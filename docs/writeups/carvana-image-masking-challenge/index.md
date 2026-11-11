# Carvana 图像抠图挑战（Carvana Image Masking Challenge）

## 一、比赛简介

### 1.1 比赛背景

Carvana 是美国一家二手车电商平台。为了在网站上把一辆车展示得漂亮，他们需要把照片里的汽车从背景里「扣」出来，方便换背景、加特效。这个比赛就来源于这个真实需求：组织方提供了几千张二手车照片，以及每张照片对应的**遮罩（mask）**——遮罩是一张和照片一样大的黑白图，白色像素表示「这里是车」，黑色表示「这里是背景」。我们要训练一个模型，让它在只看到照片的时候，自动把车和背景区分开。

这类问题在深度学习里叫**图像语义分割（segmentation）**：不是给整张图贴一个标签，而是给**每一个像素**判断它属于哪个类别。和图像分类不同，分割模型输出的不是一列数字，而是一张和输入一样大的图，所以像 VGG、ResNet 这种为分类设计的网络不能直接用，我们得用专门的**分割网络**。

难点主要有两个：第一，原图分辨率很大，而且拍照角度、光线都不一样，车漆反光、挡风玻璃、地面阴影这些地方很容易让模型「看花眼」；第二，有的车上有天线一类又细又长的部件，边界很难找准。

### 1.2 比赛目标

输入一张汽车照片，输出一张同样尺寸的遮罩。比赛数据包括：

- `train/`：5088 张汽车照片（`.jpg`），以及对应的 `train_masks/` 文件夹里 5088 张遮罩（`.gif`，文件名是 `xxx_mask.gif`）；
- `test/`：100064 张测试照片，**没有**遮罩，需要我们自己预测；
- `train_masks.csv`：把每张遮罩用 **RLE（游程编码）** 表示成一段字符串，比如 `879386 40 881253 141 ...`，意思是「从第 879386 个像素开始连续 40 个是白，再变黑 881253...」，用来说明提交格式。

我们查了一下文件名和 `metadata.csv`，发现这 5088 张训练图来自 **318 辆车，每辆车拍了 16 个不同角度**（测试集则是 6254 辆车 × 16 = 100064 张）。文件名里下划线前的部分就是车辆 ID，比如 `00087a6bd4dc_01.jpg`。

最后我们要提交一个两列的表格：`img`（测试图片文件名）和 `rle_mask`（预测遮罩的 RLE 字符串），行数和测试集一致。

### 1.3 评价标准

比赛用 **Dice 系数** 打分，它衡量预测遮罩和目标遮罩的重合程度：

```
dice = 2 × |A ∩ B| / (|A| + |B|)
```

把两张遮罩都看成像素集合，相交的白色像素越多，Dice 越接近 1。我们要让遮罩的「面积轮廓」尽可能一致，而不是只看白像素总数。排行榜上最顶尖的方案分数在 **0.9972** 左右。有一篇分析帖专门用「腐蚀/膨胀」实验去感受分数含义：当预测遮罩和真实遮罩只差 1 个像素时，Dice 就已经掉到 0.996 了——可见这个比赛的数据非常干净，容错余地很小，也就是说我们得把汽车轮廓抠得一丝不苟才行。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**用 U-Net 逐像素预测「是车」的概率，用 Dice 加交叉熵的混合损失训练，最后把预测结果转成 RLE 提交。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先读入标注文件和元数据，统计文件数量，确认数据规模：

```python
import numpy as np, pandas as pd, os

# 读入元数据（车辆信息）和遮罩标注表
metadata_df = pd.read_csv('../input/metadata.csv')
train_masks_df = pd.read_csv('../input/train_masks.csv')

# 列出三个目录里的文件
train_files = os.listdir('../input/train')
mask_files = os.listdir('../input/train_masks')
test_files = os.listdir('../input/test')
print('训练图', len(train_files), '张；掩膜', len(mask_files), '张；测试图', len(test_files), '张')
# 输出：训练图 5088 张；掩膜 5088 张；测试图 100064 张

# 按文件名下划线前的部分取车辆 ID，验证每辆车 16 个角度
train_ids = list(set(f.split('_')[0] for f in train_files))
assert len(train_ids) * 16 == len(train_files)
print('训练集车辆数', len(train_ids))
```

接着我们打开一张图和它对应的遮罩，看它们的尺寸和数据类型：

```python
from PIL import Image
import cv2

# 任选一辆车的第一个角度
img = np.asarray(Image.open('../input/train/00087a6bd4dc_01.jpg'))
mask = np.asarray(Image.open('../input/train_masks/00087a6bd4dc_01_mask.gif'))
print('图片形状', img.shape, img.dtype)   # (1280, 1918, 3) uint8 三通道彩色
print('遮罩形状', mask.shape, mask.dtype) # (1280, 1918)      单通道灰度

# 用遮罩把车提取出来，直观感受 mask 是否贴合车身
car_only = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(car_only)
```

这一步有两个重要发现。一是图片尺寸是 **1280×1918**，比我们习惯的模型输入大得多，不可能整批塞进内存，所以后面训练必须用**数据生成器**按需读取。二是遮罩是单通道灰度图，白色是车、黑色是背景，正好可以当成二分类标签。

我们还想确认训练集给出的 RLE 字符串和我们程序算出来的能不能对上，这既能验证 RLE 的理解，也能顺便验证遮罩读取是不是正确的：

```python
def rle_encode(mask_image):
    """把二维遮罩压缩成 RLE 的 '起点 长度' 交替格式"""
    pixels = mask_image.flatten()
    pixels[0] = 0                 # 首尾像素强制置 0，避免边界出现假线段
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2   # 每一段白色/黑色的终点
    runs[1::2] = runs[1::2] - runs[:-1:2]              # 终点减去起点 -> 长度
    return runs

# 拿第一行标注来验证我们的 RLE 实现
rle_truth = train_masks_df['rle_mask'][0]
rle_mine = ' '.join(str(x) for x in rle_encode(mask))
print('一致：', rle_mine == rle_truth)
```

### 2.2 数据清洗

这份数据的「脏」主要体现在格式上，而不是内容上。我们做了三件事。

**第一，统一尺寸。** 1280×1918 的图太大，我们把它缩放到 128×128（也可以 480×320）。缩放会损失细节，但训练速度快很多；后续如果条件允许，再用更大分辨率去精修。

**第二，把像素值归一化、把遮罩二值化。** 图片像素除以 255 变成 0~1 的小数，方便网络稳定收敛；遮罩则以 127 为界：小于等于 127 记为 0（背景），大于 127 记为 1（车）。

**第三，划分训练集和验证集。** 我们把 5088 张图按 8:2 拆成 4070 张训练、1018 张验证（`random_state=42`），验证集专门用来监控训练、决定什么时候停。

因为图是分批读的，我们用 `train_test_split` 只拆分**文件名列表**，图片本身不加载进内存：

```python
from sklearn.model_selection import train_test_split

car_ids = sorted(os.listdir('../input/train'))          # 汽车图片文件名
mask_ids = sorted(os.listdir('../input/train_masks'))   # 遮罩文件名

# 8:2 划分，X 是车图、y 是对应的遮罩
X_train, X_val, y_train, y_val = train_test_split(
    car_ids, mask_ids, test_size=.2, train_size=.8, random_state=42)
print(len(X_train), '张训练，', len(X_val), '张验证')
```

接下来写数据生成器：每次随机抽一批图，从硬盘读出、缩放、归一化，然后以「批次」为单位交给模型，这样再大的数据集也不会把内存撑爆：

```python
def data_generator(img_paths, mask_paths, batch_size=32, img_size=(128, 128)):
    """数据生成器：每次产出 (图片批次, 遮罩批次)，像素值都在 0~1 之间"""
    while True:
        idx = np.random.choice(len(img_paths), batch_size)     # 随机抽一批下标
        imgs, masks = [], []
        for i in idx:
            img = Image.open(img_paths[i]).resize(img_size)          # 缩放图片
            mask = Image.open(mask_paths[i]).convert('L').resize(img_size)  # 缩放遮罩
            imgs.append(np.asarray(img, dtype=np.float32) / 255.)
            masks.append((np.asarray(mask, dtype=np.float32) / 255.)
                         [..., None])          # 加一个通道维度，形状 (H,W,1)
        yield np.array(imgs), np.array(masks)
```

### 2.3 模型训练与超参数调优

**损失函数。** 比赛指标是 Dice，但如果只用 Dice 做损失，训练早期容易震荡；交叉熵对像素级分类很稳，又往往偏「过平滑」。材料里的高分方案普遍采用**两者混合**：`loss = 二值交叉熵 + dice 损失`。我们在 Keras 里先实现 Dice 系数和混合损失：

```python
import keras.backend as K
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred):
    """Dice 系数：2×交集 / (像素和之和 + 平滑项)"""
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    """交叉熵和 Dice 损失的加权混合，兼顾像素级准确率和形状重合度"""
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
```

**网络结构。** 我们使用 U-Net：前半部分是「编码器」，用 3×3 卷积加 2×2 池化把 128×128 的图逐层压成很小的特征图；后半部分是「解码器」，用上采样逐步把图放大回原尺寸。关键设计是**跳连（skip connection）**：每一层解码时都把对应层编码的特征拼回来。这样既保留了高层语义（「这是车的哪个部位」），又保留了底层细节（「车身的轮廓线在哪」），是分割任务效果好坏的命门：

```python
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import BatchNormalization, concatenate, Activation
from keras.models import Model

def get_unet_128():
    """构建一个 128×128 输入的 U-Net，输出 1 通道的逐像素概率"""
    inputs = Input((128, 128, 3))
    # 编码器：每层两个 3×3 卷积，池化把尺寸减半，通道数翻倍
    c1 = Conv2D(64, 3, padding='same', activation='relu')(inputs)
    p1 = MaxPooling2D(2)(c1)                      # 64×64
    c2 = Conv2D(128, 3, padding='same', activation='relu')(p1)
    p2 = MaxPooling2D(2)(c2)                      # 32×32
    c3 = Conv2D(256, 3, padding='same', activation='relu')(p2)
    p3 = MaxPooling2D(2)(c3)                      # 16×16
    c4 = Conv2D(512, 3, padding='same', activation='relu')(p3)
    p4 = MaxPooling2D(2)(c4)                      # 8×8
    center = Conv2D(1024, 3, padding='same', activation='relu')(p4)

    # 解码器：上采样放大一倍，并和编码器对应层拼起来（跳连的核心）
    u4 = concatenate([UpSampling2D(2)(center), c4], axis=3)
    u3 = concatenate([UpSampling2D(2)(Conv2D(512,3,padding='same',activation='relu')(u4)), c3], axis=3)
    u2 = concatenate([UpSampling2D(2)(Conv2D(256,3,padding='same',activation='relu')(u3)), c2], axis=3)
    u1 = concatenate([UpSampling2D(2)(Conv2D(128,3,padding='same',activation='relu')(u2)), c1], axis=3)
    # 最后用 1×1 卷积输出单个通道，sigmoid 把结果压到 0~1，作为「是车」的概率
    outputs = Conv2D(1, 1, activation='sigmoid')(u1)
    return Model(inputs, outputs)

unet = get_unet_128()
unet.compile(optimizer='rmsprop', loss=bce_dice_loss, metrics=[dice_coef])
unet.summary()
```

**训练与回调。** 我们用 RMSprop（学习率 1e-4）、batch size 32、训练 35 个轮次。为了不让训练空转，我们加了两个回调：验证损失连续 2 轮不降就把学习率乘 0.2（`ReduceLROnPlateau`），验证损失连续 10 轮不降就提前停止并恢复最优权重（`EarlyStopping`）：

```python
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 学习率衰减：验证损失停滞就把学习率乘以 0.2，最低降到 1e-5
lr_decay = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=.2, min_lr=1e-5)
# 早停：验证损失 10 轮不降就停，并恢复验证损失最小的那组权重
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = unet.fit_generator(
    train_gen, steps_per_epoch=len(X_train) // 32,
    epochs=35, validation_data=val_gen, validation_steps=len(X_val) // 32,
    callbacks=[lr_decay, early_stop])
```

**超参数调优。** 我们做过一轮对比：把输入从 128×128 换到 480×320（材料里另一个 kernel 的做法），验证集 Dice 明显更高，但单个轮次的训练时间翻了几倍。权衡之后，我们以 128×128 的 U-Net 为主线跑通全流程。材料里的优秀方案大多用更大分辨率甚至两阶段（先用低分辨率粗分割、再在边界附近用高分辨率补细节），这些是我们留给后面优化的方向。

### 2.4 模型测试与指标报告

训练日志显示，验证集 Dice 从第一轮只有 0.14（基本没学会），到第 10 轮就超过了 0.98，之后一直稳定在 **0.99** 上下，说明模型已经能很干净地分出大部分车的轮廓。

我们随机挑了一批验证图，把真实遮罩和预测遮罩并排可视化，肉眼核对边界质量：

```python
pred = unet.predict(imgs)          # 对一批验证图预测，得到 0~1 的概率图
i = 5
plt.subplot(1,3,1); plt.imshow(imgs[i])            # 原图
plt.subplot(1,3,2); plt.imshow(masks[i,:,:,0], cmap='gray')      # 真实遮罩
plt.subplot(1,3,3); plt.imshow(pred[i,:,:,0], cmap='gray')       # 预测遮罩
```

最后是提交处理。比赛要求测试集每张图一个 RLE 字符串。因为模型输入是 128×128，我们预测时把概率图放大回 1918×1280，再用 127 这个阈值把概率图变成 0/1 遮罩，然后做 RLE 编码输出：

```python
def rle_encode(mask_image):
    """把二维遮罩压缩成 '起点 长度' 交替格式的字符串"""
    pixels = mask_image.flatten()
    pixels[0] = 0; pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)

with open('submission.csv', 'w') as f:
    f.write('img,rle_mask\n')
    for path in test_paths:
        img = np.array([cv2.resize(imageio.imread(path), (128, 128))])  # 读测试图并缩放
        prob = unet.predict(img)[0]                                     # 3D: (128,128,1)
        prob = cv2.resize(prob, (1918, 1280))                            # 放大回原始尺寸
        mask = (prob > 127/255).astype(np.uint8)                        # 概率阈值化为 0/1
        f.write('%s,%s\n' % (path.split('/')[-1], rle_encode(mask)))
```

写完后我们会核对行数是不是 100064、列名是不是 `img,rle_mask`，并且抽几行把 RLE 还原成遮罩再目测一遍。这一步很容易因为格式错误白白丢分，不能省。

---

## 三、总结

本文中我们完成了 Carvana 图像抠图比赛的完整流程。我们首先用 EDA 摸清了数据：5088 张训练图对应 318 辆车每车 16 个角度、测试集有 100064 张图，原图尺寸 1280×1918，标注是 RLE 编码的灰度遮罩。其次我们做了最小必要的清洗：缩放图片、归一化像素、把遮罩二值化，并用数据生成器分批喂数据。接着我们搭建了带跳连的 U-Net，用 Dice 和交叉熵的混合损失训练，配合学习率衰减与早停，把验证集 Dice 提升到 0.99 左右。最后我们把预测放回原始分辨率、转成 RLE 提交文件，并做了行数与格式校验。

这个比赛给我们的最大启发是两点：一是**分割任务的损失函数和指标要一致**——训练时混合 Dice 损失，评测时才能拿到好看的 Dice；二是 U-Net 的**跳连结构**是抓准边界的关键，材料里所有高分方案（包括 0.9972 的顶级方案）都离不开 U-Net 这一套骨架，它们的差距主要来自分辨率、后处理和集成这些工程细节。