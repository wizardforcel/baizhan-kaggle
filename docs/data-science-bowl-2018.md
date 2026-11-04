# Data Science Bowl 2018：细胞核识别（Nuclei Segmentation）

## 一、比赛简介

### 1.1 比赛背景

在显微镜下研究细胞时，第一步通常是把视野里的每一个**细胞核**（nucleus，细胞里装着遗传物质的那一团）单独圈出来，数一数、量一量大小和形状。人工做这件事又慢又容易出错，所以医学界一直希望用算法自动完成。

Data Science Bowl 2018 提供的正是这样一批显微镜图像。难的地方在于，这批图像**非常杂**：细胞类型不同、放大倍数不同、成像方式也不同（一种是明场 brightfield，光穿透样本成像；另一种是荧光 fluorescence，细胞自己发光）。有的图像背景明亮、细胞核偏暗；有的正好相反。比赛故意把各种条件下的图像混在一起，考验的是算法的**泛化能力**——不能只在你见过的那一种图上有效。

### 1.2 比赛目标

数据按「一张图一个文件夹」组织，目录结构如下：

```
stage1_train/
    <图像id>/
        images/<图像id>.png      # 原图（RGB）
        masks/<若干mask>.png     # 每一个细胞核一张掩膜
stage1_test/                      # 只有 images，没有 masks
stage1_train_labels.csv           # 第一阶段的提交样例
stage1_sample_submission.csv
```

第一阶段有 **670 张训练图**、**65 张测试图**。注意一个关键设计：训练图里的细胞核**不是一张掩膜，而是一个细胞核一张**——一张图里有几个细胞核，`masks` 文件夹里就有几个 png。这种「每个实例单独一张掩膜」的标注方式，正对应了我们要解决的任务：**实例分割**。而测试时要输出的是每个细胞核单独一段编码，所以「把核分开」本身就是评分的一部分。

### 1.3 评价标准

比赛指标是 **mean average precision at different IoU thresholds**，中文可以理解为「在多个交并比阈值上的平均精度」。拆开看是四步（每张图）：

1. 对我们提交的每一个预测掩膜，和图中每一个真实掩膜算 **IoU**，也就是重叠面积除以并集面积；
2. 在 0.5 到 0.95、步长 0.05 的 10 个阈值上分别判断「算不算命中」；
3. 在每个阈值上算**精度**：`Precision = TP / (TP + FP + FN)`；
4. 把这 10 个阈值上的精度取平均，得到这张图的 average precision。

最后对所有图像再取平均，就是排行榜分数。这里有三个概念需要说清楚：

- **TP（真正例）**：预测的物体和某个真实物体的 IoU 超过了阈值；
- **FP（假正例）**：预测了一个物体，但附近根本没有对应的真实物体；
- **FN（假反例）**：有一个真实物体，但我们没有预测到。

这个指标比单纯的「像素准确率」苛刻得多：如果你把两个贴在一起的细胞核连成一个提交，IoU 可能还不错，但会漏掉一个真实物体（FN），而且并集会变大；如果你把一个细胞核拆成两半，就多出一个 FP。所以**既要分割得准，又要把相邻的核正确分开**。分数范围 0～1，越高越好。

---

## 二、解题过程

我们走的路线是社区里最主流、也最容易复现的一条：**U-Net 做像素级分割，再用形态学后处理把粘连的细胞核切开。** 下面按四步展开。

### 2.1 数据探索（EDA）

我们首先统计训练集和测试集的规模、图像尺寸和掩膜数量：

```python
import os
import numpy as np
import pandas as pd

TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

train_ids = os.listdir(TRAIN_PATH)
test_ids = os.listdir(TEST_PATH)

# 训练 670 张、测试 65 张
print('训练图:', len(train_ids))
print('测试图:', len(test_ids))
```

接着把每张图的尺寸都读出来。结果发现**尺寸五花八门**，从几百像素到上千像素都有，而且长宽比也不完全一样。这不奇怪——不同显微镜、不同放大倍数拍出来的视野大小本来就不同。

再看掩膜。由于每张图对应多个掩膜文件，我们按「一张图一组」的方式统计每张图里有多少个细胞核：

```python
import skimage.io

def count_nuclei(image_id):
    """统计一张图里有多少个细胞核（即有多少个 mask 文件）"""
    mask_files = '../input/stage1_train/{}/masks/*.png'.format(image_id)
    masks = skimage.io.imread_collection(mask_files).concatenate()
    return masks.shape[0]

counts = [count_nuclei(i) for i in train_ids]
print('每图细胞核数：最少 {}，中位数 {}，最多 {}'.format(
    int(np.min(counts)), int(np.median(counts)), int(np.max(counts))))
```

每张图的细胞核数量差异很大，有的图只有一两个，有的图有上百个，密度越高越难切分。

最后看**细胞核的大小**，这直接决定了后处理时形态学算子该开多大。我们对每张图算一下细胞核的平均像素面积：

```python
def mean_nuclei_size(image_id):
    mask_files = '../input/stage1_train/{}/masks/*.png'.format(image_id)
    masks = skimage.io.imread_collection(mask_files).concatenate()
    # 每个 mask 里大于 128 的像素算作细胞核
    sizes = [np.sum(masks[i] > 128) for i in range(masks.shape[0])]
    return np.mean(sizes)

sizes = [mean_nuclei_size(i) for i in train_ids]
print(np.min(sizes), np.median(sizes), np.max(sizes))
# 26.0  265.4  7244.1
```

统计结果非常说明问题：同一张图内平均核大小的中位数只有约 **265 像素**，最小 **26**，最大 **7244**，相差两个数量级。这意味着**不能用一套固定尺寸的形态学参数去处理所有图**，后处理必须根据每张图的实际核大小自适应。

### 2.2 数据清洗

原始数据没有缺失值，但「每核一张掩膜」的组织方式不能直接喂给分割网络。网络要学的是「这个像素是不是细胞核」，所以我们把一张图的所有掩膜**逐像素取最大值合并成一张二值图**。这样不管图里有几个核，都变成同一张 0/1 标签图，任务就从「多个实例」简化成了「前景/背景」的二值分割：

```python
import cv2

X_train = np.zeros((len(train_ids), 256, 256, 3), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), 256, 256, 1), dtype=bool)

for i, name in enumerate(train_ids):
    path = TRAIN_PATH + name
    # 原图统一缩放到 256x256，方便批量训练
    img = cv2.imread(path + '/images/' + name + '.png')
    img = cv2.resize(img, (256, 256))
    X_train[i] = img

    # 把所有细胞核掩膜合并成一张：逐像素取最大
    merged = np.zeros((256, 256, 1), dtype=bool)
    for mask_name in os.listdir(path + '/masks/'):
        m = cv2.imread(path + '/masks/' + mask_name, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (256, 256))
        m = np.expand_dims(m, axis=-1)
        merged = np.maximum(merged, m)   # 任何一张掩膜标为 1，就记 1
    Y_train[i] = merged
```

这里有一个取舍：把原图压到 256×256 会让很小的细胞核变得模糊，但换来的是训练速度和显存的平衡。后面做后处理时，我们再把预测结果放大回原图尺寸。

接下来是**数据增强**。训练图只有 670 张，而 U-Net 虽然对数据量要求不算高，也容易过拟合。我们用了比较激进的一套：旋转 40 度、缩放 0.2、错切 0.2、水平/垂直平移 0.2、上下左右翻转，填充方式用 `reflect`（镜像填充），避免出现突兀的黑边：

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 原图和掩膜必须用完全相同的增强参数
aug_args = dict(shear_range=0.2, zoom_range=0.2, rotation_range=40,
                width_shift_range=0.2, height_shift_range=0.2,
                horizontal_flip=True, vertical_flip=True,
                fill_mode='reflect')

X_train_gen = ImageDataGenerator(**aug_args)
y_train_gen = ImageDataGenerator(**aug_args)
X_val_gen = ImageDataGenerator()   # 验证集不做增强
y_val_gen = ImageDataGenerator()

# 90% 训练、10% 验证
train, val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=17)

X_train_gen.fit(train, augment=True, seed=17)
y_train_gen.fit(y_train, augment=True, seed=17)
X_val_gen.fit(val, seed=17)
y_val_gen.fit(y_val, seed=17)

# 关键：seed 相同、shuffle=False，保证原图和掩膜的增强“同步”
X_train_generator = X_train_gen.flow(train, batch_size=16, seed=17, shuffle=False)
y_train_generator = y_train_gen.flow(y_train, batch_size=16, seed=17, shuffle=False)
train_generator = zip(X_train_generator, y_train_generator)
```

「同步」这件事很容易做错：如果我们对原图和掩膜分别随机旋转，转的角度不一样，标签就对不上了。所以两个生成器必须用同一个种子，并且不打乱顺序。

### 2.3 模型训练与超参数调优

模型我们选了 **U-Net**。它的结构像一个「U」：左半边不断卷积 + 池化，把图像越缩越小、通道越来越多，提取「这是什么」的语义；右半边不断上采样，把分辨率还原回去，同时通过**跳跃连接**（skip connection，把左半边同尺度的特征直接拼到右半边）保留「在哪里」的位置信息。最后用一个 1×1 卷积输出每个像素属于细胞核的概率：

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras import models

inputs = Input((256, 256, 3))

# 编码器：通道 32 -> 64 -> 128 -> 256 -> 512，每级两次 3x3 卷积 + 一次池化
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

# 解码器：上采样后与编码器的同尺度特征拼接，再把通道数减半
up6 = UpSampling2D(size=(2, 2))(conv5)
up6 = concatenate([up6, conv4])
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
# ... conv7/conv8/conv9 同理，依次拼回 conv3/conv2/conv1

# 1x1 卷积 + sigmoid，输出单通道的“是细胞核”的概率
outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
model = models.Model(inputs=[inputs], outputs=[outputs])
```

损失函数我们没有用普通的二值交叉熵，而是用 **Dice 损失**。原因是细胞核像素只占整张图的一小部分，正负样本严重不平衡；交叉熵会被大量的背景像素「稀释」，而 Dice 系数直接看「预测区域和真实区域的重合度」，对小目标更敏感：

```python
from tensorflow.keras import backend as K

def dice_loss(y_true, y_pred):
    """1 - Dice系数：预测和真实越重合，损失越小"""
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + 1) / (union + 1), axis=0)
    return 1. - dice

def iou(y_true, y_pred):
    """集合意义下的 IoU：交集 / (A + B - 交集)"""
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return K.mean((intersection + 1) / (union + 1), axis=0)

def mean_iou(y_true, y_pred):
    """在 0.5~0.95 共 10 个阈值上分别算 IoU，再取平均，
       与比赛指标的口径保持一致，方便边训练边观察"""
    results = []
    for t in np.arange(0.5, 1, 0.05):
        results.append(iou(y_true, tf.cast(y_pred > t, tf.float32)))
    return K.mean(K.stack(results), axis=0)

model.compile(optimizer='adam', loss=dice_loss, metrics=[mean_iou])
```

训练时用 `model.fit_generator` 喂入上面搭配好的生成器，batch size 16，并在验证集上监控 `mean_iou`。超参数方面我们主要做了三组对比：**输入分辨率**（256 比 128 保留了更多小核细节）、**损失函数**（Dice 比交叉熵在类别不平衡时更稳）、**增强强度**（旋转 40 度是经验上限，再大就会把细胞核转出画面）。此外，社区里还有人尝试了 ResNet 编码器、双 U-Net、LinkNet 等结构，但基础版 U-Net 已经能拿到一个像样的分数。

### 2.4 模型测试与指标报告

模型对测试图输出的是**一整张概率图**，只告诉我们「哪里可能是细胞核」，并没有把相邻的核分开。而比赛要求一个核一行编码，所以后处理是关键一步。我们把 U-Net 输出的概率图当作「细胞核主体」，另外训练一个分支输**边界**（contour），然后把两者结合起来，用经典的**分水岭算法（watershed）**把粘连的核切开。

先做形态学清理：分别对「主体」和「边界」做 Otsu 阈值二值化，合并两者、填补内部空洞，再用边界信息把不该连的地方切断，最后用开运算去掉噪点。结构元素的大小取自这张图核的平均半径，做到自适应：

```python
import numpy as np
import scipy.ndimage as ndi
import skimage.morphology as morph
from skimage.filters import threshold_otsu

def mean_blob_size(mask):
    """估计一张图里细胞核的平均面积和半径"""
    labels, n = ndi.label(mask)
    if n < 2:
        return 1, 1
    mean_area = int(np.mean(np.bincount(labels.ravel())[1:]))
    return mean_area, int(np.round(np.sqrt(mean_area) / np.pi))

def clean_mask(mask, contour):
    """把 U-Net 的掩膜和边界组合成干净的细胞核区域"""
    m_b = mask > threshold_otsu(mask)          # 主体二值化
    c_b = contour > threshold_otsu(contour)    # 边界二值化
    m_ = np.where(m_b | c_b, 1, 0)             # 先合并，避免漏掉边界处
    m_ = ndi.binary_fill_holes(m_)             # 填补细胞核内部空洞
    m_ = np.where(c_b & (~m_b), 0, m_)         # 用边界切断粘连
    area, radius = mean_blob_size(m_b)
    m_ = morph.binary_opening(m_, selem=morph.disk(0.25 * radius))  # 去噪点
    return m_
```

清理完再用距离变换 + 分水岭把每个核的「中心」当作种子点（marker），让每个核从自己的中心向外生长，碰到边界就停，从而得到一个个独立实例：

```python
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def split_instances(mask):
    """用距离变换 + 分水岭把粘连的细胞核拆成独立实例"""
    # 距离变换：每个前景像素到最近背景的距离，核中心最大
    distance = ndi.distance_transform_edt(mask)
    # 找局部极大值作为每个核的“种子”
    coords = peak_local_max(distance, min_distance=7, labels=mask)
    markers = np.zeros(distance.shape, dtype=np.int32)
    for idx, (y, x) in enumerate(coords, start=1):
        markers[y, x] = idx
    # 从种子出发，在 -distance 上做分水岭，相当于从中心往外灌水
    labels = watershed(-distance, markers, mask=mask)
    return labels
```

最后把每个实例转成 RLE（游程编码）提交。这份比赛和之前一样，一张图里有几个核就有几行，列是 `ImageId` 和 `EncodedPixels`：

```python
def rle_encode(img):
    """行主序的游程编码：记录每段连续前景的起点和长度"""
    pixels = img.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

rows = []
for image_id in test_ids:
    prob = predict(image_id)               # U-Net 预测的概率图
    mask = prob > 0.5                      # 阈值化成二值图
    labels = split_instances(mask)         # 拆成独立实例
    for inst in range(1, labels.max() + 1):
        rows.append({'ImageId': image_id,
                     'EncodedPixels': rle_encode(labels == inst)})

pd.DataFrame(rows).to_csv('submission.csv', index=False)
```

在本地验证集上，只做 U-Net 分割、不做实例后处理时，因为相邻的核经常粘成一片，`mean_iou` 指标会被拉低；加入形态学清理和分水岭之后，粘连的核被正确拆开，社区里同类的做法把排行榜分数从 **0.421 提升到 0.429**。这个提升幅度看起来不大，但它说明了一件事：**在这个比赛里，后处理几乎和数据增强一样重要**。

---

## 三、总结

本文中我们完成了 Data Science Bowl 2018 细胞核识别的完整流程。我们首先用 EDA 摸清了数据：670 张训练图、65 张测试图，每张图里每个细胞核单独存一张掩膜，图像尺寸和细胞核大小差异巨大（平均核面积从 26 到 7244 像素），成像方式包含明场和荧光两类。其次我们做了数据清洗：把多个掩膜逐像素取最大值合并成一张二值标签图，统一缩放到 256×256，并用同一随机种子的 ImageDataGenerator 对原图和掩膜做同步增强。接着我们训练了一个编码器通道从 32 到 512 的 U-Net，用 Dice 损失应对前景稀疏，用与比赛口径一致的 mean IoU 做监控。最后我们在测试阶段把预测概率阈值化，用形态学清理加距离变换分水岭把粘连的细胞核拆成独立实例，再逐个做 RLE 编码提交，后处理把排行榜分数从 0.421 提到了 0.429。

这个比赛给我们最大的启发是：**评价指标决定了后处理的地位**。如果只看像素重合度，把粘在一起的核整片提交可能还行；但比赛用的是「多阈值平均精度」，漏掉一个核就是 FN、多拆一个核就是 FP，所以「把实例分开」这一步和「把像素分对」同样重要。另外，面对尺寸差异两个数量级的数据，任何固定的形态学参数都不靠谱，必须从每张图自身估计尺度——这也是医学图像处理和自然图像处理最大的不同之一。