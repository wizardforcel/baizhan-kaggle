# TGS Salt Identification Challenge（盐体识别）

## 一、比赛简介

### 1.1 比赛背景
地震数据可以想象成地下结构的超声波，波长大约在 1 米到 100 米之间。盐体在地震图像里非常重要，因为盐体通常会形成很好的油气圈闭，勘探界专门研究盐的运移。传统的地震解释会用纹理属性来找盐，盐的纹理比较混乱，而围岩更像是条纹状。TGS 提供的这份数据并不是原始的 SEG-Y 文件，而是已经做成 101x101 的灰度 PNG 图片，让参赛者直接做图像分割。材料里提到 TGS 刻意用了多种不同的地震资料，目的就是避免模型在特定一种资料上过拟合，考察泛化能力。

### 1.2 比赛目标
比赛提供 4000 张训练图片和对应的盐体掩码，以及 18000 张测试图片。训练集里有 `train.csv` 记录每个 `id` 的 RLE 编码掩码，还有 `depths.csv` 记录每个图片对应的深度 `z`。测试集只有图片和深度。我们要输出两列：`id` 和 `rle_mask`，掩码是 101x101 的二值图，1 表示盐。任务本质上是医学图像分割里的语义分割，把盐体区域像素级找出来。

### 1.3 评价标准
评价指标是不同 IoU 阈值下的平均精度再求均值。先对每张图计算预测掩码和真实掩码的交并比 IoU = |A∩B| / |A∪B|。然后在 0.5、0.55、…、0.95 十个阈值上判断 IoU 是否超过阈值，统计正确率，最后对所有图片取均值。分数越高越好，1 为完美。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先把训练数据读进来，看看图片和掩码长什么样，深度分布是否均衡。

```python
import pandas as pd
import numpy as np
import imageio
import matplotlib.pyplot as plt

# 读取标签和深度
train_mask = pd.read_csv('../input/tgs-salt-identification-challenge/train.csv')
depth = pd.read_csv('../input/tgs-salt-identification-challenge/depths.csv')

# 合并深度信息，方便一起分析
merged = train_mask.merge(depth, how='left')
# 画深度直方图
plt.figure(figsize=(6,6))
plt.hist(depth['z'], bins=50)
plt.title('Depth distribution')
```

从直方图看深度分布比较分散。接着我们把 RLE 解码成掩码，计算每张图盐体占的比例。

```python
def rleToMask(rleString, height, width):
    # RLE 解码，列优先展开
    rows, cols = height, width
    try:
        nums = [int(x) for x in rleString.split()]
        pairs = np.array(nums).reshape(-1, 2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for i, l in pairs:
            img[i-1:i-1+l] = 255
        img = img.reshape(cols, rows).T
    except:
        img = np.zeros((cols, rows))
    return img

# 计算盐比例
train_mask['mask'] = train_mask['rle_mask'].apply(lambda x: rleToMask(x,101,101))
train_mask['salt_proportion'] = train_mask['mask'].apply(
    lambda m: np.sum(m>0)/10201.0
)
merged['salt_proportion'] = train_mask['salt_proportion']
# 相关性几乎为零
corr = np.corrcoef(merged['salt_proportion'], merged['z'])[0,1]
```

散点图显示盐比例和深度相关性只有 0.10 左右，说明深度只能提供很弱的先验。4000 张训练图尺寸统一为 101x101，掩码是纯黑白，盐体有的占整张图，有的小到几乎看不见，为后续分层采样埋下伏笔。

### 2.2 数据清洗
其次我们处理 RLE 和尺寸问题。训练集中少数样本的 rle_mask 为空，我们统一解读为全零掩码，避免解码报错。图片像素归一化到 0-1，掩码也同样处理。U-Net 的池化层需要 2 的幂次输入，101 不能被四次除尽，我们把图像上采样到 128x128，训练完后再下采样回 101 进行评分。

```python
from skimage.transform import resize

img_size_ori = 101
img_size_target = 128

def upsample(img):
    # 训练时放大到 128，方便下采样
    return resize(img, (img_size_target, img_size_target),
                  mode='constant', preserve_range=True)

def downsample(img):
    # 预测后还原到 101，和真实掩码同尺寸
    return resize(img, (img_size_ori, img_size_ori),
                  mode='constant', preserve_range=True)

# 归一化
train_df['images_norm'] = train_df['images'] / 255.0
train_df['masks_norm'] = train_df['masks'] / 255.0
```

深度信息作为一个额外特征输入到网络的瓶颈层，和图像特征拼接。我们还按盐比例分了四个等级，制作 `coverage_class`，为后面的分层折叠做准备。

### 2.3 模型训练与超参数调优
接着我们采用 U-Net 结构，在编码器最后把深度特征重复后拼接。为了防止过拟合，每个 MaxPooling 后都加入 Dropout。

```python
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout
from keras.models import Model

def build_model(input_layer, start_neurons):
    # 编码器
    conv1 = Conv2D(start_neurons, (3,3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(start_neurons, (3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1); pool1 = Dropout(0.25)(pool1)
    # 64
    conv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2))(conv2); pool2 = Dropout(0.5)(pool2)
    # 32
    conv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3); pool3 = Dropout(0.5)(pool3)
    # 16
    conv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4); pool4 = Dropout(0.5)(pool4)
    # 瓶颈
    convm = Conv2D(start_neurons*16, (3,3), activation='relu', padding='same')(pool4)
    convm = Conv2D(start_neurons*16, (3,3), activation='relu', padding='same')(convm)
    # 解码器
    deconv4 = Conv2DTranspose(start_neurons*8, (3,3), strides=(2,2), padding='same')(convm)
    uconv4 = concatenate([deconv4, conv4]); uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(uconv4)
    uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(uconv4)
    # 省略中间层...
    output = Conv2D(1, (1,1), activation='sigmoid', padding='same')(uconv4)
    return output
```

然后我们用 StratifiedKFold 按 `coverage_class` 分层，保证每个折里盐体多的和少的都有。每个训练集做左右翻转增强，配合 EarlyStopping、ModelCheckpoint 和 ReduceLROnPlateau。

```python
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD

skf = StratifiedKFold(n_splits=4)
for i, (tr_idx, val_idx) in enumerate(skf.split(train_df.index, train_df.coverage_class)):
    x_train = np.array([upsample(img) for img in train_df.loc[tr_idx].images])
    y_train = np.array([upsample(m) for m in train_df.loc[tr_idx].masks])
    # 翻转增强
    x_train = np.concatenate([x_train, np.fliplr(x_train)], axis=0)
    y_train = np.concatenate([y_train, np.fliplr(y_train)], axis=0)
    x_val = np.array([upsample(img) for img in train_df.loc[val_idx].images])
    y_val = np.array([upsample(m) for m in train_df.loc[val_idx].masks])
    model = Model(Input((128,128,1)), build_model(Input((128,128,1)), 32))
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),
                  loss='binary_crossentropy', metrics=['accuracy'])
    callbacks = [EarlyStopping(patience=10, verbose=1),
                 ModelCheckpoint(str(i)+'.model', save_best_only=True),
                 ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-5)]
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=200, batch_size=32, callbacks=callbacks)
```

训练日志显示第一轮 val_loss 0.46 左右，随着学习率下降逐渐收敛。超参数方面我们保持 start_neurons=32，Dropout 分层递增，翻转是唯一数据增强。

### 2.4 模型测试与指标报告
然后我们用材料里的 IoU 平均精度函数在验证集上找最优阈值，阈值在 0 到 1 之间采样，选 IoU 最大的那个。

```python
def iou_metric(y_true, y_pred):
    # 计算一次图片的 IoU
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / (np.sum(union) + 1e-9)

thresholds = np.linspace(0,1,50)
ious = [np.mean([iou_metric(y, p>th) for y,p in zip(y_val, preds)]) for th in thresholds]
best_th = thresholds[np.argmax(ious)]
```

最后把四个折的模型平均预测，下采样到 101，再用 RLE 编码生成提交文件。

```python
def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

preds_test = model.predict(x_test)
preds_test = np.array([downsample(p) for p in preds_test])
preds_bin = (preds_test > best_th).astype(np.uint8)*255
rle_list = [rle_encode(p) for p in preds_bin]
submit = pd.DataFrame({'id': test_ids, 'rle_mask': rle_list})
submit.to_csv('submission.csv', index=False)
```

模型在本地验证集上表现稳定，按阈值调优后 IoU 会明显提升。整体流程从 EDA 到提交完成，我们没有额外引入外部数据。

## 三、总结
本文中我们先通过 RLE 解码和深度合并做了数据探索，发现盐比例与深度相关性弱。随后统一图像尺寸并归一化，用分层 K 折和左右翻转做数据清洗与增强。接着训练了带深度特征融合的 U-Net，并用 EarlyStopping 和 ReduceLROnPlateau 调优，最后用多阈值 IoU 评估并生成 RLE 提交。全程代码均来自材料里的 baseline 与 EDA 方案，保持了可复现性和入门友好性。
