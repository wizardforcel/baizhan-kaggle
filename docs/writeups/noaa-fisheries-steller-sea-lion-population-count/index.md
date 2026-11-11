# NOAA Fisheries Steller Sea Lion Population Count

## 一、比赛简介

### 1.1 比赛背景
这个比赛由美国国家海洋和大气管理局（NOAA）下属的渔业部门发起，任务是**给斯特勒海狮（Steller sea lion）数数**。海狮是生活在北大平洋的大型海洋哺乳动物，种群数量是生态保护的重要指标。过去这些数字靠人对着航拍照片一只一只地数，又慢又容易出错，所以主办方希望用计算机视觉把它自动化。

训练集里有 948 张航拍图像，每张图还配了一张「打点图」（TrainDotted）：同一张照片，只是每一只海狮的头顶上被人工点了一个彩色圆点，**不同颜色代表不同类别**。也就是说，我们手上有现成的、可以自动解析的标注。海狮按体型和性别分成 5 类：成年雄性（adult_males）、亚成年雄性（subadult_males）、成年雌性（adult_females）、幼体（juveniles）和幼崽（pups）。

有意思的是，比赛还提供了一个「反例」教训：训练用的打点图和训练原图在 id 为 3、7、9 的三张图上**并不完全对应**，这提醒我们数据质量不是每次都有保证，做的时候要留个心眼。

### 1.2 比赛目标
输入是一张航拍图像，输出是这张图里 5 类海狮各有多少只。训练标签存在 `Train/train.csv` 里，它是 948 行、6 列的表：一列 `train_id` 对应图像编号，另外五列分别记录五类海狮的数量。`Test` 目录下的每张测试图都需要预测同样的 5 个整数。

### 1.3 评价标准
比赛使用**均方根误差（RMSE）**作为指标：把预测的计数与人工标注的真实计数逐类相减、求平方、取平均再开根号。RMSE 越低越好。由于一张图可能有几百只海狮，RMSE 会对大的偏差特别敏感——差 50 只的惩罚不是差 5 只的 10 倍，而是 100 倍。所以与其追求「平均误差小」，不如尽量**避免极端错判**，比如把岩石误当成一整群海狮。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**先用图像差分把打点图里的圆点还原成每只海狮的位置和类别，再把大图切块，训练一个输出 5 个计数的卷积网络，最后把全图各块的预测加起来。** 下面按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来叙述。

### 2.1 数据探索（EDA）
我们首先把文件列出来，确认目录结构。
```python
import pandas as pd
import glob, os

train_data = pd.read_csv('../input/Train/train.csv')
train_imgs = sorted(glob.glob('../input/Train/*.jpg'),
                    key=lambda n: int(os.path.basename(n)[:-4]))
train_dot_imgs = sorted(glob.glob('../input/TrainDotted/*.jpg'),
                        key=lambda n: int(os.path.basename(n)[:-4]))

print(train_data.shape)     # (948, 6)
print(len(train_imgs), len(train_dot_imgs))   # 948, 948
```
训练表有 948 行，两个图像目录各 948 张图，数量对得上，说明可以按文件名一一匹配。

接着我们看 5 个类别计数的分布：
```python
# 各类别的最小、平均、最大数量
print(train_data[['adult_males', 'subadult_males', 'adult_females',
                  'juveniles', 'pups']].describe())
```
观察下来有两件事很关键：一是 `adult_females` 和 `pups` 数量最多，一张图里常有几百只；二是不同图像之间差异悬殊，有的只有十几只，有的密密麻麻，个别图的单类数量甚至可能为零。这意味着模型要既能处理稀疏场景，也要能处理拥挤场景。

最后我们随便看一张原图和对应的打点图：
```python
import cv2

raw = cv2.imread('../input/Train/0.jpg')
dotted = cv2.imread('../input/TrainDotted/0.jpg')
print(raw.shape)      # 满幅航拍图，典型尺寸如 4488 x 8128 或更大
```
打点图在原图基础上多了许多彩色圆点。这就是我们唯一靠谱的标注来源，后面所有计数都从它推导。

### 2.2 数据清洗
清洗的核心是把彩色圆点从打点图中分离出来。思路是：**打点图减去原图，剩下的就是圆点**。再按圆点的颜色判断它代表哪一类海狮。

第一步，做差并抑制噪声。直接用原图和打点图相减会残留很多因 JPEG 压缩引起的噪声，所以我们先对打点图做高斯模糊，再用亮度掩膜只保留「打点图明显更亮」的区域。
```python
import cv2
import numpy as np
import skimage.feature

def get_data(filename):
    # 读入同一张编号的原图和打点图
    image_1 = cv2.imread("../input/TrainDotted/" + filename)
    image_2 = cv2.imread("../input/Train/" + filename)

    # 先模糊打点图：圆点周边的压缩噪声被平滑，圆点本身还在
    img1 = cv2.GaussianBlur(image_1, (5, 5), 0)
    # 原图与打点图之差的绝对值
    image_3 = cv2.absdiff(image_1, image_2)

    # 只保留图像中比较亮的部分，把背景噪声剪掉
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    # 把三通道取最大值压成灰度，便于做斑点检测
    image_6 = np.max(image_4, axis=2)
    # 用高斯拉普拉斯算子找亮斑，sigma 范围 3~7 对应圆点的大小
    blobs = skimage.feature.blob_log(
        image_6, min_sigma=3, max_sigma=7,
        num_sigma=1, threshold=0.05)
    return image_2, img1, blobs
```
为什么不用连通域而用 `blob_log`？因为航拍图和圆点之间有大量二值化的杂质，纯连通域会把相邻的碎块连成一个大块；而固定尺度的亮斑检测对形状更挑剔，能避免把岩石边缘误认成海狮。

第二步，读圆点中心的颜色，映射到 5 类。材料里的颜色规则如下：
```python
if R > 225 and b < 25 and g < 25:          # 红   -> 成年雄性
    res[x1, y1, 0] += 1
elif R > 225 and b > 225 and g < 25:       # 品红 -> 亚成年雄性
    res[x1, y1, 1] += 1
elif R < 75 and b < 50 and 150 < g < 200:  # 绿   -> 幼体
    res[x1, y1, 4] += 1
elif R < 75 and 150 < b < 200 and g < 75:  # 蓝   -> 幼崽
    res[x1, y1, 3] += 1
elif 60 < R < 120 and b < 50 and g < 75:   # 棕   -> 成年雌性
    res[x1, y1, 2] += 1
```
每个圆点都被累加到一个按倍数缩放的网格里，之后这个小网格会映射到对应的图像块上。

### 2.3 模型训练与超参数调优
航拍图非常大，一整张塞进 GPU 既不现实也没必要。我们把图按比例缩小，再切成固定大小的小块（patch），让模型回答「这一格里五类各有几只」。训练时，每个 patch 对应一个长度 5 的计数向量；预测时，把同一张图所有 patch 的输出相加，就是整图的计数。

先用一个轻量版本说明流程：
```python
r = 0.4          # 缩小到 40%，降低分辨率也降低算力
width = 100      # 每个 patch 100x100

img = cv2.resize(image_2, (int(w * r), int(h * r)))
h1, w1, _ = img.shape
trainX, trainY = [], []
for i in range(w1 // width):
    for j in range(h1 // width):
        # 第 (i,j) 格的像素块
        trainX.append(img[j*width:j*width+width,
                            i*width:i*width+width, :])
        # 同一格对应的五类计数
        trainY.append(res[i, j, :])
```
切块之后，一张图通常能得到几百个 patch，样本量一下子就够了。这里有个细节：`res` 的网格索引和图像块的索引必须严格对应，否则标签会错位；材料里正是用同一套 `r` 和 `width` 来保证这一点。

然后训练一个小型卷积网络。它的输出层是 5 个线性单元——不要加 softmax，因为我们做的是**回归计数**，不是分类。
```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                 input_shape=(width, width, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))          # 100 -> 50
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))          # 50 -> 25
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='linear')) # 五个计数，线性输出
model.compile(loss='mean_squared_error', optimizer='adam')
```
损失用均方误差，和比赛的 RMSE 指标同源：最小化 MSE，等价于最小化 RMSE。

作者的完整方案做了三处升级。第一，patch 从 100x100 放大到 300x300，并且不是只缩一次，而是按 0.9 的倍数逐级缩小（0.9^0 到 0.9^4），让模型见过不同尺度；第二，把主干换成 ImageNet 预训练的 VGG16，只保留卷积部分，再接一个 1024 维全连接层：
```python
from keras.applications import VGG16
from keras.layers import LeakyReLU
from keras.models import Model

base = VGG16(weights='imagenet', include_top=False,
             input_shape=(300, 300, 3))
x = Flatten()(base.output)
x = Dense(1024)(x)
x = LeakyReLU(alpha=0.1)(x)      # LeakyReLU 避免负半轴完全死亡
preds = Dense(5, activation='linear')(x)
model = Model(base.input, preds)
```
第三，正负样本比例控制在 1:3，也就是每保留一个含海狮的 patch，就配三个纯背景 patch。这样模型不会因为背景太多而「学会永远预测 0」，同时又不至于把少量正样本淹没。

训练策略上，作者没有一上来就用大学习率，而是先用极小的学习率热身：
```python
import keras

# 第一段：小学习率热身，避免预训练权重被大梯度破坏
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.SGD(lr=1e-5, momentum=0.2))
model.fit(trainX, trainY, epochs=8, verbose=2)

# 第二段：换正常学习率继续训练
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9))
model.fit(trainX, trainY, epochs=30, verbose=2)
```
材料的训练日志显示，热身的 8 个 epoch 把损失从 37 降到 1.57，之后换 1e-4 学习率的 30 个 epoch 又把损失降到 0.37。最后再跑 20 个 epoch，损失降到 0.06。注意在 1200 秒的运行上限下，完整训练会被打断，所以作者把训练拆成多段、每段继续 `fit`——这在 Kaggle notebook 里是很实用的技巧。

数据增强对这场比赛特别重要，因为海狮在图像中的朝向、光照、拍摄角度都很多变。作者用了翻转、旋转，以及饱和度、亮度、对比度扰动。这些变换都不会改变「有几只海狮」的事实，却能让模型不把某个特定光照当成类别线索。

### 2.4 模型测试与指标报告
训练完以后，我们先把一张图切成 patch，逐个预测并求和，再和真实计数算 RMSE：
```python
def rmse(pred, target):
    return np.sqrt(((np.asarray(pred) - np.asarray(target)) ** 2).mean())

# 对每个 patch 预测五类计数，再沿 patch 维度相加
patch_preds = model.predict(test_patches)   # 形状 (n_patch, 5)
image_count = patch_preds.sum(axis=0)       # 整图五类计数

print('predicted counts:', image_count)
print('RMSE:', rmse(image_count, true_counts))
```
`sum(axis=0)` 这一步是全流程的关键：模型只学了「小格里有几只」，整图计数由求和得到。这样做的好处是，即使某张小图超出了训练时的密度范围，拆成小格后每格的分布仍然接近训练分布，泛化更稳。

材料里还提到一个基准：先随机抽几张图看一眼预测与真实标注的差距，再决定是调网络还是改切块尺度。如果误差主要来自某些特别拥挤的图，就应缩小 patch 或提高输入分辨率；如果误差来自空图被误判，则应增加背景样本比例。

最后生成提交文件：
```python
submission = pd.DataFrame(image_counts, columns=[
    'adult_males', 'subadult_males', 'adult_females',
    'juveniles', 'pups'])
submission.insert(0, 'test_id', test_ids)
submission.to_csv('submission.csv', index=False)
```
提交前检查三件事：行数是否等于测试图数量、五列是否都是非负整数（计数不能为负）、列名是否和 `sample_submission.csv` 完全一致。

---

## 三、总结
本文中我们完成了 NOAA 斯特勒海狮计数比赛的完整流程。我们首先通过 EDA 确认训练集有 948 张图、5 个类别且密度差异极大；其次用原图与打点图的差分、亮度掩膜和 `blob_log` 把彩色圆点还原成逐格计数；接着把大图缩放切块，训练一个以 VGG16 为骨干、输出 5 个线性计数的卷积回归网络，并用小学习率热身、1:3 正负样本和翻转/旋转/颜色增强来稳定训练；最后把所有 patch 的预测按图求和，用 RMSE 评估并生成提交文件。

这个比赛最大的启发是**把难题拆小**：整图数几百只海狮很难直接回归，但「这个小方格里各有几只」是模型能学会的问题，把每格的答案加起来就得到整图。另一个教训来自材料里那三张打点图与原图不一致的样本——真实数据永远可能有瑕疵，先验证标注一致性，比盲目堆模型更重要。