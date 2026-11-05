# NOAA Fisheries Steller Sea Lion Population Count

## 一、比赛简介

### 1.1 比赛背景
这个比赛由 NOAA Fisheries 发起，目标是从航拍照片中自动统计斯特勒海狮（Steller sea lion）的数量。海狮分为 5 个类别：成年雄性（adult_males）、亚成年雄性（subadult_males）、成年雌性（adult_females）、幼体（juveniles）和幼崽（pups）。训练集包含 948 张航拍图像，每张图都配有一份人工标注的「打点图」（TrainDotted），图上用不同颜色的圆点标出每只海狮的位置和类别。比赛希望用计算机视觉自动完成这个原本需要人工逐只清点的工作。

### 1.2 比赛目标
输入是一张航拍图像，输出是图像中 5 个类别各自的数量。训练集 `train.csv` 有 5 个计数列加一个 `train_id`；测试集图像需要预测同样的 5 个计数。提交文件每行对应一张测试图，包含 5 个整数列。

### 1.3 评价标准
比赛使用均方根误差（RMSE）作为评价指标，计算预测计数与人工标注真实计数之间的偏差。RMSE 越低越好。由于一张图里可能有几百只海狮，计数误差会被平方放大，所以模型既要数得准，也要避免极端的错误。

---

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读取训练表并统计图像数量。
```python
import pandas as pd
import glob, os

train_data = pd.read_csv('../input/Train/train.csv')
train_imgs = sorted(glob.glob('../input/Train/*.jpg'),
                    key=lambda name: int(os.path.basename(name)[:-4]))
train_dot_imgs = sorted(glob.glob('../input/TrainDotted/*.jpg'),
                        key=lambda name: int(os.path.basename(name)[:-4]))

print(train_data.shape)          # (948, 6)
print(len(train_imgs))
```
训练表有 948 行、6 列，说明图像数量与标注行数一一对应。我们进一步观察计数的分布。
```python
# 看看五类目标各自的平均数量和最大值
print(train_data[['adult_males','subadult_males','adult_females',
                  'juveniles','pups']].describe())
```
可以看到 `adult_females` 和 `pups` 的数量通常最多，单张图可达几百只；`adult_males` 相对较少。不同图像之间的密度差异很大，有的图只有十几只，有的图密密麻麻。

### 2.2 数据清洗
数据的「清洗」主要围绕标注展开：把打点图上的彩色圆点还原成计数。思路是拿原图和打点图做差，再检测圆点。
```python
import cv2
import numpy as np
import skimage.feature

def GetData(filename):
    # 打点图与原图
    image_1 = cv2.imread("../input/TrainDotted/" + filename)
    image_2 = cv2.imread("../input/Train/" + filename)
    img1 = cv2.GaussianBlur(image_1, (5,5), 0)
    # 两图相减，剩下的就是圆点所在区域
    image_3 = cv2.absdiff(image_1, image_2)
    # 只在打点图足够亮的地方保留差异，滤掉噪声
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
    # 取三通道最大值转成灰度，方便 blob 检测
    image_6 = np.max(image_4, axis=2)
    blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=7,
                                     num_sigma=1, threshold=0.05)
    return image_2, img1, blobs
```
这样得到的 `blobs` 就是每只海狮圆点的坐标。

### 2.3 模型训练与超参数调优
我们把大图切成一格格小图（patch），让模型学「这一格里各类海狮各有几只」，再把全图的预测加总。这样把一个超大计数问题拆成了很多小回归问题，也方便做数据增强。

首先按圆点颜色判断类别。
```python
# 根据打点图中心像素的 RGB 值判断类别
if R > 225 and b < 25 and g < 25:            # 红色 -> 成年雄性
    res[x1, y1, 0] += 1
elif R > 225 and b > 225 and g < 25:          # 品红 -> 亚成年雄性
    res[x1, y1, 1] += 1
elif R < 75 and b < 50 and 150 < g < 200:     # 绿色 -> 幼体
    res[x1, y1, 4] += 1
elif R < 75 and 150 < b < 200 and g < 75:     # 蓝色 -> 幼崽
    res[x1, y1, 3] += 1
elif 60 < R < 120 and b < 50 and g < 75:      # 棕色 -> 成年雌性
    res[x1, y1, 2] += 1
```
接着缩放整图并切块，每个块对应一个长度为 5 的计数向量。
```python
r = 0.4          # 把原图缩放到 40%，降低计算量
width = 100      # 每个 patch 100x100 像素

img = cv2.resize(image_2, (int(w*r), int(h*r)))
for i in range(int(w1 // width)):
    for j in range(int(h1 // width)):
        trainY.append(res[i, j, :])   # 这一格的 5 类计数
        trainX.append(img[j*width:j*width+width,
                            i*width:i*width+width, :])
```
然后训练一个小型卷积网络，输出 5 个实数。
```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', padding='same',
                 input_shape=(width, width, 3)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(5, activation='linear'))   # 回归 5 个计数
model.compile(loss='mean_squared_error', optimizer='adam')
```
在完整方案里，作者把 patch 放大到 300x300，并用 ImageNet 预训练的 VGG16 做主干，再加大学习率的调度：先用很小的学习率（1e-5）热身，再换 1e-4 加动量 0.9 继续训练，损失从 37 一路降到 0.06 左右。数据增强用了翻转、旋转、改变饱和度和亮度、对比度，正样本与背景样本的比例控制在 1:3。

### 2.4 模型测试与指标报告
在单张图上验证时，我们把所有 patch 的预测相加得到整图计数，再用 RMSE 衡量。
```python
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# 对测试图逐块预测并求和
pred_counts = sum(model.predict(patch) for patch in test_patches)
print("RMSE:", rmse(pred_counts, true_counts))
```
最终按测试图汇总五类计数，生成提交文件。
```python
submission = pd.DataFrame(pred_counts, columns=[
    'adult_males','subadult_males','adult_females','juveniles','pups'])
submission.insert(0, 'test_id', test_ids)
submission.to_csv('submission.csv', index=False)
```

---

## 三、总结
本文中我们完成了斯特勒海狮计数的完整流程。我们首先用 EDA 确认了 948 张训练图的五类计数分布极不均匀，其次利用原图与打点图的差分和 blob 检测把彩色圆点还原成逐格计数，接着把大图缩放到 40% 后切成 100x100（完整方案为 300x300）的 patch 训练一个输出 5 个计数的卷积回归网络，最后把所有 patch 的预测相加并用 RMSE 评估。这个比赛的关键体会是：把「整图数数」拆成「小格数数再求和」，能让问题变得可训练，也方便做数据增强。
