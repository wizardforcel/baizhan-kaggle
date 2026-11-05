# Passenger Screening Algorithm Challenge

## 一、比赛简介

### 1.1 比赛背景

Passenger Screening Algorithm Challenge 是由美国交通安全管理局 TSA 发起的比赛。比赛提供旅客安检扫描数据，目标是在安检图像中检测是否携带有危险物品。该数据集包含高分辨率全身扫描图像，每位受检者有多个角度的扫描切片（slices），每张切片为 512x660 像素，包含 16 个不同角度的图像。

比赛数据量大，每位受检者 10MB 到 2GB 不等，总共约 1200 例。比赛鼓励参赛者先从低分辨率图像开始，逐步提升性能。

### 1.2 比赛目标

给定扫描图像，输出每个威胁区域（threat zone）的危险概率。每个受检者的图像经过 22.5 度旋转，共获得 16 张扫描切片。比赛划分了 16 个威胁区域，每张切片中各个威胁区域可能可见。

标签文件为 `stage1_labels.csv`，包含每个受检者每个威胁区域的标签。

### 1.3 评价标准

比赛使用对数损失（log loss）作为评价指标，预测每个威胁区域的概率，损失越低越好。

## 二、解题过程

整场比赛围绕高分辨率扫描图像的威胁检测展开，核心难点是图像尺寸大、需要进行图像分割。我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先了解扫描数据的结构：

```python
import pandas as pd
from matplotlib import pyplot as plt

# 加载威胁区域标签
THREAT_LABELS = 'tsa_datasets/stage1/stage1_labels.csv'
labels = pd.read_csv(THREAT_LABELS)
print(labels.shape)
print(labels.head())
```

标签文件包含每个受检者每个威胁区域的标签，共有 16 个威胁区域。

我们查看威胁区域的分布情况，统计每个区域危险物品的比例。

### 2.2 数据清洗

原始扫描图像较大，需要先进行裁剪以减少计算量。我们将每个扫描图像分割为若干个局部区域（sectors），并只保留包含威胁区域的部分：

```python
import numpy as np

# 16 个区域的顶点坐标
sector01_pts = np.array([[0,160],[200,160],[200,230],[0,230]], np.int32)
sector02_pts = np.array([[0,0],[200,0],[200,160],[0,160]], np.int32)
# ... 其他区域类似
```

我们为每个威胁区域定义它在不同切片中可见的区域列表：

```python
# threat zone 1 在不同切片中的可见区域
zone_slice_list = [
    [sector01_pts, sector01_pts, ..., None, ...],
    ...
]
```

通过这种切片，我们可以将大图像裁剪为 250x250 的子图，减少了处理成本。

### 2.3 模型训练与超参数调优

我们首先使用最简单的基线：按区域均值预测。

```python
# 基于区域均值的预测
zone_means = {}
for zone in range(16):
    zone_data = []
    for sample in samples:
        # 对该区域提取特征
        features = extract_features(sample, zone)
        zone_data.append(features)
    zone_means[zone] = np.mean(zone_data)
```

随后尝试使用卷积神经网络对每个威胁区域进行单独分类。我们提取每个区域的裁剪图像并输入 CNN。

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 构建 CNN 模型
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(250,250,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
model.fit(train_images, train_labels, epochs=10, batch_size=32)
```

我们还尝试使用全扫描图 + 卷积网络的方案，但由于全扫描图过大，往往会超时。

### 2.4 模型测试与指标报告

在验证集上评估模型：

```python
from sklearn.metrics import log_loss

preds = model.predict(X_val)
loss = log_loss(y_val, preds)
print('Validation log loss:', loss)
```

我们最终用基于区域均值的基线得到 0.29089 的 log loss，作为起点。

最终生成预测文件：

```python
# 生成每个样本每个区域的预测
submission = pd.DataFrame({
    'subject_id': subject_ids,
    'zone_1': preds_zone1,
    ...
})
submission.to_csv('submission.csv', index=False)
```

## 三、总结

本文中我们完成了 Passenger Screening Algorithm Challenge 的完整流程。我们首先用 EDA 摸清了数据：扫描图像较大，每位受检者有 16 张切片，分为 16 个威胁区域。其次我们做了数据清洗，将扫描图像分割为局部区域并裁剪到 250x250。接着我们用区域均值基线和 CNN 进行训练。最后我们在验证集上得到约 0.29 的 log loss，并生成预测结果。

这个比赛给我们的最大启发是：对于超大图像数据的竞赛，首先降低数据维度（裁剪、降采样）比直接堆模型更有效，且针对特定区域进行建模是提升性能的关键。