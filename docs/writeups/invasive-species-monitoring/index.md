# Invasive Species Monitoring

## 一、比赛简介

### 1.1 比赛背景

Invasive Species Monitoring 比赛旨在通过图像自动识别入侵物种。参赛者需要判断一张图像是否包含入侵物种。数据由训练图片和标签组成，阳性样本稀少，属于二分类问题。

数据以目录形式存放，`train/` 下按文件名对应 CSV 里的标签。测试集为未标记图像，需要输出 0/1 概率。

### 1.2 比赛目标

给定一张野生动物图像，输出其为入侵物种的概率。提交文件包含图像 ID 与预测概率。

### 1.3 评价标准

比赛使用 ROC AUC（受试者工作特征曲线下面积）作为评价指标，值越高越好。它衡量模型区分正负样本的能力，与阈值无关。

## 二、解题过程

我们采用迁移学习 + 5 折交叉验证，并在训练时做轻量数据增强。

### 2.1 数据探索（EDA）

首先读取训练标签，查看样本数量与分布。

```python
import pandas as pd
import numpy as np

train_set = pd.read_csv('../input/train_labels.csv')
train_label = np.array(train_set['invasive'])
print('训练样本数:', len(train_set))
print('阳性比例:', train_label.mean())
```

数据显示阳性样本明显少于阴性样本。我们生成文件路径表用于后续读取。

```python
from pathlib import Path
path = "../input/train/"
train_files = [str(Path(path) / f"{int(i)}.jpg") for i in train_set['id']]
```

我们还检查测试集大小，确认提交格式需要概率列。

```python
test_set = pd.read_csv('../input/sample_submission.csv')
print('测试样本数:', len(test_set))
```

### 2.2 数据清洗

图像读取时统一尺寸为 800×800，并做归一化。训练生成器加入随机旋转与翻转增强。

```python
import cv2

def preprocess(img):
    img = cv2.resize(img, (800,800))
    img = img.astype('float32') / 255.
    return img

def augment(img, choice):
    if choice == 0:
        img = np.rot90(img, 1)
    elif choice == 1:
        img = np.flipud(img)
    elif choice == 2:
        img = np.rot90(img, 2)
    elif choice == 3:
        img = np.fliplr(img)
    return img
```

我们不删除异常图像，只做统一预处理，以保持数据多样性。

### 2.3 模型训练与超参数调优

采用 InceptionV3 作为骨干，冻结预训练权重，仅训练顶层分类头。

```python
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model

img_dim = (800,800,3)
input_tensor = Input(shape=img_dim)
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=img_dim)
bn = BatchNormalization()(input_tensor)
x = base_model(bn)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, output)
```

训练使用 5 折交叉验证，Adam 学习率 1e-4，早停与学习率衰减。

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)
for train_index, valid_index in kf.split(train_files):
    # 生成训练/验证生成器
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_generator(), steps_per_epoch=..., epochs=50,
                        validation_data=valid_generator(), callbacks=[EarlyStopping(patience=3)])
```

超参数包括 Dropout 0.5、batch size 5、epochs 50，通过交叉验证得到稳定 AUC。

### 2.4 模型测试与指标报告

每折训练后在验证集上计算 ROC AUC 并累加测试预测。

```python
from sklearn import metrics
roc_auc = metrics.roc_auc_score
valid_score = roc_auc(y_valid, preds_valid)
train_score = roc_auc(y_train, preds_train)
print('Val AUC:', valid_score)
```

所有折的测试预测取平均，写入提交文件。

```python
preds_test /= n_fold
test_set['invasive'] = preds_test
test_set.to_csv('submission.csv', index=False)
```

材料标题显示该方案在排行榜上达到 0.99770，说明迁移学习能有效区分入侵物种。

## 三、总结

本文中我们完成了 Invasive Species Monitoring 的图像二分类。我们首先通过 EDA 确认训练样本数与阳性比例，并构建文件路径表。其次我们统一图像尺寸至 800×800 并加入旋转、翻转增强。接着我们使用 InceptionV3 迁移学习配合 5 折交叉验证训练，Dropout 0.5 防止过拟合。最后在验证集上得到高 AUC，并在测试集上取平均预测生成提交。该方案在小样本图像分类上体现了迁移学习与交叉验证的价值。