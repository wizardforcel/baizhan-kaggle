# Painter by Numbers

## 一、比赛简介

### 1.1 比赛背景

Painter by Numbers 是 Kaggle 上一类关于艺术图像分类的比赛。比赛的目标是根据图片内容判断出该作品属于哪位画家。比赛的数据来源于各种艺术图像，包括名作复制、WikiArt 上的画作，以及 COCO 数据集的参考图像。

原始的图像数据非常庞大，约 79,433 张图像，包含各种画风和人物。数据中有些文件较大，需要在训练前进行预处理（如压缩、裁剪）。

### 1.2 比赛目标

给定一张图片，预测该图片所属的画家标签。训练数据包含 `all_data_info.csv` 中的图片信息，其中包括文件路径、画家类别等。测试集图片需要输出对应的画家类别。

输入是图像文件路径，输出是画家类别标签。每个画家对应一个编号或名称。

### 1.3 评价标准

比赛使用常规的分类准确率（accuracy）或对数损失（log loss）进行评估，具体以比赛说明为准。准确率越高，模型预测正确的比例越高。

## 二、解题过程

整场比赛的思路可以概括成：先对图像数据做清洗和预处理，再用卷积神经网络提取特征并分类。我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先了解数据集的结构：

```python
# 解压训练图片
!mkdir -p /kaggle/temp/
!cd /kaggle/temp/ && unzip -qo /kaggle/input/painter-by-numbers/train.zip
!cd /kaggle/temp/ && unzip -qn /kaggle/input/painter-by-numbers/replacements_for_corrupted_files.zip
```

由于图片数量庞大，我们先查看 `all_data_info.csv`：

```python
import pandas as pd

df = pd.read_csv('/kaggle/input/painter-by-numbers/all_data_info.csv')
print(df.head())
print('总样本数:', len(df))
```

数据中包含约 79,433 张图像，每张图像对应一个画师标签。

### 2.2 数据清洗

图像数据中包含部分损坏或过大的文件，需要先做清理。我们先删除过大的文件并替换损坏文件：

```python
# 查找大于 64MB 的文件
!find /kaggle/temp -type f -size +64M
```

对于过大的文件，我们将其从 `all_data_info.csv` 中删除以避免训练时的内存溢出。

随后我们对图片进行统一裁剪和缩放：

```python
# 压缩图像以避免内存溢出
!mogrify -resize 300x300^ -path /kaggle/working/train $file
```

由于部分图像损坏导致读取错误，我们用替换文件覆盖：

```python
# 替换损坏文件
cp -f ${file}s $file
```

这样可以得到一个可以安全读取的干净数据集。

### 2.3 模型训练与超参数调优

我们选择使用卷积神经网络进行图像分类。先用预训练模型提取特征：

```python
from torchvision import models
import torch

# 使用预训练 ResNet
model = models.resnet18(pretrained=True)
num_classes = len(set(df['painter']))
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
```

然后进行训练：

```python
# 训练流程
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

我们还尝试了 Style Transfer 相关的数据增强技巧，如 AdaIN，增加了模型的泛化能力。

### 2.4 模型测试与指标报告

在验证集上评估模型：

```python
from sklearn.metrics import classification_report, accuracy_score

# 验证集预测
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print('Test Accuracy:', acc)
```

我们同时计算混淆矩阵，观察哪些画家容易被混淆。

最后将模型应用于测试集：

```python
# 测试集预测
test_preds = model.predict(test_images)
submission = pd.DataFrame({'image_id': test_ids, 'painter': test_preds})
submission.to_csv('submission.csv', index=False)
```

## 三、总结

本文中我们完成了 Painter by Numbers 的完整流程。我们首先用 EDA 摸清了数据：约 8 万张艺术图像，包含多个画家类别。其次我们做了数据清洗，处理了损坏和过大文件，并对图像做了压缩。接着我们用预训练 ResNet 进行训练，并引入了风格迁移相关的数据增强。最后我们在验证集上得到合理准确率，并对测试集进行了预测。

这个比赛给我们的最大启发是：图像数据的前处理（压缩、替换损坏文件）在大模型训练中非常关键，且艺术图像分类受益于迁移学习和风格增强策略。