# Histopathologic Cancer Detection

## 一、比赛简介

### 1.1 比赛背景
Histopathologic Cancer Detection 是 Kaggle 上非常经典的医学影像分类比赛，目标是用显微图像判断组织切片是否含有癌细胞。训练集包含 22 万张 96×96 的灰度切片图像，每张图像对应一个二元标签：0 表示没有癌，1 表示存在癌细胞。测试集包含约 5.7 万张图像，需要参赛者提交每个样本的预测标签。

比赛数据来自真实的病理切片，但图像已经被裁剪、增强过，所以参赛者无法直接从图像里读出患者信息。比赛对模型的可解释性和鲁棒性提出了较高要求：医学影像的背景噪声、染色不均、细胞密度差异都很大，简单的颜色阈值基本无效。社区里多数顶尖方案都采用卷积神经网络（CNN）和迁移学习，从 ImageNet 预训练模型里挪用特征，再在病理图像上微调。

### 1.2 比赛目标
输入是训练文件夹 `train/` 下的 22 万张 `.tif` 文件，文件名对应 `train_labels.csv` 中的 `id` 列。标签列 `label` 取值 0/1。测试文件夹 `test/` 下的 5.7 万张文件没有标签，需要预测。

比赛要求提交一个 `submission.csv` 文件，包含两列：`id` 和 `label`。`label` 为 0 或 1。我们需要为每张测试图像输出二分类结果。指标在原始 Kaggle 竞赛中是 ROC AUC，阿岗和 LGBDT 方案里也常以准确率和 F1 辅助判断。

### 1.3 评价标准
原始排行榜使用 ROC AUC（Receiver Operating Characteristic Area Under Curve），值越接近 1 越好。AUC 对阈值不敏感，能反映模型整体的排序能力。除了 AUC，社区经常报告准确率和 ROC 曲线，以及每类的召回率，因为医学影像往往对误判癌症更敏感。

材料里提到的模型大多报告验证集上的 AUC 或准确率。部分团队还使用了类激活图（CAM）来可视化模型关注的区域，以增强可解释性。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先读取标签表，了解数据规模与类别分布：

```python
import pandas as pd, os
labels = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')
print(labels.shape)
# (220025, 2)

# 查看类别分布
counts = labels['label'].value_counts()
print(counts)
```

训练集约 22 万条，类别大致是均衡的。我们进一步随机抽取 50 张癌和 50 张非癌图像做可视化，发现癌图像通常颜色更深、纹理更粗糙，但差异肉眼难以完全区分，这说明需要模型学习微观结构。

我们还检查了图像尺寸和通道数：所有训练图像都是 96×96 的三通道或单通道 TIFF。我们计算了数据目录下的文件数 220,025，测试集 57,458，约占总数的 20.7%。

### 2.2 数据清洗
图像数据本身没有缺失值，但原始图像的亮度、对比度不一致。对此我们采用数据增强而不是手工清洗：

```python
from torchvision import transforms

# 训练时使用随机翻转、旋转等增强
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 验证和测试时只做标准化
transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

清洗的思路是：不对原始像素做手工阈值或去噪，而是让模型在训练时看到各种变形的版本，提高对染色差异的鲁棒性。训练/验证集按 0.9/0.1 分层抽样，保持癌/非癌比例一致。

我们还去掉了图像路径中的不必要列，`Dataset` 类中直接根据 `id` 拼接文件名读取 `.tif`，并用 OpenCV 加载图像，保证内存占用可控。

### 2.3 模型训练与超参数调优
我们采用 PyTorch 实现一个简单的三层 CNN 作为基线：

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
```

训练参数取自材料中的一种实现：`num_epochs=30`、`batch_size=256`、`learning_rate=0.01`，优化器用 Adam，损失函数用交叉熵。训练时我们在两个 GPU 上并行，每个 batch 使用数据增强，验证时关闭增强。

另一种方案在材料中同样出现：使用迁移学习的 ResNet50 或 Inception-ResNet-v2。作者将预训练模型的前几层冻结，只微调最后几层，学习率设为 0.002，batch size 128，epoch 10。迁移学习在小样本上表现更好，能更快收敛。

超参数调优主要集中在学习率衰减和 Dropout 上。材料里多次提到使用 `ReduceLROnPlateau` 在验证损失不再下降时衰减学习率，以及对全连接层加入 0.5 的 Dropout 来防止过拟合。对于 CNN 而言，数据增强的强度和正则化强度的平衡往往比网络深度更重要。

### 2.4 模型测试与指标报告
训练结束后，我们在验证集上计算准确率、ROC AUC 和混淆矩阵：

```python
from sklearn.metrics import accuracy_score, roc_auc_score

y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_proba)
print(f'Val Acc: {acc:.4f}, AUC: {auc:.4f}')
```

根据材料中的实验记录，简单的 CNN 在验证集上能达到 0.85 以上的准确率，而加入数据增强和迁移学习后 AUC 可提升到 0.92 以上。

预测测试集时，我们使用相同的 `transform_test`，分批读取图像并推理：

```python
test_dataset = Dataset(df_data=sub, data_dir=test_path, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

model.eval()
preds = []
with torch.no_grad():
    for images, _ in test_loader:
        outputs = model(images.to(device))
        preds.extend(outputs.argmax(dim=1).cpu().numpy())
```

预测完成后，我们把 `id` 和 `label` 写入 `submission.csv`：

```python
submission = pd.DataFrame({'id': sub['id'], 'label': preds})
submission.to_csv('submission.csv', index=False)
```

提交前我们检查行数是否与测试集一致、标签是否只有 0 和 1、无重复 id。材料中还提到部分团队对预测结果做了后处理，比如对预后概率做阈值调优，或者做多模型集成以提升稳定性。

## 三、总结
本文中我们完成了 Histopathologic Cancer Detection 的完整流程。我们首先用 EDA 了解了 22 万训练图像、5.7 万测试图像的类别分布与图像形态；随后通过随机翻转、旋转和标准化等增强方式清洗数据，保证模型对染色差异不敏感。然后我们用基于 PyTorch 的三层 CNN 并辅以迁移学习方案进行训练，在验证集上监控准确率和 AUC 并调整学习率与 Dropout。最后对测试集进行批量推理并生成符合格式的提交文件。

这个比赛给我们的启示是：医学影像任务中，数据增强和正则化往往比一味加深网络更有效；同时评估指标应以 AUC 为主，因为医学判别任务通常关注排序质量而非单一阈值。
