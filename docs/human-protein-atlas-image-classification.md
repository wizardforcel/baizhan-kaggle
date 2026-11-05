# Human Protein Atlas Image Classification
## 一、比赛简介
### 1.1 比赛背景
Human Protein Atlas 图像分类比赛来源于 Kaggle 上的 `human-protein-atlas-image-classification` 任务。比赛目的是识别细胞图像中蛋白在细胞内的定位位置。该数据集由丹麦 Human Protein Atlas 项目提供，每张图像包含四个通道的显微照片：蛋白、细胞核、微管、内质网。比赛背景强调这是一个多标签分类问题，因为一个蛋白可能同时出现在多个细胞器中。材料中 `human-protein-atlas-eda` 提到这是一个多标签分类任务，每个图像可对应多个标签。
### 1.2 比赛目标
训练文件 `train.csv` 包含 31072 个唯一图像 id 与 `Target` 列。`Target` 列内为 28 个可能标签的编号，以空格分隔。例如 `"16 0"` 表示该图像属于标签 16 和 0。图像存储在 `train/` 目录下，每张图像 id 对应四个 PNG 文件，分别命名为 `red`、`green`、`blue`、`yellow`，需要合并成 4 通道 RGBY 输入。测试集在 `test/` 目录下，需要为每个测试图像预测 28 个标签的概率或二分类结果，提交格式为每行的 28 个概率。材料中 `pretrained ResNet34 with RGBY (0.460 public LB)` 说明任务是多标签分类，输出 sigmoid 概率。
### 1.3 评价标准
比赛使用宏平均 F1 分数（macro F1）。材料中多次提到 `F1 macro`，以及 LB 分数 0.460。宏平均 F1 对 28 个类别一视同仁，即使是稀有的类别也占 1/28 的权重，因此模型需要处理类别不平衡。分数越高越好，公众榜单上的示例分数在 0.46 左右，验证集 F1 约 0.65-0.7。

## 二、解题过程
### 2.1 数据探索（EDA）
我们首先查看数据目录结构和训练表。
```python
import pandas as pd
DATA_DIR = '/kaggle/input/human-protein-atlas-image-classification'
TRAIN_CSV = DATA_DIR + '/train.csv'
df = pd.read_csv(TRAIN_CSV)
print(f"df.shape: {df.shape}")   # (31072, 2)
```
每个 id 对应四张单通道 PNG：蛋白（green）、细胞核（blue）、微管（red）、内质网（yellow）。
```python
text_labels = {
0: "Nucleoplasm",
1: "Nuclear membrane",
2: "Nucleoli",
...
27: "Rods & rings"
}
NUM_LABELS = len(text_labels)  # 28
```
我们统计每个标签的出现次数，发现类别分布极不均衡。稀有类别如 `Nucleoli fibrillar center` 出现次数很少，而 `Cytosol` 非常常见。材料中 `human-protein-atlas-eda` 还分析了每张图像的标签个数分布，发现大多数图像有 1-3 个标签。
### 2.2 数据清洗
数据本身没有缺失值，主要清洗是图像加载和标签编码。我们把四张单通道 PNG 合成 4 通道 RGBY 图像。
```python
from PIL import Image
FILTERS = ['red', 'green', 'blue', 'yellow']
def load_image(image_id, ddir):
    return Image.merge('RGBA', [
        Image.open(f"{ddir}/{image_id}_{f}.png") for f in FILTERS
    ])
```
标签编码采用多热编码。
```python
import torch
def encode(image_labels: str):
    target = torch.zeros(NUM_LABELS)
    for label in image_labels.split():
        target[int(label)] = 1
    return target
```
对于类不平衡，材料采用过采样策略：对稀有类别的图像复制多次，设置 `multi` 列表为每个类别的最小复制倍数。
```python
self.multi = [1,1,1,1,1,1,1,1,
    4,4,4,1,1,1,1,4,
    1,1,1,1,2,1,1,1,
    1,1,1,4]
```
这样训练时稀有类别的样本被多次看到，防止模型偏向常见类别。
### 2.3 模型训练与超参数调优
我们采用预训练的 ResNet34 作为骨干，并把第一层卷积分成 4 通道输入。
```python
import torch.nn as nn
from torchvision import models

network = models.resnet34(pretrained=True)
weight = network.conv1.weight.clone()
network.conv1 = nn.Conv2d(4, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
with torch.no_grad():
    network.conv1.weight[:,:3] = weight
    network.conv1.weight[:,3] = torch.mean(weight, dim=1)
network.fc = nn.Linear(network.fc.in_features, NUM_LABELS)
```
输出层用 sigmoid，不要用 softmax，因为多标签任务允许同时预测多个类别。损失函数使用 Focal Loss 来缓解类别不平衡。
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma
    def forward(self, preds, targets):
        loss = targets * torch.log(preds) * (1 - preds) ** self.gamma \
               + (1 - targets) * torch.log(1 - preds) * preds ** self.gamma
        return -torch.mean(loss)
```
训练策略采用分阶段解冻：先冻结 backbone 仅训练最后全连接层，然后再逐渐解冻前几层。材料中 `pretrained ResNet34 with RGBY (0.460 public LB)` 采用了学习率循环和梯度裁剪，验证集 F1 约 0.65-0.7，最终公共榜 0.460。
```python
# 伪代码：训练循环
for epoch in range(num_epochs):
    train_loss = ...
    val_loss, val_f1 = validate(model, val_loader)
    print(f"Epoch {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, val_score:{val_f1}")
```
材料还提到阈值调优的重要性：默认 0.5 的阈值不是最优的，对每个类别单独搜索最佳阈值可以提升宏 F1。
```python
# 对每个类别优化阈值
thresholds = []
for i in range(NUM_LABELS):
    best_f1 = 0
    best_t = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        pred = (probs[:, i] > t).astype(int)
        f1 = f1_score(true[:, i], pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    thresholds.append(best_t)
```
### 2.4 模型测试与指标报告
在测试集上，我们用相同的 4 通道加载和归一化做推理，得到每个图像 28 个类别的概率。
```python
pred = model.predict(X_test)
submission = pd.DataFrame({
    'Id': test_ids,
    'probabilities': pred
})
# 按材料中的阈值调整后生成最终提交
```
材料报告的指标：使用 ResNet34 + RGBY + 过采样 + Focal Loss 的模型在验证集上 F1 macro 约 0.65-0.7，公共榜 LB 0.460。另一篇材料 `human-protein-atlas-tune-thresholds` 强调阈值调优后可以进一步提升分数。训练过程中 F1 macro 随 epoch 缓慢上升，从 0.02 左右升到 0.22 以上。

## 三、总结
本文中我们完成了 Human Protein Atlas 图像分类的完整流程。我们首先通过 EDA 了解 31072 张 4 通道显微图像的 28 类多标签特性，发现类别严重不平衡；其次整理了 4 通道图像加载与多热标签编码，并对稀有类别做过采样；接着基于预训练 ResNet34 修改第一层为 4 通道输入，采用 sigmoid 输出与 Focal Loss，并分阶段解冻微调；最后在测试集上推理并做了阈值优化。整个过程验证了在多标签医学图像任务中，预训练模型、通道适配、类别平衡和阈值调优是提升宏 F1 的关键。
