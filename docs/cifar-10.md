# CIFAR-10 图像分类

## 一、比赛简介

### 1.1 比赛背景

CIFAR-10 是计算机视觉领域最有名的入门数据集之一：**60,000 张 32×32 的彩色图片，分成 10 个类别**——飞机（airplane）、汽车（automobile）、鸟（bird）、猫（cat）、鹿（deer）、狗（dog）、青蛙（frog）、马（horse）、船（ship）、卡车（truck），每个类别 6,000 张。

这些图片都是真实世界的彩色照片，但分辨率只有 32×32，比手机图标还小。小尺寸带来两个后果：一是同一类物体姿态、光照、背景差异很大，模型很难只靠颜色或轮廓区分；二是不同类之间又很像，比如猫和狗、汽车和卡车，在 32×32 下经常只差几个像素。

Kaggle 上的版本把数据拆成训练集 50,000 张和测试集 10,000 张，训练集带标签，测试集不带，需要提交预测。这个数据集之所以经典，是因为它足够小，能在单卡上几小时跑完；又足够难，能真实反映一个模型的好坏。

### 1.2 比赛目标

Kaggle 给的文件有四个：

- `train.7z`：50,000 张训练图，文件名是数字编号（如 `1.png`）；
- `test.7z`：10,000 张测试图；
- `trainLabels.csv`：两列 `id`、`label`，把训练图的编号和类别名对应起来；
- `sampleSubmission.csv`：提交示例，两列 `id`、`label`。

我们要为测试集的每张图预测一个类别名，按 `id,label` 的格式提交。

### 1.3 评价标准

比赛使用**分类准确率（Accuracy）**：预测对的图片数 ÷ 总图片数。10 分类，随机猜的准确率是 0.1。准确率越高越好，满分 1.0。

---

## 二、解题过程

我们的主线是：**不做从零训练，而是加载 ImageNet 预训练的 ResNet34 做微调。** 32×32 的图先放大到 224×224 以适配 ResNet 的输入，训练时用 CIFAR-10 专用的自动增强（AutoAugment），并且给网络主体和分类头设不同的学习率。材料里用这套默认配置就拿到了约 96% 的准确率，距离该数据集的最好水平只差约 3 个百分点。

### 2.1 数据探索（EDA）

我们首先解压数据并读标签，确认图片数量、类别和尺寸：

```python
import py7zr
import shutil
from pathlib import Path
import pandas as pd

root = Path("./data")
input_path = Path("../input/cifar-10")

# 用 py7zr 解压两个 .7z 文件
with py7zr.SevenZipFile(input_path / "train.7z", mode="r") as z:
    z.extractall(root)
with py7zr.SevenZipFile(input_path / "test.7z", mode="r") as z:
    z.extractall(root)
shutil.copy(input_path / "trainLabels.csv", root / "trainLabels.csv")

# 标签表：id 是图片编号，label 是类别名
labels = pd.read_csv(root / "trainLabels.csv")
print(labels.shape)                    # (50000, 2)
print(labels.label.value_counts())     # 每类 5000 张，完全均衡
```

`value_counts()` 显示 10 个类别各 5,000 张，**训练集完全均衡**。这意味着不需要做重采样或类别加权，评价指标用准确率也不会被多数类带偏。

我们还需要确认图片尺寸和通道数。CIFAR-10 的图统一是 32×32 的 RGB 三通道，`1.png` 这样的命名只是编号，不含任何类别信息，所以**标签必须靠 `trainLabels.csv` 去查**：

```python
from PIL import Image

img = Image.open(root / "train" / "1.png")
print(img.size, img.mode)    # (32, 32) RGB
```

这一点很重要：文件名没有语义，如果我们误以为文件夹名就是类别，就会把标签搞错。

### 2.2 数据清洗

这份数据的「清洗」不是去噪声，而是**把散落的图片整理成 PyTorch 能直接读的目录结构**。`ImageFolder` 要求每个类别一个子文件夹，文件夹名就是类别名。我们按 9:1 把 50,000 张训练图分成训练集和验证集，同时复制一份「训练+验证」全集，供最后重训使用：

```python
import os
from random import random

def copy_file(src_dir, dst_dir, filename):
    """把一张图复制到目标类别目录，目录不存在就创建"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_dir / filename, dst_dir / filename)

def organize_train_valid(root, labels, valid_probability=0.1):
    src = root / "original_train"
    for entry in os.scandir(src):
        if not entry.is_file():
            continue
        # 文件名去掉扩展名就是图片 id
        img_id = entry.name.split(".")[0]
        # 用 id 去标签表里查类别名
        img_class = labels[labels.id == int(img_id)].label.values[0]
        # 10% 的概率分到验证集，其余进训练集
        channel = Path("train") if random() > valid_probability else Path("valid")
        copy_file(src, root / channel / img_class, entry.name)
        # 同时复制进 train_valid，供最后全量重训
        copy_file(src, root / "train_valid" / img_class, entry.name)
```

测试集没有标签，但 `ImageFolder` 也要求有子目录，所以我们建一个假的 `undefined` 类，预测时忽略它即可：

```python
def organize_test(root):
    src = root / "original_test"
    for entry in os.scandir(src):
        if entry.is_file():
            copy_file(src, root / "test" / "undefined", entry.name)
```

接着我们要算每个颜色通道的均值和标准差，用来做标准化。这里用「分批统计再求平均」的办法，避免一次性把 50,000 张图读进内存：

```python
import torch

# 先算一个粗略的逐通道均值/标准差（分批累加）
mean = torch.zeros(3)
std = torch.zeros(3)
n = 0
for imgs, _ in dataloader:
    b = imgs.size(0)
    mean += imgs.mean(dim=(0, 2, 3)) * b
    std  += imgs.std(dim=(0, 2, 3)) * b
    n += b
mean /= n
std  /= n
```

标准化之后，每个通道的数值大致服从均值 0、方差 1，网络训练会稳定很多。

### 2.3 模型训练与超参数调优

**数据增强。** 训练时我们先把 32×32 放大到 224×224（ResNet 要求输入至少这么大），然后用 `AutoAugment` 的 CIFAR-10 策略自动做翻转、裁剪、颜色扰动等组合；验证和测试只做缩放和标准化，不做随机增强：

```python
import torchvision

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),   # 适配 ResNet 输入
    # CIFAR-10 专用的自动增强策略，随机组合多种几何/颜色变换
    torchvision.transforms.AutoAugment(
        policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

valid_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])
```

**模型。** 我们加载 ImageNet 预训练的 ResNet34，把最后的全连接层从 1000 类改成 10 类：

```python
net = torchvision.models.resnet34(pretrained=True)
# 替换分类头：ResNet34 最后一层名字是 fc
num_features = net.fc.in_features
net.fc = torch.nn.Linear(num_features, 10)
```

**分层学习率。** 这是微调的关键技巧。网络主体已经在 ImageNet 上学到了通用特征，只需要小步微调；而新的分类头是随机初始化的，需要更大的学习率快速学会。我们用 PyTorch 的**参数组**给两部分设不同学习率：

```python
lr, weight_decay, epochs = 1e-5, 5e-4, 20

# 主体参数：小学习率
params_1x = [p for name, p in net.named_parameters() if not name.startswith("fc.")]
# 分类头参数：学习率放大 10 倍
optimizer = torch.optim.Adam(
    [{"params": params_1x},
     {"params": net.fc.parameters(), "lr": lr * 10}],
    lr=lr,                    # 主体默认 1e-5
    weight_decay=weight_decay # L2 正则，抑制过拟合
)
```

这里 `weight_decay=5e-4` 相当于给权重加了 L2 惩罚，防止模型在 5 万张小图上过拟合；`lr*10` 就是平移学习率，让新分类头学得更快。

**训练循环。** 每个 epoch 在训练集上前向、反向、更新，然后在验证集上算损失和准确率：

```python
def train(net, train_loader, valid_loader, criterion, optimizer, epochs, device):
    for epoch in range(1, epochs + 1):
        net.train()
        train_correct = torch.tensor(0., device=device)
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = net(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            # 统计本 batch 猜对了多少张
            train_correct += (preds.argmax(dim=1) == y).sum()

        # 验证阶段：关闭 dropout/batchnorm 的随机性
        net.eval()
        valid_correct = torch.tensor(0., device=device)
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                preds = net(X)
                valid_correct += (preds.argmax(dim=1) == y).sum()
        print(f"Epoch {epoch}: train {100*train_correct/len(train_loader.dataset):.2f}% "
              f"valid {100*valid_correct/len(valid_loader.dataset):.2f}%")
```

**关于调参。** 材料里用的是「默认超参数」——学习率 1e-5、权重衰减 5e-4、20 个 epoch、batch size 128。作者明确说没有做大规模网格搜索，因为 ResNet34 预训练模型本身已经足够强，继续调参收益有限。我们认同这个判断：**在预训练模型上，数据增强和分层学习率比细调学习率更重要**。

### 2.4 模型测试与指标报告

先在 10% 的验证集上看训练过程。材料记录的结果是：

- 第 1 个 epoch：训练准确率 68.24%，验证准确率 90.35%；
- 第 2 个 epoch：训练 85.88%，验证 93.49%；
- 第 3 个 epoch：训练 89.37%，验证 94.52%；
- 第 20 个 epoch：训练 91.15%，验证 **95.05%**。

注意第一轮验证准确率就高达 90%，而训练准确率只有 68%——这不是过拟合，而是因为训练时开了强增强（图被随机变换，难度高），验证时用的是干净图。**验证准确率高于训练准确率，在强增强的微调里是正常现象**。

确认模型可用后，我们用「训练+验证」的 50,000 张图重新训练一次，让模型见到尽可能多的数据。20 个 epoch 后训练准确率达到 **97.29%**，总训练时间约 3,338 秒：

```python
# 在 train_valid 全集上重新构建 dataloader，重复上面的微调流程
train_valid_dataloader = torch.utils.data.DataLoader(
    train_valid_dataset, batch_size=128, shuffle=True,
    num_workers=2 * num_gpus, pin_memory=True)

# 重新初始化一个预训练 ResNet34，用同样的分层学习率训练
net = torchvision.models.resnet34(pretrained=True)
net.fc = torch.nn.Linear(net.fc.in_features, 10)
# ...（重新设优化器并调用 train）
```

最后对测试集逐批预测，把数字编号映射回类别名并提交：

```python
import pandas as pd

preds, ids = [], []
net.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        X = X.to(device)
        # 取概率最大的类别索引
        preds.extend(net(X).argmax(dim=1).cpu().numpy())
        ids.extend(y.tolist())      # 这里 y 存的是图片 id

# 把类别索引翻译成类别名（ImageFolder 的 classes 列表）
df = pd.DataFrame({"id": ids, "label": preds})
df["label"] = df["label"].apply(lambda x: train_dataset.classes[x])
df.to_csv("submission.csv", index=False)
```

材料里最终在 Kaggle 上得到约 **96%** 的测试准确率，与验证集的 95.05% 非常接近，说明验证集划分可信、没有明显泄漏。

---

## 三、总结

本文中我们完成了 CIFAR-10 图像分类的完整流程。我们首先用 EDA 确认了 50,000 张训练图、10 个类别各 5,000 张、图片统一为 32×32 RGB，且标签只存在于 `trainLabels.csv`。其次我们把散落的图片整理成按类别分文件夹的结构，按 9:1 切出验证集，并用分批统计算出每个通道的均值和标准差做标准化。接着我们把图片放大到 224×224，用 CIFAR-10 专用的 AutoAugment 做增强，加载 ImageNet 预训练的 ResNet34、替换成 10 类分类头，用 Adam 加分层学习率（主体 1e-5、分类头 1e-4）和 5e-4 权重衰减训练 20 个 epoch。最后我们在验证集上得到 95.05% 的准确率，在全量数据重训后训练准确率达到 97.29%，提交到 Kaggle 约 96%。

这个比赛给我们最直接的体会是：**预训练模型 + 合适的数据增强，往往比从零设计网络更划算**。32×32 的小图放大到 224×224 看似浪费算力，但换来了 ImageNet 上亿张图学到的通用特征，三轮就能到 94% 以上。材料里还有从零实现的 LeNet、VGG、ResNet18、SENet 等方案，它们的训练曲线更能说明「没有预训练时收敛有多慢」，也正好反衬出迁移学习的价值。
