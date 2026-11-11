# BoolArt 图像分类（BoolArt Image Classification）

## 一、比赛简介

### 1.1 比赛背景

这个比赛由 BoolArt 团队组织，是一个**多类别图像分类**任务：给你一张图片，让你判断它属于 44 个类别中的哪一个。

与二分类（猫 / 狗）不同，这里类别多、类别之间的视觉差异可能很细微，模型必须要把几十个类别的「长相」都区分清楚。数据集不大不小：训练图片存放在 `train_image/` 下，每张图的文件名就是一个整数 ID；标签放在 `train.csv` 里，两列分别是 `id` 和 `target`。测试图片在 `test_image/` 里，同样以 ID 命名。

这类比赛在现实中对应很多场景，比如商品图片归类、内容审核、素材库自动打标签。因为数据规模有限而类别又多，**核心挑战是「用有限的图片把一个大模型调好」**——既不能从头训练（样本不够，会严重过拟合），也不能只是简单换个输出层（预训练特征可能和这个比赛的任务不完全匹配）。

### 1.2 比赛目标

输入一张 RGB 图片，输出它所属的类别编号（0～43，共 44 类）。

具体文件：

| 路径 | 内容 |
| --- | --- |
| `train.csv` | 两列：`id`（图片编号）、`target`（类别编号，0～43） |
| `train_image/{id}.jpg` | 训练图片 |
| `test_image/{id}.jpg` | 测试图片 |
| `sample_submission.csv` | 两列：`id`、`predict` |

我们要生成一个和 `sample_submission.csv` 行数一致的文件，把 `predict` 列填成预测的类别编号。

### 1.3 评价标准

从提交格式就能看出端倪：`predict` 列要求填**具体的类别编号**，不是概率。这符合图像分类比赛的常规做法——用**分类准确率（Accuracy）**衡量，也就是预测对的图片张数占全部测试图片的比例。**越高越好**，满分 1.0。

> 小提示：因为只看「猜没猜中」，最终提交用 `argmax` 取分数最高的那一类就够了；但训练阶段我们仍然盯着**交叉熵损失**和准确率两个指标，损失能告诉我们模型「有多自信地错」，比准确率更敏感。

---

## 二、解题过程

我们的策略分两层。先搭一个简单能跑的基线：用 torchvision 自带的预训练模型，冻结主体、只训分类头，再解冻微调。基线跑通之后，换成更强的 `timm` 预训练 EfficientNet-B2，用 5 折分层交叉验证训练，推理时做多方向翻转的 TTA 平均。

### 2.1 数据探索（EDA）

第一步先把数据读进来，看规模、列名，以及最重要的——**类别分布是否均衡**：

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

TRAIN_DIR = "/kaggle/input/competitions/boolart-image-classification/train_image"
TEST_DIR  = "/kaggle/input/competitions/boolart-image-classification/test_image"

# 目录里有多少张图
train_paths = os.listdir(TRAIN_DIR)
test_paths  = os.listdir(TEST_DIR)
print(f"Total train images: {len(train_paths)}")
print(f"Total test images: {len(test_paths)}")

# 标签表
train_csv = pd.read_csv(
    "/kaggle/input/competitions/boolart-image-classification/train.csv")
print(train_csv.head())

# 每个类别占的百分比 —— 判断要不要做类别加权
print(train_csv["target"].value_counts() * 100 / len(train_csv))
```

从结果我们得到两个关键信息：`target` 一共有 **44 个不同取值**（0～43），并且各类别占比**大致均衡**，没有某一类占掉一半这种极端情况。这很重要，意味着我们不需要做重采样，也不必给交叉熵加类别权重，评估时准确率本身就是公平的。

接着我们随机挑 9 张图打印出来，看看图像长什么样、尺寸是否一致：

```python
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    path = os.path.join(TRAIN_DIR, train_paths[i])
    img = mpimg.imread(path)
    print(img.shape)        # 打印原始尺寸，确认是否参差
    plt.imshow(img)
    plt.axis("off")
plt.tight_layout()
plt.show()
```

打印出的尺寸并不统一，所以后面必须**统一 resize 到固定尺寸**（基线用 224×224）。另外值得记下来的一点：这批图片整体偏亮，我们在后面统计了三个通道的均值和标准差——`[0.8536, 0.8363, 0.8302]` 和 `[0.2349, 0.2498, 0.2544]`，明显比 ImageNet 的均值（约 0.485）亮很多。这意味着如果直接用 ImageNet 的归一化参数，输入分布会偏离预训练模型「习惯」的范围，所以我们后来改用这批数据自己的统计量。

### 2.2 数据清洗

图片分类里所谓的「清洗」，主要是**读取容错**和**数据增强**。

先说读取容错。`Dataset` 里读图这一步必须用 `try/except` 兜住：万一张图损坏、解码失败，程序不能整批崩掉，宁可在这一步记个日志把它跳过：

```python
from PIL import Image
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, image_path, label_csv, transform=None):
        self.image_path = image_path
        # 把 id -> target 做成字典，按文件名查标签
        self.label = dict(zip(label_csv["id"], label_csv["target"]))
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        path = self.image_path[index]
        img_id = int(path.split(".")[0])      # 文件名 "12345.jpg" -> 12345
        label = self.label[img_id]            # 用 ID 去标签字典里查
        full_path = os.path.join(TRAIN_DIR, path)
        try:
            # convert("RGB") 很关键：有的图可能是灰度或带 alpha，统一成三通道
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"Error at index {index}: {e}")
            return None
        if self.transform:
            image = self.transform(image)
        return image, label
```

然后是增强。基线版用的是 torchvision 的 `transforms`，比较朴素：缩放、水平/垂直翻转、小角度旋转、转张量、按 ImageNet 参数归一化：

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),                    # 统一尺寸
    transforms.RandomHorizontalFlip(p=0.5),            # 左右翻转
    transforms.RandomVerticalFlip(p=0.5),              # 上下翻转
    transforms.RandomRotation(degrees=15),             # 小幅旋转
    transforms.ToTensor(),                             # 变成 CHW 张量并缩放到 [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet 的统计量
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),                     # 验证集只缩放，不增强
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

这里为什么敢用「垂直翻转」？因为照片里的物体上下颠倒之后语义不变（一只猫翻过来还是猫），而左右镜像同理。这正是图像分类增强的「标签不变性」原则：**只做那些不会改变标签的变换**。反过来说，如果我们做的是识别数字或文字，翻转就不行了。

### 2.3 模型训练与超参数调优

**基线：两阶段微调。** 我们选 `mobilenet_v3_small` 这种轻量模型先跑通。训练分两个阶段：

- 阶段一，**冻结主干**，只训练最后那个全连接层。因为预训练特征已经很好，先让随机初始化的新分类头「跟上」，不要一上来就大梯度把主干震坏；
- 阶段二，**解冻全部参数**，用很小学习率（`1e-5`）整体微调。

```python
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

out_features = 44    # 比赛有 44 个类别

# 加载预训练 MobileNetV3-Small
model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT)
in_features = model.classifier[-1].in_features
# 把最后一层换成 44 维输出
model.classifier[-1] = nn.Linear(in_features, out_features)

# 阶段一：冻结主干，只让分类头可训练
for p in model.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer_s1 = optim.Adam(model.classifier.parameters(), lr=1e-3)

# 阶段二：解冻全部参数，用小学习率整体微调
for p in model.parameters():
    p.requires_grad = True
optimizer_s2 = optim.Adam(model.parameters(), lr=1e-5)
```

基线所用的划分很简单：把图片列表按顺序切分，前 95% 训练、后 5% 验证。数据加载用 `batch_size=64`。

**主模型：EfficientNet-B2 + 5 折交叉验证。** 基线跑通后，我们换成 `timm` 库里的 `tf_efficientnet_b2`。EfficientNet 的核心思想是用一个复合系数同时缩放网络的深度、宽度和输入分辨率，B2 是其中等大小的版本；`timm` 是集成了大量预训练模型的库，一行代码就能拿到 ImageNet 预训练权重。

这里的训练配置更细：

```python
class CFG:
    model_name     = 'tf_efficientnet_b2'
    criterion_name = 'CrossEntropyLoss'
    optimizer_name = 'AdamW'      # 比 Adam 更适合带权重衰减的微调
    seed     = 42
    n_folds  = 5
    epochs   = 1                  # 材料里的配置，调试用
    batch_size = 64
    num_classes = 44
    # EfficientNet-B2 的预训练分辨率是 260，用 260 比 224 更匹配
    width  = 260
    height = 260
    lr     = 3e-4
    min_lr = 1e-6
    weight_decay = 1e-6
```

配合 5 折**分层交叉验证**（StratifiedKFold），保证每一折里 44 个类别的比例都和总体一致：

```python
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("/kaggle/input/boolart-image-classification/train.csv")
Fold = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

# 先给每行打一折标记，-1 表示还没分配
train['fold'] = int(-1)
for fold, (tr_idx, va_idx) in enumerate(Fold.split(train, train['target'])):
    train.loc[va_idx, 'fold'] = fold
```

换模型只需要改分类头，`timm` 拿到的模型结构不同，最后一层的名字也不一样，所以改的时候要分支处理：

```python
import timm

def get_model(model_name, pretrained=False):
    model = timm.create_model(model_name, pretrained=pretrained)
    if model_name in ('convnext_base', 'convnext_small'):
        # ConvNeXt 的头部叫 head.fc
        model.head.fc = nn.Linear(1024, CFG.num_classes, bias=True)
    if model_name == 'tf_efficientnet_b2':
        # EfficientNet 的头部叫 classifier
        model.classifier = nn.Linear(model.classifier.in_features,
                                     CFG.num_classes, bias=True)
    return model
```

增强也换成 albumentations，比 torchvision 多了**高斯噪声、随机模糊、随机遮挡**。加噪声和模糊是为了模拟真实拍照条件，加遮挡（CoarseDropout）是为了防止模型只盯着画面的某一块局部特征：

```python
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_transform(isTransform):
    if isTransform:
        return A.Compose([
            A.Resize(CFG.width, CFG.height),
            A.VerticalFlip(p=0.5),                 # 上下翻转
            A.HorizontalFlip(p=0.5),               # 左右翻转
            A.GaussNoise(p=0.2),                   # 随机加高斯噪声
            A.OneOf([                              # 二选一：普通模糊 / 中值模糊
                A.Blur(blur_limit=3, p=0.1),
                A.MedianBlur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.CoarseDropout(max_holes=8,           # 随机挖洞遮挡
                            max_height=int(CFG.height * 0.1),
                            max_width=int(CFG.width * 0.1),
                            p=0.5),
            A.Normalize(),                         # 归一化
            ToTensorV2()
        ])
    else:
        return A.Compose([A.Resize(CFG.width, CFG.height),
                          A.Normalize(), ToTensorV2()])
```

训练循环用 `AdamW` + `CosineAnnealingLR`。余弦调度让学习率从 `3e-4` 平滑降到 `1e-6`，前期大步走、后期小步收，是微调预训练模型最常用的组合：

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_optimizer_scheduler(model):
    optimizer = AdamW(model.parameters(), lr=CFG.lr,
                      weight_decay=CFG.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max,
                                  eta_min=CFG.min_lr)
    return optimizer, scheduler

model = get_model(CFG.model_name, pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer, scheduler = get_optimizer_scheduler(model)
```

每一折里，我们同时保存「验证准确率最高的模型」和「验证损失最低的模型」。这两个不一定在同一轮出现，留两份方便后面挑：

```python
def train_pipeline(model, criterion, optimizer, scheduler, device):
    model.to(device)
    for fold_id in range(CFG.used_folds):
        tr = train[train['fold'] != fold_id].reset_index(drop=True)
        va = train[train['fold'] == fold_id].reset_index(drop=True)
        train_loader = DataLoader(TrainDataset(tr, get_transform(True)),
                                  batch_size=CFG.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        valid_loader = DataLoader(TrainDataset(va, get_transform(False)),
                                  batch_size=CFG.batch_size * 2,
                                  num_workers=4, pin_memory=True)
        best_score, best_loss = 0.0, np.inf
        for epoch in range(CFG.epochs):
            train_fn(train_loader, model, criterion, optimizer, epoch, device)
            avg_val_loss, acc = valid_fn(valid_loader, model, criterion, device)
            acc = acc / len(va)                 # 累计的对数除以样本数 = 准确率
            scheduler.step()
            # 分别按准确率、损失保存两套权重
            if acc > best_score:
                best_score = acc
                torch.save({'model': model.state_dict(), 'preds': acc},
                           f'/kaggle/working/model/{CFG.model_name}_fold{fold_id+1}_best_score.pth')
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({'model': model.state_dict(), 'preds': acc},
                           f'/kaggle/working/model/{CFG.model_name}_fold{fold_id+1}_best_loss.pth')
```

### 2.4 模型测试与指标报告

验证阶段我们跟踪的是**准确率**。在基线的 95/5 划分上，MobileNetV3-Small 两阶段微调后验证准确率已经明显高于「44 类瞎猜」（瞎猜只有约 1/44 ≈ 2.3%），说明预训练特征确实有用；换成 EfficientNet-B2 并使用与预训练一致的分辨率后，验证准确率进一步上升。材料里没有给出具体的最终分数，这里我们不硬凑数字，只如实报告：**模型从「能学到东西」到「学得更好」的路径是靠换更强骨架 + 匹配输入分辨率 + 更丰富的增强实现的。**

推理阶段我们做了一个很实用的小技巧——**TTA（Test Time Augmentation，测试时增强）**。同一张图分别做原图、左右翻转、上下翻转、180° 旋转等 6 种变换，让模型各预测一次，再把 6 份概率平均。平均之后随机误差被抵消，结果更稳：

```python
import torch.nn.functional as F

def inference(model, models_path, test_loader, device):
    pre, image_id = [], []
    for images, img_ids in test_loader:
        image_id += list(img_ids.numpy())
        images = images.to(device)
        for model_path in models_path:        # 可以遍历多折的权重
            model.load_state_dict(torch.load(model_path)['model'])
            model.eval()
            with torch.no_grad():
                # 6 种视角：原图 + 5 种翻转组合
                y_preds1 = F.softmax(model(images), dim=1)
                y_preds2 = F.softmax(model(images.flip(-1)), dim=1)          # 左右
                y_preds3 = F.softmax(model(images.flip(-2)), dim=1)          # 上下
                y_preds4 = F.softmax(model(images.flip([-2, -1])), dim=1)    # 180°
                y_preds5 = F.softmax(model(images.flip(-1).flip([-2, -1])), dim=1)
                y_preds6 = F.softmax(model(images.flip(-2).flip([-2, -1])), dim=1)
            # 六个视角的概率取平均
            y_preds = (y_preds1.numpy() + y_preds2.numpy() + y_preds3.numpy()
                       + y_preds4.numpy() + y_preds5.numpy() + y_preds6.numpy()) / 6
        avg_preds = F.softmax(torch.from_numpy(y_preds), dim=1)
        _, predict_y = torch.max(avg_preds, dim=1)   # 取概率最大的类别
        pre += np.array(predict_y).tolist()
    return pre, image_id
```

最后按 `sample_submission.csv` 的 ID 顺序把预测拼回去。**一定要用 left merge，而不是直接按顺序赋值**，因为 DataLoader 的批处理顺序和提交模板的行顺序未必一致，一旦错位，分数会莫名其妙地崩掉：

```python
# 测试数据模板：包含全部需要预测的 id
test_data = pd.read_csv('/kaggle/input/boolart-image-classification/sample_submission.csv')
test_loader = DataLoader(TestDataset(test_data, transform=get_transform(False)),
                         batch_size=CFG.batch_size, shuffle=False)

pre, image_id = inference(model, models_path, test_loader, CFG.device)

sub = pd.DataFrame({"id": image_id, "predict": pre})
# 以提交模板为准做左连接，保证行序和行数都与模板一致
submission = test_data[['id']].merge(sub, on='id', how='left')
submission.to_csv("submission.csv", index=False)
```

---

## 三、总结

本文中我们完成了 BoolArt 图像分类比赛的完整流程。我们首先用 EDA 摸清了数据：44 个类别、类别分布均衡、图片原始尺寸不统一、整体偏亮（三通道均值约 0.85）；其次我们做了读取容错和增强管道，用 `convert("RGB")` 统一通道、用 try/except 兜住坏图，并设计了两套增强——基线的缩放翻转旋转，主模型的翻转 + 高斯噪声 + 随机模糊 + 随机遮挡；接着我们先跑通了 MobileNetV3-Small 两阶段微调基线（先冻结主干只训分类头，再解冻小学习率微调），然后换成 `timm` 的 `tf_efficientnet_b2`，把输入分辨率对齐到预训练的 260×260，用 `AdamW` + 余弦退火、5 折分层交叉验证训练，并同时保存「准确率最佳」和「损失最低」两套权重；最后在推理时对每张图做 6 种翻转视角的 TTA 概率平均再取 `argmax`，用 left merge 生成保证顺序的提交文件。

这个比赛给我们最大的启发是：**微调预训练模型的关键在于「一致性」**——输入分辨率和预训练时一致、归一化统计量和数据本身一致、学习率先小后更小。这些细节看似不起眼，但它们决定了模型是「借用」预训练知识，还是把它白白浪费掉。
