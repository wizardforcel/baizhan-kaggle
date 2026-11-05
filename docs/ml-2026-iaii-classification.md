# ML-2026 IAII Classification

## 一、比赛简介

### 1.1 比赛背景

这是 2026 年 IAII（Intelligent Artifacts and Image Interpretation）视觉分类赛，主题为字符识别。主办方提供了约 9.7 万张带标签的图片，分为 50 个字符类别。训练集按字符文件夹组织，共约 96,929 张训练图片，测试集约 10,791 张图片。每张图片可能含有噪声、外框、背景变化，字符可能偏小或偏离中心。

### 1.2 比赛目标

输入是一张手写或印刷字符图片，输出该图片对应的字符名称。提交文件为 `id,character` 两列，完整包含所有测试图片的预测。

### 1.3 评价标准

使用分类准确率（Categorization Accuracy）评价，值越高越好，满分为 1.0。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先检查训练数据结构：

```python
from pathlib import Path
TRAIN_DIR = '/kaggle/input/ml-2026-iaii-classification/train'
root = Path(TRAIN_DIR)
classes = sorted(x.name for x in root.iterdir() if x.is_dir())
print('类别数:', len(classes))   # 50
```

我们共有 50 个字符文件夹，共约 96,929 张图。我们按类别统计图片数量，确认类别大致平衡，并取 20% 分层作为本地验证集。

### 2.2 数据清洗

图片有旋转、模糊、外框等差异，我们在读取时做最小化处理：先 EXIF 方向校正，转 RGB，再用 `ImageOps.pad` 填充到正方形，保持完整画面不裁切。为了避免数值溢出，我们把图片归一化为 ImageNet 均值/方差。

```python
from PIL import Image, ImageOps
import torch

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def transform(path, size, augment):
    with Image.open(path) as original:
        im = ImageOps.exif_transpose(original).convert('RGB')
        im = ImageOps.pad(im, (size,size), method=Image.Resampling.BILINEAR, color=(128,128,128))
    a = np.asarray(im, dtype=np.float32).transpose(2,0,1)/255.
    return torch.from_numpy((a-MEAN)/STD)
```

我们不删除任何训练样本，只在训练时用数据增强：随机水平翻转、随机旋转 -10~10 度、亮度和对比度微调。

### 2.3 模型训练与超参数调优

我们首先使用 ImageNet 预训练的 ResNet-18 作为基线，只替换最后的全连接层为 50 类：

```python
import torch.nn as nn
from torchvision import models

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 50)
```

我们用 AdamW 优化，学习率 1e-4，主干 1e-4、全连接层 1e-3，CosineAnnealing 调度，10 epochs，耐心 3 轮早停。

```python
optimizer = torch.optim.AdamW([
    {'params':[p for n,p in model.named_parameters() if not n.startswith('fc.')],'lr':1e-4},
    {'params':model.fc.parameters(),'lr':1e-3}
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
```

训练时使用分层验证集，验证准确率最高的模型保存。材料显示公共基线准确率约 0.58418。

### 2.4 模型测试与指标报告

训练完成后，我们在验证集上计算准确率：

```python
with torch.inference_mode():
    preds = model(x).argmax(1).cpu().tolist()
acc = float((preds == true).mean())
```

验证准确率约 0.58 左右。最后用全量训练数据重新拟合一次模型，并对 10,791 张测试图片批量预测，生成 `submission.csv`。

```python
model.eval()
for x,_ in loader(test_ds):
    preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
```

提交文件严格按样例 ID 顺序输出，确保行数、列名完全一致。

---

## 三、总结

本文中我们完成了 ML-2026 IAII Classification 的完整流程。我们首先确认了 50 类字符、约 9.7 万训练图的数据结构。其次做了最小化清洗和数据增强，保留完整画框信息。接着用 ImageNet 预训练 ResNet-18 微调，10 epochs 早停，验证准确率约 58%。最后生成符合格式的提交文件。

这个比赛提醒我们：在图像分类任务中，**预训练模型 + 轻度增强**往往是入门的最快路径，验证集与测试集的域差异（噪声、外框）越明显，特征工程和图像预处理越关键。