# PAYT-OLP AI Training Image Detection

## 一、比赛简介

### 1.1 比赛背景

PAYT-OLP AI Training Image Detection 是图像目标检测比赛，参赛者需要对训练图像中的目标进行检测并标注边界框。训练数据包含大量图像和对应的 YOLO 格式标注文件，标签格式为 `class_id x_center y_center width height`，归一化坐标。

数据存储在 `/kaggle/input/competitions/payt-olp-ai-training-image-detection` 下，分类文件夹结构为 `data/train/images`、`data/train/labels`、`data/val/images`、`data/val/labels`。测试集仅提供图像，需要预测的目标检测结果。

### 1.2 比赛目标

给定一张测试图像，输出图像中所有目标的边界框、类别和置信度。每个目标输出 6 个值：`class_id confidence x_center y_center width height`（归一化）。

### 1.3 评价标准

比赛使用 mAP（平均精度）作为评价指标，具体以 IoU 阈值 0.5 为准，mAP 越高越好。提交文件格式参考 `sample_submission.csv`，包含 `image_id` 和 `prediction_string`。

## 二、解题过程

整场比赛围绕目标检测展开，核心是从 YOLO 格式标签转换到 Faster R-CNN 训练格式，并用预训练模型微调。我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先确认数据集路径和文件数量：

```python
from pathlib import Path

ROOT = Path("/kaggle/input/competitions/payt-olp-ai-training-image-detection")
IMG_TRAIN = ROOT / "data/train/images"
LAB_TRAIN = ROOT / "data/train/labels"

print("训练图像数量:", len(list(IMG_TRAIN.iterdir())))
print("训练标签数量:", len(list(LAB_TRAIN.iterdir())))
```

确认训练图像和标签数量匹配。

### 2.2 数据清洗

我们加载样本并进行 YOLO 到 Faster R-CNN 的格式转换：

```python
def yolo_to_frcnn(text, width=1920, height=1080):
    boxes, labels = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])
        # 归一化坐标转绝对坐标
        xc, bw = xc * width, bw * width
        yc, bh = yc * height, bh * height
        x1 = max(0.0, min(xc - bw / 2, width))
        y1 = max(0.0, min(yc - bh / 2, height))
        x2 = max(0.0, min(xc + bw / 2, width))
        y2 = max(0.0, min(yc + bh / 2, height))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            labels.append(cls + 1)
    return boxes, labels
```

对于标签文件名与图像文件名不一致的情况，我们通过 stem 对齐：

```python
def load_samples(img_dir, label_dir):
    image_map = {file.stem: file.name for file in img_dir.iterdir()}
    samples = []
    for label_file in sorted(label_dir.glob("*.txt")):
        image_name = image_map.get(label_file.stem)
        if image_name is None:
            continue
        boxes, labels = yolo_to_frcnn(label_file.read_text())
        if boxes:
            samples.append((image_name, boxes, labels))
    return samples

train_samples = load_samples(IMG_TRAIN, LAB_TRAIN)
```

### 2.3 模型训练与超参数调优

我们选用 Faster R-CNN 模型，其对通用物体检测效果良好。采用 ResNet50-FPN 作为骨干网络，使用 ImageNet 预训练权重初始化骨干部分，主干不使用预训练权重：

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(
    weights=None,
    weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
    num_classes=2,
).to(DEVICE)
```

训练参数设置：

```python
BATCH_SIZE = 4
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

使用 AdamW 优化器，学习率设为 0.001：

```python
optimizer = AdamW(model.parameters(), lr=0.001)
```

训练时加入了数据增强以提升泛化能力：

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.15, 0.15),
        scale=(0.8, 1.2),
        shear=10,
    ),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.ToTensor(),
])
```

验证集仅使用 ToTensor，保持原始尺寸。训练数据集通过自定义 DetectionDataset 类加载，每次返回图像、边界框、类别标签、区域面积和 iscrowd 标记。

### 2.4 模型测试与指标报告

在验证集上测试模型：

```python
model.eval()
for image, target in val_loader:
    outputs = model(image)
```

评估完成后，我们对测试集进行推理并生成提交文件：

```python
sample_submission = pd.read_csv(SUB_PATH, dtype={"image_id": str})

# 验证测试图像是否存在
test_file_map = {file.stem: file.name for file in IMG_TEST.iterdir()}
missing_ids = [image_id for image_id in sample_submission["image_id"] if image_id not in test_file_map]
if missing_ids:
    raise FileNotFoundError(f"未找到测试图像: {missing_ids[:5]}")
```

提交文件按要求包含 `image_id` 和 `prediction_string`，其中 prediction_string 为多个目标的类、置信度、归一化坐标拼接。模型对每个测试图像输出检测结果后，将结果按样本 ID 整理到提交 CSV。

## 三、总结

本文中我们完成了 PAYT-OLP AI Training Image Detection 的完整流程。我们首先用 EDA 摸清了数据：训练集包含大量图像和 YOLO 格式标签。其次我们做了数据清洗，将 YOLO 标签转换为 Faster R-CNN 可用的格式。接着我们用 Faster R-CNN 进行了训练，并用 AdamW 优化器与学习率调度进行超参数调优。最后我们在测试集上进行推理并生成符合要求的提交文件。

这个比赛给我们的最大启发是：目标检测竞赛中，标签格式转换是关键的第一步，选择合适的检测模型并做好数据对齐可以极大提高训练效率与最终指标。