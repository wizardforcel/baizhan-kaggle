# PASCAL VOC

## 一、比赛简介

### 1.1 比赛背景

PASCAL VOC 是计算机视觉领域最经典的目标检测数据集之一，包含 20 类常见物体，如人、汽车、动物等。比赛允许使用 YOLOv8n 或 YOLOv8s 作为基线模型，不能使用更重的模型。

任务是基于提供的训练和验证集，对验证集图像进行目标检测，输出每个物体的边界框、类别和置信度。

### 1.2 比赛目标

给定训练集图像和标注，训练目标检测模型，然后在验证集上对每张图像输出检测结果。输出格式为每个检测项包含类别、置信度、归一化边界框坐标（x_min, y_min, x_max, y_max）。最终结果需要按特定格式提交。

训练数据存储在 `dataset.yaml` 文件中指定路径，验证集图像放在 `val/images` 目录下。

### 1.3 评价标准

比赛使用平均精度均值 mAP（Mean Average Precision）作为评价指标，mAP 越高表示检测效果越好。评价在不同 IoU 阈值下进行，通常使用 mAP@0.5 或 mAP@[0.5:0.95]。

## 二、解题过程

整场比赛的思路是：先根据比赛要求用 YOLOv8 训练模型，再在验证集上进行推理生成提交文件。我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先确认数据目录是否存在：

```python
import os

DATASET_ROOT = "/kaggle/input/competitions/pascal-voc"
print("数据集目录:", DATASET_ROOT)
print("是否存在:", os.path.exists(DATASET_ROOT))

# 列出验证集图像
VAL_IMAGES_DIR = os.path.join(DATASET_ROOT, "val", "images")
image_files = sorted([
    f for f in os.listdir(VAL_IMAGES_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
print("验证集图像数量:", len(image_files))
```

数据目录下有训练集和验证集，验证集图像数量符合预期。

### 2.2 数据清洗

我们确认数据文件是否完整：

```python
# 检查各目录是否存在
for p in [
    "/kaggle/input/competitions/pascal-voc",
    "/kaggle/input/pascal-voc",
]:
    print(f"{p} 存在: {os.path.exists(p)}")
```

如发现数据未挂载，可手动复制到工作目录：

```python
import shutil
DST_DATA = "/kaggle/working/pascal-voc"
if os.path.exists(DST_DATA):
    shutil.rmtree(DST_DATA)
    print("旧文件夹已删除")
shutil.copytree("/kaggle/input/competitions/pascal-voc", DST_DATA)
print("val 内容:", os.listdir(os.path.join(DST_DATA, "val")))
print("train 内容:", os.listdir(os.path.join(DST_DATA, "train")))
```

这样确保训练和验证数据都能正常访问。数据目录下包含 images 和 annotations，训练集和验证集均按标准 VOC 格式组织。

### 2.3 模型训练与超参数调优

比赛限制模型只能使用 yolov8n.pt 或 yolov8s.pt。我们选择较轻的 yolov8n：

```python
from ultralytics import YOLO

# 加载预训练 YOLOv8n
model = YOLO("yolov8n.pt")

# 开始训练
model.train(
    data=TRAIN_YAML,
    epochs=15,
    imgsz=640,
    batch=16,
    name="student_baseline",
    exist_ok=True,
)
```

训练 15 个 epoch，图像尺寸设为 640，批大小 16。

### 2.4 模型测试与指标报告

训练完成后，对验证集进行推理：

```python
rows = []
for img_file in image_files:
    img_path = os.path.join(VAL_IMAGES_DIR, img_file)
    results = model.predict(img_path, conf=0.25, verbose=False)

    pred_parts = []
    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            h, w = r.orig_shape
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                pred_parts.append(f"{cls_id} {conf:.4f} {x1/w:.4f} {y1/h:.4f} {x2/w:.4f} {y2/h:.4f}")

    pred_string = " ".join(pred_parts)
    if pred_string == "":
        pred_string = "0 0.0 0.0 0.0 0.0 0.0"

    rows.append({"image_id": img_file, "prediction_string": pred_string})

pd.DataFrame(rows).to_csv("submission.csv", index=False)
print("submission.csv 准备完成。")
```

推理时置信度阈值设为 0.25，边界框坐标归一化到 0-1，如果无检测则输出默认值。

## 三、总结

本文中我们完成了 PASCAL VOC 目标检测的完整流程。我们首先用 EDA 确认了数据目录和图像数量。其次我们做了数据清洗，确保训练和验证集正确挂载。接着我们用 YOLOv8n 在数据上训练 15 个 epoch，最后在验证集上进行推理并生成提交文件。

这个比赛给我们的最大启发是：在竞赛条件下，模型选择受限时，如何通过调整训练参数和推理阈值来最大化性能，同时保证提交格式严格符合要求。