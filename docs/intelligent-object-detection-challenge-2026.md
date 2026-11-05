# Intelligent Object Detection Challenge 2026

## 一、比赛简介

### 1.1 比赛背景

Intelligent Object Detection Challenge 2026 是一个面向通用目标检测的比赛。竞赛提供基于 VOC 格式的标注数据和 YOLO26 基线代码包，参赛者需要在此基础上训练检测模型并在测试集上预测边界框和类别。

基线包包含 `dataset_voc` 目录，下设 `VOCdevkit`、annotations 与提交样例。annotation 文件采用 JSON 格式，分别对应训练、验证和测试清单，测试清单仅含图像路径与 ID，无标签。

### 1.2 比赛目标

给定训练集图像与对应的 `voc_train.json` 标注，模型需要学会在任意输入图像中定位并分类目标。测试集提供 `voc_test_manifest.json` 与 `sample_submission.csv`，我们需要为每张测试图输出检测结果列表。

输入包含图像文件和标注 JSON，输出为每张图的检测框、置信度与类别。

### 1.3 评价标准

目标检测常用评价指标为 mAP（mean Average Precision），在不同 IoU 阈值下计算各类别的平均精度再求均值，分数越高越好。该比赛要求一次性提交所有测试图预测，排行榜按 mAP 排序。

## 二、解题过程

我们基于提供的 YOLO26 基线，按照 EDA、数据检查、训练调参、测试评测四步进行。

### 2.1 数据探索（EDA）

先确认数据目录结构和标注文件是否齐全。

```python
import os
base = 'IODC2026_baseline/dataset_voc'
# 检查训练标注
train_ann = os.path.join(base, 'annotations/voc_train.json')
val_ann = os.path.join(base, 'annotations/voc_val.json')
test_manifest = os.path.join(base, 'annotations/voc_test_manifest.json')
print('train exists:', os.path.exists(train_ann))
print('val exists:', os.path.exists(val_ann))
print('test exists:', os.path.exists(test_manifest))
```

我们读取 `voc_train.json` 的前几条记录，看类别数量和标注格式。

```python
import json
with open(train_ann) as f:
    data = json.load(f)
# 假设格式为列表，每条包含 image_id, boxes, labels
print('样本数:', len(data))
print('示例样本键:', data[0].keys())
# 类别集合
labels = set()
for item in data:
    labels.update(item['labels'])
print('类别数:', len(labels))
```

通过 EDA 可以了解图像分辨率分布、类别平衡情况，为后处理提供依据。

### 2.2 数据清洗

VOC 格式与 YOLO 格式不同，需要把 JSON 标注转换为 YOLO txt 格式。清洗主要包括坐标归一化、框过滤和类别映射。

```python
def voc_to_yolo(item, img_w, img_h):
    # item['boxes'] 为 [x_min, y_min, x_max, y_max] 绝对坐标
    lines = []
    for bbox, label in zip(item['boxes'], item['labels']):
        x_min, y_min, x_max, y_max = bbox
        # 转为中心坐标并归一化
        x_center = ((x_min + x_max) / 2) / img_w
        y_center = ((y_min + y_max) / 2) / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        lines.append(f"{label} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return '\n'.join(lines)
```

我们还对空标注样本进行检查，剔除无效框并保证训练集与验证集不泄漏。

### 2.3 模型训练与超参数调优

基线提供 `train_yolo.py`。我们首先解压基线包并按要求安装依赖，然后运行训练脚本。

```python
# 伪代码：启动基线训练
# 在项目根目录执行
# python train_yolo.py --data dataset_voc --epochs 50 --batch 16 --img 640
```

训练过程使用 YOLO26 架构，损失包含分类损失、定位损失与置信度损失。我们可以在训练脚本中调整学习率与早停。

```python
# 在 train_yolo.py 中配置超参数
config = {
    'lr': 0.001,
    'batch_size': 16,
    'img_size': 640,
    'epochs': 50,
    'weight_decay': 0.0005
}
```

我们采用验证集 `voc_val.json` 进行在线监控，观察 mAP、precision 与 recall 的变化，必要时降低学习率。

### 2.4 模型测试与指标报告

训练完成后，使用 `validate_yolo.py` 在验证集评估，并用 `predict_yolo.py` 生成测试预测。

```python
# 评估验证集
# python validate_yolo.py --weights best.pt --data dataset_voc --split val
# 输出 mAP@0.5 和 mAP@0.5:0.95
```

预测测试集并生成提交文件：

```python
# python predict_yolo.py --weights best.pt --manifest voc_test_manifest.json --out submission.csv
```

我们最后检查提交文件行数是否与测试清单一致，列名是否匹配样例。

```python
import pandas as pd
sub = pd.read_csv('sample_submission.csv')
pred = pd.read_csv('submission.csv')
print('样本数一致:', len(sub) == len(pred))
```

## 三、总结

本文中我们基于 IODC2026 YOLO26 基线完成了目标检测任务。我们首先检查了 VOC 格式数据目录与 JSON 标注，确认类别与样本数。接着编写 VOC 到 YOLO 的坐标转换清洗脚本，统一归一化格式。随后运行基线训练脚本进行 YOLO26 模型训练，并在验证集上监控 mAP。最后使用预测脚本生成提交文件并校验格式。该流程为快速起步的基线方案，后续可通过数据增强与超参数调优进一步提升 mAP。