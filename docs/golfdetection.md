# Golf Detection

## 一、比赛简介

### 1.1 比赛背景

Golf Detection（BoolArt Golf Detection）是一个高尔夫场景的目标检测比赛。与「判断这张图是不是高尔夫球场」的分类任务不同，它的核心是在画面中找出目标并给出边界框，属于典型的计算机视觉目标检测问题。比赛数据是一批高尔夫场景图片，配套 YOLO 风格的标注文件。

数据分为 `train_images/train_images`、`train_labels/train_labels` 与 `test_images/test_images` 三个目录。标注文件是纯文本，每行一个目标，格式为 `class_id center_x center_y width height`，其中坐标都除以了图像宽高做了归一化——这与 YOLOv5 官方文档的格式一致。

### 1.2 比赛目标

输入一张图片，输出画面中所有目标的类别与边界框（左上角、右下角坐标）。训练集共 3 个类别。测试集只给图片，需要按比赛要求生成检测框。数据规模不大，因此材料中的方案都只训练了很少的轮次作为演示。

### 1.3 评价标准

比赛按检测的平均精度（mAP）评分，重点看预测框与真实框的重合程度。材料中的评估代码同时打印 `mAP@.5`（IoU 阈值 0.5）与 `mAP@.5:.95`（IoU 从 0.5 到 0.95 共 10 档取平均），分数越高越好。IoU 可以理解为「两个框的交集面积除以并集面积」，完全重合为 1。

---

## 二、解题过程

我们首先尝试了 YOLOv4 方案，其次用 Faster R-CNN 做了对照。下面按 EDA、清洗、训练调参、测试报告的顺序写。

### 2.1 数据探索（EDA）

先看标注格式与坐标范围。读取一个标注文件并解析成像素坐标：

```python
h, w, _ = image.shape
boxes, class_ids = [], []
for label_info in open(label_file).readlines():
    class_id, cx, cy, bw, bh = label_info.strip().split()
    class_id = int(class_id) + 1          # 0 留给背景
    cx, cy = int(w * float(cx)), int(h * float(cy))
    bw, bh = int(w * float(bw)), int(h * float(bh))
    top_left     = (int(cx - bw / 2), int(cy - bh / 2))
    bottom_right = (int(cx + bw / 2), int(cy + bh / 2))
    boxes.append([top_left[0], top_left[1], bottom_right[0], bottom_right[1]])
    class_ids.append(class_id)
```

解析后可以看到每张图的框数量不多，目标尺寸差异大，这对检测器的小目标能力提出要求。图像分辨率不一致，训练时统一缩放到 640×640。

### 2.2 数据清洗

清洗主要有两点：一是丢弃退化框——宽或高为 0 的标注直接跳过：

```python
if top_left[0] == bottom_right[0] or top_left[1] == bottom_right[1]:
    continue   # 退化框对训练无意义
```

二是 Faster R-CNN 的 target 需要额外字段 `area` 与 `iscrowd`：

```python
area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
iscrowd = torch.zeros((len(class_ids),), dtype=torch.int64)
labels = {"boxes": boxes, "labels": class_ids, "area": area,
          "iscrowd": iscrowd, "image_id": torch.tensor([idx])}
```

另外读图时统一用 `cv2.cvtColor(..., BGR2RGB) / 255.0`，避免通道顺序和数值范围错误。

### 2.3 模型训练与超参数调优

**方案一：YOLOv4。** 我们基于 `WongKinYiu/PyTorch_YOLOv4`，选 `yolov4-pacsp` 结构（它不用 mish 激活，Kaggle 环境更容易装）。改配置只需两步：把 yolo 层的类别数改成 3，把 yolo 层前一层的 filters 改成 `(3+5)*3=24`。

```python
class Option():
    def __init__(self, file_dir):
        self.cfg = os.path.join(file_dir, 'cfg/yolov4-pacsp.cfg')
        self.data = os.path.join(file_dir, 'data/coco.yaml')
        self.img_size = [640, 640]
        self.epochs = 2
        self.batch_size = 16
```

训练循环使用混合精度、EMA 与余弦学习率：

```python
lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
scaler = amp.GradScaler(enabled=cuda)
for epoch in range(start_epoch, epochs):
    for i, (imgs, targets, paths, _) in enumerate(dataloader):
        imgs = imgs.to(device).float() / 255.0
        with amp.autocast(enabled=cuda):
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(device), model)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
```

材料里为了演示只设了 2 个 epoch、batch 16，没有使用 mosaic 等增强，实际比赛应把 epochs 拉到几十并打开增强。

**方案二：Faster R-CNN。** 用 torchvision 的 `fasterrcnn_resnet50_fpn`，把分类头替换成 4 类（3 个目标类 + 背景）：

```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                            lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)
```

```python
for images, targets, image_ids in tqdm(train_dataloader):
    images  = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    optimizer.zero_grad(); losses.backward(); optimizer.step()
```

batch 取 8，学习率 0.005，每轮乘 0.2。材料里为了节省时间只训练了 1 个 epoch、800 次迭代，因此模型远未收敛。

### 2.4 模型测试与指标报告

评估采用 YOLO 仓库自带的 `ap_per_class`，在全图尺度上计算每个类别的 P、R、mAP：

```python
conf_thres = 0.02
iou_thres  = 0.6
iouv = torch.linspace(0.5, 0.95, 10).to(device)   # 10 档 IoU 阈值
output = non_max_suppression(inf_out, conf_thres=conf_thres,
                             iou_thres=iou_thres, merge=False)
```

输出表头为 `Class Images Targets P R mAP@.5 mAP@.5:.95`。由于只训练了极少的轮次，分数偏低是正常的；材料也对 Faster R-CNN 的评估做了说明：因为只训练了不到一个 epoch，评估直接在训练集上做，结果不代表真实泛化能力。正式提交时还需要把测试集预测框按比赛格式（图片名 + 类别 + 坐标）导出成 CSV。

---

## 三、总结

本文中我们完成了 Golf Detection 的两个检测方案。我们首先通过 EDA 确认标注是 YOLO 格式的归一化中心点加宽高，并解析为像素坐标。其次在清洗阶段丢弃退化的零面积框，补齐检测框架需要的 `area`、`iscrowd` 字段。接着分别训练 YOLOv4-pacsp 与 Faster R-CNN：前者用 640 输入、batch 16、余弦学习率与 AMP，后者用 ResNet50-FPN 主干、SGD 0.005、StepLR。最后用 mAP@0.5 与 mAP@0.5:0.95 评估并说明提交格式。

这个比赛让我们看到目标检测的两个关键点：一是标注格式转换必须万无一失，坐标除以宽高还是乘回宽高差一步就会全军覆没；二是小数据集上预训练主干 + 少量微调就能跑通流程，但要拿到好成绩必须增加训练轮次和数据增强，而不是只跑演示级的 1~2 个 epoch。
