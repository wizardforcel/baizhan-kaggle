# Golf Detection

## 一、比赛简介

### 1.1 比赛背景

Golf Detection（BoolArt Golf Detection）是一个高尔夫场景的目标检测比赛。它与普通的图像分类不同：分类只需要回答“这张图里有没有某个东西”，而检测还要回答“它在哪里”，也就是要在图中画出一个紧贴目标的矩形框，并标出它属于哪一类。这类任务在体育转播、自动驾驶、工业质检里都很常见。

比赛数据直接决定了解法：图片放在 `train_images/train_images` 与 `test_images/test_images` 下，标注放在 `train_labels/train_labels` 下，是 YOLO 风格的纯文本标注。每张图对应一个 `label_xxx.txt`，每一行描述一个目标，字段为 `class_id center_x center_y width height`，其中中心点和宽高都除以了图片的宽高，落在 0~1 之间。这种“归一化中心点 + 宽高”的格式在 YOLO 系列里是标准做法，好处是同一份标注可以直接缩放到不同输入尺寸。

训练集共 3 个类别（不含背景）。材料的两个方案都围绕这个设定展开：一个是 YOLOv4，一个是 torchvision 的 Faster R-CNN。

### 1.2 比赛目标

输入是一张高尔夫场景图片，输出画面中每个目标的类别和边界框，通常表示为左上角 `(x1, y1)` 与右下角 `(x2, y2)`。测试集只提供图片，不提供标注，需要按比赛要求把检测结果写入提交文件。

模型的输入尺寸统一到 640×640。因为原图分辨率不一致，训练和推理时都要做等比缩放与填充（letterbox），否则同一目标在不同图里尺度差异太大，模型难以学习。

### 1.3 评价标准

比赛按检测的平均精度（mAP）评分。mAP 的计算依赖 IoU（Intersection over Union，交并比）：预测框与真实框的交集面积除以并集面积，完全重合为 1，完全不相交为 0。把某一类在所有图片上的预测按置信度排序，逐个判断是否为真正例，就能画出 Precision-Recall 曲线，曲线下的面积就是这一类的 AP；对所有类别取平均就是 mAP。

材料里的评估代码同时输出两个口径：`mAP@.5` 只要求 IoU 大于 0.5 就算命中，比较宽松；`mAP@.5:.95` 则在 0.5 到 0.95 之间取 10 个阈值分别算再平均，对框的贴合程度要求更高。排行榜分数越高越好。

---

## 二、解题过程

我们先读了标注和图片，确认数据格式；其次把标注解析成检测框架需要的张量；接着分别训练 YOLOv4 和 Faster R-CNN；最后用 mAP 评估并整理提交。

### 2.1 数据探索（EDA）

第一步是看一张图的标注长什么样。把归一化坐标乘回图像宽高，再转换成左上/右下角：

```python
h, w, _ = image.shape
boxes, class_ids = [], []
for line in open(label_file).readlines():
    class_id, cx, cy, bw, bh = line.strip().split()
    class_id = int(class_id) + 1                 # 0 留给背景
    cx, cy = int(w * float(cx)), int(h * float(cy))
    bw, bh = int(w * float(bw)), int(h * float(bh))
    # 中心点 + 宽高 → 左上角 / 右下角
    x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
    x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
    boxes.append([x1, y1, x2, y2])
    class_ids.append(class_id)
```

打印几张图的框数和框的大小，可以发现：每张图的标注数量不多，目标尺寸差异较大，小目标占比不低。这提示我们在缩放时不能把图缩得太小，否则小目标会只剩几个像素；同时也说明数据增强（尤其是缩放、裁剪）对提升泛化很重要，虽然材料里的演示为了省时间没有开启增强。

我们还检查了图片的通道和取值范围：OpenCV 默认按 BGR 读入，必须转成 RGB；像素值要除以 255 归一化到 0~1，否则预训练模型的输入分布会对不上。

### 2.2 数据清洗

清洗主要是两件事。

第一，丢掉“退化框”。标注里偶尔会出现宽或高为 0 的框，它们没有面积，既不能训练也不能评估：

```python
# 宽或高为 0 的框直接跳过
if x1 == x2 or y1 == y2:
    continue
```

第二，把标注整理成各框架要求的字典。Faster R-CNN 除了 `boxes` 和 `labels`，还需要 `area` 与 `iscrowd` 两个字段：

```python
area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
iscrowd = torch.zeros((len(class_ids),), dtype=torch.int64)

target = {
    "boxes": boxes,          # 形状 [N, 4]，xyxy 格式
    "labels": class_ids,     # 形状 [N]，1..3
    "area": area,            # 每个框的面积
    "iscrowd": iscrowd,      # 是否为人群框，这里全 0
    "image_id": torch.tensor([idx]),
}
```

另外，YOLO 的标签文件需要和图片一一对应。材料里的数据集类会读取图片名、推导标签路径，并缓存解析结果，避免每个 epoch 重新解析文本。

### 2.3 模型训练与超参数调优

**方案一：YOLOv4。** 我们使用 `WongKinYiu/PyTorch_YOLOv4` 仓库，选择 `yolov4-pacsp.cfg`。之所以选 pacsp 而不是原版，是因为原版用到 mish 激活，Kaggle 环境里装第三方包不方便，而 pacsp 结构不需要它。

改配置只有两步：把每个 yolo 层的 `classes` 改成 3，把 yolo 层前一层的 `filters` 改成 `(classes + 5) * 3 = 24`。前者告诉模型要预测几类，后者决定卷积输出通道数，少改一个都会在加载权重时报维度错误。

```python
class Option():
    def __init__(self, file_dir):
        self.cfg = os.path.join(file_dir, 'cfg/yolov4-pacsp.cfg')
        self.data = os.path.join(file_dir, 'data/coco.yaml')
        self.hyp = os.path.join(file_dir, 'data/hyp.scratch.yaml')
        self.img_size = [640, 640]
        self.epochs = 2
        self.batch_size = 16
```

训练主循环用混合精度（AMP）节省显存、用 EMA 平滑权重、用余弦退火调整学习率：

```python
lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
scaler = amp.GradScaler(enabled=cuda)

for epoch in range(start_epoch, epochs):
    for i, (imgs, targets, paths, _) in enumerate(dataloader):
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        with amp.autocast(enabled=cuda):
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets.to(device), model)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

材料里为了演示只设了 2 个 epoch、batch 16，也没有开 mosaic 等增强。实际比赛应把 epoch 拉到几十，并打开随机缩放、翻转、色彩抖动，否则模型很容易记住训练图。

**方案二：Faster R-CNN。** 我们用 torchvision 自带的 `fasterrcnn_resnet50_fpn`，它已经在 COCO 上预训练过，主干能提取通用视觉特征。我们只需要把分类头换成 4 类（3 个目标类 + 1 个背景）：

```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)
```

训练循环非常直接：把图片和 target 搬到 GPU，前向得到损失字典，把各项损失相加后反向传播：

```python
for images, targets, image_ids in tqdm(train_dataloader):
    images  = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()
```

batch 取 8，学习率 0.005，每轮乘 0.2。材料里为了节省时间只训练了 1 个 epoch、800 次迭代就中断，并明确说明“这是演示，结果不会好”。

### 2.4 模型测试与指标报告

评估时用 YOLO 仓库自带的 `ap_per_class`，它对每类计算 P、R、AP：

```python
conf_thres = 0.02                 # 置信度阈值，取得很低以保留更多候选框
iou_thres  = 0.6                  # NMS 的 IoU 阈值
iouv = torch.linspace(0.5, 0.95, 10).to(device)   # mAP@.5:.95 的 10 档阈值

# 推理并做非极大值抑制，去掉重叠的重复框
output = non_max_suppression(inf_out, conf_thres=conf_thres,
                             iou_thres=iou_thres, merge=False)
```

输出表头形如 `Class Images Targets P R mAP@.5 mAP@.5:.95`。P 是精确率（检出的框里有多少是真的），R 是召回率（真实目标里有多少被检出），mAP 综合两者。由于演示只训练了极少的轮次，分数偏低是正常的，不能直接拿来判断方案优劣。

提交时要把测试集的预测框按“图片名 + 类别 + 坐标”的格式导出成 CSV。每张图可能检出多个框，需要按比赛样例的列顺序整理，并注意坐标是像素值还是归一化值——这一点每个比赛要求不同，提交前必须对照样例文件核对。

---

## 三、总结

本文中我们完成了 Golf Detection 的两个检测方案。我们首先通过 EDA 确认标注是 YOLO 格式的归一化中心点加宽高，并把它解析成像素坐标的左上/右下角；同时注意到目标尺寸差异大、小目标多。其次在清洗阶段丢弃零面积的退化框，并为 Faster R-CNN 补齐 `area`、`iscrowd` 字段。接着分别训练 YOLOv4-pacsp 与 Faster R-CNN：前者用 640 输入、batch 16、余弦学习率与 AMP，后者用 ResNet50-FPN 主干、SGD 0.005、StepLR。最后用 mAP@0.5 与 mAP@0.5:0.95 评估，并说明提交格式。

这个比赛让我们看到目标检测的几个关键点。第一，**标注格式转换必须万无一失**：归一化坐标乘错一次，整个训练就会在错误的框上收敛。第二，**预训练主干 + 少量微调**能让小数据集快速跑通，但演示级的 1~2 个 epoch 只能验证流程，不能代表真实水平；要拿好成绩，必须增加训练轮次、打开数据增强，并针对小目标调整输入分辨率。第三，**评估要同时看 P、R 和 mAP**：只看其中一个数字，很容易掩盖“漏检很多”或“误检很多”的问题。
