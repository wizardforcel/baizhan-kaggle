# 超声神经分割（Ultrasound Nerve Segmentation）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的任务是：在一张颈部超声图像里，把**臂丛神经**（brachial plexus，脖子附近负责手臂感觉和运动的一束神经）从其他组织里标出来。医生在给病人做神经阻滞麻醉或者做穿刺时，需要看清神经的位置，但超声图像本身颗粒感很强、对比度又低，不同组织之间灰白程度很接近，人工判断既费时间又依赖经验。比赛希望参赛者用算法自动完成这件事。

比赛数据是一批颈部超声切片，每张图都有宽、高两个方向上的灰度信息。训练集里，每张原始超声图都配一张「掩码图」：掩码上和神经重合的像素被标成白色，其他地方是黑色。测试集只有原始超声图，没有掩码，需要参赛者自己预测。

难点主要有三个。第一，神经在图像里往往只占很小一块，大部分像素都是背景，模型很容易「偷懒」——把所有像素都预测成背景也能得到看起来不低的准确率，因此不能用普通准确率来评价。第二，超声图像有大量噪点和伪影，不同病人的体型、拍摄角度也不一样，模型必须学会忽略这些干扰。第三，训练图和测试图来自不同的切片序列，同一病人的相邻切片很像，如果简单随机划分验证集，分数会虚高。

### 1.2 比赛目标

输入是一张灰度超声图像，输出是一张同样大小的二值掩码：预测为神经的像素是 1，其他像素是 0。

训练数据里，图像文件名形如 `1_1.tif`，对应的掩码文件名是 `1_1_mask.tif`；下划线前面的数字是病人编号，后面的数字是这张图在该病人序列中的位置。测试集的文件名则是纯数字，例如 `1259.tif`。比赛要求提交一个 CSV 文件，包含两列：

- `img`：测试图片的编号；
- `pixels`：把预测掩码按行展开后，用「游程编码」（RLE）压缩成的一串数字。

游程编码的做法是：把二维掩码按列优先的顺序拉成一维，然后记录每一段连续 1 的起始位置和长度。这样提交文件会比直接存 580×420 个 0/1 小很多。

### 1.3 评价标准

比赛使用 **Dice 系数**（也叫 Dice similarity coefficient，有些地方直接叫 F1）来评价分割结果。它的定义是：

```
Dice = 2 × (预测为神经且真实也是神经的像素数)
       ÷ (预测为神经的像素总数 + 真实为神经的像素总数)
```

直观地说，就是看「预测出来的神经区域」和「真实的神经区域」重叠了多少。完全重合时 Dice 等于 1，完全不重合时等于 0。比赛把所有测试图片的 Dice 取平均，分数越高越好。

这个指标比普通准确率公平得多：如果模型把整张图都预测成背景，Dice 会直接变成 0，而不是 99%。所以我们后面训练模型时，也直接把 Dice 做成损失函数来优化。

---

## 二、解题过程

我们选择的主线是「U-Net + Dice 损失 + 数据增强」。U-Net 是一种专门做图像分割的卷积神经网络，名字里的 U 来自它的结构：左边一路不断下采样、把图像缩小并提取语义特征，右边一路不断上采样、把图像放大回原尺寸，同时把左边同尺度的特征「抄」到右边，帮助恢复边缘细节。下面按数据探索、数据清洗、模型训练、测试报告四步来说明。

### 2.1 数据探索（EDA）

我们首先把训练目录里的文件列出来，确认图像和掩码的命名关系，并统计数量：

```python
import os
import numpy as np
import pandas as pd
from PIL import Image

# 训练图片和掩码都放在 train 目录下
path = "../input/ultrasound-nerve-segmentation/train/"
file_list = os.listdir(path)

# 文件名有两种：1_1.tif 和 1_1_mask.tif
# 掩码文件名里带 "_mask"，用它来配对最方便
train_mask = [f for f in file_list if "_mask" in f]
train_image = [f.replace("_mask", "") for f in train_mask]

print("训练图片数量:", len(train_image))
print("前 5 个文件:", train_image[:5])
```

从输出可以看到，训练集一共有 **5635** 张图片，每张图片都有一张对应的掩码。文件名里的第一个数字是病人编号，第二个数字是切片编号，例如 `11_8.tif` 表示 11 号病人的第 8 张切片。这个信息很重要，因为后面划分验证集时不能把同一个病人的切片拆到两边，否则模型会「见过」验证集病人的其他切片，分数就不真实了。

接着我们随机挑一张图，把原图和掩码叠在一起看：

```python
import matplotlib.pyplot as plt

# 取 1 号病人的第一张切片
image = np.array(Image.open(path + "1_1.tif"))
mask = np.array(Image.open(path + "1_1_mask.tif"))

# 把掩码里等于 0 的地方遮掉，只显示白色的神经区域
mask_show = np.ma.masked_where(mask == 0, mask)

fig, ax = plt.subplots(1, 3, figsize=(16, 6))
ax[0].imshow(image, cmap="gray")      # 原始超声图
ax[1].imshow(mask_show, cmap="gray")  # 纯掩码
ax[2].imshow(image, cmap="gray")      # 叠加显示
ax[2].imshow(mask_show, cmap="jet", alpha=0.5)
```

看完我们就明白了两件事：第一，神经区域通常是一条细长的亮带，面积很小，所以背景像素远远多于前景像素；第二，原始图像尺寸是 580×420 左右，直接送进网络太占显存，需要先缩放到固定大小。训练集里还有一部分图片的掩码是全黑的，也就是这张切片里根本没有标注出神经，这类样本也需要保留，因为测试集同样可能出现没有神经的图。

### 2.2 数据清洗

这份数据的「清洗」主要不是去缺失值，而是把图像和掩码整理成网络能吃的格式。我们做了三件事：

1. **统一尺寸**。把所有图片缩放到 224×224，掩码用最近邻插值，保证缩放后仍然是 0/1，不会因为双线性插值产生灰色边缘。
2. **归一化**。把像素值从 0～255 除以 255，变成 0～1 之间的浮点数，让网络训练更稳定。
3. **二值化掩码**。原始掩码里神经是 255，背景是 0，我们统一映射成 1 和 0。

我们用 PyTorch 写了一个自定义 Dataset，把「读图、缩放、归一化、配对掩码」这几步封装起来：

```python
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from PIL import Image
import re

class NerveDataset(Dataset):
    def __init__(self, directory, is_test=False):
        self.files = []
        # 遍历目录，跳过掩码文件，只保留原图
        for file_name in directory.iterdir():
            if "mask" in str(file_name):
                continue
            self.files.append(self.combine_files(file_name))
        # 按文件名里的数字排序，保证每次读取顺序一致
        self.files = sorted(
            self.files,
            key=lambda f: int(re.search(r"\d+", str(f["image"])).group(0))
        )
        self.resize = Resize((224, 224), interpolation=Image.NEAREST)
        self.is_test = is_test

    def combine_files(self, file_name):
        # 原图 1_1.tif 对应掩码 1_1_mask.tif
        return {
            "image": file_name,
            "mask": ".." + str(file_name).split(".")[2] + "_mask.tif",
        }

    def __len__(self):
        return len(self.files)

    def open_as_array(self, idx):
        # 读图、缩放、加通道维，再除以 255 归一化
        raw_image = self.resize(Image.open(self.files[idx]["image"]))
        raw_image = np.stack([np.array(raw_image)], axis=2)
        return raw_image / np.iinfo(raw_image.dtype).max

    def open_mask(self, idx):
        # 掩码缩放后仍然是 255/0，这里统一变成 1/0
        raw_mask = self.resize(Image.open(self.files[idx]["mask"]))
        raw_mask = np.array(raw_mask)
        raw_mask = np.where(raw_mask == 255, 1, 0)
        return np.expand_dims(raw_mask, 0)

    def __getitem__(self, idx):
        x = torch.tensor(self.open_as_array(idx), dtype=torch.float32)
        # 为了 PyTorch 的通道优先格式，把 HWC 转成 CHW
        x = x.permute(2, 0, 1)
        if self.is_test:
            return x
        y = torch.tensor(self.open_mask(idx), dtype=torch.float32)
        return x, y
```

这里有两个容易踩坑的地方。第一，`Resize` 对掩码必须用最近邻插值，用默认的双线性插值会把 0 和 255 混成中间值，二值化时边界就糊了。第二，PyTorch 的张量默认是「通道×高×宽」，而 PIL 读出来是「高×宽×通道」，所以要用 `permute` 换一下顺序。

划分训练集和验证集时，我们按 7:3 切分：

```python
import torch
from pathlib import Path

images_path = Path("../input/ultrasound-nerve-segmentation/train")
data = NerveDataset(images_path)
print("全部样本:", len(data))

# 70% 训练，30% 验证
split_rate = 0.7
train_len = int(len(data) * split_rate)
valid_len = len(data) - train_len
train_ds, valid_ds = torch.utils.data.random_split(
    data, (train_len, valid_len)
)
print("训练集:", len(train_ds), "验证集:", len(valid_ds))
```

输出是全部 5635 张、训练 3944 张、验证 1691 张。严格来说，更好的做法是按病人编号分组切分，避免同一病人的相邻切片同时出现在训练和验证里；但在这次练习中我们先用了简单随机切分，主要目的是快速把流程跑通。

### 2.3 模型训练与超参数调优

模型部分我们用了一个现成的 U-Net 实现：`segmentation_models_pytorch` 库里的 `smp.Unet`，主干网络选 ResNet34，并加载 ImageNet 预训练权重。这样做的原因是，医学图像数据量不大，从零训练容易过拟合，而 ResNet34 在自然图像上学到的边缘、纹理特征可以迁移过来，收敛更快。

```python
import segmentation_models_pytorch as smp
import torch

# 输入是单通道灰度图，输出是单通道概率图
model = smp.Unet(
    encoder_name="resnet34",        # 主干网络用 ResNet34
    encoder_weights="imagenet",     # 加载 ImageNet 预训练权重
    in_channels=1,                  # 超声图只有灰度一个通道
    classes=1,                      # 只分割神经这一类
    activation="sigmoid"            # 输出 0~1 的概率
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

learning_rate = 0.001
epochs = 50
```

损失函数我们选 Dice Loss，而不是常用的交叉熵。原因是前面说过，神经像素占比极小，交叉熵会被大量背景像素主导，模型倾向于全部预测背景；Dice Loss 直接优化「预测区域和真实区域的重叠程度」，和比赛指标一致，对小目标更友好。

```python
import segmentation_models_pytorch as smp
import torch

# DiceLoss 内部就是 1 - Dice 系数
loss_function = smp.utils.losses.DiceLoss()

# Adam 对学习率不敏感，适合这种分割任务
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 余弦退火：学习率随着训练轮数平滑下降
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

# 监控指标用 IoU，它的趋势和 Dice 基本一致
metrics = [smp.utils.metrics.IoU()]
```

数据增强我们用了水平翻转和垂直翻转。超声图像左右翻转后解剖结构仍然合理，垂直翻转虽然现实中不常见，但可以帮助模型适应不同方向的纹理。翻转时必须让图像和掩码同步变换，否则标签就错位了：

```python
import cv2
import numpy as np

def flip_up_down(img):
    # 0 表示上下翻转
    return cv2.flip(img.copy(), 0)

def flip_right_left(img):
    # 1 表示左右翻转
    return cv2.flip(img.copy(), 1)

# 对每一对有神经的图和掩码，同时做两种翻转
for img_path, mask_path in zip(pos_img, pos_mask):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    uf_img = flip_up_down(img)
    uf_mask = flip_up_down(mask)
    rf_img = flip_right_left(img)
    rf_mask = flip_right_left(mask)

    # 新文件名加后缀，方便后面重新配对
    cv2.imwrite(img_path.replace(".jpg", "_uf.jpg"), uf_img)
    cv2.imwrite(mask_path.replace(".jpg", "_uf_mask.jpg"), uf_mask)
```

训练循环用 `segmentation_models_pytorch` 提供的 `TrainEpoch` 和 `ValidEpoch`，它们会把损失和指标自动累加、打印进度条。我们加了早停（EarlyStopping）：如果验证集损失连续 3 轮没有下降，就停止训练，避免白白浪费时间，也防止过拟合。

```python
from torch.utils.data import DataLoader
from lib.pytorchtools import EarlyStopping

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=False)

train_epoch = smp.utils.train.TrainEpoch(
    model, loss=loss_function, metrics=metrics,
    optimizer=optimizer, device=device, verbose=True
)
valid_epoch = smp.utils.train.ValidEpoch(
    model, loss=loss_function, metrics=metrics,
    device=device, verbose=True
)

stopper = EarlyStopping(patience=3)
train_loss, val_loss = [], []

for epoch in range(epochs):
    print("Epoch:", epoch)
    train_log = train_epoch.run(train_dl)   # 训练一轮
    val_log = valid_epoch.run(valid_dl)     # 验证一轮
    scheduler.step()                        # 更新学习率

    train_loss.append(train_log["dice_loss"])
    val_loss.append(val_log["dice_loss"])

    # 把验证损失交给早停器判断
    stopper(val_log["dice_loss"], model)
    if stopper.early_stop:
        print("验证损失不再下降，提前停止")
        break
```

在实际运行中，训练损失从第 0 轮的 0.56 左右一路降到 0.33 左右，验证集的 IoU 在第 8 轮前后达到约 0.45。之后验证损失开始反复，早停器在第 10 轮左右触发。这说明模型确实学到了东西，但也出现了明显的过拟合迹象——训练集和验证集的差距在拉大。我们判断主要原因是训练样本只有 3944 张，而且验证集和训练集来自同一批病人，分布比较接近，模型容易记住病人特有的纹理。

### 2.4 模型测试与指标报告

测试阶段要做三件事：对每张测试图预测概率图、把概率二值化成掩码、再按比赛要求编码成 RLE。这里有一个关键细节：训练时我们把图缩到了 224×224，但提交时必须还原成原始尺寸 580×420，否则 RLE 的位置会对不上。

```python
import cv2
import numpy as np
import torch

model.eval()  # 切换到推理模式，关闭 Dropout 等训练专用层
encodings = []

with torch.no_grad():  # 推理不需要计算梯度，省显存
    for i, image in enumerate(test_dl):
        image = image.to(device)
        pred = model(image)                      # 输出 0~1 的概率
        pred = pred[0].squeeze().cpu().numpy()   # 去掉 batch 和通道维
        pred = (pred > 0.5).astype(np.uint8)     # 阈值 0.5 二值化

        # 还原到比赛要求的原始尺寸
        pred = cv2.resize(pred, (580, 420),
                          interpolation=cv2.INTER_NEAREST)
        encodings.append((i, rle_encoding(pred)))
```

`rle_encoding` 是比赛的官方编码方式。它的规则是：把掩码按列优先拉平，找到所有值为 1 的位置，把连续的 1 合并成「起点 + 长度」的数对。起点从 1 开始计数，不是从 0：

```python
def rle_encoding(x):
    # x 是二维掩码，1 表示神经，0 表示背景
    # x.T 转置后 flatten，等价于按列优先展开
    dots = np.where(x.T.flatten() == 1)[0]

    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            # 新的一段连续区域开始，记录起点（从 1 开始）
            run_lengths.extend((b + 1, 0))
        # 当前这段长度加一
        run_lengths[-1] += 1
        prev = b
    return run_lengths
```

把每张图的编码拼成提交文件：

```python
import pandas as pd

submission = pd.DataFrame(columns=["img", "pixels"])
for i, encoding in encodings:
    # 用空格把数字连起来，和官方 sample_submission 格式一致
    submission.loc[i] = [str(i + 1), " ".join(map(str, encoding))]

submission.to_csv("submission.csv", index=False)
print(submission.head())
```

在本地验证集上，我们用同一套 Dice 公式计算分数，确认模型的输出和比赛指标是对齐的。验证集上的 Dice 大约在 0.6 上下（IoU 约 0.45，换算成 Dice 约为 `2×IoU/(1+IoU)`），说明模型已经能圈出神经的大致位置，但边界还不够精细。我们也试过把阈值从 0.5 调到 0.4 或 0.6，发现 0.5 附近比较稳；阈值太低会把周围组织误判成神经，太高则会漏掉细小的神经分支。

---

## 三、总结

本文中我们完成了超声神经分割比赛的完整流程。我们首先用 EDA 看清了数据：训练集有 5635 张 580×420 左右的灰度超声图，每张图都配一张二值掩码，神经区域细长且面积很小。其次我们做了针对性的数据清洗：统一缩放到 224×224、像素归一化、掩码二值化，并用自定义 Dataset 把读图和配对封装起来。接着我们用 ResNet34 作为主干的 U-Net、Dice Loss、Adam 优化器和余弦退火学习率训练模型，同时用翻转增强扩充样本、用早停防止过拟合。最后我们在验证集上得到约 0.45 的 IoU（约 0.6 的 Dice），并把测试图的预测结果还原到原始尺寸、编码成 RLE 提交。

这次实践让我们体会到，医学图像分割和普通图像分类很不一样：**评价指标要直接参与损失函数的设计**，否则模型会被大量背景像素带偏；**数据增强必须让图像和掩码同步变换**，否则标签会错位；**验证集的划分要按病人分组**，否则同一病人的相邻切片会造成分数虚高。这些经验比单纯调高几个百分点的分数更有价值。
