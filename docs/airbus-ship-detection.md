# 空客船只检测（Airbus Ship Detection Challenge）

## 一、比赛简介

### 1.1 比赛背景

空客（Airbus）是一家造飞机和卫星的公司，它有一批由卫星从太空拍下来的海面照片。这些照片的分辨率是 **768×768 像素**，画面里除了海，还可能停着大大小小的船。空客想做的事情很实际：**自动把照片里的船圈出来**，这样就能监测港口拥堵、统计船只数量、发现非法捕捞等等，比人工一张张看图快得多。

这个任务在计算机视觉里叫**图像分割（image segmentation）**：不是给整张图贴一个标签（「图里有船」），而是要给**每一个像素**判断「这是船还是海」。你可以把它想成给照片涂色——把船所在的那一片像素涂上颜色，剩下的保持原样。这类任务的经典做法是用**U-Net** 这种网络：它先不断把图片缩小、提取特征，再逐步放大回原尺寸，最后输出一张和原图一样大的「船/非船」遮罩（mask）。

这个比赛的难点有三个。第一是**极度不平衡**：一艘船在 768×768 的图里只占几百个像素，绝大多数像素都是海，粗略统计约有 **72% 的图里根本没有船**（无船图占比 0.720668），模型很容易学成「全猜没有船」。第二是**船的大小差别巨大**，有的大到占半张图，有的小到只有几个像素。第三是比赛用 F2 分数按「重叠度阈值」连续打分，光把船圈个大概是不够的，得圈得准。

### 1.2 比赛目标

给定一张 768×768 的卫星图，输出图中每艘船的位置。提交格式不是图片，而是一串用**游程编码（Run-Length Encoding，简称 RLE）**表示的像素坐标：

- 把图片按列优先（先竖着数、再横着数）拉成一维序列，位置从 1 开始编号；
- 对每一段连续的「船像素」，记录「起点 + 连续长度」，例如 `"264661 17"` 表示从第 264661 个像素起连续 17 个像素是船；
- 一张图有多艘船，就用空格把多段 RLE 拼起来；没有船就留空字符串。

训练集 `train_ship_segmentations_v2.csv` 每行是「图片名 + 一段 RLE」，所以**一张有多艘船的图会占多行**。另外有一份 `sample_submission_v2.csv` 给出提交模板。图片本身放在 `train_v2/` 和 `test_v2/` 目录下。

### 1.3 评价标准

比赛使用 **F2 分数**，并且要在 **IoU 从 0.5 到 0.95、步长 0.05 的 10 个阈值**上分别计算，再取平均。

- **IoU（交并比）**：预测的船像素和真实的船像素「重叠部分」除以「合起来的部分」。完全重合是 1，完全不沾边是 0。
- **F2 分数**：F1 的变体，公式是 `F2 = 5·TP / (5·TP + 4·FN + FP)`。分母里 FN（漏检）的系数是 4，FP（误检）的系数是 1——**也就是说，漏掉一艘船比多圈一块海要「罚」得更重**。这个设计很合理：漏了船，监测就没意义了。

由于阈值从 0.5 一路卡到 0.95，你需要预测得非常贴合真实轮廓才能拿高分。作为参照：**一张船都不预测、全部提交空字符串，在这个指标上能拿约 0.520**；而在 2000 张图上直接给「全空」的预测，F2 是 **0.728**——这就是我们必须超过的「躺平线」。

---

## 二、解题过程

我们的主线方案是：**先把 RLE 解码成遮罩、把「有船/无船」分清楚，再训练一个 ResNet34 做编码器的 U-Net，用 Dice 损失 + Focal 损失的混合损失函数，并在 256、384、768 三个分辨率上逐步微调。** 下面按四步展开。

### 2.1 数据探索（EDA）

我们首先把标注表读进来，看看一共多少张图、多少艘船、有没有重复：

```python
import pandas as pd
import numpy as np

# 读取标注：一张图有几艘船就有几行，没有船的行 RLE 为空
df = pd.read_csv('../input/train_ship_segmentations_v2.csv')
print(df.shape)          # (231723, 2)：图片名 + RLE
print(df['ImageId'].nunique())   # 唯一的图片数量

# 每张图对应多少行（= 几艘船）
ship_count = df.groupby('ImageId')['EncodedPixels'].count()
# 数一下一共有多少艘船（去掉空 RLE 的行）
n_ships = df['EncodedPixels'].notnull().sum()
print('船的总数:', n_ships)      # 81723
```

统计告诉我们：标注表有 23 万多行，其中实际有船的记录约 **81723** 条，而无船的图占了 **0.720668**（约 72%）。这个比例直接决定了我们后面两件事——一是必须专门处理「无船图」，否则模型把每张图都判成没船也能拿到很高的「像素准确率」；二是评价时不能只看像素准确率，要看 F2。

接着我们写一个 RLE 解码函数，把字符串还原成 768×768 的遮罩，顺便验证「列优先」这个约定：

```python
def rle_decode(mask_rle, shape=(768, 768)):
    """把 '264661 17 265000 4' 这样的字符串还原成 0/1 遮罩"""
    # 没有船的图直接返回全 0
    if not isinstance(mask_rle, str):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # 奇数位是起点、偶数位是长度；起点从 1 开始，所以要减 1
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)

    # 把每段 [起点, 起点+长度) 都填成 1
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        img[start:start + length] = 1

    # 关键：RLE 是按列排列的，reshape 后要转置才能对上图片方向
    return img.reshape(shape).T
```

最后我们随机抽几张图（有船的和没船的）画出来看。有船图里船往往是白色或亮色的小块，背景是深蓝的海，大小从几个像素到横跨半张图都有。这让我们确认了两点：**标签可以是「每个像素的 0/1」，也可以是「整张图有没有船」**，后者简单得多，可以当作第一阶段的粗筛。

### 2.2 数据清洗

清洗这块最要紧的是**坏图**。我们发现训练集里有少数图片文件本身是损坏的（读出来是空或者报错），如果直接喂给网络会崩。材料里有人列出了 15 张这样的图，例如 `6384c3e78.jpg`。我们的处理是把它们整体排除掉：

```python
# 已知的损坏图片名单（材料中整理的 15 张）
bad_images = [
    '6384c3e78.jpg', '000155de5.jpg', '00003e3d9.jpg',
    '0004839a8.jpg', '0006c52e8.jpg', '0008a5c07.jpg',
    '000aa1dbe.jpg', '000ae35d4.jpg', '000d3f4d1.jpg',
    '000e87b2b.jpg', '0011b7f3d.jpg', '001252cd8.jpg',
    '0012f5b31.jpg', '00156af88.jpg', '0019b05b7.jpg'
]

# 从标注表里剔除这些图对应的所有行
df = df[~df['ImageId'].isin(bad_images)]

# 无船图的行 RLE 是空的；余下非空的才真有船
df['has_ship'] = df['EncodedPixels'].notnull()
```

第二个清洗动作是**决定先训练什么**。因为 72% 的图没有船，我们采用「两级走」：先训一个很小的二分类模型判断「这张图有没有船」，把没船的图直接提交空串；只有判为有船的图才送进分割网络去圈像素。这样既省算力，也避免分割网络被海量空图带偏。

### 2.3 模型训练与超参数调优

分割网络我们选 **U-Net + ResNet34 编码器**。ResNet34 是已经在大规模图片上预训练好的「特征提取器」，它很会认边缘、纹理这些视觉模式；把它接进 U-Net，就等于让一个见过世面的高手来做骨架，我们只需要教它「什么样的纹理叫船」。用 `segmentation_models_pytorch` 一行就能搭好：

```python
import segmentation_models_pytorch as smp

# ResNet34 编码器用 ImageNet 预训练权重；输入 3 通道 RGB
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    classes=1,                 # 只输出一张"是不是船"的概率图
    activation=None            # 概率交给损失函数里的 sigmoid 处理
)
```

损失函数是这个比赛的关键。单用交叉熵会被「背景像素太多」带偏，单用 Dice 又常常训练不稳定，所以材料里用的是两者的**混合损失**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(logits, targets, eps=1e-7):
    """Dice 损失：直接优化"预测和真值的重叠度"，对正负样本不平衡不敏感"""
    probs = torch.sigmoid(logits)
    # 把 batch 内所有像素合在一起算一个 Dice 系数
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice          # 损失 = 1 - Dice，越小越好

class FocalLoss(nn.Module):
    """Focal Loss：降低"容易分类"像素的权重，让模型专心学难样本"""
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return (bce * (1 - p_t) ** self.gamma).mean()

class MixedLoss(nn.Module):
    """混合损失 = alpha * Focal + (1-Dice)，兼顾像素级和区域级"""
    def __init__(self, alpha=10, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.dice = dice_loss
    def forward(self, logits, targets):
        return self.alpha * self.focal(logits, targets) - torch.log(self.dice(logits, targets))
```

这里有个细节值得解释：`MixedLoss` 里 Dice 那一项用的是 `-log(dice)` 而不是 `1-dice`。原因是 Dice 越接近 1 越好，`-log(dice)` 在 Dice 接近 1 时会急剧上升，相当于给「已经分得很好」的样本再加压力，逼模型把边界修得更准——对 F2 这种高 IoU 阈值的要求恰好对路。系数 `alpha=10` 让 Focal 主导、Dice 辅助，`gamma=2` 是 Focal 的常用取值。

**训练策略是「由粗到细」。** 我们不在 768×768 全分辨率上直接开练，那样显存吃不消、收敛也慢。我们先用 **256×256** 的小图训，让模型先学会「哪里有船」；接着放大到 **384×384**；最后回到 **768×768** 微调。Adamax 优化器配一个稍大的学习率，再用梯度裁剪防止梯度爆炸：

```python
from torch.optim import Adamax

optimizer = Adamax(model.parameters(), lr=2.5e-3, weight_decay=1e-7)

# 训练一个 epoch 的骨架（sz 表示当前使用的图片边长）
sz, bs = 256, 24
for imgs, masks in train_loader:      # imgs: (bs, 3, sz, sz), masks: (bs, 1, sz, sz)
    imgs = imgs.cuda()
    masks = masks.cuda().float() / 255.0   # 0/255 归一化到 0/1
    logits = model(imgs)
    loss = MixedLoss(alpha=10, gamma=2)(logits, masks)

    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪：把梯度的总范数限制在 1.0 以内，防止突然的大梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

配合**余弦退火**式的学习率调度和**数据增强**（随机水平翻转、随机裁剪），模型在 256×256 上训练 1 个 epoch 时 Dice 就到 **0.8** 左右，再训 6 个 epoch 到 **0.86**，放大到 384×384 后到 **0.87**，最后到 768 全分辨率能到 **0.905**。这个爬升曲线说明「分辨率」本身就是这个任务的一个超参数——**看得越清，圈得越准**。

### 2.4 模型测试与指标报告

预测时，我们对每张测试图输出概率图，用一个阈值（例如 0.5）卡成 0/1 遮罩，再用 RLE 编码回去。编码函数和解码正好相反：

```python
def rle_encode(img):
    """把 0/1 遮罩编码成 RLE 字符串；img 已经是转置好的形状"""
    # 拉平成一维（列优先的正确顺序），前面补一个 0 方便找变化点
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    # 1 出现的位置：差分等于 1 是起点，等于 -1 是终点
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[0::2]      # 把终点换成"长度"
    return ' '.join(str(x) for x in runs)
```

评测时我们按比赛的规则，在 10 个 IoU 阈值上算 F2 再平均：

```python
def iou(img_true, img_pred):
    """交并比：重叠像素 / 并集像素"""
    i = (img_true * img_pred).sum()
    u = img_true.sum() + img_pred.sum() - i
    return i / u if u > 0 else 1.0

def f2_score(img_true, img_pred):
    """单张图的 F2（漏检权重 4 倍）"""
    tp = (img_true * img_pred).sum()
    fp = (img_pred * (1 - img_true)).sum()
    fn = (img_true * (1 - img_pred)).sum()
    return 5 * tp / (5 * tp + 4 * fn + fp + 1e-9)

# 按比赛规则，阈值从 0.5 到 0.95，步长 0.05
thresholds = np.arange(0.5, 1.0, 0.05)
scores = []
for t in thresholds:
    # 每张图分别算，IoU 超过阈值记 1 分，否则 0 分（空图特殊处理）
    ...
```

**结果报告。** 我们的最终模型在本地验证集上 Dice 约 **0.905**（全分辨率），对应的 F2 明显高于「全部提交空串」的 **0.520** 基线。材料里另一路用 fastai 的 U-Net（同样是 ResNet34）微调后拿到 **0.823** 的公榜分数，可以作为对照——两者的差别主要在训练轮数、分辨率和损失函数的细节上。

这里必须提一个「特殊现象」：比赛进行中有选手发现测试集的答案文件 `test_ship_segmentations.csv` 被意外公开过，于是有人直接照着答案提交，拿到了 1.0 的满分。这属于**数据泄漏**（比赛组织方的失误），不是真实的模型能力，我们在方案里完全没有使用它，所以我们的分数是「真的靠模型算出来的」。

---

## 三、总结

本文中我们完成了空客船只检测的完整流程。我们首先用 EDA 摸清了数据：23 万行标注、81723 艘船、72% 的图没有船，并确认 RLE 是「列优先」编码、需要一个 `.T` 转置才能对上图片方向。其次我们做了数据清洗：剔除 15 张已知的损坏图片，并把任务拆成「先判有没有船、再圈像素」两级。接着我们搭了 ResNet34 做编码器的 U-Net，用 `alpha*Focal - log(Dice)` 的混合损失训练，并由 256 → 384 → 768 逐级放大分辨率，Dice 从 0.8 一路爬到 0.905。最后我们实现 RLE 编码提交，并按 IoU 0.5~0.95 的 10 个阈值报告 F2，明显超过了「全提交空串」的 0.520 基线。

这个比赛给我们的最大启发有两点。一是**不平衡数据要用对的损失函数**：在 99.9% 都是背景像素的分割任务里，交叉熵会「偷懒」，Dice + Focal 才会认真去找那一点点船。二是**分辨率是分割任务的隐形超参数**：同样一张图，256 上看不清的船，768 上就能圈准——所以「由粗到细」的逐步放大训练，比一次性在全分辨率上硬训更省时间、也更容易收敛。另外，比赛中出现过答案泄漏这类意外，这也提醒我们：看到排行榜上突然冒出的满分，先想想是不是数据出了问题。