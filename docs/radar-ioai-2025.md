# 雷达感知（Radar IOAI 2025 Task 1）

## 一、比赛简介

### 1.1 比赛背景

本场比赛是 IOAI 2025（国际人工智能奥林匹克，International Olympiad in Artificial Intelligence）的第一题（Task 1），题目代号 radar-ioai-2025。比赛想解决的核心问题是：**如何用一台毫米波雷达"看见"周围环境里的人与物体**。

和摄像头不同，毫米波雷达自己发射电磁波、再接收反射回来的回波，不依赖环境光线，所以在夜间、大雾、烟尘等摄像头"瞎掉"的场景里依然工作正常，是无人驾驶和机器人感知的重要传感器。本场比赛使用的是一台 **77 GHz 调频连续波（FMCW）雷达**。所谓调频连续波，简单说就是雷达持续发出一段段频率随时间线性上升的波（每一段叫一个 chirp），通过比较发射和回波的频率差来测算目标距离，再靠多根天线之间的相位差来测算目标的角度。官方参考方案给出的雷达配置印证了这一点：扫频周期 Tr = 60 微秒，采样率 Fs = 10 MHz，每帧包含 128 个 chirp，发射天线 3 根、接收天线 4 根（合成为 12 根虚拟天线），探测距离大约 1 米到 25 米，方位角和俯仰角的范围都是 -90 度到 90 度。

雷达每发出一个帧的信号，回波经过距离傅里叶变换、角度估计（MUSIC 等算法）一系列处理后，会变成若干张**热力图（heatmap）**。热力图上的每个像素代表"这个距离、这个角度位置上有多少能量的回波"——亮的地方说明那里有物体。比赛把热力图当成"画面"，要求我们对画面上的**每一个像素**判断它属于哪一类物体，这本质上是一个经典计算机视觉问题：**语义分割（semantic segmentation）**，和无人驾驶里的车道/行人分割是同一类任务。

这个比赛的难点有两点。第一，雷达热力图和日常看到的自然图像完全不同，6 个通道的物理意义各不相同、数值范围也不固定，初学者常常不知道从哪下手。第二，画面里的前景目标（人、椅子、行李箱、墙）占的像素极少，绝大多数像素都是"背景"，类别极度不均衡，模型稍不注意就会学成"把一切都预测成背景"，而这在比赛指标下得分会非常低。

### 1.2 比赛目标

**输入。** 每个样本是一个 `.mat.pt` 文件（PyTorch 张量）。训练集放在 `training_set/training_set/` 目录下，测试集放在 `test_set/test_set/` 目录下，文件名都以 `.mat.pt` 结尾。一个文件里保存了一个 `(6, 50, 181)` 的六通道热力图张量，训练样本还额外带着一张 `(50, 181)` 的逐像素标签图。这里的 6 对应 6 个通道，50 是距离方向（对应大约 1～25 米，被裁成了 50 个距离单元），181 是方位/俯仰角方向（对应 -90 度到 90 度，每度一格），所以一张图一共 50×181 = 9050 个像素。

这 6 个通道的具体含义，官方参考代码里说得很清楚：

| 通道 | 含义 |
| --- | --- |
| 0 | 距离-方位（Range-Azimuth）静态热力图 |
| 1 | 距离-方位动态热力图 |
| 2 | 距离-俯仰（Range-Elevation）静态热力图 |
| 3 | 距离-俯仰动态热力图 |
| 4 | 距离-方位 MUSIC 平均动态概率图（取类别 1 的概率） |
| 5 | 距离-方位 MUSIC 平均静态概率图（取类别 2 的概率） |

静态通道刻画静止物体的回波，动态通道刻画运动物体，MUSIC 则是一种经典的角度估计算法——它的输出"概率图"大致表达"这个角-这个距离上有很大可能是某种目标"。前人说这张融合了静态/动态、距离/角度的热力图，就像雷达版本的"照片"。

**输出。** 对测试集里的每个样本，预测它 9050 个像素各自的类别。类别编号和含义如下：

| 标签值 | 含义 |
| --- | --- |
| -1 | 背景背景 background |
| 0 | 行李箱 suitcase |
| 1 | 椅子 chair |
| 2 | 人 person |
| 3 | 墙 wall |

注意类别编号里有 -1 这种"负数"。Python 里 Pytorch 的交叉熵损失要求类别必须从 0 开始，所以后面训练时人人都做同一件事：把标签整体 +1，把 -1～3 平移成 0～4 再训练，推理时再减回来。

### 1.3 评价标准

比赛的指标是一个**前景加权准确率**。官方参考代码里直接给出了打分函数：

```python
def competition_metric(outputs, labels):
    # 先取概率最大的那个类别作为每个像素的预测
    preds = outputs.argmax(dim=1)
    correct = (preds == labels)

    # 背景像素单独拎出来, 权重记为 1
    bg_mask = (labels == 0)
    # 前景(四类目标)像素权重记为 50
    fg_mask = (labels != 0)

    # 加权准确率 = 猜对的加权像素数 / 总加权像素数
    score = (correct[bg_mask].sum() + 50 * correct[fg_mask].sum()) / \
            (bg_mask.sum() + 50 * fg_mask.sum() + 1e-8)
    return score.item()
```

公式读起来很直白：把每个"猜对"的像素按权重累加，再除以所有像素的权重之和。**背景像素一个算 1 分，前景像素一个算 50 分**。也就是说，把一个人分割对，比把 50 个墙皮分割对还值钱。分数越高越好，满分是 1.0。

这个指标说了一件事：比赛特别看重"前景分割得准不准"。如果模型只学会"全都预测成背景"，背景那一大半像素全猜对了，但前景像素全错，按这个加权公式算出来的分数会非常难看。也正因为评价指标偏爱前景，后面几乎所有方案训练时都会给损失函数里的前景类别加权——**优化目标和评分规则保持一致**，这是我们从比赛里学到的最重要的一条经验。

---

## 二、解题过程

整场比赛的思路可以浓缩成一句话：**把 6 通道的雷达热力图当"6 通道图像"，喂给经典的编码器-解码器分割网络（U-Net 一族），输出每个像素属于 5 类目标中的哪一类；再用加权的损失函数和"选最优轮次"的小技巧，把分数推到 0.98 以上。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来展开。

### 2.1 数据探索（EDA）

第一步是先看看数据到底长什么样。我们用 glob 扫出所有训练文件，读一帧打印形状：

```python
import glob
import os
import torch

# 训练文件都放在 training_set/training_set 下, 全部以 .mat.pt 结尾
base_path = '/kaggle/input/radar-ioai-2025/training_set/training_set'
file_paths = sorted(glob.glob(os.path.join(base_path, '*.mat.pt')))
print('训练样本总数:', len(file_paths))

# 读第一帧, 前 6 个通道是热力图, 第 7 个通道是标签图
data = torch.load(file_paths[0], weights_only=True)
images = data[:6]     # [6, 50, 181]
labels = data[6]      # [50, 181]
print('热力图形状 [通道, 距离, 方位]:', images.shape)
print('标签图形状:', labels.shape)
print('标签取值范围:', int(labels.min()), '~', int(labels.max()))
```

我们看到了预期的结构：每帧是 6×50×181 的热力图加上一张 50×181 的标签图，而且标签取值范围确实是 -1～3，和比赛说明一致。

接着我们"画"出来看看——雷达画面的直观感受很重要，能帮我们判断该用什么模型。参考高分代码的做法：

```python
import matplotlib.pyplot as plt

# 挑一帧, 把前 3 个通道的热力图和最后的标签图并排画出来
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for channel in range(3):
    ax[channel].imshow(images[channel], cmap='grey')
    ax[channel].set_title(f'channel {channel}')
# 标签图 +1 是为了让背景(-1)也从 0 开始, 颜色区分更明显
ax[3].imshow(labels.squeeze() + 1, cmap='viridis')
ax[3].set_title('label map')
plt.show()
```

肉眼可见热力图和自然图像完全不同，它是"散点能量图"：亮斑集中在一片区域内，对应雷达真正照到的物体；标签图里一块块不同颜色的区域就是各个目标。这种"小目标、大背景"的画面让 EDA 的下一步目标非常明确——**统计类别分布**，验证我们对"前景稀有"的猜测：

```python
from collections import Counter

# 数一数一帧里每个类别占多少像素, 先看前 100 帧做快速统计
counts = Counter()
for f in file_paths[:100]:
    data = torch.load(f, weights_only=True)
    counts.update(data[6].numpy().reshape(-1).tolist())

total = sum(counts.values())
print('各类别像素数:', dict(counts))
print('各类别占比:', {c: round(n / total, 4) for c, n in counts.items()})
```

统计证实了我们的判断：背景（-1 类）占了绝大多数像素，四个前景类别加起来也非常少。这种极端的类别不均衡就是全场比赛要打的最大的"boss"，后面所有损失函数的选择都在对付它。

### 2.2 数据清洗

这份数据的"数据清洗"其实非常克制。我们写的 `RadarDataset` 只做了三件事：格式转换、标签平移、测试集不返回标签：

```python
from torch.utils.data import Dataset

class RadarDataset(Dataset):
    def __init__(self, file_paths, train=True):
        self.file_paths = file_paths
        self.train = train
        self.file_names = [os.path.basename(p) for p in file_paths]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], weights_only=True)

        # 取前 6 个通道的热力图, 统一转成 float 方便进网络
        images = data[:6].float()

        if self.train:
            # 训练样本带标签, 值域是 -1~3, 但交叉熵要求从 0 开始,
            # 所以整体 +1 平移成 0~4
            labels = data[6].long() + 1
            return images, labels, self.file_names[idx]
        else:
            # 测试集没有标签图, 只有文件名, 文件名后面做提交要用
            return images, self.file_names[idx]
```

为什么清洗这么少？有两个原因。第一，这些 `.mat.pt` 文件已经是官方处理好的张量，没有缺失值、没有脏文本，根本不用做常规表格那种清洗；第二，不同帧之间热力图的数值范围本来就不一致，逐样本的归一化意义不大，所以参赛者基本都只做了 `float()` 转换。我们也在构建 dataset 时把测试集和训练集分开处理，避免把任何标签信息漏给测试集。

接下来按 8:2 切出验证集，固定随机种子保证复现性：

```python
from sklearn.model_selection import train_test_split

# 全部 .mat.pt 文件, 用 20% 做本地验证集
train_paths, val_paths = train_test_split(
    file_paths, test_size=0.2, random_state=42)
```

另外还有个容易踩的坑：**181 是奇数**。U-Net 里的下采样（max pooling）每走一层分辨率除以 2，走几层就会把 181 除出小数来。我们参考别人的经验做了一个小处理：把热力图沿最后一维补 5 个像素到 186（`F.pad(x, (0, 5, 0, 0))`）再下采样，最后反正推理时还会用双线性插值把输出拉回 50×181，所以不会影响提交。

### 2.3 模型训练与超参数调优

模型方面我们采用官方参考方案（DIMSSNet）的思路：**6 个通道物理意义各不相同，就先给每个通道各配一套"编码器-解码器"，各自提取自己的特征，最后把 6 组特征融合起来逐像素分类**。这个思路比直接把 6 通道拼在一起塞进一个网络更合理——距离-方位图、距离-俯仰图、概率图的"语言"不一样，让它们各说各话再汇总，信息损失更小。

网络的基本积木是经典的"双卷积块"：

```python
import torch.nn as nn

class DoubleConv(nn.Module):
    """连续两次 3x3 卷积 + 批归一化 + ReLU, 分割网络里最常用的积木"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),          # 批归一化让训练更稳
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
```

融合 6 组特征的分类头 `Fuse_fea` 长这样：

```python
class Fuse_fea(nn.Module):
    """把 6 个通道各自提取出的 32 维特征拼起来, 逐像素分成 5 类"""
    def __init__(self):
        super().__init__()
        self.convt1 = nn.Conv2d(192, 96, 3, padding=1)   # 192 = 32 * 6 个通道
        self.convt2 = nn.Conv2d(96, 48, 3, padding=1)
        self.convt3 = nn.Conv2d(48, 24, 3, padding=1)
        self.convt4 = nn.Conv2d(24, 5, 3, padding=1)     # 最后一层输出 5 类
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feas_sra, feas_dra, feas_sre, feas_dre, feas_srd, feas_drd):
        # 把 6 组特征沿通道维拼起来: (B, 32, 50, 181) * 6 -> (B, 192, 50, 181)
        feas_ra = torch.cat(
            [feas_sra, feas_dra, feas_sre, feas_dre, feas_srd, feas_drd], dim=1)
        x = self.relu(self.convt1(feas_ra))
        x = self.relu(self.convt2(x))
        x = self.relu(self.convt3(x))
        x = self.sigmoid(self.convt4(x))   # sigmoid 把输出压到 0~1 的概率
        return x
```

> 细节说明：俯仰/动态相关分支的特征在融合前会先沿方位维求和、再广播回 181 列，相当于把"侧视图"压扁投影回"主视图"再叠加，这样 6 组特征的尺寸能对齐。完整模型用 6 对独立的编码器-解码器分别处理 6 个通道，这里我们只展示融合部分。

然后是整个比赛的关键——**损失函数**。面对"背景占绝大多数"的数据，直接上交叉熵会让模型迅速学会"全猜背景"。官方参考方案用的是 **focal loss**，它会给"被模型分错、但模型其实已经有点把握"的像素更高的权重：

```python
def _neg_loss(pred, gt):
    # 把概率限制在 (1e-5, 1-1e-5), 防止 log 出现 0 或负数
    pred = torch.clamp(pred, 1e-5, 1 - 1e-5)

    pos_inds = gt.eq(1).float()    # 前景像素
    neg_inds = gt.eq(0).float()    # 背景像素

    # (1-p)^2 这个系数就是 focal 的核心: 预测越接近对的样本权重越小,
    # 模型越没把握(难分)的样本权重越大, 让它别只顾着学简单的背景
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

    num_pos = pos_inds.float().sum()
    if num_pos == 0:               # 这一批一个前景都没有
        return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos
```

训练时我们还做了几个配套处理：标签要转成 one-hot 才能和这个损失对齐；每轮在训练集上学、在验证集上算 loss。官方配置是 batch size 4、Adam 学习率 0.001、训练 20 轮：

```python
import torch.optim as optim

criterion = FocalLoss()                      # 上面的 focal loss
optimizer = optim.Adam(dimssnet.parameters(), lr=0.001)

for epoch in range(20):
    dimssnet.train()
    running_loss = 0.0
    for images, labels, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 6 个通道分别喂给 6 套编码器-解码器
        outputs = dimssnet(images[:, 0], images[:, 1], images[:, 2],
                           images[:, 3], images[:, 4], images[:, 5])
        # 把输出尺寸对齐回 50x181 再算损失
        outputs = F.interpolate(outputs, size=(50, 181),
                                mode='bilinear', align_corners=False)
        labels = to_one_hot(labels.long(), 5)   # 标签转 one-hot
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Training Loss: {running_loss / len(train_loader):.4f}")
```

除了 focal loss，我们也参考了别的队伍的调参思路：有的队伍用**加权交叉熵**，把每个类别的损失权重定成 `[1.0, 50, 50, 50, 50]`——背景权重是 1，四类前景权重都是 50，和比赛指标的"前景权重 50"完全对上；有的队伍把优化器换成 AdamW、配上余弦退火学习率调度。数据增强方面，有人对热力图做**随机水平翻转**（雷达画面左右方向天然对称），有人把每个样本随机旋转 0 度或 90 度来扩充数据量。我们都拿来试了，确实都能提分。

这里要特别提一个不炫技但很实用的小技巧——**best epoch**。在训练中我们每一轮都在验证集上算一次比赛指标，并把"验证指标最高那一轮"的权重单独保存下来。最后推理用的是这份"最优轮次"的权重，而不是最后一轮的权重。深度学习训练后期 loss 会抖动，最后一轮的模型往往已经不是泛化最好的那一个，这一招通常能白捡零点几个百分点的分数。

### 2.4 模型测试与指标报告

先看本地验证。我们用和比赛完全一致的 `competition_metric` 在验证集上算分，用它来挑选 best epoch：

```python
def competition_metric(outputs, labels):
    """比赛评分规则: 背景像素权重 1, 前景像素权重 50"""
    preds = outputs.argmax(dim=1)
    correct = (preds == labels)
    bg_mask = (labels == 0)
    fg_mask = (labels != 0)
    score = (correct[bg_mask].sum() + 50 * correct[fg_mask].sum()) / \
            (bg_mask.sum() + 50 * fg_mask.sum() + 1e-8)
    return score.item()
```

训练完我们会把验证集里每个 batch 的预测送进这个函数累加，选出得分最高的一轮作为最终模型。从各个提交页面的记录看，这条路线上每个人的成绩是逐步爬升的：一个只有 3 层卷积的小网络就能先拿到 **0.9617** 打底；简单的 U-Net 大约在 **0.97** 附近；把编码器换成 ResNet34 预训练权重并配上加权交叉熵的方案到了 **0.973**；最后加上了 best epoch、加权损失和增强组合后，公开的最好成绩达到了 **0.98340**。

最后是推理与提交。测试集没有标签，我们对每个文件做同样的前向计算，把输出插值回 50×181，每帧得到 9050 个像素的类别。**别忘了把类别减 1**——训练时我们把 -1～3 平移到 0～4，推理得到 0～4 之后要原样减回来，否则新时代标签区间对不上，提交格式直接全错：

```python
def run_inference(model, data_loader):
    """在测试集上推理, 返回展平后的像素预测和文件名"""
    model.eval()
    predictions, filenames = [], []

    with torch.no_grad():
        for images, file_names in data_loader:
            images = images.to(device)
            outputs = model(images[:, 0], images[:, 1], images[:, 2],
                            images[:, 3], images[:, 4], images[:, 5])
            # 输出尺寸对齐到 50x181 再取 argmax
            outputs = F.interpolate(outputs, size=(50, 181),
                                    mode='bilinear', align_corners=False)
            preds = torch.argmax(outputs, dim=1) - 1   # 从 0~4 减回 -1~3

            for i, pred in enumerate(preds):
                predictions.append(pred.cpu().numpy().reshape(-1))  # 9050 个像素
                filenames.append(file_names[i])

    return predictions, filenames
```

把预测结果整理成比赛要求的 CSV 格式：每一行是一个文件名，后面 9050 列分别是 `pixel_0` 到 `pixel_9049`：

```python
import pandas as pd

results = []
for filename, pred in zip(test_filenames, test_predictions):
    row = {'filename': filename}            # 第一列是文件名
    for i, p in enumerate(pred):
        row[f'pixel_{i}'] = int(p)          # 每帧 9050 个像素各占一列
    results.append(row)

test_df = pd.DataFrame(results)
test_df.to_csv('submission.csv', index=False)
print('提交文件形状:', test_df.shape)       # (样本数, 1 + 9050)
```

提交前我们还会做一道"安全检查"：断言每帧预测长度恰好是 9050，列名是 `filename` 加 `pixel_0`...`pixel_9049`，样本行数和测试集一致。比赛里因为提交格式错误而白丢分的事很常见，这一下子几乎不花时间却非常值得。

---

## 三、总结

本文中我们完成了雷达感知（IOAI 2025 Task 1）这个语义分割比赛的完整流程。我们首先通过 EDA 弄清了数据形态：每个样本是一个 6×50×181 的六通道雷达热力图加上一张 50×181 的标签图，6 个通道分别对应距离-方位/距离-俯仰的静态、动态回波和两张 MUSIC 概率图，每帧要预测 9050 个像素、类别有背景/行李箱/椅子/人/墙五类，并且背景像素占了绝大多数。其次我们在数据清洗阶段保持克制，只做类型转换、给标签 +1、区分训练集与测试集，另外针对 181 是奇数这个下采样的坑做了补齐处理。接着我们沿官方参考方案，使用"每个通道一套编码器-解码器、最后融合特征"的 DIMSSNet 结构，配合 focal loss 缓解前景/背景严重不均衡，并用加权交叉熵、随机翻转/旋转增强、AdamW + 余弦退火等调尝试把分数一步步推高，再用 best epoch 技巧选出验证指标最高的权重。模型测试与提交阶段，我们复用了比赛的加权准确率指标做本地评估，把类别在 -1～3 和 0～4 之间正确转换，最终在排行榜上从简单的 0.9617 一路提升，公开的最高成绩达到 0.98340。

这一场比赛给我们的最大启发是：**优化目标要和评分规则对齐，而且预处理要跟着数据走**。雷达热力图不是自然图像，不能盲目套用图像的做法；前景和背景的极端不均衡，让"给损失加权"和"选最优轮次"这种小技巧变得比堆叠更大的网络更有效；比赛的加权指标从一开始就告诉我们该往哪个方向使劲——把前景分割准，比把背景分割准值钱 50 倍。理解了评价指标，就理解了大半个比赛。