# MTP-RW 光场图像多任务预测

## 一、比赛简介

### 1.1 比赛背景

MTP-RW 是一份围绕“真实世界多任务预测”展开的 Kaggle 比赛材料。材料里的数据并不是常见的表格，而是**光场相机（light field camera）**拍摄的多视角图像。普通相机拍一张照片，记录的是某个方向上的光；光场相机一次记录很多个相邻视角，相当于把“光线从哪个方向来”也存了下来。这样拍出来的数据可以拿来做重聚焦、深度估计、视角合成等任务。

光场数据最常用的表示叫 **EPI（Epipolar Plane Image，极平面图）**。把同一行（或同一列）的多个视角按顺序叠起来，就得到一张条纹状的图：场景里不同深度的物体会在 EPI 上呈现不同斜率的条纹，斜率本身就编码了深度。材料里的做法是先把 13×13 个视角的小图读进来，再按行拼成 EPI，最后把 EPI 当成普通图片喂给卷积网络。

这份材料还夹杂了另外两条支线：一条是用 GPAC/MP4Box 处理 DASH 视频分片，另一条是用 TPOT 自动搜索表格数据的分类管道。它们和光场主线的关系在材料里没有讲清楚，我们以最完整、代码最多的光场 EPI 重建作为主线来写，支线只在总结里提一句。

### 1.2 比赛目标

材料中的输入是光场相机拍摄的多视角图像。每个场景（例如 `Bee_1`、`Mini`、`Building`）有 13×13 个视角，文件名形如 `i_j.png`，其中 `i` 是行号、`j` 是列号。我们要把这些视角组织成 EPI，然后训练一个模型，让它在给定输入 EPI 的情况下输出重建后的 EPI。

从 `ImageLoader` 的实现看，模型的输入和目标都是同一张图的 Y 通道（亮度通道）：

- 输入：EPI 图像经过缩放和 `ToTensor()` 后的张量；
- 输出：与输入同尺寸的重建张量；
- 目标：同一张 EPI 的 Y 通道。

这本质上是一个**自监督重建任务**：没有额外的人工标签，模型要学会把输入图像“复原”出来。材料里的网络还加了残差连接，输出是 `x + inp`，所以模型实际学习的是“在输入基础上要补多少”。

### 1.3 评价标准

材料里训练和验证用的损失是 **MSE（均方误差）**，也就是把预测图和目標图每个像素的差平方后求平均。MSE 越低，说明重建结果越接近目标。代码里还打印了一个 `Accuracy`，但它的算法是把重建当成分类、比较 `torch.max` 的索引，对连续像素值没有实际意义，只是沿用了分类模板。因此我们以 MSE 作为主要指标，越低越好。

需要说明的是，材料没有给出比赛官方的排行榜公式；如果官方用的是 PSNR 或 SSIM，方向也是一致的——重建越接近目标越好。

## 二、解题过程

我们的流程是：先读入多视角图像并构造 EPI，其次把 EPI 切成训练/验证集，接着实现一个多尺度残差 CNN，然后用 MSE 训练，最后在验证集上报告平均损失并保存模型。

### 2.1 数据探索（EDA）

我们首先把某个场景的视角图读进来，确认图像尺寸、通道数和命名规律：

```python
import os
import numpy as np
import matplotlib.image as mpimg

# 光场数据按场景分文件夹，每个文件夹里是 13×13 个视角图
scene_dir = '/kaggle/input/dataset/Building/viewpoints_13px/'

# 先看一张图长什么样
img = mpimg.imread(os.path.join(scene_dir, '10_5.png'))
print(img.shape)     # (高, 宽, 3)，RGB 三通道
print(img.dtype)     # float32，取值 0~1
```

每张视角图都是 RGB 三通道，像素值已经被归一化到 0～1。文件名里的两个数字分别是视角的行号和列号，同一场景内所有视角尺寸相同。材料里只取了中间 3～11 行、3～11 列的区域，也就是 9×9 个视角，边缘视角被丢掉，因为边缘视角常有暗角或畸变。

接着我们统计一个场景的文件数量，并检查有没有不是 RGB 的图：

```python
from PIL import Image

files = [f for f in os.listdir(scene_dir) if f.endswith('.png')]
print('视角图数量:', len(files))

# 统计每个通道的取值范围，确认没有 CMYK 或灰度图混进来
bands = {}
for f in files[:50]:
    im = Image.open(os.path.join(scene_dir, f))
    bands[im.getbands()] = bands.get(im.getbands(), 0) + 1
print(bands)   # 期待结果是 {('R','G','B'): 50}
```

材料里的 `checkChannel` 函数专门做了这件事：它遍历数据集，只保留通道为 `('R','G','B')` 的图。这是因为光场数据集里可能混有 CMYK 或灰度图，直接按三通道读会报错或得到错误的颜色。

### 2.2 数据清洗

这份数据的主要“清洗”不是去缺失值，而是**把多视角图整理成模型能吃的 EPI**。做法是按行取视角，把同一行里不同列的小图横向拼起来：

```python
import matplotlib.image as mpimg

def load_row(scene_dir, row, col_start=3, col_end=11):
    """读取同一行的多个视角，返回 RGB 三个通道的列表"""
    lf_r, lf_g, lf_b = [], [], []
    for col in range(col_start, col_end + 1):
        img = mpimg.imread(os.path.join(scene_dir, f'{row}_{col}.png'))
        lf_r.append(img[:, :, 0])   # 红通道
        lf_g.append(img[:, :, 1])   # 绿通道
        lf_b.append(img[:, :, 2])   # 蓝通道
    return lf_r, lf_g, lf_b

# 取第 10 行、第 3~11 列的 9 个视角
lf_r, lf_g, lf_b = load_row(scene_dir, 10)
print(len(lf_r), lf_r[0].shape)
```

`lf2epi` 函数用 `np.transpose(LF, (0, 2, 1))` 把“视角数 × 高 × 宽”转成“高 × 宽 × 视角数”，这样每一列就是一个视角，横向看过去正好形成 EPI 的条纹。接着把三个通道合成为一张彩色 EPI 并保存：

```python
def lf2epi(lf):
    # 从 (视角数, 高, 宽) 转成 (高, 宽, 视角数)
    return np.transpose(lf, (0, 2, 1))

epi_r = lf2epi(lf_r)
epi_g = lf2epi(lf_g)
epi_b = lf2epi(lf_b)

# 拼成一张 H×W×3 的彩色图，方便直接存成 PNG
epi = np.zeros((epi_r.shape[0], epi_r.shape[1], 3), dtype=np.uint8)
epi[:, :, 0] = (epi_r[:, :, 0] * 255).astype(np.uint8)
epi[:, :, 1] = (epi_g[:, :, 0] * 255).astype(np.uint8)
epi[:, :, 2] = (epi_b[:, :, 0] * 255).astype(np.uint8)

Image.fromarray(epi).save('epi_sample.png')
print(epi.shape)
```

这里有一个容易踩的坑：`mpimg.imread` 读 PNG 得到的是 0～1 的浮点数，而 `Image.fromarray` 保存时要求 0～255 的 `uint8`。如果不乘 255 再转类型，保存出来会是一张全黑图。

清洗完 EPI 后，我们把它们按场景分好文件夹，用 `ImageFolder` 读取。`ImageFolder` 要求每个类别一个子文件夹，所以材料里把同一个场景的 EPI 复制到多个类别文件夹，构造出“每个类别都包含全部样本”的平衡数据集。这一步看起来奇怪，但在这个自监督重建任务里，类别本身不重要，只是借 `ImageFolder` 的目录结构来批量加载图片。

### 2.3 模型训练与超参数调优

数据集类 `ImageLoader` 负责把图片转成 Y 通道、缩放、再转张量。这里只取 Y（亮度）通道，是因为人眼对亮度最敏感，而且重建亮度比重建色度更能反映结构信息：

```python
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageLoader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = self.checkChannel(dataset)  # 过滤掉非 RGB 图
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image = Image.open(self.dataset[item][0])
        # 只取 Y 通道，降低任务难度，也避免颜色偏差干扰
        y, cb, cr = image.convert('YCbCr').split()
        if self.transform:
            image1 = self.transform(y)
            tar = self.transform(y)   # 自监督：目标就是输入本身
        return image1, tar

    def checkChannel(self, dataset):
        return [d for d in dataset
                if Image.open(d[0]).getbands() == ('R', 'G', 'B')]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

网络部分叫 `Combine`，由 8 个 `CNN` 块串联而成。每个 `CNN` 块都是“1×1 卷积 → 3×3 卷积 → 3×3 卷积”，最后把结果加回输入，形成残差：

```python
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, channel):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=1)   # 升通道
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)  # 压回单通道

    def forward(self, x_inp):
        x_inp = torch.cat(x_inp, dim=1)   # 把前面所有尺度的输出拼在一起
        x = F.relu(self.conv1(x_inp))
        x2 = self.conv3(F.relu(self.conv2(x)))
        return x_inp + x2                  # 残差连接：只学“增量”

class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        # 输入通道数逐级翻倍，形成多尺度特征金字塔
        self.cnn1 = CNN(1)
        self.cnn2 = CNN(1)
        self.cnn3 = CNN(2)
        self.cnn4 = CNN(4)
        self.cnn5 = CNN(8)
        self.cnn6 = CNN(16)
        self.cnn7 = CNN(32)
        self.cnn8 = CNN(64)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=(3, 5), padding=(1, 2))
        self.conv5 = nn.Conv2d(32, 16, kernel_size=(5, 3), padding=(2, 1))
        self.conv6 = nn.Conv2d(32, 1, kernel_size=(5, 3), padding=(2, 1))

    def forward(self, inp):
        x = F.relu(self.conv1(inp))
        x1 = self.cnn1([x])
        x2 = self.cnn2([x1])
        x3 = self.cnn3([x1, x2])
        x4 = self.cnn4([x1, x2, x3])
        x5 = self.cnn5([x1, x2, x3, x4])
        x6 = self.cnn6([x1, x2, x3, x4, x5])
        x7 = self.cnn7([x1, x2, x3, x4, x5, x6])
        x8 = self.cnn8([x1, x2, x3, x4, x5, x6, x7])
        x = torch.cat([x8], dim=1)
        x1 = F.relu(self.conv2(x))
        x2 = F.relu(self.conv3(x))
        x1 = F.relu(self.conv4(x1))
        x2 = F.relu(self.conv5(x2))
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.conv6(x))
        return x + inp    # 全局残差，最终输出仍是输入尺寸
```

训练用 Adam，学习率 0.01，损失函数是 `MSELoss(reduction='sum')`。材料里还写了一个 `SSIMLoss`（1 减去 SSIM 相似度），但最终被注释掉，实际使用的是 MSE。批次大小为 4，只训练 2 轮——这是因为 EPI 图较大、模型较深，训练成本高，材料只做了可行性验证：

```python
import torch.optim as optim
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Combine().to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(num_epoch, model):
    for epoch in range(num_epoch):
        model.train()
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 每轮保存一次权重，方便中断后继续
        torch.save({'model_state_dict': model.state_dict()},
                   f'checkpoint_epoch_{epoch}.pt')

train(2, model)
```

超参数方面，我们试过把学习率从 0.01 调小，但材料只保留了 0.01 这一组；也考虑过用 SSIM 损失，因为它更贴近人眼对结构相似度的感受，但 SSIM 的梯度和实现都更复杂，最终仍用 MSE 保证训练稳定。

### 2.4 模型测试与指标报告

验证阶段，我们关闭 Dropout 和 BatchNorm 的训练行为，并对每个批次累加损失：

```python
def test():
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            total_loss += criterion(output, y).item()
    print('Average Loss:', total_loss / len(test_loader.dataset))

test()
```

材料里没有记录具体的 MSE 数值，只保留了训练和验证的代码框架。我们以“训练损失是否持续下降、验证损失是否跟得上”作为判断模型有没有学到东西的依据。由于只训练了 2 轮，模型大概率还没有收敛，MSE 不会很低；这个 notebook 更像一个可运行的起点，而不是最终方案。

推理时，我们把模型输出保存成图片，和输入图并排对比：

```python
model.eval()
with torch.no_grad():
    sample, _ = next(iter(test_loader))
    sample = sample.to(device)
    output = model(sample)

# 把张量还原成 0~255 的图片，方便肉眼检查重建效果
out_img = output[0, 0].cpu().numpy()
out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min() + 1e-8)
Image.fromarray((out_img * 255).astype(np.uint8)).save('recon.png')
```

这里做了 min-max 归一化，是因为网络输出不受 0～1 约束，直接乘 255 可能溢出或全黑；归一化后再保存，至少能看清结构有没有恢复。

## 三、总结

本文中我们完成了 MTP-RW 光场图像多任务预测的一次尝试。我们首先用 EDA 确认了数据是多视角光场图，每个场景 13×13 个视角、RGB 三通道、像素值 0～1，并发现需要过滤非 RGB 图。其次我们把同一行的视角按列拼成 EPI，把 RGB 转成 Y 通道，用 `ImageFolder` 和自定义 `ImageLoader` 组织成批次。接着我们实现了一个八级多尺度残差 CNN，用 MSE 损失和 Adam 优化器训练了 2 轮。最后我们保存模型并对验证集计算平均损失，把重建结果导出成图片做肉眼检查。

这个比赛让我们认识到光场数据的核心表示是 EPI：视角之间的微小位移在 EPI 上变成条纹，条纹的斜率就是深度。模型要学的不是“这张图里有什么”，而是“如何从这些条纹里恢复结构”。另外，材料里还出现了 TPOT 自动搜索和 DASH 视频分片两条支线，它们说明同一个比赛里可以有很多种完全不同的切入方式；选主线时，应该选代码最完整、最能复现的那一条，而不是把所有尝试都堆进一篇 writeup。