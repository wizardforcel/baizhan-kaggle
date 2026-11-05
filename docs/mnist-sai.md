# MNIST-SAI 手写数字识别

## 一、比赛简介

### 1.1 比赛背景

MNIST 是机器学习里“Hello World”级别的数据集：6 万张小写的手写数字图片（0 到 9），每张都是 28×28 的灰度图。无数入门教程都从它开始，因为它图像小、类别少、训练快，但又能完整走一遍造模型、训练、评估、提交的流程。

`mnist-sai` 这个比赛就是 MNIST 的一个 Kaggle 版本。它与标准 MNIST 的区别在于数据文件的组织方式：没有拆成常见的 CSV，而是把所有训练图像 `x_train`、训练标签 `y_train` 和测试图像 `x_test` 打包在一个 `.npz` 文件里。这种格式在真实工程里很常见，也是这个比赛想让大家习惯的。

材料里的 notebook 用 PyTorch 实现了一个简单的卷积神经网络（CNN）。CNN 是专门处理图像的网络结构：它用一个个小“窗口”扫描图像，能自动发现“圆圈”“竖线”这类局部图案，再逐步组合成数字的识别结果。

### 1.2 比赛目标

给定一张 28×28 的手写数字灰度图，预测它是 0～9 中的哪一个数字。数据加载方式如下：

- `data['x_train']`：训练图片数组；
- `data['y_train']`：训练标签数组；
- `data['x_test']`：测试图片数组。

每张图片的像素值范围是 0～255（0 是黑色，255 是白色）。提交文件需要两列：图片编号 `Id` 和预测类别 `Category`。

### 1.3 评价标准

比赛使用**分类准确率（accuracy）**，即预测正确的图片占全部测试图片的比例。它计算的是“得分最高的类别是否猜对了”，完全不看模型对其他类别的信心高低，所以最终提交时取 softmax 输出里最大值对应的类别即可。准确率越高越好，最高是 1.0。

## 二、解题过程

我们的流程是：先把 `.npz` 读进来并归一化像素，其次把数据包装成 PyTorch 的批加载器，接着搭一个两层卷积的 CNN，训练 10 轮，最后对测试集推理并生成提交文件。

### 2.1 数据探索（EDA）

我们首先把 `.npz` 文件读进来，看每个数组的形状和数值范围：

```python
import numpy as np
import pandas as pd
import torch

# 从 npz 包里一次性取出训练图片、训练标签和测试图片
data = np.load('/kaggle/input/mnist-sai/mnist_kaggle.npz')
x_train, y_train, x_test = data['x_train'], data['y_train'], data['x_test']

print(x_train.shape)   # 训练图片数量 × 28 × 28
print(x_test.shape)    # 测试图片数量 × 28 × 28
print(y_train.shape)   # 每条训练图片对应的数字标签
print(x_train.max(), x_train.min())   # 像素值范围，确认是否需要归一化
```

输出显示训练集和测试集都是单通道的 28×28 灰度图，像素范围是 0～255。灰度图和彩色图（RGB 三通道）不同，只有一个通道，所以后面卷积层的 `in_channels=1`。

我们还检查了标签的分布：

```python
# 每个数字出现了多少次
counts = np.bincount(y_train)
print(counts)   # 10 个数字各自的样本数
```

结果说明 0～9 的样本量大致均衡。这对分类模型很友好：不需要像不平衡数据那样专门做采样或加权，直接按准确率优化即可。

### 2.2 数据清洗

MNIST 的数据本身比较“干净”，不需要修缺失值或去重。需要做的清洗只有两件小事。

第一，把像素除以 255 归一化到 0～1 之间：

```python
# 像素值从 0~255 缩放到 0~1，让网络各层输入的尺度一致，训练更稳
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
```

为什么不直接喂 0～255？因为神经网络的激活函数（比如 ReLU）对输入尺度敏感，输入太大或太小都会让梯度不稳定。归一化到 0～1 是最常见的做法。

第二，把数组换成 PyTorch 需要的格式。PyTorch 的卷积层期望输入是 `(样本数, 通道数, 高, 宽)`，而 NumPy 里通常是 `(样本数, 高, 宽)`，只有三个维度。我们用 `unsqueeze(1)` 在第二个位置插入一个“通道”维度：

```python
# 在 (样本数, 高, 宽) 的第 1 个位置插入通道维，变成 (样本数, 1, 高, 宽)
x_train = torch.from_numpy(x_train).unsqueeze(1)
x_test = torch.from_numpy(x_test).unsqueeze(1)
y_train = torch.from_numpy(y_train)
print(x_train.shape)   # (N, 1, 28, 28)
```

做完这两步就不需要别的清洗了。接着我们把数据装进 `DataLoader`，它会自动把数据切成小批次并支持乱序：

```python
from torch.utils.data import DataLoader, TensorDataset

# TensorDataset 把图片和标签一一对应打包
train_data = TensorDataset(x_train, y_train)

batch_size = 100
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
```

`shuffle=True` 让每个批次里的样本顺序随机打乱，避免模型记住批次内的固定顺序；测试时不打乱，方便后续按顺序写提交文件。

### 2.3 模型训练与超参数调优

网络是两层卷积加三层全连接：第一层卷积把 1 通道变成 10 通道，第二层把 10 变成 30 通道，每个卷积后面都跟一个 ReLU 和最大池化；全连接层把卷积得到的特征压成 250、50 维，最后输出 10 个数字的概率分数：

```python
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积：1 通道 -> 10 通道，3x3 卷积核，再接 2x2 最大池化
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        # 第二层卷积：10 通道 -> 30 通道
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 30, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        # 全连接层：把特征展平后逐层降维到 10
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(750, 250),
            torch.nn.Linear(250, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size, -1)   # 展平成一维向量，供全连接层使用
        x = self.fc(x)
        return x

model = Net()
```

为什么卷积后是 750 维？28×28 的图经过两次“3×3 卷积 + 2×2 池化”，尺寸先变小再减半，最后变成 5×5；30 个通道的 5×5 就是 30×5×5 = 750 个数值。这个数字是网络结构决定好的，写代码时只要保证 `view` 的维度对得上即可。

训练的超参数都来自材料：损失函数用交叉熵 `CrossEntropyLoss`，优化器用 `Adam`，学习率固定为 0.001，批次大小 100，共训练 10 轮。训练循环是标准的“前向 → 算损失 → 清梯度 → 反向传播 → 更新参数”：

```python
from torch.optim import Adam

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

def train(epoch):
    for e in range(epoch):
        for idx, (input, target) in enumerate(train_loader):
            optimizer.zero_grad()        # 清空上一轮的梯度
            output = model(input)        # 前向传播，得到预测分数
            loss = loss_fn(output, target)  # 计算真实标签与预测的交叉熵损失
            loss.backward()              # 反向传播，算出各参数的梯度
            optimizer.step()             # 沿梯度方向更新一次参数
            if idx % 100 == 0:
                print(e, idx, loss.item())

train(10)
```

我们没有做复杂的超参数搜索，而是直接采用材料里的默认配置。这个网络规模很小、数据很规整，Adam + lr=0.001 通常就能稳定收敛；训练过程中每 100 步打印一次损失，方便确认损失在持续下降。

### 2.4 模型测试与指标报告

训练完成后，我们进入推理阶段。先用 `torch.no_grad()` 关闭梯度计算——推理时不需要反向传播，关掉能省显存和耗时；然后对每个批次取 softmax 输出里得分最大的类别作为预测：

```python
def test():
    predictions = []
    with torch.no_grad():               # 推理阶段不需要算梯度
        for batch in test_loader:
            output = model(batch[0])
            _, predicted = torch.max(output, 1)   # 取概率最高的类别
            predictions.extend(predicted.cpu().numpy())
    return predictions

preds = test()
print(len(preds))   # 数量应等于测试图片总数
```

因为材料里没有记录验证集和测试集的具体准确率数字，我们以“训练损失逐轮下降”作为模型正常学习的信号，最后生成的提交文件结果交给排行榜评判。MNIST 这类任务上，这个规模的 CNN 通常能达到很高的准确率；如果想进一步调优，可以增加卷积层数、加入 BatchNorm，或用更小的批次训练更多轮。

最后把预测结果写进提交文件。注意 `Id` 是测试图片的顺序编号，从 0 开始：

```python
result_df = pd.DataFrame({
    "Id": range(len(preds)),
    "Category": preds
})
result_df.to_csv('/kaggle/working/submission.csv', index=False)
print(result_df.head())
```

写完之后我们再检查一遍：行数是否等于测试图片数，`Id` 是否从 0 连续编号，只能出现 0～9 这 10 个类别。提交格式错误会直接得零分，哪怕模型再好也白搭，这一步不能省。

## 三、总结

本文中我们完成了 MNIST-SAI 手写数字识别的完整流程。我们首先把 `.npz` 数据读进来，确认了训练集、测试集的形状和像素范围，并检查了标签分布大致均衡。其次做了最轻量的清洗：像素除以 255 归一化，再补一个通道维度转成 PyTorch 的 `(样本数, 1, 28, 28)` 格式，用 `DataLoader` 分批加载。接着我们用两层卷积加三层全连接的 CNN，配合交叉熵损失和 Adam 优化器训练了 10 轮。最后我们对测试集逐批推理，把 argmax 的结果与图片编号一起写入提交文件。

这个比赛让我们体会到两点：一是哪怕是最简单的基准题，把数据组织方式（`.npz`、通道维度、批次加载）理清楚就是成功的一半；二是训练和推理要区分对待，推理时关掉梯度、不打乱批次，既省资源又能保证提交顺序正确。