# TechNist 图像分类（baseline 教程）

## 一、比赛简介

### 1.1 比赛背景

TechNist 是一个以「让初学者跑通深度学习完整流程」为目的的图像分类比赛。比赛提供了训练图片文件夹 `train/`、训练标签文件 `train.csv` 和测试图片，要求参赛者开发一个模型，把输入图片分到若干类别中的一类。

这类比赛的典型特点是：**数据规模不大、类别数固定、单张图片也很小。** 因为图片小，我们可以用很小的网络（比如只有三个全连接层）就能在 CPU 上完成训练循环；因为类别数不多，交叉熵损失和准确率都很好理解。换句话说，它的定位和著名的 MNIST（手写数字识别）类似——用最简单的任务，把「读数据、做变换、搭模型、训练、评估、保存」这一整套动作练熟。

比赛的难点不在于模型要多么花哨，而在于几个容易被初学者卡住的细节：图片文件名和标签如何对应、数据放进神经网络之前要做什么变换、训练集和验证集怎么划分、怎么记录「验证集上最好的模型」。这篇 baseline 教程恰好把这几个点都覆盖到了，我们就把它的思路完整地讲一遍。

### 1.2 比赛目标

输入包括两部分：

- 图片文件：放在 `train/train/` 目录里，每一张是一张待分类的图片；
- `train.csv`：保存图片对应的标签，我们用代码里的 `OUTPUT_DIM = 15` 可以确认，任务是 **15 类分类**，也就是每张图要预测 0 到 14 之间的一个整数。

我们把这个比赛的监督信息拆开看：代码里用 `Custom_Dataset` 把图片文件和 `train.csv` 里的标签按顺序配对。`__getitem__` 返回 `(image, label)`，训练时数据加载器按这个成对的数据去迭代。

### 1.3 评价标准

这个比赛以**分类准确率（Accuracy）**为衡量标准，也就是「预测对的图片数占总图片数的比例」。推理时，模型对每张图片输出 15 个类别的分数，我们取分数最高的那个类别作为预测。预测类别和真实类别一致就算对，然后统计正确比例。

准确率越高越好。它没有像 F1 那样区分「精确率」和「召回率」，对类别均衡的数据来说足够直观；如果比赛数据存在类别不均衡，准确率就不是唯一参考，但 baseline 阶段我们先按准确率来评估拟合程度。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**写一个自定义数据集把图片和标签对应起来，用一套数据变换把图片变成统一的张量，再用一个三层全连接网络（MLP）训练 10 个 epoch，取验证集损失最小的那一版模型。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先加载库、固定随机种子，为了「每次运行都得到同样的结果」：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms

from sklearn import metrics
import numpy as np
import pandas as pd
import random

# 深度学习里很多初始化都依赖随机数，
# 固定种子后，只要代码不变，重复运行结果就一致，方便排查问题
SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```

接着我们查看训练数据的大小和标签分布：

```python
image_dir = '../input/technist/train/train/'
csv_file = '../input/technist/train.csv'

# 先看目录里有多少张图片、train.csv 有多少行，
# 两者应该一致，否则说明文件对应关系有问题
from pathlib import Path
print("images:", len(list(Path(image_dir).glob("*"))))
df = pd.read_csv(csv_file)
print("csv rows:", len(df))

# 用 value_counts 看每个类别有多少张图，检查类别是否均衡
print(df.iloc[:, 1].value_counts().sort_index())
```

`value_counts().sort_index()` 输出 15 个类别的样本数。如果某些类别特别少，训练时模型容易把它们学偏；不过作为 baseline，我们先确认「类别数是 15、每个类别都有样本」，具体不均衡程度留到后面再说。

我们还会看一眼标签列的名字和图片文件名的关系，确认 `Custom_Dataset` 里用「同一顺序」配对是否合理：

```python
print(df.head())
print(df.columns.tolist())
```

`df.head()` 会打印前几行，第 0 列是图片名（或编号），第 1 列是标签。这正是代码里 `pd.read_csv(csv_file).iloc[:, 1]` 只取第 1 列当标签的原因：标签全在固定的一列里，不需要去猜哪一列有意义。

### 2.2 数据清洗

深度学习中，「数据清洗」变成了一件更系统的事：**把每一张图片都转成大小一致、数值范围一致的张量，顺便做一点数据增强。** 这里我们用 `torchvision.transforms` 完成。

```python
# 传给 torchvision 的变换组合：
# Resize(28) 把所有图片统一缩放到 28x28，
# 因为后面的全连接网络要求输入必须是固定长度
transform_img = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),   # 把 PIL 图片变成张量，并自动把像素归一化到 [0,1]
])

# 定义自定义数据集：图片目录 + 标签 csv + 变换
class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)   # 图片名列表
        self.data = pd.read_csv(csv_file).iloc[:, 1]  # 只取标签这一列
        self.transform = transform

    def __len__(self):
        return len(self.data)          # 数据集长度 = 标签行数

    def __getitem__(self, index):
        # 按索引拼出图片完整路径，读成 PIL 图像
        image_name = os.path.join(self.image_dir, self.image_files[index])
        image = PIL.Image.open(image_name)
        label = self.data[index]       # 同一个索引对应的标签
        if self.transform:
            image = self.transform(image)
        return (image, label)          # 返回 (图片张量, 标签)
```

这段代码里最重要的是 `__getitem__` 的配对逻辑：`image_files[index]` 和 `self.data[index]` 用的是同一个 `index`。这意味着**图片文件列表的顺序必须和 csv 标签的顺序一致**，这是初学者最容易踩的坑——图片明明是对的，就因为文件排序方式和 csv 不一致，标签全错位了。

我们在数据清洗阶段**不做额外的去噪、去重**。图片分类任务里，常规的清洗就是标准化大小和数值范围；更多的手段（裁剪、翻转、旋转）属于数据增强，放在训练时做，而不是预处理时做。

训练集和验证集的划分也属于「数据处理」的一部分。我们用 `random_split` 按 8 : 2 切分：

```python
# 实例化数据集，并加载成 DataLoader
dataset = Custom_Dataset(image_dir, csv_file, transform=transform_img)

# 按 8:2 切分出训练集和验证集
train_percent = 0.8
train_size = int(train_percent * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# DataLoader 按 batch_size 打包样本，shuffle=True 让每个 epoch 顺序打乱
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, num_workers=0, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=batch_size, num_workers=0
)
```

`shuffle=True` 只加在训练集上，验证集保持原序。原因是训练时我们希望模型每个 epoch 看到不同的 batch 组合，避免在固定顺序上"背答案"；验证时结果和顺序无关，不打乱反而便于复现。

### 2.3 模型训练与超参数调优

我们接着定义模型。因为图片是 28×28 的灰度图，全连接网络把图片拉平成 784 维向量，然后经过两个隐藏层、一个输出层：

```python
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 三层全连接：784 -> 250 -> 100 -> 15
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]        # 每个 batch 的样本数
        x = x.view(batch_size, -1)     # 把 28x28x1 拉平成 784 维向量
        h_1 = F.relu(self.input_fc(x)) # 隐藏层一，非线性的关键
        h_2 = F.relu(self.hidden_fc(h_1))  # 隐藏层二
        y_pred = self.output_fc(h_2)   # 输出层不做激活，交给损失函数
        return y_pred, h_2             # 返回预测和中间向量（便于可视化）

INPUT_DIM = 28 * 28   # 输入维度 = 784
OUTPUT_DIM = 15       # 输出维度 = 15 类
model = MLP(INPUT_DIM, OUTPUT_DIM)

optimizer = optim.Adam(model.parameters())   # Adam 优化器
criterion = nn.CrossEntropyLoss()            # 多分类交叉熵损失
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)
```

几个值得解释的设计选择。**第一，输出层不加激活函数。** 交叉熵损失内部有 Softmax，所以网络输出 15 个原始分数直接传给损失函数即可，提前加 Softmax 反而会重复。**第二，用 Adam 而不是手动调学习率。** Adam 自动为每个参数调整学习率，baseline 阶段几乎不需要调超参数。**第三，模型规模很小。** 784 → 250 → 100 的容量处理 28×28 图片绰绰有余，也保证 CPU 上跑得动。

训练循环负责「一个 epoch 内把每个 batch 都走一遍」：

```python
def calculate_accuracy(y_pred, y):
    # 取分数最高的类别作为预测
    top_pred = y_pred.argmax(1, keepdim=True)
    # 和真实标签逐元素比较，算正确个数占总数的比例
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()   # 开启训练模式，Dropout/BatchNorm 才生效
    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()   # 清空上一次迭代的梯度
        y_pred, _ = model(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        loss.backward()         # 反向传播计算梯度
        optimizer.step()        # 沿梯度方向更新权重
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

`optimizer.zero_grad()` 是新手最容易漏的一行。PyTorch 默认是**累积梯度**的，不在每个 batch 前清空，梯度就会把上一个 batch 的信息带进来，相当于用一个「混着旧数据的」梯度去更新，模型会越来越乱。

验证函数和训练函数结构几乎一样，但有几个关键区别：

```python
def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()          # 评估模式
    with torch.no_grad(): # 不计算梯度，节省显存和内存
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

`torch.no_grad()` 是这个函数的核心。验证阶段我们只需要算前向结果，不需要回传梯度；关掉梯度记录既不会影响结果，又能明显加快速度。`model.eval()` 则告诉模型切换到推断模式，避免 Dropout 这类层在验证时随机丢神经元。

训练 10 个 epoch，并保存验证集损失最小的模型：

```python
EPOCHS = 10
best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    start_time = time.monotonic()
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, val_loader, criterion, device)

    # 如果验证损失比历史最好还小，就保存当前权重
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'baseline_model.pt')

    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time // 60)          # 整分钟数
    elapsed_secs = int(elapsed_time % 60)           # 去掉分钟后剩下的秒数
    print(f'Epoch: {epoch+1:02} | Time: {elapsed_mins}m {elapsed_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
```

我们**用验证集损失做「早停」的决策依据**，而不是验证准确率。原因很简单：损失是连续值，能区分「同样 90% 准确率下哪个模型更自信」；而且训练过程中损失下降通常比准确率变化更平滑，用它来比较不同 epoch 的模型更稳。

关于超参数：这个 baseline 只用了很朴素的默认值——`Adam()` 用默认学习率、batch size 64、epoch 10。我们没有做网格搜索，因为对这么小的模型和数据集，约束效果的是模型容量和是否需要更强的数据增强，而不是学习率的微小变化。等跑通流程后，再回来把这些超参数逐个调一遍，才会知道它们各自带来多少提升。

### 2.4 模型测试与指标报告

我们最后在验证集上做一次正式评估，报告准确率，然后就可以用保存的模型对测试集推理。材料没有附带测试集的推理脚本，这里我们根据训练循环的思路补一个推断流程（伪代码）：

```python
# 加载验证集上表现最好的权重
model.load_state_dict(torch.load('baseline_model.pt'))
model.eval()

# 用验证集做一次最终评估
valid_loss, valid_acc = evaluate(model, val_loader, criterion, device)
print(f'Final validation accuracy: {valid_acc*100:.2f}%')

# ---- 测试集推断（伪代码） ----
# 对测试集逐 batch 预测：
#   1) with torch.no_grad() 下拿到 y_pred；
#   2) preds = y_pred.argmax(1)，得到每个样本的类别；
#   3) 把结果写进带 id 的提交文件。
```

到这里，整个「训练 → 验证 → 保存最佳模型」的闭环就完成了。我们把它叫做「测试前报告」而不是「测试报告」，是因为：**验证集的分数用来选模型，测试集的分数才是比赛的最终成绩。** 因为测试集标签我们见不到，本地能做的就是在验证集上把流程跑通、确认模型没有明显过拟合（训练准确率远高于验证准确率就说明过拟合了），然后把权重保存下来用于推理。

---

## 三、总结

本文中我们完成了 TechNist 图像分类比赛的完整 baseline。我们首先通过读取目录和 `train.csv` 确认了数据规模与 15 个类别；其次用 `transforms` 把图片统一缩放到 28×28、转成张量，并用手写的 `Custom_Dataset` 把图片和标签正确配对，再按 8:2 划分训练与验证集；接着搭建了一个 784 → 250 → 100 → 15 的三层全连接网络，用 Adam + 交叉熵迭代训练 10 个 epoch，并在每个 epoch 后比较验证损失、只保留最优的一版权重；最后在验证集上报告准确率，确认训练与验证差距没有明显过拟合，就可以进入测试集推断阶段。

这个比赛给我们的最大启发是：**深度学习的难点经常不在「模型」而在「数据流」。** 图片和标签的对齐、变换的次序、梯度是否清零、验证时是否用 `no_grad`，这些细节任何一个出问题，模型都跑不对——而它们恰恰是 `Custom_Dataset` + DataLoader 这套组合要解决的。把这条数据流走顺，后面的改进（加大模型、加数据增强、加早停）都是在这个地基上添砖加瓦。