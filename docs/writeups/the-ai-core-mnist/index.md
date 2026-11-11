# The AI Core MNIST

## 一、比赛简介

### 1.1 比赛背景

MNIST 全称是 “Modified National Institute of Standards and Technology database”，翻译过来就是「经修改的美国国家标准与技术研究院数据集」。它收集了大量真实的手写数字，把每个数字扫描、裁剪、缩放成一张 28×28 像素的灰度小图，再由人工标注出这张图写的是 0 到 9 中的哪一个数字。因为图片小、任务直观、标签可靠，它被很多人称为计算机视觉领域的「Hello World」——就像学编程时先打印一行 Hello World 一样，学图像分类时通常也先拿 MNIST 练手。

这份材料对应的 notebook 写着 “This notebook is created for AI Core class at Imperial College”，说明它是帝国理工学院 AI Core 课程的一份作业。比赛的组织者把原始 MNIST 的二进制文件直接放出来，让参赛者自己解析文件、搭建模型、在手写数字上做十分类。相比直接给 CSV，这种形式更接近真实工程：拿到的是原始数据格式，得先弄懂文件结构才能开始建模。

这个任务看似简单，难点其实不少。不同人的书写习惯差别很大，有的「1」写得笔直，有的带一个大弯钩；有的「7」中间加一横；「4」有时封口有时不封口；「3」「5」「8」在潦草时轮廓会非常接近。再加上扫描分辨率只有 28×28，笔画细节被压缩得很厉害，模型必须从很有限的像素里抓住形状特征。也正因如此，MNIST 一直被视为检验一个图像分类流程是否跑通的标准基线。

### 1.2 比赛目标

比赛提供四个二进制文件，分成训练集和测试集两套：

- `train-images.idx3-ubyte`：训练图片；
- `train-labels.idx1-ubyte`：训练标签；
- `t10k-images.idx3-ubyte`：测试图片；
- `t10k-labels.idx1-ubyte`：测试标签。

训练集有 60000 张图，测试集有 10000 张图。每张图是 28×28 的灰度图，每个像素取 0 到 255 的整数，0 代表纯黑背景，255 代表最亮的笔迹。标签是 0 到 9 的整数，一共十个类别。

这些文件用的是 IDX 格式。图片文件开头 16 个字节是文件头：前 4 个字节是魔数（magic number），用来标识文件类型；后面三个 4 字节整数依次是图片数量、行数、列数。标签文件开头 8 个字节，前 4 个字节是魔数，后 4 个字节是标签数量。所有整数都按大端序（big-endian）存放，也就是高位字节在前。我们要做的就是读入测试图片，为每一张图预测一个数字标签。

### 1.3 评价标准

比赛使用**分类准确率（Accuracy）**作为评价标准，算法很简单：

```
准确率 = 预测正确的图片数 / 图片总数
```

例如 10000 张测试图里猜对了 9800 张，准确率就是 0.98。这个指标只看最终预测的类别，不关心模型给出的概率大小。因为一共有 10 个类别，随便乱猜的期望准确率只有 0.1，所以准确率越高越好，满分是 1.0。我们在本地评估时也沿用同一个指标，方便和最终成绩对照。

---

## 二、解题过程

整条流程可以概括成一句话：**先把 IDX 二进制文件解析成图片数组，做归一化和形状整理，再搭一个卷积神经网络训练，最后在测试集上算准确率并分析错例。** 我们按数据探索、数据清洗、模型训练与调参、模型测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先做的事情是把原始文件读进来，确认文件结构、图片尺寸和标签范围。材料里的 `loadMNIST` 函数用 NumPy 按大端序解析文件头，但它的路径写死了 `t10k`，传进去的 `prefix` 参数其实没有起作用。为了让训练集和测试集都能正确读取，我们保留了原来的解析思路，把路径改成根据 `prefix` 拼接：

```python
import numpy as np

def loadMNIST(prefix):
    # 文件里的整数采用大端序，必须先声明，否则读出来的数字会完全错乱
    intType = np.dtype('int32').newbyteorder('>')
    # 图片文件头是 4 个 int32：魔数、图片数、行数、列数
    nMetaDataBytes = 4 * intType.itemsize

    image_path = f'/kaggle/input/the-ai-core-mnist/{prefix}-images.idx3-ubyte'
    label_path = f'/kaggle/input/the-ai-core-mnist/{prefix}-labels.idx1-ubyte'

    # 先按字节读入整份文件，再从头部解析元信息
    data = np.fromfile(image_path, dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer(
        data[:nMetaDataBytes].tobytes(), intType
    )
    # 跳过文件头，把剩下的字节还原成 [图片数, 高, 宽] 的三维数组
    images = data[nMetaDataBytes:].astype('float32').reshape([nImages, width, height])

    # 标签文件头是 2 个 int32，跳过 8 个字节后就是每个样本的标签
    labels = np.fromfile(label_path, dtype='ubyte')[2 * intType.itemsize:]
    labels = labels.reshape(labels.shape[0], 1)
    return images, labels

trainingImages, trainingLabels = loadMNIST('train')
testImages, testLabels = loadMNIST('t10k')

print('训练图片', trainingImages.shape, '训练标签', trainingLabels.shape)
print('测试图片', testImages.shape, '测试标签', testLabels.shape)
```

打印结果让我们确认：训练图片是 `(60000, 28, 28)`，测试图片是 `(10000, 28, 28)`，标签分别是 `(60000, 1)` 和 `(10000, 1)`。这说明解析没有问题，图片数量和官方描述一致。

接下来我们看像素取值范围和标签分布。像素范围决定了要不要归一化，标签分布决定了数据是否均衡：

```python
import collections
import matplotlib.pyplot as plt

# 像素最小值和最大值，确认数据没有被压缩成 0/1 二值图
print('像素范围：', trainingImages.min(), trainingImages.max())

# 统计训练集里 0 到 9 各有多少张
counts = collections.Counter(trainingLabels.ravel().astype(int))
for digit in range(10):
    print(digit, counts[digit])

# 随机抽 10 张图连在一起看，直观感受手写风格的差异
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, img, label in zip(axes.ravel(), trainingImages[:10], trainingLabels[:10]):
    ax.imshow(img, cmap='gray')       # 灰度图必须用 gray 色图
    ax.set_title(int(label[0]))       # 标题写真实标签，方便对照
    ax.axis('off')
plt.show()
```

像素范围是 0 到 255，说明图片保留了完整的灰度层次，后面除以 255 就能缩放到 0 到 1。标签分布相当均匀，每个数字大约六千张，没有哪个类别特别稀少，所以我们不需要做重采样或类别加权。看图还能发现一个重要现象：同一个数字的写法五花八门，有的笔画粗、有的笔画细，有的倾斜、有的居中，这正是模型需要学习的「不变性」。

### 2.2 数据清洗

这份数据没有缺失值，也没有重复样本需要删除，所谓「清洗」主要是把 NumPy 数组整理成深度学习框架能直接吃的张量。我们做了三件事：

1. **增加通道维**。卷积层要求输入形状是 `[ batch, channel, height, width ]`。MNIST 是灰度图，通道数为 1，所以要在 28×28 前面补一个维度。
2. **归一化**。把 0 到 255 的像素除以 255，缩放到 0 到 1。这样数值范围小、分布集中，梯度更稳定，模型收敛更快。
3. **统一标签类型**。分类损失函数要求标签是 `long` 类型的类别编号，不能是浮点数。

```python
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

# unsqueeze(1) 在第二维插入通道维，形状从 [N,28,28] 变成 [N,1,28,28]
# 除以 255.0 做归一化，让每个像素落在 0 到 1 之间
X_train_all = torch.tensor(trainingImages, dtype=torch.float32).unsqueeze(1) / 255.0
y_train_all = torch.tensor(trainingLabels, dtype=torch.long).squeeze(1)

X_test = torch.tensor(testImages, dtype=torch.float32).unsqueeze(1) / 255.0
y_test = torch.tensor(testLabels, dtype=torch.long).squeeze(1)

# 把特征和标签打包成 Dataset，DataLoader 才能按批次取数据
full_dataset = TensorDataset(X_train_all, y_train_all)

# 从 60000 张训练图里切出 6000 张做验证集，剩下 54000 张用来训练
# 固定随机种子，保证每次运行切出来的验证集完全一样
train_set, val_set = random_split(
    full_dataset,
    [54000, 6000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
```

这里专门留出验证集，是因为测试集虽然带标签，但我们不应该反复拿它调参——那样等于提前偷看了考卷，最后报出来的分数会偏乐观。验证集只参与「选模型、选超参数」，测试集只在最后用一次，这样得到的指标才可信。训练集加载器打开 `shuffle=True`，每个 epoch 的打乱顺序不同，可以避免模型记住样本顺序。

### 2.3 模型训练与超参数调优

我们选择卷积神经网络（CNN）作为主模型。用大白话解释：全连接层把图片拉成一长串像素，丢掉了「哪个像素和哪个像素相邻」的位置信息；卷积层则拿一个小窗口在图上滑动，专门捕捉局部图案，比如横线、竖线、圆弧。手写数字恰好是由这些局部笔画拼起来的，所以卷积结构非常适合这个任务。

我们搭的网络有四层「特征提取」和两层「分类」：

```python
import torch.nn as nn

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个卷积块：1 通道输入，输出 32 张特征图，卷积核 3×3
        # padding=1 让输出尺寸保持 28×28，不会越卷越小
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),                 # 引入非线性，让网络能拟合复杂形状
            nn.MaxPool2d(2),           # 2×2 池化，尺寸减半到 14×14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 再减半到 7×7
        )
        # 分类头：把 64×7×7 的特征拉平后映射到 10 个类别
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),           # 随机丢弃一半神经元，抑制过拟合
            nn.Linear(128, 10),        # 输出 10 个分数，对应数字 0 到 9
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

第一个卷积层只提取 32 种基础图案，第二个卷积层在它们的基础上组合出更复杂的结构，比如「一个圆弧加一条竖线」。池化层的作用是缩小特征图，既减少计算量，也让模型对几个像素的平移不那么敏感。最后的 `Dropout(0.5)` 在训练时随机失活一部分神经元，相当于每次只让半个网络工作，能明显减轻过拟合。

训练时我们用交叉熵损失（CrossEntropyLoss）和 Adam 优化器。交叉熵衡量的是「模型给正确答案的概率有多低」，概率越低损失越大；Adam 则会根据梯度的历史自动调整每个参数的学习步长，比普通随机梯度下降更省心。

```python
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DigitCNN().to(device)

criterion = nn.CrossEntropyLoss()          # 多分类任务的标准损失
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_acc = 0.0
best_state = None
patience = 3                               # 连续 3 轮没有提升就提前停止
wait = 0

for epoch in range(15):
    model.train()                          # 切换到训练模式，Dropout 才生效
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # 清空上一轮的梯度，否则会累加
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()                    # 反向传播，算出每个参数的梯度
        optimizer.step()                   # 按梯度更新参数

        running_loss += loss.item() * images.size(0)

    # 每个 epoch 结束后在验证集上评估，只记录最好的一版权重
    model.eval()                           # 切换到评估模式，关闭 Dropout
    correct, total = 0, 0
    with torch.no_grad():                  # 评估不需要梯度，省内存也更快
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f'Epoch {epoch + 1}: loss={running_loss / len(train_set):.4f}, val_acc={val_acc:.4f}')

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = model.state_dict()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print('验证集指标不再提升，提前结束训练')
            break
```

**关于超参数调优。** 我们没有一上来就大规模网格搜索，而是先固定一组常见配置跑通，再针对最影响结果的几个参数做小范围对比。具体做法是：学习率在 `1e-2`、`1e-3`、`1e-4` 里比较，批大小在 64 和 128 里比较，Dropout 在 0.3 和 0.5 里比较，每组只训练少量 epoch 看验证集准确率的走势。学习率太大时损失会剧烈震荡甚至发散，太小则收敛很慢；批大小 64 在本机上训练速度和稳定性比较平衡；Dropout 0.5 的验证表现略好于 0.3。最终我们采用学习率 `1e-3`、批大小 64、Dropout 0.5 这一组配置。相比盲目扩大网络，把训练轮数和早停策略调好带来的收益更明显。

### 2.4 模型测试与指标报告

训练结束后，我们取出验证集上表现最好的那一版权重，在完全没有参与过训练和调参的测试集上做一次评估。除了总准确率，我们还打印了每个类别的精确率、召回率和 F1，以及混淆矩阵，目的是看清模型到底错在哪些数字上：

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 恢复验证集上最好的一版权重，而不是用最后一轮可能过拟合的权重
model.load_state_dict(best_state)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# 总准确率，和比赛的评价标准完全一致
acc = accuracy_score(all_labels, all_preds)
print(f'测试集准确率：{acc:.4f}')

# 每个数字单独的精确率、召回率、F1
print(classification_report(all_labels, all_preds, digits=4))

# 混淆矩阵：第 i 行第 j 列表示「真实是 i、被预测成 j」的数量
cm = confusion_matrix(all_labels, all_preds)
print(cm)
```

在我们这次运行中，验证集准确率稳定在 0.98 上下，测试集准确率也落在同一水平，说明模型没有明显的过拟合，验证集的选择是可信的。这个成绩在 MNIST 上属于正常水平：一个结构简单、没有做数据增强的卷积网络，通常就能达到 98% 以上，想再往上冲就要靠更深的网络、数据增强和集成。

混淆矩阵透露的信息比一个总分更有价值。错误主要集中在几组形状相近的数字上：`4` 和 `9` 容易互相混淆，因为两者都有一条竖线和一个封闭或半封闭的圈；`3`、`5`、`8` 之间也有少量互错，尤其是写得潦草时上半部分的弧线很像；`7` 偶尔会被认成 `1`，因为某些写法去掉了横杠后就是一条斜线。相反，`0` 和 `1` 的召回率通常最高，因为它们的形状和其他数字区分度大。这些错例提醒我们：如果还想继续提分，针对性地做轻微旋转、平移、缩放的数据增强，比单纯加深网络更有可能帮到这几组易混类别。

最后我们把预测结果整理成提交文件。不同平台的提交格式可能不同，核心是「样本编号 + 预测标签」两列，编号顺序要和测试集保持一致：

```python
import pandas as pd

# 用测试集在文件中的原始顺序生成编号，保证和平台要求一一对应
submission = pd.DataFrame({
    'Id': range(len(all_preds)),
    'Label': all_preds,
})
submission.to_csv('submission.csv', index=False)

# 提交前做一次最基本的自检：行数是否正确、有没有空值
print(submission.shape)
print(submission.isnull().sum())
```

提交之前我们养成了一个习惯：一定检查行数是否等于测试集大小、列名是否和样例文件一致、标签有没有越界。这些检查花不了几秒钟，却能避免因为格式问题白白丢分。

---

## 三、总结

本文中我们完成了 The AI Core MNIST 手写数字分类的完整流程。我们首先弄清了 IDX 二进制文件的格式，用大端序解析出 60000 张训练图和 10000 张测试图，并通过像素范围、标签分布和可视化确认了数据特点。其次我们做了最少量的数据清洗：补通道维、把像素归一化到 0 到 1、把标签转成 `long` 类型，并划出 6000 张验证集用于调参。接着我们搭建了一个两层卷积加两层全连接的 CNN，用交叉熵损失和 Adam 优化器训练，配合 Dropout 和早停抑制过拟合，并对学习率、批大小、Dropout 做了小范围对比。最后我们在测试集上评估，准确率稳定在 0.98 上下，并通过混淆矩阵发现错误主要集中在 `4/9`、`3/5/8`、`7/1` 这些形状相近的组合。

这个比赛给我们的最大启发是：**先把数据格式和评估流程弄对，再谈模型有多复杂**。MNIST 本身并不难，真正容易出问题的地方是文件解析的字节序、张量的形状、训练集与测试集的隔离，以及提交格式的细节。把这些基础工作做扎实，一个结构朴素的卷积网络就能拿到很稳的结果；反过来，如果数据读错了或者验证集泄漏了，再深的网络也只会给出好看却不可信的分数。
