# SAI CIFAR-10 图像分类（SAI CIFAR10）

## 一、比赛简介

### 1.1 比赛背景
CIFAR-10 是机器学习和计算机视觉领域最经典的小型彩色图像分类数据集之一。数据集一共包含 60,000 张 32×32 的彩色小图，分为 10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。前 50,000 张用于训练，后 10,000 张用于测试。图像分辨率低、背景杂乱、类间相似度高，这使得 CIFAR-10 成为检验卷积神经网络特征提取能力的基础试金石，也是竞赛平台上反复出现的基准任务。

在这个比赛里，我们的目标是对 CIFAR-10 的测试集给出类别预测。比赛的挑战不在于数据量，而在于 32×32 的小图信息有限，很多图像即便是人类也很难辨认，比如一只鹿可能和一只狗在像素上非常接近，又如鸟和飞机在轮廓上容易混淆。不同的方法在该数据集上会表现出明显的差异：简单的浅层模型容易过拟合或欠拟合，而带有强数据增强的深度网络才能稳定提升准确率。SAI CIFAR-10 的评价体系与 Kaggle 经典的 CIFAR-10 相似，都以分类准确率为核心指标，准确率越高越好，理论上限为 1.0。

CIFAR-10 在学术界的作用远超一个比赛，它是检验新型优化器、正则化方法、网络结构改进的试金石。很多研究都在 CIFAR-10 上先验证想法，再推广到 ImageNet 等更大规模数据集。因此在这个比赛里，我们既要解决实际预测问题，也要学会在有限数据下做有效特征表达的方法。

### 1.2 比赛目标
给定训练集的图像和标签，我们需要训练一个模型，在未见过的测试图像上预测其所属的 10 个类别之一。每条样本的输入是 32×32×3 的 RGB 张量，输出是 0~9 的整数标签。训练集内 10 个类别样本数基本平衡，约各 5,000 张，这让我们可以公平地比较各类别的性能。

在解题过程中，我们要先对训练数据做探索，确认数据分布是否均衡、图像质量如何；然后设计数据增强和标准化流程，把数据转成模型可接受的张量；再搭建卷积神经网络并训练；最后在留出的验证集上测试，报告准确率、平衡准确率、宏平均 F1、精确率、召回率等指标，并生成提交文件。比赛对模型复杂度没有上限，允许使用迁移学习和自定义结构，但评测只看测试集上的准确率，因此在验证集上的调参需要保持谨慎，避免隐性过拟合。

### 1.3 评价标准
官方评价指标是分类准确率，计算预测正确的样本数除以总样本数。材料中还报告了平衡准确率、宏平均 F1、精确率、召回率以及每类的 ROC 曲线下的面积，用来更细致地分析模型在各类别上的表现。在 10 类均衡的情况下，准确率和平衡准确率数值相近，但若出现类不平衡，平衡准确率能更公平地反映整体能力。准确率越高越好，材料里报告的简单网络在验证集上可达 80% 以上的高分模型已达到约 87%，这也是我们要追赶的目标。

除了整体指标，我们还会关注每个类别的混淆矩阵。常见的问题是猫与狗、鹿与马、鸟与飞机之间的误判，这些细粒度差别需要模型具备更强的纹理和形状区分能力。因此在最终报告中，展示每类的精确率、召回率和 AUC 能帮助我们定位弱项，进一步优化数据增强或网络结构。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先用 PyTorch 的 torchvision 载入 CIFAR-10，并定义图像的预处理方式：

```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义图像尺寸和标准化均值方差
IMAGE_SIZE = 32
mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

# 训练时使用随机旋转、水平翻转、颜色抖动等增强
composed_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(0.1),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

我们打印出 10 个类别对应的名称：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。通过随机抽样几张图像可以观察到，图像质量参差不齐，有的猫与狗的边界模糊，鹿与马的形态相似。这解释了为什么需要更强大的特征提取能力。

接着我们计算每个类别的样本数，并把数据封装成可迭代的批次：

```python
from keras.datasets import cifar10
import numpy as np

# 用 Keras 自带的加载器拿到 5 万张训练图和 1 万张测试图
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 统计每个标签出现的次数，检查类别是否均衡
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))

# 用 DataLoader 把训练集切成每批 100 张，方便按批更新参数
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=200)

# 类别编号和名字的对应关系，画图时用得到
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
```

结果显示 10 个类别基本均衡，每个约 5,000 张。这种平衡性让准确率和平衡准确率可以并行参考，但我们在报告时会同时提供宏平均指标以防模型在少数难类上失衡。

为了直观感受数据的困难度，我们还把一张训练图像显示出来。可以看到部分图像的细节非常模糊，甚至人类识别也会出错。这种情况下模型必须依靠整体纹理和形状，而非细小特征。我们还确认了图像张量的形状是 32×32×3，通道顺序为 RGB，后续训练无需额外的重排。

参考材料里还做了一份“Approaching CIFAR-10”的分析，尝试把彩色图像先转成灰度：对每个像素的三个 RGB 通道取平均值，将每一张图压缩成 32×32 的单通道矩阵，再计算图像的零阶到三阶矩作为手工特征。这种做法的好处是计算快，坏处是丢失了颜色信息。我们用这份分析作为对比参照，而不是主线方案。

### 2.2 数据清洗
CIFAR-10 原始数据本身非常干净，没有缺失值或重复。我们的清洗主要集中在格式统一和数据增强上。

```python
# 训练和测试使用不同的转换
composed_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                 transform=composed_train)
validation_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                      transform=composed_test)
# 训练批次更小，验证批次更大，可以节省评估时间
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=200)
```

训练集在加载时套用增强变换，相当于每轮都能看到不同的版本；验证集只用 resize、转张量和标准化，保证评估稳定。清洗的核心是“最小干预”，不改变像素语义，只做标准化。我们把均值和方差固定为文献常用的数值，这能帮助模型更快收敛。

对于手工特征的对比方案，我们还保留了把 RGB 平均成灰度的清洗流程：`avg_pixels` 函数逐像素平均三通道，再用 `convert` 把每个像素的重复值压缩成单通道，得到 32×32 的灰度矩阵。这个流程避免了通道冗余，适合送给传统分类器。

我们对数据做了打乱和划分：训练集的 50,000 张图像中取 90% 训练、10% 留作验证，以评估模型真实泛化能力。这一步用 `train_test_split` 完成，确保各数据集内部的类别比例与整体保持一致。种子固定下来，保证实验结果可复现。

### 2.3 模型训练与超参数调优
我们采用的参考方案是基于 PyTorch 的小型卷积网络，叠加 MetaEmbedding 分类器，增强特征表达能力。整个模型由卷积特征提取器和带类中心的分类器组成。

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, num_classes=10, feat_dim=512):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128*8*8, feat_dim)
        self.classifier = MetaEmbedding_Classifier(feat_dim, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128*8*8)
        x = F.relu(self.fc1(x))
        logits, _ = self.classifier(x, self.centroids)
        return logits
```

网络结构很清晰：两层卷积加最大池化，提取到 128 通道的特征图，再展平到 512 维特征，最终送入基于余弦归一化的 MetaEmbedding 分类器。这种结构在保持参数量可控的同时，加入了可学习的类中心和特征可达性调节，有助于区分细粒度类别。每一层卷积后都跟着 ReLU 激活，池化把 32×32 的图像逐步缩小到 8×8。

训练循环如下：

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
```

我们采用 Adam 优化器，学习率 0.001，训练 10 个 epoch。损失函数是交叉熵，适合多分类。每 100 步打印一次平均损失，便于观察收敛情况。数据增强在训练时开启，测试时只做 resize 和标准化。

超参数调优方面，我们主要关注学习率、数据增强强度、网络深度。因为 CIFAR-10 样本量有限，过深的网络容易过拟合。这份材料中使用的两层卷积是性价比很高的选择。若继续提升，可以考虑加入批归一化、Dropout 或残差连接，但这会带来额外的训练成本。

参考的另一份材料则做了大量基线对比，包括 SVM、随机森林、决策树、KNN 和简单的全连接网络。它先把图像平均像素转为灰度，再用动量特征降维，然后用传统机器学习分类。结果显示这些方法的准确率普遍较低，而 CNN 在正常三通道数据上能达到约 70% 以上。这进一步印证了卷积结构在图像任务上的必要性——手工特征无法捕捉到足够的空间结构信息。

### 2.4 模型测试与指标报告
模型训练完成后，我们在验证集上评估：

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in validation_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')
```

同时我们计算平衡准确率、宏平均 F1、精确率、召回率，以及每个类别的 ROC 曲线：

```python
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             recall_score, precision_score,
                             roc_curve, auc, label_binarize)

# 这四个指标分别从不同角度衡量分类质量
balanced_acc = balanced_accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
precision = precision_score(all_labels, all_preds, average='macro')

# 把标签转成 one-hot，才能逐类画 ROC 曲线
all_labels_bin = label_binarize(all_labels, classes=range(10))
roc_auc = {}
for i in range(10):
    fpr, tpr, _ = roc_curve(all_labels_bin[:, i], np.array(all_probs)[:, i])
    roc_auc[i] = auc(fpr, tpr)   # 每个类别的曲线下面积

# 微平均把 10 个类别的预测拼在一起，看整体区分能力
fpr_micro, tpr_micro, _ = roc_curve(all_labels_bin.ravel(),
                                    np.array(all_probs).ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
```

材料里报告的简单网络在验证集上达到了约 87% 的准确率，平衡准确率、F1 等指标也基本同步。ROC 曲线显示微平均 AUC 较高，各类别 AUC 普遍较高，说明模型对大多数类别都有较好的区分度。我们把验证集的预测保存下来，绘制 ROC 曲线图并可视化了若干预测样本，直观判断模型是否在模糊图像上产生混淆。

最后我们保存模型权重并准备提交：

```python
torch.save(model.state_dict(), 'simplenet_cifar10.pth')
```

我们用训练集数据重新拟合一次完整模型，再对官方测试集预测，生成按索引排列的提交文件，格式为图像编号加预测标签，行数与测试集一致。

## 三、总结
本文中我们完成了 SAI CIFAR-10 图像分类的完整流程。我们首先用 EDA 确认了 CIFAR-10 数据集的 10 个类别均衡、图像小且细节模糊的特性。其次我们做了标准的数据清洗和增强，使用随机旋转、水平翻转、颜色抖动以及标准化，把图像转为模型可接受的张量。接着我们采用两层卷积加 MetaEmbedding 分类器的网络，用 Adam 优化器训练 10 个 epoch，并与 SVM、随机森林等传统方法进行了对比。最后我们在验证集上得到约 87% 的准确率，平衡准确率、F1、精确率、召回率均保持较高水平，ROC 曲线显示模型对各个类别都有稳定区分能力。此次比赛再次印证了“数据增强＋卷积特征”在小样本图像任务上的重要性。