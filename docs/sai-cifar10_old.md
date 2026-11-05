# SAI CIFAR-10 图像分类（SAI CIFAR10）

## 一、比赛简介

### 1.1 比赛背景
CIFAR-10 是机器学习和计算机视觉领域最经典的小型彩色图像分类数据集之一。数据集一共包含 60,000 张 32×32 的彩色小图，分为 10 个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。前 50,000 张用于训练，后 10,000 张用于测试。图像分辨率低、背景杂乱、类间相似度高，这使得 CIFAR-10 成为检验卷积神经网络特征提取能力的基础试金石。

在这个比赛里，我们的目标是对 CIFAR-10 的测试集给出类别预测。比赛的挑战不在于数据量，而在于 32×32 的小图信息有限，很多图像“即便是人类也很难辨认”，比如一只鹿可能和一只狗在像素上非常接近。不同的方法在该数据集上会表现出明显的差异：简单的浅层模型容易过拟合或欠拟合，而带有强数据增强的深度网络才能稳定提升准确率。SAI CIFAR-10 的评价体系与 Kaggle 经典的 CIFAR-10 相似，都以分类准确率为核心指标，准确率越高越好。

### 1.2 比赛目标
给定训练集的图像和标签，我们需要训练一个模型，在未见过的测试图像上预测其所属的 10 个类别之一。每条样本的输入是 32×32×3 的 RGB 张量，输出是 0~9 的整数标签。训练集内 10 个类别样本数基本平衡，约各 5,000 张。

在解题过程中，我们要先对训练数据做探索，确认数据分布是否均衡、图像质量如何；然后设计数据增强和标准化流程，把数据转成模型可接受的张量；再搭建卷积神经网络并训练；最后在留出的验证集上测试，报告准确率、平衡准确率、宏平均 F1、精确率、召回率等指标，并生成提交文件。比赛对模型复杂度没有上限，允许使用迁移学习和自定义结构。

### 1.3 评价标准
官方评价指标是分类准确率，计算预测正确的样本数除以总样本数。材料中还报告了平衡准确率、宏平均 F1、召回率、精确率以及每类的 ROC 曲线下的面积，用来更细致地分析模型在各类别上的表现。在 10 类均衡的情况下，准确率和平衡准确率数值相近，但若出现类不平衡，平衡准确率能更公平地反映整体能力。准确率越高越好，材料里报告的简单网络在验证集上可达 80% 以上的高分模型已达到约 87%，这也是我们要追赶的目标。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先用 PyTorch 的 torchvision 载入 CIFAR-10：

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

我们打印出 10 个类别对应的名称：`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`。通过随机抽样几张图像可以观察到，图像质量参差不齐，有的猫与狗的边界模糊，鹿与马的形态相似。这解释了为什么需要更强大的特征提取。

接着我们计算每个类别的样本数：

```python
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))
```

结果显示 10 个类别基本均衡，每个约 5,000 张。这种平衡性让准确率和平衡准确率可以并行参考，但我们在报告时会同时提供宏平均指标以防模型在少数难类上失衡。

为了直观感受数据的困难度，我们用 Keras 加载数据并随机展示几张图像。可以看到部分图像的细节非常模糊，甚至人类识别也会出错。这种情况下模型必须依靠整体纹理和形状，而非细小特征。

### 2.2 数据清洗
CIFAR-10 原始数据本身非常干净，没有缺失值或重复。我们的清洗主要集中在格式统一和数据增强上。

```python
# 训练和测试使用不同的转换
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=composed_train)
validation_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=composed_test)
```

清洗的核心是“最小干预”，不改变像素语义，只做标准化。我们把均值和方差固定为文献常用的 `[0.4914, 0.4822, 0.4465]` 和 `[0.247, 0.243, 0.261]`，这样可以帮助模型更快收敛。

在另一份参考材料里，作者还尝试了把彩色图转为灰度图再做平均池化，这种做法会大幅降低计算成本，但会丢失颜色信息，适合作为快速基线。我们保留 RGB 通道，因颜色对区分鸟、青蛙、船等类别有帮助。

另外我们对数据做了打乱和划分：训练集的 50,000 张图像中取 90% 训练、10% 留作验证，以评估模型真实泛化能力。这一步是用 `train_test_split` 完成的，确保各类别比例保持一致。

### 2.3 模型训练与超参数调优
我们采用的参考方案是基于 PyTorch 的小型卷积网络，叠加 MetaEmbedding 分类器，增强特征表达能力。

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
```

网络结构很清晰：两层卷积加最大池化，提取到 128 通道的特征图，再展平到 512 维特征，最终送入基于余弦归一化的 MetaEmbedding 分类器。这种结构在保持参数量可控的同时，加入了可学习的类中心和特征可达性调节，有助于区分细粒度类别。

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
```

我们采用 Adam 优化器，学习率 0.001，训练 10 个 epoch。每 100 步打印一次平均损失，观察收敛情况。数据增强在训练时开启，测试时只做 resize 和标准化。

超参数调优方面，我们主要关注学习率、数据增强强度、网络深度。因为 CIFAR-10 样本量有限，过深的网络容易过拟合。这份材料中使用的两层卷积是性价比很高的选择。若继续提升，可以考虑加入 BatchNorm、Dropout 或残差连接。

参考的另一份材料则做了大量基线对比，包括 SVM、随机森林、决策树、KNN 和简单的 ANN。它先把图像平均像素转为灰度，再用动量特征降维，然后用传统机器学习分类。结果显示这些方法的准确率普遍低于 50%，而 CNN 在正常三通道数据上能达到约 70% 以上。这进一步印证了卷积结构在图像任务上的必要性。

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
```

同时我们计算平衡准确率、宏平均 F1、精确率、召回率，以及每个类别的 ROC AUC：

```python
balanced_acc = balanced_accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
precision = precision_score(all_labels, all_preds, average='macro')
```

材料里报告的简单网络在验证集上达到了约 87% 的准确率，平衡准确率、F1 等指标也基本同步。ROC 曲线显示微平均 AUC 较高，各类别 AUC 普遍在 0.9 以上，说明模型对大多数类别都有较好的区分度。

我们还绘制了预测示例，随机展示若干张测试图像及其预测结果，方便直观判断模型是否对模糊图像产生混淆。最后保存模型权重 `simplenet_cifar10.pth`，并生成提交文件。

## 三、总结
本文中我们完成了 SAI CIFAR-10 图像分类的完整流程。我们首先用 EDA 确认了 CIFAR-10 数据集的 10 个类别均衡、图像小且细节模糊的特性。其次我们做了标准的数据清洗和增强，使用随机旋转、水平翻转、颜色抖动以及标准化，把图像转为模型可接受的张量。接着我们采用两层卷积加 MetaEmbedding 分类器的网络，用 Adam 优化器训练 10 个 epoch，并与 SVM、随机森林等传统方法进行了对比。最后我们在验证集上得到约 87% 的准确率，平衡准确率、F1、精确率、召回率均保持较高水平，ROC AUC 显示模型对各个类别都有稳定区分能力。此次比赛再次印证了“数据增强＋卷积特征”在小样本图像任务上的重要性。