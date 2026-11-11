# TechX 人脸表情识别（Facial Expression Recognition）

## 一、比赛简介

### 1.1 比赛背景

人脸表情识别（Facial Expression Recognition，简称 FER）是计算机视觉里一个经典又实用的任务：给一张人脸照片，让机器判断这个人现在是生气、厌恶、害怕、开心、难过、惊讶，还是面无表情。它在人机交互、在线教育、心理健康筛查、智能客服等场景都有应用。

这个比赛的数据来自公开的人脸表情数据集，图片已经被整理成「像素字符串」的形式放在 CSV 里，而不是一张张图片文件。每张图片是 48×48 的灰度图，标签是 0 到 6 的整数，分别对应七种表情。比赛要求参赛者根据这些像素值训练模型，预测测试集中每张人脸的表情类别。

这个任务的难点有三个。第一，**表情本身有歧义**：同一张脸，不同人看可能给出不同标签，模型很难学到完全干净的边界。第二，**图片分辨率很低**：48×48 的灰度图丢失了很多细节，眼角、嘴角的细微变化可能只剩几个像素。第三，**类别不平衡**：开心的样本通常最多，厌恶的样本通常最少，模型容易偏向多数类。因此，除了把数据喂进网络，我们还要在数据增强、迁移学习和评估方式上多花心思。

### 1.2 比赛目标

输入是三张 CSV：`train.csv`、`val.csv`、`test.csv`。训练集和验证集包含两列：

- `emotion`：表情标签，取值 0 到 6；
- `pixels`：一串用空格分隔的整数，表示 48×48 灰度图的 2304 个像素值。

测试集只有图片像素，没有标签。我们要为测试集每一行预测一个 0 到 6 的类别，并按提交样例的格式写回。

七个类别按顺序对应：

```
0 Anger（生气）  1 Disgust（厌恶）  2 Fear（害怕）  3 Happy（开心）
4 Sad（难过）    5 Surprise（惊讶） 6 Neutral（中性）
```

这个编号是固定的，提交时不能自行映射，否则会出现类别错位。我们在读取数据时要保持这个顺序，并在训练时把标签直接当作整数喂给 CrossEntropyLoss。

### 1.3 评价标准

比赛使用**分类准确率（Accuracy）**作为评价指标，也就是预测正确的图片数占全部测试图片数的比例。七分类的随机猜测准确率约为 1/7 ≈ 14.3%，所以任何明显高于 14% 的模型都说明学到了东西。

准确率越高越好。它直观、容易解释，但也有一个缺点：如果某一类样本特别多，模型只要偏向这一类就能拿到不错的分数。因此我们在训练时会同时观察验证集上每一类的表现，而不是只看一个总数。

---

## 二、解题过程

我们按 EDA、数据清洗、模型训练与超参数调优、模型测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先读取训练集，看看数据长什么样：

```python
import pandas as pd
import numpy as np

TRAIN_CSV = '/kaggle/input/competitions/techx-facial-expression-recognition/train.csv'
df = pd.read_csv(TRAIN_CSV)

# 列名可能带空格或大小写不一致，先统一清理
df.columns = df.columns.str.strip().str.lower()

print(df.shape)
print(df.columns.tolist())
print(df.head(3))
```

输出里能看到 `emotion` 和 `pixels` 两列。`pixels` 是字符串，不是数字矩阵，所以 pandas 读进来时只能当作文本处理。

接着我们统计七个类别的样本数：

```python
# 每个类别有多少张图片
counts = df['emotion'].value_counts().sort_index()
print(counts)

# 用占比看清不平衡程度
print((counts / counts.sum()).round(4))
```

从统计结果能看到，不同表情的样本量并不相等。`Happy` 往往最多，`Disgust` 往往最少，这是 FER 数据集的常见现象。类别不平衡会影响模型：它会更倾向于预测多数类，少数类的召回率很低。我们在后面的训练里会用数据增强和类别权重来缓解。

然后我们把一行像素还原成图片，确认 48×48 的形状：

```python
from PIL import Image

# 取第一行的像素字符串
pixel_string = df['pixels'].iloc[0]

# 按空格切开，转成 uint8，再开方求边长
pixel_list = np.fromstring(pixel_string, dtype=np.uint8, sep=' ')
img_size = int(np.sqrt(len(pixel_list)))
print(len(pixel_list), img_size)

# reshape 成二维灰度图，并显示出来
image_array = pixel_list.reshape((img_size, img_size))
Image.fromarray(image_array).show()
```

2304 = 48 × 48，说明每张图确实是 48×48 的灰度图，只有一个通道。这个信息决定了后面的预处理：如果直接把 2304 个像素拉平喂给全连接层，会丢失空间结构；更好的做法是把它当成图片，用卷积神经网络。

### 2.2 数据清洗

这里的「清洗」主要不是去重或补缺失，而是**把 CSV 里的字符串还原成模型能吃的图片张量**，并做统一尺寸和归一化。

我们写一个自定义 Dataset：

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FacialExpressionDataset(Dataset):
    def __init__(self, csv_file, is_test=False, transform=None):
        self.df = pd.read_csv(csv_file)
        # 列名统一去空格、转小写，避免 'Pixels' 和 'pixels' 不一致
        self.df.columns = self.df.columns.str.strip().str.lower()
        self.is_test = is_test
        self.transform = transform

        # 自动找到像素列，兼容不同版本的列名
        pixel_cols = [c for c in self.df.columns if 'pixel' in c]
        self.pixel_col = pixel_cols[0] if pixel_cols else self.df.columns[1]

        # 训练/验证集还要找标签列
        if not self.is_test:
            label_cols = [c for c in self.df.columns if c in ['label', 'emotion', 'class']]
            self.label_col = label_cols[0] if label_cols else self.df.columns[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 把字符串像素还原成 48x48 的灰度图
        pixel_string = self.df[self.pixel_col].iloc[idx]
        pixel_list = np.fromstring(pixel_string, dtype=np.uint8, sep=' ')
        side = int(np.sqrt(len(pixel_list)))
        image_array = pixel_list.reshape((side, side))

        # 先转灰度，再转 RGB：因为后面要用 ImageNet 预训练模型，它要求三通道
        image = Image.fromarray(image_array).convert('L').convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, 0
        return image, int(self.df[self.label_col].iloc[idx])
```

这段代码有两个关键点。第一，`np.fromstring(..., sep=' ')` 把空格分隔的数字串一次性解析成数组，比逐字符处理快得多。第二，`.convert('L').convert('RGB')` 先转灰度再转 RGB：预训练的 EfficientNet 是在 RGB 三通道 ImageNet 上训练的，直接喂单通道会维度不匹配；而先转灰度再复制成三通道，既保留了「原图是灰度」的事实，又满足模型输入要求。

接着定义训练和验证的变换：

```python
# 训练时的增强：随机翻转、轻微旋转，模拟拍摄角度和镜像差异
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),          # EfficientNet 的标准输入尺寸
    transforms.RandomHorizontalFlip(),       # 人脸左右翻转不改变表情
    transforms.RandomRotation(15),           # 允许 ±15 度的小幅旋转
    transforms.ToTensor(),                   # 转成 [0,1] 的张量
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 均值方差
])

# 验证/测试只做确定性的缩放和归一化，不能随机增强
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

为什么用 ImageNet 的均值和方差？因为我们要加载在 ImageNet 上预训练的权重。预训练时输入是按这组均值方差归一化的，如果推理时换了另一组参数，特征分布就偏了，迁移学习的效果会打折扣。

### 2.3 模型训练与超参数调优

我们采用**迁移学习**：用 ImageNet 上预训练好的 EfficientNet-B0 当特征提取器，只把最后的分类头换成七分类。

```python
import torch.nn as nn
import torch.optim as optim
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练权重；weights='DEFAULT' 会自动下载 ImageNet 权重
model = models.efficientnet_b0(weights='DEFAULT')

# 找到分类头最后一层的输入维度
in_features = model.classifier[1].in_features

# 替换分类头：先 Dropout 再线性层，输出 7 类
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 7)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

这里有几个设计选择值得解释。**Dropout 0.4**：分类头参数量不大，但容易过拟合，尤其在数据量有限时；Dropout 随机丢弃 40% 的神经元，相当于每次训练一个不同的子网络，能提升泛化。**AdamW**：它在 Adam 的基础上把权重衰减和梯度更新解耦，是训练 Transformer 和现代卷积网络的常用优化器。**学习率 1e-3**：因为分类头是随机初始化的，需要比微调主干更大的学习率；但我们只训练 5 个 epoch，过大的学习率会震荡。

训练循环：

```python
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # 清空上一轮梯度
        outputs = model(images)        # 前向传播
        loss = criterion(outputs, labels)
        loss.backward()                # 反向传播
        optimizer.step()               # 更新参数

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100. * correct / total

    # 每个 epoch 结束后切到评估模式，在验证集上算准确率
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}: train acc {train_acc:.2f}%, "
          f"val acc {100.*val_correct/val_total:.2f}%")
```

关于超参数调优，材料里的 baseline 只训练了 **5 个 epoch**，这是一个很保守的设置：预训练模型已经很强，5 个 epoch 足以让新分类头收敛，同时避免在训练集上过拟合。我们在这 5 个 epoch 里主要观察验证准确率的走势：如果验证准确率在第 3 个 epoch 后不再上升，说明学习率或 epoch 数需要调整；如果训练准确率远高于验证准确率，说明需要更强的数据增强或更大的 Dropout。

我们也可以尝试解冻主干网络的后几层做微调，但要注意学习率要调小（比如 1e-4），否则预训练学到的边缘、纹理特征会被破坏。对入门者来说，先冻结主干、只训练分类头，是性价比最高的做法。

### 2.4 模型测试与指标报告

训练结束后，我们切到评估模式，对测试集逐批预测：

```python
model.eval()
all_preds = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())

# 用提交样例做模板，保持列名和顺序一致
submission = pd.read_csv(SAMPLE_SUB)
submission.iloc[:, 1] = all_preds
submission.to_csv('submission.csv', index=False)
```

这里用 `sample_submission.csv` 做模板，而不是自己新建 DataFrame，是为了保证列名、行序和官方要求完全一致。很多比赛提交失败，不是模型的问题，而是列名拼错或行序对不上。

指标报告方面，我们在验证集上得到的是**逐 epoch 的准确率**。这个数字的绝对值受数据难度限制：48×48 的灰度图、七类表情、类别不平衡，都会让准确率停在某个上限附近。我们更关心的是「训练准确率和验证准确率的差距」：差距小说明泛化好，差距大说明过拟合。材料里没有给出具体分数，所以我们不在这里编造排行榜数字，只报告评估流程。

如果想进一步提升，可以从三个方向入手：

1. **更强的数据增强**：随机裁剪、颜色抖动、Cutout；
2. **类别权重**：给 `Disgust` 这类少数类更高的损失权重；
3. **模型集成**：用 EfficientNet-B0、ResNet、DenseNet 多个模型投票。

但在 baseline 阶段，我们先把「字符串像素 → 图片张量 → 预训练模型 → 提交文件」这条链路走通，再逐步加复杂度。

---

## 三、总结

本文中我们完成了 TechX 人脸表情识别比赛的完整流程。我们首先用 EDA 确认了数据是 48×48 灰度图的像素字符串、七类表情且类别不平衡；其次写了自定义 Dataset，把空格分隔的像素还原成图片，并统一转成 224×224 RGB、做 ImageNet 归一化和训练增强；接着用 ImageNet 预训练的 EfficientNet-B0，替换分类头为 Dropout + 七类线性层，用 AdamW 训练 5 个 epoch；最后在验证集上逐 epoch 报告准确率，并对测试集预测生成提交文件。

这个比赛给我们的最大启发是：**迁移学习能大幅降低小数据集上的门槛。** 48×48 的灰度人脸本身信息有限，如果从零训练卷积网络，很容易过拟合；而借用在 ImageNet 上学到的通用视觉特征，再只训练一个很小的分类头，就能在有限数据上取得不错的效果。对入门者来说，理解「预训练权重 + 替换分类头 + 冻结/微调」这套流程，比记住某个具体网络结构更重要。