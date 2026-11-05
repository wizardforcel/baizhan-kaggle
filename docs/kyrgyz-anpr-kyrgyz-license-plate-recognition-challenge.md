# Kyrgyz ANPR 吉尔吉斯车牌识别挑战

## 一、比赛简介

### 1.1 比赛背景

车牌识别（ANPR，Automatic Number Plate Recognition）是计算机视觉里非常实用的一类任务。停车场自动抬杆、交通违章抓拍、高速公路收费，背后都靠它。它的做法通常分两步：第一步先用目标检测把照片里的车牌「框」出来并裁剪成小图；第二步再对小图做光学字符识别（OCR，Optical Character Recognition），把图上的字符读成文字。

这个比赛跳过第一步，直接给大家已经裁剪好的车牌小图，只考第二步：把车牌上的文字读出来。数据来自吉尔吉斯斯坦（Kyrgyzstan）的车辆牌照。吉尔吉斯车牌一般由若干数字、拉丁字母，以及表示国家的标记 `KG` 组成。

难点主要有三个：

第一，车牌上的字符是**变长的**。有的车牌 8 个字符，有的 11 个，模型不能像普通图片分类那样固定输出一个类别，而要输出一串长度不定的文字。

第二，**评价非常严格**。车牌识别里，整串文字只要错一个字符，整张图就算错。这种「全对才算对」的规则叫做精确匹配（exact match），它让任务比「字符准确率」难得多。

第三，**图像质量参差不齐**。裁剪出来的车牌尺寸、亮度、清晰度都不完全一样，需要先统一尺寸才能放进同一个批次训练。

### 1.2 比赛目标

输入是一张裁剪好的车牌图片，输出是这张车牌上的完整文字序列。训练集 `train.csv` 有两列：`path` 是图片相对路径，`label` 是车牌文字。测试集只提供图片，需要我们自己生成每张图的预测文字。

### 1.3 评价标准

比赛指标是**精确匹配准确率**：预测文字和真实文字完全一致的图片张数，除以总张数。越高越好，满分是 1.0。因为要求整串一致，所以只要有一个字符认错，这张图就没有分。

## 二、解题过程

我们的整体思路是搭一个经典的 OCR 流水线：先把图片统一成固定大小，再用卷积网络提取图像特征，接着用循环网络把特征按「从左到右」的顺序串起来，最后用 CTC 损失让模型学会在没有逐字标注的情况下对齐字符。下面按四步展开。

### 2.1 数据探索（EDA）

我们首先把训练表读进来，看看数据规模和前几行长什么样：

```python
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# 比赛数据放在这个目录下
DATA_PATH = Path("/kaggle/input/competitions/kyrgyz-anpr-kyrgyz-license-plate-recognition-challenge")

# 读训练表：两列，path 是图片路径，label 是车牌文字
train = pd.read_csv(DATA_PATH / "train.csv")

print("train shape:", train.shape)   # 样本总数和列数
print(train.head())
print(train.columns)
print(train["label"].head())
```

打印结果告诉我们，训练表只有图片路径和 OCR 目标两列，这是一道标准的「图片到文本序列」任务。

接着我们统计标签的长度和字符种类，这决定了后面模型的输出类别数：

```python
labels = train["label"].astype(str)

print("number of samples:", len(labels))        # 训练样本总数
print("min label length:", labels.str.len().min())   # 最短车牌
print("max label length:", labels.str.len().max())   # 最长车牌
print("mean label length:", labels.str.len().mean()) # 平均长度

# 把所有标签拼起来，去重后得到字符表
characters = sorted(set("".join(labels)))
print("number of unique characters:", len(characters))
print("characters:", characters)
```

我们发现车牌长度是变化的，字符表也很小，只有数字、字母和几个标记符号。**长度可变 + 字符表小**，正好是 CTC（Connectionist Temporal Classification）最擅长的场景：模型只要输出一串逐帧的字符概率，不需要事先知道每个字符在图片里的精确位置。

我们再看一下长度的分布：

```python
train["label_length"] = labels.str.len()
train["label_length"].value_counts().sort_index()
```

分布显示大部分车牌集中在 9～11 个字符，也有少量更长的。这个信息后面做训练/验证集划分时会用到。

然后我们抽 12 张图可视化，确认图片内容和标签对得上：

```python
sample_df = train.sample(12, random_state=42)   # 固定随机种子，方便复现

plt.figure(figsize=(16, 8))
for i, row in enumerate(sample_df.itertuples(), 1):
    image = Image.open(DATA_PATH / row.path).convert("RGB")
    plt.subplot(3, 4, i)
    plt.imshow(image)
    plt.title(row.label, fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.show()
```

从图上我们看到，车牌都是横向的长条形，字符从左到右排列。这提示我们输入图片应该做成「宽而扁」的形状。

最后我们抽样 1000 张图，统计原始尺寸：

```python
from collections import Counter

sample_paths = train["path"].sample(1000, random_state=42)
sizes = [Image.open(DATA_PATH / p).size for p in sample_paths]
size_counts = Counter(sizes)

print("number of unique sizes in sample:", len(size_counts))
for size, count in size_counts.most_common(10):
    print(size, count)
```

结果表明图片尺寸并不统一。所以训练前必须统一缩放，否则一个批次里的图片没法拼成一个张量。

### 2.2 数据清洗

这份数据的「清洗」主要不是去脏数据，而是**做统一的预处理**，让模型能稳定训练。

第一步，确定输入尺寸。因为车牌是横向的，我们把图片统一缩放到宽 256、高 64：

```python
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 64
```

第二步，构建字符词表。CTC 需要一个特殊的「空白符」（blank），用来表示「这一帧没有字符」。我们把它放在索引 0，真实字符从 1 开始编号：

```python
characters = sorted(set("".join(train["label"].astype(str))))
blank_token = "<BLANK>"

# 字符从 1 开始编号，0 留给 CTC 的 blank
char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# 类别数 = 真实字符数 + 1 个 blank
num_classes = len(char_to_idx) + 1
print("vocabulary size without blank:", len(char_to_idx))
print("CTC classes with blank:", num_classes)
```

第三步，把数据集按标签长度分组后再划分训练集和验证集。这里有个细节：如果直接随机划分，某些罕见长度可能全被分到一边，验证集就不具代表性。所以我们先按长度粗分档，再分层抽样：

```python
from sklearn.model_selection import train_test_split

def length_group(length):
    """把长度粗分成四档，避免分层时档位太多导致样本过少"""
    if length <= 9:
        return "short"
    if length == 10:
        return "medium"
    if length == 11:
        return "long"
    return "extra_long"

split_stratify = train["label_length"].apply(length_group)

train_df, valid_df = train_test_split(
    train,
    test_size=0.15,          # 15% 做验证
    random_state=42,         # 固定种子
    stratify=split_stratify  # 按长度档位分层，保证两边分布接近
)
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
```

第四步，封装 PyTorch 数据集，把「读图 → 缩放 → 灰度 → 转张量 → 归一化」和「标签 → 索引序列」打包在一起：

```python
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LicensePlateDataset(Dataset):
    def __init__(self, dataframe, data_path, char_to_idx, image_width, image_height):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_path = data_path
        self.char_to_idx = char_to_idx
        self.image_width = image_width
        self.image_height = image_height

        # 所有图片走同一套变换，保证输入分布一致
        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),  # 统一尺寸
            transforms.Grayscale(num_output_channels=1),      # 车牌识别用灰度就够
            transforms.ToTensor(),                            # 变成 0~1 的张量
            transforms.Normalize(mean=[0.5], std=[0.5])       # 归一化到 -1~1
        ])

    def encode_label(self, label):
        # 把字符串标签变成索引序列，供 CTC 使用
        return torch.tensor([self.char_to_idx[c] for c in label], dtype=torch.long)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(self.data_path / row["path"]).convert("RGB")
        image = self.transform(image)
        label = str(row["label"])
        return image, self.encode_label(label), label
```

因为每张车牌的标签长度不同，默认的批处理没法直接拼。我们写一个自定义的 `collate_fn`，把标签「首尾相接」拼成一维，同时单独记住每条的原始长度：

```python
def collate_fn(batch):
    images, encoded_labels, raw_labels = zip(*batch)

    images = torch.stack(images)   # 图片形状一致，可以直接堆叠

    # CTC 要求知道每条标签的真实长度
    label_lengths = torch.tensor([len(l) for l in encoded_labels], dtype=torch.long)

    # 所有标签拼成一条长序列，配合 label_lengths 使用
    labels = torch.cat(encoded_labels)

    return images, labels, label_lengths, raw_labels
```

### 2.3 模型训练与超参数调优

我们搭的模型叫 CRNN（Convolutional Recurrent Neural Network），分三段：

1. **CNN 编码器**：一层层卷积把图片变成特征图，空间尺寸不断缩小，但通道数不断变多，相当于把图像「压缩成特征」。
2. **BiLSTM 序列建模**：把特征图按宽度方向看成一串时间步，用双向 LSTM 学习字符之间的上下文关系。
3. **线性分类头**：每个时间步输出「当前是哪个字符」的分数。

```python
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # CNN 部分：用 6 层卷积，逐步提取更抽象的图像特征
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),   # 只压高度，保留宽度方向的时间步

            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        # 双向 LSTM：从左右两个方向看字符序列，理解上下文
        self.rnn = nn.LSTM(
            input_size=512 * 4,   # 特征图高度压到 4，所以通道×高度 = 512×4
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # 每个时间步输出所有字符类别的分数
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        # 把「通道×高度」拉平，宽度方向当作时间序列
        features = features.permute(0, 3, 1, 2)
        batch_size, seq_len, channels, height = features.size()
        features = features.reshape(batch_size, seq_len, channels * height)

        sequence, _ = self.rnn(features)
        logits = self.classifier(sequence)   # 形状：batch × 时间步 × 类别数
        return logits
```

损失函数用 CTC。CTC 的好处是：我们不需要告诉模型「第 1 个字符在第几列像素」，它会在训练中自动找出一种对齐方式。同时它允许输出里出现重复字符和空白符，最后解码时再把重复和空白去掉。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(num_classes=num_classes).to(device)

ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)  # 0 就是我们的 blank
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

训练一个 epoch 的函数如下。注意 CTC 要求输入是「时间步在前」的形状，所以要 `permute`；还要把每张图的输入长度告诉损失函数。另外我们加了梯度裁剪，防止 LSTM 训练时梯度爆炸：

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, labels, label_lengths, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        logits = model(images)
        # CTC 要求形状是 时间步 × batch × 类别数
        log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)

        # 每张图的输入时间步数都等于特征图宽度
        input_lengths = torch.full(
            size=(logits.size(0),), fill_value=logits.size(1),
            dtype=torch.long, device=device
        )

        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        loss.backward()

        # 梯度裁剪，避免训练不稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
```

整个训练跑 5 个 epoch，每个 epoch 记录训练损失和验证指标：

```python
EPOCHS = 5
history = []

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, ctc_loss, device)
    valid_loss, valid_acc = validate(model, valid_loader, ctc_loss, device, idx_to_char)

    history.append({"epoch": epoch, "train_loss": train_loss,
                    "valid_loss": valid_loss, "valid_acc": valid_acc})
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
          f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")
```

### 2.4 模型测试与指标报告

预测时，我们用「贪心解码」：每个时间步取概率最大的类别，然后去掉连续的重复字符和 blank。比如输出 `KK--G-77`（`-` 代表 blank）会被解码成 `KG7`。

```python
def decode_predictions(logits, idx_to_char):
    """把模型输出解码成文字：去 blank、去连续重复"""
    predictions = logits.argmax(dim=2)
    decoded_texts = []

    for prediction in predictions:
        previous_idx = None
        decoded_chars = []
        for idx in prediction.cpu().numpy():
            # 0 是 blank，跳过；和前一个相同也跳过（CTC 的合并规则）
            if idx != 0 and idx != previous_idx:
                decoded_chars.append(idx_to_char[idx])
            previous_idx = idx
        decoded_texts.append("".join(decoded_chars))

    return decoded_texts
```

验证时同时算损失和精确匹配准确率：

```python
def exact_match_accuracy(predictions, targets):
    """整串完全一致才算对"""
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(targets)

def validate(model, loader, criterion, device, idx_to_char):
    model.eval()
    total_loss = 0
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for images, labels, label_lengths, raw_labels in loader:
            images = images.to(device)
            logits = model(images)

            log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
            input_lengths = torch.full((logits.size(0),), logits.size(1),
                                       dtype=torch.long, device=device)
            total_loss += criterion(log_probs, labels.to(device),
                                    input_lengths, label_lengths.to(device)).item()

            all_predictions.extend(decode_predictions(logits, idx_to_char))
            all_targets.extend(raw_labels)

    return total_loss / len(loader), exact_match_accuracy(all_predictions, all_targets)
```

材料中记录的现象是：训练开始前，未训练的模型输出随机文字，精确匹配为 0，这是一次「 sanity check」（合理性检查），说明解码流程能跑通；训练 5 个 epoch 后，训练和验证损失都稳定下降，验证精确匹配准确率持续上升。说明模型确实在学习整串车牌，而不是只学会输出几个常见字符。

最后我们把测试目录里的图片读进来，走同样的预处理，用训练好的模型批量预测并导出提交文件：

```python
test_paths = sorted((DATA_PATH / "test").glob("*.png"))

# 测试集只需要图片和路径，不需要标签
class LicensePlateTestDataset(Dataset):
    def __init__(self, image_paths, image_width, image_height):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = self.transform(Image.open(path).convert("RGB"))
        return image, f"test/{path.name}"   # 保留提交要求的相对路径
```

## 三、总结

本文中我们完成了吉尔吉斯车牌识别（Kyrgyz ANPR）的完整 baseline。我们首先用 EDA 确认了这是一个「图片到变长文本」的 OCR 任务，并统计了标签长度、字符表和图片尺寸分布。其次我们把所有图片统一缩放成 256×64 的灰度图，构建了带 blank 的字符词表，并按长度分组分层划分训练集和验证集。接着我们搭建了 CNN + 双向 LSTM + CTC 的 CRNN 模型，用 AdamW 优化、梯度裁剪稳定训练 5 个 epoch。最后我们用贪心解码把模型输出还原成文字，以「整串全对」的精确匹配准确率评估，并导出测试集预测。

这个比赛让我们理解到：**变长序列识别不一定要逐字标注位置**，CTC 用一条损失函数就把「对齐」这件事交给模型自己学了。后续如果想提分，可以从更强的 CNN 骨干、数据增强、合成车牌数据，或者引入预训练 OCR 模型这几个方向入手。
