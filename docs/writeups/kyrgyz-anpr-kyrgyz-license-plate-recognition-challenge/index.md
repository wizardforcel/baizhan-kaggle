# 吉尔吉斯斯坦车牌识别挑战（Kyrgyz ANPR）

## 一、比赛简介

### 1.1 比赛背景

车牌自动识别（Automatic License Plate Recognition，简称 ANPR）是交通、停车场、违章抓拍等场景里的常见需求：系统拍到一辆车，自动读出它的车牌号。这个比赛针对的是吉尔吉斯斯坦的车牌。

吉尔吉斯的车牌除了数字和拉丁字母，还有一个固定标记 `<KG>`。所以车牌文本是一个「数字 + 字母 + 国家标记」组成的有序字符串，比如类似 `01KG456AB` 的格式。这带来了两个难点：

第一，这是**序列识别**问题，不是普通图像分类。分类问题只要猜出图片是哪个类别，而这里要按顺序读出每一个字符，顺序错一位就算错。

第二，评价非常严格。预测只有当**整串车牌文本和真实标签完全相同**时才算对，差一个字符都是错。这样一个「全对才算对」的指标，让每个字符的识别精度都很关键。

### 1.2 比赛目标

输入是一张裁剪好的车牌图片，输出是车牌的完整文本序列。数据文件主要有：

- `train.csv`：训练标注表，有两列——`path` 指向车牌图片文件，`label` 是车牌文本真值；
- `test` 目录：测试图片，无标签。

模型要做的事，就是从图片里读出字符序列。我们计划搭建一个**CRNN**模型：先用卷积网络（CNN）从图像里提取视觉特征，再用循环网络（LSTM）按时间顺序建模字符序列，最后用 CTC 损失对齐「图片」和「文本」。这种结构是车牌、手写体等 OCR（光学字符识别）任务的标准做法。

### 1.3 评价标准

比赛用**精确匹配准确率（exact-match accuracy）**评价：把预测的车牌文本和真实文本逐条对比，只有完全相同的才算对，用算对的条数除以总条数得到准确率。分数越高越好。

这个指标非常苛刻。它不是数「对了几个字符」，而是数「几块牌照被完整读出」。十连全对和十连错一字符，在精确匹配下就是满分和零分的差别。

## 二、解题过程

我们按照 EDA、数据清洗、模型训练与调参、模型测试与指标报告四步，搭建一个完整可复现的 PyTorch OCR 基线。整体思路是 CRNN + CTC：卷积网络读图的视觉结构，LSTM 建模字符的前后依赖，CTC 负责把「每个时间步的分类结果」对齐成「不定的文本长度」。

### 2.1 数据探索（EDA）

第一步是读训练标注表，看数据长什么样：

```python
from pathlib import Path
import pandas as pd
from PIL import Image

DATA_PATH = Path("/kaggle/input/competitions/kyrgyz-anpr-kyrgyz-license-plate-recognition-challenge")

# 读取训练标注：path 指向图片，label 是车牌文本
train = pd.read_csv(DATA_PATH / "train.csv")
print("train shape:", train.shape)
print(train.head())
```

确认了数据就是两列：图片路径和 OCR 目标。这是一个序列识别任务，模型要把每张裁剪好的车牌图片映射成它的完整文本标签。

接着分析标签的长度和字符集。车牌长度不固定，字符集是紧凑的：

```python
# 统计车牌文本长度分布
labels = train["label"].astype(str)
print("min label length:", labels.str.len().min())
print("max label length:", labels.str.len().max())
print("mean label length:", labels.str.len().mean())

# 收集全部出现过的字符，构成词表
characters = sorted(set("".join(labels)))
print("number of unique characters:", len(characters))
print("characters:", characters)

train["label_length"] = labels.str.len()
train["label_length"].value_counts().sort_index()
```

车牌长度有长有短（有几个字符的，也有十来个字符的），但字符种类不多。这种「长度不定的文本 + 紧凑字符集」正是 CTC 模型的用武之地——CTC 不要求固定位置的字符标签，天然支持变长输出。

然后抽样看一些图片，检查图像质量：

```python
import matplotlib.pyplot as plt

sample_df = train.sample(12, random_state=42)
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

肉眼观察能发现两个信息：一是车牌里的字符是否清晰、会不会有粘连；二是图片尺寸参差不齐，这提示我们后面需要统一大小。

我们再用代码确认图片尺寸的不一致性：

```python
from collections import Counter

# 抽样 1000 张，统计尺寸种类
sample_paths = train["path"].sample(1000, random_state=42)
sizes = Counter(Image.open(DATA_PATH / p).size for p in sample_paths)
print("number of unique sizes in sample:", len(sizes))
for size, count in sizes.most_common(10):
    print(size, count)
```

结果发现图片尺寸种类很多，不能直接打包成 batch 训练。由于车牌是横长的结构，我们把所有图片统一缩放成宽 256、高 64 的固定尺寸——宽高比 4:1，既接近真实车牌的比例，又适合作为「序列」按横向切块建模。

### 2.2 数据清洗

清洗的核心是把「标签文本」转成「字符索引序列」，并为 CTC 预留一个空位（blank）类。

```python
# 从训练集收集全部字符，作为词表
characters = sorted(set("".join(train["label"].astype(str))))
blank_token = "<BLANK>"   # CTC 需要的空白类

# 字符从索引 1 开始编号，把索引 0 留给 blank
char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# CTC 的类别数 = 字符数 + 1（多出的 1 是 blank）
num_classes = len(char_to_idx) + 1
print("vocabulary size without blank:", len(char_to_idx))
print("CTC classes with blank:", num_classes)
```

为什么要多一个 blank？CTC 的工作原理是：模型在每个时间步输出一个字符分布，blank 类表示「这个时间步没有字符」。训练时，一张「01KG456AB」的图会被展开成很多时间步，模型可能在中间输出大量 blank 或重复字符，CTC 会通过合并重复字符、去掉 blank 的方式还原出最终文本。所以 blank 类是 CTC 必不可少的一部分。

接着划分训练集和验证集。车牌长度有几类，直接按精确长度分层分割会因为某些长度样本太少而不稳定，所以我们把长度分成几组再分层：

```python
from sklearn.model_selection import train_test_split

# 把长度分成短/中/长/超长四组，按组分层
def length_group(length):
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
    test_size=0.15,
    random_state=42,
    stratify=split_stratify
)
print("train split:", train_df.shape)
print("valid split:", valid_df.shape)
```

验证集占 15%。按长度组分层的好处是：既不让验证集偏离整体分布，又避免了某些过稀有长度单独成组导致的分层报错。

### 2.3 模型训练与超参数调优

先建一个自定义 Dataset，把「读图 → 缩放 → 转灰度 → 归一化 → 标签编码」这几步固定下来：

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LicensePlateDataset(Dataset):
    def __init__(self, dataframe, data_path, char_to_idx, image_width, image_height):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_path = data_path
        self.char_to_idx = char_to_idx

        # 统一缩放到宽 256 高 64，转单通道灰度，再归一化
        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def encode_label(self, label):
        # 把字符串标签逐字符映射成索引序列
        return torch.tensor([self.char_to_idx[ch] for ch in label], dtype=torch.long)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(self.data_path / row["path"]).convert("RGB")
        image = self.transform(image)
        label = str(row["label"])
        return image, self.encode_label(label), label
```

因为每条车牌的字符个数不同，DataLoader 直接打包会失败。我们写一个 `collate_fn`：图片堆成张量，标签全部连成一个大张量，同时记录每条标签的长度：

```python
def collate_fn(batch):
    images, encoded_labels, raw_labels = zip(*batch)
    images = torch.stack(images)   # (batch, 1, 64, 256)
    label_lengths = torch.tensor([len(l) for l in encoded_labels], dtype=torch.long)
    labels = torch.cat(encoded_labels)   # 所有标签按顺序连成 1D
    return images, labels, label_lengths, raw_labels
```

这种「标签连成大张量 + 长度表」的格式是 CTC 损失的标准输入要求。

接下来定义 CRNN 模型。主干是几层卷积加批归一化，逐步把特征图压缩，横向保留足够的宽度供 LSTM 沿时间方向扫描：

```python
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 卷积主干：把 1x64x256 的输入压缩成 (512通道, 高4, 宽64)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                       # 高度和宽度各减半

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),         # 只压缩高度，宽度保留

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),         # 再次只压缩高度
        )

        # 双向 LSTM：沿车牌横向顺序建模字符依赖
        self.rnn = nn.LSTM(
            input_size=512 * 4,   # 每一列的通道特征
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        # 每个时间步输出一个字符类别得分
        self.classifier = nn.Linear(512, num_classes)  # 双向拼接后是 2*256

    def forward(self, x):
        features = self.cnn(x)                       # (batch, 512, 4, 64)
        features = features.permute(0, 3, 1, 2)      # 换成 (batch, 宽, 通道, 高)
        batch_size, seq_len, channels, height = features.size()
        features = features.reshape(batch_size, seq_len, channels * height)
        # 现在 (batch, 64) 沿着宽度方向形成 64 个时间步

        sequence, _ = self.rnn(features)
        logits = self.classifier(sequence)           # (batch, time, num_classes)
        return logits
```

这里有个关键设计：池化层用 `(2, 1)` 而不是 `(2, 2)`。因为车牌是横长的，我们希望高度方向不断压缩（把文字行压成一条线），但宽度方向尽量保留（宽度方向正好充当「时间轴」）。最终每个时间步的输出代表「车牌某一列的字符预测」。最后把双向 LSTM 两个方向的隐藏状态拼接成 512 维，映射成类别分数。

损失函数用 CTC。CTC 要求输入是（时间、batch、类别）的顺序，并且用对数概率：

```python
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

log_probs = logits.log_softmax(dim=2)      # 每步转成对数概率
log_probs = log_probs.permute(1, 0, 2)     # 时间在前

input_lengths = torch.full(size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long)

loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
```

这里的 `input_lengths` 是模型输出的时间步数（每张图都一样，都是 64），`label_lengths` 是每条真实车牌的字符数。CTC 会在所有可能的「对齐路径」上求联合概率，让模型学会「图片内容 + 时间步来模拟目标文本」。

**关于超参数与训练细节。** 我们这样设置：

- 图片统一 256×64，灰度单通道；
- 优化器用 AdamW，学习率 1e-3，权重衰减 1e-4；
- batch size 64，训练 5 个 epoch；
- 每个 batch 更新后做梯度裁剪（max_norm=5），防止长序列训练时梯度爆炸；
- 损失函数 `zero_infinity=True`，避免某些对齐路径概率为 0 导致的 inf。

这个配置是可以稳定跑通的基线。卷积核数量、LSTM 层数、学习率都有继续调的空间，但材料里的重点是先把一条干净的流水线跑起来。

### 2.4 模型测试与指标报告

训练好模型后，第一步是把每个时间步的输出解码成文本。CTC 的贪心解码策略很简单：每个时间步取概率最大的类别，然后**合并相邻的重复字符，并去掉 blank**：

```python
def decode_predictions(logits, idx_to_char):
    predictions = logits.argmax(dim=2)   # 每步取概率最大的类别
    decoded_texts = []
    for prediction in predictions:
        previous_idx = None
        decoded_chars = []
        for idx in prediction.cpu().numpy():
            if idx != 0 and idx != previous_idx:   # idx==0 是 blank，跳过；重复也跳过
                decoded_chars.append(idx_to_char[idx])
            previous_idx = idx
        decoded_texts.append("".join(decoded_chars))
    return decoded_texts
```

解码出来是个长度不定的字符串。为什么 blank 和重复都要去掉？因为 CTC 训练时，一张「AB」的图可能被展开成「A A blank B B」，识别时就要把重复的 A、中间的 blank、重复的 B 全部折叠回「AB」。

然后定义精确匹配准确率，并写一个验证函数：

```python
def exact_match_accuracy(predictions, targets):
    correct = sum(pred == target for pred, target in zip(predictions, targets))
    return correct / len(targets)

def validate(model, loader, criterion, device, idx_to_char):
    model.eval()
    total_loss = 0
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for images, labels, label_lengths, raw_labels in loader:
            images, labels = images.to(device), labels.to(device)
            label_lengths = label_lengths.to(device)
            logits = model(images)
            # ... 计算 CTC 损失 ...
            predictions = decode_predictions(logits, idx_to_char)
            all_predictions.extend(predictions)
            all_targets.extend(raw_labels)
    return total_loss / len(loader), exact_match_accuracy(all_predictions, all_targets)
```

每轮训练后都在验证集上算一次损失和精确匹配准确率：

```python
EPOCHS = 5
history = []
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, ctc_loss, device)
    valid_loss, valid_accuracy = validate(model, valid_loader, ctc_loss, device, idx_to_char)
    history.append({"epoch": epoch, "train_loss": train_loss,
                    "valid_loss": valid_loss, "valid_acc": valid_accuracy})
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, "
          f"valid_exact_match_accuracy={valid_accuracy:.4f}")
```

训练曲线显示：损失逐轮下降，验证集的精确匹配准确率稳步提升。第一轮后 CTC 损失已经明显下降，说明模型学到了「图像和文本序列的对齐关系」；但精确匹配准确率一开始是 0，因为整串车牌全部读对很难，需要更多轮次才能出现第一个全对的样本。到第 5 轮，验证集上大多数预测都能和真实车牌完全匹配，剩下的错误主要是相似字符之间的混淆（比如 0 和 O、1 和 I）。

训练完成后，把模型和词表一起存成 checkpoint，便于复用：

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "char_to_idx": char_to_idx,
    "idx_to_char": idx_to_char,
    "image_width": 256, "image_height": 64,
    "num_classes": num_classes,
    "history": history,
}, "/kaggle/working/crnn_anpr_baseline.pth")
```

最后对测试集做推理并生成提交文件。测试集是 `test` 目录下没有标签的图片，我们对每张图用同样的预处理和贪心解码，得到预测车牌：

```python
model.eval()
submission_rows = []
with torch.no_grad():
    for images, paths in test_loader:
        images = images.to(device)
        logits = model(images)
        predictions = decode_predictions(logits, idx_to_char)
        for path, prediction in zip(paths, predictions):
            submission_rows.append({"path": path, "label": prediction})

submission_df = pd.DataFrame(submission_rows)
print("empty predictions:", (submission_df["label"] == "").sum())
submission_df.to_csv("/kaggle/working/submission.csv", index=False)
```

提交前要检查两点：一是 `path` 列用比赛要求的相对路径（比如 `test/xxx.png`）而不是绝对路径；二是确认没有预测出空字符串。材料里最终全部测试图片都产出了非空的车牌，可以直接提交。

## 三、总结

本文中我们完成了吉尔吉斯斯坦车牌识别挑战的完整 CRNN 基线。我们首先通过 EDA 确认了这是一个序列识别任务：数据是「图片 + 车牌文本」，车牌长度不固定、图片尺寸不一、评价只看整串全对。其次做了数据清洗：把标签转成字符索引序列、为 CTC 预留 blank 类、按长度分组分层划分验证集、并把图片统一缩放成 256×64。接着搭建了 CRNN 模型——卷积主干提取视觉特征、双向 LSTM 沿横向建模字符序列、CTC 损失做对齐，配合 AdamW、梯度裁剪训练了 5 轮。最后用贪心解码和精确匹配准确率评估，对测试集逐图读出车牌并导出提交文件。

这套基线还有很大的提升空间。材料最后提到，后续可以换成更强的视觉编码器、加入数据增强（旋转、模糊、对比度调整）、用合成数据扩充，或者接入预训练 OCR 组件。结构相似字符（0/O、1/I）的混淆是当前最突出的错误来源，针对这类难例做针对性增强，是进一步提分的方向。