# Kyrgyz ANPR  Kyrgyz License Plate Recognition Challenge

## 一、比赛简介

### 1.1 比赛背景
本比赛是吉尔吉斯斯坦车牌自动识别任务。车牌图像已经被裁剪好，只需识别车牌上的文字序列。车牌包含数字、拉丁字母以及 `KG` 国家标记。识别结果需要字符顺序完全正确才算对。

### 1.2 比赛目标
给定一张裁剪的车牌图像，输出完整的车牌文本。训练集包含 `path` 图像路径和 `label` 文本标签两列。测试集仅有图像，需生成预测文本。

### 1.3 评价标准
比赛使用精确匹配准确率，即整张车牌预测文本与真实文本完全一致的比例。只有全部字符都对才算正确。

## 二、解题过程

我们采用 CRNN 结构作为 OCR baseline，CNN 提取特征，LSTM 建模时序，CTC loss 处理变长序列。

### 2.1 数据探索（EDA）
读取训练 CSV 查看样本形状和标签分布。

```python
import pandas as pd
from pathlib import Path
DATA_PATH = Path("/kaggle/input/competitions/kyrgyz-anpr-kyrgyz-license-plate-recognition-challenge")
train = pd.read_csv(DATA_PATH / "train.csv")
print("train shape:", train.shape)
print(train.head())
```

统计标签长度分布。

```python
labels = train["label"].astype(str)
print("number of samples:", len(labels))
print("min label length:", labels.str.len().min())
print("max label length:", labels.str.len().max())
characters = sorted(set("".join(labels)))
print("number of unique characters:", len(characters))
```

图像尺寸不统一，先抽样 1000 张查看分布。

```python
from PIL import Image
from collections import Counter
sample_paths = train["path"].sample(1000, random_state=42)
sizes = [Image.open(DATA_PATH / p).size for p in sample_paths]
size_counts = Counter(sizes)
print(size_counts.most_common(10))
```

### 2.2 数据清洗
统一图像大小为 256x64，灰度化并归一化。

```python
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 64

transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

构建字符词表，索引 0 留给 CTC 的 blank。

```python
characters = sorted(set("".join(train["label"].astype(str))))
blank_token = "<BLANK>"
char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
num_classes = len(char_to_idx) + 1
```

按标签长度分组分层抽样，避免长度稀有导致验证集失衡。

```python
from sklearn.model_selection import train_test_split
def length_group(length):
    if length <= 9: return "short"
    if length == 10: return "medium"
    if length == 11: return "long"
    return "extra_long"
split_stratify = train["label_length"].apply(length_group)
train_df, valid_df = train_test_split(train, test_size=0.15, stratify=split_stratify, random_state=42)
```

### 2.3 模型训练与超参数调优
构建 CRNN 模型，CNN 提取特征，BiLSTM 编码时序。

```python
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.rnn = nn.LSTM(input_size=512*4, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.classifier = nn.Linear(512, num_classes)
    def forward(self, x):
        features = self.cnn(x)
        features = features.permute(0,3,1,2).reshape(x.size(0), -1, 512*4)
        sequence, _ = self.rnn(features)
        return self.classifier(sequence)
```

使用 CTC loss 训练。

```python
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

每轮训练计算 loss 并用梯度裁剪。

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels, label_lengths, _ in loader:
        images = images.to(device); labels = labels.to(device)
        logits = model(images)
        log_probs = logits.log_softmax(dim=2).permute(1,0,2)
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
```

### 2.4 模型测试与指标报告
贪心解码得到预测文本。

```python
def decode_predictions(logits, idx_to_char):
    predictions = logits.argmax(dim=2)
    decoded = []
    for pred in predictions:
        chars = []
        prev = None
        for idx in pred.cpu().numpy():
            if idx != 0 and idx != prev:
                chars.append(idx_to_char[idx])
            prev = idx
        decoded.append("".join(chars))
    return decoded
```

验证集精确匹配准确率。

```python
def exact_match_accuracy(preds, targets):
    return sum(p==t for p,t in zip(preds, targets)) / len(targets)
```

训练 5 个 epoch 后验证准确率持续提升，loss 下降。

导出测试预测。

```python
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
with torch.no_grad():
    for images, paths in test_loader:
        logits = model(images.to(device))
        preds = decode_predictions(logits, idx_to_char)
```

## 三、总结

本文中我们完成了 Kyrgyz ANPR 的 CRNN baseline。我们首先探索了车牌图像和标签长度分布，其次统一图像尺寸并构建 CTC 词表，接着用 CNN+LSTM+CTC 训练 OCR 模型，最后在验证集上使用精确匹配评估并生成提交文件。CRNN 能学习完整的车牌序列，是车牌识别的有效入门方案。
