# Korean Gender Bias Detection 韩语性别偏见检测

## 一、比赛简介

### 1.1 比赛背景

性别偏见是指语言或文本中体现出的对特定性别的刻板印象、歧视或不平等对待。比如招聘广告里写「性别为男性优先」，或某些描述女性时使用贬低性的词汇。在社交媒体、招聘网站、评论区里，这类偏见文本如果不加以识别和过滤，会放大社会不平等。

韩语作为一种形态丰富的语言，性别偏见往往藏在词尾、敬语、使用词、甚至是隐晦的隐喻里。单纯靠词典或规则很难识别，所以比赛希望用机器学习模型让机器学会从语义和上下文里判断一段韩语评论是否含有性别偏见。

### 1.2 比赛目标

比赛的任务是二分类：判断一段韩语评论是否含有性别偏见。

训练数据文件是 `train.gender_bias.binary.csv`，开发数据文件是 `dev.gender_bias.binary.csv`。两者都有两列，`comments` 是韩语评论文本，`label` 是对应的偏见标签。测试集只提供 `comments`，需要我们输出预测结果。

从材料来看，标签在原始数据里是布尔值 `True/False`，我们需要把它转成 `1/0` 的数值标签，方便模型训练。测试集预测完后，建议再转回 `True/False` 再输出。

### 1.3 评价标准

这类文本分类比赛常用准确率和 F1 分数。准确率衡量整体猜对的比例，F1 在正负样本不平衡时更稳健。材料里 baseline 着重看开发集上的分类准确率，最终排行榜通常用准确率。分数越高越好。

## 二、解题过程

我们采用字符级 CNN 作为 baseline。字符级处理的一大好处是**不需要预先构建韩语词表**。韩语词形变化多，相同的词可能有不同拼写，字符级模型直接在字符上卷积，可以更好地捕捉子词模式，而且对拼写错误更鲁棒。

下面按 EDA、数据清洗、模型训练与调参、测试报告四步展开。

### 2.1 数据探索（EDA）

第一步是确认数据路径和列名：

```python
import os
import pandas as pd

# 列出 Kaggle 输入目录下的所有文件，确认数据是否到位
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

输出里我们能看到 `koreangenderbiasdetection/train.gender_bias.binary.csv`、`dev.gender_bias.binary.csv` 和测试集文件。

接着读训练和开发集，快速浏览几行：

```python
# 读取训练集和开发集
train_df = pd.read_csv('/kaggle/input/koreangenderbiasdetection/train.gender_bias.binary.csv')
dev_df = pd.read_csv('/kaggle/input/koreangenderbiasdetection/dev.gender_bias.binary.csv')

train_df.head()
dev_df.head()
```

我们看到两列 `comments` 和 `label`。为了方便训练，先把布尔标签转成整数：

```python
# 布尔标签转 1/0：True -> 1 代表有偏见，False -> 0 代表无偏见
train_df['label'] = train_df['label'].map(lambda x: 1 if x else 0)
dev_df['label'] = dev_df['label'].map(lambda x: 1 if x else 0)
```

然后把句子拆成字符列表，字符级模型就是在字符序列上操作：

```python
# 把每条评论拆成字符列表，韩语是按 Unicode 字符拆分
train_sentences = [list(d) for d in train_df['comments'].to_list()]
dev_sentences  = [list(d) for d in dev_df['comments'].to_list()]

train_labels = train_df['label'].to_list()
dev_labels = dev_df['label'].to_list()
```

再看一下最大长度，决定后面补齐到多长：

```python
max_len = train_df['comments'].map(len).max()
print("最大评论长度:", max_len)   # 大约在 150 个字符以内
```

### 2.2 数据清洗

这个任务的数据清洗很「轻」。我们不做分词、不做大小写转换、不做停用词过滤。因为：

- 韩语没有大小写；
- 性别偏见可能就藏在某个特定词尾或助词里，去掉它们反而会丢信息；
- 字符级模型本来就不依赖词表。

所以我们只做两件事：一是**统一长度**，二是**构建字符词表**。

首先构建词表。字符级模型需要一个「字符到索引」的映射。常见做法是先统计所有字符出现频率，取最常见的前 N 个，其余归为未知符 `<UNK>`。另外我们加入 `<PAD>` 用来填充短句子，`<SOS>`、`<EOS>` 在这个简化版里不使用。

```python
from collections import Counter

def make_dictionary(sentences, vocabulary_size=None, initial_words=['<UNK>', '<PAD>']):
    """从句子列表里统计字符频率，生成词表"""
    counter = Counter()
    for words in sentences:          # 每个句子是一个字符列表
        counter.update(words)

    if vocabulary_size is None:
        vocabulary_size = len(counter.keys())

    # 按频率排序，取前 vocabulary_size 个
    vocab_words = counter.most_common(vocabulary_size)

    # 把特殊符插在最前面，索引从 0 开始
    for initial_word in initial_words:
        vocab_words.insert(0, (initial_word, 0))

    word2idx = {word: idx for idx, (word, _) in enumerate(vocab_words)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word
```

```python
word2idx, idx2word = make_dictionary(train_sentences, initial_words=['<UNK>', '<PAD>'])
print("词表大小:", len(word2idx))
```

接下来把句子处理成固定长度。太长的句子截断，太短的句子用 `<PAD>` 补满。这样批量训练时每个样本都是同样长度的张量。

```python
def process_sentences(sentences, word2idx, sentence_length=150, padding='<PAD>'):
    """把字符列表转成索引列表，并统一到 sentence_length"""
    sentences_processed = []
    for sentence in sentences:
        if len(sentence) > sentence_length:
            fixed_sentence = sentence[:sentence_length]   # 截断
        else:
            # 补足到固定长度
            fixed_sentence = sentence + [padding] * (sentence_length - len(sentence))

        # 未知字符映射到 <UNK>
        sentence_idx = [word2idx.get(w, word2idx['<UNK>']) for w in fixed_sentence]
        sentences_processed.append(sentence_idx)
    return sentences_processed
```

```python
SENTENCE_LENGTH = 150   # 根据最大长度选择 150
train_sentences_processed = process_sentences(train_sentences, word2idx, sentence_length=SENTENCE_LENGTH)
dev_sentences_processed = process_sentences(dev_sentences, word2idx, sentence_length=SENTENCE_LENGTH)
```

### 2.3 模型训练与超参数调优

模型是 CharCNN，结构很经典：

1. **字符嵌入**：把每个字符索引转成一个 300 维的向量。
2. **多尺度卷积**：用不同大小的卷积核（3、4、5）分别捕捉不同长度的字符模式。3 表示三字符模式，5 表示五字符模式，这样能同时抓住短词和稍长短语。
3. **最大池化**：每个卷积核输出后取最大值，留下的就是该模式在整句里最强烈的信号。
4. **拼接 + 全连接**：把不同卷积核的输出拼在一起，再过一层线性分类，判断是否含有性别偏见。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, kernel_num, kernel_sizes):
        super().__init__()
        # 字符嵌入，padding_idx=0 就是 <PAD>，不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        # 多个不同大小的卷积核
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_size, kernel_num, kernel_size=k)
            for k in kernel_sizes
        ])
        # 对应地最大池化，池化窗口和卷积核同尺寸
        self.maxpools = nn.ModuleList([
            nn.MaxPool1d(kernel_size=k)
            for k in kernel_sizes
        ])

        # 三个卷积核各产生 kernel_num 个特征，加起来是 3*kernel_num 个
        # 这里按材料里计算好的 1140 做线性层输入
        self.linear = nn.Linear(1140, output_size)
        self.softmax = nn.LogSoftmax(dim=1)   # NLLLoss 需要 log 概率
        self.dropout = nn.Dropout(p=0.5)      # 防止过拟合

    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)          # batch × seq_len × embed_dim
        embedded = embedded.transpose(1, 2)   # Conv1d 要求 batch × channel × length

        pools = []
        for conv, maxpool in zip(self.convs, self.maxpools):
            feature_map = conv(embedded)      # 提取局部字符模式
            pooled = maxpool(feature_map)     # 取最大值
            pools.append(pooled)

        # 把不同卷积核的输出拼在一起
        conv_concat = torch.cat(pools, dim=-1).view(batch_size, -1)
        conv_concat = self.dropout(conv_concat)
        logits = self.linear(conv_concat)
        return self.softmax(logits)
```

创建模型并初始化：

```python
model = CharCNN(
    vocab_size=len(word2idx),
    embedding_size=300,          # 每个字符 300 维
    output_size=2,               # 二分类：有偏见 / 无偏见
    kernel_num=10,               # 每个卷积核大小产生 10 个特征
    kernel_sizes=[3, 4, 5]       # 三种多尺度
)

if torch.cuda.is_available():
    model.to('cuda')
```

数据加载器：

```python
from torch.utils.data import Dataset, DataLoader

class DatasetLoader(Dataset):
    def __init__(self, sentences_processed, labels):
        self.sentences_processed = sentences_processed
        self.labels = torch.LongTensor(labels)
    def __getitem__(self, index):
        return torch.LongTensor(self.sentences_processed[index]), self.labels[index]
    def __len__(self):
        return len(self.sentences_processed)

train_dataset = DatasetLoader(train_sentences_processed, train_labels)
dev_dataset = DatasetLoader(dev_sentences_processed, dev_labels)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=256)
```

训练循环。模型用 NLLLoss（和 LogSoftmax 配对），优化器用 Adam。每个 epoch 我们都会打印训练损失和开发集准确率。

```python
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

loss_function = nn.NLLLoss()
optimizer = Adam(model.parameters())

for epoch in range(6):   # 该 baseline 训练 6 个 epoch
    losses = []

    for batch_data, batch_label in train_dataloader:
        if torch.cuda.is_available():
            batch_data = batch_data.to('cuda')
            batch_label = batch_label.to('cuda')

        log_probs = model(batch_data)
        loss = loss_function(log_probs, batch_label)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # 简单的开发集评估
    model.eval()
    corrects = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_label in dev_dataloader:
            if torch.cuda.is_available():
                batch_data = batch_data.to('cuda')
                batch_label = batch_label.to('cuda')
            log_probs = model(batch_data)
            pred = log_probs.argmax(dim=1)
            corrects += (pred == batch_label).sum().item()
            total += batch_label.size(0)

    print(f"Epoch {epoch}: loss={sum(losses)/len(losses):.4f}, dev_acc={corrects/total:.4f}")
    model.train()
```

材料里记录的训练过程显示，开发集准确率在前几个 epoch 快速上升后趋于平缓，说明 6 个 epoch 已经足够。

### 2.4 模型测试与指标报告

训练完成后，把模型切换到评估模式，对测试集做预测。注意测试集没有标签，我们只能推理。

```python
model.eval()
test_df = pd.read_csv('/kaggle/input/koreangenderbiasdetection/test.gender_bias.no_label.csv')
test_sentences = [list(d) for d in test_df['comments'].to_list()]
test_sentences_processed = process_sentences(test_sentences, word2idx, sentence_length=SENTENCE_LENGTH)

# 测试数据集用占位标签
test_dataset = DatasetLoader(test_sentences_processed, [0]*len(test_sentences_processed))
test_dataloader = DataLoader(test_dataset, batch_size=1)

res = []
for batch_data, _ in test_dataloader:
    if torch.cuda.is_available():
        batch_data = batch_data.to('cuda')
    with torch.no_grad():
        log_probs = model(batch_data)
        pred = log_probs.argmax(dim=1).cpu().numpy()[0]
        res.append(pred)
```

把预测结果写回 DataFrame，并转成布尔值再保存：

```python
test_df['label'] = res
test_df['label'] = test_df['label'].map(lambda x: True if x else False)
test_df.to_csv('/kaggle/working/prediction.csv', index=None)
```

## 三、总结

本文中我们完成了韩语性别偏见检测的字符级 CNN baseline。我们首先把韩语评论拆成字符并构建词表，其次统一句子长度到 150 并填充。其次我们使用多尺度 CharCNN 模型训练 6 个 epoch，并在开发集上监控准确率。最后我们对测试集做推理并导出预测结果。

字符级模型不需要词表，对韩语形态变化鲁棒，是处理韩语文本分类的一个很稳健的起点。后续如果想提分，可以考虑使用预训练韩语句子模型如 KcBERT、KoELECTRA，或者加入数据增强和类不平衡处理的技巧。
