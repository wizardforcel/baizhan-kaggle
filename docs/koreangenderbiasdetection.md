# 韩语性别偏见检测（Korean Gender Bias Detection）

## 一、比赛简介

### 1.1 比赛背景

韩语日常表达里，有一些说法表面看很普通，实际上带着性别刻板印象。比如把某种职业默认成男性，或者用带贬义的词描述女性。这类「性别偏见」比赤裸裸的辱骂更隐蔽，也更难用关键词规则识别。

这个比赛的目的是训练一个模型，判断一条韩语评论里是否含有性别偏见。它和普通的文本分类任务相比有两个特点：一是偏见往往藏在语气和搭配里，不是单个词能决定的；二是标注带有主观性，同一条评论不同的人可能给出不同判断。

### 1.2 比赛目标

输入是评论正文，字段名为 `comments`。数据文件有三个：

- `train.gender_bias.binary.csv`：训练集，含 `comments` 和 `label`；
- `dev.gender_bias.binary.csv`：验证集，格式相同；
- `test.gender_bias.no_label.csv`：测试集，只有评论。

`label` 是布尔值，`True` 表示含有性别偏见，`False` 表示没有。这是一个**二分类**任务。我们需要给测试集每条评论预测 True/False，并输出预测文件。

### 1.3 评价标准

材料里没有直接给出比赛的评价指标定义，只说明了标签是二分类的 True/False。从任务类型看，这类比赛通常用准确率或 F1 来评分，分数越高越好。我们下面的训练脚本用**准确率（accuracy）**监控训练和验证过程，这也是最容易解释的指标：预测对的条数除以总条数。

## 二、解题过程

我们采用字符级卷积网络（CharCNN）作为基线。之所以按「字符」而不是按「词」处理，是因为韩语里同一个意思可以有很多种写法，按词切分容易遇到大量没见过的词；按字符处理，词表小得多，模型的泛化也更稳。下面按 EDA、清洗、训练调参、测试报告四步走。

### 2.1 数据探索（EDA）

先把训练集、验证集、测试集读进来，看列名和规模：

```python
import pandas as pd

# 三个文件的结构：训练/验证有 label，测试没有
train_df = pd.read_csv('/kaggle/input/koreangenderbiasdetection/train.gender_bias.binary.csv')
dev_df = pd.read_csv('/kaggle/input/koreangenderbiasdetection/dev.gender_bias.binary.csv')
test_df = pd.read_csv('/kaggle/input/koreangenderbiasdetection/test.gender_bias.no_label.csv')

print(train_df.head())
print(train_df.shape, dev_df.shape, test_df.shape)
```

能确认数据只有 `comments` 和 `label` 两列（测试集只有 `comments`）。接着看标签分布：

```python
# 看正负样本比例，True 表示有性别偏见
print(train_df['label'].value_counts())
print(train_df['label'].value_counts(normalize=True))
```

如果正负样本差距很大，后面训练时要注意。不过这份数据是二分类，材料里的脚本没有做额外的重采样，说明比例还在可接受范围内。

再看评论长度。这个数字决定了我们要把序列截断到多长：

```python
# 评论的最大字符数，材料里注释说最长约 150
print(train_df['comments'].map(len).max())
print(train_df['comments'].map(len).describe())
```

确认最长在 150 个字符以内，所以后面统一把序列长度设成 150。这个长度既覆盖了几乎所有评论，又不会让模型处理大量无意义的填充。

### 2.2 数据清洗

原始 `label` 是布尔值，先转成 0/1 整数，因为 PyTorch 的损失函数要的是整数类别：

```python
# True -> 1，False -> 0
train_df['label'] = train_df['label'].map(lambda x: 1 if x else 0)
dev_df['label'] = dev_df['label'].map(lambda x: 1 if x else 0)
```

然后是字符级切分。`list(d)` 会把一个字符串拆成单个字符的列表，比如 `"안녕"` 变成 `['안', '녕']`：

```python
# 按字符切分，而不是按空格切词
train_sentences = [list(d) for d in train_df['comments'].to_list()]
dev_sentences = [list(d) for d in dev_df['comments'].to_list()]

train_labels = train_df['label'].to_list()
dev_labels = dev_df['label'].to_list()
```

接着构建字符词表。我们统计训练集里每个字符出现的次数，按频率从高到低排序，再在最前面插入四个特殊符号：

```python
from collections import Counter

def make_dictionary(sentences, vocabulary_size=None,
                    initial_words=['<UNK>', '<PAD>', '<SOS>', '<EOS>']):
    counter = Counter()
    for words in sentences:
        counter.update(words)          # 统计每个字符出现次数

    if vocabulary_size is None:
        vocabulary_size = len(counter.keys())

    vocab_words = counter.most_common(vocabulary_size)
    # 特殊符号插到最前面，保证它们的编号是固定的
    for initial_word in initial_words:
        vocab_words.insert(0, (initial_word, 0))

    word2idx = {word: idx for idx, (word, count) in enumerate(vocab_words)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

word2idx, idx2word = make_dictionary(train_sentences, initial_words=['<UNK>', '<PAD>'])
print(len(word2idx))
```

`<UNK>` 用来代表没见过的字符，`<PAD>` 用来补齐短句。词表只从训练集统计，这是一个必须遵守的纪律：如果连验证集和测试集的字符一起统计，就等于提前看到了评估数据。

然后做截断和补齐，把每条评论变成固定 150 个整数：

```python
def process_sentences(sentences, word2idx, sentence_length=20, padding='<PAD>'):
    sentences_processed = []
    for sentence in sentences:
        if len(sentence) > sentence_length:
            fixed_sentence = sentence[:sentence_length]   # 超长截断
        else:
            fixed_sentence = sentence + [padding] * (sentence_length - len(sentence))  # 补 PAD

        # 词表里没有的字符统一映射到 <UNK>
        sentence_idx = [word2idx[word] if word in word2idx else word2idx['<UNK>']
                        for word in fixed_sentence]
        sentences_processed.append(sentence_idx)
    return sentences_processed

SENTENCE_LENGTH = 150
train_sentences_processed = process_sentences(train_sentences, word2idx, SENTENCE_LENGTH)
dev_sentences_processed = process_sentences(dev_sentences, word2idx, SENTENCE_LENGTH)
```

最后把数据包装成 PyTorch 的 Dataset 和 DataLoader，训练时按 batch 取数据：

```python
from torch.utils.data import Dataset, DataLoader

class DatasetLoader(Dataset):
    def __init__(self, sentences_processed, labels):
        assert len(sentences_processed) == len(labels)
        self.sentences_processed = sentences_processed
        self.labels = torch.LongTensor(labels)

    def __getitem__(self, index):
        return torch.LongTensor(self.sentences_processed[index]), self.labels[index]

    def __len__(self):
        return len(self.sentences_processed)

train_dataloader = DataLoader(DatasetLoader(train_sentences_processed, train_labels), batch_size=128)
dev_dataloader = DataLoader(DatasetLoader(dev_sentences_processed, dev_labels), batch_size=256)
```

训练 batch 用 128，验证 batch 用 256，因为验证不需要反向传播，可以开大一点加快速度。

### 2.3 模型训练与超参数调优

CharCNN 的结构是：先查字符嵌入，再用 3、4、5 三种大小的卷积核在字符序列上滑动，抓取「2 到 4 个字符组成的局部片段」，然后做最大池化，把不同长度的特征压成固定长度，拼起来送进全连接层。

```python
import torch
import torch.nn as nn

class CharCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, kernel_num, kernel_sizes):
        super().__init__()
        # 字符嵌入层，padding_idx=0 让 PAD 不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        # 每个卷积核大小对应一个 Conv1d
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_size, kernel_num, kernel_size=k)
            for k in kernel_sizes
        ])
        self.maxpools = nn.ModuleList([
            nn.MaxPool1d(kernel_size=k) for k in kernel_sizes
        ])
        self.linear = nn.Linear(1140, output_size)
        self.softmax = nn.LogSoftmax(dim=1)   # 配合 NLLLoss 使用
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)          # (batch, len, embed)
        embedded = embedded.transpose(1, 2)   # (batch, embed, len) 给 Conv1d

        pools = []
        for conv, maxpool in zip(self.convs, self.maxpools):
            feature_map = conv(embedded)
            pooled = maxpool(feature_map)
            pools.append(pooled)

        conv_concat = torch.cat(pools, dim=-1).view(batch_size, -1)
        conv_concat = self.dropout(conv_concat)   # 全连接前 dropout 防过拟合
        logits = self.linear(conv_concat)
        return self.softmax(logits)
```

超参数方面：嵌入维度 300，每种卷积核 10 个，卷积核大小 `[3,4,5]`，输出 2 类，dropout 0.5。优化器用 Adam，默认学习率，训练 6 轮。损失函数用负对数似然 `NLLLoss`，它要求模型输出的是对数概率，所以网络最后接了 `LogSoftmax`。

```python
model = CharCNN(
    vocab_size=len(word2idx),
    embedding_size=300,
    output_size=2,
    kernel_num=10,
    kernel_sizes=[3, 4, 5]
)
if torch.cuda.is_available():
    model.to('cuda')

loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())
```

训练循环里，每个 batch 做前向、算损失、反向传播、更新参数，同时统计训练准确率；每个 batch 结束后还在验证集上算一遍损失和准确率：

```python
for epoch in range(6):
    losses = []
    for i, (batch_data, batch_label) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            batch_data = batch_data.to('cuda')
            batch_label = batch_label.to('cuda')

        log_probs = model(batch_data)
        loss = loss_function(log_probs, batch_label.transpose(0, -1))

        model.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # 当前 batch 的训练准确率
        corrects = log_probs.data.cpu().numpy().argmax(axis=1) == batch_label.data.cpu().numpy()
        train_accuracy = corrects.astype(np.long).mean()
```

**关于调参。** 材料里的基线是「够用就好」的配置：6 轮、Adam 默认学习率、不做学习率衰减。我们主要关注三个超参数的影响：序列长度 150（由数据最长值决定）、嵌入维度 300（字符级任务常用值）、dropout 0.5（二分类小数据集容易过拟合，需要较强正则）。如果要继续提升，可以试更小的卷积核数量、加早停、或者改用预训练韩语模型，但材料里的基线没有做这些。

### 2.4 模型测试与指标报告

训练时每个 batch 都打印训练损失、验证损失、训练准确率、验证准确率。我们重点看验证准确率是否随轮数稳定上升；如果训练准确率一直涨、验证准确率不涨甚至下降，就说明过拟合了。

训练完成后，切到评估模式，对测试集逐条预测：

```python
model.eval()

test_sentences = [list(d) for d in test_df['comments'].to_list()]
test_sentences_processed = process_sentences(test_sentences, word2idx, SENTENCE_LENGTH)

# 测试集没有标签，先塞全 0 占位，只为复用 Dataset
test_dataset = DatasetLoader(test_sentences_processed, [0] * len(test_sentences_processed))
test_dataloader = DataLoader(test_dataset, batch_size=1)

res = []
for batch_data, batch_label in test_dataloader:
    if torch.cuda.is_available():
        batch_data = batch_data.to('cuda')
    log_probs = model(batch_data)
    res.append(log_probs.data.cpu().numpy().argmax(axis=1)[0])
```

注意这里 batch_size 设成 1，是为了让结果顺序和测试集行顺序严格对应。如果用大 batch，虽然更快，但要额外确认 DataLoader 没有打乱顺序（`shuffle=False` 默认就是不打乱），否则预测和 id 会对不上。

然后把 0/1 还原成布尔值，并保存提交文件：

```python
test_df['label'] = res
print(test_df['label'].value_counts())   # 看预测分布是否合理

# 比赛要求的格式是 True/False
test_df['label'] = test_df['label'].map(lambda x: True if x else False)
test_df.to_csv('/kaggle/working/prediction.csv', index=None, header=True)
```

材料里没有给出最终的验证准确率数值，所以这里不编造具体分数。我们能确认的是评估方式：用验证集准确率判断模型好坏，用测试集预测分布做合理性检查——如果模型把九成以上样本都判成同一类，通常说明它没学到东西，只是偏向多数类。

## 三、总结

本文中我们完成了韩语性别偏见检测的 CharCNN 基线。我们首先读取训练、验证、测试三个文件，确认任务是二分类、评论最长约 150 个字符。其次把布尔标签转成 0/1，按字符切分文本，只用训练集构建字符词表，并统一截断或补齐到 150 长度。接着用 3/4/5 三种卷积核的 CharCNN 抓取局部字符组合，配合 dropout 和 Adam 训练 6 轮。最后在验证集上监控准确率，并对测试集逐条预测，把结果还原成 True/False 写出提交文件。

这个基线最大的优点是简单、跑得快，字符级处理也避开了韩语分词难的问题。它的短板是只看局部字符片段，无法理解长距离语义；如果评论的偏见藏在上下文里，CharCNN 就容易漏判。要进一步提升，换用预训练韩语语言模型是更直接的方向。
