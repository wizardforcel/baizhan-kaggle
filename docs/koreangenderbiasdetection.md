# Korean Gender Bias Detection

## 一、比赛简介

### 1.1 比赛背景
本比赛旨在检测韩国语文本中的性别偏见。性别偏见在招聘、广告、社交评论等场景中常见，通过自动化检测可以降低偏见影响。该任务使用句子级二分类，判断文本是否包含性别偏见。

### 1.2 比赛目标
输入一段韩国语评论，输出是否含有性别偏见。训练集文件为 `train.gender_bias.binary.csv`，开发集为 `dev.gender_bias.binary.csv`，均包含 `comments` 文本和 `label` 标签。测试集只有 `comments`，需要生成预测标签。

### 1.3 评价标准
比赛采用分类准确率评估，预测正确率越高越好。

## 二、解题过程

我们采用字符级 CNN 作为 baseline，字符级处理不依赖词表，对韩国语形态变化更鲁棒。

### 2.1 数据探索（EDA）
首先确认输入目录的文件。

```python
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

读取训练和开发集，查看前几行。

```python
import pandas as pd
train_df = pd.read_csv('/kaggle/input/koreangenderbiasdetection/train.gender_bias.binary.csv')
dev_df = pd.read_csv('/kaggle/input/koreangenderbiasdetection/dev.gender_bias.binary.csv')
train_df.head()
```

标签为布尔值，转换为 0/1 编码。

```python
train_df['label'] = train_df['label'].map(lambda x: 1 if x else 0)
dev_df['label'] = dev_df['label'].map(lambda x: 1 if x else 0)
```

### 2.2 数据清洗
将句子拆分为字符列表，保持原样不做清洗。

```python
train_sentences = [list(d) for d in train_df['comments'].to_list()]
dev_sentences  = [list(d) for d in dev_df['comments'].to_list()]
```

构建字符词表，加入未知符和填充符。

```python
from collections import Counter
def make_dictionary(sentences, vocabulary_size=None, initial_words=['<UNK>', '<PAD>']):
    counter = Counter()
    for words in sentences:
        counter.update(words)
    vocab_words = counter.most_common(vocabulary_size or len(counter))
    for w in initial_words:
        vocab_words.insert(0, (w, 0))
    word2idx = {word:idx for idx, (word, count) in enumerate(vocab_words)}
    return word2idx, {v:k for k,v in word2idx.items()}
```

句子统一截断或填充到 150 字符。

```python
SENTENCE_LENGTH = 150
train_sentences_processed = process_sentences(train_sentences, word2idx, sentence_length=SENTENCE_LENGTH)
```

### 2.3 模型训练与超参数调优
使用 CharCNN 模型，嵌入后经过多卷积核和最大池化。

```python
import torch.nn as nn
class CharCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, kernel_num, kernel_sizes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_size, kernel_num, k) for k in kernel_sizes])
        self.linear = nn.Linear(1140, output_size)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        embedded = self.embedding(x).transpose(1,2)
        pools = [F.max_pool1d(conv(embedded), kernel_size).view(x.size(0), -1) for conv, kernel_size in zip(self.convs, kernel_sizes)]
        concat = torch.cat(pools, dim=1)
        return self.linear(self.dropout(concat))
```

模型参数和优化器初始化。

```python
model = CharCNN(len(word2idx), embedding_size=300, output_size=2, kernel_num=10, kernel_sizes=[3,4,5])
optimizer = torch.optim.Adam(model.parameters())
loss_function = nn.NLLLoss()
```

训练 6 个 epoch，每轮报告训练和验证准确率。

```python
for epoch in range(6):
    for batch_data, batch_label in train_dataloader:
        log_probs = model(batch_data)
        loss = loss_function(log_probs, batch_label)
        model.zero_grad(); loss.backward(); optimizer.step()
```

### 2.4 模型测试与指标报告
模型训练完成后切换到评估模式，对测试集预测。

```python
model.eval()
test_df = pd.read_csv('/kaggle/input/koreangenderbiasdetection/test.gender_bias.no_label.csv')
test_sentences = [list(d) for d in test_df['comments'].to_list()]
test_sentences_processed = process_sentences(test_sentences, word2idx, sentence_length=SENTENCE_LENGTH)
```

逐批推理并收集结果。

```python
for batch_data, _ in test_dataloader:
    log_probs = model(batch_data)
    pred = log_probs.argmax(dim=1).cpu().numpy()
    res.append(pred[0])
test_df['label'] = res
test_df.to_csv('/kaggle/working/prediction.csv', index=None)
```

## 三、总结

本文中我们完成了 Korean Gender Bias Detection 的字符级 CNN baseline。我们首先对数据进行字符切分和词表构建，其次使用 CharCNN 模型进行训练，最后对测试集生成预测结果。字符级模型在韩国语上对形态变化鲁棒，是一个有效的入门方案。
