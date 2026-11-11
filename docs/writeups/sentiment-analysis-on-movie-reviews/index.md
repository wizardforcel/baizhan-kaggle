# 电影评论情感分析（Sentiment Analysis on Movie Reviews）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的数据来自著名影评网站 Rotten Tomatoes（中文常叫「烂番茄」）。网站上的影评原本是一句句完整的话，比赛组织方用斯坦福句法分析器（Stanford parser）把每个句子拆成了很多更短的短语。比如一句话会被拆出完整句、半句、几个词，甚至单个单词。每条短语都有自己的 `PhraseId`，同一句话拆出来的短语共享一个 `SentenceId`。为了避免重复，像 `A`、`series` 这种很短又常见的短语只保留一次。

所以这个比赛的任务，就是判断一条短语表达的情感是偏负面还是偏正面。它和我们平时看到的「给整篇影评打好评/差评」不太一样：这里面对的是很碎的片段，很多短语只有一两个词，缺少上下文，判断起来更难。材料里也特别提到，否定句、讽刺、过于简短的表达、语言歧义都会让任务变得很有挑战。

### 1.2 比赛目标

训练集 `train.tsv` 有 156060 行、4 列：

- `PhraseId`：短语编号，每条短语唯一；
- `SentenceId`：句子编号，同一句话拆出的短语共用；
- `Phrase`：短语原文；
- `Sentiment`：情感标签，取值 0 到 4。

五个标签的含义分别是：

| 标签 | 含义 |
| --- | --- |
| 0 | negative（负面） |
| 1 | somewhat negative（有点负面） |
| 2 | neutral（中性） |
| 3 | somewhat positive（有点正面） |
| 4 | positive（正面） |

测试集 `test.tsv` 有 66292 行，只有前三列，没有 `Sentiment`。我们要给每条测试短语预测一个 0 到 4 的标签，并按照 `PhraseId, Sentiment` 的格式生成提交文件。`sampleSubmission.csv` 给出了提交样例。

### 1.3 评价标准

比赛使用**分类准确率（Accuracy）**作为评价指标。做法很直接：对每条测试短语预测一个类别，预测正确的条数除以总条数，就是最终得分。得分越高越好，满分是 1.0。

需要提前知道一个「偷懒基线」：训练集里情感为中性（标签 2）的短语有 79582 条，大约占一半。也就是说，如果不管输入是什么，全部猜成中性，也能拿到接近 0.5 的准确率。因此在这个比赛里，0.5 左右的分数并不说明模型学到了东西，真正有意义的提升要看能不能明显超过这个基线。

---

## 二、解题过程

整场比赛我们走了两步：先做一个基于词袋和随机森林的快速基线，确认数据和流程没问题；再上 BERT 做真正的微调。下面按数据探索、数据清洗、模型训练与调参、测试与指标报告四部分来说。

### 2.1 数据探索（EDA）

我们首先把两个 TSV 文件读进来，看看数据规模和列的含义。注意这两个文件是用制表符分隔的，不是常见的逗号分隔 CSV，所以 `read_csv` 要写 `sep='\t'`。

```python
import pandas as pd

# 训练集和测试集都是 TSV（制表符分隔），必须显式指定分隔符
train_df = pd.read_csv('train.tsv', sep='\t')
test_df = pd.read_csv('test.tsv', sep='\t')

# 查看规模：训练集 156060 行 4 列，测试集 66292 行 3 列
print(train_df.shape)   # (156060, 4)
print(test_df.shape)    # (66292, 3)

# 看列名、数据类型和是否有缺失值
print(train_df.info())
train_df.head()
```

打印结果确认：`PhraseId`、`SentenceId`、`Sentiment` 都是整数，`Phrase` 是字符串，训练集没有缺失值。测试集只少一列 `Sentiment`。这说明数据本身比较干净，主要工作量在文本建模上。

其次，我们统计五个情感标签的分布，判断类别是否均衡：

```python
# 按标签值 0~4 排序后统计数量
label_counts = train_df['Sentiment'].value_counts().sort_index()
print(label_counts)

# 算一下每个标签的占比，方便和随机猜测比较
print(label_counts / len(train_df))
```

从统计结果看，中性（2）最多，有 79582 条，接近一半；其余四类少一些，但都真实存在。这个分布告诉我们两件事：第一，模型很容易偏向中性，因为猜中性「性价比」最高；第二，只看总体准确率会被中性类主导，所以后面要同时关注模型是不是真的学会了区分正负情感。

接着，我们看短语长度。因为后面用 BERT 时要设定一个固定的最大长度，太短会截掉信息，太长又浪费显存：

```python
# 按空格切分，统计每条短语包含多少个词
train_df['word_count'] = train_df['Phrase'].str.split().str.len()

# 看最短、最长、平均值和分位数
print(train_df['word_count'].describe())

# 顺便看看最短的短语长什么样
print(train_df.nsmallest(5, 'word_count')[['Phrase', 'Sentiment']])
```

我们发现短语整体很短，大量样本只有一两个词，比如 `A`、`series`，长句子相对少。这解释了比赛难在哪里：一个孤零零的词，往往既可以是中性，也可以带情感，必须结合它原来所在的句子才能判断。但比赛把句子打散了，我们只能看到短语本身，所以模型能利用的上下文非常有限。

### 2.2 数据清洗

我们接着做文本清洗。这里要区分两条路线：如果做传统词袋模型，需要把文本规整成统一的词；如果做 BERT 微调，反而不应该清洗得太狠。

先看传统路线。我们把所有字母转小写，去掉标点，并把连续数字替换成一个统一符号 `num`：

```python
import re
import string

def clean_text(text):
    # 统一转小写，避免 Good 和 good 被当成两个不同的词
    text = text.lower()
    # 去掉标点：标点对情感判断帮助有限，还会让词表变大
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 把数字统一成 num，减少大量只出现一两次的稀有 token
    text = re.sub(r'\d+', 'num', text)
    return text

# 对训练集和测试集使用同一套清洗规则
train_df['Phrase_clean'] = train_df['Phrase'].apply(clean_text)
test_df['Phrase_clean'] = test_df['Phrase'].apply(clean_text)

# 检查清洗前后的效果
print(train_df[['Phrase', 'Phrase_clean']].head())
```

这段清洗的思路是「减少噪音、压缩词表」。对词袋模型来说，`Good` 和 `good` 没有区别，标点也不表达情感，统一处理可以避免模型把同一类信息拆成很多份。

但在 BERT 路线里，我们保留了原始 `Phrase`，没有用 `Phrase_clean`。原因有两个：一是 BERT 在预训练时见到的就是带标点、带大小写的自然文本，强行清洗反而和它的预训练分布不一致；二是否定词和标点有时会影响情感，比如 `not good` 和 `not, good` 在语义上并不完全相同。材料里的 BERT 方案也是直接拿 `Phrase` 列做分词，我们沿用了这个做法。

另外，`PhraseId` 和 `SentenceId` 这两列对预测情感没有直接帮助，传统模型里可以删掉；但在生成提交文件时，`PhraseId` 又要原样保留，所以我们只是不把它当特征，并没有从表里丢掉。

### 2.3 模型训练与超参数调优

我们先搭一个快速基线：用 TF-IDF 把短语变成向量，再喂给随机森林。它的作用是帮我们确认「文本到标签」这条链路能跑通，同时给出一个可以对比的分数。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# TF-IDF 会把词频和「这个词有多稀有」结合起来打分
# max_features=3000 表示只保留最常见的 3000 个词，控制特征维度
vectorizer = TfidfVectorizer(max_features=3000)

# 关键：只在训练集上 fit，测试集只能 transform，防止测试信息泄漏
X_train = vectorizer.fit_transform(train_df['Phrase_clean'])
X_test = vectorizer.transform(test_df['Phrase_clean'])

# 随机森林由很多棵决策树投票，训练快，适合做第一版基线
rf = RandomForestClassifier()
rf.fit(X_train, train_df['Sentiment'])

# 先看训练集上的拟合准确率
print('训练集准确率:', rf.score(X_train, train_df['Sentiment']))
```

这个基线跑起来很快，但它有两个明显短板。第一，词袋模型把每个词当成独立的，`not good` 和 `good not` 在它眼里差不多，否定关系学不到。第二，短语太短，词袋里可用的线索本来就少。因此我们决定把主要精力放在 BERT 上。

BERT 是一个预训练语言模型，可以粗略理解成「已经读过大量英文文本、对词和上下文关系有基础认识」的模型。我们不需要从零教它英语，只需要用比赛数据微调（fine-tune）它，让它把知识迁移到五分类情感任务上。

首先，我们用 BERT 自带的分词器把短语切成 subword（子词）编号，并统一补齐或截断到 64 个 token：

```python
from transformers import AutoTokenizer

# bert-base-uncased 是英文基础版 BERT，uncased 表示不区分大小写
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# batch_encode_plus 可以一次性编码整列文本
# padding='max_length' 把短句补到 64，truncation=True 把长句截到 64
encoded = tokenizer.batch_encode_plus(
    train_df['Phrase'].tolist(),
    padding='max_length',
    max_length=64,
    truncation=True,
    return_attention_mask=True
)
```

这里选 64 而不是 128 或 512，是因为前面的 EDA 已经看到短语普遍很短，64 足够覆盖绝大多数样本，同时能明显减少显存占用和训练时间。`attention_mask` 用来告诉模型哪些位置是真实单词、哪些位置只是补齐用的空白。

接着，我们把编码结果包装成 PyTorch 的 `Dataset`，并切出训练集和验证集：

```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class MovieReviewsDataset(Dataset):
    def __init__(self, df, max_len, test_only=False):
        self.text = df['Phrase'].tolist()
        self.test_only = test_only
        if not test_only:
            self.labels = df['Sentiment'].values

        # 在初始化时一次性编码整列，避免每个 epoch 重复分词
        self.encode = tokenizer.batch_encode_plus(
            self.text,
            padding='max_length',
            max_length=max_len,
            truncation=True,
            return_attention_mask=True
        )

    def __getitem__(self, i):
        input_ids = torch.tensor(self.encode['input_ids'][i])
        attention_mask = torch.tensor(self.encode['attention_mask'][i])
        if self.test_only:
            return input_ids, attention_mask
        return input_ids, attention_mask, torch.tensor(self.labels[i])

    def __len__(self):
        return len(self.text)

max_len = 64
train_dataset = MovieReviewsDataset(train_df, max_len)
test_dataset = MovieReviewsDataset(test_df, max_len, test_only=True)

# 80% 训练、20% 验证；固定随机种子，保证每次切分结果一致
lengths = [int(len(train_dataset) * 0.8), int(len(train_dataset) * 0.2)]
train_dataset, valid_dataset = random_split(
    train_dataset,
    lengths=lengths,
    generator=torch.Generator().manual_seed(42)
)

# batch_size=128 在 16GB 显存的 GPU 上比较稳妥
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)
```

然后我们定义模型。BERT 主体负责把一串 token 变成一个 768 维的语义向量，我们在这个向量后面接一个线性层，把 768 维映射成 5 个分数，分别对应五个情感类别：

```python
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained('bert-base-uncased')
        # 只加载 BERT 主体，不加载原来做「完形填空」的预训练输出头
        self.bert = AutoModel.from_pretrained('bert-base-uncased', config=config)
        # BERT 的隐藏层是 768 维，输出层改成 5 类
        self.classifier = nn.Linear(config.hidden_size, 5)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # out[1] 是 pooler_output，即 [CLS] 位置的向量，可当作整句摘要
        pooled = out[1]
        return self.classifier(pooled)

model = BertClassifier().to(device)
```

训练时我们用 AdamW 优化器，学习率设为 `2e-5`。这个值看起来很小，但 BERT 微调通常就用这个量级：太大容易把预训练学到的知识冲掉，太小又学不动。损失函数用交叉熵，它专门处理「多选一」的分类问题。我们只训练 3 个 epoch，因为 BERT 在小数据集上很容易过拟合，轮数多了验证集反而会掉。

```python
from transformers import AdamW

# BERT 微调常用小学习率，2e-5 是比较常见的起点
optimizer = AdamW(model.parameters(), lr=2e-5)
# 五分类任务用交叉熵损失
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()          # 训练模式，启用 dropout
    total_loss = 0.0
    for input_ids, attention_mask, target in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)

        optimizer.zero_grad()                 # 清掉上一轮的梯度
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, target)      # 计算损失
        loss.backward()                       # 反向传播求梯度
        optimizer.step()                      # 按梯度更新参数

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'第 {epoch + 1} 轮，平均损失 {avg_loss:.4f}')
```

### 2.4 模型测试与指标报告

训练过程中，我们在每个 epoch 结束后切到评估模式，在验证集上算准确率。评估时不需要更新参数，所以用 `torch.no_grad()` 关掉梯度计算，既省显存也更快：

```python
import numpy as np
from sklearn.metrics import accuracy_score

model.eval()               # 评估模式，关闭 dropout
all_preds, all_labels = [], []

with torch.no_grad():
    for input_ids, attention_mask, target in valid_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = model(input_ids, attention_mask)
        # argmax 取分数最高的那个类别
        batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(batch_preds)
        all_labels.extend(target.numpy())

print('验证集准确率:', accuracy_score(all_labels, all_preds))
```

我们实际看到的三轮验证准确率分别是约 **0.691**、**0.704**、**0.696**。第二轮最高，第三轮略微回落，这正是前面说的小数据集容易过拟合的表现。和「全部猜中性」的约 0.5 基线相比，0.70 说明 BERT 确实学到了情感信息，而不是只靠类别分布蒙对。另一方面，第三轮下降也提醒我们：不是训练轮数越多越好，后面如果继续调参，应该用验证集早停，而不是盲目加 epoch。

最后，我们用训练好的模型对测试集逐批预测，并生成提交文件。这里最容易出错的地方是列名和行数：提交文件必须恰好有 `PhraseId` 和 `Sentiment` 两列，行数必须和测试集 66292 行一致。

```python
model.eval()
predictions = []

with torch.no_grad():
    for input_ids, attention_mask in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = model(input_ids, attention_mask)
        predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())

# 提交格式：PhraseId 必须和测试集一一对应，Sentiment 是 0~4 的整数
submission = pd.DataFrame({
    'PhraseId': test_df['PhraseId'],
    'Sentiment': predictions
})
submission.to_csv('submission.csv', index=False)

# 检查行数和前几行，确认没有错位
print(submission.shape)
print(submission.head())
```

生成文件后我们做了两项检查：一是行数等于 66292，二是 `PhraseId` 的顺序和测试集完全一致。比赛里因为提交格式错误而拿不到分数的情况并不少见，这两步虽然琐碎，但值得每次都做。

---

## 三、总结

本文中我们完成了 Rotten Tomatoes 电影评论情感分析的完整流程。我们首先读取 TSV 数据，确认训练集有 156060 条短语、测试集有 66292 条，并统计出五个情感标签的分布，发现中性类有 79582 条、接近一半。其次我们做了针对性的文本清洗：传统模型路线统一小写、去标点、数字归一化，而 BERT 路线保留原始文本，避免破坏预训练模型熟悉的语言形式。接着我们先跑了一个 TF-IDF 加随机森林的快速基线，确认流程可用，然后把主要精力放在 BERT 微调上：用 `bert-base-uncased` 分词、统一到 64 个 token，接一个 5 类线性分类头，以 `2e-5` 的学习率训练 3 个 epoch。然后我们在验证集上得到约 0.70 的准确率，明显超过「全猜中性」的 0.5 基线。最后我们用模型对测试集预测，按 `PhraseId, Sentiment` 的格式生成了提交文件，并检查了行数和编号对齐。

这个比赛给我们最大的体会是：**同一个任务，预处理要跟着模型走**。词袋模型需要把文本洗得干净整齐，BERT 反而需要保留自然的标点和大小写；而评估时也不能只看总分，因为中性类占了一半，必须和多数类基线对比，才知道模型是不是真的学会了区分情感。
