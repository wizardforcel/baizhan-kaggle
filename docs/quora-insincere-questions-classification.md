# Quora 不真诚问题分类（Insincere Questions Classification）

## 一、比赛简介

### 1.1 比赛背景

Quora 是一个大型问答社区，任何人都可以提问。绝大多数问题是正常的求知，但也有一部分问题**不是为了求知**：它们可能是煽动性的、带有歧视的、纯属挑衅的，或者根本不是一个真问题。这类问题会伤害社区氛围，也会浪费回答者的时间。

这个比赛要做的，就是训练一个模型，自动判断一条问题是不是「不真诚」（insincere）。数据来自 Quora 线上真实的问题，人工标注了每条问题的标签。它是一个典型的**短文本二分类**任务。

难点有三个：

1. **类别极不平衡**。真诚问题占绝大多数，不真诚的只占很小一部分。模型如果偷懒「全猜真诚」，准确率能到 93% 以上，但这样的模型毫无用处。
2. **语义微妙**。很多不真诚问题表面上语法通顺、用词正常，只是立场偏激或带有预设，靠关键词很难抓。
3. **文本短、噪声多**。拼写错误、网络用语、表情符号、HTML 标签都会出现。

### 1.2 比赛目标

训练集 `train.csv` 有三列：

- `qid`：问题编号；
- `question_text`：问题正文；
- `target`：标签，0 表示真诚，1 表示不真诚。

测试集 `test.csv` 只有 `qid` 和 `question_text`。我们要为每条测试问题输出一个 `prediction`，取值 0 或 1。比赛还提供了 `embeddings.zip`，里面是预训练好的词向量，可以拿来初始化模型的嵌入层。

### 1.3 评价标准

比赛使用 **F1 分数**，并且只算「不真诚」这一类（正类）。F1 是精确率和召回率的调和平均：

```
F1 = 2 * (precision * recall) / (precision + recall)
```

直白地说，它同时惩罚两种错误：**把真诚问题误判成不真诚**（精确率下降）、**把不真诚问题漏掉**（召回率下降）。因为正类只占 6% 左右，准确率会被负类淹没，所以必须看 F1。分数越高越好，满分是 1。

## 二、解题过程

我们的主线分两层：先做一个**词袋 + 线性模型**的基线，把流程跑通、把分数标定出来；再上**预训练词向量 + 双向循环网络**的主模型，用类别加权和阈值搜索把 F1 抬上去。

### 2.1 数据探索（EDA）

我们首先把数据读进来，看规模、看标签分布：

```python
import pandas as pd

train = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test  = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')

print('训练集形状 =', train.shape)
print('测试集形状 =', test.shape)

# 标签分布：0 真诚，1 不真诚
print(train['target'].value_counts())
```

输出显示训练集有 **1,306,122** 行：`target=0` 有 **1,225,312** 条，`target=1` 只有 **80,810** 条。正类占比约 **6.2%**，负类约 **93.8%**。这个比例意味着，一个「永远预测 0」的模型准确率有 93.8%，但 F1 是 0——所以我们后面所有的评估都只看 F1，不看准确率。

接着看文本长度：

```python
# 按字符数和词数分别统计，决定后面截断到多长
train['n_chars'] = train['question_text'].str.len()
train['n_words'] = train['question_text'].str.split().str.len()

print(train[['n_chars', 'n_words']].describe(percentiles=[.5, .9, .99]))
print('最长字符数 =', train['n_chars'].max())   # 1017
print('最长词数   =', train['n_words'].max())   # 134
```

中位数只有几十个词，但最长有 **1017 个字符、134 个词**。这说明大部分问题很短，只有极少数特别长。为了控制计算量，我们会把序列截断到固定长度（后面选的是 72），超长的部分丢掉，超短的部分补零。

我们还检查了文本里的「脏东西」：

```python
import re

# 有多少条问题包含数字
has_digit = train['question_text'].str.contains(r'\d')
print('含数字比例 = {:.4%}'.format(has_digit.mean()))     # 11.2561%

# 有多少条包含标点
has_punct = train['question_text'].str.contains(r'[^\w\s]')
print('含标点比例 = {:.4%}'.format(has_punct.mean()))     # 99.9943%

# 有没有空值、重复
print('空值数 =', train['question_text'].isna().sum())
print('重复问题数 =', train['question_text'].duplicated().sum())
```

结果是：**11.2561%** 的问题含数字，**99.9943%** 含标点，空值和重复都极少。另外 EDA 里还发现少量 HTML 标签和 URL、一些全大写单词，以及 `©`、`™`、`®` 这类版权符号。这些信息决定了清洗策略：标点不能一股脑删掉（问号本身可能是有用信号），但 HTML 标签和 URL 应该清掉。

### 2.2 数据清洗

我们做的清洗分四步，原则是「**保留语义，去掉噪声**」。

第一步，统一大小写并去掉多余空白。大小写对语义影响不大，统一成小写可以显著缩小词表：

```python
import re

def clean_text(s):
    s = s.lower()                            # 统一小写，减少词表规模
    s = re.sub(r'<[^>]+>', ' ', s)           # 去掉 HTML 标签
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)   # 去掉 URL
    s = re.sub(r'\s+', ' ', s).strip()       # 多个空格压成一个
    return s

train['clean'] = train['question_text'].apply(clean_text)
test['clean']  = test['question_text'].apply(clean_text)
```

第二步，把**重复的标点**替换成一个占位词。连续三个感叹号 `!!!` 和单个感叹号情绪强度不同，但如果直接删掉，这个信息就没了。材料里的做法是用 `multiExclamation`、`multiQuestion`、`multiStop` 这样的标签替代：

```python
def normalize_punct(s):
    # 连续感叹号 -> 一个标记词，保留「情绪强烈」这个信号
    s = re.sub(r'(!)\1+', ' multiExclamation ', s)
    # 连续问号 -> 一个标记词
    s = re.sub(r'(\?)\1+', ' multiQuestion ', s)
    # 连续句号 -> 一个标记词
    s = re.sub(r'(\.)\1+', ' multiStop ', s)
    return s

train['clean'] = train['clean'].apply(normalize_punct)
```

第三步，**数字处理**。数字在很多问题里是具体信息（年份、年龄、价格），但在词袋模型里会撑大词表。我们的处理是把纯数字统一替换成 `<num>`，既保留「这里有数字」的信号，又不让每个数字都成为一个独立词：

```python
def replace_numbers(s):
    # 把连续数字替换成统一占位符
    return re.sub(r'\d+', ' <num> ', s)

train['clean'] = train['clean'].apply(replace_numbers)
```

第四步，**分词与截断**。我们按空格切词，并把每条问题截断/补齐到 72 个 token。72 这个数字来自材料里主模型的配置 `maxlen=72`，它覆盖了绝大多数问题的长度，又不至于让序列太长：

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAXLEN = 72          # 主模型统一使用的序列长度
VOCAB_MINCOUNT = 5   # 出现少于 5 次的词一律当作未知词

tokenizer = Tokenizer(num_words=None, lower=False, oov_token='<unk>')
tokenizer.fit_on_texts(train['clean'])

# 建立词表：只保留高频词，低频词不占位置
word_index = {w: i + 1 for i, (w, c) in enumerate(tokenizer.word_counts.items())
              if c >= VOCAB_MINCOUNT}

X_train = pad_sequences(
    tokenizer.texts_to_sequences(train['clean']),
    maxlen=MAXLEN, padding='post', truncating='post')
X_test = pad_sequences(
    tokenizer.texts_to_sequences(test['clean']),
    maxlen=MAXLEN, padding='post', truncating='post')

print('词表大小 =', len(word_index), '序列形状 =', X_train.shape)
```

`padding='post'` 表示在**末尾**补零，`truncating='post'` 表示超长时从**末尾**截断。这两个选择要和后面的模型结构配合——如果用双向 LSTM，末尾补零对开头的信息影响最小。

### 2.3 模型训练与超参数调优

我们先跑了一个**逻辑回归基线**：用 TF-IDF 把文本变成稀疏向量，再喂给 `LogisticRegression`。材料里这个基线的验证 F1 是 **0.6634**，XGBoost 基线是 **0.6405**。这个数字很重要，它是我们判断后续深度学习模型有没有真正提升的标尺。

主模型的结构是「**预训练词向量 + 双向 LSTM + 全连接**」。先说词向量：比赛提供的 `embeddings.zip` 里有多个预训练文件，我们把它们拼起来，再随机采样 400 维，避免维度太大拖慢训练：

```python
import numpy as np

def build_embedding_matrix(word_index, emb_paths, sample_dim=400):
    """把多个预训练词向量拼起来，取前 sample_dim 维"""
    vectors = []
    for path in emb_paths:
        # 每个文件是一个 {词: 向量} 的字典，这里简化为已加载好的形式
        vectors.append(load_vec_file(path))

    # 按词对齐后横向拼接：不同来源的词向量互补
    full = {}
    for w, i in word_index.items():
        parts = [v.get(w) for v in vectors]
        # 任何一个来源没有这个词，就用零向量占位
        parts = [p if p is not None else np.zeros_like(vectors[0][next(iter(vectors[0]))])
                 for p in parts]
        full[w] = np.concatenate(parts)

    # 固定随机种子后随机选 400 维，保证每次实验选的维度一致
    rng = np.random.RandomState(1029)
    n_total = len(next(iter(full.values())))
    dims = rng.permutation(n_total)[:sample_dim]

    # 第 0 行留给 padding，必须是全零
    matrix = np.zeros((len(word_index) + 1, sample_dim), dtype=np.float32)
    for w, i in word_index.items():
        matrix[i] = full[w][dims]
    return matrix
```

这里有两个细节值得说明：

- **第 0 行必须全零**。Keras 的嵌入层把索引 0 当作「填充位」，如果它不为零，补零的位置就会带上莫名其妙的语义。
- **随机采样 400 维**而不是取前 400 维。多个词向量拼接后，前几百维可能全来自同一个来源，随机采样能让不同来源的维度混合，信息更均衡。

接着搭模型。因为正负样本比例约 1:15，我们给每个样本一个**权重**：正类权重是 `1/正类数量`，负类权重是 `1/负类数量`。这样在算损失时，少数类不会被多数类淹没：

```python
import torch
import torch.nn as nn

# 类别权重：数量越少，权重越大
n0 = (train['target'] == 0).sum()
n1 = (train['target'] == 1).sum()
weight = {0: 1.0 / n0, 1: 1.0 / n1}

class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden=64):
        super().__init__()
        n_words, dim = embedding_matrix.shape
        # 用预训练矩阵初始化嵌入层，并冻结它，避免小数据把词向量带偏
        self.emb = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix), freeze=True, padding_idx=0)
        # 双向 LSTM：从左到右、从右到左各读一遍，再拼起来
        self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.2)
        # 双向输出是 2*hidden 维，接一个二分类输出
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        e = self.emb(x)                  # (batch, maxlen, dim)
        out, _ = self.lstm(e)            # (batch, maxlen, 2*hidden)
        # 只取最后一个时间步，代表整句的汇总表示
        h = out[:, -1, :]
        return self.fc(self.drop(h)).squeeze(-1)
```

训练配置沿用材料里的默认值：**batch size 512、5 个 epoch、学习率 1e-3、5 折交叉验证、随机种子 1029**。损失函数用 `BCEWithLogitsLoss`，它把 sigmoid 和交叉熵合在一起，数值上更稳定：

```python
import torch.optim as optim

model = BiLSTMClassifier(embedding_matrix)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    for xb, yb, wb in train_loader:       # wb 是每个样本的类别权重
        optimizer.zero_grad()
        logits = model(xb)
        # 逐样本加权：少数类的误差被放大，模型不敢忽略它
        loss = (criterion(logits, yb) * wb).mean()
        loss.backward()
        optimizer.step()
    print(f'epoch {epoch + 1} loss = {loss.item():.4f}')
```

**阈值搜索**是这类不平衡问题的关键一步。模型输出的是 0～1 的概率，默认以 0.5 为界；但在正类很少时，0.5 往往偏高，会漏掉大量不真诚问题。我们在验证集上扫描一串阈值，挑 F1 最高的那个：

```python
from sklearn.metrics import f1_score
import numpy as np

def search_threshold(y_true, y_prob):
    """在 0.1~0.9 之间找使 F1 最大的阈值"""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        pred = (y_prob > t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    print(f'最佳阈值 = {best_t:.2f}, F1 = {best_f1:.4f}')
    return best_t
```

材料里有一份阈值实验的记录：把阈值从 0.5 调到 **0.81** 时，验证 F1 是 **0.5899**。这个结果说明降低阈值（让更多样本被判为正类）确实能提升召回，但精确率会掉下来，最终 F1 未必更高。所以阈值不能凭感觉调，必须**在验证集上搜**，而且要在交叉验证的每一折上分别确认，避免过拟合到某一折。

### 2.4 模型测试与指标报告

测试阶段我们做三件事：算验证 F1、生成提交、检查分布。

```python
from sklearn.metrics import f1_score, precision_score, recall_score

model.eval()
with torch.no_grad():
    prob = torch.sigmoid(model(X_val_tensor)).numpy()

pred = (prob > best_t).astype(int)

print('F1        =', f1_score(y_val, pred))
print('Precision =', precision_score(y_val, pred))
print('Recall    =', recall_score(y_val, pred))
```

三个指标要一起看：如果 F1 还行但召回很低，说明模型太保守；如果精确率很低，说明误伤太多。比赛只看 F1，但拆开看才知道模型偏在哪一边。

接着对测试集预测并写提交文件：

```python
model.eval()
with torch.no_grad():
    test_prob = torch.sigmoid(model(X_test_tensor)).numpy()

submission = pd.DataFrame({
    'qid': test['qid'],
    'prediction': (test_prob > best_t).astype(int),   # 用验证集选出的阈值
})
submission.to_csv('submission.csv', index=False)

# 检查：行数对不对、正类比例是否离谱
print('提交行数 =', len(submission), '测试集行数 =', len(test))
print('预测正类比例 = {:.2%}'.format(submission['prediction'].mean()))
```

最后一项检查很实用：如果预测出的正类比例和训练集的 6.2% 差太远（比如 30%），基本可以断定阈值设错了或者数据泄漏了。**提交前看一眼分布**，能挡住很多低级错误。

---

## 三、总结

本文中我们完成了 Quora 不真诚问题分类的完整流程。我们首先用 EDA 摸清数据：130 万条训练样本、正类只有 80,810 条（约 6.2%）、最长问题 1017 个字符，并确认 11.2561% 含数字、99.9943% 含标点；其次我们做了保留语义的清洗——去 HTML 和 URL、统一小写、把重复标点替换成标记词、把数字换成占位符；接着我们先跑逻辑回归基线（验证 F1 0.6634）标定水平，再搭「多源预训练词向量 + 双向 LSTM」主模型，用类别权重对抗不平衡，配置为 maxlen 72、词频下限 5、batch 512、5 轮、学习率 1e-3；然后我们在验证集上搜索分类阈值，并同时报告 F1、精确率、召回率；最后我们生成提交并检查预测分布。

这个比赛给我们最大的启发是：**不平衡数据上，「准确率」是个骗人的指标**。一个 93.8% 准确率的模型可能一个正类都抓不到。真正要做的是三件事——换指标（F1）、改损失（类别加权）、调决策边界（阈值搜索）。这三步做完，模型才算真的在解决业务问题，而不是在刷一个好看的数字。
