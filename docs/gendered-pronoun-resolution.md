# Gendered Pronoun Resolution：性别代词消解（GAP）

## 一、比赛简介

### 1.1 比赛背景

在英语里，"he / she / his / her" 这类代词指代谁，通常人一眼就能看懂。但这句话："Sarah told her mother that **she** was tired" 里的 "she" 到底指 Sarah 还是她妈妈？这种歧义对人类来说可能要想一下，对机器来说更是 NLP 里出了名的老大难。这个任务叫 **代词消解（pronoun resolution / coreference resolution）**。

谷歌开源了一个专门用来考察这个问题的数据集 **GAP**（Gendered Ambiguous Pronouns），全部来自维基百科的真实句子，聚焦于**性别有歧义的代词**——也就是 A 和 B 两个人性别可能相同或相反，代词到底指谁，靠猜很容易出错。

每个样本给出一段文本、一个代词，以及两个人名 **A** 和 **B**（它们都出现在文本里，都标了起始位置）。任务要回答：这个代词指的是 A、B，还是**谁都不是（NEITHER）**。比赛要求输出三个概率，它们的和为 1。

这类任务讲究「先验」：纯粹靠两个候选人的位置距离（谁离代词近）就能得到不差的基线，但想拿高分必须真正理解句子的语义——这正好是 BERT 这类预训练语言模型的用武之地。

### 1.2 比赛目标

输入表格 `test_stage_1.tsv` / `test_stage_2.tsv`，每行有这些字段：

- `ID`：样本编号（形式如 `development-1`）；
- `Text`：英文文本；
- `Pronoun` 和 `Pronoun-offset`：代词本身，以及它在文本里的字符位置；
- `A`、`A-offset`、`B`、`B-offset`：两个候选人与各自的位置；
- `URL`：这句话来自哪篇维基百科文章。

输出是三列概率 `A`、`B`、`NEITHER`，三者相加为 1。

真实的 Google GAP 官方数据有 `gap-development.tsv`、`gap-validation.tsv`、`gap-test.tsv` 三份（很多人把这部分也读进来当训练数据，因为它是同一个数据源的原文，且带 `A-coref`/`B-coref` 标签）。测试集分两个阶段：阶段一是 2000 条，阶段二是 12359 条。

### 1.3 评价标准

比赛用 **log loss**（对数损失）评价，越低越好。它是分类问题里最常用的指标之一：对每一条样本，模型给**正确类别**打的概率越低，扣分越狠；如果你对 100% 有把握的样本只给了 0.6 的概率，一样会被重罚。

要注意的是：A、B、NEITHER 三列概率和真实标签的差距都会被计入，所以**模型必须把三列概率都校准好**，不能只把某一列概率塞满。1.0 左右是不带语义理解的特征模型的水平（比如只用位置、距离这些统计特征）；0.5 左右是微调 BERT 的基线水平；代码里做到 **0.486**，就已经是公开方案里相当不错的成绩了。

---

## 二、解题过程

我们参考了两条互补的路线：**先用一套纯手工特征 + 随机森林做基线**（验证任务理解没有问题），**再用 BERT 微调作为主线方案**。BERT 允许最多 512 个词元（token，即分词后的小片断），而我们给的文本平均长度不长——把句子稍作截断就能放进模型。

### 2.1 数据探索（EDA）

我们首先读数据、看列结构和标签分布：

```python
import pandas as pd

train = pd.read_csv('../input/test_stage_1.tsv', sep='\t')
print(train.shape)          # (2000, 9)：阶段一测试 2000 条
print(train.columns)
# ID Text Pronoun Pronoun-offset A A-offset B B-offset URL

print(train.iloc[0])
```

打印出来的第一条样本大致是这样的：

```
Text:   Zoe Telford -- played the police officer girlfriend of Simon, Maggie...
Pronoun:     her    Pronoun-offset:    274
A:   Cheryl Cassidy   A-offset:   191
B:   Pauline          B-offset:   207
```

从中可以看到任务的核心设定：**A 和 B 两个候选人都明确出现在了文本里，并且给定了字符级的偏移位置**。这决定了我们的特征工程可以围绕「位置」做文章，也决定了 BERT 方案必须把三个关键位置精确地暴露给模型。

官方 GAP 数据的标签列是 `A-coref` 和 `B-coref`（1 表示代词指代它）。我们把它转成三分类标签：

```python
# 从官方 gap 数据读标签：A-coref / B-coref / Neither
tmp = gap_df[['A-coref', 'B-coref']].copy()
tmp['Neither'] = ~(gap_df['A-coref'] | gap_df['B-coref'])   # 两者都不是
labels = tmp.values.astype(bool)
```

`NEITHER = 1 - A - B`，逻辑上三个标签只有一个是 1。从数据分布看，A 和 B 被标为正确的样本都不少，NEITHER 相对少见。这说明「谁离代词近」是一个有用的先验，但也意味着模型容易偷懒——只学「近的赢」，把语义信息丢掉。给高分方案的难点正是如何在「位置先验」和「语义理解」之间做平衡。

### 2.2 数据清洗与特征

#### 2.2.1 手工特征版的数据清洗

对于手工特征模型，我们做的最小清洗主要是**补全偏移量**和**计算相对位置**：

```python
def get_features(df):
    # 代词的结束位置 = 起始位置 + 长度
    df['Pronoun_end'] = df['Pronoun-offset'] + df['Pronoun'].map(len)
    df['A_end'] = df['A-offset'] + df['A'].map(len)
    df['B_end'] = df['B-offset'] + df['B'].map(len)

    # A、B 分别到代词的距离（带符号，负号表示在代词前面）
    df['A_dist'] = df['Pronoun-offset'] - df['A-offset']
    df['B_dist'] = df['Pronoun-offset'] - df['B-offset']

    # 谁在代词和候选之间夹着的"区间"里：代词两侧总共横跨多少字
    df['section_min'] = df[['Pronoun-offset', 'A-offset', 'B-offset']].min(axis=1)
    df['section_max'] = df[['Pronoun_end', 'A_end', 'B_end']].max(axis=1)

    # 文本区间里的单词数量（间隔越长，两个候选对代词的"竞争"越弱）
    df['word_count'] = df.apply(lambda r: len(
        r['Text'][r['section_min']:r['section_max']].split()), axis=1)
    return df
```

这里的一个技巧是**把候选名字替换成通用词**：因为候选名字千奇百怪，直接把原词给模型会让它记忆特定人名。我们把 A、B 名字替换成 `"subjectone"`、`"subjecttwo"`，再用 spaCy 统计「这两个词在句子里作为所有格（possessive，即 `his`/`her` 的用法）出现了几次」——所有格的出现和代词性别高度相关。

```python
import spacy
nlp = spacy.load('en_core_web_sm')

def count_possessive(text, word):
    doc = nlp(str(text))
    return len([tok for tok in doc
                if tok.text == word and tok.dep_ == 'poss'])

train['A_poss'] = train['Text'].map(lambda x: count_possessive(x, 'subjectone'))
train['B_poss'] = train['Text'].map(lambda x: count_possessive(x, 'subjecttwo'))
```

这一整套特征就是「看图说话」式的常识：**离得近、被所有格修饰、处在同一从句的人，更可能是代词指代的对象**。

#### 2.2.2 BERT 版的预处理：插入标记

BERT 把整句话切成词元，但候选和代词各自在原文里的位置是**字符级**的，分词之后编号就对应不上了。Matei Ionita 的方案用了一个非常聪明的办法：**在把文本送给分词器之前，先在这三个关键位置插入特殊标记 `[A]`、`[B]`、`[P]`**，分词之后这些标记占一个独立的词元，位置就变得可知了：

```python
def insert_tag(row):
    """在 A、B、代词前插入特殊标记，方便分词后定位"""
    to_insert = sorted(
        [(row['A-offset'], ' [A] '),
         (row['B-offset'], ' [B] '),
         (row['Pronoun-offset'], ' [P] ')],
        key=lambda x: x[0], reverse=True)   # 从后往前插，避免破坏偏移
    text = row['Text']
    for offset, tag in to_insert:
        text = text[:offset] + tag + text[offset:]
    return text

def tokenize(text, tokenizer):
    """返回分词结果，以及 [A]、[B]、[P] 三个标记各自的位置"""
    entries, final_tokens = {}, []
    for token in tokenizer.tokenize(text):
        if token in ('[A]', '[B]', '[P]'):
            entries[token] = len(final_tokens)   # 只记位置，不放进行列
            continue
        final_tokens.append(token)
    return final_tokens, (entries['[A]'], entries['[B]'], entries['[P]'])
```

这段代码的关键是用**待插入位置按从后往前排序**：如果先插前面的标记，后面的偏移位置就全变了；从后往前插，已处理的偏移不受影响。之后 `entries` 记录的就是三个标记在词元序列里的下标，加上 `[CLS]` 的 +1 偏移，就是模型的注意力位置。

### 2.3 模型训练与超参数调优

#### 2.3.1 基线：手工特征 + ExtraTrees

先跑一个不用深度学习的基线，验证特征方向是否对。因为这是三分类，我们用了 OneVsRest（一对剩余）策略包装 ExtraTrees 分类器，标签是三列 one-hot：

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

feat_cols = ['A_dist', 'B_dist', 'word_count', 'A_poss', 'B_poss',
             'A_min', 'A_max', 'B_min', 'B_max']   # 位置、距离、所有格等

x_train, x_valid, y_train, y_valid = train_test_split(
    train[feat_cols].fillna(-1), train[['A', 'B', 'NEITHER']],
    test_size=0.2, random_state=1)

etc = OneVsRestClassifier(
    ExtraTreesClassifier(n_jobs=-1, max_depth=9, n_estimators=1000, random_state=30))
etc.fit(x_train, y_train)
print(log_loss(y_valid, etc.predict_proba(x_valid)))
# 输出：0.919
```

这个 **0.919** 的 log loss 告诉我们三件事：一是位置特征确实有信息量（远好于乱猜的 log 3 ≈ 1.1）；二是它已经够不到「真正的消解」——0.9 级别的模型本质就是「猜离得近的那个」；三是下一步必须上能理解语义的模型。

#### 2.3.2 主线：微调 BERT

我们的主线是用 `bert-large-uncased` 微调。模型结构很简单：

- 主语 `BERT` 把整个句子编码成语义向量（每个词元一个 1024 维向量）；
- 我们早已知道 `[A]`、`[B]`、`[P]` 三个标记的位置，直接从 BERT 的输出里**取出这三个位置的向量**；
- 把三个向量拼成一个 3072 维的「特征向量」，过一个 MLP（多层感知机，即若干全连接层的堆叠）输出三类的分数。

```python
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

class Head(nn.Module):
    """把 A、B、代词三个位置的 BERT 向量组合成三分类分数"""
    def __init__(self, bert_hidden):
        super().__init__()
        self.bert_hidden = bert_hidden
        # 输入是 3 个位置的向量拼接（bert_hidden*3）
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bert_hidden * 3),   # 批归一化：稳定训练
            nn.Dropout(0.5),                   # 防过拟合
            nn.Linear(bert_hidden * 3, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 3)                  # 输出 A / B / NEITHER
        )

    def forward(self, bert_outputs, offsets):
        # gather 按位置取出 [A]、[B]、[P] 的向量
        extracted = bert_outputs.gather(
            1, offsets.unsqueeze(2).expand(-1, -1, bert_outputs.size(2))
        ).view(bert_outputs.size(0), -1)
        return self.fc(extracted)              # 拼成一个长向量分类

class GAPModel(nn.Module):
    def __init__(self, bert_model='bert-large-uncased'):
        super().__init__()
        self.bert_hidden = 1024 if 'large' in bert_model else 768
        self.bert = BertModel.from_pretrained(bert_model)
        self.head = Head(self.bert_hidden)

    def forward(self, tokens, offsets):
        bert_out, _ = self.bert(tokens,
                                attention_mask=(tokens > 0).long(),
                                output_all_encoded_layers=False)
        return self.head(bert_out, offsets)
```

这个「把三个关键位置提出来再组合」的设计，把模型的决策点**锁定在三个实体上**——它不需要从头推断哪些词是候选，只需要判断三者的关系，学习压力小了很多。损失用交叉熵（CrossEntropyLoss），它在数学上等价于多分类的 log loss：

```python
criterion = torch.nn.CrossEntropyLoss()
```

训练时我们用 KFold 交叉验证（5 折），加权重衰减（weight decay）控制过拟合，得到更稳健的分数。这也是从「单次划分验证」升级到「全量利用数据」的关键一步——原始官方数据总共只有 4454 条带标签样本，训练数据少，交叉验证既能更公平地评估，也能让模型见过所有样本。

**超参数上我们没有做大量的网格搜索**：BERT 的预训练权重已经很强，主干部分用较低的学习率微调，新加的 Head 部分相对敏感，Dropout 0.5 + 权重衰减已经能在这个小数据集上稳住。

### 2.4 模型测试与指标报告

#### 2.4.1 各阶段分数

我们在验证集上记录 log loss，并把它和测试提交分数对照：

| 方案 | log loss |
| --- | --- |
| 随机猜测（3 类均分） | 约 1.10 |
| 手工特征 + ExtraTrees | 0.919 |
| BERT 微调基线 | 约 0.54（公开榜） |
| BERT + Score Layer + KFold + WeightDecay | **0.486**（公开榜最好） |

从 0.919 到 0.486，差距几乎全来自「理解语义」这一步。这正说明手工位置特征有信息但信息量有限，而 BERT 把「谁和谁在语法上发生关系」这种信息学到了。

#### 2.4.2 生成提交

最终预测时，把 5 折模型的输出平均，再归一化到三列和为 1：

```python
def average_and_submit(models, test_loader):
    probs = np.zeros((len(test_ids), 3))
    for model in models:
        model.eval()
        with torch.no_grad():
            for tokens, offsets, _ in test_loader:
                logits = model(tokens, offsets)
                probs += torch.softmax(logits, dim=1).numpy()
    probs /= len(models)                            # 取 5 折平均

    sub = pd.DataFrame({'ID': test_ids,
                        'A': probs[:, 0],
                        'B': probs[:, 1],
                        'NEITHER': probs[:, 2]})
    sub.to_csv('submission.csv', index=False)
```

注意这里用 **softmax 而不是裸 logit 平均**：不同模型最后一层分数的量级可能不同，先把每个模型自己归一化成概率再平均，语义更稳定。

**一个容易踩的坑：型号大小写。** BERT 需要输入固定格式（首尾加 `[CLS]`/`[SEP]`，长度超 500 就截断）。我们特意把 `collate` 里的截断长度设为 500，确保长句被安全处理——否则训练中途一遇到超长样本就崩溃。数据里确实存在个别长度异常的句子，这也是社区很多 kernel 会专门处理的地方。

---

## 三、总结

本文中我们完成了 Gendered Pronoun Resolution（GAP 代词消解）的完整流程。我们首先通过 EDA 确认了任务设定：文本 + 代词 + 两个带位置的候选，三分类输出 A/B/NEITHER，用 log loss 评价。其次我们做了两类预处理——手工特征版把「位置、距离、区间单词数、所有格次数」转成表格特征；BERT 版则在分词前插入 `[A]`/`[B]`/`[P]` 三个特殊标记来可靠定位关键位置。接着我们先用手工特征 + ExtraTrees 得到 0.919 的 log loss 基线，再用 `bert-large-uncased` 微调，把三个位置的 BERT 向量拼接后过 MLP 分类，配合 KFold 与权重衰减，把公开榜分数压到 0.486。最后我们确认了 5 折模型概率平均的提交策略，并处理了超长文本截断的问题。

这个比赛给我们最大的启发是：**任务建模的「位置」很关键**。既不是把整段话丢给模型让它自己找候选，也不是只用简单位置特征——而是用 BERT 这种「能理解语义」的编码器，同时把结构信息（谁在哪儿）显式告诉它。语义和结构结合，才是这类 deep+符号混合问题的最佳解法。