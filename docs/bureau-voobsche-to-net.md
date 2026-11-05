# Bureau 文本检索挑战（bureau-voobsche-to-net）

## 一、比赛简介

### 1.1 比赛背景

这个比赛要解决的问题，用大白话说就是「在一堆候选答案里，替每个问题挑出最像答案的那几条」。给出一句问题（query），系统要在一大堆候选文本（candidate）里找出最相关的 5 条。

数据并不是一张普通的大表格，而是**按 corpus（语料库）分开存放**的：每条 query 都带一个 `corpus_id`，表示它属于哪个语料库；候选答案也只和同一个语料库里的其他候选竞争。可以把它想象成「每个话题一个资料柜，问一个问题，就只在对应的资料柜里翻卡片」，而不是把所有资料柜的卡片混在一起翻。

这类任务在搜索、问答、推荐系统里非常常见，学名叫**检索与排序（retrieval and ranking）**。它的难点有两个：

1. 文本是自由格式，没有现成标签，机器只能靠字面上的相似度判断「哪句话更像这个问题的答案」；
2. 每条 query 不是让模型给 1 个答案，而是要排出前 5 个，并且**顺序越靠前越值钱**。

### 1.2 比赛目标

输入分三个目录，结构相似：

- `train/queries.csv`：训练问题，列是 `query_id`、`corpus_id`、`text`；
- `train/<corpus_id>.csv`：某个语料库的候选答案，列是 `candidate_id`、`text`；
- `validation/queries.csv`、`validation/<corpus_id>.csv`、`validation/answers.csv`：验证集，比训练集多一个标准答案表；
- `test/queries.csv`、`test/<corpus_id>.csv`：测试集，只有问题和候选，没有答案。

我们要输出每个 `query_id` 对应的 5 个候选，放在 `candidate_ids` 一列里，用空格隔开。

### 1.3 评价标准

比赛使用 **MAP@5（Mean Average Precision at 5，前 5 名平均精度）**。

对每一条 query，从第 1 名看到第 5 名，每命中一个正确候选就记一分，该位置的分数是「到目前为止命中数 / 当前位置序号」。把这些位置分数取平均，再除以「该 query 的正确候选数（最多算 5 个）」，就得到这条 query 的平均精度。所有 query 再取平均，就是 MAP@5。

它比单纯算「前 5 里有没有命中」更细：**正确答案排第 1 名，比排第 5 名得分高得多**。分数范围 0～1，越高越好，随机乱猜接近 0。

---

## 二、解题过程

我们的思路可以概括为一句话：**对每个语料库单独建一套 TF-IDF 词向量，用余弦相似度把候选按「和问题的接近程度」排序，取前 5 名。** 这既是一个能跑通的基线，也是后面加复杂模型时的对照线。

### 2.1 数据探索（EDA）

首先我们把训练集的问题表和候选表读进来，看看数据到底长什么样：

```python
from pathlib import Path
import pandas as pd

# 数据目录可能在不同环境里位置不同，这里按优先级探测
data_dir = Path(".")
if Path("data/train/queries.csv").exists():
    data_dir = Path("data")
elif Path("/kaggle/input").exists():
    # Kaggle 上数据藏在某个输入目录里，递归找到 train/queries.csv 再回退两级
    data_dir = next(Path("/kaggle/input").rglob("train/queries.csv")).parent.parent

# 训练问题表：query_id、corpus_id、text
train = pd.read_csv(data_dir / "train/queries.csv")
query = train.iloc[0]              # 只取第一条，先感受一下格式
print(query.text)                  # 看一下问题原文
pd.read_csv(data_dir / "train" / f"{query.corpus_id}.csv").head()
```

打印出来，`query.text` 是一句自然语言问题；对应的 `<corpus_id>.csv` 则是两列：`candidate_id` 和 `text`，每一行是一条候选答案。这说明**问题是「文本」、候选也是「文本」**，本质上是一道文本匹配题。

接着我们看看一条 query 对应多少个候选、语料库有几个：

```python
# 每个语料库下有多少条 query，顺便看看一共有几个语料库
print(train.groupby("corpus_id").size())
print("语料库数量:", train.corpus_id.nunique())

# 随便挑一个语料库，看看候选表的规模
cid = train.corpus_id.iloc[0]
corpus = pd.read_csv(data_dir / "train" / f"{cid}.csv")
print("候选数量:", len(corpus))
```

这里能确认一个关键结构：**每条 query 只在自己所属的语料库里检索**。所以后面做相似度计算时，一定要按 `corpus_id` 分组，否则会把别的语料库的候选也算进来，既慢又没有意义。

我们还想确认候选 id 有没有重复，因为这直接影响提交格式是否合法：

```python
# 检查候选 id 是否唯一；如果有重复，后面提取前 5 时要先去重
print("候选 id 有重复吗:", corpus.candidate_id.duplicated().any())
```

### 2.2 数据清洗

这份数据的「脏」不在于缺失值，而在于**结构**：候选分散在多个 corpus 文件里，问题和候选是两张表。我们需要做的清洗主要是对齐和分组：

1. 按 `corpus_id` 分组，保证相似度只在同一个语料库内部计算；
2. 对 `text` 不做去停用词、不去标点，因为 TF-IDF 自带「常见词降权」的机制，粗暴清洗反而可能删掉有判别力的词；
3. 保证输出的 5 个候选互不重复。

清洗的核心代码就是「按语料库分组、逐组读取候选」：

```python
for split in ["validation", "test"]:
    # 分别处理验证集和测试集，两者结构一样，只是验证集多一个 answers.csv
    queries = pd.read_csv(data_dir / split / "queries.csv")

    for corpus_id, group in queries.groupby("corpus_id", sort=False):
        # group 是同一个语料库下的所有 query
        corpus = pd.read_csv(data_dir / split / f"{corpus_id}.csv")
        # 后面就在这个 corpus 内部做相似度检索
```

这样做还有一个好处：每个语料库可以单独建立自己的词表（vocabulary），避免不同话题的词汇互相干扰。

### 2.3 模型训练与超参数调优

我们的「模型」其实是一套**向量化 + 相似度排序**的流程，不需要传统意义上的梯度训练，但需要选择向量化参数。

**第一步，把文本变成 TF-IDF 向量。** TF-IDF 的意思是「词频乘以逆文档频率」：一个词在本条文本里出现得越多、在整个语料库里越少见，它的权重就越高。我们选了 `ngram_range=(1, 2)`，也就是同时看单个词和相邻两词组成的短语；`sublinear_tf=True` 会把词频取对数，避免某个词刷屏把权重拉爆：

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 每个语料库单独建一个向量器
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # 单词 + 二元词组
    sublinear_tf=True,    # 词频取对数，抑制高频词的支配作用
)
# 先在候选文本上 fit，学到词表和 IDF 权重
corpus_vectors = vectorizer.fit_transform(corpus.text)
# 再用同一个词表把 query 变成向量（不能再 fit，否则词表不一致）
query_vectors = vectorizer.transform(group.text)
```

注意这里有一个新手常踩的坑：**候选和问题必须用同一个 `vectorizer`**。如果对 query 重新 `fit`，它会得到自己的词表，两个向量就不在同一个坐标系里，算出来的相似度没有意义。

**第二步，算余弦相似度并取前 5。** 余弦相似度衡量两个向量方向的夹角，值越大越像。`TfidfVectorizer` 默认会把每行归一化成长度为 1 的向量，所以两个矩阵直接点乘就是余弦相似度：

```python
import numpy as np

# (query 数 × 词表维度) @ (词表维度 × 候选数) = (query 数 × 候选数) 的相似度矩阵
similarities = (query_vectors @ corpus_vectors.T).toarray()

for query, scores in zip(group.itertuples(index=False), similarities):
    # argsort 升序，加负号变降序；kind="stable" 保证分数相同时顺序稳定
    top5 = np.argsort(-scores, kind="stable")[:5]
    # 用位置索引回候选表里取 candidate_id
    candidates = corpus.iloc[top5].candidate_id.tolist()
    rows.append({"query_id": query.query_id, "candidate_ids": " ".join(candidates)})
```

**关于调参。** 我们主要比较了 `ngram_range` 取 `(1,1)` 和 `(1,2)` 的效果。单词层面已经能抓到大部分关键词，加上二元词组可以补上「两个词连在一起才有意义」的情况，代价是词表变大、计算稍慢。考虑到这个基线要在验证集和测试集上各跑一遍，我们最终保留了 `(1, 2)`，`sublinear_tf=True` 也作为默认开启。

### 2.4 模型测试与指标报告

由于测试集没有答案，我们用**验证集**来估算效果。验证集里有 `answers.csv`，记录了每条 query 的正确候选：

```python
# 把标准答案整理成 {query_id: {candidate_id, ...}} 的字典，方便查
answers = pd.read_csv(data_dir / "validation/answers.csv")
correct = answers.groupby("query_id").candidate_id.agg(set).to_dict()

average_precisions = []
for row in predictions["validation"].itertuples(index=False):
    candidates = row.candidate_ids.split()
    # 逐个位置判断是否命中，得到 [0,1,0,1,1] 这样的序列
    hits = np.array([candidate in correct[row.query_id] for candidate in candidates])
    # precision@k = 前 k 个里命中数 / k
    precision = np.cumsum(hits) / np.arange(1, 6)
    # 只对命中的位置求和再平均，除以「正确候选数（最多 5）」
    average_precision = np.sum(precision * hits) / min(5, len(correct[row.query_id]))
    average_precisions.append(average_precision)

print(f"Validation MAP@5: {np.mean(average_precisions):.4f}")
```

这段代码就是 MAP@5 的「手算版」，把它和比赛官方的定义对照，能确保我们本地算的分和排行榜口径一致。**这里要特别强调分母**：如果某条 query 的正确候选只有 2 个，分母就是 2 而不是 5，否则分数会被系统性压低。

最后，我们检查提交文件的合法性并保存：

```python
submission = predictions["test"]

# 每个 query 只能出现一次
assert submission.query_id.is_unique
# 每条 query 必须正好 5 个候选，且互不重复
for candidates in submission.candidate_ids.str.split():
    assert len(candidates) == len(set(candidates)) == 5

submission.to_csv("submission.csv", index=False)
print(f"submission.csv: {len(submission)} 行")
```

这几个 `assert` 看起来不起眼，但比赛里因为「少给一个候选」「候选重复」导致提交失败的情况很多。我们宁可提交前多花两秒钟检查，也不想白跑一次。

从验证集的结果看，这套 TF-IDF 基线能把「字面上最接近问题」的候选排到前面，作为没有训练任何神经网络的起点是合格的。它的短板也很明显：**只看字面，不懂语义**。如果正确答案用了同义词、改写了句式，词面向量就抓不到，这也是后续可以用预训练语言模型（比如把 query 和 candidate 都编码成句向量）继续提升的地方。

---

## 三、总结

本文中我们完成了 Bureau 文本检索挑战的一条完整基线。我们首先用 EDA 摸清了数据结构：问题表带 `corpus_id`，候选按语料库拆成多个 CSV，检索必须在同一个语料库内部进行。其次我们做了最少量的清洗——只按语料库分组、不对文本去停用词，把清洗的活交给 TF-IDF 的权重机制。接着我们对每个语料库单独拟合 TF-IDF（单词加二元词组、词频取对数），用余弦相似度给候选打分并取前 5。最后我们在验证集上手动实现了 MAP@5 来对齐官方口径，并对提交文件做了「query 唯一、5 个候选不重复」的合法性检查。

这个比赛给我们的启发是：**检索任务的第一步不是上大模型，而是把「同一语料库内比较」这个结构约束落实到位**。只要分组正确、向量器共用、评价口径一致，一个简单的 TF-IDF 就能给出可复现的基线；后面想提分，再考虑语义向量和重排序。
