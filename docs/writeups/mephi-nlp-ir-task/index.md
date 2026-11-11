# MEPhI NLP IR Task

## 一、比赛简介

### 1.1 比赛背景

MEPhI NLP IR Task 是 MEPhI（莫斯科工程物理学院）举办的信息检索（IR）任务。给定一个产品目录（catalog）和一组查询（query），我们要求对每个查询从目录中检索出最相关的 top-K 个产品。这类任务在推荐系统、搜索引擎里有广泛的应用，是自然语言处理与信息检索结合的经典问题。

材料来自官方 baseline 页面，标题 “MEPhI NLP IR Task baseline TF-IDF”，作者 fedor kurushin，建立了一个纯 TF-IDF + 余弦相似度的检索 pipeline。

### 1.2 比赛目标

输入为 catalog.csv 和 test.csv。catalog.csv 含 product_id、title、description 等字段；test.csv 各项为查询及其文本。输出 submission.csv，其中每一行包含 id 和 product_ids（预测出的产品编号，用空格分隔）。baseline 取 top-10 输出。

### 1.3 评价标准

材料没有给出官方指标细节，我们按信息检索任务的惯例保守理解：通常会衡量返回结果是否命中正确答案，以及命中结果是否排得足够靠前。baseline 的思路是让相似度最高的产品排在最前面，这符合这类评价标准。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取目录与测试查询，看看文档和查询的数量与文本内容：

```python
import pandas as pd
from pathlib import Path
DATA = Path('/kaggle/input/mephi-nlp-ir-task')
catalog = pd.read_csv(DATA / 'catalog.csv').fillna('')
test = pd.read_csv(DATA / 'test.csv').fillna('')
print(DATA, len(catalog), len(test))
catalog.head()
```

打印出的 catalog 大概几千条，测试查询若干条，体现了“从较小目录检索”的规模场景。catalog 文本主要由 title、description 组成，query 是一句话文本，需要与文档做相似度匹配。

### 2.2 数据清洗

文本预处理流程我们保持简单：填充缺失值为空字符串，防止向量化时报错；不删除短文本，因为 title 可能本来就短。我们不做复杂的分词和停用词过滤，让 TF-IDF 自己处理：

```python
docs = catalog['title'] + ' ' + catalog['description']
query = test['query'].fillna('')
```

这样的轻量处理保留了尽可能多的原始信息，适合作为 baseline。

### 2.3 模型训练与超参数调优

核心是 TF-IDF 向量化加余弦相似度。TF-IDF 会让出现次数少但在当前文档中突出的词获得更高权重，抑制“每个文档都有”的常见词。我们用同一套向量化器拟合文档，再变换查询，保证特征空间一致：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(lowercase=True, min_df=2)
X = vec.fit_transform(docs)
Q = vec.transform(query)
```

超参数只设置了 min_df=2（过滤文档频率太低的词），并统一转小写。之后对每个查询计算与所有文档的余弦相似度矩阵：

```python
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(Q, X)
```

### 2.4 模型测试与指标报告

我们取每行相似度最高的前 K=10 个产品，按分数降序输出 product_id：

```python
K = 10
pids = catalog['product_id'].astype(str).tolist()
rows = []
for qid, row in zip(test['id'], sims):
    idx = row.argsort()[::-1][:K]
    rows.append({'id': qid, 'product_ids': ' '.join(pids[i] for i in idx)})
sub = pd.DataFrame(rows)
sub.to_csv('submission.csv', index=False)
print(sub.head())
```

TF-IDF 能在词级向量空间里直接比较查询与文档的相似度，无需训练模型，适合作为比赛起点。

## 三、总结

本文中我们完成了 MEPhI NLP IR Task 的 baseline 流程。我们首先读取 catalog 和 test 并观察规模，其次对文本做了空值填充和转小写的清洗，接着用 TF-IDF 向量化文档与查询并计算余弦相似度，最后取每个查询相似度最高的 top-10 产品写出提交文件。这个检索流程无需训练，完全依赖词频统计，是信息检索任务最经典也最稳的起点，后续可以在此基础上引入 BM25、词向量或排序模型来提升命中率。