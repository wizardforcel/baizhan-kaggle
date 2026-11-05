# Home Depot Product Search Relevance

## 一、比赛简介

### 1.1 比赛背景
Home Depot 的线上搜索需要为每个搜索词匹配最相关的商品。比赛提供历史搜索日志，包含搜索词、商品标题、商品描述、品牌以及人工打分的 relevance。目标是构建一个可以自动给出相关性分数的模型，帮助排序。

数据来源为电商搜索点击与人工标注。商品描述长文本、品牌信息、属性信息共同构成匹配信号，存在拼写变体、单位表达、停用词等噪声。

### 1.2 比赛目标
输入为 test.csv 中的 id、product_uid、product_title、search_term（以及对应的描述与品牌信息），输出每个样本的 relevance 预测值。Relevance 为连续分数，通常 1~3，代表不相关、相关、高度相关。训练集约 74,067 条，列为 id、product_uid、product_title、search_term、relevance。

### 1.3 评价标准
比赛以均方根误差 RMSE 衡量预测与真实分数的差距，数值越低越好。RMSE 对大误差更敏感，要求模型在极端不相关上也能稳定。

---
## 二、解题过程

我们采用文本特征工程 + 树回归的思路，先提炼查询与商品之间的匹配信号，再用 RandomForest 回归。

### 2.1 数据探索（EDA）

先载入并查看训练集形态。

```python
import pandas as pd
dt_train = pd.read_csv("train.csv", encoding="ISO-8859-1")
print(dt_train.shape)               # (74067, 5)
print(dt_train['relevance'].describe())
```

relevance 分布集中在 1~3 之间，明显偏态。查询长度与标题长度差异大，短查询占比高。

```python
dt_train['search_len'] = dt_train['search_term'].str.split().str.len()
dt_train['title_len'] = dt_train['product_title'].str.split().str.len()
print(dt_train['search_len'].mean())
```

查询平均词数约 2~3，标题 5~8 个词。

### 2.2 数据清洗

文本需要统一规范。我们对 search_term、product_title、product_description、brand 做 stemming、去标点、单位统一。

```python
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def str_stem(s):
    if isinstance(s,str):
        s = s.lower()
        s = s.replace(",", "").replace("$"," ")
        s = s.replace("-"," ")
        s = " ".join([stemmer.stem(z) for z in s.split(" ")])
        return s
    return "null"
```

另外合并商品信息为 product_info，方便计算匹配计数。

```python
df_all['product_info'] = df_all['search_term'] + "\t" + \
                         df_all['product_title'] + "\t" + \
                         df_all['product_description']
```

### 2.3 模型训练与超参数调优

我们构造手工匹配特征：

```python
def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word) >= 0:
            cnt += 1
    return cnt

df_all['word_in_title'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['ratio_title'] = df_all['word_in_title'] / df_all['len_of_query']
```

查询在标题中出现次数、查询最后一个词匹配、品牌匹配等特征均被生成。特征数量约数十维。

模型选用 RandomForestRegressor，结合 TF-IDF 文本降维。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

tfidf = TfidfVectorizer(ngram_range=(1,1), stop_words='english')
tsvd = TruncatedSVD(n_components=10, random_state=2016)

rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=2016)
```

网格搜索 max_depth=20、max_features=10，2 折交叉验证得到最佳参数，CV RMSE 约 0.523。

### 2.4 模型测试与指标报告

在验证集上模型 RMSE 约 0.52，相比基准 0.47 有提升空间。最终提交使用全量训练并预测测试集。

```python
y_pred = model.predict(X_test)
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv', index=False)
```

我们检查提交文件行数与 test 保持一致，列名为 id 与 relevance。

---
## 三、总结

本文中我们完成了 Home Depot Product Search Relevance 的文本匹配建模。我们首先通过 EDA 了解查询与商品的文本长度分布。其次对文本做 stemming 与单位归一化清洗，构造查询在标题、描述、品牌中的词数比例等特征。接着用 TF-IDF+TruncatedSVD 降维的文本特征结合手工特征训练 RandomForest，并进行网格搜索调参。最后在验证集上得到 RMSE 约 0.52 的结果并导出提交。

该比赛强调特征工程：匹配计数与长度比例比单纯词袋更能捕捉相关性，清洗不当会丢失重要单位信息。
