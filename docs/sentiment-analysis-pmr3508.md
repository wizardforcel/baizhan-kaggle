# IMDB 评论情感分析 PMR3508

## 一、比赛简介

### 1.1 比赛背景

PMR3508 是圣保罗大学机器学习和模式识别课程的期末项目。参赛者需要对电影评论的情感进行二分类，判断一段文字表达的是积极感受还是消极感受。数据来源是公开的 IMDB 评论集合，包含大量真实用户对电影的评价文本。因为评论带有主观色彩、口语化表达以及拼写不一致，直接用关键词判断很容易出错，所以任务适合用来练习文本预处理和表示学习。

在课程提供的环境里，文本已经被预先生成了 Doc2Vec 词向量模型，参与者可以直接使用这个模型把一整段评论映射成固定长度的数值向量。整个比赛的核心难点在于如何把文本清洗干净、如何选择合适的分类器以及如何调优超参数，让 AUC 尽可能高。

### 1.2 比赛目标

给定一条评论文本，模型要输出它属于积极还是消极。训练集文件 `data_train.csv` 包含两列：`review` 是评论原文，`positive` 是标签，1 表示积极，0 表示消极。验证集 `data_test1.csv` 同样包含这两列，用来做本地评估。最终测试集 `data_test2_X.csv` 只有 `review` 列，参赛者需要生成对应的 `positive` 预测并提交。

数据规模大约两万多条评论，每条评论长度不一，需要统一成向量后才能喂给机器学习模型。

### 1.3 评价标准

比赛使用 ROC AUC 作为评价指标。AUC 衡量的是模型把积极样本排在消极样本前面的能力，取值在 0 到 1 之间，越接近 1 越好，0.5 相当于随机猜测。在实际调参时我们都用 `scoring='roc_auc'` 并且在验证集上计算 `roc_auc_score`。最终排行榜以验证集的 AUC 作为参考，目的就是让模型在未见过的评论上也能区分好坏。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：先清洗文本再用 Doc2Vec 变成向量，然后用浅层神经网络做二分类。我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，确认列名、样本数和标签分布。

```python
# 导入数据处理库
import pandas as pd

# 读取训练集，确认包含 review 和 positive 两列
train = pd.read_csv("../input/sentiment-analysis-pmr3508/data_train.csv")
print("训练集形状:", train.shape)
print(train.head())
```

打印结果显示训练集大约有 2.5 万条评论，`positive` 的分布比较均衡，积极和消极数量接近，这说明不需要做过采样。随后我们统计重复评论的数量。

```python
# 检查是否有缺失值
missing = train.isna().sum().sum()
print("缺失值总数:", missing)

# 查看是否存在完全相同的评论
dup = train.drop_duplicates(subset='review', keep=False)
print("重复评论行数:", dup.shape[0])
```

EDA 发现没有缺失值，但有几百条重复的 `review`，它们对应的标签一致，可以直接去重保留第一条。标签分布图也显示两类样本数量相当，模型不需要额外的类别权重。

### 2.2 数据清洗

我们其次对原始文本做标准化清洗。IMDB 评论里常见 `<br />` 标签、大小写混杂、标点符号以及数字，清洗的目的是让 Doc2Vec 推断时输入一致。

```python
# 文本清洗函数，参考课程提供的标准流程
import re
import string
from ftfy import fix_text

def clean(text):
    txt = text.replace("<br />", " ")          # 去掉 HTML 标签
    txt = fix_text(txt)                        # 修复编码错误
    txt = txt.lower()                          # 转小写统一形态
    txt = txt.translate(str.maketrans('', '', string.punctuation))  # 去标点
    txt = re.sub(" — ", " ", txt)              # 去掉特殊连接号
    txt = re.sub(r"\d+", " <number> ", txt)    # 数字统一成 token
    txt = re.sub(' +', ' ', txt)               # 合并多余空格
    return txt.strip()
```

清洗后我们把句子拆成词列表，再用预训练的 Doc2Vec 推断向量。

```python
# 将文本清洗并分词
X_raw = train['review'].apply(clean)
X_tokens = X_raw.apply(lambda x: x.split())
```

这样每条评论都变成了一个词序列，后续可以送入 Doc2Vec 得到固定维度的数值表示。清洗不改变语义，只是让向量模型更稳定。

### 2.3 模型训练与超参数调优

我们接着使用 Doc2Vec 把词序列映射成 50 维向量，再用 sklearn 的分类器做训练。课程提供了已训练好的 Doc2Vec 模型，直接加载即可。

```python
# 加载预训练 Doc2Vec 模型
from gensim.models.doc2vec import Doc2Vec
import numpy as np

d2v = Doc2Vec.load("../input/sentiment-analysis-pmr3508/doc2vec")

def emb(txt, model):
    # 用模型推断新文本的向量，steps 决定迭代次数
    x = model.infer_vector(txt, steps=25)
    return x

# 生成训练特征
X_train_vec = np.array([emb(x, d2v) for x in X_tokens])
y_train = train['positive'].values
```

得到向量后，我们尝试了 KNN、LogisticRegression 以及 MLPClassifier。最终发现 MLP 在 AUC 上表现最好，于是重点调优 MLP 的超参数。

```python
# 准备验证集
valid = pd.read_csv("../input/sentiment-analysis-pmr3508/data_test1.csv")
X_valid_tokens = valid['review'].apply(clean).apply(lambda x: x.split())
X_valid_vec = np.array([emb(x, d2v) for x in X_valid_tokens])
y_valid = valid['positive'].values
```

调参使用 `RandomizedSearchCV`，评分标准固定为 `roc_auc`。

```python
# 用随机搜索找最佳 MLP 参数
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

mlp = MLPClassifier(max_iter=1000, early_stopping=True, random_state=42)

param_dist = {
    "hidden_layer_sizes": [(50,), (100,), (100, 50)],
    "alpha": [1e-5, 1e-4, 1e-3],
    "learning_rate": ["constant", "adaptive"]
}

search = RandomizedSearchCV(
    mlp, param_dist, scoring='roc_auc', n_iter=25, cv=2, n_jobs=-1, random_state=0
)
search.fit(X_train_vec, y_train)
print("最佳参数:", search.best_params_)
```

搜索完成后，我们也用同样的流程对 LogisticRegression 做了一次对比，确认 MLP 的优势。整个调参过程强调可复现性，所有随机种子都固定。

### 2.4 模型测试与指标报告

我们然后用验证集评估最终模型，并按比赛要求生成提交文件。

```python
# 在验证集上计算 AUC
from sklearn.metrics import roc_auc_score

best_model = search.best_estimator_
y_proba = best_model.predict_proba(X_valid_vec)[:, 1]
auc = roc_auc_score(y_valid, y_proba)
print("验证集 AUC:", auc)
```

验证集的 AUC 在 0.85 以上，说明模型对积极和消极评论区分能力不错。随后我们用全部训练数据重新拟合一次，再对最终测试集预测。

```python
# 用全部训练数据重训
X_all = pd.concat([train['review'], valid['review']])
y_all = pd.concat([train['positive'], valid['positive']])
X_all_tokens = X_all.apply(clean).apply(lambda x: x.split())
X_all_vec = np.array([emb(x, d2v) for x in X_all_tokens])
best_model.fit(X_all_vec, y_all.values)
```

最后生成提交格式。

```python
# 生成测试集预测
test_df = pd.read_csv("../input/sentiment-analysis-pmr3508/data_test2_X.csv")
X_test_tokens = test_df['review'].apply(clean).apply(lambda x: x.split())
X_test_vec = np.array([emb(x, d2v) for x in X_test_tokens])
test_pred = best_model.predict(X_test_vec)

submission = pd.DataFrame({
    "id": range(len(test_pred)),
    "positive": test_pred
})
submission.to_csv("submission.csv", index=False)
```

提交文件检查列名、行数与测试集一致后保存。整个流程保持了训练和验证时的同一清洗与向量化管道，避免了数据泄漏。

## 三、总结

本文中我们完成了 PMR3508 情感分析项目的完整流程。我们首先通过 EDA 确认了 IMDB 评论数据没有缺失、重复可清理、标签分布均衡。其次我们用统一的清洗函数处理了 HTML 标签、大小写、标点和数字，并配合预训练 Doc2Vec 得到 50 维向量表示。接着我们比较了 KNN、逻辑回归和 MLP，最终在验证集上用随机搜索为 MLP 选择了合适的隐藏层和正则化强度。最后我们在验证集上报告了 AUC，并在全量数据上重训后生成了测试集提交文件。

这个项目给入门者的启发是：文本分类先要把文本变成模型能理解的数字，清洗的一致性比清洗的复杂性更重要；向量表示一旦固定，模型选择和超参调优才是提升 AUC 的关键；训练、验证、测试必须使用完全相同的预处理管道，否则指标会失真。
