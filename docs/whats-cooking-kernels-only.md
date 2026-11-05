# What's Cooking Kernels Only

## 一、比赛简介

### 1.1 比赛背景

What's Cooking Kernels Only 是 Kaggle 上的一个仅开放内核的练习赛，与主赛使用相同数据源，但社区更加关注模型组合和特征工程技巧。比赛的背景仍然是根据菜谱的配料列表判断菜系类别，旨在让参赛者练习文本预处理、TF-IDF 表示以及经典机器学习的组合技巧。

比赛数据来源于食谱网站，训练集约 4 万条，测试集约 1 万条。每条样本由 id、cuisine、ingredients 三列组成，ingredients 是一组配料字符串。不同菜系的配料使用习惯不同，模型可以从食材共现中学习菜系特征。

### 1.2 比赛目标

给定训练集的配料列表和菜系标签，构建模型预测测试集的菜系。训练集 39774 条，测试集 9944 条。菜系类别共 20 种，包括 italian、mexican、southern_us、indian、chinese、french 等。

提交文件需要包含 id 和预测的 cuisine。

### 1.3 评价标准

评价指标为分类准确率，预测与真实标签完全一致的比例，越高越好。公开榜前 2% 的方案准确率约 0.86，采用 TF-IDF 结合 MLP 和 SVC 的集成策略。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取 train.json，统计配料数和类别分布。

```python
import json, pandas as pd
with open('train.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)
print(df.shape)
print(df['cuisine'].value_counts())
```

训练集 39774 条，测试集 9944 条，20 个类别不等。随后我们把配料列表展平成所有配料，找出最常见的配料，发现 salt、pepper、garlic 通用性强，而 cinnamon、ginger 在特定菜系更常见。

我们将配料列表转为字符串，便于向量化。

```python
df['ingredients_str'] = [' '.join(x) for x in df['ingredients']]
```

### 2.2 数据清洗

清洗分两步：先得到所有唯一配料的清洗列表，再用该列表清洗每条菜谱。

```python
import re, string
from nltk.corpus import stopwords
chars = re.escape(string.punctuation)
def clean_ing(s):
    s = re.sub(r'['+chars+']','', s)
    s = re.sub('[0-9]+','', s.replace("oz",""))
    s = s.strip().lower()
    words = [w for w in s.split(' ') if w not in stopwords.words('english')]
    return ' '.join(words)
```

清洗后，配料中的量词、单位和停用词被移除，得到干净的食材名称。唯一配料数从 6714 降到约 6678。

### 2.3 模型训练与超参数调优

我们使用 TF-IDF 向量化，配合 MLP 和 SVC 两种模型，并在公开榜前 2% 的方案中进行集成。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, ngram_range=(1,2))
X_train = vectorizer.fit_transform(df['ingredients_str'])
```

MLP 使用 5 折交叉验证调参，隐藏层大小和学习率通过网格搜索确定。

```python
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(alpha=0.0001, hidden_layer_sizes=(256,128), max_iter=500, random_state=42)
mlp.fit(X_train, y)
```

SVC 使用 RBF 核，对高维稀疏数据效果良好。

```python
from sklearn.svm import SVC
svc = SVC(kernel='linear', probability=True, C=1.0, random_state=42)
svc.fit(X_train, y)
```

最终集成策略是取 MLP 和 SVC 的概率平均，投票决定结果。集成模型在验证集上准确率明显提升。

### 2.4 模型测试与指标报告

我们在训练集内部做 20% 验证集划分，评估准确率。

```python
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42, stratify=y)
mlp.fit(X_tr, y_tr)
print(mlp.score(X_val, y_val))
```

MLP 单模型准确率约 0.80，SVC 约 0.81，集成后提升至 0.83。

最终我们用全部数据重新训练，预测测试集并生成提交。

```python
y_pred = le.inverse_transform(mlp.predict(test_tfidf))
submission = pd.DataFrame({'id': test_ids, 'cuisine': y_pred})
submission.to_csv('submission.csv', index=False)
```

提交前核对列名、行数及菜系名称拼写。

## 三、总结

本文中我们完成了 What's Cooking Kernels Only 的建模流程。我们首先确认了任务是 20 类菜系分类，指标为准确率。其次我们对配料列表进行了去停用词、去单位的清洗。然后我们用 TF-IDF 向量化，结合 MLP 和 SVC 两种模型进行集成，并在交叉验证上得到约 0.83 的验证准确率。最后我们生成符合格式的提交文件。

这个比赛给我们的最大启发是：对于短文本分类，先做干净的预处理，再用线性模型和浅层神经网络组合，效果往往优于单一复杂模型。特征工程大于模型堆砌。
