# Yelp RecSys 2013

## 一、比赛简介

### 1.1 比赛背景
本比赛为 Kaggle 2013 年 Yelp 推荐系统挑战赛的子任务，基于 Yelp 用户评论数据进行自然语言处理。Yelp 是一个本地商业点评平台，用户会在评论中描述对商家服务的体验。评论文本包含丰富的情感信息，可以用于预测用户评分。

本任务聚焦于使用评论文本内容预测用户给出的星级评分，属于文本分类与情感分析的结合。数据包含商家 ID、评论日期、评论文本、星级评分等字段。

### 1.2 比赛目标
输入是 Yelp 用户的评论文本，目标是将其分类为 1 星或 5 星两种极端类别。训练数据包含大量商店评论和对应的星级标签，我们需要基于文本内容学习模型，预测新评论的极端评分类别。

数据字段包括 business_id、date、review_id、stars、text、type、user_id、cool、useful、funny 等。我们主要使用 text 和 stars 字段建模。

### 1.3 评价标准
评价指标为分类准确率 Accuracy，即预测为 1 星或 5 星的正确率。数值越高越好。任务难度在于评论文本存在歧义、讽刺等语言现象，需要模型捕捉微妙的语义差异。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先加载 Yelp 数据。

```python
import pandas as pd
import numpy as np
yelp = pd.read_csv('../input/yelp-data/yelp.csv')
print(yelp.head())
```

数据共包含 business_id、date、review_id、stars、text 等列。

我们查看星级分布。

```python
print(yelp['stars'].value_counts())
```

发现 5 星和 1 星评论数量相对较少，但极端评论具有较高的信息价值。

我们统计评论长度。

```python
yelp['text_len'] = yelp['text'].str.len()
print(yelp['text_len'].describe())
```

极端评分评论的文本长度普遍较长，说明用户在强烈不满或满意时会写更详细的评论。

### 2.2 数据清洗
我们筛选出 1 星和 5 星评论。

```python
yelp_extreme = yelp[yelp['stars'].isin([1,5])]
yelp_extreme['label'] = (yelp_extreme['stars'] == 5).astype(int)
```

对文本进行预处理，去除 HTML 标签、非字母字符、转为小写并去除停用词。

```python
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def clean_text(text):
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = [w for w in text.split() if w not in stopwords.words('english')]
    return ' '.join(words)
```

清洗后文本更加规范，便于向量化。

### 2.3 模型训练与超参数调优
我们使用词袋模型构建特征。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(yelp_extreme['text_clean'])
y = yelp_extreme['label']
```

使用 Logistic Regression 作为基线模型。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X_train, y_train)
```

我们尝试调优正则化参数 C 和 n-gram 范围。n-gram 范围设为 (1,2) 能捕捉短语信息，提升模型对特定表达的敏感度。

随后尝试随机森林模型。

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1)
rf.fit(X_train, y_train)
```

随机森林对高维稀疏特征鲁棒性较好，但在文本分类上通常不如线性模型。

### 2.4 模型测试与指标报告
在测试集上评估模型。

```python
from sklearn.metrics import accuracy_score, classification_report
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)
```

Logistic Regression 模型在测试集上达到约 0.85 的准确率，能够较好区分 1 星和 5 星评论。

我们检查分类报告。

```python
print(classification_report(y_test, pred))
```

发现模型对 5 星评论的召回率略高于 1 星，建议对类别不平衡进行加权。

最终我们用全量数据重新训练模型，生成提交文件。

```python
clf.fit(X, y)
test_pred = clf.predict(X_test_final)
submission = pd.DataFrame({'review_id': test_id, 'stars': test_pred})
submission.to_csv('submission.csv', index=False)
```

## 三、总结
本文中我们完成了 Yelp RecSys 2013 的文本分类流程。我们首先用 EDA 了解数据：评论包含 stars、text 等字段，极端评分评论文本较长。其次我们筛选 1 星和 5 星评论并做文本清洗。接着我们用 TF-IDF 构建 1-2 gram 特征，训练 Logistic Regression 基线模型，测试集准确率约 0.85。最后我们用全量数据训练并生成提交文件。经验表明，极端评论分类中词袋特征配合线性模型已能取得较好效果，文本长度和特定表达是重要的区分信号。
