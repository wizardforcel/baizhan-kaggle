# Jigsaw Toxic Comment Classification Challenge

## 一、比赛简介

### 1.1 比赛背景

Jigsaw Toxic Comment Classification Challenge 由 Google Jigsaw 发起，目标是识别网上评论中的有毒言论。数据来自 Wikipedia 讨论页评论，包含多种毒性标签如 toxic、severe_toxic、obscene、threat、insult、identity_hate。

文本数据量大、标签稀疏，属于多标签文本分类问题。

### 1.2 比赛目标

给定一段评论文本，预测其在六个维度上的毒性概率。训练集包含 `id`、`comment_text` 与六个标签列，测试集仅含 `id` 与 `comment_text`，需输出每列概率。

### 1.3 评价标准

比赛使用逐列 ROC AUC 的平均值（mean ROC AUC）作为评价指标，值越高越好。它衡量模型在各标签上的排序能力，对阈值不敏感。

## 二、解题过程

我们采用 TF-IDF 特征提取与经典线性模型，并对文本做清洗与词干化。

### 2.1 数据探索（EDA）

读取训练数据并检查缺失值与标签分布。

```python
import pandas as pd

train_df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
print(train_df.isnull().sum())
```

六个标签均无缺失。我们统计各标签的阳性样本数。

```python
print(train_df.loc[train_df.toxic==1].shape[0])      # 15294
print(train_df.loc[train_df.severe_toxic==1].shape[0])  # 1595
print(train_df.loc[train_df.obscene==1].shape[0])    # 8449
print(train_df.loc[train_df.threat==1].shape[0])     # 478
print(train_df.loc[train_df.insult==1].shape[0])     # 7877
print(train_df.loc[train_df.identity_hate==1].shape[0]) # 1405
```

标签严重不平衡，threat 与 identity_hate 阳性样本最少。

### 2.2 数据清洗

我们对文本做小写化、去链接、去停用词、词干化处理。

```python
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language='english')
stopwords_ = set(stopwords.words('english'))

def simple_clear(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]", ' ', text)
    text = re.sub('<.*?>','', text)
    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_]
    return ' '.join(text)

train_df['clean'] = train_df['comment_text'].apply(simple_clear)
```

清洗后文本更简洁，去除噪声并减少词汇表量。

### 2.3 模型训练与超参数调优

使用 TF-IDF 向量化 1-3-gram，配合 LogisticRegression 做多标签分类。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

X_raw = train_df['clean']
y = train_df.drop(['id','comment_text','clean'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

word_vectorizer = TfidfVectorizer(
    strip_accents='unicode',
    token_pattern=r'\w{1,}',
    ngram_range=(1,3),
    stop_words=stopwords_,
    sublinear_tf=True
)
X_train_transformed = word_vectorizer.fit_transform(X_train)
```

训练 OneVsRest 的 LogisticRegression。

```python
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, solver='lbfgs'))
clf.fit(X_train_transformed, y_train)
```

超参数包括 ngram 范围 (1,3)、sublinear_tf 与 stop_words，后续可调 C 值。

### 2.4 模型测试与指标报告

在验证集上计算各标签 ROC AUC 并求平均。

```python
from sklearn.metrics import roc_auc_score

y_pred_proba = clf.predict_proba(word_vectorizer.transform(X_test))
auc_scores = [roc_auc_score(y_test.iloc[:,i], y_pred_proba[:,i]) for i in range(y_test.shape[1])]
print('Mean ROC AUC:', np.mean(auc_scores))
```

对测试集生成预测。

```python
test_df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
test_clean = test_df['comment_text'].apply(simple_clear)
X_test_final = word_vectorizer.transform(test_clean)
preds = clf.predict_proba(X_test_final)
submission = pd.DataFrame(preds, columns=y.columns)
submission['id'] = test_df['id']
submission.to_csv('submission.csv', index=False)
```

## 三、总结

本文中我们完成了 Jigsaw Toxic Comment Classification 的多标签文本分类。我们首先通过 EDA 发现六个毒性标签极度不平衡，threat 仅 478 条阳性。其次我们对文本做清洗、小写、去链接、停用词与词干化。接着使用 TF-IDF 1-3-gram 配合 OneVsRest LogisticRegression 建立基线。最后在验证集上得到均值 ROC AUC 并生成提交文件。该方案证明了经典特征与线性模型在文本毒性检测上的有效性。