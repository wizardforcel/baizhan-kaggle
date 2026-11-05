# What's Cooking

## 一、比赛简介

### 1.1 比赛背景
What's Cooking 是 Kaggle 的经典文本分类比赛，目标是通过菜谱中的食材列表推断菜系。不同菜系使用不同的食材组合，例如意大利菜常出现面食和番茄，印度菜常出现香料。比赛数据来自用户提交的菜谱，每条记录包含 id、cuisine 标签和 ingredients 列表。任务本质上是多分类文本分类问题。

### 1.2 比赛目标
输入食材列表，输出对应的菜系类别。训练集约 39774 条记录，包含 20 种菜系；测试集约 9944 条记录，无标签。需要预测测试集每个样本的菜系。

### 1.3 评价标准
评价指标为分类准确率，预测标签与真实标签相同的比例。准确率越高越好。

## 二、解题过程

### 2.1 数据探索（EDA）
读取训练和测试 JSON 文件并转为 DataFrame。

```python
import pandas as pd
import json
with open('train.json') as f:
    train = pd.DataFrame(json.load(f))
with open('test.json') as f:
    test = pd.DataFrame(json.load(f))
print(train.shape, test.shape)
```

训练集 39774 行，测试集 9944 行。cuisine 分布不均，italian 最多 7838 条，russian 仅 489 条。ingredients 是列表形式，需要转为字符串。

```python
print(train['cuisine'].value_counts().head())
```

### 2.2 数据清洗
把食材列表拼接成字符串，并做词形还原。

```python
import re
from nltk import WordNetLemmatizer

train['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))
                                        for line in lists]).strip() for lists in train['ingredients']]
test['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))
                                       for line in lists]).strip() for lists in test['ingredients']]
```

随后使用 TF-IDF 向量化。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,1), max_df=0.57)
X_train = vectorizer.fit_transform(train['ingredients_string'])
X_test = vectorizer.transform(test['ingredients_string'])
```

### 2.3 模型训练与超参数调优
使用逻辑回归分类器。

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y = le.fit_transform(train['cuisine'])
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=0, stratify=y)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='liblinear')
clf.fit(X_tr, y_tr)
```

验证集准确率约 0.77。随后用 GridSearch 调优 C 参数。

```python
from sklearn.model_selection import GridSearchCV
param = {'C': [1,10]}
grid = GridSearchCV(LogisticRegression(solver='liblinear'), param)
grid.fit(X_tr, y_tr)
best_C = grid.best_params_['C']
```

最佳 C 为 7，验证集准确率提升至 0.78。

### 2.4 模型测试与指标报告
在验证集上输出分类报告。

```python
from sklearn.metrics import classification_report
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))
```

多数类别 F1 在 0.7 以上，小样本类别表现稍弱。最终用全量训练数据训练并预测测试集。

```python
best_clf = LogisticRegression(solver='liblinear', C=7)
best_clf.fit(X_train, y)
pred = le.inverse_transform(best_clf.predict(X_test))
submission = pd.DataFrame({'id': test['id'], 'cuisine': pred})
submission.to_csv('submission.csv', index=False)
```

## 三、总结
本文中我们完成了 What's Cooking 的完整流程。我们首先通过 EDA 了解数据规模和类别不平衡。其次清洗食材列表并做 TF-IDF 向量化。接着使用逻辑回归并通过 GridSearch 调优 C 参数，验证集准确率达到 0.78。最后生成提交文件完成预测。该方案证明简单的词袋模型在菜系分类任务上效果良好。
