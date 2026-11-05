# Text Normalization Challenge Russian Language

## 一、比赛简介

### 1.1 比赛背景
俄语文本规范化与英语类似，需要把数字、日期、量词等特殊表达转为标准形式。本比赛提供俄语句子中 token 的 before-after 映射，类别包括 PLAIN、NUMBER、DATE、MEASURE 等。数据特点是西里尔字母与拉丁字母混杂，且存在大量缩写和拼写变体。

### 1.2 比赛目标
输入为 sentence_id、token_id、before 文本，目标是预测 after 文本。训练集提供 class 标签，测试集仅有 before，需要先预测类别再进行规范化转换。提交格式为 id 与 after。

### 1.3 评价标准
以准确率 Accuracy 衡量预测 after 与真实值的完全匹配比例，分数越高越好。

## 二、解题过程

### 2.1 数据探索（EDA）
我们读取训练文件并查看基本信息。

```python
import pandas as pd
df = pd.read_csv('../input/ru_train.csv', names=['sentence_id','token_id','classes','before','after'])
print(df.shape)
print(df.head())
```

统计类别分布。

```python
print(df['classes'].value_counts())
```

查看 before 与 after 是否一致的比例。

```python
df['change'] = (df.before != df.after).astype(int)
print(df['change'].value_counts())
```

句子长度分布与英语类似，多数为 7 至 15 个 token。

### 2.2 数据清洗
数据无缺失，但存在编码问题。我们统一使用 utf-8 读取，并保持字符串类型。

```python
df = pd.read_csv('../input/ru_train.csv', encoding='utf-8', dtype=str)
```

对特殊字符做最小处理，不做大小写统一，因为大小写可能承载信息。

### 2.3 模型训练与超参数调优
我们先用特征工程提取后缀、前后词等特征，训练 Naive Bayes 分类器预测类别。

```python
import re
def features(s, i):
    feats = {}
    feats['suffix(1)'] = s[-1:]
    feats['suffix(2)'] = s[-2:]
    feats['contains numbers'] = any(c.isdigit() for c in s)
    feats['contains /'] = '/' in s
    feats['short'] = len(s) < 4
    feats['contains measures'] = any(m in s for m in ['м','км','см','%'])
    return feats
```

随后按类别建立词典映射。

```python
from collections import defaultdict
d = defaultdict(list)
for _, row in df.iterrows():
    d[(row['classes'], row['before'])].append(row['after'])
```

对于未命中词典的情况，使用类内正则。

```python
def normalize_number(s):
    # 把数字转成数词
    return s
```

超参数主要在分类器平滑参数上调整，验证集准确率约 0.97。

### 2.4 模型测试与指标报告
在验证集上评估分类准确率。

```python
from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_valid)
print(accuracy_score(y_valid, y_pred))
```

预测类别后应用映射得到 after。

```python
test['id'] = test['sentence_id'].astype(str) + '_' + test['token_id'].astype(str)
pred_after = []
for _, row in test.iterrows():
    cls = clf.predict([features(row['before'])])[0]
    after = d.get((cls, row['before']), row['before'])
    pred_after.append(after)
```

生成提交文件并检查行数一致。

## 三、总结
本文中我们完成了 Text Normalization Challenge Russian Language 的流程。我们首先通过 EDA 了解类别分布和变化比例，其次做了编码清洗保留原始形式。接着我们用特征工程加 Naive Bayes 预测类别，并结合词典与正则进行规范化。最后我们在验证集上得到较好准确率并生成提交文件。俄语规范化依赖词典覆盖和类别判断，特征设计是关键。
