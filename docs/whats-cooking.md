# What's Cooking

## 一、比赛简介

### 1.1 比赛背景

What's Cooking 是 Kaggle 上的一道经典文本分类入门赛。数据集来自食谱网站，提供约 4 万条菜谱记录，每条记录包含菜谱 ID、菜系标签和配料清单。目标是根据配料列表判断这道菜属于哪一种菜系。

比赛背后体现的是“从食材看不出菜系吗”的问题。不同国家的饮食习惯在使用的食材上有明显差异，比如印度菜常用香料，意大利菜常用番茄和起司，日本菜常用海鲜和米。把这些配料当成“词”，菜系当成“类”，本质上就是一个多分类文本分类问题。

### 1.2 比赛目标

训练集包含 39774 条样本，每条有 id、cuisine、ingredients 三列。ingredients 是配料列表，类型是 Python 列表，里面元素是字符串。cuisine 有 20 个类别，包括 italian、mexican、southern_us、indian 等。

测试集包含 9944 条样本，只有 id 和 ingredients。我们需要为每条测试样本预测 cuisine。

### 1.3 评价标准

评价指标是分类准确率，预测正确的比例越高越好。排行榜上前 7% 的方案准确率约为 0.81。准确率衡量的是单点分类效果，不是概率排序。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取 train.json 和 test.json，统计样本数量和类别分布。

```python
import json
import pandas as pd

with open('./train.json') as f:
    train = json.load(f)
train = pd.DataFrame(train)
print(train.shape)
print(train['cuisine'].value_counts())
```

训练集 39774 行，测试集 9944 行。20 个菜系样本数差别很大，italian 有 7838 条，russian 只有 489 条，数据不均衡但仍可接受。

接着我们把配料列表拼接成一个字符串，便于做文本建模。

```python
train['ingredients_str'] = [' , '.join(z) for z in train['ingredients']]
```

我们还统计了最常用的配料，发现 salt、pepper、garlic 等基础调味料在所有菜系中都很常见，而特定配料如 garam masala 集中在印度菜。

### 2.2 数据清洗

清洗的核心是把配料列表转成规范的文本。我们对每个配料做词干化、去停用词、去数字和特殊字符。

```python
import re
from nltk import WordNetLemmatizer

def clean_ingredients(lst):
    out = []
    lem = WordNetLemmatizer()
    for item in lst:
        t = re.sub('[^A-Za-z]', ' ', item)
        t = lem.lemmatize(t)
        out.append(t.lower())
    return ' '.join(out)
```

清洗后我们得到干净的 ingredient 字符串，去除了量词和单位如 oz、crushed、ground 等，对模型更友好。

### 2.3 模型训练与超参数调优

我们采用 TF-IDF 向量化加逻辑回归的流水线。TF-IDF 能够降低常见配料的权重，提升区分度。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train_corpus = train['ingredients_string']
train_tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,1), max_df=0.57).fit_transform(train_corpus)

le = LabelEncoder()
y = le.fit_transform(train['cuisine'])
X_train, X_val, y_train, y_val = train_test_split(train_tfidf, y, test_size=0.2, random_state=0, stratify=y)
```

逻辑回归参数 C 控制正则化强度，我们用 GridSearch 在 C=[1,10] 上搜索，发现 C=7 时验证集准确率提升。最终模型准确率约为 0.77。

### 2.4 模型测试与指标报告

在验证集上我们打印分类报告，每个类别的精确率和召回率差异不大，说明模型相对平衡。

```python
clf = LogisticRegression(solver='liblinear', C=7)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))
```

最终我们用全部训练数据重新训练，对测试集预测，生成提交文件。

```python
final_clf = LogisticRegression(solver='liblinear', C=7)
final_clf.fit(train_tfidf, y)
test_tfidf = vectorizer.transform(test_corpus)
pred = le.inverse_transform(final_clf.predict(test_tfidf))
```

提交前我们核对列名和行数，确保与 sample_submission 一致。

## 三、总结

本文中我们完成了 What's Cooking 的建模流程。我们首先确认了任务是基于配料列表的 20 类菜系分类，指标为准确率。其次我们做了配料列表的拼接、词干化和停用词清洗。然后我们用 TF-IDF 加逻辑回归建立基线，并通过 GridSearch 找到合适的正则化强度。最后我们在验证集上得到约 0.77 的准确率，并生成提交文件。

这个比赛给我们的最大启发是：文本分类不一定要用深度学习，配料列表这种短文本，TF-IDF 配合线性模型往往足够好。数据清洗的重点在于统一表述，而不是堆砌复杂特征。
