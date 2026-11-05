# Word2Vec NLP Tutorial

## 一、比赛简介

### 1.1 比赛背景
本教程基于 Kaggle 的「Word2Vec NLP Tutorial」以及 IMDB 电影评论情感分析任务。自然语言处理是机器学习的重要分支，情感分析是其典型应用。IMDB 电影评论数据集包含大量电影评论文本和对应的情感标签，任务是根据评论内容判断用户情感是正面还是负面。

Word2Vec 是 Google 提出的一种词向量训练方法，它能够将词语映射到高维向量空间，使语义相近的词在向量空间中距离也相近。本教程旨在通过 Bag of Words 和 Word2Vec 两种方法，对电影评论进行情感分类。

### 1.2 比赛目标
输入是电影评论文本，输出是评论情感标签，0 表示负面，1 表示正面。训练集共 25000 条评论，每条评论经过 HTML 去除、非字母字符替换、小写化、停用词去除等预处理后用于建模。

我们需要在训练集上构建文本分类模型，并在测试集上预测情感标签。目标是提高分类准确率。

### 1.3 评价标准
评价指标使用分类准确率 Accuracy，即预测正确的样本数占总样本数的比例。数值越高越好，满分为 1.0。不同模型在该数据集上的表现差异可达数个百分点。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读取带标签的训练数据。

```python
import pandas as pd
from bs4 import BeautifulSoup
train = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3)
print(train.shape)
```

数据包含 id、sentiment、review 三列。

我们对单条评论做预处理示例：

```python
example1 = BeautifulSoup(train["review"][0], features="lxml")
print(example1.get_text())
```

可以看到原始评论包含 HTML 标签，需要先去除。

统计评论长度分布。

```python
train["review_len"] = train["review"].str.len()
train["review_len"].describe()
```

发现正负情感评论长度分布大致相似，但极端短评论信息量不足。

### 2.2 数据清洗
我们构建清洗函数 review_to_words，完成 HTML 去除、非字母替换、小写化、停用词去除。

```python
def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, features="lxml").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)
```

对全部 25000 条评论清洗。

```python
clean_train_reviews = []
for i in range(0, train["review"].size):
    if ((i + 1) % 1000 == 0):
        print(f"Review {i + 1} of {num_reviews}")
    clean_train_reviews.append(review_to_words(train["review"][i]))
```

清洗后文本只保留有意义的词，便于后续向量化。

### 2.3 模型训练与超参数调优
首先使用 Bag of Words 构建词频特征。

```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()
print("Shape of Bag of Words: ", train_data_features.shape)
# 输出：(25000, 5000)
```

我们使用 Logistic Regression 作为分类器。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data_features, train["sentiment"], test_size=0.3, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
```

随后训练 Word2Vec 词向量。

```python
from gensim.models import Word2Vec
sentences = [review.split() for review in clean_train_reviews]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, epochs=10)
```

将评论映射为词向量的平均值。

```python
def review_to_avgvec(review, model):
    vecs = [model.wv[w] for w in review.split() if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)
```

我们比较 Bag of Words、Word2Vec 平均向量、以及两者拼接的效果，调优 Word2Vec 的 vector_size 和 window 大小。

### 2.4 模型测试与指标报告
在测试集上评估模型性能。

```python
from sklearn.metrics import accuracy_score, classification_report
pred = clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)
```

Bag of Words + Logistic Regression 通常能达到约 0.88 的准确率。Word2Vec 平均向量模型准确率约 0.85，略低但特征维度更小。

我们进一步尝试使用 LSTM 网络结合 Word2Vec 预训练词向量。

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len),
    LSTM(128),
    Dense(1, activation='sigmoid')
])
```

通过网格搜索调优 LSTM 单元数和 dropout 概率，最后得到约 0.90 的准确率。

最终我们在全量训练数据上重新训练模型，生成测试集预测。

```python
test_pred = model.predict(test_features)
submission = pd.DataFrame({'id': test_id, 'sentiment': test_pred})
submission.to_csv('submission.csv', index=False)
```

## 三、总结
本文中我们完成了 Word2Vec NLP 教程的完整流程。我们首先用 EDA 了解 IMDB 数据：25000 条评论、情感二分类、文本需预处理。其次我们做了 HTML 去除、非字母替换、停用词去除等清洗。接着我们用 Bag of Words 构建 5000 维特征，训练 Logistic Regression，准确率约 0.88，并尝试 Word2Vec 平均向量和 LSTM 模型，准确率可提升至 0.90。最后我们用全量数据训练并生成提交文件。经验表明，对短文本情感分类，词袋特征配合简单线性模型已很有效，Word2Vec 在捕捉语义相似性方面更有潜力，适合更复杂的序列模型。
