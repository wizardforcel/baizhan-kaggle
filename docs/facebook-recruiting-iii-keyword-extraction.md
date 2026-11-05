# Facebook 招募赛 III：关键词抽取（Facebook Recruiting III - Keyword Extraction）

## 一、比赛简介

### 1.1 比赛背景

Stack Overflow 是全球最大的程序员问答社区，用户提问时会自己打上几个**标签（tag）**，比如 `c#`、`java`、`php`。标签让问题能被正确归类、被懂行的人看到。这个比赛的任务就是：**只看问题的标题（Title）和正文（Body），自动预测它应该带哪些标签**。

训练数据是一份约 600 万条的问题存档，每条有 `Id`、`Title`、`Body`、`Tags` 四列，其中 `Tags` 是空格分隔的标签串。测试集只给标题和正文，要求我们输出预测的标签列表。

这是个典型的**多标签分类（multi-label classification）**问题：一条问题不是属于「一个」类别，而是同时属于多个标签。而且标签的分布极度不平衡——常用标签几十万次，长尾标签可能只出现一次。

### 1.2 比赛目标

输入是一条问题的标题（可选正文），输出一个标签集合。为了让问题可解，实践中通常把标签词表裁剪成最常见的 **N 个标签**（本次主线选 100 个），只在这 100 个里做预测。数据清洗的每个决定都会直接影响「覆盖率」：裁剪得越狠，能预测的问题越少。

### 1.3 评价标准

比赛用**多标签 F1** 衡量：预测的标签集合和真实标签集合重合得越多，分数越高。「精确率」管你预测的标签里有多少是对的，「召回率」管真实标签里你找回了多少，F1 是两者的调和平均。我们本地同时报告 accuracy、macro-F1、micro-F1 和 Hamming loss，其中 micro-F1 最接近比赛的整体口径（把所有标签位置合起来一起算）。

---

## 二、解题过程

我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步走。主线先做「TF-IDF + 线性分类器」拿到可用基线，再上 **BiLSTM** 这种能理解词序的深度模型。

### 2.1 数据探索（EDA）

首先解压读入训练集，只取需要的三列：

```python
import pandas as pd
import numpy as np

# 只读 Id/Title/Tags 三列，Body 太大先不用
train = pd.read_csv('Train.zip', usecols=['Id', 'Title', 'Tags'])
train.shape          # (6034195, 3) 约 600 万条问题
train.head()
```

第一条问题的标签是 `php image-processing file-upload upload mime-types`——一眼能看出，标签几乎都能在标题里找到线索，说明「标题里的词」是强信号。

接着统计每条问题带几个标签：

```python
# Tag_count：每个问题有几个标签，用空格切开数个数
train['Tag_count'] = train['Tags'].apply(lambda x: len(str(x).split()))
import seaborn as sns
sns.countplot(train['Tag_count'])
# 大多数问题带 2~4 个标签
```

然后统计全量标签词表和每个标签的出现频率：

```python
from sklearn.feature_extraction.text import CountVectorizer

# 用 CountVectorizer 把空格分隔的标签串变成「词袋」矩阵
tag_vectorizer = CountVectorizer(tokenizer=lambda x: str(x).split())
tag_mat = tag_vectorizer.fit_transform(train['Tags'])
tag_names = tag_vectorizer.get_feature_names()

# 全量词表有 41573 个不同标签
len(tag_names)          # 41573

# 每个标签出现的总次数，降序看前几名
tag_freq = pd.Series(tag_mat.sum(axis=0).A1, index=tag_names)
tag_freq.sort_values(ascending=False).head()
# c#            327236
# java          294484
# php           277911
# javascript    262336
# android       232395
```

这就验证了长尾假设：**4 万多个标签，但前几个高频标签就占了巨大比例**。画出的频率曲线是一条陡峭的幂律尾巴。把这组频率画成条形图或词云，排在最前面的几乎是清一色的主流编程语言和平台，这提示我们：标签预测的大头其实是「识别问题属于哪个技术栈」，其次才是「具体问的是哪个 API」。这一判断直接影响了后面的特征选择——我们优先保留标题里的技术名词，而不是泛泛的英文虚词。

更重要的是「裁剪多少标签能覆盖多少问题」。我们写了一个辅助函数：从频率最高的标签开始逐步加，每步统计「至少命中其中一个 Top-N 标签的问题」占全体的比例：

```python
def questions_covered(one_hot_tag, ntags):
    """给一个稀疏多标签矩阵，返回 Top-ntags 覆盖的问题比例"""
    nq = one_hot_tag.shape[0]                 # 总问题数
    tag_sum = one_hot_tag.sum(axis=0).tolist()[0]  # 每个标签出现次数
    # 按出现次数降序排标签序号
    tag_sorted = sorted(range(len(tag_sum)),
                        key=lambda x: tag_sum[x], reverse=True)
    # 只保留 Top-ntags 列，统计每行还剩多少标签
    topn = one_hot_tag[:, tag_sorted[:ntags]]
    tags_per_q = topn.sum(axis=1)
    # 一行标签数全为 0 = 这条问题没被覆盖
    no_tag = np.count_nonzero(tags_per_q == 0)
    return np.round((nq - no_tag) / nq * 100, 2)

# 覆盖曲线（节选）：
# #Tags   覆盖比例
# 500      91.29
# 1000     94.57
# 1500     96.11
# 2000     97.13
# 5000     98.93
```

5000 个标签能覆盖 98.93% 的问题，但要训练 5000 个输出维度代价极高。本次主线折中选了 **Top-100 标签**，把问题缩到「至少带一个 Top-100 标签」的子集（约 316 万条），代价是放弃了一些长尾问题，换来训练可行。

### 2.2 数据清洗

第一件最重要的事是**去重**。Stack Overflow 上大量问题是转载、重复提问，如果不去重，训练集和测试集会「撞题」，评估就不准：

```python
# 按 Title 去重后从 603 万降到 412 万，三分之一都是重复的
train.drop_duplicates('Title', inplace=True)
train.shape      # (4125233, 3)

# 标签为空的极少数行也去掉
train = train[train['Tags'].notnull()]
train.shape      # (4125226, 3)
```

接着对标题做文本清洗：转小写、去掉非字母字符、去停用词、做词干化（比如 `uploading` 和 `uploaded` 都变成 `upload`），让同一个词的各种变形合并：

```python
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def clean_text(sentence):
    sentence = sentence.lower()
    # 把所有非字母字符换成空格（HTML 标签、标点都会被清掉）
    sentence = re.sub(r'[^a-z]+', ' ', sentence).strip()
    words = word_tokenize(sentence)
    # 去停用词（the、a、of 这类不携带主题信息的词）
    stops = set(stopwords.words('english'))
    words = [w for w in words if w not in stops]
    # 词干化：把词还原成词根
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words]
    return ' '.join(words)

train['Title'] = train['Title'].apply(lambda x: clean_text(str(x)))
```

最后把标签矩阵裁剪成 Top-100，并把一条 Top-100 标签都没有的问题删掉：

```python
# 只保留出现次数最多的 100 个标签对应的列
y_multinomial = topn_tags(tag_vectorizer.transform(train['Tags']), 100)
# 保留至少命中一个 Top-100 标签的问题
keep = y_multinomial.sum(axis=1) != 0
y_multinomial = y_multinomial[keep.A1, :]
train = train.iloc[keep.A1, :]
# 剩下 316 万条
```

关于 `Body` 列：材料里的正文长这样 `<p>I'd like to check ...</p><pre><code>...</code></pre>`，全是 HTML 标签和代码块。它携带的信息比标题多得多，但必须先去掉 HTML、剥离代码再分词，否则 `<p>`、`<code>` 这些标记词会污染特征。本次主线只用标题，一是标题短、噪声少、训练快，二是标题里的关键词和标签重合度本来就高；用上正文是后面最容易见效的改进方向。

### 2.3 模型训练与超参数调优

**第一版：TF-IDF + One-vs-Rest SGD。** 先把标题转成 TF-IDF 向量（统计词频并压低「到处都是」的高频词），再给 100 个标签各训一个二分类器（One-vs-Rest），用的是线性模型 SGD，L1 正则让大量权重归零、只保留真正有区分力的词：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# 80% 训练、20% 测试
Xtrain, Xtest, Ym_train, Ym_test = train_test_split(
    train['Title'], y_multinomial, test_size=0.2, random_state=45)

# TF-IDF：132641 维
tfid_vec = TfidfVectorizer(tokenizer=lambda x: str(x).split())
Xtrain = tfid_vec.fit_transform(Xtrain)
Xtest = tfid_vec.transform(Xtest)

# 100 个标签各训一个线性分类器
logreg_model1 = OneVsRestClassifier(
    SGDClassifier(loss='log', alpha=0.001, penalty='l1'), n_jobs=-1)
logreg_model1.fit(Xtrain, Ym_train)
```

这一版的验证结果是 accuracy 0.123、macro-F1 0.084、micro-F1 0.274、Hamming loss 0.0135。分数不高的根源是：**线性模型只能看「词出现过没有」，看不到词序，也学不到标签之间的共现关系**，而预测时又对每个标签独立下 0.5 阈值，一有误差就放大。从规模上看也很吃紧：TF-IDF 矩阵是 2,528,076 行 × 132,641 列，而 Kaggle 内核的内存不足以同时展开 5000 个标签各自的一个分类器，材料里只训了 100 个标签就到头了。

**第二版：BiLSTM。** 我们换成能读词序的深度模型。先把标题里的词映射成整数 ID 并补齐到固定长度，再用预测练好的词向量（Embedding）把每个 ID 变成 256 维向量，交给两层双向 LSTM：

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Flatten, Dense

# 词表只保留最常见的 2 万个词，其余当未知词
t = Tokenizer(num_words=20000)
t.fit_on_texts(train['Title'].to_list())
w2num = t.word_index

# 把每条标题变成整数序列，并补齐到统一长度
docs2 = [[w2num[w] for w in doc.split()] for doc in train['Title']]
docs2 = pad_sequences(docs2)

Xtrain, Xcv, Ytrain, Ycv = train_test_split(
    docs2, y_multinomial, random_state=21, test_size=0.25)

model = Sequential()
# 词嵌入层：2 万词表 → 256 维向量，输入长度 27
model.add(Embedding(20000, 256, input_length=27))
# 第一层双向 LSTM：正向 + 反向各读一遍，捕捉上下文
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Dropout(0.4))          # 防止过拟合
# 第二层双向 LSTM
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(100, activation='sigmoid'))   # 100 个标签，各输出概率
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(Xtrain, Ytrain, epochs=10, batch_size=1024)
```

调参上主要盯着三件事：学习率用 Adam 默认值起步；Dropout 固定 0.4 防过拟合；轮数跑到 10 轮，观察到验证 loss 还在稳步下降。BiLSTM 一次前向要 380 秒，10 轮训练接近一小时，所以没有做大规模网格搜索——**先把结构跑通，再考虑加注意力、加更多轮数**是更现实的策略。

训练日志本身就是重要信号：10 轮里 loss 从 3.309 一路降到 2.580，训练准确率从 0.491 升到 0.577，而且最后一轮还在涨——这说明模型远没有收敛，如果再给更多轮数或更宽的层，分数还能往上走。反过来说，如果训练准确率早早拉满、验证分数却掉头，那就是该停下来加正则的时候。

### 2.4 模型测试与指标报告

两个模型在验证集上的表现对比：

```python
from sklearn import metrics

# 概率转成 0/1：超过 0.5 就算预测了该标签
Ycv_pred = (model.predict(Xcv) > 0.5).astype('int64')

print("Accuracy :", metrics.accuracy_score(Ycv, Ycv_pred))
print("F1 macro :", metrics.f1_score(Ycv, Ycv_pred, average='macro'))
print("F1 micro :", metrics.f1_score(Ycv, Ycv_pred, average='micro'))
print("Hamming  :", metrics.hamming_loss(Ycv, Ycv_pred))
```

```
模型            Accuracy   macro-F1   micro-F1   Hamming loss
SGD + TF-IDF    0.1230     0.0844     0.2739     0.0135
BiLSTM          0.3557     0.5729     0.6253     0.0109
```

BiLSTM 的 micro-F1 从 0.274 提到 **0.625**，macro-F1 从 0.084 提到 0.573，提升非常明显，说明词序和上下文确实是预测标签的关键。Hamming loss 也下降了，意味着「错误预测的标签位」变少。

提交时对测试集的标题走同样的清洗、编码、补齐流程，用 `model.predict` 得到每条问题在 100 个标签上的概率，取概率最高的 3~5 个标签作为答案输出。

这里有个影响分数的细节：我们本地报告结果时用的是 **0.5 固定阈值**（概率超一半才输出该标签），但线上提交往往是**按概率取 Top-k**。两种策略的取舍不同——固定阈值能控制「不预测无关标签」，Top-k 能保证每条问题都有 k 个答案，多标签 F1 在「标签少的短问题」上通常更吃 Top-k。我们两套都跑过、按验证集 micro-F1 选更优的提交。

需要诚实说明两点局限：一是这段代码在输出层用了 `categorical_crossentropy`，而多标签任务的标签之间并不互斥，严格来说应该用 `binary_crossentropy` 更合适；二是材料中只用了标题、没有用正文，正文里有大量代码和上下文，用上之后分数还能再涨——这也是材料作者在结尾说的改进方向。

---

## 三、总结

本文中我们完成了 Stack Overflow 标签预测的完整流程。我们首先用 EDA 摸清了数据：约 600 万条问题、4 万多个标签、频率呈幂律分布，Top-5000 标签能覆盖 98.93% 的问题；其次做了数据清洗，按标题去重去掉三分之一重复项，再对标题做小写化、去非字母字符、去停用词和词干化，并把标签裁剪成 Top-100、保留至少命中一个标签的 316 万条问题；接着对比了 TF-IDF + One-vs-Rest SGD 与两层 BiLSTM 两个模型；最后在验证集上得到 BiLSTM 的 micro-F1 为 0.625、macro-F1 为 0.573，明显优于线性基线的 0.274 / 0.084。

这个比赛给我们的最大启发是：**多标签任务里，「裁剪标签」和「清洗文本」比换模型更先决**。4 万个标签不可能全部学会，先用覆盖率曲线选一个合理的 N；去重和词干化直接决定了训练集质量。模型只是最后一步放大器——数据没弄清楚，再深的网络也只是更快地记住噪声。