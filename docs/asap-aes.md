# ASAP 自动作文评分（Automated Essay Scoring）

## 一、比赛简介

### 1.1 比赛背景

老师批改作文是一件极其耗时的事情。一个班五十个学生，每人一篇作文，老师要逐篇读完、打一个分数，还要保证前后标准一致——今天心情好给 8 分、明天心情差给 7 分，学生就会觉得不公平。如果能用一个模型先给出分数，老师只做复核，批改效率会高很多。这类任务叫 **自动作文评分**（Automated Essay Scoring，简称 AES）。

这个比赛由 **Hewlett Foundation**（休利特基金会）资助，在 Kaggle 上举办。训练数据是真实的学生作文，每篇都由人工评过 1～3 次分，官方把评分员的分数合并成一个最终分数，作为我们要预测的目标。为了保护隐私，作文里的人名、地名、机构名都被替换成了 `@CAPS1`、`@LOCATION1` 这样的占位符。

这个任务有三个难点。

第一，**主观性**。同一篇作文，两位老师打分不一致是常态，所以模型不可能追求「完全预测对」，能逼近人类评分员之间的一致程度就已经很好了。评价指标也专门为此设计（见 1.3）。

第二，**分数不是均匀的连续值**。大多数学生拿中间分，很高分和很低分都很少，这给回归模型带来麻烦——模型倾向于预测平均值，从而丢掉极端样本。

第三，**不同题目不能混着比**。训练集包含 8 个 essay set（可以理解成 8 道不同的作文题），每道题的评分标准、满分、学生年级都不一样。把第 1 题的 8 分和第 8 题的 8 分放在一起训练，模型会学乱。

### 1.2 比赛目标

给定一篇作文，预测它的分数。训练集一共 **12976 篇**作文，主要列如下：

| 列名 | 含义 |
| --- | --- |
| `essay_id` | 作文编号 |
| `essay_set` | 属于哪一道题（1～8） |
| `essay` | 作文正文 |
| `rater1_domain1` / `rater2_domain1` / `rater3_domain1` | 三位评分员给出的分数 |
| `domain1_score` | 官方合并后的最终分数，**这就是我们的标签** |
| `domain2_score` | 第二个评分维度的分数，只有 1800 条非空 |

八个 essay set 的样本数并不平均：第 1 题 1783 篇、第 2 题 1800 篇、第 3 题 1726 篇、第 4 题 1770 篇、第 5 题 1805 篇、第 6 题 1800 篇、第 7 题 1569 篇、第 8 题只有 723 篇。

标签的分布是：均值 6.80，标准差 8.97，最小值 0，最大值 60。**标准差比均值还大**，说明分数跨度极宽——因为不同题目的满分不同，最高的那道题能到 60 分，而简单的题目满分只有个位数。这一点直接决定了我们后面不能简单地把所有题目混在一起做一个纯回归。

### 1.3 评价标准

比赛用 **Quadratic Weighted Kappa（二次加权 Kappa，简称 QWK）** 评分。

用大白话讲，QWK 衡量的是「模型的打分和人类评分员有多一致」，并且做了两件事：

1. **离得越远罚得越重。** 真实 8 分、预测 7 分只是小错；真实 8 分、预测 2 分就是大错，惩罚按「差距的平方」增长。
2. **扣掉「瞎猜也能蒙对」的部分。** 如果评分分布本身就很集中（比如九成作文都是 8 分），那么一个永远预测 8 分的模型也能「碰对」很多，QWK 会把这种碰运气折算掉。

取值上：1.0 表示完全一致，0 表示和随机猜差不多，负数表示比随机猜还差。**越高越好。**

理解 QWK 有一个实际好处：它惩罚「差得远」，所以模型宁可预测 7.5 也不要预测 2；但它又是按整数档位算的，所以最后要把连续预测四舍五入成整数。

---

## 二、解题过程

我们的整体思路是：**先把作文变成向量，再用循环神经网络做回归**。具体来说，我们在训练集上自己训练一个 Word2Vec 词向量模型，把每篇作文里所有单词的向量取平均，得到一个 300 维的「作文向量」，然后喂给两层 LSTM 输出一个分数。下面按四步展开。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来。注意这份数据是 TSV（用制表符分隔），而且编码是 `ISO-8859-1`，不是常见的 UTF-8——如果按默认编码读，遇到特殊字符就会报错：

```python
import pandas as pd
import numpy as np

# 训练集只需要四列：编号、题目、正文、最终分数
# dropna(axis=1) 会顺手把全空的列删掉（这份表里有很多评分维度是空的）
training_data = pd.read_csv(
    "/kaggle/input/asap-aes/training_set_rel3.tsv",
    sep='\t',
    encoding='ISO-8859-1',
    usecols=['essay_id', 'essay_set', 'essay', 'domain1_score'],
).dropna(axis=1)

# 测试集和验证集同样处理，后面提交要用
test_data = pd.read_csv("/kaggle/input/asap-aes/test_set.tsv",
                        sep='\t', encoding='ISO-8859-1').dropna(axis=1)
valid_data = pd.read_csv("/kaggle/input/asap-aes/valid_set.tsv",
                         sep='\t', encoding='ISO-8859-1').dropna(axis=1)

print(training_data.shape)   # (12976, 4)
training_data.head()
```

打印出来第一行正文是这样的：

```
Dear local newspaper, I think effects computer...
```

第二行是 `Dear @CAPS1 @CAPS2, I believe that using compu...`。可以看到 `@CAPS1` 就是被替换掉的人名或机构名。这个细节后面清洗时很重要。

接着我们看标签的统计和题目的分布：

```python
# 标签的整体统计：均值、标准差、最值
print(training_data['domain1_score'].describe())
# count   12976.000000
# mean        6.800247
# std         8.970705
# min         0.000000
# max        60.000000

# 每道题各有多少篇作文
print(training_data.groupby('essay_set').size())
# 1    1783
# 2    1800
# 3    1726
# 4    1770
# 5    1805
# 6    1800
# 7    1569
# 8     723
```

这里有两个数字很关键。

`std = 8.97` 大于 `mean = 6.80`，这在自然数据里不常见。原因不是「分数两极分化」，而是**八道题的满分不同**：有的题满分 12，有的题满分 60。所以「8 分」在第 1 题可能是高分，在第 8 题却是低分。如果把八道题混在一起训练一个回归模型，模型会同时看到「8 分是好成绩」和「8 分是差成绩」两种信号，学出来的东西必然是矛盾的。

第 8 题只有 723 篇，是第 2 题（1800 篇）的四成。样本量小意味着这道题的模型更容易过拟合。

我们还检查了 `domain2_score`：只有 1800 条非空。第二个评分维度只覆盖了八道题里的一道，数据量太少，所以我们**不使用它**，只预测 `domain1_score`。

### 2.2 数据清洗

文本清洗这一步，我们要做的是「去掉噪音、保留语义」。材料里的做法是三级清洗：先正则替换掉非字母字符，再转小写，最后去停用词。

```python
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def essay_to_wordlist(essay_v, remove_stopwords):
    # 把所有非字母字符替换成空格：
    #   这样 @CAPS1、@LOCATION2 这些占位符、数字、标点会一起被清掉，
    #   它们对打分没有帮助，留着反而干扰词表
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)

    # 转小写并切词：The 和 the 应该被当成同一个词
    words = essay_v.lower().split()

    # 去停用词：the / is / at 这类词到处都有，不携带区分度
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]

    return words

# 先拿一篇试试效果
print(essay_to_wordlist(training_data['essay'][0], remove_stopwords=True)[:20])
```

处理完之后，`Dear @CAPS1 @CAPS2, I believe...` 会变成 `['dear', 'believe', ...]`——人名占位符消失了，只留下真正表达内容的词。

第二级清洗是**分句**。为什么要分句？因为我们要训练 Word2Vec，而 Word2Vec 的假设是「同一个句子里的词意思相近」。如果把整篇作文当成一句话，一篇 300 词的作文里，开头和结尾的词也会被强行认为相关，学出来的词向量会很糊。分句之后，「句内共现」这个假设才算成立：

```python
def essay_to_sentences(essay_v, remove_stopwords):
    # 用 nltk 的 punkt 分句器把长文切成一句一句
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:   # 跳过空串，否则会给 Word2Vec 喂空句子
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences
```

还有一条清洗原则要说清楚：我们**没有做拼写纠错、没有做词形还原**。原因是这份数据里的拼写错误本身可能就是评分线索（拼错多的作文往往得分低），强行「修正」等于把模型的判断依据抹掉了。

### 2.3 模型训练与超参数调优

我们的模型分成「词向量」和「打分器」两层。

**第一层：自己训练词向量。** 我们没有用现成的预训练词向量，而是在每个训练折的作文上现场训练一个 Word2Vec。这样做的理由是：学生作文的用词和新闻语料差别很大，通用词向量在这里未必贴合；而且自己训练可以保证词表里装的是本批数据真正出现的词。

```python
from gensim.models import Word2Vec
import numpy as np

# 把所有训练作文切成句子，拼成一个大列表喂给 Word2Vec
sentences = []
for essay in train_essays:
    sentences += essay_to_sentences(essay, remove_stopwords=True)

num_features = 300    # 每个词用 300 维向量表示
min_word_count = 40   # 出现少于 40 次的词直接丢弃：样本太少，向量学不准
num_workers = 8       # 8 个线程并行训练
context = 10          # 上下文窗口 10 个词：比默认的 5 宽，更能抓住长句语义
downsampling = 1e-3   # 高频词下采样，避免 the、and 这类词主导训练

model = Word2Vec(sentences,
                 workers=num_workers,
                 size=num_features,
                 min_count=min_word_count,
                 window=context,
                 sample=downsampling)
model.init_sims(replace=True)   # 把向量归一化，后面算平均时不用再处理
```

**第二层：把作文压成一个向量。** 一篇作文有几十上百个词，而我们要输出一个分数。最直接的办法是「词向量求平均」：把一个句子里所有词的 300 维向量加起来除以词数，得到这个句子的向量；再把一篇作文所有句子的向量平均，得到整篇作文的向量。

```python
def makeFeatureVec(words, model, num_features):
    """把一段文字的每个词向量求平均，得到固定长度的特征向量"""
    featureVec = np.zeros((num_features,), dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index2word)   # 词表，用来判断词是否认识

    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec, model[word])

    # 除以实际命中的词数，而不是总词数：
    # 不认识的词没参与累加，不该让它拉低平均值
    featureVec = np.divide(featureVec, num_words)
    return featureVec

def getAvgFeatureVecs(essays, model, num_features):
    """对一批作文逐个做上面的平均，返回 (作文数, 300) 的矩阵"""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays), num_features), dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter += 1
    return essayFeatureVecs
```

这一步有一个明显的代价：**求平均丢掉了词序**。「狗咬人」和「人咬狗」的平均向量完全一样。这是这套方案最大的短板，我们在总结里还会提。

**第三层：LSTM 回归器。** 网络结构是两层 LSTM 加一个输出神经元：

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def get_model():
    model = Sequential()
    # 第一层 LSTM：300 个单元，输入形状 [时间步, 特征维度]
    # dropout=0.4 丢输入，recurrent_dropout=0.4 丢循环连接，双管齐下防过拟合
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4,
                   input_shape=[1, 300], return_sequences=True))
    # 第二层 LSTM：把 300 维压到 64 维，只输出最后一步
    model.add(LSTM(64, recurrent_dropout=0.4))
    # 全连接前再丢一半，进一步抑制过拟合
    model.add(Dropout(0.5))
    # 输出一个分数；用 relu 保证预测不会是负数（分数最低是 0）
    model.add(Dense(1, activation='relu'))

    # 回归任务用均方误差做损失，rmsprop 是这个规模下比较稳的优化器
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    return model
```

这里要坦白一点：因为输入被压成了 `(1, 300)`——**只有 1 个时间步**，LSTM 其实没有「序列」可看，它退化成了一个带门控的全连接层。这不是设计上的精巧之处，而是这套方案为了省事付出的代价。真正利用词序的做法是把每个词的向量按顺序排成 `(词数, 300)` 喂进去，但那样每篇作文长度不同，需要截断或补齐，训练也会慢很多。

**交叉验证。** 数据量只有一万多篇，单次划分的运气成分太大，所以我们用 5 折交叉验证，每一折都重新训练 Word2Vec 和 LSTM：

```python
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score

cv = KFold(n_splits=5, shuffle=True)   # shuffle 打乱顺序，避免按题号切分
results = []
count = 1

for traincv, testcv in cv.split(X):
    print("--------Fold {}--------".format(count))

    # 注意这里变量名和直觉相反：testcv 是验证折，traincv 是训练折
    X_test, X_train = X.iloc[testcv], X.iloc[traincv]
    y_test, y_train = y.iloc[testcv], y.iloc[traincv]

    # 每一折都只用本折的训练文本重建词向量，防止验证集信息泄漏
    sentences = []
    for essay in X_train['essay']:
        sentences += essay_to_sentences(essay, remove_stopwords=True)
    w2v = Word2Vec(sentences, workers=8, size=300, min_count=40,
                   window=10, sample=1e-3)
    w2v.init_sims(replace=True)

    # 把作文转成 300 维向量，再 reshape 成 LSTM 要的三维形状
    trainDataVecs = getAvgFeatureVecs(
        [essay_to_wordlist(e, True) for e in X_train['essay']], w2v, 300)
    testDataVecs = getAvgFeatureVecs(
        [essay_to_wordlist(e, True) for e in X_test['essay']], w2v, 300)
    trainDataVecs = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, 300))
    testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, 300))

    lstm_model = get_model()
    # batch_size=64、epochs=2：数据少、模型大，训两轮就够，再多必然过拟合
    lstm_model.fit(trainDataVecs, y_train, batch_size=64, epochs=2)

    y_pred = lstm_model.predict(testDataVecs)
    # QWK 是按整数档位算的，所以预测值必须四舍五入成整数
    y_pred = np.around(y_pred)

    result = cohen_kappa_score(y_test.values, y_pred, weights='quadratic')
    print("Kappa Score: {}".format(result))
    results.append(result)
    count += 1
```

关于超参数，我们做了几处有依据的选择，但**没有做系统性的网格搜索**：

- `epochs=2` 是试出来的。材料里的训练日志显示，第一轮 loss 从 79.1 降到第二轮 43.8，两轮之内损失还在快速下降；但再往下训，训练损失会继续降而验证集 Kappa 不再提升——因为每折只有一万条左右样本，而模型有 81 万个参数。
- `min_word_count=40` 是为了控制词表规模。学生作文里大量词只出现一两次，这些词的向量基本是随机的，留在词表里只会给平均向量添噪音。
- 学习率、网络宽度都没有调。理由很实际：每换一组参数就要重训 5 次 Word2Vec 加 5 次 LSTM，一轮实验要几十分钟，在算力有限时把时间花在「换更好的特征」上比「微调超参数」划算。

### 2.4 模型测试与指标报告

五个折的 Kappa 分别是：

```
Fold 1: 0.7392726556880542
Fold 2: 0.7384012076017277
Fold 3: 0.7495980850853017
Fold 4: 0.7437975877154102
Fold 5: 0.7418060024273516
```

平均下来是 **0.7426**。

```python
print("Average Kappa score after a 5-fold cross validation: ",
      np.around(np.array(results).mean(), decimals=4))
# Average Kappa score after a 5-fold cross validation:  0.7426
```

这个结果要分两方面读。

**好的一面：五个折的分数非常接近**（最低 0.7384，最高 0.7496，相差不到 0.011）。说明模型的表现是稳定的，不是碰巧某一折运气好。交叉验证的意义就在这里——它告诉你「换一批数据，你大概还能拿多少分」。

**不够好的一面：0.74 说明模型只抓住了一部分规律。** 材料里没有给出人类评分员之间 Kappa 的具体数值，所以我们无法直接说「离人类还有多远」；但从任务本身看，0.74 意味着模型能分辨「明显写得好」和「明显写得差」，在中间档位上的判断就比较粗糙了。回头看我们的方案，短板很清楚：词向量求平均把词序抹平了，模型只能感知「这篇作文用了哪些词」，感知不到「这些词是怎么组织的」。而作文评分恰恰很看重组织（有没有论点、有没有转折、段落是否清晰）。

**提交环节。** 训练完成后，我们用同样的流程处理测试集：清洗 → 转向量 → 模型预测 → 取整 → 写文件。

```python
# 提交阶段不再留验证折：用全部训练数据重训一次词向量
sentences = []
for essay in training_data['essay']:
    sentences += essay_to_sentences(essay, remove_stopwords=True)
final_w2v = Word2Vec(sentences, workers=8, size=300, min_count=40,
                     window=10, sample=1e-3)

# 训练集也要用「这一套」词向量重新转向量，保证和测试集在同一空间
train_vecs = getAvgFeatureVecs(
    [essay_to_wordlist(e, True) for e in training_data['essay']], final_w2v, 300)
train_vecs = np.reshape(train_vecs, (train_vecs.shape[0], 1, 300))

# 用全量数据重训网络：材料里是直接保存第 5 折的权重去预测，
# 但每一折的 Word2Vec 都是独立训练的，第 5 折的向量空间和这里的新空间不通用，
# 混着用会让预测失真，所以我们宁可多训一次
final_model = get_model()
final_model.fit(train_vecs, y, batch_size=64, epochs=2)

# 测试集只能用训练集学到的词表，绝不能反过来在测试集上训练
test_vecs = getAvgFeatureVecs(
    [essay_to_wordlist(e, True) for e in test_data['essay']], final_w2v, 300)
test_vecs = np.reshape(test_vecs, (test_vecs.shape[0], 1, 300))

pred = np.around(final_model.predict(test_vecs)).astype(int)

# 按官方样例的格式：id + 预测分数
submission = pd.DataFrame({
    'essay_id': test_data['essay_id'] if 'essay_id' in test_data.columns
                else test_data.iloc[:, 0],
    'predicted_score': pred.ravel(),
})
submission.to_csv('submission.csv', index=False)
```

最后我们会检查两件事：预测分数是否落在合理区间（不应该出现负数，也不应该超过该题的满分），以及行数是否和样例提交一致。

---

## 三、总结

本文中我们完成了 ASAP 自动作文评分的完整流程。我们首先用 EDA 摸清了数据：12976 篇作文、8 道题、标签均值 6.80 而标准差 8.97，说明不同题目的满分差异很大；`domain2_score` 只有 1800 条非空，因此只预测 `domain1_score`。其次我们做了三级文本清洗：正则去掉包括 `@CAPS1` 占位符在内的非字母字符、转小写、去停用词，并把作文切成句子，好让 Word2Vec 的「句内共现」假设成立。接着我们在每一折训练集上现场训练 300 维 Word2Vec，把每篇作文的所有词向量求平均压缩成一个 300 维向量，再喂给「LSTM(300) → LSTM(64) → Dropout(0.5) → Dense(1)」的回归网络，损失用均方误差，训练 2 轮、批大小 64。最后我们用 5 折交叉验证评估，五折 QWK 分别是 0.7393、0.7384、0.7496、0.7438、0.7418，平均 0.7426，并把测试集预测取整后写成提交文件。

这个方案最大的教训是**特征决定了天花板**。词向量求平均这个操作把「作文向量」压成了词袋的连续版本，词序信息全部丢失，而作文评分很大程度上依赖结构和逻辑。所以 0.74 大概就是这套特征的上限，再怎么调 LSTM 也很难突破。资料里还有一份材料把所有分数重新映射到 1～10 并用同样的 LSTM 结构训练 50 轮，报告的平均 Kappa 达到 0.9217，但它的数据划分方式与我们这条主线不同（每折的样本构成不一样），两个数字不能直接比较，只能说明「分数重标定 + 更长训练」这条方向值得单独再试一次。

如果要把这个任务做得更好，方向应该是：按 essay_set 分别建模（消除满分不同带来的矛盾），以及把词向量按顺序排成真正的序列输入，让模型能看到句子结构和段落组织，而不是只看「用了哪些词」。
