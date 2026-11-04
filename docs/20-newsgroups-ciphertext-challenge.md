# 20 Newsgroups 密文挑战（Ciphertext Challenge）

## 一、比赛简介

### 1.1 比赛背景

20 Newsgroups 是机器学习圈子里非常有名的一个文本数据集。它收集了 1990 年代 Usenet 上 20 个新闻组（newsgroup）的帖子，每个新闻组对应一个话题，比如 `comp.graphics`（计算机图形）、`rec.sport.hockey`（冰球）、`sci.space`（太空）、`talk.politics.guns`（枪支政治）等等。因为话题之间差别很大，它一直被人们拿来做**文本分类**的入门练习。

这个比赛做的事情是：把 20 Newsgroups 的帖子**加密**之后交给参赛者，让大家在「看不到原文」的情况下，猜出每段密文属于哪个话题。

比赛把加密分成 4 个难度等级（difficulty）：

| 难度 | 含义 | 直观理解 |
| --- | --- | --- |
| 1 | 只对部分字符做了替换，很多字母还是原样 | 像把一篇文章里的部分字母涂黑 |
| 2 | 加密范围更大，能看出来的字母更少 | 大部分字母都被涂黑 |
| 3 | 几乎整篇都被替换，只剩少量可读字符 | 只剩零星字母能认 |
| 4 | 完全加密，整段都是乱码 | 通篇都是符号，一个字母都认不出 |

难度越高，能利用的「明文字母」越少，分类就越难。比赛希望参赛者想办法**只用密文**就把话题猜对，而不是真的把密码破译出来。

### 1.2 比赛目标

给定一条密文，预测它属于 20 个话题中的哪一个。训练集里每条样本包含四列：

- `Id`：样本编号；
- `difficulty`：加密难度（1～4）；
- `ciphertext`：加密后的文本；
- `target`：话题编号（0～19）。

测试集只有前三列，我们要给每条测试样本填上 `target`。

### 1.3 评价标准

比赛使用**分类准确率（Categorization Accuracy）**作为评价指标，也就是预测对的话题条数占全部测试样本的比例。多分类问题里，准确率就是「猜对了多少比例」，不看概率只看得分最高的那一个类别。最终排行榜上的分数越高越好，满分是 1.0。

> 小提示：因为题目按难度分层，而不同难度的难度天差地别，所以比赛中经常要**按难度分别看分数**，而不是只看一个总准确率。我们下面也是这么做的。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**先把密文当普通文本，用字符级的词袋把它的统计特征抓出来，再喂给一个线性分类器。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步是把三个文件读进来，看看数据长什么样、有多少条、难度分布是否均衡：

```python
# 导入常用库：numpy 做数值计算，pandas 读写表格
import numpy as np
import pandas as pd

# 训练集、测试集、提交样例
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')

# 看前几行，确认列名和数据格式
data.head()
```

打印出来的 `ciphertext` 大致长这样：

```
ob|IK?zzhX*L{83B3Z,FuL*Pusm$83L\t@r$$*38,8s...
```

能看到里面有各种括号、斜杠、美元符号，几乎没有正常单词，说明加密确实把字母换掉了。

接着我们统计难度的分布和每个话题的样本数：

```python
# 四个难度各有多少条样本
data['difficulty'].value_counts()
# 输出：
# 2    10024
# 4     9970
# 1     9589
# 3     9469
```

四个难度差不多各占四分之一，说明数据是**按难度均匀分配**的，不存在某个难度样本特别少的问题。

再看话题的分布：

```python
# 每个话题（0~19）各有多少条样本
data['target'].value_counts().to_frame().T
```

话题之间有一些不均衡：最多的 `17` 号有 3153 条，最少的 `6` 号只有 1236 条，相差两倍多。这在 20 个真实新闻组里是正常的——有的新闻组本来就比别的热闹。

我们还想知道难度和话题有没有关系，于是做了一张交叉表：

```python
# 行是难度，列是话题，看两者的组合分布
pd.crosstab(data['difficulty'], data['target'])
```

结果显示每个难度里 20 个话题都有，分布也比较接近，说明比赛**没有故意把某个话题只放在高难度里**。这意味着同一个分类模型可以对所有难度通用。

最后看密文长度：

```python
# 每条密文的字符数
data['ciphertext'].apply(len).describe()
# count    39052.000000
# mean       278.483356
# std         61.807633
# min          2.000000
# 50%        300.000000
# max        300.000000
```

这里透露了一个非常重要的信息：**绝大多数密文刚好是 300 个字符**，只有极少数特别短（最短只有 2 个字符）。说明原始帖子的正文被截断到 300 字符再加密。也就是说，每条样本携带的信息量是固定的、很有限的，这也解释了为什么后面准确率很难冲到很高。

### 2.2 数据清洗

这份数据本身比较「干净」：没有缺失值（`ciphertext` 和 `target` 都有值），没有需要去重的明显重复，也没有时间字段需要对齐。需要处理的主要是两点：

1. **极短的样本**。最短的密文只有 2 个字符，几乎不可能从中判断话题。我们不在训练时删除它们（每条样本都是有效标签），但心里要有数：这类样本注定预测不准。
2. **不做常规的文本清洗**。平时做文本分类会做转小写、去标点、去停用词，但这里**不能这么做**——因为密文里的标点符号本身就是加密后的字母，一旦删掉等于把信息扔了。所以我们原样保留密文字符串。

我们最终采用的清洗方式非常轻：只把 `ciphertext` 列取出来，别的都不动。这也提醒我们一个通用原则——**清洗要服务于任务**，别人家的标准流程不一定适合你手上的数据。

### 2.3 模型训练与超参数调优

这是整个比赛的核心。我们的做法分两层：

**第一层：把密文变成向量。** 正常文本分类用「词袋」（把每个单词出现的次数数出来），但密文里没有单词，所以我们改用**字符级 n-gram**：把密文切成 1～6 个字符的连续片段，统计每种片段出现的次数。虽然单个字符看不出意义，但「某些字符组合出现得多」这件事本身就是有规律的——不同的加密方式、不同的原始文本，字符组合的分布不一样，模型可以学到这种统计规律。

**第二层：分类器。** 我们用一个带核函数的线性 SVM（`SGDClassifier(loss='hinge')`），它对高维稀疏的文本特征效果稳定、训练也快。

把两步串成一个 `Pipeline`，好处是特征提取和模型训练绑在一起，交叉验证时不会泄漏，调参也方便：

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import FunctionTransformer

# 第一步：从表格里取出 ciphertext 这一列，变成一维的文本
selector = FunctionTransformer(lambda x: x['ciphertext'], validate=False)

# 第二步：字符级 n-gram 计数
#   analyzer='char'  表示按字符切，而不是按单词切
#   ngram_range=(1, 6)  表示从 1 个字符到 6 个字符的组合都统计
#   lowercase=False  密文里大小写是加密信息的一部分，不能统一转小写
vectorizer = CountVectorizer(
    analyzer='char',
    lowercase=False,
    ngram_range=(1, 6)
)

# 第三步：把计数转换成 TF-IDF 权重
#   出现次数多的片段不一定重要，TF-IDF 会压低「到处都出现」的片段
tfidf = TfidfTransformer()

# 第四步：线性 SVM 分类器
#   loss='hinge' 就是标准的 SVM 损失
#   max_iter=1000 保证迭代收敛
estimator = SGDClassifier(
    loss='hinge',
    max_iter=1000,
    random_state=0,   # 固定随机种子，结果可复现
    tol=1e-3,
    n_jobs=-1
)

model = Pipeline([
    ('selector', selector),
    ('vectorizer', vectorizer),
    ('tfidf', tfidf),
    ('estimator', estimator),
])
```

**关于数据划分。** 因为测试集没有标签，我们没法在本地直接算排行榜分数。为了能调参、能判断模型好坏，我们从训练集里再切出一部分当验证集。这里用**分层抽样**（`stratify=y`），保证切出来的验证集里 20 个话题的比例和原数据一致：

```python
from sklearn.model_selection import train_test_split

# X 是特征表（去掉 target），y 是标签
X = data.drop('target', axis=1)
y = data['target']

# 80% 训练、20% 验证；stratify=y 让两边的话题分布一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# 在训练集上拟合整个流水线
model.fit(X_train, y_train)
```

**关于调参。** 我们试过用 `GridSearchCV` 在 n-gram 范围、TF-IDF 是否使用、SVM 的正则强度等参数上搜索。但这里的现实是：特征维度极高（6 万级）、样本只有 4 万条，网格搜索每跑一组都要重新做一次字符 n-gram 统计，非常耗时。经过几次小规模试验，`ngram_range=(1, 6)` 配合默认正则强度已经比较稳，进一步细调带来的提升很小，所以我们把精力放在了「按难度分别分析」上，而不是死磕单个超参数。

### 2.4 模型测试与指标报告

先看整体表现：

```python
from sklearn.metrics import f1_score, classification_report

# 在验证集上预测
y_pred = model.predict(X_test)

# 打印每个话题的精确率、召回率、F1
print(classification_report(y_test, y_pred))

# 整体准确率（就是比赛的评分方式）
acc = (y_pred == y_test).mean()
print("Accuracy:", acc)
```

我们得到的整体准确率大约是 **0.50**，宏平均 F1 约 **0.48**。这个数字单看不高，但要看它是由什么组成的。

**关键的一步：按难度拆开看分数。** 这是我们在这个比赛里做的最有价值的分析：

```python
def get_f1_score(difficulty):
    """只挑出某个难度下的验证样本，单独算宏平均 F1"""
    mask = X_test['difficulty'] == difficulty
    return f1_score(y_test[mask], y_pred[mask], average='macro')

print("f1_score per difficulty")
for i in range(1, 5):
    print("Difficulty: {} ==> {:5f}".format(i, get_f1_score(i)))
```

结果非常说明问题：

```
Difficulty: 1 ==> 0.623181
Difficulty: 2 ==> 0.640006
Difficulty: 3 ==> 0.428171
Difficulty: 4 ==> 0.114732
```

读法很清楚：

- 难度 1、2 的密文里还保留了不少原文字符，模型能从这些「漏网之鱼」里学到话题线索，F1 在 0.6 以上；
- 难度 3 只剩零星有意义的字符，F1 掉到 0.43；
- 难度 4 完全加密，F1 只有 0.11，基本接近「瞎猜 20 选 1」（随机猜的准确率是 1/20 = 0.05），只比纯猜好一点点。

这个结果告诉我们两件事：**一是模型本身是有效的**（低难度上大幅超过随机），**二是高难度上光靠字符统计确实不够**。这也解释了为什么比赛中很多队伍会去研究「密文的加密规律」——如果能找出加密方式，把密文「部分还原」，高难度也能大幅提分。

**提交文件。** 确认模型可用后，我们用全量训练数据重新拟合一次（多喂数据总没坏处），再对测试集预测：

```python
# 用全部训练数据重新训练
model.fit(X, y)

# 对测试集做同样的流程：选列 → 字符 n-gram → TF-IDF → 预测
test_pred = model.predict(test)

# 按比赛要求的格式生成提交文件：id + topic
submission = pd.DataFrame({
    'id': test['Id'],
    'topic': test_pred
})
submission.to_csv('submission.csv', index=False)
```

最后我们还会随机挑几行提交文件看一眼，检查列名对不对、行数是不是和测试集一致——这一步看似琐碎，但比赛里因为提交格式错误白白丢分的情况并不少见。

---

## 三、总结

本文中我们完成了 20 Newsgroups 密文挑战的完整流程。我们首先用 EDA 摸清了数据：4 万条样本、4 个难度均匀分布、20 个话题、密文基本都被截断在 300 字符。其次我们做了最少量的数据清洗——因为密文里的标点本身就是加密信息，所以不转小写、不去标点、不去停用词。接着我们用字符级 1～6 gram 的 TF-IDF 特征加线性 SVM 组成流水线，并特意按难度拆开评估。最后我们在验证集上得到整体准确率约 0.50，其中难度 1、2 的 F1 在 0.6 以上，难度 4 只有 0.11，说明「只靠字符统计」这条路在完全加密的数据上会遇到天花板。

这个比赛给我们的最大启发是：**特征工程要跟着数据走**。同样是文本分类，正常文本用词袋，密文就得用字符 n-gram；同样是评估，总分会把难度的差异掩盖掉，拆开看才能知道自己的模型真实强在哪、弱在哪。
