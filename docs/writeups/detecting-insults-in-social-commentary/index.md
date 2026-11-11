# 社交媒体评论中的侮辱性言论识别（Detecting Insults in Social Commentary）

## 一、比赛简介

### 1.1 比赛背景

这个比赛由反垃圾、反滥用公司 **Impermium** 赞助（数据目录里的 `impermium_verification_set.csv` 等文件就是它提供的验证集）。比赛的场景非常贴近现实：网络评论区里总有人对别人进行人身攻击、辱骂、贴标签。平台如果靠人工审核，成本高、速度慢；靠关键词过滤，又容易误伤（比如「你真是一头猪」是侮辱，但「猪肉价格上涨」不是）。所以大家希望训练一个模型，**读一句话就能判断它是不是在侮辱别人**。

这个任务难在「侮辱」是一种语义和语气现象，不是某个词出现就算数。同样一个脏词，可能是骂人，也可能是在引用别人的话；反过来，有些句子一个脏字都没有，却充满恶意。这也是为什么参赛者大量依赖文本预处理、n-gram 特征和线性分类器来做基线。

### 1.2 比赛目标

给定一条社交媒体评论，预测它是否具有侮辱性。训练数据是一张表，主要有三列：

- `Insult`：标签，`1` 表示是侮辱性评论，`0` 表示不是；
- `Date`：评论时间，约 18% 的行是缺失的；
- `Comment`：评论文本本身。

测试集只有 `Comment`（和编号），我们要给每条测试评论预测一个 `Insult` 值（0 或 1）。

### 1.3 评价标准

比赛的官方指标是 **ROC 曲线下的面积（AUC，Area Under the Curve）**，取值范围 0～1，**越高越好**。AUC 可以粗略理解成「随机抽一条侮辱评论和一条正常评论，模型给侮辱那条打分更高的概率」——0.5 相当于瞎猜，1.0 是完美区分。因为提交的是概率值而不是硬标签，所以模型输出概率的排序质量比「卡在 0.5 这条线切」更重要。

> 小提示：材料里的 notebook 为了方便看结果，本地统计的是**准确率、加权 F1、精确率、召回率**；我们下面报告这些数字时会注明它们不是官方 AUC。

---

## 二、解题过程

我们首先读入数据、看类别和缺失情况，其次做一整套文本清洗（缩写展开、去符号、去停用词、纠正重复字母），接着用 1～3 元词袋加 TF-IDF 把文本变成向量，然后训练朴素贝叶斯和线性 SVM 两个基线，最后比较它们的本地指标。

### 2.1 数据探索（EDA）

先读表，看清样本量：

```python
import pandas as pd

# 训练集、测试集、以及官方提供的验证集
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

# 训练 3947 条、测试 2235 条
print(len(train), len(test))

# 看前几行：Insult 是 0/1 标签，Date 是时间，Comment 是评论文本
train.head()
```

接着统计缺失值和标签分布：

```python
# 每一列的缺失比例：Date 列有 18.19% 是空的，标签和文本都完整
train.isnull().sum(axis=0) / len(train)

# 数一数两类各有多少条
print(train[train.Insult == 0].count())   # 非侮辱 2898 条
print(train[train.Insult == 1].count())   # 侮辱   1049 条
```

我们得到两个很关键的事实：

1. **`Date` 缺失 18.19%**，而且时间对「这句话是不是侮辱」几乎没有直接帮助，后面可以直接丢掉；
2. **标签不均衡**：3947 条里只有 1049 条是侮辱，占 **26.6%**。这意味着一个「全部猜 0」的傻模型也能拿到 73.4% 的准确率。所以后面看准确率时要多个心眼——**准确率高于 73.4% 才算真的学到了东西**。

### 2.2 数据清洗

文本清洗是这条主线里最费功夫的部分。我们把训练集和测试集的 `Comment` 拼在一起统一清洗，保证两边走完全相同的流程（避免训练时见过的写法在测试时变成另一种样子）：

```python
# 训练集和测试集的文本拼成一个长列表，统一清洗
Data_to_clean = pd.concat([train['Comment'], test['Comment']], axis=0)
```

**第一步：展开英文缩写。** 网络评论里 `don't`、`can't`、`you're` 到处都是。如果原样保留，`don't` 和 `do not` 会被当成两个不同的词，白白丢掉共性。我们准备了一张几百条的缩写对照表（`CONTRACTION_MAP`），用正则把它们替换成完整形式：

```python
import re

def expand_contractions(sentence, contraction_mapping):
    # 把所有缩写拼成一条正则，| 表示「或」；忽略大小写
    pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                         flags=re.IGNORECASE | re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        # 查表得到展开形式；查不到就转小写再查
        expanded = contraction_mapping.get(match) \
                   or contraction_mapping.get(match.lower())
        # 保持原句首字母的大小写
        return match[0] + expanded[1:]
    return pattern.sub(expand_match, sentence)
```

**第二步：只保留字母、数字和空格。** 评论里混着大量链接、HTML 转义、表情符号控制字符（如 `\xa0`）。这些对判断侮辱没有帮助，反而会制造几万个只出现一次的「垃圾词」。所以我们用正则把非字母数字字符统统删掉：

```python
def remove_characters_before_tokenization(sentence):
    sentence = sentence.strip()
    # [^a-zA-Z0-9 ] 的意思是：不是字母、不是数字、也不是空格的字符，全部删掉
    return re.sub(r'[^a-zA-Z0-9 ]', '', sentence)
```

**第三步：分词、去停用词。** 用 NLTK 的 `word_tokenize` 切词，再删掉 `the`、`is`、`a` 这类「停用词」——它们每句话都有，区分不了侮辱与否，只会稀释特征：

```python
import nltk
stopword_list = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    # 过滤掉停用词，其余单词用空格拼回一个字符串
    tokens = [t for t in tokens if t not in stopword_list]
    return ' '.join(tokens)
```

**第四步：纠正重复字母。** 网友骂人时爱拉长音：`sooooo`、`stuupid`。同一个词被拉长成十几种写法，模型会以为是十几个不同的词。我们用一个正则不断把「连续重复的字母」压成一个，并且用 WordNet 词库检查压缩后是不是一个真词，避免把 `good` 压成 `god`：

```python
from nltk.corpus import wordnet

def remove_repeated_characters(text):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')   # 捕获连续重复的字母
    def replace(word):
        if wordnet.synsets(word):          # 已经是合法单词就不动
            return word
        new_word = repeat_pattern.sub(r'\1\2\3', word)
        # 压一次可能还不够（aaa -> a），递归直到不再变化
        return replace(new_word) if new_word != word else new_word
    return ' '.join(replace(w) for w in text.split())
```

四步走完，文本就从「带链接、带表情、大小写混乱的原始评论」变成了「干净的小写单词串」。最后把清洗后的列表按原来的长度切回训练集和测试集两部分。

### 2.3 模型训练与超参数调优

**特征工程：1～3 元词袋 + TF-IDF。** 「n 元」指的是连续 n 个词组成的短语：1 元是单个词（`stupid`），2 元是两词短语（`you stupid`），3 元是三词短语（`you are stupid`）。加入 2、3 元是为了抓住「单词本身不脏、组合起来才脏」的情况。词袋统计完之后，再用 TF-IDF 给每个词加权——**一个词如果在几乎所有评论里都出现，它的权重会被压低；只在少数评论里出现的词权重更高**，这正好符合「侮辱性表达往往是少数派」的直觉：

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# 在训练文本上建词表：统计 1~3 个连续词的组合，min_df=1 表示出现一次也保留
vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 3))
train_count = vectorizer.fit_transform(train_corpus)
# 测试集必须用同一个词表做 transform，不能重新 fit，否则特征对不上
test_count = vectorizer.transform(test_corpus)

# 把词频矩阵转成 TF-IDF 权重矩阵
tfidf = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True)
X_train = tfidf.fit_transform(train_count)
X_test  = tfidf.transform(test_count)
```

这样每一条评论就变成了一个长度约 4.9 万维的稀疏向量（材料里测试矩阵的形状是 `(2235, 49287)`）。

**两个基线模型。** 我们先用多项式朴素贝叶斯（MultinomialNB）——它假设「每个词对类别的贡献独立」，在文本分类上又快又稳，是标准的第一个基线；再用 SGDClassifier，它默认用 hinge 损失，本质是一个**线性 SVM**，对高维稀疏文本通常比朴素贝叶斯更强：

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 从 3947 条训练数据里切出 20% 做验证（stratify 保持正负比例一致）
X_tr, X_va, y_tr, y_va = train_test_split(
    X_train, train['Insult'], test_size=0.2, stratify=train['Insult'], random_state=42)

# 基线一：朴素贝叶斯
NB = MultinomialNB()
NB.fit(X_tr, y_tr)
NB_pred = NB.predict(X_va)
print(accuracy_score(y_va, NB_pred), "Accuracy")

# 基线二：线性 SVM（hinge 损失）
SDG = SGDClassifier(random_state=42)
SDG.fit(X_tr, y_tr)
SDG_pred = SDG.predict(X_va)
print(accuracy_score(y_va, SDG_pred), "Accuracy")
```

**关于调参。** 材料里的做法是先固定 `ngram_range=(1,3)` 和 TF-IDF 默认参数，把两个分类器放在同一套特征上比较。我们没有在 4.9 万维的稀疏矩阵上做大网格搜索——每次搜索都要重建词表，耗时长，而且在这个数据量（不到 4000 条）上很容易过拟合验证集。真正值得动的旋钮是「n-gram 范围」和「`min_df`」，前者决定能抓到多长的短语，后者决定要不要过滤极低频词。

### 2.4 模型测试与指标报告

在验证集上，两个模型的表现如下（材料记录的数字）：

| 模型 | 准确率 | 加权 F1 | 精确率（加权） | 召回率（加权） |
| --- | ---: | ---: | ---: | ---: |
| 多项式朴素贝叶斯 | 0.7295 | 0.6154 | 0.5322 | 0.7295 |
| 线性 SVM（SGD） | **0.7463** | **0.6545** | **0.7983** | 0.7463 |

几个值得说的点：

- **线性 SVM 全面胜出**：准确率 0.7463 比朴素贝叶斯高约 1.7 个百分点，加权 F1 从 0.6154 提到 0.6545，精确率更是从 0.53 跳到 0.80。精确率的大幅提升说明 SGD 更少「冤枉好人」——被判为侮辱的评论里，真正是侮辱的比例高得多。
- **准确率只比「全猜 0」高一点点**：全部猜 0 就有 73.4%，朴素贝叶斯的 72.95% 甚至还更低。这说明在面对不均衡数据时，**单看准确率会严重高估模型**，必须结合 F1、精确率、召回率一起看。
- **加权召回率数字和准确率一模一样**：这不是巧合，而是加权平均的定义决定的——按各类样本数加权后，召回率恰好还原成「总体猜对的比例」。理解了这一点，就不会被表格里两个相同的数字弄糊涂。

还有一点必须坦白：材料里原始代码的验证集切片（`X_training[790:]`）和训练集切片（`X_training[0:3157]`）在第 790～3157 行之间有**重叠**，同一批样本既训练又验证，这属于数据泄漏，会让上面的分数偏乐观。我们在自己的流程里改成了不重叠的 `train_test_split`，因此真实分数应当略低于材料记录值，但「SGD 优于朴素贝叶斯、且精确率差距明显」这个结论方向是稳的。

**生成提交。** 确认模型可用后，用全量训练数据重新拟合，再对测试集预测。因为官方按 AUC 评分，我们要提交的是**属于侮辱类的概率**，而不是 0/1 标签：

```python
# 用全部训练数据重新训练（多喂数据通常更稳）
SDG.fit(X_train, train['Insult'])

# predict_proba 的第二列就是「是侮辱」的概率
test_prob = SDG.predict_proba(X_test)[:, 1]

# 按提交样例的格式写文件：id + Insult 概率
submission = pd.DataFrame({'Insult': test_prob})
submission.to_csv('submission.csv', index=False)
```

---

## 三、总结

本文中我们完成了社交媒体侮辱性言论识别的完整流程。我们首先在 EDA 里确认训练集 3947 条、测试集 2235 条，`Date` 缺失 18.19%，正类只占 26.6%，从而知道「全猜 0」就有 73.4% 的准确率、必须小心评价指标；其次我们做了四步文本清洗：展开英文缩写、删除非字母数字字符、去停用词、用 WordNet 校验纠正拉长音的重复字母；接着我们用 1～3 元词袋加 TF-IDF 构造了 4.9 万维的稀疏特征，并训练朴素贝叶斯和线性 SVM 两个基线；最后在验证集上得到线性 SVM 准确率 0.7463、加权 F1 0.6545、精确率 0.7983，全面优于朴素贝叶斯的 0.7295 / 0.6154 / 0.5322。

这场比赛给我们的最大启发是：**文本清洗的每一步都要能说清「为什么」**——展开缩写是为了合并同义写法，纠重复字母是为了合并拉长音，去停用词是为了减少噪声；而评价不均衡分类任务时，准确率是会骗人的，真正该盯的是精确率和 F1，以及官方指标 AUC 所关心的概率排序质量。