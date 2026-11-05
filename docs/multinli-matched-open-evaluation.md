# MultiNLI 匹配开放域自然语言推理

## 一、比赛简介

### 1.1 比赛背景

MultiNLI（Multi-Genre Natural Language Inference）是自然语言处理里一个很有名的数据集，任务是**自然语言推理**：给两句话，一句叫前提（premise），一句叫假设（hypothesis），判断从前提能不能推出假设。

举一个直观的例子：

- 前提：一个男人在弹吉他。
- 假设：有人在演奏乐器。

这两句的关系是“蕴含”（entailment），因为前提为真时假设一定为真。如果前提是“一个男人在弹吉他”，假设是“一个女人在唱歌”，两者矛盾，就是“矛盾”（contradiction）。如果两句话既不能推出、也不矛盾，比如“一个男人在弹吉他”和“他刚吃过饭”，就是“中立”（neutral）。

MultiNLI 的特点在于“多领域”（multi-genre）：它的句子对来自小说、政府文件、演讲、电话对话、旅行指南等不同文体。同一个模型在小说上表现好，不代表在政府文件上也能行。比赛里的 `matched` 和 `mismatched` 就是用来区分“训练和测试领域相同”与“领域不同”的。

### 1.2 比赛目标

比赛提供 JSON Lines 格式的数据，每行是一个 JSON 对象。材料里读取的是测试集文件 `multinli_0.9_test_matched_unlabeled.jsonl`，主要用到这些字段：

- `genre`：句子来源的文体，例如 fiction、government、slate、telephone、travel；
- `sentence1`：前提句；
- `sentence2`：假设句（材料里的分析主要围绕 `sentence1`）。

我们要为每个句子对预测三个标签之一：蕴含、中立、矛盾。因为这是三分类，模型输出三个概率，取最大者为最终答案。

### 1.3 评价标准

比赛使用**分类准确率（accuracy）**：预测正确的句子对占全部测试对的比例。三个类别大致均衡，所以准确率不会像极不均衡数据那样虚高；随机瞎猜的准确率约为 1/3。准确率越高越好，满分 1.0。

## 二、解题过程

需要先说明：这份材料是一篇**数据分析（EDA）notebook**，只做到读取、统计、去重、长度过滤和分层采样，没有给出模型训练与提交代码。为了把流程写完整，我们在 2.3 和 2.4 里给出一个基于材料数据的基线方案，并明确标注哪些步骤是材料里的、哪些是我们按任务定义补的，不虚构任何分数。

### 2.1 数据探索（EDA）

我们首先用 pandas 直接读 JSON Lines 文件，看字段和几条样本：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# lines=True 表示每一行是一个独立的 JSON 对象
df = pd.read_json('../input/multinli_0.9_test_matched_unlabeled.jsonl', lines=True)
print(df.columns.tolist())
print(df.shape)
print(df.head(5))
```

打印结果能看到 `genre`、`sentence1`、`sentence2` 等字段。测试集没有标签列，所以这份材料只能做输入侧的分析，不能算准确率。

接着我们统计每个文体的句子数量，看看数据是不是均衡：

```python
# 每个 genre 有多少条
print(df['genre'].value_counts())

# 画成柱状图，直观看各文体的占比
df['genre'].value_counts().plot(kind='bar')
plt.title('Samples per genre')
plt.show()
```

MultiNLI 的五个文体大致均衡：fiction、government、slate、telephone、travel 各占约五分之一。这是数据集设计时刻意做的，目的是让模型不能靠“猜文体”来走捷径。

然后我们看句子长度。长度是最容易出问题的地方：太短的句子缺少上下文，太长的句子又会让模型难以处理。

```python
# 字符长度
df['length'] = df['sentence1'].apply(len)
print(df['length'].describe())

# 直方图看分布，bins=20 是常用的粗粒度
df['length'].plot.hist(alpha=0.5, bins=20)
plt.xlabel('characters')
plt.show()
```

从直方图能看到明显的长尾：大部分句子在几十到一百多个字符，但有少量句子特别长。材料还按文体分别画了长度分布，发现不同文体的长度差异很大，政府文件和小说的句子通常更长。

### 2.2 数据清洗

第一项清洗是**去重**。材料在按长度排序后检查头尾样本，发现数据里有重复的 `sentence1`：

```python
# keep=False 表示把所有重复项都标出来，而不是只留第一个
duplicates = df[df.duplicated(['sentence1'], keep=False)]
print('重复句子数:', duplicates['sentence1'].nunique())

# 按 sentence1 去重，保留第一次出现的行
data_except_d = df.drop_duplicates(['sentence1'])
print('去重前:', len(df), '去重后:', len(data_except_d))
```

去重的原因很实际：MultiNLI 的句子对由模板和人工改写生成，同一句前提可能配了多个假设。如果不去重，训练集和验证集里可能出现“同一句前提同时出现在两边”的情况，造成信息泄漏，验证分数会虚高。

第二项清洗是**按词数过滤**。材料用 spaCy 分词，统计每句的词数，只保留 3～25 个词的句子：

```python
import spacy
nlp = spacy.load('en_core_web_sm', parser=False, entity=False)

# len(nlp(x)) 得到分词后的 token 数
data_except_d['word_count'] = data_except_d['sentence1'].apply(lambda x: len(nlp(x)))
print(data_except_d['word_count'].describe())

# 至少 3 个词才能构成“主谓宾”的完整语义；超过 25 个词则太长
data_filtered = data_except_d[
    (data_except_d['word_count'] >= 3) &
    (data_except_d['word_count'] <= 25)
]
print('过滤前:', len(data_except_d), '过滤后:', len(data_filtered))
```

3 个词的下限来自“主语 + 谓语 + 宾语”最少需要三个词；25 个词的上限参考了 GOV.UK 对清晰句子的建议。注意 spaCy 会把句号也算成一个 token，所以实际词数会比直觉多一个，但影响不大。

第三项清洗是**按文体均匀采样**。过滤后各文体的数量不再相等，如果直接拿去做分析，长文体会 dominate。材料对每个文体随机抽 379 条，构造成完全均衡的数据集：

```python
# groupby('genre') 后每个组抽 379 条，保证五个文体数量一致
data_selected = (data_filtered
                 .groupby('genre')
                 .apply(lambda x: x.sample(n=379))
                 .drop(columns=['genre', 'index'])
                 .reset_index())

print(data_selected['genre'].value_counts())
```

379 是过滤后最小文体的样本数，取这个数能保证每个文体都抽得到又不浪费数据。最后得到的均衡数据集大约 1895 条（379 × 5），适合做后续的统计分析和基线实验。

### 2.3 模型训练与超参数调优

材料没有模型代码。下面是我们按 NLI 任务的标准做法补出的基线方案，用伪代码和真实可运行的 sklearn 代码表示；所有步骤都基于前面清洗出的 `data_selected`，不涉及材料里不存在的实验结果。

第一步是把前提和假设拼成一个字符串，让 TF-IDF 能同时看到两句话：

```python
# 前提和假设用分隔符拼起来，避免两句话的边界糊在一起
data_selected['pair'] = (
    data_selected['sentence1'] + ' [SEP] ' + data_selected['sentence2']
)

# 词级 1~2 元组 + 字符级 2~5 元组，兼顾词汇和拼写变化
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                           min_df=2, max_features=50000)
features = FeatureUnion([('word', word_vec), ('char', char_vec)])
```

第二步是用逻辑回归做三分类。逻辑回归对高维稀疏文本特征很稳，训练快，而且输出的是校准过的概率，方便看模型对三个类别的信心：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 分层切分，保证训练集和验证集里三个类别的比例一致
X_train, X_val, y_train, y_val = train_test_split(
    data_selected['pair'], data_selected['label'],
    test_size=0.2, stratify=data_selected['label'], random_state=42
)

clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
clf.fit(features.fit_transform(X_train), y_train)
```

如果要做更强的基线，可以再加一组“结构特征”：两句话的词重叠率、长度差、否定词数量、数字是否一致。这些特征不需要预训练模型，却能抓住 NLI 里最常见的几类推理模式：

```python
def overlap_features(premise, hypothesis):
    p = set(premise.lower().split())
    h = set(hypothesis.lower().split())
    # Jaccard 重叠率：两句话共享的词占所有词的比例
    jaccard = len(p & h) / max(len(p | h), 1)
    # 长度差：蕴含句通常不会比前提长太多
    len_diff = abs(len(p) - len(h))
    # 否定词：NLI 里 "not" 经常是矛盾信号
    neg = sum(w in {'not', 'no', 'never', 'none'} for w in p | h)
    return [jaccard, len_diff, neg]
```

超参数方面，`C` 控制正则强度：C 越大越容易过拟合，C 越小越保守。我们会在验证集上试 `C` 取 0.1、1、10 三档，选准确率最高的一档；`min_df` 和 `max_features` 则根据内存和训练时间调整。如果资源允许，可以把逻辑回归换成 BERT 微调，但在没有 GPU 的情况下，TF-IDF + 逻辑回归是一个合理起点。

### 2.4 模型测试与指标报告

因为没有官方标签，材料无法给出准确率。我们在本地只能用“留出的验证集”来估计模型好坏。评估时除了总体准确率，还要按 `genre` 拆开看：

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pred = clf.predict(features.transform(X_val))
print('Overall accuracy:', accuracy_score(y_val, pred))
print(classification_report(y_val, pred))

# 按文体拆开，检查模型是否在某个领域明显掉分
for g in data_selected['genre'].unique():
    mask = data_selected.loc[X_val.index, 'genre'] == g
    if mask.sum() > 0:
        print(g, accuracy_score(y_val[mask], pred[mask]))
```

`matched` 和 `mismatched` 的意义就在这里：模型在训练领域内准确率高，不代表跨领域也高。如果 fiction 上 80%、government 上只有 60%，说明模型学到了文体的表面特征，而不是真正的推理规律。

提交时，把测试集的 `pair` 走同样的 `features.transform`，输出三个类别的概率，再按比赛要求写 CSV。写完后要检查行数、列名、概率和是否为 1，以及有没有 NaN：

```python
proba = clf.predict_proba(features.transform(test_df['pair']))
submission = pd.DataFrame(proba, columns=['entailment', 'neutral', 'contradiction'])
submission.insert(0, 'id', test_df['id'].values)
print(submission.isnull().sum().sum())
print(submission.iloc[:, 1:].sum(axis=1).head())
submission.to_csv('submission.csv', index=False)
```

再次强调：材料本身没有训练和提交记录，上面的代码是按任务定义补的基线，不是材料作者实际跑出的结果。我们不在 writeup 里编造任何准确率数字。

## 三、总结

本文中我们完成了 MultiNLI 匹配开放域自然语言推理的数据分析，并给出了一个可复现的基线方案。我们首先用 EDA 看清了数据：五个文体大致均衡，句子长度呈长尾分布，测试集没有标签。其次我们做了三项清洗：按 `sentence1` 去重、用 spaCy 统计词数并保留 3～25 个词的句子、按文体各抽 379 条构造均衡数据集。接着我们按 NLI 任务的标准做法，把前提和假设拼成句子对，用词级和字符级 TF-IDF 提取特征，再用逻辑回归做三分类，并补充了词重叠率、长度差、否定词等结构特征。最后我们说明了如何用留出验证集和按文体拆分的准确率来评估模型，以及如何生成提交文件；由于材料没有模型结果，我们没有报告任何虚构分数。

这个比赛让我们体会到，NLI 的难点不在“句子分类”，而在“关系判断”：两句话单独看都很普通，只有把它们放在一起比较，才能知道是蕴含、中立还是矛盾。也正因为如此，数据清洗里的去重和领域拆分特别重要——如果同一句前提同时出现在训练和验证集里，模型看起来会很聪明，实际上只是记住了答案。