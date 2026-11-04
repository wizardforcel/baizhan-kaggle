# 孟加拉语代码混合情感分析（Banglish / Benglish）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的任务是判断一段社交媒体文本的**情感倾向**。它的特别之处在于文本是"代码混合"（code-mixed）的——同一句话里会混用三种写法：

- **孟加拉文**：用孟加拉字母书写，比如 `ভালো`；
- **Banglish**：用英文字母拼写孟加拉语发音，比如把"好"写成 `bhalo`、`valo` 或 `b'alo`；
- **英文**：正常的英语单词。

这种混写在孟加拉语地区的社交平台上非常普遍：同一个意思可能前一个人用孟加拉字母写，后一个人用罗马字母拼，第三个人直接夹英文。对人类来说读起来不难，对模型却是麻烦——同一个词有好几种拼法，按空格切词的"词袋"方法会因为拼法不同而把它们当成完全不同的词。

情感分三类：负面、中性、正面。数据里存在明显的类别不均衡，这也是比赛选择宏平均 F1 而不是准确率的原因。

### 1.2 比赛目标

输入一行文本，包含 `id` 和 `text` 两列；输出这一行文本的情感类别。`label` 用整数编码：

| label | 含义 |
| --- | --- |
| 0 | 负面（negative） |
| 1 | 中性（neutral） |
| 2 | 正面（positive） |

提交文件两列：`id` 和 `label`，`label` 必须是 0、1、2 这样的整数，不能是字符串。

### 1.3 评价标准

比赛使用 **Macro F1**（宏平均 F1）。F1 是精确率和召回率的调和平均，衡量"预测为某类的里面有多少是真的"和"真的某类里面有多少被找出来"两件事的综合。宏平均的意思是：先对三个类别各算一个 F1，再取算术平均，**每个类别权重相同**。

为什么不用准确率？因为如果 70% 的样本是正面，模型只要全猜正面就有 70% 的准确率，但它对负面和中性完全无能为力。宏平均会把这个模型的分数拉得很低，逼着模型三类都要照顾好。F1 越高越好，满分是 1.0。

---

## 二、解题过程

我们的方案是一个刻意保持简单的强基线：**词级 TF-IDF + 字符级 TF-IDF 拼接，喂给带类别权重的逻辑回归**。没有预训练模型，没有深度网络，但每一步都做对，后面再往上加东西时就有了可靠的对照。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，看训练集、测试集的形状，以及标签分布和文本长度。这两件事对代码混合数据特别重要：类别不均衡决定了要不要加权，长度差异决定了截断策略。

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base = '/kaggle/input/competitions/bangla-code-mixed-sentiment-analysis-banglish-benglish'
train = pd.read_csv(f'{base}/train.csv')
test  = pd.read_csv(f'{base}/test.csv')
sample_sub = pd.read_csv(f'{base}/sample_submission.csv')

print('train:', train.shape, '| test:', test.shape)
print(train.head())

# 看类别比例：不均衡有多严重，直接决定要不要 class_weight
print(train['label'].value_counts())
print(train['label'].value_counts(normalize=True).round(3))

train['text_len'] = train['text'].str.len()
print(train['text_len'].describe())

# 按类别各抽几条真实样本，亲眼看看混写长什么样
label_names = {0: 'negative', 1: 'neutral', 2: 'positive'}
for label in sorted(train['label'].unique()):
    print(f'--- {label_names[label]} ---')
    print(train[train['label'] == label]['text'].sample(3, random_state=1).to_list())
```

打印结果确认了两点。第一，三个类别的数量并不相等，多数类和少数类差距明显，所以后面必须用 `class_weight='balanced'` 让少数类的样本在损失里获得更大权重。第二，文本长度分布很宽，既有几个词的短句，也有几十个词的长句，但整体不算特别长，不需要做复杂截断。

我们还画了"文本长度按类别"的直方图，用来检查有没有"某个类别的文本系统性地更长"这种偏差。如果有，模型可能会偷懒用长度当特征，而不是真的读懂了情感。

### 2.2 数据清洗

我们其次做预处理。这里最重要的判断是**不要照搬英文文本的标准流程**。英文情感分析常用的词干提取（stemming）、英文停用词表，在代码混合数据上不但没用，还可能帮倒忙：`bhalo` 不是英文单词，词干提取器不认识它；而 `na`、`not` 这类否定词如果被当停用词删掉，情感极性会直接反转。

所以我们只做三件最保守的事：

1. 转小写——只影响英文和 Banglish 部分，孟加拉字母没有大小写，不受影响；
2. 把连续空白压成一个空格；
3. **保留表情符号和标点**。😍 和 😡 本身就是极强的情感信号，感叹号的数量也常常和情绪强度相关，删掉等于扔信息。

```python
import re

def basic_clean(text):
    text = str(text)
    text = text.lower()                      # 只影响拉丁字母，孟加拉文无大小写
    text = re.sub(r'\s+', ' ', text).strip() # 压缩空白，不删标点
    return text

train['clean_text'] = train['text'].apply(basic_clean)
test['clean_text']  = test['text'].apply(basic_clean)

print(train[['text', 'clean_text']].head())
```

我们特意**没有**做这些事：没有删表情、没有删标点、没有去停用词、没有做词干提取、没有把 Banglish 强行翻译成孟加拉文。原因很简单——在没有可靠工具的前提下，任何"加工"都可能把信号删掉，而保留原文至少不会更差。

### 2.3 模型训练与超参数调优

我们接着切分验证集。测试集没有标签，所以必须从训练集里留出一部分来本地评估。这里用**分层抽样**，保证切出来的验证集里三个类别的比例和原数据一致；否则如果验证集恰好抽到很多正面样本，分数就会虚高。

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train['clean_text'], train['label'],
    test_size=0.2, random_state=42, stratify=train['label'])
print('train:', X_train.shape, '| val:', X_val.shape)
```

特征部分我们用了两套 TF-IDF，然后把它们横向拼起来。TF-IDF 是两个统计量的乘积：词频（这个词在这条文本里出现多少次）和逆文档频率（这个词在整个语料里有多稀有）。一个词越常出现在某条文本里、又越少出现在其他文本里，它的权重就越高。

- **词级**：按空格切词，统计 1 个词和连续 2 个词的组合，最多保留 5000 个特征。它能抓住 `good`、`bhalo` 这样的完整词。
- **字符级**：按字符切，统计 2～4 个字符的连续片段，最多保留 8000 个特征。这是应对 Banglish 拼写变化的关键——`bhalo`、`valo`、`b'alo` 作为整词是完全不同的三个词，但它们共享 `al`、`lo`、`bha` 这样的字符片段，字符级特征能把它们"拉近"。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

word_vectorizer = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 2), min_df=2, max_features=5000)

char_vectorizer = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(2, 4), min_df=2, max_features=8000)

# 只在训练划分上 fit，避免验证集的词表信息泄漏进来
word_vectorizer.fit(X_train)
char_vectorizer.fit(X_train)

def transform_features(texts):
    return hstack([word_vectorizer.transform(texts),
                   char_vectorizer.transform(texts)])

X_train_feats = transform_features(X_train)
X_val_feats   = transform_features(X_val)
print('特征矩阵:', X_train_feats.shape)
```

`min_df=2` 的意思是"至少在两条文本里出现过的片段才保留"，用来过滤只出现一次、多半是拼写噪声的片段。`char_wb` 是"按词边界切字符"，它不会把两个相邻单词的字符拼成一个片段，比纯字符级更干净一些。

分类器用逻辑回归。它虽然简单，但在高维稀疏的文本特征上非常稳，训练快、不容易过拟合，而且系数可以直接解释成"这个词偏向哪个类别"。我们设 `class_weight='balanced'`，让模型按类别频率自动给少数类更高权重；`C=1.0` 是正则强度的倒数，值越小正则越强，1.0 是一个中庸的起点。

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',   # 少数类样本在损失里权重更大
    C=1.0,
    random_state=42)
clf.fit(X_train_feats, y_train)
```

### 2.4 模型测试与指标报告

我们然后在验证集上算宏平均 F1，并打印每个类别的精确率、召回率和 F1，再看混淆矩阵——分数只告诉我们"错了多少"，混淆矩阵才告诉我们"错到哪去了"。

```python
from sklearn.metrics import f1_score, classification_report, confusion_matrix

val_preds = clf.predict(X_val_feats)
macro_f1 = f1_score(y_val, val_preds, average='macro')
print(f'Validation Macro F1: {macro_f1:.4f}')

print(classification_report(y_val, val_preds,
                            target_names=['negative', 'neutral', 'positive']))

cm = confusion_matrix(y_val, val_preds, labels=[0, 1, 2])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['neg', 'neu', 'pos'],
            yticklabels=['neg', 'neu', 'pos'])
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title('Confusion Matrix (Validation)')
plt.show()
```

读这份报告时我们重点关注两件事。第一，中性类的 F1 通常最低——中性文本本身就没有强烈的情感词，模型缺少抓手；如果中性被大量误判成正面或负面，说明特征里"情感词"的权重压过了"语气平淡"的线索。第二，看负面和正面之间的互相误判：如果两边对称地错，那是模型能力问题；如果严重偏向一边，那多半是类别不均衡导致的阈值偏移。正因为先看了这些，我们才确定"加权 + 宏平均"的组合是必要的，而不是可选项。

确认验证分数合理之后，我们把验证集也并回训练集，用**全量数据**重新拟合一次向量器和分类器。多喂 20% 的数据通常只会让模型更稳，不会更差；但要注意，向量器也必须重新 fit，不能沿用之前只在 80% 数据上学的词表。

```python
# 用全量训练数据重新拟合：词表和模型都要重来
word_vectorizer_full = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 2), min_df=2, max_features=5000)
char_vectorizer_full = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(2, 4), min_df=2, max_features=8000)

word_vectorizer_full.fit(train['clean_text'])
char_vectorizer_full.fit(train['clean_text'])

def transform_full(texts):
    return hstack([word_vectorizer_full.transform(texts),
                   char_vectorizer_full.transform(texts)])

final_clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                               C=1.0, random_state=42)
final_clf.fit(transform_full(train['clean_text']), train['label'])
```

最后对测试集预测并生成提交文件。生成之后我们还会看一眼预测出来的类别分布：如果测试集上 95% 都被判成正面，而训练集只有一半左右是正面，那多半是哪里出了问题（比如向量器忘了重新 fit，或者标签顺序搞反了）。

```python
test_preds = final_clf.predict(transform_full(test['clean_text']))

submission = pd.DataFrame({'id': test['id'], 'label': test_preds})
submission.to_csv('baseline_submission.csv', index=False)

print(submission['label'].value_counts(normalize=True).round(3))
print(submission.head())
```

材料里没有给出这个基线在排行榜上的具体分数，所以我们也不编造数字。可以确定的是：它刻意保持简单，目的是给后续改进提供一个**可信的下限**。材料末尾列出的方向也很具体：把 Banglish 的拼写变体归一化、换成 XLM-RoBERTa / mBERT / BanglaBERT 这类多语言预训练模型、加否定词感知特征、处理"先夸后转折"的讽刺句式、把 TF-IDF 模型和 Transformer 做集成，以及用分层 K 折替代单次留出。

---

## 三、总结

本文中我们为孟加拉语代码混合情感分析搭了一个简单但完整的基线。我们首先用 EDA 看清了三类标签的分布不均衡、文本长度跨度大，并亲眼确认了孟加拉文、Banglish、英文混写的样子；其次只做了转小写和压缩空白这两步最保守的清洗，刻意保留表情、标点和否定词；接着用分层抽样切出 20% 验证集，把词级 1～2 gram 和字符级 2～4 gram 的 TF-IDF 拼在一起，喂给带类别权重的逻辑回归；最后在验证集上用宏平均 F1 和混淆矩阵检查每一类的表现，再用全量数据重新拟合、预测测试集并核对提交格式。

这个比赛让我们体会最深的是**预处理要跟着数据走**。同一套英文情感分析的标准流程，搬到代码混合文本上会把最有用的信号删掉；而字符级 n-gram 这种看似"笨"的特征，恰恰能化解 Banglish 拼写不统一这个最棘手的问题。先有一个诚实、可解释的基线，再去换预训练模型或者做集成，改进才有参照系。
