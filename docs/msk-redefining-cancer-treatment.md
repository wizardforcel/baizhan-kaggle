# MSK 重新定义癌症治疗（Personalized Medicine）

## 一、比赛简介

### 1.1 比赛背景

Memorial Sloan Kettering（MSK）癌症中心办过一场叫“Personalized Medicine: Redefining Cancer Treatment”的比赛。它的出发点很实际：医生拿到一份肿瘤基因检测报告，上面写着某个基因发生了某种变异，但报告不会直接告诉你“这个变异会不会致癌、该不该用靶向药”。这件事需要专家查阅大量文献才能判断。

比赛把这个问题变成了一个文本分类任务。每条样本包含三样东西：基因名（Gene）、变异写法（Variation）、以及一段从医学文献里摘出来的文字说明（Text）。参赛者要根据这些信息，把变异分到 9 个“致病等级”里。等级越高，说明这个变异与癌症的关系越明确、越值得临床关注。

难点在于：文本非常长、专业术语密集；基因和变异是类别型字段，取值很多；9 个类别严重不均衡；而且同一个基因的不同变异，文本可能几乎一样，只有变异写法不同。模型必须同时看懂“基因是什么”“变异长什么样”“文献说了什么”。

### 1.2 比赛目标

数据分成两张表：

- `training_variants.csv`：`ID`、`Gene`、`Variation`、`Class`。其中 `Class` 是 1～9 的标签，`ID` 是样本编号。
- `training_text.csv`：用 `||` 分隔的两列，`ID` 和 `Text`。`Text` 是医学文献摘要。

两张表通过 `ID` 关联。测试集同样给出 `ID`、`Gene`、`Variation` 和 `Text`，我们要为每条测试样本输出 9 个概率，分别对应 9 个类别。提交文件的列名是 `class1` 到 `class9`，每一行概率之和要等于 1。

### 1.3 评价标准

比赛使用**多分类对数损失（multi-class log loss）**，越低越好。它的计算方式是：对每条样本，取出模型给“真实类别”预测的概率，取对数再取负号，最后对所有样本求平均。

这个指标的关键在于“信心”也受惩罚：如果模型给真实类别 0.9 的概率，损失很小；如果给 0.1，损失很大；如果给 0.99 却猜错了，损失会非常大。所以模型不能只会猜对，还要知道自己有多确定。作为参照，9 个类别完全随机猜的 log loss 约为 2.20（`ln(9)`），材料里随机模型的实测值是 2.47～2.50。

## 二、解题过程

我们的主线是：先把两张表按 `ID` 合并，其次对长文本做小写化、去特殊字符、去停用词，接着用 TF-IDF 把文本变成稀疏向量，然后比较逻辑回归、朴素贝叶斯、SVM、AdaBoost 和 XGBoost，最后选验证集上 log loss 最低的 XGBoost 生成提交文件。

### 2.1 数据探索（EDA）

我们首先读入变异表和文本表，确认行数、列名和缺失情况：

```python
import pandas as pd

# 变异表：每行是一个基因变异，Class 是 1~9 的致病等级
variants = pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/training_variants.zip')
print(variants.shape)        # (3321, 4)
print(variants.head())

# 文本表用 || 分隔，不是普通逗号，所以要指定 sep 和 engine
text = pd.read_csv(
    '/kaggle/input/msk-redefining-cancer-treatment/training_text.zip',
    sep=r'\|\|', engine='python',
    names=['ID', 'TEXT'], skiprows=1
)
print(text.shape)
print(text.head())
```

变异表有 3321 行、4 列，没有缺失值。基因一共 264 种，变异写法一共 2996 种——这两个数字说明“基因”和“变异”都是典型的高基数类别特征，直接做 one-hot 会得到很稀疏的矩阵。

文本表里发现 **5 条文本缺失**，对应的 `ID` 是 1109、1277、1407、1639、2755。这 5 条样本的基因和变异还在，只是没有文献说明。后面我们会保留它们，用空字符串代替缺失文本，避免整条样本被丢掉。

接着看标签分布：

```python
# 9 个类别的样本数
print(variants['Class'].value_counts().sort_index())

# 基因出现频率：看有多少基因是“常客”
gene_counts = variants['Gene'].value_counts()
print(gene_counts.head(10))
print('基因总数:', variants['Gene'].nunique())
print('变异总数:', variants['Variation'].nunique())
```

统计结果有两个重要发现。第一，9 个类别明显不均衡，7 类样本最多，4 类最少，比例相差好几倍；如果模型只会猜多数类，准确率看起来不低，但 log loss 会很难看。第二，基因分布是长尾的：少数几个基因（比如 BRCA1、BRCA2、TP53）出现几百次，大量基因只出现一两次。稀有基因在训练集里学不到规律，这是后面必须面对的问题。

### 2.2 数据清洗

我们做的第一件事是把两张表合并：

```python
# 以变异表为主表左连接文本表，保证 3321 条变异一条不少
df = pd.merge(variants, text, on='ID', how='left')
print(df.shape)
print(df['TEXT'].isnull().sum())   # 5 条缺失
```

用左连接而不是内连接，是为了保留那 5 条没有文本的样本。它们的文本列是 NaN，后面统一填成空字符串。

第二件事是文本预处理。医学文本里有很多标点、换行、大小写混用，还有大量“the、a、is”这类停用词。我们的处理函数做四步：把非字母数字字符替换成空格、把连续空格压成一个、统一转小写、去掉英文停用词：

```python
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(s):
    if not isinstance(s, str):      # NaN 或数字直接当空文本
        return ''
    s = re.sub(r'[^a-zA-Z0-9\n]', ' ', s)  # 标点、括号等替换成空格
    s = re.sub(r'\s+', ' ', s)             # 多个空格压成一个
    s = s.lower()                          # 统一小写，避免 Gene/gene 被当成两个词
    words = [w for w in s.split() if w not in stop_words]
    return ' '.join(words)

df['TEXT'] = df['TEXT'].apply(clean_text)
df['TEXT'] = df['TEXT'].fillna('')
print(df.loc[1109, 'TEXT'][:100])
```

这里有一个取舍：基因名和变异写法本身包含数字、星号、下划线（例如 `W802*`、`Truncating Mutations`），如果清洗太狠会把它们破坏掉。所以我们的正则只作用于 `TEXT` 列，`Gene` 和 `Variation` 两列保持原样，留给后面的特征工程使用。

第三件事是检查重复。文本表里有些不同 `ID` 的样本文本几乎一样，这是同一篇文献被多个变异引用导致的，属于正常现象，不能按文本去重，否则会误删不同变异的样本。

### 2.3 模型训练与超参数调优

文本特征我们用 TF-IDF。它的思想是：一个词在本文出现得越多、在整个语料里出现得越少，就越能代表这篇文章。参数上，我们限制词频至少出现 5 次（`min_df=5`），最多保留 16000 个特征，词和词组都统计（1～3 元），并用 `sublinear_tf` 把词频取对数，避免某个词重复太多次 dominate：

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    min_df=5,                 # 出现少于 5 次的词直接丢掉，降噪
    max_features=16000,       # 最多保留 16000 个词/词组
    strip_accents='unicode',  # 去掉重音符号
    lowercase=True,
    analyzer='word',
    token_pattern=r'\w+',
    ngram_range=(1, 3),       # 单词、二元组、三元组都统计
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True,        # 词频取 1+log(tf)，压制超高频词
    stop_words='english'
)

tfidf.fit(df['TEXT'])
X = tfidf.transform(df['TEXT'])
y = df['Class'].values
print(X.shape)   # (3321, 16000)
```

我们只对 `TEXT` 做 TF-IDF，没有把 `Gene` 和 `Variation` 拼进去。原因是这两个字段基数太高，直接做词袋会产生大量只出现一两次的列，反而增加噪声。材料里另外用 `CountVectorizer` 对 `Gene` 做了 one-hot 的单特征实验，验证了“只用基因”也能学到一些信号，但单独使用效果有限，所以最终主线还是文本。

模型比较用 5 折分层交叉验证。分层（`StratifiedKFold`）保证每一折里 9 个类别的比例和原数据一致，避免某一折缺掉稀有类别。评估函数直接算 log loss 和准确率：

```python
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
import numpy as np

def evaluate(clf):
    # cross_val_predict 会为每条样本返回“没在训练中见过它”的预测概率
    probas = cross_val_predict(
        clf, X, y,
        cv=StratifiedKFold(n_splits=5, random_state=8),
        n_jobs=-1, method='predict_proba'
    )
    pred = np.argmax(probas, axis=1) + 1   # 类别从 1 开始，要加回来
    print('Log loss:', log_loss(y, probas))
    print('Accuracy:', accuracy_score(y, pred))
```

我们比较了五种模型：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

evaluate(LogisticRegression())
evaluate(MultinomialNB())
evaluate(SVC(probability=True))
evaluate(AdaBoostClassifier())
evaluate(XGBClassifier())
```

交叉验证结果如下（数字来自材料）：

| 模型 | Log loss | 准确率 |
| --- | --- | --- |
| 逻辑回归 | 1.4895 | 0.4833 |
| XGBoost | **1.4821** | **0.5002** |
| AdaBoost | 2.0670 | 0.3683 |
| 多项式朴素贝叶斯 | 2.7326 | 0.4556 |
| SVM | 1.6530 | 0.3963 |
| 随机猜 | 2.4977 | — |

XGBoost 的 log loss 最低，逻辑回归紧随其后；朴素贝叶斯虽然准确率不算最差，但 log loss 很高，说明它经常“很自信地猜错”。这就是 log loss 和准确率的区别：准确率只看对错，log loss 还看信心是否校准。

超参数方面，我们主要调了 TF-IDF 的 `min_df`、`max_features` 和 `ngram_range`。`min_df` 太小会引入大量只出现一次的词，`max_features` 太大则训练变慢、稀疏矩阵占内存。经过几组小规模试验，`min_df=5`、`max_features=16000`、`(1,3)` 元组在验证集上比较平衡，再细调带来的提升很小，所以最终固定这组参数。XGBoost 本身用默认配置（100 棵树、最大深度 3、学习率 0.1），没有做大规模网格搜索。

### 2.4 模型测试与指标报告

选定 XGBoost 后，我们用全部训练数据重新拟合，再对测试集预测：

```python
clf = XGBClassifier()
clf.fit(X, y)

# 测试集也要走同样的清洗和 TF-IDF 变换
test_df['TEXT'] = test_df['TEXT'].apply(clean_text).fillna('')
X_test = tfidf.transform(test_df['TEXT'])
proba = clf.predict_proba(X_test)
print(proba.shape)   # (测试样本数, 9)
```

在交叉验证里，XGBoost 的 log loss 是 **1.4821**，准确率 **0.5002**；作为对照，随机模型的 log loss 是 2.4977。也就是说，模型确实从文本里学到了东西，但离“临床可用”还很远。原因不难理解：9 个类别高度不均衡，稀有类别的文本证据少；而且很多变异只靠文本很难区分，必须结合基因位置、蛋白质结构等外部知识。

最后生成提交文件。比赛要求 9 列概率，列名固定为 `class1` 到 `class9`，顺序不能错：

```python
submission = pd.DataFrame(
    proba,
    columns=['class' + str(i) for i in range(1, 10)]
)
submission.insert(0, 'ID', test_df['ID'].values)

# 检查每行概率和是否为 1，以及有没有缺失
print(submission.iloc[:, 1:].sum(axis=1).head())
print(submission.isnull().sum().sum())

submission.to_csv('submission.csv', index=False)
```

我们还会检查测试集 `ID` 的顺序是否和提交文件一致。材料里测试集是单独读进来的，如果中间做过排序或过滤，`ID` 很容易错位；这种错误在排行榜上表现为“模型明明不差，分数却像随机”，排查起来很浪费时间。

## 三、总结

本文中我们完成了 MSK 重新定义癌症治疗比赛的完整流程。我们首先用 EDA 看清了数据：3321 条变异、264 种基因、2996 种变异写法、9 个不均衡的致病等级，文本表还有 5 条缺失。其次我们把两张表按 `ID` 左连接，对文本做了去特殊字符、压空格、转小写、去停用词四步清洗，同时保留 `Gene` 和 `Variation` 的原始写法。接着我们用 TF-IDF（`min_df=5`、16000 维、1～3 元组）把文本变成稀疏向量，用 5 折分层交叉验证比较了逻辑回归、朴素贝叶斯、SVM、AdaBoost 和 XGBoost。最后 XGBoost 取得最低的 log loss 1.4821 和 0.5002 的准确率，我们把 9 类概率按比赛格式写成了提交文件。

这个比赛给我们最大的启发是：**评价指标会改变模型选择**。如果只看准确率，朴素贝叶斯似乎还能接受；但一看 log loss，它“自信地猜错”的问题就暴露了。另一个体会是，医学文本分类的上限往往不在模型，而在特征——基因位置、变异类型、蛋白质结构这些信息，单靠一段文献摘要是补不回来的。
