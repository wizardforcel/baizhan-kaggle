# 第19章 文本数据比赛详解：文本分类与关键词抽取（Tradeshift / Facebook Recruiting III）

Tradeshift 是一家做企业间电子单据流转的公司。日常生活中，一张发票、一份采购单或者一份合同，往往先被扫描成图片，再由 OCR（光学字符识别）把图片里的文字识别出来。识别出来的文字并不是一整段，而是被切成很多个「文本块」，每个文本块外面画一个矩形框，记录它在页面上的位置。这个比赛要解决的问题就是：**给一个文本块，判断它属于哪一类内容**，比如它是不是发票号码、日期、金额、公司名称等等。

与 Quora 问题对去重（文本相似度、LogLoss）和 IMDb 情感分析（单文本多分类、准确率）不同，这个比赛是**多标签文本分类**：一个文本块可以同时属于多个标签，33 个标签要分别预测，每个标签输出一个 0 到 1 之间的概率。

比赛数据来自成千上万份真实单据，官方说明里提到有「数百万个单词」。每份文档会被切成若干个文本块，所有文档的文本块汇总到一起，一行就是一个文本块。因为单据的格式五花八门，语言也不统一，再加上 OCR 本身会认错字，所以数据里天然带着噪声。更麻烦的是，主办方把特征名和标签名都隐藏了，只留下 `x1` 到 `x145` 这样的编号，以及 `y1` 到 `y33` 这样的标签编号，我们不知道每一列具体代表什么，只能靠数据本身的统计规律去建模。

这个比赛还有一个特点：数据量很大。官方给出的样本总数约 210 万，其中训练集约 170 万行，测试集约 54.5 万行。特征有 145 个，标签有 33 个。对入门者来说，这是一次很好的练习——它逼着我们面对真实场景里的三个问题：数据大、特征含义不明、缺失值多。

本章我们完整走一遍 Tradeshift 比赛的解题流程：EDA → 清洗（缺失值处理、特征类型区分） → 特征工程（哈希特征作为类别、缺失标记） → 建模（在线线性模型，每个标签单独训练） → 测试与提交（id_label,pred 格式）。

顺便提一下，Facebook Recruiting III 关键词抽取是另一个文本比赛的代表：给定一段职位描述，抽取关键词，也是文本处理任务，但侧重于从非结构化文本中抽取特定信息。Tradeshift 和 Facebook Recruiting III 共同的点是都要处理噪声文本和稀有特征，区别在于 Tradeshift 是多标签分类、Facebook Recruiting III 是关键词抽取/序列标注类任务。

---

## 19.1 比赛背景与数据

### 比赛背景

Tradeshift 比赛要解决的核心问题是：给一个文本块，判断它属于哪一类内容，比如它是不是发票号码、日期、金额、公司名称等等。

### 数据结构

数据文件有四个：

| 文件 | 含义 |
|------|------|
| `train.csv` | 训练集特征，每行一个文本块，列是 `id` 加上 `x1` 到 `x145` |
| `trainLabels.csv` | 训练集标签，每行对应一个训练样本，列是 `id` 加上 `y1` 到 `y33`，取值是 0 或 1 |
| `test.csv` | 测试集特征，列和 `train.csv` 一样，但没有标签 |
| `sampleSubmission.csv` | 提交格式示例，只有两列，`id_label` 和 `pred` |

训练表是 **1,700,000 行、178 列**。178 列正好是 145 个特征加 33 个标签。

提交时要把每个测试样本的每个标签拆成一行，例如编号为 `1700001` 的样本、标签 `y1`，就写成 `1700001_y1`，后面跟上预测概率。33 个标签就要写 33 行，所以提交文件的行数是测试样本数乘以 33。

### 特征类型

这些特征大致可以分成四类：

1. **内容特征**：原始文本的加密哈希值，看起来像一串 Base64 字符串；
2. **解析特征**：这个文本块能不能被解析成数字、字母、字母数字混合等；
3. **空间特征**：文本框的位置、大小、宽高比等；
4. **关系特征**：和同一份文档里上下左右相邻文本块的关系。如果一个文本块在页面最上面，它上面没有别的文本块，这类特征就会空着。

这也解释了后面我们为什么会看到大量缺失值：不是数据坏了，而是「没有相邻文本块」这件事本身就需要用缺失来表示。

### 评价指标

比赛使用**多标签 LogLoss（对数损失）**作为评价指标，越小越好。它的意思是：对每一个测试样本的每一个标签，都拿预测概率和真实标签算一次交叉熵，然后把所有样本、所有标签的结果平均起来。

用大白话说，LogLoss 惩罚的是「confidently wrong」，也就是「很自信地猜错」。如果你给真实为 1 的标签预测了 0.99，损失很小；如果你给真实为 1 的标签预测了 0.01，损失就会非常大。反过来，给真实为 0 的标签预测 0.99，同样会被重罚。所以这个指标不只看「有没有猜对」，还看「你有多确信」。

比赛官方还对概率做了截断：预测值必须落在 `[1e-15, 1 - 1e-15]` 之间。这是为了防止出现 `log(0)` 变成无穷大。最终排行榜按 LogLoss 从低到高排序，分数越低说明模型给出的概率越接近真实情况。

> **小提示**：因为是多标签问题，33 个标签的难度并不一样。有的标签很常见，模型容易学；有的标签非常稀少，预测起来就很难。后面我们会按标签分别看损失，而不是只看一个总数。

---

## 19.2 探索性数据分析（EDA）

我们首先把训练特征和训练标签分别读进来，再按 `id` 拼成一张表，看看数据到底长什么样、有多少行、哪些列是字符串、哪些列缺失严重。

```python
import pandas as pd

# 训练特征：id 作为索引，其余 145 列是 x1 ~ x145
train_x = pd.read_csv(
    '../input/tradeshift-text-classification/train.csv.gz',
    index_col='id'
)

# 训练标签：id 作为索引，其余 33 列是 y1 ~ y33
train_y = pd.read_csv(
    '../input/tradeshift-text-classification/trainLabels.csv.gz',
    index_col='id'
)

# 按 id 左右拼接，得到「特征 + 标签」的完整训练表
train_df = pd.concat([train_x, train_y], axis=1)

# 看数据规模：行数是文本块数量，列数是 145 个特征加 33 个标签
print(train_df.shape)

# 统计每一列的数据类型，区分数值列和字符串列
print(train_df.dtypes.value_counts())

# 统计每列缺失值数量，从多到少排序，先看最严重的 10 列
missing = train_df.isnull().sum().sort_values(ascending=False)
print(missing.head(10))
```

接着我们挑几个典型列，看看它们的取值范围有多集中：

```python
# 看看 x1 这种 YES/NO 列的分布，判断是不是“一家独大”
print(train_df['x1'].value_counts())

# 再数一数每列有多少种不同取值
# 字符串列如果唯一值特别多，就不适合直接做 one-hot
nunique = train_df.nunique().sort_values(ascending=False)
print(nunique.head(15))
```

跑完这一段后看到，训练表是 **1,700,000 行、178 列**。178 列正好是 145 个特征加 33 个标签，说明数据规模和官方描述对得上。列的类型很混杂：一部分是数字，一部分是 `YES` / `NO` 这样的布尔值，还有一部分是长字符串。那些长字符串看起来像 `dqOiM6yBYgnVSezBRiQXs9bvOFnRqrtIoXRIElxD7g8=` 这样的 Base64 内容，其实就是文本的加密哈希。哈希本身没有语义，但同一个词在不同文档里会得到同一个哈希，所以它仍然可以作为「类别编号」来用。

我们接着统计缺失值，发现缺失并不是零零散散分布的，而是集中在若干列上。后面做清洗时我们看到，有 21 列在同样的 327,804 行上是空的。这个数字很关键：它说明这些列不是随机缺失，而是同一批「没有相邻文本块」的样本共同缺失。对这类缺失，简单地填一个全局平均值并不合适，因为缺失本身就带有「页面边缘」的含义。

---

## 19.3 数据清洗

### 缺失值处理：缺失本身就是信息

Tradeshift 的缺失值不应该机械地填 0 或平均值。对于关系特征列，缺失表示「没有相邻文本块」，这本身就是一种有价值的信息。我们的做法是：

1. 对数值型列，用中位数填充，并额外增加一列 `_is_missing` 标记原本是否缺失；
2. 对字符串型列，把缺失填成 `'Missing'` 作为一个独立的类别。

```python
import numpy as np

# 对数值列：中位数填充 + 缺失标记
for col in train_df.columns:
    if train_df[col].isnull().sum() > 0:
        if train_df[col].dtype in ['int64', 'float64']:
            train_df[col + '_is_missing'] = train_df[col].isnull().astype(int)
            train_df[col] = train_df[col].fillna(train_df[col].median())
        else:
            train_df[col] = train_df[col].fillna('Missing')
```

### 特征类型区分

- **数值列**：直接用，做标准化或归一化；
- **YES/NO 列**：映射为 0/1；
- **字符串列（哈希等）**：当作类别编码，用 LabelEncoder 或目标编码，不要直接做独热编码（基数太大会导致维度灾难）。

---

## 19.4 特征工程

### 哈希特征作为类别

内容特征里的 Base64 哈希字符串，虽然没有语义，但同一个哈希值代表同一个原始文本。我们可以把每个哈希值当作一个类别，用 LabelEncoder 编码成整数，或者做目标编码（用该哈希值对应的标签平均值替代）。

```python
from sklearn.preprocessing import LabelEncoder

# 对字符串列做 LabelEncoder
le = LabelEncoder()
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        train_df[col] = le.fit_transform(train_df[col].astype(str))
```

### 标签相关性

33 个标签之间可能存在相关性，比如某个文本块既是「日期」又是「金额」的可能性较低。我们可以先按每个标签单独建模，如果分数不够高，再考虑标签链（Classifier Chain）或标签权重。

---

## 19.5 模型训练与调优

### 在线线性模型：每个标签单独训练

数据量 170 万行，特征 145 列，如果用传统的批量模型（如随机森林、XGBoost），训练速度会很慢。Tradeshift 社区高分方案采用的是**在线线性模型**：每个标签单独训练一个线性模型（如 `SGDClassifier` 或 `PassiveAggressiveClassifier`），数据分批次喂入，逐步更新权重。

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# 对特征做标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df.drop([f'y{i}' for i in range(1, 34)], axis=1))

# 为每个标签训练一个独立的在线线性模型
models = {}
for label in [f'y{i}' for i in range(1, 34)]:
    y = train_df[label].values
    model = SGDClassifier(loss='log_loss', random_state=42, n_jobs=-1)
    # 分批次训练，模拟在线学习
    batch_size = 100000
    for start in range(0, len(X_train), batch_size):
        end = min(start + batch_size, len(X_train))
        model.partial_fit(X_train[start:end], y[start:end], classes=[0, 1])
    models[label] = model
```

`partial_fit` 支持增量学习，适合大规模数据。`loss='log_loss'` 使得模型输出概率，与 LogLoss 评价指标一致。

### 调优思路

- 学习率：在线模型对学习率敏感，可以尝试 `learning_rate='optimal'` 或固定值；
- 正则化：`alpha` 控制 L2 正则强度，防止过拟合；
- 类别不平衡：稀有标签正例少，可以用 `class_weight='balanced'` 或对正例进行加权。

---

## 19.6 模型测试与提交

### 生成提交文件

训练完成后，对测试集做预测，输出每个标签的概率（0~1 之间），最后按 `id_label` 格式拆成多行。

```python
# 读取测试集
test_x = pd.read_csv('../input/tradeshift-text-classification/test.csv.gz', index_col='id')

# 同样的清洗流程（中位数填充、缺失标记、LabelEncoder）
# ... 省略，与训练集保持一致 ...

# 特征标准化（用训练集的 scaler）
X_test = scaler.transform(test_x)

# 为每个标签预测概率
submission_rows = []
for label, model in models.items():
    probs = model.predict_proba(X_test)[:, 1]  # 取正类概率
    # 概率截断到 [1e-15, 1-1e-15]，防止 log(0)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)
    for idx, p in zip(test_x.index, probs):
        submission_rows.append({
            'id_label': f'{idx}_{label}',
            'pred': p
        })

submission = pd.DataFrame(submission_rows)
print(submission.head())
print("提交行数 =", len(submission))

# 核对清单
expected_rows = len(test_x) * 33
assert len(submission) == expected_rows, f"提交行数 {len(submission)} 与预期 {expected_rows} 不一致"
assert submission['pred'].isnull().sum() == 0, "预测概率包含缺失值"
assert ((submission['pred'] >= 1e-15) & (submission['pred'] <= 1 - 1e-15)).all(), "概率超出截断范围"

submission.to_csv('submission.csv', index=False)
print("提交文件已生成：submission.csv")
```

**核对说明**：
- 行数必须等于测试样本数 × 33，不能多也不能少。
- `id_label` 格式必须是 `id_ystandard`，如 `1700001_y1`。
- `pred` 必须在 `[1e-15, 1 - 1e-15]` 之间，不能出现 0 或 1。
- 不能有缺失值。

把 `submission.csv` 上传到 Kaggle 比赛页面，你就能在排行榜上看到自己的多标签 LogLoss 分数了。

---

## 19.7 小结

本章我们完成了 Tradeshift 文本分类的完整流程：

1. **比赛背景**：电子单据文本块分类，145 个特征、33 个标签，多标签分类，评价指标为多标签 LogLoss。
2. **EDA**：170 万行训练数据，特征名和标签名均隐藏（x1~x145、y1~y33），特征四类（内容/解析/空间/关系），关系特征缺失表示「没有相邻文本块」。
3. **数据清洗**：缺失本身就是信息，数值列用中位数填充并加 `_is_missing` 标记，字符串列填 `'Missing'`；特征类型区分（数值/YES-NO/字符串）。
4. **特征工程**：Base64 哈希作为类别编码，LabelEncoder 处理字符串列，避免独热编码导致维度灾难。
5. **模型训练**：在线线性模型（SGDClassifier，`loss='log_loss'`），每个标签单独训练，`partial_fit` 分批次训练适合大规模数据。
6. **模型测试与提交**：输出每个标签的概率并截断到 `[1e-15, 1-1e-15]`，按 `id_label,pred` 格式拆成多行，生成 `submission.csv`，上传前核对行数、列名、概率范围、缺失值。

这个比赛给我们的最大启发是：**大规模数据下，模型的训练效率比模型的复杂度更重要**。170 万行数据用随机森林或 XGBoost 训练会很慢，而在线线性模型分批次喂入，既能处理大数据，又能输出概率与 LogLoss 指标对齐。同时，缺失值不应该机械填充，关系特征的缺失本身就携带「页面边缘」的业务含义，保留这个信息反而能帮助模型。

文本数据部分到此结束。下一章我们将进入模型调优、集成学习与验证策略章节，讲解超参数调优、集成学习、验证策略等提升比赛分数的通用技巧。
