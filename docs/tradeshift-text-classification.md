# Tradeshift 文本分类（Tradeshift Text Classification）

## 一、比赛简介

### 1.1 比赛背景

Tradeshift 是一家做企业间电子单据流转的公司。日常生活中，一张发票、一份采购单或者一份合同，往往先被扫描成图片，再由 OCR（光学字符识别）把图片里的文字识别出来。识别出来的文字并不是一整段，而是被切成很多个「文本块」，每个文本块外面画一个矩形框，记录它在页面上的位置。这个比赛要解决的问题就是：**给一个文本块，判断它属于哪一类内容**，比如它是不是发票号码、日期、金额、公司名称等等。

比赛数据来自成千上万份真实单据，官方说明里提到有「数百万个单词」。每份文档会被切成若干个文本块，所有文档的文本块汇总到一起，一行就是一个文本块。因为单据的格式五花八门，语言也不统一，再加上 OCR 本身会认错字，所以数据里天然带着噪声。更麻烦的是，主办方把特征名和标签名都隐藏了，只留下 `x1` 到 `x145` 这样的编号，以及 `y1` 到 `y33` 这样的标签编号，我们不知道每一列具体代表什么，只能靠数据本身的统计规律去建模。

这个比赛还有一个特点：数据量很大。官方给出的样本总数约 210 万，其中训练集约 170 万行，测试集约 54.5 万行。特征有 145 个，标签有 33 个。对入门者来说，这是一次很好的练习——它逼着我们面对真实场景里的三个问题：数据大、特征含义不明、缺失值多。

### 1.2 比赛目标

比赛给出的输入是每个文本块的 145 个特征，输出是**这个文本块属于 33 个标签中每一个的概率**。注意，这不是「33 选 1」的多分类，而是**多标签分类**：一个文本块可以同时属于多个标签，所以 33 个标签要分别预测，每个标签输出一个 0 到 1 之间的概率。

数据文件有四个：

- `train.csv`：训练集特征，每行一个文本块，列是 `id` 加上 `x1` 到 `x145`；
- `trainLabels.csv`：训练集标签，每行对应一个训练样本，列是 `id` 加上 `y1` 到 `y33`，取值是 0 或 1；
- `test.csv`：测试集特征，列和 `train.csv` 一样，但没有标签；
- `sampleSubmission.csv`：提交格式示例，只有两列，`id_label` 和 `pred`。

提交时要把每个测试样本的每个标签拆成一行，例如编号为 `1700001` 的样本、标签 `y1`，就写成 `1700001_y1`，后面跟上预测概率。33 个标签就要写 33 行，所以提交文件的行数是测试样本数乘以 33。

这些特征大致可以分成四类，官方说明里写得很清楚：

- **内容特征**：原始文本的加密哈希值，看起来像一串 Base64 字符串；
- **解析特征**：这个文本块能不能被解析成数字、字母、字母数字混合等；
- **空间特征**：文本框的位置、大小、宽高比等；
- **关系特征**：和同一份文档里上下左右相邻文本块的关系。如果一个文本块在页面最上面，它上面没有别的文本块，这类特征就会空着。

这也解释了后面我们为什么会看到大量缺失值：不是数据坏了，而是「没有相邻文本块」这件事本身就需要用缺失来表示。

### 1.3 评价标准

比赛使用**多标签 LogLoss（对数损失）**作为评价指标，越小越好。它的意思是：对每一个测试样本的每一个标签，都拿预测概率和真实标签算一次交叉熵，然后把所有样本、所有标签的结果平均起来。

用大白话说，LogLoss 惩罚的是「 confidently wrong 」，也就是「很自信地猜错」。如果你给真实为 1 的标签预测了 0.99，损失很小；如果你给真实为 1 的标签预测了 0.01，损失就会非常大。反过来，给真实为 0 的标签预测 0.99，同样会被重罚。所以这个指标不只看「有没有猜对」，还看「你有多确信」。

比赛官方还对概率做了截断：预测值必须落在 `[1e-15, 1 - 1e-15]` 之间。这是为了防止出现 `log(0)` 变成无穷大。最终排行榜按 LogLoss 从低到高排序，分数越低说明模型给出的概率越接近真实情况。

> 小提示：因为是多标签问题，33 个标签的难度并不一样。有的标签很常见，模型容易学；有的标签非常稀少，预测起来就很难。后面我们会按标签分别看损失，而不是只看一个总数。

---

## 二、解题过程

整场比赛的主线可以概括成一句话：**先把 170 万行、145 列的特征整理成模型能吃的格式，再给 33 个标签各训练一个在线线性模型，最后按比赛要求输出概率。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步走。

### 2.1 数据探索（EDA）

我们首先把训练特征和训练标签分别读进来，再按 `id` 拼成一张表，看看数据到底长什么样、有多少行、哪些列是字符串、哪些列缺失严重。

```python
# 导入 pandas，用来读写 CSV 表格
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

# 按 id 左右拼接，得到“特征 + 标签”的完整训练表
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

我们跑完这一段后看到，训练表是 **1,700,000 行、178 列**。178 列正好是 145 个特征加 33 个标签，说明数据规模和官方描述对得上。列的类型很混杂：一部分是数字，一部分是 `YES` / `NO` 这样的布尔值，还有一部分是长字符串。那些长字符串看起来像 `dqOiM6yBYgnVSezBRiQXs9bvOFnRqrtIoXRIElxD7g8=` 这样的 Base64 内容，其实就是文本的加密哈希。哈希本身没有语义，但同一个词在不同文档里会得到同一个哈希，所以它仍然可以作为「类别编号」来用。

我们接着统计缺失值，发现缺失并不是零零散散分布的，而是集中在若干列上。后面做清洗时我们看到，有 21 列在同样的 327,804 行上是空的。这个数字很关键：它说明这些列不是随机缺失，而是同一批「没有相邻文本块」的样本共同缺失。对这类缺失，简单地填一个全局平均值并不合适，因为缺失本身就带有「页面边缘」的含义。

我们还注意到，有些字符串列的取值特别多，比如 `x3`、`x4`、`x34`、`x35`、`x61`、`x64`、`x65`、`x91`、`x94`、`x95` 这 10 列，每一列都有 100 种以上的不同取值。如果直接做 one-hot 编码，会一下子多出成千上万列，既占内存又容易过拟合。这个发现直接决定了下一步的清洗策略。

### 2.2 数据清洗

我们其次做的事情是把训练集和测试集拼在一起做统一处理。这样做的好处是：训练集和测试集经过完全相同的删除列、编码、填补流程，不会出现「训练时是一种格式、预测时是另一种格式」的问题。处理完之后再把它们拆开。

```python
import pandas as pd
import gc

# 读入训练集和测试集，两个文件都含 id 和 x1 ~ x145
train = pd.read_csv('../input/train.csv/train.csv')
test = pd.read_csv('../input/test.csv/test.csv')

# 加一列标记，1 表示训练样本，0 表示测试样本
train['train_identifier'] = 1
test['train_identifier'] = 0

# 上下拼接，方便对两批数据做同样的清洗
combined = pd.concat([train, test], ignore_index=True)
print(combined.shape)          # (2245082, 147)

# 拼完以后原来的两张表就不再需要了，及时释放内存
del train, test
gc.collect()

# 第一步：删掉取值超过 100 种的字符串列
# 这些列做 one-hot 会膨胀得太大，而且大部分取值只出现一两次，学不到稳定规律
redundant_cols = []
for col in combined.columns:
    if isinstance(combined[col][0], str) and combined[col].nunique() >= 100:
        redundant_cols.append(col)
combined.drop(redundant_cols, axis=1, inplace=True)

# 第二步：对“某一个取值占比超过 85%”的列，缺失值直接填 NO
# 因为这些列本来绝大多数就是 NO，填 NO 不会引入太多偏差
cols_with_monopoly = [
    'x1', 'x2', 'x11', 'x13', 'x14', 'x25', 'x32', 'x33',
    'x42', 'x44', 'x45', 'x56', 'x62', 'x63', 'x72', 'x74',
    'x75', 'x86', 'x92', 'x93', 'x102', 'x104', 'x105',
    'x116', 'x127', 'x129', 'x141'
]
for col in cols_with_monopoly:
    combined[col].fillna(value='NO', inplace=True)

# 第三步：把剩下的字符串列做 one-hot，让模型能处理类别变量
cat_features = [
    col for col in combined.columns
    if isinstance(combined[col][0], str)
]
cols_with_missing_values = [
    'x10', 'x12', 'x24', 'x26', 'x41', 'x43', 'x55', 'x57',
    'x71', 'x73', 'x85', 'x87', 'x101', 'x103', 'x115',
    'x117', 'x126', 'x128', 'x130', 'x140', 'x142'
]
reduced_cat_features = [
    col for col in cat_features
    if col not in cols_with_missing_values
]
combined = pd.get_dummies(
    combined, columns=reduced_cat_features
)

# 第四步：对剩下 21 个有缺失的列，用其他列训练一个分类器来“猜”缺失值
# 这里用 SGDClassifier，是因为数据量大，线性模型训练快、内存占用小
from sklearn.linear_model import SGDClassifier

classifier = SGDClassifier(n_jobs=8, random_state=9)
predictors = [
    col for col in combined.columns
    if col not in cols_with_missing_values + ['id', 'train_identifier']
]

for col in cols_with_missing_values:
    # 有值的行拿来训练，缺值的行拿来预测
    known = combined[combined[col].notnull()]
    unknown = combined[combined[col].isnull()]
    classifier.fit(known[predictors], known[col])
    combined.loc[unknown.index, col] = classifier.predict(unknown[predictors])
```

这一套清洗做下来，数据从 147 列变成了 166 列，用来预测的列有 143 个。我们具体做了四件事。

第一，删掉 10 个高基数类别列。高基数就是「不同取值特别多」。这类列即使做了 one-hot，大多数列也会是 0，信息密度很低，反而让模型变慢。删掉它们是一种取舍：我们牺牲了一点点可能的信息，换来了更小、更稳定的特征空间。

第二，对 27 个「一家独大」的列直接填 `NO`。这些列里超过 85% 的样本都是 `NO`，缺失值大概率也应该是 `NO`。这里用众数填补是合理的，因为填别的值反而会制造出数据里原本不存在的模式。

第三，对剩下的 29 个类别列做 one-hot。one-hot 的意思是把「一个有很多取值的列」拆成「很多个 0/1 列」，比如颜色列有红、绿、蓝三种取值，就拆成「是不是红」「是不是绿」「是不是蓝」三列。这样线性模型就能直接处理类别信息。

第四，对 21 个缺失列用 `SGDClassifier` 做迭代填补。思路是：把「这一列有值」的样本当作训练集，用其他所有列去预测这一列；训练好之后，再去预测「这一列缺失」的样本。这样做比填一个全局众数更细，因为它利用了样本之间的相似性。比如两个文本块的空间特征很像，那它们的解析特征大概率也像。

需要说明的是，这种填补方式的计算量不小。每一列都要重新训练一次分类器，21 列就是 21 次。材料里也特别标注了「超参数没有调」，说明作者清楚这一步只是为了让数据完整，并不是要把填补模型做到最好。

### 2.3 模型训练与超参数调优

我们接着进入模型训练。这个比赛官方特别提到「对在线学习（online learning）的方案很感兴趣」，而材料里的主线正好用的是一种适合大数据的在线线性模型：**FTRL**（Follow The Regularized Leader）。FTRL 可以理解成一种「边看数据边更新权重」的线性模型，它不需要把全部数据一次性塞进内存，特别适合这种上百万行、特征又很稀疏的表格数据。

因为是多标签问题，我们的做法是**给 33 个标签各训练一个二分类器**。每个分类器只回答一个问题：「这个文本块属不属于当前这个标签？」33 个分类器合起来，就得到 33 个概率。

```python
# datatable 是一个面向大数据的表格库，读 CSV 比 pandas 更省内存
import datatable as dt
from datatable.models import Ftrl

# 读入特征和标签
train = dt.fread('../input/tradeshift-text-classification/train.csv.gz')
train_labels = dt.fread('../input/tradeshift-text-classification/trainLabels.csv.gz')
test = dt.fread('../input/tradeshift-text-classification/test.csv.gz')

# id 只是样本编号，不参与建模，先单独存起来
test_ids = test['id']
del train['id']
del test['id']

# 用来一行一行攒提交结果
submission = dt.Frame()

# train_labels.names[1:] 跳过 id 列，依次取出 y1、y2 ... y33
for target in train_labels.names[1:]:
    print(f'Model for target {target}')

    # nepochs=5：把训练数据扫 5 遍
    # nbins=10**8：哈希分桶数，桶越多冲突越少，但内存也越大
    # lambda1=0.1：L1 正则，让不重要的特征权重变成 0，起到稀疏化作用
    model_ftrl = Ftrl(
        nepochs=5,
        nbins=10**8,
        lambda1=0.1
    )

    # 用当前标签训练一个二分类模型
    model_ftrl.fit(train, train_labels[target])

    # 对测试集预测，preds_test['True'] 就是“属于该标签”的概率
    preds_test = model_ftrl.predict(test)

    # 按比赛格式拼一行：id_label 形如 1700001_y1
    submission_target = dt.Frame(
        id_label=test_ids[:, dt.as_type(dt.f.id, str) + f'_{target}'],
        pred=preds_test['True']
    )
    submission.rbind(submission_target)

# 33 个标签全部跑完后，一次性写出提交文件
submission.to_csv('submission.csv')
```

这段代码里有三个关键参数，我们逐个解释。

`nepochs=5` 是训练轮数。FTRL 是Online 算法，每看一条样本就更新一次权重。扫一遍数据叫一个 epoch，扫 5 遍就是让模型有 5 次机会修正自己。轮数太少会欠拟合，轮数太多又容易过拟合，5 是一个在「效果」和「时间」之间比较平衡的选择。

`nbins=10**8` 是哈希分桶的数量。FTRL 处理类别特征时，会把特征哈希到一个固定大小的数组里。桶越多，不同特征撞到同一个桶的概率越小，模型能记住的细节越多；但桶越多，占用的内存也越大。这里取一亿个桶，是为了尽量接住 145 列里那些取值很多的类别特征。

`lambda1=0.1` 是 L1 正则的强度。L1 正则会把不重要的特征权重压到 0，相当于自动做特征选择。对高维稀疏数据来说，这一步很重要，因为大部分特征对某个标签其实没什么用，正则能帮模型把注意力集中在少数真正有信号的特征上。

关于调参，我们的做法是先把这三个参数固定在一组「能跑通、效果稳定」的值上，再观察每个标签的损失。原因很现实：33 个标签、170 万行数据，每换一组参数就要重新训练 33 次，成本很高。与其做大规模的网格搜索，不如先把主线跑通，再针对表现特别差的标签单独调整。材料里也采用了这种「先固定一组参数、逐标签训练」的策略。

### 2.4 模型测试与指标报告

我们最后要做两件事：一是用比赛官方的 LogLoss 公式检查模型，二是把预测结果整理成提交文件。

因为测试集没有标签，我们没法直接在排行榜上看到每一项的对错。为了在本地判断模型好坏，我们从训练集里留出一部分样本当验证集，用同一套流程训练和预测，再按官方公式算损失。下面这段代码就是官方 LogLoss 的实现：

```python
import numpy as np

def multi_label_logloss(y_true, y_pred):
    """
    y_true: 真实标签矩阵，形状是 (样本数, 33)，取值 0 或 1
    y_pred: 预测概率矩阵，形状和 y_true 一样，取值 0 到 1
    """
    # 比赛要求把概率截断在 [1e-15, 1-1e-15]，避免 log(0) 变成无穷大
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    # 对每个样本的每个标签算交叉熵，再对所有元素取平均
    loss = -(y_true * np.log(y_pred) +
             (1 - y_true) * np.log(1 - y_pred))
    return loss.mean()

# 假设 valid_true 是验证集真实标签，valid_pred 是模型给出的概率
score = multi_label_logloss(valid_true, valid_pred)
print('validation LogLoss:', score)

# 再按标签拆开看：哪个标签的损失特别高，就说明哪个标签最难学
for j, target in enumerate(train_labels.names[1:]):
    label_loss = multi_label_logloss(
        valid_true[:, j],
        valid_pred[:, j]
    )
    print(target, label_loss)
```

读这个结果时，我们关注两点。

第一，整体 LogLoss 越低越好。它衡量的是「预测概率和真实 0/1 之间的差距」。如果模型给正样本的概率很高、给负样本的概率很低，损失就小；如果模型把正负样本的概率预测反了，损失就会迅速变大。

第二，按标签拆开看，比只看总分更有用。33 个标签的分布通常很不均衡：有的标签在训练集里很常见，模型能学到足够的样本；有的标签可能几百条里才出现一次，模型很难学好。把每个标签的损失打印出来，我们就能知道瓶颈在哪里。如果某个标签的损失明显高于其他标签，下一步就可以针对它做重采样、调整正则强度，或者干脆给它换一个更适合的模型。

确认模型可用后，我们再用全部训练数据重新训练一次，然后对测试集生成提交文件。提交文件的格式必须和 `sampleSubmission.csv` 完全一致：第一列叫 `id_label`，第二列叫 `pred`，每一行是一个「样本编号_标签名」和一个概率。测试集有 545,082 行，33 个标签，所以提交文件大约有 1,798 万行。这个文件很大，写出来之后我们还会检查三件事：行数对不对、列名对不对、概率有没有超出 0 到 1 的范围。比赛里因为格式错误导致提交失败的情况并不少见，这一步不能省。

```python
import pandas as pd

# 读回刚刚写出的提交文件，模拟比赛系统的检查过程
sub = pd.read_csv('submission.csv')

# 行数应该等于“测试样本数 × 标签数”
print('实际行数:', sub.shape[0])
print('应有行数:', 545082 * 33)

# 列名必须和 sampleSubmission.csv 一致
print('列名:', sub.columns.tolist())

# 所有概率都必须落在 0 到 1 之间，否则提交会被判无效
print('概率范围正确:', sub['pred'].between(0, 1).all())
```

确认这三项都没问题之后，我们才把文件提交到排行榜。这个习惯看起来很小，但在数据量大的比赛里，它能帮我们避免「模型明明没问题、却因为格式错误白跑一次」的损失。

---

## 三、总结

本文中我们完成了 Tradeshift 文本分类比赛的完整流程。我们首先用 EDA 摸清了数据：训练集 170 万行、145 个特征、33 个标签，特征混杂着数值、布尔值和 Base64 哈希字符串，还有 21 列在同样的 327,804 行上缺失，说明这些缺失来自「页面边缘没有相邻文本块」这一结构原因。其次我们做了有针对性的数据清洗：删掉 10 个取值超过 100 种的高基数类别列，对 27 个主类别占比超过 85% 的列用 `NO` 填补，对剩下的 29 个类别列做 one-hot，最后用 `SGDClassifier` 逐个填补 21 个缺失列。接着我们给 33 个标签各训练一个 FTRL 二分类器，用 `nepochs=5`、`nbins=10**8`、`lambda1=0.1` 这组参数控制训练轮数、哈希桶数和正则强度。最后我们按官方 LogLoss 公式在验证集上评估，并按 `id_label,pred` 的格式生成提交文件。

这个比赛给我们最大的启发是：**面对大数据和匿名特征，先把数据流程搭稳，比追求复杂模型更重要**。145 个特征我们不知道含义，但通过「看类型、看缺失、看取值数量」这三板斧，依然能做出合理的清洗决策；33 个标签看起来很多，但拆成 33 个独立的二分类问题之后，每个问题都变得可以处理。FTRL 这种在线线性模型虽然简单，却正好适合这种「数据大、特征稀疏、要求概率输出」的场景。对入门者来说，这是一条很值得掌握的基线路线。
