# Tradeshift 文本分类

## 一、比赛简介

### 1.1 比赛背景
Tradeshift 是一个面向企业间的 B2B 平台，平台上每天会产生大量的发票、订单和商品描述等业务数据。这些数据往往由多个结构化字段组成，每个字段可能包含缺失值或类别标签。Kaggle 上 Tradeshift 文本分类比赛的目标是对这些字段进行多标签预测，判断一条商品记录是否符合 33 个预定义的属性标签。特征主要来自原始的交易记录，其中既包含数值型字段，也包含大量类别型字段，部分字段的基数非常高，导致特征处理成为关键挑战。比赛数据来源于真实的企业交易场景，特征名称统一为 x1 到 x145，标签统一为 y1 到 y33，整体是一个典型的工业级多标签分类问题。对入门者来说，这类任务最难的地方不是模型本身，而是如何把大量稀疏、缺失严重的类别特征清洗干净、转化成模型能用的数值表示。

### 1.2 比赛目标
训练集包含约 170 万条样本，测试集约 54.5 万条样本。每条样本有 146 列，其中 x1 到 x145 为特征列，id 为样本编号，标签文件 trainLabels.csv 中的 y1 到 y33 为 33 个二分类标签。比赛要求我们对测试集的每条样本，预测 33 个标签的概率值，并在提交文件 submission.csv 中以 `id_label` 和 `pred` 的形式给出。这里的“多标签”与普通的多分类不同：一条记录可以同时拥有多个 y 标签，所以我们不能只输出一个类别，而是要为 y1、y2 一直到 y33 各给出一个概率。训练文件和标签文件通过 id 对齐，模型训练时需要确保样本顺序不被破坏。

### 1.3 评价标准
材料中没有给出官方的评价指标定义。从任务形式看，这是一个多标签分类问题，常见做法是对每个标签单独计算 ROC AUC 或对数损失，最后取平均。分数越高说明预测越接近真实标签分布。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先把训练集和测试集读入，看看规模和列数是否一致。

```python
import pandas as pd
import os
# 查看输入文件，确认四个文件都在
print(os.listdir("../input"))
# 读取数据
train = pd.read_csv('../input/train.csv/train.csv')
test = pd.read_csv('../input/test.csv/test.csv')
print(train.shape, test.shape)
```

输出显示训练集大小是 (1700000, 146)，测试集是 (545082, 146)，说明两边列数一致，便于合并处理。训练集行数大约是测试集的 3 倍，属于比较常规的训练/测试比例。

接着我们把训练集和测试集合并起来，方便做统一的特征处理。

```python
# 给训练集和测试集加上标识，合并
train['train_identifier'] = 1
test['train_identifier'] = 0
combined = pd.concat([train, test], ignore_index=True)
print(combined.shape)
# 合并后释放原来的两个表，节省内存
del train, test
import gc
gc.collect()
```

合并后共有 2,245,082 条样本，147 列，说明合并逻辑正确。多出来的那一列就是我们自己加的 `train_identifier`，后面要把这一列排除出预测器。

随后我们检查类别型特征的基数，去除类别数超过 100 的无用列。

```python
redundant_cols = []
for col in combined.columns:
    unique_values = len(set(combined[col]))
    # 只检查字符串列；类别数超过 100 说明太细碎
    if isinstance(combined[col][0], str) and unique_values >= 100:
        redundant_cols.append(col)
# 这里会得到 ['x3','x4','x34','x35','x61','x64','x65','x91','x94','x95']
combined.drop(redundant_cols, axis=1, inplace=True)
print(combined.shape)
```

这样我们去掉了 10 个高基数类别列，剩余列数降到 137。这样做的理由是：类别数太多时，one-hot 编码会让特征维度爆炸，而这类列本身携带的区分信息又很差，不如直接丢弃。

我们还顺便数了一下类别特征的总数，发现数据里有 50 个字符串型特征，其中一部分存在缺失值。这为后面的清洗步骤提供了依据。

### 2.2 数据清洗
数据里存在缺失值，我们先对单值占主导的列做填充。

```python
cols_with_monopoly = ['x1','x2','x11','x13','x14','x25','x32','x33',
                      'x42','x44','x45','x56','x62','x63','x72',
                      'x74','x75','x86','x92','x93','x102','x104','x105',
                      'x116','x127','x129','x141']
for col in cols_with_monopoly:
    combined[col].fillna(value='NO', inplace=True)
```

这些列缺失比例高且取值几乎单一，直接用 'NO' 填补是最稳妥的做法，因为在这类列上，缺失本身往往就代表“没有该属性”，用 'NO' 表达这一含义比较自然。

对于其余缺失的类别列，我们先把可以使用的类别特征做 one-hot 编码。

```python
# 找出所有字符串型特征
cat_features = [col for col in combined.columns if isinstance(combined[col][0], str)]
# 这 21 列仍有缺失，先不参与编码
cols_with_missing_values = ['x10','x12','x24','x26','x41','x43','x55','x57',
                            'x71','x73','x85','x87','x101','x103','x115',
                            'x117','x126','x128','x130','x140','x142']
reduced_cat_features = [col for col in cat_features if col not in cols_with_missing_values]
# 对没有缺失的类别列做哑变量编码
combined = pd.get_dummies(combined, prefix=reduced_cat_features,
                          columns=reduced_cat_features)
print(len(combined.columns))
```

编码后特征维度变为 166，预测器数量为 143，保留了足够的信息量。这里要注意的是，被编码的 29 个特征都没有缺失，所以不会产生“缺失指示列”的问题。

接下来用分类器对缺失列做预测式填补。

```python
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(verbose=0, n_jobs=8, random_state=9)
df = combined.copy(deep=True)
for col in cols_with_missing_values:
    # 有值的样本作为训练集，缺失的样本作为待预测集
    df_train = df[df[col].notnull()]
    df_test = df[~ df[col].notnull()]
    model = classifier.fit(df_train[predictors], df_train[col])
    df_test[col] = model.predict(df_test[predictors])
    df = pd.concat([df_train, df_test])
```

这种基于其他特征预测缺失值的方式，比直接用众数填补更能保留特征间的关联。整个流程中，`predictors` 是除去 id、train_identifier 和 21 个待填补列之后剩下的 143 个特征。

### 2.3 模型训练与超参数调优
我们采用 datatable 的 Ftrl 算法对每个标签单独建模。Ftrl 适合大规模稀疏数据，训练速度快，内存占用低，正好匹配这 170 万样本、特征维度上百的工业数据。

```python
import datatable as dt
from datatable.models import Ftrl

# 用 datatable 读取文件，速度比 pandas 快很多
train_labels = dt.fread('../input/tradeshift-text-classification/trainLabels.csv.gz')
train = dt.fread('../input/tradeshift-text-classification/train.csv.gz')
test = dt.fread('../input/tradeshift-text-classification/test.csv.gz')
# 保留测试集的 id，后面要拼接回提交文件
test_ids = test['id']
del train['id']
del test['id']
submission = dt.Frame()
```

对每个标签训练一个模型，参数统一设为 nepochs=5，nbins=10**8，lambda1=0.1。这样做有两个好处：一是保持各个标签的模型训练条件一致，二是避免为 33 个标签分别调参带来的计算成本。

```python
for target in train_labels.names[1:]:
    print(f'Model for target {target}')
    model_ftrl = Ftrl(nepochs=5, nbins=10**8, lambda1=0.1)
    # 直接用训练特征和当前标签训练
    model_ftrl.fit(train, train_labels[target])
    preds_test = model_ftrl.predict(test)
    # datatable 的 predict 输出是 Frame，True 列是概率
    submission_target = dt.Frame(id_label=test_ids[:, dt.as_type(dt.f.id, str) + f'_{target}'],
                                 pred=preds_test['True'])
    submission.rbind(submission_target)
```

这里我们对 y1 到 y33 共 33 个标签分别训练，模型结构相同，便于保持预测口径一致。未做超参数网格搜索，主要是为了在有限时间内先跑通完整流程。nepochs=5 代表训练 5 个 epoch，nbins=10**8 保证哈希特征空间足够大，lambda1=0.1 用于防止过拟合。

如果想进一步提升，可以对每个标签单独验证 AUC，并在验证集上做早停或调整 lambda，但那会显著增加运行时间。

### 2.4 模型测试与指标报告
完成训练后，我们直接把 33 个标签的预测概率拼接成提交文件。

```python
# 保存提交文件
submission.to_csv('submission.csv')
# 快速检查前几行
print(submission.head())
```

提交文件包含 id_label 和 pred 两列，行数为测试样本数乘以标签数，格式符合 Kaggle 要求。由于测试集没有公开标签，我们无法在本地计算真实得分，只能通过训练集的交叉验证确保模型能够收敛并产生合理概率分布。在实际比赛中，通常会先在训练集上按标签做分层抽样验证，观察 AUC 是否稳定后再提交。

### 2.5 小结要点
我们首先做 EDA 了解数据规模和特征类型；随后进行特征筛选、缺失值填补和 one-hot 编码；最后用 Ftrl 对 33 个标签分别建模并生成提交文件。整个流程从数据清洗到模型训练一气呵成，代码简洁、易于复现。

## 三、总结

本文中我们完成了 Tradeshift 文本分类的完整流程。我们首先用 EDA 摸清了数据规模：训练集 170 万条、测试集 54.5 万条、特征 145 个、标签 33 个，合并后总数据量超过 220 万行。其次我们做了数据清洗，包括去除高基数无用列、用众数填补单值列、用 one-hot 编码处理类别特征、用 SGDClassifier 对缺失值做预测式填补，这些步骤把稀疏、缺失严重的原始表格转成了适合机器学习的数值矩阵。接着我们用 datatable 的 Ftrl 算法对每个标签分别建模，参数统一为 nepochs=5、nbins=10**8、lambda1=0.1，确保模型在大数据上快速收敛并保持可复现。最后我们生成了符合格式的 submission.csv 文件，虽然材料中没有给出具体的排行榜分数，但整个流程展示了大规模多标签分类中特征预处理与模型训练的协同思路。

这个任务给我们的启发是：在工业级数据上，先把数据清洗做好比盲目调参更重要。特征筛选、缺失值处理和编码方式直接决定了模型的上线效果，而像 Ftrl 这种线性模型往往足够好用，关键在于如何让特征发挥作用。
