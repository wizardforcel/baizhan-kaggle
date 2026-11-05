# Just the Basics Strata 2013

## 一、比赛简介

### 1.1 比赛背景

这是一场 Kaggle 在 Strata 2013 大会上举办的入门级练习赛，名字就叫《Just the Basics》。比赛的目的很简单：给初学者一个数值型二分类任务，让大家用最基础的机器学习方法从头到尾跑一遍建模流程。数据看起来是 100 维的数值特征，没有文本、没有图片，就是典型的表格数据。材料里提供的 `train.csv` 只有 600 条样本，`test.csv` 有 4000 条样本，正是那种「数据量不大、适合快速尝试」的做法。

从代码里能看出，训练标签单独放在 `train_labels.csv`，特征和标签要自己拼接。这种分开存放的方式在早期 Kaggle 入门赛里很常见，主要是为了让选手练习数据对齐。整体难度很低，重点在于学会读数据、处理缺失值、选模型、看指标，而不是堆复杂模型。

### 1.2 比赛目标

我们拿到的数据有三份：

* `train.csv`：600 行 × 100 列的训练特征，没有列名，用数字 0～99 表示特征。
* `train_labels.csv`：600 行 × 1 列的二分类标签，值只有 0 和 1。
* `test.csv`：4000 行 × 100 列的测试特征，需要我们给出预测。

我们的目标就是根据训练集学到一个映射关系，然后对测试集里的 4000 条样本给出 0/1 预测，并按比赛要求的格式提交。材料里没有给出目标字段的具体业务含义，我们只知道它是一个二分类问题，标签分布大约 42% 为 1，58% 为 0，属于轻微不平衡。

### 1.3 评价标准

材料中的 notebook 里反复用 `roc_curve` 和 `auc` 来评估分模型的好坏，也用 `model.score` 看训练集准确率。结合入门赛的常见做法，比赛的官方指标通常是 ROC AUC，即受试者操作特征曲线下面积。它衡量的是模型把正样本排在负样本前面的能力，对阈值不敏感。准确率也会作为一个参考，但 AUC 更能反映排序质量。分数越高越好，随机猜测的 AUC 是 0.5。

> 小提示：因为样本只有 600 条，验证集的划分会很大程度上影响指标，所以我们后面会用 `train_test_split` 做内部验证，而不是直接在全量训练集上看分数。

## 二、解题过程

整场比赛的思路是一句话：先摸清数据长什么样，填掉缺失值，用最简单的线性模型和树模型试一试，再用 ROC AUC 选出表现最好的方案。我们按 EDA、清洗、训练调参、测试报告四步来走。

### 2.1 数据探索（EDA）

第一步是把数据读进来，看形状、缺失情况和基本统计量。

```python
import pandas as pd
import numpy as np

# 读取训练特征，文件没有表头，按数字索引
train = pd.read_csv("/kaggle/input/just-the-basics-strata-2013/train.csv", header=None)
# 读取标签
label = pd.read_csv("/kaggle/input/just-the-basics-strata-2013/train_labels.csv", header=None)
# 读取测试特征
test = pd.read_csv("/kaggle/input/just-the-basics-strata-2013/test.csv", header=None)

print("训练特征形状:", train.shape)  # (600, 100)
print("测试特征形状:", test.shape)    # (4000, 100)
print("标签形状:", label.shape)       # (600, 1)
```

我们首先确认了数据规模：训练集只有 600 条样本，却有 100 个特征，属于特征比样本多的情况，需要注意过拟合。测试集有 4000 条，说明最终评估是大量样本上的泛化能力。

接着看缺失值和分布。

```python
# 统计训练集中所有缺失值的总数
missing_total = train.isnull().sum().sum()
print("训练集缺失值总数:", missing_total)  # 1745

# 看标签的大致分布
print(label.describe())
```

结果显示训练集里共有 1745 个缺失值，并非每行都完整，说明必须先处理缺失。标签的均值约 0.42，标准差 0.49，最大值 1 最小值 0，证实是二分类且稍微失衡。`train.describe()` 显示各特征的均值大多在 0～3 之间，标准差接近 1～1.5，有个别特征最大值超过 9000，说明存在异常大值，需要在做模型前先填缺失。

我们还用 `train.head()` 快速看了前几行，发现特征都是数值型，没有字符串，方便直接喂给 sklearn。

### 2.2 数据清洗

这份数据不需要复杂的特征工程，主要问题是缺失值和可能的尺度差异。

```python
# 用每列的均值填充缺失值，这是入门最简单且常用的做法
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

# 再次确认是否还有缺失
print("填充后缺失:", train.isnull().sum().sum())
```

我们选择用均值填充，因为样本量小，均值能保持分布的大致中心。填充后缺失值归零，模型就不会报错。

在尝试建模前，我们把特征和标签转成 numpy 数组，方便 sklearn 使用。

```python
train_X = train.to_numpy()
train_y = label.to_numpy().ravel()  # 把列向量压平成一维
```

这里 `ravel()` 很重要，sklearn 的很多分类器要求 y 是一维数组，不然会报 DataConversionWarning。我们在材料里也看到了同样的警告，先处理好可以省去很多麻烦。

### 2.3 模型训练与超参数调优

我们先用最简单的线性模型做基准。

```python
from sklearn.linear_model import LinearRegression, LogisticRegression

# 线性回归先试一下，看数值拟合程度
reg = LinearRegression()
reg.fit(train_X, train_y)
print("线性回归训练集 R2:", reg.score(train_X, train_y))
```

线性回归的训练集 R2 约 0.62，说明特征和标签有一定线性关系，但这是回归任务，实际我们需要分类。

接着换成逻辑回归，这是二分类的经典入门模型。

```python
# 逻辑回归，solver='liblinear' 适合小样本
model = LogisticRegression(solver='liblinear')
model.fit(train_X, train_y)
print("逻辑回归训练集准确率:", model.score(train_X, train_y))
```

训练集准确率约 0.94，表现不错。为了不被训练集分数骗到，我们用分层抽样划分验证集，并比较几种常用模型。

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

X_train, X_val, y_train, y_val = train_test_split(
    train_X, train_y, test_size=0.33, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(solver='liblinear'),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "RandomForestRegressor": RandomForestRegressor()
}

results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_val)
    # 用 ROC AUC 评估：把预测值（概率或回归输出）当分数排序，越高越像正类
    fpr, tpr, _ = metrics.roc_curve(y_val, y_pred, pos_label=1)
    results[name] = metrics.auc(fpr, tpr)

for name, auc in results.items():
    print(f"{name}: AUC = {auc:.4f}")
```

我们首先试了逻辑回归，其次试了 K 近邻，最后试了随机森林回归。在验证集上得到的 AUC 分别是：随机森林约 0.929，逻辑回归约 0.855，K 近邻分类约 0.675，K 近邻回归约 0.767，线性回归约 0.895。随机森林在这种小样本、多特征的任务上表现最好。这里有一个容易困惑的点：随机森林回归输出的其实是 0～1 之间的连续值（例如 0.92、0.52），它并不是严格的 0/1 分类结果，但拿来当分数算 AUC 是可以的。

由于样本只有 600 条，我们没有做系统的网格搜索，而是用默认参数横向比较了几种模型，靠「选模型 + 填缺失」来提升稳定性。材料里作者也主要是做了模型对比，没有做正式的超参数调优。

### 2.4 模型测试与指标报告

选好模型后，我们用全量训练数据重新训练，再对测试集预测并生成提交文件。

```python
# 用全部训练数据再训练一次
best_model = RandomForestRegressor()
best_model.fit(train_X, train_y)

# 对测试集预测
test_X = test.to_numpy()
test_pred = best_model.predict(test_X)

# 生成提交文件
submission = pd.DataFrame(test_pred)
submission.head()
submission.to_csv("justdoit.csv", index=False)
```

我们先在验证集上确认了指标，然后用全量数据训练以利用更多信息。最终的提交文件包含 4000 条预测值，格式与示例提交一致。

在验证集上的详细报告显示，逻辑回归的准确率在训练集上很高，但在验证集的 AUC 约 0.86；随机森林回归的 AUC 达到约 0.93，是我们目前最好的方案。这说明模型对数据是有学习能力的，但也提醒我们小样本下过拟合风险很大。

## 三、总结

本文中我们完成了 Just the Basics Strata 2013 的完整入门流程。我们首先通过 EDA 摸清了数据：训练集 600 条、100 维特征、测试集 4000 条，存在 1745 个缺失值，标签为轻度不平衡的二分类。其次我们用均值填充完成了最简单的数据清洗，保证模型能正常训练。接着我们对比了线性回归、逻辑回归、K 近邻和随机森林，用 ROC AUC 作为选模标准，最终随机森林在验证集上表现最好。最后我们用全量数据重新拟合并生成提交文件。

这个比赛给入门者的最大启发是：**先让数据跑通，再谈模型复杂化**。在样本很少的情况下，均值填充、简单模型和合理的验证划分，往往比堆模型更有效。特征比样本多的情况很容易过拟合，所以保持模型简单、认真看指标，才是入门阶段最稳的路子。
