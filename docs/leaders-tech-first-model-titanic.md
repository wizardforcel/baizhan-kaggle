# First Model: Titanic（多数类基线）

## 一、比赛简介

### 1.1 比赛背景
Titanic 是 Kaggle 上最经典的入门比赛，通常被用作机器学习的第一课。比赛提供了泰坦尼克号乘客的名单，要求根据乘客信息预测每个人是否幸存。这个比赛的意义不在于拿高分，而在于让初学者熟悉「读数据、看数据、建模型、交结果」的完整流程。本材料对应的是主办方提供的第一模型练习，核心是一个多数类基线。

### 1.2 比赛目标
数据被拆成三份：`train.csv`、`validation.csv`、`test.csv`。训练集和验证集包含 `id` 与 `survived` 两列，测试集只包含 `id`。`survived` 是 0/1 标签，1 表示幸存，0 表示未幸存。我们要给测试集的每个 `id` 预测一个 `survived` 值。

### 1.3 评价标准
评价指标是分类准确率，即预测正确的样本数除以总样本数。准确率越高越好，满分是 1.0。由于这是一个二分类问题，随机瞎猜大约只有五成准确率，多数类基线通常会略高于五成。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先定位比赛数据目录，并把三份表读进来。材料里的写法是用 `Path.rglob` 在 `/kaggle/input` 下自动寻找 `train.csv`，这样可以避免写死路径。

```python
from pathlib import Path
import pandas as pd

# 在输入目录里递归查找 train.csv，找到唯一一份
matches = list(Path('/kaggle/input').rglob('train.csv'))
assert len(matches) == 1, '只应有一份 First Model: Titanic 的数据'
data = matches[0].parent

# 分别读取训练、验证、测试三份表
train = pd.read_csv(data / 'train.csv')
validation = pd.read_csv(data / 'validation.csv')
test = pd.read_csv(data / 'test.csv')
train.head()
```

打印 `train.head()` 后可以看到，训练集只有 `id` 和 `survived` 两列。`id` 是乘客编号，只是用来对齐提交文件的，不包含任何预测信息；`survived` 才是我们要学的标签。

接下来我们统计训练集里两类标签各有多少条。

```python
# value_counts 会按出现次数从多到少排列
counts = train['survived'].value_counts()
print(counts)
```

这一步很关键：多数类基线的全部逻辑就是「找出出现次数最多的那个标签」。如果 0 比 1 多，就全部预测 0；反之全部预测 1。我们不需要看任何特征，因为基线故意不使用特征，它的作用是给后面的模型立一个「必须超过」的最低标准。

### 2.2 数据清洗
对多数类基线来说，数据清洗这一步几乎可以省略：我们只需要标签列，`id` 列保持原样用于提交。材料里也没有做缺失值填充、异常值处理或特征缩放。这里体现了一个重要原则：**清洗的复杂度应该和模型匹配**。一个只用标签的基线，没必要先花时间处理特征。

不过我们还是确认一下标签的取值是否只有 0 和 1，避免脏数据影响统计。

```python
# 确认标签只有 0/1 两种取值
assert set(train['survived'].unique()) <= {0, 1}
print('标签种类:', sorted(train['survived'].unique()))
```

如果标签里出现了 2 或者其他值，就说明数据有问题，需要先修正。确认无误后即可进入建模。

### 2.3 模型训练与超参数调优
所谓「训练」，在这里只是数一下哪一类更多。材料用了一个很巧的写法：`counts.get(1, 0) > counts.get(0, 0)`，如果 1 的数量严格大于 0，就返回 1，否则返回 0。这样在两类数量相等时会稳定地选择 0。

```python
# 取两个标签的计数，缺失时用 0 补上
counts = train['survived'].value_counts()

# 若 1 的数量严格多于 0，则多数类为 1，否则为 0（平局选 0）
majority = int(counts.get(1, 0) > counts.get(0, 0))
print('Majority class:', majority)
```

这个「模型」没有任何超参数可以调。我们唯一需要确认的是平局规则：如果训练集里 0 和 1 一样多，`>` 会返回 `False`，`int(False)` 就是 0，所以平局时选 0。规则固定、结果可复现，这正是基线应有的样子。

为了确认基线到底有多强，我们把它放到验证集上算一次准确率。

```python
# 用多数类去预测验证集，计算准确率
validation_pred = majority
val_acc = (validation['survived'] == validation_pred).mean()
print('Validation accuracy:', val_acc)
```

这一步让我们在提交之前心里有数：后面自己写的模型只要验证准确率不超过这个数，就说明还没有学到有用的东西。

### 2.4 模型测试与指标报告
最后生成提交文件。提交文件必须只有 `id` 和 `survived` 两列，行数和顺序与测试集一致，且不能带 pandas 的行索引。

```python
# 构造提交表：id 来自测试集，survived 全部填多数类
submission = pd.DataFrame({
    'id': test['id'],
    'survived': majority
})

# index=False 防止把行号写进 CSV
submission.to_csv('/kaggle/working/submission.csv', index=False)

print('Saved', len(submission), 'predictions')
submission.head()
```

保存后，在 Kaggle 上点击 Save Version，等运行结束，打开 Output 里的 `submission.csv` 提交即可。这个基线的价值在于给出一个明确的参照分数；比赛后续的任务就是用自己的模型超过它。

## 三、总结
本文中我们完成了 First Model: Titanic 的多数类基线。我们首先读取了训练、验证、测试三份数据，确认训练集只有 `id` 和 `survived` 两列。其次检查了标签取值并统计类别数量。接着用「出现次数最多的标签」作为预测值，平局时选择 0。然后在验证集上计算了基线准确率。最后按要求的格式生成 `submission.csv`。这个流程虽然简单，但完整走通了从读数据到提交的每一步，为后续建立真正的分类模型打好了基础。
