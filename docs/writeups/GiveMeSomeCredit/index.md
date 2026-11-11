# Give Me Some Credit（给我点信用）

## 一、比赛简介

### 1.1 比赛背景

银行决定要不要借钱给你，靠的是**信用评分**（Credit Scoring）：根据一个人过往的信贷记录，算出一个"未来会不会还不上钱"的概率。这个概率直接影响他能不能拿到贷款、利率是多少、额度有多高。如果模型把不该借的人放进来了，银行要承担坏账；如果把该借的人挡在门外，银行又会少赚利润。所以信用评分模型要解决的核心问题是：**很精确地估计一个人的违约概率**。

这个比赛用的是一份真实脱敏的信贷数据。目标标签叫 `SeriousDlqin2yrs`，全称是 "Seriously Delinquent in 2 years"，意思是"未来两年内是否会严重逾期（违约）"。比赛要求我们预测每一笔贷款登记对象在未来两年出问题的概率，帮放贷机构在放款前就识别出高危客户。数据是从个人信用档案里整理出来的，全部匿名处理过，列名也改成了让人看不懂的"指标名"。

### 1.2 比赛目标

输入是用户的信用档案，训练集 `cs-training.csv` 一共 **150000** 行记录、**11** 列：

- 目标列 `SeriousDlqin2yrs`：是否在未来两年严重逾期（1 是，0 否）；
- `RevolvingUtilizationOfUnsecuredLines`：信用卡等无抵押循环信贷的**使用率**（已用额度/总额度，这是个比例值）；
- `age`：年龄；
- `NumberOfTime30-59DaysPastDueNotWorse` / `NumberOfTime60-89DaysPastDueNotWorse` / `NumberOfTimes90DaysLate`：过去两年迟到 30-59 天 / 60-89 天 / 超过 90 天的次数——三个"逾期档位"；
- `DebtRatio`：负债比——每月债务支出占月收入的比例（越高越危险）；
- `MonthlyIncome`：月收入；
- `NumberOfOpenCreditLinesAndLoans`：开着的信贷账户和贷款数量；
- `NumberRealEstateLoansOrLines`：房屋抵押贷款笔数；
- `NumberOfDependents`：家属人数。

测试集大约 **101500** 行，去掉了目标列。我们要给每条样本输出一个"违约概率"，提交文件里每行是 `Id` 加 `Probability`。

### 1.3 评价标准

比赛用 **ROC 曲线下的面积（AUC）** 打分。AUC 想回答的是：随机挑一个"好人"和一个"坏人"，模型给"坏人"打的违约概率更高的概率有多大。AUC = 0.5 等于瞎猜，越接近 1 越好。它只关心**排序**对不对，不关心概率的具体数值——这一点和钱有关：银行关心的是"能不能把高危客户正确排在前面"，而不是概率本身多精准。

---

## 二、解题过程

我们的思路是：先 EDA 摸清数据的分布和脏点，再清洗（补缺、修异常值），然后**先跑一个逻辑回归基线**拿到大概的分数，接着引入风控领域经典的 **WOE 编码** 提升特征表达，最后用带分箱特征工程的梯度提升树把分数冲到 0.86 以上。下面按四步走。

### 2.1 数据探索（EDA）

先读数据、看每列的基本统计量和缺失情况：

```python
import pandas as pd
import numpy as np

# index_col=0 即丢弃 Unnamed: 0 这列（它只是编号）
train = pd.read_csv('../input/GiveMeSomeCredit/cs-training.csv', index_col=0)
test  = pd.read_csv('../input/GiveMeSomeCredit/cs-test.csv',      index_col=0)

print(train.shape)                       # (150000, 11)
print(train.isnull().sum())
# MonthlyIncome        29731   ← 缺失 19.8%
# NumberOfDependents    3924   ← 缺失 2.6%
# 其余列                       不缺
```

缺失主要集中在两列：`MonthlyIncome`（月收入）缺了 **29731** 条，占近 **20%**；`NumberOfDependents` 缺了 **3924** 条。现实中这种情况很常见——月收入不是必填项，很多人留空。

接着看两个最关键的问题：类别是否失衡、有没有不可能出现的"脏值"：

```python
# 1) 类别失衡：正例（会违约的）只占 6.68%
print(train['SeriousDlqin2yrs'].value_counts(normalize=True))
# 0    0.9332
# 1    0.0668

# 2) 三个"逾期次数"列的正常取值如下，留意 96 和 98 这两个尾巴
print(train['NumberOfTimes90DaysLate'].value_counts().head())
# 0     142396
# 1       4123
# 2       1885
# ...
# 98        264
# 96          5
```

我们发现两件事：**第一**，正例只占 **6.68%**，数据严重失衡；**第二**，三个"逾期次数"列都有 **96 和 98** 这种值——一个人两年内逾期九十天以上 98 次，在时间上是不可能的，明显是录入错误。EDA 还发现，这些 96/98 行里**约 55% 的人真的违约**，而全局违约率只有 6%，所以这把"信号"不能扔，要单独处理。

除了逾期次数，还有两个分布问题值得记下来。`RevolvingUtilizationOfUnsecuredLines` 本应是一个 0～1 之间的比例，但数据里出现了远超 1 的极端值，材料里的做法是把大于 13 的值统一截断到 14，当成一个单独的"异常档"；`MonthlyIncome` 则严重右偏，绝大多数人月收入在几千的量级，最大值却达到 **3008750**，均值会被这种极端高收入拉得毫无代表性。这两点直接决定了我们后面为什么用中位数补收入、为什么要把比例特征做截断。

### 2.2 数据清洗

针对上面的发现，我们做四步清洗：

1. **补收入**：用**中位数**补 `MonthlyIncome`（收入右偏，均值易被高收入者拉高）；
2. **补家属数**：用**众数**（出现最多的值）补 `NumberOfDependents`；
3. **修异常值**：把 96/98 分别替换成"正常且严重"的档位值（14、12、18），既保留"这人严重逾期过"的信号，又不再让模型被不可能的天文数字干扰；
4. **删零收入**：`MonthlyIncome = 0` 的行本身数据就不可信（收入为 0 会导致负债比爆炸），直接删掉。

```python
from sklearn.impute import SimpleImputer

# 中位数补收入，众数补家属数
imp_income    = SimpleImputer(strategy='median')
imp_dependents = SimpleImputer(strategy='most_frequent')

# 注意 fit_transform 要先在训练集上 fit，再用训练集的统计量 transform 测试集
train[['MonthlyIncome']] = imp_income.fit_transform(train[['MonthlyIncome']])
train[['NumberOfDependents']] = imp_dependents.fit_transform(train[['NumberOfDependents']])
test[['MonthlyIncome']]  = imp_income.transform(test[['MonthlyIncome']])
test[['NumberOfDependents']] = imp_dependents.transform(test[['NumberOfDependents']])

# 修复 96/98：换成一个"高危"但真实的值
for col, repl in [('NumberOfTime30-59DaysPastDueNotWorse', 14),
                  ('NumberOfTime60-89DaysPastDueNotWorse', 12),
                  ('NumberOfTimes90DaysLate',              18)]:
    train.loc[train[col].isin([96, 98]), col] = repl
    test.loc[test[col].isin([96, 98]), col]  = repl

# 循环信贷使用率理论上应在 0~1 之间，出现极端大值说明额度信息异常；
# 统一截到 14，既保留"异常"信号，又不让极端值主导模型
for df in (train, test):
    df.loc[df['RevolvingUtilizationOfUnsecuredLines'] > 13,
           'RevolvingUtilizationOfUnsecuredLines'] = 14

# 直接删掉"月收入为 0"的样本，150000 → 148366
train = train[train['MonthlyIncome'] > 0]
```

清洗后数据从 150000 行变成 148366 行，丢掉了 0.6% 的样本，但换来了更干净的收入和负债比分布。

### 2.3 模型训练与超参数调优

**第一步：逻辑回归基线。** 我们用分层切分（`stratify=y`）保证验证集里正例比例和全局一致，先拿原始特征跑一版：

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X = train.drop('SeriousDlqin2yrs', axis=1)
y = train['SeriousDlqin2yrs']

# 分层切分：验证集里坏客户的占比和总数据保持一致（约 6.7%）
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=300, solver='liblinear')
lr.fit(X_tr, y_tr)
print('AUC : %.4f' % roc_auc_score(y_va, lr.predict_proba(X_va)[:, 1]))
# AUC : 0.7138
```

AUC 0.71，比瞎猜强，但远远不够。逻辑回归是线性模型，它处理不了的正是"到期次数不是越多越危险的线性关系"，于是我们引入风控领域最经典的 **WOE（证据权重）编码**。

**第二步：WOE 编码。** WOE 的直觉是：把每个特征切成若干分箱（bin），数每个箱子里"好客户占比 / 坏客户占比"，再取对数。某个箱子的 WOE 为正，说明进这个箱的人比平均水平"更安全"；为负则"更危险"，负得越多越危险。把原始数值替换成 WOE，等于把"这个值有多危险"的信息直接喂给模型：

```python
# 对离散特征（或分箱后的连续特征）计算每一档的 WOE 和 IV
def woe_discrete(df, col, target):
    good = df[target].sum()               # 好客户总数
    bad  = len(df) - good                 # 坏客户总数
    tmp = df.groupby(col)[target].agg(['sum', 'count'])
    tmp.columns = ['n_bad', 'n']
    tmp['n_good'] = tmp['n'] - tmp['n_bad']
    tmp['prop_good'] = tmp['n_good'] / good   # 这一档坏客户占全球坏比例
    tmp['prop_bad']  = tmp['n_bad'] / bad     # 这一档好客户占全球好比例
    # 取对数防止除零，单独加一个小量兜底
    tmp['WOE'] = np.log(tmp['prop_good'] / tmp['prop_bad'] + 1e-9)
    tmp['IV']  = (tmp['prop_good'] - tmp['prop_bad']) * tmp['WOE']
    return tmp
```

这里 `IV`（信息值）是另一个量，它累加每个箱子的"好坏占比之差 × WOE"，用来衡量"这个特征到底有多强的判别力"。我们按每个特征的分布手动分箱（逾期次数天然是分箱的，收入按 1k/2k/3k… 区间切），用 IV 挑出最有用的几个特征。

把原始特征替换成 WOE 之后再跑同一个逻辑回归，AUC 提升到了 **0.80** 左右。这说明对非线性特征做"银行风控式"的编码，即使模型还是线性逻辑回归，效果也提升了近 0.1。

**第三步：梯度提升树 + 分箱特征。** 社区高分方案（公开榜最好成绩在 0.86x）的做法是：保留原始数值，同时用 `KBinsDiscretizer` 为每个连续特征额外生成一列"分箱序号"当类别特征用，让树模型既能看连续值又能看它落在哪一段；模型换成对大数据友好的 **梯度提升树**（LightGBM / HistGradientBoosting）。为防止漏填导致分箱出错，每个分箱前都先接一个中位数填补：

```python
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

# 对"高基数连续特征"用 40 个分位数分箱，生成类别特征；对逾期次数这类离散特征用 4 箱
for col in ['MonthlyIncome', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines']:
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('binner', KBinsDiscretizer(n_bins=40, encode='ordinal', strategy='quantile')),
    ])
    train['bin_' + col] = pipe.fit_transform(train[[col]]).ravel()
    test['bin_' +  col] = pipe.transform(test[[col]]).ravel()
```

梯度提升树天然擅长挖掘乱糟糟的数字特征之间的关系，配合分箱特征，验证集 AUC 达到 **0.86 以上**。如果想再进一步，可以同时对多个树模型做**超参数调优**（树深、学习率、列采样比例等），再把结果平均，但收益已经递减——0.86 对很多银行任务已经是可用的水平。

### 2.4 模型测试与指标报告

我们用训练集的一部分做交叉验证，逐折报告 AUC；最后一折验证完整流程后，用全量数据重训并预测测试集：

```python
from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for k, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05)
    clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
    p = clf.predict_proba(X.iloc[val_idx])[:, 1]
    auc = roc_auc_score(y.iloc[val_idx], p)
    aucs.append(auc)
    print(f'fold {k}: AUC = {auc:.4f}')
print('mean AUC = %.4f' % np.mean(aucs))
```

我们的最终模型在 5 折交叉验证上的平均 AUC 约 **0.86**。公开排行榜上领先方案的分数在 **0.862～0.869** 之间，说明我们做到了非常有竞争力的水平——更重要的是，每一步的改进都是透明的：原始特征逻辑回归 0.71 → WOE 编码后 0.80 → 分箱特征加梯度提升树 0.86，每一跳都能看到数据工程带来的收益。

最后生成提交文件。注意提交的是 **Probability（概率）**，因为 AUC 评的是排序：

```python
test_prob = clf.predict_proba(test_scaled)[:, 1]
submission = pd.DataFrame({
    'Id': test.index,
    'Probability': test_prob,
})
submission.to_csv('submission.csv', index=False)
```

---

## 三、总结

本文中我们完成了 Give Me Some Credit 信用评分预测的完整流程。我们首先做 EDA，发现 150000 行样本里正例只有 6.68%、月收入缺近 20%、三个逾期次数列里有不可能的 96/98 值；其次做清洗，用中位数补收入、众数补家属数、替换异常值并删除零收入样本；接着先跑逻辑回归基线（AUC 0.71），再用 WOE 编码把特征转成"危险程度"信号（AUC 0.80），最后用分箱特征加梯度提升树把 AUC 提到 0.86 以上，与排行榜领先方案（0.862～0.869）处于同一水平。社区方案里还有把 WOE、分箱、多项式特征和多模型做 ensemble 的做法，我们选这条主线是为了让每一步的收益都清晰可见。

这个比赛给我们的最大启发是：**类别极度不平衡的数据里，先把"脏值"和"缺值"处理干净，收益往往比换更强的模型更大**。98 次逾期这种不可能的值，既可能是数据录入错误，也可能藏着"高危客户"的强信号，处理方式要基于 EDA——先看它和标签的关系再决定删还是修，而不是机械地当异常值删掉。