# 第5章 表格数据比赛详解：信用评分（Give Me Some Credit）

Give Me Some Credit 是 Kaggle 上经典的表格二分类比赛。银行根据一个人的过往信贷记录，估算TA未来两年内是否会严重逾期（违约），从而决定是否放贷、利率多少、额度多高。模型要解决的核心问题，就是**很精确地估计一个人的违约概率**。

与房价预测（回归）不同，这是一个**二分类**任务，评价指标是 AUC（排序指标），而且正例（会违约的人）只占 6.68%，**类别极度不平衡**。本章我们按 EDA → 清洗 → 特征工程 → 建模 → 提交 的流程，完整解析这场比赛，并说明每一步改进的收益来自哪里。

---

## 5.1 比赛背景与数据

### 比赛背景

银行决定要不要借钱给一个人，靠的是信用评分：根据过往信贷记录，算出未来会不会还不上钱的概率。如果模型把不该借的人放进来，银行承担坏账；如果把该借的人挡在门外，银行少赚利润。所以信用评分模型要解决的核心问题是：**很精确地估计一个人的违约概率**。

比赛用的是一份真实脱敏的信贷数据。目标标签叫 `SeriousDlqin2yrs`，全称 "Seriously Delinquent in 2 years"，意思是"未来两年内是否会严重逾期（违约）"。我们要预测每一笔贷款登记对象在未来两年出问题的概率，帮放贷机构在放款前就识别出高危客户。数据全部匿名处理，列名改成了"指标名"。

### 数据结构

训练集 `cs-training.csv` 一共 **150000** 行记录、**11** 列：

| 列名 | 含义 |
|------|------|
| `SeriousDlqin2yrs` | 目标列：未来两年是否严重逾期（1 是，0 否） |
| `RevolvingUtilizationOfUnsecuredLines` | 信用卡等无抵押循环信贷使用率（已用额度/总额度） |
| `age` | 年龄 |
| `NumberOfTime30-59DaysPastDueNotWorse` | 过去两年迟到 30-59 天的次数 |
| `NumberOfTime60-89DaysPastDueNotWorse` | 过去两年迟到 60-89 天的次数 |
| `NumberOfTimes90DaysLate` | 过去两年迟到超过 90 天的次数 |
| `DebtRatio` | 负债比——每月债务支出占月收入比例 |
| `MonthlyIncome` | 月收入 |
| `NumberOfOpenCreditLinesAndLoans` | 开着的信贷账户和贷款数量 |
| `NumberRealEstateLoansOrLines` | 房屋抵押贷款笔数 |
| `NumberOfDependents` | 家属人数 |

测试集大约 **101500** 行，去掉了目标列。我们要给每条样本输出一个"违约概率"，提交文件里每行是 `Id` 加 `Probability`。

### 评价指标

比赛用 **ROC 曲线下的面积（AUC）** 打分。AUC 想回答的是：随机挑一个"好人"（未违约）和一个"坏人"（违约），模型给"坏人"打的违约概率更高的概率有多大。AUC = 0.5 等于瞎猜，越接近 1 越好。

AUC 只关心**排序**对不对，不关心概率的具体数值。这和银行的需求一致：银行关心的是"能不能把高危客户正确排在前面"，而不是概率本身多精准。所以提交时我们输出的是概率（0~1 之间的小数），而不是 0/1 标签。

---

## 5.2 探索性数据分析（EDA）

先读数据，看每列的基本统计量和缺失情况。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# index_col=0 即丢弃 Unnamed: 0 这列（它只是编号）
train = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv', index_col=0)
test  = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-test.csv',      index_col=0)

print("训练集形状:", train.shape)   # (150000, 11)
print("\n缺失值统计：")
print(train.isnull().sum())
```

典型输出：

```
MonthlyIncome        29731   ← 缺失约 20%
NumberOfDependents    3924   ← 缺失约 2.6%
其余列                   不缺
```

缺失主要集中在两列：`MonthlyIncome`（月收入）缺了 29731 条，占近 20%；`NumberOfDependents` 缺了 3924 条。现实中这种情况很常见——月收入不是必填项，很多人留空。

接着看两个最关键的问题：**类别是否失衡**、**有没有不可能出现的"脏值"**：

```python
# 1) 类别失衡：正例（会违约的）只占 6.68%
print(train['SeriousDlqin2yrs'].value_counts(normalize=True))
# 0    0.9332
# 1    0.0668

# 2) 三个"逾期次数"列的取值，留意 96 和 98 这两个尾巴
print(train['NumberOfTimes90DaysLate'].value_counts().head())
# 0     142396
# 1       4123
# 2       1885
# ...
# 98        264
# 96          5
```

我们发现两件事：

**第一**，正例只占 **6.68%**，数据严重失衡。模型如果全猜"不违约"，准确率就有 93.32%，但实际上什么都没学到。这时候不能只看准确率，必须用 AUC 这种排序指标，或者在建模时处理类别不平衡。

**第二**，三个"逾期次数"列都有 **96 和 98** 这种值。一个人两年内逾期九十天以上 98 次，在时间上是不可能的，明显是录入错误。但仔细观察会发现，这些 96/98 行里约 55% 的人真的违约，而全局违约率只有 6%。所以这把"信号"不能扔，要单独处理——既保留"这人严重逾期过"的信号，又不再让模型被不可能的天文数字干扰。

除了逾期次数，还有两个分布问题值得记下来：

- `RevolvingUtilizationOfUnsecuredLines` 本应是一个 0~1 之间的比例，但数据里出现了远超 1 的极端值，需要做截断处理。
- `MonthlyIncome` 严重右偏，绝大多数人月收入在几千的量级，最大值却达到 3008750，均值会被这种极端高收入拉得毫无代表性，所以填充收入要用中位数，不用均值。

---

## 5.3 数据清洗

针对 EDA 发现的问题，我们做四步清洗：

**1. 补收入**：用**中位数**补 `MonthlyIncome`（收入右偏，均值易被高收入者拉高）。

**2. 补家属数**：用**众数**（出现最多的值）补 `NumberOfDependents`。

**3. 修异常值**：把 96/98 分别替换成"正常且严重"的档位值（14、12、18），既保留"这人严重逾期过"的信号，又不再让模型被不可能的天文数字干扰。

**4. 删零收入**：`MonthlyIncome = 0` 的行本身数据不可信（收入为 0 会导致负债比爆炸），直接删掉。

以下是完整清洗代码：

```python
from sklearn.impute import SimpleImputer

# 中位数补收入，众数补家属数
imp_income    = SimpleImputer(strategy='median')
imp_dependents = SimpleImputer(strategy='most_frequent')

# 注意：fit_transform 要先在训练集上 fit，再用训练集的统计量 transform 测试集，
# 否则会发生信息泄漏（测试集的统计信息偷跑进训练过程）
train[['MonthlyIncome']] = imp_income.fit_transform(train[['MonthlyIncome']])
train[['NumberOfDependents']] = imp_dependents.fit_transform(train[['NumberOfDependents']])
test[['MonthlyIncome']]  = imp_income.transform(test[['MonthlyIncome']])
test[['NumberOfDependents']] = imp_dependents.transform(test[['NumberOfDependents']])

# 修复 96/98：换成一个"高危但真实"的值
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

# 直接删掉"月收入为 0"的样本（150000 → 148366）
train = train[train['MonthlyIncome'] > 0]

print("清洗后训练集形状:", train.shape)
```

清洗后数据从 150000 行变成 148366 行，丢掉了 0.6% 的样本，但换来了更干净的收入和负债比分布。

> **注意**：fit 只能在训练集上做，测试集只能 transform。如果在合并后的数据上统一 fit 中位数，测试集的中位数会"泄漏"进训练过程，导致验证分数虚高。

---

## 5.4 特征工程

信用评分是金融风控领域的经典任务，有该领域特有的特征工程方法：**WOE 编码（证据权重）** 和 **分箱特征**。

### WOE 编码：把特征转成"危险程度"

WOE 的直觉是：把每个特征切成若干分箱（bin），数每个箱子里"好客户占比 / 坏客户占比"，再取对数。某个箱子的 WOE 为正，说明进这个箱的人比平均水平"更安全"；为负则"更危险"，负得越多越危险。把原始数值替换成 WOE，等于把"这个值有多危险"的信息直接喂给模型。

```python
def woe_discrete(df, col, target):
    good = df[target].sum()               # 好客户总数
    bad  = len(df) - good                 # 坏客户总数
    tmp = df.groupby(col)[target].agg(['sum', 'count'])
    tmp.columns = ['n_bad', 'n']
    tmp['n_good'] = tmp['n'] - tmp['n_bad']
    tmp['prop_good'] = tmp['n_good'] / good   # 这一档好客户占全局好客户的比例
    tmp['prop_bad']  = tmp['n_bad'] / bad     # 这一档坏客户占全局坏客户的比例
    tmp['WOE'] = np.log(tmp['prop_good'] / tmp['prop_bad'] + 1e-9)
    tmp['IV']  = (tmp['prop_good'] - tmp['prop_bad']) * tmp['WOE']
    return tmp
```

这里 `IV`（信息值）累加每个箱子的"好坏占比之差 × WOE"，用来衡量"这个特征到底有多强的判别力"。IV 越大，特征越有用。

我们按每个特征的分布手动分箱（逾期次数天然是分箱的，收入按 1k/2k/3k… 区间切），用 IV 挑出最有用的几个特征。把原始特征替换成 WOE 之后，即使模型还是线性逻辑回归，效果也能明显提升。

### 分箱特征：让树模型同时看连续值和区间

梯度提升树天然擅长挖掘数字特征之间的关系，但如果我们额外生成"分箱序号"当类别特征，模型既能看连续值，又能知道某个值落在哪一段（比如收入落在哪个区间），这样能捕捉更多非线性关系。

```python
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 对连续特征用 40 个分位数分箱，生成类别特征
for col in ['MonthlyIncome', 'DebtRatio', 'RevolvingUtilizationOfUnsecuredLines']:
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('binner', KBinsDiscretizer(n_bins=40, encode='ordinal', strategy='quantile')),
    ])
    train['bin_' + col] = pipe.fit_transform(train[[col]]).ravel()
    test['bin_' +  col] = pipe.transform(test[[col]]).ravel()
```

为防止漏填导致分箱出错，每个分箱前都先接一个中位数填补。

---

## 5.5 模型训练与调优

我们按"先简单后复杂"的思路，逐步提升模型效果。

### 第一步：逻辑回归基线

用分层切分（`stratify=y`）保证验证集里正例比例和全局一致（约 6.7%），先拿原始特征跑一版：

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X = train.drop('SeriousDlqin2yrs', axis=1)
y = train['SeriousDlqin2yrs']

# 分层切分：验证集里坏客户的占比和总数据保持一致
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=300, solver='liblinear')
lr.fit(X_tr, y_tr)
print('AUC : %.4f' % roc_auc_score(y_va, lr.predict_proba(X_va)[:, 1]))
# 输出约：AUC : 0.7138
```

AUC 0.71，比瞎猜强，但远远不够。逻辑回归是线性模型，它处理不了"逾期次数不是越多越危险的线性关系"这种非线性模式。

### 第二步：WOE 编码后的逻辑回归

把原始特征替换成 WOE 编码后，再跑同一个逻辑回归，AUC 提升到了 **0.80** 左右。这说明：对非线性特征做"银行风控式"的编码，即使模型还是线性逻辑回归，效果也提升了近 0.1。

### 第三步：梯度提升树 + 分箱特征

社区高分方案的做法是：保留原始数值，同时用 `KBinsDiscretizer` 为每个连续特征额外生成一列"分箱序号"当类别特征用；模型换成对大数据友好的梯度提升树（如 `HistGradientBoostingClassifier` 或 LightGBM）。

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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

梯度提升树配合分箱特征，验证集 AUC 达到 **0.86 以上**。如果想再进一步，可以对多个树模型做超参数调优（树深、学习率、列采样比例等），再把结果平均，但收益已经递减——0.86 对很多银行任务已经是可用的水平。

**每一步的收益清晰可见**：
- 原始特征 + 逻辑回归：AUC 约 0.71
- WOE 编码 + 逻辑回归：AUC 约 0.80
- 分箱特征 + 梯度提升树：AUC 0.86 以上

---

## 5.6 模型测试与提交

5 折交叉验证平均 AUC 约 0.86，与公开排行榜领先方案（0.862～0.869）处于同一水平。最后用全量数据重训，预测测试集并生成提交文件。

注意提交的是 **Probability（概率）**，因为 AUC 评的是排序：

```python
# 用全量训练数据重训最终模型
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05)
clf.fit(X, y)

# 对测试集预测违约概率
test_prob = clf.predict_proba(test)[:, 1]

submission = pd.DataFrame({
    'Id': test.index,
    'Probability': test_prob,
})
submission.to_csv('submission.csv', index=False)

# 核对清单
assert len(submission) == len(test), "提交行数与测试集不一致"
assert list(submission.columns) == ['Id', 'Probability'], "列名不一致"
assert submission['Probability'].isnull().sum() == 0, "预测概率包含缺失值"
assert ((submission['Probability'] >= 0) & (submission['Probability'] <= 1)).all(), "概率必须在 0~1 之间"
print("提交文件已生成：submission.csv")
print(submission.head())
```

**核对说明**：
- 行数必须与测试集一致，不能多也不能少。
- 列名必须是 `Id` 和 `Probability`，与 `sample_submission.csv` 严格匹配。
- 预测概率必须在 0~1 之间，不能出现负数或大于 1 的值。
- 不能有缺失值。

把 `submission.csv` 上传到 Kaggle 比赛页面，你就能在排行榜上看到自己的 AUC 分数了。

---

## 5.7 小结

本章我们以 Give Me Some Credit 信用评分为例，完整解析了一场表格二分类比赛：

1. **比赛背景**：预测未来两年是否严重逾期，150000 行样本、11 个特征，评价指标为 AUC（排序指标）。
2. **EDA**：发现正例只有 6.68%（严重类别不平衡），`MonthlyIncome` 缺失近 20%，三个逾期次数列里有不可能的 96/98 值，循环信贷使用率有远超 1 的极端值。
3. **数据清洗**：中位数补收入、众数补家属数、替换 96/98 异常值、截断使用率、删除零收入样本。注意 fit 只在训练集上做，避免信息泄漏。
4. **特征工程**：引入 WOE 编码把特征转成"危险程度"信号，IV 用于特征筛选；用 `KBinsDiscretizer` 生成分箱特征，让树模型同时看连续值和区间。
5. **模型训练**：先跑逻辑回归基线（AUC 0.71），WOE 编码后提升到 0.80，最后用分箱特征 + 梯度提升树达到 0.86 以上。
6. **模型测试与提交**：输出违约概率（0~1 之间），按 `sample_submission` 格式生成 `submission.csv`，上传前核对行数、列名、概率范围、缺失值。

这个比赛给我们的最大启发是：**类别极度不平衡的数据里，先把"脏值"和"缺值"处理干净，收益往往比换更强的模型更大**。98 次逾期这种不可能的值，既可能是数据录入错误，也可能藏着"高危客户"的强信号，处理方式要基于 EDA——先看它和标签的关系再决定删还是修，而不是机械地当异常值删掉。

下一章我们将详解另一个表格比赛：贷款审批（Loan Approval），看看在另一个表格分类场景下，上述流程有什么共通之处和不同之处。
