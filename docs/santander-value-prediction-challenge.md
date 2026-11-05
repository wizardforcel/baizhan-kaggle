# Santander 客户价值预测挑战（Santander Value Prediction Challenge）

## 一、比赛简介

### 1.1 比赛背景

Santander 是一家总部位于西班牙的大型银行。这个比赛由 Santander 和 Kaggle 共同举办，目的是让参赛者根据客户过去的行为数据，预测每一位客户会给银行带来多少「价值」。所谓客户价值，可以理解为这位客户未来一段时间内可能为银行创造的收入或利润。银行如果能提前判断哪些客户价值高，就可以把有限的营销资源放在更值得维护的客户身上。

比赛数据最大的特点是「匿名化」和「高维稀疏」。训练集有 4993 行、4459 列，其中一列是客户编号 `ID`，一列是要预测的目标 `target`，剩下 4991 列都是经过脱敏处理的数值特征，列名像 `48df886f9`、`0deb4b6a8` 这样的随机字符串，我们无法知道每一列在真实业务里代表什么。训练集里绝大多数格子是 0，只有很少一部分是非零值，所以这是一份典型的**稀疏表格数据**。

另一个难点是样本量小、特征多。4993 行样本要对应将近 5000 个特征，特征数量几乎和样本数量一样多，模型非常容易过拟合。比赛主办方还故意在数据里留下了一些「泄漏」——某些特征列的取值和 `target` 高度相似甚至完全相同。能不能发现并利用这些泄漏，成了比赛里拉开差距的关键。当然，如果只是机械地利用泄漏而不理解数据，也很容易在测试集上翻车。

### 1.2 比赛目标

比赛要求我们根据训练集学习一个回归模型，然后对测试集里的每一位客户预测 `target`。

- 训练集：`ID`、`target` 以及 4991 个匿名数值特征；
- 测试集：`ID` 以及同样的 4991 个匿名数值特征，没有 `target`；
- 提交文件：两列，`ID` 和预测出来的 `target`。

目标值 `target` 是一个非负数值。从材料里的统计看，训练集 `target` 的最大值达到 2,000,000，而且分布很不均匀：大量客户的价值接近 0，少数客户的价值非常高。这种「长尾分布」意味着我们不能直接让模型去拟合原始数值，否则模型会被几个超大值带偏。

### 1.3 评价标准

比赛使用 **RMSLE**（Root Mean Squared Logarithmic Error，均方根对数误差）作为评价指标。它的计算方法是：

```text
RMSLE = sqrt( mean( (log1p(预测值) - log1p(真实值))^2 ) )
```

其中 `log1p(x)` 表示 `log(1 + x)`。这个指标有两个特点：

1. **先取对数再算误差**。因为对数会把大数值压缩，所以 RMSLE 对「预测值比真实值大很多」和「预测值比真实值小很多」相对更公平，不会像普通均方误差那样被少数超大值完全支配。
2. **分数越低越好**。RMSLE 等于 0 表示完美预测，数值越大表示误差越大。

也正因为评价指标是 RMSLE，材料里几乎所有方案都会先对 `target` 做 `log1p` 变换，再训练回归模型，最后用 `expm1` 把预测值变回原来的尺度。这是一个非常自然、也非常重要的技巧：**让训练时优化的目标和比赛评价的尺度保持一致**。

---

## 二、解题过程

整场比赛的主线可以概括为：**先做 EDA 看清稀疏结构，再删掉常量列和重复列，接着用对数变换训练 LightGBM，然后通过交叉验证检查 RMSLE，最后生成提交文件。** 材料里还有一条「泄漏特征」的支线，我们也会在数据清洗和模型部分讲到。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，确认数据的形状、列名和缺失情况：

```python
import numpy as np
import pandas as pd

# 读取训练集和测试集
train = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")
test = pd.read_csv("../input/santander-value-prediction-challenge/test.csv")

# 打印形状：训练集 4993 行、4459 列；测试集 49342 行、4992 列
print(train.shape, test.shape)

# 看前几行，确认列名都是匿名哈希，且大量取值为 0
print(train.head())

# 统计每一列有多少个不同的取值
unique_df = train.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]

# 只保留「只有一个取值」的列，这些列没有任何信息量
constant_df = unique_df[unique_df["unique_count"] == 1]
print("常量列数量:", constant_df.shape)
```

打印结果告诉我们：训练集是 `(4993, 4459)`，测试集是 `(49342, 4992)`。测试集比训练集大了将近十倍，但特征列是一样的。`train.head()` 里能看到大量 0，说明数据确实非常稀疏。`nunique()` 的统计显示有 **256 个常量列**——这些列在训练集里所有行的取值都一样，对预测没有任何帮助，后面可以直接删掉。

其次，我们观察目标变量的分布：

```python
import matplotlib.pyplot as plt

# 画出 target 的散点图，看看有没有明显的极端值
plt.figure(figsize=(8, 6))
plt.scatter(range(train.shape[0]), np.sort(train["target"].values))
plt.xlabel("index")
plt.ylabel("Target")
plt.title("Distribution of Target")
plt.show()

# 打印 target 的基本统计量
print(train["target"].describe())
```

从散点图可以看到，绝大多数样本的 `target` 都挤在靠近 0 的位置，只有极少数样本的值非常高，最高达到 2,000,000。这就是前面说的长尾分布。我们还注意到，`target` 里存在 0 值，而 RMSLE 里的 `log1p` 正好可以处理 0：`log1p(0) = 0`，不会出现 `log(0)` 的负无穷问题。

接着，我们检查缺失值和重复列。材料里的方案指出，数据本身没有太多常规意义上的 `NaN`，但因为大量 0 的存在，真正「有效」的信息很少。我们还用 `duplicated()` 检查了列名和列值是否有重复：

```python
# 检查每一列是否有缺失值
print("有缺失值的列数:", train.isnull().sum().sum())

# 检查列名是否重复
print("重复列名数量:", train.columns.duplicated().sum())

# 对每一列做「取值是否完全相同」的检查
# 这里用转置后的 duplicated()，可以找出内容完全一样的列
duplicated_cols = train.T.duplicated()
print("内容重复的列数:", duplicated_cols.sum())
```

这一步的结论是：数据没有明显的 `NaN`，但存在常量列和可能重复的列。对高维稀疏数据来说，删掉这些列不仅能减少计算量，还能降低模型过拟合的风险。

### 2.2 数据清洗

我们首先删掉常量列。做法很简单：从 `nunique()` 的结果里挑出 `unique_count == 1` 的列名，然后从训练集和测试集里一起删除。注意测试集也要删同样的列，否则训练和预测的特征维度对不上。

```python
# constant_df 里保存了所有常量列的列名
const_cols = constant_df["col_name"].tolist()

# 从训练集里删除常量列，同时去掉 ID 和 target
X = train.drop(const_cols + ["ID", "target"], axis=1)

# 测试集只去掉常量列和 ID，保留全部特征
X_test = test.drop(const_cols + ["ID"], axis=1)

# 目标值取对数：RMSLE 的评价对象就是 log1p(target)
y = np.log1p(train["target"].values)

print("清洗后训练特征形状:", X.shape)
print("清洗后测试特征形状:", X_test.shape)
```

删完之后，特征数量从 4991 降到了 4735 左右（4991 减去 256 个常量列）。这一步看起来只是「删无用列」，但对这个比赛很重要：样本只有 4993 行，每多一个无信息列，模型就多一个可以「死记硬背」的机会。

其次，我们处理 0 值。这里的 0 不一定代表「缺失」，更可能是「这位客户没有这项行为」。材料里的一个方案把 0 替换成 `NaN`，让 LightGBM 自己决定怎么处理缺失；另一个方案则保留 0 不动。我们选择保留 0，因为 LightGBM 对 0 和缺失值的处理方式不同，贸然替换可能会改变特征的分布。对树模型来说，0 本身就是一个有意义的取值。

接着，我们做了一件这个比赛特有的事：**寻找泄漏列**。材料里有一段代码，逐列比较特征值和 `target`：如果某一列里有很多行的取值和 `target` 完全相同，或者相对误差小于 5%，这一列就很可能是泄漏列。

```python
# 候选特征列：去掉 ID 和 target
cols = [c for c in train.columns if c not in ["ID", "target"]]

leak_cols = []
for c in cols:
    # 统计这一列有多少行和 target 完全相等
    exact_match = np.sum((train[c] == train["target"]).astype(int))
    # 统计这一列有多少行和 target 的相对误差小于 5%
    close_match = np.sum(
        ((np.abs(train[c] - train["target"]) / train["target"]) < 0.05).astype(int)
    )
    # 两个条件同时满足时，认为这一列是泄漏列
    if exact_match > 25 and close_match > 3500:
        leak_cols.append(c)

print("找到的泄漏列数量:", len(leak_cols))
```

材料里的这段筛选找到了 **73 个泄漏列**。用这 73 列单独训练模型，交叉验证 RMSLE 能从 1.4 左右降到 0.82 左右，提升非常明显。不过我们在主线里仍然先讲「正常特征」的做法，因为泄漏列是比赛数据的一个特殊现象，理解正常流程更重要；泄漏列的结果会在 2.4 节作为对比出现。

然后，我们对泄漏列做行级聚合，构造几个统计特征。思路是：既然这些列和客户价值有关，那么每一行在这些列上的非零均值、最大值、最小值、零的个数，也应该能反映客户的价值水平。

```python
# 只保留泄漏列，做行级统计
train_leak = train[leak_cols].copy()
test_leak = test[leak_cols].copy()

for df in [train_leak, test_leak]:
    # 非零值的均值：排除 0 之后，这一行「有效行为」的平均强度
    df["nz_mean"] = df.apply(lambda row: row[row != 0].mean(), axis=1)
    # 非零值的最大值和最小值
    df["nz_max"] = df.apply(lambda row: row[row != 0].max(), axis=1)
    df["nz_min"] = df.apply(lambda row: row[row != 0].min(), axis=1)
    # 这一行有多少个 0：0 越多，说明客户行为越少
    df["zero_count"] = df.apply(lambda row: len(row[row == 0]), axis=1)
    # 整行的均值、最大值、最小值
    df["row_mean"] = df.mean(axis=1)
    df["row_max"] = df.max(axis=1)
    df["row_min"] = df.min(axis=1)
```

这些聚合特征把「一行 73 个数字」压缩成了 7 个数字，既保留了客户之间的差异，又大幅降低了维度。对样本量小的比赛来说，这种「先聚合再建模」的做法往往比直接把几千列丢给模型更稳。

### 2.3 模型训练与超参数调优

我们选择 **LightGBM** 作为主模型。LightGBM 是一种梯度提升树（Gradient Boosting Decision Tree）实现：它一棵一棵地建决策树，每一棵新树都专门去修正前面所有树犯的错，最后把所有树的预测加起来。它对表格数据、稀疏数据的效果通常很好，训练速度也比传统 GBDT 快。

训练时我们做两件事：第一，把 `target` 取 `log1p`；第二，用 5 折交叉验证，每一折都在验证集上算 RMSE。因为标签已经取过对数，所以这里的 RMSE 其实就近似于 RMSLE。

```python
import lightgbm as lgb
from sklearn import model_selection

# LightGBM 的参数：学习率小一点、树多一些，通常更稳
params = {
    "objective": "regression",   # 回归任务
    "metric": "rmse",            # 验证指标用 RMSE（标签已取对数，等价于 RMSLE）
    "num_leaves": 30,            # 每棵树最多 30 个叶子，控制模型复杂度
    "learning_rate": 0.01,       # 学习率越小，每棵树贡献越小，需要更多树
    "bagging_fraction": 0.7,     # 每次迭代随机采样 70% 的行，防止过拟合
    "feature_fraction": 0.7,     # 每次迭代随机采样 70% 的列，防止过拟合
    "bagging_frequency": 5,      # 每 5 轮做一次行采样
    "bagging_seed": 2018,
    "verbosity": -1,
}

# 5 折交叉验证
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=99)
pred_test_full = 0

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 构造 LightGBM 的数据集
    lgtrain = lgb.Dataset(X_train, label=y_train)
    lgval = lgb.Dataset(X_val, label=y_val)

    # 训练，最多 1000 棵树；验证集 100 轮不提升就提前停止
    model = lgb.train(
        params,
        lgtrain,
        num_boost_round=1000,
        valid_sets=[lgval],
        early_stopping_rounds=100,
        verbose_eval=200,
    )

    # 用这一折的模型预测测试集，并累加
    pred_test_full += model.predict(X_test, num_iteration=model.best_iteration)

# 五折平均，再 expm1 变回原始尺度
pred_test_full /= 5.0
pred_test_full = np.expm1(pred_test_full)
```

这段代码里有几个关键点：

- `early_stopping_rounds=100`：如果验证集的误差连续 100 轮都不再下降，就停止训练，避免把噪声也学进去。
- `bagging_fraction` 和 `feature_fraction` 都设成 0.7：这是对付高维稀疏数据的常用手段，每次只用一部分行和一部分列，相当于让模型「不要总盯着同样的特征」。
- 五折的预测结果先取平均，再用 `expm1` 还原。因为训练时标签是 `log1p(target)`，模型输出也在对数尺度上，必须还原回来才能和真实值比较。

材料里还试过 XGBoost 和 CatBoost。XGBoost 和 LightGBM 都是梯度提升树，区别主要在树的生长方式和工程实现；CatBoost 则擅长处理类别特征，但这批特征全是匿名数值，CatBoost 的优势不明显。最终材料里的主线仍然是 LightGBM，因为它在稀疏数据上训练快、调参也相对直观。

### 2.4 模型测试与指标报告

我们首先看正常特征（不含泄漏列）的交叉验证结果。材料里 5 折的验证 RMSE 分别是 1.49247、1.48165、1.43397、1.42742、1.37178，最好的一折是 1.37178，最差的一折是 1.49247。换算成 RMSLE，大概在 **1.37 到 1.49** 之间。这个数字说明：只靠匿名特征做普通回归，模型能学到一些规律，但误差仍然很大，原因主要是样本太少、特征太稀疏。

```python
from sklearn.metrics import mean_squared_error

# 在每一折上单独计算 RMSE，看看波动有多大
fold_scores = []
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    lgtrain = lgb.Dataset(X_train, label=y_train)
    lgval = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        params, lgtrain, num_boost_round=1000,
        valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=False,
    )

    pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y_val, pred_val, squared=False)
    fold_scores.append(rmse)

print("每一折 RMSLE:", fold_scores)
print("平均 RMSLE:", np.mean(fold_scores))
```

其次，我们用前面找到的 73 个泄漏列加聚合特征，单独训练一个 LightGBM。材料里的结果非常夸张：验证 RMSE 降到了 **0.82 到 0.85** 左右，其中一折是 0.844159，另一折是 0.821953。这个提升不是模型变强了，而是数据里本来就藏着答案——某些特征列几乎就是 `target` 的「翻版」。

```python
# 只用泄漏列和聚合特征训练
leak_features = leak_cols + [
    "nz_mean", "nz_max", "nz_min", "zero_count",
    "row_mean", "row_max", "row_min",
]

X_leak = train_leak[leak_features]
X_test_leak = test_leak[leak_features]

# 同样的 5 折流程
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=99)
pred_leak = np.zeros(len(X_test_leak))

for train_index, val_index in kf.split(X_leak):
    X_train, X_val = X_leak.iloc[train_index], X_leak.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    lgtrain = lgb.Dataset(X_train, label=y_train)
    lgval = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        params, lgtrain, num_boost_round=3000,
        valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=False,
    )

    pred_leak += model.predict(X_test_leak, num_iteration=model.best_iteration)
    pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    print("验证 RMSLE:", mean_squared_error(y_val, pred_val, squared=False))

# 平均并还原尺度
pred_leak /= 5.0
pred_leak = np.expm1(pred_leak)
```

最后，我们生成提交文件。格式必须是两列：`ID` 和 `target`，行数必须和测试集一致（49342 行）。生成之后我们还会检查一下前几行、检查有没有空值、检查预测值是否出现负数——RMSLE 虽然允许 0，但负数没有业务意义，出现负数通常说明模型或还原步骤出了问题。

```python
# 用测试集的 ID 和预测值拼成提交表
submission = pd.DataFrame({
    "ID": test["ID"].values,
    "target": pred_test_full,
})

# 保存为 CSV，index=False 表示不要把行号写进去
submission.to_csv("submission.csv", index=False)

# 简单自检：行数、列名、是否有空值、最小值
print("提交文件形状:", submission.shape)
print("列名:", submission.columns.tolist())
print("空值数量:", submission.isnull().sum().sum())
print("预测最小值:", submission["target"].min())
```

我们在报告指标时会把两个结果分开说：**正常特征方案的交叉验证 RMSLE 约 1.37～1.49，泄漏特征方案约 0.82～0.85。** 前者代表模型真实的学习能力，后者代表数据泄漏带来的「捷径」。把这两个数字放在一起，才能客观地说明这个比赛的成绩是怎么来的。

---

## 三、总结

本文中我们完成了 Santander 客户价值预测挑战的完整流程。我们首先用 EDA 摸清了数据：训练集只有 4993 行，却有 4459 列，绝大多数取值是 0，目标值呈长尾分布，最高达到 2,000,000。其次我们做了数据清洗：删掉 256 个常量列，保留 0 值不动，并通过「特征值与 target 完全相等或相对误差小于 5%」的规则找到了 73 个泄漏列。接着我们对泄漏列做行级聚合，构造非零均值、最大值、最小值、零个数等统计特征，把高维稀疏的原始数据压缩成更适合小样本建模的形式。然后我们用 `log1p` 变换目标值，以 5 折交叉验证训练 LightGBM，并用早停、行采样、列采样控制过拟合。最后我们报告了两组指标：正常特征的交叉验证 RMSLE 约 1.37～1.49，泄漏特征方案约 0.82～0.85，并按 `ID, target` 的格式生成了 49342 行的提交文件。

这个比赛给我们最大的启发是：**面对高维稀疏、样本又少的数据，清洗和特征压缩比换更复杂的模型更重要。** 删常量列、做行级聚合、控制树的数量，都是在跟过拟合作斗争；而泄漏列的发现则提醒我们，比赛数据里的「异常规律」既可能是机会，也可能是陷阱，必须先用交叉验证确认它到底能不能稳定提升分数，再决定要不要把它放进最终方案。
