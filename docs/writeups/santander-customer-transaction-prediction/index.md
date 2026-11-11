# Santander 客户交易预测（Santander Customer Transaction Prediction）

## 一、比赛简介

### 1.1 比赛背景

Santander 银行想预测：哪些客户在未来会完成一笔特定交易。比赛给每个客户 200 个匿名数值特征，特征名统一叫 `var_0` 到 `var_199`，没有业务含义，也不告诉你哪个变量代表余额、收入还是交易次数。训练集有 20 万行，测试集也有 20 万行。目标 `target` 是 0/1：1 表示客户后来完成了这笔交易。

这场比赛在 Kaggle 历史上非常有名，原因有两个。第一，参赛人数超过 9000，竞争极其激烈。第二，测试集里混入了**人工合成的假样本**：20 万条测试数据中，大约 10 万条是真实客户，另外 10 万条是按训练集分布生成的假客户。主办方的目的是防止参赛者通过“猜测测试集分布”来刷分。谁能识别出真假测试样本，谁就能把特征工程做在正确的数据上。

另一个著名发现是所谓的 **magic（魔法特征）**。很长一段时间里，所有队伍都卡在 0.901 的 AUC 附近，普通的特征加减乘除、PCA、聚类、目标编码都没有明显效果。后来有人发现：对每个 `var_i`，统计它在“训练集 + 真实测试集”里每个取值出现了多少次，把这个次数作为新特征，AUC 就能明显提升。这个特征看起来平平无奇，却成了比赛的转折点。

### 1.2 比赛目标

训练集包含 `ID_code`、`target` 和 200 个 `var_i`；测试集包含 `ID_code` 和 200 个 `var_i`。我们要为测试集每个 `ID_code` 输出一个 0 到 1 之间的概率，表示该客户完成目标交易的可能性。提交文件只有两列：`ID_code` 和 `target`。

注意测试集里既有真客户也有假客户。假客户也要给预测值，否则提交行数不够；但在做特征工程和验证时，我们要尽量只使用真客户的信息，避免把人工生成的分布当成真实规律。

### 1.3 评价标准

比赛用 **ROC AUC** 评价，越高越好，0.5 相当于随机猜，1.0 是完美排序。AUC 只关心“正类样本的预测概率是否比负类高”，不关心概率的绝对值，所以提交概率即可，不需要先切成 0/1。

这个指标对类别不平衡很友好。比赛中正类比例不高，如果看准确率，全部预测 0 也能得到很高的数字；但 AUC 会惩罚这种没有排序能力的模型。我们后面所有验证、调参都以 AUC 为准。

---

## 二、解题过程

我们的主线是：先识别测试集里的真假样本，只在“训练集 + 真实测试集”上统计每个特征的取值频次，得到 200 个 count 特征；然后为每个原始特征单独训练一个小 LightGBM，输入只有“原始 var + 它的 count”，把 200 个模型的预测加起来作为最终分数。这个方案在材料里达到交叉验证 AUC 约 0.921、公开榜约 0.922。

### 2.1 数据探索（EDA）

我们首先读数据，确认形状、类型和目标分布。200 列浮点数占内存不小，所以读入时直接指定 `float32`。

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# 200 个特征统一用 float32，能把内存占用压到一半左右
dtypes = {f"var_{i}": "float32" for i in range(200)}
dtypes["target"] = "uint8"
dtypes["ID_code"] = "object"

train = pd.read_csv("../input/train.csv", dtype=dtypes)
test = pd.read_csv("../input/test.csv", dtype=dtypes)

print("train:", train.shape)   # (200000, 202)
print("test :", test.shape)    # (200000, 201)
print(train["target"].value_counts(normalize=True))
```

打印结果确认：训练集 20 万行、202 列（`ID_code`、`target` 加 200 个特征）；测试集 20 万行、201 列。目标不是 50/50，正类只占少数，所以后面不能只看准确率。

接着我们看单个特征的分布。它们大多近似正态，但每条样本的取值都有很多位小数，看起来像是连续变量。我们进一步统计“每个取值出现了多少次”，这一步是整场比赛的关键。

```python
import matplotlib.pyplot as plt

# 以 var_0 为例，看训练集和测试集的分布是否接近
plt.figure(figsize=(10, 4))
plt.hist(train["var_0"], bins=100, alpha=0.5, label="train", density=True)
plt.hist(test["var_0"], bins=100, alpha=0.5, label="test", density=True)
plt.legend()
plt.title("Distribution of var_0")
plt.show()

# 统计 var_0 每个取值出现的次数，看看有多少“只出现一次”的值
counts = train["var_0"].value_counts()
print("unique values:", len(counts))
print("values appearing once:", (counts == 1).sum())
print("max frequency:", counts.max())
```

单个特征有大量不同的取值，绝大多数值只出现一两次。这个“频次”本身在普通表格问题里通常被当成噪声，但在这场比赛中，它携带了数据生成过程留下的痕迹。材料里的分析认为，count 特征之所以有效，是因为正负类的数值在四舍五入/预处理上存在细微差异，频次把这种差异放大成了可学习的信号。我们不需要完全解释清楚原因，只要验证它确实能提高 AUC。

### 2.2 数据清洗

我们其次处理测试集里的假样本。判断方法很巧妙：对测试集的每一列，找出“在整个测试集中只出现一次”的值；如果某一行在任意一列上含有这种唯一值，就认为它是真实样本；如果一行在所有列上都没有唯一值，就认为它是合成样本。

```python
# 只取测试集的特征矩阵
test_values = test.drop(columns=["ID_code"]).values

unique_count = np.zeros_like(test_values)

# 对每一列，统计每个值出现的次数和第一次出现的位置
for feature in range(test_values.shape[1]):
    _, index_, count_ = np.unique(
        test_values[:, feature],
        return_counts=True,
        return_index=True,
    )
    # 只出现一次的值，把对应行的这一列标记为 1
    unique_count[index_[count_ == 1], feature] += 1

# 只要任意一列有唯一值，这一行就视为真实样本
real_mask = np.sum(unique_count, axis=1) > 0
real_indexes = np.argwhere(real_mask)[:, 0]
fake_indexes = np.argwhere(~real_mask)[:, 0]

print("real test rows:", len(real_indexes))  # 约 100000
print("fake test rows:", len(fake_indexes))  # 约 100000
```

这段逻辑的依据是：合成样本是按分布重采样出来的，取值容易和别的样本重复；真实样本是连续测量值，至少会在某一列上出现“全测试集唯一”的取值。材料里两个独立 notebook 都用了同一套判断，并得到“10 万真、10 万假”的结果，所以我们把它作为可靠的前置清洗步骤。

接下来构造 count 特征。注意统计范围只能是**训练集 + 真实测试集**，不能把 10 万假样本也算进去，否则频次会被人工生成的数据污染。

```python
# 先复制一份，避免修改原始 test
train_fe = train.copy()
test_fe = test.copy()

# 只把真实测试行拼到训练集后面，假行不参与频次统计
full = pd.concat(
    [train_fe.drop(columns=["target"]), test_fe.loc[real_indexes].drop(columns=["ID_code"])],
    axis=0,
    sort=False,
)

for i in range(200):
    col = f"var_{i}"
    # 统计该列在每个取值上的出现次数
    count_values = full.groupby(col)[col].count()
    # 训练集和测试集都按自己的取值去查表
    train_fe[f"new_{col}"] = train_fe[col].map(count_values)
    test_fe[f"new_{col}"] = test_fe[col].map(count_values)
    # 理论上不会有缺失，但假样本可能取到没见过的值，填 0 更安全
    test_fe[f"new_{col}"] = test_fe[f"new_{col}"].fillna(0)

print("新增特征数：", len([c for c in train_fe.columns if c.startswith("new_")]))
```

这里还有一个容易忽略的细节：`groupby(...).count()` 统计的是“该取值在 full 里出现了多少次”，映射回训练集时，训练集自己的每个值当然都能查到；测试集的假样本可能取到 full 里没有的值，`map` 会产生 `NaN`，所以我们统一填 0，表示“这个值在真实数据里从未出现过”。

### 2.3 模型训练与超参数调优

我们接着解释为什么“一个特征训练一个模型”。如果直接把 200 个原始特征加 200 个 count 特征一起喂给 LightGBM，模型会尝试所有特征之间的组合。但 `new_var_3` 只对 `var_3` 有意义，它和 `var_57` 的组合基本是噪声；400 列全量训练既慢，又容易过拟合。

材料里的做法是：每次只取一个原始特征和它对应的 count 特征，训练一个很小的 LightGBM，预测该特征单独给出的“正类概率”；200 个特征训练 200 个模型，最后把 200 列概率相加（或平均）。这样每个模型只学二维关系，训练时间从一小时降到几分钟，AUC 反而更高。

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# 固定随机种子，保证 200 个模型的切分完全一致
seed = 2319
param = {
    "num_leaves": 8,
    "min_data_in_leaf": 17,
    "learning_rate": 0.01,
    "min_sum_hessian_in_leaf": 9.67,
    "bagging_fraction": 0.8329,
    "bagging_freq": 2,
    "feature_fraction": 1,      # 关键：每个模型只有 2 列，必须两列都看
    "lambda_l1": 0.6426,
    "lambda_l2": 0.3067,
    "min_gain_to_split": 0.02832,
    "max_depth": -1,
    "seed": seed,
    "objective": "binary",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "metric": "auc",
    "is_unbalance": True,
    "boost_from_average": "false",
    "num_threads": 8,
}

features = [f"var_{i}" for i in range(200)]
X = train_fe[features]
y = train_fe["target"]

# 用同一个切分评估 200 个模型的总分
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

`feature_fraction=1` 是材料里特别强调的一个参数。默认值小于 1 时，每次迭代只随机看一部分特征；但这里每个模型本来就只有两列，如果再随机丢列，模型可能整轮都看不到 count 特征，魔法特征的作用就被削弱了。所以必须让每次迭代都使用全部两列。

下面是训练和汇总预测的核心循环。我们为每个 `var_i` 建一个 `lgb.Dataset`，训练 110 轮，预测验证集，存到 `y_hat` 的第 `i` 列。最后按行求和，得到每个客户的综合分数。

```python
iterations = 110
y_hat = np.zeros((len(X_valid), 200))

for i, feature in enumerate(features):
    # 这个模型只看原始特征和它自己的 count 特征
    feat_choices = [feature, f"new_{feature}"]
    lgb_train = lgb.Dataset(X_train[feat_choices], y_train)
    gbm = lgb.train(param, lgb_train, iterations, verbose_eval=-1)
    y_hat[:, i] = gbm.predict(X_valid[feat_choices], num_iteration=gbm.best_iteration)

# 200 个弱模型的概率相加，得到最终排序分数
sub_preds = y_hat.sum(axis=1)
print("CV AUC:", roc_auc_score(y_valid, sub_preds))
```

材料里这一步打印的交叉验证分数是 **0.921011**。相比之前所有人卡住的 0.901，这是一个明显的跃升。需要说明的是，这个 0.921 是作者用简单 `train_test_split` 而不是完整 K 折得到的，所以绝对值和排行榜分数不会完全一致；它主要用来快速比较方案。

关于超参数，材料没有做很复杂的网格搜索，而是直接给了一组在小模型上表现稳定的参数：叶子数 8、最小叶子样本 17、学习率 0.01、L1/L2 正则、行采样 0.83。因为每个模型只有两列特征，模型容量必须小，叶子太多会迅速记住训练集的频次噪声。`is_unbalance=True` 让 LightGBM 自动给少数类更高权重，缓解正负样本不均衡。我们的调参重点因此不是“把单模型调得多复杂”，而是确认三件事：count 特征有没有加、`feature_fraction` 是不是 1、每个模型是不是只喂自己的两列。

### 2.4 模型测试与指标报告

确认方案有效后，我们在全部训练数据上重训 200 个模型，对测试集逐列预测并求和。测试集里真假样本都要预测，所以这里用完整的 `test_fe`，不做行过滤。

```python
# 用全量训练数据重新训练，每个特征一个模型
test_hat = np.zeros((len(test_fe), 200))

for i, feature in enumerate(features):
    feat_choices = [feature, f"new_{feature}"]
    lgb_train = lgb.Dataset(X, y)
    gbm = lgb.train(param, lgb_train, iterations, verbose_eval=-1)
    test_hat[:, i] = gbm.predict(test_fe[feat_choices], num_iteration=gbm.best_iteration)

# 行方向求和：每个客户得到 200 个模型的累计分数
test_preds = test_hat.sum(axis=1)

submission = pd.DataFrame({
    "ID_code": test_fe["ID_code"],
    "target": test_preds,
})
submission.to_csv("submission.csv", index=False)
print(submission.head())
```

指标报告分三层。第一，本地验证 AUC 是 **0.9210**，这是 200 个模型概率求和后的排序质量。第二，材料标题给出的公开榜分数是 **0.922**，和本地验证非常接近，说明方案没有严重过拟合。第三，参赛前的普遍瓶颈是 **0.901**，加入 count 特征后提升约 0.02，这在 AUC 上是非常大的进步。

我们还会检查提交格式：行数必须是 20 万，`ID_code` 要和测试集完全一致，`target` 不能有缺失。因为最终分数是 200 列相加，任何一列的预测出现 `NaN` 都会让整行变成 `NaN`，所以提交前要用 `submission.isna().sum()` 确认一遍。

---

## 三、总结

本文中我们完成了 Santander 客户交易预测的完整流程。我们首先通过 EDA 确认数据是 20 万训练样本、20 万测试样本、200 个匿名浮点特征，目标是类别不平衡的 0/1 交易行为。其次我们识别出测试集中约 10 万条人工合成样本，只把训练集和 10 万条真实测试样本拼起来统计频次，为每个特征构造了“该取值出现了多少次”的 count 特征。接着我们没有把 400 列一起塞进一个大模型，而是为每个原始特征单独训练一个只含“原始 var + 对应 count”的小 LightGBM，并把 `feature_fraction` 设为 1，保证每次迭代都能看到魔法特征。最后我们把 200 个模型的预测概率按行求和，在本地验证集得到约 0.921 的 AUC，材料中的公开榜成绩约为 0.922，明显越过了此前 0.901 的瓶颈。

这个比赛给我们的最大启发是：**特征的价值不取决于它看起来高不高级，而取决于它是否揭示了数据生成过程**。count 特征本身只是“数数”，但它把正负类在数值精度上的细微差异放大了。另一个教训是**模型结构要服从特征结构**：既然 count 只和它自己的变量配对，那就一个变量一个模型，既快又准。最后，合成测试样本提醒我们：比赛数据不一定都是“真实用户”，先弄清数据是怎么来的，再决定怎么清洗和验证，往往比急着调参更重要。