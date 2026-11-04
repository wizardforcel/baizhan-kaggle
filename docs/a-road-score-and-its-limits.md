# A Road Score and Its Limits

## 一、比赛简介

### 1.1 比赛背景

这个比赛的数据来自 Kaggle Playground 系列的第 S5E10 期（Playground S5E10）。Playground 是 Kaggle 定期举办的练习赛，数据是**模拟生成**的，不涉及真实的人和真实的事故记录。这一期的上游数据是 Ian Too 整理的 **Simulated Roads Accident Data**（模拟道路事故数据）：每一行描述的是一条道路在某种条件下的状况，标签是一个 0 到 1 之间的数，叫 `accident_risk`（事故风险）。

这里有个容易被误解的地方，也是这份材料想讲清楚的第一件事：

> **一个 0 到 1 之间的数字，不等于一个"是/否"的标签。**

0.83 这个数，如果它来自"这条路上发生过事故吗"的记录，那它是概率；但在这份数据里，它是主办方用某种规则**算出来的一个连续分数**，只不过恰好被压在了 0 到 1 之间。把它当成二分类标签去训练，等于把"风险 0.7"和"风险 0.9"抹成同一个"1"，凭空扔掉信息。

第二件事是这份材料的核心警告：**这个分数是合成的，不能用来指导真实的道路安全决策，也不能用来估计事故发生率。** 模型告诉你"把这条路的路面类型换掉，预测风险会降"，只说明模型在这个数据上这么认为，完全不等于现实世界里换个路面就真的更安全。这就是**相关性**和**因果性**的区别，后面我们还会回到这一点。

材料的作者 Clare Laurent 把这份 notebook 定位成一份独立研究（independent study），提交的模型是**预先设定好的全特征 CatBoost 基线**，没有在排行榜上调过参。

### 1.2 比赛目标

数据目录下是三个标准文件：

- `train.csv`：训练集。第一列是样本编号，中间是道路和环境的各项属性（道路类型 `road_type`、路面状况、照明、天气、限速等），最后一列是 `accident_risk`。
- `test.csv`：测试集，列结构和训练集去掉标签后一致。
- `sample_submission.csv`：提交样例，第一列编号，第二列 `accident_risk`，要求填**连续的预测值**。

任务就是：给每条测试记录预测一个事故风险分数。

### 1.3 评价标准

这是一个**回归**问题，评价指标是 **RMSE**（Root Mean Squared Error，均方根误差）。算法很直白：

1. 对每条样本算误差：预测值减真实值；
2. 把每个误差平方（这样正负不会互相抵消，而且大错被放大）；
3. 求平均，再开根号。

公式写出来是 `sqrt(mean((y_pred - y_true)^2))`。**越低越好**，0 表示完美。

为什么最后要开根号？因为平方之后单位变了（"风险"的平方没有物理意义），开根号能把单位还原成原标签的单位，读起来更直观。这一点在后面讨论"裁剪"时会用到：**RMSE 和 MSE 对误差的排序完全一致，只是刻度不同。**

---

## 二、解题过程

整份材料的思路可以概括成一句话：**用固定的参数训练一个回归基线，用一个"什么都不学"的常数基线做对照，再用一次事先声明的消融和一组分箱误差诊断，去回答"这个模型到底学到了什么"。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步先确认数据的"合同"：目标列叫什么、提交格式是什么。

```python
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG = {
    "episode": "s5e10",
    "targets": ["accident_risk"],   # 回归目标：一个连续分数
    "family": "regression",         # 明确告诉后面的代码：这是回归，不是分类
    "ablation": ["road_type"],      # 第二个实验要拿掉的道路类型
    "date_columns": [],
}

# 按比赛 slug 在 input 目录下找数据
candidates = [p.parent for p in Path("/kaggle/input").rglob("train.csv")
              if CONFIG["episode"] in str(p).lower()]
DATA = candidates[0]

# 用样例文件的列名当契约
schema = pd.read_csv(DATA / "sample_submission.csv", nrows=3).columns.tolist()
id_col = schema[0]
assert schema[1:] == CONFIG["targets"]
```

`family` 这个字段看起来不起眼，其实是整份代码的"总开关"：后面的特征处理、模型选择、评估函数都靠它分支。把任务类型显式写进配置，比藏在某一行 `if` 里可靠得多——将来换成分类任务，只改这一处，所有分支自动切换。

第二步是抽样，并检查标签的取值范围。这一步决定了"裁剪"这个话题到底成不成立。

```python
MAX_ROWS = 120000
reservoir, rows_seen, missing = None, 0, None
rng = np.random.default_rng(20261026)

for chunk in pd.read_csv(DATA / "train.csv", chunksize=100000):
    rows_seen += len(chunk)
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)

    chunk["_sampling_priority"] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, "_sampling_priority")

train = reservoir.drop(columns="_sampling_priority").reset_index(drop=True)
y_all = train["accident_risk"].astype(float)

# 关键检查：真实标签是不是真的都落在 [0, 1] 里
print("样本数:", len(train), " 源文件行数:", rows_seen)
print("最小:", y_all.min(), "最大:", y_all.max())
print(y_all.describe())
```

这里打印的"最小/最大"不是装饰。后面我们会讨论"把预测值裁剪到 [0,1] 会不会让 RMSE 变差"，这个问题的答案**完全取决于真实标签是否都在 [0,1] 之内**。如果数据里有 1.2 或者 -0.05，裁剪就会把预测硬拉到错误的方向，误差反而变大。所以这个前置检查必须做，不能想当然。

`describe()` 还能告诉我们另一件事：均值和标准差差多少。如果所有值都挤在 0.2 附近，那这个"0 到 1 的分数"其实有效范围很窄，模型的预测也会挤在一条窄带里，这时候看 RMSE 不如看"模型有没有预测出差异"。

第三步看缺失和类别分布：

```python
# 哪些字段缺失最多？缺失率高的字段往往就是"用不上"的字段
print(missing.sort_values(ascending=False).head(8))

# 道路类型这类类别字段，看看每一类有多少条
print(train["road_type"].value_counts())
```

类别分布决定了消融实验的解读方式：如果 `road_type` 有某一类占了 90% 的样本，那"拿掉它分数掉得不多"可能只是因为**大部分样本本来就一样**，而不是因为这一列没用。

### 2.2 数据清洗

这份 notebook 的清洗原则只有一句话：**只做无状态变换**。所谓无状态，就是每个值的处理只依赖它自己，不依赖整张表的统计量。

```python
def features(frame, omit=()):
    """把原始表变成模型输入。omit 用于消融：直接不取这些列。"""
    x = frame[[c for c in feature_columns if c not in omit]].copy()

    for c in list(x.columns):
        if c in CONFIG.get("date_columns", []):
            dt = pd.to_datetime(x.pop(c), errors="coerce")
            x[c + "_year"] = dt.dt.year
            x[c + "_month"] = dt.dt.month
            x[c + "_weekday"] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 类别列：缺失变成独立的类别，而不是填众数
            x[c] = x[c].astype("string").fillna("__MISSING__").astype(str)
        else:
            # 数值列：转不动就 NaN，无穷大也当缺失
            x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return x
```

为什么"缺失变成类别"而不是填众数？因为模型的输入是"这条路的状况"，而"没记录"本身就是一种状况。填众数相当于替数据编造了一个"最普通的情况"，会系统性地把缺失样本往中间拉。CatBoost 的原生类别处理能把这个 `__MISSING__` 当成一个正常类别去学，不需要我们操心。

为什么无穷大要处理？`road_type` 这类字段没事，但如果有"每公里弯道数"这种比值型字段，分母为 0 就会产生 `inf`。树模型对 `inf` 的处理各家实现不一样，最稳的办法是统一转成缺失。

接着是标签和划分。回归任务**不能分层**（标签是连续值，没有"层"可分），所以直接随机切：

```python
regression = CONFIG["family"] == "regression"
y = train["accident_risk"].astype(float).to_numpy()

indices = np.arange(len(train))
# 回归任务给 stratify 传 None：连续标签没有类别可分
fit_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=20261026, stratify=None
)
assert set(fit_idx).isdisjoint(val_idx)
```

这里和分类任务最大的差别就是 `stratify=None`。如果硬要把连续标签分成几桶再分层，其实是把回归偷偷变成了分类，而且桶的边界怎么定又是一门玄学。直接随机切、靠固定种子保证可复现，是更干净的做法。

### 2.3 模型训练与超参数调优

模型是 **CatBoost 回归器**，参数完全固定：

```python
from catboost import CatBoostRegressor

def model_fit(x, target):
    cat_cols = x.select_dtypes(include=["object", "string"]).columns.tolist()

    params = dict(
        iterations=350,
        depth=6,
        learning_rate=0.08,
        random_seed=20261026,
        verbose=False,
        thread_count=4,
        allow_writing_files=False,
    )
    # loss_function="RMSE"：直接优化和评价口径一致的损失
    model = CatBoostRegressor(loss_function="RMSE", **params)
    model.fit(x, target, cat_features=cat_cols)
    return model
```

**为什么用 `RMSE` 作为损失，而不是 `MAE`？** 因为比赛就是按 RMSE 评分的。用 MAE 训练、用 RMSE 评价，等于"训练时追求中位数、评价时看平均数"，两者会对不上。这里的取舍很简单：**损失函数尽量和评价指标一致。**

**为什么不调参？** 材料的态度写得很明确：这是预先设定好的实验（prespecified），参数在看到分数之前就定死了。理由不是"调参没用"，而是**这份研究要回答的问题不是"能刷多高"**。如果一边调参一边看验证分数，最后得到的那个数就不再是"一个固定配方能达到的水平"，而变成"我在这个验证集上挑出来的最好结果"——后者对读者的参考价值小得多。

**常数基线。** 和分类任务一样，回归也需要一个"什么都不学"的对照：

```python
def constant_prediction(target, n):
    # 不看任何特征，永远预测训练集标签的平均值
    return np.full(n, target.mean())
```

这个基线的 RMSE 恰好等于验证集标签的**标准差**（如果训练集和验证集的均值一致）。它给出了一条非常清晰的地板线：**模型必须显著低于这个数，才说明它真的从特征里学到了东西。** 很多"看起来不错"的模型，其实连"永远报平均值"都打不过。

### 2.4 模型测试与指标报告

**第一个实验：基线对照。** 在同一个留出集上比常数预测和 CatBoost：

```python
from sklearn.metrics import mean_squared_error

def score(truth, p):
    # 回归只报 RMSE：单位和标签一致，团队里谁都能一眼看懂
    return {"RMSE": float(np.sqrt(mean_squared_error(truth, p)))}

X = features(train)
baseline = constant_prediction(y[fit_idx], len(val_idx))
model = model_fit(X.iloc[fit_idx], y[fit_idx])
pred = model.predict(X.iloc[val_idx])

results = {"constant": score(y[val_idx], baseline),
           "boosting": score(y[val_idx], pred)}
print(results)
```

**第二个实验：消融掉 `road_type`。** 还是那个规矩——名单事先写在 `CONFIG` 里，跑之前不改：

```python
omit = [c for c in CONFIG["ablation"] if c in feature_columns]
reduced = features(train, omit)
alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
results["availability_ablation"] = score(y[val_idx], alternate.predict(reduced.iloc[val_idx]))
```

这份消融有额外的意义：**现实里道路类型是最容易被改的东西**（改铺装、改等级），所以"模型有多依赖 `road_type`"看上去像是"改道路类型能带来多少收益"。但作者立刻提醒：**消融是诊断，不是部署许可，更不是干预效果的估计。** 模型在这个数据上依赖某一列，只说明这一列和标签在数据里有统计关联，不说明改动它会产生因果效应。

**第三个诊断：分箱误差，看偏差的方向。** 这是回归任务里最有用、却最常被跳过的一步：

```python
diagnostics = pd.DataFrame({
    "id": train.iloc[val_idx][id_col].to_numpy(),
    "actual": y[val_idx],
    "prediction": pred,
})
diagnostics["error"] = diagnostics.prediction - diagnostics.actual

# 按真实值分成 5 档（等频），看每一档的误差
diagnostics["target_band"] = pd.qcut(diagnostics.actual, 5, duplicates="drop")
bands = diagnostics.groupby("target_band", observed=True).agg(
    rows=("error", "size"),                     # 这一档有多少条
    MAE=("error", lambda s: s.abs().mean()),    # 平均绝对误差
    bias=("error", "mean"),                     # 平均误差：正数=系统性高估，负数=低估
)
print(bands)
```

`bias` 这一列是最有信息量的。如果低风险那一档的 `bias` 是正的、高风险那一档是负的，说明模型**把所有预测都往中间挤**——这是平方误差损失的典型行为：对极端值不敢下判断，宁可报个中间数，因为报错了平方惩罚也没那么大。这种"向均值收缩"的现象在回归里普遍存在，光看一个总体 RMSE 是看不出来的。

**第四个检查：裁剪到底会不会让误差变大？** 材料里提了一个很好的问题：如果真实值都在 `[0,1]` 之间，把预测值裁剪到 `[0,1]`，会不会让平方误差变大？

答案是不会。理由是一个纯几何事实：**把一个点投影到它所在的区间里，它到区间内任何一点的距离只会变小、不会变大。** 设真实值 `y ∈ [0,1]`，预测值 `p` 在区间外（比如 1.3），裁剪后变成 `p' = 1`，那么 `|p' - y| = 1 - y ≤ 1.3 - y = |p - y|`。平方之后大小关系不变。所以只要我们事先确认过标签确实都在 `[0,1]` 内，裁剪就是**免费的正确性提升**。

```python
# 前提检查：标签必须全在 [0,1] 内，否则下面的结论不成立
assert y.min() >= 0.0 and y.max() <= 1.0, "标签超出 [0,1]，不能盲目裁剪"

pred_clipped = np.clip(pred, 0.0, 1.0)
rmse_raw = np.sqrt(mean_squared_error(y[val_idx], pred))
rmse_clip = np.sqrt(mean_squared_error(y[val_idx], pred_clipped))
print(rmse_raw, rmse_clip)

# 数学上保证：裁剪后的 RMSE 只会 <= 裁剪前
assert rmse_clip <= rmse_raw + 1e-12
```

这段代码里最值钱的不是 `np.clip`，而是第一行的 `assert`。**裁剪"不会变差"这个结论是有前提的**，前提不成立，结论就反转。写代码时把前提写成断言，比写在注释里可靠得多。

**提交。** 最后用全量抽样数据重训，分块预测测试集：

```python
final = model_fit(X, y)
sample = pd.read_csv(DATA / "sample_submission.csv")
submission = sample.copy()
submission["accident_risk"] = submission["accident_risk"].astype(float)

offset = 0
for chunk in pd.read_csv(DATA / "test.csv", chunksize=100000):
    n = len(chunk)
    assert np.array_equal(chunk[id_col].to_numpy(), sample.iloc[offset:offset + n, 0].to_numpy())
    p = final.predict(features(chunk))
    submission.loc[submission.index[offset:offset + n], "accident_risk"] = p
    offset += n

assert offset == len(sample)
assert np.isfinite(submission["accident_risk"].to_numpy(dtype=float)).all(), "有 NaN 或 inf"
submission.to_csv("submission.csv", index=False)
```

`np.isfinite` 那一行是回归任务特有的检查：分类输出概率天然在 `[0,1]`，回归输出却可能因为特征里有极端值而爆出 `inf`。提交文件里出现 `inf`，评分系统通常会直接判错。

**关于榜单上的第一名。** 材料附了冠军 Tilii 的做法：用了多种树模型和神经网络模型，包括调过参的 **TabM**，还用**遗传编程**自动生成了 11 个特征，在集成阶段起作用；再加一个 CatBoost 残差集成和最后的爬山（hill climbing）融合，收益都很小。原作者自己也明确表示，**不确定这些微小的提升里哪些是真正有意义的**。这句话很诚实，也是我们想传达的：当提升已经小到和随机波动一个量级时，"第几名"很大程度上是运气。

---

## 三、总结

本文中我们完成了"A Road Score and Its Limits"的一份独立研究型回归基线。我们首先把任务类型、目标列和消融字段写进配置，用提交样例的列名校验数据格式。其次我们用分块读取加蓄水池抽样，在 12 万行的上限内做了固定种子的均匀抽样，并特意检查了标签的最小值和最大值——这一步决定了后面"裁剪"的结论是否成立。接着我们对特征只做无状态变换，把类别列的缺失变成独立类别、把数值列的无穷大转成缺失，回归任务不做分层、直接固定种子切出 20% 留出集。然后我们用参数完全固定的 CatBoost（350 轮、深度 6、学习率 0.08、RMSE 损失）和"永远报平均值"的常数基线对照，做了事先声明的 `road_type` 消融，又按真实值分五档统计误差，用 `bias` 一列看模型是不是在向均值收缩。最后我们验证了"标签在 `[0,1]` 内时裁剪不会增大平方误差"这个结论，分块预测测试集、检查有限性、写出提交文件。

这份材料留给我们的最大启发，是**分清三种话**：数据里观察到的事实、模型表现出的敏感性、以及现实世界的因果效应。RMSE 变好是第一种，去掉 `road_type` 分数下降是第二种，而"改路面能减少事故"是第三种——前两种都推不出第三种。材料末尾那句提醒值得记住：**这个分数是合成的，不能指导真实的道路安全决策。**
