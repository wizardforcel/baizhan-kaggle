# DDCup 能耗预测公开赛（2026pwxc00-public）

## 一、比赛简介

### 1.1 比赛背景

这个比赛给出一张**设备能耗表**：每一行是一台设备在某段工作条件下的运行记录，表里记着温度、运行时长、负载这类测量值，以及这段记录实际消耗了多少电（单位是千瓦时，kWh）。

我们要做的事是：**看着这些测量值，预测这台设备会消耗多少电**。

为什么值得做？因为能耗预测在现实里非常有用——工厂要知道明天大概用多少电、机房要判断哪台机器在偷偷费电、电网要提前安排供电。如果模型能根据当前工况估出能耗，就能提前发现异常、优化排班。

这个比赛的特别之处在于它是**合成数据**（材料里明确说「样本是独立的合成记录，没有时间顺序」）。也就是说，数据不是真的从某台设备上抄下来的，而是按一套多因素的规则生成出来的，再叠上噪声。这样设计是为了让初学者能专注在方法上：**数据是干净的，考验的是你的流程对不对，而不是你的清洗技巧有多花哨。**

另一个特点是它是**系列教程赛**：主办方配了从「第一次写 Python」到「表格 Transformer」的十课教程，从零带你走完整个流程。所以这篇 writeup 的主线，也沿着这条由浅入深的路走。

### 1.2 比赛目标

数据分三个文件：

| 文件 | 内容 |
| --- | --- |
| `train.csv` | 20000 行，带答案 `energy_kwh` |
| `test.csv` | 10000 行，没有答案 |
| `sample_submission.csv` | 提交模板，两列 `id` 和 `energy_kwh` |

训练表里的列大致分三类：

- `id`：记录编号，**只是标识，不能当特征**；
- 特征列：一共 **12 列**，都是数值，包括温度、运行时长（`operating_hours`）、负载等工况测量值；
- `energy_kwh`：目标列，要预测的小数。

所以输入是 12 个数字，输出是 1 个数字——这是一个典型的**回归**问题（预测连续数值，而不是分类）。

### 1.3 评价标准

比赛用**均方根误差（RMSE）**评分，**越低越好**。

RMSE 的算法分四步，我们用一个只有两条记录的小例子走一遍：

1. 每条记录算「预测值减真实值」，比如真实 `[10, 20]`、预测 `[12, 18]`，差是 `[2, -2]`；
2. 每个差平方，得到 `[4, 4]`——平方是为了让正负误差不会互相抵消；
3. 求平均，得到 4；
4. 开平方，得到 **2**。

为什么最后要开平方？因为平方之后单位变成了「千瓦时的平方」，没法直接理解。开方之后，RMSE 又回到和能耗一样的单位，可以说「平均每次大概差 2 度电」。

RMSE 有个重要性格：**它对大错误特别敏感**。同样是平均绝对误差为 2 的两组预测，如果错误集中砸在一条记录上（偏 4），RMSE 会从 2 涨到约 2.83。所以优化 RMSE 的模型会优先避免「个别预测离谱」，而不是追求每条都差不多准。

> 关于排行榜：材料反复提醒，公开榜分数会受环境版本、随机性、公私榜划分影响，**不能把榜单分数当成学习是否完成的唯一标准**。我们下面主要看自己切出来的验证集。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**先建立一套可信的评价流程，再在这个流程里一步步换更强的模型。** 我们首先做 EDA 看清数据，其次划分训练/验证集并建立基线，接着依次试线性模型、树模型和神经网络，然后固定方案用全量数据重训，最后生成并回读检查提交文件。

### 2.1 数据探索（EDA）

第一步是把三个文件读进来，先确认形状对不对：

```python
from pathlib import Path
import numpy as np
import pandas as pd

# 比赛数据通常挂在 Kaggle 的 input 目录下；这里按几个常见位置依次找
data_candidates = [
    Path("/kaggle/input/competitions/2026pwxc00-public"),
    Path("/kaggle/input/2026pwxc00-public"),
    Path("public_data"),
]
data_dir = next((p for p in data_candidates if (p / "train.csv").exists()), None)
if data_dir is None:
    raise FileNotFoundError("没有找到比赛数据")

train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
sample = pd.read_csv(data_dir / "sample_submission.csv")

# 训练表应该 20000 行，测试表 10000 行；对不上说明读错了文件
print("训练表：", train.shape)
print("测试表：", test.shape)
display(train.head())
```

打印结果是训练表 20000 行、测试表 10000 行。这个数量级很舒服：不用分布式，一台普通笔记本就能跑完。

接着我们用统计摘要回答三个具体问题：**每列有多少非空值、数值大概在什么范围、有没有空值**：

```python
# describe 会给出 count/mean/std/min/25%/50%/75%/max 八行
display(train.describe().round(2))

# 单独数一遍空值，不能只看 describe 的 count 就下结论
print("训练表空值数：")
print(train.isna().sum())
```

结论是：**这份数据没有缺失值**。材料里说得很直白——这是为了让第一次练习的人少踩坑，故意把数据做干净的。但我们记住一条：不能因为「这次没缺失」，就以为「真实数据也不会有缺失」。

然后画两张图，每张图只回答一个问题：

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# 左图：目标值的分布，看能耗集中在哪个区间、有没有极端值
axes[0].hist(train["energy_kwh"], bins=40, color="#287c68", edgecolor="white")
axes[0].set(xlabel="Energy (kWh)", ylabel="Record count", title="Target distribution")

# 右图：运行时长和能耗的关系，看是否存在明显趋势
axes[1].scatter(train["operating_hours"].iloc[:1000], train["energy_kwh"].iloc[:1000],
                s=10, alpha=0.3, color="#b94a55")
axes[1].set(xlabel="Operating hours", ylabel="Energy (kWh)", title="1000 records")
plt.tight_layout()
plt.show()
```

左图告诉我们目标的大致范围和形状。右图更有信息量：**时长越长，能耗整体越高**，说明 `operating_hours` 确实是有用的特征；但在同一个时长上，点会明显地上下散开——这说明除了时长，还有别的因素（温度、负载）在影响能耗，也说明数据里有噪声。

这一点决定了后面选模型的方向：**关系不是一条纯粹的直线，但也不是完全杂乱**，所以从线性模型起步是合理的，再上树模型和非线性模型会有进一步提升空间。

### 2.2 数据清洗

这份数据其实没什么好「洗」的：没有缺失值、没有文本列要编码、没有明显的时间顺序需要处理。真正需要小心的是**评价流程本身干不干净**。

我们把训练表切成两部分：

```python
from sklearn.model_selection import train_test_split

TARGET = "energy_kwh"
X = train.drop(columns=["id", TARGET])   # id 只是编号，当特征会引入无关噪声
y = train[TARGET]
feature_columns = X.columns.tolist()     # 记下列顺序，测试集要按同样顺序取列

# 80% 训练、20% 验证，random_state 固定方便不同模型之间比较
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 三条断言：X 和 y 必须严格对齐，且两边不能有重叠的行
assert X_train.index.equals(y_train.index)
assert X_valid.index.equals(y_valid.index)
assert set(X_train.index).isdisjoint(set(X_valid.index))
print(X_train.shape, X_valid.shape)
```

切出来正好是 16000 行训练、4000 行验证，输入都是 12 列。**X 和 y 必须用同一次划分**，否则会把 A 行的工况配上 B 行的能耗，模型学到的全是错的。

这里有三条纪律，材料里用三个「泄漏例子」专门强调过，我们照做：

1. **标准化只能在训练部分学均值和标准差**，再拿去变换验证集。如果先对全体数据做标准化再切分，验证集的分布信息就提前漏进了训练。
2. **绝对不把 `energy_kwh` 留在 X 里**。曾经有人这么干，验证分数好得不像话——那不是模型强，是把答案当特征了。测试集根本没有这一列。
3. **不要反复用验证集挑参数**。挑多了，验证集就变成了训练集的一部分，它给出的分数会越来越乐观。

### 2.3 模型训练与超参数调优

**第一步，建立两个「不看输入」的基线。** 基线的作用是给我们一个下限：如果模型连它都打不过，那说明流程有问题，而不是模型不够强：

```python
from sklearn.metrics import mean_squared_error

def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))

rows = []
for name, value in [("Mean", y_train.mean()), ("Median", y_train.median())]:
    rows.append({
        "method": name,
        "constant": float(value),
        # 注意：常数只能从 y_train 算，不能用 y_valid 的统计量
        "train_rmse": rmse(y_train, np.full(len(y_train), value)),
        "valid_rmse": rmse(y_valid, np.full(len(y_valid), value)),
    })
display(pd.DataFrame(rows).round(4))
```

这里最容易犯的错是拿 `y_valid.mean()` 去做预测——那等于偷看了验证集的答案。哪怕只偷看了一个平均数，验证分数也不再代表「面对未知数据」的表现。

**第二步，线性模型和 Ridge。** 线性模型假设「每个特征乘一个权重，全部加起来再加一个截距」。12 个特征就学 12 个权重加 1 个截距。Ridge 在它的基础上加了一项惩罚（L2 正则化）：权重越大，罚得越多，目的是防止模型为了拟合细小的波动而把系数拉得特别大。

两者都用 `Pipeline` 串起来，保证标准化只对训练部分拟合：

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 标准化是必需的：各列量纲差很多，不缩放的话惩罚对不同列的力度不一样
linear_model = make_pipeline(StandardScaler(), LinearRegression())
ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

comparison = pd.DataFrame([
    {"model": "Mean baseline",
     "train_rmse": rmse(y_train, np.full(len(y_train), y_train.mean())),
     "valid_rmse": rmse(y_valid, np.full(len(y_valid), y_train.mean()))},
    {"model": "LinearRegression",
     "train_rmse": rmse(y_train, linear_model.predict(X_train)),
     "valid_rmse": rmse(y_valid, linear_model.predict(X_valid))},
    {"model": "Ridge",
     "train_rmse": rmse(y_train, ridge_model.predict(X_train)),
     "valid_rmse": rmse(y_valid, ridge_model.predict(X_valid))},
])
display(comparison.round(4))
```

使用特征的线性模型明显优于均值基线——这证明 12 个特征里确实有可用的信息。LinearRegression 和 Ridge 的分数非常接近，这也正常：数据量有 16000 行，共线性不严重时，正则化的作用本来就有限。

**关于调参**，我们只动 `alpha` 这一个超参数，而且严格只改它：

```python
alpha_results = []
for alpha in [0.1, 1.0, 100.0]:
    candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    candidate.fit(X_train, y_train)
    alpha_results.append({
        "alpha": alpha,
        "train_rmse": rmse(y_train, candidate.predict(X_train)),
        "valid_rmse": rmse(y_valid, candidate.predict(X_valid)),
    })
display(pd.DataFrame(alpha_results).round(4))
```

之所以强调「只改一个」，是因为**一次改两个参数，就分不清分数变化是谁造成的**。alpha 变大时，训练误差一般会上升（模型被束缚住了），验证误差可能先降后升。如果验证误差没有明显改善，那也是有效结论——说明线性模型的正则强度不是瓶颈，该换模型形式了。

**第三步，XGBoost。** 树模型和线性模型的思路完全不同：它不问「乘多少再加起来」，而是一连串判断——「运行时长是否小于某个值？不是的话，负载是否超过某个值？」一路问到叶子，用叶子里训练样本的平均值当预测。

XGBoost 是**提升树**：它先训练一棵很浅的树，然后看「还差多少」（残差），再训练一棵新树专门去补这个差距，一轮一轮往上叠。每轮的修正幅度由一个叫**学习率**的参数控制。学习率小，每步走得稳但需要的轮数多；学习率大，走得快但容易冲过头。

我们用的配置是：最多 1000 轮、每棵树最深 4 层、学习率 0.05、每轮随机用 90% 的样本和 90% 的特征、叶子权重加 L2 正则，并且**连续 40 轮验证分数没改善就停**（这叫早停）：

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=1000,          # 最多允许 1000 轮，不是必须跑满
    max_depth=4,                # 单棵树最多分 4 层，控制复杂度
    learning_rate=0.05,         # 每轮只修正一小步，防止过冲
    subsample=0.9,              # 每轮用 90% 的样本，增加树之间的差异
    colsample_bytree=0.9,       # 每棵树用 90% 的特征，同上
    reg_lambda=1.0,             # 叶子权重的 L2 惩罚
    objective="reg:squarederror",  # 优化平方误差，和 RMSE 评价一致
    eval_metric="rmse",
    tree_method="hist",         # 直方图建树，速度快、内存省
    n_jobs=2,
    random_state=42,
    early_stopping_rounds=40,
)

# 最后一个 eval_set 才是早停用的验证集，顺序不能反，更不能把 test 放进来
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_valid, y_valid)],
          verbose=False)
```

这里有两个细节必须注意：

- `eval_set` 里放了训练集和验证集两份，是为了**画两条学习曲线**对比。但早停只认**最后一个**，所以验证集必须放最后。
- **绝对不能把 `test` 放进 `eval_set`**，因为测试集没有答案，放进去只会报错；就算能跑，也属于作弊。

早停之后，最好的那一轮往往不是最后一轮：

```python
valid_prediction = model.predict(X_valid)
print("验证 RMSE：", round(rmse(y_valid, valid_prediction), 4))

# best_iteration 从 0 开始计数，所以真实轮数要加 1
best_rounds = model.best_iteration + 1
print("最佳轮数：", best_rounds)

# 把两条曲线画出来，看验证误差在哪一轮之后不再下降
history = model.evals_result()
train_curve = history["validation_0"]["rmse"]
valid_curve = history["validation_1"]["rmse"]
```

曲线通常长这样：开头两条线都快速下降；到某一轮之后，训练线还在慢慢降，验证线却开始走平甚至回升——那一点就是模型开始「背训练数据」的地方，早停就是在那儿踩刹车。

关于深度，我们还做了单变量实验：把 `max_depth` 分别改成 2、4、6，其他条件完全不动，各跑一次，记录验证 RMSE 和最佳轮数。结论符合预期——**太浅（2 层）学不动复杂的交互，太深（6 层）更容易追逐训练细节**，4 层是个稳妥的折中。但我们没有把这个结论当成普适规律，因为一次随机划分的结果不足以说明所有情况。

**第四步，神经网络。** 为了对比，我们还试了两层隐藏层的 MLP 和一个小的表格 Transformer。MLP 固定 80/20 划分，只改隐藏层宽度（32、64、128），每个宽度跑 42、43、44 三个随机种子，取三次平均来比较，避免「恰好碰上好运气的种子」而误判。Transformer 则把 12 个特征各自映射成 32 维向量，再加一个汇总用的 CLS token，用 4 个注意力头让同一行的特征互相交换信息。

这两个实验的价值不在分数，而在**方法**：固定数据划分、固定优化器和训练轮数、多种子重复、报告平均值和波动。材料里有一句话我们很认同——「不要为了追求一个漂亮分数，偷偷删掉表现差的随机种子」。

### 2.4 模型测试与指标报告

确定方案之后，最后一步是**用全部 20000 条带答案的记录重新训练**，再对测试集预测。

这里有个容易被忽略的点：重训之后，之前的验证集已经被吃进训练数据了，所以**必须去掉早停**，并把轮数固定成刚才选出的 `best_rounds`：

```python
# 复制刚才的参数，但去掉早停（验证集已经不存在了），轮数固定为最佳轮数
final_params = model.get_params()
final_params.pop("early_stopping_rounds", None)
final_params["n_estimators"] = best_rounds

final_model = XGBRegressor(**final_params)
final_model.fit(X, y)
test_prediction = final_model.predict(test[feature_columns])
```

注意 `test[feature_columns]`——测试集取列的顺序必须和训练时**完全一致**。列顺序错了，模型不会报错，只会安静地给出错误的预测，这是最阴险的一种 bug。

生成提交文件时，我们用模板复制一份，再填预测值，并且逐条断言检查：

```python
submission = sample.copy()

# 四条断言：ID 顺序一致、行数一致、没有 NaN 或无穷大、列名正确
assert sample["id"].tolist() == test["id"].tolist(), "测试 ID 顺序与模板不一致"
assert len(test_prediction) == len(test), "预测数量不正确"
assert np.isfinite(test_prediction).all(), "预测中包含空值或无穷大"
submission["energy_kwh"] = test_prediction
assert submission.columns.tolist() == ["id", "energy_kwh"]

output_path = Path("/kaggle/working") / "submission.csv"
submission.to_csv(output_path, index=False)

# 关键一步：把文件读回来，检查落盘之后还是不是我们以为的样子
saved = pd.read_csv(output_path)
assert saved["id"].tolist() == test["id"].tolist()
assert len(saved) == len(test)
assert np.allclose(saved["energy_kwh"], test_prediction)
print("文件回读检查通过，行数：", len(saved))
```

「写出去再读回来」这一步看着多余，其实非常值得：浮点数写 CSV 可能丢精度、pandas 可能把某些 ID 读成别的类型、编码问题可能让列名变形。**在内存里对，不等于在文件里对**，而比赛只认文件。

指标报告我们记录了三样东西：

| 项目 | 内容 |
| --- | --- |
| 验证 RMSE | 在 4000 行验证集上，用只见过 16000 行训练的模型算出来 |
| 对照基线 | 均值/中位数基线的验证 RMSE，用来判断模型是否有真实增益 |
| 实验条件 | 数据划分方式、随机种子、库版本、CPU/GPU 型号 |

最后一项常被忽略，但它决定了别人能不能复现你的结果。材料里说得很清楚：`random_state=42` 只是让划分可复现，**42 本身不会让模型变好**；换一台机器、换一个库版本，分数都可能有细微差别。

---

## 三、总结

本文中我们完成了 DDCup 能耗预测公开赛的完整流程。我们首先用 EDA 看清数据：20000 行训练、10000 行测试、12 个数值特征、目标是无缺失的 `energy_kwh`，散点图显示运行时长与能耗正相关但同一时长下仍有明显分散，说明存在多因素影响和噪声。其次我们做了最少量的清洗——丢掉 `id`、切出 16000/4000 的训练验证划分、并立下三条纪律：标准化只在训练部分拟合、绝不把目标列留在特征里、不反复用验证集挑参数。接着我们按「均值基线 → 线性回归/Ridge → XGBoost → 神经网络」的顺序逐级加码：线性模型明显优于基线，Ridge 的 alpha 用单变量实验比较，XGBoost 用早停选出最佳轮数，还额外试了两层 MLP 和表格 Transformer，并用多种子重复来抵抗随机波动。然后我们固定参数、去掉早停、用全部 20000 条数据重训，对 10000 条测试记录预测。最后我们生成提交文件并把它读回来逐条核对，确认 ID 顺序、行数、列名和数值都没有问题。

这个比赛给我们的最大启发是：**在合成数据上，流程比模型重要得多**。同一份数据，用不用 Pipeline、标准化在哪一步拟合、早停看的是哪份数据、重训时有没有取消早停——这些「不产生分数」的细节，决定了一个分数是真的还是假的。相反，从 Ridge 换到 XGBoost 带来的提升，往往还不如把一次数据泄漏堵上。把评价流程做干净，再去谈模型强弱，才是正确的顺序。
