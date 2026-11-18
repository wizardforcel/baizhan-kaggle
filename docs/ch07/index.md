# 第7章 表格数据比赛详解：推土机拍卖价格预测（Blue Book for Bulldozers）

Blue Book 推土机拍卖价格预测是 Kaggle 上一个规模较大的表格回归比赛。重型机械（推土机、挖掘机、装载机）在二手市场上是通过拍卖成交的，同一台机器在不同年份、不同地区、不同拍卖行，成交价可能差得很远。拍卖行和买家都想知道：这台机器大概值多少钱？

这个比赛的数据量很大（41万行、53列），特征里既有数值列也有大量文字列，缺失值极多（有些列缺失率超过 90%），而且**时间维度很关键**——价格随年份变化，如果把不同年份的样本随机切分，模型会"偷看未来"，验证分数会虚高。所以必须按时间划分训练集和验证集。

评价指标是 **RMSLE（Root Mean Squared Log Error，均方根对数误差）**，越小越好。取对数后，误差变成了"相对误差"，大价格的绝对误差被自动放松，指标更关注"差了几倍"而不是"差了几块钱"。

本章我们完整走一遍这场比赛的解题流程：EDA → 清洗（日期拆分、文字编码、缺失值处理） → 特征工程 → 按时间切分验证集 → 随机森林建模 → 调优 → 提交。

---

## 7.1 比赛背景与数据

### 比赛背景

这个比赛由 Kaggle 与 Blue Book（一家做设备估值的企业）合作举办。数据来自真实拍卖记录：每一行是一台机器的成交记录，包含它的型号、生产年份、使用小时数、配置（有没有空调驾驶室、履带类型、铲刀宽度……）、拍卖地点、拍卖日期，以及最终的成交价。

难点有三个：

1. **时间维度很关键**。价格随年份变化（通货膨胀、市场需求），如果把 2012 年的样本和 2006 年的样本混在一起随机切分，模型会"偷看未来"，验证分数会虚高。所以必须**按时间划分**训练集和验证集。
2. **大量类别特征**。53列里有二十多列是文字描述，比如 `fiModelDesc`（型号描述）、`ProductSize`（大小级别），需要转成数字才能喂给模型。
3. **缺失值极多**。有些列缺失率超过 90%（比如 `Blade_Extension` 缺了 93.7%），直接把行丢掉会损失海量样本。

### 数据结构

比赛数据分三份，按时间切分：

| 文件 | 时间范围 | 用途 |
|------|---------|------|
| `Train.csv` | 截至 2011 年底 | 训练 |
| `Valid.csv` | 2012 年 1 月 1 日 – 4 月 30 日 | 公开榜（Public LB） |
| `Test.csv` | 2012 年 5 月 1 日 – 11 月 | 决定最终名次 |

比赛同时提供了 `TrainAndValid.csv` 把前两份合并起来，方便本地实验。

主要列包括：`SalesID`（成交编号）、`SalePrice`（目标）、`MachineID`、`ModelID`、`datasource`、`auctioneerID`、`YearMade`（生产年份）、`MachineHoursCurrentMeter`（累计工时）、`UsageBand`、`saledate`（拍卖日期），以及 `fiModelDesc`、`ProductSize`、`state`、`Enclosure`、`Hydraulics` 等描述性字段。

### 评价指标

比赛使用 **RMSLE**：

$$
\text{RMSLE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\log(1+y_i) - \log(1+\hat{y}_i)\right)^2}
$$

通俗地说：先把真实价格和预测价格都取对数，再算均方误差，最后开根号。**越小越好**。

为什么要取对数？因为推土机价格跨度很大——便宜的几千美元，贵的十几万美元。如果直接算误差，一台 10 万美元的机器预测错 1 万，和一台 1 万美元的机器预测错 1 万，惩罚居然一样，这显然不合理。取对数之后，误差变成了"相对误差"，大价格的绝对误差被自动放松，指标更关注"差了几倍"而不是"差了几块钱"。

---

## 7.2 探索性数据分析（EDA）

我们先把训练集和验证集合并读进来，看整体结构：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# low_memory=False 是因为有几列类型混杂（既有数字又有文字），
# 让 pandas 一次性读全，避免分块推断类型时报警告
df = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv',
                 low_memory=False)
print(df.shape)      # (412698, 53)
print(df.head())
```

整张表 **41万行、53列**。接着我们关心两件事：**价格怎么分布**、**时间和价格是什么关系**。

```python
# 画价格直方图，看分布是否严重右偏
df["SalePrice"].hist(bins=50)
plt.xlabel("SalePrice")
plt.ylabel("count")
plt.title("Distribution of SalePrice")
plt.show()

# 画拍卖日期和价格的关系（只取前 1000 行，否则点太密看不清）
fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000], s=1)
ax.set_xlabel("saledate")
ax.set_ylabel("SalePrice")
plt.title("SaleDate vs SalePrice (first 1000 rows)")
plt.show()
```

我们看到两个现象：

1. **价格分布明显右偏**：便宜的多、极贵的少，少数豪华机器把均值拉高。这解释了为什么比赛要用对数误差。
2. **价格随日期有起伏**：说明时间是一个真实有效的信号，必须让模型知道"这台机器是哪一年卖的"，否则它会把 2006 年的价格规律套到 2012 年。

然后看缺失情况，这是决定清洗策略的关键：

```python
# 每一列的缺失比例
missing_ratio = df.isnull().sum() / len(df)
print(missing_ratio.sort_values(ascending=False).head(10))
```

打印出的结果里有几个触目惊心的数字：

```
Blade_Extension              0.937041
Enclosure_Type               0.937041
Engine_Horsepower            0.937041
Coupler_System               0.891024
UsageBand                    0.821492
MachineHoursCurrentMeter     0.642586
```

也就是说，**有十几列四分之三以上的值是空的**。正因为如此，我们最后没有删行——删了等于把 41万条砍成几万条。我们选择"补齐 + 标记"，下面会讲。

我们还看了 `YearMade` 的分布，发现它有一批异常的取值（比如机器生产年份被填成 1000 之类的占位值）。这类错值我们通过后续按类别编码让模型自己决定要不要信它——树模型会学到"1000 这个年份对应的价格非常低"，从而自然地把它们隔离出来。

---

## 7.3 数据清洗

这一步我们做了三件事：解析日期、把文字列变成数字、处理缺失。

### 第一步：解析日期并拆成多个特征

原始 `saledate` 是一列字符串（形如 `11/16/2006 0:00`），模型没法直接用。我们先把它转成真正的 datetime，再拆出年、月、日、星期几、一年中的第几天：

```python
# 先让 pandas 把 saledate 解析成时间类型
df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce", format="mixed")

# 从日期里拆出五个特征
df["saleYear"]      = df.saledate.dt.year        # 拍卖年份：最重要的时间信号
df["saleMonth"]     = df.saledate.dt.month       # 月份：有季节性
df["saleDay"]       = df.saledate.dt.day         # 日
df["saleDayOfWeek"] = df.saledate.dt.dayofweek   # 星期几：周末拍卖可能更多/更少
df["saleDayOfYear"] = df.saledate.dt.dayofyear   # 一年中的第几天

# 拆完之后原列就没用了，删掉避免重复信息
df = df.drop("saledate", axis=1)
```

为什么把一个日期拆成五个数？因为树模型只会做"这个数大于阈值吗"这样的切分，它不理解"2012 年 1 月 17 日"这种整体概念。拆开之后，「年份 ≥ 2012」就是一次干净的切分，模型很容易学到。

### 第二步：把文字列转成数字

pandas 有个很好用的机制叫 `category`：

```python
# 把所有的 object（文字）列转成 category 类型，并声明「有序」
# 有序之后 pandas 才会给每个类别分配一个整数编码
for label, content in df.items():
    if pd.api.types.is_string_dtype(content):
        df[label] = content.astype("category").cat.as_ordered()

# 看一眼编码后的样子：8 表示「排第 8 的类别」
print(df["UsageBand"].head())
print(df["UsageBand"].cat.codes.head())
```

### 第三步：处理缺失值

数值列用**中位数**填充，同时新增一列 `xxx_is_missing` 记录「这一格原本是空的」。为什么要加这个标记？因为「缺失」本身可能就是信息——比如 `MachineHoursCurrentMeter` 缺失，可能意味着卖家根本没报工时，这类机器往往更老、更便宜。让模型看到这个标记，它就能把「缺失」当作一个独立信号来学习：

```python
# 数值列：中位数填充 + 缺失标记
for label, content in df.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # 先加一列布尔标记（True 表示原本缺了）
            df[label + "_is_missing"] = pd.isnull(content)
            # 再用中位数填上。中位数比均值抗极端值
            df[label] = content.fillna(content.median())
```

类别列则先把缺失填成一个字符串 `"None"`（当作一个新的类别），再统一转成整数编码。这里 `+1` 是因为 `cat.codes` 会把 `"None"` 编成 `-1`，而我们要保证所有编码非负：

```python
# 类别列：缺失填 "None"（作为独立类别），再编码成整数
for label, content in df.items():
    if not pd.api.types.is_numeric_dtype(content):
        df[label + "_is_missing"] = pd.isnull(content)   # 同样留缺失标记
        df[label] = pd.Categorical(content).codes + 1     # -1 -> 0
```

做完这三步，`df` 里**所有列都是数字**，可以喂给模型了。特征数从 53 涨到 **102**，多出来的主要是 `_is_missing` 标记列。

---

## 7.4 特征工程

这场比赛的特征工程主要集中在日期拆分（已在清洗步骤中完成）和缺失值标记上。文字列已经通过 category 编码转成了整数，数值列已经用中位数填充并增加了 `_is_missing` 标记。此时特征已经全部是数值，可以直接送入模型。

如果你想进一步提升，可以尝试：
- 构造"机器年龄" = 拍卖年份 - 生产年份（`saleYear - YearMade`）；
- 对 `MachineHoursCurrentMeter` 等工时特征取对数，因为工时分布通常右偏；
- 对高基数的类别特征（如 `fiModelDesc`）做目标编码或频率编码。

不过社区高分方案主要靠的是把基础清洗做扎实、按时间切分验证、用随机森林在全量数据上训练，而不是靠极其复杂的特征组合。

---

## 7.5 模型训练与调优

### 关键的一步：按时间划分验证集

这一点比任何调参都重要。训练集是 2011 年及以前，验证集取 2012 年的样本，模拟"用历史预测未来"的真实场景：

```python
# 2012 年的数据当验证集，其余当训练集
df_val   = df[df.saleYear == 2012]
df_train = df[df.saleYear != 2012]
print(len(df_val), len(df_train))     # (11573, 401125)

# 拆成特征 X 和标签 y
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
X_valid, y_valid = df_val.drop("SalePrice", axis=1),   df_val.SalePrice
print(X_train.shape, X_valid.shape)   # (401125, 102) (11573, 102)
```

然后定义评估函数。注意 `mean_squared_log_error` 要求输入非负，所以我们同时打印 MAE 和 R²，从多个角度看模型：

```python
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_true, y_pred):
    """比赛的官方指标：均方根对数误差，越小越好"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds   = model.predict(X_valid)
    return {
        "Training MAE":   mean_absolute_error(y_train, train_preds),
        "Valid MAE":      mean_absolute_error(y_valid, val_preds),
        "Training RMSLE": rmsle(y_train, train_preds),
        "Valid RMSLE":    rmsle(y_valid, val_preds),
        "Training R^2":   r2_score(y_train, train_preds),
        "Valid R^2":      r2_score(y_valid, val_preds),
    }
```

### 快速试错：限制每棵树看到的样本数

41万行、102维，训练一个完整随机森林要好几分钟，调参时会非常痛苦。我们的做法是设置 `max_samples=10000`——每棵树只随机抽 1万行来长。这样单次训练从几分钟降到 32秒，可以快速判断"这个方向对不对"：

```python
from sklearn.ensemble import RandomForestRegressor

# max_samples 限制每棵树的采样量，用少量精度换取快速迭代
fast_model = RandomForestRegressor(random_state=42, max_samples=10000)
fast_model.fit(X_train, y_train)
print(show_scores(fast_model))
```

结果：

```
{'Training MAE': 5561.30, 'Valid MAE': 7177.26,
 'Training RMSLE': 0.2577, 'Valid RMSLE': 0.2936,
 'Training R^2': 0.8607, 'Valid R^2': 0.8320}
```

验证 RMSLE 是 **0.2936**，训练 0.2577，两者差距不大，说明这个规模的模型**没有明显过拟合**，瓶颈在"每棵树看的数据太少"。

### 调参的踩坑：RandomizedSearchCV 不一定能找到最优解

接着我们用 `RandomizedSearchCV` 搜参数：

```python
from sklearn.model_selection import RandomizedSearchCV

rf_grid = {
    "n_estimators":      np.arange(10, 100, 10),
    "max_depth":         [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf":  np.arange(1, 20, 2),
    "max_features":      [0.5, 1, "sqrt", "auto"],
    "max_samples":       [10000],      # 仍然限制采样，否则搜不动
}
rs_model = RandomizedSearchCV(RandomForestRegressor(random_state=42),
                              param_distributions=rf_grid,
                              n_iter=2, cv=5, verbose=True)
rs_model.fit(X_train, y_train)
print(rs_model.best_params_)
```

这里我们踩了一个坑，值得记下来：搜出来的最优参数是 `n_estimators=10, max_depth=5, min_samples_leaf=5`，**验证 RMSLE 反而涨到了 0.4495**，比不调参还差得多。原因很清楚——`n_estimators=10` 的森林太浅、`max_depth=5` 又限制了树深，模型严重欠拟合。而搜索过程用的是「限制采样 + 5折」的配置，搜索空间里恰好没有「多树、深树」这个方向，于是把这些都不满意的结果里相对最好的那个选了出来。

这说明了一个常见误区：**交叉验证选出来的参数，未必是全局最优，只是「在你给定的候选里最好」**。我们的候选集本身就把好的区域排除掉了。

### 最终模型：用经验参数在全量数据上训练

于是我们不再依赖搜索，改用另一份材料里经过上百次试验得到的经验参数，直接用全量数据训练：

```python
# 经验参数：40 棵树、允许多特征、不限制采样量
ideal_model = RandomForestRegressor(
    n_estimators=40,
    min_samples_leaf=1,
    min_samples_split=14,
    max_features=0.5,     # 每次分裂只看一半特征，增加多样性
    max_samples=None,     # 每棵树用全部训练数据
    random_state=42,
)
ideal_model.fit(X_train, y_train)     # 约 2分31秒
print(show_scores(ideal_model))
```

结果出现了明显跃升：

```
{'Training MAE': 2953.82, 'Valid MAE': 5951.25,
 'Training RMSLE': 0.1447, 'Valid RMSLE': 0.2452,
 'Training R^2': 0.9588, 'Valid R^2': 0.8818}
```

验证 RMSLE 降到 **0.2452**，验证 R² 达到 **0.8818**。对比一下就很清楚：从"每棵树 1万样本"放开到"每棵树用全部 40万样本"，误差从 0.2936 掉到 0.2452——**数据量的作用远大于参数微调**。

训练的 MAE 是 2953、验证 MAE 是 5951，验证误差是训练的两倍。这说明模型在 2012 年的数据上确实遇到了一些"没见过的行情"，但整体仍然可用；如果再加大树的数量或深度，训练误差会继续降，验证误差却可能不降反升，所以我们停在了 40 棵树。

---

## 7.6 模型测试与提交

最终我们报告的指标是：

| 指标 | 训练集 | 验证集（2012年） |
|------|--------|------------------|
| MAE | 2953.82 | 5951.25 |
| RMSLE | 0.1447 | **0.2452** |
| R² | 0.9588 | 0.8818 |

这里 RMSLE 0.2452 的含义可以这样理解：取对数之后平均差 0.245，大概是"预测价格和真实价格平均差 1.28 倍左右"（$e^{0.245} \approx 1.28$）。对于一个价格跨度从几千到十几万的拍卖市场，这个精度已经相当可以。

### 推理时的注意事项

随机森林是树模型，输入里不能有 NaN。因为我们在训练阶段已经把所有缺失都填过了，测试集也必须走**完全相同的清洗流程**，包括：同样解析日期、同样把文字转成 category（而且类别编码必须和训练集一致）、同样按中位数填充。这一步如果漏了或者顺序错了，`predict` 会直接报错或者给出离谱的结果。

### 生成提交文件

```python
# 假设 test_df 已经走完了和训练集完全相同的清洗流程
test_pred = ideal_model.predict(test_df.drop("SalePrice", axis=1, errors="ignore"))

# 生成提交文件
submission = pd.DataFrame({
    'SalesID': test_df['SalesID'],
    'SalePrice': test_pred
})
submission.to_csv('submission.csv', index=False)

# 核对清单
assert len(submission) == len(test_df), "提交行数与测试集不一致"
assert submission['SalePrice'].isnull().sum() == 0, "预测结果包含缺失值"
assert (submission['SalePrice'] > 0).all(), "价格不能是负数"
print("提交文件已生成：submission.csv")
```

---

## 7.7 小结

本章我们完成了 Blue Book 推土机拍卖价格预测的完整流程：

1. **比赛背景**：41万行、53列，预测推土机拍卖成交价 `SalePrice`，评价指标为 RMSLE（对数空间误差，关注相对误差）。
2. **EDA**：发现价格右偏、时间与价格强相关、十几列缺失率超过 75%。
3. **数据清洗**：把 `saledate` 拆成年/月/日/星期/年积日五个特征；把文字列转成有序类别再编码成整数；数值列用中位数填充并额外生成 `_is_missing` 标记；类别列缺失填 `'None'`。
4. **特征工程**：日期拆分已经提供了时间信号，可进一步构造机器年龄等特征。
5. **模型训练与调优**：**按年份切分训练集和验证集**（2011年及以前训练、2012年验证），避免"用未来预测过去"的信息泄漏；用 `max_samples=10000` 把单次训练压到 32秒来做快速实验；发现随机搜索给出的参数反而更差（0.4495），因为候选空间里没有好的区域；最终改用经验参数（40棵树、`max_features=0.5`、不限采样）在全量数据上训练。
6. **模型测试与提交**：在 2012年验证集上得到 RMSLE **0.2452**、MAE 5951、R² 0.8818；测试集必须走完全相同的清洗流程，避免类别编码不一致导致报错。

这个比赛给我们最大的启发是：**时间序列型的数据，切分方式比模型选择更重要**。如果按随机方式切分，模型会悄悄记住"2012年的价格行情"，验证分数好看但上线就崩；只有老老实实按时间切，验证集上的 0.2452 才是可信的。第二个启发是：**调参前先确认候选空间里有没有正确答案**——那次搜索之所以"越调越差"，不是因为搜索算法有问题，而是因为我们把 `n_estimators` 的上限定在了 90、把 `max_samples` 锁死在 1万，好的解根本不在搜索范围里。

下一章我们将进入时序数据部分，讲解时序数据的通用清洗与建模方法：日期特征拆分、滞后特征、滚动统计、按时间切分验证等。
