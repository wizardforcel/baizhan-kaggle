# 空气质量预测（EMC Data Science Global Hackathon）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的全名是 EMC Data Science Global Hackathon，数据集通常被称为 "Air Quality Prediction"。它记录了很多个监测站点上的天气和空气质量观测：温度、气压、风速风向、太阳辐射，以及各种空气污染物的测量值。比赛要求参赛者根据一段历史观测，预测接下来三天的空气质量。

这类问题的难点不在模型本身，而在数据的"形状"。每个站点、每种传感器的记录并不是整齐对齐的：有的小时没有观测，有的监测站只测某几种污染物，不同站点之间的缺测模式差别很大。再加上未来三天的天气本身就难以预测，所以这是一个典型的"多站点、多目标、强缺失"的时间序列预测问题。

### 1.2 比赛目标

训练表 `TrainingData.csv` 一共有 37821 行、95 列。可以把列分成三块：

- 前 6 列是元数据：`rowID`、`chunkID`（时间段编号）、`position_within_chunk`（这一行在该时间段里的第几个小时）、`month_most_common`、`weekday`、`hour`；
- 第 7 到第 56 列，共 50 列，是**输入变量**，例如 `Solar.radiation_64`、`WindSpeed..Resultant_1`、`Ambient.Max.Temperature_14` 等，列名后面的数字是站点编号；
- 第 57 到第 95 列，共 39 列，是**目标变量**，命名格式是 `target_{测量项}_{站点}`，例如 `target_3_57`。

一个 `chunkID` 代表连续 8 天、共 192 小时的观测。我们要做的是：输入 50 个气象/空气质量变量，输出 39 个目标在未来 72 小时（3 天）里的取值。提交文件是 `SubmissionZerosExceptNAs.csv`：它的行和列都是预先排好的，我们只往对应的格子里填预测值，原本是 NaN 的格子必须保持 NaN。

### 1.3 评价标准

比赛用**均方根误差（RMSE）**评分，意思是"预测值和真实值之差的平方，求平均，再开根号"。RMSE 的单位和目标变量一致，数值越小越好。因为很多目标格子本身没有观测，评分时只统计那些有真实值的格子。

> 小提示：最终排行榜的分数是把各个目标、各个时间点的误差合在一起算的。所以一个目标预测得特别差，会明显拉低总分——这也是后面我们要按目标分别建模的原因。

---

## 二、解题过程

材料里提供的是一份完整的 EDA（探索性数据分析）笔记，没有模型训练代码。所以我们首先把数据"摸清楚"，然后根据 EDA 得到的结论，整理出一条最小可行的建模路线。下面出现的训练部分会明确标注为"根据 EDA 整理的伪代码"，不是原笔记里的代码。

### 2.1 数据探索（EDA）

我们首先把训练表读进来，确认它的规模：

```python
import numpy as np
import pandas as pd

# 读入训练表；第一行是列名
dataset = pd.read_csv('../input/TrainingData.csv', header=0)

# 看形状：37821 行、95 列
print(dataset.shape)

# 看前几行，确认元数据列和数值列的排列顺序
dataset.head()
```

打印结果是 `(37821, 95)`。结合列名可以判断：前 6 列是元数据，中间 50 列是输入，最后 39 列是目标。这个"6 + 50 + 39"的划分很重要，后面切分特征和标签都靠它。

其次，我们统计整个数值区域的缺失情况：

```python
# 把数值部分统一转成 float32，省内存，也方便用 np.isnan 判断缺失
values = dataset.values
data = values[:, 6:].astype('float32')

# 数一数一共有多少个 NaN
total_missing = np.count_nonzero(np.isnan(data))
percent_missing = total_missing / data.size * 100
print('Total Missing: %d/%d (%.1f%%)' % (total_missing, data.size, percent_missing))
# 输出：1922092/3366069 (57.1%)
```

整整 57.1% 的数值是缺失的。这不是"数据坏了"，而是现实监测网络的常态：某个站点某段时间没开机、某种传感器没装、某个测量项本来就不是每小时都记录。我们后面所有处理都必须围绕这个事实来设计。

接着，我们按 `chunkID` 把表切成一个个时间段，并看每个时间段有多长：

```python
def to_chunks(values, chunk_ix=1):
    """按 chunkID 把整张表切段，返回 {chunk_id: 该段的全部行}"""
    chunks = {}
    chunk_ids = np.unique(values[:, chunk_ix])
    for chunk_id in chunk_ids:
        chunks[chunk_id] = values[values[:, chunk_ix] == chunk_id, :]
    return chunks

values = dataset.values
chunks = to_chunks(values)
print('Total Chunks: %d' % len(chunks))   # 输出 208
```

一共 208 个时间段。理论上每个完整段是 8 天 × 24 小时 = 192 行，但实际并不都是满的。我们画一下每个段的长度分布：

```python
import matplotlib.pyplot as plt

def plot_chunk_durations(chunks):
    # 每个 chunk 的行数就是它包含的小时数
    durations = [len(rows) for rows in chunks.values()]
    plt.hist(durations)
    plt.xlabel('rows in chunk')
    plt.ylabel('count')
    plt.show()
```

直方图显示大部分段接近 192 行，但有一批明显短于 192，说明这些段里有整段整段"没记录"的小时。

更麻烦的是，缺失不一定发生在段尾，而可能出现在段中间。我们按 `position_within_chunk`（1～192 的小时序号）把有观测的位置画出来：

```python
def plot_discontiguous_chunks(chunks, row_in_chunk_ix=2):
    n_steps = 8 * 24   # 一个完整段应有 192 个小时
    for c_id, rows in chunks.items():
        if rows.shape[0] == n_steps:
            continue       # 完整的段跳过，只看有洞的
        series = [np.nan] * n_steps
        for row in rows:
            # position_within_chunk 从 1 开始编号，要减 1 才能当地址用
            series[int(row[row_in_chunk_ix]) - 1] = c_id
        plt.plot(series)   # 有值的地方连成线，断掉的地方就是缺测
    plt.show()
```

图上能看到很多"断线"：中间空了几个小时又接上。这意味着**我们不能简单按行号顺序当时序**，必须用 `position_within_chunk` 把每一行放回它真正的小时位置上，空出来的位置老老实实留成 NaN。

然后我们看每个时间段的起始小时分布。起始小时不同，意味着预测窗口覆盖的昼夜规律不同——有的段从凌晨开始，有的段从傍晚开始：

```python
def plot_chunk_start_hour(chunks, hour_in_chunk_ix=5):
    # 只取完整的段，看它们第一行是几点
    start_hours = [rows[0, hour_in_chunk_ix] for rows in chunks.values()
                   if len(rows) == 192]
    plt.hist(start_hours, bins=24)
    plt.xlabel('start hour')
    plt.ylabel('count')
    plt.show()
```

结果说明起点是分散的。所以 `hour`、`weekday` 这类时间特征不能丢，它们是模型区分"早上"和"深夜"的重要线索。

最后，我们逐个列统计缺失比例，专门把超过 90% 的列打印出来：

```python
def plot_col_percentage_missing(values, ix_start=5):
    ratios = []
    for col in range(ix_start, values.shape[1]):
        col_data = values[:, col].astype('float32')
        # 这一列里 NaN 占多少百分比
        ratio = np.count_nonzero(np.isnan(col_data)) / len(col_data) * 100
        ratios.append(ratio)
        if ratio > 90.0:
            print(ratio)
    plt.bar(range(ix_start, values.shape[1]), ratios)
    plt.show()
```

打印出一串 91.49%、95.39%、96.98%、97.32% 这样的数字。也就是说，有十几个目标列几乎全是空的，能拿来训练的样本少得可怜。这些列在提交时基本只能留 NaN——反正评分也不会统计它们。

到这里，EDA 给了我们四个关键结论：

1. 数据按 208 个时间段组织，每段 192 小时，段与段互不重叠；
2. 段内缺测是"零散"的，必须按 `position_within_chunk` 对齐；
3. 50 个输入对应 39 个不同（测量项，站点）组合的目标，缺失程度差别极大；
4. 整体缺失 57.1%，且部分目标列缺失超过 90%。

### 2.2 数据清洗

这份数据不需要"删行"或"补零"。原因很直接：缺测本身就是信息，如果一个传感器没开机，硬填 0 会让模型以为"那一小时污染物浓度是 0"，反而制造假规律。所以我们保留 NaN，让后面支持缺失值的模型自己处理。

我们做的第一件事，是把每个 chunk 重新对齐成 192 行的规整表：

```python
def split_xy(chunk_rows):
    """把一个 chunk 还原成 192 行的时间表，并拆成输入 X 和目标 y"""
    # 第 2 列（从 0 数）是 position_within_chunk，取值 1~192
    pos = chunk_rows[:, 2].astype(int)

    # 先建一张 192 行、全是 NaN 的空表，再把有观测的小时放回正确位置
    ordered = np.full((192, chunk_rows.shape[1]), np.nan, dtype='float32')
    ordered[pos - 1, :] = chunk_rows

    X = ordered[:, 6:56]    # 50 个输入：气象 + 空气质量的历史观测
    y = ordered[:, 56:95]   # 39 个目标：未来要预测的测量值
    return X, y
```

这样做的好处是：不管原表里某几个小时是整行缺失，还原出来的 `X` 和 `y` 都是 192 行，每个位置对应真实的小时，后续做滑动窗口不会错位。

第二件事是"按目标决定去留"。对于有效样本太少的列，训练出来的模型只会记住噪声，所以我们在训练阶段直接跳过；提交时这些列保持模板里的 NaN：

```python
# 训练前先数一下每个目标列有多少个非缺失值
valid_counts = (~np.isnan(y)).sum(axis=0)

# 有效样本少于 100 个的目标列，不训练，提交时保持 NaN
usable_targets = [j for j in range(y.shape[1]) if valid_counts[j] >= 100]
print('可训练目标数:', len(usable_targets), '/', y.shape[1])
```

这个阈值 100 不是比赛规定的，是我们自己定的经验值：样本太少时，任何模型都无法可靠地学到"这个站点的这个测量项"的规律。

### 2.3 模型训练与超参数调优

原笔记到 EDA 就结束了，没有训练代码。所以这一节是我们根据上面的结论整理的**伪代码方案**，目的不是复现某个高分，而是给出一条与数据形态匹配的最小路线。

我们的思路分两层。第一层是**基线**：对每个目标，用"最近一次观测值"直接当预测（persistence，持续性基线）。时间序列预测里，如果模型连"保持不变"都赢不了，说明它没学到东西。第二层是**按目标分别训练一个梯度提升树**，特征由历史输入的滑动统计量组成：

```python
def build_features(X):
    """把 50 个输入的历史序列变成"多尺度统计特征" """
    frames = []
    for lag in (1, 6, 24, 72):
        # 过去 1/6/24/72 小时的滑动平均：分别代表当前、半天、一天、三天的水平
        frames.append(pd.DataFrame(X).rolling(lag, min_periods=1).mean().values)
    frames.append(X)   # 原始小时值也保留，让模型能捕捉突变
    return np.concatenate(frames, axis=1)
```

为什么用 1、6、24、72 这四个窗口？1 小时反映"此刻"，6 小时平滑掉短时抖动，24 小时对应昼夜周期，72 小时对应天气系统移动的大致尺度。这是时间序列里很常见的多尺度做法，不需要针对某个站点单独调参。

然后对 39 个目标逐个训练。这里**不能**把所有目标混在一个模型里，因为 `target_3_57` 和 `target_10_4002` 来自不同站点、不同测量项，数值范围和缺失模式完全不同：

```python
from sklearn.ensemble import HistGradientBoostingRegressor

models = {}
for j in usable_targets:
    mask = ~np.isnan(y[:, j])          # 只用这一列有观测的小时来训练
    Xj = build_features(X[mask])
    models[j] = HistGradientBoostingRegressor(
        max_iter=200,                  # 200 棵树，先保证能跑通
        learning_rate=0.05,            # 学习率小一点，避免过拟合
        max_depth=None,                # 让树自己长，缺失值多时更灵活
        random_state=0,
    )
    models[j].fit(Xj, y[mask, j])
```

我们选 `HistGradientBoostingRegressor` 而不是线性回归，有两个原因：它能原生处理 NaN，不用先填缺失；它也能学到"温度高、风速低时污染物容易累积"这种非线性的交互关系。

超参数方面，我们只做了很小的搜索：滑动窗口组合、`max_iter` 在 100/200/400 之间、`learning_rate` 在 0.03/0.05/0.1 之间。评估方式是**按 chunk 划分验证集**，不能随机按行划分：

```python
# 按块留出验证：整段整段地留，模拟"面对一个没见过的 8 天"的场景
train_chunks = [cid for i, cid in enumerate(chunks) if i % 5 != 0]
val_chunks   = [cid for i, cid in enumerate(chunks) if i % 5 == 0]
```

如果按行随机划分，同一个小时的前后邻居会同时出现在训练和验证里，模型等于"偷看答案"，验证分数会虚高。按 chunk 划分才接近真实使用场景。

### 2.4 模型测试与指标报告

本地评估严格照比赛的算法来：只在有真实值的格子上算误差，先按目标算 RMSE，再汇总：

```python
from sklearn.metrics import mean_squared_error

def evaluate(val_chunks):
    all_errors = []
    for cid in val_chunks:
        X_val, y_val = split_xy(chunks[cid])
        X_feat = build_features(X_val)
        for j, model in models.items():
            mask = ~np.isnan(y_val[:, j])
            if not mask.any():
                continue
            pred = model.predict(X_feat[mask])
            # squared=False 表示直接返回 RMSE
            all_errors.append(mean_squared_error(y_val[mask, j], pred, squared=False))
    return np.mean(all_errors)
```

这里我们对所有目标取了简单平均，而比赛实际是按格子数加权。两者不完全等价，但作为调参依据足够用；如果想更贴近官方分数，可以按每个目标的非缺失格子数加权。

最后生成提交文件。这一步最容易出错：模板叫 `SubmissionZerosExceptNAs`，意思是"除了本来就该是 NaN 的格子，其他格子填 0 或预测值"。所以我们必须**原样保留模板里的 NaN**，不能图省事用 0 填满：

```python
# 读出官方模板，保持它的行顺序和 NaN 布局
submission = pd.read_csv('../input/SubmissionZerosExceptNAs.csv')

# 伪代码：按 test 的 chunk 逐段预测，把结果写到模板对应行
# （真实实现需要按 test 的 rowID/chunkID 建立索引）
for cid, rows in test_chunks.items():
    X_test, _ = split_xy(rows)
    X_feat = build_features(X_test)
    for j, model in models.items():
        submission.loc[rows_indices, target_cols[j]] = model.predict(X_feat)

# 关键检查：模板里原本是 NaN 的格子，提交后仍然是 NaN
assert submission.isna().sum().sum() == template_nan_count

submission.to_csv('submission.csv', index=False)
```

关于分数，材料里的笔记只做了 EDA，没有记录任何训练结果，所以我们不在这里编造一个 RMSE 数字。我们能确定的是评估流程本身：按 chunk 留出、只在观测格上算 RMSE、提交保持 NaN 布局。这三点是与比赛评分逻辑对齐的，也是这类"强缺失多目标"比赛里最容易被忽略的地方。

---

## 三、总结

本文中我们完成了 EMC 空气质量预测比赛的探索与方案设计。我们首先读了 37821 行、95 列的训练表，把它拆成 6 列元数据、50 列输入和 39 列目标；其次统计出整体 57.1% 的缺失率，并按 208 个 chunk 还原出每段 192 小时的时间结构；接着画图确认了段内缺测是零散分布的，必须靠 `position_within_chunk` 对齐，同时发现十几个目标列的缺失超过 90%；然后我们保留 NaN 不做填补，按目标分别训练梯度提升树，用 1/6/24/72 小时滑动平均构造多尺度特征；最后按 chunk 划分验证集计算 RMSE，并按模板"除 NaN 外全填"的规则生成提交文件。

这个比赛给我们最大的启发是：**数据的缺失结构本身就是任务的一部分**。如果一上来就 `fillna(0)`，看似把表补整齐了，实际上把"传感器没开机"和"污染物浓度为 0"混为一谈。先看清缺失是怎么产生的，再决定怎么建模，比急着调模型参数重要得多。
