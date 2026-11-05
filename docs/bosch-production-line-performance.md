# Bosch 生产线性能预测（Bosch Production Line Performance）

## 一、比赛简介

### 1.1 比赛背景

博世（Bosch）在自己的工厂里有一条条流水线，零件从第一站开始，依次经过几十个工位，最后下线。每一站都会记录大量传感器读数：力矩、压力、温度、时间戳……绝大多数零件都顺利通过，但偶尔会有零件在某一站出问题，最终成为**不良品**。

这个比赛要解决的就是工业质检里最经典的痛点：**能不能在零件还没下线之前，就提前预测它会不会坏？** 如果能提前发现，工厂就可以及时调整工艺参数、减少废品。

数据的难点有三个：

1. **极端的类别不平衡**。不良品只占千分之几，随便全猜「良品」准确率就能到 99.9%，所以不能用准确率衡量；
2. **数据量巨大**。数值、日期、类别三种表加起来几十 GB，一台机器的内存装不下；
3. **特征极度稀疏**。一个零件不会经过所有工位，所以大部分单元格都是空的。

### 1.2 比赛目标

数据由三对文件组成（训练/测试各一份）：

- `train_numeric.csv`：数值型传感器读数，列名形如 `L3_S32_F3851`；
- `train_date.csv`：每个工位的**时间戳**，列名形如 `L3_S32_D3851`；
- `train_categorical.csv`：类别型读数。

列名的结构是 `L{产线号}_S{工位号}_F{特征号}`：

- `L0`～`L3`：4 条生产线；
- `S0`～`S51`：共 52 个工位；
- `F####`：该工位下的具体特征编号。

`train_numeric.csv` 里还有一个 `Response` 列，就是标签：`1` 表示这个零件最终是不良品，`0` 表示良品。测试集没有 `Response`，我们要为每个 `Id` 预测它是否会失败。

### 1.3 评价标准

比赛使用 **MCC（Matthews Correlation Coefficient，马修斯相关系数）**。它是一个综合考虑混淆矩阵四个格子的指标：

```
MCC = (TP × TN - FP × FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

其中 TP 是「真不良且预测不良」，TN 是「真良品且预测良品」，FP 是「误报」，FN 是「漏报」。MCC 的取值范围是 -1 到 1：

- 1 表示完美预测；
- 0 表示和随机猜差不多；
- 负数表示预测和真实**反着来**。

MCC 的好处是：即使正负样本极度不平衡，全猜多数类也只会得到 0 分，不会像准确率那样虚高。所以这场比赛分数**越高越好，满分 1**，但实际能到 0.2 以上就已经很有竞争力了。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先确认数据的整体结构。读每个文件的前几行，统计各条产线、各个工位有多少列：

```python
import pandas as pd

# 只读表头，统计每条产线(L0~L3)各有多少列
header = pd.read_csv("../input/train_categorical.csv", nrows=1)
for line in ["L0", "L1", "L2", "L3"]:
    cols = header.columns.str.startswith(line + "_")
    print(line, cols.sum())
```

结果是类别表里 L0 有 323 列、L1 有 1227 列、L2 有 159 列、L3 有 431 列，合计两千多列；数值表有 968 个特征加一个 `Response`；日期表有 1156 个时间特征。三条产线的工位数和特征密度差别很大，L1 明显最「重」。

我们接着看标签分布：

```python
# 只读 Id 和 Response，统计不良率
resp = pd.read_csv("../input/train_numeric.csv", usecols=["Id", "Response"])
resp["Response"].mean()
```

不良率只有千分之几的量级（后续 XGBoost 用 `base_score=0.005` 作为先验，也印证了这一点）。这意味着**任何只看准确率的模型都没有意义**，必须用 MCC 这种对少数类敏感的指标。

再看日期表。日期列记录的是零件经过某个工位的时间戳，单位比较特殊（材料里的分析发现时间戳大约以 10 小时为单位），而且同一零件在同一工位可能出现多次（返修、复检）。这提示我们：**日期不只是「什么时候」，还能用来算「在每一站待了多久」「两次经过之间隔了多久」**，这些「节拍」信息对判断异常非常有价值。

### 2.2 数据清洗

数据太大，直接 `read_csv` 会把内存撑爆。我们采用**分块读取 + 转存 HDF5** 的方案：每次读 10 万行，处理完按工位/产线写到 HDF5 的不同子表里，后续实验直接从 HDF5 按列读取，速度快很多。

```python
store = pd.HDFStore("bosch_data.h5")
chunksize = 10 ** 5

# 先读表头，后面按列名分块读
header_numeric = pd.read_csv("input/train_numeric.csv.zip", nrows=1).columns

for loaded in range(1, 12 * 10**5, chunksize):
    # 按块读取数值、日期、类别三种表
    chunk_numeric = pd.read_csv("input/train_numeric.csv.zip", skiprows=loaded,
                                nrows=chunksize, names=header_numeric,
                                index_col="Id", compression="zip")
    # 对 52 个工位 + 4 条产线 + all 分别处理，写入对应子表
    for name in ["S0", "S1", "S2", "L0", "L1", "all"]:
        processed = process_numeric_chunk(chunk_numeric, name)
        store.append(name, processed)
store.close()
```

**类别表的清洗。** 类别值是形如 `T0`、`T1` 的字符串，空值表示「该零件没经过这个工位」。我们只保留至少有一个非空值的行：

```python
def process_categorical_chunk(chunk_categorical, name):
    # 取出属于该工位的列
    sub_columns = chunk_categorical.columns[chunk_categorical.columns.str.contains(r'.*' + name + '_.*')]
    sub_chunk = chunk_categorical[sub_columns]
    # 整行都是空的说明这个零件没经过这个工位，删掉
    sub_chunk.dropna(how="all", inplace=True)
    return sub_chunk
```

**日期表的清洗与聚合。** 对每条产线，我们把该线上所有工位的时间戳压成 5 个统计量：最早时间、最晚时间、第一次出现的时间、最后一次出现的时间、缺失个数，再算「最晚 - 最早」得到这条线上的**总耗时**：

```python
def process_row(row):
    first = row[row.first_valid_index()]     # 第一次经过的时间
    last  = row[row.last_valid_index()]      # 最后一次经过的时间
    NAs   = row.isnull().sum()               # 缺失的工位数
    return first, last, NAs

def process_date_chunk(chunk_date, name):
    sub_chunk = chunk_date[chunk_date.columns[chunk_date.columns.str.contains(r'.*' + name + '_.*')]]
    sub_chunk.dropna(how="all", inplace=True)
    sub_chunk[name + "_date_min"] = sub_chunk.min(axis=1)
    sub_chunk[name + "_date_max"] = sub_chunk.max(axis=1)
    sub_chunk[name + "_first"], sub_chunk[name + "_last"], sub_chunk[name + "_NAs"] = \
        zip(*sub_chunk.apply(process_row, axis=1))
    # 总耗时 = 离开该线的时刻 - 进入该线的时刻
    sub_chunk[name + "_time"] = sub_chunk[name + "_date_max"] - sub_chunk[name + "_date_min"]
    return sub_chunk
```

清洗完之后，一个零件的几百上千个原始单元格就变成了几百个「统计特征」，内存和建模难度都降下来了。

### 2.3 特征工程与模型训练

**第一层特征：工位级统计。** 对每个工位，把该工位下所有数值特征压成 `min`、`max`、`mean` 三个数——注意这里是「跨特征」的统计，表示这个零件在该工位上的整体表现。对产线（L*）和全局（all）也做同样的聚合：

```python
def process_numeric_chunk(chunk_numeric, name):
    # 选出属于该工位/产线的数值列
    if name == "all":
        sub_columns = chunk_numeric.columns[chunk_numeric.columns.str.contains(r'L\d+_S\d+_F\d+.*')]
    else:
        sub_columns = chunk_numeric.columns[chunk_numeric.columns.str.contains(r'.*' + name + '_.*')]

    sub_chunk = chunk_numeric[sub_columns]
    sub_chunk.dropna(how="all", inplace=True)

    # 跨特征聚合：这个零件在该工位的最小/最大/平均读数
    sub_chunk[name + "_numerical_min"]  = sub_chunk.min(axis=1)
    sub_chunk[name + "_numerical_max"]  = sub_chunk.max(axis=1)
    sub_chunk[name + "_numerical_mean"] = sub_chunk.mean(axis=1)
    return sub_chunk
```

**第二层特征：节拍与断点。** 在日期特征的基础上，我们再加两类「前后对比」：

- `time_dt` / `time_idt`：和上一行/下一行零件的耗时差；
- `NAs_dt` / `NAs_idt`：缺失个数的前后差。

```python
# 当前零件与相邻零件的耗时差、缺失数差
chunk[name + "_time_dt"]  = chunk[name + "_time"] - chunk[name + "_time"].shift(1)
chunk[name + "_time_idt"] = chunk[name + "_time"] - chunk[name + "_time"].shift(-1)
chunk[name + "_NAs_dt"]   = chunk[name + "_NAs"]  - chunk[name + "_NAs"].shift(1)
chunk[name + "_NAs_idt"]  = chunk[name + "_NAs"]  - chunk[name + "_NAs"].shift(-1)
```

**第三层特征：批次信息 P1。** 我们把「和前后零件进入时间相同」的零件识别为同一个批次（`P1=True`），再统计每个批次有多长（`group_len`）、当前零件在批次里的序号（`ord`）。这一招很有效，因为同一个批次常常是同一炉料、同一夹具，不良往往成批出现——材料里就提到「Failures in pairs（故障成对出现）」的现象。

**特征筛选。** 特征全量有上万列，直接训练既慢又容易过拟合。我们的做法是：先在小样本（每块采样 5%）上训一个浅层 XGBoost，按 `feature_importances_` 排出前若干列，再回到全量数据只读这些列。这样既省内存又保留了信息量最大的特征：

```python
from xgboost import XGBClassifier

# 小样本预训练，只为筛特征
clf = XGBClassifier(base_score=0.005, max_depth=3)
clf.fit(X_sample, y_sample)

# 只保留重要性大于阈值的特征
important_indices = np.where(clf.feature_importances_ > 0.005)[0]
```

**最终模型。** 主线采用二阶方案的配置：`objective="binary:hinge"`（直接输出 0/1，适合 MCC）、`max_depth=10`、`eta=0.1`、`subsample=0.9`、`colsample_bytree=0.5`，训练 65 轮并用验证集早停：

```python
import xgboost as xgb

param = {
    "nthread": 8,
    "max_depth": 10,
    "eta": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.5,
    "objective": "binary:hinge",   # 直接输出 0/1，正好匹配 MCC 评价
    "booster": "gbtree",
}
num_round = 65
evallist = [(dtrain, "train"), (dval, "eval")]
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=5)
```

### 2.4 模型测试与指标报告

我们把数据按 6:2:2 切成训练、验证、测试三份：验证集用于早停，**测试集完全不参与训练**，只在最后评估一次。评估内容就是 MCC 加混淆矩阵：

```python
from sklearn.metrics import matthews_corrcoef, confusion_matrix

y_pred = bst.predict(dtest)
print("MCC:", matthews_corrcoef(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

混淆矩阵要重点看两个数：**FP（误报）和 FN（漏报）**。工厂里这两种错误的代价不一样——漏掉一个坏件可能造成售后事故，误报太多则会浪费排查成本。MCC 相当于对两者做了平衡。

如果把主线改成输出概率的 `binary:logistic`，还要多一步**阈值搜索**——因为默认 0.5 在极度不平衡的数据上几乎全判 0。我们会在 0.01～0.99 之间扫 50 个阈值，取验证集 MCC 最高的那个：

```python
import numpy as np

thresholds = np.linspace(0.01, 0.99, 50)
mcc_scores = np.array([matthews_corrcoef(y_valid, valid_prob > thr) for thr in thresholds])
best_threshold = thresholds[mcc_scores.argmax()]

# 用最优阈值把概率变成 0/1
test_pred = (test_prob > best_threshold).astype(np.int8)
```

作为对照，公开的 XGBoost 入门方案（只用 5% 采样筛特征 + 全量训练）在 3 折交叉验证上 MCC 约 **0.193**，公开榜 **0.209**；我们在这个基础上补齐了日期聚合、批次特征和按工位分组的统计，最终提交进入前二名。

最后按官方格式写提交文件：`Id` + `Response`，并用 gzip 压缩以减小体积：

```python
sub = pd.read_csv("../input/sample_submission.csv", index_col=0)
sub["Response"] = test_pred
sub.to_csv("submission.csv.gz", compression="gzip")
```

---

## 三、总结

本文中我们完成了 Bosch 生产线性能预测的完整流程：首先通过 EDA 确认了数据的结构——4 条产线、52 个工位、上万列特征、千分之几的不良率，以及日期特征约以 10 小时为单位、同一零件可能多次经过同一工位；其次我们用「分块读取 + HDF5 转存」解决了内存问题，删掉了整行为空的记录，并把日期压成最早/最晚/首末时间/缺失数/总耗时；接着我们对每个工位和产线做数值的 min/max/mean 聚合，补充了前后零件的耗时差、缺失差和批次（P1/ord/group_len）特征，再用小样本 XGBoost 做特征筛选，最后以 `binary:hinge` 的 XGBoost 训练并早停；最后我们用 MCC 和混淆矩阵在独立测试集上评估，概率版本还额外搜索了最优阈值，公开参考方案的交叉验证 MCC 约为 0.193。

这个比赛给我们最大的启发是：**评价指标决定了模型该怎么设计**。在千分之几的不良率下，准确率是个假指标，MCC 才是真指标；而 MCC 又决定了我们不该用默认 0.5 阈值、不该全猜多数类，而应该专门去优化「那千分之几」。另外，**工业数据的价值大量藏在「时间」和「批次」里**——同一批零件一起坏，比单个传感器读数更能说明问题。