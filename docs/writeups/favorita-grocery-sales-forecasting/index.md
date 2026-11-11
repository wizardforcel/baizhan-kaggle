# Favorita 杂货销售预测（Grocery Sales Forecasting）

## 一、比赛简介

### 1.1 比赛背景

Corporación Favorita 是厄瓜多尔的一家大型连锁超市。这个比赛要求参赛者根据历史销售记录，预测未来一段时间里**每家门店、每种商品、每一天能卖出多少件**。

这是典型的**时间序列预测**问题，但比教科书里的单条时间序列复杂得多：数据里有几千家门店、几千种商品，组合起来是几十万条「门店-商品」序列，每条序列的销量又受促销、节假日、发薪日、油价等因素影响。难点主要有三：

- **数据量巨大**：训练集是上亿行级别的日销售记录，普通电脑的内存和计算都吃不消；
- **大量零销量**：很多商品在某些门店某天一件都没卖出去，序列里全是 0，模型很容易学成「永远预测 0」；
- **预测目标特殊**：比赛用的是「对数误差」，而且对易腐商品（perishable）额外加权，不是普通的最小二乘。

### 1.2 比赛目标

给定门店编号、商品编号、日期，预测这一天的销量 `unit_sales`。

数据文件包括：

- `train.csv`：历史日销量，列有 `date`（日期）、`store_nbr`（门店号）、`item_nbr`（商品号）、`unit_sales`（销量，可为负，表示退货）、`onpromotion`（当天是否促销）；
- `test.csv`：待预测的「门店 × 商品 × 日期」组合，共 **3,370,464** 行，覆盖 2017 年 8 月 16 日起的 16 天；
- `items.csv`：4100 种商品的属性，包括 `family`（品类）、`class`（类别）、`perishable`（是否易腐，1/0）；
- `stores.csv`、`oil.csv`、`holidays_events.csv`、`transactions.csv`：门店信息、每日油价、节假日、交易笔数等辅助数据。

提交文件要求为每个 `id` 填一个 `unit_sales` 预测值。

### 1.3 评价标准

比赛使用 **NWRMSLE（Normalized Weighted Root Mean Squared Logarithmic Error，归一化加权均方根对数误差）**。拆开看：

- **对数**：先对真实值和预测值都做 `log1p`（即 `log(1+x)`），再算误差。这样「从 1 件预测成 2 件」和「从 100 件预测成 200 件」的惩罚差不多，符合零售业「相对误差比绝对误差更重要」的直觉；
- **加权**：易腐商品（生鲜、面包等）的误差权重是 **1.25**，非易腐商品是 **1.0**。因为生鲜卖不掉就浪费，预测不准的代价更高；
- **归一化**：最后用权重之和做分母，使分数不受样本量影响。

公式可以写成：

```
NWRMSLE = sqrt( Σ w_i * (log1p(预测_i) - log1p(真实_i))² / Σ w_i )
```

**分数越低越好**。排行榜上 0.514 已经是很强的成绩，而「全部预测 0」这种朴素 baseline 会得到接近 1 的分数。

---

## 二、解题过程

我们的主线方案来自公开的强基线（Ceshine Lee 的 LightGBM 思路）：**把「未来 16 天」拆成 16 个独立的小预测任务，每天训练一个 LightGBM 回归模型**；特征只用「过去几天的平均销量」和「过去/未来的促销标记」；目标值先做 `log1p` 变换。最后再把 LightGBM、XGBoost、线性回归三个模型加权融合。我们按四步走。

### 2.1 数据探索（EDA）

我们首先读入训练集。训练集太大，直接全读会爆内存，所以我们只保留要用的列、把 `unit_sales` 在读取时就转成 `log1p`，并且用 `skiprows` 跳过 2016 年以前的数据：

```python
import pandas as pd
import numpy as np

df_train = pd.read_csv(
    '../input/train.csv',
    usecols=[1, 2, 3, 4, 5],          # 只取 date/store/item/unit_sales/onpromotion
    dtype={'onpromotion': bool},       # 促销标记直接读成布尔型，省内存
    # 读入时就把销量转成 log1p；销量可能为负（退货），负值一律当 0
    converters={'unit_sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0},
    parse_dates=['date'],
    skiprows=range(1, 66458909)        # 跳过 2016-01-01 之前的行
)
print(df_train.shape)
# (59038132, 5) —— 仅 2016 年以后就有约 5900 万行
```

**为什么要对销量取 `log1p`？** 因为评价指标本身就是对数误差。如果模型直接在原始销量上做回归，少数「一天卖几千件」的爆款会把损失主导掉；取对数后，大小销量被拉到相近的量级，模型会更关注「相对偏差」。而 `log1p` 里的 `+1` 是为了让 0 销量也能取对数（`log(1+0)=0`），否则会有无穷小的对数。

接着看商品表，确认商品属性和易腐标记：

```python
items = pd.read_csv('../input/items.csv').set_index('item_nbr')
print(items.shape)      # (4100, 3)：family、class、perishable
print(items['perishable'].mean())   # 易腐商品占比，用于理解加权的影响
```

再看测试集的规模和时间范围：

```python
df_test = pd.read_csv(
    '../input/test.csv',
    usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=['date']
).set_index(['store_nbr', 'item_nbr', 'date'])

print(df_test.shape)                       # (3370464, 2)
print(df_test.index.get_level_values('date').unique()[:5])
# 2017-08-16 起连续 16 天
```

EDA 得出的几个关键事实：

1. **训练集的时间跨度很长**，但越靠近预测期（2017 年 8 月）的数据越有用。所以我们只保留 **2017 年 5 月 31 日以后**的数据，约 812 万行，既够训练又跑得动；
2. **测试集有 16 天**，而且每一天的促销状态是已知的（`onpromotion` 在测试集里也给了），这是极强的特征；
3. **零销量极多**：很多「门店-商品」组合在某些天完全没有销售，这解释了为什么后面要把数据整理成「行为门店-商品、列为日期」的宽表，缺失日期补 0。

### 2.2 数据清洗与特征工程

我们把数据重排成一张**宽表**：每一行是一个「门店-商品」组合，每一列是一个日期，格子里是（对数）销量。这样「过去 3 天、7 天、14 天的平均销量」就是直接对列切片求均值，写起来非常简洁。

```python
# 只保留 2017-05-31 起 11 周的数据
df_2017 = df_train[df_train.date.isin(
    pd.date_range("2017-05-31", periods=7 * 11))].copy()
del df_train                      # 原始大表用完立刻释放，省内存

# 变成宽表：行 = (store_nbr, item_nbr)，列 = 日期
df_2017 = df_2017.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]] \
                 .unstack(level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)
print(df_2017.shape)              # 约 15.7 万个门店-商品组合 x 77 天
```

促销信息也用同样的方式整理成宽表，并且**把训练期和测试期拼在一起**，这样在构造特征时，可以同时取到「过去的促销」和「未来 16 天的促销」：

```python
# 训练期的促销宽表
promo_train = df_2017_promo.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]] \
                           .unstack(level=-1).fillna(False)
promo_train.columns = promo_train.columns.get_level_values(1)

# 测试期的促销宽表
promo_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_test.columns = promo_test.columns.get_level_values(1)
promo_test = promo_test.reindex(promo_train.index).fillna(False)

# 沿列方向拼接：一张表里既有历史促销，也有未来促销
promo_2017 = pd.concat([promo_train, promo_test], axis=1)
```

然后是特征构造函数。我们的特征是**刻意做到极简**的：只有 3 个历史均值 + 1 个历史促销计数 + 16 个未来促销标记，一共 20 列。为什么不多加特征？因为在几百万行、几十万条序列的情况下，特征一多，训练和调参的成本会成倍增加，而「近期均值 + 促销」已经能抓住销量预测的主要矛盾：

```python
from datetime import date, timedelta

def get_timespan(df, dt, minus, periods):
    """取从 dt 往前数 minus 天开始、长度为 periods 天的一段列"""
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods)]

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        # 过去 3、7、14 天的平均（对数）销量：捕捉短期水平和周规律
        "mean_3_2017":  get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017":  get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        # 过去 14 天一共促销了几天
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
    })
    # 未来 16 天每天是否促销（测试集已给出，属于"已知的未来信息"）
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[t2017 + timedelta(days=i)].values.astype(np.uint8)

    if is_train:
        y = df_2017[pd.date_range(t2017, periods=16)].values   # 未来 16 天真实销量
        return X, y
    return X
```

**训练集是怎样构造的？** 因为要预测「未来 16 天」，我们用滑动窗口造了 4 个训练样本块：从 2017-06-21 开始，每隔 7 天取一个起点，每个起点给出「未来 16 天的真实销量」当作标签。这样模型见过 4 种「从不同时间点往后预测两周」的情形，泛化更稳：

```python
t2017 = date(2017, 6, 21)
X_l, y_l = [], []
for i in range(4):                       # 4 个起点，间隔 7 天
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(t2017 + delta)
    X_l.append(X_tmp)
    y_l.append(y_tmp)

X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
# 形状约 (627160, 20)：62 万行样本、20 个特征

# 验证集：预测 2017-07-26 起的 16 天；测试集：预测 2017-08-16 起的 16 天
X_val,  y_val  = prepare_dataset(date(2017, 7, 26))
X_test         = prepare_dataset(date(2017, 8, 16), is_train=False)
```

验证集的切分方式是**按时间顺序**：训练用 6 月下旬～7 月中旬的窗口，验证用 7 月 26 日开始的 16 天，测试用 8 月 16 日开始的 16 天。绝不能随机打乱——时间序列里「用未来预测过去」就是作弊，线下分数会虚高，线上会打脸。

### 2.3 模型训练与超参数调优

我们训练 **16 个 LightGBM 模型**，第 i 个模型专门预测未来第 i 天的销量。这样每个模型只需要关心「某一天卖多少」，任务更简单，也方便把「第 i 天是否促销」作为强特征。

LightGBM 的参数是公开基线里调好的：学习率 0.05、特征采样率 0.75、行采样率 0.75、每轮都做 bagging、损失用 L2（因为我们已经在 `log1p` 空间里回归，L2 等价于优化对数误差）、最多 1000 轮、验证分数 50 轮不提升就早停：

```python
import lightgbm as lgb

params = {
    'objective': 'regression',
    'metric': 'l2',
    'learning_rate': 0.05,
    'feature_fraction': 0.75,     # 每棵树随机用 75% 的特征，抗过拟合
    'bagging_fraction': 0.75,     # 每棵树随机用 75% 的样本
    'bagging_freq': 1,
    'num_threads': 4,
}
MAX_ROUNDS = 1000

val_pred, test_pred = [], []
# 对 16 天各训练一个模型
for i in range(16):
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        # 易腐商品权重 1.25，非易腐 1.0，与评价指标的加权方式一致
        weight=pd.concat([items["perishable"]] * 4) * 0.25 + 1
    )
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=items["perishable"] * 0.25 + 1
    )
    bst = lgb.train(
        params, dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval],
        early_stopping_rounds=50,
        verbose_eval=50
    )
    val_pred.append(bst.predict(X_val,  num_iteration=bst.best_iteration))
    test_pred.append(bst.predict(X_test, num_iteration=bst.best_iteration))
```

**关于样本权重**：注意这里给 LightGBM 传了 `weight` 参数，易腐商品权重是 `perishable * 0.25 + 1 = 1.25`，非易腐是 1.0。这不是可选项——如果不加权，模型会按普通 MSE 优化，和比赛的 NWRMSLE 不一致，线上分数会吃亏。**让训练目标和评价指标保持一致**，是这个比赛最重要的经验之一。

训练日志显示第 1 天的模型在第 712 轮早停，验证集 L2 为 **0.326938**。特征重要性排名前三的是 `mean_14_2017`、`mean_7_2017`、`mean_3_2017`，其次是 `promo_0` 和 `promo_14_2017`——也就是说，**「最近两周卖了多少」和「当天是否促销」是预测的两根支柱**，这与业务直觉完全吻合。

**超参数调优**主要在三个地方：一是历史窗口的长度（试过 3/7/14 天，也试过 30 天，最终三档全保留效果最好）；二是 LightGBM 的学习率和树的数量（0.05 配 1000 轮 + 早停，比 0.1 更稳）；三是样本权重里的易腐系数（0.25 对应 1.25 倍权重，直接对齐比赛指标）。

### 2.4 模型测试与指标报告

我们实现 NWRMSLE 来评估验证集：

```python
def NWRMSLE(y, pred, w):
    """y、pred 都是原始销量（未取对数）；w 是每个样本的权重"""
    y    = np.clip(y, 0, None)          # 销量不能为负
    pred = np.clip(pred, 0, None)
    # 先 log1p，再算加权平方误差，除以权重之和后开根号
    score = np.nansum(w * ((np.log1p(pred) - np.log1p(y)) ** 2)) / w.sum()
    return np.sqrt(score)
```

单模型在验证集上的 NWRMSLE 大约在 0.6 附近。随后我们做了**模型融合**：把 LightGBM、XGBoost 和线性回归三个模型的预测按权重加权平均。之所以融合线性模型，是因为树模型擅长捕捉「促销带来的跳变」和「非线性交互」，而线性回归擅长捕捉「整体销量水平」的平滑趋势，两种结构的误差不完全相关，混在一起能互补：

```python
# 三个模型各自的测试集预测（形状相同：167515 个门店-商品组合 x 16 天）
lgb_pred = pd.read_csv('../input/dataset2/lgbSubmitY.csv', index_col=False)
xgb_pred = pd.read_csv('../input/aggr-dataset/xgbSubmitY.csv', index_col=False)
lr_pred  = pd.read_csv('../input/aggr-dataset/lrSubmitY.csv',  index_col=False)

# 按 0.45 / 0.45 / 0.10 融合（权重用验证集搜索得到）
combined = xgb_pred * 0.45 + lgb_pred * 0.45 + lr_pred * 0.10
```

融合后的排行榜成绩是 **LB 0.514**，比单用 LightGBM 有明显提升。最后把「宽表格式」的预测还原成提交文件要求的「一行一个 id」长表。这里有两个细节：一是预测值在 `log1p` 空间里，要 `expm1` 变回原始销量；二是把结果裁剪到 0～1000，防止极端预测值：

```python
import gc
gc.collect()

# 宽表 -> 长表：index 是 (store_nbr, item_nbr)，列是日期
df_preds = pd.DataFrame(
    combined.values, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

# 用测试集的 id 做左连接，保证顺序和行数与 test.csv 完全一致
submission = df_test[["id"]].join(df_preds, how="left").fillna(0)

# log1p 的逆变换 + 裁剪到合理范围
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('submission.csv', float_format='%.4f', index=None)
```

---

## 三、总结

本文中我们完成了 Favorita 杂货销售预测的完整流程。我们首先做了 EDA，确认测试集有 337 万行、覆盖 16 天，只有 2017 年 5 月以后的数据与预测期最相关，并且大量「门店-商品-日期」组合的销量为 0。其次我们做了数据清洗与特征工程：在读取时就把销量转成 `log1p`，把长表重排成「行是门店-商品、列是日期」的宽表，并只保留 3/7/14 天历史均值、14 天促销计数和未来 16 天促销标记这 20 个特征。接着我们按时间顺序切分训练/验证，为未来 16 天各训练一个 LightGBM 模型，样本权重设为易腐商品 1.25、非易腐 1.0，与 NWRMSLE 指标严格对齐，并用早停控制轮数。最后我们把 LightGBM、XGBoost、线性回归按 0.45/0.45/0.10 融合，取得 **LB 0.514** 的成绩，再经 `expm1` 逆变换和裁剪生成提交文件。这个比赛给我们的最大启发是：**训练损失必须和比赛指标一致（对数变换 + 易腐加权）**，而在海量时间序列上，滑动窗口构造样本、按预测日拆分模型、树模型与线性模型互补融合，是三条非常实用的工程套路。