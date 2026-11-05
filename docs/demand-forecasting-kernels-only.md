# 商店商品需求预测（Demand Forecasting）

## 一、比赛简介

### 1.1 比赛背景

零售行业最怕两件事：货进少了，客人买不到，生意跑了；货进多了，卖不掉，压库存、占资金。所以「明天某个门店的某个商品能卖多少件」是零售业每天都要回答的问题。这个比赛给出的就是一份真实的销售流水：**50 种商品、10 家门店、2013 年 1 月到 2017 年 12 月，共 5 年**，每一天每个「门店 + 商品」组合的销量都记在表里。

这是一个典型的**时间序列预测**问题，但它的难点不在「预测一条曲线」，而在「同时预测 500 条曲线」（50 商品 × 10 门店）。数据里有明显的规律：

- **趋势**：全店年均销量从 2013 年的 43.5 一路上升到 2017 年的 58.8，整体在增长；
- **季节性**：7 月平均销量最高（约 67.0），1、2 月最低（约 35.5、39.4），一年一个循环；
- **周内规律**：周末和工作日的销量明显不同；
- **商品与门店差异**：不同商品、不同门店的销量水平差很多。

把这些规律显式地做成特征喂给模型，是这类比赛取胜的关键。

### 1.2 比赛目标

训练表 `train.csv` 有四列：`date`（日期）、`store`（门店编号 1～10）、`item`（商品编号 1～50）、`sales`（当天销量）。测试表 `test.csv` 给出 2018 年 1～3 月的 `date / store / item`，要我们预测 `sales`。训练集 913,000 行，测试集 45,000 行（500 个组合 × 90 天）。

### 1.3 评价标准

比赛使用 **SMAPE（对称平均绝对百分比误差，Symmetric Mean Absolute Percentage Error）**，**越低越好**。它的算法是：

$$
\text{SMAPE} = \frac{200}{n}\sum \frac{|预测 - 真实|}{|预测| + |真实|}
$$

用大白话说：把每条样本「误差占两者平均值大小的比例」算出来，取平均再乘 100，结果就是百分比。它和普通百分比误差的区别是分子分母都带绝对值、且分母是「预测 + 真实」，所以**不会因为真实值很小而爆炸**，也不会偏向高估或低估。SMAPE 为 13 大致表示「平均误差是真实值的 13% 左右」。

---

## 二、解题过程

我们首先把训练集和测试集拼在一起统一处理日期、观察趋势与季节性（EDA），其次构造日期特征、滞后特征、滑动平均特征和指数加权特征（特征工程与清洗），接着用 LightGBM 做时间切分的训练与验证（模型训练与调参），最后用全量数据重训并导出提交文件（测试与报告）。

### 2.1 数据探索（EDA）

我们先读入数据，把训练集和测试集拼成一张表。之所以要拼，是因为**特征工程要按「门店 + 商品」分组做时间平移**，拼在一起才能让测试期的滞后特征也算得出来：

```python
import pandas as pd, numpy as np

train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

# 拼成一张长表（测试集的 sales 是空的），并统一转成日期类型
df = pd.concat([train, test], sort=False)
df['date'] = pd.to_datetime(df['date'])

print(df.shape)          # (958000, 5)：913000 训练 + 45000 测试
print(df.isnull().sum()) # sales 缺 45000 条，正好是测试集
```

接着按年、按月、按门店/商品分组求销量的均值，观察规律：

```python
# 按年看：整体趋势是否在增长
df.groupby('year')['sales'].agg(['sum', 'mean', 'median', 'std'])

# 按月看：是否存在季节性
df.groupby('month')['sales'].agg(['mean', 'median'])
```

打印结果非常清晰：年均销量从 2013 年的 **43.51** 涨到 2017 年的 **58.82**，五年涨了约三分之一；按月看，**7 月均值 66.99 最高，1 月 35.52、2 月 39.38 最低**，典型的「夏旺冬淡」。这两条结论直接指导了后面的特征设计——必须把「年份」和「月份」都喂给模型。

再按门店和商品分组看，销量的中位数有高有低，说明**不同组合的基准水平不一样**。这提醒我们：模型不能只学「整体曲线」，还要能区分具体的门店和商品，所以后面要对 `store`、`item` 做独热编码。

### 2.2 数据清洗与特征工程

这份数据本身很干净（没有缺失值、没有重复行），真正的「清洗」工作其实是**把时间信息变成模型能吃的数字**。我们分四步。

**第一步：日期特征。** 从 `date` 里拆出月、日、年内第几天、年内第几周、星期几、年，外加「是否周末」「是否月初」「是否月末」三个布尔标记：

```python
def create_date_features(data):
    data['month']        = data['date'].dt.month       # 月份，抓季节性
    data['day_of_month'] = data['date'].dt.day         # 几号，抓月内波动
    data['day_of_year']  = data['date'].dt.dayofyear   # 年内第几天，抓全年位置
    data['week_of_year'] = data['date'].dt.isocalendar().week  # 第几周
    data['day_of_week']  = data['date'].dt.dayofweek    # 星期几，抓周内规律
    data['year']         = data['date'].dt.year         # 年份，抓长期趋势
    data['is_wknd']      = (data['date'].dt.weekday >= 5).astype(int)  # 是否周末
    data['is_month_start'] = data['date'].dt.is_month_start.astype(int)
    data['is_month_end']   = data['date'].dt.is_month_end.astype(int)
    return data

df = create_date_features(df)
```

**第二步：滞后特征（lag）。** 「滞后 91 天」的意思是：把同一门店同一商品在 91 天前的销量搬过来，作为今天的特征。为什么要挑 91、98、105……这些数字？因为它们都是 **7 的倍数**（13 周、14 周、15 周……），这样「星期几」正好对齐——今天是周三，91 天前也是周三，销量才可比。另外 364、455、546、728 分别是 52 周、65 周、78 周、104 周，对应「一年前、一年零一季度前」等长期参考。我们还给每个滞后值加了一点随机噪声（标准差 1.6），这是一个小技巧：**让模型不要死记硬背某一天的精确数值，而是学到更稳的规律**：

```python
def random_noise(dataframe):
    # 标准差 1.6 的高斯噪声，防止模型对滞后值过度敏感
    return np.random.normal(scale=1.6, size=(len(dataframe),))

def lag_features(dataframe, lags):
    for lag in lags:
        # 按「门店 + 商品」分组，各自往前挪 lag 天
        dataframe[f'sales_lag_{lag}'] = dataframe.groupby(['store', 'item'])['sales'] \
            .transform(lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

lags = [91, 98, 105, 112, 119, 126, 182, 238, 364, 455, 546, 728]
df = lag_features(df, lags)
```

**第三步：滑动平均。** 单个滞后值只是「某一天」的快照，噪声大；滑动平均看的是「过去一段时间」的整体水平，更稳。对每个组合，先 `shift(1)` 把当天排除掉（防止偷看答案），再在窗口内取三角加权平均：

```python
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe[f'sales_roll_mean_{window}'] = dataframe.groupby(['store', 'item'])['sales'] \
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=10,
                                                    win_type='triang').mean()) \
            + random_noise(dataframe)
    return dataframe

df = roll_mean_features(df, [121, 181, 365, 546])
```

**第四步：指数加权平均（EWM）。** 滑动平均对窗口内每一天一视同仁，指数加权则是「越近的日期权重越大」，能更快跟上变化。我们对 6 个平滑系数（0.99 到 0.5）和 12 个滞后天数做组合：

```python
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            name = f'sales_ewm_alpha_{str(alpha).replace(".", "")}_lag_{lag}'
            # 先平移 lag 天，再做指数加权平均；alpha 越小越平滑
            dataframe[name] = dataframe.groupby(['store', 'item'])['sales'] \
                .transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5]
df = ewm_features(df, alphas, lags)
```

做完这几步，再由 `pd.get_dummies` 把 `store`、`item`、`day_of_week`、`month` 变成独热编码，最后对销量做 `log1p` 变换：

```python
# 把类别字段变成 0/1 列；树模型虽然能处理类别，但独热更稳
df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

# log1p = log(1+x)：压缩大销量的影响，也让误差更接近百分比
df['sales'] = np.log1p(df['sales'].values)
```

**为什么要对销量取对数？** 因为 SMAPE 本质是「相对误差」，而销量从个位数到两百多，绝对误差在大销量上会被放大。取对数之后，模型优化的 MSE 更接近「相对误差」，和比赛指标更对得上。预测时用 `expm1` 反变换回原始销量。

### 2.3 模型训练与超参数调优

**时间切分。** 时间序列不能随机切分——用「未来」预测「过去」是作弊。我们按时间切：2015-01-01 到 2016-12-31 做训练（365,500 行），2017-01-01 到 2017-03-31 做验证（45,000 行，和测试集同样长度）：

```python
train = df.loc[(df['date'] >= '2015-01-01') & (df['date'] < '2017-01-01'), :]
val   = df.loc[(df['date'] >= '2017-01-01') & (df['date'] < '2017-04-01'), :]

# 特征列：去掉日期、id、标签和 year（year 是趋势代理，测试期是 2018 没在训练里出现过）
cols = [c for c in train.columns if c not in ['date', 'id', 'sales', 'year']]
```

**SMAPE 评估函数。** LightGBM 自带的 `mae` 不是比赛指标，所以我们自己写一个 `feval`，在每个评估点把预测值反变换回原尺度（`expm1`）再算 SMAPE：

```python
def smape(preds, target):
    n = len(preds)
    # 预测和真实同时为 0 时没有意义，先排除掉
    mask = ~((preds == 0) & (target == 0))
    preds, target = preds[mask], target[mask]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    return (200 * np.sum(num / denom)) / n

def lgbm_smape(preds, train_data):
    # 模型内部在 log 空间训练，所以要先 expm1 还原
    labels = train_data.get_label()
    return 'SMAPE', smape(np.expm1(preds), np.expm1(labels)), False
```

**参数设置。** 树的数量上限给了 15,000 棵，配合 200 轮早停。`num_leaves=10` 和 `max_depth=5` 都刻意设得很小——这是因为特征里有大量高度相关的滞后项，树太深会迅速记住训练期的噪声；`learning_rate=0.01` 配上小树，是典型的「慢而稳」配置：

```python
import lightgbm as lgb

lgb_params = {
    'metric': 'mae',            # 训练时看 MAE
    'num_leaves': 10,           # 每棵树最多 10 个叶子，限制复杂度
    'learning_rate': 0.01,      # 学习率小，靠多棵树慢慢逼近
    'feature_fraction': 0.8,    # 每棵树只用 80% 特征，增加多样性
    'max_depth': 5,
    'verbose': 0,
    'num_boost_round': 15000,
    'early_stopping_rounds': 200,
    'nthread': -1,
}

lgbtrain = lgb.Dataset(X_train, label=Y_train, feature_name=cols)
lgbval   = lgb.Dataset(X_val,   label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=15000,
                  early_stopping_rounds=200,
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
print(smape(np.expm1(y_pred_val), np.expm1(Y_val)))
```

### 2.4 模型测试与指标报告

训练日志很有信息量。验证 SMAPE 从第 100 轮的 **22.01** 迅速下降到第 1000 轮的 **14.24**，之后进入「细水长流」阶段，每 1000 轮只降 0.1 左右，到 15,000 轮时是 **13.4762**：

```
[100]    training's SMAPE: 21.7764   valid_1's SMAPE: 22.013
[1000]   training's SMAPE: 13.0316   valid_1's SMAPE: 14.2393
[5000]   training's SMAPE: 12.4019   valid_1's SMAPE: 13.6191
[10000]  training's SMAPE: 12.2047   valid_1's SMAPE: 13.5043
[15000]  training's SMAPE: 12.0662   valid_1's SMAPE: 13.4762
```

两个结论：

- **最终验证 SMAPE 是 13.48**，说明平均误差约 13%，对一个横跨 500 条曲线、还要预测三个月后的任务来说，这是个相当可用的基线；
- **训练 SMAPE（12.07）和验证 SMAPE（13.48）只差 1.4 个百分点**，没有明显过拟合；同时早停一直没触发（跑到上限 15,000 轮），说明「小学习率 + 小树」还有继续压榨的空间，但边际收益已经很小。

我们还可以看特征重要性（`model.feature_importance('gain')`），排在前面的通常是近期滞后（91、98 天）和指数加权特征——这和直觉一致：**预测下周的销量，最近几周的数据最有用**，一年前同期的数据只起辅助作用。

**最终模型与提交。** 确认方案有效后，我们把训练集和验证集合并（2015 年到 2017 年底的全部数据）重新训练，树的数量沿用验证阶段找到的 `best_iteration`。为了让全量数据也能充分学习，这里把学习率调到 0.02（比验证时大一倍），相当于「同样的树数量，学得更充分」：

```python
# 用全部有标签的数据重训
train = df.loc[~df['sales'].isna()]
X_train, Y_train = train[cols], train['sales']
test = df.loc[df['sales'].isna()]
X_test = test[cols]

final_params = {
    'metric': 'mae', 'num_leaves': 10, 'learning_rate': 0.02,
    'feature_fraction': 0.8, 'max_depth': 5, 'verbose': 0, 'nthread': -1,
    'num_boost_round': model.best_iteration,   # 沿用验证阶段的最佳轮数
}

lgbtrain_all = lgb.Dataset(X_train, label=Y_train, feature_name=cols)
final_model = lgb.train(final_params, lgbtrain_all,
                        num_boost_round=model.best_iteration)

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

# 反变换回原始销量，并整理成 id + sales 的提交格式
submission = test.loc[:, ['id', 'sales']].copy()
submission['sales'] = np.expm1(test_preds)   # log1p 的逆运算
submission['id'] = submission['id'].astype(int)
submission.to_csv('submission.csv', index=False)
```

提交前我们核对了行数（45,000 行）、`id` 是否与样例一致、`sales` 是否都是非负数——这三项是这类比赛最容易踩的坑。

---

## 三、总结

本文中我们完成了商店商品需求预测的完整流程。我们首先在 EDA 里确认数据是 50 商品 × 10 门店 × 5 年共 913,000 行，并发现销量有持续上升的长期趋势和「夏高冬低」的季节性；其次我们围绕时间结构做特征工程：拆出月、日、星期等日期特征，按 7 的倍数构造 12 个滞后特征，加 4 个三角加权滑动平均窗口和 6×12 组指数加权特征，并把类别字段独热编码、销量取 log1p；接着我们按时间切出 2015～2016 训练、2017 年一季度验证，用「小叶子 + 小学习率 + 15,000 棵树 + 200 轮早停」的 LightGBM 训练，并自定义 SMAPE 作为评估函数；最后我们在验证集上得到 **SMAPE 13.48**，并用全量数据重训、反变换后生成提交文件。

这场比赛给我们的最大启发是：**时间序列预测的功夫八成在特征工程**。模型本身只是 LightGBM 的默认配置加两个正则参数，真正让 SMAPE 从 22 降到 13.5 的，是「按门店商品分组平移」「滞后天数取 7 的倍数保证星期对齐」「滑动平均先 shift 防泄漏」这些细节。另外，本地评估一定要**按时间切分**并要求自己写对比赛指标，否则分数会好看得毫无意义。