# Elo 商户类别推荐（Elo Merchant Category Recommendation）

## 一、比赛简介

### 1.1 比赛背景

Elo 是巴西的一家互联网支付公司，它的用户会围绕一张**银行卡（card_id）**产生很多笔消费流水。比赛给了我们一批卡的历史消费记录，要求我们预测每张卡主人的「忠诚度评分」（loyalty score），这个分数反映了用户在未来还会不会继续用 Elo 消费。类似的问题在商业上叫 **LTV（客户终身价值）预测**：把用户按活跃程度排序，运营就能有针对性地发优惠券、做召回。

比赛的数据来自真实支付系统，由几个文件组成：

- `train.csv` / `test.csv`：卡片级信息，一共 20 万条左右训练卡，每张卡有 `first_active_month`（首次激活月份）和几个匿名特征 `feature_1/2/3`；
- `historical_transactions.csv`：每张卡的**历史交易流水**（可以有很多行）；
- `new_merchant_transactions.csv`：每张卡在**新商户**上的交易流水。

难点在于：卡片级信息太稀疏，真正有用的信号藏在「一整年消费流水」里，得先把流水压缩成特征，模型才有东西可学。

### 1.2 比赛目标

输入是一张卡的 `card_id` 以及它关联的所有交易流水，输出是一个连续的忠诚度分数 `target`。评分规则是 **RMSE（均方根误差）**，也就是预测值和真实值之差先平方、再求平均、最后开根号。RMSE 越小越好，预测完全正确时是 0。

### 1.3 评价标准

RMSE 对「大误差」特别敏感：某个样本差 10，等效于一百个样本各差 1，所以模型不能有明显的「极端预测」。我们在本地用 5 折交叉验证估计 RMSE，Leaderboard 上的公开分数也是同一套口径。

---

## 二、解题过程

整体思路一句话：**把每张卡所有的交易流水按卡聚合（聚合）成统计特征，合并到卡片表上，再用 LightGBM 回归树去拟合忠诚度。** 我们按 EDA、数据清洗、特征工程与模型调参、测试与指标报告四步走。

### 2.1 数据探索（EDA）

首先把四个文件读进来，并解析日期列：

```python
import pandas as pd
import numpy as np
import datetime

# 卡片级数据：train/test 都解析 first_active_month 为日期
train = pd.read_csv(train_path, parse_dates=["first_active_month"])
test = pd.read_csv(test_path, parse_dates=["first_active_month"])
# 两张交易流水表：purchase_date 是交易时间
ht = pd.read_csv(historical_tr_path, parse_dates=['purchase_date'])
new_merchant = pd.read_csv(new_merchant_tr_path, parse_dates=['purchase_date'])

print(train.shape)  # (201917, 8)
print(test.shape)   # (123623, 8)
```

训练集约 20 万张卡、测试集约 12 万张卡，是典型的小样本表格回归。训练样本少意味着我们不可能靠「堆数据」取胜，**只能靠特征把消费流水里的信号尽量榨出来**。接着我们从激活月份里挖出时间特征，并着手组装流水信息：

```python
# 对 train 和 test 都做同样的时间特征
for df in [train, test]:
    df['start_year'] = df['first_active_month'].dt.year
    df['start_month'] = df['first_active_month'].dt.month
    # 从首次激活到 2018-02-01 过了多少天，衡量「卡龄」
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days

ytrain = train['target']   # 取出标签后，train 里就只留特征
del train['target']
```

EDA 里还有一个容易被忽略的细节：两张交易流水表的字段里有相当一部分是分类型（`category_2` 有 5 个枚举值、`feature_*` 有缺省范围），数值型字段的取值跨度也很大。所以很多人把「降低内存占用」当成比赛的第一课——用不到 5 亿位宽的整数就绝不占 64 位，几千万行的流水表能因此少占一半以上的内存。

### 2.2 数据清洗

流水表里有很多类别字段，我们先把它们整理成 0/1 的哑变量，方便后面积分（求和、求平均）：

```python
# 历史流水：把 'Y'/'N' 转成 1/0，浮点字段也一样
ht['authorized_flag'] = ht['authorized_flag'].map({'Y': 1, 'N': 0})
ht['category_1'] = ht['category_1'].map({'Y': 1, 'N': 0})

# category_2 是 1~5 的多分类，拆成 5 列 0/1，表示「是否属于第 k 类」
for k in range(1, 6):
    ht['category_2x{}'.format(k)] = (ht['category_2'] == k) + 0

# category_3 是 A/B/C，同样拆开
for c in ['A', 'B', 'C']:
    ht['category_3{}'.format(c)] = (ht['category_3'].astype(str) == c) + 0
```

`new_merchant_transactions.csv` 用同一套代码处理，保证两个数据源的字段口径一致。清洗的关键原则是：**分类变量宁可拆成多列 0/1，也不要直接丢给模型当整数**，否则「1 比 2 小 1 」这种不存在的顺序关系会被模型误以为有意义。

顺手加一个内存压缩函数：对每一列，看它的最大最小值能塞进哪种更窄的整数类型，就降级成那种类型。这一步不改变任何数值，只是给后续的大表合并腾出内存：

```python
def reduce_memory_usage(df, verbose=True):
    """把列的类型从 int64/float64 降级到刚好装得下的窄类型"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                # 依次尝试 int8 → int16 → int32，选装得下的最小类型
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'
              .format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
```

数据清洗还有一个环节是**异常值**的处理。材料里有专门的 notebook 研究 `purchase_amount`（交易金额）的极端值：单笔金额极大、甚至报表里出现无穷大（inf）的情况。这类极值对 LightGBM 树模型影响有限（树只是把样本分到不同桶里），但我们心里要有数：**金额最大的那几笔会把某张卡的 `purchase_amount_sum` 这类聚合统计量拉得特别高**，直接喂给深度网络（材料里的 Deep AutoViML、Fastai 方案）时会撑爆数值。所以我们的主线模型不做截断，而深度路线的作者们大多选择直接删掉含 inf 的行。

### 2.3 模型训练与超参数调优

清洗之后，我们做全场比赛最重要的一步：**按 `card_id` 分组，把流水聚合成几十个统计量**。`purchase_amount` 做 sum/median/max/min/std，`installments`（分期数）做 sum/median，城市、商户、子行业做 nunique（去重计数），还把 `purchase_date` 转成时间戳后求 max/min/极差（ptp），代表用户消费的活跃时间跨度：

```python
def aggregate_historical_transactions(history):
    # 把日期转成数字时间戳，才能参与 min/max/ptp 这类聚合
    history.loc[:, 'purchase_date'] = (
        pd.DatetimeIndex(history['purchase_date']).astype(np.int64) * 1e-9)

    agg_func = {
        'authorized_flag': ['sum', 'mean'],        # 授权交易占比
        'category_1': ['sum', 'mean'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'merchant_id': ['nunique'],                # 去过几家不同商户
        'city_id': ['nunique'],                    # 在几个城市消费
        'purchase_date': [np.ptp, 'max', 'min'],   # 消费时间跨度
        'month_lag': ['min', 'max'],               # 距参考月的滞后（新旧店主）
    }
    # 按卡分组，每一列同时做上面多种统计
    agg_history = history.groupby(['card_id']).agg(agg_func)
    # 展平多级列名，方便合并：hist_purchase_amount_sum 这种
    agg_history.columns = ['hist_' + '_'.join(col).strip()
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)

    # 再单独加一列「总交易笔数」
    df = (history.groupby('card_id').size()
          .reset_index(name='hist_transactions_count'))
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    return agg_history
```

用 `groupby(...).agg(字典)` 一次算出几十个统计量。聚合完的 `new_merchant` 流水也走同样的函数（列名前缀换成 `new_`），把两个结果分别 `left join` 到工卡表上：

```python
# 新商户流水先合并，再合并历史流水
xtrain = pd.merge(train, new_trans, on='card_id', how='left')
xtest = pd.merge(test,  new_trans, on='card_id', how='left')
print(xtrain.shape)   # (201917, 47)：8 + 39 个 new_* 特征
xtrain = pd.merge(xtrain, history, on='card_id', how='left')
xtest  = pd.merge(xtest,  history, on='card_id', how='left')
print(xtrain.shape)   # (201917, 86)：再加上 hist_* 特征
```

合并后特征从 8 列涨到 86 列，这些列才是模型真正的粮食。用 `left join` 而不是内连接是刻意的：测试集里有些卡在流水表里没有记录（`join` 后是空值），LightGBM 天然会把这些空值当作「该卡没有这种交易」处理，所以空值保留不填充反而是对的。

模型我们选 **LightGBM** 回归树：它对表格数据支持好、训练快，还能天然处理缺失值（流水少的卡在聚合列上就是空）。调参我们直接用 `optuna.integration.lightgbm` 的 LightGBMTuner，它会在训练过程中自动按贝叶斯优化搜索 `num_leaves`、`feature_fraction`、`learning_rate` 等关键参数，比手工网格快得多。整体用 5 折交叉验证，每折都早停：

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb_origin
import optuna.integration.lightgbm as lgb   # 自动调参版 LightGBM

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(xtrain))                    # 存放每折的越折预测
predictions = np.zeros(len(xtest))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(xtrain.values, ytrain.values)):
    x0, y0 = xtrain.iloc[trn_idx], ytrain[trn_idx]
    x1, y1 = xtrain.iloc[val_idx], ytrain[val_idx]
    trn_data = lgb.Dataset(x0, label=y0)
    val_data = lgb.Dataset(x1, label=y1)

    # 先跑一轮自动调参：目标 RMSE，边训练边搜最佳超参数
    best_params, tuning_history = {}, []
    lgb.train({'objective': 'regression', 'metric': 'rmse'},
              trn_data, num_boost_round=10000,
              valid_sets=[val_data], early_stopping_rounds=100,
              best_params=best_params, tuning_history=tuning_history)

    # 用调好的参数正式训练，放宽早停，允许更多轮数
    best_params['learning_rate'] = 0.05
    clf = lgb_origin.train(best_params, trn_data, num_boost_round=20000,
                           valid_names=['train', 'valid'],
                           valid_sets=[trn_data, val_data],
                           early_stopping_rounds=1000)

    # 记录本折的越折预测，并累加测试集预测（之后除以 5 取平均）
    oof[val_idx] = clf.predict(xtrain.iloc[val_idx],
                               num_iteration=clf.best_iteration)
    predictions += clf.predict(xtest, num_iteration=clf.best_iteration) / 5
```

调参日志里能看到搜索轨迹：例如 `feature_fraction` 在 0.4~0.7 之间来回试探，最后稳定在 0.5；`num_leaves` 的小叶数版本（11 左右）反而在部分折上更稳。这说明**超参数不是越大越好**，越小的模型树泛化越好。

最后提一个数据里的隐藏坑：流水表里有 `month_lag` 这个字段，表示「这笔交易发生在某个参考月份之前的第几个月」。这个参考月份怎么定义，材料里有人专门发帖研究过（标题就叫 Resolving mystery of reference date），因为它的取值直接暗示了「老卡主」和「新店买手」这种生意模式的区别。我们把它当普通数值做了 min/max 聚合，等于把「这张卡最早和最近什么时候消费」交给模型自己判断，而不是亲手去拆解参考日期的秘密。

### 2.4 模型测试与指标报告

5 折交叉验证的 RMSE 如下：

```
Fold 1 CV score: 3.68587
Fold 2 CV score: 3.63118
Fold 3 CV score: 3.60545
Fold 4 CV score: 3.79800
Fold 5 CV score: 3.63245
CV score:        3.67124
```

整体越折 RMSE 是 **3.67124**，五折之间差异不大（3.60~3.80），说明模型稳定、没有过拟合单折。提交后公开榜分数约 **3.707**，与本地 CV 对齐良好。最后把预测写成提交文件：

```python
# 组装提交文件：card_id + 预测的忠诚度
sub = pd.DataFrame({'card_id': xtest['card_id'], 'target': predictions})
sub.to_csv('sub_lgb.csv', index=False)
```

需要说明的是，我们只采用了「特征工程 + 单 LightGBM」这一条主线。材料里还有不少别的高分路线：有人用 H2O AutoML、Fastai 做深度拟合，有人做 stacking/多模型融合，还有人特别处理 `purchase_amount` 里的无穷大异常行（Deep 模型遇到 inf 会爆数值）。在另一份混合方案里,用户通过多模型 blend 把私有榜打到了 **3.61319**——这说明特征工程（比如把参考日期 `month_lag` 相关细节做细）比单纯堆模型更容易提分。

---

## 三、总结

本文中我们完成了 Elo 商户类别推荐的完整流程。我们首先用 EDA 确认了数据规模（训练 20 万张卡、测试 12 万张卡）并拆出卡龄、激活月份等时间特征；其次做了数据清洗，把 `authorized_flag`、`category_1/2/3` 等类别统一转成 0/1 哑变量；接着做了全场比赛的核心——按 `card_id` 聚合出 86 个统计特征，并用带自动调参的 5 折 LightGBM 回归模型拟合忠诚度；最后在验证集上得到 CV RMSE 3.67124，公开榜 3.707。

这个比赛给我们的最大启发是：**当单条记录本身信息量不足时，特征工程决定上限**。卡片表只有 3 个匿名特征，真正的预测力全部来自把流水「竖着读多行、横着压成一行」的聚合统计量。这也是几乎所有表格型推荐/评分比赛共同的主线思路。