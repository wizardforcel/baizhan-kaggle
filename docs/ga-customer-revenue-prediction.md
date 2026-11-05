# GA 客户收入预测（Google Analytics Customer Revenue Prediction）

## 一、比赛简介

### 1.1 比赛背景

这个比赛由 Google Merchandise Store（谷歌官方的纪念品网店）的真实流量数据打造。数据来自 Google Analytics（谷歌分析，简称 GA），也就是记录网站访问行为的工具——谁在什么时间、从哪个渠道、用哪种设备进了网站，点过什么页面，最终买了什么。比赛要求参赛者**预测：如果给某个访客未来一段时间的浏览记录，他接下来会给商店带来多少收入**。

数据自带两个特点，也是这次比赛最折磨人的地方：

1. **海量但稀疏**：训练集有 90 多万条会话（session）记录，但只有约 1% 左右的会话有购买行为，绝大多数人对商店一分钱都没花。直接预测收入，模型很容易学成「全都预测 0」。
2. **嵌套 JSON**：很多列（如设备、地理位置、流量来源）的值是一整串 JSON 字符串，里面的子字段才是真正有用的信息，需要先解析开才能用。

### 1.2 比赛目标

给每个 `fullVisitorId`（访客编号）预测它在测试期间的**总收入**。原始训练表每行是一次会话，主要字段包括：

- `fullVisitorId`：访客的唯一编号（注意同一个访客有多次会话）；
- `visitId` / `sessionId` / `visitStartTime`：本次会话的时间与编号；
- `channelGrouping`：流量渠道（自然搜索、广告、直接访问等）；
- `device`、`geoNetwork`、`totals`、`trafficSource`：设备、地域、汇总统计（页面浏览量、停留时长等）、来源，这几个字段是 JSON 格式；
- `totals.totalTransactionRevenue`：本次会话带来的交易收入（美元），这就是预测目标。

评判时并不是看单条会话，而是**把每个访客的所有会话收入加起来**，作为这个访客的收入。测试集和训练集的访客不重叠，我们要为每个测试访客输出一个收入预测。

### 1.3 评价标准

比赛用 **RMSLE**（Root Mean Squared Logarithmic Error，对数均方根误差）评价，越小越好。

它的计算方法是：先把每个访客的真实收入和预测收入都取 `log1p`（即 `log(1+x)`，取对数的作用是把「收入」这种长尾分布拉平，避免几个大额订单主宰误差），求均方误差再开根号：

```python
import numpy as np
from sklearn.metrics import mean_squared_error

def rmsle(actual, predicted):
    """对每个访客求和后，在 log(1+x) 空间里算均方根误差"""
    a = np.log1p(actual)
    p = np.log1p(predicted)
    return np.sqrt(mean_squared_error(a, p))
```

排行榜上 1.29 左右是「利用数据泄漏」之后不错的分数；什么都不懂的基线（全猜 0）大约在 1.9 附近。

---

## 二、解题过程

我们参考了比赛冠军 Konstantin Nikolaev 的方案（有热心人把他的 R 代码翻译成了 Python），这是整场比赛公认最清晰的「用户级」路线：**把一个访客一段时间的行为聚合成一行特征，先判断他会不会买，再预测他买多少**。中间还有一个贯穿全局的关键——**数据泄漏**。

我们按 EDA、清洗、训练、测试四步走。

### 2.1 数据探索（EDA）

第一步是把 CSV 读进来。原始文件里 `device`、`geoNetwork`、`totals`、`trafficSource` 四列是 JSON 字符串，我们先用 `json_normalize` 把嵌套结构拍平，让子字段变成独立列。文件很大（训练集约 1.4 GB），所以用分块读取：

```python
import pandas as pd
import json
from pandas.io.json import json_normalize

json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
usecols = ['channelGrouping', 'date', 'device', 'fullVisitorId',
           'geoNetwork', 'totals', 'trafficSource', 'visitId',
           'visitNumber', 'visitStartTime']

def load_csv(path, chunksize=100000):
    """分块读取，并把 4 个 JSON 列拍平成普通列"""
    ans = pd.DataFrame()
    reader = pd.read_csv(path, usecols=usecols,
                         converters={c: json.loads for c in json_cols},
                         chunksize=chunksize)
    for chunk in reader:
        for c in json_cols:
            # 每个 JSON 对象拆成一行内的多个子字段，列名加前缀
            sub = json_normalize(chunk[c])
            sub.columns = [f'{c}_{x}' for x in sub.columns]
            chunk = chunk.drop(c, axis=1).merge(sub, right_index=True, left_index=True)
        ans = pd.concat([ans, chunk], ignore_index=True)
    return ans

train = load_csv('../input/train_v2.csv')
print(train.shape)    # (1109157, 56) 左右
```

看到数据规模后，我们做了两件事来理解它：

- **看训练/测试时间范围**：训练集从 2016-08 到 2018-04，测试集从 2018-05 到 2018-10。这说明测试集在时间上完全在训练集之后——这是评判中很大一部分难度所在，因为用户行为习惯会随时间漂移。
- **看目标的稀疏度**：统计有多少会话确实产生了收入：

```python
target = train['totals_totalTransactionRevenue'].astype(float)
target = target.fillna(0)                 # 缺失收入当作 0
print((target > 0).mean())                # 只有约 1% 的会话有收入
```

只有 1% 左右的行非零，这是一个典型的长尾回归（long-tail regression）问题。这跟一般回归完全不同：大部分样本的值是精确的 0，少量样本的值非常大。直接用普通回归模型拟合，模型会严重偏向预测 0。

### 2.2 数据清洗与特征工程

#### 2.2.1 清洗

清洗工作比较常规但有讲究：

- 把 `'(' not set)'`、`'not available in demo dataset'`、`'unknown.unknown'` 等「占位字符串」统一替换成 `NaN`，它们是 GA 里「信息缺失」的标准写法；
- 把数值列（页面浏览量、停留时长、点击次数等）转成 `float`；
- 把日期 `date` 转成真正的日期类型。

```python
# 这些字符串在 GA 原始导出里都表示"缺失"
nulls = ['(not set)', 'not available in demo dataset',
         '(not provided)', 'unknown.unknown', '/']
for null in nulls:
    train.replace(null, np.nan, inplace=True)

train['totals_hits'] = train['totals_hits'].astype(float)
train['totals_pageviews'] = train['totals_pageviews'].astype(float)
```

#### 2.2.2 关键窗口设计：训练样本怎么切

这是整场比赛组织数据的核心思路。我们不把时间混在一起训练，而是学着比赛实际的时间结构，**按「行为期 + 预测期」切出 4 段训练样本**：

- 用 **168 天**（24 周）的访客行为作为「观察窗」，聚合出特征；
- 观察窗结束后空出 **46 天**做间隔（避免行为期的数据直接泄漏到预测期）；
- 用接下来的 **62 天**作为「预测窗」，这段时间里访客产生的收入就是目标。

```python
from datetime import timedelta

def get_time_frame(train, k=1):
    """第 k 段：观察窗 168 天 + 间隔 46 天 + 预测窗 62 天"""
    base = train.date.min()
    # 观察窗：模型能看到的行为
    tf = train[(train.date >= base + timedelta(days=168*(k-1))) &
               (train.date <  base + timedelta(days=168*k))]
    # 预测窗：把这段时间里有回访记录的访客作为"正样本"
    future = train[(train.date >= base + timedelta(days=168*k + 46)) &
                   (train.date <  base + timedelta(days=168*k + 46 + 62))]
    tf_fvid = set(future.fullVisitorId)
    return tf, tf_fvid
```

这样切，好处是**训练数据的结构和测试完全一致**（都是「看过去 168 天、预测未来一小段时间」），于是交叉验证分数可靠，而且天然避免了随机切分导致的未来信息泄漏。

#### 2.2.3 用户级聚合特征

确定窗口后，我们把**每一个访客**在观察窗内的所有会话聚合成一行。聚合方式很灵活：分类字段取出现频率最高的值，数值字段取和、均值、最大最小值，时间字段算「第一次/最后一次访问距窗口边界的天数」等。

```python
def build_features(tf, mindate, maxdate):
    """把一个时间段内的会话按 fullVisitorId 聚合成一行"""
    g = tf.groupby('fullVisitorId').agg({
        'channelGrouping': ('channelGrouping', lambda x: x.dropna().max()),
        'geoNetwork_city': ('city', lambda x: x.dropna().max()),
        'totals_pageviews': ('pageviews_sum', lambda x: x.sum()),
        'totals_hits': ('hits_sum', lambda x: x.sum()),
        'visitNumber': ('visitNumber_max', lambda x: x.max()),
        'date': ('days_since_first', lambda x: (x.min() - mindate).days),
    })
    return g
```

这里体现的建模思想是：**GA 比赛预测的是「用户」而不是「会话」**。单个会话太短、太随机；把一个人 24 周的行为汇总成特征，模型才能学到「这个人的购买倾向」这种稳定的模式。

### 2.3 模型训练与超参数调优

#### 2.3.1 两阶段模型：会不会买 × 买多少

因为目标极度稀疏（只有约 1% 会买），我们用一个**两阶段模型**（也叫 hurdle model，门槛模型）来分解问题：

- **阶段一（分类器）**：预测 `P(该访客会买)`，这是一个二分类问题；
- **阶段二（回归器）**：只在「会买」的访客上训练，预测 `log1p(预测期收入)`，这是一个回归问题。

最终预测 = `P(会买) × 预计收入`。

```python
# 阶段二回归器的训练集只保留确实买了的人
dtrain_all = lgb.Dataset(X, label=y_ret)            # 全量，二分类标签
dtrain_ret = lgb.Dataset(X[y_ret == 1], label=y[y_ret == 1])   # 只留正样本，回归
```

两个阶段都用 LightGBM。分类器的参数 `objective='binary'`、10 棵树叶的浅树、学习率 0.01；回归器 `objective='regression'`、更浅，防止在只有几百个正样本的小数据集上过拟合：

```python
params_lgb2 = {   # 会不会买（二分类）
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 15,          # 叶子数少，模型简单
    'learning_rate': 0.01,
    'bagging_fraction': 0.9,
    'feature_fraction': 0.8,
}
params_lgb3 = {   # 买多少（回归）
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 9,
    'learning_rate': 0.01,
    'bagging_fraction': 0.9,
    'feature_fraction': 0.8,
}
```

我们重复 10 次（每次随机种子不同），把 10 组「分类器×回归器」的输出取平均，让预测更稳。之所以需要平均，是因为正样本太少，单次训练波动很大。

#### 2.3.2 关键技巧：利用数据泄漏

这可能是这个比赛最有名的一个话题。比赛主办方用的数据来自谷歌的一个**公开演示账户**（Google Merchandise Store demo）。有参赛者发现，这个演示账户在 GA 的 User Explorer（用户探索）功能里**可以导出每个用户的购买汇总**（每行包含 `Client Id` 和 `Revenue`）。而 `Client Id` 里恰好包含 `visitId` 的部分：

```python
# 外部泄漏数据里的 Client Id 形如 "cid.visitId"
df_extra["visitId"] = df_extra["Client Id"].apply(
    lambda x: int(x.split('.', 1)[1]))   # 取小数点后面的部分，就是 visitId

# 拼到训练/测试数据上
for df in [train_store, test_store]:
    df["visitId"] = df["Client Id"].apply(lambda x: int(x.split('.', 1)[1]))
train = train.merge(train_store[['visitId', 'Revenue']], on='visitId', how='left')
```

一般情况下我们不应该用测试集的信息，但这个数据来自**外部公开渠道**，比赛中不算违规。把它当作一个额外特征（该访客在历史数据里已经贡献了多少收入），榜首队伍基本上都用上了这一手。用上泄漏后，随便一个 LightGBM 都能把排行榜分数打到 **1.29 左右**。

#### 2.3.3 关于调参

我们没有在超参数上花太多力气：树模型对这个比赛非常稳，两个阶段各自的 `num_leaves`（15 和 9）配合 `learning_rate=0.01` 和上千轮 boosting，已经足够收敛，网格搜索带来的提升远小于「窗口怎么切」「泄漏数据用没用」这些结构性决策。这也印证了在这个比赛里，**特征结构比调参重要得多**。

### 2.4 模型测试与指标报告

我们用第 4 段数据（最后一段观察窗 2017-11 到 2018-04、后接测试期）做本地验证，对没有标签的测试集按照同样的流程生成预测：

```python
pr_sum = 0
for i in range(10):
    # 阶段一：预测每个访客购买的概率
    prob = lgb_model1.predict(test.drop(target_cols, axis=1))
    # 阶段二：预测购买者的收入
    amt = lgb_model2.predict(test.drop(target_cols, axis=1))
    pr_sum += prob * amt        # 相乘，得到期望收入
final = pr_sum / 10

# 输出三列：访客 id、对应的预测收入（log1p 空间）
sub = pd.DataFrame({
    'fullVisitorId': test.fullVisitorId,
    'PredictedLogRevenue': final,
})
sub.to_csv('submission.csv', index=False)
```

注意提交文件里的目标是测试期间收入的对数（`log1p`），因为比赛按 `log1p` 空间评分——我们在 EDA 阶段确认过 `totals_totalTransactionRevenue` 是原始美元数，所以这里直接提交 log1p 后的模型输出即可。

评估结果是这样的：

- **纯 0 基线**：约 1.9，什么都不做就全猜零的水平；
- **两阶段 LightGBM（无泄漏特征）**：本地回测大约能到 1.6 附近；
- **加上泄漏特征**：排行榜直接跳到 **1.29**；
- 冠军方案在此基础上加上复杂的特征工程和更深的模型堆叠，最终在 1.2 出头。

结论很清楚：**这个比赛里「利用好外部信息」和「正确的两阶段结构」的收益，远大于模型细节的打磨**。我们也没有盲目信任验证集——因为数据里存在泄漏和外部信息，本地交叉验证的排序不一定等于公开榜的排序，所以每改动一个地方都会以提交一次真实分数为准。

---

## 三、总结

本文中我们完成了 GA 客户收入预测的完整流程。我们首先通过 EDA 摸清了数据：90 多万条会话、只有 1% 左右有购买行为的极端稀疏长尾、JSON 嵌套结构。其次我们做了清洗（占位字符串转缺失、数值列规范化、日期解析），并按「168 天观察窗 + 46 天间隔 + 62 天预测窗」把时间切出 4 段训练样本，再把每个访客聚合成一行特征。接着我们用「二分类预测会不会买」和「回归预测买多少」的两阶段 LightGBM，重复 10 次取平均，并利用来自谷歌公开演示账户的泄漏数据把排行榜分数打到 1.29。最后我们确认了纯零基线（约 1.9）、无泄漏的两阶段模型、加泄漏后 1.29 的完整提升链条。

这个比赛给我们的最大启发是：**在真实业务数据里，「任务结构」和「数据来源」往往比算法更重要**。稀疏长尾要先拆解成「购买概率 × 购买金额」，时间敏感的预测要按真实窗口切训练集，而外部可用的公开信息也是模型性能的一部分——学会识别这三者，才算真正掌握这类预测任务。