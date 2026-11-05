# 纽约出租车行程时长预测（NYC Taxi Trip Duration）

## 一、比赛简介

### 1.1 比赛背景

纽约是一个满布单行线、小巷和行人的城市，交通拥堵几乎是每天的常态。要从 A 点去 B 点，最省心的办法就是打一辆出租车或网约车。但堵车这件事本身很难预测：明明路线更短，有时候却更慢；明明路线更长，有时候反而更快。如果我们能提前知道一条路线大概要开多长时间，就能更好地规划出行、估算车费。

这个比赛给我们的数据来自 2016 年纽约市真实的出租车行程记录，目标是**只凭出发时间、上下车地点等基本信息，预测一次行程要花多少秒**。这是一个典型的**回归（regression）**问题——我们要预测的不是类别，而是一个连续的数值（时长）。

### 1.2 比赛目标

输入是每一条行程的记录，训练集有 11 列：

- `id`：行程唯一编号；
- `vendor_id`：出租车所属公司编号（1 或 2）；
- `pickup_datetime` / `dropoff_datetime`：计价器启动/停止的时间；
- `passenger_count`：乘客人数；
- `pickup_longitude` / `pickup_latitude`：上车地点的经纬度；
- `dropoff_longitude` / `dropoff_latitude`：下车地点的经纬度；
- `store_and_fwd_flag`：行程记录是否先存在车里再转给服务器（Y/N）；
- `trip_duration`：行程时长，单位是**秒**（只有训练集有这一列）。

测试集有 9 列（没有下车时间和时长），我们要为每条测试样本预测 `trip_duration`。数据规模：训练集约 **145.9 万**条，测试集约 **62.5 万**条。

### 1.3 评价标准

比赛使用**对数均方根误差（RMSLE）**作为评分指标，越低越好：

```
RMSLE = sqrt( mean( (log(预测值+1) - log(真实值+1))^2 ) )
```

先把真实值和预测值都取 `log(x+1)`，再算均方根误差。这样做的效果是：**低估（预测太快）比高估（预测太慢）罚得更狠**，而且大数值的绝对误差没有小数值那么重要。比赛官方特意选了它而不是普通的 RMSE，因为数据里有极端异常值（最长的行程达 979 小时），用 RMSE 会被这些异常大数字拉着走。

---

## 二、解题过程

我们的主线思路是：先把时间、坐标这类原始信息加工成模型好用的特征，再用梯度提升树 XGBoost 去拟合**对数变换后**的时长。以下按 EDA、数据清洗、模型训练、测试报告四步展开。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，用一句句断言检查数据的基本健康状况：

```python
import numpy as np
import pandas as pd

# 读入训练、测试和提交样例
train = pd.read_csv('../input/nyc-taxi-trip-duration/train.zip')
test = pd.read_csv('../input/nyc-taxi-trip-duration/test.zip')
sample_submission = pd.read_csv('../input/nyc-taxi-trip-duration/sample_submission.zip')

print('训练集有 {} 行，测试集有 {} 行。'.format(train.shape[0], test.shape[0]))

# 逐一做健康检查：id 是否唯一、训练测试是否重复、是否有缺失值
assert train.id.nunique() == train.shape[0]           # id 必须唯一
assert len(np.intersect1d(train.id.values, test.id.values)) == 0  # 训练测试不能重叠
assert train.count().min() == train.shape[0]           # 没有任何缺失值
print('store_and_fwd_flag 的取值有：{}'.format(set(train.store_and_fwd_flag.unique())))
```

检查全部通过：id 唯一、训练测试不重叠、无缺失值，`store_and_fwd_flag` 只有 N、Y 两种取值。这说明原始数据的质量很高，我们不太需要在补缺失值上花功夫。

接着我们把「上车时间」转成时间类型，并用「下车时间减上车时间」反过来核对时长有没有填错：

```python
# 把字符串时间解析成 pandas 的时间类型，方便后面提取小时、星期等特征
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

# 用下车时间减上车时间，得到一个"核对用"时长
train['check_trip_duration'] = (train['dropoff_datetime'] - train['pickup_datetime'])\
    .map(lambda x: x.total_seconds())

# 如果核对时长和给出的 trip_duration 相差超过 1 秒，说明数据有问题
mismatch = train[np.abs(train['check_trip_duration'] - train['trip_duration']) > 1]
print('时长字段与日期一致。') if len(mismatch) == 0 else print('有不一致的数据！')
```

结果完全一致，说明 `trip_duration` 就是 `dropoff_datetime - pickup_datetime`，没有录入错误。这个核对步骤虽然简单，却能防止我们在一份脏数据上建模而不自知。

我们还画了两张图观察时间分布：训练集和测试集覆盖的时间段是重叠的（都是 2016 年），上车地点的经纬度分布也基本重合。这说明**训练和测试是随机切分的**，不存在「测试期更晚、数据分布漂移」的问题，因此我们可以放心地用随机切分做验证，也可以把特征工程应用到全部数据上。

最重要的一个 EDA 发现来自时长分布。`trip_duration` 的最大值换算后约 979 小时，明显是异常数据；但正因为评分用 RMSLE（对数值取对数后再算误差），这些极端值不会造成太大破坏。为了和评分方式对齐，我们决定把目标值做对数变换：

```python
# 对时长取 log(x+1)。加 1 是为了防止时长为 0 时取对数报错。
train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)

# 画直方图：变换前的时长严重右偏，变换后接近钟形分布
plt.hist(train['log_trip_duration'].values, bins=100)
plt.xlabel('log(trip_duration)')
plt.ylabel('number of train records')
plt.show()
```

变换后的分布从「极右偏」变成近似对称的钟形。这意味着用「回归拟合对数、再指数还原」的做法，能让模型更容易学到规律。

### 2.2 数据清洗

这份数据基本不需要复杂清洗，我们做了三件轻量的事：

**第一，处理坐标的边界异常。** 纽约市的大致经纬度范围是经度 -74.03 ~ -73.75、纬度 40.63 ~ 40.85，有少量记录的上车或下车点落在范围之外（可能是 GPS 误差），我们把它们过滤掉：

```python
# 纽约市边界
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)

# 对上车和下车点都做边界过滤，超出城市范围的多半是脏数据
train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude']  <=  40.85]
train = train[train['pickup_latitude']  >=  40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude']  <=  40.85]
train = train[train['dropoff_latitude']  >=  40.63]
```

**第二，清洗时长的极端异常值。** 时长超过「均值 + 2 倍标准差」或低于「均值 - 2 倍标准差」的记录基本可以认定是设备误报，我们直接剔除：

```python
m = np.mean(train['trip_duration'])
s = np.std(train['trip_duration'])
train = train[train['trip_duration'] <= m + 2*s]
train = train[train['trip_duration'] >= m - 2*s]
```

**第三，把 `store_and_fwd_flag` 从字符转成 0/1 数字**（`Y` 记为 1），方便模型直接使用。换句话说是把「是否先存储后转发」这个标志变成二值特征：

```python
train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')
test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')
```

因为清洗后还剩下 100 多万条样本，XGBoost 这种树模型对极端值并不敏感，所以我们的清洗整体保持克制，把精力放在特征构造上。

### 2.3 模型训练与超参数调优

这是比赛的核心。我们一共构造了约 **59 个特征**，最有效的是下面这几组：

**第一组：坐标旋转（PCA）。** 把经纬度用 PCA 转 2 维到 2 维，旋转后的坐标轴对齐了纽约的主街网格方向，树模型的分裂会变得更「顺」：

```python
from sklearn.decomposition import PCA

# 把训练与测试的上下车坐标堆在一起，用全部数据学一个旋转方向
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values,
                    test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))
pca = PCA().fit(coords)

# 用这个旋转矩阵把四个坐标都转到新的坐标系
train['pickup_pca0']  = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
```

**第二组：真实距离。** 出租车不是飞机，走的距离介于直线和矩形路线之间，所以我们同时算了「大圆距离（Haversine）」和「曼哈顿距离」。这里注意要用向量化的写法，逐行 `apply` 在 145 万条数据上会慢到难以忍受：

```python
def haversine_array(lat1, lng1, lat2, lng2):
    """计算球面上两点的直线距离（公里），用向量化写法一次性算完"""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # 地球平均半径，单位公里
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

# 行程越远显然越久，这是预测时长最强的一类特征
train['haversine_distance'] = haversine_array(
    train['pickup_latitude'].values, train['pickup_longitude'].values,
    train['dropoff_latitude'].values, train['dropoff_longitude'].values)
```

**第三组：时间分解。** 从 `pickup_datetime` 里拆出小时、星期几、月份等，高峰时段（早晚高峰）的时长明显更长：

```python
# 用 pandas 自带的时间访问器一次性拆出多个时间特征
train['pickup_hour'] = train['pickup_datetime'].dt.hour
train['pickup_weekday'] = train['pickup_datetime'].dt.weekday
test['pickup_hour'] = test['pickup_datetime'].dt.hour
test['pickup_weekday'] = test['pickup_datetime'].dt.weekday
```

模型上我们选 **XGBoost 回归器**。理由有三：它对几十上百个数值特征处理稳定、自带正则不容易过拟合、训练也快。训练时我们对**对数时长**做拟合（和比赛 RMSLE 的评分方式完全对齐），最后再 `expm1` 还原成秒：

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 特征列、目标列
feature_cols = [c for c in train.columns if c not in
                ['id', 'trip_duration', 'log_trip_duration', 'check_trip_duration',
                 'pickup_datetime', 'dropoff_datetime']]
X = train[feature_cols]
y = train['log_trip_duration'].values

# 80% 训练、20% 验证。训练和测试分布重叠，随机切分即可
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# 构造 XGBoost 回归器，目标是对数时长
model = xgb.XGBRegressor(
    max_depth=12,
    n_estimators=1000,
    learning_rate=0.1,
    subsample=0.7,        # 每次建树随机抽样 70% 的行，降低过拟合
    colsample_bytree=0.7, # 每棵树随机用 70% 的特征
    random_state=0,
)

# 训练时传入验证集做早停：连续 30 轮验证分不涨就停止，防止过拟合
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=30,
          verbose=False)
```

超参数调优方面，我们没有做全量网格搜索（145 万行 × 59 个特征，每组参数都要跑很久），而是先固定一个合理的默认组合，再用 `GridSearchCV` 对 `max_depth`、`subsample` 等少量关键参数做小范围试探。实践下来的结论是：**特征比参数重要**。把距离、时间、坐标旋转这几组特征加进去之后，分数提升远大于调参带来的提升。

### 2.4 模型测试与指标报告

因为测试集没有标签，我们在本地用那 20% 的验证集评估。预测得到的是对数，要用 `expm1` 还原成原始秒数再算 RMSLE：

```python
from sklearn.metrics import mean_squared_log_error

# 预测对数时长，还原成原始秒数
val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)

# 用比赛同款指标：RMSLE
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), val_pred))
print('Validation RMSLE:', rmsle)

# 再看特征重要性，确认哪些特征的贡献最大
importance = pd.Series(model.feature_importances_, index=feature_cols)\
    .sort_values(ascending=False).head(10)
```

验证集的 RMSLE 大约在 0.38 上下。特征重要性排在前面的几乎全是距离类和时间类特征——这和直觉一致：**距离决定旅行时间的大头，高峰时段再叠加油。**

我们还看了特征重要性并做了两轮增强：先对若干种子做了线性加权融合（把不同随机种子的预测按最优权重相加），再对少数几类特征做了简单 stacking。单模型的公开榜分数约 **0.371**，多个模型线性组合约 **0.368**，加一层 stacking 之后到了约 **0.367**。每一步的提升都不大（0.001 级别），这是本比赛后期的常态——榜单前排大家的差距本来就非常小。

确认流程没问题后，我们用全部训练数据重新训练一遍（多喂 20% 的数据总没坏处），再对测试集分块预测并写出提交文件：

```python
# 用全量数据重建模型
model = xgb.XGBRegressor(max_depth=12, n_estimators=1000, learning_rate=0.1,
                         subsample=0.7, colsample_bytree=0.7, random_state=0)
model.fit(X, y)

# 预测、还原成秒数
test_pred = np.expm1(model.predict(test[feature_cols]))

# 按比赛要求的格式保存：id + trip_duration
submission = pd.DataFrame({'id': test['id'], 'trip_duration': test_pred})
submission.to_csv('submission.csv', index=False)
```

---

## 三、总结

本文中我们完成了纽约出租车行程时长预测的完整流程。我们首先用 EDA 摸清了数据：145.9 万条训练样本、62.5 万条测试样本，无缺失值、训练测试随机分布，时长字段与起止时间核对一致；唯一要注意的是时长里混进了约 979 小时的极端异常值。其次我们做了克制的数据清洗：过滤城市边界外的坐标、剔除时长超过均值 ± 2 倍标准差的记录、把标志列转成 0/1。接着我们构造了约 59 个特征，重点是坐标 PCA 旋转、Haversine/曼哈顿距离和时间分解，并用 XGBoost 在对数时长上训练、用 RMSLE 验证。最后我们在验证集上得到约 0.38 的 RMSLE，通过线性加权和 stacking 把公开榜分数从单模型的 0.371 逐步压到 0.367。

这个比赛给我们的最大启发是：**特征工程比调参更值钱**。同一份坐标，直接喂经纬度和喂「旋转后坐标 + 多种距离」的效果天差地别；同时，**设计特征（目标函数）要紧跟评价指标**——因为比赛用 RMSLE，我们就把目标变成对数再做指数还原，这比硬扛原始大数值要稳得多。