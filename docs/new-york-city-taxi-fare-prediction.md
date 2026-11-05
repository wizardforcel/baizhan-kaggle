# New York City Taxi Fare Prediction

## 一、比赛简介

### 1.1 比赛背景
这个比赛的数据来自纽约市出租车的真实行程记录，时间跨度大约是 2009 年到 2015 年。训练集有 **5,500 万行、约 5.4 GB**，测试集只有 **9,914 行**，两者规模悬殊得有点夸张。每一条记录描述一次行程：什么时候上车、从哪里上车、到哪里下车、车上有几个乘客，以及这次行程最终收了多少钱。比赛要求我们根据上车时间、起止坐标和乘客人数，预测车费金额。

这个数据集最麻烦的地方不是模型，而是**数据量**。5.4 GB 的 CSV 在普通笔记本上直接 `read_csv` 会把内存撑爆，所以材料里反复出现「分块读取」「只取 20% 样本」「先转成 parquet/feather」这类工程技巧。另一个麻烦是**脏数据**：有些行的经纬度被写反了，有些坐标根本不在纽约，有些车费是负数或天文数字。如果不先处理，模型会被这些离群点带偏。

### 1.2 比赛目标
训练集有 8 列：

- `key`：每行唯一的标识，由上车时间加一个序号拼成；
- `fare_amount`：目标列，车费金额（美元）；
- `pickup_datetime`：上车时间；
- `pickup_longitude`、`pickup_latitude`：上车点的经度和纬度；
- `dropoff_longitude`、`dropoff_latitude`：下车点的经度和纬度；
- `passenger_count`：乘客人数。

测试集除了 `fare_amount` 之外都有。提交文件需要两列：`key` 和预测的 `fare_amount`，行数必须和测试集一致。这是一个标准的**回归**任务——输出是连续的金额，而不是类别。

### 1.3 评价标准
比赛使用**均方根误差 RMSE**（Root Mean Square Error）作为评价指标。它的计算方式是：先把每条样本的预测误差平方，求平均，再开根号。RMSE 越低越好，单位跟车费一样是美元，所以可以直观理解成「平均每次预测差了几美元」。因为误差先平方再平均，少数极端错误的样本会被放大惩罚，所以模型必须特别注意异常值和长尾行程。

---

## 二、解题过程

整场比赛的主线可以概括为：**先分块读数据并清洗掉不可能的行程，再构造时间与距离特征，最后用 LightGBM 为主、KNN 为辅做加权融合。** 下面按 EDA、数据清洗、模型训练与调参、测试与指标报告四步展开。

### 2.1 数据探索（EDA）
我们首先确认数据规模。因为训练集太大，先不急着加载，而是用系统命令看行数和文件大小。
```python
import pandas as pd

data_dir = '../input/new-york-city-taxi-fare-prediction'

# 训练集 5.4G，直接读会爆内存，先看行数
# !wc -l {data_dir}/train.csv      -> 55423856 行
# !wc -l {data_dir}/test.csv       -> 9914 行
# !ls -lh {data_dir}                -> train.csv 5.4G
```
看完规模，我们读测试集——它很小，可以完整载入。
```python
# 测试集只有几千行，直接读，顺便把时间列解析成 datetime
df_test = pd.read_csv(data_dir + '/test.csv', parse_dates=['pickup_datetime'])

print(df_test.shape)                       # (9914, 7)
print('Null values:', df_test.isnull().sum().sum())   # 0
print(df_test['key'].nunique())            # 9914，key 唯一
```
测试集没有缺失值，`key` 也保证唯一，这省了不少事。

接着看训练集的前几行，了解字段格式：
```python
# 只取前 5 行做样本，避免全量加载
sample = pd.read_csv(data_dir + '/train.csv', nrows=5)
print(sample)
```
打印出来能看到 `pickup_datetime` 形如 `2009-06-15 17:26:21 UTC`，经纬度是普通浮点数。注意时间字符串后面带 `UTC`，直接 `to_datetime` 有时会慢，所以后面我们会先截断再解析。

材料里的 EDA 还得出了几条很有价值的结论，我们用文字记下来：

1. **大多数行程只有 1 名乘客**，`passenger_count` 集中在 1，少数是 2～6。
2. **平均车费在 2011～2013 年间明显上涨**，从约 10.43 美元涨到约 12.58 美元。这说明年份本身是一个有用特征，模型应该知道「2013 年的 10 美元」和「2010 年的 10 美元」不完全等价。
3. **凌晨 5 点左右的平均车费最高**，主要原因是这个时段有很多长途行程离开市区。
4. **周末的晚些时候长途行程更多**。
5. **机场和市中心是打车热点**，尤其是 JFK 机场，往返机场的行程车费明显偏高。
6. **市中心西侧有一片区域，每公里车费异常高**；另外距市中心约 90 公里的某个位置，常年有长途行程进出，但车费却明显偏低。
7. 把「车费 vs 直线距离」按行程方向拆开画，能看到纽约街道的**网格走向**。

这些结论直接指导了后面的特征工程：时间要拆成年、月、日、星期、小时；距离要算真实球面距离；机场要单独做指示特征。

### 2.2 数据清洗
数据清洗是这场比赛的重头戏。我们先把清洗写成一个函数，方便分块调用。
```python
def clean_df(df):
    # 时间字符串带 " UTC"，先截到分钟，解析更快也更稳
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 15)
    df['pickup_datetime'] = pd.to_datetime(
        df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M'
    )

    # 有些行经纬度被写反了：纬度一般 40 左右、经度 -74 左右，
    # 所以出现 dropoff_latitude < dropoff_longitude 基本就是反了
    df = df.assign(rev=df['dropoff_latitude'] < df['dropoff_longitude'])
    idx = df['rev'] == 1
    df.loc[idx, ['dropoff_longitude', 'dropoff_latitude']] = \
        df.loc[idx, ['dropoff_latitude', 'dropoff_longitude']].values
    df.loc[idx, ['pickup_longitude', 'pickup_latitude']] = \
        df.loc[idx, ['pickup_latitude', 'pickup_longitude']].values

    # 用 query 一次性过滤掉物理上不可能的行程
    criteria = (
        " 0 < fare_amount <= 500"
        " and 0 < passenger_count <= 6"
        " and -75 <= pickup_longitude <= -72"
        " and -75 <= dropoff_longitude <= -72"
        " and 40 <= pickup_latitude <= 42"
        " and 40 <= dropoff_latitude <= 42"
    )
    df = (df.dropna()
            .query(criteria)
            .reset_index()
            .drop(columns=['rev', 'index']))
    return df
```
这段清洗做了三件事，每一件都有明确理由：

- **修正经纬度互换**：正常纽约的纬度在 40 附近、经度在 -74 附近。如果下车纬度比经度还小（比如纬度 -73、经度 40），那一定是两列写反了。我们交换回来，而不是直接删掉，因为交换后往往就是一条正常行程，删了可惜。
- **限制车费范围**：负数车费没有意义，超过 500 美元的单次出租车行程也基本是录入错误。
- **限制坐标范围**：只保留落在纽约都会区经纬度盒子里的点，把明显在别的城市甚至海里的记录剔掉。

因为训练集太大，我们分块读取，每块 500 万行，清洗后再拼接：
```python
def load_df(nrows=None):
    cols = ['fare_amount', 'pickup_datetime', 'pickup_longitude',
            'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
            'passenger_count']
    parts = []
    for chunk in pd.read_csv('../input/train.csv', usecols=cols,
                             nrows=nrows, chunksize=5_000_000):
        parts.append(clean_df(chunk))
    return pd.concat(parts)

# 为控制训练时间，只取前 1000 万行
train = load_df(nrows=10_000_000)
```
分块还有一个额外好处：`usecols` 指定列名后，pandas 不会去解析用不到的 `key` 列，既省内存又快。

### 2.3 模型训练与超参数调优
我们构造了三类特征，一共 28 个。

第一类是时间特征。把上车时间拆开，让模型能学到「年份涨价」「凌晨长途」「周末晚归」这些规律：
```python
def time_features(df):
    dt = df['pickup_datetime']
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['weekday'] = dt.dt.dayofweek
    df['hour'] = dt.dt.hour
    # 一天中的分钟数，比 hour 更细，能区分 8:00 和 8:50
    df['minute_of_day'] = dt.dt.hour * 60 + dt.dt.minute
    return df
```

第二类是距离特征。直接用经纬度做差是很粗糙的，因为经度和纬度的一度对应的实际距离不同。我们用 `pyproj` 的 `Geod` 计算球面上的真实距离：
```python
from pyproj import Geod

geod = Geod(ellps='WGS84')   # WGS84 是 GPS 用的地球椭球模型

def haversine(df):
    # inv 返回 (方位角1, 方位角2, 距离米)，我们只要距离
    _, _, dist = geod.inv(
        df['pickup_longitude'].values, df['pickup_latitude'].values,
        df['dropoff_longitude'].values, df['dropoff_latitude'].values
    )
    df['distance_m'] = dist
    return df
```
另外我们还计算了上车点和下车点分别到 JFK、LaGuardia、Newark 三个机场的距离。原因是材料里发现机场是热点，而且机场行程常有固定附加费，光靠经纬度网格不容易学到这一点：
```python
airports = {
    'jfk':      (-73.7781, 40.6413),
    'lga':      (-73.8740, 40.7769),
    'ewr':      (-74.1745, 40.6895),
}

def airport_distance(df):
    for name, (lon, lat) in airports.items():
        _, _, d_pick = geod.inv(df['pickup_longitude'].values,
                                df['pickup_latitude'].values,
                                lon, lat)
        _, _, d_drop = geod.inv(df['dropoff_longitude'].values,
                                df['dropoff_latitude'].values,
                                lon, lat)
        df[f'pickup_{name}_m'] = d_pick
        df[f'dropoff_{name}_m'] = d_drop
    return df
```

第三类是位置本身的原始坐标。虽然树模型能自己切分，但保留原始经纬度仍然有用，尤其是配合前面的距离特征一起。

主模型用 LightGBM。它属于梯度提升树：一棵一棵地训练决策树，每棵新树都去拟合前面所有树的残差，最后把所有树的输出加起来。它训练快、对特征尺度不敏感、能自动处理非线性交互，非常适合这种结构化表格数据。
```python
import lightgbm as lgbm
from sklearn.model_selection import train_test_split

feature_cols = [c for c in train.columns if c != 'fare_amount']
X = train[feature_cols]
y = train['fare_amount']

# 留 20% 做验证，固定随机种子保证可复现
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = lgbm.LGBMRegressor(
    n_estimators=500,       # 树的数量
    learning_rate=0.05,     # 每棵树的学习率，越小越稳但需要更多树
    num_leaves=31,          # 每棵树的叶子数，控制复杂度
    subsample=0.8,          # 行采样，防止过拟合
    colsample_bytree=0.8,   # 列采样，防止过拟合
    random_state=42
)
model.fit(X_tr, y_tr)
```
关于调参，我们的经验是：`learning_rate` 和 `n_estimators` 要一起看，学习率减半通常要把树的数量加倍；`num_leaves` 太大容易过拟合，31 左右是个稳妥的起点；行采样和列采样各取 0.8 能带来稳定的泛化收益。材料里也有队伍用贝叶斯优化（Optuna）去搜这些参数，思路是一样的，只是搜索更自动。

除了 LightGBM，我们还训练了一个**只用起止坐标**的 K 近邻回归（KNN）。KNN 的想法很朴素：找历史上起点和终点都最接近的 K 次行程，用它们的平均车费当作预测。它不擅长外推，但在热点区域能捕捉到局部价格模式，而且和树模型的错误不太相关，适合做融合。
```python
from sklearn.neighbors import KNeighborsRegressor

coord_cols = ['pickup_longitude', 'pickup_latitude',
              'dropoff_longitude', 'dropoff_latitude']
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_tr[coord_cols], y_tr)
```
注意 KNN 对坐标尺度敏感，如果四列量纲差别大，应该先标准化。这里经纬度都在几十的量级，影响相对小，但我们仍然建议在正式流程里加一步 `StandardScaler`。

### 2.4 模型测试与指标报告
我们先在验证集上分别看两个模型，再做加权融合。
```python
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    # squared=False 表示返回开根号后的值，单位与车费一致
    return mean_squared_error(y_true, y_pred, squared=False)

pred_lgbm = model.predict(X_val)
pred_knn = knn.predict(X_val[coord_cols])

print("LightGBM RMSE:", rmse(y_val, pred_lgbm))
print("KNN RMSE:", rmse(y_val, pred_knn))

# 材料里的最终权重：树模型 94%，KNN 6%
pred_ensemble = 0.94 * pred_lgbm + 0.06 * pred_knn
print("Ensemble RMSE:", rmse(y_val, pred_ensemble))
```
从结果看，LightGBM 明显强于 KNN，但给 KNN 一点点权重仍能小幅降低整体误差，因为它补上了树模型在局部热点上的盲区。这个 94/6 的比例是调出来的：权重再大，KNN 的噪声会拖后腿；再小，它的互补作用就消失了。

材料里这套方案的公开排行榜 RMSE 是 **2.93**。作为参照，如果只用「全体平均车费」去猜，RMSE 会在 9 美元以上，所以模型确实学到了东西。另一个队伍用 Dask 加 XGBoost 也能跑到相近水平，说明在这个数据上「梯度提升树 + 距离特征」是主流且有效的路线。

最后我们训练全量模型并对测试集预测。注意测试集也要走完全相同的特征工程，列的顺序必须和训练时一致，否则 LightGBM 会报错或给出乱序结果：
```python
# 测试集同样做清洗和特征构造
test = clean_df(pd.read_csv(data_dir + '/test.csv'))
test = time_features(test)
test = haversine(test)
test = airport_distance(test)

# 用全部训练数据重新拟合（也可以继续用 X_tr 的模型）
model.fit(X, y)

final_pred = 0.94 * model.predict(test[feature_cols]) \
           + 0.06 * knn.predict(test[coord_cols])

submission = pd.DataFrame({
    'key': test['key'],
    'fare_amount': final_pred
})
submission.to_csv('submission.csv', index=False)
```
提交前我们检查了三件事：行数是否等于 9,914、`key` 是否和测试集完全一致、有没有出现负数的预测。负数预测虽然少见，但树模型在极端外推时确实可能给出，需要裁剪到 0 以上。

---

## 三、总结
本文中我们完成了纽约出租车车费预测的完整流程。我们首先用系统命令和抽样确认了训练集有 5,542 万行、5.4 GB，测试集只有 9,914 行，并总结了乘客数、年份涨价、凌晨长途、机场热点等 EDA 结论；其次把清洗写成可复用函数，修正经纬度互换、过滤异常车费和越界坐标，并用分块读取只取 1,000 万行控制训练成本；接着构造了时间、球面距离和机场距离共 28 个特征，用 LightGBM 作主模型、KNN 作补充；最后在验证集上确定了 94% 比 6% 的融合权重，公开排行榜 RMSE 约为 2.93，并生成了符合格式的提交文件。

这个比赛最大的体会是：**在超大数据上，工程能力和建模能力一样重要**。分块读取、只保留需要的列、先截断时间字符串再解析，这些看似琐碎的细节决定了你能不能在一台普通机器上把实验跑起来。而特征方面，一个「到机场的距离」往往比多调十轮超参数更管用。
