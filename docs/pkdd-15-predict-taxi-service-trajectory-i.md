# ECML/PKDD 2015 出租车行程终点预测（Predict Taxi Service Trajectory I）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的数据来自葡萄牙波尔图（Porto）的出租车公司。车上装了 GPS 设备，出租车每行驶 15 秒就记录一次自己的位置，一趟行程下来就形成一串带时间顺序的经纬度点。比赛方把 2013 年 7 月到 2014 年 6 月之间约 171 万条完整行程整理出来，交给参赛者研究。

比赛想解决的问题很贴近生活：**一辆出租车刚接上乘客、开了一小段路，我们能不能猜出他最终会停在哪里？** 对乘客来说，这意味着可以提前知道大概什么时候到；对调度中心来说，这意味着可以更合理地安排车辆。它也常被拿来做“轨迹预测”的入门题目。

难点主要有三个。第一，目的地由乘客决定，同样从市中心出发，有人去机场，有人去隔壁街区，随机性很大。第二，测试集只给出一段被截断的轨迹前缀，后面的路要我们补。第三，真实数据里混着空轨迹、GPS 漂移点和缺失记录，直接拿来训练会把模型带偏。

### 1.2 比赛目标

训练集有 1,710,670 行、9 列；测试集只有 320 行，列结构完全一样，只是没有答案。每一行的含义如下：

- `TRIP_ID`：行程编号，测试集里是 `T1`、`T2` 这样的短编号；
- `CALL_TYPE`：叫车方式。`A` 表示调度中心派单，`B` 表示乘客在出租车停靠站直接上车，`C` 表示在路边随手拦车；
- `ORIGIN_CALL`：叫车电话的编号，只在 `CALL_TYPE=A` 时有值；
- `ORIGIN_STAND`：上车的出租车停靠站编号，只在 `CALL_TYPE=B` 时有值；
- `TAXI_ID`：出租车编号；
- `TIMESTAMP`：行程开始的 Unix 时间戳，单位是秒；
- `DAY_TYPE`：日期类型，`A` 是普通日，`B` 是节假日或特殊日，`C` 是特殊日的前一天；
- `MISSING_DATA`：GPS 记录是否缺失，`True` 表示中间丢了点；
- `POLYLINE`：整条轨迹，格式是一串 `[经度, 纬度]`，每 15 秒一个点，第一个点是起点，最后一个点是终点。

我们要输出的是每个测试行程的 `LATITUDE`（纬度）和 `LONGITUDE`（经度）。提交样例告诉我们格式：三列 `TRIP_ID`、`LATITUDE`、`LONGITUDE`，一共 320 行，样例里所有坐标都被填成了同一个值 `41.146504, -8.611317`，那只是个占位符，没有实际意义。另外还附带一份 63 行的站点元数据，记录了每个停靠站的编号、名字和经纬度。

### 1.3 评价标准

官方提供的评测脚本叫 `evaluation_script.r`。它做的事情是：把每个测试行程预测出来的经纬度和真实终点经纬度放在一起，用 **Haversine 公式**算两点在地球表面上的球面距离，再对所有行程取平均。

Haversine 公式可以理解成“拿经纬度算直线距离”的标准做法：地球是个球，两个经纬度点之间的最短弧长就是它们的大圆距离。因为地球半径大约是 6371 公里，所以结果通常用公里或米来表示。这个指标**越低越好**，预测点离真实终点越近，分数越漂亮。它和我们平常见到的准确率不一样：不需要猜得一模一样，差几百米和差几公里在分数上是有区别的，但都比“完全猜错方向”要好。也正因为如此，模型输出的是连续的经纬度，而不是某个类别标签。

---

## 二、解题过程

我们的总体思路是：**先从轨迹里把起点、终点和首尾位移抠出来，做成一张普通的回归特征表，再用多输出回归模型同时预测终点的经度和纬度。** 下面按数据探索、数据清洗、模型训练与调参、测试与指标报告四步展开。我们首先把数据读进来摸清结构，其次从多个角度做可视化，接着清洗轨迹、构造特征，然后比较几种回归模型，最后生成提交文件并讨论指标背后的陷阱。

### 2.1 数据探索（EDA）

第一步是加载数据。比赛把四个 csv 都压成了 zip，我们直接用 `zipfile` 在内存里读，省去解压步骤：

```python
import zipfile
import pandas as pd

path = "/kaggle/input/pkdd-15-predict-taxi-service-trajectory-i/"

# 训练集、测试集、提交样例、站点元数据都以 zip 压缩包形式给出
# 用 zipfile 直接打开压缩包里的 csv，省去先解压到磁盘的步骤
with zipfile.ZipFile(path + "train.csv.zip") as zf:
    train = pd.read_csv(zf.open("train.csv"))
with zipfile.ZipFile(path + "test.csv.zip") as zf:
    test = pd.read_csv(zf.open("test.csv"))
with zipfile.ZipFile(path + "sampleSubmission.csv.zip") as zf:
    sample = pd.read_csv(zf.open("sampleSubmission.csv"))
with zipfile.ZipFile(path + "metaData_taxistandsID_name_GPSlocation.csv.zip") as zf:
    location = pd.read_csv(zf.open("metaData_taxistandsID_name_GPSlocation.csv"))

# 先确认四张表的大小，做到心里有数
print(train.shape)      # (1710670, 9)
print(test.shape)       # (320, 9)
print(sample.shape)     # (320, 3)
print(location.shape)   # (63, 4)

# 看哪些列缺值最多：ORIGIN_CALL 和 ORIGIN_STAND 是重灾区
print(train.isnull().sum())
```

打印结果很说明问题：训练集 171 万行，测试集只有 320 行，两者相差三个数量级；`ORIGIN_CALL` 缺了 1,345,900 个，`ORIGIN_STAND` 缺了 904,091 个，正好对应“只有 A 类才填电话、只有 B 类才填站点”的设计。`MISSING_DATA` 为 `True` 的只有 10 条，`DAY_TYPE` 在训练集里全部是 `A`，说明这一列没有区分度。再看 `CALL_TYPE` 的分布：B 有 817,881 条，C 有 528,019 条，A 有 364,770 条，站点直接上车是最常见的方式。`POLYLINE` 为空字符串 `[]` 的有 5,901 条，这些行程一个 GPS 点都没有，后面必须处理。

接着我们把时间戳拆开，看看订单在时间上的规律：

```python
import datetime

# TIMESTAMP 是 Unix 时间戳（秒），先转成可读时间，再拆出年、月、日、时、星期
train["data_time"] = [
    datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
    for t in train["TIMESTAMP"]
]
train["year"] = train["data_time"].dt.year
train["month"] = train["data_time"].dt.month
train["day"] = train["data_time"].dt.day
train["hour"] = train["data_time"].dt.hour
train["weekday"] = train["data_time"].dt.weekday   # 0 表示周一

# 按月统计订单量，比较 2013 与 2014 两年的季节变化
pivot = pd.pivot_table(
    train, index="month", columns="year",
    values="TRIP_ID", aggfunc="count"
).reset_index()

# 按星期统计订单量，看工作日与周末的差别
weekday = train.groupby("weekday")["TRIP_ID"].count()
call_type = train.groupby("CALL_TYPE")["TRIP_ID"].count()
```

从首行和末行的时间可以确认，训练数据覆盖 2013 年 7 月 1 日到 2014 年 6 月 30 日整整一年；测试集的时间戳落在 2014 年下半年，比训练集更晚。按月看，5 月订单最多，8 月最少，这和欧洲人夏天休假、城市活动节奏有关；按星期看，周五最忙。把 `delta_lon`、`delta_lat` 按月份和星期画箱线图后，我们发现各个月份、各个工作日的位移分布差别并不大，说明“星期几”这类时间特征对预测终点的直接帮助有限，真正有用的还是起点位置和行程本身的方向。

为了更直观地看目的地都集中在哪里，我们用了均值漂移（MeanShift）聚类。它是一种不需要事先指定簇数量的聚类方法：算法把每个点当成一个小窗口的中心，不断把窗口往密度更高的方向挪，最后挪不动的窗口中心就是簇心。放到地图上，簇心就是乘客最常去的那些区域：

```python
import json
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# 只保留每条轨迹的最后一个点（目的地），并转成 (纬度, 经度) 的数组
# 注意 POLYLINE 里存的是 [经度, 纬度]，这里交换顺序，和后文特征保持一致
df = pd.read_csv(
    zf.open("train.csv"),
    converters={"POLYLINE": lambda x: json.loads(x)[-1:]}
)
latlong = np.array([[p[0][1], p[0][0]] for p in df["POLYLINE"] if len(p) > 0])
print(latlong.shape)   # (1704769, 2)

# 先让算法自己估计一个带宽，再对目的地做均值漂移聚类
bw = estimate_bandwidth(latlong, quantile=0.1, n_samples=1000)
ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5)
ms.fit(latlong)
print("Clusters shape:", ms.cluster_centers_.shape)   # (131, 2)

# 用“均值 ± 6 倍标准差”剔除明显漂移的坐标点
n = 6
lat_min = latlong[:, 0].mean() - n * latlong[:, 0].std()
lat_max = latlong[:, 0].mean() + n * latlong[:, 0].std()
lon_min = latlong[:, 1].mean() - n * latlong[:, 1].std()
lon_max = latlong[:, 1].mean() + n * latlong[:, 1].std()
mask = (
    (latlong[:, 0] > lat_min) & (latlong[:, 0] < lat_max) &
    (latlong[:, 1] > lon_min) & (latlong[:, 1] < lon_max)
)
latlong = latlong[mask]

# 去掉离群点后把带宽固定成 0.001，能够得到更细的簇
bw = 0.001
ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5)
ms.fit(latlong)
print("Clusters shape:", ms.cluster_centers_.shape)   # (3419, 2)
```

按约 170 万个目的地做聚类，自动估计带宽时得到 131 个簇，粒度比较粗，大致对应城市里的几个大片区。把经纬度限制在均值加减 6 倍标准差的范围内（纬度约 40.94 到 41.39，经度约 -9.02 到 -8.22），再手动把带宽收紧到 0.001，簇的数量一下子变成 3,419 个，已经细到街区级别。这个对比告诉我们：目的地并不是随机撒在整张地图上的，而是沿着路网、围绕若干个热点区域聚集。后面做特征时，“起点坐标”之所以有用，是因为起点和终点在空间上相关——从同一个片区出发的车，终点往往落在相近的几个方向。

### 2.2 数据清洗

EDA 之后我们开始清洗。第一步是把 `POLYLINE` 这个长字符串拆成首点和末点的坐标。原始数据里它长这样：`[[-8.618643,41.141412],[-8.618499,41.141376],...]`，每个点都是 `[经度, 纬度]`，点与点之间用逗号分隔。我们不需要中间的每一个点，只要起点和终点，所以用一个正则把方括号去掉再按逗号切分即可：

```python
import re

def first_last_coord(polyline):
    """从 POLYLINE 字符串里取出首点和末点的经度、纬度"""
    if polyline == "[]":                 # 空轨迹没有坐标，先给 0 占位
        return 0.0, 0.0, 0.0, 0.0
    # 去掉字符串里的方括号，再按逗号切分
    nums = re.sub(r"[\[\]]", "", polyline).split(",")
    lon_1st, lat_1st = float(nums[0]), float(nums[1])
    lon_last, lat_last = float(nums[-2]), float(nums[-1])
    return lon_1st, lat_1st, lon_last, lat_last

# 对每一行执行，得到四列新特征
coords = train["POLYLINE"].apply(first_last_coord).tolist()
train["lon_1st"], train["lat_1st"], train["lon_last"], train["lat_last"] = zip(*coords)

# 空轨迹的 last 坐标是 0，训练时没有意义，直接删掉
train = train[train["lon_last"] != 0].reset_index(drop=True)
```

这段清洗有两个关键决定。一个是空轨迹直接丢弃：训练集里有 5,901 条 `[]`，如果保留，它们的终点会被当成经纬度 0，模型会学到一个“所有特征都指向非洲西海岸”的荒谬规律。另一个是只取首末点，而不是把整条轨迹的所有点都展开：一条轨迹可能有几十上百个点，全部展开会让表变得又宽又稀疏，而首末点已经能表达“从哪出发、往哪个方向走了多远”这两个最重要的信息。

有了首末点，我们就能构造位移特征，并做第二步清洗——去掉异常值、把类别编码成数字：

```python
# 位移特征：终点减起点，描述这趟车往哪个方向开了多远
train["delta_lon"] = train["lon_last"] - train["lon_1st"]
train["delta_lat"] = train["lat_last"] - train["lat_1st"]

# 极端值（超过 ±0.2 度）多半是 GPS 漂移或记录错误，删掉
df_ml = train.query(
    "delta_lon <= 0.2 and delta_lon >= -0.2 and "
    "delta_lat <= 0.2 and delta_lat >= -0.2"
).copy()

# 把文字类别翻译成数字，树模型才能处理
df_ml["Call_type"] = df_ml["CALL_TYPE"].map({"A": 1, "B": 2, "C": 3})

# 电话叫车、站点叫车：有值记 1，空值记 0，本质是“这个渠道是否被使用”
df_ml["ORIGIN_CALL"] = df_ml["ORIGIN_CALL"].notnull().astype(int)
df_ml["ORIGIN_STAND"] = df_ml["ORIGIN_STAND"].notnull().astype(int)

# MISSING_DATA 直接转成 0/1；DAY_TYPE 在训练集里全是 A，没有信息量，删掉
df_ml["MISSING_DATA"] = df_ml["MISSING_DATA"].astype(int)
df_ml = df_ml.drop(columns=["DAY_TYPE"])
```

这里有几个“为什么”。位移阈值定在 ±0.2 度，是因为 EDA 里画出的散点图显示绝大多数点都落在 ±0.2 以内，偶尔才出现明显偏离的孤点；0.2 度在波尔图这个纬度大约相当于二十公里，超过这个范围的行程要么是长途，要么是 GPS 记录出了问题，留着会拉偏回归线。把 `ORIGIN_CALL` 和 `ORIGIN_STAND` 从“编号”改成“有没有值”，是因为编号本身是随机的身份标识，直接当数值喂给模型没有意义，而“这单是不是电话叫的、是不是站点叫的”才是可解释的信息。`DAY_TYPE` 全为 `A`，方差为零，留着只会增加一列噪声。

### 2.3 模型训练与超参数调优

清洗完，数据还有一百多万行。为了快速迭代，我们随机抽 5 万条做原型验证，并按 7:3 切分训练集和验证集：

```python
from sklearn.model_selection import train_test_split

# 170 万条数据直接跑模型太慢，先随机抽 5 万条做原型验证
df_ml = df_ml.sample(50000, random_state=1)

features = ["Call_type", "ORIGIN_CALL", "ORIGIN_STAND", "MISSING_DATA",
            "lon_1st", "lat_1st", "delta_lon", "delta_lat"]
X = df_ml[features]
y = df_ml[["lon_last", "lat_last"]]   # 两个目标：终点经度和终点纬度

# 70% 训练、30% 验证，固定随机种子保证每次结果可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)
```

这里要特别解释一下 `MultiOutputRegressor`。普通的回归模型一次只能预测一个数，而我们要同时预测经度和纬度两个数。这个包装器的作用是：给每一个输出目标各训练一个回归器，预测时再把两个结果拼起来。经度和纬度虽然相关，但让同一个模型直接输出二维向量并不容易，拆成两个单目标任务反而简单可靠。

我们首先用随机森林做基线。随机森林由很多棵决策树组成，每棵树只看一部分样本和一部分特征，最后把所有树的预测取平均。它不需要对特征做标准化，对异常值也不敏感，很适合这种“特征量纲差别很大”的表格数据：

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# 随机森林本身只处理一个输出，用 MultiOutputRegressor 把经纬度当成两个任务并行预测
forest = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, random_state=1)
)
forest.fit(X_train, y_train)

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
```

`n_estimators=100` 表示种 100 棵树。树越多，结果越稳，但训练越慢；100 棵是常见的起点，再往上加带来的收益会迅速变小。`random_state=1` 是为了让每次运行得到同一批随机数，方便复现和比较。

为了确认随机森林是不是合适的选择，我们把同一套特征喂给另外四种模型：梯度提升树、线性回归、K 近邻和决策树。梯度提升树是一棵一棵串行地建树，每棵新树专门去修正前面所有树的残差；K 近邻是找特征空间里最相近的若干个样本，把它们的终点平均一下；决策树则是单棵深树。这样一轮对比，相当于在“模型家族”这个层面上做了一次粗调参：

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 把同一套特征喂给五种模型，比较它们在验证集上的表现
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=1),
    "GradientBoosting": GradientBoostingRegressor(random_state=0),
    "LinearRegression": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "DecisionTree": DecisionTreeRegressor(max_depth=50, random_state=1),
}

for name, model in models.items():
    reg = MultiOutputRegressor(model).fit(X_train, y_train)
    pred = reg.predict(X_test)
    print(name, "MSE:", mean_squared_error(y_test, pred),
          "R2:", r2_score(y_test, pred))
```

需要坦白的是，材料里的 notebook 虽然 `import` 了 `GridSearchCV`，但正文并没有展开真正的网格搜索结果。所以我们把“调参”如实地理解为两件事：一是固定随机种子、统一特征和划分方式，保证不同模型之间可以公平比较；二是在随机森林的树数量、决策树的最大深度这些少数几个参数上手工试了几个常见取值，最终保留 `n_estimators=100` 和 `max_depth=50`。在这个数据规模下，每跑一轮都要重新训练上万个节点，穷举式搜索的性价比很低，先把主线跑通比死磕单个参数更重要。

### 2.4 模型测试与指标报告

模型训练完，我们在验证集上同时报告均方误差（MSE）和决定系数（R²）。MSE 是预测值和真实值差值的平方的平均，数值越小越好；R² 衡量模型解释了目标多少比例的变化，越接近 1 越好，等于 0 相当于“只会猜平均值”：

```python
# 训练集和验证集分别算一遍，重点看两者差多少
print("MSE train:", mean_squared_error(y_train, y_train_pred))
print("MSE test :", mean_squared_error(y_test, y_test_pred))
print("R2  train:", r2_score(y_train, y_train_pred))
print("R2  test :", r2_score(y_test, y_test_pred))
```

把五个模型的结果整理成一张表（数字直接取自材料）：

| 模型 | 训练 MSE | 验证 MSE | 训练 R² | 验证 R² |
| --- | --- | --- | --- | --- |
| 随机森林 | 1.77e-05 | 0.004316 | 0.9805 | 0.3522 |
| 梯度提升树 | 2.96e-06 | 0.004150 | 0.9970 | 0.3767 |
| 线性回归 | 2.27e-28 | 2.27e-28 | 1.0000 | 1.0000 |
| K 近邻 | 3.96e-05 | 0.004551 | 0.9572 | 0.2969 |
| 决策树 | 0.0 | 0.004149 | 1.0000 | 0.3824 |

这张表读起来很有意思。随机森林、梯度提升树、K 近邻、决策树四种模型的验证 R² 都落在 0.30 到 0.38 之间，训练 R² 却高得多，说明它们都出现了明显的过拟合：树模型可以把训练样本“背下来”，但换到没见过的样本上，解释能力就掉到三成左右。相比之下，线性回归的训练和验证 MSE 都小到 1e-28 量级，R² 精确等于 1.0。这不是因为它更聪明，而是因为**特征里直接包含了答案**：我们定义 `delta_lon = lon_last - lon_1st`，又把这个 `delta_lon` 和 `lon_1st` 一起作为输入，那么 `lon_last = lon_1st + delta_lon` 就是一个恒等式，线性回归只要学到两个系数 1 就能完美复现标签。换句话说，目标泄漏了。

材料里另一份随机森林 notebook 得到的数字更能说明问题：训练 MSE 为 3.06e-06，验证 MSE 为 3.40e-06，训练 R² 为 0.9962，验证 R² 为 0.9946，作者自己在旁边写了一句“This score may be leaking.”。同一套特征、同一种模型，验证分数从 0.35 跳到 0.99，差距的来源正是这种“用标签算出来的特征又喂回模型”的循环。它提醒我们：**验证分数高不等于模型真的会预测，先检查特征里有没有藏着标签。**

确认流程能跑通后，我们对测试集做同样的清洗和特征构造，再让随机森林输出预测，并按比赛要求的格式写出提交文件：

```python
# 对测试集重复同一套清洗：抽首末点、算位移、类别编码
test["lon_1st"], test["lat_1st"], test["lon_last"], test["lat_last"] = zip(
    *test["POLYLINE"].apply(first_last_coord).tolist()
)

test_ml = test.copy()
test_ml["Call_type"] = test_ml["CALL_TYPE"].map({"A": 1, "B": 2, "C": 3})
test_ml["ORIGIN_CALL"] = test_ml["ORIGIN_CALL"].notnull().astype(int)
test_ml["ORIGIN_STAND"] = test_ml["ORIGIN_STAND"].notnull().astype(int)
test_ml["MISSING_DATA"] = test_ml["MISSING_DATA"].astype(int)
test_ml["delta_lon"] = test_ml["lon_last"] - test_ml["lon_1st"]
test_ml["delta_lat"] = test_ml["lat_last"] - test_ml["lat_1st"]

X_test_submit = test_ml[features]
pred = forest.predict(X_test_submit)

# 提交文件要求 TRIP_ID、LATITUDE、LONGITUDE 三列，顺序不能错
submission = pd.DataFrame({
    "TRIP_ID": test["TRIP_ID"],
    "LATITUDE": pred[:, 1],    # 第二列是纬度
    "LONGITUDE": pred[:, 0],   # 第一列是经度
})
submission.to_csv("submission.csv", index=False)
print("rows:", len(submission))   # 320，和测试集一致
```

写完之后我们会再读一遍 `submission.csv`，确认三件事：行数是 320、列名和样例完全一致、经纬度没有出现 0 或明显越界的值。这些检查看起来琐碎，但比赛里因为列顺序写反、行数对不上而白交一次的情况并不少见。

不过，这套方案有一个必须说清楚的局限。训练时我们拿“整条完整轨迹的最后一个点”当标签，测试时却只能拿到一段被截断的轨迹前缀，代码里的 `lon_last`、`lat_last` 其实是前缀里最后一个已知点，而不是真正的终点。模型学到的关系是“从起点走到前缀末端”，用它去猜更远的终点，天然会偏短。真正想在这个比赛里拿高分，应该把前缀的形态（走了多久、最后一段的方向和速度、附近的路网）作为特征，去拟合“剩余路程”，而不是简单地把前缀末端当成终点。这也是我们从材料里的高分假象中得到的最大教训：**指标要和任务对齐，验证集要模拟测试时的信息条件。**

---

## 三、总结

本文中我们完成了 ECML/PKDD 2015 出租车行程终点预测的完整流程。我们首先用 `zipfile` 读入 171 万条训练行程、320 条测试行程和 63 个站点的元数据，确认了 `ORIGIN_CALL`、`ORIGIN_STAND` 的大量缺失来自叫车方式的设计，并发现训练集里 `DAY_TYPE` 全为 `A`、`POLYLINE` 为空的有 5,901 条。其次我们把 Unix 时间戳拆成年月日时分和星期，看到 5 月订单最多、8 月最少、周五最忙，又用均值漂移对 170 万个目的地聚类：自动带宽得到 131 个大片区，收紧带宽后得到 3,419 个街区级热点。接着我们用正则从 `POLYLINE` 里抽出首末点，删掉空轨迹，构造首尾位移 `delta_lon`、`delta_lat`，把叫车方式映射成 1/2/3、把电话和站点特征转成“有/无”的 0/1，并按 ±0.2 度剔除 GPS 漂移点。然后我们抽样 5 万条、按 7:3 划分，用 `MultiOutputRegressor` 包装随机森林同时预测经纬度，并与梯度提升树、线性回归、K 近邻、决策树做了对比。最后我们在验证集上报告 MSE 和 R²，生成 320 行的提交文件，并识别出一个关键问题：线性回归 R² 等于 1.0、MSE 小到 1e-28，随机森林在另一份 notebook 里也出现 0.9946 的验证 R² 并被作者标注“可能泄漏”，根源是 `delta_lon`、`delta_lat` 由标签直接算出，等于把答案喂给了模型。

这个比赛给我们三点启发。第一，**特征工程要贴着业务走**：起点、方向、叫车方式这些信息，比堆砌几十列原始 GPS 点更有用。第二，**看到“好得不像话”的分数要先怀疑泄漏**，尤其是当某个特征是用目标变量算出来的时候。第三，**离线验证必须复现线上的信息条件**：测试集只给轨迹前缀，训练时就不该把完整终点当成已知量，否则验证分数再高也只是自欺欺人。
