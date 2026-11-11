# 共享单车需求预测（Bike Sharing Demand）

## 一、比赛简介

### 1.1 比赛背景

共享单车系统会在城市的各个站点投放自行车，用户随时可以扫码借走、骑到别处还车。运营方最头疼的问题之一是：**某个小时某个站点大概会被借走多少辆车？** 投多了浪费，投少了用户借不到。这个比赛给出一套真实的共享单车运营数据，要求我们根据日期、天气等信息，预测每个小时的租借总次数。

数据覆盖连续两年、按小时汇总的记录：每一行代表某年某月某日某个小时，里面记录了季节、是否节假日、是否工作日、天气状况、温度、体感温度、湿度、风速，以及这个小时里借出的自行车数量。比赛把每个月的 **1 号到 19 号**作为训练集，**20 号到月末**作为测试集。也就是说，训练和测试在时间上是错开的，测试集相当于"未来的一段时间"，它的分布不会和训练集完全一样——这是这场比赛最需要小心的地方。

### 1.2 比赛目标

训练集有 `datetime`（小时级时间戳）、`season`（1 春、2 夏、3 秋、4 冬）、`holiday`（是否节假日）、`workingday`（是否非周末非节假日）、`weather`（1 晴朗、2 阴雾、3 小雨雪、4 大雨雪）、`temp`（摄氏温度）、`atemp`（体感温度）、`humidity`（湿度）、`windspeed`（风速）共 9 个特征列，外加 `casual`（非注册用户租借数）、`registered`（注册用户租借数）和 `count`（总租借数）。其中 `count = casual + registered`，而 `count` 就是我们要预测的目标。

测试集只有前面 9 个特征列，没有 `casual`、`registered`、`count`。我们要给测试集的每个小时预测一个 `count`，按 `datetime + count` 的格式提交。

### 1.3 评价标准

比赛使用 **RMSLE（均方根对数误差，Root Mean Squared Logarithmic Error）**，公式是：

```
RMSLE = sqrt( mean( (log(1 + 预测值) - log(1 + 真实值))^2 ) )
```

直白地说：先把预测值和真实值都取 `log(1 + x)`，再算两者的均方误差，最后开根号。**分数越小越好。** 它的特别之处在于衡量的是"相对误差"——预测 10 辆实际 20 辆，和预测 1000 辆实际 2000 辆，惩罚一样大。所以模型在销量小的小时也要足够准，不能只盯着高峰时段。

## 二、解题过程

整场比赛的思路可以概括成：**从时间戳里挖出"小时"这个最关键的特征，对类别列做独热编码，再对销量取对数后用树模型拟合。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，确认数据规模：

```python
import numpy as np
import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sampleSubmission.csv')

# 训练集 10886 行 12 列，测试集 6493 行 9 列，提交样例 6493 行 2 列
print(train.shape, test.shape, submission.shape)
```

训练集比测试集多出来的，正是 `casual`、`registered`、`count` 这三列。接着用 `info()` 看类型，并检查缺失值：

```python
train.info()

# 把训练集和测试集拼起来统一检查缺失情况
all_data = pd.concat([train[test.columns], test])
print(all_data.isnull().sum())
```

检查结果是 9 个特征列**全都没有缺失**，这省掉了不少清洗工作，可以把精力放在特征工程上。

其次我们看目标值 `count` 的分布，这是决定要不要做对数变换的关键：

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

Ytrain = train['count']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# 左图：原始销量分布
sns.distplot(Ytrain, ax=ax1, fit=norm)
# 右图：取 log(1+销量) 后的分布
sns.distplot(np.log1p(Ytrain), ax=ax2, fit=norm)
```

左图显示原始 `count` 严重右偏：绝大多数小时只借出几辆到几十辆，少数高峰能到几百上千。右图取 `log1p` 之后，分布已经接近钟形。这个观察很重要——大多数回归模型默认最小化均方误差，而右偏的数据会让模型被少数大值主导；取对数后数据更"均衡"，正好也和比赛的 RMSLE 指标（本身就在对数空间算误差）对上了。

然后我们看时间特征。`datetime` 是一串时间戳，模型读不懂，必须拆成"年、月、日、小时、星期几"：

```python
# 把 datetime 转成真正的日期类型，再拆出各个时间分量
data = train.copy()
data['datetime'] = pd.to_datetime(data['datetime'])
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['hour'] = data['datetime'].dt.hour
data['dayofweek'] = data['datetime'].dt.dayofweek

# 看每个小时的平均租借量
sns.barplot(x='hour', y='count', data=data)
```

这张"小时—销量"图会呈现出明显的 **3 个高峰：早上 8 点、中午 12～13 点、傍晚 17～18 点**。这正好对应上班、午休、下班三个出行时段，说明共享单车有很强的通勤规律。这个发现让我们确信：`hour` 会是整个模型里最重要的特征。

### 2.2 数据清洗

清洗和特征工程在这里是一起做的。核心动作有四个：

**第一步，拆时间戳并丢掉原列。** 把 `datetime` 拆成 `year`、`month`、`day`、`hour`、`dayofweek` 之后，原始时间字符串就没有用了，直接删掉。

**第二步，把分类变量做独热编码。** `season` 和 `weather` 虽然用数字 1～4 表示，但它们并不满足"4 比 3 大一倍"这种数值关系，直接当数字喂给模型会误导模型。所以我们给每个取值开一列 0/1 标志：

```python
def preprocess(df):
    # 拆出时间特征
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['dayofweek'] = pd.to_datetime(df['datetime']).dt.dayofweek
    df['day'] = pd.to_datetime(df['datetime']).dt.day
    df['month'] = pd.to_datetime(df['datetime']).dt.month
    df['year'] = pd.to_datetime(df['datetime']).dt.year

    # weather 和 season 是分类变量，用独热编码展开成 0/1 列
    df = pd.get_dummies(df, columns=['weather', 'season'],
                        prefix=['weather', 'season'], drop_first=True)

    # 原来的时间戳已经被拆解，删掉避免干扰
    return df.drop(['datetime'], axis=1)

train_fe = preprocess(train)
test_fe = preprocess(test)
```

**第三步，处理风速的 0 值。** `windspeed` 里有一批恰好等于 0 的记录，这在真实世界里很少见，通常代表"这一小时没测到"。我们把这些 0 统一替换成该列的平均值，避免模型把"风停了"当成一个重要信号：

```python
# 只用训练集里非零的风速算平均值，避免用到测试集信息
windspeed_mean = train_fe.loc[train_fe['windspeed'] > 0, 'windspeed'].mean()

# 训练集和测试集里风速为 0 的记录，统一用这个平均值补上
for df_ in (train_fe, test_fe):
    df_.loc[df_['windspeed'] == 0, 'windspeed'] = windspeed_mean
```

**第四步，检查温度特征能不能合并。** `temp`（实际温度）和 `atemp`（体感温度）高度相关，我们顺手用主成分分析（PCA，一种把多个相关特征压缩成少数几个"综合特征"的方法）验证一下它们是不是"同一件事"：

```python
from sklearn.decomposition import PCA

# 把训练集和测试集的温度拼在一起，更完整地看两列有多像
temp_all = pd.concat([train_fe[['temp', 'atemp']], test_fe[['temp', 'atemp']]])

pca = PCA(n_components=1)
pca.fit(temp_all)
# 打印解释的方差比例：0.9939，说明一列就覆盖了 99.39% 的信息
print(pca.explained_variance_ratio_)
```

PCA 告诉我们，一个主成分就解释了 99.39% 的方差，两列温度几乎是同一件事，完全可以合并成一列。不过树模型对特征共线并不敏感，所以下面的主线里我们保留原始两列，把 PCA 当作一个可选的降维手段。

最后还有一点：`casual` 和 `registered` 只存在于训练集，测试集没有，而且 `count` 就是它俩之和。所以训练时必须把这两列从特征里删掉，否则模型会学到"把两个分项加起来"这种测试时根本用不了的捷径。

### 2.3 模型训练与超参数调优

我们首先定一个**基线**：用线性回归直接预测 `count`，看看问题有多难。为了能和比赛指标对话，我们先写好 RMSLE 的计算函数：

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmsle(y, pred):
    # 两边都取 log1p，再算均方误差的平方根
    return np.sqrt(np.mean((np.log1p(y) - np.log1p(pred)) ** 2))

def evaluate(y, pred):
    rmse_val = np.sqrt(mean_squared_error(y, pred))
    mae_val = mean_absolute_error(y, pred)
    print(f'RMSLE: {rmsle(y, pred):.3f}, RMSE: {rmse_val:.3f}, MAE: {mae_val:.3f}')
```

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 基线特征：把时间拆成数值，天气、季节保持原始编号，先不做独热
base = train.copy()
base['datetime'] = pd.to_datetime(base['datetime'])
base['year'] = base['datetime'].dt.year
base['month'] = base['datetime'].dt.month
base['day'] = base['datetime'].dt.day
base['hour'] = base['datetime'].dt.hour
base = base.drop(columns=['datetime', 'casual', 'registered'])

y = base['count']
X = base.drop(columns=['count'])

# 70% 训练、30% 验证
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

lr = LinearRegression().fit(X_tr, y_tr)
pred = lr.predict(X_val)
evaluate(y_val, pred)
# RMSLE: 1.165, RMSE: 140.900, MAE: 105.924
```

基线很差：RMSLE 1.165，平均每小时的预测误差达到 140 辆。原因有两个：一是目标值右偏，二是 `year`、`month`、`hour` 这些分类变量被当成了连续数字，模型以为"24 点比 0 点大"，其实它们是绕圈的关系。

于是我们做两处修正：**对目标取 `log1p`**，以及**把时间类变量也做独热编码**。预测时要用 `expm1`（`log1p` 的逆运算）把结果还原成真实销量。

```python
from sklearn.linear_model import Ridge, Lasso

# 目标取对数，让分布接近正态
y_log = np.log1p(y)

# 把时间、节假日、季节、天气这些分类变量都展开成独热编码
X_ohe = pd.get_dummies(X, columns=['year', 'month', 'day', 'hour',
                                   'holiday', 'workingday',
                                   'season', 'weather'])

X_tr, X_val, y_tr, y_val = train_test_split(X_ohe, y_log, test_size=0.3, random_state=0)

for model in [LinearRegression(), Ridge(), Lasso()]:
    model.fit(X_tr, y_tr)
    # 预测值要先 expm1 还原，才能和真实销量比较
    pred = np.expm1(model.predict(X_val))
    print(type(model).__name__)
    evaluate(np.expm1(y_val), pred)
# LinearRegression  RMSLE: 0.590
# Ridge             RMSLE: 0.590
# Lasso             RMSLE: 1.240
```

这一改，线性回归的 RMSLE 从 1.165 降到 **0.590**，几乎腰斩。这说明"把分类变量当分类变量对待"比换更复杂的模型更重要。

不过线性模型的极限也差不多到这里了，因为"小时—销量"是明显的三段折线，线性模型表达不了。于是我们换**树模型**：随机森林（`RandomForestRegressor`）、梯度提升（`GradientBoostingRegressor`）和 XGBoost，三棵都用 500 棵树：

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

models = [
    RandomForestRegressor(n_estimators=500),
    GradientBoostingRegressor(n_estimators=500),
    XGBRegressor(n_estimators=500),
]

for model in models:
    model.fit(X_tr.values, y_tr.values)
    pred = np.expm1(model.predict(X_val.values))
    print(type(model).__name__)
    evaluate(np.expm1(y_val), pred)
# RandomForestRegressor  RMSLE: 0.355
# GradientBoostingRegressor  RMSLE: 0.330
# XGBRegressor  RMSLE: 0.342
```

到这一步 RMSLE 已经降到了 0.33～0.36。但还有一个能"白捡分"的技巧没有用上：**既然比赛指标是在对数空间里算误差，那就干脆让模型也在对数空间里学习。** 为了确认这个技巧值多少分，我们换用另一套时间特征（保留 `hour`、`dayofweek`、`day` 的数值形式，只对 `weather`、`season` 独热），并做一组对照：同一份随机森林，一次直接预测 `count`，一次预测 `log1p(count)` 再 `expm1` 还原。

```python
from sklearn.metrics import mean_squared_log_error

# 第二套特征：时间分量保留数值形式（preprocess 已把 datetime 拆好）
y_raw = train_fe['count']
X2 = train_fe.drop(['count', 'casual', 'registered'], axis=1)

# 75% 训练、25% 验证
X_tr2, X_val2, y_tr2, y_val2 = train_test_split(X2, y_raw, test_size=0.25, random_state=42)

rf = RandomForestRegressor(n_estimators=500, n_jobs=-1)

# 对照组 1：直接预测原始 count
rf.fit(X_tr2, y_tr2)
pred = rf.predict(X_val2)
print('RF 直接预测:', mean_squared_log_error(y_val2, pred) ** 0.5)   # 0.3208

# 对照组 2：预测 log1p(count)，再 expm1 还原
rf.fit(X_tr2, np.log1p(y_tr2))
pred = np.expm1(rf.predict(X_val2))
print('RF 对数目标:', mean_squared_log_error(y_val2, pred) ** 0.5)   # 0.2928
```

随机森林从 **0.3208 降到 0.2928**，XGBoost 的收益更大：从 **0.5348 降到 0.2989**（足足改善了 0.23）。注意 XGBoost 还原后可能出现负数，而销量不可能是负的，所以要把小于 0 的预测剪成 0。

最后做一点手动的超参数微调。我们在 XGBoost 上试了学习率、行采样比例和 L1 正则，最终用 `learning_rate=0.09`、`subsample=0.7`、`alpha=0.009`、380 轮提升，把验证集 RMSLE 压到了 **0.2670**：

```python
import xgboost as xgb

# 构造 XGBoost 专用的数据格式
DM_tr = xgb.DMatrix(X_tr2, label=np.log1p(y_tr2))
DM_val = xgb.DMatrix(X_val2)

# 手调后的参数：较小的学习率 + 行采样 + 轻微 L1 正则，防止过拟合
params = {'booster': 'gbtree', 'learning_rate': 0.09,
          'subsample': 0.7, 'alpha': 0.009}
xgbr = xgb.train(dtrain=DM_tr, num_boost_round=380, params=params)

pred = np.expm1(xgbr.predict(DM_val))
pred[pred < 0] = 0                      # 销量不能为负
print(mean_squared_log_error(y_val2, pred) ** 0.5)   # 0.2670
```

我们顺便打印了随机森林的特征重要性，结果和 EDA 时的判断完全一致：**`hour` 的 importance 高达 0.75，`temp` 约 0.04，`month` 约 0.04**，其余特征都在 0.03 及以下。换句话说，这个模型七成以上的判断都来自"这是几点"。

### 2.4 模型测试与指标报告

把各个阶段的验证集 RMSLE 汇总起来，能清楚看到每一步的贡献：

| 方案 | 特征设置 | 验证集 RMSLE |
| --- | --- | --- |
| 线性回归，原始目标 | 时间当数值、不做独热 | 1.165 |
| 线性回归 + 对数目标 + 时间独热 | 时间独热 | 0.590 |
| 随机森林（500 棵） | 时间独热、目标取对数 | 0.355 |
| 梯度提升（500 棵） | 时间独热、目标取对数 | 0.330 |
| XGBoost（500 棵） | 时间独热、目标取对数 | 0.342 |
| 随机森林，直接预测原值 | 时间保留数值 | 0.3208 |
| 随机森林 + 对数目标 | 时间保留数值 | 0.2928 |
| XGBoost + 对数目标 | 时间保留数值 | 0.2989 |
| XGBoost + 对数目标 + 手调参数 | 时间保留数值 | **0.2670** |

（表里"时间独热"和"时间保留数值"是两套不同的特征工程，分别来自两个参考方案，所以中间几行不能直接横着比；同一种特征设置内部，越往下越好。）

可以看出一条清晰的主线：独热编码让线性模型从 1.165 降到 0.590；换树模型再降到 0.33 左右；换成保留数值时间的第二套特征后，对数目标又把分数推到 0.29 附近；最后手调超参数，在验证集上达到 **0.267**。

确认方案可用后，我们用全部训练数据重新训练，对测试集预测并生成提交文件。这里以树模型方案为例：

```python
# 全量训练特征：沿用第二套处理（时间保留数值形式）
X_all = train_fe.drop(['count', 'casual', 'registered'], axis=1)
y_all = train_fe['count']

# 测试集独热后可能少列，用 reindex 把列对齐、缺的补 0
test_X = test_fe.reindex(columns=X_all.columns, fill_value=0)

# 用全量训练数据重新训练（目标取对数）
final_model = xgb.XGBRegressor(
    n_estimators=380, learning_rate=0.09,
    subsample=0.7, reg_alpha=0.009, n_jobs=-1
)
final_model.fit(X_all, np.log1p(y_all))

# 预测并还原成真实销量
predictions = np.expm1(final_model.predict(test_X))
predictions[predictions < 0] = 0

# 按比赛格式组装：datetime + count
submission['count'] = predictions
submission.to_csv('submission.csv', index=False)
```

作为参照，一份采用类似思路（时间特征 + 独热编码 + 对数目标 + LightGBM）的入门方案，在公开排行榜上拿到了 **0.37235** 的 RMSLE。验证集和排行榜之间存在差距是正常的——还记得吗，测试集是每个月的 20 号到月末，属于训练集"没见过"的时间段，分布本来就会漂移。这也是为什么最后一定要用全量数据重训、并且对预测值做非负截断。

## 三、总结

本文中我们完成了共享单车需求预测的完整流程。我们首先用 EDA 摸清了数据：训练集 10886 行、测试集 6493 行、特征列没有缺失，目标 `count` 严重右偏，`count = casual + registered`，而"小时"呈现出早 8 点、午 12～13 点、晚 17～18 点三座通勤高峰；其次我们拆出年、月、小时、星期几等时间特征，对 `season` 和 `weather` 做独热编码，把风速的 0 值用均值填补，并用 PCA 验证了 `temp` 和 `atemp` 几乎是同一列（一个主成分就保留了 99.39% 的方差）；接着我们以线性回归为基线（RMSLE 1.165），通过独热编码和对数目标逐步改进，再换随机森林、梯度提升、XGBoost，并用"训练 log1p(count)、预测后 expm1 还原"的技巧对齐比赛指标，最终手调 XGBoost 在验证集上达到 0.2670；最后用全量数据重训、做非负截断后生成 `datetime + count` 格式的提交文件。

这场比赛给我们最大的启发有三点：**一是先看清指标再建模**，RMSLE 天然在对数空间工作，所以对数变换几乎是"白送"的提升；**二是时间特征才是主角**，`hour` 一个特征就贡献了 75% 的重要性，把时间戳拆对远比堆模型有效；**三是测试集是"未来的时间段"**，训练集和测试集在时间上错开意味着不能迷信验证集分数，需要用全量数据重训、并对预测做合理约束。材料里还有把 `casual` 和 `registered` 分开建模再相加、以及用时间序列或 LSTM 的思路，属于更进阶的尝试，本文按效果最好、讲得最清楚的一条主线来写。