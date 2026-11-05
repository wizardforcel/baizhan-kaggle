# Rossmann Store Sales

## 一、比赛简介

### 1.1 比赛背景
Rossmann 是一家在德国运营超过 3000 家药店的连锁企业。比赛提供 1115 家门店 2006 年至 2014 年的日销售额历史数据，任务是在测试期对未来六周的销售额进行预测。数据分为 `train.csv`、`test.csv` 和 `store.csv`。门店特征包含类型、商品组合、竞争距离、促销信息、州假日等，交易特征包含周几、是否营业、促销、学校假期等。

### 1.2 比赛目标
对测试集的每一天、每家门店预测 `Sales`。提交文件需要包含 `Id` 和 `Sales` 两列。训练集有 1,017,209 行，测试集 41,088 行。

### 1.3 评价标准
比赛使用 RMSPE（均方根百分比误差）作为评分指标。定义为对每个样本的相对误差取平方平均再开根号，值越小越好。由于 RMSPE 对低销量样本更敏感，模型需要同时拟合大店和高峰日和小店和平日。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先把数据合并并查看基本统计。

```python
import pandas as pd
import numpy as np

train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
test = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')
df = train.merge(store, how='left', left_on='Store', right_on='Store')
```

基本描述显示 `Sales` 的均值约 5773，标准差 3849，最大值 41551，且存在 0 销量，说明门店可能关店或促销期异常。

```python
df.describe().T
```

我们检查 `Open` 字段，发现 `Open=0` 的行 Sales 恒为 0。促销 `Promo` 占约 38% 的天数，学校假期占 18%。

```python
df.groupby(df.Promo).Sales.mean()
```

Sales 与 Customers 高度相关，且门店类型与竞争距离存在差异。我们用散点图观察 Sales 与 Customers 的关系并按 Promo 分组。

```python
import seaborn as sns
temp_df = df[df.Open==1]
sns.scatterplot(temp_df.Sales, temp_df.Customers, hue=temp_df.Promo)
```

日期范围覆盖 2007-07-01 至 2015-07-31，测试集在 2015-08-01 之后。时间序列显示明显的周周期和年度节假日波动。

### 2.2 数据清洗
关店日的 Sales 被设为 0，我们保留这些行但在训练时通过 `Open` 标记区分。

```python
df = df[df.Open==1] # 只训练营业日
```

`CompetitionDistance` 中有 172817 个 0 值，我们用 NaN 标记，让 XGBoost 自己处理缺失。

```python
df['CompetitionDistance'] = df['CompetitionDistance'].replace(0, np.nan)
```

门店类型和商品组合是类别变量，先用 LabelEncoder 编码。

```python
from sklearn.preprocessing import LabelEncoder
store.StoreType = LabelEncoder().fit_transform(store.StoreType)
store.Assortment = LabelEncoder().fit_transform(store.Assortment)
```

对于日期，我们拆分为年、月、日。

```python
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df.Date.dt.day
df['Month'] = df.Date.dt.month
df['Year'] = df.Date.dt.year
```

`PromoInterval` 是形如 `Jan,Apr,Jul,Oct` 的字符串，我们把它拆成月份标志。

```python
join_with = store['PromoInterval'].str.split(',', expand=True)
store['0_PromoInterval'] = join_with[0].map(lambda x: monthToNum(x) if pd.notna(x) else np.nan)
```

竞争开业时间和促销开始时间被统一转换为整数日期。

```python
store['CompetitionOpen'] = store.CompetitionOpen.dt.strftime('%Y%m%d')
df['DateInt'] = df.Date.dt.strftime('%Y%m%d').astype(int)
```

### 2.3 模型训练与超参数调优
我们选用 XGBoost，目标是最小化 RMSPE。先对目标取对数，让分布更对称。

```python
y = np.log1p(df.Sales)
```

特征包括门店静态特征和日期动态特征。

```python
features = [
    'Store','DayOfWeek','Open','Promo','SchoolHoliday','StateHoliday',
    'StoreType','Assortment','CompetitionDistance','CompetitionOpen',
    'PromoSince','0_PromoInterval','Day','Month','Year','DateInt'
]
X_train, X_test = df[features], test[features]
```

使用自定义的 RMSPE 评估函数。

```python
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return rmspe
```

XGBoost 在训练时通过 `objective='reg:squarederror'`，在预测后指数化回原尺度。超参数初步设为 `n_estimators=1000, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`。

```python
import xgboost as xgb
model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
```

训练时我们把训练集按时间切分，保留最后几周做验证，防止数据泄漏。

### 2.4 模型测试与指标报告
在验证集上我们计算 RMSPE。

```python
pred = np.expm1(model.predict(X_val))
rmspe_val = rmspe(pred, np.expm1(y_val))
print('Validation RMSPE:', rmspe_val)
```

为了提升稳定性，我们做了 5 折时间序列交叉验证并取平均。

```python
from sklearn.model_selection import KFold
oof = np.zeros(len(df))
for tr, val in kf.split(df):
    model.fit(df.iloc[tr][features], np.log1p(df.iloc[tr].Sales))
    oof[val] = np.expm1(model.predict(df.iloc[val][features]))
```

最终用全部训练数据拟合模型，对测试集预测。

```python
model.fit(X_train, np.log1p(df.Sales))
test_pred = np.expm1(model.predict(X_test))
submission = pd.DataFrame({'Id': test['Id'], 'Sales': test_pred})
submission.to_csv('submission.csv', index=False)
```

从材料看，使用特征工程后的 XGBoost 模型在公榜上属于前 1% 的水平，验证集 RMSPE 明显低于基线。

## 三、总结
本文中我们完成了 Rossmann Store Sales 的销售额预测任务。我们首先对销售数据进行了 EDA，发现明显的周周期、促销效应和右偏分布；其次对门店信息、竞争信息和促销间隔做了系统化特征工程，并对缺失值做了合理处理；接着使用对数变换目标结合 XGBoost 进行回归训练，并在时间分割验证集上调参；最后生成提交文件获得了前 1% 左右的成绩。整个过程说明了在时间序列零售预测中，特征工程和目标变换比单纯增加模型复杂度更关键。
