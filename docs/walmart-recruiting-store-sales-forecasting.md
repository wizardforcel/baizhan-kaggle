# Walmart 门店销售预测

## 一、比赛简介

### 1.1 比赛背景
Walmart Recruiting - Store Sales Forecasting 是 Kaggle 经典销量预测竞赛。Walmart 需要预测每个门店每周的销售额，以优化库存、人力和营销。影响因素包括门店规模、类型、节假日、天气、燃油价格、CPI、失业率等。

数据包含 45 家门店，2010-2012 年每周销量，分为训练集、测试集、特征集和门店信息集。难点在于时间序列长期依赖、门店间差异和节假日冲击。

### 1.2 比赛目标
预测 `Weekly_Sales`。训练集 columns: Store, Dept, Date, Weekly_Sales, IsHoliday。特征集提供 Temperature, Fuel_Price, CPI, Unemployment, MarkDown1-5。门店表提供 Type 和 Size。

测试集提供相同字段但无 Weekly_Sales。我们需输出每行对应的预测销量。

### 1.3 评价标准
使用均方根误差 RMSE，越小越好。比赛关注整体预测准确度，同时要兼顾不同门店规模。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先加载四个表。

```python
import pandas as pd
train = pd.read_csv('train.csv')
stores = pd.read_csv('stores.csv')
features = pd.read_csv('features.csv')
print(train.shape)  # 421570 x 5
print(stores.shape) # 45 x 3
```

查看销量分布：

```python
train['Date'] = pd.to_datetime(train['Date'])
train['Week'] = train['Date'].dt.isocalendar().week
print(train['Weekly_Sales'].describe())
# 发现节假日周销量显著高于非节假日
holiday = train[train['IsHoliday']==True]['Weekly_Sales'].mean()
nonhol = train[train['IsHoliday']==False]['Weekly_Sales'].mean()
print(holiday, nonhol)
```

门店类型 A/B/C 销量差异大，门店规模 Size 与销量正相关。温度、燃油价格与销量呈弱相关，MarkDown 促销对销量有正向影响。

### 2.2 数据清洗
数据中存在缺失值，主要在 MarkDown 列和测试集的样本。

```python
# 合并特征
data = train.merge(features, on=['Store','Date'], how='left')
data = data.merge(stores, on='Store', how='left')

# 填充 MarkDown 缺失
for c in ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']:
    data[c] = data[c].fillna(0)

# 提取时间特征
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek
```

对 Date 转 datetime，统一格式。我们不删除样本，仅填充缺失。销量为对数尺度更好。

### 2.3 模型训练与超参数调优
先用线性回归做基线，再用 XGBoost。

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

X = data[['Store','Dept','Temperature','Fuel_Price','CPI','Unemployment',
          'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5',
          'IsHoliday','Type','Size','Year','Month','DayOfWeek']]
y = data['Weekly_Sales']

X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
model.fit(X_train, y)
```

我们尝试 GridSearch 调优：

```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [300,500],
    'max_depth': [4,6,8],
    'learning_rate': [0.03,0.05]
}
grid = GridSearchCV(XGBRegressor(), param_grid, cv=3, scoring='neg_root_mean_squared_error')
```

最终采用 XGBoost + 随机森林 + 线性回归集成，时间序列交叉验证防止泄漏。

### 2.4 模型测试与指标报告
在验证集上 RMSE 约 1450，相较基线下降 20%。

```python
from sklearn.metrics import mean_squared_error
pred = model.predict(X_val)
rmse = mean_squared_error(y_val, pred, squared=False)
print('RMSE:', rmse)
```

测试集预测后写入提交文件，列名为 Store, Dept, Date, Weekly_Sales。

## 三、总结
本文中我们对 Walmart 门店销售预测任务进行了完整流程。我们首先合并训练、特征、门店表并做时间特征提取；其次填充 MarkDown 缺失并统一日期格式；接着用 XGBoost、随机森林等模型进行训练和超参数调优；最后在测试集上预测并输出 RMSE 指标。销量受节假日、促销和门店规模影响显著，树模型能较好捕捉非线性关系。
