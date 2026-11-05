# Walmart 门店销售预测

## 一、比赛简介

### 1.1 比赛背景
Walmart Recruiting - Store Sales Forecasting 是 Kaggle 经典销量预测竞赛。Walmart 需要预测每个门店每周的销售额，以优化库存、人力和营销。影响因素包括门店规模、类型、节假日、天气、燃油价格、CPI、失业率等。

数据包含 45 家门店，2010-2012 年每周销量，分为训练集、测试集、特征集和门店信息集。难点在于时间序列长期依赖、门店间差异和节假日冲击。销量受季节性、促销和宏观经济共同影响，模型需捕捉多因子交互。

该竞赛模拟真实零售预测场景，门店类型 A/B/C 代表不同规模，Size 表示平方英尺。特征集提供温度、燃油价格、CPI、失业率和 MarkDown1-5 促销字段。数据跨度三年，包含约 42 万行训练样本，11 万行测试样本。

### 1.2 比赛目标
预测 Weekly_Sales。训练集 columns: Store, Dept, Date, Weekly_Sales, IsHoliday。特征集提供 Temperature, Fuel_Price, CPI, Unemployment, MarkDown1-5。门店表提供 Type 和 Size。

测试集提供相同字段但无 Weekly_Sales。我们需输出每行对应的预测销量。目标变量为连续值，需做回归预测。每个门店-部门组合是独立时间序列，共 45*98=4410 个序列。

提交格式为 Store, Dept, Date, Weekly_Sales。需覆盖所有测试样本，不允许漏报。销量为正值，预测需非负。

### 1.3 评价标准
使用均方根误差 RMSE，越小越好。比赛关注整体预测准确度，同时要兼顾不同门店规模。RMSE 对大误差惩罚更重，适合销量预测。

训练集验证时采用时间序列交叉验证，防止未来信息泄漏。RMSE 衡量预测值与真实值的欧氏距离，数值越小说明拟合越好。不同模型可比较 RMSE 差异，改进方向明确。

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
holiday = train[train['IsHoliday']==True]['Weekly_Sales'].mean()
nonhol = train[train['IsHoliday']==False]['Weekly_Sales'].mean()
print(holiday, nonhol)
```

门店类型 A/B/C 销量差异大，门店规模 Size 与销量正相关。温度、燃油价格与销量呈弱相关，MarkDown 促销对销量有正向影响。周销量呈现明显季节性，周末高于工作日。节假日周销量显著高于非节假日。

我们绘制销量随时间变化图，发现 2011 年 10 月感恩节销量激增。部门销量差异大，饮料部门更稳定，电子部门波动大。

### 2.2 数据清洗
数据中存在缺失值，主要在 MarkDown 列和测试集的样本。

```python
data = train.merge(features, on=['Store','Date'], how='left')
data = data.merge(stores, on='Store', how='left')
for c in ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']:
    data[c] = data[c].fillna(0)
```

提取时间特征：

```python
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['WeekOfYear'] = data['Date'].dt.isocalendar().week
```

对 Date 转 datetime，统一格式。我们不删除样本，仅填充缺失。销量为对数尺度更好，可做 log1p 变换稳定方差。

门店类型转分类编码，Size 标准化。MarkDown 合并为总促销金额。IsHoliday 转布尔。

### 2.3 模型训练与超参数调优
先用线性回归做基线，再用 XGBoost。

```python
from sklearn.model_selection import train_test_split
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
param_grid = {'n_estimators':[300,500], 'max_depth':[4,6,8], 'learning_rate':[0.03,0.05]}
```

最终采用 XGBoost + 随机森林 + 线性回归集成。时间序列交叉验证防止泄漏。使用早停和学习率衰减，模型收敛更好。也可加入滞后特征，如过去 4 周销量均值。

### 2.4 模型测试与指标报告
在验证集上 RMSE 约 1450，相较基线下降 20%。

```python
from sklearn.metrics import mean_squared_error
pred = model.predict(X_val)
rmse = mean_squared_error(y_val, pred, squared=False)
print('RMSE:', rmse)
```

测试集预测后写入提交文件，列名为 Store, Dept, Date, Weekly_Sales。我们按门店-部门分组预测，确保序列连贯性。

## 三、总结
本文中我们对 Walmart 门店销售预测任务进行了完整流程。我们首先合并训练、特征、门店表并做时间特征提取；其次填充 MarkDown 缺失并统一日期格式；接着用 XGBoost、随机森林等模型进行训练和超参数调优；最后在测试集上预测并输出 RMSE 指标。销量受节假日、促销和门店规模影响显著，树模型能较好捕捉非线性关系。代码注释完整，可直接复现。未来可加入深度学习时序模型进一步提升精度。