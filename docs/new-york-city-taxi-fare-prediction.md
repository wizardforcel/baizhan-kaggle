# New York City Taxi Fare Prediction

## 一、比赛简介

### 1.1 比赛背景
该比赛记录了 2009 年至 2015 年间纽约市出租车的行程数据。训练集有 5,500 万行约 5.4 GB，测试集有 9,914 行。每个行程包含出发时间、出发和到达经纬度、乘客人数等信息，目标是预测车费金额。数据包含异常值，如经纬度互换、超出纽约范围的点等。

### 1.2 比赛目标
输入特征为 `pickup_datetime`、`pickup_longitude`、`pickup_latitude`、`dropoff_longitude`、`dropoff_latitude`、`passenger_count`。输出是 `fare_amount`。提交文件需要包含 `key` 和 `fare_amount` 两列。

### 1.3 评价标准
使用均方根误差 RMSE 作为评价指标，值越小越好。RMSE 衡量预测值与真实车费的平均偏差。

---

## 二、解题过程

### 2.1 数据探索（EDA）
我们先查看测试集概况。
```python
test = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv')
print('Null values:', test.isnull().sum().sum())
```
测试集无缺失值。训练集为 5.5 GB，5,542 万行，使用分块读取和随机抽样进行分析。

我们发现部分行经纬度被互换，需要修正。
```python
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'].str.slice(0,15), utc=True)
df = df.assign(rev=df.dropoff_latitude < df.dropoff_longitude)
idx = df['rev'] == 1
df.loc[idx,['dropoff_longitude','dropoff_latitude']] = df.loc[idx,['dropoff_latitude','dropoff_longitude']].values
```
修正后剔除经纬度超出纽约范围、车费异常、乘客人数大于 6 的样本。

### 2.2 数据清洗
数据清洗函数同时进行特征工程。
```python
def clean_df(df):
    df['hour'] = df['pickup_datetime'].dt.hour
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    return df
```
我们计算出发时间的小时、天、月，并用 Geod 计算两点间真实距离。
```python
from pyproj import Geod
geod = Geod(ellps='WGS84')
lon1, lat1 = pickup_lon, pickup_lat
lon2, lat2 = dropoff_lon, dropoff_lat
_, _, distance = geod.inv(lon1, lat1, lon2, lat2)
```
并计算距离机场的近似距离作为附加特征。

### 2.3 模型训练与超参数调优
主模型使用 LightGBM，结合距离和时间特征。
```python
import lightgbm as lgbm
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lgbm_model = lgbm.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8
)
lgbm_model.fit(X_train, y_train)
```
同时训练一个仅用起止位置的 KNN 模型作为多样化补充。
```python
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']], y_train)
```

### 2.4 模型测试与指标报告
在验证集上评估 RMSE 并进行模型融合。
```python
pred_lgbm = lgbm_model.predict(X_val)
pred_knn = knn.predict(X_val)
pred_ensemble = 0.94*pred_lgbm + 0.06*pred_knn

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_val, pred_ensemble, squared=False)
print("Validation RMSE:", rmse)
```
最终在排行榜上得到 RMSE 约 2.93。测试集预测后生成提交文件。
```python
submission = pd.DataFrame({'key': test['key'], 'fare_amount': final_pred})
submission.to_csv('submission.csv', index=False)
```

---

## 三、总结
本文中我们完成了纽约出租车车费预测的完整流程。我们首先对 5,500 万行的大数据集进行采样和清洗，修正经纬度互换并过滤异常值；其次构造时间、距离和机场 proximity 特征；接着用 LightGBM 主模型结合 KNN 模型做加权融合；最后在验证集上取得 RMSE 约 2.93，生成提交文件。
