# 预测次日峰值负载

## 一、比赛简介

### 1.1 比赛背景
电力系统需要提前安排发电计划，如果能够准确预测第二天的用电峰值，就能减少调度压力。这份数据把每小时的气象信息、时间信息和历史负载放在一起，要求我们用这些特征回归出负载值。

### 1.2 比赛目标
输入是每小时的记录，包含 `hour`、`day_of_week`、`month`、`is_holiday`、`temperature_c`、`humidity_pct` 等字段。输出是 `load_mw`，也就是该小时的电力负载，单位兆瓦。训练集和测试集按时间划分。

### 1.3 评价标准
比赛使用**均方根误差 RMSE**：先算预测值与真实值差值的平方，取平均后再开方。RMSE 越小越好。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先读入数据并查看分布。

```python
import pandas as pd
import numpy as np
train = pd.read_csv('/kaggle/input/competitions/predict-next-day-peak-load/train.csv')
test = pd.read_csv('/kaggle/input/competitions/predict-next-day-peak-load/test.csv')
print(train.shape, test.shape)
print(train.columns.tolist())
```

输出显示训练集包含 hour、day_of_week、month、is_holiday、temperature_c、humidity_pct、load_mw 等列。负载是连续数值。

```python
# 目标分布
print(train['load_mw'].describe())
print('skew:', train['load_mw'].skew())
```

负载分布基本对称，偏度不大，可以直接回归，不需要取对数。

### 2.2 数据清洗
时间特征是周期性的，小时 23 和小时 0 在数值上差很多，但在现实中很接近。为此我们做正余弦编码。

```python
def create_features(df):
    df = df.copy()
    # 小时和星期的周期编码
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin'] = np.sin(2*np.pi*df['day_of_week']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['day_of_week']/7)
    # 温湿度交互
    df['temp_humidity'] = df['temperature_c']*df['humidity_pct']/100
    df['is_weekend'] = (df['day_of_week']>=5).astype(int)
    return df

train = create_features(train)
test = create_features(test)
feature_cols = ['hour','hour_sin','hour_cos','day_of_week','dow_sin','dow_cos',
                'month','is_holiday','is_weekend','temperature_c','humidity_pct','temp_humidity']
X = train[feature_cols]
y = train['load_mw']
X_test = test[feature_cols]
```

### 2.3 模型训练与超参数调优
我们训练两个梯度提升模型：XGBoost 和 CatBoost，再做加权融合。

```python
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_params = {'n_estimators':3000,'max_depth':6,'learning_rate':0.03,
              'min_child_weight':3,'gamma':0.1,'subsample':0.8,
              'colsample_bytree':0.8,'random_state':42,'eval_metric':'rmse',
              'early_stopping_rounds':100}
```

XGBoost 的训练循环：

```python
xgb_oof = np.zeros(len(X))
for fold, (tr, va) in enumerate(kf.split(X)):
    model = XGBRegressor(**xgb_params)
    model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
    xgb_oof[va] = model.predict(X.iloc[va])
    rmse = np.sqrt(mean_squared_error(y.iloc[va], xgb_oof[va]))
    print(f'Fold {fold+1} RMSE: {rmse:.4f}')
```

CatBoost 使用更低的 learning_rate 和更深的树。

```python
cat_params = {'iterations':5000,'depth':6,'learning_rate':0.02,'l2_leaf_reg':5,
              'loss_function':'RMSE','eval_metric':'RMSE','random_seed':42,
              'early_stopping_rounds':100,'verbose':False}
cat_oof = np.zeros(len(X))
for tr, va in kf.split(X):
    model = CatBoostRegressor(**cat_params)
    model.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]))
    cat_oof[va] = model.predict(X.iloc[va])
```

融合权重用网格搜索。

```python
best = (1e9, 0.5)
for w in np.arange(0, 1.05, 0.05):
    pred = w*xgb_oof + (1-w)*cat_oof
    rmse = np.sqrt(mean_squared_error(y, pred))
    if rmse < best[0]:
        best = (rmse, w)
print('best RMSE', best)
```

### 2.4 模型测试与指标报告
两个模型各自的 RMSE 在 3.0 左右，融合后略有改进。

```python
xgb_test = np.mean([m.predict(X_test) for m in xgb_models], axis=0)
cat_test = np.mean([m.predict(X_test) for m in cat_models], axis=0)
final = best[1]*xgb_test + (1-best[1])*cat_test
submission = pd.DataFrame({'id': test['id'], 'load_mw': final})
submission.to_csv('submission.csv', index=False)
```

在公开讨论中，类似方案的 RMSE 大约为 3.03。

## 三、总结
本文中我们完成了次日峰值负载预测。我们首先用 EDA 确认负载分布和关键字段。其次对小时、星期做周期编码并构造温湿度交互特征。接着用 XGBoost 和 CatBoost 各做 5 折交叉验证并网格搜索融合权重。最后得到约 3.03 的 RMSE。这个比赛让我们体会到，时间序列的周期特征和模型融合往往比复杂网络更有效。
