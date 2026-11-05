# Hourly Streamflow Forecasting Hackathon

## 一、比赛简介

### 1.1 比赛背景
比赛预测美国多个流域的小时级河流水量。提供降水、短波辐射、温度、干旱指数等气象驱动，以及流域地理类别。训练分为 Track A 与 Track B，测试包含 2018 年数据。

目标是为每个流域在未来 0~3 小时的流出量做回归预测，数据具有强时序与流域异质性。

### 1.2 比赛目标
输入包括 date、basin、降水 total_precipitation、短波辐射 shortwave_radiation、温度 temperature、干旱指数 aridity、地理类别等，输出 4 个目标：Q_pred_t、Q_pred_t1、Q_pred_t2、Q_pred_t3，即当前及未来 3 小时流量的毫米/小时。

### 1.3 评价标准
使用均方根误差 RMSE 评估，数值越低越好。四个时步分别评分，最终按加权平均。

---
## 二、解题过程

我们采用特征工程 + LightGBM 的时序建模，先对气象变量做滞后与平滑，再对两条赛道做迁移学习。

### 2.1 数据探索（EDA）

载入四份文件。

```python
import pandas as pd
train_a = pd.read_csv(f"{DIR}trainTrackA.csv")
train_b = pd.read_csv(f"{DIR}trainTrackB.csv")
print(train_a.shape, train_b.shape)
```

查看日期跨度与缺失。basin 为分组键，qobs_mm_per_hour 为观测流量。降水与温度呈现季节性与日周期。

### 2.2 数据清洗

对日期做解析并剔除缺失。

```python
def engineer_features(df, is_train=True):
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['dayofyear'] = df['date'].dt.dayofyear
```

对类别列 basin、geol_1st_class、geol_2nd_class、dom_land_cover 做统一编码，保证训练与测试类别一致。

```python
for col in cat_features:
    master_cat_dtype = pd.CategoricalDtype(categories=master_categories)
    df[col] = df[col].astype(master_cat_dtype)
```

### 2.3 模型训练与超参数调优

特征工程包括周期编码与滞后。

```python
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['day_sin'] = np.sin(2*np.pi*df['dayofyear']/365.25)
```

对降水、辐射、温度做 1、3、6、12 小时滞后，并计算 6、12、24、72 小时指数加权均值 API。

```python
for col in ['total_precipitation','shortwave_radiation','temperature']:
    for lag in [1,3,6,12]:
        df[f"{col}_lag_{lag}"] = df.groupby('basin')[col].shift(lag)
```

交互特征 temp_x_aridity、radiation_x_aridity 被加入。

模型使用 LightGBM，两种目标函数并行：

- log1p 参数用于整体回归
- tweedie 参数用于迁移学习

```python
log1p_params = {"objective":"regression","learning_rate":0.08016,"num_leaves":233}
tweedie_params = {"objective":"tweedie","tweedie_variance_power":1.152,"num_leaves":235}
```

Track B 做迁移学习，先用 Track A 预训练，再在 Track B 上微调，学习率降至 0.005。

### 2.4 模型测试与指标报告

对四个时步分别训练，最后线性融合。

```python
w_log1p = 0.38
w_tweedie = 0.62
predictions[sub_col] = w_log1p*log1p_preds[target_col] + w_tweedie*tweedie_preds[target_col]
```

生成 Unique_ID = date_basin 格式的提交文件。本地交叉验证 RMSE 在不同流域波动，整体低于基准。

---
## 三、总结

本文中我们完成了 Hourly River Streamflow Forecasting 的时序回归。我们首先通过 EDA 确认气象驱动的周期性与流域差异。其次对日期做周期编码并构造滞后、API 与交互特征。接着用 LightGBM 的 log1p 与 tweedie 两套模型，并在 Track B 上做迁移学习。最后以 0.38/0.62 权重融合预测并导出提交。

该比赛说明在小样本流域上，迁移学习与时序滞后特征是提升泛化的关键。
