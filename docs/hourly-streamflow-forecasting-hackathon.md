# Hourly Streamflow Forecasting Hackathon

## 一、比赛简介

### 1.1 比赛背景
这个比赛要预测多个流域的小时级河流流量，单位毫米/小时。流量直接关系到洪水预警和调度。水从降雨到形成流量有滞后，比赛提供降水、短波辐射、温度、干旱指数等气象驱动，以及流域地理类别。数据分为 Track A 与 Track B，Track A 数据完整用于预训练，Track B 样本少采取迁移学习。

### 1.2 比赛目标
输入每个流域每小时记录，包括 date、basin、total_precipitation、shortwave_radiation、temperature、aridity、geol_1st_class、geol_2nd_class、dom_land_cover 以及训练集里的观测流量 qobs_mm_per_hour。输出四个预测列：Q_pred_t、Q_pred_t1、Q_pred_t2、Q_pred_t3，对应当前及未来 1~3 小时流量。

### 1.3 评价标准
比赛用均方根误差 RMSE 打分，数值越低越好。四个时步分别评分，最终按加权平均。RMSE 对极端洪峰误差敏感。

---
## 二、解题过程

我们按流域把时间序列排整齐，构造滞后与累积特征，用 LightGBM 分时步建模，最后加权融合。Track B 做迁移学习，先用 Track A 预训练再微调。

### 2.1 数据探索（EDA）

```python
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
train_a = pd.read_csv(f"{DIR}trainTrackA.csv")
train_b = pd.read_csv(f"{DIR}trainTrackB.csv")
test_a  = pd.read_csv(f"{DIR}testTrackA.csv")
test_b  = pd.read_csv(f"{DIR}testTrackB.csv")
sample_sub = pd.read_csv(f"{DIR}ExampleSubmission.csv")
```

数据显示是「流域 × 时间」长表，测试集仅覆盖 2018 年。降水与温度呈季节性与日周期。qobs_mm_per_hour 仅在训练集出现。

### 2.2 数据清洗

清洗核心是处理时间字段和统一类别编码。

```python
def engineer_features(df, is_train=True):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["dayofyear"] = df["date"].dt.dayofyear
    df = df.sort_values(["basin", "date"]).reset_index(drop=True)
    if is_train:
        df["target_t_0"] = df["qobs_mm_per_hour"]
        for step in [1,2,3]:
            df[f"target_t_{step}"] = df.groupby("basin")["qobs_mm_per_hour"].shift(-step)
        df = df.dropna(subset=[f"target_t_{s}" for s in range(4)])
    return df
```

类别列统一编码，保证训练测试编号含义一致。

```python
cat_features = ["basin", "geol_1st_class", "geol_2nd_class", "dom_land_cover"]
for col in cat_features:
    master_categories = pd.concat([X_train_a[col], X_train_b[col], X_test_a[col], X_test_b[col]]).dropna().unique()
    master_cat_dtype = pd.CategoricalDtype(categories=master_categories)
    for df in [X_train_a, X_train_b, X_test_a, X_test_b]:
        df[col] = df[col].astype(master_cat_dtype)
```

最后删除含缺失值行并丢弃 hour、dayofyear 原始列。

### 2.3 模型训练与超参数调优

周期编码把循环时间映射为圆。

```python
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
```

滞后特征刻画几小时前天气。

```python
for col in ["total_precipitation","shortwave_radiation","temperature"]:
    for lag in [1,3,6,12]:
        df[f"{col}_lag_{lag}"] = df.groupby("basin")[col].shift(lag)
```

累积降水指数用指数加权移动平均，半衰期 6/12/24/72 小时。

```python
for hl in [6,12,24,72]:
    df[f"API_{hl}h"] = df.groupby("basin")["total_precipitation"].transform(lambda x: x.ewm(halflife=hl, min_periods=1).mean())
```

交互特征 temp_x_aridity、radiation_x_aridity 让模型学习条件作用。

LightGBM 两套模型并行：log1p 回归压缩极端值，tweedie 适配大量零值少大值的流量分布。

```python
log1p_params = {"objective":"regression","learning_rate":0.08016,"num_leaves":233,"max_depth":10,
                "min_child_samples":495,"subsample":0.8472,"subsample_freq":4,"n_estimators":800,"random_state":42}
tweedie_base_params = {"objective":"tweedie","tweedie_variance_power":1.152,"metric":"rmse",
                       "num_leaves":235,"max_depth":5,"colsample_bytree":0.8187,"subsample":0.8674,
                       "subsample_freq":7,"reg_alpha":0.1037,"reg_lambda":0.0607,"random_state":42}
```

四个目标分别训练。Track B 迁移学习：先在 Track A 上预训练全局模型再用 Track B 微调，学习率降至 0.005。

### 2.4 模型测试与指标报告

两套预测加权融合，tweedie 占 62%。

```python
w_log1p, w_tweedie = 0.38, 0.62
for target_col, sub_col in zip(lead_times, submission_cols):
    predictions[sub_col] = w_log1p * log1p_preds[target_col] + w_tweedie * tweedie_preds[target_col]
```

生成 Unique_ID 并按样例对齐。

```python
basin_str = predictions["basin"].astype(str).str.zfill(8)
predictions["Unique_ID"] = predictions["date"].dt.strftime("%Y-%m-%d %H:%M:%S") + "_" + basin_str
final_submission = pd.merge(sample_sub[["Unique_ID"]], pred_subset, on="Unique_ID", how="left").fillna(0)
final_submission.to_csv("submission.csv", index=False)
```

本地交叉验证 RMSE 在不同流域波动，整体低于基准。负流量被截断。

---
## 三、总结

本文中我们完成了小时级河流流量预测。我们首先通过 EDA 确认数据为流域×小时长表、测试集仅 2018 年、流量偏零分布。其次按流域排序并统一类别编码，用观测流量构造 t~t+3 四个目标。接着构造周期编码、滞后、API 累积、交互特征；分别用 log1p 回归和 tweedie LightGBM 建模，并在 Track B 上做 Track A 预训练迁移微调。最后以 0.38/0.62 加权融合预测并生成提交。

这个比赛说明时间序列建模依赖滞后与累积特征，迁移学习可解决小样本流域冷启动。
