# MLCSPP（多量表股票价格预测）

## 一、比赛简介

### 1.1 比赛背景

MLCSPP 是一项多股票、多量表的金融时间序列预测任务。比赛提供 10 个 CSV 文件，每个文件对应一个股票代码（0056、1101、2327、2385、2449、2915、3005、3532、6176、9945），数据包含历史交易日期对应的价格、技术指标等数值特征。目标是预测 0056 股票的下一个收盘价（`0056_close_y`），并以 MAE / RMSE 等回归指标评估。

### 1.2 比赛目标

输入是多个股票的历史价格序列和相关特征，输出是 0056 股票的下一期收盘价。具体而言，需要把 10 个 CSV 文件合并为一个时间序列表，构造滞后特征和移动平均特征后，用回归模型预测目标列。

### 1.3 评价标准

材料中使用 `mean_absolute_error`（平均绝对误差，MAE）作为本地验证指标，MAE 越低越好。实际 Kaggle 榜单可能使用 RMSE 或 MAE，目标是降低预测值与真实收盘价的绝对偏差。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先列出所有 CSV 文件并合并：

```python
import os, pandas as pd
data_dir = '/kaggle/input/mlcspp/'
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
print('文件数:', len(files))

train_files = [
    '/kaggle/input/mlcspp/0056.csv',
    '/kaggle/input/mlcspp/1101.csv',
    '/kaggle/input/mlcspp/2327.csv',
    # ... 其余文件
]
train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
print('合并后形状:', train.shape)
print('列名:', train.columns.tolist())
```

合并后我们得到一份包含日期、多个股票收盘价、成交量、技术指标的宽表。目标列是 `0056_close_y`，用于预测。描述统计显示目标值存在缺失，部分日期无数据。

### 2.2 数据清洗

我们统一日期列名并按时间排序，处理缺失：

```python
# 兼容不同文件里的日期列名
if 'Date' in train.columns:
    train.rename(columns={'Date':'date'}, inplace=True)

train = train.sort_values('date').reset_index(drop=True)
```

接着我们删除目标全为 NaN 的行，并用前向填充和零填充处理特征缺失：

```python
target_col = '0056_close_y'
mask = train[target_col].notna()
X = train[mask]
```

### 2.3 模型训练与超参数调优

我们首先构造时间序列特征：滞后 1 天、滞后 7 天以及 5 日移动平均。

```python
train['target_lag1'] = train[target_col].shift(1)
train['target_lag7'] = train[target_col].shift(7)
train['target_ma5'] = train[target_col].rolling(window=5).mean()
```

这些特征模拟了股票价格的“惯性”——昨天价格和上周价格通常与今天价格相关。

特征选择仅保留数值列，排除日期和目标：

```python
feature_cols = [c for c in train.columns if c not in ['date', target_col]
                and train[c].dtype in ['float64','int64']]
X = train[feature_cols + ['target_lag1','target_lag7','target_ma5']]
y = train[target_col]
X = X.fillna(0)   # 简单填充
```

采用时间顺序分层：前 80% 作为训练集，后 20% 作为验证集，避免未来信息泄漏：

```python
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
```

我们使用随机森林回归作为基线，参数先设为 `n_estimators=100, max_depth=10, random_state=1`：

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

model = RandomForestRegressor(n_estimators=100, random_state=1, max_depth=10)
model.fit(X_train, y_train)
val_preds = model.predict(X_val)
mae = mean_absolute_error(y_val, val_preds)
print(f'验证 MAE: {mae:.2f}')
```

为提升性能，我们观察特征重要性，优先保留与目标相关的股票价差和滞后特征，并可以调大 `n_estimators` 或移除 `max_depth` 限制。材料中显示改进后的验证 MAE 明显低于基准标准差。

### 2.4 模型测试与指标报告

预测分布图显示模型基本抓住了价格的波动趋势。特征重要性前 10 名通常包含 `target_lag1`、`target_lag7`、`target_ma5` 以及其他相关股票的收盘价差异。

最终预测时，我们对测试数据做同样的预处理：排序、构造滞后特征、填充缺失，得到与训练一致的列集合：

```python
test = test.sort_values('date').reset_index(drop=True)
test['target_lag1'] = test[target_col].shift(1).fillna(0)
test['target_lag7'] = test[target_col].shift(7).fillna(0)
test['target_ma5'] = test[target_col].rolling(5).mean().fillna(0)

X_test = test[feature_cols + ['target_lag1','target_lag7','target_ma5']].fillna(0)
test_preds = model.predict(X_test)
```

我们把预测结果写入提交文件，并检查数值范围是否合理、是否与样本提交格式一致。

---

## 三、总结

本文中我们完成了 MLCSPP 股票价格预测的完整流程。我们首先合并了 10 个股票 CSV 并确认了目标列 `0056_close_y` 的长尾缺失。其次按时间顺序清洗、构造滞后与移动平均特征，避免数据泄漏。接着用随机森林回归做基线，并在时间顺序验证集上评估 MAE，最后生成预测文件。

这个比赛提醒我们：**金融时间序列预测的核心是严谨的时间分割和简单有效的滞后特征**，复杂的模型不一定比随机森林更有优势，先做好数据对齐和特征工程是入门最可靠的路径。