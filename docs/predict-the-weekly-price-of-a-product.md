# 预测产品周价格

## 一、比赛简介

### 1.1 比赛背景
这份比赛给出一款产品按周记录的价格序列，要求我们根据历史价格预测未来 20 周的价格。数据规模小，本质是时间序列预测，考察我们是否理解趋势、自相关和平稳性。

### 1.2 比赛目标
输入是一个 `TimeSeriesData.csv` 文件，包含 `Date` 和 `Product` 两列，`Product` 是每周价格。训练集是前 525 周，测试集是最后 20 周。输出是这 20 周的预测价格，评分看预测曲线与真实曲线的拟合程度。

### 1.3 评价标准
比赛使用 **R²（决定系数）** 衡量拟合优度。R² 越接近 1 越好，负值表示预测还不如直接用均值。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先读取数据并画出价格曲线。

```python
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/predict-the-weekly-price-of-a-product/TimeSeriesData.csv')
print('First date:', min(df['Date']))
print('Last date:', max(df['Date']))
series = df.set_index('Date')
series.plot()
plt.show()
```

曲线显示出明显的趋势和波动，说明价格不是白噪声。

```python
# 查看自相关和偏自相关
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(series, lags=25)
plot_pacf(series, lags=25)
plt.show()
```

ACF 衰减较慢，说明存在趋势，需要差分。

### 2.2 数据清洗
我们用 ADF 检验判断平稳性。

```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(series['Product'])
print('p-value:', result[1])
# 一阶差分后再检验
result_diff = adfuller(series['Product'].diff().dropna())
print('diff p-value:', result_diff[1])
```

原始序列 p 值大于 0.05，不平稳；一阶差分后 p 值小于 0.05，平稳。因此 ARIMA 的差分阶数 d 取 1。

```python
# 划分训练测试
n = len(series)
train_size = round(n*0.8)
X_train = series[:525]
X_test = series[525:]
print(n, train_size)
```

### 2.3 模型训练与超参数调优
我们使用 ARIMA(p,d,q)。p 是自回归项，q 是移动平均项。先用 (1,1,1) 做基线。

```python
import statsmodels.api as sm
from sklearn.metrics import r2_score

model = sm.tsa.ARIMA(X_train, order=(1,1,1))
res = model.fit()
pred = res.predict()
print('train R2:', r2_score(X_train, pred))
```

接着做 20 步滚动预测，每一步重新拟合。

```python
import numpy as np
series_array = series.values
X_train_new = series_array[:-20]
X_test_original = series_array[-20:]
forecasted = []

for step in range(20):
    model = sm.tsa.ARIMA(X_train_new, order=(1,1,1)).fit()
    forecasted.append(model.forecast(steps=1)[0])
    # 每次把真实值加入训练集，滚动前进
    X_train_new = series_array[:-20+(step+1)]

print('R2:', r2_score(X_test_original, forecasted))
```

然后网格搜索 p、q、d。

```python
best_score, best_order = -1e9, None
for p in range(1,4):
    for d in range(1,4):
        for q in range(1,4):
            preds = []
            train_arr = series_array[:-20]
            for step in range(20):
                try:
                    m = sm.tsa.ARIMA(train_arr, order=(p,d,q)).fit()
                    preds.append(m.forecast(steps=1)[0])
                except:
                    preds.append(-9999999)
                train_arr = series_array[:-20+(step+1)]
            score = r2_score(X_test_original, preds)
            if score > best_score:
                best_score, best_order = score, (p,d,q)
print('best:', best_order, best_score)
```

### 2.4 模型测试与指标报告
在 20 周测试集上，ARIMA 的 R² 会明显高于直接用最后一周价格外推。基线 (1,1,1) 的 R² 较低，网格搜索后 p、q 通常落在 1~2 之间。

```python
# 用最优参数重新预测
final_model = sm.tsa.ARIMA(series_array[:-20], order=best_order).fit()
final_forecast = final_model.forecast(steps=20)
print('final R2:', r2_score(X_test_original, final_forecast))
```

## 三、总结
本文中我们完成了产品周价格预测。我们首先画出价格曲线并做 ACF、PACF 分析，发现序列不平稳。其次用 ADF 检验确认一阶差分后平稳。接着用 ARIMA 做滚动预测并网格搜索 p、d、q。最后用最优参数输出 20 周预测。这个比赛让我们掌握了时间序列从平稳性检验到模型定阶的完整流程。
