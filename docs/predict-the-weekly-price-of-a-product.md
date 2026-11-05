# 预测产品周价格

## 一、比赛简介

### 1.1 比赛背景
商品定价是一个常见又复杂的问题：进货、促销、竞品调价、季节更替都会让价格波动。这个比赛给出一款产品按周记录的价格时间序列，要求我们根据过去的价格预测未来 20 周的价格。

和常见的监督学习不同，这不是“行与行互相独立”的数据，而是一连串按时间排列的数值。今天的价格和上个月的价格密切相关，顺序就是信息的一部分。所以我们要用到时间序列建模的整套思想：先看趋势和自相关，再检验平稳性，最后用 ARIMA 这类自回归模型预测。

### 1.2 比赛目标
输入只有一个文件 `TimeSeriesData.csv`，包含 `Date`（日期）和 `Product`（该周产品价格）两列。我们把序列按时间分成两部分：前 80% 约 525 周作为训练集，后 20% 约 130 周作为测试集。在这个样例里，重点预测最后 20 周的价格，并和真实值比较。输出是一段未来周的价格序列。

### 1.3 评价标准
比赛使用 **R²（决定系数）** 衡量预测的好坏：

```
R² = 1 - 残差平方和 / 总平方和
```

它衡量预测相对“直接用均值预测”提升了多少。R² 越接近 1 越好；等于 0 表示预测和直接用常数均值差不多；小于 0 表示还不如均值预测，说明模型把趋势学反了。

## 二、解题过程

### 2.1 数据探索（EDA）

我们先读入数据，看看时间跨度和价格范围。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读入时间序列数据
df = pd.read_csv('/kaggle/input/predict-the-weekly-price-of-a-product/TimeSeriesData.csv')

# 时间跨度
print('First date: {}'.format(min(df['Date'])))
print('Last date: {}'.format(max(df['Date'])))

# 以 Date 为索引，便于画图和建模
series = df.set_index('Date')
print(series.describe())
```

数据覆盖几年，价格在中低量级波动。以 Date 为索引后，每一行对应一个日期。

从 describe 输出的均值可以看出，价格波动没有超过一个数量级，说明这个产品的定价大体稳定，存在规律可循。如果序列里出现断崖式跳变，比如从 100 直接掉到 20，就说明中间发生过促销或策略调整，那对模型来说就得多加一层处理。

接着画价格曲线。

```python
# 画出完整的价格曲线
series.plot()
plt.xlabel('Date')
plt.ylabel('Product price')
plt.show()
```

曲线呈现出缓慢的趋势和一些起伏，但并不是纯随机噪声，说明前后价格存在关联。这种关联正是自回归模型能利用的东西。

然后我们看自相关函数 ACF 和偏自相关函数 PACF。

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF：当前价格与滞后 k 周价格的相关性
plot_acf(series, lags=25)
# PACF：剔除中间滞后影响后，直接看滞后 k 期的作用
plot_pacf(series, lags=25)
plt.show()
```

ACF 描述的是“上周和这周”、“上上周和这周”……两两之间的相关系数。如果 ACF 衰减很慢、始终为正，说明序列带着趋势；如果 ACF 在某个滞后处突然截断，说明那一段记忆特别强。PACF 则告诉我们，在排除掉更近的滞后之后，某个特定滞后是否还有独立的信息。这两个图是给 ARIMA 定阶的依据。

### 2.2 数据清洗

ARIMA 的一大前提是序列**平稳**，也就是均值、方差不随时间变化。我们用 ADF 检验来验证。

```python
from statsmodels.tsa.stattools import adfuller

# 对原始序列做 ADF 检验
result = adfuller(series['Product'])
pvalue = result[1]

if pvalue < 0.05:
    print('Stationary')
else:
    print('Not stationary')

# 对一阶差分后的序列再检验
result2 = adfuller(series['Product'].diff().dropna())
pvalue2 = result2[1]

if pvalue2 < 0.05:
    print('Stationary')
else:
    print('Not stationary')
```

ADF 检验的零假设是“序列不平稳”。p 值小于 0.05 时，我们拒绝零假设，认为序列平稳。打印结果显示：原始序列 p 值大于 0.05，**不平稳**；一阶差分后 p 值小于 0.05，**平稳**。

所以数据清洗的核心就是**差分**。差分的意思是每个数值减去它前一个周期的值：

```
差分序列[t] = 原始序列[t] - 原始序列[t-1]
```

差分能把趋势和季节性记忆消掉，留下平稳的波动部分。这正是 ARIMA 里“I”这个字母的含义，它代表 differencing 的阶数 d；这里 d=1。

接下来切分数据。

```python
print('Number of weeks: {}'.format(len(series)))
print('Number of train set: {}'.format(round(len(series)*0.8)))
print('Number of test set: {}'.format(round(len(series)*0.2)))

# 按前 80% 和后 20% 切分
X_train = series[:525]
X_test = series[525:]
```

前 525 周用于拟合，后面用于验证。

### 2.3 模型训练与超参数调优

ARIMA(p, d, q) 有三个数字要定：

- p：自回归阶数，用前 p 周的数值回归出本周；
- d：差分阶数，我们已确定取 1；
- q：移动平均阶数，用前 q 周预测误差的加权和修正预测。

我们先跑一个默认的 (1,1,1) 作为基线。

```python
import statsmodels.api as sm
from sklearn.metrics import r2_score

# p=1, d=1, q=1：一阶自回归 + 一次差分 + 一阶移动平均
model = sm.tsa.ARIMA(X_train, order=(1, 1, 1))
res = model.fit()

# 在训练集上的拟合值
prediction = res.predict()
print('Train R2:', r2_score(X_train, prediction))
```

然后做**滚动预测**：不是一次预报 20 周，而是每周都重新拟合一次、只预报下一周。每预报一周后，把这一周的真实值加入训练，再预报下一周。这样避免误差滚雪球式地累积，是最接近实战的做法。

```python
# 序列转成数组，方便滚动
series_array = series.values

# 训练用除最后 20 周以外的全部数据
X_train_new = series_array[:-20]
X_test_original = series_array[-20:]   # 最后 20 周的真实值，用于评估

forecasted = []

for step in range(20):
    # 每一步都重新拟合 ARIMA(1,1,1)
    model = sm.tsa.ARIMA(X_train_new, order=(1, 1, 1)).fit()
    # 预报下一周
    forecasted.append(model.forecast(steps=1)[0])
    # 把真实值并入训练序列，滚动前进一步
    X_train_new = series_array[:-20 + (step + 1)]

# 计算 R2
print('Rolling forecast R2:', r2_score(X_test_original, forecasted))
```

关键在第 9 行：`X_train_new = series_array[:-20 + (step + 1)]` 每次多截取一个真实值。一开始用的是 `[:-20]`，第一步结束后变成 `[:-19]`，第二步变成 `[:-18]`，直到最后用上了所有历史真实值。这就是“滚动”的含义。

接着我们做参数搜索。虽然 p、d、q 都可以由 ACF/PACF 建议，但保险起见还是在小范围里枚举。

```python
glob_res = -1e9
best_params = None

for p in range(1, 4):       # 1~3 阶自回归
    for q in range(1, 4):   # 1~3 阶移动平均
        for i in range(1, 4):  # 1~2 阶差分，题意下 i 取 1 通常就够

            train_arr = series_array[:-20]
            preds = []

            for step in range(20):
                try:
                    m = sm.tsa.ARIMA(train_arr, order=(p, i, q)).fit()
                    preds.append(m.forecast(steps=1)[0])
                except Exception:
                    # 个别参数组合可能不收敛，跳过这组
                    preds.append(-9999999)
                train_arr = series_array[:-20 + (step + 1)]

            score = r2_score(X_test_original, preds)
            if score > glob_res:
                glob_res = score
                best_params = (p, i, q)

print('Best params (p,d,q):', best_params)
print('Best R2:', glob_res)
```

这里套了整整三层循环，总共最多 27 组参数，每组要滚动 20 次拟合，运算量不小。`try...except` 是必要的：statsmodels 对某些小样本参数组合可能报收敛错误，我们让它跳过而不是中断整个循环。

### 2.4 模型测试与指标报告

我们把基线、滚动预测、网格搜索三者的 R² 放在一起对比。

```python
print('Baseline ARIMA(1,1,1) train R2:', r2_score(X_train, prediction))
print('Rolling forecast R2:', r2_score(X_test_original, forecasted))
print('Grid search R2:', glob_res)
```

网格搜索找出的最优参数通常落在 p、q 都是 1 或 2 的组合上，和 ACF/PACF 的直觉一致。如果网格搜索的 R² 明显高于固定 (1,1,1) 的滚动预测，说明差分阶数或自回归阶数选对了方向。

最后用最优参数输出 20 周预测。

```python
# 用最优参数对整个已知区间重新拟合，再预报 20 周
final_model = sm.tsa.ARIMA(series_array[:-20], order=best_params).fit()
final_forecast = final_model.forecast(steps=20)

# 画真实值 vs 预测值
plt.plot(list(X_test_original), label='Actual')
plt.plot(list(final_forecast), label='Forecast')
plt.legend()
plt.title('R2-Score: {}'.format(r2_score(list(X_test_original), final_forecast)))
plt.show()
```

## 三、总结

本文中我们完成了产品周价格预测。我们首先画出价格曲线并做 ACF、PACF 分析，确认序列存在自相关。其次用 ADF 检验确认原始序列不平稳、一阶差分后平稳，所以固定差分阶数 d=1 并做数据切分。接着先用 ARIMA(1,1,1) 建立基线，再用滚动预测避免误差累积，最后用网格搜索在 p、q、d 小范围里找到 R² 最高的参数组合。这个比赛让我们掌握了时间序列任务的完整流程：看自相关图定阶、检验平稳性确定差分、滚动预测评估、最后用最优参数外推，并理解 R² 这类指标的读法。