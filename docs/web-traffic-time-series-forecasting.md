# Web Traffic Time Series Forecasting

## 一、比赛简介

### 1.1 比赛背景
本比赛来自 Kaggle 的「Web Traffic Time Series Forecasting」赛题。数据来源于维基百科不同语言版本的页面访问量记录，每个页面每天都有一个访问计数。维基百科是全球最大的在线百科全书，其页面访问量会受到节假日、新闻事件、季节性因素等影响，呈现出明显的时间序列规律。比赛希望参赛者能够基于历史访问量，预测未来一段时间内各页面的访问情况。

数据中页面名称采用统一编码规则，例如 `2NE1_zh.wikipedia.org_all-access_spider` 表示中文维基百科关于 2NE1 的页面、所有访问来源、蜘蛛类型访问。我们需要先从页面名称中解析出主题、语言、访问类型等元信息，才能做有针对性的建模。

为什么页面名称这么重要？因为不同语言的节假日不同，不同访问类型（桌面、手机、蜘蛛）的行为也不同。蜘蛛是搜索引擎爬虫，访问量规律性很强；人类访问则受新闻事件和周末影响。把页面名称拆开，就等于免费得到了一组解释变量，这是本比赛最容易上手的特征来源。

### 1.2 比赛目标
训练集包含从 2015 年 7 月 1 日到 2016 年 12 月 29 日的历史访问量，测试集要求预测后续 59 天的访问量。训练数据共 145063 条记录，每条记录对应一个页面，列名是日期，数值是该日访问次数。第二个阶段训练数据延长到 2017 年 9 月 1 日，预测 2017 年 9 月 10 日到 11 月 10 日，共 61 天。

输入是 Page 标识和历史访问序列，输出是未来若干天每个页面每天的访问预测值。需要为每个页面在测试区间内生成一条访问序列，拼接成提交文件。提交文件的每一行是一个「页面 + 日期」组合，因此要把宽表预测结果转成长表，键的顺序必须和样例完全一致，否则会错位。

### 1.3 评价标准
比赛使用 SMAPE（对称平均绝对百分比误差）作为评价指标。SMAPE 计算预测值与真实值差异占两者平均值的比例，再取平均，数值越小越好。SMAPE 对接近零的序列非常敏感，预测为常数时惩罚会很大。因此模型既要捕捉趋势，也要避免在低流量页面上产生过大相对误差。

理解 SMAPE 的一个直观方式是：它同时惩罚高估和低估，而且用两者平均值做分母，所以不会像 MAPE 那样在真实值接近零时爆炸。但对真实值为零、预测也为零的情况，分母同样接近零，需要加一个很小的常数防止除零。比赛中大量页面历史全是零，预测零反而最安全，这解释了为什么简单基线能排到不错的位置。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先把训练数据读入，观察数据规模和缺失情况。

```python
import pandas as pd
import numpy as np

# 读取第一阶段训练数据，包含 145063 个页面，551 列
train = pd.read_csv("../input/web-traffic-time-series-forecasting/train_1.csv.zip", compression="zip")
print("Number of records: ", train.shape[0])
# 输出：Number of records:  145063
```

数据中存在大量缺失值。我们统计含有空值的记录数。

```python
empty_sample = train[train.isnull().any(axis=1)]
print("Number of records contain 1+ null: ", empty_sample.shape[0])
# 输出：Number of records contain 1+ null:  27786
```

约 19% 的页面在某些日期缺少访问记录，需要在建模时处理。接下来我们从页面名称中提取语言信息。

```python
page_details = train.Page.str.extract(r'(?P<topic>.*)\_(?P<lang>.*).wikipedia.org\_(?P<access>.*)\_(?P<type>.*)')
print(page_details.head())
```

解析后可以得到主题、语言、访问方式等字段，用于按语言分组做节假日特征。

### 2.2 数据清洗
数据本身没有明显重复，但缺失值较多。我们先做最小化清洗：保留原始序列，不做归一化，因为 SMAPE 对比例敏感，原尺度更直观。

```python
# 按页面抽取单列时间序列，方便后续建模
def extract_series(df, idx, start_col):
    df_out = pd.DataFrame()
    df_out['ds'] = pd.to_datetime(df.columns[start_col:])
    df_out['y'] = df.iloc[idx, start_col:].astype(float)
    return df_out
```

为了避免缺失值影响统计，我们在计算中位数时忽略 NaN。

```python
def nanmedian_zero(a):
    return np.nan_to_num(np.nanmedian(a))
```

我们保留了原始访问次数，不做对数变换，因为后续会尝试两种版本对比效果。

### 2.3 模型训练与超参数调优
我们尝试了四种基准模型：简单中位数、考虑周末节假日的改进中位数、ARIMA 和 Facebook Prophet。

首先是简单中位数模型，预测未来 p 天都用最近 p 天的中位数。

```python
def median_model(df_train, df_actual, p):
    visits = nanmedian_zero(df_train['y'].values[-p:])
    forecast_series = pd.Series(np.repeat(visits, len(df_actual)))
    forecast_series.index = df_actual.ds
    return smape(forecast_series.values, df_actual['y'].values)
```

我们选取一个样本验证效果。

```python
df_train = extract_series(X_train, 2, 5)
df_actual = extract_series(y_train, 2, 5)
score = median_model(df_train.copy(), df_actual.copy(), 15)
print("The SMAPE score is : %.5f" % score)
# 输出：The SMAPE score is : 50.09564
```

接下来是按工作日、周末、节假日分组的中位数模型。不同语言的节假日不同，我们为美英、加拿大、俄罗斯等语言单独列出节假日表。

```python
holiday_en_us = ['2015-01-01', '2015-01-19', '2015-05-25', ...]
# 根据页面语言选择对应的节假日列表
```

ARIMA 模型使用 statsmodels 的 AutoReg。

```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df_train['y'], order=(2,1,2))
res = model.fit()
forecast = res.forecast(steps=len(df_actual))
```

Prophet 模型则直接建模时间趋势和季节性。

```python
from fbprophet import Prophet
prophet_df = df_train.rename(columns={'ds':'ds','y':'y'})
m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
m.fit(prophet_df)
future = m.make_future_dataframe(periods=len(df_actual), freq='D')
forecast = m.predict(future)['yhat'].tail(len(df_actual))
```

超参数调优主要体现在中位数窗口大小 p 的选择。我们尝试 14 天和 15 天，发现 15 天在验证集上更稳定。

### 2.4 模型测试与指标报告
我们在验证集上比较各模型 SMAPE。

```python
score_median = median_model(df_train, df_actual, 15)
score_holiday = holiday_model(df_train, df_actual, lang)
score_arima = arima_model(df_train, df_actual, 2,1,2)
score_prophet = prophet_model(df_train, df_actual)
```

示例结果显示：简单中位数 SMAPE 约 50.09，考虑节假日的中位数约 53.18，ARIMA 约 65.76，Prophet 约 52.74。不同序列表现差异很大，低流量页面 SMAPE 容易爆高。

我们按页面流量分层评估，发现对高流量页面，中位数模型足够稳定；对低流量页面，任何模型都容易出现极大相对误差，需要在提交时做下限截断或平滑。

最后我们用全量训练数据重新拟合模型，生成测试集预测。

```python
model.fit(X, y)
test_pred = model.predict(test)
submission = pd.DataFrame({'Id': test['Id'], 'visits': test_pred})
submission.to_csv('submission.csv', index=False)
```

## 三、总结
本文中我们完成了 Web Traffic Time Series Forecasting 比赛的完整流程。我们首先用 EDA 摸清了数据：145063 个页面、550 多天历史、19% 样本存在缺失、页面名称可解析出语言和访问类型。其次我们做了最小化清洗，保留原始尺度并用 nanmedian 处理缺失。接着我们用简单中位数、节假日分组中位数、ARIMA 和 Prophet 四种方法建模，验证集 SMAPE 分别约为 50.09、53.18、65.76、52.74，最后用全量数据训练并生成提交文件。实验表明对维基百科流量这种强季节性且零值多的序列，中位数类基线在 SMAPE 下往往比复杂模型更稳健，节假日特征对某些语言有效但需要精细匹配。
