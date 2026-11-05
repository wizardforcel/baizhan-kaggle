# 网页流量时间序列预测

## 一、比赛简介

### 1.1 比赛背景
Web Traffic Time Series Forecasting 是 Kaggle 经典时间序列预测赛。数据为维基百科各语言版本网页的日访问量，涵盖 2015-2016 年。目标是预测未来 2 天的访问量，共有 145063 个页面时间序列。

难点在于序列长度长、季节性强、不同页面流量尺度差异巨大。部分页面为热门主页，访问量达十亿级别；大部分为长尾页面，访问量接近零。评价指标为 SMAPE，对零值敏感。

### 1.2 比赛目标
输入训练集 `train_1.csv` 和 `train_2.csv`，格式为宽表：第一列 Page，其余列为日期 2015-07-01 到 2016-12-31 的每日访问量。测试集需预测未来若干天的访问量。

目标是为每个页面生成未来天的预测值，提交列包含 Page 和各预测日期。

### 1.3 评价标准
使用对称平均绝对百分比误差 SMAPE：
```
SMAPE = 100% * mean( |actual - pred| / ((|actual|+|pred|)/2) )
```
越小越好，对零值有平滑处理。比赛要求按页面独立建模再汇总。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先读取 train_1，观察规模。

```python
import pandas as pd
train = pd.read_csv('train_1.csv.zip', compression='zip')
print(train.shape)  # 145063 x 551
print(train.head())
```

有 145063 个页面，550 天数据。语言分布不均，英文最多，其次日文、德文。

```python
import re
def get_lang(page):
    m = re.search(r'([a-z]{2})\.wikipedia\.org', page)
    return m.group(1) if m else 'na'
train['lang'] = train['Page'].map(get_lang)
print(train['lang'].value_counts())
```

流量分布呈长尾，平均周趋势存在显著周期。

### 2.2 数据清洗
数据中存在零值和缺失。

```python
# 转置为长格式
df_long = train.melt(id_vars=['Page'], var_name='date', value_name='views')
df_long['date'] = pd.to_datetime(df_long['date'])
df_long['views'] = df_long['views'].fillna(0)
```

对缺失值填充 0。我们不做删除，保持完整序列。按语言分组可提高模型稳定性。

### 2.3 模型训练与超参数调优
采用简单基线：历史均值 + 周季节性。

```python
# 计算每页面周均值
train['weekday'] = pd.to_datetime(train.columns[1:]).weekday
weekly_avg = train.groupby('weekday').mean()
```

我们用 Prophet 做快速建模：

```python
from prophet import Prophet
def forecast_page(df_page):
    df_p = df_page.reset_index()
    df_p.columns = ['ds','y']
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.fit(df_p)
    future = m.make_future_dataframe(periods=2)
    pred = m.predict(future)
    return pred['yhat'].values[-2:]
```

对 145k 页面全部拟合成本高，实际采用聚合预测：按语言和流量层级做分层预测，或用移动中位数。

```python
# 简单移动中位数基线
window = 7
rolling = train.iloc[:,1:].rolling(window, axis=1).median(axis=1)
```

也可使用 LSTM，构建序列到序列预测，对长尾页面用共享模型。

### 2.4 模型测试与指标报告
在历史滚动窗口上验证 SMAPE。

```python
def smape(y_true, y_pred):
    return 100/len(y_true)*np.sum(np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)+1e-9))
```

我们对验证集后 7 天预测，计算整体 SMAPE。热门页面用 Prophet 效果较好，长尾页面用全局中位数更稳健。

```python
submission = pd.DataFrame({
    'Page': test_pages,
    '2017-01-01': pred_day1,
    '2017-01-02': pred_day2
})
submission.to_csv('submission.csv', index=False)
```

## 三、总结
本文中我们对网页流量时间序列预测任务进行了研究。我们首先探索了 14 万多个页面、550 天的宽表数据，识别出语言分布和长尾特性；其次做缺失值填充和语言标记清洗；接着用移动中位数和 Prophet 做季节性预测，并考虑层级建模；最后以 SMAPE 评估并生成提交。流量具有强周周期，简单季节基线已能取得不错成绩，复杂模型需权衡计算成本。
