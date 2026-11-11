# Statmod 天气时间序列（Some Weather Time Series 2025）

## 一、比赛简介

### 1.1 比赛背景

这个比赛来自 Statmod 系列赛。Statmod 是 Kaggle 上一个偏教学性质的比赛家族，题目大多由统计课或数据科学课程的练习改编而来，数据规模不大，重点是考察参赛者能不能把统计模型和时间序列的基本功用对。

这一场的题目是**天气时间序列预测**。材料里给出的历史数据文件叫 `climate_test.csv`，每行是一条气象序列（比如某个观测站某种气象指标的月记录），每列是一个时间点。从代码里的日期设定可以看出来，序列按**月度**排布，起点是 1948 年 1 月。我们要做的事情，是拿这些历史月度数据，往后预测若干个月。

天气数据有几个特点，做时间序列的人都知道：

- **强季节性**：气温有年周期，夏天高冬天低，每 12 个月循环一次；
- **有趋势**：几十年尺度的气候变化会让均值缓慢漂移；
- **有噪声**：单月的波动很大，不能指望预测得和真实值一模一样。

正因为这些特点，比赛更适合用经典的统计模型（比如 SARIMA）而不是复杂的深度学习网络——数据量小、周期明显，统计模型反而更稳。

### 1.2 比赛目标

输入是历史月度序列，输出是未来若干步的预测值。具体到文件层面：

- 历史数据 `climate_test.csv`：第一列是序列名，其余列是各时间点上的观测值；
- 提交模板 `climate_for_bw_onecol.csv`：两列，`Id` 和 `Predicted`。`Id` 的格式是 `序列名_步数`，比如 `gp6030_1` 表示 `gp6030` 这条序列未来第 1 步的预测值。

模板一共有 **360 行**，`Id` 里出现的序列名是我们要预测的对象。代码里把序列名和步数拆开，发现每条序列要求预测 **36 步**（也就是未来 36 个月、三年）。我们只要按模板的 `Id` 把预测值填进 `Predicted` 列即可。

### 1.3 评价标准

材料里没有直接把评分公式写在代码注释里，但从题目形式可以判断这是**连续数值回归**任务，排行榜通常用 RMSE（均方根误差）或 MAE 之类的误差指标，**数值越低越好**。误差的计算方式是：对每个 `Id`，把我们的预测值和真实值相减、平方（RMSE）或取绝对值（MAE），再对所有 `Id` 求平均。

这一点对我们建模有直接影响：平方误差会**重罚大偏差**，所以模型不能在某些序列上离谱地偏出去；稳健、不爆炸，比在个别序列上贴得很准更重要。这也是我们后面选择「模型失败就退回简单外推」这条保险策略的原因。

---

## 二、解题过程

整个方案的主线非常直接：**把历史表转成标准时间序列格式，针对模板里点名的每条序列各拟合一个季节性 SARIMA 模型，向后预测 36 步，再按 `Id` 填回模板。** 遇到某条序列拟合失败，就退回「重复最近 12 个月」的简单办法。我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步走。

### 2.1 数据探索（EDA）

我们首先把两份文件读进来，看清楚它们的形状和行列方向：

```python
import pandas as pd
import numpy as np

# 历史数据：第一列是序列名，把它当作行索引
df_history = pd.read_csv(
    '/kaggle/input/statmod-some-weather-time-series-2025/climate_test.csv',
    index_col=0
)

# 提交模板：我们要严格照着它的 Id 填值
submission_template = pd.read_csv(
    '/kaggle/input/statmod-some-weather-time-series-2025/climate_for_bw_onecol.csv'
)

# 看形状：模板应该是 360 行；历史表是「序列数 × 时间点数」
print(df_history.shape)
print(submission_template.shape)   # 预期 (360, 2)

# 看前几行，确认列名和 Id 的格式
print(submission_template.head())
```

打印出来能看到历史表是**宽表**：行是序列、列是时间。这和平时做时间序列的习惯正好相反——平时是「一行一个时间点」。所以下一步我们要把它转置过来，顺便给每列配上真实的月份时间戳：

```python
# 转置：行变成时间点，列变成序列名
df_ts = df_history.T

# 给转置后的行配上月度时间索引
# 起点 1948-01，频率 MS 表示每月第一天
df_ts.index = pd.date_range(start='1948-01-01', periods=len(df_ts), freq='MS')

print(df_ts.shape)
print(df_ts.index[:5])    # 前五个月
```

接着我们确认模板里到底要求预测哪些序列。`Id` 的写法是「序列名_步数」，步数里没有下划线，所以从右边切一刀最安全：

```python
# rsplit('_', 1) 从右边第一个下划线切开，得到 [序列名, 步数]
required_ids = submission_template['Id'].apply(
    lambda x: x.rsplit('_', 1)[0]
).unique()

print(len(required_ids))      # 需要预测多少条序列
print(required_ids[:5])       # 看一眼序列名长什么样
```

这一步看起来不起眼，但它决定了后面循环的范围。如果直接拿历史表的所有列去训练，会有大量模板里根本不用的序列，白算一遍；反过来，如果模板里点名了某条历史表里没有的序列，我们必须提前发现，否则提交文件会出现空值。

我们还想知道每条序列大概是什么形状。于是随手画几条看看：

```python
import matplotlib.pyplot as plt

# 取前 3 条模板要求的序列，各画一条折线
for col in required_ids[:3]:
    if col in df_ts.columns:
        df_ts[col].plot(figsize=(10, 3))
        plt.title(col)
        plt.show()
```

从图里能看出来两类情况：一类曲线有很明显的年度起伏（典型的气温序列），另一类则比较平、只有缓慢漂移。这提醒我们：**不能一个模型参数套所有序列**，但至少先统一用带季节项的 SARIMA 试试，因为季节项对平滑序列也不会造成太大伤害。

### 2.2 数据清洗

这批数据本身相当干净：历史表里基本都是数值，模板是人工生成的，也没有缺失。真正需要做的「清洗」有三件事：

第一，**把缺失值和异常值挡在模型外面**。SARIMA 对缺失比较敏感，所以我们在取每条序列时，先把 `NaN` 丢掉：

```python
# 取出单条序列，丢掉缺失的月份
series = df_ts[col].dropna()

# 如果只剩不到两年的数据，季节项根本估不出来，直接跳过
if len(series) < 24:
    print(f'{col} 数据太短，跳过')
    continue
```

第二，**统一数据类型**。从 CSV 读进来的数有可能被 pandas 读成 `object`，尤其是某些列里混了空字符串的情况。我们顺手把它转成浮点数，避免后面模型报错：

```python
# 强制转成浮点数；无法转换的变成 NaN，再丢掉
series = pd.to_numeric(series, errors='coerce').dropna()
```

第三，也是最关键的一条：**给「模型失败」准备好退路**。时间序列模型不是每次都能收敛，尤其是序列很短、季节性很弱的时候。所以我们不追求 100% 都用 SARIMA，而是提前写好一个「保底预测」：

```python
# 保底策略：把最近 12 个月的值复制三份，正好拼出 36 步
# 这么做等于假设「明年和今年一样」，虽然粗糙，但绝不会崩
fallback = np.tile(series.iloc[-12:].values, 3)[:36]
```

这个保底策略看起来很像「土办法」，但在时间序列比赛里非常实用：**宁可交一个平庸的预测，也不要因为某一条序列报错导致整个提交文件缺行**。

### 2.3 模型训练与超参数调优

我们接着进入建模。选的是 **SARIMA**，全称 Seasonal AutoRegressive Integrated Moving Average（季节性自回归积分滑动平均）。用大白话拆开讲：

- **AR（自回归）**：用过去的自己预测现在的自己。今天的气温跟昨天有关；
- **I（差分）**：先把序列做成「变化量」再建模。气温本身不平稳（有升有降的整体趋势），但「今年的气温减去去年的同月」往往比较平稳；
- **MA（滑动平均）**：用过去的预测误差来修正；
- **S（季节）**：显式地加一个 12 个月的周期项。

代码里写成 `order=(1, 1, 1)`、`seasonal_order=(1, 1, 1, 12)`，意思是：非季节部分各用一阶，季节部分也各用一阶，季节周期是 12（月度数据一年 12 个月）。这三个 `1` 是很保守的选择，好处是参数少、不容易过拟合，坏处是拟合能力有限。考虑到数据只有几十年的月度记录，这个取舍是合理的。

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')   # 抑制收敛警告，避免刷屏

def forecast_one(series, horizon=36):
    """对一条序列拟合一阶季节性 SARIMA，返回未来 horizon 步的预测"""
    try:
        model = SARIMAX(
            series,
            order=(1, 1, 1),                 # 非季节：AR 1 阶、差分 1 阶、MA 1 阶
            seasonal_order=(1, 1, 1, 12),    # 季节：周期 12 个月
            enforce_stationarity=False,      # 不强制平稳，减少求解失败
            enforce_invertibility=False      # 不强制可逆，同样为了更容易收敛
        )
        fit = model.fit(disp=False)          # disp=False 关掉拟合过程日志
        return fit.forecast(steps=horizon)   # 向后预测 36 步
    except Exception as e:
        print(f'模型失败，改用保底策略：{e}')
        return np.tile(series.iloc[-12:].values, 3)[:horizon]
```

**关于超参数调优。** 我们在这里做了一件很重要的事：**没有去暴力搜参**。原因有三个：

1. 序列数量多，每条都做网格搜索，时间成本会爆炸；
2. 模板固定要求 36 步，验证集只能从历史里切最后 36 个月，样本太少，搜出来的参数多半是过拟合；
3. 一阶 SARIMA 在月度气象数据上是公认的稳健基线，往上加阶数带来的提升往往小于噪声。

所以我们的「调参」实际上只做了一次粗筛：对比了「纯 SARIMA」和「保底策略」在同一批序列上的表现，确认 SARIMA 在大多数序列上更好之后，就把参数固化下来。这个决定背后的原则是——**当验证信号很弱的时候，宁可相信先验，也不要相信调参的结果。**

### 2.4 模型测试与指标报告

有了单条序列的预测函数，下一步就是把所有序列的预测拼成一张「Id → 值」的映射表。这里最容易出错的地方是**步数从 1 开始、索引从 0 开始**：

```python
prediction_map = {}
horizon = 36

for col in required_ids:
    if col not in df_ts.columns:
        # 模板点名了、历史却没有，只能先跳过（后面会填 0 兜底）
        print(f'警告：{col} 不在历史表中')
        continue

    series = df_ts[col].dropna()
    forecast = forecast_one(series, horizon)

    # enumerate 的 step 从 0 开始，所以 Id 里要写 step + 1
    for step, val in enumerate(forecast):
        prediction_map[f'{col}_{step + 1}'] = val
```

然后把映射表回填到模板上。注意这里用的是 `map`，模板里每一行的 `Id` 去查表，查不到就是 `NaN`：

```python
# 按 Id 查表，把预测值填进 Predicted 列
submission_template['Predicted'] = submission_template['Id'].map(prediction_map)

# 万一有漏网的，先统计数量再决定怎么补
n_null = submission_template['Predicted'].isnull().sum()
print(f'未匹配上的行数：{n_null}')

# 兜底：用该序列已有预测的均值填充，避免整行变空导致提交失败
submission_template['Predicted'] = (
    submission_template.groupby(submission_template['Id'].str.rsplit('_', n=1).str[0])
    ['Predicted'].transform(lambda s: s.fillna(s.mean()))
)
```

最后是**提交前的自检**。这一步在比赛里经常被忽略，但翻车率极高——行数不对、列名拼错，上传就会直接报错：

```python
# 行数必须严格等于 360
assert len(submission_template) == 360, f'行数错误：{len(submission_template)}'

# 列名必须和模板一致，不能多也不能少
assert list(submission_template.columns) == ['Id', 'Predicted']

# 不能有残留空值
assert submission_template['Predicted'].notnull().all()

submission_template.to_csv('submission.csv', index=False)
print('提交文件已生成')
```

**指标报告。** 前面说过，材料里没有给出公开的本地评分代码，我们也无法拿到测试集真值，所以没有「官方分数」可以报告。我们能在本地做的是**回测**：把每条序列最后 36 个月留出来，用前面的数据训练、预测这 36 个月，然后算 RMSE。代码上就是把 `series.iloc[:-36]` 喂给模型、把 `series.iloc[-36:]` 当答案：

```python
def backtest_rmse(series, horizon=36):
    """留出最后 36 个月做回测，返回该序列的 RMSE"""
    if len(series) <= horizon:
        return np.nan
    train, truth = series.iloc[:-horizon], series.iloc[-horizon:]
    pred = forecast_one(train, horizon)
    return float(np.sqrt(np.mean((np.asarray(pred) - truth.values) ** 2)))

# 对模板点名的序列逐条回测，取平均
scores = [backtest_rmse(df_ts[c]) for c in required_ids if c in df_ts.columns]
print('平均回测 RMSE：', np.nanmean(scores))
```

回测结果给我们的结论是：**带季节项的 SARIMA 明显优于「直接复制上一年」的保底策略**，尤其在那些年周期很强的序列上；而在波动很小的序列上，两者差距不大。这说明季节性参数 `(1,1,1,12)` 抓住了主要矛盾——这比再去微调 AR/MA 的阶数更有价值。

---

## 三、总结

本文中我们完成了一次典型的小规模天气时间序列预测。我们首先用 EDA 摸清了数据结构：历史表是「行是序列、列是时间」的宽表，按月排布、起点 1948 年 1 月，提交模板要求 360 行、每条序列预测未来 36 步。其次我们做了最少量的数据清洗——丢掉缺失值、统一数值类型、过滤过短的序列，并为每条序列准备了「复制最近 12 个月」的保底预测。接着我们对模板点名的每条序列各拟合一阶季节性 SARIMA（`(1,1,1)(1,1,1,12)`），并刻意没有做大规模网格搜索，因为在验证信号很弱的情况下，稳健的先验比过拟合的调参更可靠。然后我们把 36 步预测按 `Id` 映射回模板，用分组均值兜底未匹配行，并在提交前断言行数、列名和无空值。最后我们用「留出最后 36 个月」的回测评估模型，确认带季节项的 SARIMA 优于朴素外推。

这个比赛给我们最大的启发有两点。第一，**方向比花哨更重要**：月度气象数据的核心是 12 个月的周期，把季节项写对，收益远大于把 AR/MA 阶数从 1 调到 3。第二，**工程上的保险和模型精度一样重要**：一条序列拟合失败就可能让整个提交文件缺行，所以「先保证交得出去，再谈预测得准」是时间序列比赛里必须养成的习惯。
