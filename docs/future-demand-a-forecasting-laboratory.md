# Future Demand: Forecasting Laboratory（预测实验室）

## 一、比赛简介

### 1.1 比赛背景

Future Demand 是 Kaggle Playground 第五季第一集的教学版，由 Walter Reade 和 Elizabeth Park 用完全合成数据制作。数据被刻意做成小而清晰，让大家能把**滚动评价、季节性和不确定性**这三个核心概念看清。训练集包含 90 条 country-store-product 系列，每条系列是一个每天的销售时间序列。

比赛用的是三张表：`train.csv`、`test.csv`、`sample_submission.csv`。`train.csv` 里有一些 `num_sold` 为空的行，这些不是销量为 0，而是「尚未观察到」；它们在训练和评分时全部排除。训练期在测试期之前，测试期的真实销量我们看不到。

数据被切分成日频面板（daily panel），只有日期、国别、门店、产品、销量。没有价格、促销、库存等外部字段。未来日期的**日历信息**（星期、季节）和**过去销售**可以用，未来实现的外汇、GDP 等不能用——因为在真实业务里，你在预测那天还不知道它们。

这个比赛不是为了拼成最复杂的模型，而是让大家学会**如何评价远期预测**，以及**滚动原点（rolling origin）**背后的逻辑。

### 1.2 比赛目标

给定训练期到某一天结束的全部信息，预测未来三年的每天销量。评测用了两组滚动原点：

- 原点 2012-12-31，预测 2013-01-01 到 2015-12-31；
- 原点 2013-12-31，预测 2014-01-01 到 2016-12-31。

提交文件格式很简单：`id, num_sold`，共 98550 行。`id` 按训练集的 id 顺序给出，每个 id 对应一个 country-store-product-日期的组合。

### 1.3 评价标准

代码里用到了两个指标：MAPE（平均绝对百分比误差）和 MAE（平均绝对误差）。在业务里 MAPE 更直观——它告诉你预测值平均偏离真实值多少百分比。MAPE 越小越好，MAE 越小越好。

一个重要提醒：两次滚动原点的预测窗口有重叠，它们**不是独立实验**。每个原点、每个三年期、每个一年期的误差都应该单独报告，不能把所有行合在一起算置信区间。

---

## 二、解题过程

这份写up 参考了代码提交「Forecasting Beyond the Last Observation」里的思路：**用两步滚动回测，一个季节基线，一个日历回归，然后固定比例融合**。全流程都遵守一个原则——只用原点之前的信息做特征。

### 2.1 数据探索（EDA）

我们首先读入数据，检查基本审计信息：

```python
import pandas as pd, json
from pathlib import Path

DATA = Path('../data')
train = pd.read_csv(DATA/'train.csv', parse_dates=['date'])
test  = pd.read_csv(DATA/'test.csv', parse_dates=['date'])

audit = {
    'train_rows': len(train),
    'test_rows' : len(test),
    'missing_targets': int(train['num_sold'].isna().sum()),
    'train_start': str(train.date.min().date()),
    'train_end'  : str(train.date.max().date()),
    'test_start' : str(test.date.min().date()),
    'test_end'   : str(test.date.min().date()),
    'series'     : int(train[['country','store','product']].drop_duplicates().shape[0])
}
print(json.dumps(audit, indent=2))
```

输出会告诉我们：训练样本有多少、测试样本有多少、缺失的销量有多少、训练期起止、测试期起止、系列数约 90 条。缺少的 `num_sold` 不是 0，而是未知——我们在拟合和评分时把它去掉。

再看一下面板是否完整：

```python
KEY = ['country','store','product']

# 检查是否有重复日期
assert not train.duplicated(['date']+KEY).any()
# 检查训练期在测试期之前
assert train.date.max() < test.date.min()
# 只保留有观测值的行
observed = train.dropna(subset=['num_sold']).copy()
assert (observed.num_sold>0).all()
```

销售值全部为正，说明 0 不是真实的「没卖掉」，而是缺失。了解这一点很重要：如果把缺失填成 0，模型会学到错误的停滞模式。

### 2.2 数据清洗与特征

这份合成数据本身干净，除了一点：**缺失目标值**要单独处理。做法不是插值，而是直接排除——模型不学习它们的标签，预测时也不计算误差。

真正需要做的是把日历信息算出来，而且只能用原点前已经知道的东西。日历特征包括：

- 星期几（day-of-week），对周末效应有用；
- 季节：用 Fourier 项表示全年周期，避免用星期几那种硬切。

Fourier 项的做法是把「一年中第几天」映射到一个单位圆上，然后用多个正弦/余弦基函数表示季节模式。

```python
import numpy as np

def calendar(frame):
    phase = 2*np.pi*(frame.date.dt.dayofyear.to_numpy()-1)/365.2425
    # 用 1~6 次的正弦和余弦，12 个季节基函数
    return np.column_stack([fun(k*phase) for k in range(1,7) 
                            for fun in (np.sin, np.cos)])
```

我们还把分类字段做成一热编码。特征组合是：

- 系列的整体效果（country、store、product 的组合）；
- 国家+星期几的交互；
- 产品+星期几的交互；
- Fourier 季节项；
- 这个系列自己的趋势项 = (日期 - 原点日期) / 365.2425 天；
- 再把 Fourier 与系列效果做交互，让不同系列有不同的季节形状。

所有这些特征在每个原点都重新拟合，保证不会用到未来信息。

### 2.3 模型训练与超参数调优

我们用了两种互补的模型，然后**固定权重融合**。

#### 2.3.1 季节性基线（SeasonalProfile）

思路很简单：对每个系列，用最近两年的中位数作为季节基线。最近两年说明我们假设最近的销售模式更能代表未来。

```python
class SeasonalProfile:
    def fit(self, frame):
        x = frame.assign(month=frame.date.dt.month, dow=frame.date.dt.dayofweek)
        recent = x.loc[x.date > x.date.max() - pd.DateOffset(years=2)]
        # 按系列、月份、星期几取中位数
        self.table  = recent.groupby(KEY+['month','dow']).num_sold.median()
        # 回退 1：只有系列的中位数
        self.fallback = recent.groupby(KEY).num_sold.median()
        # 回退 2：全局中位数
        self.global_value = float(recent.num_sold.median())
        return self

    def predict(self, frame):
        x = frame.assign(month=frame.date.dt.month, dow=frame.date.dt.dayofweek)
        p = self.table.reindex(pd.MultiIndex.from_frame(x[KEY+['month','dow']])).to_numpy()
        fb = self.fallback.reindex(pd.MultiIndex.from_frame(x[KEY])).to_numpy()
        return np.where(np.isfinite(p), p, np.where(np.isfinite(fb), fb, self.global_value))
```

这个基线的优点是**完全不依赖外部信息**，而且对突变的鲁棒。缺点是它不会学长期趋势。

#### 2.3.2 日历回归（CalendarRegression）

用 Ridge 回归预测 log 销量。因为销量通常乘性增长，取对数能让误差更对称。

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from scipy.sparse import hstack, csr_matrix

class CalendarRegression:
    def fit(self, frame):
        self.origin = frame.date.max()   # 原点是训练数据的最后一天
        self.encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float64)
        self.group   = OneHotEncoder(handle_unknown='ignore', dtype=np.float64)

        cat = self.encoder.fit_transform(categorical(frame))
        group = self.group.fit_transform(frame[['country','product']].agg('|'.join, axis=1).to_frame())
        # 岭回归，alpha=20，数值稳定
        self.reg = Ridge(alpha=20.0, solver='lsqr', tol=1e-6)
        self.reg.fit(self.matrix(frame, cat, group), np.log(frame.num_sold))
        return self

    def predict(self, frame):
        log_pred = self.reg.predict(self.matrix(frame))
        return np.exp(np.clip(log_pred, -10, 15))   # 去对数，限制极值
```

矩阵构建里，除了基础的分类和 Fourier 项，还加入了：**系列与 Fourier 的交互**，以及**系列与趋势（距原点天数）的交互**。这让模型可以学习：不同系列的季节形状不一样，长期趋势也不一样。

为什么选 alpha=20？这里没有做网格搜索，直接用一个中等正则化系数，让模型保持简单，避免在这么少的时间序列上过拟合。我们在回测里发现，进一步调参收益有限，而且会破坏「模型提前固定」的规则。

#### 2.3.3 两年滚动回测

为了让模型在上线前被合理检验，我们做了两轮滚动原点的回测，每轮预测三年，用 MAPE 和 MAE 评估：

```python
rows = []
for origin_year in [2012, 2013]:
    cutoff = pd.Timestamp(f'{origin_year}-12-31')
    fit   = observed.loc[observed.date <= cutoff]
    valid = observed.loc[(observed.date > cutoff) &
                         (observed.date <= cutoff + pd.DateOffset(years=3))]

    reg = CalendarRegression().fit(fit)
    seasonal = SeasonalProfile().fit(fit)

    p_reg    = reg.predict(valid)
    p_season = seasonal.predict(valid)
    blend    = 0.75 * p_reg + 0.25 * p_season   # 固定融合

    for horizon in [1,2,3]:
        ix = valid.date.dt.year == origin_year + horizon
        for name, p in [('seasonal_profile', p_season),
                        ('calendar_ridge', p_reg),
                        ('fixed_blend', blend)]:
            mape = mean_absolute_percentage_error(valid.num_sold[ix], p[ix])
            mae  = mean_absolute_error(valid.num_sold[ix], p[ix])
            rows.append({'origin_year':origin_year,
                         'horizon_year':horizon,
                         'model':name, 'rows':int(ix.sum()),
                         'mape':mape, 'mae':mae})
```

回测结果表明：**日历回归在短期更准，季节基线在长期更稳**。固定 75% 回归 + 25% 基线的融合在两年的误差分布上比较平滑。这也正是「固定融合」的好处：不需要在测试集上调权重。

#### 2.3.4 走向生产

回测完成后，我们用所有观测数据重新拟合一次，然后生成最终预测：

```python
final_reg = CalendarRegression().fit(observed)
final_season = SeasonalProfile().fit(observed)
prediction = 0.75 * final_reg.predict(test) + 0.25 * final_season.predict(test)

# 业务敏感性分析：假设未来年增长率 -5%、0%、+5%
years = (test.date - train.date.max()).dt.days / 365.2425
scenarios = test[['id','date','country','store','product']].copy()
for rate, name in [(-0.05,'lower_drift'), (0,'base'), (0.05,'higher_drift')]:
    scenarios[name] = prediction * (1+rate)**years
```

这一步做的不是预测区间，而是**情景分析**。真实需求规划里，我们需要知道如果增长比预期快或慢，库存要怎么调整。代码把三种情景汇总到国别层面，方便做上层决策。

提交文件生成前会检查行数和 id 一致性：

```python
submission = pd.DataFrame({'id': test.id, 'num_sold': prediction})
assert np.isfinite(prediction).all() and (prediction >= 0).all()
assert len(submission) == 98550
submission.to_csv('submission.csv', index=False)
```

### 2.4 模型测试与指标报告

回测主表 `backtest_scores.csv` 里，每一行对应一个原点、一个预测年份、一种模型。我们把 MAPE 按模型和年份平均后会发现：

- 第一年的 MAPE 明显低于第二年、第三年；
- 日历回归的 MAPE 随年份上升更快，说明长期趋势假设容易漂移；
- 季节基线的 MAPE 相对平稳；
- 融合模型在各年份的误差比较均衡。

这与业务直觉一致：**越远的未来，结构性假设越重要**。用季节中位数兜底，能避免回归模型在极端年份过度自信。

我们还应该把误差按系列拆开看，而不是只报一个总平均。不同国家、门店和产品的销量规模不同，MAPE 对低销量系列尤其敏感；MAE 则更容易被大销量系列影响。因此回测表同时保留 `origin_year`、`horizon_year`、`model` 和 `rows`，后续可以按系列继续分组。若某个系列缺少匹配的月份和星期组合，预测会依次回退到该系列整体中位数、最后才回退到全局中位数，这种回退顺序比用 0 更安全，也能避免测试集出现 NaN。

另外，我们特意在代码里留下几条审计检查：

- `train.date.max() < test.date.min()` 确保没有泄漏；
- `id` 全局唯一且训练测试不重叠；
- 模型在每个原点都重新拟合，而不是一次性拟合；
- 没有用到未来的 GDP、假日等信息。

这些检查都是 Kaggle 时间序列里常见的「踩坑点」，在真实业务中同样适用。

---

## 三、总结

本文中我们完成了 Future Demand Forecasting Laboratory 的完整流程。我们首先确认了数据结构：90 条 country-store-product 系列，训练期在测试期之前、测试覆盖未来三年，且有缺失目标值必须排除。其次我们构建了基于日历信息和 Fourier 季节基函数的特征，并设计了只有用原点前信息的滚动回测框架。接着我们训练了两种互补模型——基于最近两年中位数的季节基线和带趋势交互的日历 Ridge 回归，并用 75/25 固定比例融合。最后我们在两组滚动原点上分别报告一年、两年、三年的 MAPE 与 MAE，验证融合模型在各 horizon 上更平稳，随后用全部训练数据生成提交文件和三种增长情景的规划表。

这个练习给我们最大的启示是：**在时间序列预测里，评价方法比模型复杂度更重要**。随机切分训练/测试会让远期预测的精度被高估，只有按真实业务原点滚动评估，才能知道模型在第一个月和第三年分别有多准。预测只是起点，业务决策需要的是在不确定性下的情景和选择。
