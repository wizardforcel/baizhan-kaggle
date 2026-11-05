# 股票价格预测（Stock Price Predictor）

## 一、比赛简介

### 1.1 比赛背景

Stock Price Predictor 是一个以**股票次日涨跌方向**为主题的 Kaggle 练习赛。和很多「用机器学习预测股价」的入门项目一样，题目其实并不要求预测具体价格，而是让参赛者判断：**明天的收盘价比今天高还是低？** 这是一个二分类问题。

数据是**日线行情**：每天的开盘价（Open）、最高价（High）、最低价（Low）、收盘价（Close）。材料里的审计脚本显示，训练表和测试表都只有这几列外加日期，数据按日期升序、每天一行，没有重复日期，也没有缺失单元格。审计脚本还检查了 OHLC 的合法性（最高价不能低于开盘/收盘/最低，最低价不能高于其他），全部通过。这说明主办方给的是相对干净的历史行情。

这个赛道的难点不在数据处理，而在**目标本身的不可预测性**。有效市场假说认为日线级别的价格变动接近随机，任何模型能学到的都是微弱的统计规律。材料里也很有意思地记录了一个「诚实」的发现：代码作者用一个日历基线（按星期几统计历史上涨概率）就能拿到接近 0.7 的准确率，而复杂的 AR/ARX 模型反而不比它好多少。

### 1.2 比赛目标

输入：

- `Train.xlsx`：列 `Date, High, Low, Open, Close, Target`，`Target` 是 0/1 标签；
- `Test.xlsx`：列 `Date, High, Low, Open, Close`，没有 `Target`；
- `sample_submission.xlsx`：列 `Id, Target`，`Id` 从 0 到 364，共 365 行。

输出 `Id, Target`，365 行，`Target` 是 0 或 1。

**目标是怎么定义的？** 审计脚本用一行代码验证过：

```python
expected = (train.Close.shift(-1) > train.Close).iloc[:-1].astype(int)
```

意思是：如果**明天的收盘价高于今天**，`Target = 1`，否则为 0。审计脚本把它和训练集给出的 `Target` 做了比较，完全一致。训练集最后一行没有「明天」，所以它的标签没有意义，审计里把它排除掉了。

### 1.3 评价标准

这是二分类任务。材料里代码作者的记录显示公开排行榜分数在 0.63~0.68 之间，说明指标是**准确率（Accuracy）**，即预测正确的比例，越高越好。满分 1.0，随机猜是 0.5。

值得注意的是，股票日线涨跌的基准比例通常略偏向上涨（长期看股市有正漂移），所以「全预测 1」的朴素策略也可能拿到 0.5~0.55 左右。要想显著超过它，需要真正找到可复现的信号。

---

## 二、解题过程

我们的主线有两条，分别对应材料里的两份代码：

1. **随机森林基线**：把 OHLC 及其衍生差值当特征，直接分类；
2. **日历基线 + 全局取反**：统计每个星期几的历史上涨概率，按概率阈值预测，并在公开排行榜反馈后做一次全局 0/1 翻转。

前者是通用的机器学习做法，后者是非常「金融比赛特色」的做法。我们按 EDA、清洗、训练、报告四步把它们串起来。

### 2.1 数据探索（EDA）

我们首先读入三个 Excel 文件，并统一列名大小写：

```python
import pandas as pd
import numpy as np

BASE = '/kaggle/input/stock-price-predictor/'

def load_file(filename):
    """先找主目录，再找 StockTrends 子目录，最后尝试 CSV"""
    for path in [BASE + filename, BASE + 'StockTrends/' + filename]:
        try:
            if filename.endswith('.xlsx'):
                return pd.read_excel(path)
            return pd.read_csv(path)
        except Exception:
            continue
    return None

train_df = load_file('Train.xlsx')
test_df = load_file('Test.xlsx')
sample_sub = load_file('sample_submission.xlsx')

# 列名统一转小写，避免 Open/open 大小写问题
train_df.columns = [c.lower() for c in train_df.columns]
test_df.columns = [c.lower() for c in test_df.columns]

print(train_df.shape, test_df.shape, sample_sub.shape)
```

审计脚本记录的检查结果值得在这里复述，因为它直接决定了我们后面可以放心地做哪些假设：

- 训练表 1461 行、测试表 365 行，日期严格递增、每天一行；
- 缺失单元格为 0，重复日期为 0；
- OHLC 全部合法；
- 训练集最后一行（例如 2019-11-21）的 `Target` 来自一个指向下一行的公式 `=IF(E1462>E1461,1,0)`，而下一行不存在，所以**最后一行标签应被排除**。

再看目标分布：

```python
# 上涨/下跌比例
print(train_df['target'].value_counts(normalize=True))

# 按星期几看上涨比例，这是后面日历基线的依据
train_df['date'] = pd.to_datetime(train_df['date'])
train_df['weekday'] = train_df['date'].dt.dayofweek
print(train_df.groupby('weekday')['target'].mean())
```

如果某些星期几的上涨比例明显偏离 0.5，就说明「日历」本身携带信息。材料里的基线模型正是这么做的。

我们还想看价格和标签的关系。画一条收盘价曲线，再叠加涨跌点：

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(train_df['date'], train_df['close'])
plt.title('Close price over time')
plt.show()
```

曲线呈现出明显的趋势段和震荡段，这解释了为什么「只看日历」也能有一定的预测力——在长期上涨的市场里，工作日的上涨概率天然更高。

### 2.2 数据清洗

清洗阶段主要有三件事。

**第一，处理训练集最后一行。** 它的标签没有对应的「明天」，必须排除，否则会把「公式残留」当成监督信号：

```python
# 去掉最后一行的无效标签
train_df = train_df.iloc[:-1].copy()

# 再确认一遍标签和「次日收盘更高」完全一致
check = (train_df['close'].shift(-1) > train_df['close']).iloc[:-1].astype(int)
print('标签一致：', check.equals(train_df['target'].iloc[:-1].astype(int)))
```

**第二，数值清洗。** 虽然审计说没有缺失，但读取 Excel 时仍可能出现字符串混入。统一转成浮点、遇到无法解析的填 0：

```python
for col in ['open', 'high', 'low', 'close']:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
```

**第三，构造不会泄漏的特征。** 这是最关键的一步。做金融特征时最常见的错误是用「未来信息」：比如用整段数据的均值做标准化，或用 `shift(-1)` 造特征。我们的原则是**每个特征只能用当天及之前的数据**：

```python
def create_features(df):
    df = df.copy()
    # 当日振幅和实体：只用当天 OHLC
    df['open_close_diff'] = df['close'] - df['open']       # 当日涨跌
    df['high_low_diff'] = df['high'] - df['low']           # 当日波动幅度
    # 3 日均线：rolling 默认向后看，只用过去 3 天
    df['ma_3'] = df['close'].rolling(window=3).mean()
    # 昨日收盘：shift(1) 把昨天的值挪到今天
    df['prev_close'] = df['close'].shift(1)
    return df
```

这里每个特征的时间方向都很清楚：`rolling(3)` 看过去，`shift(1)` 看昨天。绝对不能出现 `shift(-1)` 或 `center=True`，否则就是在偷看答案。

训练集和测试集用同一个函数处理：

```python
X_train_full = create_features(train_df)
X_test_full = create_features(test_df)

features = ['open', 'high', 'low', 'close',
            'open_close_diff', 'high_low_diff', 'ma_3', 'prev_close']
# 只保留实际存在的列，兼容列名差异
features = [f for f in features if f in X_train_full.columns]

X = X_train_full[features].fillna(0)
y = train_df['target']
X_test = X_test_full[features].fillna(0)

print('特征列：', features)
print('训练集形状：', X.shape)
```

### 2.3 模型训练与超参数调优

**方案一：随机森林分类器。** 这是一个合理的基线：对特征尺度不敏感、能自动学非线性、训练快。

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,     # 100 棵树
    max_depth=10,         # 限制深度，防止把噪声背下来
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

predictions = model.predict(X_test)
```

`max_depth=10` 在这里很重要。日线数据信噪比很低，树越深越容易记住训练集里的偶然模式，验证集和测试集就会崩。限制深度是最便宜的防过拟合手段。

**方案二：日历基线。** 材料里的审计文档记录了一种更简单也更「金融」的做法：按星期几统计历史上涨概率，概率 ≥ 0.5 就预测 1：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CalendarModel:
    rates: dict      # 星期几 -> 历史上涨概率
    fallback: float  # 全局平均上涨概率

    def predict(self, frame):
        # 每一行按它的星期几查概率，查不到用全局平均
        rates = frame['date'].dt.dayofweek.map(self.rates).fillna(self.fallback)
        return (rates >= 0.5).to_numpy(dtype=int)

# 用训练集拟合：每个星期几的上涨概率
rates = train_df.groupby(train_df['date'].dt.dayofweek)['target'].mean().to_dict()
calendar = CalendarModel(rates, float(train_df['target'].mean()))
calendar_pred = calendar.predict(test_df)
```

**超参数调优方面**，材料里的记录显示作者做了严格的模型比较：用「净化窗口」（purged window）做时间序列交叉验证，对比日历基线、逻辑回归和直方图梯度提升，并设置了选择裕度——只有候选模型在**每个窗口的平均准确率**和**最差窗口准确率**上都超过基线达到一定幅度，才替换基线。这种「先定规则再选模型」的做法，比事后挑一个好看的分数要可靠得多。

材料还记录了几个尝试过的方向：

- **AR(5)/ARX(5) 回归**：用过去 5 天的收益率预测明天收益率，再转成涨跌。结果不如日历基线；
- **事件时钟**：用成交量或大幅波动事件作为「时间刻度」，再在事件之间做预测。也没有通过「提升 ≥ 1 个百分点」的验收门槛；
- **逻辑回归 + 标准化**：特征工程相同，但线性模型在低信噪比数据上往往不如日历基线稳。

最终入选的主线是日历基线。

### 2.4 模型测试与指标报告

**本地评估。** 因为测试集没有标签，我们在训练集内部用时间顺序切分（不能用随机切分，否则会泄漏），并且**在训练和验证之间留出间隔**，避免「明天」标签跨过边界：

```python
def purged_split(df, valid_start, gap=1):
    """按日期切分，并在训练/验证之间留 gap 天"""
    train = df[df['date'] < valid_start - pd.Timedelta(days=gap)]
    valid = df[df['date'] >= valid_start]
    return train, valid

train_part, valid_part = purged_split(train_df, pd.Timestamp('2018-01-01'))

# 用前半段拟合日历模型，后半段验证
rates = train_part.groupby(train_part['date'].dt.dayofweek)['target'].mean().to_dict()
m = CalendarModel(rates, float(train_part['target'].mean()))
pred = m.predict(valid_part)

acc = (pred == valid_part['target'].values).mean()
print(f'验证准确率：{acc:.5f}')
```

材料里的审计报告还给出了一个重要的参照：**「全预测多数类」在可用训练集上的准确率**。这个数字是判断模型有没有真本事的底线——如果日历模型的验证准确率不比「全猜上涨」高，那它就没有价值。

**提交文件生成。** 这里有两处材料里特别强调的坑：

```python
submission = pd.DataFrame()
submission['Id'] = range(len(predictions))
submission['Target'] = predictions

# 坑 1：必须正好 365 行，少了要补齐（补 0 只是兜底）
expected_rows = 365
if len(submission) < expected_rows:
    missing = expected_rows - len(submission)
    last_id = submission['Id'].max()
    pad = pd.DataFrame({
        'Id': range(last_id + 1, last_id + 1 + missing),
        'Target': [0] * missing
    })
    submission = pd.concat([submission, pad], ignore_index=True)

submission['Id'] = submission['Id'].astype(int)
submission.to_csv('submission.csv', index=False)
```

**坑 2：全局取反。** 材料记录了一个非常值得讨论的现象：同一套日历基线，原始方向在公开榜上只有 **0.32054**，把 0/1 全部翻转后变成 **0.67945**，排名第 2。原因是排行榜只公开了一部分测试样本，而在这部分样本上，「涨/跌」的历史比例恰好和训练期相反。

我们在写这篇 writeup 时必须诚实地说清楚：**这种全局取反是看了公开排行榜之后做的选择，不是独立验证出来的结论**。它利用了排行榜反馈，属于「对公开测试集的过拟合」，在新数据上没有任何保证。材料作者自己也明确写了 "This is not independent validation or a claim of research novelty"。作为学习者，我们应该把它当成一个**教训**而不是**技巧**：

```python
# 全局取反：仅用于理解「方向偏差」，不能当作可复现的方法
final_prediction = 1 - calendar_pred

submission = pd.DataFrame({'Id': range(len(final_prediction)),
                           'Target': final_prediction})
submission.to_csv('submission.csv', index=False)
```

**指标报告。** 综合材料，我们可以给出这样的结论：

- 随机森林基线的本地验证准确率在 0.5x 量级，基本贴着「多数类」的水平，说明 OHLC 特征在日线尺度上信息量很少；
- 日历基线在开发窗口上略高于多数类基线，但提升幅度很小；
- 公开榜 0.63835 和 0.67945 两个分数分别对应「直接提交」和「全局取反」，后者的高分离不开排行榜反馈。

---

## 三、总结

本文中我们完成了 Stock Price Predictor 的完整建模。我们首先用 EDA 和审计确认了数据结构：1461 行日线训练数据、365 行测试数据、OHLC 四列、无缺失、标签定义为「次日收盘价是否更高」，最后一行标签无效；其次做了防泄漏的特征工程，只使用 `rolling` 和 `shift(1)` 这类向后看的操作，构造了日内差值、3 日均线、昨日收盘等特征；接着对比了两条主线——限制深度的随机森林，和按星期几统计上涨概率的日历基线，并用净化窗口交叉验证和「必须超过多数类基线」的规则来选择模型；最后在生成 365 行提交文件的同时，诚实记录了公开榜 0.32054 与取反后 0.67945 之间的差异，并说明这属于排行榜反馈利用而非独立验证。

这个比赛给我们的最大启发是**对金融预测保持敬畏**。日线涨跌信噪比极低，复杂模型（ARX、梯度提升）并没有打败一个只有 5 个参数的日历基线，这本身就说明问题的上限在哪里。更重要的是，0.32 到 0.68 的跳跃不是模型变强了，而是方向赌对了——这种「分数」不能当成能力。真正可迁移的经验有三条：一是特征必须严格因果，任何 `shift(-1)` 都是作弊；二是验证必须按时间切分并留间隔，否则标签会跨边界泄漏；三是永远先算「多数类基线」，因为在不平衡或方向性明显的任务里，它才是真正的及格线。
