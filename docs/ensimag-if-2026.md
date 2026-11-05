# ENSIMAG IF 2026 电力市场算法交易（Ensimag Trading IF 2026）

## 一、比赛简介

### 1.1 比赛背景

这是法国 ENSIMAG 学院与能源交易公司 Engelhart 合办的一场「算法交易」数据科学挑战赛，时间是 2026 年 9 月 22 日。题目背景是**电力市场**：电力是一种非常特别的商品，很难储存，所以价格在任何时刻都受供需平衡（市场失衡）影响，波动剧烈。交易员做的事，就是在日前市场（DA）和实时市场之间找价差、从中套利。

比赛给我们一张 `train.csv`，里面有一列很关键的 `spread`（价差），其余列是各种市场变量；还有一张 `test.csv` 待预测。另外提供 `imbalances.csv`，记录每个时刻的市场失衡量（供需缺口）。真正的得分方式不是看预测误差，而是看你按预测去下单**能赚多少钱（PnL）**——这是金融类比赛和普通回归比赛最大的不同。

### 1.2 比赛目标

输入是每个时间点的市场变量，输出是对下一期 `spread` 的预测（`forecast`）。下单逻辑由赛方给定：当预测价差 ≥ 0 时，做多（buy）50 MW 的日前电量；预测价差 < 0 时做空（sell）50 MW。我们的目标是让累计收益最大化。

模型优劣用 **P nL（Profit and Loss，盈亏）** 衡量，单位是欧元。赛方还特别强调：当失衡量太大时（`|imbalances| > 1000`）风险过高，直接放弃交易（PnL 记为 0）。

### 1.3 评价标准

提交后按下面的规则逐行算 PnL 并累加：

```python
def calculate_pnl(row):
    pos_mw = 50                       # 每次交易固定 50 兆瓦
    # 失衡太夸张的时候强行不交易：风险太大了
    if abs(row['imbalances']) > 1000:
        return 0
    # 预测价差非负就做多（买入），赚的就是实际价差
    if row['forecast'] >= 0:
        return pos_mw * row['spread']
    # 预测价差为负就做空（卖出），赚的是负价差
    else:
        return -pos_mw * row['spread']
```

所以误差方向比误差大小更重要：**预测符号错了会直接亏钱**。

---

## 二、解题过程

赛题附带的模板只给了一条「线性回归 + 提交」的基线，提示里还点出了几条可以钻研的方向：先做数据探索、检查有没有奇怪/损坏的数据、交叉验证、以及能否加工出新的时间变量。我们沿着这些方向一步步展开。

### 2.1 数据探索（EDA）

首先把训练、测试、失衡三个数据读进来，先跑通赛方给的基线，确保提交管线没问题：

```python
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 训练集：date 解析成时间索引，目标列是 spread
train = pd.read_csv(train_path, parse_dates=['date'], index_col=0)
test = pd.read_csv(test_path, parse_dates=['date'])

# 特征和目标分离：去掉目标列，剩下全是特征
x_train = train.drop('spread', axis=1)
y_train = train['spread']
x_test = test.set_index('date').drop('ID', axis=1)

# 用 plotly 把目标价差画成曲线，直观感受波动范围和结构
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values,
                         mode='lines', name='spread'))
fig.show()
```

把价差曲线画出来之后，我们先统计各变量的分布。电力价差的特点是**剧烈、有尖峰、可能有负值**——如果画出直方图发现存在极端离群点，后面建模时要小心它们把模型拉偏。

EDA 第二件事是整体描述统计和相关性矩阵：

```python
# 每个变量的均值、标准差、min/max：一眼看出哪些变量有离谱的极值
print(train.describe())
# 特征之间、以及特征与 spread 的相关性矩阵
print(train.corr()['spread'].sort_values())
```

`corr()` 那一行尤其有用：它告诉我们哪些市场变量和 `spread` 线性地高度同涨同跌，哪些几乎无关。相关的特征要选进模型，无关的噪声列则可以丢掉，避免给模型添乱。

### 2.2 数据清洗

提示里专门问了一句「有没有奇怪或损坏的数据？模型学这些数据合适吗？」结合赛方给的示例（删掉某段时间内的行），我们做了两类处理：

一是**按时间段剔除异常历史**。有些时段市场结构突变（比如设备检修、政策切换），这些旧规律对预测未来没有帮助，反而会污染模型：

```python
def drop_bad_period(df):
    """把指定时间段内的记录删掉，只保留干净的交易时段"""
    start = '2022-01-01 00:00:00'
    end = '2022-01-02 00:00:00'
    df = df.drop(df.loc[start:end].index)   # 按时间索引切片删除
    return df
```

二是**过滤高风险时刻**。比赛直接把这条规则写进了计分函数，说明只有 `|imbalances| <= 1000` 的时段才是「可交易」的。现实中失衡巨大往往意味着价格无法预测（流动性枯竭），模型再自信也不能下单。

### 2.3 模型训练与超参数调优

赛方建议先问「线性回归真的合适吗？」。我们保留线性回归当基线，同时用 **5 折交叉验证**在本地评估模型，而不是反复提交浪费次数（比赛明文限制了提交次数）：

```python
from sklearn.model_selection import cross_val_predict

# cross_val_predict：自动把训练集切成 5 份，
# 每轮用 4 份训练、1 份预测，最终得到整段历史的「越折预测」
model = LinearRegression()                # 可以替换成更强的模型
pred_cv = cross_val_predict(model, x_train, y_train, cv=5)
```

交叉验证的价值在于**避免数据泄漏**：如果直接在训练集上预测，模型「见过」这些数据，指标会虚高；切出未见过的折来预测，才接近真实表现。得到越折预测后，把它和时间对齐，再和失衡量、真实价差拼成一张表，为下一步的回测做准备：

```python
pred_cv = pd.Series(pred_cv, index=y_train.index, name='forecast')
imbalances = pd.read_csv(imbalances_path, parse_dates=['date'], index_col=0)
# 把预测、失衡、真实价差按时间对齐
concat = pd.concat([pred_cv, imbalances, y_train], axis=1)
```

关于「能不能加新变量」，我们按提示说的方向做实验：从时间戳拆出小时、星期几、是否周末、是否尖峰时段；对某些数值变量做对数、差分变换。这类**时间变量**对电力市场很敏感——价差在峰时（傍晚）和谷时（凌晨）完全是两个世界：

```python
# 给数据框加一列：从时间索引里取出所在的小时
df['your_column_name'] = df['date'].dt.hour   # 例：18 表示傍晚 6 点

# 再加一列「是否尖峰时段」：19 点之后、3 点之前属于用电高峰
df['is_peak'] = ((df['date'].dt.hour > 19) | (df['date'].dt.hour < 3)).astype(int)
```

把「小时 x 是否高峰」这类交叉特征喂给线性模型，相当于手动教它「价差有日内节律」这件事；而对树模型来说，直接喂原始小时数它也能自己分裂学出来，所以时间类特征的设计要看你最终选哪个模型。

### 2.4 模型测试与指标报告

我们把赛方的 PnL 计算公式套到越折预测上，逐行算盈亏并累加，画出累计收益曲线：

```python
# 逐行套用赛方给定的下单与计分规则
concat['pnl'] = concat.apply(calculate_pnl, axis=1)
# 累计收益：把每天的盈亏加起来
concat['pnl_cum'] = concat['pnl'].cumsum()
print(f"Cumulated PnL: {concat['pnl_cum'].iloc[-1]}€")

# 把累计收益画出来，看曲线斜率是否稳定
fig = go.Figure()
fig.add_trace(go.Scatter(x=concat.index, y=concat['pnl_cum'],
                         mode='lines', name='Cumulated PnL'))
fig.show()
```

回测时要重点看的不是最后一个数字，而是**曲线的形态**：稳步上升说明策略靠的是模型判断力；忽上忽下、尾段剧烈摆动，说明策略在靠运气。得到满意结果后，我们对测试集的 `x_test` 做同样的预测流程，把结果放进 `submission.csv`（列名 `forecast`，索引 `ID`）提交即可。

这里要说明的是：材料本身只是赛方提供的**起步模板**，没有给出公开榜上的具体分数或前几名方案，所以我们能确定的只是「从上述流程出发做迭代」这一条路径。可以预期的高分方向是：换用梯度提升树、把失衡量作为特征直接喂给模型（而不是只用它做风险截断）、以及把「预测符号的准确率」当作比 RMSE 更贴近 PnL 的优化目标。

---

## 三、总结

本文中我们完成了 ENSIMAG IF 2026 电力定价比赛的起步流程。我们首先用 EDA 查看了价差曲线和变量分布，理解电力价差的尖峰特征；其次做了数据清洗，剔除异常时段、用失衡量过滤高风险时刻；接着以线性回归为基线做 5 折交叉验证，避免把训练集当测试集带来虚高指标，并按提示方向尝试新增时间变量；最后用赛方给定的 PnL 规则对越折预测做回测、画出累计收益曲线，再生成提交文件。

这个比赛给我们的最大启发是：**金融类问题的评价指标决定了建模目标**。这里 Score 不是预测误差而是真金白银的 PnL，于是「预测符号对不对」远比「预测值有多准」重要，建模、特征、风险过滤都必须围着交易策略本身转，而不是只盯 RMSE。