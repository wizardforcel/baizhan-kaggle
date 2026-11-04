# CiVilium 价格预测（CiVilium Price Prediction）

## 一、比赛简介

### 1.1 比赛背景

CiVilium 是一种拥有自己交易量和成交价格的数字资产（材料的说明很少，它具体是什么我们无从考证，从文件结构看很像一种加密代币的价格序列）。这个比赛要做的，就是**基于历史交易数据预测 CiVilium 下一段时间的加权价格（Weighted_Price）**。

材料里给出的是一个非常标准的「入门 starter」：只用两列最简单的原始字段，套一个线性回归，就得到一份可以提交的 baseline。它的价值不在于分数多高，而在于把「怎么把一个真实价格预测问题跑通」的骨架完整演示了一遍：读数据、分训练验证、定义损失、训练、评估、生成提交文件。我们这篇 writeup 就以这个 starter 为主线，把每一步的原理讲透，并在最后沿着它自己留下的提示（`try searching data pre-processing and rerun the code`）讨论还能怎么改进。

### 1.2 比赛目标

训练文件 `train_final.csv` 里我们关注的字段有三个：

- `Unix_Timestamp`：交易发生的时间戳（秒级）；
- `Volume_CiVilium`：该时间点的 CiVilium 成交量；
- `Weighted_Price`：加权价格，也就是我们要预测的目标（并不是简单的收盘价，而是按成交量加权过的价格）。

测试文件 `test_final.csv` 只有前两列，我们要为每一个时间点补上 `Weighted_Price`。提交文件要求包含 `Unix_Timestamp` 和 `Weighted_Price` 两列。

### 1.3 评价标准

starter 里明确用 **RMSE（Root Mean Squared Error，均方根误差）** 作为损失度量。它的计算公式是：

```text
RMSE = sqrt( (1 / N) * Σ (真实值 - 预测值)^2 )
```

直观理解就是：把每个样本预测错的量先平方（防止正负误差抵消），取平均后再开根号，结果回到和价格一样的单位（元）。RMSE 越低越好。因为平方的存在，它特别在意「错得离谱」的样本——预测错 10 元的代价是错 1 元的 100 倍，所以优化 RMSE 等于要求模型别犯大错。

> 小提示：和「不管错多大都按原样算」的 MAE 相比，RMSE 对异常值更敏感。如果数据里有几笔异常剧烈的行情，RMSE 会被这几笔带跑，这一点我们在后面改进时会用到。

---

## 二、解题过程

我们首先把整个流程拆成四步：先做 EDA 看看数据形态，再做数据预处理，接着训练线性回归模型，最后评估误差并生成提交文件。

### 2.1 数据探索（EDA）

第一步是读数据。starter 用 pandas 把训练集读进来，然后只保留「时间戳」和「成交量」两列作为特征 `X`，把加权价格作为目标 `y`：

```python
import numpy as np
import pandas as pd

# 读入训练集
df = pd.read_csv('../input/CiVilium/train_final.csv')

# 特征：时间戳 + CiVilium 成交量
X = df[['Unix_Timestamp', 'Volume_CiVilium']]
# 目标：加权价格，也就是我们要预测的东西
y = df[['Weighted_Price']]

# 看前五行，确认列名和数据长什么样
df.head()
```

虽然 EPUB 里没有附输出，但从字段命名能推断出：`Unix_Timestamp` 是单调递增的时间列（这样数据天然是按时间排好序的**时间序列**），`Volume_CiVilium` 是成交量，`Weighted_Price` 是目标价格。所谓「加权价格」可以这样理解：同一时间窗口里成交了好几笔，每一笔的成交价格和数量各不相同，把每一笔价格按「这笔成交量占全部成交量的比例」加权平均，就得到一个比单笔成交价更能代表市场行情的数字。

这里值得先想一个问题：**我们是直接把这些原始数字喂给模型，还是先观察一下它们和目标的关系？** 从直觉上说，成交量大的时候往往价格波动也大，时间戳则携带了价格随时间的趋势信息——这两个特征都和目标「沾边」，但把原始数值直接丢给线性模型，它学到的只是一条粗糙的直线，很容易错过于成交量、时间与价格之间的非线性关系。

一个合格的数据探索至少应该确认三件事：有没有缺失值、各列的量纲差多少、以及目标值本身的分布。这一步 starter 没有展开做，但它为后面的预处理埋了伏笔——我们会在数据清洗那一节看到，`StandardScaler`（标准化）和 `Pipeline`（流水线）已经被 import 进来了，却没有被用上，说明作者自己也意识到原始特征「不太够标准」。

### 2.2 数据清洗

这一节我们要做的其实很轻，但很重要：**把特征和目标的规模对齐**。

- `Unix_Timestamp` 是秒级 Unix 时间戳，动辄十几位的大整数，而 `Weighted_Price` 可能是小数，两者相差好几个数量级；
- 线性回归对特征尺度非常敏感：特征数字太大，梯度更新就会被它牵着走，或者正则化发挥失常。

解决这个问题最标准的工具就是 scikit-learn 提供的 `StandardScaler`（标准化）：它把每一列减去均值、再除以标准差，让所有特征变成「均值 0、方差 1」的量纲。配合 `Pipeline`（流水线）可以把「标准化 → 训练模型」绑定成一步，之后不论是交叉验证还是预测测试集，都不用担心漏掉预处理：

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# 流水线：先标准化，再用线性模型训练。
# 好处是 fit/predict 一次做完预处理，测试阶段不会遗漏步骤
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', LinearRegression()),
])
```

关于缺失值，starter 直接读取就用、没有专门处理，说明作者认为这份数据基本是干净的；但真实的价格数据里仍可能有个别空缺（比如某段时间没有成交记录），稳妥起见可以用 `SimpleImputer` 先填空再走流水线。原始 starter 没走这一步——这也是它自己留下「试试数据预处理再跑一遍」这句提示的原因，我们在这篇里把它补上。

还要提醒一个时间序列数据特有的坑：`train_test_split` 是**随机**割的，会把「时间上靠后的样本」随机塞进训练和验证两边。对价格预测来说，这相当于让模型见过未来再考过去，误差会显得虚低；更严谨的做法是按时间顺序，用「前 70% 的时间段训练、后 30% 的时间段验证」。starter 直接用随机切分来入门是可以的，但心里要清楚它给出的 RMSE 会偏乐观。

### 2.3 模型训练与超参数调优

模型部分，starter 的做法简单直接：按 **70% 训练、30% 验证** 划分数据，然后把线性回归模型拟合到训练集上：

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math
from sklearn.metrics import mean_squared_error

# 70% 训练、30% 验证；固定随机种子保证结果可复现
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=10086)

# 定义比赛用的损失函数 RMSE
def rmse(true, predicted):
    return math.sqrt(mean_squared_error(true, predicted))

# 初始化并训练线性回归模型
lin_reg = LinearRegression().fit(X_train, y_train)

# 分别在训练集和验证集上算误差
print(rmse(y_train, lin_reg.predict(X_train)))
print(rmse(y_val, lin_reg.predict(X_val)))
```

线性回归在这里学到的本质是一条直线：`价格 ≈ θ0 + θ1 * 时间戳 + θ2 * 成交量`。它假设价格和时间、成交量之间存在线性关系。对价格预测来说这个假设相当粗糙——价格变动往往是非线性、波动剧烈的，所以 starter 也承认「用上面的代码会得到一个还算公平的损失」（`will field a fair loss`），意思是「及格但不是高分」。

超参数调优这一步在线性回归里几乎没有可调的超参数，真正的「调优」对象是**特征**。我们会尝试：

1. **滚动窗口特征**：把「过去 N 笔的平均价 / 成交量」加进去，让模型能看到局部趋势；
2. **滞后特征**：把上一时刻的价格当特征（但要注意不能把目标本身泄漏到特征里）；
3. **换模型**：从线性回归换成更灵活的非线性回归（比如树模型或带核函数的方法），看误差能不能降下来。

starter 已经把上面的 pipeline 引好了，我们自己补一段「加滞后特征 + 对比 RMSE」的写法作为调试示范：

```python
# 构造"上一笔成交量"作为滞后特征，观察 RMSE 是否下降
# 注意：这里用的是过去时刻的成交量，不包含当前目标，不会造成泄漏
df['volume_lag1'] = df['Volume_CiVilium'].shift(1)
X2 = df[['Unix_Timestamp', 'Volume_CiVilium', 'volume_lag1']].dropna()
y2 = df.loc[X2.index, 'Weighted_Price']

X2_tr, X2_va, y2_tr, y2_va = train_test_split(X2, y2, test_size=0.3, random_state=10086)
pipe2 = Pipeline([('scaler', StandardScaler()), ('reg', LinearRegression())]).fit(X2_tr, y2_tr)
print('带滞后特征的 RMSE:', rmse(y2_va, pipe2.predict(X2_va)))
```

是否值得保留这个特征，就看验证集 RMSE 有没有真的降下来——这是「特征工程有没有用」的唯一检验标准。

### 2.4 模型测试与指标报告

模型在验证集上的误差打印出来后，就要对真正的测试集做预测并生成提交文件。starter 的收尾很清晰：用训练好（且清洗一致）的模型对 `test_final.csv` 预测，把结果写进 `Weighted_Price` 列，只保留 `Unix_Timestamp` 和 `Weighted_Price` 两列存成 `submission.csv`：

```python
# 读测试集，特征列要和训练时完全一致
df_test = pd.read_csv('../input/CiVilium/test_final.csv')
test_X = df_test[['Unix_Timestamp', 'Volume_CiVilium']]

# 在测试集上预测（用了 pipeline 就自动带上标准化步骤）
test_Y = pipeline.predict(test_X)

# 把预测值放回测试表，生成提交文件
df_test['Weighted_Price'] = test_Y
sub = df_test[['Unix_Timestamp', 'Weighted_Price']]
sub.to_csv('submission.csv', index=False)
```

检查提交文件是否合格，就看两点：列名是不是 `Unix_Timestamp, Weighted_Price`、行数是不是和测试集一致。starter 里打印了训练/验证 RMSE，但 EPUB 只保留了代码、没有保留那些输出数字，所以我们在这里不编一个数字——跑通代码后自行记录即可，用它来判断「哪种特征组合更有效」。

指标报告我们建议至少记录三样：训练集 RMSE、验证集 RMSE、以及测试集预测值的基本统计（最小值、最大值、均值）。三样放一起能看出很多问题：如果训练 RMSE 明显低于验证 RMSE，说明过拟合了（把训练数据背下来了）；如果验证 RMSE 和测试预测的分布「长得很不像训练标签」，说明特征偏移或者提交又回到了「随便猜」的水平。对一个 baseline 来说，能读懂这三个数比多刷 0.01 分更重要。

---

## 三、总结

本文中我们完成了 CiVilium 价格预测比赛的入门流程。我们首先做 EDA，确认只有时间戳、成交量两个原始特征，目标是有加权价格支撑的数值。其次做了数据清洗，注意到时间戳的量纲和目标价格差着好几个数量级，于是引入 `StandardScaler` 配合 `Pipeline` 把特征标准化，解决了线性回归对尺度敏感的问题。接着我们用 70/30 划分训练线性回归模型，并用 RMSE 作为统一的评估标准，尝试加滞后特征来观察验证集误差的变化。最后我们对测试集预测、生成标准格式的提交文件。

这个 starter 留给我们最实在的启发是：**一个再朴素的模型，只要能走得通完整的「读数据 → 训练 → 评估 → 提交」闭环，就是宝贵的起点**；接下来每一步（标准化、滞后特征、换更复杂的模型）都该用同一把尺子 RMSE 去验证它到底有没有真正变好。