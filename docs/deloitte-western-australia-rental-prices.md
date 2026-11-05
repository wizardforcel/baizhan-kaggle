# Deloitte 西澳租金预测（Western Australia Rental Prices）

## 一、比赛简介

### 1.1 比赛背景

这个比赛由德勤（Deloitte）发起，主题是**房地产租金预测**。主办方提供了一组与房产相关的特征（面积、楼层、地段距离、周边配套等），让参赛者根据这些特征预测一套房源的租金价格。房价、租金这类数据有一个很常见的特点：**绝大多数房源的价格集中在某个区间，但总有极少数房源的价格离谱地高或离谱地低**，比如一套豪宅、一笔异常交易。这些「异常点」也就是我们常说的**离群值（outlier）**。

数据里带离群值不是一件小事：很多模型（尤其是线性回归）会尽力去拟合那些少数异常点，结果反而把大多数正常样本拟合坏了。所以「要不要处理离群值、怎么处理」就成了这类价格预测比赛绕不开的问题。我们手上这份记录比赛的 notebook 标题就叫 **Outlier treatment（离群值处理）**，全文都在做一场对照实验：用三种不同的离群值处理方法，分别配合线性模型和树模型，看处理前后分数差多少。整篇 writeup 也以这条主线展开。

### 1.2 比赛目标

这是一个**回归**任务：输入是一套房源的各种特征，输出是一个连续的价格数值。实验里用到的表格有 292 列，除了 `id`、`timestamp` 这类标识和时间字段外，还有大量数值特征，例如：

- `full_sq` / `life_sq`：总面积、居住面积；
- `floor` / `max_floor` / `material`：楼层、最高楼层、建筑材质；
- `num_room` / `kitch_sq`：房间数、厨房面积；
- 一批「周边配套」特征，如 `sport_count_5000`（5 公里内运动场馆数量）、`office_sqm_5000`、`cafe_count_5000_*`、`church_count_5000` 等等；
- 最后一列 `price_doc` 是价格标签，也就是我们要预测的目标。

### 1.3 评价标准

回归比赛一般用**均方根误差（RMSE）**来打分，数值越低越好。RMSE 的算法是：把每条样本预测值和真实值的差求平方，再求平均，最后开根号。可以把它理解成「平均每个预测错得有多远」。实验代码里为了方便，直接统计了**均方误差（MSE）**——因为开根号不会改变分数的大小顺序，所以用 MSE 比较模型好坏和用 RMSE 是一样的。

---

## 二、解题过程

我们首先读入数据、做特征选择，其次对比三种离群值处理手段，接着用 5×5 交叉验证把每个方案都跑一遍，最后对比线性模型和树模型的测试结果。

### 2.1 数据探索（EDA）

我们先读入训练表看看规模和结构：

```python
import numpy as np
import pandas as pd

# matplotlib/seaborn 用来画分布图
import matplotlib.pyplot as plt
import seaborn as sns

# 从 zip 压缩包里直接读训练表，不用先解压
from zipfile import ZipFile
train = pd.read_csv(ZipFile('../input/train.csv.zip').open('train.csv'))

# (30471, 292)：3 万条样本、292 个特征
train.shape
```

打印出来的结果是 **30471 行 × 292 列**。292 列里有相当一部分是「卫生间有几间」「5 公里内咖啡馆数量」这类特征，信息很多，但在动手前我们先要判断：**哪些特征和价格关系最大**。这里我们用最朴素的办法——计算每个特征和价格的皮尔逊相关系数，然后挑相关度最高的几个：

```python
# 去掉有缺失值的列后，一次性算出所有特征之间的相关系数矩阵
correlations = train.dropna(axis=1).corr()

# iloc[-1] 取的是最后一列 price_doc 与所有特征的相关系数
# sort_values() 排序后取最大的 10 个，看谁和价格最相关
correlations.iloc[-1].sort_values()[-10:]
```

结果里和价格最相关的特征几乎都是「面积」和「配套数量」类：`full_sq`（总面积）相关系数约 0.342 排第一，之后是 `sport_count_5000`（0.295）、`trc_count_5000`（0.289）等；而和价格负相关的特征主要是「到市中心、到火车站的直线距离」（如 `zd_vokzaly_avto_km` 约 -0.284、`sadovoe_km` 约 -0.284、`kremlin_km` 约 -0.279），这也符合直觉——离市中心越远越便宜。

既然 notebook 的目的是**对比离群值处理的影响**而不是刷最高分，我们只挑这 6 个高相关特征来快速训练，把其余的先放一边：

```python
# 挑 6 个与价格相关性最强（含强负相关）的特征组成特征矩阵
features = ['sport_count_5000', 'full_sq', 'trc_count_5000',
            'zd_vokzaly_avto_km', 'sadovoe_km', 'kremlin_km']
X = train[features].values

# 价格跨度很大（几百万到上千万），取对数能让分布更像正态，
# 也能压低极端大价格对模型的拉扯
y = np.log(train['price_doc'])

# 画一下取对数后的价格分布，确认它接近钟形
sns.distplot(y)
```

价格取对数这一步很关键：原始价格分布严重右偏（少数房特别贵），而**取对数之后分布变得对称、接近正态**，模型会更好拟合。

### 2.2 数据清洗

这一步是本场比赛的核心。我们从「验一验离群值到底碍不碍事」出发，设计了三种清洗（处理）离群值的方案，它们的思路完全不同：

1. **IQR 法（四分数位距）**：把价格从低到高排，第 25% 分位（Q1）和 75% 分位（Q3）之间的跨度叫 IQR；凡是超出 `Q1 - 1.5*IQR` 到 `Q3 + 1.5*IQR` 这个范围的样本，直接**从训练集里删掉**。
2. **Z-score 法（标准差法）**：先算整列价格的均值 μ 和标准差 σ，把每条样本标准化成 z 分数；凡是 `|z| ≥ 2`（离均值超过 2 个标准差）的样本，也**删掉**。
3. **Winsorizing（缩尾法）**：和删除不同，它把极端值**往边界方向压**——低于 5% 分位的值一律改写成 5% 分位那个数，高于 95% 分位的值一律改写成 95% 分位那个数，样本数量保持不变。

代码里的实现如下，注意这三种处理都只用**训练数据**算统计量、再应用到训练数据上，绝不能让验证集的影子混进统计量里：

```python
from scipy.stats.mstats import winsorize

# 方案一：IQR。删掉价格落在 [Q1-1.5*IQR, Q3+1.5*IQR] 之外的行
def iqr(X_train, y_train):
    q1 = np.percentile(y_train, 25)      # 下四分位数
    q3 = np.percentile(y_train, 75)      # 上四分位数
    iqr = q3 - q1                        # 四分位距
    upper = q3 + 1.5 * iqr               # 上边界
    lower = q1 - 1.5 * iqr               # 下边界
    # 只保留落在这个区间内的样本
    idx = (y_train < upper) & (y_train > lower)
    return X_train[idx], y_train[idx]

# 方案二：Z-score。删掉离均值超过 2 个标准差的样本
def z_score(X_train, y_train):
    z = (y_train - np.mean(y_train)) / np.std(y_train)
    idx = np.absolute(z) < 2
    return X_train[idx], y_train[idx]

# 方案三：缩尾。把两头 5% 的极端值压到 5% / 95% 分位，而不是删除
def winsorizing(X_train, y_train):
    return X_train, winsorize(y_train, limits=[0.05, 0.05])

# 三种方法 + 一个「不处理」的对照组，后面统一拿去交叉验证
```

### 2.3 模型训练与超参数调优

为了公平地比较「不处理 / IQR / Z-score / 缩尾」四种情况，我们没有只切一次训练集就下结论，而是用了 **5 折 × 5 次重复交叉验证（RepeatedKFold）**：把数据切成 5 份，轮流拿 4 份训练、1 份验证算误差，记录每折的 MSE，并重复 5 轮来压随机波动，最后取平均：

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold

def cross_validate(X, y, model, outlier_treatment=None):
    # 5 折、重复 5 次，共 25 次训练-验证，得到更稳的平均误差
    rf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    scores = []
    for train_index, test_index in rf.split(X):
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        # 有离群值处理就先用它清洗训练集，再训练
        if outlier_treatment is not None:
            X_tr, y_tr = outlier_treatment(X_tr, y_tr)
        model.fit(X_tr, y_tr)
        # 记录这 25 次验证中的一次 MSE
        scores.append(mean_squared_error(model.predict(X_te), y_te))
    return np.mean(scores)   # 返回 25 次 MSE 的平均值
```

这个统一的 `cross_validate` 函数把「数据划分 + 离群值处理 + 训练 + 评测」串成一条流水线。值得注意两点：一是离群值处理只施加在**每次交叉验证的训练折**上，验证折一直保持原样，这样测评才不被污染；二是同一个函数既喂线性回归也喂随机森林，**对照组口径完全一致**，出来的分数才可比。

先看不处理时两类模型各自的底子：

```python
# 线性回归 + 随机森林，都不做离群值处理
print(cross_validate(X, y, LinearRegression()))          # 0.4308
print(cross_validate(X, y, RandomForestRegressor(
    random_state=0, n_estimators=10)))                   # 0.2628
```

随机森林（0.2628）明显好于线性回归（0.4308），这其实是个重要信号：**树模型对异常值天然不敏感**——因为树只是按条件划分样本，个别极端样本影响不了划分规则。而线性回归要拟合一条「直线」，极端点会把直线整体拽歪。

接着把三种离群值处理分别套到两类模型上跑一遍：

```python
for model in [LinearRegression(),
              RandomForestRegressor(random_state=0, n_estimators=10)]:
    print(type(model).__name__)
    for treatment in [iqr, z_score, winsorizing]:
        score = cross_validate(X, y, model, treatment)
        print(f'{treatment.__name__:<12} {score}')
```

### 2.4 模型测试与指标报告

上面两段代码跑完，我们得到一张非常规整的对照表（MSE，越低越好）：

**线性回归（LinearRegression）**

| 处理方式 | MSE | 与不处理相比 |
| --- | ---: | --- |
| 不处理 | 0.4308 | 基线 |
| IQR 删除 | 0.4084 | 下降约 0.022 |
| Z-score 删除 | 0.4055 | 下降约 0.025 |
| 缩尾 | **0.3904** | 下降约 0.040 |

**随机森林（RandomForestRegressor）**

| 处理方式 | MSE | 与不处理相比 |
| --- | ---: | --- |
| 不处理 | 0.2628 | 基线 |
| IQR 删除 | 0.2546 | 下降约 0.008 |
| Z-score 删除 | 0.2541 | 下降约 0.009 |
| 缩尾 | **0.2510** | 下降约 0.012 |

读法很清楚：

- **离群值处理确实有效**：在两类模型上，三种处理都比不处理分数更好，说明数据里的极端价格确实在拖累预测。
- **缩尾（winsorizing）是三者中最好的**：线性回归从 0.4308 降到 0.3904，随机森林从 0.2628 降到 0.2510。我们推测原因是删样本会损失信息，而缩尾把极端值「拉回」正常范围，既削弱了异常的影响，又保住了样本数量。
- **树模型提升幅度比线性模型小**：随机森林改进约 0.012，线性回归改进约 0.040。这与我们的预期一致——随机森林本来就不怕离群值，给它处理反而是「锦上添花」；而线性回归最怕离群值，处理后的收益最明显。

顺着这个结论，落地到本场比赛时我们就不再纠结了：**如果最终要用线性模型（比如带 L2 正则的岭回归），缩尾是第一选择；如果打算上树模型（XGBoost、LightGBM 这类），离群值处理可以省一省**，把调参的精力花在更有价值的地方。

---

## 三、总结

本文中我们完成了 Deloitte 西澳租金预测比赛的完整流程。我们首先在 EDA 里识别出「面积、配套数量、离市中心距离」是与价格最相关的 6 个特征，并对价格取对数让分布接近正态；其次我们围绕本场比赛的核心——离群值处理——清洗数据，实现了删除型的 IQR、Z-score 和保样本的缩尾三种方案；接着我们用 5×5 交叉验证把「不处理 / 三种处理」分别配线性回归和随机森林共 8 种组合统一评测；最后我们得到结论：三种处理都有效，缩尾效果最好，且线性模型对离群值更敏感、收益更大（0.4308 → 0.3904），树模型本身抗离群所以提升有限（0.2628 → 0.2510）。

这场比赛给我们的最大启发是：**处理离群值之前，先想清楚自己的模型怕不怕离群值**。树模型对异常点天生免疫，线性模型却会被拽偏，所以「要不要清洗」从来不是一道判断题，而是一道结合模型选择的单选题。