# London Airbnb 房价预测（Tabular Long-Form Competition）

## 一、比赛简介

### 1.1 比赛背景

这个比赛来自 UGCML / Harbour Space 的表格数据课程练习，题目叫「Tabular Long-Form Competition」。所谓「长表」（long-form）是相对「宽表」说的：宽表把每个特征放在一列里，长表则常常把「一个对象的多条记录」堆叠在一起。不过在这个比赛里，我们拿到的仍然是一张普通的房源表，只是课程希望我们按照完整的表格建模流程，从读数据、做特征、交叉验证一直走到生成提交文件。

比赛使用的数据是伦敦的 Airbnb 房源信息。Airbnb 是一个短租平台，房东把房子挂在上面，按晚出租。房源表里记录了这套房子能住几个人、有几间卧室、几张床、最少住几晚、收到了多少条评价、房间类型等信息。我们要根据这些信息，预测这套房源一晚的价格（单位是英镑 £）。

这个题目的难点主要有三个。第一，**价格分布很不均匀**：大部分房源价格不高，少数豪华房源价格非常高，直接预测价格会让模型被高价房「带偏」。第二，**特征类型混杂**：既有 `accommodates`、`bedrooms` 这样的数字，也有 `room_type` 这样的类别，还有 `amenities` 这种用 JSON 字符串存的长文本。第三，**很多字段有缺失值**，比如卧室数、床数，房源主人不一定填。

比赛材料里的 baseline 特意提醒我们：要注意特征工程，并且可以按照课程进度逐步升级模型，先线性模型，再树模型。我们这篇 writeup 就沿着这条主线，把 baseline 讲清楚。

### 1.2 比赛目标

输入是一张训练表和一张测试表。训练表里有房源的各种属性和目标列 `price`；测试表只有属性，没有 `price`。我们要为测试集里的每一行预测一个价格。

主要字段的含义如下：

- `id`：房源编号；
- `price`：每晚价格，单位英镑，是我们要预测的目标；
- `accommodates`：这套房子最多能住多少人；
- `bedrooms`：卧室数量；
- `beds`：床的数量；
- `minimum_nights`：最少入住晚数；
- `number_of_reviews`：累计收到的评价条数；
- `room_type`：房间类型，例如整套房子、单独房间等；
- `amenities`：设施列表，以 JSON 字符串形式保存；
- `bathrooms_text`：卫生间数量和类型，用文字描述；
- `geo_cell`：把地图切成大约 500 米 × 500 米的小格子后，给每个格子分配的匿名编号；
- `property_type`、`host_id`、评论分数、评论日期、`name`、`description` 等：房源和房东的补充信息。

输出文件必须包含两列：`id` 和 `price`。其中 `id` 要和测试集一一对应，`price` 是我们预测的每晚价格。

### 1.3 评价标准

比赛使用 **RMSLE**（Root Mean Squared Logarithmic Error，均方根对数误差）作为评价指标。它的计算方法是：先把预测值和真实值都做 `log1p` 变换，也就是 `log(1 + x)`，然后算两者差的平方、求平均、再开方。

用公式写就是：

```
RMSLE = sqrt( mean( (log1p(pred) - log1p(true))^2 ) )
```

这个指标有两个特点。第一，它衡量的是「相对误差」而不是「绝对误差」：预测 100 块、真实 110 块，和预测 1000 块、真实 1100 块，在 RMSLE 眼里差不多。第二，它对预测过低更敏感一些，因为 `log1p` 在 0 附近变化很快。

RMSLE 是**越低越好**。这也解释了为什么后面的模型要在对数尺度上思考价格：如果我们直接对 `price` 做线性回归，高价房源会主导损失；先取对数，价格的范围被压缩，模型更容易学到规律。

---

## 二、解题过程

整场比赛的思路可以概括为一句话：**先把能直接用的数值特征和类别特征整理好，用一个线性回归建立可解释的 baseline，再用交叉验证确认它没有过拟合。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，看看数据的规模和列名：

```python
import numpy as np
import pandas as pd

# 数据放在 Kaggle 的输入目录里，路径写成常量方便后面复用
DATA = "/kaggle/input/competitions/tabular-long-form-competition-ugcml-harbour-space"

# 训练集文件名里带空格和括号，读的时候要原样写
train = pd.read_csv(f"{DATA}/train (1).csv")
test = pd.read_csv(f"{DATA}/test.csv")

# 先确认行数和列数，避免把测试集当训练集读进来
print("train:", train.shape, " test:", test.shape)
train.head(3)
```

打印 `shape` 是为了先建立一个「数据有多大」的直觉。行数决定我们能不能承受复杂的模型，列数决定特征工程的规模。`head(3)` 则让我们看到每一列的真实样子，比如 `room_type` 是字符串，`amenities` 是一长串 JSON。

接着我们看目标列 `price` 的分布：

```python
# describe() 会给出均值、标准差、最小最大和四分位数
train["price"].describe()

# 单独看偏度和峰度：偏度大于 0 说明分布向右拖了一条长尾
print("skew:", train["price"].skew())
print("kurtosis:", train["price"].kurt())
```

如果偏度明显大于 0，就说明有少量很贵的房源把右边拉长了。对这个数据来说，直接最小化平方误差会让模型花很多力气去拟合这些高价点，所以后面要么对价格取对数，要么用 RMSLE 这种自带对数性质的指标。

我们还会检查缺失值：

```python
# 统计每一列的缺失比例，按从高到低排序
missing = train.isna().mean().sort_values(ascending=False)
print(missing.head(20))
```

从结果里能看到，`bedrooms`、`beds`、`bathrooms_text` 这些字段都有缺失。`amenities` 虽然不是缺失，但它是一个 JSON 字符串，不能直接喂给线性回归。`geo_cell` 的编号是被打乱过的，单独看这个数字没有意义，必须结合经纬度或者做目标编码才有用。

最后我们看一下 `room_type` 的取值：

```python
# 类别特征先看有哪些取值、各有多少条
print(train["room_type"].value_counts(dropna=False))
```

这一步很关键。`room_type` 是字符串，线性回归不认字符串，必须做 one-hot 编码。如果某个取值在训练集里出现、在测试集里没出现，或者反过来，编码后的列还要对齐，否则预测时会报维度不匹配。

### 2.2 数据清洗

这份数据的清洗原则是：**只做有把握的处理，不为了「看起来干净」而丢掉信息。**

第一，缺失值用**训练集的中位数**填充。为什么用中位数而不是均值？因为 `beds`、`bedrooms` 这类字段有长尾，均值容易被极端值拉高，中位数更稳。更重要的是，中位数只能从训练集算，不能把测试集也算进去，否则会造成数据泄漏。

第二，`minimum_nights` 和 `number_of_reviews` 做对数变换。`minimum_nights` 可能有「最少住 1000 晚」这种极端值，我们先把它截断到 365 晚，再取 `log1p`；`number_of_reviews` 也是典型的长尾分布，取对数后更接近正态。

第三，`room_type` 做 one-hot 编码，并且用 `reindex` 把测试集的列对齐到训练集。

```python
# 只选 baseline 里确定要用的数值列
NUM = ["accommodates", "bedrooms", "beds", "minimum_nights", "number_of_reviews"]

# 用训练集的中位数做填充，测试集必须复用同一组中位数
medians = train[NUM].median()

def make_features(df, medians):
    # 先填充缺失值，保证后面不会出现 NaN
    X = df[NUM].fillna(medians).copy()

    # 最少入住晚数截断到一年，再取 log1p，压住长尾
    X["minimum_nights"] = np.log1p(X["minimum_nights"].clip(upper=365))

    # 评价数同样取对数，让「几百条评价」和「几条评价」的差距不要过分离散
    X["number_of_reviews"] = np.log1p(X["number_of_reviews"])

    # 类别变量转成 0/1 列，prefix 让列名带上来源，方便解释
    rt = pd.get_dummies(df["room_type"], prefix="rt", dtype=float)
    return pd.concat([X, rt], axis=1)

# 训练集和测试集各自生成特征
X = make_features(train, medians)
X_test = make_features(test, medians)

# 关键一步：测试集可能缺少某个 room_type，补齐成 0，列顺序也保持一致
X_test = X_test.reindex(columns=X.columns, fill_value=0)
```

这段代码里最容易被忽略的是最后一行 `reindex`。如果训练集有 4 种 `room_type`，而测试集只有 3 种，`get_dummies` 生成的列数就不一样，模型一预测就会报错。`reindex` 相当于拿训练集的「列清单」去校准测试集，缺的列补 0，多的列删掉。

我们**没有**在这一步处理 `amenities`、`bathrooms_text`、`name`、`description`。原因不是它们没用，而是 baseline 的目标是先跑通一条完整的流水线。材料里也明确说，这些字段是「下一步」可以挖掘的信息。先把简单特征做扎实，再逐步加复杂度，比一上来堆几十个特征更容易排查问题。

### 2.3 模型训练与超参数调优

我们其次建立线性回归模型，并用 5 折交叉验证评估。线性回归的好处是训练快、结果可解释：每个特征的系数直接告诉我们「这个特征变大时，价格往哪个方向走」。

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

# 线性回归没有超参数，先用它作为「地板」基准
model = LinearRegression()

# 5 折交叉验证：每次拿 4 折训练、1 折验证，轮流做 5 次
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# cross_val_score 默认认为分数越高越好，所以用负的 RMSE
scores = -cross_val_score(
    model, X, train["price"], cv=kf,
    scoring="neg_root_mean_squared_error"
)

print(f"CV RMSE: {scores.mean():.4f} ± {scores.std():.4f}")
```

这里有一个细节：`cross_val_score` 的 `scoring` 参数用的是 `neg_root_mean_squared_error`，也就是「负的 RMSE」。sklearn 的约定是「分数越大越好」，所以误差前面要加负号。我们最后再取一次负，把它变回正的 RMSE。

为什么用 K 折而不是简单切一刀？因为房源价格波动大，只切一次验证集，结果可能恰好碰上几套特别贵的房子，评分会很不稳定。5 折交叉验证让每一套房源都当过一次验证数据，平均值更能代表模型的真实水平。`shuffle=True` 则是先把数据打乱再分折，避免原始文件按某种顺序排列带来的偏差。

线性回归的另一个好处是我们可以直接看系数：

```python
# 用全部训练数据拟合，再看每个特征对价格的影响方向
model.fit(X, train["price"])
coef = pd.Series(model.coef_, index=X.columns).sort_values()
print(coef)
```

系数为正，说明这个特征越大价格越高；系数为负则相反。比如 `accommodates`（可住人数）通常是正的，因为大房子更贵；`minimum_nights` 的系数可能是负的，因为要求住得越久，单晚价格往往要给折扣。看系数不是为了做因果推断，而是检查「符号是否符合常识」——如果一个明显该为正的特征变成了负的，多半是特征处理或共线性出了问题。

材料里还提到，可以沿着课程进度把线性模型换成树模型。树模型（比如随机森林、LightGBM）能自动处理非线性和特征交互，通常比线性回归强。但我们在这篇里把线性回归作为主线，原因有三点：第一，它是理解 RMSLE 和特征工程的最好起点；第二，它的训练和调参成本极低，适合反复试特征；第三，只有先有一个可信的 baseline，后面换复杂模型时才知道提升到底来自哪里。

### 2.4 模型测试与指标报告

我们接着在交叉验证的基础上做最后的拟合和预测。注意，这里有一个容易犯的错误：**评估归评估，提交归提交。** 交叉验证的分数是用来判断模型好不好的，最终提交时应该用全部训练数据重新训练一次，把尽可能多的信息用上。

```python
# 用全部训练数据重新拟合，而不是只用交叉验证里的某一折
model.fit(X, train["price"])

# 对测试集做预测
pred = model.predict(X_test)

# 价格有物理下限，先截断到 [10, 2000] 英镑，避免出现负数或离谱高价
pred = pred.clip(10, 2000)

# 按比赛要求生成两列提交文件，价格保留两位小数
submission = pd.DataFrame({
    "id": test["id"],
    "price": pred.round(2),
})
submission.to_csv("submission.csv", index=False)
submission.head()
```

`clip(10, 2000)` 是一个业务约束。线性回归是无界的，遇到特别奇怪的测试样本，可能预测出负数或者几十万英镑。负数价格没有意义，极端高价也很可能是外推造成的错误。把它截断到一个合理区间，通常能让 RMSLE 稍微好看一点，而且不会伤害正常样本。

关于指标报告，我们需要把「交叉验证分数」和「排行榜分数」分开说。交叉验证用的是 RMSE，也就是在原始价格尺度上算误差；比赛用的是 RMSLE，在对数尺度上算误差。两者不完全等价，但方向一致：交叉验证差，排行榜大概率也差。材料里的 baseline 没有给出具体的排行榜分数，所以我们不在这里编造数字，只报告流程和评估方式。

我们还会检查提交文件的格式：

```python
# 确认行数和测试集一致，列名和顺序正确
print(submission.shape)
print(submission.columns.tolist())
print(submission["price"].isna().sum())  # 应该为 0

# 看看预测值的分布，确认没有全挤在一个常数上
print(submission["price"].describe())
```

最后这一步看起来琐碎，但比赛里因为提交格式错误白白丢分的情况并不少见。行数不对、列名拼错、`id` 顺序错乱，都会让一次好好的建模前功尽弃。

---

## 三、总结

本文中我们完成了伦敦 Airbnb 房价预测比赛的完整 baseline。我们首先用 EDA 看清了数据的规模和目标分布，确认价格有明显长尾；其次做了轻量但关键的数据清洗：用训练集中位数填缺失、对 `minimum_nights` 和 `number_of_reviews` 取对数、把 `room_type` 做 one-hot 并对齐测试集列；接着用 5 折交叉验证训练线性回归，并通过系数检查特征方向是否符合常识；最后用全量训练数据重新拟合，把预测截断到合理区间，生成 `id` + `price` 的提交文件。

这个比赛给我们最大的启发是：**baseline 的价值不在于分数高，而在于把整条链路跑通。** 先确定读数据、填缺失、编码类别、交叉验证、生成提交这几步都没有问题，再去加 `amenities`、地理信息、房东特征和树模型，才不会在调试时迷失方向。材料里列出的那些「还没用上的信息」，正是后面继续提升的路线图。
