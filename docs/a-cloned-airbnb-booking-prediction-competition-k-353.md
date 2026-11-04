# A Cloned Airbnb Booking Prediction Competition - K353

## 一、比赛简介

### 1.1 比赛背景

Airbnb 早期的数据科学团队有一个很有名的公开比赛：给定房源信息和一段时间的挂牌记录，预测新上线的房源**第一次被人预订需要等多久**，或者换个角度，预测某个房源在接下来一个季度里会被订出去多少天/多少次。这个赛道之所以有趣，是因为它把"推荐系统里点击率预测"和"短期租赁的供需预测"合在了一起：同一条房源，价格、位置、房型、房东资质、季节都会影响它被订走的概率。

这个比赛叫"A Cloned Airbnb Booking Prediction Competition - K353"，从名字看是一个**克隆版**——数据被重新组织过，比赛编号是 K353，用同样的思路让人练习。材料里只给了一份参赛 notebook，没有比赛规则页的原文，所以下面关于字段含义的说明，都是从文件名和代码里的列名推断出来的最小结论。

### 1.2 比赛目标

数据目录里有五个 CSV，可以分成三组：

| 文件 | 内容 | 用在哪 |
| --- | --- | --- |
| `property_info.csv` | 房源的**静态信息**：房型、国家等（代码里主键可能叫 `id`） | 训练和测试都要 |
| `listing_2016Q1.csv` / `listing_2016Q2.csv` | 2016 年第一、二季度的**每日挂牌记录**，含 `PropertyID` 和价格 `Price` | 训练和测试都要 |
| `reserve_2016Q3_train.csv` | 2016 年第三季度的**预订标签**（目标列是最后一列） | 只用于训练 |
| `PropertyID_test.csv` | 测试集的房源编号 | 只用于预测 |

所以任务是：**用 2016 年上半年的静态信息和每日价格，预测每条房源在 2016 年第三季度的预订情况。** 提交文件的列是 `PropertyID_test` 和 `Pred`。

材料里有一句很关键的注释：比赛要求预测值必须是整数，不能有小数（it requires integer predictions）。这间接说明目标列是"次数"一类的计数，比如被预订了几次，而不是金额或时长。

### 1.3 评价标准

材料没有写排行榜的指标公式，但代码里有两条证据：验证阶段用的是**均方误差**（`mean_squared_error`），模型的优化目标写的是 `reg:squarederror` 加 `eval_metric='rmse'`。由此可以合理判断这是一个**基于平方误差的回归问题**，指标是越低越好。具体的榜单公式我们不臆测，但优化方向是清楚的：**把预测值和真实预订数的平方差压小。**

顺带说一句为什么用平方误差：预订数是计数值，误差了一两次的代价应该比误差一次之内更大，平方误差正好会把大错放得更大，逼着模型别在个别房源上错得离谱。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**把"每天的价格记录"压成几条到房源级别的统计量，和静态信息拼在一起，喂给梯度提升树。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步是把五个文件都读进来，看清各有多少行、各有哪些列：

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

base_path = "/kaggle/input/a-cloned-airbnb-booking-prediction-competition-k-353/"

# 三张"房源级"表：静态信息、训练标签、测试编号
prop_info = pd.read_csv(base_path + "property_info.csv")
train_target = pd.read_csv(base_path + "reserve_2016Q3_train.csv")
test_ids = pd.read_csv(base_path + "PropertyID_test.csv")

# 两张"每日级"表：价格随日期的流水账
q1 = pd.read_csv(base_path + "listing_2016Q1.csv")
q2 = pd.read_csv(base_path + "listing_2016Q2.csv")

# 打印行数：标签表有多少行，就说明有多少条房源有监督信号
print(prop_info.shape[0], train_target.shape[0], test_ids.shape[0])
print(train_target.columns.tolist())
```

打印出来的信息量很大。第一行告诉我们**三张表的房源数量不一样**——这意味着后面合并时一定会出现"有的房源没有价格记录""有的价格记录没有标签"的情况，必须用左连接（`how="left"`）而不是内连接，否则会把要预测的房源丢掉。

`train_target.columns.tolist()` 则告诉我们目标列叫什么。材料里的做法是"假设目标列是最后一列"，写成 `train_target.columns[-1]`。这个假设在本书的例子里成立，但它很脆弱——**字段顺序变了，代码就悄悄预测了别的列**。我们在复现时保留了这种做法，同时在打印里把选中的列名显式输出一遍，出问题能第一时间发现。

第二步是看价格列的类型：

```python
# 合并两个季度的每日记录，拿到更长的时间跨度
q1_q2 = pd.concat([q1, q2], ignore_index=True)

# 如果 Price 是字符串（含 $ 或千分位逗号），describe 会算不出统计量
print(q1_q2["Price"].dtype)
print(q1_q2[["PropertyID", "Price"]].head())
```

原始 notebook 在这里特意判断了一次 `dtype == 'object'`，说明作者遇到过价格是字符串的情况。这是一个很现实的坑：很多平台导出的 CSV 会把价格写成 `$1,200.00`，pandas 读进来就是字符串，后面求平均直接报错或者变成字符串拼接。

第三步是看目标值的分布和缺失情况：

```python
# 目标列的基本统计：均值、标准差、最小值、最大值
print(train_target.iloc[:, -1].describe())
# 静态信息里有没有缺失
print(prop_info.isna().sum().to_string())
```

这一步的目的是判断"要不要做对数变换"。如果目标值的最大值远大于均值（长尾），直接用平方误差会让模型只顾着拟合那几个大房源。不过材料里的 notebook 没有做这个变换，我们也就保持原样，只在下面提一句。

### 2.2 数据清洗

清洗主要干了三件事，都很朴素但都不能省。

**第一，清理价格字符串。** 代码里对 `Price` 做了"如果是字符串就删掉 `$` 和逗号再转 float"的处理：

```python
q1_q2 = pd.concat([q1, q2], ignore_index=True)

# 常见写法 "$1,200.00"：去掉货币符号和千分位逗号，才能转成数字
if q1_q2["Price"].dtype == "object":
    q1_q2["Price"] = (
        q1_q2["Price"]
        .replace({r"\$": "", ",": ""}, regex=True)   # 正则替换，一次清掉两种字符
        .astype(float)                                # 转成浮点数才能求均值
    )
```

这里用 `replace` 加正则而不是 `str.strip("$")`，是因为逗号可能出现在数字中间，`strip` 只能去两头的字符，去不掉中间的。

**第二，把每日流水压成房源级统计。** 一条房源在半年里可能有上百行记录，不能直接拿去训练（行数对不上）。我们按 `PropertyID` 分组，算三个数：

```python
daily_agg = q1_q2.groupby("PropertyID").agg(
    avg_price=("Price", "mean"),          # 半年平均挂牌价：代表房源定位
    max_price=("Price", "max"),           # 最高价：能反映旺季溢价能力
    days_listed=("PropertyID", "count"),  # 出现天数：挂牌多久，间接反映热度
).reset_index()
```

这三个统计量各有分工：`avg_price` 刻画房源档次，`max_price` 抓住"敢不敢在旺季抬价"，`days_listed` 则是一个行为信号——一条房源如果在半年里被反复挂牌，说明它一直在市场上找客人，这本身就与预订情况相关。

**第三，统一主键。** 静态信息表的主键可能叫 `id`，其他表叫 `PropertyID`，必须统一：

```python
# 主键名字不一致就没法 merge，先改名
if "id" in prop_info.columns:
    prop_info = prop_info.rename(columns={"id": "PropertyID"})

# 以静态信息为主表左连接，保证每条房源都保留下来
full_features = prop_info.merge(daily_agg, on="PropertyID", how="left")
```

`how="left"` 是这里的关键。如果某条房源没有价格记录，左连接会留下它、价格列填 `NaN`；换成内连接，这条房源就直接消失了——而它很可能正好是要预测的测试房源。缺失值后面交给 XGBoost 自己处理，它天生支持缺失值分裂。

### 2.3 模型训练与超参数调优

**类别编码。** 静态信息里有房型、国家这类文本列，树模型虽然理论上能处理，但 XGBoost 的默认配置要求输入是数字。材料里的做法是用 `LabelEncoder` 把每个文本列映射成整数：

```python
# 找出所有文本列（object 类型），逐列编码成 0,1,2,...
categorical_cols = full_features.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in categorical_cols:
    # astype(str) 兜底：万一有 NaN，先变成字符串 "__nan__" 再编码，不会报错
    full_features[col] = le.fit_transform(full_features[col].astype(str))
```

这里要诚实指出一个隐患：**编码器是在"训练 + 测试"的合并表上拟合的**。好处是训练和测试的类别编号完全一致（如果分开拟合，同一个"公寓"在两边可能变成不同的数字，模型会对不上）；坏处是编码器"见过"测试集，严格来说是一种轻度的信息泄漏。对于 LabelEncoder 这种只做映射、不学习分布的操作，实际影响很小，但在正式比赛里更稳妥的写法是：先用训练集拟合，遇到没见过的类别统一归到一个"未知"编号。

**拼装训练集和测试集。** 这一步要和标签表、测试编号表分别连接：

```python
train_df = train_target.merge(full_features, on="PropertyID", how="left")
test_df  = test_ids.merge(full_features, on="PropertyID", how="left")

target_col = train_target.columns[-1]      # 惯例：目标列是最后一列
X = train_df.drop(columns=["PropertyID", target_col])   # 主键是标识，不是特征
y = train_df[target_col]
X_test = test_df.drop(columns=["PropertyID"])
```

注意 `PropertyID` 被明确丢掉了。这一点很重要：房源编号是随机分配的标识符，把它当特征喂进树模型，模型会记住"编号 3021 的房源订得多"，到测试集上完全失效。**凡是"唯一标识"，都不该进特征。**

**划分验证集并训练。** 我们用 80/20 的随机划分：

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,             # 树的上限；靠 early stopping 决定实际用多少棵
    learning_rate=0.05,            # 每棵树只走一小步，配合多棵树更稳
    max_depth=6,                   # 每棵树最多 6 层，控制模型复杂度
    objective="reg:squarederror",  # 平方误差，和评价口径一致
    eval_metric="rmse",            # 监控 RMSE，数值上和 MSE 同向，但单位可读
    early_stopping_rounds=20,      # 连续 20 轮验证集没变好就停
    random_state=42,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],     # 让模型边训边看验证集
    verbose=50,                    # 每 50 轮打一次日志，方便观察是否已经收敛
)

val_preds = xgb_model.predict(X_val)
val_mse = mean_squared_error(y_val, val_preds)
print(f"Validation MSE: {val_mse:.4f}")
```

**关于超参数。** 这组参数是一个很标准的"够用就好"配置，作者没有做网格搜索。我们认同这个取舍：样本量不大、特征只有十来个，`max_depth=6` 已经能抓住大部分交互，再深就是过拟合。真正决定上限的是**特征**——比如"房源所在城市的季节系数""同类房源的均价"这些，而不是 `learning_rate` 的小数点后第三位。

`early_stopping_rounds=20` 是这里唯一"自动调参"的机制：训练 1000 棵树是上限，验证集连续 20 轮不进步就提前收工。它省掉了"到底该训多少棵树"这个超参数。

### 2.4 模型测试与指标报告

验证集上的 MSE 由上面那行 `print` 输出。材料原文里这个数字是运行时才填进 f-string 的，没有固定值，我们在这里也不编一个——**MSE 的具体数值取决于数据版本和运行环境，重要的是它在多次实验之间可以互相比较**。

预测阶段有两处必须处理：

```python
# 用训练好的模型对测试房源预测
test_preds = xgb_model.predict(X_test)

# 比赛要求整数，且预订数不可能是负数：
# 先四舍五入到整数，再把负数截到 0，最后转成 int 类型
test_preds_rounded = np.clip(np.round(test_preds), 0, None).astype(int)

submission = pd.DataFrame({
    "PropertyID_test": test_df["PropertyID"],   # 列名必须和样例完全一致
    "Pred": test_preds_rounded,
})
submission.to_csv("submission.csv", index=False)
print("submission.csv successfully created!")
```

**为什么要 `round` 再 `clip`？** 回归模型输出的是连续小数，比如 2.37 次预订。比赛要求整数，四舍五入是最自然的取整方式。但四舍五入只解决"小数"问题，不解决"负数"问题——模型完全可能对一条冷门房源吐出 -0.4。所以还要 `np.clip(..., 0, None)` 把下界压到 0。这两个操作合起来，保证了提交文件在格式上绝对不会被判错。

**提交后我们还会做三项检查**，这也是我们在这个比赛里养成的习惯：

1. **行数**：提交文件的行数必须和 `PropertyID_test.csv` 完全一致，不能多也不能少。
2. **主键**：逐行比对 `PropertyID_test` 列和样例文件，确认顺序没错位。Kaggle 按行号对齐，顺序错了分数会莫名其妙地低。
3. **取值范围**：确认没有负数、没有 NaN、全是整数。这三条任何一条不满足，提交就会失败或者被算成 0 分。

**这个方案的天花板在哪里？** 我们把半年的价格压成了三个数字，信息损失是必然的。比如"价格从三月到六月一路上涨"和"价格一直很平"最后可能算出同样的均值。如果要继续提升，方向很清楚：一是把时间序列本身的形状保留下来（比如按月取均值，得到 6 个点）；二是引入同类房源的横向比较（同城同房型的均价），这类"相对位置"特征往往比绝对值更有用；三是把目标做对数变换或分桶，缓解长尾。

---

## 三、总结

本文中我们完成了 K353 克隆版 Airbnb 预订预测比赛的一条完整基线。我们首先把五个文件读进来，确认了三张房源级表和两张每日级表的行数与列名，并注意到表之间房源数量并不相等。其次我们清理了价格列里可能存在的货币符号和千分位逗号，把半年的每日挂牌记录按房源聚合成平均价、最高价、挂牌天数三个统计量，用左连接挂在静态信息上，并用 `LabelEncoder` 把文本列编码成数字。接着我们把标签表和测试编号表分别与特征表合并、丢掉房源编号，用 80/20 划分验证集，训练了一个 1000 棵树上限、学习率 0.05、深度 6 的 XGBoost 回归模型，靠 early stopping 决定实际树数。最后我们把预测值四舍五入、下界截到 0、转成整数，按样例的列名写出提交文件，并检查了行数、主键顺序和取值范围。

这次练习让我们看清了两件事：**聚合是有代价的**，把时间序列压成三个数省事，但也把"变化的过程"丢掉了；以及**格式检查不是形式主义**，四舍五入和截断这两步如果漏了，再好的模型也拿不到分。
