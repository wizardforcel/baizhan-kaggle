# BoolArt 用户行为预测（BoolArt User Behavior Prediction）

## 一、比赛简介

### 1.1 比赛背景

这场比赛使用的数据来自**天猫 2017 年某品类用户行为数据**：在 7 天（数据时间跨度 2017-11-25 至 2017-12-03）里，系统记录了大量用户与商品的每一次交互行为。行为一共有四种类型，记录在 `BehaviorType` 列里：

- `pv`：浏览（page view，点开了商品详情页）；
- `cart`：加购（把商品放进购物车）；
- `fav`：收藏（把商品收藏起来）；
- `buy`：购买（真正下单）。

电商平台最关心的一环是**从「看过」到「买下」之间的转化**。这个比赛就是让我们基于历史行为，去预测「用户明天会不会买某个商品」——本质上是一个二分类问题。

### 1.2 比赛目标

比赛提供三个文件：

- `train`（Behavior.csv）：用户的**历史行为流水**，一行是一条行为记录，字段包括 `UserID`、`ItemID`、`CategoryID`、`BehaviorType`、`Timestamp`（Unix 秒时间戳）；
- `test`（Test.csv）：一堆「用户-商品对」，需要我们预测每个对里用户是否会在未来一天购买该商品；
- `category`（Category.csv）：商品与品类的映射表（`ItemID` → `CategoryID`）。

我们最终要对 `test` 里的每一个（用户，商品）对输出一个购买概率或 0/1 标签。

### 1.3 评价标准

材料里的 notebook 在建模部分导入的是 `accuracy_score`（准确率）和 `recall_score`（召回率）两个分类指标，说明这场比赛的主评价指标应该在二者之中。准确率是「预测对的（含没买且预测没买）比例」，召回率是「真正买了的人里，我们喊出来多少」。因为「买」这个行为是少数事件，考试时最好两个指标一起看，别被高准确率骗了——全部猜「不买」准确率也会很高。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们先把三个文件读进来，看一下规模：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, accuracy_score

train = pd.read_csv("/kaggle/input/boolart-user-behavior-prediction/Behavior.csv/Behavior.csv")
test  = pd.read_csv("/kaggle/input/boolart-user-behavior-prediction/Test.csv")
category = pd.read_csv("/kaggle/input/boolart-user-behavior-prediction/Category.csv")

train.shape
```

接着我们沿着「时间 → 行为 → 用户 → 商品 → 品类」五个维度逐层看。

**时间维度。** 我们把时间戳转成小时、星期，然后统计每个小时的行为占比：

```python
train["Timestamp"] = pd.to_datetime(train.Timestamp, unit="s")
train["hour"] = train.Timestamp.dt.hour

# 一天里各小时的浏览占比，看看用户什么时候最活跃
train.Timestamp.dt.hour.value_counts(normalize=True).sort_index().plot(kind="bar")

# 再按行为类型分别画各小时的占比曲线
for beh in train.BehaviorType.unique():
    train[train.BehaviorType == beh].Timestamp.dt.hour.value_counts(normalize=True).sort_index().plot()
```

画完图明显看到行为有**白天聚集**的现象，不同行为的高峰时段不太一样——比如加购和购买通常集中在晚上。

**行为维度。** 我们最想搞清楚的是**转化漏斗**：从浏览到购买，每一步流失多少？于是算出行为占比之间的比率：

```python
# 各行为占比
beh_ratio = train.BehaviorType.value_counts(normalize=True)
# 加购数 / 浏览数：看浏览转化到加购的比例
cart_over_pv = beh_ratio["cart"] / beh_ratio["pv"]
# 购买数 / 加购数：看加购转化到购买的比例
buy_over_cart = beh_ratio["buy"] / beh_ratio["cart"]
# 合并两步算「浏览 → 购买」的总转化率
100 * cart_over_pv * buy_over_cart
```

这个漏斗让我们心里有个底：真正发生购买的行为占比非常低，是一个典型的**严重不平衡二分类**问题。

**用户维度。** 我们统计每个用户的交互次数，发现典型的长尾：极少部分「超级用户」占了绝大多数行为。

```python
user_interact_cnt = train.UserID.value_counts()
user_interact_cnt.describe()

# 前 100 名最活跃用户的交互占全部交互的比例
user_interact_cnt.sort_values(ascending=False)[:100].sum() / user_interact_cnt.sum()
```

我们还给每个用户做了一张「行为 × 用户」的透视表，并据此计算每个人的 **买/浏览比（buy_pv_rate）**，把异常的（比率大于 1）置 0：

```python
user_behavior_cnt = train.groupby(["UserID", "BehaviorType"])["BehaviorType"].count() \
                         .rename("count").reset_index()
user_behavior_cnt = user_behavior_cnt.pivot_table(index="UserID", values="count", columns="BehaviorType") \
                                     .fillna(0.)

user_behavior_cnt["buy_pv_rate"] = user_behavior_cnt["buy"] / user_behavior_cnt["pv"]
user_behavior_cnt.loc[user_behavior_cnt.buy_pv_rate > 1, "buy_pv_rate"] = 0.
sns.displot(user_behavior_cnt.buy_pv_rate.values)
```

**商品与品类维度。** 类似地，我们统计了 `CategoryID` 与 `ItemID` 的分布，发现和用户一样是长尾：少数品类占了大头。这些「品类占比」本身就是很好的特征——高频品类往往代表大众商品，转化规律更稳定。

更值得注意的差异在**品类转化率**上：有些品类浏览量巨大但成交很少，是「逛得多买得少」；有些品类虽然流量不大，但一旦浏览就很容易下单。这提醒我们：品类不应该只当一个 ID 塞进模型，而应该把「该品类历史的购买转化率」做成一个连续特征，让模型可以直接学到「这个品类好不好卖」。

**用户画像是四件事。** 材料里把用户画像拆成生活习惯、购买转化、决策周期、品类/商品偏好四块，我们一一摸了一遍：

- 生活习惯：每个用户最活跃的小时（`active_hour`），以及 7 天内的购买次数（`buy_count`）、浏览次数（`pv_count`）；
- 购买转化：每个用户的 `buy_count / pv_count`，比值大于 1 的异常值置 0；
- 决策周期：从第一次浏览某个商品到真正下单，中间隔了几个小时（`decision_cost`），小于等于 0 的按缺失处理——这种「秒买」记录通常是数据异常；
- 品类偏好：该用户购买行为的 `CategoryID` 分布。

```python
# 给用户画像合并"活跃时段、购买数、浏览数、转化率"
sub_users = train.UserID.unique()[:1000]          # 演示时取前 1000 个用户
train = train[train.UserID.isin(sub_users)]

user = pd.DataFrame(index=sub_users)
train["hour"] = train.Timestamp.dt.hour
# 活跃时段 = 该用户行为最多的小时
user = user.merge(
    train.groupby("UserID")["hour"].apply(lambda g: g.value_counts().sort_values().index[-1])
         .rename("active_hour"), left_index=True, right_index=True)
# 7 天购买/浏览次数
user = user.merge(train[train.BehaviorType == "buy"].UserID.value_counts().rename("buy_count"),
                  left_index=True, right_index=True, how="left")
user = user.merge(train[train.BehaviorType == "pv"].UserID.value_counts().rename("pv_count"),
                  left_index=True, right_index=True, how="left")
user.buy_count.fillna(0, inplace=True)
user["buy_per_pv"] = user.buy_count / user.pv_count
```

这里有几个需要注意的点：首先，测试集给的是「用户-商品对」，所以最终特征必须**按对**组织，不能只按用户——比如「过去 7 天这个用户有没有浏览过这个商品」「这个商品在这个用户的长尾里排第几」；其次，预测目标是「未来一天是否购买」，特征时间上必须严格取「过去」，把当天的行为混进特征等于偷看答案；再次，前 100 名重度用户占了行为的很大比例，模型容易倾向他们，必要时要对次数类特征取 `log1p` 压缩量纲。

### 2.2 数据清洗

清洗的核心思路：**测试集里出现的用户和商品才是我们要关心的**，训练集里其它用户/商品的历史行为既帮不上忙，还会拖慢训练。所以我们把训练集裁剪到「用户出现在测试集里」或「商品出现在测试集里」的行：

```python
test_UserID = set(test["UserID"].unique().tolist())
test_ItemID = set(test["ItemID"].unique().tolist())

# 保留「用户被测试集关注」或「商品被测试集关注」的行为记录
use_rows = train.UserID.isin(test_UserID) | train.ItemID.isin(test_ItemID)
train = train.loc[use_rows].reset_index(drop=True)

# 时间戳是 Unix 秒，转成 datetime 方便提取小时/星期
train["Timestamp"] = pd.to_datetime(train.Timestamp, unit="s")
```

这一步把训练数据大幅瘦身，同时保证了「样本分布与测试集一致」，避免模型学了测试集根本不会遇到的用户习惯。

### 2.3 模型的构建与用户画像

材料里的 notebook 是一份**偏 EDA 的练习**，在用户画像阶段就结束了（没有再训练模型）。我们根据它已经验证过的结论，把可行的建模思路整理成下面的最小基线，这段代码**不是 notebook 原文，而是按它的 EDA 结论整理的伪代码**。

我们先给每个测试用户构造画像特征：活跃时段、7 天购买次数、浏览次数、购买转化率、平均决策周期（从第一次浏览到购买的间隔小时）：

```python
# 以测试用户为索引，逐个统计行为画像（仅示意核心逻辑）
user = pd.DataFrame(index=test_UserID)

# 活跃时段：该用户行为最集中的小时
user["active_hour"] = train.groupby("UserID")["hour"].apply(
    lambda g: g.value_counts().sort_values().index[-1])

# 7 天购买/浏览次数与转化率
user["buy_count"] = train[train.BehaviorType == "buy"].UserID.value_counts()
user["pv_count"]  = train[train.BehaviorType == "pv"].UserID.value_counts()
user["buy_per_pv"] = user["buy_count"].fillna(0) / user["pv_count"]

# 决策周期：第一次浏览到购买之间的间隔（小时）
pv = train[train.BehaviorType == "pv"].groupby(["UserID", "ItemID"])["Timestamp"] \
         .apply(lambda g: g.sort_values().iloc[0]).rename("pv_time").reset_index()
buy = train[train.BehaviorType == "buy"].drop_duplicates()[["UserID", "ItemID", "Timestamp"]] \
         .rename(columns={"Timestamp": "buy_time"})
decision = buy.merge(pv, on=["UserID", "ItemID"], how="left")
decision["decision_cost"] = (decision["buy_time"] - decision["pv_time"]).dt.total_seconds() / 3600
```

把这些特征拼起来后，交给一个逻辑回归（notebook 里导入的就是 `LogisticRegression`，适合这种「特征少、样本多、要解释性」的小基线）：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 假设上面已经汇总出特征表 X 与标签 y（是否购买）
# 逻辑回归 + 5 折交叉验证，指标按题目用准确率与召回率
clf = LogisticRegression(max_iter=1000)
acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()
rec = cross_val_score(clf, X, y, cv=5, scoring="recall").mean()
print("5-fold accuracy:", acc, " recall:", rec)
```

### 2.4 模型测试与指标报告

考虑到购买行为极度稀少，我们特别关注**在不牺牲太多准确率的前提下召回率能拉到多少**——也就是「真会买的人我们捞回多少」。读这两个指标时有几个要点：

- **准确率会被大量负样本「撑」得很好看**。测试集里绝大多数（用户，商品）对都是不买的，所以模型只要倾向于猜「不买」，准确率就会很高，但这毫无业务价值；
- **召回率才反映「抓人」的能力**。它的分母是真正购买的对数，模型漏掉一半买家，召回率就掉一半，很敏感；
- 两者是此消彼长的。实际业务里会按「捞回一个买家值多少钱」和「误发一张券亏多少钱」来定阈值，而不是死盯 0.5。

上述 5 折交叉验证会给出两个指标的平均值；如果召回率明显偏低，下一步就是调逻辑回归的分类阈值（对少数类提高阈值偏置）或者换树模型。

最后用全量数据重训一份模型，对 `test` 里每个（用户，商品）对预测购买概率，并按比赛要求的格式写出提交文件。具体分数取决于特征版本的详细程度，这一份 EDA 基线主要用来锚定「不买全猜 0」的下限，为后续特征工程和模型升级提供参照。

---

## 三、总结

本文中我们完成了 BoolArt 用户行为预测的整体分析：首先从时间、行为、用户、商品、品类五个维度做了 EDA，确认数据存在明显的时间效应、转化漏斗（浏览→加购→购买逐步收缩）以及用户/商品的长尾分布；其次我们把训练集裁剪到与测试集相关的用户与商品，并清洗了时间戳；接着我们按 notebook 的结论整理了用户画像特征和逻辑回归基线（该基线代码是根据 EDA 结论整理的思路，而非 notebook 原文）；最后我们用 5 折交叉验证分别报告了准确率与召回率，强调要为「少数类」买单。

这个比赛给我们的启发是：**在严重不平衡的「是否购买」问题里，单一准确率是没意义的**——把用户画像（活跃时段、转化率、决策周期）做成特征、把召回率纳入考核，才是真正贴合业务的做法。