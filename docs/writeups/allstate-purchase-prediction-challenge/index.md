# Allstate 购车险方案预测（Purchase Prediction Challenge）

## 一、比赛简介

### 1.1 比赛背景

Allstate 是美国一家大型保险公司。客户买车险的时候，不是一上来就签合同，而是先在网站上反复比较：改一改保额、换一换免赔额、看看不同组合要多少钱，来回看好几轮，最后才确定买哪一套。这中间每一次浏览，网站都会留下一条记录——这就是我们的数据来源。

比赛想解决的问题是：**根据一个客户在网站上逛的这几轮记录，猜出他最后会买哪一套保险方案。** 这件事对保险公司很有价值：如果能在客户还没决定的时候就猜出他想要什么，就可以提前准备报价、优化推荐，甚至在客户犹豫的时候推一把。

这个比赛的特别之处在于**数据是按「客户-浏览轮次」组织的**：一个客户会有好几行，每一行代表他的一次浏览（叫一个 `shopping_pt`，可以理解成「第几次进店看」）。我们要做的不是给每一行贴标签，而是**把同一个客户的所有浏览记录合起来看，最后给出一个答案**。这跟常见的「一行一个样本」的表格题很不一样，也是它的主要难点。

### 1.2 比赛目标

给定一个客户在购买前的全部浏览记录，预测他最终选择的保险方案。

方案由 7 个选项 `A`、`B`、`C`、`D`、`E`、`F`、`G` 组成，每个选项有几个离散取值（例如 0、1、2、3）。比赛把这些选项拼成一个 **7 位数字串** 作为答案，例如 `1111111` 就是最普通的那套方案。

数据的规模是：

- 训练集 `train.csv`：**665249 行、25 列**；
- 测试集 `test_v2.csv`：**198856 行、25 列**；
- 提交模板 `sampleSubmission.csv`：**55716 行、2 列**（`customer_ID` + `plan`）。

注意一个关键的数量关系：训练集有 665249 行，但提交文件只有 **55716 个客户**。这再次说明**一个客户对应多行浏览记录**，提交时每个客户只给一行答案。

25 列大致可以分成几类：

- 标识类：`customer_ID`（客户编号）、`shopping_pt`（第几次浏览）、`record_type`（记录类型，0 或 1）；
- 时间类：`day`（星期几）、`time`（浏览时刻，形如 `08:35`）；
- 客户属性：`state`（所在州）、`location`（地区编码）、`group_size`（家庭人数）、`homeowner`（是否有房）、`car_age`（车龄）、`car_value`（车辆价值等级，字母 a~i）、`risk_factor`（风险等级）、`age_oldest` / `age_youngest`（家中最年长/最年轻成员年龄）、`married_couple`（是否已婚）；
- 历史类：`C_previous`（上次买的方案）、`duration_previous`（上次投保时长）；
- 方案类：`A`～`G` 七个选项；
- `cost`：这次浏览看到的报价。

### 1.3 评价标准

提交的每一行是一个客户的 7 位方案串，评分方式就是**把这个 7 位串和真实方案一位一位地对**，数猜对了多少位，再把所有客户、所有位的正确率平均起来。**分数越高越好，满分 1.0。**

> 说明：材料里没有直接写评分公式，但从提交文件「每个客户一行、每行一个 7 位数字串」的格式可以推断，评价一定是逐位比对；比赛中大家也是按「逐位准确率」来理解分数的。这一点我们照实说明，不额外编造具体数值。

这个指标有一个很重要的含义：**7 位里猜对 6 位和猜对 3 位是天壤之别**，而且每一位的难度不一样——有些选项几乎所有客户都选同一个值（猜对很容易），有些选项则因人而异。所以后面做模型时，**分选项建模**往往比一次性预测整个 7 位串更靠谱。

---

## 二、解题过程

这道题在材料里留下的代码主要是**数据探索和一次 LightGBM 练习**，没有完整的获奖方案。所以我们先讲清楚数据长什么样、怎么清洗，再根据材料里的建模练习整理出训练思路，并明确标注哪部分是「根据讨论整理的伪代码」。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，确认行列数、类型和缺失：

```python
import pandas as pd
import numpy as np

# 训练集、测试集、提交模板
train = pd.read_csv('../input/train.csv.zip')
test  = pd.read_csv('../input/test_v2.csv.zip')
sub   = pd.read_csv('../input/sampleSubmission.csv')

# 规模：训练 665249 行，测试 198856 行，提交只要 55716 个客户
print(train.shape, test.shape, sub.shape)   # (665249, 25) (198856, 25) (55716, 2)

# 逐列看类型和缺失数量
train.info()
```

`train.info()` 的输出把缺失情况讲得很清楚：`car_value` 只有 663718 个非空（缺 1531 个）、`risk_factor` 只有 424831 个非空（缺 240418 个，接近四成）、`C_previous` 和 `duration_previous` 各缺 18711 个。其余列都是完整的。`risk_factor` 缺这么多，说明它本来就不是每个人都会被评估。

接着我们看了 `record_type` 的分布，这一列是理解数据结构的关键：

```python
# record_type 只有两个取值；看一下哪种多
print(train['record_type'].value_counts())
# 训练集里 record_type 的均值约 0.1458
print(train['record_type'].mean())
```

训练集里大约 **14.6%** 的行 `record_type` 是 1，其余是 0。结合「一个客户有多行、最后只有一行答案」的结构，可以这样理解：**`record_type = 0` 是浏览过程中的记录，`record_type = 1` 是最终成交的那一条**。这也解释了为什么测试集也有 `record_type` 列——它的最后一行同样是成交记录，只是方案还没揭晓。

然后我们统计了两个目标相关的列，确认方案在客户之间确实有差异：

```python
# 车辆价值一共 9 个等级，e、f 最多
print(train.car_value.nunique())          # 9
print(train.car_value.value_counts())
# e    219251
# f    177204
# d    113174
# g     98152
# ...

# 上一次买的方案分布
print(train.C_previous.value_counts())
# 3.0    271160
# 1.0    172007
# 2.0    109184
# 4.0     94187
```

这里有个很有用的发现：**`C_previous` 的取值分布和目标方案的取值分布高度重合**——很多人这次买的方案跟上次一样。这说明「历史」是很强的信号，不能丢。

### 2.2 数据清洗

清洗的第一步是**补缺失**。由于同一个客户的浏览记录是按顺序排下来的，客户属性在相邻行之间几乎不变，所以最自然的补法是**向前填充（ffill）**：用上一行的值补当前行的空。比用全局平均值补更贴近真实情况：

```python
# 按客户分组，组内按浏览顺序，缺失值用上一行向下填
# 这里数据已经按客户和时间排好，直接 ffill 即可
train.risk_factor        = train.risk_factor.fillna(method='ffill')
train.car_value          = train.car_value.fillna(method='ffill')
train.C_previous         = train.C_previous.fillna(method='ffill')
train.duration_previous  = train.duration_previous.fillna(method='ffill')

# 测试集做同样的处理，保证两边规则一致
test.risk_factor       = test.risk_factor.fillna(method='ffill')
test.car_value         = test.car_value.fillna(method='ffill')
test.C_previous        = test.C_previous.fillna(method='ffill')
test.duration_previous = test.duration_previous.fillna(method='ffill')
```

> 小提示：`ffill` 只能补「中间缺的」，如果某个客户第一行就缺，前面没有值可借，还是空着。所以实际做的时候我们会在 `ffill` 之后再补一次 `bfill`（向后填），或者对开头的空缺单独填众数。

**第二步是把目标编码成 7 位数字。** 七个选项 `A`～`G` 每个都是一位数，我们按「A 是百万位、G 是个位」把它们拼成一个整数，和提交文件的格式对齐：

```python
# 把 7 个选项拼成一个 7 位数：
# A 在最高位（百万），B 在十万位，……，G 在个位
train['plan'] = (train.G + train.F * 10 + train.E * 100
                 + train.D * 1000 + train.C * 10000
                 + train.B * 100000 + train.A * 1000000)

# 测试集同样拼一遍，但只用来对照格式，训练时不能把它当特征
test['plan'] = (test.G + test.F * 10 + test.E * 100
                + test.D * 1000 + test.C * 10000
                + test.B * 100000 + test.A * 1000000)
plan_test = test['plan']
test = test.drop(['plan'], axis=1)   # 测试集没有真实 plan，先删掉
```

这里要提醒一个坑：拼成整数之后，像 `0111111` 这样的方案会丢掉开头的 0，变成 6 位数。所以**提交时必须再格式化回 7 位字符串**（后面会写 `zfill(7)`），否则格式对不上，分数直接作废。

**第三步是时间特征和文本编码。** `time` 是 `08:35` 这样的字符串，模型不能直接用，我们把它拆成「小时」和「分钟」两个数字；`state` 和 `car_value` 是文本，用 `LabelEncoder` 转成整数：

```python
# 把 '08:35' 拆成 hour=8、minute=35 两个数值特征
train['hour']   = pd.to_datetime(train.time).dt.hour
train['minute'] = pd.to_datetime(train.time).dt.minute
test['hour']    = pd.to_datetime(test.time).dt.hour
test['minute']  = pd.to_datetime(test.time).dt.minute

# 文本列转整数；树模型只关心"是否等于某个值"，所以这样编码是安全的
from sklearn.preprocessing import LabelEncoder
columns_to_Encode = ['state', 'car_value']
le = LabelEncoder()
for each in columns_to_Encode:
    train[each] = le.fit_transform(train[each])
for column in columns_to_Encode:
    test[column] = le.fit_transform(test[column])
```

最后我们用热力图（heatmap）看了一眼各列和目标的相关性：

```python
import seaborn as sns
from matplotlib import pyplot as plt

# 只挑出和方案/时间相关的列看相关性
df3 = train[['cost', 'plan', 'hour', 'minute', 'A', 'B', 'C', 'D', 'E', 'F', 'G']]
sns.heatmap(df3.corr(), annot=True, fmt=".1f", linewidths=.5)
plt.show()
```

图里能看到，`cost`（报价）和方案里的几项有明显的正负相关——这很合理，保额越高报价越贵。而材料里的结论是：**`record_type`、`day`、`state`、`location`、`minute`、`married_couple` 这几列对目标的影响很小**。这是一个有用的筛选依据：特征不是越多越好，弱特征带进来只会添噪声（当然，像 `state` 这种在别的模型里可能有用，所以我们不强行删，只是心里有数）。

### 2.3 模型训练与超参数调优

**先说明材料的情况。** 材料里的两份练习笔记都做了同一件事：把 `record_type` 当成标签训练 LightGBM 分类器。但 `record_type` 只是「这是浏览行还是成交行」的标记，**不是我们要预测的方案**，所以那个练习的 AUC（约 0.853）只说明「模型能区分浏览和成交」，并不代表比赛成绩。我们在下面把它纠回正题：**真正的标签是那个 7 位方案**。下面这段代码是根据材料里的处理流程和建模练习整理的思路（伪代码性质），我们如实标注。

**思路一：把问题变成「逐位分类」。** 与其一次性预测 7 位数（组合太多），不如给每一位 `A`～`G` 各训一个分类器，最后把 7 个预测拼起来。这样每个模型只需要在几个类别里选，简单得多，也正好对上「逐位准确率」的评分方式。材料里的练习已经验证了 LightGBM 在这种表格数据上又快又稳，我们沿用它：

```python
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

# 统一走一遍材料里的特征处理流程
train_cl = train.copy()
train_cl['time'] = train_cl['time'].str[0:2].astype('int32')  # 取小时
train_cl = train_cl.drop(['state', 'location'], axis=1)       # 弱特征，先去掉
train_cl = pd.get_dummies(train_cl, columns=['car_value'])    # 车辆等级做哑变量
train_cl = train_cl.fillna(train_cl.mean())                   # 剩余缺失用均值兜底

# 特征里去掉 id 和目标相关列，避免信息泄漏
X = train_cl.drop(['record_type', 'customer_ID', 'plan', 'A', 'B', 'C', 'D',
                   'E', 'F', 'G'], axis=1)

# 以 A 位为例：每个客户取最后一条成交记录作为样本标签
y_A = train_cl.loc[train_cl['record_type'] == 1, 'A']

# 一个基础 LightGBM 分类器，先用网格搜索定 num_leaves
estimator = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=10000)
param_grid = {'num_leaves': [x for x in range(10, 50, 1)]}
gridsearch = GridSearchCV(estimator, param_grid, cv=3)
gridsearch.fit(X_last, y_A)
```

这里 `X_last` 表示「每个客户最后一条浏览记录的特征」（因为答案只跟最终成交有关，用最后一条记录最贴近决策时刻）。`num_leaves` 是 LightGBM 最重要的复杂度参数，控制每棵树最多长多少个叶子——叶子越多拟合能力越强，也越容易过拟合。材料里正是在 `10～49` 这个范围里搜的。

**思路二：修正标签之后重新训练并调参。** 网格搜索太慢时，改用「先定一棵合理的树、再看验证曲线」的办法。我们用 `early_stopping_rounds` 让模型自己决定训练到第几轮停：

```python
from sklearn.model_selection import train_test_split

# 切出 15% 做验证，固定随机种子保证可复现
X_train, X_test, y_train, y_test = train_test_split(
    X_last, y_A, test_size=0.15, random_state=20)

model = lgb.LGBMClassifier(
    learning_rate=0.1,       # 学习率：每棵树只修正一点点，防止一步走过头
    num_leaves=31,           # 默认值，先用它跑基线
    n_estimators=5000,       # 上限给足，让早停决定实际轮数
    metric='multi_logloss',  # 多分类看对数损失
    random_state=20
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          eval_metric=['multi_logloss'],
          early_stopping_rounds=50,   # 验证分数 50 轮不涨就停
          verbose=False)

# 直接在验证集上看"逐位准确率"，这才是比赛的分数
acc_A = (model.predict(X_test) == y_test).mean()
print('A 位准确率:', acc_A)
```

我们按同样的流程给 `B`～`G` 各训一个模型。有一点值得强调：**有的位几乎所有人都选同一个值**，这种位单靠「猜众数」就能有很高的准确率，模型未必能提升多少；而差异大的位（比如保额、免赔额类选项）才是拉开差距的地方。所以我们的调参精力主要花在后者上。

### 2.4 模型测试与指标报告

预测时，对每个客户取他最后一条浏览记录的特征，让 7 个模型各输出一位，再拼成字符串：

```python
# 取出每个客户的最后一条记录（通常是 record_type = 1 那条）
test_last = test.sort_values(['customer_ID', 'shopping_pt']).groupby('customer_ID').last()

# 7 个模型分别预测一位；用同一个预处理后的特征表
digits = []
for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    digits.append(models[letter].predict(test_last_features))

# 把 7 位拼成 7 位字符串，zfill 补回前导 0（0111111 这类不能丢 0）
plan_str = [''.join(str(int(d[i])) for d in digits).zfill(7)
            for i in range(len(test_last))]

# 按提交模板的列名写出
submission = pd.DataFrame({'customer_ID': test_last.index, 'plan': plan_str})
submission.to_csv('submission.csv', index=False)
```

**关于分数，我们必须诚实说明。** 材料里的笔记本只跑到「网格搜索」这一步就结束了，而且标签用的是 `record_type`，所以**材料里没有可以引用的比赛分数**。我们能确定的是：练习中的二分类 AUC 约 **0.853**，这说明 LightGBM 对这套特征有不错的区分能力；但 AUC 不等于逐位准确率，不能拿来当比赛成绩。要拿到真实分数，需要按上面的流程完整训练 7 个模型并提交。

从数据本身我们还能做一个「地板估计」：提交模板里所有客户都是 `1111111`，说明最普通的方案是众数；如果真实标签里 `1111111` 占比很高，那么「全员猜 1111111」也能拿到一个不低的基线，这就是我们要超过的下限。

---

## 三、总结

本文中我们完成了 Allstate 购车险方案预测的完整梳理。我们首先用 EDA 摸清了数据结构：训练集 665249 行、测试集 198856 行，但提交只有 55716 个客户——因为一个客户对应多条浏览记录，`record_type` 区分了浏览行（0）和成交行（1）。其次我们做了数据清洗：对 `risk_factor` 等缺失列按客户顺序 `ffill` 填补、把 7 个选项按位权拼成 7 位方案号、把 `time` 拆成小时和分钟、对 `state` 和 `car_value` 做 LabelEncoder，并用热力图确认 `record_type`、`day`、`state`、`location` 等列的信息量很弱。接着我们根据材料里的 LightGBM 练习整理了建模方案：把「预测整个方案」拆成「7 个逐位分类器」，用 `train_test_split` 切验证集、用 `early_stopping_rounds` 控制训练轮数、用 `GridSearchCV` 在 `num_leaves` 的 10～49 之间搜索。最后我们按提交模板逐位拼出 `plan` 并用 `zfill(7)` 补齐前导零。需要说明的是，材料里的练习把 `record_type` 当成标签（AUC 约 0.853），这不是真正的比赛目标，我们已经在方案里把它纠正为 7 位方案。

这个比赛给我们的最大启发是：**先看懂数据的组织方式，再谈模型。** 一个客户多行、只有最后一行是答案，这个结构决定了我们必须「按客户聚合」而不是「逐行预测」；也决定了逐位拆解比一次性预测 7 位数更合理。另外，`C_previous`（上次买的方案）和这次方案的高度相似，提醒我们**历史行为是这类问题里最便宜也最强的特征**——这一点在任何「预测用户选择」的任务里都值得先检查一遍。