# 活动推荐引擎挑战（Event Recommendation Engine Challenge）

## 一、比赛简介

### 1.1 比赛背景

这个比赛来自一个类 Meetup / Eventbrite 的社交活动平台：用户在应用里能看到「附近有什么活动」，并对自己感兴趣的活动按「感兴趣（Interested）」或「不感兴趣（Not Interested）」按钮。比赛的最终目标，是**学会预测用户会对哪些活动感兴趣**，从而给每个用户推荐最可能参加的活动。

主办方提供了一大套活动生态数据，除了最核心的 `train.csv` 展示日志之外，还有：

- `events.csv.gz`：每个活动的元信息（时间、地点经纬度，以及活动标题/描述里 100 个常见词干的出现次数）；
- `users.csv`：用户画像（出生年份、性别、语言、时区、所在地）；
- `user_friends.csv.gz`：用户的好友关系（社交图谱）；
- `event_attendees.csv.gz`：每个活动的报名情况（yes/maybe/invited/no 四类人员列表）。

难点在于：训练集只有约 1.5 万条展示样本，而活动、用户数量庞大，属于典型的「冷启动 + 稀疏交互」推荐问题。

### 1.2 比赛目标

`train.csv` 的每行是一次「把某活动展示给了某用户」的记录，共 6 列：

- `user`、`event`：用户和活动的编号；
- `invited`：该用户是否被邀请去这个活动（0/1）；
- `timestamp`：用户看到这个活动的近似时间（ISO-8601，误差 ±2 小时）；
- `interested`：用户是否点了「感兴趣」（1/0）；
- `not_interested`：用户是否点了「不感兴趣」（1/0）。

注意 `interested` 和 `not_interested` 可能同时为 0（用户看了但什么都没点）。测试集只有前四列，要求我们预测用户对每个候选活动的兴趣。赛方同时提供了 `random_benchmark.csv`（随机推荐）和 `event_popularity_benchmark.csv`（按活动热度推荐）两个基线文件作为对照——这种评测方式说明，比赛按**推荐排名的命中质量**打分：你推荐出来的活动是否真是用户感兴趣的那个，推得越靠前越好。

### 1.3 评价标准

比赛属于排序式评价：为每个用户/每轮曝光给出推荐，命中用户真正感兴趣的活动就得高分，同时越靠前的命中越值钱。两个 benchmark 文件的目的就是告诉我们「瞎猜能得多少分」「按热度猜能得多少分」，任何方案首先必须超过「按活动热度推荐」这条线才算有价值。

---

## 二、解题过程

材料里这份 notebook 是**完整的 EDA（探索性分析）**，还给出了各张表的字段说明。我们按 EDA、数据清洗、建模思路、测试与报告四步走；其中模型部分材料没有现成代码，我们根据数据结构整理出实验思路（用伪代码说明）。

### 2.1 数据探索（EDA）

先读入训练集，看列结构、量级和缺失情况：

```python
import numpy as np
import pandas as pd

train_df = pd.read_csv('train.csv')
train_df.head()

# 规模：15398 行，6 列
train_df.shape
# 缺失值检查：6 列全为 0，说明没有缺失
train_df.isnull().sum()
```

训练集只有 **15398** 条展示记录，且完全没有缺失值。接着我们看活动和用户的分布长什么样：

```python
# 每个活动被「感兴趣」的次数——为什么是幂律分布？
# 少数几个活动被几百人感兴趣，绝大多数只有个位数
train_df.event[train_df['interested'] == 1].value_counts()
# 955398943    118
# 2529072432    94
# 1076364848    69
# ...
# 有 2623 个不同的活动
```

结论是典型的**长尾分布**：极少数热门活动（如 955398943，118 个感兴趣）占据了大部分点击，大部分活动几乎无人问津。这直接意味着「活动热度」是一个非常强的特征——推荐通常避不开热门活动。

`invited` 字段也值得单看：把 `invited == 1` 的活动单独数一遍，出现次数明显变少（大多数活动只被邀请过 1 次），说明「邀请」本身是比「广场上刷到」稀有得多的强信号——被邀请的人参加意愿天然更强，这个字段可以直接当特征用。

再看相关性：

```python
import seaborn as sns
sns.heatmap(train_df.corr())
# 各列相关系数都很低，没有哪一列和 interested 强相关
```

结论是**典型的无强单变量**：interest 是多个弱信号叠加的结果。既然单靠某一列都不够，说明要预测兴趣，得把多张表的信息综合起来。

接着我们转向 `events.csv`。它每行是一个活动的元数据：前 9 列是 `event_id`、创建者 `user_id`、开始时间、城市/州/邮编/国家、经纬度；最后 101 列是 `count_1`~`count_100` 加 `count_other`——也就是活动标题和描述里 100 个最常见词干各出现了几次。这 100 个计数直接描述了活动「聊的是什么话题」：

```python
events = pd.read_csv('events.csv.gz', compression='gzip')
events.shape          # 2 万多个活动，110 列
events.isnull().sum() # 看哪些元数据列缺失最多
# 能派上用场的计数列，之后可以和 "event" 列关联
```

同理，`users.csv` 记录了用户的语言（locale）、出生年份、性别、时区、所在地，其中很多用户是「新用户」——没有任何交互历史，只能靠画像特征补全：

```python
users = pd.read_csv('users.csv')
users.isnull().sum()                 # 出生年份、性别等有缺失
users.locale.value_counts()          # 语言分布：英语用户占绝大多数
users.location.value_counts().iloc[:10]   # 用户集中在美国哪些城市
```

这些分布让我们对「用户的多样性能给我们多少信息」有底：如果 80% 的用户集中在少数几个城市、语言高度同质，那画像特征对预测的边际贡献就不会太大。

### 2.2 数据清洗

素材里最重要的清洗工作，是把 `event_attendees.csv.gz` 和 `user_friends.csv.gz` 里**空格分隔的长字符串**解析成真正的列表，方便统计人数：

```python
# 把 "是的（going）" 列从字符串解析成用户 id 列表
for i in range(event_attendees.shape[0]):
    attendees = str(event_attendees.yes[i]).split()   # 按空格切开
    event_attendees.yes[i] = attendees
```

对 `maybe`、`invited`、`no` 三列做同样处理。有了展开后的列表，就能聚合出「这个活动有多少个 yes、多少个 maybe」这样的计数特征，它们衡量的是活动本身的受欢迎程度，与训练集里的 `interested` 强相关。

`users.csv` 的画像字段同样被分析了分布（地域、语言的 value_counts），这些可以编码成特征补齐冷启动用户的信息。

### 2.3 模型训练与超参数调优

材料没有给出建模代码，下面是我们根据数据结构整理的**实验思路（伪代码）**：把问题转成二分类——「用户会不会对这个活动感兴趣」，再按概率排序做推荐。

```python
# 构造特征：把每张表的信息聚合到 (user, event) 这一对
def build_features(train, events, users, friends, attendees):
    feats = train[['user', 'event', 'invited']].copy()
    # 时间特征：把时间戳拆出小时、星期几，测用户活跃时段
    feats['hour'] = train['timestamp'].dt.hour
    feats['weekday'] = train['timestamp'].dt.weekday
    # 活动热度：该活动一共被多少人 yes / maybe
    feats['event_yes_cnt'] = feats['event'].map(
        attendees.groupby('event_id')['yes_cnt'].first())
    # 用户被动性：该用户历史上感兴趣的比例
    feats['user_interested_rate'] = feats['user'].map(
        train.groupby('user')['interested'].mean())
    # 社交特征：好友愿意去这个活动的数量（可选，成本高）
    return feats

# 伪代码：梯度提升树二分类
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=300, max_depth=4)
model.fit(X_train, y_train)              # y = interested
prob = model.predict_proba(X_test)[:, 1]  # 输出兴趣概率，按概率排序推荐
```

如果要用社交特征，可以这样把「好友列表」和「活动报名人数」交叉起来——用集合取交集，天然跳过重复 id：

```python
# 伪代码：某用户的好友中有几位确定去某活动
friends_set = set(user_friends.friends[uid].split())       # 好友 id 集合
going_set = set(event_attendees.yes[event_id])             # 已报名者 id 集合
friends_going = len(friends_set & going_set)               # 交集大小
```

调参重点放在三类超参数上：树的数量 `n_estimators`、深度 `max_depth`、学习率 `learning_rate`。考虑到训练集只有 1.5 万行，用带早停的交叉验证防过拟合比无限堆树更重要。数据划分上必须**按时间切分**（用最新的 20% 当验证集），因为推荐系统里「未来」的行为才是我们要预测的，随机切分会高估成绩。

模型细节上有两个小坑：一是 `interested` 的**正样本占比可能不高**（绝大多数展示都是「用户看了但没点」的 0），类别不均衡时要在损失里给正样本加权重，或者在预测后单独调阈值；二是 `invited`、`timestamp` 这类「曝光侧」信息测试集里也有，属于合法特征，而 `interested`/`not_interested` 是测试集没有的标签侧字段，构建验证集时同样要保证「未来不泄漏」。

另外一个工程细节：`timestamp` 的精度是「±2 小时」，也就是时间列本身带噪声，所以从时间戳拆出的「小时」只能当粗粒度特征，不能当精确的行为时刻去对齐别的表。而 `user_friends.csv` 解析后是用户 id 的集合，可以用 `set` 判断「某个用户的某个好友是否打算去某活动」，再把命中数量作为社交热度特征——这类特征在冷启动用户（历史交互为 0）身上尤其值钱，因为画像和社交是那时仅有的信号。

### 2.4 模型测试与指标报告

评测时，我们把每个用户的候选活动按预测概率降序排序，取第一名作为推荐结果，再和 benchmark 对照：

```python
# 按用户分组，对每组内的概率排序，取概率最高的事件作为推荐
test['rank'] = test.groupby('user')['prob'].rank(ascending=False)
test_top1 = test.loc[test['rank'] == 1]

# 与两个基准线比较命中率：
# random_benchmark：随机推荐；event_popularity_benchmark：按热度推荐
# 我们的方案至少要明显超过 event_popularity_benchmark 才算有效
```

由于材料只到 EDA 为止，我们没有能写进这里的验证分数。但依据长尾分布的事实可以预期：**热度基线已经相当强**（把最热的活动推给所有人，命中率不会太低），要超越它必须在「用户细分」上下功夫——比如用好友网络的六维特征、用矩阵分解把用户和活动映射到同一隐空间再做内积打分，这些是素材里提到但尚未展开的方向。

---

## 三、总结

本文中我们完成了活动推荐引擎挑战的数据探索与方案梳理。我们首先用 EDA 摸清了数据：训练集 15398 条、无缺失、活动和用户呈长尾分布、各列相关性低；其次做了数据清洗，把报名表、好友表里的空格分隔字符串解析成列表以提取人数类特征；接着整理了两条建模思路——把 1.5 万条记录合并出特征后训练兴趣二分类器，或用矩阵分解做协同过滤；最后按时间切分交叉验证，并与热度、随机两个 benchmark 对照评估。

这个比赛给我们的最大启发是：**推荐系统的宝藏藏在多张关联表里**。单看 `train.csv` 的六列，每列和兴趣的相关性都很弱；但一旦把活动热度、好友网络、用户画像这些外部信息合并进来，稀疏的交互数据才能被补全。这也是所有「冷启动」推荐问题的共同解法——特征不能只从一张表里挖。