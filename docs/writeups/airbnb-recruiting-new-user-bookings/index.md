# Airbnb 新用户首单目的地预测（Recruiting New User Bookings）

## 一、比赛简介

### 1.1 比赛背景

Airbnb（爱彼迎）是一个在线民宿短租平台，用户可以在这个平台上订全世界各地的民宿。一个新用户第一次在 Airbnb 上订房，会选择一个目的地（目的地可以是一个国家/地区，比如美国、法国）。搞清楚一个新人会去哪里，对平台很重要：把用户更可能感兴趣的目的地放在首页靠前的位置，用户下单就更快，平台也能更准确地预测需求。

这个比赛就是让我们用数据来猜这件事。比赛提供了两类数据：一是用户注册时留下的基本资料（性别、年龄、用什么设备注册、从哪个渠道来的等）；二是用户注册后在网站上的「行为日志」（点过什么网页、搜过几次、用了多久时间）。我们看不到用户私下的聊天内容，只能靠这些「外表特征」去推断他预订的意向。

这个任务的难点在于：注册信息和「会不会去某地旅游」之间没有直接因果关系，绝大多数用户（约六成）注册后根本没有马上预订，导致目标类别严重不平衡——好在比赛不要求我们输出唯一的答案，而是让我们给出「5 个最可能的候选目的地」，用排序质量来打分。

### 1.2 比赛目标

给定一位用户的注册资料和行为记录，预测用户首次预订时会选择哪个国家/地区作为目的地。

训练集 `train_users_2.csv` 有 **213451 行、16 列**，核心字段包括：

- `id`：用户编号；
- `date_account_created`：注册日期；
- `timestamp_first_active`：第一次活跃的时间戳；
- `date_first_booking`：第一次预订日期（**训练集里有一半以上的用户是空的**，测试集里没有这一列，因为要预测的就是这个时刻之后的事）；
- `gender`：性别；
- `age`：年龄（缺失较多）；
- `signup_method`、`signup_flow`、`language`、`affiliate_channel`、`affiliate_provider`、`first_affiliate_tracked`、`signup_app`、`first_device_type`、`first_browser`：各种注册/来源/设备相关的类别特征；
- `country_destination`：目的地标签，共有 12 类，例如美国 `US`、法国 `FR`、`NDF`（没有预订）等，这就是我们要预测的目标。

另外还有 `sessions.csv`：用户的网页行为日志，**10567737 行、6 列**，记录了每次点击发生的时间、动作类型、浏览设备等等。

### 1.3 评价标准

比赛使用 **NDCG@5（归一化折损累计增益，取前 5 名）** 作为指标。NDCG 是推荐系统里常用的评价办法，它的思想是：你可以给每个用户提交 5 个候选国家，并排好顺序，「猜中」的最高分不在话下，但**猜得越靠前，得分越高，猜得越靠后，即使猜对了也要打折扣**。NDCG 越高越好，满分是 1.0（意味着对每个用户，真实目的地都排在提交列表的第一名）。

> 小提示：正因为按「排序」算分，我们后面会坚持让模型输出**概率**而不是类别，然后按概率从大到小取前 5 个。

---

## 二、解题过程

整个方案可以概括成一句话：**把用户注册信息和网页行为都变成特征，喂给一个多分类模型，让它输出 12 个目的地的概率，最后挑概率最高的 5 个提交。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先把各个数据文件读进来，确认规模、列名和缺失情况：

```python
import pandas as pd
import numpy as np

# 读取用户注册信息；压缩包可以直接读
train = pd.read_csv('../input/train_users_2.csv.zip')
test  = pd.read_csv('../input/test_users.csv.zip')

# 行为日志：1056 万行，每个用户会占多行
sessions = pd.read_csv('../input/sessions.csv.zip')

# 先看训练集的规模、哪些列有缺失
print(train.shape)
train.info()
```

打印结果显示训练集是 213451×16，其中 `age` 只有 125461 个非空值（缺了约 8.8 万行），`date_first_booking` 只有 88908 个非空（超过一半人没订过房）。这些缺失都是「正常的缺失」——一个人本来就可能没有年龄记录、没有订单，我们不能随便删掉，而是要把「缺失」本身也当成一种信息。

接着我们看了标签的分布，这一步非常重要：

```python
# 统计每个目的地出现的次数
train.groupby('country_destination').count()['id']
# 输出（节选）：
# NDF  124543
# US    62376
# other  10094
# FR     5023
# IT     2835
# GB     2324
# ES     2249
# ...
```

`NDF`（No Destination Found，没有预订）一家独大，占了 12 个类别里的差不多六成，美国 `US` 次之。这说明如果我们**什么都不学，给每个用户固定提交最热门的 5 个国家 `["NDF","US","other","FR","IT"]`**，就能拿到 **0.85359** 的公榜分——这就是一个非常强的「地板」，也提醒我们：想从 0.85 往上爬，靠的不是把大类猜准，而是把每个用户**个性化的排序**做出来。

### 2.2 数据清洗

训练集和测试集我们合并在一起处理，保证类别编码、特征统计两边一致。清洗主要做三件事：把日期字符串拆成数值、清洗年龄异常值、把类别列转成独热（one-hot）向量：

```python
from sklearn.preprocessing import LabelEncoder

# 把训练、测试拼起来，统一做特征工程；记住训练集的行数，后面要切回去
df_all = pd.concat((train.drop('country_destination',1), test), ignore_index=True)
# date_first_booking 测试集没有，且对预测目标来说是"未来信息"，丢掉
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
# 各种类别/日期列的缺失统一填成 -1，让模型知道"这一栏没填"
df_all = df_all.fillna(-1)

# 把注册日期 '2010-06-28' 拆成年、月、日三个数字
dac = np.vstack(df_all.date_account_created.astype(str).apply(
    lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year']  = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day']   = dac[:,2]

# 年龄清洗：明显不合理的年龄（小于14岁或大于100岁）统一看成"未知"
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av < 14, av > 100), -1, av)
```

然后是独热编码。之所以不把类别列直接当成数字喂给模型，是因为像 `"facebook"`、`"google"` 这种取值之间**没有大小关系**，用 1、2、3 去表示会误导模型，独热编码则把每个取值都变成一列「是不是这个值」的 0/1 布尔列。我们对 10 个类别字段都这么做，并将原来的文本列删掉：

```python
# 需要做独热编码的类别字段
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language',
             'affiliate_channel', 'affiliate_provider',
             'first_affiliate_tracked', 'signup_app',
             'first_device_type', 'first_browser']
for f in ohe_feats:
    # 字段 f 的每个取值变成一列
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

# 目标标签也转成从 0 开始的连续整数，方便模型处理
le = LabelEncoder()
y = le.fit_transform(labels)          # labels 是训练集的 country_destination
vals = df_all.values
X = vals[:piv_train]                  # 前 piv_train 行是训练集
X_test = vals[piv_train:]             # 剩下的是测试集
```

### 2.3 模型训练与超参数调优

只靠注册资料已经能跑到 0.85 左右，但**真正拉开差距的是行为日志**。我们把每个用户的所有操作（`action`、`action_type`、`action_detail`）按时间拼成一个字符串，再对这个字符串做 TF-IDF——这样「搜索了哪些房源、点开过什么页面、浏览了多久」等行为统计就都变成了特征向量：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# 按用户把行为日志聚合成一行，每个动作变成一个逗号分隔的长字符串
sess = sessions.groupby('user_id', as_index=False).agg(lambda x: x.tolist())
# 把列表转成字符串（去掉 nan），才能用它训练 TF-IDF
def to_string(actions):
    return ",".join(str(i) for i in actions if str(i) != 'nan')
sess['action'] = sess['action'].apply(to_string)

# TF-IDF：把动作序列变成向量；min_df=10 只保留出现超过 10 次的"词"
vectorizer = TfidfVectorizer(min_df=10, max_features=5000)
vectorizer.fit(sess['action'].values)
train_action = vectorizer.transform(sess['action'].values)
test_action  = vectorizer.transform(sess_test['action'].values)

# 把行为特征和注册特征水平拼接（hstack）成一张大特征表
train_merge = hstack((train_base, train_action)).tocsr()
test_merge  = hstack((test_base,  test_action)).tocsr()
```

模型我们选 **XGBoost 多分类**。XGBoost 对「稀疏 + 高维 + 类别多」的表格特征非常拿手，训练快、效果稳。因为评分要看排序，我们用 `objective='multi:softprob'` 让模型输出每一类的**概率**。这里重点不是盲目堆参数，而是配合评价指标来调——我们先把 NDCG 在本地实现出来，用它当交叉验证的评分函数：

```python
from sklearn.metrics import make_scorer

def dcg_score(y_true, y_score, k=5):
    """折损累计增益(DCG)：按预测分数排序后，位置越靠前权重越大"""
    order = np.argsort(y_score)[::-1]      # 概率从大到小的位置
    y_true = np.take(y_true, order[:k])    # 只看前 k 名
    gain = 2 ** y_true - 1                 # 命中即得高收益
    discounts = np.log2(np.arange(len(y_true)) + 2)  # 位置越靠后折扣越大
    return np.sum(gain / discounts)

def ndcg_score(ground_truth, predictions, k=5):
    """NDCG：把 DCG 除以"理想排序"的 DCG，得到 0~1 之间的分数"""
    scores = []
    for y_true, y_score in zip(ground_truth, predictions):
        actual = dcg_score(y_true, y_score, k)
        best   = dcg_score(y_true, y_true, k)   # 理想情况：真实标签排最前
        scores.append(actual / best)
    return np.mean(scores)

# 让 XGBoost 的交叉验证直接优化 NDCG@5
ndcg_scorer = make_scorer(ndcg_score, needs_proba=True, k=5)
```

设置好模型参数并训练。这个参数组合是反复试验后比较稳的一组：树比较浅（`max_depth=3`）防止过拟合，早停靠损失函数自动停止，`random_state` 固定保证结果可复现：

```python
import xgboost as xgb

gb = xgb.XGBClassifier(
    learning_rate=0.1,        # 步长设小一点，学得更细
    max_depth=3,              # 树浅，抗过拟合
    n_estimators=100,         # 100 棵树，配合早停足够
    objective='multi:softprob',  # 输出每个类别的概率，评分要概率
    eval_metric='mlogloss',   # 训练时看多分类对数损失
    random_state=0
)
gb.fit(train_merge, y)
```

我们一开始也试过在 `max_depth`、`learning_rate`、`subsample` 等参数上用 `RandomizedSearchCV` 配 NDCG 打分做网格搜索，但特征维度高、样本量大，每组参数都要重跑一遍 TF-IDF 和训练，代价很高；实际上调来调去也就提升零点零几个百分点，收益远不如「加上行为特征」这种结构性改进大。

### 2.4 模型测试与指标报告

最后一步是生成提交文件。因为提交的是每个人猜的前 5 名，步骤是：取每个用户概率最大的 5 个国家，同一个 `id` 重复 5 次、配它的 5 个候选国家，拼成一张表：

```python
# 对测试集预测出 12 个类别的概率
pred_y = gb.predict_proba(test_merge)

ids = []   # 收集用户 id（每个 id 复制 5 遍）
cts = []   # 收集国家名（每人 5 个候选）
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    # 概率从大到小排序，取前 5 个，再映射回国家名
    cts += le.inverse_transform(np.argsort(pred_y[i])[::-1])[:5].tolist()

# 生成提交文件：两列 id、country，行数 = 测试用户数 × 5
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv', index=False)
```

看一眼输出文件开头，每个用户的前 5 名都排好了，例如某个用户得到 `NDF → US → other → FR → ES`，另一个用户 `NDF → US → other → FR → IT`——排序会随用户特征变化，这就是我们不同于固定 baseline 的地方。

这个方案在公开榜上拿到约 **0.8665**，比固定提交人类最多票的「地板」0.85359 高了 **0.013** 左右。提升主要来自两部分：一是把注册日期、年龄这些杂乱字段清洗成模型能用的特征；二是行为日志的 TF-IDF 特征——**用户搜过什么、看过什么，比他是男是女更能预测他会买哪里的票**。0.85 之后每往上爬 0.001 都需要很大的努力，这也是这个比赛分数普遍扎堆在 0.86~0.87 的原因。

---

## 三、总结

本文中我们完成了 Airbnb 新用户首单目的地预测的完整流程。我们首先用 EDA 摸清了数据：训练集 213451 行、行为日志 1056 万行，标签极不均衡，NDF 独占六成，固定提交热门国籍的 baseline 就有 0.85359。其次我们做了有针对性的数据清洗：日期拆成年月日、年龄异常值归为未知、10 个类别字段独热编码。接着我们训练了 XGBoost 多分类模型，并用自定义的 NDCG@5 作为交叉验证评分函数对齐比赛指标。最后我们按「概率前 5」的格式提交，得到约 0.8665 的分数，比 baseline 提升约 0.013。

材料里还有其它尝试值得一句话带过：有人只用注册资料、不碰行为日志（分数停在 0.85 附近）；也有人用「先判断有没有预订、再分国家」的三层分类架构，思路更复杂但收益有限。这个比赛给我们的最大启发是：**结构性的特征改进（把行为日志变成特征）远胜于堆砌超参数**——在 0.85 的高地上，加一列好特征比把树加深两层有用得多。