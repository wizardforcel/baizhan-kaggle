# Acquire 高价值顾客发现挑战（Acquire Valued Shoppers Challenge）

## 一、比赛简介

### 1.1 比赛背景

零售商家经常给顾客发优惠券（coupon / offer），希望顾客收到券之后再到店里消费一次。但钱花出去之后，商家最想知道的是：**这张券到底有没有把人拉回来？哪些顾客看到优惠券后会真正变成「回头客」？** 如果能提前预测出这一点，商家就可以只给「有把握被拉回来」的顾客发券，把预算花在刀刃上。

这场比赛用的是一家大型零售商的真实运营数据：2013 年发放的 37 种优惠券，以及顾客在这些连锁店里的一段长达一年多的交易记录——每一条记录都细到「哪个顾客、哪家店、买了哪个品牌、几件、多少钱」。比赛要求参赛者预测：**某个顾客收到某个优惠券之后，会不会再次购买**。

这场比赛的难点有三个。一是**数据量巨大**——交易流水有 3 亿 4600 多万行，普通电脑一次性根本读不进来；二是**正负样本不平衡**——大多数顾客收到券后并不会再买；三是**特征必须自己造**——原始流水里没有「这个顾客喜欢什么」这种现成字段，得靠聚合统计把顾客的行为「画像」出来。

### 1.2 比赛目标

比赛提供三张核心表：

- `offers.csv`：优惠券定义，37 行 6 列——优惠券 id、所属类目、门槛数量、公司、折扣金额、品牌；
- `trainHistory.csv`：训练集，160057 行 7 列——顾客 id（`id`）、连锁店（`chain`）、收到的优惠券（`offer`）、所在市场（`market`）、再购次数（`repeattrips`）、是否成为回头客（`repeater`，取值 t/f）、发券日期（`offerdate`）；
- `transactions.csv`：顾客的历史交易流水，共 349655789 行、11 列——顾客 id、连锁店、部门、类目、公司、品牌、日期、购买数量、金额等。

测试集是同一格式的另一批顾客记录（`testHistory.csv`），但**没有 `repeater` 标签**。我们要为每一行「顾客 × 优惠券」输出一个 0 到 1 之间的概率，字段名为 **`repeatProbability`**，表示这位顾客收到这张券后再次购买的可能性。

### 1.3 评价标准

比赛使用 **AUC（ROC 曲线下面积）** 作为评分指标，越接近 1 越好，0.5 相当于随机瞎猜。AUC 和「准确率」不一样：准确率只关心你猜对了几行，而 AUC 关心的是**排序质量**——所有真正的回头客，你给他们的概率是不是普遍高于不会回头的顾客。因为比赛要求的输出就是概率、而且正样本只占少数，用 AUC 来评价最公平。

计算方式也简单：把样本按预测概率从高到低排好队，AUC 就是「随机挑一个回头客和一个非回头客，回头客排在前面」的概率。它有一个很好的性质——**不管正样本占多少比例，只要排序对，分数就高**。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**先把 3 亿多行交易流水「压」到 16 万条顾客记录上——以顾客为主键，统计他在各个维度上的购买行为，再用这些特征训练分类器。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先读入两张小表，再用「分块读取」（`chunksize`）的方式扫一遍大表——一次读 2000 万行，读完就丢掉，只累加行数，这样内存不会爆：

```python
import pandas as pd

offers = pd.read_csv('offers.csv.gz', compression='gzip')
history = pd.read_csv('trainHistory.csv.gz', compression='gzip')
test_history = pd.read_csv('testHistory.csv.gz', compression='gzip')  # 测试集

print(offers.shape)         # (37, 6): 37 种优惠券
print(history.shape)        # (160057, 7): 16 万条顾客记录
print(test_history.shape)   # 测试集同一格式, 但没有 repeater 标签

# 分块统计交易流水总行数, 每块 2000 万行
no_entries = 0
for chunk in pd.read_csv('transactions.csv.gz', compression='gzip',
                         chunksize=2 * 10**7, engine='c'):
    no_entries += chunk.shape[0]
print(no_entries)      # 349655789
```

大表原来有 **3 亿 4965 万多行**，一次性读进来会把内存撑爆，所以「分块 + 只统计」是这种数据量下必备的检查手段。

其次我们看时间范围和标签分布。日期列读进来是字符串，先转成日期类型，才能算跨度：

```python
# 发券日期转成日期类型, 才能比较先后和算跨度
history['offerdate'] = pd.to_datetime(history['offerdate'])
print(history['offerdate'].min(), history['offerdate'].max())
# 2013-03-01 ~ 2013-04-30, 训练集的发券周期正好 60 天

# 有没有缺失值
print(history.isna().sum().sum())   # 0, 没有缺失

# 再购次数(repeattrips)的分布
history['repeattrips'].describe()
# 最大 2124 次, 这是一个极端异常值; 超过 100 次的只有 3 个顾客
```

我们注意到几件对后续建模很关键的事：一是训练集的发券时间集中在 2013 年 3–4 月的 60 天里；二是交易流水覆盖的时间从 2012 年 3 月一直到 2013 年 7 月，约 513 天，比发券期长得多——这给了我们做「发券前 N 天顾客买过什么」这类**时间窗口特征**的素材；三是 `repeattrips` 有几个被记成上千次的离群点，如果不处理，它会主导模型。

### 2.2 数据清洗

其次我们做清洗和特征工程，这是本场真正的重头戏。第一步，因为交易流水太大，我们只保留「和优惠券有关」的记录——即公司、品牌、类目出现在优惠券表或测试顾客表里的那些行，其余无关交易全部丢掉，把 3 亿多行压到可处理的规模：

```python
# 收集优惠券和测试顾客涉及的公司/品牌/类目, 组成三个集合
comp_set = set(history['company']) | set(test_history['company'])
brand_set = set(history['brand']) | set(test_history['brand'])
cat_set = set(history['category']) | set(test_history['category'])

# 分块过滤大表: 只保留和优惠券相关的交易, 其余丢掉
parts = []
for chunk in pd.read_csv('transactions.csv.gz', compression='gzip',
                         chunksize=2 * 10**7, engine='c'):
    keep = chunk.loc[chunk['company'].isin(comp_set) &
                     chunk['brand'].isin(brand_set) &
                     chunk['category'].isin(cat_set)]
    parts.append(keep)

# 把过滤后的各块拼回一张表, 后面所有聚合都基于它
offer_tx = pd.concat(parts, ignore_index=True)
offer_tx.to_csv('transactions_filtered.csv', index=False)
```

第二步，以顾客 id 为主键，从这张过滤后的表里统计两类特征：

- **全期累计特征**：某个顾客对某个公司 / 品牌 / 类目，一共买了多少次、多少件、花了多少钱；
- **时间窗口特征**：发券日期前 7 天 / 30 天 / 60 天内，他对某个公司 / 品牌 / 类目的购买次数、数量和金额。

我们把统计动作封装成一个可复用的函数，避免写几十遍相似的 `groupby`：

```python
def add_agg(df, src, keys, value, name, how):
    """
    按 keys 从 src 里分组统计, 把结果合并回主表 df, 并给没买过的人填 0。
    how='size' 时统计行数(购买次数); 否则对 value 列求和。
    """
    if how == 'size':
        feat = src.groupby(keys).size().rename(name).reset_index()
    else:
        feat = src.groupby(keys)[value].sum().rename(name).reset_index()
    df = df.merge(feat, on=keys, how='left')
    df[name] = df[name].fillna(0)   # 没买过就记 0, 不能留空
    return df

# 把每笔交易对应到「哪位顾客、哪张券的发券日」上, 才能算「发券前 N 天」
tx = offer_tx.merge(history[['id', 'offerdate']], on='id', how='inner')
tx['days_before'] = (tx['offerdate'] - pd.to_datetime(tx['date'])).dt.days

# 例: 发券前 7 天内, 顾客对每个公司的购买次数
tx7 = tx[(tx['days_before'] >= 0) & (tx['days_before'] <= 7)]
history = add_agg(history, tx7, ['id', 'offerdate', 'company'],
                  None, 'company_cnt_7', 'size')
```

第三步，处理类别列和离群值。类别列（市场、优惠券、连锁店、公司、品牌、类目）都是文本，我们转成 one-hot 哑变量；`repeattrips` 那几个上千次的异常值，我们不去修正它本身，而是把它从特征里排除——它和标签「是否回头」高度相关，留着等于直接把答案塞给模型。最后把标签 t/f 转成 0/1：

```python
# 类别列 one-hot 编码, drop_first 去掉冗余的一列
for d in ['market', 'offer', 'chain', 'company', 'brand', 'category']:
    dum = pd.get_dummies(history[d], prefix=d, prefix_sep='_', drop_first=True)
    history = pd.concat([history, dum], axis=1)

# 训练集和测试集的特征列必须完全对齐, 测试集多出来的列补 0
for col in set(test_history.columns) - set(history.columns):
    history[col] = 0

# 标签: t/f -> 1/0
history['repeater'] = (history['repeater'] == 't').astype(int)

# repeattrips 是标签的「泄露版」, 建模时必须剔掉
X = history.drop(['id', 'repeater', 'repeattrips', 'offerdate'], axis=1)
y = history['repeater']
```

### 2.3 模型训练与超参数调优

接着我们划分数据、挑模型。因为「回头客」是少数类，切分时我们用 `stratify` 按标签分层，保证训练集和验证集里正样本的比例一致，否则验证集的 AUC 会很不稳定：

```python
from sklearn.model_selection import train_test_split

# 分层抽样: 训练/验证两边的正样本比例保持一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123, stratify=history['repeater'])
```

我们首先用逻辑回归跑一个基线——它收敛稳、对高维稀疏特征不挑食，适合先判断「这些特征到底有没有信号」：

```python
from sklearn.linear_model import LogisticRegression

# 先跑个逻辑回归基线, 判断这批特征是否可用
lr = LogisticRegression(max_iter=100000, solver='lbfgs')
lr.fit(X_train, y_train)
print('LR 的验证准确率:', lr.score(X_test, y_test))
```

接下来我们做了一次「候选模型大阅兵」：把逻辑回归、LDA、KNN、决策树、朴素贝叶斯、AdaBoost、Bagging、随机森林、梯度提升等十几个常用分类器，统统放到 3 折交叉验证里，统一用准确率打分，看谁最适合这套特征：

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn import ensemble, naive_bayes, neighbors, tree, discriminant_analysis

# 把候选模型放进列表, 统一跑交叉验证
candidates = [('LogisticRegression', LogisticRegression(max_iter=100000)),
              ('KNN', neighbors.KNeighborsClassifier()),
              ('DecisionTree', tree.DecisionTreeClassifier()),
              ('GaussianNB', naive_bayes.GaussianNB()),
              ('AdaBoost', ensemble.AdaBoostClassifier()),
              ('RandomForest', ensemble.RandomForestClassifier()),
              ('GradientBoosting', ensemble.GradientBoostingClassifier())]

for name, model in candidates:
    # 3 折、每折打乱, random_state 固定保证可复现
    kfold = KFold(n_splits=3, random_state=5, shuffle=True)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    print('%s: 平均 %.3f, 标准差 %.3f' % (name, scores.mean(), scores.std()))
```

综合比较后，我们把主力放在梯度提升树（GradientBoosting）上——它擅长在「特征几十上百个、样本十几万」的表格式数据上找非线性的组合规律。接着外面套一层 `StandardScaler` 做标准化，再用网格搜索扫三个关键超参数：树深、学习率、随机种子：

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 要搜索的参数网格
param_grid = {'max_depth': [3, 5, 7],        # 树深: 3/5/7 三档都试
              'learning_rate': [0.1, 0.15],  # 学习率: 模型每轮修正的幅度
              'random_state': [10, 20, 30]}  # 随机种子: 减少单次切分的偶然性

# 标准化 + 网格搜索(2 折验证, 兼顾速度)
gbm = make_pipeline(
    StandardScaler(),
    GridSearchCV(ensemble.GradientBoostingClassifier(),
                 param_grid, cv=2, refit=True))
gbm.fit(X_train, y_train)
```

网格搜索在这类「特征多、训练一轮就要几分钟」的场景里，2 折已经够用来区分参数的优劣；把 `refit=True` 打开，搜索结束后会自动用最优参数在全部训练数据上重新训练一次。

### 2.4 模型测试与指标报告

最后进入评估。分类模型先看混淆矩阵和分类报告（精确率、召回率、F1），但这场比赛真正关心的是 AUC，所以我们也把 ROC 曲线画出来：

```python
from sklearn.metrics import roc_auc_score, roc_curve

# 用概率而不是 0/1 标签: AUC 关心的是排序
prob = gbm.predict_proba(X_test)[:, 1]

# 比赛指标: AUC, 越接近 1 越好, 0.5 等于随机猜
auc = roc_auc_score(y_test, prob)
print('AUC:', auc)

# 画 ROC 曲线: 横轴假阳率, 纵轴真阳率; 对角线就是随机水平
fpr, tpr, _ = roc_curve(y_test, prob)
plt.plot(fpr, tpr, label='GBM (AUC = %.3f)' % auc)
plt.plot([0, 1], [0, 1], 'r--', label='random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend()
```

用概率来算 AUC 是关键：如果把 `predict` 出来的 0/1 标签直接丢给 AUC，样本只剩下两种分数，ROC 曲线会变成折线，分数会被严重低估。

确认流程可用后，我们对测试顾客做同样的特征处理后预测，拼上顾客 id，输出比赛要求的 `repeatProbability`：

```python
# 测试集同样剔掉不该作为特征的列(它本来就没有 repeater/repeattrips)
drop_cols = [c for c in ['id', 'repeater', 'repeattrips', 'offerdate']
             if c in test_history.columns]
test_X = test_history.drop(columns=drop_cols)

# 按训练集的列顺序对齐; 测试集没见过的 one-hot 列直接补 0
test_X = test_X.reindex(columns=X.columns, fill_value=0)

# 每位顾客的回头概率, 取正类(回头)那一列
probs = gbm.predict_proba(test_X)[:, 1]

# 按要求拼成 id + repeatProbability 两列
final = pd.DataFrame({'id': test_history['id'], 'repeatProbability': probs})
final.to_csv('finalPredictions.csv', index=False)
```

需要说明一点：这套流程整理自竞赛社区里广为流传的「Walkthrough」笔记本，笔记本本身在复现时因为数据文件路径的问题中途中断，所以材料里没有留下它最终的 AUC 数值——我们不对分数做任何编造。我们保留下来的是这条主线的三个关键判断：**先把无关交易过滤掉、再把交易聚合成顾客级行为特征、最后用交叉验证筛选模型并以 AUC 为准评估。**

---

## 三、总结

本文中我们完成了「发现高价值回头客」的完整流程。我们首先用分块读取摸清了数据规模：37 种优惠券、160057 条顾客记录、约 3.5 亿行交易流水，训练集发券周期 60 天、交易覆盖 513 天、没有缺失值，`repeattrips` 里存在几位上千次的离群顾客。其次我们做了数据清洗与特征工程：先把交易流水按优惠券涉及的公司/品牌/类目过滤掉大部分无关记录，再以顾客为主键聚合出「全期累计 + 7/30/60 天时间窗口」的购买次数、数量与金额特征，类别列转 one-hot，标签 t/f 转 0/1，并把 `repeattrips` 从特征里剔除以防标签泄露。接着我们做分层划分，用逻辑回归打底，再用十几个分类器的交叉验证做横向比选，最终选定梯度提升并做网格搜索。最后我们以 AUC 为指标评估，并输出 `repeatProbability` 提交文件。

这个比赛给我们的最大启发是：**当原始数据大到单机放不下时，特征工程的第一步往往不是建模，而是「降维」——围绕目标把无关数据过滤掉，再把零散的交易聚合成顾客级别的行为画像，模型才有用武之地。**