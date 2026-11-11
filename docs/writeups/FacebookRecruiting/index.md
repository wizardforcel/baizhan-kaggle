# Facebook 社交网络链接预测（Facebook Recruiting：Link Prediction）

## 一、比赛简介

### 1.1 比赛背景

这是 Facebook 在 Kaggle 上举办过的一场「社交网络链接预测」挑战。简单说，Facebook 想知道**在它的好友关系网络里，哪两个人接下来会「互相关注」**。我们的数据是一张**有向社交图**：图中的每个节点是一个人，每一条有向边代表「一个人关注了另一个人」。图里只有约 190 万（1,862,220 以内）个用户节点，边与边之间没有什么文本内容，唯一的原始信息就是「谁关注了谁」。

链接预测问题在真实产品里对应的是「好友推荐」：当你打开社交 App 时，系统要猜出你可能认识谁。我们不能把图上已有的边再推荐一遍，只能去找**还没出现、但有可能会出现的边**。比赛的难点在于，图的规模很大（训练集就有 943 万条边），而且测试集里混着许多从未在训练图里出现过的「新用户」，存在明显的冷启动问题。

### 1.2 比赛目标

训练文件 `train.csv` 只有两列，每一行代表一条关注关系：

- `source_node`：发起关注的人（int）；
- `destination_node`：被关注的人（int）。

训练集一共有 9,437,519 行（条边），其中不同的 `source_node` 有 1,587,708 个，占总行数约 16.8%。测试文件 `test.csv` 给的是形状相同的候选对（约 377 万行），我们要判断其中每一对「关注关系是否真的会发生」。换句话说，这是一个**给缺失边打标签**的任务：是就是 1，不是就是 0。

### 1.3 评价标准

比赛原本是一个链接预测/推荐类任务：模型对测试集合里的每一条「候选边」给出一个概率或标签，分数越高的边越值得推荐。材料里的主线方案把问题改造成了**二分类**：对每一对 `(source_node, destination_node)`，预测「这条边会不会出现」。因为做的是推荐，**精确率和召回率都重要**——推荐一堆没人想关注的账户（低精确率）和漏掉真正会形成的关系（低召回率）都会伤用户体验，所以作者用 **F1 分数**作为工作指标：F1 是精确率和召回率的调和平均，两个都高 F1 才高。此外还配合混淆矩阵和 ROC-AUC 来观察整体表现。

> 小提示：F1 是针对某一个判断阈值算出来的，比赛里更好的做法是看模型给出的**概率排序**。不过主线方案里先证明了「分类能分得开」，排序优化是后话。

---

## 二、解题过程

思路一句话概括：**把「两节点之间是否有边」这个图论问题，重构成一个「两个节点的相似度够不够高」的二分类问题。** 我们围绕每个候选边构造一群人、图的统计特征，再用树模型分类。按 EDA、数据清洗、模型训练、测试报告的流程，我们一步步做下来。

### 2.1 数据探索（EDA）

第一步先把图读进来，看看网络的基本结构。用 `networkx` 把边列表建成有向图，然后统计每个节点的粉丝数（入度）和关注数（出度）：

```python
import pandas as pd
import networkx as nx

# 读边列表，建成有向图：边的方向是 source -> destination
data = pd.read_csv('../input/train.csv')
g = nx.from_pandas_edgelist(data, source='source_node',
                            target='destination_node', create_using=nx.DiGraph())
print(nx.info(g))           # 打印节点数、边数、平均度等信息

# 看"粉丝数"（入度）的分布，从 90 百分位一路看到 99.9 百分位
indegree = list(dict(g.in_degree()).values())
for i in range(90, 101):
    print(i, 'Percentile:', np.percentile(indegree, i))
```

分布结果非常典型：**绝大多数人只有一个很窄的粉丝圈**。材料里的 EDA 结论是，99% 的节点粉丝数不超过 40；少数「明星」节点的粉丝数远超这个数，把长尾拉得很长。整张图还被切成许多大小不一的弱连通分量，既有挤满了明星的大连通块，也有一些彼此孤立的小团体。

我们还注意到几个对建模有用的信息：不同的 `source_node` 有 1,587,708 个，只占全部行数的 16.8%——平均下来每个发起点大约只对应 6 条边，说明大量用户的关注行为非常稀疏；而测试集里有相当比例的节点在训练图里完全没露过面。这意味着**纯靠节点自身属性（粉丝数等）的模型会遇到「新用户冷启动」**，必须靠节点之间的关系结构来兜底。

### 2.2 数据清洗

这是整场方案里最关键、也最体现图论味道的一步：图数据里只有「存在的边」，**没有「不存在的边」**，而分类器恰恰需要反例来学习。所以我们得**手工造「坏边」**：随机挑两个不构成边的节点作为负样本。光随机不行，我们要求负样本之间的最短路径大于 2——不然一个「共同好友的三度关系」被当成正例的难度就失控了：

```python
import csv, random, pickle

# 把图中已有的边做成哈希表，方便 O(1) 判重
edges = {}
for row in csv.reader(open('train.csv')):
    edges[(row[0], row[1])] = 1

# 造和正例一样多的负样本：随机两个节点，要求 a != b 且它们之间没有边，
# 且图中它们的最短路径长度 > 2（保证"真的不像有边"）
missing_edges = set()
while len(missing_edges) < 9_437_519:
    a, b = random.randint(1, 1_862_220), random.randint(1, 1_862_220)
    if (a, b) in edges or a == b:
        continue
    try:
        if nx.shortest_path_length(g, source=a, target=b) <= 2:
            continue
    except nx.NetworkXNoPath:
        pass
    missing_edges.add((a, b))
```

拿到和正例数量相同的负样本后，我们把所有边按 **80/20 划分**成训练和验证。这里有一个必须小心的点：**特征必须在「去掉验证边之后的图」上计算**。如果先用完整图给验证样本算「共同好友」之类的特征，验证样本的特征里就混进了它自己这条边的信息，属于泄漏。作者的解法是先按 80/20 把正例分成两部分，其中训练的正例才被允许加入 `train_graph`，验证边的特征全部在这个修剪过的图上算：

```python
from sklearn.model_selection import train_test_split

df_pos = pd.read_csv('train.csv')                        # 正例（真实存在的边）
df_neg = pd.DataFrame(list(missing_edges),               # 负例（我们造的坏边）
                      columns=['source_node', 'destination_node'])

# 正负样本分开切 8:2，让两边各自保持 80/20 比例
train_pos, val_pos = train_test_split(df_pos, np.ones(len(df_pos)),
                                      test_size=0.2, random_state=9)
train_neg, val_neg = train_test_split(df_neg, np.zeros(len(df_neg)),
                                      test_size=0.2, random_state=9)

# 只有训练正例进入"训练图"——这是防泄漏的关键
train_graph = nx.from_pandas_edgelist(train_pos, source='source_node',
                                      target='destination_node', create_using=nx.DiGraph())
```

至此，训练集中每个样本被表示成 `(source_node, destination_node)` 加一个标签 `indicator_link`（1 正 / 0 负），数据准备完毕。

### 2.3 模型训练与超参数调优

**特征工程：把「两节点相似不相似」量化成一排数字。** 我们对每一条候选边构造了几类特征：

1. **共同关注/共同粉丝的相似度**：Jaccard 系数（交集占并集的比例）和 Cosine 相似度，都分别对「关注的人」和「粉丝」算一遍；
2. **各自的规模和交集**：source 和 destination 各自的粉丝数、关注数，以及两个人共同粉丝/共同关注的个数；
3. **图论指标**：Adamic/Adar 指数（共同邻居越多、越冷门的共同邻居权重越高）、是否互相回关、是否属于同一个弱连通分量、两人之间的最短路径长度（去掉直接边后）；
4. **点的重要性指标**：PageRank、Katz 中心性、HITS 的 hubs/authorities 分数，以及 SVD 隐含向量点积、优先连接分（preferential attachment）等；
5. **边权重特征**：受文献 `Graph-based Features for Supervised Link Prediction` 启发，给每条边定一个权重 `W = 1 / sqrt(1 + |X|)`，意思是共同邻居越多的节点对，越可能只是「都粉了一个明星」而彼此陌生，权重应被压低。

构造这些特征耗时很长，因为每一条候选边都要去图里查邻居、算最短路径，300 万级别的样本量乘以几十个图查询是实打实的开销。作者的做法是每次只抽样一部分（训练 10 万行、验证 5 万行）来算特征，并把中间结果缓存成 HDF5 文件复用，避免重复劳动。特征构造的关键代码示意如下：

```python
import math

def jaccard(a_set, b_set):
    """Jaccard 相似度：交集大小 / 并集大小，衡量两个用户朋友圈的重叠程度"""
    return len(a_set & b_set) / len(a_set | b_set)

# 对一条边 (a, b)，提取一大串数值特征
def node_features(a, b):
    a_fans = set(train_graph.predecessors(a))   # a 的粉丝
    b_fans = set(train_graph.predecessors(b))
    a_follow = set(train_graph.successors(a))   # a 关注的人
    b_follow = set(train_graph.successors(b))
    return {
        'jaccard_fans': jaccard(a_fans, b_fans),          # 粉丝圈重叠程度
        'jaccard_follow': jaccard(a_follow, b_follow),    # 关注圈重叠程度
        'n_fans_s': len(a_fans), 'n_fans_d': len(b_fans), # 粉丝数
        'n_follow_s': len(a_follow), 'n_follow_d': len(b_follow), # 关注数
        'inter_fans': len(a_fans & b_fans),               # 共同粉丝个数
        'inter_follow': len(a_follow & b_follow),         # 共同关注个数
    }
```

**模型：随机森林先做基准，XGBoost 最终上场。** 我们先用随机森林配合 `RandomizedSearchCV` 粗调参——随机森林对这类表格型特征很稳，能快速给出一个分数基线。搜索范围包括树的棵数、最大深度、叶节点最少样本数等，用 10 折交叉验证、选 F1 最高的一组：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {'n_estimators': randint(105, 125),    # 树太多训练慢，太少欠拟合
              'max_depth': randint(10, 15),         # 树太深容易过拟合
              'min_samples_split': randint(110, 190),
              'min_samples_leaf': randint(25, 65)}

clf = RandomForestClassifier(random_state=25, n_jobs=-1)
rf_random = RandomizedSearchCV(clf, param_distributions=param_dist,
                               n_iter=5, cv=10, scoring='f1', random_state=25)
rf_random.fit(df_x_train, df_y_train)
print(rf_random.best_estimator_)
```

随机森林的最好配置（深度 14、121 棵树、最小叶样本 28）能拿到还不错的交叉验证 F1，但训练集和测试集分数差得较大，过拟合倾向明显。于是我们换到 **XGBoost**，用随机搜索在更大网格里找参数，然后用 `CalibratedClassifierCV` 对概率做 sigmoid 校准，让预测概率接近真实频率，最后据此生成提交：

```python
from xgboost import XGBClassifier

# 先随机搜索一组常见超参数
params = {'min_child_weight': [1, 5, 10], 'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
          'n_estimators': [100, 200, 500, 1000, 2000], 'gamma': [0.5, 1, 1.5, 2, 5],
          'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0],
          'max_depth': [3, 4, 5]}
best = XGBClassifier(subsample=0.8, learning_rate=0.2, colsample_bytree=0.6,
                     max_depth=5, n_estimators=1000, min_child_weight=5, gamma=5)
best.fit(x_train, y_train)

# 用 sigmoid 校准把树的输出变成可信的概率，方便按概率排序推荐
clf = CalibratedClassifierCV(best, method='sigmoid')
clf.fit(x_train, y_train)
```

### 2.4 模型测试与指标报告

在 50,000 个样本的验证集（正负各半）上，我们同时看了混淆矩阵、ROC 曲线和 F1。XGBoost 最终的结果是：**训练集 F1 = 0.992，测试集 F1 = 0.912**。测试分数比训练低约 0.08，说明有过拟合，但没有断崖式下滑，仍然远高于随机猜测（二分随机猜测的 F1 基准接近 0.5），证明特征确实抓住了「两个人会不会互相关注」的结构：

```python
from sklearn.metrics import f1_score, roc_curve, auc

y_hat = clf.predict(x_test)
print('Test F1:', f1_score(y_test, y_hat))

# ROC 曲线与 AUC：看模型在不同阈值下的真阳率/假阳率表现
fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(x_test)[:, 1])
print('AUC:', auc(fpr, tpr))
```

结合前面的过程，模型的强项在**结构特征**：`jaccard`、`adar_index`、`PageRank` 差距是决定预测的主要力量；而单个节点的粉丝数等「规模类」特征，在遇到测试集里的冷启动节点时帮不上多少忙。这也是链接预测任务的普遍结论——**与其问“这个人和那个人谁更活跃”，不如问“他们的社交圈叠了多少”。**

最后把验证用的同一套特征工程作用到真正的测试对（377 万行）上，让模型输出概率并按 `seg_id` 写出提交文件。因为测试规模大，特征计算和预测都用了分批读表，避免内存卡死。

---

## 三、总结

本文中我们完成了 Facebook 社交网络链接预测比赛的完整流程。我们首先做 EDA，摸清这张有向图的结构：943 万条边、约 190 万个节点、绝大多人粉丝不超过 40、存在明显的冷启动问题。其次我们做了一件图任务特有的事——**手工构造负样本**：随机生成 943 万条「坏边」（无直接边且最短路径大于 2），并按 80/20 划分、在去掉验证边后的图上算特征，杜绝信息泄漏。接着我们为每条候选边计算 Jaccard、Cosine、Adamic/Adar、PageRank、Katz、边权重等几十个结构特征，先随机森林粗调参看基线，再用 XGBoost 加 sigmoid 概率校准上阵。最后验证集上得到测试 F1 = 0.912、训练 F1 = 0.992，用概率排序生成提交。

这场比赛的启发是：**图数据本身不给你「不存在的边」，分类器需要的反例得自己造，而怎么造决定了模型学什么**；同时，连接预测里真正有判别力的往往是「社会圈重叠程度」这类结构指标，而不是单节点的活跃度。