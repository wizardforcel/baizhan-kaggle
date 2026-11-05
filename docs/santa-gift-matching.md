# 圣诞老人礼物分配（Santa Gift Matching）

## 一、比赛简介

### 1.1 比赛背景

这是一道**组合优化**题，而不是平时常见的分类或回归题。圣诞老人要发礼物：有 100 万个孩子，每个孩子都列了一张自己最想要的 10 件礼物清单；同时有 1000 种礼物，每种礼物有 1000 份，加起来正好 100 万份，每个孩子拿一份。

难点在于两边的“喜好”都要照顾。孩子这边，如果收到了自己清单里排第一的礼物，他最高兴；收到清单外的礼物，也会有一点保底快乐。圣诞老人（或者说礼物）这边也有自己的偏好：`gift_goodkids.csv` 里记录了每种礼物最想送给的 1000 个“好孩子”。如果一份礼物送给了它最喜欢的乖孩子，礼物这一侧也高兴。问题是这些偏好互相冲突——一个孩子想要 A 礼物，但 A 礼物更想送给别的孩子。我们要在 100 万个分配上做全局权衡，让**平均幸福度**尽量高。

更麻烦的是，数据里有两类特殊的孩子：前 5001 个孩子是**三胞胎**，接下来的 40000 个孩子是**双胞胎**，同一个家庭的孩子必须收到**同一种**礼物。这些“同礼”约束让问题更难解。

### 1.2 比赛目标

输入是两张表：

- `child_wishlist.csv`：每个孩子一行。第一列是孩子编号，后面 10 列是他最想要的 10 件礼物编号（按喜好从高到低排）。一共 100 万行。
- `gift_goodkids.csv`：每种礼物一行。第一列是礼物编号，后面 1000 列是礼物最喜欢送给的孩子编号（按“乖的程度”从高到低排）。一共 1000 行。

我们要输出一张分配表，包含两列：`ChildId`（孩子编号）和 `GiftId`（分到的礼物编号）。要求每个孩子恰好拿一份礼物，每种礼物恰好发出 1000 份，三胞胎和双胞胎拿到同一种礼物。

### 1.3 评价标准

比赛使用**平均归一化幸福度（Average Normalized Happiness，简称 ANH）**作为指标。它由两部分相加：

$$
\mathrm{ANH} = \overline{child\_happiness} + \overline{gift\_happiness}
$$

孩子侧：如果孩子收到自己清单里第 $i$ 位的礼物，幸福度为 $(10-i)/10$（第一位是 1.0，第十位是 0.1）；清单外的礼物只有保底值 0.05。所有孩子取平均。礼物侧类似：如果礼物送给了它好感榜第 $i$ 位的孩子，幸福度为 $(1000-i)/1000$；榜外的孩子只有保底值 0.0005。所有礼物再平均。两个平均数相加就是 ANH，**越高越好**。排行榜上领先的分数大约在 0.94 左右。

---

## 二、解题过程

这是一个 100 万顶点的指派问题（assignment problem）。我们的做法分三层：先用简单的贪心或最小费用流得到一个还不差的解，再用匈牙利方法在局部小区块里反复改进，最后迭代到收敛。下面按四步展开。

### 2.1 数据探索（EDA）

我们首先读数据，确认两张表的形状。注意第一列是编号，所以读取时把第一列去掉，剩下的才是偏好序列。

```python
import numpy as np
import pandas as pd

N_CHILDREN = 1000000   # 孩子总数
N_GIFT_TYPE = 1000     # 礼物种类
N_GIFT_QUANTITY = 1000 # 每种礼物的份数

# child_wishlist: 第一个 100 万孩子，每行 10 个想要的礼物
kids = pd.read_csv('child_wishlist.csv', header=None).values[:, 1:]
print("Kids shape:", kids.shape)

# gift_goodkids: 1000 种礼物，每行 1000 个喜欢的孩子
gifts = pd.read_csv('gift_goodkids.csv', header=None).values[:, 1:]
print("Gifts shape:", gifts.shape)
```

打印结果是 `Kids shape: (1000000, 100)` 和 `Gifts shape: (1000, 1000)`。这说明每个孩子有 100 个想要的礼物，其中前 10 个是“最想要的”，后面 90 个可能是补充数据。比赛只用到前 10 个。

我们接着统计哪些礼物最抢手、哪些礼物最没人要。

```python
from collections import Counter

# 把所有孩子的 10 个愿望拼成一个长列表
wish_list = list(kids[:, :10].flatten())

# 计数每种礼物被多少孩子想要
popularity = Counter(wish_list)

# 最受欢迎和最不受欢迎的礼物
print(popularity.most_common(5))
print(popularity.most_common()[-5:])
```

最受欢迎的礼物被近 20000 个孩子写进愿望清单，而最不受欢迎的礼物只有几十人想要。这个差别非常大，说明礼物之间极不均衡——统一“每个礼物 1000 份”的限制下，受欢迎的礼物注定分给不了所有想要的孩子，必须有人妥协。

再看一下进度检查：特殊孩子被安排在最前面，前 5001 个是三胞胎（3 人一组），接下来的 40000 个是双胞胎（2 人一组），剩下的 954999 个是普通孩子。我们给每类孩子算一个计数，提交时用来校验“同礼”约束有没有被违反。

### 2.2 数据清洗

这题没有“缺失值”需要填，也不需要删除异常样本，更不需要做标准化。真正的清洗工作是把两张偏好表**预计算成幸福度字典**：查表很快，后面算分和建图都要用它。

```python
from collections import defaultdict

N_CHILD_PREF = 10    # 每个孩子愿望清单长度
N_GIFT_PREF = 1000   # 每种礼物的好感榜长度

CHILD_HAPPINESS = {}
for c in range(N_CHILDREN):
    # 默认值：清单外的礼物只给保底 0.05
    CHILD_HAPPINESS[c] = defaultdict(lambda: 1. / (2 * N_CHILD_PREF))
    for i, g in enumerate(kids[c, :10]):
        # 位次 i=0 最高，得 1.0；越靠后越小
        CHILD_HAPPINESS[c][g] = -1. * (N_CHILD_PREF - i) / N_CHILD_PREF

GIFT_HAPPINESS = {}
for g in range(N_GIFT_TYPE):
    # 默认值：不是礼物喜欢的孩子，只有保底 0.0005
    GIFT_HAPPINESS[g] = defaultdict(lambda: 1. / (2 * N_GIFT_PREF))
    for i, ch in enumerate(gifts[g]):
        GIFT_HAPPINESS[g][ch] = -1. * (N_GIFT_PREF - i) / N_GIFT_PREF
```

这里用负数存、计算时再取相反数，只是因为材料里的代码这么写，逻辑上把“越靠前越开心”换算成一个正分数。我们只需要记住一件事：字典的返回值就是该请求对应的幸福度，查不到的取默认保底值。

有了这两张查表，我们可以写一个独立的计分函数，用来验证任何一版提交的 ANH 是多少。

```python
def my_avg_normalized_happiness(pred):
    """pred 是 (child, gift) 的列表，返回 ANH 分数"""
    total_child = 0
    total_gift = np.zeros(N_GIFT_TYPE)
    for c, g in pred:
        total_child += -CHILD_HAPPINESS[c][g]
        total_gift[g] += -GIFT_HAPPINESS[g][c]
    nch = total_child / N_CHILDREN
    ngh = np.mean(total_gift) / N_GIFT_QUANTITY
    return nch + ngh
```

这个计分函数非常重要，后面每一步改进都要靠它来确认分数有没有真的提高。注意公式里礼物侧要除以礼物份数（1000）再求平均，相当于每份礼物的平均幸福度。

### 2.3 模型训练与超参数调优

这题没有“模型超参数”，但有**优化方法的选择**。材料里有两条主线：一条是用 Google OR-Tools 的最小费用流建图求解，从头构造一个不错的解；另一条是用匈牙利方法在局部块里做改进。我们两条都介绍。

**第一步：最小费用流建图。** 我们把“发礼物”建模成网络流：源点连每种礼物，容量 1000（每种最多发 1000 份，顺序互换无妨）；每个孩子是中间节点，需要 1 单位的流；礼物节点到孩子节点的边上有费用，费用等于“孩子幸福 + 礼物幸福”的相反数。再让最小费用流求解器找一个总费用最小的流量分配，就等价于找总幸福最大。

```python
from ortools.graph import pywrapgraph

ngift_types = 1000
neach_gift = 1000
nkids = 1000000
nwishes = 100

smcf = pywrapgraph.SimpleMinCostFlow()
snodes, enodes, capacities, ucosts = [], [], [], []

# 礼物→通用池 的边：每个礼物类型先给 1000 份货
for g in range(ngift_types):
    snodes.append(g)
    enodes.append(ngift_types)          # 通用礼物节点
    capacities.append(neach_gift)
    ucosts.append(0)

# 通用池→每个孩子 的边：每个孩子只需要 1 份
kbase = ngift_types + 1
for k in range(nkids):
    snodes.append(ngift_types)
    enodes.append(kbase + k)
    capacities.append(1)
    ucosts.append(1)

# 礼物→孩子 的边：只连孩子愿望清单里的礼物
def wishcost(w, mf):
    # w 是孩子清单位次，mf 用来给三胞胎/双胞胎加权
    return int(mf * (-2 * (nwishes - w)))

for k in range(nkids):
    for w in range(nwishes):
        g = kids[k, w]
        snodes.append(g)
        enodes.append(kbase + k)
        capacities.append(1)
        ucosts.append(int(wishcost(w, 6)))   # 普通孩子 mf=6

for i in range(len(snodes)):
    smcf.AddArcWithCapacityAndUnitCost(snodes[i], enodes[i], capacities[i], ucosts[i])
```

上面为了简洁，普通孩子的费用只写了孩子侧，材料里还对三胞胎、双胞胎按家庭分组把兄弟姐妹的快乐折进同一条弧，并加入礼物侧的好感，这份代码里没有展开。三胞胎和双胞胎的“同礼”是通过事后对解做后处理保证的：求解出首个子孩子分到的礼物后，直接把同组其他孩子也指到同一种礼物，同时从礼物库存里扣掉相应份数。材料中最终把 `gcount` 修正到每个礼物类型正的库存内，并打印每种礼物的份数检查没有超发。

**第二步：匈牙利方法局部改进。** 最小费用流能给出不错的初始解，但 100 万规模的图，边有几千万条，求解很重。更轻巧的办法是：拿一个已有的解（比如社区里公开的高分解，或我们自己跑出来的解），每次随机抽一个小块（比如 400 个孩子），在小块内部用 `scipy.optimize.linear_sum_assignment` 重新求局部最优分配。匈牙利算法是精确算法，小块上的最优点一定不差于原来的分配，所以每改进一块分数只会上升。

```python
from scipy.optimize import linear_sum_assignment

BLOCK_SIZE = 400
N_BLOCKS = (N_CHILDREN - 0) // BLOCK_SIZE   # 100万 / 400 = 2500 块

# GIFT_IDS[i] 表示第 i 张实体礼物属于哪种礼物类型（0~999）
GIFT_IDS = np.array([[g] * N_GIFT_QUANTITY for g in range(N_GIFT_TYPE)]).flatten()

def optimize_block(child_block, current_gift_ids):
    """在 child_block 指定的孩子块内部做局部最优重分配"""
    gift_block = current_gift_ids[child_block]          # 当前每一张礼物
    C = np.zeros((BLOCK_SIZE, BLOCK_SIZE))              # 代价矩阵
    for i, c in enumerate(child_block):
        for j, gid in enumerate(gift_block):
            g = GIFT_IDS[gid]                           # 实体礼物→礼物类型
            # 代价 = -(孩子幸福 + 礼物幸福)
            C[i, j] = CHILD_HAPPINESS[c][g] + GIFT_HAPPINESS[g][c]
    row_ind, col_ind = linear_sum_assignment(C)         # 匈牙利，精确解
    return child_block[row_ind], gift_block[col_ind]
```

匈牙利算法的时间复杂度是 $O(n^3)$，在整个 100 万上跑是不现实的（材料里估算要上百小时才追得上开榜第一）。所以我们对每个 400 人块单独求解，块的组合数少得多，几分钟就能扫完几千块。这个“分而治之”的思想很关键：**精确算法加在小块上，既保证了每次都有改善，又把计算量控制在了可行范围**。

### 2.4 模型测试与指标报告

我们接着实现完整流程：读取初始解 → 编码成“实体礼物”编号 → 分块做匈牙利改进 → 每轮扫完统计 ANH → 写提交。

```python
subm = pd.read_csv('cpp_sub.csv')   # 拿一个初始提交当起点
initial = my_avg_normalized_happiness(subm[['ChildId', 'GiftId']].values.tolist())
print("初始 ANH:", initial)

# 记录同一种礼物里第几份：把 (礼物类型, 份次) 编码成唯一编号
subm['gift_rank'] = subm.groupby('GiftId').cumcount()
subm['gift_id'] = subm['GiftId'] * 1000 + subm['gift_rank']

# 迭代扫描：每轮随机换一批块，杜绝顺序偏差
for i in range(1):
    current = subm['gift_id'].values
    rng = np.random.permutation(N_CHILDREN)
    for b in range(N_BLOCKS):
        child_block = rng[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE]
        cids, gids = optimize_block(child_block, current)
        current[cids] = gids
    subm['GiftId'] = GIFT_IDS[current]          # 解码成礼物类型
    anh = my_avg_normalized_happiness(subm[['ChildId', 'GiftId']].values.tolist())
    print(i, anh)
```

材料里报告了实际结果：初始解的 ANH 是 0.933724，跑完第一轮改进后变成 0.933725——提升只有约 4.3e-07，非常小。但积累下来有效：`Improve with the Hungarian method` 这篇的标题分数是 0.9375，说明作者经过很多轮、很多小时的迭代，从 0.93 级别一点点推到了 0.9375 的排行榜分数。这告诉我们，**这类优化题的收获是线性叠加、缓慢爬升的**，没有任何一个技巧能一夜之间跳过排行榜前列。

我们还做了分布分析，统计孩子侧和礼物侧各自的幸福度分布。材料里的图显示：孩子侧的幸福度集中在较高区域（中位数大约 0.86 附近），礼物侧则有个长长的低值尾巴。作者的评论是“孩子会比圣诞老人更开心”——因为孩子侧数百万条愿望的平均，比礼物侧 1000 种礼物、每种只奖赏 1000 个最乖孩子的“稀疏奖励”更容易被满足。这也解释了为什么最终的 ANH 里，孩子侧贡献约 0.86、礼物侧只贡献约 0.075。

最后我们提交。经过若干轮局部改进后，把 `subm[['ChildId', 'GiftId']]` 直接写成 CSV，列名对齐 `ChildId, GiftId`，行数必须是 100 万。提交前我们还会自检一遍：每种礼物出现次数是否为 1000 的整数份、三胞胎/双胞胎是否拿到同一种礼物。这些检查在优化竞赛里尤其重要，因为解是被几千个块拼接出来的，任何一块写错都会让总分崩盘。

---

## 三、总结

本文中我们完成了圣诞礼物分配这道组合优化题的完整流程。我们首先做 EDA，发现礼物热度极不均衡、三胞胎和双胞胎约束把问题变复杂。其次我们把两张偏好表预计算成查表字典，写了一个独立的 ANH 计分函数作为“标尺”。接着我们介绍了两条主线：一条用 OR-Tools 最小费用流从头构造初始解，兼顾孩子和礼物两侧的幸福并把家庭约束做后处理；另一条用匈牙利方法在 400 人小区块内反复求局部最优，每改进一块分数只升不降。最后我们迭代扫描所有块，把初始解从 0.9337 级别逐步提升到 0.9375 左右的排行榜分数，并检查了礼物的份数和同礼约束。

这个比赛给我们的最大启发是：**优化问题的“模型训练”其实是“局部改进”**。全局 100 万规模的无约束最优是算不动的，但拆成小块后，每个小块都能用精确算法求出最优，靠成百上千轮的微小改进把总分一点点拱上去。另一个教训是**任何改动的价值都要用同一把尺子量**——我们那个计分函数保证了每一次改动能被立刻验证，这正是这场比赛能不断迭代的前提。