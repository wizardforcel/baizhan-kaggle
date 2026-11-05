# Santa Gift Matching（圣诞礼物匹配）

## 一、比赛简介

### 1.1 比赛背景

这场比赛把圣诞老人送礼物抽象成一个大规模的优化问题。我们手里有一百万名孩子和一千种礼物。每种礼物有一千份，因此礼物总数正好也是一百万。每个孩子会列出自己最想要的礼物，Santa 也会对每种礼物列出一份「喜欢收到这件礼物的孩子」名单。我们的任务是在礼物数量有限的前提下，把礼物分给孩子，让孩子开心，也让 Santa 觉得这次分配合理。

数据文件主要是 `child_wishlist.csv` 和 `gift_goodkids.csv`。前者记录每个孩子的编号和心愿顺序，后者记录每个礼物偏好的孩子顺序。数据没有传统机器学习里的训练标签：一份礼物分配方案本身就是答案。我们要做的是搜索一个更好的方案。

比赛还包含双胞胎约束。材料中把双胞胎数量写成孩子总数的 0.004，也就是约 4000 个孩子；双胞胎必须收到同一种礼物。另一个版本的输入还出现了三胞胎，因此读取数据时必须以文件格式为准，而不能只凭名字猜列数。除此之外，绝大多数孩子都是普通的单人样本。

### 1.2 比赛目标

我们需要提交一个 CSV 文件，每行包含 `ChildId` 和 `GiftId`。`GiftId` 必须在 0 到 999 之间，并且每个礼物最多使用 1000 次。双胞胎（以及数据版本中存在的三胞胎）要共享礼物。对每个孩子来说，送到心愿表前面的礼物会带来更高的孩子幸福度；对 Santa 来说，礼物偏好名单里排名更靠前的孩子也会带来更高的 Santa 幸福度。

这个目标和普通分类不同。我们不是预测一个固定的 `target`，而是在一百万个孩子之间进行全局分配。给某个孩子最喜欢的礼物，可能会抢走另一个孩子的唯一心愿，所以每一次选择都要同时考虑全局容量。

### 1.3 评价标准

比赛使用 **Average Normalized Happiness（ANH，平均归一化幸福度）**。代码把孩子幸福度和 Santa 幸福度分别累加，再除以孩子数量和礼物种类数量，最后把两者相加。两个部分都被归一化到可比较的尺度，分数越高越好。

直观地说，孩子第一志愿对应高分，后面的志愿逐渐降低；如果一个孩子收到不在自己心愿表中的礼物，孩子幸福度接近最低。Santa 的部分也类似：某个孩子在礼物偏好名单中排名高，Santa 的幸福度就高。最终分数必须同时照顾两方，不能只让孩子开心而完全牺牲 Santa。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取两张心愿表，并确认它们的规模、编号方式和列数。材料中的原始版本每个孩子有 10 个心愿，每种礼物有 1000 个偏好孩子；v2 版本把孩子心愿扩展到了 100 列，所以我们先根据文件实际形状确定参数。

```python
import numpy as np
import pandas as pd

# 不设表头，因为第 0 列通常是编号，后面才是心愿顺序
child_pref = pd.read_csv(
    "../input/santa-gift-matching/child_wishlist.csv",
    header=None
).values

gift_pref = pd.read_csv(
    "../input/santa-gift-matching/gift_goodkids.csv",
    header=None
).values

# 用数据形状而不是手写列数，避免 v1 和 v2 格式不一致
n_children = child_pref.shape[0]
n_gifts = gift_pref.shape[0]
n_child_pref = child_pref.shape[1] - 1
n_gift_pref = gift_pref.shape[1] - 1
print(n_children, n_gifts, n_child_pref, n_gift_pref)
```

探索结果告诉我们，孩子数量是 1000000，礼物种类是 1000。每种礼物的数量固定为 1000，因此是一个严格的容量约束问题。我们还把前 4000 个孩子单独标记为双胞胎，检查同一对双胞胎的心愿是否相近。双胞胎的约束不能在最后随便修补，否则可能超过某种礼物的库存。

我们接着把心愿排名转换成幸福度。孩子在心愿表中排第 0 名时分数最高，排得越后分数越低；不在心愿表中的礼物使用一个较低的默认值。Santa 的偏好表也做同样的转换。

```python
from collections import defaultdict

# 默认值代表礼物不在心愿表中，避免字典查不到时报错
child_happiness = {}
for c in range(n_children):
    child_happiness[c] = defaultdict(lambda: 1.0 / (2 * n_child_pref))
    for rank, gift in enumerate(child_pref[c, 1:]):
        # 排名越靠前，负的成本越小，换算后的幸福度越高
        child_happiness[c][int(gift)] = -float(n_child_pref - rank) / n_child_pref

gift_happiness = {}
for g in range(n_gifts):
    gift_happiness[g] = defaultdict(lambda: 1.0 / (2 * n_gift_pref))
    for rank, child in enumerate(gift_pref[g, 1:]):
        gift_happiness[g][int(child)] = -float(n_gift_pref - rank) / n_gift_pref
```

这里使用负数保存「成本」是为了方便最小费用算法：越喜欢的组合，成本越低；越不喜欢的组合，成本越高。这样我们后面调用 Hungarian method 或最小费用流时，就可以直接寻找总成本最小的分配。

### 2.2 数据清洗

我们其次处理编号和库存。提交文件中的 `GiftId` 是礼物类型编号，而同一种礼物有 1000 份。为了进行局部交换，我们把每一份实际礼物展开成唯一的 `gift_id`，并用整除和取模检查每种类型是否恰好使用 1000 次。

```python
N_CHILDREN = 1000000
N_GIFT_TYPE = 1000
N_GIFT_QUANTITY = 1000

# 礼物类型 0 重复 1000 次，接着是类型 1，以此类推
gift_ids = np.array(
    [[g] * N_GIFT_QUANTITY for g in range(N_GIFT_TYPE)]
).flatten()

# 读入已有方案，保留 ChildId 和 GiftId 两列
submission = pd.read_csv("../input/c-submission/cpp_sub.csv")

# 按礼物类型内部排序，把同类型的第几份也编码出来
submission["gift_rank"] = submission.groupby("GiftId").rank() - 1
submission["gift_id"] = (
    submission["GiftId"] * N_GIFT_QUANTITY
    + submission["gift_rank"]
).astype(np.int64)

# 这个检查能提前发现库存超额或缺货
print(submission["GiftId"].value_counts().min())
print(submission["GiftId"].value_counts().max())
```

数据清洗的重点不是填补空值，而是保证约束合法。我们不删除任何孩子，因为每个孩子都必须获得一个礼物；也不把双胞胎拆开，因为拆开后即使 ANH 看起来更高，也不符合比赛规则。对于 v2 的三胞胎，我们使用同样的思路，把一个家庭当成一个整体处理。

我们还写了一个快速评分函数，先检查现有方案的质量，再决定优化有没有真的带来提升：

```python
def average_normalized_happiness(pred):
    # 分别累计孩子侧和 Santa 侧的幸福度
    child_total = 0.0
    gift_total = np.zeros(N_GIFT_TYPE)
    for child_id, gift_id in pred:
        child_total += -child_happiness[int(child_id)][int(gift_id)]
        gift_total[int(gift_id)] += -gift_happiness[int(gift_id)][int(child_id)]

    child_score = child_total / N_CHILDREN
    santa_score = np.mean(gift_total) / N_GIFT_TYPE
    return child_score + santa_score
```

### 2.3 模型训练与超参数调优

这一步严格来说不是训练模型，而是做组合优化。我们先采用已有的贪心或最小费用流方案作为 baseline。baseline 的 ANH 约为 0.933724239501，其中孩子侧约为 0.85920825，Santa 侧约为 0.0745159895。这个结果已经很高，但仍然存在局部交换空间。

我们接着使用 Hungarian method。它是解决「一对一分配」问题的经典算法：给定一个成本矩阵，算法会寻找总成本最小的匹配。全量矩阵太大，直接对一百万个孩子做 O(n³) 计算不可行，因此我们把普通孩子随机打乱后切成 400 个一块，只在小块内部重新匹配。

```python
from scipy.optimize import linear_sum_assignment

BLOCK_SIZE = 400

def optimize_block(child_block, current_gift_ids):
    # 当前块里每个孩子已经有一个礼物，先拿出这些候选礼物
    gift_block = current_gift_ids[child_block]
    cost = np.zeros((BLOCK_SIZE, BLOCK_SIZE))

    for i, child_id in enumerate(child_block):
        for j, gift_index in enumerate(gift_block):
            gift_id = gift_ids[gift_index]
            # 目标是让孩子和 Santa 的总成本同时变小
            cost[i, j] = (
                child_happiness[int(child_id)][int(gift_id)]
                + gift_happiness[int(gift_id)][int(child_id)]
            )

    row_ind, col_ind = linear_sum_assignment(cost)
    # 返回重新匹配后的孩子编号和实际礼物索引
    return child_block[row_ind], gift_block[col_ind]
```

`BLOCK_SIZE` 是一个重要的计算参数。块太小，能够交换的候选太少，提升有限；块太大，成本矩阵占用内存和运行时间都会迅速增加。材料中的实验使用 400，500 个块大约需要 350 秒完成一轮，单轮分数从 0.933724239501 提升到 0.933724668001。作者继续运行 8 小时，累计提升约 0.0005，最终方案约为 0.9375。

我们也尝试用局部贪心更新：每次挑出孩子当前礼物和候选礼物，计算交换前后的 ANH，如果变好就保留。它比全量 Hungarian 更容易运行，但容易陷入局部最优，所以更适合作为初始化或后处理。调参时主要观察块大小、随机种子、迭代次数以及是否跳过双胞胎，而不是学习率之类的机器学习参数。

### 2.4 模型测试与指标报告

我们最后把优化前后的方案都用同一个评分函数计算，避免因为评分代码不同造成假提升：

```python
# 读取 baseline 的 ChildId/GiftId 配对
initial = submission[["ChildId", "GiftId"]].values.tolist()
initial_score = average_normalized_happiness(initial)
print("initial ANH:", initial_score)

# 随机排列普通孩子，双胞胎单独保留，避免违反家庭约束
ordinary = np.random.permutation(range(4000, N_CHILDREN))
current = submission["gift_id"].values.copy()

# 先只跑前 500 个块，快速验证方向是否正确
for child_block in np.array_split(ordinary, 500):
    if len(child_block) != BLOCK_SIZE:
        continue
    ids, gifts = optimize_block(child_block, current)
    current[ids] = gifts

submission["GiftId"] = gift_ids[current]
new_score = average_normalized_happiness(
    submission[["ChildId", "GiftId"]].values.tolist()
)
print("new ANH:", new_score)
print("improvement:", new_score - initial_score)
```

材料中的单轮实验结果是：孩子幸福度约 0.8592102，Santa 幸福度约 0.074514468，总 ANH 为 0.933724668001，比 baseline 提升 4.285e-7。这个提升看起来很小，但全量有一百万个孩子，哪怕只改善很少一部分配对，也足以改变排行榜名次。继续运行更多块、重复多轮局部重排后，公开记录里出现了约 0.9375 的分数。

我们还画了孩子幸福度和 Santa 幸福度的直方图：

```python
child_scores = np.zeros(N_CHILDREN)
gift_scores = np.zeros(N_CHILDREN)
for child_id, gift_id in submission[["ChildId", "GiftId"]].values:
    child_scores[int(child_id)] = -child_happiness[int(child_id)][int(gift_id)]
    gift_scores[int(child_id)] = -gift_happiness[int(gift_id)][int(child_id)]

# 两个分布放在一起，检查是否一方被过度牺牲
import matplotlib.pyplot as plt
plt.hist(child_scores, bins=20, alpha=0.5, label="child")
plt.hist(gift_scores, bins=20, alpha=0.5, label="Santa")
plt.legend()
plt.title("Child and Santa happiness")
plt.show()
```

最后我们重新保存提交文件，并检查每个礼物的使用次数以及双胞胎是否相同：

```python
# 最终提交只保留比赛需要的两列
final_submission = submission[["ChildId", "GiftId"]].copy()
final_submission.to_csv("submission.csv", index=False)

# 每种礼物都应该正好使用 1000 次
counts = final_submission["GiftId"].value_counts()
assert counts.min() == 1000 and counts.max() == 1000

# 文件行数必须等于孩子数量
assert len(final_submission) == N_CHILDREN
print("submission.csv ready")
```

## 三、总结

本文中我们完成了 Santa Gift Matching 的完整优化流程。我们首先读取孩子心愿表和 Santa 偏好表，确认了一百万名孩子、一千种礼物以及每种礼物一千份的库存结构。其次我们把心愿排名转换成孩子幸福度和 Santa 幸福度，并把双胞胎作为必须同时分配的家庭处理。接着我们以贪心或最小费用方案作为 baseline，再把普通孩子分成 400 人的小块，用 Hungarian method 在块内重新匹配。最后我们用统一的 ANH 函数比较优化前后结果：baseline 约为 0.933724239501，单轮局部优化得到约 0.933724668001，持续迭代的公开方案达到约 0.9375。我们验证了库存约束和家庭约束均被严格满足，提交文件可以直接用于评测。

这个比赛的启发是，很多 Kaggle 题目不一定要训练分类器或回归器。只要目标是给大量对象分配有限资源，就应该先判断它是不是组合优化问题。全局一次求解可能太大，但「先构造可行解，再做小块局部优化」能把不可计算的问题变成可以逐步改进的方案。评价函数、库存约束和家庭约束必须贯穿整个过程，不能只追求局部幸福度。