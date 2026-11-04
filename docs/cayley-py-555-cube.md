# CayleyPy 5×5×5 魔方最优求解（CayleyPy 5x5x5 Cube Solve Optimally）

## 一、比赛简介

### 1.1 比赛背景

这是一个社区性质的 Kaggle 比赛，目标是**用尽量短的步骤把 5×5×5 魔方还原**，同时推动「在巨大图上找最短路」这类算法的进步。

理解这个比赛的关键，是先理解**凯莱图（Cayley graph）**。把魔方的每一种摆法看成一个「节点」，把每一步转动看成连接两个节点的「边」：如果从摆法 A 转一步能到摆法 B，它们之间就连一条边。因为每一步都有逆操作，所以这些边是无向的。这样一来，「用最短步数还原魔方」就等价于「在这张图里从初始摆法走到还原态的最短路径」。

问题在于这张图大到离谱。材料里给了两个参照：3×3×3 魔方的状态数是 **4.33E+19**，而 5×5×5 的群大小经 Schreier-Sims 算法算出来是 **6.198E+91**。把整张图装进内存再暴力搜最短路，完全没有希望，所以我们必须用**启发式搜索**：一边往前走，一边用某种「打分器」判断哪个状态更有希望。

还有一个容易忽略的设定：这个比赛的 5×5×5 是**图案魔方（picture cube）**，中心块也有方向，150 个贴纸彼此都不相同。因此每个状态是一个真正的群元素，逆状态合法且距离对称——这一点后面的建模会反复用到。

### 1.2 比赛目标

比赛给三个文件：

- `puzzle_info.json`：`central_state` 是还原态（`[0,1,...,149]`，150 个贴纸各有一个编号），`generators` 是 30 个生成元，也就是允许的转动（f/r/d 三个轴 × 5 层 × 顺/逆时针）；
- `test.csv`：每行一个 `initial_state`，是用随机转动序列打乱出来的初始状态，用逗号分隔的 150 个整数表示；
- `sample_submission.csv`：每个初始状态配一条 `path`，用点号分隔的转动名，比如 `f0.-r1.d2`。

我们要输出的是：对每个 `initial_state_id`，给出一条**能把该状态转回还原态**的移动序列。测试集一共 **1035 个 pid**。

生成元的作用方式很简单，就是一个「下标查表」：新状态的第 i 个贴纸，等于旧状态第 `gen[i]` 个贴纸。每个生成元都有名字前面加负号的逆元（比如 `f0` 的逆元是 `-f0`）。

### 1.3 评价标准

评分规则值得仔细读：总分是**对每个 pid 取「基准路径长度」和「我们路径长度」的较小者，再全部相加**：

```
score = Σ_p min(baseline_p, ours_p)
```

其中基准路径就来自 `sample_submission.csv`——它是当初打乱时那条路径的**逆过程**，长度从 1 到 1000 不等。pids 35 到 1034 是随机游走生成的，基准长度恰好等于 `pid - 34`，所以越靠后的 pid 基准越长、越值钱；而 0 到 34 这 35 个「Santa pid」的基准平均只有 93.6 步，基本榨不出油水。

这个 `min` 的设计很关键：**它保证我们的提交永远不会比基准更差**——如果我们解不出来，就保留基准路径；如果解出来了但更长，也还是按基准算。于是我们能拿到的收益，完全等于「每条路径比基准短了多少步」。路径还有一个硬性要求：**必须真的能把状态还原**，否则是不合法的提交。

---

## 二、解题过程

我们的路线可以概括为一句话：**先用 Hamming 距离做束搜索（beam search）跑通基线，再训练一个 transformer 给每个动作打 Q 分，最后用 Bellman 递推精修打分器，在更大的束宽下求解并逐 pid 取更短的路径。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步走。

### 2.1 数据探索（EDA）

我们首先读入 `puzzle_info.json`，确认还原态和生成元，并验证「转一步再转回来能复原」这个最基本的性质：

```python
import json, numpy as np, pandas as pd
from pathlib import Path

INPUTS = Path('/kaggle/input/cayley-py-555-cube')
info = json.load(open(INPUTS / 'puzzle_info.json'))
central_state = np.array(info['central_state'])                       # [0, 1, ..., 149]
generators = {k: np.array(v) for k, v in info['generators'].items()}  # 30 个转动
names = list(generators.keys())

def apply_gen(state, name):
    # 生成元是一张「查表下标」：新状态[i] = 旧状态[gen[i]]
    return state[generators[name]]

# 先转 f0 再转逆元 -f0，应该回到还原态，说明生成元与逆元成对
s = apply_gen(central_state, 'f0')
print('转回来了吗：', np.array_equal(apply_gen(s, '-f0'), central_state))
print('生成元个数：', len(names))       # 30 = 3 个轴 × 5 层 × 顺/逆时针
```

接着我们用广度优先搜索（BFS）从还原态往外扩几层，看看状态数怎么长。这一步既验证数据没读错，也为后面「精确端局表」的规模感打底：

```python
# 用 64 位随机哈希给状态去重（150 个整数直接当 set 的键太慢）
hv = np.random.default_rng(0).integers(-2**62, 2**62, 150, dtype=np.int64)
state_hash = lambda x: (x.astype(np.int64) * hv).sum(axis=1)

P = np.array([generators[n] for n in names])       # (30, 150) 所有生成元
frontier, seen = central_state[None, :], set()
for d in range(1, 4):
    # 每个状态展开 30 个邻居，再整体去重
    children = frontier[:, P].reshape(-1, 150)
    uniq, first = np.unique(state_hash(children), return_index=True)
    fresh = np.array([u not in seen for u in uniq.tolist()])
    frontier = children[first[fresh]]
    seen.update(uniq[fresh].tolist())
    print(f'深度 {d}: {len(frontier):,} 个新状态')
```

我们还算了一下：**30 个生成元全是奇置换**，也就是每转一步，状态的奇偶性必然翻转。这意味着凯莱图是**二部图**：一个子节点要么比父节点近一步，要么远比一步，绝不可能同距离。这个性质后来成了训练标签的理论依据——在随机路径的枢轴处，「退回一步」的距离恰好比「前进一步」少 2。

### 2.2 数据清洗

这份数据没有缺失值，所谓的「清洗」主要是**把文本还原成能计算的数组，并保证每条路径合法**。

第一，状态必须读成 `int16` 而不是 `int8`：贴纸编号是 0~149，超出了 int8 的范围，读成 int8 会变成负数，后面查嵌入表就全乱了。

第二，路径要先在原始状态上重放一遍，只有真正回到 `central_state` 才算数。这一步既是调试，也是提交前的最后一道保险：

```python
def parse_state(s):
    # 逗号分隔的字符串 -> 一维数组；必须用 int16，0~149 超出 int8
    return np.array([int(x) for x in s.split(',')], dtype=np.int16)

def parse_path(s):
    return s.split('.')          # 点号分隔的转动名

def check_path(path, state):
    # 依序施加每个转动，最后必须精确等于还原态
    for mv in path:
        state = state[generators[mv]]
    return np.array_equal(state, central_state)

test = pd.read_csv(INPUTS / 'test.csv', index_col='initial_state_id')
sub  = pd.read_csv(INPUTS / 'sample_submission.csv', index_col='initial_state_id')

pid = 40
ok = check_path(parse_path(sub.loc[pid, 'path']),
                parse_state(test.loc[pid, 'initial_state']))
print('sample 路径合法：', ok)
```

第三，我们把 150 个贴纸「压缩」成 98 个棋子 token：8 个角块（各 3 贴纸）、24 个翼棱和 12 个中棱（各 2 贴纸）、两组各 24 个中心、6 个固定中心。图案魔方上，**棋子某个面的贴纸编号就能唯一确定这是哪个块、朝向如何**，所以这个压缩是无损的。为了确认这一点，我们抽了 2000 条长度 60 的随机打乱，检查「不同状态是否得到不同的 98-token 编码」——结果是一个状态对应一个编码，没有发生碰撞。

### 2.3 模型训练与超参数调优

**第一步：Hamming 束搜索基线。** 束搜索的思路是「同时保留 W 条最有希望的路径」。每一轮把所有候选路径的末状态各展开 30 个邻居，用打分器给邻居打分，只留下得分最好的 W 个。材料里的入门 notebook 用一个很朴素的打分器——Hamming 距离，也就是「当前状态和还原态有多少个贴纸位置不同」。它便宜、能跑，但对这个任务来说太粗糙，只能当基线：

```python
from cayleypy import CayleyGraphDef, CayleyGraph, Predictor

# 用 puzzle_info 里的生成元从零构造凯莱图
graph = CayleyGraph(CayleyGraphDef.create(
    generators=[generators[n] for n in names],
    generator_names=names, central_state=central_state))

# 束搜索：beam_width=1000 表示同时保留 1000 条候选路径
res = graph.beam_search(
    start_state=initial_state,
    beam_width=1000, max_steps=100,
    predictor=Predictor(graph, 'hamming'),   # 用「错位数」当好坏分数
    beam_mode='simple', return_path=True)
print('找到路径：', res.path_found, '长度：', len(res.path))
```

Hamming 基线在小状态上能跑通，甚至比 sample 路径更短，但一放到深处的状态就不行了。原因也很直白：Hamming 只数「位置不对的贴纸」，完全不懂魔方的结构，走一步可能让错位数暂时变多、却整体更接近还原。

**第二步：训练一个 Q 打分器。** 我们换成一个 **PieceTransformer**：输入 98 个棋子 token 加 1 个 CLS token，经过 6 层 pre-norm transformer（`d_model=384`、6 个注意力头、FFN 1536、SiLU 激活），由 CLS 输出两个头——一个 **30 维的 Q 头**和一个 1 维的价值头，总参数 **10.77M**。

为什么是 30 维 Q 头而不是 1 维 V 头？因为 V 模型必须把 30 个子节点都展开、各前向一次才能打分，一个束步要 `B×30` 次前向；而 Q 头在**父节点上做一次前向**，就同时给出了全部 30 个动作的分数，束步只要 `B` 次前向。**这 30 倍的差距，正是让两百万宽度的束搜索能在一块 GPU 上跑起来的原因。** `Q(s,a)` 的含义是「从 s 走动作 a 之后，还需要多少步还原」，束搜索保留 Q 最小的子节点。

训练标签只有三种，刻意保持精简：

1. **随机游走给的稀疏 Q 标签**。从还原态出发走一条不回头的随机路径，长度 k 在 2 到 80 之间均匀抽取；在路径上选一个枢轴位置 p，此时只有两个动作有可靠标签：退回一步是 `Q=p-1`，前进一步是 `Q=p+1`。损失只在这两列上算掩码 MSE：

```python
from cube555.qtrain import SparseQSampler
import torch

# 生成元表 P 是 (30,150)；INV 记录每个动作的逆元在列表中的下标
names = list(generators.keys())
INV = np.array([names.index('-' + n if not n.startswith('-') else n[1:])
                for n in names])
perms_t = torch.as_tensor(P)
inv_t   = torch.as_tensor(INV)
solved_t = torch.arange(150, dtype=torch.uint8)

# k ~ U[2,80]，pivot_tilt=0.5 让枢轴更偏向路径深处
sampler = SparseQSampler(perms_t, inv_t, solved_t, 2, 80, 0.5,
                         torch.Generator().manual_seed(5551))
piv_s, piv_p, undo, nxt = sampler.sample(20000)
# 枢轴处只有两列有标签：退回 = p-1，前进 = p+1
print('平均枢轴深度：', float(piv_p.float().mean()))
```

这两个标签都是**真实距离的上界**：在根部（k 很小）几乎精确，越深越松。正因为如此，模型学到的不是「绝对距离」，而是**同一个状态内部 30 个动作的相对排序**——这恰好是束搜索唯一需要的东西。

2. **精确 BFS 锚点**。我们把还原态周围距离 ≤5 的全部 **10,739,017 个状态**，连同 30 个动作的精确 Q 值都算出来，每个 batch 里放 512 个、权重 1.0。锚点最重要的作用是**压住数值的绝对水平**——没有它，Bellman 自举会让 `V(还原态)` 漂到 2 附近，整个打分器就废了。

3. **对称性增广**。魔方有 48 个对称变换（24 个旋转 × 反射），在图案魔方上通过共轭作用到状态上，标签也跟着换一列。每个枢轴复制成 4 行：恒等变换加 3 个随机对称帧。用「随机」而不是贪心覆盖，是为了让同一对 (undo, next) 在整个训练过程中看到不同的变换，增加状态多样性。

一个训练步就是两组固定大小的行：`512 个枢轴 × 4 帧 = 2048 行` 做稀疏 Q，再加 `512 个锚点` 做全 30 列 MSE，总损失是

```
loss = sparse_mse + 1.0 × anchor_mse + 1.0 × value_loss
```

优化器是 AdamW，学习率恒定 `3e-4`，权重衰减 `3e-3`，梯度裁剪 1.0，bf16 混合精度。全程 **6000 个 epoch、31 亿个新鲜状态、约 285 A100 小时**。

**第三步：Bellman 精修。** 这是整个项目单项收益最大的一步。稀疏 Q 标签只是上界，深了以后误差很大；Bellman 递推能让 Q 值和模型自己的子节点**自洽**：

```python
# y(s,a) = 1 + min_a' Q_target(s∘a, a')；若子状态已还原，则 y = 0
children = torch.gather(bs[:, None, :].expand(-1, 30, -1), 2,
                        perms_t[None].expand(len(bs), -1, -1))   # (B,30,150)
child_solved = (children == torch.arange(150)).all(2)            # 子状态是否已还原

qp = net(bs)                                                     # 父状态 30 个 Q
qc = net(children.reshape(-1, 150)).reshape(len(bs), 30, 30)     # 所有子状态的 30 个 Q
y  = torch.where(child_solved, torch.zeros_like(qp), 1.0 + qc.min(2).values)

loss = ((qp - y) ** 2).mean()     # 只优化这个 MSE；lr=2e-5，只跑 2000 步
```

精修从训练好的 `epoch_005999.pt` **热启动**，2000 步、学习率 `2e-5`、无调度器，目标网络每 500 步刷新一次，batch 是 768 个随机游走状态加 256 个深度 5 锚点（权重 2.0），在一块 A100 上只花 **22.6 分钟**。热启动是关键：如果从随机初始化开始做同样的递推，会继承目标网络在深处的「平坦」，什么都传不出去——材料里在 4×4×4 上就失败过。

**调参时我们踩过的几个坑，比参数本身更值钱：**

- **更宽的束不一定更好。** 同一个打分器下，宽度 2^20 只解出 2/6，2^21 解出 6/6，2^22 掉到 3/6，2^23 是 0/1。原因是更宽的束会放进打分器**排不出优劣**的候选，它们反而挤掉了好候选。所以默认宽度就定在 2^21。
- **离线指标不能当早停标准。** 离线 top-1、pair 准确率涨了 6000 个 epoch，束搜索却可能原地踏步。真正该看的是「束搜索解出来的步数」。
- **Bellman 不是越多越好。** 2000 步最好；4000 步在难 pid 上倒退回 +19.33；10000 步甚至连其他模型都能解的 pid 都解不出。
- **训练预算要按「更新次数 × 新鲜状态数」来定。** 3000 epoch 的版本更新次数追平了对手，但新鲜状态少了 3.9 倍；翻倍到 6000 epoch 后，6 个 pid 上平均缩短了 13.33 步。

### 2.4 模型测试与指标报告

推理时的束搜索配置是：宽度 2^21、最大 300 步、历史深度 4、不允许回头、**端局深度 5**、测试时不做对称帧（因为训练时已经用随机帧代替了）。

端局（endgame）这一步很巧妙：目标不是「正好走到还原态」那一个状态，而是「走进距离 ≤5 的精确球」——**10,739,017 个状态**。一旦进入这个球，就直接从预计算的表里拼出最优下降路径。设备上判断是否入球用 64 位 Zobrist 哈希，然后在主机侧再按真实状态复核一遍；万一哈希误判，就丢掉这一帧而不是整次运行。

每条解出来的路径都必须**在原始测试状态上重放验证**，通过之后才参与提交。提交时逐 pid 取「我们」和「基准」中较短的那条，天然保证了不低于基准：

```python
def replay_ok(pid, path):
    # 把路径在原始状态上重放，只有真正回到还原态才承认
    st = test.loc[pid, 'initial_state'].copy()
    for mv in path:
        st = st[P[names.index(mv)]]
    return np.array_equal(st, central_state)

rows = []
for pid in all_pids:
    ours = best_paths.get(pid)                 # 我们解出的路径（可能没有）
    base = sample_paths[pid]                   # sample_submission 的基准路径
    if ours and replay_ok(pid, ours) and len(ours) < len(base):
        rows.append((pid, '.'.join(ours)))     # 更短且合法，替换
    else:
        rows.append((pid, '.'.join(base)))     # 否则保留基准，不亏分

pd.DataFrame(rows, columns=['initial_state_id', 'path']).to_csv('submission.csv', index=False)
```

效果上，我们在一组固定的 12 个 pid 上做了对照（前 6 个是容易的，后 6 个是旧方案解不出的）：

| 方案 | 容易 6 个的平均步数 | 难题 6 个 |
| --- | --- | --- |
| 旧方案 arm A（ResMLP + Bellman） | 120.83 | 0/6 解出 |
| 本模型 ep5999 | 133.83 | 6/6，平均 127.50 |
| 本模型 + Bellman-2k | **110.50** | **6/6，平均 108.17** |

Bellman 精修相对它自己的父模型是 **12 胜 0 负**：容易组缩短 23.33 步，难题组缩短 19.33 步。值得一提的是，旧方案 arm A 速度快得多（每个 pid 约 2.25 分钟，而本模型约 63 分钟，差了约 28 倍），所以它的价值在于**批量扫描全部 1035 个 pid**；而 transformer 的价值在于**能解出 arm A 根本解不出的状态**。

最后我们做了两件事：一是把六个不同 Bellman 长度的检查点当成六条独立的「臂」，各自跑一遍再逐 pid 取最短——**单臂最好只解出 18/24，但取并集能解出 24/24**，说明它们互补；二是在 2×T4 上把 transformer 和 ResMLP 的分数按 0.8 / 0.2 融合进同一个束搜索，让两种打分器在每一步共同排序。所有对比都在同一批 pid 上做符号检验，避免被个别异常值带偏。

---

## 三、总结

本文中我们完成了 CayleyPy 5×5×5 魔方最优求解的完整流程。我们首先用 EDA 摸清了问题：150 个贴纸构成的图案魔方、30 个生成元、群大小 6.198E+91，凯莱图是二部图且每个生成元都是奇置换。其次我们把状态读成 int16、把 150 个贴纸无损压缩成 98 个棋子 token，并写好了「路径必须在原状态上重放成功」的校验。接着我们先用 Hamming 距离的束搜索跑通基线，再训练一个 10.77M 参数的 PieceTransformer 输出 30 维 Q 值，用随机游走枢轴标签、10,739,017 个距离 ≤5 的精确锚点和 48 个对称帧训练 6000 个 epoch，最后用 2000 步 Bellman 递推精修。最后我们在 2^21 的束宽下求解，配合距离 ≤5 的精确端局拼接，逐 pid 与基准取较短者提交，在 12 个对照 pid 上把难题组从 0/6 提升到 6/6、平均 108.17 步。

这个比赛给我们的最大启发是：**在启发式搜索里，「排序」比「估准」更重要**。Q 值在深处理论上就不准（标签只是上界），但束搜索只关心同一个状态下 30 个动作谁排前面；也正因如此，判断模型好坏不能看离线损失，而要看束搜索真正解出来的步数。另一个启发是**搜索宽度不是万能钥匙**——当打分器分辨不出候选优劣时，加宽只会让噪音挤掉好答案。