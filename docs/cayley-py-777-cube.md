# CayleyPy 7×7×7 魔方最优求解（CayleyPy 7x7x7 Cube Solve Optimally）

## 一、比赛简介

### 1.1 比赛背景

这个比赛要求我们求解 **7×7×7 魔方**：给一个被打乱的状态，输出一条尽量短的还原步骤。它属于 CayleyPy 系列社区比赛，目的不只是「解魔方」，而是推动**大规模图上的路径搜索**方法。

理解这个问题最好的切入点是**凯莱图（Cayley graph）**。想象把魔方的每一个状态画成一个点；如果从状态 A 转动一步能变成状态 B，就在 A、B 之间连一条线。因为每一种转动都有逆转动，线是双向的。「把魔方还原」于是变成「从起点走到还原态那个点的最短路」。

麻烦在于这张图实在太大了。材料里给了两个直观的比较：3×3×3 魔方就有大约 **4.33E+19** 个状态，而一个 master tetraminx 大约有 **1.54E+32** 个状态。7×7×7 每面有 7×7 个色块，层数更多、可转的层也更多，图的规模只会比这些例子更夸张。把整张图存在内存里再跑最短路，根本没有可能，所以我们只能用**启发式搜索**：不追求一次看全，而是一边走一边估计「哪个方向更有希望」。

### 1.2 比赛目标

比赛要求我们为每个测试状态输出一条**转动序列**。相关文件有三个：

- `puzzle_info.json`：包含 `central_state`（还原态）和 `generators`（所有允许的动作）。每个动作都有一个逆动作，名字是原动作前面加负号，例如 `f0` 与 `-f0`；
- `test.csv`：每行一个 `initial_state`，是组委会用随机动作序列把还原态打乱得到的；
- `sample_submission.csv`：每个 `initial_state_id` 对应一条示例 `path`，动作之间用点号分隔。

动作的施加方式很直接：`generators[name]` 是一张下标表，`state[generators[name]]` 就是执行这个动作之后的新状态。我们要交的，是每个 `initial_state_id` 一行，填上我们自己找到的路径。

### 1.3 评价标准

第一条硬规则：**路径必须能把该状态转回 `central_state`**，转不回去就是无效提交。

第二条是「越短越好」。`sample_submission.csv` 的路径不是随便写的，它是**打乱过程的逆序**——也就是说，它天然合法，长度从 1 到 1000 不等。于是我们可以把它当成一条基准线：如果我们搜出来的路径更短，就替换；如果搜不到更短的，就保留原样。这样既能拿到「缩短」的收益，又不会因为瞎改而把本来合法的提交弄坏。

---

## 二、解题过程

我们的方案是：**先把 `puzzle_info.json` 读通、把路径解析与校验写对，然后用 CayleyPy 的束搜索（beam search）配一个基于 Hamming 距离的打分器，对测试集里的一部分状态尝试缩短路径。** 这个方案算力需求小，适合先把整条流水线跑通。我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步走。

### 2.1 数据探索（EDA）

我们首先加载 `puzzle_info.json`，看清还原态长什么样、动作有哪些，并做一个「转出去再转回来」的自检：

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd

INPUTS_DIR = Path('/kaggle/input/cayley-py-777-cube')
with open(INPUTS_DIR / 'puzzle_info.json') as f:
    puzzle_info = json.load(f)

central_state = np.array(puzzle_info['central_state'])          # 已还原的状态
generators = {k: np.array(v) for k, v in puzzle_info['generators'].items()}
print('动作列表：', list(generators.keys()))                    # 每个动作都有带 '-' 的逆元

def apply_gen(state, gen_name):
    # 执行一个动作：按动作对应的下标表重新排列贴纸
    return state[generators[gen_name]]

# 转一步 f0，再转 -f0，应当精确回到还原态
s = apply_gen(central_state, 'f0')
print('可逆：', np.all(apply_gen(s, '-f0') == central_state))
```

然后我们读测试集和示例提交，确认列名和规模：

```python
df_test = pd.read_csv(INPUTS_DIR / 'test.csv', index_col='initial_state_id')
df_sample = pd.read_csv(INPUTS_DIR / 'sample_submission.csv',
                        index_col='initial_state_id')

print(df_test.head())     # initial_state：逗号分隔的整数序列
print(df_sample.head())   # path：点号分隔的动作名
print('测试状态数：', len(df_test))
```

从示例提交里我们注意到两点。第一，路径长度分布跨度极大，最浅的只有 1 步，最深的接近 1000 步；第二，浅状态的路径通常就是打乱时用的那几个动作的逆序，所以看起来「短而整齐」。这意味着**不同状态的难度差了几个数量级**，我们不能用同一个搜索预算一视同仁，得先把预算花在那些「深且基准长」的状态上。

还有一个只靠尺寸就能推出来的结论：7×7×7 的层数多，每一步可选的动作也就多，搜索树的分支比小尺寸魔方更宽。同样宽度的束，在 7×7×7 上只能覆盖整个搜索空间里更小的一块。所以尺寸越大，「打分器准不准」就越关键，光靠加大束宽是追不回来的。

### 2.2 数据清洗

这份数据没有缺失值和重复行，需要清洗的地方在**解析和校验**：把 CSV 里的字符串变成数组，把路径变成动作列表，并且保证「我们认定的合法」和「评测方认定的合法」是一回事。

我们先写解析函数和校验函数：

```python
def parse_initial_state(s: str) -> np.ndarray:
    # "0,1,2,..." -> numpy 整数数组
    return np.array([int(x) for x in s.split(',')])

def parse_path(s: str) -> list:
    # "f0.-r1.d2" -> ['f0', '-r1', 'd2']
    return list(s.split('.'))

def check_path(path, initial_state) -> bool:
    # 从初始状态出发，逐个施加动作，最后必须逐元素等于还原态
    state = initial_state
    for move in path:
        state = state[generators[move]]
    return bool(np.all(state == central_state))
```

接着用一条示例路径检验我们的解析是否正确：

```python
pid = 6
initial_state = parse_initial_state(df_test.loc[pid, 'initial_state'])
path = parse_path(df_sample.loc[pid, 'path'])

print('示例路径长度：', len(path))
print('示例路径合法：', check_path(path, initial_state))
```

这一步看起来简单，但它是整个流程的地基。后面束搜索给出的每一条路径，我们都会先跑 `check_path` 再决定要不要用；提交文件写完之后，还会把所有路径整体重放一遍。原因很实在：**在「错一步就全错」的任务里，一条不合法的路径不会得低分，而是直接作废**。

### 2.3 模型训练与超参数调优

这一节我们「训练」的对象是搜索算法。思路分三步：把问题建成图、定义打分器、调搜索预算。

**第一步，建图。** CayleyPy 可以直接用 `central_state` 和 `generators` 构造凯莱图：

```python
from cayleypy import CayleyGraphDef, CayleyGraph, Predictor

gen_names = list(generators.keys())
# 用还原态和所有动作构造凯莱图：节点是状态，边是一次转动
graph = CayleyGraph(CayleyGraphDef.create(
    generators=[generators[n] for n in gen_names],
    generator_names=gen_names,
    central_state=central_state))
```

**第二步，理解束搜索。** 束搜索可以想象成「同时养一窝兔子」：我们先放出一只兔子（起点），每一轮让每只兔子往所有可能的方向各走一步，然后用打分器评价「新位置离终点有多近」，只留下最好的 W 只继续走。W 叫**束宽**；最多允许走 L 轮，叫**最大步数**。如果某一轮有兔子正好到达还原态，就把它走过的路作为答案；如果走满 L 轮还没有，就宣告失败。

它比广度优先搜索省内存（不用保存整层所有状态），又比贪心搜索稳（不会因为一步走岔就全盘皆输）。代价是：**打分器不准的时候，好路径可能在半路就被剪掉了**。

**第三步，选一个打分器。** 材料里的入门方案用 **Hamming 距离**，也就是「当前状态和还原态有多少个贴纸位置对不上」。它的好处是实现简单、计算快；坏处是它只数位置不看结构，所以经常把「暂时变差、但整体更接近」的动作误判为坏动作：

```python
pid = 6
initial_state = parse_initial_state(df_test.loc[pid, 'initial_state'])
sample_path = parse_path(df_sample.loc[pid, 'path'])

# 束宽 1000、最多 50 步，用 Hamming 距离作为「好坏」打分
result = graph.beam_search(
    start_state=initial_state,
    beam_width=1000,
    max_steps=50,
    predictor=Predictor(graph, 'hamming'),
    beam_mode='simple',
    return_path=True)

print('是否成功：', result.path_found)
print('束搜索长度：', len(result.path), ' 示例长度：', len(sample_path))
print('结果合法：', check_path(result.path, initial_state))
```

材料里的结果是：这一步能成功，而且找到的路径比示例更短。原因不难理解——示例路径是「打乱的逆序」，虽然合法，但不保证最短；束搜索只要方向大致对，就有机会把它压缩。

**调参。** 我们主要动两个旋钮：

- **束宽 W**：从 1000 往上加，能明显提高成功率，但每一步要展开和打分的状态也按比例增加，显存和时间都会吃紧；
- **最大步数 L**：浅状态用 50 步足够，深状态得放到 100 步甚至更多，否则搜索会在半路「没耐心」而失败。

材料的态度很务实：先用小预算跑通流程、拿到一个能交的结果，再把预算集中到最值钱的状态上。至于 Hamming 打分器的先天不足，靠调 W、L 只能缓解——要根治，必须换成学习出来的距离预测模型（这也是这个系列比赛的主流做法）。

### 2.4 模型测试与指标报告

接下来我们对一小批状态批量搜索。规则是：**只有束搜索成功、并且比示例路径更短，才采用新路径；否则原样保留**。这样能保证结果不会比基准差：

```python
beam_width, max_steps, first_n = 1000, 100, 10
final_paths = []

# 取第 5 到第 14 条，跳过最前面几条一步就能解的简单样本
for pid, row in df_test[5:5 + first_n].iterrows():
    sample_path = parse_path(df_sample.loc[pid, 'path'])
    state = parse_initial_state(row['initial_state'])

    result = graph.beam_search(
        start_state=state, beam_width=beam_width, max_steps=max_steps,
        predictor=Predictor(graph, 'hamming'),
        beam_mode='simple', return_path=True)

    len_sample = len(sample_path)
    len_beam = len(result.path) if result.path_found else np.inf

    # 更短才替换；失败或没更短，就保留示例路径
    if len_beam < len_sample and check_path(result.path, state):
        final_paths.append('.'.join(result.path))
    else:
        final_paths.append('.'.join(sample_path))

    print(f'{pid:>3} 成功={result.path_found} 束={len_beam} 示例={len_sample}')
```

跑完我们会看到三种情况：有些状态被明显缩短，有些状态搜索失败，还有些状态虽然解出来了但没有比示例更短。这三种情况都很正常，而且**失败并不可怕**——因为示例路径始终是兜底。

最后把三段拼起来（前面原样、中间替换、后面原样），写提交文件，并做一次全量合法性检查：

```python
new_paths = (list(df_sample[:5]['path']) + final_paths
             + list(df_sample[5 + first_n:]['path']))

df_submission = df_sample.copy()
df_submission['path'] = new_paths

bad = []
for pid, p in df_submission['path'].items():
    st = parse_initial_state(df_test.loc[pid, 'initial_state'])
    if not check_path(parse_path(p), st):     # 逐条重放
        bad.append(pid)

print('不合法路径：', bad)                     # 期望为空
df_submission.to_csv('submission.csv')
```

指标方面，我们只报告两列：**每个 pid 的路径长度**，以及**它相对示例路径的缩短量**。之所以不报「平均长度」，是因为基准长度从 1 跨到 1000，平均一下会把深状态的努力完全抹平。看逐 pid 的缩短量，才能知道我们的搜索到底在哪些状态上起了作用。

---

## 三、总结

本文中我们完成了 CayleyPy 7×7×7 魔方的求解流程。我们首先用 EDA 看清了数据结构：`puzzle_info.json` 提供还原态与动作，`test.csv` 提供打乱后的初始状态，`sample_submission.csv` 提供长度 1 到 1000 的示例路径。其次我们写好了解析与 `check_path` 校验，保证「合法」的判断和评测一致。接着我们把状态空间建成凯莱图，用 CayleyPy 的束搜索配合 Hamming 距离打分，束宽 1000、最多 100 步，对一批测试状态尝试改进。最后我们逐条比较新旧路径的长度，只保留更短且合法的结果，并在提交前对所有路径做了重放校验。

这个比赛给我们的最大启发是：**束搜索的效果几乎完全由打分器决定**。束宽和步数只能让搜索「更耐心」，却无法弥补「看不出哪条路更好」这个根本问题。Hamming 距离适合当第一版基线，但它不懂魔方的结构；下一步真正该做的，是训练一个能估计「离还原还有多少步」的模型，把它接进 `Predictor`，让搜索朝对的方向走。