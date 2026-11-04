# CayleyPy 6×6×6 魔方最优求解（CayleyPy 6x6x6 Cube Solve Optimally）

## 一、比赛简介

### 1.1 比赛背景

这是 CayleyPy 系列的一个社区比赛：给定一个被打乱的 **6×6×6 魔方**，要求我们用尽量少的转动步数把它还原。它表面上是个玩具问题，实际上是一个很大的算法问题——**在超大规模的图上找最短路径**。

要理解它，先要理解**凯莱图（Cayley graph）**这个结构。把魔方的每一种摆法看成图上的一个「节点」，把每一步转动看成一条「边」：从摆法 A 转一步能到摆法 B，A 和 B 之间就连一条边。因为每一步转动都有对应的逆转动，这些边是无向的，所以「还原魔方」就等于「在这张图上从起点走到还原态」。

这张图有多大？材料里给了参照：3×3×3 魔方大约有 **4.33E+19** 个状态，而一个 master tetraminx 大约有 **1.54E+32** 个状态。6×6×6 只会更夸张。这么大的图不可能整个装进内存，更不可能用暴力搜索枚举所有路径，因此我们只能用「一边往前走、一边猜哪条路更有希望」的**启发式搜索**——也就是这场比赛真正考的东西。

### 1.2 比赛目标

输入是每个初始状态的贴纸排列，输出是一条能把它还原的转动序列。比赛给了三个文件：

- `puzzle_info.json`：里面是 `central_state`（还原态，也就是「已经拼好」的状态）和 `generators`（生成元，也就是所有允许的转动动作）。每个生成元都有一个逆元，名字就是在原名前面加一个负号，比如 `f0` 的逆元是 `-f0`；
- `test.csv`：每行是一条 `initial_state`，是组委会用一串随机转动把还原态打乱得到的；
- `sample_submission.csv`：每个 `initial_state_id` 配一条示例 `path`，用点号分隔的转动名表示，比如 `f0.-r1.d2`。

生成元的用法非常朴素：它是一个**下标数组**。`state[generators[name]]` 就是「按照这个动作把贴纸重新排一遍」。我们要交的，就是给每个 `initial_state_id` 写一条能回到 `central_state` 的路径。

### 1.3 评价标准

最硬的规则是：**提交的路径必须真的把状态转回还原态**。只要有一条路径转不回去，它就是无效的。

在这个前提下，路径越短越好。`sample_submission.csv` 里的路径其实就是当初打乱那条路径的**逆过程**——材料里点明，这些路径的长度从 1 到 1000 不等（前几个长度很小的状态离还原态很近，是很好的调试样本）。所以我们可以把 sample 路径当成一条「基准线」：如果我们解出的路径比它短，就替换掉它；如果更短不了，就保留原样。这也是这个系列比赛最朴素的得分思路——**只赚「比基准短」的那部分**。

---

## 二、解题过程

我们的做法是：**先把数据读通、把路径校验写对，再用 CayleyPy 自带的束搜索（beam search）配合 Hamming 距离打分，对一部分测试状态尝试改进路径。** 这是一个能跑通的基线，后面再换成学习出来的打分器。我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步走。

### 2.1 数据探索（EDA）

我们首先读入 `puzzle_info.json`，拿到还原态和所有生成元，并验证「转一步再转逆元能复原」：

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd

INPUTS_DIR = Path('/kaggle/input/cayley-py-666-cube')
with open(INPUTS_DIR / 'puzzle_info.json') as f:
    puzzle_info = json.load(f)

central_state = np.array(puzzle_info['central_state'])                    # 还原态
generators = {k: np.array(v) for k, v in puzzle_info['generators'].items()}
print('生成元名字：', list(generators.keys()))

def apply_gen(state, gen_name, generators):
    # 施加一个动作：本质上就是按生成元做一次数组下标重排
    return state[generators[gen_name]]

# 转一步，再转它的逆元，应该回到还原态
new_state = apply_gen(central_state, 'f0', generators)
back = apply_gen(new_state, '-f0', generators)
print('回到还原态：', np.all(central_state == back))
```

接着我们读测试集和示例提交，看看数据结构，并挑一条示例路径验证它确实能还原：

```python
df_test = pd.read_csv(INPUTS_DIR / 'test.csv', index_col='initial_state_id')
df_sample = pd.read_csv(INPUTS_DIR / 'sample_submission.csv',
                        index_col='initial_state_id')

print(df_test.head())      # 列：initial_state（逗号分隔的整数）
print(df_sample.head())    # 列：path（点号分隔的转动名）
print('测试状态数：', len(df_test))
```

这一步还透露出一个有用的信息：`sample_submission` 的路径长度分布极不均匀——有的只有 1 步，有的接近 1000 步。浅的状态随便搜搜就能解，深的状态才是真正吃算力的地方。这也决定了后面我们不会平均用力，而是把束搜索的预算花在深处的状态上。

### 2.2 数据清洗

这份数据不需要去重、补缺失，所谓清洗就是**把文本解析成可计算的数组，并且把「合法性」检查写扎实**。原因很现实：如果解析错了（比如把状态读成了字符串），后面算出来的路径看着能跑，其实全是错的；如果合法性检查不严，交上去的路径可能根本还原不了。

我们写了三个小工具：解析状态、解析路径、校验路径：

```python
def parse_initial_state(initial_state_str: str) -> np.ndarray:
    # "1,2,3,..." -> 一维整数数组
    return np.array([int(x) for x in initial_state_str.split(',')])

def parse_path(path_str: str) -> list:
    # "f0.-r1.d2" -> ['f0', '-r1', 'd2']
    return list(path_str.split('.'))

def check_path(path, initial_state, central_state, generators) -> bool:
    # 依次施加路径里的动作，最后必须和还原态逐元素相等
    state = initial_state
    for gen_name in path:
        state = state[generators[gen_name]]
    return bool(np.all(state == central_state))
```

有了这三个函数，我们就可以对任意一行数据做端到端校验：

```python
id_to_check = 1
path_to_check = parse_path(df_sample.loc[id_to_check, 'path'])
initial_state = parse_initial_state(df_test.loc[id_to_check, 'initial_state'])

is_valid = check_path(path_to_check, initial_state, central_state, generators)
print(f'第 {id_to_check} 条示例路径合法：{is_valid}')
```

这个检查不是走过场。束搜索返回的路径、我们手工拼出来的路径、以及最后写进提交文件的每一条路径，都要过一遍 `check_path`。在这样一个「错一步就还原不了」的任务里，**校验代码比模型代码更值得先写好**。

### 2.3 模型训练与超参数调优

严格说，这一节我们训练的不是神经网络，而是**搜索算法**。但思路和调模型是一样的：先定义一个「打分函数」，再调它的「容量参数」。

**先把问题变成图。** CayleyPy 提供了把 `central_state` 和 `generators` 直接变成凯莱图的工具：

```python
from cayleypy import CayleyGraphDef, CayleyGraph, Predictor

gen_names = list(generators.keys())
graph_def = CayleyGraphDef.create(
    generators=[generators[x] for x in gen_names],   # 每个动作的置换表
    generator_names=gen_names,
    central_state=central_state)
graph = CayleyGraph(graph_def)
```

**再理解束搜索。** 它的流程可以这样描述：从起点出发，维护一个最多装 W 条候选路径的「篮子」；每一轮把篮子里每条路径的末状态都展开成所有邻居，用打分器 `p(x)` 给每个邻居打分，把「路径 + 分数」放回篮子；如果篮子超过 W 条，就丢掉分数最差的；如果某条路径已经到达还原态，就返回它；如果到了最大步数 L 还没找到，就认输。

这里有两个关键参数：**束宽 W** 决定同时保留多少条候选，**最大步数 L** 决定最多走多远。W 越大、L 越长，越不容易漏掉好路径，但内存和时间的开销也越大。打分器越好，W 和 L 就可以越小。

**打分器先用最简单的。** 材料里的入门方案用 **Hamming 距离**：数一数当前状态和还原态有多少个贴纸位置不一样。这个打分器便宜、跑得快，但它不懂魔方的结构，所以在深状态上经常「看走眼」。我们先拿它跑一个单状态试试：

```python
from tqdm.auto import tqdm

id_to_check = 13
initial_state = parse_initial_state(df_test.loc[id_to_check, 'initial_state'])
sample_path = parse_path(df_sample.loc[id_to_check, 'path'])

# 束宽 1000、最多 50 步、用 Hamming 距离当打分器
result = graph.beam_search(
    start_state=initial_state,
    beam_width=1000,
    max_steps=50,
    predictor=Predictor(graph, 'hamming'),
    beam_mode='simple',
    return_path=True)

print('找到路径：', result.path_found)
print('束搜索长度：', len(result.path), ' 示例路径长度：', len(sample_path))

# 解出来的路径也要过一遍合法性检查
print('合法：', check_path(result.path, initial_state, central_state, generators))
```

材料里这个例子的结果是：束搜索能成功，而且解出来的路径比赛给的示例更短。这说明即使是一个很粗糙的打分器，只要搜索方向大致对，也能在浅状态上带来改进。

**调参。** 我们主要调两个量：束宽和最大步数。把束宽从 1000 往上调、把最大步数从 50 提到 100，成功率会上升，但每个状态要花的时间也线性上涨。反过来，如果把束宽压得太小，篮子很快被「看起来还行、其实走不通」的路径占满，搜索就会失败。材料里的建议是：**先用小参数把流程跑通，再对真正想拿分的深状态加大预算**。至于 Hamming 这个打分器本身的缺陷，靠调 W 和 L 只能缓解、不能根治——要真正提升，得把打分器换成一个学习出来的「距离估计器」，这也是后续方案的方向。

### 2.4 模型测试与指标报告

确认单个状态能跑通后，我们对一小批测试状态批量跑束搜索，并且**逐条与示例路径比长度**：束搜索成功且更短，就采用束搜索的结果；否则保留原来的示例路径。这样最坏情况也不会比基准差：

```python
beam_width = 1000
max_steps = 100
first_n = 10

final_paths = []
# 从第 12 条开始取 10 条，避开前面那些一步就能解的「送分题」
for row_id, row in tqdm(df_test[12:12 + first_n].iterrows(), total=first_n):
    sample_path = parse_path(df_sample.loc[row_id, 'path'])
    initial_state = parse_initial_state(row['initial_state'])

    result = graph.beam_search(
        start_state=initial_state,
        beam_width=beam_width, max_steps=max_steps,
        predictor=Predictor(graph, 'hamming'),
        beam_mode='simple', return_path=True)

    len_sample = len(sample_path)
    len_beam = len(result.path) if result.path_found else np.inf

    # 只有「成功且更短」才替换，否则沿用示例路径
    if len_beam < len_sample:
        final_paths.append('.'.join(result.path))
    else:
        final_paths.append('.'.join(sample_path))

    print(f'{row_id:>3} 成功={result.path_found} 束搜索={len_beam} 示例={len_sample}')
```

从打印结果我们能直接读出三件事：第一，**束搜索在部分状态上确实更短**；第二，**它也会失败**（尤其是步数预算不够的深状态）；第三，**示例路径是可靠的兜底**，因为它本来就是打乱过程的逆序，必然合法。

最后我们把「前面原样保留 + 中间替换过的 10 条 + 后面原样保留」拼起来，写成交付文件：

```python
new_paths = (list(df_sample[:12]['path'])
             + final_paths
             + list(df_sample[12 + first_n:]['path']))

df_submission = df_sample.copy()
df_submission['path'] = new_paths

# 提交前逐条重放校验，确保没有一条路径是坏的
bad = 0
for pid, p in df_submission['path'].items():
    st = parse_initial_state(df_test.loc[pid, 'initial_state'])
    if not check_path(parse_path(p), st, central_state, generators):
        bad += 1
print('不合法路径数：', bad)

df_submission.to_csv('submission.csv')
```

指标报告我们只看两个数：**每个 pid 的路径长度**和**相对示例路径的缩短量**。把这两列并排打印出来，就能一眼看出哪些状态受益、哪些状态白跑。我们不会用「平均长度」这种把深浅状态混在一起的指标来骗自己——在基准长度从 1 到 1000 的分布上，平均值没有意义。

---

## 三、总结

本文中我们完成了 CayleyPy 6×6×6 魔方求解的入门流程。我们首先用 EDA 认识了数据结构：`puzzle_info.json` 给出还原态和生成元，`test.csv` 给出打乱后的初始状态，`sample_submission.csv` 给出长度 1 到 1000 的示例路径。其次我们把状态和路径解析成数组与列表，并写了严格的 `check_path` 校验，确保每条路径都真的能还原。接着我们把问题建模成凯莱图，用 CayleyPy 的束搜索配合 Hamming 距离打分，束宽 1000、最多 100 步，对一批测试状态尝试改进。最后我们逐条比较束搜索路径与示例路径的长度，只保留更短的结果，并在提交前对每一条路径重放校验。

这个比赛给我们的最大启发是：**搜索问题的瓶颈往往在打分器，而不在搜索本身**。Hamming 距离简单、快，但不懂魔方结构，所以只能当基线；真正要提升，需要训练一个能估计「离还原还有多远」的模型来替换它。另外，示例提交本身就是一份可靠的兜底——先保证「不比基准差」，再谈怎么变好，这是这类比赛最稳的推进方式。