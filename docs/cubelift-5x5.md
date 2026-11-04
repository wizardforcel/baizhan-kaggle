# CubeLift 5x5：用标定数据校准你的魔方求解器

## 一、比赛简介

### 1.1 比赛背景

CubeLift 系列和 3×3 的源头比赛关系很深。在 3×3×3（Face-and-Middle Layer Turns）这个比赛里，魔方有 **54 个贴纸（label）** 和 **18 个带符号的层转动作**，官方样例提交的总步数是 **500,620 步**。除了这份长解，后来又公开了一批很短的 3×3 解法：`source333_submission.csv`，只有 **21,372 步**，单条最长 24 步。CubeLift 5×5 的题目，就是把这批短解法搬到 5×5 魔方上构造出来的。

构造方法依旧是**词级平移**：把 3×3 的动作一个个换成 5×5 的动作。源里 3×3 的层 `0、1、2`，在 5×5 上对应层 `0、2、4`——也就是「隔一行取一层」。对每个轴 `a`（f、r、d），映射是 `a0→-a0`、`a1→-a2`、`a2→-a4`，符号取反是这套数据约定的一部分。一条 3×3 解法换名后长度不变，把它倒过来作用在已复原的 5×5 上，就得到一条输入。

因为这个造法，每条输入都带着一条**已知的短路线**（不超过 24 步）。但请注意：这条路线在 3×3 上最短，不代表在 5×5 上也最短；5×5 上 150 个贴纸里有 96 个是 3×3「看不到」的，可能存在更短的解。比赛把路线长度公开出来，就是让你把求解器**对准这些已知结果**去调——所以标题叫 "Calibrate Your Solver"。

### 1.2 比赛目标

数据目录里除了 `test.csv`（**1,003 条输入**，每条一个 `initial_state_id` 和 150 个逗号分隔的颜色编号 0～5）和 `sample_submission.csv`，还有一个 **`metric.py` 模块**，导出 `GENERATORS`（动作排列）、`TARGET`（目标状态）、`score`（评分函数）和 `parse_path`（路径解析）。我们可以直接从这个模块拿到所有动作和评分逻辑，比自己从 JSON 里读更稳。

我们要输出的 `path` 用句点连接动作记号。合法动作共 **30 个**：`f0`～`f4`、`r0`～`r4`、`d0`～`d4` 以及它们的负号逆动作。层 `1` 和 `3` 在造数据时用不到，但解题时完全合法。零步的输入写一个空格。

### 1.3 评价标准

排行榜把**全部提交路径的长度加起来**，总步数越小越好。参考解合计 **21,372 步**，单条最长 24 步。半转算 2 个 token，同时转两个平行层也是 2 步，换朝向不免费，目标必须是固定的 150 项颜色数组。

同样没有部分分：每条终点都必须严格等于 `TARGET`，否则校验直接失败。另外，材料特别提醒了一个对照口径：$L_i - b_i$（我的步数减参考步数）只说明「相对这条已知路线多走几步」，不是离未知最优解的差距；`L/b` 是长度比值，只在 `b > 0` 时有意义——本身就等于 0 步的输入（已经复原的）要从比值统计里排除。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把数据读进来。5×5 直接提供一个 `metric.py`，所以这次我们不再自己解析 JSON：

```python
from pathlib import Path
import csv, hashlib, json, sys, time
from collections import deque
import numpy as np
import pandas as pd

candidates = [Path('/kaggle/input/competitions/cubelift-5x5'),
              Path('/kaggle/input/cubelift-5x5'), Path('data'), Path('.')]
DATA = next(p for p in candidates if (p / 'lift_definition.json').is_file())

sys.path.insert(0, str(DATA.resolve()))     # 把数据目录加进模块搜索路径
from metric import GENERATORS, TARGET, score, parse_path

states = pd.read_csv(DATA / 'test.csv', dtype=str, keep_default_na=False)
reference = pd.read_csv(DATA / 'sample_submission.csv', dtype=str, keep_default_na=False)
reference_by_id = dict(zip(reference.initial_state_id, reference.path))

# 用官方评分函数数一下参考解总步数，顺便确认输入条数
reference_total = int(score(states, reference, 'initial_state_id'))
assert reference_total == 21372 and len(states) == 1003
```

打印的结果印证了三点：**1,003 条输入**、参考解合计 **21,372 步**、单条最长 24 步。我们还把路线长度分布打印出来，方便后面按 `reference_moves` 分组看求解器的表现。

再看一条输入的颜色分布：

```python
def parse_state(text):
    return np.array([int(x) for x in text.split(',')], dtype=np.int8)

# 150 个贴纸、6 种颜色，每种应该恰好 25 个
state = parse_state(states.initial_state.iloc[0])
print(state.shape, np.bincount(state, minlength=6))
# 输出：(150,)  [25 25 25 25 25 25]

# 参考解按长度分组：每个长度上各有几条输入
ref_lengths = pd.Series([len(parse_path(p, i)) for i, p in reference_by_id.items()])
print(ref_lengths.value_counts().sort_index())
```

颜色计数守恒（每种 25 个）先记下来，后面读状态时用来拦截坏数据。我们还用 `metric` 里的 `score` 确认了参考解全部合法、终点全部正确——把「评分函数是谁说了算」这件事交还给官方模块，避免自己实现出偏差。

### 2.2 数据清洗

5×5 这份数据的清洗，最精彩的一步是**投影核对**：既然输入是「3×3 动作换名」造出来的，那 5×5 上取层 `0、2、4`（行和列都取 0、2、4）得到的那 54 个位置，就应该和 3×3 的原状态完全一致。这条投影式很简单：5×5 每个面有 25 个位置，面编号 `f`、行 `r`、列 `c` 的全局下标是 `25*f + 5*r + c`，我们取 `r, c ∈ {0,2,4}`，于是下标就是：

```python
# 通用的动作工具：应用一条路线、给单个记号取逆、给整条路线取逆
def apply_moves(state, word):
    result = state.copy()
    for token in word:
        result = result[GENERATORS[token]]
    return result

def inverse_token(token):
    return token[1:] if token.startswith('-') else '-' + token

def inverse_word(word):
    return tuple(inverse_token(t) for t in reversed(word))

# 取每面的第 0、2、4 行和第 0、2、4 列，共 3*3=9 个点，6 个面共 54 个点
projection = np.array([25 * f + 10 * r + 2 * c
                       for f in range(6) for r in range(3) for c in range(3)])

source_info = json.loads((DATA / 'source333_puzzle_info.json').read_text())
source_generators = {k: np.array(v) for k, v in source_info['generators'].items()}

def lift_token(token):
    """3x3 记号 -> 5x5 记号：层号 0,1,2 映射到 0,2,4，并反转符号"""
    return ('' if token.startswith('-') else '-') + token.lstrip('-')[0] + str(2 * int(token[-1]))

# 1) 18 个被映射的生成元，在投影位置上必须和 3x3 的生成元完全一致
for token, permutation in source_generators.items():
    assert np.array_equal(GENERATORS[lift_token(token)][projection], projection[permutation])

# 2) 剩下 12 个层 1、3 的转动，在投影位置上必须什么都不改
for token, permutation in GENERATORS.items():
    if token[-1] in '13':
        assert np.array_equal(permutation[projection], projection)
```

这两条断言把整个「换名 + 投影」的关系固定住了：层 1、3 的转动不影响投影位置（所以 3×3 里看不到它们），层 0、2、4 的转动则和 3×3 完全对应。接着我们逐条输入核对终点：

```python
source_states = pd.read_csv(DATA / 'source333_test.csv', dtype=str, keep_default_na=False)
source_by_id = dict(zip(source_states.initial_state_id, source_states.initial_state))

for row in states.itertuples(index=False):
    full_state = parse_state(row.initial_state)
    # 3x3 原状态每条贴纸是 0..53 的编号，除以 9 得到颜色 0..5
    projected_source = parse_state(source_by_id[row.initial_state_id]) // 9
    # 5x5 的投影位置必须等于 3x3 的原颜色分布
    assert np.array_equal(full_state[projection], projected_source)
    # 播放参考路线，终点必须严格等于 150 项 TARGET
    assert np.array_equal(apply_moves(full_state, parse_path(reference_by_id[row.initial_state_id],
                                                             row.initial_state_id)), TARGET)
print('Verified 18 mapped generators, 12 projection-fixing moves, and all 1,003 endpoints.')
```

跑通这一步后，我们对这份数据就有了充分的信任：它不是随便打乱的 5×5，而是严格从 3×3 短解「长」出来的。

### 2.3 模型训练与超参数调优

这个比赛同样**没有可训练的参数**，笔记本里的 `trained_model=False` 白纸黑字写着。所以这一节我们把「模型」理解为**求解器**，「超参数」理解为**搜索预算**。求解器要满足一个接口：吃进一个完整状态，吐出一个合法路线元组，解不出来就返回 `None`。

```python
preprocess_started = time.perf_counter()
MAX_DEPTH = 2                              # 反向搜索的半径，这是唯一的预算参数
goal = np.asarray(TARGET, dtype=np.int8)
reverse_table = {goal.tobytes(): ()}       # 状态 -> 到目标的解
frontier = deque([(goal.copy(), ())])
expanded_states = 0
for_depth = {0: 1}                         # 记录每一层的状态数，方便算内存

while frontier:
    state, forward = frontier.popleft()
    if len(forward) == MAX_DEPTH:
        continue
    expanded_states += 1
    for token, permutation in GENERATORS.items():
        neighbor = state[permutation]
        key = neighbor.tobytes()
        if key in reverse_table:
            continue
        walk = forward + (token,)
        # 从目标往外走了 walk，那么从 neighbor 回到目标的解就是 walk 的逆
        reverse_table[key] = inverse_word(walk)
        for_depth[len(walk)] = for_depth.get(len(walk), 0) + 1
        frontier.append((neighbor, walk))
preprocess_seconds = time.perf_counter() - preprocess_started
```

这段代码在**预处理阶段**就把全部预算花掉：从目标（复原态）反向做两层广度优先，把所有到目标距离不超过 2 的状态和对应解法都存进字典。这样真正的查询阶段只是 `reverse_table.get(state)` 一次查表，每条输入耗时几乎为零。

为什么预算卡在深度 2？30 个合法动作，每多一层状态数最多乘 30：深度 2 只有几百个状态，深度 3 就上万，深度 4 已经到八十万量级。我们这次故意停在 2，是为了让预处理足够便宜、先把整条框架跑通。材料也明说：**绝大多数输入根本不在这个两层邻域里**（它们由十几到二十几步的路线造出来），返回 `None` 是这种小预算求解器的诚实表现，而不是 bug。

```python
# 预处理结束后看一眼表的规模，这些数字决定了查询阶段能有多快
print({'method': 'exact reverse BFS, depth 2', 'states_by_depth': for_depth,
       'expanded_states': expanded_states, 'preprocess_seconds': preprocess_seconds})

# 想加深一层要先算清楚代价：每多一层，状态数大约乘 30
for depth in (1, 2):
    print(depth, sum(count for d, count in for_depth.items() if d <= depth))

def solve_full_state(state):
    """返回一组合法动作元组；本求解器的预算就是上面那张两层反向表"""
    return reverse_table.get(np.asarray(state, dtype=np.int8).tobytes())
```

查询接口设计成 `solve_full_state` 单独的乐高块，是材料特意强调的「换求解器不换框架」：以后想换成真正的 5×5 求解器，或者加一个学习到的价值函数来引导搜索，都只替换这一个函数，读入、校验、汇报代码一行都不用改。只要新求解器也遵循「要么返回合法完整路线、要么返回 `None`」，测量框架就会自动把它纳进来。

### 2.4 模型测试与指标报告

我们把测量和导出分得很开。对每条输入，先记录**原始求解结果**（不读取任何参考解），再决定导出的路线：

```python
records, answers = [], []
for row in states.itertuples(index=False):
    identity = row.initial_state_id
    state = parse_state(row.initial_state)
    fallback = tuple(parse_path(reference_by_id[identity], identity))   # 参考路线只当兜底

    started = time.perf_counter()
    proposed = solve_full_state(state.copy())
    elapsed = time.perf_counter() - started

    valid, error = False, None
    if proposed is not None:
        # 自己查到的路线也要复核：记号合法、终点真的等于 TARGET
        proposed = tuple(proposed)
        if any(token not in GENERATORS for token in proposed):
            error = 'Unknown move token'
        elif not np.array_equal(apply_moves(state, proposed), TARGET):
            error = 'Endpoint does not equal the full fixed target'
        else:
            valid = True

    b = len(fallback)                      # 参考长度
    length = len(proposed) if valid else None
    records.append({'initial_state_id': identity, 'reference_moves': b,
                    'already_solved': b == 0, 'raw_solved': valid,
                    'raw_moves': length, 'invalid_reason': error,
                    'overhead_moves': length - b if valid else None,
                    'length_ratio': length / b if valid and b > 0 else None,
                    'solver_seconds': elapsed})

    # 导出规则：自己的解有效且不超过参考长度才用；否则回退，保证每一条都完整
    chosen = proposed if valid and len(proposed) <= b else fallback
    answers.append({'initial_state_id': identity, 'path': '.'.join(chosen) or ' '})

per_input = pd.DataFrame(records)
candidate = pd.DataFrame(answers)
full_total = int(score(states, candidate, 'initial_state_id'))   # 用官方评分函数
assert full_total <= reference_total
```

测量里的几个量要念叨一遍：`raw_solved` 是「不动用参考解也能解的条数」，`overhead_moves = L - b` 是相对参考路线的多余步数，`length_ratio = L / b` 是归一化的质量倍数。已经复原的输入（`b == 0`）要留在覆盖率和最终得分里，但只从比值统计里剔除，因为除以 0 没有意义。

报告阶段我们按参考长度分组，看原始覆盖率。因为分布直接决定了下一轮该把搜索预算花在哪：

```python
# 按参考长度分组：每一组有多少条输入、原始解出几条、平均耗时多少
grouped = per_input.groupby('reference_moves', as_index=False).agg(
    inputs=('initial_state_id', 'size'),
    raw_solved=('raw_solved', 'sum'),
    mean_solver_seconds=('solver_seconds', 'mean'))
grouped['raw_coverage'] = grouped.raw_solved / grouped.inputs

# 报告同时保留两层信息：求解器自身的表现，和最终提交的总步数
report = {
    'method': 'Exact reverse BFS to depth 2; reference fallback and per-row best valid choice',
    'trained_model': False,
    'budget': {'reverse_search_depth': MAX_DEPTH,
               'preprocessing_expanded_states': expanded_states},
    'preprocess_seconds': preprocess_seconds,
    'rows': len(states),
    'already_solved_inputs': int(per_input.already_solved.sum()),
    'raw_solved': int(per_input.raw_solved.sum()),
    'raw_failures': int((~per_input.raw_solved).sum()),
    'raw_solve_coverage': float(per_input.raw_solved.mean()),
    'raw_invalid_returns': int(per_input.invalid_reason.notna().sum()),
    'mean_overhead_on_raw_successes': per_input.loc[per_input.raw_solved, 'overhead_moves'].mean(),
    'mean_length_ratio_on_raw_successes': per_input.loc[per_input.raw_solved & (per_input.reference_moves > 0), 'length_ratio'].mean(),
    'reference_total': reference_total, 'score': full_total,
    'saved_moves_vs_reference': reference_total - full_total,
    'all_final_endpoints_valid': True,
}
```

最后把 `candidate` 写成 `submission.csv`，再读回来用 `score` 复核一遍（防止零步答案变成 NaN），同时把分组覆盖率表导出备查。

这个两层的演示**合情合理地复现了 21,372 步的参考总分、却没有任何改进**——材料作者对此毫不掩饰。这正是「标定」的意义：它给全队提供了一个真实的测量起点。以后的实验把 `solve_full_state` 换成更强的求解器，只要预算固定、成功解按参考长度分组报告、失败保留在报告里，就能用同一把尺子对比谁更接近最优。

另外提一句源头故事里的小插曲：3×3 原比赛的参考是 500,620 步，后来公开的短解只有 21,372 步，两者差了二十多倍。这提醒我们，参考文件的「最优性」永远要先打问号，标定数据标定的只是「已知可行上界」，而不是「真正的最短距离」。

---

## 三、总结

本文中我们完成了 CubeLift 5×5 的完整流程。我们首先用 EDA 确认了数据规模：1,003 条输入、150 个贴纸、30 个带符号的层转，并使用官方 `metric.py` 验证了参考解合计 21,372 步、单条最长 24 步。其次我们做了最见功力的一步清洗——投影核对：把 5×5 每面的第 0、2、4 行和第 0、2、4 列投影出来，逐条确认它与 3×3 原状态一致，并验证 18 个映射生成元、12 个投影不变转动的正确性。接着我们把搜索预算设为「反向两层 BFS」，把所有到目标的距离不超过 2 的状态和反解一次预处理成表，并把查询封装成可替换的 `solve_full_state` 接口。然后我们把原始覆盖率、多余步数、长度比值和最终得分分开测量，用官方评分函数导出并复核提交文件。最后我们得到的是一个可复现的标定基线：它可以合法复现 21,372 步的参考总分而不做任何改进，raw 覆盖率也如实地小——这是起点，不是终点。

这个比赛给我们的最大启发是：**给求解器一个固定预算，并如实报告预算外的回退次数，比盲目加搜索深度更能指导下一步实验。** 标定数据的作用不是让你刷出一个漂亮的分数，而是让你知道自己离真实上界有多远、每一步计算换来多少覆盖率——有了这把尺子，换上更强求解器时才算得清账。