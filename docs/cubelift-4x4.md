# CubeLift 4x4：用已知路线标定魔方求解器

## 一、比赛简介

### 1.1 比赛背景

魔方求解看上去是个玩具问题，实际上是组合搜索里最经典的一类难题：状态多到数不清，但我们要找的是「从当前状态回到复原状态的最短动作序列」。CayleyPy 就是一套把这类问题做成标准基准数据的工具，它把一个魔方的每个动作写成一个「位置重排表」——告诉程序「转这一下之后，第 i 个位置的颜色要从原来第几个位置拿」。于是「转一下」就变成了「按这张表重新取一遍数组」。

CubeLift 系列正是在 CayleyPy 的基础上做的放大版。组织者先在 3×3 魔方上积累了一批很短的解法：公开文件 `source333_submission.csv` 里，1,003 条解一共 **21,372 步**，单条最长 **24 步**。然后他们用这批解法去「造」更大魔方的输入。

4×4 的造法可以概括成一句话：**把 3×3 的动作文字逐个换成 4×4 的动作文字。** 3×3 的每个动作 `a0`、`a1`、`a2`（a 是 f、r、d 三个轴之一）在 4×4 上换名成 `-a0`、`-a1`、`-a2`——轴和层号都不变，只把符号反过来。一条 3×3 的解法换名之后长度完全不变，再把这条路线**倒过来**作用在已复原的 4×4 上，就得到了一个输入。所以每条输入天生带着一条已知的短路线。

这里有个很重要的细节，材料的作者反复强调：这种做法叫**词级平移（word transfer）**，不是把 3×3 的状态图嵌入到 4×4 里，也不是把状态投影过去。源路线在 3×3 上是最短的，不代表它在 4×4 上还最短；4×4 的动作多、层多，完全可能存在更短的路。这正是比赛留给参赛者的空间。

### 1.2 比赛目标

数据目录里有几个文件：`puzzle_info.json` 给出目标状态和全部动作的排列数组，`test.csv` 给出输入，`sample_submission.csv` 给出公开参考路线，另外还有 3×3 的源数据（`source333_test.csv`、`source333_submission.csv`、`source333_puzzle_info.json`）。

`test.csv` 有两列：

- `initial_state_id`：输入编号；
- `initial_state`：96 个用逗号分隔的颜色编号，取值 0～5。

我们要输出 `path` 列：用英文句点分隔的动作记号，例如 `-f0.r2.d3`。合法动作一共 **24 个**——`f0`～`f3`、`r0`～`r3`、`d0`～`d3`，以及它们前面加负号的逆动作。层号 3 在造数据时没有被用到，但解题时是合法的。

有两个容易踩的坑：零步的输入（本身就是复原态）必须写「一个 ASCII 空格」，不能留空；`path` 里的记号必须用句点连接，不是空格。

### 1.3 评价标准

排行榜把提交里**所有路线长度加起来**，总步数越小越好。也就是说这是一个最小化问题，不是准确率越高越好。参考解合计 21,372 步，单条最多 24 步。

计步规则要记牢：

- 一个带符号的四分之一转（如 `f2` 或 `-r1`）算 1 步；
- 半转要写两个 token，算 2 步；
- 同时转两个平行层也是 2 步；
- 整魔方换朝向**不免费**，目标必须是那个固定的 96 项颜色数组，不能靠「整体旋转一下就算复原」。

比赛没有部分分。任何一条输入的终点对不上，校验器就直接报错，整个提交作废。所以我们必须做到「每一条都真的解出来」，而不是「大部分对」。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先要把数据读对。Kaggle 的 `/kaggle/input` 下常常同时挂着 5×5、6×6、7×7 的数据，光靠目录名找容易找错，所以我们在候选目录里加了一个硬条件：`central_state` 的长度必须是 96。

```python
from pathlib import Path
from collections import deque
import csv, json, os, time
import numpy as np
import pandas as pd

SLUG = 'cubelift-4x4'
# 逐个候选目录试，只有 facelet 数量正好是 96 的那个才是 4x4 的数据
candidates = [Path('/kaggle/input/competitions') / SLUG,
              Path('/kaggle/input') / SLUG,
              Path('data')]
DATA = next(p for p in candidates
            if (p / 'test.csv').is_file()
            and len(json.loads((p / 'puzzle_info.json').read_text())['central_state']) == 96)

# 所有产物统一写到输出目录，Kaggle 上就是 /kaggle/working
OUTPUT = Path(os.environ.get('CUBELIFT444_OUTPUT_DIR',
              '/kaggle/working' if Path('/kaggle').exists() else 'cubelift444-output'))
OUTPUT.mkdir(parents=True, exist_ok=True)

info = json.loads((DATA / 'puzzle_info.json').read_text())
GOAL = np.array(info['central_state'], dtype=np.int8)                    # 已复原的 96 个颜色
G = {t: np.array(p, dtype=np.intp) for t, p in info['generators'].items()}

# keep_default_na=False 非常关键：零步答案是一个空格，
# 如果让 pandas 自己判断缺失值，空格会变成 NaN，后面切字符串就崩了
states = pd.read_csv(DATA / 'test.csv', dtype=str, keep_default_na=False)
reference = pd.read_csv(DATA / 'sample_submission.csv', dtype=str, keep_default_na=False)

# 3x3 的源数据也一并读进来，用来核对输入是怎么造出来的
source_states = pd.read_csv(DATA / 'source333_test.csv', dtype=str, keep_default_na=False)
source_routes = pd.read_csv(DATA / 'source333_submission.csv', dtype=str, keep_default_na=False)
G3 = {t: np.array(p, dtype=np.intp)
      for t, p in json.loads((DATA / 'source333_puzzle_info.json').read_text())['generators'].items()}

print({'inputs': len(states), 'facelets': len(GOAL), 'signed_moves': len(G)})
```

打印结果是 `facelets=96`、`signed_moves=24`，和比赛说明一致。

接下来我们看目标长什么样，再看一条输入：

```python
# 目标不是"随便哪个朝向的复原"，而是 0..5 每种颜色连续出现 16 次的固定数组
assert np.array_equal(GOAL, np.repeat(np.arange(6, dtype=np.int8), 16))

# 抽第一条输入，把字符串拆成数组，顺便数一数每种颜色有多少个
state = np.array([int(x) for x in states.initial_state.iloc[0].split(',')], dtype=np.int8)
print(state.shape, np.bincount(state, minlength=6))
# 输出：(96,)  [16 16 16 16 16 16]

# 再看参考解的长度：单条最长多少、总共多少步
lengths = [len(p.split('.')) if p.strip() else 0 for p in reference.path]
print({'max_reference_moves': max(lengths), 'reference_total': sum(lengths)})
# 输出：{'max_reference_moves': 24, 'reference_total': 21372}
```

每种颜色恰好 16 个，这是魔方的「贴纸守恒」——不管怎么转，颜色的数量不会变。这条不变量后面会写进读取函数，成为我们挡住坏数据的第一道闸门。

### 2.2 数据清洗

这份数据其实很干净：没有缺失值，没有重复行，也没有时间字段要处理。真正需要「清洗」的是**路径**。因为比赛没有部分分，只要一个记号写错、或者两个记号之间的顺序反了，终点就对不上，整个提交归零。

所以我们不做任何统计意义上的清洗——不去异常值、不做归一化、更不改颜色编号，因为一改终点就错了。我们要做的是把两个读入函数写死，让错误的输入在门口就被拦住：

```python
# 路径和状态的读入函数：宁可在这里报错，也不让坏数据流进搜索
def inverse(token):
    """取逆：带负号的去掉负号，不带负号的加上负号"""
    return token[1:] if token.startswith('-') else '-' + token

def inverse_word(w):
    """整条路线取逆：每个记号取逆，再整体倒序"""
    return tuple(inverse(t) for t in reversed(w))

def word(text, generators=G):
    """把 '-f0.r2.d3' 切成合法记号元组，零步写成空元组"""
    if not isinstance(text, str):
        raise ValueError('A path must be text; use one ASCII space for zero moves')
    result = tuple(text.strip().split('.')) if text.strip() else ()
    # 只要有一个记号不在动作表里，立刻报错，避免"看起来解出来了"的假象
    if any(t not in generators for t in result):
        raise ValueError('Illegal or empty move token')
    return result

def read_state(text):
    """读一条状态，同时检查三件事：长度 96、颜色在 0..5、每种颜色 16 个"""
    out = np.array([int(x) for x in text.split(',')], dtype=np.int16)
    if out.shape != (96,) or np.any(out < 0) or np.any(out > 5):
        raise ValueError('Expected 96 colors in 0..5')
    if not np.array_equal(np.bincount(out, minlength=6), np.repeat(16, 6)):
        raise ValueError('Expected sixteen occurrences of each color')
    return out.astype(np.int8)

def replay(state, path, generators=G):
    """按顺序走完一条路线；每次都从旧数组取，绝不原地改写"""
    out = state.copy()
    for token in path:
        out = out[generators[token]]
    return out
```

第二个清洗步骤是核对「动作」本身。材料的说明里给了一个很容易被忽略的语义：应用一个动作要写 `state = state[g]`，也就是**新状态第 i 个位置的颜色，取自旧状态第 g[i] 个位置**。这意味着必须用新数组承接结果，不能原地改写，否则索引会互相污染。

我们用三条断言把这个语义钉死：

```python
identity = np.arange(96)

for token, permutation in G.items():
    # 1) 每个动作必须是 96 个位置的一个排列（每个位置恰好被取一次）
    assert np.array_equal(np.sort(permutation), identity)
    # 2) 正转接反转等于什么都没做，这验证了我们的 inverse 写对了
    assert np.array_equal(permutation[G[inverse(token)]], identity)
    # 3) 连转四次回到原地，这是"四分之一转"的定义
    assert np.array_equal(permutation[permutation[permutation[permutation]]], identity)

# 24 个动作、96 个位置，这两个数字对上才说明数据没被换成别的魔方
assert len(G) == 24 and GOAL.shape == (96,)
```

第三条断言同时解释了一个计分细节：为什么半转要写两个 token？因为我们的动作集合里只有四分之一转，没有单独的「半转」动作。要把口径保持成 24 个记号，半转就只能老老实实写两遍，自然算 2 步。这样做的好处是整套代码只有一种动作类型，逆动作、合法性检查、搜索都不用写分支。

### 2.3 模型训练与超参数调优

这里要先说清楚一件事：**这个比赛没有可训练的模型。** 材料里写得很明白——no training、no pretrained model、no internet download。所以这一节我们说的「模型」其实是**求解器**，「超参数」是**搜索深度和搜索预算**。把它们当成模型来调，思路是一样的：先定一个明确的预算，再在这个预算下衡量效果。

我们做的第一件事，是把「输入是怎么造出来的」复现一遍。这既是数据校验，也是理解题目的关键：

```python
def lift_token(token):
    """把 3x3 的记号换成 4x4 的：轴和层号不变，符号取反"""
    axis, layer = token.lstrip('-')
    sign = '' if token.startswith('-') else '-'   # 源里的正转，到目标里变成反转
    mapped = sign + axis + layer
    assert mapped in G                            # 换名之后必须仍是合法动作
    return (mapped,)

sources = dict(zip(source_states.initial_state_id, source_states.initial_state))
source_paths = dict(zip(source_routes.initial_state_id, source_routes.path))
refs = dict(zip(reference.initial_state_id, reference.path))

source_total = 0
for row in states.itertuples(index=False):
    # 第一步：先把 3x3 的原状态和原路线跑一遍，确认来源文件本身是可信的
    original = np.array([int(x) for x in sources[row.initial_state_id].split(',')])
    source_path = word(source_paths[row.initial_state_id], G3)
    assert np.array_equal(replay(original, source_path, G3), np.arange(54))

    # 第二步：逐个记号换名，一个源动作换成一个目标动作，长度不变
    lifted = tuple(t for move in source_path for t in lift_token(move))
    assert len(lifted) == len(source_path)
    # 第三步：换名后的路线必须和公开参考解一字不差
    assert lifted == word(refs[row.initial_state_id])
    # 第四步：输入 = 从复原态出发，倒着走完这条路线
    assert np.array_equal(read_state(row.initial_state), replay(GOAL, inverse_word(lifted)))
    source_total += len(source_path)

assert source_total == 21372
```

这一步跑通，我们就确认了三件事：源数据可信、换名规则正确、每条输入真的对应一条已知的 21,372 步以内的路线。之后所有搜索都建立在这个「已知有解」的前提上。

第二件事是写求解器。我们用的是**双向广度优先搜索**：从复原态走两步，把所有能到达的状态做成一张表；再从每条输入走两步，看能不能碰上表里的状态。碰上了就把两段拼起来，长度最多 4 步。

```python
TOKENS = tuple(sorted(G))
START_DEPTH = GOAL_DEPTH = 2     # 两个方向各两步，合起来覆盖长度不超过 4 的解

def goal_table(depth):
    """从复原态做 BFS，把 depth 步以内能到达的状态和最短路线都记下来"""
    table = {GOAL.tobytes(): ()}
    queue = deque([(GOAL.copy(), ())])
    while queue:
        current, route = queue.popleft()
        if len(route) == depth:
            continue
        for token in TOKENS:
            # 刚走过去又立刻走回来，等价于什么都不做，直接跳过
            if route and token == inverse(route[-1]):
                continue
            nxt = current[G[token]]
            key = nxt.tobytes()
            if key not in table:      # BFS 第一次到达某个状态，走的就是最短路
                table[key] = route + (token,)
                queue.append((nxt, route + (token,)))
    return table

BACKWARD = goal_table(GOAL_DEPTH)
```

```python
def solve_bounded(initial):
    """从输入出发走两步，每到一个状态就查表，取最短的拼接结果"""
    queue = deque([(initial.copy(), ())])
    seen = {initial.tobytes()}
    best = None
    expanded = 0
    while queue:
        current, route = queue.popleft()
        expanded += 1
        back = BACKWARD.get(current.tobytes())
        if back is not None:
            # 前半段是正着走的 route，后半段是表路线的逆，拼起来才是完整解
            candidate = route + inverse_word(back)
            if best is None or len(candidate) < len(best):
                best = candidate
        if len(route) == START_DEPTH:
            continue
        for token in TOKENS:
            if route and token == inverse(route[-1]):
                continue
            nxt = current[G[token]]
            key = nxt.tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((nxt, route + (token,)))
    if best is not None:
        assert len(best) <= START_DEPTH + GOAL_DEPTH     # 长度上限要守住
        assert np.array_equal(replay(initial, best), GOAL)  # 拼出来的解必须真的解对
    return best, expanded
```

这两个超参数看起来不起眼，但它们决定了整个实验的成本。4×4 有 24 个动作，每多走一步，状态数最多乘 24：深度 2 大约是几百个状态，深度 3 就上万，深度 4 已经三十多万。所以深度 2 的表轻松放进内存，深度 3 开始变慢，再往上普通笔记本就跑不动了。我们把预算定在 2+2，换来的是「长度 ≤ 4 一定有解、且一定最短」这个明确的保证，代价是**绝大多数输入根本不在这个邻域里**。

我们明知这个搜索很弱还是先写它，原因有三个：它给了我们一个能跑通、能验证的搜索骨架；它把逆动作处理、失败信号、终点复核都实现了一遍；后面换成任何更强的求解器，只要替换 `solve_bounded`，外面的读入、校验、汇报代码一行都不用改。

### 2.4 模型测试与指标报告

测的时候我们把三类数字**分开记**，绝不混在一起：

1. `raw_success`：搜索自己有没有解出来（在用到参考解之前）；
2. `submitted_moves`：最终提交的这条路线有多少步；
3. `length_minus_reference`：提交步数减去参考步数，也就是 $L_i - b_i$。

这个 $L_i - b_i$ 的含义要解释清楚：它只说明「相对那条公开参考路线，我们多走了几步」，**不是**「离真正最优解还差几步」。因为真正的最近距离我们根本不知道（要证明最短需要穷举，做不到）。材料里特意提醒了这一点，我们认为这是整篇笔记最诚实的地方。

```python
records, answers = [], []
for row in states.itertuples(index=False):
    initial = read_state(row.initial_state)
    started = time.perf_counter()
    found, expanded = solve_bounded(initial)
    seconds = time.perf_counter() - started

    baseline = word(refs[row.initial_state_id])
    raw_success = found is not None
    # 规则：自己解出来了、而且比参考更短，才用自己的；否则退回参考解
    selected = found if raw_success and len(found) < len(baseline) else baseline
    assert np.array_equal(replay(initial, selected), GOAL)   # 导出前逐条复核终点

    records.append({'initial_state_id': row.initial_state_id,
                    'initially_solved': bool(np.array_equal(initial, GOAL)),
                    'raw_success': raw_success,
                    'raw_moves': len(found) if raw_success else None,
                    'expanded_states': expanded, 'search_seconds': seconds,
                    'reference_moves': len(baseline), 'submitted_moves': len(selected),
                    'length_minus_reference': len(selected) - len(baseline),
                    'used_reference_fallback': not raw_success})
    answers.append({'initial_state_id': row.initial_state_id,
                    'path': '.'.join(selected) or ' '})   # 空路线写一个空格
```

最后我们写提交文件，并且**重新读回来再校验一次**。这一步看起来多余，其实很必要：写 CSV 时如果零步答案写成了空串，重新读会变成 NaN，行数或内容就变了。

```python
def validate(answer):
    """逐条检查提交：编号齐全且不重复、每个记号合法、每条终点都等于目标"""
    if set(answer.columns) != {'initial_state_id', 'path'}:
        raise ValueError('Expected exactly initial_state_id,path')
    ids = answer.initial_state_id.tolist()
    if len(ids) != len(set(ids)) or set(ids) != set(states.initial_state_id):
        raise ValueError('Every input ID must occur exactly once')
    routes = dict(zip(ids, answer.path))
    total = 0
    for row in states.itertuples(index=False):
        path = word(routes[row.initial_state_id])
        if not np.array_equal(replay(read_state(row.initial_state), path), GOAL):
            raise ValueError(f'Unsolved 96-facelet endpoint: {row.initial_state_id}')
        total += len(path)
    return total

results = pd.DataFrame(records)
answer = pd.DataFrame(answers, columns=['initial_state_id', 'path'])
reference_total = sum(len(word(p)) for p in reference.path)   # 参考解总步数
score = validate(answer)                 # 通过校验的同时，把总步数算出来
answer.to_csv(OUTPUT / 'submission.csv', index=False,
              quoting=csv.QUOTE_ALL, lineterminator='\n')

# 把刚写出去的文件重新读回来，确认空格没有变成缺失值
exported = pd.read_csv(OUTPUT / 'submission.csv', dtype=str, keep_default_na=False)
assert validate(exported) == score

report = {'rows': len(states), 'reference_score': reference_total, 'score': score,
          'raw_solved': int(results.raw_success.sum()),
          'raw_failures': int((~results.raw_success).sum()),
          'improved_inputs': int((results.length_minus_reference < 0).sum()),
          'saved_moves': reference_total - score,
          'uses_public_reference_fallback': True, 'trained_model': False}
print(json.dumps(report, indent=2))
```

报告里最关键的是 `raw_solved` 和 `saved_moves` 这两个数：前者告诉我们「不看参考解，我们的求解器到底能解几条」；后者告诉我们「相比参考解省了几步」。因为深度四的搜索只覆盖极短的解，而每条输入都是由十几到二十几步的路线造出来的，`raw_solved` 会很小，`saved_moves` 大概率是 0。

这不是失败，而是**标定**：我们量出了一个明确的起点，知道「预算 2+2 能换来什么」。下次把搜索换成更强的、或者加一个学习到的价值函数，只要用同一套接口重跑，就能直接和这次的数字对比。

---

## 三、总结

本文中我们完成了 CubeLift 4×4 的完整流程。我们首先用 EDA 确认了数据规模：96 个贴纸、6 种颜色、24 个带符号的层转，每种颜色恰好 16 个，目标是固定的颜色数组。其次我们做了针对性的「清洗」——不是去异常值，而是把状态读入、路径解析、动作语义（`state = state[g]`，必须用新数组）用断言钉死，因为比赛没有部分分，一个记号写错就全盘作废。接着我们复现了「词级平移」的构造过程：把 3×3 的每个动作换名成 4×4 的动作、符号取反，再取逆作用在复原态上，逐条验证了 21,372 步、单条最长 24 步的参考解。然后我们用双向宽度优先搜索做了一个「深度四以内保证最短」的求解器，超参数就是两个方向的搜索深度。最后我们把搜索结果和公开参考解取短者导出，并把 raw 覆盖率、总步数、相对参考的步数差分成三个独立指标写进报告。

这个比赛给我们的最大启发是：**在没法证明最优的问题里，把「量了什么」和「拿到了什么」分开说，比报一个好看的分数更重要。** 一个完整的 `submission.csv` 不代表求解器解出了每一条输入；把回退用了几次、raw 覆盖多少如实写进报告，才是可复现、可对比的标定。
