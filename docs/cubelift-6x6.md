# CubeLift 6x6：测搜索，而不只是测最终提交文件

## 一、比赛简介

### 1.1 比赛背景

CubeLift 系列把 3×3 魔方的已知短解法搬到更大的魔方上，构造出一条条「带着标准答案」的输入。6×6 是四个版本里最特别的一个：**它的边长是偶数**，不像 4×4 那样层号可以直接照搬——偶数阶魔方没有单个正中心层，层的选法必须重新约定。

组织者的做法是：3×3 的层 `0、1、2` 在 6×6 上映射到层 `0、2、5`（也就是数过去的第 1、3、6 层），对每个轴 `a`（f、r、d）：`a0→-a0`、`a1→-a2`、`a2→-a5`。符号取反照旧。之所以选 `0、2、5` 这三层，是因为它们既包含两个最外层，又把中间的位置错开，偶数阶没有对称中心，这个选择是明确的约定，而不是什么「对称网格」的自然结果。

这也解释了笔记本标题为什么是「测搜索，而不只是测最终提交文件」：整份数据只有一条已知长短的参考路线当尺子，真正能拉开差距的是——在同样的计算预算下，你的求解器多解出几条、每条多走几步。所以我们后面所有实验都在量这两件事，而不是只盯着一个总分。

每个 3×3 动作换成一个 6×6 动作，一条 3×3 解法换名后长度不变，倒过来作用在已复原的 6×6 上，就得到一条输入。公开参考解合计仍然 **21,372 步**，单条最长 **24 步**。和另外几个版本一样，这些是**构造出来的可行路线**，不是最短距离的证明：源路线在 3×3 上最短，不代表在 6×6 上最短。

### 1.2 比赛目标

`test.csv` 的每条输入包含一个 `initial_state_id` 和 **216 个**逗号分隔的颜色编号（0～5）。要输出的 `path` 用句点连接动作记号。

这版有 **36 个**合法动作：`f0`～`f5`、`r0`～`r5`、`d0`～`d5` 以及它们的负号逆动作。造数据只用层 `0、2、5`，解题时 6 个层全部合法。半转算 2 个 token，同时转两层也是 2 步，换朝向不免费，目标必须是固定的 216 项颜色数组。还有一条不变量：每种颜色恰好 **36 个**副本，分布在固定的 6 个面上。

### 1.3 评价标准

排行榜把全部提交路径的总长度相加，**越小越好**。参考解合计 21,372 步、单条最多 24 步。仍然是零容忍校验：每条终点必须严格等于目标状态，否则整体报错。

材料在评价标准上额外强调了**会计口径**：搜索本身的原始成功率（`raw_solved`）、每条输入的扩展状态数和耗时、以及「提交步数 − 参考步数」的 $L_i - b_i$，要分开记录、分开报告。最终 CSV 总步数只是几个指标之一。$L_i - b_i$ 的正值只说明相对那条公开路线多走了几步，不说明离未知最优解差多远的精确距离。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先定位数据。和 4×4 一样，`/kaggle/input` 下可能同时挂着好几个尺寸的数据，我们用 facelet 数等于 216 来判断这是 6×6：

```python
from pathlib import Path
from collections import deque
import csv, json, os, time
import numpy as np
import pandas as pd

SLUG = 'cubelift-6x6'
candidates = [Path('/kaggle/input/competitions') / SLUG,
              Path('/kaggle/input') / SLUG,
              Path('data')]
DATA = next(p for p in candidates
            if (p / 'test.csv').is_file()
            and len(json.loads((p / 'puzzle_info.json').read_text())['central_state']) == 216)

# 产物统一写到输出目录，Kaggle 上就是 /kaggle/working
OUTPUT = Path(os.environ.get('CUBELIFT666_OUTPUT_DIR',
              '/kaggle/working' if Path('/kaggle').exists() else 'cubelift666-output'))
OUTPUT.mkdir(parents=True, exist_ok=True)

info = json.loads((DATA / 'puzzle_info.json').read_text())
GOAL = np.array(info['central_state'], dtype=np.intp)                   # 已复原的 216 个颜色
G = {t: np.array(p, dtype=np.intp) for t, p in info['generators'].items()}

states = pd.read_csv(DATA / 'test.csv', dtype=str, keep_default_na=False)
reference = pd.read_csv(DATA / 'sample_submission.csv', dtype=str, keep_default_na=False)
refs = dict(zip(reference.initial_state_id, reference.path))

print({'inputs': len(states), 'facelets': len(GOAL), 'signed_moves': len(G)})
```

预期输出：`facelets=216`、`signed_moves=36`。我们再看目标数组和一条输入的颜色分布：

```python
# 目标是固定数组：0..5 每种颜色 36 个副本，按面分块排列
assert np.array_equal(GOAL, np.repeat(np.arange(6, dtype=np.intp), 36))

# 抽一条输入，数颜色
state = np.array([int(x) for x in states.initial_state.iloc[0].split(',')], dtype=np.intp)
print(state.shape, np.bincount(state, minlength=6))
# 输出：(216,)  [36 36 36 36 36 36]

# 参考解的长度信息：单条上限与总步数
lengths = [len(p.split('.')) if p.strip() else 0 for p in reference.path]
print({'max_reference_moves': max(lengths), 'reference_total': sum(lengths)})
# 输出：{'max_reference_moves': 24, 'reference_total': 21372}
```

颜色守恒在这里依然成立。这告诉我们：魔方转动只交换位置、不改颜色数，这个数量约束可以成为后面读状态时的第一道防线。

### 2.2 数据清洗

这份数据没有缺失值、没有重复，需要清洗的是**读入函数和路径解析**。我们把三条检查写死在读取里：形状必须是 216、颜色必须在 0～5、每种颜色必须恰好 36 个。任何一条不满足，直接报错，避免坏数据混进后面的搜索：

```python
# 读入函数：把"形状、颜色范围、颜色计数"三件事一次查完
def read_state(text):
    """读一条状态并检查：长度 216、颜色 0..5、每种恰好 36 个"""
    out = np.array([int(x) for x in text.split(',')], dtype=np.intp)
    if out.shape != (216,) or np.any(out < 0) or np.any(out > 5):
        raise ValueError('Expected 216 colors in 0..5')
    if not np.array_equal(np.bincount(out, minlength=6), np.repeat(36, 6)):
        raise ValueError('Expected thirty-six occurrences of each color')
    return out

def inverse(token):
    """取逆动作：带负号去掉负号，不带负号加上负号"""
    return token[1:] if token.startswith('-') else '-' + token

def inverse_word(w):
    """整条路线取逆：逐个记号取逆再整体倒序"""
    return tuple(inverse(t) for t in reversed(w))

def word(text, generators=G):
    """把 '-f0.r2.d3' 切成合法记号元组"""
    if not isinstance(text, str):
        raise ValueError('A path must be text; use one ASCII space for zero moves')
    result = tuple(text.strip().split('.')) if text.strip() else ()
    if any(t not in generators for t in result):
        raise ValueError('Illegal or empty move token')
    return result

def replay(state, path, generators=G):
    """按顺序播放路线；每次从旧数组取新数组，绝不原地改写"""
    out = state.copy()
    for token in path:
        out = out[generators[token]]
    return out
```

接着我们核对手里的 36 个转动数组。「应用一个转动」的语义是 `state = state[perm]`，也就是新状态的第 `i` 位取自旧状态的第 `perm[i]` 位。我们用三条断言验证每个数组的属性：

```python
identity = np.arange(216)
assert len(G) == 36                                                  # 36 个带符号的层转
for token, permutation in G.items():
    # 每个数组都必须是 216 个位置的一个排列
    assert np.array_equal(np.sort(permutation), identity)
    # 正转接反转等于不动，说明 inverse 与数组配对正确
    assert np.array_equal(permutation[G[inverse(token)]], identity)
    # 连转四次回到原点，这是四分之一转的定义
    assert np.array_equal(permutation[permutation[permutation[permutation]]], identity)

# 目标必须是固定的 216 项颜色数组，不是"随便转一下都算复原"
assert GOAL.shape == (216,) and GOAL.max() == 5
```

36 个数组全部通过，说明动作集合自洽：没有写错的下标，逆动作互相配对，四分之一转的定义成立。到这一步，数据面和动作面都洗干净了。

### 2.3 模型训练与超参数调优

这个比赛没有可训练的模型（`trained_model=False`），所以这节讲的「模型」是**求解器**，「超参数」是**搜索预算**。我们沿用整系列的套路：先复现「词级平移」的构造，再做一个小而精确的搜索。

第一步是复现构造。6×6 和 4×4 不同，层号不是照搬，而是要查一张映射表：

```python
def lift_token(token):
    """把 3x3 记号换成 6x6 记号：层 0,1,2 映射到 0,2,5，符号取反"""
    axis, layer = token.lstrip('-')
    sign = '' if token.startswith('-') else '-'                    # 源正转 -> 目标反转
    mapped = sign + axis + str((0, 2, 5)[int(layer)])              # 层号查表
    assert mapped in G
    return (mapped,)

source_states = pd.read_csv(DATA / 'source333_test.csv', dtype=str, keep_default_na=False)
source_routes = pd.read_csv(DATA / 'source333_submission.csv', dtype=str, keep_default_na=False)
sources = dict(zip(source_states.initial_state_id, source_states.initial_state))
source_paths = dict(zip(source_routes.initial_state_id, source_routes.path))
G3 = {t: np.array(p, dtype=np.intp)
      for t, p in json.loads((DATA / 'source333_puzzle_info.json').read_text())['generators'].items()}

source_total = 0
for row in states.itertuples(index=False):
    # 1) 复现 3x3 的原状态和原路线，确认源文件可信
    original = np.array([int(x) for x in sources[row.initial_state_id].split(',')])
    source_path = word(source_paths[row.initial_state_id], G3)
    assert np.array_equal(replay(original, source_path, G3), np.arange(54))
    # 2) 换名后长度不变，且和公开参考解完全一致
    lifted = tuple(t for move in source_path for t in lift_token(move))
    assert lifted == word(refs[row.initial_state_id])
    assert len(lifted) == len(source_path)
    # 3) 输入 = 从复原态倒着走换名后的路线
    assert np.array_equal(read_state(row.initial_state), replay(GOAL, inverse_word(lifted)))
    source_total += len(source_path)

assert source_total == 21372
```

跑通之后我们确信：每条输入都确实由一条 21,372 步总长、单条最多 24 步的已知路线构造出来，「有解」这件事是可靠的。

第二步是求解器。我们采用**双向广度优先搜索**：从目标反向走 2 步建表，从每条输入正向走 2 步，两边碰头就把路线接上，总共最长 4 步。6×6 有 36 个动作，比 4×4 的 24 个更宽，同样的深度下状态数量更大，所以我们把预处理和查询分开计时，方便评估预算成本：

```python
TOKENS = tuple(sorted(G))
START_DEPTH = GOAL_DEPTH = 2     # 超参数：两个方向各走几步

def goal_table(depth):
    """从复原态 BFS 到 depth 步，记录每个可达状态的最短路线"""
    table = {GOAL.tobytes(): ()}
    queue = deque([(GOAL.copy(), ())])
    while queue:
        current, route = queue.popleft()
        if len(route) == depth:
            continue
        for token in TOKENS:
            if route and token == inverse(route[-1]):   # 立刻原路返回没意义，剪掉
                continue
            nxt = current[G[token]]
            key = nxt.tobytes()
            if key not in table:                        # BFS 首次到达即最短
                table[key] = route + (token,)
                queue.append((nxt, route + (token,)))
    return table

table_started = time.perf_counter()
BACKWARD = goal_table(GOAL_DEPTH)        # 预处理：只做一次，之后所有输入共用
table_seconds = time.perf_counter() - table_started
```

```python
def solve_bounded(initial):
    """从输入走 2 步，查反向表拼接，返回长度 ≤ 4 的解或 None"""
    queue = deque([(initial.copy(), ())])
    seen = {initial.tobytes()}
    best = None
    expanded = 0
    while queue:
        current, route = queue.popleft()
        expanded += 1
        back = BACKWARD.get(current.tobytes())
        if back is not None:
            candidate = route + inverse_word(back)       # 前半段正向 + 后半段反向逆
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
        assert len(best) <= START_DEPTH + GOAL_DEPTH     # 拼接结果长度有硬上限
        assert np.array_equal(replay(initial, best), GOAL)  # 拼出来的解必须真的解对
    return best, expanded
```

调参的道理和 4×4 相同：36 个动作让每深一层状态数最多乘 36，深度 2 是一千多个状态，深度 3 就到四万多，深度 4 已是百万量级。深度 2+2 是个人电脑上划算的预算。这个「≤ 4 步有解且最短」的保证能力很弱——绝大多数输入都远在 4 步之外——但它是现有算力下最稳的四步。

我们还额外用反向表自身做了一次自检：表里所有距离不超过 1 的状态，`solve_bounded` 都必须找回相同长度的解。这相当于单元测试，防止「建表逻辑」和「查询逻辑」各写各的、互相矛盾：

```python
# 表里的 BFS 路径是最短的，solve_bounded 必须恢复这些距离
checked = 0
for key, route in BACKWARD.items():
    if len(route) <= 1:
        solution, _ = solve_bounded(np.frombuffer(key, dtype=np.intp))
        assert solution is not None              # 距离 1 以内不可能解不出
        assert len(solution) == len(route)       # 长度必须和 BFS 距离一致
        checked += 1
print({'goal_table_states': len(BACKWARD), 'checked_near_goal': checked,
       'table_seconds': table_seconds})
```

### 2.4 模型测试与指标报告

测试的核心原则是**先量再退**：搜索绝不读参考解；每条输入先记录原始成功与否，再决定是否退回参考。我们把 `L_i - b_i`、搜索耗时、扩展状态数都写进逐条记录：

```python
records, answers = [], []
for row in states.itertuples(index=False):
    initial = read_state(row.initial_state)
    started = time.perf_counter()
    found, expanded = solve_bounded(initial)
    seconds = time.perf_counter() - started

    baseline = word(refs[row.initial_state_id])     # 参考路线只用于兜底和对比
    raw_success = found is not None
    selected = found if raw_success and len(found) < len(baseline) else baseline
    assert np.array_equal(replay(initial, selected), GOAL)   # 导出前逐条复核

    records.append({'initial_state_id': row.initial_state_id,
                    'initially_solved': bool(np.array_equal(initial, GOAL)),
                    'raw_success': raw_success,
                    'raw_moves': len(found) if raw_success else None,
                    'expanded_states': expanded, 'search_seconds': seconds,
                    'reference_moves': len(baseline), 'submitted_moves': len(selected),
                    'length_minus_reference': len(selected) - len(baseline),
                    'used_reference_fallback': not raw_success})
    answers.append({'initial_state_id': row.initial_state_id,
                    'path': '.'.join(selected) or ' '})
```

报告写成 JSON 而不是只打一行 print。它同时保留两个层面：**求解器自身的原始表现**和**完整提交的总步数**，二者绝不混为一谈：

```python
def validate(answer):
    """逐条重放，终点必须等于目标；通过校验就返回总步数"""
    ids = answer.initial_state_id.tolist()
    if len(ids) != len(set(ids)) or set(ids) != set(states.initial_state_id):
        raise ValueError('Every input ID must occur exactly once')
    routes = dict(zip(ids, answer.path))
    total = 0
    for row in states.itertuples(index=False):
        path = word(routes[row.initial_state_id])
        if not np.array_equal(replay(read_state(row.initial_state), path), GOAL):
            raise ValueError(f'Unsolved 216-facelet endpoint: {row.initial_state_id}')
        total += len(path)
    return total

results = pd.DataFrame(records)
answer = pd.DataFrame(answers, columns=['initial_state_id', 'path'])
reference_total = sum(len(word(p)) for p in reference.path)   # 参考解总步数
score = validate(answer)
answer.to_csv(OUTPUT / 'submission.csv', index=False,
              quoting=csv.QUOTE_ALL, lineterminator='\n')
exported = pd.read_csv(OUTPUT / 'submission.csv', dtype=str, keep_default_na=False)
assert validate(exported) == score        # 重新读回，确认空格没变成 NaN

report = {'competition': SLUG, 'rows': len(states),
          'reference_score': reference_total, 'score': score,
          'solver': 'exact bidirectional breadth-first search',
          'complete_through_depth': START_DEPTH + GOAL_DEPTH,
          'raw_solved': int(results.raw_success.sum()),
          'raw_failures': int((~results.raw_success).sum()),
          'initially_solved': int(results.initially_solved.sum()),
          'raw_solved_initially_unsolved': int((results.raw_success & ~results.initially_solved).sum()),
          'improved_inputs': int((results.length_minus_reference < 0).sum()),
          'saved_moves': reference_total - score,
          'uses_public_reference_fallback': True, 'trained_model': False}
(OUTPUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
```

看报告时有三个数要摆在一起读：

- `raw_solved`：在动用参考解之前，搜索自己解出了几条。因为预算只有 2+2，而输入都是十几到二十几步的构造产物，这个数会很小；
- `raw_failures`：回退发生了多少次。它**不是**错误，而是预算的边界——把失败如实留在这个字段里，正是材料强调的「honest failure signal」；
- `saved_moves`：相对参考省下的总步数，大概率是 0。这不丢人，它只是确认了「这个预算下的搜索打不过参考解」这个事实，给后续实验一个诚实的基线。

笔记本最后还会打一张交叉汇总表，行是「输入本来是否已复原」、列是「原始是否解出」，交叉格子里是输入条数和扩展状态总数。这张表能一眼看出两件事：有没有本来就不用解、却被算进成功率的输入；以及真正「本来没复原、又被自己解出来」的条数有多少（报告里对应 `raw_solved_initially_unsolved`）。把这张表留下来，下次加深搜索时就能直接看出多花的预算主要补上了哪一段。

下次换上更强的求解器（更大搜索、或带学习到的价值函数）时，接口不变、校验不变、报告字段不变，只有 `raw_solved` 和 `saved_moves` 变大——这才是这套框架最值钱的地方。

---

## 三、总结

本文中我们完成了 CubeLift 6×6 的完整流程。我们首先用 EDA 确认了数据规模：216 个贴纸、6 种颜色、每种 36 个、36 个带符号的层转。其次我们做了针对性的清洗：把「形状 216、颜色 0..5、每种 36 个」写进读取函数，用三条断言（是排列、逆配对、转四次回原点）验证了全部 36 个转动数组，并确认了 `state = state[perm]` 的取数语义。接着我们复现了偶数阶特有的「层映射表」：源层 `0、1、2` 映射到目标层 `0、2、5`，符号取反，逐条验证了 21,372 步、单条最长 24 步的参考解。然后我们用双向广度优先搜索（两个方向各 2 步，合起来保证长度 ≤ 4 的解）构建了求解器，并把预处理与查询分离计时、加上自检断言。最后我们把原始成功率、回退次数、相对参考步数差和总步数分开写进 `validation.json`，导出并复核提交文件。

这个比赛给我们的最大启发是：**搜索的原始覆盖率应该在回退之前测量，并在报告里独立存在。** 一个能通过全部校验的完整提交，并不证明求解器解出了每条输入；把 `raw_failures` 和 0 步的 `saved_moves` 如实摆出来，才让每个后续实验都有可比对的起点。