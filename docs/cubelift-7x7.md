# CubeLift 7x7：已知路线、更大的状态

## 一、比赛简介

### 1.1 比赛背景

CubeLift 7×7 是整个系列里状态最大的版本：**294 个贴纸（facelet）**、6 种颜色、**42 个带符号的单层四分之一转**。它的数据来源仍然是 3×3：从一条已知的短源路线出发，把每个动作逐个换名，再把换好的路线倒过来作用在已复原的 7×7 上，就产生一条输入。所以题目的核心不是「从零开始打乱」，而是「已知一条短路线，看你能否用自己的求解器找回一条同样短、甚至更短的路」。

换名规则在这一版也很具体。源 3×3 的三层 `0、1、2` 在 7×7 上变成 `0、3、6`（从 1 开始数是第 1、4、7 层），对每个轴 `a`（f、r、d）：

```
a0 → -a0
a1 → -a3
a2 → -a6
```

符号取反是这套数组约定的一部分。**一个源动作换成一个目标动作，长度完全不变**——源里的一个中间层转动，不会被展开成一个「宽转」。因此参考解的总步数和源解一样，是 **21,372 步**，单条最多 **24 步**。

造数据只用了层 `0、3、6`，层 `1、2、4、5` 在生成时完全没用到，但解题时全部合法。计步规则和系列其他版本一致：半转算 2 个 token，同时转两个平行层也算 2 步，整魔方换朝向不免费，目标就是那个固定的 294 项颜色数组。

### 1.2 比赛目标

`test.csv` 每行是一个 `initial_state_id` 加 **294 个**逗号分隔的颜色编号（0～5）。我们要填 `path` 列：句点分隔的动作记号。合法动作 42 个：`f0`～`f6`、`r0`～`r6`、`d0`～`d6` 与各自的负号逆动作。零步输入写一个空格，不能留空。

有一个容易忽略的工程细节：294 超过了 8 位无符号整数的上限 255，所以**动作排列数组不能用 `uint8` 存下标**，否则下标会被截断、转动直接错位。颜色值本身只有 0～5，仍然可以用 8 位存。我们后面会用一句断言专门守住这件事。

### 1.3 评价标准

排行榜把所有提交路径长度加起来，**越小越好**。参考解 21,372 步、单条最长 24 步。校验是零容忍的：每条输入的终点都必须严格等于目标状态。

材料在指标口径上给了一个明确的定义：对每条输入 $i$，记提交长度为 $L_i$、已知参考长度为 $b_i$，那么 $L_i - b_i$ 是「相对这条已知可行路线上多走几步」。它**不是**离未知最优解的精确距离——要知道真实最短距离得把整个状态空间搜穿，做不到。另外，本身已经复原的输入（$b_i = 0$）要留在覆盖率和最终计分里，但要从 $L/b$ 这类比值统计里排除。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把数据读进来。7×7 的数据目录靠 294 这个数字来确认：

```python
from pathlib import Path
from collections import deque
import csv, json, os, time
import numpy as np
import pandas as pd

SLUG = 'cubelift-7x7'
candidates = [Path('/kaggle/input/competitions') / SLUG,
              Path('/kaggle/input') / SLUG,
              Path('data')]
DATA = next(p for p in candidates
            if (p / 'test.csv').is_file()
            and len(json.loads((p / 'puzzle_info.json').read_text())['central_state']) == 294)

# 产物统一写到输出目录，Kaggle 上就是 /kaggle/working
OUTPUT = Path(os.environ.get('CUBELIFT777_OUTPUT_DIR',
              '/kaggle/working' if Path('/kaggle').exists() else 'cubelift777-output'))
OUTPUT.mkdir(parents=True, exist_ok=True)

info = json.loads((DATA / 'puzzle_info.json').read_text())
GOAL = np.array(info['central_state'], dtype=np.int8)                  # 294 个颜色，值 0..5
G = {t: np.array(p, dtype=np.intp) for t, p in info['generators'].items()}

states = pd.read_csv(DATA / 'test.csv', dtype=str, keep_default_na=False)
reference = pd.read_csv(DATA / 'sample_submission.csv', dtype=str, keep_default_na=False)
refs = dict(zip(reference.initial_state_id, reference.path))
print({'inputs': len(states), 'facelets': len(GOAL), 'signed_moves': len(G)})
```

打印结果确认 `facelets=294`、`signed_moves=42`。接着看目标数组和颜色计数：

```python
# 目标是固定数组：0..5 每种颜色 49 个，按面分块
assert np.array_equal(GOAL, np.repeat(np.arange(6, dtype=np.int8), 49))

# 看一条输入：294 个数，颜色计数必须是 49 个一组
state = np.array([int(x) for x in states.initial_state.iloc[0].split(',')], dtype=np.int8)
print(state.shape, np.bincount(state, minlength=6))
# 输出：(294,)  [49 49 49 49 49 49]

# 参考解的长度信息：单条最长与总步数
lengths = [len(p.split('.')) if p.strip() else 0 for p in reference.path]
print({'max_reference_moves': max(lengths), 'reference_total': sum(lengths)})
# 输出：{'max_reference_moves': 24, 'reference_total': 21372}
```

颜色守恒在这里同样成立。294 除以 6 正好是 49，说明六种颜色在这份数据里是平均分配的，每种各占 49 个贴纸。

### 2.2 数据清洗

这份数据没有缺失值、没有重复行，需要「清洗」的仍是两件事：把坏输入挡在门外，和把动作数组的类型钉死。

第一件，读取和路径解析。我们沿用系列里的做法：形状、颜色范围、颜色计数三项一起查：

```python
def read_state(text):
    """读一条状态并检查：294 个、颜色 0..5、每种颜色 49 个"""
    out = np.array([int(x) for x in text.split(',')], dtype=np.int16)
    if out.shape != (294,) or np.any(out < 0) or np.any(out > 5):
        raise ValueError('Expected 294 colors in 0..5')
    if not np.array_equal(np.bincount(out, minlength=6), np.repeat(49, 6)):
        raise ValueError('Expected forty-nine occurrences of each color')
    return out.astype(np.int8)      # 校验完再压成 int8，颜色值只占 8 位

def inverse(token):
    return token[1:] if token.startswith('-') else '-' + token

def inverse_word(w):
    return tuple(inverse(t) for t in reversed(w))

def word(text, generators=G):
    """把 '-f0.r3.d6' 切成合法记号元组，零步写成空元组"""
    if not isinstance(text, str):
        raise ValueError('A path must be text; use one ASCII space for zero moves')
    result = tuple(text.strip().split('.')) if text.strip() else ()
    if any(t not in generators for t in result):
        raise ValueError('Illegal or empty move token')
    return result

def replay(state, path, generators=G):
    out = state.copy()
    for token in path:
        out = out[generators[token]]     # 用新数组承接，不能原地改
    return out
```

第二件，动作数组的类型检查。这是我们这一版专门补上的一条，因为 294 个位置的下标范围是 0～293，如果排列数组被存成 `uint8`，下标超过 255 的部分会回绕，转动就变成「转错了位置」——这种错误不会报错，只会让终点差几个贴纸，非常难查：

```python
identity = np.arange(294)
assert len(G) == 42                                   # 42 个带符号的单层四分之一转

for token, permutation in G.items():
    assert np.array_equal(np.sort(permutation), identity)             # 是 294 个位置的排列
    assert np.array_equal(permutation[G[inverse(token)]], identity)   # 逆动作配对正确
    # 连转四次回到原位，这是"四分之一转"的定义
    assert np.array_equal(permutation[permutation[permutation[permutation]]], identity)
    # 下标必须能安全容纳 0..293：一定要 intp，绝不能是 uint8
    assert permutation.dtype == np.dtype(np.intp) and int(permutation.max()) == 293
```

这五条断言一起把「动作集合是自洽的、而且索引不会溢出」这件事锁死了。之后再怎么做搜索，我们都不用担心是数据本身有问题。

### 2.3 模型训练与超参数调优

仍然先说明：这个比赛没有训练、没有预训练权重、不需要联网。所以「模型」指的是**求解器**，「超参数」指的是**搜索深度和预算**。

我们做的第一件事是复现构造过程，确认「每条输入真的有一条已知短解」：

```python
def lift_token(token):
    """3x3 记号 -> 7x7 记号：层 0,1,2 映射到 0,3,6，符号取反"""
    plain = token.lstrip('-')
    axis, source_layer = plain[0], int(plain[1:])
    target_layer = {0: 0, 1: 3, 2: 6}[source_layer]     # 层号查表，不按比例缩放
    sign = '' if token.startswith('-') else '-'        # 源正转 -> 目标反转
    mapped = sign + axis + str(target_layer)
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
    # 先把 3x3 的原状态和原路线复现，确认源文件本身没问题
    original = np.array([int(x) for x in sources[row.initial_state_id].split(',')])
    source_path = word(source_paths[row.initial_state_id], G3)
    assert np.array_equal(replay(original, source_path, G3), np.arange(54))

    # 换名：一个源 token 换成一个目标 token，长度不变
    lifted = tuple(t for move in source_path for t in lift_token(move))
    assert len(lifted) == len(source_path)
    assert lifted == word(refs[row.initial_state_id])

    # 输入 = 从复原态出发，倒着走换名后的路线
    assert np.array_equal(read_state(row.initial_state), replay(GOAL, inverse_word(lifted)))
    source_total += len(source_path)

assert source_total == 21372
```

这一步跑完，我们不仅确认了参考解合法，还确认了「层号是查表映射、不是等比缩放」这个关键约定——如果按 7/3 的比例去放，`a1` 会落到层 2、`a2` 落到层 4 附近，而不是约定的层 3 和层 6，整条构造就对不上了。

第二件事是写求解器。我们依然用**双向宽度优先搜索**：从目标反向走两步建表，从输入正向走两步，碰头就拼接，保证长度不超过 4 的解都能找到、而且最短。

```python
TOKENS = tuple(sorted(G))
START_DEPTH = GOAL_DEPTH = 2      # 超参数：两个方向各 2 步，合计保证 ≤ 4 步

def goal_table(depth):
    """从复原态 BFS 到 depth 步，记下每个可达状态的最短路线"""
    table = {GOAL.tobytes(): ()}
    queue = deque([(GOAL.copy(), ())])
    while queue:
        current, route = queue.popleft()
        if len(route) == depth:
            continue
        for token in TOKENS:
            if route and token == inverse(route[-1]):    # 立刻走回头路，等价于空转
                continue
            nxt = current[G[token]]
            key = nxt.tobytes()
            if key not in table:                         # BFS 第一次到达就是最短路
                table[key] = route + (token,)
                queue.append((nxt, route + (token,)))
    return table

BACKWARD = goal_table(GOAL_DEPTH)
```

```python
def solve_bounded(initial):
    """从输入走 2 步查表拼接；解不出返回 None，这是我们唯一的失败信号"""
    queue = deque([(initial.copy(), ())])
    seen = {initial.tobytes()}
    best, expanded = None, 0
    while queue:
        current, route = queue.popleft()
        expanded += 1
        back = BACKWARD.get(current.tobytes())
        if back is not None:
            candidate = route + inverse_word(back)    # 正向段 + 反向段的逆
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
        assert len(best) <= START_DEPTH + GOAL_DEPTH
        assert np.array_equal(replay(initial, best), GOAL)   # 拼出来必须真解对
    return best, expanded
```

超参数的意义和前面几版一样：42 个动作意味着每深一层状态数最多乘 42——深度 2 是一千多个，深度 3 七万多，深度 4 三百多万。所以 2+2 是普通环境里划算的预算。它的能力边界也很清楚：只覆盖长度不超过 4 的解，而输入都是由十几到二十几步的构造路线产生的，绝大多数输入落在邻域之外。

为了确认建表和查询没有各写各的，我们又加了一段自检：反向表里距离不超过 1 的状态，`solve_bounded` 必须找回同等长度的解。

```python
# 表里的路线是 BFS 最短的，查询函数必须能恢复这些距离
checked = 0
for key, route in BACKWARD.items():
    if len(route) <= 1:
        solution, _ = solve_bounded(np.frombuffer(key, dtype=np.int8))
        assert solution is not None                     # 距离 1 以内不可能解不出
        assert len(solution) == len(route)              # 长度必须和 BFS 距离一致
        checked += 1

# 这张表是后面所有输入共用的，先把它的大小和自检条数打出来存档
print({'goal_table_states': len(BACKWARD), 'checked_near_goal': checked})
```

### 2.4 模型测试与指标报告

测试时我们坚持一个顺序：**先记录原始结果，再决定回退**。搜索代码从头到尾不读参考解，参考只在比较和兜底时出现：

```python
records, answers = [], []
for row in states.itertuples(index=False):
    initial = read_state(row.initial_state)
    started = time.perf_counter()
    found, expanded = solve_bounded(initial)
    seconds = time.perf_counter() - started

    baseline = word(refs[row.initial_state_id])
    raw_success = found is not None
    # 自己的解有效且更短才用；否则用参考，保证提交完整
    selected = found if raw_success and len(found) < len(baseline) else baseline
    assert np.array_equal(replay(initial, selected), GOAL)

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

注意 `raw_moves` 和 `submitted_moves` 是两个字段，不能合成一个：前者是求解器自己的成绩，后者是最终提交的成绩。把两者混在一起，报告就失去了意义。

最后是校验和导出。我们写文件之后**重新读回来再验一遍**，因为零步答案写的是空格，pandas 默认会把空格读成缺失值，行数或内容可能悄悄变掉：

```python
def validate(answer):
    """逐条检查编号与终点，通过就返回总步数"""
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
            raise ValueError(f'Unsolved 294-facelet endpoint: {row.initial_state_id}')
        total += len(path)
    return total

results = pd.DataFrame(records)
answer = pd.DataFrame(answers, columns=['initial_state_id', 'path'])
reference_total = sum(len(word(p)) for p in reference.path)   # 参考解总步数
score = validate(answer)
answer.to_csv(OUTPUT / 'submission.csv', index=False,
              quoting=csv.QUOTE_ALL, lineterminator='\n')
exported = pd.read_csv(OUTPUT / 'submission.csv', dtype=str, keep_default_na=False)
assert validate(exported) == score

report = {'competition': SLUG, 'rows': len(states), 'facelets': len(GOAL),
          'signed_moves': len(G), 'reference_score': reference_total, 'score': score,
          'complete_through_depth': START_DEPTH + GOAL_DEPTH,
          'raw_solved': int(results.raw_success.sum()),
          'raw_failures': int((~results.raw_success).sum()),
          'improved_inputs': int((results.length_minus_reference < 0).sum()),
          'saved_moves': reference_total - score,
          'uses_public_reference_fallback': True, 'trained_model': False}
```

报告里最该看的是三项：`raw_solved`（回退前自己解出几条）、`raw_failures`（退了几次）、`saved_moves`（相对参考总共省几步）。由于预算只有 4 步、输入长度普遍在十几步以上，`raw_solved` 会很小、`saved_moves` 大概率是 0。这正是我们想留下的**诚实刻度**：完整提交能过校验，不等于求解器独立解出了全部输入。

关于数据来源也要交代一句。7×7 的动作数组是用官方 CayleyPy 源码函数（记录在 `generator_provenance.json` 的提交号里）生成的，但材料也实事求是地说明：原版 7×7 比赛的文件并没有被独立下载对照，只是同一个生成函数能对上原版 4×4 和 5×5 的各项正数组。所以这份 7×7 数据的可信度建立在「生成函数一致」上，而不是建立在原文件比对一致上。把这一点明明白白交代出来，比假装它已经被完整验证要可信得多。

---

## 三、总结

本文中我们完成了 CubeLift 7×7 的完整流程。我们首先用 EDA 确认了数据规模：294 个贴纸、6 种颜色、每种 49 个、42 个带符号的层转。其次我们做了两层清洗：读入时检查形状、颜色范围与颜色计数；动作上验证每个数组是 294 位的排列、逆动作配对、四转复原，并额外断言下标类型必须是 `np.intp`，因为 294 已经超过 8 位无符号整数的范围，用 `uint8` 存下标会静默地把转动算错。接着我们复现了层映射表（源层 `0、1、2` → 目标层 `0、3、6`，符号取反），逐条验证了 21,372 步、单条最长 24 步的参考解。然后我们用双向广度优先搜索（两个方向各 2 步，保证长度 ≤ 4 的解最短）构建求解器，并用反向表做距离自检。最后我们把原始成功率、回退次数、相对参考步数差与最终总步数分开报告，导出并重新读回校验提交文件。

这个比赛给我们的最大启发是：**规模变大时，最先出问题的往往不是算法，而是数据类型的边界。** 一个 255 之后的下标回绕不会抛异常，只会让终点差几个贴纸；只有把「排列是排列、下标是 intp、逆动作配对、四转复原」这些约束写成断言，才能在小搜索之前先保证大魔方本身是对的。随后的一切——预算、覆盖率、步数差——才有意义。
