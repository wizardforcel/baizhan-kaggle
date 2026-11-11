# CubeLift 9x9

## 一、比赛简介

### 1.1 比赛背景

魔方求解是一个典型的**搜索问题**：给定一个打乱的状态（六个面颜色混杂），输出一串转动把它转回「每面同色」。3×3 魔方的状态数还有成熟算法可以处理；边长一旦放大，状态数会以指数级继续爆炸，穷举彻底没戏。所以近年的研究方向是**用机器学习给搜索提供方向感**——CubeLift 系列比赛就来自这条路线（材料提到的 CayleyPy 是一个把深度学习用在魔方与群论问题上的开源项目）。

CubeLift 的设计很省事：先给你一个 3×3 魔方上的**已知短解法**，再按固定规则把这条解法「搬运」到更大的魔方上，用搬运后的解法倒着走一遍，就造出了一道题。于是每道题的参考答案长度是已知的。

9×9 的情况是：**486 个面片、6 种颜色、54 个带符号的单层 90 度转动**。材料公开的参考解一共 **21,372 步**，单个输入最多 **24 步**——注意这和 8×8 那一场的总数完全一样，因为搬运是「一个 token 换一个 token」，长度守恒。

材料开篇的标题是 **"known routes, larger states"**（已知路线，更大的状态），副标题是 "Start with a short source route. Transfer its turns. Measure how well your solver finds a way back."（从一条短的源解法出发，搬运它的转动，测量你的求解器能把路找回来多少）。这句话点明了比赛的核心：**不训练模型，不加载预训练权重，不联网下载**，交付的是正确的转动语义、一个小而可靠的搜索、一个完整的校验器，以及一份诚实的标定报告。

### 1.2 比赛目标

9×9 魔方每面 81 格，6 种颜色各 81 个，一共 486 个面片。题目用逗号分隔的 486 个整数（取值 0～5）表示一个状态。

我们要交的是：每个 `initial_state_id` 对应一条能复原它的转动路线。

转动记号的规则和 8×8 一致：

| 记号 | 含义 |
| --- | --- |
| `f` / `r` / `d` | 沿 front / right / down 三个轴之一转动 |
| 后面的数字 `0`～`8` | 转第几层（9×9 每轴 9 层） |
| 前缀 `-` | 逆时针 90 度 |
| `.` | 转动之间的分隔符 |
| 空格 | 一步都不走（已经复原） |

三个轴 × 9 层 × 2 个方向 = **54 个合法转动**：`f0`～`f8`、`r0`～`r8`、`d0`～`d8` 加上各自的逆。材料还给了一条重要的计价规则：**半转算两个 token，同时转两个平行层也算两个 token，整体翻转魔方不是免费的**——所以想省步数，就只能老老实实用单层 90 度转。

### 1.3 评价标准

指标是**提交路线的总 token 数，越短越好**。官方参考解总长 21,372，单个输入最多 24 步。

对第 *i* 个输入，记我们提交的长度为 *L_i*、参考长度为 *b_i*，那么 *L_i* − *b_i* 表示「相对于这条已知可行路线多走了几步」。材料特别提醒：这个差值是**相对开销**，不是相对真正最优解的距离——真正的最优解没人知道。这个口径贯穿了我们的整个报告。

---

## 二、解题过程

我们的路线是：**先把 54 个转动数组和整个搬运规则验证到底，再用一个有明确深度上限的双向广度优先搜索去解题，最后把「搜索独立解出了多少」和「提交是否有效」分成两个数字报告。** 需要提前说明：这个比赛没有模型可训练，2.3 节的「模型」指的是搜索器。

### 2.1 数据探索（EDA）

我们首先自动定位数据目录。判断标准是「`central_state` 正好 486 个面片」——这样即使机器上同时挂着 3×3 和 8×8 的数据，也不会读错：

```python
from pathlib import Path
from collections import deque
import csv, hashlib, json, os, time
import numpy as np
import pandas as pd

SLUG = 'cubelift-9x9'
candidates = [Path('/kaggle/input/competitions') / SLUG,
              Path('/kaggle/input') / SLUG,
              Path('outputs/cubelift-9x9/data'), Path('data')]
# 只接受面片数正好 486 的那一份数据
DATA = next(p for p in candidates
            if (p / 'test.csv').is_file()
            and (p / 'lift_definition.json').is_file()
            and len(json.loads((p / 'puzzle_info.json').read_text())['central_state']) == 486)

info = json.loads((DATA / 'puzzle_info.json').read_text())
# 颜色值本身很小，用 int8 就够存
GOAL = np.array(info['central_state'], dtype=np.int8)
# 但转动数组是"位置索引"，必须用 np.intp，见下文
G = {t: np.array(p, dtype=np.intp) for t, p in info['generators'].items()}

states = pd.read_csv(DATA / 'test.csv', dtype=str, keep_default_na=False)
print({'inputs': len(states), 'facelets': len(GOAL), 'signed_moves': len(G)})
```

这里有一个**9×9 独有的坑**，值得单独讲：486 个位置超过了无符号 8 位整数能表示的上限 255。如果把转动数组存成 `uint8`，那么索引 256 以上的位置会**静默溢出**回 0～229 的范围，程序不报错，但转动结果全错。所以材料明确要求转动数组用 `np.intp`；而颜色值 0～5 本身很小，用 8 位存储完全够。换句话说：**存颜色可以省，存位置绝不能省**。

接下来我们确认颜色分布和复原态的形状：

```python
# 取第一行状态，切成整数数组
first = np.array([int(x) for x in states.initial_state.iloc[0].split(',')], dtype=np.int16)

# 9x9 每面 81 格，6 种颜色应该各出现 81 次
print(np.bincount(first, minlength=6))
# 输出：[81 81 81 81 81 81]

# 复原态是按颜色分块的 0..5，每块 81 个
assert np.array_equal(GOAL, np.repeat(np.arange(6), 81))

# 看转动记号：9 个层号，正反两个方向
print(sorted(G)[:6])
# ['-d0', '-d1', '-d2', '-d3', '-d4', '-d5']
```

这一步还揭示了一个和 8×8 不同的结构事实：**9 是奇数，所以 9×9 魔方真的有一个正中心层（第 4 层）**，而 8×8 没有。这个差异直接影响了搬运规则能选哪些层，下一节会讲。

### 2.2 数据清洗

数据本身干净，没有缺失和重复，所以我们的「清洗」其实全是**校验**。理由很简单：转动数组只要错一个位置，后面所有搜索都是在错误的世界里跑，越跑越错。我们先写工具函数：

```python
def inverse(token):
    """求逆转动：带 - 的去掉 -，不带 - 的加上 -"""
    return token[1:] if token.startswith('-') else '-' + token

def inverse_word(word):
    """整条路线的逆：每个转动取逆，再把顺序反过来"""
    return tuple(inverse(t) for t in reversed(word))

def replay(state, path, generators=G):
    """按路线转一遍：新状态 = 旧状态[这一步的置换数组]"""
    out = state.copy()
    for token in path:
        out = out[generators[token]]
    return out

def read_state(text):
    """把一行颜色串变成数组，并做三重合法性检查"""
    out = np.array([int(x) for x in text.split(',')], dtype=np.int16)
    if out.shape != (486,):                                  # 必须正好 486 个面片
        raise ValueError('Expected 486 facelets')
    if np.any(out < 0) or np.any(out > 5):                   # 颜色只能是 0..5
        raise ValueError('Expected colors in 0..5')
    # 每种颜色必须正好 81 个，否则状态物理上不可能存在
    if not np.array_equal(np.bincount(out, minlength=6), np.repeat(81, 6)):
        raise ValueError('Expected eighty-one of each color')
    return out.astype(np.int8)                               # 校验完再压成 int8
```

然后我们对 54 个转动逐一做三条断言：

```python
identity = np.arange(486)
for token, perm in G.items():
    # 1) 排序后等于 0..485：说明它是一个不含重复、不缺位置的置换
    assert np.array_equal(np.sort(perm), identity)
    # 2) 转一下再转它的逆，回到原样
    assert np.array_equal(perm[G[inverse(token)]], identity)
    # 3) 转四次回到原样（90 度转动的阶是 4）
    assert np.array_equal(perm[perm[perm[perm]]], identity)

# 9x9 的转动数组最大值必须正好是 485，再一次确认没被 uint8 截断
assert len(G) == 54
assert all(g.dtype == np.dtype(np.intp) and int(g.max()) == 485 for g in G.values())
```

最后那句 `int(g.max()) == 485` 是专门用来抓溢出 bug 的——如果某个数组被存成了 `uint8`，它的最大值会变成 229 之类的奇怪数字，这条断言会立刻失败。

### 2.3 模型训练与超参数调优

这个比赛没有可训练的模型，我们的「模型」是搜索器。造搜索器之前，先要搞清题目是怎么生成的——因为**题目是答案的逆**。

搬运规则是这样的：3×3 的层 0、1、2 分别映射到 9×9 的 **0、4、8** 层（从 1 开始数是第 1、5、9 层），并且**符号取反**：`a0 → -a0`、`a1 → -a4`、`a2 → -a8`；如果源 token 本身带 `-`，符号再翻回来。

```python
def lift_token(token):
    """3x3 的一个转动 -> 9x9 的一个物理转动，长度不变"""
    plain = token.lstrip('-')
    axis, source_layer = plain[0], int(plain[1:])
    target_layer = {0: 0, 1: 4, 2: 8}[source_layer]     # 源 0/1/2 层 -> 目标 0/4/8 层
    sign = '' if token.startswith('-') else '-'         # 符号取反
    mapped = sign + axis + str(target_layer)
    assert mapped in G                                  # 搬过去必须仍然合法
    return (mapped,)
```

这里有个关键点：**3×3 的中间层（第 1 层）搬到 9×9 的第 4 层以后，仍然只是一个单层转动，不会被展开成一个「宽转」**。一个源 token 就换一个目标 token，长度严格守恒，所以参考解总长还是 21,372，最长 24 步。

这也意味着 9×9 的 9 个层里，只有 0、4、8 三层在生成时用过，**1、2、3、5、6、7 这六层在题目生成时从未出现，但对求解器完全开放**。这是一个可以利用的信息：如果一条解法里出现了这些「未用层」，说明求解器真正在目标魔方上做搜索，而不是死记搬运规则。

我们把整条链路验一遍：

```python
sources = dict(zip(source_states.initial_state_id, source_states.initial_state))
source_paths = dict(zip(source_routes.initial_state_id, source_routes.path))
refs = dict(zip(reference.initial_state_id, reference.path))

source_total = 0
for row in states.itertuples(index=False):
    original = np.array([int(x) for x in sources[row.initial_state_id].split(',')])
    source_path = word(source_paths[row.initial_state_id], G3)   # 3x3 官方解法
    # 1) 源解法在源魔方上确实能复原
    assert np.array_equal(replay(original, source_path, G3), np.arange(54))
    # 2) 搬运后与官方参考解逐 token 一致，且长度不变
    lifted = tuple(t for move in source_path for t in lift_token(move))
    assert lifted == word(refs[row.initial_state_id])
    assert len(lifted) == len(source_path)
    # 3) 题目构造：从复原态倒走搬运后的解法，得到输入的初始状态
    assert np.array_equal(read_state(row.initial_state),
                          replay(GOAL, inverse_word(lifted)))
    source_total += len(source_path)

assert source_total == 21372
assert max(len(word(p)) for p in reference.path) == 24
```

验证通过后我们手里就有一条可靠的兜底路线。但比赛要求**先测量搜索、再使用参考**，所以兜底只用于保证提交有效，不参与「搜索能力」的计分。

我们的搜索器是**双向广度优先搜索**。从复原态出发把所有「两步以内」可达的状态和路线存成一张表；对每道题，从初始状态出发也展开两步以内，一旦某个状态在表里出现就说明两头接上了，整条路线长度不超过 4 步。因为两边都只展开两层，这个搜索对「长度 ≤ 4 的解」是**完备**的：

```python
TOKENS = tuple(sorted(G))
START_DEPTH = GOAL_DEPTH = 2

def goal_table(depth):
    """从复原态广度优先展开 depth 层，返回 {状态字节串: 到达路线}"""
    table = {GOAL.tobytes(): ()}
    queue = deque([(GOAL.copy(), ())])
    while queue:
        current, route = queue.popleft()
        if len(route) == depth:
            continue
        for token in TOKENS:
            if route and token == inverse(route[-1]):   # 立刻转回去是浪费，剪掉
                continue
            nxt = current[G[token]]
            key = nxt.tobytes()                          # 字节串当字典键，查得快
            if key not in table:
                path = route + (token,)
                table[key] = path
                queue.append((nxt, path))
    return table

BACKWARD = goal_table(GOAL_DEPTH)
```

正向那一半边展开边查表：

```python
def solve_bounded(initial):
    """正向展开两层，一路查后表；接上就拼成完整路线，并校验它真能复原"""
    queue = deque([(initial.copy(), ())])
    seen = {initial.tobytes()}
    best, expanded = None, 0                     # expanded 记录搜索成本
    while queue:
        current, route = queue.popleft()
        expanded += 1
        back = BACKWARD.get(current.tobytes())
        if back is not None:
            # 后表是从复原态出发的，拼接时要取逆
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
    if best is not None:                         # 找到就必须真的能复原
        assert np.array_equal(replay(initial, best), GOAL)
    return best, expanded
```

**关于调参。** 唯一的超参数就是两个深度。我们把它们都设成 2：54 个转动从一端展开两层，去掉「刚转完又立刻转回去」的重复，大约是 1 + 54 + 54×53 ≈ 2900 个状态，建表几乎瞬间完成；再往上加一层会乘上约 52，涨到十五万量级，而题目最长有 24 步，多这一层换来的覆盖率提升很有限。所以这个搜索器的定位是**标定用的最小搜索**——它有正确的展开、正确的逆路线拼接和明确的失败信号，但它不是实用的 9×9 求解器。材料也说了，把 `solve_bounded` 换成一个真正的求解器时，输入、校验、报告这三块代码一行都不用改。

### 2.4 模型测试与指标报告

主循环把「搜索的原始结果」和「最终提交」分开记录：

```python
records, answers = [], []
for row in states.itertuples(index=False):
    initial = read_state(row.initial_state)
    found, expanded = solve_bounded(initial)          # 搜索器只看输入
    baseline = word(refs[row.initial_state_id])       # 参考解此时才登场
    raw_success = found is not None
    # 搜索成功且比参考短就用自己的，否则用参考兜底
    selected = found if raw_success and len(found) < len(baseline) else baseline
    assert np.array_equal(replay(initial, selected), GOAL)   # 最终答案必须有效
    records.append({'initial_state_id': row.initial_state_id,
                    'initially_solved': bool(np.array_equal(initial, GOAL)),
                    'raw_success': raw_success,
                    'raw_moves': len(found) if raw_success else None,
                    'expanded_states': expanded,
                    'reference_moves': len(baseline),
                    'submitted_moves': len(selected),
                    'length_minus_reference': len(selected) - len(baseline),
                    'used_reference_fallback': not raw_success})
    answers.append({'initial_state_id': row.initial_state_id,
                    'path': '.'.join(selected) or ' '})     # 空路线写成空格
```

两个细节我们特意处理了。第一，空路线必须写成**一个空格**——如果写成空字符串，读回来会被当成缺失值。第二，**「提交有效」不等于「搜索解出」**：有参考解兜底，提交可以 100% 有效，但搜索的成功率可能很低，报告里这两个数字必须分开列。

写文件时强制加引号，写完再读回来二次校验：

```python
answer.to_csv(OUTPUT / 'submission.csv', index=False,
              quoting=csv.QUOTE_ALL, lineterminator='\n')
# 重新打开真正的产物：空格不能被读成 NaN
exported = pd.read_csv(OUTPUT / 'submission.csv', dtype=str, keep_default_na=False)
assert validate(exported) == score

report = {'competition': SLUG, 'facelets': len(GOAL), 'signed_moves': len(G),
          'reference_score': reference_total, 'score': score,
          'solver': 'exact bidirectional breadth-first search',
          'complete_through_depth': START_DEPTH + GOAL_DEPTH,
          'raw_solved': int(results.raw_success.sum()),
          'raw_failures': len(states) - int(results.raw_success.sum()),
          'saved_moves': reference_total - score,
          'uses_public_reference_fallback': True, 'trained_model': False,
          'sha256': hashlib.sha256((OUTPUT / 'submission.csv').read_bytes()).hexdigest()}
(OUTPUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
```

我们最后会同时看三份东西：`submission.csv`（提交物）、`per_input_results.csv`（逐题明细：展开了多少状态、耗时、比参考长还是短）、`validation.json`（总报告）。材料的口径很清楚：**排行榜分数包含公开参考解兜底，衡量的是提交质量而非搜索能力**；要看搜索能力，就看 `raw_solved` 和 `raw_failures`。

还有一点材料主动披露的**来源诚实性**问题，值得记一笔：9×9 的转动数组是用**未经修改的官方 CayleyPy 源函数**生成的，提交记录在 `generator_provenance.json` 里；同一个函数生成的结果与下载到的原版 4×4、5×5 比赛中的每一个正方向数组都吻合；但**原始的 9×9 比赛文件本身没有被独立下载核对过**。这种「说清楚哪部分验证过、哪部分没有」的做法，正是这个比赛想教的。

---

## 三、总结

本文中我们完成了 CubeLift 9x9 的完整流程。我们首先通过 EDA 确认了题目规模：486 个面片、6 种颜色各 81 个、54 个合法转动，并发现 486 超过 8 位无符号整数上限这个隐藏陷阱，据此决定转动数组用 `np.intp`、颜色用 8 位。其次我们做了全量校验：状态长度、颜色计数、转动数组的置换性质、逆转动关系和「转四次回原样」，还专门加了一条最大值必须等于 485 的断言来抓溢出。接着我们理清了搬运规则：3×3 的第 0、1、2 层映射到 9×9 的 0、4、8 层并取反，中间层不会被扩成宽转，一个 token 换一个 token，所以参考解仍是 21,372 步、最长 24 步，同时 1、2、3、5、6、7 层在生成时未用、对求解器开放。然后我们用双向广度优先搜索各展开两层，保证对长度 ≤ 4 的解完备，并记录每题展开的状态数和成败。最后我们把搜索成功且更短的路线与参考解兜底合并、写盘、读回二次校验，并输出一份把搜索表现与提交表现分开的报告。

这个比赛和 8×8 那一场最大的区别不在规模，而在**奇偶性带来的结构差异**：奇数阶有真正的中心层，偶数阶没有，所以搬运只能挑三个边缘附近的层；再加上 486 这个数字跨过了 8 位整数边界，一个不起眼的 `dtype` 选择就能让整个程序静默出错。这提醒我们：**在魔方这类「状态即数组」的问题里，类型和索引的边界比算法更先决定成败。**
