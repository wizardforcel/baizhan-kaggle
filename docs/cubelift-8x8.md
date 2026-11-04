# CubeLift 8x8

## 一、比赛简介

### 1.1 比赛背景

魔方（Rubik's Cube）大家都见过：一个 3×3×3 的立方体，六个面各有一种颜色，拧乱之后要靠一串转动把它转回「每面同色」的样子。从数学上说，这是一个**搜索问题**——给定一个打乱的状态，找出一串转动把状态变回复原态。

3×3 魔方的状态数虽然也很吓人，但还有成熟的算法可以处理。把边长放大到 8×8 以后，状态数会以指数级继续膨胀，穷举就彻底不可能了。所以这个领域近年来的主流思路是：**用机器学习给搜索提供「方向感」**，让程序不要瞎试。CubeLift 系列比赛就来自这个研究方向（材料中提到的 CayleyPy 就是一个把深度学习用在魔方、群论这类组合问题上的开源项目）。

CubeLift 的做法很有意思：它不是让你从零解一个 8×8 魔方，而是先给你一个 3×3 魔方上的**已知短解法**，再按一套固定规则把这条解法「搬运」到 8×8 上，用搬运后的解法去构造题目。于是每道题的答案长度是已知的——材料里公开的参考解一共 **21,372 步**，单个输入最多 **24 步**。

这里要特别强调材料开篇那句话：**"measure the search, not just the final CSV"**（测量你的搜索，而不只是交一个 CSV）。也就是说，比赛关心的不只是你最后交上去的路线有多短，还关心你的搜索真正独立解出了多少题、花了多少计算量。这个比赛**不训练任何模型**，不加载预训练权重，也不需要联网——它的核心交付物是「正确的转动语义 + 一个可测试的小搜索器 + 一个完整的校验器 + 一份诚实的标定报告」。

### 1.2 比赛目标

8×8 魔方有 384 个「面片」（facelet，就是贴在表面上的小色块），6 种颜色各 64 个。题目用一串 0～5 的整数表示一个状态，按顺序列出全部 384 个位置的颜色。

我们要交的东西很简单：对每个输入状态，给出一条能把它转回复原态的**转动序列**。

转动记号的规则是：

| 记号 | 含义 |
| --- | --- |
| `f` / `r` / `d` | 沿三个轴之一转动（front / right / down） |
| 后面的数字 `0`～`7` | 转第几层（8×8 魔方每轴有 8 层） |
| 前缀 `-` | 逆时针转 90 度；不带 `-` 是顺时针 |
| `.` | 两个转动之间的分隔符 |
| 空格 | 表示这一步都不走（已经复原） |

比如 `f0.r3.-d7` 表示「f 轴第 0 层顺时针 → r 轴第 3 层顺时针 → d 轴第 7 层逆时针」。三个轴 × 8 层 × 2 个方向 = **48 个合法转动**。

数据目录里一共有这些文件：

- `test.csv`：每行一个 `initial_state_id` 和对应的 `initial_state`（384 个逗号分隔的整数）；
- `sample_submission.csv`：官方公开的参考解，列是 `initial_state_id` 和 `path`；
- `puzzle_info.json`：描述复原态 `central_state` 和 48 个转动对应的置换数组；
- `lift_definition.json`：描述 3×3 到 8×8 的搬运规则；
- `source333_test.csv`、`source333_submission.csv`、`source333_puzzle_info.json`：原始的 3×3 题目、解法和转动定义；
- `provenance.json`：记录这些数组是从哪个官方来源生成的。

### 1.3 评价标准

评价指标是**提交路线的总长度**，按 token 数计算，**越短越好**。一条路线里有几个转动就是几个 token，半转（180 度）算两个 token，同时转两层也算两个 token——换句话说，只有「单层 90 度」才划算，这直接决定了我们后面只用 48 个单层转动。

材料给出了一个可以直接对照的基准：参考解总长 21,372，单个输入最多 24 步。对第 *i* 个输入，记我们的长度是 *L_i*、参考长度是 *b_i*，那么 *L_i* − *b_i* 就是「相对于这条已知可行路线多走了多少」。注意这个差值**不是**相对于真正最优解的差距（真最优解谁也不知道），只是相对于一条已知可行路线的开销，这一点材料反复强调，我们后面写报告时也照这个口径。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**先把魔方的转动语义和搬运规则彻底验证正确，再用一个「有明确能力边界」的双向广度优先搜索去解题，最后诚实地报告搜索独立解出了多少、失败了多少。** 我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。需要提前说明的是，这个比赛没有「模型」可训练，2.3 节里的「模型」指的是我们的**搜索器**。

### 2.1 数据探索（EDA）

第一步是把数据找出来并读进来。Kaggle 的数据路径有好几种可能，我们写了一段自动探测的代码，并且用「面片数必须正好是 384」作为判据，防止误读到 9×9 或者 3×3 的数据：

```python
from pathlib import Path
from collections import deque
import csv, hashlib, json, os, time
import numpy as np
import pandas as pd

# 数据可能被挂在几个不同的路径下，我们逐个试探
# 只接受同时含有 test.csv、lift_definition.json，
# 并且 central_state 正好 384 个面片的那一份，避免读错比赛
SLUG = 'cubelift-8x8'
candidates = [Path('/kaggle/input/competitions') / SLUG,
              Path('/kaggle/input') / SLUG,
              Path('outputs/cubelift-8x8/data'), Path('data')]
DATA = next(p for p in candidates
            if (p / 'test.csv').is_file()
            and (p / 'lift_definition.json').is_file()
            and len(json.loads((p / 'puzzle_info.json').read_text())['central_state']) == 384)

info = json.loads((DATA / 'puzzle_info.json').read_text())
GOAL = np.array(info['central_state'], dtype=np.intp)                       # 复原态
G = {t: np.array(p, dtype=np.intp) for t, p in info['generators'].items()}  # 48 个转动

states = pd.read_csv(DATA / 'test.csv', dtype=str, keep_default_na=False)
print({'inputs': len(states), 'facelets': len(GOAL), 'signed_moves': len(G)})
```

读进来之后，我们先确认三件事：面片总数是 384、转动总数是 48、颜色分布均匀。颜色分布这一步很关键——如果某个状态里某种颜色出现了 65 次，那它在物理上根本不可能存在，说明数据被污染了：

```python
# 把第一行状态从 "0,1,2,..." 的字符串切成整数数组
first = np.array([int(x) for x in states.initial_state.iloc[0].split(',')], dtype=np.intp)

# 8x8 魔方每个面 64 格，6 种颜色应该各出现 64 次
print(np.bincount(first, minlength=6))
# 输出：[64 64 64 64 64 64]

# 复原态就是按颜色分块排列的 0..5，每块 64 个
assert np.array_equal(GOAL, np.repeat(np.arange(6), 64))

# 看一眼转动记号的命名规律：三个轴、8 层、正反两个方向
print(sorted(G)[:8])
# ['-f0', '-f1', '-f2', '-f3', '-f4', '-f5', '-f6', '-f7']
```

这一步还告诉我们一个隐含事实：**8×8 是偶数阶魔方，没有单独的中心层**。所以搬运规则里只能挑三个「离边缘有距离」的层来做映射，这一点在 2.3 节会具体讲。

### 2.2 数据清洗

这份数据没有缺失值、没有重复行，传统意义上的「清洗」几乎不用做。真正的清洗工作在于**验证**：魔方的转动数组只要错一位，后面所有搜索都是白费。所以我们写了三个工具函数，并且对 48 个转动逐一做了三重自检。

```python
def inverse(token):
    """求一个转动的逆：带 - 的去掉 -，不带的加上 -"""
    return token[1:] if token.startswith('-') else '-' + token

def inverse_word(word):
    """一整条路线的逆：每个转动取逆，并且顺序反过来"""
    return tuple(inverse(t) for t in reversed(word))

def read_state(text):
    """把一行逗号分隔的颜色串变成数组，并做三重合法性检查"""
    out = np.array([int(x) for x in text.split(',')], dtype=np.intp)
    if out.shape != (384,):                                   # 必须正好 384 个面片
        raise ValueError('Expected 384 facelets')
    if np.any(out < 0) or np.any(out > 5):                    # 颜色只能是 0..5
        raise ValueError('Expected colors in 0..5')
    # 每种颜色必须正好 64 个，否则这个状态物理上不可能存在
    if not np.array_equal(np.bincount(out, minlength=6), np.repeat(64, 6)):
        raise ValueError('Expected sixty-four of each color')
    return out

def replay(state, path, generators=G):
    """按路线转动一遍：每一步都是 新状态 = 旧状态[该转动的置换数组]"""
    out = state.copy()
    for token in path:
        out = out[generators[token]]
    return out
```

接着我们对 48 个转动数组做自检。每条断言都对应一个数学事实，只要有一条不成立，就说明数组有问题：

```python
identity = np.arange(384)
for token, perm in G.items():
    # 每个转动本质是一个置换数组，排序后应该等于 0..383，说明没有漏掉或重复位置
    assert np.array_equal(np.sort(perm), identity)
    # 转一下再转它的逆，必须回到原样
    assert np.array_equal(perm[G[inverse(token)]], identity)
    # 90 度转动连做四次必须回到原样（也就是"阶为 4"）
    assert np.array_equal(perm[perm[perm[perm]]], identity)
```

这一步看起来啰嗦，但它体现了一个很重要的工程习惯：**在写任何聪明的算法之前，先把最基础的数据契约钉死**。后面所有结论都建立在这 48 个数组正确的前提上，如果它们错了，再好的搜索也只是在错误的世界里跑得很快。

### 2.3 模型训练与超参数调优

如前所述，这个比赛不训练模型，我们的「模型」是一个搜索器。在造搜索器之前，先要把题目是怎么生成的搞清楚，因为**题目本身就是答案的逆**。

搬运规则是这样的：对每个轴 *a*（取 f、r、d），3×3 的第 0、1、2 层分别映射到 8×8 的 **0、3、7** 层，并且**符号取反**——源里的 `a0` 变成目标里的 `-a0`，`a1` 变成 `-a3`，`a2` 变成 `-a7`。逆转动（带 `-` 的）映射后符号再翻回来。

```python
def lift_token(token):
    """把一个 3x3 的转动记号搬到 8x8 上：轴不变，层号换成 0/3/7，符号取反"""
    axis, layer = token.lstrip('-')
    sign = '' if token.startswith('-') else '-'
    mapped = sign + axis + str((0, 3, 7)[int(layer)])
    assert mapped in G              # 搬过去必须仍然是合法转动
    return (mapped,)
```

一个源 token 对应一个目标 token，长度完全不变。所以 3×3 的参考解总长 21,372 搬过来以后还是 21,372，最长的一条 24 步。我们用下面这段代码把这条链路完整验了一遍：先在 3×3 上确认源解法真的能把源状态复原，再把解法搬运过去，确认它和官方 `sample_submission.csv` 逐 token 一模一样，最后确认**题目的构造方式**——从复原态出发，倒着走一遍搬运后的解法，就能得到题目给的那个打乱状态：

```python
sources = dict(zip(source_states.initial_state_id, source_states.initial_state))
source_paths = dict(zip(source_routes.initial_state_id, source_routes.path))
refs = dict(zip(reference.initial_state_id, reference.path))

source_total = 0
for row in states.itertuples(index=False):
    original = np.array([int(x) for x in sources[row.initial_state_id].split(',')])
    source_path = word(source_paths[row.initial_state_id], G3)      # 3x3 的官方解法
    assert np.array_equal(replay(original, source_path, G3), np.arange(54))
    # 逐 token 搬运：源解法 -> 目标解法
    lifted = tuple(t for move in source_path for t in lift_token(move))
    assert lifted == word(refs[row.initial_state_id])               # 与官方参考解一致
    # 题目构造：复原态倒走一遍目标解法，应当得到输入的初始状态
    assert np.array_equal(read_state(row.initial_state),
                          replay(GOAL, inverse_word(lifted)))
    source_total += len(source_path)

assert source_total == 21372
```

这段验证通过以后，我们就有了一个可靠的「兜底答案」：官方参考解。但比赛明确要求**先测量搜索、再使用参考**，所以兜底不能替我们打分。

真正属于我们自己的搜索器是一个**双向广度优先搜索**。思路很朴素：从复原态出发，把所有「走两步以内」能到达的状态和对应路线存成一张表；对每个题目，从它的初始状态出发也走两步以内，一旦某个状态在表里出现过，就说明两头接上了，路线就是「前两步 + 后半段的逆」，总长不超过 4 步。因为两边都只展开两层，所以这个搜索**对「长度 ≤ 4 的解」是完备的**——只要存在这样的解，它一定能找到最短的那条。

```python
TOKENS = tuple(sorted(G))
START_DEPTH = GOAL_DEPTH = 2

def goal_table(depth):
    """从复原态做广度优先，把所有 depth 步以内可达的状态存成 {状态字节: 路线}"""
    table = {GOAL.tobytes(): ()}
    queue = deque([(GOAL.copy(), ())])
    while queue:
        current, route = queue.popleft()
        if len(route) == depth:
            continue
        for token in TOKENS:
            # 刚转完又立刻转回去是浪费，剪掉
            if route and token == inverse(route[-1]):
                continue
            nxt = current[G[token]]
            key = nxt.tobytes()          # 用字节串当字典键，比数组快得多
            if key not in table:
                path = route + (token,)
                table[key] = path
                queue.append((nxt, path))
    return table

BACKWARD = goal_table(GOAL_DEPTH)
```

有了后面的表，前半段就可以边展开边查表：

```python
def solve_bounded(initial):
    """从初始状态正向展开两层，每到达一个状态就查后表；接上了就拼成一条完整路线"""
    queue = deque([(initial.copy(), ())])
    seen = {initial.tobytes()}
    best = None
    expanded = 0                          # 记录展开了多少状态，用来衡量搜索成本
    while queue:
        current, route = queue.popleft()
        expanded += 1
        back = BACKWARD.get(current.tobytes())
        if back is not None:
            candidate = route + inverse_word(back)   # 后半段要取逆：表是从复原态出发的
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
    # 只要找到了，就必须真的能复原，否则立刻报错而不是悄悄交上去
    if best is not None:
        assert len(best) <= START_DEPTH + GOAL_DEPTH
        assert np.array_equal(replay(initial, best), GOAL)
    return best, expanded
```

**关于「调参」。** 这里唯一的超参数就是两个深度。我们把 `START_DEPTH` 和 `GOAL_DEPTH` 都设成 2：48 个转动从复原态展开两层，去掉「刚转完又立刻转回去」的重复，一共只有 1 + 48 + 48×47 ≈ 2300 个状态，普通笔记本几秒钟就能把表建好；再往上加一层就要乘上约 47，涨到十万量级，而题目最长有 24 步，多这一层对整体覆盖率的帮助其实很有限。所以 2+2 = 4 步是一个「成本可控、又能覆盖一部分短题」的折中。所以这个搜索器的定位是**标定用的最小可用搜索**：它提供了正确的展开循环、正确的逆路线处理、明确的失败信号，但它不是实用的 8×8 求解器。材料也明确说了，换成一个更强的求解器时，输入、校验、报告这三块代码可以原样复用。

### 2.4 模型测试与指标报告

主循环对每个输入跑一次搜索，把「原始结果」和「最终提交」严格分开记录：

```python
records, answers = [], []
for row in states.itertuples(index=False):
    initial = read_state(row.initial_state)
    found, expanded = solve_bounded(initial)        # 搜索器只看输入，绝不偷看参考解
    baseline = word(refs[row.initial_state_id])     # 参考解只在这里才登场
    raw_success = found is not None
    # 策略：搜索成功且比参考解短，就用搜索的；否则用参考解兜底
    selected = found if raw_success and len(found) < len(baseline) else baseline
    assert np.array_equal(replay(initial, selected), GOAL)   # 最终答案必须真的能复原
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
                    'path': '.'.join(selected) or ' '})    # 空路线要写成空格
```

这里有两个容易踩的坑，我们都做了处理。第一，空路线必须写成**一个空格**，如果写成空字符串，读回来会被 pandas 当成缺失值。第二，「原始成功率」和「最终提交是否有效」是两件事：参考解兜底能让提交 100% 有效，但这**不代表我们的搜索解出了 100%**。材料特别强调这一点，所以报告里 `raw_solved` 和 `all_full_endpoints_valid` 是分开列的。

写文件时我们用了强制引号，写完再读回来重新校验一遍：

```python
answer.to_csv(OUTPUT / 'submission.csv', index=False,
              quoting=csv.QUOTE_ALL, lineterminator='\n')
# 重新打开真正的产物再验一次：空白字符串不能变成 NaN
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

最后我们会把三个文件一起看：`submission.csv` 是提交物，`per_input_results.csv` 是逐题明细（每题展开了多少状态、花了多少秒、比参考短还是长），`validation.json` 是总报告。材料反复强调的口径是：**最终的排行榜分数包含了公开参考解的兜底，所以它衡量的是「提交质量」，不是「搜索能力」**；要看搜索能力，得去看 `raw_solved` 和 `raw_failures`。

---

## 三、总结

本文中我们完成了 CubeLift 8x8 的完整流程。我们首先用 EDA 摸清了题目：384 个面片、6 种颜色各 64 个、48 个合法的单层带符号 90 度转动。其次我们做了「验证式清洗」——写函数把状态合法性、转动数组的置换性质、逆转动关系和「转四次回原样」逐条断言，先把数据契约钉死。接着我们搞清了题目构造：3×3 的第 0、1、2 层映射到 8×8 的 0、3、7 层且符号取反，一个源 token 换一个目标 token，所以参考解总长仍是 21,372、最长 24 步；我们逐题验证了整条搬运链路和「倒走解法生成题目」的构造方式。然后我们写了双向广度优先搜索，两边各展开两层，保证对长度 ≤ 4 的解完备，并把搜索成本和搜索成败都记录下来。最后我们把搜索成功的短路线和参考解兜底合并成提交，重新读回文件做二次校验，并输出一份把「搜索表现」和「提交表现」分开的报告。

这个比赛给我们的最大启发是：**在一个「答案已知」的标定任务里，最重要的不是把分数刷高，而是把测量做诚实**。参考解兜底可以让提交永远有效，但如果报告里不把原始成功率和兜底率分开，人就会误以为自己解出了所有题。把「我做到了什么」和「我借用了什么」分清楚，比多省几步更有价值。
