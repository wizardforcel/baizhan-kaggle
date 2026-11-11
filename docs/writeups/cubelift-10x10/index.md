# CubeLift 10x10 魔方求解（CubeLift 10x10）

## 一、比赛简介

### 1.1 比赛背景

魔方是一种「置换谜题」：一堆小方块被固定在一个结构上，你每转动一层，方块的位置就被重新排列一次。目标是把打乱的状态转回整齐的状态。

CubeLift 10x10 是 CayleyPy 系列「Lift」谜题的放大版。它不是一个真的 10×10×10 的实体魔方，而是一个数学上定义好的抽象谜题：

- 一个「状态」由 **600 个色块**（facelet）组成；
- 一共 **6 种颜色**，每种颜色恰好出现 **100 次**；
- 一共有 **60 种合法动作**（move），分别是 f、r、d 三个轴、每个轴 10 层（0～9）的正转，以及它们的逆时针版本（名字前面加负号，比如 `-f3`）。

所谓「正转」，就是把这个轴上的某一层转动 90 度。转四次回到原样，所以每个动作的逆动作就是把同一层反着转一次。

这个比赛和常见的机器学习比赛很不一样：**它没有「训练数据」可学**。每条测试样本给一个打乱的状态，你要输出一串动作，让这个状态经过这串动作后回到整齐状态。比赛比的不是预测准不准，而是**你的解法有多短**。

### 1.2 比赛目标

数据文件夹里有这几样东西：

- `puzzle_info.json`：谜题定义，包含目标状态 `central_state`（600 个颜色值）和 60 个动作对应的置换数组 `generators`；
- `test.csv`：每行一个待解状态，列为 `initial_state_id` 和 `initial_state`（600 个用逗号分隔的颜色编号）；
- `sample_submission.csv`：官方给的参考解，列为 `initial_state_id` 和 `path`；
- `source333_test.csv`、`source333_submission.csv`、`source333_puzzle_info.json`：一个 3×3 的「源谜题」及其官方解。

输出格式是：每个 `initial_state_id` 对应一个 `path`，动作之间用英文句点 `.` 连接；如果某个状态本来就已还原，path 写一个空格表示「零步」。

这个比赛最有意思的地方在于 `sample_submission.csv` 不是空的，它是一份**真实可用的解**，总共 21372 步，单个输入最多 24 步。也就是说，你至少要做得跟它一样好，才有意义。

### 1.3 评价标准

分数是**所有输入路径长度之和**，**越低越好**。每一步动作算一个 token，半圈（转 180 度）要写两次，所以要算两个 token。同一个状态用两步解开，就比用十步解开好五倍。

但「短」有一个前提：**路径必须真的能把状态解开**。评分时会从初始状态开始逐个执行动作，检查 600 个色块是否全部回到目标颜色。路径再短，解不开就是零分。所以这个比赛的本质是：**在保证正确的前提下，把总步数压到最低**。

---

## 二、解题过程

材料里的这份 notebook 走的是「能验证的基线」路线：不训练任何模型，而是写一个**深度有限的精确搜索器**，能解开的就用自己搜出来的更短路径，解不开的就退回官方参考解，并且对每一条输出都做完整校验。

### 2.1 数据探索（EDA）

我们首先读入谜题定义，确认三个最基本的事实：状态有多长、动作有多少个、目标状态长什么样。

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("/kaggle/input/cubelift-10x10")

info  = json.loads((DATA / "puzzle_info.json").read_text())
GOAL  = np.array(info["central_state"], dtype=np.int8)          # 目标状态，600 个颜色
G     = {token: np.array(p, dtype=np.intp)                     # 每个动作是一个置换数组
         for token, p in info["generators"].items()}

states    = pd.read_csv(DATA / "test.csv", dtype=str, keep_default_na=False)
reference = pd.read_csv(DATA / "sample_submission.csv", dtype=str, keep_default_na=False)

print({"inputs": len(states), "facelets": len(GOAL), "signed_moves": len(G)})
# {'inputs': ..., 'facelets': 600, 'signed_moves': 60}
```

这里的 `generators` 是「置换」的数组表示。可以这样理解：如果 `p = G["f3"]`，那么执行完 `f3` 之后，新状态的第 `i` 个位置上的颜色，等于旧状态的第 `p[i]` 个位置上的颜色。用一个数组就能完整描述「哪块挪到哪」，不需要真的去模拟三维几何。

接下来我们验证目标状态和动作的合法性。这一步看着枯燥，但它决定了后面所有代码能不能信：

```python
def inverse(token):
    # 逆动作：有负号就去掉，没负号就加上。f3 的逆是 -f3，-f3 的逆是 f3
    return token[1:] if token.startswith("-") else "-" + token

# 目标状态应该是 6 种颜色各 100 个
counts = np.bincount(GOAL, minlength=6)
assert list(counts) == [100] * 6, counts

identity = np.arange(600)

for token, perm in G.items():
    # 1) 置换必须是 0~599 的一个排列：排序后刚好是 0,1,...,599
    assert np.array_equal(np.sort(perm), identity)
    # 2) 逆动作必须真的能抵消它：先做 perm 再做逆，等于没动
    assert np.array_equal(perm[G[inverse(token)]], identity)
    # 3) 做四次必须回到原点：转 90 度 × 4 = 360 度
    assert np.array_equal(perm[perm[perm[perm]]], identity)
```

第 3 条断言特别能说明问题：它验证了「每个动作确实是 90 度转动」。如果某个动作被错误地定义成 180 度，做两次就回原点了，这条检查会立刻失败。

我们接着看官方参考解的长度分布。总步数是 21372，单个输入最多 24 步，这个数字很重要——它告诉我们**官方基线大概是什么水平**，我们的搜索器至少要能在这个水平上不倒退。

```python
# 每条参考路径用 '.' 分隔，切出来的段数就是步数
ref_lengths = reference["path"].apply(lambda s: 0 if s.strip() == "" else len(s.split(".")))
print(ref_lengths.describe())
print("总步数:", ref_lengths.sum())      # 21372
print("单条最长:", ref_lengths.max())    # 24
```

这一步还顺带确认了「空路径」的写法：已还原的状态在文件里是一个空格，不是空字符串也不是 NaN。读文件时如果开了默认的缺失值处理，空格会被读成 NaN，后面全部出错——所以读 CSV 时我们统一加 `keep_default_na=False`。

### 2.2 数据清洗

对普通比赛来说，「清洗」是补缺失、去异常；对这个比赛来说，清洗就是**把所有输入和输出都严格校验一遍**，让任何不合法的数据在第一时间暴露，而不是等到评分时才发现。

我们首先定义「动作单词」的解析和回放：

```python
def inverse_word(word):
    # 一整串动作的逆：先倒序，再逐个取逆（inverse 前面已经定义）
    # 例："f0.r1" 的逆是 "-r1.-f0"
    return tuple(inverse(t) for t in reversed(word))

def word(text):
    # 把 "f0.r1.-d9" 这样的字符串解析成元组；空格代表零步
    if not isinstance(text, str):
        raise ValueError("路径必须是字符串")
    result = tuple(text.strip().split(".")) if text.strip() else ()
    if any(t not in G for t in result):
        raise ValueError("出现非法动作")
    return result

def replay(state, path):
    # 从初始状态出发，逐个动作套用置换，得到最终状态
    out = state.copy()
    for token in path:
        out = out[G[token]]
    return out
```

有了这三个函数，我们就能写状态读取的校验。注意这里检查得非常细：长度对不对、颜色值在不在 0～5、每种颜色是不是刚好 100 个。任何一条不满足，说明这一行数据本身就有问题，与其硬着头皮算，不如当场报错。

```python
def read_state(text):
    out = np.array([int(x) for x in text.split(",")], dtype=np.int16)
    if out.shape != (600,):
        raise ValueError("状态长度必须是 600")
    if np.any(out < 0) or np.any(out > 5):
        raise ValueError("颜色值必须在 0~5")
    if not np.array_equal(np.bincount(out, minlength=6), np.repeat(100, 6)):
        raise ValueError("每种颜色必须恰好 100 个")
    return out.astype(np.int8)
```

然后是整份提交文件的校验。这个函数是后面反复调用的「总闸门」：它不仅检查格式，还会**真的把每条路径走一遍**，确认终点等于目标状态：

```python
def validate(answer):
    # 列名必须严格是这两列
    if set(answer.columns) != {"initial_state_id", "path"}:
        raise ValueError("列名不对")
    ids = answer["initial_state_id"].tolist()
    # 每个 ID 必须出现且只出现一次，不能多也不能少
    if len(ids) != len(set(ids)) or set(ids) != set(states["initial_state_id"]):
        raise ValueError("ID 集合不匹配")

    routes = dict(zip(ids, answer["path"]))
    total = 0
    for row in states.itertuples(index=False):
        path = word(routes[row["initial_state_id"]])
        if not np.array_equal(replay(read_state(row["initial_state"]), path), GOAL):
            raise ValueError("这条路径解不开: " + row["initial_state_id"])
        total += len(path)      # 分数 = 所有路径长度之和

    return total
```

`validate` 返回的就是比赛的分数。我们把它设计成「只要有一条不对就抛异常」，这样在本地跑通，就等价于提交能过校验，不用拿提交次数去试错。

### 2.3 模型训练与超参数调优

这个比赛没有模型可训练。我们的「模型」是一个**双向广度优先搜索**（bidirectional BFS）。为了讲清楚它，先解释 BFS：

BFS（广度优先搜索）就像往水里扔石子看波纹扩散：从起点出发，先看「走 1 步能到的所有状态」，再看「走 2 步能到的」，一层一层往外扩。因为是一层一层来的，所以第一次碰到目标状态时，走的步数一定是最少的。

双向 BFS 是从两头同时扩：一边从目标状态往外扩两步，存成一张表；另一边从每个输入往外扩两步。如果两边在某个状态「碰头」，把两段路径拼起来，就是一条完整解。两边各扩 2 步，拼起来最多 4 步，所以这个搜索器**能保证找到所有 4 步以内的最优解**。

```python
from collections import deque

TOKENS = tuple(sorted(G))
START_DEPTH = GOAL_DEPTH = 2       # 两边各扩 2 层，拼起来最多 4 步

def goal_table(depth):
    # 从目标状态出发，把所有 depth 步以内能到达的状态存进字典
    # 键是状态字节串（方便哈希），值是到达它的动作序列
    table = {GOAL.tobytes(): ()}
    queue = deque([(GOAL.copy(), ())])

    while queue:
        current, route = queue.popleft()
        if len(route) == depth:
            continue                     # 到达深度上限就不再往下扩
        for token in TOKENS:
            # 剪枝：刚做完 f0 又做 -f0，等于白走一步
            if route and token == inverse(route[-1]):
                continue
            nxt = current[G[token]]
            key = nxt.tobytes()
            if key not in table:         # 第一次到达才是最短，后来的不用存
                path = route + (token,)
                table[key] = path
                queue.append((nxt, path))
    return table

BACKWARD = goal_table(GOAL_DEPTH)
print("目标表状态数:", len(BACKWARD))
```

这里有两个关键设计。第一，**用字节串当字典键**。600 个 int8 组成的 numpy 数组不能直接哈希，转成 `bytes` 就可以，而且比较起来非常快。第二，**第一次到达才记录**。BFS 是按层扩散的，第一次遇到某个状态时走的步数一定最少，后面再遇到就不用覆盖了——这保证了表里存的都是最短路。

从输入这一侧，我们做同样的事，但每扩展到一个状态，就查一下它有没有出现在目标表里：

```python
def solve_bounded(initial):
    queue = deque([(initial.copy(), ())])
    seen = {initial.tobytes()}
    best = None
    expanded = 0

    while queue:
        current, route = queue.popleft()
        expanded += 1

        back = BACKWARD.get(current.tobytes())
        if back is not None:
            # 碰头了：正向路径 + 反向路径的逆，拼成完整解
            candidate = route + inverse_word(back)
            if best is None or len(candidate) < len(best):
                best = candidate

        if len(route) == START_DEPTH:
            continue                     # 深度到顶，不再往下扩
        for token in TOKENS:
            if route and token == inverse(route[-1]):
                continue
            nxt = current[G[token]]
            key = nxt.tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((nxt, route + (token,)))
    return best, expanded                # 找不到时 best 是 None
```

为什么反向路径要取逆、还要倒序？因为目标表里存的是「从目标走到中间状态」的动作，而我们需要的是「从中间状态走回目标」，方向正好相反，所以整串动作要反过来做。

最后我们对表里的短路径做一次自检：目标表里距离为 0 或 1 的状态，`solve_bounded` 必须能搜出同样长度的解。这个测试能抓出「拼路径时方向搞反」这类最常见的 bug：

```python
for key, route in BACKWARD.items():
    if len(route) <= 1:
        solution, _ = solve_bounded(np.frombuffer(key, dtype=np.int8))
        assert len(solution) == len(route)
```

关于「超参数」，这里其实只有两个：两边的搜索深度。我们把两边都设为 2，理由是**成本可控**：60 个动作、深度 2，正向最多扩 60×59 ≈ 3500 个状态，目标表也是同样量级，内存和时间都很轻。如果把深度加到 3，状态数会涨到几十万甚至上百万，普通 notebook 就跑不动了。所以这个搜索器是「在能跑完的前提下，尽可能多解几个」的折中，notebook 里也明确说了它**不是**一个通用的 10×10 求解器。

### 2.4 模型测试与指标报告

推理阶段，我们对每一条输入都跑一次 `solve_bounded`，并且记录三件事：有没有搜到、扩了多少个状态、花了多少秒。这三项是判断搜索器值不值得继续投入的依据。

```python
records, answers = [], []
raw_successes = 0

for row in states.itertuples(index=False):
    initial = read_state(row["initial_state"])
    found, expanded = solve_bounded(initial)

    baseline = word(reference.loc[reference["initial_state_id"] == row["initial_state_id"],
                                  "path"].iloc[0])

    raw_success = found is not None
    # 搜到了、而且比参考解短，才用自己的；否则一律用参考解
    selected = found if raw_success and len(found) < len(baseline) else baseline

    # 选出来的路径必须真的能解开，否则宁可报错也不要提交
    assert np.array_equal(replay(initial, selected), GOAL)

    records.append({
        "initial_state_id": row["initial_state_id"],
        "raw_success": raw_success,
        "raw_moves": len(found) if raw_success else None,
        "expanded_states": expanded,
        "reference_moves": len(baseline),
        "submitted_moves": len(selected),
        "length_minus_reference": len(selected) - len(baseline),
        "used_reference_fallback": not raw_success,
    })
    answers.append({"initial_state_id": row["initial_state_id"],
                    "path": ".".join(selected) or " "})   # 零步写空格
```

这段代码体现了这场比赛最重要的一个原则：**先测量，再用兜底**。搜索器解不开的输入，我们不硬凑，直接退回官方参考解——这样提交一定是合法的、完整的，同时我们又能诚实地说「有多少条是我自己解出来的」。

导出时有两个细节容易踩坑。第一，路径里可能出现以负号开头的动作（比如 `-f3`），如果导出时不加引号，某些解析器会把它当成公式；所以要用 `QUOTE_ALL` 把所有字段都加上引号。第二，重新读回来时不能把空格当成缺失值：

```python
import csv

answer = pd.DataFrame(answers, columns=["initial_state_id", "path"])
score = validate(answer)

answer.to_csv(OUTPUT / "submission.csv", index=False,
              quoting=csv.QUOTE_ALL, lineterminator="\n")

# 把刚写出去的文件重新读一遍再校验：写盘环节也可能出问题
exported = pd.read_csv(OUTPUT / "submission.csv", dtype=str, keep_default_na=False)
assert validate(exported) == score
print("总分:", score, " 参考解总分: 21372")
```

最后我们写一份 `validation.json`，把关键指标都落盘：总步数、参考解步数、省下的步数、搜索原始成功率、总扩展状态数、耗时、文件哈希。这份报告的价值在于**可复现**：别人拿到同一个文件，可以逐项对照，而不是只看到一个「我分数很高」的说法。

指标报告我们分两层看：

- **原始成功率**：搜索器独立解开了多少条输入。这是「方法本身有多强」的度量，不掺参考解的水分。
- **提交总步数**：最终提交文件的总长度，和参考解的 21372 对比。因为兜底用的是参考解，这个数字只可能持平或更小，不会更差。

材料里的 notebook 还特别强调了一句：对第 i 条输入，`L_i - b_i`（提交长度减参考长度）是「相对这条已知可行路径的差距」，**不是**「距离真正最优解有多远」——因为谁也不知道真正的最短解是多少步。这个区分很重要，它决定了我们不会把「比参考解短」吹成「接近最优」。

---

## 三、总结

本文中我们完成了 CubeLift 10x10 的完整求解流程。我们首先解析 `puzzle_info.json`，确认状态有 600 个色块、6 种颜色各 100 个、共 60 个带符号的合法动作，并逐条验证每个动作都是合法的置换、做四次能回到原点；其次我们把「清洗」做成了严格校验：解析动作单词、回放路径、检查状态长度和颜色计数、检查提交的 ID 集合与列名，并写了一个「走完全部 600 个位置」的 `validate` 函数当总闸门；接着我们用双向广度优先搜索作为求解器，从目标状态扩 2 层建表、从每个输入扩 2 层查表，保证能找到所有 4 步以内的最短解，同时明确承认它在更深的状态上会失败；最后我们对每条输入先测量搜索覆盖率和耗时，搜到更短的解就用，搜不到就退回官方参考解，导出后重新读回再校验一遍，并输出包含总分、覆盖率、扩展状态数的验证报告。

这个比赛让我们看到，**「没有训练数据」的比赛同样可以做得很工程化**。真正决定成败的不是模型有多花哨，而是三件事：状态和动作的表示是否严谨、验证是否覆盖到每一个色块、以及有没有诚实地区分「我自己解出来的」和「我抄参考解的」。参考解总步数是 21372、单条最长 24 步，这个基线摆在那里，任何声称「大幅超越」的说法，都应该先问一句：你的路径，真的每一步都验证过吗？
