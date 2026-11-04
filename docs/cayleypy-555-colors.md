# CayleyPy 555 色彩魔方（555 Colors）

## 一、比赛简介

### 1.1 比赛背景

CayleyPy 是一个专门研究「魔方类谜题」怎么用计算机求解的开源项目。名字里的 Cayley 指的是「凯莱图」（Cayley graph）：把魔方的每一种状态看成一个点，把每个动作看成连接两个点的一条边，整个状态空间就变成了一张巨大无比的网。求解魔方，就是在这张网里从起点走到终点，而且希望走得越短越好。

这个比赛用的是 5×5×5 的六色魔方，也就是常说的五阶魔方。一个五阶魔方每个面有 5×5=25 个小格，六个面加起来是 150 个格子，每个格子叫一个「贴纸」（facelet）。所以一个魔方状态可以写成长度 150 的数组，数组里每个数字是 0～5 的颜色编号，而且每种颜色正好出现 25 次。

动作一共有 30 个，叫做「带符号的单层四分之一转」：f、r、d 三个方向各 5 层，每层再分正转和反转，3×5×2=30 个。这里要特别注意，中间的层也能转，所以不能只拧最外面那一层；另外，反向转不是「免费」的，它同样算一步。

这个比赛有三个难点。第一，状态空间大到根本不可能穷举，五阶魔方的合法状态数量是一个天文数字。第二，每条路径必须严格合法，动作要按题目给的置换数组作用在状态上，写错一个动作整条路径就作废。第三，分数是路径长度之和，越短越好，所以不是「能还原就行」，而是要尽量短。

### 1.2 比赛目标

输入是 `test.csv` 里的 `initial_state` 列：150 个用逗号分隔的颜色值（0～5）。输出是 `path` 列：用英文句点 `.` 分隔的动作名，比如 `f0.-r3` 表示先做 `f0`，再做 `-r3`（r3 的反向）。每个 `initial_state_id` 对应一行。

比赛附带四个文件：

- `test.csv`：`initial_state_id` 和 `initial_state`，一共 1035 个待求解状态；
- `sample_submission.csv`：官方提供的一条 baseline 路径，可以直接提交；
- `puzzle_info.json`：`generators`（30 个长度为 150 的置换数组）和 `central_state`（目标状态）；
- `metric.py`：官方评分器。

这里要把「置换」讲清楚。一个动作不是一个「转动公式」，而是一个长度为 150 的数组 g。执行动作就是把状态重新排列成 `s[g]`。数组里的数字是「位置下标」，不是颜色，这一点非常容易搞混——如果把它当成颜色去做 `argsort`，结果一定是错的。

### 1.3 评价标准

评分函数把所有 1035 条路径的长度加起来。一个四分之一转算 1 步，半转算 2 步，整体旋转魔方也不免费。分数越低越好。官方 baseline 的总分是 **503,776 步**。

因为分数是「总和」，所以某一条路径特别长会拖累整体；也因为它是总和，我们只要保证每一条都不比 baseline 长，整体就不可能变差。

---

## 二、解题过程

材料里其实有两条路线：一条是训练价值网络、用束搜索加终局表来求解；另一条是纯 CPU 的「精确化简」。我们选第二条作为主线，因为它不需要 GPU、每一步都能验证、结果也更好讲清楚：它把官方 baseline 的 503,776 步压到了 **456,796 步**。神经网络那条路线我们在总结里一句话带过。

### 2.1 数据探索（EDA）

我们首先把数据和官方评分器加载进来，顺手做一次「对账」：确认测试集正好 1035 行、baseline 文件的 SHA-256 和材料里记录的一致、官方评分器给 baseline 打出的分数正好是 503,776。这样做是为了让后面所有比较都有一个可信的参照系。

```python
from pathlib import Path
import hashlib, importlib.util
import pandas as pd

# 比赛把数据分散在 /kaggle/input 下，先自动找到同时有 test.csv 和提交样例的目录
DATA = next(p.parent for p in Path('/kaggle/input').rglob('test.csv')
            if (p.parent / 'sample_submission.csv').is_file())

states   = pd.read_csv(DATA / 'test.csv', dtype=str)
baseline = pd.read_csv(DATA / 'sample_submission.csv', dtype=str)

# 直接加载官方评分器，而不是自己重写计分逻辑，避免“自己算自己”
spec = importlib.util.spec_from_file_location('published_metric', DATA / 'metric.py')
metric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric)
score = metric.score

# 三方对账：行数、文件指纹、baseline 总分，任何一项对不上就停下
assert len(states) == 1035
assert hashlib.sha256((DATA / 'sample_submission.csv').read_bytes()).hexdigest() \
       == 'd1c856ab0ad73fc55c090be3e0641b4dde0bc258c67acb003a223422259c057a'
assert int(score(states, baseline, 'initial_state_id')) == 503776
print('baseline =', 503776)
```

这段对账让我们确认了两件事：第一，官方评分器是唯一的权威，任何「我自己数出来的长度」都不如直接调 `metric` 可靠；第二，baseline 本身是合法的，所以后面的化简只要保证「不比原来长」，最坏情况也不会比 baseline 差。

接着我们检查 `puzzle_info.json` 里的 30 个动作到底长什么样：

```python
import json
import numpy as np

puzzle = json.loads((DATA / 'puzzle_info.json').read_text())
NAMES = list(puzzle['generators'])   # 30 个动作名，例如 f0、-f0、r4 ...
# 把 30 个动作堆成一个 30×150 的数组，每一行是一个置换
GP = np.array([puzzle['generators'][n] for n in NAMES], dtype=np.intp)
TARGET = np.asarray(puzzle['central_state'], dtype=np.uint8)

assert GP.shape == (30, 150)                                   # 动作数 × 位置数
assert np.array_equal(np.bincount(TARGET, minlength=6), np.full(6, 25))  # 六色各 25 个
identity = np.arange(150)
for g in GP:
    assert np.array_equal(np.sort(g), identity)                # 每个动作是 0..149 的排列

# 每个动作都要能找到逆动作：g 与它的逆复合后必须回到恒等
inverse = np.array([next(j for j, h in enumerate(GP)
                        if np.array_equal(g[h], identity))
                    for g in GP])
print('actions:', len(NAMES), 'inverse ok:', inverse.shape)
```

这段检查做了四件事：动作数组必须是 30×150；目标状态必须六种颜色各 25 个；每个动作必须是 0～149 的一个排列（排序后正好等于 0..149）；每个动作都能找到逆动作。最后一条尤其重要，因为后面做「两动作桥」时要用到逆动作。

### 2.2 数据清洗

这份数据没有缺失值，也没有需要去重的行。所谓「清洗」，主要是把 baseline 的路径字符串变成干净的记号序列，并且拒绝任何不认识的记号。

```python
def parse_path(text):
    """把 baseline 的 path 字符串变成动作名列表。"""
    if not isinstance(text, str) or not text.strip():
        return []                    # 空路径表示“当前已经是目标状态”
    tokens = text.split('.')         # 官方用句点分隔动作
    for t in tokens:
        assert t in NAMES, t         # 遇到不认识的记号立刻报错，绝不静默丢弃
    return tokens

def apply_path(state, tokens):
    """按顺序执行动作，返回新的 150 维颜色状态。"""
    out = np.asarray(state, dtype=np.uint8).copy()
    for name in tokens:
        out = out[GP[NAMES.index(name)]]   # 动作是“按位置重排”，不是对颜色做加减
    return out

# 逐行拆开 baseline，确认拆完之后总步数和官方分数一致
paths = [parse_path(t) for t in baseline['path']]
assert sum(map(len, paths)) == 503776
print('rows:', len(paths))
```

我们做了三层检查：路径按 `.` 切开后，每个记号必须在 30 个合法动作名里；空路径合法地表示「已经在目标状态」，我们统一写成空列表或一个空格；每个记号都必须能通过 `apply_path` 作用在 150 个位置上。做完这些，`path` 就变成了一个纯字符串列表，后面的代数运算只在这个列表上做。

### 2.3 模型训练与超参数调优

这一节没有神经网络。我们把「化简算法」当成模型，把「用哪几个算子、按什么顺序、迭代几轮」当成超参数。核心是三个算子：代数合并、删环、走捷径，外加一个两动作桥。

**第一个算子：代数合并。** 如果两个层动作的置换可以交换，那么它们中间夹着的动作也能被跨过去，把同层动作合并成一次幂，再对 4 取模。反向转等价于正向做 3 次；半转要写成两个同向转，因为它算 2 步。删环则是利用「颜色状态」：如果同一个 150 维颜色状态在路径上出现了两次，中间那一段一定是绕了个圈，可以直接删掉。注意这里看的是颜色，不是贴纸编号——贴纸可以有多种排列方式，但颜色必须回到原样。

```python
BASES = [axis + str(i) for axis in "frd" for i in range(5)]   # f0..f4, r0..r4, d0..d4
BP = [puzzle['generators'][n] for n in BASES]
# 两个层动作是否可交换：把置换按不同顺序复合，结果一样就可交换
COMMUTE = np.array([[np.array_equal(a[b], b[a]) for b in BP] for a in BP])
BASEIDX = {n: i for i, n in enumerate(BASES)}

def algebraic(tokens):
    """把可交换的相邻动作合并成一次幂，再把幂化到 0~3 次。"""
    stack = []
    for token in tokens:
        negative = token.startswith('-')
        base = BASEIDX[token[1:] if negative else token]
        exponent = 3 if negative else 1        # 反向 = 正向做三次
        j = len(stack) - 1
        while j >= 0 and COMMUTE[base, stack[j][0]]:
            if stack[j][0] == base:            # 找到同层动作，指数相加
                exponent = (exponent + stack[j][1]) % 4   # 四次一循环
                stack.pop(j)
                break
            j -= 1
        if exponent:
            stack.append((base, exponent))
    out = []
    for base, exponent in stack:
        name = BASES[base]
        # 2 次要写成两个正转（半转算 2 步），3 次写成一个反转
        out.extend([name] * 2 if exponent == 2
                   else [name if exponent == 1 else '-' + name])
    return out

def erase_loops(initial, tokens):
    """同一个颜色状态第二次出现时，中间那段一定在绕圈，直接删掉。"""
    states = [np.array(initial, dtype=np.uint8)]
    seen = {states[0].tobytes(): 0}
    out = []
    for token in tokens:
        state = states[-1][GP[NAMES.index(token)]]
        key = state.tobytes()
        if key in seen:                        # 回到了以前见过的颜色状态
            j = seen[key]
            for removed in states[j + 1:]:     # 从索引表里清掉被删的状态
                del seen[removed.tobytes()]
            del states[j + 1:]                 # 状态序列回退到第一次出现的位置
            del out[j:]                        # 动作序列同样回退
        else:
            out.append(token)
            states.append(state)
            seen[key] = len(out)
    return out
```

**第二个算子：一步捷径。** 我们把「已经走过的所有状态」看成一张有向图：如果某个早一点的状态能一步走到后面某个状态，就直接连一条边。然后在这张图上做一次最短路（动态规划），就能把路径里绕远的部分换成直连。

```python
def one_step_shortcuts(initial, tokens):
    """在“走过的状态”上建图：任意早状态若能一步到后面的状态，就直接连边。"""
    states = [np.array(initial, dtype=np.uint8)]
    for token in tokens:
        states.append(states[-1][GP[NAMES.index(token)]])

    # 颜色状态 -> 它在路径里的位置，用来判断“后面是否又经过这里”
    lookup = {s.tobytes(): j for j, s in enumerate(states)}
    cost = np.arange(len(states), dtype=np.int32)      # 不走捷径时第 i 步就是 i
    prev = [(j - 1, tokens[j - 1]) if j else None for j in range(len(states))]

    for i, state in enumerate(states[:-1]):
        if cost[i] + 1 < cost[i + 1]:                  # 正常往前走一步
            cost[i + 1] = cost[i] + 1
            prev[i + 1] = (i, tokens[i])
        for g, neighbor in enumerate(state[GP]):       # 看哪个邻居在后面出现过
            j = lookup.get(neighbor.tobytes(), -1)
            if j > i + 1 and cost[i] + 1 < cost[j]:    # 只许向前跳，避免绕死循环
                cost[j] = cost[i] + 1
                prev[j] = (i, NAMES[g])

    out = []
    at = len(tokens)
    while at:                      # 从终点倒着把动作接回来
        at, token = prev[at]
        out.append(token)
    return out[::-1]
```

**第三个算子：两动作桥。** 假设有两条路线分别从 A 和 B 出发，各自一步走到了同一个中间状态 U：A 做 a 到 U，B 做 b 到 U。那么从 A 先做 a、再做 b 的逆，就能直接到达 B。我们用一个哈希表记录「每个中间状态最早被谁、用哪个动作到达过」，就能把这种两跳连接找出来。因为用的是完整的 150 维颜色状态做键，不存在「差不多就算匹配」的风险。

```python
def inverse_token(name):
    return name[1:] if name.startswith('-') else '-' + name

def two_edge(initial, tokens):
    """两动作桥：A --a--> U 且 B --b--> U，则 A 做 a 再做 b 的逆就到 B。"""
    states = [np.asarray(initial, dtype=np.uint8)]
    for name in tokens:
        states.append(states[-1][GP[NAMES.index(name)]])

    neighbor_map = {}       # 中间状态 U -> (到达 U 的最短代价, 来自第几步, 用哪个动作)
    costs, previous = [], []
    for j, state in enumerate(states):
        best = costs[-1] + 1 if j else 0
        record = (j - 1, [tokens[j - 1]]) if j else None

        direct = neighbor_map.get(state.tobytes())       # 情况一：原来一步就能到这
        if direct is not None and direct[0] + 1 < best:
            best = direct[0] + 1
            record = (direct[1], [NAMES[direct[2]]])

        neighbors = state[GP]
        keys = [s.tobytes() for s in neighbors]
        for g, key in enumerate(keys):                   # 情况二：通过共同邻居走两步
            first = neighbor_map.get(key)
            if first is not None and first[0] + 2 < best:
                best = first[0] + 2
                record = (first[1], [NAMES[first[2]], inverse_token(NAMES[g])])

        costs.append(best)
        previous.append(record)
        for g, key in enumerate(keys):                   # 把当前状态的一步邻居登记进去
            first = neighbor_map.get(key)
            if first is None or best < first[0]:
                neighbor_map[key] = (best, j, g)

    pieces = []
    at = len(tokens)
    while at:
        at, segment = previous[at]
        pieces.append(segment)
    return [name for segment in reversed(pieces) for name in segment]
```

在超参数上，我们实际调的是「算子顺序和迭代次数」：先做 5 轮代数合并，再做一次「删环 + 一步捷径」，接着 5 轮「代数 + 删环」，最后 3 轮「两动作桥 + 代数 + 删环」。每一轮都要求长度严格下降才继续，不下降就提前停止。这样每一行的计算量都是可控的，而且不会出现「越优化越长」的怪事。

### 2.4 模型测试与指标报告

最后一步是把 1035 行全部跑一遍，并且用官方评分器整体重放。任何一条路径不合法，评分器会直接报错；我们还会额外断言「评分器算出来的总分」等于「我们自己数的长度之和」，防止两边口径不一致。

```python
rows, before_lengths, after_lengths = [], [], []

for state_row, path_row in zip(states.itertuples(index=False),
                               baseline.itertuples(index=False)):
    assert state_row.initial_state_id == path_row.initial_state_id
    initial = np.fromstring(state_row.initial_state, dtype=np.uint8, sep=',')
    tokens = parse_path(path_row.path)
    before_lengths.append(len(tokens))

    # 第一轮：只做代数合并，直到长度不再下降
    for _ in range(5):
        candidate = algebraic(tokens)
        if len(candidate) == len(tokens):
            tokens = candidate
            break
        tokens = candidate

    # 第二轮：删环 + 一步捷径
    tokens = one_step_shortcuts(initial, erase_loops(initial, tokens))

    # 第三轮：代数与删环交替压缩
    for _ in range(5):
        candidate = erase_loops(initial, algebraic(tokens))
        if len(candidate) == len(tokens):
            tokens = candidate
            break
        tokens = candidate

    # 第四轮：两动作桥，再用代数和删环收尾
    for _ in range(3):
        candidate = erase_loops(initial, algebraic(two_edge(initial, tokens)))
        if len(candidate) == len(tokens):
            tokens = candidate
            break
        tokens = candidate

    assert len(tokens) <= before_lengths[-1]        # 化简只能变短，绝不能变长
    after_lengths.append(len(tokens))
    rows.append({'initial_state_id': path_row.initial_state_id,
                 'path': '.'.join(tokens) or ' '})  # 空路径写成一个空格

submission = pd.DataFrame(rows)

# 用官方评分器重放全部 1035 条，任何一条不合法都会在这里暴露
total = int(score(states, submission, 'initial_state_id'))
assert total == sum(after_lengths)                  # 评分器结果 = 我们自己数的长度
assert total == 456796
submission.to_csv('submission.csv', index=False)
print('baseline 503776 ->', total, 'saved', 503776 - total)
```

最终我们得到：

- 总步数从 **503,776** 降到 **456,796**，一共省下 **46,980** 步，约 **9.33%**；
- 每一行都不比 baseline 长，所以整体分数不可能变差；
- 全部 1035 行都通过官方评分器重放。

除了 `submission.csv`，我们还写了一份 `verification.json` 和一份 `per_input_lengths.csv`，记录每行的「化简前 / 化简后 / 省了多少」。这样如果以后想换方法，可以直接对比哪些输入还有压缩空间。

需要说明的是，这个方法只能在「当前构造出的图」里保证最优：它不证明全局最短路径。材料里也明确写了，这些路径是「这个受限方法能找到的最好结果」，而不是数学意义上的最短解。

---

## 三、总结

本文中我们完成了 CayleyPy 555 色彩魔方的求解。我们首先用 EDA 和官方评分器对账，确认了 1035 个待求解状态、30 个带符号单层转、目标状态六色各 25 个，以及 baseline 的 503,776 步。其次我们把 baseline 的路径字符串解析成干净的记号序列，做了合法性清洗。接着我们用三个精确算子——可交换动作的代数合并、颜色状态删环、一步捷径图搜索，再加上一个两动作桥——反复迭代压缩路径。最后我们用官方评分器重放全部 1035 条，把总分从 503,776 降到 456,796，省下 46,980 步，约 9.33%，并断言没有任何一行变长。

这个比赛的启发是：**优化不一定要靠更复杂的模型，先把「已有的合法解」吃干榨净，往往性价比最高**。材料里另一条路线训练了价值网络，用束搜索加终局表来求解，还加载了作者已经训练好的模型；那条路线更通用，但工程量大得多。对我们这种入门者来说，先把「怎么定义状态、怎么验证路径、怎么用官方评分器」这三件事搞明白，比追新模型更重要。
