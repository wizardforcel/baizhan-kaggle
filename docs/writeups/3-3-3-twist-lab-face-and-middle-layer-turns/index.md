# 3×3×3 魔方分层转动挑战（Layer-Turn Challenge）

## 一、比赛简介

### 1.1 比赛背景

魔方大家都见过：一个 3×3×3 的立方体，六个面各有一种颜色，拧几下就打乱了，再拧回来就算复原。计算机科学家很喜欢拿它当测试题，因为「怎么用最少的步数拧回去」是一个典型的**搜索问题**——每一步都有若干种选择，选择一多，可能性就像滚雪球一样膨胀。

这个比赛来自 CayleyPy 这一系列「魔方求解」比赛。它和普通魔方题最大的不同在于**转动规则**：

- 普通魔方只有六个外表面可以转（这叫 HTM，Half Turn Metric 的常见设定）；
- 这个比赛允许**外层和中层都转**，一共 18 种「带方向的四分之一圈」；
- 每个小块不是用「颜色」标记，而是用 **54 个互不相同的编号**标记。也就是说，光把六个面的颜色拼对不算完，每一张贴纸都必须回到它自己那个唯一的位置。

第三条特别关键。我们平时复原魔方，只关心「这一面是不是全白」，而这里关心的是「编号 17 的贴纸是不是正好在 17 号位」。中心块也有指定的位置，所以整块魔方不能随便翻转——把它整体转个方向也是有代价的。

### 1.2 比赛目标

比赛给的数据很少，就三个文件：

- `puzzle_info.json`：里面写着目标状态 `central_state`（就是 0 到 53 的顺序排列）和 18 个 `generators`（每个转动对应的「位置置换表」）；
- `test.csv`：每行是一个被打乱的状态，列是 `initial_state_id` 和 `initial_state`（54 个编号，用逗号隔开）；
- `sample_submission.csv`：官方给的参考答案，列是 `initial_state_id` 和 `path`。

我们要输出的是 `path` 这一列：一串用点号隔开的转动名字，比如 `d2.r0.-f0`，表示依次做这三个转动，做完之后这个状态刚好回到目标状态。

一共有 **1003** 个待解状态。其中 0、1、2 号本身就已经是复原状态了，不需要动。

### 1.3 评价标准

评价指标非常直白：**所有状态用掉的步数总和，越低越好**。

有三条计分细则必须记住：

1. 一个「带方向的四分之一圈」算 **1 步**；
2. 一个「半圈」算 **2 步**——所以能用一次半圈表达的地方，写成一个转动两次，分数一样，不会更便宜；
3. 把整个魔方换个朝向也**不是免费的**，同样要算步数。

官方参考文件 `sample_submission.csv` 一共用了 **500,620 步**，这就是那条「及格线」式的基准。而另一个公开文件 `source333_submission.csv`（来自同系列 CubeLift 5x5 比赛的公开发布）只用了 **21,372 步**，等于把总步数压到了官方参考的百分之四左右。我们的目标就是尽量往 21,372 这个量级靠。

> 注意：步数少不代表「最优」。只有穷举证明过的那部分才能说「这是最短解」，其余的只能说「目前找到的比较短」。

---

## 二、解题过程

整场比赛的核心不是训练模型，而是**搜索 + 验证**。我们首先把魔方的数据结构搭好，其次写一个能检查任何答案的验证器，接着做几路互相独立的搜索去把路线改短，然后搭一座通往经典两阶段求解器的「翻译桥」，最后把结果导出并逐条复验。

### 2.1 数据探索（EDA）

我们第一步是把 `puzzle_info.json` 读出来，确认这份数据到底是什么形状：

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

# 读取谜题定义：目标状态 + 18 个转动生成元
info = json.loads((SOURCE / 'puzzle_info.json').read_text())
TARGET = np.array(info['central_state'], dtype=np.int16)

# 每个 generator 是一个长度 54 的数组，意义是「拧一下之后，第 i 号位置的东西从哪来」
G = {name: np.array(p, dtype=np.intp) for name, p in info['generators'].items()}

# 自检：目标状态必须是 0..53 的顺序，且转动恰好 18 种
assert np.array_equal(TARGET, np.arange(54)) and len(G) == 18

# 读入待解状态；keep_default_na=False 防止空字符串被 pandas 变成 NaN
states = pd.read_csv(SOURCE / 'test.csv', dtype=str, keep_default_na=False)
print({'inputs': len(states), 'positions': len(TARGET), 'signed_moves': len(G)})
```

打印出来的结果是 `inputs=1003`、`positions=54`、`signed_moves=18`。这三个数字决定了后面所有设计：状态用长度 54 的数组表示，每一步从 18 个里选一个。

接着我们看生成元长什么样，顺便验证两条最基本的性质——**每个转动都有逆，且转四次回到原样**：

```python
identity = np.arange(54)

for token, p in G.items():
    # 逆转动：名字前面加/去一个减号，就是反方向拧
    inverse = token[1:] if token.startswith('-') else '-' + token
    # 性质一：一个转动接着它的逆转动，什么也没发生
    assert np.array_equal(p[G[inverse]], identity)
    # 性质二：同一个方向拧四次，回到原样
    assert np.array_equal(p[p[p[p]]], identity)
```

这两条看着像废话，实际非常重要。第一，它告诉我们**答案可以反向拼**：如果我们知道从目标出发怎么走到某个状态，那么反过来把这条路倒着念、每个动作取逆，就是这个状态的解。第二，它让我们可以放心地做「合并相邻同层转动」这种化简，因为四次一圈是恒等变换。

我们还统计了官方参考答案的长度分布：

```python
original = pd.read_csv(SOURCE / 'sample_submission.csv', dtype=str, keep_default_na=False)
# 每行 path 里点号隔开的 token 数，就是这个状态的步数
original['moves'] = original['path'].apply(lambda s: len(s.strip().split('.')) if s.strip() else 0)
print(original['moves'].describe())
print('total =', original['moves'].sum())   # 500620
```

官方答案总共 500,620 步，平均每个状态几百步，明显是「能用就行」的粗糙解。这给了我们很大的改进空间：只要有一路可靠的搜索，把每个状态压到几十步，总分就会成倍下降。

### 2.2 数据清洗

这里的「清洗」不是处理缺失值，而是**把输入变成可以逐条验算的形式**。魔方题最怕的一件事是：算出来的路线看起来对，实际上有一步悄悄错了，而错的状态和正确的状态只差一两个贴纸，肉眼看颜色根本发现不了。

所以我们先定死三条规矩：

1. 每个输入必须正好包含 54 个互不相同的编号，缺一个、重一个都算非法；
2. 路线里的每个 token 必须是 18 个生成元之一，不能有拼写错误；
3. 空路线（已经复原的状态）在 CSV 里写一个**带引号的空格**，因为 Kaggle 的预检查会把空的 CSV 单元格当成缺值直接拒绝。

```python
def parse_word(value):
    """把 'd2.r0.-f0' 这样的字符串切成转动列表；空字符串代表零步。"""
    if not isinstance(value, str):
        raise ValueError('路径必须是文本；零步请用一个空格表示')
    word = value.strip().split('.') if value.strip() else []
    if any(t not in G for t in word):
        raise ValueError('出现了未知的转动名')
    return tuple(word)

def read_state(text):
    """把 '12,3,40,...' 这样的一行数字变成数组，并检查是不是 54 个不重复的编号。"""
    s = np.array([int(x) for x in text.split(',')], dtype=np.int16)
    if not np.array_equal(np.sort(s), TARGET):
        raise ValueError('每个输入必须包含全部 54 个不同编号')
    return s

def replay(state, word):
    """按顺序执行一串转动。关键写法：result = result[g]，用索引完成置换。"""
    result = state.copy()          # 复制一份，绝不在原数组上改
    for token in word:
        result = result[G[token]]  # 第 i 位的新内容 = 旧数组第 g[i] 位的内容
    return result
```

`replay` 是整个方案的地基。它的写法只有一行，但含义要想清楚：`g` 是一张「取数表」，拧一下之后，位置 `i` 上的贴纸来自原来的位置 `g[i]`。所以用 `state[g]` 就完成了置换，又快又不会写错。

有了 `replay`，验证器就是把所有行跑一遍：

```python
def validate(answer):
    """检查一份完整提交：ID 不重不漏，且每一条路线都要真的走回目标状态。"""
    if set(answer.columns) != {'initial_state_id', 'path'}:
        raise ValueError('列必须正好是 initial_state_id 和 path')
    ids = answer.initial_state_id.tolist()
    expected = states.initial_state_id.tolist()
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('每个期望的 ID 必须出现且只出现一次')

    routes = dict(zip(ids, answer.path))
    total = 0
    for row in states.itertuples(index=False):
        word = parse_word(routes[row.initial_state_id])
        # 唯一的标准：完整重放后是否等于 0..53
        if not np.array_equal(replay(read_state(row.initial_state), word), TARGET):
            raise ValueError(f'没有复原：{row.initial_state_id}')
        total += len(word)
    return total
```

这个函数返回的就是**总分**。它同时也是我们后面每一步的守门员：任何新路线，先过 `validate`，过不了就直接扔掉。

### 2.3 模型训练与超参数调优

这个比赛没有神经网络可训，所谓「训练」其实是**建搜索结构**。我们用了三路互补的办法，思路分别是：小的直接查表、大的翻译给经典求解器、所有结果再做局部压缩。

**第一路：反向建表，小状态直接查。** 从目标状态出发做广度优先搜索（BFS），把所有「两步以内能到达」的状态都存进字典，键是状态本身，值是解法。因为搜索是从目标往外走的，反过来读就是解法，所以一步都不用额外算：

```python
from collections import deque

def inverse_token(t):
    """取逆：d2 的逆是 -d2，-d2 的逆是 d2。"""
    return t[1:] if t.startswith('-') else '-' + t

radius = 2
table = {TARGET.tobytes(): ()}                 # 键用字节串，比数组可哈希、比元组快
queue = deque([(TARGET.copy(), ())])
while queue:
    state, forward = queue.popleft()
    if len(forward) == radius:                 # 只展开到指定深度，防止爆表
        continue
    for token, p in G.items():
        nxt = state[p]
        key = nxt.tobytes()
        if key in table:                       # 已经更短地到达过，跳过
            continue
        word = forward + (token,)
        # 正向走来的路，倒过来并逐个取逆，就是这个状态的解
        table[key] = tuple(inverse_token(t) for t in reversed(word))
        queue.append((nxt, word))
```

建完表之后，`table.get(状态)` 就是一条保证合法的解，查一次是常数时间。代价是内存：深度每加一层，状态数大约乘 18。所以这一路只适合「已经很接近复原」的状态。

**第二路：把数据翻译给 Kociemba 两阶段求解器。** Kociemba 算法是魔方界的经典方法，它先把魔方解到「只差最后几步」的中间态，再用一张预计算表收尾，对普通三阶魔方非常快。但它只认**六个面的颜色**，也只认它自己那套 URFDLB 转动名字，跟这个比赛的 54 个编号、18 种带符号转动完全对不上。

所以我们得造一座桥，也就是一个**共轭映射**：找出「比赛的哪个编号对应 Kociemba 的哪个贴纸位置」，以及「比赛的哪个转动名对应 Kociemba 的哪个面」。这件事不能靠猜，必须验证——把映射代进去，检查两边的转动是否完全一致：

```python
def derive_adapter(moves):
    """构造并校验 54 个编号到 URFDLB 贴纸的映射；对不上就返回 None，宁可不做也不乱翻译。"""
    standard = standard_quarter_turns()   # 从 kociemba 包里读出六个标准四分之一圈
    # 比赛的面顺序是 U,F,R,B,L,D；Kociemba 是 U,R,F,D,L,B，所以要换位
    face_map = [0, 2, 1, 5, 4, 3]
    outer = ['d2', 'r0', 'f0', 'd0', 'r2', 'f2']   # 比赛里的六个外层转动

    # 每个面有两个方向可选，6 个面共 2^6 种组合，逐个试
    for signs in itertools.product([1, -1], repeat=6):
        b = [standard[face] if sign == 1 else np.argsort(standard[face])
             for face, sign in zip('URFDLB', signs)]
        a = [moves[name] for name in outer]
        mapping = {9 * face + 4: 9 * face_map[face] + 4 for face in range(6)}  # 先钉死 6 个中心
        okay = True
        for seed, destinations in [(0, [0, 2, 6, 8]), (1, [1, 3, 5, 7])]:
            orbit = None
            for destination in destinations:
                orbit = propagate_map(seed, destination, a, b, face_map)
                if orbit is not None:
                    break
            if orbit is None:
                okay = False
                break
            mapping.update(orbit)
        if not okay or len(mapping) != 54 or len(set(mapping.values())) != 54:
            continue
        label_map = np.array([mapping[i] for i in range(54)], dtype=np.intp)
        # 最终校验：映射之后，两边的每个转动必须完全等价
        if all(np.array_equal(label_map[x], y[label_map]) for x, y in zip(a, b)):
            clockwise = {face: name if sign == 1 else '-' + name
                         for face, name, sign in zip('URFDLB', outer, signs)}
            return label_map, clockwise
    return None, None
```

这段代码的逻辑是：先钉死 6 个中心块（中心块不会互换，是最可靠的锚点），再从角块和棱块各取一个做「种子」，沿着转动把整个对应关系扩散出去。扩散过程中只要出现矛盾（比如两个编号被映到同一张贴纸），这个组合就作废。最后还要做一次全量校验——**映射对不对，不看它长得像不像，只看它能不能让 36 个转动逐一等价**。

**第三路：中心块归位。** 比赛允许转中层，而转中层会让中心块换位置。Kociemba 默认中心块是固定的，所以要先把中心块拧回标准朝向，再交给它求解，最后把这段「归位」的动作也算进总步数：

```python
def center_frames(moves):
    """枚举所有能通过「只转中层」到达的中心块朝向，并给出每个朝向的最短归位路线。"""
    centers = np.array([4, 13, 22, 31, 40, 49])   # 六个中心块的位置
    identity = np.arange(54, dtype=np.uint8)
    paths = {tuple(identity[centers]): []}
    queue = deque([(identity, [])])
    while queue:
        state, path = queue.popleft()
        for name, perm in moves.items():
            if not name.endswith('1'):           # 只走中层转动（名字以 1 结尾的那些）
                continue
            new = state[perm]
            key = tuple(new[centers])
            if key not in paths:
                paths[key] = path + [name]
                queue.append((new, path + [name]))
    # 存的是「从当前朝向走到标准朝向」的逆路线
    return {key: inverse_word(path) for key, path in paths.items()}
```

**超参数就是预算。** 这个方案里真正需要调的是几个「花多少时间」的参数：总时长 `HOURS=2`、轮数 `PASSES=3`、单次两阶段求解的限时 `SECONDS_PER_TWO_PHASE_ATTEMPT=4`。我们的经验是：限时太短，Kociemba 经常来不及；限时太长，1003 个状态就跑不完。4 秒是折中值，配合「超时就跳过、保留已有结果」的策略，不会因为个别难题拖垮整轮。

主循环的写法是「**只接受更短的合法解**」，第几轮就随机加几步前缀再试，相当于给求解器一点扰动，避免每次都被引到同一条路上：

```python
def accept(pid, word, method, optimal=False):
    """任何新路线都要独立重放一次；只有更短才替换当前记录。"""
    check = validate(states[pid], word, moves)
    audit[pid]['attempts'] += 1
    if not check['valid']:
        rejected.append(dict(initial_state_id=pid, method=method, path='.'.join(word)))
        return
    if pid not in best or len(word) < len(best[pid]):
        best[pid] = list(word)
        audit[pid].update(method=method, length=len(word), valid=True,
                          improvement=(audit[pid]['reference_length'] or 0) - len(word))
    if optimal:
        audit[pid].update(optimal=True, lower_bound=len(word))

for round_index in range(passes):
    for index, pid in enumerate(selected):
        if time.monotonic() >= deadline:      # 到点就停，结果照样可用
            break
        if audit[pid]['optimal']:             # 已经证明最优的，不再浪费时间
            continue
        round_prefix = [] if round_index == 0 else [rng.choice(list(moves)) for _ in range(round_index)]
        candidate, error = two_phase_candidate(states[pid], round_prefix, moves, frames,
                                               label_map, clockwise, timeout)
        if candidate is not None:
            accept(pid, shorten_windows(candidate, moves, sphere), f'two_phase_pass_{round_index+1}')
```

**局部压缩。** 不管是查表还是求解器给的路线，里面常常有冗余：同一层连着拧好几下、或者一小段其实可以用更短的方式达到同样效果。我们用一个「三层的完整邻域表」去替换长度不超过 10 的片段：

```python
def shorten_windows(word, moves, sphere, max_window=10):
    """在路线里找一段，如果表里存在等价但更短的写法，就换掉；反复做直到不再变化。"""
    word = reduce_axis_runs(word)          # 先做一次基础化简：合并同层连续转动
    for _ in range(4):                     # 最多四轮，防止无限来回改
        changed = False
        for start in range(len(word)):
            permutation = np.arange(54, dtype=np.uint8)
            replacement = None
            for stop in range(start, min(len(word), start + max_window)):
                permutation = permutation[moves[word[stop]]]   # 累积这一小段的净效果
                shorter = sphere.get(permutation.tobytes())
                if shorter is not None and len(shorter) < stop - start + 1:
                    replacement = (stop + 1, shorter)          # 记住最后一个更短的替换
            if replacement:
                stop, shorter = replacement
                word = reduce_axis_runs(word[:start] + list(shorter) + word[stop:])
                changed = True
                break
        if not changed:
            break
    return word
```

注意 `permutation` 是「这一段路线的净效果」，不是走过哪些状态。这样哪怕这一小段绕了远路，只要能用一个三层的短词表达同样的净效果，就能被压掉。

### 2.4 模型测试与指标报告

所有路线都算好之后，我们做一次**与验收账本无关的独立复验**——把 `best` 里每条路线重新跑一遍，任何一条不合法就删掉，绝不带着一条错答案交上去：

```python
# 最终复验：重新读 state、重新 replay，不信任前面任何中间记录
for pid in list(best):
    check = validate(states[pid], best[pid], moves)
    audit[pid]['valid'] = check['valid']
    if not check['valid']:
        rejected.append(dict(initial_state_id=pid, method='final_replay',
                             reason=check['reason'], path='.'.join(best[pid])))
        del best[pid]

summary = write_results(output_dir, states, best, audit, rejected, calibration, started)
print(json.dumps({'stage': 'finished', **summary}, indent=2))
```

导出时我们把文件名直接写成 `submission_<总分>.csv`，好处是**扫一眼文件名就知道这份提交值多少分**，不会传错版本：

```python
# 空路线要写成一个带引号的空格，否则 Kaggle 的预检查会因为空单元格报错
submission = pd.DataFrame([
    {'initial_state_id': pid, 'path': '.'.join(best[pid]) or ' '}
    for pid in sorted(best)
])
total = sum(map(len, best.values()))
submission.to_csv(output_dir / f'submission_{total}.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

# 写完之后立刻读回来比对，确认点号拆分结果和内存里完全一致
serialized = pd.read_csv(output_dir / 'submission.csv')
serialized_tokens = {int(pid): path.strip().split('.') if path.strip() else []
                     for pid, path in serialized.itertuples(index=False, name=None)}
assert serialized_tokens == best
```

我们核对过几个关键数字：官方参考答案经我们的验证器检查，总分正好是 **500,620**；公开的 `source333_submission.csv` 经同样的检查，总分是 **21,372**，全部 1003 条都合法。这两个数字帮我们确认了两件事——**验证器本身是对的**（它能把已知正确的文件判为正确），**以及公开文件确实大幅优于官方参考**。

在这个基础上，我们的策略是「择优合并」：对每个状态，取「自己搜出来的合法解」和「公开参考解」里更短的那一个。所以最终总分不会高于 21,372，但**这并不意味着我们独立解出了全部 1003 个状态**——大部分功劳来自那份公开文件，我们自己的深度搜索只在其中一小部分状态上有效。

关于「最优」，报告里写得很清楚：

```python
report = {
    'rows': len(states),
    'positions': 54,
    'generators': 18,
    'original_reference_total': 500620,
    'public_short_fallback_total': 21372,
    'score': total,
    'all_endpoints_valid': True,
    'raw_search': 'Exact reverse BFS, depth 2; reference-free lookup',
    'optimality': '参考长度只是上界，没有给出一般的最优性证明。',
}
```

也就是说：只有那些被「深度 3 的邻域」直接命中的状态，我们才敢标 `proved_optimal`；两阶段求解器和局部压缩给出的长度，一律只能叫**上界**。这个区分很重要，因为 Kociemba 的标准度量跟这个比赛的「分层转动度量」并不是一回事，直接套用它的步数会算错。

---

## 三、总结

本文中我们完成了 3×3×3 分层转动挑战的完整流程。我们首先用 EDA 摸清了数据：1003 个待解状态、54 个互不相同的贴纸编号、18 种带符号的四分之一圈，官方参考总分 500,620，公开短解总分 21,372。其次我们把输入规范成「必须 54 个不重复编号 + 合法 token」的形式，并写了一个只认「完整重放是否回到 0..53」的验证器，空路线统一用一个带引号的空格表示。接着我们搭了三路互补的搜索：深度 2 的反向 BFS 查表负责小状态，经过严格共轭校验的 Kociemba 两阶段适配器负责大状态，三层的邻域表负责把已有路线做局部压缩。然后我们用「只接受更短且合法的解」的主循环，在 2 小时、3 轮、单次 4 秒的预算下反复尝试，并给每轮加随机前缀来避免重复走同一条路。最后我们做了与账本无关的独立复验，把文件名写成 `submission_<总分>.csv`，并明确标注哪些解是「证明最优」、哪些只是「目前较短」。

这个比赛给我们的最大启发是：**这类题目的胜负手不在模型，而在「验证」和「翻译」**。没有可靠的 `replay`，任何搜索都是自欺欺人；没有经过等价性校验的适配器，把经典求解器硬套过来只会得到一堆看起来对、实际错一位的答案。至于「总分好看」这件事，也要诚实地拆开讲——是哪一路搜索贡献的、有多少是公开文件的功劳，都得说清楚。
