# 2026 IEEE Big Data 交通流基准赛（TrafficFlowBench）

## 一、比赛简介

### 1.1 比赛背景

这个比赛来自 2026 年 IEEE Big Data 会议，名字叫 **TrafficFlowBench（交通流基准）**。它给出一批高速公路检测器的历史数据，要求参赛者同时完成四个互相关联的任务。

先说数据长什么样。比赛包含 **10 条有方向的高速公路走廊**，每条走廊上每隔 0.1～0.4 公里装一个检测器，每 5 分钟记录一次**速度、流量、占有率**，外加一个数据质量字段。一条走廊一天有 288 个时间格（24 小时 × 12），数据覆盖 **273 天**。也就是说，全部数据是「走廊 × 链路 × 时间格」组成的一张巨大的三维表。

这些数据是**模拟器生成的**，不是真实公路。这一点非常重要：它意味着数据里的物理规律比现实世界更「干净」——比如自由流时所有车都严格开在 117.5 km/h，基本通行图（fundamental diagram，描述「流量—密度—速度」关系的曲线）是一个完美的三角形，拐点之前没有任何弧度。理解这一点，能帮我们判断哪些规律可以放心用、哪些不能想当然。

比赛的四个任务是：

| 任务 | 内容 | 权重 |
| --- | --- | --- |
| Task 1 | 状态重建：每天随机挖掉 20%/30%/50% 的格子，把它们的速度和流量补回来 | 0.35 |
| Task 2 | 排队预测：给 60 分钟历史，预测未来 30 分钟每条链路是否排队 | 0.30 |
| Task 3 | 物理一致性：在 Task 1 的答案上检查车辆守恒和通行图 | 0.15 |
| Task 4 | OD 反推：根据路段计数反推各条路径的流量 | 0.20 |

难度在于：**这四个任务不是独立的**。Task 3 直接拿 Task 1 的答案打分，所以 Task 1 补得离谱，Task 3 一定跟着崩。而且比赛只要求 30 分钟内的预测，不要求预测明天或下个月——这看起来简单了，实际上意味着「长期趋势」这套方法完全用不上，必须老老实实做短时补全和短时判别。

### 1.2 比赛目标

最终要提交一个 `submission.csv`，它是由 `submission_key.csv` 这张「答题卡」决定的：每一行是一个待填的格子，标明它属于哪个任务、哪条走廊、哪个时间、哪条链路。官方样本的规模是 **6,980,503 行**，拆开来是：

- state 任务 6,735,795 行（速度 + 流量）；
- queue 任务 174,000 行（0/1）；
- odme 任务 70,708 行（路径流量）。

我们要交的列是 `submission_id`、`task`、`speed_kmh`、`flow_vph`、`queue_pred`、`path_flow`——每个任务只填自己那几列，其余留空（写 0.0）。**行数、顺序、自然键必须和答题卡完全一致**，这是硬性格式要求。

### 1.3 评价标准

总分是四个任务分数的加权和：

```
S_total = 0.35 × S_state + 0.30 × S_queue + 0.15 × S_physics + 0.20 × S_ODME
```

四个分项各自怎么算，需要逐个说清楚：

**S_state（状态重建）** 由速度误差和流量误差两部分组成，按「区域」分别算再平均：

```
S_state = 0.54 × max(0, 1 − RMSE_v / 25) + 0.46 × max(0, 1 − RMSE_q / 600)
```

速度误差除以 25（km/h）、流量误差除以 600（veh/h），意思是「误差达到这个量级，这一项就得 0 分」。所以速度每差 1 km/h，大约损失 0.54/25 ≈ 0.022 的分数；流量每差 1 veh/h，损失约 0.46/600 ≈ 0.00077。**速度比流量金贵得多**。

**S_queue（排队预测）** 用**时空 IoU**：把预测的「排队格子」和真实的「排队格子」当成两张图，算交集除以并集。按窗口算，再对 onset（拥堵开始）和 ongoing（拥堵持续）两类条件取平均。

**S_physics（物理一致性）** 是两项的加权：`S_physics = (S_FD + 2 × S_LWR) / 3`，其中 S_FD 衡量预测点离三角形通行图有多远，S_LWR 衡量车辆守恒（相邻时刻的车辆数变化是否等于流入减流出）。

**S_ODME（OD 反推）** 由四项组成：路径流量误差 0.45、链路拟合 0.25、偏离基准 0.15、终点吸引分布 0.15。

绝对分数**越高越好，满分 1.0**。榜单上 0.85 已经很强，0.87 属于前列，公开榜首在 0.928 左右。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**先把四个任务拆开各自做到最好，再用「哪些结论是量出来的」来决定往哪里投时间。** 我们首先做 EDA 摸清数据形态，其次建立插值基线并做数据增强，接着分别训练状态重建模型和排队分类模型、解 OD 反推，然后针对 Task 3 做密度后处理，最后组装、校验并提交。

### 2.1 数据探索（EDA）

第一步是把数据目录结构摸清楚。比赛发布包里有 `config/corridors.json`，它列出了所有走廊和它们所属的「家族」：

```python
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd

# 数据可能挂在几个位置，按顺序找第一个含 corridors.json 的目录
candidates = [
    '/kaggle/input/2026-ieee-big-data-traffic-flow-bench/kaggle_public',
    '/kaggle/input/competitions/2026-ieee-big-data-traffic-flow-bench/kaggle_public',
]
RELEASE = next(Path(c) for c in candidates if (Path(c) / 'config' / 'corridors.json').exists())

manifest = json.loads((RELEASE / 'config' / 'corridors.json').read_text(encoding='utf-8'))
PANELS = [p['corridor_id'] for p in manifest['panels']]
print(f'走廊数量：{len(PANELS)}')      # 10
print(PANELS)
```

确认是 10 条走廊之后，我们读一条走廊某一天的原始状态表，看一条记录到底包含什么：

```python
panel, date = 'D12_I5_N', '2031_01_23'
f = glob.glob(f'{RELEASE}/corridors/{panel}/train/mainline_states/*/'
              f'synthetic_mainline_{date}.parquet')[0]
d = pd.read_parquet(f)

# 只有 is_score_eligible 为真的格子才参与打分，先过滤掉
d = d[d.is_score_eligible.astype(bool)]
print(d[['timestamp', 'link_id', 'speed_kmh', 'flow_vph']].head())

# 用「流量 ÷ 速度」估算密度（veh/km），这是后面所有物理检查的基础量
d = d.dropna(subset=['speed_kmh', 'flow_vph'])
d['density'] = d['flow_vph'] / d['speed_kmh'].clip(lower=1.0)
print('密度分位数：', d.density.quantile([0.05, 0.5, 0.95]).round(1).to_dict())
print('速度分位数：', d.speed_kmh.quantile([0.05, 0.5, 0.95]).round(1).to_dict())
```

这段代码透露了三个关键事实：

1. **每天 288 个时间格**，5 分钟一格，没有秒级数据；
2. **存在 `is_score_eligible` 字段**，不是所有格子都参与评分，特征里必须把这个「质量」也告诉模型，否则模型会把一个「有值但不计分」的读数当成可靠观测；
3. **密度可以由流量和速度直接算出来**，不需要额外数据。这个换算在后面 Task 3 的守恒检查里会反复用到。

我们还把一条走廊一天的速度画成「时间—空间」热力图（横轴是公里数，纵轴是小时）。图上一眼能看到两块明显的「楔形」：一块早上 7:30 从 33 公里处的瓶颈生出，尾部一路向上游拉长十几个小时；另一块中午从 67 公里处开始。**这正是拥堵的形状——头部固定不动，尾部缓慢向上游生长。**

这些观察直接决定了后面的建模策略：拥堵是「局部、连续、有固定出生地」的，所以补全和预测都应该围绕**时空邻域**做，而不是指望一个全局公式。

### 2.2 数据清洗

这份数据的「脏」不在于缺失值，而在于**缺失是人为制造的**，而且训练时看到的缺失模式和测试时不一样。所以我们的清洗分三步：建插值基线、造缺失增强、加物理边界。

**第一步，时间插值基线。** 速度和流量在时间上高度连续，相邻 5 分钟的差值很小，所以直接用「前后最近的已知值线性插值」就能得到一个不差的起点。模型不从这个基线开始学，而是只学**残差**（真实值减基线），这样学习任务轻得多：

```python
def temporal_baseline(frame, link_ids, col):
    """按链路做时间方向线性插值，得到 (288, 链路数) 的基线矩阵。"""
    mat = frame.pivot_table(index='timestamp', columns='link_id', values=col)
    mat = mat.reindex(columns=link_ids)
    # 时间上前后插值；两端的空缺用最近值兜底
    mat = mat.interpolate(method='time', limit_direction='both')
    return mat.to_numpy(dtype=np.float32)

# 对速度和流量各做一次，得到 v0 和 q0
v0 = temporal_baseline(d, sorted(d.link_id.unique()), 'speed_kmh')
q0 = temporal_baseline(d, sorted(d.link_id.unique()), 'flow_vph')
```

选「时间方向」而不是「空间方向」，是因为同一条链路前后时刻的相关性远高于同一时刻相邻链路——车流是随时间演化的，不是瞬间横向传播的。

**第二步，缺失增强。** 训练集是完整的，测试集却被挖掉了 20%/30%/50%。如果直接拿完整数据训练，模型会过度依赖「周围都有值」这个假象。所以我们**在训练时主动制造缺失**：每天随机选一个遮挡率，再按这个比例把合格的速度/流量读数抹掉，然后在这个残缺版本上算特征。标签保持不变——我们改的是「模型能看到什么」，不是编造交通数据。

**第三步，物理边界。** 预测出来的速度不能 ≤ 0，否则 Task 3 算密度时会除以零；流量不能为负。所以最后统一裁剪：速度夹在 5～135 km/h，流量不小于 0。这一步看起来粗暴，但它同时保护了 Task 1 和 Task 3 两项分数。

```python
# 物理护栏：速度下限 5 km/h，避免 q/v 出现无穷大；上限 135 km/h 超过自由流速度
state_df['speed_kmh'] = np.clip(state_df['speed_kmh'], 5.0, 135.0)
# 流量不能是负数
state_df['flow_vph'] = np.clip(state_df['flow_vph'], 0.0, None)
```

### 2.3 模型训练与超参数调优

**Task 1：两个深度的 CatBoost 学残差。**

特征分两块，一共 94 个。第一块 54 个，描述时间（几点、星期几）、链路属性（长度、车道数、自由流速度、通行能力）、插值基线值、到最近观测的距离，以及上下游各两条有检测器的链路。第二块 40 个，来自 **10 个时间偏移**：±5、±10、±15、±30、±60 分钟。每个偏移都取速度、单车道流量、流量速度比，以及一个「这个读数是否计分」的标志。

这 40 个特征是这个方案的核心。因为 Task 1 是**离线补全**，未来的观测是拿得到的——一个被挖掉的格子，它前后各一小时的值都可以用来判断它该是多少。

```python
OFFSETS = (-60, -30, -15, -10, -5, 5, 10, 15, 30, 60)   # 单位：分钟

def offset_features(frame, link_ids, offsets=OFFSETS):
    """把每个时间偏移上的速度/流量/密度比搬成特征列。"""
    out = {}
    pivot_v = frame.pivot_table(index='timestamp', columns='link_id', values='speed_kmh')
    pivot_q = frame.pivot_table(index='timestamp', columns='link_id', values='flow_vph')
    pivot_e = frame.pivot_table(index='timestamp', columns='link_id', values='is_score_eligible')

    for off in offsets:
        # 5 分钟一格，所以偏移量要换算成格数
        steps = off // 5
        v = pivot_v.shift(steps).to_numpy(dtype=np.float32)
        q = pivot_q.shift(steps).to_numpy(dtype=np.float32)
        out[f'v_{off}'] = v
        out[f'q_{off}'] = q
        # 流量除以速度 ≈ 密度，是物理上更有意义的量
        out[f'k_{off}'] = q / np.maximum(v, 1.0)
        # 质量标志：有值但不计分的读数，可信度要打折
        out[f'e_{off}'] = pivot_e.shift(steps).to_numpy(dtype=np.float32)
    return out
```

模型学的是两个残差：`r_v = v − v0` 和 `r_q = q/L − q0`，其中 L 是车道数。**流量按车道归一化**，是因为比赛的流量误差是按单车道算的；一个三车道的链路，基线 500 veh/h/lane 加上残差 40，还原回去就是 1620 veh/h。

我们训练了两个 CatBoost 回归器，损失都是 RMSE：一个深度 8、跑 9000 轮，一个深度 10、跑 4500 轮，学习率都是 0.06。两个深度捕捉不同层级的特征交互，最后**等权平均**：

```python
from catboost import CatBoostRegressor

def fit_residual_models(X, r_v, r_q):
    """分别对速度和流量残差各训练两个深度的模型。"""
    models = {}
    for target_name, y in (('v', r_v), ('q', r_q)):
        for depth, rounds in ((8, 9000), (10, 4500)):
            m = CatBoostRegressor(
                loss_function='RMSE',
                depth=depth,
                iterations=rounds,
                learning_rate=0.06,
                random_seed=42,
                verbose=False,
            )
            m.fit(X, y)
            models[(target_name, depth)] = m
    return models

def predict_state(models, X, v0, q0, lanes):
    """两个深度等权平均，再把残差加回基线，流量乘回车道数。"""
    v_hat = v0 + 0.5 * models[('v', 8)].predict(X) + 0.5 * models[('v', 10)].predict(X)
    q_hat = lanes * (q0 + 0.5 * models[('q', 8)].predict(X) + 0.5 * models[('q', 10)].predict(X))
    return np.clip(v_hat, 5, 135), np.clip(q_hat, 0, None)
```

用 6 月到 1 月的数据、按刚才说的遮挡增强，一共造出 **367.5 万行**训练样本；2 月用一套固定的遮挡方案、42 万个目标做留出验证。固定留出集是为了让不同版本的模型可比——如果每次验证的遮挡都不一样，分数变化就分不清是模型变好了还是运气变好了。

结果，2 月的状态代理分数从 **0.934905 提升到 0.941802**，十条走廊全部有提升。

**Task 2：规则和树模型分工。**

排队预测的第一个发现来自数据本身：**在官方选出的 onset 窗口里，拥堵只出现在最后一个时间步（T+30）**。40 个官方训练窗口里有 39 个都是这样，起点时刻一个排队链路都没有，而被预测为排队的那几条链路，在起点时速度比阈值还高 40～58 km/h。

这意味着两件事：第一，**在 T+5 到 T+25 之间预测「排队」一定是假阳性**，所以那几步直接输出 0；第二，onset 根本不可能靠「速度下降趋势」预测——真值只在最后一刻才出现。

于是 onset 用「**固定瓶颈的地点记忆**」：统计每个链路在每个小时里出现排队的频率，取频率最高的前 K 条，只在 T+30 那一步点亮。再加一道闸门：如果起点时刻流量/通行能力的最大值已经 ≥ 1（超容量却没排队），说明这个窗口不会形成排队，直接全 0。这条闸门在训练集上的分离度是 0.994 对 1.003，非常干净。

```python
HORIZON, HIST_CELLS = 6, 12          # 未来 6 步（30 分钟），历史 12 格（60 分钟）

def predict_onset(S, F, table, cap, hour, topk=3, gate_ratio=1.01):
    """S/F 是 (12, 链路数) 的历史矩阵；返回 (链路数, 6) 的 0/1 预测。"""
    valid_last = np.isfinite(S[-1])
    freq = table['freq'][:, hour]                 # 该小时的历史排队频率
    # 闸门：起点就超容量却没排队 → 这个窗口不会形成排队
    if np.isfinite(F[-1]).any():
        ratio = np.where(valid_last, F[-1], np.nan) / np.where(valid_last, cap, np.nan)
        if np.nanmax(ratio) >= gate_ratio:
            return np.zeros((len(freq), HORIZON), bool)

    # 只在最后一格点亮历史最常堵的前 K 条链路
    cands = np.where(table['sensored'] & (freq > 0))[0]
    cands = cands[np.argsort(-freq[cands])][:topk]
    pred = np.zeros((len(freq), HORIZON), bool)
    pred[cands, HORIZON - 1] = True
    return pred
```

ongoing 窗口是另一回事：拥堵已经存在，任务变成判断它下一步还在不在。这个用 LightGBM 做逐（链路, 步）分类，特征是历史速度统计、与阈值的距离、减速度、历史排队数、流量/容量比、瓶颈频率、位置、小时、步号。训练集是 80 个官方窗口、8.7 万行、正例约 8%。

单看 IoU，规则在 onset 上更强（0.440 对 0.322），LightGBM 在 ongoing 上强得多（0.638 对 0.442），所以最终用**混合**：onset 走规则，ongoing 走模型，整体 IoU 从 0.441 提到 **0.539**。

最后再加一道「**一维空间形态学闭运算**」：如果一条链路左右两边都判为排队，中间那条也判为排队。这符合高速公路上拥堵连续分布的物理常识，能把孤立的噪声点和被切断的队列接回去。

**Task 4：非负最小二乘。**

OD 反推的本质是解一个线性方程组：每条路径的流量乘以「路径是否经过某链路」的 0/1 矩阵，加起来应该等于观测到的链路计数。未知数比方程多，所以加一个 L2 正则，把解往「弱先验」拉：

```
min_{f≥0}  ½‖Af − c‖² + (λ/2)‖f − b‖²
```

关键是**必须用当前评分 split 自己的链路计数**，而不是训练月份的。这一点是这道题最容易犯的错：题目给的 `base_od.csv` 是私有月份的表，拿它来拟合公开月份，分数会莫名其妙地低。

```python
from scipy.optimize import nnls

def solve_odme_nnls(A, counts, measured, prior, lam=0.05):
    """A: 链路×路径关联矩阵；counts: 观测计数；prior: 弱先验路径流量。"""
    # 把正则项写成「额外的行」：sqrt(lam)*I 和 sqrt(lam)*prior
    # 这样带正则的最小二乘就变成标准 NNLS 形式，直接调库即可
    n_path = A.shape[1]
    aa = np.vstack([A[measured], np.sqrt(lam) * np.eye(n_path)])
    bb = np.concatenate([counts[measured], np.sqrt(lam) * prior])
    f, _ = nnls(aa, bb)          # 自动保证所有路径流量非负
    return f
```

NNLS 的好处是**天然满足「流量不能为负」**这个物理约束，不用先解再裁剪。我们把解出来的路径流量和官方 `public_reference` 对比，差异小于 1e-8——说明求解器本身没问题，剩下的分差全在 Task 1 和 Task 2 上。

### 2.4 模型测试与指标报告

先看主线成绩：把状态模型、排队混合模型和 NNLS 组合起来，公开榜（3 月）拿到 **0.86517**，在 320 支队伍里排第 48（2026 年 10 月 5 日的记录）。

但真正值得讲的是**后面的 Task 3 后处理**，它把分数从 0.86517 推到了 **0.87141**。

思路来自对 Task 3 打分公式的拆解。官方从 Task 1 的答案反算密度：

```
N = (q / v) × L
```

然后在相邻 5 分钟之间检查守恒：`ΔN` 应该等于边界流量差 `rhs`。问题是，被挖掉的格子没有观测，`q̂/v̂` 的误差会直接变成 `N` 的误差。而**密度在时间上极其连续**——相邻两步的变化远小于 1%。所以与其相信模型算出来的 `q̂/v̂`，不如**在前后已知的 N 之间做插值**，把预测拉过去。

```python
def interp_density(N, ok, rows, cols):
    """在时间方向找前后最近的已知格子，按位置线性插值出 N。"""
    T = N.shape[0]
    pos = np.arange(T)[:, None]
    # 前一个已知位置 / 后一个已知位置
    prev = np.maximum.accumulate(np.where(ok, pos, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, pos, T)[::-1], axis=0)[::-1]
    p, n = prev[rows, cols], nxt[rows, cols]
    has = (p >= 0) & (n < T)
    pp, nn = np.maximum(p, 0), np.minimum(n, T - 1)
    t = (rows - pp) / np.maximum(nn - pp, 1)          # 在两个锚点之间的相对位置
    Ni = N[pp, cols] * (1 - t) + N[nn, cols] * t      # 线性插值
    return np.where(has, Ni, np.nan), nn - pp - 1     # 第二项是「中间隔了几个空格」
```

拿到插值密度之后，把它和模型密度取对数差，再按比例分摊到速度或流量上：

```python
def adjust_density(v_hat, q_hat, Ni, run, L, w=1.0, alpha=1.0, max_run=2):
    """把预测密度拉向插值密度。alpha=1 全部靠改流量，alpha=0 全部靠改速度。"""
    k_hat = q_hat / np.maximum(v_hat, 1.0)
    k_i = Ni / L
    ok = np.isfinite(k_i) & (k_i > 0) & (k_hat > 0) & (run <= max_run)
    # 对数空间的差，乘上强度 w
    d = np.where(ok, np.log(k_i) - np.log(np.maximum(k_hat, 1e-9)), 0.0) * w
    v_new = np.clip(v_hat * np.exp(-(1 - alpha) * d), 1, 180)
    q_new = np.clip(q_hat * np.exp(alpha * d), 0, 150000)
    return v_new, q_new
```

这里有三个超参数要选：强度 `w`（改多少）、分流 `alpha`（改速度还是改流量）、最大间隔 `max_run`（跨太多空格就不信插值）。我们没有拍脑袋定，而是在 2 月留出集上做网格搜索：`w ∈ {0.25, 0.5, 0.75, 1.0}`，`alpha ∈ {0, 0.25, 0.5, 0.75, 1}`，`max_run ∈ {1, 2, 4, 999}`，一共 80 组。

评分时直接用官方公式：

```python
# 状态分：速度 RMSE 除以 25，流量 RMSE 除以 600，各自截断到 0
s_state = 0.54 * max(0, 1 - rmse_v / 25) + 0.46 * max(0, 1 - rmse_q_lane / 600)

# 守恒分：ΔN 与 rhs 的偏差，除以真实 ΔN 的总量
s_lwr = max(0, 1 - np.abs(dN - dNt)[tr].sum() / max(np.abs(dNt)[tr].sum(), 1e-9))

# 通行图分：预测点与三角形 FD 曲线的偏差
s_fd = max(0, 1 - np.abs(ql - qfd)[used].sum() / max(np.abs(ql)[used].sum(), 1e-9))

# 估算总分增益（物理分 = (S_FD + 2*S_LWR)/3）
est_lb = 0.35 * dS_state + 0.15 * (dS_FD / 3 + 2 * dS_LWR / 3)
```

我们设了一条纪律：**只有估算增益超过 0.001，才真的去改提交文件**。因为本地测出来的提升，很可能到线上就没了。这个「先量再改」的习惯，帮我们避免了好几次盲改。

最终这一套拿到公开榜 **0.87141**。

关于分项，我们对 0.86520 那一版做过一次精确拆解：`0.86520 − 0.35 × 0.9408 = 0.53593`，也就是排队、物理、OD 三项加起来贡献 0.53593。由此反推出自洽的一组值：

| 任务 | 当前分数 | 满分还能加多少 |
| --- | --- | --- |
| state | 0.941 | +0.021 |
| queue | 0.887 | +0.034 |
| physics | ≈0.66 | +0.051 |
| ODME | ≈0.854 | +0.029 |

这张表非常有用：它告诉我们**物理分是剩余空间最大的一块**（+0.051），但也告诉我们它最难碰——因为本地根本复现不了官方的守恒评分。材料里记录了一个残酷的事实：即使把「真实观测场」直接交上去，在本地回退评分下 S_LWR 也约等于 0。所以我们没有在物理上赌，而是把力气花在密度插值这种「有把握的局部改进」上。

最后是提交组装。`submission_key.csv` 有近 700 万行，直接读进内存再合并容易爆，所以我们**分块流式写出**，边读边匹配边追加：

```python
KEYS = {
    'state': ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'],
    'queue': ['window_id', 'timestamp', 'link_id'],
    'odme':  ['panel', 'departure_time', 'path_id'],
}
VALUES = {'state': ['speed_kmh', 'flow_vph'], 'queue': ['queue_pred'], 'odme': ['path_flow']}

row_count = 0
for chunk in pd.read_csv(DATA / 'submission_key.csv', dtype=str, chunksize=500_000):
    out = chunk[['submission_id', 'task']].copy()
    for col in ('speed_kmh', 'flow_vph', 'queue_pred', 'path_flow'):
        out[col] = 0.0
    for task in KEYS:
        mask = chunk.task == task
        if not mask.any():
            continue
        wanted = pd.MultiIndex.from_frame(chunk.loc[mask, KEYS[task]])
        matched = tables[task].reindex(wanted)      # 按自然键对齐，不是按行号
        assert matched.notna().all().all(), f'{task} 有缺失预测'
        out.loc[mask, VALUES[task]] = matched.to_numpy()
    out.to_csv(SUB, mode='w' if row_count == 0 else 'a',
               header=row_count == 0, index=False)
    row_count += len(out)

# 三条总校验：总行数、各任务行数、取值合法性
assert row_count == 6_980_503
assert counts == {'state': 6_735_795, 'queue': 174_000, 'odme': 70_708}
assert set(queue_df.queue_pred.unique()) <= {0, 1}
```

按**自然键对齐**而不是按行号，是这里最关键的一步。答题卡的行顺序不保证和我们的预测表一致，一旦错位，整份提交就是错的，但格式检查完全发现不了。

---

## 三、总结

本文中我们完成了 2026 IEEE Big Data 交通流基准赛的完整流程。我们首先用 EDA 摸清了数据：10 条走廊、273 天、5 分钟一格，一天 288 个时间格，速度由 117.5 km/h 的自由流和三角形通行图支配，拥堵呈「头部固定、尾部上游生长」的楔形。其次我们建了时间插值基线，用 20%/30%/50% 的随机遮挡做数据增强，并给预测加上速度 ≥ 5 km/h、流量 ≥ 0 的物理护栏。接着我们分任务建模：Task 1 用 54 个基础特征加 10 个时间偏移的 40 个上下文特征，训练深度 8 和深度 10 两个 CatBoost 学残差并等权平均，2 月代理分数从 0.934905 提到 0.941802；Task 2 利用「onset 只在 T+30 出现」这个结构规律，onset 走瓶颈地点记忆加超容量闸门、ongoing 走 LightGBM，混合后 IoU 到 0.539；Task 4 用带 L2 正则的 NNLS，并且坚持用当前 split 自己的链路计数。然后我们针对 Task 3 发现「密度在时间上变化不到 1%」，用前后锚点插值密度去修正 Task 1 的输出，只在估算增益超过 0.001 时才改动提交。最后我们分块流式组装 6,980,503 行提交并用自然键对齐校验，公开榜从 0.86517 提升到 0.87141。

这个比赛给我们的最大启发是：**在这种多任务加权赛里，「知道哪一分能拿、哪一分拿不到」比「模型多强」更重要**。我们做过拆解，物理分还有 0.051 的空间，看起来最诱人，但本地根本没法复现官方评分，盲目投入就是赌博；反倒是排队和状态这两块，只要老老实实挖数据里的结构规律（onset 只出现在最后一步、拥堵不向上游传播、密度在时间上几乎不变），分数就能稳定地涨。**先把评测复现清楚，再决定往哪儿使劲**，这是我们从这场比赛里带走的最重要的一条经验。
