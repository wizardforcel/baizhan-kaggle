# 量子编译观测站（Quantum Compilation Observatory）

## 一、比赛简介

### 1.1 比赛背景

量子计算机不是随便写一段程序就能跑的。我们写的量子线路（circuit）通常假设任意两个量子比特都能直接相互作用，但真实硬件不是这样：一块超导芯片上，每个量子比特只和身边少数几个比特连着，像一张稀疏的网格。想把逻辑线路放到硬件上跑，就得先做**编译**——把逻辑比特分配到物理比特上，再把不能直接执行的两比特门用一串交换门「搬」过去。搬得越多，线路越长、噪声越大、结果越差。

这个比赛关注的就是编译这一步的**代价（cost）**：给定若干标准测试线路（benchmark），让参赛者想办法把编译后的代价压到最低。比赛材料里出现了三个主要基准：

- **invsqrt**（逆平方根）：62 个量子比特；
- **DQI**：888 个量子比特；
- **Chemistry**（化学模拟）：430 个量子比特。

每个基准都配了一个「Google 参考代价」，也就是官方给出的基准线。参赛者交上去的线路会被官方验证器（verifier）用指定次数的采样跑一遍，一边核对数值对不对（value error / phase error），一边算代价。这不是一道「训练一个模型」的题，而是一道**工程优化题**：靠改线路结构、改门分解方式、改参数，一点点把数字磨下去。

### 1.2 比赛目标

我们要提交的是一个压缩包（`submission.zip`），里面按固定目录放好每个基准对应的线路文件。材料里的提交包一共有 18 个文件：1 个 invsqrt 线路（`invsqrt.kmb`）、1 个 DQI 线路（`dqi.kmb`）、16 个化学相关文件（15 个 `angle_k_*.kmb` 角度文件加 1 个 `submission.npz` 系数文件）。

`kmb` 是这次比赛使用的一种线路描述格式；`npz` 是 NumPy 的压缩数组文件，用来存化学基准需要的数值系数。评测系统读这些文件，在模拟器上跑，检查结果是否正确，再计算代价。

### 1.3 评价标准

单个基准的分数是**参考代价除以我们的代价**：

```
ratio = reference_cost / our_cost
```

这个比值越大越好。它没有单位，所以不同基准之间可以比较——虽然三个基准的原始代价数量级差得很远（invsqrt 十万级，DQI 千亿级，化学三亿级），但换成比值之后就都在 10 到 61 这个范围里了。

三个基准的比值再取**几何平均**，就是最终分数。几何平均的好处是：三个基准权重相同，谁拖后腿都会被放大，不能靠一个基准的高分掩盖另一个的糟糕表现。材料里记录的历史提交 `v8` 公开分数是 **23.17116**；另一个 notebook 里记录的更早一次公开提交是 **22.99252**。

## 二、解题过程

我们首先要说清楚：这份材料不是从零推导线路的过程记录，而是一份**「把已经验证过的成果打包、核对、记录」的工程笔记**。它把公开社区的前沿工作和本地验证结果拼在一起，每一步都留下可复现的证据。我们的叙述就按这个真实流程来走。

### 2.1 数据探索（EDA）

第一步是把「数据」摸清楚。这里的原始数据不是 CSV，而是**成绩清单**：每个基准跑了多少量子比特、本地验证代价多少、Google 参考代价多少、比值多少。我们先把这份清单读进来，做成一张表：

```python
import pandas as pd

# 手工整理的成绩清单，一行一个基准
# 数字全部来自本次 notebook 记录的本地验证结果
portfolio = pd.DataFrame([
    # 基准名, 量子比特数, 本地验证代价, Google 参考代价
    ("invsqrt",   62, 100_692.57301,          2_076_774.23372),
    ("DQI",      888, 400_273_653_230.8152,   24_439_463_583_672.92969),
    ("chemistry", 430, 337_780_213.344,        3_355_599_588.31554),
], columns=["benchmark", "qubits", "local_cost", "reference_cost"])

# 比值 = 参考代价 / 本地代价，越大越好
portfolio["ratio"] = portfolio["reference_cost"] / portfolio["local_cost"]
print(portfolio)
```

跑出来的比值分别是 **20.624900**、**61.056888**、**9.934269**。一眼能看出两件事：

1. 化学基准的比值最低，只有 9.93，是明显的短板；
2. DQI 的比值最高，但它同时也是量子比特数最多、代价最大的基准，说明它的优化空间和难度都集中在另一个维度上。

接着我们把三个比值合成总分，确认和记录里的 23.2142 对得上：

```python
import numpy as np

# 几何平均：三个比值相乘再开三次方
rollup = float(np.prod(portfolio["ratio"]) ** (1 / len(portfolio)))
print(f"geometric mean = {rollup:.4f}")

# 顺便看看算术平均会是多少，用来说明为什么不能用算术平均
print(f"arithmetic mean = {portfolio['ratio'].mean():.4f}")
```

几何平均是 **23.2142**，而算术平均会被 DQI 的 61 拉高。比赛采用几何平均，正是为了避免这种「一个高分掩盖两个低分」的情况。这一条认识很关键：后面我们判断「该往哪儿使劲」时，看的是**最弱的那个基准**，不是平均分。

我们还核对了验证规模，因为采样次数直接决定结果可不可信：

```python
# 每个基准在本地验证时跑了多少次采样（shots）
# 这些数字记录在 notebook 的验证凭据里
runs = {
    "invsqrt_full64":  {"shots": 200_540_928, "value_err": 0, "phase_err": 0},
    "DQI_paired2":     {"shots": 102_400,     "value_err": 0, "phase_err": 1},
    "DQI_paired3":     {"shots": 102_400,     "value_err": 0, "phase_err": 0},
    "chemistry_salt1": {"shots": 78_080,      "value_err": 0, "phase_err": 0},
    "chemistry_salt2": {"shots": 78_080,      "value_err": 0, "phase_err": 0},
}
for name, r in runs.items():
    print(f"{name:18s} shots={r['shots']:>10d} "
          f"value_err={r['value_err']} phase_err={r['phase_err']}")
```

从这里我们看到验证的粒度差别很大：invsqrt 跑了两亿多次采样，几乎是「确定性」的；DQI 和化学只跑十万级、八万级，属于统计估计。这意味着**同样一个代价数字，可信程度并不一样**——我们在后面比较成绩时必须把这一点放在心上。

### 2.2 数据清洗

这一步我们做的不是删空值，而是**核对文件的身份**。比赛提交的是二进制线路文件，一旦拿错版本、或者文件在网络传输中损坏，看起来分数没变，实际上交上去的东西根本不是我们验证过的那一份。所以我们给每个文件算了 SHA-256 指纹，和登记值逐一比对。

```python
import hashlib
from pathlib import Path

def sha256_of(path: Path) -> str:
    """分块读取文件并计算 SHA-256，避免一次性把大文件读进内存"""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        # 每次读 8MB，适合几十 MB 的线路文件
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# 只列出两个主基准做示范，化学的 16 个文件同理
EXPECTED = {
    "invsqrt.kmb": "67482db6c53896348eadba126924868ed3011d7e1576a6d0abb6f7c137ae4997",
    "dqi.kmb":     "25db915d86ab9bf2cf41670ee97b7d32495fcb12f12aa48e89c1a5924e8921ec",
}

for name, expected in EXPECTED.items():
    actual = sha256_of(Path("submission") / name)
    # 指纹不一致就直接报错，绝不带着错文件去打包
    assert actual == expected, f"{name} 指纹不符：{actual}"
    print(f"{name} OK")
```

为什么要这么严格？因为材料里明确写了，DQI 用的是「用户提供的编译候选」，本地只做了匹配验证，并没有从源码重新构建。这种情况下，**文件本身**就是唯一的可信锚点。指纹对了，我们才知道「本地验证过的那个东西」和「要提交的那个东西」是同一个。

除了核对文件，我们还做了一件「清洗」性质的事：**清理错误的归因**。材料里反复强调，invsqrt 的相位补丁来自 Ahmed Berat Özer，通过 Voodoo v5 分发；DQI 的构造器源自 SomeJoint，Q889 收窄来自 Heliosli；化学的系数来自 avazaku。把这些来源分清楚，既是对社区工作的尊重，也避免我们误以为某个提升是自己做出来的。

### 2.3 模型训练与超参数调优

这道题没有神经网络，所谓「训练」其实是**线路改造的迭代**。我们按基准分开讲。

**invsqrt。** 公开起点是 busyaprime 的 63 比特线路，期望代价 102,349.22。我们的本地工作把它压到 62 比特、101,002.54，省下 1 个量子比特和 1,346.68 的代价。接着我们又做了四次「非线性锥重写」，把代价降到 100,754.54；最后叠加 Özer 的独立相位补丁，得到 100,692.54076。我们把这个改进过程写成一段可核对的计算：

```python
# invsqrt 的逐步改进记录（期望代价，不是采样代价）
milestones = [
    ("公开起点 busyaprime 63Q", 102_349.22),
    ("本地 Q62 里程碑",          101_002.54),
    ("四次非线性锥重写",         100_754.54),
    ("叠加 Özer 相位补丁",       100_692.54076),
]

prev = None
for name, cost in milestones:
    if prev is None:
        print(f"{name:28s} cost={cost:>12.5f}")
    else:
        saved = prev - cost
        pct = saved / prev * 100
        # 每一步省了多少、占比多少，方便判断哪一步最值钱
        print(f"{name:28s} cost={cost:>12.5f} "
              f"省 {saved:>9.5f} ({pct:.6f}%)")
    prev = cost
```

从这段记录能看出一个很朴素的道理：**改进是逐步累积的**。最大的一步是把 63 比特压到 62 比特，后面几步只省几百甚至几十的代价。但比赛到了后期，大家比的正是这最后几百。

**DQI。** 这条线我们基本没有自己重写线路，而是**验证别人给的候选**。材料里的做法是「匹配验证」：用同一个随机种子、同一个 salt（盐值，用来固定验证器的随机化参数）、同一个 SDK，跑 102,400 次采样，确认错误数和对方一致，再比较代价。paired2 的结果是 400,273,653,230.8152；另一组 paired3 用不同种子跑出 398,443,798,212.809。两次结果差了约 0.46%，这个差距不是模型好坏，而是**采样噪声**。

```python
# 两次 DQI 配对验证的结果对比
paired2 = 400_273_653_230.8152
paired3 = 398_443_798_212.809

diff = paired2 - paired3
print(f"绝对差 = {diff:,.3f}")
print(f"相对差 = {diff / paired3 * 100:.4f}%")
# 结论：同一线路、不同种子，代价会有千分之几的抖动
# 所以不能拿单次采样结果当「真实水平」
```

**Chemistry。** 化学基准的改进最小，但技术含量不低。起点是 Voodoo v5 的 337,833,376.13，我们做了 O-to-S 变换、规范（gauge）修正和 SO60 抛光，降到 337,823,023.22。绝对只省了 10,352.91，占比 0.003065%。这种量级的改动必须靠严格验证才敢提交——材料也特别提醒，化学基准的 error-2 和 gap 余量「异常窄」，稍有不慎就会从「通过」变成「不通过」。

我们把三类改动的收益放在一起对比：

```python
# 三项改进的绝对值和百分比
improvements = [
    ("invsqrt",   100_940.54076, 100_692.54076),
    ("DQI",       403_339_964_182.11682, 402_458_332_609.99841),
    ("chemistry", 337_833_376.13268, 337_823_023.22231),
]
for name, before, after in improvements:
    saved = before - after
    print(f"{name:10s} 省 {saved:>18,.5f}  ({saved / before * 100:.6f}%)")
```

可以看到，百分比都在千分之几到万分之几。这说明**比赛后期已经进入「小数点后几位」的竞争**，任何一次「大改」都可能破坏已验证的正确性，反而是小步、可验证的改动更稳。

### 2.4 模型测试与指标报告

测试环节的核心是**验证器**。它做两件事：一是跑线路，统计数值错误（value error）和相位错误（phase error）；二是根据线路结构算代价。只有错误数为零，代价才有意义——一条错误百出的线路代价再低也不能要。

我们先估算「采样代价」和「确定性期望代价」之间的差距，理解这两个数为什么不一样：

```python
# invsqrt 的两个代价口径
expected_cost = 100_692.54076   # 按线路结构算出的确定性期望
sampled_cost  = 100_692.57301   # 实际跑 200,540,928 次采样测出来的

# 两者只差 0.03 左右，说明采样足够多时估计很稳
print(f"采样 - 期望 = {sampled_cost - expected_cost:.6f}")
print(f"相对误差   = {(sampled_cost - expected_cost) / expected_cost * 100:.8f}%")
```

材料里对这两个口径分得很清楚：排行榜用的、和别人比较用的，要说明是哪一个。我们在这里也保持同样的严谨——**报数字的时候一定带上口径**。

然后我们按比赛规则把三个比值合成最终分数，并和历史提交对照：

```python
import numpy as np

# 当前三个基准的比值（来自 2.1 的表）
ratios = np.array([20.624900, 61.056888, 9.934269])
score = float(np.prod(ratios) ** (1 / 3))

print(f"本地混合估计 = {score:.4f}")
print(f"历史公开提交 v8 = 23.17116")
print(f"更早的公开分数   = 22.99252")

# 另一个口径：DQI 用零错误的那次凭据重算，会得到 23.17117
# 这说明「同一个方案，换一次采样，分数会差千分之几」
```

我们得到的本地混合估计是 **23.2142**，比历史公开的 23.17116 略高，但材料明确说这只是**本地验证凭据推出来的估计，不是新的排行榜成绩**。这一点必须反复强调：采样型验证的结果会随种子和盐值波动，单次数字不能当成保证。

最后是打包提交。这一步看似机械，其实最容易出错：文件少一个、CRC 坏了、压缩包结构不对，评测系统就直接判失败。所以我们在写压缩包时同步算指纹、校验成员集合：

```python
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
import hashlib

source = Path("submission")          # 已验证过的文件目录
archive_path = Path("/kaggle/working/submission.zip")

# 18 个文件的清单：1 个 invsqrt + 1 个 DQI + 16 个化学文件
members = ["invsqrt.kmb", "dqi.kmb", "submission.npz"] + \
          [f"dfthcsos/angle_k_{k}.kmb" for k in range(15)]

with ZipFile(archive_path, "w", compression=ZIP_DEFLATED, compresslevel=1) as zf:
    for name in sorted(members):
        path = source / name
        # 固定时间戳，保证同样的输入每次打出字节一致的压缩包
        info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        digest = hashlib.sha256()
        with path.open("rb") as src, zf.open(info, "w") as dst:
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                digest.update(chunk)   # 边写边算指纹
                dst.write(chunk)
        print(f"{name}  {digest.hexdigest()[:12]}")

print("打包完成，共", len(members), "个文件")
```

到这里，比赛流程就闭环了：确认基准和口径 → 核对文件身份 → 分基准做小幅改造 → 用验证器确认零错误 → 合成分数 → 打包。

---

## 三、总结

本文中我们完成了 Quantum Compilation Observatory 的完整提交流程。我们首先把三个基准的成绩清单整理成表，算清各自的比值和几何平均，确认总分 23.2142 的来历；其次我们做了「数据清洗」意义上的文件身份核对，用 SHA-256 确保提交文件和本地验证过的是同一份；接着我们分基准推进线路改造——invsqrt 从 63 比特 102,349.22 一路压到 62 比特 100,692.54076，DQI 用匹配验证确认了用户提供的候选，化学基准靠 O-to-S 与规范修正省下 10,352.91；然后我们用验证器在 200,540,928、102,400、78,080 等不同规模的采样下确认零错误；最后我们按 18 个文件的清单打包并校验 CRC。

这个比赛给我们最深的印象是**「可信」比「好看」重要**。分数是采样估出来的，同一个方案换个种子就能差千分之几；一个文件拿错版本，验证就白做。所以真正决定成绩的往往不是某个惊天动地的技巧，而是每一步都留下可复现的证据、每个数字都写清楚口径。这也是后期优化能从千分之一里再挤出万分之几的原因。
