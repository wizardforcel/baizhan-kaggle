# 量子编译观测台（Quantum Compilation Observatory）

## 一、比赛简介

### 1.1 比赛背景

量子计算机和普通电脑有一个根本区别：普通电脑的比特非 0 即 1，而量子比特（qubit）可以同时处于 0 和 1 的叠加态。这种能力让量子计算机在某些问题上理论上更快，但也带来一个麻烦——**写在纸上的量子程序，不能直接搬到真机上跑**。

原因在于，纸上的量子程序（叫「逻辑电路」）假设任意两个量子比特都能直接相互作用；而真实的量子芯片上，只有物理上连了线的比特才能互相操作，这种「谁和谁相连」的图叫**拓扑（topology）**。所以必须做一步翻译：把逻辑比特分配到物理比特上，把不能直接做的双比特门用一串能做的门「绕」出来。这一整套翻译工作叫**量子编译（quantum compilation）**，而「把逻辑比特安排到哪个物理比特」这一步叫**初始映射（initial mapping）**。

这个比赛（Quantum Compilation Observatory，简称 QCC）就是围绕着「编译出来的电路够不够省」展开的。它固定了几个基准电路（benchmark），参赛者要提交自己的编译结果，官方用一个**验证器（verifier）**去检查：电路功能是否还和原来等价，以及它的「代价（cost）」是多少。材料里出现了三个基准：

- **invsqrt**：倒数平方根（inverse square root）电路，规模较小，最终做到 62 个量子比特；
- **DQI**：材料里只用缩写称呼，规模很大，做到 888～889 个量子比特；
- **chemistry**：化学模拟电路，430 个量子比特。

这三个基准的难度和评分尺度完全不同——材料特意提醒：「原始代价使用各任务自己的计分定义，**不能跨行直接比较**」。所以最终成绩不是把三个数字加起来，而是把三个「相对倍数」做几何平均。

### 1.2 比赛目标

输入是官方给定的基准电路（以 QASM 等格式描述）和一份 Google 参考实现；输出是**我们编译后的电路文件**，要求：

1. **功能等价**：编译后的电路在验证器上跑指定次数的采样（shots），测出的结果必须和原始电路一致，错误数（value error / phase error）通常要求为 0；
2. **代价尽可能低**：在等价的前提下，验证器算出的 cost 越小越好；
3. **可复现**：提交要附带 seed、salt（盐值，用来固定随机采样）、SDK 版本、artifacts 哈希等，让别人能重跑出同样的数字。

材料里的提交还遵循一个「注册表快照」的约定：notebook 会把当前 main 分支的 commit、artifact 哈希、本地验证凭据（receipt）、上游署名和已知局限一起打包进数据集，保证「我提交的是哪一版、依据是什么」有据可查。

### 1.3 评价标准

评分分两层。

**第一层：单个基准的比值。** 对每个基准，记 Google 参考代价为 $C_{\text{ref}}$，我们本地验证得到的代价为 $C_{\text{local}}$，则该基准的倍数为：

$$
\text{ratio} = \frac{C_{\text{ref}}}{C_{\text{local}}}
$$

这个比值是**无量纲**的（因为分子分母同单位，约掉了），所以三个基准才能放在一起比较。比值越大越好——它表示「我们的实现比 Google 参考省了多少倍」。

**第二层：总分的几何平均。** 三个基准的比值取几何平均：

$$
\text{score} = \sqrt[3]{\text{ratio}_{\text{invsqrt}} \times \text{ratio}_{\text{DQI}} \times \text{ratio}_{\text{chemistry}}}
$$

几何平均而不是算术平均，是为了防止某一个基准的比值特别大就把总分拉飞——它要求三个基准都表现好，总分才高。**分数越高越好。**

材料里给出的三个比值和总分是：invsqrt 20.624900×、DQI 61.056888×、chemistry 9.934269×，几何平均滚出 **23.2142**（作者明确说明这是「本地混合配置的估计值，不是公开分数或保证」）；另一个 notebook 记录的本地估计是 **23.13580**，此前已提交的公开分数是 **22.99252**，更早的 v8 是 **23.17116**。

---

## 二、解题过程

这个比赛不是「训练一个模型」，而是「**一点点把电路改小、把代价压下去，并且每一步都用验证器证明自己没改坏**」。所以我们按 EDA（看清三个基准的家底）、数据清洗（剔除不可信的验证结果）、优化与调参（改电路、调近似）、测试与报告（跑验证、记录 receipt）四步走。

### 2.1 数据探索（EDA）

第一步是把「家底」摊开看：三个基准各自多少比特、参考代价多少、我们当前做到多少、比值多少。材料里的 notebook 第一件事就是定位挂载的数据集，并读出清单文件：

```python
from pathlib import Path
from IPython.display import Image, display

# 数据集可能挂在两个不同路径下（Kaggle 数据集与 notebook 输出），依次尝试
roots = [Path("/kaggle/input/qcc-local-sota-20261031"),
         Path("/kaggle/input/datasets/yoyobar/qcc-local-sota-20261031")]
# 找到真正含有 portfolio-manifest.json 的那个目录，找不到就报错，避免静默用错版本
DATASET = next((p for p in roots if (p / "portfolio-manifest.json").is_file()), None)
if DATASET is None:
    raise FileNotFoundError("Attach the public yoyobar/qcc-local-sota-20261031 dataset.")

# 清单里记录了每个基准的 qubits / 本地代价 / 参考代价 / 比值
display(Image(filename=str(DATASET / "submission-meme.png"), width=680))
```

为什么要先找 `portfolio-manifest.json`？因为这类比赛最容易犯的错不是算法错，而是**版本错**——你以为自己在看最新结果，实际上引用的是三天前的快照。清单文件相当于「这一版到底包含什么」的目录，先确认它，后面的数字才有意义。

把三个基准摊成一张表，我们能立刻看出它们的性格差异：

| 基准 | 量子比特 | 本地验证代价 | Google 参考代价 | 比值 |
| --- | --- | --- | --- | --- |
| invsqrt | 62 | 100,692.57301 | 2,076,774.23372 | 20.624900× |
| DQI | 888 | 400,273,653,230.8152 | 24,439,463,583,672.92969 | 61.056888× |
| chemistry | 430 | 337,780,213.344 | 3,355,599,588.31554 | 9.934269× |

读法很清楚：

- **DQI 的比值最高（61 倍）**，因为它规模最大、参考实现离最优最远，可压缩的空间也最大；
- **chemistry 的比值最低（9.93 倍）**，而且材料反复提醒它的「error-2 和 gap 余量异常窄」，意思是它已经逼近某个边界，稍有不慎就会验证失败；
- **invsqrt 规模最小（62 比特）**，但它的改进被精确到小数点后五位——材料里一次优化只省下 310 的代价，占总量的 0.3%。

接着我们把「总分怎么算」写成代码，方便每次改动后立刻估算：

```python
import math

# 三个基准的（参考代价, 本地代价）二元组
benchmarks = {
    "invsqrt":   (2_076_774.23372,        100_692.57301),
    "DQI":       (24_439_463_583_672.92969, 400_273_653_230.8152),
    "chemistry": (3_355_599_588.31554,      337_780_213.344),
}

ratios = {}
for name, (ref, local) in benchmarks.items():
    ratios[name] = ref / local           # 无量纲倍数：参考比我们贵多少倍
    print(f"{name:9s} ratio = {ratios[name]:.6f}x")

# 几何平均：三个比值相乘再开三次方，避免某个基准一枝独秀拉高总分
score = math.prod(ratios.values()) ** (1 / 3)
print(f"geometric-mean rollup = {score:.4f}")
```

这段代码跑出来就是材料里那个 **23.2142**。我们特意把它单独写出来，因为后面每改一次电路，都要重新算一遍这个数，判断「这次改动到底值不值得提交」。

### 2.2 数据清洗

量子编译里的「数据清洗」不是处理缺失值，而是**筛掉不可信的验证结果**。材料里有一条非常关键的纪律：同一个基准可能跑出好几个分数，但只有「seed、salt、SDK 版本、错误数都对齐」的那一次才能拿来比较。

举个材料里的例子：DQI 有一份「fixed-salt」的 receipt 出现了 **1 个 value error**，它的本地代价是 404,306,639,504.52698；而另一份 **zero-error** 的 receipt 本地代价是 402,458,332,609.99841。前者算出来的总分是 23.13580，后者是 23.17117。两个数字差得不多，但**含义完全不同**——带错误的那个不能当作「等价实现」来宣传。所以我们的清洗规则是：

```python
def accept_receipt(receipt):
    """判断一份验证凭据能不能进入正式统计。
       返回 (是否采纳, 原因)。宁可少一个分数，也不能混入不可比的数字。"""
    # 1) 错误数必须为 0：value error 和 phase error 任意一个非零都意味着功能不等价
    if receipt["value_errors"] != 0 or receipt["phase_errors"] != 0:
        return False, "存在 value/phase 错误，功能不等价"
    # 2) 采样次数必须达到该基准的最低要求
    if receipt["shots"] < receipt["min_shots"]:
        return False, "采样次数不足，统计不可信"
    # 3) seed / salt / SDK 必须与对照实验一致，否则不是同一个实验
    if (receipt["seed"], receipt["salt"]) != (receipt["ref_seed"], receipt["ref_salt"]):
        return False, "seed/salt 不一致，无法与基线对比"
    return True, "ok"
```

清洗的第二个动作是**核对样本量**。三个基准的验证强度不同：invsqrt 的 full64 验证跑了 **200,540,928 shots**，DQI 每次记录跑 **102,400 shots**，chemistry 两个盐值各跑 **78,080 shots**。shots 越多，统计噪声越小；但 shots 也不是越多越好——它直接决定验证时间。所以我们按基准分别设定门槛，而不是一刀切。

第三个动作是**承认局限**。材料反复写「这是本地验证证据（local verifier evidence）」「不是公开分数、预测或保证」。我们把这类限定语原样保留在报告里，因为一旦把「本地估计」说成「排行榜分数」，后面所有的对比都会失真。

### 2.3 模型训练与超参数调优

这是整个比赛的主战场。我们的策略是：**先在公开基础上做「结构性」改进（减比特、改电路结构），再做「数值性」微调（相位、系数），每一步都用验证器卡住等价性。**

**第一步：invsqrt 的比特缩减。** 公开起点是 busyaprime 基于 mrmeows / team selenium 的 **63 比特、期望代价 102,349.22**。我们把它压到 **62 比特、101,002.54**——省下 1 个量子比特和 1,346.68 的代价（1.315771%）。省一个比特为什么这么值钱？因为少一个比特，后续所有的双比特门、映射、路由都可能跟着变简单，代价是指数式往下掉的。

```python
# 伪代码：根据材料描述整理的「减比特 + 重写」流程，不是原始实现
def reduce_qubit(circuit, target_qubits=62):
    # 1) lookup（查表）重写：把可以用经典查表替代的小段落移出量子电路
    circuit = rewrite_lookup_tables(circuit)
    # 2) normalization（归一化）：把重复/冗余的门序列合并
    circuit = normalize_gates(circuit)
    # 3) tail schedule（尾部调度）：重排电路末尾的门，减少跨比特依赖
    circuit = reschedule_tail(circuit)
    assert circuit.num_qubits <= target_qubits, "比特数没有降下来"
    return circuit
```

**第二步：四次非线性锥重写。** 在 101,002.54 的基础上，我们又做了四次「非线性锥重写」（nonlinear-cone rewrite）——可以理解成「把一小片电路当成一个整体，换一种等价但更省的写法」。四次改写合计把期望代价降到 **100,754.54076**。

**第三步：引入上游的相位补丁。** Ahmed Berat Özer 有一个单独署名、在 manwithacat 的 Voodoo v5 里分发的「单 Toffoli 相位补丁」。我们把它组合进来，期望代价再降 **62**，到达 **100,692.54076**。相对上一里程碑 101,002.54076，总共省 310，降幅 **0.306923%**，其中四次锥重写贡献 248、相位补丁贡献 62。这一步的关键是**署名分开算**：不能把别人的 62 算成自己的。

```python
# 伪代码：把「自己的改写」和「上游补丁」分开记账，避免夸大贡献
milestones = []

base = 101_002.54076
milestones.append(("Q62 base (ours)", base))

after_cones = 100_754.54076
milestones.append(("4x nonlinear-cone (ours)", after_cones))

after_patch = 100_692.54076
milestones.append(("Ozer phase patch (upstream)", after_patch))

for name, cost in milestones:
    print(f"{name:35s} cost={cost:.5f}")

# 分别计算两段节省，而不是只报一个总数
print("cone rewrites saved:", round(base - after_cones, 5))
print("upstream patch saved:", round(after_cones - after_patch, 5))
```

**第四步：DQI 的候选电路验证。** DQI 这条线我们**没有自己重建**，而是拿到一份用户提供的已编译 Q888 候选（哈希 `25db915d…`），它带有「norm-spill 改动」和「有意的 gi8 D=0 极点近似」。我们做的是**匹配验证**：用同一 seed（2026103003）、同一 salt（`dqi-r2-confirmation-20261030`）、同一 SDK，跑 102,400 shots，结果 0/0 错误，代价从 **401,141,196,953.7628** 降到 **400,273,653,230.8152**，降幅 **0.216268917%**。这里有一条重要纪律：**不适用它的源码补丁、不独立重建，就不能声称自己「实现」了它**——材料里明确写了这一点。

**第五步：chemistry 的系数微调。** chemistry 用的是 SomeJoint 的全 XOR 化学加载器和 Heliosli 发布的角度电路；数值输入来自 avazaku 的 spectral-trust notebook（基于 Heliosli V28 因子数据），15 个 Heliosli 加载器原样使用，只替换 avazaku 的数值系数 NPZ。这一步把代价从 **337,780,864.96436** 降到 **337,780,213.34423**，只省 **651.62013**，降幅 **0.000192912%**——小数点后第四位。为什么这么小的改动还要做？因为 chemistry 的比值只有 9.93，任何一点节省都直接进总分；但也正因为余量窄，改动必须用两个盐值各 78,080 shots 验证通过才敢留。

**关于参数调优。** 这个比赛的「超参数」是：批次的采样 shots 数、seed/salt 的选择、近似的极点位置（如 gi8 D=0）、以及是否启用某项重写。我们的取舍原则是：**先保证 0 错误，再谈代价**。shots 太少会让统计噪声盖过真实改进；近似太激进会让 error-2 余量消失。材料里 chemistry 的「gap 余量异常窄」就是典型的警告信号。

### 2.4 模型测试与指标报告

优化做完，必须用验证器「验货」。这一步的核心是**可复现的 receipt**：把 seed、salt、shots、错误数、代价全部记下来，任何人拿着同样的参数都能跑出同样的结果。

```python
# 伪代码：一次完整的本地验证记录
def run_verification(benchmark, circuit, seed, salt, shots):
    result = verifier.run(
        circuit=circuit,
        benchmark=benchmark,   # 用该基准自己的计分定义，不能跨基准比较
        seed=seed,             # 固定随机种子，保证采样可复现
        salt=salt,             # 盐值：换一个盐值等于换一组随机样本
        shots=shots,           # 采样次数，越多统计越稳、耗时越长
    )
    receipt = {
        "benchmark": benchmark,
        "cost": result.cost,
        "value_errors": result.value_errors,   # 必须为 0
        "phase_errors": result.phase_errors,   # 必须为 0
        "shots": shots,
        "seed": seed,
        "salt": salt,
        "sdk": result.sdk_version,
    }
    return receipt
```

三个基准的实测记录如下（全部来自材料）：

- **invsqrt**：full64 验证通过 **200,540,928 shots**，采样代价 **100,692.57301**，与确定性期望代价 100,692.54076 只差 0.03225（这是采样噪声，不是实现差异）。
- **DQI**：paired2（seed 2026103002，salt `dqi-r2-paired-20261030`）跑 **102,400 shots**，**0 value / 1 phase error**；paired3（seed 2026103003，salt `dqi-r2-confirmation-20261030`）跑 **102,400 shots**，**0/0 错误**，代价 398,443,798,212.809。材料明确说 paired3 是补充证据，不混进总分。
- **chemistry**：两个本地盐值各 **78,080 shots** 通过，代价 **337,780,213.344**，但 error-2 与 gap 余量异常窄。

最后是打包提交。材料里的 notebook 会把清单、receipt、源码 commit、哈希一起打成 zip：

```python
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib, json

def build_submission(dataset_dir, out_zip="submission.zip"):
    files = ["portfolio-manifest.json", "SOURCES.md", "receipts.json"]
    with ZipFile(out_zip, "w", ZIP_DEFLATED) as zf:
        for name in files:
            path = dataset_dir / name
            zf.write(path, arcname=name)          # 保持相对路径，解压后结构不乱
    # 对整个包算一个哈希，写进提交说明，方便别人核对「我拿到的就是这一版」
    digest = hashlib.sha256(open(out_zip, "rb").read()).hexdigest()
    print("submission sha256:", digest[:16])
    return digest
```

**指标报告的读法。** 我们最终看三件事：第一，**三个比值**分别是多少，谁强谁弱；第二，**几何平均总分**是多少（本地估计 23.2142，此前公开分 22.99252）；第三，**每条结论的证据强度**——是 0 错误的全量验证，还是带 1 个错误的固定盐值结果。材料把这三层分得很清楚，我们照抄这个纪律：**分数可以估计，证据不能含糊。**

---

## 三、总结

本文中我们完成了量子编译观测台（QCC）的完整优化流程。我们首先做 EDA，把 invsqrt（62 比特）、DQI（888 比特）、chemistry（430 比特）三个基准的本地代价、参考代价和比值摊成一张表，并用几何平均公式复现了 23.2142 的本地估计总分。其次我们做了数据清洗，建立了「错误数必须为 0、shots 必须达标、seed/salt/SDK 必须对齐」三条采纳规则，把带 1 个 value error 的 DQI receipt 明确排除在正式统计之外。接着我们分两条线优化：invsqrt 先减到 62 比特、再做四次非线性锥重写、最后组合 Özer 的相位补丁，把期望代价从 102,349.22 一路压到 100,692.54076；DQI 和 chemistry 则是在公开候选上做匹配验证与系数微调，分别降 0.216% 和 0.00019%。然后我们按基准分别跑了 200,540,928、102,400、78,080 shots 的验证，确保 0 错误。最后我们把清单、receipt、源码哈希打包成可复现的提交。

这个比赛给我们的最大启发是：**在精密优化里，「可复现」和「分数」一样重要**。同样是 23 分，一份带 0 错误、seed/salt 齐全的 receipt，和一份「我跑出来大概是这样」的口头结论，价值完全不同。另一个启发是**贡献要分开记账**——自己的锥重写省了 248，上游补丁省了 62，就必须写成 248 和 62，而不是含糊地说「我们省了 310」。这种诚实不会让分数变低，反而让每一次进步都站得住脚。
