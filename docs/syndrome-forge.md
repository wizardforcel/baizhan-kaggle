# syndrome-forge

## 一、比赛简介

### 1.1 比赛背景

syndrome-forge 这份材料记录的不是传统的表格预测或图像分类，而是一类量子计算基准优化任务。材料中的 notebook 直接采用了当前的 main research SOTA（主要研究最优方案）作为提交工件，并把经过本地验证的量子电路文件打包提交。对初学者来说，可以把量子电路先理解成一串对量子比特进行操作的程序：量子比特数量越少，电路通常越紧凑；验证器成本越低，说明电路执行目标计算时消耗的资源越少。

材料把任务分成三个基准：Invsqrt、DQI 和 Chemistry。Invsqrt 是 inverse square root（平方根倒数）方向的基准；DQI 使用量子比特电路和 KMB 工件；Chemistry 则使用角度电路加载器与数值系数文件。三类基准的原始成本定义不同，所以材料特别提醒我们，不能把不同基准的原始成本直接横向比较。我们需要关注每个基准相对于它自己的 Google reference cost（Google 参考成本）的比例。

这个任务的难点有三层。第一，提交内容不是一个普通的 CSV，而是多个二进制电路或数值文件，文件身份必须保持稳定。第二，电路必须通过验证器，不能只看文件能否成功压缩。第三，优化是在已有公开工作、用户提供工件和本地改写之间进行组合，最终还要明确哪些部分是本地改进、哪些部分来自公开贡献。材料也明确说明，这个 portfolio 采用了公开和用户提供的工作，并不是从零独立重新发明全部电路。

### 1.2 比赛目标

我们的目标是针对三个基准准备能通过验证的提交工件，并尽量降低各自的 verifier cost。材料给出的当前组合包括：Invsqrt 使用 Q62 电路，DQI 使用 Q888 的编译 KMB，Chemistry 使用 15 个 Heliosli loader 对应的角度文件和一个 `submission.npz` 数值文件。这里的 Q62、Q888、Q430 表示材料记录的量子比特规模，分别对应 Invsqrt、DQI 和 Chemistry 的基准规模记录。

提交文件不是由训练一个分类器产生的标签，而是由电路优化和工件管理产生的压缩包。材料要求最终压缩包恰好包含 18 个文件：1 个 `invsqrt.kmb`、1 个 `dqi.kmb`、15 个 `angle_k_*.kmb` 角度文件以及 1 个 Chemistry 的 `submission.npz`。因此，我们把“输入”理解为公开基线、用户提供的候选工件、系数数据和本地修改；把“输出”理解为通过校验、哈希一致、成员集合正确的 `submission.zip`。

```python
from pathlib import Path

# 我们先定位带有 portfolio 清单的数据集，避免把文件路径写死在某一台机器上
roots = [
    Path("/kaggle/input/qcc-local-sota-20261031"),
    Path("/kaggle/input/datasets/yoyobar/qcc-local-sota-20261031"),
]
DATASET = next(
    (path for path in roots if (path / "portfolio-manifest.json").is_file()),
    None,
)
if DATASET is None:
    raise FileNotFoundError("没有找到包含 portfolio-manifest.json 的数据集")

# 清单是我们后续核对来源、提交文件和限制条件的入口
print("数据集目录:", DATASET)
print("提交目录:", DATASET / "submission")
```

我们首先寻找 `portfolio-manifest.json`，其次确认 submission 目录存在，接着把清单作为所有后续检查的依据。这样做的目的不是把任意文件塞进压缩包，而是固定本次实验使用的 registry snapshot（注册表快照）。材料还说明，当前 notebook 使用的是 2026-10-31 的 `qcc-local-sota` 数据集快照，提交面对的是已经接受的工件，并不重新运行完整验证器，也不直接提交排行榜。

### 1.3 评价标准

材料列出了三个维度：Qubits、Local verifier cost 和 Google reference cost。Qubits 是电路使用的量子比特数量；Local verifier cost 是我们在本地验证器上测到的任务成本；Google reference cost 是对应基准的参考成本。核心比较量是 `reference / local`，也就是参考成本除以本地成本。这个比值越大，表示我们的本地成本相对参考成本越低，结果越好。由于三个基准的成本定义不同，材料明确指出原始成本不能跨行比较。

在记录的当前组合中，Invsqrt 的 Q62 本地期望成本为 100,692.57301 的采样结果，对应期望成本 100,692.54076，参考成本为 2,076,774.23372，比例为 20.624900 倍。DQI 的 Q888 本地成本为 400,273,653,230.8152，参考成本为 24,439,463,583,672.92969，比例为 61.056888 倍。Chemistry 的 Q430 本地成本为 337,780,213.344，参考成本为 3,355,599,588.31554，比例为 9.934269 倍。三个 main-registry 分数的几何平均汇总为 23.2142，但材料强调它只是本地 mixed-profile estimate，不是公开排行榜分数、预测或保证。

```python
# 我们把材料中的三条基准记录整理成表格，便于初学者理解评价方向
benchmarks = [
    {"name": "Invsqrt", "qubits": 62, "local": 100692.57301,
     "reference": 2076774.23372, "ratio": 20.624900},
    {"name": "DQI", "qubits": 888, "local": 400273653230.8152,
     "reference": 24439463583672.92969, "ratio": 61.056888},
    {"name": "Chemistry", "qubits": 430, "local": 337780213.344,
     "reference": 3355599588.31554, "ratio": 9.934269},
]
for item in benchmarks:
    # 比例越大，代表相对参考成本越有优势；原始成本只在同一基准内比较
    print(item["name"], "Q", item["qubits"], "ratio", item["ratio"])
```

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把数据探索理解成“弄清楚本次提交有哪些基准、哪些工件、哪些来源和哪些验证记录”。材料中的清单不是传统数据表，但它同样承担了 EDA 的作用：它告诉我们当前 main portfolio 有 Invsqrt、DQI、Chemistry 三行基准；告诉我们每行的量子比特数、成本和参考成本；还告诉我们提交 ZIP 要包含 18 个文件。

从时间线看，Invsqrt 的公开基础来自 busyaprime 的 63-qubit 电路，该线路又沿用了 mrmeows / team selenium 的工作；本地工作先把候选降到 Q62 和 101,002.54076 的期望成本，再通过四次 nonlinear-cone rewrite（非线性锥改写）达到 100,754.54076，最后和独立的 one-Toffoli phase patch（一次 Toffoli 相位修补）组合到 100,692.54076。DQI 的公开谱系包括 SomeJoint 的 897-qubit builder/decoder、Q889 two-halves source 和 Heliosli 的 Q889 narrowing；当前使用的是用户提供的 Q888 compiled KMB。Chemistry 则使用 SomeJoint 的 full-XOR loaders、Heliosli 的角度电路和 avazaku 的数值系数 NPZ。

```python
import json

# 我们读取清单而不是凭文件名猜测实验信息，避免把不同版本工件混在一起
manifest = json.loads((DATASET / "portfolio-manifest.json").read_text())
print("清单字段:", sorted(manifest.keys()))

# 提交目录中的文件数量是一个重要的探索结果，后面要和 18 这个预期值核对
submission_files = sorted(
    str(path.relative_to(DATASET / "submission"))
    for path in (DATASET / "submission").rglob("*")
    if path.is_file()
)
print("发现文件数:", len(submission_files))
for name in submission_files:
    print(name)
```

我们其次检查目录和清单，接着按基准记录资源规模与成本，最后确认提交文件的组成。这里没有把二进制 KMB 当成普通文本解析，因为材料没有提供 KMB 的字段定义；我们只使用 notebook 明确给出的文件名、数量、哈希和验证结果。这样既能完成必要的探索，也不会凭空解释未公开的电路内部结构。

### 2.2 数据清洗

在这个比赛里，数据清洗主要是工件完整性检查，而不是删除缺失行。材料给出的清洗原则非常明确：每个输入文件都要流式读取，计算 SHA-256，并和固定的期望哈希比较；压缩后还要检查 ZIP 成员集合、成员数量、CRC，以及解压后再次计算哈希。哈希可以理解为文件的指纹，只要文件内容有一个字节变化，指纹就可能不同。

我们没有重新编译用户提供的 DQI Q888 工件，也没有把它说成本地独立发明。材料说明这个候选包含 norm-spill change 和有意的 gi8 D=0 pole approximation，本地只是用 paired2、paired3 做了验证。Chemistry 的 15 个 Heliosli loader 保持不变，数值系数来自 avazaku 的 notebook。这样的清洗和来源管理，重点是防止提交了错误版本，而不是让文件看起来更“干净”。

```python
import hashlib

# 我们用分块读取计算指纹，避免一次性把大型二进制电路全部放进内存

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# 只要实际指纹和清单指纹不同，我们就立即停止，避免错误工件进入提交包
for name, expected_sha256 in EXPECTED.items():
    actual_sha256 = sha256_file(DATASET / "submission" / name)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"文件指纹不一致: {name}")
print("所有输入文件的 SHA-256 均匹配")
```

我们首先按 8 MiB 分块读取文件，其次计算 SHA-256，接着逐个和材料固定的 `EXPECTED` 映射比较。这个过程把“文件存在”提升为“文件内容就是我们想要的版本”。材料列出的 18 个固定文件中，15 个是 `dfthcsos/angle_k_0.kmb` 到 `angle_k_14.kmb`，另外是 `dfthcsos/submission.npz`、`dqi.kmb` 和 `invsqrt.kmb`。任何一个成员缺失、替换或内容变化，都不能继续打包。

### 2.3 模型训练与超参数调优

本比赛没有机器学习意义上的训练集、标签和神经网络模型，所以这里的“模型训练”对应电路候选的构造、改写与筛选。我们首先沿用公开基线，其次采用本地电路改写，接着组合经过单独验证的公开 phase patch。Invsqrt 的改进路径最清楚：从 Q62、101,002.54076 的候选出发，四个 nonlinear-cone rewrite 把期望成本降到 100,754.54076，再和 one-Toffoli phase correction 组合到 100,692.54076。这里的“超参数调优”可以理解为在电路改写方案、候选来源和组合方式之间做实验选择，而不是调学习率或树深度。

DQI 的处理更保守。我们没有应用源码补丁，也没有独立重建 Q888 KMB，而是接受用户提供的编译候选并做 paired2、paired3 两套本地验证。Chemistry 的处理也强调复用：15 个 loader 保持 Heliosli 版本不变，只使用 avazaku 的 numeric spectral-trust factor NPZ。材料还特别提醒，Chemistry 的 error-2 和 gap margin 非常窄，因此不能只因为一次通过就宣称整个输入域都等价。

```python
# 这里用伪代码表达我们筛选电路候选的过程；材料没有提供可独立运行的优化器
candidates = [
    {"name": "Invsqrt-Q62-cone", "expected_cost": 100754.54076},
    {"name": "Invsqrt-Q62-plus-phase-patch", "expected_cost": 100692.54076},
]

best = None
for candidate in candidates:
    # 我们先比较候选成本，再把验证通过作为不可省略的安全条件
    passed = run_local_verifier(candidate)  # 代表材料中的本地验证步骤
    if passed and (best is None or candidate["expected_cost"] < best["expected_cost"]):
        best = candidate

if best is None:
    raise RuntimeError("没有找到同时满足成本和验证条件的候选")
print("选择候选:", best["name"])
```

我们其次把“成本更低”与“验证通过”放在同等重要的位置：成本再低，若电路没有通过验证，也不能成为提交版本。接着我们比较同一基准内部的候选，而不是拿 Chemistry 的原始成本和 DQI 直接排序。最后，我们保留来源和限制：Invsqrt 的本地研究由用户与 GPT-6 Astra 协作完成；DQI 的 Q888 候选归属于用户提供工件；Chemistry 的系数候选归属于 avazaku。这样调优结果才有可复现的边界。

### 2.4 模型测试与指标报告

我们首先在本地验证器上测试电路是否正确，其次用 shots（重复采样次数）观察成本记录，接着按基准分别报告结果。Invsqrt 的 full64 验证通过了 200,540,928 shots，采样成本为 100,692.57301，确定性期望成本为 100,692.54076。DQI 的 paired2 运行使用 seed `2026103002` 和 salt `dqi-r2-paired-20261030`，102,400 shots 的错误计数为 0/1；paired3 使用 seed `2026103003` 和另一个 salt，102,400 shots 的错误计数为 0/0，补充分数为 398,443,798,212.809。Chemistry 在两个 78,080-shot salt 上通过，成本为 337,780,213.344，但 error-2 和 gap margin 仍然特别窄。

然后我们把提交 ZIP 当作最终测试对象。材料中的打包代码会按照排序后的文件名写入临时 ZIP，完成后替换正式文件；再检查成员数必须为 18、名称不能重复、名称顺序要和预期一致、CRC 不能损坏，并逐个重新计算压缩包内文件的 SHA-256。最后记录输出 ZIP 的 SHA-256 和已验证输入文件数量。注意，这一步只是打包和身份验证，不会重新运行完整 verifier，也不会向比赛提交分数。

```python
from zipfile import ZipFile

# 我们检查压缩包的结构，防止电路正确但提交格式错误
with ZipFile(archive_path, "r") as archive:
    names = archive.namelist()
    if len(names) != 18 or len(set(names)) != 18:
        raise ValueError("提交必须恰好包含 18 个不重复文件")
    if names != sorted(EXPECTED):
        raise ValueError("提交成员名称或顺序不符合预期")

    # CRC 检查压缩包内容是否损坏；返回 None 才表示全部成员正常
    corrupt = archive.testzip()
    if corrupt is not None:
        raise ValueError(f"压缩包成员损坏: {corrupt}")

    # 我们再次核对压缩包内的哈希，确认压缩过程没有改变成员身份
    for name, expected_sha256 in EXPECTED.items():
        packaged_sha256 = sha256_file_from_zip(archive, name)
        if packaged_sha256 != expected_sha256:
            raise ValueError(f"压缩包内指纹变化: {name}")

print("提交压缩包结构、CRC 和成员指纹检查通过")
```

最后我们报告材料中能确认的指标：Invsqrt 的 reference/local 为 20.624900 倍，DQI 为 61.056888 倍，Chemistry 为 9.934269 倍，三个 main-registry 分数的几何平均 rollup 为 23.2142。材料同时记录了上一里程碑的改进：Invsqrt 从 101,002.54076 降到 100,692.54076，减少 310.00000、降幅 0.306923%；DQI 从 401,141,196,953.7628 降到 400,273,653,230.8152，降幅 0.216268917%；Chemistry 从 337,780,864.96436 降到 337,780,213.34423，减少 651.62013、降幅 0.000192912%。这些是本地验证和匹配比较结果，不应被表述成新的公开排行榜分数；材料明确说明最新已完成的公开比赛条目 v8 得分为 23.17116，而本 portfolio 没有新的 leaderboard score。

## 三、总结

本文中我们完成了 syndrome-forge 材料所记录的量子电路提交流程。我们首先通过 EDA 梳理了 Invsqrt、DQI、Chemistry 三个基准、量子比特规模、成本和参考成本；其次把数据清洗落实为文件来源、SHA-256、ZIP 成员、CRC 和工件版本检查，而不是随意修改二进制电路；接着我们沿着公开基线、用户候选和本地改写的路线选择电路，Invsqrt 使用四次 nonlinear-cone rewrite 与独立 phase patch 的组合，DQI 和 Chemistry 则保留材料标注的来源与限制；然后我们用不同 seed、salt 和 shots 做本地验证，分别报告成本、错误计数和边界风险；最后我们把 18 个已验证文件打包成提交 ZIP，并再次核对压缩包身份。

这份材料给我们的最大启发是，量子电路比赛的“模型”不一定是神经网络。电路改写、候选组合、参考成本比较和严格的工件校验，同样构成一条完整的实验链。我们还必须把“结果好”与“结果可追溯”同时考虑：Invsqrt 的局部优化有明确的改进数字，DQI 的候选明确标注为用户提供，Chemistry 的通过结果也保留了 error-2 和 gap margin 很窄的警告。只有把分数、验证、来源和限制放在一起，writeup 才不会把本地证据夸大成未经证明的全域结论。
