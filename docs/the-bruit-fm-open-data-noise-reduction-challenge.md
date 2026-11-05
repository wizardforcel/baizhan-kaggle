# BRUIT-FM 开放数据噪声抑制挑战（BRUIT-FM Open Data Noise Reduction Challenge）

## 一、比赛简介

### 1.1 比赛背景

地震仪会持续记录地面的微小振动。为了看清地面到底怎么动，台站通常同时记录三个方向：一个是垂直方向，记作 `LHZ`；另外两个是水平方向，记作 `LH1` 和 `LH2`。除了这三个方向，台站还可能记录气压等辅助信息，例如本次数据里的 `LDG` 通道。

真实的地震记录并不“干净”。垂直分量 `LHZ` 上除了真正的地面运动，还会混进很多别的东西：台站轻微倾斜会让水平分量出现变化，这种变化又会“串”到垂直分量上；气压变化、海洋波浪造成的低频扰动也会污染垂直记录。比赛的名字叫“噪声抑制”，要做的就是把这些和辅助通道有关系的噪声从 `LHZ` 里减掉，同时尽量不伤到真正的地震信号。

本次比赛提供的是一份合成的含噪地震数据，文件名是 `SYNTH_noisy.mseed`。MiniSEED 是地震领域常用的一种二进制数据格式，里面的采样值用 STEIM1 方式压缩。STEIM1 不直接存每一个采样值，而是存相邻采样点之间的“差分”，因为地震信号变化比较平缓，差分往往很小，压缩率就高。这个格式对做机器学习的人来说不太友好：它不是 CSV，不能直接 `read_csv`；比赛 notebook 又要求在没有外网的 Kaggle 环境里运行，不能随便调用 ObsPy 之类的专业库。因此，参赛者需要自己把二进制数据解出来。

比赛把任务分成两个目标。第一个叫 `Z_compliance`，主要针对与台站倾斜、海洋低频扰动有关的“顺应噪声”（compliance noise）。从代码里的通道组合看，这个目标认为 `LH1` 和 `LH2` 能够线性解释 `LHZ` 中那部分相干的倾斜污染。第二个叫 `Z_earthquake`，针对地震信号场景，除了水平通道，还把气压相关的 `LDG` 一起拿来解释 `LHZ` 中的相干污染。两个目标共用同一条原始 `LHZ`，但使用的辅助通道和滤波尺度不同。

这个比赛的难点主要有三个。第一，数据是二进制格式，读取和解析本身就是一道门槛；第二，`SYNTH` 的标准答案不公开，参赛者无法在本地直接算出排行榜分数；第三，噪声不是随机的白噪声，而是和辅助通道“同源”的相干噪声，所以简单的低通、高通滤波往往不够，需要利用多通道之间的关系来做估计。

### 1.2 比赛目标

比赛的输入是一份含噪的 MiniSEED 文件，里面包含 `LH1`、`LH2`、`LHZ`、`LDG` 四个通道。我们要输出一个名为 `submission.csv` 的表格，列名固定为：

- `id`：样本编号；
- `Z_compliance`：去噪后的垂直分量，对应顺应噪声目标；
- `Z_earthquake`：去噪后的垂直分量，对应地震信号目标。

提交文件必须正好有 6000 行。官方提交构造方式是把整段数据切成 10 个固定窗口，每个窗口取 600 个采样点，10 × 600 正好是 6000。窗口起点依次是 0、8640、17280、25920、34560、43200、51840、60480、69120、77760。每一行的 `id` 由“窗口编号_窗口内采样点编号”组成，例如第 0 个窗口的第 5 个点写成 `0_5`。两个预测列都必须是有限的浮点数，不能出现空值或无穷大。

换句话说，比赛要的不是“把整段信号降噪后随便交上去”，而是严格按照官方窗口格式，交出两套清洗后的 `LHZ` 序列。

### 1.3 评价标准

`SYNTH` 的答案不提供给参赛者，所以真实排行榜分数只有提交之后才能看到。比赛 notebook 里附带了一个“指标辅助函数”，它模仿官方公布的 dB 风格评分公式，方便参赛者用本地训练数据做参考。

评分的基本思想是比较两个误差：一个是“含噪原始信号”与干净答案之间的误差，记作 `err_noisy`；另一个是“我们提交的清洗信号”与干净答案之间的误差，记作 `err_sub`。如果清洗后的误差更小，就说明去噪有效；如果误差反而变大，就说明我们过度减噪、把有用信号也削掉了。

公式里有一个噪声下限 `NOISE_FLOOR_COUNTS = 64.0`。当 `err_noisy` 本身已经小于等于 64 时，说明原始数据已经足够干净，此时只要 `err_sub` 不大于 `err_noisy` 就给满分 100，否则给一个极大的负数作为惩罚。一般情况下，先算两个 dB 值：

- `best_db = 20 * log10(err_noisy / 64.0)`，表示“把误差一路降到噪声下限”能带来多少 dB 的提升；
- `achieved_db = 20 * log10(err_noisy / err_sub)`，表示我们实际取得了多少 dB 的提升。

最终分数是 `min(100, 100 * achieved_db / best_db)`。直观地说，分数越高越好，满分 100；如果 `err_sub` 比 `err_noisy` 还大，`achieved_db` 会是负数，分数也会变成负数，代表这次提交还不如不处理。

> 小提示：dB 是“分贝”，在这里用来表示两个误差的比值。误差缩小 10 倍，大约相当于提升 20 dB。用对数是为了让“误差缩小一倍”和“误差缩小一百倍”在分数上有比较平滑的区别。

---

## 二、解题过程

整场比赛我们采用的是一条信号处理路线：先把 MiniSEED 解成四个通道的采样序列，再用多通道频域维纳滤波，把 `LHZ` 中能被辅助通道线性解释的成分减掉；针对两个目标分别选择不同的辅助通道和不同的分析窗口，最后按官方窗口格式拼出 6000 行提交文件。下面按 EDA、数据清洗、模型训练与调参、测试与指标报告四步展开。

### 2.1 数据探索（EDA）

我们首先做的是找到数据文件。Kaggle 通常把比赛数据挂在 `/kaggle/input` 下面，但也可能把官方 ZIP 作为 notebook 的输入附件。为了不依赖固定的绝对路径，我们写了一个小函数，先在几个常见根目录里递归找文件名，找不到再去 ZIP 成员里找。这样无论数据是解压好的还是压缩包，代码都能跑。

```python
from pathlib import Path
import zipfile

# 官方要求的 10 个窗口起点，每个窗口 600 个采样点，合计 6000 行
WINDOW_START_SAMPLES = [0, 8640, 17280, 25920, 34560,
                        43200, 51840, 60480, 69120, 77760]
WINDOW_SAMPLES = 600
EXPECTED_ROWS = 6000

# Kaggle 数据可能在这些位置，按顺序查找
SEARCH_ROOTS = [Path("/kaggle/input"), Path("/mnt/data"), Path(".")]

def locate_file_or_zip_member(filename):
    """先在目录里找文件；找不到再去 zip 里找同名成员。"""
    for root in SEARCH_ROOTS:
        if root.exists():
            for p in root.rglob(filename):
                if p.is_file():
                    return ("path", p)

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for zp in root.rglob("*.zip"):
            with zipfile.ZipFile(zp) as zf:
                matches = [n for n in zf.namelist()
                           if n == filename or n.endswith("/" + filename)]
                if matches:
                    # 优先选路径里带 CHALLENGE/SYNTH 的那份，避免拿到别的副本
                    matches.sort(key=lambda n: ("CHALLENGE/SYNTH" not in n, len(n)))
                    return ("zip", (zp, matches[0]))

    raise FileNotFoundError(filename)

synth_source = locate_file_or_zip_member("SYNTH_noisy.mseed")
```

拿到文件之后，我们关心的第一件事是“里面到底有哪些通道、每个通道有多少点”。MiniSEED 的每个记录头里都写着通道名和采样点数，所以我们按通道把所有记录拼起来，打印每个通道的长度和标准差。标准差可以粗略反映信号幅度：如果某个通道几乎是一条直线，标准差会非常小；如果波动很大，标准差就大。

```python
# 读取原始字节；解码函数的具体实现在 2.2 节
synth_bytes = read_source_bytes(synth_source)
channels = decode_mseed_v2_steim1(
    synth_bytes,
    wanted_channels={"LH1", "LH2", "LHZ", "LDG"},
)

# 逐个通道检查：四个通道都必须存在，且长度要覆盖最后一个窗口
for ch in ["LH1", "LH2", "LHZ", "LDG"]:
    x = channels[ch]
    print(f"{ch}: {len(x):,} 个采样点，标准差={np.std(x):,.2f}")

# 最后一个窗口的终点是 77760 + 600 = 78360，长度不够就无法构造提交
assert len(channels["LHZ"]) >= 78360
```

通过这一步，我们确认了四件事。第一，四个目标通道都在，没有缺通道；第二，每条通道的采样点数量足够覆盖 10 个窗口，最远要用到第 78360 个采样点；第三，各个通道的幅度量级不一样，`LHZ` 是我们要处理的主信号，`LH1`、`LH2`、`LDG` 是辅助信号；第四，数据里没有可以直接读取的标签列，`SYNTH` 的干净答案确实不可见。这也意味着后面的“调参”不能靠普通的训练集/验证集划分，而要靠本地另外准备的 MASS 训练数据做检查。

我们还确认了官方窗口的排列方式：10 个窗口不是连续的整段，而是从整条序列里挑了 10 个固定起点，每个起点只取 600 个点。提交时 `id` 必须严格按这个顺序生成，否则即使信号算得再准，格式对不上也会被判错。

### 2.2 数据清洗

这一节说的“清洗”不是删表格里的缺失值，而是把二进制数据正确还原成可以计算的浮点序列。MiniSEED 的一个记录由“记录头 + 数据区”组成，数据区里的 STEIM1 压缩帧每 64 字节一组：第一个 32 位字是控制字，后面 15 个是数据字。控制字里每 2 位描述一个数据字的编码方式：编码 1 表示这个字里塞了 4 个 8 位差分，编码 2 表示塞了 2 个 16 位差分，编码 3 表示整个 32 位字就是一个差分。

解码时最容易踩的坑是“第一个差分”。STEIM1 记录在开头会显式存一个起点 `X0`，第一个差分描述的是“上一条记录的最后一个采样值”到 `X0` 的差，它并不属于本记录内部的采样序列。因此从 `X0` 往后重建采样点时，必须把第一个差分丢掉，否则整条序列会整体偏移一个点。

```python
import struct
import numpy as np

def _signed(x, bits):
    """把无符号整数按补码解释成有符号整数。"""
    return x - (1 << bits) if x >= (1 << (bits - 1)) else x

def decode_steim1_record(record, npts, data_offset, word_order=1):
    """把一个 STEIM1 记录还原成 npts 个整数采样值。"""
    endian = ">" if word_order == 1 else "<"
    x0 = xn = None
    diffs = []

    # 每 64 字节是一帧：1 个控制字 + 15 个数据字
    for off in range(data_offset, len(record), 64):
        frame = record[off:off + 64]
        if len(frame) < 64:
            break
        words = struct.unpack(endian + "16I", frame)
        control = words[0]

        for wi in range(1, 16):
            code = (control >> (30 - 2 * wi)) & 0x03
            word = words[wi]

            # 第一帧的第 1、2 个数据字分别是起点 X0 和终点 XN
            if off == data_offset and wi == 1:
                x0 = _signed(word, 32)
                continue
            if off == data_offset and wi == 2:
                xn = _signed(word, 32)
                continue

            if code == 1:          # 一个字里 4 个 8 位差分
                diffs += [_signed((word >> 24) & 0xFF, 8),
                          _signed((word >> 16) & 0xFF, 8),
                          _signed((word >> 8) & 0xFF, 8),
                          _signed(word & 0xFF, 8)]
            elif code == 2:        # 一个字里 2 个 16 位差分
                diffs += [_signed((word >> 16) & 0xFFFF, 16),
                          _signed(word & 0xFFFF, 16)]
            elif code == 3:        # 整个字就是 1 个 32 位差分
                diffs.append(_signed(word, 32))

    # 第一个差分连接的是上一条记录的末尾和本记录的 X0，必须丢弃
    diffs = diffs[1:]
    out = np.empty(npts, dtype=np.float64)
    out[0] = x0
    out[1:] = x0 + np.cumsum(np.asarray(diffs[:npts - 1], dtype=np.int64))

    # XN 是记录头给出的最后一个采样值，用它检查解码有没有错位
    if xn is not None and int(out[-1]) != int(xn):
        raise ValueError("STEIM1 XN 校验失败")
    return out
```

单个记录解出来只是第一步，整份文件里有很多记录，而且通道是混在一起的。我们继续遍历文件：每个 64 字节记录头里有通道名、采样点数、数据区偏移和 Blockette 偏移；顺着 Blockette 链找到类型 1000 的块，就能读到编码方式、字节序和记录长度。这里只接受编码 10，也就是 STEIM1；遇到别的编码就明确报错，避免在不知情的情况下算出错误结果。

```python
import collections

def decode_mseed_v2_steim1(data, wanted_channels=None):
    """按记录遍历 MiniSEED v2，把需要的通道分别拼接起来。"""
    by_channel = collections.defaultdict(list)
    pos = 0

    while pos + 64 <= len(data):
        header = data[pos:pos + 64]
        channel = header[15:18].decode("ascii", errors="replace").strip()
        npts = struct.unpack(">H", header[30:32])[0]
        data_offset = struct.unpack(">H", header[44:46])[0]
        blockette_offset = struct.unpack(">H", header[46:48])[0]

        # 顺着 Blockette 链找类型 1000，它描述数据编码和记录长度
        bo = blockette_offset
        encoding = None
        word_order = 1
        record_length_exp = None
        seen = set()
        while bo and bo not in seen:
            seen.add(bo)
            bh = data[pos + bo:pos + bo + 8]
            if len(bh) < 8:
                break
            btype, next_bo = struct.unpack(">HH", bh[:4])
            if btype == 1000:
                encoding = bh[4]
                word_order = bh[5]
                record_length_exp = bh[6]
                break
            bo = next_bo

        record_length = 2 ** record_length_exp
        record = data[pos:pos + record_length]
        if len(record) < record_length:
            break

        # 本 notebook 只处理 STEIM1（编码 10）
        if encoding != 10:
            raise ValueError(f"期望 STEIM1，实际编码是 {encoding}")

        if wanted_channels is None or channel in wanted_channels:
            arr = decode_steim1_record(record, npts, data_offset, word_order)
            by_channel[channel].append(arr)

        pos += record_length

    # 同一通道可能被拆成很多条记录，这里按顺序拼回一条长序列
    return {ch: np.concatenate(parts) for ch, parts in by_channel.items()}
```

解码完成后，我们只保留 `LH1`、`LH2`、`LHZ`、`LDG` 四个通道。没有做重采样，也没有做幅度归一化，因为两个目标需要保持原始的相对幅度关系；如果擅自把每个通道标准化，辅助通道和目标通道之间的比例就会被破坏，后面的线性滤波系数也会失去意义。数据本身没有缺失值，也没有重复记录需要去重，真正需要“清洗”的主要就是解码、丢弃错误的首差分、校验 `XN`、以及按通道拼接这几件事。

### 2.3 模型训练与超参数调优

这里的“模型”不是神经网络，而是一个按频率点求解的自适应线性滤波器。我们先把 `LHZ` 和辅助通道都做短时傅里叶变换（STFT）：把长信号切成很多短段，每段加汉宁窗，再变到频域。这样得到的谱是“频率 × 时间”的二维复数矩阵，可以看清每个时刻、每个频率上各通道的强弱和相位。

然后对每一个频率点单独解一个小型最小二乘问题。设辅助通道在这一频率、这一时刻的谱组成向量 `x`，目标通道的谱是 `z`。我们估计一个复数系数向量 `h`，让 `h` 的共轭转置乘 `x` 尽量接近 `z`。用矩阵写法就是先算辅助通道的协方差 `Rxx` 和辅助通道与目标通道的互相关 `rxz`，再解 `(Rxx + ridge * I) h = rxz`。解出来的 `h` 就代表“辅助通道怎样线性组合，才能最好地预测目标通道中与它们相干的那部分”。最后从 `Z` 里减掉这个预测噪声，再逆变换回时域。

```python
from scipy import signal

def multichannel_wiener_clean(z, predictors, nperseg, ridge=1e-6, fs=1.0):
    """频域多通道维纳滤波：减掉 LHZ 中与辅助通道相干的成分。"""
    noverlap = nperseg // 2

    # 目标通道的短时傅里叶变换
    freqs, _, Z = signal.stft(
        z, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        boundary="zeros", padded=True,
    )

    # 每个辅助通道也做同样的变换，保证频率点和时间点一一对应
    X_list = []
    for x in predictors:
        _, _, X = signal.stft(
            x, fs=fs, window="hann",
            nperseg=nperseg, noverlap=noverlap,
            boundary="zeros", padded=True,
        )
        X_list.append(X)
    X = np.stack(X_list, axis=0)

    predicted_noise = np.zeros_like(Z)
    p = len(predictors)
    eye = np.eye(p, dtype=np.complex128)

    # 每个频率点单独求一组线性系数
    for fi in range(len(freqs)):
        xf = X[:, fi, :]       # 该频率上所有辅助通道的谱
        zf = Z[fi, :]          # 该频率上目标通道的谱

        Rxx = (xf @ xf.conj().T) / xf.shape[1]
        scale = float(np.trace(Rxx).real / max(p, 1))
        # 加一个很小的岭项，避免协方差矩阵奇异
        A = Rxx + ridge * max(scale, 1e-12) * eye
        rxz = (xf @ zf.conj()) / xf.shape[1]

        h = np.linalg.solve(A, rxz)
        predicted_noise[fi, :] = np.conj(h) @ xf

    # 从目标谱里减掉预测噪声，再变回时域
    cleaned_spec = Z - predicted_noise
    _, cleaned = signal.istft(
        cleaned_spec, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        input_onesided=True, boundary=True,
    )
    return cleaned[:len(z)]
```

我们的两个目标使用不同的辅助通道组合。对 `Z_compliance`，我们只用 `LH1` 和 `LH2` 作为预测变量，因为这一目标主要想减掉水平倾斜能够解释的那部分垂直污染；如果把气压通道也放进去，可能会把一些不该减的低频成分一起减掉。对 `Z_earthquake`，我们把 `LH1`、`LH2`、`LDG` 三个通道都放进去，让模型同时利用水平和气压信息来估计相干污染，尽量保留地震信号本身。

超参数主要有两个：分析窗口长度 `nperseg` 和岭正则系数 `ridge`。`nperseg` 决定 STFT 每一段有多长：窗口短，时间分辨率高，能抓住地震信号的快速变化，但频率分辨率低；窗口长，频率分辨率高，适合处理低频的顺应噪声，但时间上会变模糊。因为两个目标关注的尺度不同，我们没有用同一个窗口，而是做了多分辨率组合。

```python
z = channels["LHZ"]
horiz = [channels["LH1"], channels["LH2"]]
all_aux = [channels["LH1"], channels["LH2"], channels["LDG"]]

# 顺应噪声目标：长窗口看低频，短窗口补细节，两者各占一半
compl_8192 = multichannel_wiener_clean(z, horiz, nperseg=8192)
compl_1024 = multichannel_wiener_clean(z, horiz, nperseg=1024)
z_compliance_full = 0.50 * compl_8192 + 0.50 * compl_1024

# 地震目标：三个尺度组合，中间尺度 512 权重最大
eq_256 = multichannel_wiener_clean(z, all_aux, nperseg=256)
eq_512 = multichannel_wiener_clean(z, all_aux, nperseg=512)
eq_1024 = multichannel_wiener_clean(z, all_aux, nperseg=1024)
z_earthquake_full = 0.20 * eq_256 + 0.60 * eq_512 + 0.20 * eq_1024
```

这些权重不是拍脑袋定的。材料里明确说，集成方案是从本地 MASS 训练检查中选出来的；`ridge` 也特意取得非常小，只有 `1e-6`，因为更强的收缩会降低本地 MASS 验证表现。我们理解这背后的原因：辅助通道和垂直通道之间的相干关系本来就很强，正则太大会把真实的传递系数压小，噪声减不干净；正则太小又可能解出数值上不稳定的系数，所以取一个“刚好能防止矩阵奇异”的小值比较合适。

另外，两个目标的窗口选择也体现了取舍。顺应噪声主要是低频、长周期的，所以用 8192 点长窗口能看得更准；但长窗口会抹平短时变化，因此再加一个 1024 点窗口做补充。地震信号含有较短时间的瞬态，256 点窗口能保留这些细节；512 点窗口在细节和稳定性之间比较平衡，所以给了 0.6 的权重。我们最后把不同尺度的输出按权重相加，而不是只选一个窗口，是为了减少单一尺度带来的偏差。

### 2.4 模型测试与指标报告

因为 `SYNTH` 的答案不公开，我们没有办法在本地算出真正的排行榜分数。材料里也没有记录提交后的公开分数，所以这里不编造一个具体成绩。我们能做的是两件事：一是用本地 MASS 训练数据检查去噪前后的变化，二是用辅助函数按官方公式估算“如果知道答案，分数大概怎么算”。

在生成提交文件之前，我们先打印原始 `LHZ`、顺应目标输出、地震目标输出的标准差，确认三个序列都不是全零，也没有出现数值爆炸。然后按官方窗口格式逐行构造表格。这里我们特别注意三点：列顺序必须是 `id`、`Z_compliance`、`Z_earthquake`；行数必须正好 6000；`id` 不能重复。代码里用 `assert` 把这些条件写死，避免因为格式问题白白丢分。

```python
import pandas as pd

rows = []
for window_idx, start in enumerate(WINDOW_START_SAMPLES):
    end = start + WINDOW_SAMPLES
    if end > len(z):
        raise ValueError(f"窗口 {window_idx} 超出数据长度")

    # 每个窗口贡献 600 行，id 由窗口编号和窗口内下标组成
    for sample_idx in range(start, end):
        rows.append({
            "id": f"{window_idx}_{sample_idx}",
            "Z_compliance": float(z_compliance_full[sample_idx]),
            "Z_earthquake": float(z_earthquake_full[sample_idx]),
        })

submission = pd.DataFrame(
    rows, columns=["id", "Z_compliance", "Z_earthquake"]
)

# 提交前做格式自检，任何一项不满足都直接报错
assert list(submission.columns) == ["id", "Z_compliance", "Z_earthquake"]
assert len(submission) == EXPECTED_ROWS
assert submission["id"].is_unique
assert np.isfinite(
    submission[["Z_compliance", "Z_earthquake"]].to_numpy()
).all()

# 同时写 CSV 和 ZIP，方便 Notebook 提交或文件上传两种方式
submission.to_csv("submission.csv", index=False)
```

指标辅助函数把官方 dB 公式复现了一遍。它先计算含噪误差和提交误差，再按 1.3 节说的比例换成分数。函数里还准备了去线性趋势和 RMS 的工具：地震信号本身可能有缓慢漂移，如果直接算差值，漂移会主导误差，所以先去掉线性趋势再求均方根更合理。

```python
NOISE_FLOOR_COUNTS = 64.0

def remove_linear_trend(diff, sample_index):
    """去掉差值里的线性漂移，避免缓慢趋势主导误差。"""
    slope, offset = np.polyfit(sample_index, diff, 1)
    return diff - (offset + slope * sample_index)

def rms(x):
    """均方根误差：先平方、再平均、最后开方。"""
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))

def db_score(err_sub, err_noisy, floor=NOISE_FLOOR_COUNTS):
    """按官方 dB 风格公式估算分数，满分 100。"""
    if err_noisy <= floor:
        # 原始数据已经足够干净：不比它差就给满分，否则重罚
        return 100.0 if err_sub <= err_noisy else -1e6

    best_db = 20.0 * np.log10(err_noisy / floor)
    achieved_db = (
        20.0 * np.log10(err_noisy / err_sub)
        if err_sub > 0 else float("inf")
    )
    # 超过 100 的部分截断，保证分数不超过满分
    return min(100.0, 100.0 * achieved_db / best_db)
```

从本地检查的角度看，我们最关心的是“减噪有没有减过头”。如果 `Z_earthquake` 的标准差比原始 `LHZ` 小太多，说明地震信号可能被一起削掉了；如果几乎不变，说明滤波器太保守，噪声没减掉。多分辨率集成的好处就在这里：不同窗口的输出互相补位，单一窗口的过度或不足会被平均掉一部分。最终提交文件满足 6000 行、列名正确、`id` 唯一、数值有限这四个条件，可以走 Notebook 提交，也可以把 ZIP 里的 `submission.csv` 从文件上传入口交上去。

---

## 三、总结

本文中我们完成了 BRUIT-FM 开放数据噪声抑制挑战的完整流程。我们首先定位比赛数据，确认 `SYNTH_noisy.mseed` 可能藏在目录或 ZIP 里，并记住了官方 10 个窗口、每个 600 点、合计 6000 行的提交格式。其次我们自己写了 MiniSEED v2 / STEIM1 解码器，按记录头找到 Blockette 1000，按控制字拆出 8 位、16 位、32 位差分，特别处理了“第一个差分不属于本记录”这个容易错位的细节，并用 `XN` 做校验。接着我们把四个通道拼成完整序列，只保留 `LH1`、`LH2`、`LHZ`、`LDG`，不做重采样和归一化。然后我们实现了频域多通道维纳滤波，对每个频率点解一个小型最小二乘，用 `LH1 + LH2` 估计顺应噪声目标，用 `LH1 + LH2 + LDG` 估计地震目标，并通过 256、512、1024 等多个分析尺度做加权集成。最后我们按官方窗口拼出 6000 行提交文件，用断言检查列名、行数、`id` 唯一性和数值有限性，并复现了 dB 风格评分公式作为本地参考。

这个比赛给我们的最大启发是：**面对专业领域的数据，先把数据格式弄懂，比急着上模型更重要。** 如果 MiniSEED 解错一个差分，后面的滤波再精致也没有意义；反过来，只要把通道关系和物理含义理清，一个并不复杂的线性滤波就能做出合理的去噪基线。另一个启发是“评估要诚实”：标准答案不公开时，不能编造排行榜分数，而应该用本地验证数据和公式辅助函数来判断方案是否站得住脚。
