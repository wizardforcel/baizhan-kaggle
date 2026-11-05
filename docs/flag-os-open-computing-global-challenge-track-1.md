# FlagOS 开放计算全球挑战赛 Track 1（SGLang 推理内核优化）

## 一、比赛简介

### 1.1 比赛背景

大语言模型（LLM）在线上服务时，每说出一个字都要把整个模型前向跑一遍。为了同时服务很多用户，业界常用 **SGLang** 这类推理框架：它把一次请求拆成「预填充（prefill）」和「逐字生成（decode）」两个阶段，前者一次性读完用户的问题，后者一个 token 接一个 token 地往外吐答案。

很多人以为推理慢是因为矩阵乘法太多，其实在真实的服务里，真正拖后腿的常常是夹在注意力层和多层感知机（MLP）之间的一批「小算子」：残差相加、RMSNorm 归一化、SwiGLU 激活、旋转位置编码（RoPE）等等。这些算子有个共同特点——**算术量很小，但要反复读写显存**。显存（HBM）的带宽是有限的，数据搬来搬去的时间远远大于真正做加减乘除的时间，这类算子就被称为「访存受限（memory-bandwidth-bound）」算子。

FlagOS 是一个面向多种 AI 芯片的统一计算平台，本赛道要求参赛者用 **Triton** 把这些小算子合并成「融合内核（fused kernel）」。Triton 是一种用 Python 写 GPU 内核的语言，写起来比手写 CUDA 简单，又能直接控制每个线程块读哪些数据、算哪些数据。

### 1.2 比赛目标

比赛给出若干个算子的接口签名和计算语义，参赛者要交出接口相同、结果正确的 Triton 实现。核心是三个算子：

- **融合 Add + RMSNorm**：输入 `hidden_states`（形状 `[token 数, 隐藏维度]`）、`residual`（同形状）、`weight`（形状 `[隐藏维度]`）和 `eps`，输出归一化结果以及残差相加后的中间结果 `h`；
- **融合 SiLU-And-Mul（SwiGLU 门）**：输入 `gate` 和 `up` 两块张量，输出 `silu(gate) * up`；
- **融合 RoPE**：把旋转位置编码的 sin/cos 缓存同时作用到 Query 和 Key 的头张量上。

除了算对，还要求**可移植**：同一份内核要能在不同厂商的芯片上运行；在没有 GPU 的 CPU 校验机上，则自动退回 PyTorch/NumPy 的向量化实现，保证「一份代码，到处能跑」。

### 1.3 评价标准

比赛看两件事，而且是**先及格、再比快**：

1. **正确性**：内核输出必须和 FP32 参考实现一致，误差要求 `atol <= 1e-3`、`rtol <= 1e-3`。这一条不通过，性能再好也白搭。
2. **性能**：在满足正确性的前提下，比较内核在不同芯片上的运行时间。对访存受限的算子，最有说服力的指标是**有效带宽**（单位时间搬运了多少 GB 数据）以及相对标准 PyTorch 实现的加速比。

换句话说，这是一个「既要算对、又要少搬数据」的比赛。

---

## 二、解题过程

我们首先把每个算子在标准实现里的访存次数数清楚，其次设计融合方案把多趟读写压成一趟，接着写出 Triton 内核并做自动调参，最后用参考实现对拍、测量有效带宽。下面按四步展开。

### 2.1 数据探索（EDA）

这里的「数据」不是一张表格，而是**算子的数据流**。我们先看最典型的 `Add + RMSNorm`，把未融合的 PyTorch 写法拆开，数一数显存往返了几次：

1. 读 `hidden_states`、读 `residual`，相加之后把结果写回显存（第 1 次读、第 1 次写）；
2. 为了算均方根，再把刚写下的和读回来，平方后求和（第 2 次读）；
3. 最后再读一遍，乘以 `rsqrt(var + eps)` 和权重，写出结果（第 2 次写）。

合计 2 读 2 写，一共四趟全局显存访问。而理想的融合实现只需要「读输入 → 写输出」两趟，理论上能把显存流量砍掉一半。材料里说的 `cutting global memory traffic by 50%`，指的就是这件事。

为了有个对照基线，我们用 NumPy 写了一份参考实现，顺便测了一下它的耗时和有效带宽：

```python
import time
import numpy as np

def ref_fused_add_rmsnorm(x, residual, weight, eps=1e-6):
    """参考实现：严格按 RMSNorm 的数学定义逐条计算"""
    # 残差相加。标准 PyTorch 会把这一步的结果写到显存，再读回来
    h = x.astype(np.float32) + residual.astype(np.float32)
    # 均方根：先平方，再沿最后一维（隐藏维度）求平均
    var = np.mean(h * h, axis=-1, keepdims=True)
    # rsqrt 就是 1/sqrt；最后按元素乘上可学习的权重
    normed = h * (1.0 / np.sqrt(var + eps)) * weight.astype(np.float32)
    return normed.astype(x.dtype), h.astype(x.dtype)

rng = np.random.default_rng(42)          # 固定种子：每次测的都是同一批数
for n_tokens, hidden_dim in [(512, 4096), (2048, 4096)]:
    x = rng.standard_normal((n_tokens, hidden_dim), dtype=np.float32)
    res = rng.standard_normal((n_tokens, hidden_dim), dtype=np.float32)
    w = rng.standard_normal((hidden_dim,), dtype=np.float32)

    t0 = time.perf_counter()
    out, h_out = ref_fused_add_rmsnorm(x, res, w)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # 有效带宽：按「4 倍输入字节数」估算一次完整往返搬运的数据量
    gbps = (4 * x.nbytes) / (dt_ms / 1000.0) / 1e9
    print(f"[{n_tokens}x{hidden_dim}] time={dt_ms:.2f} ms, BW={gbps:.2f} GB/s")
```

我们特意选了两组形状：`512×4096` 对应「单个请求的预填充」，`2048×4096` 对应「批量并发」。规模不同，内核块大小的最优取值也不同，这为后面的调参提供了依据。

接着我们用同样的方式看 SwiGLU。它的公式是 `silu(gate) * up`，其中 `silu(x) = x / (1 + exp(-x))`。如果拆成 sigmoid、除法、乘法三个算子，每一步的中间结果都要落一次显存；融合成一个内核后，读两块、写一块就够了。我们也先写了 NumPy 参考版本：

```python
def ref_swiglu(gate, up):
    """SwiGLU 的参考实现，后面用来和 Triton 内核逐元素对拍"""
    g = gate.astype(np.float32)     # 统一升到 FP32，避免低精度误差累积
    u = up.astype(np.float32)
    silu = g / (1.0 + np.exp(-g))   # SiLU：x * sigmoid(x) 的等价写法
    return (silu * u).astype(gate.dtype)   # 按原数据类型返回
```

最后我们看了 RoPE。朴素写法会先把 Q、K 的前后半段 `cat` 成一个临时张量再旋转，这个临时张量本身就是一次额外的显存分配和拷贝。分析下来，三个算子的问题本质相同：**中间结果不必要地落盘**。这就是我们整个方案的出发点。

### 2.2 数据清洗

内核开发里的「清洗」，指的是把输入数据的各种边界情况处理干净，否则数值会对不上。我们做了三件事：

1. **累加统一用 FP32**。输入可能是 float16 或 bfloat16，如果直接用低精度做平方和，误差会一点点累积，最后超过 `rtol=1e-3`。所以我们在内核里先把数据 `to(tl.float32)`，算完再按原类型写回。
2. **越界元素用 mask 屏蔽**。Triton 的块大小必须是编译期常量（比如 4096），但真实维度不一定刚好整除。对超出的部分要设 `mask`，并用 `other` 指定填充值：相加时填 `0.0`，读权重时填 `1.0`。如果权重也填 0，那些位置会被错误地乘成 0，结果就错了。
3. **`eps` 不能省**。当某一行的值恰好全是 0 时，方差为 0，`rsqrt(0)` 会得到无穷大。加上一个很小的 `eps`，数值才稳定。

```python
import triton
import triton.language as tl

@triton.jit
def _fused_add_rmsnorm_kernel(
    X_ptr, Res_ptr, W_ptr, Out_ptr, H_ptr,
    stride_x, N, eps,
    BLOCK_N: tl.constexpr,          # 块宽必须是编译期常量
):
    row = tl.program_id(0)          # 一个 program 负责一行
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N                 # 超出真实宽度的列全部屏蔽掉

    # other=0.0：越界位置当 0 参与相加，不会污染这一行的和
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Res_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    h = x + r

    # 中间结果必须写出去：调用方还要拿它做残差连接
    tl.store(H_ptr + row * stride_x + cols, h, mask=mask)

    # 方差在 FP32 下累加，避免低精度输入带来的误差
    var = tl.sum(h * h, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)

    # other=1.0：越界位置的权重补 1，乘上去相当于不改变数值
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    out = h * rstd * w
    tl.store(Out_ptr + row * stride_x + cols, out, mask=mask)
```

这段内核的关键在于：`h` 只在寄存器里算一次，方差也用同一份 `h` 算，**没有再回到显存里读第二遍**。四趟访存因此压成了两趟。另外，`stride_x` 是显式传进来的，所以即使输入不是连续存储（比如来自转置过的张量），内核也能正确寻址——这是「可移植」的一部分。

### 2.3 模型训练与超参数调优

这里的「训练」不是梯度下降，而是**给内核挑块大小和线程数**。Triton 的 `tl.autotune` 可以在一组候选配置里自动试跑，挑出最快的那个。

第一条支线是 SwiGLU。我们把它写成一个二维块内核，用 `BLOCK_M × BLOCK_N` 同时切分行和列：

```python
@triton.autotune(
    configs=[
        # BLOCK_M/BLOCK_N 是二维分块大小，num_warps 是每个块的线程束数
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=8),
    ],
    key=['M', 'N'],                  # 形状变了就重新挑一次配置
)
@triton.jit
def swiglu_fused_kernel(Gate_ptr, Up_ptr, Out_ptr, M, N,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)

    g = tl.load(Gate_ptr + rows[:, None] * N + cols[None, :], mask=mask).to(tl.float32)
    u = tl.load(Up_ptr   + rows[:, None] * N + cols[None, :], mask=mask).to(tl.float32)

    # SiLU 写成 x / (1 + exp(-x))，少一次函数调用，也少一个中间张量
    silu = g / (1.0 + tl.exp(-g))
    tl.store(Out_ptr + rows[:, None] * N + cols[None, :], silu * u, mask=mask)
```

调参时我们发现一个规律：`BLOCK_N` 不宜超过隐藏维度太多，否则会有一半线程在空转处理被 mask 掉的列；而 `num_warps` 在 4 和 8 之间切换，效果比继续加大块更明显。这也印证了材料里只给 `num_warps ∈ {4, 8}` 两组候选的做法——访存受限的核，线程开太多反而互相抢带宽。

第二条支线是 RoPE。我们直接把旋转作用在 Q、K 上，不生成任何拼接后的临时张量。Q 和 K 共用同一份 cos/sin 缓存，也是一处省流量的细节：

```python
@triton.jit
def fused_rope_qk_kernel(Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
                         stride, N, HALF: tl.constexpr):
    row = tl.program_id(0)
    idx = tl.arange(0, HALF)         # 只处理前半段，后半段由配对元素换位得到
    # cos/sin 各读一次，Q 和 K 共用，避免重复加载
    c = tl.load(Cos_ptr + idx).to(tl.float32)
    s = tl.load(Sin_ptr + idx).to(tl.float32)

    q1 = tl.load(Q_ptr + row * stride + idx).to(tl.float32)
    q2 = tl.load(Q_ptr + row * stride + idx + HALF).to(tl.float32)
    # 旋转公式：x' = x*cos - y*sin，y' = x*sin + y*cos
    tl.store(Q_ptr + row * stride + idx,        q1 * c - q2 * s)
    tl.store(Q_ptr + row * stride + idx + HALF, q1 * s + q2 * c)

    k1 = tl.load(K_ptr + row * stride + idx).to(tl.float32)
    k2 = tl.load(K_ptr + row * stride + idx + HALF).to(tl.float32)
    tl.store(K_ptr + row * stride + idx,        k1 * c - k2 * s)
    tl.store(K_ptr + row * stride + idx + HALF, k1 * s + k2 * c)
```

三个内核都写完后，我们统一用一个带 fallback 的入口把它们包起来。这样在 GPU 上走 Triton，在没有 GPU 的校验机上自动退回 NumPy：

```python
def fused_add_rmsnorm(x, residual, weight, eps=1e-6):
    """统一入口：有 CUDA 走 Triton，否则走 NumPy 参考实现"""
    if HAS_TRITON and x.is_cuda:
        grid = (x.shape[0],)                  # 按行切分网格，一行一个 program
        out = torch.empty_like(x)
        h   = torch.empty_like(x)
        _fused_add_rmsnorm_kernel[grid](
            x, residual, weight, out, h,
            x.stride(0), x.shape[1], eps,
            # 块宽取不小于真实宽度的 2 的幂，保证一个块能覆盖一整行
            BLOCK_N=triton.next_power_of_2(x.shape[1]),
        )
        return out, h
    return ref_fused_add_rmsnorm(x.cpu().numpy(),
                                 residual.cpu().numpy(),
                                 weight.cpu().numpy(), eps)
```

### 2.4 模型测试与指标报告

最后一步是「对拍」：用同一组随机输入，分别跑参考实现和 Triton 内核，逐元素比较误差。误差判定严格按比赛给的容差来：

```python
def check(name, got, ref, atol=1e-3, rtol=1e-3):
    """按比赛容差判定内核是否正确"""
    max_abs = np.max(np.abs(got - ref))
    # allclose 同时考虑绝对误差和相对误差
    ok = np.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"{name}: max_abs_err={max_abs:.2e}, pass={ok}")
    return ok

rng = np.random.default_rng(42)
x   = rng.standard_normal((2048, 4096), dtype=np.float32)
res = rng.standard_normal((2048, 4096), dtype=np.float32)
w   = rng.standard_normal((4096,), dtype=np.float32)

ref_out, ref_h = ref_fused_add_rmsnorm(x, res, w)
check("FusedAddRMSNorm / out", out_np, ref_out)   # out_np 是内核回传的结果
check("FusedAddRMSNorm / h",   h_np,   ref_h)

gate = rng.standard_normal((2048, 4096), dtype=np.float32)
up   = rng.standard_normal((2048, 4096), dtype=np.float32)
check("SwiGLU", swiglu_np, ref_swiglu(gate, up))
```

在我们的验证里，融合后的 `Add+RMSNorm` 与参考实现的**最大绝对误差是 0**（材料里的输出就是 `max_abs_err=0.00e+00`）。这说明把中间结果留在寄存器、只在最后写一次显存，并不会引入额外误差——因为数学运算的顺序和精度都没有改变，改变的只是数据放在哪里。

性能方面，四趟访存变成两趟后，理论显存流量下降 50%。我们在 `512×4096` 和 `2048×4096` 两种规模下分别测了时间：小规模时内核启动开销占比更高，大规模时更接近带宽上限。最终提交时，我们按比赛接口把三个内核打包，并保留了 CPU 回退分支，确保在任何校验环境下都能跑通。

---

## 三、总结

本文中我们完成了 FlagOS 开放计算全球挑战赛 Track 1 的 SGLang 推理内核优化。我们首先通过访存分析发现，`Add + RMSNorm` 这类算子在标准实现里要往返显存四趟，是典型的带宽瓶颈；其次我们做了输入规范化，统一用 FP32 累加、用 mask 处理越界、保留 `eps` 防止除零；接着我们用 Triton 写出融合内核，把残差相加、方差累加、`rsqrt` 和权重缩放压进一次读写，并用 `tl.autotune` 在块大小和 `num_warps` 上自动挑配置，同时把 SwiGLU 和 RoPE 也做成融合/原地版本；最后我们按 `atol<=1e-3`、`rtol<=1e-3` 的标准与参考实现对拍，最大绝对误差为 0，显存流量下降约 50%。

这个比赛让我们最直观地体会到：**优化推理性能，很多时候不是在减少计算，而是在减少搬运**。同样的算术，换个写法少读两遍显存，速度就能明显提升；而要让内核在多种芯片上都成立，还得把块大小、线程数和边界处理都做成可配置、可回退的形式。
