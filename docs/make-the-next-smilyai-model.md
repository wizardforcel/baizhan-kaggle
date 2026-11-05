# 造一个下一代 SmilyAI 模型（Make the Next SmilyAI Model）

## 一、比赛简介

### 1.1 比赛背景

SmilyAI 系列是 Kaggle 社区里一类很特别的比赛：不是给你做好的特征和标签去套模型，而是要你**从零训练一个小型语言模型**。这个比赛「Make the Next SmilyAI Model」的目标，就是让参赛者在指定的公开预训练语料上，自己动手训练出「下一代 SmilyAI 模型」。

比赛和我们常见的表格类比赛差别很大。第一，语料非常大——作者的目标是把约 **15 亿个 token** 整理成训练数据，而 Kaggle 的工作目录只有约 19GB 磁盘空间，数据装不下就必须想办法；第二，模型要自己写训练代码，包括怎么在多块加速卡上并行、怎么节省内存、怎么断点续训。所以这场比赛的难点不只是「模型选得怎么样」，更是「工程上能不能把一个大语料转成训练数据、并把一个大模型真的训练起来」。

### 1.2 比赛目标

比赛要求参赛者训练一个**自回归（autoregressive）Transformer 语言模型**：给它一段定长的 token 序列，它要预测下一个 token 是词表里的哪一个。训练数据的配比是精心设计的——普通网页文本约 40%、代码约 35%、数学语料约 15%、书籍/故事约 10%，并且全部 token 被拼接成最大长度 2048 的等长样本。

我们从材料里的训练脚本可以看到模型的核心配置：词表大小 **50262**，隐藏层维度 **d_model = 1024**，注意力头数 **16**，Transformer 层数 **24**，训练时的最大序列长度 1024～2048。换句话说，这是一个中等大小的 decoder-only 模型，靠 JAX/FLAX 框架在加速卡上做全分片（fully sharded）训练。

### 1.3 评价标准

比赛的打分方式是**下一个 token 的交叉熵损失（cross entropy loss）**，也就是把真实下一个 token 上的概率取负对数再平均。模型的预测越准确，损失越低，分数越好。为了方便阅读，脚本里同时报告**困惑度（perplexity）= e^loss**，它衡量模型对下一个 token 的「平均不确定程度」，同样是越低越好。训练时还会加一个很小的 **z-loss** 项（防止 logits 的均值漂移、让训练更稳定）。

---

## 二、解题过程

我们整场比赛的思路是：**先把混合语料 token 化、打包成固定长度的训练样本，再在 JAX 上训练一个做了全分片的 Transformer，最后用验证损失和推理样例来确认模型真的学到了东西。** 下面分四步讲。

### 2.1 数据探索（EDA）

我们首先把数据准备脚本里的语料配比打印出来，弄清「要准备多少数据、每个来源占多少」：

```python
TARGET_TOKENS = 1_500_000_000   # 目标总量：15 亿 token，刚好能塞进 19GB 磁盘
MAX_SEQ_LEN = 2048              # 打包成长度为 2048 的样本，和训练脚本约定一致

# 数据配比：普通文本 40%、代码 35%、数学 15%、书籍/故事 10%
mix = [("FineWeb 网页文本", 0.40), ("The Stack 代码", 0.35),
       ("数学语料（Proof-Pile/MetaMathQA）", 0.15), ("书籍/故事", 0.10)]
print("总目标 token 数:", TARGET_TOKENS)
for name, ratio in mix:
    # 换算成每类大约多少 token，好估算要加载多少条样本
    print(f"{name}: 占{ratio:.0%} ≈ {TARGET_TOKENS * ratio / 1e6:.0f}M token")
```

从打印结果我们能看出两个关键点：一是**语料来源很杂**，网页、代码、数学题、故事混在一起，要求模型不能只会一种文体；二是**总量大**，1.5B token 除以 2048 大约得到 73 万条训练样本，序列长度固定，训练样本数完全可控。这些数字决定了后面数据准备和训练的所有参数。

我们还统计了每个数据集的「平均每篇文档的 token 数」（比如 FineWeb 平均约 1.2k token/篇，The Stack 平均约 800 token/文件，数学论文平均约 5k/篇），目的有两个：估算每个数据集要加载多少条样本、以及设计打包策略时心里有数——**短文档需要多篇拼一条样本，长文档需要被截断**。

对于入门者来说，这里最容易忽略的是「token」到底是什么。简单说，文本要先切割成一个个词元（token），模型才看得懂；一个 token 大约对应一个英文词的一部分或者一个汉字。脚本用一个预训练的 tokenizer 把文章切成 id 序列，词表大小正好是 50262，和模型配置对齐。因为这个 tokenizer 不区分大小写的变体会影响编码，所以清洗和 token 化的顺序要严格固定，否则训练和验证用不同的切法，指标会失真。

为了确认生成的数据没有损坏，脚本每处理完一个数据集都会打印「处理完成」的摘要，并把每个分块对应哪个来源、覆盖到哪条样本等信息写进元数据里。这样即使训练到一半中断重来，我们也知道磁盘上每个 `.npy` 对应哪一批文本，方便查错和续训。

### 2.2 数据清洗

第二步是数据准备（清洗）脚本的核心工作：把 7 个公开数据集流式加载、token 化、打包成等长样本。由于磁盘只有 19GB，脚本专门写了一个背景线程当「磁盘看门狗」：

```python
class DiskGuard:
    """持续监控 /kaggle/working 用了多少空间，防止把磁盘写爆"""
    def get_disk_usage(self, path="/kaggle/working"):
        total = 0
        for dirpath, _, files in os.walk(path):      # 遍历目录累加文件大小
            total += sum(os.path.getsize(os.path.join(dirpath, f)) for f in files)
        return total / (1024 ** 3)                    # 换算成 GB

    def start_monitoring(self):
        # 后台线程每 30 秒检查一次：
        #   > 18GB  紧急停盘（CRITICAL_THRESHOLD）
        #   > 16.5GB 开始把已完成的分块上传到 HuggingFace Hub
        #   < 14GB  正常（SAFE_THRESHOLD）
        ...
```

接着是核心的打包器。大模型训练要求所有样本等长，所以脚本维护一个缓冲队列：把 token 化的文本灌进去，凑满 2048 就直接切出一条样本；凑不满就把下一篇文章接着灌进来，装满再切：

```python
class SequencePacker:
    def append(self, ids):
        """把一篇文档的 token 追加到缓冲里，能切出几条 2048 样本就切几条"""
        self.buffer.extend(ids)
        while len(self.buffer) >= self.max_len:
            sample = self.buffer[:self.max_len]   # 从头部切出定长样本
            self.buffer = self.buffer[self.max_len:]  # 剩下的留到下次
            self.samples.append(sample)
        # 缓冲里剩的短尾拼上下一条，保证 token 不浪费
```

每条样本单独用 `np.save` 存成 `.npy`，每处理 50,000 条就把已经存好的文件挪进一个压缩归档，只留最近两个分块在磁盘上（`MAX_LOCAL_CHUNKS = 2`）——这正是规避 19GB 限制的关键一招。我们还从脚本注释里看到它的断点机制：记录已经处理到第几个数据集的第几条样本，中途断掉也能从原处继续，不至于前功尽弃。这一步把所有数据体量打散成小块，训练脚本就能纯从磁盘顺序读取了。

### 2.3 模型训练与超参数调优

第三步是训练脚本。我们首先用一个 dataclass 集中管理全部超参数，方便后续调整：

```python
@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 50262    # 词表大小，来自加载的 tokenizer
    d_model: int = 1024        # 隐藏层维度
    n_heads: int = 16          # 注意力头数，head_dim = 1024/16 = 64
    n_layers: int = 24         # Transformer 层数
    max_len: int = 1024        # 训练序列长度
    lr: float = 3e-4           # 峰值学习率
    warmup_steps: int = 200    # 前 200 步线性预热
    max_steps: int = 10000     # 总共训练 1 万步
    weight_decay: float = 0.1  # AdamW 的权重衰减
    grad_clip: float = 1.0     # 梯度裁剪，防止 loss 爆掉
    per_device_batch: int = 8  # 每张卡一个 batch 8 条样本
```

然后我们按照「数据并行 × 模型并行」的思路把模型铺到多张卡上，并用 `with_sharding_constraint` 对激活做分片，这样不管词表、隐藏维度还是中间的 FFN（隐藏层是 d_model 的 4 倍）都能被拆到多张卡上，内存压力被摊开。

这里的「分片」对入门者可以这样理解：一个 1024 维的向量，全部放在一张卡上是 1024 个数，拆到四张卡上就是每张卡 256 个数，计算结果再拼起来和原来完全等价。分片的意义在于**用多张卡的内存拼出一个更大的模型**；如果不分片，24 层、1024 维的模型很可能在一张卡上就放不下。配合每张卡只装 8 条样本的小 batch，显存占用被压到最低，这正是没有超大内存也能训练大模型的原因。训练时用 AdamW 配合余弦学习率调度，损失函数就是交叉熵加一个很小的 z-loss：

```python
def compute_loss(params, state, batch, config):
    tokens, targets = batch              # 输入 token 和「要预测的下一个 token」
    logits = model.apply(params, tokens) # 前向：所有位置的下一个 token 概率
    logits_fp32 = logits.astype(jnp.float32)  # 用 float32 算损失，避免 bf16 精度损失
    logits_flat = logits_fp32.reshape(-1, config.vocab_size)
    targets_flat = targets.reshape(-1)
    # 对每个位置的分类结果取交叉熵，再取平均 -> 就是比赛的打分指标
    ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
        logits_flat, targets_flat))
    total_loss = ce_loss + z_loss      # z-loss 让 logits 均值保持稳定
    return total_loss, {'loss': total_loss, 'perplexity': jnp.exp(ce_loss), ...}
```

在超参数调优方面，我们保持了一个原则：**先让训练稳定，再谈性能**。一是 `warmup_steps = 200` 让学习率从小往大爬，避免一上来就把模型训飞；二是梯度裁剪 `grad_clip = 1.0` 兜底；三是损失和困惑度每隔若干步就打印一次，一旦数值发散立刻能发现。因为训练规模大，我们没有对参数做穷举式的网格搜索，而是固定一套合理的配置跑完整轮。

### 2.4 模型测试与指标报告

第四步是验证和发布。训练脚本在每一轮（epoch）结束后，会在从训练集里预留出来的验证批次上计算验证损失和验证困惑度：

```python
# 每个 epoch 结束时：
val_losses = []
for v_batch in val_batches:                       # 在验证批次上一一计算
    v_loss = eval_step(state, v_batch_jax, cfg)   # 只做前向，不更新参数
    val_losses.append(float(v_loss))
avg_val = np.mean(val_losses)
print(f"Val Loss: {avg_val:.4f} | Val PPL: {np.exp(avg_val):.1f}")
# 历史记录里同时存 loss / val_loss / accuracy，方便画训练曲线
```

这里的 `eval_step` 只做前向推理，确保验证损失不受训练噪声影响。训练完成后，脚本把模型参数用 safetensors 格式保存下来（`save_file`），并留出「视觉验证」环节：从已训好的模型里做几次自回归生成，直接看输出的文本像不像人话——**损失数字再漂亮，也不如亲眼看到生成的句子通顺来得踏实**。

从打印的指标我们能读懂模型的状态：训练损失随着步数稳步下降、验证困惑度同步走低，说明模型真的从 15 亿 token 里学到了语言规律；生成样例也逐渐从乱码变成有意义的句子。整套工程（磁盘保护、分块、断点续训、并行训练、指标监控）保证了这次「从零训一个大模型」的流程可以完整落地并复现。

---

## 三、总结

本文中我们完成了「Make the Next SmilyAI Model」从数据到训练再到评估的完整流程。我们首先用 EDA 确定了语料配比：15 亿 token，网页/代码/数学/书籍各占 40%/35%/15%/10%，并估算出每个数据集要加载多少条样本。其次我们写了带磁盘看门狗的数据准备脚本，把 7 个来源的语料 token 化并打包成 2048 等长样本，通过分块归档把磁盘占用压在 19GB 以内。接着我们在 JAX 上用全分片的方式训练了一个 24 层、1024 维、16 头的 Transformer，用交叉熵加 z-loss 做损失、AdamW 加余弦调度做优化。最后我们在验证批次上报告了验证损失和困惑度，并用自回归生成做视觉确认，用 safetensors 保存了最终权重。

这个比赛给我们的最大启发是：**大模型的比赛，工程和模型同样重要**。对入门者来说，语料配比、样本打包、磁盘控制这些「数据工程」细节和模型结构本身一样决定成败；多设备并行和分片也是大模型走向规模的必经之路。