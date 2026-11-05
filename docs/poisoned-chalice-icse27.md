# Poisoned-Chalice ICSE 27

## 一、比赛简介

### 1.1 比赛背景
我们面对的是一个关于大语言模型隐私泄露的研究型比赛，比赛全称为 Poisoned-Chalice ICSE 2027。ICSE 是指国际软件工程会议，本次 Kaggle 比赛是基于 ICSE 2027 的公开数据和任务设计而来。核心问题是成员推断攻击 Membership Inference Attack，简称 MIA。

大白话说，模型在训练的时候会记住训练集里见过的文本。当我们把一段代码喂给模型时，如果这段代码正好在训练数据里，模型给出的困惑度通常会更低、损失更小。反之，从未见过的数据，模型会更“困惑”。攻击者利用这种差异，判断某条样本是否属于训练集。代码是从 Go、Java、Python、Ruby、Rust 五种编程语言的开源仓库采集中来，形式上是单文件代码片段，样本编号为 sample_id，内容放在 content 字段，验证集还附带一个 membership 标签，用于评测。

这类攻击的意义在于隐私合规：如果一个模型能泄露它是否见过某个用户的私有代码，就可能间接暴露训练数据的来源。比赛的出发点是让参赛者复现最基本的损失攻击，对模型是否存在记忆进行定量评估，然后在 Kaggle 上提交评分。

### 1.2 比赛目标
比赛的目标很明确：给定验证集中的代码片段，我们要预测每个样本是否属于模型的训练集。提交文件只需要两列，sample_id 和 membership_score。评测时系统会把我们的得分与真实的 membership 标签进行排序比较。分数越高，说明我们能越好地分辨“见过”和“没见过”的样本。

具体流程里，有两个人协作：第一个人用 Hugging Face Hub 或 Kaggle 输入接口加载数据，数据分散在五个语言子集下，分别读取 train 和 validation。第二个阶段是调用一个公开的开源代码模型，通常是 bigcode/starcoder2-3b，对每一个文本计算自回归语言建模的交叉熵损失，再把损失取负作为成员得分。最后把得分导出成 CSV，sample_id 由 pandas 自动写入索引，membership_score 作为评分列。

从用户材料来看，提供了 Starter Kit 的示例，演示了如何通过 HF Hub 加载 Poisoned-Chalice/ICSE-2027-public 数据集，也演示了 Kaggle 上的 CSV 加载路径。比赛没有要求训练自己的模型，而是把 Attack 的实现作为核心能力来测试。

### 1.3 评价标准
评测指标是 ROC 曲线下面积 AUC。AUC 不关心绝对阈值，只关心排序：随机抽一个真成员和一个真非成员，我们的得分是否能把前者排在后者前面。AUC = 0.5 表示随机猜测，AUC = 1.0 表示完美排序。损失攻击的直觉就是“训练样本损失更小，得分更高”，因此把负损失作为得分时，成员的得分分布理应整体右移。

需要提醒一点，验证集里并非所有样本都有 membership 标签，Starter Kit 里对标签缺失的样本会用 dropna 过滤。AUC 只在标签完整的那部分样本上计算。ROC 曲线也会被绘制，横轴是假阳性率，纵轴是真阳性率，图中会标出 AUC 数值以供直观比较。

## 二、解题过程

### 2.1 EDA
我们先把五个语言的数据读进来，观察数据的基本结构和标签分布。因为比赛材料以 HF Hub 为主，我们复现了 Starter Kit 的加载方式，先把每个语言的 train 和 validation 分别转成 pandas 并加上 sample_id 索引。

```python
from datasets import load_dataset
import pandas as pd

# 用来暂存每个语言的训练和验证片段
train_list = []
val_list = []

# 比赛共涉及五种编程语言
for lan in ['Go','Java','Python','Ruby','Rust']:
    ds = load_dataset("Poisoned-Chalice/ICSE-2027-public", lan)
    # 把数据集的 train/validation 换成 DataFrame，sample_id 设为索引方便对齐
    train_list.append(ds['train'].to_pandas().set_index('sample_id'))
    val_list.append(ds['validation'].to_pandas().set_index('sample_id'))

# 拼接成整体数据集
train = pd.concat(train_list)
valid = pd.concat(val_list)

print(train.shape, valid.shape)
print(valid.columns)
```

我们发现每个样本只有 sample_id、content、membership 三列，content 都是纯文本代码。样本编号是连续的但跨语言会重复，因此拼接前保持独立索引很重要。

接下来检查验证集的基本情况：代码长度、标签缺失比例、语言分布。我们把文本长度用字符数统计，并把 membership 标签计数。

```python
# 计算文本长度分布，以便了解截断是否会丢失信息
valid['text_len'] = valid['content'].astype(str).str.len()

# 统计语言来源，train 和 valid 都是按语言分块加载的，后续可据此检查偏置
print(valid['text_len'].describe())

# membership 标签统计，注意部分样本没有标签
print(valid['membership'].value_counts(dropna=False))
```

材料里没有给出验证集标签的精确比例，但 Starter Kit 明确写了要先用 dropna 去掉没有 membership 的样本，说明确实存在一部分无标签数据。代码长度也没有被统计成表，我们只知道模型的截断长度设为 512 个 token，超过这个长度的片段会被截断，因而损失只反映前 512 个 token 的信息。

我们还随机抽取几个样本打印前 200 字符，确认内容是可读的代码块，没有明显的二进制污染或空内容。这些检查让我们对后续计算损失时的截断策略有了预期。

### 2.2 数据清洗
在计算损失前，需要把数据整理成模型可用的格式。首先统一索引并去重，避免同一个 sample_id 被重复计分。其次处理缺失内容。

```python
# 去除内容为空或仅包含空白的样本，防止 token 化后产生长度为零的输入
valid = valid[valid['content'].astype(str).str.strip().ne('')].copy()

# 如果有重复样本编号，保留第一次出现
valid = valid[~valid.index.duplicated(keep='first')]

# 为了避免 token 长度为零导致损失异常，稍后在评分函数里会检查 token 长度 <2
print(f'清洗后验证集样本数：{len(valid)}')
```

对于训练集，我们的任务不涉及主动训练，但为了与验证集对齐，我们也做了相同的去重和空值检查，确保后续拼接不会产生对齐错误。

文本预处理方面，比赛提供的代码片段已经是清理过的源文件，没有需要额外做分词或去注释。我们保留原始内容，因为损失攻击依赖模型对原始分布的敏感度，过度清洗会抹平记忆信号。

截断策略也在这里确定。我们统一使用 512 个 token 作为最大长度，这在 starcoder2 的 16384 上限之下能够保证显存占用可控，同时覆盖绝大多数片段。超过长度的样本会被 tokenizer 截断，从开头保留前 512 token，这是一种常用的长度归一化方式。

最后我们把 membership 标签转成布尔值，方便后续 AUC 计算。

```python
# membership 标签通常为 "member" 或空值，转成布尔标签
valid['label'] = valid['membership'].apply(lambda x: x == "member")

# 只保留有标签的样本用于评测，评分阶段仍需对全部样本输出得分
evaluated = valid.dropna(subset=['membership'])
print(evaluated['label'].value_counts(normalize=True))
```

标签比例在材料中没有给出具体数字，我们只从代码中知道用于 AUC 计算的是 dropna 后的有标签样本。

### 2.3 模型训练与超参数调优
损失攻击本身不需要额外训练模型，我们直接使用公开的 bigcode/starcoder2-3b 作为目标模型。模型的大小和结构直接决定了记忆能力，因此超参数调优主要体现在计算设置上，而不是传统意义上的训练参数搜索。

我们首先设置了模型和环境的配置，确保在 GPU 上使用半精度加速推理。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "bigcode/starcoder2-3b"
MAX_LENGTH = 512            # 截断长度，平衡信息保留和显存
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

print(f'使用设备 {DEVICE}，数据类型 {DTYPE}')

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
model.eval()  # 推理模式，关闭 dropout 和梯度
```

核心的评分函数是计算单个文本的负损失。注意我们用 `labels = input_ids` 让模型做自回归语言建模，返回的 loss 就是交叉熵损失。损失越小表示模型越“熟悉”，因此我们取负数作为 membership_score，得分越高越像成员。

```python
import numpy as np

@torch.inference_mode()
def membership_score(text):
    # 对文本进行 token 化，截断到 MAX_LENGTH
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    input_ids = tokens["input_ids"]
    # 长度小于2的样本无法计算有意义的损失，标记为 NaN
    if input_ids.shape[1] < 2:
        return np.nan
    # 计算语言建模损失
    loss = model(**tokens, labels=tokens["input_ids"], use_cache=False).loss
    return -loss.item()
```

我们没有做并行批处理，保持逐条计算以降低显存峰值，这与 Starter Kit 保持一致。虽然在 T4 显卡上可以用批处理加速，但在写说明时我们强调代码的简洁性。

超参数方面，我们尝试了两个方向的调优想法并在代码中留下接口。第一是 MAX_LENGTH 的选择，512 是默认值，如果显存允许可以提升到 1024 或 2048 以保留更多上下文。第二是模型大小的替代，例如 starcoder2-7b 或其他代码模型，模型越大记忆效应越明显，但计算成本也越高。

在实际调参时，我们可以对验证子集采样，用不同的 MAX_LENGTH 计算得分并观察 AUC 的变化趋势，找到在性能和速度之间的甜点。Starter Kit 里最终固定为 512，也是为了可复现性。

### 2.4 模型测试与指标报告
评分完成之后，我们把得分写入验证集，然后用 sklearn 的 roc_auc_score 和 roc_curve 计算指标。

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# 对验证集逐条打分
valid["membership_score"] = [membership_score(t) for t in valid["content"]]
# 过滤掉评分失败的样本
evaluated = valid.dropna(subset=["membership_score", "membership"])

auc = roc_auc_score(evaluated["label"], evaluated["membership_score"])
fpr, tpr, _ = roc_curve(evaluated["label"], evaluated["membership_score"])
print(f"ROC-AUC: {auc:.4f}")
```

我们把 ROC 曲线画出来，用红色曲线表示损失攻击的性能，灰色虚线表示随机基线。

```python
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f"Loss Attack (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="0.5", label="random")
ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
ax.grid(alpha=0.25)
ax.legend(loc="lower right")
fig.tight_layout()
fig.show()
```

AUC 数值在材料中未给出具体数字，Starter Kit 只展示了如何计算和绘制 ROC 曲线，说明损失攻击的 AUC 应高于随机基线，具体数值依赖运行环境和模型版本。

提交格式要求非常简单，我们只要导出 sample_id 作为索引，membership_score 作为列。

```python
valid.to_csv('loss_submission.csv', columns=['membership_score'])
```

这样生成的 CSV 包含完整的 sample_id 与得分序列，评测系统会读取并计算整体 AUC。提交后我们还会检查文件行数是否与验证集一致，确认没有丢失样本。

## 三、总结
本文中我们先介绍了 Poisoned-Chalice ICSE 2027 比赛的背景，这是一个围绕大语言模型成员推断攻击的隐私评估任务，目标是判断代码片段是否出现在训练集中，评价标准是 ROC-AUC。接着我们在解题过程中，先用 HF Hub 加载了 Go、Java、Python、Ruby、Rust 五种语言的公开数据集，并对验证集进行了基本的长度分布和标签比例探索，然后清洗了空内容和重复样本，统一了截断长度。模型部分没有进行传统训练，而是直接使用 bigcode/starcoder2-3b 计算自回归损失，负损失作为成员得分，并通过 MAX_LENGTH、数据类型等计算参数做了有限的调优。最后我们对验证集打分、计算 AUC 并绘制 ROC 曲线，生成了符合 Kaggle 规则的提交文件。
