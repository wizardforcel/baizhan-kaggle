# Simple Fine-Tuning Competition using LoRA

## 一、比赛简介

### 1.1 比赛背景

这是一场围绕“用 LoRA 微调大语言模型”的入门比赛。任务很具体：给一句英文，让模型把它翻译成法语。比赛提供训练集、验证集和测试集三个 CSV 文件，每一行都是一段 Alpaca 风格的文本，里面包含 `### Instruction`、`### Input`、`### Response` 三个标记。`### Input` 后面是英文原句，`### Response` 后面是法语参考译文。

大语言模型（LLM）通常有几十亿个参数。如果做“全量微调”，也就是把所有权重都更新一遍，需要为每个参数保存梯度 and 优化器状态，显存开销非常大，普通 Kaggle GPU 根本放不下。LoRA（Low-Rank Adaptation，低秩适配）提供了一条捷径：冻结原来的大模型权重，只在部分层旁边插入两个很小的矩阵，训练时只更新这两个小矩阵。这样可训练参数往往只占总参数的百分之一不到，却能让模型学会新任务。

比赛还有几条明确约束：只能使用比赛提供的数据，不能引入外部语料；不能做数据增强；只允许 LoRA 微调，不允许全量微调；任何预训练基础模型都可以使用；提交文件必须是 UTF-8 编码的 `id,french` 两列；评价指标是精确匹配准确率。材料里的方案选择 Mistral-7B 作为基础模型，用 PEFT 库实现 LoRA。

### 1.2 比赛目标

训练集和验证集的每一行都有完整的英文输入和法语参考输出，测试集只有英文输入（以及一个 `id`）。我们要让模型读入提示词，生成对应的法语译文，然后把结果写进提交文件。

这里有一个容易被忽略的细节：评价是“精确匹配”（exact match），也就是模型输出必须和参考答案在规范化之后一字不差。多一个句号、少一个大写字母、词序稍有不同，都会被判错。所以除了模型本身，输出格式的后处理也同样重要。

### 1.3 评价标准

准确率的计算方式是：预测译文与参考译文完全相同（比较前统一转小写、去掉多余空白）的样本数，除以测试样本总数。越高越好，满分是 1.0。它不奖励“意思差不多”的翻译，只奖励和参考答案逐字一致的输出。这解释了为什么材料花了大量篇幅做格式分析：如果参考答案 99% 都以大写字母开头、以句号结尾，那么模型输出也必须遵守同样的习惯，否则即使翻译正确也会被判错。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先加载三个 CSV 文件，看看行数和原始文本的精确样子。这里用 `repr()` 而不是直接打印，因为普通打印会把换行、行尾空格、引号等“看不见的字符”隐藏起来，而精确匹配恰恰对这些字符敏感。

```python
import pandas as pd
from datasets import load_dataset

DATA_FILES = {
    "train": "/kaggle/input/.../train.csv",
    "validation": "/kaggle/input/.../validation.csv",
    "test": "/kaggle/input/.../test.csv",
}
ds = load_dataset("csv", data_files=DATA_FILES)
train_df = pd.read_csv(DATA_FILES["train"])
val_df = pd.read_csv(DATA_FILES["validation"])
test_df = pd.read_csv(DATA_FILES["test"])

print("Rows:", len(train_df), len(val_df), len(test_df))
print(repr(train_df["text"].iloc[0]))  # repr 能看到换行和空格
```

我们其次把每行拆成三部分：提示词 `src`、法语参考 `ref`、纯英文输入 `input_only`。拆分靠 `### Response:` 标记，英文输入再用正则从 `### Input:` 后面提取。这样后续做重叠分析和推理时都能复用同一套函数。

```python
import re
RESPONSE_MARKER = "### Response:"

def split_text(text):
    if RESPONSE_MARKER in text:
        p, r = text.split(RESPONSE_MARKER, 1)
        return p.strip(), r.strip()
    return text.strip(), ""

def extract_input(src):
    m = re.search(r"### Input:\s*(.*)", src, flags=re.DOTALL)
    return (m.group(1) if m else src).strip()

for df in (train_df, val_df, test_df):
    parsed = df["text"].apply(split_text)
    df["src"] = parsed.apply(lambda x: x[0])
    df["ref"] = parsed.apply(lambda x: x[1])
    df["input_only"] = df["src"].apply(extract_input)
```

EDA 里最关键的一步是检查训练集和测试集的英文输入有多少重叠。如果测试句子几乎都在训练集里出现过，那么任务本质上更接近“记忆查表”，高分是可以预期的；如果大量句子从未出现，才是真正的泛化挑战。

```python
def norm(s):
    return " ".join(s.lower().split())

train_in = set(train_df["input_only"].map(norm))
test_norm = test_df["input_only"].map(norm)
unseen = ~test_norm.isin(train_in)

print("Unique inputs in train:", train_df["input_only"].map(norm).nunique())
print("Test seen in train:", (~unseen).sum(), "/", len(test_df))
print("Test never seen:", unseen.sum(), "/", len(test_df))
```

材料的结果显示，大约 99% 的测试输入在训练集中出现过。这直接决定了策略：模型不需要从零学会法语，只需要把训练集中“英文→法语”的对应关系记住，并对少数没见过的句子做合理翻译。因此高准确率是合理的，而训练轮数、LoRA 容量和输出格式才是决定分数的关键。

我们还统计了参考译文的长度和格式指纹。参考译文绝大多数不超过 70 个词，所以生成时把最大新词数设为 96 已经足够；同时统计“是否大写开头”“是否以句号、感叹号或问号结尾”“是否有首尾空格”，为后面的确定性后处理提供依据。

```python
refs = train_df["ref"]
last = refs.str.strip().str[-1]
print("starts uppercase:", refs.str[0].str.isupper().mean())
print("ends with punctuation:", last.isin(list(".!?")).mean())
print("has leading/trailing space:", (refs != refs.str.strip()).mean())
```

### 2.2 数据清洗

这份数据的清洗不是删行，而是“把文本整理成模型能学的格式”。我们接着做两件事：构造训练标签时只对回答部分计算损失；推理时对生成结果做统一后处理。

第一件事很关键。Alpaca 格式的输入包含指令、英文句子和回答。如果让模型对整段文本都计算损失，它会把大量算力花在“复述提示词”上，而不是学习翻译。所以我们把提示词部分的标签全部设成 `-100`，PyTorch 的损失函数会自动忽略这些位置；只有回答部分的 token 参与损失计算。

```python
MAX_LENGTH = 512

def tokenize_function(examples):
    input_ids_batch, attention_batch, labels_batch = [], [], []
    for text in examples["text"]:
        if RESPONSE_MARKER in text:
            prompt, response = text.split(RESPONSE_MARKER, 1)
            prompt = prompt + RESPONSE_MARKER + "\n"
            response = response.lstrip("\n")
        else:
            prompt, response = text, ""

        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        # 回答末尾补 EOS，模型才会学会“什么时候停”
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        input_ids = (prompt_ids + response_ids)[:MAX_LENGTH]
        # 提示词位置填 -100，不参与损失
        labels = ([-100] * len(prompt_ids) + response_ids)[:MAX_LENGTH]

        input_ids_batch.append(input_ids)
        attention_batch.append([1] * len(input_ids))
        labels_batch.append(labels)
    return {"input_ids": input_ids_batch,
            "attention_mask": attention_batch,
            "labels": labels_batch}
```

第二件事是输出后处理。因为精确匹配对格式极其敏感，我们统一规定：只取生成结果的第一行；去掉首尾空白；首字母大写；如果结尾不是 `.`、`!`、`?` 就补一个句号。这不是“作弊”，而是把 EDA 中观察到的参考答案格式固化成规则，减少模型因为格式差异被误判。

```python
def postprocess(pred):
    p = pred.strip().split("\n")[0].strip()  # 只保留第一行
    if not p:
        return p
    p = p[0].upper() + p[1:]                 # 首字母大写
    if p[-1] not in ".!?":
        p += "."                             # 补上句号
    return p
```

另外，分词器的 `pad_token` 需要特别处理。材料明确指出不能用 `eos` 当 `pad`：如果两者相同，数据整理器会把真正的结束符也当成填充而屏蔽掉，模型就永远学不会停止生成。因此这里用 `<unk>` 作为填充符号。

### 2.3 模型训练与超参数调优

我们然后配置 LoRA。它的核心公式是 `W' = W + (α/r)·B·A`：原来的权重 `W` 冻结不动，只训练两个低秩矩阵 `A` 和 `B`。`B` 初始化为 0，所以训练开始时模型输出和原模型完全一致，相当于从预训练状态平滑起步。秩 `r` 控制适配器的容量，`lora_alpha` 控制更新幅度，`target_modules` 决定往哪些层插适配器。

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,                       # 低秩矩阵的秩，越大容量越强
    lora_alpha=32,              # 常用约定 alpha = 2 * r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",      # 因果语言模型
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 查看可训练参数占比
```

材料提到，最初只用 `r=8` 且只适配 `q_proj`、`v_proj` 时，准确率已经能到约 0.99；后来为了增强记忆能力，改成 `r=16` 并适配所有注意力与 MLP 投影层。因为数据集本身高度可记忆，过拟合风险不大，所以扩大容量是划算的。

训练参数方面，单卡显存有限，所以每张卡 batch size 设为 4，再用 8 步梯度累积，等效 batch size 为 32；学习率 2e-4，余弦调度，3% 预热，权重衰减 0.01，训练 2 个完整 epoch，使用 fp16 混合精度。材料特意关闭了 `load_best_model_at_end`，原因是验证损失最低的检查点并不等于精确匹配最好的检查点：损失关心概率，精确匹配关心最终字符串，两者不完全一致。

```python
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

training_args = TrainingArguments(
    output_dir="/kaggle/working/results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   # 等效 batch size = 4 × 8 = 32
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0.03,
    weight_decay=0.01,
    num_train_epochs=2,
    fp16=True,
    group_by_length=True,            # 长度相近的样本放一起，减少 padding
    eval_strategy="steps",
    eval_steps=150,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        label_pad_token_id=-100, padding=True),
)
trainer.train()
model.save_pretrained("/kaggle/working/fine_tuned_lora")  # 只保存适配器
```

这里没有做大规模超参数搜索，因为影响因素很明确：数据几乎可记忆，所以训练是否充分、适配器容量是否足够、输出格式是否对齐，比微调学习率的小数点更重要。若要继续调，优先尝试增加 epoch、提高 `r`，或者适当放宽早停耐心。

### 2.4 模型测试与指标报告

我们最后做批量推理。生成时把 `padding_side` 设为 `left`，这是 decoder-only 模型批量生成的要求：所有序列必须从右侧对齐，新 token 才能接着同一列继续生成。`do_sample=False` 表示贪心解码，保证同样输入得到确定输出，便于复现和精确匹配。

```python
import torch
model.eval()
tokenizer.padding_side = "left"
BATCH_SIZE, MAX_NEW_TOKENS = 16, 96
predictions = []

for start in range(0, len(prompts), BATCH_SIZE):
    batch = prompts[start:start + BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True,
                       truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # 只解码新生成的部分，不把提示词本身算进结果
    gen_only = outputs[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    predictions.extend(postprocess(d) for d in decoded)

assert len(predictions) == len(test_df)
```

如果本地测试集带有参考答案，我们还能算一个本地精确匹配分数，作为提交前的 sanity check。比较前统一小写并压缩空白，避免因为大小写或多余空格产生假错误。材料没有给出一个固定的榜单分数，因此我们只报告“本地有参考时会打印准确率”，不编造具体数值。

```python
def n(t):
    return " ".join(t.lower().strip().split())

if has_reference:
    acc = sum(n(p) == n(r) for p, r in zip(predictions, references)) / len(predictions)
    print(f"Local exact-match accuracy: {acc*100:.2f}%")
```

我们还做了错误分析，把失败样本分成四类：第一类是“格式问题”，词都一样，只是标点、大小写或空格不同，可以通过加强后处理解决；第二类是“训练集中本身有歧义”，同一个英文句子对应多个法语参考，模型无法知道该选哪个，可以用多数投票查表补救；第三类是“见过但模型答错”，说明记忆不充分，可以增加训练轮数或提高秩；第四类是“完全没见过”，属于真正的泛化，模型可能给出语义正确但措辞不同的译文，这类很难靠精确匹配拿分。

```python
from collections import Counter
train_map = train_df.groupby("input_only")["ref"].apply(list).to_dict()

for pred, ref, prompt in zip(predictions, references, prompts):
    if norm(pred) == norm(ref):
        continue
    eng = extract_input(prompt.split(RESPONSE_MARKER)[0])
    refs_in_train = train_map.get(eng)
    if pred.strip() == ref.strip():
        etype = "formatting"
    elif refs_in_train is None:
        etype = "unseen"
    elif len(set(map(norm, refs_in_train))) > 1:
        etype = "ambiguous in train"
    else:
        etype = "seen, model wrong"
    print(etype, "|", eng[:50], "|", pred[:50])
```

最后按比赛要求写出 `id,french` 两列，使用 UTF-8 编码，保证法语重音字符不会损坏。写完后检查行数是否和测试集一致、id 顺序是否保持原样。

```python
submission = pd.DataFrame({"id": test_df["id"], "french": predictions})
submission.to_csv("/kaggle/working/submission.csv",
                  index=False, encoding="utf-8")
assert len(submission) == len(test_df)
print(submission.head())
```

## 三、总结

本文中我们完成了一次基于 LoRA 的英法翻译微调。我们首先做了 EDA，发现约 99% 的测试英文输入在训练集中出现过，因此任务更接近“记忆映射”而不是开放域翻译；同时我们统计了参考译文的长度和标点习惯。其次我们把 Alpaca 格式拆成提示词和回答，构造只对回答计算损失的标签，并写了一套确定性的输出后处理。接着我们用 Mistral-7B 加 LoRA 适配器训练，设置 r=16、覆盖注意力和 MLP 投影层，用梯度累积在有限显存下得到等效 batch size 32，训练 2 个 epoch。最后我们做贪心批量生成、错误分类和提交文件校验。

这个比赛让我们理解了 LoRA 的实用价值：它把“微调 70 亿参数”变成“训练一小撮适配器参数”，让普通 GPU 也能完成大模型任务。另一个重要收获是，评价指标会反过来塑造整个流程。精确匹配让格式后处理和错误分析变得和模型本身一样重要；如果只看“翻译得对不对”，就会忽略大量因为一个句号而丢分的样本。
