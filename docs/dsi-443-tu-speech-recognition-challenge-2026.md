# 泰语语音识别挑战（DSI-443 TU Speech Recognition Challenge 2026）

## 一、比赛简介

### 1.1 比赛背景

这是一个泰语自动语音识别（Automatic Speech Recognition，简称 ASR）比赛。ASR 要做的事情很直白：给一段录音，把它说的话转成文字。我们平时用的语音输入、会议字幕、语音助手，背后都是 ASR。

泰语给 ASR 带来了两个额外难点。第一，泰语书写**词与词之间不加空格**，一句话连成一片，所以"一个词从哪里开始、到哪里结束"本身就需要分词工具来判断；这直接影响评价指标怎么算。第二，泰语里有大量梵语、巴利语借词和人名地名，同一个发音可能对应好几种写法，模型很容易在专有名词上出错。

材料来自一份名为 "Thai ASR — Fresh Start + Vocabulary" 的 Kaggle notebook。作者没有沿用旧方案，而是从预训练模型重新开始：先测基线，再用 LoRA 做小步微调，最后尝试用外部词表给解码器"提词"。

### 1.2 比赛目标

输入是一段泰语音频，输出是对应的泰语文本。比赛数据提供三个 CSV：

- `train.csv`：带参考文本的训练音频；
- `dev.csv`：带参考文本的开发集，用来本地评估和选模型；
- `test.csv`：没有参考文本，需要我们生成提交文件。

除了比赛数据，方案里还用了两份外部资源：FLEURS 的泰语训练集（2,602 条录音，第一次下载约 1.95 GB），以及 LST20 语料（整理后主词表有 1,050,110 条），后者不直接拿来训练音频，而是用于"检索提示词"。

### 1.3 评价标准

比赛用**词错误率（Word Error Rate，WER）**评分，越低越好。WER 的算法是：

```
WER = (替换词数 + 删除词数 + 插入词数) / 参考文本的总词数
```

直观理解就是"把模型输出改成正确答案，最少要改几个词，再除以答案长度"。完全预测对是 0%，完全乱说可能超过 100%（因为插入太多）。因为泰语要先分词再比词，所以分词器不同，WER 也会有细微差别——这也是方案里特意固定 `pythainlp` 版本的原因之一。除了 WER，我们还同时报告字符错误率 CER，它按字符比，不受分词影响，适合看"拼写层面"错得多不多。

---

## 二、解题过程

整个方案可以概括成一条流水线：**先把数据整理干净，测出预训练模型的基线，再用 LoRA 小步微调，按完整开发集的 WER 选 checkpoint，最后用词表提示做一次可选增强。** 我们按四步展开。

### 2.1 数据探索（EDA）

我们首先在 `/kaggle/input` 下扫描，找出同时包含 `train.csv`、`dev.csv`、`test.csv` 的目录。比赛环境里经常同时挂着好几个数据集，这一步是为了防止拿错文件夹：

```python
from pathlib import Path

# 找出所有"三个 CSV 都齐"的目录，作为候选数据根目录
roots = sorted({
    p.parent
    for p in Path("/kaggle/input").rglob("train.csv")
    if (p.parent / "dev.csv").is_file()
    and (p.parent / "test.csv").is_file()
})

print("找到的数据目录:", roots)

# 必须恰好有一个，否则说明 Input 没挂全，直接停下，别浪费 GPU
if len(roots) != 1:
    raise RuntimeError("请先补全比赛数据，或检查上面列出的目录")
```

确认只有一组数据后，我们进入"准备阶段"：把所有音频统一转成单声道 16 kHz，缓存成 PCM，顺便统计时长、检查重复文件、把超过 30 秒的长音频单独标出来。准备阶段结束后会输出一份汇总：

```python
import json

# 读取准备阶段的汇总：各切分保留了多少条、因为什么原因排除了多少条
prepared = json.loads(
    (PREP / 'preparation_summary.json').read_text(encoding='utf-8')
)

print('可训练条数:', prepared['train_counts'])
print('训练总数:', prepared['train_total'])
print('按原因排除:', prepared['excluded_by_reason'])
```

从材料里的设置可以看到，开发集是**完整的 240 个文件**，不是抽样。这一点很关键：ASR 的 WER 在几百条音频上波动很大，只拿几十条做验证，选出来的模型很可能只是"运气好"。

我们也检查了标签文本的 Unicode 形态。泰语有个特殊字符叫 "sara am"（ำ），它在不同输入法里可能被存成"单个字符"或"两个字符的组合"，看起来一样，但分词和比对时会出问题。所以准备阶段把标签统一成 NFC 形式，并且只修这个明确的 Unicode 问题，**不强行把阿拉伯数字改成泰文数字，也不改人名拼写**——那些属于参考文本本身，擅自改写等于篡改答案。

### 2.2 数据清洗

清洗的原则是"只做不会改变语义的规范化"。我们做了四件事：

1. **统一采样格式**：全部转成单声道、16 kHz 的 PCM。Whisper 系列模型就是按 16 kHz 单声道训练的，不统一会在输入端引入不必要的偏差。
2. **去重**：对音频内容算哈希，同一段录音如果同时出现在训练集和开发集，会造成"验证集泄漏"，必须找出来。
3. **切长音频**：超过 30 秒的文件单独处理。Whisper 的输入窗口是 30 秒，一条几分钟的音频如果不切，后面的内容会被截断。
4. **规范化标签**：NFC + 只修 sara am，其他原样保留。

这些规则集中写在一份配置里，保证"数据怎么处理"是可复现的，而不是散落在代码各处：

```python
CFG = {
    'repo': 'nectec/Pathumma-whisper-th-large-v3',   # 泰语预训练 Whisper
    'revision': '6d5e4f2fc7fc2c05447b3626dd346b1793682eb9',  # 锁定版本，避免上游更新
    'seed': 443,
    'use_fleurs': True,          # 额外使用 FLEURS 泰语训练集
    'max_steps': 100,            # 先跑 100 步的试点，不是完整收敛
    'learning_rate': 1e-5,       # LoRA 常用的小学习率
    'rank': 8,                   # LoRA 的秩，越小可训练参数越少
    'grad_accum': 8,             # 梯度累积 8 次，等效 batch = 1 × 8
    'warmup_steps': 10,
    'eval_every': 25,            # 每 25 步在完整 Dev 上评一次
    'patience': 3,               # 连续 3 次不提升就停
    'num_beams': 3,              # 解码用 3 束搜索
    'previous_text': True,       # 长音频分段解码时带上文
    'max_length': 448,
    'label_policy': 'NFC_and_Thai_sara_am_only',  # 只做最小规范化
    'peak_normalize': False,     # 不做峰值归一化，保留原始音量
}
```

这里有两个容易被忽略的细节。`revision` 锁定到具体的 commit，是为了"今天跑和下周跑得到同一个模型"；`peak_normalize=False` 是因为音量归一化会改变音频的绝对幅度，而预训练模型见过的数据分布未必做过这一步，贸然归一化反而可能变差。

### 2.3 模型训练与超参数调优

我们首先测**基线**：不训练，直接拿 `Pathumma-whisper-th-large-v3` 在完整开发集上解码，得到 WER 和 CER。这一步的意义是给后面的训练定一个"及格线"——如果微调后还不如原模型，那就说明训练配置有问题，应该回去查数据，而不是继续加步数。

```python
# 在完整 Dev（240 个文件）上评估预训练模型
run_logged(
    [sys.executable, str(RUNTIME / 'fresh_asr_worker.py'), 'baseline', str(CONFIG)],
    RUN / 'baseline.log'
)

baseline = json.loads((RUN / 'pretrained/summary.json').read_text(encoding='utf-8'))
print(f"Baseline WER: {baseline['wer']*100:.2f}% | CER: {baseline['cer']*100:.2f}%")
```

接着是微调。我们用的是 **LoRA**（Low-Rank Adaptation）。用大白话解释：Whisper large-v3 有十几亿参数，全部重训既慢又容易把原有的泰语能力"洗掉"；LoRA 的做法是在模型的部分层旁边挂一个小矩阵，只训练这个小矩阵，主干参数冻结。这样显存占用小、训练快，也更不容易过拟合。我们设置 `rank=8`、学习率 `1e-5`、FP16 混合精度、batch size 1 配 8 次梯度累积，等效 batch 是 8。

为什么只跑 100 步？因为这是一次**有边界的试点实验**：100 步 × 8 = 约 800 个样本，而且采样是"按来源平衡、带轻微 Latin/数字加权"的，同一个文件可能被重复抽到。作者在笔记里明确写了，这还没看到全部训练数据，不能当作完整收敛的结果。

训练过程中每 25 步就在完整开发集上评一次 WER，保存 WER 最低的 checkpoint，同时把"第 0 步"（也就是未训练的基座模型）也作为候选：

```python
# 训练并在 0/25/50/75/100 步各评估一次，按 Dev WER 选最优
run_logged(
    [sys.executable, str(RUNTIME / 'fresh_asr_worker.py'), 'train', str(CONFIG)],
    RUN / 'training.log'
)

selected = json.loads((RUN / 'selected_model.json').read_text(encoding='utf-8'))
print('选中的模型:', selected)

# 如果选中的是 step 0，说明基座模型更好，不需要任何 LoRA 权重
if selected['step'] == 0:
    print('基座模型胜出，本次不产生适配器')
```

这里我们特意把"不训练"也放进候选，是因为小数据微调经常出现"越训越差"的情况。与其强行提交一个训坏的模型，不如老老实实承认基座更好。

**词表提示（Vocabulary Retrieval）**是这套方案里比较特别的一步。思路是：先用第一遍解码得到一段粗略文本，从里面挑出可能的专有名词、英文词、数字，然后到 105 万条的外部词表里检索，最多取 6 个候选词，拼成不超过 96 个 token 的提示，再把提示作为 `prompt_ids` 喂给模型做第二遍解码。这相当于在解码时告诉模型："这段话里可能出现这几个词。"它不保证检索到的词就一定是说出来的词，所以最终是否启用，必须在完整开发集上比较"开"和"关"两种 WER，谁低用谁：

```python
VOCAB_SETTINGS = {
    'vocab_root': str(BASE / 'vocabulary_resources'),
    'use_lst20': True,
    'max_hint_terms': 6,      # 最多给 6 个提示词
    'max_prompt_tokens': 96,  # 提示长度上限，防止挤占音频编码
}

# 在完整 Dev 上比较"用词表"和"不用词表"的 WER
run_logged(
    [sys.executable, str(RUNTIME / 'fresh_asr_vocab.py'), 'dev', str(VOCAB_CONFIG)],
    RUN / 'vocabulary_dev.log'
)
```

超参数方面，我们只固定了一组值，没有做大规模网格搜索。原因很实际：每次在完整开发集上评估都要几十分钟，100 步试点能覆盖的组合很有限。作者也强调，这是一个"多组件配方"，无法从单轮实验里推断出某一个组件单独的贡献。

### 2.4 模型测试与指标报告

训练结束后，我们先看开发集上的 WER 曲线，再和历史上的集成方案对比。材料里给出的参照是：**旧集成方案的 Dev WER 为 12.68%，也就是 5,653 个词里错了 717 个**。这个数字是"要打败的目标"，不是本次实验的成绩：

```python
history = json.loads((RUN / 'training_dev_history.json').read_text(encoding='utf-8'))
display(pd.DataFrame(history).assign(wer_percent=lambda x: x.wer * 100))

# 如果本次 Dev WER 高于 12.68%，说明没有打赢旧集成
if selected['dev_wer'] > 717 / 5653:
    print('本次实验未超过旧集成，下面的 CSV 只能算实验候选')
```

接着我们做**逐文件误差分析**，把基线和选中模型在同一批开发文件上的词错误数放在一起比：

```python
baseline_rows = read_csv(RUN / 'pretrained/predictions.csv')
winner_rows = read_csv(winner_dir / 'predictions.csv')
lookup = {r['audio']: r for r in baseline_rows}

error_rows = []
for row in winner_rows:
    old = lookup[row['audio']]
    error_rows.append({
        'audio': row['audio'],
        'reference': row['reference'],
        'baseline_hypothesis': old['hypothesis'],
        'selected_hypothesis': row['hypothesis'],
        'baseline_word_errors': int(old['word_errors']),
        'selected_word_errors': int(row['word_errors']),
        # 负数代表这个文件变好了，正数代表变差了
        'error_change': int(row['word_errors']) - int(old['word_errors']),
    })

error_table = pd.DataFrame(error_rows).sort_values('selected_word_errors', ascending=False)
print('变好的文件数:', int((error_table.error_change < 0).sum()))
print('变差的文件数:', int((error_table.error_change > 0).sum()))
```

这张表比一个总分有用得多：如果变好的和变差的数量差不多，说明模型只是"把错误挪了个位置"，并没有真正学到东西；如果变差集中在某类音频（比如长音频、带英文术语的音频），就知道下一步该补什么数据。

最后，我们用选中的模型解码测试集，生成 `submission.csv`，并检查每一行 ID 都能对上：

```python
if RUN_TEST:
    run_logged(
        [sys.executable, str(RUNTIME / 'fresh_asr_worker.py'), 'test', str(CONFIG)],
        RUN / 'test.log'
    )
    # 把最终提交文件复制到工作目录根下，方便手动下载
    shutil.copy2(RUN / 'submission.csv', BASE / 'submission.csv')
    print('请手动下载并提交这个 CSV:', BASE / 'submission.csv')
```

这里有一个必须说清楚的限制：**开发集既用来选 checkpoint，又用来报告分数**，所以这个 WER 不是独立的泛化估计，真实测试集上的表现通常会更差一些。材料里也特意提醒了这一点。要得到无偏估计，得单独留一份从不参与选择的测试集，而本方案没有这样做。

---

## 三、总结

本文中我们完成了泰语语音识别比赛的一次完整实验。我们首先扫描数据目录，确认训练/开发/测试三份 CSV 齐全，并统计出开发集有 240 个文件；其次把音频统一成单声道 16 kHz、缓存 PCM、按哈希去重、切分超长音频，并且只对标签做 NFC 和 sara am 的最小规范化；接着以 `Pathumma-whisper-th-large-v3` 为基座测出基线，用 rank 8、学习率 1e-5 的 LoRA 跑 100 步试点，每 25 步在完整开发集上评估并按 WER 选模型，同时尝试用 105 万条外部词表给解码器提供最多 6 个提示词；最后做逐文件误差对比，与旧集成 12.68% 的开发集 WER 对照，并解码测试集生成提交文件。

这个比赛让我们体会到两件事。第一，**先把基线测出来再训练**，比盲目调参省时间得多；第二，**小数据微调不保证一定变好**，把"不训练"也放进候选、用完整开发集而不是抽样来决定去留，是对实验结果更诚实的态度。
