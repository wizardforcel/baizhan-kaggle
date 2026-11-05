# Pharma-Semantic Challenge - Datahack 3.0

## 一、比赛简介

### 1.1 比赛背景

药物相互作用是临床用药安全中非常关键的一环。当病人同时服用两种或多种药物时，一种药物可能会影响另一种药物在体内的吸收、代谢或排泄，从而让药效变强变弱，甚至引发不良反应。举例来说，如果药物 A 抑制了肝脏里负责代谢的另一酶的活性，药物 B 就无法被正常代谢，血液里的浓度可能因此升高，带来毒性风险。这些知识一般被整理成结构化的数据库，但真正给医生或者病人看的时候，需要用自然语言把机制说清楚，比如「当药物 B 与药物 A 合用时，其血清浓度可能升高」。

Datahack 3.0 举办的 Pharma-Semantic Challenge 正是围绕这个需求展开的：它把药物相互作用描述的生成本身当作一个机器学习任务，让参赛者根据两味药物的属性描述，自动生成一句通顺、准确的相互作用说明。这属于医药领域的自然语言生成问题，核心是把两段结构化的药物属性文本融合起来，输出一段人类可读的句子，辅助临床决策与用药知识普及。

这场比赛所依赖的深度学习技术是序列到序列模型，也就是 seq2seq。这类模型一端读入完整的输入文本，另一端逐步输出目标文本，中间的编码器把输入「理解」为向量，解码器再根据这个向量逐词生成句子。谷歌提出的 T5 模型把几乎所有自然语言任务都统一成了「输入一段文字，输出一段文字」的格式，非常适合这种任务。材料给出的演示使用 T5-small 这个相对小巧的版本，参数量不大，运行速度快，对入门者友好。

### 1.2 比赛目标

从材料代码可以还原出比赛的数据形状。训练集包含五个字段：`ID_1`、`ID_2`、`Input_Context_1`、`Input_Context_2` 和 `Interaction Description`。前两个字段是两种药物的编号，`Input_Context_1` 描述第一种药物的特性，比如它是哪种代谢酶的抑制剂、会不会影响心率、是否与血浆蛋白结合等；`Input_Context_2` 描述第二种药物的特性；`Interaction Description` 则是这两味药联用时相互作用的标准描述句子，也就是我们要求模型生成的答案。

测试集只有前四个字段，我们需要为每一对药物生成对应的 `Interaction Description`。提交文件里，每一行包含两个字段：`id` 和 `Interaction Description`。`id` 的规则是从材料代码里看出来的，需要把 `ID_1` 和 `ID_2` 用下划线拼接起来，例如 `DH_005_DH_006`。简单概括就是：比赛的输入是两段药物属性文本，输出是一段相互作用描述，任务本质是一个受控的文本生成问题。

材料为了演示方便，直接在脚本里创建了自包含的演示数据，避免依赖外部 CSV 文件，这样任何人都能一键复现完整流程。演示数据只有几条样本，覆盖了酶抑制、心率变化、蛋白结合三种常见场景，虽然规模小，但把输入输出的格式展示得非常清楚。

```python
# 生成演示用训练数据，方便在无外部数据时跑通流程
train_data = {
    'ID_1': ['DH_001', 'DH_003', 'DH_009'],
    'ID_2': ['DH_002', 'DH_004', 'DH_010'],
    'Input_Context_1': ['[DRUG_A] is a CYP3A4 inhibitor.', '[DRUG_A] increases heart rate.', '[DRUG_A] binds to plasma proteins.'],
    'Input_Context_2': ['[DRUG_B] is metabolized by CYP3A4.', '[DRUG_B] is a beta-blocker.', '[DRUG_B] is highly protein bound.'],
    'Interaction Description': [
        'The serum concentration of [DRUG_B] can be increased when it is combined with [DRUG_A].',
        'The therapeutic efficacy of [DRUG_B] can be decreased when used in combination with [DRUG_A].',
        '[DRUG_A] may decrease the excretion rate of [DRUG_B].'
    ]
}
train_df = pd.DataFrame(train_data)
```

从上面的字典可以看到，`[DRUG_A]` 和 `[DRUG_B]` 是占位符，真实比赛里它们会被具体的药物名替换。材料用占位符的好处是既保留了句子的语法结构，又不涉及真实药物数据，安全且易于展示。

### 1.3 评价标准

材料笔记里没有给出官方评分脚本和具体指标，因此我们在本地不编造分数，只做流程复现。从任务形态判断，这是一个文本生成类竞赛，官方一般会用文本重合度或者语义相似度来衡量生成句子与参考句子的接近程度。常见的一类指标是 BLEU，它统计生成句子里的 n 元词组（也就是连续的 n 个词）有多少出现在参考句子里，重合得越多分越高，是机器翻译领域最常用的指标之一。另一类常用的指标是 ROUGE，它更看重参考句子里有多少内容被生成句子保留下来，常用于摘要任务。此外，有的生成竞赛还会用基于句向量的语义相似度打分，即使词语不完全相同，只要语义接近也能得高分。

一般来说，这类指标都是分数越高越好，满分接近 1.0。我们在本地可以做的替代做法是：在训练集里留出一小部分作为验证集，用 BLEU 或 ROUGE 自己算一个参考分数，来判断模型改进的方向。材料里的演示代码没有做这一步，所以下面我们严格按材料里的流程来写，不额外编造任何数字。

## 二、解题过程

整场比赛的思路可以概括成一句话：把两段药物属性文本融合成一段输入，用 T5 做条件文本生成，训练目标是相互作用描述。我们严格按材料里的代码复现整个流程，并分成四步来叙述：数据探索、数据清洗、模型训练与超参数调优、模型测试与指标报告。

### 2.1 数据探索（EDA）

我们首先加载了常用的数值计算和数据处理库，并遍历 Kaggle 环境的输入目录，看看有没有现成的数据文件可以使用。

```python
# 导入数值计算和数据处理的库
import numpy as np
import pandas as pd

# 遍历 /kaggle/input 目录，打印所有文件路径，确认数据位置
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

在真实比赛里，这个目录里会有训练集、测试集和提交样例文件，遍历目录可以帮助我们确认文件名是否与预期一致。材料为了让笔记本能够脱离外部数据独立运行，直接在脚本里构造了演示数据。我们把演示数据转成 pandas 的 DataFrame 对象，便于用统一的接口操作。

```python
# 训练用演示数据：两段上下文 + 一条目标句子
train_data = {
    'ID_1': ['DH_001', 'DH_003', 'DH_009'],
    'ID_2': ['DH_002', 'DH_004', 'DH_010'],
    'Input_Context_1': ['[DRUG_A] is a CYP3A4 inhibitor.', '[DRUG_A] increases heart rate.', '[DRUG_A] binds to plasma proteins.'],
    'Input_Context_2': ['[DRUG_B] is metabolized by CYP3A4.', '[DRUG_B] is a beta-blocker.', '[DRUG_B] is highly protein bound.'],
    'Interaction Description': [
        'The serum concentration of [DRUG_B] can be increased when it is combined with [DRUG_A].',
        'The therapeutic efficacy of [DRUG_B] can be decreased when used in combination with [DRUG_A].',
        '[DRUG_A] may decrease the excretion rate of [DRUG_B].'
    ]
}
train_df = pd.DataFrame(train_data)

# 测试用演示数据：只有两段上下文，没有目标句子
test_data = {
    'ID_1': ['DH_005', 'DH_007'],
    'ID_2': ['DH_006', 'DH_008'],
    'Input_Context_1': ['[DRUG_A] is a strong CYP2D6 inhibitor.', '[DRUG_A] reduces blood pressure.'],
    'Input_Context_2': ['[DRUG_B] relies on CYP2D6 for activation.', '[DRUG_B] also reduces blood pressure.']
}
test_df = pd.DataFrame(test_data)
```

通过观察数据，我们可以确认几点：第一，训练集有 3 条样本、测试集有 2 条样本，规模只够演示。第二，`Input_Context_1` 和 `Input_Context_2` 描述的确实是两味不同药物的特性，比如第一条里药物 A 是 CYP3A4 抑制剂，药物 B 由 CYP3A4 代谢，两者正好对应「抑制代谢酶 → 血清浓度升高」的关系。第三，目标句子总是以「The serum concentration of ... can be ...」这类固定句式出现，说明这是一个控制比较强的生成任务，模型只要能学会关键机制和措辞，就很容易生成合理答案。

### 2.2 数据清洗

其次，我们把两段上下文融合成单条输入字符串。为什么不直接分别送入模型？因为 T5 这类模型通常把整段输入当成一个整体来编码，如果拆成两段再拼接，很难用第二个分隔符以外的机制表示「这两段分别属于两种药物」。材料采用的做法是在字符串前面加上 `Drug A properties:` 和 `Drug B properties:` 这样的前缀，中间用竖线分隔，这样模型从格式上就能区分哪段对应哪味药。

```python
# 把两段上下文合并成单条字符串，方便模型读取
train_df['Merged_Context'] = "Drug A properties: " + train_df['Input_Context_1'] + " | Drug B properties: " + train_df['Input_Context_2']
test_df['Merged_Context'] = "Drug A properties: " + test_df['Input_Context_1'] + " | Drug B properties: " + test_df['Input_Context_2']

# 按比赛规则生成测试集 id：用下划线拼接两个药物编号
test_df['id'] = test_df['ID_1'].astype(str) + "_" + test_df['ID_2'].astype(str)
```

这里我们没有做去停用词、去标点、转小写等常规文本清洗，原因是这些符号和大小写在语义生成任务里同样承载信息，贸然删除会伤害模型对句子结构的理解。生成任务与分类任务不同，分类任务关心的是判别性特征，而生成任务需要尽量保留原文信息，才能产出流畅的句子。所以我们只做了「合并上下文」和「生成 id」两步，其余信息原样保留。

### 2.3 模型训练与超参数调优

接着，我们准备模型。材料使用的是 `t5-small`，它是 T5 家族里最小的版本。T5 是一个预训练模型，它预先在大规模语料上学过「把输入转成输出」的通用能力，我们在此基础上做微调，也就是用比赛数据再训练一小段时间，让模型掌握药物相互作用领域的表达方式。这样做比从零训练又快又稳，是迁移学习的典型应用。

```python
# 导入 PyTorch 和 Hugging Face Transformers
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# 指定使用 T5 的小模型版本
MODEL_NAME = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
```

关键的一步是扩展词表。`[DRUG_A]` 和 `[DRUG_B]` 这两个占位符虽然写法像「词」，但 T5 的分词器是按照子词切分的，如果不对它们做特殊处理，分词器可能会把 `[`、`DRUG`、`_`、`A`、`]` 拆成好几个片段，导致模型无法把它们当作一个整体来理解。材料里把它们加入特殊词表，并让模型重新调整词嵌入矩阵的大小。

```python
# 把两个药物占位符注册为特殊词，避免被分词器切碎
special_tokens_dict = {'additional_special_tokens': ['[DRUG_A]', '[DRUG_B]']}
tokenizer.add_special_tokens(special_tokens_dict)

# 词表变大后，模型最末层的词向量矩阵也要跟着变大
model.resize_token_embeddings(len(tokenizer))
```

接下来我们把数据转换成 Hugging Face 的数据集格式，并写一个预处理函数。预处理函数里做两件事：一是给输入加上 `predict interaction:` 这样一个任务前缀，告诉模型我们的任务类型；二是把输入和目标句子都转成 token 序列（也就是模型认识的数字编码）。输入最长限制在 128 个 token，目标最长限制在 64 个 token，超过的部分截断，防止显存浪费。

```python
# 将 DataFrame 转成 Hugging Face Dataset 格式
train_dataset = Dataset.from_pandas(train_df[['Merged_Context', 'Interaction Description']])
test_dataset = Dataset.from_pandas(test_df[['id', 'Merged_Context']])

# 输入与目标的最大 token 长度
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 64

def preprocess_function(examples):
    # 给每个输入加上任务前缀
    inputs = ["predict interaction: " + str(doc) for doc in examples['Merged_Context']]
    # 分词：超过长度截断
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    # 目标句子同样转成 token，作为训练标签
    labels = tokenizer(text_target=[str(t) for t in examples['Interaction Description']], max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 对训练数据集批量执行预处理，得到可以直接训练的样本
tokenized_train = train_dataset.map(preprocess_function, batched=True)
```

预处理完成后，我们配置训练参数并创建 Trainer。材料里学习率用了 2e-5，这是一个 T5 微调常用的取值；每块 GPU 的批大小只有 2，因为演示机器显存有限且样本极少；只训练 1 个 epoch，保证「立刻跑完」。`predict_with_generate=True` 表示评测时用真正的解码方式生成文本，而不是简单的分类。

```python
# 训练参数：输出目录、学习率、批大小、epoch 数
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    predict_with_generate=True,
    report_to="none"
)

# 数据整理器：自动给同一批内的样本做 padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    data_collator=data_collator
)

print("Starting training on dummy data...")
trainer.train()
```

关于超参数调优，材料里并没有做网格搜索，因为演示数据本来就只够把流程跑通。真实比赛里，我们会把训练集切出一部分当验证集，用 BLEU 或 ROUGE 评估，然后对学习率、batch 大小、epoch 数、束搜索宽度等参数做小范围搜索。需要提醒的是，医药文本的生成错误可能带来安全风险，所以宁可保守地多评估几轮，也不要盲目调大学习率追求速度。

### 2.4 模型测试与指标报告

最后，我们用训练好的模型对测试集逐条生成描述。推理时先判断是否有 GPU 可用，把模型搬到对应设备上，然后进入评估模式。评估模式下模型关闭了训练相关的随机性，行为更稳定，适合推理。

```python
print("Generating predictions...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

predictions = []
for context in test_df['Merged_Context']:
    inputs = "predict interaction: " + str(context)
    # 把单条文本转成 token，并放到模型所在的设备上
    input_ids = tokenizer(inputs, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).input_ids.to(device)
    # 生成参数：最多 64 个新词，束搜索宽度 2，满足终止条件就尽早停止
    outputs = model.generate(input_ids, max_new_tokens=MAX_TARGET_LENGTH, num_beams=2, early_stopping=True)
    # 把输出的 token 解码回文本，跳过特殊词
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(pred_text)
```

这里我们用到了束搜索（beam search）：每一步不只保留得分最高的一个候选，而是同时维护多个候选，生成结束时再从中挑整体最优的句子。束宽度设为 2，属于比较轻量的配置，速度和质量都比较平衡。`early_stopping=True` 能让模型在遇到结束符号时提前停止生成，而不是非要生成满 64 个词。

生成结束后，我们按比赛要求组织成提交文件：一列是 `id`，一列是 `Interaction Description`，然后保存为 CSV。`index=False` 表示不写入多余的行号，保证格式和比赛模板一致。

```python
# 组织成提交的 DataFrame
submission_df = pd.DataFrame({
    "id": test_df["id"],
    "Interaction Description": predictions
})

# 保存为 CSV 提交文件
submission_df.to_csv("submission.csv", index=False)
print("Saved submission.csv successfully!")
print(submission_df.head())
```

材料里没有给出量化的指标报告，只打印了提交文件的前几行。我们可以在本地对生成的句子做人工对照：看模型是否保留了药物名称 `[DRUG_A]`、`[DRUG_B]`，是否抓住了「抑制代谢酶」、「影响排泄」这些关键机制，句式是否完整。真实比赛中，我们会用验证集计算 BLEU 或 ROUGE 分数，作为调参的依据。这里我们严格依据材料，不编造任何分数。

## 三、总结

本文中我们基于 Kaggle 上公开的 Pharma-Semantic Challenge - Datahack 3.0 演示笔记，完整复现了药物相互作用语义生成的端到端流程。我们首先用 EDA 查看了数据格式，确认训练集包含两段药物属性上下文和一句标准相互作用描述，测试集需要为每对药物生成描述。其次我们做了最小的数据清洗，把两段上下文融合成带前缀的单条输入，并按比赛规则生成提交用的 id。接着我们使用 T5-small 作为序列到序列模型，扩展特殊药物占位符词表，编写预处理函数，以 2e-5 的学习率微调了 1 个 epoch。最后我们用束搜索对测试集逐条生成描述，整理成 `id` 与 `Interaction Description` 两列的提交文件。

材料提供的只是演示基线，我们额外讨论了真实比赛中常见的改进方向：用验证集计算 BLEU/ROUGE、对超参数做网格搜索、换更大的预训练模型、增加训练数据等。这个比赛给我们的启发是，文本生成任务的完整流程可以简洁地概括为「数据融合、序列到序列微调、生成与提交」三步，理解每一步在做什么，比堆砌复杂的技巧更重要。