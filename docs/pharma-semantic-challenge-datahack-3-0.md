# Pharma-Semantic Challenge - Datahack 3.0

## 一、比赛简介

### 1.1 比赛背景
药物相互作用是临床用药安全的重要环节，医生和药师需要快速理解两味药物联用时可能出现的疗效变化或副作用。传统的相互作用知识库条目是自然语言描述，由药物代谢酶、蛋白结合、作用机制等多个属性信息推导而来。Datahack 3.0 举办的 Pharma-Semantic Challenge 把任务抽象为语义生成：给定两味药物的属性描述，模型需要生成一段通顺、准确的相互作用说明。这类任务属于医药自然语言生成，目标是把结构化的药物属性转成人类可读的解释，辅助决策和知识普及。材料里提供了一个自包含的演示实现，用 T5 小模型展示端到端流程。

### 1.2 比赛目标
从代码材料来看，训练集包含以下字段：`ID_1`、`ID_2`、`Input_Context_1`、`Input_Context_2`、`Interaction Description`。其中 `Input_Context_1` 和 `Input_Context_2` 分别描述两味药物的药理特性，如代谢酶抑制、蛋白结合等，`Interaction Description` 是需要生成的目标句子。测试集只有 `ID_1`、`ID_2`、`Input_Context_1`、`Input_Context_2`，参赛者要为每组药物对生成 `Interaction Description`，并按规则把 `ID_1` 与 `ID_2` 用下划线拼接为提交 `id`。输出文件需要两列：`id` 与 `Interaction Description`。

材料中为了让笔记本自包含，先创建了演示数据：
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
我们可以看到训练演示数据只有 3 条样本，分别对应抑制酶、影响心率、蛋白结合三种场景，便于入门者快速复现。

### 1.3 评价标准
材料笔记中没有给出官方评分脚本和具体指标。任务形态上属于序列到序列的自然语言生成，因此在实际比赛中通常会用 BLEU、ROUGE 等文本重合度指标，或基于语义相似度的嵌入距离来衡量生成描述与参考描述的贴近程度。演示代码里只做到生成并保存提交文件，没有本地量化报告。我们在写这篇笔记时也不编造分数，仅按材料中可复现的流程进行重述。

## 二、解题过程
整体思路是把两段药物上下文融合成一段输入，用 T5 做条件的文本生成，训练时把生成目标设为相互作用描述。代码材料把流程拆成环境准备、数据融合、模型准备、训练与推理四步，我们严格按此复现。

### 2.1 数据探索（EDA）
我们首先查看 Kaggle 环境里的输入目录，确认是否有外部数据，并导入分析库。
```python
# 导入数值计算和数据处理库
import numpy as np
import pandas as pd

# 打印 /kaggle/input 下的文件列表，确认数据位置
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
材料里为了自包含，直接在脚本里创建演示数据。接下来我们把演示数据转成 DataFrame 并预览结构。
```python
# 训练数据示例
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

# 测试演示数据
test_data = {
    'ID_1': ['DH_005', 'DH_007'],
    'ID_2': ['DH_006', 'DH_008'],
    'Input_Context_1': ['[DRUG_A] is a strong CYP2D6 inhibitor.', '[DRUG_A] reduces blood pressure.'],
    'Input_Context_2': ['[DRUG_B] relies on CYP2D6 for activation.', '[DRUG_B] also reduces blood pressure.']
}
test_df = pd.DataFrame(test_data)
```
我们看到训练集有 3 行、测试集有 2 行，两列上下文描述了药物属性，训练目标是自然语言的相互作用句子。数据规模仅用于演示，真实比赛中样本量会大很多。

### 2.2 数据清洗
我们其次把两段上下文融合成一条模型输入，同时按比赛规则生成提交用的 id。
```python
# 把两段上下文拼成单条字符串，方便模型读取
train_df['Merged_Context'] = "Drug A properties: " + train_df['Input_Context_1'] + " | Drug B properties: " + train_df['Input_Context_2']
test_df['Merged_Context'] = "Drug A properties: " + test_df['Input_Context_1'] + " | Drug B properties: " + test_df['Input_Context_2']

# 按规则生成测试集 id：ID_1 + "_" + ID_2
test_df['id'] = test_df['ID_1'].astype(str) + "_" + test_df['ID_2'].astype(str)
```
这样清洗后，模型只需要读取 `Merged_Context` 作为输入，目标是 `Interaction Description`。我们没有做去停用词或大小写转换，保持原文信息完整，这对药物名称和酶名至关重要。

### 2.3 模型训练与超参数调优
接着我们准备序列到序列模型并进行词表扩展。
```python
# 导入 PyTorch 和 Hugging Face 工具
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# 用 T5-small 作为基础模型，模型名字固定
MODEL_NAME = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 材料里把 [DRUG_A]、[DRUG_B] 设为特殊词，避免被分词器拆散
special_tokens_dict = {'additional_special_tokens': ['[DRUG_A]', '[DRUG_B]']}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
```
词表扩展后，我们把 DataFrame 转成 Hugging Face Dataset 并定义分词函数。
```python
# 转成 Hugging Face Dataset 格式
train_dataset = Dataset.from_pandas(train_df[['Merged_Context', 'Interaction Description']])
test_dataset = Dataset.from_pandas(test_df[['id', 'Merged_Context']])

# 输入和目标的最大长度设置
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 64

def preprocess_function(examples):
    # 前缀提示模型这是预测相互作用的任务
    inputs = ["predict interaction: " + str(doc) for doc in examples['Merged_Context']]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    # 把目标句子也分词
    labels = tokenizer(text_target=[str(t) for t in examples['Interaction Description']], max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```
数据分词完成后，我们配置训练参数并实例化 Trainer。材料里采用 1 个 epoch、极小的 batch size，保证演示能快速跑完。
```python
# 分词后的训练集
tokenized_train = train_dataset.map(preprocess_function, batched=True)

# 训练参数：学习率 2e-5，batch size 2，只跑 1 个 epoch
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    predict_with_generate=True,
    report_to="none"
)

# 数据整理器，自动填充 padding
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
我们没有做超参数网格搜索，材料的目的就是演示端到端流程。真实场景下会尝试更大的模型、更多 epoch、调整学习率，并在验证集上用 ROUGE/BLEU 评估。

### 2.4 模型测试与指标报告
然后我们用训练好的模型对测试集生成描述，并按比赛格式保存提交文件。
```python
print("Generating predictions...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

predictions = []
for context in test_df['Merged_Context']:
    inputs = "predict interaction: " + str(context)
    input_ids = tokenizer(inputs, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).input_ids.to(device)
    # 生成参数：最大新词数 64，束搜索宽度 2，早停止
    outputs = model.generate(input_ids, max_new_tokens=MAX_TARGET_LENGTH, num_beams=2, early_stopping=True)
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(pred_text)
```
生成结果后，我们组织成提交 DataFrame。
```python
# 按比赛要求创建提交文件：id 与 Interaction Description
submission_df = pd.DataFrame({
    "id": test_df["id"],
    "Interaction Description": predictions
})

submission_df.to_csv("submission.csv", index=False)
print("Saved submission.csv successfully!")
print(submission_df.head())
```
材料里没有给出定量指标报告，仅打印提交文件的前几行。我们可以把生成的文本与参考文本人工对照，检查是否保留了药物名称、作用方向等关键信息。真实比赛中会用独立验证集计算 BLEU/ROUGE，并在排行榜上展示分数。

## 三、总结
本文中我们基于 Kaggle 上公开的 Pharma-Semantic Challenge - Datahack 3.0 演示笔记，复现了药物相互作用语义生成的端到端流程。我们首先用 EDA 查看了自包含的演示数据结构，确认输入是两段药物属性上下文，输出是自然语言相互作用描述。其次我们做了最小的数据清洗，把两段上下文融合成单条输入并按规则生成提交 id。接着我们用 T5-small 作为序列到序列模型，扩展特殊药物标记，进行分词、训练参数配置和 1 epoch 的快速训练。最后我们对测试集生成预测句子并保存为 submission.csv。整个实现强调可复现性与清晰注释，适合编程和深度学习入门者理解医药文本生成任务的典型流程。
