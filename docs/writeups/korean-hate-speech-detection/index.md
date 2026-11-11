# 韩语仇恨言论检测（Korean Hate Speech Detection）

## 一、比赛简介

### 1.1 比赛背景

网络评论里经常出现辱骂、人身攻击和针对特定群体的仇恨言论。平台需要自动识别这些内容，才能及时处理。韩语的难点在于：网络用语多、缩略语多、同一句话可能因为语气助词不同而含义完全相反，单纯靠关键词表很容易误判。

这个比赛提供了一批韩语评论，要求参赛者把它们分成三类：`none`（正常）、`offensive`（冒犯性言论）、`hate`（仇恨言论）。三类的边界并不总是清晰：`offensive` 可能只是骂人，`hate` 则往往针对某个群体。这种「相邻类别难区分」正是本场比赛的核心难点。

### 1.2 比赛目标

输入是评论正文，字段名为 `comments`。输出是三类标签之一。比赛提供的文件包括：

- `train.hate.csv`：训练集，含 `comments` 和 `label` 两列；
- `dev.hate.csv`：验证集，格式和训练集相同；
- `test.hate.no_label.csv`：测试集，只有评论没有标签。

我们需要给测试集每条评论预测一个标签，并按比赛要求生成提交文件。提交时标签要转成数字：材料里的映射是 `none→0`、`offensive→1`、`hate→2`。

### 1.3 评价标准

这是一个三分类任务，材料中的训练脚本用**准确率（accuracy）**和**宏平均 F1（macro F1）**两个指标来监控模型，并把宏平均 F1 作为挑选最佳模型的依据。宏平均 F1 的做法是：先对每个类别单独算 F1，再取平均。它和普通准确率的区别在于，每个类别权重相同，不会因为 `none` 样本多就把少数类的糟糕表现掩盖掉。分数越高越好。

## 二、解题过程

我们走两条线：先做一个从零训练的 TextCNN 基线，理解数据和任务；再对比社区里用 KoELECTRA 微调的方案，看预训练模型能带来多大提升。下面按 EDA、清洗、训练调参、测试报告四步展开。

### 2.1 数据探索（EDA）

第一步是读训练集，看标签分布和评论样子：

```python
import pandas as pd

# 读取训练集和验证集
df = pd.read_csv("/kaggle/input/korean-hate-speech-detection/train.hate.csv")
val_df = pd.read_csv("/kaggle/input/korean-hate-speech-detection/dev.hate.csv")

# 看前几行，确认 comments 和 label 两列
print(df.head())

# 统计三类各有多少条
print(df['label'].value_counts())
```

从 `value_counts` 能看出三类并不均衡：`none` 最多，`hate` 通常最少。这种不均衡会直接影响训练——如果不管它，模型会把大部分样本都猜成 `none`，准确率看起来不错，但宏平均 F1 很低。

接着把字符串标签转成数字，并切分训练/验证集：

```python
from sklearn.model_selection import train_test_split

# 三类标签映射成 0/1/2
label_map = {"none": 0, "offensive": 1, "hate": 2}
df["label_num"] = df["label"].map(label_map)

# 分层抽样：保证训练集和验证集里三类比例一致
X_train, X_test, y_train, y_test = train_test_split(
    df["comments"].values,
    df["label_num"].values,
    test_size=0.2,
    random_state=42,
    stratify=df["label_num"]
)
```

这里用 `stratify` 很关键。如果不分层，随机切分可能让本来就不多的 `hate` 样本在验证集里只剩个位数，评估结果会非常不稳定。

我们再看评论长度：

```python
# 统计每条评论的字符数，决定后面截断到多长
lengths = df['comments'].astype(str).str.len()
print(lengths.describe())
print(lengths.quantile([0.5, 0.9, 0.99]))
```

大多数评论不长，99% 分位数也在几十到一百多个字符之间。这让我们有信心把最大长度设得比较小，既省显存又不丢太多信息。

### 2.2 数据清洗

我们做了最基础的清洗：转字符串、去首尾空白。没有做形态素分析，也没有删停用词，原因后面会解释。

然后是构建词表。因为这是从零训练的基线，没有预训练分词器，我们先用空格切词：

```python
# 简单的空格分词，适合快速搭基线
def tokenize(text):
    text = str(text)
    text = text.strip()
    return text.split()

# 只用训练集建词表，防止验证集信息泄漏
word2idx = {"<pad>": 0, "<unk>": 1}
idx2word = ["<pad>", "<unk>"]

for text in X_train:
    for w in tokenize(text):
        if w not in word2idx:
            word2idx[w] = len(idx2word)
            idx2word.append(w)

vocab_size = len(word2idx)
print("vocab_size:", vocab_size)
```

词表只从训练集构建，这一点很重要。如果连验证集、测试集的词一起统计，词表里就会混入评估时才能看到的信息，相当于考试前偷看答案。

接着把每条评论转成固定长度的整数序列：

```python
max_len = 40  # 最大长度，超过截断，不足补 0

def text_to_ids(text):
    words = tokenize(text)
    ids = []
    for w in words:
        ids.append(word2idx.get(w, word2idx["<unk>"]))  # 未见过的词用 <unk>
    if len(ids) > max_len:
        ids = ids[:max_len]          # 截断
    else:
        ids += [0] * (max_len - len(ids))  # 补 pad
    return ids

X_train_ids = np.array([text_to_ids(t) for t in X_train], dtype=np.int64)
X_test_ids = np.array([text_to_ids(t) for t in X_test], dtype=np.int64)
```

这里有一个取舍：空格分词会把韩语的助词、词尾都当成独立词，词表会变得很大，而且同一个词根的不同变形会被视为不同词。这正是后面要换用 KoELECTRA 的原因——预训练模型用子词分词（subword），能自动处理这种变形。

### 2.3 模型训练与超参数调优

我们的基线是 TextCNN。它的思路很简单：用不同大小的卷积核在词序列上滑动，抓取 3 个词、4 个词、5 个词组成的局部片段，再对每个片段取最大值，最后拼起来分类。对于「骂人短语」这种局部模式，TextCNN 特别合适。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, num_classes, dropout_rate):
        super(TextCNN, self).__init__()
        # 词嵌入层，padding_idx=0 保证 pad 不参与梯度
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 三个不同卷积核，分别抓 3/4/5 个词的片段
        self.conv_1 = nn.Conv1d(embedding_dim, num_filters, kernel_sizes[0])
        self.conv_2 = nn.Conv1d(embedding_dim, num_filters, kernel_sizes[1])
        self.conv_3 = nn.Conv1d(embedding_dim, num_filters, kernel_sizes[2])
        # 三层 dropout，分别放在嵌入后、池化后、全连接前
        self.dropout_1 = nn.Dropout(p=dropout_rate[0])
        self.dropout_2 = nn.Dropout(p=dropout_rate[1])
        self.dropout_3 = nn.Dropout(p=dropout_rate[2])
        self.classifier = nn.Linear(num_filters * 3, num_classes)

    def forward(self, x):
        x = self.dropout_1(self.embedding(x))   # (batch, len, dim)
        x = x.transpose(1, 2)                    # 转成 (batch, dim, len) 给 Conv1d
        # 每个卷积核：卷积 -> ReLU -> 全局最大池化
        c1 = torch.max(F.relu(self.conv_1(x)), dim=2)[0]
        c2 = torch.max(F.relu(self.conv_2(x)), dim=2)[0]
        c3 = torch.max(F.relu(self.conv_3(x)), dim=2)[0]
        h = torch.cat([c1, c2, c3], dim=1)       # 拼接三路特征
        h = self.dropout_3(h)
        return self.classifier(h)
```

超参数上，我们设嵌入维度 100、每种卷积核 64 个、卷积核大小 `[3,4,5]`、dropout `[0.1,0.2,0.5]`、类别数 3。优化器用 Adam，学习率 0.001，训练 5 轮。

针对类别不均衡，我们给交叉熵损失加了类别权重：

```python
# 类别越少，权重越大，让模型不敢忽视少数类
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * num_classes
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

训练循环每一轮都记录训练损失、训练准确率，并在验证集上算准确率：

```python
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

    # 每轮结束后在验证集上评估
    model.eval()
    correct, count = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = torch.argmax(model(x_batch), dim=1)
            correct += (preds == y_batch).sum().item()
            count += x_batch.size(0)
    print("Epoch:", epoch + 1, "Test Acc:", round(correct / count, 4))
```

**关于调参。** 我们主要动了三个地方：一是类别权重，解决不均衡；二是最大长度 40，平衡显存和信息量；三是 dropout 分三层设置，嵌入层只要 0.1（不要过早丢掉词信息），全连接前用 0.5（防止过拟合）。进一步调卷积核数量或学习率带来的提升有限，所以我们把精力转向更强的模型。

**对比方案：KoELECTRA 微调。** TextCNN 从零训练，词表是空格切出来的，对韩语变形不友好。材料里另一个方案直接微调 `monologg/koelectra-base-v3-discriminator`：用预训练分词器把评论切成子词，最大长度设为 64，加载 `ElectraForSequenceClassification`，训练 5 轮、batch size 64、开启 fp16，并用宏平均 F1 挑选最佳轮次。这个方案站在预训练模型已经学过的韩语知识上，通常比从零训练的小网络更稳。

关键的训练配置用 `TrainingArguments` 一次说明：

```python
from transformers import ElectraForSequenceClassification, Trainer, TrainingArguments

# 加载多标签的分类头（本案例里做了二值化：hate/offensive -> 1，none -> 0）
model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,               # 训练 5 轮
    per_device_train_batch_size=64,   # T4 显卡能容纳的批量大小
    fp16=True,                        # 半精度训练，速度约提升一倍
    eval_strategy="epoch",            # 每个 epoch 结束后在验证集上评估
    load_best_model_at_end=True,      # 训练结束自动恢复效果最好的那个
    metric_for_best_model="f1",       # 用 F1 而非 loss 挑模型
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)
trainer.train()
```

这里两个设置值得注意。一是开启 `fp16` 半精度训练：T4 这类推理显卡的加速靠的是张量核心，半精度能明显缩短训练时间；二是 `load_best_model_at_end` 配合 `metric_for_best_model="f1"`，保证我们拿到的是整个训练过程中 F1 最高的状态，而不是最后一个 epoch 的状态——后者往往已经开始过拟合。

`compute_metrics` 里同时算准确率和宏平均 F1：

```python
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')   # 不均衡场景看宏平均 F1
    return {'accuracy': acc, 'f1': f1}
```

把宏平均 F1 写进 `compute_metrics` 而不是只看准确率，是因为类别不均衡时准确率会骗人：即使少数类全猜错，只要 `none` 类猜对得多，准确率还是很高。

**快速基线：直接调用现成模型。** 材料里还有一个更省事的做法：用 `transformers` 的 `pipeline` 加载 `beomi/beep-KcELECTRA-base-hate`，对测试集逐条推理，再把字符串标签映射成数字提交。它不需要训练，适合先拿到一个能提交的结果，再慢慢优化。

### 2.4 模型测试与指标报告

验证集评估我们用了 `classification_report` 和混淆矩阵，这样能看到每一类的精确率、召回率和 F1，而不是只有一个总数：

```python
from sklearn.metrics import classification_report, confusion_matrix

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        logits = model(x_batch.to(device))
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(y_batch.numpy())

# 每一类的 precision / recall / f1 都能看到
print(classification_report(all_labels, all_preds, digits=4))
# 混淆矩阵能看出哪两类最容易混
print(confusion_matrix(all_labels, all_preds))
```

我们重点看混淆矩阵的对角线以外。经验上，`offensive` 和 `hate` 之间的混淆最多，因为两者都带攻击性词汇，区别主要在「是否针对群体」。如果 `hate` 的召回率明显偏低，就说明模型倾向于把它降级成 `offensive`，这时可以提高 `hate` 的类别权重。

材料里的训练脚本没有给出最终的准确率或 F1 数值，所以我们在这里不编造具体分数。能确认的是评估流程：用宏平均 F1 选最佳模型，同时报告准确率，并用分类报告和混淆矩阵做逐类分析。

如果采用 `pipeline` 方案，提交前要做标签映射：

```python
LABEL_DIC = {'none': 0, 'offensive': 1, 'hate': 2}

# 把模型输出的字符串标签转成比赛要求的数字
df['label'] = df['label'].map(lambda x: LABEL_DIC[x])

# 必须检查映射失败的行，NaN 会让提交直接失效
print(df['label'].isna().sum())

df.to_csv('./submission.csv', index=None)
```

材料里的 KoELECTRA 微调方案还额外做了可解释性分析：把最后一层注意力权重取出来，按词归一化后画成热力图，看模型把注意力放在哪些词上。比如输入一句带辱骂的评论，热力图会高亮骂人的词；输入一句普通问候，注意力则比较分散。这一步不直接提分，但能帮我们判断模型是不是学到了「关键词匹配」而不是真正的语义。

## 三、总结

本文中我们完成了韩语仇恨言论检测的完整流程。我们首先通过 EDA 确认三类标签不均衡、评论普遍较短，并用分层抽样切分训练验证集。其次做了最基础的清洗，只用训练集构建词表，把评论转成固定长度序列。接着用 TextCNN 建立从零训练的基线，通过三路卷积抓取局部辱骂模式，并用类别加权交叉熵缓解不均衡。然后对比了 KoELECTRA 微调方案和现成模型直接推理方案，前者用子词分词和预训练知识，通常效果更好。最后用宏平均 F1 选模型，并用分类报告和混淆矩阵逐类分析。

这个比赛给我们的最大启发是：**类别不均衡时，看总数会骗人**。准确率很高不代表模型好，必须拆到每一类去看 F1。另外，韩语这类形态丰富的语言，子词分词比空格分词更合适，这也是预训练模型普遍强于从零训练小网络的原因。把这两点结合起来，就是文本分类比赛里最实用的方法论：选对分词与预训练模型，再认真看每一类的指标。
