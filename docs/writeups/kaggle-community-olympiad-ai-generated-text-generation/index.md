# Kaggle Community Olympiad AI Generated Text Generation

## 一、比赛简介

### 1.1 比赛背景

AI 生成文本在日常应用中越来越常见，区分人类写的文本和模型生成的文本变得重要。Kaggle Community Olympiad 系列比赛提供了大量与 AI 社区相关的数据集，本比赛聚焦在 AI 生成文本的辨别上。材料中用户提交的代码显示，训练数据包含一段文本和对应的标签，目标是判断这段文本是人写的还是 AI 生成的。

### 1.2 比赛目标

从材料代码中可以看出数据格式：

- 训练集 `train.csv` 包含 `text` 列和 `label` 列。`text` 是待判断的文本，`label` 是对应的类别。我们在代码里用 `train_data["text"]` 和 `train_data["label"]` 取出这两列。
- 测试集 `test.csv` 包含 `id` 列和 `text` 列，没有标签，需要我们预测出 `label`。

比赛目标是：给定一段文本，输出 0 或 1 的标签，判断它是人类生成还是 AI 生成。提交文件需要包含 `id` 和 `label` 两列。

### 1.3 评价标准

材料中没有明确写出比赛使用的评价指标。我们从代码实现做最小合理推断：模型使用 `nn.BCEWithLogitsLoss` 训练，预测时用 `torch.sigmoid` 后阈值 0.5 判别为 0 或 1，输出的是硬标签。因此比赛是二分类任务，评价应围绕分类正确率或相关二分类指标展开，具体指标名称需要以官方说明为准。

---

## 二、解题过程

整场比赛思路可以概括为：先把文本用预训练语言模型编码，再用简单的全连接网络做二分类。我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步是把训练集和测试集读进来，确认列名和基本格式：

```python
import numpy as np
import pandas as pd
import os

TRAIN_PATH: str = "/kaggle/input/competitions/kaggle-community-olympiad-ai-generated-text-generation/train.csv"
TEST_PATH: str = "/kaggle/input/competitions/kaggle-community-olympiad-ai-generated-text-generation/test.csv"

train_data = pd.read_csv(TRAIN_PATH)  # 读取训练数据
test_data = pd.read_csv(TEST_PATH)    # 读取测试数据

train_data.head()  # 查看前几行，确认有 text 和 label 两列
```

我们首先读取两个 CSV 文件，用 `head()` 看样例确认列名正确。材料中代码后续使用了 `train_data["label"]` 做分层抽样，说明标签列存在且为数值型，`train_data["text"]` 为字符串文本。测试集只有 `id` 和 `text`，符合提交要求。

### 2.2 数据清洗

在材料中没有进行复杂的字符清洗或特征工程，清洗主要体现在数据读取和分块时的数据准备：

```python
from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(
    [x for x in range(len(train_data))],
    train_size = 0.8,
    shuffle = True,
    random_state = 14,
    stratify = train_data["label"]  # 按标签分层，保持比例一致
)

texts = train_data["text"].tolist()
targets = train_data["label"].tolist()
```

我们其次把样本索引按 0.8 训练 0.2 验证拆分，并用 `stratify` 保证验证集里人类和 AI 文本的比例与整体一致，避免类别失衡。文本本身没有做去停用词或小写化，因为后续使用预训练分词器，会在分词阶段自动处理。

在数据加载类里，文本会统一截断或补齐到 128 长度：

```python
class DataStream(Dataset):
    def __getitem__(self, index: int):
        text = self.texts[index]
        target = self.targets[index]
        inputs = self.tokenizer(text = text, max_length = 128,
                                padding = "max_length", truncation = True,
                                return_tensors = "pt")
        inputs = { k: v.squeeze(0) for k, v in inputs.items()}
        return inputs, float(target)
```

这是数据清洗的核心：对过长的文本做截断，对过短的文本做填充，保证所有输入长度一致，方便批量训练。

### 2.3 模型训练与超参数调优

我们采用 `microsoft/deberta-v3-small` 作为 backbone，使用 Transformer 提取文本语义，再接一个全连接头做二分类。

模型结构如下：

```python
from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn

model_id = "microsoft/deberta-v3-small"
backbone = AutoModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

class Model(nn.Module):
    def __init__(self, backbone = backbone):
        super().__init__()
        self.backbone = backbone
        hidden_size = self.backbone.config.hidden_size
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size),                 # 归一化稳定训练
            nn.Linear(hidden_size, hidden_size*4),     # 扩增特征
            nn.SiLU(inplace = True),                   # 激活函数
            nn.Linear(hidden_size*4, 1)                # 输出单个数值
        )
    def forward(self, data):
        hidden_states = self.backbone(**data).last_hidden_state
        B, C, H = hidden_states.shape
        mask = data["attention_mask"].view(B, C, 1).expand(B, C, H)
        embedding = (hidden_states * mask).sum(dim = 1) / mask.sum(dim = 1)  # 均值池化
        return self.fc(embedding).flatten()
```

我们接着使用不同的学习率训练 backbone 和分类头：backbone 用 1e-5 慢速微调，分类头用 1e-3 快速学习，配合 weight decay 防止过拟合。

```python
n_epochs = 5
model = Model()
param_groups = [
    {"params": model.backbone.parameters(), "lr": 1e-5, "weight_decay": 0.0},
    {"params": model.fc.parameters(), "lr": 1e-3, "weight_decay": 1e-3},
]
optimizer = torch.optim.AdamW(param_groups)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = n_epochs, eta_min = 1e-5)
```

训练时使用二分类交叉熵损失 `BCEWithLogitsLoss`，并用 `CosineAnnealingLR` 让学习率平滑下降。材料中调用 `model_trainer.train` 进行训练，保存最佳 checkpoint 到 `/kaggle/working/TrainingSummary/best_model.pth`。

### 2.4 模型测试与指标报告

训练完成后，我们加载最佳模型对测试集预测：

```python
DEVICE = "cuda"
checkpoint = torch.load("/kaggle/working/TrainingSummary/best_model.pth")
eval_model = Model()
eval_model.load_state_dict(checkpoint["model_state_dict"])
eval_model.eval()
eval_model.to(DEVICE)

TestStream = DataStream(texts = test_data["text"],
                        targets = [0 for _ in range(len(test_data))],
                        indices = [x for x in range(len(test_data))])
```

预测时对每个 batch 做 sigmoid 变换并用 0.5 阈值二值化：

```python
preds = []
for data, _ in tqdm(TestLoader):
    data = {k: v.to(DEVICE) for k, v in data.items()}
    with torch.inference_mode():
        outputs = (torch.sigmoid(eval_model(data)) > 0.5).long()
    preds.extend(outputs.tolist())
```

最后生成提交文件：

```python
submission_1 = pd.DataFrame({
    "id": test_data["id"],
    "label": preds
})
submission_1.to_csv("submission.csv", index = False)
```

我们最后检查 `submission.csv` 的列名和行数是否与测试集一致，确保格式正确。材料中没有给出具体的验证集指标数值，因此这里只展示预测流程，实际效果需要在本地验证集上计算准确率或 ROC AUC 等指标来评估。

---

## 三、总结

本文中我们完成了 Kaggle Community Olympiad AI Generated Text Generation 的 baseline 流程。我们首先读取了训练和测试数据，确认文本和标签结构。其次我们做分层抽样划分验证集，并通过分词器的 padding 与 truncation 保证输入长度统一。接着我们用 DeBERTa-v3-small 作为特征提取器，配合 LayerNorm、线性层和 SiLU 构成的简单分类头进行微调，采用不同的学习率策略训练 5 个 epoch。最后我们加载最佳模型对测试集进行推理，用 sigmoid 阈值 0.5 生成硬标签并输出提交文件。

整体方案属于易上手的 baseline，后续可以考虑增大模型、调整 max_length、加入数据增强或做集成提升效果。
