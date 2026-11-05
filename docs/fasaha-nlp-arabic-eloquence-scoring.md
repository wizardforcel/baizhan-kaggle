# FasahaNLP：阿拉伯语修辞优雅度评分（Arabic Eloquence Scoring）

## 一、比赛简介

### 1.1 比赛背景

Fasaha（فصاحة，读作「法萨哈」）在阿拉伯语里是「雄辩、纯正」的意思。阿拉伯语有一条很长的历史脉络：从古典诗歌、宗教经典，到近代的「现代标准阿拉伯语」，再到今天各地的方言，不同文本的「语体纯正程度」差别很大。这个比赛想做的事，就是让模型给一段阿拉伯语文本打一个 **0 到 1 之间的「优雅度分数」（fasaha_score）**——越接近 1 表示语体越古典、越纯正；越接近 0 表示越口语、越像方言，甚至是乱码。

也就是说，这本质上是一个**文本回归（text regression）**任务：输入一段阿拉伯语，输出一个连续分数。它和常见的「文本分类」不同，模型不是在挑类别，而是在估计一个连续的程度值，难度更高。

数据来自真实的阿拉伯语语料，覆盖了多个语体层次。比赛还特别规定：测试集里**不含《古兰经》（Qur'an）文本**——这是训练集里分数最高、最特殊的一档，测试时不出现，所以预测时不能把测试样本打成满分。

### 1.2 比赛目标

给定一段阿拉伯语文本，预测它的 `fasaha_score`。

数据文件有：

- `train.csv`：训练集，包含 `id`、`text`（阿拉伯语文本）、`fasaha_score`（标签，0～1）；
- `test.csv`：测试集，只有 `id` 和 `text`；
- `sample_submission.csv`：提交格式样例。

训练集的分数落在 6 个「语体档位」（rubric band）上，每个档位对应一个固定分数：

| 档位 | 含义 | 对应分数 |
| --- | --- | --- |
| `dialect` | 方言/口语 | 0.10 |
| `msa` | 现代标准阿拉伯语 | 0.40 |
| `late_heritage` | 近代传承文本 | 0.70 |
| `ca` | 古典阿拉伯语（Classical Arabic） | 0.85 |
| `quran` | 《古兰经》文本 | 1.00 |
| `garbled` | 残缺/乱码文本 | 不在固定档位上 |

我们要输出的就是这些分数之间的连续值。这些档位也决定了后面验证集的切分方式——按档位分层抽样，才能保证每一折里各语体比例一致。

### 1.3 评价标准

比赛采用一个自定义指标，代码里叫 **BW-MAE（Band-Weighted Mean Absolute Error，档位加权平均绝对误差）**。它的计算分三步：

1. 先算普通的绝对误差 `|真实值 - 预测值|`；
2. 再看真实值、预测值分别落在哪个档位，算出两者的「档位差」；
3. 按档位差查一张惩罚权重表，把误差乘上权重后取平均。

惩罚权重是 `[1.0, 1.5, 2.5, 3.5, 4.0, 4.0]`，档位差越大，权重越高。这很好理解：把 0.85 的古典文本预测成 0.80，只差 0.05，无伤大雅；但把 0.85 预测成 0.10（方言），虽然数值只差 0.75，性质上却是「把阳春白雪认成了下里巴人」，必须重罚。

**这个指标越低越好**，0 是完美。注意它和普通 MAE 一样是「误差」而不是「准确率」，所以排行榜上 0.0259 比 0.0948 好。

---

## 二、解题过程

我们的路线是：**先做充分的文本 EDA，理解阿拉伯语的特殊之处（变音符号、分词方式、语体线索），再用预训练的阿拉伯语语言模型 AraELECTRA 做回归微调。** 我们按四步走。

### 2.1 数据探索（EDA）

我们首先把训练集、测试集读进来，检查形状、缺失值和重复文本：

```python
import pandas as pd

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print("train shape:", df_train.shape, "test shape:", df_test.shape)
print("train null:", df_train.isna().sum().to_dict())
print("test null:", df_test.isna().sum().to_dict())
print("重复文本数:", df_train['text'].duplicated().sum())
```

接着我们把连续的分数映射回 6 个档位，看看各语体的样本量分布。这一步很重要，因为如果某一档样本极少，分层交叉验证和模型学习都会出问题：

```python
import numpy as np

def map_score_to_band(scores):
    # 按分数区间映射到语体档位；用 isclose 判断 1.00，避免浮点误差
    conditions = [
        scores.isna(),
        np.isclose(scores, 1.00, atol=1e-5),
        scores >= 0.85,
        scores >= 0.70,
        scores >= 0.40,
        scores >= 0.10,
    ]
    choices = ["unknown", "quran", "ca", "late_heritage", "msa", "dialect"]
    return pd.Series(np.select(conditions, choices, default="garbled"),
                     index=scores.index)

df_train["band"] = map_score_to_band(df_train["fasaha_score"])
print(df_train["fasaha_score"].describe())          # 看分数的均值、分位数
print(df_train["band"].value_counts())              # 各档位的样本数
```

**阿拉伯语文本有一个很强的语体信号：变音符号（tashkeel / diacritics）。** 阿拉伯文字平时书写通常省略短元音符号，但古典文本、宗教文本为了准确诵读，往往会标上密密麻麻的变音符号（如 fatha、damma、kasra、shadda）。因此我们统计了「变音符号密度」——变音符号数除以总字符数：

```python
import re

# Unicode 里的阿拉伯语变音符号区间
ARABIC_DIACRITICS_REGEX = re.compile(r"[ً-ْٰ]")

def extract_orthographic_features(text):
    text = str(text)
    total_chars = max(len(text), 1)
    num_diacritics = len(ARABIC_DIACRITICS_REGEX.findall(text))
    num_shaddah = text.count("ّ")                        # 叠音符
    num_tanween = sum(text.count(c) for c in ["ً", "ٌ", "ٍ"])  # 鼻音符
    return {
        "diacritic_density": num_diacritics / total_chars,
        "shaddah_density":  num_shaddah / total_chars,
        "tanween_density":  num_tanween / total_chars,
    }

ortho_df = df_train["text"].apply(extract_orthographic_features).apply(pd.Series)
df_train = pd.concat([df_train, ortho_df], axis=1)

# 按档位看变音符号密度：古典/宗教档位通常明显更高
print(df_train.groupby("band")["diacritic_density"].agg(["mean", "median", "std"]))
```

我们还比较了训练集和测试集的词表重合度（OOV 率），并统计测试集里是否可能出现《古兰经》文本。结论是测试集按比赛约束不含 `quran` 档，所以后面预测时要给分数设一个上限。

最后一项 EDA 是**分词器基准测试**。阿拉伯语的构词法很复杂，一个词往往能拆出好几个「子词」，不同预训练模型的分词器效率差别很大。我们用同一批样本文本，测量每个分词器的「生育率」（fertility，子词数 / 词数，越低越好）：

```python
from transformers import AutoTokenizer

candidates = [
    "aubmindlab/araelectra-base-discriminator",
    "aubmindlab/bert-large-arabertv02",
    "UBC-NLP/MARBERTv2",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix",
]

sample_texts = df_train["text"].dropna().sample(500, random_state=42).tolist()
total_words = sum(len(t.split()) for t in sample_texts)

for model_id in candidates:
    tok = AutoTokenizer.from_pretrained(model_id)
    total_tokens = sum(len(tok.tokenize(t)) for t in sample_texts)
    print(model_id.split("/")[-1], "fertility:", round(total_tokens / total_words, 3))
```

分词器把同样的文本切成越少的子词，说明它对阿拉伯语越「懂行」，同样的 256 个 token 能装下更多内容。这一步帮我们选定了后续使用的主干模型。

### 2.2 数据清洗

这份数据的清洗比较轻，主要是三点：

1. **空值统一处理**：`text` 列用空字符串填充，避免向量化时报错。
2. **不删除变音符号**。通常做阿拉伯语文本预处理的第一反应是「去掉所有 tashkeel 做归一化」，但在这里**绝对不能删**——变音符号密度本身就是区分语体的强特征，删掉等于把最有用的信号扔掉。
3. **截断长度要合理**。我们统计了词数的分位数，按阿拉伯语分词器「一词约 1.3 个子词」的经验换算，最终把最大长度定为 **256 个 token**。这个长度能覆盖绝大多数样本，又不会让显存爆掉。

```python
# 统计词数分位数，帮助确定合理的最大序列长度
percentiles = [50, 75, 90, 95, 98, 99, 100]
p_train = np.percentile(df_train["word_count"], percentiles)
# 按「1 词约 1.3 子词」估算子词数，再取 16 的整数倍
subword_estimate = int(np.ceil(p_train[4] * 1.30 / 16) * 16)
max_len = min(subword_estimate, 512)
```

### 2.3 模型训练与超参数调优

我们的主力模型是 **AraELECTRA**（`aubmindlab/araelectra-base-discriminator`）。ELECTRA 的训练方式和普通 BERT 不同：它让一个小「生成器」负责把句子里的词替换掉，再让「判别器」去判断每个词是真是假。这样每个词都能提供训练信号，样本利用率高，在阿拉伯语理解基准 ALUE 上表现很好，适合语料不算特别大的场景。

因为要输出连续分数，我们把预训练模型顶部的分类头改成**单输出回归头**（`num_labels=1`），损失函数用 **Smooth L1（Huber）损失**。它在大误差时像普通 L1（绝对值误差）一样稳健，在小误差附近又像 L2 一样平滑、好优化，和我们「按绝对误差打分」的评价指标很匹配。

```python
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

model_name = "aubmindlab/araelectra-base-discriminator"
config = AutoConfig.from_pretrained(model_name, num_labels=1)  # 单输出 = 回归
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

# Smooth L1：大误差时近似 MAE，小误差时平滑可导
criterion = nn.SmoothL1Loss()

# 参数分组：偏置和 LayerNorm 不做权重衰减，其余参数加 0.01 的 L2
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
```

超参数方面：学习率 **2e-5**（微调预训练模型的标准量级，太大容易把预训练知识冲掉），批大小 **32**，训练 **2 轮**（这么小的学习率下，2 轮已经足够让模型适应任务，再多容易过拟合），前 10% 的步数做**学习率预热**，然后用余弦退火慢慢降下来：

```python
from transformers import get_cosine_schedule_with_warmup

total_steps = len(train_loader) * epochs          # 总训练步数
warmup_steps = int(total_steps * 0.1)             # 前 10% 步数预热
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps,
    num_training_steps=total_steps)
```

训练循环里还用了两个技巧：**FP16 混合精度**（`torch.amp.autocast`）让显存占用减半、速度更快；**梯度裁剪**（`clip_grad_norm_`，上限 1.0）防止梯度爆炸：

```python
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, batch["target"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
```

**验证集怎么切？** 用 **5 折分层交叉验证**，分层的依据是语体档位，保证每一折里方言、标准语、古典语的比例都和整体一致。为了贴合测试集「没有古兰经」的特点，我们额外计算了一个「去掉 quran 档」的验证分数，并用它来挑最好的模型；如果只看包含古兰经的分数，会高估模型在实际测试集上的表现。

### 2.4 模型测试与指标报告

我们先把评价指标写成函数，方便每一折复现：

```python
PENALTY_WEIGHTS = np.array([1.0, 1.5, 2.5, 3.5, 4.0, 4.0])

def score_to_band(scores):
    arr = np.asarray(scores, dtype=np.float64)
    bands = np.zeros_like(arr, dtype=np.int64)
    bands[arr >= 0.10] = 1     # dialect
    bands[arr >= 0.40] = 2     # msa
    bands[arr >= 0.70] = 3     # late_heritage
    bands[arr >= 0.85] = 4     # ca
    bands[np.isclose(arr, 1.00, atol=1e-5)] = 5   # quran
    return bands

def compute_bw_mae(y_true, y_pred):
    # 先把预测和真值夹到 [0,1]，防止越界拉大误差
    y_t = np.clip(np.asarray(y_true, dtype=float), 0.0, 1.0)
    y_p = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)
    band_diff = np.minimum(np.abs(score_to_band(y_t) - score_to_band(y_p)), 4)
    weights = np.take(PENALTY_WEIGHTS, band_diff)   # 档位差越大，权重越高
    return float(np.mean(weights * np.abs(y_t - y_p)))
```

作为对照，我们首先做了一个**只做特征工程、不做深度学习的基线**：用 TF-IDF（词级 1～2 元语法，最多 25000 个特征）加上三列手工特征（字符数、词数、变音符号密度），再喂给岭回归（Ridge）：

```python
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# 文本特征：词级 1-2 gram，sublinear_tf 压低高频词的权重
word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)
X_word_train = word_vec.fit_transform(df_train["text"].fillna("").astype(str))

# 稠密特征：字符数、词数、变音符号密度，做标准化
X_dense_train = StandardScaler().fit_transform(extract_dense_features(df_train))

# 稀疏 + 稠密拼接
X_train = hstack([X_word_train, X_dense_train]).tocsr()

# 5 折分层交叉验证
for train_idx, val_idx in skf.split(df_train, df_train["band"]):
    model = Ridge(alpha=1.0)
    model.fit(X_train[train_idx], y_train[train_idx])
    val_pred = model.predict(X_train[val_idx])
```

这个 TF-IDF + Ridge 基线在排行榜上的成绩是 **BW-MAE 0.0948**。它证明了「变音符号 + 词汇 n-gram」确实携带语体信息，但也说明纯词袋模型的天花板不高——它读不懂句子结构。

换成 AraELECTRA 微调后，排行榜成绩降到 **BW-MAE 0.0259**，比基线好了约 3.7 倍。这个提升主要来自预训练模型对阿拉伯语语法和语体搭配的理解：它不只是数「哪些词出现了」，而是理解了「这些词以什么方式组合」。

最后生成提交文件时有一个**必须做的截断**：因为测试集里没有古兰经文本，模型不该输出 1.00 分，所以把预测上限压到 0.985：

```python
# 测试集不含古兰经，上限设为 0.985；下限 0
test_preds = np.clip(np.array(test_preds), 0.0, 0.985)

submission = pd.DataFrame({
    "id": df_test["id"],
    "fasaha_score": np.round(test_preds, 4)   # 保留 4 位小数
})
submission.to_csv("submission.csv", index=False)
```

---

## 三、总结

本文中我们完成了阿拉伯语修辞优雅度评分的完整流程。我们首先做了 EDA，把连续分数映射到六个语体档位，发现**变音符号密度**是区分语体的强信号，并用词数分位数确定了 256 的最大序列长度，还用「子词生育率」基准挑选了分词效率最高的阿拉伯语预训练模型。其次我们做了克制的数据清洗：统一空值、**保留变音符号**、不做过度归一化。接着我们对比了两条路线——TF-IDF + 手工特征的岭回归基线（LB 0.0948）和 AraELECTRA 回归微调（单输出回归头、Smooth L1 损失、AdamW、2e-5 学习率、余弦退火、FP16 混合精度、梯度裁剪），并用按语体分层的 5 折交叉验证和「去掉古兰经」的验证口径来选模型。最后 AraELECTRA 取得 **LB 0.0259** 的成绩，提交时把预测上限截到 0.985 以符合「测试集不含古兰经」的约束。这个比赛给我们的启发是：**在低资源语言任务上，手工语言知识（变音符号）和预训练语言模型是互补的**，而评价指标里的「档位惩罚」提醒我们，回归任务有时也要关心「错到了哪个程度」，而不只是数值差多少。