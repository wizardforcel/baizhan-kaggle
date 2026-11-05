# K-IPTAB 韩国专利无效预测（Korean Patent Invalidation Prediction）

## 一、比赛简介

### 1.1 比赛背景

这个比赛要解决的是一个知识产权领域的真实问题：**一件专利被人提出无效宣告之后，最后到底会被判“无效”还是“有效”？**

在专利制度里，一项技术被授予专利之后，其他人如果认为它不该被授权，可以向专利审判机构提出“无效宣告”。审判机构会看双方的书面材料，最后作出裁决。这些材料包括：专利本身的权利要求书、用来证明这项技术早就存在的“现有技术”说明书、双方当事人的辩论意见，以及提出无效所依据的法律理由。

比赛把一批真实的韩语法律文书交给参赛者，让我们训练模型去预测审判结果。它有两个很现实的难点：

- **文本特别长。** 材料里提到，单篇文书最长可以达到约 60000 个 token（token 可以粗略理解为“词块”）。而常见的 BERT 类模型一般只能处理 512 个 token，差距非常大。
- **领域非常专业。** 文本是韩语，而且充满专利、法律术语，通用模型不一定看得懂。

此外，两类结果的数量并不一定均衡，这也是比赛选择 ROC AUC 作为指标的原因。

### 1.2 比赛目标

训练集里每一行是一起专利无效案件，包含下面几列：

| 列名 | 含义 |
| --- | --- |
| `id` | 案件编号 |
| `claims` | 专利权利要求书，说明这项专利到底保护什么 |
| `prior_art_specifications` | 现有技术说明书，用来证明相关技术此前已经存在 |
| `party_arguments` | 当事人的主张和辩论理由 |
| `grounds` | 提出无效所依据的法律事由 |
| `label` | 结果，取值为 `INVALID`（无效）或 `VALID`（有效） |

测试集没有 `label`，我们要为每个 `id` 输出一个 `INVALID` 列的数值，表示“这个案件被判无效的概率”。

这里有一个关键点：按照数据说明，模型真正允许使用的输入只有**上下文**，也就是 `claims` 加 `prior_art_specifications`。`party_arguments` 和 `grounds` 是跟结论高度相关的“答案侧”信息，比赛中不应该拿来当特征。

### 1.3 评价标准

比赛使用 **ROC AUC**（Area Under the ROC Curve）作为评价指标，越高越好，完美值是 1.0，随机乱猜大约是 0.5。

用大白话解释：模型给每个案件输出一个“无效概率”，AUC 衡量的是——随机抽一个真正的无效案件和一个真正的有效案件，模型给无效案件打的分数更高的概率有多大。它只看排序，不需要我们提前定一个“大于 0.5 就算无效”的阈值，所以对类别不平衡比较宽容。

这也意味着提交的 `INVALID` 列应该是**概率**，而不是 0/1 的硬标签。硬标签只在我们自己算准确率、F1 时才会用到。

---

## 二、解题过程

我们沿着“先看懂数据，再把数据弄干净，然后从简单基线走到微调大模型，最后生成提交文件”这条线来做。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，看看数据规模、列名和标签分布：

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = "/kaggle/input/competitions/k-iptab-korean-patent-invalidation-prediction/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

# 先确认行列数、列名，避免后面用错字段
print("Train shape:", train.shape, "| Test shape:", test.shape)
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

# 标签分布：既看条数，也看百分比，判断是否类别不平衡
label_count = train["label"].value_counts()
label_pct = train["label"].value_counts(normalize=True) * 100
print(label_count)
print(label_pct)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.countplot(data=train, x="label", hue="label", palette="viridis", ax=ax[0])
ax[0].set_title("Distribution (Count)")
ax[0].set_ylabel("Count")

# 饼图把两类占比画出来，一眼就能看出是否悬殊
ax[1].pie(label_count, labels=["INVALID", "VALID"], autopct="%1.1f%%",
          colors=["#2ecc71", "#e74c3c"], explode=(0, 0.1), shadow=True)
ax[1].set_title("Distribution (Percentage)")
plt.tight_layout()
plt.show()
```

从列名可以看到，测试集原本也带着 `party_arguments` 和 `grounds`。这两个字段跟审判结论关系太近，等会儿必须处理掉，否则模型等于在“偷看答案”。

接着我们关心文本到底有多长。我们按照大模型训练时的方式，把四个字段拼成一段提示文本，再用分词器数一数 token 数：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

# 按照“字段名 + 内容”的方式拼接，让模型知道每一段是什么
def build_full_text(row):
    return (f"Claims: {row['claims']} [SEP] "
            f"Prior Art: {row['prior_art_specifications']} [SEP] "
            f"Arguments: {row['party_arguments']} [SEP] "
            f"Grounds: {row['grounds']}")

train["full_text"] = train.apply(build_full_text, axis=1)

# truncation=False：这里故意不截断，就是想看看真实长度分布
lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["full_text"]]

plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
```

直方图通常会出现很长的右尾：大部分案件的长度还在可控范围内，但少数案件能冲到几万 token。材料里明确说最长约 60000 token，远超普通 BERT 的 512 上限。这直接决定了后面的策略——要么截断，要么换支持长文本的模型。我们最终选择 `ModernBERT-base`，并把最大长度设为 1024，这是在显存、速度和“能保留多少信息”之间的折中。

### 2.2 数据清洗

这一步我们做了三件事，每一件都直接影响分数是否可信。

**第一，堵住答案泄漏。** 测试集里的 `party_arguments` 和 `grounds` 是跟裁决理由直接相关的字段。按照数据说明，模型输入应该只有 `claims` 和 `prior_art_specifications`。如果训练时用了这四个字段、测试时却只有两个字段，模型学到的模式在测试时根本用不上；更糟的是，这会让离线验证分数虚高。所以我们在训练和测试里统一只保留两个上下文列。

**第二，修正提交文件的行数。** 材料里记录了一个数据缺陷：`sample_submission.csv` 有 362 行，但测试集只有 359 个唯一 `id`。多出来的三行是 `2009100002929`、`2013100003420`、`2021100001901` 被重复计算了。如果不先去重，最后提交要么行数对不上，要么同一个案件被提交两次。

**第三，统一标签和文本格式。** 我们把 `INVALID` 映射成 1、`VALID` 映射成 0，这样模型输出的就是“无效概率”；拼接文本前先填空串，避免缺失值导致报错。

```python
# 只保留允许使用的上下文列，彻底避开 party_arguments / grounds 的泄漏
def make_text(df):
    # fillna("")：法律文书可能缺字段，先补空串，保证字符串拼接不报错
    return (df["claims"].fillna("") + " [SEP] " +
            df["prior_art_specifications"].fillna("")).astype(str)

train["text"] = make_text(train)
test["text"] = make_text(test)

# INVALID 记为 1，正好对应提交文件里的 P(INVALID)
y = train["label"].map({"INVALID": 1, "VALID": 0}).astype(int).values

# 样例提交先按 id 去重，保证和 359 个测试 id 一一对应
subm = pd.read_csv(path + "sample_submission.csv").drop_duplicates("id")
print("test unique ids:", test["id"].nunique())
print("sample_submission rows after dedup:", len(subm))
```

### 2.3 模型训练与超参数调优

我们准备了两条路线：一条是轻量的 TF-IDF 加逻辑回归基线，用来快速确认“文本里到底有没有可学的信号”；另一条是微调预训练语言模型，争取更好的效果。此外还有一个只做工程自检的表格模型。

**第一条路线：字符级 TF-IDF + 逻辑回归。**

韩语跟中文一样，词与词之间通常不加空格，直接按“单词”切分很容易切错。所以我们改用**字符级 n-gram**：把文本切成连续 2 到 5 个字符的片段，统计这些片段出现的频率。这样不需要韩语分词器，也不需要联网下载词典。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# char_wb：在字符边界内切 n-gram，对韩语这种无空格语言比较稳
# max_features 限制特征数量，避免几十万维把内存撑爆
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                      min_df=3, max_features=200_000, sublinear_tf=True)
Xtr = vec.fit_transform(train["text"])
Xte = vec.transform(test["text"])

# class_weight="balanced"：让少数类在损失里权重更大，缓解类别不平衡
clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")

# 5 折分层交叉验证：每折的类别比例都和原数据一致
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = cross_val_predict(clf, Xtr, y, cv=cv, method="predict_proba")[:, 1]

# 用袋外预测算分数，比“训练集自评”诚实得多
print("CV AUC = %.4f | ACC = %.4f | F1 = %.4f" % (
    roc_auc_score(y, oof),
    accuracy_score(y, (oof > 0.5).astype(int)),
    f1_score(y, (oof > 0.5).astype(int))))
```

这里特意用 `cross_val_predict` 而不是先训练再在训练集上打分：前者每个样本的预测都来自没见过它的那一折模型，能避免“自己考自己”的虚高。

**第二条路线：微调 ModernBERT。**

我们把任务当成两个类别的文本分类，用 `AutoModelForSequenceClassification` 加载预训练模型，再在比赛数据上做监督微调（SFT，Supervised Fine-Tuning，也就是“用带标签的数据继续训练预训练模型”）。

```python
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 1024

# 分层切出 20% 做验证集，保证验证集里两类比例与训练集一致
tr_df, val_df = train_test_split(
    train, test_size=0.2, stratify=train["label"], random_state=42
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    # 截断到 1024：长文档只保留前 1024 个 token，是显存与信息量的折中
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

tok_train = Dataset.from_pandas(tr_df).map(tokenize_function, batched=True)
tok_val = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

# 只保留模型真正需要的三列，避免把原始文本也塞进张量
cols = ["input_ids", "attention_mask", "label"]
tok_train.set_format(type="torch", columns=cols)
tok_val.set_format(type="torch", columns=cols)
```

训练参数方面，我们用 3 个 epoch、学习率 2e-5、批大小 16，并开启混合精度 `fp16` 来省显存。每隔 100 步在验证集上算一次 AUC，并且保留验证 AUC 最好的那个检查点：

```python
from transformers import (AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

# num_labels=2：无效 / 有效两类
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

args = TrainingArguments(
    output_dir="./ver_1",
    eval_strategy="steps",          # 按步数而不是按 epoch 评估，反馈更及时
    save_strategy="steps",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,             # 微调大模型常用的较小学习率
    logging_steps=50,
    eval_steps=100,
    save_steps=100,
    save_total_limit=1,             # 只留最新/最好的一个检查点，省磁盘
    metric_for_best_model="roc_auc",
    greater_is_better=True,
    load_best_model_at_end=True,    # 训练结束后自动回到最佳检查点
    fp16=True,
    report_to="none",
)

def metrics(pred):
    labels = pred.label_ids
    probs = softmax(pred.predictions, axis=1)
    # 这里标签是 VALID=1、INVALID=0，所以第 1 列是 VALID 概率
    return {"roc_auc": roc_auc_score(labels, probs[:, 1])}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=metrics,
)
trainer.train()
```

**第三条路线：microModel 表格基线。** 材料里还有一个很小的模型：把文本列做序数编码（给每个不同取值编一个整数），再喂给 CatBoost、LightGBM、XGBoost 三个树模型，按 0.45、0.25、0.30 加权平均。它完全不理解韩语文本，作者自己也说它只是“冒烟测试”，用来确认数据管道能跑通，所以我们不把它当主方案。

```python
from sklearn.preprocessing import OrdinalEncoder
import catboost, lightgbm, xgboost

# 输入只取测试集里除 id 外的列，避免把标签带进来
feats = [c for c in test.columns if c != "id"]
X = train[feats].copy()
Xte = test[feats].copy()

# 序数编码：把字符串变成整数；没见过的取值统一记成 -1
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
cat_cols = test.select_dtypes(include="object").columns.tolist()
X[cat_cols] = encoder.fit_transform(X[cat_cols])
Xte[cat_cols] = encoder.transform(Xte[cat_cols])

# 三个树模型各自回归，再按经验权重融合
preds_cat = catboost.CatBoostRegressor(verbose=False).fit(X, y).predict(Xte)
preds_lgb = lightgbm.LGBMRegressor(verbosity=-1).fit(X, y).predict(Xte)
preds_xgb = xgboost.XGBRegressor(verbosity=0).fit(X, y).predict(Xte)

ensemble = 0.45 * preds_cat + 0.25 * preds_lgb + 0.30 * preds_xgb
ensemble = ensemble.clip(0, 1)   # 回归输出可能越界，裁剪回 [0, 1] 概率范围
```

### 2.4 模型测试与指标报告

训练完成后，我们先在验证集上看 AUC。AUC 只关心排序，所以不需要调整 0.5 阈值；如果还想看准确率和 F1，才需要把概率按 0.5 切成硬标签。材料里的 notebook 主要展示流程，没有在正文记录最终分数，所以这里不写具体数字，只说明我们看什么：**验证集 AUC 是否明显高于 0.5、训练损失是否稳定下降、验证损失有没有过早反弹**。如果验证 AUC 停止提升，就说明 3 个 epoch 已经够了，继续训练只会过拟合。

确认模型可用后，我们对测试集做同样的文本拼接和分词，再输出“无效概率”：

```python
from datasets import Dataset
from scipy.special import softmax

# 测试集同样只拼 claims + prior_art_specifications，保持与训练一致
test["text"] = make_text(test)
tok_test = Dataset.from_pandas(test[["text"]]).map(
    tokenize_function, batched=True
)

pred = trainer.predict(tok_test)
probs = softmax(pred.predictions, axis=1)

# 训练时 VALID=1、INVALID=0，因此第 0 列才是 P(INVALID)
submission = pd.DataFrame({
    "id": test["id"],
    "INVALID": probs[:, 0],
})
submission.to_csv("submission.csv", index=False)
```

最后我们做一遍提交前检查。这一步看起来琐碎，但比赛里因为行数、列名、重复 id 出错而白白丢分的情况并不少见：

```python
# 行数应与测试集唯一 id 数一致（这里应为 359）
print("rows:", len(submission))
print("unique ids:", submission["id"].nunique())
print("duplicated ids:", int(submission["id"].duplicated().sum()))

# 概率应落在 [0, 1]，且分布不应全部挤在 0 或 1
print(submission["INVALID"].describe())

# 对照样例提交的列名和 id 顺序
print(submission.head())
```

如果 `rows` 是 359、`duplicated ids` 是 0、`INVALID` 的均值和分位数都在合理范围内，这份提交就可以交上去了。

---

## 三、总结

本文中我们完成了 K-IPTAB 韩国专利无效预测的完整流程。我们首先通过 EDA 确认了数据结构：每起案件由权利要求、现有技术说明书、当事人主张和无效事由等韩语法律文本组成，单篇最长可达约 60000 个 token，远超普通 BERT 的处理上限。其次我们做了关键的数据清洗：去掉测试集里会泄漏答案的 `party_arguments` 和 `grounds`，把训练和测试的输入统一成 `claims` 加 `prior_art_specifications`，并把有 362 行的样例提交按 id 去重到 359 行。接着我们先跑了一个字符级 2～5 gram 的 TF-IDF 加逻辑回归基线，用 5 折分层交叉验证得到诚实的袋外分数；然后微调 `ModernBERT-base`，把最大长度设为 1024，用 3 个 epoch、2e-5 学习率和验证集 AUC 早停来选择最佳检查点。最后我们按 `id + P(INVALID)` 的格式生成提交文件，并检查了行数、唯一 id 和概率范围。

这个比赛给我们最大的启发是：**先把数据泄漏堵住，再谈模型效果**。如果训练时用了答案侧字段、测试时却没有，离线分数再高也不可信；同样，提交文件的行数和 id 对不上，模型再好也会被判无效。在这两个前提之下，字符 n-gram 的线性基线适合快速验证信号，而支持长文本的预训练模型才是冲击更高 AUC 的主力。
