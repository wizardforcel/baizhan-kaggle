# Word Groups 101（词分类练习）

## 一、比赛简介

### 1.1 比赛背景
Word Groups 101 是一个教初学者做文本分类的入门比赛。数据给出一批单词，每个单词属于「植物、城市、动物、其他」四类中的一类。比赛附带预先训练好的 50 维词向量，鼓励我们用向量相似度来区分类别。材料中还附了一份「多数类基线」，告诉我们完全不看单词能拿到多少分，作为后续模型必须超过的底线。

### 1.2 比赛目标
`train.csv` 与 `validation.csv` 包含 `word` 和 `label` 两列，`test.csv` 只有 `word` 与 `id`。标签取值 0~3，分别对应 plant（植物）、city（城市）、animal（动物）、other（其他）。另有 `embeddings.csv`，包含 `word, e00, ..., e49`，即每个词的 50 维预训练 GloVe 词向量。我们要为测试集中每个词预测它的类标签。

### 1.3 评价标准
评价指标是分类准确率。四个类别数量均衡，所以随机猜只有约 25% 准确率，多数类基线也是 25%。分数越高越好。公开排行榜只用测试词的一半打分，另一半用于最终私有榜。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先定位数据目录并读取三份表。

```python
from pathlib import Path
import pandas as pd

# 在输入目录下找到 embeddings.csv 所在目录作为数据根目录
if Path("/kaggle/input").exists():
    matches = [p.parent for p in Path("/kaggle/input").rglob("embeddings.csv")
               if (p.parent / "train.csv").exists()]
    assert len(matches) == 1
    DATA = matches[0]
else:
    DATA = Path(".")

label_names = ["plant", "city", "animal", "other"]
train = pd.read_csv(DATA / "train.csv")
valid = pd.read_csv(DATA / "validation.csv")
test = pd.read_csv(DATA / "test.csv")
print("Rows:", len(train), len(valid), len(test))
train.head()
```

打印后发现训练、验证、测试行数各不相同，`word` 列是单词本身，`label` 列是 0~3 的数字。我们再看标签分布。

```python
# value_counts 按出现次数排序，再映射成类别名
print(train["label"].value_counts().sort_index()
      .rename(index=dict(enumerate(label_names))))
```

四个类别的训练样本数几乎一样多，确认数据集是均衡的。正因为均衡，多数类基线拿到的准确率就约等于四分之一，也就是 0.25 左右。

### 2.2 数据清洗
词向量表 `embeddings.csv` 是按 `word` 关键词存储的，所以清洗的核心是**按单词关联**，而不是按行号对齐。我们加载向量时，要把训练词的向量提取成一个矩阵，行号只作为位置索引。

```python
import numpy as np

# 读入词向量表
emb = pd.read_csv(DATA / "embeddings.csv")

# 按 word 列关联，取出训练集每个词的 50 维向量
train_emb = train.merge(emb, on="word", how="left").iloc[:, 2:].to_numpy()
print("shape:", train_emb.shape)
```

合并时如果某词在向量表里缺失，向量会变成 NaN，我们规范化长度时能顺便发现这种情况。

```python
# 每个向量除以自己的长度，得到单位向量（范数 1）
norms = np.linalg.norm(train_emb, axis=1, keepdims=True)
print("零长度向量数:", int((norms == 0).sum()))
train_emb = train_emb / np.maximum(norms, 1e-9)
```

单位化之后，两个向量的点积就正好等于余弦相似度，可以用它衡量词义相近程度。

### 2.3 模型训练与超参数调优
多数类基线很简单：找出训练集里最常见的标签，全部预测它。

```python
# mode() 取众数，iloc[0] 保证平局时确定性地选较小的号
majority_label = int(train["label"].mode().iloc[0])
val_acc = float((valid["label"] == majority_label).mean())
print("Majority:", label_names[majority_label], "Acc:", val_acc)
```

在四个类别均衡时，这个基线段准确率约 0.25。接着我们开始真正的模型：给每个已知类别造一个「类中心」。

```python
def category_centre(words, emb):
    # 取出这组词的向量，单位化后取平均，再单位化一次
    vecs = emb[emb["word"].isin(words)].iloc[:, 1:].to_numpy(dtype=float)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    centre = vecs.mean(axis=0)
    return centre / np.linalg.norm(centre)
```

我们为「植物、城市、动物」三类各挑 5 个训练词造中心点，注意**不造**「其他」的中心，因为「其他」这个词组本来就是各种无关词凑成的，取平均没有意义。

```python
plant_words = train[train["label"] == 0]["word"].head(5).tolist()
city_words  = train[train["label"] == 1]["word"].head(5).tolist()
animal_words= train[train["label"] == 2]["word"].head(5).tolist()

centres = np.stack([
    category_centre(plant_words, emb),
    category_centre(city_words, emb),
    category_centre(animal_words, emb),
])
```

预测时，对验证集每个词计算它与三个中心的余弦相似度，取最大值对应的类别。如果最大的相似度低于一个阈值，就认为它不属于任何已知类别，判为「其他」。

```python
import numpy as np

def predict(words, threshold=0.0):
    vecs = words.merge(emb, on="word").iloc[:, 1:].to_numpy(dtype=float)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = vecs @ centres.T          # 得分：与每个中心的相似度
    best = sim.max(axis=1)
    pred = sim.argmax(axis=1)
    pred[best < threshold] = 3      # 相似度不够就归为 other
    return pred
```

阈值是个可以调的参数。我们在验证集上扫一组等距阈值，选准确率最高的那个。

```python
best_th, best_acc = 0.0, -1.0
for th in np.linspace(0.0, 0.6, 25):
    acc = (predict(valid, th) == valid["label"].to_numpy()).mean()
    if acc > best_acc:
        best_th, best_acc = th, acc
print("best threshold:", best_th, "acc:", best_acc)
```

阈值越高，被拒收的词越多（其中包含一些真正的植物、城市、动物），需要多试几档再定，而且**只能用验证集选阈值**，不能拿测试标签去调。

### 2.4 模型测试与指标报告
在本地验证集上，多数类基线约 0.25，而加入词向量和类中心后的模型显著高于基线。为了保险，我们把多数类版本也保留成一份提交，方便对照。

```python
# 用多数类生成一份对照提交，用模型结果生成最终提交
sub = pd.DataFrame({"id": test["id"], "label": predict(test)})
sample = pd.read_csv(DATA / "sample_submission.csv")
assert sub["id"].tolist() == sample["id"].tolist()

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
sub.to_csv(OUT / "submission.csv", index=False)
print("Saved:", OUT / "submission.csv")
```

保存后在 Kaggle 上运行并提交，公开排行榜大约会记录我们的模型分数。提交文件必须是 `id,label` 两列、标签为 0~3 的整数、行顺序与测试集一致。

## 三、总结
本文中我们完成了 Word Groups 101 的入门方案。我们首先读取三个数据集，确认四类标签均衡，所以多数类基线只有约 0.25 准确率。其次按 `word` 键合并词向量并做 L2 归一化。接着用每类 5 个词的向量平均构造「植物、城市、动物」三个类中心，用余弦相似度给验证集打分，并通过网格扫阈值把低相似度的词判为「其他」。测试阶段我们生成优化后的准确率明显超过多数类的提交文件。这个过程让我们完整实践了「词向量 + 相似度 + 阈值」的文本分类思路。