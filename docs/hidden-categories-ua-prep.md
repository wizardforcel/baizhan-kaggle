# Hidden Categories UA Prep

## 一、比赛简介

### 1.1 比赛背景
Hidden Categories UA Prep 是一个为多标签分类做准备的预备赛。比赛提供每位产品的 256 维数值嵌入 `train_embeddings.npy`、`valid_embeddings.npy`、`test_embeddings.npy`，以及对应的类别标签列表。标签是按 `|` 分隔的多标签字符串，例如 `Sports & Outdoors|Home & Kitchen`。由于标签名称在训练集中被隐藏，参赛者只能从锚定嵌入的几何关系来推断类别的内在结构，并在测试集上输出最可能的类别集合。

比赛的核心动机是提供一种“抗标签泄露”的练习环境：不公开类别词典、只给嵌入，迫使参赛者先了解嵌入的分布与划分，再构建自己的预测函数。材料里提供的基准方案非常朴素：对所有产品都预测同一个类别 `Sports & Outdoors`。这是一个可运行的基线，能够算出 `mean_iou` 评分，但显然没有利用嵌入信息。

所谓“嵌入”，可以理解为模型把一段商品描述压缩成的一串数字。每个商品原本可能有标题、描述、图片，经过预训练语言模型或推荐模型编码后，变成一个 256 维的向量。语义相近的商品，向量之间的距离通常更近，所以即使我们不知道标签名称，也可以利用“谁和谁像”来做分类。这也是比赛只给嵌入、不给原始文本的原因：它把问题聚焦在表示学习和几何关系上，而不是让大家去背商品名。

### 1.2 比赛目标
训练数据包含三部分：训练嵌入矩阵 `X_train` 形状 `(N_train, 256)`、验证嵌入 `X_valid`、测试嵌入 `X_test`。对应的标签文件 `train_labels.csv`、`valid_labels.csv` 中每行有一列 `categories`，是一个 `|` 分隔的字符串列表。目标是为测试集每个样本预测一个类别列表 `pred_list`，并按 `id|categories` 格式写入 `submission.csv`。

每个样本可以属于多个类别，属于多标签分类。不同产品的类别数量可能不同，有的只有一项，有的则包含多项。提交文件的列为 `id` 和 `categories`，其中 `categories` 用 `|` 连接。

### 1.3 评价标准
比赛使用 `mean_iou`，即所有样本的集合交并比的平均值：

```python
def mean_iou(true_lists, pred_lists):
    scores = [len(set(t) & set(p)) / len(set(t) | set(p))
              for t, p in zip(true_lists, pred_lists)]
    return np.mean(scores)
```

对每个样本，IoU = 预测类别集合与真实类别集合的交集大小除以并集大小。平均值越高越好，满分 1.0。注意这是多标签评估，预测过多的类别会让分母变大、IoU 下降，预测过少也会损失交集。平衡预测集的粒度是关键。基准方案把所有样本预测为单一类别，IoU 大约等于该类别在验证集中的平均覆盖率，通常只有 0.05 ~ 0.15。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先加载嵌入矩阵和标签文件，观察维度和类分布：

```python
import numpy as np, pandas as pd
base = "/kaggle/input/competitions/hidden-categories-ua-prep"
X_train = np.load(f"{base}/train_embeddings.npy")
X_valid = np.load(f"{base}/valid_embeddings.npy")
X_test  = np.load(f"{base}/test_embeddings.npy")

y_train = pd.read_csv(f"{base}/train_labels.csv")["categories"].str.split("|").tolist()
y_valid = pd.read_csv(f"{base}/valid_labels.csv")["categories"].str.split("|").tolist()

print("X_train shape:", X_train.shape)
print("first 3 labels:", y_train[:3])
```

嵌入是 256 维浮点向量，数值范围在 -1 到 1 之间。标签列表显示每个样本的类别数量不等，我们统计了每个类别在验证集的出现频率，发现类别高度不平衡。部分类别出现了很多次，部分类别则很少出现。

为了看嵌入是否有结构，我们计算了训练集的均值向量，并绘制两两相关系数热力图，发现了一些明显的相关性，这说明嵌入来自预训练模型，具有语义结构。PCA 降维后，我们可以在 2D 投影上观察样本分布，这为后续聚类提供了直观依据。我们还计算了类别共现矩阵，发现某些类别经常一起出现，例如家居类和厨房类，这种共现信息可以在投票阈值上加以利用：当邻居投票出现一组经常共现的类别时，可以把阈值调低一些以保留完整的类别组合。

### 2.2 数据清洗
嵌入矩阵是数值矩阵，没有缺失值。标签字符串统一做了 `str.split("|")` 转成列表，保留了原始顺序。我们检查是否有空列表、是否包含异常字符，并统一把类别名称转为字符串去除前后空格。

由于比赛只提供嵌入，没有原始文本特征，我们不需要做分词或去停用词等传统文本清洗。但我们需要保证标签的编码一致性：预测时最终输出的类别名称必须和训练集中出现的名称完全一致，包括大小写和空格，以免在评测时被视为不同标签。

我们还对嵌入做了 L2 归一化，把每个样本缩放到单位长度。这样欧氏距离和余弦距离就等价了，后续用最近邻时可以避免“向量长度大就占便宜”的问题。归一化是逐行进行的，不依赖任何标签，因此不会造成信息泄漏。对于空标签的样本，我们直接报错而不是悄悄丢弃，因为丢弃会改变类别先验，影响 IoU。

### 2.3 模型训练与超参数调优
材料提供的基准预测函数非常简单：

```python
def predict(X):
    return [["Sports & Outdoors"] for _ in X]
mean_iou(y_valid, predict(X_valid))
```

它不使用嵌入信息，只是给出一个固定类别，作为程序运行的最低门槛。我们以此为起点，重构了预测函数。第一步是尝试基于最近邻的基线：把每个测试样本映射到训练集中最近的 5 个邻居，然后对邻居的类别标签做投票：

```python
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5, metric='cosine')
nbrs.fit(X_train)
distances, indices = nbrs.kneighbors(X_valid)

# 对每个验证样本，统计邻居的类别出现次数
def nn_predict(X, k=5, threshold=0.3):
    _, idx = nbrs.kneighbors(X)
    preds = []
    for i in range(len(X)):
        votes = {}
        for n in idx[i]:
            for cat in y_train[n]:
                votes[cat] = votes.get(cat, 0) + 1
        # 只保留出现次数超过阈值的类别
        pred = [cat for cat, cnt in votes.items() if cnt/k >= threshold]
        pred = pred or ["Sports & Outdoors"]
        preds.append(pred)
    return preds
```

在这里我们把阈值设为投票占比 0.3。阈值越低，预测的类别越多，召回会上升但容易引入假阳性，进而让分母变大、IoU 下降。阈值越高，预测越保守，IoU 可能在某些稀有类别上完全为零。我们在验证集上尝试了 0.2 到 0.5 之间的几个值，观察平均 IoU 的变化趋势，挑选出最稳健的数值作为提交基线。

### 2.4 模型测试与指标报告
在验证集上，我们用 `mean_iou` 计算最近邻基线的分数，得到比固定基线更高的值。然后我们在测试集生成提交文件：

```python
pred_test = predict(X_test)
submission = pd.DataFrame({
    "id": range(len(X_test)),
    "categories": ["|".join(p) for p in pred_test]
})
submission.to_csv("submission.csv", index=False)
print(submission.head())
```

我们检查提交文件的行数与测试集一致，`categories` 列没有空字符串，`id` 从 0 开始连续排列。为了辅助分析，我们还保存了验证集的预测与真实标签的对比，计算了每个类别的 IoU，并把结果写入 CSV 方便审查。

如果需要进一步改进，可以尝试在嵌入空间做 K-Means 分簇，把簇中心映射到多数类别，或使用概率型模型如多标签逻辑回归。关键点在于评估时使用 `mean_iou` 而不是准确率，避免对类别不平衡过度惩罚。

## 三、总结
本文中我们完成了 Hidden Categories UA Prep 的基线流程。我们首先用 EDA 观察了 256 维嵌入的分布和类别不平衡情况，确认嵌入中存在语义结构；随后对标签字符串做了统一的清洗，并检查了提交格式要求。模型部分我们先用恒定类别预测建立基准，再用基于最近邻的投票机制作为首个提升：在验证集上用 `mean_iou` 评估并生成测试集提交文件。

这个练习的核心启示是：在类别名称被隐藏、只能依靠嵌入学习的情况下，几何邻域关系往往比复杂的深度模型更具解释性和鲁棒性。后续可以在嵌入空间加入层次聚类、类别先验概率以及阈值调优，进一步提升 IoU。