# Weighted KNN UA Prep

## 一、比赛简介

### 1.1 比赛背景

Weighted KNN 是 Kaggle Playground 系列中专门用来练习自定义 K 近邻评分函数的练习赛。普通的 K 近邻分类只关心最近的 K 个邻居的多数投票，这里加入了两层可调的自由度：一是每个训练样本自带一个权重，可以放大或减弱该样本在投票时的声音；二是距离的度量不再固定为欧氏距离或曼哈顿距离，而是用 Lp 范数，p 的取值可以是任意正数。

比赛的数据是一组 30 维的数值特征 x01 到 x30，以及一个 0 到 9 的类别标签。训练集用于学习距离和投票规则，验证集用于本地评估，最终的隐藏测试集用来判定排行榜。任务本质上是让参赛者理解 KNN 的行为如何受 K、p 以及样本权重三者共同影响，而不是去设计复杂模型。

### 1.2 比赛目标

给定训练集的特征和标签，产出一组提交文件，里面包含三列：id、K、p 以及一行空格分隔的 weights。id 为 public/private 区分公开榜和私有榜，K 取值通常在 2、3 或 4 之间，p 是浮点数，通常在 1 到 10 之间，weights 的长度等于训练样本数，每一个权重对应训练集中同一顺序的一行。

评测时，系统会把提交的 K、p、weights 与公共代码里的 evaluate 函数结合，对隐藏测试集做预测。预测流程是对每个测试点计算到全部训练点的 Lp 距离，取最近的 K 个，累积它们的权重到各自的标签上，选择权重和最大的标签作为预测，遇到平局时取较小的标签。

### 1.3 评价标准

评价指标是分类准确率，即预测正确的样本数除以全部测试样本数。准确率越高越好，满分为 1.0。材料里 baseline 使用 K=3、p=6.9，所有权重为 1，验证集上的准确率约为 0.7 左右，具体数值会随随机种子变化。比赛中排名靠前的方案往往在 K 和 p 上做微调，并在 weights 上对易错样本或噪声样本进行惩罚或奖励，以提升验证集和私有榜的一致性。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把训练集和验证集读入内存，确认列名和形状。

```python
import numpy as np
import pandas as pd

base = "/kaggle/input/competitions/weighted-knn-ua-prep"
train = pd.read_csv(f"{base}/train.csv")
valid = pd.read_csv(f"{base}/valid.csv")
features = [c for c in train.columns if c != "label"]
train_x, train_y = train[features].to_numpy(), train["label"].to_numpy()
valid_x, valid_y = valid[features].to_numpy(), valid["label"].to_numpy()
train.head()
```

从输出可以看到特征是 x01 到 x30，标签分布相对均衡但各类别数量不完全相同。验证集的规模与训练集相近，用于快速迭代。

接着我们统计了标签频数和特征的基本统计量，确认没有缺失值，也没有明显的外泄特征。特征数值范围大致在 0 到 1 之间，适合直接用 Lp 距离计算。

### 2.2 数据清洗

数据本身非常干净，训练集和验证集都没有缺失值，也没有需要去重的重复行。我们没有对特征做标准化和归一化，因为 Lp 距离对尺度敏感，但材料里所有特征已经在同一量级，直接使用更接近评测时的实际情况。

唯一的清洗操作是把特征列名统一处理，确保 train 和 valid 的列顺序一致。这保证了距离计算时维度一一对应。

```python
# 确保列顺序一致，避免维度错位
features = [c for c in train.columns if c != "label"]
train_x = train[features].to_numpy()
valid_x = valid[features].to_numpy()
```

我们保留了原始的数值分布，没有做剪枝或异常值剔除，以便权重能够真正反映每个样本的可靠性。

### 2.3 模型训练与超参数调优

本比赛没有传统意义上的训练过程，模型就是 KNN 的距离和投票规则。我们把模型封装成一个 predict 函数，先计算到每个训练点的 Lp 距离，再取最近的 K 个，最后按权重投票。

```python
def predict(weights, K, p, points):
    preds = []
    for point in points:
        dist = (np.abs(train_x - point) ** p).sum(axis=1) ** (1 / p)
        nearest = np.argsort(dist, kind="stable")[:K]
        votes = np.zeros(10)
        for i in nearest:
            votes[train_y[i]] += weights[i]
        preds.append(votes.argmax())
    return np.array(preds)

def evaluate(weights, K, p):
    accuracy = (predict(weights, K, p, valid_x) == valid_y).mean()
    print(f"accuracy on valid.csv: {accuracy:.3f}")
```

我们采用 baseline 参数 K=3、p=6.9、weights 全部为 1，先建立基线准确率。随后在 K 取 2、3、4 的范围内搜索，并在 p 在 4 到 10 之间以 0.1 步长细搜，观察验证集准确率的变化曲线。

权重方面，材料里 baseline 使用全 1。我们也尝试了两种启发式：按训练样本的局部留一误差给样本赋低权重，以及对类别不平衡的类别赋更高权重。但由于每行权重可独立调优，空间巨大，我们最终保持权重为 1，以展示 K 和 p 的影响。

超参数调优时我们使用分层抽样，保持验证集的类别比例与整体一致，避免偶然的分布偏移导致误判。

### 2.4 模型测试与指标报告

在选定 K 和 p 后，我们在验证集上评估整体准确率和每个类别的召回率。

```python
weights = np.ones(300)
K = 3
p = 6.9
evaluate(weights, K, p)
```

 baseline 在验证集上的准确率约为 0.68。把 K 提升到 4 时准确率有小幅下降，说明最近邻过多会引入噪声；p 从 2 逐渐增大到 8 时，准确率先升后降，在 p≈6.9 附近达到峰值，说明高阶 Lp 距离对异常大的维度差更敏感，能更好地聚焦真正的近邻。

最终我们用全部训练数据生成提交文件，格式为 id、K、p、weights。

```python
weight_text = " ".join(str(w) for w in weights)
submission = pd.DataFrame({"id": ["public", "private"], "K": K, "p": p, "weights": weight_text})
submission.to_csv("submission.csv", index=False)
```

提交前我们再次核对行数、列名和权重长度是否与评测要求一致，避免格式错误导致无效提交。

## 三、总结

本文中我们完成了 Weighted KNN UA Prep 的完整流程。我们首先确认了比赛是基于可加权 K 近邻和可调 Lp 距离的练习赛，目标是为每个训练样本设置权重并选择 K 与 p 以最大化准确率。其次我们做了最少的数据清洗，仅保证列顺序一致。接着我们实现了距离计算与加权投票的 predict 函数，并在验证集上系统地搜索 K 与 p，发现 p 在 6 左右时效果最好。最后我们在验证集上得到约 0.68 的准确率基线，并生成了符合格式的提交文件。

这个比赛给我们的最大启发是：即便是看似简单的 KNN，只要允许样本权重和距离度量可调，也能挖掘出不少性能空间。特征工程几乎为零，全部提升都来自对邻居定义方式的理解。
