# 最小训练集挑战（Smallest Training Set - UA Prep）

## 一、比赛简介

### 1.1 比赛背景

我们平时参加 Kaggle 比赛，套路一般是「主办方给一份固定的训练集，我们想办法把模型调得更好」。这个比赛把方向反了过来：**模型是固定的，能被我们选择的只有训练数据**。

比赛给出一张 2000 行左右的训练表 `train.csv`，每行是一个样本，30 个特征列分别叫 `s01` 到 `s30`，最后一列 `label` 是答案。参赛者要做的事情不是提交预测值，而是提交「我用了哪些行、哪些列」。主办方的评分程序拿到我们交的行号和列名后，会用一个**事先写死的模型**在这份子集上训练，再去隐藏测试集上打分。

这个设定想考察的东西很纯粹：在真实项目里，标注数据和采集特征都是要花钱的。如果能用 50 行、3 列就达到和 2000 行、30 列差不多的效果，那就省下了大量成本。所以比赛名字叫「Smallest Training Set」，直译就是「最小的训练集」。`ua-prep` 是比赛系列的名字，可以理解成一类以「数据准备」为主题的练习赛。

这个比赛还有一个有意思的地方：因为模型不允许换，调参的空间被彻底堵死了。我们不能再靠「换个更强的模型」提分，只能老老实实想清楚**哪些样本有代表性、哪些特征真正有用**。

### 1.2 比赛目标

输入是两份表格：

- `train.csv`：2000 行、30 个特征列 `s01`～`s30`，外加一列 `label`；
- `valid.csv`：结构相同，用来在本地模拟评分。

输出不是每个测试样本的预测，而是一份两行的 `submission.csv`：

| id | rows | columns |
| --- | --- | --- |
| public | 选中的行号，空格分隔 | 选中的列名，空格分隔 |
| private | 选中的行号，空格分隔 | 选中的列名，空格分隔 |

也就是说，我们要回答两个问题：**用哪几行？用哪几列？**

### 1.3 评价标准

评分公式分成两部分，最后相乘。第一部分是「省了多少数据」，叫 `size_score`：

```text
size_score = 0.5 * log(2000 / 行数) / log(2000)
           + 0.5 * log(30   / 列数) / log(30)
```

行数和列数越少，两个对数项越大，`size_score` 越高。用满 2000 行时第一项是 `log(1)=0`，用满 30 列时第二项也是 0，所以**全量数据的 size_score 恰好是 0**。

第二部分是「模型还准不准」，叫 `factor`：

```text
factor = clip((F1 - 0.80) / 0.10, 0, 1)
```

F1 低于 0.80 时 `factor` 是 0，整个成绩归零；F1 高于 0.90 时 `factor` 是 1；中间线性过渡。最终分数是 `size_score * factor`，越高越好。

这个设计非常巧妙：它逼着我们在「少用数据」和「保持准确」之间找平衡。只交一行一列，`size_score` 很高，但 F1 大概率崩掉，乘 0 还是 0；交全量数据，F1 再高，乘 0 也是 0。

---

## 二、解题过程

整场比赛我们做的事情可以概括成一句话：**把评分程序原样复现到本地，然后在这套评分函数上做子集搜索。** 下面按 EDA、清洗、搜索与验证、指标报告四步展开。

### 2.1 数据探索（EDA）

我们首先把训练集和验证集读进来，确认行列规模、缺失情况和标签分布：

```python
import numpy as np
import pandas as pd

base = "/kaggle/input/competitions/smallest-training-set-ua-prep"
train = pd.read_csv(f"{base}/train.csv")
valid = pd.read_csv(f"{base}/valid.csv")

# 先看规模：行数决定了 size_score 的分母，必须心里有数
print(train.shape, valid.shape)
print(train.columns.tolist())

# 每个特征列有多少个不同取值
# 只有 1 个取值的列对决策树没有任何用处，可以直接不选
nunique = train.drop(columns="label").nunique().sort_values()
print(nunique.head(10))

# 缺失率：评分程序用中位数填补，缺失太多的列信息量会打折
missing = train.isna().mean().sort_values(ascending=False)
print(missing.head(10))

# 标签分布：如果正负样本不平衡，F1 会比准确率更敏感
print(train["label"].value_counts(normalize=True))
```

这一步得到三个结论。第一，训练集是 2000 行、30 个特征，和评分公式里的分母完全对得上。第二，特征列里可能存在常量列，这类列选进去只会白白增加列数、拉低 `size_score`。第三，标签是二分类，且评分用的是 F1，所以我们对少数类的召回要格外关注。

我们接着看验证集和训练集的分布是否一致。这一步很重要：如果两份数据分布差得远，本地 `evaluate` 的分数就不能代表隐藏测试集的表现。

```python
# 对每个特征比较训练集和验证集的均值，差异过大说明分布可能漂移
summary = pd.DataFrame({
    "train_mean": train.drop(columns="label").mean(),
    "valid_mean": valid.drop(columns="label").mean(),
})
summary["gap"] = (summary["train_mean"] - summary["valid_mean"]).abs()
print(summary.sort_values("gap", ascending=False).head())
```

我们关心的不是小数点后的细微差别，而是有没有某一列在两份数据里量级完全不同。如果有，那这一列在验证集上的表现就不可信，搜索时应当谨慎。

### 2.2 数据清洗

这个比赛的数据清洗和平时不太一样：**我们不能改变评分程序内部的填补逻辑**，因为那段代码是固定的。我们能做的清洗只有两类。

第一类是**去掉无效列**。常数列、几乎全空的列、以及和标签完全无关的列，选进去只会增加列数。我们用一个很轻的规则筛一遍：

```python
feature_cols = [c for c in train.columns if c != "label"]

# 规则 1：取值只有一个的列，决策树永远切不出信息增益
constant_cols = [c for c in feature_cols if train[c].nunique(dropna=True) <= 1]

# 规则 2：缺失率超过一半的列，中位数填补后大部分是同一个值
too_missing = [c for c in feature_cols if train[c].isna().mean() > 0.5]

# 规则 3：与标签的相关系数几乎为 0 的列，先放进观察名单而不是直接删
corr = train[feature_cols + ["label"]].corr()["label"].drop("label")
weak_cols = corr[corr.abs() < 0.01].index.tolist()

print("常数列:", constant_cols)
print("缺失过多:", too_missing)
print("弱相关:", weak_cols)
```

第二类是**处理标签缺失**。评分程序直接拿 `train.loc[rows, "label"]` 去训练，如果选中的行里标签是空的，程序会报错。所以我们在挑选行号之前，先把标签为空的行排除掉：

```python
# 只保留标签非空的行，并重新拿到它们在原表中的行号
valid_rows = train.index[train["label"].notna()].to_numpy()
print("可用行数:", len(valid_rows))

# 注意：提交的是原始行号，不是重排后的位置
# 所以后面所有操作都要基于 train.index，不能 reset_index
```

这里有一个容易踩的坑：如果我们中途 `reset_index(drop=True)`，行号就和原始 CSV 对不上了，提交上去评委会在错误的数据上训练。我们全程保留原始索引，就是为了避免这个问题。

### 2.3 模型训练与超参数调优

这个比赛没有「模型训练」的传统含义，因为模型是主办方定死的。我们首先要做的，是把评分程序一字不差地搬到本地：

```python
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

def evaluate(rows, columns):
    # 这就是评分程序用的固定模型：中位数填补 + 决策树
    # class_weight="balanced" 让少数类在划分时获得更高权重
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
    ])
    model.fit(train.loc[rows, columns], train.loc[rows, "label"])

    # 用 valid.csv 模拟隐藏测试集
    f1 = f1_score(valid["label"], model.predict(valid[columns]))

    # 省数据的得分：行、列各占一半权重
    size_score = (0.5 * np.log(2000 / len(rows)) / np.log(2000)
                  + 0.5 * np.log(30 / len(columns)) / np.log(30))
    # 准确率门槛：F1 0.80 以下不得分，0.90 以上拿满
    factor = np.clip((f1 - 0.80) / 0.10, 0, 1)
    return size_score * factor, f1, size_score
```

有了这个函数，问题就变成了一个**离散优化问题**：在 30 个列、2000 行的组合空间里找分数最高的子集。我们采用贪心前向搜索，一次只加一列，选让分数提升最大的那一列：

```python
def greedy_columns(max_cols=30):
    chosen = []
    best_score = 0.0
    history = []

    for _ in range(max_cols):
        candidates = []
        for c in feature_cols:
            if c in chosen:
                continue
            # 暂时用全部可用行，先只比较列的作用
            s, f1, size = evaluate(valid_rows, chosen + [c])
            candidates.append((s, f1, c))

        # 按分数排序，取这一轮最好的列
        candidates.sort(reverse=True)
        s, f1, c = candidates[0]
        if s <= best_score:      # 再加列只会让 size_score 变小，收益为负就停
            break
        chosen.append(c)
        best_score = s
        history.append((len(chosen), c, s, f1))
        print(f"加入 {c}: 分数={s:.4f}, F1={f1:.4f}")

    return chosen, history
```

我们其次处理行。行数和列数在公式里是对称的，但作用不同：列决定了模型能看见什么信息，行决定了模型能见多少样本。我们的做法是从全量行开始，每次按固定比例减少行数，只要 F1 还能保持在 0.90 以上就继续减：

```python
def shrink_rows(columns, min_rows=10):
    # 从全量可用行开始，每次保留 80%
    rows = valid_rows.copy()
    best_rows, best_score = rows, evaluate(rows, columns)[0]

    while len(rows) > min_rows:
        keep = max(min_rows, int(len(rows) * 0.8))
        # 随机抽样，但要固定种子，保证结果可复现
        rng = np.random.default_rng(42)
        rows = rng.choice(valid_rows, size=keep, replace=False)

        s, f1, size = evaluate(rows, columns)
        print(f"行数={keep}, F1={f1:.4f}, 分数={s:.4f}")

        # F1 一旦跌破 0.90，factor 就开始打折，此时停手
        if f1 < 0.90:
            break
        if s > best_score:
            best_rows, best_score = rows, s
    return best_rows, best_score
```

这里的关键判断是「什么时候停」。因为 `factor` 在 F1 高于 0.90 时恒等于 1，所以在这个区间里继续减行是纯赚；一旦跌破 0.90，每减一行都在拿准确率换尺寸分，就要看边际收益了。我们没有在排行榜上反复试提交，而是把 `valid.csv` 当作唯一的裁判，这样可以避免过拟合公共榜。

### 2.4 模型测试与指标报告

最后我们把选出来的行和列写进提交文件。格式看着简单，但 `rows` 和 `columns` 都必须是空格分隔的字符串，写错了评委解析不出来：

```python
chosen_cols, _ = greedy_columns()
chosen_rows, final_score = shrink_rows(chosen_cols)

# 行号必须是原始索引，并转成字符串
row_text = " ".join(str(r) for r in chosen_rows)
column_text = " ".join(chosen_cols)

submission = pd.DataFrame({
    "id": ["public", "private"],
    "rows": [row_text, row_text],
    "columns": [column_text, column_text],
})
submission.to_csv("submission.csv", index=False)
print(submission.shape)          # 必须是 (2, 3)
print(len(chosen_rows), "行", len(chosen_cols), "列")
```

关于结果，材料里的基线 notebook 说得很直白：**它用满全部 2000 行和 30 列，所以分数是 0**。这不是模型不行，而是 `size_score` 的两个对数项都变成了 `log(1)=0`。这也从反面说明，这个比赛真正的抓手是「减法」而不是「加法」。

我们可以用公式估一下不同选择的量级。假设 F1 能保持在 0.90 以上、`factor=1`：

- 用 100 行、5 列：行项约 `0.5*log(20)/log(2000)=0.197`，列项约 `0.5*log(6)/log(30)=0.263`，合计约 0.46；
- 用 50 行、3 列：行项约 `0.5*log(40)/log(2000)=0.253`，列项约 `0.5*log(10)/log(30)=0.329`，合计约 0.58；
- 用 20 行、2 列：行项约 `0.5*log(100)/log(2000)=0.303`，列项约 `0.5*log(15)/log(30)=0.398`，合计约 0.70。

可以看到，想在 0.9 以上，行和列都得压到很小的数量，而这时候决策树很容易过拟合，F1 会掉。**真正的难点不是算分，而是在极小的数据上还能保住 F1。**

我们最后还做了一次「压力测试」：把选中的行打乱顺序重新训练，看 F1 是否稳定。如果换个随机种子分数就大幅波动，说明选中的样本太特殊，换到隐藏测试集上大概率会崩。这个检查没有写在提交文件里，但它是我们判断「这套子集能不能信」的主要依据。

---

## 三、总结

本文中我们完成了 Smallest Training Set 比赛的完整流程。我们首先通过 EDA 确认了数据规模是 2000 行、30 个特征、二分类标签，并找出了常数列、高缺失列和弱相关列。其次我们做了非常克制的数据清洗：只剔除无效列和标签缺失的行，并且全程保留原始行号，避免提交时行号错位。接着我们把评分程序原样复刻到本地，用贪心前向搜索挑列、用逐步减半的方式挑行，把「选数据」变成了一个可以量化比较的优化问题。最后我们按公式估算了不同子集规模的得分量级，并指出 F1 0.90 这条线是整场比赛的分水岭。

这个比赛给我们最大的启发是：**模型不是唯一的变量，数据本身就是超参数**。平时我们习惯把训练集当成理所当然的全集，但当我们被迫做减法时，才会认真去想每一行、每一列到底贡献了什么。这种「数据预算」的意识，在标注成本很高的真实项目里，比多调两个超参数值钱得多。
