# Reducing Commercial Aviation Fatalities

## 一、比赛简介

### 1.1 比赛背景

商业航空的事故率已经很低，但每一次事故的代价都极其惨重。研究人员发现，很多事故并不是飞机本身出了故障，而是飞行员在关键时刻的**认知状态**出了问题：注意力过度集中在一个仪表上、被突然的声响吓到、或者因为思考别的事情而漏看了警告。如果能在飞行过程中实时判断飞行员处于哪种状态，就有可能在危险发生前提醒他，从而减少事故。

这个比赛的数据来自飞行模拟器实验。实验人员让飞行员佩戴多种生理传感器，记录他们在模拟飞行时的身体信号，同时人为制造三类会干扰注意力的情境：

- **Channelized Attention（CA，通道化注意力）**：让被试专注于一个任务，忽略其他所有信息，比如玩一个很投入的拼图游戏；
- **Diverted Attention（DA，注意力转移）**：让被试一边监控仪表，一边分心去解数学题；
- **Startle/Surprise（SS，惊吓/意外）**：让被试看带有突然惊吓镜头的电影片段。

除了这三种“被干扰”的状态，还有一种是飞行员的**基线状态（baseline）**，也就是没有受到特殊干扰时的正常状态。

比赛要求参赛者根据飞行员在每一时刻的生理信号，判断他当前处于上述四种状态中的哪一种。这些信号包括脑电图（EEG）、心电图（ECG）、呼吸（respiration）和皮肤电反应（GSR）。用大白话说，就是**通过身体信号去“读心”**，看飞行员现在是不是走神了、被吓到了，还是正常飞行。

### 1.2 比赛目标

训练集和测试集都是表格数据，每一行代表一个很短的时间片，列可以分成三类：

- **标识列**：`crew`（机组编号）、`experiment`（实验类型）、`time`（实验内的时间）、`seat`（座位，0 表示左座、1 表示右座）；
- **生理信号列**：20 个 `eeg_*` 脑电通道，加上 `ecg`（心电）、`r`（呼吸）、`gsr`（皮肤电）；
- **标签列**：`event`，取值为 `A`、`B`、`C`、`D`。

标签和状态的对应关系是：

| 标签 | 含义 |
| --- | --- |
| A | 基线状态（baseline） |
| B | 惊吓/意外（SS） |
| C | 通道化注意力（CA） |
| D | 注意力转移（DA） |

训练集一共有 4,867,421 行、28 列，内存占用约 464.2 MB，并且**没有缺失值**。测试集只有特征列，没有 `event`，我们要为测试集的每一行输出四个类别的概率。

需要注意的是，训练集里的实验类型有 `CA`、`DA`、`SS` 三种，而测试集对应的是 `LOFT`（Line Oriented Flight Training，完整航线飞行训练）。这意味着测试数据和训练数据来自不同的实验场景，模型不能简单地记住“某个实验对应某个标签”，而必须真正从生理信号里学到规律。

### 1.3 评价标准

比赛使用**多分类对数损失（multi-class log loss）**，也叫交叉熵损失。它的计算方式是：

$$
\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{4} y_{ik}\log(p_{ik})
$$

其中 $N$ 是样本数，$y_{ik}$ 是第 $i$ 个样本是否属于第 $k$ 类（属于为 1，否则为 0），$p_{ik}$ 是模型预测它属于第 $k$ 类的概率。

这个指标的含义是：模型给“正确答案”的概率越高，损失越小；如果模型非常自信地给出了错误答案，损失会被罚得很重。所以光把类别猜对还不够，**概率要校准得好**。分数越低越好，满分是 0，实际比赛中很难接近 0。

---

## 二、解题过程

整场比赛我们走的是“**先把数据看清楚，再按飞行员做归一化，最后用 LightGBM 做多分类**”这条路线。下面按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来讲。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，并且在读取时就指定好数据类型。数据有将近五百万行，如果用默认的 `float64` 会吃掉大量内存，所以脑电、心电、呼吸、皮肤电这些信号列统一用 `float32`，`crew` 和 `seat` 用 `int8`：

```python
import pandas as pd
import numpy as np

# 先声明各列的数据类型：小整数用 int8，信号用 float32，能省一半以上内存
dtypes = {
    "crew": "int8",
    "experiment": "category",
    "time": "float32",
    "seat": "int8",
    "ecg": "float32",
    "r": "float32",       # r 是呼吸信号
    "gsr": "float32",     # gsr 是皮肤电反应
    "event": "category",
}

# 20 个脑电通道名字比较长，用循环批量加进 dtypes
eeg_cols = [
    "eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6",
    "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3",
    "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4",
    "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2",
]
for col in eeg_cols:
    dtypes[col] = "float32"

train_df = pd.read_csv("../input/train.csv", dtype=dtypes)
test_df = pd.read_csv("../input/test.csv", dtype=dtypes)

# 确认行列数：训练集 4,867,421 行、28 列，且没有缺失值
print(train_df.shape)
print(train_df.info())
```

`info()` 的输出告诉我们两件重要的事：第一，训练集有 4,867,421 条记录，数据量很大；第二，所有列都是非空的，也就是说这份数据本身不需要做缺失值填补。

其次，我们统计了标签的分布，并把实验类型和标签做了交叉表：

```python
# 四种状态各有多少条样本
print(train_df["event"].value_counts())

# 行是实验类型，列是状态，看每个实验里会出现哪些状态
print(pd.crosstab(train_df["experiment"], train_df["event"]))
```

交叉表的结果非常有信息量：

```
event          A        B        C        D
experiment
CA          5690        0  1652686        0
DA       1423064        0        0   235329
SS       1420055   130597        0        0
```

可以看到，`CA` 实验里只出现 A（基线）和 C（通道化注意力），`DA` 实验里只出现 A 和 D，`SS` 实验里只出现 A 和 B。这说明**实验类型几乎决定了标签的取值范围**。如果测试集也是同样的实验类型，模型只要看 `experiment` 这一列就能大幅提分；但测试集是 `LOFT`，所以这条“捷径”在正式评测中并不成立。这个发现提醒我们：后面评估模型时，不能盲目相信“带上实验类型就能得高分”，否则线下分数会严重高估线上表现。

接着，我们画了标签计数图和脑电信号按状态分组的箱线图：

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 看四个状态是否均衡
sns.countplot(x="event", data=train_df)
plt.title("Target repartition")
plt.show()

# 对每个脑电通道，看不同状态下的取值分布有没有差别
# 数据太大，只随机抽 5 万行画图，既够看趋势又不至于卡死
for eeg in eeg_cols:
    sns.boxplot(x="event", y=eeg, data=train_df.sample(50000), showfliers=False)
    plt.title(eeg)
    plt.show()
```

从箱线图里我们看到，不同状态之间，部分脑电通道的中位数和四分位范围确实有可见差异，但重叠也很多。这说明单靠某一个通道很难把状态分开，必须把多个通道组合起来，交给树模型去自动寻找组合规律。

我们还检查了 `time` 这一列。它在训练集和测试集里的分布并不一致，而且这个“时间”是模拟器内部的时间，和真实实验时间没有对应关系，因此我们最后**没有把 `time` 放进特征**。

另外，我们统计了飞行员人数：把 `crew` 和 `seat` 组合起来可以唯一标识一位飞行员，训练集里一共有 18 位不同的飞行员。这个数字对下一步的归一化很关键。

### 2.2 数据清洗

这份数据的“清洗”主要不是补缺失值，而是**处理不同飞行员之间的个体差异**。

生理信号的绝对幅度因人而异：有的人脑电整体偏大，有的人皮肤电基线偏低。如果不做处理，模型可能学到的是“这位飞行员的个人特征”，而不是“这种认知状态的特征”。材料里的做法是：**对每一位飞行员、每一个信号列，单独做 Min-Max 归一化**，把每个人的信号都缩放到 0 到 1 之间。

我们首先构造一个 `pilot` 列来唯一标识飞行员，然后按飞行员分组做缩放：

```python
from sklearn.preprocessing import MinMaxScaler

# 用 crew 和 seat 组合出飞行员编号
# 乘 100 是为了把两个数拼成一个不会冲突的整数
train_df["pilot"] = 100 * train_df["seat"] + train_df["crew"]
test_df["pilot"] = 100 * test_df["seat"] + test_df["crew"]

# 需要归一化的信号列：20 个脑电 + 心电 + 呼吸 + 皮肤电
features_n = eeg_cols + ["ecg", "r", "gsr"]

def normalize_by_pilots(df):
    """按飞行员分组做 Min-Max 缩放，消除个人基线差异"""
    for pilot in df["pilot"].unique():
        ids = df[df["pilot"] == pilot].index
        scaler = MinMaxScaler()
        # 只对该飞行员的这些行、这些信号列做缩放
        df.loc[ids, features_n] = scaler.fit_transform(df.loc[ids, features_n])
    return df

train_df = normalize_by_pilots(train_df)
test_df = normalize_by_pilots(test_df)
```

这里有一个细节：训练集和测试集是**分别**做归一化的，而不是用训练集的参数去变换测试集。原因是测试集里可能出现训练集没有的飞行员，或者同一位飞行员在不同实验里的信号范围不同。按人分别缩放，至少能保证每个人自己的信号被拉到同一量纲上。

除了归一化，我们还把标签从字母转成整数，因为 LightGBM 的多分类目标要求标签是 0 到 3 的整数：

```python
# LightGBM 需要 0、1、2、3 这样的整数标签
label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
train_df["event"] = train_df["event"].map(label_map)

# 最终喂给模型的特征：机组、座位，加上 23 个归一化后的生理信号
features = ["crew", "seat"] + features_n
```

我们没有使用 `experiment` 作为特征。虽然它在训练集里和标签高度相关，但测试集是全新的 `LOFT` 实验，把实验类型喂进去只会让模型学到无法迁移的规律。

### 2.3 模型训练与超参数调优

我们接着把训练集按 8:2 切成训练部分和验证部分。这里用固定的随机种子，保证每次实验的划分一致，方便比较不同模型：

```python
from sklearn.model_selection import train_test_split

# 80% 训练、20% 验证；random_state 固定，结果可复现
train_part, val_part = train_test_split(
    train_df, test_size=0.2, random_state=420
)
print("训练样本数：", train_part.shape[0])
# 输出：训练样本数：3893936
```

切完之后，训练部分有 3,893,936 条样本，验证部分有 973,485 条。

模型我们选的是 **LightGBM**。它是一种梯度提升树（GBDT）的实现，可以理解成“一棵接一棵地训练决策树，每一棵新树都专门去修正前面所有树犯的错误”。它对表格数据非常友好，训练速度快，还能直接输出多分类概率。

我们使用的参数如下：

```python
import lightgbm as lgb

params = {
    "objective": "multiclass",   # 多分类任务
    "num_class": 4,              # 一共 A、B、C、D 四类
    "metric": "multi_error",     # 训练过程中监控分类错误率
    "num_leaves": 30,            # 每棵树最多 30 个叶子，控制模型复杂度
    "min_child_weight": 50,      # 一个叶子至少要有 50 的样本权重，防止过拟合
    "learning_rate": 0.1,        # 学习率，每棵树只走一小步
    "bagging_fraction": 0.7,     # 每轮只随机用 70% 的样本建树
    "feature_fraction": 0.7,     # 每轮只随机用 70% 的特征建树
    "bagging_seed": 420,
    "verbosity": -1,
}

lg_train = lgb.Dataset(train_part[features], label=train_part["event"])
lg_val = lgb.Dataset(val_part[features], label=val_part["event"])

# 最多训练 1000 轮；如果验证集指标连续 50 轮不提升就提前停止
model = lgb.train(
    params,
    lg_train,
    num_boost_round=1000,
    valid_sets=[lg_val],
    early_stopping_rounds=50,
    verbose_eval=100,
)
```

训练日志里，验证集的 `multi_error` 随着轮数增加一路下降：

```
[100]  valid_0's multi_error: 0.0821512
[200]  valid_0's multi_error: 0.068337
[300]  valid_0's multi_error: 0.0598006
[400]  valid_0's multi_error: 0.0539228
[500]  valid_0's multi_error: 0.0499443
[600]  valid_0's multi_error: 0.0467958
[700]  valid_0's multi_error: 0.044165
[800]  valid_0's multi_error: 0.0419852
[900]  valid_0's multi_error: 0.0399277
[1000] valid_0's multi_error: 0.0383067
```

到第 1000 轮时，分类错误率降到了约 3.83%，而且还没有触发提前停止，说明模型仍有继续提升的空间。不过我们为了控制训练时间，先停在了 1000 轮。

关于超参数调优，我们也试过用网格搜索去调逻辑回归、随机森林和 XGBoost，但在这份近五百万行的数据上，每一组参数都要重新训练很久，网格搜索的成本非常高。相比之下，LightGBM 在默认参数附近就已经能给出比较稳定的结果，所以我们把主要精力放在了特征处理和验证方式上，而不是无休止地搜参数。

为了确认哪些信号真正有用，我们还画了特征重要性图：

```python
# 画出 LightGBM 认为最重要的特征
fig, ax = plt.subplots(figsize=(12, 10))
lgb.plot_importance(model, height=0.8, ax=ax)
ax.grid(False)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.title("Importance of the Features of our LightGBM Model")
plt.show()
```

从重要性排序看，若干脑电通道排在前面，`ecg`、`r`、`gsr` 也有贡献，而 `crew` 和 `seat` 的贡献相对有限。这符合直觉：真正区分认知状态的是大脑和自主神经系统的活动，而不是飞行员坐在哪个座位。

### 2.4 模型测试与指标报告

训练完成后，我们用验证集计算比赛真正关心的指标——多分类对数损失：

```python
from sklearn.metrics import log_loss, confusion_matrix

# 用验证集特征预测四类概率，每一行四个数，加起来等于 1
pred_val = model.predict(val_part[features], num_iteration=model.best_iteration)

# 计算多分类 log loss，越低越好
val_loss = log_loss(val_part["event"].values, pred_val)
print("Log loss on validation data :", round(val_loss, 3))
# 输出：Log loss on validation data : 0.108
```

验证集上的 log loss 是 **0.108**。这个数字可以这样理解：如果模型完全瞎猜，四类等概率时 log loss 约为 1.386；0.108 说明模型给正确答案的概率已经相当高。

我们接着看了混淆矩阵，确认模型主要错在哪里：

```python
# 取概率最大的类别作为预测类别，再和真实标签对比
conf_mat = confusion_matrix(
    pred_val.argmax(axis=1),   # 模型预测的类别
    val_part["event"].values   # 真实类别
)
print(conf_mat)
```

混淆矩阵显示，绝大多数样本都落在对角线上，错误主要集中在“基线状态”和某些干扰状态之间。这也是合理的：基线状态本身就没有强烈的生理特征，容易和其他状态混淆。

最后，我们把模型应用到测试集，并按比赛要求生成提交文件。提交文件需要 `id` 列和 `A`、`B`、`C`、`D` 四列概率：

```python
# 对测试集预测四类概率
pred_test = model.predict(test_df[features], num_iteration=model.best_iteration)

# 拼成提交表：第一列是行号 id，后面四列是概率
submission = pd.DataFrame(
    np.concatenate((np.arange(len(test_df))[:, None], pred_test), axis=1),
    columns=["id", "A", "B", "C", "D"],
)
submission["id"] = submission["id"].astype(int)

# 保存到本地，之后上传到 Kaggle
submission.to_csv("submission.csv", index=False)
print(submission.head())
```

这样，一条从原始生理信号到最终概率提交的完整流程就跑通了。

---

## 三、总结

本文中我们做了一次从生理信号预测飞行员认知状态的完整尝试。我们首先阅读了比赛数据，确认训练集有 4,867,421 行、28 列、没有缺失值，并通过交叉表发现训练集里的实验类型和标签几乎是一一对应的，而测试集是全新的 `LOFT` 实验，因此我们刻意没有把 `experiment` 当作特征，避免模型学到无法迁移的“捷径”。

其次，我们针对不同飞行员之间信号幅度差异很大的问题，按 `crew` 和 `seat` 组合出飞行员编号，对每位飞行员的 20 个脑电通道以及心电、呼吸、皮肤电分别做 Min-Max 归一化，让模型关注状态差异而不是个人基线。

接着，我们把数据按 8:2 切成训练集和验证集，使用 LightGBM 做四分类，参数上采用 30 个叶子、学习率 0.1、行采样和列采样各 0.7，并设置 50 轮早停。训练过程中验证集的分类错误率从 8.2% 一路降到 3.83%。

然后，我们在验证集上计算了比赛指标 log loss，得到 0.108，并用混淆矩阵检查了错误分布；最后把模型应用到测试集，输出了包含 `id` 和四类概率的提交文件。

回顾整个方案，优点是比较直接、训练速度可以接受、结果也容易解释；不足是我们把每一行当成独立样本，**没有利用时间上的连续性**。飞行员的认知状态不会在一秒之内突然跳变，前后时刻的信号显然有关联。如果继续改进，可以把相邻若干秒的信号拼成序列，用 LSTM 或一维卷积网络来建模，或者对 LightGBM 的预测结果做时间上的平滑。此外，材料里还有参赛者尝试过逻辑回归、LDA、QDA 等模型，验证集 log loss 分别在 0.92、0.93、3.18 附近，都明显不如树模型，这也从侧面说明非线性的组合特征对这份数据很重要。

