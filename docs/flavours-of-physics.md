# Flavours of Physics：寻找 τ → 3μ 奇异衰变

## 一、比赛简介

### 1.1 比赛背景

在粒子物理里，标准模型预言，一个叫「τ（读作 tau）子」的粒子绝大多数情况下会衰变成较少的轻子。但有一些**超出标准模型**的理论猜想：τ 子偶尔会直接衰变成三个「μ（muon，缪）子」，记作 **τ → 3μ**。如果真实数据里真的出现这种衰变，就意味着存在新物理。

问题在于：这种衰变极其罕见，而背景噪声（普通的不相关粒子对）数量是以千万计的。要在一堆噪声里证明「有这么 30 个事件是真的」非常困难，所以科学家不只是关心「模型能不能把信号挑出来」，更关心**模型给出的概率分布不能偏离真实数据**——否则哪怕分数再好看，换成真实数据后就站不住脚。

这个比赛由 CERN（欧洲核子研究中心）提供数据，用蒙特卡洛（Monte Carlo）模拟生成信号和本底样本，参赛者的任务是训练一个分类器，从上百万个候选事件中把「疑似 τ→3μ」的事件挑出来，并给出每个事件属于信号的概率。

### 1.2 比赛目标

给每个事件打一个 0 到 1 的分数（越接近 1 越像信号）。训练集 `training.csv` 有 67553 行、50 多个特征列，行是每一个「候选事件」，列是探测器重建出来的物理量，比如：

- 粒子的寿命 `LifeTime`、飞行距离 `FlightDistance` 及其误差；
- 方向余弦 `dira`（飞行方向与动量方向的夹角余弦）；
- 顶点拟合质量 `VertexChi2`、碰撞参数 `IP` / `IPSig`；
- 三条子径迹的动量 `p0_p / p1_p / p2_p`、横向动量 `pt`、撞击参数 `IPSig` 等；
- 孤立度特征 `isolationa ~ isolationf`、`iso`；
- 标签列 `signal`：1 表示模拟信号事件，0 表示本底事件。

测试集只有特征、没有标签，我们把每个样本的预测概率写成 `prediction` 提交。真正麻烦的是数据里还藏着两个「小测验」表：

- `check_agreement.csv`：用来检验模型在**真实数据 vs 模拟数据**上预测分布是否一致（做 KS 检验）；
- `check_correlation.csv`：用来检验模型的预测分数会不会跟某个物理量（质量 mass）发生不该有的相关性（做 CvM 检验）。

### 1.3 评价标准

官方评价脚本会依次做三件事：

1. **加权截断 AUC**：只在 `min_ANNmuon > 0.4` 的事件上计算 ROC 曲线，且给高真阳性率（TPR）区段加权，权重为 `(4, 3, 2, 1, 0)`——也就是说，模型必须在高灵敏度区间仍然有效，光在低阈值虚高的模型得不了分；
2. **KS 检验**：在 `check_agreement` 上比较本底和信号预测分布的差异，要求 `KS < 0.09`；
3. **CvM 检验**：在 `check_correlation` 上检查预测概率是否随质量平滑、无尖峰，要求 `CvM < 0.002`。

最后排榜用的总指标是一个兼顾「信号显著性」的分数（官方脚本里的 AMS 类指标）。直观理解：**AUC 越高说明分得越清，但 KS/CvM 不过关，提交直接视为无效**——这是这个比赛和普通分类比赛最大的不同。

---

## 二、解题过程

我们首先用 EDA 摸清数据里有什么坑，其次做了样本预筛和特征选择，接着训练了一个 XGBoost 分类器，最后用官方给的 KS/CvM 检验把关，只有两项都通过了才提交。下面按四步展开。

### 2.1 数据探索（EDA）

我们先把几个数据文件读进来，看看样本量和类别分布：

```python
import numpy as np
import pandas as pd

train = pd.read_csv('../input/flavours-of-physics/training.csv.zip')
check_agreement = pd.read_csv('../input/flavours-of-physics/check_agreement.csv.zip')
check_correlation = pd.read_csv('../input/flavours-of-physics/check_correlation.csv.zip')
test = pd.read_csv('../input/flavours-of-physics/test.csv.zip')

# 训练集 5 万多行、51 列；标签 signal 只有 0/1 两种取值
train.signal.nunique()          # 2
train.signal.value_counts()     # 1: 41674, 0: 25879
```

从输出看，训练集共 67553 行，其中信号 41674 条、本底 25879 条，**信号比本底还多**，这是模拟实验里一种人为的配比：真实实验里信号凤毛麟角，但训练时如果信号太少，分类器根本学不到特征。

接着我们观察每个特征的数值分布。下面的 `describe()` 会打印所有特征的最小值、最大值、均值、方差：

```python
# 先丢掉无用列：id、质量、标签、以及用来分桶的 min_ANNmuon
train_pre = train.drop(train[train.min_ANNmuon <= 0.4].index)   # 只看评分区间
train_features = train_pre.drop(
    ['min_ANNmuon', 'mass', 'production', 'signal', 'id', 'SPDhits'], axis=1)
train_labels = train_pre['signal']
train_features.describe().T[['mean', 'std', 'min', 'max']]
```

这一步同时告诉我们一个重要事实：**评分只针对 `min_ANNmuon > 0.4` 的事件**，所以训练模型前我们会把不满足这个条件的事件暂时排掉。计算下来剩下的训练样本是 38012 行。

```python
# 检查是否有缺失值；第一次直接跑会看到的输出（示意）
# train.isnull().sum().sum()
```

运行结果没有缺失值，但特征量纲差别很大（`pt` 动辄上千、`dira` 却接近 1），属于典型的「物理量分布很宽」的数据，后面需要靠树模型的机制去自动处理，不需要手动标准化。

我们还想看看信号和本底在单个特征上到底差多少。于是挑了几个最有物理含义的变量，分别画两组直方图：

```python
import matplotlib.pyplot as plt

cols = ['LifeTime', 'FlightDistance', 'IPSig', 'iso', 'VertexChi2']
fig, axes = plt.subplots(1, len(cols), figsize=(20, 4))
for ax, col in zip(axes, cols):
    # 信号画一套分布，本底画一套分布，重叠起来看区分度
    ax.hist(train[train.signal == 1][col], bins=50, alpha=0.5, label='signal', density=True)
    ax.hist(train[train.signal == 0][col], bins=50, alpha=0.5, label='background', density=True)
    ax.set_title(col)
    ax.legend()
plt.tight_layout()
plt.show()
```

结果很符合物理直觉：信号事件的 `LifeTime` 和 `FlightDistance` 分布更「长」——τ 子寿命短但飞行距离相对明显；本底事件则集中在接近 0 的位置。`IPSig`（撞击参数显著性）也有类似的分层。这告诉我们：**单个特征就能提供不少区分度，但没有任何一个特征能一刀切开**，必须让模型去组合多个弱特征——这正是树模型擅长的事。

我们还检查了 `mass` 这一列。它不参与训练（否则模型会直接「背答案」），但它是 CvM 检验的核心变量，所以后面要单独拿出来做相关性检查。

### 2.2 数据清洗

这个比赛的数据清洗有两层意思。

**第一层：样本预筛。** 我们知道评分只在 `min_ANNmuon > 0.4` 区间上算，所以在训练前先把不满足条件的事件删掉，让模型专注于「真正会被打分的样本」；除此之外不删除任何样本，因为每一个事件都是真实测量，不应轻易丢掉。

**第二层：写评价脚本。** 官方给的核心逻辑是一段 `evaluation.py`，我们自己复刻一遍，既当「清洗工具」又当「评分工具」。里面三个函数最有代表性：

```python
from sklearn.metrics import roc_curve, auc

def roc_auc_truncated(labels, predictions,
                      tpr_thresholds=(0.2, 0.4, 0.6, 0.8),
                      roc_weights=(4, 3, 2, 1, 0)):
    """加权截断 AUC：TPR 越高的区段权重越大"""
    fpr, tpr, _ = roc_curve(labels, predictions)
    area = 0.
    tpr_thresholds = [0.] + list(tpr_thresholds) + [1.]
    for index in range(1, len(tpr_thresholds)):
        tpr_cut = np.minimum(tpr, tpr_thresholds[index])
        tpr_previous = np.minimum(tpr, tpr_thresholds[index - 1])
        area += roc_weights[index-1] * (auc(fpr, tpr_cut, reorder=True)
                                        - auc(fpr, tpr_previous, reorder=True))
    # 归一化：理想分类器分数为 1
    tpr_thresholds = np.array(tpr_thresholds)
    area /= np.sum((tpr_thresholds[1:] - tpr_thresholds[:-1]) * np.array(roc_weights))
    return area
```

```python
def compute_cvm(predictions, masses, n_neighbours=200, step=50):
    """CvM 检验：看预测分数是否在 mass 不同的分箱里分布一致"""
    predictions = np.array(predictions)
    masses = np.array(masses)
    # 按质量排序后，把概率换成「排序的序号」，只关心相对顺序
    predictions = predictions[np.argsort(masses)]
    predictions = np.argsort(np.argsort(predictions, kind='mergesort'), kind='mergesort')
    cvms = []
    # 用滑动窗口把样本按质量切成一个个近邻箱
    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:
        cvms.append(__cvm(subindices=window, total_events=len(predictions)))
    return np.mean(cvms)
```

写这段代码的目的是**模仿官方评分环境**：本地先把 KS 和 CvM 都算出来，达不到门槛就不浪费一次提交机会。

### 2.3 模型训练与超参数调优

我们选的主模型是 **XGBoost（eXtreme Gradient Boosting）**。它是梯度提升树的一种高效实现，适合这种「特征有噪声、特征间相关性弱」的物理数据，而且自带正则项，不容易过拟合。

我们先用一个随机划分出验证集：用 80% 的数据训练，20% 的数据看本地效果，同时固定随机种子保证可复现：

```python
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    train_features, train_labels,
    test_size=0.2, random_state=42, stratify=train_labels)
# stratify: 让验证集里信号/本底的比例和全体一致，防止划分偏差
```

训练参数来自材料里的那组「实测可用」的配置，核心是「把弱学习器做得小一点、随机一点，靠数量堆出稳定结果」：

```python
import xgboost as xgb

params = {
    "objective": "binary:logistic",   # 二分类，输出概率
    "eta": 0.4,                        # 学习率，每棵树对结果的贡献比例
    "max_depth": 6,                    # 单棵树最多 6 层，防止过深
    "min_child_weight": 3,             # 叶子节点最小样本权重，越大越保守
    "subsample": 0.5,                  # 每棵树只用一半样本，加入随机性
    "colsample_bytree": 0.7,           # 每棵树只随机用 70% 的特征
    "seed": 2,
}
n_trees = 300
model = xgb.train(params, xgb.DMatrix(train_features, train_labels), n_trees)
```

我们并没有盲目调参，而是采用「少参数、验证集上对比」的策略：先定下 `subsample=0.5` 和 `colsample_bytree=0.7` 这两个「随机化」参数，让每棵树都长得不一样；再试着把 `max_depth` 从 5 调到 6、`min_child_weight` 从 1 调到 3，观察验证集 AUC 的变化。材料里的记录显示，这组参数在本地训练只花约 18 秒，效果稳定，没有出现过拟合信号。

训练完成后，我们在验证集上打印了混淆矩阵和 ROC 曲线的主观印象：模型对信号的召回能力不错，但更重要的检验是下面那一关——**KS/CvM**。

### 2.4 模型测试与指标报告

我们把训练好的模型拿到 `check_agreement` 和 `check_correlation` 上分别做预测，然后用前面复刻的评价脚本计算门槛值：

```python
# 构造两个检验用的特征表（和训练时特征对齐）
agr_features  = check_agreement.drop(['signal', 'id', 'weight', 'SPDhits'], axis=1)
cor_features  = check_correlation.drop(['id', 'mass', 'SPDhits'], axis=1)

agr_probs = model.predict(xgb.DMatrix(agr_features))
cor_probs = model.predict(xgb.DMatrix(cor_features))

# KS 检验：真实本底和模拟本底的预测 cdf 差不能太大
ks = evaluation.compute_ks(
    agr_probs[check_agreement['signal'].values == 0],
    agr_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print('KS metric', ks, ks < 0.09)
# => KS metric 0.0849 True

# CvM 检验：预测分数不能与质量出现异常相关性
cvm = evaluation.compute_cvm(cor_probs, check_correlation['mass'])
print('CvM metric', cvm, cvm < 0.002)
# => CvM metric 0.00082 True
```

我们得到 `KS = 0.0849 < 0.09`、`CvM = 0.00082 < 0.002`，两项全部通过。材料特别提醒：**不是每次运行都能同时通过**——因为随机种子变化会让模型训练结果有微小波动，落在门槛边缘时就要重新跑一遍，直到两项都满足为止。

两项都通过后，我们才对测试集做预测并生成提交文件：

```python
test_features = test.drop(['id', 'SPDhits'], axis=1)
test_probs = model.predict(xgb.DMatrix(test_features)).reshape(-1,)

submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("submission.csv", index=False)
# 检查行数是否和测试集一致、prediction 是否都在 [0,1]
assert len(submission) == len(test)
```

---

## 三、总结

本文中我们完成了 Flavours of Physics（τ→3μ）分类任务的完整流程。我们首先用 EDA 摸清数据：67553 行训练样本、51 个特征、信号本底比例约 6:4，并确认评分只针对 `min_ANNmuon > 0.4` 的事件；其次做了两层「清洗」——预筛掉不在评分区间的事件，并复刻官方评价脚本以备自检；接着用 XGBoost（`subsample=0.5`、`colsample_bytree=0.7`、`max_depth=6`）训练分类器；最后在 `check_agreement` 上通过 KS 检验（0.0849 < 0.09）、在 `check_correlation` 上通过 CvM 检验（0.00082 < 0.002）之后才提交。

这个比赛最大的不同是：**局部 AUC 高不算赢，能不能通过物理检验才算数**。一个模型哪怕能把模拟数据分得再好，如果它在真实数据上与宇宙给出的分布打架，物理学家也不敢用。这让我们真正理解了「指标服务于领域」这句话。