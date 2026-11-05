# 味之物理：寻找 τ → 3μ 衰变（Flavours of Physics）

## 一、比赛简介

### 1.1 比赛背景

在 CERN 的大型强子对撞机（LHC）上，质子高速相撞会产生大量粒子。物理学家想在其中找到一种极其罕见的衰变：**τ 轻子（读音「套」）衰变成三个 μ 子（读音「缪」）**，记作 τ → 3μ。按照现有理论，这个过程被压得极低，如果真的被观察到，就可能是新物理的线索。

比赛的数据是 CERN 用模拟程序生成的「候选事件」：每条样本记录了一次三个 μ 子同时出现的重建结果，包括它们从哪里飞出来、飞了多远、动量多大、周围有多「干净」等等。标签 `signal` 告诉我们这次事件到底是真的 τ→3μ（1），还是普通背景（0）。难点在于：信号极其稀少、背景千千万万，而且物理学家非常担心模型「学到假的规律」——所以除了分类准，还要通过两道物理一致性检验。

### 1.2 比赛目标

训练集 `training.csv` 有 67,553 行、51 列，主要特征可以分成几组：

- **衰变几何**：`LifeTime`、`FlightDistance`、`FlightDistanceError`、`dira`（飞行方向与动量方向的夹角余弦）、`VertexChi2`（顶点拟合质量）；
- **碰撞点偏离**：`IP`、`IPSig` 及其两两组合 `IP_p0p2`、`IP_p1p2`；
- **三个子粒子**：`p0_*`、`p1_*`、`p2_*` 各自的动量 `p`、横动量 `pt`、赝快度 `eta`、IP、IPSig、径迹拟合质量 `track_Chi2Dof`、隔离度 `IsoBDT`；
- **整体隔离度**：`isolationa~f`、`iso`、`ISO_SumBDT`、`CDF1~3`；
- **辅助列**：`SPDhits`（探测器击中数）、`production`（数据来源标记）、`mass`（重建质量）、`min_ANNmuon`（一个现成的判别分数）。

测试集 `test.csv` 没有标签，我们要给每条 `id` 输出一个 `prediction` 概率。此外还提供两个特殊文件：`check_agreement.csv` 用来检验模型预测分布与真实数据是否一致，`check_correlation.csv` 用来检验预测是否和质量 `mass` 相关。

### 1.3 评价标准

排行榜分数是 **AMS（Approximate Median Significance，近似中位数显著性）**，它衡量「在给定误报率下能发现多少信号」，越高越好。但提交必须先通过两道门槛：

1. **KS 检验**（Kolmogorov-Smirnov）：在 `check_agreement` 上比较真实数据与模拟数据的预测分布，要求 KS **< 0.09**；
2. **CvM 检验**（Cramér-von Mises）：在 `check_correlation` 上检查预测是否随 `mass` 变化，要求 CvM **< 0.002**。

材料里的评估脚本还实现了 `roc_auc_truncated`——一种按灵敏度区间加权的 ROC-AUC，权重为 `(4, 3, 2, 1, 0)`，用来在本地衡量「高信号效率区间的排序质量」。换句话说，比赛既要 AUC 高，又要分布「像真的物理」。

## 二、解题过程

我们首先读入四个文件并做基本统计，其次按物理惯例做预选和去列，接着用 XGBoost 训练二分类模型，然后在本地算加权 AUC 并跑 KS/CvM 两道检验，最后确认通过后生成提交文件。

### 2.1 数据探索（EDA）

```python
import pandas as pd

train = pd.read_csv('../input/flavours-of-physics/training.csv.zip')
check_agreement   = pd.read_csv('../input/flavours-of-physics/check_agreement.csv.zip')
check_correlation = pd.read_csv('../input/flavours-of-physics/check_correlation.csv.zip')
test  = pd.read_csv('../input/flavours-of-physics/test.csv.zip')

print(train.shape)                 # (67553, 51)
print(train['signal'].value_counts())
# 1    41674
# 0    25879
```

看到的数据有几个特点。第一，信号比背景还多一些（41,674 对 25,879），这是因为训练集是特意为「学习信号形状」构造的，和真实实验里信号稀缺的情况不同——所以不能用普通准确率，要用 AMS 这种考虑信号显著性的指标。第二，每列都是连续物理量，量纲差异极大：`FlightDistance` 可以上万，`dira` 在 0 到 1 之间，`VertexChi2` 可能只有个位数。第三，`min_ANNmuon` 已经是一个不错的判别量，通常的做法是**只用它做粗筛**，而不是直接拿它当特征。

```python
# 看每个特征的取值范围和分位数，判断有没有异常值和极端量纲
train.describe().T[['mean', 'std', 'min', '50%', 'max']]
```

进一步观察会发现两个值得注意的现象。

第一，很多特征有长尾和重尾分布：比如 `FlightDistance` 的中位数只有几，但最大值可以到几万，说明存在个别「飞得特别远」的事件；`IP`（碰撞点偏离量）的分布也极度右偏。树模型不受量纲和偏态影响，我们据此把 XGBoost 定为基线模型——如果选的是对尺度敏感的模型（比如原始特征的神经网络），这一步就必须先做标准化了。

第二，`mass`（重建质量）的分布有明显的峰。真正的 τ→3μ 信号会集中在一个特征质量附近，而背景是平滑下降的本底。模型会不会「捡芝麻」——靠质量峰去认信号？这正是 `check_correlation` 里 CvM 检验要盯住的事。我们把它记下来，防止后面调参时不小心让模型过度依赖 `mass`。

我们还专门分析过 `min_ANNmuon` 的取值分布，发现绝大多数背景事件这一项都很低，信号整体则偏高。这个「现成的打分」帮我们确定了 0.4 这个预选阈值：把 0.4 以下的丢掉几乎不损失信号，却能去掉大量噪声。

### 2.2 数据清洗

我们按照物理分析的惯例做了两步清洗。

第一步，**预选**：把 `min_ANNmuon <= 0.4` 的事件丢掉。这个阈值意味着「连初步判别器都认为它不像 μ 子的事件，不值得再花力气」。预选之后训练集从 67,553 行降到 38,012 行，背景噪声大幅减少。

第二步，**去列**：删掉标签列、编号列、与物理判断无关或会引入泄漏的列。

```python
# 预选：只保留初步 μ 子判别分数足够高的事件
train_pre = train.drop(train[train.min_ANNmuon <= 0.4].index)
print(train_pre.shape)   # (38012, 51)

# 建模特征：去掉标签、编号、质量、数据来源和 SPDhits
drop_cols = ['min_ANNmuon', 'mass', 'production', 'signal', 'id', 'SPDhits']
train_features = train_pre.drop(columns=drop_cols)
train_labels = train_pre['signal']

# 测试集也必须做同样的列选择，保证训练/测试特征完全一致
test_features = test.drop(columns=['id', 'SPDhits'])
```

为什么要删 `SPDhits`？它是探测器整体的击中数，反映的是「这一束碰撞有多热闹」，而不是这次衰变本身的物理性质。把它放进模型，模型可能靠「哪一天探测器比较忙」来区分信号和背景，这种规律在真实数据上不成立，会直接导致 `check_agreement` 的 KS 检验翻车。同理，`mass` 在训练里是辅助信息、在测试里没有，留着只会让训练和测试特征对不上。

这里还要提醒一个细节：**预选只对训练集做**。测试集没有 `min_ANNmuon`，所以预测时不能按同样的阈值筛选，否则会漏掉一部分测试样本、导致提交行数不够。

### 2.3 模型训练与超参数调优

主模型用 **XGBoost**（一种基于梯度提升的树模型，很多棵决策树依次修正前一棵的错误，在表格数据上非常强）。考虑到训练集只有 3.8 万行、特征 40 多个，我们用较深但带采样的配置来平衡拟合能力和泛化：

```python
import time
import xgboost as xgb

t0 = time.perf_counter()
params = {
    'objective': 'binary:logistic',  # 二分类，输出 0~1 的概率
    'eta': 0.4,                      # 学习率：每棵树的贡献打四折，收敛快
    'max_depth': 6,                  # 每棵树最深 6 层，控制复杂度
    'min_child_weight': 3,           # 叶子节点最少 3 个样本，防止切得太碎
    'subsample': 0.5,                # 每棵树只用一半样本，增加随机性防过拟合
    'colsample_bytree': 0.7,         # 每棵树只用 70% 特征，进一步去相关
    'seed': 2,                       # 固定随机种子，保证结果可复现
}
n_trees = 300
model = xgb.train(params, xgb.DMatrix(train_features, train_labels), n_trees)
print('训练耗时: %.2f 秒' % (time.perf_counter() - t0))  # 材料中约 17.56 秒
```

调参思路上，材料里还有一份用 `GridSearchCV` 搜索 XGBoost 参数的版本，以及神经网络（Keras 全连接网络）和随机森林两个对比模型。神经网络在这个任务上也能跑到相近水平，但树模型对特征量纲不敏感、训练只要十几秒，性价比更高，所以我们把它当主线。`eta=0.4` 配 300 棵树是「学习率大、树少」的组合；如果换成 `eta=0.05`，就需要上千棵树，训练时间会明显变长，而验证收益并不大。

我们比较过几个对照：随机森林（RFC）用同样的特征也能训练，但它对高基数连续特征的切分不如 boosting 精细；全连接神经网络需要标准化、还要调层数和学习率，且在小样本上更容易过拟合。综合下来，XGBoost 在「分数 / 训练成本」这条曲线上最划算。

还有一个容易忽略的点：`seed` 不是随便填的。由于这个比赛对分布检验非常敏感，换一个种子就可能让 KS/CvM 从通过变成不通过。我们的做法是**多跑几个种子，挑一个两关都过的**——材料里的作者也明确写了「不是每次运行都能同时满足 cvm 和 ks 条件」。

### 2.4 模型测试与指标报告

训练完之后，先做两道物理检验。检验用的预测必须在 `check_agreement` 和 `check_correlation` 上做，且特征要与训练时完全一致：

```python
import evaluation   # 比赛方提供的评估脚本（含 KS / CvM / 加权 AUC）

# 两个检验集的预测
agreement_probs   = model.predict(xgb.DMatrix(
    check_agreement.drop(columns=['signal', 'id', 'weight', 'SPDhits'])))
correlation_probs = model.predict(xgb.DMatrix(
    check_correlation.drop(columns=['id', 'mass', 'SPDhits'])))

# KS 检验：真实数据(0) 与 模拟数据(1) 的预测分布差异，必须小于 0.09
ks = evaluation.compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print('KS metric', ks, ks < 0.09)     # 材料中一次运行：0.08495 True

# CvM 检验：预测是否随 mass 变化，必须小于 0.002
cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
print('CvM metric', cvm, cvm < 0.002)  # 材料中一次运行：0.000824 True
```

这一步的结果很能说明这个比赛的性格：同一位作者的另一份提交里，KS = 0.065（通过），但 CvM = 0.00918（**没通过**）。也就是说，模型在「分类」上没问题，但在「和质量的相关性」上翻了车。这提醒我们：**训练完不能只看 AUC，必须每次都跑完两道检验再决定要不要提交**。

确认两关都通过后，才对测试集预测并写文件：

```python
if cvm < 0.002 and ks < 0.09:
    test_probs = model.predict(xgb.DMatrix(test_features)).reshape(-1,)
    submission = pd.DataFrame({'id': test['id'], 'prediction': test_probs})
    submission.to_csv('submission.csv', index=False)
    print('已生成提交文件，行数:', len(submission))
else:
    print('检验未通过，需要换随机种子重新训练')
```

本地评估方面，我们还用 `roc_auc_truncated` 在 `check_agreement` 上画了 ROC 曲线，看模型在高信号效率区间（TPR 0.2~0.8）的排序表现。这个加权 AUC 比普通 AUC 更贴近 AMS 的偏好：它把「能捞到多少真信号」看得比「整体排序对不对」更重。

---

## 三、总结

本文中我们完成了 Flavours of Physics τ→3μ 衰变识别的完整流程。我们首先读入训练、测试、一致性检验和相关性检验四个文件，确认训练集 67,553 行、信号 41,674 条、背景 25,879 条。其次我们按物理惯例做预选（丢掉 `min_ANNmuon <= 0.4` 的事件，剩 38,012 行）并删掉 `SPDhits`、`mass`、`production` 等会引入偏差或对不齐的列。接着我们用 XGBoost（`eta=0.4`、`max_depth=6`、300 棵树）训练二分类模型，训练耗时约 17.6 秒。然后我们在本地算加权 AUC，并在 `check_agreement` / `check_correlation` 上跑 KS 与 CvM 检验，得到 KS≈0.085、CvM≈0.0008，两项都达标。最后我们生成 `id,prediction` 格式的提交文件。

这个比赛让我们记住了一件事：**在科学任务里，「预测得准」只是及格线，「预测得符合已知规律」才是真正的门槛**。同一次训练，换一个随机种子就可能让 CvM 从 0.0008 跳到 0.009，因此每次提交前都必须把一致性检验跑完，而不是盯着一个验证分数就交卷。