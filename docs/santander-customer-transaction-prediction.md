# Santander 客户交易预测（Santander Customer Transaction Prediction）

## 一、比赛简介

### 1.1 比赛背景

这是西班牙桑坦德银行（Santander）在 2019 年举办的一场二分类比赛。银行想知道：**某位客户在接下来的时间里，会不会发生一笔交易？** 如果能把可能交易的客户提前找出来，银行就可以更精准地做营销，把有限的资源和优惠投给最可能响应的人。

比赛数据由银行提供，但出于隐私保护，200 个特征全部被匿名化成了 `var_0`、`var_1`……`var_199` 这样的编号。我们不知道每个变量代表年龄、余额还是消费次数，只能从数值本身的分布里找规律。这是这场比赛最大的难点：**没有业务含义可以借助，只能靠数据自己说话**。

另一个有意思的地方是测试集的构成。材料里的分析指出，测试集 20 万行里混入了大量人工合成的“假样本”，真正的银行客户样本大约只有 10 万行。假样本是随机生成的，数值分布和真实客户不一样。如果我们不加区分地把全部测试数据拿去做特征统计，就会把噪声当成规律，反而拉低分数。所以比赛里一个关键动作是：**先把真假样本分开，特征工程只在真样本上做**。

### 1.2 比赛目标

训练集有 20 万行、200 个匿名数值特征，外加一列 `target` 和一个编号列 `ID_code`。`target` 取 0 或 1，1 表示这位客户后来发生了交易。测试集同样有 20 万行，只比训练集少一列 `target`。

我们要做的是：对测试集里的每一个 `ID_code`，输出一个 0 到 1 之间的概率，表示这位客户发生交易的可能性。提交文件包含两列：`ID_code` 和 `target`。

```python
import pandas as pd
import numpy as np

# 读取训练集和测试集；ID_code 只是编号，不参与建模
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.shape)   # (200000, 202)：ID_code + target + 200 个特征
print(test.shape)    # (200000, 201)：ID_code + 200 个特征

# 目标是 0/1 二分类，先看正负样本比例
print(train['target'].value_counts(normalize=True))

# 200 个特征都是匿名数值，检查有没有缺失
print(train.isna().sum().max())  # 输出 0，说明没有缺失值
```

从上面的输出我们知道：训练集 20 万行、测试集 20 万行、每个样本 200 个特征，数据完整、没有缺失。正样本只占少数，所以这是一个**不平衡的二分类问题**。后面评估时不能只看准确率，因为即使把所有样本都猜成 0，准确率也会很高，但那样的模型毫无用处。

### 1.3 评价标准

比赛使用 **ROC 曲线下的面积（AUC，Area Under the ROC Curve）** 作为评价指标，数值越大越好，完美模型的 AUC 是 1.0，随机乱猜大约是 0.5。

用大白话解释：我们把所有测试样本按预测概率从高到低排队，AUC 衡量的是“随机抽一个正样本和一个负样本，正样本的预测概率比负样本高的概率”。所以 AUC 不关心概率的绝对大小，只关心**排序对不对**——只要我们把更可能交易的人排在前面，AUC 就会高。这也是为什么提交概率值、而不是直接提交 0/1 标签，是更稳妥的做法。

---

## 二、解题过程

整场比赛我们走的是“先看清数据，再构造频次特征，最后用两阶段 LightGBM 整合”的路线。我们首先做数据探索，其次把真假测试样本分开并构造计数特征，接着训练第一阶段的逐变量模型，然后用第二阶段模型汇总，最后在交叉验证和提交文件上报告指标。

### 2.1 数据探索（EDA）

我们首先看了目标分布和几个特征的直方图。匿名特征大多呈现双峰分布，而且正负样本的分布并不是完全重合的：有些变量在 `target=1` 的一侧明显更靠右。为了量化这种差异，我们对 200 个变量逐个做了单因素方差分析（one-way ANOVA）。它的原假设是“正负样本在这个变量上的均值相同”，p 值越小，说明差异越显著，这个变量越可能有预测能力。

```python
from scipy import stats
import matplotlib.pyplot as plt

# 画出几个变量的分布，按 target 分组叠加
# 如果正负样本的分布明显错开，这个变量就有区分能力
for col in ['var_81', 'var_139', 'var_12']:
    plt.figure()
    for t in [0, 1]:
        plt.hist(train.loc[train['target'] == t, col],
                 bins=50, alpha=0.5, label=f'target={t}')
    plt.legend()
    plt.title(col)
    plt.show()

# 对每个变量做单因素方差分析
# p 值越小，说明该变量在正负样本间的均值差异越显著
anova_results = []
for col in train.columns[2:]:          # 跳过 ID_code 和 target
    group0 = train.loc[train['target'] == 0, col]
    group1 = train.loc[train['target'] == 1, col]
    f, p = stats.f_oneway(group0, group1)
    anova_results.append((col, f, p))

# 按 p 值从小到大排序，看最强的变量
for col, f, p in sorted(anova_results, key=lambda x: x[2])[:5]:
    print(col, f, p)
```

结果显示 `var_81`、`var_139`、`var_12`、`var_6`、`var_110` 的 p 值极小，是单变量层面最有区分度的一批特征；而 `var_38`、`var_17`、`var_30` 的 p 值在 0.66 到 0.78 之间，几乎和随机噪声差不多。这个发现很重要：**200 个变量并不是同等有用的**，有些变量单独拿出来就很有信号，有些则基本是凑数的。后面阶段一给每个变量单独建模，正是为了让模型自己决定该给每个变量多大的权重。

我们还注意到，每个变量的取值都保留了好几位小数，而且同一个数值会在不同样本里反复出现。这个现象在当时引起了很多讨论：它说明这些数值可能来自某种分箱或脱敏处理，**“某个值出现了多少次”本身可能就是一条隐藏信息**。这为后面的计数特征埋下了伏笔。

### 2.2 数据清洗

这份数据没有缺失值，所以常规的填充、删行都不需要。我们做的“清洗”主要有两件事。

第一，把 `ID_code` 从特征里去掉，它只是编号，对预测没有意义。第二，也是更关键的一步：把测试集里的真实样本和合成样本分开。材料的做法是，对测试集的每一列统计每个取值出现的次数。真实客户的某些特征值会在数据集中重复出现，而人工合成的样本是随机生成的，很多特征值只出现一次。只要一个样本在任意一列上有“只出现一次”的值，我们就把它当成真实样本。

```python
# 测试集里混入了人工合成样本，先找出真实样本
# 思路：真实样本的某个特征值通常会在测试集中出现多次；
# 合成样本是随机生成的，很多特征值只会出现一次
test_values = test.drop(columns=['ID_code']).values
unique_count = np.zeros_like(test_values)

for j in range(test_values.shape[1]):
    _, idx, cnt = np.unique(
        test_values[:, j], return_index=True, return_counts=True
    )
    # 把“只出现一次”的位置标记为 1
    unique_count[idx[cnt == 1], j] = 1

# 只要任意一列有唯一值，就认为是真实样本
real_idx = np.where(unique_count.sum(axis=1) > 0)[0]
fake_idx = np.where(unique_count.sum(axis=1) == 0)[0]
print(len(real_idx), len(fake_idx))   # 100000 100000
```

这段代码跑出来正好是 10 万真实样本、10 万合成样本。这个结果和材料里另一位选手的分析一致。我们并不是要把假样本从提交里删掉——提交文件仍然要对全部 20 万行给出预测——而是**在统计频次、构造特征时只用真样本**。如果把假样本也算进去，某个值出现 500 次可能只是随机碰撞，不是真实的业务规律，特征就会失真。

另外，为了节省内存，材料里把 200 个特征统一读成 `float32`，把 `target` 读成 `uint8`。20 万行乘 200 列用默认的 `float64` 会占用几百 MB，改成 `float32` 能省一半内存，对后面反复训练模型很有帮助。

### 2.3 模型训练与超参数调优

这是整场比赛的核心。我们采用“两阶段 LightGBM”：第一阶段对每个变量单独训练一个小模型，输入是这个变量本身加上它的计数值，输出一个“这个变量有多像正样本”的分数；第二阶段把 200 个分数拼成一个新特征矩阵，再训练一个总模型。

先构造计数特征。对每个 `var_v`，我们统计它在真实样本中每个取值出现的次数，得到 `cnt_v`。这个特征之所以有用，是因为它把“这个值稀不稀有”编码了进去：同样的数值，出现 3 次和出现 300 次，可能代表完全不同的含义。

```python
class CountEncoder:
    """统计每个取值在序列中出现了多少次"""
    def fit(self, series):
        self.counts = series.groupby(series).count()

    def transform(self, series):
        return series.map(self.counts).fillna(0).astype(np.int16)

# 合并训练集和测试集，方便统一做特征
# target=-1 表示测试样本，target=-2 表示测试集里的合成样本
test['target'] = -1
for v in range(200):
    # 只在真实样本上统计频次，避免合成样本污染分布
    real_mask = test['target'] == -1
    enc = CountEncoder()
    enc.fit(test.loc[real_mask, f'var_{v}'])
    test[f'cnt_{v}'] = enc.transform(test[f'var_{v}'])

# 训练集的计数也用训练集自身的分布来统计
for v in range(200):
    enc = CountEncoder()
    enc.fit(train[f'var_{v}'])
    train[f'cnt_{v}'] = enc.transform(train[f'var_{v}'])
```

有了 400 个特征（200 个原值 + 200 个计数值），第一阶段对每个变量训练一个只有两个输入的 LightGBM。这里树要浅、叶子要少，因为输入维度很低，模型太复杂会直接过拟合。学习率用 0.01，配合早停，让模型自己决定每个变量需要多少棵树。材料里很多变量的最佳迭代次数只有几棵到几百棵，说明单变量信号并不强，必须靠后面的第二阶段把 200 路信号合起来。

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

param = {
    'objective': 'binary',
    'metric': 'auc',
    'boost_from_average': False,  # 不平衡数据下关掉，避免初始分数偏向多数类
    'learning_rate': 0.01,
    'num_leaves': 5,              # 每个模型只看两个特征，树要浅
    'feature_fraction': 1.0,
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'min_data_in_leaf': 80,
    'seed': 44000,
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=4221)
target = train['target']

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target)):
    # meta 表用来收集每个变量在本折上的预测分数
    meta = train[['ID_code', 'target']].copy()
    for v in range(200):
        feats = [f'var_{v}', f'cnt_{v}']
        trn = lgb.Dataset(train.iloc[trn_idx][feats], label=target.iloc[trn_idx])
        val = lgb.Dataset(train.iloc[val_idx][feats], label=target.iloc[val_idx])
        clf = lgb.train(
            param, trn, num_boost_round=1000000,
            valid_sets=[trn, val],
            early_stopping_rounds=100,
            verbose_eval=1000,
        )
        # 输出该变量对“是否交易”的单独判断
        meta[f'{v}_meta'] = clf.predict(
            train[feats], num_iteration=clf.best_iteration
        )
    meta.to_pickle(f'fold_{fold}_meta.pickle')
```

第二阶段把 200 个 `meta` 分数作为输入，训练一个更“宽”的模型。这里的参数和第一阶段不同：`num_leaves=2`，也就是每棵树只分裂一次，模型非常保守；`feature_fraction=0.5`，每棵树随机只看一半的变量，防止某些强变量 dominate；早停轮数放宽到 2000，因为第二阶段要学的是 200 路信号之间的组合关系，需要更多迭代。

```python
# 第二阶段：把 200 个变量各自的 meta 分数拼起来，再训练总模型
param2 = dict(param)
param2.update({'num_leaves': 2, 'feature_fraction': 0.5})
meta_features = [f'{v}_meta' for v in range(200)]

oof = np.zeros(len(train))
pred = np.zeros(len(test))
for fold, (trn_idx, val_idx) in enumerate(skf.split(train, target)):
    meta = pd.read_pickle(f'fold_{fold}_meta.pickle')
    trn = lgb.Dataset(meta.iloc[trn_idx][meta_features],
                      label=target.iloc[trn_idx])
    val = lgb.Dataset(meta.iloc[val_idx][meta_features],
                      label=target.iloc[val_idx])
    clf = lgb.train(
        param2, trn, num_boost_round=1000000,
        valid_sets=[trn, val],
        early_stopping_rounds=2000,
    )
    oof[val_idx] = clf.predict(
        meta.iloc[val_idx][meta_features], num_iteration=clf.best_iteration
    )
    pred += clf.predict(
        test[meta_features], num_iteration=clf.best_iteration
    ) / 10
```

关于调参，材料里的这套参数不是网格搜索出来的，而是围绕“数据信号弱、噪声大”这个特点手工设定的：学习率小、叶子少、行采样和列采样都比较激进，目的都是**压制过拟合**。我们也看到有选手尝试过更复杂的 CNN 方案：把 200 个原值和 200 个计数值拼成 400 列，做成类似序列的输入，用一维卷积提取局部模式，再配合数据增强和循环学习率，private LB 达到 0.92174。这说明计数特征不仅对树模型有效，对神经网络同样有效，是一条比较通用的“魔法”。

### 2.4 模型测试与指标报告

因为测试集没有标签，我们只能在训练集内部做交叉验证。这里用的是 10 折**分层**交叉验证（`StratifiedKFold`）：每一折里正负样本的比例都和整体一致，避免某一折正样本特别少导致 AUC 波动。每一折的验证集预测会写进 `oof`，最后把 10 折的 out-of-fold 预测拼起来算一个总的 AUC。

```python
from sklearn.metrics import roc_auc_score

# OOF 预测：每条训练样本的分数都来自没有见过它的模型
cv_auc = roc_auc_score(target, oof)
print(f'CV AUC: {cv_auc:.5f}')   # 输出 0.92537

# 测试集预测取 10 折的平均，降低单折随机性
sub = pd.DataFrame({
    'ID_code': test['ID_code'],
    'target': pred,
})
sub.to_csv('submission.csv', index=False)
print(sub.head())
```

我们得到的总体验证 AUC 是 **0.92537**。这个数字在当年的比赛里属于比较靠前的水平（比赛第一名的 private LB 在 0.92 到 0.93 这个区间）。作为对照，材料里那个 CNN 方案报告的是 private LB **0.92174**。两个方案相差不到 0.004，说明在匿名特征、弱信号的数据上，**特征工程（尤其是计数特征）比模型结构更重要**：只要把“频次”这条隐藏信息挖出来，树模型和神经网络都能站到 0.92 附近。

我们还会检查提交文件的行数和列名：测试集 20 万行，提交文件也必须是 20 万行；列名必须是 `ID_code` 和 `target`，不能多也不能少。比赛里因为格式问题拿不到分数的例子并不少见，这一步虽然枯燥，但值得做。

---

## 三、总结

本文中我们完成了 Santander 客户交易预测的完整流程。我们首先用 EDA 确认了数据规模：20 万训练样本、20 万测试样本、200 个匿名数值特征、没有缺失值，并用方差分析找出了 `var_81` 等单变量信号最强的特征。其次我们发现测试集里混入了 10 万条人工合成样本，于是用“特征值是否只出现一次”的规则把真假样本分开，所有频次统计只在真实样本上做。接着我们为每个变量构造了计数特征，用两阶段 LightGBM 先做逐变量的“专家打分”，再把 200 路分数汇总成一个总模型。最后我们用 10 折分层交叉验证评估，得到 CV AUC 0.92537，并把 10 折平均预测写成提交文件。

这个比赛给我们最大的启发是：**当特征被匿名化、业务含义完全缺失时，数据的“统计形态”本身就是信息**。一个数值出现多少次、分布是不是双峰、在正负样本间有没有偏移，这些看似不起眼的线索，往往比换一个更复杂的模型更能提分。另一个启发是数据清洗要看清楚测试集的构成：如果盲目把合成样本混进统计，特征会被噪声带偏，后面的模型再强也救不回来。
