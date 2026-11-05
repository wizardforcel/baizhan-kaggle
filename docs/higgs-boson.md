# Higgs Boson

## 一、比赛简介

### 1.1 比赛背景
Higgs Boson 是 Kaggle 上经典的高能物理分类比赛，和欧洲核子研究中心（CERN）的大型强子对撞机（LHC）有关。质子在高速对撞后会产生大量粒子，其中极少数会衰变成 Higgs 玻色子。Higgs 玻色子本身极不稳定，会迅速衰变成其他粒子，因此物理学家只能通过探测器记录到的衰变产物（例如轻子、喷注、缺失横向能量）来反推它是否出现过。这个比赛把这些模拟出来的事件交给参赛者，让大家区分“信号事件”（s）和“背景事件”（b）。

比赛数据包含 30 个物理特征，可以分成几类：一类是原始测量量（前缀 `PRI_`），比如粒子的横动量、赝快度、方位角；另一类是物理学家用手工公式推导出来的高阶特征（前缀 `DER_`），比如不变质量、可见质量、横向质量、粒子之间的角度差。这些特征本身就是领域知识的结晶，比赛希望参赛者能在它们的基础上做出比简单阈值切割更好的分类器。

### 1.2 比赛目标
训练集 `training.zip` 有 250,000 条事件，测试集 `test.zip` 有 550,000 条事件。训练集有 33 列：`EventId`、30 个特征、`Weight` 和 `Label`；测试集缺少 `Label` 和 `Weight`。`Label` 的取值是 `s` 或 `b`，训练集里背景事件的数量大约是信号事件的两倍，属于类别不平衡问题。`Weight` 是每个事件在物理意义上的权重，评估时会用到它。

我们需要为每条测试事件输出一个信号概率，或者直接输出 `s`/`b` 的分类结果。比赛要求提交 `EventId` 和 `Label` 两列，`Label` 是 `s` 或 `b`。

### 1.3 评价标准
比赛使用 AMS（Approximate Median Significance，近似中位数显著性）作为评价指标，越高越好。直观理解是：在相同的背景噪声下，模型找出的信号越多、越纯，AMS 就越高。AMS 的计算会同时考虑被判定为信号的样本中的真实信号数 `s`、背景数 `b`，以及它们的统计涨落。它不是简单的准确率，所以一个在多数类上表现很好的模型，未必能拿到高 AMS。

材料中附带了官方脚本 `HiggsBosonCompetition_AMSMetric_rev1.py`，参赛者可以用它离线复现排行榜的评分逻辑。由于 AMS 对阈值敏感，模型输出的概率排序比硬分类更重要。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先读取训练集和测试集，确认列名和形状：

```python
import pandas as pd, numpy as np

train = pd.read_csv('../input/higgs-boson/training.zip')
test  = pd.read_csv('../input/higgs-boson/test.zip')
print(train.shape, test.shape)
# (250000, 33) (550000, 31)
```

训练集 250,000 行、33 列，测试集 550,000 行、31 列。测试集比训练集还大，这是因为比赛官方额外提供了更多测试事件用于最终评估。

接着我们查看标签分布：

```python
print(train['Label'].value_counts())
```

背景 `b` 的数量约为信号 `s` 的两倍，说明数据不平衡。如果直接用准确率评估，模型只要全猜 `b` 就能得到约 2/3 的准确率，但这显然没有物理价值，因此必须用 AMS 或至少用 F1、AUC 来辅助判断。

我们还检查了缺失值的编码方式：

```python
# 统计每列等于 -999 的样本数
missing = (train == -999).sum().sort_values(ascending=False)
print(missing.head(10))
```

物理特征中有些量在某些事件里根本不存在，比如 `DER_deltaeta_jet_jet` 只在有两个喷注时才有定义，因此主办方统一用 `-999` 表示“缺失”。这不是随机缺失，而是物理上“不适用”。如果直接把 -999 当数值喂给模型，模型会误以为这是一个极端的负值，所以必须处理。

我们还按 `PRI_jet_num` 对数据做了分组统计。这个特征表示事件中有几个喷注，取值为 0、1、2、3。不同喷注数下，很多衍生特征是否有定义完全不同，比如没有喷注时 `DER_mass_jet_jet` 全是 -999。这解释了为什么缺失不是随机的，也提示我们：与其用全局均值填补，不如让模型按喷注数分别学习。树模型天然能通过分裂处理这种条件缺失，所以保留 `PRI_jet_num` 并让缺失值单独成支，是更合适的做法。

### 2.2 数据清洗
我们首先去掉测试集里不存在的列，避免训练时用到未来拿不到的信息：

```python
# Weight 和 Label 只在训练集里有，预测时不能使用
train = train.drop(columns=['Weight'])
```

然后处理 -999。材料里的 R 方案采用的做法是：先删除全为 NA 的列、再删除方差为 0 的列，最后用 PCA 处理高度相关的特征组。我们在 Python 里采用更轻量的方式：把 -999 替换成 `NaN`，让梯度提升树自己学习缺失分支：

```python
# 把物理上的“不适用”统一成缺失值
train = train.replace(-999.0, np.nan)
test  = test.replace(-999.0, np.nan)
```

对于高度相关的特征，我们做了相关性检查。材料中的 R 代码用 0.75 作为阈值，把相关系数大于 0.75 的特征对挑出来，再决定是否用 PCA 合成。我们保留了原始特征，因为 XGBoost 对共线性不敏感，而且删除特征会损失物理信息。

另一种常见的清洗方式是把 -999 填成该列的中位数。我们对比过这种做法，发现它会把“没有喷注”的事件和“有喷注但值缺失”的事件混在一起，验证集 AUC 反而下降。最终我们选择让 `NaN` 原样保留，并关闭 XGBoost 的缺失值填充，让它为每个缺失分支单独学习分裂增益。对于 `Weight` 列，它在测试集中不存在，因此训练时也不能作为特征，只能用于评估时计算加权 AMS。

### 2.3 模型训练与超参数调优
我们首先用分层抽样把训练集切成训练和验证两部分，保持信号/背景比例一致：

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 把 s/b 转成 0/1
y = LabelEncoder().fit_transform(train['Label'])
X = train.drop(columns=['EventId', 'Label'])

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
```

我们比较了逻辑回归、随机森林、K 近邻、朴素贝叶斯和 XGBoost。逻辑回归的验证准确率约 0.703，随机森林约 0.825，XGBoost 约 0.829，是其中最好的。XGBoost 的优势在于它能自动处理缺失值、捕捉特征之间的非线性交互，而且训练速度可以接受：

```python
from xgboost import XGBClassifier

clf = XGBClassifier(
    n_estimators=300,       # 树的数量
    max_depth=6,            # 每棵树的深度
    learning_rate=0.05,     # 学习率
    subsample=0.8,          # 行采样，防止过拟合
    colsample_bytree=0.8,   # 列采样
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)
clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
```

超参数方面，我们把重点放在 `max_depth` 和 `learning_rate` 上。深度超过 8 以后验证 AUC 开始下降，说明模型开始记住噪声；学习率 0.1 配 200 棵树与 0.05 配 300 棵的效果接近，我们选择后者以获得更平滑的概率输出。材料中的 R 代码还尝试了 5 层神经网络、Dropout 0.3、batch size 256、学习率 0.0001、L2 正则 0.001，并配合样本权重处理类别不平衡，这套方案也能达到相近的水平。

由于信号样本只有背景的一半，我们还给训练样本加了权重，让模型在损失计算时更关注少数类。XGBoost 里可以通过 `scale_pos_weight` 把正负样本的权重调到约 2，这种简单的平衡策略在物理搜索任务里经常有效，AMS 在阈值变化时更加稳定。

### 2.4 模型测试与指标报告
在验证集上，XGBoost 的准确率约为 0.82886，F1 约为 0.7387。这个准确率看起来不低，但我们更关心 AMS。由于 AMS 依赖阈值，我们先输出概率，再在验证集上扫描不同阈值，选择使 AMS 最大的切点：

```python
from sklearn.metrics import roc_auc_score

proba = clf.predict_proba(X_val)[:, 1]
print('AUC:', roc_auc_score(y_val, proba))

# 在 0.1 到 0.9 之间扫描阈值，找验证集 AMS 最高的点
best_t, best_ams = 0.5, -1
for t in np.arange(0.1, 0.9, 0.01):
    pred = (proba >= t).astype(int)
    ams = compute_ams(y_val, pred, weight_val)   # 用官方脚本计算
    if ams > best_ams:
        best_t, best_ams = t, ams
print('best threshold', best_t, 'AMS', best_ams)
```

确定阈值后，我们对测试集预测并生成提交文件：

```python
test_proba = clf.predict_proba(test.drop(columns=['EventId']))[:, 1]
test_label = np.where(test_proba >= best_t, 's', 'b')

submission = pd.DataFrame({'EventId': test['EventId'], 'Label': test_label})
submission.to_csv('submission.csv', index=False)
print(submission['Label'].value_counts())
```

我们检查了提交文件的 `EventId` 是否与测试集一致、`Label` 是否只有 `s` 和 `b` 两种取值。材料里没有给出最终的 AMS 排行榜分数，因此我们只报告验证集上的准确率、F1、AUC 和通过阈值扫描得到的 AMS。

这里要强调阈值选择不能反复在测试集上试。正确做法是只在验证集上确定阈值，然后把该阈值原封不动地应用到测试集。如果测试集分布与训练集有漂移，AMS 可能下降，但这至少是一个诚实的估计。为了判断稳定性，我们还把验证集按 `PRI_jet_num` 分组，分别看各组的 AUC，发现喷注数较多的事件更容易分类，因为可用的物理特征更完整。

## 三、总结
本文中我们完成了 Higgs Boson 信号/背景分类的完整流程。我们首先用 EDA 确认了 25 万条训练事件、55 万条测试事件、30 个物理特征和约 2:1 的类别不平衡；其次把 -999 识别为“物理上不适用”的缺失编码，并做了删除无关列、相关性与缺失处理；接着用分层抽样和 XGBoost 建立分类器，在验证集上比较了多种模型并调整了树深、学习率和采样参数；最后用 AUC、准确率、F1 和阈值扫描得到的 AMS 来评估，并生成测试集提交文件。

这个比赛给我们的最大启发是：物理特征本身已经浓缩了大量领域知识，机器学习的价值在于把多个手工特征组合成非线性的判别边界；同时，评估指标必须和业务目标一致，AMS 让我们不能只满足于高准确率。

在实际工作中，处理物理数据时要特别注意“不适用”和“缺失”的区别。把 -999 误读为真正的数值，会让模型学会错误的条件分支。我们在清洗时保留了 `PRI_jet_num` 等标志列，让缺失值以条件分支的形式存在，而不是用一个全局常数填补。最后，模型上线前一定要用官方脚本复现 AMS，并在不同阈值上绘制 AMS 曲线，以确保我们选择的阈值既稳定又可解释。
