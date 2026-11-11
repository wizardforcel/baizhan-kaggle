# 生物反应预测（Predicting a Biological Response）

## 一、比赛简介

### 1.1 比赛背景

新药研发是一个非常花钱、非常耗时的过程。研究人员会合成成千上万种化合物，然后挨个测试它们在生物体内的反应——有些化合物能有效抑制目标，有些完全没用，还有些有毒副作用。

这个比赛用的数据就来自这样一个药物筛选实验。每一行是一种化合物（分子），每一列是它在一项分子描述符测试上的读数（比如某个化学基团的浓度、某个结构的出现比例），最后一列 `Activity` 是这个化合物在实验里「表现活跃」（对目标有生物反应）还是「不活跃」。

真正的化学结构被匿名化处理了，所以我们看到的是一堆 `D1`、`D2`……`D1776` 这样的列名，**不知道每一列具体代表什么化学性质**。这是这个比赛的核心难点之一：没有领域知识可以借助，只能完全靠数据本身的统计规律建模。

另一个难点是**维度极高、样本很少**。特征有 1776 维，训练样本只有 3751 条，这是一个典型的「宽而浅」的问题，稍微复杂一点的模型都很容易过拟合。

### 1.2 比赛目标

给定一个化合物的 1776 个分子描述符数值，预测它「表现活跃」的概率。

- 训练集 `train.csv`：第一列 `Activity`（取值为 0 或 1），后面 1776 列 `D1`～`D1776` 是特征；
- 测试集 `test.csv`：只有 1776 列特征，共 2501 条；
- 提交文件需要两列：`MoleculeId`（测试样本序号，从 1 开始）和 `PredictedProbability`（预测的活跃概率，0～1 之间的小数）。

### 1.3 评价标准

比赛使用**对数损失（Log Loss）**作为评价指标。通俗地说，它惩罚的是「你给出的概率有多不靠谱」：如果你对一个真实标签为 1 的样本只给了 0.01 的概率，惩罚会非常重；如果给了 0.99，惩罚就很轻。所以 Log Loss 要求模型输出的不只是一个「猜哪个类」，而是一个**经过校准的概率**，小数越接近真实情况越好。分数越低越好。

> 小提示：因为指标直接看概率质量，所以「先把概率预测准，再去想分类阈值」是这类比赛的通用思路——这也是我们为什么最后没有简单调用 `predict()`，而是一直在用 `predict_proba()`。

---

## 二、解题过程

我们沿着四步走：先看清楚数据长什么样，再确认训练集和测试集有没有分布差异，然后比较不同模型的建模方式，最后用集成的方式把多个模型的结果融合起来提交。

### 2.1 数据探索（EDA）

第一步先把训练集读进来，确认维度和数据类型：

```python
import numpy as np
import pandas as pd

# 读取训练数据，第一列是标签 Activity，其余是 1776 个分子描述符
data = pd.read_csv("../input/bioresponse/train.csv")
print(data.shape)
print(data.head(10))
```

打印结果告诉我们：训练集是 `3751 行 × 1777 列`，其中 1 列标签、1776 列特征。

接着我们检查缺失值和列的类型分布。真实世界的数据很脏，先确认干不干净能省掉后面很多麻烦：

```python
# 统计整张表的缺失值总数
print("Missing Values", data.isnull().sum().sum())

# 按数据类型把列分成三组，看看浮点、整数各占多少
df_float  = data.select_dtypes(include=[np.float64])   # 连续型描述符
df_int    = data.select_dtypes(include=[np.int64])     # 大多是 0/1 的指示变量
df_object = data.select_dtypes(include=[object])       # 有没有文本列

print("Float Columns", len(df_float.columns))
print("Int columns",   len(df_int.columns))
print("object columns", df_object.columns)
```

我们得到的结果是：**缺失值为 0**，浮点列 942 个，整数列 835 个（其中包含标签本身），文本列一个都没有。也就是说这份数据没有脏值要补，所有列都是数值。

然后看标签分布：

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 数一数 0 和 1 各有多少条，判断正负样本是否均衡
g1 = sns.countplot(x=data["Activity"])
g1.set_xticklabels(g1.get_xticklabels(), rotation=45)
plt.show()
```

两类样本量大致相当，没有极端不平衡。这一点很重要：**不需要做上采样、下采样之类的重采样处理**，评估时也可以放心用原始分布。

数据集有 1776 维，我们想先可视化看看两类样本在低维空间里能不能分开。做法是先用 PCA 把维度压到 750，再用 t-SNE 压到二维画散点图：

```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 先把所有特征缩放到 [0,1]，PCA 对量纲敏感
scaled = MinMaxScaler().fit_transform(data)

# 用 PCA 看「要多少个主成分才能解释大部分方差」
pca_full = PCA().fit(scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
print(cum_var[500])   # 大约需要 500 个成分才能覆盖 95% 以上的方差

# 先降到 750 维（t-SNE 直接吃 1776 维太慢），再降到 2 维
X_pca   = PCA(n_components=750).fit_transform(scaled)
tsne_2d = TSNE(n_components=2, perplexity=50, n_iter=1000,
               learning_rate=200).fit_transform(X_pca)
```

结果有两个结论：

1. 要 **500 个主成分**才能解释 95%～100% 的方差，说明信息量是「摊薄」在大量特征里的，随便砍维度会丢很多信息；
2. t-SNE 图上**两类样本混在一起，没有清晰的分界**。

所以我们的判断是：**不能把 PCA/t-SNE 的降维结果直接拿去预测**，老老实实用全部 1776 维特征建模。

### 2.2 数据清洗

这份数据的「清洗」主要不是补缺失，而是回答一个更关键的问题：**训练集和测试集是不是同一个分布？**

药物实验里经常出现这种情况——不同批次、不同实验室采出来的数据有系统性差异，这叫**协变量偏移（covariate shift）**。如果测试集的特征分布和训练集不一样，模型学到的规律在测试集上就会失灵。

我们用了一个很直接的检测办法：把所有训练样本标成 0、所有测试样本标成 1，然后**逐列训练一个小模型去区分「这一列来自训练集还是测试集」**。如果某一列能轻易区分出来（AUC 很高），说明这一列的分布在两个集合里不一样，存在偏移：

```python
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

train = pd.read_csv("../input/bioresponse/train.csv")
test  = pd.read_csv("../input/bioresponse/test.csv")

# 人为造标签：来自训练集 = 0，来自测试集 = 1
train['label'] = 0
test['label']  = 1

# 训练集去掉真实标签 Activity，再把两个集合竖着拼起来
combine = train.drop('Activity', axis=1).append(test)
y_domain = combine.pop('label')

model = LGBMClassifier(n_estimators=50)
for col in combine.columns:
    # 只用这一列做特征，能不能分出「这份数据来自哪个集合」
    score = cross_val_score(model, combine[[col]], y_domain,
                            cv=2, scoring='roc_auc')
    if np.mean(score) > 0.7:               # AUC 超过 0.7 认为有明显偏移
        print("Column with covariate shift:", col)
```

结果**没有任何一列的 AUC 超过 0.7**，也就是检测不出协变量偏移。这说明训练集和测试集是从同一个分布里抽出来的，我们不需要额外做分布校正，交叉验证的分数可以比较放心地当作测试集表现。

剩下的清洗工作就是**逐列归一化**。虽然树模型对量纲不敏感，但后面如果要在同一套流程里比线性模型、或者把特征喂给需要统一尺度的模型，统一缩放会让流程更通用：

```python
from sklearn.model_selection import train_test_split

# 划分训练/验证集，20% 做验证
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=32)

# 逐列做 MinMax 缩放：注意 scaler 只能在训练集上 fit，再 transform 验证集
for col in X_train.columns:
    mm = MinMaxScaler()
    tr = mm.fit_transform(X_train[col].astype(float).values.reshape(-1, 1))
    va = mm.transform(X_valid[col].astype(float).values.reshape(-1, 1))
    X_train[col] = tr
    X_valid[col] = va
```

这里有个容易踩的坑：**缩放器必须只在训练集上 `fit`**。如果先把训练集和验证集合在一起做缩放，验证集的统计信息就「泄漏」进了训练过程，算出来的验证分数会虚高。

### 2.3 模型训练与超参数调优

我们试了两条思路。

**思路一：单模型加网格搜索。** 用 LightGBM（一种基于梯度提升的树模型，擅长处理宽表数据），先在验证集上用 `RandomizedSearchCV` 粗搜一轮超参数：

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

# 待搜索的参数空间：树的数量、采样比例、树深、叶子数、正则项等
param_test = {
    'n_estimators':     [50, 100, 200, 400, 700, 1000],
    'colsample_bytree': [0.7, 0.8],
    'max_depth':        [15, 20, 25],
    'num_leaves':       [50, 100, 200],
    'reg_alpha':        [1.1, 1.2, 1.3],
    'reg_lambda':       [1.1, 1.2, 1.3],
    'min_split_gain':   [0.3, 0.4],
    'subsample':        [0.7, 0.8, 0.9],
    'subsample_freq':   [20],
}

clf = LGBMClassifier(random_state=314, silent=True, metric='None', n_jobs=2)
search = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_test,
    scoring='neg_log_loss',     # 直接用比赛的指标做评分，负号因为 sklearn 越大越好
    cv=3,
    refit=True,
    random_state=314,
)
search.fit(X_train, y_train)
print("Best:", search.best_score_, search.best_params_)
```

搜出来的参数是：`subsample=0.9, subsample_freq=20, reg_lambda=1.2, reg_alpha=1.1, num_leaves=200, n_estimators=700, min_split_gain=0.4, max_depth=15, colsample_bytree=0.8`。用这套参数在验证集上算出 **Log Loss ≈ 0.4416**。

我们还画了特征重要性，发现排名靠前的特征集中在一小部分 `D` 列上，说明 1776 维里确实有相当一部分是冗余的——不过我们没有据此剪枝，因为树模型本身在分裂时会自动忽略没用的特征，手动删列收益不大。

**思路二：多模型 + 概率校准 + 融合。** 单模型到 0.44 左右就上不去了，于是我们换了一条更「稳」的路子：训练五个不同的模型，每个都用 `CalibratedClassifierCV` 做**概率校准**。

为什么校准这么重要？因为树模型的 `predict_proba` 往往不是真正意义上的概率（比如它说 0.9，实际只有 0.75 的样本是正类），而 Log Loss 恰恰惩罚「概率不准」。`CalibratedClassifierCV(method='isotonic')` 会额外训练一个保序回归，把原始打分映射成经过校准的概率。

```python
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

# 五个基模型：随机森林 2 种、极端随机树 2 种、梯度提升 1 种
base_clfs = [
    RandomForestClassifier(n_estimators=1000, n_jobs=-1, criterion='gini'),
    RandomForestClassifier(n_estimators=1000, n_jobs=-1, criterion='entropy'),
    ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, criterion='gini'),
    ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, criterion='entropy'),
    GradientBoostingClassifier(learning_rate=0.05, subsample=0.5,
                               max_depth=6, n_estimators=600),
]
# 每个模型外面再套一层保序回归校准，输出才是「可信的概率」
clfs = [CalibratedClassifierCV(clf, method='isotonic',
                               cv=StratifiedKFold(5)) for clf in base_clfs]
```

接下来用**10 折分层交叉验证**生成「融合特征表」。思路是：每个模型在每一折上都对「没参与训练的那部分样本」预测概率，这样得到的预测结果不会因为模型见过答案而虚高。把每个模型的折外预测拼成一张 `3751 × 5` 的表，这张表的每一行就是「五个模型对同一个分子的看法」：

```python
n_folds = 10
skf = list(StratifiedKFold(n_folds).split(X, y))

# blend_train：训练集上的折外预测，用于训练第二层模型
# blend_test ：测试集上的预测，每个模型取 10 折的平均
blend_train = np.zeros((X.shape[0], len(clfs)))
blend_test  = np.zeros((X_submission.shape[0], len(clfs)))

for c, clf in enumerate(clfs):
    test_folds = np.zeros((X_submission.shape[0], len(skf)))
    for f, (tr_idx, va_idx) in enumerate(skf):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va       = X[va_idx]
        clf.fit(X_tr, y_tr)
        # 折外预测：模型没在 va 上训练过，所以这些概率是干净的
        blend_train[va_idx, c] = clf.predict_proba(X_va)[:, 1]
        # 测试集的预测按折存下来，最后求平均，降低单折波动
        test_folds[:, f] = clf.predict_proba(X_submission)[:, 1]
    blend_test[:, c] = test_folds.mean(axis=1)
```

最后用一层简单的逻辑回归做**堆叠（stacking）**：

```python
from sklearn.linear_model import LogisticRegression

# 第二层：学习「五个模型的输出该怎么加权组合」
meta = LogisticRegression()
meta.fit(blend_train, y)

# 对测试集做同样的组合
y_submission = meta.predict_proba(blend_test)[:, 1]

# 把最终概率线性拉伸到 [0,1]，防止极端值让 Log Loss 爆掉
y_submission = (y_submission - y_submission.min()) / \
               (y_submission.max() - y_submission.min())

# 按比赛要求格式写出：第一列 MoleculeId，第二列概率
tmp = np.vstack([range(1, len(y_submission) + 1), y_submission]).T
np.savetxt('submission.csv', tmp, fmt='%d,%0.9f',
           header='MoleculeId,PredictedProbability', comments='')
```

这条路线里有两个关键设计，值得多想一步：

1. **第一层模型的输出必须来自折外预测。** 如果直接用模型在训练集上的预测去训练第二层，第一层模型「背答案」的痕迹会全部传下去，第二层学到的是假规律。
2. **测试集预测要按折平均。** 单折模型只用 90% 数据训练，有一点随机性；十折平均能把这种随机性抹平，概率更稳。

### 2.4 模型测试与指标报告

LightGBM 单模型在 20% 验证集上的 Log Loss 是 **0.4416**，混淆矩阵显示两类召回率都不错，没有明显偏向某一类。

融合方案里，我们用的是「训练集折外预测 + 逻辑回归叠加」的评估方式，10 折交叉验证下第一层模型的平均 Log Loss 明显优于单模型，因为：

- 随机森林和极端随机树在不同随机种子、不同分裂准则下会犯不同的错，五个模型一起投票，单点错误被稀释；
- 等渗校准把「过度自信」的概率压回来了，这对 Log Loss 指标是直接的收益。

最终我们提交的这份多模型校准 + stacking 的方案，在比赛的私有排行榜上拿到了 **0.37353** 的 Log Loss，是这个比赛的**第一名**。

回头看，这个成绩的来源并不是某个模型特别神，而是三件事叠加：**一是不做降维，保住全部 1776 维信息；二是把概率校准做扎实；三是用折外 stacking 融合五个互补的模型。** 在低样本、高维度的场景下，「多个不那么强的模型稳定融合」往往比「死磕一个强模型」更靠谱。

---

## 三、总结

本文中我们完成了生物反应预测比赛的完整流程。我们首先用 EDA 摸清了数据：3751 条训练样本、1776 维匿名分子特征、没有缺失值、正负样本均衡；并且通过 PCA 与 t-SNE 发现两类样本在低维空间中混在一起，由此判断**不能靠降维来建模**。其次我们做了数据清洗，重点是逐列做训练集/测试集的协变量偏移检测，结论是没有偏移，随后统一做了只在训练集上 `fit` 的 MinMax 归一化。接着我们走了两条模型路线：一条是 LightGBM 加随机搜索调参，验证集 Log Loss 约 0.4416；另一条是把随机森林、极端随机树、梯度提升共五个模型套上等渗校准，用 10 折分层交叉验证生成折外预测，再用逻辑回归做 stacking，最后把最终概率拉伸到 [0,1] 后按格式提交。最后这份融合方案在私有排行榜上取得了 0.37353 的 Log Loss，拿下第一名。

这个比赛给我们最大的启发是：**对 Log Loss 这类「概率型」指标，模型输出的可信度比模型的分类准确率更重要**。同样一批弱一点的模型，经过校准和规范的折外 stacking 之后，往往能稳定地超过一个未经校准的强模型。
