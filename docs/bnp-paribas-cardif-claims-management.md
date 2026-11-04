# BNP Paribas Cardif 理赔加速预测

## 一、比赛简介

### 1.1 比赛背景

BNP Paribas Cardif 是一家做保险的法国公司。客户出险之后会提交理赔申请，公司要决定：这份理赔是走「加速通道」快速处理，还是走普通流程慢慢审核。走加速通道的客户体验更好、结案更快；但如果判断错了，把不该加速的理赔放过去，可能带来欺诈和财务损失；而把该加速的漏掉，客户等待时间长，对品牌的伤害更大。

这场比赛要做的就是：根据理赔提交时能拿到的信息，预测这份理赔**是否应该被加速处理**。所以标签 `target` 的含义是「这份理赔是否属于加速处理的那一类」，1 表示是，0 表示否。

数据是脱敏过的：除了 `ID` 和 `target`，其余 131 列全部叫 `v1` 到 `v131`，具体业务含义不公开。训练集 **114,321 行 × 133 列**，测试集 **114,393 行 × 132 列**（少一列 `target`）。这些匿名特征有三个鲜明特点：

- **缺失严重**：很多列只有约 64,489 个非空值，也就是说将近一半是空的；
- **数值被缩放并截断在 20**：`describe()` 里大量特征的 `max` 正好是 20.000001，说明原始值经过变换后被封顶；
- **类别列不少**：`v3`、`v22`、`v24`、`v30`、`v31` 等 19 列是文字，其中 `v22` 是两三个字母组成的代码。

另外一个关键事实是**正类占绝大多数**：`target` 的均值是 **0.761199**，也就是约 76% 的理赔属于加速类。这和常见的「少数正类」问题相反，后面的类别权重、评估视角都要跟着调整。

### 1.2 比赛目标

输入是 `train.csv`：`ID`、`target` 和 `v1`～`v131`。输出是每条测试理赔属于加速类的概率，提交文件两列：`ID` 和 `target`。测试集没有标签，我们要填 0～1 的概率。

### 1.3 评价标准

比赛使用 **LogLoss（对数损失）**，材料里的 AutoML 输出直接写着 `The task is binary_classification with evaluation metric logloss`。LogLoss 衡量的是「预测概率和真实标签差多远」：猜得越自信又越错，惩罚越大；**越低越好**。这和 AUC 不同——AUC 只看排序，LogLoss 还看概率是否校准。所以在这个比赛里，把概率「压得太极端」是很危险的，模型必须学会输出接近真实发生率的概率。

## 二、解题过程

我们的主线是：**先通过 EDA 摸清缺失规律和最强特征，再做标签编码与特征筛选，用带类别权重的 LightGBM 做随机搜索调参，最后用 LogLoss 和 Gini 两个视角评估、并用集成模型提交。** 下面按四步走。

### 2.1 数据探索（EDA）

第一步读数据、看整体数值形态：

```python
import pandas as pd
import numpy as np

# 训练集是压缩包里的 CSV，pandas 可以直接读 .zip
train = pd.read_csv('../input/bnp-paribas-cardif-claims-management/train.csv.zip')
test  = pd.read_csv('../input/bnp-paribas-cardif-claims-management/test.csv.zip')

print(train.shape, test.shape)          # (114321, 133) 和 (114393, 132)

# 逐列统计：均值、标准差、最小值、分位数、最大值
desc = train.describe().T
print(desc[['count', 'mean', 'std', 'min', '50%', 'max']].head(20))

# 标签均值就是正类比例：约 0.761
print('target mean =', train['target'].mean())
print(train['target'].value_counts())
```

`describe()` 透露了两个重要信息。一是 `count` 这一列：像 `v1` 只有 64,489 个非空值，而 `v10`、`v12`、`v14` 等有 114,000 多个——**缺失不是随机的**，是成组出现的，可能对应「这类信息只在某些理赔流程里才采集」。二是 `max` 大量停在 20，说明数值已经被截断，极端值不需要额外处理（对树模型尤其如此）。

接着专门看缺失和标签的关系，这是本场 EDA 最有价值的发现：

```python
# 每个字段缺失多少行；同时按标签分组看缺失率的差异
missing_all = train.isnull().mean()
missing_by_target = train.groupby('target').apply(lambda g: g.isnull().mean())

# 找出「正类缺失率 - 负类缺失率」最大的字段
diff = (missing_by_target.loc[1] - missing_by_target.loc[0]).sort_values(ascending=False)
print(diff.head(10))
print(diff.tail(10))

# v50 是公认最强的单特征，单独看它的分布
import matplotlib.pyplot as plt
train[train['target'] == 0]['v50'].hist(bins=50, alpha=0.6, label='target=0')
train[train['target'] == 1]['v50'].hist(bins=50, alpha=0.6, label='target=1')
plt.legend(); plt.title('v50 按标签的分布'); plt.show()
```

结论很有意思：**大部分特征在「加速类」样本里缺失得更多**。一份早期的 EDA 材料对此的解释是：也许不是「缺信息导致快速处理」，而是「信息齐全的理赔反而要走慢流程」。这个反直觉的现象提醒我们，缺失本身就是一个强特征，不能随便用均值填掉。另外 `v50` 的分布能明显区分两类标签，是全场最强的单变量。

`v22` 是字母代码。材料里有人把它按「二十六进制」转成整数（`A=1, B=2, ..., Z=26`，多位字母按位权展开），转完之后直方图上出现了清晰的结构。这说明它不是一个随便的类别编号，而是有顺序含义的代码，值得特殊处理。

最后我们做了一次**对抗验证**，检查训练集和测试集是否同分布：

```python
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 把训练集标 1、测试集标 0，然后让模型去区分「这行来自哪个集合」
xdat = pd.concat([train.drop(columns=['ID', 'target']),
                  test.drop(columns=['ID'])], axis=0)
y_set = np.r_[np.ones(len(train)), np.zeros(len(test))]
for c in xdat.select_dtypes(include='object').columns:
    xdat[c] = pd.factorize(xdat[c])[0]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
clf = xgb.XGBClassifier(learning_rate=0.05, max_depth=4, n_estimators=100,
                        subsample=0.9, colsample_bytree=0.9, n_jobs=-1)
for tr, va in skf.split(xdat, y_set):
    clf.fit(xdat.iloc[tr], y_set[tr], eval_metric='logloss')
    print('对抗验证 AUC:', roc_auc_score(y_set[va], clf.predict_proba(xdat.iloc[va])[:, 1]))
```

如果这个 AUC 接近 0.5，说明两个集合混在一起分不出来；如果接近 1，说明分布差异巨大、本地验证不可信。材料跑出来每折在 **0.545～0.552** 之间，整体约 **0.572**，属于「略有差异但不算严重」，可以放心用随机切分做验证。

### 2.2 数据清洗

清洗的第一件事是编码 19 个类别列。这里有个容易忽略的坑：**测试集里可能出现训练集没有的类别值**。如果直接用 `LabelEncoder` 分别 fit 训练和测试，两边编码会错位；正确做法是只在训练集上 fit，遇到没见过的值统一映射到 `unknown`：

```python
from sklearn import preprocessing

cols_cat = ['v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66',
            'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112',
            'v113', 'v125']

encoders = {}
for cat in cols_cat:
    le = preprocessing.LabelEncoder()
    # 只在训练集上拟合，记住「哪些类别是训练时见过的」
    le.fit(train[cat].astype(str))
    encoders[cat] = le

for cat in cols_cat:
    le = encoders[cat]
    # 按「编码表 → 序号」造一个字典，测试集没见过的一律归为 'unknown'
    mapping = {v: i for i, v in enumerate(le.classes_)}
    mapping['unknown'] = len(mapping)                       # 排在最后面的新类别
    train[cat] = train[cat].astype(str).map(mapping)
    test[cat]  = test[cat].astype(str).map(lambda s: mapping.get(s, mapping['unknown']))
```

为什么不直接把缺失补成均值或众数？因为 LightGBM 原生支持缺失值，它会为缺失单独学一个分支；而「缺失模式」本身（比如一批字段同时为空）正好像上面 EDA 发现的那样有预测力。用均值填，等于把这个信号擦掉。材料里也有别的做法（用众数/均值填补），效果不一定差，但会丢掉这层信息。

第二件事是划出训练、验证、测试三块。我们保留一份完全独立的验证集来盯 LogLoss，避免调参时把验证集用旧：

```python
# 70% 训练 / 15% 验证 / 15% 留出测试，分层保证正类比例一致
from sklearn.model_selection import train_test_split
train_part, rest = train_test_split(
    train, test_size=0.3, random_state=42, stratify=train['target'])
valid_part, test_part = train_test_split(
    rest, test_size=0.5, random_state=42, stratify=rest['target'])

X_cols = [c for c in train.columns if c not in ('ID', 'target')]
y_tr = train_part['target'].values
y_va = valid_part['target'].values
```

### 2.3 模型训练与超参数调优

131 个特征里有不少是噪音。我们先做一次**递归特征消除（RFE）**：让 LightGBM 先训练一遍，按重要性把特征排个序，只保留排名靠前的一批。这样既能降维，也能减少过拟合：

```python
import lightgbm as lgbm
from sklearn.feature_selection import RFE

# 用 LightGBM 当「评委」，递归地删掉最不重要的特征
rfe = RFE(lgbm.LGBMClassifier(verbosity=-1, n_jobs=-1))
rfe.fit(train_part[X_cols], y_tr)

# support_ 是布尔数组：True 表示这个特征被保留
print('保留特征数:', rfe.support_.sum(), '/', len(X_cols))
X_cols = train_part[X_cols].columns[rfe.get_support(1)]
```

接着处理类别不平衡。这里正类占 76%，所以加权方向要反过来——**给负类（0）相对更高的权重**。材料里的写法非常直观：`{0: 1, 1: 0.3137}`，也就是正类的权重是负类的 0.3137 倍，正好是 `负类比例 / 正类比例`：

```python
counts = train_part['target'].value_counts(normalize=True)
# 权重与类别频率成反比：少见的那一类（这里是 0）权重更高
class_weight = {0: 1, 1: counts[0] / counts[1]}
print(class_weight)     # {0: 1, 1: 0.3137...}

model = lgbm.LGBMClassifier(class_weight=class_weight,
                            verbosity=-1, n_jobs=-1)
```

然后做随机搜索。材料里没有用 `GridSearchCV`，而是自己写了一个「暴力随机搜索」：把各参数的候选值做笛卡尔积，随机抽 20 组来跑，同时记录训练集和验证集的 Gini。Gini 就是 `2 × AUC - 1`，是风控领域常用的写法，和 AUC 单调等价，但数值上更「展开」：

```python
from sklearn.metrics import roc_auc_score
import numpy as np

def gini(y_true, prob):
    # Gini = 2*AUC - 1：AUC 0.5 对应 Gini 0，AUC 1 对应 Gini 1
    return 2 * roc_auc_score(y_true, prob) - 1

params_grid = {
    'max_depth': [2, 4, 6],
    'n_estimators': [50, 100],
    'subsample': [0.25, 0.5, 0.75],
    'min_gain_to_split': [60, 80, 100],
    'reg_alpha': [0.5, 0.7, 1.0],
    'learning_rate': [0.1, 0.2, 0.3],
    'colsample_bytree': [0.2, 0.4, 0.6],
}

best = None
rng = np.random.default_rng(0)
for _ in range(20):     # 只抽 20 组：单棵树训练很快，但没必要穷举
    cfg = {k: v[rng.integers(len(v))] for k, v in params_grid.items()}
    model.set_params(**cfg)
    model.fit(train_part[X_cols], y_tr)
    g_va = gini(y_va, model.predict_proba(valid_part[X_cols])[:, 1])
    if best is None or g_va > best[0]:
        best = (g_va, cfg)
    print(round(g_va, 4), cfg)

print('最好的验证 Gini:', best)

# 用最好的一组参数重新拟合，供后面报告使用
model.set_params(**best[1])
model.fit(train_part[X_cols], y_tr)
```

搜索出来的最优组合是 `max_depth=6, n_estimators=100, subsample=0.5, min_gain_to_split=80, reg_alpha=0.5, learning_rate=0.1, colsample_bytree=0.6`，对应的验证 Gini 在 **0.465 左右**。换成 AUC 大约是 **0.732**。这个模型的树不深、轮数不多、采样比例偏低，整体是「保守型」配置——在缺失严重、特征含义未知的数据上，这是好事。

### 2.4 模型测试与指标报告

我们先在训练集、验证集、测试集三块上同时报告 Gini，检查有没有过拟合：

```python
def report_gini(model, parts):
    """同一套模型，在三块数据上各算一次 Gini，差距过大说明过拟合"""
    for name, part in parts.items():
        prob = model.predict_proba(part[X_cols])[:, 1]
        print(name, 'Gini =', round(gini(part['target'], prob), 6))

report_gini(model, {'train': train_part, 'valid': valid_part, 'test': test_part})
```

材料里的结果是：训练集 **0.464656**、验证集 **0.463168**、测试（留出）集 **0.453845**。三个数字非常接近，说明模型没有明显过拟合；测试集略低一点是正常现象，因为那一块样本更少。

接着按 0.5 阈值把概率转成标签，看分类报告：

```python
from sklearn.metrics import classification_report, roc_curve

prob_va = model.predict_proba(valid_part[X_cols])[:, 1]
pred_va = (prob_va >= 0.5).astype(int)
print(classification_report(valid_part['target'], pred_va, digits=2))

# ROC 曲线：把排序能力和阈值选择分开看
fpr, tpr, _ = roc_curve(valid_part['target'], prob_va)
plt.plot(fpr, tpr, label='LGBM (AUC=%.3f)' % roc_auc_score(valid_part['target'], prob_va))
plt.plot([0, 1], [0, 1], 'r--'); plt.legend(); plt.show()
```

材料里 0.5 阈值下的准确率是 **0.65**；类别 0 的精确率/召回率是 0.37/0.68，类别 1 是 0.86/0.64。读法是这样的：模型对「非加速」这一类（少数类）精确率不高——一旦它说「这个不该加速」，只有 37% 是对的；但对多数类的把握很足。在 LogLoss 的评价下，我们更关心概率本身，而不是这个 0.5 阈值下的硬分类，所以这个报告主要用于理解模型行为，而不是调阈值。

为了进一步压 LogLoss，我们还试了 AutoML 自动调参与模型集成。材料里的 Optuna 自动搜索给出的验证 LogLoss 是：

| 模型 | 验证 LogLoss |
| --- | --- |
| LightGBM | 0.458348 |
| XGBoost | 0.462022 |
| RandomForest | 0.470014 |
| CatBoost | 0.446348 |
| CatBoost + GoldenFeatures | 0.446006 |
| 集成（ensemble） | 0.44422 |
| 堆叠集成（stacked ensemble） | **0.442319** |

可以看到，单模型里 CatBoost 最好（0.446348），而把它和 LightGBM、XGBoost 堆叠之后能到 **0.442319**，是所有方案里最低的。这符合 LogLoss 比赛的一般规律：**单个模型再强，也不如几个差异化的模型取长补短**。

最后生成提交文件：

```python
# 用表现最好的概率输出（单模型或集成），保持 sample_submission 的列名
sub = pd.read_csv('../input/bnp-paribas-cardif-claims-management/sample_submission.csv.zip')
prob_test = model.predict_proba(test[X_cols])[:, 1]

sub['target'] = prob_test
# 提交前检查：行数一致、没有空值、概率都在 0~1 之间
assert len(sub) == len(test)
assert sub['target'].notna().all()
assert sub['target'].between(0, 1).all()
sub.to_csv('submission.csv', index=False)
```

## 三、总结

本文中我们完成了 BNP Paribas Cardif 理赔加速预测的完整流程。我们首先用 EDA 看清了数据：114,321 行训练样本、131 个匿名特征、约 76% 的正类，数值被截断在 20，大量字段有近半缺失，而且「加速类」样本反而缺失得更多；其次用只在训练集上拟合的 LabelEncoder 处理 19 个类别列（没见过的值统一映射为 `unknown`），保留缺失让 LightGBM 自己学，并用对抗验证确认训练/测试分布差异不大（AUC 约 0.55）；接着用 RFE 筛特征、按类别频率的倒数设置权重（正类权重约 0.3137），再做 20 组随机参数搜索，得到 Gini 约 0.465（AUC 约 0.732）的 LightGBM；最后在训练/验证/测试三块上报告 Gini（0.464656 / 0.463168 / 0.453845）、查看分类报告，并对比了 AutoML 的单模型与集成结果，其中 CatBoost 单模型 LogLoss 0.446348，堆叠集成达到最低的 0.442319。

这个比赛给我们最大的启发是：**评价指标决定优化方向**。如果只看 AUC，我们会拼命把排序做对；但这里比的是 LogLoss，概率校准同样重要，所以集成、保守的参数、对缺失的处理方式都要围绕「输出可信的概率」来设计。另外一点是：**缺失值不是垃圾，是信息**——在这个数据集里，「哪些字段为空」本身就和理赔是否加速强烈相关，随手用均值填掉，等于把最有价值的线索扔了。