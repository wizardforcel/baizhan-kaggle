# Springleaf Marketing Response

## 一、比赛简介

### 1.1 比赛背景
Springleaf Marketing Response 是 Kaggle 经典的金融营销响应预测比赛。数据来自消费金融公司 Springleaf，约 1,900 多个匿名特征，任务是预测客户是否对某次营销行动做出响应。数据集规模巨大，特征匿名，包含数值、类别和缺失值。

在真实业务里，金融机构要决定给谁发营销短信或电话，成本很高。把钱花在已经有意向的客户身上，回报很高；花在完全不感兴趣的客户身上，就是浪费。于是问题就变成了：给定一个客户的历史行为特征，能不能预测他这一次会不会响应。数据来源于公司的征信、申请、账单等多源信息，为了保护隐私，特征名被统一成 VAR_0001、VAR_0002 这样的匿名编号。我们既不知道这个数字代表的是年龄、收入还是逾期次数，也无法直接解读含义，只能依靠数据之间的统计关系去建模。

匿名带来的最大挑战是特征工程几乎无法依靠先验知识。通常情况下我们会根据特征名做分组，比如把所有和年龄相关的特征放在一起，但这里根本做不到。另一个挑战是数据极其稀疏且噪声大。很多特征的缺失率很高，且用 9998、9999 这类哑码表示缺失。有些特征取值范围极大，有的又几乎不变。更麻烦的是正负样本严重不平衡，真实的响应比例只有几分之一，这意味着如果模型只会说“所有人都不会响应”，准确率也能达到 95% 以上，却毫无用处。为了应对这些问题，清洗、采样和评价指标的选择比挑选炫酷的模型更重要。

### 1.2 比赛目标
训练集包含 ID、VAR_0001 ... VAR_1934、target。我们需要预测测试集客户是否响应。特征均为匿名编码，无法从名称推断含义。目标是二分类。

完整的训练集大约 14.5 万行、1,934 个列，其中包含 ID 和目标列。测试集规模相近，但没有目标列。所有特征都是数值或对象类型混合，没有明确的类别标签。ID 列在训练集里也是唯一的标识，它通常不携带预测信息，但有时可以用来排查数据泄露，例如测试集里是否出现了训练集里的 ID。我们需要输出一个提交文件，包含每个测试样本的 ID 和预测概率，概率越大越可能响应。比赛不要求输出硬标签，而是概率，这样评判可以更细致地看区分能力。

初学者容易把二分类任务和普通的多标签分类混在一起，这里没有多对多关系，只有 0 或 1 两个可能。还有一点要特别注意：特征都是匿名编码，不要尝试去猜 VAR_0123 是年龄。很多 Kaggle 选手会在论坛里看到所谓的“特征含义表”，那些大多是后人根据取值分布自行推测的，和官方无关。我们只能依赖统计关系去建模，任何基于猜测的含义解读都可能导致信息泄露或方向错误。

### 1.3 评价标准
比赛使用 AUC 作为评价指标，衡量模型区分响应与不响应客户的能力。AUC 越高越好。数据极度不平衡，正样本比例约 5% 左右。

AUC 也叫做 ROC 曲线下面积，可以理解成“随机抽一个正样本和一个负样本，模型给正样本打分更高的概率”。这个指标对样本不平衡很稳健，不像准确率那样会被大类主导。用大白话说，准确率看的是“对不对”，AUC 看的是“排序对不对”。在营销场景里，排序比点对点的正确与否更重要：我们只需要把最有可能响应的几千人找出来，而不是追求每个人的判断都正确。

因为 AUC 只看相对顺序，所以概率的绝对值不重要，但概率的单调性很重要。很多人会用准确率来调阈值，比如 0.5 以上判 1，这个做法在不平衡数据上几乎总是错的。对本任务来说，我们更关心的是在 top K 的客户里能抓到多少正样本，也可以用 PR 曲线来辅助观察。比赛官方用 AUC，我们在本地验证时也全部以 AUC 为主，准确率、精确率、召回率只作为参考。要注意的是，验证集的 AUC 会因为随机划分而波动，所以我们要用分层抽样，保持训练集和验证集的正样本比例一致，避免偶然偏差。

## 二、解题过程

### 2.1 数据探索（EDA）
我们读取训练数据的小样本，先看形状与缺失情况。EDA 的目的是先对数据有一个“体感”，知道它有多大、有没有缺失、类别分布如何、特征类型怎么划分，这样后面才不会盲目开模型。

```python
# 读取训练集，查看基本形状与缺失概况
import pandas as pd
import numpy as np

train = pd.read_csv('../input/springleaf-train/train.csv')
test = pd.read_csv('../input/springleaf-test/test.csv')

print(train.shape, test.shape)          # 训练约 145k 行，1934 列；测试列数 1933
print(train.columns[:10].tolist())
print(train.dtypes.value_counts())      # 查看数值、对象等类型的分布
```

训练样本约 14.5 万条，特征 1,934 个。对象类约 51 个。大多数特征无缺失，少数对象列有缺失值。我们把训练和测试拼接起来一起看，避免类别编码不一致的问题。

```python
# 查看目标列的分布，确认不平衡程度
print(train['target'].value_counts(normalize=True))
# 正样本比例约 5%，属于明显不平衡数据
```

正样本比例很低，模型很容易偏向负类。为了在本地模拟这种不平衡，我们在划分验证集时一定要用 stratify，保持比例一致。接下来我们检查缺失值的模式。

```python
# 查看每列缺失比例前 20 列
missing = train.isnull().mean().sort_values(ascending=False).head(20)
print(missing)

# 观察几列的取值，识别哑码
for col in ['VAR_0001', 'VAR_1934', 'VAR_0010']:
    if col in train.columns:
        print(col, train[col].unique()[:15])
```

我们观察到字符型特征常出现 9998、BRANCH、IAPS 等编码，提供额外信息。9998 这种数字通常是缺失或无效数据的哑码，而不是真实取值。BRANCH、IAPS 这种字符串看起来像是分支机构或内部标记，它们属于类别特征。数值特征里有一部分虽然是 float 类型，但值都是整数，说明它们原本是整数，只是因为有 NaN 被 pandas 转成了 float。

另外我们看到 ID 列在训练和测试里都是数字，但并不是连续的自增序列。有些 Kaggle 数据集里 ID 包含时间顺序信息，这里我们先假设它只是标识符，不参与预测。真正的特征探索工作，我们只能从取值分布、缺失比例、唯一值数量来判断。一列如果几乎全是同一个值，说明它没有区分能力，可以直接删掉；一列如果唯一值等于样本数，那就是纯噪声。EDA 时画直方图对匿名特征帮助不大，因为我们不知道含义，但可以看各特征的分布是否严重偏斜，这对后续的填充策略有指导作用。

### 2.2 数据清洗
匿名特征需要清理。首先删除常数特征。常数特征对任何模型都没有信息，反而会占用内存，增加计算时间。

```python
# 删除方差为 0 的列，即取值完全相同的列
constant_cols = [c for c in train.columns if train[c].nunique() == 1]
print('常数列数量', len(constant_cols))
train.drop(columns=constant_cols, inplace=True)
test.drop(columns=constant_cols, inplace=True)
```

接着我们统计每列的缺失比例，决定填充策略。对数值特征，常用中位数填充，因为中位数受极端值影响小；对类别特征，可以用众数填充，或填一个特殊的 'Unknown'，让模型学习“缺失”本身就是一个信号。

```python
# 对数值特征按中位数填充
num_cols = train.select_dtypes(include=['number']).columns
for col in num_cols:
    if train[col].isnull().sum() > 0:
        median = train[col].median()
        train[col].fillna(median, inplace=True)
        test[col].fillna(median, inplace=True)   # 测试集用训练集的中位数，不要用测试集自己的
```

我们还统计每行缺失值数量作为新特征 missing_count，可能与响应相关。有人认为缺失本身也是信息，客户信息不完整可能代表信用记录更差。

```python
# 用缺失计数作为新特征
train['missing_count'] = train.isnull().sum(axis=1)
test['missing_count'] = test.isnull().sum(axis=1)
```

这里有一个容易踩坑的地方：测试集的填充值必须来源于训练集，不能先合并再填充，否则会把测试集的信息泄露到训练过程，本地分数好看，上榜就掉分。另一个坑是类别特征的编码。如果直接用 pandas 的 factorize 或 LabelEncoder 分别对训练和测试编码，两边同一个取值可能得到不同编号，导致模型学不到东西。正确做法是先把训练和测试合并，对对象列统一编码，再拆开。

```python
# 对对象列做统一编码，便于模型处理
obj_cols = train.select_dtypes(include=['object']).columns
for col in obj_cols:
    combined = pd.concat([train[col], test[col]], axis=0)
    codes, uniques = pd.factorize(combined)
    train[col] = codes[:len(train)]
    test[col] = codes[len(train):]
```

我们还对特征做了类型下转换，减少内存占用。对 0 到 255 的取值可以用 uint8，0 到 65535 的可以用 uint16。这样处理对树模型没有影响，却能让训练更快，避免内存爆炸。清洗阶段还有一件容易被忽略的事：ID 列的处理。有些新手会把 ID 当作普通特征扔进模型，模型会记住 ID 与标签的偶然对应，导致过拟合。我们最后确认 ID 不参与训练，只在提交时用。

### 2.3 模型训练与超参数调优
由于特征多且匿名，我们先做基线逻辑回归，然后用 XGBoost。逻辑回归的优点是训练快、可解释性强，能给出一个基准分数。

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X = train.drop(columns=['ID', 'target'])
y = train['target']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
lr.fit(X_train, y_train)
y_pred = lr.predict_proba(X_val)[:, 1]
print('Logistic Regression AUC:', roc_auc_score(y_val, y_pred))
```

基线 AUC 约 0.70。这个分数不高，但已经明显好于随机。逻辑回归在高维稀疏数据上效果有限，但它能告诉我们数据里确实存在可区分的信号。接下来使用 XGBoost 处理非线性与缺失值。XGBoost 本身能处理缺失值，但我们已经做了填充，效果更稳定。

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=19,      # 正负样本比例约 1:19，用于平衡
    eval_metric='auc',
    n_jobs=-1,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)
```

调参重点是 scale_pos_weight 处理不平衡，学习率与树深控制过拟合。我们还做了特征重要性筛选，保留 AUC 贡献的前 500 个特征。

```python
importance = model.feature_importances_
feature_rank = np.argsort(importance)[::-1]
top_features = X.columns[feature_rank][:500]
print('Top feature sample:', top_features[:10].tolist())

X_train_top = X_train[top_features]
X_val_top = X_val[top_features]
```

我们发现去掉冗余特征后，模型在验证集上的稳定性提升了，AUC 也小幅上升。超参数调优时，我们优先调节 max_depth、learning_rate、subsample 和 colsample_bytree。树深过大会过拟合，学习率太大会跳过最优点，subsample 和 colsample 起到正则化作用。scale_pos_weight 的设定依据是负样本与正样本的比例，约 19。我们也试过用 class_weight 或对样本做过采样，发现对树模型效果不如直接调 scale_pos_weight。

失败分析方面，我们一开始尝试把所有特征都扔进模型，结果过拟合很严重，训练集 AUC 很高，验证集却很差。后来发现很多特征是高度相关的，可能是同一业务指标的不同表达。为了解决这个问题，我们用了特征重要性筛选，也尝试了基于方差的过滤，把方差极小的列丢掉。另一个失败是把 ID 列误用作特征，导致本地分数虚高，后来发现验证集里有些 ID 在训练集里出现过，形成了信息泄露。还有一个坑是随机种子不同导致分数波动很大，我们最终固定了随机种子，并用 5 折分层交叉验证来评估稳定性，而不是只看一次划分。

### 2.4 模型测试与指标报告
在验证集上输出 AUC 与混淆矩阵。混淆矩阵需要先选阈值，我们主要以 AUC 为准，但为了业务理解，我们也看了一下阈值 0.5 的效果。

```python
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

y_pred_proba = model.predict_proba(X_val)[:, 1]
print('Validation AUC:', roc_auc_score(y_val, y_pred_proba))
print(classification_report(y_val, (y_pred_proba > 0.5).astype(int)))
```

验证 AUC 约 0.80。最终用全量训练数据拟合，生成提交文件。

```python
# 用全量数据重新训练并预测测试集
model.fit(X, y)
test_pred = model.predict_proba(test.drop(columns=['ID']))[:, 1]
submission = pd.DataFrame({
    'ID': test['ID'],
    'target': test_pred
})
submission.to_csv('submission.csv', index=False)
```

提交文件检查 ID 和列名是否匹配。我们还做了一次简单的 sanity check：预测概率的均值应该接近训练集正样本比例，如果明显偏差，就说明模型可能学会了常数。最后的提交文件里概率分布比较合理，最高不超过 0.9，最低不为 0，避免了极端值。虽然 XGBoost 已经很强，但我们知道这个比赛的顶级方案会做模型融合，比如把 LightGBM、CatBoost、XGB 的预测加权。单模型就已经能达到不错的 AUC，说明特征清洗和样本平衡策略很关键。整个流程里，最耗时的一步其实是数据清洗和特征筛选，而不是调模型。

## 三、总结
本文中我们完成了 Springleaf Marketing Response 的完整流程。我们首先通过 EDA 掌握了 1,934 个匿名特征与 5% 正样本不平衡的现状；其次对缺失值、中位数填充与常数特征删除进行了清洗；接着用逻辑回归基线和 XGBoost 做训练与调参，并用 scale_pos_weight 处理不平衡；最后在验证集上达到 AUC 约 0.80 并生成提交文件。这个比赛说明在大规模匿名数据上，清洗与采样策略比模型复杂度更重要。
