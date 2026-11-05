# 哥伦比亚企业破产预测（Colombia Bankrupt）

## 一、比赛简介

### 1.1 比赛背景

企业破产预测是信用风险领域最经典的问题之一。银行、供应商、投资人在把钱借出去或把货发出去之前，都想知道对方**未来一段时间内会不会倒下**。如果能提前识别高风险企业，就可以减少坏账、调整授信额度，或者要求更多担保。

这个比赛给出一批哥伦比亚企业的经营与财务记录，让我们根据这些指标判断企业是否会触发破产事件。它的现实意义很直接：破产不是一瞬间发生的，在正式倒闭之前，企业的财务比率往往会先恶化——负债率上升、现金流变紧、盈利能力下滑。模型要做的，就是从这些数字里提前捕捉信号。

这类任务的难点主要有三个：

1. **样本不均衡**：真正破产的企业是少数，绝大多数企业正常经营。如果模型只会预测「不会破产」，准确率会很高，但完全没有用。
2. **缺失值多**：财务数据常常报不全，某些指标对小企业根本不适用，缺失本身也可能携带信息。
3. **类别型字段**：行业、地区、企业类型这类文字字段需要先编码，才能进模型。

### 1.2 比赛目标

训练集是一张表，每一行是一家企业（或一个企业-年份观测），其中 `event` 是标签：发生破产事件为 1，否则为 0。其余列是特征，既有数值型（各种财务比率、规模指标），也有类别型（代码里通过 `select_dtypes(include=["object"])` 识别出来的文字列）。

测试集结构相同但没有 `event`，我们要给每一行输出一个 **0 到 1 之间的概率**，表示这家企业破产的可能性。提交文件是在样例文件的基础上加一列 `prediction`。

### 1.3 评价标准

代码里交叉验证用的是 `scoring="roc_auc"`，也就是说评价指标是 **ROC AUC**。

用大白话说：把所有企业按模型给出的破产概率从高到低排队，随机取一个真破产的和一个没破产的，AUC 就是「真破产的排在前面」的概率。

- AUC = 0.5：模型没有区分能力，和抛硬币一样；
- AUC = 1.0：完美排序，所有破产企业都排在正常企业前面；
- 实际比赛中 0.7～0.9 都是常见的范围。

AUC 只关心**排序**，不关心概率的绝对大小。这对风险场景很合适：业务上往往只需要「把最危险的 100 家企业挑出来重点审查」，而不需要精确知道某家企业的破产概率到底是 3% 还是 5%。但也正因为如此，AUC 高不代表概率校准好——这一点我们在最后会再提。

---

## 二、解题过程

这份材料给的是一个**干净利落的基线**：读取数据 → 数值列中位数填补、类别列众数填补并独热编码 → 直方图梯度提升树 → 5 折分层交叉验证看 AUC → 全量重训 → 输出概率。我们按 EDA、清洗、训练、测试四步把它讲透。

### 2.1 数据探索（EDA）

第一步是把三个文件读进来，确认行列数、标签分布、缺失情况和字段类型：

```python
import pandas as pd
import numpy as np

TRAIN_PATH = '/kaggle/input/competitions/colombia-bankrupt/train.csv'
TEST_PATH  = '/kaggle/input/competitions/colombia-bankrupt/test.csv'
SUB_PATH   = '/kaggle/input/competitions/colombia-bankrupt/example_submission.csv'

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SUB_PATH)

print('训练集:', train.shape, '测试集:', test.shape)
print('提交样例列:', submission.columns.tolist())
```

打印形状能立刻发现两件事：训练集和测试集的特征列应当一致，提交样例只有「标识列 + 待填列」。这一步看起来简单，但它决定了后面所有代码的接口——比如特征列必须从 `train` 里去掉 `event` 得到，而不是手工列一遍，否则很容易漏列或错列。

接着看标签分布和缺失比例：

```python
TARGET = 'event'
print(train[TARGET].value_counts(normalize=True))

# 每一列的缺失比例，按从高到低排
na_ratio = train.isna().mean().sort_values(ascending=False)
print(na_ratio.head(15))
```

标签分布几乎肯定是**不均衡**的（破产是少数事件），这提醒我们不能用准确率评估，必须用 AUC 这类对类别比例不敏感（相对而言）的排序指标。缺失比例则告诉我们：哪些字段需要重点处理，以及「缺失」是不是成片出现——如果某些财务指标只在特定行业缺失，那缺失本身就和行业强相关，简单填一个中位数可能会抹掉这层信息。

再看字段类型，把列分成数值型和类别型两组：

```python
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in train.columns if c not in cat_cols + [TARGET]]

print('类别列:', cat_cols)
print('数值列数量:', len(num_cols))

# 数值列的统计摘要，重点看量纲和极端值
train[num_cols].describe().T.head(20)
```

这一步的产出直接喂给后面的预处理管道：**数值列走中位数填补，类别列走众数填补加独热编码**。为什么要分开处理？因为中位数对数字有意义，对文字没有意义；而独热编码会把一个文字列变成若干 0/1 列，不能和数值列混在一起填补。

### 2.2 数据清洗

我们不在管道外面手工清洗，而是把清洗步骤**全部放进 scikit-learn 的 `Pipeline`**。这样做有两个好处：

1. **避免信息泄漏**：交叉验证时，填补用的中位数、众数、编码用的类别表，都只在当前折的训练部分上计算，验证折完全不参与。如果先在整张表上 `fillna(train.median())`，验证折的分布信息就漏进训练了，CV 分数会虚高。
2. **训练和预测走同一条路**：测试集直接 `pipe.predict_proba(test)`，不用担心「训练时填了、预测时忘了填」。

具体配置是一个 `ColumnTransformer`，数值和类别各走一条支路：

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        # 数值列：用中位数填补。中位数比均值抗极端值，
        # 财务比率经常有离群点，用均值会被拉偏
        ('num', SimpleImputer(strategy='median'), num_cols),

        # 类别列：先填众数，再做独热编码
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # handle_unknown='ignore'：测试集出现训练时没见过的类别，
            # 编码成全 0，而不是直接报错
            ('ohe', OneHotEncoder(handle_unknown='ignore')),
        ]), cat_cols),
    ]
)
```

几个设计选择的理由：

- **中位数而不是均值**：财务指标分布常常长尾，少数极端值会把均值拉得离谱；中位数只关心「排中间的那个」，更稳。
- **众数填类别**：类别列的缺失没法用数字补，取出现最多的类别是简单且不容易引入偏见的做法。也可以把缺失单列成一类（`fillna('__MISSING__')`），但这份基线选了更保守的方案。
- **`handle_unknown='ignore'`**：这是实战中必须加的参数。测试集里如果出现训练集没见过的行业或地区，独热编码默认会抛错，整个预测就崩了；忽略未知类别则输出全 0，模型仍能靠其他特征做判断。

需要说明的是，我们没有做特征缩放。因为后面用的是基于树的模型，树只关心「某个特征是否大于某个阈值」，对特征做加减乘除不会改变分裂结果，标准化在这里是多余的计算。

### 2.3 模型训练与超参数调优

模型选用 **`HistGradientBoostingClassifier`**，即直方图梯度提升树（Histogram-based Gradient Boosting）。它的思路是：把每个连续特征离散成若干个区间（直方图），在区间上找最佳分裂点，而不是对每个取值都试一遍。这样训练速度快、内存占用小，而且原生支持缺失值，非常适合中等规模的表格数据。

超参数是固定的一组，没有做大规模搜索：

```python
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier(
    learning_rate=0.03,     # 学习率小一点，配合更多轮数，通常更稳
    max_depth=6,            # 树深 6，能拟合中等复杂度的交互
    max_iter=500,           # 最多 500 轮提升
    min_samples_leaf=20,    # 叶子至少 20 个样本，抑制过拟合
    random_state=42,        # 固定随机种子，结果可复现
)

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', model),
])
```

为什么是这几个值？

- `learning_rate=0.03` 配 `max_iter=500`：小学习率让每棵树只修正一点点，靠多轮累积逼近目标，泛化通常比「大学习率 + 少轮数」好；
- `max_depth=6`：表格数据里深度 4～8 是常见区间，太浅学不到交互，太深容易记住噪声；
- `min_samples_leaf=20`：每个叶子至少要有 20 个样本才允许分裂，这是一道防止「为个别企业量身定做规则」的闸门；
- 没有设 `early_stopping`：500 轮是上限，配合较小学习率一般不会严重过拟合；如果数据量很小，可以打开早停用验证集决定轮数。

**验证方式**是 5 折**分层**交叉验证（`StratifiedKFold`）。「分层」的意思是每一折里正负样本的比例都和全表一致——破产企业本来就少，如果随机切分导致某一折里几乎没有正例，那一折的 AUC 会非常不稳定。

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

print(f'CV AUC: {auc.mean():.5f} ± {auc.std():.5f}')
```

注意这里直接把**整个管道**交给 `cross_val_score`，而不是先手动填补再交叉验证。这正是管道存在的意义：每一折内部都会重新 `fit` 一次预处理器，验证折的统计量绝不会泄漏到训练里。

`n_jobs=-1` 表示用满所有 CPU 核心并行跑 5 折。材料里没有给出具体的 CV AUC 数值，我们也不编造；方法上可以确定的是，这个分数比「用全表统计量预处理后再 CV」得到的数字更可信。

关于调参，这份基线刻意保持克制：先建立一个**能跑通、能复现、没有泄漏**的参照点，再去想提升。对入门者来说，这比一上来堆十几个模型、调几十组参数更有价值——因为你得先知道「正常水平」在哪里，才能判断后面的改动是真提升还是噪声。

### 2.4 模型测试与指标报告

交叉验证确认流程没问题后，我们用**全部训练数据**重新拟合一次管道，再对测试集输出概率：

```python
# 用全量训练数据重训（CV 只用来评估，不用来产出最终模型）
pipe.fit(X, y)

# 取第 1 列，即「破产」这一类的概率
preds = pipe.predict_proba(test)[:, 1]

# 填进提交样例，保持原有的行顺序和标识列不动
submission['prediction'] = preds
submission.to_csv('submission.csv', index=False)

print(submission.head())
```

这里有三个容易出错的点，我们逐一检查：

1. **必须取 `[:, 1]`**。`predict_proba` 返回的是二维数组，第 0 列是「不破产」的概率，第 1 列才是「破产」的概率。取错列的话，AUC 会变成 `1 - 正确值`，分数直接崩到 0.1 左右。
2. **必须用样例文件的顺序**。提交时按行比对，如果我们自己重新排序、或者按 `test` 的索引赋值，行就会错位，所有预测都张冠李戴。这里直接在 `submission` 副本上加一列，天然保持了原顺序。
3. **必须检查概率范围**。可以加一行 `assert submission['prediction'].between(0, 1).all()`，确保没有 NaN、没有越界值。

**指标报告。** 比赛只看 AUC，但作为风险模型，我们至少应该同时看几样东西，才能判断它是否真的可用：

- **AUC**：整体排序能力；
- **PR-AUC（平均精确率）**：在正例极少时，PR 曲线比 ROC 更能反映「前 K 个高风险企业里抓到多少真破产」；
- **KS 统计量**：风控行业常用，衡量好坏样本累积分布的最大差距；
- **校准**：把预测概率分箱，看「预测 10% 的组里实际破产率是不是约 10%」。AUC 高但校准差，意味着概率不能直接拿去做定价。

材料里的基线只报告了 AUC，其余指标属于我们建议的后续工作。需要诚实地说：**这份材料没有给出任何具体分数**，所以我们不写「CV AUC 达到 0.xx」这类没有出处的数字。它的价值在于展示了一条结构正确的基线流水线。

---

## 三、总结

本文中我们完成了哥伦比亚企业破产预测的一个基线方案。我们首先读取训练集、测试集和提交样例，确认行列结构与标签分布，并按数据类型把特征分成数值列和类别列。其次我们把清洗封装进 `ColumnTransformer`：数值列用中位数填补以抵抗长尾极端值，类别列用众数填补后做独热编码，并用 `handle_unknown='ignore'` 兜住测试集里的新类别，整个流程放在 `Pipeline` 里以避免交叉验证时的信息泄漏。接着我们用直方图梯度提升树（学习率 0.03、深度 6、最多 500 轮、叶子最少 20 样本）做 5 折分层交叉验证，以 ROC AUC 为评估指标。最后我们用全量数据重训，取 `predict_proba` 的第二列作为破产概率，按样例文件的原始顺序写入 `prediction` 列并生成提交文件。

这个基线给我们最大的启发是：**在表格类风险预测里，「流程正确」比「模型花哨」更重要**。填补和编码必须放进管道、交叉验证必须分层、概率必须取对列、提交必须对齐行序——这四件事任何一件做错，分数都会失真，而且错得悄无声息。等这条基线稳稳跑通之后，再去尝试类别权重、早停、特征选择或者模型融合，才知道每一步到底带来了多少真实提升。