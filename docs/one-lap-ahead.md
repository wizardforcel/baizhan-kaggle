# One Lap Ahead：预测车手下一圈是否进站

## 一、比赛简介

### 1.1 比赛背景

一级方程式（F1）比赛中，进站（pit stop）是影响比赛结果的关键决策：换胎能提升圈速，但进站本身又要损失二十多秒。车队需要在「现在进站会不会划算」之间做判断。如果我们能提前一圈预测某位车手下一圈是否会进站，就能为策略分析提供一个有用的信号。

这个比赛的数据来自 Kaggle Playground 系列 S6E5，是一个**合成的 F1 策略数据集**。所谓「合成」，是指数据是按照真实比赛的统计规律生成的，而不是从真实比赛直接抓取。比赛被设计成一个二元分类任务：给出一圈结束时的各种状态（名次、轮胎、圈速、天气、赛道等），预测下一圈该车手是否进站。

需要强调的是：**预测「进站」并不等于找到「最优策略」**。前者只是一个观察层面的预测，后者需要考虑整场比赛的时间收益。作者在 notebook 里特意把这一点写在了显眼位置，提醒读者不要把预测结果直接当作决策建议。

### 1.2 比赛目标

输入是一张表，每行代表某位车手在某一圈的记录，包含比赛年份、比赛名称、车手、车队、圈数、当前名次、轮胎 compound、轮胎寿命、圈速、天气、赛道状态等字段。输出是一个二值标签 `PitNextLap`：

- `1`：该车手下一圈进站；
- `0`：下一圈不进站。

提交文件的第一列是样本 `id`，第二列是 `PitNextLap` 的预测概率。

### 1.3 评价标准

比赛使用 **ROC AUC（接收者操作特征曲线下面积）** 作为评分指标，越高越好。AUC 的含义是：随机抽一个正样本和一个负样本，模型给正样本打分更高的概率。0.5 相当于瞎猜，1.0 是完美排序。它只关心「排序对不对」，不要求概率校准得很准，因此特别适合类别不平衡的问题——进站是少数事件。

---

## 二、解题过程

我们按照「只使用训练集、预先固定实验设置、再做可用性消融」的原则来做。整个过程没有用排行榜分数调参。

### 2.1 数据探索（EDA）

我们首先读入三份文件，确认提交格式和字段：

```python
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')

# 提交文件的列顺序就是标准答案的列顺序：id + 目标
print(sample.columns.tolist())   # ['id', 'PitNextLap']

# 看训练集规模、目标分布、缺失情况
print(train.shape)
print(train['PitNextLap'].value_counts(normalize=True))
print(train.isna().sum().sort_values(ascending=False).head(10))
```

目标分布通常是明显不平衡的：绝大多数圈不会进站。因此我们后面评估时不能只看准确率（全猜 0 也能很高），必须看 AUC 和混淆矩阵。

我们还把训练集按「比赛 + 年份」分组，检查不同比赛、不同年份的样本量和进站率。这一步有两个目的：一是看数据有没有年份漂移，二是为后面的「跨赛事/跨年份验证」做准备。

```python
# 把 Race 和 Year 拼成一个事件组，统计每组的样本数和进站率
train['event'] = train['Race'].astype(str) + '_' + train['Year'].astype(str)
stats = train.groupby('event')['PitNextLap'].agg(['size', 'mean'])
print(stats.sort_values('size', ascending=False).head())
```

结果能看到不同比赛之间的进站率差别很明显（有的赛道就是更容易进站），这意味着**事件组是一个重要的验证维度**，随机切分会高估模型在新比赛上的表现。

### 2.2 数据清洗

这份数据整体干净，目标列没有缺失（notebook 里用断言强制了这一点：`assert not train[targets].isna().any().any()`），因此不需要补标签。我们主要做了三件事：

**第一，有界均匀采样。** 原始训练集可能很大，为了在有限内存里快速迭代，我们用固定随机种子做一次无放回的均匀采样，上限 12 万行：

```python
MAX_ROWS = 120000
rng = np.random.default_rng(SEED)

# 每一行生成一个随机数，按这个数取最小的 12 万行
train['_sampling_priority'] = rng.random(len(train))
train = train.nsmallest(MAX_ROWS, '_sampling_priority')
train = train.drop(columns='_sampling_priority').reset_index(drop=True)
```

这是「均匀随机采样」，不是按目标分层，因此不会人为改变进站比例。notebook 里也明确写了：样本上限是为了省内存，不代表全部训练数据都被用上了。

**第二，处理非数值字段。** 车手、车队、赛道、轮胎类型等都是字符串，CatBoost 可以直接吃类别特征，所以我们没有做 one-hot，而是把它们转成字符串并保留。数值字段里的 `inf` 统一替换成 `NaN`：

```python
def clean_features(frame):
    x = frame.copy()
    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]):
            # 字符串列：缺失填成一个显式的 "__MISSING__" 类别
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列：无穷大不是有效数值，替换成缺失
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return x
```

**第三，完全不使用验证集做预处理拟合。** 我们的清洗和特征变换都是「逐行无状态」的，不涉及目标统计量，也不在整张表上算均值，因此不会把验证集的信息泄漏进训练。

### 2.3 模型训练与超参数调优

我们把 12 万行按 80/20 切分，分类任务用分层抽样保证两边的正负比例一致：

```python
from sklearn.model_selection import train_test_split

X = clean_features(train.drop(columns=['id', 'PitNextLap']))
y = train['PitNextLap'].to_numpy()

indices = np.arange(len(train))
fit_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=SEED, stratify=y)
```

模型使用 **CatBoost 分类器**。选它的原因是：数据里大量字段是类别型的（车手、车队、赛道、轮胎），CatBoost 内置了类别特征处理和有序提升，不需要我们手工编码。参数在训练前就固定下来，避免看了验证分数再回头改：

```python
from catboost import CatBoostClassifier

cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

model = CatBoostClassifier(
    iterations=350,        # 树的轮数
    depth=6,               # 每棵树深度
    learning_rate=0.08,    # 学习率
    loss_function='Logloss',
    random_seed=SEED,
    verbose=False,
    thread_count=4,
    allow_writing_files=False,
)

model.fit(X.iloc[fit_idx], y[fit_idx], cat_features=cat_cols)
```

**基线对比。** 我们同时算了一个「常数基线」：用训练集的正类比例作为所有样本的预测概率。这个基线在 AUC 上等于 0.5，用来确认模型是否真的学到了东西：

```python
# 常数基线：所有样本都预测成训练集的正类比例
baseline_prob = y[fit_idx].mean()
from sklearn.metrics import roc_auc_score
print('常数基线 AUC:', roc_auc_score(y[val_idx], [baseline_prob] * len(val_idx)))
```

**消融实验。** 为了检查模型是否依赖了「不完全可用」的字段，我们去掉 `PitStop`（当前是否已进站）重训一次，在同一个验证集上比较 AUC：

```python
# 去掉 PitStop 字段，其余设置完全不变
X_ablation = X.drop(columns=['PitStop'], errors='ignore')
alt_model = CatBoostClassifier(iterations=350, depth=6, learning_rate=0.08,
                               loss_function='Logloss', random_seed=SEED, verbose=False)
alt_model.fit(X_ablation.iloc[fit_idx], y[fit_idx],
              cat_features=[c for c in cat_cols if c != 'PitStop'])
```

这一步的目的不是找更高分，而是回答一个更根本的问题：**如果去掉一个可能「太贴近答案」的字段，模型还能不能工作？** 如果 AUC 大幅下降，说明模型过度依赖这个字段，上线到真实场景时要格外小心。

**跨赛事验证。** 最后我们按 `Race_Year` 分组做一次 GroupShuffleSplit，让验证集里的比赛年份在训练时完全没见过。这模拟的是「用旧赛季预测新赛季」的真实场景：

```python
from sklearn.model_selection import GroupShuffleSplit

groups = train['Race'].astype(str) + '_' + train['Year'].astype(str)
gi, gv = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                random_state=SEED).split(X, y, groups))
gm = CatBoostClassifier(iterations=350, depth=6, learning_rate=0.08,
                        loss_function='Logloss', random_seed=SEED, verbose=False)
gm.fit(X.iloc[gi], y[gi], cat_features=cat_cols)
```

如果跨赛事 AUC 比同分布验证低不少，就说明模型对特定赛道/年份有过拟合，这正是我们做分组验证的意义。

### 2.4 模型测试与指标报告

验证集上我们报告 AUC、准确率、平衡准确率、对数损失和混淆矩阵：

```python
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             balanced_accuracy_score, log_loss, confusion_matrix)

prob = model.predict_proba(X.iloc[val_idx])[:, 1]
pred = (prob >= 0.5).astype(int)

print('AUC:', roc_auc_score(y[val_idx], prob))
print('Accuracy:', accuracy_score(y[val_idx], pred))
print('Balanced accuracy:', balanced_accuracy_score(y[val_idx], pred))
print('Log loss:', log_loss(y[val_idx], prob))
print(confusion_matrix(y[val_idx], pred))
```

正类只占少数，所以「准确率」会显得不错，但「平衡准确率」和混淆矩阵才能反映对少数类（真的进站）的召回能力。我们的模型在这两个指标上都明显优于全猜 0 的基线，说明它确实学到了进站前的模式。

确认模型可用后，我们用全部 12 万行重训，并对测试集**分块**推理。分块是为了避免一次性把几十万行读进内存：

```python
final = CatBoostClassifier(iterations=350, depth=6, learning_rate=0.08,
                           loss_function='Logloss', random_seed=SEED, verbose=False)
final.fit(X, y, cat_features=cat_cols)

submission = sample.copy()
offset = 0
for chunk in pd.read_csv('test.csv', chunksize=100000):
    n = len(chunk)
    p = final.predict_proba(clean_features(chunk[X.columns]))[:, 1]

    # 把这一块的预测写回提交表对应位置
    submission.loc[submission.index[offset:offset+n], 'PitNextLap'] = p
    offset += n

# 收尾检查：行数一致、没有缺失
assert offset == len(sample)
assert not submission['PitNextLap'].isna().any()
submission.to_csv('submission.csv', index=False)
```

注意测试集的列顺序必须和训练时完全一致，否则 CatBoost 会因为特征对不上而报错。我们顺手做了断言 `chunk.columns == train.columns - target`，提前把这类低级错误挡在门外。

---

## 三、总结

本文中我们完成了 One Lap Ahead 的完整建模流程。我们首先用 EDA 看清了数据：每行是一位车手一圈的状态，目标是下一圈是否进站，正类明显偏少，不同比赛的进站率差异很大。其次我们做了无状态的数据清洗：固定随机种子的均匀采样、把字符串字段保留为类别、把无穷大替换成缺失，并严格保证验证集不参与任何预处理拟合。接着我们用 CatBoost 在 12 万行样本上训练，固定 350 轮、深度 6、学习率 0.08，与常数基线对比，并做了 `PitStop` 字段消融和跨赛事年份验证。最后我们报告了 AUC、平衡准确率和混淆矩阵，用全量样本重训并对测试集分块推理，生成了符合格式的提交文件。

这个比赛给我们的最大启发是：**验证方式比模型复杂度更值得花心思**。同一份数据，随机切分和按赛事/年份切分给出的分数可能差很多；只有后者才能回答「模型到了新赛季还灵不灵」。另外，做预测和做决策是两件事——预测下一圈会进站，并不意味着我们应该建议车队进站，把这个边界说清楚，是数据工作者的基本诚实。