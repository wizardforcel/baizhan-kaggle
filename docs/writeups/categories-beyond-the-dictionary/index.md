# 词典之外的类别（Categories Beyond the Dictionary）

## 一、比赛简介

### 1.1 比赛背景

这个比赛来自 Kaggle 的 Playground S4E8 系列——一种用真实问题「合成」出来的比赛数据，方便大家练习特征工程和数据清洗。数据主题是**蘑菇分类**：根据菌盖、菌柄、颜色、季节等特征，判断一朵蘑菇是**可食用**还是**有毒**。它的上游来源可以追溯到公开的 UCI Mushroom 数据集；参与整理的作者特别提醒：这只是数据清洗练习，**绝不能拿这个模型去判断真实蘑菇能不能吃**。

这个比赛标题「词典之外的类别」点出了真正的难点：类别字段里充满了**意想不到的取值**。分类变量里可能混入了拼写错误、多余空格、大小写不一致，甚至是完全未知的取值。比赛的用意不是教我们默认定稿一个「清洗规则」，而是让我们**慎重对待清洗本身**：先原样保留原始数据，再设计一个可审计的规则，然后在独立留出的数据上检验「清洗到底有没有帮上忙」。

### 1.2 比赛目标

输入是 `train.csv`，每条样本是一朵蘑菇的各种特征，以及一个二分类标签 `class`（比如可食用/有毒）。特征里主要是分类变量——例如 `cap-diameter`（菌盖直径）、`stem-height`（菌柄高度）、`stem-width`（菌柄宽度）、`does-bruise-or-bleed`（是否变蓝/流血）、`season`（季节）等；这些列里出现了规范词典之外的取值，需要我们处理。测试集 `test.csv` 只有特征没有标签，我们要为每一行预测出 `class`，并按照 `sample_submission.csv` 的格式提交。

### 1.3 评价标准

它采用 **MCC（Matthews 相关系数）** 作为主指标。对二分类问题，MCC 从**四个混淆计数**算出：真正例 TP、真负例 TN、假正例 FP、假负例 FN。它的取值范围是 -1 到 1：1 表示预测和真实完全一致，0 表示瞎猜，-1 表示完全相反。

只用准确率很容易被误导——如果数据里可食用的蘑菇占了九成，那么「全部回答可食用」的正确率就有九成，但模型其实什么都没学会。MCC 同时看四个计数，标签不均衡时也能真实反映模型区分两类的能力。这也是这个 notebook 特意在 `score()` 里同时算出**准确率、平衡准确率、log_loss、AUC、MCC** 的原因：一个指标不够看，要多个指标互相印证。

---

## 二、解题过程

我们把这个比赛当作一个「**预先指定实验方案、再在留出集上检验**」的练习：固定一组 CatBoost 超参数，不碰排行榜，在固定的 20% 留出集上与「常数基线」对比，再做一次去掉 `cap-shape` 这个特征的消融实验。我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步走。

### 2.1 数据探索（EDA）

我们首先只读不写地检查输入，确定任务家族（这里是二分类 MCC 任务）、目标列和提交格式，然后分块读取训练集——用带种子的「蓄水池采样」把样本上限压到 120,000 行，既保证内存安全，又保证抽样不偏向任何区块：

```python
import numpy as np, pandas as pd
from pathlib import Path

# 比赛配置：目标列 class，任务家族 mcc（二分类 + MCC 指标）
CONFIG = {'episode': 's4e8', 'targets': ['class'], 'family': 'mcc',
          'ablation': ['cap-shape'], 'date_columns': []}
SEED = 20261026
MAX_ROWS = 120_000                 # 训练采样上限，省内存而非精简数据
np.random.seed(SEED)

# 分块读取 train.csv，用随机数做蓄水池，始终只保留 12 万行
reservoir, rng = None, np.random.default_rng(SEED)
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100_000):
    chunk['_r'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk])
    reservoir = reservoir.nsmallest(MAX_ROWS, '_r')   # 只留随机分最小的 12 万行
train = reservoir.drop(columns='_r').reset_index(drop=True)
print(train.shape[0], '行被采入')
```

我们还顺手统计了每列的缺失情况，写成 `missingness.csv`。这一步让我们心里有数：哪些类别字段存在空值，后面清洗时要统一处理。

接着我们确认目标 `class` 确实是两类，并把类别文本编码成 0/1：

```python
classes = sorted(train['class'].unique().tolist())
mapping = {v: i for i, v in enumerate(classes)}      # 把文本标签映射成整数
y = train['class'].map(mapping).to_numpy()
print('类别编码：', mapping, '；类别数：', len(classes))   # 断言必须有且只有两类
```

### 2.2 数据清洗

这份数据的「脏」，重点是分类列里那些**规范之外的取值**。我们采取的清洗策略是一套「不进特征、不学全局统计」的无状态转换：数值列转成数值并处理无穷值；分类列统一转成字符串，并把缺失值填成一个显眼的 `__MISSING__` 标记。这样设计有两个目的——第一，**保留原始取值**，不把"可能有效、可能损坏、可能缺失"的取值一锅端替换成众数；第二，转换可逆、可审计，评审随时能还原出最初的源数据：

```python
feature_columns = [c for c in train.columns if c not in CONFIG['targets'] + ['id']]

def features(frame, omit=()):
    """无状态特征转换：分类列字符串化，缺失填显式标记，不做全局统计"""
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if not pd.api.types.is_numeric_dtype(x[c]):
            # 分类列：转字符串，空值填 '__MISSING__'
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列：尽量转 float，无穷值变成 NaN
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return x

X = features(train)
```

**关于「词典之外的类别」，我们的观点是：不能一概而论。** 一个陌生的取值可能是有效的新类别，可能是损坏的数据，也可能是本来就缺失。如果无脑把不常见的取值都替换成最常见的词，会把「两类长得不一样却不同的对象」误合并成同一类。所以本方案没有强行使「清洗规则」，而是让梯度提升模型在保留原始类别的基础上自己学会处理低频类别——这样我们提交的基线更诚实，再把「去掉 `cap-shape` 会怎样」作为一个独立的消融实验去回答。

### 2.3 模型训练与超参数调优

**数据划分。** 我们按 8:2 切分，并且用 `stratify=y` 做分层抽样，让留出集里两类的比例和全量一致。关键纪律是：**留出集的标签绝不参与预处理、早停或调参**，所有结果都在固定划分上一次性算出：

```python
from sklearn.model_selection import train_test_split

indices = np.arange(len(train))
fit_idx, val_idx = train_test_split(
    indices, test_size=.2, random_state=SEED, stratify=y)
print(len(fit_idx), '行训练，', len(val_idx), '行验证（分层抽样）')
```

**基线：常数预测。** 我们先算一个「什么特征都不用」的下限——直接用训练集里的类别比例当概率，看 MCC 能拿多少。这个下限很重要，因为只有比常数基线明显更好，才证明特征和模型真的起作用：

```python
def constant_prediction(target, n):
    """常数基线：直接输出训练集的类别比例，不学任何特征"""
    return np.tile(np.bincount(target, minlength=len(classes)) / len(target), (n, 1))
```

**主模型：固定参数的 CatBoost。** 我们用 CatBoost 分类器，超参数**提前定死**：`iterations=350`、`depth=6`、`learning_rate=0.08`、固定随机种子。之所以选 CatBoost，是因为它对分类特征支持很好（可以直接吃字符串列，不用手动 one-hot），刚好契合本比赛「类别列又杂又多」的特点。训练时我们显式告诉它哪些列是分类列：

```python
from catboost import CatBoostClassifier

# 超参数提前定死，不在留出集上反复搜索，避免选择偏差
params = dict(iterations=350, depth=6, learning_rate=.08,
              random_seed=SEED, verbose=False, thread_count=4,
              allow_writing_files=False)
model = CatBoostClassifier(loss_function='Logloss', **params)
# cat_features 直接告诉模型哪些列是类别列，省去手工 one-hot
model.fit(X.iloc[fit_idx], y[fit_idx],
          cat_features=X.select_dtypes(['object', 'string']).columns)
```

**调参策略：不调。** 这里反而要强调，我们**没有**用交叉验证去搜索超参数，也没有对着排行榜魔改配置——这就是材料里反复强调的「预指定实验」精神：在看过分数之前把方案定下来，避免在留出集上反复试探造成「选择偏差」。我们最多做的对比实验只有两个：常数基线 vs CatBoost，以及去掉 `cap-shape` 后的消融。

### 2.4 模型测试与指标报告

评分函数我们一次写好：对二分类同时算准确率、平衡准确率、log_loss、AUC 和 MCC。MCC 用的是 sklearn 官方的 `matthews_corrcoef`，它的值由四个混淆计数共同决定：

```python
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             log_loss, roc_auc_score, matthews_corrcoef)

def score(truth, p):
    """对验证集同时报告多个二分类指标，重点是 MCC"""
    labels = p.argmax(axis=1)                          # 概率取最大者作为预测类别
    return {
        'accuracy': accuracy_score(truth, labels),
        'balanced_accuracy': balanced_accuracy_score(truth, labels),
        'log_loss': log_loss(truth, p, labels=range(len(classes))),
        'AUC': roc_auc_score(truth, p[:, 1]),
        'MCC': matthews_corrcoef(truth, labels),       # 综合四个混淆计数
    }

pred = model.predict_proba(X.iloc[val_idx])
results = {
    'constant': score(y[val_idx], constant_prediction(y[fit_idx], len(val_idx))),
    'boosting': score(y[val_idx], pred),
}
```

第二个实验是**消融**：在同样的留出集上，把 `cap-shape`（菌盖形状）从特征里拿掉，重新训练同样的模型，比较指标变化。答案「先看数据再说」：如果去掉后指标几乎不变，说明这个字段信息量不大；如果明显变差，说明它是有效信号。无论哪个结果，我们都会把完整诊断保存下来——即使消融的结果反而更好，这个事实本身也值得记录，因为它是「哪些字段真正重要」的证据：

```python
omit = [c for c in CONFIG['ablation'] if c in feature_columns]
reduced = features(train, omit)                        # 丢掉 cap-shape 的特征表
alt = CatBoostClassifier(loss_function='Logloss', **params)
alt.fit(reduced.iloc[fit_idx], y[fit_idx], cat_features=reduced.select_dtypes(['object', 'string']).columns)
results['availability_ablation'] = score(y[val_idx], alt.predict_proba(reduced.iloc[val_idx]))
```

**生成提交文件。** 确认方案可用后，我们用采样后的样本（上限 12 万行）重新拟合一遍模型，再对测试集**分块**推理（每块 10 万行），一块一块地填进提交表里。因为测试文件的顺序可能被破坏，我们每次写入都断言块内 `id` 顺序和 `sample_submission.csv` 完全一致，最后再做一次「读回来 → 对比列名和行数」的往返校验：

```python
final = CatBoostClassifier(loss_function='Logloss', **params)
final.fit(X, y, cat_features=X.select_dtypes(['object', 'string']).columns)

sample = pd.read_csv(DATA/'sample_submission.csv')
submission, offset = sample.copy(), 0
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100_000):
    # 断言测试块和提交表这一段 id 顺序完全一致，避免错位
    n = len(chunk)
    assert np.array_equal(chunk[id_col].to_numpy(), sample.iloc[offset:offset+n, 0].to_numpy())
    p = final.predict_proba(features(chunk))
    submission.loc[submission.index[offset:offset+n], 'class'] = np.asarray(classes)[p.argmax(1)]
    offset += n

# 往返校验：确认写出去的提交能原样读回、列名和格式都正确
submission.to_csv('submission.csv', index=False)
roundtrip = pd.read_csv('submission.csv')
assert list(roundtrip.columns) == list(sample.columns) and len(roundtrip) == len(sample)
```

关于结果的解读，材料本身没有记录那张验证表的**具体数字**——它只把流程和代码完整留下来，强调「先看结果、再解释」。我们能确定的是：比较常数基线和 CatBoost 的 MCC，就能判断这套清洗+建模方案到底带来了多少增益；而消融实验能告诉我们 `cap-shape` 是否值得留。我们把结果表（`validation_scores.csv`）、混淆矩阵和逐样本诊断都导出保存，这是可复现实验的底线。

---

## 三、总结

本文中我们学习了「词典之外的类别」这个以数据清洗为主题的比赛。我们首先做了 EDA：分块蓄水池采样到 12 万行，确认二分类目标和缺失情况。其次我们坚持「无状态清洗」——保留原始类别取值、缺失值填显式标记，而不是盲目替换成众数。接着我们用固定超参数的 CatBoost（350 轮、深度 6、学习率 0.08）与常数基线对比，并用去掉 `cap-shape` 的消融实验验证特征价值，全程不碰排行榜。最后我们分块对测试集推理，做了严格的提交格式往返校验。

这个比赛给我们的最大启发是：**清洗要可审计、结论要在独立留出的数据上检验**。陌生类别可能是噪音也可能是信号，与其武断地「清洗」掉，不如让模型在保留原始取值的前提下自己判断，再用消融实验去验证每一个清洗决定到底值不值。这也呼应了作者反复强调的一句话——这是数据清洗练习，不是对真实蘑菇的食品安全建议。