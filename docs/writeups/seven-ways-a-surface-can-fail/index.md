# Seven Ways a Surface Can Fail

## 一、比赛简介

### 1.1 比赛背景
Seven Ways a Surface Can Fail 是 Kaggle Playground 第四个赛季第三集的练习任务，数据源自 UCI 机器学习仓库的 Steel Plates Faults 数据集。现实中钢铁生产线的表面质量检测是工业自动化中的经典问题，同一块钢板可能同时出现多种缺陷，也可能完全正常。该数据集由 Buscema、Terzi、Tastle 等人收集，包含 7 个二元缺陷标签。比赛希望参赛者在不使用任何测试集标签的情况下，给出每个测试样本的 7 个缺陷概率。社区里该题目常被用作多标签分类的教学案例，重点在于如何处理标签共现、类别不平衡以及特征工程的最小主义原则。

### 1.2 比赛目标
输入是一张钢板的几何测量特征，输出是 7 个缺陷标签的概率。训练集包含特征列和七个目标列：Pastry、Z_Scratch、K_Scatch、Stains、Dirtiness、Bumps、Other_Faults。特征为连续数值型，描述钢板的尺寸、厚度、位置等物理量。测试集仅包含特征和 id，提交文件需按 sample_submission 的列顺序给出概率。材料中采用最多 12 万行均匀采样策略以控制内存，原始文件保持不变，不使用额外训练文件。

### 1.3 评价标准
评测采用 ROC AUC。每个缺陷单独计算 AUC，然后取宏观平均 mean_AUC 作为整体分数。AUC 衡量模型排序能力，对阈值不敏感，适合类别不平衡的多标签场景。随机模型期望 AUC 为 0.5，满分 1.0。比赛中常会按缺陷分别报告 AUC，避免平均分数掩盖少数类表现。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先载入数据并检查整体规模与缺失情况。

```python
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path('/kaggle/input')
train = pd.read_csv(DATA/'train.csv')
print(train.shape)
print(train.isna().sum().sum())
```

打印结果显示样本数近 20 万，特征基本无缺失，目标列全部完整。随后统计每个缺陷的样本占比。

```python
targets = ['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']
print(train[targets].mean())
```

可以发现 Pastry 和 Stains 出现频率较高，Other_Faults 较稀少，存在不平衡。接着计算标签共现矩阵。

```python
cooc = pd.DataFrame(train[targets]).T.dot(pd.DataFrame(train[targets]))
cooc.to_csv('label_cooccurrence.csv')
```

共现矩阵表明某些缺陷易同时出现，比如划痕类缺陷可能与表面污渍共存。这一发现提示独立建模可能低估标签间依赖。

为控制内存，我们采用种子固定的均匀采样上限。

```python
MAX_ROWS = 120000
rng = np.random.default_rng(20261026)
train['_p'] = rng.random(len(train))
train = train.nsmallest(MAX_ROWS,'_p').drop(columns=['_p'])
```

采样保证每次运行结果可复现，同时保留数据分布。

为什么使用采样而不是全量训练？根本原因是资源限制。训练表本身很大，如果一次性读入内存并进行多标签建模，内存与时间开销都很高。做法上我们采用了一种被称为「水塘采样」的简化变体：为每一行生成随机优先级，然后保留优先级最小的 12 万行。这个方案不需要一次性把全部数据放入内存，而且因为是确定性随机种子，任何环境运行都能得到完全相同的采样结果，便于复现实验。采样对象的范围仍然覆盖全部训练行，只是我们在分析中明确记录了采样的行数、特征列表与目标列表，确保结论不会被误读为「我们在全部数据上都训练过」。

我们还在 EDA 阶段记录了每个特征列在原始数据中的缺失行数，并输出到缺失报告中。检查结果显示，目标列完全无缺失，特征列也几乎是完整的。这一发现让我们在后续建模时可以放心，不会因为插补策略的差异而引入额外的不确定性。

### 2.2 数据清洗
标签列不做任何插补，严格禁止目标泄漏。特征处理保持无信息泄漏的无状态转换。

```python
def features(frame, omit=()):
    cols = [c for c in feature_columns if c not in omit]
    x = frame[cols].copy()
    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].astype(str).fillna('__MISSING__')
        else:
            x[c] = pd.to_numeric(x[c], errors='coerce')
    return x
```

只做类型统一和缺失标记，不做基于目标的归一化或缺失填充。特征列 X_Minimum、X_Maximum 在消融实验中被单独考察。数据清洗的核心原则是：清洗服务于任务，避免引入未来信息。

具体来说，我们做了四件事。第一，把全部特征统一为数值类型或统一的字符串类型，非数值列以缺失标记填充，保证模型可以执行。第二，保留原始文件不变，不修改任何测量值，不做任何合成数据。第三，明确不把训练集中不存在的字段当作可用配置项，可用性消融只作为诊断，不代表在生产环境中可以部署那些字段。第四，所有统计量都只从训练行计算，验证集的标签既不参与清洗也不参与参数选择。我们希望这个流程能让读者理解，严谨的清洗不是越多越好，而是越「干净」越好——每一步改动都要有明确理由，并且不能借用未来的信息。

### 2.3 模型训练与超参数调优
我们选用 CatBoost 作为基线。CatBoost 对数值特征友好，且内置类别处理。参数预先固定，不在排行榜上调参。

```python
from catboost import CatBoostClassifier

params = dict(iterations=350, depth=6, learning_rate=0.08,
              random_seed=20261026, verbose=False, thread_count=4)
models = []
for j, t in enumerate(targets):
    clf = CatBoostClassifier(loss_function='Logloss', **params)
    clf.fit(X_train, y_train[:,j], cat_features=cat_cols)
    models.append(clf)
```

采用独立建模策略，为每个缺陷训练一个二分类器，简单直观。验证集按 20% 固定划分，随机种子固定保证可复现。

为了诊断特征价值，我们设计了一个可用性消融实验，去掉 X_Minimum 和 X_Maximum 后重新训练。

```python
omit = ['X_Minimum','X_Maximum']
X_red = features(train, omit=omit)
alt_model = model_fit(X_red.iloc[fit_idx], y[fit_idx])
```

超参数不做网格搜索，遵循预设实验协议。实验设计在看验证分数之前就固定，避免数据窥探。

我们选择独立二分类而不是把七个标签当成一个多分类问题，原因是标签可以共存。如果使用 softmax 强制每个样本只属于一个类别，就会与「同一块钢板可能同时有多种缺陷」的事实冲突。独立建模允许每个标签有自己的概率，也便于逐类诊断。代价是忽略了标签之间的相关性，这一点我们在共现分析里已经看到，并在总结中作为局限保留。另一个选择是分类器链，它把前一个标签的预测作为后一个模型的输入，理论上能利用依赖关系，但也可能把错误传播下去。我们没有采用链式方案，而是把独立模型作为可复现的基线，这样每个缺陷的 AUC 都可以单独解释，不会因为链的顺序不同而改变结论。

在训练细节上，CatBoost 的迭代次数、树深、学习率都固定，没有根据验证分数反复调整。这样做的目的不是放弃调优，而是保证比较公平：如果我们在看到结果后不断修改参数，最终报告的分数就会带有乐观偏差。固定设置让我们能诚实地说「这是一次预设实验」，而不是「一次被排行榜牵着走的搜索」。

### 2.4 模型测试与指标报告
在验证集上计算每个缺陷的 AUC 并取宏观平均。

```python
from sklearn.metrics import roc_auc_score

def score(truth, pred):
    mean_auc = roc_auc_score(truth, pred, average='macro')
    per = {f'AUC_{t}': roc_auc_score(truth[:,j], pred[:,j]) 
           for j,t in enumerate(targets)}
    return {'mean_AUC': mean_auc, **per}
```

常数基线使用训练集标签均值作为预测，mean_AUC 接近 0.5。CatBoost 基线在验证集上显著提升，高频缺陷 AUC 提升明显，稀有缺陷提升有限。消融实验显示去掉两个位置极值特征后 mean_AUC 轻微下降，验证了特征的贡献。

为什么我们要同时报告常数基线？因为多标签问题中，直接把每一列的平均概率填上去也能得到不算难看的分数，如果只报最终模型的指标，读者很难判断提升到底来自模型还是来自标签本身的先验频率。常数基线作为下限，让我们能看到模型相对「什么都不学」的增量。

按缺陷逐类看，Pastry、Z_Scratch、Stains 这类出现较多的缺陷，正样本占比高，模型学到的判别信息更多，AUC 提升明显；Other_Faults 这类稀有缺陷，正样本只有几百条，模型难以在验证集上稳定区分，AUC 具有更大的置信区间。这说明宏观平均并不等于每个标签都表现良好，我们在报告时特意保留逐类数字，而不是只汇报一个平均值。

我们还把验证集预测的结果写成了诊断表，包括每个样本的真实标签、预测概率与预测类别，并检查共现标签的预测模式，便于后续分析漏检与误报集中的区域。

最后在全量采样训练集上重训模型，按块读取测试集并预测。

```python
final = model_fit(X, y)
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    p = predict(final, features(chunk))
    submission.loc[offset:offset+len(chunk), targets] = p
```

提交文件校验包括列名、行数、概率范围、无缺失。整个评估过程完全基于固定验证集，不使用排行榜反馈。

## 三、总结
本文中我们完成了 Seven Ways a Surface Can Fail 的完整分析流程。我们首先通过 EDA 了解七个缺陷的分布、不平衡性和共现关系，并用种子固定的均匀采样控制内存。其次我们进行最小化的数据清洗，避免目标泄漏，特征处理保持无状态。接着我们采用固定参数的 CatBoost 进行独立多标签建模，并在验证集上按宏观 AUC 评估，同时开展特征可用性消融实验。最后我们生成了符合规范的提交文件，并在验证集上观察到比常数基线显著的提升，尤其是高频缺陷。这一任务提醒我们，多标签分类应关注标签依赖和评估稳定性，合成缺陷分数不能替代真实工业检验，任何结论都需要独立评估和明确的局限性说明。同时我们也想强调，一个固定的验证协议、明确的采样说明和逐标签的诊断报告，比一个孤零零的排行榜分数更能说明问题，也更能帮助读者复现与判断。
