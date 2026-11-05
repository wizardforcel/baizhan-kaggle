# Light, Labels and Rare Objects

## 一、比赛简介

### 1.1 比赛背景
Light, Labels and Rare Objects 是 Kaggle Playground Series S6E6 的合成天文分类任务。数据来源于 Stellar Classification Dataset SDSS17 的模拟，包含光度、颜色指数、光谱类型等测量特征。目标是对观测对象进行类别判别，其中部分类别样本稀少，形成“稀有对象”问题。任务强调在保持可解释性的同时，对稀有类别提供可见的误差率。

### 1.2 比赛目标
给定训练表 train.csv，表中包含 id 列和目标列 class，目标是对测试集 test.csv 预测 class。训练集规模较大，材料基线使用最多 120000 条均匀抽样，特征列为除 id、class 外的所有列。CONFIG 中明确 targets=['class']，family='balanced'，ablation 关注 spectral_type 与 galaxy_population。

### 1.3 评价标准
基线脚本采用单标签分类的 accuracy、balanced_accuracy 与 log_loss 作为验证指标，并在二分类时附加 AUC、MCC。材料强调提供逐类召回与标识符删减实验，衡量稀有类别的表现。比赛核心是分类正确率与类别平衡性，数值越高越好。

## 二、解题过程

### 2.1 数据探索（EDA）
我们采用确定性水库抽样控制内存。

```python
import pandas as pd, numpy as np
MAX_ROWS = 120000
reservoir = None
# 逐 chunk 读取并均匀抽样
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    reservoir = pd.concat([reservoir, chunk]) if reservoir is not None else chunk
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')
train = reservoir.drop(columns='_sampling_priority')
```

抽样后检查目标分布是否唯一，确认不存在标签缺失。

```python
print('Class encoding:', {v:i for i,v in enumerate(sorted(train['class'].unique()))})
```

```python
missing = train.isna().sum()
```

missingness.csv 记录每个字段的缺失行数，供后续处理参考。

### 2.2 数据清洗
特征预处理采用无状态变换，避免使用目标统计量。

```python
def features(frame, omit=()):
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors='coerce')
    return x
```

数值列转数，对象列填充缺失字符串。对于日期列（本任务无）会拆成年月周。ablation 实验会剔除 spectral_type 与 galaxy_population。

### 2.3 模型训练与超参数调优
采用固定超参数的 CatBoost 分类器，避免在排行榜上调参。

```python
from catboost import CatBoostClassifier
params = dict(iterations=350, depth=6, learning_rate=.08,
              random_seed=SEED, verbose=False, thread_count=4)
model = CatBoostClassifier(loss_function='MultiClass',
                          auto_class_weights='Balanced', **params)
model.fit(X.iloc[fit_idx], y[fit_idx])
```

训练使用 80/20 分层划分，stratify=y 保证各类别比例一致。常数基线用于对比。

```python
baseline = np.tile(np.bincount(y[fit_idx], minlength=len(classes))/len(y[fit_idx]), (len(val_idx),1))
```

第二实验在同一划分上移除 ablation 字段或减小训练预算，保持预设不变。

### 2.4 模型测试与指标报告
验证集上计算 accuracy、balanced_accuracy、log_loss，并输出混淆矩阵。

```python
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
pred = model.predict(X.iloc[val_idx])
labels = pred.argmax(axis=1)
result = {'accuracy': accuracy_score(y[val_idx], labels),
          'balanced_accuracy': balanced_accuracy_score(y[val_idx], labels),
          'log_loss': log_loss(y[val_idx], pred)}
```

最后用全量抽样集 refit，逐 chunk 预测测试集并保存 submission.csv，验证列名与行数与 sample_submission 一致。

```python
final = model_fit(X, y)
submission.to_csv(OUT/'submission.csv', index=False)
```

## 三、总结
本文中我们完成了 Light, Labels and Rare Objects 的基线复现。我们首先以确定性水库抽样限制数据规模并记录缺失情况，接着采用无状态特征处理并保留类别平衡权重。随后使用固定参数的 CatBoost MultiClass 模型在 80/20 分层划分上训练，并与常数基线以及字段删减的 ablation 进行对比。最后以全量抽样集重训练并生成符合规范的提交文件。材料强调这是学习基线而非可部署的发现流水线，稀有类别的评估需要在召回与混淆矩阵中具体呈现。
