# Retention Signals from Scores to Decisions

## 一、比赛简介

### 1.1 比赛背景
这是一个保留客户的评分与决策研究。数据来源为 Kaggle Playground S4E1 的合成数据，任务是预测客户是否流失 `Exited`。虽然数据是合成的，但它保留了真实的银行客户流失特征，适合作为可解释性基准的教学案例。材料明确指出，真实的营销利润需要治疗成本和结果证据，本数据集没有这些信息，因此分数只能用于排序，不能直接等于收益。

### 1.2 比赛目标
输入每条客户记录的特征，输出 `Exited` 的概率。训练集包含 客户 ID、地理、性别、年龄、账户余额、持卡数、产品数、信贷额度、活跃账户数以及流失标签。测试集没有标签，需要提交概率预测。

### 1.3 评价标准
主要的评价指标是 ROC AUC，即模型对正负样本的排序能力。材料中还计算了 Average Precision、Log Loss、Brier Score，用来补充概率校准的评估。所有指标越高越好，但 Log Loss 越小越好。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先定位数据文件并做基本审计。

```python
from pathlib import Path
import pandas as pd
import json

DATA = Path('../data')
train = pd.read_csv(DATA/'train.csv')
test = pd.read_csv(DATA/'test.csv')
audit = {
    'train_rows': len(train),
    'test_rows': len(test),
    'positive_count': int(train.Exited.sum()),
    'prevalence': float(train.Exited.mean())
}
print(json.dumps(audit, indent=2))
```

审计显示训练集记录数、测试集记录数、流失率等。我们还用 SQL 汇总地理和产品数的流失率。

```python
import sqlite3
with sqlite3.connect(':memory:') as con:
    train[['id','CustomerId','Geography','NumOfProducts','Exited']].to_sql('records', con, index=False)
    sql_summary = pd.read_sql_query("""
        SELECT Geography, NumOfProducts, COUNT(*) AS records,
               SUM(Exited) AS exits, AVG(1.0*Exited) AS exit_rate
        FROM records GROUP BY Geography, NumOfProducts
    """, con)
```

这帮助我们确认不同分段的流失率差异，为后面分段分析提供依据。

### 2.2 数据清洗
材料中明确排除了 `id`、`CustomerId` 和 `Surname` 三列，以避免模型依赖身份捷径。

```python
excluded = ['id','CustomerId','Surname','Exited']
features = [c for c in train.columns if c not in excluded]
categorical = ['Geography','Gender']
numeric = [c for c in features if c not in categorical]
```

分类列被转为字符串，数值列保持原样。我们没有做缺失值填充，因为数据是合成的且无缺失。

```python
for c in categorical:
    train[c] = train[c].astype(str)
```

为了后续分组压力测试，我们保留了 `CustomerId` 的副本，但训练时不使用。

### 2.3 模型训练与超参数调优
我们建立了三种基线：恒定先验、逻辑回归、CatBoost。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline

def linear_model():
    prep = ColumnTransformer([
        ('numeric', StandardScaler(), numeric),
        ('category', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])
    return make_pipeline(prep, LogisticRegression(max_iter=1500, C=1.0))
```

CatBoost 模型使用固定超参数。

```python
from catboost import CatBoostClassifier

def model(seed):
    return CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05,
        loss_function='Logloss', l2_leaf_reg=5, random_seed=seed,
        verbose=False, allow_writing_files=False)
```

我们采用 3 折分层交叉验证，保证每折的流失率一致。

```python
from sklearn.model_selection import StratifiedKFold

oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=20261025)
for fold, (fit, valid) in enumerate(splitter.split(X, y)):
    cb = model(SEED+fold)
    cb.fit(X.iloc[fit], y[fit], cat_features=categorical)
    oof[valid] = cb.predict_proba(X.iloc[valid])[:,1]
    test_pred += cb.predict_proba(X_test)[:,1] / 3
```

超参数在跑完后不做搜索，保持教学可复现性。

### 2.4 模型测试与指标报告
我们计算了 OOF 的 ROC AUC、Average Precision、Log Loss 和 Brier Score。

```python
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

def metrics(y, p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return {
        'roc_auc': roc_auc_score(y, p),
        'average_precision': average_precision_score(y, p),
        'log_loss': log_loss(y, p),
        'brier': brier_score_loss(y, p)
    }
```

为了展示决策能力，我们模拟了只能审查 10% 记录的场景。

```python
CAPACITY = 0.10
k = int(np.ceil(CAPACITY*len(y)))
selected = np.argsort(-oof)[:k]
capacity_report = {
    'precision': float(y[selected].mean()),
    'recall': float(y[selected].sum()/y.sum()),
    'lift': float(y[selected].mean()/y.mean())
}
```

材料还做了 sigmoid 校准实验和分组压力测试，用 GroupShuffleSplit 保证验证集中不出现训练集的客户 ID。

```python
from sklearn.model_selection import GroupShuffleSplit
gfit, gvalid = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=20261025).split(X, y, groups=train.CustomerId))
```

最终提交文件为三折 CatBoost 的平均概率。

```python
submission = pd.DataFrame({'id': sample.id, 'Exited': test_pred})
submission.to_csv('submission.csv', index=False)
```

## 三、总结
本文中我们完成了 Retention Signals 的基准实验。我们首先通过审计和 SQL 分段确认了数据的流失率分布；其次排除了身份列并对分类与数值特征做了标准化处理；接着用 3 折 CatBoost 与逻辑回归基线进行训练，并通过容量限制报表评估决策价值；最后在未见客户 ID 的分组压力测试下验证了模型的泛化风险。整个流程强调了在合成数据上，排序质量、概率校准与部署约束之间的权衡。
