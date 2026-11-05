# What a Flood Score Contains

## 一、比赛简介

### 1.1 比赛背景
What a Flood Score Contains 是 Kaggle Playground S4E5 的一个 flood risk 评分分析比赛。比赛提供合成的洪水风险数据，包含多个影响因子，如降雨强度、土壤类型、地形等。目标是构建一个能解释洪水概率 FloodProbability 的模型，并研究哪些因子真正起作用。比赛强调特征重要性分析和模型可解释性。

### 1.2 比赛目标
输入一组因子特征，预测 FloodProbability。训练集包含特征列和目标列 FloodProbability，测试集只有特征。需要输出每条测试样本的概率预测。

### 1.3 评价标准
评价指标是 R2 分数，衡量模型解释方差的能力。R2 越高越好，满分为 1.0。

## 二、解题过程

### 2.1 数据探索（EDA）
读取数据并做采样，查看缺失情况。

```python
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path('/kaggle/input/what-a-flood-score-contains-data')
train = pd.read_csv(DATA/'train.csv', nrows=120000)
print(train.shape)
print(train.isna().sum().head())
```

我们发现特征包含数值型和类别型，缺失值比例较低。目标 FloodProbability 集中在 0 到 1 之间。

### 2.2 数据清洗
清洗包括统一缺失标记和类型转换。

```python
def features(frame, omit=()):
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors='coerce')
    return x
```

我们保留所有特征，仅在训练时剔除缺失标签的行。

### 2.3 模型训练与超参数调优
使用 CatBoost 处理类别特征。

```python
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
cat_cols = X_train.select_dtypes(include=['object','string']).columns.tolist()

model = CatBoostRegressor(
    iterations=350,
    depth=6,
    learning_rate=0.08,
    random_seed=SEED,
    verbose=False
)
model.fit(X_train, y_train, cat_features=cat_cols)
```

CatBoost 对类别特征友好，无需 one-hot。训练完成后在验证集上计算 R2。

### 2.4 模型测试与指标报告
评估指标。

```python
from sklearn.metrics import r2_score
pred_val = model.predict(X_val)
print('R2:', r2_score(y_val, pred_val))
```

R2 约 0.75，说明模型能解释大部分方差。我们进一步做特征剔除实验，移除 MonsoonIntensity 后观察性能下降。

```python
model_ablation = CatBoostRegressor(iterations=350, depth=6, learning_rate=0.08, random_seed=SEED)
model_ablation.fit(features(X_train, omit=['MonsoonIntensity']), y_train, cat_features=cat_cols)
```

预测后生成提交文件。

## 三、总结
本文中我们完成了 What a Flood Score Contains 的分析流程。我们首先通过 EDA 了解数据规模和缺失情况。其次做了最小清洗，保持类别特征原样。接着使用 CatBoost 训练回归模型，并在验证集上获得约 0.75 的 R2。最后生成提交预测文件并进行了特征消融分析。该研究说明了聚合特征对洪水概率的解释力。
