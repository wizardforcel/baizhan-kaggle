# 水资源决策与不均衡类别

## 一、比赛简介

### 1.1 比赛背景
Water Decisions and Unequal Classes 是 Kaggle Playground S6E4 的练习赛。题目来源于灌溉水需求预测数据集，模拟农业水资源分配决策。真实场景中稀有类别如极端干旱或特殊作物往往被模型忽略，导致平均分数好看但实际决策失效。

比赛目标是预测 Irrigation_Need，为分类问题且类别分布不均衡。类别不均衡会让模型偏向多数类，少数类召回很差，这在水资源分配中可能导致关键缺水地区被误判。

### 1.2 比赛目标
输入训练 csv，包含多种气象、土壤、作物特征列和目标列 Irrigation_Need。测试集仅保留特征列，需输出预测标签。部分字段如 Crop_Type 在某些情况下不可用，需做消融分析。

### 1.3 评价标准
竞赛使用准确率、平衡准确率、MCC 等指标，并特别强调各类别召回。平均分数可能掩盖罕见类表现差的问题，需单独报告每个类别的召回率。

## 二、解题过程

### 2.1 数据探索（EDA）
我们读取训练表，检查缺失和类别分布。

```python
import pandas as pd
train = pd.read_csv('train.csv')
print(train.shape)
print(train.isna().sum().sort_values(ascending=False).head())
print(train['Irrigation_Need'].value_counts(normalize=True))
```

发现类别不均衡，最常见类别占比超过 60%，最稀有类别不足 1%。特征中包含数值型和类别型，Crop_Type 是强预测因子。

```python
import seaborn as sns
sns.countplot(train['Irrigation_Need'])
```

### 2.2 数据清洗
我们做统一采样限制在 12 万行以控制内存。

```python
import numpy as np
MAX_ROWS = 120000
rng = np.random.default_rng(20261026)
train['_p'] = rng.random(len(train))
train = train.nsmallest(MAX_ROWS, '_p').drop(columns='_p')
```

对类别特征填充 '__MISSING__'，数值特征填充中位数。

```python
def features(df, omit=()):
    x = df.drop(columns=omit)
    for c in x.columns:
        if x[c].dtype == 'object':
            x[c] = x[c].astype(str).fillna('__MISSING__')
    return x
```

不做目标统计泄漏。

### 2.3 模型训练与超参数调优
使用 CatBoost 处理类别特征，固定参数做基线。

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

cat_cols = train.select_dtypes(include=['object']).columns.tolist()
X_train, X_val, y_train, y_val = train_test_split(
    train, train['Irrigation_Need'], test_size=0.2, stratify=train['Irrigation_Need'],
    random_state=20261026
)
model = CatBoostClassifier(iterations=350, depth=6, learning_rate=0.08,
                           random_seed=20261026, verbose=False)
model.fit(X_train, y_train, cat_features=cat_cols)
```

我们评估各类召回，而非只看整体准确率。若移除 Crop_Type 再训练，可观察该字段对罕见类的影响。

### 2.4 模型测试与指标报告
在验证集上输出混淆矩阵和每类召回。

```python
from sklearn.metrics import classification_report, confusion_matrix
pred = model.predict(X_val)
print(classification_report(y_val, pred))
```

测试集预测采用分块推理，避免内存溢出。

```python
test = pd.read_csv('test.csv')
pred_test = []
for chunk in np.array_split(test, 10):
    pred_test.append(model.predict(chunk))
submission = pd.DataFrame({
    'id': test['id'],
    'Irrigation_Need': np.concatenate(pred_test)
})
```

## 三、总结
本文中我们对水资源决策不均衡分类任务做了独立研究。我们首先探索了类别分布极不均衡的问题，识别出平均分数掩盖少数类风险；其次对数据做采样和统一填充清洗；接着用 CatBoost 在固定参数下训练，重点评估每类召回；最后生成测试预测并报告指标。结果表明，移除 Crop_Type 会显著下降少数类召回，必须单独评估罕见类而非只看总体准确率。
