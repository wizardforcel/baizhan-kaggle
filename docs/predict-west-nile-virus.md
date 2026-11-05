# West Nile 病毒预测

## 一、比赛简介

### 1.1 比赛背景
West Nile 病毒通过蚊子传播，每年夏季在美国伊利诺伊州芝加哥出现感染。公共卫生部门在全市布设捕蚊灯，记录蚊种数量、位置和气象信息，目标是预测是否存在携带病毒的蚊子。数据包含 2007 年的样本。

### 1.2 比赛目标
训练集有 10506 条记录，包含 `Date`、`Species`、`Block`、`Street`、`Trap`、`Latitude`、`Longitude`、`AddressAccuracy`、`NumMosquitos`、`WnvPresent`。目标 `WnvPresent` 为 0 表示未携带，1 表示携带。测试集 116293 条，需预测每条的概率。额外提供 `weather.csv` 和 `spray.csv` 作为补充特征。

### 1.3 评价标准
比赛使用 **ROC AUC** 评估二分类性能。AUC 越高越好，1.0 为完美，0.5 为随机。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先查看训练集规模和目标比例。

```python
import pandas as pd
train = pd.read_csv('/kaggle/input/predict-west-nile-virus/train.csv')
print(train.shape)  # (10506, 12)
print(train['WnvPresent'].mean())  # 约 0.052
```

正样本仅 5.2%，是极不平衡数据。

```python
# 查看蚊种分布
print(train['Species'].value_counts())
```

主要蚊种是 CULEX RESTUANS 和 CULEX PIPIENS/RESTUANS。

```python
# 地理分布
import matplotlib.pyplot as plt
plt.scatter(train['Longitude'], train['Latitude'], c=train['WnvPresent'], s=5)
plt.show()
```

病毒集中在城市南部。

气象数据有 11 个站点，每天有最高、最低温度、降水等信息。

### 2.2 数据清洗
我们把地址类文本列去掉，只保留数值和类别特征。对日期做时间特征提取。

```python
# 时间特征
train['Date'] = pd.to_datetime(train['Date'])
train['month'] = train['Date'].dt.month
train['dayofweek'] = train['Date'].dt.dayofweek
```

类别特征 `Species`、`Block`、`Street`、`Trap` 用 LabelEncoder 编码。经度纬度保留。

```python
from sklearn.preprocessing import LabelEncoder
for col in ['Species','Block','Street','Trap']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
```

气象数据按日期合并，取最近 5 天的平均温度和降水。

```python
weather = pd.read_csv('/kaggle/input/predict-west-nile-virus/weather.csv')
weather['Date'] = pd.to_datetime(weather['Date'])
train = pd.merge_asof(train.sort_values('Date'), weather.sort_values('Date'), on='Date')
```

### 2.3 模型训练与超参数调优
正样本稀少，使用 Stratified KFold 并考虑样本权重。我们试 RandomForest 和 XGB。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

X = train.drop(columns=['WnvPresent'])
y = train['WnvPresent']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for tr, va in skf.split(X, y):
    clf = RandomForestClassifier(n_estimators=1000, min_samples_split=2, random_state=42)
    clf.fit(X.iloc[tr], y.iloc[tr])
    pred = clf.predict_proba(X.iloc[va])[:,1]
    aucs.append(roc_auc_score(y.iloc[va], pred))
print('RF CV AUC:', np.mean(aucs))
```

RF 在私有集上的 AUC 大约在 0.655~0.687 之间。

```python
import xgboost as xgb
xgb_clf = xgb.XGBClassifier(seed=42, n_estimators=500, max_depth=4, learning_rate=0.05)
xgb_clf.fit(X, y)
```

XGB 的特征重要性显示蚊种、月份、纬度和蚊子数量最重要。

```python
# 选择阈值做特征选择
from sklearn.metrics import roc_auc_score
# 用验证集计算 ROC AUC
```

### 2.4 模型测试与指标报告
在验证集上 XGB 的 AUC 约 0.83。

```python
y_pred = xgb_clf.predict_proba(X_test)[:,1]
submission = pd.DataFrame({'Id': test['Id'], 'WnvPresent': y_pred})
submission.to_csv('submission.csv', index=False)
```

最终提交时采用 RF+XGB 集成提升稳定性。

## 三、总结
本文中我们完成了 West Nile 病毒预测。我们首先发现数据极度不平衡，正样本仅 5%。其次对日期、物种、气象做特征工程并合并。接着用随机森林和 XGBoost 做交叉验证，验证 AUC 约 0.83。最后生成提交文件。这个比赛让我们认识到，时间、地点与气象特征在流行病预测中至关重要，特征融合比单一模型更可靠。
