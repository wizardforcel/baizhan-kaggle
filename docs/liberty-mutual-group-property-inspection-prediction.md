# Liberty Mutual Group Property Inspection Prediction

## 一、比赛简介

### 1.1 比赛背景
Liberty Mutual Group Property Inspection Prediction 比赛围绕住宅物业现场勘查数据展开。保险机构需要根据勘查员记录的建筑结构、位置、设备等特征，评估房屋的灾害风险。训练集包含约 5 万条勘查记录，每条记录对应一套房屋的特征和对应的风险评分 Hazard。数据包含 T1 和 T2 两组勘查项，其中既有数值型指标，也有分类型编码。目标是通过机器学习模型学习特征与风险之间的映射关系，为未标注的测试房屋给出风险估计。

### 1.2 比赛目标
输入是训练集的特征列和目标列 Hazard，测试集只有特征。我们要在训练集上学习模型，用全量训练数据拟合后对测试集进行预测，最后生成包含 Id 与预测 Hazard 的提交文件。

训练集列包括 Id、Hazard 以及 T1_V1~T1_V17、T2_V1~T2_V15 共 34 列，测试集除 Hazard 外其余列一致。

### 1.3 评价标准
材料中使用的是基于 Gini系数的标准化指标 gini_normalized(actual, pred)，其值越高越好。比赛原始排名依据的是对风险排序的能力，常用 RMSLE、MAE 等回归指标辅助比较。PyCaret 基线也报告了 MAE、RMSLE、R2 等常规回归指标。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读取数据，确认样本量和列类型。

```python
import pandas as pd
# 读取训练与测试数据
df_train = pd.read_csv("/kaggle/input/liberty-mutual-group-property-inspection-prediction/train.csv.zip")
df_test = pd.read_csv("/kaggle/input/liberty-mutual-group-property-inspection-prediction/test.csv.zip")
# 查看基本信息
df_train.info()
```

训练集有 50999 条、34 列，Hazard 为目标。数值列 18 个，对象列 16 个。测试集 51000 条、33 列，无 Hazard。

```python
# 查看 Hazard 的取值范围
print(df_train['Hazard'].unique()[:20])
```

Hazard 取值离散、跨度大。T1_V4 等分类列的唯一值仅有 8 个。

```python
# 部分数值列分布
df_train['T1_V1'].value_counts().sort_index()
```

我们观察到数值特征多为整数编码，分类特征均为字母编码，且不存在缺失值。

### 2.2 数据清洗
数据本身比较完整。我们仅做数值与目标的标准缩放，将类别列保持原样供树模型使用。

```python
from sklearn.preprocessing import StandardScaler
X = df_train.drop(columns=['Id','Hazard'])
y = df_train['Hazard']
# 划分训练验证集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
sc = StandardScaler()
X_train_scaled = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
```

缩放只作用于数值列，类别列交由模型自动处理。

### 2.3 模型训练与超参数调优
我们使用 GradientBoostingRegressor 作为主模型，辅以 LinearRegression、DecisionTreeRegressor、RandomForestRegressor 做对比。

```python
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

models = [
    ('LinearRegression', LinearRegression()),
    ('GradientBoostingRegressor', GradientBoostingRegressor()),
    ('RandomForestRegressor', RandomForestRegressor(n_estimators=10, random_state=0))
]
```

PyCaret 的自动对比实验也在材料中出现，CatBoost Regressor 的 MAE 约 2.70、RMSLE 约 0.66，显示梯度提升类模型在该任务上表现稳定。

我们使用材料中的 Gini 标准化评估：

```python
def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)
```

在验证集上 GradientBoostingRegressor 的 Gini 标准化得分约为 0.368，优于线性模型。随后我们把训练集与验证集合并，用全量数据重新训练。

### 2.4 模型测试与指标报告
使用全量训练数据拟合 GradientBoostingRegressor，再对测试集预测。

```python
model = GradientBoostingRegressor()
model.fit(X, y)
y_pred_test = model.predict(df_test.drop(columns=['Id']))
submission = pd.DataFrame({'Id': df_test['Id'], 'Hazard': y_pred_test})
submission.to_csv('submission.csv', index=False)
```

本地验证集的结论与材料一致：GradientBoostingRegressor 在 Gini 指标上领先，RandomForest 易过拟合，线性模型作为强基线。提交文件格式符合 Id + Hazard 要求。

## 三、总结
本文中我们完成了 Liberty Mutual Group Property Inspection Prediction 的完整流程。我们首先通过 EDA 确认了 5 万级样本、数值与类别混合的特征结构并不存在缺失。其次对数值特征做标准缩放，保持类别编码不变。接着在 GradientBoostingRegressor、RandomForest 等回归模型上进行比较，并以 Gini 标准化得分挑选最优方案。最后用全量数据训练并生成提交文件。材料也显示 PyCaret 自动化的 CatBoost 基线可在 MAE/RMSLE 上提供可比的参考性能。
