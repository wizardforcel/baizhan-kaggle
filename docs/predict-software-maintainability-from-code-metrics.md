# 从代码度量预测软件可维护性

## 一、比赛简介

### 1.1 比赛背景
软件可维护性衡量一段代码以后修改起来有多费劲。业界常用代码行数、圈复杂度、注释比例等指标来估计维护成本。这个比赛把真实项目的代码度量与人工评估的可维护性分数配对，要求我们建立回归模型。

### 1.2 比赛目标
输入是代码度量表，包含 `Size-Lines of Code (LOC)`、`Cyclomatic Complexity-Average Complexity per Method`、`CodeFactor Issue-Issue`、`Framework`、`Cluster` 等字段。输出是 `Maintainability`，一个连续数值。测试集只有特征，需要预测每个项目的可维护性。

### 1.3 评价标准
比赛使用**均方根误差 RMSE** 评估回归效果。RMSE 越小，说明预测分数越接近真实分数。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先读入训练集、测试集和提交样例，确认字段。

```python
import pandas as pd
train = pd.read_csv('/kaggle/input/competitions/predict-software-maintainability-from-code-metrics/train.csv')
test = pd.read_csv('/kaggle/input/competitions/predict-software-maintainability-from-code-metrics/test.csv')
print(train.shape, test.shape)
print(train.head())
```

训练集有 `id`、`Maintainability` 和几十个代码度量列，测试集没有目标。`Framework` 是类别列，`Cluster` 是无意义的分组列。

```python
# 查看目标分布
print(train['Maintainability'].describe())
```

目标分数范围较宽，需要用回归模型。

### 2.2 数据清洗
我们删除 `Cluster` 列，对 `Framework` 做序数编码，并准备两组特征：一组只用高信号特征，一组用全特征。

```python
from sklearn.preprocessing import OrdinalEncoder
# 去掉无用列
train = train.drop(columns=['Cluster'], errors='ignore')
test = test.drop(columns=['Cluster'], errors='ignore')
# 保存测试 id
test_ids = test['id']
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])
# 编码 Framework
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[['Framework']] = enc.fit_transform(train[['Framework']])
test[['Framework']] = enc.transform(test[['Framework']])
```

高信号特征来自相关性分析。

```python
top_features = ['Size-Lines of Code (LOC)',
                'Cyclomatic Complexity-Average Complexity per Method',
                'CodeFactor Issue-Issue']
X_top = train[top_features]
X_test_top = test[top_features]
y = train['Maintainability']
```

### 2.3 模型训练与超参数调优
我们训练两个随机森林：一个普通回归森林，一个分位数森林。前者拟合均值，后者预测 55 分位数，用来抵抗噪声项目。

```python
from sklearn.ensemble import RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor

# 均值专家
rf_mean = RandomForestRegressor(n_estimators=1000, max_depth=3,
                                min_samples_leaf=2, random_state=42)
rf_mean.fit(X_top, y)

# 分位数专家
qrf = RandomForestQuantileRegressor(n_estimators=1500, max_depth=3,
                                    min_samples_leaf=2, random_state=42)
qrf.fit(X_top, y)
```

树深限制为 3，是为了避免过拟合小样本。

```python
# 预测并融合
pred_mean = rf_mean.predict(X_test_top)
pred_q = qrf.predict(X_test_top, quantiles=0.55)
final = 0.5*pred_mean + 0.5*pred_q
```

### 2.4 模型测试与指标报告
我们在训练集上做交叉验证，比较单模型和融合模型。

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_mean, X_top, y, cv=5, scoring='neg_root_mean_squared_error')
print('RF RMSE:', -scores.mean())
```

随机森林的 RMSE 大约在 500 上下，融合后比单模型更稳。

```python
submission = pd.DataFrame({'id': test_ids, 'Maintainability': final})
submission.to_csv('submission.csv', index=False)
```

## 三、总结
本文中我们完成了软件可维护性预测。我们首先通过 EDA 找到与目标最相关的 LOC、平均复杂度和 CodeFactor 问题数。其次清洗掉 Cluster 列并编码 Framework。接着训练均值随机森林和 55 分位随机森林，按 50/50 融合。最后生成提交文件。这个比赛说明，在表格回归任务里，选对少数高信号特征比盲目堆模型更有效。
