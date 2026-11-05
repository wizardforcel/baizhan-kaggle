# House Prices Regression

## 一、比赛简介

### 1.1 比赛背景
经典 Ames 房价数据集用于回归预测。训练集约 1460 条，79 个特征，包含房屋结构、位置、质量等信息。测试集需预测 SalePrice。

特征包含数值型与类别型，存在缺失值、异常值与偏态。

### 1.2 比赛目标
给定房屋特征，预测销售价格 SalePrice。训练集包含 Id 与 SalePrice，测试集无 SalePrice。需要输出 Id 与预测价格。

### 1.3 评价标准
以均方根误差 RMSE 评估，且对 SalePrice 取对数后更稳健。数值越低越好。

---
## 二、解题过程

我们按 EDA、清洗、特征工程、模型训练四步进行，采用 Ridge 回归做基线。

### 2.1 数据探索（EDA）

```python
train = pd.read_csv('/kaggle/input/competitions/house-prices-regression/train.csv')
print(train.shape)               # (1460, 81)
print(train['SalePrice'].describe())
```

SalePrice 呈现右偏，对数变换后更近似正态。缺失值在 PoolQC、GarageYrBlt、LotFrontage 等列较多。

```python
missing = all_data.isnull().sum()
print(missing[missing>0].head(20))
```

### 2.2 数据清洗

对缺失值按语义填充。

```python
none_cols = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu']
for col in none_cols:
    all_data[col] = all_data[col].fillna('None')
zero_cols = ['GarageYrBlt','GarageArea','GarageCars']
for col in zero_cols:
    all_data[col] = all_data[col].fillna(0)
```

LotFrontage 用 Neighborhood 中位数填充，其余类别用众数，数值用中位数。

```python
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
```

### 2.3 模型训练与超参数调优

特征工程构造组合特征。

```python
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5*all_data['HalfBath']
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
```

有序类别映射为数值，其余类别做 LabelEncoder。

```python
ordinal_map = {
    'ExterQual': {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
    'KitchenQual': {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
}
```

目标取对数。

```python
y_train = np.log1p(train['SalePrice'])
```

模型选择 Ridge。

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=10.0, random_state=42)
```

5 折交叉验证。

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(rmse_scores.mean(), rmse_scores.std())
```

### 2.4 模型测试与指标报告

在对数空间 5 折 RMSE 约 0.13，转换回原始价格后预测范围合理。

```python
model.fit(X_train, y_train)
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': y_pred})
submission.to_csv('submission.csv', index=False)
```

---
## 三、总结

本文中我们完成了 House Prices Regression 的基线流程。我们首先通过 EDA 发现 SalePrice 右偏且存在多处缺失。其次按语义填充缺失并构造 TotalSF、TotalBath、HouseAge 等特征。接着用对数目标训练 Ridge 回归，通过 5 折交叉验证评估。最后在对数空间得到稳定 RMSE 并导出提交。

该比赛表明在结构化数据上，合理清洗与特征组合比复杂模型更关键。
