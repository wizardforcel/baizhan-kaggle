# 第4章 表格数据比赛详解：房价预测（House Prices）

House Prices（房价预测）是 Kaggle 上最经典的入门表格比赛之一，数据来自美国爱荷华州艾姆斯（Ames）市的房屋成交记录。每一行代表一套房子，包含 79 个特征，涵盖房屋结构、位置、质量、装修情况等信息，目标是预测房屋销售价格 `SalePrice`。

本章我们把第3章讲的通用清洗与建模方法，完整地应用到 House Prices 比赛中，并对比说明每一步的决策理由。你可以把这一章当作「表格比赛解题的完整示例」，后面参加其他表格比赛时，可以套用同样的流程。

---

## 4.1 比赛背景与数据

### 比赛背景

House Prices 比赛由房屋成交数据组成，每套房子有 79 个特征，包括：
- 房屋结构：总面积、房间数、卫浴数、地下室面积、车库面积等
- 位置：街区（Neighborhood）、区划（MSZoning）
- 质量：整体质量（OverallQual）、外墙质量（ExterQual）、厨房质量（KitchenQual）等
- 时间：建成年份（YearBuilt）、翻修年份（YearRemodAdd）、出售年份（YrSold）

训练集约 1460 条记录，测试集约 1459 条。每条记录都有唯一编号 `Id`，训练集包含目标列 `SalePrice`，测试集没有。

### 评价指标

比赛使用 RMSE（均方根误差）评估，且社区里普遍对 `SalePrice` 取对数后计算 RMSE（「log 空间 RMSE」）。对数变换让大房子的绝对误差在评分中不至于压倒小房子，分数越低越好。

---

## 4.2 探索性数据分析（EDA）

在清洗和建模之前，先把数据摊开看，了解目标变量分布、特征相关性、缺失模式。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
train = pd.read_csv('/kaggle/input/house-prices-regression/train.csv')
test  = pd.read_csv('/kaggle/input/house-prices-regression/test.csv')
sample_sub = pd.read_csv('/kaggle/input/house-prices-regression/sample_submission.csv')

print("训练集形状:", train.shape)   # (1460, 81)
print("测试集形状:", test.shape)   # (1459, 80)
```

### 查看目标变量 SalePrice 的分布

```python
plt.figure(figsize=(8, 5))
plt.hist(train['SalePrice'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('SalePrice')
plt.ylabel('Count')
plt.title('Distribution of SalePrice')
plt.show()

print(train['SalePrice'].describe())
```

**观察**：
- `SalePrice` 呈现**右偏分布**，均值被少数高价房拉高。
- 对数变换后更接近正态分布，适合用对数空间 RMSE 做评估。

### 查看特征与目标的相关性

```python
# 只看数值型特征与 SalePrice 的相关系数
corr = train.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
print(corr.head(15))
```

典型输出：

```
SalePrice        1.000000
OverallQual      0.793985
GrLivArea        0.708671
GarageCars       0.642407
GarageArea       0.623431
TotalBsmtSF      0.613581
1stFlrSF         0.605852
```

**观察**：`OverallQual`（房屋整体质量）与房价相关性最高（0.79），说明质量越好的房子通常越贵。`GrLivArea`（地上面积）也与房价强正相关。

### 查看缺失值模式

```python
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing.head(15))
```

**观察**：
- `PoolQC`、`MiscFeature`、`Alley`、`Fence`、`FireplaceQu` 缺失较多，这些缺失通常表示「没有泳池/没有围栏/没有壁炉」，而非数据损坏。
- `LotFrontage`、`GarageYrBlt` 等也有缺失，需要按语义处理。

---

## 4.3 数据清洗

按照第3章的方法，按语义填充缺失值，而不是简单删行。

```python
# 保存 Id
train_id = train['Id']
test_id  = test['Id']

# 分离特征和目标
train_features = train.drop(['Id', 'SalePrice'], axis=1)
test_features  = test.drop(['Id'], axis=1)
y_train = np.log1p(train['SalePrice'])  # 目标取对数

# 合并训练集和测试集统一清洗
all_data = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)

# 1. 缺失表示「没有这个东西」：填 'None'
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
for col in none_cols:
    all_data[col] = all_data[col].fillna('None')

# 2. 缺失表示数量为 0：填 0
zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
             'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for col in zero_cols:
    all_data[col] = all_data[col].fillna(0)

# 3. LotFrontage 用 Neighborhood 分组中位数填充
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# 4. 其余缺失：类别用众数，数值用中位数
for col in all_data.columns:
    if all_data[col].isnull().sum() > 0:
        if all_data[col].dtype == 'object':
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
        else:
            all_data[col] = all_data[col].fillna(all_data[col].median())

print("清洗后剩余缺失值总数:", all_data.isnull().sum().sum())
```

**决策理由**：
- `PoolQC` 等缺失填 `'None'`，保留「没有泳池」的信息，而不是强行填一个质量等级。
- `LotFrontage` 按 `Neighborhood` 分组填中位数，因为不同街区的地产线长度分布不同，全局中位数不够精确。
- 目标变量取 `log1p` 变换，让右偏的房价分布更接近正态，模型更容易拟合。

---

## 4.4 特征工程

把现有列组合出更有信息量的新特征，并处理类别变量。

```python
# 1. 构造组合特征
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + \
                        all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['IsRemodeled'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)

# 2. 序号类别用有序映射
ordinal_map = {
    'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'FireplaceQu': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
}
for col, mapping in ordinal_map.items():
    if col in all_data.columns:
        all_data[col] = all_data[col].map(mapping)

# 3. 非序号类别用独热编码
all_data = pd.get_dummies(all_data, drop_first=True)

print("特征工程后特征数:", all_data.shape[1])
```

**决策理由**：
- `TotalSF` 把地下室、一楼、二楼面积合并，更好地捕捉居住规模。
- `TotalBath` 把全卫浴和半卫浴合并，半卫浴按 0.5 计算，更符合实际使用价值。
- `HouseAge` 直接反映房子新旧，比单独看 `YearBuilt` 更直观。
- `IsRemodeled` 区分翻修房，翻修房通常价格更高。
- 质量等级（`ExterQual` 等）有明确的优劣顺序，用有序映射保留这种关系。

---

## 4.5 模型训练与调优

将数据拆回训练集和测试集，训练模型并用交叉验证评估。

```python
# 拆回训练集和测试集
n_train = len(train_features)
X_train = all_data.iloc[:n_train]
X_test  = all_data.iloc[n_train:]

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

# 岭回归基线
ridge = Ridge(alpha=10.0, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(ridge, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"岭回归 5折 RMSE (log空间): {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")

# 也可以试随机森林作为对比
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
scores_rf = cross_val_score(rf, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
rmse_rf = np.sqrt(-scores_rf)
print(f"随机森林 5折 RMSE (log空间): {rmse_rf.mean():.4f} (±{rmse_rf.std():.4f})")
```

**决策理由**：
- 选用 Ridge 回归做基线：L2 正则能防范多重共线性，在小样本（1460 条）上比普通线性回归更稳。
- 也试过 RandomForest 做对比：树模型在小样本上容易过拟合，Ridge 在对数空间的 5 折 RMSE 更稳定。
- 5 折交叉验证避免单次划分的运气成分，得到更可靠的估计。

---

## 4.6 模型测试与提交

训练完成后，对测试集做预测，还原回原始价格空间，生成提交文件。

```python
# 训练最终模型
ridge.fit(X_train, y_train)

# 对测试集预测（log空间）
test_pred_log = ridge.predict(X_test)

# 还原回原始价格空间
test_pred = np.expm1(test_pred_log)

# 生成提交文件
submission = pd.DataFrame({
    'Id': test_id,
    'SalePrice': test_pred
})

# 上线前核对清单
assert len(submission) == len(test), "提交行数与测试集不一致"
assert list(submission.columns) == ['Id', 'SalePrice'], "列名不一致"
assert submission['SalePrice'].isnull().sum() == 0, "预测结果包含缺失值"
assert (submission['SalePrice'] > 0).all(), "房价不能是负数"

submission.to_csv('submission.csv', index=False)
print("提交文件已生成：submission.csv")
print(submission.head())
print(f"\n预测房价范围：{test_pred.min():.2f} ~ {test_pred.max():.2f}")
print(f"预测房价均值：{test_pred.mean():.2f}")
```

**核对说明**：
- 行数必须与测试集一致（1459 行），不能多也不能少。
- 列名必须是 `Id` 和 `SalePrice`，与 `sample_submission.csv` 严格匹配。
- `Id` 顺序与测试集一一对应，不能打乱。
- 预测结果不能有缺失值，房价不能是负数。

把 `submission.csv` 上传到 Kaggle 比赛页面，你就能在排行榜上看到自己的名次了。

---

## 4.7 小结

本章我们以 House Prices 房价预测为例，完整走了一遍表格比赛解题流程：

1. **比赛背景**：Ames 房价数据集，1460 条训练记录、79 个特征，预测 `SalePrice`，评价指标为 log 空间 RMSE。
2. **EDA**：发现 `SalePrice` 右偏，`OverallQual` 与房价相关性最高，`PoolQC` 等缺失通常表示「没有该设施」。
3. **数据清洗**：按语义填充缺失值（没有泳池填 `'None'`，车库面积填 0，`LotFrontage` 按街区分组填中位数），目标变量取 `log1p` 变换。
4. **特征工程**：构造 `TotalSF`、`TotalBath`、`HouseAge`、`IsRemodeled` 等组合特征；质量等级用有序映射，街区等非序号类别用独热编码。
5. **模型训练**：Ridge 回归做基线，L2 正则防范多重共线性；5 折交叉验证获得稳定的 RMSE 估计；随机森林作为对比。
6. **模型测试与提交**：对测试集预测，用 `expm1` 还原价格，按 `sample_submission` 格式生成 `submission.csv`，上传前检查行数、列名、ID 顺序、缺失值和数值范围。

这个流程是表格比赛的通用模板。下一章我们将详解另一个表格比赛：信用评分（Give Me Some Credit），看看在二分类场景下，上述流程有什么共通之处和不同之处。
