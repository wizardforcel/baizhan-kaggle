# 第3章 表格数据的通用清洗与建模方法

表格数据是 Kaggle 中最常见的格式：每一行是一个样本，每一列是一个特征。房价预测、信用评分、贷款审批、销量预测、点击率预测，绝大多数入门比赛都是表格数据。

本章我们讲表格数据的「通用清洗与建模方法」：缺失值怎么处理、类别文字怎么编成数字、怎么把现有列组合出新特征、数值怎么变换让分布更友好、选什么模型做基线、怎么评估和调参、最后怎么生成提交文件。这些方法适用于绝大多数表格比赛，是后面每个具体比赛章节的基础。

我们以 House Prices（房价预测）作为贯穿示例，带你从原始数据走到可以提交的基线模型。所有代码都可以直接在 Kaggle Notebooks 里运行。

---

## 3.1 缺失值处理：空不等于坏，按语义填充

缺失值是表格数据中最常见的问题。新手最容易犯的错误是：一看到缺失就把整行删掉，或者一律填成 0。这样做很危险，因为**缺失不等于「数据坏了」**，缺失本身往往有业务含义。

### 不同缺失含义：没有泳池 ≠ 数据丢失

以 House Prices 为例：
- `PoolQC`（泳池质量）缺失，通常意味着这套房子**没有泳池**，而不是测量数据丢失。
- `GarageYrBlt`（车库建成年份）缺失，通常意味着**没有车库**。
- `LotFrontage`（街道前面的地产线长度）缺失，可能是真正的测量缺失，可以用同一街区（Neighborhood）的中位数填充。

如果你把这些缺失一律填成 0 或删除，就会把「没有泳池」这个有用信息弄丢了。

### 按语义填充缺失值

常用的填充策略有：

| 策略 | 适用场景 | 示例 |
|------|---------|------|
| 填 `'None'` | 缺失表示「没有这个东西」 | `PoolQC`、`GarageType`、`Fence` |
| 填 `0` | 缺失表示数量为 0 | `GarageArea`、`GarageCars` |
| 填中位数 | 数值型，受极端值影响小 | `LotFrontage`、`GrLivArea` |
| 填众数 | 类别型，取出现次数最多的类别 | `MSZoning`、`Exterior1st` |
| 分组中位数 | 数值型，按某个类别分组后填该组中位数 | `LotFrontage` 按 `Neighborhood` 分组填 |
| 增加 `_is_missing` 标记 | 缺失本身就是信号 | 新增布尔列标记原本是否缺失 |

以下代码展示如何按语义填充 House Prices 的缺失值：

```python
import pandas as pd
import numpy as np

# 读取训练集和测试集，合并方便统一处理
train = pd.read_csv('/kaggle/input/house-prices-regression/train.csv')
test  = pd.read_csv('/kaggle/input/house-prices-regression/test.csv')

# 保存 Id，稍后用于生成提交文件
train_id = train['Id']
test_id  = test['Id']

# 去掉 Id 和目标列，只保留特征
train_features = train.drop(['Id', 'SalePrice'], axis=1)
test_features  = test.drop(['Id'], axis=1)

# 合并训练集和测试集，统一做清洗
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

# 4. 其余类别型用众数填充，数值型用中位数填充
for col in all_data.columns:
    if all_data[col].isnull().sum() > 0:
        if all_data[col].dtype == 'object':
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
        else:
            all_data[col] = all_data[col].fillna(all_data[col].median())

print("清洗后剩余缺失值总数:", all_data.isnull().sum().sum())
```

运行后，所有缺失值都被处理了，剩余缺失值总数应该为 0。

### 缺失标记：让缺失本身成为特征

有时候，「缺失」本身就是预测目标的有用信号。比如 `GarageYrBlt` 缺失可能意味着房子很老、从来没建过车库，这本身跟房价有关。除了填充，我们可以额外增加一列 `_is_missing` 标记原本是否缺失：

```python
for col in all_data.columns:
    if all_data[col].isnull().sum() > 0:
        all_data[col + '_is_missing'] = all_data[col].isnull().astype(int)
        # 然后再填充缺失值
        if all_data[col].dtype == 'object':
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
        else:
            all_data[col] = all_data[col].fillna(all_data[col].median())
```

这样模型既能看到填充后的数值，也能知道「这个值原来是不是缺的」。

---

## 3.2 类别变量编码：让模型读得懂文字

机器学习模型只认数字，不能直接处理「East」「West」这样的文字。类别变量编码的目的是把文字类别转成数字，但不同的类别应该用不同的编码方式。

### 序号类别用有序映射

有些类别本身有大小顺序，比如房屋质量：
- `Ex`（优秀）> `Gd`（良好）> `TA`（一般）> `Fa`（一般差）> `Po`（很差）

这种类别应该用**有序映射**，把等级转成对应的数字，保留顺序信息：

```python
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
```

### 非序号类别用独热编码

有些类别没有大小顺序，比如街区（`Neighborhood`）：`CollgCr` 不比 `OldTown` 「大」或「小」。这种类别应该用**独热编码（One-Hot Encoding）**：把一列展开成多列，每列对应一个类别，有该类别就填 1，没有就填 0。

```python
# 对所有剩余的 object 类型列做独热编码
all_data = pd.get_dummies(all_data, drop_first=True)
```

`drop_first=True` 表示去掉第一个类别作为基准，避免「虚拟变量陷阱」（多重共线性）。比如原来有 3 个街区，独热编码后只保留 2 列，第 3 列可以由前两列推断出来。

### 独热编码的注意事项

独热编码会把一列变成很多列。如果某个类别型特征的基数很大（比如有 50 个不同的街区），独热编码后会增加 50 列，导致特征维度过高，模型训练变慢、容易过拟合。遇到这种情况，可以考虑：
- 把出现次数很少的类别合并为 `'Other'`；
- 使用目标编码（Target Encoding）：用该类别对应的目标变量平均值替代类别名称；
- 直接删除基数过大且与目标相关性低的特征。

---

## 3.3 特征构造：把已有列组合出新列

有时候，原始特征单独看不明显，但把两个或多个特征组合起来，就能产生更有信息量的新特征。特征构造是表格数据比赛中提升分数的重要手段。

### 常见的组合模式

| 模式 | 示例 | 含义 |
|------|------|------|
| 加法 | `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | 总楼面面积 |
| 加法 | `TotalBath = FullBath + 0.5 * HalfBath` | 总卫浴数（半浴按 0.5 算） |
| 减法 | `HouseAge = YrSold - YearBuilt` | 房龄 |
| 乘法 | `PricePerSqFt = SalePrice / GrLivArea` | 每平方英尺价格 |
| 比率 | `RoomPerArea = TotRmsAbvGrd / GrLivArea` | 每平方英尺房间数 |

```python
# 构造总面积
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# 构造总卫浴数
all_data['TotalBath'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + \
                        all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']

# 构造房龄
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']

# 构造是否翻新
all_data['IsRemodeled'] = (all_data['YearRemodAdd'] != all_data['YearBuilt']).astype(int)
```

### 为什么组合特征有效？

- `TotalSF` 把地下室、一楼、二楼面积合并，更好地捕捉居住规模，比单独看某一层面积更直观。
- `HouseAge` 直接反映房子新旧，比单独看 `YearBuilt` 更有预测力（因为同一年建成的房子，在不同年份卖出，房龄不同）。
- `IsRemodeled` 用于区分翻修房，翻修房通常价格更高。

构造完新特征后，记得把原始特征中不再需要的可以删除，保持特征集简洁。

---

## 3.4 数值变换：对数变换与标准化，让分布更友好

有些特征的分布不太友好，比如房价严重右偏。数值变换可以让分布更接近正态，模型更容易拟合。

### 对数变换（log1p）

对于右偏分布的目标变量或特征，取对数可以压缩大值，让分布更对称。`np.log1p(x) = log(1 + x)`，之所以用 log1p 而不是 log，是为了安全处理 0 值。

```python
# 对目标变量 SalePrice 取对数（在训练集上做）
y_train = np.log1p(train['SalePrice'])
```

对数变换后，房价分布从右偏变得更接近正态。House Prices 比赛的评价指标是 RMSE，社区里常用「log 空间 RMSE」作为参考，就是基于这个考虑。

### 标准化与归一化

如果特征之间的量纲差异很大（比如 `LotFrontage` 几十到几百，`OverallQual` 1 到 10），基于距离的模型（如 KNN、SVM、神经网络）会受量纲大的特征主导。这时候需要标准化或归一化，让所有特征落在同一量级。

- **标准化（StandardScaler）**：减去均值，除以标准差，使数据均值为 0、标准差为 1。适合分布接近正态的特征。
- **归一化（MinMaxScaler）**：缩放到 [0, 1] 区间。适合不确定分布、或需要界定范围的场景。

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化
scaler = StandardScaler()
all_data[['GrLivArea', 'TotalSF']] = scaler.fit_transform(all_data[['GrLivArea', 'TotalSF']])

# 归一化
minmax_scaler = MinMaxScaler()
all_data[['OverallQual']] = minmax_scaler.fit_transform(all_data[['OverallQual']])
```

不过要注意：树模型（随机森林、XGBoost）对特征的量纲不敏感，不需要标准化；线性模型（岭回归、Lasso）和神经网络通常需要标准化。

---

## 3.5 表格数据的基线模型：线性回归与岭回归

有了清洗后的数据和构造后的特征，就可以训练第一个模型了。不要一上来就用复杂模型，先跑一个简单的基线，建立可比较的分数，后续的改进才有参照。

### 线性回归

线性回归是最简单的回归模型：假设目标变量与各特征呈线性关系。它快、可解释，是很好的基线。

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# 将合并后的数据拆回训练集和测试集
n_train = len(train_features)
X_train = all_data.iloc[:n_train]
X_test  = all_data.iloc[n_train:]

# 线性回归基线
lr = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 交叉验证，用负均方误差作为评分
scores = cross_val_score(lr, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print("线性回归 5折 RMSE (log空间):", rmse_scores.mean(), rmse_scores.std())
```

### 岭回归（Ridge）

线性回归的问题在于：如果特征之间高度相关（多重共线性），模型系数会变得很大、不稳定。岭回归在线性回归的基础上加了 **L2 正则化**，惩罚系数过大，能防范多重共线性，通常比普通线性回归更稳。

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=10.0, random_state=42)
scores = cross_val_score(ridge, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print("岭回归 5折 RMSE (log空间):", rmse_scores.mean(), rmse_scores.std())
```

`alpha` 是正则强度，alpha 越大，正则越强，系数被压得越小。可以尝试不同的 alpha 值（如 1, 10, 100），看哪个在验证集上表现更好。

### 5折交叉验证

不要只划分一次训练集/验证集，那样容易受运气影响（万一划分时验证集恰好很简单或很难）。K 折交叉验证把数据切成 K 份，每份都当一次验证集，最后取平均分，结果更可靠。上面的代码已经用了 5 折交叉验证。

---

## 3.6 表格数据的主力模型：随机森林与 XGBoost

线性回归和岭回归虽然快，但只能捕捉线性关系。表格数据比赛中，效果通常最好的主力模型是**树模型**：随机森林和 XGBoost。

### 决策树的直观原理

决策树像玩「猜是谁」游戏：不断地用特征把数据切分，每一层选一个特征和一个阈值，把数据分成两堆，直到每堆里的样本目标值差不多。比如预测房价，先切分「房屋质量是否大于 7」，再切分「地上面积是否大于 1500」，以此类推。

### 随机森林

随机森林是很多棵决策树的集合。每棵树只看部分数据和部分特征，最后取所有树的预测值的平均。这样可以减少单棵树的过拟合，结果更稳。

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_train)
print("随机森林训练集 RMSE (log空间):", np.sqrt(mean_squared_error(y_train, rf_pred)))
```

### XGBoost

XGBoost（极端梯度提升）是表格数据竞赛的「标配」。它也是集成很多棵树，但不是随机建树，而是每棵新树专门去纠正前面棵树的误差（梯度提升）。XGBoost 通常比随机森林效果更好，但调参空间也更大。

```python
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
```

XGBoost 的常用超参数：
- `n_estimators`：树的数量，越多通常越好，但容易过拟合。
- `max_depth`：每棵树的最大深度，越深越容易拟合细节，也越容易过拟合。
- `learning_rate`：学习率，每棵树的贡献缩放比例，越小越稳，但需要更多棵树。
- `subsample`：每棵树随机抽取的样本比例，0.8 表示用 80% 的样本。
- `colsample_bytree`：每棵树随机抽取的特征比例，0.8 表示用 80% 的特征。

树模型对缺失值和异常值有天然耐受性，不需要像线性模型那样严格标准化，这是它们在表格比赛中受欢迎的原因之一。

---

## 3.7 模型评估与调参：交叉验证下找不骗人的分数

训练完模型，怎么知道好不好？不能只看训练集上的分数，因为模型可能只是「背诵」了训练数据（过拟合），在新数据上表现很差。

### 为什么要用交叉验证

单次划分训练集/验证集可能运气好（验证集恰好简单）或运气差（验证集恰好难）。K 折交叉验证把数据切成 K 份，每份都当一次验证集，最后取平均分，结果更稳、更不骗人。

```python
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, X, y, cv=kf):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    return rmse_scores.mean(), rmse_scores.std()

# 比较不同模型
lr_rmse, lr_std = evaluate_model(LinearRegression(), X_train, y_train)
ridge_rmse, ridge_std = evaluate_model(Ridge(alpha=10.0), X_train, y_train)
rf_rmse, rf_std = evaluate_model(RandomForestRegressor(n_estimators=100, random_state=42), X_train, y_train)

print(f"线性回归: RMSE = {lr_rmse:.4f} (±{lr_std:.4f})")
print(f"岭回归:   RMSE = {ridge_rmse:.4f} (±{ridge_std:.4f})")
print(f"随机森林: RMSE = {rf_rmse:.4f} (±{rf_std:.4f})")
```

通过交叉验证，你可以客观比较不同模型，选出在验证集上表现更稳的那个。

### 调参的基本思路

调参不是盲目尝试，而是有策略地在候选参数里找表现更好的组合：

1. 先跑基线，知道当前分数。
2. 只改一个超参数，观察验证集分数变化。
3. 如果分数提升，保留这个改动；如果没有提升或分数下降，撤销。
4. 重复上述过程，逐步优化。

常用的调参工具有 `GridSearchCV`（穷举所有组合）和 `RandomizedSearchCV`（随机采样）。下面是一个简单的随机搜索示例：

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, 10, None],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=20,  # 尝试 20 组参数组合
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
random_search.fit(X_train, y_train)
print("最佳参数:", random_search.best_params_)
print("最佳 RMSE:", np.sqrt(-random_search.best_score_))
```

调参时要注意：验证集分数提升了，训练集分数可能也会变化。如果训练分数很高但验证分数不高，就是过拟合，应该减少模型复杂度或增加正则。

---

## 3.8 生成提交文件与上线前核对清单

训练好模型后，对测试集做预测，生成符合比赛要求的提交文件。

### 生成提交文件

```python
# 用训练好的模型对测试集预测（log空间）
test_pred_log = xgb_model.predict(X_test)

# 还原回原始价格空间
test_pred = np.expm1(test_pred_log)

# 按照 sample_submission 格式生成提交文件
submission = pd.DataFrame({
    'Id': test_id,
    'SalePrice': test_pred
})

submission.to_csv('submission.csv', index=False)
print("提交文件已生成：submission.csv")
print(submission.head())
```

注意：如果目标变量做了对数变换，预测完后必须用 `np.expm1` 还原回原始价格，否则提交的价格全是对数后的值，完全不对。

### 上线前核对清单

在点击上传之前，务必检查以下几点，避免低级错误导致提交失败：

| 检查项 | 为什么重要 |
|--------|-----------|
| 行数与测试集一致 | 多行或少行都会被平台拒绝 |
| 列名与 sample_submission 一致 | 列名错了一字母也无法评分 |
| ID 顺序与测试集一致 | 预测值要和正确的 Id 对应 |
| 无缺失值 | 预测结果里不能有 NaN |
| 数值范围合理 | 房价不可能是负数或几亿 |

```python
# 核对清单代码
assert len(submission) == len(test), "提交行数与测试集不一致"
assert list(submission.columns) == list(pd.read_csv('/kaggle/input/house-prices-regression/sample_submission.csv').columns), "列名不一致"
assert submission['SalePrice'].isnull().sum() == 0, "预测结果包含缺失值"
assert (submission['SalePrice'] > 0).all(), "房价不能是负数"
print("所有核对项通过，可以安全上传。")
```

---

## 3.9 小结

本章我们学习了表格数据的通用清洗与建模方法：

1. **缺失值处理**：空不等于坏，按语义填充。没有泳池填 `'None'`，数量为 0 填 `0`，数值型用中位数，类别型用众数，还可以按分组填充或增加 `_is_missing` 标记。
2. **类别变量编码**：有序类别（如质量等级）用有序映射，非序号类别（如街区）用独热编码。注意独热编码的基数过大问题。
3. **特征构造**：把原始列组合出新特征，如总面积 `TotalSF`、总卫浴数 `TotalBath`、房龄 `HouseAge`、是否翻修 `IsRemodeled`。
4. **数值变换**：对数变换（`log1p`）让右偏分布更接近正态；标准化让不同量纲的特征落在同一量级。树模型通常不需要标准化，线性模型和神经网络通常需要。
5. **基线模型**：先跑线性回归和岭回归建立基准分数。岭回归的 L2 正则防范多重共线性。
6. **主力模型**：随机森林和 XGBoost 是表格比赛的标配，对缺失值和异常值耐受性好，通常效果优于线性模型。
7. **评估与调参**：用 K 折交叉验证获得更可靠的分数，避免单次划分的运气成分。调参时一次只改一个超参数，以验证集分数为准。
8. **提交文件与核对**：预测后记得做逆变换（如 `expm1`），上传前检查行数、列名、ID 顺序、缺失值和数值范围。

下一章开始进入具体比赛的详解：我们将以 House Prices 房价预测为例，把本章讲的通用方法完整地应用一遍，并对比说明每一步的决策理由。
