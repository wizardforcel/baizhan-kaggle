# Housing Price Prediction 2026

## 一、比赛简介

### 1.1 比赛背景
比赛使用经典 Boston Housing 数据集的变体，预测房屋中位数房价 MEDV。特征为 13 个数值型变量，包含犯罪率 CRIM、区位 ZN、工业用地 INDUS、房龄 AGE、房间数 RM 等。数据覆盖波士顿多个分区的房屋，存在异常值与非正态分布。

### 1.2 比赛目标
给定 13 个特征，预测 MEDV，即房屋中位数自住单价（单位 $1000）。训练集与测试集分别提供，目标为回归。

### 1.3 评价标准
使用均方误差 MSE 评估，数值越低越好。模型需对异常值与尺度差异敏感。

---
## 二、解题过程

我们采用清洗、稳健缩放、异常值处理、线性/梯度提升组合的方式。

### 2.1 数据探索（EDA）

```python
train = pd.read_csv('/kaggle/input/competitions/housing-price-prediction-2026/train.csv')
print(train.shape)              # 约 506, 14
train.describe()
```

CRIM、LSTAT 等呈现右偏，RM 与 MEDV 正相关，LSTAT 与 MEDV 负相关。箱线图显示明显异常值。

```python
columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
for col in columns:
    sns.boxplot(data=train, x=col)
```

### 2.2 数据清洗

先用 RobustScaler 处理尺度差异。

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
```

异常值用 IQR 方法置为 NaN 后再用模型填补。

```python
for col in X_train.columns:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    X_train[col] = np.where((X_train[col] < lower) | (X_train[col] > upper), np.nan, X_train[col])
```

缺失值用基于其他特征的 HistGradientBoostingRegressor 填补。

```python
def fill_missing_values(df, target_col, feature_col):
    clf = HistGradientBoostingRegressor(random_state=42)
    clf.fit(train_data[feature_col], train_data[target_col])
    df.loc[df[target_col].isnull(), target_col] = clf.predict(test_data[feature_col])
```

### 2.3 模型训练与超参数调优

基线使用 LinearRegression。

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_score = mean_squared_error(y_test, lr.predict(x_test))
print(f"Linear Regression MSE:{lr_score}")
```

随后尝试 HistGradientBoostingRegressor，自动处理非线性关系。

```python
from sklearn.ensemble import HistGradientBoostingRegressor
hgb = HistGradientBoostingRegressor(random_state=42)
hgb.fit(X_train, y_train)
```

### 2.4 模型测试与指标报告

LinearRegression 在清洗后 MSE 得到下降。箱线图显示清洗后特征分布更紧凑，模型预测稳定性提升。

```python
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

最终预测导出为 CSV，包含 Id 与 MEDV。

```python
submission = pd.DataFrame({'Id': test.index, 'MEDV': preds})
submission.to_csv('submission.csv', index=False)
```

---
## 三、总结

本文中我们完成了 Housing Price Prediction 2026 的回归建模。我们首先通过 EDA 发现 CRIM、LSTAT 等特征右偏并存在异常值。其次用 RobustScaler 稳健缩放并以 IQR 标记异常，再用梯度提升模型填补缺失。接着用 LinearRegression 作基线并尝试 HistGradientBoosting 提升非线性拟合。最后输出 MSE 评估结果并生成提交文件。

该比赛说明在小样本回归中，数据清洗与尺度处理比模型复杂度更关键。
