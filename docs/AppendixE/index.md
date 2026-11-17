# 附录E scikit-learn 机器学习快速入门

scikit-learn（通常简称 sklearn）是 Python 里最经典的机器学习库。它集成了大量常用算法（线性回归、随机森林、梯度提升、SVM、KNN 等等），也自带数据预处理、模型评估、超参数调优等工具。Kaggle 上的很多表格数据比赛，baseline 代码基本都是 sklearn 写的。

本附录面向零基础初学者，用大白话讲清楚 sklearn 最核心的使用模式：** fit → predict → score**，以及数据预处理、划分训练集/测试集、交叉验证等常用操作。

> **Mermaid 图表：scikit-learn 典型使用流程**

```mermaid
flowchart LR
    A[读取数据] --> B[划分训练集/测试集]
    B --> C[特征预处理 缺失值/编码/缩放]
    C --> D[创建模型并 fit 训练]
    D --> E[在测试集上 predict 预测]
    E --> F[用评估指标 score 打分]
    F --> G[提交预测结果]
    style D fill:#bbf,stroke:#333
```

---

## E.1 安装与导入

```bash
pip install scikit-learn
```

Kaggle Notebooks 里 sklearn 已经预装，直接导入：

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score
```

---

## E.2 sklearn 的核心使用模式

sklearn 所有模型都遵循同一套模式，学会了一个模型，其他模型也都会用了：

1. **创建模型对象**：`model = 某个模型()`
2. **训练（拟合）**：`model.fit(X_train, y_train)` —— 让模型从训练数据里学规律
3. **预测**：`model.predict(X_test)` —— 对新数据做预测
4. **评估**：把预测值和真实值比较，看准不准

下面用一个最简单的例子说明：**波士顿房价预测**（回归问题）。

```python
from sklearn.datasets import load_boston   # sklearn 自带的示例数据集
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. 加载数据
boston = load_boston()
X = boston.data          # 特征：房屋的各种属性（犯罪率、房间数等）
y = boston.target        # 目标：房价

# 2. 划分训练集和测试集（80% 训练，20% 测试）
# random_state 设固定值，保证每次划分结果一样，方便复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 创建模型并训练
model = LinearRegression()   # 线性回归
model.fit(X_train, y_train)  # 训练：让模型学习 X 和 y 的关系

# 4. 预测
y_pred = model.predict(X_test)

# 5. 评估：计算均方根误差（RMSE）
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("测试集 RMSE:", rmse)

# 6. 看模型在训练集和测试集上的得分（R^2 分数，越接近1越好）
print("训练集得分:", model.score(X_train, y_train))
print("测试集得分:", model.score(X_test, y_test))
```

> 💡 **大白话**：`fit` 就是“学习”，`predict` 就是“应用所学去预测新数据”。`score` 返回的是模型自带的默认评估指标，回归模型默认是 R² 分数（越接近1越好），分类模型默认是准确率（accuracy）。

---

## E.3 回归 vs 分类

sklearn 把常见的机器学习问题分为两大类：

| 问题类型 | 目标变量 y 是什么 | 常用模型 | 常用评估指标 |
|----------|------------------|----------|--------------|
| **回归（Regression）** | 连续数值（如房价、气温、销售额） | 线性回归、随机森林回归、梯度提升回归 | RMSE、MAE、R² |
| **分类（Classification）** | 离散类别（如是否点击、猫/狗、信用等级） | 逻辑回归、随机森林分类、SVM、KNN | 准确率、F1、AUC-ROC |

### 回归模型示例：随机森林回归

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("随机森林回归 RMSE:", rmse)
```

`n_estimators=100` 表示构建100棵决策树，棵数越多通常越准，但训练越慢。

### 分类模型示例：随机森林分类

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 假设 y 是 0/1 的类别标签
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("测试集准确率:", acc)
```

---

## E.4 数据预处理

机器学习模型对数据有要求：不能有缺失值、特征要全是数字、不同特征的量级最好差不多。sklearn 提供了常用的预处理工具。

### 1. 缺失值填充：`SimpleImputer`

```python
from sklearn.impute import SimpleImputer

# 用每列的平均值填充缺失值
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

> ⚠️ **关键点**：`fit_transform` 只用在训练集上（学习填充值，比如算出每列的平均值）；`transform` 用在测试集上（用训练集学到的平均值去填充）。绝对不能在测试集上 `fit`，否则测试集的信息会“泄露”到训练过程里，导致成绩虚高。

常用的 `strategy`：
- `"mean"`：用平均值填充（适合数值型）
- `"median"`：用中位数填充（对异常值不敏感）
- `"most_frequent"`：用众数填充（适合类别型）
- `"constant"`：用固定值填充，配合 `fill_value` 参数

### 2. 类别特征编码：`LabelEncoder` / `OneHotEncoder`

模型只能处理数字，类别特征（如“城市”、“颜色”）要先编码成数字。

```python
from sklearn.preprocessing import LabelEncoder

# 标签编码：把“北京/上海/广州”变成 0/1/2
le = LabelEncoder()
df["城市"] = le.fit_transform(df["城市"])
```

如果类别之间没有大小顺序（比如城市），更常用 **独热编码（One-Hot Encoding）**：把一个类别列展开成多个 0/1 列。

```python
import pandas as pd

# pandas 的 get_dummies 最方便
df = pd.get_dummies(df, columns=["城市"], drop_first=True)
```

`drop_first=True` 表示删掉第一个哑变量，避免“虚拟变量陷阱”（多重共线性）。

### 3. 特征缩放：`StandardScaler` / `MinMaxScaler`

有些模型（如 KNN、SVM、神经网络、线性回归）对特征的量级很敏感。如果一个特征是 0~1，另一个是 0~100000，后者会主导计算结果。需要把所有特征缩放到差不多的范围。

```python
from sklearn.preprocessing import StandardScaler

# 标准化：均值为0，标准差为1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

`StandardScaler` 把每列变成“均值0、标准差1”的分布；`MinMaxScaler` 则把每列缩放到 `[0, 1]` 区间。

> ⚠️ 同样要记住：`fit_transform` 只在训练集，`transform` 只在测试集。

---

## E.5 交叉验证：`cross_val_score`

单次划分训练集/测试集的结果可能有运气成分（比如刚好测试集里都是简单样本）。**交叉验证**把数据分成好几份，轮流当测试集，最后取平均分，结果更稳可靠。

```python
from sklearn.model_selection import cross_val_score

model = RandomForestRegressor(n_estimators=100, random_state=42)

# 5折交叉验证，返回每一折的得分
scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")

print("每一折的得分:", scores)
print("平均得分:", scores.mean())
```

注意：回归问题的 `scoring="neg_root_mean_squared_error"` 返回的是**负的 RMSE**（sklearn 约定得分越大越好，所以把 RMSE 取负）。实际 RMSE 是 `-scores.mean()`。

交叉验证常用的 `cv` 值是 5 或 10，表示分成 5 份或 10 份。

---

## E.6 模型评估指标

sklearn 的 `sklearn.metrics` 模块提供了各种评估指标。

### 回归常用指标

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# RMSE：均方根误差，越小越好，和目标变量单位一致，好理解
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# MAE：平均绝对误差，越小越好，对异常值不敏感
mae = mean_absolute_error(y_test, y_pred)

# R²：决定系数，越接近1越好，可解释为“模型解释了多少比例的方差"
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
```

### 分类常用指标

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# 准确率：预测对的占总样本的比例
acc = accuracy_score(y_test, y_pred)

# F1分数：精确率和召回率的调和平均，类别不平衡时比准确率更可靠
f1 = f1_score(y_test, y_pred)

# AUC-ROC：衡量模型区分正负样本的能力，越接近1越好（只适合二分类）
# 需要传入预测的概率，而不是预测的类别
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

# 混淆矩阵：直观看出预测对了多少、错在了哪里
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f"准确率: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
```

---

## E.7 超参数调参：`GridSearchCV`

每个模型都有一些“超参数”，比如随机森林的树的数量 `n_estimators`、最大深度 `max_depth` 等。这些参数不是模型自动学出来的，需要你手动指定，而且不同数据集最佳值不同。

**网格搜索（Grid Search）**会把你指定的参数组合全部跑一遍，找出效果最好的一组。

```python
from sklearn.model_selection import GridSearchCV

model = RandomForestRegressor(random_state=42)

# 定义要尝试的参数组合
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

# 5折交叉验证 + 网格搜索
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_root_mean_squared_error")
grid_search.fit(X_train, y_train)

# 看最好的参数和最好的得分
print("最佳参数:", grid_search.best_params_)
print("最佳得分:", -grid_search.best_score_)

# 用最佳参数的模型做预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

> ⚠️ 网格搜索会训练很多次模型（参数组合数 × cv 折数），数据大时会很慢。可以先用粗参数范围快速试，再在好的区域附近细调。

---

## E.8 保存和加载模型

训练好的模型可以保存下来，以后直接加载使用，不用重新训练。

```python
import joblib

# 保存模型
joblib.dump(model, "model.pkl")

# 加载模型
model = joblib.load("model.pkl")

# 直接预测
y_pred = model.predict(X_test)
```

`joblib` 是 sklearn 官方推荐的序列化工具，比 pickle 更适合大模型。

---

## E.9 小结

本附录快速过了一遍 scikit-learn 最常用的操作：

1. **核心模式**：`fit()` 训练 → `predict()` 预测 → `score()` 评估。
2. **划分数据**：`train_test_split(X, y, test_size=0.2, random_state=42)`。
3. **回归模型**：`LinearRegression`、`RandomForestRegressor` 等，评估指标 RMSE/MAE/R²。
4. **分类模型**：`LogisticRegression`、`RandomForestClassifier` 等，评估指标 准确率/F1/AUC。
5. **缺失值处理**：`SimpleImputer(strategy="mean/median/most_frequent")`。
6. **类别编码**：`LabelEncoder` 或 pandas `get_dummies()`。
7. **特征缩放**：`StandardScaler()` 或 `MinMaxScaler()`。
8. **关键原则**：`fit_transform` 只在训练集，`transform` 只在测试集，防止数据泄露。
9. **交叉验证**：`cross_val_score(model, X, y, cv=5)`，结果更稳。
10. **超参数调优**：`GridSearchCV(param_grid, cv=5)` 暴力搜索最佳参数。
11. **保存模型**：`joblib.dump/load()`。

掌握这些，你就可以写出一个完整的 Kaggle 表格数据 baseline 了：读数据 → 预处理 → 划分数据 → 训练模型 → 交叉验证 → 调参 → 预测 → 生成提交文件。

---

> **下一步建议**：sklearn 能解决绝大多数表格数据比赛。如果你想尝试更强的模型，可以继续学习 XGBoost、LightGBM、CatBoost 等梯度提升树库，它们是 Kaggle 表格比赛的“标配”神器。建议继续阅读正文第21章《模型调参与集成学习》。