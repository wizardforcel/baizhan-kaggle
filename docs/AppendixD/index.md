# 附录D scikit-learn快速参考

scikit-learn（通常简称 sklearn）是 Python 里最经典的机器学习库。它集成了大量常用算法（线性回归、随机森林、梯度提升、SVM、KNN 等等），也自带数据预处理、模型评估、超参数调优等工具。Kaggle 上的很多表格数据比赛，baseline 代码基本都是 sklearn 写的。

本附录面向零基础初学者，用大白话讲清楚 sklearn 最核心的使用模式：**fit → predict → score**，以及数据预处理、特征选择、划分训练集/测试集、交叉验证、管道与评估等常用操作。

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

## D.1 数据预处理

机器学习模型对数据有要求：不能有缺失值、特征要全是数字、不同特征的量级最好差不多。sklearn 提供了常用的预处理工具，都在 `sklearn.preprocessing` 模块里。

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

| strategy | 说明 |
|----------|------|
| `"mean"` | 用平均值填充（适合数值型） |
| `"median"` | 用中位数填充（对异常值不敏感） |
| `"most_frequent"` | 用众数填充（适合类别型） |
| `"constant"` | 用固定值填充，配合 `fill_value` 参数 |

### 2. 数值缩放：`StandardScaler` / `MinMaxScaler`

有些模型（如 KNN、SVM、神经网络、线性回归）对特征的量级很敏感。如果一个特征是 0~1，另一个是 0~100000，后者会主导计算结果。需要把所有特征缩放到差不多的范围。

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化：均值为 0，标准差为 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 归一化：缩放到 [0, 1] 区间
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 3. 类别特征编码：`OneHotEncoder` / `OrdinalEncoder` / `LabelEncoder`

模型只能处理数字，类别特征（如“城市”、“颜色”）要先编码成数字。

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

# 独热编码：把一个类别列展开成多个 0/1 列（适合无序类别，如城市）
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(df[["城市"]])

# 有序编码：把有顺序的类别映射为数字（如 质量等级：差 < 中 < 好）
encoder = OrdinalEncoder()
encoded = encoder.fit_transform(df[["质量等级"]])

# 标签编码：把每个类别映射为一个整数（只适合目标变量 y，不适合特征 X）
le = LabelEncoder()
y = le.fit_transform(y)
```

> 💡 **小窍门**：在 pandas 里也可以用 `pd.get_dummies(df, columns=["城市"])` 做独热编码，更简单方便，Kaggle Notebook 里很常用。

其他预处理类：

| 类 | 说明 |
|----|------|
| `RobustScaler` | 用中位数和四分位距缩放，对异常值不敏感 |
| `MaxAbsScaler` | 缩放到 [-1, 1]，适合已经居中于 0 的数据 |
| `Normalizer` | 对每个样本（行）归一化到单位向量 |
| `Binarizer` | 按阈值二值化，大于阈值为 1 否则为 0 |
| `PolynomialFeatures` | 生成多项式特征和交互特征 |
| `KBinsDiscretizer` | 把连续特征分桶成离散区间 |
| `FunctionTransformer` | 用自定义函数做变换 |
| `PowerTransformer` | 做 Yeo-Johnson 或 Box-Cox 变换，让分布更接近正态 |
| `QuantileTransformer` | 按分位数把特征变换成均匀分布或正态分布 |

---

## D.2 特征选择

从很多特征里挑出最有用的几个，减少噪声和过拟合。都在 `sklearn.feature_selection` 模块里。

```python
from sklearn.feature_selection import SelectKBest, f_classif

# 选 K 个最好的特征（基于方差分析 F 值）
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)

# 查看被选中的特征下标
print(selector.get_support(indices=True))
```

常用的特征选择类：

| 类 | 说明 |
|----|------|
| `SelectKBest` | 选 K 个评分最高的特征 |
| `SelectPercentile` | 选评分最高的百分比特征 |
| `RFE` | 递归特征消除，每次删掉最不重要的特征 |
| `RFECV` | 带交叉验证的递归特征消除，自动选最佳特征数 |
| `VarianceThreshold` | 删除方差低于阈值的特征（如常量列） |
| `SelectFromModel` | 基于模型重要性选特征（如随机森林的 `feature_importances_`） |

```python
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# 删除方差为 0 的常量列
selector = VarianceThreshold(threshold=0)
X_new = selector.fit_transform(X)

# 基于模型重要性选特征
rf = RandomForestClassifier()
selector = SelectFromModel(rf, threshold="median")
X_new = selector.fit_transform(X, y)
```

---

## D.3 模型选择与交叉验证

把数据分成训练和验证，交叉验证更稳，网格搜索帮你找最好的超参数。都在 `sklearn.model_selection` 模块里。

### 1. 划分训练集/测试集

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- `test_size`：测试集占比，如 0.2 表示 20%。
- `random_state`：随机种子，设固定值保证每次划分结果一样，方便复现。
- `stratify=y`：分层抽样，保持各类别比例一致（分类任务常用）。

### 2. 交叉验证

单次划分训练集/测试集的结果可能有运气成分。**交叉验证**把数据分成好几份，轮流当测试集，最后取平均分，结果更稳可靠。

```python
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate

# K 折交叉验证
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
print("每一折得分:", scores)
print("平均得分:", scores.mean())

# 分层 K 折（保持类别比例，分类任务常用）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 时间序列分割（时序数据，不能随机打乱）
cv = TimeSeriesSplit(n_splits=5)

# 组 K 折（同一组的样本不跨折，如同一用户的多条记录）
cv = GroupKFold(n_splits=5)
```

常用的交叉验证类：

| 类 | 说明 |
|----|------|
| `KFold` | 随机 K 折，适合回归或平衡分类 |
| `StratifiedKFold` | 分层 K 折，保持类别比例，分类任务常用 |
| `TimeSeriesSplit` | 时序分割，训练在前验证在后，防止未来信息泄露 |
| `GroupKFold` | 组 K 折，同一组不跨折 |

`cross_val_score` 常用的 `scoring`：

| 任务 | scoring | 说明 |
|------|---------|------|
| 回归 | `"neg_root_mean_squared_error"` | 负 RMSE（越接近 0 越好） |
| 回归 | `"neg_mean_absolute_error"` | 负 MAE |
| 回归 | `"r2"` | R² 分数（越接近 1 越好） |
| 分类 | `"accuracy"` | 准确率 |
| 分类 | `"f1"` | F1 分数 |
| 分类 | `"roc_auc"` | AUC-ROC（二分类） |

### 3. 超参数调优：`GridSearchCV` / `RandomizedSearchCV`

每个模型都有一些“超参数”，需要你手动指定。网格搜索会把你指定的参数组合全部跑一遍，找出效果最好的一组。

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

# 网格搜索：穷举所有组合
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

print("最佳参数:", grid_search.best_params_)
print("最佳得分:", grid_search.best_score_)

best_model = grid_search.best_estimator_
```

| 类 | 说明 |
|----|------|
| `GridSearchCV` | 穷举所有参数组合，计算代价大但 exhaustive |
| `RandomizedSearchCV` | 随机采样参数组合，计算代价小，适合参数空间大的情况 |
| `learning_curve` | 绘制学习曲线，判断过拟合或欠拟合 |
| `validation_curve` | 绘制验证曲线，看某个参数对分数的影响 |

---

## D.4 回归模型

预测连续数值（房价、销量、温度）的模型。都在 `sklearn.linear_model`、`sklearn.tree`、`sklearn.ensemble` 等模块里。

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# 线性回归
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 岭回归（L2 正则，防止多重共线性）
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 随机森林回归
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

常用的回归类：

| 类 | 说明 |
|----|------|
| `LinearRegression` | 线性回归，最简单的基线 |
| `Ridge` | 岭回归，L2 正则化 |
| `Lasso` | 套索回归，L1 正则化，可做特征选择 |
| `ElasticNet` | 弹性网，L1+L2 正则化 |
| `SGDRegressor` | 基于随机梯度下降的线性回归，适合大数据 |
| `DecisionTreeRegressor` | 决策树回归 |
| `RandomForestRegressor` | 随机森林回归，多棵树平均 |
| `GradientBoostingRegressor` | 梯度提升回归，依次纠错 |
| `SVR` | 支持向量回归 |
| `KNeighborsRegressor` | K 近邻回归 |

回归模型都有 `.score(X, y)` 方法，默认返回 **R² 分数**（越接近 1 越好）。

---

## D.5 分类模型

预测类别（是否点击、是否违约、猫/狗）的模型。

```python
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# 逻辑回归（二分类基线）
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # 预测概率

# 随机森林分类
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

常用的分类类：

| 类 | 说明 |
|----|------|
| `LogisticRegression` | 逻辑回归，分类基线，可输出概率 |
| `SGDClassifier` | 基于随机梯度下降的线性分类器，适合大数据 |
| `DecisionTreeClassifier` | 决策树分类 |
| `RandomForestClassifier` | 随机森林分类，多棵树投票 |
| `GradientBoostingClassifier` | 梯度提升分类，依次纠错 |
| `SVC` | 支持向量分类（核技巧，适合小数据） |
| `LinearSVC` | 线性支持向量分类，适合大数据 |
| `KNeighborsClassifier` | K 近邻分类 |
| `GaussianNB` | 高斯朴素贝叶斯，适合连续特征 |
| `MultinomialNB` | 多项式朴素贝叶斯，适合离散计数特征（如文本） |
| `BernoulliNB` | 伯努利朴素贝叶斯，适合 0/1 特征 |

分类模型都有 `.score(X, y)` 方法，默认返回 **准确率**（accuracy）。

---

## D.6 集成模型

多个模型合在一起投票或平均，通常比单个模型更稳、分数更高。都在 `sklearn.ensemble` 模块里。

```python
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

# 软投票分类：多个模型预测概率取平均
vc = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb), ('lgb', lgb)],
    voting='soft'
)
vc.fit(X_train, y_train)

# Stacking：第一层模型预测结果作为第二层模型的输入
stack = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    final_estimator=LogisticRegression()
)
stack.fit(X_train, y_train)
```

常用的集成类：

| 类 | 说明 |
|----|------|
| `VotingClassifier` / `VotingRegressor` | 投票/平均集成，硬投票（多数票）或软投票（概率平均） |
| `StackingClassifier` / `StackingRegressor` | 堆叠集成，第一层预测喂给第二层元学习器 |
| `BaggingClassifier` / `BaggingRegressor` | 装袋集成，自助采样+独立训练+平均/投票 |
| `AdaBoostClassifier` / `AdaBoostRegressor` | 自适应提升，依次关注被错分的样本 |

---

## D.7 聚类与降维

聚类把相似的样本分到一组，降维把高维数据压缩到低维便于可视化。

```python
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, NMF
from sklearn.manifold import TSNE

# K-Means 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# PCA 降维（主成分分析）
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# t-SNE 降维（适合可视化）
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X)
```

常用的聚类类：

| 类 | 说明 |
|----|------|
| `KMeans` | K 均值聚类，指定 K 个簇 |
| `AgglomerativeClustering` | 层次聚类，自底向上合并 |
| `DBSCAN` | 基于密度的聚类，自动确定簇数，能识别噪声 |
| `GaussianMixture` | 高斯混合模型，软聚类（每个样本属于各簇的概率） |

常用的降维类：

| 类 | 说明 |
|----|------|
| `PCA` | 主成分分析，线性降维，保留方差最大的方向 |
| `KernelPCA` | 带核技巧的 PCA，非线性降维 |
| `TruncatedSVD` | 截断奇异值分解，适合稀疏矩阵（如文本 TF-IDF） |
| `NMF` | 非负矩阵分解，适合非负数据 |
| `TSNE` | t-分布随机邻嵌入，适合二维/三维可视化 |

---

## D.8 管道与评估

Pipeline 把预处理和模型串起来，避免数据泄露，代码更简洁。评估函数算出模型好不好。

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

# Pipeline：标准化 + 随机森林
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipe.fit(X_train, y_train)
print(accuracy_score(y_test, pipe.predict(X_test)))

# 或者用 make_pipeline 更简洁
pipe = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
```

### 回归评估指标

```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))    # 均方根误差，越小越好
mae = mean_absolute_error(y_test, y_pred)             # 平均绝对误差，越小越好
r2 = r2_score(y_test, y_pred)                         # R² 分数，越接近 1 越好
```

### 分类评估指标

```python
acc = accuracy_score(y_test, y_pred)                  # 准确率
f1 = f1_score(y_test, y_pred)                         # F1 分数，类别不平衡时比准确率更可靠
auc = roc_auc_score(y_test, y_proba)                  # AUC-ROC，二分类，越接近 1 越好

cm = confusion_matrix(y_test, y_pred)                 # 混淆矩阵
print(classification_report(y_test, y_pred))          # 分类报告：精确率、召回率、F1、支持数

# ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
```

常用的评估类与函数：

| 类/函数 | 说明 |
|---------|------|
| `Pipeline` / `make_pipeline` | 把预处理和模型串起来，避免泄露 |
| `FeatureUnion` | 把多个特征提取器的结果拼在一起 |
| `make_column_transformer` | 对不同列应用不同的预处理 |
| `accuracy_score` | 准确率 |
| `precision_score` | 精确率 |
| `recall_score` | 召回率 |
| `f1_score` | F1 分数 |
| `roc_auc_score` | AUC-ROC |
| `mean_squared_error` | 均方误差 |
| `mean_absolute_error` | 平均绝对误差 |
| `r2_score` | R² 分数 |
| `confusion_matrix` | 混淆矩阵 |
| `classification_report` | 分类报告 |
| `roc_curve` | ROC 曲线 |
| `precision_recall_curve` | 精确率-召回率曲线 |

---

## D.9 小结

本附录快速过了一遍 scikit-learn 最常用的操作：

1. **核心模式**：`fit()` 训练 → `predict()` 预测 → `score()` 评估。
2. **数据预处理**：`SimpleImputer`（缺失值）、`StandardScaler`/`MinMaxScaler`（缩放）、`OneHotEncoder`/`OrdinalEncoder`/`LabelEncoder`（类别编码）。
3. **特征选择**：`SelectKBest`、`RFE`/`RFECV`、`VarianceThreshold`、`SelectFromModel`。
4. **划分数据**：`train_test_split(X, y, test_size=0.2, random_state=42)`。
5. **交叉验证**：`cross_val_score`，`KFold`/`StratifiedKFold`/`TimeSeriesSplit`/`GroupKFold`。
6. **超参数调优**：`GridSearchCV`（穷举）或 `RandomizedSearchCV`（随机采样）。
7. **回归模型**：`LinearRegression`、`Ridge`、`Lasso`、`RandomForestRegressor`、`GradientBoostingRegressor`、`SVR`、`KNeighborsRegressor`，评估 RMSE/MAE/R²。
8. **分类模型**：`LogisticRegression`、`RandomForestClassifier`、`GradientBoostingClassifier`、`SVC`、`KNeighborsClassifier`、`GaussianNB` 等，评估 准确率/F1/AUC。
9. **集成模型**：`VotingClassifier/Regressor`、`StackingClassifier/Regressor`、`BaggingClassifier/Regressor`、`AdaBoostClassifier/Regressor`。
10. **聚类与降维**：`KMeans`、`DBSCAN`、`GaussianMixture`、`PCA`、`TSNE`、`TruncatedSVD`。
11. **管道与评估**：`Pipeline` 串联预处理和模型；`accuracy_score`/`f1_score`/`roc_auc_score`/`mean_squared_error`/`confusion_matrix`/`classification_report` 等评估函数。

> ⚠️ **关键原则**：`fit_transform` 只在训练集，`transform` 只在测试集，防止数据泄露。

掌握这些，你就可以写出一个完整的 Kaggle 表格数据 baseline 了：读数据 → 预处理 → 划分数据 → 训练模型 → 交叉验证 → 调参 → 预测 → 生成提交文件。

---

> **下一步建议**：scikit-learn 能解决绝大多数表格数据比赛。如果你想尝试更强的模型，可以继续学习 XGBoost、LightGBM、CatBoost 等梯度提升树库，它们是 Kaggle 表格比赛的“标配”神器；如果要做图像、文本等深度学习比赛，建议继续阅读附录 E（PyTorch 快速参考）。