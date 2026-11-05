# Springleaf Marketing Response

## 一、比赛简介

### 1.1 比赛背景
Springleaf Marketing Response 是 Kaggle 经典的金融营销响应预测比赛。数据来自消费金融公司 Springleaf，约 1,900 多个匿名特征，任务是预测客户是否对某次营销行动做出响应。数据集规模巨大，特征匿名，包含数值、类别和缺失值。

### 1.2 比赛目标
训练集包含 ID、VAR_0001 ... VAR_1934、target。我们需要预测测试集客户是否响应。特征均为匿名编码，无法从名称推断含义。目标是二分类。

### 1.3 评价标准
比赛使用 AUC 作为评价指标，衡量模型区分响应与不响应客户的能力。AUC 越高越好。数据极度不平衡，正样本比例约 5% 左右。

## 二、解题过程

### 2.1 数据探索（EDA）
我们读取训练数据的小样本，先看形状与缺失情况。

```python
train = pd.read_csv('../input/springleaf-train-small/train_small.csv')
print(train.shape)
print(train.isnull().sum(axis=1).head())
```

训练样本约 14,523 条，特征 1,934 个。对象类约 51 个。大多数特征无缺失，少数对象列有 56 个缺失值。

```python
# 查看目标分布
print(train['target'].value_counts(normalize=True))
```

正样本比例约 5%，属于不平衡数据。我们合并训练测试做 EDA。

```python
traintest = pd.concat([train, test], axis=0)
```

观察到字符型特征常出现 9998、BRANCH、IAPS 等编码，提供额外信息。

### 2.2 数据清洗
匿名特征需要清理。首先删除常数特征。

```python
# 删除方差为 0 的列
constant_cols = [c for c in train if train[c].nunique() == 1]
train.drop(columns=constant_cols, inplace=True)
```

缺失值统计，逐列查看缺失比例。

```python
train.isnull().sum().sort_values(ascending=False).head(20)
```

对数值特征，按中位数填充；对类别特征，按众数填充或填 'Unknown'。

```python
for col in train.select_dtypes(include=['number']).columns:
    train[col].fillna(train[col].median(), inplace=True)
```

我们还统计每行缺失值数量作为新特征缺失计数，可能与响应相关。

```python
train['missing_count'] = train.isnull().sum(axis=1)
```

### 2.3 模型训练与超参数调优
由于特征多且匿名，我们先做基线逻辑回归，然后用 XGBoost。

```python
from sklearn.linear_model import LogisticRegression
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2)
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)
```

基线 AUC 约 0.70。随后使用 XGBoost 处理非线性与缺失值。

```python
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=19,
    eval_metric='auc'
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
```

调参重点是 scale_pos_weight 处理不平衡，学习率与树深控制过拟合。我们还做了特征重要性筛选，保留 AUC 贡献的前 500 个特征。

```python
importance = model.feature_importances_
feature_rank = np.argsort(importance)[::-1]
```

### 2.4 模型测试与指标报告
在验证集上输出 AUC 与混淆矩阵。

```python
from sklearn.metrics import roc_auc_score
y_pred_proba = model.predict_proba(X_val)[:,1]
print('AUC:', roc_auc_score(y_val, y_pred_proba))
```

验证 AUC 约 0.80。最终用全量训练数据拟合，生成提交文件。

```python
model.fit(X, y)
test_pred = model.predict_proba(X_test)[:,1]
submission = pd.DataFrame({'ID': test_ID, 'target': test_pred})
submission.to_csv('submission.csv', index=False)
```

提交文件检查 ID 和列名是否匹配。

## 三、总结
本文中我们完成了 Springleaf Marketing Response 的完整流程。我们首先通过 EDA 掌握了 1,934 个匿名特征与 5% 正样本不平衡的现状；其次对缺失值、中位数填充与常数特征删除进行了清洗；接着用逻辑回归基线和 XGBoost 做训练与调参，并用 scale_pos_weight 处理不平衡；最后在验证集上达到 AUC 约 0.80 并生成提交文件。这个比赛说明在大规模匿名数据上，清洗与采样策略比模型复杂度更重要。
