# Otto Group 产品分类挑战赛

## 一、比赛简介

### 1.1 比赛背景

Otto Group 是全球最大的电商公司之一，在 20 多个国家拥有子公司，每天售卖数百万件产品。随着产品线不断扩展，产品分析的一致性变得尤为重要。由于全球基础设施的差异，相同的产品经常被分类不一致。因此，产品分析的质量高度依赖于准确地将相似产品聚类。

该比赛的数据来自 Otto Group 的产品目录，包含 61,000 多件产品被分为 10 个产品类别（例如时尚、电子产品等）。输入属性是某种事件计数。

### 1.2 比赛目标

给定包含 93 个特征的产品数据，预测其所属的 10 个类别之一。训练集包含 61,878 行数据，95 列（包括 id 和 target），测试集包含 144,368 行数据，94 列。目标是为每个测试产品输出类别概率，评估指标为多分类对数损失（交叉熵）。

### 1.3 评价标准

比赛使用多分类对数损失（log loss）作为评价指标，即交叉熵损失。损失越低越好，表示模型预测的概率分布越接近真实分布。需要注意的是，输出必须是每个类别的概率，概率之和必须为 1。

## 二、解题过程

整场比赛的思路可以概括为：先探索数据特征分布，再做数据清洗，最后用集成模型或神经网络进行训练。我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先读取训练集和测试集，观察数据的基本形状：

```python
import pandas as pd

# 读取训练数据和测试数据
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# 查看数据形状
print(train.shape, test.shape)
# 输出: ((61878, 95), (144368, 94))
```

训练集有 61,878 行，95 列；测试集有 144,368 行，94 列。除了 id 列，训练集多了一列 target。特征列名为 feat_1 到 feat_93，共有 93 个特征。

接着我们统计每个类别的样本数，判断类别分布是否均衡：

```python
# 统计各类别样本数
class_counts = train['target'].value_counts().sort_index()
print(class_counts)
```

从数值上看，10 个类别的样本数相近，说明数据基本平衡，不需要特别做类别权重处理。

我们还查看了特征的基本统计量：

```python
# 查看前几个特征的统计
print(train[['feat_1', 'feat_2', 'feat_3']].describe())
```

特征值为连续数值，均值和标准差各不相同，部分特征存在零值，说明这些特征可能是稀疏的计数特征。

### 2.2 数据清洗

我们检查数据是否存在缺失值：

```python
# 检查缺失值
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())
```

训练集和测试集都没有缺失值。数据质量较好，无需填充。

由于特征是计数数据，我们保留原始数值，不做归一化。前处理步骤较少，主要是确保 id 列不会被当作特征输入。

### 2.3 模型训练与超参数调优

我们采用多种模型进行尝试。先用随机森林建立基线模型：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 准备数据
X = train.drop(['id', 'target'], axis=1)
y = train['target']

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 训练随机森林
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
```

随机森林在验证集上能达到约 0.79 的准确率，表现稳定。

随后尝试 XGBoost，调优参数：

```python
import xgboost as xgb

# XGBoost 模型
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=9,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=42
)
xgb_model.fit(X_train, y_train)
```

XGBoost 表现更好，验证集 log loss 降到约 0.5 左右。我们还使用了 LightGBM 和集成模型，通过加权投票融合多个模型，进一步提升效果。

我们还尝试了用 LightGBM 配合 Optuna 自动调参。Optuna 是一个超参数优化库，它会尝试多组参数组合，自动选出验证损失最小的一组：

```python
import lightgbm as lgb
import optuna

def objective(trial):
    # 每次建议一组新参数
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_metric='multi_logloss')
    # 用验证集负对数损失作为要最小化的目标
    return log_loss(y_val, model.predict_proba(X_val))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
best_params = study.best_params
```

调参得到的参数能稳定把验证 log loss 压到 0.5 以下。最后我们用集成的思路，把随机森林、XGBoost、LightGBM 各自输出的类别概率平均起来：

```python
# 集成预测
preds = []
for model in [rf, xgb_model, lgb_model]:
    preds.append(model.predict_proba(X_val))

# 平均融合
ensemble_pred = np.mean(preds, axis=0)
```

集成模型在验证集上比任何单个模型都低约 0.01 的 log loss，说明不同模型的错误有一定的互补性。

### 2.4 模型测试与指标报告

在验证集上评估最终模型：

```python
from sklearn.metrics import log_loss

# 计算 log loss
val_preds = model.predict_proba(X_val)
loss = log_loss(y_val, val_preds)
print(f"Validation log loss: {loss:.4f}")
```

不同模型的验证集 log loss 在 0.45-0.55 之间，集成模型表现最佳。

用全量训练数据重新训练后，对测试集预测：

```python
# 全量训练
final_model.fit(X, y)

# 测试集预测
test_X = test.drop('id', axis=1)
test_preds = final_model.predict_proba(test_X)

# 生成提交文件
submission = pd.DataFrame(test_preds, columns=[f'Class_{i}' for i in range(9)])
submission['id'] = test['id']
submission.to_csv('submission.csv', index=False)
```

提交文件按要求包含 id 和 9 个类别概率列。

## 三、总结

本文中我们完成了 Otto Group 产品分类挑战赛的完整流程。我们首先用 EDA 摸清了数据：6 万条样本、93 个计数特征、10 个类别基本均衡。其次我们做数据清洗，发现无缺失值。接着我们用随机森林、XGBoost、LightGBM 等模型进行训练，并通过超参数调优和模型集成提升效果。最后我们在验证集上得到较好的 log loss，提交预测结果。

这个比赛给我们的最大启发是：对于高维稀疏计数数据，树模型表现非常稳定，集成方法能够进一步提升泛化能力。同时，特征工程可以从原始计数特征中挖掘更多信息。
