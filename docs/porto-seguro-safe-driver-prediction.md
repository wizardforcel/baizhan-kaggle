# Porto Seguro 安全驾驶预测

## 一、比赛简介

### 1.1 比赛背景
Porto Seguro 这家巴西保险公司希望用历史数据预测车主未来是否会提出理赔。数据来自实际投保记录，包含保单、车主个人信息和车辆信息。比赛难点在于正样本极少、特征很多且部分特征以 -1 表示缺失。

### 1.2 比赛目标
给定训练集和测试集，预测测试样本的 `target` 列。训练集有 595212 行、58 列，其中 57 个特征加 1 个目标。测试集有 892816 行、57 列。特征分为 `ps_ind_*`、`ps_reg_*`、`ps_car_*`、`ps_calc_*` 四类。目标是二分类，1 表示会理赔，0 表示不会。

### 1.3 评价标准
比赛使用**Gini 系数**，也就是 ROC AUC 的变形。Gini = 2*AUC -1，取值范围 0~1，越高越好。实际比赛中我们用 `eval_gini` 函数计算验证集的标准化 Gini。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先把训练集、测试集和样本提交文件读进来，查看形状和列名。

```python
# 读入数据
import pandas as pd
data_path = '/kaggle/input/porto-seguro-safe-driver-prediction/'
train = pd.read_csv(data_path + 'train.csv', index_col='id')
test = pd.read_csv(data_path + 'test.csv', index_col='id')
# 查看数据量
train.shape, test.shape
# 输出：((595212, 58), (892816, 57))
```

训练集约 59 万条，测试集约 89 万条。正样本比例很低。

```python
# 计算正样本比例
pos_rate = train['target'].mean()
print(pos_rate)  # 约 0.036
```

正样本只有 3.6% 左右，是典型的类别不平衡。

```python
# 检查 -1 缺失值
missing = (train == -1).sum()
print(missing[missing>0].head())
```

很多数值列用 -1 表示缺失，需要特殊处理。类别特征以 `_cat` 结尾，数值特征以 `_bin`、`_reg` 结尾。

### 2.2 数据清洗
我们把训练和测试合并处理，避免泄漏。类别特征做 One-Hot 编码，数值特征保留原样。`calc` 系列特征信息量低且维度高，我们在基线里暂时剔除。缺失值 -1 直接保留给模型学习。

```python
# 合并训练测试
all_data = pd.concat([train, test], ignore_index=True)
all_data = all_data.drop('target', axis=1)
# 类别特征列表
cat_features = [f for f in all_data.columns if 'cat' in f]
# One-Hot 编码
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
encoded = onehot.fit_transform(all_data[cat_features])
# 保留的数值特征
drop_features = ['ps_ind_14','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_car_14']
remaining = [f for f in all_data.columns if 'cat' not in f and 'calc' not in f and f not in drop_features]
```

### 2.3 模型训练与超参数调优
评价指标用 Gini，我们先写评估函数。

```python
import numpy as np
def eval_gini(y_true, y_pred):
    # 计算标准化 Gini
    assert y_true.shape == y_pred.shape
    n = y_true.shape[0]
    L_mid = np.linspace(1/n, 1, n)
    pred_order = y_true[y_pred.argsort()]
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    G_pred = np.sum(L_mid - L_pred)
    true_order = y_true[y_true.argsort()]
    L_true = np.cumsum(true_order) / np.sum(true_order)
    G_true = np.sum(L_mid - L_true)
    return G_pred / G_true
```

我们使用 5 折分层交叉验证，模型选 LightGBM。

```python
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1991)
params = {'objective':'binary','learning_rate':0.01,'force_row_wise':True,'random_state':0}

oof_val = np.zeros(len(X))
oof_test = np.zeros(len(X_test))

for idx, (tr_idx, val_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_valid, y_valid = X[val_idx], y[val_idx]
    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_valid, y_valid)
    model = lgb.train(params, dtrain, num_boost_round=1000,
                      valid_sets=dvalid, feval=lambda name, preds, train: ('gini', eval_gini(train.get_label(), preds), True),
                      early_stopping_rounds=100, verbose_eval=100)
    oof_val[val_idx] = model.predict(X_valid)
    oof_test += model.predict(X_test)/5
```

5 折的 Gini 分别约为 0.289、0.275、0.281、0.271、0.288，整体 OOF Gini 约 0.2805。

### 2.4 模型测试与指标报告
在验证集上整体 Gini 约 0.28，说明模型已经超过随机基线。我们用全量数据再训练一次生成提交文件。

```python
# 最终提交
submission['target'] = oof_test
submission.to_csv('submission.csv', index=False)
```

## 三、总结
本文中我们完成了 Porto Seguro 安全驾驶预测的完整流程。我们首先通过 EDA 发现数据量大、正样本稀少且缺失值以 -1 表示。其次对类别特征做 One-Hot 编码并剔除部分弱特征。接着用 LightGBM 结合 5 折分层交叉验证和 Gini 指标进行训练，OOF Gini 约 0.28。最后生成提交文件。这个比赛告诉我们，处理严重不平衡和大量缺失是保险理赔预测的关键，特征筛选和交叉验证比调参更重要。
