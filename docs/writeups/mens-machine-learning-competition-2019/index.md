# NCAA Men's Basketball Tournament (March Madness 2019)

## 一、比赛简介

### 1.1 比赛背景

Men's Machine Learning Competition 2019 延续了 Kaggle 的 March Madness 系列。与 2018 版相比，它新增了 daily RPI 数据、Massey 排名、play-by-play 逐回合数据等丰富资源。比赛要求为 2019 年锦标赛的所有可能配对估计获胜概率。

数据被组织成多个文件夹：datafiles（主要比赛表）、masseyordinals（Massey 评级）、playbyplay_XXXX（逐回合数据）、daily-rpi-2019（每日 RPI 关键指标的预加工数据）。

### 1.2 比赛目标

对每一场 Stage1 比赛预测主队获胜概率。提交格式为 Id、Pred。模型输入是按“配对”组织的特征表，例如 train_stage1_Prepped.csv 里已经包含双方场均得分 PPG_1/PPG_2、对阵场均失分 OPPG、三分命中率 FG3_PCT、AOE/ADE/AEE 以及 seed_diff、target 等列。

### 1.3 评价标准

仍是 Log Loss，越低越好，0/1 错误预测会让 loss 变为无穷大。MAHER starter 特别提醒提交概率不要取极值。

## 二、解题过程

### 2.1 数据探索（EDA）

我们先用 seaborn 和 pandas 浏览 daily-rpi 提供的预加工数据，打印每张表的前几行：

```python
import pandas as pd
st1_tr = pd.read_csv('../input/daily-rpi-2019/train_stage1_Prepped.csv')
st1_tt = pd.read_csv('../input/daily-rpi-2019/test_stage1_Prepped.csv')
print(st1_tr.shape)
st1_tr.head()
```

可以看到 train 表按“队1 vs 队2”展开，每一行是对一场比赛的对比特征，包括 PPG_diff、OPPG_diff、FG3_PCT_diff、seed_diff 等，target 列是 0/1 标签，输入端很友好。

### 2.2 数据清洗

很多比赛特征是差值和比例，可能出现缺失值。我们先检查缺失情况并统一填充：

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in st1_tr.columns:
    if st1_tr[col].dtype == 'object':
        st1_tr[col] = le.fit_transform(st1_tr[col].astype(str))
```

再对字段做相关性检查，去掉几乎常量的列，避免冗余特征干扰树模型。

### 2.3 模型训练与超参数调优

我们选用 XGBoost 分类器，先做默认参数的 baseline，再用 train_test_split 划分，观察 log loss：

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
X_train, X_val, y_train, y_val = train_test_split(feats, target, test_size=0.2)
model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4)
model.fit(X_train, y_train)
val_loss = log_loss(y_val, model.predict_proba(X_val)[:, 1])
print('Log Loss: ', val_loss)
```

我们尝试不同的树深度与学习率组合寻找较稳定参数。

### 2.4 模型测试与指标报告

在验证集上计算 Log Loss 并观察特征重要性，确认 seed_diff、PPG_diff 等对胜负影响最大：

```python
imp = pd.DataFrame({'feature': feats.columns, 'importance': model.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
```

最后对测试集预测 0/1 概率写出提交文件。由于 log loss 对极端预测惩罚大，我们对预测概率做裁剪，限制在 0.05~0.95 之间：

```python
p = model.predict_proba(X_test)[:, 1]
p = np.clip(p, 0.05, 0.95)
pd.DataFrame({'Id': test_ids, 'Pred': p}).to_csv('submission.csv', index=False)
```

## 三、总结

本文中我们完成了 2019 March Madness 的完整流程。我们首先通过 EDA 发现 daily-rpi 已把数据加工成“配对差值”表，其次对缺失值和常数列做了清洗，接着用 XGBoost 分类器配合深度与学习率的组合调参训练，最后用 Log Loss 评估并裁剪概率写出提交。2019 版加入的预加工特征表让 feature engineering 更轻松，但核心仍是“把每支球队的真实实力拟合到胜负概率上”。