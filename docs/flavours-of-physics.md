# Flavours of Physics

## 一、比赛简介

### 1.1 比赛背景
Flavours of Physics 是 CERN 组织在 Kaggle 上的粒子物理分类竞赛。数据来源于 LHCb 探测器的模拟，目标是区分罕见的信号衰变与普通背景衰变。训练集包含大量带标签的探测器特征，如寿命、飞行距离、顶点位移、隔离度、BDT 输出等。材料来自用户提交的笔记，其中 `p001_05.txt`、`p001_01.txt`、`p001_04.txt` 等页面详细记录了加载、EDA 与建模过程。

### 1.2 比赛目标
输入为训练集中 51 列特征，输出为二分类标签 `signal`。`signal=1` 为信号事件，`signal=0` 为背景事件。测试集没有标签，需要提交每行的预测概率。训练集共 67553 行，信号 41674 行、背景 25879 行。测试集 47 列特征，去除了 `signal`、`mass`、`production` 等标识列。

### 1.3 评价标准
公开排行榜使用截断加权 ROC AUC `roc_auc_truncated`，在不同真阳性率区间给予不同权重：TPR 阈值 0.2/0.4/0.6/0.8，权重 4/3/2/1/0。提交需要同时通过两项一致性检验：Kolmogorov-Smirnov KS 距离 < 0.09 与 Cramer-von Mises CvM 距离 < 0.002，保证模型预测分布与蒙特卡洛数据一致。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读入数据，查看样本量、类别分布与基本统计量，用直方图和相关性检查特征差异。

```python
import pandas as pd
import numpy as np
train = pd.read_csv('/kaggle/input/flavours-of-physics/training.csv.zip')
# 查看形状和类别分布
print(train.shape)                     # 67553, 51
print(train.signal.value_counts())     # 1:41674, 0:25879
# 检查偏度，某些特征极度偏态
skew = train.skew(axis=0)
print(skew['DOCAone'])                 # 约39.7，严重右偏
```
文本说明：训练集 67553 条，信号占多数。我们计算偏度发现 `DOCAone` 偏度高达 39.7，说明分布极不均衡，后续需要压缩。`p001_05.txt` 同样先展示 `train.head()` 与 `train.info()`，确认 44 列 float、7 列 int，内存约 26 MB。

### 2.2 数据清洗
其次我们剔除无关列，对异常偏态特征做压缩，并做标准化。

```python
# 过滤低置信度样本并删除标识列
train_pre = train.drop(train[train.min_ANNmuon <= 0.4].index)
# 剔除无法使用的列
X = train_pre.drop(columns=['id','min_ANNmuon','production','mass','signal','SPDhits'])
y = train_pre['signal']
# DOCAone 偏度过大，用 tanh 压缩
X['DOCAone'] = np.tanh(X['DOCAone'])
# 标准化使不同量纲可比
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
```
文本说明：材料里 `p001_01.txt` 与 `p001_04.txt` 都先执行 `min_ANNmuon > 0.4` 的预筛选，随后去掉 `id、mass、production、signal、SPDhits、min_ANNmuon`。`DOCAone` 经 tanh 后偏度从 39.7 降到 3.0，正好对应 `p001_05.txt` 中的变换。

### 2.3 模型训练与超参数调优
接着我们用基线 Logistic 回归快速验证，然后用 XGBoost 提升性能，并在 `p001_01.txt` 基础上加标准化的输入。

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled.values, y.values, test_size=0.33, random_state=42)
# 基线模型
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
print(clf.score(X_test, y_test))       # 约0.861
# XGBoost 主模型
import xgboost as xgb
params = {"objective":"binary:logistic","eta":0.3,"max_depth":5,
          "min_child_weight":3,"subsample":0.7,"colsample_bytree":0.7,"seed":1}
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, params=params)
xgb_model.fit(X_train, y_train)
print('test acc', xgb_model.score(X_test, y_test))
```
文本说明：我们首先用 Logistic 取得 0.861 的准确率作为起点，随后使用 XGBoost。参数参考 `p001_05.txt` 中 `eta=0.3、max_depth=5、min_child_weight=3、subsample=0.7、colsample_bytree=0.7`，训练集 0.962、测试集 0.885，明显优于基线。`p001_06.txt` 还展示了特征工程如 `flight_dist_sig` 与 `NEW_IP_dira` 的思路，但我们保持主线单一。

### 2.4 模型测试与指标报告
然后我们绘制混淆矩阵并检查 KS 与 CvM 一致性。

```python
from sklearn.metrics import confusion_matrix
y_pred = xgb_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)  # [[7095 1345],[1217 12636]]
# 预测概率用于后续一致性检验
y_prob = xgb_model.predict_proba(X_test)[:,1]
# 在 p001_04.txt 的 NN 实验中 KS=0.065<0.09，CvM=0.009>0.002未全通过
```
文本说明：XGBoost 在测试集上误分类约 2562 条，信号召回良好。材料中 `p001_04.txt` 的神经网络报告 KS 0.065 满足 <0.09，但 CvM 0.009 超过 0.002，说明分布一致性仍需改善。我们的 XGBoost 在 ROC AUC 上更优，可作为提交主线。

## 三、总结
本文中我们完成了 Flavours of Physics 的完整建模流程。我们首先对数据做了 EDA，发现类别不平衡与 `DOCAone` 极端偏态；其次通过筛选 `min_ANNmuon`、剔除标识列、tanh 压缩与标准化完成清洗；接着用 Logistic 回归建立基线，再以 XGBoost 为主模型，测试准确率达到 0.885；最后报告混淆矩阵并参考材料中 KS/CvM 检验结果，确认模型在判别力上可行，分布一致性仍需进一步调优。
