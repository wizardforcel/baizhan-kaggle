# StayAlert 驾驶疲劳检测（StayAlert）

## 一、比赛简介

### 1.1 比赛背景

StayAlert 是 Kaggle 上一个经典的**实时驾驶疲劳检测**比赛，源自福特汽车（Ford）与学术机构联合开展的模拟驾驶实验。比赛名称也常被写作 Ford Stay Alert 或 FordStayAlert，核心任务是根据驾驶过程中的多模态传感器信号，在线判断驾驶员是否进入疲劳或注意力分散状态。

数据来自模拟驾驶舱实验，每一场试验由一名被试在一定时长内进行驾驶操作，过程中采集大量生理和车辆行为特征。特征大致可分为三组：

- **P1~P8**：生理信号类，通常与心率、心率变异性、皮肤电、呼吸等有关；
- **E1~E11**：眼部相关信号，通常与眨眼频率、眼睑下垂度、眼动轨迹、视线偏离等有关；
- **V1~V11**：车辆行为信号，通常与方向盘转角、油门踏板、刹车压力、车道偏移量等有关。

疲劳的典型征兆是：眼睑逐渐下垂、眨眼间隔变长且幅度增大、心率变异性降低、方向盘握力出现微小颤抖、车道保持能力下降。这些微妙的变化往往不会以单一特征呈现，而是多特征共同偏移。

这个比赛在入门教学里非常出名，原因有三：**数据非常干净**，几乎没有缺失值；**标签明确**，每条观测都有 0/1 标签；**不平衡严重**，疲劳样本远少于正常样本。材料里多个笔记本都在讨论 Undersampling、NearMiss、SMOTE 等不平衡处理手法，可见类别不平衡是贯穿整场比赛的核心难点。如果直接训练，会让模型偏向多数类，把所有样本都预测为「未疲劳」。

### 1.2 比赛目标

输入两张表：

- 训练集 `fordTrain.csv`：含 `TrialID`（试验编号）、`ObsNum`（观察序号）、`IsAlert`（目标，0 表示正常警觉，1 表示疲劳/分心）、`P1~P8`（生理）、`E1~E11`（眼部）、`V1~V11`（车辆）共 30 余个特征列；
- 测试集 `fordTest.csv`：特征列与训练集完全一致，但 `IsAlert` 列为空，需要我们预测。

我们要为每条测试记录预测 `IsAlert`（0/1），生成 `TrialID, ObsNum, IsAlert` 三列的提交文件。标签是二元分类，提交格式严格对应样例。

### 1.3 评价标准

材料里没有直接写出官方评分公式，但从代码中引用的 `average_precision_score`、`confusion_matrix`、`classification_report`、`accuracy_score` 以及不平衡处理的讨论，可以推断官方指标很可能是 **AUC、Average Precision（PR AUC）或 F1**，而非简单准确率。

因为数据不平衡（疲劳样本约占 5%~10%），如果用准确率做指标，模型只要全预测为 0 也能拿到 90% 以上的准确率，这显然不公平。所以排行榜通常采用 **F1、ROC AUC、PR AUC**。材料里多个 notebook 对不平衡做 SMOTE、NearMiss 处理，正是针对 F1/AUC 这类在少数类上更敏感的指标。

---

## 二、解题过程

我们的主线基于参考 notebook `FordStayAlert`、`LG,RF and DT`、`FordAlert-1` 等思路：先做 EDA 确认无缺失，再检查类别分布，选用随机森林分类器做基线，并在模型里加入 `class_weight='balanced'` 来处理不平衡。我们按 EDA、清洗、训练、报告四步展开。

### 2.1 数据探索（EDA）

首先检查是否缺失值。这是 StayAlert 比赛的一个亮点，数据极其干净：

```python
import pandas as pd

train = pd.read_csv('/kaggle/input/stayalert/fordTrain.csv', index_col=1)
test = pd.read_csv('/kaggle/input/stayalert/fordTest.csv')

print('训练集缺失值：')
print(train.isna().sum().sum())   # 材料输出都是 0

print('测试集缺失值：')
print(test.isna().sum().sum())    # 同样是 0
```

每一列的缺失值都为 0，说明根本不需要缺失值补全。

接着看目标列分布，了解不平衡程度：

```python
print(train['IsAlert'].value_counts())
print(train['IsAlert'].value_counts(normalize=True))
```

从多数尝试来看，`IsAlert` 为 1 的比例大约在 5% 到 10% 之间，属于明显的类别不平衡。

我们再看特征概览，确认每列的尺度范围：

```python
print(train.describe())
```

`P1~P8`、`E1~E11`、`V1~V11` 都是数值，范围从 0 到几千不等。这类数据适合树模型，因为树模型对尺度不敏感。

为了直观了解特征与标签的关系，我们可以画几个直方图或核密度图：

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(data=train, x='E1', hue='IsAlert', fill=True)
plt.title('E1 眼部特征分布 by Alert')
plt.show()
```

眼部特征 `E1~E11` 通常和 `IsAlert` 关联最强，图上会有明显的分离。生理特征 `P` 通常次之，车辆特征 `V` 的区分度相对较弱。

我们还可以用相关系数矩阵快速定位关键特征：

```python
corr = train.corr()['IsAlert'].abs().sort_values(ascending=False)
print(corr.head(15))
```

这一步能帮助我们在解释模型时给出合理的依据：不是模型随机构造，而是特征本身有生理意义。

还需要说明的是批次数据的规模。训练集 `fordTrain.csv` 有 60 多万行（材料里 `train.describe()` 的 count 约 604329），所有特征列都没有缺失。60 万条样本对随机森林而言并不算大，单机训练很快，这也是我们在教学基线里放心用随机森林而不是更重的深度网络的原因之一。

综合 EDA 的观察，我们得到三个关键结论：**一是数据无缺失，省去了补全的麻烦；二是类别不平衡明显，必须通过类权重或采样来应对；三是眼部特征与标签的相关性整体高于生理和车辆特征，建模时可优先关注。** 这三个结论直接决定了后面清洗和建模的选择。值得补充的是，`ObsNum` 反映的是同一试验内部的时间顺序，如果我们想做序列建模，还可以把同一 `TrialID` 的观测做成窗口特征，比如计算最近 5 个观测的均值和方差，这能进一步捕捉疲劳累积效应。材料里没有做窗口特征，这里只是为后续进阶留一个思路。

### 2.2 数据清洗

数据本身无需常规清洗。主要工作是特征分离、防止数据泄漏以及不平衡策略选择。

我们先把 ID 列去掉，只保留预测用的特征：

```python
feature_cols = [c for c in train.columns if c.startswith(('P','E','V'))]

X = train[feature_cols]
y = train['IsAlert']

X_test = test[feature_cols]
```

另外，`TrialID` 代表试验编号，同一试验里的多条观测时间上是连续的。为了防止数据泄漏，我们在划分验证集时应该按 `TrialID` 分层，避免同一个试验的样本同时出现在训练集和验证集里：

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=train['TrialID']))

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
```

这一步虽不常见于入门笔记本，但在真实场景里非常重要，否则验证集的性能会被高估。

关于不平衡处理，材料里有多种建议。基线阶段我们先不做采样，先看模型表现，再决定是否需要采样。因为随机森林本身对类别不平衡有一定容忍度，直接跑是可行的起点。

### 2.3 模型训练与超参数调优

我们选用随机森林分类器作为基线。原因：对混合信号鲁棒、不用标准化、对缺失值友好、容易解释。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,          # 200 棵树，足够稳定
    max_depth=None,            # 不限制深度
    class_weight='balanced',   # 自动给少数类更高权重
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
```

这里用 `class_weight='balanced'` 是在不做采样的前提下平衡类别。它会根据类别频率自动加权，等价于在损失里对少数类放大权重。

为了做超参数调优，我们可以先粗略搜索最关键的两个参数：

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
}

grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    param_grid,
    scoring='average_precision',
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train)
print('最佳参数：', grid.best_params_)
print('最佳平均精度：', grid.best_score_)
```

在实际笔记本里，作者还尝试过决策树、逻辑回归、KNN 等模型，最终随机森林在平均精度上表现最稳。

如果想进一步提升不平衡表现，可以用 SMOTE 生成合成样本：

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

model_smote = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
model_smote.fit(X_res, y_res)
print('SMOTE 后类别分布：', pd.Series(y_res).value_counts())
```

对比实验通常会发现：**SMOTE 能让 PR AUC 提升 1~3 个百分点**，但对 AUC 的提升有限，因为 SMOTE 容易引入噪声。另一个常见做法是 NearMiss undersampling，能在不增加样本量的情况下提高模型对少数类的关注。

我们也可以用 `GridSearchCV` 对比「class_weight」和「SMOTE + 无权重」两种策略，选择在验证集上平均精度更高的那一套。

### 2.4 模型测试与指标报告

训练好模型后，先在验证集上评估：

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, average_precision_score

y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

print('验证集准确率：', accuracy_score(y_val, y_pred))
print('平均精度：', average_precision_score(y_val, y_proba))
print(classification_report(y_val, y_pred))
```

因为类别不平衡，`average_precision` 和 `F1` 更能反映少数类预测质量。材料里也强调了这一点。

对模型的特征重要性进行分析，有助于解释模型在学什么：

```python
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances.sort_values(ascending=False).head(10)
```

通常 `E` 系列的眼部特征会占前几位，这符合直觉——眼睛是疲劳最早反映的信号。

我们也可以画混淆矩阵，看看模型主要犯什么错：

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

最后生成提交文件：

```python
test_pred = model.predict(X_test)

submission = pd.DataFrame({
    'TrialID': test['TrialID'],
    'ObsNum': test['ObsNum'],
    'IsAlert': test_pred
})

submission.to_csv('submission.csv', index=False)
print(submission.head())
```

提交前检查：行数和测试集一致、列名正确、无缺失、无类型错误。

为了让模型部署更稳健，还可以做阈值调优。随机森林输出的是概率，我们不一定非要把概率 0.5 作为划分点。针对疲劳检测这种安全任务，漏报比误报更危险，所以可以把阈值从 0.5 降到 0.3 左右，用来降低漏报率。

```python
threshold = 0.3
y_pred_thresh = (y_proba >= threshold).astype(int)

print(classification_report(y_val, y_pred_thresh))
```

这一步能让我们在保证召回率的同时，适当控制误报率。不同阈值下 F1 曲线也能帮助我们找到最合适的平衡点。

## 三、总结

本文中我们完成了 StayAlert 驾驶疲劳检测任务的完整流程。我们首先通过 EDA 确认数据无缺失且存在明显类别不平衡；其次在数据清洗阶段主要做特征分离、按 `TrialID` 分层拆分以防止泄漏，并讨论了 Undersampling/NearMiss/SMOTE 的不平衡处理思路；接着采用随机森林分类器作为基线，使用 `class_weight='balanced'` 和平均精度作为优化目标，并通过网格搜索确定了合适的 `n_estimators` 和 `max_depth`；最后在验证集上评估了准确率、平均精度和 F1 指标，并按格式生成了提交文件。

这个比赛给我们最大的启发是：在不平衡数据上，不要迷信准确率。全预测多数类可以拿到很高的准确率，但对疲劳检测这类安全相关任务几乎毫无价值。用平均精度、F1、ROC AUC 评估，并在模型里加入类权重或 SMOTE，才是真正提升少数类识别能力的做法。同时，StayAlert 提醒我们特征工程的物理意义非常重要：眼部信号 `E1~E11` 往往比生理信号更具判别力，理解特征的来源能帮助我们更快找到关键变量，也让模型解释更可信。
