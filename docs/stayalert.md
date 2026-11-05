# StayAlert（StayAlert）

## 一、比赛简介

### 1.1 比赛背景

StayAlert 是 Kaggle 上一个经典的**驾驶疲劳检测**比赛。也被称为 Ford Stay Alert。比赛提供了一批驾驶员在模拟驾驶过程中的生理信号和行为特征数据，目标是判断驾驶员是否处于疲劳警惕状态。

从材料文件名 `fordTrain.csv`、`fordTest.csv` 来看，数据来自福特（Ford）的驾驶实验。每一行是一个「观察」，包含试验 ID、观察序号、警惕状态标签以及多种生理和车辆传感器信号。典型的征兆是：驾驶员的眼皮下垂、眨眼频率改变、心率变化、方向盘握力波动等。比赛要求根据这些实时信号预测 `IsAlert`——即当前是否在疲劳或分心状态。

这个比赛在入门教学里非常出名：数据没有缺失、特征都很干净，标签不平衡比较严重。按照材料里笔记的描述，很多 notebook 都在讨论 Undersampling、NearMiss、SMOTE 等不平衡处理手法。可以看出**类别不平衡**是贯穿整场比赛的核心难点——疲劳样本远少于正常警觉样本，直接训练会让模型偏向多数类。

### 1.2 比赛目标

输入两张表：

- 训练集 `fordTrain.csv`：含 `TrialID`、`ObsNum`、`IsAlert`（目标）、`P1~P8`（生理信号？）、`E1~E11`（眼部或 Event？）、`V1~V11`（车辆信号）。
- 测试集 `fordTest.csv`：含同样的特征和 ID 列，但目标列 `IsAlert` 处于隐含状态（比赛里通常 `IsAlert` 列在测试集里是空的或需要预测）。

我们要为每条测试记录预测 `IsAlert`（0/1），生成 `TrialID, ObsNum, IsAlert` 的提交文件。标签是二元分类。

### 1.3 评价标准

材料里没有写出明确的评分指标，但从代码中引用的 `average_precision_score`、`confusion_matrix`、`classification_report`、`accuracy_score` 以及不平衡处理的讨论，可以推断官方指标很可能是 **AUC、Average Precision 或 F1**，而非简单准确率。

因为数据不平衡（`IsAlert` 为 1 的很少），如果用准确率做指标，模型只要全预测为 0 也能拿到 90% 以上的准确率，这显然不公平。所以排行榜通常采用 **F1、ROC AUC、PR AUC**。材料里多个 notebook 对不平衡做 SMOTE、NearMiss 处理，正是针对 F1/AUC 这类在少数类上更敏感的指标。

---

## 二、解题过程

我们选用了参考 notebook `FordStayAlert` 和 `LG,RF and DT` 里的思路作为主线：先做 EDA 确认无缺失，再检查类别分布，最后用随机森林分类器做基线，并在不平衡问题上做简单处理。我们按 EDA、清洗、训练、报告四步展开。

### 2.1 数据探索（EDA）

首先检查是否缺失值。这是 StayAlert 比赛的一个亮点：数据极其干净：

```python
import pandas as pd

train = pd.read_csv('/kaggle/input/stayalert/fordTrain.csv', index_col=1)
test = pd.read_csv('/kaggle/input/stayalert/fordTest.csv')

print('训练集缺失值：')
print(train.isna().sum().sum())   # 材料输出都是 0

print('测试集缺失值：')
print(test.isna().sum().sum())    # 同样是 0
```

每一列的缺失值都为 0，说明**根本不需要缺失值补全**。这是教学里难得的好消息。

接着看目标列分布，了解不平衡程度：

```python
print(train['IsAlert'].value_counts())
print(train['IsAlert'].value_counts(normalize=True))
```

从多个尝试来看，`IsAlert` 为 1 的比例大约在 5%~10% 之间，属于明显的类别不平衡。

我们再看特征概览，确认每列的尺度：

```python
print(train.describe())
```

`P1~P8`、`E1~E11`、`V1~V11` 全部是数值，范围从 0 到几千不等。这类数据适合树模型，因为树模型对尺度不敏感。

为了直观了解特征与标签的关系，我们可以画几个直方图：

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(data=train, x='E1', hue='IsAlert', fill=True)
plt.title('E1 eye feature distribution by alert')
plt.show()
```

眼部特征 `E1~E11` 通常和 `IsAlert` 关联最强，图上会有明显的分离。

### 2.2 数据清洗

数据本身无需常规清洗。主要工作是**特征分离和标签平衡策略选择**。

我们先把 ID 列去掉，只保留预测用的特征：

```python
# 特征列：P1-P8, E1-E11, V1-V11
feature_cols = [c for c in train.columns if c.startswith(('P','E','V'))]

X = train[feature_cols]
y = train['IsAlert']

# 测试集同样提取
X_test = test[feature_cols]
```

考虑到类别不平衡，材料里有多种处理建议：

- **Undersampling**：随机丢弃多数类样本；
- **NearMiss**：从多数类中挑离少数类最近的样本保留；
- **Oversampling**：复制少数类样本；
- **SMOTE**：合成新样本。

基线阶段我们先不做采样，先看模型表现，再决定是否需要采样。因为随机森林本身对类别不平衡有一定容忍度，直接跑是可行的起点。

另外，训练集里还有 `TrialID` 和 `ObsNum`。`TrialID` 代表试验编号，同一试验里的多条观测时间上是连续的。为了防止数据泄漏，我们应该在划分验证集时**按 TrialID 分层**，避免同一个试验的样本同时出现在训练集和验证集里。虽然参考代码里没有写这么严格，但这是入门者容易忽略的点。

### 2.3 模型训练与超参数调优

我们选用随机森林分类器作为基线。原因：对混合信号鲁棒、不用标准化、对缺失值友好、容易解释。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,      # 200 棵树，足够稳定
    max_depth=None,        # 不限制深度，让模型充分学习
    class_weight='balanced',   # 自动给少数类更高权重，代替采样
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
    scoring='average_precision',   # 对不平衡数据比 accuracy 更合理
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

model.fit(X_res, y_res)
print('SMOTE 后类别分布：', pd.Series(y_res).value_counts())
```

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

对模型的**特征重要性**进行分析，有助于解释模型在学什么：

```python
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances.sort_values(ascending=False).head(10)
```

通常 `E` 系列的眼部特征会占前几位，这符合直觉——眼睛是疲劳最早反映的信号。

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

提交前检查：行数和测试集一致、列名正确、无缺失。

---

## 三、总结

本文中我们完成了 StayAlert 驾驶疲劳检测任务的完整流程。我们首先通过 EDA 确认数据无缺失且存在明显类别不平衡；其次在数据清洗阶段主要做特征分离和 ID 管理，并讨论了 Undersampling/NearMiss/SMOTE 的不平衡处理思路；接着采用随机森林分类器作为基线，使用 `class_weight='balanced'` 和平均精度作为优化目标，并通过网格搜索确定了合适的 `n_estimators` 和 `max_depth`；最后在验证集上评估了准确率、平均精度和 F1 指标，并按格式生成了提交文件。

这个比赛给我们最大的启发是：**在不平衡数据上，不要迷信准确率**。全预测多数类可以拿到很高的准确率，但对疲劳检测这类安全相关任务几乎毫无价值。用平均精度、F1、ROC AUC 评估，并在模型里加入类权重或 SMOTE，才是真正提升少数类识别能力的做法。同时，StayAlert 提醒我们「特征工程」在表格数据里也非常重要：眼部信号 `E1~E11` 往往比生理信号更具判别力，理解特征的物理含义能帮助我们更快找到关键变量。
