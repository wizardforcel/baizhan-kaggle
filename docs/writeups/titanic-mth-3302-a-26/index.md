# Titanic MTH-3302-A-26

## 一、比赛简介

### 1.1 比赛背景

Titanic 是 Kaggle 上最经典的入门比赛，几乎每个学机器学习的人都做过。它的背景是 1912 年泰坦尼克号沉船事件：船上两千多人，只有一部分人活了下来。比赛把乘客名单整理成表格，让参赛者根据性别、年龄、舱位、票价等字段，预测某位乘客是否幸存。因为数据小、字段直观、结果容易解释，它常被用作分类问题的第一课。即使到今天，泰坦尼克仍然是检验新手流程是否完善的试金石：一个干净的流程能稳定达到 0.78 以上的准确率，说明数据清洗、特征工程、模型评估都没有明显失误。

这个比赛目录下的数据被重新组织过，训练文件叫 `Train.csv`，测试文件叫 `Test.csv`，放在 `titanic-mth-3302-a-26` 目录下。材料中两份笔记本展示了两种思路：第一份非常简洁，只用了四个特征和一个随机森林；第二份做了一整套预处理、探索性分析、交叉验证、多模型比较和 PyTorch 神经网络。我们以第二份为主线，因为它更能体现完整的建模流程。

### 1.2 比赛目标

输入是一位乘客的结构化信息，包括 `Pclass`（舱位等级）、`Sex`（性别）、`Age`（年龄）、`SibSp`（同船兄弟姐妹或配偶数）、`Parch`（同船父母或子女数）、`Fare`（票价）、`Embarked`（登船港口）、`Name`、`Ticket`、`Cabin` 等。目标列是 `Survived`，取值 0 表示遇难，1 表示幸存。

测试集有 418 位乘客，没有 `Survived` 列。我们要为每位乘客输出一个 0 或 1，提交文件包含 `PassengerId` 和 `Survived` 两列。训练集原本有 891 行，但材料中的预处理会删掉 `Age` 缺失的行，所以实际进入模型的行数会少一些。除了预测本身，我们还要保证输出行数和 `PassengerId` 顺序与测试文件一致，否则即使模型预测得很准，提交也会被判无效。这一点在材料的两份笔记本里都反复体现：先保存好 ID，再动特征表。

### 1.3 评价标准

这是一个二分类问题，比赛使用准确率（Accuracy）作为评价指标，也就是预测正确的乘客比例。准确率越高越好，1.0 是满分。材料内部还使用了混淆矩阵、精确率、召回率、F1 和 ROC AUC 来更细致地看模型表现。

准确率有一个特点：当两类样本不平衡时，它可能误导人。泰坦尼克数据里遇难者大约占六成，幸存者约四成，比例还算均衡，所以准确率仍然是一个合理的总指标。但在调参时，我们会同时看训练准确率和验证准确率，避免模型在训练集上背答案。如果只看训练准确率，一棵很深的树可以轻松达到 0.95 以上，但它在验证集上可能只有 0.78，这种差距就是过拟合的信号。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读入训练集和测试集，把 `PassengerId` 单独存起来，因为提交文件要用它，而模型训练时不需要它。然后删掉三个文本列：`Name`、`Ticket`、`Cabin`。这三列原始形式不规则，直接喂给模型没有意义；其中 `Cabin` 的缺失又特别多，保留反而增加噪声。

```python
import pandas as pd

train_df = pd.read_csv('/kaggle/input/titanic-mth-3302-a-26/Train.csv')
test_df = pd.read_csv('/kaggle/input/titanic-mth-3302-a-26/Test.csv')

ID = test_df['PassengerId']                       # 提交时要用
drop_cols = ['Name', 'Ticket', 'Cabin']
train_df = train_df.drop(drop_cols + ['PassengerId'], axis=1)
test_df = test_df.drop(drop_cols + ['PassengerId'], axis=1)
```

接着我们看缺失情况。`Age` 缺失最多，训练集约两成；`Embarked` 只缺两行；`Fare` 在测试集里可能缺一行。材料里的第一版代码用 `dropna()` 处理 `Age`，结果训练集少了 179 行、测试集少了 87 行，导致提交不完整。后来改成用中位数填补，这是一个重要的修正。

```python
print(train_df.isna().sum())
print(test_df.isna().sum())
```

然后我们画箱线图和直方图，看每个数值特征的分布和离群点。箱线图能直观看到 `Fare` 有很长的右尾：少数头等舱票价极高，大部分乘客票价很低。直方图则显示 `Age` 大致呈钟形，但有一个婴儿峰。我们还画了相关矩阵，看哪些特征和 `Survived` 关系最强。从常识和图表都能看出，`Sex` 与 `Survived` 的相关性最高，其次是 `Pclass` 和 `Fare`。

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for ax, col in zip(axes.ravel(), train_df.columns):
    sns.histplot(train_df[col], kde=True, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
```

### 2.2 数据清洗

数据清洗分四步：缺失值填补、类别编码、离群值处理、标准化。

缺失值方面，`Embarked` 用训练集的众数填补，`Fare` 用同舱位等级的中位数填补，`Age` 用训练集的中位数填补。所有统计量都只在训练集上计算，再应用到测试集，避免测试信息泄漏。

```python
# 这些统计量只能从训练集算，不能碰测试集
embarked_mode = train_df['Embarked'].mode()[0]
age_median = train_df['Age'].median()
fare_median_pclass = train_df.groupby('Pclass')['Fare'].median()

train_df['Embarked'] = train_df['Embarked'].fillna(embarked_mode)
test_df['Age'] = test_df['Age'].fillna(age_median)
test_df['Embarked'] = test_df['Embarked'].fillna(embarked_mode)
test_df['Fare'] = test_df['Fare'].fillna(
    test_df['Pclass'].map(fare_median_pclass))
```

类别编码方面，`Sex` 映射成 female=0、male=1，`Embarked` 映射成 C=0、Q=1、S=2。材料强调要用固定映射，而不是在测试集上单独 `fit` 一个 `LabelEncoder`，否则训练集和测试集可能得到不一致的编号。

```python
sex_map = {'female': 0, 'male': 1}
emb_map = {'C': 0, 'Q': 1, 'S': 2}

for df in (train_df, test_df):
    df['Sex'] = df['Sex'].map(sex_map)
    df['Embarked'] = df['Embarked'].map(emb_map)
```

离群值方面，我们只处理 `Fare`。用 IQR 规则：低于 Q1 - 1.5×IQR 或高于 Q3 + 1.5×IQR 的值，分别截断到边界。注意边界必须从训练集算，再应用到验证集和测试集；材料里修正过一个 bug：旧代码把 Age 的边界套到了所有列上，这是典型的「一刀切」错误。

```python
def fit_outlier_bounds(X_train, cols, IQR=1.5):
    bounds = {}
    for col in cols:
        q1, q3 = X_train[col].quantile(0.25), X_train[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - IQR * iqr, q3 + IQR * iqr)
    return bounds

def apply_outlier_bounds(df, bounds):
    df = df.copy()
    for col, (low, high) in bounds.items():
        df[col] = df[col].clip(lower=low, upper=high)
    return df

bounds = fit_outlier_bounds(X_train, ['Fare'])
X_train_out = apply_outlier_bounds(X_train, bounds)
X_test_out = apply_outlier_bounds(X_test, bounds)
```

最后做标准化。标准化的意思是把每列减去均值、除以标准差，让所有特征处在同一量级。树模型其实不要求标准化，但逻辑回归、LDA 和神经网络对尺度敏感，所以统一做一次更省事。同样，`StandardScaler` 只在训练集上 `fit`。

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_out)
X_test_scaled = scaler.transform(X_test_out)   # 用训练集的均值和标准差
```

### 2.3 模型训练与超参数调优

我们先用 5 折交叉验证把数据切成五份，取其中一份作为验证。材料特别修正了「循环结束后变量默认指向最后一折」的问题，改成显式取 `folds[-1]`。

```python
from sklearn.model_selection import KFold

X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(kf.split(X, y))
train_index, test_index = folds[-1]          # 明确指定用最后一折

X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
```

基线模型是 XGBoost。XGBoost 是梯度提升树的一种，意思是训练很多棵树，每棵新树都去修正前面模型的错误。我们先跑一个基础配置：100 棵树、默认深度、随机种子固定。

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    n_estimators=100,       # 旧版只用了 5 棵，明显欠拟合
)
model.fit(X_train_out, y_train)
```

然后我们用 `Pipeline` 加 `GridSearchCV` 做系统调参。Pipeline 的作用是把「标准化 + 分类器」绑成一个整体，交叉验证时每一折内部重新 fit scaler，避免数据泄漏。`param_grid` 里放了四个候选模型：逻辑回归、XGBoost、线性判别分析和随机森林，每个模型各有一组超参数。

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(eval_metric='logloss', random_state=42)),
])

param_grid = [
    {'classifier': [LogisticRegression(max_iter=1000)],
     'classifier__C': [0.01, 0.1, 1, 10]},
    {'classifier': [XGBClassifier(random_state=42, eval_metric='logloss')],
     'classifier__n_estimators': [100, 200, 300],
     'classifier__learning_rate': [0.01, 0.1, 0.2],
     'classifier__max_depth': [3, 4, 7]},
    {'classifier': [LinearDiscriminantAnalysis()],
     'classifier__solver': ['svd', 'lsqr']},
    {'classifier': [RandomForestClassifier(random_state=42)],
     'classifier__n_estimators': [100, 200, 300],
     'classifier__max_depth': [None, 5, 10]},
]

grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                           scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_out, y_train)
print('Best params:', grid_search.best_params_)
print('Best CV accuracy:', grid_search.best_score_)
```

最后我们还在 PyTorch 里搭了一个全连接网络做对比。网络结构是 5 个隐藏层，宽度依次为 512、256、128、64、32，每层后面接 ReLU 和 Dropout，最后用一个神经元加 Sigmoid 输出幸存概率。损失函数是二元交叉熵，优化器是 Adam，并加了 weight decay 做正则化。神经网络的好处是它能自动学习特征之间的非线性交互，比如「女性且儿童」可能比单独看性别更有信息量。缺点是需要调学习率、Dropout 比例、训练轮数，而且在小数据上容易过拟合。所以材料里先用早停和 Dropout 控制复杂度，再用不同学习率做对照，最终选出效果最好的学习率。

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Deep(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, 32)
        self.layer6 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop(self.relu(self.layer1(x)))
        x = self.drop(self.relu(self.layer2(x)))
        x = self.drop(self.relu(self.layer3(x)))
        x = self.drop(self.relu(self.layer4(x)))
        x = self.drop(self.relu(self.layer5(x)))
        return self.sigmoid(self.layer6(x))
```

学习率我们试了 0.0003、0.001、0.003、0.01 四档，每档训练 100 轮。这里材料修正了一个关键错误：旧代码只建了一个模型，四档学习率依次在同一组权重上继续训练，导致后面的比较不公平。正确做法是每一档都新建一个模型，并固定随机种子。

```python
lrs = [0.0003, 0.001, 0.003, 0.01]
result = {}

for lr in lrs:
    torch.manual_seed(42)                 # 每档都重置种子
    model_lr = Deep(X_train_scaled.shape[1])   # 每档都新建模型
    # 训练与评估略，记录每档最好的验证准确率
    result[lr] = train_and_eval(model_lr, lr)
```

### 2.4 模型测试与指标报告

评估时我们同时打印训练集和验证集的准确率、分类报告和混淆矩阵。分类报告包含精确率、召回率和 F1：精确率是「预测为幸存的人里有多少真的幸存」，召回率是「真的幸存者里有多少被找出来」。在救援场景里，召回率往往更值得关注，因为漏掉一个可救援的人代价更高。

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

for name, Xs, ys in [('TRAIN', X_train_out, y_train),
                     ('VALID', X_test_out, y_test)]:
    pred = model.predict(Xs)
    print(name, accuracy_score(ys, pred))
    print(classification_report(ys, pred))
    print(confusion_matrix(ys, pred))
```

如果训练准确率远高于验证准确率，说明过拟合；如果两者都很低，说明欠拟合。材料里提到的一个典型问题是树太浅或树太少：`n_estimators=5` 时模型明显欠拟合，改成 100 后训练和验证都提升。

最终提交时，我们在完整训练集上重新拟合最佳模型（如果是 Pipeline，就直接喂未标准化的原始特征，因为 scaler 在 Pipeline 内部），再对测试集预测，生成 418 行的提交文件。

```python
best_model.fit(X_full_out, y)
preds = best_model.predict(X_tests)

submission = pd.DataFrame({
    'PassengerId': ID,
    'Survived': preds,
})
submission.to_csv('submission.csv', index=False)
print(submission.shape)          # 应该是 (418, 2)
```

如果走 PyTorch 线路，流程类似：用全量训练集拟合最终的 scaler 和离群边界，训练最终网络，输出概率后以 0.5 为阈值转成 0/1，再写提交文件。材料中的提交检查打印了 `value_counts()`，确认预测里两类都有，没有全部塌到一边。

关于阈值，我们这里直接用了 0.5，因为训练数据的正负比例大致接近四六开，并没有极端不平衡。如果幸存者只占很小一部分，0.5 就会让模型倾向于全部预测为 0，这时需要根据验证集上的 F1 或召回率重新选阈值。我们在报告里把「概率质量」和「阈值选择」分开看：前者由模型和特征决定，后者只是一个业务上的取舍，不能靠反复提交排行榜来碰运气。

## 三、总结

本文中我们完成了 Titanic MTH-3302-A-26 的完整分类流程。我们首先读入 `Train.csv` 和 `Test.csv`，删除 `Name`、`Ticket`、`Cabin` 三个文本列，并通过箱线图、直方图和相关矩阵确认 `Sex`、`Pclass`、`Fare` 是最重要的信号；其次用训练集统计量填补 `Age`、`Fare`、`Embarked`，用固定映射编码类别，用 IQR 规则截断 `Fare` 的离群值，并只对训练集拟合标准化；接着用 5 折交叉验证划分数据，先跑 XGBoost 基线，再用 Pipeline 加 GridSearchCV 比较逻辑回归、XGBoost、LDA 和随机森林，另外用 PyTorch 五层全连接网络做对照；最后在完整训练集上重拟合最佳模型，对 418 位测试乘客预测并生成提交文件。

这个练习让我们看到，泰坦尼克这种小数据比赛的关键不是模型多复杂，而是流程是否干净：缺失值不能随手 `dropna`，编码不能训练集和测试集各搞一套，离群值边界和 scaler 都只能从训练集学。材料里那些「修正」条目，本质上都在防同一件事——信息泄漏。
