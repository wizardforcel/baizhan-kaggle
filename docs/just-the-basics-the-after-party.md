# Just the Basics: The After Party（垃圾邮件检测）

## 一、比赛简介

### 1.1 比赛背景

这是一场面向初学者的垃圾邮件识别比赛。名字里的 Just the Basics 就是「只讲最基础的东西」的意思；从参赛笔记标题里的 Strata2013After-party 看，它和 2013 年 Strata 大会期间的入门赛有关，After Party 是入门赛之后的加场。

比赛方从一批邮件里抽出了 100 个数值特征，把每封邮件变成一行 100 个数字，然后要求参赛者判断这封邮件是垃圾邮件还是正常邮件。特征的具体含义没有公开，列名只有 0 到 99，所以这是一道典型的「只给数字、不给业务解释」的题。

这场比赛有几个明显的难点。第一，训练集只有 600 封邮件，而测试集有 4000 封，样本少、特征多，模型很容易过拟合。第二，100 个特征都是匿名的，我们没法靠常识去构造新特征。第三，数据里有缺失值，而且不同列的量纲相差很大，有的列在 0.0 几附近，有的列能到几千。第四，评价指标是 ROC AUC，它关心的是「排序」而不是简单的对错，这会影响我们最后怎么输出预测。

### 1.2 比赛目标

比赛给了三个文件：

| 文件 | 内容 | 形状 |
| --- | --- | --- |
| `train.csv` | 600 封训练邮件的 100 个特征 | 600 × 100 |
| `train_labels.csv` | 600 封训练邮件的标签 | 600 × 1 |
| `test.csv` | 4000 封测试邮件的 100 个特征 | 4000 × 100 |

标签只有两种取值：`1` 表示垃圾邮件（spam），`0` 表示正常邮件（normal）。我们需要给 `test.csv` 里的 4000 封邮件各输出一个预测。比赛说明里写得很清楚：预测可以是 0/1 这样的硬标签，也可以是连续的数字，提交文件不需要表头。

### 1.3 评价标准

比赛用 **ROC 曲线下面积（Area Under the ROC Curve，简称 AUC）** 来评分。

ROC 曲线的横轴是「假阳性率」，也就是正常邮件被误判成垃圾邮件的比例；纵轴是「真阳性率」，也就是垃圾邮件被正确找出来的比例。曲线越往左上角鼓，说明模型越好。AUC 就是这条曲线下面的面积，取值在 0 到 1 之间。

用大白话说：随机拿一封垃圾邮件和一封正常邮件，让模型各打一个分，AUC 就是「垃圾邮件的分数高于正常邮件」的概率。0.5 相当于闭着眼睛瞎猜，1.0 是完美排序。所以 AUC 越高越好。

也正因为 AUC 只看排序，输出「这封邮件是垃圾邮件的概率」通常比直接输出 0 或 1 更有利，因为概率里保留了更多粗细信息。

---

## 二、解题过程

整场比赛我们走了一条很朴素的路线：先把数据摸清楚，补上缺失值，统一量纲，然后拿几个经典模型做交叉验证对比，最后选中随机森林，并用 AUC 来调参。下面按 EDA、数据清洗、模型训练与调参、测试与指标报告四步展开。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，确认数据的形状和格式。这里有个小坑：文件本身没有表头，读取时必须显式写 `header=None`。

```python
# 导入常用库
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")  # 关掉无关警告，输出更干净

# 三个文件都没有表头，必须写 header=None
# 否则 pandas 会把第一条邮件当成列名，训练集只剩下 599 行
train = pd.read_csv("../input/just-the-basics-the-after-party/train.csv", header=None)
test = pd.read_csv("../input/just-the-basics-the-after-party/test.csv", header=None)
labels = pd.read_csv("../input/just-the-basics-the-after-party/train_labels.csv", header=None)

print("train:", train.shape)    # (600, 100)
print("test :", test.shape)     # (4000, 100)
print("label:", labels.shape)   # (600, 1)
```

有一份参赛笔记就是用默认方式读取的，结果 `info()` 显示训练集只有 599 行，比实际少一条，原因正是第一行被当成了列名。这个坑很小，但会让训练数据白白少一条，所以我们在读入时就把它堵住。

接下来我们统计缺失值，并看看每一列的数值范围。

```python
# 缺失值总数：训练集 1745 个，测试集 11882 个
print("训练集缺失值：", train.isnull().sum().sum())
print("测试集缺失值：", test.isnull().sum().sum())

# 看数值范围：不同列的量纲差得很远
# 有的列在 0.0x 附近，有的列能到几千
train.describe().T[["mean", "std", "min", "max"]].head(5)
```

从统计结果能看到，同样是特征，第 0 列很多值在 0.1 左右，第 2 列却能出现 45、94、1955 这样的数字。量纲差这么大，像 KNN 这种靠「距离」做判断的模型会吃大亏，所以后面必须做标准化。

我们还画了一张 100 个特征的相关性热力图，想看看有没有两列几乎一模一样、可以删掉。

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 先标准化再算相关，避免「数值大」的列把相关系数带偏
X_tmp = StandardScaler().fit_transform(train.fillna(train.mean()))

plt.figure(figsize=(12, 12))
sns.heatmap(pd.DataFrame(X_tmp).corr(), cmap="coolwarm")
plt.title("100 个特征的相关性")
plt.show()
```

因为特征都是匿名的，我们没有根据热力图删列，而是保留全部 100 列交给模型自己判断。对树模型来说，多几列不相关的特征通常也不会带来太大伤害。

### 2.2 数据清洗

我们其次处理缺失值和数据格式。这里我们选择用**中位数**填补缺失值。

```python
# 给 100 列起名字 0~99，方便按列循环
colums = list(range(100))
train.columns = colums
test.columns = colums

# 用中位数补缺失值：中位数不容易被极端大值拉走
for i in colums:
    train[i] = train[i].fillna(train[i].median())
    test[i] = test[i].fillna(test[i].median())

# 标签原来是 (600, 1) 的表格，转成一维数组，sklearn 才认
y = np.ravel(labels)

print(train.isnull().sum().sum(), test.isnull().sum().sum())  # 0 0
print(y.shape)  # (600,)
```

有的笔记用均值填补，有的用中位数。考虑到部分列的最大值到了上千，均值容易被极端值拉高，我们主线采用中位数，稳一些。填补完之后，训练集和测试集都没有缺失值了。

然后我们做标准化。标准化的意思是：把每一列减去自己的均值，再除以标准差，让所有列都变成「平均 0、波动 1」的尺度。

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train)    # 训练集：先算均值和标准差
test_scaled = scaler.transform(test)      # 测试集：沿用训练集的参数

# 注意不能对 test 再 fit_transform 一次
# 那样会把测试集的统计信息带进模型，属于数据泄漏
```

这里有个容易犯的错误：有些代码对训练集和测试集分别 `fit_transform`。这样做等于让模型提前知道了测试集的均值和方差，虽然泄漏不算严重，但评估结果会偏乐观。正确做法是只在训练集上 `fit`，测试集只做 `transform`。

### 2.3 模型训练与超参数调优

我们接着从一个最朴素的 KNN 开始，看看这份数据大概有多难。KNN 的思路是：把新邮件和训练集里所有邮件比一比，找最像的 k 封，看它们多数是垃圾还是正常。

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 留出 25% 做验证，random_state 固定保证每次划分一样
X_tr, X_val, y_tr, y_val = train_test_split(
    train, y, test_size=0.25, random_state=0)

# 先不标准化，直接找 3 个最近的邻居
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_tr, y_tr)
print("未标准化 KNN 准确率：", round(knn.score(X_val, y_val), 4))
# 约 0.667
```

不标准化时，3 个邻居的准确率只有 66.7% 左右。我们接着把三种缩放方式和 k 的取值一起扫一遍：

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

# 比较三种缩放方式，并用 10 折交叉验证挑 k
best = {"acc": 0, "k": None, "scaler": None}
for name, scaler in [("原始", None),
                     ("标准化", StandardScaler()),
                     ("最小最大", MinMaxScaler()),
                     ("归一化", Normalizer())]:
    for k in range(1, 20):
        model = KNeighborsClassifier(n_neighbors=k)
        if scaler is not None:
            # 把缩放和 KNN 包成流水线，交叉验证时不会泄漏
            model = make_pipeline(scaler, model)
        acc = cross_val_score(model, train, y, cv=10, scoring="accuracy").mean()
        if acc > best["acc"]:
            best = {"acc": acc, "k": k, "scaler": name}

print(best)
# 约 {'acc': 0.7667, 'k': 11, 'scaler': '标准化'}
```

标准化之后，11 个邻居的交叉验证准确率提高到了 76.7% 左右。这个对比很直观地说明了：**量纲不统一时，距离会被大数值的列主导，KNN 找出来的「近邻」其实是假近邻。**

不过 KNN 的上限不高。我们又把逻辑回归、随机梯度下降、决策树、随机森林、支持向量机放在同一份训练/验证划分上比了一遍：

```python
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 统一用标准化后的数据，保证比较公平
X_tr_s = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)

models = [
    LogisticRegression(random_state=0),
    SGDClassifier(random_state=0),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(random_state=0),
    SVC(random_state=0),
]

for m in models:
    m.fit(X_tr_s, y_tr)
    pred = m.predict(X_val_s)
    print(type(m).__name__, round(accuracy_score(y_val, pred) * 100, 2))
```

在 75%/25% 的划分上，我们得到的验证准确率分别是：逻辑回归 82.00%、SGD 78.67%、决策树 65.33%、随机森林 91.33%、SVM 81.33%。随机森林明显领先，决策树则因为太容易过拟合而垫底。

我们还试了一个简单的全连接神经网络，结构是 100 维输入、1024 个隐藏单元、1 个 sigmoid 输出：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# 一个很简单的全连接网络：100 维输入 -> 1024 个隐藏单元 -> 1 个概率
model = Sequential([
    Flatten(input_shape=(100,)),
    Dense(1024, activation="relu"),
    Dense(1, activation="sigmoid"),
])
model.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=0.01),
              metrics=["accuracy"])
model.fit(X_tr_s, y_tr, epochs=50,
          validation_data=(X_val_s, y_val), verbose=0)
print("神经网络验证准确率约 84.67%")
```

神经网络拿到 84.67%，比逻辑回归好，但没有超过随机森林。这也不奇怪：训练集只有 450 条（75% 划分），对神经网络来说太少了，参数一多就记样本而不是学规律。

真正的主线是围绕比赛指标 AUC 来做网格搜索。我们首先试了带 L1 正则的逻辑回归和岭分类器：

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# 逻辑回归：L1 正则会让一部分权重变成 0，相当于顺便做特征筛选
lr = LogisticRegression(solver="liblinear", penalty="l1", random_state=1)
lr_search = GridSearchCV(
    lr, {"C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10]},
    scoring="roc_auc", cv=3)   # 直接用比赛指标 AUC 调参
lr_search.fit(train, y)
print(lr_search.best_params_, round(lr_search.best_score_, 4))
# {'C': 0.5} 0.9313

# 岭分类器：L2 正则，和逻辑回归互补
ridge = RidgeClassifier(random_state=1)
ridge_search = GridSearchCV(
    ridge, {"alpha": [0.01, 0.05, 0.1, 0.5, 1, 2, 5]},
    scoring="roc_auc", cv=3)
ridge_search.fit(train, y)
print(ridge_search.best_params_, round(ridge_search.best_score_, 4))
# {'alpha': 5} 0.9032
```

这里的 `C` 和 `alpha` 都是控制正则化强弱的参数：值越小，模型越保守，越不容易过拟合；值越大，模型越敢去拟合训练数据。逻辑回归在 `C=0.5` 时交叉验证 AUC 是 0.9313，岭分类在 `alpha=5` 时是 0.9032。

接着我们试随机森林和 XGBoost。随机森林由很多棵决策树投票组成，单棵树容易过拟合，但很多棵树平均一下会稳很多；XGBoost 则是一棵接一棵地训练，后一棵专门纠正前一棵的错误。

```python
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# 随机森林：先限制树深，避免每棵树把训练集背下来
rf = RandomForestClassifier(max_depth=4, random_state=1)
rf_search = GridSearchCV(
    rf,
    {"n_estimators": list(range(20, 100, 5)),
     "min_weight_fraction_leaf": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]},
    scoring="roc_auc", cv=3)
rf_search.fit(train, y)
print(rf_search.best_params_, round(rf_search.best_score_, 4))
# {'min_weight_fraction_leaf': 0.001, 'n_estimators': 20} 0.9514

# XGBoost：一棵一棵地纠正前一棵树的错误
xgb_search = GridSearchCV(
    xgb.XGBClassifier(random_state=1, min_child_weight=3),
    {"max_depth": list(range(1, 7)),
     "learning_rate": [0.01, 0.05, 0.1, 0.5, 1, 1.5],
     "n_estimators": list(range(10, 100, 5))},
    scoring="roc_auc", cv=3)
xgb_search.fit(train, y)
print(xgb_search.best_params_, round(xgb_search.best_score_, 4))
# {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 60} 0.9483
```

随机森林在 `max_depth=4`、`n_estimators=20`、`min_weight_fraction_leaf=0.001` 时，交叉验证 AUC 达到 0.9514；XGBoost 最好的一组是 `learning_rate=0.1`、`max_depth=2`、`n_estimators=60`，AUC 为 0.9483。两者很接近，随机森林略胜一点，所以我们最终选随机森林作为主模型。

为什么把树深定在 4？我们单独做了一条「树深 vs AUC」的曲线：

```python
from sklearn.model_selection import validation_curve
import numpy as np

# 固定其他参数，只让树深从 1 变到 10
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=1), train, y,
    param_name="max_depth", param_range=range(1, 11),
    cv=3, scoring="roc_auc")

print("训练集 AUC：", np.round(train_scores.mean(axis=1), 3))
print("验证集 AUC：", np.round(val_scores.mean(axis=1), 3))
```

结果是：训练集 AUC 从 0.956 一路涨到 1.0，而验证集一直在 0.94 到 0.95 之间打转。树越深，训练集分数越漂亮，但验证集不再跟着涨，说明再深下去只是在背训练样本。树深 4 附近已经接近验证集的最好水平，所以我们把 `max_depth` 定在 4，牺牲一点训练集分数，换更好的泛化。

### 2.4 模型测试与指标报告

我们把几个候选模型的交叉验证 AUC 汇总一下：

| 模型 | 关键参数 | 交叉验证 AUC |
| --- | --- | --- |
| 逻辑回归（L1） | `C=0.5` | 0.9313 |
| 岭分类器 | `alpha=5` | 0.9032 |
| XGBoost | `learning_rate=0.1, max_depth=2, n_estimators=60` | 0.9483 |
| 随机森林 | `max_depth=4, n_estimators=20` | 0.9514 |

随机森林的 0.9514 是这里面最高的，所以它成为我们的最终模型。这个数字的含义是：随机抽一封垃圾邮件和一封正常邮件，模型有大约 95% 的概率把垃圾邮件排在前面。

除了 AUC，我们也看了准确率：随机森林在 75%/25% 划分上约 91.33%，在另一次 70%/30% 划分上约 92.22%。这两个数字不能直接和 AUC 比较，因为准确率只在「概率大于 0.5 就算垃圾邮件」这一个阈值上算对错，而 AUC 看的是所有阈值下的整体排序能力。AUC 高，说明模型的分数排序靠谱；准确率则更依赖阈值选得合不合适。

作为对照，KNN 从 66.7% 提升到 76.7%，说明标准化确实有用；神经网络 84.67%，说明在小样本上它并没有想象中那么强。这也提醒我们：**模型不是越复杂越好，数据量小的时候，随机森林这种「一群树投票」的方法往往更稳。**

确认模型可用后，我们用全部 600 条训练数据重新训练一次，然后对 4000 条测试数据输出垃圾邮件的概率：

```python
# 用全部 600 条训练数据重新训练，数据多一点总没坏处
best_rf = rf_search.best_estimator_
best_rf.fit(train, y)

# 比赛按 AUC 排序，输出「是垃圾邮件的概率」比硬输出 0/1 更细腻
proba = best_rf.predict_proba(test)[:, 1]
print(proba.shape)  # (4000,)

# 比赛说明不需要表头，所以 header=False
submission = pd.DataFrame(proba)
submission.to_csv("submission.csv", index=False, header=False)
```

最后我们检查了提交文件：一共 4000 行，每行一个 0 到 1 之间的小数，没有表头，和比赛要求对得上。材料里没有给出排行榜分数，所以我们只报告本地的交叉验证结果，不编造线上成绩。

---

## 三、总结

本文中我们完成了一场垃圾邮件检测比赛的完整流程。我们首先用 EDA 摸清了数据：训练集 600 封、测试集 4000 封，每封邮件有 100 个匿名数值特征，训练集有 1745 个缺失值，测试集有 11882 个缺失值，而且不同列的量纲相差很大。其次我们做了数据清洗：读取时用 `header=None` 保住全部样本，用中位数填补缺失值，把标签转成一维数组，并且只在训练集上拟合标准化参数，避免测试集信息泄漏。接着我们从 KNN 入手，发现标准化能把 KNN 的准确率从 66.7% 提到 76.7%；然后把逻辑回归、SGD、决策树、随机森林、SVM 和一个小神经网络放在一起比较，随机森林的验证准确率约 91.33%，已经明显领先。然后我们围绕比赛指标 AUC 做网格搜索，逻辑回归、岭分类、XGBoost 和随机森林的交叉验证 AUC 分别是 0.9313、0.9032、0.9483 和 0.9514，最终选中随机森林，并用验证曲线把树深定在 4，防止它把训练集背下来。最后我们用全部训练数据重新拟合，对测试集输出垃圾邮件概率，生成了 4000 行的提交文件。

这场比赛给我们最大的启发是：**评价指标会反过来影响你怎么做模型、怎么输出结果。** 如果只看准确率，我们可能直接把 0/1 标签交上去；但比赛比的是 AUC，输出概率才能把「有多像垃圾邮件」的细微差别保留下来。另一个启发是，面对匿名特征和小样本，花哨的模型不一定赢，老老实实做缺失值处理、标准化、交叉验证和防过拟合，反而更可靠。
