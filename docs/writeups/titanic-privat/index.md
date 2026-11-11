# Titanic Privat

## 一、比赛简介

### 1.1 比赛背景

Titanic Privat 是一个以泰坦尼克号沉船为主题的入门分类比赛，数据被放在 `titanic-privat` 目录下，训练文件是 `train.csv`，测试文件是 `test.csv`。和公开的 Kaggle Titanic 比赛一样，它要求参赛者根据乘客的舱位、性别、年龄、同行亲属数量、票价、登船港口等信息，预测每位乘客是否幸存。

材料中的笔记本来自瑞士一所高校 Data Science 课程的练习。作者是德语区的学生小组，作业要求是「建立一个统计模型，在留出测试集上尽可能准确」，并且明确要求使用决策树。材料里的第二份笔记本则进一步尝试了随机森林，并把第一次提交的成绩 0.78742 作为起点，讨论「分类器选对了吗、超参数选对了吗」两个问题。这种从课程作业出发、逐步改进的叙述，很适合作为入门者的参考。

数据字段包括 `pclass`（舱位等级）、`sex`（性别）、`sibsp`（同船兄弟姐妹或配偶数）、`parch`（同船父母或子女数）、`embarked`（登船港口）、`fare`（票价）、`age`（年龄），以及被删掉的 `name`、`ticket`、`cabin`。目标列是 `survived`，0 表示遇难，1 表示幸存。数据以 `Id` 作为索引列，训练集 891 行、10 个特征列，测试集 418 行。

### 1.2 比赛目标

输入是一位乘客的特征，输出是 0 或 1 的幸存预测。提交文件只有一列 `survived`，索引用测试集的 `Id`。因为课程作业强调「留出测试集」，所以材料中先用 `train_test_split` 从训练集切出 70% 训练、30% 验证，用验证集判断模型好坏，最后再在完整训练集上重训并预测测试集。

这个比赛的特殊之处在于数据规模很小。891 行训练数据，切掉 30% 验证后只剩六百多行。小数据意味着模型很容易过拟合：一棵不加限制的决策树可以把训练集准确率做到 97% 以上，但在新数据上可能只有 78% 左右。因此整个流程的重点不是「把训练集拟合得多好」，而是「如何诚实地估计泛化能力」。

### 1.3 评价标准

比赛使用准确率（Accuracy）作为评价指标，也就是预测正确的比例，越高越好。材料里第一次提交的公开成绩是 0.78742，可以理解为在测试集上约 78.7% 的乘客被预测正确。

除了准确率，材料还用交叉验证分数来选超参数。交叉验证的思路是：把训练集切成 k 份，每次拿 k-1 份训练、剩下 1 份验证，重复 k 次，最后把 k 个分数平均。材料里用了 `cv=10`，也就是十折交叉验证。这样做比单次切分更稳定，因为它把「运气好切到容易的验证集」的影响平均掉了。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把数据读进来。训练集用 `Id` 作为索引，这样后面的特征表天然带索引，提交时不用再手动对齐。然后我们把 `survived` 单独拿出来当标签，剩下的列作为特征。

```python
import pandas as pd

def import_data():
    # 用 Id 作为索引，提交时可以直接复用
    dftrain = pd.read_csv('/kaggle/input/titanic-privat/train.csv',
                          sep=',', index_col='Id')
    Xtest = pd.read_csv('/kaggle/input/titanic-privat/test.csv',
                        sep=',', index_col='Id')

    Xtrain = dftrain.drop('survived', axis=1)   # 特征
    ytrain = dftrain.survived                    # 标签
    return Xtrain, ytrain, Xtest
```

读完后我们看形状：训练特征 891 行、10 列，测试集 418 行、9 列（少一列标签）。接着看每列的非空数量，发现 `age` 缺失较多，`cabin` 缺失更多，`embarked` 只缺两行。`name` 和 `ticket` 是自由文本，直接放进决策树没有意义。

```python
Xtrain, ytrain, Xtest = import_data()
print(Xtrain.shape)        # (891, 10)
print(Xtrain.isna().sum())
print(Xtest.shape)         # (418, 9)
```

从业务常识看，`sex` 应该是最强的特征，因为救生艇优先让妇女和儿童上；`pclass` 也很重要，头等舱乘客离甲板更近、更有机会获救；`fare` 与 `pclass` 高度相关，可以作为舱位的补充。`sibsp` 和 `parch` 反映家庭规模，太小或太大都可能降低逃生概率。`embarked` 则可能反映登船位置和舱位分布。

### 2.2 数据清洗

清洗分三步：删列、填补缺失、编码类别。

删列方面，`cabin` 缺失太多，`ticket` 格式混乱，`name` 是自由文本，三者都不适合直接喂给决策树，所以直接删除。注意训练集和测试集要删同样的列，保持特征顺序一致。

```python
# 这些列信息密度低或格式不规则，先删掉
for df in (Xtrain, Xtest):
    del df['cabin']
    del df['ticket']
    del df['name']
```

填补缺失方面，我们按列的类型分开处理。数值列 `fare`、`age` 用中位数填补，因为中位数对极端值不敏感；类别列 `pclass`、`sex`、`sibsp`、`parch`、`embarked` 用众数填补。关键点是：`SimpleImputer` 只在训练集上 `fit`，然后对测试集只 `transform`。如果对测试集重新 `fit`，就会把测试集的分布信息带进模型。

```python
from sklearn.impute import SimpleImputer

numerische_features = ['fare', 'age']
qualitative_features = ['pclass', 'sex', 'sibsp', 'parch', 'embarked']

imp_n = SimpleImputer(strategy='median')       # 数值列用中位数
imp_q = SimpleImputer(strategy='most_frequent') # 类别列用众数

imp_n.fit(Xtrain[numerische_features])         # 只在训练集上学习
imp_q.fit(Xtrain[qualitative_features])

Xtrain[numerische_features] = imp_n.transform(Xtrain[numerische_features])
Xtest[numerische_features] = imp_n.transform(Xtest[numerische_features])

Xtrain[qualitative_features] = imp_q.transform(Xtrain[qualitative_features])
Xtest[qualitative_features] = imp_q.transform(Xtest[qualitative_features])
```

编码方面，决策树只能处理数值，所以要把 `sex` 和 `embarked` 转成整数。这里同样要用训练集 `fit`、测试集 `transform`，保证 `female` 在两边的编号一致。

```python
from sklearn.preprocessing import LabelEncoder

le_sex = LabelEncoder()
Xtrain['sex'] = le_sex.fit_transform(Xtrain['sex'])
Xtest['sex'] = le_sex.transform(Xtest['sex'])

le_emb = LabelEncoder()
Xtrain['embarked'] = le_emb.fit_transform(Xtrain['embarked'])
Xtest['embarked'] = le_emb.transform(Xtest['embarked'])
```

材料里还做了一个标准化实验：`X_std = (Xtrain - Xtrain.mean()) / Xtrain.std()`，并检查均值是否接近 0、标准差是否接近 1。对决策树来说标准化不是必须的，因为树只关心特征的排序和分裂点，不关心绝对尺度；但对后续的线性模型或神经网络会有帮助。

### 2.3 模型训练与超参数调优

我们先用一个不加限制的决策树看训练集能拟合到什么程度。`max_depth=50` 基本等于让树长到不能再长。

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=50)
clf.fit(Xtrain, ytrain)
print(clf.score(Xtrain, ytrain))   # 训练集准确率 0.9775
```

训练集 97.75% 看起来很高，但这不是好消息。一棵树如果对每个训练样本都单独长出一条路径，它等于把答案背下来了，换一批数据就会露馅。材料里紧接着用随机森林做对比：100 棵树、每棵最大深度 5。

```python
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(
    n_estimators=100,   # 100 棵决策树
    max_depth=5,        # 每棵树最多 5 层，限制复杂度
    random_state=1,
)
random_forest.fit(Xtrain, ytrain)
print(random_forest.score(Xtrain, ytrain))   # 训练集准确率 0.8484
```

随机森林的训练准确率反而更低（84.84%），但这不代表它更差。它通过「每棵树只看部分样本和部分特征、最后投票」的方式降低了方差，训练集分数下降换来的是验证集和测试集上的稳定。这正是集成学习的基本思想：多个弱模型一起投票，比一个过拟合的强模型更可靠。

为了选树的深度，我们做十折交叉验证。对每个候选深度，训练一次、评估十次，打印平均分和标准差。

```python
from sklearn.model_selection import cross_val_score
import numpy as np

for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 200]:
    clf = DecisionTreeClassifier(max_depth=m)
    scores = cross_val_score(clf, Xtrain, ytrain, cv=10)
    print(f'Tiefe {m}: {np.mean(scores):1.3f} +/- {np.std(scores):1.3f}')
```

这段输出会告诉我们两件事。第一，深度从 1 增加到 3、4 时，交叉验证分数明显上升，说明模型在学真实规律。第二，深度继续增加后，训练分数继续上升，但交叉验证分数趋于平缓甚至下降，说明开始过拟合。我们选那个「交叉验证平均分最高、标准差又不太大」的深度，而不是训练分数最高的深度。

材料还画了验证曲线：横轴是 `max_depth`，纵轴是分数，一条线是训练分数，一条是交叉验证分数，阴影是标准差。理想图形是两条线先一起上升，然后训练线继续上升、验证线走平，两条线的分叉点就是复杂度合适的区域。

```python
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

param_range = np.logspace(0, 3, 50)   # 1 到 1000，对数刻度
train_scores, valid_scores = validation_curve(
    DecisionTreeClassifier(), Xtrain, ytrain,
    param_name='max_depth', param_range=param_range,
    cv=10, scoring='accuracy')

plt.semilogx(param_range, train_scores.mean(axis=1), label='Training score')
plt.semilogx(param_range, valid_scores.mean(axis=1), label='Cross-validation score')
plt.xlabel('max_depth')
plt.ylabel('Score')
plt.legend()
plt.show()
```

最后，材料还用 `tree.export_graphviz` 把一棵浅树画出来，节点上写着分裂条件和类别，方便解释「为什么这个人被预测为幸存」。可解释性是决策树相对神经网络的最大优势：你可以顺着一条路径给非技术人员讲清楚判断依据。

### 2.4 模型测试与指标报告

调完参后，我们在完整训练集上重新训练，再对测试集预测。预测结果是 0/1 的一维数组，长度 418。提交文件的索引是 `Id`，列名是 `survived`。

```python
clf = DecisionTreeClassifier(max_depth=5, criterion='entropy')
clf.fit(Xtrain, ytrain)

yhat = clf.predict(Xtest).astype('int')
print(yhat.shape)      # (418,)
print(yhat.dtype)      # int

submission = pd.Series(yhat, name='survived')
submission.index.name = 'Id'
submission.to_csv('submission.csv')
```

材料里第一次提交的成绩是 0.78742。这个分数在泰坦尼克比赛中属于「流程正确但没有精雕细琢」的水平：它说明缺失值处理、编码、模型选择都没有大错，但也说明特征工程还有空间。作者随后提出的两个问题很关键：分类器选对了吗？超参数选对了吗？

我们对比了两种模型的表现。深树在训练集上 97.75%，但这不是泛化能力；随机森林训练集 84.84%，交叉验证更稳定。测试集上两者差距没有训练集那么夸张，说明深树的过拟合主要发生在训练集内部。实际选择时，我们更倾向于交叉验证分数高、方差小的配置，而不是训练分数最高的配置。

材料还检查了预测结果的分布。如果模型把所有乘客都预测成 0，准确率也能有六成左右，但这种模型没有实际价值。所以我们打印 `value_counts()`，确认 0 和 1 都有出现，并且比例大致合理。这一步在类别不平衡的数据上尤其重要。

```python
print(pd.Series(yhat).value_counts())
```

## 三、总结

本文中我们完成了 Titanic Privat 的入门分类流程。我们首先用 `Id` 作为索引读入 `train.csv` 和 `test.csv`，确认训练集 891 行、测试集 418 行，并识别出 `age`、`cabin` 的缺失问题；其次删除 `name`、`ticket`、`cabin` 三个不规则列，对数值列用中位数、类别列用众数填补，所有 imputer 和 encoder 都只在训练集上 `fit`；接着先用深树观察过拟合，再用 100 棵、深度 5 的随机森林降低方差，并用十折交叉验证和验证曲线选择树的深度；最后在完整训练集上重训，输出 418 行提交文件，并参考第一次提交 0.78742 的成绩讨论改进方向。

这个练习最重要的收获是：训练集准确率高不等于模型好。深树 97.75% 的训练准确率是「背答案」，随机森林 84.84% 反而更可信。交叉验证、验证曲线、预测分布检查，这些工具的作用就是帮我们区分「真的学会了」和「只是记住了」。另一个收获是流程纪律：所有填补和编码都只能在训练集上拟合，否则验证分数会被高估，到了真实测试集就会原形毕露。
