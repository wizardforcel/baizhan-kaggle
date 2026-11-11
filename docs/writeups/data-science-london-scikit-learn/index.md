# Data Science London + Scikit-learn：入门级二分类（Binary Classification）

## 一、比赛简介

### 1.1 比赛背景

Data Science London 是伦敦的一个数据科学社区。2013 年前后，他们在 Kaggle 上办了一场**入门级**比赛，目的很朴素：让刚开始学机器学习的人，用最常用的 Python 库 **scikit-learn** 走一遍完整的分类流程。

为了让大家把注意力放在「怎么建模」而不是「怎么理解业务」上，比赛特意没有给真实的业务数据，而是给了一份**匿名的数值特征表**：每一行是一个样本，每一列是一个已经算好的特征，而特征代表什么、样本是什么东西，比赛都没有说。标签只有 0 和 1 两类。这有点像考试卷上只写着「甲类」和「乙类」，你不需要知道它们具体是什么，只需要把两类分开。

也正因为数据被匿名化，这个比赛最大的特点就是：**没法靠业务知识做特征工程**。你只能从数据的数值分布本身去找规律。这一点决定了后面整篇 writeup 的主角——不是复杂的神经网络，而是「数据长什么样」这件事。

### 1.2 比赛目标

比赛提供三个文件：

- `train.csv`：1000 行 × 40 列，训练样本的特征；
- `trainLabels.csv`：1000 行 × 1 列，每个训练样本的类别标签（0 或 1）；
- `test.csv`：9000 行 × 40 列，测试样本的特征。

也就是说，这是一道标准的**二分类**题：用 1000 条有标签的样本训练一个模型，去预测 9000 条没有标签的样本属于 0 还是 1。训练集只有 1000 条，测试集却有 9000 条，这个 1:9 的比例提醒我们：模型必须**泛化得好**，光在训练集上把分刷到 1.0 没有任何意义。

### 1.3 评价标准

评价指标是**分类准确率（accuracy）**：

```
accuracy = 预测正确的样本数 / 总样本数
```

分数范围 0～1，越高越好。这是一个非常直白的指标：预测对多少个，就得分多少。需要注意两点：

- 因为两类样本几乎各占一半（训练集里 0 类约 51%、1 类约 49%），所以「闭着眼睛全猜 0」也有大约 **0.51** 的准确率。这就是我们必须超过的基线。
- 准确率对「每类都一样重要」这件事是公平的，不会因为某一类少就给它加权，所以不用像处理不平衡数据那样特别设计损失函数。

比赛在 Kaggle 上以「Public / Private Leaderboard」的形式排名，榜上靠前的方案准确率普遍在 **0.98 以上**，而普通调参的 SVM、随机森林大概在 0.90 左右。中间这段差距，就是这篇 writeup 要讲的核心故事。

---

## 二、解题过程

我们的整体路线是：**先用几个常见分类器摸清基线水平，再通过观察特征的分布发现「数据是由若干个高斯分布混合而成」这一关键线索，最后用高斯混合模型（GMM）把原始特征换成「属于各个高斯成分的概率」，再在这组新特征上做网格搜索。** 下面分四步走。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，确认形状和标签分布：

```python
import pandas as pd
import numpy as np

# header=None：文件第一行就是数据，没有列名
train = pd.read_csv('../input/train.csv', header=None)
train_label = pd.read_csv('../input/trainLabels.csv', header=None)
test = pd.read_csv('../input/test.csv', header=None)

# 训练 (1000, 40)、标签 (1000, 1)、测试 (9000, 40)
print(train.shape, train_label.shape, test.shape)

# 标签只有 0 和 1，两类几乎各占一半
print(train_label[0].value_counts(normalize=True))
# 0    0.51
# 1    0.49
```

这一步告诉我们两件事。第一，**没有缺失值**——40 列全部是完整的 1000 个数，不需要做填补。第二，**类别基本平衡**，所以准确率这个指标不会被某一类主导，也不需要做重采样。

接下来我们看每一列的统计量，尤其是标准差：

```python
# 看 40 个特征的均值、标准差、最小最大值
desc = train.describe()
print(desc.loc[['mean', 'std', 'min', 'max']].T.head(12))
```

结果很有意思：**大部分列的均值都在 0 附近、标准差都在 1 左右**，看起来像是做过标准化的；但有几列的标准差明显偏大——材料里看到的有 **4.54、2.12、2.23、2.24、2.12、2.05**，最大值甚至到了 17.6、7.6、7.1。这说明各列的**尺度不齐**：如果直接用基于距离的模型（比如 KNN、SVM），尺度大的列会喧宾夺主，把距离计算带偏。

再往下，我们把某一列画成直方图，并叠上一条正态分布曲线：

```python
import matplotlib.pyplot as plt
from scipy.stats import norm

# 随便挑一列（sample(axis=1) 随机取一列）
random_feature = train.sample(axis=1)
min_, max_ = random_feature.min(), random_feature.max()
mean, std = random_feature.mean(), random_feature.std()

# 用该列的均值和标准差画一条理论正态曲线
fitted_gaussian = norm.pdf(np.linspace(min_, max_, 1000), loc=mean, scale=std)

plt.figure(figsize=(8, 6))
plt.hist(random_feature, bins=100, density=True, label='Feature data')
plt.plot(np.linspace(min_, max_, 1000), fitted_gaussian, lw=5, label='Gaussian fit')
plt.legend()
plt.show()
```

这一步是整个比赛的转折点。我们发现：**每一列的形状都像高斯分布**——中间高、两边低、近似对称，跟理论曲线贴合得相当好。如果只是「一列一个高斯」，那它本身没什么可做的；但联想到标签是两类，而两类各占一半，一个很自然的猜想就出现了：**这份数据可能是由若干个高斯分布混合生成的，每一类数据对应其中几个高斯成分。** 这个猜想后面被证明是这场比赛最大的金矿。

### 2.2 数据清洗

严格来说这份数据并不「脏」：没有缺失值、没有重复行、标签也没有异常值。所以这里的「清洗」主要是**格式整理和尺度处理**。

第一个小问题是标签的形状：`trainLabels.csv` 读进来是 `(1000, 1)` 的 DataFrame，而 scikit-learn 的分类器要求标签是一维数组。我们把它拉平：

```python
# to_numpy().ravel()：DataFrame → numpy 数组 → 压成一维
y = train_label.to_numpy().ravel()
print(y.shape)   # (1000,)
```

第二个问题是尺度。我们对比了两种做法。一种是**标准化（StandardScaler）**，把每列减去均值、除以标准差，变成均值 0、方差 1；另一种是**归一化（Normalizer）**，它不改变每列自身的尺度，而是把每一**行**（每个样本的 40 维向量）缩放到单位长度。这两种做法的效果差别很大，后面我们在验证时会看到：

```python
from sklearn.preprocessing import StandardScaler, Normalizer

# 标准化：按列减均值除标准差
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train)

# 注意：测试集要用训练集拟合出来的 scaler 去 transform，
# 而不是自己 fit，否则训练和测试的尺度基准就不一致了
X_test_scaled = scaler.transform(test)

# 归一化：按行把每个样本缩放到单位范数
X_normalized = Normalizer().fit_transform(train)
```

除了尺度，我们还试过 **PCA 降维**。40 维特征降到 12 维之后，信息损失不大，但训练会快一些，而且能去掉一些噪声方向：

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=12)
X_pca = pca.fit_transform(X_scaled)
print('保留的方差比例:', pca.explained_variance_ratio_.sum())
```

不过后面的实验表明，PCA 主要起到「稳定模型」的作用，真正把分数从 0.90 抬到 0.99 的，是下面的 GMM 步骤。

### 2.3 模型训练与超参数调优

**第一步：建立基线。** 我们先用最常用的几个分类器在同一份划分上比一比，心里有个底：

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# 70% 训练、30% 留出验证
x_tr, x_va, y_tr, y_va = train_test_split(train, y, test_size=0.3, random_state=101)

models = {
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(gamma='auto'),
    'XGBoost': XGBClassifier(),
}
for name, model in models.items():
    model.fit(x_tr, y_tr)
    print(name, accuracy_score(y_va, model.predict(x_va)))
```

在这份 70/30 的划分上，结果是 **KNN 0.917、SVM 0.903、XGBoost 0.877、随机森林 0.867、决策树 0.75**。在另一份 75/25 的划分上，数字略有不同（SVM 0.908、KNN 0.884、决策树 0.764），但结论一致：**SVM 和 KNN 是这批模型里最好的两个，决策树最弱。** 这个量级（0.90 左右）就是「常规做法的天花板」。

**第二步：试尺度和降维。** 我们用 10 折交叉验证，把「原始特征 / 归一化 / PCA」三种预处理和三个分类器交叉组合：

```python
from sklearn.model_selection import cross_val_score

for X_name, X in [('原始', train), ('归一化', X_normalized), ('PCA12', X_pca)]:
    for name, model in [('SVC', SVC()), ('DT', DecisionTreeClassifier()),
                        ('KNN', KNeighborsClassifier())]:
        # cv=10：把训练集切成 10 份，轮流拿 1 份做验证
        scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
        print(X_name, name, round(scores.mean(), 3))
```

结果里有一个值得注意的现象：归一化之后，SVC 只有 0.808、决策树 0.822，反而比原始特征还差；而 PCA(12) 之后，SVC 回到 0.905、KNN 0.908。这说明**对这份数据，简单的行归一化会破坏有用的信息**，而 PCA 是更稳妥的降维方式。但不管怎么组合，分数都在 0.90 上下打转。

**第三步：GMM 特征工程（关键一步）。** 回到 EDA 里的那个发现：数据像是由多个高斯分布混合而成的。既然如此，与其直接把 40 个原始数值喂给分类器，不如问一个更有信息量的问题：**这个样本更可能属于哪个高斯成分？属于每个成分的概率分别是多少？**

我们就在**训练集 + 测试集合并后的 10000 行**上拟合一个高斯混合模型。这一步是**无监督**的——不使用任何标签，只是让 GMM 去描述这 10000 个点整体是由哪几个高斯团块组成的。成分个数和协方差类型都用 AIC 自动挑：

```python
from sklearn.mixture import GaussianMixture

# 训练集和测试集拼在一起，共 10000 行、40 列
x_all = np.r_[train, test]
print(x_all.shape)   # (10000, 40)

lowest_aic = np.inf
best_gmm = None

# 成分数 1~6，协方差类型 4 种，穷举后取 AIC 最小的
for cov_type in ['spherical', 'tied', 'diag', 'full']:
    for n_components in range(1, 7):
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type=cov_type,
                              random_state=0)
        gmm.fit(x_all)
        aic = gmm.aic(x_all)      # AIC 越小，模型对数据的解释越好且不过复杂
        if aic < lowest_aic:
            lowest_aic = aic
            best_gmm = gmm

print(best_gmm.n_components, best_gmm.covariance_type)   # 4 full
```

AIC（赤池信息量准则）是一种「既要拟合得好、又不喜欢太复杂」的打分方式，数值越小越好。穷举的结果是 **4 个成分、full 协方差**。这正好呼应了「两类标签、每类可能对应两个高斯团块」的猜想。

有了这个 GMM，我们把每个样本的 40 维原始特征，替换成它属于这 4 个成分的**后验概率**，也就是 4 维的新特征：

```python
# predict_proba 返回每个样本属于各成分的概率，形状 (n_samples, 4)
gmm_train = best_gmm.predict_proba(train)
gmm_test = best_gmm.predict_proba(test)

print(gmm_train.shape, gmm_test.shape)   # (1000, 4) (9000, 4)

# 新特征也做一次标准化，让各列尺度一致
scaler = StandardScaler()
gmm_train = scaler.fit_transform(gmm_train)
gmm_test = scaler.transform(gmm_test)
```

这一步的妙处在于：40 个「含义未知、尺度各异」的原始数值，被压缩成了 4 个「含义明确、都在 0～1 之间」的概率。分类器不再需要自己从 40 维里艰难地摸索边界，只需要在 4 维概率空间里画一条分界线——而这正好是 GMM 最擅长表达的形状。

**第四步：在新特征上做网格搜索。** 我们在 4 维的 `gmm_train` 上重新跑一遍常见分类器，并用 `GridSearchCV` 找超参数：

```python
from sklearn.model_selection import GridSearchCV

# SVM：试不同的惩罚系数 C 和核函数
svm_grid = GridSearchCV(
    SVC(),
    {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']},
    scoring='accuracy', cv=5, n_jobs=-1)
svm_grid.fit(gmm_train, y)
print('SVM', svm_grid.best_params_, svm_grid.best_score_)   # {'C': 1, 'kernel': 'linear'} 0.996

# KNN：试邻居数
knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    {'n_neighbors': range(1, 11)},
    scoring='accuracy', cv=5, n_jobs=-1)
knn_grid.fit(gmm_train, y)
print('KNN', knn_grid.best_params_, knn_grid.best_score_)    # {'n_neighbors': 3} 0.996

# 随机森林：试树的棵数和最大深度
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=0),
    {'n_estimators': [10, 50, 100, 200, 400], 'max_depth': [3, 10, 20, 40]},
    scoring='accuracy', cv=5, n_jobs=-1)
rf_grid.fit(gmm_train, y)
print('RF', rf_grid.best_params_, rf_grid.best_score_)       # {'max_depth': 3, 'n_estimators': 10} 0.996
```

三个模型的交叉验证分数**全部达到 0.996**，比之前的 0.90 一下跳了将近 10 个百分点。而且最优参数都很「朴素」：SVM 用线性核、C=1；KNN 只用 3 个邻居；随机森林只要 10 棵树、深度 3。这反过来印证了一件事——**不是模型变强了，是特征变对了**。当特征本身已经把两类分得很开时，简单的模型就足够了；反之，再复杂的模型在 40 维的原始空间里也只能做到 0.90。

### 2.4 模型测试与指标报告

因为三个模型分数一样，我们选了随机森林来做最终提交（它对参数不敏感、也不容易过拟合），用网格搜索得到的最优参数重新在全部 1000 条训练样本上训练：

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 用最优参数在全部训练数据上重新拟合
forest = RandomForestClassifier(random_state=0, max_depth=3, n_estimators=10)
forest.fit(gmm_train, y)

# 预测 9000 条测试样本
pred = forest.predict(gmm_test)

# 提交格式：Id 从 1 数到 9000，Solution 是预测类别
submission = pd.DataFrame({'Id': np.arange(1, len(pred) + 1),
                           'Solution': pred})
submission.to_csv('submission.csv', index=False)
```

这里有一个容易出错的地方：**测试集必须先经过和训练集完全相同的 GMM 变换**。也就是说，测试集不能自己再拟合一个 GMM，必须用训练时那个 `best_gmm` 去 `predict_proba`，再用同一个 `scaler` 做标准化。否则训练和测试的特征空间会对不上，模型看到的就完全是另一组数了。这正是我们前面把 GMM 拟合在「训练+测试合并的 10000 行」上的原因——它让两个数据集天然处在同一个概率空间里。

从指标上看，交叉验证准确率 **0.996** 意味着在 1000 条训练样本上，只有大约 4 条会被分错；而在 Kaggle 排行榜上，「随机森林 + GMM 特征」这一类做法能够稳定进入前列（材料里那篇题为 *RandomForest, top 4%* 的 notebook 用的正是这个思路）。作为对比，不做 GMM、只用原始特征加调参的方案最好的结果也只有 0.90 出头。

值得强调的一点是：**GMM 在拟合时完全没有用标签。** 它只是根据 10000 个点的空间分布，找出「数据自然聚成几个团」。这个信息是从特征本身提取的，跟答案无关，所以不存在「在测试集上偷看标签」的问题——它属于一种合法的**无监督特征工程**，在 Kaggle 中叫做「利用测试集分布做变换」，是被规则允许的。

---

## 三、总结

本文中我们完成了 Data Science London + Scikit-learn 这场入门二分类比赛的完整流程。我们首先用 EDA 摸清了数据：训练集 1000 行 × 40 列、测试集 9000 行 × 40 列、标签 0/1 各占约一半，没有缺失值，但各列尺度不齐（标准差从 1 到 4.5 不等），而且每一列的形状都接近高斯分布。其次我们做了数据清洗和尺度处理：把标签拉平成一维，比较了标准化、行归一化和 PCA 降维三种做法，发现简单的行归一化反而会伤害模型。接着我们建立了基线，KNN、SVM、随机森林、XGBoost、决策树在 70/30 划分上的准确率都在 0.75～0.92 之间，这就是常规做法的天花板。然后我们抓住「数据像多个高斯分布的混合」这一线索，在训练集与测试集合并的 10000 行上无监督地拟合高斯混合模型，用 AIC 在成分数 1～6 与四种协方差类型中选出最优的 4 成分模型，并把每个样本的 40 维原始特征替换成它属于各成分的后验概率。最后我们在这 4 维新特征上做网格搜索，SVM、KNN、随机森林的交叉验证准确率全部达到 **0.996**，并用最优参数的随机森林生成 `Id,Solution` 格式的提交文件。

这个比赛给我们最大的启发是：**特征的质量往往比模型的复杂度重要得多。** 40 个匿名数值丢给任何分类器都只能做到 0.90 左右，而把它们换成 4 个高斯成分的后验概率之后，连一棵深度 3 的小树都能做到 0.996。原因是原始特征把「生成数据的机制」藏了起来，而 GMM 把这层机制显式地还原了出来。另一个启发是：**无监督方法可以用测试集的数据分布**。GMM 不需要标签，所以在全部 10000 个样本上拟合既合法又有效，它让训练和测试共享同一个特征空间，避免了分布漂移。对入门者来说，这也是一个很好的示范——不需要深度学习，不需要调几十个超参数，只要认真看数据的形状，就能把分数提高一大截。
