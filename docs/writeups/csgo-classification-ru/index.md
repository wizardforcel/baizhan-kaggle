# CS:GO 回合胜负预测（CS:GO Classification RU）

## 一、比赛简介

### 1.1 比赛背景

CS:GO（Counter-Strike: Global Offensive，反恐精英：全球攻势）是一款非常流行的第一人称射击游戏。每一局比赛里，玩家分成两个阵营：

- **T（Terrorists，恐怖分子）**：负责安放炸弹；
- **CT（Counter-Terrorists，反恐精英）**：负责阻止安放或者拆除炸弹。

每一「回合」（round）结束时，总有一方获胜。比赛想做的事情，就是**只看一回合进行过程中的各项统计数据，提前判断哪一方会赢**。

每一行记录的是某一局对战在某个时刻的统计数据，包含每个选手的击杀数、伤害量、经济状况、存活情况等等，再加上这一局打的是哪张地图。因为地图不同，攻守双方的天然优势也不同——有些图 CT 好守，有些图 T 好攻——所以地图名本身就是一个很重要的线索。

这个任务的难点在于：**特征多、量纲差异大、而且大量特征严重偏斜**。比如「击杀数」绝大多数时候是 0，偶尔出现一个 30，分布拖着一条长长的尾巴；不同特征之间的数值范围也差了几个数量级。不做处理直接喂给模型，效果会很差。

### 1.2 比赛目标

训练集和测试集的结构相同，每一行代表一次对局/回合的快照，主要包含：

- `id`：样本编号（测试集有，训练集也有，但不参与建模）；
- `map`：地图名称，是文字类别（比如 `de_dust2` 这类）；
- 大约 95 个数值特征：选手和队伍的各项统计量；
- `target`：训练集独有的标签，取值是字符串 `"CT"` 或 `"T"`。

我们要给测试集的每一行预测一个 `target`，填回 `sample_submission.csv` 的对应位置。

### 1.3 评价标准

这是一个二分类问题，比赛用**分类准确率（Accuracy）**衡量：预测对的样本数除以总样本数，越高越好。材料里的 notebook 也都围绕准确率做优化——一份用 PyCaret 的 `compare_models(sort='Accuracy')` 选模型，另一份在融合模型里直接用 `accuracy_score` 当优化目标，这从侧面印证了评分方式。

准确率的好处是直观：0.7 就代表「十局里猜对七局」。坏处是它只看「最终猜的那一类」，不看概率，所以做模型融合时我们要先把几个模型的概率平均起来，再取最大的那一类。

---

## 二、解题过程

我们沿着「先看懂分布，再修分布，最后用树模型融合」这条线走。材料里有三份 notebook，我们以讲得最完整、思路最清晰的融合模型为主线，另外两份在总结里简单带过。

### 2.1 数据探索（EDA）

我们首先读入数据，把标签从字符串转成 0/1，方便后面的模型处理：

```python
import pandas as pd
import numpy as np

train = pd.read_csv("/kaggle/input/csgo-classification-ru/train.csv")
test  = pd.read_csv("/kaggle/input/csgo-classification-ru/test.csv")
sub   = pd.read_csv("/kaggle/input/csgo-classification-ru/sample_submission.csv")

print("Shape train =", train.shape)
print("Shape test  =", test.shape)

test = test.drop("id", axis=1)      # id 只是编号，留着会误导模型

# 把 "CT" / "T" 变成 1 / 0：CT 赢记 1，T 赢记 0
ytrain = train["target"]
y = np.zeros(ytrain.shape[0])
for i in range(ytrain.shape[0]):
    y[i] = 1 if ytrain[i] == "CT" else 0
```

转成数字之后，我们就能用 `sns.displot(y)` 看一眼两类样本的比例。这一步很重要：如果 CT 和 T 的样本数差得离谱（比如 9:1），那模型只要无脑猜多数类就能拿到 0.9 的准确率，这种「假高分」必须先排除掉。

其次我们按数据类型把列分成两组。`select_dtypes` 会自动帮我们区分「文字列」和「数字列」，不用手写列名：

```python
import seaborn as sns

# object 类型就是文字列，这里只有 map 和 target
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
num_cols = train.select_dtypes(exclude=["object"]).columns.tolist()

print(cat_cols)    # ['map', 'target']
print(len(num_cols))   # 约 95 个数值特征

# 训练集和测试集拼在一起做独热编码，保证两边的地图列完全对齐
data = pd.concat([train, test], axis=0)
data = pd.get_dummies(data, columns=["map"])
```

`pd.get_dummies` 就是「独热编码」：把 `map` 这一列文字拆成很多列 0/1，比如 `map_de_dust2`、`map_de_mirage`，是这张图就填 1，否则填 0。训练集和测试集必须**一起**做，否则训练集有 `de_dust2` 列、测试集没有，特征就对不上了。

接着我们把所有数值特征的分布画出来。这一步是整场比赛最关键的分析：很多特征零点堆积、尾巴极长，直接喂模型等于让模型去拟合噪音。

```python
import matplotlib.pyplot as plt

tmp_train = data.loc[:, num_cols].dropna()

def reject_outliers2(data, m=4):
    # 只保留距离均值 4 个标准差以内的点，并且把 0 去掉
    # 去 0 是为了看清"非零部分"到底长什么样，不然图会被一根大柱子挡住
    data = data[abs(data - np.mean(data)) < m * np.std(data)]
    return data.loc[data != 0]

fig, axes = plt.subplots(19, 5, sharey=False)   # 19×5 = 95 个格子，一个特征一个
colors = sns.color_palette("hls", 95)
deal_list = []                                  # 记录"形状太怪"的特征

for ix, var in enumerate(tmp_train.columns):
    row, col = divmod(ix, 5)
    tmpVar = reject_outliers2(tmp_train[var])
    sk = round(tmpVar.skew(), 2)       # 偏度：分布往哪边歪
    ku = round(tmpVar.kurt(), 2)       # 峰度：尖不尖、尾巴厚不厚
    if abs(sk) > 2.5 or abs(ku) > 2.5:
        deal_list.append(var)          # 偏得太厉害，后面要单独做变换
    sns.distplot(tmpVar, kde=True, ax=axes[row, col], color=colors[ix])
    axes[row, col].set_title(var)
    axes[row, col].set_xlabel("skew:" + str(sk) + " kurt:" + str(ku))
plt.tight_layout()
```

偏度（skewness）衡量分布左右不对称的程度，峰度（kurtosis）衡量分布是不是尖峰厚尾。绝对值超过 2.5 就说明这个特征离正态分布很远。我们把这类特征记进 `deal_list`，清洗阶段重点处理它们——这就是「用数据决定清洗策略」，而不是所有列一律 log 一下了事。

### 2.2 数据清洗

我们做的第一件事，是把拼在一起的数据按 `target` 是否为空重新切开——训练集有标签、测试集没有，这是天然的切分点：

```python
# concat 之后，测试集的 target 是 NaN，正好用来把两边分开
train = data[data["target"].notna()].reset_index(drop=True)
test  = data[data["target"].isna()].reset_index(drop=True)
```

第二件事是补缺失值。测试集的部分数值特征存在空值，而六个模型对缺失值的处理方式并不统一，为了让融合时的输入完全一致，我们统一在建模前填掉。这里用**中位数**而不是均值：

```python
for col in num_cols:
    # 中位数对极端值不敏感：就算有几个 1000，也不会把填补值拉飞
    test[col] = test[col].fillna(test[col].median())
```

第三件事，是给 `deal_list` 里的偏斜特征挑一个「最不偏」的变换。候选有三个：`log1p`（对数）、`sqrt`（开方）、平方。判断标准是**变换后峰度的绝对值最小**——峰度越接近 0，分布越接近正态，模型越好学。

```python
for col in deal_list:
    logLot   = np.log1p(train[col]).kurt()      # log(1+x)：能压长尾，但要求 x >= 0
    rootLot  = np.sqrt(train[col]).kurt()       # 开方：压缩力度比 log 温和
    cubicLot = np.power(train[col], 2).kurt()   # 平方：反过来放大差异
    minVal = min([logLot, rootLot, cubicLot])

    # 谁的峰度绝对值最小就用谁，训练集和测试集必须用同一个变换
    if logLot == minVal:
        train[col] = np.log1p(train[col]);  test[col] = np.log1p(test[col])
    elif rootLot == minVal:
        train[col] = np.sqrt(train[col]);   test[col] = np.sqrt(test[col])
    elif cubicLot == minVal:
        train[col] = np.power(train[col], 2); test[col] = np.power(test[col], 2)
```

这种「按列自适应」的变换，比无脑对所有列做 log 要好。因为有的特征本来就接近对称，硬做 log 反而会把它弄歪。

最后清理无穷大和非数值。`np.log1p` 遇到负数会产生 NaN，除以 0 会产生 inf，这些值一旦进入模型就会报错或者算出垃圾：

```python
train = train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
test  = test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
```

最后我们把特征表转成 numpy 数组。树模型只认数值矩阵，而且后面用 `KFold` 给出的行号切片时，DataFrame 的 `[]` 是按列名取的，容易切错；转成数组后 `train[tr_ix]` 就是老老实实按行取：

```python
# 标签已经单独存在 y 里，这里把 target 列去掉
train = train.drop("target", axis=1).values
test  = test.drop("target", axis=1).values

print(train.shape, test.shape, y.shape)
```

### 2.3 模型训练与超参数调优

我们选了三类树模型：XGBoost、RandomForest、LightGBM。树模型的优点是**不需要特征缩放**，对偏斜和异常值也比较宽容，很适合这种表格数据。

调参我们用的是**贝叶斯优化**（Bayesian Optimization）。普通网格搜索是把所有组合都试一遍，参数一多就组合爆炸；贝叶斯优化的思路是：先随机试几组，根据结果建一个「哪里可能更好」的概率模型，下一组就挑最有希望的位置去试。它的两个关键参数是 `init_points`（先随机试几组）和 `n_iter`（之后按模型建议再试几组）。

优化的目标不是训练集准确率，而是 **5 折交叉验证的 ROC AUC**。AUC 衡量的是「模型给正例打的分是不是普遍高于负例」，它对阈值不敏感，比准确率更稳，不容易被某一折的偶然结果带偏：

```python
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
import numpy as np

def xgb_evaluate(n_estimators, max_depth, subsample, colsample_bytree, learning_rate):
    model = XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),          # 贝叶斯优化给的是小数，树深必须是整数
        subsample=subsample,               # 每棵树随机抽多少比例样本
        colsample_bytree=colsample_bytree, # 每棵树随机抽多少比例特征
        learning_rate=learning_rate,
        eval_metric="auc",
        n_jobs=-1,
    )
    scores = []
    kf = KFold(n_splits=5)                 # 5 折：轮流拿 1/5 当验证集
    for tr_ix, va_ix in kf.split(train):
        model.fit(train[tr_ix], y[tr_ix])
        pred = model.predict_proba(train[va_ix])[:, 1]   # 取"是 CT"的概率
        scores.append(roc_auc_score(y[va_ix], pred))
    return np.mean(scores)

pbounds = {"max_depth": (5, 15), "n_estimators": (100, 800),
           "subsample": (0.001, 1), "colsample_bytree": (0.001, 1),
           "learning_rate": (0.05, 0.35)}

xgb_bo = BayesianOptimization(f=xgb_evaluate, pbounds=pbounds, random_state=2, verbose=2)
xgb_bo.maximize(init_points=2, n_iter=3, acq="ei")   # 先试 2 组，再按建议试 3 组
params_xgb1 = xgb_bo.max["params"]
```

同样的套路我们复制给了 RandomForest 和 LightGBM，只是搜索空间不同，比如 LightGBM 多了一个 `num_leaves`（叶子数），深度范围收得更窄（3～7），学习率范围拉得更宽（0.0001～0.5）。LightGBM 用叶子生长策略，叶子数比深度更能控制复杂度，所以要单独调。

单个模型调好之后，我们做了一个**加权融合**：同时训练两个 XGBoost（一个调过参、一个默认参数）、两个随机森林、两个 LightGBM，六个模型各自输出「是 CT 的概率」，再按权重加权平均。权重本身也是一组超参数，同样交给贝叶斯优化，目标换成 5 折准确率。

```python
from sklearn.metrics import accuracy_score

def model_fusion(a, b, c, d, e, f):
    # 六个模型：调参版 + 默认版，成对出现，保证多样性
    xgb1 = XGBClassifier(**params_xgb1, n_jobs=-1)
    xgb2 = XGBClassifier(n_jobs=-1)
    rf1  = RandomForestClassifier(**params_rf, n_jobs=-1)
    rf2  = RandomForestClassifier(n_jobs=-1)
    lgb1 = LGBMClassifier(**params_lgb, n_jobs=-1)
    lgb2 = LGBMClassifier(n_jobs=-1)

    scores = []
    kf = KFold(n_splits=5)
    for tr_ix, va_ix in kf.split(train):
        X_tr, X_va = train[tr_ix], train[va_ix]
        Y_tr, Y_va = y[tr_ix], y[va_ix]
        for m in (xgb1, xgb2, rf1, rf2, lgb1, lgb2):
            m.fit(X_tr, Y_tr)

        # 六个概率加权平均，权重之和归一化；argmax 取概率大的那一类
        proba = (a * xgb1.predict_proba(X_va) + b * xgb2.predict_proba(X_va) +
                 c * rf1.predict_proba(X_va)  + d * rf2.predict_proba(X_va)  +
                 e * lgb1.predict_proba(X_va) + f * lgb2.predict_proba(X_va))
        pred = np.argmax(proba / (a + b + c + d + e + f), axis=1)
        scores.append(accuracy_score(Y_va, pred))
    return np.mean(scores)

pbounds = {k: (0.001, 1) for k in "abcdef"}   # 六个权重都在 0.001~1 之间搜
fusion_bo = BayesianOptimization(f=model_fusion, pbounds=pbounds, random_state=1, verbose=2)
fusion_bo.maximize(init_points=2, n_iter=3)
params_ = fusion_bo.max["params"]
```

为什么要融合？因为三个模型的「犯错方式」不一样：XGBoost 和 LightGBM 都是梯度提升，容易在相似的样本上一起错；随机森林是 bagging，方差更大但偏差方向不同。加权平均之后，某一类模型的个别错误会被其他模型抵消掉，准确率通常会比任何单模型高一点点。

**关于数据划分。** 我们用的是 `KFold` 而不是 `StratifiedKFold`，在类别比较均衡时两者差别不大；如果两类比例悬殊，建议换成分层版，保证每折里 CT/T 的比例和整体一致。

### 2.4 模型测试与指标报告

调参结束后，我们用全部训练数据把六个模型重新训练一遍，再对测试集预测。注意这里**不是**用交叉验证里的某一折模型，而是用全量数据重训——多喂数据总没坏处。

```python
# 按调好的参数重新构建六个模型（融合函数里的模型是局部的，这里要重建一份）
xgb1 = XGBClassifier(**params_xgb1, n_jobs=-1)
xgb2 = XGBClassifier(n_jobs=-1)
rf1  = RandomForestClassifier(**params_rf, n_jobs=-1)
rf2  = RandomForestClassifier(n_jobs=-1)
lgb1 = LGBMClassifier(**params_lgb, n_jobs=-1)
lgb2 = LGBMClassifier(n_jobs=-1)

# 全量重训
for m in (xgb1, xgb2, rf1, rf2, lgb1, lgb2):
    m.fit(train, y)

# 测试集上也做同样的加权平均
proba = (params_["a"] * xgb1.predict_proba(test) + params_["b"] * xgb2.predict_proba(test) +
         params_["c"] * rf1.predict_proba(test)  + params_["d"] * rf2.predict_proba(test)  +
         params_["e"] * lgb1.predict_proba(test) + params_["f"] * lgb2.predict_proba(test))
pred = np.argmax(proba / sum(params_.values()), axis=1)
```

最后把 0/1 还原成比赛要求的字符串。这里要注意方向不能反：我们编码时是 `CT → 1`，所以 1 要还原成 `"CT"`，0 还原成 `"T"`：

```python
for i in range(pred.shape[0]):
    sub.loc[i, "target"] = "T" if pred[i] == 0 else "CT"

sub.to_csv("sample_submission.csv", index=False)
```

指标报告方面，我们统计的是 5 折交叉验证的平均准确率（`accuracy_score` 的均值），这是最能反映模型真实水平的数字——它没有见过验证折的数据，比训练集准确率可信得多。材料里的 notebook 没有保留最终的打印结果，所以这里不写具体数值；但流程本身是完整的：调参看 AUC，验收看准确率，两者分工明确。

材料里还有两份思路不同的 notebook，可以对照着看：

- 一份走「暴力特征工程 + AutoML」路线：对 95 个数值特征各做 9 种变换（log、log1p、Yeo-Johnson、分位数变换、开方、MaxAbs、MinMax、Robust、归一化），再用 `PolynomialFeatures` 生成二阶交叉项，用 `VarianceThreshold(0.4)` 删掉方差过小的列，最后交给 PyCaret 自动比较模型、调参、融合、堆叠。它说明了「特征变换堆得多」也能出效果，但代价是特征维度爆炸、训练极慢。
- 一份是最朴素的基线：`pd.get_dummies` 之后直接 `DecisionTreeClassifier().fit()`，一棵不剪枝的决策树。它的作用不是拿高分，而是快速验证「数据能不能被树模型学会」——能跑通、能出提交文件，就说明数据格式没问题。

---

## 三、总结

本文中我们完成了 CS:GO 回合胜负预测的完整流程。我们首先把 `CT`/`T` 编码成 1/0，按数据类型拆出数值列和类别列，把训练集与测试集拼在一起做地图的独热编码，并逐列画出分布、用偏度和峰度圈出「形状太怪」的特征；其次我们用中位数填补测试集缺失值，对偏斜特征按「变换后峰度最小」的原则自适应地选择 log1p、开方或平方，并清理无穷大；接着我们用贝叶斯优化分别调 XGBoost、随机森林、LightGBM 的超参数，优化目标是 5 折 ROC AUC，再把六个模型（调参版与默认版各三个）的概率加权平均，权重同样用贝叶斯优化以准确率为目标搜索；最后在全量数据上重训、预测测试集、把 0/1 还原成 `T`/`CT` 并生成提交文件。

这个比赛给我们的体会是：**表格数据上，「看懂分布」比「换更复杂的模型」更值钱**。95 个特征里有大量零点堆积和长尾，如果不先把它们变换到接近正态，再强的模型也只能学到噪音。另一条经验是评估指标要分层使用：调参阶段用对阈值不敏感的 AUC，验收阶段用比赛真正考察的准确率，各司其职，才不会被某一折的偶然波动带偏。
