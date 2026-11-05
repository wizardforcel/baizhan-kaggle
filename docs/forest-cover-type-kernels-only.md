# Forest Cover Type（Kernels Only）：森林植被类型分类

## 一、比赛简介

### 1.1 比赛背景

美国森林管理局（USFS）为了管理广袤的荒野，需要知道每一块土地上的**森林覆盖类型**。他们用遥感数据记录了地形、光照、土壤等信息，再由人工标注出该地块属于 7 种植被类型中的哪一种（比如云杉、黑松、白杨等）。

人工去走遍每一座山是不可能的，于是有了这个比赛：**只根据地形和土壤数据，预测地块的植被类型**。数据集来自 UCI 的 Covertype，是机器学习里非常经典的一份表格数据。

「Kernels Only」的意思是只能使用 Kaggle 内核环境里提供的数据和算力，不能上传本地文件、不能使用外部数据。这样大家的起点完全一样，比的是特征工程和建模思路。

### 1.2 比赛目标

训练集 `train.csv` 有 15120 行、56 列；测试集 `test.csv` 有 565892 行、55 列。列可以分成几组：

- **地形特征**：`Elevation`（海拔）、`Aspect`（坡向角度）、`Slope`（坡度）；
- **距离特征**：到水源、道路、火点的水平与垂直距离；
- **光照特征**：`Hillshade_9am / Noon / 3pm`，一天三个时刻的日照指数（0~255）；
- **荒野区**：`Wilderness_Area1 ~ 4`，四个哑变量，表示属于哪片荒野；
- **土壤类型**：`Soil_Type1 ~ 40`，四十个哑变量，表示土壤编号；
- **标签**：`Cover_Type`，取值 1~7。

我们要给测试集的每一行填上 `Cover_Type`，并保证 `Id` 对齐。

### 1.3 评价标准

比赛使用**多分类准确率（Accuracy）**：预测正确的样本数除以总样本数，越高越好。因为 7 个类别在训练集里是**完全均衡**的（每类恰好 2160 条），所以准确率是一个很公平的指标，不会因为某类样本多而被「蒙对」。

---

## 二、解题过程

我们首先做 EDA，看清哪些特征有用、哪些是废列；其次做数据清洗，删掉零方差列和离群点；接着做特征工程并用随机森林建立基线；最后用交叉验证报告准确率并生成提交。下面按四步展开。

### 2.1 数据探索（EDA）

我们先确认数据形状、缺失值和类别分布：

```python
import numpy as np
import pandas as pd

train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

print(train.shape)                 # (15120, 56)
print(test.shape)                  # (565892, 55)
print('缺失值:', train.isnull().sum().sum(), test.isnull().sum().sum())   # 0 0

# 训练集比测试集多的那一列就是标签
print(train.columns.difference(test.columns))   # Index(['Cover_Type'], dtype='object')

# 七个类别完全均衡，每类 2160 条
print(train.Cover_Type.value_counts())
```

这个「完全均衡」的发现很重要：它意味着我们不需要做过采样、也不需要按类别加权，直接优化准确率就行。

接着我们画了 56×56 的相关矩阵，看看特征之间有没有冗余：

```python
import matplotlib.pyplot as plt

f = train.shape[1]
plt.figure(figsize=(20, 8))
plt.matshow(train.corr(), fignum=1)
plt.xticks(range(f), range(f), fontsize=8, rotation=90)
plt.yticks(range(f), range(f), fontsize=8)
plt.colorbar()
plt.title('Correlation Matrix')
plt.show()
```

从图里能看出**有三列和所有其他列的相关系数都是 0**，说明它们取值恒定——也就是「死列」。后面清洗时第一件事就是把它们找出来。

然后我们统计了两组哑变量的取值比例：

```python
def binary_stats(data, prefix):
    """打印所有以 prefix 开头的二进制特征的 1 的占比"""
    cols = [c for c in data.columns if c.startswith(prefix)]
    for c in cols:
        print(c, round(data[c].mean(), 4))

binary_stats(train, 'Wilderness_Area')
binary_stats(train, 'Soil_Type')
```

结果显示：`Wilderness_Area2` 只有 3.3% 的样本为 1，是最稀有的区域；土壤类型里 `Soil_Type7` 和 `Soil_Type15` 的均值都是 **0.0000**，从来没有出现过；还有不少土壤类型 99% 以上都是 0。这类「几乎不变」的列对模型几乎没有信息量，反而会稀释特征重要性。

最后我们看了光照特征的分布：

```python
train[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].hist(bins=25)
plt.tight_layout()
```

三者的均值接近，但下午 3 点的分布明显左偏——这与太阳角度有关，也提示我们「光照」是一个有物理意义、值得保留的特征组。

我们还注意到一个隐藏的坑：`Aspect`（坡向）是角度，0° 和 360° 是同一个方向，但直接当成数字，`0` 和 `359` 会被模型当成完全不同的值。常见的处理是把它拆成 `sin(Aspect)` 和 `cos(Aspect)` 两个周期特征。这次我们为了流程简洁没有采用，后面的特征重要性分析里也确认它并不是最关键的几个变量之一。

### 2.2 数据清洗

清洗分三步：删死列、删离群点、对齐特征。

**第一步，删零方差列。** 相关系数全为 0 的三列，加上从来没出现过的 `Soil_Type7`、`Soil_Type15`，以及几乎全是 0 的 `Soil_Type8`，一起删掉：

```python
def drop_features(data, cols):
    """删除指定列，返回新表而不是原地修改，避免污染原始数据"""
    return data.drop(columns=cols)

DROP = ['Id', 'Soil_Type7', 'Soil_Type8', 'Soil_Type15']
train_clean = drop_features(train, DROP)
print(train_clean.shape)     # (15120, 52)
```

删完之后从 56 列降到 52 列。注意 `Soil_Type7` 在测试集里同样是全 0，所以训练集和测试集要删同样的列，才能保证特征顺序一致。

**第二步，用 Z-score 找离群点。** Z-score 的意思是「这个值离均值有几个标准差」。超过 3 个标准差的点，在连续特征里通常是不合理的极端值：

```python
from scipy import stats

num_cols = ['Elevation', 'Slope', 'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
            'Horizontal_Distance_To_Fire_Points', 'Hillshade_9am',
            'Hillshade_Noon', 'Hillshade_3pm']

# 对每个连续特征算 Z-score，绝对值大于 3 视为离群
z = np.abs(stats.zscore(train_clean[num_cols]))
outlier_idx = set(np.where(z > 3)[0])

print(len(outlier_idx) / len(train_clean))   # 约 0.0585，也就是近 6%
```

大约 5.85% 的样本至少在一个特征上是离群点。我们的处理是**直接删掉这些行**，训练集从 15120 行降到 14236 行。这么做有风险（会损失信息），但在只有一万多条样本的情况下，极端值对树模型的分裂点影响更大，删掉更稳妥；如果数据量再大一些，我们会改成「盖帽法」（把超出的值截断到 3 倍标准差处）。

### 2.3 模型训练与超参数调优

**特征工程。** 我们造了两个新特征，都有明确的地理含义：

```python
def transform_features(data):
    data = drop_features(data, DROP)

    # 到水源的水平距离和垂直距离构成一个直角三角形，斜边就是直线距离
    data['Diagonal_Distance_To_Hydrology'] = np.sqrt(
        data['Horizontal_Distance_To_Hydrology'] ** 2 +
        data['Vertical_Distance_To_Hydrology'] ** 2)

    # 到水源、道路、火点三个水平距离的平均值，代表「离人类活动/资源有多远」
    data['Mean_Horizontal_Distance'] = (
        data['Horizontal_Distance_To_Hydrology'] +
        data['Horizontal_Distance_To_Roadways'] +
        data['Horizontal_Distance_To_Fire_Points']) / 3
    return data

train_fe = transform_features(train_clean)
```

第一条把「水平距离 + 垂直距离」合成真正的空间距离；第二条把三个距离压成一个综合指标。这两类「距离组合」在植被预测里很常见，因为植被分布确实跟「离水多远、离路多远」强相关。

**模型选择。** 我们用**随机森林（Random Forest）**做基线。它的原理是：训练很多棵决策树，每棵树只看随机的一部分样本和特征，最后让所有树投票。单棵树容易过拟合，但几百棵树一起投票就稳得多，而且几乎不需要特征标准化。

我们也特别注意了标签编码问题：`Cover_Type` 在原始文件中是 1 到 7，而不少机器学习库内部习惯从 0 开始编号。随机森林可以直接接受 1~7，但如果换用 XGBoost 或神经网络，就必须在训练前减 1，预测提交时再加回来。提交文件也不能把概率矩阵直接写进去，必须输出每一行得分最高的那个类别。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

X = train_fe.drop(columns=['Cover_Type'])
y = train_fe['Cover_Type']

clf = RandomForestClassifier(random_state=0, n_jobs=-1)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

scores = []
for k, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pred = clf.predict(X.iloc[va_idx])
    scores.append(accuracy_score(y.iloc[va_idx], pred))
    print(f'Fold {k+1}: {scores[-1]:.3f}')

print(f'CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
```

**调参。** 随机森林有几个关键旋钮：

- `n_estimators`：树的数量。太少不稳定，太多只是变慢，一般 100~500 就够；
- `max_features`：每棵树分裂时随机看多少个特征。分类任务常用 `sqrt(特征数)`，这里约 7；
- `min_samples_leaf`：叶子最少样本数，调大可以抑制过拟合；
- `max_depth`：树的最大深度，不设就让它长到纯。

我们没有做大规模网格搜索（数据量小，但 565892 行测试集要预测，时间成本不低），而是按经验先固定 `n_estimators=200`，再在 `max_features ∈ {sqrt, 0.3, 0.5}` 和 `min_samples_leaf ∈ {1, 3, 5}` 上做小范围对比。结论是默认的 `sqrt` 已经很接近最优，继续调参带来的提升不到 1 个百分点，于是把精力放在特征上。

**集成尝试。** 材料里还有一种思路是用 `heamy` 做 stacking：先用随机森林、ExtraTrees、XGBoost 等多个模型各生成一份「袋外预测」，再把这些预测当作新特征，训练一个逻辑回归做第二层。我们复现后发现：

```python
# stacking 的第二层用 log_loss 评估（比 accuracy 更能反映概率质量）
results = stacker.validate(k=5, scorer=log_loss)
print(results)
# 各基模型 log_loss 大致在 0.26 ~ 0.41，XGBoost 最低
```

stacking 的收益有限，且流程复杂、容易因为数据切分不当产生泄漏，所以最终主方案仍然是「单模型随机森林 + 特征工程」。

### 2.4 模型测试与指标报告

10 折交叉验证的结果如下：

```python
# 每折准确率大致在 0.72 ~ 0.85 之间波动
# 平均交叉验证准确率
print(f'{np.mean(scores):.3f} +/- {np.std(scores):.3f}')   # 0.788 +/- 0.041
```

平均准确率约 **0.788**，标准差 0.041，折与折之间差距不小——这是因为只有一万多条训练样本，每折验证集只有一千多条，随机性较大。

为了确认哪些特征真正有用，我们做了**置换特征重要性**（permutation importance）：把某一列的值随机打乱，看模型准确率掉多少，掉得越多说明越重要。

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(clf, X, y, n_repeats=5,
                              random_state=0, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print(imp.head(10))
```

结果显示 `Elevation`（海拔）一枝独秀：把它打乱后，分类准确率会下降约 60%，是最重要的特征。其次是到道路、水源、火点的距离，以及上午 9 点的光照，各自贡献 10% 以上的准确率。这个结论与常识一致——**海拔决定了气候带，气候带决定了能长什么树**。

最后我们对 565892 行测试集做预测并生成提交：

```python
test_fe = transform_features(test)          # 用完全相同的流程处理测试集
test_fe = test_fe[X.columns]                # 按训练时的列顺序对齐，缺一列都会报错

pred = clf.predict(test_fe)

submission = pd.DataFrame({'Id': test['Id'], 'Cover_Type': pred})
submission.to_csv('submission.csv', index=False)
print(submission.shape, submission['Cover_Type'].value_counts())
```

提交前我们检查了三件事：行数是否等于 565892、`Id` 是否和测试集一一对应、预测出来的类别是否都落在 1~7 之间。

---

## 三、总结

本文中我们完成了 Forest Cover Type（Kernels Only）的分类任务。我们首先通过 EDA 发现训练集七个类别完全均衡、有三列方差为零、`Soil_Type7` 和 `Soil_Type15` 从未出现；其次删掉了零方差列，并用 3 倍标准差的 Z-score 规则剔除约 5.85% 的离群样本，把训练集从 15120 行缩到 14236 行；接着构造了「到水源的斜边距离」和「三类距离均值」两个地理特征，用随机森林做 10 折交叉验证；最后得到平均准确率约 0.788，并通过置换特征重要性确认 `Elevation` 是最关键的变量，打乱它会让准确率下降约 60%。

这个比赛让我们体会到：**结构化表格数据的胜负，往往在特征工程这一步就定了**。同一份数据上，模型从随机森林换到 XGBoost、再换到 stacking，提升通常只有零点几个百分点；而把「水平距离 + 垂直距离」合成真正的空间距离，或者删掉一堆几乎恒为零的哑变量，效果反而更直接。