# Forest Cover Type（Kernels Only）

## 一、比赛简介

### 1.1 比赛背景

这个比赛用的是一份很经典的森林数据集（Covertype）。每一行代表一块林地，记录的是这块地的**地形**和**土壤**测量值：海拔多高、坡朝哪个方向、坡有多陡、离水源多远、离道路多远、离点火点多远、一天中三个时刻的山体阴影，以及它属于哪个荒野区、哪种土壤类型。我们要根据这些信息判断这块地上**主要长的是哪种树**。

数据集里的树被归成 7 个类别（`Cover_Type` 取值 1 到 7）。这些类别的真实含义（云杉、冷杉、松树等）对建模并不重要，我们只需要把它们当成 7 个编号。

这个比赛最大的难点不在模型，而在**数据规模的不对称**：训练集只有 15,120 行，测试集却有 565,892 行，测试集是训练集的 37 倍多。更麻烦的是，两边并不是从同一个分布里随机切出来的——材料里有人明确指出，`Elevation`（海拔）在训练集和测试集上的分布差别很大，以至于他干脆把这一列删掉，看模型能不能更稳。这种「训练集小、测试集大、分布还有偏移」的设定，意味着我们在本地交叉验证拿到的分数，只能当作参考，不能当成排行榜分数的精确预测。

`Kernels Only` 的意思是：所有提交都必须在 Kaggle 的 Notebook（当时叫 Kernel）环境里跑完并生成，不能在自己电脑上算好再上传。这逼着大家把代码写得更省内存、更省时间。

### 1.2 比赛目标

输入是一张宽表，训练集 56 列、测试集 55 列（少的那一列就是标签）。列可以分成四组：

- **连续的地形量**：`Elevation`、`Aspect`（坡向角度）、`Slope`（坡度）、`Horizontal_Distance_To_Hydrology`、`Vertical_Distance_To_Hydrology`、`Horizontal_Distance_To_Roadways`、`Horizontal_Distance_To_Fire_Points`、`Hillshade_9am`、`Hillshade_Noon`、`Hillshade_3pm`；
- **荒野区**：`Wilderness_Area1` 到 `Wilderness_Area4`，4 列 0/1；
- **土壤类型**：`Soil_Type1` 到 `Soil_Type40`，40 列 0/1；
- **编号与标签**：`Id` 和 `Cover_Type`。

我们要为测试集的每一行预测一个 1 到 7 的整数，写成 `Id, Cover_Type` 两列提交。

### 1.3 评价标准

比赛用**多分类准确率**（accuracy）评分：预测对的样本数除以总样本数，越高越好。7 个类别在训练集里是完全平衡的（每类 2160 条），所以「瞎猜一个类」的准确率是 1/7，大约 14.3%，这就是我们的分数下限。

因为指标只看「猜没猜中」，不看概率，所以最终提交的必须是类别编号，而不是概率。这一点在写代码时很容易出错，后面我们会专门提醒。

---

## 二、解题过程

整场比赛我们是这样推进的：我们首先把数据读进来，确认它的形状、类别分布和每一列的含义；其次做最小限度的清洗，扔掉没有信息量的列和明显的离群点；接着在原始特征上补几个由距离推导出来的几何特征；然后训练随机森林并用 10 折交叉验证评估；最后用同一套变换处理测试集、生成提交文件。下面按这四步展开。

### 2.1 数据探索（EDA）

第一步是确认数据规模。我们读入训练集和测试集，打印形状和标签分布：

```python
import pandas as pd
import numpy as np

# 训练集和测试集都是压缩包里的 csv，pandas 可以直接读 zip
train = pd.read_csv('../input/forest-cover-type-kernels-only/train.csv.zip')
test = pd.read_csv('../input/forest-cover-type-kernels-only/test.csv.zip')

print(train.shape)   # (15120, 56)
print(test.shape)    # (565892, 55)
print('缺失值:', train.isnull().sum().sum(), test.isnull().sum().sum())  # 都是 0

# 看看 7 个类别各有多少条
print(train['Cover_Type'].value_counts())
# 1~7 每类都是 2160 条，完全平衡
```

这一步给出三个关键事实：训练集只有 1.5 万行、测试集有 56 万行、两类数据都没有缺失值。类别完全平衡是一件好事——我们不需要做重采样，也不用担心模型偏向多数类。

接着我们确认每一列到底是什么类型。虽然 `info()` 显示 56 列全是 `int64`，但它们的物理含义完全不同：有的是连续测量值，有的是 0/1 标记。分清楚这一点，后面才不会把土壤类型当成「数字大小」去用：

```python
# 找出所有以某个前缀开头的列，例如所有 Soil_Type*
def get_common_prefix_columns(data, prefix):
    return [c for c in data.columns if set(prefix).issubset(c)]

# 统计每个 0/1 特征里「1」占多大比例，用数字代替画 40 张直方图
def get_binary_stats(data, prefix):
    for feature in get_common_prefix_columns(data, prefix):
        ratio = data[feature].value_counts(normalize=True).to_dict()
        print(feature, ratio)

get_binary_stats(train, 'Wilderness_Area')
# Area1: 23.79%  Area2: 3.30%  Area3: 41.99%  Area4: 30.92%
```

荒野区里 `Wilderness_Area3` 最常见（约 42%），`Wilderness_Area2` 最罕见（约 3.3%）。土壤类型则更极端，大部分 `Soil_Type` 的「1」占比都不到 5%，说明土壤是非常稀疏的类别特征。

然后我们画了 56×56 的相关矩阵，想看看有没有冗余特征：

```python
import matplotlib.pyplot as plt

f = plt.figure(figsize=(20, 8.27))
plt.matshow(train.corr(), fignum=f.number)
plt.xticks(range(train.shape[1]), range(train.shape[1]), fontsize=8, rotation=90)
plt.yticks(range(train.shape[1]), range(train.shape[1]), fontsize=8)
plt.colorbar()
plt.title('Correlation Matrix')
```

矩阵里立刻暴露出一个问题：有 3 列和所有其他列的相关系数都是 0，说明它们的值根本不变。回去核对二值统计，正是 `Soil_Type7`、`Soil_Type8`、`Soil_Type15`——前两个几乎全是 0，第三个全是 0。这类「常数特征」对模型没有任何贡献，只会增加维度。

最后我们按树种类别看了看山体阴影的均值：

```python
# 每个 Cover_Type 在三个时刻的平均 Hillshade
print(train.groupby('Cover_Type').mean()[
    ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
])
```

结果里 `Hillshade_9am` 的类别差异很明显：类型 4 的平均值约 227.97，类型 6 只有约 193.56；`Hillshade_3pm` 也从类型 6 的约 147.68 到类型 4 的约 111.81。这说明「光照条件」确实携带了树种信息，值得保留。

### 2.2 数据清洗

这份数据本身很干净：没有缺失值、没有重复行、没有文本需要解析。我们的清洗只做了三件事。

**第一，删掉 `Id`。** 它只是行号，和树种没有因果关系。把行号喂给模型，模型可能会记住「第几行是什么树」，这在训练集上看起来很好，到了测试集就完全失效。

**第二，删掉三个常数土壤列。** 就是 EDA 里发现的 `Soil_Type7`、`Soil_Type8`、`Soil_Type15`。理由很简单：一列如果所有样本都是 0，它连一次分裂都产生不了。

**第三，处理离群点。** 我们对连续特征做 Z-score 检查，也就是看每个值离均值有几个标准差，把超过 3 个标准差的点标出来：

```python
from scipy import stats

# 只对真正的连续列做检查，0/1 列不参与
num_cols = ['Elevation', 'Aspect', 'Slope',
            'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points']

outlier_idx = set()
for col in num_cols:
    z = np.abs(stats.zscore(train[col]))
    # z > 3 表示这个值离均值超过 3 倍标准差，属于极端值
    outlier_idx.update(train.index[z > 3].tolist())

print('离群样本占比:', len(outlier_idx) / train.shape[0])  # 约 0.0585

# 直接删除这些行，剩下 14236 行
train = train.drop(index=list(outlier_idx))
print(train.shape)   # (14236, 56)
```

大约 5.85% 的样本至少有一个特征超过 3 个标准差。我们选择直接删除，而不是用均值或中位数填补：因为这里的极端值更可能是测量错误（比如距离特别离谱），硬填一个平均值反而会制造出不存在的样本。代价是要丢掉 6% 的训练数据，对本来就小的训练集来说是有点心疼的，这也解释了为什么后面交叉验证的方差会比较大。

删完之后，再去掉 `Id` 和三个常数土壤列，训练集变成 14,236 行、52 列。

### 2.3 模型训练与超参数调优

**先做特征工程。** 原始特征里有两组距离是成对出现的：水平距离和垂直距离。它们其实描述的是同一个几何关系——从地块到水源，既可以拆成「水平走多远、垂直升降多少」，也可以合成一条直线距离。我们补的第一个特征就是这条直线距离：

```python
def diagonal_distance(data, x, y):
    """用勾股定理把水平距离和垂直距离合成直线距离"""
    return np.sqrt(data[x] ** 2 + data[y] ** 2)

def mean_value(data, feature_list):
    """对若干个距离取平均，表示整体上的远近程度"""
    total = 0
    for feature in feature_list:
        total += data[feature]
    return total / len(feature_list)

def transform_features(data):
    # Id 和三个常数土壤列没有信息量，统一在这里删掉
    data = data.drop(columns=['Id', 'Soil_Type7', 'Soil_Type8', 'Soil_Type15'])
    # 三个水平距离（水源、道路、火点）取平均
    horizontal = get_common_prefix_columns(data, 'Horizontal')
    data['Mean_Horizontal_Distance'] = mean_value(data, horizontal)
    # 到水源的水平距离与垂直距离合成直线距离
    hydrology = get_common_prefix_columns(data, 'Hydrology')
    data['Diagonal_Distance_To_Hydrology'] = diagonal_distance(
        data, hydrology[0], hydrology[1])
    return data

train = transform_features(train)
print(train.shape)   # (14236, 54)
```

为什么这两个特征有用？树模型每次分裂只能沿一个特征切一刀，切出来的边界是「横平竖直」的。而「离水源的直线距离」是一个斜着的方向，模型自己很难用水平和垂直两个特征拼出来。我们提前算好，等于把一个斜线方向变成了一个可以直接切的特征。

**再选模型。** 我们的主线是随机森林（Random Forest）。它的做法是训练很多棵决策树，每棵树只看一部分样本和一部分特征，最后让所有树投票。对这类「几十个特征、样本量不大、特征含义各异」的表格数据，随机森林几乎不用调参就能给出一个很稳的基线，而且不需要对特征做标准化。

我们用分层 10 折交叉验证来评估。分层的含义是：每一折里 7 个类别的比例都和原始数据一致，避免某一折碰巧缺了某个类别：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

Y = train['Cover_Type']
X = train.drop(columns=['Cover_Type'])

def train_predict(clf, X_train, X_valid, y_train):
    clf.fit(X_train, y_train)
    return clf.predict(X_valid)

clf = RandomForestClassifier(random_state=0)
skf = StratifiedKFold(n_splits=10)      # 10 折，每折验证集约 1400 行
scores = []

for k, (tr, va) in enumerate(skf.split(X, Y)):
    pred = train_predict(clf, X.iloc[tr], X.iloc[va], Y.iloc[tr])
    score = accuracy_score(Y.iloc[va], pred)
    scores.append(score)
    print('Fold %2d  Accuracy: %.3f' % (k + 1, score))

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
```

10 折的结果是 **0.788 ± 0.041**。逐折看，最好的一折 0.845，最差的一折只有 0.719，波动相当大。这不是模型不稳定，而是每折验证集只有一千四百多条，抽到几条「长得像别的类」的样本，分数就会明显抖动。所以我们更相信「均值 ± 标准差」这个区间，而不是某一折的具体数字。

**关于超参数。** 这条主线用的是 scikit-learn 的默认参数，没有再调。原因是训练集太小，调参很容易过拟合到交叉验证的某一折上。真正值得调的是「用哪些特征」，而不是「树建多深」。

材料里还有另一条更强的路线，值得单独说：有人先把特征从 55 列扩到 103 列——除了我们上面那两个距离特征，还加了早午、午后的明暗差、三个时刻的亮度组合，以及把几个距离按固定宽度分箱（例如把到火点的距离每 717 米归成一档）。然后他对 `Aspect`、`Slope`、各个距离和 `Hillshade` 做了 L2 归一化，并且**主动删掉了 `Elevation`**，理由是它在训练集和测试集上分布差别太大：

```python
from sklearn.preprocessing import normalize
from heamy.dataset import Dataset
from heamy.estimator import Classifier
from heamy.pipeline import ModelsPipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

TARGET, ID, NFOLDS, SEED = 'Cover_Type', 'Id', 5, 1337

def load_and_process_dataset():
    tr = pd.read_csv('../input/train.csv.zip')
    te = pd.read_csv('../input/test.csv.zip')
    # XGBoost 要求标签从 0 开始，所以先减 1
    y = tr[TARGET].ravel() - 1
    tr, te = tr.drop([ID, TARGET], axis=1), te.drop([ID], axis=1)
    tr, te = add_feats(tr), add_feats(te)          # 加衍生特征和分箱特征
    cols = ['Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points']
    # 距离类特征量纲差别大，逐行做 L2 归一化后再喂给线性模型和神经网络
    tr[cols] = normalize(tr[cols])
    te[cols] = normalize(te[cols])
    tr, te = tr.drop('Elevation', axis=1), te.drop('Elevation', axis=1)
    return {'X_train': tr.values, 'X_test': te.values, 'y_train': y}

dataset = Dataset(preprocessor=load_and_process_dataset, use_cache=True)

# 第一层：5 个基模型，各自 5 折输出「折外概率」
rf  = Classifier(dataset=dataset, estimator=RandomForestClassifier,
                 parameters={'n_estimators': 200, 'criterion': 'entropy',
                             'random_state': 0}, name='rf')
et  = Classifier(dataset=dataset, estimator=ExtraTreesClassifier,
                 parameters={'n_estimators': 200, 'criterion': 'gini',
                             'random_state': 0}, name='et')
lgbc = Classifier(dataset=dataset, estimator=LGBMClassifier,
                  parameters={'n_estimators': 200, 'learning_rate': 0.1},
                  name='lgbc')
logr = Classifier(dataset=dataset, estimator=LogisticRegression,
                  parameters={'solver': 'liblinear', 'multi_class': 'ovr', 'C': 1},
                  name='logr')
gnb = Classifier(dataset=dataset, estimator=GaussianNB, name='gnb')

# 把第一层的输出拼成新特征，第二层再用逻辑回归学一遍
pipeline = ModelsPipeline(rf, et, lgbc, logr, gnb)
stack_ds = pipeline.stack(k=NFOLDS, seed=SEED)
stacker = Classifier(dataset=stack_ds, estimator=LogisticRegression,
                     parameters={'C': 5, 'solver': 'liblinear',
                                 'multi_class': 'ovr', 'random_state': SEED})
```

这条堆叠（stacking）路线在第二层用 `log_loss` 评估，得到 0.2628 ± 0.0059。`log_loss` 衡量的是「预测概率离真实标签有多远」，越小越好；它比准确率更细腻，能看出模型是不是「虽然猜对了，但很心虚」。需要强调的是，0.2628 和 0.788 不是同一个指标，不能直接比较，我们在这里只是用它来判断堆叠有没有带来更平滑的概率输出。

### 2.4 模型测试与指标报告

交叉验证确认流程没问题后，我们用全部清洗后的训练数据重新训练一次，再对测试集做**完全相同的变换**——这一点非常关键，特征工程里用到的列名、计算方式必须和训练时一致：

```python
# 用全部训练数据重训（不再留验证集）
final_model = RandomForestClassifier(random_state=0)
final_model.fit(X, Y)

# 测试集先做同样的清洗和特征工程，再预测
test_data = pd.read_csv('../input/forest-cover-type-kernels-only/test.csv.zip')
test_data = transform_features(test_data)
print(test_data.shape)          # (565892, 53)

y_test = final_model.predict(test_data)

# 按提交格式写出 Id 和 Cover_Type 两列
submission = pd.DataFrame({'Id': test_data['Id'], 'Cover_Type': y_test})
submission.to_csv('submission.csv', index=False)
```

这里有两个容易踩的坑。第一，`transform_features` 会删掉 `Id`，所以我们必须**先**把 `Id` 存下来，否则提交文件就没有主键了。第二，如果改用 XGBoost 之类的模型，它内部要求标签从 0 开始编号，预测出来要记得 `+1`，否则提交的类别会整体错一位——材料里那条堆叠路线就特意写了 `np.round(np.argmax(preds_proba, axis=1)).astype(int) + 1`。

最终我们的本地成绩是 10 折交叉验证准确率 **0.788 ± 0.041**。考虑到训练集只有一万四千行、测试集有 56 万行，而且两边的海拔分布还不一致，这个数字应当理解成「模型大致能猜对八成地块」的量级，而不是对排行榜名次的承诺。真正上线时，模型面对的是一批分布略有偏移的新数据，稳健性比多零点几个百分点更重要——这也是那条堆叠路线宁愿删掉最强特征 `Elevation` 的原因。

---

## 三、总结

本文中我们完成了 Forest Cover Type（Kernels Only）比赛的完整流程。我们首先用 EDA 摸清了数据：训练集 15,120 行、测试集 565,892 行、7 个类别完全平衡、没有任何缺失值，并发现 `Soil_Type7/8/15` 三列几乎是常数、`Hillshade` 在不同树种间有明显差异。其次我们做了很轻的清洗：删掉行号 `Id`、删掉三列常数土壤特征、用 3 倍标准差的 Z-score 规则剔除约 5.85% 的离群样本，训练集剩下 14,236 行。接着我们补了两个几何特征——到水源的直线距离和三个水平距离的均值——把「斜着的关系」变成树模型可以直接切的一列。然后我们用默认参数的随机森林做 10 折分层交叉验证，得到 0.788 ± 0.041；同时我们也尝试了另一条路线：把特征扩到 103 列、做 L2 归一化、去掉分布不一致的 `Elevation`，再用随机森林、ExtraTrees、LightGBM、逻辑回归、朴素贝叶斯做第一层、逻辑回归做第二层堆叠，第二层的 `log_loss` 为 0.2628 ± 0.0059。最后我们用全量数据重训、对测试集做同样的变换并生成 `Id, Cover_Type` 提交文件。

这个比赛给我们最大的启发是：**当训练集和测试集的分布不一致时，「删掉最强特征」有时反而是一种保护**。`Elevation` 在训练集里几乎能单独决定树种，但它一旦在测试集上漂移，模型就会错得很自信。与其把一个不可靠的强特征喂进去，不如让模型多依赖几个彼此印证的中等特征。
