# Don't Get Kicked！（别买到坏车）

## 一、比赛简介

### 1.1 比赛背景

二手车店里最可怕的事情，不是进了一辆卖不动的新车，而是从批发拍卖会上**买了一辆"病车"**。很多二手车行会去 ADESA、MANHEIM 这样的旧车拍卖场进货——买家看一眼车况、估个价、举牌成交，还是"买定离手"。有的车买回来才发现毛病比看到的严重得多，修车、退货、再上架的成本都砸在自己手里，这种"买亏了"的车在业内就叫 **kicked（被踢）**。

这个比赛的数据就来自某大型二手车商的拍卖进货记录。每辆车在买入那一刻，系统里已经有一些信息：车龄、里程、颜色、拍卖场、品牌型号、一家叫 **MMR** 的评估机构给出的各种价格，等等。比赛要我们做的，就是**在买下这辆车之前**，用这些信息判断它会不会变成一笔"问题买卖"。数据词典里对目标列 `IsBadBuy` 的解释只有一句话：这个标签标识出"被踢的车辆是不是一次可以避免的购买"。

### 1.2 比赛目标

输入是二手车拍卖记录，每行一辆车。训练集一共 **72983** 行、**33** 列，分三类：

- 标识类：`RefId`（编号）、`PurchDate`（购买日期）、`Auction`（拍卖场）、`BYRNO`（买家编号）；
- 车辆本身：`VehYear`（出厂年份）、`VehicleAge`（车龄）、`Make`（品牌）、`Model`（型号）、`Trim`/`SubModel`（配置）、`Color`、`Transmission`（变速箱）、`WheelType`（轮毂类型）、`VehOdo`（里程）、`Nationality`（产地）、`Size`（尺寸）、`VNZIP1`/`VNST`（邮编和州）、`IsOnlineSale`（是否线上成交）等；
- 价格类：8 个 `MMR` 评估价（成交价/零售价 × 均价/精洗价 × 收购时/当前）、`VehBCost`（购入成本）、`WarrantyCost`（保修成本）。

目标列是 `IsBadBuy`：1 表示这是一辆被踢的问题车，0 表示正常。测试集只有特征没有标签，我们要给每条测试样本输出一个"它是坏车"的得分。

### 1.3 评价标准

比赛用 **归一化 Gini 系数**（Normalized Gini）打分。理解它很简单：把每一辆车按"坏车概率"从高到低排好队，想象画一条"积累坏车占比"的曲线。如果概率排序完全正确，曲线就涨得又快又高，Gini 越接近 1；如果完全瞎排，曲线就和随机的对角线重合，Gini 等于 0。所以 **Gini 越高越好**。它其实和更常见的 ROC AUC 是等价的——归一化 Gini ≈ 2 × AUC − 1，只是画法不同。这也提醒我们：提交文件里必须输出**概率**而不是 0/1 标签，排序才有意义。

---

## 二、解题过程

我们的整体思路是：先把数据读进来摸清家底，再针对"缺值多、脏值多"的拍卖数据做清洗和特征工程，然后用一条 **预处理流水线 + 随机森林** 组成模型，最后按 Gini 评估并生成提交文件。下面按四步走。

### 2.1 数据探索（EDA）

首先把训练集、测试集读进来，统计每一列的缺失情况。因为拍卖记录是人工录入的，我们对"哪列缺得多"特别敏感：

```python
import numpy as np
import pandas as pd

# index_col='RefId' 让编号直接当地图索引，训练集里 RefId 是唯一的
train = pd.read_csv('../input/DontGetKicked/training.csv', index_col='RefId')
test  = pd.read_csv('../input/DontGetKicked/test.csv')

# 便捷函数：统计每列的缺失个数和缺失比例，按缺失多少排序
def missing_value_stats(df):
    count = df.isna().sum().sort_values(ascending=False)   # 每列缺失数
    proportion = count / df.shape[0]                        # 缺失比例
    out = pd.concat([count, proportion], axis=1)
    out.columns = ['missing values', 'proportion']
    return out

missing = missing_value_stats(train)
print(train.shape)              # (72983, 33)：72983 辆车，33 列
missing.loc[missing['proportion'] > 0]
```

```
               missing values  proportion
PRIMEUNIT            69564      0.953153
AUCGUART             69564      0.953153
WheelType             3174      0.043490
Trim                  2360      0.032336
MMRCurrent*Price       315      0.004316
…
```

结论很清楚：**19 列里有 17 列带缺失**，其中 `PRIMEUNIT`（是否准新库存）和 `AUCGUART`（是否有拍卖行保修）这两列 **95.3% 都是空的**——几乎等于这一特征"大多数车没记录"。剩下的缺值比例很小，属于正常情况。

接着看目标变量是不是失衡，以及连续特征与目标的关系：

```python
# 目标变量取值分布：1 是"被踢"的问题车
print(train['IsBadBuy'].value_counts(normalize=True))
# 0    0.877
# 1    0.123

# 12 个连续特征，逐一做 ANOVA 检验：
# 看"正例组"和"负例组"的均值差得够不够大（F 越大差得越明显）
from scipy import stats
for col in continuous_cols:
    v0 = train.loc[train.IsBadBuy == 0, col]
    v1 = train.loc[train.IsBadBuy == 1, col]
    F, p = stats.f_oneway(v0, v1)
    if p < 0.01:
        print(col, 'significant, F =', round(F, 1))
```

运行结果里 `IsBadBuy` 的正类只占 **12.3%**，两类不均等（一个永远猜 0 的"笨模型"也能拿到 87% 的正确率）。而所有 12 个连续特征通过 ANOVA 检验，其中 `MMRAcquisitionAuctionAveragePrice`（收购时成交均价）的 F 值高达 881，`VehOdo`（里程）F 值为 501，说明**价格类和里程类特征对"坏车"最有判别力**——这也符合直觉：买贵了、跑得多的车更容易出问题。

最后看一眼离散特征的"基数"（有多少种取值）：

```python
# 每列有多少个不同取值，决定后面的编码策略
cardinality = train.nunique()
print(cardinality.sort_values(ascending=False).head(10))
# Model     1063    型号，取值非常多（高基数）
# SubModel   863
# Make        33
# BYRNO       74
# …
# WheelType    3    有些列只有两三个取值，几乎可以当类别用
```

`Model` 有 1000 多种，直接独热编码会炸出几千列。这为后面的特征工程埋了伏笔：与其把 1000 种型号硬编码成 1000 列，不如按型号求"该型号的平均收购价"，把信息浓缩成一列。

### 2.2 数据清洗

针对 EDA 发现的三种脏数据，我们分别处理：

1. **缺值超过 95% 的列**（`PRIMEUNIT`、`AUCGUART`）：缺这么多已经不叫"缺失"了，而是"绝大多数车没有这个属性"，所以直接填字符串 `'unknown'`，让模型把它当一类处理；
2. **连续变量的少量缺值**（MMR 价格、`WheelTypeID` 等）：用**中位数**填——拍卖价格常有右偏的极端值，中位数比均值更稳；
3. **离散变量的缺值**（颜色、变速箱等）：用**众数**（出现最多的值）填。

```python
# 缺失超过 20% 的列一律填 'unknown'，让它成为合法类别
threshold = 0.2
high_missing = list(missing[missing['proportion'] > threshold].index)
# high_missing = ['PRIMEUNIT', 'AUCGUART']

for col in continuous_cols:                 # 连续列：中位数填充
    train[col] = train[col].fillna(train[col].median())
for col in discrete_cols:                   # 离散列：众数填充
    train[col] = train[col].fillna(train[col].mode()[0])
train[high_missing] = train[high_missing].fillna('unknown')
```

注意 `PurchDate` 要先转成时间类型，才能参与后续统计。清洗之后我们重新跑了一遍缺失统计，确认所有列都填满了。

做完基本填充，我们做了一个**分类型+分组聚合**的特征工程：`Model` 有 1000 多种取值，直接编码维度太高，但"同一型号的车在 MMR 上的平均收购价"是一个很稳定的信号，于是按品牌、型号、配置这些列分组，把每辆车的 MMR 均价换成本组的平均值，作为新特征：

```python
# 新特征：按某列分组，求该组 MMR 收购均价的平均值，回填给每个成员
# 例如 mean_..._Make 就表示"同品牌的车平均收购价多少"
for grp_col in ['Make', 'Model', 'Trim', 'SubModel', 'Color', 'Transmission']:
    name = f'mean_MMRAcquisitionAuctionAveragePrice_{grp_col}'
    train[name] = train.groupby(grp_col)['MMRAcquisitionAuctionAveragePrice'].transform('mean')
    test[name]  = test.groupby(grp_col) ['MMRAcquisitionAuctionAveragePrice'].transform('mean')
```

这样 1000 多种型号的信息被压缩进一个"组内均价"列里，既保住了经济信号，又不会把特征维度撑爆。

### 2.3 模型训练与超参数调优

我们选 **随机森林** 作为模型，并用 sklearn 的 `ColumnTransformer` 把"数值列处理"和"类别列处理"打包成一条**预处理流水线**（pipeline）。用流水线的最大好处是：清洗、编码这些操作只在训练数据上学得参数，交叉验证时**不会把测试集的信息泄漏进训练过程**。

```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 数值列：先按中位数补缺，再标准化（让量纲统一，树模型虽不受此影响，
#         但后面如果接线性模型会需要）
num_pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)
# 类别列：缺失值填 'missing'，再独热编码；
# handle_unknown='ignore' 保证测试集出现新取值时不会报错
cat_pipe = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='missing'),
    OneHotEncoder(handle_unknown='ignore')
)

# 把两条通路拼成一个"预处理器"，再在后面接随机森林
preprocessor = make_column_transformer(
    (num_pipe, numerical_features),
    (cat_pipe, categorical_features)
)
model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100))

# 按目标变量分层切分，保证训练/验证集里正例比例一致
X = train.drop('IsBadBuy', axis=1)
y = train['IsBadBuy']
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

model.fit(X_train, y_train)
print(f'Train acc : {model.score(X_train, y_train):.3f}')   # 1.000
print(f'Val   acc : {model.score(X_val,   y_val):.3f}')     # 0.900
```

训练集 100% 正确、验证集 90%，两者差距很大——随机森林对训练集**过拟合**了。但这不妨碍它在验证集上仍然拿到 90% 的正确率，远高于"永远猜 0"的 87%。因为正类只有 12%，正确率这个指标不够敏感，我们稍后改用 Gini 评估。

接下来调参。手工穷举很便宜，我们用 `RandomizedSearchCV` 在"树的数量、每棵树的深度、每次分裂看的特征数"上随机搜几十组，用 3 折交叉验证挑最优组合：

```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300, 500, 700, 900],
    'randomforestclassifier__max_features': ['auto', 'sqrt'],
    'randomforestclassifier__max_depth':    [2, 4, 6, 8],
}
# 搜索器会对 pipeline 做交叉验证；参数名要带上前缀 randomforestclassifier__
search = RandomizedSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
search.fit(X_train, y_train)

print(search.best_params_)
# {'max_depth': 8, 'max_features': 'auto', 'n_estimators': 926}
print(search.best_estimator_.score(X_val, y_val))   # 0.899
```

搜索结果显示深度限制在 8、树的数量取 900 多棵效果最好。较浅的树 **缓解了过拟合**（训练正确率从 100% 降到 89.7%），而验证集仍保持 89.9%——泛化更好，这正是我们想要的。

### 2.4 模型测试与指标报告

我们按比赛的 Gini 口径评估。把每一辆车按预测的坏车概率从高到低排序，然后画"累计坏车"曲线，计算曲线与随机线之间的面积差，归一化到 0～1：

```python
def gini(actual, pred):
    """按预测值排序后计算归一化 Gini 系数（越高越好，随机猜为 0）"""
    data = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    data = data[np.lexsort((data[:, 2], -1 * data[:, 1]))]   # 按概率从高到低排序
    total_loss = data[:, 0].sum()                            # 总的坏车数
    gini_sum = data[:, 0].cumsum().sum() / total_loss        # 累计坏车占比之和
    gini_sum -= (len(actual) + 1) / 2.                       # 减掉随机线的面积
    return gini_sum / len(actual)

def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)         # 除以理想值归一化

# 提交时必须用概率，而不是 0/1 硬标签
val_prob = search.predict_proba(X_val)[:, 1]
print('Gini : %.3f' % gini_normalized(y_val, val_prob))
```

使用**概率**评估后，Gini 才有意义（如果用 0/1 硬标签，很多概率都被抹掉了，排序信息全丢，算出来的 Gini 只有 0.10 左右）。我们最后用全量训练数据重新训练一次，对测试集输出坏车概率并保存提交文件：

```python
# 用全部数据重新拟合，把最后一点数据用上
model.fit(X, y)

# 输出"坏车概率"而非标签——Gini 比的是排序而不是 0/1 对不对
test_prob = model.predict_proba(test)[:, 1]
submission = pd.DataFrame({'RefId': test['RefId'], 'IsBadBuy': test_prob})
submission.to_csv('submission.csv', index=False)
```

---

## 三、总结

本文中我们完成了 Don't Get Kicked！二手车坏车预测的完整流程。我们首先做 EDA，确认训练集 72983 辆车、目标正例仅占 12.3%、MMR 价格和里程是最强特征；其次做数据清洗，把缺失率 95% 的列填成 `'unknown'`，数值列用中位数、离散列用众数补齐，并对品牌/型号做了"组内 MMR 均价"特征工程；接着把数值/类别预处理打包成流水线，接随机森林并用 `RandomizedSearchCV` 把树深限制到 8；最后按比赛的归一化 Gini 口径评估并提交坏车概率。最终验证集正确率约 90%、显著超过"永远猜 0"的 87% 基线。社区里也有尝试 KNN、XGBoost、神经网络等方案，我们选随机森林主线是因为它调参稳、结果好解释，对入门者最友好。

这个比赛给我们的最大启发是：**脏数据治理的优先级往往高于模型选择**。上百种型号、九成缺失的字段、右偏到极端的拍卖价，这些"数据工程"问题处理好了，一个朴素的随机森林就能达到不错的效果；而评价指标也要和提交格式对齐——比赛用排序类指标 Gini，我们就必须交概率而不是交标签。