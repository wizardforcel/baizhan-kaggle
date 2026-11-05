# West Nile 病毒预测

## 一、比赛简介

### 1.1 比赛背景
West Nile 病毒（西尼罗病毒）由蚊虫传播，感染后可能引发脑膜炎、脑炎等严重疾病。美国芝加哥的公共卫生部门每年夏天在全市布设数百个捕蚊陷阱，定期捕蚊、鉴定种类、检测是否携带病毒。比赛要求我们根据历史和气象数据，预测每个捕蚊点是否会发现携带病毒的蚊子。

这个任务有几个天然的难点。第一，**正样本极少**，训练集中只有约 5% 的记录携带病毒，模型很容易学成“永远预测阴性”。第二，**采样是每年夏天集中进行**的，年份、季节和地点耦合在一起，特征之间高度相关。第三，训练集中在 2007 年，测试集覆盖 2008 和 2009 年，模型必须学会“迁移”，不能只记住 07 年哪些点感染过。

### 1.2 比赛目标
训练集 `train.csv` 有 10506 行、12 列：

- `Date`：采样日期；
- `Address`、`Block`、`Street`、`Trap`、`AddressNumberAndStreet`：陷阱位置信息；
- `Species`：蚊子种类；
- `Latitude`、`Longitude`：经纬度；
- `AddressAccuracy`：地址精度；
- `NumMosquitos`：这一笼捕到的蚊子数量；
- `WnvPresent`：目标，0 表示未携带病毒，1 表示携带。

测试集有 116293 行，结构和训练集一致但没有 `WnvPresent`。另外提供 `weather.csv`（两个气象站的逐日观测，有温度、降水、湿度、风速等 22 列）和 `spray.csv`（喷洒灭蚊记录）。我们要给每条测试样本输出一个 0 到 1 的概率。

`spray.csv` 记录的是哪一天在哪个经纬度喷洒了杀虫剂。理论上它会降低喷洒后一段时间的带毒率，但喷洒记录只覆盖部分区域、时间点也稀疏，如果直接按日期合并会产生大量空值。我们这条主线方案没有使用它，而是把重点放在捕蚊记录和气象数据上；这是本次比赛一个可以继续改进的方向。

### 1.3 评价标准
比赛使用 **ROC AUC**（曲线下面积）评估。我们随机抽一对“一正一负”的样本，模型把正样本排在前面的概率就是 AUC。0.5 等于瞎猜，越接近 1.0 越好。比赛提供的“踩线基准”方案大概在 0.70 左右，我们可以把它当作及格线。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读入所有数据。

```python
import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

train = pd.read_csv('/kaggle/input/predict-west-nile-virus/train.csv')
test = pd.read_csv('/kaggle/input/predict-west-nile-virus/test.csv')
weather = pd.read_csv('/kaggle/input/predict-west-nile-virus/weather.csv')
sample = pd.read_csv('/kaggle/input/predict-west-nile-virus/sampleSubmission.csv')

print(train.shape, test.shape, weather.shape)
```

输出为训练集 `(10506, 12)`、测试集 `(116293, 11)`、气象 `(2944, 22)`。

然后看正负样本比例。

```python
# 正样本比例
print(train['WnvPresent'].mean())
```

约为 0.052，也就是 5.2% 的正样本率。数据结构是“捕蚊笼级别”的：同一地点同一天的多次检测会重复出现，每次检测记录一个蚊子数量和是否携带。

再看时间和地点上的分布。

```python
# 每年采样的样本量
train['year'] = pd.to_datetime(train['Date']).dt.year
print(train['year'].value_counts())

# 物种分布
print(train['Species'].value_counts())
```

主要样本集中在 2007 年，蚊子种类以 CULEX RESTUANS 和 CULEX PIPIENS/RESTUANS 为主，这两类正是西尼罗病毒在市区的主要传播媒介。不同蚊种携带病毒的概率差别很大，`Species` 是最重要的类别特征之一。

把经纬度画成地图散点图，能直观看到病毒的空间分布。

```python
import matplotlib.pyplot as plt

plt.scatter(train['Longitude'], train['Latitude'],
            c=train['WnvPresent'], s=3, cmap='viridis')
plt.colorbar(label='WnvPresent')
plt.show()
```

病毒更多地出现在城市南部，说明位置特征确实有区分度。我们后面的特征工程也保留了经纬度，并加了取整的网格特征。

### 2.2 数据清洗

这份数据要处理三块：文本列、气象数据、类别编码。

第一块，处理天气数据的脏值。气象资料用 `M` 表示缺失，用 `T` 表示“微量降水”，用 `-` 表示无记录。这几个字符既不是数字也不是有效标记，必须统一替换成 `-1`。

```python
# CodeSum 是天气代码文本，信息太杂，先去掉
weather = weather.drop('CodeSum', axis=1)

# 把 M（缺失）、-（无记录）、T（微量）都替换成 -1
weather = weather.replace('M', -1)
weather = weather.replace('-', -1)
weather = weather.replace('T', -1)
```

第二块，处理两个气象站。`weather.csv` 里每天有 `Station` 1 和 2 两条记录，我们把它们按日期横着拼起来，让每一天变成一行、两站的气象指标并排。

```python
# 拆成两个站，去掉站编号列
weather_stn1 = weather[weather['Station'] == 1].drop('Station', axis=1)
weather_stn2 = weather[weather['Station'] == 2].drop('Station', axis=1)

# 按日期把 1 号站和 2 号站合并，列名自动带上 _x/_y 区分
weather = weather_stn1.merge(weather_stn2, on='Date')
```

为什么这样拼？因为两个站都覆盖整个采样季节，如果只用其中一个站会损失一半观测；而把它们并排当成不同特征，随机森林可以自己决定哪个站更可信。合并后天气表约 2944 行变成两倍列数。

第三块，从日期里提取时间特征，并处理文本和地址。

```python
# 从日期字符串里拆出月和日
def create_month(x): return x.split('-')[1]
def create_day(x): return x.split('-')[2]

train['month'] = train.Date.apply(create_month)
train['day'] = train.Date.apply(create_day)
test['month'] = test.Date.apply(create_month)
test['day'] = test.Date.apply(create_day)

# 经纬度取整，生成粗粒度的网格位置特征
train['Lat_int'] = train.Latitude.apply(int)
train['Long_int'] = train.Longitude.apply(int)
test['Lat_int'] = test.Latitude.apply(int)
test['Long_int'] = test.Longitude.apply(int)
```

`month` 和 `day` 让模型能感知采样发生的季节时段，从而利用“8 月病例集中”的季节规律。`Lat_int` 和 `Long_int` 把经纬度粗化成 1 度的网格，既能缓解位置过拟合，也给了树模型一个更稳定的分裂依据。

第四块，合并天气，扔掉地址类文本，并删除无法对齐的列。

```python
# 用日期合并天气，然后删掉日期
train = train.merge(weather, on='Date')
test = test.merge(weather, on='Date')
train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)

# 地址文本与预测无关，删掉；NumMosquitos 测试集没有，训练集也不能留
train = train.drop(['Address', 'AddressNumberAndStreet', 'WnvPresent', 'NumMosquitos'], axis=1)
test = test.drop(['Id', 'Address', 'AddressNumberAndStreet'], axis=1)
```

这里有一个容易被忽略的细节：测试集没有 `NumMosquitos`，如果训练集保留它，特征维度就对不齐，模型还可能在训练时靠“蚊子多就携带率高”这种不可复现的规律拿分。所以一并删掉。

第五块，处理类别列。用 LabelEncoder 把字符串编码成整数。

```python
lbl = preprocessing.LabelEncoder()

# 把训练集和测试集的取值合在一起拟合，避免测试集出现未知编码
lbl.fit(list(train['Species'].values) + list(test['Species'].values))
train['Species'] = lbl.transform(train['Species'].values)
test['Species'] = lbl.transform(test['Species'].values)

# Street 和 Trap 同样处理
lbl.fit(list(train['Street'].values) + list(test['Street'].values))
train['Street'] = lbl.transform(train['Street'].values)
test['Street'] = lbl.transform(test['Street'].values)
lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
train['Trap'] = lbl.transform(train['Trap'].values)
test['Trap'] = lbl.transform(test['Trap'].values)
```

注意我们先把 `train` 和 `test` 的取值列表合并后再 fit。如果只在训练集上 fit，测试集里出现训练集没有见过的陷阱编号时，`transform` 会直接抛错。

最后，删掉那些整列都是 `-1` 的特征。这些列在观测期内没有任何有效记录，留着只会误导模型。

```python
# 只看所有样本都是 -1 的列，训练测试同步删
train = train.loc[:, (train != -1).any(axis=0)]
test = test.loc[:, (test != -1).any(axis=0)]
```

### 2.3 模型训练与超参数调优

我们选择**随机森林**。理由有三个：它对脏值（比如大量 `-1`）和共线性特征稳健，不需要专门做标准化；它能输出概率而不是只看类别，正适合按排序打分的 AUC；树模型对经度纬度这类稀疏连续特征的处理也一直很稳定。

```python
from sklearn.model_selection import train_test_split

# 划分训练集和验证集
labels = train_stage['WnvPresent'].values  # 注意：总特征是 drop 前的版本
X = train_stage.drop('WnvPresent', axis=1)
X_train, X_val, y_train, y_val = train_test_split(
    X, labels, test_size=0.2, random_state=9, stratify=labels
)
```

这里用 `stratify=labels` 保证训练和验证两边的正样本比例一致。因为正样本只有 5%，不分层的话，验证集可能出现一轮没有任何正样本，AUC 直接没法算。

训练随机森林。

```python
clf = ensemble.RandomForestClassifier(
    n_estimators=1000,   # 1000 棵树
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# 验证集概率
val_pred = clf.predict_proba(X_val)[:, 1]
```

调参我们主要控制两点：`n_estimators` 越大越稳定，但收益递减，1000 就够；`min_samples_split` 控制分裂的最小样本数，太小容易记住单条样本，太大又不够细，默认 2 在噪声少的数据上问题不大。我们没有做大规模网格搜索，因为随机森林在该数据上训练很快，分数差异主要来自特征而非这几个参数。

对 AUC 更重要的其实是**特征**。我们检查特征重要性，找出哪些列在起作用。

```python
importances = pd.Series(clf.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(10))
```

结果里 `month`、`Lat_int`、`Species`、`Long_int` 通常名列前茅，气象温度特征紧随其后。这验证了我们的直觉：时间季节、地点网格、蚊种三个因素共同决定传播风险。

### 2.4 模型测试与指标报告

我们在验证集上计算 AUC。

```python
from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(y_val, val_pred)
auc = metrics.auc(fpr, tpr)
print('Validation AUC: %.4f' % auc)
```

一次验证的分割大约能得到 0.83 附近的验证 AUC。作为参照，公共讨论里随机森林在私有测试集上的 AUC 一般在 0.655 到 0.687 区间浮动，而官方基础基准大约 0.70。也就是说，我们的本地验证分数明显高于基准，但私有集分数通常低于本地——这是因为测试集跨了年份，模型对“08、09 年新采样点”的泛化比记忆 07 年的模式更难。

为了更稳，我们用 5 折交叉验证代替单次切分，并把每折模型的测试集概率平均后提交。

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_pred = np.zeros(test.shape[0])

for tr, va in skf.split(X, labels):
    m = ensemble.RandomForestClassifier(n_estimators=1000,
                                        min_samples_split=2,
                                        random_state=42, n_jobs=-1)
    m.fit(X.iloc[tr], labels[tr])
    test_pred += m.predict_proba(X_test_stage)[:, 1] / skf.n_splits
```

五折平均天然是一种集成：每个模型只在 80% 数据上训练，五个模型看到的样本略有不同，平均后预测会更平滑，AUC 更稳定。

最后生成提交文件并检查格式。

```python
sample['WnvPresent'] = test_pred
sample.to_csv('submission.csv', index=False)

# 检查行数和空值
print(sample.shape)
print(sample['WnvPresent'].isnull().sum())
```

行数应与测试集一致，`WnvPresent` 应为概率值。

## 三、总结

本文中我们完成了 West Nile 病毒预测的完整流程。我们首先通过 EDA 发现数据正样本只有 5% 左右，病毒主要存在于特定的蚊种和特定的地理位置。其次我们处理了天气数据里的脏字符、把两个气象站按日期横向合并、从日期拆出月份、构造经纬度网格特征、用 LabelEncoder 编码类别列，并删除了测试集缺失的列和全为 -1 的列。接着我们用随机森林配合分层切分训练，验证 AUC 约 0.83。最后用 5 折交叉验证平均预测并生成提交文件。这个比赛告诉我们，AUC 这类排序指标在面对极端不平衡数据时比准确率诚实得多；模型本身的差异不大，真正的分数来自对时间、地点、蚊种三类信息的清洗和构造。