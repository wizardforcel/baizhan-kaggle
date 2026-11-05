# TalkingData 移动用户人口属性预测

## 一、比赛简介

### 1.1 比赛背景

TalkingData 是中国的一家移动大数据平台，它会收集手机上的应用安装、使用事件，以及手机品牌、型号等设备信息。这个比赛要求参赛者**根据用户的行为数据，推测用户的性别和年龄段**。在广告投放里，知道用户是「男性 25 到 34 岁」还是「女性 19 到 23 岁」，直接决定了应该推什么广告、出什么价。

比赛给出的数据非常贴近真实业务：设备表记录手机品牌和型号，事件表记录每台设备上发生过的应用事件，应用标签表记录每个应用被打上的类别标签。我们要把这些「行为碎片」拼成一个用户的画像。

这个任务的难点在于：**数据是多张表、稀疏且高维的**。一台设备可能装了几十个应用，一个应用又对应多个标签；如果直接展开成宽表，特征维度会迅速膨胀到几万列。另一个难点是标签不是单一的二分类，而是「性别 × 年龄段」组合出来的多个类别，属于多分类问题。

### 1.2 比赛目标

输入包括六张表：

- `gender_age_train.csv`：训练集，包含 `device_id`、`gender`、`age` 和 `group`，其中 `group` 是我们要预测的组合标签；
- `gender_age_test.csv`：测试集，只有 `device_id`；
- `phone_brand_device_model.csv`：设备 ID 对应的手机品牌和型号；
- `events.csv`：事件 ID、设备 ID、时间戳、经纬度；
- `app_events.csv`：事件 ID、应用 ID、是否激活；
- `app_labels.csv` 和 `label_categories.csv`：应用 ID 对应的标签，以及标签的中文类别名。

输出是为每个测试设备预测它属于哪个 `group`。从材料里的建模代码可以看到，`num_class = 12`，也就是共有 12 个「性别 + 年龄段」组合类别。

### 1.3 评价标准

比赛使用**多分类对数损失（Multi-class Log Loss）**作为评价指标。模型对每个样本输出 12 个类别的概率，log loss 会惩罚「给正确答案分配的概率太低」的情况。公式是：

```
LogLoss = - (1/N) * Σ log( p(真实类别) )
```

这个指标**越低越好**。它和准确率不同：准确率只看「猜没猜中」，log loss 还看「猜中的时候有多自信」。如果一个模型总是给出 0.99 的把握却猜错，log loss 会非常难看。因此我们不仅要预测对，还要让概率校准得合理。

---

## 二、解题过程

我们按 EDA、数据清洗、模型训练与超参数调优、模型测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先读入应用标签表，看看标签体系有多大：

```python
import pandas as pd
import numpy as np

app_labels = pd.read_csv('../input/app_labels.csv')
label_categories = pd.read_csv('../input/label_categories.csv')

# 看行数和列数：app_labels 是「应用-标签」的多对多关系
print(app_labels.shape)
print(app_labels.nunique())

# label_categories 把标签 ID 翻译成人类可读的类别名
print(label_categories.shape)
print(label_categories.head())
```

从材料的输出可以看到，`app_labels` 有约 46 万行，包含 11 万个不同的应用和 507 个不同的标签；`label_categories` 有 930 行，其中 835 个是不同的类别名。也就是说，一个应用可以被打上多个标签，一个标签也可以被多个应用共用。

接着看最常出现的标签：

```python
# 把标签 ID 翻译成类别名，再统计每个类别出现的次数
apps_labels_cats = pd.merge(app_labels, label_categories, how='left')
print(apps_labels_cats['category'].value_counts().head(25))
```

输出里排名靠前的有 `Industry tag`、`Custom label`、`Tencent`、`game`、`Property Industry 2.0` 等，说明标签体系里既有很粗的行业分类，也有很细的游戏类型。这个信息告诉我们：**应用标签可以作为用户的兴趣画像**，比如一个用户装了很多游戏类应用，就更可能是年轻男性。

然后我们看设备品牌和型号：

```python
phone = pd.read_csv('../input/phone_brand_device_model.csv')
print(phone.shape)
print(phone['phone_brand'].value_counts().head(10))

# 同一个 device_id 可能重复出现，先去重，避免后面 merge 时行数翻倍
phone = phone.drop_duplicates('device_id', keep='first')
```

材料里提到设备表存在重复的 `device_id`，如果不去重，后面把品牌拼到训练集时会出现一对多，样本数被悄悄放大。

最后看训练集和测试集的规模：

```python
gatrain = pd.read_csv('../input/gender_age_train.csv', index_col='device_id')
gatest = pd.read_csv('../input/gender_age_test.csv', index_col='device_id')

print(gatrain.shape, gatest.shape)
print(gatrain['group'].value_counts())
```

从材料里的特征矩阵可以确认：训练集约 74,645 台设备，测试集约 112,071 台设备，测试集比训练集还大。这也是为什么特征工程必须用稀疏矩阵——如果做成稠密的 7 万 × 2 万表格，内存会直接爆掉。

### 2.2 数据清洗

这一节的核心是**把多张表拼成「一台设备一行」的特征矩阵**。我们分三块做：品牌、型号、应用、标签。

品牌和型号使用 one-hot 的稀疏表示。我们不去新建一个巨大的 DataFrame，而是直接构造 CSR 稀疏矩阵：

```python
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

# 给训练集和测试集各加一个行号，后面构造稀疏矩阵时要用
gatrain['trainrow'] = np.arange(gatrain.shape[0])
gatest['testrow'] = np.arange(gatest.shape[0])

# 品牌编码：把字符串品牌转成 0,1,2,... 的整数
brandencoder = LabelEncoder().fit(phone['phone_brand'])
phone['brand'] = brandencoder.transform(phone['phone_brand'])
gatrain['brand'] = phone['brand']
gatest['brand'] = phone['brand']

# 构造稀疏矩阵：第 i 行、第 brand 列填 1
Xtr_brand = csr_matrix(
    (np.ones(gatrain.shape[0]), (gatrain['trainrow'], gatrain['brand']))
)
Xte_brand = csr_matrix(
    (np.ones(gatest.shape[0]), (gatest['testrow'], gatest['brand']))
)
```

`csr_matrix((data, (row_ind, col_ind)))` 这个构造方式很实用：它不要求我们先把整张表填好，只需要告诉它「哪些位置是 1」。品牌只有 131 种，所以品牌矩阵是 74,645 × 131，非常小。

型号特征同理，但要把「品牌 + 型号」拼起来再做编码，因为不同品牌可能有同名型号：

```python
# 品牌和型号拼接，避免不同品牌下的同名型号被当成同一个
m = phone['phone_brand'].str.cat(phone['device_model'])
modelencoder = LabelEncoder().fit(m)
phone['model'] = modelencoder.transform(m)

gatrain['model'] = phone['model']
gatest['model'] = phone['model']

Xtr_model = csr_matrix(
    (np.ones(gatrain.shape[0]), (gatrain['trainrow'], gatrain['model']))
)
Xte_model = csr_matrix(
    (np.ones(gatest.shape[0]), (gatest['testrow'], gatest['model']))
)
```

型号有 1667 种，比品牌细得多。手机型号和价格、用户群有很强相关性，比如高端机型用户可能年龄偏大、收入偏高，所以这是很有价值的特征。

在构造特征时我们还做了两个小清洗。第一，设备表里同一台设备的记录可能不止一条，如果重复行在品牌和型号上不一致，我们用 `drop_duplicates('device_id', keep='first')` 丢掉后面出现的记录，避免同一设备拥有多个身份。第二，`app_events` 中 `is_active` 字段有时会是空值，我们在读取时强制转成 bool，这样空值会被当作 False 处理，避免后面聚合时出现 NaN 导致行数膨胀。这两个清洗步骤都很轻，但能显著减少特征矩阵里的噪声。

应用特征需要先把 `events` 和 `app_events` 连起来：

```python
events = pd.read_csv('../input/events.csv', parse_dates=['timestamp'], index_col='event_id')
appevents = pd.read_csv('../input/app_events.csv',
                        usecols=['event_id','app_id','is_active'],
                        dtype={'is_active': bool})

# 应用 ID 编码
appencoder = LabelEncoder().fit(appevents['app_id'])
appevents['app'] = appencoder.transform(appevents['app_id'])
napps = len(appencoder.classes_)

# 通过 event_id 把 device_id 接到 app_events 上，
# 再按「设备-应用」聚合，得到每台设备装过哪些应用
deviceapps = (appevents.merge(events[['device_id']], how='left',
                              left_on='event_id', right_index=True)
              .groupby(['device_id','app'])['app'].agg(['size'])
              .reset_index())
```

这段代码是整个特征工程的核心。`events` 里一行是一次事件，`app_events` 里一行是「某次事件涉及哪个应用」，两者用 `event_id` 连接后，就能知道「哪台设备用过哪个应用」。`groupby(['device_id','app'])` 去重，保证一台设备对一个应用只算一次。

最后把应用标签也拼进来：

```python
# 只保留在 app_events 里出现过的应用，减少无用标签
applabels = app_labels.loc[app_labels['app_id'].isin(appevents['app_id'].unique())].copy()
applabels['app'] = appencoder.transform(applabels['app_id'])

labelencoder = LabelEncoder().fit(applabels['label_id'])
applabels['label'] = labelencoder.transform(applabels['label_id'])
nlabels = len(labelencoder.classes_)

# 设备 -> 应用 -> 标签，按「设备-标签」聚合
devicelabels = (deviceapps[['device_id','app']]
                .merge(applabels[['app','label']])
                .groupby(['device_id','label'])['app'].agg(['size'])
                .reset_index())
```

标签只有 492 个，比应用的 19,237 个少得多，所以标签特征是「低维但语义强」的补充。

### 2.3 模型训练与超参数调优

我们把四组稀疏特征横向拼起来：

```python
# hstack 按列拼接稀疏矩阵，format='csr' 方便后面做逻辑回归
Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format='csr')
Xtest = hstack((Xte_brand, Xte_model, Xte_app, Xte_label), format='csr')

print(Xtrain.shape, Xtest.shape)
```

材料里的结果显示，拼完之后训练集是 74,645 × 21,527，非零元素约 270 万个。这个规模用稠密矩阵要 74,645 × 21,527 × 8 字节，超过 12 GB；用稀疏矩阵只存非零值，内存占用小一个数量级。

模型方面，材料的主线是**正则化逻辑回归**（`LogisticRegression`），它对高维稀疏特征很合适：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# 把 group 字符串编码成 0~11 的整数
targetencoder = LabelEncoder().fit(gatrain['group'])
y = targetencoder.transform(gatrain['group'])
nclasses = len(targetencoder.classes_)

# 多分类逻辑回归；C 越小正则越强
clf = LogisticRegression(
    C=1.0,
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=200,
    n_jobs=-1
)
clf.fit(Xtrain, y)
```

为什么逻辑回归能打败更复杂的模型？因为这里的特征维度远大于样本量（21,527 列 vs 7 万行），而且特征非常稀疏。逻辑回归带 L2 正则，相当于给每个特征系数加一个「不要太大」的约束，能有效防止过拟合。材料里作者也试过 XGBoost 的 `gblinear`，结果反而略差——这提醒我们：**不是模型越复杂越好，特征和正则化匹配才是关键。**

关于超参数，主要调 `C`（正则强度的倒数）和 `max_iter`（迭代次数）。`C` 太大，模型会记住训练集的噪声；`C` 太小，模型欠拟合。我们用验证集 log loss 来选：

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for C in [0.1, 0.3, 1.0, 3.0]:
    losses = []
    for tr, va in skf.split(Xtrain, y):
        model = LogisticRegression(C=C, multi_class='multinomial',
                                   solver='lbfgs', max_iter=200)
        model.fit(Xtrain[tr], y[tr])
        proba = model.predict_proba(Xtrain[va])
        losses.append(log_loss(y[va], proba))
    print(C, np.mean(losses))
```

这段代码会打印每个 `C` 对应的平均 log loss，我们选最低的那个。注意 `predict_proba` 输出的概率要按 `model.classes_` 的顺序理解，sklearn 会自动对齐，不需要我们手动排序。

### 2.4 模型测试与指标报告

确定 `C` 后，我们用全部训练数据重新拟合，再对测试集输出概率：

```python
clf = LogisticRegression(C=best_C, multi_class='multinomial',
                         solver='lbfgs', max_iter=200)
clf.fit(Xtrain, y)

# 输出 12 类的概率，而不是硬分类
proba = clf.predict_proba(Xtest)

# 按提交格式写：第一列是 device_id，后面 12 列是各类概率
submission = pd.DataFrame(proba, columns=targetencoder.classes_)
submission.insert(0, 'device_id', gatest.index)
submission.to_csv('submission.csv', index=False)
```

提交 log loss 类比赛时，**一定要提交概率而不是 0/1 标签**。如果提交硬分类，log loss 会变成无穷大（因为给真实类别的概率是 0），排行榜直接判负。

指标报告方面，我们报告交叉验证的平均 log loss。材料里的 XGBoost 实验在 5 折下大约在 2.3 附近，逻辑回归的基线与之接近；具体数值会随特征组合和 `C` 变化，我们不把某一次的结果当作绝对结论。更重要的是看「特征有没有用」：去掉应用标签，log loss 会变差；去掉手机型号，也会变差。这说明品牌、型号、应用、标签四类特征是互补的。

---

## 三、总结

本文中我们完成了 TalkingData 移动用户人口属性预测的完整流程。我们首先用 EDA 摸清了七张表的关系：应用标签有约 46 万行、11 万个应用、507 个标签，设备表存在重复 ID 需要去重；其次用稀疏矩阵把品牌、型号、应用、标签四类特征拼成 21,527 维的「一台设备一行」矩阵；接着用带 L2 正则的多分类逻辑回归建模，并用分层 5 折交叉验证调正则强度；最后输出 12 类概率生成提交文件。

这个比赛给我们的最大启发是：**高维稀疏场景下，简单模型 + 好特征往往比复杂模型更有效。** 21,527 维特征、7 万条样本，逻辑回归靠正则化就能稳住；而更复杂的树模型反而容易在稀疏 one-hot 上过拟合。把多张表正确连接、把行为转成画像，才是这类用户画像比赛真正的胜负手。做好稀疏矩阵的构建，能让内存占用和训练时间都可控，是工程落地的关键一步。