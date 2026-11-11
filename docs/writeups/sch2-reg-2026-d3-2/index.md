# 加州房价回归预测（sch2-reg-2026-d3-2）

## 一、比赛简介

### 1.1 比赛背景

加州房价（California Housing）是表格机器学习里很常见的一份数据。它把加州划分成很多小区块，每个区块记录收入、房龄、房间数量、入住人数和经纬度等信息，最后给出这个区块的房价中位数。因为房价是连续数值，所以这是一个典型的**回归**问题：模型要做的不是从几个类别里挑一个，而是猜一个具体的数。

这次比赛的名字是 `sch2-reg-2026-d3-2`，其中 `reg` 就是 regression（回归）的缩写。比赛提供的文件包括 `train.csv`、`test.csv` 和 `sample_submission.csv`。训练集带有目标列，测试集没有，我们要根据训练集学出的规律给测试集补上预测值。

这个任务看起来简单，实际上有几个难点。第一，特征的单位和范围差别很大：收入是几万美元的量级，房间数是几个，经纬度是几十度，模型如果不注意就会偏向数值大的特征。第二，地理位置对房价的影响非常强，而且不是简单的直线关系，同一个纬度上靠海和在内陆可能完全是两个世界。第三，房价分布里既有普通住宅，也有少量高价街区，极端值会把误差放大。第四，训练集和测试集来自同一份数据的划分，只有保证两边做完全相同的处理，线下验证才有意义。

### 1.2 比赛目标

训练集里每一行是一个街区样本，包含 `id`、若干特征列和目标列 `MedHouseVal`。从代码里用到的列来看，特征主要有：

- `MedInc`：街区收入中位数；
- `HouseAge`：房龄中位数；
- `AveRooms`：平均房间数；
- `AveBedrms`：平均卧室数；
- `AveOccup`：平均入住人数；
- `Latitude`、`Longitude`：纬度和经度。

目标列 `MedHouseVal` 是房价中位数，是一个连续值。测试集只有 `id` 和特征列，没有 `MedHouseVal`。我们要输出一个表格，包含测试集的 `id` 和预测出来的 `MedHouseVal`，行数和顺序要与 `sample_submission.csv` 对得上。

### 1.3 评价标准

比赛使用**均方根误差（RMSE，Root Mean Squared Error）**作为评价指标。它的算法是：先把每条样本的预测值和真实值相减，把差值平方，对所有样本求平均，最后开平方。写成公式就是：

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

其中 `y_i` 是真实房价，`\hat{y}_i` 是模型预测的房价，`n` 是样本数。RMSE 的单位和目标值一样，所以可以直观理解成「平均来看，我们的预测离真实房价差了多少」。因为误差先平方再平均，大误差会被放大，所以 RMSE 对「个别预测得特别离谱」的情况很敏感。这个指标**越低越好**，完美预测时等于 0。

材料里的代码也印证了这一点：XGBoost 版本用 `neg_root_mean_squared_error` 做交叉验证，CatBoost 版本直接计算 `sqrt(mean_squared_error(...))`，都是围绕 RMSE 展开的。

---

## 二、解题过程

整场比赛我们的主线是：先把数据看清，做最必要的清洗，再构造比例、距离和交互特征，最后用五折交叉验证训练 CatBoost，并把五折的测试预测平均后提交。下面按步骤展开。

### 2.1 数据探索（EDA）

我们首先把三张表读进来，确认数据规模、列名、类型、缺失情况和目标分布。这个阶段不急着建模，先把「数据长什么样」弄清楚，后面才不会在错误的假设上白费力气。

```python
# 导入表格处理、数值计算和画图要用的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 三个文件分别对应训练集、测试集和提交样例
train = pd.read_csv('/kaggle/input/sch2-reg-2026-d3-2/train.csv')
test = pd.read_csv('/kaggle/input/sch2-reg-2026-d3-2/test.csv')
submission = pd.read_csv('/kaggle/input/sch2-reg-2026-d3-2/sample_submission.csv')

# 先看行数和列数，确认训练集比测试集多了目标列
print('train shape:', train.shape)
print('test shape:', test.shape)

# 看前几行，确认每个特征大概是什么量级
print(train.head())

# info 能看出每列的数据类型，以及有没有非数值列需要额外处理
print(train.info())

# 缺失值数量：后面决定是删除还是填补
print(train.isnull().sum())

# 目标值的分布：重点看范围、均值和极端值
print(train['MedHouseVal'].describe())

# 相关性热力图帮我们快速扫一遍特征之间、特征和目标之间的关系
plt.figure(figsize=(10, 8))
sns.heatmap(train.corr(), cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# 经纬度是地理位置，单独画散点图，用颜色表示房价高低
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=train, x='Longitude', y='Latitude',
    hue='MedHouseVal', palette='viridis', s=15
)
plt.title('House Value by Location')
plt.show()
```

从打印结果我们知道，训练集和测试集的列基本对得上，训练集多出来的 `MedHouseVal` 正是目标列，测试集则保留了 `id` 用于提交。所有特征都是数值型，没有需要做词袋或者编码的文本列，这对我们是个好消息：可以直接进梯度提升树。

相关性热力图的作用不是让我们背下某个相关系数，而是先建立一个大致印象：哪些特征值得重点关注，哪些特征之间可能重复表达同一件事。这里要特别注意，**经纬度的相关系数看起来可能不高，但这不代表它们没用**。房价和位置的关系是典型的非线性关系，同一个经度上，靠海和在内陆的价格可能差很多，单纯的线性相关系数捕捉不到这种模式。所以我们在后面的特征工程里没有删掉经纬度，而是把它们转换成「到几个大城市中心的距离」这类更容易被树模型利用的形式。

目标值的分布也值得看一眼。房价是连续的正数，范围跨度比较大，少数高价街区会把整体均值拉高。因为 RMSE 会平方误差，这些高价样本一旦预测偏了，对总分的影响会比普通样本大得多。这提醒我们两件事：一是不要随便把高价样本当异常值删掉，它们也是真实要预测的对象；二是评估时要看整体 RMSE，而不能只看大多数普通房子的误差。

### 2.2 数据清洗

其次我们做数据清洗。这份数据本身是结构化的表格，清洗工作量不大，但有几个细节必须处理好，否则后面交叉验证的分数会虚高。

```python
# 测试集的 id 要单独保存，提交文件需要它，但它不能当特征
test_ids = test['id'].copy()

# 目标列单独拿出来，特征表里只保留真正的特征
y = train['MedHouseVal'].copy()
X = train.drop(columns=['id', 'MedHouseVal'])
X_test = test.drop(columns=['id'])

# 保证训练集和测试集的列顺序完全一致，避免预测时列错位
X_test = X_test[X.columns]

# 数值特征用训练集的中位数填补
# 用中位数而不是均值，是因为房价相关特征往往有极端值，均值容易被拉偏
for col in X.columns:
    median_value = X[col].median()
    X[col] = X[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)

# 检查重复行：如果训练集里有很多重复样本，交叉验证会高估模型
print('duplicated rows:', X.duplicated().sum())

# 这里故意不删除异常值：高价街区是真实样本，树模型对极端值也有一定鲁棒性
# 删除它们反而会让模型学不到高价区域的规律
```

我们做清洗时遵循三个原则。第一，`id` 只是编号，和房价没有因果关系，必须从特征里去掉；但测试集的 `id` 要留着，最后提交时按原顺序拼回去。第二，训练集和测试集必须用同一套列顺序、同一套缺失值填补规则，否则模型在测试集上看到的特征含义可能和训练时不一样。第三，缺失值用**训练集的中位数**填补，而不是用整个数据集的中位数，这样做是为了模拟真实场景：测试集的信息在训练时是看不到的。

关于异常值，我们没有采用「删掉最贵的百分之一」这种做法。房价数据里的高价街区不是录入错误，而是真实存在的市场现象；如果把它们删掉，模型会倾向于把房价往中间预测，遇到真正的高价样本时误差反而更大。梯度提升树本身对单点异常值不像线性回归那么敏感，所以我们选择保留全部样本，把重点放在特征表达和正则化上。

另外，材料里有一版代码用「纬度乘经度」造了一个 `Coord` 特征，但在测试集那一行写成了 `train['Latitude'] * test['Longitude']`，把训练集的纬度和测试集的经度拼在了一起。这种写法在行数不一致或者顺序不同时会直接报错，即使碰巧能跑通，特征的含义也已经错位了。我们在自己的流程里没有沿用这个写法，而是按行使用同一个样本自己的经纬度，或者干脆交给后面的距离特征来处理。

### 2.3 模型训练与超参数调优

接着我们进入特征工程和模型训练。原始特征是「平均房间数」「平均卧室数」这样的总量，它们本身有信息，但不够直接。我们想把它们组合成更有物理含义的指标，让模型少走弯路。

```python
def engineer_features(df):
    """把原始表格转换成更有表达力的特征表"""
    df_eng = df.copy()

    # 卧室占房间的比例：反映户型结构，比例比绝对数量更稳定
    df_eng['Bedrms_per_Room'] = df_eng['AveBedrms'] / df_eng['AveRooms']

    # 人均房间数和人均卧室数：把「房子大小」和「住多少人」结合起来
    # 同样大小的房子，住的人越多越拥挤，房价的规律也不同
    df_eng['Rooms_per_Person'] = df_eng['AveRooms'] / df_eng['AveOccup']
    df_eng['Bedrms_per_Person'] = df_eng['AveBedrms'] / df_eng['AveOccup']

    # 加州几个主要城市中心的经纬度
    # 直接给模型经纬度，树模型要切很多刀才能圈出一个区域；
    # 改成「到城市中心的距离」，位置信息一下子就变得可比较了
    cities = {
        'LA': (34.0522, -118.2437),
        'SF': (37.7749, -122.4194),
        'SD': (32.7157, -117.1611),
        'SJ': (37.3382, -121.8863),
    }

    for city, (lat, lon) in cities.items():
        # 这里用平面欧氏距离做近似：加州范围不算大，作为相对远近的度量足够
        df_eng[f'Dist_{city}'] = np.sqrt(
            (df_eng['Latitude'] - lat) ** 2 +
            (df_eng['Longitude'] - lon) ** 2
        )

    # 交互特征：让模型能同时利用「收入高」和「房子新/大」这两个条件
    # 树模型虽然能自己学组合，但显式给出来可以减少它需要的分裂次数
    df_eng['MedInc_x_HouseAge'] = df_eng['MedInc'] * df_eng['HouseAge']
    df_eng['MedInc_x_AveRooms'] = df_eng['MedInc'] * df_eng['AveRooms']

    return df_eng

# 训练集和测试集必须调用同一个函数，保证特征定义完全一致
X = engineer_features(X)
X_test = engineer_features(X_test)
```

我们做这些特征的出发点很简单：**让每一列都尽量接近人理解房价时用的概念**。比如「卧室占房间的比例」比单独看平均卧室数更能说明这是大户型还是小户型；「人均房间数」把房子大小和家庭人口放在一起，能反映居住拥挤程度；「到洛杉矶、旧金山、圣迭戈、圣何塞的距离」把两个冷冰冰的经纬度变成了「离大城市有多远」；「收入乘房龄」「收入乘房间数」则是让模型直接看到组合效应，而不是从零开始用很多次分裂去逼近。

严格来说，经纬度之间的距离应该用球面距离（Haversine）计算，但加州的范围相对地球半径很小，用平面欧氏距离做近似，排序上不会差太多，而且实现简单、计算快。我们更看重的是「相对远近」这个信息，而不是精确到公里的距离。

然后我们比较了三种梯度提升树：XGBoost、LightGBM 和 CatBoost。材料里三套代码都出现了。XGBoost 版本用 `train_test_split` 切出 20% 验证集，参数是 `n_estimators=2000`、`learning_rate=0.01`、`max_depth=6`、`subsample=0.8`、`colsample_bytree=0.8`，并用 `tree_method='hist'` 加速。LightGBM 版本在注释里试了 `num_leaves=64`、`n_estimators=3000` 和 `10000` 等配置。CatBoost 版本则用 5 折交叉验证，每折都记录验证 RMSE，并把五折的测试预测取平均。

我们最终把 CatBoost 五折作为主线，原因是它的评估最完整：每一折都有独立的验证集，训练时用 `early_stopping_rounds` 监控验证误差，测试集预测又是五折平均，比只切一次验证集更不容易受某一次随机划分的影响。

```python
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 五折交叉验证：每折都用不同的 80% 训练、20% 验证
# shuffle=True 打乱顺序，避免数据原本的排列带来偏差
FOLDS = 5
SEED = 42
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

# oof 用来保存「每条训练样本被当作验证样本时」的预测
# 这样每个预测都来自没见过这条样本的模型，评估才诚实
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = CatBoostRegressor(
        loss_function='RMSE',      # 直接优化比赛指标
        eval_metric='RMSE',
        iterations=2000,           # 上限给足，靠早停决定实际轮数
        learning_rate=0.03,        # 比 0.1 慢，但通常更稳
        depth=6,                   # 控制树的复杂度，防止记住噪声
        l2_leaf_reg=3,             # L2 正则，抑制叶子节点权重过大
        random_state=SEED,
        verbose=0,
        allow_writing_files=False  # 不在工作目录里生成 catboost_info
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,  # 验证集 100 轮不提升就停
        verbose=False
    )

    # 验证集预测：存进 oof 对应位置，后面统一算 OOF RMSE
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred

    # 测试集预测：每折都预测一次，最后取平均，降低单模型的波动
    test_preds += model.predict(X_test) / FOLDS

    score = np.sqrt(mean_squared_error(y_val, val_pred))
    scores.append(score)
    print(f'Fold {fold + 1} RMSE: {score:.5f}')

print(f'Average RMSE: {np.mean(scores):.5f}')
```

调参时我们没有一上来就跑大网格。原因很实际：五折交叉验证意味着每试一组参数要训练五个模型，CatBoost 在 2000 轮的上限下并不便宜，盲目网格搜索会把时间浪费在明显不合理的组合上。我们的做法是先定一个合理的起点，再看验证曲线微调。

具体来说，`learning_rate=0.03` 是一个比较折中的值：太大容易在验证集上抖动，太小则需要更多轮数。`iterations=2000` 只是上限，真正训练到多少轮由 `early_stopping_rounds=100` 决定，这样既不会欠拟合，也不会无意义地一直跑。`depth=6` 限制单棵树的最大深度，深度越大模型越容易记住训练集里的偶然细节。`l2_leaf_reg=3` 给叶子节点的输出加一点惩罚，配合早停一起控制过拟合。我们也没有对特征做标准化，因为树模型只看特征的排序和切分点，量纲不影响结果。

XGBoost 那一版则用了更小的学习率 `0.01` 和更多的树 `2000`，同时用 `subsample=0.8` 和 `colsample_bytree=0.8` 做行采样和列采样。这两种采样相当于每次建树只看一部分数据和一部分特征，能增加模型之间的差异性，降低过拟合。LightGBM 的 `num_leaves=64` 控制每棵树的叶子数，思路和 `max_depth` 类似，只是 LightGBM 用的是叶子数量而不是深度来限制复杂度。这些配置我们都在小规模验证里试过，最后选择 CatBoost 五折作为提交主线。

### 2.4 模型测试与指标报告

最后我们做模型测试和指标报告。因为测试集没有真实标签，我们不能直接算排行榜分数，所以主要看两样东西：训练集上的**折外预测（OOF）RMSE**，以及五折之间的分数波动。

```python
from sklearn.metrics import mean_squared_error

# OOF RMSE：把所有验证折的预测拼起来，相当于每个样本都由没见过的模型预测
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f'OOF RMSE: {oof_rmse:.5f}')

# 每一折的分数：看波动，而不只看平均值
for i, score in enumerate(scores, start=1):
    print(f'Fold {i}: {score:.5f}')
print(f'Fold mean: {np.mean(scores):.5f}, std: {np.std(scores):.5f}')

# 均值基线：如果模型连「全部预测成平均房价」都赢不了，说明特征没起作用
baseline_pred = np.full(len(y), y.mean())
baseline_rmse = np.sqrt(mean_squared_error(y, baseline_pred))
print(f'Baseline RMSE: {baseline_rmse:.5f}')

# 残差 = 真实值 - 预测值，看有没有系统性的高估或低估
residual = y - oof_preds
print(residual.describe())
```

我们看指标时分三层。第一层是 OOF RMSE 和均值基线的对比。均值基线就是「不管什么房子，都猜训练集的平均房价」，它的 RMSE 代表完全不利用特征时的水平。如果我们的模型只比基线好一点点，那说明特征工程没有带来真正的信息；只有明显低于基线，才能说明模型学到了收入、位置、房龄这些变量和房价之间的关系。

第二层是五折之间的波动。如果五折的 RMSE 都差不多，说明模型对数据划分不敏感，结果比较可信；如果某一折特别差，就要回头检查是不是那一折里高价样本特别多，或者某些特征在那一折里分布异常。我们不会只挑最好看的那一折来报告，而是看均值和标准差。

第三层是残差的分布。残差描述的是「模型在哪些样本上偏得厉害」。如果残差整体偏离 0，说明模型有系统性偏差；如果残差的尾部很长，说明少数高价或低价样本的误差很大，这和 RMSE 对极端值敏感的特点是吻合的。材料里没有保留具体的分数输出，所以我们在这里不写虚构的 RMSE 数字，而是把评估方法和判断标准讲清楚。

最后是生成提交文件。我们把五折模型对测试集的预测取平均，按 `sample_submission.csv` 的格式填回去，并做几项检查：

```python
# 以提交样例为模板，保证列名和行数完全符合比赛要求
submission = pd.read_csv('/kaggle/input/sch2-reg-2026-d3-2/sample_submission.csv')

# 五折的测试预测已经在上面的循环里累加并平均好了
submission['MedHouseVal'] = test_preds

# 提交前检查：行数、id 顺序、列名，任何一项不对都可能被判格式错误
assert len(submission) == len(test), '提交行数和测试集不一致'
assert submission['id'].equals(test['id']), 'id 顺序和测试集不一致'
assert list(submission.columns) == ['id', 'MedHouseVal'], '列名不符合要求'

submission.to_csv('submission.csv', index=False)
print('submission.csv saved')
```

这几行检查看起来琐碎，但比赛里因为提交文件行数不对、id 顺序错乱、列名拼错而白白丢分的情况并不少见。尤其是我们前面为了做特征工程对 `X_test` 做过列重排，提交时更要确认 `id` 还是原来的顺序。五折平均的预测比单模型稳定，因为不同折的模型在不同的数据子集上训练，犯的错误不完全一样，平均之后可以互相抵消一部分。

---

## 三、总结

本文中我们完成了 sch2-reg-2026-d3-2 加州房价回归比赛的完整流程。我们首先用 EDA 确认了数据是表格型回归任务，训练集比测试集多一个连续目标列 `MedHouseVal`，特征以收入、房龄、房间数、入住人数和经纬度为主，并用相关性热力图和地理位置散点图建立了对数据的直观认识。其次我们做了必要的数据清洗：把 `id` 从特征里去掉但保留测试集 id 用于提交，用训练集中位数填补缺失值，保证训练集和测试集列顺序一致，并且没有随意删除高价样本。接着我们构造了卧室占比、人均房间数、到四大城市中心的距离以及收入和房龄、房间数的交互特征，让原始列更接近人理解房价的方式。然后我们用五折交叉验证训练 CatBoost，配合 RMSE 损失、早停和 L2 正则控制过拟合，并把五折的测试预测取平均。最后我们用 OOF RMSE、均值基线和折间波动来评估模型，按提交样例生成文件，并检查行数、id 顺序和列名。

这个比赛给我们的最大启发是：**表格回归的分数往往不是靠一个神奇模型，而是靠对数据的理解**。经纬度本身只是两个数字，换成「到城市中心的距离」后才有明确含义；平均房间数和平均入住人数分开看很普通，组合成人均房间数后才能反映居住拥挤程度。另一个体会是评估要诚实：OOF 预测、五折波动和均值基线，这三样结合起来看，比盯着一个孤立的验证分数可靠得多。
