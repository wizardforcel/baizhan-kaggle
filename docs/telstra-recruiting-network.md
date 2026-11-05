# Telstra 网络故障严重程度预测（Telstra Recruiting Network）

## 一、比赛简介

### 1.1 比赛背景

Telstra 是澳大利亚最大的电信运营商。这个比赛是 Telstra 当年在 Kaggle 上举办的招聘赛，目的是从参赛者中挑选有潜力的数据科学人才，因此整道题被设计成「像在 Telstra 数据科学团队里工作」的样子：数据来自真实的网络服务日志，参赛者要判断一次网络故障到底有多严重。

这里的「严重程度」被分成三档：

- `0`：没有故障，或者只是一次很快恢复的小抖动；
- `1`：出现了若干次故障；
- `2`：严重故障，可能导致服务大面积中断。

难点主要有三个。第一，数据被拆成好几张表，主表只有地点和标签，故障特征分散在另外四张表里，必须按 `id` 拼起来。第二，特征几乎全是类别型，而且类别很多，比如 `location`、`log_feature`、`event_type` 都有成百上千种取值。第三，训练集只有 7381 条，样本量不大，模型很容易过拟合，所以调参和验证要格外小心。

### 1.2 比赛目标

比赛提供了六张表：

| 文件 | 内容 | 规模 |
| --- | --- | --- |
| `train.csv` | 训练集：`id`、`location`、`fault_severity` | 7381 行 |
| `test.csv` | 测试集：`id`、`location` | 11171 行 |
| `log_feature.csv` | 日志特征和出现次数 `volume` | 58671 行 |
| `event_type.csv` | 事件类型 | 31170 行 |
| `resource_type.csv` | 资源类型 | 21076 行 |
| `severity_type.csv` | 严重类型 | 18552 行 |

输入是一条网络故障记录对应的地点、日志特征、事件类型、资源类型和严重类型；输出是这条记录属于 0、1、2 三类的概率，提交文件里对应 `predict_0`、`predict_1`、`predict_2` 三列。测试集只给 `id` 和 `location`，其余信息都要从辅助表里查出来。

### 1.3 评价标准

比赛使用**多分类对数损失（multi-class log loss）**评分。用大白话说：对每条测试样本，我们给出三个类别的概率，评分只看「真实类别那一档的概率有多高」——概率越接近 1，扣分越少；概率越接近 0，扣分越重。最后把所有样本的扣分取平均，数值越小越好。

这个指标有两个直接后果。首先，我们不能只提交 0/1/2 的硬标签，必须提交概率，因为指标要用到概率。其次，「很自信却猜错」的惩罚远大于「不太确定但没猜错」，所以模型输出的概率要尽量校准，不能一味追求准确率。材料里的提交样例正是三列概率，这也印证了这一点。

---

## 二、解题过程

整场比赛的主线可以概括为：**先把六张表按 id 合并成一张宽表，再把类别列的前缀清理掉，接着用 LightGBM 和 CatBoost 两个梯度提升树模型做三分类，最后用贝叶斯优化和网格搜索调参，并输出概率。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步展开。

### 2.1 数据探索（EDA）

我们首先把六张表全部读进来，然后按 `id` 一张一张合并。这里不能用简单的上下拼接，因为同一个 `id` 在 `log_feature`、`event_type` 这些表里可能出现多行，必须用 `merge` 按共同的 `id` 列做连接：

```python
import pandas as pd
import numpy as np

# 依次读入比赛提供的六张表
train = pd.read_csv('../input/telstra-recruiting-network/train.csv.zip')
test = pd.read_csv('../input/telstra-recruiting-network/test.csv.zip')
feature = pd.read_csv('../input/telstra-recruiting-network/log_feature.csv.zip')
event = pd.read_csv('../input/telstra-recruiting-network/event_type.csv.zip')
resource = pd.read_csv('../input/telstra-recruiting-network/resource_type.csv.zip')
severity = pd.read_csv('../input/telstra-recruiting-network/severity_type.csv.zip')

# 主表只有 id/location/标签，其余信息都靠 id 连接补进来
# 使用 merge 而不是 concat，是因为辅助表里同一个 id 可能有多条记录
merge_1 = pd.merge(train, feature)
merge_2 = pd.merge(merge_1, event)
merge_3 = pd.merge(merge_2, resource)
merge_4 = pd.merge(merge_3, severity)

# 合并后一个 id 会膨胀成多行，先看看规模
print(merge_4.shape)   # 材料中输出 (61839, 8)
```

打印结果说明：原始训练集只有 7381 条，但合并后变成 61839 行，原因就是同一个故障记录对应多条日志特征和事件记录。这也提醒我们，后面必须先按 `id` 去重，否则同一条样本会被重复计算很多次，标签分布和训练都会失真。

接着我们看目标变量的分布。因为只有三档，最直观的是画饼图；再看取值较少的两个类别特征和数值列 `volume`，判断它们和标签有没有明显关系：

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 目标变量三类的占比：先确认类别是否均衡
plt.figure(figsize=(3, 3))
plt.pie(x=train.fault_severity.value_counts().values,
        labels=train.fault_severity.value_counts().index,
        colors=('aliceblue', 'lightsteelblue', 'pink'),
        autopct='%.2f%%')
plt.title('fault_severity')
plt.legend()
plt.show()

# resource_type 和 severity_type 取值少，画交叉柱状图看它们与标签的关系
less_level = train.iloc[:, [3, 4, 6]]
count = 1
for col in less_level.columns[:-1]:
    plt.subplot(1, 2, count)
    temp = pd.crosstab(less_level.fault_severity, less_level[col])
    temp1 = temp.T.stack().reset_index()
    sns.barplot(x=temp1[col], y=temp1[0],
                hue=temp1.fault_severity, palette='PuBu')
    count += 1

# volume 是数值列，按三类标签分别画直方图
sns.distplot(train[train.fault_severity == 0]['volume'],
             kde=False, label='NoFault', bins=3)
sns.distplot(train[train.fault_severity == 1]['volume'],
             kde=False, label='Several Faults', bins=3)
sns.distplot(train[train.fault_severity == 2]['volume'],
             kde=False, label='Serious Faults', bins=3)
plt.legend()
```

从饼图可以看到三类并不是完全等量，0 类样本最多，2 类相对少一些，这属于典型的不平衡三分类。`resource_type` 和 `severity_type` 的柱状图则告诉我们：某些资源类型在严重故障里出现得更多。`volume` 的直方图呈现明显的长尾，大部分样本的日志次数很低，少数样本的 `volume` 特别大。

我们最后做一轮常规检查：看数据类型、每列缺失值和非空数量。`location`、`log_feature`、`event_type` 的取值特别多，直接画图看不清，所以我们只确认它们不是数值列，后面统一交给树模型处理：

```python
# 看每列类型和非空数量，确认没有意外的数值/字符串混用
train.info()

# 逐列统计缺失值
print(train.isna().sum(axis=0))
# 材料中合并后的结果：所有列缺失数都是 0

# volume 的统计量：最小值 1，中位数 2，最大值 79，长尾很明显
print(train.describe())

# 偏度检查：volume 的偏度约 3.34，属于右偏，但没有超过 10
print(train.skew())
```

EDA 阶段我们得到三个关键认识：数据本身没有缺失值；训练集只有七千多条，样本很小；`volume` 有明显离群值。这三点直接决定了后面的清洗策略。

### 2.2 数据清洗

我们其次处理合并造成的重复行。既然一个故障记录会因为多条日志而膨胀，那么最合理的做法是每个 `id` 只保留一行。材料里保留下来的这一行是合并后该 `id` 的第一条日志，因此 `log_feature` 和 `volume` 只是该记录众多日志中的一条代表，这也是这份数据本身的一个局限。

```python
# 合并后同一 id 出现多行，保留一行即可，否则同一条样本会被重复训练
merge_4.drop_duplicates(subset='id', inplace=True)
print(merge_4.shape)   # 材料中输出 (7381, 8)

# 把 id 设为索引，防止它被模型当成一个普通特征
train = merge_4.set_index(merge_4.id).drop('id', axis=1)

# 类别列形如 "location 118"，树模型只关心编号 118，去掉前綴
# split(' ')[-1] 取空格后的最后一段，再整列替换成这一段
removal = train.iloc[:, [0, 2, 4, 5, 6]].apply(
    lambda i: i.apply(lambda x: x.replace(x, x.split(' ')[-1]))
)

# 清理后的五列类别特征 + volume + 标签，重新拼成训练表
train = pd.concat([removal, train.iloc[:, [3, 1]]], axis=1)
```

去掉前缀这一步看起来不起眼，但对树模型很重要。原来 `location 118` 是字符串，模型只能把它当成一个很长的类别名；去掉前缀后只剩 `118`，既方便统一处理，也方便后面转成 `category` 类型。测试集也要做同样的处理，否则训练和预测的特征格式对不上。

接着处理 `volume` 的离群值。我们使用 z-score：如果一个值偏离均值超过 3 个标准差，就认为它太极端，先标成缺失，再用剩余样本的均值填补。这样既没有被极端值带偏，也没有直接删掉样本：

```python
from scipy import stats

# 用 z-score 找出偏离均值超过 3 个标准差的 volume
x = np.abs(stats.zscore(train.volume)) < 3
train.volume = np.where(x, train.volume, np.nan)   # 异常值先标记为缺失
print(train.isna().sum())   # 材料中 volume 出现 104 个缺失

# 数据量不大，直接拿剩余样本的均值填补，避免删样本
train.volume = train.volume.replace(
    np.nan,
    train.volume.sum() / (len(train.volume) - train.volume.isna().sum())
)
```

这样处理后，`volume` 的 104 个极端值被均值替代，训练表重新变成零缺失。我们没有做标准化或取对数，原因是后面用的是树模型，树模型只关心特征的大小顺序，不关心具体尺度，所以这一步可以省掉。

### 2.3 模型训练与超参数调优

我们接着把五个类别列统一转成 pandas 的 `category` 类型，然后把数据切成训练集和验证集。这一步很关键：测试集没有标签，我们只能从训练集里留出一部分来模拟「未知数据」，否则调参就是盲调。

```python
from sklearn.model_selection import train_test_split

# 前五列是类别特征，转成 category 后 LightGBM 会按类别处理
train.iloc[:, 0:5] = train.iloc[:, 0:5].astype('category')

x = train.iloc[:, 0:-1]    # 特征
y = train.fault_severity   # 三分类标签

# 留 20% 作验证集，固定 random_state 保证每次划分一致
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=1
)
```

然后我们跑两个基线模型：LightGBM 和 CatBoost。它们都是「梯度提升树」，可以理解成很多棵决策树排着队，后一棵专门纠正前一棵的错误。两者都原生支持类别特征，不需要我们手动做独热编码，非常适合这种「类别多、样本少」的表格数据：

```python
from lightgbm import LGBMClassifier as lgbc
from catboost import CatBoostClassifier as cbc
from sklearn.metrics import accuracy_score

# LightGBM 基线：先给一组不算激进的参数
gbm_b = lgbc(objective='multiclass', max_depth=6,
             reg_lambda=3.0, random_state=1)
gbm_b = gbm_b.fit(X=x_train, y=y_train,
                  eval_set=(x_val, y_val),
                  early_stopping_rounds=50, verbose=10)

# CatBoost 基线：直接告诉它哪些列是类别特征
cat_b = cbc(objective='MultiClass', learning_rate=0.1,
            n_estimators=100, random_state=1)
cat_b = cat_b.fit(X=x_train, y=y_train,
                  eval_set=(x_val, y_val),
                  cat_features=np.where(x_train.dtypes != np.float)[0],
                  early_stopping_rounds=50, verbose=10)

# 两个模型都在验证集上算准确率，先摸个底
for model in [gbm_b, cat_b]:
    y_pred = model.predict(x_val)
    print(type(model).__name__, accuracy_score(y_val, y_pred))
```

材料里的基线结果是：LightGBM 验证准确率约 `0.700068`，CatBoost 约 `0.702099`，两者非常接近。这说明数据本身的信息量有限，换模型带来的差异不大，真正能拉开差距的是特征和调参。

我们先对 LightGBM 做贝叶斯优化。贝叶斯优化和网格搜索的区别是：网格搜索把所有参数组合挨个试一遍，贝叶斯优化则会根据已经试过的结果，猜测「下一组参数大概在哪个方向更好」，因此用更少的次数就能找到不错的组合：

```python
import lightgbm as lgb
from bayes_opt import BayesianOptimization

# 把训练数据包装成 LightGBM 的 Dataset 对象
train_set = lgb.Dataset(data=x_train, label=y_train)

def lgb_eval(learning_rate, n_estimators, max_depth, num_leaves,
             min_data_in_leaf, bagging_fraction, bagging_freq,
             feature_fraction, lambda_l1, lambda_l2):
    # 优化器传入的是连续小数，树模型要求整数，所以四舍五入
    params = {'objective': 'multiclass', 'num_class': 3,
              'seed': 1, 'verbose': -1}
    params['learning_rate'] = learning_rate
    params['n_estimators'] = round(n_estimators)
    params['max_depth'] = round(max_depth)
    params['num_leaves'] = round(num_leaves)
    params['min_data_in_leaf'] = round(min_data_in_leaf)
    params['bagging_freq'] = round(bagging_freq)
    params['bagging_fraction'] = min(bagging_fraction, 1.0)
    params['feature_fraction'] = min(feature_fraction, 1.0)
    params['lambda_l1'] = lambda_l1
    params['lambda_l2'] = lambda_l2

    # 5 折交叉验证，返回验证集 logloss 的相反数（优化器只做最大化）
    cv_result = lgb.cv(params, train_set, nfold=5,
                       early_stopping_rounds=50, verbose_eval=50)
    return -(min(cv_result['valid multi_logloss-mean']))

lgb_BO_1 = BayesianOptimization(
    lgb_eval,
    {'learning_rate': (0.05, 0.2), 'n_estimators': (10, 500),
     'max_depth': (3, 8), 'num_leaves': (7, 255),
     'min_data_in_leaf': (18, 22), 'bagging_fraction': (0.8, 1),
     'bagging_freq': (1, 5), 'feature_fraction': (0.8, 1),
     'lambda_l1': (0.1, 3), 'lambda_l2': (0.1, 3)},
    random_state=1)
lgb_BO_1.maximize()
```

贝叶斯优化把 LightGBM 的验证 logloss 从基线的 `0.634234` 降到 `0.632163`。之后我们还试过用网格搜索单独调学习率和树的数量，但效果反而略差：只用网格搜索调参得到 `0.636552`，再结合交叉验证得到 `0.633481`。材料里的结论是「所有参数一起用贝叶斯优化搜索更好」，所以我们把主线放在这条路上。

CatBoost 这边我们试了三种配置：带类别权重的 Ordered 模式、带类别权重的 Plain 模式，以及不带类别权重的 Ordered 模式。其中「按类别平衡权重」是为了缓解 0/1/2 三类样本不均衡的问题。最后我们选出一组参数，用早停重新训练，让模型自己在验证集上决定什么时候停：

```python
import catboost as cb

# 训练集和验证集都要包装成 CatBoost 的 Pool，并声明类别列下标
train_pool = cb.Pool(data=x_train, label=y_train,
                     cat_features=np.where(x_train.dtypes != np.float)[0])
val_pool = cb.Pool(data=x_val, label=y_val,
                   cat_features=np.where(x_val.dtypes != np.float)[0])

# 代入贝叶斯优化搜到的参数，先用 300 棵树训练
cat_f2 = cbc(objective='MultiClass',
             learning_rate=0.05716226472904498,
             n_estimators=300, max_depth=9,
             reg_lambda=1.4979851182199662, random_state=1)
cat_f2 = cat_f2.fit(X=train_pool, eval_set=val_pool,
                    use_best_model=True,
                    early_stopping_rounds=50, verbose=10)

# 早停会在验证集指标不再变好时停下，保留最好那一轮
print('best_iteration', cat_f2.best_iteration_)
print('best_score', cat_f2.best_score_)
```

早停的结果是最好一轮出现在第 191 轮，CatBoost 自动把模型收缩到前 192 棵树，验证集上的 MultiClass 指标为 `0.6086256973`。这也说明 300 棵树给多了，真正有用的大约只有三分之二。

### 2.4 模型测试与指标报告

我们然后在验证集上做最终评估。这里要说明一点：材料里既打印了准确率，也打印了 `log_loss`，但那段 `log_loss` 是把预测结果先转成 one-hot 硬标签再计算的，数值会达到 10 左右，和 Kaggle 用概率计算的对数损失不是一回事。因此我们主要看两个更可靠的数字：验证准确率，以及 CatBoost 自己早停时报告的 MultiClass 损失。

```python
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder

# 训练集和验证集上的准确率，用来判断有没有过拟合
train_acc_f2 = cat_f2.score(x_train, y_train)
y_pred = cat_f2.predict(x_val)
val_acc_f2 = accuracy_score(y_val, y_pred)

print('train_acc_final2: ', train_acc_f2)      # 材料中 0.7854
print('val_acc_final2: ', val_acc_f2)          # 材料中 0.7062
print('best validation score: ', cat_f2.best_score_)
```

最终 CatBoost 在训练集上的准确率约 `0.7854`，验证集约 `0.7062`，两者相差约 8 个百分点，说明存在一定过拟合，但还在可接受范围。和基线相比，验证准确率从 `0.702099` 提升到 `0.706161`，提升不算大，符合「这份数据本身信息有限」的判断。

我们还看了最终模型的特征重要性，确认模型有没有学到合理的东西：

```python
# 取出每个特征的重要性分数，并从大到小排序
cat_f2_im = cat_f2.feature_importances_
im_ind = np.argsort(cat_f2_im)[::-1]

for f in range(x_train.shape[1]):
    print(f + 1, x.columns[im_ind[f]], cat_f2_im[im_ind[f]])
```

材料里的排序是：`location` 约 25.79、`log_feature` 约 19.84、`severity_type` 约 16.59、`resource_type` 约 13.44、`volume` 约 13.11、`event_type` 约 11.23。六个特征的重要性差距不算特别悬殊，说明模型是综合多列信息在做判断，而不是只依赖某一个字段。其中地点和日志特征排前两位，也符合「不同地区的网络故障模式不同」的直觉。

最后生成提交文件。因为比赛按概率算对数损失，我们必须用 `predict_proba` 而不是 `predict`，并且保持测试集的顺序与 `id` 对齐：

```python
# predict_proba 输出三个类别的概率，正好对应提交文件的三列
y_pred = cat_f2.predict_proba(test)

submission = pd.DataFrame(
    y_pred, columns=['predict_0', 'predict_1', 'predict_2']
)
re_test = test.reset_index()   # 把原来的 id 取回来
submission = pd.concat([re_test['id'], submission], axis=1)
submission.to_csv('submission.csv', index=0)
```

提交前我们检查了三件事：行数是否等于测试集的 11171 行、三列概率是否每行加起来约等于 1、`id` 是否和测试集完全一致。这三步看起来琐碎，却是比赛里最容易翻车的地方。

---

## 三、总结

本文中我们完成了 Telstra 网络故障严重程度预测的完整流程。我们首先把六张表按 `id` 合并，发现合并后样本从 7381 条膨胀到 61839 行，于是按 `id` 去重恢复成一条记录一行。其次我们做了 EDA，确认三类标签不均衡、`volume` 有明显长尾、类别特征取值很多。接着我们清理了类别列的前缀，用 z-score 处理 `volume` 的 104 个离群值，并把类别列转成 `category` 类型。然后我们用 LightGBM 和 CatBoost 做基线，验证准确率分别在 0.700 和 0.702 左右；再用贝叶斯优化搜索参数，把 LightGBM 的验证 logloss 从 0.6342 降到 0.6322。最后我们用早停训练 CatBoost，得到训练准确率约 0.7854、验证准确率约 0.7062，并按三列概率格式生成提交文件。

这个比赛给我们最大的启发是：**表格比赛里，数据和验证比模型花哨程度更重要**。这份数据只有七千多条、特征几乎全是高基数类别，换更复杂的模型提升很有限；反而是去重、清理前缀、处理离群值、留出可靠的验证集这些「笨功夫」，决定了模型能不能稳定地泛化。另一个细节是评价指标——对数损失要的是概率，所以提交时必须输出 `predict_proba`，并且要理解「自信地猜错」会被重罚，这样才不会在最后一步丢掉不该丢的分。
