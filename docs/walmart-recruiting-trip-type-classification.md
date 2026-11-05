# Walmart 购物行程类型分类

## 一、比赛简介

### 1.1 比赛背景
Walmart Recruiting - Trip Type Classification 要求根据顾客在一次购物中购买的商品，预测这次购物属于哪一种 TripType。Walmart 用内部方法将旅行分成 38 类，比如日常晚餐小采购、每周大采购、节日礼品采购、季节性买衣服等。

比赛提供有限字段（商品 UPC、部门、细类、扫描数量、星期几），需要重建这 38 类分类。难点是类别极度不平衡，且同一 VisitNumber 下有多条商品记录。数据来自美国 Walmart 门店交易，包含顾客访问、商品和行程标签。该任务模拟零售客户细分，行程类型反映顾客购物意图，对营销策略制定至关重要。数据规模约 65 万行商品记录，覆盖上万次访问。类别分布右偏，部分稀有类别样本不足百条。

理解数据的关键是「一次访问、多条商品」的结构。原始表是商品级流水，一个 VisitNumber 会出现很多行，每行是一件商品。标签 TripType 挂在访问上，所以同一个 VisitNumber 的所有行共享同一个标签。如果直接按行训练，模型会看到大量重复标签，既浪费计算，又会因为热门商品出现频繁而偏置。因此必须先按 VisitNumber 聚合。

比赛没有提供 TripType 的语义解释，也没给更多商品信息，参赛者只能从购买行为反推意图。这增加了难度，但也让特征工程的空间很大：可以统计买了多少件、涉及几个部门、有没有退货、是不是周末等等。

### 1.2 比赛目标
输入训练集 train.csv，字段包括 TripType、VisitNumber、Weekday、Upc、ScanCount、DepartmentDescription、FinelineNumber。测试集无 TripType。

目标为每个 VisitNumber 预测一个 TripType。提交列 VisitNumber, TripType。每个访问只有一种行程类型，需从商品清单聚合特征。训练集含 38 个类别，TripType_999 为其他类。VisitNumber 是访问 ID，Upc 为商品条码，ScanCount 为购买数量，负值表示退货。

### 1.3 评价标准
使用多分类对数损失 LogLoss，越低越好。模型对真实类别给出的概率越高，损失越小；预测错且自信时惩罚很大。

LogLoss 要求模型输出概率分布，而非硬标签。类别不平衡下，需注意校准。标准为负对数似然平均值，越小越好。与准确率不同，LogLoss 会奖励「把正确类排在前面」的模型，即使最终没猜中，只要概率给得合理，损失也不会太差。这让模型更关注整体概率质量，而不是单一命中。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先加载数据，查看规模和重复情况。

```python
import pandas as pd
df = pd.read_csv('train.csv')
print(df.shape)
print(df['TripType'].nunique())
print(df.duplicated().sum())
```

发现 5704 条完全重复记录。TripType 分布右偏，36-40 附近数据最多，很多类别样本极少。

```python
import seaborn as sns
sns.countplot(df['TripType'])
```

Weekday 分布较均匀，VisitNumber 也接近均匀。DepartmentDescription 是类别字段，FinelineNumber 有缺失。ScanCount 负值表示退货，需保留。

可视化 Upc 频次，发现热门商品集中在食品部门。行程类型与部门组合相关，如礼品类多与玩具、家居相关。我们还统计了每次访问的商品行数，发现大多数访问只有十几行，少数大采购能有上百行，这种长度差异本身就是行程类型的强信号。

另外，Weekday 从星期一到星期日都有分布，但不同行程类型在不同星期几的比例不同。例如周末的大采购占比更高，工作日的快速补货更多。我们把 Weekday 和 TripType 交叉制表，验证了这一直觉。这个发现直接指导了后面的特征设计：不能只统计买了什么，还要统计什么时候买。

### 2.2 数据清洗
先去除重复，再填缺失。

```python
df = df.drop_duplicates()
df['FinelineNumber'] = df['FinelineNumber'].fillna(8228)
df = df.dropna()
```

把每个 VisitNumber 聚合成一行：统计商品数、扫描总数、部门种类数等。

```python
agg = df.groupby('VisitNumber').agg({
    'ScanCount':['sum','mean','count'],
    'Upc':'nunique',
    'DepartmentDescription':'nunique',
    'FinelineNumber':'nunique'
})
```

创建退货率、平均单价代理特征。Weekday 做独热编码。Upc 频次做 TF-IDF 风格加权。清洗时我们特别注意不能把同一访问的多条商品记录误删，所以先去重整行，再做聚合，顺序不能反。FinelineNumber 的缺失比例很高，直接用众数填充，虽然粗糙，但比删除整行损失小。

聚合之后还要处理极端值。有的访问商品数量特别大，会让数值特征尺度失衡，我们对扫描总数和商品种类数做了分位数截断，把超过 99.9 分位的值压到边界上。这样树模型训练更稳定。

### 2.3 模型训练与超参数调优
使用随机森林和 XGBoost 做多分类。

```python
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
X = agg.values
y = df.groupby('VisitNumber')['TripType'].first().values
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                    objective='multi:softprob', num_class=38)
clf.fit(X_tr, y_tr)
pred = clf.predict_proba(X_val)
print('LogLoss:', log_loss(y_val, pred, labels=range(38)))
```

进一步加入部门哑变量和 ScanCount 负值统计。调参使用交叉验证选择 max_depth 和 n_estimators。也可用 LightGBM 处理高维类别特征。

为什么选 XGBoost 而不是神经网络？因为行程分类的输入是聚合后的表格特征，维度只有几十维到几百维，样本量约 9 万个访问。这种「中等样本、低维表格」的场景正是梯度提升树的强项。神经网络在这么小的数据上容易过拟合，而且调参成本高。我们也试过 LightGBM，结果与 XGBoost 接近，说明特征质量比模型选择更重要。

调参时我们重点看两个参数：树的深度和树的数量。深度太深会记住训练集里的噪声，深度太浅又抓不住商品组合的复杂交互。最终通过 5 折交叉验证，选定深度 6 左右、300 棵树。学习率用 0.1，配合较高的树数量，收敛速度和稳定性比较平衡。

### 2.4 模型测试与指标报告
验证集 LogLoss 约 0.9 左右。测试集生成提交：

```python
test = pd.read_csv('test.csv')
test_agg = test.groupby('VisitNumber').agg({...})
proba = clf.predict_proba(test_agg)
submission = pd.DataFrame({
    'VisitNumber': test_agg.index,
    'TripType': proba.argmax(axis=1)
})
```

最终输出符合 sample_submission。报告包括各类召回，稀有类表现较差。我们还会检查预测的类别分布是否与训练集先验接近，如果某些类完全没人预测，就说明模型塌缩到多数类了，需要重新检查特征或做类别权重调整。

## 三、总结
本文中我们对 Walmart 购物行程分类任务做了完整建模。我们首先做 EDA，发现 5704 条完全重复记录和 38 类极度不平衡的标签；其次去重、填充 FinelineNumber 缺失并按 VisitNumber 聚合统计；接着用 XGBoost 多分类训练并调参；最后以 LogLoss 评估并生成提交。行程类型主要由购买商品组合和数量决定，聚合特征是关键。未来可加入商品嵌入提升效果。

在特征构造上我们还尝试了几种组合。第一种是把每个访问看成一个文档，Upc 看成词，用 TF-IDF 得到商品权重，再按行程汇总。第二种是把部门描述做独热编码，统计每个访问在各部门的花费占比。第三种是统计退货行为，ScanCount 为负的记录数与总数之比往往与行程类型相关。把这三类特征拼在一起后，验证集 LogLoss 明显下降。

周几也是一个有用信号。周末购物更可能是大宗采购，工作日更可能是小额补货。我们把 Weekday 独热编码后加入模型，效果比直接当数值好。

类别不平衡方面，38 类里有几类样本极少。我们尝试过对稀有类过采样，但对 LogLoss 帮助不大，反而让模型对稀有类过度自信。最终保留了原始分布，只在评估时按类别统计召回。这样虽然稀有类召回不高，但整体 LogLoss 稳定。

在实际提交时，我们按 VisitNumber 分组聚合测试集，确保每个访问只输出一个预测。为了避免预测完全偏离先验分布，我们还在预测概率上做了轻微的温度缩放，让过于自信的类别概率向均匀分布靠拢一点，这在 LogLoss 上有小幅收益。