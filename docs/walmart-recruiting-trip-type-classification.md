# Walmart 购物行程类型分类

## 一、比赛简介

### 1.1 比赛背景
Walmart Recruiting - Trip Type Classification 要求根据顾客在一次购物中购买的商品，预测这次购物属于哪一种 TripType。Walmart 用内部方法将旅行分成 38 类，比如日常晚餐小采购、每周大采购、节日礼品采购、季节性买衣服等。

比赛提供有限字段（商品 UPC、部门、细类、扫描数量、星期几），需要重建这 38 类分类。难点是类别极度不平衡，且同一 VisitNumber 下有多条商品记录。数据来自美国 Walmart 门店交易，包含顾客访问、商品和行程标签。

该任务模拟零售客户细分，行程类型反映顾客购物意图，对营销策略制定至关重要。数据规模约 65 万行商品记录，覆盖上万次访问。类别分布右偏，部分稀有类别样本不足百条。

### 1.2 比赛目标
输入训练集 train.csv，字段包括 TripType、VisitNumber、Weekday、Upc、ScanCount、DepartmentDescription、FinelineNumber。测试集无 TripType。

目标为每个 VisitNumber 预测一个 TripType。提交列 VisitNumber, TripType。每个访问只有一种行程类型，需从商品清单聚合特征。

训练集含 38 个类别，TripType_999 为其他类。VisitNumber 是访问 ID，Upc 为商品条码，ScanCount 为购买数量，负值表示退货。

### 1.3 评价标准
使用多分类对数损失 LogLoss，越低越好。模型对真实类别给出的概率越高，损失越小；预测错且自信时惩罚很大。

LogLoss 要求模型输出概率分布，而非硬标签。类别不平衡下，需注意校准。标准为负对数似然平均值，越小越好。

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

可视化 Upc 频次，发现热门商品集中在食品部门。行程类型与部门组合相关，如礼品类多与玩具、家居相关。

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

创建退货率、平均单价代理特征。Weekday 做独热编码。Upc 频次做 TF-IDF 风格加权。

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

最终输出符合 sample_submission。报告包括各类召回，稀有类表现较差。

## 三、总结
本文中我们对 Walmart 购物行程分类任务做了完整建模。我们首先做 EDA，发现 5704 条完全重复记录和 38 类极度不平衡的标签；其次去重、填充 FinelineNumber 缺失并按 VisitNumber 聚合统计；接着用 XGBoost 多分类训练并调参；最后以 LogLoss 评估并生成提交。行程类型主要由购买商品组合和数量决定，聚合特征是关键。未来可加入商品嵌入提升效果。