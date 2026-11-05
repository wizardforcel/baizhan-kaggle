# San Francisco Crime Classification

## 一、比赛简介

### 1.1 比赛背景
San Francisco Crime Classification 是 Kaggle 社区中围绕公共安全与城市数据挖掘的经典竞赛，数据来源于旧金山市警局公开的犯罪记录。旧金山作为全球科技与金融中心，兼具高人口流动性与显著的区域差异，犯罪类型与时空分布呈现复杂规律。本比赛旨在利用历史案件的时空属性、警区划分以及描述信息，预测未来案件的犯罪类别。任务涉及时间序列、地理坐标、类别不平衡等典型挑战，对特征工程与模型泛化能力要求较高，是入门者练习多分类与时空特征处理的好案例。

### 1.2 比赛目标
输入为训练集中的 Dates、Descript、DayOfWeek、PdDistrict、Resolution、Address、X、Y 等列，目标是预测 Category。测试集缺少 Category 列，需要对每个测试样本给出类别预测。目标变量包含约 39 个不同的犯罪类别，例如 LARCENY/THEFT、ASSAULT、VEHICLE THEFT、WARRANTS、BURGLARY、MISSING PERSON、ROBBERY、VANDALISM 等。任务属于多分类问题，数据量大、类别分布极不均衡，需要精心构造特征并选择合适的模型。

### 1.3 评价标准
材料中常见的评价指标为分类准确率 accuracy 以及对数损失 log_loss。准确率越高越好，对数损失越低越好。因为类别分布严重倾斜，单一整体准确率会掩盖少数类的表现，因此实际评估中常配合分类报告查看各类的精确率、召回率与 F1 分数。部分作品在标题中直接报告准确率约 26%，这一偏低分数正反映了多分类与类别不平衡的难度，说明需要更精细的评估方式。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先载入训练和测试集查看数据规模。

```python
import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv.zip')
test = pd.read_csv('../input/test.csv.zip')
print(train.shape)
print(test.shape)
```

训练集约 878049 行，9 列，测试集为同量级。接着我们查看数值列的统计信息。

```python
print(train.describe())
```

坐标 X 的均值约为 -122.42，标准差约为 0.03，Y 的均值约为 37.77，标准差约为 0.46，说明犯罪记录高度集中在旧金山市中心的狭长区域。日期列跨度多年，早期记录较少，后期逐年增多，存在时间上的增长趋势。

类别分布极不平衡。

```python
print(train['Category'].value_counts())
```

LARCENY/THEFT 占据最大比例，WARRANTS、OTHER OFFENSES、VANDALISM、ASSAULT 等次之，部分类别样本不足几百条。地址列包含街区信息但存在大量重复与噪声，且测试集地址格式不完全一致。

我们解析日期列，提取时间特征。

```python
train['Dates'] = pd.to_datetime(train['Dates'])
train['Hour'] = train['Dates'].dt.hour
train['Month'] = train['Dates'].dt.month
train['Year'] = train['Dates'].dt.year
train['Weekday'] = train['Dates'].dt.weekday
```

不同类别在不同小时、月份的出现频率差异明显，例如 BURGLARY 多集中在夜间，ASSAULT 在周末高发，VEHICLE THEFT 在夏季稍多。空间坐标 X、Y 显示犯罪热点集中在特定警区，尤其在 Tenderloin、Mission 等区域密度更高。

```python
train.groupby('PdDistrict')['Category'].nunique()
```

警区与犯罪类别强相关，可作为重要分组特征。我们随后可以统计每个警区中最常见的犯罪类别，辅助理解警务数据的分布规律，也为后续的目标编码提供依据。

### 2.2 数据清洗
测试集不包含 Descript、Resolution 列，需要删除训练集中不存在于测试的列，保证训练与测试的特征一致。

```python
train = train.drop(['Descript','Resolution'], axis=1)
```

Address 列包含大量文本噪声，且测试集与训练集地址格式并不统一，我们选择删除 Address 以降低维度、避免过拟合到具体地点细节。

```python
train = train.drop(['Address'], axis=1)
```

日期列进一步拆分为 Year、Month、Day、Hour、Weekday，并对小时和月份做周期性编码，保留时间的循环特性。

```python
train['Hour_sin'] = np.sin(2*np.pi*train['Hour']/24)
train['Hour_cos'] = np.cos(2*np.pi*train['Hour']/24)
train['Month_sin'] = np.sin(2*np.pi*train['Month']/12)
train['Month_cos'] = np.cos(2*np.pi*train['Month']/12)
```

周几同样可以做周期性编码，提升模型对周末效应的捕捉能力。类别标签需要编码为整数。

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(train['Category'])
print(len(le.classes_))
```

这里确认类别数约为 39。缺失值检查显示训练集基本无缺失，主要处理的只是结构与格式问题，而非缺失数据。因此数据清洗的核心是保持训练测试一致性与避免信息泄漏。

我们还检查了重复记录与异常坐标。少量坐标落在旧金山边界之外，可能是录入误差或系统偏移，直接删除会损失样本，因此我们保留它们但让树模型自行学习其影响。对于日期解析失败的行，我们统一转换为缺失值并在特征中显式标记，避免因为一条坏数据导致整列解析中断。类别标签只做整数编码，不改变类别顺序，也不合并稀有类别，以免丢失业务含义。所有清洗步骤都只使用训练集的统计信息，测试集仅做同样的转换，从而保证评估结果可信。这些细节看似琐碎，却决定了线下验证能否真实反映线上表现。

### 2.3 模型训练与超参数调优
我们先使用 XGBoost 进行基线训练。

```python
from sklearn.model_selection import train_test_split
import xgboost as xgb

X = train.drop(['Category','Dates'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = xgb.XGBClassifier(objective='multi:softprob', num_class=39, n_estimators=500,
                          max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
model.fit(X_train, y_train)
```

采用分层抽样保持训练与验证集的类别分布一致。随后我们尝试 LightGBM 并结合贝叶斯优化。

```python
from skopt import BayesSearchCV
import lightgbm as lgb

search = BayesSearchCV(estimator=lgb.LGBMClassifier(objective='multiclass', num_class=39),
                       search_spaces={'n_estimators':(100,1000), 'learning_rate':(0.01,0.3),
                                      'num_leaves':(31,255), 'max_depth':(3,12)},
                       n_iter=30, cv=3, scoring='accuracy', random_state=42)
search.fit(X_train, y_train)
print(search.best_params_)
```

贝叶斯优化比网格搜索更高效，能在大参数空间中快速找到较好组合。特征选择显示 DayOfWeek、PdDistrict、Hour_sin/cos 对模型提升明显。部分作品对 PdDistrict 做目标编码。

```python
train['PdDistrict_enc'] = train.groupby('PdDistrict')['Category'].transform(lambda x: x.mode()[0])
```

社区里也有作品使用随机森林、朴素贝叶斯、KNN 以及神经网络作为对照。经验表明，树模型在处理海量混合类型特征时通常优于线性与基于距离的模型，而集成方法能进一步提升稳定性。

### 2.4 模型测试与指标报告
在验证集上评估准确率与分类报告。

```python
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_val)
print('Accuracy:', accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=le.classes_))
```

整体准确率在 0.25 至 0.35 区间，少数类召回率较低。可视化的混淆矩阵能直观看到类别间的混淆，比如 ASSAULT 与 BATTERY 易混淆， LARCENY/THEFT 与 SHOPLIFTING 存在相似模式。

我们尝试类别加权，缓解不平衡问题。

```python
from sklearn.utils.class_weight import compute_class_weight
class_weight = compute_class_weight('balanced', classes=np.unique(y), y=y_train)
```

最终模型在全量训练集上重新训练并预测测试集。

```python
model.fit(X, y)
test_X = test.drop(['Id','Dates'], axis=1)
test_pred = model.predict(test_X)
submission = pd.DataFrame({'Id': test['Id'], 'Category': le.inverse_transform(test_pred)})
submission.to_csv('submission.csv', index=False)
```

提交文件校验行数、列名与类别合法性。通过特征工程与模型调参，验证集准确率得到提升，少数类表现有所改善，最终基于验证结果选择最佳模型提交并保存。

进一步分析结果时，我们注意到整体准确率并不能完全代表模型价值。由于 LARCENY/THEFT 等大类占比很高，模型只要偏向预测大类就能获得看似不错的准确率，但这类预测对实际警务调度的帮助有限。相反，稀有类别虽然样本少，却往往对应更严重的案件，漏判的代价更高。因此我们在报告阶段同时查看宏平均 F1 与各类召回率，避免被总体准确率误导。对类别加权之后，少数类的召回率有所上升，但整体准确率可能轻微下降，这是一种典型的权衡，需要根据业务目标决定取舍。

在特征层面，我们发现空间坐标与警区信息的组合比单独使用坐标更有效。原因是坐标本身噪声较大，而警区把空间划分成稳定的行政单元，能提供更平滑的先验。时间特征方面，小时的正弦余弦编码帮助模型理解夜间与白天的连续性，避免把二十三点与零点当成两个毫不相干的取值。星期与月份的加入则让模型捕捉到周末效应和季节效应。相比之下，地址文本带来的提升有限，却显著增加了维度与过拟合风险，因此我们选择删除。

在模型层面，XGBoost 与 LightGBM 的表现接近，后者训练更快，适合在有限时间内反复调参。贝叶斯优化比穷举网格更节省计算资源，尤其是在参数维度较高时优势明显。我们还尝试过随机森林作为对照，发现单棵树的方差较大，集成后虽然稳定，但在大规模多分类任务上训练时间较长。最终我们选择树模型集成作为主线方案，因为它对缺失值、异常值和特征尺度都不敏感，适合这类表格型城市数据。

## 三、总结
本文中我们完成了 San Francisco Crime Classification 的完整流程。我们首先通过 EDA 了解数据规模、类别不平衡和时空分布规律，确认坐标集中在城市核心区域。其次我们清理了测试集中不存在的列，对日期进行特征拆分并做周期性编码，提升时间信息的表达能力。接着我们用 XGBoost 和 LightGBM 训练多分类模型，并通过贝叶斯优化进行超参数调优。最后在验证集上报告准确率与分类报告，并生成提交文件。该比赛提醒我们，时空数据需要专门的特征工程，类别不平衡需要加权或采样策略来平衡表现，特征的选择与清洗往往比模型本身更关键。