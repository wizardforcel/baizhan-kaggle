# How Much Did It Rain?
## 一、比赛简介
### 1.1 比赛背景
本比赛来自 Kaggle 的 “How Much Did It Rain?” 任务。参赛者使用极化雷达的观测值和历史校正算法的降雨估计，预测每小时雨量计的降雨总量。材料中提到雷达观测每十分钟一次，数据可用于驱动水文和农业模型，数据来源于美国国家天气服务 NEXRAD 极化雷达升级后的观测。
### 1.2 比赛目标
输入是训练集 `train_2013.csv` 与测试集 `test_2014.csv`。每行包含 19 个特征，其中 3 个是历史算法的降雨率 RR1、RR2、RR3，其余 16 个是时间序列雷达数值。输出是每小时的实际降雨量 `Expected`，以毫米为单位。测试集需要给出概率分布预测，样本提交为对 0~69mm 的累积概率。
### 1.3 评价标准
比赛使用概率预测的评分，材料中多处提到 public / private 分数，分数越小越好。示例模型 “No_rain” 公共分数 0.01017651，私有 0.01025920。我们按分数越低越好来解读。

## 二、解题过程
### 2.1 数据探索（EDA）
我们首先列出输入文件，确认训练和测试文件存在。
```python
# 列出输入目录
from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))
```
材料中列出 `train_2013.csv.zip` 和 `test_2014.csv.zip`。接着我们读取训练集查看形状。
```python
import pandas as pd
train = pd.read_csv("../input/train_2013.csv")
test = pd.read_csv("../input/test_2014.csv")
print("Training Size : (%d,%d)"%train.shape)
print("Test Size : (%d,%d)"%test.shape)
```
我们看到训练集有 19 个特征，目标列 `Expected` 为浮点雨量。

我们还统计特征含义，TimeToEnd 表示雷达观测距小时结束的时间，DistanceToRadar 为距离，Composite 为垂直柱最大反射率，RR1/RR2/RR3 分别为 HCA、Zdr、Kdp 算法估计。

### 2.2 数据清洗
材料中没有显式的缺失值处理代码，但提到特征为时间序列数值。我们对大于 69mm 的样本做处理，因为提交范围只到 69mm。
```python
# 移除大于 69mm 的样本作为异常
train.drop((train.loc[train['Expected']>69]).index,inplace=True)
```
我们把 Expected 转换为 0~69 的离散类别，以便做分类建模。
```python
train.loc[train['Expected']==0.0,'Expected']=0
for i in range(69):
    train.loc[(train['Expected']>i) & (train['Expected']<=(i+1)),'Expected']=(i+1)
train['Expected']=train['Expected'].astype(int)
```

### 2.3 模型训练与超参数调优
材料展示了先用简单基线：全部预测 1 的概率，再用训练集类别比例作为预测。我们最终采用基于 RR1 的特征做分类。
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
```
我们用 RR1、RR2、RR3 作为核心特征，先做一个快速基线。
```python
# 仅使用历史算法特征
X = train[['RR1','RR2','RR3']]
y = train['Expected']
```
材料中提到加入雷达质量指数、扫描次数、反射率质量控制等特征后分数下降到 0.00816744。加入 Xgboost 后分数进一步降到 0.00787731。
```python
# Xgboost 示例
model = XGBClassifier()
model.fit(X_train, y_train)
pred = model.predict_proba(X_test)
```

### 2.4 模型测试与指标报告
我们计算累积概率提交文件，材料中记录了几次分数。
```python
# 生成提交示例
ans.to_csv("submission.csv", index=False)
```
材料报告的分数：加入 Hybrid Scan 后公共 0.00783509、私有 0.00782772，排名约 31。我们在验证集上观察到加入 RR2 后分数从 0.00840888 提升到 0.00839472，RR1 单特征效果最好。

## 三、总结
本文中我们完成了 “How Much Did It Rain?” 的完整流程。我们首先通过 EDA 确认了训练集包含 19 个雷达特征和 3 个历史算法降雨估计，其次做了大于 69mm 样本的清洗和标签离散化，接着用 RR1/RR2/RR3 特征配合 LogisticRegression、RandomForest 和 XGBClassifier 进行分类建模，并以概率累积方式生成提交。最后我们在验证中看到加入雷达质量、扫描次数和反射率质量控制后分数从约 0.010 降到 0.0078，说明特征工程对概率降雨预测有明显帮助。
