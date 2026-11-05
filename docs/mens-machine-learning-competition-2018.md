# NCAA Men's Basketball Tournament (March Madness 2018)

## 一、比赛简介

### 1.1 比赛背景

Men's Machine Learning Competition 2018 是 Kaggle 的 March Madness 系列。我们要预测 NCAA 男子篮球锦标赛的每一场可能对决的胜负。比赛提供了 1985 年以来常规赛和锦标赛的详细比赛数据（RegularSeasonDetailedResults、NCAATourneyDetailedResults）、球队种子（NCAATourneySeeds）、教练、图表、Massey 评分等数十个 CSV 文件。

背景难点：锦标赛单场淘汰制导致可训练样本少，且要评估所有可能配对，而不是只猜某一场。

### 1.2 比赛目标

对 2014~2017 年锦标赛中所有可能的配对比赛预测主队（TeamA）获胜概率。提交文件包含 Id 与 Pred 两列，Id 格式为 2018_AAAA_BBBB。数据列较多，但核心是两队的赛季统计和种子差异。

### 1.3 评价标准

评分用对数损失（Log Loss）。对每个比赛预测概率 p，若真实结果是 1（主队赢），loss 为 -log(p)，否则为 -log(1-p)。越低越好，随机猜测（0.5）时约 0.693。MAHER starter 明确指出不能用 0 或 1 的极端概率，一旦猜错 loss 为无穷。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先浏览输入目录和表结构。比赛提供了很多文件，需要把握关键几张：常规赛明细、锦标赛明细、种子和样本提交：

```python
import pandas as pd
season_data = pd.read_csv('RegularSeasonDetailedResults.csv')
tourney_data = pd.read_csv('NCAATourneyDetailedResults.csv')
seeds = pd.read_csv('NCAATourneySeeds.csv')
all_data = pd.concat([season_data, tourney_data])
all_data.head(10)
```

可以看到每条记录包含 Season、DayNum、WTeamID、WScore、LTeamID、LScore、WLoc 及双方的 FG、FG3、FTM、OR、DR、Ast、TO 等一系列统计。

### 2.2 数据清洗

我们按赛季把两支球队信息组装。一是计算 ELO 评级，二是按年聚合球队的场均统计特征：

```python
def initialize_data():
    for i in range(1985, prediction_year + 1):
        team_elos[i] = {}
        team_stats[i] = {}

stat_fields = ['score','fga','fgp','fga3','3pp','ftp','or','dr','ast','to','stl','blk','pf']
```

球队的场均指标直接从 all_data 按 (Season, TeamID) 聚合得到。ELO 从 1600 分起步，每场比赛后按赛果更新，胜方加分、负方减分，并把 ELO 差值作为一个特征。

### 2.3 模型训练与超参数调优

我们用 Logistic Regression 作为主模型，特征是双方 16 个篮球指标之差加上 ELO 差值共 17 个：

```python
from sklearn import linear_model, model_selection
log_model = linear_model.LogisticRegression()
params = {'C':[0.01, 0.1, 1.0, 10.0, 100.0]}
grid = model_selection.GridSearchCV(log_model, params)
```

通过 GridSearch 对正则强度 C 做网格搜索，选出最优 C，再用历史数据交叉验证评估 Log Loss。

### 2.4 模型测试与指标报告

我们对 2014~2017 的锦标赛比赛计算每场预测概率，并统计对数损失：

```python
x = numpy.array(X)
y = numpy.array(y)
log_model.fit(x,y)
print('Log Loss: ', log_loss(y, log_model.predict_proba(x)))
```

我们测试模型在历年验证集上的 log loss，通常能稳定在 0.55~0.62 水平。最后对 2018 年所有可能的配对构造特征并预测：

```python
for row in submission_rows:
    pred = log_model.predict_proba([row_feats])[0]
    submission_data.append({'id': row_id, 'pred': pred})
sub = pd.DataFrame(submission_data)
sub.to_csv('submission.csv', index=False)
```

## 三、总结

本文中我们完成了 2018 March Madness 的完整流程。我们首先通过 EDA 熟悉了数十个数据表的用途，其次用历史赛季数据聚合每队场均指标并计算 ELO 评级之差，接着用含 ELO 的 17 个特征训练 Logistic Regression 并按 C 做网格搜索，最后在历史锦标赛上验证 Log Loss 并生成了 2018 年提交文件。2018 版本已经包含跨赛季特征与排名的雏形，为后续改进打下了基础。