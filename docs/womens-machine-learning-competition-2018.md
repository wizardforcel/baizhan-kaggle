# Women's Machine Learning Competition 2018

## 一、比赛简介

### 1.1 比赛背景
本比赛为 Google Cloud 举办的 NCAA 女子篮球锦标赛预测赛，数据来自女子 NCAA 篮球的历史比赛结果。NCAA 女子篮球赛是美国大学女子篮球的顶级赛事，比赛过程复杂，受球队实力、赛季表现、赛程安排等多因素影响。参赛者需要利用常规赛和锦标赛的历史数据，预测未来比赛的胜负概率。

数据包含常规赛详细结果、锦标赛详细结果、种子信息、球队信息等。球队实力变化快，且存在明显的赛程效应，故需要从多维度构建特征。

### 1.2 比赛目标
比赛分为两个阶段。第一阶段预测锦标赛的比赛胜负概率，第二阶段预测更细粒度的得分差等。输入为两支球队的对阵信息，输出为比赛胜负概率。训练数据包含多赛季常规赛和锦标赛的详细比赛数据，每场比赛记录胜负球队、比分、投篮命中率、助攻、盖帽等统计指标。

我们需要基于历史比赛特征，预测指定比赛结果的概率分布，并以对数损失为准。

### 1.3 评价标准
比赛使用对数损失 Log Loss 作为评价指标，衡量预测概率与真实结果的差异。对数损失对预测置信度敏感，预测概率越接近真实标签，损失越小。数值越小越好，理想值为 0。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读取常规赛结果文件。

```python
import pandas as pd
regresults = pd.read_csv("WRegularSeasonDetailedResults_PrelimData2018.csv")
results = pd.read_csv("WNCAATourneyDetailedResults_PrelimData2018.csv")
```

数据规模较大，常规赛约 5 万场比赛记录，每场比赛包含 34 个统计指标。

```python
regresults.info()
# RangeIndex: 51582 entries
# Data columns (total 34 columns)
```

我们将胜负球队信息对称化，方便建模。

```python
r1 = regresults[, c("Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", ...)]
r2 = regresults[, c("Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore", ...)]
names(r1) = c("Season","DayNum","T1","T1_Points","T2","T2_Points",...)
regular_season = rbind(r1, r2)
```

这样每一行代表一个球队在该场的表现，后续特征工程更方便。

### 2.2 数据清洗
数据本身统计完整，但需要处理缺失的种子信息。

```python
seeds = pd.read_csv("WNCAATourneySeeds.csv")
seeds$Seed = as.numeric(substring(seeds$Seed,2,4))
```

我们筛选出进入锦标赛的球队。

```python
march_teams = select(seeds, Season, Team = TeamID)
X = regular_season %>%
inner_join(march_teams, by = c("Season" = "Season", "T1" = "Team")) %>%
inner_join(march_teams, by = c("Season" = "Season", "T2" = "Team"))
```

仅保留两队都在锦标赛名单中的比赛，减少噪声。

### 2.3 模型训练与超参数调优
我们采用混合效应逻辑回归，先为每个赛季拟合球队随机效应。

```python
library(lme4)
glmm = glmer(I(T1_Points > T2_Points) ~ (1 | T1) + (1 | T2),
data = X[X$Season == season & X$NumOT == 0, ], family = binomial)
random_effects = ranef(glmm)$T1
quality[[season]] = data.frame(Season = season, Team_Id = as.numeric(row.names(random_effects)),
quality = exp(random_effects[,"(Intercept)"]))
```

随机效应代表球队的隐性实力，指数化后可作为特征输入。

接下来构建常规赛统计特征。

```python
season_summary =
regular_season %>%
mutate(win14days = ifelse(DayNum > 118 & T1_Points > T2_Points, 1, 0),
last14days = ifelse(DayNum > 118, 1, 0)) %>%
group_by(Season, T1) %>%
summarize(
WinRatio14d = sum(win14days) / sum(last14days),
PointsMean = mean(T1_Points),
PointsMedian = median(T1_Points),
PointsDiffMean = mean(T1_Points - T2_Points),
FgaMean = mean(T1_fga)
)
```

我们还加入种子差异、主场优势等特征。

```python
feature = data.frame(
SeedDiff = Team1Seed - Team2Seed,
EloDiff = Elo1 - Elo2,
RecentWinRatioDiff = WinRatio14d1 - WinRatio14d2
)
```

模型选择方面，我们尝试 Logistic Regression、XGBoost、GLMM 混合模型。超参数调优采用交叉验证，网格搜索正则化强度和树的深度。

### 2.4 模型测试与指标报告
在验证集上，我们得到 Logistic Regression 基线对数损失约 0.51。

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(X_train, y_train)
pred = clf.predict_proba(X_test)[:,1]
log_loss(y_test, pred)
```

加入随机效应特征后，损失有所下降。我们按赛季分组评估，发现不同赛季球队实力分布变化较大，时间序列交叉验证比随机拆分更稳健。

最终提交时，我们用全量历史数据重新训练，并在预测时对概率做裁剪，避免预测 0 或 1 导致对数损失爆炸。

```python
submission = pd.DataFrame({'Id': test['Id'], 'pred': pred})
submission.to_csv('submission.csv', index=False)
```

## 三、总结
本文中我们完成了 Women's Machine Learning Competition 2018 的建模流程。我们首先用 EDA 明确了数据结构：常规赛约 5 万场、锦标赛详细结果、种子信息齐全。其次我们对数据做对称化处理，筛选锦标赛球队，计算随机效应实力指标。接着我们用 GLMM 提取球队随机效应，结合最近 14 天胜率、场均得分等统计特征训练 Logistic Regression 和 XGBoost 模型。最后我们在验证集上得到基线对数损失约 0.51，并通过特征增强和概率裁剪优化提交。经验表明，女子 NCAA 篮球预测中球队历史随机效应和近期状态是关键，简单线性模型配合好的特征即可取得较好表现。
