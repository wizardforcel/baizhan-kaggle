# Kobe Bryant 射门选择

## 一、比赛简介

### 1.1 比赛背景

科比·布莱恩特是 NBA 历史上最伟大的球员之一，他整个 20 年的职业生涯都在洛杉矶湖人队度过。这个比赛提供科比在职业生涯中每一记投篮的数据，包括投篮位置、比赛时间、对手、比分、赛季等。题目要求在没有任何结果信息的情况下，预测科比是否会投中。

数据包含 3 万多条投篮记录，覆盖 1996-2016 年的 20 个赛季。每个样本都是一次投篮尝试，有很多空间和时间特征可以用来判断投篮难度。

### 1.2 比赛目标

给定一次投篮的上下文特征，预测该投篮是否命中。输入特征包括：

- action_type：投篮动作类型，比如 Jump Shot、Dunk
- combined_shot_type：投篮类型，Jump Shot、Dunk、Layup 等
- shot_distance：距离篮筐的距离，单位英尺
- shot_zone_basic / shot_zone_area / shot_zone_range：投篮区域划分，比如 Above the Break 3、Back Court
- loc_x / loc_y：投篮点在球场坐标系中的位置
- period、minutes_remaining、seconds_remaining：比赛时间和节数
- season、playoffs、opponent、matchup：赛季、是否季后赛、对手、主客场
- shot_type：2 分或 3 分

输出是二分类标签 shot_made_flag，1 表示命中，0 表示未命中。测试集没有标签，需要我们预测。

### 1.3 评价标准

比赛使用对数损失 Log Loss 作为评价指标。对数损失衡量模型预测概率的准确性，越小越好。模型不仅要猜对，更要给出合理的置信度。

## 二、解题过程

我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

先读取数据，看看规模和列名：

```python
# 读取投篮数据，设 shot_id 为索引
data = pd.read_csv('../input/data.csv')
data.set_index('shot_id', inplace=True)
print(data.shape)
```

数据有 30697 条记录，25 个特征。我们检查标签分布：

```python
# 看命中率大概多少
data['shot_made_flag'].value_counts(normalize=True)
```

命中率约 45% 左右，小于一半，属于二分类不平衡问题。

再看投篮距离分布：

```python
# 投篮距离的描述统计
print(data['shot_distance'].describe())
```

平均距离约 15 英尺，标准差约 7 英尺。我们画位置散点图，发现大部分投篮集中在弧线内，只有少数是远距离 3 分。

位置特征有两个表示：地图坐标 lon/lat 和球场坐标 loc_x/loc_y。我们发现 lon/lat 基本重复，可以只保留 loc_x/loc_y：

```python
# 用极坐标转换，更符合半圆形球场
shots$loc_r <- sqrt((shots$loc_x)^2 + (shots$loc_y)^2)
shots$loc_theta <- atan(shots$loc_y/shots$loc_x)
```

把笛卡尔坐标转成极坐标后，r 表示距离，theta 表示角度，更符合篮球场地形状。

时间特征方面，我们把剩余额外转成总秒数：

```python
# 合并分钟和秒，得到每节剩余总秒数
shots$time_remaining <- (shots$minutes_remaining * 60) + shots$seconds_remaining
shots$lastminutes <- ifelse(shots$time_remaining <= 180, 1, 0)
```

lastminutes 表示是否在节末最后 3 分钟，这段时间压力大，投篮难度可能不同。

### 2.2 数据清洗

训练集和测试集的区分方式是 shot_made_flag 是否为 NA：

```python
# NA 的就是测试集
test.id <- shots %>% filter(is.na(shot_made_flag)) %>% select(shot_id)
shots_filtered_train <- shots_filtered %>% filter(!is.na(shot_made_flag))
shots_filtered_test <- shots_filtered %>% filter(is.na(shot_made_flag))
```

我们删除了永远不变的列，比如 team_id、team_name，因为科比一直效力湖人。同时删除原始坐标 lon/lat，只保留转换后的 loc_r 和 loc_theta。

对于赛季列，原始格式是 "1996-97"，我们只取后两位数字，简化为分类变量：

```python
shots$season <- sapply(shots$season, function(x){str_extract(x, "(?<=[:punct:])[:digit:]{2}")})
```

对于 matchup 变量，我们提取主客场信息：

```python
shots$home <- as.numeric(grepl("vs.", shots$matchup, fixed = TRUE))
shots$away <- as.numeric(grepl("@", shots$matchup, fixed = TRUE))
```

这些衍生特征能让模型更好地捕捉比赛情境。

### 2.3 模型训练与超参数调优

我们选用 XGBoost 做基线。先把数据转成矩阵：

```python
train.y <- shots_filtered_train$shot_made_flag
trainM <- data.matrix(shots_filtered_train, rownames.force = NA)
dtrain <- xgb.DMatrix(data = trainM, label = train.y, missing = NaN)
```

参数设置上，objective 选 binary:logistic，做二分类。eta 设为 0.0375，学习率较小，防止过拟合。max_depth 设为 5，限制树深度。subsample 和 colsample_bytree 分别设为 0.8 和 0.6，做随机采样增强泛化。

我们用 10 折交叉验证并早停：

```python
param <- list(objective = "binary:logistic",
    booster = "gbtree",
    eval_metric = "logloss",
    eta = .0375,
    max_depth = 5,
    subsample = .8,
    colsample_bytree = .6
)
clf <- xgb.cv(params = param,
    data = dtrain,
    nrounds = 500,
    nfold = 10,
    early_stopping_rounds = 20,
    maximize = FALSE)
```

交叉验证结果显示训练 log loss 从 0.69 降到约 0.56，测试 log loss 最好在 0.601 左右，第 268 轮停止。说明模型在 268 轮后开始过拟合。

我们还加入了职业生涯阶段特征，比如是否在全明星赛季、是否有戒指等：

```python
shots$first_team <- ifelse((shots$game_num >= 395 & shots$game_num <= 474) | ..., 1, 0)
shots$postachilles <- ifelse(shots$game_num > 1452, 1, 0)
```

这些特征让模型能区分科比不同阶段的投篮手感。

### 2.4 模型测试与指标报告

最佳迭代的 log loss 指标如下：

```
iter train_logloss_mean train_logloss_std test_logloss_mean test_logloss_std
1:  268           0.566653       0.001488       0.6010992       0.01022511
```

测试集 log loss 约 0.601，训练集 0.567，差距不大，说明泛化良好。

我们用最佳轮数重新训练模型：

```python
xgb.model <- xgb.train(params = param,
    data = dtrain,
    nrounds = bestRound,
    watchlist = watchlist)
```

预测后生成提交文件：

```python
pred <- predict(xgb.model, dtest)
submission <- data.frame(shot_id = test.id, shot_made_flag = pred)
write.csv(submission, 'submission.csv', row.names = FALSE)
```

最终我们提交了基于 XGBoost 的预测结果，并在本地验证中获得稳定的 log loss。

## 三、总结

本文中我们完成了 Kobe Bryant 射门选择的完整流程。我们首先通过 EDA 发现投篮位置和时间是关键特征，并把笛卡尔坐标转成极坐标。接着做了主客场、节末压力等特征工程。随后用 XGBoost 建立基线，通过 10 折交叉验证和早停得到最佳 268 轮，测试 log loss 约 0.601。最后生成预测文件提交。

这个比赛告诉我们，空间特征的合理转换比原始坐标更有价值。同时加入球员生涯阶段的上下文特征，能显著提升预测效果。
