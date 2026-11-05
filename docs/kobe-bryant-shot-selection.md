# Kobe Bryant 射门选择（Kobe Bryant Shot Selection）

## 一、比赛简介

### 1.1 比赛背景

科比·布莱恩特是 NBA 历史上最伟大的球员之一，他 20 年的职业生涯都在洛杉矶湖人队度过，留下了无数经典时刻。这个比赛来自他真实的比赛数据：每一记投篮的时间、位置、当时的比分、对手、赛季等都被记录了下来。

题目要求我们在**不知道投篮结果的情况下**，预测科比这球投不投得进。也就是说，只看「这记投篮是在什么情况下出手的」，判断它有多大的概率命中。这不仅是一个有趣的数据题，也暗合真实篮球导播和教练组在场上做的判断——同样的位置，不同的防守强度和比赛阶段，投篮难度完全不同。

数据包含 3 万多条投篮记录，覆盖 1996-2016 年的 20 个赛季。每个样本就是一次投篮尝试，带有很多空间特征和时间特征。难点在于这些特征之间关系复杂：同样是「离篮筐 25 英尺」，空位三分和压哨三分完全不同。

### 1.2 比赛目标

给定一次投篮的上下文特征，预测该投篮是否命中。这是一个二分类问题，主要输入特征包括：

- `action_type`：投篮动作类型，比如 Jump Shot、Dunk、Layup；
- `combined_shot_type`：更粗的投篮类型分类；
- `shot_distance`：距离篮筐的距离，单位是英尺；
- `shot_zone_basic` / `shot_zone_area` / `shot_zone_range`：投篮区域的划分，比如禁区、三分线外、后场；
- `loc_x` / `loc_y`：投篮点在标准球场坐标系中的位置；
- `period`、`minutes_remaining`、`seconds_remaining`：第几节、每节还剩多少时间；
- `season`、`playoffs`、`opponent`、`matchup`：赛季、是否季后赛、对手和主客场；
- `shot_type`：2 分球还是 3 分球。

输出是一个二值标签 `shot_made_flag`：1 表示命中，0 表示未命中。训练集有标签，测试集没有，需要我们来预测。

### 1.3 评价标准

比赛使用**对数损失（Log Loss）**作为评价指标。它衡量的是模型给出的预测概率和真实标签的吻合程度：如果你对命中球给出 0.95 的高概率，扣分就少；如果给 0.2 的低概率，扣分就多。它比普通准确率更严格，因为**模型不仅要猜对，还要给出靠谱的置信度**。分数越低越好，纯猜 50% 的 log loss 是 0.693。

## 二、解题过程

我们采用了一个经典的思路：先做充分的 EDA 理解投篮的空间分布，再对坐标和时间做特征工程，最后用 XGBoost 加 10 折交叉验证得到稳定的预测概率。下面按四步展开。

### 2.1 数据探索（EDA）

第一步是读取数据，看整体规模：

```python
import pandas as pd

# 读取投篮数据，把 shot_id 设为行索引
data = pd.read_csv('../input/data.csv')
data.set_index('shot_id', inplace=True)
print(data.shape)
```

数据有 30697 条记录。我们关心每列的情况，特别是标签的分布：

```python
# 看命中与未命中的比例
data['shot_made_flag'].value_counts(normalize=True)
```

命中率大约在 45% 左右，略低于一半。这意味着问题略有类别不平衡，但不算严重——我们不需要特别复杂的不平衡处理，用模型预测概率本身就能解决。

再看投篮距离：

```python
# 投篮距离的描述统计
print(data['shot_distance'].describe())
```

平均出手距离约 15 英尺，标准差约 7 英尺，说明大部分投篮是距离适中的跳投，远距离的零星有一些。距离显然是判断命中率最直观的特征。

接着我们画投篮位置散点图。发现大部分点集中在球场的弧线以内，只有少数超远距离出手。我们看到原始数据有两套位置坐标：地图坐标 `lon/lat` 和球场坐标 `loc_x/loc_y`。前者基本是后者的冗余，因此只保留 `loc_x/loc_y` 就够了。

为了让位置特征更贴合篮球场这个半圆形的空间，我们把直角坐标转成极坐标：

```python
# 把 (x, y) 转成 (距离 r, 角度 theta)
shots$loc_r <- sqrt((shots$loc_x)^2 + (shots$loc_y)^2)
shots$loc_theta <- atan(shots$loc_y / shots$loc_x)
```

`loc_r` 表示离篮筐多远，`loc_theta` 表示从哪个角度出手。对投篮难度来说，「距离 + 角度」比「x、y」更直观，也更容易被树模型利用。

时间特征方面，我们把「还剩几分钟几秒」合并成总的剩余秒数，并造一个「是否临近节末」的开关：

```python
# 合并分钟和秒，得到每节剩余的总秒数
shots$time_remaining <- (shots$minutes_remaining * 60) + shots$seconds_remaining

# 最后 3 分钟（180 秒）压力更大，单独标记出来
shots$lastminutes <- ifelse(shots$time_remaining <= 180, 1, 0)
```

### 2.2 数据清洗

这份数据的「清洗」主要是两类：处理测试集划分、删除冗余列。

训练集和测试集的区分方式是很有技巧的——**`shot_made_flag` 为空的记录就是测试集**。所以我们按是否缺失标签来切分：

```python
# 标签缺失的就是测试集
test.id <- shots %>% filter(is.na(shot_made_flag)) %>% select(shot_id)

# 标签存在的才是训练集
shots_filtered_train <- shots_filtered %>% filter(!is.na(shot_made_flag))
shots_filtered_test <- shots_filtered %>% filter(is.na(shot_made_flag))
```

随后我们删除了几乎不变化的列，比如 `team_id`、`team_name`——科比二十年都在湖人，这两列对区分投篮毫无帮助。同时删掉冗余的 `lon/lat` 原始坐标，只保留转换后的 `loc_r` 和 `loc_theta`。

赛季列原始格式是 `"1996-97"`，我们把它简化成两位数年份，当作分类特征：

```python
# 从 "1996-97" 里提取出后两位 "97"
shots$season <- sapply(shots$season, function(x){str_extract(x, "(?<=[:punct:])[:digit:]{2}")})
```

`matchup` 列里藏着主客场信息：`vs.` 表示主场，`@` 表示客场。我们把它拆成两个独立的指示变量：

```python
# 包含 "vs." 的是主场
shots$home <- as.numeric(grepl("vs.", shots$matchup, fixed = TRUE))
# 包含 "@" 的是客场
shots$away <- as.numeric(grepl("@", shots$matchup, fixed = TRUE))
```

这些特征能让模型区分「主场还是客场」这种情境差别——球员在主客场的状态确实有差异。

另外，根据比赛进程，我们还加入了几个与职业生涯阶段有关的手工特征，比如是否在全明星赛季、是否在跟腱重伤之后复出等。这些特征试图粗略刻画「科比职业生涯不同阶段的手感」：巅峰期和生涯末期，同样的出手位置命中率完全不同。

### 2.3 模型训练与超参数调优

我们选用 XGBoost 作为主要模型，因为它对表格特征、混合类型（数值 + 分类）支持好，训练快，泛化也不错。先把数据转成 XGBoost 需要的矩阵格式：

```python
train.y <- shots_filtered_train$shot_made_flag
trainM <- data.matrix(shots_filtered_train, rownames.force = NA)
dtrain <- xgb.DMatrix(data = trainM, label = train.y, missing = NaN)
```

参数设置上我们做了几处精心选择：

- `objective = "binary:logistic"`：二分类，输出 0~1 的概率；
- `eta = 0.0375`：学习率取得比较小，让每棵树的贡献都小一点，减少过拟合；
- `max_depth = 5`：限制树的深度，避免学到太细的噪声；
- `subsample = 0.8` 和 `colsample_bytree = 0.6`：每棵树只随机用 80% 的样本、60% 的特征，这相当于给模型加噪声，增强泛化。

我们用 10 折交叉验证配合早停来决定训练多少轮：

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
    nrounds = 500,             # 最多训练 500 轮
    nfold = 10,                # 10 折交叉验证
    early_stopping_rounds = 20, # 连续 20 轮没变好就停
    maximize = FALSE)
```

交叉验证告诉我们：训练 log loss 从 0.69 一路下降到约 0.56，验证集 log loss 最好到了大约 0.601，且第 268 轮之后验证分数不再下降。这个信息很关键——它告诉我们**268 轮是「欠拟合到过拟合」的转折点**，再往后训练只是在背数据。

### 2.4 模型测试与指标报告

最佳状态下，交叉验证各折的平均指标如下（取自材料里的日志）：

```
iter train_logloss_mean train_logloss_std test_logloss_mean test_logloss_std
 268           0.566653       0.001488        0.6010992       0.01022511
```

注意训练 log loss（0.567）和验证 log loss（0.601）只相差约 0.03，标准差也很小，说明模型没有严重过拟合，10 折之间的表现也比较稳定。这正好印证了早停起了作用。

我们用这一最佳轮数重新在完整训练集上训练模型：

```python
xgb.model <- xgb.train(params = param,
    data = dtrain,
    nrounds = bestRound,     # 268 轮
    watchlist = watchlist)
```

最后对测试集预测并生成提交文件：

```python
pred <- predict(xgb.model, dtest)   # 得到每个样本的命中概率
submission <- data.frame(shot_id = test.id, shot_made_flag = pred)
write.csv(submission, 'submission.csv', row.names = FALSE)
```

## 三、总结

本文中我们完成了 Kobe Bryant 射门选择挑战的完整流程。我们首先通过 EDA 摸清了数据：3 万多条投篮记录、命中率约 45%、空间特征和时间特征并存。其次我们把位置坐标转成极坐标、合并出剩余时间和节末标记、提取主客场信息，并加入职业生涯阶段特征。接着我们用 XGBoost 配合 10 折交叉验证和早停，在 268 轮处得到验证 log loss 约 0.601，然后用最佳轮数重新训练并生成提交文件。

这个比赛告诉我们：**空间特征的合理转换（极坐标）比原始坐标更有价值**，而交叉验证配合早停能帮我们精确地找到「不过拟合」的训练轮数，这两点在很多表格类比赛里都通用。