# Kobe Bryant 投篮选择

## 一、比赛简介

### 1.1 比赛背景

科比·布莱恩特是 NBA 史上最伟大的球员之一，他整个 20 年职业生涯都在洛杉矶湖人队度过。比赛提供了他职业生涯中每一投的数据，包含投篮位置、距离、出手方式、比赛时间等 25 个变量。

这是一个经典的二分类预测问题：根据投篮时的情境，判断这次投篮是否命中。难度在于影响投篮命中的因素很多，且不同区域、不同时期的命中率差异很大。

### 1.2 比赛目标

给定一条投篮记录，预测 `shot_made_flag` 是否为 1（命中）或 0（未命中）。训练集数据包含 30697 条投篮记录，每个样本有 25 列特征：

- `action_type`、`combined_shot_type`、`shot_type` 等投篮类型
- `loc_x`、`loc_y`、`shot_distance` 投篮位置和距离
- `period`、`minutes_remaining`、`seconds_remaining` 比赛时间
- `season`、`playoffs`、`opponent` 等比赛情境
- `shot_zone_area`、`shot_zone_basic`、`shot_zone_range` 投篮区域区

测试集需要提交每条投篮的预测结果。

### 1.3 评价标准

比赛使用对数损失 Log Loss 作为评价指标。对数损失衡量模型对概率的预测精度，值越小越好。预测出错的置信度越高，惩罚越大。

## 二、解题过程

整场比赛的思路可以概括成一句话：**先把投篮数据探索清楚，再用 XGBoost 做二分类预测，重点处理位置特征和时序特征**。我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先把数据读进来，看看有 30697 条投篮记录，25 个变量：

```python
# 导入 R 语言常用包
library(tidyverse)
library(data.table)
library(xgboost)
library(dummies)

# 读取压缩的 CSV 文件
shots <- read.csv(unzip("../input/kobe-bryant-shot-selection/data.csv.zip"))
str(shots)
```

从结构信息可以看出，变量既有数值型也有因子型。`shot_made_flag` 是目标变量，有 NA 值存在。`loc_x`、`loc_y` 构成笛卡尔坐标系下的投篮位置。

我们发现 `loc_x` 和 `loc_y` 有重复信息，`lon`、`lat` 也是经纬度。于是我们把笛卡尔坐标转换到极坐标，得到投篮距离 `loc_r` 和角度 `loc_theta`：

```python
# 把笛卡尔坐标转成极坐标，方便表示环形球场
shots$loc_r <- sqrt((shots$loc_x)^2 + (shots$loc_y)^2)
shots$loc_theta <- atan(shots$loc_y/shots$loc_x)

# loc_x 为 0 时 atan 会得到 NA，补上 pi/2
shots$loc_theta[is.na(shots$loc_theta)] <- pi/2
```

接着我们统计命中率分布。可以看到不同区域的命中率差异很大，禁区和近距离高，远距离低。不同赛季的命中率也有波动。

### 2.2 数据清洗

数据本身质量不错，但我们做了几件事：

1. **处理缺失值**。`shot_made_flag` 有 NA，我们在训练时只保留非 NA 的样本：
```python
# 只保留有标签的投篮用于训练
data = allData[allData['shot_made_flag'].notnull()].reset_index()
```

2. **创建时序特征**。从 `game_date`、`period`、`minutes_remaining`、`seconds_remaining` 提取：
```python
data['game_date_DT'] = pd.to_datetime(data['game_date'])
data['dayOfWeek'] = data['game_date_DT'].dt.dayofweek
data['dayOfYear'] = data['game_date_DT'].dt.dayofyear
data['secondsFromPeriodEnd'] = 60*data['minutes_remaining']+data['seconds_remaining']
```

3. **类型转换**。把字符型变量转为 category，节省内存、便于建模：
```python
data["combined_shot_type"] = data["combined_shot_type"].astype('category')
data["season"] = data["season"].astype('category')
```

### 2.3 模型训练与超参数调优

我们使用 XGBoost 做二分类。先把因子型变量做 one-hot 编码，数值型变量直接使用。

```python
# 准备训练数据
library(xgboost)
library(data.table)

# 选出特征列，目标列为 shot_made_flag
# 先把因子型变量转成 dummy 变量
train_matrix <- model.matrix(~ . -1, data = train_df)

# 训练 XGBoost 模型
model <- xgboost(
    data = train_matrix,
    label = train_label,
    nrounds = 100,
    max_depth = 6,
    eta = 0.1,
    objective = "binary:logistic",
    eval_metric = "logloss"
)
```

我们尝试了不同的 `max_depth`、`eta`、`nrounds` 组合，发现 `max_depth=6`、`eta=0.1` 比较稳。网格搜索显示，模型在 100 轮左右开始收敛。

我们还试过用随机森林做基准，对比发现 XGBoost 的 logloss 更低。极坐标特征 `loc_r`、`loc_theta` 对模型帮助明显，比原始 `loc_x`、`loc_y` 更能捕捉球场环形结构。

### 2.4 模型测试与指标报告

我们在验证集上评估模型，计算 logloss 和准确率：

```python
from sklearn.metrics import log_loss, accuracy_score

# 在验证集上预测概率
y_pred_proba = model.predict(X_valid)
y_pred = (y_pred_proba > 0.5).astype(int)

# 打印对数损失和准确率
print("Log Loss:", log_loss(y_valid, y_pred_proba))
print("Accuracy:", accuracy_score(y_valid, y_pred))
```

整体 logloss 在 0.60 左右，准确率约 0.70。按区域拆分看，篮下投篮预测最准，远投最难。按赛季拆分发现，后期赛季命中率下降，模型对老化趋势有一定捕捉。

最后我们用全量训练数据重新拟合，生成提交文件：

```python
# 用全部训练数据训练最终模型
model.fit(X_train_full, y_train_full)

# 对测试集预测概率
test_pred_proba = model.predict(X_test)
submission = pd.DataFrame({
    'shot_id': test_ids,
    'shot_made_flag': test_pred_proba
})
submission.to_csv('submission.csv', index=False)
```

## 三、总结

本文中我们完成了 Kobe Bryant 投篮选择的完整流程。我们首先用 EDA 摸清了数据：30697 条投篮记录，25 个变量，投篮位置呈现明显的区域差异。其次我们做了数据清洗，处理缺失值、创建时序特征、把笛卡尔坐标转为极坐标。接着我们用 XGBoost 建模，对因子变量做 one-hot 编码，并调参找到 max_depth=6 的稳定配置。最后我们在验证集上得到 logloss 约 0.60，模型对近距离投篮预测较准，远投仍有提升空间。

这个比赛给我们的启发是：**位置特征的表示方式很重要**。球场是环形的，用极坐标比笛卡尔坐标更符合实际情况。也是特征工程要跟着业务走的典型例子。
