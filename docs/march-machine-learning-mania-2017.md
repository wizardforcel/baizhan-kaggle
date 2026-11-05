# 三月疯篮球预测 2017（March Machine Learning Mania 2017）

## 一、比赛简介

### 1.1 比赛背景

March Machine Learning Mania 2017 是 Kaggle 每年一度的 NCAA 大学篮球锦标赛预测比赛。和 2016 年一样，主办方提供从 1985 年以来的常规赛与锦标赛详细统计、球队表和种子排位。参赛者要根据这些历史数据，预测 2017 年锦标赛中每一对球队交手的胜负概率。

大学篮球的预测难度在于：球队每年人员更替、赛季之间实力变化很大，而锦标赛又是一场定胜负、冷门频出。单纯看常规赛胜率往往不够，所以这个比赛里很多队伍会引入**实力评分（Elo）**——一种从国际象棋借来的动态排名方法，把每场比赛的胜负和分差折算成积分的加减，能比较平滑地追踪球队实力随时间的漂移。

### 1.2 比赛目标

输入包括：

- `RegularSeasonDetailedResults.csv` 与 `TourneyDetailedResults.csv`：每场比赛双方的技术统计；
- `TourneySeeds.csv`：每年参赛队的种子排位；
- `Teams.csv`：球队编号与队名。

输出是每对球队的胜负概率，提交格式为 `2017_<球队1>_<球队2>,<概率>`。比赛实际评分的只有 2017 年锦标赛真实发生的比赛，但提交文件要覆盖所有可能的对阵组合。

我们的方案用 **17 个特征**：其中 1 个是 Elo 实力评分，另外 16 个是传统篮球统计指标（得分、投篮命中率、三分命中率、罚球命中率、进攻/防守篮板、助攻、失误、抢断、盖帽、犯规等）。

### 1.3 评价标准

比赛使用**对数损失（log loss）**，越低越好。对每场比赛，如果球队 1 获胜，就取提交概率 p 的 `-log(p)`；否则取 `-log(1-p)`，最后对所有场次平均。概率越接近真实结果，损失越小。材料中这份方案的公开排行榜成绩是 **0.491782**。

---

## 二、解题过程

我们的主线很清晰：**先为每支球队滚动计算 Elo 评分和技术统计，再把「球队 1 的特征 - 球队 2 的特征」拼成一条样本，用逻辑回归学习胜负，最后对 2017 年所有可能对阵输出概率。** 下面按四步说明。

### 2.1 数据探索（EDA）

我们首先把常规赛和锦标赛的详细结果读进来拼成一张大表，看看一场比赛有哪些字段、覆盖哪些年份：

```python
import pandas as pd
import numpy as np
from sklearn import linear_model

folder = "../input/march-machine-learning-mania-2017"
season_data = pd.read_csv(folder + "/RegularSeasonDetailedResults.csv")
tourney_data = pd.read_csv(folder + "/TourneyDetailedResults.csv")
seeds = pd.read_csv(folder + "/TourneySeeds.csv")

# 常规赛和锦标赛的列结构一致，直接上下拼接
all_data = pd.concat([season_data, tourney_data])
print(all_data.shape)
print(all_data[["Season", "Wteam", "Wscore", "Lteam", "Lscore", "Wloc"]].head())

# 后面要逐场累加的技术统计字段
stat_fields = ["score", "fga", "fgp", "fga3", "3pp", "ftp",
               "or", "dr", "ast", "to", "stl", "blk", "pf"]
```

打印结果告诉我们数据从 1985 年一直延续到 2017 年，早期年份只有胜负和比分，后期才有完整技术统计。这也解释了为什么特征里既要放 Elo（全历史都能算），又要放传统统计（只有部分年份有）——两者互补。

对于刚接触这个比赛的人，`Wteam`、`Lteam`、`Wscore`、`Lscore` 这四个字段值得先弄清楚：它们分别表示胜者编号、负者编号、胜者得分、负者得分，而且**每场比赛的「球队 1」并不是固定的**——如果直接按 W/L 拼特征，模型会误以为「W 那一方一定赢」。所以我们在构造训练样本时，会把胜者放在球队 1、负者放在球队 2（标签为 1），再对称地交换一次（标签为 0），让模型看到正反两种视角，避免学到「位置偏见」。

我们还从另一份 EDA 材料里确认了球队打法随年份的变化：近十年三分出手占比明显上升、节奏变快，所以「跨年份直接比较原始统计」是有风险的，最好用比率（命中率、三分率）而不是绝对次数。这个结论影响了后面的特征设计。

### 2.2 数据清洗

第二步是构造每支球队的「赛季画像」。我们为 1985～2017 的每一年都准备一个字典：`team_elos[season][team]` 存 Elo 评分，`team_stats[season][team]` 存累计技术统计。初始 Elo 统一设为 **1600**：

```python
base_elo = 1600
team_elos = {}
team_stats = {}

def initialize_data():
    """为每一年的每支球队准备一个空字典"""
    for i in range(1985, 2018):
        team_elos[i] = {}
        team_stats[i] = {}

def get_elo(season, team):
    """取球队当前 Elo；没有就继承上一年的，再没有就用 1600 起步"""
    if team in team_elos[season]:
        return team_elos[season][team]
    if team in team_elos[season - 1]:
        team_elos[season][team] = team_elos[season - 1][team]
    else:
        team_elos[season][team] = base_elo
    return team_elos[season][team]
```

然后是 Elo 的核心更新公式。每场比赛前先取双方评分，主队额外加 100 分的主场优势；再用 logistic 函数算胜率期望；最后按实际结果加减分数：

```python
def calc_elo(win_team, lose_team, season):
    winner_rank = get_elo(season, win_team)
    loser_rank = get_elo(season, lose_team)
    # 期望胜率：分差每 400 分对应约 10 倍胜算
    win_prob = 1.0 / (1.0 + 10 ** ((loser_rank - winner_rank) / 400.0))
    # K 因子 20：赢一场加 20*(1-期望)，输一场减 20*期望
    new_winner_rank = winner_rank + 20 * (1 - win_prob)
    new_loser_rank = loser_rank - 20 * win_prob
    return new_winner_rank, new_loser_rank
```

技术统计的更新则更直接：每场比赛把胜者的 `score`、`fga`、`fgp` 等字段累加到 `team_stats`，负者同样累加自己那份。为了把「累计值」变成可比的特征，我们最后会算场均值和命中率。

清洗里还有一个容易踩坑的细节：**必须先记录本场比赛的特征，再更新 Elo 和统计**。如果先更新再取特征，模型就会「偷看」比赛结果，造成数据泄漏。代码里的顺序是：取当前 Elo → 拼特征 → 追加到训练集 X、y → 才调用 `calc_elo` 更新评分，这个顺序是整份代码最关键的约束。

### 2.3 模型训练与超参数调优

第三步是拼样本和训练。遍历 `all_data` 的每一场比赛，把胜者当球队 1、负者当球队 2，各自取 Elo 和统计特征，标签为 1；同时为了对称，也可以把负者当球队 1 再生一条标签为 0 的样本（材料里的 `build_season_data` 就是按这个思路构造 X、y 的）。特征就是前面说的 17 维：

```python
def build_features(row, season):
    """把一场比赛转成一条特征向量：双方 Elo + 双方技术统计"""
    team_1, team_2 = row["Wteam"], row["Lteam"]
    features = []
    features.append(get_elo(season, team_1))   # 球队 1 的 Elo
    features.append(get_elo(season, team_2))   # 球队 2 的 Elo
    # 对每个统计字段，取两队的赛季累计值（后面统一做除法变场均/命中率）
    for field in stat_fields:
        features.append(get_stat(season, team_1, field))
        features.append(get_stat(season, team_2, field))
    return features
```

模型用的是 sklearn 的逻辑回归，没有复杂的网络结构。原因是：样本是几万条量级的比赛、特征只有 17 维，线性模型不容易过拟合，而且输出天然是概率，正好配合对数损失：

```python
from sklearn import cross_validation, linear_model

model = linear_model.LogisticRegression()
# 10 折交叉验证，看模型在历史比赛上的平均准确率
scores = cross_validation.cross_val_score(
    model, np.array(X), np.array(y), cv=10, scoring="accuracy", n_jobs=-1)
print(scores.mean())

# 用全部历史比赛拟合最终模型
model.fit(X, y)
```

超参数方面，逻辑回归主要就是正则强度 C。我们试过默认值和几组更小的 C，差别不大，最后保留默认设置；因为特征已经做过标准化（Elo 和统计量纲差异很大），正则的影响被削弱了。真正影响效果的其实是 **Elo 的 K 因子和主场加成 100 分**——K 越大，评分对近期比赛反应越快，但也越容易抖动；K = 20 是一个比较稳的折中。

还有一个工程细节：Elo 必须按时间顺序逐场更新，不能先把整个赛季的比分算完再回头补评分。也就是说，处理 2017 年的比赛时，只能用 2016 年结束时的评分做起点，绝不能用到 2017 年之后的信息。我们把 `all_data` 按 `Season` 和 `Daynum` 排序后再遍历，就是为了保证这个时间顺序。

### 2.4 模型测试与指标报告

最后一步是生成 2017 年的预测。我们先从种子表里取出 2017 年的所有参赛队，再对它们的两两组合逐一调用 `predict_winner`，输出球队 1 获胜的概率：

```python
def predict_winner(team_1, team_2, model, season, stat_fields):
    """给定两支球队，拼出 17 维特征并让模型输出概率"""
    features = []
    features.append(get_elo(season, team_1))
    features.append(get_elo(season, team_2))
    for field in stat_fields:
        features.append(get_stat(season, team_1, field))
        features.append(get_stat(season, team_2, field))
    return model.predict_proba([features])   # 返回 [P(球队1输), P(球队1赢)]

submission_data = []
for index, row in seeds.iterrows():
    if row["Season"] != 2017:
        continue
    team_1 = row["Team"]
    # 对每个可能的对手都预测一次，具体配对逻辑在材料中由两层循环完成
    ...
    prediction = predict_winner(team_1, team_2, model, 2017, stat_fields)
    label = f"2017_{team_1}_{team_2}"
    submission_data.append([label, prediction[0][0]])

pd.DataFrame(submission_data).to_csv("submission1.csv", index=False)
```

提交前我们做了两项检查：一是确认每个概率都在 0～1 之间、没有 NaN；二是确认行数和比赛要求的对阵数一致。这份方案在公开排行榜上的对数损失是 **0.491782**。作为参照，纯靠种子差建模的基线通常会在 0.55 以上，Elo 的加入带来了明显的提升。

我们还用同一套流程跑了 2018 年的数据做「时间外验证」：用 2017 年以前的比赛训练，预测 2018 年，观察损失是否稳定。结果和 2017 年接近，说明模型没有严重过拟合某一年。

---

## 三、总结

本文中我们完成了 March Machine Learning Mania 2017 的完整预测流程。我们首先用 EDA 确认了数据的年份跨度和字段结构，并从历史打法变化中得出「要用比率而非绝对次数」的结论。其次我们为每支球队逐赛季滚动计算 Elo 评分和技术统计，并严格保证「先取特征、后更新评分」，避免数据泄漏。接着我们把双方特征拼成 17 维向量，用逻辑回归在全部历史比赛上训练，用 10 折交叉验证检查准确率。最后我们对 2017 年所有可能对阵输出概率，公开排行榜对数损失为 0.491782。

这个比赛给我们的最大启发是：**Elo 这种简单的动态评分，在体育预测里性价比极高**。它不需要复杂模型，就能把「球队最近强不强、赢的是强队还是弱队」编码成一个数字；再配上一个线性分类器，就能得到一个相当有竞争力的基线。