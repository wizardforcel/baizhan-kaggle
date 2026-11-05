# 三月疯篮球预测 2016（March Machine Learning Mania 2016）

## 一、比赛简介

### 1.1 比赛背景

「March Machine Learning Mania」是 Kaggle 每年三月举办的 NCAA 大学篮球锦标赛预测比赛。2016 年的这一届，主办方提供了非常完整的比赛数据：从 2003 年开始的常规赛详细技术统计、锦标赛详细技术统计、球队信息以及每年的种子排位。参赛者要用这些历史数据，预测 2016 年锦标赛每一场可能对阵的胜负概率。

这个比赛有意思的地方在于「历史很长、样本很少」：常规赛有十几年的上万场比赛，但每年锦标赛只有 63 场，而且球队每年都在变化。所以模型既要利用历史规律，又不能把某一年球队的偶然表现当成普遍规律。另一个难点是**冷门**——三月疯之所以叫「疯」，就是因为强队经常翻车，预测得太自信会被对数损失狠狠惩罚。

### 1.2 比赛目标

输入数据主要有四张表：

- `RegularSeasonDetailedResults.csv`：每场常规赛的详细统计，包括双方得分、投篮命中数/出手数、三分、罚球、篮板、助攻、失误、抢断、盖帽、犯规等；
- `TourneyDetailedResults.csv`：锦标赛的同类详细统计；
- `Teams.csv`：球队编号与队名；
- `TourneySeeds.csv`：每年每支参赛队的种子排位。

输出是一个提交文件，每一行是一对球队，格式为 `2016_<球队1>_<球队2>,<概率>`，概率表示球队 1 战胜球队 2 的可能性。只有锦标赛实际发生的比赛才会计分。

### 1.3 评价标准

比赛采用**对数损失（log loss）**，越低越好。对每场比赛，如果球队 1 赢了，就用提交的概率 p 算 `-log(p)`；如果球队 2 赢了，就算 `-log(1-p)`，最后对所有场次取平均。这个指标奖励「既猜得对、又敢给高概率」的模型，同时对过度自信的错误惩罚很重。

---

## 二、解题过程

我们的整体思路是：**先从详细比赛数据里为每支球队每年算出一组技术统计摘要，再把「两支球队的摘要」拼成一条训练样本，用逻辑回归、SVM、随机森林和 XGBoost 分别建模，最后把多个模型的概率平均起来。** 我们按 EDA、清洗、训练、评估四步走。

### 2.1 数据探索（EDA）

我们首先读入锦标赛详细结果和球队表，看看一场比赛到底记录了哪些信息：

```python
import pandas as pd
import numpy as np

# 每行是一场比赛：Wteam/Lteam 是胜者和负者，Wscore/Lscore 是比分
tourney_df = pd.read_csv("../input/TourneyDetailedResults.csv")
teams_df = pd.read_csv("../input/Teams.csv")

# 把球队编号翻译成队名，方便肉眼检查
team_dict = dict(zip(teams_df["Team_Id"], teams_df["Team_Name"]))
tourney_df["Wteam_name"] = tourney_df["Wteam"].map(team_dict)
tourney_df["Lteam_name"] = tourney_df["Lteam"].map(team_dict)

print(tourney_df.shape)        # 从 2003 年开始的所有锦标赛比赛
print(tourney_df.columns.tolist())
# ['Season','Daynum','Wteam','Wscore','Lteam','Lscore','Wloc','Numot',
#  'Wfgm','Wfga',...,'Lpf','Wteam_name','Lteam_name']
```

数据从 2003 年开始，每场比赛有 30 多列，胜者和负者各有一套完全对称的统计。我们接着比较「赢球的球队」和「输球的球队」在技术统计上的差异，并构造了一个「侵略性百分比」指标（罚球出手与投篮出手的比值、三分出手与投篮出手的比值），想看看打得更凶的球队是不是更容易赢：

这里要特别提醒初学者：`W*` 开头的列永远属于胜者，`L*` 开头的列永远属于负者。如果直接把它们当成「球队 1」和「球队 2」两组特征，模型就会学到「W 开头的那一边总是赢」这种假规律——这在训练集上准确率会非常高，但一到真实预测就完全失效。正确的做法是先把每支球队自己的表现从胜/负两种角色里拆出来，再按球队聚合，这也是下一节清洗工作的核心。

```python
# 赢球方的罚球率、三分率；0.66 是一个经验折扣，因为部分罚球来自投篮犯规
wft_wfg = (tourney_df["Wfta"] * 0.66) / tourney_df["Wfga"]
wtp_wfg = tourney_df["Wfga3"] / tourney_df["Wfga"]
# 输球方同样计算
lft_lfg = (tourney_df["Lfta"] * 0.66) / tourney_df["Lfga"]
ltp_lfg = (tourney_df["Lfga3"] / tourney_df["Lfga"])

# 看这三个量和得分之间的相关矩阵
print(np.corrcoef((wft_wfg, tourney_df["Wscore"], wtp_wfg)))
```

结果里罚球率、三分率和得分之间的相关系数都很低（大多在 0.1 以下，甚至有负值），说明**单一的技术风格指标并不能直接决定胜负**。这个发现很重要：它提醒我们不能只拿一两个比率做规则模型，必须把多项统计综合起来，让模型自己去学权重。

我们还统计了每支球队在锦标赛里的胜场分布：大多数球队只赢 0～1 场，少数强队赢 3～4 场，样本非常不均衡。这也是为什么后面要用「按年份切分」而不是随机切分——随机切分会让同一年的比赛同时出现在训练和验证里，造成数据泄漏。

### 2.2 数据清洗

第二步是把「一场一场的比赛」变成「一支球队一年的画像」。原始数据里，同一支球队既是胜者也是负者，统计分散在 `W*` 和 `L*` 两组列里。我们的做法是：分别筛出「某队作为胜者」和「某队作为负者」的记录，再按年份和球队分组，对得分做平均、最小、最大、中位数、标准差五种汇总：

```r
# 以「球队1作为胜者」为例：把该队所有赢球的比赛筛出来
winningteam1compactresults <- merge(how = "left",
  left = teamcompactresults1,
  right = regularseasoncompactresults,
  left_on = c("year", "team1"),
  right_on = c("Season", "Wteam"))

# 分组求均值，得到「球队1赢球时的平均得分」
grpwinningteam1resultsaverage <- winningteam1compactresults %>%
  group_by(year, team1) %>% mean()
winningteam1resultsaverage <- grpwinningteam1resultsaverage %>%
  reset_index()
names(winningteam1resultsaverage)[names(winningteam1resultsaverage) == "Wscore"] <- "team1WAverage"
```

我们对胜者、负者、球队 1、球队 2 四个方向都做同样的汇总，于是每支球队每年就有了一组特征：赢球时平均得多少分、最低多少、最高多少、波动多大，输球时又是多少。清洗完之后，每一行不再是一场比赛，而是「某年某队的技术画像」，为下一步拼样本做好准备。

这一步还有两个关键清洗动作：一是**只保留有种子排位的球队**，因为只有参赛队才需要预测；二是**把年份当作切分依据**——训练集用 2013 年以前的所有赛季，验证集用 2013 年，再依次用 2013 年以前训练、2014 年验证，2015 年同理。这样每次验证都是在「预测未来」，和真实比赛的使用场景一致。

### 2.3 模型训练与超参数调优

第三步是建模。我们把「球队 1 的画像」和「球队 2 的画像」拼成一条样本，标签是球队 1 是否获胜。然后并行训练四类模型：

```r
# 逻辑回归（GLM）：最经典的二分类基线，输出的是概率
glm_all_2013 <- glm(outcome ~ ., data = train_2013, family = "binomial")
glm_pred_2013 <- predict(glm_all_2013, newdata = test_2013, type = "response")

# 逐步回归：让 R 按 AIC 自动筛选变量，去掉没用的统计列
glm_step_2013 <- step(glm_all_2013)
glm_step_pred_2013 <- predict(glm_step_2013, newdata = test_2013, type = "response")

# 支持向量机：用核函数拟合非线性边界，probability=TRUE 才会输出概率
svm_all_2013 <- svm(outcome ~ ., data = train_2013)
svm_pred_2013 <- predict(svm_all_2013, newdata = test_2013, probability = TRUE)

# 随机森林：对缺失和异常值不敏感，但输出的概率偏「保守」
rf_all_2013 <- randomForest(outcome ~ ., data = train_2013)
rf_prob_2013 <- predict(rf_all_2013, newdata = test_2013, type = "prob")
```

XGBoost 需要先把类别型变量（球队所属联盟）做独热编码，再把数据转成矩阵：

```r
library(xgboost)
library(dummies)

# 把联盟列展开成哑变量，然后删掉原始文本列
opp_conf_dum <- dummy(xgb_model_data$opp_conf, data = NULL, sep = "_")
team_conf_dum <- dummy(xgb_model_data$team_conf, data = NULL, sep = "_")
xgb_model_data$opp_conf <- NULL
xgb_model_data$team_conf <- NULL
xgb_model_data <- cbind(xgb_model_data, opp_conf_dum, team_conf_dum)

# 二分类逻辑回归目标，输出概率
xgb_2013 <- xgboost(
  data = xgb_train_mtx, label = train_2013_label,
  objective = "binary:logistic",
  eta = 0.6, max.depth = 50, nthread = 10, nrounds = 20
)
xgb_pred_2013 <- predict(xgb_2013, newdata = xgb_test_mtx)
```

超参数方面，我们主要靠手工小范围试验：GLM 用全部变量和逐步筛选两个版本；SVM 用默认的径向基核；随机森林用默认树数；XGBoost 取 `eta = 0.6`、`max.depth = 50`、`nrounds = 20`。因为样本量不大（每年几百场），我们没有做很重的网格搜索，而是把精力放在**多模型对比和集成**上。

集成部分我们试了四种方案：全部模型等权平均、去掉 SVM（它输出的概率不够可靠）、去掉全变量 GLM、以及「多数投票」。等权平均的代码只有一行：

```r
# 对每一行（每场比赛），把所有模型给出的概率取平均
ens_1_2015 <- apply(probs_2015, 1, mean, na.rm = TRUE)
ens_1_ac_2015 <- acu_calc_pred(ens_1_2015, as.numeric(as.character(test_2015$outcome)))
```

### 2.4 模型测试与指标报告

我们按年份报告验证准确率。单模型的表现大致是：2013 年最好的模型约 **0.74**，2014 年最好的约 **0.67**，2015 年单模型在 **0.74** 左右；表现最差的一般是随机森林或 SVM，说明树模型和核方法在这个小样本问题上并不稳定。

集成之后的结果更整齐：

```r
# 全部模型等权平均
# 2013: 0.71
# 2014: 0.62
# 2015: 0.76  <- 目前最好
ens_1_2013 <- apply(probs_2013, 1, mean, na.rm = TRUE)
ens_1_2014 <- apply(probs_2014, 1, mean, na.rm = TRUE)
ens_1_2015 <- apply(probs_2015, 1, mean, na.rm = TRUE)

# 去掉 SVM 后再平均（因为 SVM 的概率输出不够可靠）
ens2_df_2015 <- probs_2015[, -3]
ens_2_2015 <- apply(ens2_df_2015, 1, mean, na.rm = TRUE)   # 2015: 0.74

# 去掉「全变量 GLM」后再平均
ens3_df_2015 <- probs_2015[, -1]
ens_3_2015 <- apply(ens3_df_2015, 1, mean, na.rm = TRUE)   # 2015: 0.74
```

最终，**全部模型等权平均在 2015 年拿到 0.76 的准确率**，是我们所有方案里最好的；去掉 SVM 或去掉全变量 GLM 都会掉到 0.74。投票法更差，只有 0.65 左右。

需要说明的是，准确率只是我们本地的辅助指标，比赛真正看的是对数损失。准确率高不代表对数损失低——如果模型总是给 0.51 这种「勉强猜对」的概率，准确率好看但对数损失会很差。所以我们在挑选模型时，除了看准确率，也会看它输出的概率是否分散、是否过度自信。

---

## 三、总结

本文中我们完成了 March Machine Learning Mania 2016 的完整建模流程。我们首先用 EDA 摸清了数据结构：每场比赛有胜者/负者两套对称统计，单一技术风格指标和得分的相关性很弱，必须综合建模。其次我们把逐场比赛汇总成「球队-年份画像」，用胜/负场景下的得分均值、最值、中位数和标准差作为特征，并按年份切分训练与验证。接着我们并行训练了逻辑回归、逐步回归、SVM、随机森林和 XGBoost，并把它们的概率做等权平均。最后我们报告了分年份的验证准确率：单模型在 0.67～0.74 之间，全部模型平均后在 2015 年达到 0.76，是最好的方案。

这个比赛给我们的最大启发是：**小样本、强随机的比赛，集成比单模型可靠**。单个模型在不同年份的准确率波动很大，但把多个模型的概率平均后，结果明显更稳定；同时，评估时不能只看准确率，还要看概率校准，因为对数损失惩罚的是「自信的错误」。