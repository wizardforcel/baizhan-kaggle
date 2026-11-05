# Facebook Recruiting IV: Human or Robot？

## 一、比赛简介

### 1.1 比赛背景

这是一场来自 Facebook Recruiting（脸书招聘）系列的比赛。在线拍卖网站上有一个让人头疼的问题：**机器人（bot）冒充真人去竞拍**，扰乱竞价秩序。机器人是指用脚本自动出价的程序，区别真人和机器人，对平台的安全很重要。

比赛给了三份数据：

- `train.csv`：2013 个竞拍者（bidder），每人有一个标签，告诉你他是「真人」还是「机器人」；
- `bids.csv`：全部 765 万条出价记录，每一条都记录了某次出价发生在哪个拍卖会、用的什么设备、IP 是什么等等；
- `test.csv`：4700 个待预测的竞拍者，没有标签。

难点在于：训练集只给了竞拍者的基本信息，真正能反映「行为"的数据全都藏在 765 万条出价记录里，我们必须自己动手把出价记录**聚合成每个竞拍者的行为特征**。

### 1.2 比赛目标

给定一个竞拍者的编号，预测这个人是真人还是机器人。

训练集 `train.csv` 有四列：

- `bidder_id`：竞拍者编号；
- `payment_account`：付款账号；
- `address`：地址；
- `outcome`：标签，`0` 代表真人（Human），`1` 代表机器人（Robot）。

出价记录 `bids.csv` 每个用户的每条出价一行，列有：`auction`（拍卖会）、`merchandise`（商品类别）、`device`（设备）、`time`（出价时间戳）、`country`（国家）、`ip`（IP 地址）、`url`（页面链接）等。我们要做的，就是把每个人名下所有的出价行为汇总成特征，再来判断他是人还是程序。

### 1.3 评价标准

比赛用 **AUC（ROC 曲线下面积）** 作为主评价指标。AUC 通俗说就是「把机器人和真人随机各挑一个出来，模型能正确把机器人排在更前面的概率」，取值在 0～1 之间，越接近 1 越强，乱猜是 0.5。因为这里「机器人」只有 103 个（占 5%），样本极度不均衡，单独看准确率没有意义，必须用 AUC 这类不依赖阈值的指标来比较模型。

> 小提示：这份数据的一个坑是——训练集里恰好也有每个竞拍者的基本信息表，也有人直接把它和出价表合并后就拿来训练，结果「准确率 100%」，其实是把不可直接用于测试的信息泄露进了模型。我们下面会走正规路线：把每个用户的出价行为汇总成稳健的统计特征。

---

## 二、解题过程

整体思路一句话：**把 765 万条出价记录，按竞拍者分组，汇总成「一个人用了多少台设备、多少个 IP、多少个付款账号、出价时间多分散」之类的行为特征，再用这些特征训练分类模型。** 我们按四步走。

### 2.1 数据探索（EDA）

我们首先把三份文件解压并读进来，看看每张表长什么样：

```python
import numpy as np
import pandas as pd

# 训练集与测试集：每个竞拍者一行
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# 出价记录：每个出价一行，约 765 万行
bids_df = pd.read_csv('../input/bids.csv')

# 查看每张表的基本信息
train_df.info()   # 2013 行 x 4 列
bids_df.info()    # 7656334 行 x 9 列
```

打印出来能看到 `train_df` 只有 `bidder_id`、`payment_account`、`address`、`outcome` 四列，而 `bids_df` 才是信息大户：每一行记录了这个人某次出价的拍卖会、设备、IP、国家、URL、时间戳等。

我们接着看标签分布，这是决定后面怎么做特征和模型的关键：

```python
# 统计真人（0）和机器人（1）的数量
train_df['outcome'].value_counts()
# 0.0    1910
# 1.0     103
# 说明：机器人只占 5% 左右，类别严重不均衡
```

103 个机器人对 1910 个真人，比例约 1:18。这意味着如果我们死磕「准确率」，一个"全猜真人"的模型就能拿到 95% 的准确率，但毫无用处。**必须用 ROC / AUC 来评估，并且训练时要小心类别不平衡的影响。**

为了看机器人和真人在行为上到底差在哪，我们把训练集和出价记录合并，单独把机器人挑出来，看看他们集中出现在哪些国家、IP、设备上：

```python
# 以 bidder_id 为键，把训练集和出价记录合并
df = pd.merge(train_df, bids_df)

# 只留下机器人的出价记录
mask = df['outcome'] == 1.0
df_bot = df[mask]

# 画图：机器人出价主要来自哪几个国家 / 设备 / IP / URL
import seaborn as sns
sns.countplot(x="country", data=df_bot,
              order=df.country.value_counts().iloc[:3].index)
sns.countplot(x="device", data=df_bot,
              order=df.device.value_counts().iloc[:3].index)
```

从图中能看到一个明显规律：**机器人的出价高度集中**。它们往往只来自极少数几个国家、几台设备和几个 IP。这给我们的直觉是：机器人的行为「太重复、太单一」，真人则会用五花八门的设备、来自不同地方。后面我们就把这种「多样性 / 重复度」量化成特征。

### 2.2 数据清洗与特征工程

出价记录有 765 万行，但我们要预测的对象是竞拍者，所以核心工作是**按竞拍者聚合**——把一个人的所有出价行为压缩成一行特征。

我们首先检查有没有重复、缺失值，然后按 `bidder_id` 分组，统计每个用户在这些维度上的**独立取值数量**和**时间分散程度**：

```python
# 按竞拍者分组，把每个字段上"不同的值有多少个"数出来
agg = bids_df.groupby('bidder_id').agg({
    'payment_account': 'nunique',   # 用了几个不同的付款账号
    'auction':        'nunique',    # 参与了多少场不同的拍卖
    'device':         'nunique',    # 用了多少台不同设备
    'country':        'nunique',    # 来自多少个不同国家
    'ip':             'nunique',    # 用了多少个不同 IP
    'url':            'nunique',    # 访问了多少个不同链接
    'time':           'std',        # 出价时间的标准差，衡量时间分散程度
})
agg.columns = [
    'n_similiar_payment_account', 'n_auction', 'n_device',
    'n_country', 'n_ip', 'n_url', 'time_sd'
]
```

这里的逻辑是：

- **独立取值越多**，说明这个人行为越「杂」，更像真人；
- **取值很少但出价却很多**，很可能是脚本在重复固定的几个动作，更像机器人；
- `time_sd` 统计出价时间的标准差：真人出价时间有早有晚、散得开，机器人往往集中在某几个瞬间，时间差小。

聚合后的特征表长这样（每个用户一行）：

| 列名 | 含义 |
| --- | --- |
| `n_similiar_payment_account` | 使用过相似的付款账号数量 |
| `n_auction` | 参与的不同拍卖会数量 |
| `n_device` | 使用过的不同设备数量 |
| `n_country` | 来自的不同国家数量 |
| `n_ip` | 使用过的不同 IP 数量 |
| `n_url` | 访问过的不同页面数量 |
| `time_sd` | 出价时间的标准差 |

把聚合结果拼回标签，得到 1984 个有效样本（清洗掉了一些没有有效出价记录或信息重复的用户），每个样本只有 7 个数值特征 + 1 个标签。

由于特征数值范围差异巨大（如 `n_ip` 最大 123，而 `time_sd` 却是万亿级别的大数），直接送进模型会让数值大的特征主导距离计算，我们接下来在建模前做标准化。同时因为类别不均衡，我们做**分层切分**——从机器人和真人里各取 80% 做训练、20% 做验证，保证两边验证集里机器人的比例都是平衡的：

```python
# 把机器人和真人各随机抽 80% 当训练集，剩下的当验证集
robot_names = robot_users
human_names = human_users
train = data[data.index.isin(robot_train + human_train)]
valid = data[data.index.isin(robot_valid + human_valid)]

# 输出量级：Total 1984 / Train 1586 / Valid 398
```

### 2.3 模型训练与超参数调优

我们共比较了 7 种分类器：线性判别分析（LDA）、二次判别分析（QDA）、逻辑回归（GLM）、朴素贝叶斯（NB）、K 近邻（KNN）、带径向基核的支持向量机（SVM）和随机森林（RF）。

由于大多数量级差异大，我们统一先做了**标准化**（中心化 + 缩放，让每个特征均值 0、方差 1）。评估采用 **10 折交叉验证、重复 10 次**，指标用 ROC：

```r
# R 的 caret 包：统一接口训练各种分类模型
fitControl <- trainControl(
  method = "repeatedcv",  # 重复交叉验证
  number = 10,            # 10 折
  repeats = 10,           # 重复 10 次，得到更稳定的估计
  classProbs = TRUE,      # 输出类别概率（算 ROC 需要）
  summaryFunction = twoClassSummary  # 用 ROC/灵敏度/特异度作为汇总指标
)

# 统一的建模函数：换 model_name 就能换算法
train_model <- function(model_name, tune = NULL) {
  train(outcome ~ ., data = train,
    preProcess = c("center", "scale"),    # 先标准化再训练
    trControl = fitControl,
    metric = "ROC",                       # 用 AUC 选最优参数
    method = model_name,
    tuneGrid = tune
  )
}

# 7 个模型：LDA / QDA / GLM / NB 不用调参，KNN / SVM / RF 需要调
model_lda <- train_model("lda")
model_glm <- train_model("glm")
model_nb  <- train_model("naive_bayes",
                         tune = data.frame(laplace=1, usekernel=FALSE, adjust=0))
model_knn <- train_model("knn", tune = data.frame(k = c(40)))   # 40 近邻
model_svm <- train_model("svmRadial",
                         tune = data.frame(C=0.5, sigma=1.0))   # 径向基核
model_rf  <- train_model("rf", tune = data.frame(mtry = c(2)))  # 每节点试 2 个特征
```

交叉验证（100 次重抽样的平均 ROC）结果是：

| 模型 | CV 平均 ROC |
| --- | --- |
| RF（随机森林） | **0.873** |
| KNN（40 近邻） | 0.790 |
| svmRadial（径向基 SVM） | 0.765 |
| LDA | 0.746 |
| QDA | 0.725 |
| GLM（逻辑回归） | 0.708 |
| NB（朴素贝叶斯） | 0.690 |

随机森林明显领先。它的最终模型是 500 棵树、每次分裂随机试 2 个特征，袋外（OOB）错误率 4.67%。这里有个值得注意的现象：随机森林的混淆矩阵显示它把**绝大多数机器人都漏掉了**（只对训练数据而言）——这正是类别不均衡的通病，模型倾向于把样本都判成占多数的「真人」，所以单纯看准确率会非常乐观，必须看 ROC。

### 2.4 模型测试与指标报告

我们用验证集（训练阶段没见过的 398 个竞拍者）来公平比较各模型：

```r
# 在验证集上算每个模型的 ROC
pred_ROC <- function(model, X) {
  pred <- data.frame(
    obs   = X$outcome,
    pred  = predict(model, X),
    Human = predict(model, X, type="prob")["Human"],
    Robot = predict(model, X, type="prob")["Robot"]
  )
  twoClassSummary(pred, lev = c("Human", "Robot"))[["ROC"]]
}

data.frame(
  Methods = c("RF","SVM","KNN","LDA","QDA","NB","GLM"),
  ROC = c(pred_ROC(final_rf, valid),  pred_ROC(final_svm, valid),
          pred_ROC(final_knn, valid), pred_ROC(final_lda, valid),
          pred_ROC(final_qda, valid), pred_ROC(final_nb,  valid),
          pred_ROC(final_glm, valid))
)
```

在验证集上的 ROC 排名：

| 模型 | 验证集 ROC |
| --- | --- |
| **RF（随机森林）** | **0.9641** |
| SVM | 0.9214 |
| KNN | 0.8685 |
| LDA | 0.7809 |
| QDA | 0.7680 |
| NB | 0.7660 |
| GLM | 0.7490 |

随机森林再次夺冠，验证集 ROC 达到 **0.964**。我们再看它的预测明细：

```text
          Human Robot        # 行是预测，列是真实
  Human   377     1
  Robot     0    20
```

真实为机器人的 20 个验证样本全部被正确找出来，真人 378 个里只误判了 1 个。这说明在验证集上，模型已经能相当可靠地区分两类人。

查看随机森林的特征重要性后发现，贡献最大的特征往往是 `n_similiar_payment_account`（相似付款账号数）和 `n_ip`、`n_url` 这类「行为多样性」指标——这和我们 EDA 时的直觉完全吻合：**机器人动作高度重复，真人行为各式各样**。

最后把从训练数据上学到的标准化参数和最优模型应用到整个测试集上，输出每个 `bidder_id` 是否为机器人的预测概率并生成提交文件：

```r
# 用全量数据再训练一次最终模型，并预测测试集
final_rf <- train_model("rf", tune = data.frame(mtry = c(2)))

# 生成提交文件：bidder_id + 预测为机器人的概率
submission <- data.frame(
  bidder_id = test$bidder_id,
  outcome   = predict(final_rf, test, type = "prob")[["Robot"]]
)
write.csv(submission, "submission.csv", row.names = FALSE)
```

---

## 三、总结

本文中我们完成了 Facebook Recruiting「人类还是机器人」的完整流程。我们首先做了 EDA，发现训练集只有 2013 个竞拍者、机器人仅占 5%，行为规律集中在出价记录里。其次我们做了数据清洗和特征工程，把 765 万条出价记录按竞拍者聚合成 7 个行为统计特征（独立设备数、IP 数、付款账号数、出价时间标准差等）。接着我们用 10 折重复交叉验证同时比较了 7 种分类模型，确认随机森林最优（CV ROC 约 0.873，最优参数 mtry=2）。最后我们在验证集上测得随机森林 ROC 0.964，能找回全部 20 个验证机器人。这套方案的核心经验是：**不要迷信准确率（全猜真人也 95%）**，在类别不均衡的任务里要用 ROC/AUC 评估，并且真正有价值的信息要靠「把行为记录聚合成行为特征」来挖掘。