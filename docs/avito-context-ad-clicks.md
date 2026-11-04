# Avito 上下文广告点击预测（Context Ad Clicks）

## 一、比赛简介

### 1.1 比赛背景

Avito 是一家做分类信息网站的公司：用户在站内搜索「二手自行车」「租房」「修水管」，网站一边返回结果，一边在结果旁边插入广告。这种**跟着搜索关键词一起出现的广告**叫「上下文广告」（contextual ad）——广告内容和用户此刻的搜索意图相关，所以点击率通常比随机投放高。

比赛要解决的问题是：**给定一次广告展示，预测用户会不会点它。** 数据来自 Avito 真实的搜索与广告日志，按表拆成了好几个文件。

这份数据的形态和常见的「一张 CSV」比赛很不一样，难点也在这里：

1. **数据分散在多张表里**，需要自己按 ID 关联。资料里给出的文件有 `AdsInfo.tsv`（广告信息）、`Category.tsv`（类目）、`Location.tsv`（地区）、`SearchInfo.tsv`（搜索信息）、`UserInfo.tsv`（用户信息）、`VisitsStream.tsv`（访问流）、`PhoneRequestsStream.tsv`（电话请求流），以及主表 `trainSearchStream.tsv` / `testSearchStream.tsv`。此外还有一个打包好的 `database.sqlite`，把所有表装在一个 SQLite 数据库里。
2. **数据量很大**。`trainSearchStream` 里光是「上下文广告」这一类（`ObjectType=3`）就有 **190,157,735 条**记录。材料里明确提醒：用 SQL 把这一批取出来「要花几分钟」。
3. **点击是稀有事件**。广告展示远远多于点击，所以模型不能只看准确率——一个「永远预测不点」的模型准确率极高，但毫无用处。

另外一个很重要的细节：数据里已经提供了一列 **`HistCTR`**，可以理解为「这个广告位、这个搜索上下文历史上的点击率」。比赛还额外给了一份 `sampleSubmission_HistCTR.csv`，从文件名看，官方的样例提交就是直接拿 `HistCTR` 当预测值。这等于告诉我们：**`HistCTR` 是最强的现成特征，任何模型都应该先和「直接用 HistCTR」比一比。**

### 1.2 比赛目标

给定一次广告展示，预测它被点击的概率。资料里没有完整列出主表的所有列——那篇脚本只用 SQL 取了自己需要的几列，所以我们也只介绍这几列：

| 列名 | 含义 |
| --- | --- |
| `ObjectType` | 展示对象的类型，`3` 表示上下文广告 |
| `HistCTR` | 历史点击率，官方预先算好的特征 |
| `IsClick` | 标签，1 表示点击，0 表示没有 |

各张辅助表互相通过 ID 关联，比如 `UserInfo.tsv` 以 `UserID` 为主键，记录每个用户的浏览器、操作系统、设备信息。测试表 `testSearchStream` 用 `TestId` 代替 `IsClick`，同时保留 `ObjectType` 和 `HistCTR`。我们要输出每条测试记录被点击的概率，提交两列：`ID` 和 `IsClick`。

### 1.3 评价标准

比赛用 **Logloss（对数损失）**，越低越好。公式是：

$$\text{Logloss} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

其中 $y_i$ 是真实标签（0 或 1），$\hat{p}_i$ 是模型给「点击」的概率。

用大白话解释：对每条样本，看模型给正确答案打了多少概率，取对数再取负。模型越确信正确答案，分数越接近 0；模型越确信错误答案，分数飞得越快。

这个指标有两个必须注意的地方：

- **它惩罚「过度自信」。** 说「这条 99% 会点」结果没点，罚得比说「这条 55% 会点」重得多。所以模型输出的必须是**校准过的概率**，不能只是「排序分」。
- **它不能出现 0 或 1。** 因为 log(0) 是负无穷，只要有一条预测是 0 而真实是 1，整个分数就变成无穷大。所以材料里的损失函数特意做了截断，把概率夹在 $[10^{-15}, 1-10^{-15}]$ 之间。

---

## 二、解题过程

要先说明一件事：这个比赛在资料里的可用页面很少，只有六篇，其中四篇只是「列出输入文件」，真正有建模内容的是**一篇用 R 写的逻辑回归脚本**。那篇脚本在 Kaggle 的 Python 环境里直接运行会报 `SyntaxError`（因为 R 的 `<-` 赋值在 Python 里不合法）。所以我们把它的逻辑**原样翻译成 Python**，并在下面每一步标明哪些做法来自原文、哪些是把 R 语法改写成 Python。

我们的主线非常朴素：**只用 `HistCTR` 这一列做逻辑回归**。听起来简单得过分，但它给出了一个可以对照的基线，也把「多表关联」这个工程问题拆解得清清楚楚。

### 2.1 数据探索（EDA）

我们首先确认手上到底有哪些文件：

```python
import os

# 列出输入目录：先看清有哪些表，再决定怎么读
print(sorted(os.listdir('../input/')))
# ['AdsInfo.tsv', 'Category.tsv', 'Location.tsv', 'PhoneRequestsStream.tsv',
#  'SearchInfo.tsv', 'UserInfo.tsv', 'VisitsStream.tsv', 'database.sqlite',
#  'sampleSubmission.csv', 'sampleSubmission_HistCTR.csv',
#  'testSearchStream.tsv', 'trainSearchStream.tsv']
```

可以看到同一份数据给了两种形式：**一堆 TSV** 和 **一个 SQLite 数据库**。TSV 适合单表读取，SQLite 适合做多表关联查询——后面我们两种都会用到。

读 TSV 时踩过一个很典型的坑。`UserInfo.tsv` 是用**制表符**分隔的，但 `read_csv` 默认按逗号分隔，于是整行被读成了一列：

```python
import pandas as pd

# 错误示范：不指定分隔符，制表符会被当成普通字符
df_wrong = pd.read_csv("../input/UserInfo.tsv")
print(df_wrong.head(2))
# 输出只有一列，值是 "1\t44073\t30\t2019\t9" 这样的整串

# 正确做法：显式指定 sep="\t"
df_user = pd.read_csv("../input/UserInfo.tsv", sep="\t")
print(df_user.head())
#    UserID  UserAgentID  UserAgentOSID  UserDeviceID  UserAgentFamilyID
# 0       1        44073             30          2019                  9
# 1       2        12505             20          2014                 85
```

这个小坑值得记下来：**读文件的第一步永远是确认分隔符和编码**。如果直接把「一整行当一列」的表喂进模型，后面所有特征都会变成字符串常量，模型什么也学不到，而且不报错——排查起来非常费时间。

`UserInfo` 只有五个字段：用户 ID、浏览器 Agent ID、操作系统 ID、设备 ID、设备家族 ID。这告诉我们一件事：**用户信息已经被匿名化成 ID 了**，没有年龄、性别这类可以直接用的画像。

接着我们看看主表里「上下文广告」占多少。资料里的做法是直接用 SQL 在主表上过滤：

```python
import sqlite3
import pandas as pd

# 连接打包好的 SQLite 数据库
conn = sqlite3.connect("../input/database.sqlite")

# 先看有哪些表，确认表名和 TSV 文件是否一一对应
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print(tables)

# 只取上下文广告（ObjectType=3）
# 这一步在原文里明确标注「要花几分钟」，因为表里有上亿行
query = "SELECT COUNT(*) AS n FROM trainSearchStream WHERE ObjectType=3"
print(pd.read_sql(query, conn))
# 约 190,157,735 条
```

一亿九千万行——这个数字决定了我们的策略：**不可能把全表拉进 pandas 再处理**，必须让数据库先做过滤、先在 SQL 层把数据量砍下来。

我们还想知道 `HistCTR` 长什么样，因为整条主线都围着它转：

```python
# 取一小批上下文广告样本，看看 HistCTR 的分布
sample = pd.read_sql(
    "SELECT HistCTR, IsClick FROM trainSearchStream WHERE ObjectType=3 LIMIT 1000000",
    conn,
)

print(sample['HistCTR'].describe())
# 关注：均值、最小值、最大值，以及有多少条是 0
print("HistCTR 为 0 的比例:", (sample['HistCTR'] == 0).mean())

# 顺带看标签比例：点击相对于展示非常稀少
print("点击率:", sample['IsClick'].mean())
```

`HistCTR` 作为「历史点击率」，天然就是一个 0～1 之间的概率。这非常重要：**它和我们最终要预测的东西是同一个量纲**，所以可以直接拿来做逻辑回归的唯一输入，也可以直接拿去当提交值当基线。

### 2.2 数据清洗

这份数据的清洗，核心不是「补缺失值、去重复行」，而是**在 SQL 层就把无关数据筛掉**。原文的做法只有一句查询：

```python
# 一次性取出上下文广告的记录。
# 原文在 Kaggle 上必须限量，因为不限量的话内存直接爆掉；
# 这里用 LIMIT 先取一千万条，做探索和训练都够用
sql = """
SELECT HistCTR, IsClick
FROM   trainSearchStream
WHERE  ObjectType = 3
LIMIT  10000000
"""
train_ctx = pd.read_sql(sql, conn)

print(train_ctx.shape)     # (10000000, 2)
```

为什么一定要加 `ObjectType=3`？因为 `trainSearchStream` 里混着多种展示对象（上下文广告只是其中一类）。不同 `ObjectType` 的点击行为完全不同——把「上下文广告」和别的类型混在一起训练，模型学到的平均点击率会被稀释，预测也就偏了。**先按业务含义切出子集，是做 CTR 类任务的第一步清洗。**

第二步是**分层抽样**。一亿九千万行里我们只想要一百万行来训练，但绝不能简单随机抽——万一抽出来的点击比例和全量差很远，模型就会系统性偏差。原文用 R 的 `createDataPartition` 按标签分层抽样，并在抽样后**打印两边的点击率做对比**：

```python
from sklearn.model_selection import train_test_split

# 分层抽样：按 IsClick 分层，保证抽样后的点击率和全量一致
sample_df, _ = train_test_split(
    train_ctx,
    train_size=1_000_000,       # 目标样本量
    stratify=train_ctx['IsClick'],
    random_state=42,
)

# 关键的一步：抽样后一定要核对点击率有没有走样
print("全量点击率:", train_ctx['IsClick'].mean())
print("抽样点击率:", sample_df['IsClick'].mean())
```

原文里这两行打印是特意留着的，理由是：**抽样是「悄悄出错」的典型环节**——代码不报错，但样本分布已经偏了。养成「抽样后核对关键比例」的习惯，能省掉很多莫名其妙的分数下降。

第三步是划分训练集和验证集，同样按标签分层：

```python
# 80% 训练、20% 验证；stratify 保证两边点击率一致
train_df, valid_df = train_test_split(
    sample_df,
    test_size=0.20,
    stratify=sample_df['IsClick'],
    random_state=42,
)

print(train_df.shape, valid_df.shape)
```

清洗到这里就结束了。我们没有做缺失值填充，也没有做异常值处理——因为这份数据的主要问题是「量太大」和「表太多」，而不是「脏」。**清洗的重点应该跟着数据的实际毛病走**，这句话在这个比赛里体现得特别明显。

### 2.3 模型训练与超参数调优

我们的模型是**一元逻辑回归**：输入只有 `HistCTR` 一个特征，输出点击概率。

为什么只用一个特征？有两个理由。

第一，原文就是这么做的，而且它报告了公开榜 0.05104 的成绩——这是一个**有据可查的基线**，值得先复现。

第二，`HistCTR` 本身已经浓缩了大量信息。从名字看，它是官方用历史日志统计出来的点击率；考虑到它的前缀是「Hist」（历史）而粒度是「这次上下文展示」，它很可能已经把「这个广告位在这个搜索上下文下的历史表现」聚合进去了。所以哪怕只用它一个特征，模型也不会是「瞎猜」。

模型的形式是：

$$\hat{p} = \sigma(a + b \cdot \text{HistCTR}) = \frac{1}{1 + e^{-(a + b \cdot \text{HistCTR})}}$$

逻辑回归会把输出压到 0～1 之间，正好满足概率的要求。原文用 R 的 `glm(..., family="binomial")` 拟合，我们翻译成 Python：

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# 逻辑回归只需要一列特征，注意 sklearn 要求二维数组，所以用 [['HistCTR']]
X_train = train_df[['HistCTR']].values
y_train = train_df['IsClick'].values

model = LogisticRegression(
    penalty='l2',        # L2 正则，防止系数被极少数极端样本带跑
    C=1.0,               # 正则强度的倒数，1.0 是默认的中等强度
    fit_intercept=True,  # 保留截距：基准点击率不是 0，必须有截距来吸收
    solver='liblinear',  # 小特征数下稳定且快
    max_iter=1000,       # 提高迭代上限，避免「未收敛」警告
)
model.fit(X_train, y_train)

# 看系数：coef 是 HistCTR 的斜率，intercept 是截距
print("系数:", model.coef_, "截距:", model.intercept_)
```

原文在拟合之后还做了 `summary(model)`，目的不是看分数，而是**检查系数是否显著**——如果 `HistCTR` 的系数不显著（p 值很大），说明这个特征其实没提供信息，整个方案就站不住。我们在 Python 里用 `statsmodels` 可以做同样的事：

```python
import statsmodels.api as sm

# statsmodels 默认不加截距，必须手动 add_constant
X_sm = sm.add_constant(train_df['HistCTR'])
glm = sm.GLM(train_df['IsClick'], X_sm, family=sm.families.Binomial()).fit()

# summary 里重点看两件事：coef 的符号（应该是正的，历史点击率高则更可能点）
# 和 P>|z| 这一列（应该远小于 0.05）
print(glm.summary())
```

这一步的意义在于：**建模不只是拟合，还要验证「你的假设成不成立」**。我们假设「历史点击率高的广告更容易被点」，如果系数是负的或者不显著，这个假设就被数据推翻了，后面再怎么调参都是白费。

顺便说一句「调参」：这个模型只有两个参数（斜率、截距），没有树深、学习率、迭代轮数这些可调项，所以**这场比赛我们没做超参数搜索**。原文也没有做。这也是选择简单模型的好处——省掉了调参的时间，可以把精力放在「数据怎么取、特征怎么来」上。

### 2.4 模型测试与指标报告

评估之前，我们先把比赛的损失函数写对。原文里的 `logloss` 函数有一段很容易被忽略的细节：

```python
def logloss(y, y_hat):
    """按比赛定义计算 Logloss，并对概率做截断"""
    threshold = 10 ** (-15)
    # 把预测概率夹在 [1e-15, 1-1e-15] 之间：
    # 否则一旦出现 0 或 1，log(0) 会变成 -inf，整个分数就报废了
    y_hat = np.clip(y_hat, threshold, 1 - threshold)

    # 逐条计算 -[y*log(p) + (1-y)*log(1-p)]，再取平均
    loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return loss
```

这段代码只有五行，但每一行都有原因。尤其 `np.clip` 那一行：**很多比赛提交失败不是因为模型差，而是因为某一条预测恰好是 0 或 1**。

接着在验证集上算分：

```python
# 预测验证集的点击概率
valid_pred = model.predict_proba(valid_df[['HistCTR']].values)[:, 1]

# 用比赛的指标评估
print("验证集 Logloss:", logloss(valid_df['IsClick'].values, valid_pred))
# 同时报告一个 AUC，方便判断「排序能力」和「概率校准」是否一致
from sklearn.metrics import roc_auc_score
print("验证集 AUC:", roc_auc_score(valid_df['IsClick'].values, valid_pred))
```

结果怎么读？

- **Logloss 要和「常数预测」比。** 如果整体点击率是 $c$，那么一个永远输出 $c$ 的模型 Logloss 约为 $-c\log c - (1-c)\log(1-c)$。我们的模型必须明显低于这个数，才算学到了东西。
- **Logloss 要和「直接用 HistCTR」比。** 这是这个比赛最容易被忽视的对照。如果逻辑回归的分数还不如把 `HistCTR` 原样交上去，那说明这层回归不但没帮忙，还引入了偏差——因为线性变换 $a + b \cdot \text{HistCTR}$ 有可能把概率整体推歪。
- **AUC 和 Logloss 要一起看。** AUC 只关心排序，不受概率尺度影响；Logloss 关心绝对概率。如果 AUC 不错但 Logloss 很差，问题几乎一定出在**概率校准**上（比如系数把概率整体抬高了），而不是模型没学到东西。

最后生成提交文件。注意测试集也要先按 `ObjectType=3` 过滤：

```python
# 测试集同样只取上下文广告；TestId 就是提交文件里的 ID
test_ctx = pd.read_sql(
    "SELECT TestId, HistCTR FROM testSearchStream WHERE ObjectType=3",
    conn,
)

# 用同一个模型预测概率
test_pred = model.predict_proba(test_ctx[['HistCTR']].values)[:, 1]

# 按比赛要求的两列格式写文件
submission = pd.DataFrame({'ID': test_ctx['TestId'], 'IsClick': test_pred})
submission.to_csv('glm_submission.csv', index=False)

# 收尾：关闭数据库连接
conn.close()
```

关于成绩，原文的作者给了一个很有参考价值的说明：这份脚本在 Kaggle 上为了能跑完，**限制了读入行数、也缩小了样本量**；而在本地用**完整数据库 + 1000 万条样本**训练时，公开榜拿到 **0.05104**。也就是说，这个只有一列特征的逻辑回归，已经是一个非常强的基线——后面的复杂模型必须显著低于 0.05104 才算有价值。

---

## 三、总结

本文中我们完成了 Avito 上下文广告点击预测的一套基线流程。我们首先用 EDA 摸清了数据的组织方式：数据被拆成十来张 TSV，同时打包成一个 SQLite 数据库，主表里 `ObjectType=3` 的上下文广告有约 1.9 亿条，官方还预先给了一列 `HistCTR` 和一份 `sampleSubmission_HistCTR.csv` 作为现成基线；读 `UserInfo.tsv` 时因为没指定 `sep="\t"` 把整行读成了一列，这个坑也记录了下来。其次我们在 SQL 层完成了主要清洗：只取 `ObjectType=3` 的子集，按标签分层抽出一百万行并**打印抽样前后的点击率做核对**，再按 8:2 分层划分训练集和验证集。接着我们只用一个特征 `HistCTR` 训练逻辑回归，并用 `statsmodels` 的 `summary` 检查系数符号和显著性，这一步代替了并不存在的「超参数调优」。最后我们写了一个带 `1e-15` 截断的 Logloss 函数，在验证集上同时报告 Logloss、AUC，并与「常数预测」「直接用 HistCTR」两个基线对照，再把测试集概率按 `ID`、`IsClick` 两列写成提交文件；原文作者报告，用完整数据加 1000 万条样本时公开榜成绩是 0.05104。

这个比赛留给我们三点体会。第一，**别急着上模型，先看数据是怎么组织的**：这份数据的关键难点是「多表 + 上亿行」，如果不在 SQL 层先过滤、先抽样，后面的模型再精巧也跑不起来。第二，**永远先做一个最简单的基线**：一列 `HistCTR` 加一个逻辑回归就能拿到 0.05104，这件事本身就是标尺——它告诉我们后面所有特征的增量到底值多少。第三，**指标要和模型输出对齐**：Logloss 要求校准过的概率，所以必须做截断、必须核对抽样后的点击率、必须把 Logloss 和 AUC 一起看；只看其中一个，很容易给出一个「排序不错但分数很难看」的模型却不知道原因。资料的可用页面很少，这篇 writeup 的代码是把原文的 R 脚本按同样逻辑翻译成 Python 得到的，主线之外没有更多可对照的方案，这也是我们在总结里要如实说明的一点。
