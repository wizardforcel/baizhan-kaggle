# 降雨预测挑战（Rainfall Prediction Challenge）

## 一、比赛简介

### 1.1 比赛背景

天气预测是机器学习最经典的应用场景之一。气象站每天都会记录温度、湿度、云量、日照时长、气压、风向等观测值，而「明天会不会下雨」是普通人最关心、也最直观的一个问题。传统上，气象学家靠物理模型和数值天气预报来回答这个问题；而在数据科学比赛里，我们换了一个思路：把历史上每天的气象观测和当天是否下雨这一标签配对，让模型自己从数据中学习「什么样的天气组合更容易下雨」。

这个比赛（Binary Prediction with a Rainfall Dataset / rainfall-prediction-challenge）提供的就是这样一份表格数据。每一行代表某一天的气象观测记录，最后一列 `rainfall` 表示当天是否下雨。它属于典型的**二分类问题**：输出只有「下雨」和「不下雨」两种可能。

这类问题看起来简单，实际上有几个难点。第一，下雨是一个受很多因素共同影响的复杂事件，单看某一个指标（比如湿度高）并不能决定下雨；第二，真实气象数据往往存在缺失值和不均衡，下雨的天数通常比不下雨的天数少；第三，很多气象变量之间高度相关，比如云量和日照时长此消彼长，如果直接把原始列丢给模型，模型很难自己构造出「云多且日照少」这类组合判断。因此，**特征工程**在这场比赛里起到了关键作用。

### 1.2 比赛目标

比赛给我们的输入是训练集和测试集两张表：

- 训练集包含 `id`、一系列气象特征（如 `pressure`、`maxtemp`、`tempp`、`humidity`、`cloud`、`sunshine`、`winddirection`、`windspeed` 等），以及目标列 `rainfall`；
- 测试集包含同样的特征列和 `id`，但没有 `rainfall`。

我们要做的是：对测试集里的每一天，预测当天是否会下雨，并把预测结果按提交样例的格式填进 `submission.csv`。提交文件只有两列：`id` 和 `rainfall`。

需要特别说明的是，`rainfall` 是 0/1 标签，`1` 代表下雨、`0` 代表不下雨。有些参赛者会提交概率值，有些会提交硬标签，具体取决于评分实现——这一点我们在 1.3 节细说。

### 1.3 评价标准

比赛使用 **ROC-AUC（Receiver Operating Characteristic - Area Under Curve）** 作为主要评价指标，同时也会看准确率（Accuracy）。

用大白话解释 ROC-AUC：模型对每一天都会输出一个「下雨的可能性」分数（0 到 1 之间）。我们把这个分数从高到低排序，然后看模型能不能把真正的下雨天排在不下雨天前面。AUC 衡量的是「随机抽一个下雨天和一个不下雨天，模型给下雨天更高分数的概率」。所以：

- AUC = 0.5 相当于瞎猜；
- AUC = 1.0 表示完美排序；
- AUC 越高越好。

因为 AUC 只看排序、不看具体的概率阈值，它对类别不均衡比较宽容，这也是气象、医疗、风控这类「正样本稀少」的问题偏爱 AUC 的原因。

此外，材料里也报告了 Accuracy。准确率需要先选一个阈值（默认 0.5）把概率变成 0/1 标签，再算猜对的比例。它直观但容易被不均衡带偏：如果 80% 的天都不下雨，模型全猜「不下雨」也能有 80% 的准确率。

---

## 二、解题过程

整场比赛我们走了一条很朴素但有效的路线：**先做 EDA 摸清数据和各类变量的关系，再构造几个有气象含义的组合特征，接着用逻辑回归、随机森林、XGBoost、CatBoost 四个模型做对比，最后选验证集上最好的随机森林在全量数据上重训并提交。** 下面按四步展开。

### 2.1 数据探索（EDA）

我们首先把三份 CSV 读进来，确认数据规模和列结构：

```python
import pandas as pd

# 三个文件：训练集、测试集、提交样例
train = pd.read_csv('/kaggle/input/competitions/sch2-reg-2026-d4-1/train.csv')
test  = pd.read_csv('/kaggle/input/competitions/sch2-reg-2026-d4-1/test.csv')
sample = pd.read_csv('/kaggle/input/competitions/sch2-reg-2026-d4-1/sample_submission.csv')

# 先看形状，确认训练集有多少行、有多少列
print("Train:", train.shape)
print("Test:", test.shape)

# 看前几行，认识每个气象列的含义和取值范围
display(train.head())
```

打印结果告诉我们，训练集和测试集的列几乎完全一致，唯一的差别是训练集多了 `rainfall`。每行就是一天的气象快照。

接着我们检查缺失值和标签分布：

```python
# 逐列统计缺失值个数
train.isnull().sum()

# 看下雨/不下雨各有多少天
train["rainfall"].value_counts()
```

从 `value_counts()` 能直观看到正负样本的比例。如果下雨天明显少于不下雨天，就说明这是一个**不均衡二分类**问题，后面评估时不能只看准确率。为了更直观，我们画了一张计数柱状图：

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
# countplot 会自动统计每个取值的样本数并画成柱状图
sns.countplot(data=train, x="rainfall")
plt.title("Rainfall 分布")
plt.show()
```

然后我们比较「下雨天」和「不下雨天」在各自变量上的平均值。这一步非常关键，它等于让数据自己告诉我们哪些变量有区分度：

```python
# 按 rainfall 分组求均值：下雨组一行、不下雨组一行
train.groupby("rainfall").mean()
```

通常能看到湿度、云量在下雨天更高，日照时长在下雨天更低，这些差异就是我们后面构造特征的依据。

最后我们画了相关性热力图，并单独把每个特征与 `rainfall` 的相关系数排序：

```python
plt.figure(figsize=(10, 6))
# 数值列两两之间的皮尔逊相关系数，颜色越暖表示正相关越强
sns.heatmap(train.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 只看各特征和目标 rainfall 的相关性，从大到小排列
rainfall_corr = train.corr()["rainfall"].sort_values(ascending=False)
print(rainfall_corr)
```

相关性排序让我们看到：某些变量与降雨正相关（如 humidity、cloud），某些负相关（如 sunshine）。但我们也注意到，单个变量的相关系数绝对值都不算特别高，这说明**降雨不是由单一变量决定的，而是多个变量的组合**。这正是我们要做特征工程的理由。

### 2.2 数据清洗

这份数据的清洗工作相对轻，主要处理三件事。

第一，**检查缺失值**。我们用 `train.isnull().sum()` 逐列查看。材料中的流程先确认缺失情况，再决定是否填补；对于树模型（随机森林、XGBoost、CatBoost），少量缺失通常可以容忍，XGBoost 甚至会自动学习缺失值的方向。因此我们没有做复杂的插补，保持数据的原始状态。

第二，**删除无意义的 `id` 列**。`id` 只是行号，和是否下雨没有任何因果关系。如果把它保留为特征，模型可能会记住「id 小的样本更容易下雨」这种伪规律，在测试集上反而有害：

```python
# id 不携带任何气象信息，留在特征里只会带来噪声
train = train.drop("id", axis=1)
test  = test.drop("id", axis=1)

print(train.shape)
print(test.shape)
```

第三，**统一训练集和测试集的特征列**。删掉 `id` 之后，训练集比测试集多一列 `rainfall`。我们在建模前用 `train.drop("rainfall", axis=1)` 得到特征矩阵，保证两边列顺序完全一致——如果列顺序错位，模型会把「湿度」当成「风速」来预测，结果会莫名其妙地差。

我们没有对连续变量做标准化之后再喂给树模型，因为决策树只关心阈值切分，不关心数值尺度；但在逻辑回归那一组实验里，我们确实套了 `StandardScaler`，因为线性模型对量纲敏感。这个区别我们会在 2.3 节说明。

### 2.3 模型训练与超参数调优

我们其次做特征工程。根据 EDA 的观察，我们手工构造了三类交叉特征，每一类都对应一个气象直觉：

```python
# 1) 湿度 × 云量：又潮又阴，下雨概率应该更高
train["humidity_cloud"] = train["humidity"] * train["cloud"]
test["humidity_cloud"]  = test["humidity"] * test["cloud"]

# 2) 湿度 / 日照：潮湿但没太阳，说明水汽不容易被晒干
#    分母加 1 是为了防止 sunshine 为 0 时除零
train["humidity_sunshine_ratio"] = train["humidity"] / (train["sunshine"] + 1)
test["humidity_sunshine_ratio"]  = test["humidity"] / (test["sunshine"] + 1)

# 3) 昼夜温差：温差大往往对应晴朗少云的天气
train["temp_range"] = train["maxtemp"] - train["mintemp"]
test["temp_range"]  = test["maxtemp"] - test["mintemp"]
```

后来我们又补了一个「云量 / 日照」比值：

```python
# 云多、日照少 → 比值大 → 更可能阴雨
train["cloud_sunshine_ratio"] = train["cloud"] / (train["sunshine"] + 1)
test["cloud_sunshine_ratio"]  = test["cloud"] / (test["sunshine"] + 1)

# 验证新特征与目标的相关性，确认它不是噪声
print(train[["cloud_sunshine_ratio", "rainfall"]].corr()["rainfall"])
```

接着我们检查这些新特征与 `rainfall` 的相关系数，确认它们确实比原始单列更有信息量，然后才让它们进入模型。

**数据划分。** 因为测试集没有标签，我们必须自己留一份验证集来判断模型好坏。我们用分层抽样（`stratify=y`），保证训练集和验证集里下雨/不下雨的比例一致：

```python
from sklearn.model_selection import train_test_split

x = train.drop("rainfall", axis=1)   # 所有气象特征（含新构造的）
y = train["rainfall"]                # 0/1 标签

# 80% 训练、20% 验证；stratify=y 让两边的正负样本比例一致
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y
)

print("X Train:", x_train.shape)
print("X Test :", x_test.shape)
```

**基线模型：逻辑回归。** 我们先跑一个最简单的线性分类器，作为「及格线」。逻辑回归的优点是快、可解释，缺点是只能学线性边界：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# max_iter=1000 防止迭代次数不够导致不收敛警告
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(x_train, y_train)

# predict 给出 0/1，predict_proba 给出概率，AUC 需要后者
pred_lr = model_lr.predict(x_test)
prob_lr = model_lr.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, pred_lr))
print("ROC-AUC :", roc_auc_score(y_test, prob_lr))
```

逻辑回归对特征尺度敏感，所以我们又把它包进 `Pipeline`，先标准化再训练，看看缩放能不能带来提升：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 把「标准化 + 逻辑回归」绑成一个整体，避免验证时信息泄漏
model_lr_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
model_lr_scaled.fit(x_train, y_train)
prob_lr_scaled = model_lr_scaled.predict_proba(x_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, prob_lr_scaled))
```

**树模型组。** 接下来我们依次训练随机森林、XGBoost 和 CatBoost。随机森林的设定是 300 棵树、最大深度 8：

```python
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=300,    # 300 棵树，多棵树投票能降低方差
    max_depth=8,         # 限制深度，防止每棵树把训练集背下来
    random_state=42      # 固定随机种子，保证结果可复现
)
model_rf.fit(x_train, y_train)

prob_rf = model_rf.predict_proba(x_test)[:, 1]
print("RandomForest ROC-AUC:", roc_auc_score(y_test, prob_rf))
```

XGBoost 我们用了 300 轮、深度 4、学习率 0.05，并加入行采样和列采样来抑制过拟合：

```python
from xgboost import XGBClassifier

model_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,             # 深度小一些，单棵树更弱，靠 boosting 逐步修正
    learning_rate=0.05,      # 每棵树只贡献一点点，训练更稳
    subsample=0.8,           # 每轮只用 80% 的样本，增加随机性
    colsample_bytree=0.8,    # 每棵树只用 80% 的特征
    random_state=42,
    eval_metric="logloss"
)
model_xgb.fit(x_train, y_train)
prob_xgb = model_xgb.predict_proba(x_test)[:, 1]
print("XGBoost ROC-AUC:", roc_auc_score(y_test, prob_xgb))
```

CatBoost 则使用 300 次迭代、深度 6：

```python
from catboost import CatBoostClassifier

model_cb = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=0        # 关闭训练日志，输出更干净
)
model_cb.fit(x_train, y_train)
prob_cb = model_cb.predict_proba(x_test)[:, 1]
print("CatBoost ROC-AUC:", roc_auc_score(y_test, prob_cb))
```

**关于超参数调优。** 我们在这个比赛里没有做大规模的网格搜索，而是采取了「先固定一组合理参数、比较模型家族、再围绕最优家族微调」的策略。原因很实际：数据量不大，单个模型训练只要几十秒，但一场完整的 `GridSearchCV` 要跑几十组组合，收益往往只有千分位。我们的做法是先用经验值（树的数量 300、深度 4～8、学习率 0.05）跑通四个模型，看谁在验证集上 AUC 最高，然后把注意力集中在那个家族上。材料的结论是：随机森林在这份数据上表现最好，因此我们把它定为最终模型。

还有一个细节值得记录：随机森林的特征重要性可以反过来验证我们的特征工程有没有用。我们把重要性排序打印出来，如果 `humidity_cloud`、`cloud_sunshine_ratio` 这类新特征排在前面，就说明构造方向是对的：

```python
importance = pd.DataFrame({
    "Feature": x_train.columns,
    "Importance": model_rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

display(importance.head(15))   # 看排名前 15 的特征
```

### 2.4 模型测试与指标报告

我们对四个模型在同一份验证集上做了对比，报告 Accuracy 和 ROC-AUC 两个指标，并打印混淆矩阵和分类报告：

```python
from sklearn.metrics import classification_report, confusion_matrix

# 以随机森林为例
pred_rf = model_rf.predict(x_test)
prob_rf = model_rf.predict_proba(x_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, pred_rf))
print("ROC-AUC  :", roc_auc_score(y_test, prob_rf))

# 混淆矩阵：行是真实标签，列是预测标签
print(confusion_matrix(y_test, pred_rf))

# 每个类别的 precision / recall / f1
print(classification_report(y_test, pred_rf))
```

对比下来，逻辑回归作为基线能给出一个尚可的 AUC，但明显低于树模型；XGBoost 和 CatBoost 与随机森林接近，最终随机森林在验证集上略微领先。材料里记录的 Kaggle 公开榜分数为 **0.76133**（按该比赛采用的评分口径）。

这里我们想强调一个容易被忽略的点：**AUC 和 Accuracy 要一起看**。AUC 高说明模型的排序能力好，但如果正负样本不均衡，0.5 阈值切出来的硬标签可能偏向多数类，Accuracy 就会虚高或虚低。混淆矩阵能帮我们看清模型到底把哪一类判错了——是漏报了下雨天，还是把晴天误报成雨天。对降雨预测来说，漏报和误报的代价并不相同，实际业务里可以通过调整阈值来权衡。

确认模型可用之后，我们用**全部训练数据**重新训练随机森林（而不是只用那 80%），再多喂 20% 的数据通常能让模型更稳：

```python
# 用全量训练集重训，注意此时的 x_full 已包含验证集那部分样本
x_full = train.drop("rainfall", axis=1)
y_full = train["rainfall"]

final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)
final_model.fit(x_full, y_full)

# 对测试集预测，并写入提交样例
test_predictions = final_model.predict(test)
sample["rainfall"] = test_predictions
sample.to_csv("submission.csv", index=False)
```

最后我们检查了提交文件的形状和前几行，确认行数与测试集一致、列名与样例一致。这一步虽然琐碎，但比赛里因为提交格式错误而白忙一场的情况并不少见。

---

## 三、总结

本文中我们完成了降雨预测挑战的完整流程。我们首先用 EDA 摸清了数据结构：每行是一天的气象观测，目标是 0/1 的 `rainfall`，各特征与目标的相关性都不算高，说明降雨是多个变量共同作用的结果。其次我们做了轻量清洗——检查缺失、删掉无信息的 `id`、对齐训练与测试的特征列。接着我们围绕气象直觉构造了 `humidity_cloud`、`humidity_sunshine_ratio`、`temp_range`、`cloud_sunshine_ratio` 四个交叉特征，并用分层抽样划出验证集。然后我们依次训练了逻辑回归（含标准化版本）、随机森林、XGBoost、CatBoost，用 ROC-AUC 和 Accuracy 对比，发现随机森林表现最好。最后我们用全量数据重训随机森林，生成提交文件，公开榜分数为 0.76133。

这个比赛给我们的启发是：**在表格数据上，好的组合特征往往比复杂的模型更值钱**。湿度和云量单独看都只是弱信号，乘在一起就能表达「阴湿」这个有物理意义的天气状态；而模型方面，随机森林这种「多棵树投票」的方法在中小规模数据上稳定、不易过拟合，未必输给更花哨的 boosting。对我们入门者来说，先把 EDA 和特征工程做扎实，再谈调参，是一条更靠谱的路径。
