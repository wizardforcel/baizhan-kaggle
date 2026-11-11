# Deep-Pharma Multimodal Intelligence Challenge（DataHack 3.0）：药物联用风险预测

## 一、比赛简介

### 1.1 比赛背景

这个比赛关注的是**药物联用（drug-drug interaction）**的风险。现实中，一个病人往往同时吃好几种药，药物之间可能互相影响：有的只是让药效变弱，有的会产生严重的副作用，甚至危及生命。如果在开药之前就能预测「这两种药一起吃会怎样」，医生就能提前避开风险。

比赛把每种药的信息用**多种模态**描述：

- **SMILES**：一种用字符表示分子结构的写法，比如 `CC(=O)OC1=CC=CC=C1C(=O)O` 就是阿司匹林。它本质是一串文本，但里面编码了原子的连接关系；
- **作用机制（Mechanism）**：这种药在体内是怎么起作用的，通常是一段自然语言描述；
- **毒性（Toxicity）**：已知的毒副作用描述，也是文本。

这些信息形式完全不同——一个是类代码的字符串，一个是英文描述，一个是医学文本。把它们放进同一个模型，就是标题里说的「多模态」（multimodal）。

### 1.2 比赛目标

数据放在三个 CSV 里：

- `train.csv`：训练数据，每行是一对药物，带各种特征列；
- `test.csv`：测试数据，列和训练集一样，但没有答案；
- `sample_submission.csv`：提交样例，规定了输出格式。

每行有一个 `Pair_ID` 作为编号。要预测的东西有三组：

1. **`Severity`**：这一对药物联用的严重程度，是一个类别标签；
2. **`Target_Binary_*`**：50 个二分类目标，每个对应一种副作用，取值 0/1，表示这种副作用会不会出现；
3. **`Target_PRR_*`**：50 个连续目标，是对应副作用的 **PRR（比例报告比）**。PRR 是药物警戒里常用的统计量，用来衡量「吃这个药的人报告某副作用的比例，比不吃的人高多少倍」，数值越大说明关联越强。

所以这其实是一道「一个多分类 + 50 个二分类 + 50 个回归」的混合题。

### 1.3 评价标准

材料里的 notebook 没有写明排行榜的公式，我们只能从输出结构推断：三组目标的量纲和含义完全不同，不可能用一个指标衡量。合理的做法是**分组评估**——`Severity` 用分类指标（如宏平均 F1），50 个 `Target_Binary` 用每个标签的 F1 再平均，50 个 `Target_PRR` 用回归指标（如 RMSE 或 MAE）。最终排行分数多半是这几项的加权组合。

因此我们在本地也按三组分别报告，而不是硬凑一个总分。这一点很重要：如果只盯住其中一组，很容易把另外两组做坏。

---

## 二、解题过程

我们的整体思路非常朴素：**把所有模态的信息拼成一段文本，用 TF-IDF 变成向量，再给三类任务各配一个合适的模型。** 好处是实现简单、跑得快；代价是丢掉了 SMILES 里的结构信息，这一点我们在总结里会再谈。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，看清列的结构：

```python
import pandas as pd

base = "/kaggle/input/deep-pharma-multimodal-intelligence-challenge-data-hack-3-0/"
train = pd.read_csv(base + "train.csv")
test = pd.read_csv(base + "test.csv")
sub = pd.read_csv(base + "sample_submission.csv")

print(train.shape, test.shape, sub.shape)
print(sub.columns.tolist()[:10])   # 看输出列长什么样
```

第一次看列名会有点晕，因为目标列很多。我们用一个简单规则把它们分开：**答案列由提交样例决定**，凡是出现在 `sample_submission.csv` 里、又不是 `Pair_ID` 的列，都是要预测的目标：

```python
# 从提交样例里自动识别目标列，避免手写 100 个列名
target_cols = sub.columns.drop("Pair_ID").tolist()

# 按前缀分成三类
binary_targets = [c for c in target_cols if c.startswith("Target_Binary")]
prr_targets    = [c for c in target_cols if c.startswith("Target_PRR")]

print("严重程度列:", "Severity" in target_cols)
print("二分类目标数:", len(binary_targets))   # 50
print("PRR 目标数:", len(prr_targets))        # 50
```

这样做的好处是：如果主办方改了目标个数，代码不用动。特征列则反过来取——`test.csv` 里除了 `Pair_ID` 之外的列：

```python
feature_cols = [c for c in test.columns if c != "Pair_ID"]
print(len(feature_cols))

# 看看缺失情况：文本列经常是空的
print(train[feature_cols].isna().mean().sort_values(ascending=False).head(10))
```

我们重点看三件事：**每列缺失多少**（文本描述经常缺）、**`Severity` 的类别分布**（是否严重不平衡）、**50 个二分类目标的正例率**（副作用大多是罕见事件，正例可能只有百分之几）。前两件事决定要不要做类别加权，第三件事决定了不能用准确率评估二分类——全猜 0 也能有很高的准确率。

### 2.2 数据清洗

清洗分两块：文本缺失和回归标签缺失。

**文本缺失**：`SMILES`、`Mechanism`、`Toxicity` 这些列只要有空值，拼字符串时就会变成 `nan` 这个单词，白白污染词表。所以我们统一填成空字符串：

```python
def create_text_feature(df):
    """把一行里的所有特征列拼成一个「文档」。
       TF-IDF 只认文本，所以多模态融合的第一步是转成文本。"""
    df_filled = df[feature_cols].fillna("")   # 空值填空串，避免出现 "nan"
    # astype(str) 保证数字列也能拼；agg(' '.join) 把每行所有列连起来
    return df_filled.astype(str).agg(" ".join, axis=1)

X_train_text = create_text_feature(train)
X_test_text  = create_text_feature(test)

print(X_train_text.iloc[0][:300])   # 抽查一条，确认拼接顺序
```

注意这里**训练集和测试集必须用同一套 `feature_cols`**，顺序也要一致，否则同一个位置上的词含义就变了。

**回归标签缺失**：50 个 PRR 列里，很多药物对根本没有对应副作用的报告，值就是空的。我们没有删除这些行（删除会丢掉大量样本），而是用 0 填充：

```python
# PRR 缺失意味着「没有观察到这个副作用」，填 0 是一个合理的基线
y_train_prr = train[prr_targets].fillna(0)
```

填 0 而不是填均值，是因为 PRR 的含义是「高出多少倍」，没有报告约等于没有升高；填均值反而会人为制造出「中等风险」的假样本。

**最后一步是截断。** PRR 在定义上不可能是负数，但回归模型不知道这个约束，可能吐出 -0.3 这样的值。我们在预测之后统一截断：

```python
import numpy as np
pred_prr = np.clip(pred_prr, 0, None)   # 小于 0 的一律拉回 0
```

### 2.3 模型训练与超参数调优

**第一步，把文本变成向量。** 我们用 TF-IDF，它给每个词打一个权重：在本条记录里出现得越多、在整个语料里越少见，权重越高。`max_features=3000` 是出于显存的考虑——词表再大，Kaggle 的免费环境就容易爆内存：

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
X_train_vec = tfidf.fit_transform(X_train_text)   # 只在训练集上 fit
X_test_vec  = tfidf.transform(X_test_text)        # 测试集只做 transform
```

这里有个常见错误要避开：**`fit` 只能用训练集**。如果在测试集上也 `fit`，测试集的词频信息就泄漏进了模型，本地分数会虚高，线上却掉下来。

**第二步，三个任务各配一个模型。** 为什么不共用一个模型？因为三个任务的输出性质不同：一个是多分类，一个是 50 个独立的二分类，一个是 50 个连续值。强行用一个网络同时输出，需要精心设计损失权重；分开建模更简单，也更容易分别调。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.linear_model import LogisticRegression, Ridge

# 任务 1：严重程度是多分类，用随机森林
# class_weight='balanced' 是因为严重等级通常不平衡，稀有等级要被加权
clf_severity = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
clf_severity.fit(X_train_vec, train["Severity"])

# 任务 2：50 个二分类，用 MultiOutputClassifier 包一层逻辑回归
# 每个副作用单独训练一个分类器，互不干扰
clf_binary = MultiOutputClassifier(
    LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=-1))
clf_binary.fit(X_train_vec, train[binary_targets])

# 任务 3：50 个连续值，用 MultiOutputRegressor 包一层岭回归
reg_prr = MultiOutputRegressor(Ridge(alpha=1.0))
reg_prr.fit(X_train_vec, y_train_prr)
```

三个选择各有理由：

- 随机森林能自动处理特征之间的非线性组合，对 3000 维稀疏向量也不敏感，适合当多分类的「稳妥选项」；
- 逻辑回归在文本分类上是经典强基线，训练快、系数可解释，50 个标签各训一个也不会太慢；
- 岭回归（Ridge）就是加了 L2 正则的线性回归。`alpha=1.0` 控制正则强度：越大越保守，系数被压得越接近 0，能防止 3000 维特征上的过拟合。

**关于调参。** 这次我们只做了很轻的调整，主要卡在两个地方：`max_features`（3000 是速度和效果的折中）和 `alpha`（岭回归的正则强度）。没有做网格搜索的原因是：50 个输出意味着每跑一次就是 50 个模型，搜索成本是普通任务的几十倍。更划算的做法是先固定模型，把力气花在特征上——比如把 SMILES 单独抽出来做字符 n-gram，或者对文本做词形还原。这些我们留在了后续改进里。

### 2.4 模型测试与指标报告

预测和拼接提交文件的代码如下：

```python
# 三组预测分别做
pred_severity = clf_severity.predict(X_test_vec)
pred_binary   = clf_binary.predict(X_test_vec)       # 形状 (n_test, 50)
pred_prr      = reg_prr.predict(X_test_vec)          # 形状 (n_test, 50)
pred_prr      = np.clip(pred_prr, 0, None)           # 负值截断

# 按样例的列顺序拼回去
submission = pd.DataFrame({"Pair_ID": test["Pair_ID"]})
submission["Severity"] = pred_severity
submission = pd.concat(
    [submission, pd.DataFrame(pred_binary, columns=binary_targets)], axis=1)
submission = pd.concat(
    [submission, pd.DataFrame(pred_prr, columns=prr_targets)], axis=1)

# 最后一步最关键：完全按 sample_submission 的列顺序排
submission = submission[sub.columns]
submission.to_csv("submission.csv", index=False)
```

`submission = submission[sub.columns]` 这一行看着不起眼，却是最容易丢分的地方。多输出比赛的列名有上百个，手工拼很容易顺序错乱；而评审脚本通常按列位置取值，列错位了分数会莫名其妙地归零。

本地评估我们分三组做：

```python
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np

# 多分类：宏平均 F1，每个严重等级权重相同
print("Severity macro-F1:",
      f1_score(y_val_sev, pred_val_sev, average="macro"))

# 50 个二分类：每个标签单独算 F1 再平均（samples 平均不适合这里）
print("Binary macro-F1:",
      f1_score(y_val_bin, pred_val_bin, average="macro"))

# 50 个回归：先算 MSE 再开根号，单位和 PRR 一致
print("PRR RMSE:", np.sqrt(mean_squared_error(y_val_prr, pred_val_prr)))
```

这里用**宏平均**而不是准确率，是因为副作用大多是罕见事件：一个标签 95% 都是 0，全猜 0 也有 0.95 的准确率，但 F1 会是 0。宏平均会把每个标签同等对待，罕见副作用预测失败时分数会明显下降，更能反映真实水平。

需要说明的是，材料里这条 notebook 没有保存任何运行输出，所以这里不写具体的分数。我们能确认的只有流程本身：3000 维 TF-IDF、100 棵树、50 个逻辑回归、50 个岭回归，以及按三组分别评估的设计。

---

## 三、总结

本文中我们完成了 Deep-Pharma 多模态药物联用预测的一次基线尝试。我们首先读入训练集、测试集和提交样例，用提交样例的列名自动识别出三组目标——1 个严重程度、50 个副作用二分类、50 个 PRR 回归。其次我们做了两项清洗：把文本特征的空值填成空串并拼成一条文档，把 PRR 的缺失值填 0，并在预测后把负值截断。接着我们用 3000 维 TF-IDF 把多模态信息压成向量，为三类任务分别训练随机森林、多输出逻辑回归和多输出岭回归。最后我们按样例的列顺序生成提交文件，并用宏平均 F1 和 RMSE 分组评估。

这个比赛让我们最深的体会是：**「多模态」不等于「上大模型」**。把 SMILES、机制、毒性拼成一段文本喂给 TF-IDF，是一种非常粗糙的融合——它把分子结构当成了普通单词，丢掉了括号和原子连接所表达的结构信息。真正想提分，应该给 SMILES 单独做字符级 n-gram 或者用图神经网络，再和文本特征在模型层面融合。另一个体会是**多输出任务要分组看指标**：一个总分会把三组的差异抹平，分开看才知道自己到底强在哪、弱在哪。
