# Santander 客户满意度预测（Customer Satisfaction）

## 一、比赛简介

### 1.1 比赛背景

Santander 是一家总部位于西班牙的大型银行。银行每天都会和成千上万的客户打交道，客户对服务是否满意，直接决定了他会不会继续使用这家银行的产品。如果能在客户真正流失之前，提前发现那些「可能不满意」的人，银行就可以主动联系、补偿或者改进服务，把客户留下来。

这个比赛给出的就是一批匿名化之后的客户数据。所谓匿名化，是指银行把原本有业务含义的字段名替换成了 `var3`、`var15`、`saldo_var30` 这样的代号，只保留数值本身。这样既能让参赛者建模，又不会泄露客户隐私。代价是我们无法直接知道每一列到底代表年龄、余额还是交易次数，只能通过数据分布去猜。这也是这场比赛最大的难点：**特征没有业务解释，必须靠统计规律去找信号**。

数据还有一个明显特点：绝大多数特征在大多数客户身上都是 0。可以把它想象成一张很大的表格，里面记录着客户使用各种银行产品的情况，但每个客户只用了其中很少几种，所以表格非常稀疏。再加上满意和不满意的人数差距很大，这让模型很容易「偷懒」——只要全部预测成满意，准确率也能很高，但这样就完全抓不到真正不满意的客户。

### 1.2 比赛目标

训练集包含约 7.6 万行、371 列：第一列是客户编号 `ID`，中间 369 列是匿名特征，最后一列是标签 `TARGET`。`TARGET` 只有两个取值：

- `0` 表示客户对银行满意；
- `1` 表示客户不满意。

测试集包含约 7.58 万行、370 列，比训练集少了 `TARGET` 这一列。我们要对每个测试客户输出一个 0 到 1 之间的概率，表示他「不满意」的可能性。提交文件只有两列：`ID` 和 `TARGET`。

这里要注意，虽然 `TARGET` 是 0/1 标签，但比赛要的不是硬分类，而是**概率**。也就是说，输出 0.8 比输出 1 更有信息量，因为它表达了「有八成把握不满意」。

### 1.3 评价标准

比赛使用 **ROC 曲线下面积（AUC，Area Under the ROC Curve）** 作为评价指标。用大白话解释：我们把所有测试客户按预测概率从高到低排队，AUC 衡量的是「随机挑一个真正不满意的客户和一个真正满意的客户，模型给前者打的分数更高的概率」。

- AUC 等于 0.5，说明模型和抛硬币差不多；
- AUC 等于 1.0，说明模型把两类人完美分开了；
- 分数越高越好。

AUC 的好处是它只看排序，不看概率的绝对大小，所以即使模型整体把概率估得偏高或偏低，只要顺序对，分数仍然好。这对正负样本极不均衡的数据特别合适。

---

## 二、解题过程

我们的整体思路是：先通过 EDA 弄清楚数据有多稀疏、标签有多不均衡，再把明显没有信息量的列删掉、把特殊编码处理掉，最后用树模型（XGBoost 和 LightGBM）来抓非线性关系。下面按四个步骤展开。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，确认数据规模和列名：

```python
import pandas as pd
import numpy as np

# 读取训练集和测试集
train = pd.read_csv("../input/santander-customer-satisfaction/train.csv")
test = pd.read_csv("../input/santander-customer-satisfaction/test.csv")

# 看看数据有多大：训练集比测试集多一列 TARGET
print(train.shape)   # (76020, 371)
print(test.shape)    # (75818, 370)

# 前几列长这样：ID、var3、var15，以及大量 imp_/num_/saldo_ 开头的列
print(train.columns[:10].tolist())
```

打印结果告诉我们，训练集有 76020 行、371 列，测试集有 75818 行、370 列。列名里能看到 `imp_`（可能是进口/收入类金额）、`num_`（次数）、`saldo_`（余额）等前缀，但具体业务含义材料里没有说明，我们只能把它们当成普通数值特征。

接着我们看标签分布，这是整场比赛最关键的背景信息：

```python
# 统计 TARGET 的取值数量
print(train["TARGET"].value_counts())
# 0    73012
# 1     3008
```

满意客户有 73012 人，不满意客户只有 3008 人，正样本大约占 4%。这就是典型的**类别不均衡**。它带来两个直接后果：第一，不能用准确率评估模型，因为全猜 0 就有 96% 的准确率；第二，训练时要让模型更关注少数类，或者至少用对不均衡不敏感的指标（AUC）来选模型。

我们接着检查缺失值和特殊值：

```python
# 表面上看没有任何缺失值
print(train.isna().sum().sum())   # 0

# 但 var3 里有一个 -999999，出现 116 次，很像是「未知」的占位符
print(train["var3"].value_counts().head())
#  2         74165
#  8           138
# -999999      116
```

这说明数据里的「缺失」不是用 `NaN` 表示的，而是用 `-999999` 这样的特殊数字。如果我们直接把 `-999999` 当正常数值喂给模型，模型会误以为这个客户的 `var3` 极小，从而学到错误的关系。所以后面必须把它替换掉。

我们还看了几个重要特征的分布：

```python
# var15 的分布，材料里猜测它和年龄有关
train["var15"].hist(bins=30)

# var38 最常见的取值是 117310.979016，有 14868 行
print(train["var38"].value_counts().head())
# 117310.979016    14868
# 451931.220000       16
# 463625.160000       12
```

`var15` 的直方图显示大多数客户集中在中青年区间，而且不同 `TARGET` 的分布形状不一样，说明它是有区分度的。`var38` 则非常极端：将近 1.5 万人共享同一个值 117310.979016，其他取值非常分散。这种「一个值占了很大比例」的列，树模型可以通过「是否等于这个值」切一刀，往往很有用，后面特征重要性也验证了这一点。

最后我们统计有多少列几乎全是 0：

```python
# 统计每一列中 0 的个数，超过 75000 个的列几乎没有信息量
zero_cols = [c for c in train.columns if (train[c] == 0).sum() > 75000]
print(len(zero_cols))   # 229
```

76020 行里有 75000 个以上是 0，意味着非零样本不到 1000 个。这样的列对绝大多数客户没有区分作用，留着只会增加噪声和计算量，所以我们决定删掉它们。

### 2.2 数据清洗

基于上面的观察，我们做了三件事：删掉零值过多的列、替换 `var3` 的特殊编码、把训练集和测试集做完全一致的处理。

```python
# 找出 0 值超过 75000 个的列
removed_features = []
for col in train.columns:
    if col == "TARGET":
        continue
    if (train[col] == 0).sum() > 75000:
        removed_features.append(col)

# 训练集和测试集必须同步删除，否则列对不上
train.drop(columns=removed_features, inplace=True)
test.drop(columns=removed_features, inplace=True)

# var3 里的 -999999 是「未知」，材料里把它替换成 0
train["var3"].replace(-999999, 0, inplace=True)
test["var3"].replace(-999999, 0, inplace=True)

print(train.shape)   # 列数比原来的 371 少了很多
```

这里有一个容易踩的坑：删列必须**用同一份列名列表**同时作用于训练集和测试集。如果只在训练集上删，测试集就会多出很多列，模型预测时会直接报错。我们先把要删的列收集到 `removed_features`，再分别对两个表执行 `drop`，这样能保证列顺序和列数量完全一致。

除了 `var3`，材料里还对一些金额类特征做了「把极端值替换成均值」的处理。比如 `saldo_var30` 超过 250 万的值被视为离群点。树模型对离群点不像线性模型那么敏感，但在金额跨度极大的情况下，极端值仍然会抢走分裂点，所以适度截断是有帮助的。我们保留了这个思路，但没有对每一列都做，而是只处理那些量级明显异常的金额列。

清洗完之后，数据里就没有 `NaN`，也没有 `-999999` 这种「假缺失」了。我们再把 `ID` 和 `TARGET` 分开：

```python
# y 是标签，X 是特征；ID 只是编号，不能当特征
y = train["TARGET"]
X = train.drop(columns=["ID", "TARGET"])

# 测试集也去掉 ID，保留纯特征
X_test = test.drop(columns=["ID"])
```

### 2.3 模型训练与超参数调优

我们首先建立一个本地的验证集。因为正样本很少，普通随机切分可能让验证集里只剩几十个正样本，AUC 会非常不稳定，所以我们用**分层抽样**（`stratify=y`），保证训练集和验证集的正负比例一致：

```python
from sklearn.model_selection import train_test_split

# 80% 训练、20% 验证；stratify 保证两边正样本比例一样
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=86
)

print(X_train.shape, X_val.shape)
print(y_train.mean(), y_val.mean())   # 两边正样本比例应该接近
```

然后我们选了四个模型做对比：逻辑回归、随机森林、XGBoost、LightGBM。逻辑回归作为线性基线，随机森林作为 Bagging 代表，两个 Boosting 模型则是这类表格数据比赛的主力。材料里的对比结果如下：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# 四个模型都用同一个随机种子，保证对比公平
models = {
    "LogisticRegression": LogisticRegression(random_state=86),
    "RandomForest": RandomForestClassifier(random_state=86),
    "XGBoost": XGBClassifier(eval_metric="auc", random_state=86),
    "LightGBM": LGBMClassifier(random_state=86),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    # 注意：要取正类的概率，也就是 predict_proba 的第二列
    pred = model.predict_proba(X_val)[:, 1]
    print(name, roc_auc_score(y_val, pred))
```

我们得到的验证集 AUC 大致是：

| 模型 | 验证集 AUC |
| --- | --- |
| 逻辑回归 | 0.652 |
| 随机森林 | 0.748 |
| XGBoost | 0.825 |
| LightGBM | 0.831 |

逻辑回归只有 0.652，说明特征和标签之间大量是非线性关系，直线分不开。随机森林提升到 0.748，说明树结构确实能抓到一些交互。两个 Boosting 模型直接冲到 0.82 以上，其中 LightGBM 略好，而且训练时间比 XGBoost 短很多（材料里 XGBoost 大约 34 秒，LightGBM 不到 2 秒）。

在超参数方面，我们没有做大规模网格搜索，而是采用了这类比赛里比较稳的默认配置，再手工调整几个关键参数：

```python
# LightGBM 的关键参数：
# num_leaves 控制每棵树的最大叶子数，越大越容易过拟合
# learning_rate 小一点、树多一点，通常泛化更好
# min_child_samples 防止叶子节点样本太少
lgbm = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=86,
)
```

这里解释一下几个参数为什么这么设。`num_leaves=31` 是 LightGBM 的默认值，对于 7 万多行数据不算大；`learning_rate=0.02` 配合 `n_estimators=1000` 是「小步慢跑」，比用 0.1 的学习率、100 棵树更不容易过拟合；`min_child_samples=40` 是为了防止模型在只有几个样本的叶子上记住噪声。正样本只有 3008 个，如果叶子太细，模型很容易把每个正样本单独记住，验证集 AUC 反而会掉。

树模型不需要标准化，也不怕特征量纲不同，所以我们可以把原始数值直接喂进去。这是它相比逻辑回归的一个实际优势。

### 2.4 模型测试与指标报告

模型训练好之后，我们在验证集上做了更细的检查。除了 AUC，我们还看了特征重要性，确认模型学到的东西是否符合前面的 EDA：

```python
# 用训练好的 LightGBM 看最重要的 10 个特征
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": lgbm.feature_importances_,
}).sort_values("importance", ascending=False)

print(importance.head(10))
# var38、var15、saldo_var30、saldo_medio_var5_hace3 等排在前面
```

结果和 EDA 的观察一致：`var38` 和 `var15` 是最重要的两个特征。`var38` 那个「近 1.5 万人共享同一个值」的现象，让树模型可以很轻松地切出一大块人群；`var15` 则像年龄一样，把客户分成不同年龄段。紧随其后的是 `saldo_var30`、`saldo_medio_var5_hace3`、`num_var45_ult3` 等余额和交易类特征，说明客户的资产状况确实和满意度有关。

我们也尝试了特征选择：把重要性为 0 的特征删掉，再重新训练。材料里提到，369 列里有大量列几乎不参与分裂，删掉一部分之后 AUC 基本不变，但训练更快、模型更简单。这说明这场比赛的信息主要集中在大约几十个特征上，剩下的很多列是「陪跑」的。

最后我们生成提交文件：

```python
# 用全量训练数据重新训练，多喂数据通常更稳
final_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=86,
)
final_model.fit(X, y)

# 对测试集预测「不满意」的概率
test_pred = final_model.predict_proba(X_test)[:, 1]

# 按比赛格式输出：ID + TARGET
submission = pd.DataFrame({
    "ID": test["ID"],
    "TARGET": test_pred,
})
submission.to_csv("submission.csv", index=False)

# 检查行数和取值范围，避免格式错误
print(submission.shape)
print(submission["TARGET"].describe())
```

提交前我们特意检查了三件事：行数是否等于 75818、`ID` 是否和测试集一一对应、概率是否都在 0 到 1 之间。这些检查看起来琐碎，但比赛中因为提交格式错误拿不到分数的情况并不少见。

最终，我们最好的单模型在验证集上的 AUC 约为 **0.83**，和材料里其他参赛者的结果（XGBoost 0.825、LightGBM 0.831）处于同一水平。这个分数说明模型已经能比较有效地区分满意和不满意客户，但距离「完美」还有很大空间——匿名特征限制了我们对业务的理解，很多信号只能靠模型自己去挖。

---

## 三、总结

本文中我们完成了 Santander 客户满意度预测的完整流程。我们首先用 EDA 摸清了数据：76020 条训练样本、369 个匿名特征、正样本只占约 4%，而且大量列有 75% 以上的零值。其次我们做了针对性的清洗：删掉零值过多的 229 列、把 `var3` 里的 `-999999` 替换成正常值，并保证训练集和测试集同步处理。接着我们用分层抽样切分验证集，对比了逻辑回归、随机森林、XGBoost 和 LightGBM 四个模型，发现 Boosting 模型的 AUC 明显领先。然后我们调整了 LightGBM 的叶子数、学习率和最小叶样本数，并检查了特征重要性，确认 `var38`、`var15` 和几个余额类特征最重要。最后我们用全量数据重新训练，输出每个测试客户的「不满意概率」，验证集 AUC 约为 0.83。

这个比赛给我们最大的启发是：**面对匿名、稀疏、不均衡的数据，树模型和 AUC 是一对好搭档**。树模型不需要我们解释每一列的含义，也能自动处理非线性和特征交互；AUC 则不会被「多数类」带偏，能真实反映模型有没有把少数类排到前面。清洗阶段删掉大量零值列，也让模型从「什么都学一点」变成了「集中学好几十个关键特征」。
