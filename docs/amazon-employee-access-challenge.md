# Amazon 员工访问权限预测（Employee Access Challenge）

## 一、比赛简介

### 1.1 比赛背景

在任何一家公司里，新员工入职后往往第一件事就是申请电脑权限：要打开哪些系统、能改哪些数据、能看哪些报表。这些权限一般由主管手动审批——员工发现「登不进某个系统」，再去找主管开通，一来一回既耽误员工干活，也占用主管的时间。员工换岗时，同样的流程还得再走一遍。

Amazon 在 Kaggle 上办了这场比赛：给出一批历史审批记录，每条记录描述「某个岗位的员工申请访问某个资源，最后批了还是没批」，希望参赛者训练出一个模型，在新员工入职或换岗时**自动判断这条权限该不该给**。材料里把这类模型叫作 auto-access model（自动权限模型），目标是把人工审批的介入降到最低。

这个比赛有两个明显的难点。

第一，**所有特征都是类别型的整数编号**。`RESOURCE` 是资源编号，`MGR_ID` 是主管编号，`ROLE_DEPTNAME` 是部门编号……这些数字的大小没有数学含义，编号 30000 并不比编号 20000「大」。如果把它们当成普通数值喂给模型，模型会凭空学出「30000 比 20000 更重要」这种不存在的规律。

第二，**有几列的可取值特别多**，术语叫「高基数」（high cardinality）。材料里统计出 `RESOURCE` 有 7518 个不同取值、`MGR_ID` 有 4243 个、`ROLE_FAMILY_DESC` 有 2358 个。如果直接做 one-hot 编码（每个取值拉出一列 0/1），特征维度会瞬间冲到几万维，而且大部分格子都是 0，模型很难从这种极稀疏的数据里学到东西。

### 1.2 比赛目标

给定一条审批记录，预测它被批准的概率。训练集里的列如下：

| 列名 | 含义 |
| --- | --- |
| `ACTION` | 标签，1 表示资源被批准，0 表示没有批准 |
| `RESOURCE` | 每个资源一个 ID |
| `MGR_ID` | 该员工主管的 ID，一个员工同时只有一个主管 |
| `ROLE_ROLLUP_1` | 公司角色分组类别 1（例如「美国工程」） |
| `ROLE_ROLLUP_2` | 公司角色分组类别 2（例如「美国零售」） |
| `ROLE_DEPTNAME` | 角色所属部门（例如「零售」） |
| `ROLE_TITLE` | 职位的业务名称（例如「高级工程零售经理」） |
| `ROLE_FAMILY_DESC` | 职位族的详细描述 |
| `ROLE_FAMILY` | 职位族（例如「零售经理」） |
| `ROLE_CODE` | 职位代码，每个职位唯一 |

训练集是 32769 行 × 10 列（9 个特征加标签），测试集是 58921 行（多一个 `id` 列，没有 `ACTION`）。我们要给每条测试记录填上 `ACTION` 的概率。

### 1.3 评价标准

比赛用 **AUC（Area Under the ROC Curve，ROC 曲线下面积）** 评分。

用大白话解释：我们从测试集里随便挑一条「批准」的记录和一条「拒绝」的记录，把两条都交给模型打分，如果模型给「批准」那条的分数更高，就算它赢一次。AUC 就是「赢的次数占所有配对的比例」。所以：

- 0.5 表示和抛硬币一样，什么都没学到；
- 1.0 表示每对都能排对，完美；
- 分数越高越好。

AUC 只看排序、不看概率的绝对值，所以它特别适合类别很不均衡的数据——这一点在下面会很关键。

---

## 二、解题过程

整场比赛我们的主线可以概括成一句话：**把九列整数编号原封不动交给一个「认识类别」的梯度提升树模型（CatBoost），让它自己去学每一列的统计规律。** 我们先用 EDA 摸清数据，再处理掉一列重复信息，然后训练 CatBoost 并和几种常见的编码方案做对比，最后提交并分析指标。

### 2.1 数据探索（EDA）

我们首先把数据读进来。材料里提到一个坑：Kaggle 上这份比赛的 `test.csv` 当时出了故障，行数接近 100 万而不是应有的 5.6 万。好在 CatBoost 库自带这份数据集，所以我们直接从 `catboost.datasets` 里取：

```python
# numpy 做数值计算，pandas 读写表格
import numpy as np
import pandas as pd

# 直接使用 CatBoost 内置的 amazon 数据集，绕开 Kaggle 上损坏的 test.csv
from catboost.datasets import amazon
train, test = amazon()

# 打印形状，先确认行列数是否符合预期
print("Train shape: {}, Test shape: {}".format(train.shape, test.shape))
# 输出：Train shape: (32769, 10), Test shape: (58921, 10)

# 看前几行，确认每一列都是整数编号
train.head(5)
```

打印出来的样子是这样的（只截取一部分）：`ACTION=1, RESOURCE=39353, MGR_ID=85475, ROLE_ROLLUP_1=117961, ...`。可以看到确实全是五位、六位的整数编号，没有一列是「年龄」「金额」这种连续数值。

接着我们数一数每一列有多少个不同取值，同时看看标签的分布：

```python
# 每一列的唯一值个数：这一步决定了后面该用哪种编码方式
train.apply(lambda x: len(x.unique()))
# ACTION                 2
# RESOURCE            7518
# MGR_ID              4243
# ROLE_ROLLUP_1        128
# ROLE_ROLLUP_2        177
# ROLE_DEPTNAME        449
# ROLE_TITLE           343
# ROLE_FAMILY_DESC    2358
# ROLE_FAMILY           67
# ROLE_CODE            343

# 标签的均值就是「批准」的比例
print(train['ACTION'].mean())
# 输出：0.942139
```

这两个结果都非常有信息量。

第一个：`RESOURCE`、`MGR_ID`、`ROLE_FAMILY_DESC` 是典型的高基数列，而 `ROLE_ROLLUP_1`（128 个）、`ROLE_FAMILY`（67 个）取值很少。同一种编码方式不可能同时适合这两类列，这是后面要反复权衡的地方。

第二个：**94.2% 的申请都会被批准**，也就是说正负样本比例大约是 16 : 1。这意味着一个「闭着眼睛全批」的模型准确率就有 94%，看起来很漂亮，但其实什么都没学到。这也解释了比赛为什么用 AUC 而不是准确率——AUC 不会被这种「多数类猜对」的把戏骗到。

### 2.2 数据清洗

这份数据本身很干净：材料里检查过，训练集和测试集都**没有缺失值**，也**没有重复行**。所以清洗工作只剩下两件事。

第一件事是找重复信息。我们发现 `ROLE_TITLE` 和 `ROLE_CODE` 的唯一值个数都正好是 343，这有点可疑——两个不相干的列不太可能刚好一样多。于是我们做一次交叉计数验证：

```python
# 如果「两列一起分组」的组数 == 「单列分组」的组数，
# 说明两列是一一对应的，其中一列是冗余的
col1, col2 = 'ROLE_CODE', 'ROLE_TITLE'
pair = len(train.groupby([col1, col2]).size())
single = len(train.groupby([col1]).size())
print(col1, col2, pair, single)
# 输出：ROLE_CODE ROLE_TITLE 343 343
```

结果 343 == 343，证明两列是严格的一对一关系：每个 `ROLE_CODE` 只对应一个 `ROLE_TITLE`，反过来也一样。既然信息完全重复，留一列就够了，我们删掉 `ROLE_CODE`。

有另一份材料选择删 `ROLE_TITLE` 保留 `ROLE_CODE`，效果上没有区别——反正剩下那一列携带的信息是一样的。

第二件事是把「整数编号」显式转换成「字符串类别」。这一步是为了统一类型：训练集和测试集分开读进来时，只要有一边因为某个原因被解析成了别的类型（比如出现缺失值就变成浮点），CatBoost 就会报「特征类型不一致」。统一转成字符串之后，我们再用 `cat_features` 声明这几列是类别，就不会有歧义。

```python
# 删掉和 ROLE_TITLE 完全重复的 ROLE_CODE
train.drop('ROLE_CODE', axis=1, inplace=True)
test.drop('ROLE_CODE', axis=1, inplace=True)

# 拆分特征和标签
y = train['ACTION']
X = train.drop('ACTION', axis=1)

# 转成字符串，明确告诉模型「这是类别，不是数值大小」
X = X.astype(str)
test_X = test.drop('id', axis=1).astype(str)

# 80% 训练、20% 验证
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)
```

### 2.3 模型训练与超参数调优

我们采用的模型是 **CatBoost**（Category Boosting）。它是 Yandex 开发的梯度提升树库，最大的特点就是**原生支持类别特征**：只要告诉它哪几列是类别，它内部会自动做一套叫 target statistics（目标统计）的编码，不需要我们手动 one-hot，也不需要在几万维的稀疏矩阵上折腾。

材料里对这个模型的描述是：数据会先被量化（quantize）成若干个 bin，算法自己决定 bin 的边界；然后一棵一棵地建「对称树」（symmetric tree），后一棵树修正前一棵树的错误。这些预处理都在库内部完成，我们只需要调用 `fit`。

训练代码非常短：

```python
from catboost import CatBoostClassifier

# 一共 8 列特征，全部是类别型
cat_features = [*range(8)]

model = CatBoostClassifier(
    custom_metric=['TotalF1'],   # 只用于观察，不影响训练目标
    early_stopping_rounds=100,  # 验证集指标 100 轮不提升就停，防止白跑
    eval_metric='AUC',          # 用 AUC 挑「最好的那一轮」，和比赛指标一致
)

model.fit(
    X_train, y_train,
    cat_features=cat_features,          # 声明哪些列按类别处理
    eval_set=(X_val, y_val),            # 提供验证集，才能做早停和选模型
    use_best_model=True,                # 保存验证集上最好的那一轮，而不是最后一轮
    plot=True,
    verbose=False,
)
```

几个参数值得说明一下为什么这么写：

- `eval_metric='AUC'`：CatBoost 默认会优化 Logloss，但比赛看的是 AUC。把 `eval_metric` 设成 AUC，早停和「选最好模型」就都按比赛的指标来，不会出现「logloss 降了但 AUC 没升」的尴尬。
- `early_stopping_rounds=100`：树模型很容易越训越过拟合。给 100 轮耐心值，是因为提升类模型前期波动大，耐心太小会过早停下。
- `use_best_model=True`：训练结束时内存里留下的是最后一轮的模型，而最后一轮可能已经过拟合了，所以要显式要求它保留验证集上最好的那轮。

**结果。** 这个只写了三行参数的默认模型，在验证集上拿到 **AUC 0.9027、Logloss 0.1352**，提交到 Kaggle 后公开榜分数是 **0.90373**。材料里提到冠军的分数大约 92%，也就是说一个几乎没调参的模型就已经拿到九成以上的水平。

**调参有没有用？** 材料里用 Hyperopt（一个自动搜超参数的库）在 `depth`、`border_count`、`l2_leaf_reg`、`scale_pos_weight` 上搜了 50 组，结论很干脆：**没有提升**。作者的原话是默认参数给出的 loss 已经是最好的。这给我们一个很实用的提醒：**当模型和数据的「形态」匹配时（这里是树模型 + 原生类别特征），调参的收益往往远小于选对模型。**

**横向对比。** 为了确认 CatBoost 确实适合这份数据，材料里还比了其它方案，我们用表格汇总：

| 方案 | 验证指标（AUC 越高越好，Logloss 越低越好） |
| --- | --- |
| one-hot 编码 + 逻辑回归 | AUC 0.8636 |
| 频次编码（count encoding）+ ExtraTrees | AUC 0.8782 |
| 目标编码 + 交叉验证加噪 | AUC 0.8542 |
| CatBoost 默认参数 | **AUC 0.9027** |
| XGBoost 默认参数 | Logloss 0.1555 |
| LightGBM 默认参数 | Logloss 0.1638 |

其中最值得说的是**频次编码**，因为它简单到不可思议：把每个类别替换成「这个类别在数据里出现了多少次」。逻辑是——出现得多的资源，往往是公共资源，批准规律更稳定；只出现过一两次的资源，模型对它几乎没有把握。代码只有几行：

```python
def get_freq_encoding(dataset):
    """把每一列替换成该取值出现的次数"""
    new_dataset = pd.DataFrame()
    for c in dataset.columns:
        # groupby + size 统计每个取值出现几次
        data = dataset.groupby([c]).size().reset_index()
        # merge 回原表，注意用 left join，保证行数不变
        new_dataset[c + "_freq"] = dataset[[c]].merge(data, on=c, how="left")[0]
    return new_dataset

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

# 把类别列换成频次，得到 8 列纯数值特征
# 这里为了演示方便直接在全部训练集上算频次：它只用到特征本身、不碰标签，
# 所以这种程度的「信息共享」通常被认为可以接受
X_freq = get_freq_encoding(X)

# 用 ExtraTrees 做 5 折交叉验证
stats = cross_validate(
    ExtraTreesClassifier(n_estimators=300, n_jobs=3, random_state=5436),
    X_freq, y,
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=4141),
    return_train_score=True,
)
print(stats['test_score'].mean())
# 输出约 0.8782
```

0.8782 已经比逻辑回归好了，但离 CatBoost 的 0.9027 还差一截。差距的来源很直观：频次编码只保留了「这个类别有多常见」，丢掉了「这个类别的批准率是多少」；CatBoost 内部的目标统计恰好补上了后者。

### 2.4 模型测试与指标报告

我们首先在验证集上把指标算全，而不是只看一个 AUC：

```python
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             log_loss, classification_report)

y_pred_prob = model.predict_proba(X_val)     # 第一列是 0 的概率，第二列是 1 的概率
y_pred = (y_pred_prob[:, 1] > 0.5).astype(int)  # 按 0.5 阈值切成 0/1

print(confusion_matrix(y_val, y_pred))
# [[ 143  249]
#  [  57 6105]]

print("AUC:", roc_auc_score(y_val, y_pred_prob[:, 1]))   # 0.9026809725837756
print("Logloss:", log_loss(y_val, y_pred_prob))          # 0.13516505504697254
print(classification_report(y_val, y_pred))
```

混淆矩阵很有说服力：6162 条真正被批准的申请里，模型抓对了 6105 条（召回率 0.99）；392 条本该拒绝的申请里，只抓对 143 条（召回率 0.36）。

初看会以为「拒绝类学得很差」，但结合 94% 的正样本比例就能理解：**拒绝样本太少了，模型没有足够的例子去学它的边界**。而在 AUC 这个指标下这并不致命——AUC 只关心「批准样本的分数是否普遍高于拒绝样本」，模型不需要在 0.5 这个固定阈值上把两类分得一样准。这也提醒我们：**换指标就要换读法**，不能用准确率时代的直觉去解读 AUC 时代的混淆矩阵。

最后我们生成提交文件。注意预测时只保留概率列，行数要和样例提交严格对齐：

```python
# 对测试集预测概率
y_test_prob = model.predict_proba(test_X)

# 样例提交文件决定了列名和顺序
sub = pd.read_csv("../input/amazon-employee-access-challenge/sampleSubmission.csv")
print(sum(test['id'] == sub['Id']), test.shape)   # (58921, (58921, 9))

# 第二列才是「批准」的概率
sub['Action'] = y_test_prob[:, 1]
sub.to_csv('amazon1.csv', index=False, header=True)
```

把 `sub['Id']` 和 `test['id']` 逐行比对一遍，确认 58921 条完全对齐——这行检查看着琐碎，但提交文件错位是比赛里最常见的低级失误。

---

## 三、总结

本文中我们完成了 Amazon 员工访问权限预测的完整流程。我们首先用 EDA 摸清了数据的基本形状：32769 条训练样本、58921 条测试样本，9 列全是类别型整数编号，其中 `RESOURCE`、`MGR_ID`、`ROLE_FAMILY_DESC` 是高基数列，标签有 94.2% 是「批准」，类别严重不均衡。其次我们做了很轻的清洗：确认没有缺失值和重复行，通过交叉计数发现 `ROLE_TITLE` 与 `ROLE_CODE` 是 343 对 343 的严格一一对应，删掉其中一列，再把整数编号转成字符串并声明为类别。接着我们用 CatBoost 原生处理类别特征，只设了早停、评估指标和「用最好模型」三个参数，就在验证集上拿到 AUC 0.9027、Logloss 0.1352，公开榜 0.90373；用 Hyperopt 搜 50 组超参数没有带来提升。最后我们在验证集上完整报告了混淆矩阵、AUC、Logloss 和分类报告，并把测试集预测按样例格式写成提交文件。

这场比赛给我们最大的启发是：**特征的类型决定了模型的选型**。同样一份表格，用 one-hot 加逻辑回归只有 0.8636，换成频次编码能到 0.8782，而让 CatBoost 直接吃原始类别能到 0.9027——差距不在调参，而在「模型是否天生理解类别」。另外两个小结论也值得记住：类别极不均衡时，准确率会骗人，要看 AUC；高基数类别列不要硬做 one-hot，要么用频次这类统计编码，要么用支持类别特征的树模型。
