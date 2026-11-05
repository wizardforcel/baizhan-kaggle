# Santander 客户满意度预测（Santander Customer Satisfaction）

## 一、比赛简介

### 1.1 比赛背景

银行最怕的不是客户已经离开，而是客户已经不满意、却还没有公开抱怨。Santander 希望通过历史业务数据尽早找出“不开心的客户”，这样客服团队就可以在客户流失之前主动联系。这个比赛把真实客户信息做了匿名化处理，参赛者只能看到 `var3`、`var15`、`saldo_var5`、`imp_op_var39_comer_ult1` 这类没有业务含义的字段，不能直接知道某一列代表余额、消费还是产品状态。

数据是典型的表格二分类。训练集包含 76020 行、371 列，测试集包含 75818 行、370 列；训练集比测试集多出来的那一列就是目标 `TARGET`。特征全部是整数或浮点数，没有字符串列。很多特征是 0/1 指示变量，也有金额、次数、余额等连续变量。匿名特征多、变量之间可能重复、目标类别不平衡，是这个比赛的主要难点。

### 1.2 比赛目标

我们的输入是客户的匿名特征和一个不参与建模的 `ID`。训练集里有 `TARGET`：0 表示满意，1 表示不满意。测试集没有 `TARGET`。我们要为测试集每位客户输出他“不满意”的概率，提交文件包含 `ID` 和 `TARGET` 两列。这里提交的是概率而不是硬分类标签，模型应该尽量把真正不满意的客户排在前面。

### 1.3 评价标准

比赛用 **ROC AUC（ROC 曲线下面积）**评价。AUC 可以理解为：随机抽一个不满意客户和一个满意客户，模型把不满意客户预测得更高的概率。随机排序的 AUC 约为 0.5，完美排序为 1，分数越高越好。它不要求我们先选一个 0.5 的分类阈值，因此比准确率更适合类别不平衡场景：即使不满意客户很少，只要概率排序正确，AUC 仍然会奖励模型。

---

## 二、解题过程

我们的主线是：先用 EDA 找出类别不平衡、全零特征和异常编码，再删除没有信息的列、修正明显的伪缺失值；随后用带 AUC 目标的梯度提升树训练，并通过分层验证、过采样或欠采样比较不同方案，最后输出测试集概率。

### 2.1 数据探索（EDA）

我们首先读入训练集和测试集，确认行列数、数据类型和目标分布。`ID` 只是客户编号，不能让模型把它当成有业务意义的变量。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取训练集和测试集；训练集多一列 TARGET
train_df = pd.read_csv("../input/santander-customer-satisfaction/train.csv")
test_df = pd.read_csv("../input/santander-customer-satisfaction/test.csv")

print("Train Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
print(train_df.dtypes.value_counts())

# 目标分布：先看是否严重偏斜，再决定是否重采样
print(train_df["TARGET"].value_counts())
sns.countplot(x="TARGET", data=train_df)
plt.title("TARGET distribution")
plt.show()
```

材料显示训练集是 `(76020, 371)`，测试集是 `(75818, 370)`。特征类型是 259 个整数列和 111 个浮点列，没有字符串，所以我们不需要文本编码或类别嵌入。目标计数图显示类别明显不平衡：0 类远多于 1 类。如果只看准确率，一个永远预测 0 的模型也可能看起来很高，但它完全找不出不满意客户，所以我们后面坚持看 AUC、召回率和混淆矩阵。

我们接着统计每个特征中 0 的数量。很多匿名变量本质上是“是否拥有某项业务”的指示，0 占比过高时几乎没有区分度。

```python
# 统计每一列等于 0 的行数
zero_counts = (train_df.drop(columns="TARGET") == 0).sum()

# 材料中的基线把超过 75000 个 0 的列列为待删除特征
removed_features = zero_counts[zero_counts > 75000].index.tolist()
print("待删除特征数量:", len(removed_features))
print(removed_features[:10])

# 看每列的不同取值数量，发现很多列只有 1 或 2 个值
nunique = train_df.drop(columns="TARGET").nunique().sort_values()
print(nunique.head(20))
```

EDA 的一个明确结论是：有 229 个特征的 0 数量超过 75000。它们大部分时间都是常数，既增加内存又增加树的分裂候选，删除它们是一个有依据的简化。我们还看到许多列只有一个或两个不同值，这些列也需要结合验证分数决定是否保留，不能因为匿名就盲目删除所有二值列。

### 2.2 数据清洗

我们其次删除 `ID` 和高度稀疏的特征。删除列时训练集和测试集必须同步，否则模型会遇到列不一致。材料还指出数据没有普通的空值，但 `var3` 的最小值是 `-999999`，这不是正常业务数值，而是缺失值的特殊编码。

```python
# ID 只用于提交匹配，不作为学习特征
train_id = train_df["ID"].copy()
test_id = test_df["ID"].copy()

# 去掉 ID 和目标列，得到纯特征表
X = train_df.drop(columns=["ID", "TARGET"]).copy()
X_test = test_df.drop(columns=["ID"]).copy()
y = train_df["TARGET"].copy()

# 训练、测试使用同一份待删除列清单，避免列错位
X = X.drop(columns=removed_features)
X_test = X_test.drop(columns=removed_features)

# var3 的 -999999 表示异常缺失，把它还原成 0
# 0 是该变量正常分布中更合理的下界
X["var3"] = X["var3"].replace(-999999, 0)
X_test["var3"] = X_test["var3"].replace(-999999, 0)
```

`var3` 的处理不能照搬普通的“删除异常行”：如果我们删除这些客户，就会损失训练样本；把它替换成 0 能保留样本，同时让树模型看到一个统一的缺失状态。对于其它异常值，材料中列出了按列设置阈值的做法，例如金额列超过某个上限、次数列超过某个上限时用该列均值替代。我们只在明确知道阈值来自材料的列上这样做，避免为了让分布好看而随意截断真实的高消费客户。

下面把“按列替换极端值”的逻辑写成可复用函数。函数只修改明确列出的规则，且不碰目标列。

```python
def replace_extreme_values(df, rules):
    """按照材料中给出的列名和阈值，用训练集列均值修正极端值"""
    df = df.copy()
    for rule in rules:
        col = rule["col"]
        value = rule["value"]
        if col not in df.columns:
            continue
        # 用当前训练表的均值作为保守替代，避免新增极端点
        mean_value = df[col].mean()
        mask = df[col] > value
        df.loc[mask, col] = mean_value
    return df

# 真实项目里 rules 来自 EDA 后的阈值清单；
# 测试集也要使用同一套列名和阈值
rules = [
    {"col": "imp_ent_var16_ult1", "value": 7000},
    {"col": "imp_op_var39_comer_ult1", "value": 7000},
    {"col": "num_var5", "value": 8},
    {"col": "saldo_var5", "value": 500000},
]
X = replace_extreme_values(X, rules)
X_test = replace_extreme_values(X_test, rules)
```

重采样也属于数据清洗的一部分。由于正类很少，我们准备三种训练数据：原始数据、随机欠采样（减少多数类 0）、随机过采样（复制少数类 1）。注意重采样只能发生在训练折里，不能先对全数据复制再切分，否则同一条样本可能同时出现在训练和验证中，AUC 会虚高。

### 2.3 模型训练与超参数调优

我们接着先建立一个逻辑回归或树模型基线，再使用 LightGBM、XGBoost 比较。梯度提升树很适合这类匿名表格：它能自动学习非线性关系和特征交互，不要求我们手动把所有变量缩放到同一范围。训练时把评价指标设为 `auc`，并让验证集早停。

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# 分层切分，保证验证集也保留少数的 TARGET=1
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# LightGBM 直接以二分类 AUC 为目标，不把概率硬切成 0/1
model = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    random_state=42,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

valid_proba = model.predict_proba(X_valid)[:, 1]
print("Validation ROC_AUC:", roc_auc_score(y_valid, valid_proba))
```

材料中的一个 LightGBM/逻辑回归建模 notebook 在本地验证集打印了 `ROC_AUC: 0.7897`，这给我们一个可复现的量级参考。这个数字不是最终排行榜分数，也不是所有方案都能得到的保证；它只说明在正确处理目标不平衡、使用概率输出后，模型明显优于随机排序。

为了调参，我们围绕 `num_leaves`、`learning_rate`、`n_estimators`、`max_depth`、行采样和列采样做网格或随机搜索，所有试验都用同一份分层切分和 AUC 比较。调参时我们不追求训练集 AUC 越高越好，因为树越多、叶子越复杂，训练分数很容易接近 1，却可能在验证集下降。

```python
from sklearn.model_selection import GridSearchCV

# 搜索空间保持小一些，先找方向，再做更细搜索
param_grid = {
    "num_leaves": [15, 31],
    "learning_rate": [0.01, 0.05],
    "max_depth": [-1, 5],
}

search = GridSearchCV(
    lgb.LGBMClassifier(objective="binary", metric="auc", random_state=42),
    param_grid=param_grid,
    scoring="roc_auc",       # 类别不平衡时用 AUC，而不是 accuracy
    cv=3,
    n_jobs=-1,
)
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print("Best CV AUC:", search.best_score_)
```

对于重采样，我们分别训练原始、欠采样和过采样版本，然后只保留验证 AUC 更高的方案。SMOTE 也可以生成少数类的合成样本，但匿名特征中有大量二值变量，合成以后可能产生不自然的小数，所以材料同时讨论了随机过采样、随机欠采样和 Tomek Links；我们以验证结果为准，不预设哪一种一定最好。

### 2.4 模型测试与指标报告

最后我们用选择出的模型在全部清洗后的训练集上重新拟合，再预测测试集概率。因为 AUC 只看排序，所以不能对概率做四舍五入，也不能只输出 0 和 1。

```python
# 选出搜索得到的最佳模型，并用全部训练数据重新训练
best_model = search.best_estimator_
best_model.fit(X, y)

# 第 1 列是 TARGET=0 的概率，第 2 列才是“不满意”的概率
test_proba = best_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "ID": test_id,
    "TARGET": test_proba,
})
submission.to_csv("submission.csv", index=False)

# 提交前做三项格式检查
print(submission.head())
print("行数是否一致：", len(submission) == len(test_df))
print("是否有空值：", submission.isna().any().any())
print("概率范围：", submission["TARGET"].min(), submission["TARGET"].max())
```

指标报告包含三部分。第一是本地分层验证的 ROC AUC，材料里明确记录过 0.7897；第二是不同重采样方法的验证 AUC，用来选择最终训练策略；第三是提交文件的概率范围、行数和空值检查。我们不把训练集 AUC 当成绩，也不把分类准确率当比赛指标。若需要观察某个阈值下的业务效果，可以额外打印 precision、recall 和混淆矩阵，但排行榜最终仍以概率排序的 AUC 为准。

---

## 三、总结

本文中我们完成了 Santander 客户满意度预测的完整流程。我们首先通过 EDA 确认训练集有 76020 行、测试集有 75818 行，特征是 259 个整数列和 111 个浮点列，目标类别明显不平衡。其次我们删除 `ID` 和 0 数量超过 75000 的无信息列，把 `var3` 中的 `-999999` 改成 0，并根据材料里的阈值规则处理少量极端值。接着我们用分层切分、LightGBM 和 XGBoost 方向的梯度提升树训练概率模型，用 AUC 做早停和网格搜索评分，同时比较原始数据、欠采样和过采样。最后我们在验证集上用 ROC AUC 报告结果，材料中的一个方案达到 0.7897 的本地验证 AUC，再用全量数据训练并生成 `ID,TARGET` 提交文件。

这个比赛给我们的启发是：匿名数据并不等于没有结构。列名虽然看不懂，但缺失编码、零值比例、分布异常和目标不平衡仍然可以通过统计分析发现。另一个重要教训是：业务上“找出不满意客户”不能只看准确率，因为少数类会被多数类淹没；使用概率和 AUC，才能真正评价模型有没有把高风险客户排在前面。