# 银行客户流失二分类（Binary Classification with a Bank Dataset）

## 一、比赛简介

### 1.1 比赛背景

银行每天都在做电话营销：客户经理拿着一份名单，挨个打电话，向客户推荐定期存款（term deposit）产品。问题是，大部分客户会拒绝，只有少数人会说「好」。如果能把「大概率会接受」的客户提前挑出来，银行就能把有限的人力集中在最值得打的那批人身上，省下大量无效外呼。

这个比赛要解决的就是这件事。数据来自银行的电话营销记录，每一行对应一次与客户的沟通，包含客户的年龄、职业、婚姻状况、教育程度、账户余额、是否有房贷/个人贷款、这次沟通的时长、联系月份、之前营销活动的次数和结果等等，最后一列 `y` 表示这位客户最终**是否认购了定期存款**。

这是一个典型的**二分类 + 类别不均衡**问题。认购的客户通常只占少数，如果我们只看准确率，模型只要全猜「不认购」就能拿到很高的分数，但这样的模型对银行毫无价值——它一个潜在客户都没找出来。因此比赛采用 ROC-AUC 作为评价标准，我们后面会解释为什么它比准确率更适合这里。

另一个特点是**混合类型特征多**：既有 `age`、`balance`、`duration` 这样的连续数值，也有 `job`、`marital`、`education`、`month`、`poutcome` 这样的类别变量。怎么把类别变量合理地编码进模型，是这场比赛的主要工程量之一。

### 1.2 比赛目标

输入是训练集和测试集两张表：

- 训练集包含 `id`、一组客户与营销活动特征，以及目标列 `y`（`yes`/`no` 或 1/0）；
- 测试集包含同样的特征和 `id`，但没有 `y`。

我们要对测试集里的每一次营销沟通，预测客户是否会认购，并按提交样例的格式输出 `id` 和 `y`。`y` 可以提交 0/1 硬标签，也可以提交概率值；因为评分用 ROC-AUC，提交概率通常更占优势（AUC 只看排序，概率保留了更多信息）。

### 1.3 评价标准

比赛使用 **ROC-AUC** 作为评价指标，越高越好。

用大白话说：模型会给每个客户输出一个「认购可能性」分数。把所有客户按这个分数从高到低排队，AUC 衡量的是「随机挑一个真的认购了的客户，和一个没认购的客户，模型给前者的分数更高的概率」。AUC = 0.5 等于随机乱排，AUC = 1.0 是完美排序。

AUC 的好处是**不受阈值和类别比例影响**：不管我们把「认定为会认购」的门槛定在 0.3 还是 0.7，排序质量不变。这对银行场景很实用——业务方可以按人力预算决定只打前 10% 或前 20% 的客户，而模型只需要把真正有意向的人排在前面。

材料里另外报告了 **PR-AUC（平均精确率，Average Precision）**。当正样本很少时，PR-AUC 比 ROC-AUC 更敏感，因为它直接看「被我们挑出来的人里有多少是真的」，所以我们也把它作为辅助指标一起看。

---

## 二、解题过程

我们的整体思路是：**先通过 EDA 和箱线图了解数值分布、处理极端值，再把类别变量分成「有序」和「无序」两类分别编码，接着在同一套 5 折交叉验证下比较逻辑回归、XGBoost、LightGBM，最后把两个树模型做概率平均。** 下面分四步讲。

### 2.1 数据探索（EDA）

我们首先把训练集读进来，并单独取出所有数值列，方便观察分布：

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("/kaggle/input/sch2-reg-2026-d4-2/train.csv")

# 只保留数值列，方便做统计和画箱线图
out_df = train.select_dtypes(exclude=["object"])

# 逐个看几个关键列的取值分布
for col in ["previous", "age", "balance", "duration", "campaign", "pdays"]:
    print(col, out_df[col].value_counts().head())
```

从 `value_counts()` 里我们注意到两件事。第一，`pdays`（距离上次营销活动结束的天数）大量取值为 `-1`，这其实是一个「占位符」，表示这位客户**之前从未被联系过**，而不是「负一天」。第二，`balance` 有负值（透支），`duration`（通话时长）差异极大，从几秒到几十分钟都有。

接着我们画箱线图，直观看每个数值列的离群情况：

```python
import matplotlib.pyplot as plt

n = 1
plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.3)
for col in out_df.columns:
    if n <= 9:
        plt.subplot(3, 3, n)
        plt.boxplot(out_df[col])   # 箱线图会把超出须线的点标成离群点
        plt.title(col)
        n += 1
plt.show()
```

箱线图清楚显示 `balance`、`duration`、`campaign`、`previous` 都有很长的右尾。这些极端值可能来自真实的大客户，也可能是录入异常，需要谨慎处理。

然后我们看目标分布：

```python
import seaborn as sns

# 统计 y 的两个类别各有多少
print(train["y"].value_counts())

# 画柱状图，直观感受不均衡程度
sns.countplot(x=train["y"])
plt.show()
```

结果确认了我们的判断：`no` 远多于 `yes`，这是一个不均衡问题。这也解释了为什么不能只看准确率。

最后我们检查类别列的取值个数，为后面的编码方案做准备：

```python
# 每个类别列有多少种取值
for col in ["poutcome", "month", "contact", "loan", "housing",
            "default", "education", "marital", "job"]:
    print(col, train[col].nunique(), train[col].unique()[:6])
```

统计下来：`poutcome` 有 4 类，`month` 有 12 类，`contact` 有 3 类，`education` 有 4 类且含 `unknown`，`job` 有 12 类。这些信息决定了我们后面用 one-hot 还是序数编码。

### 2.2 数据清洗

我们其次处理异常值和缺失编码。

**第一步：用 IQR 规则裁掉极端值。** IQR 是「四分位距」，即第 75 百分位减第 25 百分位。统计学上把超出 `Q1 - 1.5×IQR` 或 `Q3 + 1.5×IQR` 的点视为离群点。我们对几个数值列做了截断式过滤：

```python
int_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]

for col in int_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    low_limit = Q1 - 1.5 * IQR
    up_limit  = Q3 + 1.5 * IQR
    # 只保留落在上下界之内的行
    train = train[(train[col] >= low_limit) & (train[col] <= up_limit)]

print(train.shape)   # 行数会变少，说明确实裁掉了不少极端样本
```

这里我们要说明一个取舍：删除离群样本会损失一部分数据，但极端值对线性模型和距离类模型的影响很大，而对树模型影响较小。因为后面要同时跑逻辑回归和树模型，我们选择了统一裁掉，保证比较公平。

**第二步：处理「伪缺失」和低信息列。** 我们发现 `contact`（联系方式）和 `default`（是否有违约）这两列取值高度集中，区分度很低，于是直接删掉：

```python
# 把训练集和测试集拼在一起处理，保证编码一致
cum_df = pd.concat([train.drop("y", axis=1), test], axis=0)

# contact 和 default 信息量太低，删掉可以减少噪声维度
cum_df.drop(columns=["contact", "default"], axis=1, inplace=True)
```

**第三步：把二值类别转成 0/1。** `loan`、`housing` 只有 yes/no 两种取值，直接映射最省事：

```python
# yes → 1，no → 0
cum_df["loan"]    = cum_df["loan"].replace({"no": 0, "yes": 1})
cum_df["housing"] = cum_df["housing"].replace({"no": 0, "yes": 1})
```

**第四步：`poutcome` 用序数编码。** 上一次营销活动的结果有明显的强弱顺序：`unknown` < `other` < `failure` < `success`。与其做 one-hot 把顺序信息打散，不如直接给它们编号：

```python
# 按「结果好坏」排序赋码：unknown=0, other=1, failure=2, success=3
cum_df["poutcome"] = cum_df["poutcome"].replace(
    {"unknown": 0, "failure": 2, "success": 3, "other": 1}
)
```

**第五步：剩下的无序类别做 one-hot。** `job`、`marital`、`education`、`month` 之间没有大小关系，用哑变量（dummy）最合适：

```python
# 取出所有仍然是字符串的列
obj_df = cum_df.select_dtypes(include=["object"])

# get_dummies 会为每个取值生成一列 0/1，dtype=int 避免出现 True/False
obj_df = pd.get_dummies(obj_df, dtype=int)

# 数值列和哑变量列拼回去
int_df = cum_df.select_dtypes(exclude=["object"])
final_df = pd.concat([int_df, obj_df], axis=1)
```

拼接完成后，我们再按原来的行数把 `final_df` 切回训练部分和测试部分，并丢掉第一列 `id`（它不是特征）：

```python
# 前 train.shape[0] 行是训练集，后面是测试集
train_df = final_df.iloc[:train.shape[0], 1:]
test_df  = final_df.iloc[train.shape[0]:, 1:]

y = train["y"]
```

到这里，所有特征都变成了数值，可以直接喂给任何模型。我们保留了 `pdays` 的 `-1`，因为树模型可以自己学到「-1 代表从未联系」这个特殊含义；在另一条更精细的方案里，参赛者把 `pdays == -1` 单独抽成一个 `pdays_cont` 标志位，再把 `-1` 替换成 0，效果也不错。

### 2.3 模型训练与超参数调优

我们接着划分数据并训练模型。先留出 25% 做验证：

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(
    train_df, y, test_size=0.25, random_state=42
)

# 线性模型对尺度敏感，需要标准化；树模型则不需要
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)
test_df_scaled = scaler.transform(test_df)
```

**基线：逻辑回归。** 我们先跑一个标准实现，拿到参考分数：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

lg = LogisticRegression(max_iter=1000)
lg.fit(x_train_scaled, y_train)

# 注意这里传的是 predict 出的 0/1，而不是概率
lg_pred = lg.predict(x_test_scaled)
print("Logistic ROC-AUC:", roc_auc_score(y_test, lg_pred))
```

为了理解逻辑回归内部到底在做什么，我们还手写了一个简化版：用梯度下降最小化交叉熵。这既是对照，也是一次很好的练习：

```python
class LogiReg:
    """手写逻辑回归：sigmoid + 梯度下降"""
    def __init__(self, learning_rate=0.001, epochs=200):
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, x_train, y_train):
        # 在最左边插一列 1，把截距也当成一个权重来学
        x_train = np.insert(x_train, 0, 1, axis=1)
        w = np.ones(x_train.shape[1])
        N = x_train.shape[0]

        for _ in range(self.epochs):
            z = np.dot(x_train, w)
            y_hat = 1 / (1 + np.exp(-z))       # sigmoid，把分数压到 0~1
            error = y_train - y_hat            # 预测与真实标签的差
            grad = np.dot(error, x_train) * (-1 / N)  # 交叉熵的梯度
            w = w - self.lr * grad             # 沿梯度反方向走一小步

        self.intercept_ = w[0]
        self.coef_ = w[1:]

    def predict(self, x_test):
        raw = np.dot(x_test, self.coef_) + self.intercept_
        # 大于 0 判为 1，否则判为 0
        return np.array([1 if v > 0 else 0 for v in raw])
```

**树模型：XGBoost。** XGBoost 在表格数据上几乎从不让人失望。我们设置了 600 棵树、深度 3，并开启行/列采样：

```python
from xgboost import XGBClassifier

xc = XGBClassifier(
    n_estimators=600,      # 600 轮 boosting
    max_depth=3,           # 树很浅，靠多轮累加提升
    subsample=0.5,         # 每棵树随机用一半样本
    colsample_bytree=0.5,  # 每棵树随机用一半特征
    random_state=42
)
xc.fit(x_train_scaled, y_train)
xc_pred = xc.predict(x_test_scaled)
print("XGBoost ROC-AUC:", roc_auc_score(y_test, xc_pred))
```

**树模型：决策树。** 我们另外训练了一棵单树作为对照，用熵作为分裂准则，并通过 `min_samples_split=300` 强制每次分裂至少需要 300 个样本，防止树长得太细：

```python
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(
    criterion="entropy",      # 用信息增益选分裂点
    max_depth=25,
    min_samples_split=300,    # 一个节点至少 300 个样本才允许继续分裂
    min_samples_leaf=2,
    splitter="best",
    random_state=42
)
dtc.fit(x_train_scaled, y_train)
print("DecisionTree ROC-AUC:", roc_auc_score(y_test, dtc.predict(x_test_scaled)))
```

**更稳的方案：5 折交叉验证 + 概率平均。** 单次 `train_test_split` 的验证结果波动较大，尤其在不均衡数据上。因此我们最终采用的是另一条更规范的路线：用 `StratifiedKFold` 做 5 折，每折都保持正负样本比例一致；同时用 LightGBM 和 XGBoost 各跑一遍，最后把两者的预测概率按 0.5/0.5 平均：

```python
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# oof 用来存「每个样本在它没参与训练的那一折上」的预测
oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y), 1):
    X_tr, X_va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # LightGBM：叶子数 31、学习率 0.05，配合早停
    m_lgb = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=31,
        max_depth=8, random_state=42, n_jobs=-1, verbose=-1
    )
    m_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)], eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    oof_lgb[va_idx] = m_lgb.predict_proba(X_va)[:, 1]

    # XGBoost：用 scale_pos_weight 补偿正负样本不均衡
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    m_xgb = XGBClassifier(
        n_estimators=300, tree_method="hist",
        scale_pos_weight=spw, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    m_xgb.fit(X_tr, y_tr)
    oof_xgb[va_idx] = m_xgb.predict_proba(X_va)[:, 1]

    print(f"Fold {fold} LGB AUC: {roc_auc_score(y_va, oof_lgb[va_idx]):.4f}")

# 两个模型的 OOF 概率做简单平均
oof_blend = 0.5 * oof_lgb + 0.5 * oof_xgb

print("Blend ROC:", roc_auc_score(y, oof_blend))
print("Blend PR :", average_precision_score(y, oof_blend))
```

这里 `scale_pos_weight` 是一个很实用的参数：它等于「负样本数 / 正样本数」，会让 XGBoost 在算损失时给正样本更高的权重，相当于告诉模型「漏掉一个真正会认购的客户，比误判一个不会认购的客户更可惜」。

**关于调参。** 我们试过的方向包括树的数量（300 到 1000）、学习率（0.02 到 0.1）、叶子数（31 到 127）、以及前面提到的 `scale_pos_weight`。结论是：在这份数据上，**把类别比例通过权重喂给模型**带来的 AUC 提升，比单纯加深树要明显；而两个模型做概率平均又能再稳定提升一点，因为 LightGBM 和 XGBoost 的误差不完全相同，平均之后可以互相抵消一部分。

### 2.4 模型测试与指标报告

我们用 OOF（out-of-fold）预测来评估最终方案。OOF 的好处是每个样本的预测都来自「没见过它」的模型，等价于一次完整的交叉验证，比单次划分更可靠：

```python
from sklearn.metrics import roc_auc_score, average_precision_score

# 单个模型的表现
print("LGB  ROC:", roc_auc_score(y, oof_lgb),
      "PR:", average_precision_score(y, oof_lgb))
print("XGB  ROC:", roc_auc_score(y, oof_xgb),
      "PR:", average_precision_score(y, oof_xgb))

# 融合后的表现
print("Blend ROC:", roc_auc_score(y, oof_blend),
      "PR:", average_precision_score(y, oof_blend))
```

从结果看，两个树模型的 ROC-AUC 都明显高于逻辑回归基线，融合之后又略有提升；PR-AUC 的提升幅度通常比 ROC-AUC 更明显，因为正样本稀少时 PR 曲线对「排在前面的那批人抓得准不准」更敏感。

确认方案有效后，我们用全部训练数据重新拟合两个模型，再对测试集预测并融合：

```python
# 在完整训练集上重训
m_lgb_full = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, num_leaves=31,
    max_depth=8, random_state=42, n_jobs=-1, verbose=-1
).fit(train_df, y)

spw_full = (y == 0).sum() / (y == 1).sum()
m_xgb_full = XGBClassifier(
    n_estimators=300, tree_method="hist",
    scale_pos_weight=spw_full, random_state=42, n_jobs=-1
).fit(train_df, y)

# 分别预测概率，再按权重平均
p_lgb = m_lgb_full.predict_proba(test_df)[:, 1]
p_xgb = m_xgb_full.predict_proba(test_df)[:, 1]
p_blend = 0.5 * p_lgb + 0.5 * p_xgb

# 按 sample_submission 的列名写文件
sample = pd.read_csv("/kaggle/input/sch2-reg-2026-d4-2/sample_submission.csv")
target_col = [c for c in sample.columns if c != "id"][0]
sample[target_col] = p_blend
sample.to_csv("submission.csv", index=False)
```

我们最终提交的是**概率**而不是 0/1 标签。原因是 ROC-AUC 只关心排序，概率里保留了「这个客户比那个客户更可能认购」的相对信息；如果压成 0/1，大量客户会并列，排序信息就丢了。这也是 AUC 类比赛的一个通用技巧。

---

## 三、总结

本文中我们完成了银行客户流失（定期存款认购）二分类比赛的完整流程。我们首先用 EDA 和箱线图摸清了数值分布，发现 `pdays = -1` 是「从未联系」的占位符、`balance` 存在负值、多个变量有长尾。其次我们用 IQR 规则裁掉极端值，删掉信息量过低的 `contact` 和 `default`，把 yes/no 映射成 0/1、把 `poutcome` 按结果好坏做序数编码、其余无序类别做 one-hot。接着我们在同一套 5 折分层交叉验证下比较了逻辑回归、XGBoost、LightGBM，并用 `scale_pos_weight` 处理类别不均衡。最后我们把 LightGBM 和 XGBoost 的 OOF 概率按 0.5/0.5 融合，用 ROC-AUC 与 PR-AUC 两个指标确认效果，再用全量数据重训并提交概率值。

这个比赛让我们体会到三件事。第一，**清洗要看模型**：极端值对线性模型伤害大，对树模型影响小，统一裁掉是为了公平比较。第二，**编码要尊重语义**：`poutcome` 有明确的好坏顺序，用序数编码比 one-hot 更省维度也更有信息；`job` 没有顺序，才该用 one-hot。第三，**不均衡问题要用对工具**：`scale_pos_weight` 和 `class_weight` 让模型在训练时就重视少数类，而评估时用 AUC/PR-AUC 而不是准确率，才能看清模型真正的排序能力。
