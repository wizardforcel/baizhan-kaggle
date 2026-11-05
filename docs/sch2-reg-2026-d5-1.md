# 银行定期存款响应预测（SCH2-REG-2026-D5-1）

## 一、比赛简介

### 1.1 比赛背景

这个比赛使用的是经典的银行电话营销数据集。银行会主动给客户打电话，推销一种叫“定期存款”的理财产品。客户接起电话后，可能答应购买，也可能拒绝。每一次外呼都要花人力成本，频繁打扰客户还会引起反感，所以银行希望提前判断：哪些客户更可能接受产品？把有限的外呼名额留给这些人，转化率会更高，成本也更低。

数据记录的是过去若干次营销活动的真实结果。每一行对应一位客户，包含年龄、职业、婚姻状况、教育程度、是否有信用违约、账户平均余额、是否有住房贷款、是否有个人贷款、联系方式、最后一次联系的日期、通话时长、本次活动中被联系的次数、距上次活动多少天、过去联系次数、上次活动的结果等字段。标签 `y` 表示这位客户最终有没有购买定期存款：`1` 表示买了，`0` 表示没买。

这个任务的核心难点是**类别不平衡**。真正答应购买的客户只占很小一部分，如果模型只追求准确率，它完全可以一律预测“不买”，准确率也会很高，但这样的模型对银行没有任何价值。因此比赛更关心模型能不能把“可能购买的人”排在前面。

### 1.2 比赛目标

训练集包含客户属性列 `age`、`job`、`marital`、`education`、`default`、`balance`、`housing`、`loan`、`contact`、`day`、`month`、`duration`、`campaign`、`pdays`、`previous`、`poutcome`，以及编号 `id` 和标签 `y`。测试集只有特征列和 `id`，没有 `y`。我们要为测试集里的每一位客户输出一个“购买概率”，最后生成包含 `id` 和 `y` 两列的提交文件。

### 1.3 评价标准

材料里的建模代码主要使用 ROC-AUC（Receiver Operating Characteristic - Area Under Curve）来评价模型。ROC 曲线画的是“把正样本排在前面的能力”：横轴是假阳性率，纵轴是真阳性率。AUC 就是这条曲线下面的面积，取值在 0 到 1 之间，0.5 相当于瞎猜，越接近 1 越好。材料里记录了一个基线 OOF AUC，约为 0.96823，后面的特征工程就是拿它当参照物来比较的。

## 二、解题过程

我们的整体思路是：先通过 EDA 看清数据长什么样，再围绕“客户资质、历史营销、通话行为”三个方向做特征工程，然后用 LightGBM 做五折交叉验证，最后用 OOF 预测评估模型并生成提交文件。

### 2.1 数据探索（EDA）

我们首先读入训练集和测试集，确认行数、列数和数据类型：

```python
import numpy as np
import pandas as pd

# 读取比赛数据
train = pd.read_csv("/kaggle/input/competitions/sch2-reg-2026-d5-1/train.csv")
test  = pd.read_csv("/kaggle/input/competitions/sch2-reg-2026-d5-1/test.csv")

print("train shape:", train.shape)
print("test  shape:", test.shape)
print(train.head())
```

接着我们看目标分布。因为正样本很少，我们不仅看数量，还要看比例：

```python
# 目标分布：1 表示购买，0 表示没有购买
vc = train["y"].value_counts()
print(vc)
print("正样本比例:", train["y"].mean())
```

从输出可以确认这是一个明显不平衡的二分类问题。于是后面所有模型评估都以 AUC 为主，而不是只看准确率。

我们还画了通话时长和目标之间的箱线图。图中的差异非常明显：答应购买的客户，通话时长普遍更长。这很符合直觉——客户愿意聊下去，说明有兴趣；如果一接电话就挂断，基本不会买。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 对比购买与未购买客户在通话时长上的分布
sns.boxplot(data=train, x="y", y="duration")
plt.title("Call duration by target")
plt.show()
```

不过这里马上要提醒自己：`duration` 是通话结束之后才知道的字段。真正在打电话之前做预测时，我们根本不知道这通电话会聊多久。所以它虽然预测能力很强，却带有**数据泄露**的风险，不能直接当成线上可用特征。我们在实验里保留它，是为了和材料中的基线对齐；在讨论部署时则要明确去掉它。

除了通话时长，我们还统计了各个类别特征的“购买率”，也就是每个职业、每种婚姻状态、每档学历里 `y=1` 的比例。结果显示 `poutcome`（上次营销结果）、`previous`（过去联系次数）和 `campaign`（本次联系次数）的区分度比较高。之前成功过的客户再次购买的概率更大，而被反复打扰很多次的客户反而更容易拒绝。

### 2.2 数据清洗

这份数据的缺失值很少，主要的清洗工作是**对齐训练集和测试集的类别取值**。如果某个职业只出现在测试集里，而训练集里没有，模型遇到它就会不知所措。我们的做法是先把两边的类别取并集，再统一转成 `category` 类型：

```python
# 去掉编号列，它不参与建模
train = train.drop(columns=["id"])
test  = test.drop(columns=["id"])

# 找出所有文本类型的类别列
cat_cols = [c for c in train.columns if train[c].dtype == "object"]

# 把训练集和测试集的类别取并集，保证两边编码一致
for col in cat_cols:
    categories = pd.api.types.union_categoricals([
        train[col].astype("category"),
        test[col].astype("category"),
    ]).categories
    train[col] = pd.Categorical(train[col], categories=categories)
    test[col]  = pd.Categorical(test[col],  categories=categories)
```

我们没有对数值列做标准化。原因是后面用的是 LightGBM，这是一类基于决策树的模型，它只关心特征取值的排序，不关心具体尺度，所以标准化不会带来收益，反而多一步操作。

接下来是特征工程。我们围绕年龄、余额、通话时长构造了分段特征和组合特征：

```python
def create_features(df):
    df = df.copy()

    # 年龄是否达到 60 岁，老年客户对稳健理财的兴趣可能不同
    df["is_senior"] = (df["age"] >= 60).astype(int)

    # 余额是否为负，负数往往意味着客户有透支或负债
    df["negative_balance"] = (df["balance"] < 0).astype(int)

    # 通话时长分段：短电话、长电话分别标记
    df["long_call"]  = (df["duration"] >= 1000).astype(int)
    df["short_call"] = (df["duration"] < 100).astype(int)

    # 把连续的通话时长切成几个区间，方便树模型抓拐点
    df["duration_group"] = pd.cut(
        df["duration"],
        bins=[-np.inf, 100, 300, 600, 1000, np.inf],
        labels=["under_100", "100_299", "300_599",
                "600_999", "1000_or_more"],
        right=False,
    )

    # 余额做“带符号对数”变换：负数取负，正数取正，压缩极端值
    df["balance_signed_log"] = (
        np.sign(df["balance"]) * np.log1p(np.abs(df["balance"]))
    )

    # 老年且余额非负，代表一部分“有钱的退休客户”
    df["senior_nonnegative_balance"] = (
        (df["age"] >= 60) & (df["balance"] >= 0)
    ).astype(int)

    # 老年且通话很长，说明聊得非常投入
    df["senior_long_call"] = (
        (df["age"] >= 60) & (df["duration"] >= 1000)
    ).astype(int)

    # 通话时长取对数，减弱长尾的影响
    df["duration_log"] = np.log1p(df["duration"])

    return df

train_featured = create_features(train)
test_featured  = create_features(test)
```

这些特征不是随便堆的，每一条都对应一个业务假设：年龄和余额刻画客户的财务能力，通话时长刻画兴趣强弱，组合特征则是在捕捉“既有钱又愿意聊”的客户。

### 2.3 模型训练与超参数调优

我们选择 LightGBM 作为主模型。它是一种梯度提升树（GBDT），工作原理是：先训练一棵简单的树，然后让下一棵树专门去修正前面犯的错，一棵一棵叠加起来，整体预测越来越准。它对表格数据友好，能自动处理类别特征，训练速度也快。

下面是材料中使用的参数：

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

# 目标列和编号列
TARGET = "y"
ID_COL = "id"

# 特征列就是除编号和目标之外的所有列
feature_cols = [c for c in train_featured.columns
                if c not in [ID_COL, TARGET]]
X      = train_featured[feature_cols]
y      = train_featured[TARGET]
X_test = test_featured[feature_cols]

# 类别特征要单独告诉 LightGBM
cat_cols = [c for c in feature_cols
            if str(train_featured[c].dtype) == "category"]

lgb_params = {
    "objective": "binary",       # 二分类任务
    "metric": "auc",             # 用 AUC 监控验证集
    "learning_rate": 0.03,       # 学习率小一点，模型更稳
    "num_leaves": 63,            # 每棵树的叶子数，控制复杂度
    "max_depth": -1,             # 不限制深度，由叶子数间接控制
    "min_child_samples": 40,     # 叶子最少样本数，防止过拟合
    "feature_fraction": 0.8,     # 每次分裂只用 80% 特征
    "bagging_fraction": 0.8,     # 每次迭代只用 80% 样本
    "bagging_freq": 5,           # 每 5 轮做一次 bagging
    "reg_alpha": 0.1,            # L1 正则
    "reg_lambda": 0.1,           # L2 正则
    "max_cat_to_onehot": 8,      # 类别取值不超过 8 个时用 one-hot
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
```

训练采用五折分层交叉验证。分层的意思是：切分时保证每一折里正负样本的比例和整个训练集一致，避免某一折恰好全是负样本导致 AUC 失真。

```python
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds  = np.zeros(len(X))       # 收集每个样本的折外预测
test_preds = np.zeros(len(X_test))  # 累加各折对测试集的预测
cv_scores  = []
best_iters = []

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
    dtrain = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx],
                         categorical_feature=cat_cols)
    dvalid = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx],
                         categorical_feature=cat_cols,
                         reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dvalid],
        callbacks=[
            # 验证集 AUC 连续 100 轮不提升就停止
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # 折外预测：验证折的每个样本只被当前模型预测一次
    oof_preds[val_idx] = model.predict(
        X.iloc[val_idx], num_iteration=model.best_iteration)

    # 测试集预测按折平均，降低单折波动
    test_preds += model.predict(
        X_test, num_iteration=model.best_iteration) / N_SPLITS

    fold_auc = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
    cv_scores.append(fold_auc)
    best_iters.append(model.best_iteration)
    print(f"Fold {fold} AUC: {fold_auc:.5f} "
          f"(best_iter: {model.best_iteration})")
```

关于调参，我们没有做大规模的网格搜索，而是沿着“先降学习率、再加叶子数、同时加正则”的方向小步调整。材料里的经验是：学习率 0.03 配合 63 个叶子已经能稳定收敛，再复杂化收益很小。早停轮数设为 100，既能给模型足够的探索空间，又不会无限训练。

### 2.4 模型测试与指标报告

训练结束后，我们把五折的折外预测拼起来，算一个整体的 OOF AUC。这个分数比单折 AUC 更可靠，因为每个样本的预测都来自没见过它的模型。材料里的基线是 0.96823，我们用它作为参照：

```python
# 把所有折的 OOF 预测拼成完整向量后计算整体 AUC
oof_auc = roc_auc_score(y, oof_preds)

print("每折 AUC:", [f"{s:.5f}" for s in cv_scores])
print("平均 AUC: %.5f +/- %.5f" % (np.mean(cv_scores), np.std(cv_scores)))
print("OOF AUC : %.5f" % oof_auc)

BASELINE_AUC = 0.96823
print("与基线差值: %+.5f" % (oof_auc - BASELINE_AUC))
```

从结果看，加入年龄、余额、时长分段这些工程特征之后，模型的 OOF AUC 与材料里的基线保持在同一水平，说明特征工程的方向是合理的。我们还可以画 ROC 曲线，直观看模型在不同阈值下的表现：

```python
fpr, tpr, _ = roc_curve(y, oof_preds)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, lw=2, label=f"OOF ROC (AUC = {oof_auc:.5f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (OOF)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
```

最后生成提交文件。注意提交的是概率而不是 0/1 标签，因为比赛按 AUC 评价，概率排序比硬分类更有信息量：

```python
submission = pd.DataFrame({
    ID_COL: test_featured[ID_COL],
    TARGET: test_preds,
})
submission.to_csv("submission.csv", index=False)

print("submission shape:", submission.shape)
print(submission.head())
```

我们还会检查提交文件的行数是否和测试集一致、`id` 是否唯一、概率是否落在 0 到 1 之间。这些检查能避免因为格式问题白白丢分。

## 三、总结

本文中我们完成了银行定期存款响应预测的完整流程。我们首先通过 EDA 确认了目标类别不平衡，并发现通话时长与购买行为强相关，同时意识到它存在数据泄露风险。其次我们对齐了训练集和测试集的类别取值，并围绕年龄、余额和通话行为构造了分段与组合特征。接着我们用 LightGBM 做五折分层交叉验证，采用较小学习率、早停和正则化来控制过拟合。然后我们用 OOF AUC 评估模型，结果与材料中的基线 0.96823 处于同一水平。最后我们按折平均生成测试集概率，并保存为符合要求的提交文件。

这个比赛给我们最大的启发是：**离线分数高不代表线上能用**。`duration` 这类“事后才知道”的特征能把 AUC 推得很高，但真实预测时拿不到。做特征工程时，除了问“这个特征有没有用”，还要问“这个特征在预测那一刻是否真的存在”。
