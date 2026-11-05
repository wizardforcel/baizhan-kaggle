# 预测数据科学家薪资（Salary Prediction）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的数据来自 GlassDoor 上的职位招聘信息。GlassDoor 是一个让员工给公司打分的网站，招聘方也会在上面发布岗位。每条记录描述一个岗位：职位名称、公司所在地、公司总部、公司规模、成立年份、行业、营收、竞争对手，以及一段很长的职位描述。我们要根据这些信息，预测这个岗位的平均年薪。

薪资预测和普通的分类问题不一样。分类问题输出的是“是或否”“A 类或 B 类”，而这里输出的是一个连续的数字，所以它属于**回归问题**。回归问题常用的损失函数是均方误差，也就是把每个预测值和真实值的差平方后求平均。

比赛的真实难点不在模型本身，而在**特征**。表格里大部分列都是文字：职位名称、地点、公司描述、职位描述。如果直接把文字丢掉，只剩下评分和成立年份两个数字，模型能学到的信息非常少。怎么把文字变成模型能吃的数字，是整场比赛的核心。

### 1.2 比赛目标

训练集和测试集都是 CSV 文件。训练集里包含下面这些列：

- `id`：每条样本的编号；
- `Job Title`：职位名称，例如 Data Scientist、Data Engineer；
- `Job Description`：职位描述，一段较长的英文文本；
- `Rating`：公司在 GlassDoor 上的评分，数值，可能有缺失；
- `Company Name`：公司名称；
- `Location`：岗位所在城市和州；
- `Headquarters`：公司总部所在城市和州，缺失时用 `-1` 表示；
- `Size`：公司规模，例如 `201 to 500 employees`；
- `Founded`：公司成立年份，缺失时用 `-1` 表示；
- `Type of ownership`：公司所有制类型；
- `Industry`、`Sector`：行业和更大类别的行业；
- `Revenue`：公司营收区间；
- `Competitors`：竞争对手列表；
- `posting_date`：招聘信息发布日期；
- `avg_salary`：目标列，平均年薪，单位是千美元。

测试集没有 `avg_salary` 这一列。我们需要为测试集里的每个 `id` 预测一个 `avg_salary`，最后保存成两列的提交文件：`id` 和 `avg_salary`。

### 1.3 评价标准

比赛使用**均方误差（Mean Squared Error，简称 MSE）**作为评价指标。它的算法是：

$$
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

其中 $y_i$ 是第 $i$ 条样本的真实薪资，$\hat{y}_i$ 是模型预测的薪资，$n$ 是样本总数。平方会把大的错误放得更大，所以模型会特别怕“离谱”的预测。MSE 越小越好，没有上限，完美预测时等于 0。

实际看结果时，我们也会开平方得到 RMSE（均方根误差）。RMSE 和薪资的单位一样，都是千美元，读起来更直观。比如 RMSE 等于 20，就表示预测平均会差两万美元左右。注意这个解释只是近似，不是严格的平均误差。

---

## 二、解题过程

我们的整体思路是：先把原始表格里能直接用的数字和类别整理出来，再把职位名称和职位描述拼成一段文本，用预训练语言模型把文本压成向量，最后用 LightGBM 做回归，并用 Optuna 自动搜超参数。下面按 EDA、数据清洗、模型训练与调参、测试与报告四步展开。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，看看数据规模、列类型和缺失情况。

```python
import pandas as pd
import numpy as np

# 读取比赛提供的两张表
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 看形状：多少行、多少列
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# 看每一列的类型和非空数量，快速定位缺失
train.info()

# 看每一列有多少缺失值
print(train.isna().sum().sort_values(ascending=False))

# 看目标列的五数概括：最小值、四分位数、中位数、最大值
print(train["avg_salary"].describe())
```

读完之后我们发现两件事。第一，`Rating` 和 `Founded` 是仅有的两个直接可用的数值列，其余大多是字符串。第二，`Founded` 和 `Headquarters` 用 `-1` 表示“没有数据”，这不是真实的年份或地点，必须在清洗时单独处理。

接着我们看目标变量的分布。

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 左图：原始薪资分布
# 右图：取 log1p 之后的分布
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.histplot(train["avg_salary"], bins=30, kde=True, ax=axes[0])
axes[0].set_title("Distribution of Average Salary (Raw)")
axes[0].set_xlabel("Salary ($K)")

# log1p(x) = log(1 + x)，对 0 也安全
sns.histplot(np.log1p(train["avg_salary"]), bins=30, kde=True,
             ax=axes[1], color="coral")
axes[1].set_title("Distribution of Average Salary (Log1p Transformed)")
axes[1].set_xlabel("Log(Salary + 1)")

plt.tight_layout()
plt.show()
```

原始薪资明显右偏：大多数岗位集中在中间，少数高薪岗位拖出一条很长的尾巴。右偏会让 MSE 训练不稳定，因为少数极大值会产生巨大的平方误差。取 `log1p` 之后，分布接近钟形，模型更容易学习。于是我们决定**训练时对目标取对数，预测后再用 `expm1` 变回来**。

我们还想知道地点和职位类型对薪资有没有影响。

```python
# 从 Location 里取出州名：按逗号切开，取最后一段
train_viz = train.copy()
train_viz["state"] = train_viz["Location"].apply(
    lambda x: str(x).split(",")[-1].strip() if pd.notnull(x) else "Unknown"
)

# 把职位名称粗分成四类，方便画图
def group_job(title):
    title = str(title).lower()
    if "scientist" in title:
        return "Data Scientist"
    if "engineer" in title:
        return "Data Engineer"
    if "analyst" in title:
        return "Data Analyst"
    return "Other"

train_viz["Role"] = train_viz["Job Title"].apply(group_job)

# 按州求平均薪资，取最高的 10 个
top_states = (train_viz.groupby("state")["avg_salary"]
              .mean().sort_values(ascending=False).head(10))
print(top_states)
```

结果显示，不同州的平均薪资差距明显，加州、纽约这类科技公司密集的地方更高；数据科学家、数据工程师、数据分析师三类岗位的薪资分布也不一样。这提醒我们：`Location` 和 `Job Title` 不能简单丢掉，应该拆成更有意义的类别特征。

### 2.2 数据清洗

我们其次处理缺失值和“伪缺失”。`Rating` 的缺失直接用训练集的中位数填补；`Founded` 里的 `-1` 先替换成 `NaN`，再算公司年龄；地点、总部、规模、营收这些字符串列则拆成结构化字段。

```python
# Rating 用训练集的中位数填补，测试集也用同一个中位数，
# 保证训练和测试的处理方式一致
rating_median = train["Rating"].median()
train["Rating"] = train["Rating"].fillna(rating_median)
test["Rating"] = test["Rating"].fillna(rating_median)

# 把 Location 拆成城市和州
for df in [train, test]:
    df["state"] = df["Location"].apply(
        lambda x: str(x).split(",")[-1].strip() if pd.notnull(x) else "Unknown"
    )
    # -1 不是真实成立年份，先按缺失处理
    df["company_age"] = 2025 - df["Founded"]
    df.loc[df["Founded"] < 0, "company_age"] = -1
```

这里用 2025 减成立年份，是因为材料里的 notebook 就是这么算的，代表“公司到现在大概存在了多少年”。这个数字不是精确年龄，但能表达“老公司还是新公司”这个信号。`-1` 单独保留成一个特殊值，让树模型自己决定怎么处理。

`Size` 和 `Revenue` 原本是区间字符串，例如 `201 to 500 employees`、`$10+ billion (USD)`。树模型不能直接理解字符串，我们把它映射成有序的整数：公司越大、营收越高，数字越大。

```python
def get_size(x):
    """把公司规模区间映射成 1~7 的有序数字"""
    if "1 to 50 employees" in x:
        return 1
    if "51 to 200 employees" in x:
        return 2
    if "201 to 500 employees" in x:
        return 3
    if "501 to 1000 employees" in x:
        return 4
    if "1001 to 5000 employees" in x:
        return 5
    if "5001 to 10000 employees" in x:
        return 6
    if "10000+ employees" in x:
        return 7
    return np.nan  # Unknown、-1 等无法识别的值留空

for df in [train, test]:
    df["size_num"] = df["Size"].apply(get_size)
    # 竞争对手按逗号分隔，数量本身就是一个特征
    df["cnt_competitors"] = df["Competitors"].apply(
        lambda x: len(x.split(",")) if x != "-1" else np.nan
    )
```

我们接着从职位名称和职位描述里抽信号。职位名称里的 `senior`、`lead`、`junior` 代表资历；描述里如果出现 `phd`、`masters`，说明学历要求更高；描述里如果写了“X years of experience”，可以直接抽出一个经验年限。职位描述的长度也能反映岗位的复杂程度。

```python
import re

def extract_years_experience(text):
    """从职位描述里找 '3+ years of experience' 这类说法"""
    match = re.search(
        r"(\d+)\s*(?:\+|-\s*\d+)?\s*(?:years?|yrs?)(?:\s+of)?\s+experience",
        str(text).lower(),
    )
    return float(match.group(1)) if match else -1.0  # 找不到就用 -1 表示未知

for df in [train, test]:
    df["years_experience"] = df["Job Description"].apply(extract_years_experience)
    desc_lower = df["Job Description"].str.lower().fillna("")
    # 是否明确要求博士/硕士
    df["req_phd"] = desc_lower.str.contains(r"\b(phd|ph\.d\.|doctorate)\b").astype(int)
    df["req_masters"] = desc_lower.str.contains(r"\b(masters?|m\.s\.|m\.a\.|ms|ma)\b").astype(int)
    # 职位描述长度：长描述通常意味着岗位更专业
    df["description_len"] = df["Job Description"].fillna("").apply(len)
```

资历我们编码成三级：2 表示 senior/lead/manager/director/principal，0 表示 junior，1 表示普通岗位。职位类型则粗分成数据科学家、数据工程师、数据分析师和其他。这样类别数量不会太多，模型不会因为稀有类别而过拟合。

```python
for df in [train, test]:
    # 资历：2 高级，1 普通，0 初级
    df["seniority"] = df["Job Title"].apply(
        lambda x: 2 if any(w in str(x).lower() for w in
                           ["senior", "sr", "lead", "manager", "director",
                            "principal", "staff"])
        else (0 if any(w in str(x).lower() for w in ["junior", "jr"]) else 1)
    )
    # 职位类型粗分，避免上百种原始 title 造成维度爆炸
    df["job_type"] = df["Job Title"].apply(
        lambda x: "data_scientist"
        if any(w in str(x).lower() for w in ["scientist", "machine learning", "ml"])
        else ("data_engineer"
              if any(w in str(x).lower() for w in ["engineer", "architect"])
              else ("data_analyst"
                    if any(w in str(x).lower() for w in ["analyst", "analytics"])
                    else "other"))
    )
```

### 2.3 模型训练与超参数调优

我们接着解决最大的问题：职位描述是一大段自然语言，怎么变成特征。直接用词袋（把每个单词出现的次数数出来）会丢掉词序和语义，而且词汇表会很大。材料里的做法是用 `sentence-transformers` 里的 `all-MiniLM-L6-v2` 模型，把“职位名称 + 职位描述”编码成一个固定长度的向量。这个模型是在大量文本上预训练过的，能把语义相近的句子映射到相近的位置。

```python
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# 把职位名称和描述拼在一起，让模型同时看到岗位名和职责
for df in [train, test]:
    df["text"] = (df["Job Title"] + " " + df["Job Description"]).str.lower().fillna("")

# 加载预训练句向量模型
model_sbert = SentenceTransformer("all-MiniLM-L6-v2")

# 把每条文本编码成向量；batch_size 控制一次处理多少条
train_emb = model_sbert.encode(train["text"].tolist(), batch_size=32,
                               show_progress_bar=True)
test_emb = model_sbert.encode(test["text"].tolist(), batch_size=32,
                              show_progress_bar=True)

# 向量维度通常几百维，用 PCA 压到 30 维，降低过拟合和计算量
pca = PCA(n_components=30, random_state=42)
train_pca = pd.DataFrame(pca.fit_transform(train_emb),
                         columns=[f"pca_{i}" for i in range(30)])
test_pca = pd.DataFrame(pca.transform(test_emb),
                        columns=[f"pca_{i}" for i in range(30)])
print("30 个主成分解释的方差比例：", pca.explained_variance_ratio_.sum())
```

注意 PCA 只能在训练集上 `fit`，测试集只能 `transform`。如果先把训练集和测试集拼起来再 fit，就会把测试集的信息泄漏进特征里，交叉验证的分数会虚高。这是很多初学者容易犯的错误。

然后我们把数值特征、类别特征和文本主成分拼成一张大表。类别列转成 `category` 类型，LightGBM 可以直接处理，不需要独热编码。

```python
import lightgbm as lgbm

num_features = ["Rating", "company_age", "years_experience",
                "req_phd", "req_masters", "seniority",
                "description_len", "cnt_competitors", "size_num"]
cat_features = ["state", "Sector", "Type of ownership", "job_type"]

for df in [train, test]:
    for col in cat_features:
        df[col] = df[col].astype("category")

# 横向拼接：左边是表格特征，右边是文本主成分
X_train = pd.concat([train[num_features + cat_features], train_pca], axis=1)
X_test = pd.concat([test[num_features + cat_features], test_pca], axis=1)

# 目标取 log1p，和 EDA 的结论保持一致
y = np.log1p(train["avg_salary"])
```

超参数我们交给 Optuna 自动搜索。搜索空间包括学习率、叶子数、最大深度、行采样比例、列采样比例等。每一组参数都用 3 折交叉验证评估，用 RMSE 作为目标，越小越好。Optuna 的剪枝器会提前停掉明显不好的试验，节省时间。

```python
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from optuna.integration import LightGBMPruningCallback

def objective(trial):
    # 每个 trial 建议一组超参数
    params = {
        "n_estimators": 10000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 45),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 60),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
        "random_state": 42,
        "verbosity": -1,
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr, val in kf.split(X_train):
        model = lgbm.LGBMRegressor(**params)
        model.fit(
            X_train.iloc[tr], y.iloc[tr],
            eval_set=[(X_train.iloc[val], y.iloc[val])],
            eval_metric="rmse",
            callbacks=[
                lgbm.early_stopping(50, verbose=False),          # 50 轮不提升就停
                LightGBMPruningCallback(trial, "l2", "valid_0"),  # 提前剪枝
            ],
        )
        scores.append(root_mean_squared_error(
            y.iloc[val], model.predict(X_train.iloc[val])))
    return np.mean(scores)

study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=50, max_resource=2500, reduction_factor=3),
)
study.optimize(objective, n_trials=50)
print("最佳交叉验证 RMSE（对数尺度）：", study.best_value)
print("最佳参数：", study.best_params)
```

这段代码里有两个关键点。第一，`early_stopping` 让模型在验证集不再提升时停下，避免把树建得太多导致过拟合。第二，`LightGBMPruningCallback` 把每一轮的验证分数报告给 Optuna，Optuna 判断这组参数大概率不如之前的试验，就提前终止，不用跑满 10000 棵树。

### 2.4 模型测试与指标报告

最后我们用最佳参数做 5 折交叉验证。每一折训练一个模型，对测试集预测一次，把 5 个模型的预测取平均。平均能降低单个模型因为数据划分不同而产生的波动，通常比只用一折更稳。

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = np.zeros(len(test))

for tr, val in kf.split(X_train):
    model = lgbm.LGBMRegressor(**study.best_params)
    model.fit(
        X_train.iloc[tr], y.iloc[tr],
        eval_set=[(X_train.iloc[val], y.iloc[val])],
        callbacks=[lgbm.early_stopping(50, verbose=False)],
    )
    # 预测的是对数薪资，先 expm1 变回千美元，再累加
    preds += np.expm1(model.predict(X_test)) / kf.n_splits

submission = pd.DataFrame({"id": test["id"], "avg_salary": preds})
submission.to_csv("submission.csv", index=False)
print(submission.head())
```

这里最容易错的是单位。训练时 `y = log1p(avg_salary)`，模型输出的是对数值；如果直接把这个数写进提交文件，预测会变成个位数，跟真实薪资差得非常远。必须用 `np.expm1` 还原。`expm1` 是 `exp(x) - 1`，和 `log1p` 正好互逆。

指标报告方面，材料里的 notebook 打印了“最佳交叉验证 RMSE（对数尺度）”和最佳参数，但没有给出公开排行榜的具体分数。因此我们在这里只报告**本地 5 折交叉验证的 RMSE**，不编造排行榜数字。评估时我们统一在对数尺度上算 RMSE，因为训练目标就是对数值；如果要在原始薪资尺度上汇报，需要先 `expm1` 再算，两种口径不能混着比较。

我们还会检查提交文件的行数是否等于测试集行数、`id` 是否一一对应、有没有空值。这些检查很枯燥，但比赛里因为提交格式错误而白忙一场的情况并不少见。

---

## 三、总结

本文中我们完成了 GlassDoor 薪资预测比赛的完整流程。我们首先做 EDA，发现薪资右偏严重、地点和职位类型对薪资影响明显，于是决定对目标取 `log1p`。其次我们做数据清洗：用中位数补 `Rating`，把 `Founded` 的 `-1` 当作缺失，把 `Location`、`Headquarters` 拆成城市和州，把公司规模和营收区间映射成有序数字，并从职位描述里抽出经验年限、学历要求和描述长度。接着我们用 SentenceTransformer 把职位名称和描述编码成语义向量，再用 PCA 压到 30 维，和表格特征拼在一起。然后我们用 LightGBM 做回归，用 Optuna 搜索学习率、叶子数、深度、采样比例等超参数，并用早停和剪枝控制训练成本。最后我们用 5 折交叉验证训练五个模型，对测试集预测后取平均，用 `expm1` 还原成真实薪资并生成提交文件。

这个比赛给我们最大的启发是：**表格比赛的上限往往由特征决定，而不是由模型决定**。同样的 LightGBM，只喂 `Rating` 和 `Founded` 两个数字，和喂“结构化字段 + 文本语义向量”相比，效果会差很多。另一个教训是**目标变换要成对使用**：训练时 `log1p`，预测时就必须 `expm1`，否则提交的数值会完全不对。最后，文本嵌入加 PCA 是一种很实用的做法：它既保留了语义信息，又把维度控制在模型能承受的范围内，适合入门者在自己电脑上复现。
