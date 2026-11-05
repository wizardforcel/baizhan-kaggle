# 票房预测（Box Office Competition）

## 一、比赛简介

### 1.1 比赛背景

这场比赛的数据来自电影数据库 TMDB（The Movie Database）。每行是一部电影，除了片名、上映日期这些基本信息，还有**预算、片长、类型、演员、制作公司、发行国家、关键词**等一大串与影片制作相关的元数据。我们的目标是根据这些「上映前就已知」的信息，预测电影在**全球**能拿到多少票房（`revenue`）。

票房的分布非常极端：绝大多数电影是小成本小回报，只有极少数爆款能赚得盆满钵满。就像材料里说的——**收入严重「偏斜」（lopsided）：大量小片子，偶尔几部巨无霸**。这种长尾分布对回归模型很不友好：如果直接预测美元数字，模型会把精力都花在拟合几个天文数字上，小成本电影反而没人管。

### 1.2 比赛目标

训练集 `train.csv` 每一行是一部电影，主要列有：

- `title`、`release_date`：片名和上映日期；
- `budget`：制作预算（美元，但很多片子的预算是 0，表示「未知」）；
- `runtime`：片长（分钟）；
- `genres`、`production_companies`、`cast`、`crew`、`Keywords` 等：以**字符串**形式存放的列表数据（比如 `[{'id': 18, 'name': 'Drama'}]`）；
- `revenue`：全球票房（只有训练集有，是我们要预测的目标）。

测试集没有 `revenue`，我们要给每部电影（按 `id` 区分）预测一个票房数字并提交。

### 1.3 评价标准

比赛使用 **RMSLE（Root Mean Squared Logarithmic Error，均方根对数误差）**：

```
RMSLE = sqrt( 1/N * Σ (log(1 + 预测) - log(1 + 真实))² )
```

理解它的关键是「先取对数再算误差」。`log` 可以把爆款和小众片的差距压缩到同一个量级，这样**预测出错 10% 和 1000% 的票价相比不再被放大**，模型也不会只顾着拟合头部大片。**RMSLE 越低越好**，理论上最小是 0（全部预测精确）。

> 小技巧：如果我们直接把目标取对数 `y = log(1 + revenue)`，那么在 `y` 上算普通的均方根误差（RMSE），得到的数值就是比赛的 RMSLE——这就简化了本地评估的复杂度。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们读入三个文件，先看训练集的规模和几部典型电影：

```python
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")

train[["title", "budget", "runtime", "release_date", "revenue"]].head(10)
```

票房分布一眼看过去就很不均匀。我们画了两张直方图对比「原始票房」和「对数票房」：

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(train["revenue"], bins=50)
axes[0].set_title("Revenue")
# 取对数后，指数级的长尾被"拉直"，分布看起来均匀多了
axes[1].hist(np.log1p(train["revenue"]), bins=50)
axes[1].set_title("log(1 + revenue)")
plt.show()
```

结论很清楚：**选择 `log(1 + revenue)` 作为目标，比直接回归美元数字健康得多**。另外我们还发现预算和票房有明显的正相关——预算是目前最强的一个单特征，但不是所有电影都有预算（很多预算是 0），所以后面建模时要同时处理「有预算」和「没预算」两种情况。

除了预算，我们还从几个角度扫了一眼：

- **类型**：剧情、喜剧、动作是数量最多的三类，但「数量多」不等于「赚钱多」，类型更要看的是它和预算、档期的组合；
- **片长**：大部分电影集中在 90～120 分钟，极长或极短的片子都很少，属于长尾里的离群点；
- **上映时间**：月份分布并不均匀，暑期和年末档明显更密集，这提示我们「上映档期」可能是一个有用特征；
- **语言与系列**：英语片占多数，属于某个系列（`belongs_to_collection`）的电影通常有前作观众基础，票房也更集中。

### 2.2 数据清洗

数据有三类「脏」的地方，我们逐个处理：

1. **列表被存成了一段字符串**。`genres` 长这样 `[{'id': 18, 'name': 'Drama'}]`，模型没法直接用。我们用 `ast.literal_eval` 把它解析成 Python 列表，只保留每个元素的 `name`：
2. **两位数的年份有歧义**。`2/20/15` 是 2015 年还是 1915 年？Python 默认会读成 2045……我们把晚于 2019 的都退回 100 年（这份数据不可能有未来电影）；
3. **预算为 0 表示「未知」**而不是真的一分钱没花。小于 1000 美元的预算都当缺失值处理。

```python
import ast

LIST_COLUMNS = ["belongs_to_collection", "genres", "production_companies", "production_countries",
                "spoken_languages", "Keywords", "cast", "crew"]

def parse_list(text):
    """把 "[{'id': 18, 'name': 'Drama'}]" 变成 ['Drama']"""
    if pd.isna(text):
        return []
    return [item["name"] for item in ast.literal_eval(text)]

def clean(df):
    df = df.copy()
    # 1. 字符串列表 -> 真正的列表
    for col in LIST_COLUMNS:
        df[col] = df[col].apply(parse_list)

    # 2. 两位数年份修正：晚于 2019 的一律退回 100 年
    dates = pd.to_datetime(df["release_date"], format="%m/%d/%y", errors="coerce")
    df["release_date"] = dates.where(dates.dt.year <= 2019,
                                     dates - pd.DateOffset(years=100))

    # 3. 预算为 0（或只有几美元）视为缺失
    df["budget"] = df["budget"].where(df["budget"] >= 1000)
    return df

train_clean = clean(train)
test_clean  = clean(test)
```

清洗完还要留意一种「标签脏」：训练集里有少数电影的票房只有 1 美元这种明显不合理的值，它们多半是源数据录入错误。模型是照着标签学的，这种极端错标会把它往错误方向拽。材料给出的建议就是把这部分样本过滤掉（比如只保留 `revenue >= 1000` 的电影）再重新训练对比：

```python
print("票房低于 1000 美元的异常样本数:", (train_clean["revenue"] < 1000).sum())
# 把这些明显是录入错误的样本剔除（用注释开关，方便对比前后分数）
# train_clean = train_clean[train_clean["revenue"] >= 1000].reset_index(drop=True)
```

这是典型的「先怀疑标签，再怀疑模型」：当模型怎么调都卡在一个分数上时，回头检查训练标签的质量往往比继续调参更有效。

### 2.3 特征工程与模型训练

**特征工程的原则**：只使用「上映前就已知」的信息。材料特意强调**不能用 `popularity`**——它是电影上映后才由用户行为算出来的，用它等于「偷看未来」，比赛规则不允许。

我们组的特征分四类：

- **钱和时间**：`log_budget`（预算取对数）、`budget_missing`（预算是否缺失）、`year`/`month`/`day_of_week`（上映年份、月份、星期几）、`runtime`；
- **营销与系列**：是否属于一个电影系列（`in_collection`）、有没有官网（`has_homepage`）、有没有宣传语（`has_tagline`）、是不是英语片（`is_english`）；
- **规模计数**：类型、演员、剧组、关键词、制作公司的数量；
- **热门类型的独热编码**：从训练数据里统计出最常用的 10 个类型，每个类型做一列 0/1。

```python
TOP_GENRES = train_clean["genres"].explode().value_counts().head(10).index

def make_features(df):
    X = pd.DataFrame(index=df.index)

    X["log_budget"]     = np.log1p(df["budget"])              # 预算取对数，和票房同一量纲
    X["budget_missing"] = df["budget"].isna().astype(int)     # 没有预算本身也是个信号
    X["year"]           = df["release_date"].dt.year
    X["month"]          = df["release_date"].dt.month
    X["day_of_week"]    = df["release_date"].dt.dayofweek
    X["runtime"]        = df["runtime"]

    X["in_collection"]  = (df["belongs_to_collection"].apply(len) > 0).astype(int)
    X["has_homepage"]   = df["homepage"].notna().astype(int)
    X["has_tagline"]    = df["tagline"].notna().astype(int)
    X["is_english"]     = (df["original_language"] == "en").astype(int)

    X["num_genres"]     = df["genres"].apply(len)
    X["num_cast"]       = df["cast"].apply(len)
    X["num_crew"]       = df["crew"].apply(len)
    X["num_keywords"]   = df["Keywords"].apply(len)
    X["num_companies"]  = df["production_companies"].apply(len)

    # 常用类型各做一列 0/1
    for genre in TOP_GENRES:
        X["genre_" + genre.replace(" ", "_")] = df["genres"].apply(lambda g: int(genre in g))
    return X

X     = make_features(train_clean)
X_test = make_features(test_clean)
y     = np.log1p(train_clean["revenue"])     # 目标取对数，正好对上 RMSLE
```

**基线。** 建模之前先立一个下限：对每一部电影都猜「训练集中位数」。任何模型打不过它就没有意义：

```python
def rmse(predicted, actual):
    return np.sqrt(np.mean((np.asarray(predicted) - np.asarray(actual)) ** 2))

baseline = rmse(np.full(len(y), y.median()), y)
print(f"Baseline RMSLE (同一中位数猜到底): {baseline:.4f}")
```

**正式模型：LightGBM。** LightGBM 是梯度提升树库，它训练几百棵小决策树，每棵纠正上一棵的残差；它能原生处理缺失值，所以我们不需要提前把空值都填上。做 5 折交叉验证评估：

```python
import lightgbm as lgb
from sklearn.model_selection import KFold

folds = KFold(n_splits=5, shuffle=True, random_state=42)

def make_lgbm():
    return lgb.LGBMRegressor(
        n_estimators=500,      # 树的数量
        learning_rate=0.03,    # 每棵树的修正步长，调小更稳但要更多树
        num_leaves=15,         # 每棵树的叶子数量，越小越不容易过拟合
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42,
    )

def cross_validate(make_model):
    scores = []
    for train_rows, check_rows in folds.split(X):
        model = make_model()
        model.fit(X.iloc[train_rows], y.iloc[train_rows])
        scores.append(rmse(model.predict(X.iloc[check_rows]), y.iloc[check_rows]))
    print("每折 RMSLE:", np.round(scores, 3))
    print(f"平均 RMSLE: {np.mean(scores):.4f}")
    return np.mean(scores)

lgbm_score = cross_validate(make_lgbm)
print(f"相对基线的提升: {baseline - lgbm_score:.3f}")
```

几个超参数的设计意图说明一下：`n_estimators=500` 配 `learning_rate=0.03` 是「多树 + 小步长」的组合，比「少树 + 大步长」更不容易过拟合；`num_leaves=15` 限制了每棵树的复杂度，叶子越多模型越激进；`subsample=0.8` 和 `colsample_bytree=0.8` 分别是行采样和列采样，相当于给每棵树一点「随机性」，让森林整体更稳。由于 `y` 里我们还是保留了缺失的特征值（比如没有预算的电影），LightGBM 能原生处理 `NaN`，所以不需要额外填充。

5 折交叉验证除了给一个平均分，还要注意**看每折之间的离散程度**：如果五折分数忽高忽低，说明模型对数据划分很敏感，这时候加正则（降 `num_leaves`、升 `subsample`）比加特征更优先。任何想上线的新特征或新参数，都要先过这一关。

训练完我们画特征重要性图，看看模型到底押注在哪些特征上（不出意外，预算、类型数量、上映年份排在最前面），这也能指导下一轮迭代：

```python
model = make_lgbm().fit(X, y)
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importance.plot.barh(figsize=(7, 8), title="Feature importance")
plt.show()
```

### 2.4 模型测试与指标报告

5 折交叉验证给出了模型在「没见过的数据」上的平均 RMSLE，相对「中位数基线」有明显的下降，说明特征是有效的。两个改进方向在材料里都被点名：

- 清洗层：部分训练样本的票房只有几美元，明显是源数据错误，把它们删掉可以让模型更专注（`train = train[train["revenue"] >= 1000]`）；
- 模型层：学完 LightGBM 再学一个 CatBoost，把两份预测取平均（ensemble），通常还能再往下压一点分数。

任何改动都遵循**同一套回测流程**：改了 → 重新跑 5 折 → 分数没有下降才保留。

确认模型可用后，用**全量训练数据**重新训练一版，预测测试集，再记得把对数还原成美元（`expm1`），并确保提交文件的行数、`id` 顺序与官方样例一致：

```python
final_model = make_lgbm().fit(X, y)

predictions = np.expm1(final_model.predict(X_test))   # 还原成美元
predictions = predictions.clip(min=0)                 # 票房不能为负

submission = pd.DataFrame({"id": test["id"], "revenue": predictions})
submission.to_csv("submission.csv", index=False)

# 行数与顺序对齐校验，防止交一个格式错的提交
assert len(submission) == len(sample) and set(submission["id"]) == set(sample["id"])
print("Saved submission.csv with", len(submission), "predictions")
```

---

## 三、总结

本文中我们完成了票房预测的完整流程：首先通过 EDA 发现票房呈严重长尾分布，因此选择 `log(1 + revenue)` 作为回归目标以对齐 RMSLE 评价指标；其次我们清洗了三类脏数据——把字符串形式的列表解析成真列表、修正两位数年份的歧义、把 0 预算当作缺失；接着我们构建了「钱和时间 + 营销 + 规模计数 + 热门类型独热」四类特征，坚持以「上映前已知」为原则，并故意排除了上映后才有的 `popularity`；然后我们用 LightGBM 做 5 折交叉验证，用 RMSLE 对比中位数基线验证模型有效性，并用特征重要性指导迭代；最后我们全量重训、把对数预测还原成美元并做了提交格式校验。

这个比赛给我们最大的启发是：**目标函数的形状决定了建模的全部**——因为评价指标是 RMSLE，我们就取对数目标；因为票房分布长尾，我们就重点处理预算缺失并放弃估算不出的爆款。凡是和评分方式脱节的技巧，都是白费功夫。