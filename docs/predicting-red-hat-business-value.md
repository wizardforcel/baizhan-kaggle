# Red Hat 商务价值预测

## 一、比赛简介

### 1.1 比赛背景

Red Hat 商务价值预测比赛把企业客户与客户行为放在一起，要求我们从历史活动中判断一次活动有没有带来预期的商务结果。材料把任务描述成一个二分类问题：活动表中的 `outcome` 是结果标签，取值表示这次活动之后是否在固定时间窗口内完成了结果。数据不是一张表，而是两张表。`people.csv` 记录每个客户以及客户特征，每行对应一个唯一的 `people_id`；`act_train.csv` 和 `act_test.csv` 记录客户参与过的活动，每行对应一个唯一的 `activity_id`，并包含活动类别、活动特征和日期。这个设计很接近真实业务：客户信息相对稳定，活动信息随着时间不断增加，我们必须用 `people_id` 把两类信息接起来。

这场比赛的难点首先来自数据规模。材料中的探索显示，客户表有 189118 行，训练活动有 2197291 行，测试活动有 498687 行。其次，字段中有许多类别型变量，值常常以类似 `type 1` 的字符串形式保存，还夹杂布尔变量、日期和一个数值变量。再次，活动表中的部分特征只在某一类活动中出现，所以缺失并不一定表示数据损坏，而可能是业务字段本来就不适用。我们不能简单地删除这些列，否则会丢掉活动类型信息。

### 1.2 比赛目标

我们的输入是客户信息和活动信息。客户信息包括 `people_id`、`group_1`、`char_1` 到 `char_38` 以及客户日期；活动信息包括 `activity_id`、`people_id`、活动日期、`activity_category`、活动特征和训练标签 `outcome`。我们要先用 `people_id` 合并训练活动与客户特征，再为测试活动预测一个结果概率。材料明确提出的业务目标是找出“谁、在什么时候、通过什么活动”更可能产生 Red Hat 所关心的商务价值，因此输出不是简单的规则，而是每条测试活动对应的连续概率，方便按照可能性排序。

### 1.3 评价标准

材料多次说明比赛用 AUC 衡量分类性能。AUC 可以理解为：随机抽取一个真正有结果的活动和一个没有结果的活动，模型把前者排在前面的概率。它只关注排序，不要求我们把概率恰好校准到某个具体数值；AUC 越高，说明模型越能把潜在高价值活动排在前面。我们在本地可以用验证集计算 `roc_auc_score`，最后提交文件需要保留测试活动的标识以及模型预测的概率。由于测试集没有公开标签，最终排行榜分数只能由比赛平台计算，我们不能在材料没有给出的地方补写具体成绩。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取两张训练表和一张测试表，确认列名、样本量、数据类型，并把客户表和活动表的关系弄清楚。探索时我们重点观察三件事：客户与活动的规模，标签是否有明显偏斜，以及活动类型和日期是否可能影响结果。材料中的 EDA 还提出了几个合理假设：不同活动带来的价值可能不同；一年中的不同时间可能有不同成功率；某些客户群体可能更有价值；客户特征和活动特征都可能包含信号。

```python
# 读取客户、训练活动和测试活动，先保留原始表便于核对字段
import pandas as pd
import numpy as np

people = pd.read_csv("../input/predicting-red-hat-business-value/people.csv.zip")
activity = pd.read_csv("../input/predicting-red-hat-business-value/act_train.csv.zip")
test = pd.read_csv("../input/predicting-red-hat-business-value/act_test.csv.zip")

# 用 shape 认识规模，避免把客户行数误当成活动行数
print("people:", people.shape)
print("activity:", activity.shape)
print("test:", test.shape)

# 查看标签数量，二分类任务必须先知道正负样本是否失衡
print(activity["outcome"].value_counts(dropna=False))

# 检查客户键是否唯一，people_id 应该能唯一定位客户
print(people["people_id"].nunique(), len(people))
```

我们其次检查缺失值和字段类型。材料指出客户表没有明显缺失，但活动表的 `act_char_1` 到 `act_char_9` 有相同数量的缺失，因为这些字段只对活动类型 1 有意义；`act_char_10` 也存在缺失。这个现象提醒我们，缺失模式本身可能和活动类别有关，不能把所有空值都当成随机漏填。我们还可以按 `outcome` 分组统计客户和活动类别，观察正负结果是否集中在少数客户或少数活动上。

```python
# 找出每张表真正存在缺失的列，重点关注缺失模式
people_missing = people.isna().sum()
activity_missing = activity.isna().sum()
print("people missing:")
print(people_missing[people_missing > 0])
print("activity missing:")
print(activity_missing[activity_missing > 0])

# 按结果查看活动类别数量，帮助我们判断类别是否带来初步信号
category_outcome = pd.crosstab(
    activity["activity_category"], activity["outcome"],
    margins=True
)
print(category_outcome)

# 按客户统计成功活动次数，观察是否存在重复活动行为
success_by_person = (
    activity.loc[activity["outcome"] == 1]
    .groupby("people_id")["activity_id"]
    .count()
    .sort_values(ascending=False)
)
print(success_by_person.head(20))
```

### 2.2 数据清洗

我们首先把两个表中的日期转为日期类型，再拆出月份和年份。这样做不是为了凭空增加信息，而是把“发生在什么时候”变成模型可以直接使用的数字。材料中的一个方案使用 `month_people`、`year_people`、`month_act` 和 `year_act`，然后去掉原始日期列。我们其次处理类别字段：如果字段是对象类型，就先用 `type 0` 填补缺失，再取字符串中的数字部分；布尔字段转换为 0/1。这里的核心原则是保留业务类别，而不是把它们当作自然语言。

```python
# 将日期拆成月和年，让模型能够学习季节和时间阶段
people["date"] = pd.to_datetime(people["date"], format="%Y-%m-%d")
people["month_people"] = people["date"].dt.month
people["year_people"] = people["date"].dt.year
people = people.drop("date", axis=1)

activity["date"] = pd.to_datetime(activity["date"], format="%Y-%m-%d")
activity["month_act"] = activity["date"].dt.month
activity["year_act"] = activity["date"].dt.year
activity = activity.drop("date", axis=1)

# 保留日期拆分后的字段，同时删除只用于连接或标识的原始列
print(people[["month_people", "year_people"]].head())
print(activity[["month_act", "year_act"]].head())
```

接着我们把类别字符串和布尔值统一编码。材料给出的 `act_data_treatment` 会跳过 `people_id`、`activity_id`、日期派生列、`char_38` 和 `outcome`，对对象列填补后取第二个空格分隔字段，对布尔列转成整数。实际使用时，我们还需要注意训练和测试使用一致的转换逻辑；如果某个类别只在测试出现，应该采用能处理未知类别的编码方式。材料中的主线使用 LightGBM 的类别特征能力，因此不必把每一个类别展开成大量独热列。

```python
# 统一处理类别和布尔字段，减少字符串带来的训练负担

def act_data_treatment(dataset):
    dataset = dataset.copy()
    skip_cols = ["people_id", "activity_id", "date", "char_38", "outcome"]
    for col in dataset.columns:
        if col in skip_cols:
            continue
        if dataset[col].dtype == "object":
            # 缺失类别保留为一个单独类型，避免直接丢掉样本
            dataset[col] = dataset[col].fillna("type 0")
            dataset[col] = dataset[col].apply(
                lambda value: value.split(" ")[1]
            ).astype(np.int32)
        elif dataset[col].dtype == "bool":
            # 布尔字段转成 0/1，树模型可以直接使用
            dataset[col] = dataset[col].astype(np.int8)
    return dataset

activity = act_data_treatment(activity)
people = act_data_treatment(people)
print(activity.dtypes.value_counts())
```

最后我们按 `people_id` 合并两张表，并删除只用来连接或标识的列。材料中的合并采用内连接，得到统一训练表；另一份方案也显示，合并后可以直接对缺失列进行检查。我们不删除活动类型 1 的空字段，因为它们反映的是字段适用范围。对于模型不需要的 `activity_id` 和 `people_id`，我们只在建模特征中排除，提交时仍要保留测试活动的身份。

```python
# 将客户静态特征拼到每一条活动上，形成一行一活动的训练矩阵
train = pd.merge(activity, people, how="inner", on="people_id")

# outcome 是标签，连接键和活动编号不是预测特征
features = [
    col for col in train.columns
    if col not in ["outcome", "activity_id", "people_id"]
]
X = train[features]
y = train["outcome"]

print("merged shape:", train.shape)
print("feature count:", len(features))
print("target mean:", y.mean())
```

### 2.3 模型训练与超参数调优

我们首先选用 LightGBM，因为材料中的多个方案都把它用于二分类，并且它能够直接处理高维表格数据和类别特征。我们把训练数据分成训练部分和验证部分，验证集只用于比较模型，不参与拟合。材料中的一个方案使用 75% 训练、25% 验证的随机划分，另一个方案展示了 `GridSearchCV`、`StratifiedKFold` 和 `LGBMClassifier` 的组合。由于目标是 AUC，训练时使用二分类目标，并把 `binary_logloss`、`auc` 或 `roc` 作为监控指标。

```python
# 切分验证集，让我们能在提交前检查排序能力
from sklearn.model_selection import train_test_split
import lightgbm as lgbm

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

# 告诉 LightGBM 哪些列是类别特征，避免错误地把类别当连续数值
categorical_features = [
    col for col in features if X_train[col].dtype.name in ["int8", "int32"]
]
train_set = lgbm.Dataset(
    X_train, y_train, categorical_feature=categorical_features
)
valid_set = lgbm.Dataset(
    X_valid, y_valid, categorical_feature=categorical_features
)

params = {
    "learning_rate": 0.03,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": ["binary_logloss", "auc", "roc"],
    "min_gain_to_split": 0.25,
    "verbose": 10,
}
```

我们其次控制模型复杂度和训练轮数。材料中的 LightGBM 方案设置学习率 `0.03`、`num_iterations` 为 `10100`、`max_bin` 为 `30000`，并以最多 `15000` 轮训练，同时用验证集观察指标。这里的数字都来自材料，我们不把它们解释成普适最优值，而是作为这条方案的可复现实验配置。对于入门者来说，学习率可以理解为每棵树修正前面错误的步子大小；轮数越多，模型越有机会拟合，但也更容易过拟合。

```python
# 用材料中的参数训练梯度提升树，并在验证集上监控 AUC
model = lgbm.train(
    params,
    train_set=train_set,
    num_boost_round=15000,
    valid_sets=[train_set, valid_set],
    verbose_eval=10,
)

# 预测概率而不是直接输出 0/1，AUC 需要完整排序信息
valid_prob = model.predict(X_valid)
print("validation probability range:", valid_prob.min(), valid_prob.max())
```

我们最后可以把调参理解成对学习率、树数量、叶子数量和采样比例的有控制比较。材料提到使用 `GridSearchCV` 和分层交叉验证，也列出了 LightGBM 的 sklearn 接口。实际操作时，我们优先固定数据处理流程，再在有限范围内比较参数，并用 AUC 选择方案，而不是根据一次随机切分的偶然结果反复试验。若采用神经网络、随机森林或 XGBoost，也要保持同样的验证规则，才能公平比较。

```python
# 用 sklearn 接口保留一个可扩展的参数搜索入口
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

estimator = LGBMClassifier(
    objective="binary",
    learning_rate=0.03,
    n_estimators=1000,
    random_state=0,
)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# scoring 使用 roc_auc，和比赛的评价方向保持一致
scores = cross_val_score(
    estimator, X, y, cv=folds, scoring="roc_auc", n_jobs=-1
)
print("fold AUC:", scores)
print("mean AUC:", scores.mean())
```

### 2.4 模型测试与指标报告

我们首先在验证集上计算 AUC，并检查预测概率是否包含缺失值。因为材料没有给出最终排行榜分数，我们只报告训练流程和验证方式，不编造具体结果。我们其次用全部训练活动重新拟合模型，再对合并后的测试活动预测。预测必须使用和训练完全相同的日期拆分、类别处理、字段顺序和客户连接方式，否则模型看到的列含义会变化。

```python
# 计算验证集 AUC，验证模型是否能把正样本排在前面
from sklearn.metrics import roc_auc_score

valid_auc = roc_auc_score(y_valid, valid_prob)
print("validation AUC:", valid_auc)
print("missing prediction:", np.isnan(valid_prob).sum())

# 查看不同阈值下的正例比例，帮助理解概率输出而非只看一个阈值
for threshold in [0.2, 0.5, 0.8]:
    predicted_label = (valid_prob >= threshold).astype(int)
    print(threshold, predicted_label.mean())
    # 阈值越低，被判为正例的样本越多
    print("positives:", int(predicted_label.sum()))
```

然后我们为测试表执行同样的预处理和合并，使用全量训练数据训练最终模型。材料中的提交思路是保留测试活动的 `activity_id`，把预测值写入 `outcome` 列，再保存为 CSV。这里的预测值是概率，不是手工设定的标签；这样既符合 AUC 的排序要求，也保留了业务方按风险或潜力排序的空间。

```python
# 对测试活动执行与训练完全一致的时间和类别处理
people_test = people.copy()
test_processed = act_data_treatment(test.copy())

# 合并客户特征，保证测试矩阵和训练矩阵字段一致
test_merged = pd.merge(
    test_processed, people_test, how="inner", on="people_id"
)
X_test = test_merged[features]

# 使用全部训练数据重新拟合，减少只使用验证划分带来的信息损失
full_set = lgbm.Dataset(
    X, y, categorical_feature=categorical_features
)
final_model = lgbm.train(
    params, full_set, num_boost_round=15000
)
test_prob = final_model.predict(X_test)
```

最后我们检查提交文件的列名、行数和预测范围。若测试活动排序在合并时发生变化，必须按 `activity_id` 找回原始顺序；材料明确强调提交需要包含测试活动的标识。由于不同 notebook 对列名写法可能略有差异，我们要以比赛提供的 `sample_submission.csv` 为准，而不是凭经验猜列名。

```python
# 按样例提交文件的结构生成结果，避免列名或顺序错误
sample = pd.read_csv(
    "../input/predicting-red-hat-business-value/sample_submission.csv.zip"
)
submission = sample.copy()
submission["outcome"] = test_prob

# 提交前做三项机械检查：行数、缺失值、预测范围
assert len(submission) == len(sample)
assert submission["outcome"].notna().all()
assert ((submission["outcome"] >= 0) &
        (submission["outcome"] <= 1)).all()
submission.to_csv("submission.csv", index=False)
print(submission.head())
```

## 三、总结

本文中我们完成了 Red Hat 商务价值预测的完整流程。我们首先认识到数据由客户表和活动表组成，训练活动有 2197291 行，测试活动有 498687 行，客户表有 189118 行；其次通过 EDA 检查了标签、活动类别、客户重复和缺失模式，理解了活动类型 1 的部分空字段是字段适用范围造成的。接着我们把日期拆成年、月，把类别字符串和布尔值编码，再用 `people_id` 合并客户与活动，保留了能表达业务行为的字段。然后我们用 LightGBM 二分类模型学习非线性的表格关系，并用分层验证和 AUC 监控排序效果；最后我们使用全量训练数据预测测试活动，按照样例格式生成概率提交文件。

这场比赛带给我们的主要经验是，业务表格的关键往往不在某一个复杂模型，而在数据关系是否处理正确。客户级字段必须准确连接到活动级记录，缺失值必须结合业务含义解释，日期不能在读取后被悄悄丢掉，类别编码也必须在训练和测试之间保持一致。我们还看到，AUC 任务应保留预测概率，不能只输出硬分类标签。材料中还出现了神经网络、随机森林、XGBoost 和分组折叠等其他探索方向，但本文选择了信息完整、流程清楚的 LightGBM 主线，便于初学者复现实验并进一步学习。
