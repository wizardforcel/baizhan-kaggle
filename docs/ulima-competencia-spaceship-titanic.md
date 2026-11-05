# ULIMA Spaceship Titanic

## 一、比赛简介

### 1.1 比赛背景
Spaceship Titanic 是 Kaggle 上一个以太空旅行事故为背景的二分类预测比赛。材料里出现的代码都在读取 `train.csv` 和 `test.csv`，训练集包含乘客的姓名、家乡星球、是否在冷冻舱、舱位号、目的地、年龄、是否为 VIP、以及五项消费记录等信息。比赛的任务是根据这些上船前的乘客特征，判断乘客在事故发生后是否被成功转移 `Transported`。

代码里对 `PassengerId` 的处理显示编号是 `gggg_pp` 形式，`gggg` 代表一个乘客组，`pp` 代表组内序号，说明乘坐是按家庭或团体进行的。`Cabin` 列被拆分为甲板、舱号、侧边，说明舱位位置也可能和获救概率有关。总体来说，这是一个典型的表格数据分类问题，难点在于大量缺失值和类别特征需要手工提取。

### 1.2 比赛目标
给定一条乘客记录，预测其 `Transported` 是否为 True。训练集每条样本包含：
- `PassengerId`：乘客编号
- `HomePlanet`：家乡星球
- `CryoSleep`：是否冷冻睡眠
- `Cabin`：舱位编码
- `Destination`：目的地星球
- `Age`：年龄
- `VIP`：是否 VIP
- `RoomService`、`FoodCourt`、`ShoppingMall`、`Spa`、`VRDeck`：五项消费金额
- `Name`：姓名
- `Transported`：是否被转移，仅训练集有

测试集没有 `Transported` 列，我们需要输出 `PassengerId` 和预测的 `Transported`。

### 1.3 评价标准
材料里反复使用 `accuracy_score` 做模型评估，并在 5 折分层交叉验证里计算每折准确率。最终提交文件用布尔值判断，排行榜上用准确率，越高越好。

## 二、解题过程

整场比赛的思路可以概括成一句话：先把原始表格做特征工程，把缺失值和文本字段转成数值，再用树模型做预测。我们按 EDA、数据清洗、模型训练与调优、测试与报告四步来走。

### 2.1 数据探索（EDA）
我们首先把训练集、测试集、样本提交文件读进来，确认文件是否找得到、列名是否一致。
```python
import os
import pandas as pd

# 遍历输入目录自动找到 train.csv 和 test.csv
train_path = None
test_path = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files:
        train_path = os.path.join(root, "train.csv")
    if "test.csv" in files:
        test_path = os.path.join(root, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train Shape :", train.shape)  # 看样本数和特征数
print("Test Shape  :", test.shape)
print("列名:", train.columns.tolist())
```
这一步让我们确认数据加载成功，并知道有多少列需要处理。这里我们特意使用 `os.walk` 去找文件，而不是把路径写死，是因为 Kaggle 上挂载的数据目录名字会变化，写死路径很容易在换环境后找不到文件；自动搜索虽然多几行代码，却能让 Notebook 更稳。

为了知道哪些列缺失严重，我们统计了每个列的缺失数和缺失比例。
```python
# 统计缺失值数量和占比
missing_table = pd.DataFrame({
    "Missing": train.isnull().sum(),
    "Percentage": (train.isnull().mean() * 100).round(2)
})
missing_table = missing_table[missing_table["Missing"] > 0].sort_values("Percentage", ascending=False)
missing_table
```
从结果看，`Age`、`Cabin`、五项消费记录都有不少缺失，`HomePlanet`、`CryoSleep` 等类别列也有缺失。这说明不能直接丢弃，需要先填补再建模。我们特别关注消费列的缺失，因为一个人如果处于冷冻睡眠状态，大概率不会在船上消费，所以“消费为空”本身可能就暗示了 `CryoSleep`，后续可以考虑把它们填成 0，而不是简单用均值代替。

我们还查看了目标分布。
```python
print(train["Transported"].value_counts())
```
目标不是完全平衡，模型需要对正负样本都敏感。

### 2.2 数据清洗
材料里采用的清洗策略是：先把训练集和测试集合并做统一编码，避免测试集出现训练集没见过的类别；再对每种字段做针对性处理。

我们首先把类别字段的缺失填为 `Unknown`，并把布尔字段转为字符串。
```python
full = pd.concat([train.drop(columns=["Transported"]), test], ignore_index=True)

for col in ["HomePlanet", "Destination"]:
    full[col] = full[col].fillna("Unknown").astype(str)

for col in ["CryoSleep", "VIP"]:
    full[col] = full[col].replace({True:"True", False:"False"})
    full[col] = full[col].fillna("Unknown").astype(str)
```
这样后续做独热编码时不会报错。

乘客组信息从 `PassengerId` 解析出来。
```python
id_parts = full["PassengerId"].astype(str).str.split("_")
full["GroupID"] = id_parts.str[0]
full["GroupMemberNo"] = pd.to_numeric(id_parts.str[1], errors="coerce")
group_counts = full["GroupID"].value_counts()
full["GroupSize"] = full["GroupID"].map(group_counts).astype(float)
full["IsAlone"] = (full["GroupSize"] == 1).astype(int)
```
组大小和是否独自出行是常用特征。材料里还进一步计算了每个姓氏出现的次数，用作家族规模的代理变量，因为 Spaceship Titanic 的故事里家庭成员常在一起行动，家人的数量可能影响被救助的优先级。姓名字符长度也在特征里，名字很短或很长都可能和数据录入质量有关，保留下来让模型自己判断。
```python
clean_name = full["Name"].fillna("Unknown Unknown").astype(str)
full["Surname"] = clean_name.str.split().str[-1]
surname_counts = full["Surname"].value_counts()
full["FamilySizeBySurname"] = full["Surname"].map(surname_counts).astype(float)
full["NameLength"] = clean_name.str.len().astype(float)
```


舱位信息拆分为甲板、舱号、侧边。
```python
cabin = full["Cabin"].fillna("U/-1/U").astype(str).str.split("/", expand=True)
full["Deck"] = cabin[0].fillna("U").replace("", "U")
full["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
full["Side"] = cabin[2].fillna("U").replace("", "U")
```
舱号再分箱成 `CabinRegion`，让模型更容易学习位置规律。

消费类特征先把缺失填 0，再做汇总。
```python
spend_columns = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
spending = full[spend_columns].fillna(0)
full["TotalSpend"] = spending.sum(axis=1)
full["NoSpend"] = (full["TotalSpend"] == 0).astype(int)
full["AmenitiesUsed"] = (spending > 0).sum(axis=1)
full["LuxurySpend"] = spending["RoomService"] + spending["Spa"] + spending["VRDeck"]
full["FoodShoppingSpend"] = spending["FoodCourt"] + spending["ShoppingMall"]
```
对消费做 log1p 变换，减少极端值影响。

年龄也做了分箱。
```python
full["IsChild"] = (full["Age"] < 13).astype(int)
full["IsYoung"] = (full["Age"] < 25).astype(int)
full["AgeBand"] = pd.cut(full["Age"], bins=[-np.inf,12,18,25,35,50,65,np.inf],
                         labels=["Child","Teen","Young","YoungAdult","Adult","MiddleAge","Senior"])
```
最后对所有数值列用中位数填补，类别列用 `Unknown` 填补，再做 `pd.get_dummies` 一键独热编码。这样训练集和测试集的列对齐。

### 2.3 模型训练与超参数调优
清洗完特征后，我们用分层 5 折交叉验证避免数据泄露，并对每折模型求概率平均。
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import numpy as np

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def train_cv_model(model, model_name):
    oof_probability = np.zeros(len(X))
    test_probability = np.zeros(len(X_test))
    fold_scores = []
    for fold, (train_index, valid_index) in enumerate(cv.split(X, y), start=1):
        current_model = clone(model)
        current_model.fit(X.iloc[train_index], y.iloc[train_index])
        valid_prob = current_model.predict_proba(X.iloc[valid_index])[:,1]
        oof_probability[valid_index] = valid_prob
        fold_pred = (valid_prob >= 0.50).astype(int)
        fold_acc = accuracy_score(y.iloc[valid_index], fold_pred)
        fold_scores.append(fold_acc)
        test_probability += current_model.predict_proba(X_test)[:,1] / cv.n_splits
    # 用 OOF 概率搜索最佳阈值
    thresholds = np.arange(0.40, 0.601, 0.005)
    best_acc = 0
    best_thr = 0.5
    for thr in thresholds:
        acc = accuracy_score(y, (oof_probability >= thr).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    print(f"{model_name} 平均折准确率 {np.mean(fold_scores):.5f}, 最佳阈值 {best_thr:.3f}")
    return oof_probability, test_probability, best_thr
```
这段代码保证我们选阈值时不会泄漏。

材料里尝试了多个树模型，首选是 `HistGradientBoostingClassifier`。
```python
from sklearn.ensemble import HistGradientBoostingClassifier

model_1 = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=400,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)
oof_1, test_prob_1, thr_1 = train_cv_model(model_1, "HistGradientBoosting")
```
参数选择强调正则化和小叶子样本，避免过拟合。

随后也尝试了 `RandomForest`、`ExtraTrees`、`GradientBoosting`，并用类似流程训练。
```python
from sklearn.ensemble import RandomForestClassifier

model_2 = RandomForestClassifier(
    n_estimators=700,
    max_depth=13,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```
超参数调优主要靠经验和交叉验证的平均准确率，没有做耗时的网格搜索。

在另一个简易方案里，材料也展示了用 `TotalSpend` 做特征工程后，用 XGBoost 做快速验证。
```python
from xgboost import XGBClassifier

spend_cols = ["RoomService","FoodCourt","ShoppingMall","Spa","VRDeck"]
for df in [train, test]:
    df[spend_cols] = df[spend_cols].fillna(0)
    df["Age"] = df["Age"].fillna(train["Age"].median())
    df["TotalSpend"] = df[spend_cols].sum(axis=1)

model_xgb = XGBClassifier(n_estimators=150, learning_rate=0.05, random_state=42, eval_metric="logloss")
model_xgb.fit(X, y)
```
这说明即使是简单的消费汇总特征也能带来提升。

### 2.4 模型测试与指标报告
训练完成后，我们用全量训练数据重新拟合，并在测试集上生成概率，通过最优阈值转为布尔预测。
```python
def create_submission(test_probability, threshold, filename):
    prediction = test_probability >= threshold
    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Transported": prediction.astype(bool)
    })
    # 检查形状和列名
    assert len(submission) == len(sample_submission)
    assert submission.columns.tolist() == sample_submission.columns.tolist()
    submission.to_csv("/kaggle/working/" + filename, index=False)
    print("✅ Submission created:", filename)
    return submission
```
生成文件后会打印预测分布，确认没有全是同一类的错误。

我们还做了简单集成验证：用逻辑回归、随机森林、带特征工程的随机森林、XGBoost 四个模型投票。
```python
votes = pd.DataFrame({"p1": pred1, "p2": pred2, "p3": pred3, "p4": pred4})
final_prediction = votes.sum(axis=1) >= 2
```
投票策略在材料里作为第二种提交方案出现，说明作者在追求稳定性的同时也尝试了集成。

## 三、总结
本文中我们完成了 ULIMA Spaceship Titanic 比赛的完整流程。我们首先用 EDA 摸清了数据的缺失情况和目标分布，其次在数据清洗阶段做了乘客组解析、舱位拆分、消费汇总和年龄分箱等特征工程。然后我们用分层 5 折交叉验证训练了 HistGradientBoosting、RandomForest、ExtraTrees 等树模型，并通过 OOF 概率搜索最优阈值。最后我们生成了符合格式的提交文件，并在验证集上报告了准确率。

从材料来看，特征工程是提升的关键：从原始舱位编码和消费字段中提取总消费、无消费、奢侈消费等指标，比直接使用原始列效果更好。模型方面，梯度提升类模型在这一套特征上表现稳定，集成投票也能进一步降低方差。这个比赛给我们的最大启发是：表格数据的清洗和手工特征要服务于业务逻辑，小而明确的特征往往比盲目堆模型更有用。
