# 预测歌曲每分钟节拍数

## 一、比赛简介

### 1.1 比赛背景

我们要预测的是一首歌曲每分钟的节拍数，也就是音乐里的 BPM。BPM 决定了音乐的快慢，对 DJ、健身、制作人等人都很重要。材料没有提供歌曲的原始音频，而是给出一张表格。训练集和测试集都有同一批特征，只有一个目标列 `BeatsPerMinute` 在训练集出现。特征的中文名称可以从代码里看到，例如 `RhythmScore`、`AudioLoudness`、`VocalContent`、`AcousticQuality`、`InstrumentalScore`、`LivePerformanceLikelihood`、`MoodScore`、`TrackDurationMs` 和 `Energy`。这些特征像是音频分析后得到的数值指标，而不是歌词或专辑信息。

这场比赛的背景是回归预测：给定音频生成的数值特征，猜一个连续的节拍数。材料中的 EDA 表明特征的分布各不相同。有些特征相对均匀，有些集中在低端还有大量零值。目标 `BeatsPerMinute` 的分布也左右不对称，中间集中在 118 到 120 左右，低端的异常值有 197 个，高端的异常值有 1944 个。特征之间的关系看起来并不是完全线性，例如 `AudioLoudness` 随 BPM 变化时有广泛的散点。我们不能假设每个特征都和 BPM 呈直线关系。

### 1.2 比赛目标

我们输入的是每首歌的数值和类别特征，输出是对应的 BPM 数值。训练集含有 `id`、`BeatsPerMinute` 和若干特征列；测试集只有 `id` 和特征列。我们要把 `BeatsPerMinute` 的预测值填到 `id` 所在的提交文件中。材料中的代码把目标命名为 `TARGET = "BeatsPerMinute"`，ID 列命名为 `ID_COL = "id"`。我们需要保证在训练和预测时使用完全相同的特征集合，不能在训练时用了测试集刚刚增加的特征，也不能把目标列误当作特征。

### 1.3 评价标准

比赛依据均方根误差 RMSE 评判。我们可以用 `mean_squared_error` 计算 MSE，再开方得到 RMSE。RMSE 的单位和 BPM 相同，因此误差可以直观理解成“平均偏离多少拍”。材料中的代码使用 5 折交叉验证来估计 RMSE，并把多折的平均值当成模型表现的稳定参考。RMSE 越小越好，说明预测值越接近真实值。与分类不同的地方是，这里不关注准确率，而是关注连续数值的距离。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取训练集和测试集，快速了解行数、列数和数据类型。探索的第一步是确认 `id` 是否唯一，以及特征是否包含常数列。常数列对模型没有贡献，还会浪费计算时间。我们其次检查数值缺失和离群值。材料中的一个方案计算每列的上下四分位间距，并把超出 1.5 倍 IQR 的值标记为异常值。对于 `RhythmScore`、`AudioLoudness` 和 `VocalContent`，发现不同程度的异常值比例。

```python
# 读取训练和测试文件，先不做清洗，确认基本信息
import pandas as pd
import numpy as np

TRAIN_PATH = "/kaggle/input/predicting-the-beats-per-minute-of-songs/train.csv"
TEST_PATH = "/kaggle/input/predicting-the-beats-per-minute-of-songs/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("train shape:", train.shape)
print("test shape:", test.shape)
print(train.columns.tolist())

# 检查 id 是否唯一，避免同一个样本在训练集中出现多次
print("train id unique:", train["id"].is_unique)
print("test id unique:", test["id"].is_unique)
```

我们接着看目标分布和每个特征的单变量分布。材料用箱线图和直方图组合来查看特征的偏态和异常值比例。例如 `AudioLoudness` 分布左偏，数值集中在 -12 到 -5，右端有尖峰；`VocalContent` 极度右偏，大约 70% 的样本集中在最低点。我们还要观察特征与目标的关系，材料使用 hex 联合图来展示 `RhythmScore`、`AudioLoudness` 等特征和 BPM 的二维密度。

```python
# 观察目标 BPM 的基本统计
TARGET = "BeatsPerMinute"
print(train[TARGET].describe())

# 逐列检查唯一值数量，找出常数特征准备删除
features = [c for c in train.columns if c not in ["id", TARGET]]
nunique = train[features].nunique()
const_feats = nunique[nunique <= 1].index.tolist()
print("constant features:", const_feats)

# 简单检查每列缺失比例
print(train[features].isna().mean().sort_values(ascending=False).head(10))
```

最后我们需要确定数值特征和类别特征的划分。材料里在某些特征上使用 `LabelEncoder`，说明特征中可能包含对象类型。我们还需要在训练集和测试集上联合拟合编码，避免训练集没有而测试集有的类别产生映射错误。

```python
# 区分数值特征和类别特征，便于后续分别处理
num_cols = train[features].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train[features].select_dtypes(include=["object"]).columns.tolist()
print("numeric:", num_cols)
print("categorical:", cat_cols)

# 目标基本统计帮助我们理解预测范围
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,4))
sns.histplot(train[TARGET], bins=50)
plt.title("BPM distribution")
plt.show()
```

### 2.2 数据清洗

我们首先去除训练集中可能存在的重复 ID。材料中的代码使用 `train.drop_duplicates(ID_COL, inplace=True)`，这一步防止同一个样本影响模型。我们其次删除常数特征，因为它们无法提供区分信息。再处理缺失值：对于数值列，用训练集的中位数填补测试集和训练集，保证填补值只来自训练集；对于类别列，用字符串 `"None"` 填充后，再进行编码。这种做法避免了把缺失当成一个新类别后在测试集中出现不一致。

```python
# 去重并删除常数列，准备进入建模
train.drop_duplicates("id", inplace=True)
features = [c for c in train.columns if c not in ["id", TARGET]]
nunique = train[features].nunique()
features = [f for f in features if train[f].nunique() > 1]

# 处理缺失值：先用中位数对数值列填补
num_cols = train[features].select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if train[col].isna().any() or test[col].isna().any():
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col] = test[col].fillna(med)
print("missing after fill:", train[features].isna().sum().sum())
```

我们接着做简单的特征工程。材料中使用了行级别统计：对数值特征按行计算均值和标准差，作为两个新特征。这一类特征可以捕捉一首歌整体特征的综合水平，例如能量和响度如果都在同一行偏高，行均值会上升。我们随后为类别列准备 LabelEncoder。材料先把所有值转成字符串，统一填充 `"None"`，然后联合训练和测试一起拟合，保证编码一致。

```python
# 行级别统计：捕捉同一首歌所有特征的整体水平
train["r_mean"] = train[num_cols].mean(axis=1)
train["r_std"]  = train[num_cols].std(axis=1).fillna(0)
test["r_mean"]  = test[num_cols].mean(axis=1)
test["r_std"]   = test[num_cols].std(axis=1).fillna(0)
features.extend(["r_mean", "r_std"])
print("new features added:", ["r_mean", "r_std"])

# 对类别特征做 LabelEncoder，保持训练和测试编码一致
from sklearn.preprocessing import LabelEncoder
cat_cols = train[features].select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    le = LabelEncoder()
    train[c] = train[c].astype(str).fillna("None")
    test[c] = test[c].astype(str).fillna("None")
    combined = pd.concat([train[c], test[c]], axis=0)
    le.fit(combined)
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])
```

材料中还有从 `TrackDurationMs` 派生出分钟数，以及创建 `Energy_Loudness` 和 `Vocal_Ratio` 的例子。我们可以在此基础上继续构造有意义的组合特征，但要避免使用目标信息泄露。所有变换必须先在训练集上计算，再应用到测试集。

```python
# 单位转换：毫秒转分钟，让数值规模更符合直觉
train["TrackDurationMin"] = train["TrackDurationMs"] / 60000
test["TrackDurationMin"] = test["TrackDurationMs"] / 60000
train.drop(columns=["TrackDurationMs"], inplace=True)
test.drop(columns=["TrackDurationMs"], inplace=True)

# 构造交互特征：能量和响度的乘积，声乐与器乐的相对比例
train["Energy_Loudness"] = train["Energy"] * train["AudioLoudness"]
train["Vocal_Ratio"] = train["VocalContent"] / (
    train["VocalContent"] + train["InstrumentalScore"] + 1e-6
)
test["Energy_Loudness"] = test["Energy"] * test["AudioLoudness"]
test["Vocal_Ratio"] = test["VocalContent"] / (
    test["VocalContent"] + test["InstrumentalScore"] + 1e-6
)
features = [c for c in train.columns if c not in ["id", TARGET]]
print("feature count:", len(features))
```

### 2.3 模型训练与超参数调优

我们首先选择 LightGBM 作为基线回归模型。材料中的代码使用 `lgb.train` 以及 `lgb.LGBMRegressor`，都设定了学习率在 0.02 左右，子采样和列采样在 0.6 到 0.8 之间。我们用 5 折交叉验证评估稳定性。材料中还展示了 CatBoost 配合 LightGBM 做模型融合，融合权重为 0.5 + 0.5。

```python
# 用 5 折交叉验证评估回归模型，先单模型
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np

SEED = 2025
NFOLDS = 5
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 64,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "seed": SEED,
    "verbosity": -1,
}

kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
test_preds = np.zeros(len(test))
```

我们其次在每折中用早期停止控制训练轮数，避免过拟合。材料使用 `lgb.early_stopping(stopping_rounds=100)`，并把训练集和验证集都交给模型监控。训练结束后再在验证折上计算 RMSE，累积得到总体 OOF RMSE。

```python
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
    X_tr, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
    y_tr, y_val = train.iloc[tr_idx][TARGET], train.iloc[val_idx][TARGET]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=0)
        ]
    )
    oof[val_idx] = model.predict(X_val)
    test_preds += model.predict(test[features]) / NFOLDS

    fold_rmse = rmse(y_val, oof[val_idx])
    print(f"fold {fold+1} RMSE: {fold_rmse:.5f}")

overall_rmse = rmse(train[TARGET], oof)
print("OOF RMSE:", overall_rmse)
```

材料还展示了贝叶斯优化来调参：使用 `bayes_opt` 的 `BayesianOptimization`，在 `n_estimators`、`learning_rate`、`num_leaves`、`max_depth`、`min_child_samples`、`feature_fraction`、`bagging_fraction` 等超参数范围内搜索，每次用 5 折交叉验证的平均 RMSE 作目标。我们可以理解这是在有限次数内，比网格搜索更智能地寻找好的参数组合。对于入门者，先用默认或简单参数跑通流程，再考虑优化。

```python
# 贝叶斯调参与网格搜索的不同思路，后者用固定范围
from bayes_opt import BayesianOptimization

def lgb_cv_rmse(n_estimators, learning_rate, num_leaves,
                max_depth, min_child_samples,
                feature_fraction, bagging_fraction):
    params = {
        "n_estimators": int(n_estimators),
        "learning_rate": learning_rate,
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "min_child_samples": int(min_child_samples),
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []
    for tr_idx, va_idx in cv.split(train):
        X_tr, X_va = train.iloc[tr_idx][features], train.iloc[va_idx][features]
        y_tr, y_va = train.iloc[tr_idx][TARGET], train.iloc[va_idx][TARGET]
        model = lgb.LGBMRegressor(**params, objective="regression", random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
    return -np.mean(rmses)

# 只展示参数范围定义，实际跑优化需要时间
pbounds = {
    "n_estimators": (300, 2000),
    "learning_rate": (0.01, 0.2),
    "num_leaves": (20, 150),
    "max_depth": (3, 15),
    "min_child_samples": (5, 100),
    "feature_fraction": (0.6, 1.0),
    "bagging_fraction": (0.6, 1.0),
}
```

最后我们可以使用材料里的模型融合思路。LightGBM 和 CatBoost 的 OOF 预测各占 0.5，融合后得到最终测试预测。这样做可以降低单一模型偏差，并在不使用测试集标签的情况下评估效果。

```python
# 融合两个模型，展示多模型组合的常见做法
from catboost import CatBoostRegressor

oof_lgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))
test_lgb = np.zeros(len(test))
test_cat = np.zeros(len(test))

cat_params = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "RMSE",
    "random_seed": SEED,
    "early_stopping_rounds": 200,
    "verbose": False,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
    X_tr, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
    y_tr, y_val = train.iloc[tr_idx][TARGET], train.iloc[val_idx][TARGET]

    model_lgb = lgb.train(lgb_params, lgb.Dataset(X_tr, y_tr),
                          num_boost_round=5000,
                          valid_sets=[lgb.Dataset(X_val, y_val)],
                          callbacks=[lgb.early_stopping(stopping_rounds=100),
                                     lgb.log_evaluation(period=0)])
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    test_lgb += model_lgb.predict(test[features]) / NFOLDS

    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    oof_cat[val_idx] = model_cat.predict(X_val)
    test_cat += model_cat.predict(test[features]) / NFOLDS

ensemble_oof = 0.5 * (oof_lgb + oof_cat)
ensemble_test = 0.5 * (test_lgb + test_cat)
print("ensemble OOF RMSE:", rmse(train[TARGET], ensemble_oof))
```

### 2.4 模型测试与指标报告

我们首先计算 OOF RMSE，评估模型在训练数据上的泛化能力。材料中计算了整体 OOF LGB RMSE、OOF Cat RMSE 以及融合后的 RMSE。整个流程中我们没有使用测试集的标签，所以所有指标都只能在训练集内部产生。我们还要把预测结果保存为提交文件，列名为 `id` 和 `BeatsPerMinute`。

```python
# 汇总指标并生成提交文件
submission = pd.DataFrame({
    "id": test["id"],
    TARGET: ensemble_test
})
submission.to_csv("submission.csv", index=False)
print(submission.head())

# 简单自检：确保没有缺失值，数值在合理范围内
print("submission shape:", submission.shape)
print("missing:", submission.isna().sum().sum())
print("value range:", submission[TARGET].min(), submission[TARGET].max())
```

我们其次检查 OOF 预测与真实值的散点分布，判断模型是否存在系统性偏差。材料没有给出具体数值，我们只能说明方法：用折线图或残差直方图观察预测误差分布。如果低 BPM 或高 BPM 的误差明显更大，可以考虑加权损失或分段建模。

```python
# 可视化残差，帮助发现偏差模式
residual = train[TARGET] - ensemble_oof
plt.figure(figsize=(8,4))
sns.histplot(residual, bins=50)
plt.title("Residual distribution")
plt.show()

# 与特征相关的残差分析
plt.figure(figsize=(8,4))
sns.scatterplot(x=train["Energy"], y=residual)
plt.title("Residual vs Energy")
plt.show()
```

最后我们确认提交文件格式：列名必须与 `sample_submission.csv` 保持一致，行数必须与测试集相同。材料中的代码在最后输出总耗时并打印提交文件保存路径。我们不为了追求数值而编造分数，指标只在代码中以计算方式呈现。

```python
# 最后的格式检查，确保不会因为格式错误丢分
sample = pd.read_csv("/kaggle/input/predicting-the-beats-per-minute-of-songs/sample_submission.csv")
assert list(submission.columns) == list(sample.columns)
assert len(submission) == len(sample)
print("format check passed")
```

## 三、总结

本文中我们完成了预测歌曲每分钟节拍数的完整流程。我们首先通过 EDA 了解特征分布和目标 BPM 的集中区间，确认了特征各不相同且存在离群值；其次进行数据清洗，去除重复和常数特征，用中位数填补缺失，并构造行均值、行标准差、分钟转换、能量响度乘积和声乐比例等新特征。然后我们采用 LightGBM 和 CatBoost 做 5 折交叉验证训练，用早期停止避免过拟合，并尝试贝叶斯优化寻找更好参数；最后我们计算 OOF RMSE 评估泛化，用 0.5 融合两个模型生成测试预测，并输出符合要求的提交文件。

这场比赛给我们的主要启示是，音频特征虽然数值化，但不同特征的尺度和分布差异很大，预处理和特征工程在回归任务中同样重要。交叉验证和模型融合能够提升稳定性，避免单一模型随折的波动。未来可以探索更精细的异常值处理、分段回归或加权损失，但在入门阶段，先把流程走通、把指标可复现地衡量出来，比盲目堆模型更重要。
