# 超导体临界温度的预测（Critical Temperature of Superconductors）

## 一、比赛简介

### 1.1 比赛背景

1911 年，物理学家卡末林·昂内斯发现：把汞冷却到零下 269℃（4 开尔文）左右，它的电阻会**突然变成零**。这就是**超导电性**。后来人们又发现，很多材料的电阻降到零的温度不止一个，而是各不相同，这个「正好变成超导的温度」就叫**临界温度**，记作 Tc，单位是开尔文（K）。

临界温度越高，超导材料越实用——理想情况下，我们希望有一种材料在常温下就能超导，那将彻底改变输电、磁悬浮、核磁共振这些行业。然而，已知的超导材料有上万种，它们的化学组成千差万别，临界温度从几开尔文到一百多开尔文不等。**能不能只看材料的化学成分，就预言它的临界温度？** 这正是这个比赛要回答的问题。

数据来自日本国家材料科学研究所（NIMS）维护的超导材料数据库，一共收录了 **21263** 种超导材料。对每种材料，除了完整的化学式，还给了 8 类基础化学性质的统计量——原子质量、电离能、原子半径、密度、熔化比热、电子亲和能、热导率、化合价，并且每类又细分出「绝对值」「平均值」「加权平均值」等多个维度。

难点有三个：

1. **目标分布严重偏斜**。材料里的直方图显示，大部分材料的临界温度集中在 0～25 K，但有一条向右拖得很长的尾巴，最远到 175 K 左右，75～100 K 之间还有一个小「鼓包」。这个分布不是正态的，直接回归很容易被长尾的少量高温材料牵着走。
2. **配方高维且稀疏**。化学式被拆成**一列一个元素**的计数表，一个简单化合物只在少数几列非零，剩下几百列全是 0。模型要从这堆稀疏计数里找出「哪个元素加得多会让 Tc 变高」。
3. **同一种材料反复出现**。同一化学式会在数据里出现很多次（只是统计口径不同），如果按普通随机切分，训练集和验证集里会各出现一批「长得几乎一样」的样本，验证分数虚高——这是隐藏的**数据泄漏**。

### 1.2 比赛目标

数据以两对文件给出：

- `train.csv`（**17010** 行）和 `test.csv`（**4253** 行）：每行是一种材料的 8 类化学性质统计量；
- `formula_train.csv` 和 `formula_test.csv`：同一批材料的化学式拆成的元素计数表，一列一个元素（`O`、`Cu`、`Ba`、`Fe`、`Bi`……），值是该元素在化学式里的原子个数；
- `material` 列：材料的化学式原文，它既是唯一标识，也充当「分组键」；
- 目标列 `critical_temp`：临界温度（K），在 `formula_train.csv` 里也有一份，拼接时要删掉一个。

我们要给测试集的每一条预测一个临界温度，提交文件包含 `index` 和 `critical_temp` 两列，`index` 就是测试集的行号。

**怎么读这个题目？** 两份文件描述的是同一批材料：`train.csv` 是「性质统计量」，`formula_train.csv` 是「成分构成」。真正能打的信息是把两者**横着拼起来**——用化合物的成分去解释它的临界温度。这也是材料里三份方案共同的第一步。

### 1.3 评价标准

这个比赛用 **RMSE**（Root Mean Squared Error，均方根误差）打分。材料里的证据很明确：训练目标设成 `metric='rmse'`，验证时用 `mean_squared_error(..., squared=False)` 计算，作者还在文件夹里标注了榜单分数 **83.19**。

RMSE 的算法是：把每条样本的「预测值减真实值」取平方，这样正负误差不会互相抵消，还能放大那些猜得很离谱的样本；对全部样本取平均，最后开平方根回到和温度相同的单位（开尔文）：

```
RMSE = sqrt( (1/n) · Σ (y_pred - y_true)² )
```

**这个指标越低越好**，单位就是 K，满分是 0。直观地感受一下尺度：比赛的中位数方案分数在 100～115 左右（材料里提到，即便用了 Box-Cox 变换也没能明显改善），而这份方案做到了 **83.19**。

---

## 二、解题过程

我们用一条「从粗到精」的路线：先拼表、做 EDA，再靠手工特征工程把化学式压成几条能用的统计量，然后用 **LightGBM + Optuna 调参 + 分组交叉验证** 作为主力，XGBoost 作为对照。我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

**第一步：读入两对文件，按行拼接。** 性质表和配方表行数相同、顺序一一对应，可以直接按左右拼：

```python
import pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pref = "/kaggle/input/critical-temperature/"
train = pd.read_csv(pref + "train.csv")
formula_train = pd.read_csv(pref + "formula_train.csv")

print(f"Train dataset shape: {train.shape}")                # (17010, ...)
print(f"Train_formula dataset shape: {formula_train.shape}")

# formula_train 里也有 critical_temp，删掉避免拼出两列同名的目标
formula_train = formula_train.drop(columns=['critical_temp'])
train_full = pd.concat([train, formula_train], axis=1)

# material（化学式原文）先留着当分组键，建模时才丢
# train_full.drop(columns=['material'], inplace=True)
print(f"Full Train dataset shape: {train_full.shape}")
```

这里有一个细节：**`drop(columns=['material'])` 先不执行**。原因留在下面模型一节解释，它会作为分组键参与交叉验证。

**第二步：数缺失、看分布。** 材料先确认了最基础的两件事——有没有缺失，目标长什么样：

```python
# 先摸清底细：有缺失才需要填补，这里四张表都是 0 缺失
print("Number of missing values in train datasets:")
print(train.isna().sum())
print(formula_train.isna().sum())

# 目标分布直方图
plt.figure(figsize=(6, 4))
sns.histplot(train_full['critical_temp'], kde=True, bins=30)
plt.title("Распределение таргета")
plt.show()
```

结论有两层：

1. **数据干净，不需要填补**。四张表都没有缺失值，这是标准比赛数据；真正的预处理重点不在「补洞」，而在「怎么切分」和「怎么造特征」。
2. **目标严重右偏**。绝大多数临界温度在 0～25 K，右尾拉长到约 175 K，75～100 K 有个小鼓包。材料里明确建议这类分布不要指望「变正态再打」——作者试过对数变换和 Box-Cox，分数并没有变好，所以干脆直接对原始目标回归。

**第三步：算相关性和异常值。** 两个动作能立刻告诉你「哪些特征值得看」：

```python
# 只看数值列，算每个特征与目标的皮尔逊相关系数，按绝对值排前 10
correlations = train_props.corr()["critical_temp"].abs() \
    .sort_values(ascending=False)[1:11]
sns.barplot(x=correlations.values, y=correlations.index)

# 异常值普查：用四分位距法则逐列统计超出 1.5 倍 IQR 的样本占比
for col in numerical_features:
    Q1 = train_full[col].quantile(0.25)
    Q3 = train_full[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = train_full[(train_full[col] < Q1 - 1.5 * IQR) |
                          (train_full[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: 异常占比 {len(outliers)/len(train_full)*100:.2f}%")
```

异常值几乎每列都有。材料在这里做了一个很关键的决定：**不删除异常样本**。理由是「剔除异常特征可能导致信息丢失」——在超导数据里，极端值往往对应的是特殊组分材料，恰恰是最有价值的样本。正确的做法是选一个能扛异常值的模型（树模型天然比线性模型抗造），而不是把异常值删掉。

### 2.2 数据清洗

清洗在这里不是删缺失、填空缺，而是**把化学式变成模型能用的数字**。配方表里一列一个元素，加起来几百列，大部分是 0，直接丢给树模型不是不行，但效率太低。材料里最亮眼的操作是做**配方统计特征**：

```python
def engineer_features(props_df, formula_df):
    # 去掉非特征列，注意 errors='ignore' 让训练/测试可以共用这个函数
    f_clean = formula_df.drop(columns=["material", "critical_temp"], errors="ignore")
    p_clean = props_df.drop(columns=["critical_temp"], errors="ignore")

    total_atoms = f_clean.sum(axis=1)                 # 一种材料一共含多少个原子
    unique_elements = (f_clean > 0).sum(axis=1)       # 由几种元素组成

    epsilon = 1e-9                                    # 防除零
    engineered = pd.DataFrame({
        "total_atoms": total_atoms,
        "unique_elements": unique_elements,
        # 关键元素占比：有些元素的相对含量强烈影响临界温度
        "O_ratio": f_clean.get("O", 0)  / (total_atoms + epsilon),
        "Cu_ratio": f_clean.get("Cu", 0) / (total_atoms + epsilon),
        "Ba_ratio": f_clean.get("Ba", 0) / (total_atoms + epsilon),
        "Fe_ratio": f_clean.get("Fe", 0) / (total_atoms + epsilon),
        "Bi_ratio": f_clean.get("Bi", 0) / (total_atoms + epsilon),
    })

    # 原始配方计数 + 属性统计量 + 新造的统计特征，三块拼在一起
    return pd.concat([p_clean, f_clean, engineered], axis=1)

X_train = engineer_features(train_props, train_formula)
X_test  = engineer_features(test_props, test_formula)
```

新造的 7 个特征有一个共同的动机：**绝对原子数无法跨物质比较，占比可以**。「甲材料含 4 个氧原子」和「乙材料含 4 个氧原子」，在几百列的计数表里是一样的；但「氧原子占了原子总数的一半」和「只占四分之一」，才是真正能区分超导性能的相对量。为什么专挑 O、Cu、Ba、Fe、Bi 这五种？因为铜氧化物（Cu–O）和铁基超导体（Fe）是已知的最高温超导家族，材料作者凭领域知识把注意力集中在这些主力元素上。把 5 个占比和化学式原始计数一起放进去，等于既给了树模型「重写」的答案，也给了它自己重新组合的空间。

清洗阶段我们特意没做两件事，都值得说明：

- **没有做归一化/标准化**。树模型按特征值切分，不依赖尺度，标准化只会浪费时间。
- **没有砍掉长尾元素列**。树模型天然能忽略从不被选中的列；真正需要防的是「这条铁基材料在训练集出现、同款在验证集也出现」——这事交给分组交叉验证解决，而不是删列。

### 2.3 模型训练与超参数调优

**第一步：先跑两个基线，确认方向。** 材料的第一份方案按 7:3 切分，比较了线性回归和梯度提升：

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 线性回归并不适用：超导关系高度非线性
model = LinearRegression().fit(X_train, y_train)
print('R2 score:', r2_score(y_test, model.predict(X_test)))

# 梯度提升显著更好——材料原话：“Градиентный бустинг работает значительно лучше”
model = GradientBoostingRegressor().fit(X_train, y_train)
print('R2 score:', r2_score(y_test, model.predict(X_test)))
```

两次 R² 的差距让结论一目了然：**用树集成，别用线性模型**。考虑到特征是稀疏散点、目标和特征之间是强非线性关系（氢含量和铜含量的影响互相纠缠），线性回归把关系强行当直线拟，自然处处失手。这就是为什么后面两份方案都改用 XGBoost / LightGBM。

**第二步：把「切分」这个环节做到位。** 这是整份材料里最重要的一课。同一化学式的材料会在数据里出现多次，如果普通随机切分，验证集里会出现一批训练集的「近亲」，模型顺着化学式抄答案，分数虚高。解决方法是**按 material 分组切**：

```python
from sklearn.model_selection import GroupKFold, train_test_split
import pandas as pd

RANDOM_STATE = 42
y_full = train_full["critical_temp"]
# material 作为分组键：同一化学式的所有行必须待在同一个组里
groups_full = train_full["material"]

# 1) 分层 hold-out：把目标按十分位数分箱，再按箱分层切分
#    保证验证集覆盖到目标的所有取值水平，避免长尾全被切进训练集
y_binned = pd.qcut(y_full, q=10, duplicates='drop')
X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
    X_full, y_full, groups_full,
    test_size=0.2, random_state=RANDOM_STATE, stratify=y_binned)

# 2) GroupKFold：5 折，同一组的样本绝不会同时出现在训练折和验证折
gkf = GroupKFold(n_splits=5)
```

- **`stratify=y_binned` 为什么做**：目标分布偏斜，随机切分有可能把 75～100 K 那批「鼓包」样本全分到训练集，验证集里只有 0～25 K 的常见样本。分层保证验证集里每一档临界温度都有代表，RMSE 才稳定可信。
- **`GroupKFold` 为什么做**：它是防泄漏的具体手段。材料里用了一个很形象的比喻——如果不分组，模型在验证集上看到了和训练集「化学式几乎一样」的材料，RMSE 会因为「抄答案」而造假地低，上线立刻被打回原形。**分组之后，验证分数才接近 Kaggle 榜单上的真实分数。**

**第三步：用 Optuna 给 LightGBM 调参。** 调参范围覆盖学习率、树结构、采样比例、正则项，每次 trial 用 GroupKFold 的平均 RMSE 当目标：

```python
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

def objective(trial):
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        # 每个超参数给出一个合理范围，Optuna 用贝叶斯优化在此搜索
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves":     trial.suggest_int("num_leaves", 16, 128),
        "max_depth":      trial.suggest_int("max_depth", 3, 12),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 1, 5),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "reg_alpha":   trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":  trial.suggest_float("reg_lambda", 0.0, 1.0),
        "n_estimators": 2000,          # 上限给足，实际轮数交给早停决定
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    rmses = []
    for tr_idx, va_idx in gkf.split(X_train, y_train, groups_train):
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx],
                  eval_set=[(X_train.iloc[va_idx], y_train.iloc[va_idx])],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(stopping_rounds=100)])
        rmses.append(mean_squared_error(y_train.iloc[va_idx],
                                        model.predict(X_train.iloc[va_idx]),
                                        squared=False))
    return np.mean(rmses)              # Optuna 的目标：5 折平均 RMSE 越小越好

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)  # 材料里以 5 次 trial 演示，正式解跑 25 次
```

这里值得讲清楚两个「多此一举」的设计，它们都是防止调参本身过拟合：

- **每个 trial 额外训一个 holdout 模型**。Optuna 每试一组参数，除了在 5 折上算平均 RMSE，还会把参数在完整训练集上训一遍、在分层 hold-out 上再算一次 RMSE。如果出现「折上 RMSE 很低、hold-out 却明显变差」，就说明这组参数**在调参时就过拟合了**。这个「旁路监督」不直接参与选择，但能让你亲眼看到 Optuna 有没有走偏。
- **早停 `stopping_rounds=100`**。`max_depth` 和 `num_leaves` 大了很容易过拟合，早停让模型自己决定在多少棵树时停——验证集连续 100 轮不改善就停，树的数量因此成为由参数推断出来的量，而不是拍脑袋定的量。

**第四步：对照组 XGBoost。** 材料的另一份方案用 XGBoost 走了同样的 5 折流程，参数由 Optuna 给出：

```python
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

xgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.008508649256010067,   # 很小的学习率，靠大量树慢慢学
    "max_depth": 13,
    "subsample": 0.8097270804253619,          # 行采样约 0.81
    "colsample_bytree": 0.7224382440244262,   # 列采样约 0.72
    "min_child_weight": 4,
    "gamma": 1.2120600998883369,              # 分裂增益的阈值，越大越保守
    "tree_method": "hist", "device": "cuda",
    "early_stopping_rounds": 100, "random_state": 42, "n_jobs": -1,
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X_train))
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    model = XGBRegressor(**xgb_params)
    model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx],
              eval_set=[(X_train.iloc[va_idx], y_train.iloc[va_idx])], verbose=250)
    oof_predictions[va_idx] = model.predict(X_train.iloc[va_idx])
    test_preds += model.predict(X_test) / kf.n_splits   # 5 折预测平均
```

这里有几个值得记住的设定：`learning_rate` 只有 0.0085，配合 `n_estimators=2000` 和早停，是经典的「小步慢走」策略；`subsample` 和 `colsample_bytree` 都在 0.7 以上，属于温和的行列采样；`gamma=1.21` 让分裂更保守。这组参数的共同气质是**偏保守**——在数据规模不大、目标又极端偏斜的任务上，稳比激进更好。

**第五步：最终模型与预测。** 选完参数后，把模型在整个训练集上重新训一遍，并顺手把预测值下限裁到 0——临界温度不可能是负数：

```python
best_params = study.best_params
best_params.update({"boosting_type": "gbdt", "objective": "regression",
                    "metric": "rmse", "n_estimators": 2000,
                    "random_state": RANDOM_STATE, "verbose": -1})

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_full, y_full, eval_set=[(X_val, y_val)], eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=100)])

y_test_pred = final_model.predict(test_full.drop(columns=["material"]).copy())
y_test_pred = np.clip(y_test_pred, 0, None)   # 负温度在物理上没有意义，直接裁掉

submission = pd.DataFrame({"index": test_full.index, "critical_temp": y_test_pred})
submission.to_csv("submission.csv", index=False)
```

### 2.4 模型测试与指标报告

**验证指标。** 训练收尾时，材料在分层 hold-out 上同时打印 MSE 和 RMSE，并记录下最初几组基线模型的分数是 **100～115**（即便用了 Box-Cox 变换），而最终方案的榜单分数是 **83.19**。这 20 个 K 的差距，主要来自两处——**手工配方特征**（让树少做很多无用分裂）和**分组交叉验证**（切分不再泄漏，调参有了统一、可信的度量）。

**诊断图。** 材料画了两张图来检查预测质量：

```python
# 左边：真实值 vs 预测值散点，越贴对角线越好
plt.scatter(y, oof_predictions, alpha=0.1, color="teal")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)

# 右边：残差（真实-预测）分布，越接近 0 的尖峰越好
residuals = y - oof_predictions
sns.histplot(residuals, bins=100, kde=True, color="darkorange")
```

如果散点图在高温端明显离开对角线，说明模型在长尾上系统性低估——这正是目标右偏的典型症状；如果残差分布围绕 0 对称且集中，说明误差没有明显的系统性偏向。这两张图是判断「偏差大还是方差大」的快速入口。

**特征重要性。** LightGBM 直接用 `feature_importances_` 给出排名，材料画了 Top-10 条形图：

```python
importances = pd.Series(final_model.feature_importances_, index=X_full.columns)
top_20 = importances.sort_values(ascending=False).head(20)
sns.barplot(x=top_20.values, y=top_20.index)
```

看这个排名能验证一个判断：手工造的**元素占比特征**（`O_ratio`、`Cu_ratio`、`Ba_ratio` 等）和**原子总数**这类配方统计量，如果出现在前列，说明「把计数变成占比」这条路走对了；如果排在前面全是原始性质列，就要回去检查占比是不是被除零之类的错误污染了。

---

## 三、总结

本文中我们完成了超导体临界温度预测的完整流程。我们首先用 EDA 摸清了数据：性质表和化学式表各自成对，训练集 17010 行、测试集 4253 行，四张表都没有缺失；目标的分布严重右偏，大部分在 0～25 K、右尾拖到约 175 K，并且几乎每列都存在异常值——我们决定不删异常，而是选抗异常值的树模型。其次我们做了关键的清洗与特征工程：把两份文件按行拼接、只保留一份 `critical_temp`，并手工造了 `total_atoms`、`unique_elements` 以及 O/Cu/Ba/Fe/Bi 五种元素占比特征，把稀疏的化学式计数压成能横向比较的相对量。接着我们跑了三个层次的模型：先用线性回归和梯度提升做基线，确认「树集成显著优于线性模型」；再引入 Optuna 给 LightGBM 调参，过程中用目标十分位分层切分验证集、用 `GroupKFold` 按化学式分组防止泄漏，每组参数还会在独立 hold-out 上旁路监督过拟合；同时以 XGBoost（学习率 0.0085、2000 棵树上限、早停）做 5 折对照。最后我们在验证集上报告 RMSE，把预测值下限裁到 0 后按 `index` + `critical_temp` 生成提交文件，最终榜单分数 83.19，而基线水平为 100～115。

这个比赛给我们最大的启发是：**「怎么切数据」有时候比「用什么模型」更重要。** 三份方案用的模型不同，但真正拉开分数差距的，是 `GroupKFold` 挡住了同化学式的泄漏、分层切分让验证集覆盖了每个温度档。调参器的每个决策都是看着 CV 分数做的，如果这个分数本身是假的，那整个调参过程就是在向错误的目标前进。另一个启发是，**领域知识能直接变成特征**：用化学家都知道的「铜氧化物和铁基是高温超导主力」选中五个主力元素算占比，比让模型从几百列稀疏计数里自己悟要省事得多——这也是为什么在这个比赛里，好的特征工程比模型结构更值钱。