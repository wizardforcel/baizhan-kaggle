# 五款游戏销量预测（5 of the Game Sales Competition）

## 一、比赛简介

### 1.1 比赛背景

这是一个典型的「教学向」回归比赛：组织者把某款游戏销量数据做成 `train.csv` 和 `test_private_shifted.csv` 两份文件，让学生用它练习完整的机器学习流程。从 `test_private_shifted` 这个名字就能看出，「shifted」说明测试集是**故意移动（偏移）过的**——它的特征分布和训练集不完全一致。为什么老师要故意这么设计？因为在真实世界里，「未来的数据」几乎永远不会和「训练时看到的数据」长得一模一样：厂商每年发布新作，游戏发行年份会整体往后移，媒体评分习惯也会随年份变化。一个只在旧分布上训练得很好的模型，遇到新分布往往立刻失准。这就是现实里机器学习模型「上线即退化」的典型原因，比赛用考题的形式把这件事摆到了台面上。

比赛对应的模板按照五个模块组织，我们这份材料是最难的终章（最后一个 Block，标注 DOUBLE POINTS）。它给定了四个必须完成的任务：第一，用 KS 检验找出发生漂移的特征；第二，用一小部分「新数据」重新微调模型；第三，把三个模型做堆叠（stacking）；第四，用 SHAP 解释模型、找出最重要的三个特征。

### 1.2 比赛目标

训练文件 `train.csv` 里我们明确知道存在两列数值特征：

- `release_year`：游戏发行的年份；
- `critic_score`：媒体给游戏打的评分。

我们要预测的目标是某个连续数值（模板里用通用变量名 `y` 表示，最可能是销量这类回归目标，列名模板没有写死）。测试文件 `test_private_shifted.csv` 结构相同，我们要为它补上预测值。模板还提示我们其他列可能有缺失值（它提前 import 了 `SimpleImputer`），这些细节放在清洗环节处理。

### 1.3 评价标准

比赛用 **RMSE（均方根误差）** 衡量预测好坏：

```text
RMSE = sqrt( (1 / N) * Σ (真实值 - 预测值)^2 )
```

RMSE 越低越好。它把误差先平方再平均再开根号，因此**特别惩罚「错得离谱」的预测**——一次大错会拖累整体分数，这正好配合本次比赛「先想办法把分布漂移纠正过来、再追求低误差」的意图。

> 小提示：因为误差被平方了，RMSE 的单位虽然和销量一样大，但数量级感受会比 MAE（平均绝对误差）「重」。看到分数时先想清楚它是开过根号的，别被吓到。

---

## 二、解题过程

材料本身是一个需要补全代码的课程模板：每个任务给一段「学生写的示例」作为起点，最后的结果字段（RMSE、TOP-3 特征）留了空位等学生填。所以这篇 writeup 记录的是**完整的工作流和必须报告的指标**，而不是编造的分数。我们首先做 EDA 找出漂移特征，其次做清洗与标准化，接着完成「微调」和「堆叠」两个建模任务，最后用 SHAP 解释模型。

### 2.1 数据探索（EDA）

第一步是把两份数据读进来，确认行数，再用 **KS 检验**逐一比较训练集和测试集的每个数值特征分布。KS 检验的全名是 Kolmogorov–Smirnov 检验，它回答「两堆数的分布是不是同一个」：输出一个 p 值，p 越小说明两堆数越不像、越可能是漂移了。课程里约定 p < 0.05 就算显著漂移：

```python
import pandas as pd
from scipy.stats import ks_2samp

# 读训练集和测试集（测试集被做过偏移，分布可能和训练集不一样）
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test_private_shifted.csv')
print('Train:', len(df_train), 'Test:', len(df_test))

# 对每个数值特征做 KS 检验：比较训练集、测试集的分布
numeric_cols = ['release_year', 'critic_score']
drift_detected = []
for col in numeric_cols:
    stat, p = ks_2samp(df_train[col].dropna(), df_test[col].dropna())
    # p 越小，说明两个分布的差异越不可能是碰巧，越是真漂移
    if p < 0.05:
        drift_detected.append(col)
    print(f'{col}: KS={stat:.4f}, p={p:.4f}')

print('发生漂移的特征:', drift_detected)
```

这个检验的结论直接影响后面怎么做：**哪些特征漂移了，模型在测试集上就越不可信，需要在「适应新数据」的环节重点照顾。** 模板给出的答案里，`release_year` 已被标成漂移特征（`block5_drift_detected = "release_year"`），说明时间类特征最容易随数据更新而整体偏移。

我们可以把 KS 检验想象成两张「累计分布图」的比对：把训练集某个特征的值从小到大排好、画成一条从 0 走到 1 的阶梯曲线，测试集也画一条，然后看这两条曲线在竖直方向上**离得最远的地方差了多少**。差得越远，统计量 `stat` 越大；再结合两边的样本量，就能算出「如果两堆数其实来自同一个分布，出现这么大差距的概率有多大」，这个概率就是 p 值。所以 p < 0.05 并不是「差 5%」，而是「纯属巧合的可能性不到 5%」。理解了这一点，我们就知道为什么 KS 检验特别适合做漂移预警——它不关心数值的绝对大小，只关心两个集合的「形状」像不像。

### 2.2 数据清洗

数据清洗这一步，模板替我们把工具都准备好了：`SimpleImputer` 用来填缺失值，`StandardScaler` 用来标准化。对这道题来说这两件事都值得做，原因是：`critic_score` 可能有缺失（不是每款游戏都被媒体评分）；而 `release_year` 是上千量级的大数、`critic_score` 是 0~100 的小数，量纲差着数量级，先把它们标准化到「均值 0、方差 1」会让后面所有模型都更稳定：

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 先用中位数填缺失值（中位数比均值抗异常值），再做标准化。
# 用流水线打包，保证后面换模型、做交叉验证时不会漏掉预处理
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
```

注意清洗顺序：**先填空、再标准化**。如果先标准化，缺失值会把均值/方差算歪；反过来则没有这个问题。这也是为什么洗数据要按「缺失 → 量纲」的顺序来。

### 2.3 模型训练与超参数调优

数据准备好了，进入两个建模任务。

**任务二：模型适应新数据。** 思路是把训练数据里的「新数据」单独圈 20% 出来，只让模型在这部分上重新拟合（适应漂移后的分布），然后在剩余的 80% 上评估 RMSE，看到底适应之后准不准：

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 特征和标签（这里 X、y 来自前面的特征工程）
X = df_train.drop('target', axis=1)
y = df_train['target']

# 用 20% 的样本当"新数据"训练，剩下 80% 验证模型适应得好不好
X_fit, X_eval, y_fit, y_eval = train_test_split(
    X, y, test_size=0.8, random_state=42)

# 在 20% 新数据上微调一个随机森林回归
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(preprocessor.fit_transform(X_fit), y_fit)

# 在 80% 的验证数据上算 RMSE
pred = model.predict(preprocessor.transform(X_eval))
rmse_adapted = np.sqrt(mean_squared_error(y_eval, pred))
print(f'适应后的 RMSE: {rmse_adapted:.3f}')
```

**任务三：堆叠三个模型。** 堆叠（stacking）的思想是：单个模型各有盲区，不同模型犯的错往往不重合。于是我们先让 XGBoost、随机森林、梯度提升三个模型各自预测，然后**把它们的预测值当新特征**，再训练一个小模型（元模型）学习「怎么组合这些预测效果最好」。为了不让任何一份数据既被基模型看过又被元模型学过，我们用 3 折：每折只把「这一折的预测」喂给元模型，防止信息泄漏：

```python
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

kf = KFold(n_splits=3, shuffle=True, random_state=42)
meta_features = np.zeros((len(X), 3))

# 3 折交叉：每一折都用"没见过该折"的模型来预测，保证元特征不泄漏
for tr_idx, va_idx in kf.split(X):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr = y.iloc[tr_idx]
    # 三个基模型，各自只训练这一折的训练部分
    xgb_m = xgb.XGBRegressor(n_estimators=50, random_state=42).fit(X_tr, y_tr)
    rf_m = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_tr, y_tr)
    gb_m = GradientBoostingRegressor(n_estimators=50, random_state=42).fit(X_tr, y_tr)
    # 把这一折的验证预测填进元特征表
    meta_features[va_idx, 0] = xgb_m.predict(X_va)
    meta_features[va_idx, 1] = rf_m.predict(X_va)
    meta_features[va_idx, 2] = gb_m.predict(X_va)

# 元模型：用线性回归去学"三个预测各该占多少权重"
meta_model = LinearRegression().fit(meta_features, y)
```

到这里，推理时的完整链路就是：**数据 → 标准化 → 三个基模型各出一个预测 → 线性组合 → 最终预测。**

为什么这样能比单模型更好？因为三个基模型学出来的「偏见」方向不一样：XGBoost 用一层层提升去拟合残差，随机森林靠多棵树投票来降方差，梯度提升则更关注每一步该补多少。当它们在同一份数据上各自犯下**不完全重合**的错误时，元模型的线性组合就像「三个臭皮匠」——某个人偏高了，另一个人可能正好偏低，平均之后误差互相抵消。如果三个模型长得一模一样（比如都是同一族的树、同样的参数、同样的训练子集），那堆叠就没意义了；所以选模型时要刻意挑「思路不同」的组合。另外，元模型用最简单的线性回归就够了——它只需要学「三份预测各占多少权重」，若再上复杂模型，反而容易把基模型的噪声也学进去。

### 2.4 模型测试与指标报告

最后一步是解释模型、给出结论。我们训练一个单独的 XGBoost 作为「最佳模型」，用 **SHAP** 分析它。SHAP 的做法是把模型的预测**分摊回每个特征头上**：每个特征对本次预测「贡献了多少」就是一个 SHAP 值，把所有样本的贡献绝对值取平均，就能排出「哪些特征真正在影响预测」。模板要求给出 TOP-3：

```python
import shap

# 训练一个专门用来解释的 XGBoost 模型
best_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
best_model.fit(preprocessor.transform(X), y)

# 树模型用 TreeExplainer：X[:100] 表示只取前 100 行样本算 SHAP，省时间
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(preprocessor.transform(X[:100]))

# 每个特征贡献的绝对值取平均，贡献越大排在越前面
shap_importance = np.abs(shap_values).mean(axis=0)
top_indices = np.argsort(shap_importance)[-3:][::-1]
top3 = [X.columns[i] for i in top_indices]
print('TOP-3 特征:', top3)
```

报告阶段，我们把三件事放在一起看：任务二的 `rmse_adapted` 是多少、任务三 `block5_stacking_rmse` 是多少、任务四的 TOP-3 特征是什么。模板把这些字段都留成了空位（比如初始 `block5_stacking_rmse = 0.0`），需要我们真正跑通代码后填进去。对我们的交付来说，重点不是背一个分数，而是确认流程闭合：**漂移特征被找出 → 模型在 20% 新数据上适应 → 三个模型堆叠出最终预测 → SHAP 告诉我们预测主要靠哪些特征。** 如果 TOP-3 里出现了 `release_year`，正好和 2.1 里的 KS 检验结论互相印证——这个特征漂移最严重，同时也最重要，那后续就该优先处理它的口径。

---

## 三、总结

本文中我们按课程模板完成了「五款游戏销量预测」终章的四个任务。我们首先做 EDA，用 KS 检验比较训练集与测试集的分布，确认 `release_year` 存在显著漂移；其次做数据清洗，按「中位数填空 → 标准化」的顺序构造预处理流水线，解决缺失值和量纲问题；接着完成两个建模任务——用 20% 新数据微调随机森林、以及用 3 折交叉不泄漏的方式堆叠 XGBoost/随机森林/梯度提升三个模型并用线性回归做元模型；最后用 SHAP 找出 TOP-3 重要特征，并把 RMSE 作为贯穿全程的衡量标准。

这份材料给我们的启发是：**真实比赛里「特征分布漂移」不是可以忽略的细节，而是决定模型上线后还灵不灵的关键**。先检验漂移、再用少量新数据适应、最后用可解释性工具验证特征是否可靠，这套顺序本身就是一套可以复用的工程方法论。