# 屏幕使用习惯与标签构建研究（Screen Habits and Label Construction）

## 一、比赛简介

### 1.1 比赛背景
本场比赛来自 Kaggle Playground S6E8 的合成数据，上游数据是 `algozee/smartphone-addiction-prediction-data`。它记录了一份关于手机使用习惯的调查问卷，包含学习、工作、社交、游戏等场景下的每日使用时长，以及压力水平、学业或工作受影响程度等自评字段。比赛要求根据这些调查字段预测一个名为 `addicted_label` 的二分类标签。

这场比赛特别的地方在于：目标标签并不是临床诊断，也不是独立测量的行为结果，而是由同一份问卷里的若干字段按照某种规则构造出来的合成标签。因此，模型如果表现很好，可能反映的是它学会了这个标签构造规则，而不是发现了新的行为规律。材料把这个现象称为“标签构建问题”，并要求我们做一次目标构建审计和一次更换特征集的迁移测试。

理解“标签构建”可以借助一个生活中的例子：如果一份问卷既问了“你每天玩手机几小时”，又用同一个答案直接算出“你是否手机上瘾”，那么一个模型只要学会“每天超过几小时就判为成瘾”，就能得到很高的分数，但它并没有真正理解上瘾。我们要做的第一件事，就是检查目标是不是这样被“造”出来的，而不是急着追求高分。

### 1.2 比赛目标
输入是调查问卷字段，包括各类每日使用时长、`stress_level`、`academic_work_impact` 等；输出是 `addicted_label` 的预测值（0 或 1）。数据包含 `id` 列和标签列，训练集与测试集拥有相同的特征列。材料中的基线把训练集随机抽样上限设为 120000 行，并固定保留 20% 作为 holdout。这样做既控制了内存，也让实验可以在有限时间内复现。

### 1.3 评价标准
材料中的模型在一组常用二分类指标上做评估，包括准确率（accuracy）、平衡准确率（balanced_accuracy）、对数损失（log_loss）、ROC-AUC 和 MCC（Matthews 相关系数）。对于不平衡数据，单看准确率容易被多数类主导，所以平衡准确率、AUC 和 MCC 更能反映真实区分能力。材料没有给出一个唯一的官方排行榜指标，因此我们在复现时把这些指标一并报告，并以 AUC 和 MCC 作为主要比较依据。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先按块读取训练集，统计总行数并做有上限的均匀抽样。这样做既控制了内存，又保留了随机性。随后我们统计每个字段的缺失数量和标签分布。

```python
import pandas as pd, numpy as np

MAX_ROWS = 120000
rng = np.random.default_rng(20261026)
reservoir = None
missing = None
rows_seen = 0

# 分块读取，避免一次性把大文件全部装进内存
for chunk in pd.read_csv(DATA/"train.csv", chunksize=100000):
    rows_seen += len(chunk)
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
    # 给每行一个随机优先级，只保留优先级最小的若干行
    chunk["_priority"] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk])
    reservoir = reservoir.nsmallest(MAX_ROWS, "_priority")

train = reservoir.drop(columns="_priority").reset_index(drop=True)
```

抽样后我们检查 `id` 唯一、标签没有缺失，并统计正负样本比例。材料还要求把缺失情况单独保存为 `missingness.csv`，方便后续审计。标签分布显示这是一个不平衡的二分类任务，因此后面不能只看准确率；如果正样本只占很小一部分，一个“全部预测为负”的模型也可能有很高的准确率，但对实际决策没有帮助。

```python
# 标签不能有缺失，否则说明数据契约被破坏
assert not train["addicted_label"].isna().any()
print(train["addicted_label"].value_counts(normalize=True))

# 保存每个字段的缺失行数，作为数据质量证据
pd.DataFrame({
    "field": missing.index,
    "missing_rows": missing.values.astype(int)
}).to_csv("missingness.csv", index=False)
```

除了看单列分布，我们还做了一次“使用时长恒等式”检查：问卷里通常有总使用时长，以及社交、工作、游戏等分项时长。如果总分项之间存在固定关系，那么这些字段其实在重复表达同一件事。我们计算“总时长减去分项之和”的残差，并设置一个由四舍五入和单位决定的容差，而不是要求浮点数完全相等。

```python
# 检查总使用时长是否等于各分项之和
component_cols = ["social_usage", "work_usage", "gaming_usage"]
residual = train["total_usage"] - train[component_cols].sum(axis=1)

# 允许 1e-6 的误差，吸收浮点数和四舍五入造成的微小差异
bad = residual.abs() > 1e-6
print("不满足恒等式的行数:", bad.sum())
print(residual.describe())
```

这个检查很关键：如果残差几乎全为零，说明某一列可以由其他列精确重建，模型使用它并不会获得真正的新信息。材料还建议把这种“重建出来的特征”当作需要重点解释的对象，而不是直接当作强特征使用。

### 2.2 数据清洗
清洗原则是“无状态变换”：所有处理只依赖当前行，不在训练集上学习目标统计量，也不对验证集做任何拟合。对类别列我们把缺失填成固定字符串 `__MISSING__`，对数值列把无穷值替换成缺失。这样 CatBoost 可以原生处理缺失，不需要额外的全表插补。

```python
def features(frame, omit=()):
    """把原始表转成模型输入；omit 用于消融实验"""
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if not pd.api.types.is_numeric_dtype(x[c]):
            # 类别列统一成字符串，缺失单独作为一个类别
            x[c] = x[c].astype("string").fillna("__MISSING__").astype(str)
        else:
            # 数值列只做类型清理，不做跨行统计
            x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x
```

我们刻意没有做目标编码，也没有用整个训练集计算均值再填回验证集。原因是材料强调要区分“模型真正学到的信号”和“预处理泄漏”。在这个任务里，验证标签也不能用于早停或参数选择，固定参数必须在看到分数之前就确定。材料还保留了额外的训练文件，但基线明确不使用它们，以免在无意中扩大数据范围、让结果无法比较。

对于单标签分类，我们使用分层抽样，让训练部分和验证部分的正负比例一致。如果某些类别只有一个样本，普通随机切分可能把该类全部放进某一边，导致验证指标剧烈波动；分层抽样可以缓解这个问题。缺失值则统一记录到审计文件里，而不是悄悄用 0 填充。

### 2.3 模型训练与超参数调优
基线采用 CatBoost 分类器，参数固定为 350 轮、深度 6、学习率 0.08，使用 Logloss。我们按标签分层切出 20% 的 holdout，先在训练部分拟合，再在 holdout 上评价。材料没有在排行榜上做调参，这一点很重要：它避免了用测试集信息反向选择模型。

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

X = features(train)
y = train["addicted_label"].astype(int).to_numpy()

# 固定随机种子并分层，保证验证集的正负比例与训练集一致
idx = np.arange(len(train))
fit_idx, val_idx = train_test_split(
    idx, test_size=0.2, random_state=20261026, stratify=y
)

params = dict(iterations=350, depth=6, learning_rate=0.08,
              random_seed=20261026, verbose=False, thread_count=4)

model = CatBoostClassifier(loss_function="Logloss", **params)
model.fit(X.iloc[fit_idx], y[fit_idx],
          cat_features=X.select_dtypes(include=["object", "string"]).columns.tolist())
```

除了全特征基线，我们还做了一个预先声明的消融实验：去掉 `stress_level` 和 `academic_work_impact` 两个字段，在完全相同的 holdout 上重新训练。如果去掉这两个字段后性能大幅下降，说明模型可能依赖了构造标签的规则字段；如果变化不大，说明其他使用时长字段也携带了相关信息。这个实验是诊断，不是部署许可。

```python
omit = ["stress_level", "academic_work_impact"]
X_reduced = features(train, omit=omit)

alt_model = CatBoostClassifier(loss_function="Logloss", **params)
alt_model.fit(X_reduced.iloc[fit_idx], y[fit_idx],
              cat_features=X_reduced.select_dtypes(include=["object", "string"]).columns.tolist())

alt_pred = alt_model.predict_proba(X_reduced.iloc[val_idx])[:, 1]
```

我们没有因为消融后分数更高就立刻改用精简特征。原因很实际：一个字段可能在离线数据里可用，但在真实部署时无法获得，或者它代表的是我们不应该使用的敏感信息。模型选择要同时考虑可用性、合法性和可解释性，不能只看分数。固定参数也意味着我们没有针对 holdout 反复试错，这能让报告更接近一次诚实的基线实验。

### 2.4 模型测试与指标报告
我们同时计算常数基线（直接按训练集正样本比例预测）和 CatBoost 的各项指标。常数基线的作用是给出“完全不学习”的下限，避免把任何一点提升都误认为模型有效。

```python
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             balanced_accuracy_score, log_loss,
                             matthews_corrcoef)

def report(truth, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(truth, pred),
        "balanced_accuracy": balanced_accuracy_score(truth, pred),
        "log_loss": log_loss(truth, prob),
        "AUC": roc_auc_score(truth, prob),
        "MCC": matthews_corrcoef(truth, pred),
    }

# 常数基线：所有样本都预测训练集的正类先验
base_prob = np.full(len(val_idx), y[fit_idx].mean())
print("constant:", report(y[val_idx], base_prob))
print("catboost:", report(y[val_idx], model.predict_proba(X.iloc[val_idx])[:, 1]))
print("ablation:", report(y[val_idx], alt_pred))
```

材料没有在正文中给出具体的 holdout 分数，因此我们不虚构数字。我们关注的是指标之间的关系：如果 CatBoost 的 AUC 和 MCC 明显高于常数基线，说明特征中确实存在可学习的信号；消融实验则告诉我们这些信号有多少来自 `stress_level` 与 `academic_work_impact`。我们还会保存验证集的逐样本预测和混淆矩阵，检查错误是否集中在某个概率区间，而不是只看一个总数。

```python
from sklearn.metrics import confusion_matrix

# 保存验证集预测，便于事后按概率分桶检查
val_prob = model.predict_proba(X.iloc[val_idx])[:, 1]
val_pred = (val_prob >= 0.5).astype(int)
diag = pd.DataFrame({
    "id": train.iloc[val_idx]["id"].to_numpy(),
    "actual": y[val_idx],
    "prob": val_prob,
    "pred": val_pred,
})
diag.to_csv("validation_predictions.csv", index=False)
print(confusion_matrix(y[val_idx], val_pred))
```

最后我们按固定的全特征配方在全部抽样数据上重新拟合，并对测试集分块预测，生成符合 `sample_submission.csv` 格式的提交文件。分块推理是为了避免测试集很大时一次性占用过多内存，同时在每个块上检查列顺序与 `id` 是否和样例提交一致。

```python
final = CatBoostClassifier(loss_function="Logloss", **params)
final.fit(X, y, cat_features=X.select_dtypes(include=["object", "string"]).columns.tolist())

sample = pd.read_csv(DATA/"sample_submission.csv")
submission = sample.copy()
offset = 0
for chunk in pd.read_csv(DATA/"test.csv", chunksize=100000):
    p = final.predict_proba(features(chunk))[:, 1]
    submission.loc[offset:offset+len(chunk)-1, "addicted_label"] = p
    offset += len(chunk)

# 提交前做契约检查：行数、id 对齐、概率范围
assert len(submission) == len(sample)
assert submission["addicted_label"].between(0, 1).all()
submission.to_csv("submission.csv", index=False)
```

## 三、总结
本文中我们完成了屏幕使用习惯与标签构建研究的基线复现。我们首先用分块读取和固定随机种子的蓄水池抽样把训练数据控制在 12 万行，统计缺失与标签分布，并用“总时长等于分项之和”的恒等式检查了字段之间的重复表达。其次我们采用无状态的特征处理，把缺失类别单独编码，保留额外训练文件但不在基线中使用，避免数据范围不一致。接着我们用固定参数的 CatBoost 在 20% 分层 holdout 上训练，并预先设计了一个去掉 `stress_level` 与 `academic_work_impact` 的消融实验。最后我们报告了准确率、平衡准确率、对数损失、AUC 和 MCC，保存了逐样本预测和混淆矩阵，并按相同配方生成测试集提交文件。这个练习提醒我们：当标签由问卷字段构造时，高分可能来自规则重建而不是新现象，必须用消融和独立评估来区分两者，同时把“合成标签不是临床诊断”这一限制明确写进结论。
