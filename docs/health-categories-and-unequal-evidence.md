# Health Categories and Unequal Evidence

## 一、比赛简介

### 1.1 比赛背景
本项比赛源自 Kaggle Playground S6E7，是一个关于“健康类别与不平等证据”的独立研究项目。原始数据来自 Ziya: College Student Health Behavior Dataset，比赛在标准分类任务的基础上加入了三个教育目标：报告每类别的召回率、报告各组样本量，并在每种率旁边报告其分母，强调小样本组的证据不足。主办方刻意保留了合成标签的局限，并公开声明这些标签不能用于实际健康评估，只作为方法学练习。

比赛的核心问题是“不同人群在健康行为类别上是否被公平地覆盖”，因此不仅要看整体准确率，更要看类别之间的表现是否平衡。材料里提到“Balanced accuracy equalizes class contributions, but a small group still has less evidence”，说明即使我们把每个类别权重一样，小样本类别的估计仍然不稳定。

为了让参赛者直面不平等证据，比赛规定所有模型必须在固定的 20% 分层验证集上报告每类别的召回率，并在每个率旁边列出对应的样本量。提交前还要输出一个数据收集建议，说明如果要提高对小样本类别的预测可靠性，需要采集哪类人群、多少样本。这一设计把技术指标和研究伦理连接在了一起，避免模型只优化整体分数而忽视脆弱群体。

### 1.2 比赛目标
输入是一张包含学生匿名特征的表格，目标列是 `health_condition`，是一个单标签分类问题。每个学生对应一行特征和一个类别标签。特征中包含人口学、生活方式、身体指标等字段，部分字段可能在部署时不可用。比赛要求我们对测试集进行预测，并提交每行的类别。

数据量很大，材料中使用了一个有界均匀采样策略，最大保留 120,000 行训练样本，使用随机数生成器按 `seeded uniform reservoir` 抽样，保证抽样可复现。验证集采用固定的 20% 分层抽样，`stratify` 保证每个类别的比例保持一致。额外训练文件被保留但不使用，目的只是演示数据收集的复杂性。

### 1.3 评价标准
比赛以 `balanced_accuracy` 为主要指标，即每个类别召回率的平均值。相较于普通准确率，balanced accuracy 对类别不平衡更公平，因为它对每个类别贡献相同权重。材料中还要求报告 `log_loss`、`accuracy`、`AUC` 等诊断指标，并生成混淆矩阵。

与此同时，比赛强调“证据不等同于证据量”。例如，某类别只有两条样本，即使 100% 召回，也不能当作稳定结论。因此最终报告必须同时给出类别计数、召回率以及对应的样本量。数据收集建议也是交付物的一部分。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先读取原始训练表并按块抽样，避免整表读入内存：

```python
from pathlib import Path
import pandas as pd, numpy as np
MAX_ROWS = 120000
reservoir = None
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    # 随机抽样优先级，保证均匀
    chunk['_priority'] = np.random.default_rng(SEED).random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk])
    reservoir = reservoir.nsmallest(MAX_ROWS, '_priority')
train = reservoir.drop(columns=['_priority'])
```

抽样完成后我们检查缺失值分布：

```python
missing = train.isna().sum()
pd.DataFrame({'field': missing.index, 'missing_rows': missing.values}).to_csv('missingness.csv')
```

材料里提到特征中包含缺失值，但我们不会对训练标签做隐式填补。类别分布显示多个健康类别，其中部分类别样本量较小，这是后续关注不平等证据的核心。

我们还计算了每个类别的样本计数，并可视化类别间的比例。EDA 的结论是：类别不平衡明显，少数类别的样本量明显少于多数类别，整体数据包含大量类别型特征，需要统一为字符串处理。日期型特征在此数据集中不存在。

在了解分布之后，我们检查了特征的取值范围。数值型特征中存在一些极端值和长尾分布，这对树模型影响不大，但如果未来改用线性模型就会是个问题。类别型特征中有大量唯一值，这意味着把它们做成 one-hot 编码会产生维度爆炸，必须使用支持类别列的模型。材料也指出，某些字段在部署时不可用，因此我们要留意哪些字段贡献大，以便在上线时权衡数据获取成本。

### 2.2 数据清洗
特征预处理必须完全无监督，不能使用目标统计量。我们定义了一个通用的 `features` 函数：

```python
def features(frame, omit=()):
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]):
            # 非数值统一转为字符串占位符
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors='coerce')
    return x
```

对于数值列，我们把无穷值转成 NaN，但不做基于目标的均值填补；对于类别列，我们用特殊字符串 `__MISSING__` 表示缺失，让 CatBoost 直接处理。材料强调“Stateless transforms. No target statistics”，也就是说任何填补都不能依赖标签。这个原则在医疗数据上尤其重要，避免泄漏未来信息。

我们在训练和验证划分之前完成特征转换，确保验证集与训练集使用同一套规则。划分采用 `train_test_split`，`test_size=0.2`，`stratify=y`，以保证小类别也在验证集中有代表。我们不使用交叉验证，因为比赛要求固定验证集以保持实验可重复。

除特征清洗外，我们还保留了可重复性的审计信息：采样行数、种子、特征列名、缺失值文件、标签映射等，全部写入 `run_manifest.json`，以便后续审查。

这里要特别说明随机种子和分层的作用。因为训练集只抽样 12 万行，如果每次抽样不同，类别比例就会波动，小类别的召回率尤其不稳定。固定种子后，任何人都能复现同一条训练集；分层则保证验证集里的类别比例与抽样后的训练集一致。两者结合，才能让“证据”的比较有意义。材料还明确禁止使用排行榜标签调参，这是为了防止过拟合测试集。

### 2.3 模型训练与超参数调优
比赛明确规定模型参数提前固定，不在看到验证分数后调参。我们采用 CatBoost 作为基线，因为它对类别特征天然友好，且支持 `auto_class_weights='Balanced'`：

```python
from catboost import CatBoostClassifier
cat_cols = x.select_dtypes(include=['object','string']).columns.tolist()
params = dict(
    iterations=350,
    depth=6,
    learning_rate=0.08,
    random_seed=SEED,
    verbose=False,
    thread_count=4,
    allow_writing_files=False
)
model = CatBoostClassifier(
    loss_function='MultiClass',
    auto_class_weights='Balanced',
    **params
)
model.fit(X.iloc[fit_idx], y[fit_idx], cat_features=cat_cols)
```

训练前后我们不做早期停止，也不看验证分数。固定迭代次数是为了保证每次运行结果一致。材料同时定义了一个常数基线作为对照：

```python
def constant_prediction(target, n):
    # 常数基线：每个类别的先验概率
    return np.tile(np.bincount(target)/len(target), (n, 1))
```

为了验证“证据不等同于样本量”这一假设，我们做了第二实验：去除 `bmi` 和 `physical_activity_level` 两个特征。它们在部署场景中可能不可用，我们想看看移除后各类别召回率的变化：

```python
omit = ['bmi', 'physical_activity_level']
reduced = features(train, omit=omit)
alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
```

这个删减实验在相同验证集上进行，目的是评估字段可用性对公平性的影响。材料强调它只是诊断，不是调参许可。

如果删掉 `bmi` 和 `physical_activity_level` 后 balanced accuracy 明显下降，说明模型对这两个字段依赖较强；如果下降很小，则说明其他特征已经包含了足够信息。无论结果如何，我们都不能因为删减后分数更高就改用删减模型，因为比赛预设的提交模型是“全特征基线”。这种预先注册实验的做法可以减少“看了结果再编故事”的风险，让结论更可信。

### 2.4 模型测试与指标报告
模型训练完成后，我们在验证集上计算指标：

```python
from sklearn.metrics import balanced_accuracy_score, accuracy_score, log_loss
labels = pred.argmax(axis=1)
result = {
    'accuracy': accuracy_score(y_val, labels),
    'balanced_accuracy': balanced_accuracy_score(y_val, labels),
    'log_loss': log_loss(y_val, pred)
}
```

我们还输出每个类别的召回率，并在报告中标注对应的样本数。材料里特别强调“Report denominators beside every rate”。因此我们在 `validation_predictions.csv` 中保留了每行的 `id`、`actual`、`prediction`、`prob_*`，并生成混淆矩阵供人工审查。

随后我们对测试集做分块预测，保证 `sample_submission` 的列顺序和 id 完全一致：

```python
final = model_fit(features(train), y)
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    pred = predict(final, features(chunk))
    submission.loc[offset:offset+len(chunk), targets] = pred
```

提交前我们做完整性检查：行数一致、无 NaN、概率和为 1、id 唯一。所有检查通过后，模型被视为可提交的基础线。

## 三、总结
本文中我们完成了 Health Categories and Unequal Evidence 的基线建模。我们首先用有界均匀采样和分层抽样构建了可复现的训练验证集，并对类别型特征进行无监督清洗。接着用固定参数的 CatBoost 建立平衡权重模型，并通过常数基线和移除 `bmi`、`physical_activity_level` 的删减实验来校准对证据量和字段可用性的理解。最后我们在验证集上报告整体 balanced accuracy、每类别召回率及其样本量，并按规范生成了测试集提交文件。

这个练习的最大收获是：分类指标不能脱离样本量解读。我们学会了在报告中同时展示率及其分母，也明白了在合成标签上取得的高准确率并不具备医疗决策的合法性。后续研究应聚焦于数据收集的公平性、时间依赖性和独立评估。

在具体写法上，我们遵循了三条原则：第一，所有预处理只使用训练集，验证集只用于最终评估；第二，任何字段删减实验都提前声明，不根据结果反复挑模型；第三，报告中同时保留点估计和样本量，让读者能判断证据强度。这三条原则虽然简单，却能显著降低结论被误读的概率。对入门者来说，最值得记住的是：当某个类别的召回率看起来很高时，先看它的分母；当某个差距看起来很大时，先看它的不确定性和比较次数。只有把“率”和“证据量”放在一起，报告才经得起追问。这套方法对任何“先分组、后比较”的任务都适用，是连接统计推断与业务决策的桥梁。掌握它，比多背几个模型名称更有长期价值。
