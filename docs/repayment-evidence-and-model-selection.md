# Repayment Evidence and Model Selection（还款证据与模型选择）

## 一、比赛简介

### 1.1 比赛背景

这个比赛围绕“贷款是否能够还清”这一类二分类预测问题展开。材料说明，比赛使用的是 Kaggle Playground S5E11 的未修改合成数据，原始比赛贡献者为 Walter Reade 和 Elizabeth Park，数据来源许可为 CC BY 4.0。上游资料被称为 Loan Prediction Dataset 2025。这里的“合成”很重要：数据可以用来学习表格建模、验证集划分和模型选择，但不能直接当成真实贷款审批依据。真实借贷场景还需要外部数据、独立验证、风险治理和人工审核。

这场比赛的特别之处，不只是训练一个模型然后追求一个最高分。题目强调了一个常见但容易忽略的问题：如果我们不断尝试很多模型和特征，验证集上的微小提升可能只是随机波动。最后选择出来的“最好模型”不一定真的最好，这就是模型选择中的证据问题。因此材料把任务设计成一个可复现的独立研究：我们要保留试验记录，使用固定的数据抽样和固定的留出集，比较常数基线与 CatBoost 基线，并记录特征消融实验，而不是根据排行榜反复调参。

材料还明确规定，最多使用 120000 行均匀抽样的训练数据。这个上限是为了控制内存和运行时间，不表示原始训练文件只有这些行，也不表示模型已经利用了所有可用样本。额外的训练文件会被保留在数据目录中，但本次固定基线不使用它们。这样的约束让每次实验都更容易重现，也让我们能够清楚地回答“模型到底用哪些数据训练”的问题。

### 1.2 比赛目标

我们的输入是比赛提供的 `train.csv`、`test.csv` 和 `sample_submission.csv`。提交样例的第一列被当作 ID，后续列是目标列。当前配置中目标列是 `loan_paid_back`，任务族是 `binary`，也就是二分类任务：模型需要给每个样本预测贷款是否还清的概率或类别。代码会从提交样例自动读出列名，并检查样例中的目标列是否与配置一致，这样可以避免把列名写错。

训练数据中的其他列都是候选特征。特征可能包含数值列，也可能包含类别列。代码没有把某一个固定字段写死成数值变量，而是根据 pandas 的数据类型来处理：数值列转成数值，非数值列转成字符串类别。配置中还列出了一个 `grade_subgrade` 消融字段，第二个实验会在相同训练/验证划分上去掉这个字段，观察它是否真的提供了稳定证据。需要注意，消融实验只是诊断，不等于我们可以在实际部署时随便使用或删除字段。

我们首先建立一个可审计的配置和随机种子：

```python
# 用一个配置字典集中记录比赛、目标列和预先规定的消融字段
CONFIG = {
    'episode': 's5e11',
    'title': 'Repayment Evidence and Model Selection',
    'competition_slug': 'repayment-evidence-and-model-selection',
    'dataset_slug': 'repayment-evidence-and-model-selection-data',
    'targets': ['loan_paid_back'],
    'family': 'binary',
    'ablation': ['grade_subgrade'],
    'date_columns': []
}

# 固定随机种子，使抽样、划分和 CatBoost 训练可以重现
SEED = 20261026
MAX_ROWS = 120000
```

### 1.3 评价标准

材料中的评分函数会根据任务类型选择指标。对于本比赛的二分类任务，代码计算 `accuracy`、`balanced_accuracy`、`log_loss`，并额外计算 ROC AUC 和 MCC。准确率是预测类别正确的比例；平衡准确率会分别计算正类和负类的识别效果，再进行平均，类别不平衡时更有参考价值；Log Loss 会惩罚过度自信的错误概率；ROC AUC 衡量模型把正样本排在负样本前面的能力；MCC 是综合真阳性、真阴性、假阳性和假阴性的相关系数。

我们不能把任何一个指标单独当成真实风险能力。尤其是这份数据是合成数据，材料没有给出本次运行的具体验证分数，也没有允许我们使用排行榜标签。因此我们只报告固定留出集上由程序实际生成的指标，并同时保留常数基线、完整特征模型和 `grade_subgrade` 消融模型的结果。这样可以回答两个问题：模型是否超过了最简单的基线，以及去掉指定字段后性能变化是否足以改变我们的判断。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取提交样例来确认 ID 列和目标列，再分块读取训练文件。分块读取的意思是一次只把一部分数据放进内存，而不是把整个文件一次性载入。每个块都统计缺失值，并给每一行生成一个固定随机数作为抽样优先级。我们不断保留优先级最小的 120000 行，这是一种有固定随机种子的均匀蓄水池式抽样。与此同时，我们记录原始文件看见了多少行、最终保留了多少行以及有哪些额外文件。

```python
from pathlib import Path
import numpy as np
import pandas as pd

# 从提交样例取得列契约，避免手写列名造成提交格式错误
schema = pd.read_csv(DATA / 'sample_submission.csv', nrows=3).columns.tolist()
id_col = schema[0]
targets = CONFIG['targets']
assert schema[1:] == targets

# 分块读取训练数据，控制内存，并记录完整文件的缺失情况
reservoir = None
rows_seen = 0
missing = None
rng = np.random.default_rng(SEED)
for chunk in pd.read_csv(DATA / 'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    # 只保留固定上限，防止抽样过程把内存越积越大
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
assert train[id_col].is_unique
assert not train[targets].isna().any().any()
feature_columns = [c for c in train if c not in targets + [id_col]]
```

接着我们检查 ID 是否唯一、目标列是否缺失，并把缺失统计保存为 `missingness.csv`。代码对目标缺失直接报错，而不是悄悄填补标签。原因很简单：标签是我们要学习的答案，如果标签不确定，强行填补会把人为猜测混进训练证据。材料还会把所有额外 CSV 文件的名字放进审计记录，明确说明本次基线没有使用它们。

在 EDA 阶段我们还应该观察数值列和类别列的类型、目标类别数量、每个类别的样本量以及缺失列分布。材料没有提供某次运行的具体计数，因此我们不把样本量、类别比例或分数写成未经执行验证的数字；我们让程序输出这些信息，并将它们随运行产物保存下来。对于初学者来说，这一步的核心是先问清楚“数据有多少行、标签是否齐全、特征是什么类型”，再决定模型和清洗方式。

### 2.2 数据清洗

我们其次进行尽量少而明确的数据清洗。`features` 函数只接收训练或测试框架，并根据要消融的字段选择特征。数值列先转为数值，正无穷和负无穷替换成缺失；字符串列转成字符串并把缺失值标记成 `__MISSING__`。这样做保留了“缺失本身可能有意义”的信息，也让 CatBoost 能够稳定接收类别特征。配置中的日期列列表当前为空，所以本次运行没有日期拆分；如果配置声明了日期列，代码会拆成年、月和星期几。

```python
# 这个函数只做不依赖标签的转换，避免把验证集信息泄露到训练过程

def features(frame, omit=()):
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):
            # 日期不能直接当作普通文字，拆出容易理解的时间字段
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c + '_year'] = dt.dt.year
            x[c + '_month'] = dt.dt.month
            x[c + '_weekday'] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 类别缺失单独留一个标记，避免把缺失误当成普通类别
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值转换失败或无穷值统一变成缺失，交给模型处理
            x[c] = pd.to_numeric(x[c], errors='coerce')
            x[c] = x[c].replace([np.inf, -np.inf], np.nan)
    return x

X = features(train)
```

我们不会按照普通自然语言文本的习惯去除停用词，也不会把类别列随意编码成目标均值。材料特别强调“无目标统计”和“无全表插值”，因为这些操作如果使用了验证标签，就会造成数据泄露。对一个入门者来说，数据泄露可以理解成考试前偷看答案：验证分数会变好，但真正遇到新数据时并不会同样好。

### 2.3 模型训练与超参数调优

我们接着使用固定的 20% 留出集。对于单标签分类，`train_test_split` 使用 `stratify=y`，让训练部分和验证部分的类别比例尽量一致。验证标签只用于最后评估，不参与预处理、提前停止或参数选择。这样做牺牲了一部分可用于拟合的数据，却换来了较清楚的比较依据。

```python
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

# 把原始标签映射为从零开始的整数，方便统一计算分类指标
classes = sorted(train[targets[0]].unique().tolist())
mapping = {v: i for i, v in enumerate(classes)}
y = train[targets[0]].map(mapping).to_numpy()
assert len(classes) == 2

# 固定留出集，并用分层抽样保持两个类别的比例
indices = np.arange(len(train))
fit_idx, val_idx = train_test_split(
    indices, test_size=.2, random_state=SEED, stratify=y
)

# 只根据训练特征的类型找出类别列，CatBoost 会原生处理这些列
cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
params = dict(
    iterations=350,
    depth=6,
    learning_rate=.08,
    random_seed=SEED,
    verbose=False,
    thread_count=4,
    allow_writing_files=False
)
model = CatBoostClassifier(
    loss_function='Logloss',
    **params
)
model.fit(X.iloc[fit_idx], y[fit_idx], cat_features=cat_cols)
```

CatBoost 是适合表格数据的梯度提升模型。我们可以把它理解成许多棵小决策树依次纠正前面模型的错误。固定设置为 350 次迭代、深度 6、学习率 0.08，并使用固定随机种子和 4 个线程。这里没有根据排行榜反复调参，也没有把留出集分数拿来无限试验；这些参数在实验设计中预先规定，目的就是让“模型选择”有证据链，而不是只留下一个漂亮的分数。

为了知道模型是否真的有用，我们先建立常数基线。二分类时，常数基线输出训练部分中各类别的比例。它不看任何特征，只回答“如果我们永远按照训练集类别比例猜，验证指标是多少”。随后我们在完全相同的验证行上比较常数基线和 CatBoost。

```python
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# 常数基线不读取特征，只复制训练部分的类别比例
baseline = np.tile(
    np.bincount(y[fit_idx], minlength=len(classes)) / len(y[fit_idx]),
    (len(val_idx), 1)
)

# 用概率计算多种指标，避免只盯着一个分数
pred = model.predict_proba(X.iloc[val_idx])
labels = pred.argmax(axis=1)
results = {
    'accuracy': float(accuracy_score(y[val_idx], labels)),
    'balanced_accuracy': float(balanced_accuracy_score(y[val_idx], labels)),
    'log_loss': float(log_loss(y[val_idx], pred, labels=np.arange(len(classes)))),
    'AUC': float(roc_auc_score(y[val_idx], pred[:, 1])),
    'MCC': float(matthews_corrcoef(y[val_idx], labels))
}
```

我们还按预先写入配置的方案做第二个实验：如果 `grade_subgrade` 存在，就在同一组 `fit_idx` 和 `val_idx` 上去掉它，再重新训练一个 CatBoost。相同划分非常关键，因为如果两个实验用了不同验证行，分数差异可能来自样本不同，而不是字段本身。我们保留完整模型和消融模型的结果，即使消融后分数更高，也不会自动把它改成最终提交模型。材料明确规定，提交模型仍然是完整特征基线。

### 2.4 模型测试与指标报告

我们最后先保存验证预测、混淆矩阵和验证分数，再用完整的抽样训练数据重新拟合固定的完整特征模型。混淆矩阵把真实类别和预测类别交叉统计，可以帮助我们看出错误主要集中在哪一类。对二分类问题，我们还保留每一行的预测概率，而不是只保留 0 或 1，这样后续可以检查概率是否过度自信。

材料没有给出一次实际执行后的 `validation_scores.csv` 内容，所以我们不虚构某个准确率、AUC 或 MCC。正确的报告方式是运行脚本后读取该文件，同时报告 `constant`、`boosting` 和 `availability_ablation`（如果存在）三行，并说明所有结果都来自同一固定留出集。这样的结果比只写一个排行榜分数更有解释力，因为读者可以看到模型相对最简单方案的增益，也可以看到去掉字段后的变化。

```python
from sklearn.metrics import confusion_matrix

# 保存验证集上每一行的真实标签、预测标签和两个类别的概率
validation_predictions = pd.DataFrame({
    'id': train.iloc[val_idx][id_col].to_numpy(),
    'actual': y[val_idx],
    'prediction': labels
})
for j, c in enumerate(classes):
    validation_predictions['prob_' + str(c)] = pred[:, j]
validation_predictions.to_csv(OUT / 'validation_predictions.csv', index=False)

# 混淆矩阵保留类别名称，便于检查哪一类更容易被错判
pd.DataFrame(
    confusion_matrix(y[val_idx], labels),
    index=classes,
    columns=classes
).to_csv(OUT / 'confusion_matrix.csv')
```

提交阶段使用全量的抽样训练集 `X, y` 拟合最终模型，然后分块读取测试集。分块预测可以避免测试文件过大时一次性占满内存。每个测试块都要检查列顺序和 ID 是否与提交样例对应，再把概率写入目标列。二分类提交的是正类概率，因此代码取 `p[:, 1]`。完成后我们还要检查提交行数、列名、ID 顺序、缺失值和概率范围，并进行一次读回测试，确认保存后的 CSV 与样例契约一致。

```python
# 固定完整特征配方，不因为消融结果更好就偷偷改变提交方案
final = CatBoostClassifier(loss_function='Logloss', **params)
final.fit(X, y, cat_features=cat_cols)

sample = pd.read_csv(DATA / 'sample_submission.csv')
submission = sample.copy()
offset = 0
for chunk in pd.read_csv(DATA / 'test.csv', chunksize=100000):
    # 先验证列和 ID 的顺序，再写入预测，避免错位提交
    assert chunk.columns.tolist() == [c for c in train.columns if c not in targets]
    n = len(chunk)
    assert np.array_equal(chunk[id_col].to_numpy(), sample.iloc[offset:offset+n, 0].to_numpy())
    p = final.predict_proba(features(chunk))[:, 1]
    submission.loc[submission.index[offset:offset+n], targets[0]] = p
    offset += n

assert offset == len(sample)
assert submission[id_col].is_unique
assert not submission[targets].isna().any().any()
assert np.isfinite(submission[targets].to_numpy(dtype=float)).all()
assert submission[targets[0]].between(0, 1).all()
submission.to_csv(OUT / 'submission.csv', index=False)
```

我们还保存 `run_manifest.json`，记录随机种子、抽样审计、验证结果、消融字段、提交模型说明、运行时间和 Python、pandas、scikit-learn、CatBoost 版本。这样即使两个实验使用同一个文件名，也可以依靠试验 ID、配置、代码版本、数据版本、划分 ID、环境和预测产物来区分它们。材料提出的进一步方向包括建立实验台账、使用没有标签泄露的 OOF 预测进行正则化 stacking，以及在大量试验后用独立评估处理选择偏差；这些属于后续学习扩展，不是本次基线已经完成的内容。

## 三、总结

本文中我们完成了 Repayment Evidence and Model Selection 比赛的一套可复现基线。我们首先读取未修改的训练、测试和提交样例，根据提交契约确认 ID 与目标列；其次用固定随机种子做最多 120000 行的均匀抽样，并统计缺失值、检查 ID 唯一性以及目标标签完整性；接着我们用不依赖标签的转换函数处理数值列、类别列和缺失值，避免把验证信息泄露到训练过程；然后我们用固定 20% 分层留出集训练 CatBoost，并把类别列交给模型原生处理。

在模型选择方面，我们同时保留了不看特征的常数基线、固定设置的完整特征模型，以及去掉 `grade_subgrade` 的预先规定消融实验。我们没有根据排行榜标签调参，也没有把一次留出结果当成真实部署保证。最后我们按块预测测试集，检查列名、ID 顺序、概率范围、行数和读回结果，再生成 `submission.csv`，并把审计信息、验证预测、混淆矩阵和运行版本写入输出文件。

这份材料给我们的主要启发是：模型分数只是证据的一部分，模型是如何被选出来的同样重要。不断挑选细小提升会放大随机噪声，因此我们需要固定划分、固定配置、保留失败试验，并在最终选择后使用独立评估。对于合成还款目标，我们可以把它当成学习和恢复练习，但不能据此直接做真实借贷决定；真正的贷款系统还需要外部验证、群体与时间划分、不确定性分析、治理流程和持续监控。
