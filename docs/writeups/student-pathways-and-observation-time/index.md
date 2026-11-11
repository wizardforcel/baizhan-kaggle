# Student Pathways and Observation Time

## 一、比赛简介

### 1.1 比赛背景

```python
# 用配置明确实验属于哪一场比赛，避免把别的比赛的数据混进来
CONFIG = {
    'episode': 's4e6',
    'title': 'Student Pathways and Observation Time',
    'competition_slug': 'student-pathways-and-observation-time',
    'dataset_slug': 'student-pathways-and-observation-time-data',
    'targets': ['Target'],
    'family': 'class',
}
```

我们首先要把比赛放在正确的背景里理解。材料说明，这是一场 Kaggle Playground S4E6 的练习，使用的文件是没有改动的合成比赛文件；其上游数据集的主题是预测学生的辍学和学业成功。数据贡献者包括 Valentim Realinho、Mónica Vieira Martins、Jorge Machado 和 Luís Baptista，比赛贡献者是 Walter Reade 与 Ashley Chow，数据以 CC BY 4.0 许可发布。这里的“预测学生结果”只能理解为历史结果建模练习，不能把它解释成学生能力的判断，更不能据此限制学生接受教育的机会。

这份材料还有一个很重要的教育研究提醒：模型训练所使用的信息必须和实际决策时间一致。第二学期结束以后才出现的字段，不能拿来宣称自己完成了“入学时的早期预警”。因此，我们不只关注模型能不能得到分数，还要说明预测截止时间、结果观察窗口以及哪些字段当时已经可用。材料把这项独立研究定位为学习基线，而不是一个已经可以直接部署的学生支持系统。

### 1.2 比赛目标

```python
# 从提交样例读取比赛要求，确认第一列是编号，其余列是需要预测的目标
schema = pd.read_csv(DATA / 'sample_submission.csv', nrows=3).columns.tolist()
id_col = schema[0]
targets = ['Target']
assert schema[1:] == targets

# 训练表中除编号和目标以外的字段，才是模型可以使用的输入
feature_columns = [c for c in train if c not in targets + [id_col]]
```

其次，我们把目标说清楚。训练集包含一列编号、一组学生相关特征以及名为 `Target` 的目标列；测试集没有 `Target`，我们需要根据训练数据学习分类规律，再为测试集填入目标类别。材料把任务族设为 `class`，所以这里是单标签分类：每一行对应一个学生记录，模型要从多个可能类别中选择一个结果。

材料没有给出完整的数据字典，也没有列出所有特征的含义，因此我们不擅自解释每一列代表什么。我们只依据代码确认：特征列由训练表自动得到，目标列明确是 `Target`，编号列来自 `sample_submission.csv`。另外，文件夹中可能还有额外训练 CSV 文件，但本次基线特意保留它们而不使用；这保证了“哪些行真正参与训练”是可以审计的。

这里还要区分两个概念。比赛上的预测目标，是对给定表格中的 `Target` 做分类；教育研究中的“入学时预测”或“学期末支持”，则是不同的决策任务。若换了决策时间，就要重新定义可用特征、符合资格的人群和还剩多长的结果观察期，不能只把同一个模型换个名字。

### 1.3 评价标准

```python
# 分类任务同时报告几种互补指标：不能只看一个总准确率
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.metrics import roc_auc_score, matthews_corrcoef

labels = p.argmax(axis=1)  # 概率最大的类别作为最终预测
result = {
    'accuracy': float(accuracy_score(truth, labels)),
    'balanced_accuracy': float(balanced_accuracy_score(truth, labels)),
    'log_loss': float(log_loss(truth, p, labels=np.arange(len(classes)))),
}
# 只有二分类时，材料才额外计算 AUC 和 MCC
if len(classes) == 2:
    result['AUC'] = float(roc_auc_score(truth, p[:, 1]))
    result['MCC'] = float(matthews_corrcoef(truth, labels))
```

接着，我们依据材料中的 `score` 函数说明评价方式。因为本题是多分类任务，代码会计算 `accuracy`、`balanced_accuracy` 和 `log_loss`。准确率表示预测类别与真实类别相同的比例；平衡准确率会分别看各类别再取平均，类别数量不均衡时更有参考价值；交叉熵损失 `log_loss` 不只看猜没猜对，还会惩罚模型对错误类别过度自信，数值越低越好。材料没有给出这次运行的具体排行榜分数或验证分数，所以我们不编写一个虚构的分数，而是如实报告评价流程和输出文件。

如果类别数恰好是二类，材料还会计算 AUC 和 Matthews 相关系数；本题的代码会先把类别排序并编码，再根据类别数量决定使用二分类或多分类损失。最终提交必须遵守样例提交文件的列顺序，目标列不能有缺失值，编号必须唯一，并且保存后的文件还要重新读回检查。

## 二、解题过程

### 2.1 数据探索（EDA）

```python
from pathlib import Path
import numpy as np
import pandas as pd

# 固定随机种子，让抽样和后续划分每次都能复现
SEED = 20261026
MAX_ROWS = 120000
rng = np.random.default_rng(SEED)
reservoir = None
rows_seen = 0
missing = None

# 分块读取，避免一次性把原始训练表全部放进内存
for chunk in pd.read_csv(DATA / 'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
    # 为每行生成随机优先级，保留优先级最小的固定数量
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
print(train.shape, train.dtypes)
print(missing.sort_values(ascending=False))
print(train['Target'].value_counts(dropna=False))
```

我们首先进行 EDA，也就是先观察数据的形状、字段类型、缺失情况和目标分布，而不是马上训练模型。材料采用了分块读取：每次读入 `100000` 行，同时累计原始行数和每个字段的缺失行数。为了节省内存，代码使用固定随机种子 `20261026` 做均匀蓄水池抽样，最多保留 `120000` 行。这个上限只是本次实验的内存控制策略，不代表原始文件只有这么多行，也不代表所有可用训练行都被模型拟合。

我们还要检查编号是否唯一、目标是否缺失。材料明确要求不能悄悄给标签填补一个值；如果 `Target` 有缺失就直接报错。缺失统计会保存为 `missingness.csv`，审计信息会记录原始行数、抽样行数、抽样方式、特征列、目标列以及是否使用了额外文件。这样我们不仅知道“模型跑出来了什么”，也知道“模型到底看过哪些数据”。

### 2.2 数据清洗

```python
def features(frame, omit=()):
    # 只选模型允许使用的字段；omit 用于做预先声明的可用性实验
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):
            # 日期字段拆成简单时间成分，避免直接把日期字符串交给模型
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c + '_year'] = dt.dt.year
            x[c + '_month'] = dt.dt.month
            x[c + '_weekday'] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 类别文本统一为字符串，并把缺失明确标成一个类别
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列转成数字，非法值和无穷值改成缺失供模型处理
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return x

X = features(train)
assert not train[['Target']].isna().any().any()
```

其次，我们做任务所需的最小清洗。当前配置中的 `date_columns` 是空列表，所以这次不会凭空创造日期特征；代码保留了通用的日期处理逻辑，只有配置明确声明日期列时才会执行。数值字段被转换为数值，正负无穷被改为缺失；非数值字段被转成字符串，缺失值用 `__MISSING__` 明确标记。这样做的目的不是把所有缺失都强行补成平均数，而是让 CatBoost 能区分“原本就是一个类别”和“这里没有记录”。

我们不对标签做静默填补，也不把整张表的目标统计量带入特征。`features` 是无状态转换：它只依照输入行和预先确定的字段列表处理数据，不从验证集学习规则。材料还特别指出，某些额外字段的删除只是“可用性消融诊断”，不能自动成为部署许可。尤其是第二学期批准数、第二学期成绩和第二学期考核数，只有在相应决策时间已经结束后才能使用；把它们放入入学时模型会造成时间泄漏。

### 2.3 模型训练与超参数调优

```python
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

# 先把类别名称映射为整数，便于分类器训练，同时保留原类别顺序
classes = sorted(train['Target'].unique().tolist())
mapping = {v: i for i, v in enumerate(classes)}
y = train['Target'].map(mapping).to_numpy()
indices = np.arange(len(train))
fit_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=SEED, stratify=y
)

def model_fit(x, target):
    # CatBoost 能直接处理类别列，减少手工独热编码带来的复杂度
    cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()
    params = dict(
        iterations=350, depth=6, learning_rate=0.08,
        random_seed=SEED, verbose=False,
        thread_count=4, allow_writing_files=False
    )
    model = CatBoostClassifier(
        loss_function='Logloss' if len(classes) == 2 else 'MultiClass',
        **params
    )
    model.fit(x, target, cat_features=cat_cols)
    return model

model = model_fit(X.iloc[fit_idx], y[fit_idx])
pred = model.predict_proba(X.iloc[val_idx])
```

接着，我们把数据分成拟合集和验证集。材料使用固定的 `20%` 留出集，并对单标签分类采用分层抽样，保证各类别在两边的比例尽量一致。验证标签不参与预处理、提前停止或参数选择；这条规则很重要，因为如果我们一边看验证答案一边改模型，验证分数就不再是独立检查。

主模型是 CatBoost 分类器。固定设置为 `350` 次迭代、树深 `6`、学习率 `0.08`、随机种子 `20261026`、4 个线程，并关闭写文件。CatBoost 的实际便利之处是可以把字符串类别列直接交给 `cat_features`，我们不必先把每个类别手工展开成许多 0/1 列。二分类使用 `Logloss`，多分类使用 `MultiClass`。

这次实验不在排行榜上调参。材料预先规定了固定参数，并且只在相同验证划分上做一个可证伪的第二实验：如果配置中有消融字段，就删去第二学期相关字段后重新训练；否则就使用更小的训练预算。完整特征模型仍然是预先指定的提交模型，消融结果只是关于字段可用性的诊断，不是“哪个验证分数高就自动上线”的理由。

### 2.4 模型测试与指标报告

```python
# 先用训练集类别比例构造常数基线，作为“什么规律也没学”的参照
baseline = np.tile(
    np.bincount(y[fit_idx], minlength=len(classes)) / len(fit_idx),
    (len(val_idx), 1)
)

# 把预测概率转为类别并保存验证结果，便于复查每一行的错误
labels = pred.argmax(axis=1)
diagnostics = pd.DataFrame({
    'id': train.iloc[val_idx][id_col].to_numpy(),
    'actual': y[val_idx],
    'prediction': labels,
})
for j, c in enumerate(classes):
    diagnostics['prob_' + str(c)] = pred[:, j]

diagnostics.to_csv(OUT / 'validation_predictions.csv', index=False)
```

然后，我们在固定验证集上比较常数基线和 CatBoost，并把结果写入 `validation_scores.csv`。材料中的运行逻辑会同时保存逐行预测、混淆矩阵和类别概率。由于页面材料只展示了代码与实验契约，没有展示这次运行的 `validation_scores.csv` 内容，我们不能声称某个模型取得了多少准确率，也不能把别人的名次当成自己的结果。材料提到 Matt OP 的报告名次为第 6，但同时明确说那是另一份账户的发现，不是本独立基线复现出的分数。

最后，我们用全量的 `120000` 行上限样本重新拟合固定的全特征模型，再分块读取测试集进行推理。每个测试块都要检查列名顺序、编号是否和提交样例对应；预测完成后检查目标没有缺失、编号唯一，并把保存的 `submission.csv` 重新读回，确认列名和行数与样例完全一致。分块推理的意义是控制内存，不改变提交顺序，也不把测试标签偷偷用于训练。

```python
# 全量拟合固定的全特征方案；消融实验不会替换提交模型
final = model_fit(X, y)
sample = pd.read_csv(DATA / 'sample_submission.csv')
submission = sample.copy()
offset = 0

# 测试集也采用和训练集完全一致的特征转换，并按块写回预测
for chunk in pd.read_csv(DATA / 'test.csv', chunksize=100000):
    assert chunk.columns.tolist() == [c for c in train.columns if c not in ['Target']]
    n = len(chunk)
    assert np.array_equal(chunk[id_col].to_numpy(), sample.iloc[offset:offset+n, 0].to_numpy())
    p = final.predict_proba(features(chunk))
    values = np.asarray(classes)[p.argmax(axis=1)][:, None]
    submission.loc[submission.index[offset:offset+n], 'Target'] = values[:, 0]
    offset += n

assert offset == len(sample)
assert submission[id_col].is_unique
assert not submission[['Target']].isna().any().any()
submission.to_csv(OUT / 'submission.csv', index=False)
```

我们最后还会生成 `run_manifest.json`，记录随机种子、抽样审计、验证结果、消融字段、提交模型说明、验证方式、源文件哈希和软件版本。这样的报告能让别人知道这只是一次固定留出集上的学习基线，而不是对未来学生结果的部署保证。材料要求下一步研究分别构造入学时和学期末的特征清单，并记录未完成观察窗口的学生，不能把尚未跟踪完的人直接标成“成功”。

## 三、总结

本文中我们完成了 Student Pathways and Observation Time 的一套可审计分类基线。我们首先确认了比赛来源、目标列 `Target` 和数据许可背景，并强调历史结果预测不能代表学生能力，也不能用来限制教育机会。其次我们用分块读取、固定种子和最多 `120000` 行的均匀蓄水池抽样控制内存，同时保存缺失统计和训练行审计。接着我们使用无状态特征转换处理数值、类别和缺失值，保持标签不被静默填补，并警惕第二学期字段与入学时决策之间的时间泄漏。然后我们用固定 `20%` 分层留出、CatBoost 的 `350` 次迭代、深度 `6` 和学习率 `0.08` 建立全特征模型，并用常数基线及预先声明的字段消融做比较。最后我们在没有查看排行榜标签的前提下分块预测测试集，检查提交契约，保存预测、混淆矩阵、验证分数和运行清单。

这份材料最值得保留的结论不是一个没有证据的分数，而是研究设计本身：预测截止时间决定哪些信息可以进入模型，结果观察窗口决定标签是否已经可靠，符合资格的人群决定不同 landmark 分数能否公平比较。如果继续学习，我们应当分别比较入学和学期末人群，报告共同人群上的结果，并在存在删失或未完成随访时考虑时间到事件的表达。更进一步，预测“谁可能辍学”不等于知道“谁会从辅导中获益”；学生支持项目还要考虑班级或同伴之间的溢出效应，以及学习、福祉、负担和公平机会等学生中心的结果。
