# Rain Signals in a Small Sample

## 一、比赛简介

### 1.1 比赛背景

这是一个来自 Kaggle Playground Series S5E3 的练习赛，官方名称为 Rain Signals in a Small Sample。数据是合成表格，源自 Subho117 的 Rainfall Prediction using Machine Learning，再由 Clare Laurent 重新整理为独立教学研究。比赛的重点不是追求最高排行榜分数，而是**在小样本下如何诚实地评估模型**。

论文式的说明里反复提到：样本很少时，复杂的搜索很容易把验证集“学掉”。我们需要把数据拆分、采样、评估的做法当成结果来报告，而不是事后调参。原始数据被提示是一个 weather-table 标签，它不是可校准的本地天气预报，没有时间与地点验证，不能直接当作部署依据。

在小样本情境里，模型差异往往被随机波动掩盖。我们必须先说明采样假设、拆分方式和评估频率，然后再看提升是否稳定。练习赛还特别提醒：不要把验证集分数当成部署保证，也不要把天气表的标签直接解释为本地预报，这需要时间与地点校验。价值在于学习如何报告不确定性，而不是做出一个看似漂亮的排行榜分数。

整个实验遵循一个固定的协议：只用训练行进行预处理、早停和参数选择；固定的 CatBoost 参数、固定的 20% 保留验证集、用均匀的 reservoir 采样限制内存。额外的训练文件保留但不使用。样本上限是为了节省内存，不是声称使用了全部训练行。

我们把这个协议当成研究假设的一部分来记录。所有配置都事先写死：采样种子、划分种子、模型迭代次数都固定下来，避免看到分数后再调整。正是这种先行承诺，让结果可以被复现，也让我们能够比较不同评估方式带来的差异，而不是把运气当成能力。

### 1.2 比赛目标

我们要做的是一个二分类问题，目标列只有一列 rainfall，family 为 binary。训练文件 train.csv 包含若干特征、id 列和标签 rainfall。测试文件 test.csv 只有特征和 id，sample_submission.csv 给出提交的列顺序。

在配置里明确写出：

```python
CONFIG = {
    'episode': 's5e3',
    'title': 'Rain Signals in a Small Sample',
    'competition_slug': 'rain-signals-in-a-small-sample',
    'dataset_slug': 'rain-signals-in-a-small-sample-data',
    'targets': ['rainfall'],
    'family': 'binary',
    'ablation': ['day'],
    'date_columns': []
}
```

也就是说，我们的输入是带 id 的表格，输出是每个测试样本的降雨概率或标签，列名必须与 sample_submission 保持一致。因为目标是二分类，提交的概率越接近真实降雨情况越好；而训练阶段我们还会把概率转成 0/1 标签，检查准确率和混淆矩阵。

在正式动手之前，我们先把数据的契约固定下来：训练文件、测试文件和样例提交文件的列顺序必须一一对应。读到测试集时我们会先断言它的列与训练集去掉目标后的列一致，避免在数据拼接时出现字段错位。这个检查看起来琐碎，但在表格比赛里，列顺序错误往往会让整个提交作废。

### 1.3 评价标准

根据公开的二分类降雨预测任务描述和本 notebook 的 score 函数，评估以 ROC 曲线下面积 AUC 为主，记分时越大越好。训练脚本里还会同时计算准确率、平衡准确率、对数损失和 Matthews 相关系数，用来描述小样本下的不确定性。我们不把这些指标当成调参依据，只是报告。

AUC 的含义是：随机抽一个真实降雨的样本和一个真实未降雨的样本，模型给降雨样本打出更高概率的可能性。它不像准确率那样依赖阈值，因此适合不平衡的二分类数据。至于对数损失，它惩罚的是概率本身的“自信程度”，即使预测方向对了，如果概率给得太夸张也会被扣分。我们会在验证集上一并观察这些指标，但它们只是诊断工具，不是拿来搜索参数的目标。

## 二、解题过程

整条思路可以概括为：先用受控采样读入数据并检查缺失，再做轻量特征处理，接着用预设的 CatBoost 基线在分层保留集上训练，最后在全样本上重拟并分块预测，我们按 EDA、清洗、训练调优、测试报告四步来走。

### 2.1 数据探索（EDA）

我们首先确定数据来源并用确定性 reservoir 限制样本量，避免把整个表一次性载入内存。数据通过 chunksize 逐块读取，同时记录缺失情况。

```python
# 设置采样上限，超过的样本用确定性 reservoir 保留，保证可复现
MAX_ROWS = 120000
rng = np.random.default_rng(SEED)
reservoir = None
rows_seen = 0
missing = None

for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    # 统计本块缺失数，用来事后写 missingness.csv
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
    # 用随机优先级做均匀采样
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
```

采样完成后我们打印审计信息，包括看到的总行数、实际采样的行数、特征列表和目标列，并把缺失统计保存到 missingness.csv。这一步让我们知道哪些字段缺失严重，是否需要填充或转换。

reservoir 采样的好处是内存可控且可复现。我们用同一个种子生成每块数据的随机优先级，然后保留优先级最小的 MAX_ROWS 行，这样最终集合是对原始流的均匀子集，而不是先加载再随机抽样。这种方法在数据很大时尤其重要，它保证了实验可以在笔记本环境里稳定运行，也让别人可以复现同样的样本。

随后我们检查 id 是否唯一、标签是否缺失，并把特征列定义为所有非目标、非 id 的列。这种轻量 EDA 不引入目标信息，避免泄漏。我们刻意不画直方图或相关性热力图，因为在合成天气表里，相关性可能随时间漂移，过早依赖特定特征会产生虚假信心。

### 2.2 数据清洗

数据本身没有明显的重复，但有两种类型需要统一处理：字符型和数值型。我们写了一个无状态的 features 函数，不学习目标统计，也不在全表上做插补。

```python
def features(frame, omit=()):
    # 保留不在 omit 列表里的特征列
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):
            # 日期列拆成年、月、周几
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c+'_year'] = dt.dt.year
            x[c+'_month'] = dt.dt.month
            x[c+'_weekday'] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 非数值列统一成字符串，缺失填占位符
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列转浮点，无穷值改为 NaN
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return x
```

清洗原则很简单：不要删除信息，也不根据目标做统计填充。字符列的缺失用显式占位符保留，数值列的异常用 NaN 保留，让 CatBoost 自己处理。day 字段被列为 ablation 对象，我们准备在第二实验里去掉它，看是否影响评估。

这里的清洗与常规文本处理不同。我们不会对字符列做小写、去重或标准化，因为类别名称的大小写和写法本身可能是信息。同样，我们不做全局缺失插补，因为插补统计量往往来自全表，会带来目标泄漏。对于数值列，我们只做类型强制转换和无穷值清理，其他异常保留为 NaN，让模型决定如何利用缺失模式。这个策略在小样本里更安全，因为任何人为的平滑都可能掩盖稀有信号。

### 2.3 模型训练与超参数调优

我们不做超参数搜索，采用预设的固定 CatBoost 配置，避免在验证集上反复试探。

```python
def model_fit(x, target):
    cat_cols = x.select_dtypes(include=['object','string']).columns.tolist()
    params = dict(iterations=350, depth=6, learning_rate=.08, random_seed=SEED,
                  verbose=False, thread_count=4, allow_writing_files=False)
    # binary 任务使用 Logloss
    model = CatBoostClassifier(loss_function='Logloss', **params)
    model.fit(x, target, cat_features=cat_cols)
    return model
```

数据划分采用 80/20 分层抽样，确保训练和验证集的标签比例一致。

```python
indices = np.arange(len(train))
strata = y  # binary 单标签分层
fit_idx, val_idx = train_test_split(indices, test_size=.2, random_state=SEED, stratify=strata)

X = features(train)
model = model_fit(X.iloc[fit_idx], y[fit_idx])
pred = model.predict_proba(X.iloc[val_idx])
```

我们选择 CatBoost 的原因在于它原生支持类别特征，能直接处理字符串列而不需要手工编码。参数 iterations=350、depth=6、learning_rate=0.08 都是事先写死的值，不是通过搜索得到的。这种固定配置让实验更像一个基准测试，而不是一次调参竞赛。我们还把 random_seed 固定为 20261026，保证每次运行结果一致。

为了研究小样本不确定性，我们还做了一个预设的第二实验：从特征里去掉 day 列，重新在同一份划分上训练并评估。另一个备选是缩小训练预算到约一半数据，比较模型稳定性的变化。这些都是事先写死，不是看完分数才决定。我们记录所有结果，即使去掉 day 后表现更好也不会据此改变主提交，因为可用性是部署前提。

### 2.4 模型测试与指标报告

先用常量预测做一个最简单的基准，再与 CatBoost 基线对比。

```python
def constant_prediction(target, n):
    # 常量基线：用训练集均值作为预测
    const = target.mean()
    return np.full(n, const)

baseline = constant_prediction(y[fit_idx], len(val_idx))
results = {'constant':score(y[val_idx], baseline), 'boosting':score(y[val_idx], pred)}
pd.DataFrame(results).T.to_csv(OUT/'validation_scores.csv')
```

score 函数会返回准确率、平衡准确率、对数损失、AUC、MCC 等多个指标，并在验证集上保存详细预测用于后续诊断。即使某个指标提升不大，我们也完整记录，避免只报告好的一面。验证预测会以 CSV 形式保存，方便后续检查错误模式，比如降雨被系统性低估的情况。

提交阶段我们把模型在全部采样数据上重拟一遍，再分块预测测试集，确保列顺序、id 顺序和样本数量与样例文件一致。

```python
final = model_fit(X, y)
submission = sample.copy()
offset = 0
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    p = final.predict_proba(features(chunk))
    values = p[:,1,None]  # 二分类取正类概率
    submission.loc[submission.index[offset:offset+len(chunk)], targets] = values
    offset += len(chunk)

submission.to_csv(OUT/'submission.csv', index=False)
```

分块预测是为了避免一次加载整个测试集导致内存溢出。每块预测后我们会断言当前块的 id 与 sample_submission 的对应行完全一致，防止在拼接时出现错位。最后我们会检查概率是否全部落在 0 到 1 之间，并确认没有 NaN 值存在，确保提交文件通过 Kaggle 的格式校验。

```python
final = model_fit(X, y)
submission = sample.copy()
offset = 0
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    p = final.predict_proba(features(chunk))
    values = p[:,1,None]  # 二分类取正类概率
    submission.loc[submission.index[offset:offset+len(chunk)], targets] = values
    offset += len(chunk)

submission.to_csv(OUT/'submission.csv', index=False)
```

最后我们检查 submission.csv 的列名、行数、id 唯一性和数值范围是否在 0~1 之间，并把运行清单写入 run_manifest.json，记录种子、版本、采样方式和验证方案。

## 三、总结

本文中我们完成了 Rain Signals in a Small Sample 的完整实验流程。我们首先用确定性 reservoir 采样把训练表限制在 12 万行以内，并统计了缺失情况。其次我们做了无状态的数据清洗，把非数值列统一为字符串并用 __MISSING__ 占位，数值列转为浮点。接着我们使用固定参数的 CatBoost 构建预设基线，在分层 80/20 保留集上训练，并额外做掉 day 特征的 ablation 实验来检查可用性影响。最后我们在验证集上报告了常量基线和提升模型的多个指标，并在全样本上重拟模型后分块生成提交文件并校验格式。

这个小样本练习让我们看到，即使模型本身很简单，评估方式的严谨性也决定了结论的可信度。样本小、特征有限时，先把采样、分层、固定协议写死，再谈提升，才是负责的做法。

我们还意识到，单次 20% 保留集只能算学习用的基线。真正面向角色的研究还需要说明分组结构、时间顺序、不确定性和独立评估。例如，如果数据按地点或年份聚集，随机分层会让相邻记录同时出现在训练和验证里，分数就会虚高。在我们的基线里，这一点被明确写成限制，而不是隐藏起来。

从结果呈现的角度看，常量基线虽然朴素，却能告诉我们模型是否真的学到东西。如果 CatBoost 只比常量均值好一点点，那就说明信号很弱，或者验证集太小、噪声太大。把这种对比写进报告，比单独展示一个漂亮的 AUC 更诚实，也更适合小样本场景。
