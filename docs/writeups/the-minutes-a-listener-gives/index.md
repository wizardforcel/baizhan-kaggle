# The Minutes a Listener Gives

## 一、比赛简介

### 1.1 比赛背景

The Minutes a Listener Gives 是一次建立在 Kaggle Playground S5E4 之上的独立研究。原始的 Playground 比赛叫 Podcast Listening Time Prediction，意思是「播客聆听时长预测」。比赛的组织者是 Walter Reade 和 Elizabeth Park，数据采用 Apache 2.0 协议发布。更上游的数据来自 Yash Patel 整理的同名数据集。所谓 Playground，是 Kaggle 专门给入门者练手的一系列合成比赛：数据由程序生成，规模可控，排行榜压力小，适合把完整的建模流程走一遍。

这次研究并没有沿用原比赛「冲榜」的思路，而是把重点放在方法和可复现性上。作者 Clare Laurent 重新组织了笔记本，明确固定随机种子、固定采样上限、固定模型参数，并且在看验证分数之前就把实验方案写死。这样做的好处是：我们看到的分数不是反复试出来的，而是对一套事先声明流程的检验。对于刚学机器学习的人来说，这种「先定规则、再看结果」的习惯，比多拿零点几分更重要。

数据本身是合成的，描述的是播客节目和听众行为。合成数据有一个特点：字段之间的关系是程序设定的，缺失也往往是人为注入的，所以不能把在它上面学到的规律直接当成现实世界的因果结论。材料里反复强调一句话：聆听时长的预测，不能证明是哪些内容导致了听众参与。这一点我们在最后还会回到。

### 1.2 比赛目标

任务是一个标准的回归问题。每条训练样本由若干特征列组成，其中一列是目标 `Listening_Time_minutes`，表示某位听众在某个节目上花掉的分钟数。我们要训练一个模型，读入测试样本的特征，输出一个连续的分钟数。测试集没有目标列，最终按比赛要求的格式提交 id 加预测值。

材料里的配置把这些信息写得很清楚：`targets` 只有一项 `Listening_Time_minutes`，`family` 是 `regression`，没有日期列，预设的消融字段是 `Episode_Sentiment`。训练样本并不是全部拿来用，而是设定了一个 12 万行的上限。这个上限的用途是节省内存，并不代表原始数据只有 12 万行；审计文件会同时记录「读到了多少行」和「实际用了多少行」，以免读者误以为模型见过了全部数据。

实验只使用官方提供的 train.csv 训练、test.csv 推理，额外的训练文件虽然保留在目录里，但不参与这个基线。这一点也是可复现性的一部分：说清楚哪些数据进了模型，哪些没有。

### 1.3 评价标准

这是一个回归任务，所以评价标准不是「分类对了几个」，而是「预测值和真实值差多少」。材料中的评分函数对 `regression` 家族返回 RMSE，也就是均方根误差：先把每个样本的误差平方，求平均，再开根号。它的单位和目标一样，都是分钟，所以可以直白地理解成「平均偏离多少分钟」。RMSE 越低越好。

如果任务是 `logreg` 家族，评分会换成 RMSLE，也就是先对预测值和真实值取 `log1p` 再算均方根误差，用来降低大数值样本的权重。这个比赛用的是普通 RMSE，没有走对数变换。

除了总 RMSE，材料还要求看分段误差。原因很简单：一个总平均数会把不同情况混在一起。短节目和长节目的预测难度不一样，如果模型在长节目上总是偏低，总 RMSE 可能看起来还行，但实际上存在系统性偏差。所以后面我们会按真实值的五分位切段，分别算 MAE 和平均误差，看看模型到底在哪些区间失灵。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先做的事情不是画图，而是把数据「安全地」读进来。原始 train.csv 可能很大，直接一次性读入会占用大量内存。材料采用的办法是分块读取加工确定性蓄水池采样：每次读 10 万行，为每行生成一个固定种子的随机数，然后只保留随机数最小的 12 万行。这样无论文件多大，内存里始终只有 12 万行左右，而且同样的种子每次抽到的行完全一样。

```python
from pathlib import Path
import pandas as pd
import numpy as np

MAX_ROWS = 120000                 # 内存上限：只保留 12 万行
SEED = 20261026                   # 固定种子，保证每次抽样结果一致
rng = np.random.default_rng(SEED)
reservoir = None

# 分块读取，避免一次性把大文件塞进内存
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    # 给每一行一个随机优先级，用固定种子生成
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk])
    # 只保留优先级最小的 12 万行，相当于均匀抽样
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
```

采样完成后，我们统计每个字段的缺失行数，写进 `missingness.csv`。同时生成一份审计信息，记录原始行数、采样后行数、特征列表、目标列表、采样方式，以及目录里还有哪些额外的 csv 文件没有被使用。这份审计看起来琐碎，但它是「别人能不能复现」的关键证据。

```python
audit = {
    'source_rows': rows_seen,          # 原始文件一共有多少行
    'sampled_rows': len(train),        # 实际进入模型的多少行
    'sampling': 'seeded uniform reservoir',
    'features': feature_columns,       # 除 id 和目标外的所有列
    'targets': targets,
    'source_files_unchanged': True,    # 原始文件没有被修改
}
```

接下来我们检查两个最基本的约束：id 必须唯一，目标列不能有缺失。前者防止重复样本导致验证分数虚高，后者防止我们不知不觉把没有标签的行也拿去训练。

```python
assert train[id_col].is_unique              # id 不能重复
assert not train[targets].isna().any().any()  # 目标列不能有缺失
feature_columns = [c for c in train if c not in targets + [id_col]]
```

最后我们按 8:2 切分训练集和验证集。因为这是回归任务，没有类别可以分层，所以直接随机切分，固定 `random_state`。

```python
from sklearn.model_selection import train_test_split

indices = np.arange(len(train))
fit_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=SEED)
assert set(fit_idx).isdisjoint(val_idx)   # 两边不能有重叠
```

这一步得到的验证集就是后面所有实验的唯一考场。材料特别规定，验证标签不能用于预处理、早停或选参数，否则验证分数就不再是「没见过的数据」上的分数。

### 2.2 数据清洗

这份数据的清洗原则可以概括成一句话：只做不依赖标签的变换。所谓不依赖标签，是指每一步只看特征本身，不看目标值，也不看验证集。这样就不会出现「用未来信息预测未来」的泄漏。

具体做法分两类。数值列用 `pd.to_numeric` 强制转换，把无法解析的值变成 NaN，再把正负无穷替换成 NaN，因为无穷大会让很多模型直接报错。非数值列先转成 pandas 的 string 类型，缺失填成字符串 `'__MISSING__'`，再转成普通字符串。为什么要专门造一个 `'__MISSING__'` 而不是直接填众数？因为「这个字段本来就缺失」本身可能就是信息，把它显式标出来，树模型可以自己学到一个分支。

```python
def features(frame, omit=()):
    # omit 用来做消融：临时去掉某些字段
    x = frame[[c for c in feature_columns if c not in omit]].copy()

    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):
            # 如果有日期列，就拆成年、月、星期几三个数值特征
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c+'_year'] = dt.dt.year
            x[c+'_month'] = dt.dt.month
            x[c+'_weekday'] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 类别列：缺失单独标成一个类别，保留「缺失」这个信息
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列：无法解析的转成 NaN，无穷也当成 NaN
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return x
```

这里还有一个容易忽略的点：这个函数对训练集、验证集、测试集是同一套逻辑，没有任何一步需要「先 fit 再 transform」。这种无状态变换的好处是不会因为数据切分不同而产生不同的映射，也不会把验证集的信息带进训练。

材料还明确规定，目标列不能填补。代码里用断言守住这条线：

```python
assert not train[targets].isna().any().any(), 'Do not silently impute a label.'
```

对有监督学习来说，标签缺失的行没法算损失，硬填一个猜测值等于往训练集里掺假。正确做法是要么丢掉，要么单独处理，绝不能悄悄填。

### 2.3 模型训练与超参数调优

我们使用的模型是 CatBoost 回归器。CatBoost 是梯度提升树的一种实现，通俗地说，它把很多棵浅层决策树串起来，每一棵新树都专门去修正前面所有树的残差，最后把所有树的输出加起来。它对表格数据、类别特征和缺失值都比较友好，所以常被当作表格比赛的「默认强基线」。

材料里的参数是事先固定的：350 轮迭代、树深 6、学习率 0.08，随机种子固定，线程数 4。所谓「固定」是指不看验证分数再改，这样验证分数才有意义。

```python
from catboost import CatBoostRegressor

params = dict(
    iterations=350,          # 建 350 棵树
    depth=6,                 # 每棵树最深 6 层，控制模型复杂度
    learning_rate=0.08,      # 学习率，越大越快但越容易过拟合
    random_seed=SEED,
    verbose=False,
    thread_count=4,
    allow_writing_files=False,
)

# 把字符串类型的列交给 CatBoost 当类别特征处理
cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()
model = CatBoostRegressor(loss_function='RMSE', **params)
model.fit(x, target, cat_features=cat_cols)
```

除了 boosting 模型，我们还准备了一个常数基线：不管输入是什么，都预测训练集的平均聆听时长。这个基线看起来「傻」，但它非常重要。如果一个复杂模型连「全部猜平均值」都赢不了，说明特征里根本没有可学的信号，或者流程有 bug。材料把常数基线和 boosting 放在同一张表里比较，就是为了让提升幅度一目了然。

```python
# 常数基线：所有验证样本都预测训练集均值
baseline = np.full(len(val_idx), y[fit_idx].mean())

# 同一个评分函数分别评估两个方案
results = {
    'constant': score(y[val_idx], baseline),
    'boosting': score(y[val_idx], pred),
}
```

第二个实验是预先声明的消融：把 `Episode_Sentiment` 这一列去掉，在完全相同的训练行和验证行上再训练一次。消融的意思是「拿掉一个零件，看看机器还转不转」。如果去掉之后分数几乎不变，说明这个字段贡献很小；如果明显变差，说明它有用。注意消融的结论只用于理解模型，不自动等于「可以上线时不要这个字段」。

```python
omit = [c for c in CONFIG.get('ablation', []) if c in feature_columns]
if omit:
    reduced = features(train, omit)                 # 去掉指定字段
    alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
    results['availability_ablation'] = score(
        y[val_idx], predict(alternate, reduced.iloc[val_idx]))
```

### 2.4 模型测试与指标报告

验证阶段我们先算整体 RMSE，然后按真实聆听时长分成五个区间，分别统计样本数、平均绝对误差和平均偏差。平均偏差的定义是「预测减真实」，正数表示整体高估，负数表示整体低估。这样做能回答一个总 RMSE 回答不了的问题：模型是均匀地差，还是只在某一类节目上差。

```python
diagnostics = pd.DataFrame({
    'id': train.iloc[val_idx][id_col].to_numpy(),
    'actual': y[val_idx],
    'prediction': pred,
})
diagnostics['error'] = diagnostics.prediction - diagnostics.actual
# 按真实值切成五个数量差不多的区间
diagnostics['target_band'] = pd.qcut(diagnostics.actual, 5, duplicates='drop')

diagnostics.groupby('target_band', observed=True).agg(
    rows=('error', 'size'),
    MAE=('error', lambda s: s.abs().mean()),
    bias=('error', 'mean'),
).to_csv(OUT/'error_bands.csv')
```

确认流程没问题后，我们用全部 12 万行重新训练最终模型。这里「全部」仍然是指采样后的 12 万行，不是原始全量。然后分块读取测试集，逐块预测，并逐块检查 id 顺序和提交样例完全一致。

```python
final = model_fit(X, y)

sample = pd.read_csv(DATA/'sample_submission.csv')
submission = sample.copy()
offset = 0

for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    # 测试集列顺序必须和训练集去掉目标后一致
    assert chunk.columns.tolist() == [c for c in train.columns if c not in targets]
    p = predict(final, features(chunk))
    n = len(chunk)
    submission.loc[submission.index[offset:offset+n], targets] = p
    offset += n

assert offset == len(sample)
submission.to_csv(OUT/'submission.csv', index=False)
```

最后我们做一次「回读校验」：把刚写出的 submission.csv 重新读进来，确认列名和样例完全一致、行数一致、id 顺序一致、目标列没有缺失、数值都是有限值。这些检查看起来是工程细节，但在真实比赛里，格式错误导致提交失败的情况非常常见。

关于分数，材料没有给出一个可以复述的具体 RMSE 数值，所以我们不编造。我们能确定的是：实验把常数基线和 CatBoost 的结果都写进了 `validation_scores.csv`，并且明确说明提交的是「固定全特征基线」，消融结果只作为诊断。材料还提到，排行榜第 3 名的 Johannes Heller 使用了五折 KFold、目标编码交互、多种基模型，并把 LightGBM 堆叠与爬山权重结合起来。我们的独立基线不声称复现那个成绩，这一点在写作时必须诚实。

## 三、总结

本文中我们完成了 The Minutes a Listener Gives 的完整实验流程。我们首先用固定种子的分块蓄水池采样把训练数据限制在 12 万行，并记录缺失与审计信息；其次只做无状态清洗，数值列清洗无穷值，类别列把缺失显式标成单独类别，绝不对目标做填补；接着用固定参数的 CatBoost 回归器，在 20% holdout 上与常数均值基线比较，并预先声明了移除 `Episode_Sentiment` 的消融实验；然后按真实值五分位统计 MAE 与偏差，检查模型在不同时长区间是否系统性失灵；最后在全量采样数据上重拟合，分块预测测试集，并对提交文件做了列名、行数、id 顺序和数值合法性校验。

这个练习给我们最大的启发是「约束先于结果」。固定采样、固定切分、固定参数、固定消融，看起来让模型变「笨」了，但它让每一个分数都能被解释。与此同时，材料反复提醒：预测聆听时长不等于找到让听众更投入的原因。要回答因果问题，需要随机实验或可识别的准实验设计，而不是一个回归模型。
