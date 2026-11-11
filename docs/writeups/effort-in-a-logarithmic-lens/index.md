# 对数视角下的热量消耗预测（Effort in a Logarithmic Lens）

## 一、比赛简介

### 1.1 比赛背景

这个比赛来自 Kaggle 的 Playground Series 第五季第五集（简称 S5E5）。Playground 系列的特点是：数据由 Kaggle 用生成模型"仿造"真实数据集而来，规模适中、列类型丰富，适合练习表格数据的完整流程。

S5E5 的原始来源是 "Calories Burnt Prediction" 数据集。任务是根据一次运动的生理和活动指标，预测这次运动消耗了多少卡路里。输入大致包括运动时长、心率、体温、年龄、身高、体重、性别等字段。因为数据是合成的，它比真实可穿戴设备记录的数据干净得多，但特征之间的关系仍然保留了"运动越久、心率越高，消耗越大"这类基本规律。

材料是一份名为 "Effort in a Logarithmic Lens" 的独立研究笔记，作者 Clare Laurent。它没有追求排行榜高分，而是把重点放在三件事上：**单位是否一致、损失函数是否和目标匹配、结论的适用边界在哪里**。这套思路对入门者很有参考价值。

### 1.2 比赛目标

训练表 `train.csv` 里有一列 ID、若干特征列，以及目标列 `Calories`。测试表 `test.csv` 只有特征，没有 `Calories`。提交文件 `sample_submission.csv` 只有两列：第一列是 ID，第二列是 `Calories`。

我们要做的就是给测试集的每一行填上一个预测的卡路里值。由于卡路里不可能为负，提交值还必须非负。

### 1.3 评价标准

比赛用**均方根对数误差（Root Mean Squared Logarithmic Error，RMSLE）**评分，越低越好。它的算法分三步：

1. 把预测值和真实值都做 `log1p`（先加 1 再取对数，避免 0 取对数变成负无穷）；
2. 两者相减，平方，求平均；
3. 开根号。

写成公式就是：

```
RMSLE = sqrt( mean( (log1p(pred) - log1p(actual))^2 ) )
```

用大白话理解：它衡量的是"相对误差"而不是"绝对误差"。预测 100、真实 200，和预测 1000、真实 2000，在 RMSLE 眼里错得一样多（都差一倍），但在普通 RMSE 眼里后者错得严重得多。所以这个指标特别适合"数值跨度大、大值小值都有"的目标，比如卡路里。

这也直接决定了建模方式：与其在原始尺度上 Minimize 误差，不如**先对目标取对数，在 log 空间里训练回归模型，预测完再变换回来**。这正是笔记标题里 "Logarithmic Lens"（对数视角）的意思。

---

## 二、解题过程

我们的整体路线是：读入并抽样 → 审计缺失 → 构造无泄漏特征 → 固定训练/验证划分 → 用常数基线作对照 → 训练 CatBoost 回归 → 做一次预先声明的消融实验 → 分块预测并严格校验提交文件。下面逐步展开。

### 2.1 数据探索（EDA）

我们首先确认提交模板的列结构。这一步看着简单，却能避免后面"预测错列、交上去 0 分"的低级错误：

```python
targets = CONFIG['targets']          # ['Calories']
schema = pd.read_csv(DATA / 'sample_submission.csv', nrows=3).columns.tolist()
id_col = schema[0]                   # 第一列是 ID
# 模板除 ID 外必须恰好是我们要预测的目标列
assert schema[1:] == targets
```

其次，因为原始训练表可能很大，我们采用**分块读取 + 蓄水池抽样**的方式，只保留 12 万行。直接 `read_csv` 全量读进来会占很多内存，而蓄水池抽样可以在只扫一遍文件的前提下，得到一份近似均匀的随机样本：

```python
MAX_ROWS = 120000
reservoir = None
rows_seen = 0
missing = None
rng = np.random.default_rng(SEED)

# 每块 10 万行，逐块读，避免一次性载入
for chunk in pd.read_csv(DATA / 'train.csv', chunksize=100000):
    rows_seen += len(chunk)

    # 顺便累计每一列的缺失数；后面会写成缺失审计表
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)

    # 给每行一个随机优先级，只保留优先级最小的 12 万行
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat(
        [reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
print('扫描行数:', rows_seen, '| 实际使用:', len(train))
```

这里必须强调：**抽样是为了省内存，不是在宣称"全量数据都训练过了"**。笔记里专门把 `source_rows` 和 `sampled_rows` 都记进审计文件，就是为了不让读者误以为这是全量结果。

我们同时检查了两件事：ID 是否唯一，目标是否缺失：

```python
assert train[id_col].is_unique                 # ID 不能重复，否则提交会错位
assert not train[targets].isna().any().any()   # 目标不能缺，缺了不能偷偷填补
```

目标列有缺失时，正确做法是删掉那一行或者换数据源，而不是用平均值填上——拿"编造出来的标签"训练，等于让模型学假答案。

最后我们把每列的缺失数写成 `missingness.csv`。这份表让我们知道哪些字段大面积缺失，但请注意：**我们只是记录，不在全表上做插补**。因为 CatBoost 本身能处理缺失值，而在全表上算均值/中位数再填，会把验证集的信息泄漏进训练。

### 2.2 数据清洗

这份数据的清洗原则是"**无状态变换**"：每一步只依赖当前这一行，绝不使用全表的统计量。这样训练集和测试集可以走同一段代码，不会出现"用测试集信息预处理训练集"的泄漏。

具体做法集中在 `features()` 函数里：

```python
def features(frame, omit=()):
    """把原始表变成模型能吃的特征；omit 用于消融实验，临时去掉某些列"""
    x = frame[[c for c in feature_columns if c not in omit]].copy()

    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):
            # 有日期列时拆成 年/月/星期，而不是把日期字符串直接丢给模型
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c + '_year'] = dt.dt.year
            x[c + '_month'] = dt.dt.month
            x[c + '_weekday'] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 类别列统一转字符串，缺失单独标成 __MISSING__，不猜成某一类
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列强制转数字，无穷大改成 NaN，交给 CatBoost 处理
            x[c] = pd.to_numeric(x[c], errors='coerce').replace(
                [np.inf, -np.inf], np.nan)
    return x
```

这里有几个设计理由：

- **类别缺失单独成类**。把缺失填成出现次数最多的类别，等于告诉模型"这个样本属于那一类"，这是假信息；标成 `__MISSING__`，模型能学到"缺失本身可能和结果有关"。
- **无穷大转 NaN**。物理量里出现 `inf` 通常是除法出错，不是真实观测，不能当极端值喂给模型。
- **日期拆字段**。日期字符串对树模型没有直接意义，拆成年、月、星期才有用。这个比赛 `date_columns` 是空列表，所以这段逻辑没有实际触发，但保留它是为了让同一份代码能复用到其他 Playground 赛题。

### 2.3 模型训练与超参数调优

我们首先固定一个 **20% 的留出验证集**。对回归任务不做分层抽样，直接随机切：

```python
indices = np.arange(len(train))
fit_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=SEED, stratify=None)

# 训练索引和验证索引不能有交集，否则验证分数没有意义
assert set(fit_idx).isdisjoint(val_idx)
```

这个划分**在查看任何验证分数之前就定死了**。笔记里反复强调 "prespecified"（预先声明）：先定实验，再看结果。否则人很容易根据分数反复改划分、改参数，最后把验证集"调"成一个好看的数字。

然后我们训练一个**常数基线**：把所有验证样本都预测成训练集目标的平均值。这是最低标准——如果 CatBoost 连"所有样本都猜平均值"都赢不了，说明特征或代码有问题：

```python
def constant_prediction(target, n):
    # 在 log 空间求均值，再变换回原始尺度，保证和 RMSLE 的视角一致
    const = np.expm1(np.log1p(target).mean())
    return np.full(n, const)

baseline = constant_prediction(y[fit_idx], len(val_idx))
```

正式模型用 CatBoost 回归。关键设置是：**目标先取 log1p，损失仍用 RMSE**。因为 RMSE 在 log 空间里最小化，等价于在原始空间里最小化 RMSLE：

```python
def model_fit(x, target):
    cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()
    params = dict(
        iterations=350,        # 350 棵树，固定值，不在排行榜上反复试
        depth=6,               # 深度 6，兼顾拟合能力和速度
        learning_rate=0.08,
        random_seed=SEED,
        verbose=False,
        thread_count=4,
        allow_writing_files=False,
    )
    model = CatBoostRegressor(loss_function='RMSE', **params)
    # 注意：训练时喂 log1p(y)，预测后再 expm1 变回来
    model.fit(x, np.log1p(target), cat_features=cat_cols)
    return model

def predict(model, x):
    # expm1 是 log1p 的逆运算；再截断到 0，保证卡路里不为负
    return np.maximum(np.expm1(model.predict(x)), 0)
```

为什么用 `log1p` 而不是 `log`？因为卡路里有可能接近 0，直接取 `log(0)` 会变成负无穷。`log1p(x) = log(1+x)` 在 0 附近仍然有定义。需要提醒的是：`log1p` 只是一个**数值变换**，它和"物理上的幂律关系"不是一回事——如果把 `log1p` 的结果当成严格的幂律模型去解释弹性系数，在接近 0 的地方会失真。这一点是笔记在"中级"讨论里特意区分开的。

关于超参数调优，我们的态度比较克制：`iterations=350`、`depth=6`、`learning_rate=0.08` 是**事先固定**的，没有用排行榜分数去反推。原因是这个比赛的数据是合成的，排行榜和本地验证的分布未必一致，盯着榜单调参很容易过拟合公共榜。我们更愿意把有限的时间花在"实验设计是否公平"上。

为此我们做了第二组实验：**去掉 `Body_Temp` 这一列，在完全相同的训练/验证行上重训**。这是一次"可用性消融"（availability ablation），用来回答"模型有多依赖体温这个字段"：

```python
omit = [c for c in CONFIG.get('ablation', []) if c in feature_columns]

if omit:
    reduced = features(train, omit)
    alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
    alt_pred = predict(alternate, reduced.iloc[val_idx])
    results['availability_ablation'] = score(y[val_idx], alt_pred)
```

请注意：消融是**诊断**，不是"发现去掉更好就删掉它"。如果真实场景里体温拿不到，那才需要考虑替代方案；如果只是本地分数高一点，不能成为删除一个有意义字段的理由。笔记里也明确写了 "availability ablations are diagnostics, not permission to deploy unavailable fields"。

### 2.4 模型测试与指标报告

本地评估严格按 RMSLE 计算：

```python
from sklearn.metrics import mean_squared_log_error

def score(truth, p):
    # 先保证预测非负，再套 RMSLE 公式
    return {'RMSLE': float(np.sqrt(mean_squared_log_error(truth, np.maximum(p, 0))))}

results = {
    'constant': score(y[val_idx], baseline),
    'boosting': score(y[val_idx], predict(model, X.iloc[val_idx])),
}
print(results)
```

我们把常数基线和 CatBoost 两行结果都保留下来，写进 `validation_scores.csv`。即使消融实验的结果更好，也照样保留，不挑好看的报。这是笔记的一个核心态度：**诊断结果要完整公开，不能只晒赢的那一行**。

除了总分，我们还按真实值的分位数把验证集切成五段，看每一段的平均绝对误差和平均偏差：

```python
diagnostics = pd.DataFrame({
    'id': train.iloc[val_idx][id_col].to_numpy(),
    'actual': y[val_idx],
    'prediction': pred,
})
diagnostics['error'] = diagnostics.prediction - diagnostics.actual
diagnostics['target_band'] = pd.qcut(diagnostics.actual, 5, duplicates='drop')

# 按真实值高低分段，看模型是不是"小值准、大值飘"
diagnostics.groupby('target_band', observed=True).agg(
    rows=('error', 'size'),
    MAE=('error', lambda s: s.abs().mean()),
    bias=('error', 'mean'),
).to_csv(OUT / 'error_bands.csv')
```

这一步常常能暴露总分掩盖的问题：比如高卡路里段系统性低估，说明模型在极端值上被"拉平均"了；低卡路里段偏差大，可能是 log 变换在接近 0 时的数值问题。

确认方案可用后，我们用全部抽样数据重新拟合，再对测试集做**分块预测**。分块不只是为了省内存，也是为了逐块校验 ID 顺序：

```python
final = model_fit(X, y)
sample = pd.read_csv(DATA / 'sample_submission.csv')
submission = sample.copy()
offset = 0

for chunk in pd.read_csv(DATA / 'test.csv', chunksize=100000):
    n = len(chunk)
    # 测试块的特征列必须和训练时一致，顺序也要一致
    assert chunk.columns.tolist() == [c for c in train.columns if c not in targets]
    # 这一块的 ID 必须和模板对应位置完全一致
    assert np.array_equal(chunk[id_col].to_numpy(),
                          sample.iloc[offset:offset + n, 0].to_numpy())

    p = predict(final, features(chunk))
    submission.loc[submission.index[offset:offset + n], 'Calories'] = p
    offset += n

# 行数、ID、非负、有限值，四项都检查
assert offset == len(sample) and submission[id_col].is_unique
assert not submission[targets].isna().any().any()
assert np.isfinite(submission[targets].to_numpy(dtype=float)).all()
assert (submission['Calories'] >= 0).all()
```

最后把提交文件重新读回来，做一次"往返校验"：列名和模板完全一致、行数一致、ID 逐一相等。很多比赛里，预测做得再好，提交格式错一列就是 0 分，这一步花几秒钟，能省掉一次白跑。

关于成绩，材料里的笔记只给出了流程，没有记录本次实验的具体 RMSLE 数值，所以我们不在这里编造。作为外部参照，笔记引用了该比赛公开分享的第一名方案：Chris Deotte 用 GPU 爬山从几百个模型里选出七个互补模型，结合对数变换、成对运算、分箱类别、分组 z-score 和残差学习，报告 CV RMSLE 0.05880、private 0.05841。这是作者的比赛发现，不是通用配方；我们的笔记是独立基线，并不声称复现这个分数。

---

## 三、总结

本文中我们完成了 S5E5 卡路里预测比赛的一次完整基线。我们首先从提交模板确认了 ID 和 `Calories` 两列的结构，用分块读取加 12 万行蓄水池抽样的方式控制内存，并审计了每列缺失；其次用无状态的特征函数处理类别、数值和日期，缺失单独成类而不做全表插补；接着固定 20% 留出集，先算常数基线，再用 CatBoost 对 `log1p(Calories)` 做回归、预测后 `expm1` 变回原尺度并截断非负，另做了一次去掉 `Body_Temp` 的消融实验；最后按 RMSLE 报告常数基线、 boosting 和消融三组结果，按真实值分位检查误差分布，分块预测并逐项校验提交文件。

这个比赛让我们记住一句话：**指标决定变换，变换决定解释**。因为评分是 RMSLE，我们在 log 空间训练；但 log 变换只是让数值更"听话"，不能反过来把它当成物理规律。把实验预先声明、把失败的结果也保留下来，比多刷 0.001 分更有价值。
