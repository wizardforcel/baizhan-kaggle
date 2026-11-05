# 相对尺度上的保费预测

## 一、比赛简介

### 1.1 比赛背景

这个比赛来自 Kaggle Playground 系列的第 4 季第 12 集（S4E12），题目是预测保险保费的金额（`Premium Amount`）。Playground 系列是 Kaggle 官方推出的一类练习赛：数据由合成生成器批量产生，规模大、字段多、噪声也不小，但结构接近真实业务表格。它的目的不是解决某一个具体行业难题，而是让参赛者在接近实战的数据上练习完整的建模流程——读数据、处理缺失、做特征、选模型、看误差。

保险定价有一个非常现实的特点：**同样的绝对误差，对便宜保单和昂贵保单的意义完全不同**。如果一份保单的真实保费是 100 元，你预测成 200 元，误差 100 元，相对误差是 100%；如果另一份保单真实保费是 10000 元，你预测成 10100 元，误差也是 100 元，相对误差只有 1%。从业务角度看，前者是灾难，后者几乎可以接受。所以评估保费模型时，不能只看绝对误差，而要看**相对误差**，这也正是本比赛使用对数尺度指标的原因。

本笔记本的作者特别强调，这是一项独立的教育性研究：它使用固定、事先指定的模型设置，不根据排行榜分数反复调参，目的是展示一个诚实、可复现的基线，而不是追求名次。作者还明确提醒，预测出来的保费不能作为真实保险报价或理赔成本的依据，因为合成数据和真实精算数据之间隔着巨大的鸿沟。这种「把限制说清楚」的态度，本身就是数据科学工作的一部分。

### 1.2 比赛目标

输入是一张包含投保人信息、保单信息等字段的表格，输出是每份保单的保费金额 `Premium Amount`。提交文件的第一列是 ID，第二列是目标列。我们从 `sample_submission.csv` 读出列名来确认目标列，而不是把列名写死在代码里：

```python
# 从提交样例中读出 ID 列名和目标列名，防止写死列名
schema = pd.read_csv(DATA / 'sample_submission.csv', nrows=3).columns.tolist()
id_col = schema[0]              # 第一列是样本编号
assert schema[1:] == targets    # targets = ['Premium Amount']
```

训练集很大，为了控制内存，作者设定最多使用 12 万行，并且用固定随机种子做均匀抽样。也就是说，我们拟合的并不是全部训练数据，而是一个有界的、可复现的子集。作者在文档里明确写了这一点，没有假装自己用了全量数据——这也是诚实研究的一部分：抽样上限是为了省内存，不代表所有可用数据都被用上了。

### 1.3 评价标准

比赛使用均方根对数误差 RMSLE（Root Mean Squared Logarithmic Error）。它的计算方式是：对真实值和预测值都先做 `log1p` 变换，也就是 `log(1+x)`，然后算差值、平方、求平均、再开方。写成公式就是 RMSLE 等于「`log1p` 预测值与 `log1p` 真实值之差的平方的平均」的平方根。

为什么用 `log1p` 而不是直接 `log`？因为保费可能为 0，而 `log(0)` 是负无穷，没法计算；`log1p(0)` 恰好等于 0，有良好定义。这也是为什么后面训练模型时，我们直接在 `log1p` 空间里回归——这样训练目标和评价指标就是同一件事，模型优化的方向不会跑偏。

RMSLE 越小越好。它对**相对误差**敏感：同样差 100 元，在小额保单上会被放大，在大额保单上会被缩小。此外，因为取了 log，预测偏高和偏低在指标上比 RMSE 更对称，不会像平方误差那样对高估和低估给出极不对称的惩罚。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先以分块的方式读取训练集，一边读一边统计缺失情况，并用固定随机种子的「蓄水池抽样」把数据量限制在 12 万行以内：

```python
MAX_ROWS = 120000
reservoir = None
rows_seen = 0
missing = None
rng = np.random.default_rng(SEED)

# 每块 10 万行地读，避免一次性把大文件塞进内存
for chunk in pd.read_csv(DATA / 'train.csv', chunksize=100000):
    rows_seen += len(chunk)                       # 记录原始总行数
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)

    # 给每一行分配一个随机优先级，最后只保留优先级最小的 12 万行
    # 这样得到的是均匀随机子集，不会偏向某一段数据
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
```

这样做的原因很实际：完整训练集很大，普通笔记本内存吃不消；而均匀抽样不会系统性地偏向某一类样本，比「只读前 12 万行」更公平。抽样完成之后，我们做了几条硬性断言：ID 必须唯一，目标列不能有缺失。

```python
# 硬性检查：ID 唯一、标签不缺失
assert train[id_col].is_unique
assert not train[targets].isna().any().any(), 'Do not silently impute a label.'
```

这条断言看起来很苛刻，但很有必要：标签缺失和特征缺失是两码事。特征缺失可以用中位数、众数或者让树模型自己处理，标签缺失却没有任何可靠办法补——你不可能凭空知道一份保单的真实保费。悄悄用均值填标签，只会让模型学到假答案。

我们把每列的缺失数量单独存成 `missingness.csv`，方便复盘。然后看目标 `Premium Amount` 的分布和零值比例：

```python
# 描述统计：均值、中位数、最大值，以及零保费占比
print(train['Premium Amount'].describe())
print('零值比例:', (train['Premium Amount'] == 0).mean())

# 直方图直观展示右偏
plt.hist(train['Premium Amount'], bins=50, edgecolor='black')
plt.xlabel('Premium Amount')
plt.ylabel('Count')
plt.title('Distribution of Premium Amount')
plt.show()
```

结果显示目标明显右偏：大部分保单集中在中低金额，少数高额保单把右尾拉得很长，同时还存在零保费样本。这解释了为什么必须用 `log1p`：一方面零值需要被定义，另一方面对数变换能把长尾压平，让模型更关注相对差异而不是绝对差异。

我们还查看了日期字段 `Policy Start Date`。它记录保单的起始日期，可能隐含季节、年份等业务规律，所以我们后续把它拆成年、月、星期几三个数值特征。作者在实验设计里把这一列列为「可得性消融」的对象：在某些真实部署场景里，保单起始日期可能在预测时还拿不到，所以需要单独看看去掉它之后模型会怎样。

### 2.2 数据清洗

清洗采用「无状态变换」：每一行只看自己，不跨行统计目标信息，避免验证集信息泄漏。具体来说，日期列拆成年、月、星期几；非数值列转成字符串并把缺失填成一个特殊标记 `__MISSING__`；数值列统一转成数字，并把正负无穷替换成缺失：

```python
def features(frame, omit=()):
    # omit 用于消融实验：临时去掉某些列
    x = frame[[c for c in feature_columns if c not in omit]].copy()

    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):
            # 日期列：拆成三个简单数值特征，树模型可以直接用
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c + '_year'] = dt.dt.year
            x[c + '_month'] = dt.dt.month
            x[c + '_weekday'] = dt.dt.dayofweek

        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 缺失本身可能携带信息，用专门的字符串标记代替直接丢弃
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)

        else:
            # 把无法解析的值和无穷大统一变成 NaN，交给模型处理
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)

    return x
```

我们没有做全表填补，也没有做目标编码（用目标均值去替换类别）。原因有两点：第一，CatBoost 这类树模型能直接处理缺失值，把缺失交给模型去学，通常比强行填一个统计量更稳；第二，目标编码如果不在训练折内拟合，会把验证集的标签信息泄漏进特征，线下分数会虚高。作者的整个设计都强调「验证标签不参与预处理、早停或参数选择」，这条原则贯穿始终。

### 2.3 模型训练与超参数调优

我们把数据固定切成 80% 训练、20% 验证，并且只把训练部分的标签用于拟合、早停和参数选择。模型使用 CatBoost 回归器，参数事先固定，不在验证集上反复挑：

```python
def model_fit(x, target):
    # 把字符串列当作类别特征交给 CatBoost，省去手动独热编码
    cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()

    params = dict(
        iterations=350,        # 最多 350 轮提升
        depth=6,               # 每棵树深度 6
        learning_rate=.08,     # 学习率
        random_seed=SEED,
        verbose=False,
        thread_count=4,
        allow_writing_files=False
    )

    model = CatBoostRegressor(loss_function='RMSE', **params)
    # 关键：在 log1p 空间训练，与 RMSLE 指标对齐
    model.fit(x, np.log1p(target), cat_features=cat_cols)
    return model

def predict(model, x):
    # 预测出来的是 log1p 空间的值，用 expm1 还原，并保证非负
    p = model.predict(x)
    return np.maximum(np.expm1(p), 0)
```

为什么用 `expm1` 而不是 `exp`？因为训练时用的是 `log1p(y)`，它的反函数是 `exp(z)-1`，而 `expm1(z)` 正好等于 `exp(z)-1`，并且在 z 接近 0 时数值更稳定；如果直接用 `exp(z)`，会凭空多出 1 的偏移，小额保费的预测会整体偏高。

为什么参数是固定的？因为作者把「不根据排行榜调参」当成实验设计的一部分。模型的效果应该在看到分数之前就由预设决定，否则你其实是在用测试集做模型选择。350 轮、深度 6、学习率 0.08 这组值不算激进，属于稳妥的基线配置。

除了主模型，我们还做了一个可证伪的对照实验：在完全相同的训练行和验证行上，移除 `Policy Start Date` 这一列再训练一次，看看去掉这个可能在实际业务中拿不到的字段后结果如何变化：

```python
# 消融实验：在相同划分上，用去掉指定列的特征重新训练
omit = [c for c in CONFIG.get('ablation', []) if c in feature_columns]
if omit:
    reduced = features(train, omit)
    alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
    alt_pred = predict(alternate, reduced.iloc[val_idx])
    results['availability_ablation'] = score(y[val_idx], alt_pred)
```

作者强调，这种「可得性消融」只是诊断，不等于可以放心地把不可用字段部署上线。如果去掉某个字段后分数掉很多，说明模型在真实场景里可能不可用；如果掉得很少，也不能反过来证明该字段在业务上不重要。

### 2.4 模型测试与指标报告

我们先在验证集上计算两组分数：一组是「只用训练集对数均值」的常数基线，一组是 CatBoost：

```python
def constant_prediction(target, n):
    # logreg 情形下的常数基线：先取 log1p 均值，再反变换回原尺度
    const = np.expm1(np.log1p(target).mean())
    return np.full(n, const)

baseline = constant_prediction(y[fit_idx], len(val_idx))
model = model_fit(X.iloc[fit_idx], y[fit_idx])
pred = predict(model, X.iloc[val_idx])

results = {
    'constant': score(y[val_idx], baseline),
    'boosting': score(y[val_idx], pred)
}
```

其中 `score` 对 logreg 家族返回的就是 RMSLE。常数基线的意义在于给你一个「什么都不学」的下限：如果 CatBoost 连它都赢不了，说明特征里根本没有可用信号；如果能赢，赢多少就是模型真正带来的价值。

只看一个总分还不够。我们还按真实保费的五分位区间切分，统计每个区间的样本数、平均绝对误差和平均偏差：

```python
diagnostics = pd.DataFrame({
    'id': train.iloc[val_idx][id_col].to_numpy(),
    'actual': y[val_idx],
    'prediction': pred
})
diagnostics['error'] = diagnostics.prediction - diagnostics.actual

# 按真实值做五分位分箱（duplicates='drop' 防止大量零值导致分箱失败）
diagnostics['target_band'] = pd.qcut(diagnostics.actual, 5, duplicates='drop')

diagnostics.groupby('target_band', observed=True).agg(
    rows=('error', 'size'),
    MAE=('error', lambda s: s.abs().mean()),
    bias=('error', 'mean')
).to_csv(OUT / 'error_bands.csv')
```

这个表非常有用。总 RMSLE 好看，不代表在便宜保单上就没有系统性高估或低估。如果最低价位档的偏差是正数且很大，说明模型在小额保单上系统性高估；如果最高价位档的 MAE 远高于其他档，说明模型在高价段不可靠。只有分区间看清楚，才知道模型能不能真的拿去用。

最后我们在 12 万行抽样上重新拟合全特征模型，分块读测试集、逐块断言列顺序与样例一致，再写出提交文件：

```python
final = model_fit(X, y)
submission = pd.read_csv(DATA / 'sample_submission.csv')
offset = 0

for chunk in pd.read_csv(DATA / 'test.csv', chunksize=100000):
    # 断言测试块列和训练特征完全对应，防止列错位
    assert chunk.columns.tolist() == [c for c in train.columns if c not in targets]
    n = len(chunk)

    # 用同一个 predict 函数，保证 log1p 反变换一致
    p = predict(final, features(chunk))
    submission.loc[submission.index[offset:offset+n], 'Premium Amount'] = p
    offset += n

# 提交前做往返校验：列名、行数、ID 顺序、无缺失、数值有限
assert offset == len(sample) and submission[id_col].is_unique
assert not submission[targets].isna().any().any()
assert np.isfinite(submission[targets].to_numpy(dtype=float)).all()
submission.to_csv(OUT / 'submission.csv', index=False)
```

每次运行还会把配置、随机种子、库版本、运行时长写进 `run_manifest.json`，并记录源文件的 SHA-256。这样别人拿到笔记本，可以确认我们用的是同一份数据、同样的设置，结果可复现。

作者在文末提到，排行榜第一名 Chris Deotte 的报告里用了单个 XGBoost 加 611 个特征、类别交互、嵌套折的目标编码与计数编码、保单日期分解和 GPU 加速特征搜索。这些是原作者在竞赛中的经验，不代表我们可以复现，也不应该直接照搬；我们的目标是把一个诚实基线讲清楚，而不是复刻冠军方案。

## 三、总结

本文中我们完成了相对尺度保费预测的完整研究。我们首先用分块读取加固定种子蓄水池抽样，把数据控制在 12 万行并把缺失统计落盘，同时用断言保证 ID 唯一、标签不缺失；其次采用只依赖当前行的无状态特征变换，把日期拆成年度/月度/星期，把文本缺失保留成显式标记，避免任何跨行统计导致的验证泄漏；接着用固定超参数的 CatBoost 在 `log1p` 空间回归，并用 `expm1` 加非负裁剪还原预测，另外做了去掉保单起始日期的可得性消融对照；然后在固定 20% 留出集上与常数基线对比 RMSLE，并按保费五分位报告误差与偏差；最后分块预测测试集、校验提交格式并保存运行清单。

这个比赛让我们理解了三件事。第一，**评价指标决定目标变换**：既然比赛用 RMSLE，就应该在 `log1p` 空间训练，并记得用 `expm1` 反变换，否则预测会整体偏移。第二，**诚实的研究要控制变量**：固定超参数、固定划分、事先声明对照实验，才能让分数说明问题，而不是让调参技巧掩盖问题。第三，**隐私与可得性要提前想**：作者把「某些字段在部署时可能拿不到」作为消融实验单独提出来，这种思维方式比多刷零点几分更有价值。对我们初学者来说，这篇 notebook 是一个很好的范本：它没有炫技，却把抽样、缺失、目标变换、误差分档和可复现性这些工程细节都讲清楚了。
