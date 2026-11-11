# 带解释的价格预测

## 一、比赛简介

### 1.1 比赛背景

这个比赛是 Kaggle Playground 系列第 4 季第 9 集（S4E9），任务是根据二手车的登记信息预测它的价格。Playground 的数据来自合成生成器，字段结构和真实二手车平台很像：品牌、车型、年份、里程、燃料类型、发动机描述、变速箱、颜色、事故记录、产权状态等，目标列是 `price`。

笔记本作者把它写成一个「价格审计」练习：一位二手车分析师不仅要给出一辆车看起来合理的估价，还要能说明这个估价为什么可信、在哪些价位上不可信。这个角度比单纯刷 RMSE 更有意思，因为真实世界的二手车数据有两个棘手问题。

第一，**价格是重尾分布**。绝大多数车集中在几万块的区间，少数豪车价格极高。平方误差会把这些豪车的残差放大得不成比例，模型为了压低总误差，可能牺牲普通车的精度，或者干脆不敢预测高价。所以光看一个总体 RMSE 会掩盖问题，必须按价位分档看误差。

第二，**描述字段是非结构化字符串**。比如发动机一栏可能写成 `300.0HP 3.0L V6`，把马力、排量和气缸布局混在一起；也可能是 `unknown`，甚至对电动车来说根本没有排量。如果直接把它当类别处理，信息的利用率很低；如果粗暴地用空格切分，又会把数字和单位切错。因此必须写一个带单位约束的解析器，并保留「缺失」和「字段不适用」的区别。

作者反复声明，这是独立的教育性研究，模型设置事先固定，不根据排行榜调参；预测出来的交易价格既不是认证估价，也不能解释为某个车辆特征的因果效应。这种克制很重要：一辆车贵，可能是因为品牌溢价，也可能是因为车况好，模型学到的相关性不等于因果关系。

### 1.2 比赛目标

输入是一张二手车信息表，输出是每辆车的成交价格 `price`。提交文件的第一列是 ID，第二列是 `price`。我们从提交样例读取列名，确认目标列：

```python
# 从提交样例读取列名，确认 ID 列与目标列
schema = pd.read_csv(DATA / 'sample_submission.csv', nrows=3).columns.tolist()
id_col = schema[0]              # 第一列是样本编号
assert schema[1:] == targets    # targets = ['price']
```

和另一篇同样来自 Playground 的研究一样，作者把训练数据限制在最多 12 万行，使用固定种子的均匀抽样。这样做的原因不是数据少，而是内存有限；作者在文档里明确声明「这不是说所有训练行都被拟合了」，避免读者误以为用了全量数据。

### 1.3 评价标准

比赛用均方根误差 RMSE 评价，数值越低越好。RMSE 的算法是「误差平方的平均值再开根号」，所以它对大误差特别敏感，单位又和价格一致，可以直接理解成「平均差多少钱」。

但正如背景里说的，价格重尾，单看 RMSE 容易被少数豪车带偏。所以作者除了报告整体 RMSE，还按真实价格的价值区间切分，分别报告平均绝对误差（MAE）和平均偏差（bias）。MAE 对极端值没那么敏感，能反映普通车的典型误差；bias 的符号则告诉我们模型是系统性高估还是低估。把这三个量放在一起看，才能区分「平均误差变大」和「相对误差变差」这两件不同的事。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先分块读入训练集，统计每一列的缺失数量，并把缺失表保存下来。读取时使用固定种子的蓄水池抽样，把最多 12 万行留在内存里：

```python
MAX_ROWS = 120000
reservoir = None
rows_seen = 0
missing = None
rng = np.random.default_rng(SEED)

for chunk in pd.read_csv(DATA / 'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)

    # 随机优先级，最后保留优先级最小的 12 万行，得到均匀子集
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
```

我们断言每辆车的 ID 唯一，并且 `price` 没有缺失——标签缺失不能靠插值糊弄过去。

接着查看价格分布。它明显右偏：绝大多数车集中在较低价位，少数豪华车把右尾拉得很长。这提醒我们，后面按价格分箱看误差是必要的。我们还查看了 `engine` 列的原始写法：

```python
# 抽样看发动机字段的原始写法，了解需要解析哪些模式
print(train['engine'].sample(20, random_state=SEED).tolist())
```

输出里能看到类似 `300.0HP 3.0L V6`、`unknown` 这样的取值，也有只写了马力或只写了排量的情况。这说明整体当类别会浪费信息，用空格切分又会把单位弄错，所以必须写解析器。

我们还检查了日期字段。这个数据集在配置里把 `date_columns` 设为空，说明作者没有需要拆解的日期列；但我们保留了这个通用分支，方便其他 Playground 比赛复用同一套代码。类别字段则统一转成字符串，把缺失填成 `__MISSING__`，这样「缺失」本身成为一个独立类别，而不是被悄悄丢掉。

### 2.2 数据清洗

清洗的核心是「无状态变换」：每一行只看自己，不跨行学习统计量，防止验证集信息泄漏。日期列拆成年、月、星期；文本列缺失填特殊标记；数值列统一类型并处理无穷：

```python
def features(frame, omit=()):
    # omit 用于消融实验：临时去掉某些列
    x = frame[[c for c in feature_columns if c not in omit]].copy()

    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c + '_year'] = dt.dt.year
            x[c + '_month'] = dt.dt.month
            x[c + '_weekday'] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 用显式标记保留「缺失」这一事实
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)

    return x
```

针对 `engine` 列，我们用带单位约束的正则分别抽出 horsepower 和排量，并写了一条自测断言，确保 `300.0HP 3.0L V6` 能正确解析出 `300.0`：

```python
if CONFIG['episode'] == 's4e9' and 'engine' in x:
    # 只抓 "数字+HP" 和 "数字+L"，避免把年份、里程等其他数字误当马力或排量
    x['engine_hp'] = x['engine'].str.extract(r'([0-9.]+)HP', expand=False).astype(float)
    x['engine_litres'] = x['engine'].str.extract(r'([0-9.]+)L', expand=False).astype(float)

    # 用一个已知样例做夹具，解析错了立刻报错，而不是悄悄产生错误特征
    assert pd.Series(['300.0HP 3.0L V6', 'unknown']).str.extract(
        r'([0-9.]+)HP', expand=False).iloc[0] == '300.0'
```

这里的正则 `([0-9.]+)HP` 只匹配「数字后面紧跟 HP」，所以 `300.0HP` 能被抽出来，而 `3.0L` 不会被误抓；第二行 `unknown` 匹配不到，结果就是缺失。作者特别提醒，电动车没有排量是「字段不适用」，不是「格式错误」，所以解析失败时保留缺失并把原文留着，而不是硬塞一个 0。塞 0 会让模型误以为电动车是「零排量燃油车」，反而制造噪声。

### 2.3 模型训练与超参数调优

我们固定划分 20% 作为留出集，训练集占 80%，且验证标签不参与预处理、早停或参数选择。模型用 CatBoost 回归，参数事先写死：

```python
def model_fit(x, target):
    # 字符串列交给 CatBoost 当类别特征处理
    cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()

    params = dict(
        iterations=350,
        depth=6,
        learning_rate=.08,
        random_seed=SEED,
        verbose=False,
        thread_count=4,
        allow_writing_files=False
    )

    model = CatBoostRegressor(loss_function='RMSE', **params)
    model.fit(x, target, cat_features=cat_cols)   # 直接优化 RMSE
    return model
```

为什么选 CatBoost？因为二手车数据里类别字段多（品牌、车型、颜色、变速箱等），而 CatBoost 对类别特征有专门的处理机制，不用我们手动做独热编码，也不会因为类别取值太多而维度爆炸。参数方面，350 轮、深度 6、学习率 0.08 是比较保守的配置，适合作为一个可复现基线。

我们没有在排行榜上反复试探，而是把「全特征 + 固定参数」作为提交模型。第二个实验是事先规定的消融：在同样的训练行和验证行上，去掉 `engine` 列重新训练，观察解析出来的发动机信息到底有没有帮助：

```python
# 消融实验：去掉 engine 列，在完全相同的数据划分上重训
omit = [c for c in CONFIG.get('ablation', []) if c in feature_columns]
if omit:
    reduced = features(train, omit)
    alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
    alt_pred = predict(alternate, reduced.iloc[val_idx])
    results['availability_ablation'] = {'RMSE': float(np.sqrt(
        mean_squared_error(y[val_idx], alt_pred)))}
```

作者强调，这种对照只是诊断，不是「字段不可得也能部署」的许可证。如果去掉发动机后分数明显变差，说明这个字段很有价值；如果没差多少，也不能说发动机不重要，可能只是它提供的信息被其他字段（比如车型）替代了。

### 2.4 模型测试与指标报告

验证阶段我们同时计算常数基线（训练集价格均值）和 CatBoost 的 RMSE：

```python
X = features(train)

# 常数基线：不管输入什么，都预测训练集价格均值
baseline = np.full(len(val_idx), y[fit_idx].mean())

# 全特征 CatBoost
model = model_fit(X.iloc[fit_idx], y[fit_idx])
pred = predict(model, X.iloc[val_idx])

results = {
    'constant': {'RMSE': float(np.sqrt(mean_squared_error(y[val_idx], baseline)))},
    'boosting': {'RMSE': float(np.sqrt(mean_squared_error(y[val_idx], pred)))}
}
```

常数基线的意义在于给出「什么都不学」的下限。如果 CatBoost 赢不了均值，说明特征里没有可用信号；如果能赢，赢多少就是模型的价值。

仅看一个 RMSE 不够，我们按真实价格做五分位分箱，统计每档的样本数、平均绝对误差和平均偏差：

```python
diagnostics = pd.DataFrame({
    'id': train.iloc[val_idx][id_col].to_numpy(),
    'actual': y[val_idx],
    'prediction': pred
})
diagnostics['error'] = diagnostics.prediction - diagnostics.actual
diagnostics['target_band'] = pd.qcut(diagnostics.actual, 5, duplicates='drop')

diagnostics.groupby('target_band', observed=True).agg(
    rows=('error', 'size'),
    MAE=('error', lambda s: s.abs().mean()),
    bias=('error', 'mean')
).to_csv(OUT / 'error_bands.csv')
```

这个表能回答「是不是少数豪车把误差撑大了」：如果最高价位档的 MAE 远高于其他档，就说明模型在高价段不可靠；如果偏差符号一致，就说明存在系统性高估或低估。举个例子，如果最低价位档 bias 为正、最高价位档 bias 为负，那就是典型的「向中间收缩」：模型不敢预测太便宜或太贵的车，全都往均值靠。

最后用全特征模型在抽样数据上重训，分块预测测试集，并做提交文件的往返校验：

```python
final = model_fit(X, y)
sample = pd.read_csv(DATA / 'sample_submission.csv')
submission = sample.copy()
offset = 0

for chunk in pd.read_csv(DATA / 'test.csv', chunksize=100000):
    # 断言测试块列与训练特征一致，防止列错位
    assert chunk.columns.tolist() == [c for c in train.columns if c not in targets]
    n = len(chunk)
    p = predict(final, features(chunk))
    submission.loc[submission.index[offset:offset+n], 'price'] = p
    offset += n

# 写完后再读回来做往返校验，确认行列和数值都没问题
assert offset == len(sample) and submission[id_col].is_unique
assert np.isfinite(submission['price'].to_numpy(dtype=float)).all()
submission.to_csv(OUT / 'submission.csv', index=False)
```

作者在文末提到，排行榜第一名的报告用了「异常价格分类器作为额外特征、折内中位数目标编码、SVR/LightGBM/分类器/XGBoost 的神经堆叠」等做法。这些是原作者的经验，不是我们必须复现的配方；而且其中一些技巧（比如目标编码）如果实现不严谨，很容易泄漏。我们的目标是把一个可解释、可复现的基线讲清楚。

## 三、总结

本文中我们完成了二手车价格预测的价格审计式研究。我们首先用分块读取加蓄水池抽样控制内存，统计每列缺失并把缺失表落盘；其次采用无状态特征变换，专门为 `engine` 写了带单位约束的正则解析器，并用夹具自测保证解析正确，同时保留「缺失」与「字段不适用」的区别；接着用固定超参数的 CatBoost 直接优化 RMSE，再做去掉发动机字段的消融对照；然后在固定 20% 留出集上同时报告常数基线与模型分数，并按价格五分位分析误差和偏差；最后分块预测测试集、做往返校验并保存结果。

这个比赛最大的收获是：面对重尾的价格数据，光看一个总体 RMSE 会掩盖高价段的失败，必须把误差拆到价值区间里看，才能判断模型给出的价格是否真的可以解释。另一个收获是解析器要「敢报错」：给正则配上夹具断言，比事后发现特征全错要省事得多。对我们初学者来说，这个 notebook 提醒我们，数据清洗不是随便填填补补，而是要把字段的业务含义想清楚——电动车没有排量是正常的，不该被当成坏数据。
