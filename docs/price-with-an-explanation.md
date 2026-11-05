# 带解释的价格预测

## 一、比赛简介

### 1.1 比赛背景

这个比赛是 Kaggle Playground 系列第 4 季第 9 集（S4E9），任务是根据二手车的登记信息预测它的价格。笔记本作者把它写成一个「价格审计」练习：一位二手车分析师不仅要给出一辆车看起来合理的估价，还要能说明这个估价为什么可信、在哪些价位上不可信。真实世界的二手车数据里，豪车数量少但价格极高，会主导平方误差；描述字段（比如发动机）又是非结构化的字符串，必须小心解析。因此这个比赛真正考验的是数据清洗和误差分析，而不是把模型调得多复杂。

作者反复声明，这是独立的教育性研究，模型设置事先固定，不根据排行榜调参；预测出来的交易价格既不是认证估价，也不能解释为某个车辆特征的因果效应。

### 1.2 比赛目标

输入是一张二手车信息表，输出是每辆车的成交价格 `price`。提交文件的第一列是 ID，第二列是 `price`：

```python
# 从提交样例读取列名，确认 ID 列与目标列
schema = pd.read_csv(DATA/'sample_submission.csv', nrows=3).columns.tolist()
id_col = schema[0]
assert schema[1:] == targets    # targets = ['price']
```

### 1.3 评价标准

比赛用均方根误差 RMSE 评价，数值越低越好。由于价格是重尾分布，少数豪车会产生很大的平方误差，所以作者除了报告整体 RMSE，还按真实价格的价值区间切分，分别报告平均绝对误差和偏差，用来区分「平均误差变大」和「相对误差变差」这两件不同的事。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先分块读入训练集，统计每一列的缺失数量，并把缺失表保存下来。读取时使用固定种子的蓄水池抽样，把最多 12 万行留在内存里：

```python
MAX_ROWS = 120000
reservoir = None; rows_seen = 0; missing = None
rng = np.random.default_rng(SEED)
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')
train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
```

我们断言每辆车的 ID 唯一，并且 `price` 没有缺失——标签缺失不能靠插值糊弄过去。

接着查看价格分布。它明显右偏：绝大多数车集中在较低价位，少数豪华车把右尾拉得很长。这提醒我们，后面按价格分箱看误差是必要的。我们还查看了 `engine` 列的原始写法，发现它把马力、排量和气缸布局混在一个字符串里，例如 `300.0HP 3.0L V6`，也有 `unknown` 这样的取值，所以必须写解析器而不是整体当类别。

### 2.2 数据清洗

清洗的核心是「无状态变换」：每一行只看自己，不跨行学习统计量，防止验证集信息泄漏。日期列拆成年、月、星期；文本列缺失填特殊标记；数值列统一类型并处理无穷：

```python
def features(frame, omit=()):
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c+'_year'] = dt.dt.year
            x[c+'_month'] = dt.dt.month
            x[c+'_weekday'] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 用显式标记保留「缺失」这一事实
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf,-np.inf], np.nan)
    return x
```

针对 `engine` 列，我们用带单位约束的正则分别抽出 horsepower 和排量，并写了一条自测断言，确保 `300.0HP 3.0L V6` 能正确解析出 `300.0`：

```python
if CONFIG['episode'] == 's4e9' and 'engine' in x:
    # 只抓 "数字+HP" 和 "数字+L"，避免把其他数字误当马力或排量
    x['engine_hp'] = x['engine'].str.extract(r'([0-9.]+)HP', expand=False).astype(float)
    x['engine_litres'] = x['engine'].str.extract(r'([0-9.]+)L', expand=False).astype(float)
    # 用一个已知样例做夹具，解析错了立刻报错
    assert pd.Series(['300.0HP 3.0L V6', 'unknown']).str.extract(
        r'([0-9.]+)HP', expand=False).iloc[0] == '300.0'
```

作者特别提醒，电动车没有排量是「字段不适用」，不是「格式错误」，所以解析失败时保留缺失并把原文留着，而不是硬塞一个 0。

### 2.3 模型训练与超参数调优

我们固定划分 20% 作为留出集，训练集占 80%，且验证标签不参与预处理、早停或参数选择。模型用 CatBoost 回归，参数事先写死：

```python
def model_fit(x, target):
    cat_cols = x.select_dtypes(include=['object','string']).columns.tolist()
    params = dict(iterations=350, depth=6, learning_rate=.08,
                  random_seed=SEED, verbose=False, thread_count=4)
    model = CatBoostRegressor(loss_function='RMSE', **params)
    model.fit(x, target, cat_features=cat_cols)   # 直接优化 RMSE
    return model
```

我们没有在排行榜上反复试探，而是把「全特征 + 固定参数」作为提交模型。第二个实验是事先规定的消融：在同样的训练行和验证行上，去掉 `engine` 列重新训练，观察解析出来的发动机信息到底有没有帮助。作者强调，这种对照只是诊断，不是「字段不可得也能部署」的许可证。

### 2.4 模型测试与指标报告

验证阶段我们同时计算常数基线（训练集价格均值）和 CatBoost 的 RMSE：

```python
X = features(train)
baseline = np.full(len(val_idx), y[fit_idx].mean())   # 常数基线
model = model_fit(X.iloc[fit_idx], y[fit_idx])
pred = predict(model, X.iloc[val_idx])
results = {'constant': {'RMSE': float(np.sqrt(mean_squared_error(y[val_idx], baseline)))},
           'boosting': {'RMSE': float(np.sqrt(mean_squared_error(y[val_idx], pred)))}}
```

仅看一个 RMSE 不够，我们按真实价格做五分位分箱，统计每档的样本数、平均绝对误差和平均偏差：

```python
diagnostics = pd.DataFrame({'id': train.iloc[val_idx][id_col].to_numpy(),
                            'actual': y[val_idx], 'prediction': pred})
diagnostics['error'] = diagnostics.prediction - diagnostics.actual
diagnostics['target_band'] = pd.qcut(diagnostics.actual, 5, duplicates='drop')
diagnostics.groupby('target_band', observed=True).agg(
    rows=('error', 'size'),
    MAE=('error', lambda s: s.abs().mean()),
    bias=('error', 'mean')
).to_csv(OUT/'error_bands.csv')
```

这个表能回答「是不是少数豪车把误差撑大了」：如果最高价位档的 MAE 远高于其他档，就说明模型在高价段不可靠；如果偏差符号一致，就说明存在系统性高估或低估。

最后用全特征模型在抽样数据上重训，分块预测测试集，并做提交文件的往返校验：

```python
final = model_fit(X, y)
sample = pd.read_csv(DATA/'sample_submission.csv')
submission = sample.copy(); offset = 0
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    assert chunk.columns.tolist() == [c for c in train.columns if c not in targets]
    n = len(chunk)
    p = predict(final, features(chunk))
    submission.loc[submission.index[offset:offset+n], 'price'] = p
    offset += n
assert offset == len(sample) and submission[id_col].is_unique
assert np.isfinite(submission['price'].to_numpy(dtype=float)).all()
submission.to_csv(OUT/'submission.csv', index=False)
```

作者在文末提到，排行榜第一名的报告用了「异常价格分类器作为额外特征、折内中位数目标编码、SVR/LightGBM/分类器/XGBoost 的神经堆叠」等做法，但那是原作者的经验，不是我们必须复现的配方。

## 三、总结

本文中我们完成了二手车价格预测的价格审计式研究。我们首先用分块读取加蓄水池抽样控制内存，并统计缺失；其次采用无状态特征变换，专门为 `engine` 写了带单位约束的正则解析器并用夹具自测；接着用固定超参数的 CatBoost 直接优化 RMSE，再做去掉发动机字段的消融对照；然后在固定 20% 留出集上同时报告常数基线与模型分数，并按价格五分位分析误差和偏差；最后分块预测并校验提交格式。

这个比赛最大的收获是：面对重尾的价格数据，光看一个总体 RMSE 会掩盖高价段的失败，必须把误差拆到价值区间里看，才能判断模型给出的价格是否真的可以解释。
