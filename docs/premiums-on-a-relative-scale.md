# 相对尺度上的保费预测

## 一、比赛简介

### 1.1 比赛背景

这个比赛来自 Kaggle Playground 系列的第 4 季第 12 集（S4E12），题目是预测保险保费的金额（Premium Amount）。Playground 系列的特点是数据由合成生成器产生，规模大、字段多、噪声也不小，适合练习完整的建模流程。保险定价的核心难点在于：同一个绝对误差，对于便宜的保单和昂贵的保单意义完全不同，因此评价指标通常采用对数尺度，让小金额和大金额的误差被公平地看待。

本笔记本的作者特别强调，这是一项独立的教育性研究：它使用固定的、事先指定的模型设置，不根据排行榜分数调参，目的是展示一个诚实、可复现的基线，而不是追求名次。作者还明确提醒，预测出来的保费不能作为真实保险报价或理赔成本的依据。

### 1.2 比赛目标

输入是一张包含投保人信息、保单信息等字段的表格，输出是每份保单的保费金额。提交文件的第一列是 ID，第二列是 `Premium Amount`。我们从 `sample_submission.csv` 读出列名来确认目标列：

```python
# 从提交样例中读出 ID 列名和目标列名，防止写死列名
schema = pd.read_csv(DATA/'sample_submission.csv', nrows=3).columns.tolist()
id_col = schema[0]
assert schema[1:] == targets   # targets = ['Premium Amount']
```

### 1.3 评价标准

比赛使用均方根对数误差 RMSLE（Root Mean Squared Logarithmic Error）。它的计算方式是对真实值和预测值都先取 `log1p`，算差值平方、求平均、再开方。由于取了对数，预测偏高和偏低在指标上更对称，而且小金额上的相对误差会被放大。因此我们的主线模型直接在 `log1p` 空间里做回归，这样训练目标和评价指标是一致的。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先以分块的方式读取训练集，一边读一边统计缺失情况，并用固定随机种子的蓄水池抽样把数据量限制在 12 万行以内：

```python
MAX_ROWS = 120000
reservoir = None; rows_seen = 0; missing = None
rng = np.random.default_rng(SEED)
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    rows_seen += len(chunk)                       # 记录原始总行数
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
    # 给每行一个随机优先级，最后保留优先级最小的 12 万行
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')
train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
```

这样做的原因很实际：完整训练集很大，普通笔记本内存吃不消；而均匀抽样不会系统性地偏向某类样本。我们同时断言 ID 唯一、标签没有缺失，绝不悄悄填补标签。

我们把缺失统计单独存成 `missingness.csv`，并用描述性统计查看目标 `Premium Amount` 的分布和零值比例。结果显示目标右偏明显，且存在零保费，这正是必须用 `log1p` 而不是 `log` 的原因——`log(0)` 是负无穷，而 `log1p(0)=0` 有定义。

### 2.2 数据清洗

清洗采用「无状态变换」：只看当前行，不跨行统计目标信息，避免验证集信息泄漏。日期列拆成年、月、星期几；非数值列转成字符串并把缺失填成一个特殊标记 `__MISSING__`；数值列统一转成数字，并把正负无穷替换成缺失：

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
            # 缺失本身可能携带信息，所以用专门的字符串标记而不是直接丢
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf,-np.inf], np.nan)
    return x
```

我们没有做全表填补，因为树模型（CatBoost）可以直接处理缺失值，把缺失交给模型去学更安全。

### 2.3 模型训练与超参数调优

我们把数据固定切成 80% 训练、20% 验证，并且只把训练部分的标签用于拟合、早停和参数选择。模型使用 CatBoost 回归器，参数是事先固定的，不在验证集上反复挑：

```python
def model_fit(x, target):
    # 把字符串列当作类别特征交给 CatBoost
    cat_cols = x.select_dtypes(include=['object','string']).columns.tolist()
    params = dict(iterations=350, depth=6, learning_rate=.08,
                  random_seed=SEED, verbose=False, thread_count=4)
    model = CatBoostRegressor(loss_function='RMSE', **params)
    # 关键：在 log1p 空间训练，与 RMSLE 指标对齐
    model.fit(x, np.log1p(target), cat_features=cat_cols)
    return model

def predict(model, x):
    # 预测出来的是 log1p 空间的值，用 expm1 还原，并保证非负
    p = model.predict(x)
    return np.maximum(np.expm1(p), 0)
```

为什么用 `expm1` 而不是 `exp`？因为训练时用的是 `log1p(y)`，它的反函数是 `exp(z)-1`，而 `expm1(z)` 正好等于 `exp(z)-1`，在 z 接近 0 时数值更稳定；如果直接用 `exp(z)`，会凭空多出 1 的偏移。

除了主模型，我们还做了一个可证伪的对照实验：在完全相同的验证行上，移除 `Policy Start Date` 这一列再训练一次，看看去掉这个可能在实际业务中拿不到的字段后结果如何变化。作者强调，这种「可得性消融」只是诊断，不等于可以放心地把不可用字段部署上线。

### 2.4 模型测试与指标报告

我们先在验证集上计算两组分数：一组是「只用训练集均值/对数均值」的常数基线，一组是 CatBoost：

```python
def constant_prediction(target, n):
    # logreg 情形下的常数基线：先取 log1p 均值，再反变换回原尺度
    const = np.expm1(np.log1p(target).mean())
    return np.full(n, const)

baseline = constant_prediction(y[fit_idx], len(val_idx))
model = model_fit(X.iloc[fit_idx], y[fit_idx])
pred = predict(model, X.iloc[val_idx])
results = {'constant': score(y[val_idx], baseline),
           'boosting': score(y[val_idx], pred)}
```

其中 `score` 对 logreg 家族返回的就是 RMSLE。除了总分，我们还按真实保费的五分位区间切分，统计每个区间的平均绝对误差和偏差：

```python
diagnostics = pd.DataFrame({'id': train.iloc[val_idx][id_col].to_numpy(),
                            'actual': y[val_idx], 'prediction': pred})
diagnostics['error'] = diagnostics.prediction - diagnostics.actual
diagnostics['target_band'] = pd.qcut(diagnostics.actual, 5, duplicates='drop')
diagnostics.groupby('target_band', observed=True).agg(
    rows=('error','size'), MAE=('error', lambda s: s.abs().mean()), bias=('error','mean')
).to_csv(OUT/'error_bands.csv')
```

分区间看误差很重要：总 RMSLE 好看，不代表在便宜保单上就没有系统性高估或低估。

最后我们在 12 万行抽样上重新拟合全特征模型，分块读测试集、逐块断言 ID 顺序与样例一致，再写出提交文件：

```python
final = model_fit(X, y)
submission = pd.read_csv(DATA/'sample_submission.csv')
offset = 0
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    # 断言测试块列和训练特征完全对应，防止列错位
    assert chunk.columns.tolist() == [c for c in train.columns if c not in targets]
    n = len(chunk)
    p = predict(final, features(chunk))
    submission.loc[submission.index[offset:offset+n], 'Premium Amount'] = p
    offset += n
submission.to_csv(OUT/'submission.csv', index=False)
```

作者在文末提到，排行榜第一名 Chris Deotte 的报告里用了单个 XGBoost 加 611 个特征、类别交互、嵌套折的目标编码与计数编码、保单日期分解和 GPU 加速特征搜索。这些是原作者的经验，不代表我们可以复现，我们的目标是把一个诚实基线讲清楚。

## 三、总结

本文中我们完成了相对尺度保费预测的完整研究。我们首先用分块读取加固定种子蓄水池抽样，把数据控制在 12 万行并把缺失统计落盘；其次采用只依赖当前行的无状态特征变换，避免验证信息泄漏；接着用固定超参数的 CatBoost 在 `log1p` 空间回归，并用 `expm1` 加非负裁剪还原预测；然后在固定 20% 留出集上与常数基线对比，并按保费五分位报告误差与偏差；最后分块预测测试集、校验提交格式并保存。

这个比赛让我们理解了两件事：一是评价指标决定目标变换，RMSLE 就应该在 `log1p` 空间训练；二是诚实的研究要把缺失机制、对照实验和分区间误差都摆出来，而不是只报一个好看的分数。
