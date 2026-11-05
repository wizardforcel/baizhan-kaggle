# The Cost of a Packed Bag

## 一、比赛简介

### 1.1 比赛背景

这个比赛源自 Kaggle Playground S5E2，官方名称是 The Cost of a Packed Bag。原始数据集来自 Souradip Pal 的 Student Bag Price Prediction Dataset，由 Kaggle 贡献者 Walter Reade 和 Elizabeth Park 整合成竞赛文件，采用 MIT 许可。材料里强调 More data help only when their provenance and validation role are understood。也就是说，多加数据只有在搞清楚数据来源和它在验证里的角色时才有帮助。

作者 Clare Laurent 写了独立研究笔记，用来演示一个可复现的基线，而不是追排行榜。研究里把额外的训练文件保留在数据契约里但不使用，重点是先在固定验证集上比较常数基线和 CatBoost 基线，再做一个可证伪的第二次实验。材料反复提醒，预测的包价是描述性的，并不代表干预后客户愿意付的价格。

材料还把额外数据的来源问题讲得很具体：在合并额外的背包训练文件之前，必须记录它的来源、许可证、字段结构、标签定义、行数，以及它与主训练集和评估集的关系。校验和只能说明用了哪一份字节，不能单独证明数据质量或来源合法。合并两张表之后还应保留一个来源指示列，方便做分布审计和误差审计；至于要不要把这个来源列放进预测模型，则是另一个需要单独测试的选择。

竞赛文件保持未修改，训练集很大，基线笔记里为了省内存只用均匀抽样限制在最多 12 万行，整个流程用固定随机种子 20261026，验证集固定为 20%，不做任何排行榜上的调参。

### 1.2 比赛目标

任务是回归预测。配置文件里写明了：

```python
CONFIG = {
    'episode': 's5e2',
    'title': 'The Cost of a Packed Bag',
    'competition_slug': 'the-cost-of-a-packed-bag',
    'targets': ['Price'],
    'family': 'regression'
}
```

输入是一张表，包含一个 ID 列和若干包的属性特征列，输出是 Price，单位是价格。训练集只有一部分样本有标签 Price，测试集没有标签，我们需要给出每行测试样本的价格预测。

材料里提到会保留额外的训练文件，但基线实验里不使用它们。特征列是训练表里除 id 列和目标列外的所有列。笔记里特别点名 Weight Capacity (kg) 这一列，后面会用来做可用性消融。

sample_submission.csv 的第一列是 ID，后面的列应当与 targets 完全一致，代码里用 assert schema[1:] == targets 来确认这一点。否则后续写提交文件时列名容易对不上，也容易在比赛中因格式错误丢分。

### 1.3 评价标准

比赛族是回归。笔记里的 score 函数对回归返回 RMSE，即均方根误差，值越小越好。

```python
def score(truth, p):
    if family == 'regression':
        return {'RMSE': float(np.sqrt(mean_squared_error(truth, p)))}
```

除了模型预测，笔记还计算常数预测基线，用训练集目标的均值作为整体预测。验证集的分数会保存到 validation_scores.csv，同时保存验证预测和误差带 error_bands.csv。整个评估在固定的 20% 保留集上完成，不使用测试集标签。

RMSE 可以用大白话理解为“预测值和真实值平均差多远”。计算时先把每条误差平方，避免正负误差互相抵消，再求平均并开平方，所以结果仍然和价格使用同一单位。平方会放大较大的错误，因此特别大的偏差会明显拉低模型表现。

## 二、解题过程

我们按 EDA、数据清洗、模型训练与超参数调优、模型测试与指标报告四步来走。叙述用第一人称复数，代码来自材料里的基线笔记，注释用中文解释为什么这么写。

### 2.1 数据探索（EDA）

我们首先要把数据读进来，同时记录缺失情况。为了控制内存，笔记采用确定性的水库抽样，最多保留 12 万行。

```python
MAX_ROWS = 120000
reservoir = None
rng = np.random.default_rng(SEED)
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    # 记录每个列的缺失行数
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
    # 每个块生成采样优先级
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk])
    # 只保留优先级最小的 MAX_ROWS 行
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')
train = reservoir.drop(columns='_sampling_priority')
```

这样既能遍历全量数据，又保证抽样是均匀且可复现的。抽样结束后我们会输出一份审计信息，包含源文件总行数、抽样后行数、特征列列表、目标列、是否使用额外文件等。

```python
audit = {
    'source_rows': rows_seen,
    'sampled_rows': len(train),
    'sampling': 'seeded uniform reservoir',
    'features': feature_columns,
    'targets': targets,
    'source_files_unchanged': True
}
```

我们把缺失情况单独保存到 missingness.csv，字段是 field 和 missing_rows。EDA 的目的不是看分布直方图，而是确认数据契约：ID 唯一、目标无缺失、额外文件被列出但未使用。

### 2.2 数据清洗

数据本身没有缺失目标值，ID 唯一。清洗的核心是把特征统一成模型可用的类型，同时不使用目标统计量做预处理，保持无泄漏。

```python
def features(frame, omit=()):
    # 只保留特征列，可以传入 omit 去掉某些列做消融
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c+'_year'] = dt.dt.year
            x[c+'_month'] = dt.dt.month
            x[c+'_weekday'] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 非数值统一转字符串，缺失用 __MISSING__ 占位
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列强制转数值，非法值变 NaN
            x[c] = pd.to_numeric(x[c], errors='coerce')
            x[c] = x[c].replace([np.inf, -np.inf], np.nan)
    return x
```

我们不做全局均值填补，不做基于目标的编码，所有转换都是无状态的。非数值列保留原始文本，CatBoost 会自动处理类别特征。这样清洗服务于模型，而不是盲目套用常规的文本清洗流程。

### 2.3 模型训练与超参数调优

训练采用固定的 CatBoost 回归基线，不在排行榜上调参。先做分层划分，回归任务不分层，只是固定随机种子保证可复现。

```python
indices = np.arange(len(train))
fit_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=SEED, stratify=None
)
```

模型拟合函数里明确写死超参数，方便他人复现。

```python
def model_fit(x, target):
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
    model = CatBoostRegressor(loss_function='RMSE', **params)
    model.fit(x, target, cat_features=cat_cols)
    return model
```

我们先用常数基线作为对照，常数就是训练集目标的均值。

```python
X = features(train)
baseline = constant_prediction(y[fit_idx], len(val_idx))
model = model_fit(X.iloc[fit_idx], y[fit_idx])
pred = predict(model, X.iloc[val_idx])
results = {
    'constant': score(y[val_idx], baseline),
    'boosting': score(y[val_idx], pred)
}
```

因为特征维度高且 CatBoost 对类别特征友好，我们不做网格搜索。材料里的做法是把精力放在实验设计上，而不是死磕超参数。

第二次实验是可证伪的消融。我们按配置文件指定的列去掉 Weight Capacity (kg)，在完全相同的验证集上重新训练。

```python
omit = [c for c in CONFIG.get('ablation', []) if c in feature_columns]
if omit:
    reduced = features(train, omit)
    alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
    alt_pred = predict(alternate, reduced.iloc[val_idx])
    results['availability_ablation'] = score(y[val_idx], alt_pred)
```

消融的目的是诊断这项特征的可获得性影响，而不是为了改进排行榜分数。所有实验都记录在 validation_scores.csv 里。

### 2.4 模型测试与指标报告

验证结束后我们会输出详细的诊断文件。对回归任务，我们保存每条验证样本的 id、实际值、预测值、误差，并按实际值的五分位分组计算 MAE 和偏差。

```python
diagnostics = pd.DataFrame({
    'id': train.iloc[val_idx][id_col].to_numpy(),
    'actual': y[val_idx],
    'prediction': pred
})
diagnostics['error'] = diagnostics.prediction - diagnostics.actual
diagnostics['target_band'] = pd.qcut(diagnostics.actual, 5, duplicates='drop')
diagnostics.groupby('target_band').agg(
    rows=('error','size'),
    MAE=('error', lambda s: s.abs().mean()),
    bias=('error','mean')
).to_csv(OUT/'error_bands.csv')
```

最后我们用全量抽样数据重新拟合一次固定的全特征基线，再分块预测测试集，保证提交格式与 sample_submission.csv 完全一致。

```python
final = model_fit(X, y)
sample = pd.read_csv(DATA/'sample_submission.csv')
submission = sample.copy()
offset = 0
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    # 特征转换和预测
    p = predict(final, features(chunk))
    values = p[:, None]
    # 写入对应位置，避免索引错位
    submission.loc[submission.index[offset:offset+len(chunk)], targets] = values
    offset += len(chunk)
submission.to_csv(OUT/'submission.csv', index=False)
```

提交前我们会检查列名、行数、ID 唯一性、目标无 NaN、无无穷大，并校验证书哈希，最后写出 run_manifest.json 记录种子、版本、审计信息和耗时。整个流程不使用验证标签做预处理或早停，完全符合事先规定的实验协议。

## 三、总结

本文中我们完成了 The Cost of a Packed Bag 的基线复现。我们首先用水库抽样读入训练数据并记录缺失和审计信息，其次用无状态的特征转换把类别和数值列统一处理，接着用固定超参数的 CatBoost 回归在 80/20 划分上训练，并用常数均值作为对照，最后在完全相同的验证集上运行 Weight Capacity (kg) 的可用性消融，并在全量抽样数据上重拟合后分块生成提交文件。

这个比赛给我们的最大启发是数据契约和实验设计的重要性。同样是回归，模型分数只有在固定验证集、固定采样、固定超参数的前提下才可比较。特征的可获得性消融提醒我们，实际部署时某些列可能不可用，模型必须在这种情况下依然可评估。预测的价格只是描述性的，不能直接等同于客户在干预下的支付意愿，这一点在报告里必须保持可见。
