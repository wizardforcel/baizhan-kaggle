# Response at Scale

## 一、比赛简介

### 1.1 比赛背景
这是基于 Kaggle Playground 赛季第 4 季第 7 轮的独立研究版本，原题为“保险交叉销售（二元分类）”。原始数据来源于 Annant Kumar Singh 的 Health Insurance Cross Sell Prediction 数据集，由 Kaggle Playground 的 Walter Reade 与 Ashley Chow 发布，使用 CC BY 4.0 许可证。

大白话说，这个比赛的背景是一个保险营销表：公司已经有健康保险客户，现在想推销车辆保险给他们。表格足够大，数据搬运和实验纪律本身就成了建模任务的一部分。比赛的核心问题是：客户是否会响应（Response=1）车辆保险的推销，而不是是否应该联系。

研究报告里特别强调，响应倾向并不能证明联系客户带来的增量收益。研究的目的不是为了追排行榜，而是比较不同训练样本规模下的分数-运行时间权衡，保持实验可复现。

### 1.2 比赛目标
给定客户的人口统计、车辆和保单信息，预测目标列 Response 是否为 1。训练表包含标识列 id 和目标列 Response，其余为特征列。测试集只有 id 和特征列，需要输出每个客户被响应概率的 0~1 值。

从代码里的 CONFIG 可以看到：
```python
CONFIG = {'episode': 's4e7', 'title': 'Response at Scale', 'competition_slug': 'response-at-scale',
          'targets': ['Response'], 'family': 'binary', 'ablation': [], 'date_columns': []}
```
说明这是一个单标签二分类任务，目标只有一个列 Response。研究采用统一随机预留采样，上限 120000 行训练，固定 20% 保留集用于验证，不在排行榜上调参。

### 1.3 评价标准
在独立研究里，我们同时报告训练常数基准和固定 CatBoost 基线的多项指标。代码里的 score 函数对二分类返回：准确率 accuracy、平衡准确率 balanced_accuracy、对数损失 log_loss、AUC 以及 Matthews 相关系数 MCC。AUC 越高越好，log_loss 越低越好，MCC 与 AUC 类似越高越好。提交文件要求输出 0~1 之间的概率，体现了基于概率的评价思路。

报告中强调，这里用一个固定的公开保留集做学习基准，不代表部署时的保证。模型是预先设定的全特征基线，ablation 只是诊断性的。

## 二、解题过程

整套思路是：先用有界均匀采样把大表放进内存，再用无状态特征处理保持实验可复现，最后用固定超参数的 CatBoost 做二分类，并在相同验证集上比较常数基准、全样本基线、小样本基线三者的分数。

### 2.1 数据探索（EDA）
我们首先读取原始训练表，统计总体行数、缺失行数，并用确定性的水库采样限制内存。采样不是取前 N 行，而是给每行一个随机优先级，保证在流式读取中每个训练行被均匀抽中。

```python
# 读取分块并做确定性水库采样，上限 120000 行
MAX_ROWS = 120000
reservoir = None
rng = np.random.default_rng(SEED)
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    chunk['_sampling_priority'] = rng.random(len(chunk))  # 给每行随机优先级
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk])
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')  # 保留优先级最小的 MAX_ROWS 行
train = reservoir.drop(columns='_sampling_priority')
```

我们同时累计每列的缺失行数并导出 missingness.csv，方便后续判断哪个字段缺失最多。代码里断言 id 列唯一且目标列没有缺失，保证标签是干净的。

随后我们打印一份审计信息，包括来源总行数、采样后行数、采样方式、特征列表和额外文件清单。这一版的研究使用单一公开保留集、分层抽样保持类别比例。

### 2.2 数据清洗
我们不做基于目标的统计填补，保持无状态变换。第二步是定义 features 函数，对非数值列统一转为字符串并用 '__MISSING__' 填充缺失，对数值列做强制转换并把无穷值转成 NaN。

```python
def features(frame, omit=()):
    # 只保留特征列，省略列可以做 ablation 研究
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        # 非数值转为字符串，保持缺失标记
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors='coerce')  # 强制为数值，不合法的转为 NaN
    return x
```

我们发现数据本身比较干净，主要工作是把分类特征让 CatBoost 能直接识别，数值特征统一类型。清洗是为了保持实验可复现，而不是做泄露性的目标统计拟合。

### 2.3 模型训练与超参数调优
我们使用固定超参数的 CatBoost 分类器，不做网格搜索。超参数在实验开始前就确定：iterations=350，depth=6，learning_rate=0.08，random_seed 固定，线程数为 4。分类特征由 select_dtypes 自动识别。

```python
def model_fit(x, target):
    cat_cols = x.select_dtypes(include=['object','string']).columns.tolist()  # 自动找分类列
    params = dict(iterations=350, depth=6, learning_rate=.08, random_seed=SEED,
                  verbose=False, thread_count=4, allow_writing_files=False)
    model = CatBoostClassifier(loss_function='Logloss', **params)
    model.fit(x, target, cat_features=cat_cols)  # 直接拟合，不看验证集
    return model
```

训练集与验证集的划分采用一次性的 train_test_split，比例 80/20，随机种子固定，并对单标签分类做分层抽样。

```python
indices = np.arange(len(train))
strata = y if not regression else None
fit_idx, val_idx = train_test_split(indices, test_size=.2, random_state=SEED, stratify=strata)
```

为了建立基准，我们先做常数预测：用训练集里正类的频率作为全部验证样本的预测值。

```python
def constant_prediction(target, n):
    # 常数基准，等于训练集的类别频率
    return np.tile(np.bincount(target, minlength=len(classes))/len(target), (n,1))
baseline = constant_prediction(y[fit_idx], len(val_idx))
```

随后用全特征基线模型拟合并预测验证集，得到 boosting 结果。这一步不调参，只是让模型和基准在同一保留集上比较。

训练后我们进行第二项预先指定的实验：用更小的训练预算在同一保留集上评估。代码取前面 25000 个训练索引的模型。

```python
small_idx = fit_idx[:min(25000, len(fit_idx)//2)]
alternate = model_fit(X.iloc[small_idx], y[small_idx])
results['smaller_training_budget'] = score(y[val_idx], predict(alternate, X.iloc[val_idx]))
```

这项实验帮助我们看到样本规模与分数、运行时间的权衡。

### 2.4 模型测试与指标报告
在验证阶段，我们把模型在验证集上的预测保存为 validation_predictions.csv，并输出混淆矩阵。指标同时包含准确率、平衡准确率、log_loss、AUC、MCC，便于从多个角度理解二分类表现。

最终提交时，我们在全部有界采样数据上重新拟合一次固定模型，然后分块读取测试集做预测，保证内存占用可控并与 sample_submission 的 id 顺序严格对应。

```python
final = model_fit(X, y)  # 在全部采样数据上再拟合一次
submission = sample.copy()
offset = 0
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    n = len(chunk)
    p = predict(final, features(chunk))  # 得到概率
    values = p[:,1,None]  # 二分类取正类概率
    submission.loc[submission.index[offset:offset+n], targets] = values
    offset += n
```

提交前我们校验列名、行数、id 唯一性、没有缺失值以及概率在 0~1 之间。所有结果、审计信息、模型配置和运行时间都写入 run_manifest.json，确保实验可追溯。

## 三、总结
本文中我们基于 Kaggle Playground S4E7 的保险交叉销售数据集，按照“Response at Scale”独立研究的设计，完成了有界均匀采样、EDA、数据清洗、固定超参数 CatBoost 训练与验证以及小样本对比的完整流程。我们首先用确定性水库采样把大表限制在 120000 行内并统计缺失，其次用无状态特征处理统一分类与数值类型，接着用固定 350 次迭代、深度 6 的 CatBoost 在分层 20% 保留集上比较常数基准与全特征基线，并在同一保留集上用 25000 行的更小训练预算做对照实验，最后用分块推理生成符合格式的提交文件并保存验证报告与清单。整个研究强调实验纪律、样本规模与运行时间的权衡，并明确响应倾向不能等同于增量收益这一局限。
