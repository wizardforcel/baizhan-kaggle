# the-choice-of-an-electric-car: 大型电动车选择学习研究

## 一、比赛简介

### 1.1 比赛背景

本比赛源自 Kaggle Playground S6E9 系列挑战，由 Yao Yan、Walter Reade、Elizabeth Park 等人贡献原始竞赛文件。比赛的上游研究素材来自 Omkar Kadam 的论文《EV Adoption Behavior and Range Anxiety》，研究电动车采用行为和续航焦虑问题。

比赛提供了一个合成数据集，目的是让参赛者通过机器学习方法来预测消费者的电动车购买行为。研究背景强调了三个不同概念之间的区别：对车辆的兴趣、对购买的意愿以及实际购买行为，这三者是不同的标签。比赛要求参赛者判断数据源提供的是三者中的哪一种。

本研究由 Clare Laurent 提出新的教育性框架，并独立编写了分析笔记本，不是原始数据集的再发布，也无原贡献者的背书。原始数据集遵循 CC BY 4.0 授权。

这是一个关于消费者行为预测的经典二分类问题，聚焦于电动车接受度这一热点社会议题。数据来源于 EV 采用行为研究，包含了影响购买决策的多种特征。研究特别强调，不同的标签定义会带来完全不同的模型解释：兴趣标签反映的是用户的好奇心和信息收集行为，意向标签反映的是购买意愿的声明，而实际购买标签反映的是真实的消费行为。这三者之间存在显著的差距，同一个特征在不同标签下可能扮演完全不同的角色。

比赛框架还包含明确的合同声明，要求参赛者保持对来源文件的尊重，不对原始数据做未声明的修改，也要在报告中明确指出额外训练文件的未使用情况。这种做法确保了研究的透明性和可审计性。

### 1.2 比赛目标

比赛的核心目标是预测消费者是否会购买电动车（Will_Buy_EV）。给定训练数据，我们需要学习特征到标签的映射关系，然后对测试集预测每个样本的购买概率或类别。

根据配置文件中的设定：
- 比赛标识：s6e9
- 标题：The Choice of an Electric Car
- 目标列：Will_Buy_EV
- 问题类型：二分类
- 特性提取引出：Annual_Income_USD（年收入）
- 数据集名称：the-choice-of-an-electric-car-data

训练文件包含特征列、标识列和目标列 Will_Buy_EV。测试文件包含相同的特征列和标识列，但不包含目标列。提交文件需要以样本 ID 为索引，给出对测试集每个样本的预测结果。

比赛要求交付一个排名报告，其中包含特征削减（feature-ablation）和场景转移（shift scenario）分析。这意味着不仅仅要给出模型预测，还要分析哪些特征对模型性能影响最大，以及模型在不同场景下表现如何。

### 1.3 评价标准

本研究采用多项评价指标来全面评估模型性能，而非单一指标：

- **ROC AUC**：接收者操作特征曲线下面积，是二分类中最常用的评价指标，对类别不平衡不敏感
- **Accuracy**：准确率，即预测正确的样本比例
- **Balanced Accuracy**：平衡准确率，对类别不平衡做了校正
- **MCC（Matthews Correlation Coefficient）**：马修斯相关系数，综合考虑了真阳性、假阳性、真阴性、假阴性
- **Log Loss**：对数损失，衡量预测概率与真实标签的匹配度

其中 ROC AUC 是主要参考指标。比赛还特别强调不进行排行榜调参，所有模型设置在查看分数之前就已固定，以保证实验的公平性和可重复性。

## 二、解题过程

我们按 EDA、数据清洗、模型训练与调优、测试报告四步来走。整个流程基于固定 CatBoost 设置，强调可重复性和独立评估。

### 2.1 数据探索（EDA）

我们首先检查数据来源是否正确，并统计缺失值情况。这一步是为了确保数据完整性，以及了解样本量规模。

```python
# 设置随机种子，保证结果可复现
SEED = 20261026
np.random.seed(SEED)

# 找到数据集位置，优先检查 SHA256.json 文件
candidates = list(Path('/kaggle/input').rglob('SHA256.json'))
# 如果找不到就用 train.csv 定位

# 采用确定性 reservoir 采样，限制内存使用而非原始文件
MAX_ROWS = 120000
reservoir = None
rows_seen = 0

# 分块读取训练数据，每块 10 万行
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    # 统计每列的缺失值数量
    m = chunk.isna().sum()
    chunk['_sampling_priority'] = rng.random(len(chunk))
    # 保持 reservoir 中最小优先级的 MAX_ROWS 行
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk])
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns=['_sampling_priority']).reset_index(drop=True)
```

这段代码体现了我们对数据规模的控制策略。原始训练数据可能非常大，我们采用 reservoir 采样只保留 12 万行，这既节省内存，也能保证采样均匀。我们还统计每列的缺失值，输出 missingness.csv 文件以供参考。

接着我们确认数据的基本属性：

```python
# 确认样本 ID 唯一，避免重复样本导致信息泄漏
assert train[id_col].is_unique

# 确认目标列没有缺失值，标签必须完整
assert not train[targets].isna().any().any()

# 记录审计信息，包括来源行数、采样行数、特征列表
audit = {
    'source_rows': rows_seen,
    'sampled_rows': len(train),
    'sampling': 'seeded uniform reservoir',
    'features': feature_columns
}
```

通过审计信息，我们知道自己到底使用了多少原始数据。例如，可能原始训练集有数百万行，我们只采样 12 万行进行实验，这种说明对于结果解释很重要。

### 2.2 数据清洗

数据清洗的核心原则是无目标泄漏、状态无依赖处理。我们不使用目标统计信息，也不在全表上做插补。

```python
def features(frame, omit=()):
    # 复制特征子集，支持特征削减实验
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    
    for c in list(x.columns):
        # 如果是日期列，拆分为年、月、周
        if c in CONFIG.get('date_columns', []):
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c+'_year'] = dt.dt.year
            x[c+'_month'] = dt.dt.month
            x[c+'_weekday'] = dt.dt.dayofweek
        # 非数值列转为字符串并填补缺失
        elif not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列先转为数值格式，处理无穷大
            x[c] = pd.to_numeric(x[c], errors='coerce')
            x[c] = x[c].replace([np.inf, -np.inf], np.nan)
    return x
```

这个特征处理函数体现了谨慎的数据清洗思路。日期列会被拆分为年、月、周特征，方便模型学习时间模式。非数值列统一转为字符串，缺失值用特殊标记 __MISSING__ 填充，而不是丢弃。数值列则处理无穷大，避免后续模型报错。

对于特征削减实验，我们可以传入 omit 参数，例如移除 Annual_Income_USD 来测试收入信息的重要性：

```python
omit = [c for c in CONFIG.get('ablation', []) if c in feature_columns]
reduced = features(train, omit)  # 移除指定列后的特征
```

这种设计让我们可以公平地比较完整特征和削减特征的模型性能。

### 2.3 模型训练与超参数调优

我们采用固定 CatBoost 参数，不进行排行榜调参。模型在实验开始前就已确定设置。

```python
def model_fit(x, target):
    # 识别类别特征，让 CatBoost 自动处理
    cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # 固定参数，不根据验证集表现调整
    params = dict(
        iterations=350,          # 迭代次数
        depth=6,                 # 树深度
        learning_rate=.08,       # 学习率
        random_seed=SEED,        # 随机种子
        verbose=False,           # 不输出训练日志
        thread_count=4,          # 线程数
        allow_writing_files=False
    )
    
    # 二分类使用 Logloss 损失
    model = CatBoostClassifier(
        loss_function='Logloss',
        auto_class_weights='Balanced' if family == 'balanced' else None,
        **params
    )
    model.fit(x, target, cat_features=cat_cols)
    return model
```

选择 CatBoost 是因为它对类别特征有天然支持，无需手动编码。参数设置相对保守：350 次迭代、深度 6，在 Kaggle 比赛和工业部署中是比较常见的基础配置。learning_rate 设为 0.08 平衡了收敛速度和稳定性。

数据划分采用分层抽样，保证训练集和验证集的类别比例一致：

```python
indices = np.arange(len(train))
strata = y  # 分层依据

fit_idx, val_idx = train_test_split(
    indices, 
    test_size=.2, 
    random_state=SEED, 
    stratify=strata
)

# 确保两集合互不相交
assert set(fit_idx).isdisjoint(val_idx)
```

20% 的验证集保留用于评估，训练集用于拟合模型。这种固定划分避免了数据泄漏，也让实验结果可复现。

我们还建立了常数预测基线作为对比：

```python
def constant_prediction(target, n):
    # 基线预测始终输出训练集的平均标签比例
    const = target.mean()
    return np.tile(np.bincount(target, minlength=len(classes)) / len(target), (n, 1))
```

基线模型不使用任何特征，总是预测训练集中正类的平均比例。这帮助我们判断模型学到了真实信息还是只是拟合了标签分布。

### 2.4 模型测试与指标报告

验证阶段我们同时评估常量基线和 CatBoost 模型：

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

得分函数综合计算多个指标：

```python
def score(truth, p):
    labels = p.argmax(axis=1)
    result = {
        'accuracy': accuracy_score(truth, labels),
        'balanced_accuracy': balanced_accuracy_score(truth, labels),
        'log_loss': log_loss(truth, p, labels=np.arange(len(classes)))
    }
    # 二分类额外计算 AUC 和 MCC
    if len(classes) == 2:
        result.update(
            AUC=float(roc_auc_score(truth, p[:, 1])),
            MCC=float(matthews_corrcoef(truth, labels))
        )
    return result
```

我们还进行特征削减实验，检验 Annual_Income_USD 的重要性：

```python
omit = [c for c in CONFIG.get('ablation', []) if c in feature_columns]
if omit:
    reduced = features(train, omit)
    alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
    alt_pred = predict(alternate, reduced.iloc[val_idx])
    results['availability_ablation'] = score(y[val_idx], alt_pred)
```

该实验模拟了收入数据不可用时的情况，结果具有诊断价值，帮助我们了解哪些特征对模型至关重要。注意这只是诊断，不是部署许可。

最终我们在全量采样数据上重新拟合模型，并分块预测测试集：

```python
final = model_fit(X, y)
sample = pd.read_csv(DATA/'sample_submission.csv')
submission = sample.copy()

offset = 0
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    # 验证列名一致性
    assert chunk.columns.tolist() == [c for c in train.columns if c not in targets]
    # 验证 ID 顺序正确
    assert np.array_equal(chunk[id_col].to_numpy(), sample.iloc[offset:offset+n, 0].to_numpy())
    
    p = predict(final, features(chunk))
    # 对于二分类，取正类概率
    values = p[:, 1, None]
    
    # 写入提交文件
    submission.loc[submission.index[offset:offset+n], targets] = values[:, 0]
    offset += n
```

最终提交文件经过严格校验：行数匹配、列名正确、无缺失值、概率值在 0-1 区间。我们还生成 run_manifest.json，记录完整的实验配置、版本信息和结果，便于复现。

## 三、总结

本文中我们完成了 the-choice-of-an-electric-car 比赛的完整研究流程。我们首先通过 reservoir 采样控制数据规模，并统计缺失值情况，确保数据完整性。其次我们实施了状态无依赖的特征处理，对日期、数值、类别列分别处理，缺失值用特殊标记填充。接着我们使用固定参数的 CatBoost 分类器进行训练，采用 20% 分层验证集评估，并建立常数预测基线作为对比。我们还通过特征削减实验检验了 Annual_Income_USD 的重要性，并最终在全量采样数据上生成提交文件。

本次研究还包含三个层次的学习延伸，对于入门者、中级者和高级者分别提出不同的技能要求。入门者需要理解先验概率、赔率和基率等基础概念，并能解释当购买概率为 0.2 时赔率为 0.25，这是概率与赔率不同尺度的体现。中级者需要掌握标签漂移、事前校正和可逆性等概念，理解为什么几乎奇异的混淆矩阵会破坏黑盒迁移估计，以及为什么先验校正不能修复特征-结果关系的改变。高级者则需要考虑模型跨市场部署时的概念漂移、延迟标签问题和回滚机制，建立完善的监控契约。

本研究的局限性在于：采样限制可能无法代表全量数据集；固定参数没有进行超参数优化；验证集只是学习基线，不能保证部署性能；研究表格不能确立新补贴或定价政策下的需求。各参与者应明确各自的研究边界和独立评估要求。

同时我们也认识到模型部署需要考虑更多因素：延迟标签问题、分布漂移监测、回滚机制等。CatBoost 基线提供了可复现的起点，但真正的购买预测需要结合业务理解和持续监控。研究明确强调，表格结果不能建立新的补贴或定价政策下的需求预测，这提醒我们在解释模型结果时必须保持谦逊，承认数据和模型的边界。
