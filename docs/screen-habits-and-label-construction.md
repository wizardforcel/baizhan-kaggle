# Screen Habits and Label Construction

## 一、比赛简介

### 1.1 比赛背景
Screen Habits and Label Construction 是 Kaggle Playground S6E8 的一个独立研究型题目，源头数据来自 algozee 的 smartphone-addiction-prediction-data。比赛的出发点不是预测真正的成瘾诊断，而是一次目标构造审计。原始合成标签 addicted_label 是从问卷字段里按规则生成的，如果我们直接用同样的问卷字段去预测标签，模型很可能只是在复现构造规则，而不是发现新的现象。组织方希望参赛者写一份目标构造审计报告，并做一次迁移测试——换掉特征集合，看看性能会不会掉。合成标签不是临床诊断，不能当成成瘾的医学判断。

我们拿到的数据来自 Kaggle 比赛 screen-habits-and-label-construction，数据文件是 screen-habits-and-label-construction-data。比赛明确说标签是合成的，且提供了一个固定的基线实验规范，禁止在排行榜上调参。

### 1.2 比赛目标
训练集里每行是一个用户的问卷记录，含有若干屏幕使用习惯字段和一个二分类标签 addicted_label。测试集只有特征列，没有标签。我们需要输出每个测试样本的 addicted_label 概率或类别。题目规定不能使用额外的训练文件，采样要做但保留审计信息。我们要做的不仅是提交预测，还要解释标签是否被特征直接定义。

### 1.3 评价标准
比赛家族是 binary。基线规范里使用 AUC、balanced_accuracy、MCC、log_loss 等指标来评估。主要关注 AUC，即 ROC 曲线下面积，越高越好。规范中还要求做常数基线对比，以及移除 stress_level 和 academic_work_impact 后的可用性消融。提交文件必须和 sample_submission 结构一致，概率值在 0 到 1 之间，且不能有缺失。

## 二、解题过程

整场比赛的思路可以概括成一句话：按规范做一个固定设置的 CatBoost 基线，记录采样、划分、消融和验证分数，用来审计标签是否被特征定义。我们按 EDA、数据清洗、模型训练与超参数调优、模型测试与指标报告四步来走。

### 2.1 数据探索（EDA）
我们首先读取样本提交文件确定列名和 ID 列，然后用分块阅读和种子化的水箱采样限制内存，只保留最多 12 万行均匀采样。采样过程中同时统计缺失行数，以便后续审计。

```python
# 读取样本提交，确定 ID 列和目标列
schema = pd.read_csv(DATA/'sample_submission.csv', nrows=3).columns.tolist()
id_col = schema[0]
assert schema[1:] == targets

# 水箱采样，限制内存同时保持均匀性
MAX_ROWS = 120000
reservoir = None; rows_seen = 0
rng = np.random.default_rng(SEED)
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk])
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')
train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
```

采样结束后我们检查 ID 唯一性和标签无缺失，并把缺失统计导出为 missingness.csv。feature_columns 是除 ID 和目标外的所有列。我们发现源文件行数很大，采样后约 12 万行用于训练，这是规范规定的内存限制，不是数据量不足。

接着我们统计标签分布并做 20% 的分层划分，保证验证集的正负比例与整体一致。

```python
# 确定类别并编码
classes = sorted(train[targets[0]].unique().tolist())
mapping = {v:i for i,v in enumerate(classes)}
y = train[targets[0]].map(mapping).to_numpy()

# 分层划分，保证类别比例稳定
indices = np.arange(len(train))
fit_idx, val_idx = train_test_split(indices, test_size=.2,
                                    random_state=SEED, stratify=y)
```

我们还做了缺失统计表格并打印审计信息，包括来源行数、采样行数、特征列表和额外文件清单。这些信息为后续审计提供了可追溯性。

### 2.2 数据清洗
题目要求的清洗是无状态的，不能用目标统计做填充。我们对数值列做强制转换，对非数值列统一填补 __MISSING__ 标记，并把日期列拆成年、月、周几。整个过程不学习整体统计，避免泄漏。

```python
# 无状态特征转换，不使用目标统计
def features(frame, omit=()):
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c+'_year'] = dt.dt.year
            x[c+'_month'] = dt.dt.month
            x[c+'_weekday'] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf,-np.inf], np.nan)
    return x
```

我们不对标签做任何插补，训练集中标签必须完整。特征工程保持极简，目的就是让模型性能变化可以归因于特征可用性而不是清洗技巧。对于类别型特征，CatBoost 能够自动处理，我们只是统一标记缺失。

清洗完成后我们得到 X = features(train)，作为后续建模的输入。整个清洗过程可复现，因为种子固定。

### 2.3 模型训练与超参数调优
规范要求模型参数固定，不在排行榜上调参。我们采用 CatBoostClassifier，iterations=350，depth=6，learning_rate=0.08，random_seed 固定，线程数 4。没有做早停，参数在实验开始前就确定。

```python
# 固定参数的模型拟合函数
def model_fit(x, target):
    cat_cols = x.select_dtypes(include=['object','string']).columns.tolist()
    params = dict(iterations=350, depth=6, learning_rate=.08,
                  random_seed=SEED, verbose=False, thread_count=4,
                  allow_writing_files=False)
    model = CatBoostClassifier(loss_function='Logloss', **params)
    model.fit(x, target, cat_features=cat_cols)
    return model
```

我们还准备了预测和评分函数，评分包含 accuracy、balanced_accuracy、log_loss 和 AUC、MCC。常数基线用于提供学习基线，方便判断模型是否超过平凡预测。

```python
# 预测函数，统一接口
def predict(model, x):
    return model.predict_proba(x)

# 评分函数，返回关键指标字典
def score(truth, p):
    labels = p.argmax(axis=1)
    result = {'accuracy': float(accuracy_score(truth, labels)),
              'balanced_accuracy': float(balanced_accuracy_score(truth, labels)),
              'log_loss': float(log_loss(truth, p, labels=np.arange(len(classes))))}
    result.update(AUC=float(roc_auc_score(truth, p[:,1])),
                  MCC=float(matthews_corrcoef(truth, labels)))
    return result
```

训练时我们先用常数基线得到基准分数，然后用全特征训练模型。我们还按规范做了可用性消融，移除 stress_level 和 academic_work_impact 两个字段，重新训练并比较分数。

```python
# 常数基线用于对比
baseline = constant_prediction(y[fit_idx], len(val_idx))
model = model_fit(X.iloc[fit_idx], y[fit_idx])
pred = predict(model, X.iloc[val_idx])
results = {'constant': score(y[val_idx], baseline),
           'boosting': score(y[val_idx], pred)}

# 预先指定的消融实验
omit = [c for c in CONFIG.get('ablation',[]) if c in feature_columns]
if omit:
    reduced = features(train, omit)
    alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
    alt_pred = predict(alternate, reduced.iloc[val_idx])
    results['availability_ablation'] = score(y[val_idx], alt_pred)
```

消融实验的结果被完整记录，即使性能更好我们也不自动部署，因为可用性是独立的决策。整个流程强调先声明实验，再看分数，避免数据窥探。

### 2.4 模型测试与指标报告
验证分数写入 validation_scores.csv，混淆矩阵和验证预测也导出。接下来我们在采样后的全量数据上重新拟合固定模型，用分块推理预测测试集，保证 ID 顺序和提交契约一致。

```python
# 全特征基线再拟合，用于最终提交
final = model_fit(X, y)
sample = pd.read_csv(DATA/'sample_submission.csv')
submission = sample.copy()
offset = 0
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    p = predict(final, features(chunk))
    values = p[:,1,None]   # 二分类取正类概率
    submission.loc[submission.index[offset:offset+n], targets[0]] = values[:,0]
    offset += n
submission.to_csv(OUT/'submission.csv', index=False)
```

提交前我们检查列名、行数、ID 唯一性、值非空且概率在 0 到 1 之间。运行日志还会打印耗时、版本和审计信息。我们在验证集上看到提升明显优于常数基线，消融后分数下降，说明 stress_level 和 academic_work_impact 提供了部分预测力，但正如规范所说，这只是诊断，不是部署许可。

## 三、总结
本文中我们按 Kaggle Playground S6E8 的规范复现了一个固定参数的 CatBoost 基线，使用种子化的水箱采样限制在 12 万行，分层 20% 验证，并做了预先声明的特征消融。我们通过无状态特征转换保证了可复现性，用常数基线和消融分数审计了标签构造的影响。我们最终生成了符合提交契约的 submission.csv，并在验证集上记录了 accuracy、balanced_accuracy、AUC、MCC 和 log_loss。整个流程强调学习基线、独立评估和可用性审计，而不是在排行榜上调优。
