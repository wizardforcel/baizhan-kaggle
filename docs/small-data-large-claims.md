# Small Data, Large Claims（小数据，大结论）

## 一、比赛简介

### 1.1 比赛背景

这个比赛是 Kaggle Playground S5E7 的一个“独立研究”版本，题目叫 Small Data, Large Claims，直译过来就是“小数据，大结论”。它的原始数据来自 Rakesh Kapilavayi 发布的 Extrovert vs. Introvert Behavior Data，也就是“外向者与内向者行为数据”，许可协议是 CC BY-SA 4.0。数据是合成的小型表格，每一行描述一个人的日常行为习惯，目标是根据这些行为判断这个人更偏内向还是更外向。

比赛的组织者是 Walter Reade 和 Elizabeth Park。它并不是一个鼓励大家拼命刷排行榜的比赛，恰恰相反，它想提醒我们：小数据集很容易让人把一次“记得住的分数”误当成“可重复的结论”。notebook 里反复强调，这个练习不能判断一个人的性格，也不能判断一个人是否适合某个岗位、是否具有就业潜力。性格标签本身只是一个不完美的构念，也就是人为定义出来、并不能完全代表真实的人的一个标签。

所以整篇的主线是：在训练集内部划出一个固定的留出集，比较一个“只猜多数类”的常数基线和一个固定参数的 CatBoost 模型，并且把不确定性、可用性和结论边界讲清楚，而不是在排行榜上反复调参。

### 1.2 比赛目标

训练表里包含样本编号 `id`、若干行为特征，以及目标列 `Personality`。行为特征包括独处时间、参加社交活动的频率、是否害怕当众讲话（`Stage_fear`）、社交之后是否感到疲惫（`Drained_after_socializing`）等。测试表只有 `id` 和特征，要求我们给每个测试样本填上对应的 `Personality`。

提交样例 `sample_submission.csv` 只有两列：第一列是 `id`，第二列是 `Personality`。代码里用 `assert schema[1:] == targets` 检查提交格式，确保输出列没有跑偏。材料还提到，数据目录里可能附带额外的训练文件，但 baseline 只是把它们保留下来，并不使用；对训练行数设置上限也只是为了节省内存，并不代表所有可用训练行都参与了拟合。

### 1.3 评价标准

比赛按分类准确率（accuracy）排名，也就是预测正确的样本条数占全部测试样本的比例，越高越好。不过材料里的评分函数不只计算 accuracy，还会记录 balanced accuracy、AUC 和 MCC。原因是：如果类别不均衡，模型只要一直猜多数类，accuracy 也可能不低；balanced accuracy 把两个类别看得一样重，AUC 看的是概率排序，MCC 则同时考虑真正例、真负例、假正例和假负例。

我们会在同一个固定留出集上，同时报告常数基线和 CatBoost 的这几个指标，避免只用一个好看的数字下结论。

## 二、解题过程

我们首先确认提交格式和训练数据规模，其次累计缺失值并做有上限的均匀抽样，接着把特征整理成 CatBoost 能直接使用的形式，然后固定参数训练模型并与常数基线比较，最后在测试集上分块推理、检查提交文件。

### 2.1 数据探索（EDA）

第一步，我们读提交样例，确认输出列；再分块读取训练集，一边读一边统计缺失，一边做蓄水池抽样。

```python
# 先读提交样例，确认列名和输出格式；只读 3 行就够了
schema = pd.read_csv(DATA / 'sample_submission.csv', nrows=3).columns.tolist()
id_col = schema[0]              # 第一列是样本编号
assert schema[1:] == targets    # 第二列必须正好是目标列 Personality

# 分块读取训练集：即使原文件很大，也不会一次性占满内存
MAX_ROWS = 120000
reservoir = None
rows_seen = 0
missing = None
rng = np.random.default_rng(SEED)

for chunk in pd.read_csv(DATA / 'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    # 按列累计缺失值，后面单独保存成 missingness.csv
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
    # 给每一行一个随机优先级，只保留优先级最小的若干行
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
```

这里用到了“分块读取”和“蓄水池抽样”。分块读取的意思是每次只把一小段 CSV 读进内存；蓄水池抽样则是给每一行分配一个随机数，最后只保留随机数最小的那些行。这样得到的样本是均匀的，而且只要随机种子固定，结果就可以复现。我们把上限设为 120000 行，是为了控制内存，而不是宣称全部训练数据都被用上了。

接着，我们检查样本编号是否唯一、标签是否有缺失，并查看类别分布和缺失情况。

```python
# 样本编号必须唯一，否则后面的提交对齐会出问题
assert train[id_col].is_unique

# 目标列不能有缺失，否则无法判断这一行到底该学什么
assert not train[targets].isna().any().any()

# 特征就是除 id 和目标以外的所有列
feature_columns = [c for c in train if c not in targets + [id_col]]

# 看类别是否均衡：这决定了“常数基线”能拿多少准确率
class_counts = train['Personality'].value_counts()
print(class_counts)
print(class_counts / len(train))

# 把每个字段的缺失行数保存下来，方便复查
pd.DataFrame({
    'field': missing.index,
    'missing_rows': missing.values.astype(int)
}).to_csv(OUT / 'missingness.csv', index=False)
```

我们要特别关注多数类的占比。如果多数类占了绝对优势，那么一个“永远猜多数类”的模型也能得到不低的准确率。所以后面必须把常数基线放在同一张表里比较，不能只看 CatBoost 的 accuracy。

### 2.2 数据清洗

我们的清洗原则是“无状态”：不学习任何目标统计量，不用整张表的均值或中位数去填补缺失，也不让验证集标签参与预处理。所有变换只依赖当前这一列本身。

```python
def features(frame, omit=()):
    # 只保留特征列；omit 用于消融实验，临时删掉某些字段
    x = frame[[c for c in feature_columns if c not in omit]].copy()

    for c in list(x.columns):
        if not pd.api.types.is_numeric_dtype(x[c]):
            # 类别/文本列统一转成字符串，缺失值单独标成 __MISSING__
            # 这样 CatBoost 能把“缺失”当成一个类别来学习
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列强制转成数值，并把正负无穷变成缺失
            x[c] = pd.to_numeric(x[c], errors='coerce')
            x[c] = x[c].replace([np.inf, -np.inf], np.nan)

    return x

# 全特征版本，后面训练和提交都用它
X = features(train)
```

这里没有做 target encoding，也就是没有用目标值去构造特征；也没有做整表填补。这样做是为了避免“信息泄漏”：如果验证集的信息以任何形式进入预处理，验证分数就会虚高。CatBoost 本身可以处理缺失值和类别特征，所以我们只做最基础的格式统一。

然后，我们把标签编码成 0 和 1，并做固定比例的、按类别分层的留出划分。

```python
# 排序后建立稳定的类别到整数映射，避免每次运行编码顺序变化
classes = sorted(train['Personality'].unique().tolist())
mapping = {v: i for i, v in enumerate(classes)}
y = train['Personality'].map(mapping).to_numpy()
assert len(classes) == 2       # 这是一个二分类任务

indices = np.arange(len(train))
fit_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,             # 固定留出 20% 作为验证集
    random_state=SEED,
    stratify=y,                # 分层抽样，保持两边的类别比例接近
)

# 训练索引和验证索引不能有重叠
assert set(fit_idx).isdisjoint(val_idx)
```

这一步只做一次，之后不再反复换切分。材料里说得很清楚：验证标签不用于预处理、不用于早停、也不用于参数选择；固定设置是在看分数之前就定好的。

### 2.3 模型训练与超参数调优

这一节的关键不是“调出最高分”，而是“预先固定实验”。材料明确说明：不在排行榜上调参；固定设置先定义好实验，然后再看分数。我们使用的模型是 CatBoostClassifier，参数为 350 轮迭代、树深 6、学习率 0.08，随机种子固定，线程数为 4。

```python
def model_fit(x, target):
    # CatBoost 可以直接处理字符串列，先自动找出类别特征
    cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()

    # 这些参数是材料里预先固定的，没有在验证集上做网格搜索
    params = dict(
        iterations=350,
        depth=6,
        learning_rate=0.08,
        random_seed=SEED,
        verbose=False,
        thread_count=4,
        allow_writing_files=False,
    )

    # 二分类使用 Logloss，训练时把类别特征的位置告诉模型
    model = CatBoostClassifier(loss_function='Logloss', **params)
    model.fit(x, target, cat_features=cat_cols)
    return model
```

我们刻意没有做超参数搜索。原因很简单：如果我们在同一个验证集上反复试参数，验证集就会慢慢变成训练集的一部分，最后得到的分数会偏乐观。小数据上这种风险尤其大，因为样本少，一次切分带来的波动可能比参数带来的提升还大。

为了知道模型到底有没有学到东西，我们先做一个常数基线。

```python
def constant_prediction(target, n):
    # 找出训练集里出现次数最多的类别
    values, counts = np.unique(target, return_counts=True)
    majority = values[np.argmax(counts)]

    # 验证集的每一行都预测成这个多数类
    p = np.zeros((n, len(values)))
    p[:, np.argmax(counts)] = 1.0
    return p

# 只用训练部分的标签来构造基线，验证标签不参与
baseline = constant_prediction(y[fit_idx], len(val_idx))
```

常数基线回答了一个很基本的问题：如果完全不看特征，只根据训练集的类别比例来猜，准确率能有多少？CatBoost 必须明显超过这个参照，才能说明特征真的提供了信息；如果只高一点点，就要结合样本量和波动来谨慎解释。

我们用的评分函数同时记录多个指标。

```python
def predict(model, x):
    # 返回每个类别的概率，行是样本，列是类别
    return model.predict_proba(x)

def score(truth, p):
    labels = p.argmax(axis=1)
    result = {
        'accuracy': accuracy_score(truth, labels),
        'balanced_accuracy': balanced_accuracy_score(truth, labels),
    }
    # 二分类时再补充 AUC 和 MCC
    if len(classes) == 2:
        result['AUC'] = roc_auc_score(truth, p[:, 1])
        result['MCC'] = matthews_corrcoef(truth, labels)
    return result
```

accuracy 是主指标；balanced accuracy 让两个类别权重相同；AUC 看的是正类概率的排序质量；MCC 则综合了混淆矩阵的四个格子。把这几个指标放在一起看，比只盯着一个 accuracy 更不容易被类别不均衡误导。

接着，我们做第二个实验：在完全相同的划分上，删掉 `Stage_fear` 和 `Drained_after_socializing` 这两个字段，看看模型有多依赖它们。

```python
# 用 omit 参数临时去掉两个自报字段
X_reduced = features(train, omit=('Stage_fear', 'Drained_after_socializing'))

# 仍然使用同一组训练索引，保证和全特征模型可比
alt = model_fit(X_reduced.iloc[fit_idx], y[fit_idx])
alt_pred = predict(alt, X_reduced.iloc[val_idx])

# 这只是一种诊断，不是自动调参，也不代表可以部署不可用字段
ablation_score = score(y[val_idx], alt_pred)
print(ablation_score)
```

材料特别提醒：可用性消融只是诊断，不是部署许可；提交模型仍然是预先指定的全特征 baseline。换句话说，即使删掉某些字段后分数变化不大，也不能因此就认为在真实场景中可以不用这些字段，因为真实场景还要考虑这些字段是否可得、是否合规。

### 2.4 模型测试与指标报告

在固定留出集上，我们把常数基线和 CatBoost 放在一起评分，并保存混淆矩阵。

```python
# 用训练部分拟合全特征模型
model = model_fit(X.iloc[fit_idx], y[fit_idx])
pred = predict(model, X.iloc[val_idx])

# 同一张表里放常数基线和 CatBoost，方便直接比较
results = {
    'constant': score(y[val_idx], baseline),
    'boosting': score(y[val_idx], pred),
}
print(pd.DataFrame(results).T)

# 混淆矩阵能看出模型主要把哪一类认成了哪一类
cm = confusion_matrix(y[val_idx], pred.argmax(axis=1))
pd.DataFrame(cm, index=classes, columns=classes).to_csv(
    OUT / 'confusion_matrix.csv'
)
pd.DataFrame(results).T.to_csv(OUT / 'validation_scores.csv')
```

我们没有在这里写出具体的分数，因为分数只代表这一次固定留出集上的结果。材料的态度很明确：一个留出集只是学习用的基线，不是部署保证；真要用于实际决策，还需要说明分组、时间、不确定性和独立评估。对于小数据，换一个随机种子或者换一次切分，结果都可能波动，所以更重要的是把实验流程和边界讲清楚。

最后，我们在全部抽样训练集上重新拟合固定配方，对测试集分块预测，并逐行检查提交文件。

```python
# 提交模型仍然是全特征、固定参数的 CatBoost
final = model_fit(X, y)

sample = pd.read_csv(DATA / 'sample_submission.csv')
submission = sample.copy()
offset = 0

for chunk in pd.read_csv(DATA / 'test.csv', chunksize=100000):
    # 测试集的特征列顺序必须和训练时一致
    assert chunk.columns.tolist() == [c for c in train.columns if c not in targets]

    n = len(chunk)
    # 测试集 id 的顺序必须和提交样例完全一致
    assert np.array_equal(
        chunk[id_col].to_numpy(),
        sample.iloc[offset:offset + n, 0].to_numpy()
    )

    # 分块预测，再按类别概率取最大的那个类
    p = predict(final, features(chunk))
    values = np.asarray(classes)[p.argmax(axis=1)][:, None]
    submission.loc[submission.index[offset:offset + n], targets] = values
    offset += n

# 最后检查行数、id 唯一性和目标列完整性
assert offset == len(sample)
assert submission[id_col].is_unique
assert not submission[targets].isna().any().any()
submission.to_csv(OUT / 'submission.csv', index=False)
```

这段代码有几个“保险”：测试特征列顺序必须和训练一致；测试 id 顺序必须和提交样例一致；分块写入可以避免一次性读入全部测试集；最后再检查行数、编号唯一性和目标列是否有缺失。只有这些检查都通过，提交文件才是可复现的。

## 三、总结

本文中我们围绕 Small Data, Large Claims 这个基于 Kaggle Playground S5E7 的独立研究，完整走了一遍小型表格分类的流程。我们首先读取提交样例确认输出格式，并用分块读取加固定种子蓄水池抽样的方式，把训练数据限制在最多 120000 行；其次统计每个字段的缺失情况，检查样本编号唯一性和标签完整性；接着用无状态的方式整理特征，把类别列的缺失标成 `__MISSING__`，把数值列的无穷值转成缺失，并做固定 20% 的分层留出；然后使用预先固定的 CatBoost 参数训练模型，同时构造只猜多数类的常数基线，在同一留出集上比较 accuracy、balanced accuracy、AUC 和 MCC；最后在全量抽样数据上重新拟合，对测试集分块预测，并逐项检查提交文件。

我们得到的最大体会是：小数据最危险的地方，不是模型不够复杂，而是人很容易被一个记得住的分数说服。所以这篇 writeup 里，我们刻意没有在排行榜或验证集上反复调参，也没有把一次留出集的结果当成部署保证。消融实验删掉 `Stage_fear` 和 `Drained_after_socializing`，只是为了观察模型对字段的依赖，不能据此认为真实业务中可以不用这些字段。性格标签本身也只是一个人为定义的构念，这个练习不能用来判断一个人的性格、岗位适配度或就业潜力。把基线、指标、切分方式和结论边界都写清楚，比单纯追求一个更高的数字更重要。
