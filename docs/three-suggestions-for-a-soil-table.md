# Three Suggestions for a Soil Table

## 一、比赛简介

### 1.1 比赛背景

Three Suggestions for a Soil Table 是一次建立在 Kaggle Playground S5E6 之上的独立研究。原始的合成比赛任务是 Fertilizer prediction，也就是「肥料推荐预测」，由 Walter Reade 和 Elizabeth Park 组织，数据以 CC0 公共领域协议发布，上游链接指向 Irakoze Ntawigenga Kelly 的同名数据集。Playground 系列的特点是数据由程序合成、规模适中、允许反复练习，排行榜成绩不是唯一目的。

这次研究的作者 Clare Laurent 没有去追求排行榜名次，而是把重点放在「排名指标到底怎么算」这件事上。很多入门者做分类时只盯着准确率，也就是「第一名猜对了没有」。但在这个比赛里，提交的不是一个答案，而是三个候选答案，评分要看你把正确答案排在第几位。材料开篇就说得很直白：单一的最佳标签和一份有用的候选名单，回答的是两个不同的问题，我们要学会评估正确答案在候选列表中的位置。

正因为数据是合成的，材料也提醒我们：一个合成的肥料标签，不等于农业上的处方，更不等于做过对照试验。模型学到的是数据生成程序里的统计关联，不能直接拿去指导施肥。这个限定看起来扫兴，但它划清了「预测」和「决策」的边界。

### 1.2 比赛目标

输入是一条训练样本，包含土壤、作物、气候等特征列；目标列是 `Fertilizer Name`，也就是该样本对应的肥料名称。类别数量在材料中通过编码映射打印出来，结合材料里提到的「七个类别指示变量」，可以理解为这是一个七分类问题。

测试阶段的输出形式和普通分类不同：每条测试样本要给出三个不同的肥料名称，用空格分隔。也就是说，模型不能只输出一个 argmax，而要输出一个按概率排序的前三名单。这个名单必须满足两个条件：第一，三个名字互不重复，否则等于浪费位置；第二，顺序有意义，排第一和排第三的得分不一样。

训练仍然只使用官方 train.csv，测试使用 test.csv。材料中的配置写着 `targets: ['Fertilizer Name']`、`family: 'ranking'`，预设的消融字段是 `Soil Type` 和 `Crop Type`。样本同样设定 12 万行上限，用固定种子的蓄水池采样控制内存。

### 1.3 评价标准

比赛使用 MAP@3，全称是 Mean Average Precision at 3，中文可以叫「前三名平均精度」。它的计算规则是：对每一条样本，看正确标签出现在预测列表的第几位；如果排第一，这条样本得 1 分；排第二，得 1/2 分；排第三，得 1/3 分；没进前三，得 0 分。最后把所有样本的得分求平均。

```python
top = np.argsort(-p, axis=1, kind='stable')[:, :3]   # 每行取概率最高的三个类别
# 正确类别在第 k 位就得 1/k 分，k 从 1 开始
row_scores = ((top == truth[:, None]) / np.arange(1, 4)).sum(axis=1)
MAP3 = row_scores.mean()
```

这个指标有两个容易踩的坑。第一，它奖励「把正确答案往前放」，而不只是「放进前三」。把正确答案从第三位挪到第一位，分数会从 0.33 涨到 1.0，所以模型输出的概率排序很关键。第二，如果提交里出现重复标签，重复的位置等于作废，严重时还会违反提交格式。材料专门要求手工实现 MAP@3，并用「命中、未命中、重复、换序」几组小例子来验证自己的实现没有写错。

材料还给出了一个参照：排行榜第一名 Chris Deotte 报告的私有 MAP@3 是 0.38652。他把八个输入当作类别特征，对最多四个输入的组合做编码，生成七个类别指示变量，再加上原始数据聚合、堆叠 OOF 预测、重复种子和爬山权重。我们的笔记本是独立基线，不声称复现这个分数。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先做的还是安全读入。和上篇一样，用固定种子的分块蓄水池采样，把训练数据限制在 12 万行。这样做的目的不是「数据太多用不完」，而是让内存占用可预测，同时保证每次运行抽到的行完全一致。

```python
MAX_ROWS = 120000
SEED = 20261026
rng = np.random.default_rng(SEED)
reservoir = None

for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    # 给每行一个固定种子生成的随机优先级
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk])
    # 只留下优先级最小的 12 万行
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
```

接着我们读取 `sample_submission.csv` 的前三行，确认提交文件的列结构。第一列是 id，后面是目标列。代码用断言保证 schema 和目标配置一致，避免「训练时叫一个名字、提交时叫另一个名字」的低级错误。

```python
schema = pd.read_csv(DATA/'sample_submission.csv', nrows=3).columns.tolist()
id_col = schema[0]
assert schema[1:] == targets          # 提交列必须和目标列一致
assert train[id_col].is_unique        # 训练 id 唯一
assert not train[targets].isna().any().any()   # 目标不能缺失
```

然后我们看类别分布。做法是把目标列映射成从 0 开始的整数，打印映射表，并统计每个类别出现的次数。映射表很重要，因为后面预测出来的是概率向量，要按索引还原成肥料名称；如果映射顺序和提交时的类别顺序不一致，分数会莫名其妙地低。

```python
classes = sorted(train[targets[0]].unique().tolist())
mapping = {v: i for i, v in enumerate(classes)}
y = train[targets[0]].map(mapping).to_numpy()

print('Class encoding:', mapping)
print(pd.Series(y).value_counts().sort_index())
```

从材料给出的类别编码逻辑可以确认，这是一个单标签多分类问题，而不是多标签。每条样本只有一个正确的肥料名称，因此 MAP@3 里「相关标签」只有一个，第 k 位的贡献就是 1/k，不需要像多标签检索那样累加多个命中。

### 2.2 数据清洗

清洗仍然遵循「无状态、不看标签」的原则。数值列统一转成数值类型，无法解析的变成 NaN，正负无穷也变成 NaN。非数值列转成字符串，缺失填成 `'__MISSING__'`。这样做的理由是：缺失本身可能携带信息，比如某个土壤字段没填，可能对应某类记录方式；把它显式标成一个类别，树模型可以在分裂时把「缺失」单独分到一边。

```python
def features(frame, omit=()):
    # omit 用于消融实验：临时去掉 Soil Type、Crop Type 等字段
    x = frame[[c for c in feature_columns if c not in omit]].copy()

    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):
            # 日期列拆成三个数值特征，树模型才能直接使用
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c+'_year'] = dt.dt.year
            x[c+'_month'] = dt.dt.month
            x[c+'_weekday'] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 类别列：缺失单独作为一个类别保留下来
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列：坏值、无穷值统一变成 NaN，交给 CatBoost 处理
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return x
```

这里没有做独热编码，也没有做标准化。原因是我们选的模型是 CatBoost，它可以直接处理类别特征，内部会做一种叫「有序目标编码」的处理；对树模型来说，特征的绝对尺度也不重要，所以标准化不是必须的。

标签同样不允许填补。代码里用断言守住这条线，因为一条没有正确肥料名称的样本，无法计算 MAP@3，也无法提供有效梯度。

```python
assert not train[targets].isna().any().any(), 'Do not silently impute a label.'
```

### 2.3 模型训练与超参数调优

我们使用 CatBoost 多分类器。CatBoost 属于梯度提升树家族：它训练很多棵决策树，每棵新树都去拟合前面模型的残差，最后把结果加起来。对表格数据来说，这种模型通常不需要太多特征工程就能拿到不错的基线。

参数是预先固定的：350 轮迭代、深度 6、学习率 0.08、固定随机种子、4 个线程。材料明确说「不在排行榜上调参」，也就是说这些参数在看验证分数之前就定好了，后面的分数是对这套配置的评价，而不是搜索出来的最优值。

```python
from catboost import CatBoostClassifier

params = dict(
    iterations=350,        # 建 350 棵树
    depth=6,               # 每棵树最深 6 层
    learning_rate=0.08,    # 学习率
    random_seed=SEED,
    verbose=False,
    thread_count=4,
)

cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()
model = CatBoostClassifier(loss_function='MultiClass', **params)
model.fit(x, target, cat_features=cat_cols)
```

模型输出的不是单个类别，而是每个类别的概率。`predict_proba` 返回一个形状为「样本数 × 类别数」的矩阵，每一行加起来等于 1。我们把这个矩阵按行降序排序，取前三个索引，就得到候选名单。

```python
proba = model.predict_proba(x_valid)
top3 = np.argsort(-proba, axis=1, kind='stable')[:, :3]
```

这里的 `kind='stable'` 是稳定排序，意思是概率完全相同时，保持原来的类别顺序。为什么要在意这个细节？因为如果每次运行的 tie-break 规则不一样，同样的输入可能给出不同的名单，复现性就被破坏了。材料在讨论部分专门问了「如何把类别概率变成可复现的候选名单」，答案就是：降序排序、保留类别索引映射、对完全相同的概率用明确的确定性规则。

除了 boosting 模型，我们还构造了常数基线。分类任务的常数基线不是「猜一个固定类别」，而是「输出训练集的类别频率」。对每一条验证样本都给出同样的概率分布，然后取前三。这个基线可以告诉我们：如果完全不看特征，MAP@3 能拿多少。

```python
# 训练集里每个类别出现的比例
freq = np.bincount(y[fit_idx], minlength=len(classes)) / len(fit_idx)
# 所有验证样本都重复这个分布
baseline = np.tile(freq, (len(val_idx), 1))
results = {'constant': score(y[val_idx], baseline)}
```

第二个实验是预先声明的消融：去掉 `Soil Type` 和 `Crop Type` 两列，在相同的训练行和验证行上重新训练。这个实验回答的问题是「这两列到底贡献了多少」。如果去掉之后 MAP@3 掉得厉害，说明土壤和作物类型是强特征；如果几乎不变，说明模型主要靠别的字段。材料强调，这种消融是诊断，不是「上线时就可以不要这些字段」的许可。

```python
omit = [c for c in CONFIG.get('ablation', []) if c in feature_columns]
reduced = features(train, omit)
alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
alt_pred = predict(alternate, reduced.iloc[val_idx])
results['availability_ablation'] = score(y[val_idx], alt_pred)
```

### 2.4 模型测试与指标报告

验证阶段，我们同时报告 MAP@3、准确率和 log loss。准确率只看 argmax 对不对，MAP@3 看前三名的排序质量，log loss 看概率校准得好不好。三个指标一起看，才能判断模型是真的变好了，还是只是「第一名碰巧对了」。

```python
labels = proba.argmax(axis=1)
result = {
    'accuracy': accuracy_score(y[val_idx], labels),
    'log_loss': log_loss(y[val_idx], proba, labels=np.arange(len(classes))),
}
top = np.argsort(-proba, axis=1, kind='stable')[:, :3]
result['MAP@3'] = np.mean(
    ((top == y[val_idx][:, None]) / np.arange(1, 4)).sum(axis=1))
```

材料里还有一个很重要的检查：手工实现 MAP@3，并用小例子验证。我们构造了「命中第一位」「命中第二位」「命中第三位」「完全未命中」「重复标签」「换序」几组固定输入，手算期望分数，再和函数输出对比。只有手算和代码一致，才敢相信排行榜上的分数。

确认流程之后，我们在 12 万行采样数据上重拟合最终模型，分块读取测试集进行推理。每一块都要检查列顺序和提交样例一致，预测后把概率转成前三名，再拼成空格分隔的字符串。

```python
final = model_fit(X, y)
submission = pd.read_csv(DATA/'sample_submission.csv')
offset = 0

for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    p = predict(final, features(chunk))
    top = np.argsort(-p, axis=1, kind='stable')[:, :3]
    # 把类别索引还原成肥料名称，用空格连接
    values = np.array([' '.join(str(classes[j]) for j in row) for row in top])
    # 每个名单必须恰好三个互不相同的名字
    assert all(len(set(row.split())) == 3 for row in values)
    submission.loc[offset:offset+len(chunk), targets[0]] = values
    offset += len(chunk)

assert offset == len(submission)
submission.to_csv(OUT/'submission.csv', index=False)
```

最后我们回读提交文件，确认列名、行数、id 顺序都正确。材料没有给出这个基线具体的 MAP@3 数值，所以我们只报告流程和对照关系，不编造分数。

## 三、总结

本文中我们完成了 Three Suggestions for a Soil Table 的独立基线研究。我们首先用固定种子的分块蓄水池采样把数据限制在 12 万行，并检查提交 schema、id 唯一性和标签缺失；其次做无状态清洗，把类别缺失显式标成单独类别，数值坏值和无穷值统一转成 NaN；接着用固定参数的 CatBoost 多分类器输出类别概率，在相同 holdout 上与类别频率常数基线比较，并预先声明了移除 `Soil Type`、`Crop Type` 的消融；然后手工实现并验证 MAP@3，用命中、未命中、重复和换序的小例子检查公式；最后在全量采样数据上重拟合，分块预测，保证每条提交恰好包含三个互不相同的肥料名称。

这个练习让我们真正理解了一件事：排名指标不是「准确率的变体」，它对位置有明确的、非线性的奖励。把正确答案从第三位提到第一位，价值是第三位的三倍。也正因为如此，模型输出的概率排序、类别索引映射、tie-break 规则都必须小心处理。材料最后还提醒，合成肥料标签不是农学处方；预测「记录过的选择」和判断「哪种肥料真的更好」，是两个完全不同的问题。
