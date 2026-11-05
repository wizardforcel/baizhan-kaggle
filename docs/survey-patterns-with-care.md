# Survey Patterns With Care

## 一、比赛简介

### 1.1 比赛背景

这个比赛要处理的是心理健康问卷调查数据。本次研究来自一份名为 "Survey Patterns With Care"（谨慎对待问卷模式）的独立 notebook，它强调一个核心观点：**问卷的回答是"在特定措辞和特定抽样方式下得到的测量结果"**，当我们拿它去预测心理健康标签（比如是否抑郁）时，必须保留这种区分，不能想当然地把问卷答案当成客观事实。

比赛使用的数据是合成（synthetic）的，来自 Kaggle Playground 系列赛 S4E11（配置里写的就是 `episode: 's4e11'`），比赛贡献者是 Walter Reade 和 Ashley Chow，许可证为 CC0（公共领域），比赛文件保持原样未被改动。它背后的原始来源是 Suman Sharma 的 《Depression Survey/Dataset for Analysis》（抑郁症问卷数据集），问卷的原始作者保留数据的署名权。材料反复强调：**这是一个合成问卷练习，不能用来评估任何真实个体的心理健康状况**。

难点在于：问卷数据是典型的表格数据，但充满文本类别的问卷选项、缺失值和类别不均衡，而且"抑郁标签"本身是由回答构造出来的，一不小心就会把相关性当成因果。材料还提出了两条纪律：保存一份"清洗日志"（cleaning log），并且证明每一次修复对"留出集准确率"（held-out accuracy）的影响。

### 1.2 比赛目标

输入是一张问卷回答表。训练集包含一个 id 列、若干特征列（feature_columns，即除目标列之外的所有问卷问题列），以及一个目标列 `Depression`（是否抑郁）。测试集没有标签，我们要给每条测试样本输出一个 `Depression` 的预测。提交文件 `sample_submission.csv` 的第一列是 id，后面的列就是目标列。

数据是逐块（chunk）读取的，训练行数上限是 12 万行（`MAX_ROWS = 120000`）。如果不做上限，把整个训练文件一次性读进内存很可能爆内存，所以材料采用"种子化的均匀水库抽样（seeded uniform reservoir）"来抽取一个有代表性的子集。目标列不允许有缺失，材料里还显式断言"不要悄悄填补标签"（Do not silently impute a label）。

### 1.3 评价标准

比赛按"留出集准确率（held-out accuracy）"来评价，也就是预测正确的样本数占样本总数的比例，越高越好。notebook 里的 `score()` 函数除了算准确率，还会额外报告 balanced accuracy（平衡准确率）、log loss（对数损失）、AUC 和 MCC，作为更完整的诊断指标。因为类别可能不均衡，只看准确率会被多数类带偏，所以这些辅助指标用来观察模型是否真的学到了有用的东西。

材料强调了两条原则：一是验证只用本地"固定的 20% 留出集"，模型从头到尾没有看过排行榜（leaderboard）标签；二是"先在检查分数之前把实验设定写死"，避免在排行榜或验证集上反复试错造成信息泄漏。

---

## 二、解题过程

整条主线的思路是：**先抽样摸清数据，再统一清洗特征，接着用一个固定参数的 CatBoost 分类器与"常数基线"对比，然后做去除敏感特征的可证伪实验，最后分块预测测试集并核对提交契约**。我们按四步来走。

### 2.1 数据探索（EDA）

我们首先做数据探索（EDA）。由于训练文件很大，我们不是一次性读入，而是按每块 10 万行逐块读取，并用种子化的水库抽样把行数压到最多 12 万行；同时逐块累计每一列的缺失行数，得到全表的缺失统计。

```python
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

MAX_ROWS = 120000        # 样本上限：防止把整个大文件塞进内存
reservoir = None         # 水库：最终保留下来的训练子集
rows_seen = 0            # 已读过的总行数，用于审计
missing = None           # 各列缺失行数的累计
SEED = 20261026
rng = np.random.default_rng(SEED)   # 用固定随机种子，保证抽样可复现

for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    # 累计每一列的缺失数：第一次出现时直接取，之后按列对齐相加
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
    # 给每一行打一个随机优先级，最后只保留优先级最小的 12 万行
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
# 审计信息：原始多少行、抽了多少行、用了什么抽样方法
audit = {'source_rows': rows_seen, 'sampled_rows': len(train),
         'sampling': 'seeded uniform reservoir', 'targets': ['Depression']}
print(json.dumps(audit, indent=2))
```

这里关键点有两个：一是**抽样是可复现的**（每次都带 `SEED`），二是**抽样比例不受数据块的顺序影响**——`nsmallest` 按随机优先级挑全局最小的 12 万行，等效于从全部数据里做均匀随机抽样，而不是简单取前 12 万行，避免了文件顺序带来的偏差。

我们把缺失统计写进 `missingness.csv`，再记录标签的类别编码，并断言目标列没有缺失：

```python
# 把每一列的缺失行数存成"缺失报告"，之后清洗时对照使用
pd.DataFrame({'field': missing.index,
              'missing_rows': missing.values.astype(int)}).to_csv(OUT/'missingness.csv', index=False)

# 目标列必须完整：允许 0/1 二分类或多分类，但绝不能有 NaN
assert not train[['Depression']].isna().any().any(), '不要悄悄填补标签'

# 特征列 = 除 id 和目标列之外的所有列；这类问卷特征既有数值也有文本选项
feature_columns = [c for c in train if c not in ['Depression'] + [Schema.id]]
```

因为我们不去碰排行榜，唯一的验证手段就是后面切出的留出集，所以**缺失报告和标签断言**是 EDA 阶段最重要的两道防线，也是"清洗日志"的起点。

### 2.2 数据清洗

其次，我们做数据清洗。清洗策略全部收在 `features()` 函数里，并且刻意设计成"无状态变换"——不在全表上学任何统计量（不做目标编码、不做全表均值填充），避免验证集的信息渗进训练过程。处理规则分三类：日期列拆成年/月/星期（本比赛 `date_columns` 为空，不会触发）；文本列统一转成字符串并用 `'__MISSING__'` 填充缺失；数值列转成 float 并把正负无穷转成缺失。

```python
def features(frame, omit=()):
    # 只保留特征列；omit 传进来时用于第二个实验里去掉某些敏感字段
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):          # 日期列：拆出年、月、星期特征
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c + '_year'] = dt.dt.year
            x[c + '_month'] = dt.dt.month
            x[c + '_weekday'] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):    # 文本列：缺失填成显式哨兵
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:                                            # 数值列：把无穷值转成缺失
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return x
```

设计决策说明：文本列的缺失我们既不删样本也不用均值填充，而是填成字符串 `'__MISSING__'`，让 CatBoost 把它当作一个"缺失"类别去学——在问卷里"没有回答这一问题"本身可能就是一种信号；数值列把无穷值转成 NaN 后交给模型自行处理。整张表不学习任何统计量，保证清洗阶段不产生信息泄漏。

关于缺失，材料里专门的 "Middle：问卷选择与缺失作答" 一节还提到了三个概念：response-bias（作答偏差）、complete-case（完整样本分析）、MNAR（非随机缺失）。如果只保留回答完整的样本（complete-case），而"回答得完不完整"本身和人的特征或结果相关，那么被分析的群体就变了，模型会引入偏差；同时，光靠观测数据无法证明缺失是随机的，所以更稳妥的做法是设定几种合理的缺失场景分别跑一遍，看结论是否稳健。我们采用的是后一种思路：不删不填，把缺失本身暴露给模型。

### 2.3 模型训练与超参数调优

接着，我们切出验证集并训练模型。材料的主线是"先把常数基线做出来，再跑固定参数的 CatBoost"，两份成绩放在一起对比，才能回答"模型到底有没有学到东西"。

```python
# 单标签分类：对类别做编号，并断言至少有两个类别
classes = sorted(train['Depression'].unique().tolist())
mapping = {v: i for i, v in enumerate(classes)}
y = train['Depression'].map(mapping).to_numpy()

# 切 80% 训练 / 20% 留出；对单标签分类用分层抽样，让两边类别比例一致
indices = np.arange(len(train))
fit_idx, val_idx = train_test_split(indices, test_size=.20,
                                    random_state=SEED, stratify=y)
```

然后是 `model_fit`：它收集仍为字符串的列作为类别特征（`cat_features`），用固定参数训练 CatBoost。这里**没有网格搜索**，参数是开工前就定好的，目的就是先看固定设定下的基线水平：

```python
def model_fit(x, target):
    # 类别特征 = 尚为字符串（文本选项）的列
    cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()
    # 固定超参数：350 轮、深度 6、学习率 0.08；验证分数不参与选参
    params = dict(iterations=350, depth=6, learning_rate=.08,
                  random_seed=SEED, verbose=False, thread_count=4,
                  allow_writing_files=False)
    # 二分类用 Logloss，多分类用 MultiClass
    loss = 'Logloss' if len(classes) == 2 else 'MultiClass'
    model = CatBoostClassifier(loss_function=loss, **params)
    model.fit(x, target, cat_features=cat_cols)
    return model
```

**关于超参数**：材料特意强调"在检查分数之前先规定好实验设定"——验证标签不参与预处理、不参与早停、不参与参数选择。这并不是偷懒，而是为了保证"这个留出分数确实就是这套固定设定产生的"；如果先看了分数再回头调参数，等于在验证集上反复试用，分数会虚高，对测试集就失去参考价值。

随后我们做了**第二个可证伪实验**：把问卷里一个极其敏感的问题 "Have you ever had suicidal thoughts ?"（你是否曾有过自杀念头）从特征里拿掉，在完全相同的留出行上重新训练一次，叫作 availability ablation（可用性消融），目的是看清：如果现实中不能使用这个字段（比如出于隐私考虑该字段不可得），模型会损失多少。材料说得非常克制：**消融只是诊断工具，不是"部署时就可以放心用这些字段"的许可**。

```python
# 第二个实验在查看分数之前就规定好：去掉敏感字段，重训一次
omit = [c for c in CONFIG.get('ablation', []) if c in feature_columns]
if omit:
    reduced = features(train, omit)                        # 清洗时去掉该列
    alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
    alt_pred = predict(alternate, reduced.iloc[val_idx])
    results['availability_ablation'] = score(y[val_idx], alt_pred)  # 同一批留出行
```

### 2.4 模型测试与指标报告

然后，我们在留出集上测试并生成指标报告。评分函数覆盖分类常用的指标：准确率、平衡准确率、对数损失，二分类再补 AUC 和 MCC。对照对象是"常数基线"——对分类问题，它等于"每类都按训练集比例去猜"，代表"不做任何学习"时的表现：

```python
def score(truth, p):
    labels = p.argmax(axis=1)                                  # 分类取概率最高的一类
    result = {
        'accuracy': accuracy_score(truth, labels),             # 准确率：主要指标
        'balanced_accuracy': balanced_accuracy_score(truth, labels),  # 对不均衡更公平
        'log_loss': log_loss(truth, p, labels=np.arange(len(classes))),
    }
    if len(classes) == 2:                                       # 二分类再补 AUC 与 MCC
        result.update(AUC=roc_auc_score(truth, p[:, 1]),
                      MCC=matthews_corrcoef(truth, labels))
    return result

def constant_prediction(target, n):
    # 常数基线：按标签在训练集里的比例生成预测，复制 n 份
    return np.tile(np.bincount(target, minlength=len(classes)) / len(target), (n, 1))

X = features(train)
baseline = constant_prediction(y[fit_idx], len(val_idx))       # 基线预测
model = model_fit(X.iloc[fit_idx], y[fit_idx])                 # CatBoost 预测
results = {'constant': score(y[val_idx], baseline),
           'boosting': score(y[val_idx], predict(model, X.iloc[val_idx]))}
```

我们把 `constant` 和 `boosting` 两行成绩写进 `validation_scores.csv`；把留出集的真实标签、预测类别、各类别概率写进 `validation_predictions.csv`；把混淆矩阵写进 `confusion_matrix.csv`。通常常数基线的准确率约等于多数类占比，而 CatBoost 若能明显超过它，就说明问卷特征里确实存在可学习的信号。

最后，我们对全部训练样本重新拟合一次，再**分块**预测测试集（每块 10 万行，避免内存峰值），并逐条核对提交文件的行数、id 顺序和列名：

```python
final = model_fit(X, y)                  # 用全量 12 万行样本重训固定配方
sample = pd.read_csv(DATA/'sample_submission.csv')
submission = sample.copy(); offset = 0
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    n = len(chunk)
    # 核对测试集列顺序与 id 顺序必须和样例文件完全一致
    assert chunk.columns.tolist() == [c for c in train.columns if c not in ['Depression']]
    assert np.array_equal(chunk[id_col].to_numpy(),
                          sample.iloc[offset:offset + n, 0].to_numpy())
    p = predict(final, features(chunk))  # 把概率转成类别标签，写回对应行
    submission.loc[submission.index[offset:offset + n],
                   ['Depression']] = np.asarray(classes)[p.argmax(axis=1)][:, None]
    offset += n
# 提交行数、id 唯一性、目标列无缺失这些契约逐条验证
assert offset == len(sample) and submission[id_col].is_unique
assert not submission[['Depression']].isna().any().any()
submission.to_csv(OUT/'submission.csv', index=False)
```

材料把这次提交定性为"基线结果，不等于证明这些特征可以在部署时照单全收"。整个实验把配置、采样审计、验证成绩、消融字段、声称、运行时长和软件版本全部写进 `run_manifest.json`，做到完全可复现。

---

## 三、总结

本文中我们完成了一次"谨慎对待问卷模式"的独立研究。我们首先用种子化的均匀水库抽样，把大问卷数据缩到最多 12 万行，并累计出全表缺失报告；其次用无状态的特征清洗（文本缺失填哨兵、数值无穷转缺失、不学全表统计量）避免信息泄漏，同时用完整样本分析、作答偏差、非随机缺失这几个视角审视缺失处理方式；接着把固定的 350 轮、深度 6 的 CatBoost 与"按比例猜"的常数基线同台对比，并预先声明第二个可证伪实验——去掉 "Have you ever had suicidal thoughts ?" 这一敏感字段做可用性消融；然后我们在同一批留出行上产出准确率、平衡准确率、对数损失、AUC 和 MCC 报告；最后分块预测测试集并逐条核对提交契约，全程不使用排行榜标签。

notebook 的旁路设计也值得一提：它把人分成 Junior/Middle/Senior 三档练习——Junior 练文本类别的规范化（Unicode、casefold、碰撞），Middle 练问卷选择与缺失机制（response-bias、complete-case、MNAR），Senior 练隐私与负责任解释（minimization、access-control、disclosure），提醒小样本亚组的报告也要避免暴露个体。材料还提到原赛公开第一名的方案（Mahdi Ravaghi 用 XGBoost、LightGBM 跑四条数据处理管线，再用 AutoGluon 的 OOF stacking 选出 24 个模型），那是作者的个人比赛结论，不是放之四海皆准的配方；我们的 notebook 是独立基线，并不声称复现那个分数。最后请务必记住：这份合成问卷练习，不能用来评估任何真实个体的心理健康。