# Tempo From a Table（从表格里预测节拍）

## 一、比赛简介

### 1.1 比赛背景

Tempo From a Table 是 Kaggle Playground 系列比赛中的一员（代号 S5E9）。Playground 是 Kaggle 官方用**合成数据**（synthetically generated data，也就是机器按一定规则生成、故意带有某种规律的假数据）做的练习赛，目的是让参赛者可以在接近真实比赛的流程里练习建模，数据也容易获取。这个比赛的贡献者是 Walter Reade 和 Elizabeth Park。

比赛的名字有两层意思：Tempo（节拍）指音乐里的速度，通常用**BPM**（Beats Per Minute，每分钟节拍数）来表示；From a Table（从一张表格）则点明了一个关键前提——**数据和音频没有任何关系**，我们要做的只是从一个普通的表格（tabular）里预测 BPM。材料里反复强调这句话：`Tabular tempo prediction is not an audio-analysis system unless audio is actually provided.` 意思是说，除非真的给你音频，否则不能把它当成音频分析任务，老老实实做表格回归就行。

这个比赛有一个值得注意的难点。音乐里的节拍是有歧义的：同一段节奏，你既可以把它数成每秒一拍，也可以数成每半秒一拍，也就是 **60 BPM 和 120 BPM 可能听起来都合理**（一个是「正统」速度，一个是它的「倍速」）。这类「正好差一倍」的错误叫做**半速 / 双速误差（half-tempo / double-tempo alias）**。换句话说，一个只报平均数的模型，很可能在这种歧义上出系统性偏差。所以比赛的主线思路是：**在加任何花哨特征之前，先诊断预测残差的规律**。

### 1.2 比赛目标

这是一个**回归任务**。输入是一张表格，其中有一个样本编号列（id），若干特征列，以及我们真正要预测的目标列：

- `BeatsPerMinute`：一条样本对应的节拍数（BPM），是浮点数。

训练集里每一行都给出一首（或一段）音乐的表格化特征以及它的 BPM 标签。测试集和训练集的特征完全一样，但没有 `BeatsPerMinute` 这一列。我们要做的，就是给测试集每一行预测一个 BPM 值，然后按照 `sample_submission.csv` 的格式（id 顺序保持一致）写入提交文件。

### 1.3 评价标准

本场比赛使用 **RMSE（Root Mean Squared Error，均方根误差）** 作为评价指标。它的计算方法是：先计算每个样本「预测值减去真实值」的误差，把误差平方后求平均，再开平方根。小学过的概念是「方差开根号」，这里只是把「方差」换成「误差平方的平均」。RMSE **越小越好**，模型预测得越准，这一项分数越低。

> 小提示：RMSE 对「偏差很大的样本」很敏感——因为要先平方。所以如果有一批样本集体犯「半速 / 双速」的错误，RMSE 会被拉得很高，这正是我们后面要重点检查残差的原因。

---

## 二、解题过程

我们的整体思路可以概括成一句话：**用一个配置固定、不做任何调参的 CatBoost 回归模型作为基线，同时训练一个「永远输出平均值」的常数模型作对比，再用残差分析来诊断节拍歧义。** 下面按 EDA、数据清洗、模型训练、测试报告四步来讲。

### 2.1 数据探索（EDA）

第一步，我们把训练集读进来，看看它有多大、有哪些列、有没有缺失值。训练集的完整文件可能很大（Playground 比赛常常是几十万行），所以材料里的做法是**分块读取，并用「蓄水池采样」只保留最多 120,000 行**，这样既省内存，又不破坏随机性：

```python
# 导入常用库：numpy 做数值计算，pandas 读写表格
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor        # 我们要用的梯度提升树模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 比赛配置：目标是 BeatsPerMinute，这是回归任务
CONFIG = {'targets': ['BeatsPerMinute'], 'family': 'regression'}
SEED = 20261026                                 # 固定随机种子，结果可复现
np.random.seed(SEED)

# 限制训练样本上限，防止内存爆炸
MAX_ROWS = 120000
reservoir = None                                # 蓄水池：始终只保留最好的 120000 行
rows_seen = 0                                   # 记录原始训练集一共有多少行
rng = np.random.default_rng(SEED)               # 独立的随机数发生器，保证可复现
for chunk in pd.read_csv('train.csv', chunksize=100000):   # 每 10 万行读一次
    rows_seen += len(chunk)
    # 给每一行随机一个优先级，值越小越优先保留
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk])
    # nsmallest 只留下优先级最小的 120000 行，多余的行被丢弃
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')
train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
```

读完之后，我们把「源数据行数、最终采样行数、特征列、目标列」整理成一份审计报告，同时统计每一列的缺失值数量。这一步能让我们快速确认两件事：**样本规模有多大，以及哪一列数据不完整**：

```python
# 把每一列的缺失值数量存下来，方便诊断
missing = train.isna().sum()
pd.DataFrame({'field': missing.index,
              'missing_rows': missing.values}).to_csv('missingness.csv', index=False)

# 打印这次实验的数据概览（audit 这份字典后来也会写进清单）
audit = {'source_rows': rows_seen, 'sampled_rows': len(train),
         'features': feature_columns, 'targets': CONFIG['targets']}
print(audit)
```

这里需要说明：材料里没有把我们看到的缺少数值列出来，所以我们不编造具体的缺失比例。但**检查缺失、把检查结果落成文件**这个动作本身就是 EDA 的关键——后面所有特征才会交给模型。

### 2.2 数据清洗

这份表格的清洗方式很轻。材料里的 `features()` 函数把整个清洗流程封装了起来，我们只需要调用它就能得到模型可以直接吃进去的特征表：

```python
# 特征清洗：输入原始表，输出模型能用的特征表
def features(frame, omit=()):
    # omit 表示要临时去掉的列，后面做消融实验会用上
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if not pd.api.types.is_numeric_dtype(x[c]):
            # 非数值列统一转成字符串，缺失值填成 __MISSING__，
            # 这样 CatBoost 能把它们当成一个「是缺失」的特殊类别
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列强制转成 float，并拒绝无穷大（变成 NaN）
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return x

X = features(train)      # 一次调用，得到全部特征
```

这段代码的要点在于**分类特征和缺失值的处理方式**。CatBoost 天生支持类别型特征和缺失值，所以我们对类别列做的只是「把缺失变成字符串 `__MISSING__`」，而不是去猜测填补。对数值列则要做防御性转换：防止文本或无穷大混进模型。注意我们**没有做任何目标编码或全表填充**——因为验证集的标签不能提前用来学习任何统计量，否则会造成信息泄漏。

### 2.3 模型训练与超参数调优

这一节我们做三件事：切出验证集、训练「常数基线」、训练真正的 CatBoost 模型。

首先把数据按 8:2 切分，20% 留作验证集。因为这是回归问题，切分时不需要分层（分层主要是分类任务保证类别比例用）：

```python
# 目标值转成浮点数
y = train[CONFIG['targets'][0]].astype(float).to_numpy()
indices = np.arange(len(train))

# 随机切分：80% 训练、20% 验证
fit_idx, val_idx = train_test_split(indices, test_size=.2, random_state=SEED)
```

接着写一个「常数基线」。它不学任何东西，只是永远输出训练集 BPM 的平均值。虽然它显然预测不准，但它是一个很好的**参照物**：如果我们的模型连常数基线都打不过，那说明模型完全没学到规律；只有明显超过它，模型才是有用的：

```python
# 常数预测：不论哪一行，都输出训练集目标的均值
def constant_prediction(target, n):
    const = target.mean()          # 训练集 BPM 的平均值
    return np.full(n, const)       # 生成 n 个一模一样的预测

baseline = constant_prediction(y[fit_idx], len(val_idx))
```

然后是正式模型。这个比赛有一条很重要的原则：**参数是事先固定的，不根据验证集来回调**。材料里把 CatBoost 的参数写死为：350 轮迭代、树深度 6、学习率 0.08。这样做的理由是，如果看着验证集分数去改参数，就相当于「偷偷看了答案」，最后拿到排行榜上的分数会失真。我们忠实照做：

```python
# 固定的模型配方：不调参，一次性训练
def model_fit(x, target):
    params = dict(iterations=350, depth=6, learning_rate=.08,
                  random_seed=SEED, verbose=False, thread_count=4,
                  allow_writing_files=False)
    # 回归任务用 RMSE 作为损失函数，也就是比赛的评价指标
    model = CatBoostRegressor(loss_function='RMSE', **params)
    model.fit(x, target)           # x 里已经包含类别型字符串列，交给 CatBoost 自行处理
    return model

model = model_fit(X.iloc[fit_idx], y[fit_idx])   # 只在训练子集上拟合
pred = model.predict(X.iloc[val_idx])            # 在验证子集上预测
```

我们把「常数基线」和「CatBoost」两种预测放在同一个函数里评分，方便对比：

```python
# RMSE：预测与真实的误差平方后取平均，再开根号
def score(truth, p):
    return {'RMSE': float(np.sqrt(mean_squared_error(truth, p)))}

results = {'constant': score(y[val_idx], baseline), 'boosting': score(y[val_idx], pred)}
print(results)
```

材料给出的是完整脚本，但没有附上运行后的数字，所以我们在这里**不编造具体的 RMSE 数值**。读者应该记住的判断方式是：拿到打印结果后，先看 `boosting` 是否明显小于 `constant`——如果差得不多，说明特征或模型没有学到有效信息。

### 2.4 模型测试与指标报告

模型测试阶段我们做三件事：残差分档诊断、半速 / 双速检查、以及去掉一个特征再跑一遍（消融实验）。

**第一，残差分档。** RMSE 是一个总分，它会把不同情况混在一起。我们把验证集按真实 BPM 分成 5 档，分别看每一档的平均误差（bias）和平均绝对误差（MAE）。如果某些档位偏差特别大，说明模型在那一档有系统性问题：

```python
# 把预测和真实值放在同一张表里，算出残差
diagnostics = pd.DataFrame({'actual': y[val_idx], 'prediction': pred})
# 残差 = 预测 - 真实，负数表示低估，正数表示高估
diagnostics['error'] = diagnostics['prediction'] - diagnostics['actual']
# 按真实 BPM 用分位数切成 5 档，这里用 duplicates='drop' 防止有重复分位点报错
diagnostics['target_band'] = pd.qcut(diagnostics['actual'], 5, duplicates='drop')

# 每档的样本数、平均绝对误差、平均偏差
diagnostics.groupby('target_band', observed=True).agg(
    rows=('error', 'size'),
    MAE=('error', lambda s: s.abs().mean()),
    bias=('error', 'mean'),
)
```

**第二，半速 / 双速检查。** 这是本场比赛最有特色的分析。回想 1.1 节说的节拍歧义：60 BPM 和 120 BPM 对同一段节奏都可能成立。材料给我们的检查方法很明确：**看「预测值 / 真实值」的比值，是否聚集在 0.5 或 2 附近**（用真实的非零目标来算，并且明确单位是 BPM）。下面这段诊断代码就是根据材料的这个建议整理的思路：

```python
# 半速 / 双速检查：先排除真实值为 0 的行（0 没法做除法）
valid = diagnostics[diagnostics['actual'] > 0].copy()
# 预测 / 真实：比值接近 1 是正常，接近 0.5 是“预测成半速”，接近 2 是“预测成双速”
valid['ratio'] = valid['prediction'] / valid['actual']

# 统计三种情况各有多少条
ok      = ((valid['ratio'] - 1.0).abs() < 0.05).sum()
half    = ((valid['ratio'] - 0.5).abs() < 0.05).sum()
double  = ((valid['ratio'] - 2.0).abs() < 0.05).sum()
print(f'正常: {ok}, 半速: {half}, 双速: {double}')
```

如果 `half` 或 `double` 的条数明显偏多，说明模型系统性「看低了一半速度」或「看高了一倍速度」——这不是随机误差，而是节拍歧义导致的**系统性偏差**。找到这种模式，比盲目加特征重要得多。

**第三，消融实验。** 材料里设计的第二个实验很克制：只把 `RhythmScore` 这一列去掉，在**完全相同**的验证集上再训练一遍，比较 RMSE。这个实验回答的问题是：「`RhythmScore` 这个字段到底有没有用？」注意消融只是诊断，不等于说我们要根据结果马上调模型：

```python
# 第二个实验：去掉 RhythmScore，在同样的划分上重跑一遍
omit = ['RhythmScore']
reduced = features(train, omit)
alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
alt_pred = alternate.predict(reduced.iloc[val_idx])
results['availability_ablation'] = score(y[val_idx], alt_pred)
print(results)
```

最后，基线模型确认没问题之后，我们用**全部 120,000 条采样样本**重新训练一次（训练数据多喂一点总没坏处），然后**分块预测测试集**——因为测试集也可能很大，一次读进来会占内存。预测时特别注意保证 id 顺序和 `sample_submission.csv` 完全一致，这是提交格式的硬性要求：

```python
# 用完整采样样本重训一次，作为最终提交模型
final = model_fit(X, y)

sample = pd.read_csv('sample_submission.csv')
submission = sample.copy()
offset = 0
# 分块读测试集，逐块预测并填回对应位置
for chunk in pd.read_csv('test.csv', chunksize=100000):
    p = final.predict(features(chunk))
    submission.loc[submission.index[offset:offset+len(chunk)], 'BeatsPerMinute'] = p
    offset += len(chunk)

# 提交前自检：没有缺失值、id 不重复、行数和样例一致
assert not submission['BeatsPerMinute'].isna().any()
assert submission['id'].is_unique
submission.to_csv('submission.csv', index=False)
```

到这里，一次完整的、可复现的提交就生成了。整个过程我们没有接触排行榜上的隐藏标签，也没有用验证集去「调」参数，保证了分数是可信的。

---

## 三、总结

本文中我们完成了一场「从表格预测节拍」的回归比赛。我们首先用蓄水池采样的方式读取了训练数据（最多保留 120,000 行），并统计了每一列的缺失情况；其次我们用 `features()` 函数做过很轻的数据清洗——类别列缺失值转成 `__MISSING__`、数值列防御性转 float；接着我们按照材料给定的固定配方训练了 CatBoost 回归模型（350 轮、深度 6、学习率 0.08），同时训练了一个「输出平均值」的常数基线作为参照；然后我们在固定的 20% 验证集上对比了两者的 RMSE，做了残差分档分析，并用「预测 / 真实」的比例检查是否存在半速（0.5 倍）和双速（2 倍）的系统性偏差；最后我们还做了一个只去掉 `RhythmScore` 的消融实验，并用全量采样样本重训、分块预测测试集、校验格式后生成提交文件。

这场练习赛给我们的最大收获有两点。第一，**总分不可信，要看残差的结构**：RMSE 一个数字可能把「半速 / 双速」这种规律性错误全部掩盖掉，只有把残差按档位拆开、按比例检查，才能发现模型真正的弱点。第二，**调参之前要先定好「游戏规则」**：参数事先固定、消融和基线都在同一份验证集上对比、验证集标签绝不参与预处理——这样才能保证我们测到的分数是模型真实水平的反映，而不是「看着答案作答」的幻觉。