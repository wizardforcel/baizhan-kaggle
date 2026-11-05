# 临床标签作为学习问题（Clinical Label as a Learning Problem）

## 一、比赛简介

### 1.1 比赛背景

这份材料对应的题目是 Kaggle Playground 系列里的 **S6E2 —— 心脏病预测**。Playground 比赛的特点是：数据由主办方合成（synthetic），结构模仿真实场景，但做了脱敏处理，任何人都能免费参加，适合练手。

原始数据来自 `neurocipher/heartdisease`，主办方是 Yao Yan、Walter Reade 和 Elizabeth Park，数据以 CC BY 4.0 协议发布。合成数据里有很多「看起来很医学」的列名，比如胆固醇、心电图结果、铊（Thallium）负荷试验结果等。这里必须先把话说清楚：**这是一份合成数据，列名只是模仿临床，不能据此做任何诊断或治疗判断**。材料本身在开头就反复强调这一点，我们在写作时也保留这个立场。

这份 notebook 的定位不是「冲榜」，而是一个**可复现的学习基线**：它想回答的问题是「在这张表上，一个固定配置的梯度提升模型，比一个只会猜平均值的常数基线好多少？」，以及「去掉某个特征后，分数会怎么变？」——用一个可证伪的实验来检验「哪些信息真的有用」。

### 1.2 比赛目标

输入是一张表格，每一行是一个受试者，列是各种检查指标；输出是一个概率：这个人患有心脏病（`Heart Disease`）的可能性有多大。提交文件只有两列：`id` 和 `Heart Disease`，值是 0 到 1 之间的概率。

数据量不小（原始训练集被切成 10 万行一块分块读取），而且有缺失值、类别型字段。材料里的做法是：从训练集里**均匀随机抽最多 12 万行**来训练，剩下的不参与，目的是控制内存和运行时间。

### 1.3 评价标准

比赛指标是 **ROC AUC**（Area Under the ROC Curve）。用大白话说：把所有样本按预测概率从高到低排好，随机挑一个真病人和一个健康人，AUC 就是「病人排在健康人前面」的概率。0.5 等于瞎猜，1.0 是完美排序。

AUC 只看**排序**，不看概率的绝对大小。所以材料里特意提醒：一个模型 AUC 高，不代表它的概率可以直接当「患病风险百分比」用——后者需要看校准（calibration）。

---

## 二、解题过程

整个 notebook 的设计很克制：**先定规则，再看分数**。所有超参数、抽样上限、验证集划分、消融特征，都在看验证分数之前写死；不根据排行榜调模型。下面按 EDA、清洗、训练、报告四步讲。

### 2.1 数据探索（EDA）

第一步不是建模，而是**做数据审计**：数据从哪来、多少行、缺多少、标签有没有缺。因为我们用分块读取，顺手记录每块的缺失计数并累加：

```python
import pandas as pd, numpy as np, json
from pathlib import Path

DATA = Path('/kaggle/input/clinical-label-as-a-learning-problem-data')
SEED = 20261026
MAX_ROWS = 120000          # 抽样上限，先定死，避免后面偷偷调
rng = np.random.default_rng(SEED)

missing = None
rows_seen = 0
reservoir = None
# 分块读，块大小 10 万行；大表整读容易爆内存
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
    # 给每行一个随机优先级，最后留下优先级最小的 12 万行
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
print(rows_seen, len(train))
```

这里用了一个叫**蓄水池抽样**的技巧：边读边随机保留，保证最终留下的是全量数据的一个均匀随机子集，而不是「只读前 12 万行」。后者会让样本偏向文件开头，可能引入顺序偏差。

接着检查三件事：ID 是否唯一、标签是否有缺失、类别是否只有两种。材料里的原则是「**绝不悄悄填补标签**」——标签缺了就报错停下，而不是用 0 或众数糊过去。

```python
# 主键必须唯一，否则后面的验证集划分会串行
assert train['id'].is_unique
# 标签缺失不能静默填补
assert not train['Heart Disease'].isna().any().any()

# 看正负样本比例，决定后面是否需要 class_weight
print(train['Heart Disease'].value_counts(normalize=True))

# 把缺失情况单独存一份，方便回头查是哪些字段缺
pd.DataFrame({'field': missing.index,
              'missing_rows': missing.values.astype(int)}
            ).to_csv('missingness.csv', index=False)
```

我们还把「用了哪些文件、抽了多少行、原始文件是否改动」写进一个 JSON 审计记录。这一步在真实项目里很重要：**别人拿到你的 notebook，应该能知道你到底喂了什么数据**。

材料里还有一个值得注意的细节：数据目录里除了 `train.csv`、`test.csv`、`sample_submission.csv`，可能还躺着额外的训练文件。这份基线**明确不使用它们**，并在审计里列出文件名。用不用是一回事，隐瞒不隐瞒是另一回事。

### 2.2 数据清洗

这份数据的清洗原则是「**无状态**」：任何变换都不能依赖验证集或全表的统计量，否则验证分数会虚高。

具体做三件事：

1. 数值列：强制转成数字，把 `inf` / `-inf` 换成缺失（`NaN`）。CatBoost 原生支持缺失值，不需要我们填。
2. 类别列：转成字符串，缺失填一个特殊标记 `'__MISSING__'`。这样「缺失」本身成为一种类别信息，而不是被平均值抹掉。
3. 日期列：如果配置里有日期字段，就拆出年、月、星期几，然后把原列删掉。

```python
def features(frame, omit=()):
    """把原始表变成模型输入；omit 用于消融实验（去掉指定列）"""
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if c in CONFIG.get('date_columns', []):
            dt = pd.to_datetime(x.pop(c), errors='coerce')
            x[c+'_year']    = dt.dt.year
            x[c+'_month']   = dt.dt.month
            x[c+'_weekday'] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 类别列：缺失单独成一类，保留「没测」这个信息
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列：无穷值不是有效测量，先变成缺失
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return x
```

注意这个函数**没有任何 fit 步骤**：同样的输入永远得到同样的输出，不存在「用训练集均值填补测试集」这种会泄漏的操作。CatBoost 拿到类别列后，会在内部用目标统计做编码，但那是在训练折内部完成的。

然后划验证集。用固定随机种子做 80/20 切分，并对标签做**分层抽样**（`stratify=y`），保证验证集的正负比例和训练集一致——否则如果验证集里病人特别少，AUC 会抖得很厉害。

```python
from sklearn.model_selection import train_test_split

y = train['Heart Disease'].astype(int).to_numpy()
indices = np.arange(len(train))
fit_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=SEED, stratify=y)
assert set(fit_idx).isdisjoint(val_idx)   # 双重保险：两边绝不重叠
```

这里还有一个隐含的纪律：**验证集只切一次**。很多比赛选手会反复换验证集、换种子，直到分数好看为止，这本质上是在验证集上调参，最后排行榜会打脸。材料里的做法是固定一个 holdout，从头用到尾。

### 2.3 模型训练与超参数调优

模型用 **CatBoost**，一种梯度提升决策树。它的特点是：对类别特征有原生支持，不需要独热编码；对缺失值友好；在小中型表格数据上通常开箱即用就很强。

超参数是**事先写死**的，没有网格搜索：

- `iterations=350`：树的数量；
- `depth=6`：每棵树的深度；
- `learning_rate=0.08`：学习率；
- `random_seed=SEED`：固定种子；
- `thread_count=4`：限制线程数，保证在 Kaggle 上稳定。

```python
from catboost import CatBoostClassifier

def model_fit(x, target):
    # CatBoost 需要知道哪些列是类别列
    cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()
    params = dict(
        iterations=350,        # 350 棵树，够用且不至于太慢
        depth=6,               # 深度 6 是表格数据的常用起点
        learning_rate=0.08,
        random_seed=20261026,  # 固定种子，结果可复现
        verbose=False,
        thread_count=4,
        allow_writing_files=False,  # 不在磁盘上留临时文件
    )
    model = CatBoostClassifier(loss_function='Logloss', **params)
    model.fit(x, target, cat_features=cat_cols)
    return model
```

**为什么要设一个常数基线？** 因为只有和「什么都不学」比，才知道模型到底学没学到东西。常数基线就是把训练集的正例比例当成所有人的预测概率。对二分类来说，这个预测对所有样本都一样，排序完全随机，所以它的 AUC 理论上就是 **0.5**。如果 CatBoost 的 AUC 没有明显高于 0.5，那这个模型就没有价值。

```python
def constant_prediction(target, n):
    """常数基线：所有人都预测成训练集的正例比例"""
    p = np.bincount(target, minlength=2) / len(target)
    return np.tile(p, (n, 1))   # 每行都是同一个概率向量

def predict_proba(model, x):
    """取正类的概率，即第 1 列"""
    return model.predict_proba(x)[:, 1]
```

**可证伪的第二实验：去掉「铊」（Thallium）。** 这是整份材料最有意思的设计。铊负荷试验是心脏检查里信息量很高的项目，但它在很多场景下并不可得（设备、成本、流程限制）。如果我们去掉它之后模型分数几乎不变，说明模型没有依赖这个「现实中可能拿不到」的字段；如果分数掉很多，那就说明模型的强依赖建立在一个部署时未必存在的条件上——这是**可用性风险**，不是「再调调参就能解决」的问题。

材料强调：这个消融是**诊断**，不是「既然去掉也行，那就部署一个没有该字段的模型」的许可。实验结论要如实报告，哪怕备选方案分数更高，也不能反过来修改主方案。

```python
# 第二实验在查看验证分数之前就指定好：去掉 Thallium
omit = [c for c in ['Thallium'] if c in feature_columns]
reduced = features(train, omit)

# 用完全相同的训练行和验证行，只改特征
alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
alt_pred  = predict_proba(alternate, reduced.iloc[val_idx])

from sklearn.metrics import roc_auc_score
print('full :', roc_auc_score(y[val_idx], pred))
print('ablate:', roc_auc_score(y[val_idx], alt_pred))
```

同一份数据、同一个切分、同一组超参数，唯一变量是特征集合——这才叫**受控实验**。如果训练行换了、种子换了，分数差异就分不清是特征造成的还是随机波动造成的。

### 2.4 模型测试与指标报告

验证阶段我们报告的指标不止 AUC，还有一组「概率质量」指标，材料里的 `score` 函数一次算全：

```python
from sklearn.metrics import (roc_auc_score, log_loss, accuracy_score,
                             balanced_accuracy_score, matthews_corrcoef,
                             confusion_matrix)

def score(truth, p):
    labels = (p >= 0.5).astype(int)   # 概率转成 0/1 预测
    return {
        'AUC': float(roc_auc_score(truth, p)),          # 排序能力
        'log_loss': float(log_loss(truth, p)),          # 概率校准 + 排序
        'accuracy': float(accuracy_score(truth, labels)),
        'balanced_accuracy': float(balanced_accuracy_score(truth, labels)),
        'MCC': float(matthews_corrcoef(truth, labels)), # 对类别不平衡更稳健
    }

results = {
    'constant': score(y[val_idx], baseline),   # 常数基线
    'boosting': score(y[val_idx], pred),       # CatBoost
}
```

为什么 AUC 之外还要看 log loss 和 MCC？

- **AUC** 只关心排序，把所有预测概率整体压扁或拉伸都不影响它；
- **log loss** 关心概率的绝对数值，预测「0.9 而实际是 0」会被重罚，因此能反映校准；
- **MCC**（马修斯相关系数）同时考虑了四个格子（真阳、真阴、假阳、假阴），在正负样本不均衡时比 accuracy 更诚实——一个把所有样本都预测成「健康」的模型，accuracy 可能很高，但 MCC 接近 0。

材料还导出了**混淆矩阵**和逐样本的预测表：

```python
# 混淆矩阵：看清错误集中在哪一类
cm = confusion_matrix(y[val_idx], (pred >= 0.5).astype(int))
pd.DataFrame(cm, index=['actual_0', 'actual_1'],
                 columns=['pred_0', 'pred_1']).to_csv('confusion_matrix.csv')

# 逐样本留档：id、真实标签、预测概率
diag = pd.DataFrame({
    'id': train.iloc[val_idx]['id'].to_numpy(),
    'actual': y[val_idx],
    'prediction': pred,
})
diag.to_csv('validation_predictions.csv', index=False)
```

这份逐样本文件很有用：后续如果要做校准曲线（把预测概率分箱，看每箱的实际患病率），直接拿它就能画，不用重跑模型。

**最终提交。** 验证通过后，用**全部抽样数据**重新训练一次，再对测试集分块预测。分块的原因和读数据一样：测试集可能很大，一次性读进来内存吃不消。

```python
# 用全部 12 万行重训，特征配置保持「全特征」不变
final = model_fit(X, y)

sample = pd.read_csv(DATA/'sample_submission.csv')
submission = sample.copy()
offset = 0

for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    # 测试集列顺序必须和训练集一致（去掉标签列）
    assert chunk.columns.tolist() == [c for c in train.columns if c != 'Heart Disease']
    n = len(chunk)
    # 行顺序必须和 sample_submission 对齐，否则分数会张冠李戴
    assert np.array_equal(chunk['id'].to_numpy(),
                          sample.iloc[offset:offset+n, 0].to_numpy())
    p = predict_proba(final, features(chunk))
    submission.loc[submission.index[offset:offset+n], 'Heart Disease'] = p
    offset += n

assert offset == len(sample)
assert not submission['Heart Disease'].isna().any()
submission.to_csv('submission.csv', index=False)
```

最后做一次「读回来再检查」的往返验证：重新读入 `submission.csv`，确认列名和样例完全一致、行数一致、ID 顺序一致、概率都在 0 到 1 之间。这些检查看起来啰嗦，但提交格式错误是比赛里最常见的白丢分方式。

关于分数，材料里没有给出具体的验证集 AUC 数值，我们也不编造。能确定的只有方法层面的结论：CatBoost 作为非线性模型，在绝大多数表格任务上都会明显优于常数基线（后者的 AUC 恒为 0.5）；而「去掉铊之后掉多少」这个数字，才是这份实验真正想让人看到的东西——它衡量的是模型对「可能拿不到的字段」的依赖程度。

材料里还提到一个外部参照：另一位作者 Bala Baskar 在这场比赛里拿到第 10 名，用的方法是 10 折分层交叉验证产生 out-of-fold 预测、数值与类别两种表示、折内交互特征、原始数据统计量，最后在 logit 空间做集成。这份 notebook 明确声明**不声称复现那个成绩**，只是独立基线。这种「把别人的成绩和自己的基线分开说」的态度，本身也是做研究该有的样子。

---

## 三、总结

本文中我们完成了一次以「临床标签」为主题的可复现建模练习。我们首先做了数据审计：分块读取、记录全量行数与逐列缺失、用蓄水池抽样均匀保留 12 万行，并确认 ID 唯一、标签无缺失、类别只有两种。其次我们采用无状态的特征处理——数值列转数字并把无穷值变缺失、类别列把缺失单独标成一类、日期列拆出年月日，全程不使用验证集或全表统计量。接着我们用事先写死的 CatBoost 配置（350 棵树、深度 6、学习率 0.08）训练，并设置常数基线作为参照，在同一个固定 20% 分层 holdout 上比较；第二个实验在完全相同的切分下去掉 Thallium 字段，用来检验模型对「可能不可得特征」的依赖。最后我们把 AUC、log loss、balanced accuracy、MCC 和混淆矩阵一并报告，导出逐样本预测留档，再用全量抽样数据重训、分块预测、多重校验后生成提交文件。

这个练习给我们最大的启发是：**在医疗相关的数据上，「分数高」远不是终点**。AUC 高只说明排序好，不代表概率能当风险用；一个字段在训练集里很强，不代表部署时拿得到；一次 holdout 上的好成绩，也不等于通过了外部验证。材料里反复强调的「不做诊断、不做治疗、不部署」，不是客套话，而是这类任务的基本边界。
