# 合约、流失与迁移学习（Contracts, Churn and Transfer）

## 一、比赛简介

### 1.1 比赛背景

电信运营商最怕的事情之一，是用户「用着用着就走了」。这种用户主动停止使用服务的行为叫**流失（churn）**。获取一个新客户的成本远高于留住一个老客户，所以运营商愿意花钱预测「谁可能走」，然后提前做挽留：送流量、降月费、推荐更合适的套餐。

这个比赛的数据来自 Kaggle Playground 系列 **S6E3**，主办方是 Yao Yan、Walter Reade 和 Elizabeth Park，上游数据是 IBM 的电信客户流失样例数据集（`thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm`），以 CC BY-NC 4.0 协议发布，**仅限非商业教育用途**。数据是合成的，结构却高度仿真：每一行是一个电信客户，列包括：

- **人口信息**：性别、是否老年人、是否有伴侣、是否有子女；
- **账户信息**：在网时长（tenure）、合约类型（Contract）、付款方式、是否无纸化账单、月费、总费用；
- **服务信息**：电话、多条线路、网络服务、在线安全、在线备份、设备保护、技术支持、流媒体电视、流媒体电影。

材料给这份 notebook 的定位很特别：它不是普通的「刷分」笔记，而是一次**迁移实验**。作者前一个任务做的是银行数据，现在换到电信数据，问的是——「上一题里哪些假设还能用，哪些一换场景就失效？」这比单看分数有意思得多。

### 1.2 比赛目标

输入是一个客户的全部属性，输出是 `Churn` 列：这个客户会不会流失。提交文件只有两列，第一列是 `id`，第二列是 `Churn`，要求填 **0 到 1 之间的概率**（不是 0/1 硬标签）。因为业务上需要按风险排序，概率比标签更有用。

数据量不小，材料里用**分块读取 + 蓄水池抽样**，最多保留 12 万行训练，以控制内存和运行时间。

### 1.3 评价标准

指标是 **ROC AUC**。把所有客户按预测流失概率排队，随机抽一个真流失的、一个没流失的，AUC 就是「真流失的排得更靠前」的概率。0.5 等于乱猜，1.0 是完美排序。

AUC 的优点是**不受类别比例影响**：流失用户通常只占两三成，如果只看准确率，「全猜不流失」就能有七八十分，毫无意义；AUC 则专门衡量「能不能把两类人分开」。

材料里还反复强调一句话：**流失概率不等于「发一张优惠券能挽回他的概率」**。前者是相关性预测，后者是因果效应。用预测模型去指导干预，还需要额外的因果实验，这一点我们放在最后说。

---

## 二、解题过程

整份 notebook 的骨架和上一题的银行任务几乎一样：审计数据 → 无状态特征处理 → 固定 CatBoost → 固定 holdout 对比常数基线 → 消融实验 → 分块预测与格式校验。**一样的是流程，不一样的是「为什么」。** 我们按四步来讲。

### 2.1 数据探索（EDA）

第一步是把原始表读进来，并且**在读取过程中就做审计**：用了多少行、缺了多少、抽了多少。因为数据是分块读的，我们顺便把每块的缺失计数累加起来：

```python
import pandas as pd, numpy as np, json
from pathlib import Path

DATA = Path('/kaggle/input/contracts-churn-and-transfer-data')
SEED = 20261026
MAX_ROWS = 120000
rng = np.random.default_rng(SEED)

reservoir, missing, rows_seen = None, None, 0
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)

    # 给每行一个随机数，按它排序后只留前 12 万行
    # 效果等价于「从全量里均匀随机抽 12 万」，而不是只读文件开头
    chunk['_sampling_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_sampling_priority')

train = reservoir.drop(columns='_sampling_priority').reset_index(drop=True)
print(f'源数据 {rows_seen} 行，抽样后 {len(train)} 行')
```

接着检查标签和几个关键字段的分布：

```python
TARGET = 'Churn'
print(train[TARGET].value_counts(normalize=True))

# 合约类型是最值得看的类别字段：按月、一年、两年
print(train['Contract'].value_counts())

# 在网时长与流失的关系（分组看均值）
print(train.groupby('Contract')[TARGET].mean())
```

这一步给出两个重要观察：

1. **流失比例不高**，数据不均衡，必须用 AUC 而不是准确率；
2. **合约类型与流失强相关**：按月付费（month-to-month）的客户流失率明显高于签了一两年合约的客户。原因很直观——长期合约有违约金、有绑定优惠，客户「想走也得掂量」；按月合约随时能停，流失门槛最低。

这个发现直接催生了后面的消融实验：**如果去掉 `Contract`，模型会掉多少？**

材料还检查了缺失情况：这份合成数据整体比较完整，但仍有少量缺失，需要处理。原则和上一题一致：**标签缺失绝不能悄悄填补**，要直接报错停下。

```python
# 主键唯一，否则验证集划分会串行
assert train['id'].is_unique
# 标签有缺失就停下，不要用众数糊过去
assert not train[TARGET].isna().any().any()

# 把每列缺失数单独存一份，方便回查
pd.DataFrame({'field': missing.index,
              'missing_rows': missing.values.astype(int)}
            ).to_csv('missingness.csv', index=False)
```

我们还把「源数据多少行、抽样多少行、用了哪些文件、原始文件是否改动」写进一份审计记录（audit）。这是这份 notebook 最值得学的地方之一：**它不只告诉你结果，还告诉你结果是怎么来的**。别人拿到你的 notebook，应该能判断你有没有偷偷用了额外数据。

### 2.2 数据清洗

清洗的原则是**无状态**：所有变换都不依赖验证集，也不依赖全表统计量。这样验证分数才可信。

对这份电信数据，具体做三件事：

1. **数值列**：转成数字，把 `inf` / `-inf` 当缺失处理（财务里的「无穷大比率」通常是除零造成的，不是真实值）；
2. **类别列**：转成字符串，缺失单独标成 `'__MISSING__'`，让「没填」本身成为一种类别；
3. **不做手工编码**：不把 `Month-to-month` 映射成 0、`One year` 映射成 1、`Two year` 映射成 2。为什么？因为这种整数编码隐含了「两年合约 = 两个月合约的两倍」这种虚假的数值关系，对线性模型是灾难；对 CatBoost 虽然影响小，但也没必要。

```python
def features(frame, omit=()):
    """无状态特征处理；omit 用于消融实验，去掉指定列"""
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if not pd.api.types.is_numeric_dtype(x[c]):
            # 类别列：缺失单独成一类，保留「未填写」这个信息
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列：无穷值不是有效测量，先变成缺失交给 CatBoost
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return x
```

然后划验证集。用固定种子做 80/20 切分，并**按标签分层**：

```python
from sklearn.model_selection import train_test_split

y = train[TARGET].astype(int).to_numpy()
indices = np.arange(len(train))
fit_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=SEED, stratify=y)

# 双重保险：训练和验证绝不允许重叠
assert set(fit_idx).isdisjoint(val_idx)
```

分层的必要性：流失用户本来就少，如果随机切分碰巧让验证集里流失用户特别少，AUC 会抖得厉害，调参就变成了看运气。分层能保证两边比例一致。

还有一个纪律：**只切一次 holdout，不反复换**。很多选手会不停换种子、换切分，直到分数好看，这实际上是在验证集上过拟合。这份 notebook 从头到尾只用这一个固定切分。

### 2.3 模型训练与超参数调优

模型用 **CatBoost**。它属于梯度提升树家族：一棵一棵地加树，每棵新树专门修正前面所有树的残差。CatBoost 的特色是对**类别特征**有原生支持——它会在内部用「目标统计」的方式把类别转成数值，而不是简单独热，所以在类别多、样本少的时候通常比别的库稳。

超参数是**看分数之前就写死**的：

```python
from catboost import CatBoostClassifier

def model_fit(x, target):
    # 让 CatBoost 知道哪些列是文字列，它会自己处理
    cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()

    params = dict(
        iterations=350,      # 350 棵树
        depth=6,             # 每棵树深度 6
        learning_rate=0.08,  # 学习率
        random_seed=20261026,
        verbose=False,
        thread_count=4,
        allow_writing_files=False,   # 不产生临时文件
    )
    model = CatBoostClassifier(loss_function='Logloss', **params)
    model.fit(x, target, cat_features=cat_cols)
    return model
```

`loss_function='Logloss'` 就是二分类的对数损失，它优化的正是「预测概率与真实 0/1 的交叉熵」。注意这里**没有用 `auto_class_weights='Balanced'`**——虽然数据不均衡，但平衡权重会改变概率的绝对水平（把预测概率整体抬高），而 AUC 只看排序，加权对 AUC 帮助有限，却可能让概率失去校准意义。这是一个「指标决定预处理」的例子。

**常数基线。** 和上一题一样，我们先造一个「什么都不学」的模型：所有人都预测训练集的流失比例。对二分类来说，这个预测对每个客户都相同，排序完全随机，所以它的 **AUC 恒为 0.5**。这是判断 CatBoost 有没有学到东西的最低门槛。

```python
def constant_prediction(target, n):
    """常数基线：每行都是训练集的正例比例"""
    p = np.bincount(target, minlength=2) / len(target)
    return np.tile(p, (n, 1))
```

**第二个实验：去掉 `Contract`。** 这是整份材料设计的核心。前面 EDA 已经看到合约类型和流失强相关，那么：

- 如果去掉它之后 AUC 只掉一点点，说明模型从「在网时长、月费、服务数量」等字段里已经能间接推断出合约类型，`Contract` 是冗余的；
- 如果掉很多，说明模型严重依赖这个字段——而在真实业务里，合约类型也许并不是随时可得（比如新客户还没签约、或者数据同步延迟），这就是**可用性风险**。

材料强调：消融是**诊断**，不是「既然去掉也能跑，那就上线一个没有该字段的模型」的许可证。结论要如实报告，哪怕备选方案分数更高，也不能反过来改主方案。

```python
# 第二个实验在看分数之前就确定：去掉 Contract
reduced  = features(train, omit=['Contract'])
alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
alt_pred  = predict_proba(alternate, reduced.iloc[val_idx])

from sklearn.metrics import roc_auc_score
print('full  :', roc_auc_score(y[val_idx], pred))
print('ablate:', roc_auc_score(y[val_idx], alt_pred))
```

同一批训练行、同一批验证行、同一组超参数，唯一变量是特征集合——只有这样才能把「分数差异」归因到「特征差异」上。

### 2.4 模型测试与指标报告

验证阶段不能只看一个数。材料的评分函数一次给出五个指标：

```python
from sklearn.metrics import (roc_auc_score, log_loss, accuracy_score,
                             balanced_accuracy_score, matthews_corrcoef)

def score(truth, p):
    labels = (p >= 0.5).astype(int)
    return {
        'AUC': float(roc_auc_score(truth, p)),                  # 排序能力（比赛指标）
        'log_loss': float(log_loss(truth, p)),                  # 概率质量
        'accuracy': float(accuracy_score(truth, labels)),       # 阈值 0.5 下的准确率
        'balanced_accuracy': float(balanced_accuracy_score(truth, labels)),
        'MCC': float(matthews_corrcoef(truth, labels)),         # 不平衡下更稳健
    }

results = {'constant': score(y[val_idx], baseline),
           'boosting': score(y[val_idx], pred)}
```

为什么要都看？

- **AUC** 是比赛指标，但它对概率的绝对大小完全不敏感；把所有人的预测概率开平方，AUC 一点不变；
- **log loss** 会惩罚「自信地预测错」，因此能反映概率有没有校准；
- **balanced accuracy** 是正负两类召回率的平均，不会被多数类淹没；
- **MCC** 同时看混淆矩阵的四个格子，「全猜不流失」的模型 MCC 接近 0，一眼就能看出问题。

材料还导出混淆矩阵和逐样本预测表：

```python
from sklearn.metrics import confusion_matrix

pd.DataFrame(confusion_matrix(y[val_idx], (pred >= 0.5).astype(int)),
             index=['actual_0', 'actual_1'],
             columns=['pred_0', 'pred_1']).to_csv('confusion_matrix.csv')

pd.DataFrame({'id': train.iloc[val_idx]['id'].to_numpy(),
              'actual': y[val_idx],
              'prediction': pred}).to_csv('validation_predictions.csv', index=False)
```

逐样本文件是后续做**校准曲线**、**按风险分位看挽留收益**的基础，不必重跑模型。

**最终提交。** 用全部 12 万行重训，再对测试集分块预测。分块是为了控制内存，同时也顺便做了格式校验：

```python
final = model_fit(X, y)                     # 全量重训
sample = pd.read_csv(DATA/'sample_submission.csv')
submission = sample.copy()
offset = 0

for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    # 测试集列顺序必须和训练特征一致（去掉标签列）
    assert chunk.columns.tolist() == [c for c in train.columns if c != TARGET]
    n = len(chunk)
    # 行顺序必须和样例文件一致，否则预测会张冠李戴
    assert np.array_equal(chunk['id'].to_numpy(),
                          sample.iloc[offset:offset+n, 0].to_numpy())

    p = predict_proba(final, features(chunk))
    submission.loc[submission.index[offset:offset+n], 'Churn'] = p
    offset += n

assert offset == len(sample)
assert submission['Churn'].between(0, 1).all()   # 概率必须在 0~1
submission.to_csv('submission.csv', index=False)
```

最后再把保存的文件读回来，核对列名、行数、ID 顺序。这些检查很啰嗦，但提交格式错误是比赛里最常见的白丢分方式。

**关于分数。** 材料里没有给出具体的验证 AUC 数值，我们不编造。方法上可以确定的是：CatBoost 作为非线性模型，在含大量类别字段的表格上通常会明显优于常数基线（后者的 AUC 恒为 0.5）；而「去掉 Contract 后 AUC 掉多少」这个数字，才是这次迁移实验真正想让人看到的东西。

材料里还提到一个外部参照：选手 Traiko Dinev 在这场比赛里拿到第 3 名，做法是把 100 份 out-of-fold 预测文件（来自自己模型和公开 notebook）做**简单线性回归集成**，反而打败了他的爬山法和岭回归方案；一个浅层类别树模型提供了有用的多样性。作者也提醒：混用不同的折数和随机种子，可能把 CV 与排行榜的差距拉大。这份 notebook 明确声明**不复现该成绩**，只是独立基线。把「别人的名次」和「自己的基线」分开说，是研究该有的诚实。

---

## 三、总结

本文中我们完成了一次从银行任务迁移到电信流失预测的练习。我们首先做了数据审计：分块读取、累加逐列缺失、用蓄水池抽样均匀保留最多 12 万行，并确认 ID 唯一、标签无缺失。其次我们采用无状态清洗：数值列转数字并把无穷值变缺失，类别列把缺失单独标成一类、不做会引入虚假顺序的整数编码。接着我们用预设的 CatBoost（350 轮、深度 6、学习率 0.08、Logloss）训练，以常数基线为参照，在固定 20% 分层 holdout 上比较；第二个实验在完全相同的切分下去掉 `Contract` 字段，用来检验模型对合约类型的依赖。最后我们同时报告 AUC、log loss、balanced accuracy、MCC 和混淆矩阵，导出逐样本预测，再用全量数据重训、分块预测、多重校验后生成提交文件。

这次迁移给我们的最大启发是：**流程可以照搬，假设不能照搬**。银行任务里「用固定 holdout + 常数基线 + 无状态预处理」这套纪律在电信数据上依然成立；但「哪些特征强、缺失意味着什么、概率能不能直接指导决策」这些问题，每换一个场景都要重新回答。尤其是最后一点——模型给出的是「谁会走」的相关性预测，而不是「给谁发券能留住」的因果效果，把两者混为一谈，是数据科学里最昂贵也最常见的错误。