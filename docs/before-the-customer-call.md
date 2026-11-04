# 打电话之前：银行客户认购预测（Before the Customer Call）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的材料来自 Kaggle Playground 系列第 5 季第 8 场（S5E8），题目叫 Before the Customer Call。它的原始数据是 UCI 的 Bank Marketing 数据集：一家葡萄牙银行做电话营销，向客户推销定期存款（term deposit，就是约定存期、利率比活期高的一种存款），记录每次通话前后的客户信息和最终有没有认购。Playground 版本用生成模型做了脱敏和扩增，字段含义保持不变。材料的版权说明写得很细：比赛文件是 Apache 2.0，上游 UCI 数据集是 CC BY 4.0，数据集的原始作者是 S. Moro、P. Rita、P. Cortez，比赛的组织者是 Walter Reade 和 Elizabeth Park。

这个题目最值得讲的地方不是模型，而是「**字段什么时候才能拿到**」。营销记录里有些字段是通话之后才产生的，最典型的是 `duration`（这通电话打了多久）。用 `duration` 预测「有没有认购」，分数会很好看，因为聊得久的人本来就更容易买；但真实场景里，我们要在**拨电话之前**就决定打给谁，那时候根本不知道这通电话会打多久。所以作者把任务定义成：只用通话前就能拿到的信息建模，再把 `duration` 单独拿掉做一次对照实验，看分数差多少。

材料里还有一句很重要的提醒：**「和认购相关」不等于「多打一次电话能带来认购」**。模型学到的是相关性，不是因果效应；拿它去回答「该不该多打一通电话」，需要另外的实验设计。作者反复强调，字段可用性的消融实验只是「诊断」，不是「可以上线」的许可证。

### 1.2 比赛目标

输入是一张营销记录表，每一行是一位客户以及和这次营销活动相关的信息；输出是 `y`，0/1 表示客户最终有没有认购定期存款。提交样例只有两列：`id` 和 `y`，也就是说我们要给每一条测试数据填一个预测值（概率）。

材料没有逐个列出字段名，notebook 用通用方式读取所有列：除 `id` 和 `y` 之外的列全部当特征。数据规模也没有固定写出，作者用分块读取加蓄水池抽样，把训练集上限控制在 **120,000 行**——这是一个内存预算，不代表原始文件只有这么多行。作者还声明：额外提供的训练文件保留下来但没有用于基线。

### 1.3 评价标准

notebook 的评分函数对二分类会同时输出 accuracy（准确率）、balanced accuracy（平衡准确率，正负类各自召回率的平均）、log loss（对数损失）、AUC 和 MCC（马修斯相关系数）。Playground 的二分类比赛按 **ROC-AUC** 排名，AUC 衡量的是「把正样本排在负样本前面的能力」，0.5 等于瞎猜，越接近 1 越好。作者明确说：**没有用排行榜标签调参**，模型参数在看分数之前就固定了。

## 二、解题过程

整场比赛的思路可以概括成：**先划定「打电话之前拿得到」的字段，用一个固定参数的 CatBoost 做基线，再和「常数预测」以及「去掉 duration」两个对照比较。** 我们按 EDA、清洗、训练、测试四步走。

### 2.1 数据探索（EDA）

第一步是把配置、数据和审计信息固定下来。我们不整表读入，而是分块读、按固定随机种子做均匀抽样：

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

SEED = 20261026                 # 固定种子：换机器、换时间跑，抽样结果也一样
MAX_ROWS = 120000               # 内存上限，不是原始数据量的声明
TARGET = 'y'
DATA = Path('/kaggle/input/before-the-customer-call')   # 数据所在目录

# 训练集分块读取，先累计缺失值，再做「蓄水池」式的均匀抽样
missing = None
rows_seen = 0
reservoir = None
rng = np.random.default_rng(SEED)

for chunk in pd.read_csv(DATA / 'train.csv', chunksize=100000):
    rows_seen += len(chunk)                       # 记录真实读到了多少行
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)

    # 给每行一个随机优先级，只保留优先级最小的 12 万行
    chunk['_priority'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_priority')

train = reservoir.drop(columns='_priority').reset_index(drop=True)

# 两条硬性检查：id 不能重复，标签不能有空——缺标签的样本宁可不要，也不能瞎补
assert train['id'].is_unique
assert not train[TARGET].isna().any()
```

为什么用「随机优先级取最小的 N 行」，而不是直接取前 N 行？因为数据文件很可能按某种顺序排列（比如按时间），取前 N 行会引入偏差；给每行一个随机数再取最小的，等价于均匀抽样，又不需要把整表装进内存。

接着把缺失情况落盘，并做审计记录：

```python
# 每个字段缺失多少行，写成一个文件，方便回头解释「为什么这么补」
pd.DataFrame({'field': missing.index,
              'missing_rows': missing.values.astype(int)}
             ).to_csv('missingness.csv', index=False)

feature_columns = [c for c in train.columns if c not in [TARGET, 'id']]

# 审计清单：样本来源多少行、实际用了多少行、用了哪些额外文件
audit = {
    'source_rows': rows_seen,
    'sampled_rows': len(train),
    'sampling': 'seeded uniform reservoir',
    'features': feature_columns,
    'source_files_unchanged': True,
}
print(json.dumps(audit, indent=2))
```

这段看起来「不建模」，其实是比赛里最值钱的部分：**先证明自己用的是哪些行、哪些列**。后面所有分数都挂在这份审计记录上，别人才知道结果是怎么来的。

### 2.2 数据清洗

清洗的核心原则是「**无状态**」：所有变换只依赖当前这一行，不依赖训练集的统计量，也不依赖验证集。这样验证集的分数才干净。

```python
def features(frame, omit=()):
    """把原始表变成模型能吃的表；omit 用于消融实验，临时去掉某些列"""
    x = frame[[c for c in feature_columns if c not in omit]].copy()

    for c in list(x.columns):
        if not pd.api.types.is_numeric_dtype(x[c]):
            # 类别列统一转成字符串，缺失值单独占一个类别
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列：转不动就变 NaN；无穷大也当缺失，否则 CatBoost 会报错
            x[c] = pd.to_numeric(x[c], errors='coerce')
            x[c] = x[c].replace([np.inf, -np.inf], np.nan)
    return x
```

我们没有做「用中位数/众数把缺失补上」这一步。原因有两个：一是 CatBoost 自己能处理数值缺失，会把缺失当成一个特殊分支；二是如果用全表统计量补缺失，等于把验证集的信息悄悄带进了训练。类别列的缺失则显式变成 `'__MISSING__'` 这个字符串，让模型自己学「缺失」有没有预测力。

标签方面也做了一次统一编码：

```python
# 把标签映射成 0/1，并打印出来确认顺序——正类是哪一个，直接影响提交的概率含义
classes = sorted(train[TARGET].unique().tolist())
mapping = {v: i for i, v in enumerate(classes)}
y = train[TARGET].map(mapping).to_numpy()
assert len(classes) == 2, '这是二分类比赛'
print('类别编码:', mapping)
```

这一步看似简单，但「正类到底是 0 还是 1」搞反了，提交的 AUC 会直接掉到 0.5 以下，是新手最常踩的坑之一。

### 2.3 模型训练与超参数调优

作者刻意把实验设计成「先定参数，再看分数」。模型是 CatBoost，参数固定为 350 轮、深度 6、学习率 0.08，不做任何基于验证集的调参：

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

X = features(train)
idx = np.arange(len(train))

# 分层切出 20% 做留出集：正负比例和整体一致，分数才不会因抽样波动而误导
fit_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)

def model_fit(x, target):
    cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()
    params = dict(
        iterations=350,          # 固定轮数，不用验证集早停，避免「偷看」验证集
        depth=6,                 # 树深，控制模型复杂度
        learning_rate=0.08,
        random_seed=20261026,
        verbose=False,
        thread_count=4,
        allow_writing_files=False,
    )
    model = CatBoostClassifier(loss_function='Logloss', **params)
    # cat_features 告诉 CatBoost 哪些列是类别列，它内部会做目标统计编码
    model.fit(x, target, cat_features=cat_cols)
    return model

model = model_fit(X.iloc[fit_idx], y[fit_idx])
pred = model.predict_proba(X.iloc[val_idx])[:, 1]   # 留出集上的预测概率
```

为什么不做调参？因为这是一个**基线研究**：作者想在「参数不动」的前提下，观察「字段可用性」带来的分数变化。如果一边调参一边看验证集，最后就分不清分数提升是模型变强了，还是字段泄漏了。

为了判断模型到底有没有学到东西，我们还准备了一个「常数预测」基线：永远输出训练集的正类比例。任何模型如果连这个都赢不了，说明它什么也没学到。

```python
def constant_prediction(target, n):
    """最笨的基线：所有人都给同一个概率（训练集的正类比例）"""
    rate = np.bincount(target, minlength=2)[1] / len(target)
    return np.full(n, rate)

baseline = constant_prediction(y[fit_idx], len(val_idx))
# 常数基线在 AUC 上恒为 0.5，用它当「零分线」最直观
print('baseline prob =', baseline[0])
```

接着做本场比赛最有价值的对照实验：把 `duration` 去掉，在**同一批**留出样本上重新训练、重新打分。

```python
# 先声明要消融的字段，再看分数——顺序不能反，否则就是「看着答案找解释」
omit = ['duration']
reduced = features(train, omit)

alt_model = model_fit(reduced.iloc[fit_idx], y[fit_idx])
alt_pred = alt_model.predict_proba(reduced.iloc[val_idx])[:, 1]

print('全特征 AUC :', roc_auc_score(y[val_idx], pred))
print('去掉 duration AUC:', roc_auc_score(y[val_idx], alt_pred))
```

如果两个分数差得很多，就说明 `duration` 里藏着大量「通话后信息」，全特征模型的高分是不可部署的；如果差得不多，说明模型主要靠通话前字段在工作，结论更可信。作者的态度很明确：**消融结果只能用来判断字段是否可用，不能反过来当成「上线也没关系」的理由**。

### 2.4 模型测试与指标报告

评估阶段我们把「常数基线」和「CatBoost」放在一起报告，这样分数才有参照物：

```python
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, matthews_corrcoef

def score(truth, prob):
    """一次算齐几个常用二分类指标，避免只看一个数下结论"""
    label = (prob >= 0.5).astype(int)          # 概率超过 0.5 判为正类
    return {
        'AUC': float(roc_auc_score(truth, prob)),         # 排序能力，主指标
        'log_loss': float(log_loss(truth, prob)),         # 概率校准，越低越好
        'accuracy': float(accuracy_score(truth, label)),
        'MCC': float(matthews_corrcoef(truth, label)),    # 不平衡数据下比准确率可靠
    }

results = {
    'constant': score(y[val_idx], baseline),
    'boosting': score(y[val_idx], pred),
}
print(json.dumps(results, indent=2))
```

材料里没有保存这次运行的输出，所以具体数值需要在本地跑出来；这里我们只强调**怎么读**：CatBoost 相对常数基线的提升，代表模型真的抓到了信号；`duration` 消融带来的下降，代表那个字段的「含金量」；log loss 和 MCC 则提醒我们，AUC 高不代表概率准、也不代表在不平衡数据上的分类表现好。

最后是测试集推理。测试集可能有上百万行，整表读入会撑爆内存，所以分块预测，并且每一块都核对 `id` 顺序：

```python
sample = pd.read_csv(DATA / 'sample_submission.csv')
submission = sample.copy()
offset = 0

# 用全量训练数据重训一次固定配方，作为最终提交模型
final = model_fit(X, y)

for chunk in pd.read_csv(DATA / 'test.csv', chunksize=100000):
    # 列顺序必须和训练时一致，否则模型会拿错特征
    assert chunk.columns.tolist() == [c for c in train.columns if c != TARGET]

    n = len(chunk)
    # 提交文件的行顺序必须和测试文件完全一致，错一行就全错
    assert np.array_equal(chunk['id'].to_numpy(), sample.iloc[offset:offset + n, 0].to_numpy())

    p = final.predict_proba(features(chunk))[:, 1]
    submission.loc[submission.index[offset:offset + n], TARGET] = p
    offset += n

# 最终校验：行数对得上、id 不重复、没有空值、概率在 0~1 之间
assert offset == len(sample)
assert submission['id'].is_unique
assert submission[TARGET].between(0, 1).all()
submission.to_csv('submission.csv', index=False)
```

分块推理还有一个好处：每一块都能单独检查 `id` 是否和提交样例对得上。如果中间某一块错位，断言会立刻炸掉，而不是等到提交之后才发现整列预测都挪了一行。

## 三、总结

本文中我们完成了 Before the Customer Call 这场营销认购预测比赛的完整流程。我们首先用分块读取和带种子的均匀抽样把训练集控制在 12 万行，并记录了真实行数、缺失情况和特征清单；其次采用无状态的特征变换，类别缺失单独编码、数值缺失交给 CatBoost 处理，避免验证集信息泄漏；接着固定 CatBoost 的 350 轮、深度 6、学习率 0.08，用 20% 分层留出集评估，并和「常数预测」基线对比；然后做了去掉 `duration` 的消融实验，用同一批留出样本判断这个通话后字段的影响；最后分块预测测试集，逐块核对 id 顺序，并检查提交文件的行数与概率范围。

这个比赛给我们最大的启发是：**先问「这个字段在预测时刻拿得到吗」，再问「模型准不准」**。同样的数据，加上 `duration` 分数会漂亮很多，但那样的模型在拨电话之前根本用不了；而「相关不等于因果」这句话，决定了模型输出只能用来排序候选客户，不能直接当作「多打一通电话就能提高认购率」的证据。
