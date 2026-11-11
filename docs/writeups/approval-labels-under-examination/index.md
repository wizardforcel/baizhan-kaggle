# 审批标签再审视（Approval Labels Under Examination）

## 一、比赛简介

### 1.1 比赛背景

贷款审批是银行最常见的自动化决策之一：申请人填一张表，系统判断"批"还是"不批"。这个比赛的数据来自 Kaggle Playground 系列第 S4E10 期（Loan Approval Prediction），原始数据由 Akshay Kumar 提供，比赛组织者把它整理成合成数据，供大家练习二分类建模。

这份 notebook 的作者 Clare Laurent 给比赛加了一层新的解读，标题叫"审批标签再审视"。她的核心观点是：**一个模型完全可以学会复现历史上的审批流程，却完全没有衡量申请人的还款风险**。换句话说，模型学到的是"过去谁被批准了"，而不是"谁将来能还上钱"。这两件事听起来很像，实际上不是一回事。

因此这份材料要求我们交付的不只是一个提交文件，还包括一张"模型卡"（model card，一份说明模型做了什么、没做什么、适用于哪里的说明书），并且必须把"审批倾向"（approval propensity）和"还款能力"（creditworthiness）区分开。材料里反复强调一句话：审批不等于还款，这个练习不能用来批准或拒绝真实的贷款。

### 1.2 比赛目标

输入是一张贷款申请表，每一行是一位申请人，输出是 `loan_status`，表示这笔贷款在历史上是否被批准（0 = 拒绝，1 = 批准）。

训练集 `train.csv` 的列包括：

- `id`：样本编号，唯一；
- 申请人信息：年龄、性别、受教育程度、年收入、工作年限、住房情况等；
- 贷款信息：贷款金额、用途、贷款等级 `loan_grade`、利率 `loan_int_rate`、贷款占收入的比例等；
- 信用信息：历史违约标记、信用历史长度等；
- `loan_status`：标签，也是我们要预测的目标。

测试集只有特征列，没有 `loan_status`；`sample_submission.csv` 的第一列是 `id`，第二列是 `loan_status`，我们要按这个格式填好提交。

### 1.3 评价标准

这是一个正负样本不平衡的二分类问题，排行榜按 ROC AUC 排序。AUC 衡量的是"随机抽一个正样本和一个负样本，模型给正样本打的分更高的概率"，它只看排序、不看阈值，0.5 等于瞎猜，1.0 是完美。

不过这份材料没有只盯一个指标。notebook 里同时报告了五项：准确率（accuracy）、平衡准确率（balanced accuracy）、对数损失（log loss）、AUC 和马修斯相关系数（MCC）。原因很直白：如果绝大多数申请都被批准，一个"全部猜批准"的模型准确率也会很高，看起来很美，但它什么也没学到。材料里专门提醒，报告准确率时一定要同时给出类别数量和常数基线的分数。

---

## 二、解题过程

我们的总思路是：先用一个常数基线把"不作为"的分数钉住，再用固定超参数的 CatBoost 做真正的模型，然后做一个"删特征"的对照实验，最后按提交格式做分块推理并逐项校验。整个过程不使用排行榜分数调参。

### 2.1 数据探索（EDA）

我们首先确认"提交合同"：读 `sample_submission.csv` 的前三行，看清第一列是 `id`、第二列是 `loan_status`。然后分块读取 `train.csv`，用一个固定种子的均匀抽样把训练行数压到 12 万行以内，同时顺手统计每一列的缺失数量：

```python
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20261026
DATA = Path('/kaggle/input/approval-labels-under-examination-data')

# 先读提交样例的前三行，确认“提交合同”：第一列 id，第二列必须是标签
schema = pd.read_csv(DATA / 'sample_submission.csv', nrows=3).columns.tolist()
id_col, targets = schema[0], schema[1:]
assert targets == ['loan_status']

# train.csv 很大，分块读，避免一次性把整张表塞进内存
MAX_ROWS = 120000
reservoir, rows_seen, missing = None, 0, None
rng = np.random.default_rng(SEED)
for chunk in pd.read_csv(DATA / 'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
    # 每行抽一个随机数，只保留随机数最小的 12 万行，等价于均匀抽样
    chunk['_p'] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, '_p')
train = reservoir.drop(columns='_p').reset_index(drop=True)

# 标签不能缺、id 不能重：这是后面所有实验的前提
assert train[id_col].is_unique
assert not train[targets].isna().any().any()
feature_columns = [c for c in train if c not in targets + [id_col]]
print('source rows:', rows_seen, '| sampled:', len(train))
print(train['loan_status'].value_counts(normalize=True))
```

这里要说明两点。第一，抽样是为了省内存，**不是**声称我们用上了全部训练数据，所以要把"源文件行数"和"实际拟合行数"都记录下来。第二，缺失统计单独存成一张表，而不是急着填补——先看清楚哪些列缺、缺多少，再决定怎么处理。

打印标签比例可以看到，被批准的样本明显多于被拒绝的样本。这解释了为什么后面不能只看准确率：多数类本身就占了很大一块。

### 2.2 数据清洗

这份数据没有日期列，因此不需要拆年、月、星期。我们做的清洗只有一件事：**无状态的特征变换**。所谓无状态，就是每一行只根据自己这一行的值做转换，不去学习全表的均值、众数或者目标编码。原因是材料里的一条硬规则：验证集的标签不能用于预处理、早停或选参。如果在全表上算统计量再填补，验证集的信息就悄悄漏进训练里了。

具体做法是：类别列统一转成字符串，缺失单独记成 `__MISSING__`；数值列强制转成数字，把正负无穷替换成缺失，剩下的交给 CatBoost 自己处理。

```python
# feature_columns 是去掉 id 和标签之后的全部特征名
def features(frame, omit=()):
    """无状态特征变换：只依赖当前这一行，不学习任何全表统计量。"""
    cols = [c for c in feature_columns if c not in omit]
    x = frame[cols].copy()
    for c in list(x.columns):
        if not pd.api.types.is_numeric_dtype(x[c]):
            # 类别列：缺失本身作为一个取值保留，不用众数填
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            # 数值列：把无穷统一成缺失，让模型自己决定怎么分叉
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return x
```

我们特意**没有**做目标编码、没有按全表均值填补、也没有删除"看起来异常"的行。这份材料的态度是：清洗的每一步都要能说清楚为什么，而不是把一套标准流程无脑套上去。

### 2.3 模型训练与超参数调优

我们其次搭起对照实验。基线是一个"常数模型"：它不看任何特征，直接输出训练集里的类别频率。这个模型代表"完全没学"的水平，是判断后面模型有没有真本事的标尺。

主模型用 CatBoost，超参数在看分数之前就写死：350 轮迭代、深度 6、学习率 0.08、4 个线程。数据按 80/20 切分，用分层抽样保证两边的正负比例一致。

```python
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

X = features(train)
classes = sorted(train['loan_status'].unique())          # [0, 1]
y = train['loan_status'].map({c: i for i, c in enumerate(classes)}).to_numpy()

# 固定 20% 留出集；stratify 保证两边正负比例一致
idx = np.arange(len(train))
fit_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
assert set(fit_idx).isdisjoint(val_idx)

def model_fit(x, target):
    cat_cols = x.select_dtypes(include=['object', 'string']).columns.tolist()
    # 超参数在看验证分数之前就定好，避免“用验证集调参”的隐性泄漏
    params = dict(iterations=350, depth=6, learning_rate=0.08,
                  random_seed=SEED, verbose=False, thread_count=4,
                  allow_writing_files=False)
    model = CatBoostClassifier(loss_function='Logloss', **params)
    model.fit(x, target, cat_features=cat_cols)   # 类别列交给 CatBoost 自己编码
    return model

model = model_fit(X.iloc[fit_idx], y[fit_idx])
```

模型本身没有任何花哨的地方，这份材料的重点也不在刷分，而在"可证伪的实验设计"。所以我们紧接着做了第二个实验：把 `loan_grade`（贷款等级）和 `loan_int_rate`（利率）这两个字段删掉，用完全相同的切分、完全相同的超参数再训一次。

为什么偏偏删这两个？因为在真实的审批流程里，等级和利率往往是在审批过程中甚至审批之后才定下来的，它们离"决定"太近，很可能直接编码了审批结果。删掉它们再看分数，就能判断模型是不是在"抄答案"，而不是真的从申请人的收入和负债里学到东西。材料把这种实验叫"可得性消融"（availability ablation），并强调它只是诊断，不代表这些字段真的能在部署时拿到。

### 2.4 模型测试与指标报告

我们接着在留出集上同时算五项指标，并把常数基线、完整模型、删特征模型放在一张表里对比：

```python
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             log_loss, roc_auc_score, matthews_corrcoef)

def score(truth, p):
    labels = p.argmax(axis=1)
    return {
        'accuracy': accuracy_score(truth, labels),
        'balanced_accuracy': balanced_accuracy_score(truth, labels),
        'log_loss': log_loss(truth, p, labels=np.arange(len(classes))),
        'AUC': roc_auc_score(truth, p[:, 1]),
        'MCC': matthews_corrcoef(truth, labels),
    }

# 常数基线：直接输出训练集的类别频率，代表“完全没学”的下限
freq = np.bincount(y[fit_idx], minlength=2) / len(fit_idx)
baseline = np.tile(freq, (len(val_idx), 1))
pred = model.predict_proba(X.iloc[val_idx])
results = {'constant': score(y[val_idx], baseline), 'boosting': score(y[val_idx], pred)}

# 第二个实验：去掉两个和审批决定最接近的字段，切分与超参数完全不变
omit = ['loan_grade', 'loan_int_rate']
reduced = features(train, omit)
alt = model_fit(reduced.iloc[fit_idx], y[fit_idx])
results['ablation'] = score(y[val_idx], alt.predict_proba(reduced.iloc[val_idx]))
print(pd.DataFrame(results).T)
```

读这张表的方法是：先看常数基线的平衡准确率（大约就是 0.5）和 AUC（0.5），再看完整模型比它高多少。如果删掉两个字段之后 AUC 掉得很多，说明模型的"本事"有相当一部分来自这两个和决定直接相关的字段，而不是来自申请人的真实偿付能力；如果掉得不多，说明收入、负债比、信用历史这些信息本身就够用。无论结果如何，材料要求我们**保留全部诊断结果**，不能因为另一个实验更好看就把它藏起来。

最后是提交。提交的模型仍然是那个"全特征 + 固定超参数"的版本，删特征实验只作为证据：

```python
# 提交模型固定为全特征基线；消融结果只写进报告，不用来改提交
final = model_fit(X, y)
sample = pd.read_csv(DATA / 'sample_submission.csv')
submission = sample.copy()
submission['loan_status'] = submission['loan_status'].astype(float)

offset = 0
for chunk in pd.read_csv(DATA / 'test.csv', chunksize=100000):
    n = len(chunk)
    # 逐块核对 id 顺序，防止行错位这种不会报错的“静默错误”
    assert np.array_equal(chunk[id_col].to_numpy(),
                          sample.iloc[offset:offset + n, 0].to_numpy())
    p = final.predict_proba(features(chunk))[:, 1]      # 提交正类概率
    submission.loc[submission.index[offset:offset + n], 'loan_status'] = p
    offset += n

assert offset == len(sample)
assert not submission['loan_status'].isna().any()
submission.to_csv('submission.csv', index=False)

# 再读回来核对一次列名和行数，格式错误在比赛里非常常见
roundtrip = pd.read_csv('submission.csv')
assert roundtrip.columns.tolist() == schema and len(roundtrip) == len(sample)
```

最后我们把这次实验的"身份信息"写进一份清单。有了它，别人拿到结果时能立刻知道样本量、模型配方和验证方式，而不是只看到一个孤零零的提交文件：

```python
import json, platform, sklearn, catboost

manifest = {
    'seed': SEED,
    'source_rows': rows_seen,
    'sampled_rows': len(train),          # 实际参与拟合的行数，不是源文件行数
    'submitted_model': '固定全特征 CatBoost，350 轮，深度 6',
    'validation': '单个固定 20% 留出集，单标签分类使用分层抽样',
    'ablation_fields': omit,             # 只作诊断，不用于改提交
    'extra_training_files_used': False,
    'versions': {'python': platform.python_version(),
                 'pandas': pd.__version__,
                 'sklearn': sklearn.__version__,
                 'catboost': catboost.__version__},
}
Path('run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
```

这份清单就是"模型卡"的骨架：它明确写出了我们用了多少数据、用了什么模型、在什么条件下验证过。材料反复强调，一次留出集只是学习基线，**不是部署保证**；真正要用于有后果的决策，还必须说明分组方式、时间跨度、不确定性以及独立评估的安排。

---

## 三、总结

本文中我们完成了一次"把标签含义放在模型之前"的贷款审批建模练习。我们首先确认提交格式，分块读取数据并做固定种子的 12 万行均匀抽样，同时记录缺失情况；其次用无状态变换处理类别列和数值列，坚决不学全表统计量、不做目标编码；接着用常数基线和固定超参数的 CatBoost 做对比，并额外做了删除 `loan_grade`、`loan_int_rate` 的可得性消融；然后在留出集上同时报告准确率、平衡准确率、对数损失、AUC 和 MCC，最后用全特征模型分块推理、逐项校验提交文件。

这个比赛给我们最大的启发不是某个模型技巧，而是**先问标签是什么意思，再问模型怎么选**。`loan_status` 记录的是历史上的审批决定，不是还款事实；在审批数据上训练出来的模型，最多只能复现过去的审批倾向，不能证明谁真的会还钱。材料里那句"审批不是还款"值得我们一直记着：一份只在一个留出集上验过的基线，只是学习的起点，真正要用于有后果的决策，还需要分组、时间、不确定性和独立评估。
