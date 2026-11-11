# BoolArt 用户违约预测（BoolArt User Default Prediction）

## 一、比赛简介

### 1.1 比赛背景

这场比赛的任务是**信用卡用户违约预测**：银行手里有大量客户过去若干个月的信用卡账单记录，希望提前判断哪些客户未来会违约（default，也就是还不上钱）。比赛的原始数据来自 Kaggle 上著名的信用卡违约预测赛（材料里的代码注释也直接给出了 `amex-default-prediction` 的讨论帖链接）。

这类数据有两个鲜明特点：

1. **一个客户有多行**。每个 `customer_ID` 对应 1～13 个月的月度账单，每个月一行；而标签只有一个（这个客户最终是否违约）。所以建模的第一步不是「训练模型」，而是**把多行账单汇总成一个客户级别的向量**。
2. **特征全部匿名且加了噪音**。列名只告诉你类别前缀，不告诉你具体业务含义；数值特征还被人为加入了 `uniform(0.00, 0.01)` 的随机噪音，防止有人反推原始金额。

### 1.2 比赛目标

数据分三个文件：

- `train.parquet`：训练集账单明细，一行是「某个客户某个月」的记录；
- `train_labels.csv`：每个 `customer_ID` 对应的 `target`（0 = 未违约，1 = 违约）；
- `test.parquet`：测试集账单明细，结构与训练集相同，但没有标签。

特征按前缀分成几大组（材料里的注释写得很清楚）：

| 前缀 | 含义 |
| --- | --- |
| `D_*` | Delinquency，逾期相关 |
| `S_*` | Spend，消费相关 |
| `P_*` | Payment，还款相关 |
| `B_*` | Balance，余额相关 |
| `R_*` | Risk，风险相关 |

其中数值特征 177 个，类别特征 11 个（`B_30`、`B_38`、`D_114`、`D_116`、`D_117`、`D_120`、`D_126`、`D_63`、`D_64`、`D_66`、`D_68`），日期列是 `S_2`。我们要给每个测试客户输出一个**违约概率**。

### 1.3 评价标准

这是一个二分类问题：提交每个客户属于「违约」的概率，模型排序越准越好。材料里的 XGBoost 用 `logloss`（对数损失，越小越好）和 `auc`（曲线下面积，越大越好）两个指标监控训练，并按最后一个指标做早停。因此我们同样以 **AUC 为主、logloss 为辅**来选择模型和轮数。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们先把三个文件读进来，并把 `S_2` 从字符串转成日期：

```python
import pandas as pd
import numpy as np
import cudf, cupy

train = pd.read_parquet("../input/boolart-user-default-prediction/train.parquet")
test  = pd.read_parquet("../input/boolart-user-default-prediction/test.parquet")
targets = pd.read_csv("../input/boolart-user-default-prediction/train_labels.csv")

train['S_2'] = pd.to_datetime(train['S_2'])
test['S_2']  = pd.to_datetime(test['S_2'])
```

**先看标签和客户生命周期。** 我们把每个客户的首次账单日期（近似理解为「开户/注册时间」）合到标签表上，再看不同注册月份的违约率和客户数量：

```python
customer_first_date = train.groupby('customer_ID')['S_2'].min()
targets = targets.merge(customer_first_date.reset_index(), on='customer_ID', how='left')

# 注册月份 vs 违约率：越晚进来的客户，违约率是否不同？
targets.groupby('S_2')['target'].mean().sort_index().plot()
# 注册月份 vs 客户数量：各月的样本量是否均衡？
targets.groupby('S_2')['target'].count().sort_index().plot()
```

这一步很关键：如果训练集和测试集的客户「入网时间」分布不同，模型学到的规律就可能失效。

**再看类别特征。** 对 11 个类别列，我们都画了两张图：左边的柱状图是取值分布，右边是该取值下的平均违约率：

```python
def plot_category(col):
    f, axes = plt.subplots(1, 2, figsize=(16, 2))
    value_counts = train[col].value_counts()
    value_counts.plot(kind='bar', xlabel=f'{col} value counts', ax=axes[0])
    # 每个类别的平均 target，能直接看出「哪类客户更危险」
    train.groupby(col)['target'].mean().loc[value_counts.index] \
         .plot(kind='bar', xlabel=f'{col} target mean', ax=axes[1])

for col in ['B_30', 'B_38', 'D_114', 'D_116', 'D_117',
            'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']:
    plot_category(col)
```

**类别 × 类别交叉。** 单个类别的区分力有限，我们又两两组合画热力图（格子里是该组合的平均违约率），发现了一些很有价值的组合。例如 `D_68` 与 `D_63` 交叉出的某个格子违约率只有 0.029/0.038，基本可以断定「这类客户不可能违约」：

```python
def plot_category_cross_heatmap(col1, col2):
    # 交叉组合的平均违约率，异常低的格子代表"几乎不会违约"的人群
    cross_mean = train.groupby([col1, col2])['target'].mean()
    sns.heatmap(cross_mean.reset_index().pivot(index=col1, columns=col2, values='target'),
                annot=True)

plot_category_cross_heatmap('D_68', 'D_63')
```

当然，交叉格子还要看样本量——频次太低的格子，均值不可信，这是我们在读热力图时反复提醒自己的一点。

**训练集/测试集分布对比。** 我们对每个类别列分别画 train 与 test 的取值分布，发现总体差异不大，但 `D_120` 的分布差异比较明显，而且它在时间上有缓慢上升的趋势；`D_66` 在测试集里几乎不出现 0 类别（如果不考虑缺失值，它接近无效特征）。这类结论会直接影响我们要不要保留某个特征。

### 2.2 数据清洗

清洗分三块。

第一，**把 `customer_ID` 从 16 进制字符串压成整数**，把 `S_2` 从日期压成「年月序号」。后者用一个基准（`12 * 年 + 月 - 24207`）换算，训练集落在 0～12，测试集自然落在 13 以后，顺便还能当「客户处在生命周期第几个月」用：

```python
def process_customer_columns(df):
    # 只取 ID 的后 16 位十六进制转成 int64，节省内存
    df['customer_ID'] = df['customer_ID'].str[-16:].apply(int, base=16).astype('int64')
    year  = pd.to_numeric(df['S_2'].str[:4])
    month = pd.to_numeric(df['S_2'].str[5:7])
    # 折算成连续的"月份编号"，训练集为 0~12，测试集顺延
    df['S_2'] = year.mul(12).add(month).sub(24207).astype('int8')
    return df
```

第二，**丢掉整列无信息或重复的特征**。材料里直接删掉了 `B_29`，因为它的取值和其它列高度重复；EDA 里也发现 `D_66` 在测试集里几乎没有非缺失值。

第三，**缺失值处理**。数值特征本身加了 `uniform(0, 0.01)` 的噪音，很多「看起来不同」的小数其实是噪音。我们对聚合后的结果统一填一个哨兵值 `-127`（能塞进 int8，又远离正常取值范围），让 XGBoost 自己决定怎么切；只有少数几列（`D_50`、`S_23` 的 mean/std/last）可能低于 -127，单独填 `-32783`：

```python
NAN_VALUE = -127          # 能放进 int8 的哨兵值

nan_col = ['D_50_mean', 'D_50_std', 'D_50_last',
           'S_23_mean', 'S_23_std', 'S_23_last']
df[nan_col] = df[nan_col].fillna(-32783)   # 这几列正常值可能低于 -127
df = df.fillna(NAN_VALUE)
```

### 2.3 模型训练与超参数调优

**第一步：把多行账单压成一行客户特征。** 这是整场比赛的核心工作量。我们对每个客户做以下几类聚合：

- 数值特征：`mean`（均值）、`std`（波动）、`last`（最近一个月）；
- 类别特征：`last`（最近一个月的状态）；
- 派生特征：`curLevel`（最后值在 min～max 中的相对位置）、`magnitude`（max - min，波动幅度）、`last-mean`（最后值偏离均值多少）；
- 类别派生：`match`（最后两个月是否相同）；
- 日期派生：`S_2` 的最小值、最大值、出现次数（判断是不是短期客户）；
- 缺失统计：整段账单里的缺失总数、最后一行缺失数。

```python
groups = df.groupby("customer_ID")

# 1) 数值特征的均值/标准差/最后值
num_agg = groups[num_features].agg(['mean', 'std', 'last'])
num_agg.columns = ['_'.join(x) for x in num_agg.columns]

# 2) 最后值在历史区间中的相对位置：0=历史最低，1=历史最高
cur_level = (groups[num_features].last() - groups[num_features].min()) / \
            (groups[num_features].max() - groups[num_features].min())
cur_level = cur_level.fillna(0)
cur_level.columns = [x + '_curLevel' for x in cur_level.columns]

# 3) 波动幅度：历史最大值 - 历史最小值
magnitude = (groups[num_features].max() - groups[num_features].min()).fillna(0)
magnitude.columns = [x + '_magnitude' for x in magnitude.columns]

# 4) 最后值 - 均值：最近一个月比平时高还是低
last_mean = (groups[num_features].last() - groups[num_features].mean()).fillna(0)
last_mean.columns = [x + '_last-mean' for x in last_mean.columns]

# 5) 类别特征：最后一个月是否与上个月相同（状态是否稳定）
cat_match = 1 + groups[cat_features].nth(-1) - groups[cat_features].nth(-2)
cat_match = cat_match.fillna(1)
cat_match[cat_match != 1] = 0
cat_match.columns = [x + '_match' for x in cat_match.columns]
```

**第二步：加「趋势」特征 HMA。** 普通均值会把最近的变化抹平。我们用了 Hull Moving Average（HMA，一种加权移动平均，越近的月份权重越大、还会减去更早的均值项来放大趋势），分别用最近 1、3、5、7、9、11、13 个月计算，再把结果拼起来。这样模型不仅知道「现在多少」，还知道「在往哪个方向走」。类似地，`S_2` 用「最小值 +12 - 最大值」做归一化，就能区分「新客户」和「中间有断档的老客户」。

```python
def hma3(groups, columns):
    # 用最近 3 个月算加权平均，越近权重越大；负号项用来抵消趋势滞后
    return (5/6) * groups[columns].nth(-1) \
         + (1/3) * groups[columns].nth(-2) \
         - (1/6) * groups[columns].nth(-3)

def hma5(groups, columns):
    return (2/3)  * groups[columns].nth(-1) \
         + (2/5)  * groups[columns].nth(-2) \
         + (2/15) * groups[columns].nth(-3) \
         - (2/15) * groups[columns].nth(-4) \
         - (1/15) * groups[columns].nth(-5)
```

**第三步：训练 XGBoost。** 我们用的是作者称为 **pyramid（金字塔）** 的训练策略：不是一次性训练一棵提升树，而是分 6 层——前几层用「很多棵树 × 很小权重」堆出一个多样的森林，后面几层逐层降低森林比例、提高提升树比例，最后一层是纯提升树（权重 0 表示不再混合前一层输出）。这样相当于先「广撒网」再「精修残差」。

```python
xgb_params = {
    'max_depth': 7,
    'subsample': 0.75,
    'colsample_bytree': 0.35,   # 列采样较低，配合小学习率可显著降过拟合
    'gamma': 1.5,
    'lambda': 70,               # L2 正则很强
    'min_child_weight': 8,
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],   # 按最后一个指标早停
    'tree_method': 'gpu_hist',
    'random_state': SEED,
}

# 每一层：(森林树数, 提升树轮数, 学习率调整, 与上一层输出的混合权重)
pyramid_layers = [(100,   10, 1.56,  0.5),
                  ( 20,   50, 1.3,   2/3),
                  (  1, 1000, 1.25,  0.75),
                  (  1, 1000, 1.125, 0.875),
                  (  1, 3000, 1.0,   1),
                  (  1, 9000, 0.5,   0)]
```

训练时做 **5 折交叉验证**：每一折用 4 份训练、1 份验证，把验证集的预测值保存成 OOF（out-of-fold，折外预测），这样每个训练客户都有一个「没见过它的模型」给出的预测，可以用来评估真实效果。中间层之间用 `output_margin=True` 把上一层的原始打分传给下一层当初始预测，这是 XGBoost 官方 demo 里的 `boost_from_prediction` 用法。

```python
from sklearn.model_selection import KFold
import xgboost as xgb

FOLDS = 5
skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))

for fold, (tr_idx, va_idx) in enumerate(skf.split(train)):
    X_tr, y_tr = train.loc[tr_idx, features], train.loc[tr_idx, 'target']
    X_va, y_va = train.loc[va_idx, features], train.loc[va_idx, 'target']

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)

    margin = None
    for trees, rounds, adj_eta, w in pyramid_layers:
        # 每一层都从上一层的输出（margin）继续，而不是从零开始
        if margin is not None:
            dtrain.set_base_margin(margin)
        model = xgb.train({**xgb_params, 'num_parallel_tree': trees},
                          dtrain, rounds, evals=[(dvalid, 'valid')])
        margin = model.predict(dtrain, output_margin=True)

    oof[va_idx] = model.predict(dvalid)
```

### 2.4 模型测试与指标报告

5 折跑完后，我们把 OOF 预测拼起来，画出分布并计算整体 AUC；同时导出特征重要性，看看哪些变量真正被模型用上了（这也能反过来验证我们的特征工程是否合理）。这些中间产物（`oof_xgb_v1.csv`、`xgb_feature_importance_v1.csv`）都存了下来，方便后续融合不同版本的特征。

```python
from sklearn.metrics import roc_auc_score

print("OOF AUC:", roc_auc_score(train['target'], oof))

import matplotlib.pyplot as plt
plt.hist(oof, bins=100)      # 看预测概率的分布是否被压得太偏
plt.show()
```

**预测测试集。** 测试集被切成两段分别做特征工程（因为原始 parquet 很大，一次读不进内存），所以预测也是分两段做，最后按 `customer_ID` 合并。提交时以 `sample_submission.csv` 的 `customer_ID` 顺序为准左连接，保证行序完全一致：

```python
sub = pd.read_csv('../input/boolart-user-default-prediction/sample_submission.csv')[['customer_ID']]
test_pred = pd.concat([pred_part0, pred_part1])          # 两段测试集的预测
test_pred = pd.DataFrame({'customer_ID': customers, 'prediction': test_pred})

# 左连接，保证提交文件的顺序与官方样例一致
sub = sub.merge(test_pred, on='customer_ID', how='left')
sub.to_csv('submission.csv', index=False)
```

由于 XGBoost 的早停基于验证集 AUC，模型在验证集上的表现和最终提交所需的概率尺度是对齐的，不需要额外做概率校准。

---

## 三、总结

本文中我们完成了 BoolArt 用户违约预测的完整流程：首先通过 EDA 摸清了数据的结构——每个客户 1～13 行月度账单、177 个数值特征 + 11 个类别特征、特征带有人为噪音，并重点分析了注册时间与违约率、类别取值与违约率、类别交叉组合（如 `D_68`×`D_63`）以及训练集/测试集的分布差异；其次我们做了针对性的数据清洗：ID 压成整数、`S_2` 压成月份序号、删掉 `B_29`、用哨兵值处理缺失；接着我们把每个客户的多行账单聚合出均值/标准差/最后值/相对位置/波动幅度/最后值减均值/类别稳定性等特征，再加 HMA 趋势特征，最后用 6 层「金字塔」XGBoost 做 5 折交叉验证训练；最后我们用 OOF AUC 评估模型、导出特征重要性，并按官方样例顺序生成提交文件。

这个比赛给我们最大的启发是：**多行变一行（aggregation）是这类「行为序列 → 单标签」任务的主战场**。模型反而是现成的 XGBoost，真正拉开差距的是你从「均值、最后值、波动、趋势」里能压出多少信息，以及能不能识别出像 `D_68`×`D_63` 这种「几乎不会违约」的强规则人群。