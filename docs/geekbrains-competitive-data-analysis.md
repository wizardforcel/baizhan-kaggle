# GeekBrains 竞赛数据分析（GeekBrains Competitive Data Analysis）

## 一、比赛简介

### 1.1 比赛背景

这是一个由俄罗斯在线教育平台 GeekBrains 组织的「教学型」Kaggle 比赛，用来练习**信用风险预测**：根据银行客户申请贷款时提供的资料，预测这个客户未来会不会出现还款问题（违约）。

这类问题在金融风控里非常常见，通常是整个风控系统的入口——银行要在放款前决定「批不批准、批多少、利率多少」。数据是**表格型数据**（不像图片、文本那样需要神经网络），一张表一行，有很多列，这种问题用梯度提升树（GBDT）系列模型往往效果最好。

数据一共有 6 张表，模拟了银行系统里真实的关联数据结构：

| 文件 | 内容 |
| --- | --- |
| `train.csv` | 申请编号 + 目标变量（0 = 正常还款，1 = 有问题的客户） |
| `test.csv` | 申请编号，要预测 `TARGET` 的那些申请 |
| `bki.csv` | 信用局（征信机构）里该客户过往贷款的记录，一条客户可能有很多条 |
| `client_profile.csv` | 客户资料：性别、子女数、收入、申请金额、月供、教育、婚姻、居住地规模、年龄、工作年限、车龄、各种信用评分、各类查询次数等 |
| `payments.csv` | 客户此前贷款的还款记录 |
| `applications_history.csv` | 客户此前的贷款申请历史 |

难点在于**多张表怎么合并**。训练表里一个客户只有一行，但信用局表和还款记录表里同一个客户可能有几十行。如果不会聚合，就只能用到年龄、收入这类表层信息，白白浪费最有预测力的历史还款记录。

### 1.2 比赛目标

输入是一条贷款申请及其关联的客户资料和历史记录，输出该申请出现还款问题（`TARGET = 1`）的**概率**。提交文件包含两列：`APPLICATION_NUMBER` 和 `TARGET`。

### 1.3 评价标准

比赛用 **ROC AUC**（接收者操作特征曲线下的面积）评价，越高越好。

用大白话解释：把模型对所有申请打的分排序，AUC 衡量的是「随机抽一个正常客户和一个问题客户，模型给问题客户的分更高的概率」。完全瞎猜是 0.5，满分是 1.0。AUC 的好处是它只看**排序**，不看分数的绝对值大小，所以不需要像 RMSE 那样处理量纲问题。

---

## 二、解题过程

我们参考了社区里一篇结构清晰的「stacking models」笔记本：**先把客户资料做特征工程，训练多个不同超参数的 LightGBM，用 5 折交叉验证产出 out-of-fold 预测；再把这几组预测当作新特征，喂给一个线性判别分析（LDA）做第二层；然后把信用局数据聚合成客户级特征，重复一遍，最后按「有没有客户资料」把两份结果拼起来提交**。我们按数据探索、清洗、训练调参、测试报告四步展开。

### 2.1 数据探索（EDA）

我们首先读入 6 张表，看看每张表的规模：

```python
import pandas as pd

PATH = '/kaggle/input/geekbrains-competitive-data-analysis/'

raw_train = pd.read_csv(PATH + 'train.csv')
raw_test  = pd.read_csv(PATH + 'test.csv')
raw_cp    = pd.read_csv(PATH + 'client_profile.csv')
bureau    = pd.read_csv(PATH + 'bki.csv')

print(raw_train.shape, raw_test.shape)   # 训练表和测试表：各一行一个申请
print(raw_cp.shape)                      # 客户资料，只有部分申请有
print(bureau.shape)                      # 信用局表：一行一笔历史贷款，远多于客户数
```

从形状可以看出两件关键的事：

1. **目标变量高度不平衡**。我们看一下比例：

```python
print(raw_train['TARGET'].value_counts(normalize=True))
```

正常还款的比例占绝大多数，问题客户只占一小部分。这意味着**不能用准确率评价**——全猜 0 的准确率也能很高，但没有任何价值。这也是比赛选 AUC 的原因之一。

2. **`client_profile` 并非每个申请都有**。我们做了个交集检查：

```python
cp_ids = set(raw_cp['APPLICATION_NUMBER'])
train_with_cp = raw_train['APPLICATION_NUMBER'].isin(cp_ids).mean()
print('训练集中有客户资料的比例:', train_with_cp)
```

这个发现直接影响了提交策略：**测试集中有一部分申请根本没有客户资料**，对它们只能用信用局等历史数据；对另一部分，两边信息都有。所以后面我们干脆训练「是否含客户资料」的两套模型，最后按 id 拼接。

我们还看了目标变量与关键数值列的关系（比如收入、申请金额、年龄），由于数据已经匿名化处理，很多列含义并不直观，我们主要看**缺失率**和**分布形状**，没有对单个列做深入的可解释分析。

### 2.2 数据清洗与特征工程

#### 2.2.1 基础清洗

我们先对客户资料做统一处理：

```python
import numpy as np

def profiling(df):
    # 一个只会出现在"无工作"客户上的占位值 365243，代表"无"，换成缺失
    df['DAYS_ON_LAST_JOB'].replace(365243, np.nan, inplace=True)
    # 月供占申请金额的比例，比原始金额更能反映客户的还款压力
    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMOUNT_CREDIT'] / df['AMOUNT_ANNUITY']
    # 把性别和婚姻状态拼成一个新类别，方便模型一次学到"组合"效应
    df['NEW_STATUS'] = df['GENDER'] + df['FAMILY_STATUS'].astype(str)
    # 分类变量做独热编码，缺失值也单独作为一类
    df, cat_cols = get_encoded(df, nan_as_category=False)
    return df
```

这里处理了一个经典陷阱：**`DAYS_ON_LAST_JOB` 里的 365243 天（约 1000 年）不是真实值**，而是表示「没有工作」。如果把它当普通数值，会让模型以为存在一个极端年长的客户，需要映射成缺失值。

独热编码（one-hot encoding）的作用是：把「性别」这种字符串列拆成「是否是男性」「是否是女性」两列 0/1 值，这样树模型可以直接拿来切分。

#### 2.2.2 派生特征

第二版我们加入了客户画像的比值特征：

```python
def create_client_profile_features(X):
    # 月供相对年龄、贷款相对年龄、收入相对年龄：衡量财务负担与人生阶段的匹配
    X['RATIO_ANNUITY_TO_AGE']    = X['AMOUNT_ANNUITY'] / X['AGE']
    X['RATIO_CREDIT_TO_AGE']     = X['AMOUNT_CREDIT'] / X['AGE']
    X['RATIO_SALARY_TO_AGE']     = X['TOTAL_SALARY'] / X['AGE']
    # 年龄 / 工作年限、车龄 / 工作年限：衡量稳定程度
    X['RATIO_AGE_TO_EXPERIENCE'] = X['AGE'] / X['DAYS_ON_LAST_JOB']
    X['RATIO_CAR_TO_EXPERIENCE'] = X['OWN_CAR_AGE'] / X['DAYS_ON_LAST_JOB']
    X['RATIO_CAR_TO_AGE']        = X['OWN_CAR_AGE'] / X['AGE']
    return X
```

树模型虽然能自己学出「收入高 + 年龄大」这样的组合，但**直接的比值特征能更容易地表达「相对水平」**，比如「同样借 100 万，25 岁的人比 45 岁的人压力大得多」。我们在后面比较两版特征的效果时，会用到这两组。

#### 2.2.3 信用局数据聚合

信用局表是「一个客户多行」，必须聚合成「一个客户一行」，才能和主表合并。我们按客户 id 分组，对逾期相关的金额取平均值，并把「有未结清贷款」和「已结清贷款」分开统计：

```python
def bki(nan_as_category=True):
    bureau = pd.read_csv(PATH + 'bki.csv')
    bureau, bureau_cat = get_encoded(bureau, nan_as_category)
    bureau.drop(['BUREAU_ID'], axis=1, inplace=True)   # 这个 id 对预测无意义

    num_agg = {
        'CREDIT_DAY_OVERDUE': ['mean'],          # 平均逾期天数
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],      # 平均最大逾期金额
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],      # 平均逾期总额
    }
    # 分类变量（如信用状态）也取"平均值"，等于统计各状态的占比
    cat_agg = {cat: ['mean'] for cat in bureau_cat}

    bureau_agg = bureau.groupby(COLUMN_ID).agg({**num_agg, **cat_agg})
    bureau_agg.columns = ['BURO_' + c[0] + '_' + c[1].upper() for c in bureau_agg.columns]

    # 只看"活跃"贷款：当前还欠着钱，最能反映当前还款压力
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby(COLUMN_ID).agg(num_agg)
    active_agg.columns = ['ACTIVE_' + c[0] + '_' + c[1].upper() for c in active_agg.columns]
    bureau_agg = bureau_agg.join(active_agg, how='left', on=COLUMN_ID)

    # 只看"已结清"贷款：反映历史上是否顺利还完
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby(COLUMN_ID).agg(num_agg)
    closed_agg.columns = ['CLOSED_' + c[0] + '_' + c[1].upper() for c in closed_agg.columns]
    bureau_agg = bureau_agg.join(closed_agg, how='left', on=COLUMN_ID)
    return bureau_agg
```

这一步的思想很值得记：**汇总时别只算全部，还要按业务含义分组**。全部贷款的平均逾期天数会被「已经还完的贷款」稀释；单独看「当前未结清贷款」的逾期情况，才真正反映客户眼下的还款压力。

### 2.3 模型训练与超参数调优

#### 2.3.1 交叉验证与 OOF 预测

我们用 5 折交叉验证（KFold）来评估和生成第二层特征。**关键在于 out-of-fold（OOF）预测**：第 k 折验证集上的预测，只由另外 4 折训练出来的模型给出。这样每一行的预测都「没见过自己」，用它来训练第二层模型才不会欺骗自己：

```python
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

def lightgbm_cross_validation(params, X, y, cv):
    estimators, folds_scores = [], []
    oof_preds = np.zeros(X.shape[0])          # 每个样本一份"折外"预测
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(x_train, y_train,
                  eval_set=[(x_valid, y_valid)],
                  eval_metric='auc',
                  early_stopping_rounds=50)   # 50 轮不提升就停
        oof_preds[valid_idx] = model.predict_proba(x_valid)[:, 1]
        score = roc_auc_score(y_valid, oof_preds[valid_idx])
        folds_scores.append(round(score, 5))
        print(f'Fold {fold+1}, Valid score = {score:.5f}')
        estimators.append(model)
    print(f'Mean score: {np.mean(folds_scores):.5f}')
    return estimators, oof_preds
```

测试集预测则取 5 个折模型预测的平均：

```python
prediction = np.zeros(test.shape[0])
for model in estimators:
    prediction += model.predict_proba(test)[:, 1]
prediction /= N_SPLITS
```

#### 2.3.2 多组超参数的 LightGBM

为了让第二层有「多样」的输入，我们训练了三组不同超参数的 LightGBM：一组偏深、正则强，一组偏浅，一组介于两者之间。

```python
lgbm_params = {           # 主力模型：较深、带正则
    'metric': 'auc', 'boosting_type': 'gbdt',
    'n_estimators': 1000, 'learning_rate': 0.03,
    'num_leaves': 42, 'max_depth': 10,
    'reg_alpha': 0.05, 'reg_lambda': 0.06,
    'subsample': 0.8, 'min_child_weight': 60,
    'random_state': 42,
}
lgbm_params1 = {          # 浅一些、学习率更低
    'metric': 'auc', 'boosting_type': 'gbdt',
    'n_estimators': 1000, 'learning_rate': 0.02,
    'max_depth': 8, 'random_state': 42,
}
lgbm_params2 = {          # 最浅：5 层，叶子 32
    'metric': 'auc', 'boosting_type': 'gbdt',
    'n_estimators': 1000, 'learning_rate': 0.05,
    'num_leaves': 32, 'max_depth': 5,
    'min_child_weight': 42, 'random_state': 42,
}
```

为什么不要只用一个超参数组？因为后面要把它们叠加，**不同深度的模型会犯不同的错**，合在一起才有互补性。如果三个模型预测高度相关，叠加就没有意义。

我们还尝试过用 Optuna 自动调参 CatBoost，搜索空间是学习率（0.01～0.1）和深度（这里固定为 3），每次试验训练 1000 棵树，最多 10 次试验或 600 秒：

```python
import optuna, catboost as cb
from sklearn.model_selection import train_test_split

def objective(trial):
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=42)
    param = {
        'n_estimators': 1000, 'eval_metric': 'AUC',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 3, 3),
    }
    gbm = cb.CatBoostClassifier(**param)
    gbm.fit(train_x, train_y, cat_features, eval_set=[(valid_x, valid_y)], verbose=0)
    preds = gbm.predict_proba(valid_x.fillna(-1))[:, 1]
    return roc_auc_score(valid_y, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10, timeout=600)
```

Optuna 是一个自动调参工具，用贝叶斯方法选择下一组参数，比暴力网格搜索省时间。不过它单独跑在客户资料表上，属于「多一种思路」的尝试，主线还是 LightGBM 叠加。

#### 2.3.3 第二层模型：LDA

第一层的输出是几列「预测概率」，我们让线性判别分析（LDA，一种经典的线性分类器）来学习这几列预测的**最优组合权重**：

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

scores = pd.DataFrame({'lgbm': oof_preds_lgbm,
                       'lgbm1': oof_preds_lgbm1,
                       'lgbm2': oof_preds_lgbm2})

lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto',
                                 store_covariance=True, n_components=1)
estimators_lda, oof_preds_lda, mean_score = cross_validation(
    model=lda, X=scores, y=target, cv=cv_strategy)
```

`solver='eigen'` + `shrinkage='auto'` 是 LDA 在特征少但可能有共线性的情况下的稳定配置。第二层之所以用简单的线性模型而不是再来一个复杂的树模型，是因为**第二层的输入已经是高度浓缩的预测信号，越简单的组合越不容易过拟合**。

#### 2.3.4 两套特征 × 两套模型，按 id 拼接

我们两次重复了上面的流程：

- **基准版（只用客户资料）**：用一版特征（`profiling`），得到 `submission_base`；
- **加信用局版**：在 `profiling` 特征基础上，再 join 聚合后的 `bki` 特征，得到 `submission_bki`。

最后，按「测试集里有没有客户资料」把两份结果拼在一起：

```python
submission_join = submission_bki.copy()
cp_index = raw_cp['APPLICATION_NUMBER'].to_list()
# 有客户资料的申请，用基准版；没有的，用含信用局的版本
submission_join.loc[submission_join[COLUMN_ID].isin(cp_index)] = \
    submission_base.loc[submission_base[COLUMN_ID].isin(cp_index)]
submission_join.to_csv('submission.csv', index=False)
```

为什么要这么拼？因为「有客户资料」和「没有客户资料」两种测试样本的**数据分布和可用特征集不同**，用同一套模型预测它们，相当于让模型去处理它没见过的情况。分开建模、分别预测，虽然代码多了一点，但逻辑更严谨。

### 2.4 模型测试与指标报告

我们在 5 折交叉验证下记录了每个折、每种方法的 AUC：

- 每组 LightGBM 的折平均 AUC 在 **0.7 几** 的量级；
- 三组 LightGBM 的 OOF 预测之间的相关性图（热力图）显示，它们彼此相关但并不完全重叠，这正是叠加能起作用的条件；
- 第二层 LDA 的折平均 AUC 比单组 LightGBM 略高，说明**简单叠加有一定的稳定作用**；
- 加入信用局聚合特征后，第二层模型的 AUC 进一步提升，因为逾期历史是比客户资料更直接的风险信号。

我们还比较了几种不同的「合并预测」方式，用来验证取平均是否最优：

```python
from scipy.stats import gmean

def mean_scores(scores, target):
    # 1) 算术平均
    print('AMean:', roc_auc_score(target, scores.mean(axis=1)))
    # 2) 几何平均：对"一个模型高、另一个模型极低"的情况更敏感
    print('GMean:', roc_auc_score(target, gmean(scores, axis=1)))
    # 3) 先转成名次再平均：忽略分数尺度差异，只看排序
    print('Rank :', roc_auc_score(target, scores.rank().mean(axis=1)))
```

对 AUC 这类基于排序的指标，**「先排名再平均」往往比直接平均概率更稳健**，因为不同模型的概率校准程度不一样。这也是很多比赛里最终集成会采用 rank average 的原因。

---

## 三、总结

本文中我们完成了 GeekBrains 竞赛数据分析的完整流程。我们首先通过 EDA 认清了数据：6 张互相关联的表、高度不平衡的二分类目标、客户资料在部分申请上缺失、信用局表是一对多关系。其次我们做了清洗（占位值转缺失、分类变量独热编码）和特征工程（月供/收入/年龄的比值特征，信用局贷款按整体、活跃、已结清三种口径聚合）。接着我们用三组不同超参数的 LightGBM 做 5 折交叉验证、产出折外预测，再用 LDA 做第二层叠加，并区分「有无客户资料」两套测试集分别建模、最终按 id 拼接。最后我们以 AUC 评估各组模型，用相关性和多种合并方式比较了叠加的收益。

这个比赛给我们最大的启发是：**表格数据竞赛的胜负，往往取决于「多表聚合」和「多模型叠加」这两件事**。把一对多的历史记录按业务含义聚合成一行，让每个客户的历史真正进入模型；用折外预测避免叠加时的信息泄漏；再用简单模型去组合复杂模型的输出——这三步比把单个模型调参调到极致更有效。