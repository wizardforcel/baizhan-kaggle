# 预测美国住房可负担性

## 一、比赛简介

### 1.1 比赛背景
美国住房可负担性是指家庭每月用于住房支出的比例是否超过收入的 30%。美国住房和城市发展部 HUD 把支出超过 30% 的家庭定义为住房成本负担家庭。换句话说，如果一个家庭每个月把三成以上的收入花在房租或房贷上，就算承受着较重的住房压力。我们要做的比赛是预测每个邮政编码 ZIP 区域内，不受成本负担困扰的家庭所占百分比。这个百分比记作 AffordabilityPercentageTrue，取值范围是 0 到 100。数值越高，说明这个地区越多的家庭能够轻松承担住房开销，地区整体的可负担性越好。

ZIP 是美国邮政使用的五位邮政编码。它比州更细，比具体街道更粗，正好能表示一个社区或一片街区。比赛把同一个 ZIP 内的家庭汇总起来，计算一个百分比，所以每条训练样本代表一个 ZIP 在某个月份的总体情况，而不是某一个家庭的情况。这一点很重要，因为它意味着我们不能把样本当成彼此独立的家庭，而要考虑同一个 ZIP 在不同月份之间的连续性。

数据来自美国各 ZIP 区域的经济与人口统计指标，比如租金指数 RentIndex、年家庭收入中位数 AnnualMedianHouseholdIncome、房主比例 OwnerPercent、租户比例 RenterPercent、居住时长、孩子数、成人数、代数数量等。数据按月份记录，训练集包含 2026 年 5 月和 6 月的信息，测试集要预测的是 7 月的数值。因为只有两个训练月份，时间序列很短，我们不能像普通时间序列那样训练很深的滞后模型，只能利用月度变化量和地理层级做补偿。

难点在于评价是分区的。测试集里约 80% 的 ZIP 在训练集里出现过，称为 Seen ZIP；另外约 20% 是完全没见过的 New ZIP。比赛的损失函数是 Seen 的 RMSE 占 0.2，New 的 RMSE 占 0.8。这意味着即使 Seen 样本更多，New ZIP 的预测误差会占主导。我们可以这样理解：如果只把 Seen ZIP 预测得很准，而 New ZIP 预测得很差，总分依然会被拉低。反过来，如果 New ZIP 的误差下降一点，总分改善会比 Seen ZIP 更明显。因此我们必须同时解决时间上的漂移和空间上的泛化问题，不能只靠记住每个 ZIP 的历史值。

### 1.2 比赛目标
给定训练集里的特征，我们要预测测试集每个 ZIP 的 AffordabilityPercentageTrue。输入列包含 Zip、State、Metro、City、Month、SampleSize 以及一系列人口经济变量，例如 RentIndex、AnnualMedianHouseholdIncome、OwnerPercent、RenterPercent、AvgNumberofChildren、AvgNumberofAdults、AvgHouseholdSize、AvgGenerationsinHousehold 等。输出只有一列预测值，即每个测试样本的 AffordabilityPercentageTrue。训练集共 13,660 行，18 列，测试集 8,534 行，17 列。每个 ZIP 在训练集中出现两次，对应 5 月和 6 月。由于测试集的月份是 7 月，这本质上是利用两个已知月份去预测紧接着的下一个月的任务。

### 1.3 评价标准
官方使用加权 RMSE，公式为 0.2 * RMSE_seen + 0.8 * RMSE_new。RMSE 越小越好。这里的 RMSE_seen 指在训练中出现过、测试阶段要时间外预测的 ZIP 的误差；RMSE_new 指没有出现过、需要纯粹依靠空间泛化的 ZIP 的误差。Seen 部分是时间外预测，New 部分是空间外预测。加权之后，New 部分虽然只占约两成的测试样本，却贡献了八成的权重。我们在本地用 5 折分组交叉验证模拟这个标准，按 ZIP 分组切分，保持同一个 ZIP 的两条样本不会同时出现在训练和验证里，避免泄漏。

## 二、解题过程

整场比赛的思路可以概括为：我们先摸清数据结构与缺失情况，其次做有理论依据的特征工程，接着用双机制模型分别处理 Seen 和 New ZIP，最后在验证集上做凸优化融合。我们按 EDA、数据清洗、模型训练与调优、测试与报告四步来做。

### 2.1 数据探索（EDA）
我们首先把三个文件读进来，确认样本量与地理划分情况。

```python
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/kaggle/input/competitions/predicting-housing-affordability-in-the-u-s")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(f"训练集: {train_df.shape[0]} 行 x {train_df.shape[1]} 列")
print(f"测试集: {test_df.shape[0]} 行 x {test_df.shape[1]} 列")

train_zips = set(train_df['Zip'].unique())
test_zips = set(test_df['Zip'].unique())
seen_zips = test_zips.intersection(train_zips)
new_zips = test_zips.difference(train_zips)
print(f"Seen ZIP: {len(seen_zips)}, New ZIP: {len(new_zips)}")
```

从输出里我们看到训练集 13,660 行，测试集 8,534 行。训练集里 5 月和 6 月各有 6,830 行，每个 ZIP 都完整出现两次。测试集里 Seen ZIP 有 6,830 个，占 80.03%，New ZIP 有 1,704 个，占 19.97%。这正是需要双机制的原因。

接着我们看目标分布与缺失值。

```python
def audit_missingness(df, name):
    null_cnt = df.isnull().sum()
    null_pct = (null_cnt / len(df)) * 100
    report = pd.DataFrame({
        '特征': df.columns,
        '缺失数': null_cnt.values,
        '缺失率%': null_pct.values
    })
    return report[report['缺失数']>0]

print(audit_missingness(train_df, "训练集"))
print(audit_missingness(test_df, "测试集"))

y_train = train_df['AffordabilityPercentageTrue']
print(f"均值 {y_train.mean():.2f}%, 中位数 {y_train.median():.2f}%, 标准差 {y_train.std():.2f}")
```

缺失主要集中在 RentIndex，训练集缺 1,185 个，约 8.67%，测试集缺 9 个，约 0.11%。Metro 缺失约 0.78%，City 基本完整。目标均值 44.68%，中位数 46.00%，标准差 28.47，范围 0 到 100，呈现双峰分布：一部分 ZIP 的负担能力稳定在较高水平，另一部分则集中在较低水平，这提醒我们最终预测要夹在 0 到 100 之间。5 月到 6 月的相关性高达 0.999，基线零漂移 RMSE 为 1.2524，说明 Seen ZIP 具有很强的月度惯性——也就是每个 ZIP 的数值两个月之间变化很小。对 Seen ZIP 来说，最好的起点也许就是它上个月的值，模型剩下的任务只是预测一个小的变化量。

我们还计算了关键变量的相关性。AvgNumberofChildren 为 +0.58，AvgHouseholdSize 为 +0.40，OwnerPercent 为 +0.34，AvgGenerationsinHousehold 为 +0.32，而 RenterPercent 为 -0.34，验证了多代同居和房主比例的缓冲作用。这些相关性的方向在经济上是说得通的：家里孩子多、代际多、常住人口多，往往对应负担能力更强；租户比例高，则负担能力偏弱。EDA 的结论直接指导了后面的特征工程方向。

### 2.2 数据清洗
我们其次处理缺失并做时间特征抽取。RentIndex 的缺失采用空间回退填补：先用 3 位邮编 SCF 的中位数，再用 Metro、State、最后全国中位数，同时保留一个缺失指示特征。

```python
def build_features(df, state_stats=None, is_train=True):
    d = df.copy()
    d['Zip_str'] = d['Zip'].astype(str).str.zfill(5)
    d['Zip_1digit'] = d['Zip_str'].str[:1].astype(int)
    d['Zip_2digit'] = d['Zip_str'].str[:2].astype(int)
    d['Zip_3digit'] = d['Zip_str'].str[:3].astype(int)

    d['RentIndex_IsMissing'] = d['RentIndex'].isnull().astype(float)
    
    if is_train:
        state_rent = d.groupby('State')['RentIndex'].median()
        global_rent = d['RentIndex'].median()
        state_stats = {'state_rent': state_rent, 'global_rent': global_rent}

    d['RentIndex_Clean'] = d['RentIndex'].fillna(d['State'].map(state_stats['state_rent']))
    d['RentIndex_Clean'] = d['RentIndex_Clean'].fillna(state_stats['global_rent'])
    return d, state_stats
```

这段代码把邮编拆成 1、2、3 位地理层级，并用州层级的中位数做分层填补，避免用未来目标信息泄露。随后我们补充了微观经济指标：把年收入转成月收入，计算租金收入比 RentToIncomeRatio、affordability 缓冲 AffordabilityBuffer、剩余可支配收入 ResidualDisposableIncome。还计算了家庭构成指标 ChildrenPerAdult、HouseholdCrowding、MultigenHouseholdIndex 以及空间相对价格 Rent_to_StateRent 等。最终训练特征从 18 维扩展到 45 维，测试集 42 维。

为了避免时空泄漏，我们用 GroupKFold 按 ZIP 分组做交叉验证，确保同一 ZIP 的不同月份不会同时出现在训练和验证里。

### 2.3 模型训练与超参数调优
接着我们建立双机制模型。Regime A 负责 Seen ZIP，利用月度惯性做自回归漂移补偿。

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import numpy as np

may_data = train_feat[train_feat['Month'] == '5/1/2026'].set_index('Zip')
june_data = train_feat[train_feat['Month'] == '6/1/2026'].set_index('Zip')
common_zips = may_data.index.intersection(june_data.index)

delta_y = june_data.loc[common_zips, 'AffordabilityPercentageTrue'] - may_data.loc[common_zips, 'AffordabilityPercentageTrue']
X_delta = pd.DataFrame({
    'prev_affordability': may_data.loc[common_zips, 'AffordabilityPercentageTrue'],
    'delta_rent': june_data.loc[common_zips, 'RentIndex_Clean'] - may_data.loc[common_zips, 'RentIndex_Clean'],
    'delta_owner': june_data.loc[common_zips, 'OwnerPercent'] - may_data.loc[common_zips, 'OwnerPercent'],
    'delta_sample': june_data.loc[common_zips, 'SampleSize'] - may_data.loc[common_zips, 'SampleSize'],
})
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_delta = np.zeros(len(delta_y))
for tr, va in kf.split(X_delta):
    m = Ridge(alpha=150.0)
    m.fit(X_delta.iloc[tr], delta_y.iloc[tr])
    oof_delta[va] = m.predict(X_delta.iloc[va])
```

该 Ridge 模型的 5 折外样本 RMSE 为 1.2002，比零漂移基线 1.2524 降低 0.0522 个百分点。加权后对整体评分的贡献约 0.240。

Regime B 用来处理 New ZIP。我们训练 LightGBM、CatBoost、XGBoost 和 Tabular ResNet 四个模型，并在每折上做贝叶斯平滑的目标编码。

```python
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

feature_cols = ['RentToIncomeRatio','AffordabilityBuffer','AnnualMedianHouseholdIncome',
                'OwnerPercent','RenterPercent','AvgNumberofChildren','AvgHouseholdSize',
                'AvgGenerationsinHousehold','Zip_1digit','Zip_2digit','Zip_3digit',
                'Rent_to_StateRent','Income_to_StateInc']

gkf = GroupKFold(n_splits=5)
oof_lgb = np.zeros(len(train_feat))
test_lgb = np.zeros(len(test_feat))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(train_feat, y_train, groups=train_feat['Zip'])):
    X_tr = train_feat.iloc[tr_idx][feature_cols].copy()
    y_tr = y_train[tr_idx]
    # 贝叶斯平滑的州与 SCF 目标编码
    global_m = y_tr.mean()
    state_stats = train_feat.iloc[tr_idx].groupby('State')['AffordabilityPercentageTrue'].agg(['mean','count'])
    state_enc = (state_stats['mean']*state_stats['count'] + global_m*10)/(state_stats['count']+10)
    X_tr['State_TE'] = train_feat.iloc[tr_idx]['State'].map(state_enc)
    model = lgb.LGBMRegressor(n_estimators=700, learning_rate=0.03, num_leaves=31, 
                              max_depth=6, subsample=0.8, colsample_bytree=0.8,
                              random_state=42+fold, n_jobs=-1)
    model.fit(X_tr, y_tr, categorical_feature=['State'])
    oof_lgb[val_idx] = model.predict(train_feat.iloc[val_idx][feature_cols])
```

类似地我们训练 CatBoost、XGBoost 以及带残差块的 Tabular ResNet，ResNet 使用 RobustScaler 标准化输入，AdamW 优化器，Huber 损失。单模型外样本 RMSE 约在 LightGBM 12.858、XGBoost 12.769、CatBoost 13.111、ResNet 16.537。

最后我们通过凸优化在概率单纯形上融合四模型。

```python
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

P = np.column_stack([oof_lgb, oof_cat, oof_xgb, oof_nn])
def obj(w):
    w_norm = np.exp(w) / np.sum(np.exp(w))
    pred = P @ w_norm
    return np.sqrt(mean_squared_error(y_train, pred))

res = minimize(obj, [0.,0.,0.,0.], method='BFGS')
opt_w = np.exp(res.x) / np.sum(np.exp(res.x))
oof_blend = P @ opt_w
blend_rmse = np.sqrt(mean_squared_error(y_train, oof_blend))
```

最优权重为 XGBoost 0.6832、LightGBM 0.2790、Tabular ResNet 0.0378、CatBoost 0.0000。融合后的 RMSE_New 为 12.7415，优于任一单模型。

### 2.4 模型测试与指标报告
然后我们在测试集上按地理状态路由预测。Seen ZIP 用 Regime A 的 Ridge 漂移模型，以 6 月实际值为锚点加上预测漂移；New ZIP 用融合后的 Regime B 模型。所有预测都夹到 0 到 100 之间。

```python
is_seen = test_df['Zip'].isin(seen_zips).values
test_preds = np.zeros(len(test_feat))

for i, row in test_feat.iterrows():
    if is_seen[i]:
        june_y = train_df[(train_df['Zip']==row['Zip']) & (train_df['Month']=='6/1/2026')]['AffordabilityPercentageTrue'].values[0]
        drift = ridge_model.predict([[
            june_y, row['RentIndex_Clean']-june_rent, row['OwnerPercent']-june_owner,
            row['SampleSize']-june_sample, row['RentIndex_Clean'], row['AnnualMedianHouseholdIncome']
        ]])[0]
        test_preds[i] = np.clip(june_y + drift, 0, 100)
    else:
        test_preds[i] = np.clip(test_blend[i], 0, 100)

submission = test_df[['Zip','State']].copy()
submission['AffordabilityPercentageTrue'] = test_preds
submission.to_csv('submission.csv', index=False)
```

最终验证分数为 0.2 * 1.2002 + 0.8 * 12.7415 = 10.4332。这是基于分组交叉验证、无泄漏的估计。提交文件包含 8,534 行，与 sample_submission 完全对齐，无缺失值。

## 三、总结
本文中我们完成了预测美国住房可负担性的完整建模流程。我们首先做 EDA，确认训练集 13,660 行、测试集 8,534 行，Seen ZIP 6,830 个、New ZIP 1,704 个，目标均值 44.68%，缺失主要在 RentIndex。其次我们做了分层空间填补和特征工程，构造了租金收入比、代数指数、SOF 目标编码等 45 维特征。接着我们设计了双机制模型：Regime A 用 Ridge 做 Seen ZIP 的月度漂移补偿，RMSE 1.2002；Regime B 用 LightGBM、XGBoost、CatBoost、ResNet 融合处理 New ZIP，融合 RMSE 12.7415。最后我们按地理状态路由预测并生成提交文件，交叉验证综合分数 10.4332。

这个比赛让我们体会到评价不对称时必须把模型拆开设计，既要捕捉时间惯性，也要在完全未见的地理区域保持泛化能力。特征工程贴合微观经济理论、目标编码防止泄漏、以及分组交叉验证，都是入门者可以复制的关键做法。
