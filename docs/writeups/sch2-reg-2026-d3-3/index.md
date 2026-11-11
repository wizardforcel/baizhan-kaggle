# Regression with an Insurance Dataset（保险数据集回归）

## 一、比赛简介

### 1.1 比赛背景

保险（insurance）行业每天都在回答同一个问题：一个客户该收多少保费？保费定低了，保险公司要亏钱；定高了，客户就跑去别家投保。所以保险公司积累了大量客户的个人信息和保单信息，希望能**根据这些数据科学地估计每位客户该缴纳的保费金额**。

Kaggle 上的 "Regression with an Insurance Dataset"（保险数据集回归）比赛正是这样一个任务：主办方提供了一批客户的档案和保单数据，要求参赛者**只用表格特征**，预测每位客户应缴纳的保费（Premium Amount）。这类"给一张表、预测一个数值"的问题，在机器学习里叫作**回归（Regression）**。它属于入门级却是非常实用的一类任务。

难点主要有两个：第一，影响因素很多，年龄、收入、健康状况、既往理赔等等都相关，而且这些因素和保费之间大多不是简单的线性关系；第二，保费的分布很偏——大多数人保费低，少数人保费特别高，直接去预测大额样本容易出大偏差。因此这个比赛的比拼重点，落在**特征工程**和**对目标做合适的变换**上。

### 1.2 比赛目标

输入是一张客户信息表，主要有下面几类字段：

- **数值型**：Age（年龄）、Annual Income（年收入）、Number of Dependents（抚养人数）、Health Score（健康评分）、Vehicle Age（车龄）、Credit Score（信用分）、Insurance Duration（保险期限）、Previous Claims（既往理赔次数）等；
- **类别型**：Gender（性别）、Smoking Status（是否吸烟）、Marital Status（婚姻状况）、Education Level（学历）、Occupation（职业）、Exercise Frequency（运动频率）、Customer Feedback（客户评价）、Location（地区）、Policy Type（险种）、Property Type（房产类型）等；
- **时间型**：Policy Start Date（保单生效日期）；
- 每行还有一个 `id`，以及要预测的目标列 `Premium Amount`（保费金额）。

训练集给出 `id` + 所有特征 + `Premium Amount`；测试集只有 `id` + 所有特征。我们要为每条测试样本预测 `Premium Amount`，最后按 `id`、`Premium Amount` 两列提交。

### 1.3 评价标准

比赛用 **RMSLE（Root Mean Squared Logarithmic Error，对数均方根误差）** 打分，公式是：

$$RMSLE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(\ln(1+p_i)-\ln(1+a_i)\right)^2}$$

其中 $p_i$ 是预测值，$a_i$ 是真实值，$\ln$ 是自然对数。直观理解：普通 RMSE 直接比较"预测和真实相差多少"，而 RMSLE 比较的是"预测和真实差几个数量级"。对于保费这种跨度很大的金额，RMSLE 可以避免大额样本天然更容易产生大误差的偏差。分数**越低越好**，0 分表示完美。

这里有一个非常关键、本次比赛所有提交都用到的手法：**先对保费取 log1p（即 $\log(1+y)$），在 log 尺度上做回归——此时回归的 RMSE 恰好就等于 RMSLE；预测时再用 expm1 逆变换还原回原始金额。** 这样做还有一个额外好处：取对数后偏态分布被大幅拉平，模型不再被少数极端大额样本牵着走。

---

## 二、解题过程

整场比赛我们的主线是：**加载数据 → EDA 摸清字段与缺失 → 清洗与特征工程 → log 变换目标 → 交叉验证训练 → 集成 → 提交**。下面按四步展开。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，看看数据规模、列名和字段类型。

```python
# 导入常用库：pandas 处理表格，numpy 做数值计算
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")   # 关掉警告，让输出干净一些

# 读取训练集、测试集、官方提交样例
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")

# 看一眼数据规模
print("Train shape :", train.shape)
print("Test shape  :", test.shape)

# 看训练集前几行和字段类型，确认列结构
print(train.head())
print(train.dtypes)
```

输出显示训练集和测试集的**列结构一致**，只是训练集多了目标列 `Premium Amount`。字段里既有连续数值（年龄、收入、信用分……），也有文本类别（婚姻状况、职业……），还有一个日期列，是典型的"结构化表格"数据，一共二十来个字段。

其次，我们检查每条样本的缺失情况，因为缺失值会直接影响模型能不能训练。我们干脆把训练集和测试集拼在一起统计，样本更多、结论更可信：

```python
import matplotlib.pyplot as plt

# 拼接两个数据集，统一统计缺失
df = pd.concat([train, test])

# 计算每个字段的缺失比例（百分比），从小到大排序
null = (df.isnull().mean() * 100).sort_values(ascending=True)

# 画横向条形图：横轴是缺失比例，纵轴是字段名
plt.figure(figsize=(10, 6))
bars = plt.barh(null.index, null, color='#DC143C')
for i, v in enumerate(null):
    plt.text(v + 0.5, i, f"%{v:.1f}", va='center', fontsize=9)
plt.xlabel('NaN (%)')
plt.title('各列缺失数据比例')
plt.show()
```

从图里可以看到不少字段都有缺失，其中 `Occupation`（职业）缺失得尤其明显，`Previous Claims`（既往理赔）等字段也有缺失。这说明**数据清洗是绕不开的一步**——原始数据不是现成能直接喂给模型的。

### 2.2 数据清洗

接着我们处理缺失值。规则很简单：数值列用**中位数**填充（中位数不受极端值影响，比均值更稳），类别列用**众数**填充；某个字段缺失特别多就单独填 `Unknown`。对 `Previous Claims` 我们额外记了一个"是否缺失"的标记列，把"缺失"这件事本身也当成一个特征交给模型。

```python
# 数值列：用训练集的中位数填充
# 注意只统计训练集，否则会把测试集的信息泄漏进训练过程
num_cols = ['Age', 'Annual Income', 'Number of Dependents',
            'Health Score', 'Vehicle Age', 'Credit Score',
            'Insurance Duration']
df[num_cols] = df[num_cols].fillna(df.loc[train, num_cols].median())

# 既往理赔：先加一列“是否缺失”标记，再填中位数
df['Previous_Claims_Missing'] = df['Previous Claims'].isna().astype(int)
df['Previous Claims'] = df['Previous Claims'].fillna(
    df.loc[train, 'Previous Claims'].median())

# 类别列：用训练集众数填充
cat_cols = ['Marital Status', 'Customer Feedback']
df[cat_cols] = df[cat_cols].fillna(df.loc[train, cat_cols].mode().iloc[0])

# 职业缺失最多，统一填 Unknown
df['Occupation'] = df['Occupation'].fillna('Unknown')
```

填完缺失之后，我们继续做**特征工程**：先把日期列拆成年、月、星期几等周期特征，再构造几个保险领域里有业务含义的衍生特征——人均收入（抚养的人越多，同样收入下可支配的钱越少）、年均索赔次数（出险频率越高风险越大）、年龄×健康分（衡量身体状况的综合风险）等等。这些衍生特征能让树模型更容易找到好的分裂点。

```python
# 把保单生效日期转成时间类型，拆出周期特征
df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
df['Policy_Year']      = df['Policy Start Date'].dt.year
df['Policy_Month']     = df['Policy Start Date'].dt.month
df['Policy_DayOfWeek'] = df['Policy Start Date'].dt.dayofweek  # 0=周一
df['Policy_Age_Days']  = (pd.Timestamp('2026-04-19') - df['Policy Start Date']).dt.days

# 日期已拆成多个特征，原始日期列可以扔掉了
df = df.drop(columns=['Policy Start Date'])

# 新增业务特征：人均收入、年均索赔、年龄×健康分、总风险时长
df['Income_Per_Person']      = df['Annual Income'] / (df['Number of Dependents'] + 1)
df['Claims_Per_Year']        = df['Previous Claims'] / df['Insurance Duration']
df['Age_Health_Interaction'] = df['Age'] * df['Health Score']
df['Total_Risk_Duration']    = df['Vehicle Age'] + df['Insurance Duration']
```

然后是**编码**：模型只能吃数字，类别文本必须先变成数。我们分三种情况处理——二分类直接映射成 0/1；有顺序的类别（学历、客户评价、运动频率）按等级映射，保留大小关系；没有顺序的类别（婚姻状况、职业等）用 One-Hot 编码展开成一堆 0/1 列，让模型知道它们是"并列"的、彼此没有大小之分。

```python
# 二分类：直接映射成 0/1
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Smoking Status'] = df['Smoking Status'].map({'No': 0, 'Yes': 1})

# 有序类别：按等级高低映射成数字，顺序信息保留下来
education = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
feedback  = {'Poor': 0, 'Average': 1, 'Good': 2}
exercise  = {'Rarely': 0, 'Monthly': 1, 'Weekly': 2, 'Daily': 3}
df['Education Level']    = df['Education Level'].map(education)
df['Customer Feedback']  = df['Customer Feedback'].map(feedback)
df['Exercise Frequency'] = df['Exercise Frequency'].map(exercise)

# 无序类别：One-Hot 编码，drop_first 去掉第一列，避免特征彼此共线
nominal_cols = ['Marital Status', 'Occupation', 'Location',
                'Policy Type', 'Property Type']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
```

经过上面三步，`df` 里已经全是数值、没有缺失，可以直接进入建模环节了。

### 2.3 模型训练与超参数调优

然后，我们按目标列有没有值，把拼接好的表拆回训练集和测试集，并做最关键的一步：**log1p 目标变换**。前面 1.3 节说过，对保费取对数之后回归的 RMSE 就等于比赛要的 RMSLE；同时保费分布特别偏，取对数后分布被拉平，模型好学了不止一点。

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, root_mean_squared_error

# 拆回训练集和测试集
train = df[df['Premium Amount'].notna()]
test  = df[df['Premium Amount'].isna()]

# 特征矩阵去掉 id 和目标列
X = train.drop(columns=['id', 'Premium Amount'])
y = train['Premium Amount']
X_test = test.drop(columns=['id', 'Premium Amount'])

# 关键变换：log1p 即 log(1+y)，在 log 尺度上回归，RMSE 就等于 RMSLE
y_log = np.log1p(y)

# 5 折交叉验证：把数据切成 5 份，轮流拿 1 份验证、4 份训练，
# 最后取平均，评估结果比单次划分更稳定、更可信
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))       # OOF：把每折验证集的预测拼起来
test_preds = np.zeros(len(X_test))
```

我们首先没急着上复杂模型，而是先跑了一个社区里流传的**轻量基线**：直接 XGBoost + 交叉验证，配合早停。XGBoost 是梯度提升树（把很多棵小决策树一棵棵串起来、每棵都去纠正上一棵的残差）的经典实现，对表格数据几乎是默认的强基线。这里用 `tree_method='hist'`（直方图加速）让训练更快，`early_stopping_rounds=100` 表示验证集连续 100 轮不提升就提前收工——省时间也防过拟合。

```python
import xgboost as xgb

# 基线：XGBoost + 5 折交叉验证，每折单独训练、单独早停
for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y_log)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y_log.iloc[tr_idx], y_log.iloc[va_idx]

    model = xgb.XGBRegressor(
        n_estimators=2000,        # 最多 2000 棵树
        learning_rate=0.05,       # 学习率：每棵树只贡献 5% 的修正，学慢更稳
        max_depth=6,              # 每棵树的深度上限，防止单棵树太深
        subsample=0.8,            # 每棵树随机取 80% 样本，抗过拟合
        colsample_bytree=0.8,     # 每棵树随机取 80% 特征
        random_state=42,          # 固定随机种子，结果可复现
        tree_method='hist',       # 直方图近似分箱，显著加速训练
        early_stopping_rounds=100 # 验证集连续 100 轮不提升就停止
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_pred = model.predict(X_va)
    # 在 log 尺度上算 RMSE，数值上就等于要交的 RMSLE
    score = np.sqrt(mean_squared_error(y_va, val_pred))
    print(f"Fold {fold+1} RMSLE: {score:.5f}")
```

基线能跑通之后，我们再换**主力方案**：LightGBM 和 XGBoost 双模型。LightGBM 也是梯度提升树，但用"按叶子生长"的策略和直方图算法，训练更快、内存更省。两者参数上有意做了互补：LightGBM 放宽结构（`num_leaves=127`、不设 `max_depth`），靠 L1/L2 正则约束；XGBoost 保持 `max_depth=6`，用 `gamma`、`reg_alpha`、`reg_lambda` 控制复杂度。两个模型共用同一套 5 折切分，各自把每折测试集预测平均起来，还额外存下 OOF 预测，供最后集成使用。

```python
import lightgbm as lgb

# LightGBM 参数：叶节点多、正则足，适合细粒度拟合
lgb_params = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.03,         # 更小的学习率，配合更多棵树
    "num_leaves": 127,             # 每棵树最多 127 个叶节点
    "min_child_samples": 20,
    "feature_fraction": 0.8,       # 每棵树随机用 80% 特征
    "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l1": 0.1, "lambda_l2": 0.1,  # L1/L2 正则
    "n_estimators": 3000, "random_state": 42, "n_jobs": -1,
}

# 为两个模型分别准备 OOF 预测和测试集预测数组
oof_lgb, oof_xgb = np.zeros(len(X)), np.zeros(len(X))
pred_lgb, pred_xgb = np.zeros(len(X_test)), np.zeros(len(X_test))

# LightGBM 的 5 折训练循环
for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y_log)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y_log.iloc[tr_idx], y_log.iloc[va_idx]

    m_lgb = lgb.LGBMRegressor(**lgb_params)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

    oof_lgb[va_idx] = m_lgb.predict(X_va)                         # 验证集预测，留作评估
    pred_lgb += m_lgb.predict(X_test) / kf.get_n_splits()         # 测试集预测取平均

print("LightGBM OOF RMSE:", root_mean_squared_error(y_log, oof_lgb))
```

XGBoost 的训练循环和上面完全一样，只是参数换了，这里只把参数贴出来，循环照抄即可：

```python
xgb_params = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "learning_rate": 0.03, "max_depth": 6, "n_estimators": 3000,
    "min_child_weight": 5,           # 叶子最少样本权重，数值越大越保守
    "subsample": 0.8, "colsample_bytree": 0.8,
    "gamma": 0.1,                    # 分裂所需的最小增益，越大越不容易分裂
    "reg_alpha": 0.1, "reg_lambda": 1.0,   # L1/L2 正则
    "random_state": 42, "n_jobs": -1, "tree_method": "hist",
}

# 循环结构与上面 LightGBM 一致，只需把 m_lgb 换成
# xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100)，
# 并分别写入 oof_xgb、pred_xgb，这里不再重复贴一遍
```

顺带说一句我们做过的**模型横向对比**：社区里有份方案用一个函数一口气测了线性回归、Ridge、Lasso、ElasticNet、KNN、ExtraTrees、决策树、梯度提升、XGBoost 共九种模型——线性模型先标准化再喂，树模型直接喂，最后按 R² 排序。实测对比中**梯度提升树一族明显占优**，线性模型在这种多特征、多非线性的数据上吃亏；轮换目标从原始保费换成 log 变换后的值重测一遍后，我们确认 log 变换对多数模型都是稳赚不赔的，于是把主模型锁定在 LightGBM 和 XGBoost 上。所谓"调参"，除了学习率、树深、采样比例、正则强度这些常规项之外，**早停帮我们把树的数量自动选好**——这比手动试出固定 `n_estimators` 更省事、也更抗过拟合。

### 2.4 模型测试与指标报告

最后是集成与提交。**集成（Ensemble）** 的思路是"三个臭皮匠顶个诸葛亮"：LightGBM 和 XGBoost 结构不同，犯的错不完全一样，把它们的预测按权重混合往往比单个模型更稳。我们先试 0.6:0.4 的加权混合；再更进一步试**堆叠（Stacking）**：把两个模型的 OOF 预测当作两个新特征喂给一个简单的 Ridge 回归，让模型自己学出最优权重。最后比较两种方式在 OOF 上的 RMSE（即 RMSLE），分数低的那个胜出。

```python
# —— 第一种集成：加权混合 ——
LGB_WEIGHT, XGB_WEIGHT = 0.6, 0.4          # 权重可以先手调再微调
oof_blend  = LGB_WEIGHT * oof_lgb + XGB_WEIGHT * oof_xgb
pred_blend = LGB_WEIGHT * pred_lgb + XGB_WEIGHT * pred_xgb
blend_score = root_mean_squared_error(y_log, oof_blend)
print(f"Blended OOF RMSE: {blend_score:.5f}")

# —— 第二种集成：Ridge 堆叠，用 OOF 预测当特征，学出最优权重 ——
from sklearn.linear_model import Ridge
stacker_X_train = np.column_stack([oof_lgb, oof_xgb])
stacker_X_test  = np.column_stack([pred_lgb, pred_xgb])
ridge = Ridge(alpha=1.0)
ridge.fit(stacker_X_train, y_log)
stacked_score = root_mean_squared_error(y_log, ridge.predict(stacker_X_train))
print(f"Ridge Stacked OOF RMSE: {stacked_score:.5f}")

# —— 谁的 OOF 分数低就用谁 ——
if stacked_score < blend_score:
    final_pred_log = ridge.predict(stacker_X_test)
else:
    final_pred_log = pred_blend

# —— 逆变换并提交 ——
# expm1 是 log1p 的反函数，把 log 尺度的预测还原成真实保费金额
final_pred = np.expm1(final_pred_log)
final_pred = np.clip(final_pred, 0, None)   # 保费不可能为负，负数直接压成 0

submission = pd.DataFrame({
    'id': test['id'],
    'Premium Amount': final_pred
})
submission.to_csv('submission.csv', index=False)
print(submission.head(10))
```

提交前我们再检查一遍：提交文件的行数和测试集一致、两列名 `id` 和 `Premium Amount` 与官方样例一致。比赛里因为格式不对白白丢分的情况可不少，这一眼值得看。拿到 OOF 分数后，我们还会画一下特征重要度，看看模型主要靠哪些特征决策，方便下一轮迭代时决定特征工程该往哪个方向使劲。

---

## 三、总结

本文中我们完成了保险数据集回归任务的完整流程。我们首先通过 EDA 摸清了数据：这是一张典型的表格回归数据集，包含年龄、收入、健康评分等数值字段，性别、学历、职业等类别字段以及保单生效日期，共二十来个字段，多个字段存在缺失；其次我们按"数值列填中位数、类别列填众数、缺失特别多的单独填 Unknown"的规则清洗数据，随后完成日期拆分、人均收入、年均索赔等衍生特征构造和三类编码；接着我们确立"log1p 目标变换 + 5 折交叉验证"的基调，先以 XGBoost 快速跑通基线，再训练 LightGBM 与 XGBoost 两个主力模型；然后我们比较 0.6:0.4 加权混合与 Ridge 堆叠两种集成方式，选择 OOF 分数更优者作为最终预测；最后用 expm1 逆变换还原保费、把负数压成 0，生成提交文件。

同一场比赛里还有更省事的做法——比如只做一个 XGBoost、类别全部 Label 编码就直接提交易，也能拿到合理的分数；我们展示的是特征更充分、评估更严谨的完整路线。这个比赛给我们最大的启发是：**表格回归的胜负手不在模型选得多花哨，而在目标变换（log1p）、特征工程和交叉验证评估这三件基本功上**，把这些做扎实，再配合集成，就能稳定地压住 RMSLE。

> 说明：本文代码综合整理自比赛页面的多份公开提交，材料中的代码只打印各折分数而未展示具体数值，因此正文不出具体分数；所有参数和流程均可在材料中找到出处。