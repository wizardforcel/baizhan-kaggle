# Flavours of Physics：寻找 τ → 3μ 衰变（Kernels Only）

## 一、比赛简介

### 1.1 比赛背景

这个比赛来自欧洲核子研究中心 CERN 的大型强子对撞机 LHC。粒子物理里有一个叫「τ 轻子」（读作 tau）的粒子，它很重、寿命极短，会衰变成其他粒子。标准模型预言，τ 衰变成三个 μ 子（μ 是比 τ 轻很多的同类粒子）这件事几乎不会发生——物理学家一直没观察到它。如果真观察到了，就可能说明存在标准模型之外的新物理。

难点在于：这种衰变即使存在也极其稀有，实验里绝大多数事件都是「背景」（普通过程产生的、长得很像的干扰事件）。比赛给出的训练集是「模拟出来的信号事件」加上「真实数据里的背景事件」混合而成：信号标签为 1，背景标签为 0。换句话说，我们要教模型在大量长得相似的噪声里，找出那一点点可能存在的稀有信号。

### 1.2 比赛目标

输入是一次粒子碰撞事件的物理量，输出是「这次碰撞里发生 τ → 3μ 衰变的概率」。训练集 `training.csv` 里每一行是一个事件，包含几十列物理特征，例如：

- `LifeTime`：衰变粒子的寿命；
- `FlightDistance`、`FlightDistanceError`：飞行距离及其误差；
- `IP`、`IPSig`：碰撞点到粒子的最近距离（impact parameter）及其显著性；
- `dira`：飞行方向与动量方向的夹角余弦；
- `VertexChi2`：衰变顶点的拟合质量；
- `pt`、`p0_p`、`p1_p`、`p2_p`：各个粒子的横向动量、动量；
- `isolationa` 到 `isolationf`、`iso`、`ISO_SumBDT`、`p0_IsoBDT` 等：隔离度，用来衡量这个粒子周围有没有别的粒子跟着；
- `p0_track_Chi2Dof`、`p0_IP`、`p0_IPSig`、`p0_pt` 等：三个子粒子各自的轨迹质量、最近距离和动量；
- `SPDhits`：探测器上被击中的像素数；
- `signal`：标签，1 是信号、0 是背景；
- `mass`、`production`、`min_ANNmuon`：只用于检验，不作为特征。

测试集 `test.csv` 没有 `signal`，也没有 `mass`、`production`、`min_ANNmuon` 这几列，我们要给每个 `id` 填一个 0 到 1 之间的 `prediction`。

### 1.3 评价标准

排行榜使用的是加权截断 ROC AUC，可以理解成对 ROC 曲线下面积做加权，重点看低假阳性率、低真阳性率区间。数值越大表示区分信号和背景的能力越强。

但 AUC 不能随便刷。比赛额外设了两道「体检」，模型必须先通过才有资格参与评分：

但 AMS 不能随便刷。比赛额外设了两道「体检」，模型必须先通过才有资格参与评分：

1. **KS 检验（Kolmogorov–Smirnov）**：用一份叫 `check_agreement` 的对照数据，比较「真实数据」和「模拟数据」上模型打分的分布。KS 值要求 **小于 0.09**。如果模型只是在死记模拟数据的特征，真实数据和模拟数据就会分得很开，KS 值会超标。
2. **CvM 检验（Cramér–von Mises）**：用 `check_correlation` 数据，检查模型的预测值和事件重建质量 `mass` 有没有相关性，要求 **小于 0.002**。也就是说，模型不能靠「质量」这条线索作弊——因为在真实搜索里，质量必须被「盲」掉。

所以这个比赛的真正难点不是单纯把 AUC 刷高，而是**在高 AUC 的同时保持对模拟数据和真实数据一视同仁，并且不依赖质量**。

---

## 二、解题过程

我们的主线是：先把原始物理量做一轮特征工程，其次用 LightGBM 训练一个强基线并用贝叶斯优化调参，接着用 UGBC（一种带「平坦性损失」的梯度提升）压住 CvM，最后用 KS/CvM 两道检验把关并生成提交文件。

### 2.1 数据探索（EDA）

我们首先把四个文件都读进来，看看规模和数据长什么样：

```python
import pandas as pd
import numpy as np

# 训练集、测试集，以及两份用于「体检」的对照数据
train_set = pd.read_csv('../input/training.csv.zip', index_col='id')
test = pd.read_csv('../input/test.csv.zip', index_col='id')
check_agreement = pd.read_csv('../input/check_agreement.csv', index_col='id')
check_correlation = pd.read_csv('../input/check_correlation.csv', index_col='id')

# 看形状：训练集约 6.7 万行，测试集约 85 万行
print(train_set.shape, test.shape)
print(check_agreement.shape, check_correlation.shape)

# 看标签比例和缺失情况
print(train_set['signal'].value_counts())
print(train_set.isnull().sum().sum())
```

打印结果告诉我们：训练集约 67,553 行、51 列；测试集约 855,819 行，规模是训练集的十倍还多；`check_agreement` 约 33 万行，`check_correlation` 只有 5 千多行。训练集里 `signal` 只有 0 和 1 两种取值，没有缺失值。

接着我们看各物理量的取值范围，重点是那些「可以互相组合」的量：

```python
# 只挑选一部分有代表性的列看统计量，避免输出太长
cols = ['LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError',
        'IP', 'IPSig', 'VertexChi2', 'pt', 'iso', 'SPDhits']
train_set[cols].describe().T
```

从统计量里能看出两件事。第一，`dira` 基本都在 1 附近，信号事件的三个 μ 子应该来自同一个顶点，方向非常一致；背景事件则更散。第二，`LifeTime`、`FlightDistance`、`IP` 这些量的量纲差异很大，而且`FlightDistance` 的误差列 `FlightDistanceError` 本身就暗示了「距离除以误差」是一个更有物理意义的信噪比。这为我们后面的特征工程提供了直接依据：**不要只把原始列喂给模型，要把物理上相关的量组合起来**。

我们还特意对比了训练集和测试集的列名。测试集少了 `mass`、`production`、`min_ANNmuon`、`signal` 四列，这意味着任何依赖这些列的训练代码，都必须在上测试集之前把它们去掉。很多初学者在这里踩坑：本地交叉验证时模型偷偷用上了 `mass`，分数漂亮，一提交就掉下来，因为测试集根本没有这一列。

### 2.2 数据清洗

这份数据的「脏」不在缺失值，而在**冗余和不该用的列**。我们做了两件事。

第一，删掉不能用的列。`mass`、`production`、`min_ANNmuon` 是训练集独有的辅助列，测试集里没有，留着会让训练和测试的特征对不上；`SPDhits` 在测试集里也没有，同样删掉。此外还有一些原始列在特征工程之后变成了冗余，我们也一并剔除。

第二，做特征工程。这是本方案的核心。我们按物理含义把原始量组合成新的信噪比和极值特征：

```python
def add_features(df):
    # 飞行距离除以误差，再平方：距离的「显著性」，越大越像真的飞出去了
    df['flight_dist_sig2'] = (df['FlightDistance'] / df['FlightDistanceError']) ** 2
    # 三个子粒子轨迹拟合质量里最差的那个，用来抓「有一个粒子很假」的情况
    df['p_track_Chi2Dof_MAX'] = df[['p0_track_Chi2Dof',
                                    'p1_track_Chi2Dof',
                                    'p2_track_Chi2Dof']].max(axis=1)
    # 飞行距离除以三粒子总动量：单位动量能飞多远
    df['NEW_FD_SUMP'] = df['FlightDistance'] / (df['p0_p'] + df['p1_p'] + df['p2_p'])
    # 寿命乘以平均最近距离：寿命长且远离原点的更可能是信号
    df['NEW5_lt'] = df['LifeTime'] * (df['p0_IP'] + df['p1_IP'] + df['p2_IP']) / 3
    df['flight_dist_sig'] = df['FlightDistance'] / df['FlightDistanceError']
    # 最近距离与方向一致性的乘积
    df['NEW_IP_dira'] = df['IP'] * df['dira']
    # 整体 IP 与两两 IP 的比值，衡量三个粒子是否来自同一顶点
    df['p0p2_ip_ratio'] = df['IP'] / df['IP_p0p2']
    df['p1p2_ip_ratio'] = df['IP'] / df['IP_p1p2']
    # 三个 DOCA 里最大的那个（DOCA 越小越像来自同一顶点）
    df['DCA_MAX'] = df[['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    # 隔离度的最小值：只要有一个子粒子被别的粒子「跟着」，就值得怀疑
    df['iso_bdt_min'] = df[['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df[['isolationa', 'isolationb', 'isolationc',
                        'isolationd', 'isolatione', 'isolationf']].min(axis=1)
    # 隔离度连乘，放大「六个方向都很干净」这件事
    df['NEW_iso_abc'] = df['isolationa'] * df['isolationb'] * df['isolationc']
    df['NEW_iso_def'] = df['isolationd'] * df['isolatione'] * df['isolationf']
    # 三个粒子的 IP、动量、IPSig 求和
    df['NEW_pN_IP'] = df['p0_IP'] + df['p1_IP'] + df['p2_IP']
    df['NEW_pN_p'] = df['p0_p'] + df['p1_p'] + df['p2_p']
    df['NEW_pN_IPSig'] = df['p0_IPSig'] + df['p1_IPSig'] + df['p2_IPSig']
    # 距离除以寿命，等价于「速度」的一种度量
    df['NEW_FD_LT'] = df['FlightDistance'] / df['LifeTime']
    return df

train_add = add_features(train_set)
test_add = add_features(test)
```

加完这些列之后，训练集从 51 列变成 68 列。然后我们写一份「保留清单」，只留下信息量最高的一批列，把原始的那些单粒子量删掉，避免同一信息被重复计算：

```python
# 要删掉的列：id 已经作为索引、辅助列、以及被新特征替代的原始列
filter_out = ['min_ANNmuon', 'production', 'mass', 'signal', 'SPDhits',
              'CDF1', 'CDF2', 'CDF3',
              'isolationa', 'isolationb', 'isolationc',
              'isolationd', 'isolatione', 'isolationf',
              'p0_pt', 'p1_pt', 'p2_pt', 'p0_p', 'p1_p', 'p2_p',
              'p0_eta', 'p1_eta', 'p2_eta',
              'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT',
              'p0_IP', 'p1_IP', 'p2_IP', 'IP_p0p2', 'IP_p1p2',
              'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof',
              'p0_IPSig', 'p1_IPSig', 'p2_IPSig',
              'DOCAone', 'DOCAtwo', 'DOCAthree']

# 只保留新造出来的那批列
col = [f for f in train_add.columns if f not in filter_out]
print(len(col))   # 训练和测试用完全相同的 col 列表，保证特征对齐
```

### 2.3 模型训练与超参数调优

我们选 LightGBM 作为主力模型。它属于「梯度提升树」：一棵树一棵树地加，每一棵新树都专门去修正前面所有树犯的错，最后把很多棵树的输出加起来。它训练快、能自动处理特征之间的非线性关系，很适合这种表格型物理数据。

调参用的是**贝叶斯优化**。普通网格搜索是把参数组合一个一个试，很费时间；贝叶斯优化会先试几组，然后根据结果推测「哪一块参数空间更可能出好成绩」，把下一次试验放在那里。

```python
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score

# 把 DataFrame 包成 LightGBM 的数据集，标签是 signal
train_data = lgb.Dataset(train_add[col], label=train_set['signal'])

def lgb_eval(num_leaves, feature_fraction, bagging_fraction,
             max_depth, lambda_l1, lambda_l2,
             min_split_gain, min_child_weight):
    # 基础参数：二分类，最多 50000 棵树，学习率很小，靠早停决定何时停
    params = {'application': 'binary',
              'num_iterations': 50000,
              'learning_rate': 0.0001,
              'early_stopping_round': 100,
              'metric': 'auc'}
    # 把贝叶斯优化传来的浮点数整理成合法取值
    params['num_leaves'] = int(round(num_leaves))
    params['feature_fraction'] = max(min(feature_fraction, 1), 0)
    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    params['max_depth'] = int(round(max_depth))
    params['lambda_l1'] = max(lambda_l1, 0)
    params['lambda_l2'] = max(lambda_l2, 0)
    # 5 折交叉验证，返回平均 AUC 作为这一组参数的分数
    cv = lgb.cv(params, train_data, nfold=5, seed=42,
                stratified=True, metrics=['auc'])
    return cv['auc-mean'][-1]

# 搜索范围：叶子数、列采样、行采样、树深、L1/L2、最小分裂增益、叶子最小权重
lgbBO = BayesianOptimization(lgb_eval, {
    'num_leaves': (20, 614),
    'feature_fraction': (0.4, 1.0),
    'bagging_fraction': (0.4, 1.0),
    'max_depth': (3, 15),
    'lambda_l1': (0, 10),
    'lambda_l2': (0, 10),
    'min_split_gain': (0, 0.1),
    'min_child_weight': (1, 50),
}, random_state=42)

lgbBO.maximize(init_points=20, n_iter=30)
```

搜索输出里，最好的两组交叉验证 AUC 分别是 **0.9134** 和 **0.9181**，对应的参数大致是叶子数 550 左右、树深 7 到 9、`feature_fraction` 0.84 以上、`bagging_fraction` 0.89 左右。我们取其中表现更稳的一组，固定下来做最终训练：

```python
# 固定最终参数（数字来自上面的贝叶斯搜索结果）
params = {'application': 'binary',
          'learning_rate': 0.0001,
          'metric': 'auc',
          'num_leaves': 45,
          'feature_fraction': 0.8,
          'bagging_fraction': 1.0,
          'max_depth': 10,
          'lambda_l1': 4.638,
          'lambda_l2': 0.0,
          'min_split_gain': 0.042,
          'min_child_weight': 5.0}

train = lgb.Dataset(train_add[col], label=train_set['signal'])

# 先跑 5 折交叉验证，看看 AUC 随树数量的变化
cv = lgb.cv(params, train, nfold=5, seed=42,
            stratified=True, metrics=['auc'])
# 取平均 AUC 最高那一轮对应的树数量
best_niter = int(np.argmax(cv['auc-mean']))
print('Best CV AUC:', cv['auc-mean'][best_niter])
```

这一个配置下我们得到的交叉验证 AUC 约 **0.9133**。注意这里的 AUC 只是「排序能力」的本地估计，排行榜真正看的是加权截断 AUC，两者不能直接换算。AUC 的意思是：随机抽一个信号、再随机抽一个背景，模型把信号排在背景前面的概率。它适合在没有测试集答案时比较模型，但它并不关心模型在什么阈值上截断，也不直接反映最终能保留多少稀有信号。因此我们不会因为 AUC 高就跳过 KS 和 CvM 检验。

此外，贝叶斯搜索中的 `feature_fraction` 和 `bagging_fraction` 是两种随机抽样：前者让每棵树只看一部分列，后者让每棵树只看一部分样本。这样做能减少树与树之间的相似性，也能降低过拟合风险；`lambda_l1`、`lambda_l2` 则像给模型加「复杂度罚分」，防止它为了记住训练集而建立过于曲折的决策边界。：**UGBC（Uniform Gradient Boosting Classifier）**，它来自高能物理常用的 `hep_ml` 库。它和普通梯度提升的区别在于损失函数里多了一项「平坦性约束」：要求模型在 `mass` 的各个区间里给出的平均预测尽量一致。这正是 CvM 检验想考察的东西，所以用它对冲 CvM 超标的风险：

```python
from hep_ml.gradientboosting import UGradientBoostingClassifier
from hep_ml.losses import BinFlatnessLossFunction

# 把 mass 也作为一列传进去，但只用于「平坦性」约束，不作为普通分裂特征
ugbc_features = col + ['lines', 'line3']

# BinFlatnessLossFunction 会把 mass 分成 15 个区间，
# 要求背景（uniform_label=0）在每个区间里的预测均值尽量相同
loss = BinFlatnessLossFunction(['mass'], n_bins=15,
                               uniform_label=0,
                               fl_coefficient=15, power=2)

ugbc = UGradientBoostingClassifier(
    loss=loss,
    n_estimators=250,     # 250 棵树
    max_depth=8,
    learning_rate=0.15,
    subsample=0.7,
    random_state=123,
)
ugbc.fit(train_add[col + ['mass']], train_set['signal'])
```

### 2.4 模型测试与指标报告

本地没有测试集标签，也没有直接算 AMS 的官方脚本，所以我们用两道「体检」来判断模型能不能安全提交。

```python
# 用训练好的 LightGBM 给两份对照数据打分
agreement_probs = clf.predict(check_agreement[col])
correlation_probs = clf.predict(check_correlation[col])

# KS 检验：真实数据（signal=0）和模拟数据（signal=1）的打分分布差多少
ks = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

# CvM 检验：把事件按 mass 排序，看不同质量区间里的打分分布是否一致
cvm = compute_cvm(correlation_probs, check_correlation['mass'])

print('KS metric:', ks, ks < 0.09)       # 门槛 0.09
print('CvM metric:', cvm, cvm < 0.002)   # 门槛 0.002
```

我们跑出来的是 **KS = 0.0405**、**CvM = 0.00181**，两项都通过了门槛，其中 KS 还有比较大的余量，CvM 则贴近上限，说明「不依赖质量」这件事确实更难做到。作为对照，UGBC 版本的 KS 在 0.0716 到 0.0873 之间，同样合格；它的价值在于额外提供了 250 棵树的预测，可以和 LightGBM 做平均来降低方差。

确认体检通过后，我们对测试集打分并按比赛格式输出：

```python
# id 已经在读取时作为索引，这里直接取出来做提交的第一列
result = pd.DataFrame({'id': test.index})
result['prediction'] = clf.predict(test_add[col])

# 检查行数和取值范围，避免格式错误白白丢分
assert len(result) == 855819
assert result['prediction'].between(0, 1).all()
result.to_csv('result.csv', index=False)
print(result.head())
```

最终提交文件的每一行就是「事件 id + 0 到 1 的预测概率」。如果要把两个模型合起来，只需把 `clf.predict` 和 `ugbc.predict_proba(...)[:, 1]` 的结果按权重平均，再重新跑一遍 KS/CvM 体检即可——**任何合并后的新预测都必须重新体检**，否则前面的努力会在排行榜上被判无效。

---

## 三、总结

本文中我们完成了 Flavours of Physics τ → 3μ 衰变比赛的完整流程。我们首先用 EDA 摸清了数据结构：训练集约 6.7 万行、测试集约 85 万行，信号是模拟数据、背景是真实数据，还额外提供了用于 KS 和 CvM 检验的两份对照集。其次我们做了有针对性的特征工程，把飞行距离与误差、三个子粒子的 IP/动量/隔离度组合成 17 个信噪比和极值特征，并删掉测试集里不存在的辅助列。接着我们用 LightGBM 配合贝叶斯优化调参，交叉验证 AUC 约 0.913，又用带平坦性损失的 UGBC 作为互补模型来压制质量相关性。最后我们按比赛规则跑 KS（0.0405 < 0.09）和 CvM（0.00181 < 0.002）两道检验，通过后才生成提交文件。

这个比赛最值得记住的一点是：**评价指标不只是「预测得准」，还包括「预测得讲道理」**。在一个从没观察到的物理现象上，模型如果只会背模拟数据的特征，就会在真实数据上失效；KS 和 CvM 就是把这种「作弊式拟合」挡在门外的两道闸。特征工程也应当从物理含义出发，而不是盲目堆列。
