# 味之物理（Flavours of Physics）Kernels-Only 竞赛

## 一、比赛简介

### 1.1 比赛背景

在大型强子对撞机 LHC 上，物理学家们最热衷的一件事就是寻找**新物理**的信号。标准模型预言，带电轻子味道的破坏是极其罕见的，所以如果实验观察到 τ 轻子衰变成三个 μ 子（记作 **τ → 3μ**），就可能是新粒子在作怪。这个比赛的数据来自 CERN 的模拟（monte carlo），每条样本是一次「衰变事件」的记录，我们手里只有探测器重建出来的运动学量——例如飞行距离、动量、各种角度——并不知道这次事件到底是真实的 τ→3μ 信号，还是普通的背景噪声。

因为信号极其稀罕、噪声极其普遍，再加上物理学家非常在意**不要把噪声误判成信号**，这个任务远比普通的二分类难。比赛的难度还体现在：它不光要求分类准，还要求模型在「机器学习里学到的规律」和「物理上已知的规律」之间保持一致，防止模型学到过拟合的假规律。

### 1.2 比赛目标

输入是一张特征表，每行代表一个候选事件，核心列包括：

- `LifeTime`、`FlightDistance`、`FlightDistanceError`：与 η 子（tau）飞行相关的寿命、距离及误差；
- `IP`、`IPSig`：碰撞点（primary vertex）的偏离量及其显著性；
- `VertexChi2`、`pt`、`dira` 等：顶点拟合质量、横动量、角度；
- 三个子粒子（p0/p1/p2）各自的动量、IP、隔离度等 `p*_*` 列；
- `mass`：重建出的不变量质量；
- `min_ANNmuon`：一个预先给出的判别分数，很多解法直接用它做数据预选；
- `signal`：标签，1 是信号（真正的 τ→3μ），0 是背景。

输出是每条测试事件属于信号的**概率**。数据量上，训练集有约 6.7 万行，测试集约 85 万行，另有 `check_agreement`（约 33 万行）与 `check_correlation`（约 5 千行）两份专门用来做物理一致性检验的数据。

### 1.3 评价标准

比赛先用 **加权 ROC-AUC**（把 ROC 曲线按不同灵敏度区间加权求和，`roc_auc_truncated`）衡量分类能力，CERN 用这套加权方法模拟真实实验里「信号占比低」的场景。但光 AUC 高还不够，提交还要过两道「物理安检」：

1. **KS 检验**（Kolmogorov-Smirnov）：在 `check_agreement` 上，模型对真实数据拟合背景的预测分布，要和模拟数据给出的分布一致，要求 KS 距离 **< 0.09**；
2. **CvM 检验**（Cramér-von Mises）：在 `check_correlation` 上，模型的预测不能和质量变量 `mass` 强相关（否则说明模型学到的是假的信号形状），要求 CvM **< 0.002**。

简而言之：**既要分得开信号与背景，又要保证预测的分布像「真实物理」**。这两道安检是解法里必须显式处理的难点。

---

## 二、解题过程

我们首先对数据做 EDA，其次构造了一批新的物理量特征并清洗冗余列，接着用一个 LightGBM 模型配合贝叶斯优化调参作为主模型，然后补充一个带「平面化损失」的 UGBC 模型做集成并逐项过安检，最后报告 AUC 与 KS/CvM 指标。

### 2.1 数据探索（EDA）

我们先读入五个文件，看各数据集的行数和特征数量：

```python
import pandas as pd

train_set = pd.read_csv('../input/training.csv', index_col='id')
test      = pd.read_csv('../input/test.csv', index_col='id')
check_agreement = pd.read_csv('../input/check_agreement.csv', index_col='id')
check_correlation = pd.read_csv('../input/check_correlation.csv', index_col='id')

print('训练集:', train_set.shape)          # (67553, 51)
print('测试集:', test.shape)              # (855819, 48)
print('一致性检验集:', check_agreement.shape)  # (331147, 50)
print('相关性检验集:', check_correlation.shape) # (5514, 46)
```

肉眼看到的关键事实有三个。第一，训练里信号和背景严重不均衡，背景占了绝大多数——这是粒子物理数据的常态。第二，`min_ANNmuon` 这个列几乎是现成的判别分数，直接用太浪费，很多队伍只拿它做数据预选（丢掉该分极低的样本），而把真正的判别能力留给后续特征。第三，训练集列数（51）比测试集（48）多 3 个，因为 `signal`、`mass` 是训练特有的，`min_ANNmuon` 在测试集里没有——建模时必须保持一致的特征集合。

我们还对单列分布做过分析，绝大多数物理量都有长尾，且事件质量 `mass` 的分布有明显峰结构。这提醒我们，后面做特征时不要只取原始值，要多做比值、乘积、取最大最小这类组合变换。

### 2.2 数据清洗

这道题的数据没有缺失值需要填充，真正的「清洗」是**特征压缩与重建**。我们看原始列里成组的特征太多：3 个子粒子各有一堆 `p0_*`、`p1_*`、`p2_*` 列，6 个隔离度 `isolationa~f`，还有 3 个 CDF 分类器输出。我们没必要把它们全部原样塞给模型，而是提炼成更有判别力的新特征：

```python
def add_features(df):
    # 飞行的「距离/误差」比值再做平方，本质是显著的飞行距离
    df['flight_dist_sig2'] = (df['FlightDistance']             / df['FlightDistanceError']) ** 2
    df['flight_dist_sig']  = df['FlightDistance'] / df['FlightDistanceError']
    # 三个子粒子拟合质量里最差的那个（取最大），最差的一环往往最能反映信号
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    # 飞行距离除以三子粒子动量之和、寿命乘平均 IP 等组合量
    df['NEW_FD_SUMP'] = df['FlightDistance'] / (df['p0_p'] + df['p1_p'] + df['p2_p'])
    df['NEW5_lt'] = df['LifeTime'] * (df['p0_IP'] + df['p1_IP'] + df['p2_IP']) / 3
    df['NEW_IP_dira'] = df['IP'] * df['dira']
    # 碰撞点偏离量之间的比值，源自前人的高分解法
    df['p0p2_ip_ratio'] = df['IP'] / df['IP_p0p2']
    df['p1p2_ip_ratio'] = df['IP'] / df['IP_p1p2']
    # 三条子径迹的 DCA、隔离度、IsoBDT 的最大/最小聚合
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    # 隔离度乘积与求和，刻画「三条径迹整体有多干净」
    df['NEW_iso_abc'] = df['isolationa'] * df['isolationb'] * df['isolationc']
    df['NEW_iso_def'] = df['isolationd'] * df['isolatione'] * df['isolationf']
    df['NEW_pN_IP']   = df['p0_IP'] + df['p1_IP'] + df['p2_IP']
    df['NEW_pN_p']    = df['p0_p'] + df['p1_p'] + df['p2_p']
    return df

train_set = add_features(train_set)
test = add_features(test)
```

这里有一个人在讲流程的人在别的方案里总结过的关键经验：**重构出来的 28 个「新特征 + 精选原始特征」组合，比直接把全部原始列喂给树模型效果更好**。因为树模型虽然不吃特征尺度，但面对 40 多个强相关的原始列，仍然容易被冗余信息干扰；而物理上合理的组合量本身就是「特征工程」最有力的形式。

### 2.3 模型训练与超参数调优

主模型选 **LightGBM**，它训练快、内存小，在 67 万级别的行数上没有压力。超参数我们用**贝叶斯优化**（`bayes_opt`）搜索，用 5 折交叉验证的平均 AUC 作为目标函数：

```python
from bayes_opt import BayesianOptimization
import lightgbm as lgb

def lgb_eval(num_leaves, feature_fraction, bagging_fraction,
             max_depth, lambda_l1, lambda_l2, min_child_weight, min_split_gain):
    # 每次迭代都用同一套通用参数，只换待搜索的超参数
    params = {'application': 'binary', 'num_iterations': 50000,
              'learning_rate': 0.0001, 'early_stopping_round': 100, 'metric': 'auc'}
    params['num_leaves'] = int(round(num_leaves))
    params['feature_fraction'] = max(min(feature_fraction, 1), 0)
    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    params['max_depth'] = int(round(max_depth))
    params['min_child_weight'] = min_child_weight
    params['min_split_gain'] = min_split_gain
    params['lambda_l1'], params['lambda_l2'] = max(lambda_l1, 0), max(lambda_l2, 0)
    # 5 折 CV，返回 5 折平均 AUC（贝叶斯优化的目标值）
    cv_result = lgb.cv(params, train_data, nfold=5, seed=42,
                       stratified=True, metrics=['auc'])
    return max(cv_result['auc-mean'])

# 每个超参数给一个区间，贝叶斯优化会在区间内智能取样
lgbBO = BayesianOptimization(
    lgb_eval, {
        'num_leaves': (20, 614),
        'feature_fraction': (0.1, 1.0),
        'bagging_fraction': (0.1, 1.0),
        'max_depth': (3, 12),
        'lambda_l1': (0, 5), 'lambda_l2': (0, 5),
        'min_child_weight': (0, 40), 'min_split_gain': (0, 1),
    })
lgbBO.maximize(init_points=20, n_iter=30)
```

贝叶斯优化跑下来的 5 折 CV AUC 在 **0.913～0.918** 之间。相比纯随机网格搜索，贝叶斯优化只用了很少的迭代就收敛到好参数，因为它会记住「哪些参数组合效果差」并绕开它们。

训练完成后，还要再验证两件事：一是取 CV 最好的迭代次数重新训练，二是对验证集单独算一次 AUC。材料里这一套跑出的验证 AUC 为 **0.916** 左右，说明模型在训练数据之外仍然有很强的判别力。

### 2.4 模型测试与指标报告

训练之外的关键一步是**过安检**。我们复用 CERN 提供的评估代码，把预测概率打进 `check_agreement` 和 `check_correlation`，分别算 KS 和 CvM：

```python
def compute_ks(data_prediction, mc_prediction, weights_data, weights_mc):
    """KS 距离：真实数据与模拟数据的预测分布差多远（越大越可疑）"""
    ...  # 统计上对两个样本做 ROC 变换后取最大差

ks  = compute_ks(agreement_probs[train_signal == 0], agreement_probs[train_signal == 1],
                 agreement_weights[train_signal == 0], agreement_weights[train_signal == 1])
print('KS metric :', ks, ks < 0.09)   # KS metric : 0.0405 True

cvm = compute_cvm(correlation_probs, check_correlation['mass'])
print('CVM metric :', cvm, cvm < 0.002)  # CvM metric : 0.0018 True
```

KP 世界的经验是，单独一个普通分类器常常在 KS 或 CvM 上翻车：模型在 `mass` 附近过度灵敏，就会把「假信号峰」当信号。为了稳定过安检，我们还引入了 `hep_ml` 库的 **UGBC（UniformGradientBoostingClassifier）**，它的损失函数 `BinFlatnessLossFunction` 会强制「信号率在不同 mass 区间保持平坦」，从根上抑制假的相关性：

```python
from hep_ml.gradientboosting import UGradientBoostingClassifier
from hep_ml.losses import BinFlatnessLossFunction

# n_bins=15 表示把质量均匀切成 15 段，损失强制各段判别结果一致
loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0,
                               fl_coefficient=15, power=2)
ugbc = UGradientBoostingClassifier(loss=loss, n_estimators=250,
                                   max_depth=8, learning_rate=0.15,
                                   subsample=0.7, random_state=123)
ugbc.fit(X_tr[ugbc_features + ['mass']], y_tr)
```

UGBC 的好处是它把「不要和 mass 相关」直接写进训练目标，而不是事后看运气。最终我们把这套 UGBC 的预测和主 LightGBM 的预测做加权平均（集成），既提高了 AUC，也让两关更稳定地通过。最后一步是把测试集的预测概率按比赛格式写成 `prediction` 列提交：

```python
result = pd.DataFrame({'id': test.index})
result['prediction'] = final_ensemble_predictions
result.to_csv('submission.csv', index=False)
```

---

## 三、总结

本文中我们完成了 Flavours of Physics Kernels-Only 竞赛的完整流程。我们首先通过 EDA 摸清了数据：约 6.7 万条训练事件、85 万条测试事件，以及 `check_agreement`/`check_correlation` 这两份专用于物理一致性检验的数据。其次我们做了轻量的数据清洗：不处理缺失值，而是把成组的原始物理量重构为 28 个更有判别力的组合特征，并剔除 `min_ANNmuon`、`mass` 等冗余或不应参与训练的列。接着我们用 LightGBM 配合贝叶斯优化在 5 折 CV 上把 AUC 搜到 0.913～0.918，再叠加一个带 mass 平坦化损失的 UGBC 模型做集成。然后我们在验证集上报告了约 0.916 的 AUC，并用 KS（<0.09）与 CvM（<0.002）两项检验确认模型没有学到假的物理相关性。最后把集成预测写成提交文件。

这段经历给我们最大的启发是：**粒子物理竞赛的难点不在分类器本身，而在「分布一致性」**。一个模型分得再开，只要它的预测分布和真实物理对不上，算法师的成绩就会归零。把「物理约束」直接编码进损失函数，比事后打补丁有效得多。