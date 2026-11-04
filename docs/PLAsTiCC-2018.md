# PLAsTiCC 2018 天文光变曲线分类

## 一、比赛简介

### 1.1 比赛背景

抬头看夜空，绝大多数星星的亮度几乎不变，但有一小撮天体是**会变亮变暗**的：超新星爆发会在几周内亮度暴涨再衰减，造父变星、天琴座 RR 型变星会周期性脉动，类星体则会不规则地闪烁。天文学家把这些"亮度随时间变化"的曲线叫做**光变曲线（light curve）**。

现代巡天望远镜（比如正在建设的 LSST，大口径综合巡天望远镜）每晚要给几百万个天体拍照片，逐个用光谱仪确认类型根本来不及，只能靠**测光**——也就是在各波段上反复测量亮度。PLAsTiCC 这个名字来自 "Photometric LSST Astronomical Time-Series Classification Challenge"，比赛用 LSST 的模拟数据，让大家仅凭测光得到的**多波段光变曲线**，判断每个天体属于哪一类。

这个任务的难点有三层：第一，**观测是稀疏且不均匀的**——望远镜不是均匀采样，有的天体只在几个晚上被拍到；第二，**不同类别在光变形状上长得很像**，比如不同类型的超新星；第三，**类别极度不平衡**，有的类型只有几十个样本。

### 1.2 比赛目标

比赛提供两份训练数据：

- `training_set.csv`：**1421705** 行光变观测，6 列——`object_id`（天体编号）、`mjd`（观测时刻，用修正儒略日表示）、`passband`（波段，u,g,r,i,z,y 六个波段分别编码为 0～5）、`flux`（测到的光流量，即亮度）、`flux_err`（这次测量的误差）、`detected`（这次观测是否显著探测到目标）；
- `training_set_metadata.csv`：**7848** 个天体，12 列——天球坐标 `ra`/`decl`、银道坐标 `gal_l`/`gal_b`、是否在深场 `ddf`、宿主星系的红移 `hostgal_specz`/`hostgal_photoz`/`hostgal_photoz_err`、距离模数 `distmod`、银河消光 `mwebv`，以及目标列 `target`。

`target` 在训练集里一共有 **14** 个类别：`[92, 88, 42, 90, 65, 16, 67, 95, 62, 15, 52, 6, 64, 53]`。测试集里还多出一个 **99** 类，它是"其他"类别——那些不属于上述 14 类的天体都会被打上 99，而训练集里**一个 99 类样本都没有**。这个设计是整个比赛最难的地方：模型必须学会对"从没见过的东西"也给出一个不自信的概率。

### 1.3 评价标准

比赛用 **加权多分类对数损失（Weighted Multi-class Log Loss）**，**越低越好**。它的算法可以拆成三步理解：

1. 对每个类别，取所有真实属于该类样本的 log loss 平均，得到这一类的损失；
2. 给每个类别乘一个**权重**——`15`、`64`、`99` 这几个稀有或没见过的类权重是 2，其余是 1；
3. 按权重求加权平均。

这个指标比单纯的准确率"狠"得多：交叉熵会**重罚"自信地答错"**。如果你说"我 99% 确定这是 52 类"，结果它是 90 类，损失会爆炸。所以模型必须学会"不确定的时候就分散下注"。

---

## 二、解题过程

我们的主线是：**先把一条条光变曲线压缩成每个天体一行的统计特征，再用 LightGBM 做多分类，并用比赛专用的加权对数损失来训练和验证。** 下面按四步走。

### 2.1 数据探索（EDA）

先把光变表和元数据读进来，看看体量和字段：

```python
import pandas as pd
import numpy as np

# 光变表：一行 = 某个天体在某个时刻、某个波段的一次观测
ts = pd.read_csv('../input/training_set.csv')
meta = pd.read_csv('../input/training_set_metadata.csv')

print(ts.shape)                      # (1421705, 6)：142 万次观测
print(ts.columns.tolist())
# ['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected']
print(meta.shape)                    # (7848, 12)：7848 个天体
print(meta['target'].nunique())      # 14 个类别
```

一个天体（`object_id`）对应**多行**观测——这正是"时间序列"的本质。接着看类别分布和缺失情况：

```python
# 每个类别的天体个数（柱状图），看看谁多谁少
print(meta['target'].value_counts().sort_values(ascending=False))
# 90 类最多，42、16、15 次之；53 类只有寥寥几个
# 训练集里没有 99 类，99 只存在于测试集

# 元数据里的缺失比例：对训练集和测试集要分别看
for col in ['hostgal_specz', 'hostgal_photoz', 'distmod']:
    tr = meta[col].isna().mean()
    print(f'{col}: 训练集缺失 {tr:.2%}')
# hostgal_specz  训练集 0.00%，测试集 96.55%
# distmod        训练集 29.63%，测试集 11.18%
```

我们得到几个关键认识：**第一**，类别极度不平衡——有的类只有几十个样本，所以训练时必须给稀有类**加权**；**第二**，光谱红移 `hostgal_specz` 在测试集里 **96.55% 缺失**，几乎没法直接用；**第三**，同一张表里既有"每个天体一行"的元数据，也有"每次观测一行"的光变数据，必须靠 `object_id` 把它们对齐。

### 2.2 数据清洗

严格说这份模拟数据没有"错值"，清洗的重点是**把不规则的时间序列整理成规整的特征表**。我们做了三件事：

1. **丢掉无用列**：天球坐标 `ra`/`decl`、银道坐标 `gal_l`/`gal_b`、深场标记 `ddf` 对分类帮助很小，反而容易让模型记住"训练集里某片天空有什么"，所以直接删；
2. **给负流量兜底**：天文学里流量为负是正常的（噪声压过了信号），但取对数会得到 NaN，所以对 `flux ≤ 0` 的情况统一填 0；
3. **按天体聚合**：把 142 万行观测按 `object_id` 分组，每个天体压缩成一行。这样训练集就从 142 万行变成 7848 行，和元数据一一对应。

```python
# 计算"星等"：magnitude = -2.5 * log10(flux)
# 负值流量不能取 log，所以先 clip 到 0 再 log，避免产生 NaN
ts['magnitude'] = -2.5 * np.log(ts['flux'].clip(lower=0) + 1e-6)

# 分波段拆列：每个波段单独一列流量，方便后续按波段统计
for p in range(6):
    ts[f'flux_{p}'] = ts.loc[ts.passband == p, 'flux']

# 删掉对分类没帮助的坐标列，只留红移、距离、消光等物理量
meta = meta.drop(columns=['ra', 'decl', 'gal_l', 'gal_b', 'ddf'])
```

这里有个容易踩的坑：`flux` 为负说明那一次测量里目标比背景暗，直接取 log 会变成 NaN；而如果我们把负流量整体删掉，又会丢失"这个天体很暗"的重要信息。正确做法是**先限制到 0 再取对数**，让负流量变成一个大星等值。

### 2.3 模型训练与超参数调优

这一节是核心，我们分三步：造特征、写指标、训练模型。

**第一步：把时间序列压成统计特征。** 对每个天体、每个波段，我们统计流量的最小值、最大值、均值、中位数、标准差、偏度、求和，以及"流量/误差的平方"——最后这个量很关键，它近似等于**信噪比的平方**，能反映这条曲线有多"清晰"：

```python
# 信噪比的平方：flux²/flux_err²，信号越强、误差越小，值越大
ts['flux_ratio_sq'] = (ts['flux'] / ts['flux_err']) ** 2
# 用信噪比平方给流量加权，等于"高置信度的观测说了算"
ts['flux_by_flux_ratio_sq'] = ts['flux'] * ts['flux_ratio_sq']

# 按天体聚合：每个 object_id 的每个统计量各算一列
aggs = {
    'flux':    ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_err':['min', 'std'],
    'detected':['mean'],
    'flux_ratio_sq': ['sum', 'skew'],
    'flux_by_flux_ratio_sq': ['sum', 'skew'],
    'magnitude': ['min', 'max', 'mean', 'std', 'skew'],
    'mjd': ['mean'],
}
agg = ts.groupby('object_id').agg(aggs)
# 把多级列名展平，例如 flux_min、flux_max
agg.columns = [f'{c}_{a}' for c, a in agg.columns]
```

除了这些基础统计量，我们还构造了一批**物理意义明确**的衍生特征：不同波段之间峰值的差（反映颜色/温度）、最亮和最暗星等之差除以时间间隔（反映变光快慢）、以及"加权平均流量 = Σ(流量×信噪比²) / Σ(信噪比²)"。这一套组合下来，每个天体最终得到 **136 个特征**。

**第二步：实现比赛指标。** 官方指标不能直接调库，我们按定义手写一个函数，供模型训练时做 early-stopping 的监控：

```python
def multi_weighted_logloss(y_true, y_preds):
    """PLAsTiCC 的加权多分类对数损失，越低越好"""
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6:1, 15:2, 16:1, 42:1, 52:1, 53:1, 62:1,
                    64:2, 65:1, 67:1, 88:1, 90:1, 92:1, 95:1}
    if len(np.unique(y_true)) > 14:          # 测试集含 99 类时补上权重
        classes.append(99); class_weight[99] = 2

    y_p = np.clip(y_preds, 1e-15, 1 - 1e-15)             # 防止 log(0)
    y_ohe = pd.get_dummies(y_true)                        # one-hot 编码
    y_log_ones = np.sum(y_ohe.values * np.log(y_p), axis=0)  # 每类的对数损失和
    nb_pos = y_ohe.sum(axis=0).values.astype(float)      # 每类样本数
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos                # 按类平均后再乘权重
    return -np.sum(y_w) / np.sum(class_arr)              # 加权平均
```

**第三步：训练 LightGBM。** 我们用 5 折**分层交叉验证**（`StratifiedKFold`），每折训练一个 LightGBM；为了让稀有类不被淹没，给每条样本的损失乘上"总样本数 / 该类样本数"的权重：

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

X = agg.join(meta.set_index('object_id'), how='inner')
y = X.pop('target')

# 每类的样本权重：类越稀有，单条样本的权重越大
w = y.value_counts()
sample_weight = y.map({c: len(y) / n for c, n in w.items()})

params = dict(objective='multiclass', num_class=14, metric='multi_logloss',
              learning_rate=0.0267, n_estimators=500, max_depth=7,
              num_leaves=7, colsample_bytree=0.5, subsample=0.75,
              min_child_weight=100.0, reg_alpha=0.1, reg_lambda=0.00023,
              n_jobs=-1, verbosity=-1)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
oof = np.zeros((len(X), 14))                 # 收集每折在验证集上的预测
for trn, val in folds.split(X, y):
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X.iloc[trn], y.iloc[trn],
            eval_set=[(X.iloc[val], y.iloc[val])],
            eval_metric=lambda yt, yp: ('wloss', multi_weighted_logloss(
                pd.Series(yt), yp.reshape(len(yt), 14, order='F')), False),
            early_stopping_rounds=50,
            sample_weight=sample_weight.iloc[trn])
    oof[val] = clf.predict_proba(X.iloc[val], num_iteration=clf.best_iteration_)

print('CV weighted logloss = %.5f' % multi_weighted_logloss(y, oof))
```

为什么把树做得这么"浅"（`max_depth=7`、`num_leaves=7`）？因为这 7848 个样本、136 个特征很容易过拟合，浅树配合较低的学习率和列采样，泛化更稳。需要说明的是，早停监控必须用比赛指标，而不是默认的 `multi_logloss`——否则模型会朝"普通准确率"优化，而比赛真正罚的是稀有类的错误。

### 2.4 模型测试与指标报告

5 折跑完后，每折的验证集损失分别是 0.55459、0.50493、0.55741、0.49636、0.53624，合并所有折的 out-of-fold 预测，整体 **CV 加权对数损失为 0.52974**。这个数字很符合直觉：样本充足的类别损失较低，而数据稀少的类会把整体拉高。对应的公开排行榜成绩约 **0.943**（榜单越低越好）。和随机乱猜相比，这个结果说明模型确实从光变形状里学到了东西。

测试集的数据量很大，一次性读入内存会让程序直接崩掉，所以预测时必须**分块**处理——我们按每块 40615 行逐块读入、逐块预测，再把结果拼起来。另外，**99 类必须单独构造**：既然训练集里没有"其他"类，我们就用"这个天体不属于任何已知类的概率"来代表它——把各类概率的补数乘起来：

```python
# 99 类的概率 = Π(1 - 每个已知类的概率)，再按均值缩放
preds_99 = np.ones(preds.shape[0])
for i in range(preds.shape[1]):
    preds_99 *= (1 - preds[:, i])
# 0.14 是社区调出来的缩放系数，让 99 类概率量级和别的类匹配
preds_99 = 0.14 * preds_99 / np.mean(preds_99)

sub = pd.DataFrame(preds, columns=[f'class_{c}' for c in classes])
sub['object_id'] = test_ids
sub['class_99'] = preds_99
sub.to_csv('submission.csv', index=False)
```

这一步是整个比赛"没有训练数据的类别"的标准解法：不预测它，而是从"其他类都不像"反推它。

---

## 三、总结

本文中我们完成了 PLAsTiCC 2018 天文光变曲线分类的完整流程。我们首先做 EDA，弄清 142 万次观测如何按天体聚合成 7848 个样本、14 个类别严重不平衡、测试集还有训练集里从没出现过的 99 类；其次做清洗与表征，删掉无用的坐标列、把负流量安全地转成星等、按 `object_id` 把时间序列压缩成一张特征表；接着手工构造 136 个统计与物理特征，手写加权多分类对数损失，用 5 折分层交叉验证训练浅层 LightGBM，并对稀有类做样本加权；最后得到 CV 损失 0.52974、公开榜约 0.943，并用"各类概率补数相乘"的办法构造 99 类的预测。材料中另一条很有代表性的路线是第三名方案的 1D CNN：把光变曲线插值成 128 个时间点 × 18 个通道，用 3 层卷积加全局最大池化，再接一个小 MLP 融合元数据；我们在总结里提一句，是为了说明同一份数据既可以用"手工特征 + 树模型"也能用"端到端深度学习"。

这个比赛给我们的最大启发是：**当类别不平衡且存在"训练集里没有的类"时，评价函数本身就是模型设计的一部分**。只优化普通准确率会忽略稀有类，只预测已知类会在 99 类上吃大亏；把比赛指标原样实现出来、给稀有类加权、用补数构造未知类，这三件事做对了，比换更复杂的网络更有价值。