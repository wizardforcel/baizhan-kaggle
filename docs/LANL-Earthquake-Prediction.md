# LANL 地震预测（LANL Earthquake Prediction）

## 一、比赛简介

### 1.1 比赛背景

这个比赛由美国洛斯阿拉莫斯国家实验室（LANL）在 2019 年发起，是当年比较经典的时间序列回归比赛之一。它研究的不是真实的大地震，而是**实验室里制造出来的「微型地震」**：把一块带裂缝的岩石样本放到机器里，对它逐渐施加剪切力，模拟地壳断层的受力过程。随着压力不断累积，岩石会在某个时刻突然断裂、释放能量，这个过程被仪器录成一条非常长的**声学信号（acoustic signal）**。整份训练数据就是一条连续的信号流，记录了「加压→断裂→重新加压」循环反复进行的过程。

这个比赛的难点很直白：原始训练数据有 **629,145,481 行（约 6.3 亿行）**，光是 `train.csv` 就有大约 10GB，普通电脑连一次性读进内存都做不到；更麻烦的是，测试集并不是普通的分割，而是从信号里**随机截出来的 150,000 个点**，彼此之间时间上不连续，所以不能指望模型靠「照葫芦画瓢」记住固定规律。

### 1.2 比赛目标

训练集中每一行是一条采样记录，共有两列：

- `acoustic_data`：声学信号的数值（`int16` 类型）；
- `time_to_failure`：从当前时刻到下一次岩石破坏（failure）的**剩余秒数**（`float64` 类型），也就是我们要预测的目标。

测试阶段会拿到 2,624 个文件，每个文件里是一个 150,000 长的信号片段、只有 `acoustic_data` 一列。我们要对**每个片段输出一个数**：这个片段对应的时刻距离下一次破坏还有多少秒。换句话说，模型读入一长串「岩石发出的声音」，输出一个倒计时。

### 1.3 评价标准

比赛使用 **MAE（Mean Absolute Error，平均绝对误差）** 作为评价指标，计算公式是预测值与真实值之差取绝对值后求平均：

```text
MAE = (1 / N) * Σ |真实值 - 预测值|
```

直观理解就是「平均每个样本预测差了多少秒」。MAE 越低越好，为 0 代表完全预测准确。之所以用 MAE 而不是 RMSE（均方误差），是因为 MAE 不像 RMSE 那样对个别特别大的误差耿耿于怀，更符合「整体到底偏了多少」的直觉。

另外有一个细节对策略影响很大：2,624 个测试片段里，只有 13%（341 个）被用来算公开榜（Public Leaderboard）分数，剩下 87% 留到比赛结束算私榜。也就是说，公开榜分数只是抽样结果，波动可能很大，所以当时很多队伍宁可做多模型集成来求稳定。

---

## 二、解题过程

整场比赛我们大致走了这几步：**先做 EDA 读懂信号和倒计时的关系，再解决「数据太大装不进内存」的难题，接着人工构造一批统计特征，最后用梯度提升树模型做验证和预测。**

### 2.1 数据探索（EDA）

第一步是读数据。因为文件有 6 亿多行、10GB 大，我们没法一口气读进来，所以用 `chunksize` 分块读取，并按测试集的长度把训练信号切成一段一段的。目标值取每个片段**最后一行**的 `time_to_failure`，因为测试集给定的是一个 150,000 长度的片段，我们也用同样长度来划分，才能保证训练和测试的「形状」一致：

```python
import numpy as np
import pandas as pd

total = 629_145_481          # 训练数据总行数
chunksize = 150_000          # 每个片段大小，和测试片段保持一致
segments = total // chunksize # 一共能切成多少段：4194 段

# 分块读取：把 10GB 的大文件按 15 万行一块地喂给程序，避免内存爆掉
chunks = pd.read_csv('../input/train.csv',
                     dtype={'acoustic_data': np.int16,
                            'time_to_failure': np.float64},
                     chunksize=chunksize)

# 每个片段我们只提取极少数统计量做快速观察，目标是每段的"最后时刻离破坏还有几秒"
train_feats = []
y = pd.DataFrame(index=range(segments), columns=['time_to_failure'])
for i, chunk in enumerate(chunks):
    ad = chunk['acoustic_data']
    y.loc[i, 'time_to_failure'] = chunk['time_to_failure'].values[-1]
    train_feats.append({'max': ad.max(), 'mean': ad.mean()})
X = pd.DataFrame(train_feats)
```

把每个片段的 `max`、`mean` 和倒计时画在同一张图上后，我们得出几个非常重要的观察：

1. **`mean` 几乎说明不了问题**：信号均值非常小（小于 6），画出来基本是平的，看不出趋势——因为岩石缓慢受力时信号很安静，只有临近断裂才剧烈波动。
2. **`max` 能看出明显的周期性**：每个「地震过程（session）」末尾，信号会突然冲到一个很高的值，然后猛跌回去——这就是一次破坏发生。整个训练序列里大约发生了 16 次破坏（不同 EDA 统计略有出入）。
3. **倒计时曲线是锯齿状向下**：破坏一发生，`time_to_failure` 立刻被重置成下一次破坏的剩余时间，重新开始倒数。

这些观察告诉我们：**一个片段离破坏越近，信号通常越「激动」**，这是我们构造特征的依据。数据探索还顺带回答了「数据里有多少次破坏」这类常识性问题，帮我们校准了对目标分布的直觉。

### 2.2 数据清洗

这一节看起来像「清洗」，其实核心是**把 6 亿行降成几千个训练样本**，顺便避免信息泄漏。

最直接的做法是像上面那样，把整个信号平均切成 **4,194 个互不重叠的 150k 片段**，每个片段变成一个样本。互不重叠意味着这些片段之间没有任何相同的信号点，训练和验证之间不容易「泄漏」。不过我们最终采用的思路更精细：从整条信号里分 6 段，每段内部随机抽起点的方式得到 24,000 个起始位置，再为每个起点取其后 150,000 个点拼成样本。这样做训练样本从 4 千多变成 2 万 4 千，模型学到的东西更多，而且可以开 6 个进程并行算特征：

```python
SIG_LEN = 150_000            # 每个样本的片段长度
NUM_THREADS = 6              # 用 6 个进程并行干活
NUM_SEG_PER_PROC = 4000      # 每个进程抽 4000 个片段

# 先按等长把原始信号切成 6 大块，方便多进程各自读取，不需要每次载入全部 10GB
def split_raw_data():
    df = pd.read_csv('train.csv')
    max_start_index = len(df.index) - SIG_LEN
    slice_len = int(max_start_index / NUM_THREADS)
    for i in range(NUM_THREADS):
        df0 = df.iloc[slice_len * i: (slice_len * (i + 1)) + SIG_LEN]
        df0.to_csv('raw_data_%d.csv' % i, index=False)

# 在每个大块内部随机抽 4000 个起始位置——分层的意思就是整条信号哪里都可能抽到
def build_rnd_idxs():
    rnd_idxs = np.zeros((NUM_THREADS, NUM_SEG_PER_PROC), dtype=np.int32)
    for i in range(NUM_THREADS):
        np.random.seed(5591 + i)
        rnd_idxs[i, :] = np.random.randint(0, 100_000_000, NUM_SEG_PER_PROC)
    return rnd_idxs
```

在数据清洗这个环节我们要特别注意**不要破坏时间顺序**：抽取片段时只能往前、不能跨越重大破坏点去取一段「看起来像测试片段」的信号，否则模型在验证集上会显得很厉害、上了真实的排行榜却露馅。这也是后来我们反复吃的教训之一。

### 2.3 模型训练与超参数调优

先造特征。生成了 24,000 个样本后，我们对每个 150k 片段做了一堆统计变换，最终得到约 **900 个特征**。特征大致分几类：

1. **时域统计**：均值、标准差、最大值、最小值、差分均值、首尾 1 万/5 万点的统计量等；
2. **趋势特征**：对片段做一元线性回归，取斜率，刻画信号「是在变大还是变小」；
3. **STA/LTA 特征**：短时均值/长时均值之比，这是地震学里的经典指标，用来突出信号突然变强的时刻；
4. **频域特征**：做 FFT 后按频带切块，统计每个频带幅值的分位数、相位等。

这些特征里，STA/LTA 和频带统计最有代表性，示例代码长这样：

```python
from scipy.signal import butter, lfilter

# STA/LTA：长窗口均值更新得慢，短窗口均值更新得快，
# 一旦信号突然增强，短窗口均值会立刻盖过长窗口均值，比值变大 → 捕捉"异常前兆"
def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)
    lta = sta.copy()
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    sta[:length_lta - 1] = 0
    dtiny = np.finfo(0.0).tiny          # 防止除以 0
    lta[lta < dtiny] = dtiny
    return sta / lta

# 把信号按频段切开：EDA 显示 2 万 Hz 以上的信号基本是噪声，
# 于是在 0~2 万 Hz 之间用带通滤波器切成 8 个频带，分别算统计量
def band_features(xc):
    bands = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]
    feats = {}
    for lo, hi in zip(bands[:-1], bands[1:]):
        b, a = butter(4, (lo / 75000, hi / 75000), btype='bandpass')
        seg = lfilter(b, a, xc)          # 用 4 阶巴特沃斯滤波器截出这个频带
        feats['mean_%d_%d' % (lo, hi)] = seg.mean()
        feats['std_%d_%d' % (lo, hi)] = seg.std()
        feats['max_%d_%d' % (lo, hi)] = seg.max()
    return feats
```

900 个特征里并不是个个都有用。原版方案里，我们**先用 Pearson 相关系数筛特征**：只保留与 `time_to_failure` 相关性显著（p 值 ≤ 0.05）的特征，把剩下的扔掉。这一步看着简单，实际带来不小提升——例如 LightGBM 模型从 1.439 提到 1.434，XGBoost 从 1.467 提到 1.440。

模型我们主要用 LightGBM 和 XGBoost 两种梯度提升树，配 8 折交叉验证，并把每一折预测在测试集上的结果平均——这样既用上了全部训练数据，又让单折的偶然波动互相抵消：

```python
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

params = {'num_leaves': 21,          # 叶子数控制树的复杂度，设小一点防止过拟合
          'learning_rate': 0.001,    # 学习率小、树多，是梯度提升树的常见搭配
          'feature_fraction': 0.91,  # 每棵树只看 91% 的特征，增加随机性
          'metric': 'mae',           # 直接对着比赛指标 MAE 优化
          'objective': 'regression',
          'verbosity': -1}

# 8 折交叉验证：每折训练一个模型，预测值取平均
predictions = np.zeros(len(test_X))
folds = KFold(n_splits=8, shuffle=True, random_state=42)
for trn_idx, val_idx in folds.split(train_X, train_y):
    model = lgb.LGBMRegressor(**params, n_estimators=80000, n_jobs=-1)
    model.fit(train_X.iloc[trn_idx], train_y.iloc[trn_idx],
              eval_set=[(train_X.iloc[val_idx], train_y.iloc[val_idx])],
              eval_metric='mae', early_stopping_rounds=200)  # 验证集 MAE 连续 200 轮不开就停
    predictions += model.predict(test_X, num_iteration=model.best_iteration_) / folds.n_splits
```

调参时要诚实地说：这个比赛里 **CV 分数和排行榜分数经常对不上**。比如用非重叠切片做验证时，XGBoost 的 CV 平均 MAE 高达 2.253，但它在公开榜上的分数却是 1.437；参考脚本（Preda 的 LightGBM）CV 是 2.082，公开榜却只有 1.556。原因是公开榜只有 341 个样点、抽样误差大。所以我们不敢盲目迷信 CV，只能把超参数往「更不容易过拟合」的方向调，比如把 `min_data_in_leaf` 调大、降低树深度。

### 2.4 模型测试与指标报告

单模型大约跑到公开榜 1.434（LightGBM 最好）和 1.437（XGBoost 最好）后，我们对多模型做了**简单平均集成**。三个差异较大的模型——两个训练方法不同的 LightGBM/XGBoost，再加一个用遗传编程（gplearn）做特征合成的模型——预测值直接取平均。最终提交得到的公开榜分数是 **1.392**，在 3,200 多个参赛队里当时能进前 1%。

作为对照，我们还验证了两个有意思的极端做法：

- **「一个特征、不用机器学习」**的方案：对所有片段算信息熵，把熵裁剪到 `[2.37, 2.66]` 之间再做一次一元线性变换，就得到 MAE 2.112、私榜第 9 的成绩。这说明「离破坏越近、信号越混乱」这个规律到底有多强。
- **当年私榜第一名**的脚本：用 LightGBM 配 Gamma 回归目标做 5 次 bagging、每次 10~15 折交叉验证，最后还加了一个目标数据泄漏式的后处理（把预测均值乘 `6.31 / mean(predtest)`），私榜分数达到 2.252。

提交格式很简单，把 2,624 个片段的预测填进 `sample_submission.csv` 的 `time_to_failure` 列，一行对应一个 `seg_id` 即可：

```python
sub = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
sub['time_to_failure'] = predictions
sub.to_csv('submission.csv')
```

---

## 三、总结

本文中我们完成了 LANL 地震预测比赛的完整流程。我们首先用分块读取做 EDA，发现信号均值很小、但「越接近破坏信号越乱」的规律，并整理出大约 16 次破坏的时间结构。其次我们解决了大数据的难题：把 6.3 亿行的连续信号切分成与测试片段同样长度（150k）的样本，还在 6 段信号里分层随机抽取、凑出 24,000 个样本，并用多进程把特征计算压到一晚上跑完。接着我们构造了约 900 个时域、频域和 STA/LTA 特征，再用 Pearson 相关系数筛掉不显著的特征，最后用 LightGBM 和 XGBoost 做 8 折交叉验证的集成，公开榜分数做到 1.392、进入当时前 1%。

这个比赛让我们学到两点：**一是面对超大数据，先想清楚怎么把原始信号压缩成模型能吃的样本，决策质量远高于堆模型**；**二是当 CV 和排行榜不一致时，要敢于怀疑验证方法本身**——这个比赛里非重叠切片与非随机切片的 CV 差出两个量级，却都能在公开榜拿到差不多的分数，这种背离本身就是重要的数据特征。