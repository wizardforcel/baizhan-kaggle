# JOAI Competition 2026：用老鼠的脑活动预测拉杆位置

## 一、比赛简介

### 1.1 比赛背景

这场比赛来自 Kaggle 的 Playground 系列，是 JOAI 2026 活动配套的一道练习题。它的数据不是常见的房价、销量或图片，而是一段真实的神经科学实验记录：研究人员把老鼠放进实验箱里，先播放一个声音，老鼠如果在这个声音之后去拉一根拉杆，就能得到一滴水作为奖励。实验过程中，研究人员用「钙成像」的方法记录老鼠大脑的活动。

所谓钙成像，可以理解成给神经细胞「拍视频」。神经细胞一旦活跃，细胞内的钙离子浓度就会发生变化，这个变化会被转换成光的强弱，于是我们就得到了一串随时间变化的数字。比赛数据里，每个脑区都对应一列数字，`_l` 表示左脑，`_r` 表示右脑，一共 44 个脑区特征。于是问题就变成了：**看着这一串脑活动的数字，能不能反推出老鼠此刻把拉杆拉到了什么位置？**

数据被切成了一个个「样本」（`sample_id`），一个样本大致对应一次试验中的一小段连续时间。每行数据还带有 `mouse_id`（第几只老鼠）、`day_n`（实验第几天）、`time`（时间，单位秒）等辅助信息。我们要预测的目标列叫 `lever`，也就是拉杆的位置。测试集里没有 `lever` 这一列，我们需要自己预测出来并提交。

### 1.2 比赛目标

比赛是一个**回归任务**：对测试集里的每一行（每一个时间点），预测出对应的 `lever` 数值。训练集提供 `lever` 作为标准答案，测试集只给脑活动特征，需要我们输出一个和 `sample_submission.csv` 同样长度、同样 `id` 顺序的 CSV 文件，其中包含 `id` 和 `lever` 两列，而且不需要行索引。

换句话说，这是一个「序列到序列」的问题：输入是一段时间的脑活动矩阵（时间 × 44 个特征），输出是同样长度的一段拉杆位置序列（时间 × 1）。

### 1.3 评价标准

比赛的评分指标是**均方误差**（Mean Squared Error，简称 MSE）。它的定义是：

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(\text{预测值}_i - \text{真实值}_i)^2
$$

也就是把每一个时间点的「预测值减真实值」平方，再全部加起来求平均。MSE 越小，说明预测越准。平方的作用是让正负误差都变成正数，同时让偏差特别大的点受到更重的惩罚。为了判断模型到底有没有学到东西，我们还会做一个「基线」对比：如果什么都不预测，干脆把训练集 `lever` 的平均值填给每一个点，也能算出一个 MSE，模型必须比这个基线明显更低才有意义。

## 二、解题过程

我们首先把数据读进来，确认它的形状和每一列的含义；其次做探索性数据分析，弄清数据的长相；接着做数据清洗并把交叉验证的分组设计好；然后依次训练 LightGBM、加入滚动特征、训练 LSTM 神经网络，最后把几个模型的结果平均起来提交。

### 2.1 EDA（探索性数据分析）

我们先把训练集、测试集和提交样例读进来，看看它们的行数列数。三个文件的分工很直接：训练集带答案，测试集用来预测，提交样例告诉我们输出格式。

```python
from pathlib import Path          # 路径操作
import numpy as np                # 数值计算
import pandas as pd               # 表格数据处理
import matplotlib.pyplot as plt   # 画图

data_dir = Path('/kaggle/input/playground-joai-competition-2026')

# 分别读取训练集、测试集、提交样例
train = pd.read_csv(data_dir / 'train.csv')
test = pd.read_csv(data_dir / 'test.csv')
sample_sub = pd.read_csv(data_dir / 'sample_submission.csv')

print('训练集形状:', train.shape)   # (行数, 列数)
print('测试集形状:', test.shape)
print('提交样例形状:', sample_sub.shape)

# 看看前几行，注意 sample_id 切换时 id 末尾会从 0 重新开始
print(train.head(40))
```

读完以后我们确认了几件事。第一，训练集和测试集的比例大约是 **6:4**，训练数据稍微多一点。第二，两边都**没有缺失值**，省去了填补的麻烦。第三，训练集和测试集里出现的是**同一批老鼠**，实验一共做了 15 天，每一天的数据量大致相同。第四，`sample_id` 才是真正的「预测单位」，同一个 `sample_id` 里的行是一段连续的时间序列。

接着我们统计每个样本有多少个时间步，以及相邻两帧的时间间隔。

```python
# 每个样本包含多少个时间步
timesteps = train.groupby('sample_id').size()
print('每样本时间步 最小/最大/平均:',
      timesteps.min(), timesteps.max(), timesteps.mean())

# 计算相邻时间点的间隔，判断采样频率
diffs = []
for _, group in train.groupby('sample_id'):
    diffs.extend(np.diff(group['time'].values))
diffs = np.array(diffs)
print('平均帧间隔(秒):', diffs.mean(), '约等于', round(1 / diffs.mean()), 'fps')
```

结果告诉我们，每个样本大约有 20 到 40 个时间步，平均帧间隔约为 1/30 秒，也就是说数据是 **30fps** 录制的，一个样本大概只覆盖 0.7 到 1.3 秒。这个信息很关键：它说明单点信息量有限，必须把相邻时间点的上下文利用起来。

然后我们重点看目标列 `lever` 的分布。

```python
# lever 的基本统计量和最常出现的值
print(train['lever'].describe())
print(train['lever'].value_counts().head(10))

# 画直方图看分布
plt.hist(train['lever'], bins=100)
plt.xlabel('lever'); plt.ylabel('频数'); plt.title('lever 分布')
plt.show()
```

图上能看到两个「尖峰」：一个在 0 附近，另一个在 4 附近；而 `lever = 0.141372` 这个值出现得特别多，占了全体数据的很大一部分。结合实验场景可以理解，这正是老鼠**没有拉杆、处于静止状态**时的读数。这也提醒我们，模型如果能学会「多数时候不动、少数时候拉动」，就已经抓住了一大半规律。

我们还按老鼠分组看了 `lever` 的均值和标准差，以及每只老鼠落在「静止值」附近的比例。结果显示，不同老鼠之间差异很大：有的老鼠超过 70% 的时间都停在静止值，有的则几乎不停。这说明 `mouse_id` 本身是有信息量的，后面应该把它当作类别特征交给模型。

```python
# 按老鼠统计 lever 的均值和标准差
stats = train.groupby('mouse_id')['lever'].agg(['mean', 'std'])
print(stats.sort_values('mean'))

# 每只老鼠处于静止值(0.141372)附近的比例
special = 0.141372
ratio = train.groupby('mouse_id')['lever'].apply(
    lambda x: (np.abs(x - special) < 1e-4).mean() * 100)
print(ratio.sort_values(ascending=False))
```

再看 44 个脑活动特征。它们大多以 0 为中心、近似正态分布，右侧有一条稍微长一点的尾巴；不同老鼠的特征分布也略有偏移。我们把某一个样本的 `lever` 和几个运动相关脑区（`MOp_l`、`MOp_r`、`MOs_l`、`MOs_r`）画在同一条时间轴上，可以看到拉杆动作前后脑活动确实会变化，但单看某一个脑区和拉杆之间并没有特别直白的同步关系。最后我们算了每个特征和 `lever` 的相关系数，发现**没有一个特征特别突出**，但几乎每个特征都有微弱的关联。这暗示应该把 44 个特征整体交给模型，而不是只挑几个。

### 2.2 数据清洗

数据本身很干净，没有缺失值，所以这一节的重点不是「补洞」，而是三件事：**压缩内存**、**把顺序排对**、**设计不会泄漏的验证方式**。

时间序列数据行数很大，如果每列都用 64 位浮点数存，内存会很吃紧。我们写了一个函数，把整数列和浮点列尽量往更小的类型转换，只要不溢出就降级。

```python
def reduce_mem_usage(df):
    """把每一列转换成能容纳其取值的最小数据类型，节省内存"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                # 整数列：按取值范围选择 int8/16/32/64
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                # 浮点列：按取值范围选择 float16/32/64
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
```

第二件事是排序。做滚动特征、按样本切分时，必须保证同一 `sample_id` 内部是按 `time` 升序排列的，否则「上一帧」和「下一帧」就乱了。我们用 `sort_values(['sample_id', 'time'])` 统一排序。

第三件事最关键：**交叉验证要怎么切**。如果像普通表格数据那样随机按行切分，同一个 `sample_id` 的相邻帧会同时出现在训练集和验证集里，模型等于提前看到了答案，验证分数会虚高。正确做法是**按 `sample_id` 分组切分**，保证同一个样本的所有时间步只落在一边。这正好对应 `GroupKFold`。

```python
from sklearn.model_selection import GroupKFold

# 按 sample_id 分组，5 折交叉验证，避免同一段序列被拆到两边
gkf = GroupKFold(n_splits=5)
groups = train['sample_id']

# mouse_id 和 day_n 当作类别特征，交给 LightGBM 原生处理
for c in ['mouse_id', 'day_n']:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')
```

### 2.3 模型训练与超参数调优

我们依次尝试了三种思路：先用 LightGBM 做基础版，再给它加上滚动特征，最后用 LSTM 直接吃整段序列，最后把三者平均。

**（1）LightGBM 基础版。** LightGBM 是梯度提升决策树（GBDT）的代表实现。它的想法很朴素：一棵树预测得不准没关系，下一棵树专门去学「上一棵树的误差」，很多棵树加起来就越来越准。对于这种表格式数据，它训练快、不容易过拟合，是很好的第一选择。

```python
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# 去掉 id、sample_id、目标列，剩下的都当特征
drop_cols = ['id', 'sample_id', 'lever']
features = [c for c in train.columns if c not in drop_cols]

params = {
    'objective': 'regression',   # 回归任务
    'metric': 'l2',              # 用 MSE 作为监控指标
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,       # 学习率，越小越稳但需要更多轮
    'num_leaves': 31,            # 每棵树的叶子数，控制复杂度
    'random_state': 42,
    'verbosity': -1,
}

oof = np.zeros(len(train))       # 存放折外预测
test_pred = np.zeros(len(test))  # 存放测试集预测（多折平均）

for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(train, train['lever'], groups=groups)):
    X_tr, y_tr = train.iloc[tr_idx][features], train.iloc[tr_idx]['lever']
    X_va, y_va = train.iloc[va_idx][features], train.iloc[va_idx]['lever']

    # 先算一个「全填平均值」的基线分数，方便对比
    base = mean_squared_error(y_va, np.full(len(y_va), y_tr.mean()))

    dtrain = lgb.Dataset(X_tr, y_tr)
    dvalid = lgb.Dataset(X_va, y_va)
    model = lgb.train(
        params, dtrain, num_boost_round=1000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50, verbose=False)],  # 50 轮不提升就停
    )
    oof[va_idx] = model.predict(X_va)
    test_pred += model.predict(test[features]) / 5   # 5 折结果累加后取平均

    print(f'Fold {fold}: 基线 {base:.6f}, 模型 {model.best_score["valid_0"]["l2"]:.6f}')

print('整体折外 MSE:', mean_squared_error(train['lever'], oof))
```

在这个设置下，教程给出的 5 折平均分数约为 **1.160874（上下浮动约 0.038596）**。作为参照，第一个折的「全填平均值」基线是 1.785078，模型在该折上能降到 1.17436 左右，说明模型确实学到了模式，但还有空间。

**（2）加入滚动特征。** 既然数据是时间序列，我们就可以把「当前时刻前后一小段窗口」的统计量也做成特征。具体做法是对每个 `sample_id`、每个脑区，计算一个宽度为 5 的滑动窗口内的均值、标准差、最小值和最大值。44 个特征 × 4 种统计量 = **176 个新特征**。

```python
window_size = 5

# 按 sample_id 分组，对 44 个特征一次性做滚动统计
rolling = (
    train.sort_values(['sample_id', 'time'])
    .groupby('sample_id')[feature_cols]
    .rolling(window=window_size, center=True, min_periods=1)  # 居中窗口
    .agg(['mean', 'std', 'min', 'max'])
)

# 把多级列名压平，例如 MOp_l_mean_5
rolling.columns = [f'{col}_{stat}_{window_size}' for col, stat in rolling.columns]
rolling = rolling.reset_index(level=0, drop=True)  # 丢掉多出来的 sample_id 索引

# 和原来的 id、lever、sample_id 拼在一起
train_roll = pd.concat([train[['id', 'lever', 'sample_id']], rolling], axis=1)
```

这里有个小技巧值得说明：如果老老实实地写两层 `for` 循环、一个样本一个特征地算，代码清楚但很慢，大约要跑十分钟；而用 `groupby(...).rolling(...).agg(...)` 让 Pandas 在底层批量计算，只要三十秒左右。特征名里带上窗口大小，是为了将来换窗口大小时不冲突。

把这份新特征喂给同样的 `GroupKFold` 和 LightGBM，教程观察到每个折的分数都**稳定地改善了大约 0.05**。幅度不算大，但方向明确：时间上下文确实是有用的。

**（3）LSTM 神经网络。** 前两种方法本质上是「把每个时间点单独看」，滚动特征只是手动补一点上下文。LSTM 则不同，它天生就是处理序列的：输入一整段 `(时间, 44)` 的矩阵，输出一整段 `(时间, 1)` 的预测。它内部有「门」结构，可以决定记住什么、忘掉什么，因此能捕捉较长时间跨度上的依赖。

因为不同样本的序列长度不一样，我们需要先做**填充**（padding）：把短序列用 0 补到同一长度，再用一个「掩码」在算损失时把补出来的 0 排除掉，否则模型会被这些假数据带偏。

```python
import torch
import torch.nn as nn

class TimeSeriesNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        # LSTM 层：batch_first=True 表示输入形状是 (batch, 序列长度, 特征数)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 把每个时间步的隐藏状态映射成 1 个预测值
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, 序列长度, hidden_size)
        return self.fc(out)          # (batch, 序列长度, 1)

model = TimeSeriesNN(input_size=44, hidden_size=16, output_size=1)
```

训练部分用 PyTorch Lightning 来组织，把「模型结构、损失函数、优化器」写在一个类里，代码会清爽很多。关键点还是那个掩码：只在真实时间步上算 MSE。

```python
class TimeSeriesModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TimeSeriesNN(input_size=44, hidden_size=16, output_size=1)
        self.criterion = nn.MSELoss(reduction='none')  # 保留每个元素的损失

    def _masked_loss(self, x, y, lengths):
        batch_size, seq_len = x.shape[:2]
        # 生成掩码：位置序号 < 真实长度 的地方为 1，填充部分为 0
        mask = torch.arange(seq_len).expand(batch_size, seq_len)
        mask = (mask < lengths.unsqueeze(1)).unsqueeze(-1)
        loss = self.criterion(self.model(x), y)
        return (loss * mask.float()).sum() / mask.sum()  # 只对真实位置求平均

    def training_step(self, batch, batch_idx):
        x, y, lengths, ids = batch
        return self._masked_loss(x, y, lengths)

    def validation_step(self, batch, batch_idx):
        x, y, lengths, ids = batch
        loss = self._masked_loss(x, y, lengths)
        self.log('valid/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)  # 学习率是超参数
```

超参数方面，我们用了批量大小 32、隐藏层维度 16、学习率 0.01、训练 10 个 epoch，优化器是 Adam。教程也特别强调，这些「不会被优化器自动更新、需要人手动设定」的量就是超参数，包括学习率、隐藏层大小、层数、训练轮数等。它们直接决定模型能学多好，值得单独调。训练时用 `ModelCheckpoint` 只保存验证损失最低的那一版权重，最后用这份最好的权重去预测测试集。

**（4）集成。** 最后我们把「基础 LightGBM」「LightGBM + 滚动特征」「LSTM」三份测试集预测直接取平均。这种做法叫 bagging 式的简单平均，思路是：不同模型犯的错误不完全一样，平均之后各自的随机误差会互相抵消一部分，结果通常比任何单个模型都稳。

```python
sub_ensemble = sub_baseline.copy()
sub_ensemble['lever'] = (
    sub_baseline['lever'] + sub_rolling['lever'] + sub_nn['lever']
) / 3   # 三个模型简单平均
```

### 2.4 模型测试与指标报告

我们统一用「折外预测」（out-of-fold，OOF）来评估：每一折的验证集预测拼起来，覆盖整个训练集，再整体算一次 MSE。这样得到的分数比单看某一折更可靠，也不会因为某一折运气好而误判。

```python
# 用全部训练集的折外预测算整体 MSE
oof_mse = mean_squared_error(train['lever'], oof)
print('OOF MSE:', oof_mse)

# 生成提交文件：用测试集的 id，填入多折平均后的预测
submission = sample_sub.copy()
submission['lever'] = test_pred
submission.to_csv('submission.csv', index=False)  # index=False，不要行号
```

我们得到的主要结果如下：

- **基线**：把训练集 `lever` 均值填满所有点，第一折 MSE 约为 1.785078。
- **LightGBM 基础版**：5 折交叉验证平均 MSE 约为 **1.160874 ± 0.038596**，相比基线下降明显。
- **LightGBM + 滚动特征**：在完全相同的折划分下，每一折又稳定改善约 **0.05**。
- **LSTM**：10 个 epoch 内，验证集 MSE 从 1.785078 一路降到约 **1.238787**，与 GBDT 大致相当；教程指出如果继续增加 epoch，还有进一步下降的可能。
- **三模型平均**：作为最终提交方案，利用了不同模型的互补性。

从预测曲线看，模型已经能大致跟上拉杆位置的起伏，但在动作切换的瞬间仍然偏平滑，这也是这类时间序列回归常见的难点。测试集预测的顺序也要特别注意：LSTM 是按 `sample_id` 输出的，顺序和 `test.csv` 并不一致，必须按 `id` 做一次合并对齐，否则提交文件会整体错位。

## 三、总结

本文中我们围绕「从老鼠脑活动预测拉杆位置」这个回归任务，完整走了一遍流程。我们首先读取数据，确认了训练/测试约为 6:4、没有缺失值、训练与测试共享同一批老鼠、每个样本有 20 到 40 个 30fps 的时间步；其次做了探索性数据分析，发现 `lever` 在 0 和 4 附近有明显聚集、静止值 0.141372 占比很高、不同老鼠的行为习惯差异很大、单个脑区与目标的相关性都很弱；接着做了数据清洗，压缩了内存、按 `sample_id` 和 `time` 排序，并用 `GroupKFold` 按样本分组切分，避免相邻帧泄漏答案；然后依次训练了 LightGBM 基础模型、加入 176 个滚动统计特征的 LightGBM、以及直接处理整段序列的 LSTM，并把三者做了简单平均集成；最后用折外预测报告了指标，其中 LightGBM 基础版交叉验证 MSE 约为 1.160874，滚动特征带来每折约 0.05 的稳定提升，LSTM 在 10 个 epoch 内把验证损失从 1.785 降到约 1.239。这场比赛最值得记住的经验有两点：一是时间序列任务一定要按组切分验证集，否则分数会骗人；二是特征工程和模型结构要配合数据的「时序」本性，滚动窗口和循环神经网络都是在补足上下文信息。沿着这些方向继续加窗口统计量、延长训练轮数、尝试更多模型的组合，分数还有继续提升的余地。
