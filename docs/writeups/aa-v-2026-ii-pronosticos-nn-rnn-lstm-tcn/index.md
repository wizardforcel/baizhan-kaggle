# 哥伦比亚登革热周病例预测（AA V 2026-II）

## 一、比赛简介

### 1.1 比赛背景

登革热是一种由蚊子传播的传染病，病例数会随着雨季、气温和蚊虫密度起起落落。哥伦比亚的卫生部门按「流行病学周」（ semana epidemiológica，一年分 52 周）统计每个省（departamento）报告的病例数。把这些数字按周排起来，就得到了一条时间序列：横轴是周，纵轴是当周新增病例。

这个比赛希望参赛者用**神经网络**来预测未来的病例数。比赛名字里那串缩写 `NN / RNN / LSTM / TCN` 就是四类常见的序列模型：普通前馈网络、循环网络、长短期记忆网络、时间卷积网络。主办方把这几条路线都列出来，意思是「你可以从最简单的开始，一路试到更擅长处理时序的结构」。

数据以 parquet 格式给出。parquet 是一种列式存储格式，比 csv 小、读得快，在数据量大的比赛里很常见。除了病例数，表里还带了降水（`precip_chirps_mm`）这类外生变量——降水多、积水多，蚊子就多，病例往往跟着涨，这是做登革热预测时最常被用到的辅助信息。

这个任务有几个难点：

- **序列不止一条。** 每个省都有自己的曲线，全国是一条、安蒂奥基亚省是另一条。不能把它们混在一起当成一条序列。
- **序列不长。** 一年只有 52 个点，能用来训练的历史年份有限。
- **爆发很突然。** 大部分周病例数很低，少数几周突然飙升，误差很容易被这些峰值主导。
- **特征里有「标识」和「特征」混在一起。** 省编号、年份、周次这些列如果直接喂给模型，模型会把它们当成有大小关系的数字，反而学歪。

### 1.2 比赛目标

训练集每一行是「某个省、某一年、某一周」的记录，主要列有：

- `anio`：年份；
- `semana`：一年中的第几周；
- `id_departamento`：省的编号；
- `departamento`：省名；
- `casos_dengue`：当周登革热病例数，也就是要预测的目标；
- `precip_chirps_mm`：当周降水量；
- `semana_inicio`、`semana_fin`：这一周的起止日期；
- `ID`：这条记录的编号。

测试集有同样的列，但没有 `casos_dengue`。我们要给测试集每一行填上一个预测值，提交文件里两列：`id` 和 `dengue`。注意训练集里目标叫 `casos_dengue`，提交文件里叫 `dengue`，名字不一样，写代码时要以提交样例为准。

### 1.3 评价标准

材料里没有直接写出评分公式。不过从代码里能确认两件事：一是训练时用的损失函数是**均方误差 MSE**（预测值和真实值之差的平方取平均），二是作者导入了 `mean_squared_error`、`mean_absolute_error`、`r2_score` 和 `sqrt` 这几个评估函数。由此可以推断，这是一个**回归任务**，排行榜用的是某种基于误差的指标，最常见的就是 RMSE（均方根误差），数值越小越好。

RMSE 的大白话解释：先把每个预测的误差平方、求平均，再开根号。它和病例数同量纲，所以可以理解成「平均一次猜错多少例」。因为误差先平方再平均，少数几个预测得特别离谱的周会把分数拉得很差，这也意味着**峰值预测不准，代价很大**。

---

## 二、解题过程

整场比赛我们走了一条「先跑通、再换模型」的路线：首先把数据整理成「每个省一条按时间排好的序列」，其次用 5 周滑动窗口把序列切成监督学习的样本，接着用一个简单的前馈网络（MLP）把流程跑通，然后用时间卷积网络（TCN）替换掉 MLP 来捕捉时序结构，最后把测试集接上历史窗口做预测并生成提交文件。

### 2.1 数据探索（EDA）

我们首先把训练集读进来，并把「年 + 周」拼成一个可以排序的时间标签：

```python
import numpy as np
import pandas as pd

# parquet 是列式存储，比 csv 小且读得快，比赛直接给了这个格式
df = pd.read_parquet(
    '/kaggle/input/competitions/aa-v-2026-ii-pronosticos-nn-rnn-lstm-tcn/train.parquet')
print(df.shape)

# 把"年 + 周"拼成 6 位字符串，例如 2024 年第 3 周 -> "202403"
# 只按 semana 排序会跨年错位（第 52 周后面接第 1 周），必须带上年份
df['fecha'] = df['anio'].astype(str) + df['semana'].astype(str).str.zfill(2)
df = df.sort_values(by=['fecha', 'id_departamento'])
df.head(2)
```

这里用 `str.zfill(2)` 把周次补成两位，是因为字符串排序是逐位比较的：如果不补零，`"202410"` 会排在 `"20249"` 前面，时间顺序就乱了。这个小细节在拼接时间键时很容易踩坑。

接着我们挑出安蒂奥基亚省（Antioquia）作为样本，把病例曲线和降水画在同一张图上。两者的数量级差得很远，所以必须用双坐标轴：

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

departamento = 'Antioquia'
# 先把字符串日期转成真正的时间类型，再按周汇总
serie = (df.loc[df['departamento'] == departamento]
           .assign(semana_inicio=lambda d: pd.to_datetime(d['semana_inicio']))
           .groupby('semana_inicio', as_index=False)
           .agg(casos_dengue=('casos_dengue', 'sum'),
                precip=('precip_chirps_mm', 'mean')))

fig = make_subplots(specs=[[{'secondary_y': True}]])
# 降水用柱子放右轴，病例用折线放左轴，避免一条线被压成平的
fig.add_trace(go.Bar(x=serie['semana_inicio'], y=serie['precip'],
                     name='降水 (mm)'), secondary_y=True)
fig.add_trace(go.Scatter(x=serie['semana_inicio'], y=serie['casos_dengue'],
                         name='病例', mode='lines'), secondary_y=False)
fig.update_yaxes(title_text='病例', secondary_y=False)
fig.update_yaxes(title_text='降水 (mm)', secondary_y=True, showgrid=False)
fig.show()
```

这张图要回答一个具体问题：病例高峰和降水高峰是否对得上。如果两者大体同步，降水就是一个有用的辅助特征；如果对不上，就说明降水只能当参考，不能当主信号。登革热这类疾病通常有明显的季节性，我们要先在数据里确认这一点，再决定要不要把「周次」的信息显式喂给模型。

然后我们把「年份 × 周次」做成一张热力图，直接看每年第几周容易爆发：

```python
import plotly.express as px

dep = df[df['departamento'] == 'Antioquia']
# 行是年份、列是周次，颜色深浅就是当周病例数
tabla = dep.pivot_table(index='anio', columns='semana',
                        values='casos_dengue', aggfunc='sum')
px.imshow(tabla, aspect='auto', color_continuous_scale='Reds',
          labels=dict(x='流行病学周', y='年份', color='病例'),
          title='病例按年份与周次的分布').show()
```

这张图要回答的是：爆发是否集中在固定的周次。如果红色深色带年年出现在相近的位置，就说明「第几周」本身携带季节性信息，而不是随机散布。**这种年复一年的周期性，正是序列模型能利用的东西**——模型看到前几周在涨，就能推断接下来几周还会继续涨。

我们还对比了 Antioquia、Valle Del Cauca、Cundinamarca、Santander、Risaralda 五个省最近六年的曲线。这一步是为了判断：各省的峰值是不是同一时间出现。如果整体走势相似，就可以共享一个模型、靠「省编号」区分；如果各省完全各走各的，就必须给每个省单独建模。

### 2.2 数据清洗

这份数据的清洗工作不重，核心是**把「标识列」和「特征列」分开**，并且严格按时间切分训练集和验证集：

```python
SEQUENCE_LENGTH = 5          # 用过去 5 周预测下一周
TARGET_COLUMN = 'casos_dengue'
GROUP_COLUMN = 'id_departamento'

# 年份、周次、省编号、ID、省名、起止日期都是"标识"而不是"特征"：
# 它们没有连续的大小含义，喂进网络只会让模型记住编号
features_cols = df.columns.difference([
    'anio', 'semana', TARGET_COLUMN, GROUP_COLUMN,
    'ID', 'fecha', 'departamento', 'semana_fin', 'semana_inicio'])

# 按年份切分：2024 年之前训练，2024 年及以后验证
# 时间序列不能随机切分，否则"用未来预测过去"，验证分数会虚高
train_df = df[df.anio < 2024].copy()
val_df = df[df.anio >= 2024].copy()

# 标准化只在训练集上 fit，再套用到验证集
# 如果先对全量数据 fit，验证集的均值方差信息就泄漏进训练过程了
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train_df[features_cols])
train_df[features_cols] = scaler.transform(train_df[features_cols])
val_df[features_cols] = scaler.transform(val_df[features_cols])
```

这里有两个决定值得说明。第一，**按年份切分而不是随机切分**。时序任务里如果随机打乱，模型训练时可能已经见过「未来」的样本，验证分数会好得离谱，上线就崩。第二，**标准化只用训练集拟合**。`StandardScaler` 会记住均值和标准差，如果拿全量数据去拟合，等于把验证集和测试集的分布信息提前告诉了模型，这也是一种泄漏。

此外我们检查了特征里是否有缺失值。降水这类气象数据有时会缺周，如果直接喂给网络会产生 NaN，训练会直接报错或者梯度变成 NaN，所以这一步必须先确认。

### 2.3 模型训练与超参数调优

我们首先把「多条序列」变成「多条监督样本」。做法是滑动窗口：对每个省，取连续 5 周的所有特征作为输入，第 6 周的病例数作为标签，然后窗口往后挪一周，重复下去。

```python
import torch
from torch.utils.data import Dataset

class DengueDataset(Dataset):
    """把一个 DataFrame 切成滑动窗口样本。"""
    def __init__(self, df, sequence_length, is_train=True):
        self.sequence_length = sequence_length
        self.is_train = is_train
        self.data = []

        # 按省分组，每个省是一条独立序列，绝不能跨省滑窗
        for _, group in df.groupby(GROUP_COLUMN):
            group = group.reset_index(drop=True)
            # 序列长度必须大于窗口长度，否则连一个样本都凑不出来
            if len(group) <= sequence_length:
                continue
            for i in range(len(group) - sequence_length):
                # flatten 把 5 周 × n 个特征压成一维，供 MLP 使用
                seq_x = group.loc[i:i + sequence_length - 1,
                                  features_cols].values.flatten()
                if is_train:
                    # 训练时标签是下一周的病例数
                    self.data.append(
                        (seq_x, group.loc[i + sequence_length, TARGET_COLUMN]))
                else:
                    # 推理时没有标签，用 ID 记住这一行是谁，方便拼提交文件
                    self.data.append(
                        (seq_x, group.loc[i + sequence_length, 'ID']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, label = self.data[idx]
        if self.is_train:
            return (torch.tensor(x, dtype=torch.float32),
                    torch.tensor(label, dtype=torch.float32))
        return torch.tensor(x, dtype=torch.float32), label
```

`is_train` 这个开关很关键：训练集返回 `(X, y)`，测试集返回 `(X, ID)`。同一套滑窗逻辑复用两次，推理时才能把预测值和原始行对上号。

第一个模型是最简单的前馈网络（MLP）：把 5×n 个数字摊平，过一层 64 维的隐藏层，再输出一个数。

```python
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),                 # 非线性激活，否则两层线性等于一层
            nn.Linear(hidden_size, 1),
            nn.ReLU()                  # 病例数不可能为负，用 ReLU 卡住下界
        )

    def forward(self, x):
        return self.fc(x).squeeze()    # squeeze 去掉多余的维度，变成 [batch]

# 输入维度 = 窗口长度 × 特征数
input_size = len(features_cols) * SEQUENCE_LENGTH
model = SimpleMLP(input_size).to(device)
criterion = nn.MSELoss()               # 回归任务的标准损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()          # 每轮必须先清空上一轮的梯度
        loss = criterion(model(x_batch), y_batch)
        loss.backward()                # 反向传播算梯度
        optimizer.step()               # 按梯度更新权重
```

MLP 能跑通，但它把 5 周的数据摊平成一个长向量，**丢掉了「先后顺序」**：第 1 周和第 5 周在它眼里只是两个位置不同的数字。为了显式建模时间，我们换成了 TCN（时间卷积网络）。TCN 的核心是「因果卷积」——第 t 个输出只能看到 t 及以前的数据，不能偷看未来：

```python
class CausalConv1d(nn.Module):
    """带膨胀的因果一维卷积：只在左边补零，保证不看未来。"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # 卷积核越大、膨胀越大，能看到的历史就越长
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              dilation=dilation, padding=self.padding)

    def forward(self, x):
        out = self.conv(x)
        # 右边多出来的 padding 要切掉，否则输出会比输入长
        return out[:, :, :-self.padding] if self.padding > 0 else out

class TCNBlock(nn.Module):
    """两层因果卷积 + 残差连接。"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.GroupNorm(1, out_channels),   # 归一化，训练更稳定
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 通道数变了就用 1x1 卷积对齐，残差才能相加
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        # 残差连接让梯度能直接回传，深层网络才训得动
        return nn.ReLU()(self.net(x) + self.residual(x))
```

TCN 的另一个关键设计是**膨胀率指数增长**：第 1 层膨胀 1，第 2 层膨胀 2，第 3 层膨胀 4，第 4 层膨胀 8。这样 4 层就能覆盖很长的时间跨度，而参数量不会爆炸。

```python
class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i      # 1, 2, 4, 8：感受野逐层翻倍
            in_ch = input_size if i == 0 else num_channels[i - 1]
            layers.append(TCNBlock(in_ch, num_channels[i],
                                   kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # 输入是 [batch, 周, 特征]，Conv1d 要的是 [batch, 通道, 周]
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]        # 只取最后一个时间步做预测
        return self.fc(out).squeeze(-1)
```

训练 TCN 时我们加了两个技巧：**梯度裁剪**和**学习率衰减**。前者防止某一批数据让梯度突然爆炸，后者在验证损失停滞时把学习率减半，让模型在最优解附近细调：

```python
NUM_CHANNELS = [32, 32, 64, 64]   # 4 层，每层的卷积核数量
N_EPOCHS, LR = 30, 1e-3

model_tcn = TCN(len(features_cols), NUM_CHANNELS,
                kernel_size=3, dropout=0.1).to(device)
optimizer = optim.Adam(model_tcn.parameters(), lr=LR)
# 验证损失 5 轮不降就把学习率乘 0.5，避免一直在最优点附近震荡
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5)

best_val_loss, best_state = float('inf'), None
for epoch in range(N_EPOCHS):
    model_tcn.train()
    for x_batch, y_batch in train_loader_tcn:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model_tcn(x_batch), y_batch)
        loss.backward()
        # 梯度裁剪：把所有梯度的范数压到 1.0 以内，防止训练炸掉
        nn.utils.clip_grad_norm_(model_tcn.parameters(), max_norm=1.0)
        optimizer.step()

    model_tcn.eval()               # 关掉 dropout，验证结果才稳定
    val_loss = 0.0
    with torch.no_grad():          # 验证不需要梯度，省显存也更快
        for x_batch, y_batch in val_loader_tcn:
            val_loss += criterion(model_tcn(x_batch.to(device)),
                                  y_batch.to(device)).item()
    avg_val = val_loss / len(val_loader_tcn)
    scheduler.step(avg_val)

    # 只保留验证损失最低的那一版权重，防止过拟合
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        best_state = {k: v.clone() for k, v in model_tcn.state_dict().items()}
```

训练过程中我们会把每个 epoch 的训练 MSE 和验证 MSE 打印出来对照。判断过拟合的方法很朴素：训练 MSE 一直降、验证 MSE 开始升，就是该停的信号。这次我们保留了验证损失最低的 checkpoint，而不是最后一轮的权重。

### 2.4 模型测试与指标报告

测试集有一个现实问题：**每条序列的开头几周没有足够的历史可以组成窗口**。我们的处理是把验证集里每个省的最后 5 行接到测试集前面，这样测试集的第一周也有「过去 5 周」可用。TCN 用的数据集类和前面几乎一样，唯一的区别是**不做 `flatten`**，保留 `[周, 特征]` 的二维形状，好让卷积在时间维度上滑动：

```python
import pandas as pd

df_test = pd.read_parquet(
    '/kaggle/input/competitions/aa-v-2026-ii-pronosticos-nn-rnn-lstm-tcn/test.parquet')
df_test = df_test.sort_values(by=[GROUP_COLUMN, 'anio', 'semana'])
# 注意：这里必须用训练时 fit 好的同一个 scaler，不能重新 fit
df_test[features_cols] = scaler.transform(df_test[features_cols])

# 每个省取验证集最后 5 行，拼到测试集前面当"历史"
historical = val_df.groupby(GROUP_COLUMN).tail(SEQUENCE_LENGTH)
df_test_extended = pd.concat([historical, df_test], ignore_index=True)
df_test_extended = (df_test_extended
                    .sort_values([GROUP_COLUMN, 'anio', 'semana'])
                    .reset_index(drop=True))

test_loader = DataLoader(
    DengueTCNDataset(df_test_extended, SEQUENCE_LENGTH, is_train=False),
    batch_size=64, shuffle=False)   # 推理时绝不能打乱顺序
```

这里有个容易忽略的细节：推理阶段用的 `scaler` 必须是训练阶段拟合的那个对象。如果对测试集重新 `fit`，标准化基准就变了，模型看到的数值分布和训练时不一致，预测会系统性偏移。

最后生成提交文件。TCN 的输出我们做了两个后处理：**把负数截成 0**（病例数不可能为负），**四舍五入成整数**（病例数是计数）：

```python
model_tcn.eval()
all_preds, all_ids = [], []
with torch.no_grad():
    for x_batch, id_batch in test_loader:
        preds = model_tcn(x_batch.to(device)).cpu().numpy()
        all_preds.extend(preds)
        all_ids.extend(id_batch)

submission = pd.DataFrame({'id': all_ids, 'dengue': all_preds})
# 网络输出是连续值，可能小于 0；病例数必须是自然数，所以截断再取整
submission['dengue'] = submission['dengue'].clip(lower=0).round().astype(int)
submission.to_csv('submission_tcn.csv', index=False)
print(submission.shape)
```

关于分数，材料里没有留下具体数字，所以我们不在这里编造。能报告的是训练过程本身：我们全程盯着验证集的 MSE，MLP 和 TCN 都用同一套按年份切分的数据、同一个 scaler、同一个窗口长度，这样两者的对比才是公平的。从设计上讲，TCN 更可能跟上峰值：MLP 把 5 周摊平成一个向量，对「连续几周在涨」这种顺序模式不敏感，容易把峰值削平；TCN 的因果卷积能沿着时间方向逐层累积信息，输出的起伏更贴近输入曲线的走势。不过这只是结构上的推断，材料里没有给出两种模型的分数对比，我们不下「TCN 一定更好」的结论。

---

## 三、总结

本文中我们完成了哥伦比亚登革热周病例预测的完整流程。我们首先把「年 + 周」拼成可排序的时间键，按省分组、按时间排序；其次用折线图、双轴降水图和「年份 × 周次」热力图看清了病例的季节性和爆发规律；接着把标识列剔除、只用训练集拟合标准化器、按年份而不是随机切分训练验证集；然后先用一个把 5 周窗口摊平的前馈网络跑通流程，再换成带因果卷积、膨胀卷积和残差连接的 TCN；最后把验证集的历史窗口接到测试集前面，用同一个 scaler 做推理，并对输出做非负截断和取整。

这个比赛给我们的最大启发是：**时序任务里「怎么切数据」比「换多复杂的模型」更重要**。按年份切分、只用训练集拟合标准化器、推理时不打乱顺序，这三条如果做错，验证分数会好看得虚假。而 TCN 相对 MLP 的优势，本质上是它保留并利用了「先后顺序」这一结构信息，而不是简单地堆了更多参数。
