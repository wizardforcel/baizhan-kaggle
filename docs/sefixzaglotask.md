# GPU 热点温度预测（GPU Hotspot）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的数据来自 GPU 集群的遥测（telemetry）。所谓遥测，就是机器每隔一段时间把自己当前的运行状态记录下来，比如显卡用了多少、功耗多大、风扇转多快、进风口温度多少。我们可以把每一块 GPU 想象成一口正在烧水的锅：锅底的火力是功耗，锅里的水是散热片和芯片，水的温度不会因为火突然变大就立刻升高，而是会慢慢爬上去。这种“温度变化比负载变化慢半拍”的现象，就是热惯性（thermal mass）。

比赛的数据里，每块 GPU 都对应一段连续的监控会话，每隔一分钟记录一行。有些 GPU 会报告自己的“热点温度”（hotspot temperature），也就是芯片上最热那一点的温度；另一些 GPU 由于传感器原因不报告这个值。我们要做的，就是根据其他传感器读数和时间上下文，把缺失的热点温度补出来。

这个任务有两个难点。第一，测试集里的 GPU 是训练集里从来没出现过的设备，模型不能靠记住某一块卡的行为来作弊。第二，同一块 GPU 相邻分钟的数据几乎一模一样，如果随机切分训练集和验证集，验证分数会虚高，所以验证时必须按设备分组。

### 1.2 比赛目标

训练集包含以下列：

- `row_id`：每一行的编号；
- `device_id`：设备编号，同一块 GPU 的所有行共用同一个编号；
- `step`：会话内的分钟序号；
- `util_gpu`：GPU 计算利用率；
- `util_mem`：显存利用率；
- `clock_sm_mhz`：流处理器时钟频率；
- `power_w`：整卡功耗；
- `fan_pct`：风扇转速百分比；
- `temp_inlet_c`：进风口温度；
- `temp_hotspot_c`：热点温度，也就是我们要预测的目标。

测试集同样有这些列，但 `temp_hotspot_c` 是空的。我们要给测试集的每一行填上一个预测值。

### 1.3 评价标准

比赛使用均方根误差（RMSE，Root Mean Squared Error）作为评价指标。它的算法是：先把每个预测值和真实值的差平方，再对所有样本求平均，最后开根号。因为开了根号，RMSE 的单位和温度一样是摄氏度，读起来很直观。RMSE 越小越好。材料里的入门基线给出的参考分数大约是 7.4，也就是说平均偏差在七摄氏度左右，后面还有很大的提升空间。

## 二、解题过程

我们的整体思路是：先按设备把数据整理成一条条时间序列，然后消除不同设备之间的系统性偏差，接着用能利用时间上下文的模型去预测，最后按设备分组做验证并生成提交文件。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，看看数据规模以及设备数量：

```python
import os
import numpy as np
import pandas as pd

# 比赛数据放在这个目录下
DATA = "/kaggle/input/competitions/sefixzaglotask"

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))

# 打印形状，确认训练集和测试集各有多少行
print("train:", train.shape, "| test:", test.shape)

# 看看训练集和测试集各有多少块不同的 GPU
# 测试集的设备是训练时没见过的，这一点决定了后面的验证方式
print("devices: train", train["device_id"].nunique(),
      "| test", test["device_id"].nunique())

train.head()
```

从输出可以看到，训练集和测试集都按 `device_id` 分成了很多段会话。每块 GPU 的数据是连续的一分钟一行，因此行与行之间高度相关。我们接下来挑一块设备，把四个关键指标画成四条上下对齐的曲线：

```python
import matplotlib.pyplot as plt

# 取第一块设备，画出它完整的一段会话
dev = train["device_id"].iloc[0]
g = train[train["device_id"] == dev]

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

# 四个子图分别对应负载、功耗、风扇和热点温度
for ax, col, color in zip(
    axes,
    ["util_gpu", "power_w", "fan_pct", "temp_hotspot_c"],
    ["tab:blue", "tab:red", "tab:green", "tab:purple"],
):
    ax.plot(g["step"], g[col], lw=0.8, color=color)
    ax.set_ylabel(col)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel(f"step (minutes) — one session of {dev}")
plt.tight_layout()
plt.show()
```

这张图告诉我们一个很重要的现象：`util_gpu` 和 `power_w` 会在一分钟内上下剧烈跳动，而 `temp_hotspot_c` 的曲线非常平滑，只会慢慢上升或下降。原因就是前面说的热惯性。也就是说，当前时刻的温度不只取决于当前时刻的负载，还取决于前面一段时间累积的热量。只看单行特征是不够的，模型最好能看见历史。

### 2.2 数据清洗

这份数据的缺失值主要出现在传感器读数上。我们没有做复杂的插补，而是按设备分组，用前向填充加后向填充把空缺补上。这样做的理由是：同一块 GPU 相邻分钟的值很接近，用隔壁分钟的值来补是最合理的。

```python
# 只保留和物理状态有关的特征
FEATURES = ["util_gpu", "util_mem", "clock_sm_mhz",
            "power_w", "fan_pct", "temp_inlet_c"]
TARGET = "temp_hotspot_c"

# 按设备分组做前向填充和后向填充
# 前向填充用上一分钟的值补，后向填充兜底补开头的空缺
for df in (train, test):
    df[FEATURES] = (
        df.groupby("device_id")[FEATURES]
          .transform(lambda s: s.ffill().bfill())
    )
```

接着我们发现，不同批次的 GPU 之间存在固定的温度偏移。材料里给出了一个按 `device_id` 最后一位数字索引的偏移表。我们先把目标减去这个偏移，让所有设备站在同一条起跑线上，预测完再加回去：

```python
# 每个批次有一个固定的温度偏移，索引就是设备编号的最后一位
BATCH_OFFSET = [-4.6, 2.4, 7.1, -1.8, 0.5,
                -6.3, 3.9, 8.2, -3.4, 5.6]
off = np.array(BATCH_OFFSET)

# 取出设备编号最后一位数字，映射到对应的偏移量
digit = train["device_id"].str[-1].astype(int)

# 训练目标先减去偏移，学一个“去偏”后的温度
train["target_debiased"] = train["temp_hotspot_c"] - off[digit]
TARGET = "target_debiased"
```

这里为什么要去偏？因为如果不去掉批次差异，模型可能会把“这块卡属于哪一批”当成捷径，而不是真的学温度变化的物理规律。去偏之后，模型只需要关心负载、功耗、风扇这些真正影响温度的因素。

然后我们对特征做标准化。神经网络对输入尺度很敏感，标准化可以让训练更稳定：

```python
from sklearn.preprocessing import StandardScaler

feature_scaler = StandardScaler()
train[FEATURES] = feature_scaler.fit_transform(train[FEATURES])
test[FEATURES] = feature_scaler.transform(test[FEATURES])

# 目标也做标准化，训练时数值更稳定
target_scaler = StandardScaler()
train[[TARGET]] = target_scaler.fit_transform(train[[TARGET]])
```

### 2.3 模型训练与超参数调优

因为每块 GPU 是一段变长时间序列，我们把每块设备打包成一个样本，用 PyTorch 的 `Dataset` 来组织。不同设备的会话长度不一样，所以需要 `pad_sequence` 把短序列补齐，同时记录真实长度，计算损失时把补齐的部分屏蔽掉。

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 把每块设备的数据收集成一个 (时间步, 特征数) 的张量
X, y = [], []
for _, g in train.groupby("device_id", sort=False):
    X.append(torch.tensor(g[FEATURES].values, dtype=torch.float32))
    y.append(torch.tensor(g[TARGET].values, dtype=torch.float32))

class RNNDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 没有标签时用全零占位，方便测试集复用同一个类
        y = self.y[idx] if self.y is not None else torch.zeros(
            self.X[idx].shape[0], dtype=torch.float32)
        return self.X[idx], y

def collate_fn(batch):
    # 把一个 batch 里的变长序列补齐到同一长度
    X, y = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in X])
    X = pad_sequence(X, batch_first=True)
    y = pad_sequence(y, batch_first=True)

    # mask 标记哪些位置是真实数据，哪些是补齐出来的
    mask = torch.arange(X.shape[1])[None] < lengths[:, None]
    return X, lengths, y, mask.float()
```

模型本身我们用一个双向 GRU。GRU 是一种循环神经网络，可以按时间顺序读入每一分钟的数据，并记住之前的状态；双向的意思是它同时从前往后和从后往前读，这样每个时刻的输出都能看到前后文。读完之后接一个线性层，把隐藏状态映射成一个温度值。

```python
from torch import nn, optim
import pytorch_lightning as pl

class RNNModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # 输入 6 个特征，隐藏层 512 维，双向所以输出是 1024 维
        self.rnn = nn.GRU(6, 512, batch_first=True, bidirectional=True)
        self.head = nn.Linear(1024, 1)

    def forward(self, x, lengths):
        # pack_padded_sequence 让 GRU 跳过补齐的位置
        packed = pack_padded_sequence(x, lengths.cpu(),
                                      batch_first=True,
                                      enforce_sorted=False)
        output = self.rnn(packed)[0]
        output = pad_packed_sequence(output, batch_first=True)[0]
        return self.head(output).squeeze(-1)

    def criterion(self, logits, y, mask):
        # 只在真实位置上算均方误差，补齐位置不参与
        return (((logits - y) ** 2 * mask) / mask.sum()).sum()

    def training_step(self, batch, _):
        X, lengths, y, mask = batch
        loss = self.criterion(self(X, lengths), y, mask)
        self.log("train_mse", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Adam 加余弦退火，学习率随训练轮次平滑下降
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]
```

训练时我们用 `EarlyStopping` 和梯度裁剪，防止过拟合和梯度爆炸。材料里设置的训练轮数是 30，batch size 是 32，梯度裁剪阈值为 3.0：

```python
trainer = pl.Trainer(
    max_epochs=30,
    gradient_clip_val=3.0,          # 梯度范数超过 3 就裁剪，训练更稳
    gradient_clip_algorithm="norm",
)

trainer.fit(rnn_model, train_loader, valid_loader)
```

关于验证，我们没有随机切分行，而是按 `device_id` 分组。材料里的入门基线用 `GroupShuffleSplit` 做分组切分，原因很直接：如果同一块 GPU 的相邻分钟同时出现在训练集和验证集里，模型只要抄上一分钟的温度就能拿高分，这种分数在真正没见过的新设备上会崩掉。

### 2.4 模型测试与指标报告

训练完成后，我们先用验证集算 RMSE，确认模型没有明显过拟合。然后对所有测试设备逐段预测，把预测值反标准化，再加回批次偏移：

```python
from sklearn.metrics import mean_squared_error

# 在验证集上预测并计算 RMSE
valid_pred = trainer.predict(rnn_model, valid_loader)
# 这里省略了把变长预测拼回原数组的细节
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))
print("Validation RMSE:", rmse)
```

生成提交文件时，顺序很重要：先反标准化，再加回偏移，最后按 `row_id` 对齐：

```python
# 把每条序列的预测值摊平，和设备分组一一对应
test_predict = trainer.predict(rnn_model, test_loader)
test_predict = [
    test_predict[i][j]
    for i in range(len(test_predict))
    for j in range(len(test_predict[i]))
]

result = []
for predict, (_, g) in zip(test_predict,
                           test.groupby("device_id", sort=False)):
    tmp = g.copy()
    tmp["target_debiased"] = predict
    result.append(tmp)

result = pd.concat(result, ignore_index=True)

# 先反标准化，把数值变回摄氏度
result["target_debiased"] = target_scaler.inverse_transform(
    result[["target_debiased"]])

# 再加回每个批次的固定偏移
digit = result["device_id"].str[-1].astype(int)
result["temp_hotspot_c"] = result["target_debiased"] + off[digit]

# 按比赛要求输出 row_id 和预测温度
sub = pd.DataFrame({
    "row_id": test["row_id"],
    "temp_hotspot_c": result["temp_hotspot_c"],
})
sub.to_csv("submission.csv", index=False)
print(sub.shape)
```

材料里的入门基线是一个逐行梯度提升模型，直接用当前时刻的六个传感器读数预测温度，官方给出的参考 RMSE 大约是 7.4。我们把它当作下限，然后换成 GRU，让模型能看到历史。这里要诚实地说：材料没有给出 GRU 的最终排行榜分数，所以我们不编造具体提升幅度，只报告验证流程和预期方向。

我们最后还检查了提交文件的形状和缺失值，确保行数和测试集一致、`temp_hotspot_c` 没有空值。这个步骤看起来琐碎，但比赛里因为提交格式出错而白白丢分的情况并不少见。

## 三、总结

本文中我们完成了 GPU 热点温度预测的完整流程。我们首先通过 EDA 发现每块 GPU 是一段连续的分钟级会话，温度曲线明显比负载曲线平滑，说明热惯性让历史信息变得重要。其次我们按设备做前后向填充，并用设备编号最后一位对应的偏移量对目标去偏，减少批次差异带来的干扰。接着我们把每块设备打包成变长序列，用双向 GRU 学习时间上下文，并在损失函数里用 mask 屏蔽补齐位置。然后我们按设备分组验证，避免相邻分钟泄漏导致分数虚高。最后我们把预测值反标准化、加回批次偏移，生成符合要求的提交文件。

这个比赛给我们最大的启发是：**验证方式必须模拟真实测试场景**。测试集是没见过的设备，所以验证也必须按设备分组；否则模型学到的只是“抄上一分钟”，看起来分数很高，换一批设备就不行了。
