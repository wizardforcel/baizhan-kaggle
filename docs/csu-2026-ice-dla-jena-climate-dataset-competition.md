# CSU 2026 耶拿气候数据集竞赛（Jena Climate Dataset）

## 一、比赛简介

### 1.1 比赛背景

德国小城耶拿（Jena）有一个长期运行的气象观测站，它每隔 10 分钟记录一次当地的天气：气温、气压、湿度、风速、风向、降水量等等。这样连续记录多年，就形成了一条很长的时间序列。因为数据质量高、覆盖时间长，这份「耶拿气候数据集」在机器学习圈子里被广泛用来练习时间序列预测。

这个比赛的任务很聚焦：**根据过去若干个时刻的观测，预测当前时刻的气温**。气温是一串连续变化的数字，而且相邻时刻高度相关——十分钟前 20 度，现在大概率还是 20 度左右。这种「用历史预测现在」的问题，正好是循环神经网络（RNN）最擅长的场景。

### 1.2 比赛目标

比赛提供两个文件：

- `train_val.csv`：按时间顺序排列的历史观测，包含一列时间戳（datetime）、若干数值气象特征，以及目标列 `T (degC)`（摄氏度气温）；
- `test.csv`：每一行是一个待预测样本，列名形如 `feat [t-11]`、`feat [t-10]`……`feat [t-1]`，表示「这个特征在 11 步前、10 步前……1 步前的值」，另外还有一个样本编号列。

也就是说，训练数据是「一条连续的长序列」，测试数据是「很多个已经切好的窗口」。我们要为每个测试窗口输出一个预测值：窗口结束时那一刻的气温。

这里有个容易忽略的细节：**训练集和测试集的形状完全不同**。训练集是一行一个时刻，测试集是一行一个窗口。所以训练时我们要自己用滑动窗口把长序列切成样本，预测时则要把一行里的几十上百个滞后列重新排列成一个「时间 × 特征」的小矩阵。

### 1.3 评价标准

比赛用 **RMSE（均方根误差）** 衡量预测值与真实气温的差距，单位是摄氏度，**越低越好**。计算方式是：把每个样本的误差平方、求平均、再开根号。平方会放大大的错误，所以 RMSE 对「偶尔预测得离谱」特别敏感——这也是做气温预测时最需要防的问题。

材料里的基线除了 RMSE 还打印了 MAE（平均绝对误差），单位同样是摄氏度。MAE 对异常值没那么敏感，两个一起看，能判断模型是「普遍偏一点」还是「偶尔错得离谱」。

---

## 二、解题过程

我们的方案是一条标准的时序深度学习流水线：**先把长序列按时间切分并标准化，再用滑动窗口造样本，训练一个两层 LSTM，最后把测试集的滞后列重组成窗口做推理。** 下面分四步讲。

### 2.1 数据探索（EDA）

我们首先读入 `train_val.csv`，然后面对的第一个问题是：**列名不固定**。这类比赛的数据文件经常换列名，硬编码 `"Date Time"` 很容易在某次更新后直接报错。所以我们写了两个「找列」函数，按优先级猜测：

```python
import pandas as pd
import numpy as np

df = pd.read_csv(TRAINVAL_CSV, low_memory=False)

def find_datetime_col(df):
    # 优先找名字里同时包含 date 和 time 的列
    cands = [c for c in df.columns if "date" in c.lower() and "time" in c.lower()]
    # 再按常见写法逐个试
    for c in ["date time", "date_time", "datetime", "Date Time", "time"]:
        if c in df.columns:
            return c
    return cands[0] if cands else None

def find_target_col(df):
    # 气温列的常见写法都列出来，实在找不到就找名字里带 degc 的
    for c in ["T (degC)", "T(degC)", "T_degC", "T", "temp", "temperature"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "degc" in c.lower():
            return c
    return None

dt_col = find_datetime_col(df)
target_col = find_target_col(df)
print("时间列:", dt_col, " 目标列:", target_col)
```

找到之后，我们做了一次**数值列盘点**：除了时间列和目标列，剩下的数值列全部当特征。这样做的好处是不用手工挑特征，也不会因为漏掉某一列而浪费信息；坏处是可能混进一些没用的列，但对 LSTM 来说，多几个特征通常比少几个更安全。

接着我们检查时间列的连续性：转成 datetime、排序、看有没有解析失败的空值。气象数据在采集时偶尔会断档，如果时间戳乱序，后面按 80/20 切分就会把「未来」切进训练集，造成数据泄漏。

```python
df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
before = len(df)
df = df.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)
print("解析失败被丢掉的行数:", before - len(df))
print("时间范围:", df[dt_col].min(), "→", df[dt_col].max())
```

### 2.2 数据清洗

这份数据的清洗不复杂，但有三件事必须做对：**按时间排序切分、只用训练集拟合标准化器、目标值也要标准化**。

第一，切分不能随机。时间序列的相邻样本高度相关，随机切分会让验证集里混进「训练集的邻居」，分数虚高。我们按时间顺序取前 80% 做训练、后 20% 做验证：

```python
# 特征列 = 所有数值列里去掉目标列
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in num_cols if c != target_col]

n = len(df)
split_ix = int(n * 0.8)          # 前 80% 训练，后 20% 验证

X_train_raw = df.iloc[:split_ix][feature_cols].values.astype(np.float32)
y_train_raw = df.iloc[:split_ix][target_col].values.astype(np.float32)
X_val_raw   = df.iloc[split_ix:][feature_cols].values.astype(np.float32)
y_val_raw   = df.iloc[split_ix:][target_col].values.astype(np.float32)
```

第二，标准化。神经网络的输入最好都在 0 附近、量级接近，否则气压（约 1000）会完全盖住湿度（约 0.5）。关键是**标准化器只能在训练集上 fit，再 transform 验证集和测试集**——如果用全体数据 fit，验证集的均值方差就泄漏进训练过程了：

```python
from sklearn.preprocessing import StandardScaler

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train_raw)   # fit + transform
X_val   = x_scaler.transform(X_val_raw)         # 只 transform

# 目标值也要标准化：让 MSE 损失处在合理量级，收敛更稳
y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
y_val   = y_scaler.transform(y_val_raw.reshape(-1, 1)).reshape(-1)
```

第三，造滑动窗口。LSTM 要吃的输入是「长度为 L 的一段序列」，所以我们要把一维的时间序列切成一个个 `(L, 特征数)` 的小块，标签取窗口之后那一个时刻的值：

```python
def make_sequences(X_scaled, y_scaled, L):
    Xs, ys = [], []
    for i in range(L, len(X_scaled)):
        Xs.append(X_scaled[i-L:i])   # 过去 L 个时刻，形状 (L, F)
        ys.append(y_scaled[i])       # 紧接着的那个时刻，标量
    return np.stack(Xs).astype(np.float32), np.array(ys, dtype=np.float32)

X_tr, y_tr = make_sequences(X_train, y_train, SEQ_LEN)   # SEQ_LEN = 12
X_va, y_va = make_sequences(X_val,   y_val,   SEQ_LEN)
print("训练窗口数:", len(X_tr), " 验证窗口数:", len(X_va))
```

注意验证集也要单独切窗口，而不是从训练序列的尾部借 12 个点——虽然那样能多出 12 个样本，但会让验证集和训练集共享数据，分数不可信。

### 2.3 模型训练与超参数调优

我们用的模型是 **LSTM**（长短期记忆网络）。普通 RNN 在处理长序列时，早期的信息会被后面的计算「冲淡」，出现梯度消失；LSTM 内部有一套「门」结构（遗忘门、输入门、输出门），可以自己决定「哪些旧信息该保留、哪些新信息该写入」。对气温这种既有短期波动、又有日周期的数据，这种「选择性记忆」很合适。

模型结构很简单：两层 LSTM 提取时序特征，只取最后一个时刻的输出，接一个线性层映射成一个数：

```python
import torch
from torch import nn

class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,   # 每个时刻有多少个特征
            hidden_size=hidden,      # 隐藏状态维度
            num_layers=layers,       # 两层，第二层在第一层的输出上再提取一次
            batch_first=True,        # 输入形状是 (batch, 序列长, 特征数)
            dropout=dropout,         # 层间随机丢弃，防止过拟合
        )
        self.head = nn.Linear(hidden, 1)   # 把隐藏状态映射成一个温度值

    def forward(self, x):
        out, _ = self.lstm(x)      # out: (B, L, H)
        out = out[:, -1, :]        # 只取最后时刻，因为它"看过"整段历史
        return self.head(out).squeeze(-1)   # (B,)
```

训练配置是：批量 256、30 轮（epoch）、Adam 优化器、学习率 0.001、损失函数 MSE。这几个数字不是拍脑袋来的——批量 256 能充分利用 GPU，30 轮配合 0.001 的学习率在大多数气象数据上够收敛，再多就容易过拟合。

```python
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=256, shuffle=True)
val_loader   = DataLoader(SeqDataset(X_va, y_va), batch_size=512, shuffle=False)

model = LSTMRegressor(n_features=len(feature_cols)).to(device)
crit  = nn.MSELoss()
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
```

训练循环里我们做了一件重要的事：**每一轮都在验证集上算一次损失，只保存验证损失最低的那一版权重**。这叫「早停式保存」（checkpointing），能防止模型在训练集上越背越熟、在验证集上越来越差：

```python
best_val = float("inf")
for ep in range(1, 31):
    model.train()
    tr_sum, tr_n = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()               # 清空上一轮的梯度
        loss = crit(model(xb), yb)    # 预测值与真实值的均方误差
        loss.backward()               # 反向传播求梯度
        opt.step()                    # 更新参数
        tr_sum += loss.item() * len(xb); tr_n += len(xb)

    model.eval()
    va_sum, va_n = 0.0, 0
    with torch.no_grad():             # 验证不需要建计算图，省显存
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            va_sum += crit(model(xb), yb).item() * len(xb); va_n += len(xb)
    va_mse = va_sum / va_n

    if va_mse < best_val:             # 只留最好的一版
        best_val = va_mse
        torch.save({"model_state_dict": model.state_dict(),
                    "feature_cols": feature_cols,
                    "x_scaler": x_scaler, "y_scaler": y_scaler}, CKPT_PATH)
    print(f"Epoch {ep:02d} | train MSE {tr_sum/tr_n:.6f} | val MSE {va_mse:.6f}")
```

关于超参数，影响最大的是三个：序列长度（窗口太短看不到趋势，太长又容易过拟合）、隐藏层维度（64 是性价比比较高的起点）、学习率（1e-3 太小收敛慢，1e-2 又容易震荡）。由于训练一轮就要几十秒到几分钟，大规模网格搜索不现实，所以我们先在少量轮次上粗筛，再用选定的配置跑满 30 轮。

### 2.4 模型测试与指标报告

验证阶段，我们把模型输出反标准化回摄氏度，再算 RMSE 和 MAE：

```python
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

preds_s, truth_s = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        preds_s.append(model(xb.to(device)).cpu().numpy())
        truth_s.append(yb.numpy())

preds = y_scaler.inverse_transform(np.concatenate(preds_s).reshape(-1, 1)).reshape(-1)
truth = y_scaler.inverse_transform(np.concatenate(truth_s).reshape(-1, 1)).reshape(-1)

RMSE = float(np.sqrt(np.mean((preds - truth) ** 2)))   # 单位：摄氏度
MAE  = float(np.mean(np.abs(preds - truth)))
print(f"Validation RMSE: {RMSE:.4f} degC, MAE: {MAE:.4f} degC")
```

这段代码打印出来的就是模型在「没见过的未来 20% 数据」上的真实误差。我们还可以画一条「真实值 vs 预测值」的曲线，只看前 300 个点，肉眼检查模型有没有系统性偏移（比如整体偏低）。

测试集的推理是整份代码里最容易出错的地方，因为**测试集的列是「扁平」的滞后列，模型要的是「立体」的窗口**。对每一行，我们按 `[t-11]` 到 `[t-1]` 的顺序把特征一列列取出来，拼成 `(11, 特征数)` 的矩阵：

```python
id_col = next((c for c in ["ID", "Id", "id", "sample_id"] if c in df_test.columns), None)
if id_col is None:
    df_test.insert(0, "ID", np.arange(1, len(df_test) + 1))
    id_col = "ID"

lags_to_use = list(range(11, 0, -1))     # t-11, t-10, ..., t-1
feat_means = x_scaler.mean_.reshape(1, -1)   # 缺失值就用训练集均值补

pred_rows = []
for _, row in df_test.iterrows():
    seq = np.zeros((len(lags_to_use), len(feature_cols)), dtype=np.float32)
    for t_i, k in enumerate(lags_to_use):
        vec = np.array([float(row[f"[t-{k}] {f}"]) for f in feature_cols]).reshape(1, -1)
        vec = np.where(np.isnan(vec), feat_means, vec)      # 补缺失
        seq[t_i, :] = x_scaler.transform(vec).squeeze(0)    # 用训练集的缩放器

    with torch.no_grad():
        inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred_s = model(inp).cpu().numpy().squeeze()

    pred_degC = y_scaler.inverse_transform(np.array(pred_s).reshape(-1, 1))[0, 0]
    pred_rows.append({id_col: row[id_col], "pred_T_degC": pred_degC})

pd.DataFrame(pred_rows).to_csv(SUBMIT_PATH, index=False)
```

这里有一个我们在材料里发现的**小坑**：代码里 `SEQ_LEN = 12`，但 `lags_to_use` 只有 11 个值（`range(11, 0, -1)` 是 11、10……1，共 11 个）。也就是说训练时的窗口是 12 步，推理时喂进去的是 11 步。LSTM 对输入长度没有硬性要求，代码能跑通，但「训练看 12 步、预测看 11 步」在严格意义上是不一致的。正确做法是把 `SEQ_LEN` 改成 11，或者把滞后列补到 12 个。我们在这里把它指出来，是因为这种「差一位」的 bug 在时序比赛里非常常见，而且不报错，只会悄悄拉低分数。

最后一个检查：提交文件的行数必须等于测试集行数，列名必须是编号列加 `pred_T_degC`，不能有重复 ID。这三点确认无误，提交才算完成。

---

## 三、总结

本文中我们完成了 CSU 2026 耶拿气候数据集竞赛的完整流程。我们首先用「按名字猜列」的方式稳健地找到时间列和目标列，盘点所有数值特征并检查时间戳是否连续；其次按时间顺序 80/20 切分数据，只用训练集拟合标准化器，把目标值也做标准化，再用滑动窗口把长序列切成 `(12, 特征数)` 的样本；接着搭建两层 LSTM（隐藏维度 64、dropout 0.1），用 MSE 损失和 Adam 优化器训练 30 轮，并只保存验证损失最低的权重；最后把测试集的 `[t-11]`…`[t-1]` 滞后列重组成窗口，用训练集的缩放器标准化、缺失值用训练均值填补，预测后反标准化回摄氏度、按编号写进提交文件。

这个比赛提醒我们两件事。第一，**时间序列的切分必须按时间走**，随机切分会让模型偷看未来，分数好看但没用。第二，**训练和推理的输入形状一定要对齐**——材料里那个 12 步对 11 步的错位就是活生生的例子，它不会报错，只会让你的排行榜分数莫名其妙地低一截。做完模型之后，多花五分钟核对「训练吃什么、预测吃什么」，往往比再调一轮超参数更值得。
