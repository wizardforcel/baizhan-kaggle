# Grasp and Lift EEG Detection

## 一、比赛简介

### 1.1 比赛背景

Grasp and Lift EEG Detection 是一个脑电信号（EEG）事件检测比赛。人在做「抓握并抬起物体」这个动作时，大脑会产生有规律的电位变化，头戴式电极阵列可以记录下来。比赛给出一段段 32 通道的 EEG 时序数据，要求逐时间点预测 6 个动作事件是否正在发生。

这 6 个事件是：

- `HandStart`：手开始移动
- `FirstDigitTouch`：手指第一次接触物体
- `BothStartLoadPhase`：双手开始承重
- `LiftOff`：物体离开桌面
- `Replace`：物体放回
- `BothReleased`：双手松开

动作按时间顺序发生，事件之间存在前后依赖。数据采样率 500 Hz，每条 `subjX_seriesY_data.csv` 有 32 列通道数据和一列 `id`，对应的 `_events.csv` 有 6 列 0/1 标签。

### 1.2 比赛目标

输入是一段 EEG 时序，输出是每个时间点上 6 个事件的概率（0~1）。训练集既有 `data` 也有 `events`，测试集只有 `data`，需要按行 id 和 6 个事件名生成提交文件。

```python
labels = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase',
          'LiftOff', 'Replace', 'BothReleased']
```

### 1.3 评价标准

比赛用 6 个事件各自的 ROC AUC 取平均作为评价指标。AUC（曲线下面积）衡量「正样本得分高于负样本」的程度，0.5 等于随机猜，1.0 是完美排序。因为事件是极度不平衡的（大部分时间没有动作），用准确率会「全猜 0」就得到 0.9 以上的假高分，所以必须用 AUC。

---

## 二、解题过程

我们首先做信号可视化，其次做降噪和标准化，接着用 1D 卷积网络训练，最后用验证集的 AUC 报告结果。

### 2.1 数据探索（EDA）

先解压并读取一条数据与对应事件，统计各事件为正的样本数：

```python
labels = ['HandStart','FirstDigitTouch','BothStartLoadPhase',
          'LiftOff','Replace','BothReleased']
# 训练文件形如 train/subj1_series1_data.csv 与 _events.csv
x, y = read_csv('train/subj1_series1_data.csv',
                'train/subj1_series1_events.csv')
print(x.shape, y.shape)       # (行数, 32) 与 (行数, 6)
```

画 6 个事件的正样本计数柱状图，可以看到 `HandStart`、`FirstDigitTouch` 等前期事件出现较多，后面的事件相对少，整体非常不平衡。再画一段原始信号与降噪后信号的对比，确认高频噪声明显：

```python
import pywt, matplotlib.pyplot as plt
signal = pd.read_csv('train/subj1_series1_data.csv').drop("id", axis=1)
filtered = wavelet_denoising(signal.values, wavelet='db2', level=3)
plt.plot(signal.iloc[:10000, 1], label='signal', alpha=0.5)
plt.plot(filtered[:10000, 1], label='reconstructed', color='k')
plt.title('Denoising with DWT')
plt.legend(); plt.show()
```

### 2.2 数据清洗

脑电图的清洗分三步。第一步做小波降噪：把信号分解到不同尺度，对小波系数做硬阈值处理再重建，去掉高频噪声而保留动作相关的低频成分：

```python
def wavelet_denoising(x, wavelet='db2', level=3):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard')
                 for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')
```

第二步做标准化：每个通道减去均值、除以标准差，让不同电极的量纲可比。第三步丢弃「长时间没有事件」的数据块。材料把时间轴按 1000 个点切块，块内平均事件率低于 0.01 就整块扔掉：

```python
chunk_size = 1000
threshold = 0.01
index = []
for i in range(len(gt)):
    for j in range(0, gt[i].shape[1], chunk_size):
        if np.mean(gt[i][:, j:j+chunk_size]) < threshold:
            continue                       # 无事件块不进训练
        index.extend([(i, k) for k in range(j, min(gt[i].shape[1], j+chunk_size))])
```

这一步丢掉了 8859 个块、约 0.5% 的数据，却显著减少了「全是 0」的样本，缓解类别不平衡。

### 2.3 模型训练与超参数调优

我们采用滑窗方式构造样本：每个样本是截止到当前时刻的 1024 个采样点（`in_len=2**10`）× 32 通道，标签是当前时刻的 6 个事件。训练时对标签做软截断，把小于 0.02 的值抬到 0.02，避免 BCE 在极端不平衡时数值不稳。

```python
class EEGSignalDataset(Dataset):
    def __getitem__(self, i):
        i, j = self.index[i]
        raw = self.data[i][:, max(0, j - in_len + 1): j + 1]
        pad = in_len - raw.shape[1]
        if pad:                                   # 开头不足 1024 点就左侧补 0
            raw = np.pad(raw, ((0, 0), (pad, 0)), 'constant')
        label = self.gt[i][:, j].copy()
        if self.soft_label:
            label[label < .02] = .02
        return torch.from_numpy(raw.astype(np.float32)), \
               torch.from_numpy(label.astype(np.float32))
```

模型是纯 1D 卷积网络：先两三层卷积提取局部波形，再堆叠 6 个 block（每块 `MaxPool1d(2,2)` + `Dropout(0.1)` + 两层 5 宽卷积），最后用 1×1 卷积把通道压到 6，接 Sigmoid 输出每个事件的概率：

```python
class NNet(nn.Module):
    def __init__(self, in_channels=32, out_channels=6):
        super().__init__()
        self.hidden = 32
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 5, padding=2),
            nn.Conv1d(32, 32, 16, stride=16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 32, 7, padding=3),
        )
        for i in range(6):
            self.net.add_module(f'conv{i}', self.__block(32, 32))
        self.net.add_module('final', nn.Sequential(
            nn.Conv1d(32, out_channels, 1),
            nn.Sigmoid()))
    def __block(self, inch, outch):
        return nn.Sequential(
            nn.MaxPool1d(2, 2), nn.Dropout(p=0.1),
            nn.Conv1d(inch, outch, 5, padding=2), nn.LeakyReLU(0.1),
            nn.Conv1d(outch, outch, 5, padding=2), nn.LeakyReLU(0.1))
```

训练设置为 Adam（`lr=0.002`，betas 0.5/0.99）、batch 1024、损失 `BCELoss`，材料中只跑了 1 个 epoch 作为演示：

```python
nnet = NNet().to(device)
loss_fnc = nn.BCELoss()
adam = optim.Adam(nnet.parameters(), lr=0.002, betas=(0.5, 0.99))
for x, y in dataloader:
    x, y = x.to(device), y.to(device)
    loss = loss_fnc(nnet(x).squeeze(dim=-1), y)
    adam.zero_grad(); loss.backward(); adam.step()
```

### 2.4 模型测试与指标报告

我们把最后两个 series 留作验证，其余用于训练。预测后用 `roc_auc_score` 一次性计算 6 个事件的 AUC：

```python
nnet.eval()
y_pred = []
with torch.no_grad():
    for x, _ in testloader:
        y_pred.append(nnet(x.to(device)).cpu().numpy())
y_pred = np.concatenate(y_pred, axis=0).squeeze(axis=-1)
valid_gt = np.concatenate(valid_gt, axis=1)
print('auc roc: ', metrics.roc_auc_score(valid_gt.T, y_pred))
# auc roc: 0.8975334954123739
```

验证集 AUC 约 **0.8975**，训练集 AUC 约 **0.9625**，说明模型确实学到了动作模式，但也有一些过拟合。作为对照，材料里的传统机器学习方案（小波降噪 + 标准化 + `OneVsRestClassifier(SGDClassifier)`）在验证集上 AUC 只有 **0.5007**（等于随机），虽然准确率有 0.875，但那是因为绝大多数时刻标签都是 0。这个对比清楚地说明：不平衡数据上用准确率会被骗，必须看 AUC。

---

## 三、总结

本文中我们完成了 Grasp and Lift EEG Detection 的完整流程。我们首先做 EDA，确认 32 通道、6 个按时间顺序发生的事件，以及严重不平衡的标签分布。其次用 db2 小波做降噪、逐通道标准化，并丢弃无事件的时间块。接着用 1024 点滑窗构造样本，软标签缓解不平衡，训练一个堆叠 6 个 block 的 1D 卷积网络。最后在留出的两个 series 上得到验证 AUC 约 0.8975、训练 AUC 约 0.9625，并对比传统 SGD 方案的 0.5007。

这个比赛给了我们两个重要教训：一是时序信号任务里「滑窗 + 卷积」是非常自然且有效的结构，噪声和量纲要先处理掉；二是评价指标必须和数据的分布匹配，在正负样本极不均衡时，准确率会给出完全错误的乐观印象，只有 AUC 这类排序指标才能反映真实能力。
