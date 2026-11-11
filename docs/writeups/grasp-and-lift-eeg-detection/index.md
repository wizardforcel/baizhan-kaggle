# Grasp and Lift EEG Detection

## 一、比赛简介

### 1.1 比赛背景

Grasp and Lift EEG Detection 是一个脑电信号（Electroencephalography，EEG）事件检测比赛。人在做“抓握并抬起物体”这个动作时，大脑皮层会产生有规律的电位变化，贴在头皮上的电极阵列可以记录到这些微弱的电压波动。比赛给出的就是这样的多通道时序数据，要求我们逐时间点预测 6 个动作事件是否正在发生。

这 6 个事件按动作的先后顺序是：

- `HandStart`：手开始移动；
- `FirstDigitTouch`：手指第一次接触物体；
- `BothStartLoadPhase`：双手开始承重；
- `LiftOff`：物体离开桌面；
- `Replace`：物体放回；
- `BothReleased`：双手松开。

它们不是互相独立的：`LiftOff` 通常发生在 `FirstDigitTouch` 之后，`Replace` 又跟在 `LiftOff` 后面。这种时间上的先后关系是重要的先验，但模型如果只看单个时间点，就学不到它，因此需要给模型足够的上下文窗口。

数据采样率是 500 Hz，也就是每秒 500 个时间点。每条记录 `subjX_seriesY_data.csv` 有 32 列通道值和一列 `id`；对应的事件文件 `subjX_seriesY_events.csv` 有 6 列 0/1 标签。训练集和测试集按受试者和场次（series）组织，测试集只有 data 文件，没有 events 文件。

### 1.2 比赛目标

输入是一段 32 通道的 EEG 时序，输出是每个时间点上 6 个事件各自的概率（0~1）。提交文件的行 id 与测试数据一致，列名为 6 个事件名。

```python
labels = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase',
          'LiftOff', 'Replace', 'BothReleased']
```

因为多个事件可以同时为 1（例如接触物体的同时开始承重），这是一个多标签（multi-label）问题，而不是多分类。多分类同一时刻只能选一个类，多标签则允许 6 个输出都独立地取 0 或 1。

### 1.3 评价标准

比赛使用 6 个事件各自的 ROC AUC 取平均作为指标。AUC（Area Under the ROC Curve）衡量的是“随机取一个正样本和一个负样本，模型给正样本打分更高的概率”，0.5 表示和随机猜一样，1.0 表示完美排序。它只关心排序，不关心概率的绝对大小，也不依赖某个固定阈值。

为什么不用准确率？因为 6 个事件在整个时间轴上大部分时刻都是 0，正样本占比很低。假设正样本只有 5%，一个“永远预测 0”的模型准确率也有 95%，但它毫无用处。AUC 对类别不平衡不敏感，能真实反映模型区分事件与背景的能力。材料里的验证代码正是用 `roc_auc_score` 一次性计算 6 个事件的 AUC：

```python
from sklearn import metrics
print('auc roc: ', metrics.roc_auc_score(valid_gt.T, y_pred))
```

---

## 二、解题过程

我们的主线是：先做信号可视化与统计分析，其次做小波降噪、标准化和无效片段剔除，接着用 1D 卷积网络做多标签分类，最后在留出的 series 上报告 AUC，并与传统机器学习基线对比。

### 2.1 数据探索（EDA）

首先解压数据并读取一条记录。读取函数把 data 与 events 分别读成矩阵，去掉 `id` 列后转置，得到“通道 × 时间”的形状：

```python
def read_csv(data, events):
    x = pd.read_csv(data)
    y = pd.read_csv(events)
    x = x.iloc[:, 1:].values      # 去掉 id 列
    y = y.iloc[:, 1:].values
    return x, y                   # 形状均为 [时间点, 通道/事件]
```

接着统计 6 个事件为正的样本数：

```python
ye = pd.DataFrame(y_train, columns=labels)
sns.barplot(x=labels, y=ye.sum().values)
plt.title("Number of samples labeled as active (1)")
plt.xticks(rotation=45)
plt.show()
```

柱状图显示 6 个事件的正样本数差异明显，整体占全部时间点的比例很低。这再次说明为什么必须用 AUC：如果只看准确率，模型只要全部输出 0 就能拿到很高的分数。

然后我们画原始信号与降噪信号的对比：

```python
signal = pd.read_csv('train/subj1_series1_data.csv').drop("id", axis=1)
filtered = wavelet_denoising(signal.values, wavelet='db2', level=3)

plt.plot(signal.iloc[:10000, 1], label='signal', alpha=0.5)
plt.plot(filtered[:10000, 1], label='reconstructed', color='k')
plt.title('Denoising with DWT')
plt.legend(); plt.show()
```

从图上看，原始信号上有明显的高频毛刺，降噪后的曲线更平滑，但保留了慢速的波动趋势。EEG 的有用信息大多集中在较低频段，高频成分更多是肌电噪声和工频干扰，所以降噪是合理的第一步。

### 2.2 数据清洗

脑电信号的清洗分三步。

**第一步：小波降噪。** 小波变换把信号分解成不同尺度的系数，噪声通常集中在细节系数里。我们对细节系数做硬阈值处理，小于阈值的系数直接置零，再重建信号：

```python
def wavelet_denoising(x, wavelet='db2', level=3):
    coeff = pywt.wavedec(x, wavelet, mode="per")      # 3 层小波分解
    sigma = (1 / 0.6745) * madev(coeff[-level])       # 用最高频系数估计噪声水平
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))     # 通用阈值
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard')
                 for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')   # 重建

def madev(d, axis=None):
    """平均绝对偏差，用来稳健地估计噪声标准差"""
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)
```

阈值公式 `sigma * sqrt(2*log(n))` 是经典的 VisuShrink 阈值，`sigma` 用最高频系数的平均绝对偏差估计，对异常值不敏感。

**第二步：标准化。** 每个通道减去均值、除以标准差，让不同电极的数值尺度一致。这一步对神经网络很重要：如果某个通道的量级是另一个的几十倍，梯度会被它主导，其他通道的信息就学不进去。

**第三步：丢弃无事件片段。** 材料把时间轴按 1000 个点切块，如果某块内 6 个事件的平均值都低于 0.01，就整块丢弃：

```python
chunk_size = 1000
threshold = 0.01
index = []
for i in range(len(gt)):
    for j in range(0, gt[i].shape[1], chunk_size):
        if np.mean(gt[i][:, j:j+chunk_size]) < threshold:
            continue                     # 整块没有事件，不参与训练
        index.extend([(i, k) for k in range(j, min(gt[i].shape[1], j+chunk_size))])
```

这一步丢掉了 8859 个块、约 0.5% 的数据，但显著降低了“全零样本”的比例，让每个 batch 里都有足够多的正样本，训练更稳定。

还有一点是标签的“软截断”。原始 events 是硬 0/1 标签，但材料在训练时把小于 0.02 的值抬到 0.02，相当于给负样本留了一点概率空间。这样做的原因是 BCE 在预测概率接近 0 或 1 时会饱和，梯度接近消失；把标签限制在 (0, 1) 开区间内，能让模型在训练早期更快地调整。验证时则用原始硬标签，保证 AUC 的计算与比赛口径一致。

全流程的标准化参数（均值与标准差）也应当只在训练集上估计，再应用到验证集和测试集。材料里预先保存了 `mean.npy` 与 `std.npy` 两个文件，正是为了避免在推理时重新 `fit`，否则不同批次的数据分布差异会让模型输出偏移。

### 2.3 模型训练与超参数调优

我们采用滑窗构造样本：每个样本取当前时刻之前的 1024 个采样点（约 2 秒）× 32 通道，标签是当前时刻的 6 个事件。训练时对标签做软截断，把小于 0.02 的值抬到 0.02，防止 BCE 在极端不平衡时数值不稳：

```python
class EEGSignalDataset(Dataset):
    def __getitem__(self, i):
        i, j = self.index[i]
        raw = self.data[i][:, max(0, j - in_len + 1): j + 1]
        pad = in_len - raw.shape[1]
        if pad:                                  # 开头不足 1024 点就左侧补 0
            raw = np.pad(raw, ((0, 0), (pad, 0)), 'constant')
        label = self.gt[i][:, j].copy()
        if self.soft_label:
            label[label < .02] = .02             # 软标签，避免 log(0)
        return (torch.from_numpy(raw.astype(np.float32)),
                torch.from_numpy(label.astype(np.float32)))
```

模型是纯 1D 卷积网络。前三层负责提取局部波形：先用 5 宽卷积把 32 通道混合，再用 16 宽、步长 16 的卷积把时间轴压缩 16 倍，然后用 7 宽卷积扩大感受野。之后堆 6 个 block，每个 block 先 `MaxPool1d(2,2)` 把时间减半，再接两层 5 宽卷积，最后用 1×1 卷积输出 6 个通道并过 Sigmoid：

```python
class NNet(nn.Module):
    def __init__(self, in_channels=32, out_channels=6):
        super().__init__()
        self.hidden = 32
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 5, padding=2),
            nn.Conv1d(32, 32, 16, stride=16),      # 时间下采样 16 倍
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 32, 7, padding=3),
        )
        for i in range(6):
            self.net.add_module(f'conv{i}', self.__block(32, 32))
        self.net.add_module('final', nn.Sequential(
            nn.Conv1d(32, out_channels, 1),
            nn.Sigmoid()))                          # 每个事件独立的概率
    def __block(self, inch, outch):
        return nn.Sequential(
            nn.MaxPool1d(2, 2), nn.Dropout(p=0.1),
            nn.Conv1d(inch, outch, 5, padding=2), nn.LeakyReLU(0.1),
            nn.Conv1d(outch, outch, 5, padding=2), nn.LeakyReLU(0.1))
```

这样设计的直觉是：池化层不断扩大感受野，让后面的卷积能看到更长的上下文；`Dropout(0.1)` 抑制过拟合；最后的 1×1 卷积把每个时间位置的特征映射成 6 个事件的得分。

选择 1024 点作为输入窗口，是因为采样率是 500 Hz，1024 点大约覆盖 2 秒。动作事件本身通常持续几百毫秒到一秒，2 秒的上下文足以包含“动作前”和“动作后”的波形变化。如果窗口太短，模型只能看到动作的瞬间，区分不了方向；如果窗口太长，反而把大量无关的背景也带进来，还会增加显存和计算量。材料里的 1024 是经验值与算力之间的折中，实际比赛可以在验证集上比较 512、1024、2048 等窗口大小。

另一个细节是滑窗的步长。训练时我们对每个时间点都构造一个样本，相邻样本之间有 1023 个点重叠，这会让数据量看起来很大、样本高度相关。好在这里的目的不是统计独立性，而是让模型见到所有时间位置；验证时同样逐点预测，因此 AUC 的计算覆盖了完整时间轴。代价是训练一个 epoch 就要跑数百万个样本，所以材料只跑 1 个 epoch 作为演示，真正比赛需要用更大的 GPU 和更多轮数。

训练设置为 Adam（`lr=0.002`，betas 0.5/0.99）、batch 1024、损失 `BCELoss`，材料中只跑了 1 个 epoch 作为演示：

```python
nnet = NNet().to(device)
loss_fnc = nn.BCELoss()
adam = optim.Adam(nnet.parameters(), lr=0.002, betas=(0.5, 0.99))

for x, y in dataloader:
    x, y = x.to(device), y.to(device)
    loss = loss_fnc(nnet(x).squeeze(dim=-1), y)
    adam.zero_grad()
    loss.backward()
    adam.step()
```

### 2.4 模型测试与指标报告

我们把最后两个 series 留作验证，其余用于训练。预测后把每个 batch 的概率拼接起来，再用 `roc_auc_score` 计算：

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

验证集 AUC 约 **0.8975**，训练集 AUC 约 **0.9625**。两者有差距，说明模型有一定过拟合；在 1 个 epoch 的演示设置下这个结果已经说明卷积结构能抓到动作相关的波形。

作为对照，材料里的传统机器学习方案（小波降噪 + 标准化 + `OneVsRestClassifier(SGDClassifier)`）在验证集上 AUC 只有 **0.5007**，基本等于随机；但它的准确率有 **0.875**。这两组数字放在一起非常有教育意义：0.875 的准确率听起来不错，其实是“全猜 0”也能达到的水平；AUC 0.5 才暴露了模型没有学到任何排序信息。

---

## 三、总结

本文中我们完成了 Grasp and Lift EEG Detection 的完整流程。我们首先做 EDA，确认数据是 32 通道、500 Hz 的多标签时序，6 个事件按动作顺序发生且正样本极度稀少。其次用 db2 小波做降噪、逐通道标准化，并丢弃无事件的时间块。接着用 1024 点滑窗构造样本，对标签做软截断，训练一个堆叠 6 个 block 的 1D 卷积网络。最后在留出的两个 series 上得到验证 AUC 约 0.8975、训练 AUC 约 0.9625，并与传统 SGD 基线的 0.5007 做对比。

这个比赛给了我们两个重要教训。第一，**时序信号任务里“滑窗 + 卷积”是非常自然的结构**：卷积能提取局部波形，池化能扩大感受野，二者配合正好对应 EEG 的物理特性。第二，**评价指标必须和数据的分布匹配**：在正负样本极不均衡时，准确率会给出完全错误的乐观印象，只有 AUC 这类排序指标才能反映模型的真实能力。此外，降噪和标准化这些看似“传统”的预处理，在深度模型上依然能显著影响训练稳定性，不能因为用了神经网络就跳过。