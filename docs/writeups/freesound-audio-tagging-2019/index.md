# Freesound Audio Tagging 2019（音频打标签）

## 一、比赛简介

### 1.1 比赛背景

Freesound 是一个著名的免费音效网站，里面有大量用户上传的录音：雨声、吉他、狗叫、汽车喇叭、洗衣机……每一段录音都带着网友自己填的标签。这个比赛由 Freesound 和声学研究机构（CVSSP）联合举办，要求参赛者训练一个模型：**听一段几秒钟的音频，判断它里面出现了哪些声音**。

和常见的「一张图里有一只猫」不同，一段音频里往往同时有好几种声音，比如「雨声 + 打雷」「吉他 + 人声」。所以这是一个**多标签分类**（multi-label classification）问题——答案不是一个类别，而是一组类别。比赛一共有 80 个候选类别。

数据分两部分，这是这个比赛最关键的设计：

- **curated（精编集）**：人工挑选并人工标注的干净录音，标签可信，但只有 4970 条，数量很少；
- **noisy（噪声集）**：从网站上批量抓取、用网页标签自动生成的录音，有 19815 条，数量多但标签很脏，而且录音本身经常混着别的环境噪声。

也就是说，比赛的真正难点不是「模型不会做音频分类」，而是**怎么把一大堆脏数据利用起来，弥补干净数据太少的短板**。

### 1.2 比赛目标

输入是一段 `.wav` 音频文件，输出是该音频包含的 80 个类别各自的概率。训练集有两张表：

- `train_curated.csv`：两列 `fname`（文件名）和 `labels`（标签，多个标签用逗号隔开，例如 `Strum,Acoustic_guitar`），共 4970 行；
- `train_noisy.csv`：同样是 `fname` 和 `labels` 两列，共 19815 行。

测试集只给文件名，我们要为每个测试音频输出 80 个概率值，列名就是 80 个类别名。

### 1.3 评价标准

比赛用 **lwlrap**（label-weighted label-ranking average precision，按标签加权的排序平均精度）来评分，越高越好，满分 1.0。

这个指标名字很长，但意思很好理解。对**一个音频**来说：

1. 把 80 个类别按模型给的分数从高到低排个序；
2. 找出这段音频真正含有的那几个类别，看它们排在第几名；
3. 对每个正确答案，算「从第 1 名到它所在名次之间，命中了几个正确答案」除以「它的名次」，这就是该答案的精度。

举例：正确答案是 A 和 B。模型排序是 A 第一、C 第二、B 第三。那么 A 的精度是 1/1 = 1.0，B 的精度是 2/3 ≈ 0.667。

把所有音频、所有正确答案的精度平均起来，就是最终的 lwlrap。和普通的准确率相比，它有两个好处：**一是多标签也能算**，二是**它奖励「把正确答案排在前面」**，哪怕没排第一，只要排在很靠前也能拿不错的分数。

> 一个直观感受：随机乱猜时 lwlrap 很低；而比赛里能到 0.7 以上就已经是相当好的成绩了。

---

## 二、解题过程

我们参考的是比赛中**第 7 名队伍（kaggler-ja Shirogane）公开的方案**。整条主线可以概括成：**把音频转成对数梅尔频谱图，当成图片喂给 CNN；先用脏数据「预热」模型，再用干净数据精调；最后多个模型、多次裁剪取平均。**

我们按 EDA、数据清洗、模型训练与调参、测试与报告四步走。

### 2.1 数据探索（EDA）

我们首先把两张表读进来，看看规模和标签的样子：

```python
import pandas as pd

# 两张训练表：精编集和噪声集
curated_df = pd.read_csv('../input/train_curated.csv')
noisy_df = pd.read_csv('../input/train_noisy.csv')

# 看形状：(样本数, 列数)
print(curated_df.shape)   # (4970, 2)
print(noisy_df.shape)     # (19815, 2)

# 看前几行，注意 labels 列是逗号分隔的多个标签
curated_df.head()
```

打印出来后 `labels` 长这样：`Bathtub_(filling_or_washing)`、`Marimba_and_xylophone,Glockenspiel`、`Water_tap_and_faucet,Sink_(filling_or_washing)`。可以看出**一段音频确实可能同时属于好几个类别**，所以后面模型的输出层要用 sigmoid 而不是 softmax——每个类别各自独立判断「有还是没有」，而不是在 80 个类别里选一个。

接着我们看一下有多少段音频含多个标签：

```python
# 按逗号切开，数每段音频有几个标签
label_counts = curated_df['labels'].str.split(',').apply(len)

# 看看 1 个标签、2 个标签……各有多少条
print(label_counts.value_counts().sort_index())
```

结果显示大多数音频只有一个标签，两个标签的占一部分，三个以上的很少。这说明多标签虽然存在，但**大部分样本还是「一音频一主标签」**，这让我们对任务的难度有了底。

由于数据是 `.wav` 文件，我们还顺手听了几段、画了波形，确认采样率是 44100 Hz，每段音频长度不一（有的 2 秒，有的 30 秒）。这一点很重要——**CNN 要求输入尺寸固定**，所以后面必须做裁剪或补零。

### 2.2 数据清洗与预处理

这个比赛的数据清洗分两块：**音频层面的统一**和**脏数据的处理**。

**第一块：把任意长度的音频变成固定长度的「图片」。** 我们的做法是先统一采样率、去掉首尾静音、截断或居中补零到固定长度，再转成对数梅尔频谱图（log-mel spectrogram）。梅尔频谱图可以理解为「把声音的频率轴按人耳的感受重新划分，再画成一张随时间变化的能量热力图」，横轴是时间、纵轴是频率、颜色深浅是能量强弱。这样一张图就能直接喂给图像 CNN。

```python
import numpy as np
import librosa

def read_audio(conf, pathname, trim_long_data):
    """读音频，并统一成固定长度"""
    y, sr = librosa.load(pathname, sr=conf.sampling_rate)
    if 0 < len(y):
        y, _ = librosa.effects.trim(y)          # 去掉首尾的静音部分
    if len(y) > conf.samples:
        if trim_long_data:
            y = y[0:0 + conf.samples]           # 太长就只要开头一段
        else:
            pass
    else:
        # 太短就在两边对称补零，保证长度一致
        padding = conf.samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), conf.padmode)
    return y

def audio_to_melspectrogram(conf, audio):
    """把波形转成对数梅尔频谱图"""
    spectrogram = librosa.feature.melspectrogram(
        audio,
        sr=conf.sampling_rate,
        n_mels=conf.n_mels,      # 梅尔频带数量，也就是图片的高度
        hop_length=conf.hop_length,
        n_fft=conf.n_fft,
        fmin=conf.fmin, fmax=conf.fmax)
    # 转成分贝（对数）尺度，人耳对能量的感知本来就是对数的
    spectrogram = librosa.power_to_db(spectrogram)
    return spectrogram.astype(np.float32)
```

参数统一放在一个类里，方便全流程引用：

```python
class conf:
    sampling_rate = 44100
    duration = 2                     # 每段取 2 秒
    hop_length = 347 * duration      # 这样算下来时间轴正好 128 格
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128                     # 频率轴 128 格
    n_fft = n_mels * 20
    padmode = 'constant'
    samples = sampling_rate * duration
```

这样每段音频就变成一张 **128 × 128** 的灰度图（频率 128 行、时间 128 列），正好适合当 CNN 的输入。

**第二块：噪声集怎么用。** `train_noisy` 的标签是网页标签自动生成的，错误率不低。直接拿它训练会把模型带偏。我们的处理思路不是「清洗掉脏样本」，而是**把它降级使用**：只在训练的第一阶段（预热）用它，第二阶段用干净的 curated 数据精调。这样脏数据的作用是让模型先「见多识广」，而不是决定最终决策边界。

**第三块：数据增强。** curated 只有 4970 条，太少，必须扩增。我们用开源音频工具 `sox`（Kaggle 环境自带），对每条音频做了 7 种变换，每种生成一份同样大小的数据集：

```python
import subprocess as sp

# 音高上下移动 500 音分（约 5 个半音）
sp.run(["sox", "004ca909.wav", "out.wav", "pitch", "-500"])
sp.run(["sox", "004ca909.wav", "out.wav", "pitch", "+500"])

# 淡入淡出，改变音量包络
sp.run(["sox", "004ca909.wav", "out.wav", "fade", "q", "3"])

# 加混响
sp.run(["sox", "004ca909.wav", "out.wav", "reverb"])

# 提高/削弱高低频
sp.run(["sox", "004ca909.wav", "out.wav", "gain", "-h", "treble", "+20", "bass", "+20"])

# 对某个频段做均衡
sp.run(["sox", "004ca909.wav", "out.wav", "gain", "-h", "equalizer", "2400", "3q", "8"])
```

我们挑的都是**「改变了听感、但还能认出是同一个东西」**的变换——比如把吉他升调，听起来还是吉他；但如果加太多噪声，标签就不成立了。7 份增强数据 + 1 份原始数据，训练集直接变成原来的 8 倍。

### 2.3 模型训练与超参数调优

#### 2.3.1 裁剪策略：先随机裁，再「往响的地方裁」

频谱图比 128×128 宽，需要裁剪。最初我们像大多数公开方案一样**随机裁**，但很快发现一个问题：很多声音的关键信息集中在前几秒，也有很多在中段，**很少出现在结尾**。我们试过只取最后几秒，分数掉了 0.03～0.04。

于是我们设计了「强度自适应裁剪」（strength adaptive crop）：先用滑动窗口算出频谱图**每个位置的能量总和**，然后按能量高低做加权随机抽样，让裁剪窗口更容易落在声音响的部分。

```python
def _get_strength(x):
    """算出每个滑动窗口位置的能量总和，供加权抽样使用"""
    time_dim, base_dim = x.shape[1], x.shape[0]
    x_sum = []
    for j in range(0, time_dim - base_dim + 1):
        # 只统计前 84 行（主要频段），避免无关高频干扰
        x_temp = x[0:84, j:j + base_dim].sum()
        x_sum.append(x_temp)
    return x_sum

class FATTrainDataset(Dataset):
    def __init__(self, mels, labels, transforms, strength_list):
        self.mels = mels
        self.labels = labels
        self.transforms = transforms
        self.strength_list = strength_list   # 预先算好的能量列表

    def __getitem__(self, idx):
        x = self.mels[idx]
        x_sum = self.strength_list[idx][0]
        height = self.strength_list[idx][1]  # 能量最大值 - 最小值
        x_min = self.strength_list[idx][3]
        time_dim, base_dim = self.strength_list[idx][4], self.strength_list[idx][5]

        # 反复抽，直到抽到的位置「能量够高」为止
        flag = True
        while flag:
            crop = random.randint(0, int(height * (time_dim - base_dim)))
            if height == 0:
                crop_idx = 0
            else:
                crop_idx = int(crop // height)
            value = int(crop % height)
            if ((x_sum[crop_idx] - x_min) - value) >= 0:
                flag = False
            crop = crop_idx

        image = Image.fromarray(x, mode='RGB')
        image = image.crop([crop, 0, crop + base_dim, base_dim])
        return self.transforms(image).div_(255), torch.from_numpy(self.labels[idx]).float()
```

这一招的效果很直接：**交叉验证分数涨了 0.01，排行榜涨了 0.004～0.005**。数值不大，但在比赛后期每 0.001 都很宝贵。

#### 2.3.2 模型结构：带注意力的自建 CNN

我们的主力模型是从 CVSSP 官方 baseline（一个 9 层 CNN）改出来的。改成 5 个卷积块，每个块里加**空间注意力 + 通道注意力（scSE）**，让模型自己学习「图上哪个区域、哪个特征通道更重要」：

```python
class SCse(nn.Module):
    """空间注意力 + 通道注意力，两者相加"""
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)   # 空间：给每个像素位置打分
        self.catt = GAB(dim)                  # 通道：给每个特征通道打分
    def forward(self, x):
        return self.satt(x) + self.catt(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ConvBlock, self).__init__()
        # 实验发现 (2,2) 的卷积核比常见的 (3,3) 效果更好
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2),
                               stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(2, 2),
                               stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = SCse(out_channels)
    def forward(self, inp, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(inp)))
        x = F.relu_(self.scse(self.bn2(self.conv2(x))))
        # 池化时把 max 和 avg 加起来，信息更全
        if pool_type == "both":
            x = F.max_pool2d(x, kernel_size=pool_size) + F.avg_pool2d(x, kernel_size=pool_size)
        return x
```

这里有一个和普通图像分类很不一样的设计细节：**频谱图的时间轴和频率轴不能同等对待**。在自然图片里，猫挪到画面顶部还是猫；但在频谱图里，同一个模式出现在低频和高频代表完全不同的声音。所以我们的池化是**交替沿时间轴、沿频率轴**做的（第一块不池化，第二块池时间轴，第三块池频率轴，如此交替），并且用**空间金字塔池化**把特征图切成不同大小的块分别聚合，保留「位置」信息。

这一套设计让交叉验证的 lwlrap 提升了 **0.015～0.03**，是整个模型改进里最大的一笔。

我们还训练了两个 Inception v3：一个是标准的 3 通道版本（直接把频谱图复制成 RGB 三通道），另一个把第一层改成 1 通道。两者结果有差异，正好用来做集成。

#### 2.3.3 批内混合（MixUp）

除了音频层面的增强，我们还在**图片层面**做混合：随机取两张频谱图，按一个随机比例 alpha 叠加成一张，标签也按同样比例混合。

```python
def _get_blend_arr(x_trn, y_trn, lam=np.random.beta(1.0, 1.0)):
    """把两张频谱图按比例混合，标签也按比例混合"""
    blend_arr_x, blend_arr_y = [], []
    for i in range(1000):
        r1 = random.randint(0, len(x_trn) - 1)
        r2 = random.randint(0, len(x_trn) - 1)
        img0 = array2img(x_trn[r1])       # 先各自随机裁一块
        img1 = array2img(x_trn[r2])
        img_blend = Image.blend(img0, img1, alpha=lam)   # 按 lam 叠加
        y_blend = y_trn[r1] * (1 - lam) + y_trn[r2] * lam  # 标签同比混合
        blend_arr_x.append(np.asarray(img_blend))
        blend_arr_y.append(y_blend)
    return blend_arr_x, blend_arr_y
```

这样模型见到的「样本」数量等于无限多，而且被逼着学「多种声音共存」的表示，很契合多标签任务。

#### 2.3.4 两阶段训练：先脏后净

这是整个方案最核心的战术。我们不是把 curated 和 noisy 混在一起训练，而是**分两个阶段**：

**阶段一：用 noisy 数据预热（pre-train）。** 训练设置沿用当时社区里公认好用的组合：

```python
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = Adam(model.parameters(), lr=1e-3)
# 学习率按余弦曲线在 1e-3 到 1e-5 之间循环，每 10 个 epoch 一个周期
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
# 多标签用带 logits 的二元交叉熵，比先 sigmoid 再算更稳定
criterion = torch.nn.BCEWithLogitsLoss()
```

预热阶段我们试了两种验证方式：一种是用全部 noisy 训练、用 curated 验证；另一种是把 noisy 自己切成训练/验证。结果第二种略好一点，不过差距很小，我们判断其中有一部分是随机种子的运气。

**阶段二：用 curated 数据精调（fine-tune）。** 把阶段一的权重拿来当初始值，在 8 倍扩增后的 curated 数据上继续训练。这里用 5 折交叉验证，因为 7 份增强数据和原数据**索引一一对应**，只要按同样的索引切分，就不会发生数据泄漏；验证集只用原始数据，不用增强数据（增强数据太大，而且评估原始数据更接近测试分布）。

我们特意验证了预训练的价值：

- 自建 CNN：交叉验证 lwlrap 从 **0.83 提升到 0.88**；
- Inception v3：从 **0.86 提升到 0.87**，排行榜上大约提升 **0.01**。

也就是说，**脏数据本身单独用很差，但作为「预训练素材」非常值钱**。这让 4970 条干净样本发挥出了远超其数量的作用。

#### 2.3.5 验证时的 TTA

因为我们训练时用的是随机裁剪，同一个音频每次裁到的片段不同，预测结果就会波动。如果验证时只裁一次，分数会抖得没法判断哪个 epoch 最好。所以验证时我们也做 **TTA**（test-time augmentation，测试时增强）：对同一个音频裁多个不同位置，把预测取平均。

```python
class FATValidDataset(Dataset):
    """验证集：每个样本重复 tta 次，每次随机裁一个位置"""
    def __init__(self, fnames, mels, labels, transforms, strength_list, tta=5):
        self.fnames = fnames
        self.mels = mels
        self.labels = labels
        self.transforms = transforms
        self.strength_list = strength_list
        self.tta = tta

    def __len__(self):
        return len(self.fnames) * self.tta   # 数据集长度翻 tta 倍

    def __getitem__(self, idx):
        new_idx = idx % len(self.fnames)     # 同一段音频被取 tta 次
        x = self.mels[new_idx]
        # ……（与训练集相同的强度自适应裁剪逻辑）
        image = Image.fromarray(x, mode='RGB')
        image = image.crop([crop, 0, crop + base_dim, base_dim])
        return self.transforms(image).div_(255), \
               torch.from_numpy(self.labels[new_idx]).float(), self.fnames[new_idx]
```

预测完之后按文件名分组求平均，就得到每个音频的最终分数。

### 2.4 模型测试与指标报告

#### 2.4.1 先把评分函数写对

比赛官方给了 lwlrap 的实现，我们直接拿来当本地评估函数：

```python
import numpy as np

def _one_sample_positive_class_precisions(scores, truth):
    """单个样本：算出每个正确答案的精度"""
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)     # 真正的类别下标
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    retrieved_classes = np.argsort(scores)[::-1]      # 按分数从高到低排序
    class_rankings = np.zeros(num_classes, dtype=np.int64)
    class_rankings[retrieved_classes] = range(num_classes)   # 每个类别的名次

    retrieved_class_true = np.zeros(num_classes, dtype=bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    # 精度 = 到该名次为止命中的正确答案数 / 该名次
    precision_at_hits = (retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
                         (1 + class_rankings[pos_class_indices].astype(float)))
    return pos_class_indices, precision_at_hits

def calculate_per_class_lwlrap(truth, scores):
    """整个数据集：按类别算精度，再按类别出现频率加权平均"""
    num_samples, num_classes = scores.shape
    precisions = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        idx, p = _one_sample_positive_class_precisions(scores[i, :], truth[i, :])
        precisions[i, idx] = p
    labels_per_class = np.sum(truth > 0, axis=0)          # 每个类别出现了几次
    weight_per_class = labels_per_class / labels_per_class.sum()
    per_class_lwlrap = precisions.sum(axis=0) / np.maximum(1, labels_per_class)
    # 按类别频率加权，得到最终总分
    return per_class_lwlrap, weight_per_class
```

理解这段代码的关键就是 2.3 节说的那个直观规则：**正确答案排得越靠前，精度越高；多个类别的分数再按各自出现次数加权**。这样样本多的类别话语权大，符合比赛定义。

#### 2.4.2 各种技巧的贡献

我们把每一招都单独测了一遍，得到下面这组对照（同一模型下，其他条件不变）：

| 设置 | 交叉验证 lwlrap | 排行榜 lwlrap |
| --- | --- | --- |
| 不做增强 | 0.800 | 0.690 |
| 一部分增强（降调、淡入淡出、混响、削弱高低频） | 0.829 | 0.720 |
| 全部增强 | 0.825 | 0.72 以上 |
| 全部增强 + noisy 预热 | 0.88 | — |

从表里能读出两件事：**一是增强确实有用**，但每种增强单独的收益并不一样，而且组合起来未必是简单相加（「全部增强」的交叉验证反而略低于「一部分增强」，说明增强过头会引入分布偏移）；**二是预训练才是真正的大头**，一下把交叉验证从 0.83 推到 0.88。

#### 2.4.3 集成与提交

最后一步是把所有模型合起来。我们试过很多分数在 0.700～0.710 的模型，发现一个问题：**是「多堆几个模型」划算，还是「多做一些 TTA」划算？** 实验结论是 Inception v3 比较不稳定，需要 20～25 次 TTA 才能稳定下来，而继续加入低分模型带来的收益有限。

于是最终方案定为：**每折训练 3 个模型（自建 CNN + 3 通道 Inception v3 + 1 通道 Inception v3），5 折共 15 个模型，每个模型 20 次 TTA，全部等权平均**。

```python
# 对每个模型、每个折，先做 tta 次预测，再按文件名求平均
preds = []
for model, loader in all_model_loaders:          # 15 个模型
    model.eval()
    fold_pred = []
    with torch.no_grad():
        for images, _, fnames in loader:         # 每个样本出现 20 次
            logits = model(images)
            fold_pred.append((fnames, torch.sigmoid(logits)))
    preds.append(average_by_fname(fold_pred))    # 同一文件名的 20 次结果取平均

# 15 个模型等权平均，得到最终提交概率
final = sum(preds) / len(preds)
submission = pd.DataFrame(final, columns=class_names)
submission.insert(0, 'fname', test_fnames)
submission.to_csv('submission.csv', index=False)
```

等权而不是按分数加权，是因为各模型分数接近，加权容易过拟合验证集。提交前我们核对了行数、列名和 `sample_submission.csv` 是否一致——这个比赛类别名里带括号和逗号，写错一个字符整份文件就作废。

---

## 三、总结

本文中我们完成了 Freesound Audio Tagging 2019 的完整方案复现。我们首先通过 EDA 确认了这是 80 类的多标签问题，精编集只有 4970 条、噪声集有 19815 条但标签很脏。其次我们把音频统一裁成 2 秒、转成 128×128 的对数梅尔频谱图，并用 sox 做了音高、淡入淡出、混响、高低频、均衡等 7 种增强，把干净数据扩成 8 倍。接着我们设计了带空间/通道注意力的 5 块 CNN，用交替方向的池化加空间金字塔池化保留频谱图的位置信息，配合「往响的地方裁」的自适应裁剪和批内图像混合；训练上用带余弦退火学习率的 Adam 和 BCEWithLogitsLoss，**先拿噪声集预热、再拿精编集精调**。最后我们把 5 折 × 3 个模型共 15 个模型、每个 20 次 TTA 等权集成，交叉验证 lwlrap 从不用增强时的 0.800 提升到 0.88，排行榜从 0.690 提升到 0.72 以上。

这个比赛留给我们最重要的一条经验是：**数据的「质量」和「用途」可以分开看**。噪声集直接训练是负担，但当成预训练素材就是财富——同一份数据，换个用法，价值完全不同。
