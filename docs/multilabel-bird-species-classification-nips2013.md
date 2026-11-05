# NIPS 2013 多标签鸟类物种分类

## 一、比赛简介

### 1.1 比赛背景

NIPS 2013 多标签鸟类物种分类（Multilabel Bird Species Classification）是音频领域的经典比赛。它的数据和 MLSP 2013 鸟类比赛类似，都来自野外录音，但任务形式不同：这里的一条录音可以同时标注多种鸟，而且标签是按“物种 + 叫声类型”细分的，例如 `Aegcau_call` 表示某种鸟的“呼叫”，`Alaarv_song` 表示另一种鸟的“鸣唱”。同一只鸟的 call 和 song 在声学上差别很大，所以模型必须把它们当成不同的类别。

比赛提供的原始数据是 WAV 音频和 MFCC 特征，但材料里的做法没有直接对 MFCC 建模，而是先把音频转成**梅尔声谱图**（mel spectrogram），把一维的声音变成二维的“图片”，再用图像分类的卷积网络来做。这个思路在音频领域叫“声谱图 + CNN”，好处是可以直接借用成熟的图像模型（比如 ResNet），也能用图像增强手段扩充数据。

难点主要有三个：一是多标签，一条录音可能同时有多个正确答案；二是类别极多且极不均衡，很多类别在训练集里只有几十条；三是录音长度不一，背景噪声大，鸟叫只占一小段。

### 1.2 比赛目标

数据包含：

- `NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV`：`train/` 下有 687 条 WAV，`test/` 下有 1000 条 WAV；
- `NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS`：训练标签 CSV，文件名 `nips4b_birdchallenge_train_labels.csv`；
- `NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_MFCC`：预先算好的 MFCC 特征。

标签表的第一列是文件名，后面每一列是一个类别，取值 0/1 表示该录音里有没有这个类别。我们要为 1000 条测试录音的每个类别输出一个概率。

### 1.3 评价标准

官方评价使用 **ROC 曲线下面积（AUC）**，对每个类别分别计算再取平均。AUC 衡量的是排序能力：模型给“正样本”的概率是否普遍高于“负样本”。因为是多标签，每个类别独立算一次，最后取宏平均，所以稀有类别和常见类别权重相同，不能只把常见类做好。

材料里的 fastai 训练过程用了一个更方便监控的指标 `accuracy_thresh`：把预测概率按阈值 0.2 二值化，再和真实标签比对算准确率。它和官方 AUC 不完全一样，但能快速看出模型有没有在学。

## 二、解题过程

我们的主线是：先把 WAV 批量转成梅尔声谱图 PNG，其次用 fastai 的多标签数据加载器读入图片和标签，接着用 ResNet50 在 128 和 256 两种尺寸上微调，最后对测试集预测并按比赛格式写提交文件。

### 2.1 数据探索（EDA）

我们首先看标签表的结构。它不是标准的“一行一个标签”，而是一行一个文件、后面跟很多个 0/1 列：

```python
import pandas as pd

label_path = ('../input/multilabel-bird-species-classification-nips2013/'
              'nips4b_bird_challenge_train_labels/'
              'NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/'
              'nips4b_birdchallenge_train_labels.csv')

# 前两行是表头说明，跳过；第一列是文件名，后面是各个类别的 0/1
df = pd.read_csv(label_path, skiprows=[0, 2])
print(df.shape)
print(df.columns[:10].tolist())
print(df.head())
```

从输出可以看到，第一列是录音文件名，后面几十列都是类别名，列名由“物种代码 + 叫声类型”组成。`df.shape` 的行数约等于 687（训练音频数），列数则是类别数加一。每行的取值只有 0 和 1，少数行可能一个 1 都没有，说明这段录音没有目标鸟。

接着我们统计每个类别有多少正样本：

```python
# 去掉文件名列，只留 0/1 标签
labels = df.drop(columns=df.columns[0])
counts = labels.sum().sort_values(ascending=False)

print(counts.head(10))    # 最常见的类别
print(counts.tail(10))    # 最稀有的类别
print('正样本为 0 的类别数:', (counts == 0).sum())
```

结果呈现出典型的长尾分布：少数类别有上百条正样本，大量类别只有个位数，甚至有些类别在训练集里一条正样本都没有。这解释了为什么后面不能只看总体准确率——模型只要把所有类别都猜 0，也能得到很高的“准确率”，但对 AUC 毫无帮助。

然后我们看一条音频的声谱图，确认鸟叫在时间上的分布：

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

wav_path = ('../input/nips4b_bird_challenge_train_test_wav/'
            'NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/'
            'train/nips4b_birds_trainfile015.wav')
clip, sr = librosa.load(wav_path, sr=None)

# 梅尔声谱图：横轴是时间，纵轴是梅尔频率，颜色是能量
S = librosa.feature.melspectrogram(y=clip, sr=sr)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
plt.colorbar()
plt.show()
```

声谱图上能看到鸟叫对应一段段明亮的纹理，而背景噪声是均匀的暗色。把这种图交给 CNN，卷积核就能学到“哪种纹理对应哪种鸟”。

### 2.2 数据清洗

这份数据的清洗主要是格式转换：把 WAV 变成统一的 PNG 图片。我们写了一个函数，对每条音频算梅尔谱、转 dB、画成图并保存：

```python
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def create_spectrogram(fn_audio, fn_gram, zoom=1):
    # sr=None 表示保持原始采样率，不做重采样
    clip, sample_rate = librosa.load(fn_audio, sr=None)

    # 梅尔谱：把人耳不敏感的线性频率换成梅尔刻度
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)

    # 图的大小跟着声谱图尺寸走，dpi=1 是为了不让 matplotlib 额外缩放
    fig = plt.figure(figsize=tuple(reversed(S.shape)), dpi=1)
    plt.gca().set_axis_off()   # 去掉坐标轴，避免把刻度画进图片
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    fig.savefig(fn_gram, dpi=zoom, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
```

这里有两个关键点。第一，`set_axis_off()` 和 `bbox_inches='tight'` 是为了得到一张“纯图”，没有白边、没有坐标轴，否则 CNN 会把坐标轴的数字当成特征。第二，`dpi=zoom` 控制输出分辨率，zoom=2 时图片更清晰，但文件也更大；材料里训练和测试都用 zoom=2。

接着批量处理 train 和 test：

```python
from fastprogress import progress_bar

def audios_to_spectrograms(from_path, to_path, folder='', zoom=2):
    (to_path / folder).mkdir(parents=True, exist_ok=True)
    fns = list((from_path / folder).glob('*.wav'))
    for src in progress_bar(fns):
        dest = to_path / folder / (src.stem + '.png')
        create_spectrogram(src, dest, zoom)

for ds in ('train', 'test'):
    audios_to_spectrograms(wav_dir, spect_dir, ds, zoom=2)
```

转换完成后，训练集得到 687 张图，测试集得到 1000 张图。图片文件名和原 WAV 同名，只是后缀变成 `.png`，这样标签表里的文件名可以直接对应上。

标签方面，我们不做人工平衡。多标签任务里，过采样少数类别容易把同一张图重复很多次，导致过拟合；我们改用带阈值的准确率来监控稀有类别，并在最终预测时保留概率而不是硬标签。

### 2.3 模型训练与超参数调优

我们用 fastai 的 `ImageList` 从 DataFrame 构建数据集。关键是 `label_from_df(cols=...)`：它把标签表里除文件名外的所有列都当成多标签目标，模型输出会是一个向量，每个类别一个 sigmoid 概率。

```python
from fastai.vision import *
from functools import partial

np.random.seed(42)

# 从 DataFrame 建图片列表，按 8:2 随机切分训练/验证
src = (ImageList.from_df(df, spect_dir, folder='train', suffix='.png')
       .split_by_rand_pct(0.2)
       .label_from_df(cols=df.columns[1:].tolist()))

# 声谱图不能左右翻转：翻转会把时间轴倒过来，鸟叫的时序结构就毁了
tfms = get_transforms(do_flip=False, max_rotate=None, max_warp=None)

data = (src.transform(tfms, size=128)
        .databunch(num_workers=0)
        .normalize(imagenet_stats))
```

`do_flip=False` 是音频转图像任务里特别重要的一条：普通图像左右翻转通常无害，但声谱图的横轴是时间，翻转等于把叫声倒放，语义完全变了。同理，大幅度旋转和透视变换也不适合，所以材料只保留了温和的裁剪和亮度变化。

模型用 ResNet50（ImageNet 预训练），损失函数是 fastai 多标签默认的 BCEWithLogitsLoss，监控指标是阈值 0.2 的准确率：

```python
from fastai.vision import models

arch = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)
learn = cnn_learner(data, arch, metrics=acc_02, path='.')

# 先用 lr_find 找学习率，避免手拍
learn.lr_find()
learn.recorder.plot(suggestion=True)

lr = 2.29e-02
learn.fit_one_cycle(5, slice(lr))
```

第一阶段的 5 轮结果如下（来自材料）：

| 轮次 | train_loss | valid_loss | accuracy_thresh |
| --- | --- | --- | --- |
| 0 | 0.8004 | 0.5986 | 0.1056 |
| 1 | 0.5293 | 0.1277 | 0.9792 |
| 2 | 0.3780 | 0.1429 | 0.9763 |
| 3 | 0.2957 | 0.1090 | 0.9739 |
| 4 | 0.2417 | 0.1034 | 0.9724 |

第 0 轮准确率只有 0.106，第 1 轮跳到 0.979，之后稳定在 0.97 左右。这个跳变说明预训练权重在第 1 轮就完成了“从通用图像特征到声谱图特征”的迁移。

接着解冻整个网络做微调，并把图片尺寸从 128 提高到 256。尺寸变大后，声谱图的时间细节和频率细节都更清楚，但显存和计算量也更大：

```python
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)

# 解冻后用更小的学习率，避免把预训练权重破坏掉
learn.fit_one_cycle(5, slice(3.02e-03, lr/5))
learn.save('stage-2-rn50')

# 换成 256×256 的图，再训练 10 轮
data_256 = (src.transform(tfms, size=256)
            .databunch(num_workers=0)
            .normalize(imagenet_stats))
learn.data = data_256
learn.freeze()
learn.fit_one_cycle(5, slice(5e-03))
learn.unfreeze()
learn.fit_one_cycle(5, slice(1.58e-06, lr/5))
learn.fit_one_cycle(5, slice(1.58e-06, lr/5))
```

在 256 尺寸下，验证集 `accuracy_thresh` 从 0.9790 逐步升到 **0.9812**，验证损失从 0.0830 降到 0.0646。提升幅度不大，但方向是稳定的：更大的输入让模型能看清更细的频谱纹理。

超参数方面，我们主要调整了学习率、训练尺寸和冻结/解冻策略。`lr_find` 给出的最小梯度学习率是 2.29e-02（128 尺寸）和 6.31e-07（256 尺寸的解冻阶段），我们据此设置了分层学习率。数据增强只用了不破坏时间轴的变换。

### 2.4 模型测试与指标报告

训练完成后，我们用导出的 learner 对测试集推理：

```python
# 测试集图片单独建一个 ImageList，不需要标签
test = ImageList.from_folder(spect_dir / 'test')
predictor = load_learner('.', test=test, num_workers=0)

# get_preds 返回每个类别的 sigmoid 概率
preds, _ = predictor.get_preds(ds_type=DatasetType.Test)
print(preds.shape)   # (1000, 类别数)
```

在验证集上，最终的 `accuracy_thresh` 是 **0.9812**，验证损失 **0.0646**。需要提醒的是，这个准确率是在阈值 0.2 下算的，而且验证集里大量样本的标签全是 0，所以它会被“容易猜对的负样本”拉高；官方的 AUC 会更严格地衡量每个类别的排序质量。材料没有给出官方 AUC 的数字，我们只能以验证损失和阈值准确率作为参考。

最后生成提交文件。材料里的提交格式是每行一个“测试文件 + 类别编号”，列名为 `ID` 和 `Probability`：

```python
import itertools

# 去掉第 0 列（背景/无效类），只保留真正的鸟种概率
fpreds = preds[:, 1:].reshape(-1, )

# 每个测试文件对应多个类别，ID 拼成“文件名.wav_classnumber_类别号”
names = [f.stem for f in predictor.data.test_ds.items]
fnames = [x + '.wav_classnumber_' + str(i)
          for x in names for i in range(1, len(data.classes))]

test_df = pd.DataFrame({'ID': fnames, 'Probability': fpreds},
                       columns=['ID', 'Probability'])
test_df.to_csv('submission.csv', index=False)
print(test_df.shape)
```

这里的 `reshape(-1)` 有一个隐含前提：`preds` 的行顺序和 `names` 的顺序一致，列顺序和 `range(1, len(data.classes))` 一致。FastAI 的 `get_preds` 默认保持测试集顺序，所以这个写法可行；但我们在提交前会抽查几个 `ID`，确认文件名和类别号没有错位。

## 三、总结

本文中我们完成了 NIPS 2013 多标签鸟类物种分类的完整流程。我们首先用 EDA 看清了标签结构：一行一个音频、后面是几十个 0/1 类别列，类别分布是典型的长尾，很多类别只有个位数样本。其次我们把 WAV 批量转成梅尔声谱图 PNG，转换时关掉坐标轴、去掉白边，并保持原始采样率。接着我们用 fastai 的多标签数据加载器读入图片和标签，以 ResNet50 为骨干，在 128 和 256 两种尺寸上先冻结后解冻地微调，数据增强特意禁用左右翻转以保护时间轴。最后我们在验证集上把 `accuracy_thresh` 从 0.9792 提升到 0.9812，验证损失降到 0.0646，并把 1000 条测试录音的逐类概率写成了提交文件。

这个比赛让我们体会到“音频当图像做”这条路的威力：一旦转成声谱图，就能直接复用 ImageNet 预训练模型，省掉大量从零训练的成本。但也要小心，图像领域的常规增强（翻转、旋转）在声谱图上可能改变语义，数据增强必须跟着数据的物理含义走。