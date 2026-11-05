# Freesound Audio Tagging

## 一、比赛简介

### 1.1 比赛背景

Freesound Audio Tagging 是一个声音分类比赛。数据来自音频文件，训练表中给出文件名、声音标签以及标签是否经过人工核验。声音类别既有乐器，也有日常环境声，例如 `Hi-hat`、`Saxophone`、`Trumpet`、`Laughter`、`Bark` 和 `Fireworks`。材料中的训练表有 9473 条记录，标签一共有 41 类。

这个任务的难点不只是“把声音变成数字”。第一，音频时长不一致，有的片段很短，有的片段很长；第二，录音的开头和结尾可能包含静音；第三，训练数据中既有人工核验的清晰样本，也有标签较弱甚至不可靠的样本；第四，声音的重要内容不一定出现在同一个时间位置。材料中的解决方案因此同时考虑了数据清洗、不同时间范围的声谱特征，以及多个模型结果的集成。

### 1.2 比赛目标

我们需要读取 `audio_train` 和 `audio_test` 中的声音文件。训练表 `train.csv` 的主要列是 `fname`、`label` 和 `manually_verified`；测试表 `sample_submission.csv` 提供待预测文件名。我们的目标是为每个测试音频输出最可能的标签排序，提交文件中的标签使用空格连接。

为了让神经网络能够处理声音，我们先把波形转换成梅尔频谱图。梅尔频谱图可以理解为一张“声音图片”：横轴表示时间，纵轴表示人耳更容易理解的频率，像素值表示这一时刻的能量。这样，二维卷积网络就可以像识别图片一样识别声音模式。

### 1.3 评价标准

比赛使用 Mean Average Precision，简称 MAP。每条音频提交排名最高的三个标签：如果正确标签排在第一位，得分最高；排在第二或第三位仍然得分，但分数更低。材料中的示例把三个位置的权重写成 `11/18`、`5/18`、`2/18`，如果只保留两个或一个预测，则分别使用 `3/4、1/4` 或 `1` 的权重。最后对所有音频的得分取平均，MAP 越高越好。因此模型不只要猜中，还要尽量把正确标签排在前三名的前面。

## 二、解题过程

### 2.1 数据探索（EDA）

```python
import os
import numpy as np
import pandas as pd

# 先读取表格和文件列表，确认比赛数据是否能一一对应
train = pd.read_csv('../input/freesound-audio-tagging/train.csv')
test = pd.read_csv('../input/freesound-audio-tagging/sample_submission.csv')
print(train.shape)
print(train.label.nunique())
print(train.manually_verified.value_counts(normalize=True))
print(os.listdir('../input'))
```

我们首先查看表格形状、标签数量和文件目录。训练表的形状是 `(9473, 3)`，标签数量是 41。`manually_verified` 的统计结果显示，值为 1 的人工核验样本约占 39.16%，其余约占 60.84%。这说明训练数据中同时存在可信度不同的样本，后面不能把所有记录简单地看成同样干净。

接着我们随机播放音频，并观察波形和频谱图。材料中的例子显示，某个声音片段在 22050 采样率下长度为 22932 个采样点，另一个片段长度为 125685 个采样点；对应的梅尔频谱图时间宽度也不同。这种差异说明网络输入必须统一大小。

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 播放和绘图是为了确认音频确实被正确读取，而不是直接相信文件名
signal, sr = librosa.load('../input/freesound-audio-tagging/audio_train/0048fd00.wav')
mel = librosa.feature.melspectrogram(y=signal, sr=sr)
mel_db = librosa.power_to_db(mel, ref=np.max)
print(signal.shape, sr, mel_db.shape)
librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.show()
```

我们还发现不同类别的样本数量并不均衡。少数类别很难仅靠随机切分得到稳定的验证结果，所以训练阶段需要做类别平衡。另一方面，声音有时在片段的开头就包含主要信息，有时重要内容出现在中间或后面，这直接影响后面两套特征方案的设计。

### 2.2 数据清洗

```python
import numpy as np
import librosa

# 先删除经过人工检查、确认会伤害训练的异常样本
train_blacklist_index = np.load(EXTRA / 'train_blacklist.npy')
print('Blacklist has %d items.' % len(train_blacklist_index))

# 读取音频时去掉两端静音，减少无信息区域对模型的干扰
def read_audio(conf, pathname):
    y, sr = librosa.load(pathname, sr=conf['sampling_rate'])
    if 0 < len(y):
        y, _ = librosa.effects.trim(y)
    return y
```

我们采用“先训练、再检查错误样本”的办法清洗数据。模型预测失败后，我们逐条检查声音，把静音、音质很差、内容过于混乱或与标签无关的样本放入黑名单。材料给出的黑名单共有 59 条记录。训练时只使用不在黑名单中的样本，这比盲目删除大量数据更稳妥。

其次，我们把音频变成统一长度。如果音频太长，方案 LH 取开头部分；如果音频不够长，就在两端补零。方案 X 则保留更长的声音，并把长的梅尔频谱按窗口切成多个片段，同时记录 `index_map`，这样预测后还能把同一个原始音频的多个窗口合并起来。

```python
import numpy as np
import librosa

def audio_to_melspectrogram(conf, audio):
    # 梅尔频谱保留人耳更有意义的频率结构，功率转分贝后更适合网络学习
    spectrogram = librosa.feature.melspectrogram(
        audio, sr=conf['sampling_rate'], n_mels=conf['n_mels'],
        hop_length=conf['hop_length'], n_fft=conf['n_fft'],
        fmin=conf['fmin'], fmax=conf['fmax'])
    return librosa.power_to_db(spectrogram).astype(np.float32)
```

### 2.3 模型训练与超参数调优

我们首先准备两种互补的输入。LH 使用 44100 Hz、4 秒、128 个梅尔频带和 882 的 hop length，重点观察声音开头的高分辨率细节。X 使用 26000 Hz、6 秒、48 个梅尔频带和 520 的 hop length，尽量覆盖完整音频，但采用更粗的频率分辨率。这样做是因为材料中的观察表明：很多样本的有效信息位于开头，但只看开头又会漏掉后半段的关键声音。

```python
# LH：高分辨率，集中观察声音开头
confLH['sampling_rate'] = 44100
confLH['duration'] = 4
confLH['hop_length'] = 882
confLH['n_mels'] = 128
confLH['audio_split'] = 'head'

# X：覆盖更长音频，允许后续切分并合并结果
confX['sampling_rate'] = 26000
confX['duration'] = 6
confX['hop_length'] = 520
confX['n_mels'] = 48
confX['audio_split'] = 'dont_crop'
```

模型使用 AlexNet 风格的二维卷积网络。我们先用较大的卷积核提取局部声音纹理，再用多层 3×3 卷积继续组合特征，最后经过全连接层输出 41 类的 softmax 概率。Dropout 用来降低模型记住训练样本的风险。

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(48, 11, input_shape=input_shape, strides=(2, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(3, strides=(1, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, 5, strides=(2, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(3, strides=2))
model.add(BatchNormalization())
model.add(Conv2D(192, 3, activation='relu', padding='same'))
model.add(Conv2D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling2D(3, strides=(1, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
```

我们其次处理类别不均衡。每个交叉验证折中，先用随机过采样把训练集各类别补到当前最大类别的数量，再使用 `MixupGenerator` 把两条频谱和对应标签按比例混合。生成器还可以配合随机擦除、横向平移和翻转，让模型不依赖某个固定时间位置。正式配置使用 5 折、批大小 32、学习率 0.0001，最多训练 200 轮；每折保存验证损失最好的权重。

```python
from imblearn.over_sampling import RandomOverSampler

# 按类别补齐样本数量，避免大类别主导损失函数
Xidx = [[i] for i in range(len(_Xtrain))]
y_cls = [np.argmax(one) for one in _ytrain]
nsamples = max(y_cls.count(cls) for cls in set(y_cls))
flat_ratio = {cls: nsamples for cls in set(y_cls)}
Xidx_resampled, y_resampled = RandomOverSampler(
    ratio=flat_ratio, random_state=42).fit_sample(Xidx, y_cls)

# Mixup 让模型看到连续的标签组合，降低过拟合
train_generator = MixupGenerator(
    _Xtrain, _ytrain, alpha=1.0, batch_size=32,
    datagen=aug_datagen)()
```

### 2.4 模型测试与指标报告

我们最后对每个折的测试窗口进行预测。一个原始音频可能对应多个窗口，所以先按照 `idx_map` 找回原始文件，再对同一文件的多个概率向量计算几何平均。几何平均能让多个窗口共同支持的类别保持较高概率，同时减少某一个窗口偶然误判的影响。

```python
def geometric_mean_preds(_preds):
    # 同一原始音频的多个窗口必须共同参与，不能只取某一个窗口
    preds = _preds.copy()
    for i in range(1, preds.shape[0]):
        preds[0] = np.multiply(preds[0], preds[i])
    return preds[0]

def get_unified_preds(preds, pred_idx, N):
    # 按原始音频编号合并窗口，恢复到提交所需的一行一个音频
    results = []
    for idx in range(N):
        this_preds = preds[np.where(pred_idx == idx)]
        results.append(geometric_mean_preds(this_preds))
    return np.array(results)
```

接着，我们对每个样本取概率最高的三个类别，并按 MAP@3 的规则生成提交结果。材料中的最佳方案把 LH 和 X 看作互补：LH 擅长开头的细节，X 擅长覆盖完整音频。最终对不同折、不同窗口得到的概率做几何均值集成。材料报告的最终结果为 public 0.950、private 0.917；训练集的人工核验样本准确率在不同折中约为 0.963 到 0.981，集成后的示例为 0.981。

## 三、总结

本文中我们完成了 Freesound Audio Tagging 的一条完整解题路线。我们首先检查了 9473 条训练记录、41 个声音类别和人工核验字段，其次通过静音裁剪、统一长度和 59 条黑名单清理不可靠音频，接着把波形转换为梅尔频谱图，用 LH 与 X 两种时间范围互补的方案训练卷积网络，并通过过采样、Mixup、随机擦除和 5 折训练改善泛化，最后把多个窗口和多个折的概率用几何均值合并，按前三名标签计算 MAP@3。

这个比赛说明，音频分类的效果不仅取决于网络层数，也取决于我们怎样保留时间信息、怎样处理脏数据，以及怎样把同一个声音的多个观察结果组合起来。