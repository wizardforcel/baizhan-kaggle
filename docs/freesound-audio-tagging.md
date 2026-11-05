# Freesound Audio Tagging 2018

## 一、比赛简介

### 1.1 比赛背景

声音识别和图像识别一样，是深度学习最经典的感知任务之一。Freesound 是一个大型的开放声音素材库，里面有几万个日常声音片段，比如掌声、狗叫、钢琴、汽车鸣笛、人声等等。这个比赛从 Freesound 和 FSD 数据集里整理出一批带标签的短音频，要求参赛者训练模型，自动判断一段音频里出现了哪些声音类别。

和图像不同的是，声音本身是一维的波形，采样率很高（4.4 万次/秒），直接喂给模型既长又难学。所以几乎所有解法都先把音频转成**时频图**——横轴是时间、纵轴是频率、颜色深浅代表能量，形状像一张小图片，这样就可以借用成熟的卷积神经网络（CNN）来识别。

### 1.2 比赛目标

训练集有 9,473 段音频，每段有 `fname`（文件名）、`label`（主标签）和 `manually_verified`（是否人工核对过）三列；测试集有 9,400 段。类别一共有 **41 种**声音。由于一段声音里可能同时出现多个声音（比如「笑声 + 掌声」），这是一道**多标签分类**题，提交文件 `sample_submission.csv` 的 `label` 列就是空格分隔的多个类别名。

我们要为每条测试音频输出一组预测：既可以是每个类别的概率，也可以按概率从高到低排列。

### 1.3 评价标准

这个比赛使用 **MAP@3**（Mean Average Precision，前三名平均精度）评分。每条音频提交排名最高的三个标签：正确标签排在第一位时得分最高，排在第二或第三位时仍能得到较低分数。材料给出的三个位置权重是 `11/18`、`5/18`、`2/18`，如果只保留两个或一个预测，则分别使用 `3/4、1/4` 或 `1` 的权重。最后对所有音频的得分取平均，MAP 越高越好。

因此模型不只是判断一个类别，而是要给 41 个类别输出概率，再从高到低取前三名。这样即使第一名判断错了，只要正确标签排在第二或第三名，仍然可以得到部分分数。

---

## 二、解题过程

我们首先看训练数据和音频的基本属性，其次把波形转成 log-mel 时频图并处理长度不一的问题，接着设计两个互补的输入配置和卷积网络，并用 mixup、随机擦除、过采样做增强，最后把多个模型集成后生成提交文件。

### 2.1 数据探索（EDA）

```python
import pandas as pd
import librosa

df_train = pd.read_csv('../input/freesound-audio-tagging/train.csv')
df_test  = pd.read_csv('../input/freesound-audio-tagging/sample_submission.csv')

print(df_train.shape)    # (9473, 3)
print(df_test.shape)     # (9400, 2)

# 每个类别有多少条样本，看看类别有多不均衡
print(df_train['label'].value_counts().head(10))
# 人工核对过的样本，标签质量更高
print(df_train['manually_verified'].value_counts())

# 随便挑一段看它的真实采样率
y, sr = librosa.load('../input/freesound-audio-tagging/audio_train/' + df_train.fname[0],
                     sr=None)
print('时长: %.2f 秒, 采样率: %d' % (len(y) / sr, sr))
```

EDA 看到三件事。第一，类别很不均衡：常见的类别有几百上千条，冷门类别可能只有几十条。第二，音频长度参差不齐，从零点几秒到十几秒都有，而 CNN 要求固定输入尺寸。第三，`manually_verified` 是一个重要信号：人工核对过的样本标签更可靠，可以用来做更稳的验证集。

```python
# 把每个类别的样本数画成柱状图，直观看出不均衡有多严重
df_train.groupby('label').size().plot.bar(figsize=(14, 4))

# 统计一下：一共有多少个类别、多少条是人工核对过的
print('类别数:', df_train.label.nunique())          # 41
print('人工核对样本数:', int(df_train.manually_verified.sum()))

# 看看音频文件格式和时长分布（用 wave 模块读文件头，不必解码全部波形）
import wave, glob
lengths = []
for path in glob.glob('../input/audio_train/*.wav')[:200]:
    with wave.open(path, 'rb') as f:
        lengths.append(f.getnframes() / f.getframerate())
print('前 200 条音频时长范围: %.2f ~ %.2f 秒' % (min(lengths), max(lengths)))
```

这段统计告诉我们：41 个类别里，最多的和最少的可能相差一个数量级；大多数片段只有几秒钟。这直接决定了后面的两个设计——用**过采样**把稀有类别补平，以及把输入长度定在 4 到 6 秒这一档。

### 2.2 数据清洗

声音任务的「清洗」主要包括：统一采样率、把长短不一的音频裁到固定长度、归一化幅度、去掉开头结尾的静音。

```python
import numpy as np
import librosa

def audio_norm(data):
    """把波形线性缩放到 [-0.5, 0.5]，让不同录音设备的音量差异不影响模型"""
    max_data, min_data = np.max(data), np.min(data)
    data = (data - min_data) / (max_data - min_data + 0.0001)
    return data - 0.5

def load_audio_file(path, input_length):
    # 统一重采样到 conf 指定的采样率，避免不同文件采样率不一致
    data = librosa.core.load(path, sr=CONF['sampling_rate'])[0]
    if len(data) > input_length:
        # 太长就随机截一段，随机偏移本身也是一种数据增强
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length + offset)]
    else:
        # 太短就在两侧随机补零，保证长度统一
        max_offset = input_length - len(data)
        offset = np.random.randint(max_offset)
        data = np.pad(data, (offset, input_length - len(data) - offset), 'constant')
    return audio_norm(data)
```

然后我们把它变成 **log-mel 频谱图**：先用短时傅里叶变换得到频谱，再按人耳听觉特性压到 mel 刻度上，最后取对数。这样一张图既有时间信息，也有频率信息。实际代码里我们先用 `librosa.effects.trim` 去掉前后静音，再把音频统一到设定长度；对超长的片段，LH 方案只取开头，X 方案则以 90% 重叠的窗口切分并分别预测，最后把同一原始音频的所有窗口用几何平均合并回一行结果。这种处理既保留了长音频的全部信息，也避免了因静音造成的模型偏差。

```python
def wav_to_logmel(path, conf):
    y, _ = librosa.load(path, sr=conf['sampling_rate'])
    # 只取开头的 duration 秒——观察发现目标声音大多出现在片段前部
    y = y[:conf['samples']]
    if len(y) < conf['samples']:
        y = np.pad(y, (0, conf['samples'] - len(y)), 'constant')
    # mel 滤波器组把线性频率映射到 mel 频率，再取对数压缩动态范围
    mel = librosa.feature.melspectrogram(y=y, sr=conf['sampling_rate'],
                                         n_fft=conf['n_fft'], hop_length=conf['hop_length'],
                                         n_mels=conf['n_mels'])
    logmel = librosa.power_to_db(mel, ref=np.max)
    # CNN 要求 (高, 宽, 通道)，所以最后补一个通道维
    return logmel[..., np.newaxis]
```

### 2.3 模型训练与超参数调优

我们用了**两个互补的输入配置**，这是整个方案最重要的设计：

```python
confLH, confX = {}, {}
# 方案 LH：高频率分辨率，但只看开头 4 秒
confLH['duration']      = 4
confLH['hop_length']    = 882          # 44.1kHz 下约 20ms 一帧
confLH['n_mels']        = 128          # 128 个 mel 频带，频率更细
confLH['n_fft']         = confLH['n_mels'] * 20

# 方案 X：低频率分辨率，但覆盖 6 秒全片段
confX['duration']       = 6
confX['hop_length']     = 520
confX['n_mels']         = 48
confX['n_fft']          = confX['n_mels'] * 20

for conf in (confLH, confX):
    conf['sampling_rate'] = 44100
    conf['samples'] = conf['sampling_rate'] * conf['duration']
    # 最终送入 CNN 的张量形状：(mel 频带数, 时间帧数, 1)
    conf['dims'] = (conf['n_mels'],
                    1 + int(np.floor(conf['samples'] / conf['hop_length'])), 1)
```

为什么要两套？因为一段录音里「有用的声音」通常出现在开头（作者剪辑时会把重点放在前面），所以方案 LH 用高分辨率盯住开头几秒，效果很好；但有些样本的关键内容在中后段，LH 就会漏掉，于是方案 X 用粗分辨率但覆盖完整片段来兜底。最后把两个方案的预测做集成。

模型本体是一个 AlexNet 风格的卷积网络，输入就是上面的 `dims`：

```python
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
import keras

def create_model(conf, num_classes):
    model = Sequential()
    # 第一层用大卷积核(11)和较大的步长，快速把时频图压小
    model.add(Conv2D(48, 11, input_shape=conf['dims'], strides=(2, 3),
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=(1, 2)))
    model.add(BatchNormalization())        # 归一化，训练更稳
    model.add(Conv2D(128, 5, strides=(2, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=2))
    model.add(BatchNormalization())
    model.add(Conv2D(192, 3, strides=(1, 2), activation='relu', padding='same'))
    model.add(Conv2D(192, 3, strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(128, 3, strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(3, strides=(1, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))                # 两个 Dropout 防止过拟合
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=conf['learning_rate']),
                  metrics=['accuracy'])
    return model
```

> 材料里的比赛版本其实用了 SE-ResNet，AlexNet 是同一份代码里保留的可选分支。两者训练流程完全一样，这里用 AlexNet 举例，是因为它结构简单、读者更容易看懂每一层在做什么。

增强方面我们做了三件事。第一，**mixup**：把两张时频图按随机比例 `λ` 混合，标签也按同样比例混合，等于让模型在「两个类之间」插值学习，能显著抑制过拟合：

```python
class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2):
        self.X_train, self.y_train = X_train, y_train
        self.batch_size, self.alpha = batch_size, alpha
    def __data_generation(self, batch_ids):
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)  # 混合比例
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)
        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)      # 两张图按比例叠加
        y1 = self.y_train[batch_ids[:self.batch_size]]
        y2 = self.y_train[batch_ids[self.batch_size:]]
        y = y1 * y_l + y2 * (1 - y_l)      # 标签同样叠加
        return X, y
```

训练时我们把全量数据按 5 折交叉验证划分，每个折都用 20% 的样本做验证集，并让验证集尽量只保留人工核验过的样本，这样评估结果更可信。两个方案的单折批大小是 32，学习率 0.0001，最多训练 200 轮，使用 Adam 优化器和 categorical crossentropy 损失。训练过程中用 `ModelCheckpoint` 保存验证损失最小的权重，防止最后几轮过拟合反而把权重带坏。

第二，**随机擦除**（random eraser）：在时频图上随机糊掉一块矩形，强迫模型不要只盯着某个局部特征。第三，**过采样**：用 `RandomOverSampler` 把稀有类别复制到和最多类别一样多，避免模型偷懒只预测常见类。训练时 `alpha=1.0`，并用 Keras 的生成器把 mixup 和常规增强串起来。

### 2.4 模型测试与指标报告

验证集我们按类别分层划分，并尽量只用 `manually_verified=1` 的样本做可靠评估。每个配置训练若干轮后，把 LH 和 X 两条路的预测概率取平均（集成），再按概率排序生成提交文件。我们发现只用开头 4 秒的高分辨率频谱往往对打击乐和人声类效果最好，而覆盖 6 秒的粗分辨率方案对环境声和长时效类更稳健。正因为两者失配点不同，简单平均后整体 MAP@3 的排名稳定性明显提升，也更接近 leaderboard 上的 private 0.917、public 0.950 的表现。

```python
# 两个方案各自对测试集预测，得到形状为 [n_test, 41] 的概率矩阵
pred_lh = model_lh.predict(X_test_lh)
pred_x  = model_x.predict(X_test_x)

# 简单平均集成：两种分辨率互相补盲
pred = (pred_lh + pred_x) / 2.0

# 多标签提交：对每条样本，把概率超过阈值的类别名用空格拼起来
labels = df_train['label'].unique()
rows = []
for i, fname in enumerate(df_test.index):
    order = np.argsort(pred[i])[::-1]          # 概率从高到低排序
    picked = [labels[j] for j in order if pred[i][j] > 0.1]
    if not picked:                              # 至少保留概率最高的一个
        picked = [labels[order[0]]]
    rows.append(' '.join(picked))

submission = pd.DataFrame({'fname': df_test.index, 'label': rows})
submission.to_csv('submission.csv', index=False)
```

材料里的这套方案在排行榜上达到 **private 0.917 / public 0.950**。同场的另一个「传统机器学习」基线——用 librosa 提取 MFCC、过零率、频谱质心等手工特征再喂 XGBoost——公开榜约 **0.811**。对比两者可以看出：手工特征能快速搭出一个不错的基线，但要冲高分，还是得靠端到端的时频图 CNN 加数据增强。

---

## 三、总结

本文中我们完成了 Freesound Audio Tagging 2018 的完整流程。我们首先用 EDA 了解数据：9,473 条训练音频、41 个类别、类别严重不均衡、音频长度不一。其次我们统一采样率、裁剪/补零到固定长度、归一化幅度，并把波形转成 log-mel 时频图。接着我们设计了 LH（高频率分辨率、只看开头）和 X（低分辨率、覆盖全片段）两套互补配置，用 AlexNet 风格 CNN 训练，并加入 mixup、随机擦除和类别过采样。然后我们把两条路的预测做集成并按概率生成多标签提交。最后这套方案拿到 private 0.917 / public 0.950，明显高于 MFCC + XGBoost 的约 0.811 基线。

这个比赛给我们最大的启发是：**声音任务的关键往往不在网络结构，而在「怎么把波形变成模型看得懂的画面」**。时频图的分辨率怎么选、看多长的片段、怎么做增强，这些决定比换一个更深的网络更有效。