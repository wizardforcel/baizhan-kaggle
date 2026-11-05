# Freesound 音频标签竞赛

## 一、比赛简介

### 1.1 比赛背景
Freesound 是一家存放创意声音片段的开源社区。2018 年的 Freesound Audio Tagging 比赛提供了一批真实录音，让参赛者根据音频内容猜出它属于哪几个音乐/音效标签，比如 `Hi-hat`、`Violin_or_fiddle`、`Applause`、`Squeak` 等。比赛共 41 个标签，训练集 9473 条音频，测试集 9400 条，每条音频时长不等，部分样本经过人工核验（`manually_verified`）。

音频标签任务属于多标签分类：一段声音可能同时属于多个类别。难点在于标签分布不均衡——有的标签出现几百次，有的仅十几次；音频质量参差，有静音、噪音、录音切得太短等「脏数据」。该比赛使用的评价指标是 lwlrap（label-weighted label-ranking average precision），简单说就是先对每个样本的标签做排序，再按标签出现频率加权求平均精度，值越高越好，满分 1.0。

### 1.2 比赛目标
给定训练集 `train.csv`，每条记录包含 `fname`（文件名）和 `label`（逗号分隔的标签组成的字符串），以及 `manually_verified` 标记。测试集只有文件名，需要我们输出每个文件的多标签预测。提交格式是 `fname,label`，其中 label 为空格连接的标签名。

### 1.3 评价标准
官方使用 lwlrap。直白讲，它先对每个样本的标签按模型给出的置信度排序，计算标签排序的平均精度，再按标签在训练集中的出现次数加权平均。加权是为了让稀有标签也被重视。分数越高越好。

## 二、解题过程

我们首先做数据探索，其次做清洗，接着用声谱+CNN训练并调整参数，最后用集成预测并报告指标。

### 2.1 数据探索（EDA）

```python
import pandas as pd, glob, os
train = pd.read_csv('../input/train.csv')
print(train.shape)
print(train.label.nunique())
print(train.manually_verified.value_counts())
# 输出：(9473, 3)  41类  verified 为 1 的样本约 15 %
```

从文件列表可知训练音频 9473 个，测试 9400 个，标签类别 41 个。手动核验样本约占 15%，这些样本标签更可信。我们还随机采样播放几条，发现很多样本开头结尾有沉默，长短差异很大。

### 2.2 数据清洗

```python
# 黑名单处理
train_blacklist_index = np.load(EXTRA / 'train_blacklist.npy')
print('Blacklist has %d items.' % len(train_blacklist_index))
```

我们在训练中发现 59 条不可靠的音频：静音、音质极差或标签明显错误。采用「先建模→预测错误样本并手工检查→加入黑名单→重新训练」的迭代方式，最终保留一份 59 项的黑名单并在建模时移除。

音频读取时先用 librosa 的 trim 去除前后静音：

```python
y, sr = librosa.load(pathname, sr=conf['sampling_rate'])
y, _ = librosa.effects.trim(y)
```

这样可以避免静音段带入声谱。长音频会截取或填充到统一长度，保证输入尺寸一致。

### 2.3 模型训练与超参数调优

我们采用两种声谱方案并行，最后集成。方案 LH 用 44100 Hz、4 秒、128 mel、hop 882（20 ms），取音频开头；方案 X 用 26000 Hz、6 秒、48 mel、hop 520，整段不裁剪。

```python
# 方案配置
confLH['sampling_rate'] = 44100
confLH['duration'] = 4
confLH['hop_length'] = 882
confLH['n_mels'] = 128

confX['sampling_rate'] = 26000
confX['duration'] = 6
confX['hop_length'] = 520
confX['n_mels'] = 48
```

声谱转换：

```python
spec = librosa.feature.melspectrogram(audio,
    sr=conf['sampling_rate'],
    n_mels=conf['n_mels'],
    hop_length=conf['hop_length'])
spec = librosa.power_to_db(spec).astype(np.float32)
```

模型使用 AlexNet 风格的 2D 卷积：

```python
model = Sequential()
model.add(Conv2D(48, 11, input_shape=input_shape, strides=(2,3), activation='relu', padding='same'))
model.add(MaxPooling2D(3, strides=(1,2)))
model.add(BatchNormalization())
model.add(Conv2D(128, 5, strides=(2,3), activation='relu', padding='same'))
model.add(MaxPooling2D(3, strides=2))
model.add(BatchNormalization())
model.add(Conv2D(192, 3, strides=(1,2), activation='relu', padding='same'))
model.add(Conv2D(192, 3, strides=(1,1), activation='relu', padding='same'))
model.add(Conv2D(128, 3, strides=(1,1), activation='relu', padding='same'))
model.add(MaxPooling2D(3, strides=(1,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
```

训练使用 Mixup 数据增强：

```python
train_generator = MixupGenerator(_Xtrain, _ytrain, alpha=1.0, batch_size=conf['batch_size'])
```

并对类别做过采样平衡，使每类样本数大致相等。采用 5 折交叉验证，每个折批大小 32，迭代约 200 轮，使用 Adam 优化器和 categorical crossentropy 损失。

### 2.4 模型测试与指标报告

每个折训练完成后对验证集和测试集分别预测，单折验证集在人工核验样本上的准确率约 0.96~0.98。测试集预测采用几何均值集成：

```python
def pred_geometric_mean(preds_set):
    predictions = np.ones_like(preds_set[0])
    for p in preds_set:
        predictions *= p
    predictions = predictions ** (1. / len(preds_set))
    return predictions
```

方案 LH 的优势在样本开头信息丰富，方案 X 覆盖整段音频，两者互补。最终集成在公开榜获得约 0.950，私有榜约 0.917。

## 三、总结

本文中我们完成了 Freesound 音频标签竞赛的完整流程。我们首先通过 EDA 发现 9473 条训练音频、41 个标签且分布不均，并手工剔除了 59 条脏数据；其次对音频做静音裁剪和 mel 声谱转换，设计了 LH 高分辨率短时和 X 低分辨率长时两种特征；接着用 Mixup 与过采样训练了 5 折 CNN，并在人工核验集上得到约 0.98 的准确率；最后用几何均值集成两套方案，在排行榜上取得约 public 0.950 / private 0.917 的成绩。

这个比赛提醒我们：音频任务里特征的时间截取策略和数据清洗同样关键，同一套声谱不同参数配合集成，能明显提升多标签音频的识别效果。
