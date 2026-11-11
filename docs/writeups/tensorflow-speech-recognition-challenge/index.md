# TensorFlow Speech Recognition Challenge

## 一、比赛简介

### 1.1 比赛背景

这是一个用短语音识别关键词的比赛。训练数据放在 `train/audio` 目录中，目录名就是语音类别，文件是 WAV 音频。材料中的代码把类别整理成 `yes`、`no`、`up`、`down`、`left`、`right`、`on`、`off`、`stop`、`go`、`silence` 和 `unknown`。其中，原始目录里还有 `_background_noise_`，它不是一个要识别的单词，而是背景噪声，所以我们把它当作 `silence`。

语音数据和表格数据不一样。我们不能只看一行数字判断类别，而要研究声音随时间变化的波形。一个音频文件可以看成一串采样点，采样率为 16000 时，1 秒声音大约对应 16000 个采样值。材料中的探索还显示，训练目录有 31 个标签目录、总共 64727 条录音；但提交模型使用的是整理后的 12 类。比赛的难点主要有三个：有些录音不到 1 秒，有些文件属于背景噪声或未知词，而且音频长度和音量并不完全一致。

### 1.2 比赛目标

我们要读取训练集中的 WAV 文件，学习声音和关键词之间的关系，然后给测试集中的每个文件预测一个标签。模型的输入不是文件名本身，而是音频波形转换出来的特征。材料中出现了两条路线：一条路线直接把 16000 个采样点送进 TensorFlow 的短时傅里叶变换，再交给卷积网络；另一条路线先生成对数频谱图，把频谱图当成一张图片交给二维 CNN。

训练文件名中带有说话人的标识，材料还按照 `validation_list.txt` 把同一说话人的文件分到验证集，从而减少训练集和验证集之间的说话人重叠。我们最后需要输出测试文件名和预测标签，保存成 `sub.csv` 或 `submission.csv`，供 Kaggle 评分。

### 1.3 评价标准

材料没有在代码中给出完整的官方指标公式，因此我们不额外猜测公式。我们把 Kaggle leaderboard（排行榜）分数作为最终指标，并用本地验证集准确率、分类报告和每类准确率来检查模型。材料标题直接报告了若干结果：轻量 CNN 页面写明 LB 0.74，TensorFlow Estimator 基线页面写明 LB 0.72；这些数字是对应页面的排行榜记录，不代表所有模型都会得到相同分数。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先导入音频处理、数值计算和绘图工具，并读取一个 `yes` 文件。先确认采样率和采样值的形状非常重要，因为后面的窗口长度、频谱图尺寸都依赖采样率。材料中的示例文件采样率是 16000，波形是 `int16` 数组。

```python
import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy import signal
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 我们先读取一个已知类别的文件，确认音频格式再设计后续处理
train_audio_path = '/kaggle/input/tensorflow-speech-recognition-challenge/train/audio'
filename = 'yes/0a7c2a8d_nohash_0.wav'
sample_rate, samples = wavfile.read(join(train_audio_path, filename))

# 输出采样率和波形数组，避免把错误的采样率带入频谱计算
print(sample_rate)
print(samples.shape)
```

接着我们把声音转换成对数频谱图。短时傅里叶变换会把连续声音切成很多小窗口，在每个窗口里计算不同频率的能量。这样得到的图像横轴是时间，纵轴是频率，颜色或数值表示能量。对数变换可以压缩过大的能量差，让弱一些的声音细节也能被看到。

```python
def log_specgram(audio, smaple_rate, window_size=20, step_size=10, eps=1e-10):
    # 20 毫秒窗口和 10 毫秒步长，能同时保留时间和频率变化
    nperseg = int(round(window_size * smaple_rate / 1e3))
    noverlap = int(round(step_size * smaple_rate / 1e3))
    freqs, times, spec = signal.spectrogram(
        audio, fs=smaple_rate, window='hann',
        nperseg=nperseg, noverlap=noverlap, detrend=False)
    # 加 eps 是为了避免能量为 0 时直接取对数产生无穷值
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

freqs, times, spec = log_specgram(samples, sample_rate)
print(freqs.shape, times.shape, spec.shape)

# 画出波形和频谱图，直观看语音在时间和频率上的变化
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.plot(samples)
ax1.set_title('Raw wave of ' + filename)
ax2 = fig.add_subplot(212)
ax2.imshow(spec.T, aspect='auto', origin='lower',
          extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_title('Spectrogram of ' + filename)
plt.show()
```

我们还统计标签目录和文件数量。材料显示目录数量是 31，所有目录中的 WAV 文件一共有 64727 条。这一步让我们知道原始目录并不等于最终类别数，也提醒我们后续必须处理未知词和背景噪声。

```python
dirs = [f for f in os.listdir(train_audio_path) if isdir(join(train_audio_path, f))]
dirs.sort()
print('Number of labels: ' + str(len(dirs)))

# 每个目录分别统计文件数，观察类别是否大小差异明显
number_of_recordings = []
for direct in dirs:
    waves = [f for f in os.listdir(join(train_audio_path, direct))
             if f.endswith('.wav')]
    number_of_recordings.append(len(waves))

# 总量可以帮助我们估计内存和训练时间
print(sum(number_of_recordings))
```

最后，我们查看 MFCC。MFCC 是把声音频谱压缩成更适合表示人类发音特点的一组数字，适合用来观察语音差异。材料还计算了二阶差分，帮助描述频谱变化速度；不过主线 CNN 使用的是频谱图，因此这里把 MFCC 作为探索工具，而不是强行和 CNN 混用。

```python
# 先把频谱转成 Mel 频率尺度，Mel 更接近人耳对频率的感受
S = librosa.feature.melspectrogram(
    samples.astype(float), sr=sample_rate, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)

# 用 13 个 MFCC 系数保存紧凑的语音特征
mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)
print(mfcc.shape, delta2_mfcc.shape)

# 可视化 MFCC，检查特征是否能随时间产生有规律的纹理
librosa.display.specshow(delta2_mfcc)
plt.ylabel('MFCC coeffs')
plt.xlabel('Time')
plt.title('MFCC')
plt.colorbar()
plt.tight_layout()
```

### 2.2 数据清洗

我们其次处理标签。模型最终识别的是 12 个类别，合法标签在材料中由 `POSSIBLE_LABELS` 给出。我们把 `_background_noise_` 改成 `silence`，把不在合法列表中的目录统一改成 `unknown`。这样做的原因是：模型需要固定数量的输出类别，不能让每一个陌生词都产生一个新类别。

```python
import re
from glob import glob

# 只保留比赛主任务中的 12 个输出类别，方便最后做 softmax 分类
POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}

def normalize_label(label):
    # 背景噪声没有具体单词含义，因此统一视作静音
    if label == '_background_noise_':
        return 'silence'
    # 其他不在目标列表的词，统一放入 unknown，避免类别数失控
    if label not in set(POSSIBLE_LABELS):
        return 'unknown'
    return label
```

我们然后使用训练集提供的验证文件列表。文件名中包含说话人 ID，材料用正则表达式取出这个 ID，并按 ID 划分训练集和验证集。这样验证集不只是随机抽取文件，而是尽量模拟没有见过说话人的情况。

```python
def load_data(data_dir):
    # 返回 (类别编号、说话人编号、文件路径) 的训练和验证列表
    pattern = re.compile('(.+/)?(\\w+)/([^_]+)_.+wav')
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))
    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()

    # 先收集验证文件对应的说话人，防止同一人同时出现在两边
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = normalize_label(r.group(2)), r.group(3)
            sample = (name2id[label], uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)
    print('There are {} train and {} val samples'.format(len(train), len(val)))
    return train, val
```

我们还要把所有音频整理为相同长度。轻量 CNN 页面把 1 秒定义为 `L = 16000` 个采样点：短文件在前面补 0，长文件只取一个 1 秒片段；端到端生成器则从较长文件中随机截取 1 秒。统一长度的原因很直接：神经网络的输入张量形状必须一致。

```python
def pad_audio(samples):
    # 长度足够时不改动，避免无意义地改变原始声音
    if len(samples) >= L:
        return samples
    # 短于 1 秒时补 0，让每条记录都能组成同样大小的输入
    return np.pad(samples, pad_width=(L - len(samples), 0),
                  mode='constant', constant_values=(0, 0))

def chop_audio(samples, L=16000, num=20):
    # 背景噪声文件很长，随机切片可以制造多个 1 秒静音样本
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]

def label_transform(labels):
    nlabels = []
    for label in labels:
        # 标签转换必须和训练时的 12 类定义保持一致
        nlabels.append(normalize_label(label))
    return pd.get_dummies(pd.Series(nlabels))
```

端到端方案还特别增加了静音样本。对于普通类别，每个文件产生一个训练片段；对于 `silence`，材料把 `samples_per_file` 设为 20。我们不把这个数字解释成普遍最优值，只把它作为材料中的一个处理策略：静音样本容易被其他类别数量压过，所以适当增加它的出现次数。

```python
def data_generator(data, params, mode='train'):
    def generator():
        if mode == 'train':
            # 每轮打乱顺序，减少固定文件顺序带来的影响
            np.random.shuffle(data)
        for (label_id, uid, fname) in data:
            try:
                _, wav = wavfile.read(fname)
                wav = wav.astype(np.float32) / np.iinfo(np.int16).max
                L = 16000
                if len(wav) < L:
                    continue
                # 静音样本多生成一些，避免模型忽视 silence 类
                samples_per_file = 1 if label_id != name2id['silence'] else 20
                for _ in range(samples_per_file):
                    beg = np.random.randint(0, len(wav) - L) if len(wav) > L else 0
                    yield dict(target=np.int32(label_id), wav=wav[beg:beg + L])
            except Exception as err:
                print(err, label_id, uid, fname)
    return generator
```

### 2.3 模型训练与超参数调优

我们接着建立第一个基线：把音频变成频谱图，再用二维卷积网络处理。卷积层可以理解成许多小窗口，它们在频谱图上寻找局部纹理，例如某个频率范围内的短时间变化。材料中的 CNN 使用输入形状 `(99, 81, 1)`，对应 8000 采样率下生成的频谱图和一个灰度通道。

```python
# 先把 1 秒音频重采样到 8000，减少频谱尺寸和显存占用
new_sample_rate = 8000
labels, fnames = list_wavs_fname(train_data_path)
x_train, y_train = [], []
for label, fname in zip(labels, fnames):
    sample_rate, samples = wavfile.read(os.path.join(train_data_path, label, fname))
    samples = pad_audio(samples)
    n_samples = chop_audio(samples) if len(samples) > 16000 else [samples]
    for samples in n_samples:
        resampled = signal.resample(
            samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        y_train.append(label)
        x_train.append(specgram)

# Conv2D 需要四维输入，因此最后增加通道维度
x_train = np.array(x_train)
x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))
y_train = label_transform(y_train).values
```

然后我们定义轻量 CNN。材料中先做批归一化，再堆叠卷积、池化和 Dropout，最后用全连接层输出 12 个类别。Dropout 的作用是训练时随机关闭一部分连接，降低模型只记住训练样本的风险。

```python
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout
from keras.layers import MaxPooling2D, BatchNormalization

# 频谱图是二维结构，所以我们使用 Conv2D 而不是普通全连接网络
input_shape = (99, 81, 1)
nclass = 12
inp = Input(shape=input_shape)
norm_inp = BatchNormalization()(inp)
img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(norm_inp)
img_1 = Convolution2D(8, kernel_size=2, activation=activations.relu)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(rate=0.2)(img_1)
img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
img_1 = Convolution2D(16, kernel_size=3, activation=activations.relu)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(rate=0.2)(img_1)
img_1 = Flatten()(img_1)
dense_1 = Dense(128, activation=activations.relu)(img_1)
dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)
model = models.Model(inputs=inp, outputs=dense_1)
```

我们用分层之外的简单随机划分作为材料中的训练示例：验证集占 10%，随机种子为 2017，batch size 为 16，训练 3 个 epoch。轻量 CNN 页面还说明，若要复现 LB 0.74，需要把 epoch 调到 5、把 `chop_audio(num=1000)`，并把卷积层参数加倍；这属于材料报告的调参方向，我们不把它说成必然适合所有环境。

```python
from sklearn.model_selection import train_test_split

# 先留出验证集，用来观察训练是否过拟合
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=0.1, random_state=2017)

# Adam 适合先建立稳定基线，分类损失与 one-hot 标签对应
model.compile(optimizer=optimizers.Adam(),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16,
         validation_data=(x_valid, y_valid),
         epochs=3, shuffle=True, verbose=2)
model.save(os.path.join(model_path, 'cnn.model'))
```

我们还参考端到端 TF Estimator 基线，使用四个卷积阶段，并把最大池化和平均池化的结果混合。材料给出的参数包括 batch size 64、学习率 `1e-3`、Dropout 保留率 0.5、梯度裁剪 15.0。学习率按 10000 步衰减，优化器是带 Nesterov 的 Momentum。这样做的好处是参数集中写在 `params` 中，后续调整更容易。

```python
def baseline(x, params, is_training):
    # 每层通道数逐步增加，让网络从简单纹理学到复杂模式
    x = layers.batch_norm(x, is_training=is_training)
    for i in range(4):
        x = layers.conv2d(x, 16 * (2 ** i), 3, 1,
                          activation_fn=tf.nn.elu,
                          normalizer_fn=layers.batch_norm if params.use_batch_norm else None,
                          normalizer_params={'is_training': is_training})
        x = layers.max_pool2d(x, 2, 2)
    # 同时保留最大响应和平均响应，减少只依赖单个局部峰值
    mpool = tf.reduce_max(x, axis=[1, 2], keep_dims=True)
    apool = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)
    x = 0.5 * (mpool + apool)
    x = layers.conv2d(x, 128, 1, 1, activation_fn=tf.nn.elu)
    x = tf.nn.dropout(x, keep_prob=params.keep_prob if is_training else 1.0)
    logits = layers.conv2d(x, params.num_classes, 1, 1, activation_fn=None)
    return tf.squeeze(logits, [1, 2])

params = dict(seed=2018, batch_size=64, keep_prob=0.5,
              learning_rate=1e-3, clip_gradients=15.0,
              use_batch_norm=True, num_classes=len(POSSIBLE_LABELS))
hparams = tf.contrib.training.HParams(**params)
```

### 2.4 模型测试与指标报告

我们最后把测试声音按训练时完全相同的方式处理：读取、补齐到 1 秒、重采样、生成频谱图、增加通道维度。测试文件太多时不应一次性全部放进内存，因此材料提供了批量生成器，每次只产生一批图片和文件名。

```python
def test_data_generator(batch=16):
    fpaths = glob(os.path.join(test_data_path, '*wav'))
    i = 0
    imgs, fnames = [], []
    for path in fpaths:
        rate, samples = wavfile.read(path)
        samples = pad_audio(samples)
        resampled = signal.resample(
            samples, int(new_sample_rate / rate * samples.shape[0]))
        _, _, specgram = log_specgram(resampled,
                                      sample_rate=new_sample_rate)
        imgs.append(specgram)
        fnames.append(path.split('\\')[-1])
        i += 1
        if i == batch:
            # 每批都补上通道维度，保持和训练输入一致
            yield fnames, np.array(imgs).reshape(tuple(list(np.array(imgs).shape) + [1]))
            i, imgs, fnames = 0, [], []
    if i < batch and imgs:
        arr = np.array(imgs)
        yield fnames, arr.reshape(tuple(list(arr.shape) + [1]))
```

我们加载保存好的模型并预测每一批测试数据。`argmax` 取概率最大的类别，再通过 `label_index` 把类别编号还原成标签名称。这里的关键是训练和测试必须使用同一套标签顺序，否则即使概率计算正确，输出名称也会错位。

```python
from keras.models import load_model

# 加载验证效果最好的模型，避免重新训练后参数发生变化
saved_model = load_model('cnn.model')
index, results = [], []
for fnames, imgs in test_data_generator(batch=32):
    predicts = saved_model.predict(imgs)
    predicts = np.argmax(predicts, axis=1)
    predicts = [label_index[p] for p in predicts]
    index.extend(fnames)
    results.extend(predicts)

# 按材料中的字段名生成提交文件，index 保证与预测结果一一对应
df = pd.DataFrame(columns=['fname', 'label'])
df['fname'] = index
df['label'] = results
df.to_csv(os.path.join(out_path, 'sub.csv'), index=False)
```

本地评估时，我们用保存的模型在验证集上调用 `evaluate`，再输出分类报告和混淆矩阵。分类报告可以看每个关键词分别识别得好不好，混淆矩阵可以看哪些词最容易互相混淆。材料没有给出一份统一的本地准确率表，所以我们只报告材料明确出现的排行榜结果：端到端 TF Estimator 页面标题为 LB 0.72，轻量 CNN 页面标题为 LB 0.74。轻量 CNN 页面还指出，使用更高的训练轮数、更多音频切片、原始 16000 采样率、更深的 CNN 或 RNN，可能继续改进，但它也提醒单纯 CNN 距离 LB 0.88 仍有差距。

```python
from sklearn.metrics import confusion_matrix, classification_report

# 先在验证集上计算损失和准确率，确认模型不是只会输出一个类别
results = saved_model.evaluate(X_test, y_test, batch_size=128)
print('test loss, test acc:', results)

# 把 one-hot 标签和预测编号都还原成类别名称，报告更容易阅读
prediction = saved_model.predict(X_test)
pred = [classes_encoded[np.argmax(p)] for p in prediction]
true_val = [classes_encoded[np.argmax(p)] for p in y_test]
print(classification_report(true_val, pred, target_names=classes_encoded))

# 混淆矩阵用于定位具体的混淆类别，而不只看一个总分
print(confusion_matrix(true_val, pred))
```

## 三、总结

本文中我们完成了 TensorFlow Speech Recognition Challenge 的一条完整入门路线。我们首先读取 WAV 文件，检查 16000 的采样率和波形形状，然后用短时傅里叶变换生成对数频谱图，并用 MFCC 辅助观察语音特征。其次我们把背景噪声归为 `silence`，把不在目标列表中的词归为 `unknown`，再把短音频补到 1 秒、把长音频切成固定长度。接着我们把音频重采样到 8000，生成频谱图，用带批归一化、池化和 Dropout 的轻量 CNN 训练 12 类分类器；同时参考 TF Estimator 基线，理解了批量生成器、STFT、学习率衰减和梯度裁剪的作用。最后我们按相同流程处理测试文件，分批预测并生成提交文件，用验证集的准确率、分类报告和混淆矩阵检查结果。

材料中明确报告的排行榜结果包括轻量 CNN 的 LB 0.74 和端到端 TF Estimator 的 LB 0.72。我们不把这些分数扩展成没有证据的本地指标，而是把它们当作相应方案的参考。这个比赛给我们的主要启发是：声音任务的关键不只是选择网络，还包括统一音频长度、正确处理静音和未知类别、控制内存，以及让训练和测试的特征处理完全一致。