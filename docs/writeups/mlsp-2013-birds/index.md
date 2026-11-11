# MLSP 2013 鸟类分类挑战

## 一、比赛简介

### 1.1 比赛背景

MLSP 2013 Bird Classification Challenge 是一个音频分类比赛。比赛的音频来自野外的自动录音设备，录音里可能同时出现好几种鸟的叫声，也可能只有风声、虫鸣，甚至什么都没有。参赛者要做的，是听一段录音，判断这段录音里出现了哪些鸟。

这件事比想象中难。人耳能分辨的鸟叫，在计算机眼里只是一串按时间排列的振幅数字；同一只鸟在不同距离、不同天气下叫声也会变化；更麻烦的是，一段几分钟的录音里往往只有几秒钟真正有鸟叫，剩下都是背景噪声。传统做法是先把声音变成“声谱图”，再从声谱图里提取统计特征，最后交给分类器。

这份材料来自 Kaggle 上的一个公开 notebook，作者使用了 `librosa` 做音频特征提取，并用 Keras 搭建了一个全连接神经网络。我们按这个思路，把整个比赛流程走一遍。

### 1.2 比赛目标

比赛提供三类文件：

- `species_list.txt`：19 种鸟的编号、四字母代码和英文名，例如 0 号是 Brown Creeper（褐旋木雀），1 号是 Pacific Wren（太平洋鹪鹩）。
- `rec_id2filename.txt`：645 条录音的编号与文件名对应表。
- `rec_labels_test_hidden.txt`：训练标签，格式是“录音编号,物种编号”。有些录音标了多个物种，有些标了 `?` 表示未公开，标 `-1` 表示这段录音里没有目标鸟。

我们要为每条测试录音、每个鸟种输出一个概率，表示“这段录音里出现该鸟种的可能性”。训练集有 322 条带特征和标签的录音，测试集有 323 条。

### 1.3 评价标准

比赛使用 ROC 曲线下面积（AUC）作为指标，对 19 个鸟种分别计算再取平均。AUC 可以理解为：随机抽一条“有这种鸟”的录音和一条“没有这种鸟”的录音，模型给前者的概率高于后者的概率。AUC 越高越好，0.5 相当于瞎猜，1.0 是完美排序。因为是多标签问题，一条录音可以同时对应多个鸟种，所以不能只看一个总体准确率。

## 二、解题过程

我们的主线是：先把 WAV 变成 193 维的音频特征向量，再用一个带 Dropout 的全连接网络做多分类，最后把 softmax 概率整理成提交格式。下面按 EDA、清洗、训练、测试四步展开。

### 2.1 数据探索（EDA）

我们首先把物种表和录音表读进来，确认类别数量与文件命名规则：

```python
import pandas as pd
import numpy as np

# 物种表：class_id 是模型要预测的编号
species = pd.read_csv('../input/data-burung/essential_data/species_list.txt')
print(species.shape)          # (19, 4)，共 19 种鸟
print(species.head())

# 录音编号与文件名对应表
file_name = pd.read_csv('../input/data-burung/essential_data/rec_id2filename.txt')
print(file_name.shape)        # (645, 2)，645 条录音
print(file_name.head())
```

物种表告诉我们一共有 19 个目标类别。录音表则有 645 行，文件名类似 `PC1_20090513_050000_0010`，其中前半段是录音设备编号和日期，后半段是当天的第几次录音。文件名本身不含鸟种信息，不能直接拿来当特征。

接着我们看标签文件。它的每一行不是标准的 CSV，而是用分号分隔、逗号再分隔的多标签格式：

```python
# 标签文件用分号分隔每条记录，第一个字段是录音编号，后面是物种编号
raw = pd.read_csv('../input/data-burung/essential_data/rec_labels_test_hidden.txt', sep=';')

id_ls, label_ls = [], []
for i in range(len(raw)):
    labels = raw.iloc[i, 0].split(',')   # 例如 "12,3,7"
    id_ls.append(labels.pop(0))          # 第一个是录音编号
    label_ls.append(labels[0] if len(labels) else -1)  # 没有标签就记 -1

label_df = pd.DataFrame({'rec_id': id_ls, 'label': label_ls})
print(label_df['label'].value_counts(normalize=True))
```

统计结果很说明问题：约 50.1% 的记录标签是 `?`，22.2% 是 `-1`，剩下才是具体鸟种，而且单个鸟种的占比大多不到 5%。也就是说，**有明确物种标签的样本非常少，类别极不均衡**。这决定了我们不能简单追求总体准确率，否则模型把所有样本都猜成“没有鸟”也能得到很高的数字。

我们还想知道音频本身长什么样，于是随机挑一条录音画波形和声谱图：

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

wav_path = '../input/data-burung/essential_data/src_wavs/PC10_20090513_054500_0010.wav'
data, sample_rate = librosa.load(wav_path)

print(sample_rate)   # 22050，所有录音统一到 22.05 kHz
plt.figure(figsize=(14, 5))
librosa.display.waveplot(data, sr=sample_rate)
plt.show()
```

波形图显示，鸟叫只集中在少数时间段，大部分时间振幅很低。这提醒我们：如果直接把整段音频压成一个平均值，鸟叫的瞬间信息会被大量静音稀释。后面的特征提取之所以要对每一帧取统计量，就是为了在保留时间变化的同时把长度不一的音频压成固定维度。

### 2.2 数据清洗

这份数据的“脏”主要体现在标签，而不是音频。我们做了三件事。

第一，把标签统一成整数。`?` 表示官方未公开，不能当成一个鸟种；`-1` 表示这段录音没有目标鸟。我们把 `?` 从训练集中剔除，把 `-1` 保留为一个“无鸟”类别，最后用 `LabelEncoder` 重新编号：

```python
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

# 只保留有明确标签的录音用于训练
train_df = label_df[label_df['label'] != '?'].copy()
train_df['label'] = train_df['label'].astype(int)

# 把 -1 和 0~18 统一映射到 0~19，共 20 个输出单元
lb = LabelEncoder()
y = to_categorical(lb.fit_transform(train_df['label']))
print(y.shape)   # (322, 20)
```

这里用 20 维而不是 19 维，是因为 `-1`（无鸟）也占一个位置；多出来的一个单元让网络可以明确表示“这段录音没有目标鸟”，而不是被迫在 19 种鸟里硬选一个。

第二，统一采样率。`librosa.load` 默认会把音频重采样到 22050 Hz，这样所有录音的特征尺度一致，不会因为设备不同导致频率轴对不上。

第三，检查缺失与长度。材料里的 645 条录音都能正常读取，没有出现文件损坏；但长度差异很大，从几秒到几分钟都有。我们没有做截断或补零，而是用“对时间轴取平均”的方式把变长音频压成定长向量，这样就不需要人为对齐长度。

### 2.3 模型训练与超参数调优

我们接着做特征提取。核心思路是把音频切成很多短帧，对每一帧算多种声学特征，再对时间轴求平均。材料里用了五组特征：

```python
def extract_features(filename):
    # 读取音频；kaiser_fast 是一种较快的重采样窗
    X, sample_rate = librosa.load(
        '../input/data-burung/essential_data/src_wavs/' + filename + '.wav',
        res_type='kaiser_fast'
    )

    # 1) 40 维梅尔频率倒谱系数（MFCC），描述音色
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    # 2) 12 维色度特征，描述音高类别
    stft = np.abs(librosa.stft(X))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # 3) 128 维梅尔谱，描述能量在不同频率上的分布
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # 4) 7 维频谱对比度，描述谱峰与谱谷的差距
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # 5) 6 维 tonnetz，描述和声关系
    tonnetz = np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

    return mfccs, chroma, mel, contrast, tonnetz
```

五组特征的长度分别是 40、12、128、7、6，加起来正好 193 维。我们把它们首尾拼接：

```python
features_label = bird_train_df['filename'].apply(extract_features)
features_test = bird_test_df['filename'].apply(extract_features)

# 把五段特征拼成一个 193 维向量
X = np.array([np.concatenate(f[:5]) for f in features_label])
X_test = np.array([np.concatenate(f[:5]) for f in features_test])
print(X.shape)        # (322, 193)
print(X_test.shape)   # (323, 193)
```

因为不同特征的量纲差别很大（MFCC 可能是几百，tonnetz 只有个位数），我们做了标准化。注意标准化器只在训练集上拟合，再套用到验证集和测试集，避免把测试集的分布信息泄漏进来：

```python
from sklearn.preprocessing import StandardScaler

# 按时间顺序切分：前 257 条训练，后 65 条验证
X_train, y_train = X[:257], y[:257]
X_val, y_val = X[257:], y[257:]

ss = StandardScaler()
X_train = ss.fit_transform(X_train)   # 只在训练集上学习均值和方差
X_val = ss.transform(X_val)
X_test = ss.transform(X_test)
```

然后是网络结构。我们用一个四层全连接网络：输入 193 维，两个 128 维隐藏层，输出 20 维 softmax。Dropout 从 0.1 逐步加大到 0.5，目的是防止小样本过拟合：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(193, input_shape=(193,), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))   # 20 类：19 种鸟 + 无鸟

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')

# 验证损失长期不降就提前停，省得白跑
early_stop = EarlyStopping(monitor='val_loss', patience=100, mode='auto')
```

超参数方面，我们主要调整了 Dropout 比例和网络宽度。材料里的设置是 193-128-128-20，参数量不大，配合 0.5 的 Dropout 足以应付 322 条样本。批次大小设为 5，训练 10 轮；学习率用 Adam 的默认值。我们没有做大规模网格搜索，因为样本太少，搜索出来的“最优”很可能只是验证集的噪声。

### 2.4 模型测试与指标报告

训练过程如下：

```python
history = model.fit(X_train, y_train,
                    batch_size=5,
                    epochs=10,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop])
```

10 轮里，训练准确率从 0.315 升到 0.619，验证准确率最高到 **0.585**（第 6、7、8 轮），之后验证损失开始回升。这个数字不算高，但考虑到只有 257 条训练样本、20 个类别，而且验证集里还有大量 `-1` 和稀有鸟种，能到 0.58 已经说明特征里确实有可分的信息。

我们还在验证集上看了损失曲线：训练损失一路下降，验证损失在第 6 轮降到 1.677 后反弹到 1.730。这是典型的过拟合信号，也是我们加大 Dropout、加早停的原因。

最后生成提交文件。比赛要求三列：`rec_id`、`species`、`probability`。我们先对测试集做预测，再把 20 维概率按物种编号展开：

```python
# 对测试集预测，得到每条录音在 20 个类别上的概率
preds = model.predict(X_test)

submission = pd.read_csv('../input/data-burung/essential_data/sample_submission.csv')
# 按 sample_submission 的行顺序，把模型概率填进去
for i, row in submission.iterrows():
    rec = int(row['rec_id'])
    sp = int(row['species'])
    submission.loc[i, 'probability'] = preds[rec, sp]

# 比赛要求 id = rec_id * 100 + species
submission.insert(0, 'id', submission['rec_id'] * 100 + submission['species'])
submission = submission.drop(['rec_id', 'species'], axis=1)
submission.to_csv('submission.csv', index=False)
```

这里有一个容易忽略的细节：`preds` 的行号是测试集在 `bird_test_df` 里的顺序，不一定和 `sample_submission` 的 `rec_id` 一一对应。材料里的写法是直接用 `rec_id` 当下标，前提是测试集按 `rec_id` 排序且从 0 开始。我们在实际提交前会额外检查行数和 `id` 是否唯一，避免因为顺序错位导致分数异常。

## 三、总结

本文中我们完成了 MLSP 2013 鸟类分类挑战的一次完整尝试。我们首先用 EDA 看清了数据：645 条录音、19 种鸟、标签里有一半是未公开的 `?`，明确的物种标签只占少数，类别极不均衡。其次我们做了针对性的清洗：剔除 `?`、把 `-1` 保留为“无鸟”类、统一采样率，并把变长音频压成定长特征。接着我们用 MFCC、色度、梅尔谱、频谱对比度和 tonnetz 五组特征拼成 193 维向量，标准化后喂给一个 193-128-128-20 的全连接网络，用 Dropout 和早停控制过拟合。最后我们在验证集上得到约 0.585 的准确率，并把 softmax 概率整理成比赛要求的提交格式。

这个比赛让我们体会到音频任务的第一个门槛不是模型，而是**如何把声音变成表格**。同一段鸟叫，可以用 MFCC 描述音色，用梅尔谱描述能量分布，用 chroma 描述音高；这些特征各有侧重，拼在一起才比较稳。另一个体会是，多标签、强不均衡的数据不能只看准确率，AUC 和按类别拆开的指标才能反映模型到底学到了什么。
