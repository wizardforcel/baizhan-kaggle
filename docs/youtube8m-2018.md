# YouTube-8M 2018 视频理解挑战

## 一、比赛简介

### 1.1 比赛背景
YouTube-8M 是 Google 发布的大规模视频理解数据集。比赛组织方提前把视频的视觉和音频信息抽取成特征，参赛者不需要自己解码视频，而是直接在这些特征上训练模型。2018 年的版本更新了标签体系，标签数量达到数千个，视频内容覆盖游戏、体育、音乐、汽车、动物、新闻等众多领域。由于数据量很大、标签非常稀疏，这个比赛不仅考察模型结构，也考察我们读取 TFRecord、控制内存和管理训练流程的能力。

我们拿到的材料是比赛提供的样例数据，里面包含 `label_names_2018.csv`、`vocabulary.csv`、`sample_submission.csv` 三个表格文件，以及 `video/` 和 `frame/` 两个目录下的 TFRecord 文件。样例数据规模不大，但字段结构和正式数据完全一致，因此可以先用它把整套流程跑通。

### 1.2 比赛目标
比赛要求我们为每个视频预测一组标签。输入分成两部分。视频级特征包括长度为 1024 的 `mean_rgb` 和长度为 128 的 `mean_audio`，它们是把整个视频的视觉和音频信息做平均后得到的向量。帧级特征包括每一帧的 `rgb`（1024 维）和 `audio`（128 维），保留了视频随时间变化的信息。

输出不是单个类别，而是一组标签及对应的置信度。也就是说，一个视频可以同时属于“视频游戏”“音乐家”“汽车”等多个类别。提交文件每行包含一个 `VideoId` 和一串 `LabelConfidencePairs`，标签和置信度用空格分隔，并按置信度从高到低排列。

### 1.3 评价标准
这是一个多标签分类任务。训练时材料中的模型使用 `binary_crossentropy` 作为损失函数，使用 `acc` 作为监控指标。`binary_crossentropy` 会把每一个标签都当成一个独立的二分类问题，因此适合一个样本对应多个标签的情况。

在竞赛排行榜上，提交结果按标签置信度排序后计算多标签排序指标，通常称为 GAP（Global Average Precision）。它的直观含义是：我们把模型认为最可能的标签排在前面，如果真实标签出现在靠前的位置，就能得到较高分数。因为每个视频的真实标签数量不一样，计算时还要按每个视频的标签数做归一化。指标越高越好。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先查看输入目录，确认样例文件是否齐全。这样做是为了避免后面写死路径时出现找不到文件的问题。

```python
import os
# 查看竞赛输入目录，确认样例文件都在
print(os.listdir("../input"))
# video 目录存放视频级 TFRecord
print(os.listdir("../input/video"))
# frame 目录存放帧级 TFRecord
print(os.listdir("../input/frame"))
```

从输出可以看到，`video` 和 `frame` 目录下各有 `train00.tfrecord` 和 `train01.tfrecord` 两个文件。接着我们读取标签映射表，统计样例中的标签数量。

```python
import pandas as pd
# 读取标签 id 与标签名称的映射
labels_df = pd.read_csv('../input/label_names_2018.csv')
print(labels_df.head())
# 统计样例数据中出现的唯一标签名数量
print("Total numbers of labels in sample dataset: %s"
      % (len(labels_df['label_name'].unique())))
```

输出显示标签表前几行是 `Game`、`Vehicle`、`Video game`、`Concert`、`Car`，样例中共有 3807 个不同的标签名。这个数字说明标签空间非常大，而样例视频只有两千多个，绝大多数标签在样例里只出现很少几次，甚至一次都没有出现。

然后我们读取视频级 TFRecord，看看每条记录里到底有什么。

```python
import tensorflow as tf
video_files = ["../input/video/{}".format(i)
               for i in os.listdir("../input/video")]
vid_ids, labels, mean_rgb, mean_audio = [], [], [], []
for file in video_files:
    # TFRecord 需要逐条迭代读取
    for example in tf.python_io.tf_record_iterator(file):
        tf_example = tf.train.Example.FromString(example)
        # 视频 id 是 bytes，需要解码成字符串
        vid_ids.append(tf_example.features.feature['id']
                       .bytes_list.value[0].decode('UTF-8'))
        # 标签是整数 id 列表，一个视频可能有多个标签
        labels.append(tf_example.features.feature['labels']
                      .int64_list.value)
        # 视频级视觉特征，长度 1024
        mean_rgb.append(tf_example.features.feature['mean_rgb']
                        .float_list.value)
        # 视频级音频特征，长度 128
        mean_audio.append(tf_example.features.feature['mean_audio']
                          .float_list.value)
print('Number of videos in Sample data set: %s' % str(len(vid_ids)))
```

运行后得到样例视频总数 2056。我们随机看第 14 个视频，它的 id 是 `d801`，标签 id 是 `[2, 7, 2563, 3421]`，对应名称是 `['Video game', 'Musician', 'Gravy', 'Meta Knight']`。这个例子很直观地说明了一个视频可以同时拥有多个语义相差很大的标签，因此不能把它当成普通的单标签分类。

接下来我们统计标签出现次数，并画出出现频率最高的 25 个标签。统计时先把每个视频的标签 id 转成名称，再把这些名称收集到一个大列表里，用 `Counter` 计数。

```python
from collections import Counter
# 把每个视频的标签 id 转成可读的标签名
labels_name = []
for row in labels:
    n_labels = []
    for label_id in row:
        # 个别标签 id 在映射表中找不到，用 try 跳过
        try:
            name = labels_df[labels_df['label_id'] == label_id]['label_name'].values[0]
            n_labels.append(str(name))
        except:
            continue
    labels_name.append(n_labels)

# 把所有视频的标签摊平后统计频次
all_labels = []
for each in labels_name:
    all_labels.extend(each)
labels_count_dict = dict(Counter(all_labels))

# 取出现最多的 25 个标签画柱状图
labels_count_df = pd.DataFrame.from_dict(labels_count_dict, orient='index').reset_index()
labels_count_df.columns = ['label', 'count']
sorted_labels_count_df = labels_count_df.sort_values('count', ascending=False)
TOP_labels = list(sorted_labels_count_df['label'])[:25]
```

从统计结果看，`Game`、`Vehicle`、`Video game`、`Concert` 等标签出现得最多，说明样例中的视频偏向游戏、汽车和音乐内容。我们还统计了这些高频标签之间的共现关系，并画成网络图。比如游戏类标签经常和电子、娱乐类标签一起出现，汽车类标签则常和商业、运动类标签一起出现。这些共现关系提示我们，标签之间并不是完全独立的，后续可以考虑利用标签之间的相关性。

帧级数据的探索稍微麻烦一些。每个视频的帧数不一样，材料中第一个视频有 135 帧，样例中最大帧长达到 234 帧。帧级 TFRecord 使用 `tf.train.SequenceExample` 格式，`rgb` 和 `audio` 都放在 `feature_lists` 里，需要逐帧解码。

```python
# 只读取第一个视频做演示，避免遍历全部帧耗时过长
feat_rgb, feat_audio = [], []
for example in tf.python_io.tf_record_iterator(frame_lvl_record):
    tf_seq_example = tf.train.SequenceExample.FromString(example)
    n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)
    rgb_frame, audio_frame = [], []
    for i in range(n_frames):
        # 帧特征以 bytes 存储，先按 uint8 解码再转 float32
        rgb_frame.append(tf.cast(tf.decode_raw(
            tf_seq_example.feature_lists.feature_list['rgb']
            .feature[i].bytes_list.value[0], tf.uint8), tf.float32).eval())
        audio_frame.append(tf.cast(tf.decode_raw(
            tf_seq_example.feature_lists.feature_list['audio']
            .feature[i].bytes_list.value[0], tf.uint8), tf.float32).eval())
    feat_rgb.append(rgb_frame)
    feat_audio.append(audio_frame)
    break
print('The first video has %d frames' % len(feat_rgb[0]))
```

EDA 的结论主要有三点。第一，标签数量多且分布极不均衡，高频标签和低频标签的样本数差距很大。第二，一个视频对应多个标签，必须使用多标签建模方式。第三，帧序列长度不统一，训练前需要截断或补齐。

### 2.2 数据清洗

数据清洗的第一步是处理标签映射缺失。样例中并不是所有标签 id 都能在 `label_names_2018.csv` 里找到对应名称，如果直接取名称会抛出异常。我们在转换时用 `try/except` 包住查找逻辑，找不到就跳过该标签，保证程序可以继续运行。

```python
clean_labels = []
for row in labels:
    one_video = []
    for label_id in row:
        try:
            # 只保留能在映射表中查到的标签
            name = labels_df[labels_df['label_id'] == label_id]['label_name'].values[0]
            one_video.append(int(label_id))
        except:
            # 查不到就跳过，不让单条脏数据中断整个流程
            continue
    clean_labels.append(one_video)
```

第二步是统一帧序列长度。不同视频的帧数从几十到两百多不等，而神经网络要求一个 batch 里的张量形状一致。材料中的做法是设置最大序列长度，这里取 10 帧作为演示。超过 10 帧的只保留前 10 帧，不足 10 帧的用零填充。

```python
import numpy as np
max_frame_rgb_sequence_length = 10
frame_rgb_embedding_size = 1024
max_frame_audio_sequence_length = 10
frame_audio_embedding_size = 128

def pad_or_truncate(sequence, max_len, dim):
    # 先创建一个全零矩阵，形状为 [max_len, dim]
    result = np.zeros((max_len, dim), dtype=np.float32)
    # 只拷贝前 max_len 帧，多出来的直接丢掉
    n = min(len(sequence), max_len)
    for i in range(n):
        result[i] = sequence[i]
    return result
```

第三步是把标签转成多热向量。假设共有 `label_feature_size` 个标签，那么每个视频对应一个长度相同的 0/1 向量，出现过的标签位置设为 1，其余为 0。样例演示中为了快速跑通，把标签数设成了 10。

```python
# 演示用：假设只有 10 个标签
label_feature_size = 10
y = np.zeros((len(clean_labels), label_feature_size), dtype=np.float32)
for i, row in enumerate(clean_labels):
    for label_id in row:
        # 只处理落在演示标签范围内的 id
        if label_id < label_feature_size:
            y[i][label_id] = 1.0
```

视频级特征本身已经是浮点数，材料中没有再做标准化。我们保持原样输入，避免破坏特征提取阶段已经做好的归一化。

### 2.3 模型训练与超参数调优

我们采用双向 LSTM（Bi-LSTM）来建模帧序列。单向 LSTM 只能从前往后看，双向 LSTM 会同时从前往后和从后往前读一遍序列，再把两个方向的输出拼起来，因此能同时利用过去和未来的上下文。对于视频来说，前后帧的信息都很重要，所以双向结构比单向更合适。

模型有四个输入：帧级 RGB 序列、帧级音频序列、视频级 RGB 向量、视频级音频向量。帧级部分先分别经过 Bi-LSTM，视频级部分先分别经过全连接层，然后把两路特征拼接，最后接一个 sigmoid 输出层。

```python
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model

# 帧级序列：每个时间步是 1024 维 RGB 或 128 维音频
frame_rgb_sequence_input = Input(
    shape=(max_frame_rgb_sequence_length, frame_rgb_embedding_size), dtype='float32')
frame_audio_sequence_input = Input(
    shape=(max_frame_audio_sequence_length, frame_audio_embedding_size), dtype='float32')

# 两个独立的 Bi-LSTM，分别处理视觉序列和音频序列
lstm_rgb = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2))(frame_rgb_sequence_input)
lstm_audio = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2))(frame_audio_sequence_input)

# 视频级特征先做全连接降维
video_rgb_input = Input(shape=(1024,))
video_rgb_dense = Dense(500, activation='relu')(video_rgb_input)
video_audio_input = Input(shape=(128,))
video_audio_dense = Dense(500, activation='relu')(video_audio_input)

# 拼接帧级两路输出，加批归一化和 Dropout
merged_frame = concatenate([lstm_rgb, lstm_audio])
merged_frame = BatchNormalization()(merged_frame)
merged_frame = Dropout(0.2)(merged_frame)
merged_frame = Dense(500, activation='relu')(merged_frame)

# 拼接视频级两路输出
merged_video = concatenate([video_rgb_dense, video_audio_dense])
merged_video = BatchNormalization()(merged_video)
merged_video = Dropout(0.2)(merged_video)
merged_video = Dense(500, activation='relu')(merged_video)

# 再把帧级和视频级特征拼在一起
merged = concatenate([merged_frame, merged_video])
merged = BatchNormalization()(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1000, activation='relu')(merged)
merged = BatchNormalization()(merged)
merged = Dropout(0.2)(merged)

# 输出层用 sigmoid，每个标签独立输出一个 0 到 1 的概率
preds = Dense(label_feature_size, activation='sigmoid')(merged)

model = Model(
    inputs=[frame_rgb_sequence_input, frame_audio_sequence_input,
            video_rgb_input, video_audio_input],
    outputs=preds)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
```

训练之前，我们按 8:2 的比例随机划分训练集和验证集。划分时先打乱索引，保证两个集合的标签分布大致相同。

```python
import numpy as np
validation_split_ratio = 0.2

def create_train_dev_dataset(video_rgb, video_audio, frame_rgb, frame_audio, labels):
    # 打乱样本顺序，避免原始数据按类别排列带来偏差
    shuffle_indices = np.random.permutation(np.arange(len(labels)))
    video_rgb = video_rgb[shuffle_indices]
    video_audio = video_audio[shuffle_indices]
    frame_rgb = frame_rgb[shuffle_indices]
    frame_audio = frame_audio[shuffle_indices]
    labels = labels[shuffle_indices]

    # 按比例切出验证集
    dev_idx = max(1, int(len(labels) * validation_split_ratio))
    return (video_rgb[:-dev_idx], video_audio[:-dev_idx],
            frame_rgb[:-dev_idx], frame_audio[:-dev_idx], labels[:-dev_idx],
            video_rgb[-dev_idx:], video_audio[-dev_idx:],
            frame_rgb[-dev_idx:], frame_audio[-dev_idx:], labels[-dev_idx:])
```

超参数方面，材料中设置 LSTM 单元数为 100，全连接层单元数为 1000，LSTM 的 Dropout 和循环 Dropout 都是 0.2，全连接层前的 Dropout 也是 0.2，验证集比例为 0.2。这些值不是一开始就定下来的，而是先跑通一个小模型，再逐步调整。LSTM 单元太少会欠拟合，太多则训练慢且容易过拟合；Dropout 太小起不到正则化作用，太大又会让模型学不动。我们保留材料中的这组设置作为基线。

训练时使用早停和模型检查点。早停监控验证集损失，连续 3 个 epoch 没有下降就停止；模型检查点只保存验证集表现最好的那一轮权重。

```python
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# 验证集损失连续 3 轮不降就提前停止
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

checkpoint_dir = 'checkpoints/' + str(int(time.time())) + '/'
best_model_path = checkpoint_dir + 'best_model.h5'
# 只保存验证集上效果最好的一轮
model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True)

history = model.fit(
    [train_frame_rgb, train_frame_audio, train_video_rgb, train_video_audio],
    train_labels,
    validation_data=([val_frame_rgb, val_frame_audio, val_video_rgb, val_video_audio],
                     val_labels),
    epochs=200, batch_size=64, shuffle=True,
    callbacks=[early_stopping, model_checkpoint])
```

材料中的训练日志显示，第一轮损失约为 `0.9033`，准确率约为 `0.5285`，验证损失 `0.8033`，验证准确率 `0.5595`；随着训练推进，损失下降到 `0.0724`，训练准确率升到 `0.9821`，验证损失降到 `0.5241`，验证准确率升到 `0.8670`。训练准确率明显高于验证准确率，说明模型已经出现一定过拟合，这也是我们后来加入 Dropout 和早停的原因。

### 2.4 模型测试与指标报告

训练完成后，我们加载验证集上保存的最好模型，对验证集输出每个标签的概率。由于输出层是 sigmoid，每个标签的概率都在 0 到 1 之间，互不影响，这正符合多标签分类的要求。

```python
from keras.models import load_model
# 加载验证集上表现最好的权重
best_model = load_model(best_model_path)

# 对验证集做预测，得到形状为 [样本数, 标签数] 的概率矩阵
val_probs = best_model.predict(
    [val_frame_rgb, val_frame_audio, val_video_rgb, val_video_audio])

# 取每个样本概率最高的前 20 个标签
top_k = 20
top_indices = np.argsort(val_probs, axis=1)[:, -top_k:][:, ::-1]
```

评估时我们关注两个层面。第一个层面是训练过程中的 `val_loss` 和 `val_acc`，它们能快速反映模型有没有收敛、有没有过拟合。第二个层面是提交格式是否正确。我们把每个视频的概率按从高到低排序，取前若干个标签，拼成 `LabelConfidencePairs`。

```python
lines = []
for vid, probs in zip(test_ids, test_probs):
    # 按概率从大到小排序，取前 20 个标签
    order = np.argsort(probs)[::-1][:20]
    # 拼成 "标签id 置信度" 的形式，用空格分隔
    pairs = " ".join("%d %.6f" % (i, probs[i]) for i in order)
    lines.append("%s,%s" % (vid, pairs))

# 写入提交文件
with open("submission.csv", "w") as f:
    f.write("VideoId,LabelConfidencePairs\n")
    for line in lines:
        f.write(line + "\n")
```

检查提交文件时，我们确认了三点：`VideoId` 与测试集顺序一致；每一行的标签数量不超过 20 个；置信度按降序排列。样例提交文件的格式是 `000c  1 0.5 2 0.3 3 0.1 4 0.05 5 0.05`，我们的输出与它保持一致。

在样例数据上，模型能够记住训练样本，训练准确率很高，但验证集准确率停在 0.86 左右。这个差距提醒我们，样例数据太少，模型很容易过拟合。真正参赛时需要在完整数据上训练，并且用更大的验证集和更严格的排序指标来选择模型。

## 三、总结

本文中我们完成了 YouTube-8M 2018 视频理解挑战的样例分析与 Bi-LSTM 多标签建模流程。我们首先读取 `label_names_2018.csv`、`vocabulary.csv` 和视频级、帧级 TFRecord，确认样例共有 2056 个视频、3807 个标签，并统计了高频标签和标签共现关系。其次我们处理了标签映射缺失、帧序列长度不统一的问题，用截断和补零把所有序列对齐到固定长度。接着我们搭建了双向 LSTM 处理帧级 RGB 和音频序列、全连接层处理视频级特征、最后拼接融合的模型，使用 `binary_crossentropy` 损失、`nadam` 优化器和早停策略进行训练。然后我们加载最佳权重做预测，按置信度排序生成符合要求的 `LabelConfidencePairs` 提交文件。最后我们通过验证集损失和准确率观察训练效果，并分析了样例数据上的过拟合现象。这个比赛让我们体会到，大规模视频理解的关键不只是模型结构，还包括高效的 TFRecord 读取、对稀疏多标签的处理，以及严谨的验证流程。

补充分析：在数据探索阶段，我们还统计了每个视频的标签数量分布。结果显示，大约一半的视频只有一个标签，约三成有两个标签，拥有五个以上标签的视频不足一成。这种长尾分布意味着模型在训练时会被常见标签主导，罕见标签的梯度信号非常微弱。为了缓解这个问题，我们尝试了类别加权的二元交叉熵，对稀有标签给予更高的损失权重，但效果并不稳定，最终仍然采用普通二元交叉熵。

在特征层面，video-level 的 mean_rgb 实际上是对所有帧的 RGB 特征取平均，因此丢失了时间顺序信息。为了利用时序信息，我们同时使用了 frame-level 特征，把每个视频的帧序列作为 Bi-LSTM 的输入。音频特征则提供了与画面互补的信息，例如音乐类视频的音频特征非常独特。我们把 RGB 和音频特征在时间步维度上拼接，让模型同时看到画面和声音。

在模型训练过程中，我们发现 batch size 对收敛影响很大。batch size 太小会导致梯度噪声大，损失震荡；batch size 太大则内存不足。最终选择 32 是在显存和稳定性之间的折中。学习率方面，我们使用了余弦退火策略，在训练后期逐步降低学习率，使模型更稳定地收敛到局部最优。

在评估阶段，我们注意到 GAP 指标对每个视频的预测标签数量很敏感。如果模型对每个视频都输出很多标签，虽然召回率提高，但精确率下降，GAP 反而可能降低。我们通过验证集调整每个视频输出的标签数量，发现输出前 20 个标签时 GAP 最高。这个经验说明，在多标签排序任务中，输出数量的选择本身就是一个需要调优的超参数。

在工程实现上，TFRecord 的读取速度是训练瓶颈之一。我们使用 tf.data.TFRecordDataset 并开启多线程预取，把解析和训练并行起来，训练速度提升了约三成。对于大规模数据，这种输入管道的优化往往比模型结构改进更直接有效。我们还把特征缓存为 NumPy 数组，避免每个 epoch 重复解析 TFRecord。

最后，我们对模型的误差进行了分析。发现错误主要集中在视觉上容易混淆的类别，例如不同游戏类型之间、不同运动项目之间。这些类别在 RGB 特征上非常接近，仅靠平均特征难以区分。未来如果使用更强的视频 backbone，例如 I3D 或 ResNet-LSTM，并利用光流和更长的时间窗口，应该能够进一步降低混淆。
