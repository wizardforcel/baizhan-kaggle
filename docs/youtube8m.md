# YouTube-8M 视频标签分类

## 一、比赛简介

### 1.1 比赛背景

YouTube-8M 是 Google 发布的一个大规模视频理解数据集。它的每条样本不是一张图片，而是一段 YouTube 视频，并且视频已经被提前抽成了数值特征。比赛要做的事情是：给定一段视频的视觉特征和音频特征，预测这段视频属于哪些内容类别。因为一段视频可能同时讲游戏、汽车和音乐，所以一条样本往往对应多个标签，这属于**多标签分类**，而不是“从一堆类别里只选一个”的普通分类。

材料中的 EDA 写得很清楚：数据分成 `video_level` 和 `frame_level` 两套。`video_level` 把整段视频压缩成一个向量，读取更方便；`frame_level` 保留每一帧的特征，信息更细，但文件更大、读取更慢。视频级特征里，`mean_rgb` 是 1024 维的视觉特征，`mean_audio` 是 128 维的音频特征。比赛还提供了 `label_names.csv`，把标签编号翻译成人类可读的名称，例如 0 号是 `Games`、4 号是 `Car`、9 号是 `Music video`。

这个题目的难点主要有三个。第一，类别非常多，材料的模型输出层直接设置了 4716 个输出单元，说明标签空间很大。第二，标签不均衡，少数热门类别出现得很多，大量长尾类别出现得很少。第三，视频特征维度高，一条样本就有 1024 + 128 个数值，直接训练容易过拟合，所以需要先做特征压缩或全连接层降维。

### 1.2 比赛目标

训练数据以 TFRecord 格式存放。每个视频级样本包含：

- `video_id`：视频编号；
- `labels`：该视频对应的标签编号列表，可能不止一个；
- `mean_rgb`：1024 维视觉特征；
- `mean_audio`：128 维音频特征。

帧级样本还包含每一帧的 `rgb` 与 `audio` 特征，材料的 EDA 读取了第一条视频，发现它有 161 帧。比赛要求我们为测试集中的每个 `video_id` 输出若干组“标签编号 + 置信度”对。材料中的提交代码取置信度最高的 20 个标签，拼成 `LabelConfidencePairs` 字段，最终提交文件包含 `VideoId` 和 `LabelConfidencePairs` 两列。

### 1.3 评价标准

材料中的 `Notebook18fa8571e3` 直接实现了比赛使用的 **GAP（Global Average Precision）指标**，并且把每次预测限制在前 20 个标签，也就是常说的 GAP@20。它的计算过程可以拆成两步。

第一步，对单个视频把所有类别的预测分数从高到低排序，只保留前 20 个。然后按顺序往下看：如果当前标签真的是这个视频的标签，就把“命中数除以当前位置”累加一次，最后再除以该视频真实标签数（最多按 20 个算）。这其实就是在算单个视频的 Average Precision。

第二步，把所有视频的 AP 求平均，得到 GAP。GAP 越高越好，满分是 1。它和普通准确率的不同在于：模型不需要把 4716 个类别全排对，只要把最可能的 20 个标签排好，就能拿到不错的分数。这个指标也解释了为什么提交文件只保留前 20 个标签。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先用 TensorFlow 的 `tf_record_iterator` 逐条读取视频级 TFRecord。TFRecord 可以理解成一种把很多样本打包在一起的大文件，里面每条记录都要先解析成 `tf.train.Example`，再按字段名取出 `video_id`、`labels`、`mean_rgb` 和 `mean_audio`。材料中的 EDA 读取 `train-1.tfrecord`，一共得到 1178 条视频，并确认视觉特征长度是 1024、音频特征长度是 128。

```python
import tensorflow as tf
import numpy as np
import pandas as pd

# 视频级 TFRecord：一条记录就是一段视频的汇总特征
video_lvl_record = "../input/video_level/train-1.tfrecord"

vid_ids, labels, mean_rgb, mean_audio = [], [], [], []
for example in tf.python_io.tf_record_iterator(video_lvl_record):
    # 把二进制记录解析成 TensorFlow 的 Example 结构
    tf_example = tf.train.Example.FromString(example)

    # video_id 是字节串，需要解码成普通字符串
    vid = tf_example.features.feature['video_id'].bytes_list.value[0]
    vid_ids.append(vid.decode(encoding='UTF-8'))

    # labels 是整数列表，一个视频可以有多个标签
    labels.append(list(tf_example.features.feature['labels'].int64_list.value))

    # mean_rgb 和 mean_audio 是定长浮点向量
    mean_rgb.append(list(tf_example.features.feature['mean_rgb'].float_list.value))
    mean_audio.append(list(tf_example.features.feature['mean_audio'].float_list.value))

print('视频数量:', len(mean_rgb))
print('视觉特征长度:', len(mean_rgb[0]))
print('音频特征长度:', len(mean_audio[0]))
```

我们其次检查标签表。材料的 EDA 用 `label_names.csv` 把编号映射成名称，并统计每个标签出现的次数。需要注意，`label_names.csv` 并不是每个编号都有名字：在读取 10 个训练 TFRecord 时，材料发现 21 个标签编号在名称表里缺失。这个现象本身很重要，它提醒我们不能假设“标签编号一定能翻译成名称”，后面构造训练标签时应该直接使用整数编号。

```python
# 标签名称表：label_id 是模型使用的整数编号，label_name 是人类可读名称
labels_df = pd.read_csv('../input/label_names.csv')

# 统计名称表里有多少个不同的类别名称
print('不同标签名称数量:', labels_df['label_name'].nunique())

# 把所有训练记录里的标签编号汇总，观察哪些编号没有对应名称
all_label_ids = set()
for fn in ["../input/video_level/train-{}.tfrecord".format(i) for i in range(10)]:
    for example in tf.python_io.tf_record_iterator(fn):
        ex = tf.train.Example.FromString(example)
        all_label_ids.update(ex.features.feature['labels'].int64_list.value)

known_ids = set(labels_df['label_id'])
missing_ids = sorted(all_label_ids - known_ids)
print('缺失名称的标签编号数量:', len(missing_ids))
print('前几个缺失编号:', missing_ids[:10])
```

接着我们查看每个视频带多少个标签，以及最常见的标签是什么。多标签数据里，大部分样本可能只有一两个标签，少量样本有十几个标签；这种分布会直接影响损失函数和阈值选择。材料把标签数量画成分布图，并统计了出现频率最高的类别，例如 `Cosmetics`、`Games`、`Car`、`Vehicle`、`Animal` 等。

```python
from collections import Counter
import matplotlib.pyplot as plt

# 每个视频的标签数量
label_counts = [len(x) for x in labels]

plt.hist(label_counts, bins=30)
plt.xlabel('每个视频的标签数量')
plt.ylabel('视频数量')
plt.title('标签数量分布')
plt.show()

# 统计所有标签编号的出现次数，取最常见的 10 个
top_n = Counter([item for sublist in labels for item in sublist]).most_common(10)
print('最常见标签编号:', [x[0] for x in top_n])
print('对应次数:', [x[1] for x in top_n])
```

最后我们观察帧级数据。视频级特征是把所有帧平均之后得到的，帧级特征则保留时间顺序。材料只读取了第一条视频，因为它有 161 帧，逐帧解码比较耗时。读取帧级数据要用 `SequenceExample`，而不是 `Example`，因为每帧是一个序列中的一个元素。

```python
# 帧级 TFRecord 使用 SequenceExample，因为一条视频包含多帧
frame_lvl_record = "../input/frame_level/train-1.tfrecord"

for example in tf.python_io.tf_record_iterator(frame_lvl_record):
    seq_example = tf.train.SequenceExample.FromString(example)
    n_frames = len(seq_example.feature_lists.feature_list['audio'].feature)
    print('第一条视频的帧数:', n_frames)
    # 只读第一条就退出，避免把整个文件解码进内存
    break
```

### 2.2 数据清洗

我们然后处理标签。因为模型输出层固定为 4716 个类别，而 `labels` 字段只给出当前视频命中的编号，所以要把“编号列表”转换成“长度为 4716 的 0/1 向量”。这种向量叫 multi-hot 编码，意思是多个位置可以同时为 1。材料中的 `tf_itr` 正是这样做的：先创建一个全是 0 的 `np.zeros(4716)`，再把当前视频的每个标签位置设为 1。

```python
def labels_to_multihot(label_ids, num_classes=4716):
    # 先全部置 0，表示默认不属于任何类别
    out = np.zeros(num_classes, dtype=np.int8)
    for y in label_ids:
        # 一个视频可以命中多个类别，所以这里不是互斥的 one-hot
        out[y] = 1
    return out

# 用第一条样本检查编码结果
example_vector = labels_to_multihot(labels[0])
print('向量长度:', example_vector.shape)
print('为 1 的位置:', np.where(example_vector == 1)[0])
```

我们没有因为名称缺失就删掉这些样本。原因是模型训练只依赖整数标签，名称表缺失只影响人类阅读，不影响监督信号。反过来，如果把缺失名称的样本删掉，反而会丢掉一部分有效数据。对于视频特征，材料也没有做标准化或异常值删除，而是直接送入网络；这让我们可以把注意力放在网络结构和训练流程上。

### 2.3 模型训练与超参数调优

我们接着搭建一个双输入的全连接网络。视觉特征和音频特征的维度不同，所以材料用两个 `Input` 分别接收 128 维音频和 1024 维视觉向量，各自经过一个 `fc_block` 升到 1024 维，再拼接起来，最后接一个 4716 维的 `sigmoid` 输出。`sigmoid` 会把每个类别的分数压到 0 到 1 之间，正好可以当作“这个类别出现的概率”。

`fc_block` 里依次用了全连接层、批归一化、LeakyReLU 和 Dropout。批归一化让每一层的输入尺度更稳定，训练更容易收敛；LeakyReLU 是激活函数，让网络能学习非线性关系；Dropout 在训练时随机丢弃一部分神经元，能减轻过拟合。材料设置的丢弃比例是 0.2。

```python
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, merge
from keras.layers.advanced_activations import LeakyReLU


def fc_block(x, n=1024, d=0.2):
    # 全连接层把输入映射到统一宽度，方便后续拼接
    x = Dense(n, init='glorot_normal')(x)
    # 批归一化稳定数值尺度，减少训练初期的震荡
    x = BatchNormalization()(x)
    # LeakyReLU 在负数区保留一个小斜率，避免神经元完全失活
    x = LeakyReLU()(x)
    # Dropout 随机丢弃一部分连接，降低过拟合风险
    x = Dropout(d)(x)
    return x


def build_mod():
    # 音频分支：输入 128 维，输出 1024 维表示
    in1 = Input((128,), name='x1')
    x1 = fc_block(in1)

    # 视觉分支：输入 1024 维，同样映射到 1024 维表示
    in2 = Input((1024,), name='x2')
    x2 = fc_block(in2)

    # 把音频和视觉表示拼接，让模型同时利用两种信息
    x = merge([x1, x2], mode='concat', concat_axis=1)
    x = fc_block(x)

    # 4716 个类别各自输出一个概率，sigmoid 适合多标签场景
    out = Dense(4716, activation='sigmoid', name='output')(x)

    model = Model(input=[in1, in2], output=out)
    # 材料使用 categorical_crossentropy；多标签任务通常也可用 binary_crossentropy
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model
```

训练时，材料用 `tf_itr` 按批次读取数据，每批 10×1024 条，最多训练 100 个 epoch。每训练 10 轮，就在验证集上算一次 GAP；如果当前 GAP 更高，就把权重保存下来。这里的“调参”不是网格搜索，而是用验证集 GAP 做模型选择：保留验证集表现最好的权重，避免最后一轮过拟合的模型被拿去提交。

```python
def train():
    if not os.path.exists('weights'):
        os.mkdir('weights')

    batch = 10 * 1024      # 每批一万条左右，兼顾显存和梯度稳定性
    n_itr = 10             # 每 10 次参数更新做一次验证
    n_eph = 100            # 最多训练 100 轮

    # 验证集只取一次，后续重复使用，保证比较公平
    _, x1_val, x2_val, y_val = next(tf_itr('val', 10000))
    model = build_mod()

    cnt = 0
    for e in range(n_eph):
        for d in tf_itr('train', batch):
            _, x1_trn, x2_trn, y_trn = d
            # 用一小批数据更新一次参数
            model.train_on_batch({'x1': x1_trn, 'x2': x2_trn},
                                 {'output': y_trn})
            cnt += 1

            if cnt % n_itr == 0:
                # 在验证集上预测 4716 个类别的分数
                y_prd = model.predict({'x1': x1_val, 'x2': x2_val},
                                      verbose=False, batch_size=100)
                g = gap(y_prd, y_val)
                print('val GAP %0.5f; epoch: %d; iters: %d' % (g, e, cnt))
                # 用 GAP 给权重文件命名，方便最后挑最好的一个
                model.save_weights('weights/%0.5f_%d_%d.h5' % (g, e, cnt))
```

我们在这个阶段主要调整三个量：每个全连接块的宽度、Dropout 比例和批大小。材料给出的 `n=1024`、`d=0.2`、`batch=10*1024` 是一组可运行的基线。宽度再大，显存压力会明显上升；Dropout 太小容易过拟合，太大又会让模型学不动。由于每次验证都要跑一遍完整验证集，材料没有做很细的网格搜索，而是先保证训练流程稳定，再根据验证 GAP 保存最佳权重。

### 2.4 模型测试与指标报告

我们最后实现 GAP 的计算，并用它检查验证集。单条视频的 AP@20 先按预测分数从高到低排序，只检查前 20 个位置；每遇到一个真实标签，就把“当前命中数 / 当前位置”累加，再乘以“1 / 真实标签数”作为这次命中的权重。把所有视频的 AP 取平均，就是 GAP。

```python
def ap_at_n(data):
    # predictions 是 4716 个类别的分数，actuals 是 0/1 真实标签
    predictions, actuals = data
    n = 20

    # 按分数从高到低排序，只关心前 20 个候选
    sortidx = np.argsort(predictions)[::-1]
    numpos = np.size(np.where(actuals > 0))
    if numpos == 0:
        return 0

    # 每个真实标签贡献同样的召回增量
    numpos = min(numpos, n)
    delta_recall = 1.0 / numpos
    poscount = 0.0
    ap = 0.0

    r = min(len(sortidx), n)
    for i in range(r):
        if actuals[sortidx[i]] > 0:
            poscount += 1
            # 命中越靠前，poscount/(i+1) 越大，AP 越高
            ap += poscount / (i + 1) * delta_recall
    return ap


def gap(pred, actual):
    # 对每个视频分别算 AP，再对所有视频求平均
    pairs = zip(list(pred), list(actual))
    with Pool() as pool:
        all_ap = pool.map(ap_at_n, pairs)
    return np.mean(all_ap)
```

预测阶段，我们先从 `weights/` 目录里挑出文件名排序最后的权重。材料用 GAP 作为文件名前缀，所以按文件名排序时，分数高的权重会排在后面，取最后一个就相当于取验证集最好的模型。然后对测试集逐批预测，每条视频取分数最高的 20 个类别，拼成 `标签编号 置信度` 的字符串。

```python
def conv_pred(el):
    # 只保留分数最高的 20 个类别
    t = 20
    idx = np.argsort(el)[::-1]
    return ' '.join(['{} {:0.5f}'.format(i, el[i]) for i in idx[:t]])


def predict():
    model = build_mod()
    # 权重文件名以 GAP 开头，排序后最后一个就是验证集最好的
    wfn = sorted(glob.glob('weights/*.h5'))[-1]
    model.load_weights(wfn)
    print('加载权重:', wfn)

    # 读取测试集，得到视频编号、音频特征、视觉特征
    idx, x1_val, x2_val, _ = next(tf_itr('test', 700640))
    ypd = model.predict({'x1': x1_val, 'x2': x2_val},
                        verbose=1, batch_size=32)

    # 把每个视频的预测转成比赛要求的字符串格式
    with Pool() as pool:
        out = pool.map(conv_pred, list(ypd))

    df = pd.DataFrame.from_dict({
        'VideoId': idx,
        'LabelConfidencePairs': out
    })
    df.to_csv('subm1', header=True, index=False,
              columns=['VideoId', 'LabelConfidencePairs'])
```

材料没有给出最终提交的 GAP 数值，所以我们不虚构排行榜分数。我们能够确认的是：验证阶段会用 `gap(y_prd, y_val)` 打印每一轮的 GAP，并且只保存 GAP 较高的权重；测试阶段则按比赛格式输出前 20 个标签。对于入门者来说，更重要的是理解这条链路：TFRecord 解析、multi-hot 标签、双输入网络、验证集选模型、前 20 个标签生成提交文件。

## 三、总结

本文中我们完成了 YouTube-8M 视频标签分类的一套完整流程。我们首先用 TFRecord 读取视频级样本，确认每条视频有 1024 维视觉特征和 128 维音频特征，并查看了标签数量分布和常见类别。其次我们把整数标签列表转换成 4716 维的 multi-hot 向量，并且没有因为 `label_names.csv` 缺少部分名称就删除样本。接着我们搭建了音频和视觉双输入的全连接网络，用批归一化、LeakyReLU 和 Dropout 稳定训练，采用 Adam 优化器和 sigmoid 多标签输出。然后我们实现了 AP@20 与 GAP，用验证集 GAP 选择并保存最佳权重。最后我们加载最佳权重，对测试集预测前 20 个标签，生成包含 `VideoId` 和 `LabelConfidencePairs` 的提交文件。

这个比赛让我们体会到，多标签任务和普通单标签分类的差别不只在损失函数，还在评价方式。GAP@20 只要求把最可能的 20 个标签排好，因此提交文件保留前 20 个候选，而不是只给一个答案。另一个重要经验是：标签名称表可能不完整，但训练标签本身仍然有效，数据清洗要分清“展示用的名称”和“模型用的编号”。只要抓住这两点，即使只是全连接网络，也能建立起一个可运行、可验证的 YouTube-8M 基线。
