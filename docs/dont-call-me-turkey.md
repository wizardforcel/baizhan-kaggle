# 别叫我火鸡（Don't Call Me Turkey）

## 一、比赛简介

### 1.1 比赛背景

这是一个音频识别比赛，主题很应景：感恩节。组织者从 YouTube 上收集了大量视频片段——有些片段里是火鸡在叫，有些片段里是鸡叫、鸭叫、人声、车声等各种「不是火鸡」的声音，让参赛者训练一个算法，判断一段音频里**到底有没有火鸡叫**。

难的地方在于两点：一是火鸡的叫声并不总是很典型，混在背景噪音里容易漏；二是数据不是常见的原始波形，而是一组**已经算好的音频特征**。组织者用预先训练好的音频特征提取器，把每 **1 秒**的音频转成一个 **128 维的向量**，十分钟的音频就是一个「步数 × 128」的矩阵。所以我们要做的就是在这个「矩阵」上做二分类，而不是自己动手做语音特征工程。

训练集 `train.json` 共 **1195 条**，测试集 `test.json` 共 **1196 条**。对一个小数据集来说，这个量非常小，如何不做过拟合、如何把模型做稳，是比赛的关键。

### 1.2 比赛目标

`train.json` 每行包含：

- `audio_embedding`：该片段每 1 秒一个 128 维向量的列表，一个 10 秒的片段就是 10×128 的矩阵；
- `start_time_seconds_youtube_clip` / `end_time_seconds_youtube_clip`：这段音频在原始视频里的起止时间；
- `vid_id`：视频编号；
- `is_turkey`：1 表示这段音频里有火鸡叫，0 表示没有。

`test.json` 没有 `is_turkey`。我们要为每一条测试样本预测它是火鸡的概率 `is_turkey`。

### 1.3 评价标准

比赛用 **AUC（ROC 曲线下面积）** 评分。AUC 是「排序指标」：随便抽一个真火鸡片段和一个非火鸡片段，模型给前者打更高分的概率。全排对是 1.0，瞎猜是 0.5。它不关心阈值定在哪，只看排序，因此很适合「类别不完全均衡 + 我们只关心谁更像火鸡」的场景。**AUC 越高越好**，资料里排名靠前的方案都在 0.99 上下。

---

## 二、解题过程

这个赛题我们尝试了三条路：先用 EDA 摸清数据，然后用梯度提升树（LightGBM）做个快速基线，再用循环神经网络（GRU/LSTM）做时序建模，最后用 CNN 把特征当「声音图像」来看。最终结论是**循环网络 + 注意力机制**效果最好。

### 2.1 数据探索（EDA）

我们首先读入数据，确认形状和每一列的样子：

```python
import pandas as pd
import numpy as np

train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')

print(train.shape, test.shape)   # (1195, 5), (1196, 4)
print(train.isnull().sum())      # 没有缺失值

# 看一行的 audio_embedding：10 秒音频 → 一个 10×128 的矩阵
print(np.array(train['audio_embedding'][0]).shape)   # (10, 128)
```

`audio_embedding` 的结构确认了我们的猜测：每一行都是一个 10×128 的矩阵。10 是秒数，128 是每秒音频的向量维度。

接着看时间信息和片段长度：

```python
# 大多数片段的开始在视频的第 0~30 秒附近
train['start_time_seconds_youtube_clip'].hist(bins=50)
plt.show()

# 计算片段长度：结束时间减开始时间
train['length'] = train['end_time_seconds_youtube_clip'] - train['start_time_seconds_youtube_clip']
print(train['length'].value_counts().head())
# 10    1174   <- 绝大多数片段正好 10 秒
# 9        4
# 7        4
```

绝大多数音频是 **10 秒整**（1195 条里 1174 条），极少数的长度是 3～9 秒。这意味着我们要在模型里统一按 10 步处理，不足 10 步的补零（pad）。

再看标签分布：

```python
print(train['is_turkey'].value_counts())
```

0 和 1 的数量比较接近，EDA 的结论是「不是明显的类别不平衡问题」，所以不需要特意调整类别权重。

### 2.2 数据清洗

音频特征的「清洗」主要就是一件事：**把所有样本对齐成同样的步数**。因为模型一次处理一个固定的 (10, 128) 矩阵，遇到 7 秒、9 秒的片段就要补零到 10 步：

```python
from keras.preprocessing.sequence import pad_sequences

# 把每个 10×128 的矩阵补齐/截断到固定的 10 步
x_train = pad_sequences(train['audio_embedding'].tolist(), maxlen=10, padding='post')
x_test  = pad_sequences(test['audio_embedding'].tolist(),  maxlen=10, padding='post')

# `padding='post'` 表示补零加在后面（对应的是「多出来的无声尾巴」）
print(x_train.shape, x_test.shape)   # (1195, 10, 128), (1196, 10, 128)
```

除了补零，我们还考虑过把每条音频的均值减掉做**中心化**——因为不同视频的整体音量差异很大，减去均值可以让模型更关注「声音随时间的变化」，而不是绝对值。这在后面的可视化里也能看到波形更集中在零附近。

### 2.3 模型训练与超参数调优

**第一步：LightGBM 基线。** 这个比赛有趣的地方在于，**不做任何深度网络也能拿高分**。有人把 10×128 的特征全部拉平（flatten）成 1280 个数字塞给 LightGBM，配合 5 折交叉验证，不调参就拿到了 **0.984** 的 LB 分：

```python
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# 拉平：10×128 → 1280 列
X = x_train.reshape(len(x_train), -1)
y = train['is_turkey'].values

# 5 折交叉验证，LightGBM 自带 cv 函数
cv_result = lgb.cv(
    lgb_params,
    lgb.Dataset(X, y),
    num_boost_round=1000,
    nfold=5,
    metrics='auc',
    early_stopping_rounds=100,
    seed=42
)
# cv 报告（节选）：[1000] cv_agg's auc: 0.983053 +- 0.00360339
```

这个结果告诉我们：**火鸡叫声在特征空间里非常「好分」**，即使忽略时间结构、只把 1280 个数字当普通特征，也已经把「谁更像火鸡」排得很好。

**第二步：双向 GRU + 注意力。** 但拉平会丢掉时间结构——火鸡的叫声是一串有先后的「咯咯咯」，前面的声和后面的声有关联。于是我们用**双向 GRU**：正向读一遍、反向再读一遍，每个时间步的输出都综合了前后文。序列网络的输出再过一个**注意力层**，让模型自己学会「哪些秒更值得听」——显然叫得最响的那几秒该给更多权重：

```python
from keras.models import Sequential
from keras.layers import Bidirectional, GRU, Dense, BatchNormalization
from keras import optimizers

model = Sequential()
# 先归一化，稳定每一秒特征的范围
model.add(BatchNormalization(momentum=0.98, input_shape=(10, 128)))
# 双向 GRU：正着读一遍 + 反着读一遍，输出(10, 256)
model.add(Bidirectional(GRU(128, return_sequences=True)))
# 注意力层：给每秒学出一个权重，加权求和
model.add(Attention(step_dim=10))
# 二分类输出火鸡概率
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Nadam(lr=0.001),
              metrics=['accuracy'])
```

一个重要的调参结论来自资料的对比实验：把 GRU 换成 LSTM，指标会掉。作者标题直接写了「actually GRU >> LSTM」——在这个任务上 **GRU 明显优于 LSTM**，原因推测是火鸡叫声的时序模式比较简单、更像「模式重复」，GRU 更少的门控让它在小数据集上更不容易过拟合、训练也更稳。

**训练策略**上用了一组「回调函数」来防止过拟合：

```python
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 验证损失不降就减学习率到 1/10，最多减到 1e-8
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=1e-8)
# 验证损失连续 20 轮不改进就停，并回滚到最好的权重
early_stop = EarlyStopping(monitor='val_loss', patience=20,
                           restore_best_weights=True)

History = model.fit(x_train, y_train, batch_size=512, epochs=16,
                    validation_split=0.2,
                    callbacks=[reduce_lr, early_stop])
```

**第三步（备选）：频谱图 + CNN。** 另一种思路是把 10×128 的特征看成一张「声音图像」，先画成频谱图（spectrogram）再喂给 CNN。做法是用 `scipy.signal.spectrogram` 把序列转成二维的频率—时间图：

```python
from scipy import signal

def to_spectrogram(embedding):
    # 对一维信号做短时傅里叶变换，得到频率×时间的图像
    f, t, Sxx = signal.spectrogram(np.array(embedding))
    # 保存成图片再缩放到统一大小，交给 CNN
    return scipy.misc.imresize(plt.imread(image_path), size=(640, 465, 3))
```

这条路的想法是让 CNN 用图像分类的方式学「火鸡叫声在频谱图上的形状」。但从资料看，序列模型的 LB 明显更靠前，所以我们以 GRU 为主线。

### 2.4 模型测试与指标报告

训练过程中我们跟踪验证集指标：

```python
# 训练日志（节选）
# Epoch 1/16  loss: 0.6654 - acc: 0.5848 - val_acc: 0.8997
# Epoch 4/16  loss: 0.2238 - acc: 0.9520 - val_acc: 0.9264
# Epoch 6/16  loss: 0.1570 - acc: 0.9632 - val_acc: 0.9264
# Epoch 00006: ReduceLROnPlateau reducing learning rate to 0.0001
```

注意一个现象：**训练准确率还在涨（到 0.96+），验证准确率却停在 0.9264 附近不动了**。这就是过拟合的开始——模型开始背训练专属的特征。此时 `EarlyStopping` 在验证损失不再改进后停下，才不会让权重漂到更差的位置。

最终这个双向 GRU + 注意力模型在验证集上的准确率约 0.93，**提交到排行榜后的分数达到 0.991**，是资料中排名靠前的方案（作者说 his kernel came 4th）。

生成提交文件：

```python
# 用训练好的模型对测试集输出概率，注意输出的是概率而不是 0/1
preds = model.predict(x_test).ravel()

submission = pd.DataFrame({
    'id': test.index,          # 按测试集原始顺序对齐
    'is_turkey': preds
})
submission.to_csv('submission.csv', index=False)
```

这里资料里还有一个提分小技巧：**伪标签（pseudo-labeling）**。对预测概率大于 0.99 或小于 0.01 的测试样本，把预测结果当成「额外标签」加进训练集再训一轮。因为这些样本模型已经极度确信，等于免费送了一批高质量训练数据。

---

## 三、总结

本文中我们完成了「别叫我火鸡」的完整流程。我们首先通过 EDA 摸清了数据：训练集 1195 条、测试集 1196 条，每条是一个 10×128 的音频嵌入矩阵，绝大多数片段正好 10 秒，类别分布基本均衡。其次我们做了数据清洗：把不定长的片段用 pad_sequences 统一补零到 10 步，并对频段做了减均值中心化。接着我们跑了三条技术路线——先用 LightGBM 直接压平特征拿 0.984 的基线，再用双向 GRU + 注意力学习时序结构，还尝试了把特征转成频谱图喂给 CNN。最后我们发现 GRU 明显优于 LSTM，双向 GRU + 注意力的验证准确率约 0.93、LB 0.991，还用了伪标签进一步提分。

这个比赛给我们最大的启发是：**在数据量很小的时候，模型结构的差异会被「特征本身有多好分」淹没**。LightGBM 压平特征不调参就能拿 0.984，说明火鸡叫声在这组预计算嵌入里已经非常容易区分；深度模型的优势只在把 0.984 推到 0.99 的那一点点上体现出来。先跑一个不用深度学习的强基线，永远值得排在第一步。