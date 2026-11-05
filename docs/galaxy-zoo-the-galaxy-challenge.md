# Galaxy Zoo 星系形态分类挑战（The Galaxy Challenge）

## 一、比赛简介

### 1.1 比赛背景

Galaxy Zoo（星系动物园）是一个著名的「公民科学」项目：天文学家把斯隆数字巡天（Sloan Digital Sky Survey）拍摄的海量星系图片放到网上，请普通网友帮忙看图片、回答几个问题——这个星系是圆的还是椭圆的？有没有旋臂？是正面朝向我们还是侧对着我们？

因为一个问题会有成千上万个网友来答，同一个星系就会得到一大堆「投票」。把这些投票统计起来，就得到每个选项的**得票比例**。这个比例非常宝贵：它相当于「人群的共识」，比任何单个专家的判断都更稳定。

这个比赛要做的就是：**给一张星系图片，直接预测出网友对 37 个问题的得票比例**。注意，这不是普通的「这张图是猫还是狗」，而是**同时回归 37 个 0 到 1 之间的连续数值**。所以它是一个多输出回归（multi-output regression）问题。

数据的难点在于：

- 图片是真实的天空照片，噪声大、亮度低、星系常常很小、背景还有别的星体干扰；
- 37 个输出之间并不独立（比如「完全圆」和「雪茄形」在物理上互斥），模型必须同时兼顾；
- 训练图片只有 6 万多张，而 37 维输出让模型很容易过拟合。

### 1.2 比赛目标

输入是一张星系图片（比赛中统一缩放到 100×100 像素）。输出是 37 个数值，对应 `Class1.1` 到 `Class7.3` 这些列，每一列表示「网友选择该答案的比例」。例如 `Class1.1` 是「光滑、圆形」的比例，`Class7.3` 是「雪茄形」的比例。

训练数据包含三部分：

- `images_training_rev1.zip`：61578 张训练图片，文件名就是 `GalaxyID`；
- `training_solutions_rev1.csv`：每个 `GalaxyID` 对应的 37 个比例值；
- `images_test_rev1.zip`：79975 张测试图片。

另外还提供了几个「基准文件」（`all_zeros_benchmark`、`all_ones_benchmark`、`central_pixel_benchmark`），可以理解为官方给的「笨办法参考答案」，用来判断模型有没有比瞎猜强。

### 1.3 评价标准

比赛用 **RMSE**（Root Mean Squared Error，均方根误差）评价，越低越好。具体做法是把 37 个输出的误差全部平方、求平均、再开根号。因为它对所有输出一视同仁，所以模型不能只把几个大类预测准，冷门的小类也必须照顾到。

需要留意的是：预测值应该在 0～1 之间，`all_zeros` 基准就是把所有输出都填 0，`all_ones` 就是全填 1。如果你的 RMSE 还不如 `all_zeros`，说明模型完全没学到东西。

---

## 二、解题过程

我们参考了一份结构清晰的 Keras 实现：用 TensorFlow 的 `tf.data` 流水线读图，用一个「四层卷积 + 两层全连接」的小型 CNN 直接回归 37 个比例，配上翻转、旋转、缩放、对比度四种图像增强。这条路线不追求极致分数，但每一步都可解释，非常适合入门。

### 2.1 数据探索（EDA）

我们首先解压数据，看看图片和标签长什么样：

```python
import os
import numpy as np
import pandas as pd

# 解压训练图、测试图和标签（-n 表示不覆盖已存在文件）
!unzip -q -n ../input/images_training_rev1.zip -d ../temp/
!unzip -q -n ../input/images_test_rev1.zip -d ../temp/
labels_pd = pd.read_csv('../input/training_solutions_rev1.zip', compression='zip')

# 看看标签表的结构
print(labels_pd.shape)     # (61578, 38)：1 列 GalaxyID + 37 列比例
labels_pd.head()
```

打印出来能看到：第一列是 `GalaxyID`，后面 37 列是 `Class1.1`、`Class1.2`、……、`Class7.3`。每一行的 37 个值基本都在 0～1 之间，但**一行加起来不一定是 1**——因为不同问题的选项集合不一样，比例是分组统计的。

接着我们列出图片文件，确认数量和尺寸：

```python
# 收集所有训练图片路径
images_path = tf.io.gfile.glob('../temp/images_training_rev1/*')
print('Number of total samples:', len(images_path))   # 61578

# 随机抽一张读进来，看尺寸
import tensorflow as tf
sample = tf.io.decode_image(tf.io.read_file(images_path[0]), dtype=tf.float32)
print(sample.shape)   # 原始尺寸不一，需要统一缩放
```

原始图片大小并不统一，但比赛说明是 100×100 左右。为了让 CNN 能批量处理，我们统一用 `resize_with_crop_or_pad` 调整到 100×100×3：

```python
def get_image_and_features(image_path, training=True):
    image = tf.io.decode_image(tf.io.read_file(image_path), dtype=tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, 100, 100)   # 统一成 100×100

    if training:
        # 文件名就是 GalaxyID，比如 .../100008.jpg
        name = tf.strings.split(image_path, os.path.sep)[3]
        galaxy_id = int(tf.strings.substr(name, 0, tf.strings.length(name) - 4))  # 去掉 .jpg
        row = labels_pd.loc[labels_pd['GalaxyID'] == galaxy_id]
        features = tf.cast(row.iloc[0, 1:].values, tf.float32)   # 37 个比例
        return image, features
    return image
```

这一步也顺便验证了标签和图片是一一对应的：按文件名里的数字去标签表查，能查到且只有一行。**如果这里对不上，后面训练全白费**，所以我们特意做了检查。

### 2.2 数据清洗与预处理

这个比赛的「清洗」主要不是处理脏数据，而是**把数据整理成能高效喂给 GPU 的格式**。

**第一步：划分训练集和验证集。** 我们没有用随机打乱后的行索引，而是先打乱文件路径，再按 8:2 切。这样验证集的星系和训练集完全不重叠：

```python
seed = 54
tf.random.set_seed(seed)
images_path = tf.random.shuffle(images_path)

split = int(np.floor(len(images_path) * 0.8))
train_paths = images_path[:split]    # 约 49262 张
val_paths   = images_path[split:]    # 约 12316 张
```

**第二步：用 `tf.data` 搭流水线。** 图片在硬盘上，一张一张读会很慢，所以用 `Dataset.map` 并行读取、`batch` 组批、`prefetch` 预取下一批，让 GPU 不用等数据：

```python
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

def dataset_preprocessing(paths, training=True):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if training:
        # py_function 允许在 tf.data 里调用普通 Python 函数（读图 + 查表）
        ds = ds.map(lambda x: tf.py_function(get_image_and_features,
                                             inp=[x],
                                             Tout=(tf.float32, tf.float32)),
                    num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(2048)      # 打乱缓冲区，避免顺序影响
    else:
        ds = ds.map(lambda x: tf.py_function(get_image_and_features,
                                             inp=[x, False], Tout=tf.float32),
                    num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)   # 组批 + 预取
```

`tf.py_function` 是这里的关键：TensorFlow 自带的算子没法做「按文件名查 pandas 表」这种操作，所以用 `py_function` 把它包起来。代价是不能用图编译加速，但读图本来瓶颈就在 IO 上，实践下来够用。

**第三步：图像增强。** 星系图片有一个天然的对称性——把图片左右翻、上下翻，它还是一个星系；稍微旋转一下，也不改变星系的形态类别。所以我们用了四种增强：

```python
data_augmentation = tf.keras.Sequential(name='augmentation', layers=[
    # 水平/垂直翻转：星系没有"正着放"的概念
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    # 小幅旋转：拍摄角度本身是随机的
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
    # 随机缩放：星系在画面里的大小不一
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2, 0.2),
    # 轻微对比度变化：模拟不同的曝光条件
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.05),
])
```

这里有个容易犯的错误：**旋转角度不能太大**。如果转 180 度，本来「侧对着我们」的星系看起来还是侧对，但转 90 度就可能把一个椭圆星系转成另一个朝向，虽然物理上没变，但像素分布变了，模型会困惑。我们用 0.3（大约 ±30 度以内）比较保守。

### 2.3 模型训练与超参数调优

#### 2.3.1 网络结构

我们的模型是一个小型 CNN，结构非常直白：四个卷积块，每个块是「卷积 → 批归一化 → 最大池化 → Dropout」，然后接两层全连接，最后输出 37 个值并用 sigmoid 压到 0～1。

```python
model = tf.keras.models.Sequential(name="galaxyClassifier", layers=[
    # 第 1 块：7×7 大卷积核，先抓大尺度结构（星系整体轮廓）
    layers.Conv2D(16, (7, 7), activation='relu', input_shape=(100, 100, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    # 第 2 块：6×6，继续提取中尺度特征（旋臂、棒状结构）
    layers.Conv2D(32, (6, 6), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    # 第 3 块：5×5
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    # 第 4 块：3×3，提取细节纹理
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    # 37 个输出，用 sigmoid 保证都在 0~1，符合"比例"的物理含义
    layers.Dense(37, activation='sigmoid')
])
```

设计上有几点考虑：

- **卷积核从大到小递减**（7→6→5→3）：先看整体形状，再看局部纹理，符合「先粗后细」的直觉；
- **通道数从 16 翻倍到 128**：越深特征越多，这是 CNN 的常规做法；
- **每个卷积块后都加 Dropout**：37 维输出很容易过拟合，Dropout 是最便宜的防过拟合手段；
- **最后一层用 sigmoid 而不是 softmax**：因为 37 个输出不是互斥的 37 选 1，而是各自独立的比例；用 softmax 会强制它们加起来等于 1，与真实标签不符。

整个模型只有约 **87.8 万参数**，对 6 万张图来说规模合适，不至于训练不动。

#### 2.3.2 损失函数与优化器

损失用均方误差（MSE），评价指标用 RMSE——两者是同一件事的平方关系，优化 MSE 就是在优化 RMSE：

```python
with strategy.scope():
    model = create_model((100, 100, 3), use_augmentation=True)
    # 学习率 1e-3 是 Adam 的常用起点
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
```

#### 2.3.3 训练策略

我们训练了 10 个 epoch。因为图片读盘慢，一个 epoch 要将近 10 分钟，整个训练约 **1.67 小时**。训练中准备了三个回调（callback，训练过程中自动执行的钩子函数）：

```python
# 1) 验证损失 7 个 epoch 不降，学习率乘 0.2
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=7, min_lr=1e-5)

# 2) 每 10 个 epoch 学习率乘 0.8（这里训练只有 10 轮，基本不会触发）
def decay_schedule(epoch, lr):
    return lr * 0.8 if (epoch % 10 == 0) and (epoch != 0) else lr
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(decay_schedule)

# 3) 连续 7 个 epoch 验证损失不降就提前停止
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', patience=7)

history = model.fit(train_dataset, validation_data=val_dataset,
                    epochs=10, callbacks=[], verbose=True)
```

说实话，这次训练里回调列表是空的——10 个 epoch 本来就不长，而且验证损失一直在缓慢下降，没有触发提前停止。但我们把回调写好放在代码里，是为了以后换更大模型、更长训练时可以直接用。

从训练曲线看，模型学得很「稳」：

- 第 1 个 epoch：训练 RMSE 0.144，验证 RMSE 0.127；
- 第 5 个 epoch：训练 0.115，验证 0.110；
- 第 10 个 epoch：训练 0.110，验证 0.111。

训练和验证的误差始终贴得很近，说明**没有明显过拟合**；但两者到后面几乎不再下降，说明这个模型容量已经到顶了，想再提升就得换更深的网络或更好的预训练权重。

### 2.4 模型测试与指标报告

训练完成后，我们在验证集上做最终评估：

```python
results = model.evaluate(val_dataset, verbose=1)
print("Test RMSE: {:.3f}".format(results[1]))
# 输出：Test RMSE: 0.111
```

验证集 RMSE 约 **0.111**。这是什么概念？我们可以和官方给的 `all_zeros` 基准比：如果 37 个输出全填 0，误差就等于标签本身的均方根，通常在 0.2 以上（因为大多数标签的值并不接近 0）。也就是说，我们的模型把误差压到了「全零基准」的一半左右，确实学到了星系形态与图片之间的关系。

然后我们对 79975 张测试图片做预测：

```python
# 测试集不需要标签，training=False
test_paths = tf.io.gfile.glob('../temp/images_test_rev1/*')
test_dataset = dataset_preprocessing(test_paths, training=False)

result = model.predict(test_dataset, verbose=1)
print(result.shape)     # (79975, 37)
```

得到 `(79975, 37)` 的矩阵后，按 `sample_submission.csv` 的列顺序整理成提交文件。这里有一个必须检查的点：**列顺序**。`result` 的第 0 列对应 `Class1.1`，第 36 列对应 `Class7.3`，顺序必须和标签 CSV 里 `iloc[:, 1:]` 的顺序完全一致——因为我们是按这个顺序取标签训练的，顺序天然对齐，但仍要核对一遍列名。

**我们尝试过的其他改进。** 材料里还有一条思路：把 37 个回归目标简化成 7 个「形态大类」的分类问题——用 `Class7.1`、`Class7.2`、`Class7.3` 判断整体形状（圆、介于两者之间、雪茄形），用 `Class2.1` 判断是否侧对，用 `Class3.1/3.2` 与 `Class4.1` 组合判断是否有旋臂、是否有棒。只保留那些得票比例超过 90% 的「高置信样本」，把回归变成分类，可以更快拿到一个能用的模型。但代价是丢掉了大量信息——37 维比例本身就是这个比赛的核心，所以我们主线仍然保留完整的 37 维回归。

---

## 三、总结

本文中我们完成了 Galaxy Zoo 星系形态分类挑战的完整流程。我们首先通过 EDA 确认了数据的形态：61578 张 100×100 的星系图片对应 37 个 0～1 的投票比例，测试集有 79975 张图。其次我们用 `tf.data` 搭了并行读图、组批、预取的流水线，并加入翻转、旋转、缩放、对比度四种适合天文图像的增强。接着我们构建了一个四层卷积加两层全连接的小型 CNN，卷积核从 7×7 递减到 3×3，输出层用 sigmoid 回归 37 个比例，以 MSE 为损失、Adam 为优化器训练 10 个 epoch。最后我们得到验证集 RMSE 约 0.111，明显优于全零基准，并生成了 79975×37 的提交矩阵。

这个比赛给我们最大的启发是：**「回归比例」和「分类」看起来像，其实完全不同**。用 softmax 强行让 37 个输出加起来等于 1 会违背数据本身的统计方式；而 sigmoid 加 MSE、再配合适合图像的几何增强，才是与问题结构匹配的做法。