# Cdiscount 商品图片分类（Image Classification Challenge）

## 一、比赛简介

### 1.1 比赛背景

Cdiscount 是法国的一家大型电商平台，这个比赛就是他们把自己的商品图片拿出来，让参赛者训练模型自动判断「这张图是哪一类商品」。对电商来说这件事很实际：平台的商品目录有上百万条，人工给每件商品打标签成本极高，如果能用图片自动分类，就能大幅降低上新和纠错的成本。

根据比赛数据页的说明，数据集大概是这样的规模：接近 **900 万件商品**（差不多是平台商品目录的一半），**1500 多万张** 180×180 分辨率的图片，涵盖 **5000 多个类目**。这在当年算是非常「极端」的多分类任务——不是 10 类、100 类，而是 5270 类，很多类目之间长得还很像。

数据以 BSON（Binary JSON，一种二进制的 JSON 格式）存放。训练文件 `train.bson` 里有 **7,069,896 条**商品记录，每条记录包含三样东西：

- `_id`：商品编号；
- `category_id`：商品类目编号，测试集里没有；
- `imgs`：一个列表，里面装 1～4 张图片，每张图片又是一个字典，键 `picture` 存着 JPEG 的二进制字节串。

测试文件 `test.bson` 只有 `_id` 和 `imgs`，没有 `category_id`。另外还有一个 `category_names.csv`，给出了三级类目层级：一级最粗（比如「手机 - GPS」），二级居中，三级最细（比如「手机壳 - 保险杠」）。模型要预测的 `category_id` 对应最细的那一级。

这个比赛最大的难点不是模型结构，而是**数据量**：7 百万条商品、15 百万张图片，光是把 BSON 读一遍就要很久，内存和磁盘 I/O 才是真正的瓶颈。

### 1.2 比赛目标

输入是一张（或一组）商品图片，输出它属于 5270 个类目中的哪一个。提交文件包含 `_id` 和 `category_id` 两列，每个测试商品一行。

### 1.3 评价标准

比赛按**分类准确率（categorization accuracy）**评分：预测的 `category_id` 和真实类目完全一致的商品，占总数的比例。因为是 5270 选 1，随机乱猜的准确率只有大约 0.0002，所以哪怕只有 0.6 的准确率，也已经说明模型学到了东西。这个指标越高越好。

需要提醒的是，材料里所有模型训练日志和排行榜分数（0.65、0.68）都是这个准确率口径下的数字，不要和「top-5 准确率」混为一谈。

---

## 二、解题过程

整场比赛的主线可以概括成一句话：**先把 BSON 里的图片高效地读出来，再用在 ImageNet 上预训练好的卷积网络做迁移学习，把最后的分类头换成 5270 类。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先写一个最小的 BSON 读取程序，看看一条记录到底长什么样。这里用 `bson.decode_file_iter`，它是一个生成器，一条一条吐数据，不会把几十 GB 的文件一次性读进内存。

```python
import io, bson
import numpy as np, pandas as pd
import cv2

# 用生成器逐条读，避免把整个 train.bson 读进内存
data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))

rows = []
for d in data:
    product_id  = d['_id']
    category_id = d['category_id']      # 测试集没有这一列
    for pic in d['imgs']:               # 一件商品有 1~4 张图
        # picture 存的是 JPEG 字节串，要先解码成像素数组
        arr = np.frombuffer(pic['picture'], np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        rows.append((product_id, category_id, img))

df = pd.DataFrame(rows, columns=['product_id', 'category_id', 'img'])
print(df.shape)
print('商品数:', df['product_id'].nunique())
print('类目数:', df['category_id'].nunique())
```

打印结果告诉我们两件事：图片确实是 180×180 的彩色图；同一件商品的多张图共用同一个 `category_id`，所以「按商品」和「按图片」是两种不同的统计口径，做验证集划分时一定要按商品切，不能让同一件商品的图同时出现在训练和验证里，否则分数会虚高。

接着我们看类目表，确认三级层级的含义：

```python
cat = pd.read_csv('../input/category_names.csv')
print(cat.shape)        # 5270 行，一行一个类目
print(cat.head())

# 一级、二级、三级是逐层细化的：level3 才对应我们要预测的 category_id
levels = cat[['category_id', 'category_level1', 'category_level2', 'category_level3']]

# 统计训练集里每类有多少件商品，检查类别是否均衡
counts = df.drop_duplicates('product_id')['category_id'].value_counts()
print(counts.describe())
print('商品最多的一类:', counts.index[0], counts.iloc[0])
```

类别分布明显不均衡：热门类目的商品数以万计，冷门类目只有几十件。材料里有人专门做了「把所有商品都猜成出现最多的那个类目」的 baseline，正说明这种偏斜足够严重，否则这种 baseline 根本毫无意义。

### 2.2 数据清洗

这份数据没有「缺失值」的概念，真正的清洗工作全在 I/O 上。如果每次训练都从头解 BSON，一个 epoch 可能要好几个小时。所以我们做了两件事：第一，给每个商品建一张「偏移量表」，之后可以 O(1) 随机读取；第二，把图片预先切成固定格式的 `.bin` 文件，训练时直接按批读字节。

先看偏移量表：

```python
import struct
from tqdm import tqdm

# train.bson 是「4 字节长度 + BSON 文档」的连续流。
# 先扫一遍，记下每件商品的 (起始偏移, 长度)，之后想读谁就 seek 到谁
IDS_MAPPING = {}
num_dicts = 7_069_896          # 数据页给出的商品总数
with open('../input/train.bson', 'rb') as f, tqdm(total=num_dicts) as bar:
    offset = 0
    while True:
        bar.update()
        f.seek(offset)
        head = f.read(4)       # 前 4 个字节是这一条的长度
        if len(head) == 0:
            break              # 读到文件末尾
        length = struct.unpack('<i', head)[0]
        f.seek(offset)
        item = bson.BSON(f.read(length)).decode()
        IDS_MAPPING[item['_id']] = (offset, length)
        offset += length

def get_item(item_id):
    """随机读取任意一件商品，用于抽查和可视化。"""
    offset, length = IDS_MAPPING[item_id]
    with open('../input/train.bson', 'rb') as f:
        f.seek(offset)
        return bson.BSON(f.read(length)).decode()
```

材料里的实测是：扫一遍建表大约 3 分钟。这 3 分钟很值——之后不管做可视化、检查坏图，还是按 ID 抽样，都不用再从头解析整个文件。

然后是训练用的生成器。我们自己切好的 `.bin` 文件格式很简单：每张图前面写 8 个字节，前 4 个字节是类别编号，后 4 个字节是图片字节数，紧接着是 JPEG 数据。这样按顺序读就能还原出「图 + 标签」。

```python
import threading
from keras.preprocessing.image import Iterator, load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input

class BinFileIterator(Iterator):
    """从切好的 .bin 文件按批读图，避免每个 epoch 重新解 BSON。"""
    def __init__(self, bin_file, samples, target_size=(180, 180),
                 batch_size=128, num_class=5270):
        self.file = open(bin_file, 'rb')
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)
        self.num_class = num_class
        self.lock = threading.Lock()    # 多个线程读同一个文件句柄，必须加锁
        super().__init__(samples, batch_size, shuffle=False, seed=None)

    def _flow(self, index_array):
        X = np.zeros((len(index_array),) + self.image_shape, dtype='float32')
        Y = np.zeros((len(index_array), self.num_class), dtype='float32')
        for i, j in enumerate(index_array):
            with self.lock:
                head = self.file.read(8)          # 8 字节：类别编号 + 图片长度
                if len(head) < 8:                 # 读到头就绕回文件开头
                    self.file.seek(0)
                    head = self.file.read(8)
                cls, length = struct.unpack('<ii', head)
                buf = self.file.read(length)
            img = load_img(io.BytesIO(buf), target_size=self.target_size)
            X[i] = img_to_array(img)
            Y[i, cls] = 1                         # 转成 one-hot 标签
        # Inception / Xception 要求把像素从 0~255 缩放到 -1~1
        return preprocess_input(X), Y

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._flow(index_array[0])
```

这里有一个容易踩的坑：`self.file` 是共享的文件句柄，多个 worker 同时 `read` 会让字节错位，所以必须在读 8 字节头和读图片体的整段过程里都加锁。

### 2.3 模型训练与超参数调优

7 百万件商品、5270 个类目，从零训练一个卷积网络几乎不可能收敛。所以我们用**迁移学习**：拿在 ImageNet（1000 类自然图像）上预训练好的网络，扔掉它原来的分类头，换成 5270 类的新头，再用商品图片微调。

材料里的主线是 InceptionV3，输入直接设成 180×180×3，和比赛图片一致，省掉一次缩放：

```python
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model

# 输入固定 180x180x3，和比赛图片分辨率一致
inp = Input(shape=(180, 180, 3))
# include_top=False：只保留卷积特征提取器，扔掉 ImageNet 的 1000 类分类头
base = InceptionV3(input_tensor=inp, weights='imagenet', include_top=False)

x = base.output
x = GlobalAveragePooling2D()(x)          # 全局平均池化，比 Flatten 参数少得多
pred = Dense(5270, activation='softmax')(x)   # 5270 个商品类目
model = Model(inputs=base.input, outputs=pred)

# 第一阶段只训练新分类头，卷积层全部冻结，避免一上来就把预训练权重带偏
for layer in base.layers:
    layer.trainable = False
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

后面效果更好的版本换成了 Xception，并且用 3 张 GPU 并行训练。它的训练流程分三步，层层解冻，学习率越来越小：

```python
from keras.applications.xception import Xception
from keras.optimizers import SGD, Adam
from keras.metrics import top_k_categorical_accuracy

base = Xception(include_top=False, weights='imagenet', input_shape=(180, 180, 3))
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)                  # 5270 类、每类样本有限，Dropout 抑制过拟合
out = Dense(len(classnames), activation='softmax')(x)
model = Model(base.input, out)

# 第一步：冻结整个 Xception，只训练分类头，用较大的学习率快速找到方向
for layer in base.layers:
    layer.trainable = False
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.001, momentum=0.9),
              metrics=[top_k_categorical_accuracy, 'accuracy'])
model.fit_generator(train_generator,
                    steps_per_epoch=math.ceil(2_000_000 / batch_size),
                    epochs=3, validation_data=validation_generator,
                    validation_steps=math.ceil(10_000 / batch_size))

# 第二步：只解冻后面几段（block10~block14），用更小的学习率小心微调
for layer in base.layers:
    if layer.name.split('_')[0] in ['block%d' % i for i in range(10, 15)]:
        layer.trainable = True
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.00025),
              metrics=[top_k_categorical_accuracy, 'accuracy'])
model.fit_generator(train_generator,
                    steps_per_epoch=math.ceil(2_000_000 / batch_size),
                    epochs=8, initial_epoch=3,
                    validation_data=validation_generator,
                    validation_steps=math.ceil(10_000 / batch_size))
```

最后一步是全部解冻，并且让学习率每两轮乘 0.95 慢慢衰减。这个「先冻后解」的顺序很关键：如果一上来就全量微调，预训练权重会被打散，小数据集上很容易崩。

```python
from keras import backend as K

# 第三步：全部解冻，边训练边指数衰减学习率，让模型慢慢收敛
for layer in base.layers:
    layer.trainable = True
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.00025),
              metrics=[top_k_categorical_accuracy, 'accuracy'])

init_epochs = 8                     # 前两步已经训练了 8 轮
for i in range(100):
    # 每两轮把学习率乘以 0.95，前期步子大、后期步子小
    K.set_value(model.optimizer.lr, 0.95 * K.get_value(model.optimizer.lr))
    model.fit_generator(train_generator,
                        steps_per_epoch=math.ceil(2_000_000 / batch_size),
                        initial_epoch=init_epochs + i * 2,
                        epochs=init_epochs + (i + 1) * 2,
                        validation_data=validation_generator,
                        validation_steps=math.ceil(10_000 / batch_size))
```

在超参数上，我们主要调了这几个：batch_size 用 3 张 GPU 合计 258（每张卡 86）；每个 epoch 只抽 200 万张图，而不是把 700 万张全跑一遍，这样一轮的时间可控；验证集固定 1 万张图。数据增强用的是「随机缩放 10%、随机平移 10%、水平翻转、像素缩放到 0～1」，没有做颜色抖动——材料里的经验是「这个网络不太过拟合，不需要太重的增强」。

### 2.4 模型测试与指标报告

训练过程中我们主要通过验证集准确率来判断模型是不是在进步。材料里的训练日志是这样的：

```python
# 材料中每个 epoch 结束会打印训练 / 验证的 loss 和 accuracy
log = [
    # epoch, train_loss, train_acc, valid_loss, valid_acc
    (1, 4.243194, 0.320435, 3.313541, 0.434122),
    (2, 2.952012, 0.480100, 2.863293, 0.496674),
]
for epoch, tr_loss, tr_acc, va_loss, va_acc in log:
    print(f'epoch {epoch}: train acc {tr_acc:.3f}  valid acc {va_acc:.3f}')
```

两个 epoch 之后，验证准确率从 0.434 涨到 0.497，训练准确率从 0.320 涨到 0.480，loss 也在稳定下降。这说明模型确实在学习，而不是在瞎猜。最终材料里给出的排行榜成绩是：VGG 风格的基线卷积网络约 **0.65**，Xception 版本在私有榜上达到 **0.68+**。

生成提交文件时，`sample_submission.csv` 的索引就是 `_id`，我们只需要把预测的类目填进 `category_id` 列。材料里还展示了一个「最朴素」的提交方式，用来在模型跑不动时保底：

```python
# 读入官方提交样例，索引列就是 _id
submission = pd.read_csv('../input/sample_submission.csv', index_col='_id')

# 兜底策略：所有商品都猜出现次数最多的那一类，先保证提交格式合法
most_frequent_guess = 1000018296
submission['category_id'] = most_frequent_guess
submission.to_csv('submission.csv.gz', compression='gzip')
```

这一步看起来不起眼，但在比赛里非常实用：先有一份能提交的文件，再去慢慢优化模型，心里会踏实很多。

---

## 三、总结

本文中我们完成了 Cdiscount 商品图片分类的完整流程。我们首先用 BSON 流式读取摸清了数据：7,069,896 条商品记录、每件 1～4 张 180×180 图片、5270 个类目、三级类目层级，以及明显的类别不均衡。其次我们把 I/O 当成主要清洗工作：建了偏移量表做随机读取，又切出定长的 `.bin` 文件配合加锁的自定义生成器，避免每个 epoch 重解 BSON。接着我们用 InceptionV3 做迁移学习，先冻结卷积层只训分类头，再逐段解冻；效果更好的版本换成 Xception，用 3 张 GPU、batch 258、SGD 起步、Adam 微调、学习率每两轮乘 0.95。最后我们通过验证集准确率监控训练，得到 VGG 基线约 0.65、Xception 约 0.68+ 的排行榜成绩，并用官方提交样例格式生成提交文件。

这个比赛给我们的最大启发是：**在大规模图像任务里，数据管道往往比模型结构更决定成败**。同样是 InceptionV3，如果每个 epoch 都从头解 BSON，训练根本跑不完几轮；而把读取、缓存、多进程、随机访问这些工程问题解决好，模型才有机会发挥。另外，「先冻结、后解冻、再衰减学习率」这套微调节奏，也是迁移学习里非常值得记住的模板。
