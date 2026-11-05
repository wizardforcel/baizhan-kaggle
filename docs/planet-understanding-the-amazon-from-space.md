# Planet: 从太空理解亚马逊（Understanding the Amazon from Space）

## 一、比赛简介

### 1.1 比赛背景

这个比赛由 Planet 和 SCCON 联合发起，目的很直接：用卫星图像监测亚马逊雨林的砍伐和气候变化。Planet 是一家自己设计、制造并运营地球成像卫星的公司，它的卫星群能提供 3～5 米分辨率的影像。作为对比，过去研究森林变化主要依赖 Landsat（30 米像素）和 MODIS（250 米像素），分辨率太粗，小范围的砍伐和森林退化很容易被漏掉。分辨率提高到 3～5 米之后，才有机会从一张图里看出农田、道路、居住区、矿场这些人类活动的痕迹。

材料里有一句话很能说明紧迫性：世界上每分钟消失的森林面积相当于 48 个足球场，而亚马逊流域占了其中最大的一份。森林消失会带来生物多样性下降、栖息地丧失和气候变化等一系列问题；如果能更快、更准地知道哪里发生了砍伐，政府和当地机构就能更及时地响应。

数据最初包含红、绿、蓝和近红外（NIR）四个波段，通过 torrent 分发。后来 torrent 被关闭，社区里有人保存了完整的 RGB 版 PNG 图片，但 NIR 的 TIF 文件已经拿不到了。NIR 对森林监测很有价值，可以计算 NDVI（归一化植被指数），更清楚地看树冠和估算碳吸收；问题是常见的预训练模型都是在 RGB 三通道图片上训练的，要用 NIR 就得从零训练一个模型。综合成本和效果，我们这条主线只使用 RGB 图片。

标注数据大约 4 万张，材料里列出训练图片 40479 张、测试图片 40669 张。数据量不算大，却要做 17 类的多标签分类，所以几乎所有参赛者都选择“加载在 ImageNet 上预训练好的模型，再在 Planet 数据上微调”这条迁移学习路线。另一个大麻烦是类别极不平衡：超过 90% 的标签里都含有“原始森林（primary）”，而“传统采矿（conventional mine）”只占 0.25%。材料还提到，像 Blooming、Conventional Mine 这样的标签总共不到 500 个样本，Burn Down 甚至不到 100 个。稀有类样本少，却恰恰是发现人类活动的关键，也是能不能把分数做上去的关键。

### 1.2 比赛目标

比赛要给每张卫星图打上若干标签。注意这不是“一张图只能选一个类”的单标签分类，而是多标签分类（multilabel classification）：一张图可以同时是“晴天 + 原始森林 + 河流”，也可以同时是“有雾 + 农田 + 道路 + 居住区”。材料里说，每张图至少有 2 个标签，最多有 9 个标签，全部标签一共 17 类。

这 17 类大致可以分成两组。一组描述天气和成像质量，比如 `haze`（雾）、`clear`（晴）、`cloudy`（多云）、`partly_cloudy`（局部多云）；另一组描述地表，比如 `primary`（原始森林）、`agriculture`（农田）、`water`（水）、`road`（道路）、`habitation`（居住区）、`blooming`（开花）、`conventional_mine`（传统矿）、`slash_burn`（烧荒）等。天气标签会影响地表标签的判断，比如云层盖住地面时，模型很难看清下面到底是森林还是农田。

训练数据由两个文件组成：图片目录 `train-jpg` 和标签表 `train_classes.csv`（另一版材料里叫 `train_v2.csv`）。标签表只有两列：

- `image_name`：图片名，例如 `train_0`；
- `tags`：用空格分隔的标签串，例如 `haze primary` 或 `agriculture clear primary water`。

测试集只有图片和 `sample_submission_v2.csv` 里的 `image_name` 列，我们要给每个测试样本补上 `tags` 列，多个标签之间同样用空格隔开。

### 1.3 评价标准

比赛使用 **F2 score** 作为评价指标。要理解 F2，先要理解两个基础概念：

- **精确率（precision）**：模型预测出来的标签里，有多少是真的。比如模型说一张图有 4 个标签，其中 3 个是对的，精确率就是 3/4。
- **召回率（recall）**：真实存在的标签里，有多少被模型找出来了。比如一张图真实有 5 个标签，模型只找到 3 个，召回率就是 3/5。

F-beta 分数是精确率和召回率的加权调和平均：

```
F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
```

当 `beta = 2` 时就是 F2。这个公式里 recall 前面乘了 `beta^2 = 4`，所以召回率的权重更大。换成大白话：**漏掉一个真实标签的惩罚，比多报一个标签的惩罚更重。** 比赛选择 F2 是有现实理由的——漏掉一片正在被砍伐的森林，后果比多标注一片“疑似被砍伐”严重得多。

具体到多标签任务，评分时会拿一张图的真实标签集合和预测标签集合做比较，数出真正例（TP）、假正例（FP）、假负例（FN），算出这张图的 F2，再对所有测试图取平均。分数范围是 0 到 1，越高越好。材料里的 0.90 指的就是官方测试集上的 F2，当届比赛冠军的成绩是 0.93。也正因为指标偏向召回，我们不能只盯着“预测对了多少”，更要想办法把稀有标签找出来。

---

## 二、解题过程

我们把整场比赛拆成四步：先做数据探索，搞清楚标签长什么样、分布有多偏；再做数据清洗，把空格分隔的标签串变成 17 维的 0/1 向量，按多标签分层的方式切分训练集和验证集，并统一图片通道；接着做模型训练与调参，用预训练卷积网络做迁移学习，以二分类交叉熵为损失、以 F2 为模型选择依据；最后在验证集上搜阈值、算指标、生成提交文件。主线有两条前后衔接的方案：先跑通 PyTorch 版的 EfficientNet-B5 基线，再换成 TFRecord + MobileNetV2 的数据流水线，后者在官方测试集上拿到 0.90。

### 2.1 数据探索（EDA）

我们首先把标签表读进来，统计训练集、测试集的规模和每个标签出现的次数。做多标签任务之前，必须先知道“哪些标签很常见、哪些标签几乎见不到”，否则模型会悄悄退化成“只预测常见标签”的懒模型。

```python
import os
import pandas as pd
from collections import Counter

# 数据目录里，训练图、测试图和标签表是分开放的
path = "../input/planets-dataset/planet/planet/"
path_train = os.path.join(path, "train-jpg")
path_test = os.path.join(path, "test-jpg")

# 先核对图片数量，确认训练图和标签表能对得上
print(f"train files: {len(os.listdir(path_train))}, "
      f"test files: {len(os.listdir(path_test))}")

# 标签表只有两列：图片名 image_name 和标签串 tags
df_class = pd.read_csv(os.path.join(path, "train_classes.csv"))
print(df_class.shape)

# 标签串用空格分隔，例如 "agriculture clear primary water"
# 先按空格切开，得到每张图对应的标签列表
df_class["list_tags"] = df_class["tags"].str.split(" ")

# 把所有图的标签摊平成一个长列表，再统计每个标签出现的总次数
row_tags = df_class["list_tags"].values
tags = [tag for row in row_tags for tag in row]
counter_tags = Counter(tags)

# 按出现次数从少到多排序，稀有标签会排在最前面
df_tags = pd.DataFrame(
    {"tag": counter_tags.keys(), "total": counter_tags.values()}
).sort_values("total")
print(df_tags)
```

运行后我们看到训练图片 40479 张、测试图片 40669 张，标签表是 40479 行 2 列，和训练图片数量一致。统计结果里，`primary` 和 `agriculture` 的出现次数远远高于其他标签，而 `conventional_mine`、`blooming`、`burn` 这类标签的柱子短得几乎看不见。这就是材料里反复强调的类别不平衡：多数类占了绝大多数，少数类可能只有几十到几百个样本。如果直接拿全部数据训练、只看总体准确率，模型很可能学会“无脑预测 primary”，分数看起来不差，实际上对监测砍伐毫无用处。

只看数字还不够，我们接着为每个标签挑一张包含它的图，拼成一个网格，亲眼看看这些标签在图像上是什么样子。

```python
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# 读入训练表，并把图片名拼成完整路径
df_train = pd.read_csv("../input/train_v2.csv")
df_train["image_name"] = "../input/train-jpg/" + df_train["image_name"] + ".jpg"
df_train["tags2"] = df_train["tags"].str.split(" ")

# 收集全部出现过的标签，作为后面画图和编码的类别表
labels = sorted({tag for row in df_train["tags2"] for tag in row})

# 每个标签挑第一张包含它的图片，方便对照“标签名”和“画面”
images_title = [
    df_train[df_train["tags"].str.contains(label)].iloc[0]["image_name"]
    for label in labels
]

# 5 行 4 列的画布，逐张显示图片并写上对应标签
fig, axs = plt.subplots(5, 4, figsize=(15, 20))
axs = axs.ravel()
for i, (image_name, label) in enumerate(zip(images_title, labels)):
    img = mpimg.imread(image_name)   # 读入一张 RGB 图片
    axs[i].imshow(img)               # 显示原图
    axs[i].set_title(label)          # 标题写这个标签的名字
    axs[i].axis("off")               # 关掉坐标轴，画面更干净
plt.show()
```

看图时我们注意到几件事。第一，同一个画面里往往同时出现好几个标签，比如一张“晴天 + 原始森林 + 河流”的图，说明标签之间不是互斥关系，不能用一个 softmax 输出层解决。第二，天气对判断影响很大：`haze` 的图灰蒙蒙的，`cloudy`、`partly_cloudy` 的图有大片白云遮住地面，人眼都不容易分辨云下面是什么。第三，人类活动的痕迹在图上通常很小，可能只是几条细路、一小片规则排列的农田，这正是 30 米分辨率的 Landsat 难以发现、而 3～5 米分辨率的 Planet 影像才有希望捕捉的目标。这些观察直接决定了后面的建模方式：必须用多标签输出、必须做数据增强、必须单独监控稀有类。

### 2.2 数据清洗

我们其次处理标签和图片。这份数据的“脏”不在缺失值，而在标签的组织方式和图片的通道格式：标签是一整串用空格隔开的文本，模型没法直接吃；图片可能有四个通道，预训练网络却只认三个通道。我们先把标签串切开，再用 `MultiLabelBinarizer` 把每张图变成一个 17 维的 0/1 向量，某一维是 1 就表示这张图有这个标签。

切分训练集和验证集时，普通的随机切分或普通 K 折会带来一个问题：稀有标签的样本本来就少，随机一切，可能某一折里一个稀有样本都没有，验证分数就会剧烈波动。材料里用了多标签分层 K 折（`MultilabelStratifiedKFold`），它在切分时尽量让每一折的标签分布和全体一致，稀有类也能在每折里分到一些样本。

```python
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# 把“标签列表”编码成 17 维 0/1 向量
# 例如 [primary, water] 会变成 primary 和 water 两列为 1，其余为 0
label_encoder = MultiLabelBinarizer()
Y = label_encoder.fit_transform(df_train["tags2"]).astype(float)

# 多标签分层 K 折：切分时同时考虑 17 个标签的分布
# 普通 KFold 可能让稀有标签在某一折里完全消失，导致验证分数不可信
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)

for train_index, valid_index in mskf.split(df_train["image_name"].values, Y):
    # 按索引取出这一折的训练图片、验证图片和对应标签
    X_train = df_train["image_name"].iloc[train_index].values
    X_valid = df_train["image_name"].iloc[valid_index].values
    y_train, y_valid = Y[train_index], Y[valid_index]

    # 打印每折的标签计数，检查稀有类有没有被切没
    print(y_train.sum(0))
    print(y_valid.sum(0))
```

接着处理图片通道。材料里特别提到一个坑：数据里的 JPG 是 RGBA 四通道（红、绿、蓝、透明度），而 TensorFlow 自带的 `tf.io.decode_jpeg` 只支持 1 通道或 3 通道，直接拿去解码会得到一张全黑的图。PyTorch 这边相对简单，用 PIL 打开后加一句 `.convert("RGB")` 就能把四通道压成三通道；TensorFlow 那边则要先用 matplotlib 把图读成数组，再转成字节写进 TFRecord，读取时用 `tf.io.parse_tensor` 还原，而不是用 `decode_jpeg`。

```python
import torch
from torch.utils.data import Dataset
from PIL import Image

class PlantDataset(Dataset):
    """把“图片路径 + 17 维标签向量”包装成 PyTorch 数据集"""
    def __init__(self, paths, labels, transform=None):
        self.paths = paths          # 图片路径数组
        self.labels = labels        # 形状为 (样本数, 17) 的 0/1 矩阵
        self.transform = transform  # 训练时的数据增强，验证时只做缩放和归一化

    def __getitem__(self, index):
        # 关键一步：RGBA 四通道转成 RGB 三通道，否则预训练模型无法接收
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        # 标签转成 float 张量，配合二分类交叉熵使用
        return img, torch.from_numpy(self.labels[index]).float()

    def __len__(self):
        return len(self.paths)
```

我们只做了这些必要的清洗，没有删除稀有类样本，也没有做粗暴的过采样。原因是测试集和训练集来自同一批卫星影像，分布本来就一致；盲目复制稀有类会让模型对它们过度自信，反而在真实数据上产生大量误报。对付不平衡，我们更愿意把力气花在“选对指标”和“调好每个标签的判定阈值”上。

### 2.3 模型训练与超参数调优

我们接着搭第一个 PyTorch 基线。思路是迁移学习：拿一个在 ImageNet 上训练好的卷积网络，把最后那层“1000 类输出”换成“17 类输出”，其余权重保留，再在 Planet 数据上微调。材料里的基线用的是 `efficientnet-b5`。代码里变量名沿用了材料中的 `self.resnet`，但它实际装的是 EfficientNet，这一点看代码注释就能明白。

```python
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class PlantNet(nn.Module):
    """用 ImageNet 预训练的 EfficientNet-B5 做 17 类多标签分类"""
    def __init__(self, num_classes=17):
        super(PlantNet, self).__init__()
        # from_pretrained 会加载在普通图片上训练好的权重
        model = EfficientNet.from_pretrained("efficientnet-b5")
        # 只替换最后的全连接层：2048 维特征 → 17 个标签分数
        model._fc = nn.Linear(2048, num_classes)
        # 变量名沿用材料里的 resnet，实际骨干网络是 EfficientNet
        self.resnet = model

    def forward(self, img):
        # 输出 17 个实数（logits），每个数表示对应标签“存在”的分数
        # 训练时直接送进带 logits 的损失函数，推理时再手动过 sigmoid
        out = self.resnet(img)
        return out
```

为什么用 17 个输出而不是 1 个多分类输出？因为一张图可以同时有多个标签，17 个标签是 17 个独立的“是/否”问题。对应的损失函数是二分类交叉熵（binary cross-entropy），它对每一个输出位单独计算“预测概率和真实 0/1 差多少”，再求和。材料里也明确说，因为每个标签可以看成一个独立的二分类问题，所以选二分类交叉熵。训练循环本身很标准：

```python
import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PlantNet().to(device)

# 多标签的每个输出位独立判断，所以用二分类交叉熵
# BCEWithLogitsLoss 把 sigmoid 和交叉熵合在一起，数值上更稳定
criterion = BCEWithLogitsLoss()
# 先用 Adam 的默认学习率把流程跑通，再根据验证集 F2 调整
optimizer = Adam(model.parameters())

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()  # 切到训练模式，dropout 和 BatchNorm 正常起作用
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()          # 清空上一批留下的梯度
        output = model(data)           # 前向传播，得到 17 个分数
        loss = criterion(output, target)  # 与真实 0/1 标签算损失
        loss.backward()                # 反向传播，求每个参数的梯度
        optimizer.step()               # 按梯度更新参数
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/"
                  f"{len(train_loader.dataset)}] Loss: {loss.item():.6f}")
```

第一版基线能跑通，但训练很慢。材料里说得很实在：几万张图片、一个卷积网络，再加上 Google Colab 有限的资源，读数据成了瓶颈——每个 epoch 都要把 JPG 重新解码一遍，GPU 经常在等 CPU。于是我们对数据流水线做了改造。第一步是把原始图片提前序列化成 TFRecord，这是一种二进制文件格式，训练时直接读字节，省掉反复读文件和解码的时间。第二步是用 `tf.data.Dataset` 的 `shuffle`、`map`、`batch`、`prefetch` 把读盘、预处理、送 GPU 这几件事重叠起来做，避免 GPU 空等。材料里还试过 `cache`，想把整个数据集缓存进内存，但因为内存不够失败了。

这里还有一个必须绕开的坑：RGBA 图片不能直接用 `tf.io.decode_jpeg`。我们的做法是用 matplotlib 读图，转成字节写进 TFRecord；读的时候用 `tf.io.parse_tensor` 把字节还原成张量，再 reshape 回“高 × 宽 × 通道”的形状。下面这段是根据材料里的流水线描述整理的伪代码，缓冲区和批大小要按实际机器的内存、显存来定：

```python
import tensorflow as tf
import matplotlib.pyplot as plt

def _bytes_feature(value):
    """把 Python 字节串包装成 TFRecord 能存的 Feature"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecord(image_paths, labels, out_path):
    """把图片和标签提前写进二进制文件，训练时就不必反复解码 JPG"""
    with tf.io.TFRecordWriter(out_path) as writer:
        for path, label in zip(image_paths, labels):
            # 用 matplotlib 读图，避开 tf.io.decode_jpeg 不支持 RGBA 的问题
            image = plt.imread(path).astype("float32")
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": _bytes_feature(image.tobytes()),
                "label": _bytes_feature(label.astype("float32").tobytes()),
            }))
            writer.write(example.SerializeToString())

def parse_example(serialized):
    """从 TFRecord 里读出一条样本，还原成图片张量和标签张量"""
    features = tf.io.parse_single_example(serialized, features={
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    })
    # 材料里明确说不用 decode_jpeg，而用 parse_tensor 还原
    image = tf.io.parse_tensor(features["image"], out_type=tf.float32)
    label = tf.io.parse_tensor(features["label"], out_type=tf.float32)
    # 再 reshape 回“高 × 宽 × 通道”，具体尺寸按数据实际情况填
    image = tf.reshape(image, [height, width, channels])
    return image, label

# shuffle → map → batch → prefetch，让读盘和 GPU 计算并行
dataset = (tf.data.TFRecordDataset("train.tfrec")
           .shuffle(buffer_size=shuffle_buffer)
           .map(parse_example)
           .batch(batch_size)
           .prefetch(tf.data.AUTOTUNE))
```

在模型上，我们换成了更轻的 MobileNetV2。EfficientNet-B5 精度不错，但参数量大、训练慢；MobileNetV2 用深度可分离卷积，计算量小得多，在 Colab 这种资源受限的环境里能多跑几轮、多试几组参数。材料标题里的 `[0.90] MobileNetV2 Optimized with TFRecord` 说的就是这条路线：MobileNetV2 骨干 + 17 路 sigmoid 输出 + TFRecord 数据流水线，最后在官方测试集上得到 0.90。下面这段模型代码是我们按这条主线整理的写法，结构上保持了“预训练骨干 + 全局平均池化 + 17 个 sigmoid 输出”。

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from tensorflow_addons.metrics import FBetaScore

# 加载 ImageNet 预训练的 MobileNetV2，去掉原来的 1000 类分类头
# 输入尺寸按数据里的实际图片大小设置，通道必须是 RGB 三通道
base = MobileNetV2(weights="imagenet", include_top=False,
                   input_shape=(img_height, img_width, 3))
# 在 Planet 数据上微调整个网络，而不只是训练最后一层
base.trainable = True

# 全局平均池化把每个特征图压成一个数，减少参数量、抑制过拟合
x = layers.GlobalAveragePooling2D()(base.output)
# 17 个输出对应 17 个标签，sigmoid 让每个输出独立落在 0~1 之间
outputs = layers.Dense(17, activation="sigmoid")(x)
model = Model(base.input, outputs)

# 多标签 + 17 个独立标签 → 二分类交叉熵
# FBetaScore(beta=2.0) 就是比赛用的 F2，训练时可以直接盯着它看
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=[FBetaScore(beta=2.0)])
```

超参数方面，我们主要调了学习率、批大小、数据增强的强度和训练轮数。数据增强用了随机翻转、随机裁剪和亮度扰动，因为卫星图旋转、翻转之后语义不变，而亮度扰动可以模拟不同天气和光照。调参的判据不是训练损失，而是验证集 F2：训练损失降了但 F2 不涨，说明模型只是在记住多数类，这种“假进步”不能要。材料里的最终结果也印证了这一点：训练集损失 0.09、F2 0.90，验证集损失 0.11、F2 0.89，两者很接近，说明模型没有严重过拟合。

### 2.4 模型测试与指标报告

我们最后在验证集上做预测和评分。模型输出的是 17 个 logits，先过 sigmoid 变成 0～1 的概率，再按阈值判断每个标签“出现/不出现”。评分用 scikit-learn 的 `fbeta_score`，把 `beta` 设为 2，`average` 设为 `samples`——也就是先对每张图单独算 F2，再对所有图取平均，和比赛的算法一致。

```python
import numpy as np
import torch
from sklearn.metrics import fbeta_score

model.eval()  # 切到推理模式，关掉 dropout，固定 BatchNorm 的统计量
all_probs, all_targets = [], []

with torch.no_grad():  # 推理不需要梯度，省显存也更快
    for data, target in valid_loader:
        data = data.to(device)
        # 模型输出 logits，sigmoid 后变成每个标签的概率
        probs = torch.sigmoid(model(data)).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(target.numpy())

y_prob = np.vstack(all_probs)      # 拼成 (验证样本数, 17)
y_true = np.vstack(all_targets)    # 真实 0/1 标签

# 阈值 0.5：概率超过一半就认为这个标签存在
preds = (y_prob >= 0.5).astype(int)

# beta=2 让召回率权重更大；average="samples" 表示逐图算 F2 再平均
f2 = fbeta_score(y_true, preds, beta=2, average="samples")
print("Validation F2:", f2)

# 再按标签逐个算 F2，专门盯住稀有类有没有被模型忽略
per_label_f2 = fbeta_score(y_true, preds, beta=2, average=None)
for name, score in zip(label_encoder.classes_, per_label_f2):
    print(f"{name:20s} {score:.4f}")
```

只看总体 F2 是不够的。多数类样本多，即使稀有类全错，总体分数也可能看起来不错；所以我们会同时打印每个标签的 F2，重点看 `conventional_mine`、`blooming`、`burn` 这些稀有类。如果某个稀有类的 F2 接近 0，说明模型基本没学会它，这时候可以单独调低这个标签的判定阈值——阈值降低，模型更愿意报出这个标签，召回率上升，正好符合 F2 偏爱召回的方向。材料里提到有专门的“最佳阈值搜索”思路，我们也是在验证集上按标签扫一遍阈值，选使该标签 F2 最高的切点，再把整组阈值用到测试集。

材料里这条 MobileNetV2 + TFRecord 主线的最终成绩是：官方测试集 F2 为 **0.90**，当届冠军是 **0.93**。训练集 F2 0.90、验证集 F2 0.89，和测试集 0.90 很接近，说明验证集切分比较可信，模型也没有在测试集上“翻车”。和冠军的差距不算大，但从材料看，剩下的提升空间主要在稀有标签、云雾干扰和更充分的训练/集成上。

确认模型可用后，我们把预测结果还原成标签名，生成提交文件。这里有两个容易出错的地方：一是 `inverse_transform` 只能处理 0/1 矩阵，不能直接喂概率；二是标签之间必须用空格连接，列名必须是 `image_name` 和 `tags`。

```python
# 把 17 维 0/1 预测还原成标签名列表
# inverse_transform 会把每一行里为 1 的位置映射回对应的标签
pred_tags = label_encoder.inverse_transform(preds)

# 测试图片名从 sample_submission_v2.csv 里按顺序取，保证行对齐
test_names = pd.read_csv("../input/sample_submission_v2.csv")["image_name"]

# 一行一个图片名，多个标签用空格拼起来
submission = pd.DataFrame({
    "image_name": test_names,
    "tags": [" ".join(tags) for tags in pred_tags],
})
submission.to_csv("submission.csv", index=False)
```

提交前我们还会做两项检查：行数是否和测试集一致，标签名是否都在 17 类里面。这两件事看起来琐碎，但比赛里因为格式错误拿不到分数的情况并不少见。

---

## 三、总结

本文中我们完成了 Planet“从太空理解亚马逊”比赛的完整流程。我们首先用 EDA 摸清了数据：训练图片 40479 张、测试图片 40669 张，每张图有 2～9 个标签、共 17 类，类别严重不平衡，`primary` 出现在 90% 以上的标签里，而 `conventional_mine` 只占 0.25%，部分稀有标签不到 500 个样本甚至不到 100 个。其次我们做了数据清洗：把空格分隔的标签串编码成 17 维 0/1 向量，用多标签分层 K 折切分训练集和验证集，把 RGBA 四通道图片统一转成 RGB，并绕开了 `tf.io.decode_jpeg` 不支持 RGBA 的坑。接着我们用 ImageNet 预训练的 EfficientNet-B5 跑通 PyTorch 基线，以二分类交叉熵为损失，再把数据流水线改造成 TFRecord + `tf.data.Dataset`，模型换成更轻的 MobileNetV2，用验证集 F2 而不是训练损失来选模型和调阈值。最后我们在官方测试集上得到 0.90 的 F2，训练集 F2 0.90、验证集 F2 0.89，与测试集基本一致；当届冠军是 0.93。

这个比赛给我们最大的启发有三点。第一，**多标签任务要把每个标签当成独立的二分类问题**，用 sigmoid 加二分类交叉熵，而不是 softmax 加多分类交叉熵，否则标签之间会互相抢概率。第二，**不平衡数据要先选对指标**：F2 把召回率放在更重要的位置，正好对应“宁可多报、不可漏报”的业务需求；再配合逐标签阈值调整，比盲目过采样更可控。第三，**数据读取往往是深度学习的隐形瓶颈**：把 JPG 提前转成 TFRecord，再用 `prefetch` 让读盘和计算重叠，训练速度的提升可能比换一个更复杂的模型还明显。材料里其他参赛者还尝试过 fast.ai、Keras、ResNet 等路线，但核心思想都一样——预训练卷积网络做迁移学习，多标签输出，用 F2 盯住稀有类。
