# CVPR 2018 自动驾驶挑战赛（Autonomous Driving）

## 一、比赛简介

### 1.1 比赛背景

自动驾驶汽车在行驶时要「看懂」周围的路况，其中最基本的一项能力就是：从车载摄像头拍到的画面里，把路上出现的车、行人等物体一个一个找出来，像涂色一样把每个物体占的那片像素涂上标记。这个任务在计算机视觉里叫**语义/实例分割**（image segmentation），是自动驾驶感知模块的核心前置步骤。

这场比赛由 CVPR 2018 自动驾驶研讨会上组织。数据来自一辆车绕城行驶时，用安装在车上的摄像机从驾驶视角录制的视频。我们把视频拆成一帧一帧的图像，参赛者要做的就是给每一帧图像里的物体画掩膜（mask）。因为画面来自真实的公路场景，物体互相遮挡、距离远近悬殊、天气光线多变，所以这个任务相当有挑战性。

### 1.2 比赛目标

输入是一张行车场景的彩色图像，输出是图像中每个物体对应的「掩膜」——一张和原图同样大小的图，物体所在的像素被标记出来。

数据集分三个目录加一个提交样例：

- `train_color/`：约 3.9 万张训练彩色图（jpg），共 92.3 GB；
- `train_label/`：与训练图一一对应的掩膜标注（png）；
- `test/`：1917 张测试图，文件名是一串哈希值；
- `sample_submission.csv`：提交格式样例。

训练图的文件名形如 `170908_061523257_Camera_5.jpg`：前半段是拍摄日期时间，后半段说明这是哪台相机拍的，同一时刻通常有 Camera 5 和 Camera 6 两台相机同时记录。对应的掩膜文件名是 `170908_061523257_Camera_5_instanceIds.png`。

掩膜的编码方式非常巧妙，一个像素值同时包含了「类别」和「实例编号」两层信息：

```
像素值 P = 类别标签 × 1000 + 实例编号
```

例如像素值 33000 表示「第 0 号车」（类别 33 是 car，实例 0），33001 表示「第 1 号车」。这样同一类里的多辆车可以分开，一张图里每个物体都有自己独立的编号。

比赛中主要考虑的物体类别有：轿车（car）、摩托车（motorbicycle）、自行车（bicycle）、行人（person）、卡车（truck）、公交车（bus）、三轮车（tricycle）等 7 类。

### 1.3 评价标准

提交文件按「实例」来组织，每一行代表一个检测到的物体，列的顺序固定为：

| 列名 | 含义 |
| --- | --- |
| ImageId | 图像编号 |
| LabelId | 物体类别编号 |
| Confidence | 置信度 |
| PixelCount | 掩膜的像素数 |
| EncodedPixels | 掩膜的 RLE 编码（把连续像素段压缩成「起点+长度」的字符串） |

比赛的评分本质上是比较我们预测的掩膜和真实掩膜的重合程度，用的核心指标是 **IoU（Intersection over Union，交并比）**：两个区域重叠的面积除以两者并集的面积，越接近 1 说明画得越准。许多选手在本地用平均 IoU（mean IoU）作为模型的训练与验证指标，跟比赛的思路保持一致。

---

## 二、解题过程

整场比赛的思路可以概括为：**先读懂标签里「类别+实例」合一的编码，把任务化成二值分割问题，再训练一个 U-Net 网络输出车辆掩膜，最后把连续的概率图切分成独立物体并做 RLE 编码提交。** 下面按四步来走。

### 2.1 数据探索（EDA）

我们首先快速看一眼数据文件的结构，并分析文件名：

```python
import os
# 列一下输入目录，看看比赛给了哪些东西
os.listdir('../input')
# 输出：['test', 'train_label', 'sample_submission.csv', 'train_color']

# 统计 train_color 目录里有多少张 jpg、总共多大
img_files = os.listdir('../input/train_color')
print(len(img_files))   # 39221
```

训练彩色图有 39221 张，体量有几十 GB。命中注定的第一印象是：**单张图很大、数量很多，训练的难点首先在「怎么喂给模型」而不是模型本身。**

接下来我们看掩膜里到底存了什么。掩膜是 16 位 PNG，直接当普通图片读会变成灰度图，信息就丢了。正确的打开方式是把每个像素的值除以 1000 取整数部分，得到类别：

```python
import numpy as np
import pandas as pd
from PIL import Image

# 类别编号 → 类别名字的对照表，来自比赛数据说明
classdict = {0:'others', 33:'car', 34:'motorbicycle', 35:'bicycle',
             36:'person', 38:'truck', 39:'bus', 40:'tricycle',
             49:'road', 65:'traffic_cone', 113:'vegatation', 255:'unlabeled'}

# 读一张掩膜，像素值 // 1000 得到类别号
tlabel = np.asarray(Image.open(
    '../input/train_label/170908_061523257_Camera_5_instanceIds.png')) // 1000

# 统计每个类别出现的次数，并转成一行 DataFrame 方便看
cls, counts = np.unique(tlabel, return_counts=True)
d = dict(zip(cls, counts))
df = pd.DataFrame.from_dict(d, orient='index').transpose()
df.rename(columns=classdict, inplace=True)
df
```

从结果可以看到，一张图里通常同时存在 several种物体，比如同时有车、公交车和交通锥。我们再用一个脚本把**全部**训练掩膜读一遍，统计每个类别在所有图中出现的频率，得到类别分布：

```python
from glob import glob

# 把每个类的像素占比做成一行数据，合并成整份训练集的类别统计
def get_label_info(in_path):
    idx_image = np.asarray(Image.open(in_path)) // 1000   # 提取类别号
    count_dict = {k: (idx_image == k).sum() / idx_image.size
                  for k in np.unique(idx_image)}
    return count_dict

label_paths = glob('../input/train_label/*.png')
all_counts = [get_label_info(p) for p in label_paths]
class_freq = pd.DataFrame(all_counts).fillna(0)
class_freq.sum().sort_values(ascending=False)
```

统计结果最突出的一点是：**car（轿车）出现的面积占比远超其他类**，而很多类别（比如桥、隧道、墙）在绝大多数图里压根不出现。这就是典型的**类别极度不平衡**——如果直接拿所有类别一起训练，模型会偏向数量多的车，忽略罕见类。这决定了我们后面的策略：先聚焦数量最多、也最重要的车辆。

### 2.2 数据清洗

清洗的第一个任务是把「类别+实例」混合编码的掩膜变成模型能用的干净格式。我们的做法是：把像素值除以 1000 得到类别，然后只关心「是不是车辆」，把它压成一个二值图（像素是车 = 1，不是 = 0）。这就把一个多类别分割问题简化成了二值分割，门槛低很多：

```python
from skimage.io import imread

# 把车、卡车、公交车等「车辆」类挑出来
car_classes = ['bus', 'car', 'bus_group', 'car_groups', 'truck', 'truck_group']
class_dict = {'car': 33, 'bus': 39, 'truck': 38,
              'car_groups': 161, 'bus_group': 167, 'truck_group': 166}
car_idx = [v for k, v in class_dict.items() if k in car_classes]

def read_label_image(in_path):
    """读掩膜，只保留车辆像素，输出 0/1 二值图"""
    idx_image = imread(in_path) // 1000          # 取类别号
    return np.isin(idx_image.ravel(), car_idx) \
        .reshape(idx_image.shape).astype(np.float32)
```

两个细节值得一提：一是这里用 `skimage` 而不是 PIL 来读图，因为 `skimage` 能正确处理 16 位无符号整数，而 PIL 对 16 位图支持不好；二是我们的读图函数把它包进一个「把路径换成掩膜」的读图器，配合 Keras 的图片生成器使用，这样 `ImageDataGenerator` 就能一边做数据增强、一边自动把彩色图和掩膜一起准备好。

针对类别不平衡，我们还做了**样本筛选**：每张图先算一下「车辆的占比」，然后只拿车辆最多的前 1000 张图来训练。这样模型看到的几乎每一张图里都有车，训练信号充足，代价是放弃了一些几乎纯背景的图：

```python
# 计算每张图的车辆面积占比，按降序排，取前 1000 张
def total_car_vol(in_row):
    return sum(in_row[k] for k in car_classes)

group_df['total_vehicle'] = group_df.apply(total_car_vol, 1)
train_df = group_df.sort_values('total_vehicle', ascending=False).head(1000)
```

### 2.3 模型训练与超参数调优

模型我们选择了 **U-Net**，它是最经典的分割网络之一：左边是卷积层逐层下采样提取特征（编码器），右边用转置卷积逐层上采样恢复分辨率（解码器），中间把同尺度的特征**拼接（skip connection）**起来，让定位信息和语义信息互补。

我们先用 `train_test_split` 把 1000 张图按 3:1 划分成训练集（750 张）和留出集（250 张），并构造两个数据流：一个产生 RGB 彩色图，一个产生对应掩膜，两者用同一个随机种子保证成对：

```python
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

# 划分训练/留出集，固定随机种子保证可复现
train_split_df, valid_split_df = train_test_split(
    train_df, random_state=2018, test_size=0.25)

# 统一缩放到 512x512：车辆往往比较小，高分辨率才能看清
IMG_SIZE = (512, 512)

# 彩色图的数据增强：轻微平移、旋转、缩放、水平翻转，并做 VGG 预处理的归一化
img_gen_args = dict(horizontal_flip=True, height_shift_range=0.05,
                    width_shift_range=0.02, rotation_range=3,
                    shear_range=0.01, zoom_range=0.05, fill_mode='nearest')
rgb_gen = ImageDataGenerator(preprocessing_function=preprocess_input, **img_gen_args)
lab_gen = ImageDataGenerator(**img_gen_args)   # 掩膜只增强、不做颜色归一化
```

接下来是模型的骨干。我们用 Keras 搭了一个带**全连接层（FCL）分支**的 U-Net：在瓶颈处除了正常的卷积特征，还加了一条「展平 → 全连接 → 重塑」的路径，相当于是让网络对整张图有一个全局的感知，再和局部特征拼在一起用。通道数从 8 逐层翻倍到 128，深度很浅，方便在有限的显存里跑 512×512 的图：

```python
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout, Flatten, Reshape, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

inputs = Input(IMG_SIZE + (3,))
s = BatchNormalization()(inputs)
s = Dropout(0.5)(s)

# 编码器：三层卷积后用 ReLU，再接最大池化下采样
c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(s)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)
# ... c2/c3/c4 依次把通道数翻倍（16/32/64），同样接池化
c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

# 全连接分支：把最深层特征展平，过两层全连接再重塑回 2 通道
flat_c5 = Dropout(0.5)(Flatten()(c5))
fcl_c5 = Dropout(0.5)(Dense(256)(flat_c5))
out_shape = c5._keras_shape[1:3] + (2,)
fcl_img = Reshape(out_shape)(Dense(np.prod(out_shape))(fcl_c5))

# 和卷积特征拼起来，作为解码器的输入
new_c5 = concatenate([c5, fcl_img])

# 解码器：转置卷积上采样，与同尺度编码器特征拼接
u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(new_c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
# ... 依次上采样回 512x512

# 最后用一个 1x1 卷积输出单通道，sigmoid 把输出压到 0~1 当成概率
outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
model = Model(inputs=[inputs], outputs=[outputs])
```

参数总量约 154 万，属于轻量级网络。训练时我们用二值交叉熵作为损失，并让 Keras 同时计算 IoU 指标来实时观察分割质量：

```python
import tensorflow as tf
from keras import backend as K

def mean_iou(y_true, y_pred):
    """把 0/1 掩膜的 IoU 作为训练监控指标"""
    preds = K.cast(K.greater(y_pred, 0.5), 'float32')  # 概率>0.5 视为正样本
    smooth = 1e-5
    intersection = K.sum(y_true * preds) + smooth
    union = K.sum(y_true) + K.sum(preds) - intersection + smooth
    return intersection / union

# 编译：二值交叉熵损失 + mean IoU 指标
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

model.fit_generator(train_and_lab_gen, epochs=30, steps_per_epoch=100,
                    validation_data=valid_and_lab_gen,
                    validation_steps=30)
```

之所以用生成器 `fit_generator` 而不是一次性载入所有数据，是因为原图太大、内存装不下去，必须边读边训练。超参数上，除了上面提到的 512×512 分辨率，我们还通过实验固定了 batch size = 8，这是平衡「显存占用」和「梯度稳定」之后的折中。

### 2.4 模型测试与指标报告

训练完成后，我们把测试集的 1917 张图逐张喂给模型，得到每张图的车辆概率图。但比赛要求的是**按实例**提交——同一张图里三辆车要分三行,每辆车一个掩膜。所以我们先用一个阈值（默认 0.5）把概率图转成二值图，再用 `skimage` 的连通域标记 `label()` 把连在一起的像素分成一个个独立物体：

```python
from skimage.morphology import label

def rle_encoding(x):
    """把一张二值图转成 RLE 编码（行主序）"""
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    """把概率图按连通域切分成独立掩膜，每个物体生成一段 RLE"""
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
```

然后把每个物体的 RLE 装进提交文件。行数不是固定的——图里有多少个物体就有多少行，所以 `ImageId` 要重复多次，`LabelId` 统一填车辆类，`Confidence` 和 `PixelCount` 分别填置信度和掩膜像素数：

```python
new_test_ids, rles = [], []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# 按比赛规定的列顺序组装 DataFrame 并保存
submission = pd.DataFrame()
submission['ImageId'] = new_test_ids
submission['LabelId'] = 33
submission['Confidence'] = 1
submission['PixelCount'] = len(rles)
submission['EncodedPixels'] = pd.Series(rles).apply(
    lambda x: ' '.join(str(y) for y in x))
submission.to_csv('submission.csv', index=False)
```

在留出集上观察 mean IoU 时，我们发现模型确实学到了「车长什么样」，但掩膜的边界还不够干净，作者也在原始讨论里注明「模型还需要更多训练和后处理」。这也符合预期——这是一个典型的需要大算力反复调优的比赛，而我们的这一版主要验证了整条**数据读取 → 二值分割 → 实例切分 → RLE 提交**的流水线是通的。

---

## 三、总结

本文中我们完成了 CVPR 2018 自动驾驶挑战赛的完整入门流程。我们首先用 EDA 摸清了数据：3.9 万张车载摄像头图像、掩膜用「类别×1000+实例」的方式把类别和实例编码在一个像素值里、测试集 1917 张、车辆类占据了绝大多数标注。其次我们做了数据清洗：把混合编码转成「是否车辆」的二值掩膜，按车辆占比筛选出前 1000 张车辆密集的图减小类别不平衡，并用 skimage 解决 16 位掩膜读取的问题。接着我们训练了一个带全连接分支的 U-Net，在 512×512 分辨率下做二值分割，用 mean IoU 监控训练。最后我们通过阈值二值化和连通域标记把预测概率切分成独立实例，再转成 RLE 编码并按规定格式提交。

除了这套方案，社区里还流行「U-Net + 预训练 ResNet50 编码器」、Pyramid Scene Parsing Network（PSPNet）以及纯 OpenCV 的 HOG 特征方案等做法，但都万变不离其宗：先在像素级别把物体分割出来，再想办法把实例拆开。这个比赛给我们最大的启发是：**掩膜是「类别+实例」合一的编码**，读懂数据格式往往比调模型更重要；另外对于极度不平衡的分割任务，聚焦最重要的那一小类、跑通整套流程，比一上来就追求全类别精度更现实。