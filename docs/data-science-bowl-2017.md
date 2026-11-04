# Data Science Bowl 2017：肺癌检测（Lung Cancer Detection）

## 一、比赛简介

### 1.1 比赛背景

肺癌是全球死亡率最高的癌症之一。医生判断一个人有没有肺癌，最常用的手段是做**CT（计算机断层扫描）**：机器绕着人体旋转拍摄，得到一层一层的横截面图像，一层叫一个「切片」（slice），把它们叠起来就是一个三维的人体「切片卷」。

Data Science Bowl 2017 由 Kaggle 与多家医疗机构合作举办，提供的正是这样一批胸部 CT 扫描。比赛的核心问题是：**仅凭这次扫描的影像，能不能预测这位患者在接下来一年内会被诊断出肺癌？** 如果能提前一年发现，治疗的成功率会高很多，这就是这个比赛的现实意义。

数据格式是医学影像领域通用的 **DICOM**（`.dcm`）。每个患者一个文件夹，里面是该患者一次扫描的所有切片。

### 1.2 比赛目标

- **训练集**：`stage1_labels.csv`，两列 `id`（患者编号）和 `cancer`（0 或 1，表示一年内是否确诊肺癌），共 1397 位患者。
- **测试集**：只给患者编号，不给标签，我们要为每位患者输出一个 0～1 之间的概率。
- **图像**：每个患者一个目录，目录名就是患者编号，里面全是 `.dcm` 切片文件。

所以这是一个**二分类 + 概率输出**的问题：输入是一整卷三维 CT，输出是一个患癌概率。

### 1.3 评价标准

比赛使用 **LogLoss（对数损失，也叫二元交叉熵）** 作为评价指标：

```
LogLoss = -(1/N) × Σ [ y·log(p) + (1-y)·log(1-p) ]
```

其中 `y` 是真实标签（0 或 1），`p` 是我们预测的概率，`N` 是患者数。这个指标的意思是：**你给正确答案的概率越高，惩罚越小；给得越离谱，惩罚越大**。所以 LogLoss 是**越低越好**，而且它对「过度自信的错误」惩罚特别重——预测 0.99 但真实是 0，损失会非常大。

正因为 LogLoss 只看概率、不看对错，材料里有一个非常重要的观察：假设我们不做任何模型，把所有人都预测成训练集的平均患癌率 25.91%，训练集上的 LogLoss 是 **0.5721**，提交到排行榜得到 **0.60235**。这是我们必须超过的「傻瓜基线」。

---

## 二、解题过程

这类比赛最大的工作量其实不在模型，而在**把原始 CT 变成模型能吃的张量**。我们按四步来：先探索数据、再做一长串医学影像预处理、接着训练一个三维卷积网络、最后生成概率提交。

### 2.1 数据探索（EDA）

我们首先看输入目录和数据规模：

```python
import os, glob
import numpy as np
import pandas as pd

# 输入目录里有：样例图像、训练标签、提交样例
os.listdir('../input')
# ['sample_images', 'stage1_labels.csv', 'stage1_sample_submission.csv']

# 看看每个患者有多少张切片
for d in os.listdir('../input/sample_images')[:3]:
    print(d, len(os.listdir('../input/sample_images/' + d)))

# 整体统计：患者数 和 dcm 文件总数
print(len(os.listdir('../input/sample_images')))
print(len(glob.glob('../input/sample_images/*/*.dcm')))
```

样例数据只有 20 位患者、3604 个 dcm 文件；正式训练集约 1400 位患者。每位患者的切片数量差别很大，材料里看到最少 110 张、最多 435 张。每张切片默认是 512×512 像素，单个 dcm 文件大小几乎是固定的 0.525 MB，说明文件没有压缩、信息量相当。

接着看标签。这一步非常关键，因为我们要先知道「瞎猜」能拿多少分：

```python
df_train = pd.read_csv('../input/stage1_labels.csv')
print('训练患者数:', len(df_train))          # 1397
print('患癌比例: {:.2f}%'.format(df_train.cancer.mean() * 100))  # 25.91%

# 用训练集平均患癌率作为所有人的预测概率，算一下 LogLoss
from sklearn.metrics import log_loss
naive = np.full(len(df_train), df_train.cancer.mean())
print('训练集 LogLoss:', log_loss(df_train.cancer, naive))   # 0.5721
```

患癌比例只有约四分之一，属于**类别不平衡**。这也解释了为什么「全猜 0.26」这种毫无技术含量的提交也能拿到 0.60 的分数——LogLoss 对这类「温和的概率」并不苛刻，所以真正的模型必须明显优于 0.60 才算有效。

我们还检查了一个常见陷阱：**患者编号和标签有没有关联**。如果编号是随机分配的，那么按编号排序后标签应该没有规律；我们画了滑动平均曲线来验证，并没有发现可利用的泄漏。这提醒我们不要浪费时间去找「排序规律」这类捷径。

### 2.2 数据清洗

这一步是整个比赛的胜负手。原始 DICOM 有三个「不干净」的地方必须处理：

**问题一：切片顺序是乱的。** 文件夹里的 dcm 文件不一定按人体从上到下的顺序排列，必须按切片在三维空间中的位置排序：

```python
import pydicom
import numpy as np

def load_scan(path):
    """读入一个患者目录下的所有切片，并按 Z 轴位置排好序"""
    slices = [pydicom.dcmread(path + '/' + s) for s in os.listdir(path)]
    # ImagePositionPatient[2] 是切片在 Z 方向上的物理坐标
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    # 用相邻两片的间距补上「层厚」这个缺失的元数据
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
                                 slices[1].ImagePositionPatient[2])
    except Exception:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices
```

**问题二：像素值不是人看的单位。** CT 的原始像素值只是机器读数，需要按文件里存的斜率和截距换算成 **HU（Hounsfield Unit，亨氏单位）**，这是衡量组织密度的物理单位。此外，圆柱形扫描范围之外的区域被填成了 -2000，要把它改成代表空气的 0：

```python
def get_pixels_hu(slices):
    """把原始像素值换算成 HU 单位"""
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    image[image == -2000] = 0        # 扫描边界外的填充值改成空气

    for i in range(len(slices)):
        intercept = slices[i].RescaleIntercept   # 截距，一般 -1024
        slope = slices[i].RescaleSlope           # 斜率，一般 1
        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.int16)
        image[i] += np.int16(intercept)
    return np.array(image, dtype=np.int16)
```

**问题三：每台机器的「像素物理尺寸」不一样。** 同一台 CT 扫出来的体素，可能是 0.5mm×0.5mm×2.5mm，也可能 0.7mm×0.7mm×1.5mm。如果直接喂给网络，模型会误以为不同患者的器官大小不一样。解决办法是**重采样到各向同性（isotropic）的 1mm×1mm×1mm**：

```python
import scipy.ndimage

def resample(image, scan, new_spacing=[1, 1, 1]):
    """把整卷影像重采样到 1mm 各向同性分辨率"""
    # 当前体素在三个方向上的物理尺寸（Z 方向用层厚，XY 用 PixelSpacing）
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing),
                       dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_shape = np.round(image.shape * resize_factor)
    # 因为取整，实际缩放比例要再算一次，避免尺寸对不上
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor,
                                             mode='nearest')
    return image, new_spacing
```

**问题四：整卷图像绝大部分是无关的身体组织。** 我们只关心肺，所以用一套基于 HU 阈值和连通域的方法把肺抠出来。思路很朴素：肺里是空气，HU 很低（小于 -320），先把低密度区域二值化；然后找出「人体外部的空气」并填掉；再对每一层取最大的连通区域，把肺内部被血管、结节占据的实心结构补齐；最后只保留最大的一块气腔，去掉体内其他零散的气泡：

```python
from skimage import measure, morphology

def largest_label_volume(im, bg=-1):
    """返回出现次数最多的那个连通域编号"""
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    return vals[np.argmax(counts)] if len(counts) > 0 else None

def segment_lung_mask(image, fill_lung_structures=True):
    # 小于 -320 HU 的算「空气」，+1 是为了避开 0 这个背景值
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # 左上角那个点几乎一定在人体外面，用它代表「外部空气」
    background_label = labels[0, 0, 0]
    binary_image[background_label == labels] = 2   # 把外部空气标记为 2

    if fill_lung_structures:
        # 每一层取最大的实心区域，把肺内的血管/结节一起包含进来
        for i, axial_slice in enumerate(binary_image):
            labeling = measure.label(axial_slice - 1)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None:
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1              # 变回真正的 0/1
    binary_image = 1 - binary_image  # 翻转：肺变成 1
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:          # 只留最大一块气腔
        binary_image[labels != l_max] = 0
    return binary_image
```

得到肺掩膜后，材料里特别提醒要**做一点膨胀**：因为有些结节贴在肺壁上，严格按肺的内部抠会把它切掉。

**最后是归一化。** HU 范围大约 -1024 到 2000，但 400 以上都是骨头，对判断肺结节没有帮助，所以我们把范围裁到 [-1000, 400] 再线性映射到 [0, 1]，最后减去全数据集的平均值 0.25 做**零中心化**：

```python
MIN_BOUND, MAX_BOUND = -1000.0, 400.0
PIXEL_MEAN = 0.25

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.0
    image[image < 0] = 0.0
    return image

def zero_center(image):
    return image - PIXEL_MEAN
```

这里有个容易踩的坑：**不要按单张图做零中心化**。普通照片的亮度对比度各不相同，但 CT 的 HU 是经过机器标定的绝对物理量，按图减去各自的均值反而把有用的密度信息破坏了。所以要用全数据集的统一均值。

**统一尺寸。** 预处理完的体数据每卷的层数还不一样（110～435 层），而网络需要固定输入。我们的办法是：把每张切片缩放到 50×50，再把整卷按顺序切成 20 段，每段内的切片取平均，得到固定 20 层的体数据。平均相当于稍微「加厚」了每一层，但结节这种结构仍然会保留下来：

```python
import cv2

IMG_PX_SIZE = 50
HM_SLICES = 20

def chunks(l, n):
    """把列表尽量平均地切成 n 段"""
    k, m = divmod(len(l), n)
    return [l[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def prepare_patient(path):
    slices = load_scan(path)
    volume = get_pixels_hu(slices)               # 转 HU
    volume, _ = resample(volume, slices)         # 重采样到 1mm
    # 每层缩放到 50x50，再按 20 段求平均，得到 50x50x20
    slices_50 = [cv2.resize(np.array(s), (IMG_PX_SIZE, IMG_PX_SIZE)) for s in volume]
    volume_20 = np.array([np.mean(c, axis=0) for c in chunks(slices_50, HM_SLICES)])
    return normalize(volume_20)
```

把这些处理好的 50×50×20 体数据一次性存成 `.npy` 文件，之后训练就直接加载，省去每次重复解码 DICOM 的时间。

### 2.3 模型训练与超参数调优

模型上，我们选择了一个简单的 **3D 卷积神经网络（3D CNN）**。普通 CNN 在二维平面上滑动卷积核，3D CNN 的卷积核在长、宽、深三个方向同时滑动，正好对应 CT 的三维结构。网络结构是两层卷积加两层池化，再接一个全连接层输出二分类：

```python
import tensorflow as tf

IMG_SIZE_PX, SLICE_COUNT, n_classes = 50, 20, 2

def conv3d(x, W):
    # 三维卷积，步长全为 1，padding 用 SAME 保持尺寸不变
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def maxpool3d(x):
    # 三个方向都做 2 倍下采样
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='SAME')

def convolutional_neural_network(x):
    # 第一层：3x3x3 卷积核，1 通道输入，输出 32 个特征图
    weights = {'W_conv1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
               # 第二层：32 通道输入，输出 64 个特征图
               'W_conv2': tf.Variable(tf.random_normal([3, 3, 3, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([54080, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}
    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    # 把输入整理成 [batch, 50, 50, 20, 1]
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])
    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)
    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    # 展平后接全连接层，dropout 防止过拟合
    fc = tf.reshape(conv2, [-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate=0.8)
    return tf.matmul(fc, weights['out']) + biases['out']
```

训练时用标准的 softmax 交叉熵和 Adam 优化器，学习率 1e-3，batch size 10，训练 10 轮。我们按患者编号简单切分，最后 100 位患者留作验证集：

```python
much_data = np.load('muchdata-50-50-20.npy')  # 预处理好的 (体数据, 标签) 列表
train_data, validation_data = much_data[:-100], much_data[-100:]

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    # 二分类的 softmax 交叉熵，等价于 LogLoss 的损失形式
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10):
            for data in train_data:
                X, Y = data[0], data[1]
                sess.run([optimizer, cost], feed_dict={x: X, y: Y})
            # 每轮结束后在留出集上算一次准确率
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Epoch', epoch + 1,
                  'val acc:', accuracy.eval({x: [i[0] for i in validation_data],
                                             y: [i[1] for i in validation_data]}))
```

超参数方面，我们主要调了三个：**图像分辨率**（512 → 150 → 50，分辨率越低越省显存但细节越少）、**层数**（20 层，太少看不到肺的完整形状，太多显存吃不消）、以及**归一化的上下界**（最终固定在 -1000～400）。这个网络直白到有点「土」，但它是第一次把三维 CT 端到端喂给卷积网络的尝试，跑通本身就很有价值。

### 2.4 模型测试与指标报告

训练完成后，我们对测试集的每位患者做同样的预处理（读 DICOM → 转 HU → 重采样 → 缩放到 50×50×20 → 归一化），再用模型输出「患癌」那一类的概率：

```python
test_ids = pd.read_csv('../input/stage1_sample_submission.csv')['id'].values
preds = []

for pid in test_ids:
    volume = prepare_patient('../input/stage1/' + pid)   # 与训练一致的预处理
    volume = np.expand_dims(volume, axis=0)              # 加 batch 维
    prob = sess.run(tf.nn.softmax(prediction), feed_dict={x: volume})
    preds.append(prob[0, 1])                              # 取类别 1 的概率

submission = pd.DataFrame({'id': test_ids, 'cancer': preds})
submission.to_csv('submission.csv', index=False)
```

在验证集上，模型的表现会明显受到**样本量太小**的限制：每个患者只有一个 0/1 标签，而一次扫描有几百层图像，网络很容易记住训练患者而不是学到通用的病灶特征。材料的作者也坦率地提到，这一版主要还是「第一次跑通」，需要更多的数据和训练。从排行榜的角度看，这一类简单方案的分数往往停留在 0.6 左右，跟「全猜平均患癌率」的 0.60235 差距不大——这个结果虽然不好看，但它告诉我们一个关键结论：**这个比赛里真正拉开差距的是预处理和特征设计，而不是把网络堆深。**

另外值得记录的是，比赛期间还出现过一条「旁门左道」：因为排行榜会返回 LogLoss 分数，而 LogLoss 的数值理论上携带了测试标签的信息，于是有人设计了一组精心挑选的概率值，用十几次提交的分数反推出公开测试集的患者标签（材料里的 "Perfect Score" 脚本，198 个测试标签、14 次提交）。这属于利用评测机制的信息泄漏，不是真正的建模，我们只把它当作「评价指标设计」的反面教材来看：**如果指标反馈的信息量太大，比赛就可能被刷分而不是靠模型取胜。**

---

## 三、总结

本文中我们完成了 Data Science Bowl 2017 肺癌检测比赛的完整流程。我们首先用 EDA 摸清了数据：1397 位训练患者、患癌率 25.91%、每人一次 CT 扫描包含约 100～400 张 512×512 的 DICOM 切片，并算出「全猜平均概率」的 LogLoss 基线是 0.5721（排行榜 0.60235）。其次我们做了这个比赛最重的部分——医学影像清洗：按空间位置排序切片、把像素值换算成 HU、重采样到 1mm 各向同性分辨率、用 HU 阈值加连通域分析和形态学操作抠出肺部、把 HU 裁到 [-1000, 400] 并零中心化，最后把不定长的三维体积压成固定的 50×50×20。接着我们搭了一个两层 3D 卷积加全连接的 TensorFlow 网络，用 Adam、学习率 1e-3、batch 10 训练 10 轮，并按患者留出 100 人做验证。最后我们对测试患者逐个预处理并输出患癌概率，生成 `id,cancer` 格式的提交文件。

这个比赛给我们最大的启发有两点。第一，**医学影像比赛里 80% 的功夫在预处理**：切片顺序、HU 换算、体素间距、肺部分割，任何一步做错，后面的模型再好也没用。第二，**小样本 + LogLoss 是一个很诚实的组合**：它不会让你靠调参侥幸刷分，而是逼你把数据本身理解透。虽然这一版 3D CNN 分数只跟基线打平，但整条预处理流水线是后续所有改进（更大的网络、LUNA16 外部数据、结节检测候选框等）的基础。