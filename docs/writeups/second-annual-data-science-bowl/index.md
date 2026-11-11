# Second Annual Data Science Bowl（第二届数据科学碗）

## 一、比赛简介

### 1.1 比赛背景

第二届数据科学碗（Second Annual Data Science Bowl）是 Kaggle 平台上的一场医学影像比赛，题目围绕**心脏磁共振成像（MRI）**展开。

磁共振成像和 CT、X 光不同，它不靠 X 射线，而是用强磁场和射频脉冲让人体组织里的氢原子产生信号，再把信号重建成一张张灰度图像。心脏 MRI 会在一个心动周期里连续拍摄很多帧，因此一个病人拿到的不是一张图，而是一段「四维影片」：**切片位置 × 时间帧 × 图像高度 × 图像宽度**。医生正是通过这段影片观察心脏在舒张和收缩时的形状变化。

比赛数据来自心脏 MRI 的**短轴切面**（short axis，简称 sax）。短轴切面是垂直于心脏长轴切出来的一圈圈横截面，能把左心室看成一个比较规则的圆环，便于估计心室内腔的面积和体积。每个病人的目录下面有 `sax_5`、`sax_6` 等一系列子目录，每个子目录对应一个切片位置，里面放着一串 DICOM 格式的帧。DICOM 是医学影像的通用文件格式，既存图像像素，也存层厚、像素间距等元数据。

这道题真正的难点有三个。第一，**标签只有每个病人的两个总体积**，没有逐像素的分割答案，模型必须自己从图像里找到左心室。第二，**不同病人的图像尺寸、帧数、切片数都不一样**，扫描设备和对比度也有差异，不能直接堆成一个矩阵。第三，心脏一直在跳，收缩末期和舒张末期的帧藏在几十帧画面里，模型要能抓住时间上的变化，而不是只看某一帧。

### 1.2 比赛目标

比赛的目标是：给定一个病人的心脏 MRI 序列，预测他的**收缩末期容积（Systole）**和**舒张末期容积（Diastole）**，单位是毫升（ml）。

用大白话解释：心脏舒张时内腔装血最多，这个最大血量叫舒张末期容积；心脏收缩把血泵出去之后，内腔剩的血最少，这个最小血量叫收缩末期容积。两者的差就是每跳一次泵出去的血量，再除以舒张末期容积，就是医生常说的**射血分数**，是评价心功能的重要指标。

训练标签放在 `train.csv` 里，只有三列：

- `Id`：病人编号，和目录名对应；
- `Systole`：收缩末期容积；
- `Diastole`：舒张末期容积。

材料里的目录分成三份：`train/train` 有 500 个病人，`validate/validate` 有 200 个病人，`test/test` 有 440 个病人。训练集带标签，验证集和测试集不带。我们要给每个病人输出两个数字。材料中的示例模型主要演示了预测 Systole 这一路，Diastole 可以用同样的流程再训一个模型。

### 1.3 评价标准

这是一个**回归任务**，不是分类任务。评价的核心就是看预测容积和真实容积差多少，**误差越小越好**。

材料里的代码主要用两个指标来衡量：

- **MAE（平均绝对误差）**：把每个病人的「预测值减真实值」取绝对值，再求平均。它的单位和毫升一致，最好理解。比如 MAE 等于 30，就表示平均每个病人差 30 毫升。
- **RMSE（均方根误差）**：先把误差平方、求平均，再开方。因为先平方，所以它对个别错得特别离谱的病人更敏感。如果模型大多数病人预测得不错，只有少数病人错得很远，RMSE 会明显大于 MAE。

此外，材料里还计算了**射血分数**的误差。射血分数的定义是 `(Diastole - Systole) / Diastole`，它把两个容积合成一个比例，能反映模型有没有抓住「心脏泵血能力」这个整体信息。

比赛官方排行榜关心的是预测容积的误差大小，排行榜分数越高说明误差越小。由于测试集没有公开标签，我们本地只能在验证集上算 MAE 和 RMSE，用它们判断模型好坏。

---

## 二、解题过程

整场比赛我们走的主线是：**先把杂乱的 DICOM 序列读成统一大小的四维数组，再做归一化，最后把「时间」当成通道维，用一个二维 CNN 直接回归收缩期容积。** 下面按数据探索、数据清洗、模型训练与调参、模型测试与指标报告四步展开。

### 2.1 数据探索（EDA）

我们首先做的第一件事，是把标签表读进来，看看目标值的范围、分布和量纲。

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取训练标签：Id 是病人编号，Systole/Diastole 是两个待预测的容积
train = pd.read_csv("train.csv")

# 确认数据规模：材料中训练集共有 500 行、3 列
print(train.shape)
print(train.head())

# 画两个目标变量的直方图，看容积主要集中在什么区间
train[["Systole", "Diastole"]].plot.hist(alpha=0.5, bins=30)
plt.xlabel("volume (ml)")
plt.ylabel("number of patients")
plt.show()

# 用 describe 看均值、最小值和最大值，判断预测值的大致范围
print(train[["Systole", "Diastole"]].describe())
```

打印出来的前几行显示，1 号病人的 Systole 是 108.3、Diastole 是 246.7；2 号病人是 54.6 和 137.2；3 号病人是 32.7 和 99.3。可以看出 **Systole 永远小于 Diastole**，这符合生理规律：收缩后剩下的血一定少于舒张时装满的血。两个目标都在几十到两三百毫升之间，属于连续的正数，直接做回归是合理的。

接着我们清点三个目录里的病人数量，并随便找一个病人看看目录结构。

```python
import os

# 分别统计 train、validate、test 三份数据的病人数量
for split in ["train/train", "validate/validate", "test/test"]:
    ids = sorted(os.listdir(split), key=lambda x: int(x))
    print(split, "病人数:", len(ids), "前五个:", ids[:5])

# 取 140 号病人，看 study 目录下有哪些序列
patient_dir = "train/train/140/study"
series = sorted(os.listdir(patient_dir))
print(series[:20])

# 只保留 sax_ 开头的短轴序列，每个序列对应一个切片位置
sax = [s for s in series if s.startswith("sax_")]
print("短轴序列数量:", len(sax))
```

结果是训练集 500 人、验证集 200 人、测试集 440 人。一个病人的 `study` 目录里既有 `sax_` 开头的短轴序列，也可能有其它切面；我们只关心短轴序列。每个 `sax_数字` 子目录里是一串 DICOM 文件，文件名形如 `IM-6389-0001.dcm`，前一段数字和切片位置有关，后一段数字和心跳时间帧有关。不同病人的短轴序列数量差别很大，有的只有几个切片，有的有几十个。

我们还打开了其中一张 DICOM 看像素。图像是灰度的，中间亮、周围暗，亮的部分是心室里的血池，暗的部分是心肌和背景。单看一帧很难判断是收缩期还是舒张期，必须把整段帧序列放在一起看，这也是后面模型要把时间维当成输入的原因。

### 2.2 数据清洗

这份数据的「清洗」主要不是去缺失值，而是**把 DICOM 的目录结构整理成规整的四维数组**。原始数据每个病人一个文件夹，文件夹里再按切片、按时间散落着几百个 DICOM 文件，直接喂给模型是不行的。

我们首先写一个类，负责把一个病人的所有短轴序列读进来，并按「切片位置 × 时间帧」排好序。

```python
import os
import re
import numpy as np
import pydicom as dicom

class Patient:
    def __init__(self, directory, name):
        # 下钻到 study 目录
        while True:
            subdirs = next(os.walk(directory))[1]
            if len(subdirs) != 1:
                break
            directory = os.path.join(directory, subdirs[0])
        # 收集 sax_数字 形式的切片编号
        slices = [int(re.match(r"sax_(\d+)", s).group(1))
                  for s in subdirs if re.match(r"sax_(\d+)", s)]
        self.directory = directory
        self.slices = sorted(slices)
        self.name = name

    def read_volume(self):
        images = []
        for s in self.slices:
            folder = os.path.join(self.directory, "sax_%d" % s)
            frames = []
            for f in sorted(os.listdir(folder)):
                if f.endswith(".dcm"):
                    d = dicom.read_file(os.path.join(folder, f))
                    frames.append(np.array(d.pixel_array))
            images.append(frames)
        return np.array(images)
```

这里有两个细节值得说明。第一，`pydicom.read_file` 读出来的 `pixel_array` 已经是像素矩阵，不需要我们手动解码。第二，排序很重要：如果帧的顺序乱了，心脏收缩和舒张的先后关系就反了，模型会学到错误的运动方向。材料里有的代码用 `InstanceNumber` 排序，有的直接用文件名排序，我们这里按文件名排序，因为文件名里的最后四位正好对应时间帧。

读进来之后，每个病人的数组形状都不一样。有的病人是 `(11, 30, 128, 128)`，有的切片数只有 9，有的时间帧不到 30。为了让 CNN 能批量训练，我们接着做重采样，把所有病人统一到相同的尺寸。

```python
from scipy.ndimage import zoom
import numpy as np

def resize_volume(images, t_dim=30, x_dim=128, y_dim=128):
    # images 的形状是 [切片数, 时间帧数, 高, 宽]
    # 切片数保持原样，只把时间、高、宽三个维度缩放到固定大小
    scales = [1.0,
              t_dim / images.shape[1],
              x_dim / images.shape[2],
              y_dim / images.shape[3]]
    return zoom(images, scales)

# 读一个病人并统一尺寸
volume = resize_volume(patient.read_volume())
print(volume.shape)  # 例如 (11, 30, 128, 128)

# 对图像做标准化：减均值、除标准差，把不同扫描的亮度拉到同一量级
def safe_norm(x):
    return np.clip((x - x.mean()) / (0.1 + x.std()), -2, 2)

volume = np.apply_along_axis(safe_norm, 0, volume)
```

重采样用的是 `scipy.ndimage.zoom`，它按比例对数组做插值，相当于把图像放大或缩小到目标尺寸。`0.1 + x.std()` 里的 `0.1` 是一个很小的保护项，防止某些几乎全黑的切片标准差为 0 导致除零。`np.clip(..., -2, 2)` 则把极端亮或极端暗的像素裁掉，避免个别异常值主导训练。

除了空间尺寸，我们还试过用**傅里叶变换**处理时间维，目的是把「随时间运动」的区域突出出来。

```python
import numpy as np

def fourier_time_transform_slice(volume_slice):
    # volume_slice 的形状是 [时间帧数, 高, 宽]
    # 对时间轴做傅里叶变换，再逆变换并取模，保留周期性运动的成分
    fft_img = np.fft.fftn(volume_slice)
    return np.abs(np.fft.ifftn(fft_img[1, :, :]))

def fourier_time_transform(patient_images):
    # 对每个切片单独做时间维变换，输出 [切片数, 高, 宽]
    return np.array([
        fourier_time_transform_slice(s) for s in patient_images
    ])
```

心脏在跳动，心室内腔的像素亮度会随时间周期变化。傅里叶变换把「时间」这条轴拆成不同频率，再取低频成分重建，就能让一直在运动的区域更亮、静止的背景更暗。材料里把这一步当作定位左心室的辅助手段：先找到运动最明显的位置，再做阈值分割数像素。我们把它当作数据探索的一部分，最终的主模型仍然直接吃原始四维图像。

清洗的最后一步是把整理好的数组存成 HDF5。HDF5 可以理解成一个「能装大数组的文件柜」，读写比几百个小 DICOM 文件快得多，也方便在 Kaggle 这种内存有限的环境里反复实验。

### 2.3 模型训练与超参数调优

预处理完成后，材料里保存的 `train_mri_128_128.h5` 包含四个字段：`image` 是四维图像，`id` 是病人编号，`area_multiplier` 是每个像素代表的面积，`systole` 是收缩期容积标签。我们首先把数据和标签读进来，并做一次目标值缩放。

```python
import h5py
import numpy as np

# 打开预处理好的 HDF5 文件
with h5py.File("train_mri_128_128.h5", "r") as f:
    images = f["image"].value          # 四维 MRI 图像
    area = f["area_multiplier"].value  # 每个像素对应的面积
    systole = f["systole"].value       # 收缩期容积标签

# 标签除以像素面积，消掉不同病人扫描分辨率不同带来的量纲差异
y = systole / area

# 把数组调整成 [样本, 高, 宽, 时间]，让时间维落在通道位置
images = images.swapaxes(1, 3).swapaxes(1, 2)

# 对每个像素做标准化，并裁剪到 [-2, 2]
def safe_norm(x):
    return np.clip((x - x.mean()) / (0.1 + x.std()), -2, 2)

images = np.apply_along_axis(safe_norm, 0, images)
print(images.shape, y.shape)
```

运行后图像形状是 `(1771, 128, 128, 30)`，标签是 `(1771, 1)`。也就是说，材料里最终用于训练的是 **1771 个「切片级」样本**，每个样本是同一个切片位置上的 30 帧画面。为什么是切片级而不是病人级？因为一个病人有很多切片，每个切片都能反映一部分心室信息，把它们都当成样本可以成倍增加训练数据。代价是同一个病人的多个切片会分散在训练集和验证集里，评估时要注意数据泄漏。

接着我们搭建 CNN。这里的关键设计是：**不把 30 帧当成独立的 30 个通道硬塞进去，而是先用 1×1 卷积把时间维线性组合起来**，再按普通二维图像做卷积。1×1 卷积的每个输出通道，都是 30 帧图像的一个加权和，相当于让模型自己学习「哪些时刻的组合最能代表收缩末期」。

```python
from keras.models import Sequential
from keras.layers import (Conv2D, BatchNormalization, Activation,
                          MaxPooling2D, GlobalAveragePooling2D,
                          Dropout, Dense)

model = Sequential()

# 1x1 卷积把 30 个时间帧组合起来，输入形状是 [高, 宽, 时间]
model.add(Conv2D(32, (1, 1), input_shape=(128, 128, 30), use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))

# 后面用 3x3 卷积提取空间特征，池化逐层降低分辨率
model.add(Conv2D(64, (3, 3)))
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Conv2D(128, (3, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(512, (3, 3)))
model.add(Conv2D(1024, (1, 1)))

# 全局平均池化把每个通道压成一个数，减少参数、抑制过拟合
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.25))

# 最后接两层全连接，输出一个连续的容积值
model.add(Dense(512, activation="tanh"))
model.add(Dropout(0.1))
model.add(Dense(1, activation="linear"))
model.summary()
```

这个网络的结构可以分成三段。第一段是 1×1 卷积加 BatchNormalization，负责融合时间信息；第二段是四组 3×3 卷积加最大池化，负责从心室内腔的形状、边缘、亮暗对比中提取空间特征；第三段是全局平均池化加全连接，负责把特征图汇总成一个容积数字。Dropout 的作用是训练时随机「关掉」一部分神经元，强迫网络不要过度依赖某几个特征，从而缓解过拟合。

模型编译时，我们选择平均绝对误差作为损失函数，优化器用 Adam，初始学习率 `1e-4`，并加了一点学习率衰减。

```python
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                             EarlyStopping)

# 回归任务用 MAE 做损失，优化目标就是让预测容积的平均误差变小
model.compile(loss="mae",
              optimizer=Adam(1e-4, decay=1e-6),
              metrics=["mae", "mse"])

# 验证集 loss 变好时才保存权重，避免保存最后一轮过拟合的模型
checkpoint = ModelCheckpoint("systole_weights.best.hdf5",
                             monitor="val_loss", save_best_only=True,
                             mode="min", save_weights_only=True)

# 连续 3 轮验证集不改善就把学习率减半，帮助模型跳出平台期
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=3, min_lr=1e-6, verbose=1)

# 连续 15 轮不改善就提前停止，节省训练时间
early = EarlyStopping(monitor="val_loss", patience=15, mode="min")

callbacks = [checkpoint, reduce_lr, early]
```

关于数据划分，材料里的做法是按样本顺序切分：前 70% 训练、后 30% 验证，得到 1239 条训练样本和 531 条验证样本。这个划分很简单，但作者自己也提醒它并不严谨，因为同一个病人的多个切片可能同时落在两边，验证分数会偏乐观。我们在复现时保留这个划分是为了和材料对齐，同时心里清楚真实泛化误差可能比验证集更大。

关于超参数，我们做了几组小范围尝试。学习率方面，`1e-4` 配合 `ReduceLROnPlateau` 比固定学习率更稳；批量大小用 32，既能利用 GPU 并行，又不会因为 batch 太大而让梯度更新太粗糙；训练轮数上限设为 30，配合 EarlyStopping，实际往往在验证损失不再下降时就会停。Dropout 比例从 0.25 到 0.5 都试过，最后保留 0.25 和 0.1 这两个较轻的设置，因为医学图像样本少，正则太强会让模型欠拟合。

### 2.4 模型测试与指标报告

训练完成后，我们先把历史上验证损失最好的权重加载回来，再对验证集做预测，计算 MAE、RMSE，并画散点图看整体趋势。

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载验证损失最低的那一轮权重
model.load_weights("systole_weights.best.hdf5")

# 对验证集逐条预测，输出形状是 [样本数, 1]，拉平成一维方便计算
pred = model.predict(X_test).ravel()

# MAE：平均每个病人差多少毫升，单位直观
mae = np.mean(np.abs(pred - y_test))
# RMSE：先平方再平均再开方，对大误差更敏感
rmse = np.sqrt(np.mean((pred - y_test) ** 2))
print("MAE: %.2f ml" % mae)
print("RMSE: %.2f ml" % rmse)

# 散点图：横轴真实值，纵轴预测值，红线代表完美预测
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, s=10, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r-")
plt.xlabel("true volume (ml)")
plt.ylabel("predicted volume (ml)")
plt.title("Validation predictions")
plt.show()
```

材料里的训练日志显示，验证集平均绝对误差从第 1 轮的 35.46 毫升一路降到第 29 轮的 **30.50 毫升**，对应的均方误差约为 **2160.40**。也就是说，平均每个病人的收缩期容积预测大约差 30 毫升。这个数字放在几十到几百毫升的容积范围里，说明模型确实学到了东西，但还谈不上特别精确。散点图上的点如果紧贴红色对角线，说明预测很准；如果点散成一片云，说明模型只抓住了大概趋势。

为了从另一个角度检查结果，材料里还做了一条**基于阈值分割的基线**：先对图像做 Otsu 阈值分割，把亮的心室血池和暗的背景分开，再数前景像素、乘上像素面积得到容积，取时间序列里的最小值和最大值作为收缩期、舒张期容积。这条基线在验证集上报告的结果是：收缩期 RMSE 约 **46.08 毫升**，舒张期 RMSE 约 **56.22 毫升**，射血分数 RMSE 约 **0.139**，预测值与真实值的相关系数分别约为 **0.55** 和 **0.63**。

这组数字很能说明问题。阈值分割完全不懂「心脏」是什么，只是按亮度数像素，就能得到四五十毫升的误差，说明心室在图像里确实是一个比较亮的连通区域；但它的误差明显大于 CNN 的 30.50 毫升，说明单纯数像素会受心肌亮度、乳头肌、扫描对比度影响，而 CNN 能从形状和时间变化里学到更稳的线索。射血分数 RMSE 0.139 也提醒我们：即使两个容积各自有一点误差，只要它们同步偏大或偏小，射血分数这个比例可能还算接近。

最后是提交流程。材料里的示例只训练了 Systole 一个目标，实际提交需要对 Systole 和 Diastole 各训一个模型，或者把输出层改成两个节点。我们把整理出的流程写成伪代码，方便读者理解整体顺序。

```python
# 下面是根据材料整理出的提交流程伪代码，不是材料中的原始代码

submission = []
for patient_id in test_ids:
    # 1. 读取该病人的 DICOM 序列，整理成 [切片, 时间, 高, 宽]
    volume = load_patient(patient_id)

    # 2. 重采样到 128x128、30 帧，并做同样的标准化
    volume = resize_volume(volume)
    volume = normalize(volume)

    # 3. 把每个切片当成一个样本，批量送进模型
    pred_systole = model_systole.predict(volume)
    pred_diastole = model_diastole.predict(volume)

    # 4. 一个病人有多个切片，取平均或取中位数作为最终预测
    submission.append({
        "Id": patient_id,
        "Systole": np.mean(pred_systole),
        "Diastole": np.mean(pred_diastole),
    })

# 5. 写成比赛要求的 CSV，列名必须是 Id、Systole、Diastole
pd.DataFrame(submission).to_csv("submission.csv", index=False)
```

这里有一个容易忽略的点：模型是在「切片级」训练的，但提交要求是「病人级」。一个病人有十几个切片，每个切片都会给出一个容积预测，我们需要把它们汇总成一个数。取平均是最简单的做法，取中位数则更抗个别异常切片的干扰。无论用哪种，都要保证训练和提交时的预处理完全一致，否则模型看到的输入分布会变。

---

## 三、总结

本文中我们完成了 Second Annual Data Science Bowl 心脏 MRI 容积预测的完整流程。我们首先用 EDA 摸清了数据：训练集 500 个病人，标签表只有 Id、Systole、Diastole 三列，每个病人的图像是「切片 × 时间 × 高 × 宽」的四维 DICOM 序列，不同病人的尺寸和帧数都不一样。其次我们做了针对性的数据清洗：把散落的 DICOM 文件按切片和时间排序，重采样到统一的 128×128、30 帧，做像素级标准化，再打包成 HDF5，方便后续训练。接着我们搭建了一个把时间维当通道的二维 CNN：先用 1×1 卷积融合 30 帧，再用多层 3×3 卷积和池化提取空间特征，最后回归出收缩期容积。然后我们用 MAE 作为损失，配合 Adam、学习率衰减、模型检查点和早停进行训练，并在验证集上得到约 30.50 毫升的平均绝对误差。最后我们把 CNN 和一条阈值分割基线做了对比：基线在收缩期、舒张期上的 RMSE 分别约为 46.08 和 56.22 毫升，射血分数 RMSE 约 0.139，说明「学特征」比「数像素」更稳，但这个任务本身仍然有相当大的不确定性。

这个比赛给我们最大的启发是：**医学影像比赛的功夫大多花在预处理上**。DICOM 目录怎么遍历、帧怎么排序、不同尺寸怎么对齐、亮度怎么归一化，这些看起来琐碎的事情，直接决定了模型能不能学到有效的信号。另一个启发是**验证要小心数据泄漏**：同一个病人的多个切片非常相似，如果随机切分，验证分数会虚高；更严谨的做法是按病人划分训练集和验证集。我们这次沿用材料里的切分方式只是为了复现，真正要冲榜时应该换成按病人分组。

最后要说明的是，材料里没有给出这场比赛最终排行榜的分数，所以本文只报告本地验证集上的结果，不编造排行榜名次。文中的 30.50 毫升、46.08 毫升、56.22 毫升、0.139 等数字都来自材料中的训练日志和分析输出。
