# RSNA 肺炎检测挑战赛（Pneumonia Detection Challenge）

## 一、比赛简介

### 1.1 比赛背景

肺炎是一种常见的肺部感染，医学上通常用**胸部 X 光片（Chest X-ray）**来筛查和诊断。放射科医生每天要看成千上万张片子，如果能让机器先帮你「圈出」可疑的病灶，医生就能把精力集中在真正需要判断的地方。这正是 RSNA（北美放射学会）在 Kaggle 上举办肺炎检测挑战赛的初衷。

本次比赛的数据来自真实的 X 光影像，以 **DICOM** 格式存储。DICOM 是医学影像的标准格式，一个 `.dcm` 文件里面除了图像本身的像素值，还带了一长串元数据头（比如拍摄体位、像素间距等）。正因为是医学图像，比赛还专门做了隐私处理，病人的姓名等敏感信息被脱敏。

任务的核心难点有两个：

1. **图像中包含的病灶叫「肺不透明度（Lung Opacity）」**，指肺部区域出现的一片发白（不透光）的区域。它可能意味着肺炎，但其他一些疾病也会表现为不透明，医生标注时也不总是意见一致，所以数据本身带有一定的「噪声」。
2. **一份片子可能同时有好几个病灶。** 同一个病人的同一张 X 光片上，可能画着两个甚至更多个边界框（bounding box），我们后面会看到这样的例子。

比赛分为两个阶段：第一阶段（Stage 1）先让大家预测「这张片子是否患肺炎」，第二阶段（Stage 2）再让大家把病灶的具体位置框出来。

### 1.2 比赛目标

**输入**：一张胸片的 DICOM 文件（像素是 1024×1024 的 8 位灰度图），用文件名里的 `patientId` 与标签表关联。

**Stage 1 输出**：每个病人对应一个二分类标签 `Target`，1 表示「有肺炎」，0 表示「没有」。

**Stage 2 输出**：每个病人一行提交，列是 `patientId, PredictionString`。`PredictionString` 里按顺序写出「类别、置信度、x、y、width、height」，一个病人可以预测多个框，用空格分隔。例如：

```
1 0.95 264 152 213 379 1 0.42 562 152 256 453
```

**训练标签文件 `stage_2_train_labels.csv`** 内容如下（Stage 1 的结构完全相同）：

- `patientId`：病人编号，一一对应一张图；
- `x, y`：边界框左上角的坐标；
- `width, height`：边界框的宽和高；
- `Target`：0 或 1。`Target=0` 说明这张片子没有肺炎，此时 `x, y, width, height` 都是空值（NaN）；`Target=1` 时这些坐标有值。

另外一个辅助文件 `stage_2_detailed_class_info.csv` 把负样本进一步拆分成两类：

- `Normal`：完全正常；
- `No Lung Opacity / Not Normal`：虽然不算肺炎，但本身也不正常（有其他病灶）；
- `Lung Opacity`：有肺炎（对应 `Target=1`）。

**评价标准**

### 1.3 评价标准

Stage 1 用的是普通的**分类准确率**，猜对的比例越高越好。

Stage 2 才是比赛的重点，用的是**不同 IoU 阈值下的平均精度均值（mean Average Precision, mAP）**。这里需要拆开讲：

- **IoU（交并比）**：我们预测一个框，医生标注一个框，两者重叠面积占「并集面积」的比例就是 IoU。两个框完全重叠时 IoU=1，完全不沾边时 IoU=0。预测框只有和某个真实框的 IoU 超过阈值，才算是「命中」。
- **精度（Precision）**：在所有预测出的框里，真正命中的比例。
- **平均精度（AP）**：按置信度从高到低逐步放出预测框，画出「精度-召回率」曲线，取曲线下面积。
- **最终得分**：在 IoU＝0.4、0.45、0.5、……、0.75 这一组阈值上各算一次 mAP 再做平均。这种「多阈值」的评分方法比只卡一个 IoU 更严格也更全面：框得不够准，即使边界框里的内容对了也会掉分。

为了能在本地验证模型，我们需要自己在脚本里实现一个 `map_iou` 函数，模拟比赛的评分方式，这一部分我们在 2.4 节会写到。

---

## 二、解题过程

整个比赛我们采用的是「**先分类、再定位**」的思路：先用一个好判别的模型把「有没有肺炎」分出来，再在阳性图片上找框、做后处理。下面按 EDA、数据清洗、模型训练、测试报告四步走。

### 2.1 数据探索（EDA）

首先我们把两个 CSV 读进来，看看行数、列数和空值情况：

```python
import pandas as pd
import numpy as np

# 主标签表：病人 + 边界框 + 二分类目标
df_box = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
# 辅助表：把负样本进一步细分成 Normal / Not Normal
df_aux = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')

print('主表行数（框数）：', df_box.shape[0])
print('唯一病人数：', df_box['patientId'].nunique())
# 看前几行，确认列名和格式
print(df_box.head())
```

主表里一个病人可能有**多行**——每一行代表一个边界框，因此「框数」大于「病人数」。

接着数一下正负样本的比例，并查看边界框坐标的空值情况：

```python
# 计算 Target 的分布：0 是负样本，1 是正样本
print(df_box.groupby('Target').size() / df_box.shape[0])

# 检查边界框四列的缺失值数量
print(df_box[['x', 'y', 'width', 'height']].isnull().sum())
```

结果很清楚：`Target=0` 的样本约占 70%，负样本的边界框坐标**全部是 NaN**，正样本（约占 30%）的坐标则都有值。这说明标签表的设计是「没有肺炎就不填框」，空值是有意义的，不能随便删掉或填 0。

我们再看看辅助表里的三类划分：

```python
# 三类各自的占比
print(df_aux.groupby('class').size() / df_aux.shape[0])
```

输出约等于：`Lung Opacity` 31%、`No Lung Opacity / Not Normal` 40%、`Normal` 29%。也就是说负样本内部并不是铁板一块，其中 40% 的片子其实也有别的异常。这提醒我们：即使二分类模型把样本判成「无肺炎」，这些样本在临床上也不是完全正常，好在比赛评分只看 `Target`。

最后检查一个关键点：**每个病人是否都能找到对应的图像文件**，同时看一眼 DICOM 的元数据：

```python
import os
from glob import glob

# 图像文件以 .dcm 后缀存，文件名就是 patientId
files = os.listdir('../input/rsna-pneumonia-detection-challenge/stage_2_train_images')
# 断言：标签里的病人和图像文件一一对应
assert sorted(df_box['patientId'].unique()) == sorted([f[:-4] for f in files])

# 用 pydicom 读取一张片子，看看元数据
import pydicom
d = pydicom.read_file('../input/rsna-pneumonia-detection-challenge/stage_2_train_images/' + files[0])
print(d.Rows, d.Columns)   # 例如 1024 1024
print(d.Modality, d.ViewPosition)  # 例如 CR / PA
```

得到的图像是 1024×1024 的单通道灰度图，拍摄体位信息（PA 后前位 / AP 前后位）被保留在 `ViewPosition` 里。这些元数据看似无关，但后面有队伍靠它分析出了「拍摄体位泄漏（ViewPosition 泄漏）」这样的小 tricks，说明 EDA 挖得越细，能利用的信息越多。

### 2.2 数据清洗

数据集本身比较干净：病人 ID 与图像文件能一一对上，`Target` 没有缺失，唯一「缺」的是负样本的边界框坐标。因此我们的清洗策略是：

1. **不删除空值行**。负样本的 NaN 是故意的，删掉它们就破坏了标签结构。做二分类时我们直接用 `Target` 当标签即可。
2. **合并辅助表**。辅助表和主表的行顺序完全一致（`patientId` 逐行相同），所以直接按列拼接，把 `class` 字段并进来备用：

```python
# 两个数据表的 patientId 顺序一致，可以按列直接拼
assert df_box['patientId'].tolist() == df_aux['patientId'].tolist()
df_train = pd.concat([df_box, df_aux.drop(labels=['patientId'], axis=1)], axis=1)
print(df_train.groupby(['class', 'Target']).size())
```

3. **DICOM 读取与统一尺寸**。图像直接训练内存放不下（3 万张 1024×1024），我们写一个数据生成器，边读边用，并把每张图缩放到 128×128；DICOM 原始像素是单通道灰度，转成三通道 RGB 才能喂给预训练的 ResNet50：

```python
import pydicom, numpy as np, cv2

def load_and_resize(path, size=128):
    """读取一张 DICOM，转成三通道并缩放到统一尺寸"""
    ds = pydicom.read_file(path)           # 读 DICOM
    img = ds.pixel_array                    # 取出像素数组，1024x1024 灰度
    img = np.stack((img,) * 3, -1)          # 单通道复制三份，凑成 RGB 三通道
    return cv2.resize(img, (size, size))    # 缩放到 128x128 供模型使用
```

数据集我们按病人切分训练集/验证集，避免同一个病人的多张图同时出现在两边而影响评估的公平性。

### 2.3 模型训练与超参数调优

我们的主力模型选择**迁移学习**：直接使用在 ImageNet 上预训练好的 **ResNet50**，去掉最后的全连接层，把它当作一个现成的「特征提取器」，后面接上自定义全连接层。原因很简单：医学图像数据集再怎么大也比不上 ImageNet 的规模，预训练权重已经学会了通用图像的纹理、边缘等基础特征，我们只需要在它基础之上微调各部分即可，训练快而且容易收敛。

网络主体代码如下：

```python
from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, UpSampling2D

model = Sequential()
# 预训练的 ResNet50，只取卷积主体（include_top=False），输入 128x128x3
model.add(ResNet50(input_shape=(128, 128, 3), include_top=False, weights='imagenet'))
# 之后接若干 Dense + 上采样层，逐级把特征图放大并压缩成单通道输出
model.add(Dense(1024, activation='relu'))
model.add(UpSampling2D())
model.add(Dense(512, activation='relu'))
model.add(UpSampling2D())
model.add(Dense(256, activation='relu'))
model.add(UpSampling2D())
model.add(Dense(64, activation='relu'))
model.add(UpSampling2D())
model.add(Dense(8, activation='relu'))
# 最终输出一个 sigmoid 概率：表示该位置/该图有肺炎的可能性
model.add(Dense(1, activation='sigmoid'))

# ResNet 主体冻结，不参与训练，只训练后面的全连接层
model.layers[0].trainable = False
model.summary()

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

训练时我们加上了三个经典的回调函数：**EarlyStopping**（验证损失连续 3 轮不下降就提前停）、**ModelCheckpoint**（只保存验证损失最好的一轮权重）、**ReduceLROnPlateau**（损失不降就把学习率乘 0.8）：

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

earlystop = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint('/kaggle/working/best-{epoch:02d}-{val_accuracy:.2f}.hdf5',
                             monitor='val_loss', save_best_only=True, save_weights_only=True)
# 验证集损失 2 轮没有变好，就把学习率缩小，帮助损失越过小谷底
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2,
                             min_lr=0.0001)

history = model.fit(train_trans,
                    validation_data=valid_trans,
                    epochs=10,
                    callbacks=[earlystop, checkpoint, reduceLR])
```

关于**超参数调优**，我们也尝试过用 `GridSearchCV` 在批大小、学习率等参数上做网格搜索，但每个组合都要把 3 万张图重新过一遍网络，训练一次要很久，中途还容易因为会话超时中断而丢失结果。于是我们改成手工调参：批大小设为 16、图像尺寸 128，观察训练/验证的准确率和损失曲线来决定要不要继续跑。经验是：对医学任务，**先把预训练主体的权重冻结、只训头部**，比一开始就放开全网络更容易得到稳定的验证集分数。

### 2.4 模型测试与指标报告

训练阶段我们额外加了一个 `mean_iou` 指标，用来观察预测的像素级重叠情况：

```python
import tensorflow as tf

def mean_iou(y_true, y_pred):
    """逐像素计算预测与真实掩膜的 IoU，作为辅助观察指标"""
    y_pred = tf.round(y_pred)                        # 概率 >0.5 判为正
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    smooth = tf.ones(tf.shape(intersect))            # 加一个小平滑值，避免除零
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))
```

在验证集上我们画了训练/验证的 accuracy、loss 和 mean_iou 三条曲线，最终模型的验证准确率大约在 **97%** 左右（保存权重时的文件名就带上了 `0.97`）。作为对照，随机瞎猜只有 50%，所以模型确实学到了「有没有肺炎」这个信息。

接下来是**本地复现比赛评分**。对于 Stage 2，光有分类不够，还必须框出位置。我们按比赛的规则实现 `iou` 和 `map_iou` 函数，用来在一小批图上打分：

```python
import numpy as np

def iou(box1, box2):
    """计算两个框的交并比，box 格式：(x1, y1, w, h)"""
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2
    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1 = max(x11, x21), max(y11, y21)
    xi2, yi2 = min(x12, x22), min(y12, y22)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0                                       # 两个框不重叠
    intersect = (xi2 - xi1) * (yi2 - yi1)              # 重叠面积
    union = area1 + area2 - intersect                  # 并集面积
    return intersect / union

def map_iou(boxes_true, boxes_pred, scores, thresholds):
    """在多个 IoU 阈值下计算平均精度，模拟比赛 mAP"""
    # 先按置信度从高到低给预测框排序
    boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
    map_total = 0
    for t in thresholds:                               # 遍历每个 IoU 阈值
        matched_bt = set()                             # 已匹配的真实框下标
        tp, fn = 0, 0
        for bt in boxes_true:
            matched = False
            for j, bp in enumerate(boxes_pred):
                if iou(bt, bp) >= t and not matched and j not in matched_bt:
                    matched = True; tp += 1            # 命中，计一个真正例
                    matched_bt.add(j)                  # 每个预测框只用一次
            if not matched:
                fn += 1                                # 没有匹配上，漏检
        fp = len(boxes_pred) - len(matched_bt)         # 多报的框算误检
        map_total += tp / (tp + fn + fp)
    return map_total / len(thresholds)
```

这个函数能出的最大值是 1.0，越接近 1 越好。它告诉我们两个重要的事情：

- **IoU 阈值越高，框要画得越准。** 阈值 0.4 时，框稍微偏一点还能拿分；到 0.75 时，预测框必须和真实框基本重合才行。
- **误检（FP）和漏检（FN）都一样受罚。** `tp + fp + fn` 是每张图里所有框的总数，报多了、报少了分数都会掉。

所以在准备提交前，还有一个经常被忽视的重头戏——**后处理（post-processing）**。强弱不一的模型会输出一堆互相重叠、置信度高低混杂的框，直接用肯定不行。我们会做下面几件事之一：

- **非极大值抑制（NMS）**：两个框的重叠面积超过阈值（比如 0.05）时，只留下置信度更高的那个，把一堆重复的框压成一个；
- **置信度过滤**：只保留置信度超过一定数值（比如训练调出的阈值）的框，剩下的全当作「图片没有肺炎」；
- **多模型融合**：把检测模型（Mask R-CNN、YOLO）和分割模型（U-Net）的预测结果做一个投票/加权合并，因为不同模型各自的犯错模式不一样，融合之后往往比单模型更稳。

后处理里的这些阈值本身也算超参数，我们会在验证集上扫一遍找出最优组合，确认验证集分数稳定了，再用全部训练数据重新训练一次，并生成提交文件。提交格式必须是两列 `patientId` 和 `PredictionString`，`PredictionString` 里每个框写成 `0 置信度 x y w h`，多个框用空格隔开；如果一个病人没有预测任何框，`PredictionString` 留空。

---

## 三、总结

本文中我们完成了 RSNA 肺炎检测挑战赛的完整流程。我们首先通过 EDA 摸清了数据：主标签表约 3 万行、2.6 万多个病人，正样本约占 30%，负样本的边界框坐标全部为空，辅助表把负样本进一步分为 Normal 和 Not Normal 两类。其次，我们做了有分寸的数据清洗：不去删 NaN，而是利用「空值即无框」这个编码规则，并写了 DICOM 读取生成器把图像统一缩放到 128×128。接着我们采用冻结预训练 ResNet50 主体的迁移学习策略做二分类，配合 EarlyStopping、ModelCheckpoint、ReduceLROnPlateau 三个回调稳定训练，并在验证集上达到约 97% 的准确率。最后我们实现了比赛的 mAP 评分函数做本地验证，并注意到后处理（NMS、置信度过滤、多模型融合）对最终排名的巨大影响。

这个比赛给我们最大的启发是：**检测类任务的功夫一半在模型、一半在输出后处理。** 光有高识别率的骨干网络还不够，IoU 多阈值评分对「框得准」的执念，决定了我们必须认真对待每一个边界框的取舍。同时，DICOM 元数据这类「额外信息」也常常被人忽略，深挖它们往往能带来意想不到的收获。