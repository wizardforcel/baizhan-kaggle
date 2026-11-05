# Facial Keypoints Detection（人脸关键点检测）

## 一、比赛简介

### 1.1 比赛背景

人脸关键点检测（facial keypoint detection）是计算机视觉里一个很经典的任务：给一张人脸照片，找出眼睛、鼻子、嘴角、眉毛等部位的**精确像素坐标**。它是很多应用的基础——比如手机相机的人脸贴纸、美颜、表情动画，都要先知道「眼睛在哪、嘴角在哪」，才能把特效贴到正确的位置。

比赛数据来自真实人脸照片：每张图是 **96×96 像素的灰度图**，标注了 **15 个关键点**（每个点有 x、y 两个坐标，所以一共 30 个数字）。这些坐标都是相对于图片左上角的像素位置，范围在 0～96 之间。

这个任务的难点在于：

- **每个点的标注并不完整**。训练集里 28 列坐标存在缺失值，有的图只标了眼睛、没标嘴角，有的图全部 15 个点都标了；
- **人和人长得不一样**。脸的大小、朝向、表情都会让同一个关键点的像素位置变化很大，模型不能死记坐标，得学会「长得像脸的地方」；
- **这是回归问题不是分类问题**。我们要输出连续的小数坐标，而不是类别。

### 1.2 比赛目标

输入一张 96×96 的灰度人脸图，输出 15 个关键点的坐标（30 个实数）。

具体的数据文件有四个：

- `training.zip` 解压得到 `training.csv`，7049 行，每行一张图 + 30 个坐标列 + 一列 `Image`；
- `test.zip` 解压得到 `test.csv`，1783 行，只有 `Image`，没有坐标；
- `IdLookupTable.csv`：告诉我们要提交哪些「图片 + 关键点」组合；
- `SampleSubmission.csv`：提交格式样例。

`training.csv` 的 30 个坐标列名字很有规律，都是「部位 + `_x` / `_y`」，例如 `left_eye_center_x`、`nose_tip_y`、`mouth_center_bottom_lip_x`。`Image` 列是一整行用空格分隔的像素值，共 96×96 = 9216 个数字。

提交文件不是「每张图一行」，而是 `IdLookupTable.csv` 里每一行一个 (ImageId, FeatureName)，要求填上该图该部位的坐标值 `Location`。所以一张测试图可能只要求提交其中几个点。

### 1.3 评价标准

比赛用 **均方根误差（RMSE，Root Mean Squared Error）** 衡量预测坐标和真实坐标的差距。通俗说，就是把所有要预测的坐标点，算「预测值减真实值」的平方、取平均、再开根号。单位是**像素**：RMSE 等于 4 就意味着平均每个关键点偏了大约 4 个像素。**这个指标越低越好**，0 是完美。

---

## 二、解题过程

我们的总体思路是：**把「找关键点」当作「对 30 个数字做回归」，用一个卷积神经网络（CNN）直接看图输出 30 个坐标。** 卷积层负责从图像里提取「眼角是尖的」「鼻头是圆的」这类形状特征，最后的全连接层把这些特征映射成坐标。我们按四步走。

### 2.1 数据探索（EDA）

我们首先解压数据、把训练集和测试集读进来，看看规模：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('training.csv')   # 7049 张带标注的人脸
test_data  = pd.read_csv('test.csv')       # 1783 张待预测的人脸
lookid_data = pd.read_csv('../input/facial-keypoints-detection/IdLookupTable.csv')

# 把前 5 行转置过来看，列太多，转置后更直观
train_data.head().T
```

打印出来能看到：前 30 列是坐标（如 `left_eye_center_x = 66.03`、`left_eye_center_y = 39.00`），最后一列 `Image` 是长长的像素串。坐标值都在 0～96 之间，符合 96×96 图片的坐标系。

我们接着检查缺失情况，这是这份数据最重要的特点：

```python
# 统计每一列的缺失比例
all_data_na = train_data.isnull().sum() / len(train_data)
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
missing_data.head(20)
```

结果显示 **30 个坐标列里有 28 列存在缺失**。这是关键发现：不是每张图都标注了全部 15 个点。很多图只标了 4 个眼睛中心点，其余 11 个点是空的。这会影响我们后面清洗和评价的方式。

我们还想亲眼确认「图片长什么样、关键点在哪」，于是把像素串还原成 96×96 的矩阵画出来，再把坐标点标上去：

```python
# Image 列是按空格分隔的字符串，先切分、空值补 0，再转成浮点数
img = train_data['Image'][0].split(' ')
img = ['0' if x == '' else x for x in img]
image = np.array(img, dtype='float').reshape(96, 96)

plt.imshow(image, cmap='gray')
# 只取前 4 个点（左右眼中心）画绿点，验证坐标和图像对得上
plt.plot(train_data.iloc[0, 0:4:2], train_data.iloc[0, 1:4:2], 'go')
plt.show()
```

这一步很值得做：它让我们确信「坐标列确实是像素位置」，也直观看到人脸在 96×96 画框里大致居中、方向不一。

### 2.2 数据清洗

清洗要解决两个问题：**缺失的坐标**和**字符串格式的像素**。

对于缺失坐标，我们有两个选择：一是把含有缺失的行全部删掉——但这样会丢掉大部分训练数据；二是做填充。我们选择填充，而且用了「前向填充和后向填充取平均」的办法，比单纯用前值填充更稳：

```python
# 对每一行先做前向填充（用上面一行的值补），再做后向填充（用下面一行的值补），
# 然后两者取平均；这样比只用 ffill 更平滑，实验效果也更好
train_data = train_data.fillna(
    pd.concat([train_data.ffill(), train_data.bfill()]).groupby(level=0).mean()
)

# 确认此时已经没有缺失了
train_data.isnull().any().value_counts()
# False    31   —— 30 个坐标列 + Image 列，全部有值
```

这里要说明一个取舍：填充让「所有行都能参与训练」，但也会引入一些不精确的标签。好在缺失的坐标大多是同一张图上不太常考的次要关键点，填充值取自同一行相邻的标注，误差相对可控。

接着把像素字符串变成真正的图像张量。要注意字符串里可能有连续空格，切分后会产生空字符串，需要替换成 0：

```python
# 把每行的像素串切分成 9216 个数，空字符串补 0
imag = []
for i in range(len(train_data)):
    img = train_data['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img]
    imag.append(img)

# 转成 float，再重塑成 (样本数, 96, 96, 1)，最后 1 表示单通道灰度图
image_list = np.array(imag, dtype='float')
X_train = image_list.reshape(-1, 96, 96, 1)

# 标签就是去掉 Image 列后的 30 个坐标
training = train_data.drop('Image', axis=1)
y_train = np.array(training, dtype='float')
```

测试集用完全相同的办法处理，得到 `X_test`（1783 张图）。

### 2.3 模型训练与超参数调优

我们用一个「一路卷积 + 末端全连接」的 CNN。结构上是 5 个卷积块：每个块里连续两个 3×3 卷积，配 `LeakyReLU` 激活和批归一化（BatchNormalization），再用 2×2 最大池化把尺寸减半。通道数依次是 32 → 64 → 128 → 256 → 512。

为什么这么设计？卷积核负责「局部看形状」——第一层只能看到 3×3 的小块边缘，越往后感受野越大，能组合出「眼睛」「鼻梁」这类结构；池化负责逐步缩小尺寸，让最后的全连接层能承受。批归一化则让训练更稳、收敛更快。

```python
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, BatchNormalization
from keras.layers import Flatten, Dense, Dropout

model = Sequential()

# 第 1 个卷积块：32 通道，输出尺寸仍是 96x96
model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False,
                        input_shape=(96, 96, 1)))
model.add(LeakyReLU(alpha=0.1))          # 负半轴给一个小斜率，避免神经元"死掉"
model.add(BatchNormalization())
model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))   # 96 -> 48

# 第 2 个卷积块：64 通道
model.add(Convolution2D(64, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))   # 48 -> 24

# 第 3 个卷积块：128 通道
model.add(Convolution2D(128, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))   # 24 -> 12

# 第 4 个卷积块：256 通道
model.add(Convolution2D(256, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))   # 12 -> 6

# 第 5 个卷积块：512 通道，不再池化
model.add(Convolution2D(512, (3, 3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())

# 收尾：拉平 -> 512 维全连接 -> Dropout -> 输出 30 个坐标
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))                  # 随机丢掉 10% 神经元，抑制过拟合
model.add(Dense(30))                     # 30 = 15 个关键点 x 2 个坐标

model.summary()   # 总参数量约 1417 万
```

输出层不接激活函数，因为它要直接输出任意实数坐标（回归任务），而不是概率。

**损失函数**选择均方误差（MSE），这和比赛的 RMSE 指标是同一类的度量——让 MSE 变小，RMSE 自然跟着变小。**优化器**用 Adam，它在各种视觉任务里通常收敛快、对学习率不敏感。训练时把 20% 的数据留作验证：

```python
# 编译：Adam 优化器 + 均方误差损失
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# 训练：50 轮，每批 512 张，留 20% 做验证
log = model.fit(X_train, y_train,
                epochs=50,
                batch_size=512,
                validation_split=0.2)
```

训练日志里，第一轮验证损失是 1739，第二轮降到 847，第三轮 658，第四轮 368……损失快速下降，说明网络确实在学「从像素到坐标」的映射。你可能会注意到日志里还打了 `accuracy`，那只是因为 Keras 默认带了这个指标；对回归任务来说 `accuracy` 没有实际意义，我们只看 loss。

**关于超参数**：我们主要调整了网络深度/宽度、Dropout 比例和批大小。更大的批能让训练更快但泛化略差，512 是一个折中；Dropout 从 0 调到 0.1 能缓解验证损失反弹。轮数上 50 轮已经能看到验证损失趋于平缓，再多容易过拟合。

### 2.4 模型测试与指标报告

模型在验证集上表现稳定后，我们用它预测测试集，但这里有一个必须处理的细节：**提交文件是按 `IdLookupTable` 组织的**，每行只问「某张图的某个关键点」，所以要从 30 个输出里挑出对应的那一个：

```python
# 预测全部 1783 张测试图的 30 个坐标
preds = model.predict(X_test)

# IdLookupTable 每行给出 ImageId 和 FeatureName
id_lookup = pd.read_csv('../input/facial-keypoints-detection/IdLookupTable.csv')
feature_names = list(training.columns)   # 30 个坐标列的名字，顺序与模型输出一致

locations = []
for _, row in id_lookup.iterrows():
    image_id = int(row['ImageId']) - 1   # 表格里的编号从 1 开始
    feature_index = feature_names.index(row['FeatureName'])
    locations.append(preds[image_id][feature_index])

submission = pd.DataFrame({
    'RowId': id_lookup['RowId'],
    'Location': locations
})
submission.to_csv('submission.csv', index=False)
```

这里如果用错了顺序（比如把 `ImageId` 当成从 0 开始），或者 `FeatureName` 对错了列，提交出去会全线崩盘，所以这个「编号对齐」的检查必须做。

最终这份 CNN 方案在 Kaggle 排行榜上的成绩是：**公开榜（public）RMSE 4.55126，私有榜（private）RMSE 4.24206**。也就是说，平均每个关键点的预测位置离真实位置大约 4 个像素——在 96×96 的人脸上，这个误差已经能让人一眼看出预测点落在正确的器官上。

我们也试过另一条路线：用 PyTorch + EfficientNet 做迁移学习，配合 Albumentations 做「旋转、剪切、亮度变化」等数据增强。这里的关键是增强时必须**同时变换图片和关键点坐标**，否则标签就错位了：

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transforms = A.Compose([
    A.Resize(width=96, height=96),
    A.Rotate(limit=15, p=0.8),                     # 随机旋转 ±15 度
    A.RandomBrightnessContrast(p=0.2),             # 随机亮度/对比度
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, p=0.2),
    A.Normalize(mean=[0.4897]*3, std=[0.2330]*3),   # 用数据集统计量归一化
    ToTensorV2(),
],
# 关键点格式指定为 xy，增强时坐标会跟着图片一起变
keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))
```

同时在损失里把「缺失的坐标」掩掉——数据清洗时我们用 -1 标记缺失位置，算损失时只对有效坐标回传梯度：

```python
# 只对 targets != -1 的位置计算损失，缺失的关键点不参与训练
loss = loss_fn(scores[targets != -1], targets[targets != -1])
```

最后把预测坐标截断到 0～96，防止网络输出越界：

```python
preds = torch.clip(model(image).squeeze(0), 0.0, 96.0)
```

这条迁移学习路线的思路是「借用在大规模图像上预训练好的特征提取器」，在小数据集上往往比从零训练更稳；而前面从零训练的 CNN 方案结构简单、代码短，更适合入门理解。

---

## 三、总结

本文中我们完成了人脸关键点检测的完整流程。我们首先做了 EDA，确认训练集有 7049 张 96×96 灰度人脸图，每张图标注 15 个关键点（30 个坐标），并且发现 28 个坐标列存在缺失。其次我们做了数据清洗：用前向填充和后向填充取平均来补齐缺失坐标，把空格分隔的像素字符串还原成 96×96 的单通道图像张量。接着我们搭了一个 5 段式卷积神经网络（通道 32→64→128→256→512，LeakyReLU + 批归一化 + 池化，末端全连接 512→30），用 MSE 损失和 Adam 优化器训练 50 轮。最后我们按 `IdLookupTable` 对齐图片和部位生成提交文件，在排行榜上取得公开榜 RMSE 4.55126、私有榜 RMSE 4.24206 的成绩；我们还实践了「图片与关键点同步增强」的 PyTorch + Albumentations 迁移学习路线。这个比赛教会我们的是：**回归任务的关键在于输出层不接激活、损失函数要和评价指标同向**，以及数据增强时必须把标签一起变换。