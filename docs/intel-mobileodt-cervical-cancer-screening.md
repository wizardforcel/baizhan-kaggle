# Intel MobileODT 宫颈癌筛查

## 一、比赛简介

### 1.1 比赛背景

宫颈癌筛查依赖医生识别宫颈的转化区（transformation zone），但这个区域有时不易看见。Intel MobileODT 比赛提供了用移动设备拍摄的宫颈图像数据，任务是把宫颈分为三种正常类型（Type 1、Type 2、Type 3）。这三种类型在病理上都是正常的，但进一步检查的必要性不同：转化区不易看见时需要转诊。自动分类可以提升筛查效率和一致性。

数据来自 MobileODT 采集的临床图像。图像格式为 JPG，形状不一致，部分是圆形裁剪，部分是矩形。

### 1.2 比赛目标

给定一张宫颈图像，预测它属于 Type 1、Type 2 或 Type 3 中的哪一种。提交文件要求输出每个类别的概率。

训练集中 `train/Type_1`、`train/Type_2`、`train/Type_3` 各自存放 JPG。比赛还提供了 `additional` 增量数据。我们需要输出三列概率。

### 1.3 评价标准

比赛使用分类对数损失（categorical log loss）作为评价指标。损失越低越好，完美预测为 0。模型需要输出校准良好的概率，而不是硬标签。

## 二、解题过程

本次解题思路：先统计数据分布并查看图像形态，再统一裁剪/缩放尺寸，使用基于 ResNet 的迁移学习进行三分类，最后在验证集上监控损失并生成概率提交。

### 2.1 数据探索（EDA）

我们先列出训练目录下的文件，统计每个类型有多少张图。

```python
# 导入 pandas 和 glob 用来读取目录
import pandas as pd
from glob import glob

basepath = '../input/train/'
# 收集三种类型的所有图片路径
all_cervix_images = []
for path in sorted(glob(basepath + "*")):
    cervix_type = path.split("/")[-1]
    cervix_images = sorted(glob(basepath + cervix_type + "/*"))
    all_cervix_images = all_cervix_images + cervix_images

all_cervix_images = pd.DataFrame({'imagepath': all_cervix_images})
all_cervix_images['type'] = all_cervix_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)
print('训练集总图像数:', all_cervix_images.shape[0])
# 输出：1481
```

可以看到训练集共 1481 张图。我们再合并 `additional`：

```python
base_a = '../input/additional/'
all_cervix_images_a = []
for path in sorted(glob(base_a + "*")):
    cervix_type = path.split("/")[-1]
    cervix_images = sorted(glob(base_a + cervix_type + "/*"))
    all_cervix_images_a = all_cervix_images_a + cervix_images

all_cervix_images_a = pd.DataFrame({'imagepath': all_cervix_images_a})
all_cervix_images_a['type'] = all_cervix_images_a.apply(lambda row: row.imagepath.split("/")[-2], axis=1)
print('增量数据:', all_cervix_images_a.shape[0])  # 6734
```

总可用数据约 8215 张。Type 2 在训练集中占比超过 50%，Type 1 少于 20%，存在类别不均衡。图像尺寸不一，需要统一处理。

### 2.2 数据清洗

图像形态各异，不能直接喂入网络。我们统一缩放至 64×64，并在读取时裁剪/填充以保持大小。

```python
import cv2
import random
np.random.seed(2016)
random.seed(2016)

conf = dict()
conf['image_shape'] = (64, 64)  # 统一输入尺寸
conf['train_valid_fraction'] = 0.75

def read_and_resize(f):
    img = cv2.imread(f)  # 读取彩色图
    img = cv2.resize(img, conf['image_shape'])  # 缩放到 64x64
    return img / 255.0
```

我们还把所有路径打平成列表，然后按 0.75/0.25 划分训练/验证。注意我们没有删除数据，只做尺寸统一。

```python
filepaths = ['../input/train/Type_1/', '../input/train/Type_2/', '../input/train/Type_3/']
allFiles = []
for fp in filepaths:
    files = glob.glob(fp + '*.jpg')
    allFiles = allFiles + files

random.shuffle(allFiles)
split_point = int(round(conf['train_valid_fraction']*len(allFiles)))
train_list = allFiles[:split_point]
valid_list = allFiles[split_point:]
print('训练:', len(train_list), '验证:', len(valid_list))
# 训练 1111, 验证 370
```

### 2.3 模型训练与超参数调优

我们使用 ResNet 34 的改进版作为骨干，输入三通道图，输出三类 softmax。

```python
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 简化版 ResNet 构造逻辑，实际使用 ResnetBuilder.build_resnet_34
img_channels = 3
model = ResnetBuilder.build_resnet_34((img_channels, 64, 64), 3)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

训练时我们用 `fit_generator`，并设置早停：

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=0),
    ModelCheckpoint('cervical_best.hdf5', monitor='val_loss', save_best_only=True, verbose=0)
]

# batch_generator_train 负责读取、缩放并生成 one-hot 标签
fit = model.fit_generator(
    generator=batch_generator_train(train_list, batch_size=1),
    nb_epoch=1,
    samples_per_epoch=3,
    validation_data=batch_generator_train(valid_list, batch_size=1),
    nb_val_samples=1,
    callbacks=callbacks
)
```

超参数方面，我们先固定图像尺寸 64×64、Batch Size 1、早停 patience 3，训练验证比例 0.75。后续可尝试增加数据增强（如旋转、翻转）或使用增量数据来平衡类别。

### 2.4 模型测试与指标报告

在验证集上评估损失与准确率，并用全量训练数据重新拟合后预测测试集。

```python
# 加载最优权重
model.load_weights('cervical_best.hdf5')

# 在验证集上评估
val_loss, val_acc = model.evaluate_generator(
    batch_generator_train(valid_list, 1),
    nb_val_samples=len(valid_list)
)
print('val_loss:', val_loss, 'val_acc:', val_acc)
```

预测阶段我们把测试图像同样缩放至 64×64，取 softmax 概率并按提交格式输出。

```python
import pandas as pd
# 假设 test_files 已读取
# preds = model.predict_generator(test_generator(), nb_val_samples=len(test))
submission = pd.DataFrame({
    'image': [os.path.basename(f) for f in test_files],
    'Type_1': preds[:,0],
    'Type_2': preds[:,1],
    'Type_3': preds[:,2]
})
submission.to_csv('submission.csv', index=False)
```

## 三、总结

本文中我们完成了 Intel MobileODT 宫颈癌筛查的完整流程。我们首先用 EDA 摸清数据：仅训练集 1481 张图，Type 2 占比最高，Type 1 最少，加入增量数据后约 8215 张；图像形状不统一。其次我们统一把图像缩放到 64×64 并按 0.75 划分验证集。接着我们用 ResNet 34 搭建三分类网络，采用 categorical_crossentropy 和早停进行训练。最后在验证集上监控损失并生成概率提交，说明在小样本医疗图像上结构化预处理和早停是稳健的起点。