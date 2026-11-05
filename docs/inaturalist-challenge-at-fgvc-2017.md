# iNaturalist Challenge at FGVC 2017

## 一、比赛简介

### 1.1 比赛背景
iNaturalist 是一个公民科学平台，收集了大量由自然爱好者拍摄的野生动植物照片。iNaturalist Challenge at FGVC 2017 由细粒度视觉分类研讨会举办，目的是让机器学习模型能够从普通照片中识别出物种。为了让模型学会分辨相似物种，比赛提供了一个包含数百万张图片的数据集。材料中出现的图片路径说明数据来源是 Kaggle 的输入文件夹，包含 `train2017.json`、`val2017.json`、`test2017.json`，以及 `train_val_images_mini`，说明我们可以先用小规模图片做探索。

这个比赛的难点在于物种数量非常多，许多物种外观差异很小，容易混淆。数据来源于用户上传，拍摄角度、光线、背景都非常多样。

### 1.2 比赛目标
比赛提供训练集和验证集的 JSON 标注文件，记录每张图的 `imageId` 和对应的物种标签。测试集只有图片，我们需要为每张测试图片预测出物种类别。

从代码片段来看，输入目录里有 `train2017.json`、`test2017.json`、`train_val_images_mini`。说明我们可以通过 `imageId` 找到对应的图片文件，并读取标签。数据集规模相当大，需要高效的读取和模型训练流程。

### 1.3 评价标准
材料比较稀少，没有直接给出指标定义。iNaturalist 类比赛通常使用 Top-1 分类准确率，即预测的最可能类别与真实类别是否一致。因为物种数量非常多，随机猜测的准确率接近于零，模型必须学到细微的形态差异。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先查看输入目录，确认有哪些文件。

```python
from subprocess import check_output
import os

# 列出输入目录里的文件，确认数据清单
files = check_output(["ls", "../input/"]).decode("utf8")
print(files)
```
输出包含 `kaggle_sample_submission.csv`、`test2017.json`、`train2017.json`、`train_val_images_mini`、`val2017.json`。这说明训练、验证、测试都有 JSON 标注，图片保存在 `train_val_images_mini`。

接着我们统计图片总数和标签分布。

```python
import json
import pandas as pd

# 用 json 加载标注，而不是用 CSV，避免嵌套结构丢失
with open('../input/train2017.json') as f:
    train = json.load(f)
with open('../input/val2017.json') as f:
    val = json.load(f)

print('train images:', len(train['images']))
print('val images:', len(val['images']))
```
虽然具体数字在材料里未给出，我们可以先假设数量在上万级别，后续可以用实际加载来确认。

我们还可以用 seaborn 画出标签分布，观察类别是否平衡。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 提取物种标签列表
labels = [item['category'] for item in train['annotations']]
# 统计每个物种的出现次数
label_counts = pd.Series(labels).value_counts()
sns.barplot(x=label_counts.index[:30], y=label_counts.values[:30])
plt.title('Top 30 species by count')
plt.xticks(rotation=90)
plt.show()
```
细粒度任务中，长尾分布很常见，少数常见物种占多数样本，很多稀有物种只有几张图片。EDA 能帮我们了解是否需要重采样或者加权损失。

### 2.2 数据清洗
我们首先把 JSON 的嵌套结构摊平，得到一张图片一个标签的表格。

```python
# 训练标注一般有 image_id 与 category 两个字段
train_df = pd.DataFrame(train['annotations'])
val_df = pd.DataFrame(val['annotations'])
train_df.head()
```
随后检查是否存在 `NaN` 或重复 `imageId`。

```python
# 检查缺失值
print(train_df.isnull().sum())

# 去除重复标注，避免同张图被计两次
train_df = train_df.drop_duplicates(subset=['image_id'])
```
图片的读取速度是瓶颈，我们可以用 `ImageDataGenerator` 配合预处理。

```python
from keras.preprocessing.image import ImageDataGenerator

# 训练时做随机翻转、裁剪；验证时只做归一化
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
```
这样清洗后，数据就具备了可复现的训练流水线。

### 2.3 模型训练与超参数调优
我们选用 ResNet50 或 DenseNet121 作为骨干网络，用 ImageNet 预训练权重启动。

```python
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

# 加载预训练主干，冻结底层的特征提取器
base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base.layers[:-10]:
    layer.trainable = False

# 在主干上接全连接层，输出物种数量
x = GlobalAveragePooling2D()(base.output)
out = Dense(num_classes, activation='softmax')(x)
model = Model(base.input, out)
```
细粒度任务需要更精细的特征，冻结较少的层、减小学习率对性能有帮助。我们先用较小的学习率训练分类头，再解冻部分层微调。

```python
from keras.optimizers import Adam

# 先训练分类头
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
# 之后解冻并用更小学习率微调
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
```
超参数方面，我们关注图像尺寸、批次大小和学习率。材料里没有给出具体数值，我们采用 224×224 输入、批大小 32、学习率 1e-4 作为起点，并在验证集上监控准确率。类别不平衡时，可以在损失函数里加上类别权重。

```python
# 类别权重根据频次倒数计算
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced',
                                     classes=np.unique(train_df['category']),
                                     y=train_df['category'])
```
### 2.4 模型测试与指标报告
我们在验证集上评估 Top-1 准确率。

```python
from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

# 生成验证集预测
y_pred = model.predict(val_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

acc = accuracy_score(y_true, y_pred_classes)
print('Validation accuracy:', acc)
```
也可以按类别拆分，看看长尾类别的表现。

```python
# 计算每个物种的准确率，找出表现最差的几个
report = classification_report(y_true, y_pred_classes, output_dict=True)
per_class = pd.DataFrame(report).T
worst = per_class.sort_values('precision').head(10)
print(worst)
```
最后，我们用全量训练数据重新拟合，并在测试集上生成提交文件。

```python
# 重新拟合
model.fit(train_generator, validation_data=val_generator, epochs=20)

# 预测测试集
test_pred = model.predict(test_generator)
test_labels = np.argmax(test_pred, axis=1)

submission = pd.DataFrame({
    'image_id': test_df['image_id'],
    'category': test_labels
})
submission.to_csv('submission.csv', index=False)
```
我们检查行数与 sample submission 一致，避免格式错误。

## 三、总结

本文中我们完成了 iNaturalist Challenge at FGVC 2017 的细粒度物种识别流程。我们首先通过 EDA 确认了数据包含 train、val、test 三个 JSON 标注以及小规模图片目录，初步了解了物种分布的长尾特点。其次我们把 JSON 标注展平为表格，删除重复和缺失，并使用 ImageDataGenerator 做预处理与增广。然后我们采用 ResNet50 预训练模型，在分类头上先训练再微调部分层，并考虑类别权重解决不平衡问题。最后我们在验证集上计算 Top-1 准确率，并生成测试集的提交文件。

这个比赛提醒我们，细粒度分类的核心是特征表达和数据清洗。类别很多且相似时，预训练权重和数据增广比单纯加深网络更有效；同时检查提交格式、避免重复标注，才能把模型性能真正转化为排行榜分数。
