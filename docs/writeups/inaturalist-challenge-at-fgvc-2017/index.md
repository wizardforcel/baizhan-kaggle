# iNaturalist Challenge at FGVC 2017

## 一、比赛简介

### 1.1 比赛背景
iNaturalist 是一个面向自然观察者的公民科学平台，用户上传自己拍到的动植物照片，并请社区帮忙鉴定物种。几年下来，平台积累了大量带物种标签的图片，这为训练细粒度图像分类器提供了宝贵数据。iNaturalist Challenge at FGVC 2017 由 FGVC（Fine-Grained Visual Categorization）研讨会主办，目标就是让机器从这些自然照片中识别物种。

这个任务和常见的 ImageNet 分类很不一样。ImageNet 的类别是狗、猫、汽车这种差别很大的物体，而这里的类别是成千上万个物种，很多物种之间只差一个斑点、一条纹路或者一个翅膀形状。照片又来自普通用户，存在大量遮挡、模糊、逆光和奇怪的角度，背景里还有树枝、叶子、石头等干扰。因此模型必须具备极强的细粒度区分能力。

这些困难也让比赛非常适合研究迁移学习：从 ImageNet 预训练模型出发，把通用视觉特征迁移到生物物种识别上，再通过数据增广和微调适配自然场景。

### 1.2 比赛目标
数据在材料中以 JSON 标注形式出现：`train2017.json` 保存训练集图片和类别，`val2017.json` 保存验证集，`test2017.json` 保存测试集。图片本身放在 `train_val_images_mini` 目录下，说明这是一个缩小版的训练/验证图片集，方便在 notebook 里快速实验。

每张图片有 `image_id` 和对应的物种标签。训练时我们要学习图片到物种的映射；预测时给每个测试图片输出一个物种 ID。分类的类别数很多（材料里没有给出准确数字，从文件名和比赛规模看是数千类），而且各类样本数量差别很大。

最终提交格式一般是 `image_id, category` 两列，每个测试图片一行。

### 1.3 评价标准
材料没有直接给出指标定义，但这类细粒度物种识别比赛通常采用 Top-1 准确率：模型输出的最可能类别与真实类别相同就算正确。Top-5 准确率有时也会作为辅助参考，即真实类别出现在前五个预测里就算对。

因为类别数量巨大，随机猜中的概率极低，所以准确率能很直观地反映模型是否真的学到了物种特征。比赛排行榜以 Top-1 准确率排序，越高越好。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先列出输入目录，确认文件结构。

```python
from subprocess import check_output
import os

# 查看 Kaggle 输入目录，确认有哪些数据文件
files = check_output(["ls", "../input/"]).decode("utf8")
print(files)
```
输出里有 `kaggle_sample_submission.csv`、`test2017.json`、`train2017.json`、`train_val_images_mini`、`val2017.json`。这说明训练、验证、测试都有 JSON 标注，图片则在 `train_val_images_mini` 里。

我们接着读取 JSON，统计图片数量和类别数。

```python
import json
import pandas as pd

with open('../input/train2017.json') as f:
    train = json.load(f)
with open('../input/val2017.json') as f:
    val = json.load(f)

# images 记录图片，annotations 记录标签
print('train images:', len(train['images']))
print('val images:',   len(val['images']))
print('train annotations:', len(train['annotations']))
```
从 JSON 的键可以看出，训练和验证都包含 `images`、`annotations`、`categories` 三个数组。`categories` 里存着物种 ID 和名称，`annotations` 里把 `image_id` 与 `category_id` 连起来。

我们用一个表格把标注摊平。

```python
# 把嵌套 JSON 变成一张图片-标签表
ann = pd.DataFrame(train['annotations'])
img = pd.DataFrame(train['images'])
train_df = ann.merge(img, left_on='image_id', right_on='id')
train_df[['image_id', 'category_id', 'file_name']].head()
```
这样每行就是一张图的一个标签，后续统计和读图都方便。

接着看类别分布。

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 统计每个物种的样本数
counts = train_df['category_id'].value_counts()
print('类别数:', counts.shape[0])
print('最多:', counts.iloc[0], '最少:', counts.iloc[-1])

# 画出前 30 个类别的样本数
sns.barplot(x=counts.index[:30], y=counts.values[:30])
plt.xticks(rotation=90)
plt.title('Top 30 species frequency')
plt.show()
```
典型的自然数据集会呈现长尾分布：常见物种有上千张图片，稀有物种可能只有几张。长尾意味着模型容易偏向头部类别，稀有物种的召回率会比较低。

我们还查看了图片尺寸和颜色分布。

```python
from PIL import Image
import numpy as np

# 随机抽几张图片看尺寸
sample_files = train_df['file_name'].sample(5, random_state=0)
for f in sample_files:
    img = Image.open('../input/train_val_images_mini/' + f)
    print(f, img.size, img.mode)
```
自然照片的尺寸和长宽比差异很大，有竖拍也有横拍，有近景也有远景。这提示我们在预处理阶段必须统一裁剪或缩放，同时用数据增广模拟这些变化。

### 2.2 数据清洗
我们其次清洗标注和图片路径。先检查缺失值和重复。

```python
# 检查标注表是否有缺失
print(train_df.isnull().sum())

# 同一个图片重复标注同一类别时只保留一条
train_df = train_df.drop_duplicates(subset=['image_id', 'category_id'])
```
对于分类任务，一张图通常只有一个物种标签，所以如果 `annotations` 里出现多条，我们取第一条或者按置信度保留最可信的一条。

接着我们把类别 ID 重新编号成 0 到 N-1 的连续整数，因为 Keras 的损失函数要求标签是连续索引。

```python
# 建立类别 ID 到连续索引的映射
categories = pd.DataFrame(train['categories'])
cat_ids = sorted(categories['id'].unique())
cat_to_idx = {c: i for i, c in enumerate(cat_ids)}
idx_to_cat = {i: c for c, i in cat_to_idx.items()}

train_df['label'] = train_df['category_id'].map(cat_to_idx)
```
然后我们清洗图片路径。材料中的图片目录是 `train_val_images_mini`，文件名在 JSON 的 `file_name` 字段里。有些文件名可能带子目录，我们统一拼接。

```python
IMG_DIR = '../input/train_val_images_mini/'

# 只保留文件真实存在的记录，避免训练时报 FileNotFoundError
import os
train_df['path'] = train_df['file_name'].apply(lambda x: os.path.join(IMG_DIR, x))
train_df = train_df[train_df['path'].apply(os.path.exists)]
```
最后我们把数据按类别分层切分。虽然官方已经给了验证集，但我们还是从训练集里再切一小部分做快速验证，方便调参。

```python
from sklearn.model_selection import train_test_split

# 分层抽样，保证验证集里每个物种都有样本
train_idx, val_idx = train_test_split(
    train_df.index, test_size=0.15,
    stratify=train_df['label'], random_state=42)
```

### 2.3 模型训练与超参数调优
我们采用迁移学习。自然物种识别数据量有限，从零训练很难收敛，而 ImageNet 预训练模型已经学会了边缘、纹理、形状等通用特征，正好可以迁移。

```python
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model

# 加载 ResNet50 主干，去掉顶层分类头
base = ResNet50(weights='imagenet', include_top=False,
                input_shape=(224, 224, 3))
# 先冻结主干，只训练新加的分类头
for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)     # 把特征图压成一个向量
x = Dropout(0.3)(x)                 # 防止过拟合
out = Dense(num_classes, activation='softmax')(x)
model = Model(base.input, out)
```
这里的 `GlobalAveragePooling2D` 是把卷积层输出的每个通道取平均，得到一个固定长度向量。相比直接展平，它参数更少，不容易过拟合。

接着编译模型。第一阶段用较大的学习率训练分类头。

```python
from keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
```
数据增广是自然图像任务的关键。我们使用随机旋转、缩放、水平翻转和亮度变化，模拟用户拍摄时的各种条件。

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    validation_split=0.15)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)
```
注意验证集只做缩放，不做随机增广，否则评估结果会不稳定。

训练完分类头后，我们解冻 ResNet50 的最后几个卷积块，用很小的学习率微调。原因是底层学的边缘和纹理对物种也有用，但高层特征需要适配本任务。

```python
# 解冻最后 20 层
for layer in base.layers[-20:]:
    layer.trainable = True

# 微调阶段学习率要小，避免破坏预训练权重
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
超参数方面，我们重点调整图像尺寸、批大小和学习率。224×224 是常用起点；如果显存不足就降到 160×160。批大小在 16 到 64 之间试。类别不平衡时，可以给损失函数加类别权重。

```python
from sklearn.utils.class_weight import compute_class_weight

# 按类别频率的倒数计算权重，缓解长尾问题
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=train_df['label'].values)
class_weight = dict(enumerate(weights))
```
我们没有采用特别复杂的网络结构，因为材料只提供了图片和标注，没有预训练检测器或多模态信息。对入门者来说，ResNet50 + 增广 + 微调已经是一个可靠的强基线。

### 2.4 模型测试与指标报告
训练完成后，我们在验证集上计算准确率。

```python
# 评估验证集
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path', y_col='label',
    target_size=(224, 224),
    batch_size=32, class_mode='sparse', shuffle=False)

loss, acc = model.evaluate(val_generator)
print('Validation accuracy:', acc)
```
为了看清模型在哪些物种上表现差，我们输出每个类别的精确率和召回率。

```python
from sklearn.metrics import classification_report

# 预测整个验证集
val_generator.reset()
y_prob = model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
y_true = val_generator.classes

report = classification_report(y_true, y_pred, output_dict=True)
per_class = pd.DataFrame(report).T
print(per_class.sort_values('recall').head(10))
```
通常头部物种的召回率很高，尾部物种很低。这说明模型还能进一步提升，方向包括重采样、focal loss 或者按类别调整阈值。

训练过程中我们使用了检查点（checkpoint）和学习率衰减。检查点保存验证集准确率最高的模型权重，学习率衰减则在验证损失不再下降时把学习率减半，这两者可以避免训练后期在最优解附近来回震荡。

```python
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    # 只保存验证准确率最高的模型
    ModelCheckpoint('best_model.h5', monitor='val_accuracy',
                    save_best_only=True, mode='max'),
    # 验证损失停滞时降低学习率
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-7)
]
```

我们还在训练前做了小规模试跑，用几千张图片先跑 1 到 2 个 epoch，确认数据管线、标签映射和图片路径都没问题，再扩大到全量数据。这样做比一开始就全量训练更节省时间，也更容易定位错误。

最后我们用全量训练数据重新训练，对测试集做预测并生成提交。

```python
# 在全量训练集上再训练若干轮
model.fit(train_generator, epochs=10)

# 预测测试集
test_generator.reset()
test_prob = model.predict(test_generator, verbose=1)
test_idx = np.argmax(test_prob, axis=1)

submission = pd.DataFrame({
    'image_id': test_df['image_id'],
    'category': [idx_to_cat[i] for i in test_idx]
})
submission.to_csv('submission.csv', index=False)
```
我们检查提交行数是否等于测试图片数，并确认 `category` 用的是原始类别 ID，而不是模型内部的连续索引。

如果条件允许，我们还会做多模型投票或单模型多尺度预测。例如对同一张图片取 224×224 和 256×256 两个尺寸分别预测，再把概率平均，通常能小幅提升 Top-1 准确率。材料没有给出具体的排行榜数字，因此我们不虚构最终分数，只把验证集上的准确率作为相对比较的依据。由于类别非常多，即使准确率只有 60%～70%，也远高于随机猜测，说明模型确实学到了物种的形态特征。

## 三、总结

本文中我们完成了 iNaturalist Challenge at FGVC 2017 的物种识别流程。我们首先通过 EDA 确认了数据由 train、val、test 三个 JSON 和一份小规模图片目录组成，了解到物种数量多、类别分布长尾、图片尺寸和拍摄条件差异大。其次我们把 JSON 标注展平成表格，删除重复和缺失，建立类别到连续索引的映射，并过滤不存在的图片路径。然后我们采用 ResNet50 迁移学习，先冻结主干训练分类头，再解冻最后几层微调，同时使用旋转、缩放、翻转、亮度变化等数据增广，并用类别权重缓解不平衡。最后我们在验证集上计算 Top-1 准确率和逐类召回率，并按格式生成提交文件。

这个比赛让我们体会到，细粒度分类的难点不在网络有多深，而在数据质量和特征细节。预训练模型、强数据增广和类别不平衡处理是三个最有效的抓手；同时要按类别看指标，否则总准确率会被头部物种掩盖，尾部物种的真实表现看不出来。
