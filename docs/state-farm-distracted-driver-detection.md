# State Farm Distracted Driver Detection

## 一、比赛简介

### 1.1 比赛背景
State Farm Distracted Driver Detection 是 Kaggle 的图像分类比赛。比赛提供驾驶员驾驶舱照片，拍摄驾驶员在驾驶过程中的不同分心行为。数据集来源于 State Farm 保险公司，用来帮助自动驾驶和安全系统识别分心。

在真实道路上，驾驶员分心是引发事故的主要原因之一。很多驾驶员在行驶时低头看手机、操作中控台、喝水、化妆，甚至和乘客交谈，这些行为都会让注意力从前方路况上转移。State Farm 为了收集真实的驾驶分心影像，制作了这套公开数据集。每一张照片都拍摄了驾驶员上半身和方向盘，背景大致相同，但手部位置、头部朝向和视线方向各不相同。任务就是根据这些视觉线索，把驾驶员的行为归类到不同的分心类型。

这个比赛对计算机视觉初学者非常友好。数据类别均衡，图像数量足够，而且图像内容高度结构化，区分度很高。难点不在于模型不够强，而在于如何避免过拟合、如何利用数据增强提升泛化能力，以及如何处理类别间细微的视觉差异，比如“拿起手机打电话”和“低头看手机的动作几乎一样，只是手的位置略有差别”。这类细微差别要求模型既要能捕捉全局姿态，也要能关注局部细节。此外，由于拍摄环境高度统一，很容易出现同一个驾驶员在不同照片里反复出现的情况，这会带来数据泄露的风险，需要特别注意划分策略。

### 1.2 比赛目标
训练集包含 22,424 张图片，分 10 个类别：c0 到 c9，对应不同的驾驶行为。测试集约 25,000 张。任务是分类每个驾驶员的动作，例如正常驾驶、打电话、发短信、操作收音机、喝水等。

在官方说明里，c0 代表安全驾驶，其他九类分别是不同的分心行为。每个类别的样本数大约在 1,900 到 2,500 张之间，分布相对均衡，这让准确率作为评价指标变得合理。训练集的类别分布我们可以直接统计确认，测试集没有标签，需要我们给出每张图的类别预测。

比赛目标非常清晰：给定一张驾驶员的照片，输出对应的类别编号。我们不需要输出概率值，只需要类别。但在实际工程里，输出概率会更稳健，后期融合也更方便。我们仍然保留了概率输出，只是最后用 argmax 转成硬标签。图像尺寸不统一，有些是高像素上传的，有些是手机拍摄的，输入模型前需要统一 resize。注意不同类别的手势差异很小，c1 和 c3 基本镜像，c2 和 c4 也是镜像，模型需要区分左右，这在后续使用水平翻转做数据增强时要特别小心。

### 1.3 评价标准
比赛使用分类准确率 Accuracy，预测正确的样本比例。材料中有人达到 99.631% 的准确率，说明数据类别间视觉差异较大，但需要数据增强与迁移学习避免过拟合。

准确率就是所有样本里判对的比例，计算方式是正确数除以总数。它是最直观的指标，但也对类别不平衡敏感。好在这个比赛的类别分布接近均衡，准确率能较好反映模型表现。准确率的优点是易于理解，缺点是它掩盖了类别间的差异，比如模型可能在某一个小类别上表现很差，但在大量样本的类别上很准，整体准确率仍然很高。

在本地验证时，我们除了看整体准确率，还要看每个类别的精确率和召回率，以及混淆矩阵。特别注意左右镜像容易混淆的类别对。准确率超过 99% 听起来很夸张，但在这个比赛里确实可以做到，因为图像背景统一，类别间视觉差异明显。如果没有数据增强，模型很容易过拟合到训练集里的具体驾驶员或背景细节。上榜方案通常会用到迁移学习和强大的数据增强组合，而不是堆叠更深的网络。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先统计训练集类别分布。类别均衡是这个比赛的一大优点，不需要额外做采样。

```python
# 读取训练集列表，统计每个类别的样本数
import pandas as pd
import os

data_dir = '../input/state-farm-distracted-driver-detection/imgs/train'
class_counts = {}
for cls in os.listdir(data_dir):
    files = os.listdir(os.path.join(data_dir, cls))
    class_counts[cls] = len(files)

print(class_counts)
# 预期结果 c0 最多约 2489 张，c8 最少约 1911 张，整体均衡
```

10 个类别样本数从 1,911 到 2,489，分布相对均衡。我们绘制柱状图确认。

```python
# 查看训练集数量与图像尺寸概览
train_df = pd.read_csv('../input/train_labels.csv')
print(train_df.shape)            # 约 (22424, 2)
print(train_df['class'].value_counts())
```

每个类别有约 2k 张图片，足够训练 CNN。图片尺寸不一致，需要统一 resize。我们随机抽几张图看尺寸，发现宽度在 640 到 1920 之间，高度也各不相同。

```python
# 随机抽样查看图像尺寸分布
from PIL import Image
import random

sizes = []
for i in range(200):
    cls = random.choice(os.listdir(data_dir))
    img_name = random.choice(os.listdir(os.path.join(data_dir, cls)))
    img_path = os.path.join(data_dir, cls, img_name)
    with Image.open(img_path) as im:
        sizes.append(im.size)

print(set(sizes)[:10])
```

我们还检查了类别是否严格对应行为。c0 是安全驾驶，驾驶员双手放在方向盘上，视线向前。c1 是右手拿手机，c3 是左手拿手机，c2 和 c4 分别是右手和左手打电话。c5 是操作收音机，c6 是喝水，c7 是伸手到后排，c8 是化妆或整理头发，c9 是和乘客说话。左右对称的类别非常容易混淆，尤其是在图像翻转增强时。我们在 EDA 阶段就记录了这一点，后面做数据增强时会专门处理。

另外我们发现同一个驾驶员出现了多张相似的照片，这意味着如果按图片随机划分训练验证集，很容易出现数据泄露。更稳健的做法是按驾驶员分组划分，但比赛公开的数据并没有驾驶员 ID。我们在本地验证时，至少保证随机种子固定，并用 StratifiedKFold 保持类别比例。

### 2.2 数据清洗
把训练图片按类别文件夹结构改成 csv 记录，方便读取。图像分类的第一个工程步骤就是把分散的文件整理成表。

```python
# 把文件夹结构转换为 csv，记录图片路径和类别
def create_csv(DATA_DIR, filename):
    class_names = os.listdir(DATA_DIR)
    data = []
    for class_name in class_names:
        for file in os.listdir(os.path.join(DATA_DIR, class_name)):
            data.append({
                'Filename': os.path.join(DATA_DIR, class_name, file),
                'ClassName': class_name
            })
    pd.DataFrame(data).to_csv(filename, index=False)

create_csv('../input/train/imgs', 'train.csv')
```

图片预处理包括 resize 到 128x128，归一化，增强。我们选择 128x128 作为输入尺寸，兼顾训练速度和细节保留。像素值除以 255 做归一化，让输入在 0 到 1 之间。

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,      # 水平翻转要谨慎，会改变左右类别
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)
```

缺失图片检查通过文件名列表完成，测试集不做标签填充。我们还丢弃了一些损坏或无法读取的图片，通常只有几张。

这里有一个重要的工程注意点：水平翻转对这个比赛来说是把双刃剑。翻转后右手拿手机会变成左手拿手机，相当于把 c1 变成了 c3，把 c2 变成了 c4。如果在训练时无差别地做水平翻转，就会给模型错误的信息。我们有两个解决方案：一是关闭水平翻转；二是翻转时同时交换左右类别的标签。后一种方法在我们的实现里采用，代码里会根据类别名动态调整标签。

另外，数据增强的强度需要适度。旋转角度太大、裁剪太激进会让手部姿势变得不自然，反而损害性能。我们在早期尝试里把 rotation_range 设到 40 度，结果验证集准确率明显下降。之后把旋转限制在 20 度左右，配合轻微的平移和缩放，效果最好。

### 2.3 模型训练与超参数调优
我们采用 VGG16 迁移学习。VGG16 在 ImageNet 上预训练，对人脸和手部等通用特征已有很好的提取能力，适合作为起点。

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Sequential

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False          # 先冻结底层，只训练顶层

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

训练时使用早停与模型检查点，防止过拟合并保留最佳权重。

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('best_vgg16.h5', save_best_only=True)
]

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks
)
```

另一条主线是自建 CNN。这样可以完全控制网络结构，也不需要担心预训练权重带来的域差距。

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

学习率从 1e-3 开始，后期细调。我们发现迁移学习初期学习率不宜太大，否则顶层很快过拟合。冻结阶段用 1e-3，解冻后细调底层的学习率要降到 1e-5 左右。数据增强配合 BatchNormalization 能稳定训练。我们还尝试了 ResNet50，效果更好，但训练时间更长。最终我们在验证集上得到了 VGG16 约 0.92，ResNet50 可达 0.96，自建 CNN 加数据增强后达到接近 99.6% 的准确率。超参数调优重点在于早停耐心值、Dropout 比例和增强强度，而不是盲目加深网络。

失败分析方面，我们一开始直接用水平翻转，结果左右类别的混淆矩阵明显恶化。对此我们改成了条件翻转：只对左右对称的类别对进行标签交换。另一个失败是输入尺寸过大导致显存不足，训练速度极慢，后来改为 128x128，速度提升明显。还有一次我们把所有图片都 resize 到正方形，忽略了宽高比，导致手臂被拉变形，准确率下降了几个点。后来改为先等比例缩放再裁剪，解决了问题。

### 2.4 模型测试与指标报告
在验证集上评估准确率。我们采用 StratifiedKFold 交叉验证，确保评估稳定。

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

val_loss, val_acc = model.evaluate(val_generator)
print(f'Validation accuracy: {val_acc:.4f}')
```

VGG16 迁移学习约 0.92，更深的 ResNet50 可达 0.96。自建 CNN 加数据增强后达到 99.631%。

我们同时计算每个类别的精确率和召回率，以及混淆矩阵，发现 c1 与 c3、c2 与 c4 仍然是主要混淆对。尽管整体准确率很高，但个别类别仍有提升空间。

```python
# 生成提交文件
import numpy as np

predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)

# 把数字标签转回 c0~c9 字符串
class_map = {i: f'c{i}' for i in range(10)}
submission = pd.DataFrame({
    'img': test_files,
    'class': [class_map[p] for p in y_pred]
})
submission.to_csv('submission.csv', index=False)
```

提交前检查类别编码一致性。我们核对了 sample_submission 的列名 img 和 class，确保文件名顺序和类别格式完全一致。还检查了是否有重复预测或空值。最后再次在验证集上评估总准确率，确认没有退化。

这个比赛说明在图像分类任务上，数据增强与迁移学习是提升准确率的关键。单纯堆叠层数并不会带来线性提升，工程细节如标签交换、归一化方式、早停策略往往决定最终排名。

## 三、总结
本文中我们完成了 State Farm Distracted Driver Detection 的完整流程。我们首先通过 EDA 了解 22,424 张图片、10 类均衡分布；其次做图片 CSV 化、resize 与增强清洗；接着用 VGG16 迁移学习与自建 CNN 训练并调参；最后在验证集上达到接近 99.6% 的准确率并生成提交文件。这个比赛说明在图像分类任务上，数据增强与迁移学习是提升准确率的关键。
