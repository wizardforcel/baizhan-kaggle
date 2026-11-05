# NOAA 露脊鲸个体识别（Right Whale Recognition）

## 一、比赛简介

### 1.1 比赛背景

北大西洋露脊鲸（North Atlantic right whale）是世界上最濒危的鲸类之一。研究人员需要长期跟踪每一头鲸鱼的数量、迁徙和生存状况，而跟踪的前提是**认出它是谁**。

露脊鲸的头部有一块块粗糙的白色角质斑块，叫「胼胝」（callosity）。这些斑块的位置和形状因个体而异，就像人的指纹，研究人员正是靠俯拍照片上的胼胝图案来给每头鲸鱼编号。传统做法是请专家一张张看照片、手工比对，费时费力，也难以应对每年积累的海量航拍图。

这个比赛由 NOAA（美国国家海洋和大气管理局）发起，希望用计算机视觉自动完成「看照片认鲸鱼」这件事：给一张航拍照片，判断它是哪一头已知的露脊鲸个体。对入门者来说，这是一个很典型的**细粒度图像分类**问题——类别之间的差别不是「有没有角」「几条腿」这种大特征，而是头顶花纹的细微差别。

### 1.2 比赛目标

输入是一张从飞机上俯拍的露脊鲸照片（RGB 图像），输出这头鲸鱼对应的个体编号。训练集是带有专家标注的航拍图，每张图对应一头已知的鲸鱼；测试集只有图片，我们要为每张测试图片预测它的个体类别。

比赛材料里给出的 notebook 只是一个极简的「读数据」脚手架，它做的事情是遍历 `/kaggle/input` 目录、把所有输入文件路径打印出来，用于确认数据挂载成功：

```python
import os

# 遍历输入目录，把每个文件的完整路径打印出来
# 目的：确认比赛数据已经正确挂载，并看清目录结构
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

这段代码本身没有建模，但它是所有 Kaggle 项目的第一步：**先搞清楚数据在哪、长什么样**。后面我们围绕它补齐了读取图片、划分验证集和训练分类器的流程。

### 1.3 评价标准

比赛使用**多分类对数损失（multi-class log loss）**作为评分指标，越低越好。它衡量的不是「猜对没有」，而是「给正确答案分配了多高的概率」：

```
logloss = - (1/N) * Σ log( p(真实类别) )
```

如果模型给真实类别分配的概率接近 1，损失就很小；如果只给 0.1，损失就会明显变大。因此光「猜对」不够，还要**对自己的判断有合理的置信度**。这也是为什么细粒度分类里常用带温度缩放的概率校准——过度自信会被 log loss 狠狠惩罚。

> 说明：提供的材料只有一份读取目录的脚手架，没有完整的建模代码。下面的方案是根据该比赛的公开任务描述和图像分类的通用做法整理的思路，代码为示意性的骨架，不是材料里的原始实现。

---

## 二、解题过程

我们的思路分四步：先把图片读成统一尺寸的张量，再把数据集按个体分层切分，然后用卷积网络（或迁移学习）提取特征分类，最后用 log loss 评估并生成提交文件。

### 2.1 数据探索（EDA）

我们首先在脚手架的基础上，把每张图片的路径和它的标签整理成一张表，并统计每个个体有多少张照片：

```python
import pandas as pd
from pathlib import Path

# 假设训练图片放在 train/ 下，标注文件是 train.csv
train_dir = Path('/kaggle/input/noaa-right-whale-recognition/train')
labels = pd.read_csv('/kaggle/input/noaa-right-whale-recognition/train.csv')

# 统计每个鲸鱼个体有多少张训练图
counts = labels['whale_id'].value_counts()
print('个体总数：', counts.shape[0])
print('图片最多的个体：', counts.head())
print('图片最少的个体：', counts.tail())
```

细粒度识别的一个典型特点是**长尾分布**：少数「明星鲸鱼」被拍到的次数很多，大多数个体只有寥寥几张照片。这直接影响我们后面的划分策略——不能简单地随机切分，否则验证集里可能出现训练时完全没见过的个体。

我们还抽查了图片的尺寸和拍摄角度。航拍照片的鲸鱼朝向、大小、光照差异很大，有的只露出背部，有的被浪花遮住一半，这些都是模型要克服的困难。

### 2.2 数据清洗

图像数据没有表格里的缺失值，但有几个必须处理的点：

**第一，统一尺寸与归一化。** 不同照片分辨率不一，我们统一缩放到固定大小（例如 224×224），并把像素值从 0~255 压到 0~1：

```python
import tensorflow as tf

IMG_SIZE = (224, 224)

def load_image(path):
    """读一张图，缩放到统一尺寸并归一化"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img / 255.0   # 归一化到 [0,1]
```

**第二，按个体划分训练/验证集。** 这是本比赛最关键的一步。同一头鲸鱼可能有多张照片，如果随机切分，验证集里会出现训练时见过的个体，分数会虚高。正确做法是**按 `whale_id` 分组切分**：同一头鲸鱼的所有照片只能出现在训练侧或验证侧：

```python
from sklearn.model_selection import GroupShuffleSplit

# 以 whale_id 为组切分，保证同一个个体的照片不跨集合
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(labels, groups=labels['whale_id']))
```

**第三，剔除无效图片。** 有些航拍图里鲸鱼几乎不可见（只有水花或尾巴），这些样本对训练帮助不大，可以在统计后过滤掉；测试集则必须全部预测，不能丢。

### 2.3 模型训练与超参数调优

图像分类的主流做法是卷积神经网络（CNN）。我们从零搭一个小型 CNN 作为基线，再对比迁移学习的效果。

**基线 CNN 的骨架：**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    # 三组「卷积 + 池化」，逐步提取从边缘到纹理的特征
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),          # 抑制过拟合
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax'),  # 每个个体一个输出
])
```

编译时用 Adam 优化器和分类交叉熵：

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
```

**迁移学习是更实际的路线。** 露脊鲸的训练图只有几千张，从零训练很容易过拟合；而 ImageNet 上预训练的网络已经学会了识别边缘、纹理和形状，把它拿来微调往往能明显提升。做法是冻结卷积基、只训练最后的分类头，然后再解冻顶部几层做小学习率微调：

```python
base = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base.trainable = False   # 第一阶段先冻结卷积基

model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

# 小学习率微调，避免把预训练权重破坏掉
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

超参数方面，我们主要调整三个：输入分辨率（224 或 320，花纹越细越吃分辨率）、学习率（分类头用 1e-3，微调用 1e-4）、以及数据增强的强度（随机裁剪、水平翻转、亮度扰动）。因为类别多、样本少，强增强和早停几乎是必需的。

### 2.4 模型测试与指标报告

评估时用比赛同款的多分类对数损失，同时看准确率作为辅助：

```python
from sklearn.metrics import log_loss, accuracy_score
import numpy as np

# 在按个体划分出来的验证集上预测概率
probs = model.predict(val_images)
loss = log_loss(val_labels, probs, labels=list(range(num_classes)))
acc = accuracy_score(val_labels, np.argmax(probs, axis=1))
print(f'验证集 log loss: {loss:.4f}, 准确率: {acc:.4f}')
```

由于 log loss 对「过度自信的错误」惩罚很重，我们还会检查模型输出的最大概率分布：如果大量样本都以 0.99 的概率押一个类别，一旦押错损失会爆掉，说明需要做温度缩放或标签平滑。

最后对测试集逐张预测，输出每张图片概率最高的个体编号：

```python
submission = pd.DataFrame({
    'Image': test_files,
    'whale_id': [class_names[i] for i in np.argmax(test_probs, axis=1)],
})
submission.to_csv('submission.csv', index=False)
```

---

## 三、总结

本文中我们整理了 NOAA 露脊鲸个体识别比赛的完整思路。我们首先根据材料里的脚手架确认了数据目录，用 EDA 统计了个体数量和长尾分布；其次做了图像统一缩放与归一化，并特别强调**按个体分组划分验证集**，避免同一头鲸鱼跨集合造成泄漏；接着我们用卷积网络和迁移学习两条路线搭建分类器，配合数据增强、Dropout 和早停控制过拟合；最后用多分类对数损失评估，并生成提交文件。

这个比赛给我们的最大启发是：**细粒度分类的难点不在「分类器」，而在「数据和验证」**。类别之间的差异只有头顶花纹那么小，样本又少又不均衡，只有把数据划分、增强和概率校准做扎实，模型的能力才谈得上发挥。另外，材料里的 notebook 虽然只有十几行「列目录」的代码，却也提醒我们：任何项目都应该从确认数据开始，这一步偷懒，后面会加倍还回来。