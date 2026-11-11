# 图像分类项目：识别牛、象、马、蜘蛛（Open CV & TF Project 1, Round 6）

## 一、比赛简介

### 1.1 比赛背景

这是一个面向深度学习入门者的**图像分类**练习项目。图像分类要解决的问题很直观：给模型一张图片，让它说出图里是什么。模型能认出的类别，取决于我们喂给它的训练数据。

本项目的训练数据包含四类动物：**奶牛（cow）、大象（elephant）、马（horse）、蜘蛛（spider）**。数据按类别放在不同文件夹里，这种「一个子文件夹一个类别」的目录结构，正好是 TensorFlow/Keras 的 `image_dataset_from_directory` 默认支持的格式，省去了自己写标签映射的麻烦。

比赛分成两个 notebook（Round 6 版本）：

- Part 1「Data Understanding & Pipeline Check」：理解数据、准备数据、先在**小样本**上把整条流水线跑通；
- Part 2「Train an Image Classifier From Scratch」：用完整数据训练模型，要求**不允许使用任何预训练权重**，从零搭建网络。

这种安排是工程上非常值得学习的习惯：先在小数据上验证「读数据→预处理→建模→评估→生成提交」这条链路没有 bug，再投入时间做全量训练。否则几小时的训练可能因为一个路径写错而白费。

### 1.2 比赛目标

输入是一张动物图片（RGB），输出它属于四类中的哪一类。具体任务：

- Part 1（25 分）：完成数据探索、数据准备、训练配置，并展示若干容易出错的地方；
- Part 2（75 分）：从零设计并训练一个卷积神经网络，在公开测试集上达到 **≥ 80% 的准确率**，画出混淆矩阵，并生成 Kaggle 提交文件。

提交文件是 CSV，每行包含图片文件名和预测类别。测试集没有标签，最终分数由 Kaggle 公开/私有排行榜给出。

### 1.3 评价标准

评价指标是**分类准确率（accuracy）**，即预测正确的图片数占全部图片的比例，越高越好。项目要求公开测试集准确率达到 80% 以上。由于四个类别大致均衡，准确率是一个直观且合理的指标。比赛还要求提交混淆矩阵，用来观察模型是否在某一类上特别容易混淆（比如把马认成牛）。

---

## 二、解题过程

我们的主线是「先用小样本跑通流水线，再从零训练一个卷积网络，最后评估并提交」。下面按四步展开。

### 2.1 数据探索（EDA）

我们首先固定随机种子，保证每次运行结果一致，并检查运行环境里可用的 GPU 数量：

```python
import os, random
import numpy as np
import tensorflow as tf

def set_seeds(seed=42):
    """把所有随机源都固定住，让实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"   # 强制 TensorFlow 用确定性算子

    # 打开显存按需增长，避免一上来就把 GPU 显存占满
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

set_seeds(42)

# 多 GPU 环境下用镜像策略，把模型复制到每块卡上同步训练
strategy = tf.distribute.MirroredStrategy()
print('可用设备数：', strategy.num_replicas_in_sync)
```

接着我们看看数据目录长什么样：每个类别一个文件夹，文件夹名就是类别名。我们统计每个类别的图片数，确认数据是否均衡：

```python
import pathlib

data_root = pathlib.Path('/kaggle/input/competitions/'
                         'open-cv-tf-project-1-image-classification-round-6/dataset')

# 列出四个类别文件夹，并数一数每个文件夹里有多少张图
for class_dir in sorted(data_root.iterdir()):
    if class_dir.is_dir():
        n_images = len(list(class_dir.glob('*')))
        print(f'{class_dir.name}: {n_images} 张图片')
```

如果四类数量接近，说明没有明显的类别不平衡，用准确率评估是合适的。

我们还随机抽几张图片显示出来，看看尺寸和内容。图像分类项目里一个常见的坑是：**训练图片尺寸不统一**。Keras 的 `image_dataset_from_directory` 会在读取时统一缩放，但我们仍要确认目标尺寸（例如 224×224 或 128×256）不会把动物拉变形。

```python
import matplotlib.pyplot as plt

# 取一个 batch 的图片和标签，可视化检查
ds = tf.keras.utils.image_dataset_from_directory(
    data_root, image_size=(224, 224), batch_size=9, shuffle=True, seed=42)

for images, labels in ds.take(1):
    plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(int(labels[i]))
        plt.axis('off')
    plt.show()
```

这里我们看到标签是整数 0~3，对应文件夹按字母序排列的四个类别。实际写代码时，我们会把这个顺序记录下来，提交时把整数映射回类别名。

### 2.2 数据清洗

图像数据没有传统意义上的「缺失值」，但有几处必须处理：

**第一，统一尺寸。** 原始图片大小不一，我们在读取时就统一缩放到固定尺寸。Part 1 的配置里写的是 `(128, 256, 3)`，Part 2 中为了配合网络结构，我们使用 `(224, 224, 3)`：

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 训练集：打乱顺序，避免同一类的图片连续出现
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_root / 'train',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)

# 验证集：不打乱，方便复现评估结果
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_root / 'val',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
```

**第二，把像素值缩放到 [0,1]。** 原始像素是 0~255 的整数，直接喂进网络会让梯度数值过大、训练不稳。我们用一个 `Rescaling` 层把它压到 0~1：

```python
# Rescaling 层可以嵌进模型里，推理时也会自动做同样的缩放
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
```

**第三，用缓存和预取加速。** 图片从磁盘读出来比较慢，我们把解码后的 batch 缓存在内存里，并让 CPU 提前准备下一批数据：

```python
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

**第四，数据增强。** 训练图片数量有限，我们通过随机翻转、旋转、缩放来「造」更多样本，降低过拟合：

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),   # 水平翻转，动物左右对称
    tf.keras.layers.RandomRotation(0.1),        # 小角度旋转
    tf.keras.layers.RandomZoom(0.1),            # 随机缩放
])
```

注意：数据增强只加在训练集上，验证集和测试集绝不能做随机变换，否则评估结果会不稳定。

### 2.3 模型训练与超参数调优

按照 Part 1 的要求，我们**先用一个小样本把流水线跑通**：从训练集里取几百张图片，搭一个很浅的网络训练 1~2 个 epoch，确认 loss 在下降、预测形状正确，再上全量数据。

```python
# 小样本：只取前 10 个 batch，快速验证整条链路
small_train = train_ds.take(10)
small_val   = val_ds.take(2)

# 一个极简模型，仅仅用来检查数据形状和训练流程
test_model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation='softmax'),  # 四类输出
])
test_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
test_model.fit(small_train, validation_data=small_val, epochs=1)
```

这一步的价值在于「早发现错误」：如果图片形状、标签范围、输出维度有任何不对，这里几秒钟就会报错，而不是等到全量训练几小时后才发现。

确认流水线没问题后，我们搭建真正用于比赛的卷积网络。结构是经典的「卷积 → 池化」堆叠，最后接全连接层和 softmax：

```python
model = tf.keras.Sequential([
    # 输入是 224x224 的 RGB 图，先归一化
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    data_augmentation,

    # 第一组卷积：32 个 3x3 卷积核，提取边缘、颜色等低级特征
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # 第二组：64 个卷积核，提取更复杂的局部结构
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # 第三组：128 个卷积核
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    # Dropout 随机丢弃 50% 的神经元，抑制过拟合
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax'),   # 四分类
])
```

编译时我们用 Adam 优化器和稀疏交叉熵损失。稀疏交叉熵适用于标签是整数（0/1/2/3）而不是 one-hot 向量的情况：

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',  # 标签是整数，用 sparse 版本
    metrics=['accuracy'],
)
```

训练配置方面，Part 1 的模板给出 `BATCH_SIZE = 4 * num_replicas`、`EPOCHS = 2`、`LEARNING_RATE = 0.1` 作为起点；实际训练时我们通常把 epoch 提高到 10~20，并配合两个回调：

```python
callbacks = [
    # 保存验证损失最低的那一版权重，训练结束后用它做预测
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_loss',
        save_best_only=True, verbose=1),
    # 验证损失连续 3 轮不下降就提前停止，省时间也防过拟合
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
)
```

关于超参数调优：我们没有做大规模网格搜索，而是抓住几个关键杠杆——学习率用 1e-3（Adam 的常用默认值），卷积核数量按 32→64→128 递增，Dropout 取 0.5。对入门项目来说，**先把网络结构和数据流水线弄对，再谈调参**，收益更大。

### 2.4 模型测试与指标报告

训练结束后，我们用保存的最佳模型在验证集上评估，并画出 loss/accuracy 曲线和混淆矩阵：

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# 在验证集上评估
val_loss, val_acc = model.evaluate(val_ds)
print(f'验证集 loss: {val_loss:.4f}, 准确率: {val_acc:.4f}')

# 收集所有验证样本的预测结果
y_true, y_pred = [], []
for images, labels in val_ds:
    probs = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(probs, axis=1))

# 画混淆矩阵：对角线越亮说明该类判得越准
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap='Blues')
plt.show()
```

混淆矩阵能直接告诉我们错误集中在哪里。如果「马」和「牛」之间互相混淆得多，说明这两类的外观确实接近，后续可以考虑更强的数据增强或更深的网络。

接着我们对测试集逐张预测，生成提交文件。测试图片按文件名读取，输出每个文件对应的类别名：

```python
import pandas as pd
from PIL import Image

test_dir = data_root / 'test'
rows = []
for img_path in sorted(test_dir.glob('*.jpg')):
    # 用和训练时相同的尺寸读入，并补一个 batch 维度
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    arr = tf.expand_dims(arr, axis=0)

    prob = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(prob))
    rows.append({'filename': img_path.name, 'label': class_names[pred_idx]})

pd.DataFrame(rows).to_csv('submission.csv', index=False)
```

其中 `class_names` 要和我们训练时 `image_dataset_from_directory` 给出的类别顺序一致——这个顺序通常按文件夹名字母排序，即 `['cow', 'elephant', 'horse', 'spider']`。提交前我们还会核对行数是否等于测试图片数、列名是否和样例文件一致。

从 Round 6 的 starter 来看，项目要求公开榜达到 80% 以上。从零训练一个三层卷积网络，配合数据增强和早停，通常可以在验证集上达到接近或超过这个水平；如果分数不够，材料里还提供了另一个方向——**迁移学习**（使用预训练模型），但在 Part 2 中这是被明确禁止的，所以我们把它留作后续练习。

---

## 三、总结

本文中我们完成了图像分类项目 Round 6 的完整流程。我们首先用 EDA 确认了四类动物（牛、象、马、蜘蛛）的数据目录结构、类别均衡性和图片尺寸，并用随机种子与显存增长设置保证了实验可复现。其次我们做了图像数据的必要清洗：统一缩放、像素归一化、缓存预取，并只在训练集上施加随机翻转/旋转/缩放的数据增强。接着我们先用小样本把「读数据→建模→评估」的流水线跑通，再从零搭建了一个三层卷积网络（32/64/128 通道 + 池化 + Dropout + softmax），用 Adam、稀疏交叉熵、ModelCheckpoint 和 EarlyStopping 训练。最后我们在验证集上报告准确率、画出混淆矩阵，并用最佳模型对测试图片逐张预测，生成了符合要求的提交文件。

这个比赛给我们的最大启发是：**先跑通，再跑好**。小样本流水线检查花不了几分钟，却能挡住绝大多数低级错误；而从零训练卷积网络的过程，也让我们真正看清了「卷积提特征、池化降尺寸、全连接做分类」这套组合拳是怎么工作的。等这套基本功扎实了，再去用迁移学习或更大的模型，才不会只是「会调用 API」。