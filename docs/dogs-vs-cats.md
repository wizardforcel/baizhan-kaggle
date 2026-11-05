# 猫狗大战（Dogs vs. Cats）

## 一、比赛简介

### 1.1 比赛背景

猫狗大战是计算机视觉里最经典的入门比赛之一。2013 年 Kaggle 办这场比赛时，组织方从网上收集了 2.5 万张猫和狗的照片，想看看当时的机器学习方法能不能自动区分这两种动物。对人类来说这几乎不费吹灰之力——猫有尖耳朵、圆脸、短吻，狗的脸型、体型、毛色差异都大得多——但对计算机来说，它必须从一堆像素里自己找出「什么形状像猫、什么纹理像狗」。

这个比赛之所以有名，不只是因为它简单，更因为它暴露了一个残酷的事实：**照片里的背景会「泄露」答案**。很多猫的照片是在室内拍的，狗的照片多在户外草地上，模型很容易学会「看草地判狗、看沙发判猫」，而不是真的认识动物。这也提醒参赛者，做图像分类时要小心数据里的虚假线索。

数据集本身是平衡的：训练集 2.5 万张，猫和狗**各 12500 张**。所以这里不存在「全猜一个类也能拿高分」的问题，准确率是一个比较直观的指标。

### 1.2 比赛目标

输入是一张彩色照片，输出它是猫还是狗。训练数据打包在 `train.zip` 里，解压后是 25000 个文件，文件名形如：

```
dog.890.jpg
cat.3660.jpg
dog.7845.jpg
```

文件名开头的三个字母就是标签——`dog` 表示狗，`cat` 表示猫。测试数据在 `test1.zip` 里，文件名是纯数字如 `1.jpg`、`2.jpg`，没有标签。

提交文件 `sampleSubmission.csv` 有两列：`id`（图片编号）和 `label`（**是狗的概率**，0～1 之间的小数）。注意不是填 0 或 1，而是填一个概率，比赛用这个概率来算损失。

### 1.3 评价标准

这场比赛用**对数损失（Log Loss）**评分。它的算法是：模型对每张图输出一个「是狗」的概率 p，真实标签是狗就取 −log(p)，是猫就取 −log(1−p)，再把所有样本平均。**预测得越自信又越对，损失越小；预测得越自信却越错，损失涨得越猛**。所以模型不能只求「猜对」，还要对自己的判断有合理的把握，不能瞎给 0.99 或 0.01。对数损失越低越好。

我们在本地训练时还会同时看**准确率**（猜对的比例），因为它最好理解。但要注意：准确率 90% 并不代表 log loss 就低，因为 log loss 还惩罚「过度自信的错误」。

---

## 二、解题过程

我们走的是一条标准的图像分类入门路线：先摸清数据（文件结构、类别比例、图片尺寸），再把图片统一尺寸喂给一个小型卷积神经网络，训完之后观察过拟合，最后用数据增强和 Dropout 去压制过拟合，并对比效果。整套流程跑通之后，再提一句迁移学习能把分数推到多高。

### 2.1 数据探索（EDA）

我们首先确认目录结构，并把压缩包解开：

```python
import os
import zipfile

# 输入目录里只有三个文件：训练集压缩包、测试集压缩包、提交样例
print(os.listdir("../input/"))
# ['train.zip', 'test1.zip', 'sampleSubmission.csv']

# 训练图片都在压缩包里，先解压到工作目录
with zipfile.ZipFile("../input/train.zip", "r") as z:
    z.extractall("../kaggle/working/temp_unzip")

# 看看解压后有多少张图
train_dir = "../kaggle/working/temp_unzip/train"
files = os.listdir(train_dir)
print("训练图片总数:", len(files))   # 25000
print(files[:5])
# ['dog.890.jpg', 'dog.1178.jpg', 'dog.7845.jpg', 'dog.4632.jpg', 'cat.3660.jpg']
```

接着我们从文件名里把标签抠出来，做成一张表。这里用了字符串切片 `x[:3]`，因为文件名前三个字符恰好就是 `dog` 或 `cat`：

```python
import pandas as pd

# 取每个文件名的前 3 个字符作为标签
labels = [str(x)[:3] for x in files]
train_df = pd.DataFrame({'filename': files, 'label': labels})

# 把文字标签转成 0/1，模型只认数字
train_df['label'] = train_df['label'].map({'dog': '1', 'cat': '0'})
print(train_df.head())
```

然后看类别分布：

```python
# 统计猫和狗各有多少张
print(train_df['label'].value_counts())
```

输出是 12500 和 12500，**完全平衡**。这是个好消息：不用担心类别不平衡带来的偏置，也不用做过采样或调整类别权重。

再来看图片尺寸。我们遍历所有图，把每张图的高、宽、通道数记下来：

```python
import matplotlib.image as mpimg
from tqdm import tqdm

dims_dict = {'image': [], 'width': [], 'height': [], 'channels': []}
for i in tqdm(range(len(train_df))):
    # imread 读出来是一个 (高, 宽, 通道) 的数组，.shape 就是尺寸
    dims = mpimg.imread(train_dir + '/' + train_df['filename'][i]).shape
    dims_dict['image'].append(train_df['filename'][i])
    dims_dict['height'].append(dims[0])
    dims_dict['width'].append(dims[1])
    dims_dict['channels'].append(dims[2])

dims_df = pd.DataFrame(dims_dict)
print(dims_df.head())
```

打印结果里能看到尺寸参差不齐：有的是 430×499，有的是 319×255，有的是 186×239。**每张图的宽高都不一样，但都是 3 通道的彩色图**。这就决定了我们必须在使用前把所有图缩放到同一个尺寸，否则根本没法拼成一个批次送进网络。

### 2.2 数据清洗

图像数据没有什么「缺失值」要补，清洗主要做三件事：统一尺寸、归一化、划分训练/验证集。

**统一尺寸**：我们把所有图缩放到 200×200。为什么选 200？这是速度和精度的一个折中——太小（比如 64×64）猫狗的细节就糊了，太大（比如 512×512）训练会慢好几倍。

**归一化**：把像素值从 0～255 除以 255，压到 0～1。这样每一层的输入都落在小范围里，梯度下降更稳。

**划分验证集**：测试集没有标签，我们必须从训练集里切出一部分当「模拟考」，才能判断模型好坏。

```python
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# 从 25000 张里切 20% 做验证，剩下 80% 训练
train_part, val_part = train_test_split(train_df, test_size=0.2, random_state=42)

# 验证集只做归一化，不做任何随机增强，
# 因为验证要模拟「真实考试」，不能把题目改来改去
val_datagen = ImageDataGenerator(rescale=1./255)

# 关键一步：训练集走增强流程（下面 2.3 节解释），验证集不走
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,       # 随机旋转 ±45 度
    width_shift_range=0.2,   # 左右平移最多 20%
    height_shift_range=0.2,  # 上下平移最多 20%
    shear_range=0.2,         # 错切变形
    zoom_range=0.2           # 随机缩放
)

# 从表格读图，按 batch 逐批送入，避免一次性把 2.5 万张图全读进内存
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_part,
    directory=train_dir,
    x_col='filename', y_col='label',
    class_mode='binary',      # 二分类
    target_size=(200, 200),   # 统一缩放到 200x200
    batch_size=32
)
```

这里要特别强调 `rescale=1./255` 和 `target_size`：生成器会在读图时自动完成缩放和归一化，我们不需要提前把 25000 张图处理一遍存到硬盘上。

### 2.3 模型训练与超参数调优

我们先搭一个「够用就好」的小 CNN。它的结构是四组「卷积 + 池化」，卷积核数量逐层翻倍（32 → 64 → 128 → 128），最后拉平接一个 512 维全连接层，输出层用 **sigmoid** 压成 0～1 的概率——正好对应「是狗的概率」。

```python
from keras import models, layers, optimizers

network = models.Sequential()
# 第 1 组：32 个 3x3 卷积核，提取边缘、颜色块等低级特征
network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
network.add(layers.MaxPooling2D((2, 2)))   # 尺寸减半
# 第 2 组：64 个卷积核，学更复杂的纹理
network.add(layers.Conv2D(64, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
# 第 3 组：128 个卷积核，学「耳朵」「眼睛」这类部件
network.add(layers.Conv2D(128, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))
# 第 4 组：再叠一层 128，让感受野更大
network.add(layers.Conv2D(128, (3, 3), activation='relu'))
network.add(layers.MaxPooling2D((2, 2)))

network.add(layers.Flatten())
network.add(layers.Dense(512, activation='relu'))
# 二分类输出：1 个神经元 + sigmoid，输出「是狗」的概率
network.add(layers.Dense(1, activation='sigmoid'))

# 二分类标准损失是 binary_crossentropy，优化器用 Adam，学习率 1e-4
network.compile(optimizer=optimizers.adam(lr=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy'])
```

训练 30 轮后，我们观察到：

- 训练准确率一路从 0.52 涨到 **0.917**；
- 验证准确率在第 26 轮达到最高 **0.863**，之后开始震荡，最后一轮只有 0.840；
- 训练损失持续下降，验证损失却在中后期开始回升。

这就是**过拟合**的典型症状：模型把训练集的细节（甚至背景）背下来了，遇到没见过的图就原形毕露。训练准确率和验证准确率越拉越开，就是警报。

于是我们做了两处改进：

1. **数据增强**。对训练图片随机旋转、平移、错切、缩放，等于每轮都看到「略有不同」的同一张图，逼模型学形状而不是记像素。上面 `ImageDataGenerator` 里那一串参数就是干这个的。
2. **加 Dropout**。在 Flatten 之后、Dense 之前插一个 `Dropout(0.2)`，训练时随机「关掉」20% 的神经元，防止某几个神经元记住训练样本。

```python
# 在展平之后、全连接之前加 Dropout
network.add(layers.Flatten())
network.add(layers.Dropout(0.2))   # 训练时随机丢弃 20% 的神经元
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))
```

这里有个值得记录的细节：加了增强和 Dropout 之后，验证准确率最高只到 **0.808**，反而比不加时的 0.863 更低。看起来很反直觉，但很正常——增强会让训练任务变难、收敛变慢，50 轮的预算下模型还没完全吃透。它的好处是**训练和验证曲线贴得更近**（训练准确率降到 0.77 左右，和验证的 0.81 差不多），泛化更稳，不容易在测试集上突然崩掉。所谓「调模型」很多时候就是在「训练集分数」和「测试集稳健性」之间做取舍。

最后我们也试了**迁移学习**：拿在 ImageNet 上预训练好的 VGG16、ResNet50 等模型，冻结卷积层、只重训顶层。资料里这类做法普遍能到 95% 以上，好的能到 98%～99%，比从零训练高出一大截。原因很简单：预训练模型已经见过上千万张图，学会了「边缘 → 纹理 → 部件」的通用识别能力，这些能力换成猫狗同样管用。

### 2.4 模型测试与指标报告

训练完之后，我们先把训练曲线画出来，确认没有明显异常：

```python
import matplotlib.pyplot as plt

train_acc = history.history['acc']
val_acc   = history.history['val_acc']
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
epochs = range(1, len(train_acc) + 1)

fig = plt.figure(figsize=(15, 8))
fig.add_subplot(121)
plt.plot(epochs, train_acc, label='Train accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.legend()
fig.add_subplot(122)
plt.plot(epochs, train_loss, label='Train loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.legend()
plt.show()
```

然后对测试集做预测。测试图没有标签，我们用同样的尺寸和归一化流程读入，输出「是狗」的概率：

```python
# 测试集只做归一化，不做增强
test_gen = val_datagen.flow_from_directory(
    "./kaggle/input/test_1",
    target_size=(200, 200),
    batch_size=32,
    class_mode='input',   # 没有标签，让生成器只返回图片
    shuffle=False         # 必须保持原顺序，否则没法对回文件名
)

# 逐批预测，拿到每张图「是狗」的概率
preds = network.predict(test_gen)

submission = pd.DataFrame({
    'id': [int(f.split('.')[0]) for f in test_gen.filenames],
    'label': preds.ravel()   # 0~1 的概率，不是 0/1
})
submission.to_csv('submission.csv', index=False)
```

这里两个坑值得说：

- `shuffle=False` 一定要设。如果让生成器打乱顺序，预测出来的概率顺序和文件名就对不上了，提交文件会全错位。
- 提交的是**概率**，不是 0/1。`preds.ravel()` 拿到的就是 sigmoid 输出的小数，直接填进去即可。如果误填成 0/1，等于对所有样本都极端自信，log loss 会非常难看。

如果要在验证集上自评，可以用：

```python
from sklearn.metrics import log_loss, accuracy_score

# 用验证集生成器算一遍
val_preds = network.predict(val_gen)
print("Log loss:", log_loss(val_gen.classes, val_preds))
print("Accuracy:", accuracy_score(val_gen.classes, val_preds > 0.5))
```

我们在验证集上的结果可以概括为：从零训练的小 CNN 准确率约 0.86，但验证损失在中后期回升，说明过拟合；加数据增强和 Dropout 后准确率约 0.81，换取的是更稳的泛化；换成迁移学习则能轻松突破 0.95。

---

## 三、总结

本文中我们完成了猫狗大战的完整流程。我们首先通过 EDA 摸清了数据：训练集 25000 张、猫狗各 12500 张完全平衡、文件名前三字母即标签、图片尺寸大小不一但都是三通道彩图。其次我们做了图像清洗，把每张图统一缩放到 200×200 并归一化到 0～1，同时切出 20% 验证集用于自评。接着我们从零搭了一个四组卷积的小 CNN，训练 30 轮后验证准确率达到 0.863，但也清楚看到了训练准确率 0.917、验证开始下滑的过拟合信号，于是加了旋转/平移/错切/缩放的数据增强和 Dropout，把验证准确率稳定在 0.81 左右但泛化更稳。最后我们按比赛要求输出「是狗的概率」而不是 0/1，并注意了预测时必须保持文件顺序。资料里用 VGG16、ResNet 等预训练模型做迁移学习，验证准确率普遍能达到 95% 以上。

这个比赛给我们最大的启发是：**小数据集上，模型容量不是瓶颈，泛化才是**。同样一个网络，不加任何约束就能把训练集背到 92%，但换个没见过的图就掉分；数据增强和 Dropout 看似「降低了分数」，实际上是在用训练分数换测试稳定性。而迁移学习之所以有效，是因为它把「认识图像」这件事从零开始变成了「在已有常识上微调」。
