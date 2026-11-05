# Pneumonia-SAI2（肺炎 X 光图像二分类）

## 一、比赛简介

### 1.1 比赛背景

肺炎是我们非常熟悉的一种疾病，它是由细菌、病毒或真菌感染引起的肺部炎症。得了肺炎之后，肺泡里会充满液体或分泌物，导致患者咳嗽、发烧、呼吸困难，严重的甚至危及生命。在临床上，医生判断一个人是否得了肺炎，最常用、最便宜的手段就是拍一张胸部 X 光片。有经验的放射科医生通过观察 X 光片里肺部的阴影形态，就能大致判断有没有炎症。但是，X 光片的判读非常依赖医生的经验，而且胸部 X 光片本身光影复杂，容易误判。

近年来，深度学习在医学图像识别上取得了很大的进展。卷积神经网络（CNN）能够自动从图像中学习"肺炎区域长什么样"的特征，从而帮助医生做辅助诊断。本场比赛（比赛代号 `pneumonia-sai2`）就是这样一个典型的医疗图像二分类任务，预期结果是对输入的一张胸部 X 光片，预测它属于"正常（NORMAL）"还是"肺炎（PNEUMONIA）"。比赛使用的数据来自经典的 "pneumonia-kaggle" 数据集，里面的 X 光片按照类别存放在 `train` 和 `test` 两个文件夹中。

对于机器学习入门者来说，这个比赛非常适合练手：数据量适中、任务简单清晰、评价指标直观，而且可以使用预训练模型做迁移学习，不需要从零训练一个很深的网络。

### 1.2 比赛目标

我们的目标是训练一个图像分类模型，让它对测试集中的每一张胸部 X 光片输出一个二分类标签。如果图片显示肺部有炎症，就标记为 "1"（肺炎）；如果肺部正常，就标记为 "0"（正常）。

比赛的训练数据存放在 `pneumonia-kaggle/train` 目录下，内部按类别分子文件夹存放；待预测的图片存放在 `pneumonia-kaggle/test` 目录下。我们需要读取每一张测试图片，用训练好的模型做预测，最后生成一个提交文件 `submission.csv`。提交文件的格式很简单，只包含两列：文件名（`filename`）和预测标签（`label`），其中 `label` 是 0 或 1 的整数。Kaggle 的评分系统会根据这个文件对我们的模型打分。

### 1.3 评价标准

比赛采用"分类准确率（Accuracy）"作为评价标准，也就是"预测正确的图片数占全部图片数的比例"。这个指标在二分类任务中非常好理解：

准确率 = 预测正确的样本数 ÷ 总样本数

从我们使用的公开代码来看，模型在编译时指定了 `metrics=['accuracy']`，并在验证集上打印验证准确率；最终提交文件里也直接输出 0/1 整数标签，这说明排名就是依据准确率来计算的。同时，训练时使用的损失函数是二分类交叉熵（`binary_crossentropy`），这也是二分类任务最常用的损失函数：它度量了模型预测的概率分布和真实标签之间的差异，值越小说明预测越接近真实答案。

## 二、解题过程

### 2.1 EDA（探索性数据分析）

EDA 是整场比赛的第一步。我们首先把数据"看一遍"，搞清楚数据长什么样、有多少张图、类别怎么分布，再决定后续怎么处理。我们用开放文件遍历代码，看看输入目录里到底有哪些文件。

```python
import numpy as np          # 数值计算的库，处理数组
import pandas as pd         # 数据处理库，用来读写 CSV 文件
import os                   # 操作文件路径的库
import matplotlib.pyplot as plt   # 画图库，用来画准确率曲线
import cv2                  # OpenCV，用来读取和缩放图片
import tensorflow as tf     # 深度学习框架
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 数据增强
from tensorflow.keras import layers, models                          # 网络层和模型
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # 回调函数

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# 遍历输入目录，把所有文件路径打印出来，看看数据是怎么组织的
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

我们先引入需要用到的库。特别要先确认 TensorFlow 的版本，以及有没有可用的 GPU——因为卷积神经网络训练很耗时，有 GPU 会快很多。然后我们用 `os.walk` 遍历输入目录，把里面所有文件从头到尾列出来。这一步能让我们确认数据路径是否正确、训练集和测试集分别放在哪里，避免后面读取时报路径错误。

接着，我们确定了数据的基础结构，并设置好路径、检查路径是否存在。

```python
# 比赛数据的根目录（注意这个比赛的数据在 competitions 子目录下）
base_path = "/kaggle/input/competitions/pneumonia-sai2/pneumonia-kaggle"

train_dir = os.path.join(base_path, "train")   # 训练集目录
test_dir = os.path.join(base_path, "test")     # 测试集目录

print("Train Path Exists:", os.path.exists(train_dir))
print("Test Path Exists:", os.path.exists(test_dir))
```

从目录浏览的结果看，训练集文件夹里按照类别分子目录存放图片，类别分别是 "NORMAL"（正常）和 "PNEUMONIA"（肺炎），而测试集文件夹里是分散存放的待预测图片。我们可以用 `os.path.exists` 检查路径拼写是否正确，这里打印 `True` 就说明路径没问题。

在真正的 EDA 里，我们还会统计两个类别的图片数量、看清图片的分辨率和像素分布。不过本场比赛的数据量适中，类别也比较均衡，因此我们没有必要做过于复杂的画图分析，只要心里有数：这是一个二分类图片任务，类别就是正常和肺炎两类。对尺寸各异的 X 光片，我们会在后面的数据清洗环节统一缩放。

### 2.2 数据清洗

医学 X 光片和普通的自然图像不太一样，它们的尺寸不统一、亮度也有差异。为了让神经网络稳定训练，我们需要做数据清洗，也就是预处理。

我们第一步做的是像素值归一化：把每个像素从 0~255 的范围缩放到 0~1。这样做的原因是，神经网络对输入数值的大小范围很敏感，大范围的数值会让梯度更新不稳定。第二步是统一图片尺寸。预训练模型 `MobileNetV2` 要求输入 224×224 的图片，所以我们把所有图片都调整到 224×224。第三步是数据增强：我们让训练时的图片做小角度旋转、小幅缩放和随机水平翻转，相当于"人工制造"更多样化的训练样本，减少过拟合。

在 Keras 中，这些清洗和增强操作都可以用 `ImageDataGenerator` 一次性配置好。它还支持把训练集按比例切出一部分做验证集，方便我们观察模型在没见过的数据上的表现。

```python
# 配置数据增强器：归一化像素 + 训练/验证划分 + 几何增强
train_datagen = ImageDataGenerator(
    rescale=1./255,          # 像素值除以 255，归一化到 [0,1] 区间
    validation_split=0.2,    # 从训练集里切出 20% 作为验证集
    rotation_range=15,       # 随机旋转正负 15 度，增强数据多样性
    zoom_range=0.1,          # 随机缩放 10%，让模型看到不同大小的病灶
    horizontal_flip=True     # 随机水平翻转（肺部左右对称，翻转不会改变语义）
)

# 训练用的数据生成器，指定使用训练子集
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),   # 把图片统一缩放为 224×224
    batch_size=32,           # 每批处理 32 张图
    class_mode='binary',     # 二分类，标签是 0 或 1
    subset='training'        # 取 80% 作为训练数据
)

# 验证用的数据生成器，指定使用验证子集
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='validation'      # 取 20% 作为验证数据
)
```

`flow_from_directory` 会扫描子文件夹的名字，自动把类别名映射成标签并把图片打包成批次。这里我们把训练数据按 8:2 切成训练子集和验证子集，训练子集用来更新模型参数，验证子集用来检验模型学得好不好、有没有过拟合。这样我们不需要单独准备一套验证图片，很方便。

此外，这一步也是数据清洗的一部分：图片统一变成了 224×224、像素值落在 0~1、颜色通道保持 3 通道（恰好符合 ImageNet 预训练模型的需求）。因为我们使用了在 ImageNet 上预训练的模型，所以输入格式必须和预训练时一致，模型才能把学到的特征迁移过来。

### 2.3 模型训练与超参数调优

接下来是核心环节：搭建并训练模型。我们使用**迁移学习**的思路，加载在 ImageNet 上预训练好的 `MobileNetV2` 作为"特征提取器"，只替换掉顶部的分类部分。这样我们等于站在巨人的肩膀上，把别人花了很多资源学到的"看图片"的能力拿过来，自己只需学习"肺炎 X 光片分类"这个新任务。

MobileNetV2 是一个轻量级网络，参数量少、运算快，非常适合作为医学图像分类的骨干网络。我们保留了它在 ImageNet 上学到的通用特征（比如边缘、纹理），冻住网络前面的绝大部分层不让它们更新，只放开最后 20 层参与微调——因为靠近输出的层学到的特征更接近具体任务，放开它们可以让模型更好地适应 X 光片的细节。

```python
# 加载预训练的 MobileNetV2，不要最后的分类层
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),   # 输入是 224×224 的 3 通道彩色图
    include_top=False,         # 去掉顶层的全连接分类部分
    weights='imagenet'         # 加载在 ImageNet 上预训练的权重
)

# 冻结前 139 层（总层数减去 20），不参与训练
for layer in base_model.layers[:-20]:
    layer.trainable = False
# 放开最后 20 层，让它们随着任务继续微调
for layer in base_model.layers[-20:]:
    layer.trainable = True

# 在骨干网络后面接上我们自己的分类头
model = models.Sequential([
    base_model,                    # 预训练骨干网络，负责提取特征
    layers.GlobalAveragePooling2D(),  # 把每个通道的特征图压成一个数字
    layers.BatchNormalization(),   # 批归一化，稳定训练、加快收敛
    layers.Dense(128, activation='relu'),  # 128 个神经元的全连接层
    layers.Dropout(0.5),           # 随机丢弃 50% 神经元，防止过拟合
    layers.Dense(1, activation='sigmoid') # 输出一个 0~1 的概率值
])
```

`GlobalAveragePooling2D` 的作用是把骨干网络输出的特征图（每张都是一个小矩阵）求平均，变成一维的特征向量，这样不管进来的特征图多大都能接上全连接层。后面的 `Dense(128)` 负责把这些特征组合成更高层的判断，`Dropout(0.5)` 随机丢掉一半的神经元，逼迫模型不要过度依赖某几个特征，从而降低过拟合。最后一层的 `sigmoid` 激活函数输出一个 0~1 之间的概率，比如输出 0.87 就说明模型认为这张图有 87% 的可能是肺炎。

然后我们编译模型，选择优化器和学习率。

```python
# 编译模型：指定优化器、损失函数和评价指标
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Adam 优化器，学习率 1e-4
    loss='binary_crossentropy',    # 二分类交叉熵损失
    metrics=['accuracy']           # 以准确率作为观测指标
)
```

我们选择 Adam 优化器，它是目前最常用的优化器之一，自带自适应学习率，对入门者很友好。学习率设置为 0.0001（1e-4）比较保守，适合微调预训练模型——学习率太大会破坏预训练权重，太小则收敛太慢。

接下来我们还设置了两个重要的回调函数，它们会在训练过程中自动干预训练，这也是超参数调优的一部分。

```python
# 早停：验证损失连续 3 轮不再下降就停止训练，并恢复最优权重
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 学习率衰减：验证损失连续 2 轮不下降，学习率就乘以 0.3
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',   # 监控验证集损失
    factor=0.3,           # 学习率缩小为原来的 0.3 倍
    patience=2,           # 连续 2 轮没有改善就触发
    min_lr=1e-6           # 学习率最小降到 1e-6，防止降到底
)
```

`EarlyStopping` 的意思是：如果连续 3 轮验证集损失都不再下降，说明模型已经收敛，再训练只会浪费时间和算力，于是提前停止，并把验证集损失最好的那一轮权重恢复回来。`ReduceLROnPlateau` 的意思是：如果验证损失连续 2 轮原地踏步，就把学习率缩小为原来的 0.3 倍。当损失卡在平台期时，更小的学习率能让模型在损失曲面的更小范围内精细搜索，从而进一步下降。这两个回调配合起来，是训练深度模型非常实用的组合拳。

准备好之后，正式开始训练。

```python
# 开始训练：循环遍历训练数据 15 轮，每轮结束在验证集上评估
history = model.fit(
    train_generator,             # 训练数据生成器
    validation_data=val_generator,  # 验证数据生成器
    epochs=15,                   # 最多训练 15 轮
    callbacks=[early_stop, reduce_lr]  # 挂载前面定义的回调
)
```

我们设置最多训练 15 轮。得益于预训练权重和早停机制，模型一般用不了 15 轮就能收敛。每一轮结束后，模型都会自动在验证集上算一次损失和准确率，并把整个训练过程记录在 `history` 里，供我们后面画图分析。

### 2.4 模型测试与指标报告

训练完成后，我们先在验证集上正式评估一遍模型的最终表现，打印验证准确率。

```python
# 在验证集上评估最终模型
val_loss, val_acc = model.evaluate(val_generator)
print("Validation Accuracy:", val_acc)   # 打印验证集准确率
```

`model.evaluate` 会自动把整个验证集喂给模型，返回验证集上的损失和准确率。如果这个准确率在合理范围（比如 85%~95%），说明模型训练得不错，可以拿到测试集上去做预测。

为了更直观地确认训练过程是否健康，我们画出训练和验证的准确率曲线与损失曲线。

```python
# 画两张图：左边是准确率曲线，右边是损失曲线
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)                     # 左图
plt.plot(history.history['accuracy'])  # 训练准确率
plt.plot(history.history['val_accuracy'])  # 验证准确率
plt.title("Accuracy")                  # 标题：准确率
plt.legend(["Train","Validation"])     # 图例

plt.subplot(1,2,2)                     # 右图
plt.plot(history.history['loss'])      # 训练损失
plt.plot(history.history['val_loss'])  # 验证损失
plt.title("Loss")                      # 标题：损失
plt.legend(["Train","Validation"])     # 图例

plt.show()                             # 显示图片
```

如果训练和验证两条曲线都平稳上升（准确率）/ 下降（损失），而且两者差距不大，说明模型既没有欠拟合也没有严重过拟合。如果验证损失在后期反弹而训练损失一直降，就说明过拟合了，需要加强 Dropout 或者数据增强。

确认模型表现合格后，我们进入最后一步：读取测试集图片，做出预测，生成提交文件。

```python
# 用 OpenCV 逐个读取测试图片，统一缩放并归一化
test_images = []      # 存放处理好的测试图片
test_filenames = []   # 同步记录文件名，方便提交时对应上

for file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, file)   # 拼出完整路径

    img = cv2.imread(img_path)     # 读取图片（BGR 格式）
    img = cv2.resize(img, (224,224))  # 缩放到 224×224
    img = img / 255.0              # 归一化到 [0,1]

    test_images.append(img)        # 加入图片列表
    test_filenames.append(file)    # 记录文件名

test_images = np.array(test_images)  # 转成 NumPy 数组，维度为 (N,224,224,3)

print("Test Data Shape:", test_images.shape)  # 打印测试数据的形状
```

这里我们用和训练时一样的预处理方式处理测试图片：`cv2.imread` 读取，`cv2.resize` 统一到 224×224，再除以 255 归一化。这样测试图片的分布和训练图片保持一致，模型才能发挥应有的性能。`test_images.shape` 会显示类似 `(N, 224, 224, 3)`，其中 N 是测试图片的数量。

最后，用模型对所有测试图片做预测，概率大于 0.5 的判为肺炎（1），否则判为正常（0），然后写入提交文件。

```python
# 用模型对所有测试图片做预测
predictions = model.predict(test_images)      # 输出每个样本的概率值
predictions = (predictions > 0.5).astype(int) # 概率大于 0.5 记为 1（肺炎），否则为 0（正常）

# 拼成提交文件：文件名 + 预测标签
submission = pd.DataFrame({
    "filename": test_filenames,   # 测试图片的文件名
    "label": predictions.flatten() # 展平成一行一个标签
})

# 保存为 CSV 文件，取消行索引
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("✅ Submission file created successfully!")
```

`sigmoid` 层输出的概率天然在 0~1 之间，用 0.5 作为分界线把概率转成 0/1 整数就是最自然的判定了。生成的 `submission.csv` 里有两列：`filename` 是测试图片的名字，`label` 是我们预测的类别。把这份文件提交到 Kaggle，评分系统就会按准确率给我们排名。

## 三、总结

本文中我们完成了一场完整的肺炎 X 光图像二分类比赛（Pneumonia-SAI2）。我们首先介绍了比赛的背景：肺炎是常见的肺部感染疾病，胸部 X 光片是筛查肺炎的重要手段，本场比赛要求对 X 光片做"正常/肺炎"的二分类，评价标准是分类准确率。其次，我们做了 EDA，用目录遍历确认了数据组织方式：训练集按 NORMAL 和 PNEUMONIA 两个类别分子文件夹存放，测试集是分散的待预测图片。接着我们在数据清洗阶段利用 `ImageDataGenerator` 完成了像素归一化、统一缩放为 224×224，并加入旋转、缩放、水平翻转等数据增强手段，同时按 8:2 划分训练集和验证集。然后我们采用迁移学习策略，加载 ImageNet 预训练的 MobileNetV2 作为骨干网络，冻结大部分层、微调最后 20 层，在顶部接上全局平均池化、批归一化、全连接层和 Dropout 组成的分类头，并用 Adam 优化器配合早停和学习率衰减回调训练模型。最后我们在验证集上评估模型并打印准确率，绘制训练与验证的准确率、损失曲线，读取测试图片做预测，把概率按 0.5 阈值转成二分类标签，生成 `submission.csv` 提交文件。整个流程从数据探索、数据清洗、模型训练到最终提交一环扣一环，是一个很适合入门者模仿的医疗图像分类基线方案。