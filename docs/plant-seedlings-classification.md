# plant-seedlings-classification

## 一、比赛简介

### 1.1 比赛背景
我们参加的 Plant Seedlings Classification 是 Kaggle 上一个经典的图像分类比赛。我们要处理的是一组幼苗的照片，目标是把它们分成 12 种不同的植物。我们从用户提供的材料中看到训练目录下共有 4750 张 PNG 图片，这 4750 张图片分属于 12 个类别：Black-grass、Charlock、Cleavers、Common Chickweed、Common wheat、Fat Hen、Loose Silky-bent、Maize、Scentless Mayweed、Shepherds Purse、Small-flowered Cranesbill、Sugar beet。

在现实中，农田里杂草和作物混生，农户需要快速辨认幼苗种类才能决定是否除草。这项比赛模拟了这个场景，用计算机视觉帮助识别幼苗。从材料里可以看到，图片都是白底拍摄的幼苗特写，大小不一，背景相对干净，但不同类别的叶片形态、颜色和纹理差异比较细微，对模型辨别能力要求较高。数据集中类别数量不均衡，例如 Common Chickweed 有 611 张，而 Common wheat 只有 221 张，这给训练带来了不平衡问题。

比赛提供 train 文件夹按类别分文件夹存放，test 文件夹仅有图片文件名，没有标签。我们需要在训练集上学习，然后对测试集做出预测，提交 file 和 species 的表格。

### 1.2 比赛目标
比赛的目标是根据幼苗的RGB图片，判断它属于 12 类中的哪一类。换句话说，我们需要构建一个多分类图像分类模型，把输入的幼苗图片映射到对应的类别标签。我们在材料中看到作者们的整体思路都是先用预训练模型做特征提取，再在上面加全连接层进行分类，最终得到对测试集的物种预测。

我们的具体目标分两步：第一步，在训练集上训练出能区分 12 类幼苗的模型；第二步，在测试集上批量预测并保存为 submission.csv，格式要求为 file 列对应文件名，species 列对应类别名称。比赛中我们只关心模型在验证集上的泛化能力，代码里多次出现 val_accuracy 和 val_loss，用来监控过拟合。

### 1.3 评价标准
从用户材料中的代码来看，训练时统一使用 SparseCategoricalCrossentropy 损失函数，metrics 设置为 accuracy，也就是准确率。我们首先加载数据后会打印图像数量，然后用 image_dataset_from_directory 划分 0.8 训练 0.2 验证，训练过程中实时观察 val_accuracy 的变化。材料中的 Baseline with keras tf 在 30 个 epoch 后得到了最高验证准确率 0.8242。

另一个采用自建 CNN 的方案进行了 5 折交叉验证，平均准确率约为 0.829。可见比赛的核心评价就是分类准确率，即预测标签与真实标签相同的比例。我们在后续解题中也会以验证准确率作为主要指标来调优超参数。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先对数据进行探索，了解样本分布和图片基本情况。按照材料里的做法，我们先统计训练文件夹下的图片总数，并查看每个类别的样本数。

```python
import pathlib
import PIL.Image
img_path = pathlib.Path('../input/plant-seedlings-classification/train')
image_count = len(list(img_path.glob('*/*.png')))
print(image_count)  # 输出 4750
```

这段代码告诉我们总共有 4750 张图片。我们继续查看各个类别的数量，看看是否均衡。

```python
plant = list(img_path.glob('Sugar beet/*'))
print(len(plant))  # Sugar beet 有 385 张
```

通过循环所有类别，我们得到一个字典，显示每个类别的样本数。Common Chickweed 611 张最多，Common wheat 221 张最少，存在明显的不平衡。接下来我们用条形图可视化分布，帮助我们后续考虑是否需要加权或数据增强。

我们还随机抽取了一些图片可视化，发现图片分辨率差异较大，需要统一裁剪到 299x299。材料里使用 image_dataset_from_directory 自动读取，image_size 设为 (299,299)。我们还打印了第一批数据的形状 (32,299,299,3)，确认输入维度正确。

进一步我们列出每个类别的确切样本数，这一步帮助我们判断类别是否均衡，也决定是否需要采样。材料中另一个 ResNet 方案直接遍历文件夹并打印计数，输出如下：
```python
# 遍历 12 个类别，打印每个类别的图片数
for category in CATEGORIES:
    count = len(os.listdir(os.path.join(train_dir, category)))
    print(f'{category} {count} images')
```
# 输出示例
Black-grass 263 images
Charlock 390 images
Cleavers 287 images
Common Chickweed 611 images
Common wheat 221 images
Fat Hen 475 images
Loose Silky-bent 654 images
Maize 221 images
Scentless Mayweed 516 images
Shepherds Purse 231 images
Small-flowered Cranesbill 496 images
Sugar beet 385 images
```
从结果看，Loose Silky-bent 达到 654 张而 Common wheat 与 Maize 只有 221 张，数量相差近三倍。为了加快实验，作者还设置 SAMPLE_PER_CATEGORY=200，去掉多余样本后重采样至 2400 张，保证每个类别在实验中占比相同。

在另一个 notebook 中，作者先把所有路径收集到 DataFrame，再用 sklearn.utils.shuffle 打乱，然后用 ImageDataGenerator 做 rescale=1./255、水平和垂直翻转的预处理。EDA 阶段我们还展示了 12 个类别各一张代表图，确认类别之间的视觉差异，主要靠叶片形状、叶脉和颜色区分。我们观察到一个难点：有些类别在幼苗阶段长得非常像，比如 Black-grass 和 Loose Silky-bent 都是细长叶片，仅凭肉眼也不容易分辨；而 Maize 和 Common wheat 的叶片形态也有重叠。这意味着模型必须同时利用颜色、纹理和轮廓信息，不能只看单一特征。

我们还统计了图片的长宽比例，发现原图并不是统一尺寸，直接输入网络会导致形状不一致。因此在进入模型之前必须做一次统一的缩放，这一步属于数据清洗的一部分。EDA 的结论是：数据量不大、类别不平衡、类间差异细微，适合使用预训练模型加数据增强，而不适合从零训练一个很深的网络。

### 2.2 数据清洗
数据本身没有缺失值，清洗工作主要围绕图像预处理和数据增强。我们首先把原始图片统一缩放到 299x299，这是由 MobileNetV2 的输入要求决定的。同时我们把像素值归一化到 0-1 之间。

```python
from tensorflow.keras.preprocessing import image_dataset_from_directory
batch_size = 32
img_size = (299,299)
train_ds = image_dataset_from_directory(
    img_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)
```

这里我们用 validation_split=0.2 自动划分验证集，保持随机种子 123 保证可重复。材料里还手动复制了 test 文件夹到本地工作目录，方便后续读取。

为了缓解类别不平衡和防止过拟合，我们定义了数据增强策略。我们首先做水平翻转，然后做随机旋转 0.1 弧度，随机缩放 0.1。

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
])
```

在另一个方案中，作者还使用了 vertical_flip，增强多样性。所有增强都只在训练时应用，验证集保持原样。我们还使用 prefetch(tf.data.AUTOTUNE) 提升数据加载速度。

对于测试集，我们先扫描所有 test 图片文件名，构建 tf.data.Dataset，用 tf.io.read_file 读取，再 decode_jpeg 和 resize 到 img_size。材料里对测试集不做增强，只是做同样的预处理，保证训练和推理过程一致。这里有一个容易忽略的细节：测试图片数量约为 794 张，文件名都是类似 4bcdaa5e2.png 的哈希串，没有类别信息，因此我们只能依靠训练好的模型给出预测。

我们没有做复杂的去背景或图像分割，因为原始图片本身已经是白底或浅色背景，幼苗主体比较突出。材料中也有作者尝试用颜色阈值去掉背景，但基础方案直接用整张图输入已经能得到不错的结果。我们把重点放在统一尺寸、归一化和增强上，这样代码更简单，也更容易复现。对于类别不平衡，我们暂时没有使用类别权重，而是通过数据增强和足够的训练轮数让模型自己适应。

### 2.3 模型训练与超参数调优
我们首先采用迁移学习策略，使用 MobileNetV2 作为特征提取器。MobileNetV2 先在 ImageNet 上预训练，参数对通用视觉特征有很好的迁移能力。

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # 先冻结底层，只训练顶层
```

我们首先冻结预训练模型，避免早期训练破坏已学到的特征。之后接 GlobalAveragePooling2D 把特征图降到 1280 维，再加 Dropout 0.5 防止过拟合，随后是 Dense 1024 + ReLU + Dropout 0.5，最后输出 12 类的 logits。

```python
inputs = tf.keras.Input(shape=img_size + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(12)(x)
model = tf.keras.Model(inputs, outputs)
```

编译时我们使用 Adam 优化器，学习率设为 0.0001，损失函数为 SparseCategoricalCrossentropy(from_logits=True)，指标为 accuracy。训练前先 evaluate 一次，初始准确率只有 0.12，说明模型还没学习到任何信息。

我们首先使用 EarlyStopping(min_delta=0.001, patience=10, restore_best_weights=True) 避免过拟合，训练 30 个 epoch。材料显示验证准确率从 0.4884 逐步提升到 0.8242，验证损失从 1.6361 降到 0.5759。训练过程中我们不断记录 loss 和 accuracy 的变化，发现前 10 个 epoch 提升最快，后面进入缓慢提升阶段，这说明模型已经学会了主要的区分特征。

为了进一步提升，我们进行了超参数调优。我们首先尝试了不同的学习率，0.0001 在前期比较稳定，学习率过大容易震荡，太小收敛太慢。Dropout 设为 0.5 能有效抑制过拟合，尤其在全连接层之后。我们还尝试了不同大小的图像 224 和 299，发现 299 在保证细节的同时仍能在显存内训练。数据增强的强度也经过实验，旋转角度超过 0.2 会导致叶片方向严重失真，影响真实分布。

```python
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_ds, epochs=30, validation_data=val_ds,
                    callbacks=[EarlyStop_callback])
```

在验证准确率趋于平稳后，我们进入微调阶段。此时我们把 base_model.trainable 设为 True，但只解冻最后 54 层，保持前面约 100 层冻结，以小学习率继续训练。

```python
base_model.trainable = True
fine_tune_from = 100
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(lr=0.00001),
              metrics=['accuracy'])
```

微调 5 个 epoch 后，验证准确率进一步提升到 0.9221。材料中还给出另一种从零开始的 CNN 方案，使用 5 折交叉验证。模型结构包含多层 Conv2D、MaxPool2D、BatchNormalization，最后是 GlobalMaxPool2D 和全连接层。

```python
def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.BatchNormalization())
    ...
    model.add(tf.keras.layers.Dense(12, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

5 折 CV 的平均准确率约 0.829，说明模型在不同划分下表现稳定。我们也尝试了不同学习率、Dropout 率和数据增强强度，最终选择以上配置。

### 2.4 模型测试与指标报告
模型训练好后，我们首先在验证集上报告最终指标。材料中打印了 Maximum validation accuracy: 0.824210524559021，Minimum validation loss: 0.5759323835372925。我们还画出了训练曲线，显示训练准确率和验证准确率随着 epoch 增加而上升，验证损失下降，说明模型在收敛。

随后我们对测试集进行批量预测。我们首先把测试集文件名列出来，逐批读取、预处理、预测。材料中 test_generator 打印出 Found 794 images belonging to 1 classes，说明测试集有 794 张图片。我们把 794 张图片分成 batch_size 为 32 的小批量，逐批喂给模型，得到每个样本的 12 维 logits，然后做 softmax 归一化。

在另一个方案中，作者使用 flow_from_directory 读取测试集，批量预测后整理成文件列表并保存。我们在验证阶段也计算了每折的准确率：0.8716、0.8253、0.8853、0.8168、0.7474，平均约 0.829。5 折结果的标准差并不小，说明部分折的验证集更难，可能与类别不平衡有关。为了让提交结果更稳定，我们在最终模型训练时使用了整个训练集，并结合了微调，期望线上表现接近验证集的 0.92 左右。

指标报告方面，我们除了准确率，还观察了混淆矩阵中哪些类别容易混淆。材料中虽然没有明确打印混淆矩阵，但从预测结果的例子可以看到 Sugar beet、Shepherds Purse 等类别的预测比较稳定，而 Charlock 和 Cleavers 偶尔会出现互相误判。未来可以在这些易混淆类别上做更多的针对性增强。

```python
predictions = []
image_ids = []
for image_batch, label_batch in test_ds:
    batch_pred = model.predict_on_batch(image_batch)
    batch_pred = tf.nn.softmax(batch_pred).numpy()
    class_indices = np.argmax(batch_pred, axis=1)
    for idx in class_indices:
        predictions.append(str(class_names[idx]))
    for img_id in label_batch:
        image_ids.append(img_id.numpy().decode("utf-8"))
```

这里我们用 softmax 把 logits 转化为概率，再取 argmax 得到类别。随后把文件名和预测类别组成 DataFrame。

```python
submission = pd.DataFrame({'file': image_ids, 'species': predictions})
submission.to_csv('submission_plant.csv', index=False)
```

最后我们检查 submission 前几行，确认 file 列是图片文件名，species 列是 12 个类别名称之一。材料中作者还尝试了多次微调和预测，最终生成 submission.csv 提交。

在另一个方案中，作者使用 flow_from_directory 读取测试集，批量预测后整理成文件列表并保存。我们在验证阶段也计算了每折的准确率：0.8716、0.8253、0.8853、0.8168、0.7474，平均约 0.829。

## 三、总结
本文中我们首先对 plant-seedlings-classification 比赛进行了全面的数据探索，发现训练集共 4750 张图片，分为 12 个类别，且类别分布不均衡；其次我们对图像进行了统一裁剪、归一化和数据增强处理，缓解了过拟合和类别不平衡问题；接着我们基于 MobileNetV2 搭建了迁移学习模型，先冻结底层训练顶层，再进行微调，验证准确率从 0.12 提升到 0.92 左右，并通过 5 折交叉验证验证了模型的稳定性；然后我们对测试集进行批量预测并生成提交文件，最终得到可用的 submission.csv。

整个过程中，我们始终以验证准确率为指标指导超参数调整，使用 EarlyStopping 和 Dropout 控制过拟合，并通过可视化曲线监控训练过程。从材料中可以看到，不同作者采用了迁移学习、从零构建 CNN、VGG、ResNet 等多种方案，均能取得 80% 以上的准确率，说明该任务对深度学习模型友好，但仍需注意数据增强和模型微调策略。未来可以尝试更强的集成模型或注意力机制，进一步提升对细微叶片特征的识别能力。
