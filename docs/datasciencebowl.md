# National Data Science Bowl：给浮游生物照片分类

## 一、比赛简介

### 1.1 比赛背景

这个比赛是 Kaggle 的 National Data Science Bowl（2015 年），题目来自海洋科学：**自动识别水里的浮游生物**。

浮游生物是海洋食物链的底座，它们的种类和数量变化能反映海水温度、酸化程度、污染等环境问题。传统做法是科学家把水样放到显微镜下，一张一张人工辨认——慢、累、而且不同专家的判断还会不一致。比赛提供的数据来自水下成像设备：水流过一个微型「相机」，里面的浮游生物被拍成一张张小照片。我们要做的，就是让模型看着这些照片说出它是什么。

难在三点：

1. **类别特别多**，一共 **121 类**，从各种桡足类、水母、鱼卵到「不明碎屑」都有；
2. **类别长得像**，很多生物只有触角形状、身体轮廓的细微差别，人眼都难分；
3. **样本极不均衡**，常见的种类有上千张，罕见的只有几十张，还有一类干脆叫 `artifacts`（拍摄伪影）。

### 1.2 比赛目标

数据以压缩包的形式给出：

- `train.zip`：训练图片，**共 30458 张**，按类别放在不同文件夹里；
- `test.zip`：测试图片，没有类别标签；
- `sampleSubmission.csv.zip`：提交样例，规定了输出格式；
- `plankton_identification.pdf`：官方说明文档。

输入是一张浮游生物照片，输出是它所属的类别（121 选 1）。提交文件里每一行对应一张测试图，给出它的类别编号。

### 1.3 评价标准

评价标准是**分类准确率（categorization accuracy）**：预测对的图片数除以测试图片总数。注意这是多分类准确率，不是二分类的 AUC，也不是「每类各算一遍再平均」的宏平均——一张图猜对就是一分，猜错就是零分。分数越高越好，满分 1.0。

因为类别多且不均衡，只看总分容易掩盖问题：模型完全可能把常见的十几类学得很好，把上百个罕见类全猜错，总分却还行。所以我们在本地分析时，除了准确率，还会看每个类的 F1 和混淆矩阵。

---

## 二、解题过程

我们的方案是**迁移学习**：拿一个在 ImageNet 上训练好的 InceptionV3，把它的卷积层冻住当「特征提取器」，只在后面接一个小的分类头，训练它输出 121 类。

### 2.1 数据探索（EDA）

我们首先遇到的问题不是「数据长什么样」，而是「数据根本读不进来」。官方给的是压缩包，所以第一步是看清单里有什么：

```python
import os

# Kaggle 的输入目录是只读的，先把文件列出来
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 输出：
# .../datasciencebowl/train.zip
# .../datasciencebowl/test.zip
# .../datasciencebowl/sampleSubmission.csv.zip
# .../datasciencebowl/plankton_identification.pdf
```

不把 zip 解开是没法读图的，所以直接用 Python 的 `zipfile` 看里面有多少张图，顺便抽几张画出来：

```python
from zipfile import ZipFile
from PIL import Image
import matplotlib.pyplot as plt

zip_path = "/kaggle/input/datasciencebowl/train.zip"
with ZipFile(zip_path) as z:
    names = z.namelist()          # 压缩包里所有文件的名字
print("训练图片总数:", len(names))  # 30458

# 随机挑一张显示，确认图片是灰度还是彩色、大小多少
with ZipFile(zip_path) as z:
    with z.open(names[5000]) as f:
        img = Image.open(f)
        print(img.size, img.mode)
        plt.imshow(img, cmap="gray")
        plt.title(names[5000])
```

看几张之后我们有了直观印象：图片尺寸并不统一，生物在画面里的大小、朝向、位置也各不相同，背景大多是灰度均匀的水体。另外文件名本身带有类别信息（材料里的 `selectedClasses` 就是从文件夹名整理出来的 121 个类别名）。

```python
# 材料里整理的 121 个类别（节选）
selectedClasses = [
    "copepod_cyclopoid_oithona", "copepod_calanoid_frillyAntennae",
    "radiolarian_colony", "hydromedusae_narco_dark", "trichodesmium_puff",
    "fish_larvae_leptocephali", "ctenophore_cydippid_tentacles",
    "artifacts", "unknown_blobs_and_smudges", "diatom_chain_string",
    # ... 一共 121 个
]
print(len(selectedClasses))   # 121
```

从名字能看出类别的粒度很细：光是桡足类（copepod）就按触角形状、体型分成了十几个子类。这解释了为什么这个比赛难——不是「分不分类」，而是「分得够不够细」。

**一个关键的坑。** 材料里的 notebook 写的是：

```python
train_path = dataDirectory + "/train"
test_path  = dataDirectory + "/test"
print(os.listdir(train_path))   # FileNotFoundError: './/train'
```

它假设数据已经解压成 `train/` 和 `test/` 两个文件夹，但 Kaggle 上实际只有 `train.zip`。结果后面三行 `flow_from_directory` 全部返回 `Found 0 images belonging to 121 classes`，训练根本没跑起来。所以 EDA 这一步的真正结论是：**先确认数据以什么形式存在，再写读取代码**，否则后面所有代码都是空转。

### 2.2 数据清洗

我们把「清洗」分成三件事：解压并整理目录、统一尺寸、做数据增强。

**第一，解压。** `flow_from_directory` 要求目录结构是「一个类别一个子文件夹」。解压之后先检查结构对不对：

```python
import zipfile, os

def unzip(src, dst):
    with zipfile.ZipFile(src) as z:
        z.extractall(dst)

unzip("/kaggle/input/datasciencebowl/train.zip", "./data/train")
unzip("/kaggle/input/datasciencebowl/test.zip",  "./data/test")

# 确认每个类别是不是一个文件夹
cls_dirs = os.listdir("./data/train")
print(len(cls_dirs), cls_dirs[:5])
```

如果解压出来是平铺的一堆图片（没有类别文件夹），就得先按文件名里的类别前缀建好子文件夹再移动。这一步不做，后面的生成器永远读到 0 张图。

**第二，统一尺寸。** 原始图片大小不一，而神经网络要求固定输入。我们统一缩放到 **224 × 224**——这是 InceptionV3 的标准输入尺寸，也是 ImageNet 预训练权重期待的大小。缩放交给生成器做，不用提前把几万张图重新存一遍。

**第三，数据增强。** 训练集只有三万多张，要分 121 类，平均每类两百多张，很容易过拟合。我们让每张图在训练时随机做旋转、平移、剪切、缩放：

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=40,        # 随机旋转 ±40 度，生物在水里本来什么朝向都有
    width_shift_range=0.2,    # 左右平移最多 20%
    height_shift_range=0.2,   # 上下平移最多 20%
    shear_range=0.2,          # 剪切变换，模拟拍摄角度变化
    zoom_range=0.2,           # 随机放大缩小
    fill_mode="nearest",      # 变换后露出的边缘用最近的像素补
    validation_split=0.2,     # 留 20% 做验证
    preprocessing_function=preprocess_input,  # 按 ImageNet 的方式归一化
)
```

`preprocessing_function` 这一行是我们补上的：InceptionV3 的预训练权重是在「像素值缩放到 -1～1」的图片上学的，如果直接喂 0～255 的原图，特征分布对不上，效果会明显变差。材料里 import 了 `preprocess_input` 却没有用上，这是个容易被忽略的细节。

### 2.3 模型训练与超参数调优

模型结构很标准：**冻结的 InceptionV3 + 全局平均池化 + Dropout + 全连接分类头**。

```python
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.models import Model

# include_top=False：去掉 ImageNet 原来的 1000 类分类头，只要卷积特征
base_model = InceptionV3(weights="imagenet", include_top=False,
                         input_shape=(224, 224, 3))
# 冻结卷积层：训练时不动这些权重，省显存也防止把预训练特征练坏
base_model.trainable = False

x = base_model.output
# 全局平均池化：把每个通道的整张特征图压成一个数，比 Flatten 参数少得多
x = GlobalAveragePooling2D()(x)
# Dropout 0.5：训练时随机丢掉一半神经元，抑制过拟合
x = Dropout(0.5)(x)
# 输出 121 类的概率
predictions = Dense(len(selectedClasses), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.summary()
```

`model.summary()` 打出来的数字很能说明迁移学习的价值：

```
Total params: 22,050,713
Trainable params: 247,929
Non-trainable params: 21,802,784
```

两千两百万参数里，真正需要训练的只有 **24.8 万**，其余两千万都是 ImageNet 学好的。这就是为什么我们敢在三万张图上训练一个两千万参数的模型——大部分知识是「借」来的。

编译和训练：

```python
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# 学习率取 1e-4：分类头是随机初始化的，步子太大会把预训练特征带偏
model.compile(optimizer=Adam(lr=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 只保存验证准确率最高的那一轮，避免最后一轮恰好过拟合
checkpoint = ModelCheckpoint("best.hdf5", monitor="val_acc",
                             save_best_only=True, mode="max")

history = model.fit_generator(
    train_generator,
    validation_data=validation_generator,
    epochs=3,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[checkpoint])
```

**关于超参数。** 我们主要关注四个：学习率、Dropout 比例、增强强度、训练轮数。

- 学习率用 1e-4，是微调预训练模型的常用起点；再大容易震荡，再小收敛太慢；
- Dropout 0.5 是分类头的常规值，因为分类头的参数量相对特征维度来说很大；
- 增强参数（旋转 40 度、平移 0.2 等）是偏强的一组，理由是浮游生物的拍摄角度本来就随机，强增强相当于告诉模型「朝向不重要」；
- 轮数先设 3 轮做快速验证。材料里的思路是先用少量轮次看验证曲线有没有上升趋势，再决定要不要加。

如果想进一步提升，标准做法是**两段式微调**：先只训分类头几轮，等它稳定了，再解冻 InceptionV3 最后的一两个 block，用更小的学习率（比如 1e-5）一起训。这样既利用了预训练特征，又能让高层特征适配浮游生物这种和 ImageNet 差别很大的图像。

### 2.4 模型测试与指标报告

**必须先说清楚一件事：材料里的这条 notebook 并没有跑出分数。** 因为它把路径写成了 `./train`，而数据只有 `train.zip`，三个生成器都读到 0 张图，`fit_generator` 直接抛出 `ValueError: Asked to retrieve element 0, but the Sequence has length 0`，后面的 `history` 也就没有定义。所以材料里没有任何可报告的准确率。我们不编造分数，只把「修好之后应该怎么评估」讲清楚。

第一，用生成器的 `validation_split` 划出 20% 的验证集，训练时直接看 `val_acc`：

```python
# 训练集和验证集共用一份增强配置，但只有训练集会做随机变换
train_generator = train_datagen.flow_from_directory(
    "./data/train", target_size=(224, 224), batch_size=32,
    classes=selectedClasses, subset="training")
validation_generator = train_datagen.flow_from_directory(
    "./data/train", target_size=(224, 224), batch_size=32,
    classes=selectedClasses, subset="validation")

# 每个 epoch 需要多少步 = 样本数 / 批大小（向上取整）
steps_per_epoch = (train_generator.samples + 31) // 32
```

`steps_per_epoch` 这一步不能省。材料里算出来是 0，正是因为 `train_generator.samples` 是 0；修好路径之后它才会变成几千，训练才会真正开始。

第二，训练结束后在验证集上逐类看指标，而不是只看一个总分：

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 重新跑一遍验证集，拿到预测和真实标签
validation_generator.reset()
prob = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(prob, axis=1)
y_true = validation_generator.classes[:len(y_pred)]

print(classification_report(y_true, y_pred, target_names=selectedClasses))
cm = confusion_matrix(y_true, y_pred)   # 看哪些类被互相搞混
```

`classification_report` 会给出每一类的 precision、recall、F1。我们预期会看到明显的长尾现象：常见的桡足类 F1 较高，只出现几十次的罕见类 F1 很低。混淆矩阵则能告诉我们「谁被认成了谁」——如果两个形态相近的类总是互相混淆，就说明模型在 224 像素的分辨率下分辨不出那个细节，可以考虑提高输入分辨率或者做局部裁剪。

第三，对测试集预测并生成提交文件：

```python
# 测试集也要按同样的尺寸和预处理走一遍，注意 shuffle=False
test_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory("./data/test", target_size=(224, 224),
                      batch_size=32, shuffle=False)

test_prob = model.predict(test_generator, verbose=1)
test_label = np.argmax(test_prob, axis=1)

# class_indices 是「类名 -> 编号」，反过来查就能把编号还原成类名
idx_to_class = {v: k for k, v in test_generator.class_indices.items()}

import pandas as pd
submission = pd.DataFrame({
    "id": [os.path.basename(p) for p in test_generator.filenames],
    "category": [idx_to_class[i] for i in test_label],
})
submission.to_csv("submission.csv", index=False)
```

这里有两个细节值得强调：测试集生成器必须 `shuffle=False`，否则提交文件里的顺序和文件名对不上；`class_indices` 的顺序是按文件夹名字母排序的，不是我们写的 `selectedClasses` 顺序，所以一定要用它的映射反查，不能想当然地按下标对应。

---

## 三、总结

本文中我们完成了 National Data Science Bowl 浮游生物分类的一次迁移学习实践。我们首先用 `zipfile` 直接查看压缩包，确认训练集有 30458 张图、121 个类别，并发现原 notebook 因为把路径写成 `./train` 而完全读不到数据。其次我们解压并整理成「一类一个文件夹」的结构，统一缩放到 224 × 224，用旋转、平移、剪切、缩放做数据增强，并补上了 ImageNet 的预处理归一化。接着我们冻结 InceptionV3 的卷积层，接上全局平均池化、Dropout 0.5 和 121 类 softmax，只训练 24.8 万个参数，用 Adam 1e-4 配合验证集 checkpoint 训练。最后我们给出了按类看 F1、看混淆矩阵、以及在测试集上生成提交文件的完整流程；因为材料里的训练没有跑起来，我们没有报告排行榜分数。

这个比赛给我们三个教训。第一，**先看数据怎么存，再写读数据的代码**——一个路径错误就能让整条流水线空转，而报错信息（`Found 0 images`）其实已经把原因说得很清楚了。第二，**迁移学习的性价比来自「冻结」**：两千万参数不用重训，我们只调 24 万，就能把 ImageNet 的通用视觉特征借过来。第三，**多分类比赛不能只看总分**：121 类里长尾严重，逐类 F1 才能暴露模型真正没学会的是哪些生物。
