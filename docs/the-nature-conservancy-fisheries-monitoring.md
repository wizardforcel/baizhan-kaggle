# The Nature Conservancy Fisheries Monitoring

## 一、比赛简介

### 1.1 比赛背景

The Nature Conservancy Fisheries Monitoring 是 Kaggle 在 2017 年前后举办的一场计算机视觉比赛。它的现实背景是海洋渔业监管：很多远洋渔船在甲板上装有摄像头，靠岸后会把录像交上来，但录像时长惊人，靠人工逐帧查看成本很高。比赛希望参赛者训练图像分类模型，自动识别画面里出现的是哪种鱼、有没有鱼，从而帮助机构判断捕捞是否合规。

训练数据是一批从渔船摄像头里截取的静态图片，按鱼类分文件夹存放。材料里列出的文件夹有 `ALB`、`BET`、`DOL`、`LAG`、`NoF`、`OTHER`、`SHARK`、`YFT`，外加一个 macOS 生成的 `.DS_Store`。前几个缩写分别对应长鳍金枪鱼、大眼金枪鱼、鲯鳅、 triggerfish、无鱼、其他、鲨鱼和黄鳍金枪鱼。`NoF` 是 No Fish 的缩写，表示画面里没有鱼。比赛通常分为两个阶段：第一阶段用一批图片训练并提交，第二阶段换一批更难、更贴近真实分布的图片，防止队伍过拟合第一阶段的样本。

这类比赛难在哪？第一，同一种鱼在不同角度、光照、遮挡下差别很大；第二，背景是甲板、海水、箱子，容易干扰；第三，类别不平衡，有的鱼种样本很多，有的很少；第四，测试集的拍摄船和训练集可能不同，模型要能泛化到「没见过的船」。

### 1.2 比赛目标

输入是一张 JPEG 图片，输出是它属于各个类别的概率。材料中的代码先遍历 `train/` 下的每个类别文件夹，把图片路径和标签收集成两个列表，再用 `LabelEncoder` 把类别名编码成 0 到 7 的整数，最后用 `to_categorical` 做成 one-hot 向量。

训练集一共 3777 张图片，其中训练划分 2832 张、验证划分 945 张。测试集 Stage 1 有 1000 张图片。提交文件 `sample_submission_stg1.csv` 的第一列是图片名，后面八列是八个类别的概率。每条样本的八个概率应当加起来等于 1。

### 1.3 评价标准

比赛使用多分类对数损失（multi-class log loss），越低越好。它的直觉解释是：模型给正确答案的概率越高，惩罚越小；如果给正确答案的概率很低，惩罚会迅速变大。公式是

```
logloss = - (1/N) * sum( log(p_correct) )
```

其中 `p_correct` 是模型给真实类别的预测概率。因为取了 log，概率从 0.9 降到 0.5 的惩罚，远小于从 0.1 降到 0.01。这个指标既惩罚「答错」，也惩罚「过度自信地答错」，所以模型输出的概率必须校准，不能只追求 argmax 正确。

材料里还出现了一个 KNN 方案的标题，写着 `log-loss = 1.65074`，可以作为「简单方法大概在什么水平」的参照。我们下面这条主线用的是迁移学习加数据增强，验证集上的最佳 val_loss 约为 0.0712，对应的验证准确率约为 0.979。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把训练目录读进来，看看一共有多少类、每类多少张。代码遍历 `train/` 下的子文件夹，跳过以点开头的隐藏文件，把每张 `.jpg` 的完整路径和它所在的文件夹名分别存起来。

```python
from os import listdir
from tqdm import tqdm

def loadBatchImages(path):
    catList = listdir(path)
    loadedImages, loadedLabels = [], []
    for cat in catList:
        if cat.startswith('.'):        # 跳过 .DS_Store 这类隐藏文件
            continue
        deepPath = path + cat + '/'
        for img in tqdm(listdir(deepPath)):
            loadedImages.append(deepPath + img)
            loadedLabels.append(cat)   # 文件夹名就是类别名
    return loadedImages, loadedLabels

loadedImages, loadedLabels = loadBatchImages('../input/train/')
```

读完后我们统计类别数。材料里打印出来是 8 类，和八个文件夹一一对应。接着我们把类别名编码成整数，再转成 one-hot 向量。one-hot 的意思是用一个长度等于类别数的向量表示类别，正确类别的位置是 1，其余位置是 0。

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

num_classes = len(np.unique(loadedLabels))     # 结果是 8
encoder = LabelEncoder()
encoded = encoder.fit_transform(np.asarray(loadedLabels))
labels_Hot = to_categorical(encoded, num_classes=num_classes)
```

从遍历的进度条可以看出类别分布并不均衡：最多的类别有 1719 张，最少的只有 67 张，差了二十多倍。这种不平衡在真实渔业数据里很正常，因为某些鱼更常被捕获。它会影响训练：如果直接按原始分布训练，模型会倾向于预测多数类。我们后面在划分验证集时保留这个分布，并在评估时同时看 loss 和 accuracy，避免只看准确率被多数类带偏。

图片尺寸方面，原始图片分辨率不一，但模型输入被统一缩放到 128×128。材料中的生成器输出形状是 `(128, 128, 3)`，也就是宽 128、高 128、三个颜色通道。

### 2.2 数据清洗

图像任务的「清洗」和表格任务不太一样。我们不做删行、填补缺失值，而是做尺寸统一、数值归一化和数据增强。材料使用 Keras 的 `ImageDataGenerator`，在训练时实时做变换，而不是提前把增强后的图片写到磁盘。这样做节省空间，而且每个 epoch 看到的增强版本都不一样，相当于免费扩大了数据集。

```python
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)

core_idg = ImageDataGenerator(
    samplewise_center=True,            # 每张图减去自己的均值
    samplewise_std_normalization=True, # 再除以自己的标准差
    horizontal_flip=True,              # 水平翻转：鱼左右朝向不重要
    vertical_flip=False,               # 不上下翻转：甲板方向有现实含义
    height_shift_range=0.05,           # 轻微上下平移
    width_shift_range=0.1,             # 轻微左右平移
    rotation_range=5,                  # 小角度旋转
    shear_range=0.1,                   # 轻微错切
    fill_mode='reflect',               # 变换后空出来的像素用镜像填补
    zoom_range=0.15,                   # 随机缩放
)
```

为什么用 `samplewise_center` 而不是全局均值？因为渔船照片的亮度差异很大，有的在强光下，有的在阴影里。逐张减均值、除标准差，能把每张图的亮度和对比度拉到大致相同的范围，减少光照带来的干扰。翻转和缩放则模拟了拍摄角度、距离的变化，让模型不要死记某一张图的构图。

接着我们把图片路径和 one-hot 标签组成 DataFrame，再封装一个 `flow_from_dataframe`，让生成器按 DataFrame 里的路径读图。材料先按 75% / 25% 切成训练和验证两部分，然后用固定的验证批次做评估，保证每次评估用的是同一批图片。

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame({'path': loadedImages, 'labels': list(labels_Hot)})
train_df, valid_df = train_test_split(df, test_size=0.25, random_state=2018)

train_gen = flow_from_dataframe(core_idg, train_df, path_col='path',
                                y_col='labels', target_size=IMG_SIZE,
                                batch_size=128)
valid_gen = flow_from_dataframe(core_idg, valid_df, path_col='path',
                                y_col='labels', target_size=IMG_SIZE,
                                batch_size=256)
```

注意验证集也用了同一个生成器，但验证时不应该做随机增强。严谨的做法是给验证集单独建一个只做归一化、不做随机变换的生成器。材料把验证图片一次性取成一个大 batch，这样同一张图每次评估的结果一致，便于比较不同 epoch 的 loss。

### 2.3 模型训练与超参数调优

我们采用迁移学习。迁移学习的意思是：不在 3777 张小图上从零训练一个卷积网络，而是拿一个已经在 ImageNet（上百万张通用图片）上训练好的网络，把它学到的「边缘、纹理、形状」特征借过来用。材料选的是 VGG16，去掉它原来的分类头，只保留卷积部分。

```python
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Flatten, Dense, BatchNormalization

img_dim = (128, 128, 3)
input_tensor = Input(shape=img_dim)

# 先对输入做批归一化，让数值更稳定
bn = BatchNormalization()(input_tensor)

# 载入 VGG16 的卷积部分，include_top=False 表示不要原来的 1000 类分类头
base_model = VGG16(include_top=False, input_shape=img_dim)
x = base_model(bn)
x = Flatten()(x)                       # 把 4×4×512 的特征图拉平
output = Dense(num_classes, activation='softmax')(x)  # 8 类概率

model = Model(input_tensor, output)
```

这个模型一共有约 1478 万个参数，其中绝大部分来自 VGG16 的卷积层，最后那个全连接层只有 65544 个参数。`softmax` 的作用是把输出变成八个 0 到 1 之间、加起来等于 1 的概率。

训练时我们使用 Adam 优化器，学习率设为 1e-4，损失函数是 `binary_crossentropy`，评估指标是准确率。这里材料用的是二分类交叉熵配 softmax 输出，属于常见写法；更严格地说，多分类 one-hot 标签应该用 `categorical_crossentropy`，两者在数值上非常接近，但概念上后者更对口。

```python
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam

callbacks = [
    # 验证 loss 连续 3 轮没有明显下降就停，防止过拟合
    EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-4, verbose=1),
    # 验证 loss 停滞时把学习率乘以 0.1，帮助模型跳出平台
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1,
                      min_lr=1e-7, verbose=1),
    # 只保存验证 loss 最好的那一轮权重
    ModelCheckpoint('weights.best.hdf5', save_best_only=True, verbose=1),
]

model.compile(optimizer=Adam(lr=1e-4),
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(train_gen, steps_per_epoch=len(train_df)/128,
                    validation_data=(test_X, test_Y),
                    epochs=25, callbacks=callbacks)
```

超参数方面，我们没有做大规模网格搜索，而是靠两个回调函数控制训练：早停决定「什么时候停」，学习率衰减决定「步子多大」。批次大小 128 是在显存和梯度稳定性之间的折中。训练 25 轮的上限只是给早停一个边界，实际在第 20 轮左右就触发了早停。

从材料记录的训练日志看，第 1 轮验证 loss 是 0.2708、准确率 0.8933；到第 10 轮验证 loss 降到 0.0785、准确率 0.9754；第 17 轮达到最好的 0.0712 左右，准确率约 0.9788。之后 loss 基本不再下降，学习率被反复衰减到 1e-7，早停生效。这个曲线说明模型在前几轮学得很快，后面进入平台期，继续训练只会增加过拟合风险。

### 2.4 模型测试与指标报告

训练结束后，我们先把验证 loss 最好的权重加载回来，再对测试集做推理。测试图片放在 `test_stg1/` 下，文件名形如 `img_00837.jpg`。提交文件要求第一列是去掉扩展名的图片名，所以我们要从路径里提取 basename 再去掉 `.jpg`。

```python
import os
from glob import glob

model.load_weights('weights.best.hdf5')

test_paths = glob('../input/test_stg1/*.jpg')
X_test = pd.DataFrame({'path': test_paths})
X_test['image'] = X_test['path'].map(
    lambda x: os.path.splitext(os.path.basename(x))[0])
```

然后我们用测试生成器读图、预测，得到每张图的八维概率向量，写进提交表。

```python
test_gen = flow_from_dataframe(core_idg, X_test, path_col='path',
                               y_col=None, target_size=IMG_SIZE,
                               batch_size=256, class_mode=None,
                               shuffle=False)

proba = model.predict_generator(test_gen, verbose=1)
submission = pd.DataFrame(proba, columns=encoder.classes_)
submission.insert(0, 'image', X_test['image'].values)
submission.to_csv('submission.csv', index=False)
```

推理时最重要的细节是 `shuffle=False`。如果打乱顺序，预测结果和图片名就对不上了，提交上去等于随机分配。材料在整理测试 DataFrame 时保持了 `glob` 的原始顺序，预测时也禁止打乱，最后按列名对齐，这样才安全。

指标报告方面，我们主要看两个数：验证 loss 和验证准确率。最佳轮次的 val_loss 约为 0.0712，val_acc 约为 0.979。这个准确率看起来很高，但要结合类别不平衡来读：如果多数类占了很大比例，一个只会猜多数类的模型也能拿到不低的准确率。所以我们同时看 log loss，它会对「给正确答案的概率不够高」持续惩罚，更难被多数类糊弄。

材料里还提到其他队伍的思路，比如用 KNN 做基线（标题里 log-loss 为 1.65074）、把多个模型的预测做集成（`Ensemble your models for better score`）、以及专门检测「船 ID」来利用同一艘船的图片相似性。这些方案说明：在这个比赛里，单靠一个 CNN 不够，理解数据的采集结构（同一艘船、同一批航次）往往能带来额外提升。我们的主线是一个清晰可复现的迁移学习基线，不追求排行榜名次。

## 三、总结

本文中我们完成了 The Nature Conservancy Fisheries Monitoring 的图像分类流程。我们首先遍历训练文件夹，确认共有 8 个类别、约 3777 张图片，并发现类别分布很不均衡；其次用 `ImageDataGenerator` 做逐样本归一化和随机增强，把图片统一到 128×128，并按 75% / 25% 划分训练与验证；接着用预训练的 VGG16 卷积部分加一个 8 类 softmax 头做迁移学习，配合 Adam、早停、学习率衰减和最优权重保存；最后加载最佳权重，对 1000 张测试图片做不打乱顺序的推理，输出符合提交格式的八列概率。

这条主线让我们看到迁移学习的价值：在只有几千张图片的情况下，从零训练很容易过拟合，而借用一个在百万张图片上学过特征的网络，几轮之内就能把验证 loss 降到 0.07 左右。同时也暴露出局限：验证集准确率很高，不代表在第二阶段的新船上一样好；要真正提升泛化，需要更多样的数据、更强的增强，或者利用「同一艘船」这种结构信息。材料中的集成方案和船 ID 检测，正是沿着这个方向走的。
