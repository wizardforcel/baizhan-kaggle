# SAI 血管分割 2

## 一、比赛简介

### 1.1 比赛背景
SAI Vessel Segmentation 2 是一个医学图像语义分割比赛，目标是对视网膜血管图像进行像素级分割，区分血管与背景。该任务来源于医学影像分析领域，血管分割是眼底疾病筛查、糖尿病视网膜病变检测等研究的基础步骤。视网膜血管在显微镜下呈现为细长的管状结构，宽度可能只有几像素，长度却能贯穿整幅图像，这给传统基于形状的分割方法带来很大挑战。

材料里使用的公开思路基于 U-Net 架构，结合 VGG16 预训练编码器做特征提取，再通过上采样和跳跃连接恢复精细的血管结构。U-Net 的设计初衷就是为了解决医学图像分割中目标边界模糊、细节缺失的问题，编码器负责捕捉上下文语义，解码器通过跳跃连接把低级细节信息融合回来。视网膜图像中血管占像素比例很小，存在显著的类别不平衡，网络需要在小目标上保持高召回，同时又不能把背景噪声误判为血管。

### 1.2 比赛目标
我们需要对每张 512×512 的 RGB 训练图像生成对应的二值血管掩码。

训练集文件放在 `/kaggle/input/sai-vessel-segmentation2/all/train` 目录下，图像为 `.tif` 格式，掩码为 `.gif` 格式。测试集放在同目录下的 test 文件夹，用于生成提交。材料里的代码使用 PIL 打开图像，统一 resize 到 512×512，并在训练时使用随机旋转和裁剪作为数据增强。

每个样本包含：
- 输入图像：512×512×3；
- 标签掩码：512×512×1，二值。

图像读取时先用 PIL 打开，再转成 numpy 数组并归一化到 0~1，掩码则通过自定义函数把大于 0 的值都转为 1，确保标签是严格二值的。模型需要学习从原始眼底图像到血管像素图的映射，在推理时对测试集的每张图像输出一个同尺寸的二值掩码。

### 1.3 评价标准
材料里自定义并使用了 Dice 系数作为分割质量的核心指标，Dice 越高越好，同时也监测了整体像素准确率。Dice 系数衡量预测掩码与真实掩码的重叠程度，定义为 2×交集 / (预测和 + 真实和)，对类别不平衡的分割任务比单纯的准确率更可靠。因为血管像素只占整幅图像一小部分，准确率很容易被背景占主导而被高估，而 Dice 关注的是正样本的重叠程度，更能反映模型是否真的把血管找出来。

训练时还使用了 Binary Crossentropy 作为损失函数，从两方面共同优化。Binary Crossentropy 对每个像素做二分类似然估计，能够稳定训练；Dice 系数则作为监控指标，帮助我们观察模型在细小结构上的表现。实际部署时，通常需要把模型输出的概率图用 0.5 阈值二值化再计算 Dice。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先列出数据目录，确认图像和掩码数量是否一一对应。

```python
# 列出目录下所有文件
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

数据生成函数通过 glob 收集所有 `.tif` 图像和 `.gif` 掩码，并按文件名排序确保配对。

```python
TRAIN_PATH = '/kaggle/input/sai-vessel-segmentation2/all/train'
image_files = glob(os.path.join(TRAIN_PATH, '*.tif'))
mask_files = glob(os.path.join(TRAIN_PATH, '*.gif'))
DATASET_SIZE = len(image_files)
```

图像与掩码数量一致，材料里没有出现缺失文件。我们抽取一条样本做可视化，图像经过 resize 到 512×512 并归一化到 0~1，掩码同样 resize 后二值化。

```python
image = PIL.Image.open(train_image).resize(IMAGE_SIZE)
image = np.array(image) / 255

mask = PIL.Image.open(train_mask).resize(IMAGE_SIZE)
mask = np.array(mask) / 255
```

从可视化可知血管呈纤细的线状结构，背景占绝大多数，说明后期训练需要关注正样本的召回。我们还注意到掩码的像素值不是严格的 0 与 255，而是需要把大于 0 的像素统一视作血管。

```python
# 快速统计正负像素比例
pos_ratio = mask.mean()
print(f"血管像素占比: {pos_ratio:.4f}")
```

典型情况下正样本占比不到 10%，属于强不平衡。

### 2.2 数据清洗
原始图像和掩码尺寸一致，但需要统一预处理。材料里的一个关键细节是把掩码中所有非零灰度值统一转成 0/1 标签，因为眼底图像的掩码可能带有灰度渐变或标注误差，直接当二值标签用会让模型学到错误的信号。

```python
def convert_to_label(mask):
    # 把大于 0 的像素映射为 1，其余为 0
    label_func = np.vectorize(lambda p: 1 if p > 0 else p)
    return label_func(mask)
```

图像归一化除以 255，掩码重新 reshape 成单通道，与网络的输出形状对齐。

```python
image = image / 255
mask_img = mask_img.reshape((mask_img.shape[0], mask_img.shape[1], 1))
```

数据增强在 tf.data pipeline 里实现，包含随机旋转 90 度以及随机裁剪。随机旋转同时作用于图像和掩码，保持对应关系。旋转角度从 1 到 4 之间随机取整，对应旋转 90 度的整数倍。

```python
def random_rotate(img, mask):
    rand_rotate_num = tf.random.uniform(shape=(), minval=1, maxval=5, dtype=tf.int32)
    img = tf.image.rot90(img, k=rand_rotate_num)
    mask = tf.image.rot90(mask, k=rand_rotate_num)
    return img, mask
```

裁剪采用中心裁剪，随机决定是否使用 0.5 的中心区域，增加模型对血管位置的鲁棒性。

```python
def random_crop(img, mask):
    # 一半概率裁剪中央一半，另一半保持原图
    if tf.random.uniform(shape=()) >= 0.5:
        return crop_central(img, 0.5), crop_central(mask, 0.5)
    else:
        return img, mask
```

我们用生成器从磁盘读取图片，通过 tf.data 构建输入流。训练集与验证集按 0.8:0.2 划分，使用 dataset.take 和 dataset.skip，配合 shuffle 使训练顺序随机化。

### 2.3 模型训练与超参数调优
我们采用基于 VGG16 编码器的 U-Net 架构。VGG16 作为预训练骨干提供强特征，解码器通过反卷积逐层上采样并与编码器特征拼接。跳跃连接把编码器不同深度的特征图直接输入解码器，帮助恢复血管的细小结构。材料里使用 ImageNet 预训练权重初始化 VGG16，然后接上自定义的解码器。解码器使用 Conv2DTranspose 做上采样，Conv2D 做特征融合，并使用 LeakyReLU 和 BatchNormalization 保持训练稳定。

```python
def create_model(input_shape):
    VGG16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    last_layer = VGG16.output

    # 上采样 + 跳跃连接示意
    model = Conv2DTranspose(256, (2,2), strides=(2,2))(last_layer)
    concat_1 = concatenate([model, VGG16.get_layer("block5_conv3").output])
    ...
    model = Conv2D(1, (3,3), padding='same', activation='sigmoid')(model)
    return Model(VGG16.input, model)
```

模型从最深的 block5 开始反卷积，每次上采样一步就与对应编码器层拼接一次，然后用 3x3 卷积融合特征。最终输出一层单通道 sigmoid，表示每个像素是血管的概率。

训练参数：
- 批次大小 4，图像 512×512，显存受限所以批次较小；
- 优化器 Adam，学习率 1e-4；
- 损失 BinaryCrossentropy，指标 dice_coef 与 accuracy；
- 训练轮数 30，steps_per_epoch 按数据集大小计算；
- 使用 ModelCheckpoint 保存验证准确率最好的权重。

模型构建时先读取图像，再通过 data_augmentation pipeline 进行随机旋转和裁剪。tf.data 的 map 操作使用 num_parallel_calls=AUTO 加速预处理，训练集重复且打乱，保证每个 epoch 都能看到不同顺序的数据。

```python
vgg16_unet_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=BinaryCrossentropy(from_logits=True),
    metrics=[dice_coef, 'accuracy']
)

vgg16_unet_model.fit(
    train_dataset,
    batch_size=BATCH_SIZE,
    steps_per_epoch=train_size,
    epochs=30,
    validation_data=val_dataset
)
```

自定义 Dice 系数实现如下，用于训练监控：

```python
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
```

我们没有做复杂的超参数网格搜索，而是依赖数据增强和预训练编码器来稳定训练。材料里也没有显示学习率调度，但 Adam 的自适应特性在初期训练中足够稳定。训练时使用 from_logits=True 的 BinaryCrossentropy，配合最后的 sigmoid 激活，能保持梯度稳定。

### 2.4 模型测试与指标报告
训练结束后，我们加载最佳权重，对验证集样本做预测并可视化。

```python
model = create_model(IMAGE_SHAPE)
model.load_weights('/kaggle/working/vgg16-unet-bestpoint.weights.h5')

pred = model.predict(image)
pred = conv_func(pred)  # 0.5 阈值二值化
```

模型输出是概率图，我们用一个向量化函数把大于 0.5 的概率判为血管、其余判为背景。

对比预测掩码与真实掩码，计算 Dice 值：

```python
dice = (2. * intersection + K.epsilon()) / (K.sum(label_f) + K.sum(p_f) + K.epsilon())
print("DICE:", dice)
```

材料里展示了预测结果与真实掩码的并排图，血管的连续性和细小分支是主要关注点。从结果看，模型对主干的粗血管识别较好，细小的末梢血管有时会被漏掉。为最终提交，我们对测试集全部图像进行推理，生成二值掩码并保存为所需格式。

```python
# 对测试集循环推理
for i, test_image in enumerate(test_images):
    out = model.predict(preprocess(test_image)[None])
    out = (out > 0.5).astype(np.uint8)
    save_mask(out, f"pred_{i}.png")
```

## 三、总结

本文中我们完成了 SAI Vessel Segmentation 2 的解题流程。我们首先通过 EDA 确认了 512×512 的眼底图像与二值血管掩码的数据配对情况；其次进行了图像归一化、掩码二值化和随机旋转、裁剪的数据增强；接着搭建了基于 VGG16 编码器的 U-Net 模型，使用 Dice 系数监控分割质量并以 Adam 1e-4 训练 30 轮；最后通过加载最佳权重对验证集和测试集进行推理并生成掩码预测。

整体思路体现了医学图像分割中预训练骨干加跳跃连接的经典范式，并在类别不平衡的血管线上保持了较好的结构保持能力。这个比赛给我们的启发有三点：一是分割任务的评估必须使用兼顾正样本重叠的指标，单纯准确率会掩盖漏检；二是掩码数据在进入网络前要统一二值化，避免灰度噪声干扰训练；三是对纤细目标，解码器配合多个尺度的跳跃连接比单一尺度的特征更有表现力，这也是 U-Net 之所以在医学分割里经久不衰的原因。
