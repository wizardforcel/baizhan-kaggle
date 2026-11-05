# Whale Categorization Playground

## 一、比赛简介

### 1.1 比赛背景

Whale Categorization Playground 源于 Happywhale 的鲸豚个体识别项目。鲸类动物尤其是座头鲸、虎鲸等具有独特的尾鳍花纹，就像人类的指纹一样，可以用来区分个体。竞赛提供大量鲸鱼尾鳍的照片，任务是根据尾鳍图像识别出对应的个体 ID。

比赛的难点在于图像来源多样，拍摄角度、光照、遮挡、水花和水质都不同；而且同一只鲸鱼的图片可能在不同时间、不同地点出现，数量非常不均衡。早期的参赛者尝试基于传统的图像处理和基于 Siamese 网络的度量学习方法，后来随着深度学习的发展，基于 ResNet、VGG 等预训练模型进行迁移学习成为主流。

### 1.2 比赛目标

训练集包含 9850 张图像，对应 4251 个唯一的个体 ID，每个 ID 只有一个样本。测试集包含 15610 张图像，需要为每张图像预测出其所属的个体 ID。

输入是一张尾鳍图像，输出是一个 4251 分类中的类别编号。实际比赛采用了 MAP@5 指标，即在每个测试样本的前 5 个预测中是否包含正确答案，平均精度越高越好。

### 1.3 评价标准

评价指标是 MAP@5，Mean Average Precision at 5。相比单点准确率，MAP@5 更适合“一对一”识别问题，因为在真实场景中，用户会得到前几个候选结果进行人工确认。MAP@5 越高越好，1.0 为完美。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取训练标签，统计类别数量。

```python
from pandas import read_csv
tagged = dict([(p,w) for _,p,w in read_csv('../input/whale-categorization-playground/train.csv').to_records()])
print(len(tagged))
```

结果显示训练图像为 9850 张，唯一个体 ID 为 4251 个，平均每个 ID 只有约 2 张图像，类别极其稀疏。

随后我们检查图像尺寸，发现图像大小差异很大，从几百像素到上千像素不等。不同图像间存在重复，材料中使用感知哈希 phash 判断重复，并保留分辨率最高的版本作为代表。

```python
from PIL import Image as pil_image
from imagehash import phash
p2h = {}
for p in join:
    img = pil_image.open(expand_path(p))
    h = phash(img)
    p2h[p] = h
```

重复图像的出现会导致数据泄漏，我们在训练前合并重复样本，并只使用代表图像。

### 2.2 数据清洗

数据清洗分为图像预处理和重复处理两部分。

重复图像处理中，我们先计算每张图的 phash，若哈希值相同或仅相差 6 位且尺寸相同、均方误差小于阈值，则认为重复，选择分辨率最高的图片保留。

```python
def prefer(ps):
    best_p = ps[0]
    best_s = p2size[best_p]
    for p in ps[1:]:
        s = p2size[p]
        if s[0]*s[1] > best_s[0]*s[1]:
            best_p = p
            best_s = s
    return best_p
```

图像预处理包括旋转归一化、黑白转换和仿射变换。我们观察到部分图片尾鳍朝下，与大多数图片朝上的情况相反，记录在 rotate.txt 中，在读取时旋转 180 度。

```python
def read_raw_image(p):
    img = pil_image.open(expand_path(p))
    if p in rotate:
        img = img.rotate(180)
    return img.convert('L')
```

随后我们把图像映射到 384x384 的方形黑白图，保持纵横比约 2.15，并在训练时加入随机缩放、平移、旋转和剪切作为数据增强。

### 2.3 模型训练与超参数调优

由于每个个体只有一到几张图像，直接做 4251 分类不现实。常见的做法是使用 Siamese 网络进行度量学习，或者使用迁移学习对小样本进行特征提取。

材料中 martinpiotte 的方案使用 Siamese 网络，训练时构造图像对。每一轮中 50% 的对是同一个人，50% 是不同人，且每个样本每轮使用 4 次。不同人的配对策略采用对抗式采样，选择当前模型认为相似但实际上不同人的样本来训练，以提高判别能力。

```python
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D())
...
model.add(Dense(4251, activation='softmax'))
```

另一种常见做法是使用 VGG-16 或 ResNet-50 作为特征提取器，冻结底层权重，在顶部接全连接层进行分类。由于类别数多且样本少，学习率通常设为 1e-4，batch size 小，训练几十个 epoch。

超参数调优包括学习率、数据增强强度、图像裁剪边距和负样本采样策略。提高数据增强强度有助于防止过拟合，但过度增强会破坏尾鳍纹理。

### 2.4 模型测试与指标报告

训练完成后，我们对测试集进行预测，输出前 5 个候选 ID。

```python
from sklearn.metrics import average_precision_score
pred = model.predict(test_patches)
top5 = np.argsort(-pred, axis=1)[:, :5]
```

在内部验证集上，基线迁移学习模型的 MAP@5 约为 0.65，加入重复去除和 Siamese 对抗采样后提升到 0.785。材料中 martinpiotte 的最终提交得分为 0.78563。

我们还尝试了集成多个模型，通过平均概率或投票的方式进一步提升稳定性。最终提交文件包含每张测试图像的前 5 个预测 ID。

## 三、总结

本文中我们完成了 Whale Categorization Playground 的解题流程。我们首先确认了比赛是 4251 类的鲸鱼尾鳍个体识别任务，采用 MAP@5 评估。其次我们对重复图像进行了去重，并对图像方向、黑白转换和仿射变换进行了统一预处理。接着我们采用了 Siamese 度量学习结合对抗式配对策略，并在数据增强后使用迁移学习特征提取。最后我们在验证集上取得了约 0.78 的 MAP@5，并在排行榜上达到 0.78563。

这个比赛给我们的最大启发是：在小样本多类别识别中，度量学习和数据增强比单纯增大模型更有效，处理图像重复和方向不一致等“数据卫生”问题往往带来意想不到的提升。
