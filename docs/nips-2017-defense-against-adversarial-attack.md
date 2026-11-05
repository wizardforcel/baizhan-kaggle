# NIPS 2017 Defense Against Adversarial Attack

## 一、比赛简介

### 1.1 比赛背景
NIPS 2017 对抗学习挑战是 2017 年 NIPS 会议上由 Google Brain 团队发起的一组竞赛，共包含三个相关比赛：非定向攻击、定向攻击和**防御**。背景是这样一个现象：现在的图像分类器很容易被「对抗样本」骗过。对抗样本就是在原图上加一点肉眼几乎察觉不到的小改动，改动之后的图和原图看起来一模一样，但分类器却会以很高的置信度给出完全错误的类别。

这个现象很反直觉：人眼觉得还是同一张图，模型却彻底认不出来了。这说明模型的「理解」和人类不同——它可能依赖了一些我们看不见的脆弱线索。防御挑战的目标，就是训练出对这些对抗攻击更鲁棒（robust）的分类器，也就是说，即使输入被恶意篡改，它也能尽量保持正确的分类。

主办方把这套机制设计成了一个循环：攻击比赛的参赛者负责生成对抗样本，防御比赛的参赛者负责训练抗攻击的模型，双方互相对抗、互相检验。

### 1.2 比赛目标
防御比赛使用 ImageNet 类别的 1001 类图像，图像被缩放到 299x299 的 RGB 格式。开发的材料包括：

- `images.csv`：每张图的 `ImageId`、`TrueLabel`（真实类别）和 `TargetClass`（攻击方指定的目标类别）；
- `categories.csv`：类别编号对应的名称；
- `Inception V3` 预训练模型权重：官方提供的被攻击基准分类器。

我们要做的是训练（或微调）一个分类器：它在干净的原始图像上保持高准确率，同时在被对抗扰动后的图像上不轻易犯错。

比赛中通常还会提供 `sample_defense_submission`，用于检查提交格式，说明提交内容就是每张测试图一个预测的类别编号。

### 1.3 评价标准
防御挑战的评分主线是：攻击方生成对抗图像，喂给防御方训练出的模型，看模型还能不能分对。最终排行榜反映的就是模型在攻击面前的平均分类准确率，越高越好。因为攻击方会用最强的攻击手段，所以只对某一种攻击鲁棒是不够的，模型要经得起多种扰动方式的考验。

需要特别说明的是：这里我们手上只有官方的「Getting Started」教程材料，它完整给出了攻击的生成代码，但对防御本身只描述了目标。所以下面的解题过程，是我们依据教程里 FGSM 攻击的代码、结合通用的对抗训练思想整理出的**代表性流程**，不是某个具体冠军方案的复刻。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**先会制造攻击，才能理解该怎么防御。** 所以我们先把 FGSM 攻击在 Inception V3 上跑通，观察模型被耍的过程；再把对抗样本混进训练，让模型见识「敌人」，从而变得更皮实。下面按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）
首先加载环境，导入官方教程里用到的库并定义参数。
```python
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.misc import imread
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim

checkpoint_path = "../input/inception-v3/inception_v3.ckpt"
input_dir = "../input/nips-2017-adversarial-learning-development-set/images/"

max_epsilon = 16.0        # 允许的最大像素扰动（0~255 尺度）
image_width = 299
image_height = 299
batch_size = 16
num_classes = 1001
# 输入归一化到 [-1,1] 后，16/255 的扰动要乘以 2 换算
eps = 2.0 * max_epsilon / 255.0
batch_shape = [batch_size, image_height, image_width, 3]
```
这里有个换算要特别注意：图像像素原本在 0～255，教程把它线性映射到 -1～1。所以在 -1～1 的空间里，16/255 的扰动上限要写成 `2.0 * 16.0 / 255.0`。漏掉这个 2，扰动幅度就减半了。

接着我们写一个函数，按批把目录里的 PNG 读进来并归一化：
```python
def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    for filepath in sorted(tf.gfile.Glob(os.path.join(input_dir, '*.png'))):
        with tf.gfile.Open(filepath, "rb") as f:
            # 乘以 2/255 再减 1，把 [0,255] 映射到 [-1,1]
            images[idx, :, :, :] = imread(f, mode='RGB').astype(np.float) \
                * 2.0 / 255.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images
```
用生成器的好处是：开发集图像可能很多，逐批加载就不会一次把内存占满。

然后我们读取类别表和每张图的标签：
```python
categories = pd.read_csv(
    "../input/nips-2017-adversarial-learning-development-set/categories.csv")
image_classes = pd.read_csv(
    "../input/nips-2017-adversarial-learning-development-set/images.csv")

filenames, images = next(load_images(input_dir, batch_shape))
meta = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(
    image_classes, on="ImageId")
true_classes = meta["TrueLabel"].tolist()

# 把类别编号翻译成名称，例如 388 -> giant panda
print(pd.DataFrame({"CategoryId": true_classes})
      .merge(categories, on="CategoryId")["CategoryName"].tolist())
```
EDA 阶段的重点观察是：**原始图像用 Inception V3 分类，绝大多数是对的**（比如一张图正确识别为 giant panda / 大熊猫）。这给了我们一个干净的出发点——我们的分类器本身不差，问题全出在扰动上。

### 2.2 数据清洗
对抗攻防对「输入范围」极其敏感，所以清洗环节的核心是保证所有图像都落在单调一致的区间里。

我们用一个定制的归一化变换，保证送入模型前的张量一定在 [-1, 1]：
```python
class LeNormalize(object):
    """按 Google Inception 的风格，把 [0,1] 张量线性映射到 [-1,1]"""
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)   # (x-0.5)*2
        return tensor
```
同时，在生成对抗样本之后，我们必须把结果**裁剪**回合法的像素范围，否则图像会出现越界值，后续无论攻击还是防御都会失真：
```python
# 攻击/防御产出的图像都裁剪到 [-1, 1]
x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
```
另外一个常规检查是确认图像尺寸：Inception V3 期望 299x299，如果直接喂 224x224 或其他尺寸，卷积层会报错。所以我们统一在调用模型之前缩放和中心裁剪。

### 2.3 模型训练与超参数调优
（见下文）