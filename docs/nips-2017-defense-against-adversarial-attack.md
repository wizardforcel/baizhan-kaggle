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
我们先构造一个 Inception V3 的推理封装，用官方 checkpoint 加载权重。
```python
class InceptionModel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        # 第一次调用时建图，后面复用同一套网络参数
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input, num_classes=self.num_classes,
                is_training=False, reuse=reuse)
        self.built = True
        return end_points['Predictions'].op.inputs[0]
```
官方教程使用 CleverHans 的 `FastGradientMethod` 生成非定向样本。所谓非定向，就是只要求模型分错，不关心它具体错成哪一类：
```python
from cleverhans.attacks import FastGradientMethod

with tf.Graph().as_default():
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    model = InceptionModel(num_classes)
    fgsm = FastGradientMethod(model)
    # 沿着增大分类损失的方向走一步
    x_adv = fgsm.generate(x_input, eps=eps,
                          clip_min=-1.0, clip_max=1.0)
```
真正用于防御时，我们把生成出来的 `x_adv` 和干净图像配成一对，训练分类器同时看两种输入。这个方法叫**对抗训练**，大白话就是让模型在学习过程中不断做模拟考试，而且题目故意带一点最容易让它错的干扰。
```python
# 伪代码：防御训练同时使用干净样本和对抗样本
for clean_images, labels in train_loader:
    adv_images = generate_fgsm(clean_images, labels)

    # 两种输入共用标签，模型要对两种图都做对
    clean_loss = cross_entropy(model(clean_images), labels)
    adv_loss = cross_entropy(model(adv_images), labels)
    loss = 0.5 * clean_loss + 0.5 * adv_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
材料给出的攻击参数是 `max_epsilon=16`，如果使用迭代攻击，可以把总预算拆成多步，每一步只走很小的梯度符号。调参时我们主要关注三个量：epsilon 越大，训练出来的模型通常越鲁棒，但干净图像准确率可能下降；对抗样本在每个 batch 中的比例越高，模型越重视鲁棒性，但训练也更慢；学习率过大会让模型在两种目标之间来回震荡。因此我们从小 epsilon、小学习率开始，逐步增加攻击强度。

防御模型的调参不能只看干净验证集准确率。如果只在干净图像上选模型，很可能选出一个「平时考 100 分、遇到攻击就交白卷」的模型。我们要同时记录干净准确率和对抗准确率，并倾向于选择两者都不差的检查点。

### 2.4 模型测试与指标报告
我们先用原始 Inception V3 测量攻击前后的变化，确认评估脚本本身没有问题。
```python
with tf.Graph().as_default():
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        _, end_points = inception.inception_v3(
            x_input, num_classes=num_classes, is_training=False)
    predicted_labels = tf.argmax(end_points['Predictions'], 1)

    # 用同一个 checkpoint 对三种输入分别预测
    predicted_classes = sess.run(predicted_labels,
                                 feed_dict={x_input: images})
    predicted_adv_classes = sess.run(predicted_labels,
                                     feed_dict={x_input: adversarial_images})
```
然后计算两个最基本的指标：
```python
# 干净准确率：原始图像上有多少预测正确
clean_accuracy = np.mean(predicted_classes == true_classes)
# 对抗准确率：加入扰动后仍然预测正确的比例
adv_accuracy = np.mean(predicted_adv_classes == true_classes)
# 攻击成功率就是对抗准确率的补数（非定向情况下）
attack_success = 1.0 - adv_accuracy
print(clean_accuracy, adv_accuracy, attack_success)
```
教程中的示例非常直观：原始图像是 giant panda，模型预测正确；加上几乎看不出的非定向扰动后，预测变成 Samoyed。原始图像是 gondola（贡多拉），扰动后可能变成 dam（堤坝）。这说明干净准确率高并不代表模型具有安全性。

对防御模型，我们按同样的方法测量，但同时在多种扰动强度下测试：
```python
for eps_test in [4.0, 8.0, 16.0]:
    # 测试时也要把 0~255 的 epsilon 换算到 [-1,1]
    scaled_eps = 2.0 * eps_test / 255.0
    adv = fgsm.generate(x_input, eps=scaled_eps,
                        clip_min=-1.0, clip_max=1.0)
    pred = sess.run(predicted_labels,
                    feed_dict={x_input: adv, y_input: true_classes})
    print('epsilon =', eps_test,
          'accuracy =', np.mean(pred == true_classes))
```
如果一个模型只在 epsilon=4 时有效，到 epsilon=16 就完全失效，说明它的鲁棒范围太窄。我们还会按类别统计错误，观察某些类别是否特别脆弱。最后确认提交格式：每行一个图像 ID 和类别预测，列名与 `sample_defense_submission` 一致，行数不变。

---

## 三、总结
本文中我们完成了 NIPS 2017 防御对抗攻击挑战的入门流程。我们首先认识了对抗样本和三项挑战之间的关系，并确认开发集是 299x299 的 1001 类图像；其次把图像归一化到 Inception V3 要求的 [-1,1] 区间，严格检查尺寸和裁剪范围；接着复现了 CleverHans 的 FGSM 攻击，并根据教程整理出「干净样本 + 对抗样本」的对抗训练方案；最后分别报告干净准确率、对抗准确率和攻击成功率，还在多个 epsilon 下测试了模型的鲁棒范围。

这个比赛告诉我们，**模型准确率和模型安全性是两回事**。一个模型在普通测试集上表现完美，仍可能被一条很小的梯度扰动击穿。防御不能靠最后加一个过滤器，而要在训练阶段就让模型见过攻击、学会依赖更稳定的图像线索。材料本身只提供了 Getting Started 代码，所以本文没有虚构一个排行榜分数，而是把可以由材料直接验证的攻击流程和防御评估流程完整整理出来。