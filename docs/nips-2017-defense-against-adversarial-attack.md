# NIPS 2017 Defense Against Adversarial Attack

## 一、比赛简介

### 1.1 比赛背景
NIPS 2017 对抗学习挑战包含三个相关竞赛：非定向攻击、定向攻击和防御。图像分类器很容易被精心制作的对抗图像欺骗，这些图像对原始图像添加几乎人眼不可察觉的小改动，却导致分类器以高置信度给出错误的类别。防御挑战的目标是训练在面对前两个挑战产生的对抗攻击时仍保持正确分类的模型。

### 1.2 比赛目标
防御挑战要求参赛者使用训练集图像和标签开发一个分类模型，在对抗扰动下保持高准确率。比赛提供开发集图像、类别映射和对抗扰动示例供调试。最终评估使用隐藏的对抗攻击生成并在测试集上进行。

### 1.3 评价标准
防御挑战的评估基于模型在面对攻击图像时的分类准确率。准确率越高越好。比赛与攻击挑战相互关联：攻击挑战的评分取决于它们能多好地欺骗防御模型，而防御挑战的评分取决于它们能多好地抵抗攻击。

---

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先查看开发集图像和类别。
```python
import pandas as pd
categories = pd.read_csv("../input/nips-2017-adversarial-learning-development-set/categories.csv")
image_classes = pd.read_csv("../input/nips-2017-adversarial-learning-development-set/images.csv")
```
开发集中包含 1001 个类别的 299x299 RGB 图像。我们使用 Inception V3 模型作为基准分类器。
```python
from tensorflow.contrib.slim.nets import inception
model = inception.inception_v3(x_input, num_classes=1001, is_training=False)
```

### 2.2 数据清洗
图像数据已预处理至 [-1, 1] 范围。我们对输入进行归一化并裁剪。
```python
images = images.astype(np.float) * 2.0 / 255.0 - 1.0
images = np.clip(images, -1.0, 1.0)
```
对抗图像在生成后也做相同裁剪，确保输入有效。

### 2.3 模型训练与超参数调优
防御模型通常采用对抗训练。我们展示如何生成 FGSM 攻击作为训练信号。
```python
from cleverhans.attacks import FastGradientMethod
fgsm = FastGradientMethod(model)
eps = 2.0 * 16.0 / 255.0
x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)
```
训练中使用原始图像和对抗图像混合输入，提升模型鲁棒性。
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([x_clean, x_adv], y, epochs=10, batch_size=32)
```
在超参数上，我们尝试减小学习率并增加对抗样本比例。

### 2.4 模型测试与指标报告
在开发集上评估模型对原始图像和对抗图像的准确率。
```python
pred_clean = sess.run(predicted_labels, feed_dict={x_input: clean_images})
pred_adv = sess.run(predicted_labels, feed_dict={x_input: adv_images})
acc_clean = np.mean(pred_clean == true_classes)
acc_adv = np.mean(pred_adv == true_classes)
print("Clean accuracy:", acc_clean)
print("Adversarial accuracy:", acc_adv)
```
防御模型在原始图像上保持高准确率的同时，对抗准确率显著提升。

---

## 三、总结
本文中我们介绍了 NIPS 2017 防御对抗攻击挑战。我们首先对开发集图像和类别进行初步探索，其次说明了对抗图像的生成原理，接着展示了基于对抗训练的稳健模型训练方法，最后在开发集上验证了模型对原始和对抗样本的分类效果。该挑战强调将鲁棒性纳入模型设计的重要性。
