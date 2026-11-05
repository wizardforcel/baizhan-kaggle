# NIPS 2017 Non-Targeted Adversarial Attack

## 一、比赛简介

### 1.1 比赛背景
NIPS 2017 对抗学习挑战中的非定向攻击（Non-Targeted Adversarial Attack）比赛，要求参赛者生成对抗图像，使得图像分类器把原本正确的图像判成任意一个错误类别即可，不限定要骗成哪一类。比赛基于 ImageNet 的 1001 类图像，使用 Inception V3 作为被攻击的分类器。这类研究揭示了深度模型的一个根本弱点：人眼几乎看不出差别的小扰动，就能让模型以高置信度输出错误结果。

### 1.2 比赛目标
给定开发集的 299x299 图像和真实标签，生成修改后的图像，要求修改幅度受 L-infinity 范数约束（每个像素的改动不超过某个 epsilon）。提交的对抗图像随后会被放进防御挑战中训练出的模型上进行评估。也就是说，攻击的目标是让尽可能多的图像被分错。

### 1.3 评价标准
非定向攻击的得分取决于对抗图像欺骗目标分类器的比例，也就是攻击成功率。成功率越高，说明攻击越强。图像扰动通常限制在 epsilon=16/255 的范围内，超出范围的改动无效。

---

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先加载开发集图像、类别表与标签。
```python
import pandas as pd
import numpy as np

# 读取类别名称和每张图的真实标签、目标标签
categories = pd.read_csv("../input/nips-2017-adversarial-learning-development-set/categories.csv")
image_classes = pd.read_csv("../input/nips-2017-adversarial-learning-development-set/images.csv")
```
开发集使用 `images.csv` 中的 `TrueLabel` 作为原始类别，图像统一缩放到 299x299 并归一化到 [-1, 1]。
```python
def default_inception_transform(img_size):
    # 缩放 -> 中心裁剪 -> 转张量 -> 归一到 [-1,1]
    return transforms.Compose([
        transforms.Scale(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        LeNormalize(),
    ])
```
`LeNormalize` 把像素从 [0,1] 线性映射到 [-1,1]，这是 Google Inception 系列的标准输入格式。

### 2.2 数据清洗
对抗攻击对输入范围很敏感，所以我们统一裁剪。
```python
# 每个像素最多改动 16/255，先换算到 [-1,1] 尺度
max_epsilon = 16.0
eps = 2.0 * max_epsilon / 255.0

# 生成对抗样本后必须裁剪回合法像素范围
input_adv = torch.clamp(input_adv, -1.0, 1.0)
```
这里乘 2 是因为输入范围宽度是 2（-1 到 1）。如果不裁剪，提交的图像会出现越界像素，评估时会出问题。

### 2.3 模型训练与超参数调优
我们攻击的是预训练的 Inception V3，不需要自己训练，只需要把梯度反传回输入图像。
```python
import torch
from torch import autograd
from torch.autograd.gradcheck import zero_gradients

class AttackIterative:
    def __init__(self, max_epsilon=16, norm=float('inf'), num_steps=10):
        self.eps = 2.0 * max_epsilon / 255.0
        self.num_steps = num_steps
        # 每一步的步长，把总预算平均分成 num_steps 份
        self.step_alpha = self.eps / self.num_steps
        self.loss_fn = torch.nn.CrossEntropyLoss()
```
核心是迭代快速梯度符号法（I-FGSM）：每一步沿损失对输入的梯度方向走一小步。
```python
def non_target_attack(self, model, x, targets):
    input_var = autograd.Variable(x, requires_grad=True)
    for step in range(self.num_steps):
        zero_gradients(input_var)          # 清空上一步的梯度
        output = model(input_var)          # 前向，得到各类别分数
        # 非定向攻击：把当前预测的最可能类别当作要“远离”的目标
        targets_var = output.data.max(1)[1]
        loss = self.loss_fn(output, targets_var)
        loss.backward()                    # 求损失对输入图像的梯度
        # 只取梯度的符号，保证每一步改动幅度一致
        normed_grad = self.step_alpha * torch.sign(input_var.grad.data)
        step_adv = input_var.data + normed_grad
        # 把累计扰动限制在 [-eps, eps] 内
        total_adv = torch.clamp(step_adv - x, -self.eps, self.eps)
        input_var.data = torch.clamp(x + total_adv, -1.0, 1.0)
    return input_var.data
```
关键超参数包括 `num_steps`（迭代步数）、`max_epsilon`（扰动上限）和步长。步数越多攻击越强，但也越慢。材料中的实现默认 10 步，并把 epsilon 固定为 16/255。需要注意，第一轮迭代时 `targets_var` 取的是模型自己预测的类别，之后沿这个类别的损失做梯度上升。

### 2.4 模型测试与指标报告
生成对抗图像后，我们让原始 Inception V3 重新分类，统计预测结果。
```python
# 原始图像、非定向对抗图像的预测
pred_clean = predict(model, images)
pred_adv = predict(model, adv_images)

# 攻击成功率 = 对抗图像被分错的比例
success_rate = (pred_adv != true_classes).mean()
print("Attack success rate:", success_rate)
```
经验上，原始图像几乎全部被正确分类，而加了扰动后，模型会把熊猫认成萨摩耶、把吊舱认成水坝、把蝴蝶认成环蝶等。肉眼对比两张图几乎看不出差别，这正是对抗样本可怕的地方。

---

## 三、总结
本文中我们完成了 NIPS 2017 非定向对抗攻击的思路与实现。我们首先了解开发集的图像与标签格式，其次把图像统一归一化到 [-1,1] 并设定 16/255 的扰动预算，接着用迭代快速梯度符号法对 Inception V3 的输入求梯度并逐步叠加扰动，最后统计对抗图像被分错的比例来衡量攻击效果。这个比赛让我们直观看到：模型的高准确率并不等于真正理解了图像。
