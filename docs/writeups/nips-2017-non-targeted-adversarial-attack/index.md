# NIPS 2017 Non-Targeted Adversarial Attack

## 一、比赛简介

### 1.1 比赛背景
NIPS 2017 对抗学习挑战由 Google Brain 团队发起，包含三个互相咬合的比赛：非定向攻击、定向攻击和防御。这个比赛是其中的**非定向攻击**（Non-Targeted Adversarial Attack）。

所谓对抗样本，是指在原图上叠加一层精心计算的、幅度极小的扰动。小到什么程度呢？把原图和对抗图并排放在一起，人眼几乎分辨不出差别，但图像分类器会以很高的置信度给出完全错误的答案。非定向攻击的要求最宽松：**只要模型分错就行**，错成猫、错成狗、错成汽车都算成功，不指定错误类别。相比之下，定向攻击要求模型必须认成攻击者指定的某一类，难度更高。

比赛基于 ImageNet 的 1001 类图像，使用 Inception V3 作为被攻击的基准分类器。参赛者要提交一批对抗图像，主办方会把这些图像放进防御挑战训练出的模型上评估。换句话说，攻击方和防御方是在同一个战场上互相出题。

### 1.2 比赛目标
比赛提供开发集（development set），包含：

- `images/` 目录下的一批 299x299 PNG 图像；
- `images.csv`：每张图的 `ImageId`、`TrueLabel`（真实类别）和 `TargetClass`（定向攻击才用得到的目标类别）；
- `categories.csv`：1001 个类别编号到名称的映射。

我们要生成每张图的对抗版本，并保证扰动不超过规定的 L-infinity 上限。提交的是一批图像文件，文件名与原图对应。评估时，这些图像会被送进目标分类器，看有多少被分错。

这里解释一下 L-infinity 范数：它衡量的是「所有像素改动量里最大的那一个」。比如限制为 16/255，意思是任何一个像素通道的改动都不能超过 16 个灰度级（灰度级范围是 0～255）。这个约束很严格，但也正因为严格，攻击才显得可怕——只改这么一点，模型就崩了。

### 1.3 评价标准
非定向攻击的得分取决于**攻击成功率**：对抗图像被目标分类器分错的比例。成功率越高越好。评估时通常还要求扰动在 epsilon=16/255 以内，超出预算的提交无效。因为防御模型每届都在变强，所以攻击方法也必须不断迭代；材料里也提到，攻击挑战的评分取决于它们能多好地欺骗防御挑战中的分类器。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**沿着分类损失对输入图像的梯度方向走，一步步把图像推离正确类别。** 这个方法叫快速梯度符号法（FGSM），迭代版本叫 I-FGSM。下面按 EDA、数据清洗、模型训练与调参、测试与指标报告的框架来叙述。

### 2.1 数据探索（EDA）
我们首先把开发集的元数据读进来，看看标签结构。
```python
import pandas as pd
import numpy as np

# 每张图的真实类别和目标类别（定向攻击用）
image_classes = pd.read_csv(
    "../input/nips-2017-adversarial-learning-development-set/images.csv")

# 类别编号 -> 人类可读的名称
categories = pd.read_csv(
    "../input/nips-2017-adversarial-learning-development-set/categories.csv")

print(image_classes.head())
print(image_classes['TrueLabel'].nunique())   # 开发集里出现的真实类别数
```
`TrueLabel` 是这张图本来的类别，取值范围是 1～1000（Inception V3 的输出有 1001 维，多出来的一维是背景类）。`ImageId` 是文件名去掉 `.png` 后缀，提交时要靠它和原图对上。

接着我们确认图像尺寸和取值范围：
```python
from PIL import Image

# 随便打开一张图，确认是 RGB、299x299
img = Image.open("../input/nips-2017-adversarial-learning-development-set/images/"
                 + image_classes['ImageId'][0] + ".png")
print(img.size, img.mode)     # 期望 (299, 299) RGB
```
尺寸必须是 299x299，因为 Inception V3 的全连接层前面是 8x8 的全局平均池化，输入尺寸不对会直接报维度错误。

### 2.2 数据清洗
对抗攻击的「清洗」其实是对输入的规范化：把所有图像变成模型期望的格式，并且始终把像素限制在合法范围内。

先做归一化。Inception 系列模型的训练输入是 [-1, 1]，所以我们把 [0, 1] 的张量线性映射过去：
```python
import torchvision.transforms as transforms

class LeNormalize(object):
    """把 [0,1] 的像素线性映射到 [-1,1]，这是 Inception 的标准输入"""
    def __call__(self, tensor):
        for t in tensor:          # 逐通道处理
            t.sub_(0.5).mul_(2.0) # (x - 0.5) * 2
        return tensor

def default_inception_transform(img_size):
    return transforms.Compose([
        transforms.Scale(img_size),        # 缩放到 299
        transforms.CenterCrop(img_size),   # 中心裁剪，保证方形
        transforms.ToTensor(),             # PIL -> [0,1] 张量
        LeNormalize(),                     # [0,1] -> [-1,1]
    ])
```

然后是扰动预算的换算。比赛说的 epsilon 是 0～255 尺度上的，而模型输入是 [-1, 1]，范围宽度是 2，所以要乘 2：
```python
max_epsilon = 16.0
# 0~255 尺度的 16 对应 [-1,1] 尺度的 2*16/255
eps = 2.0 * max_epsilon / 255.0

# 生成对抗样本后必须裁剪回 [-1,1]，否则会出现非法像素
x_adv = torch.clamp(x_adv, -1.0, 1.0)
```
这个裁剪步骤非常重要。如果不裁剪，某些像素会被推到 1 以上或 -1 以下，保存成图片时会被截断，实际扰动可能超过预算，提交就无效了。

### 2.3 模型训练与超参数调优
这里不是训练一个分类器，而是加载一个已经训练好的 Inception V3，并对输入图像求梯度。
```python
import torch
from torch import autograd
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn

# 下载/加载 ImageNet 预训练 Inception V3
model = inception_v3(pretrained=True)
model.eval()                     # 攻击时关闭 dropout 和 batch norm 更新
loss_fn = nn.CrossEntropyLoss()
```
非定向 I-FGSM 的公式是：

`x_adv = x + alpha * sign(∇x J(x, y_true))`

也就是沿着增大真实类别损失的方向走。材料中的实现使用 10 步迭代，每一步把梯度归一成符号，再把累计扰动截断到 epsilon 范围内：
```python
class AttackIterative:
    def __init__(self, max_epsilon=16, num_steps=10):
        self.eps = 2.0 * max_epsilon / 255.0
        self.num_steps = num_steps
        # 把总扰动预算平均分给每一步，避免第一步走太远
        self.step_alpha = self.eps / self.num_steps
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def non_target_attack(self, model, x):
        input_var = autograd.Variable(x, requires_grad=True)
        for step in range(self.num_steps):
            zero_gradients(input_var)          # 每一轮重新求梯度
            output = model(input_var)
            # 第一轮取当前模型最相信的类别，之后不断把它推远
            target = output.data.max(1)[1]
            loss = self.loss_fn(output, target)
            loss.backward()

            # L-infinity 约束：只保留每个像素梯度的正负号
            grad_step = self.step_alpha * torch.sign(input_var.grad.data)
            candidate = input_var.data + grad_step
            total = torch.clamp(candidate - x, -self.eps, self.eps)
            # 既满足扰动预算，又满足合法像素范围
            input_var.data = torch.clamp(x + total, -1.0, 1.0)
        return input_var.data
```
这里的 `num_steps=10` 和 `max_epsilon=16` 是材料中给出的默认配置。步数越多，梯度方向的近似通常越好，但每张图要多做几次前向和反向，速度更慢。我们可以把 `num_steps` 在 5、10、20 之间比较；epsilon 则应严格遵守比赛上限，不能为了分数偷偷放大。

I-FGSM 的关键是「每一步都从原图出发限制累计扰动」，而不是让每一步继续无限叠加。如果只写 `input_var += grad` 而不做 `clamp(total, -eps, eps)`，跑几十步后图像已经不再是「肉眼相似的原图」，那就不是合规的对抗样本了。

材料还实现了定向版本 I-TFGSM：把梯度方向反过来，最小化指定假类别的损失，公式是 `x_adv = x - alpha * sign(∇x J(x, y_fake))`。但这篇 writeup 的主线是非定向攻击，我们只把定向版本作为对照，不把它混进主提交。

### 2.4 模型测试与指标报告
生成对抗图像后，我们用同一个 Inception V3 重新分类，比较原图和对抗图的预测。
```python
def predict(model, images):
    with torch.no_grad():
        logits = model(images)
        return logits.argmax(dim=1).cpu().numpy()

clean_pred = predict(model, images)
adv_images = attack.non_target_attack(model, images)
adv_pred = predict(model, adv_images)

# 非定向攻击只关心有没有分错
clean_acc = np.mean(clean_pred == true_labels)
adv_acc = np.mean(adv_pred == true_labels)
attack_success = 1.0 - adv_acc
print('clean accuracy:', clean_acc)
print('adversarial accuracy:', adv_acc)
print('attack success rate:', attack_success)
```
教程中的示例显示，原始的大熊猫被正确识别为 giant panda，加上扰动后变成 Samoyed；原始贡多拉被正确识别，扰动后变成 dam。原图和对抗图看起来几乎一样，但预测类别完全换了，这就是攻击成功。

我们还要检查扰动是否真的符合预算：
```python
# 在模型输入尺度上检查最大改动量
max_delta = (adv_images - images).abs().max().item()
print('max perturbation:', max_delta)
assert max_delta <= eps + 1e-6
```
最后把对抗图像保存下来，文件名沿用原始 `ImageId`，并随机打开几张肉眼检查。若图像颜色明显变化，通常说明归一化或裁剪写错了，而不是攻击更强。

在另一个教程 notebook 中，作者还用 Fashion-MNIST 的 28x28 图像和逻辑回归演示了「最大化输出与真实标签差异」的攻击思路。这是很好的教学补充，但我们最终采用的是 Inception V3 + I-FGSM，因为它与 NIPS 2017 比赛的输入和评估模型一致。

---

## 三、总结
本文中我们完成了 NIPS 2017 非定向对抗攻击的入门实现。我们首先确认了 1001 类、299x299 图像和 `TrueLabel` 元数据，其次把输入统一归一化到 [-1,1] 并把 16/255 的扰动预算换算到模型尺度，接着加载预训练 Inception V3，用 10 步迭代快速梯度符号法逐步增大分类损失，同时把累计扰动限制在 L-infinity 范围内；最后比较干净准确率、对抗准确率和攻击成功率，并检查最大像素改动是否超预算。

这个比赛最重要的启发是：**攻击不需要改变图像的语义，只需要利用模型的梯度弱点**。对人来说，图像还是大熊猫；对模型来说，几个看不见的小数点改动足以让它变成萨摩耶。另一个实际经验是，输入归一化、epsilon 换算和 `clip` 这些工程细节与梯度公式同样重要，任何一个写错都会让「看起来能跑」的攻击变成不合规的提交。