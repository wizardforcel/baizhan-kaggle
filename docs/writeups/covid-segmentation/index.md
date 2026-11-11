# COVID-19 肺部 CT 病灶分割（COVID Segmentation）

## 一、比赛简介

### 1.1 比赛背景

新冠病毒感染后，病人的肺部 CT 上会出现一些特征性的阴影。医生看片时主要关注两种病灶：

- **磨玻璃影**（Ground Glass Opacity，简称 GGO）：肺野里一片淡淡的、半透明的白色，像磨砂玻璃，说明肺泡里有渗出但还没完全实变；
- **实变影**（Consolidation）：更白、更实的区域，说明肺泡已经被渗出物填满。

这两种病灶的**面积占比**直接反映病情轻重，医生需要一层一层地勾画出它们的边界，再算「病灶占了肺的百分之几」。这个工作极其耗时：一个病人的 CT 动辄上百层，一层层手工描边要几十分钟，而且不同医生的勾画还会有差异。

这个比赛就是让算法来做这件事：**输入一张肺部 CT 切片，输出一张同样大小的图，标出每个像素属于哪一类**。这类任务在深度学习里叫**语义分割**（semantic segmentation）——「语义」是指给像素赋予含义，「分割」是指把图像切成不同的区域。

比赛的难点集中在三处：

1. **正负样本极度不平衡**。一张切片里绝大多数像素是背景或者正常肺组织，真正的病灶可能只占百分之几甚至不到百分之一。模型如果偷懒全预测成背景，损失也能降得很低，但完全没用。
2. **数据来源杂、标注风格不统一**。训练数据来自两个公开数据集：MedSeg 和 Radiopedia，一个 100 张、一个 829 张，标注精细程度和拍摄参数都不一样。
3. **评测只看两个目标类别**。掩码有四个通道（磨玻璃影、实变影、其他肺组织、背景），但比赛只关心前两类，所以后两类在提交时会被直接丢掉。

### 1.2 比赛目标

数据以五个 `.npy` 文件的形式给出（NumPy 的数组存储格式，读进来就是现成的多维数组）：

- `images_medseg.npy`：MedSeg 的 CT 图像，**100 张**，每张 **512×512**，单通道灰度；
- `masks_medseg.npy`：对应的掩码，形状是 `(100, 512, 512, 4)`，最后一维是 4 个类别通道，值是布尔型；
- `images_radiopedia.npy`：Radiopedia 的 CT 图像，**829 张**，同样是 512×512；
- `masks_radiopedia.npy`：对应掩码，`(829, 512, 512, 4)`；
- `test_images_medseg.npy`：测试图像，**10 张**，512×512。

四个掩码通道的顺序是固定的：`0 = 磨玻璃影`、`1 = 实变影`、`2 = 其他肺组织`、`3 = 背景`。注意第 2、3 类**不需要预测**，模型只要把前两类分准就行。

我们要输出的是**逐像素的二值预测**：把 10 张测试图按 `(图像, 行, 列, 类别)` 的顺序拉平成长表，每行一个像素一个类别，值填 0 或 1。按 10 × 512 × 512 × 2 算，提交文件一共有 5,242,880 行。

### 1.3 评价标准

材料里的代码用 **mIoU**（mean Intersection over Union，平均交并比）来评估模型，并且靠它挑最好的 checkpoint。IoU 的意思是「预测出来的区域」和「真实区域」的**重叠程度**：

```
IoU = 交集面积 / 并集面积
```

如果预测和真实完全重合，交等于并，IoU = 1；如果完全不沾边，IoU = 0。多个类别各算一个 IoU，再取平均，就是 mIoU。**这个指标越高越好**，满分是 1。

用 IoU 而不是准确率（accuracy）是有道理的：假设一张图里 99% 是背景，模型全猜背景也能拿到 99% 的准确率，但 IoU 里病灶那一类的交并比是 0，平均下来分数立刻露馅。**在不平衡数据上，IoU 比准确率诚实得多。**

---

## 二、解题过程

我们的主线方案可以概括成一句话：**用 U-Net 架构 + 在 ImageNet 上预训练过的 EfficientNet 编码器，把 CT 当成单通道图像做四分类分割，最后只提交前两个类别。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步我们把五个数组读进来，先看形状和数值范围：

```python
import numpy as np
import os

prefix = '/kaggle/input/covid-segmentation/'

# 两个来源的训练数据，加一份测试图
images_medseg      = np.load(os.path.join(prefix, 'images_medseg.npy')).astype(np.float32)
masks_medseg       = np.load(os.path.join(prefix, 'masks_medseg.npy')).astype(np.int8)
images_radiopedia  = np.load(os.path.join(prefix, 'images_radiopedia.npy')).astype(np.float32)
masks_radiopedia   = np.load(os.path.join(prefix, 'masks_radiopedia.npy')).astype(np.int8)
test_images_medseg = np.load(os.path.join(prefix, 'test_images_medseg.npy')).astype(np.float32)

# 形状：图像 (N, 512, 512)，掩码 (N, 512, 512, 4)
print(images_medseg.shape, masks_medseg.shape)
print(images_radiopedia.shape, masks_radiopedia.shape)
print(test_images_medseg.shape)

# CT 值的原始范围：Hounsfield 单位，空气约 -1000，骨头可以到 +1000 以上
print(images_radiopedia.min(), images_radiopedia.max())
```

三个数字确认了：MedSeg **100 张**、Radiopedia **829 张**、测试 **10 张**，全部是 512×512 单通道，掩码是 4 通道布尔值。CT 的原始取值是 **Hounsfield 单位（HU）**——放射学里描述组织密度的标准刻度，空气约 -1000，水是 0，骨头能到上千。所以图像的数值范围是几百到上千，和我们平时处理的 0～255 的图片完全不是一个量级。

接着我们做两件更关键的检查：**每类到底占多少像素**，以及**类别之间有没有重叠**：

```python
class_names = ["ground_glass", "consolidations", "lungs_other", "background"]

# 统计每个类别通道里 True 的像素总数，看看不平衡到什么程度
for i, name in enumerate(class_names):
    print(name, np.sum(masks_medseg[..., i]))

# 检查重叠：一个像素同时属于多个类别，会让 argmax 的编码方式出问题
channel_sum = np.sum(masks_medseg, axis=-1)
print("多类别像素:", np.count_nonzero(channel_sum > 1))
print("无类别像素:", np.count_nonzero(channel_sum == 0))

# 单独看两个目标类是否互相重叠
print("磨玻璃 ∩ 实变:", np.count_nonzero(masks_medseg[..., 0] & masks_medseg[..., 1]))
```

结果有两个重要发现：

1. **类别极不平衡**。`background` 的像素数远远超过 `ground_glass` 和 `consolidations`，两个目标病灶加起来只占很小一部分。这解释了为什么后面必须用 IoU 而不是准确率来评估。
2. **两个目标类之间完全没有重叠**（交集为 0）。虽然存在少量像素同时被标成多个通道，但那些都发生在「目标类 vs 其他肺组织」之间，前两类互不冲突。

第二点非常关键，它决定了我们的建模方式：既然两个病灶不会同时出现在同一个像素上，我们就可以把四个通道**压成一个整数标签**（0/1/2/3），做成一个标准的四分类分割问题，而不是训练两个独立的二分类模型。这能让一个网络同时学两类病灶，也顺带学会了「这是背景」的判断。

### 2.2 数据清洗

这份数据的「干净」程度超出预期：**没有 NaN、没有 Inf、图像和掩码数量一一对应、空间尺寸完全一致**。所以常规意义上的「删缺失、补空缺」在这里无事可做。真正的清洗工作集中在**强度归一化**和**标签编码**两件事上。

**第一件：把 CT 强度拉到合理范围。** CT 里有一些极端值（金属伪影、扫描仪噪声）会把整体分布拖歪，所以先做截断，再用分位数估算均值和标准差：

```python
def preprocess_images(images_arr, mean_std=None):
    # 先复制一份，避免改动原始数组
    images = np.asarray(images_arr, dtype=np.float32).copy()

    # 按放射学惯例截断：低于 -1500（空气以下）和高于 500（骨头以上）的都压平
    np.clip(images, -1500, 500, out=images)

    if mean_std is None:
        # 只用 5% 到 95% 之间的像素来算统计量，避开残余的极端值
        min_perc, max_perc = np.percentile(images, [5, 95])
        valid = images[(images > min_perc) & (images < max_perc)]
        mean, std = valid.mean(), valid.std()
    else:
        mean, std = mean_std          # 复用别的数据集算好的参数

    # z-score 标准化：减均值除标准差，把数据变成均值 0、方差 1
    images = (images - mean) / std
    return images, (mean, std)

# 关键：只在 Radiopedia 上算统计量，另外两个集直接套用
images_radiopedia, mean_std = preprocess_images(images_radiopedia)
images_medseg, _            = preprocess_images(images_medseg, mean_std)
test_images_medseg, _       = preprocess_images(test_images_medseg, mean_std)
```

这里有一个很容易做错、但后果很严重的细节：**均值和标准差只在 Radiopedia 上算一次，然后套用到 MedSeg 和测试集上**。材料里算出来的值是均值约 **-451.65**、标准差约 **458.57**。为什么不各自算各自的？因为如果测试集用自己的统计量归一化，就等于「偷看」了测试集的整体分布，本地验证和线上表现的对应关系会被破坏；而且模型在训练时见到的数值尺度必须和推理时一致，否则同一块组织在训练和测试里会变成不同的数字。这是 CT 预处理的通行做法。

**第二件：把四通道掩码压成单通道标签。** 分割网络的输出是「每个像素属于哪一类」的一个整数，所以要把 `(H, W, 4)` 的布尔张量变成 `(H, W)`：

```python
def onehot_to_mask(mask, palette):
    """(H, W, K) 的独热掩码 → (H, W) 的整数标签"""
    # argmax 沿通道取最大值的下标，正好就是类别编号
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    return np.uint8(colour_codes[x.astype(np.uint8)])

palette = [[0], [1], [2], [3]]
masks_radiopedia_recover = onehot_to_mask(masks_radiopedia, palette).squeeze()
masks_medseg_recover     = onehot_to_mask(masks_medseg, palette).squeeze()

print(masks_medseg_recover.shape)   # (100, 512, 512)
```

因为前面已经确认两个目标类不重叠，这一步用 `argmax` 是安全的——不会出现「一个像素被强行归到某一类、丢掉另一个类」的情况。

**第三件：划分训练集和验证集。** 我们沿用了材料里 baseline 的划分方式，好处是后面所有实验都能和它直接对比：

```python
VAL_SIZE = 24
val_indexes   = np.arange(VAL_SIZE)                       # 前 24 张 MedSeg 当验证
train_indexes = np.arange(VAL_SIZE, len(images_medseg))   # 剩下 76 张

# 训练集 = MedSeg 的 76 张 + Radiopedia 全部 829 张
train_images = np.concatenate((images_medseg[train_indexes], images_radiopedia))
train_masks  = np.concatenate((masks_medseg_recover[train_indexes],
                               masks_radiopedia_recover))
val_images   = images_medseg[val_indexes]
val_masks    = masks_medseg_recover[val_indexes]
```

这里有个权衡值得说明：把两个数据集混在一起训练，等于让模型多见 829 张图，泛化更好；但**验证集只用 MedSeg 的前 24 张**，因为测试集也是 MedSeg 风格——用同源的 24 张当验证，才能真实反映「在测试集上会怎样」。如果拿 Radiopedia 当验证，分数可能虚高。

### 2.3 模型训练与超参数调优

**网络结构：U-Net + EfficientNet-B2。** U-Net 是为医学图像分割设计的经典结构，形状像字母 U：左边一串卷积不断下采样（缩小尺寸、提取语义），右边一串上采样（恢复尺寸、还原位置），左右同一层级之间用「跳跃连接」把细节直接送过去。这样既有「全局判断这是肺还是背景」的能力，又有「边界画得准」的能力。

我们不用从零搭，而是用 `segmentation_models_pytorch`（简称 smp）这个库，它把「编码器」和「解码器」拆开，可以自由组合：

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name='efficientnet-b2',   # 编码器：EfficientNet-B2
    encoder_weights='imagenet',       # 用 ImageNet 预训练权重初始化
    in_channels=1,                    # CT 是单通道灰度，不是 RGB 三通道
    classes=4,                        # 输出 4 类：磨玻璃/实变/其他肺/背景
    activation=None,                  # 不做 softmax，交给损失函数里的 CrossEntropy
    encoder_depth=5,                  # 编码器分 5 个尺度
    decoder_channels=[256, 128, 64, 32, 16]  # 解码器每一级的通道数
)
```

`encoder_weights='imagenet'` 是这里最重要的一行：EfficientNet 在几百万张自然图片上学到的「边缘、纹理、形状」特征，对 CT 同样有用。虽然 ImageNet 是彩色的、CT 是灰度的，但底层特征（比如「这里是边界」）是通用的。**在小数据集上，预训练权重往往比多调几个超参数更有用**。

**数据增强：** 训练集只有 905 张图，必须靠增强来「变出」更多样本。我们的增强只有三个，但都很对症：

```python
import albumentations
import cv2

SOURCE_SIZE = 512
TARGET_SIZE = 256

train_augs = albumentations.Compose([
    # 任意角度旋转：CT 切片的拍摄角度不固定，病灶方向没有意义
    albumentations.Rotate(limit=360, p=0.9, border_mode=cv2.BORDER_REPLICATE),
    # 随机裁剪 75%~100% 再缩放到 256：模拟不同的病灶大小和视野
    albumentations.RandomSizedCrop((int(SOURCE_SIZE * 0.75), SOURCE_SIZE),
                                   (TARGET_SIZE, TARGET_SIZE),
                                   interpolation=cv2.INTER_NEAREST),
    # 水平翻转：左右肺没有本质区别，翻转不改变语义
    albumentations.HorizontalFlip(p=0.5),
])

# 验证集只做缩放，不做任何随机变换——评估必须可复现
val_augs = albumentations.Compose([
    albumentations.Resize(TARGET_SIZE, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
])
```

这里有两个要注意的地方。第一，**掩码的插值必须用最近邻**（`INTER_NEAREST`）。掩码里存的是类别编号 0/1/2/3，如果用双线性插值，会算出 1.7 这种不存在的类别，标签就全乱了。第二，**旋转 360 度**看起来夸张（等于随便转），但对 CT 切片是合理的：一张轴位切片旋转之后，仍然是肉眼可辨的肺部图像，只是方向不同。这类「语义不变」的增强，效果通常比调网络结构更明显。

**训练设置：** 损失函数用交叉熵（四分类的标准选择），优化器用 AdamW，学习率调度用 OneCycle：

```python
import torch
import torch.nn as nn

max_lr       = 1e-2
epoch        = 10
weight_decay = 1e-3

# 四分类逐像素交叉熵：网络输出 (N, 4, H, W)，标签是 (N, H, W) 的整数
criterion = nn.CrossEntropyLoss()

# AdamW = Adam + 解耦的权重衰减，比原版 Adam 正则效果更干净
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

# OneCycle：学习率先从很低升到 max_lr，再一路降回去
# 前期快速找到方向，后期小步收敛，很适合短轮次训练
sched = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr, epochs=epoch, steps_per_epoch=len(train_dataloader))
```

**早停。** 训练循环里我们自己实现了一个「连续 7 次没进步就停」的规则，并顺手把验证 mIoU 最高的模型存下来：

```python
min_loss, not_improve = np.inf, 0

for e in range(epoch):
    # ... 训练一个 epoch，然后在验证集上算 val_loss 和 val_miou ...

    if val_loss < min_loss:
        min_loss = val_loss
        # 文件名里带上验证 mIoU，方便事后挑最好的那一份
        torch.save(model, f'Unet_efficientnet_b2_mIoU-{val_miou:.3f}.pt')
    else:
        not_improve += 1
        if not_improve == 7:
            print('连续 7 次没进步，停止训练')
            break
```

「保存文件名带指标」是个很实用的小习惯：训练完不用记日志，直接看文件名就知道哪个 checkpoint 最好。在不平衡分割任务里，**val_loss 最低不一定 mIoU 最高**——因为 loss 会被大量背景像素主导，所以我们两个都记录，最终按 mIoU 挑模型。

### 2.4 模型测试与指标报告

**先算 IoU。** 我们把 IoU 写成一个函数，逐类算完再取平均：

```python
import torch.nn.functional as F

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=4):
    """pred_mask: 网络原始输出 (N,4,H,W)；mask: 整数标签 (N,H,W)"""
    with torch.no_grad():
        # 先 softmax 再过 argmax，得到每个像素的预测类别
        pred_mask = torch.argmax(F.softmax(pred_mask, dim=1), dim=1).view(-1)
        mask = mask.view(-1)

        iou_per_class = []
        for clas in range(n_classes):
            true_class = pred_mask == clas     # 我预测是这个类的像素
            true_label = mask == clas          # 真实是这个类的像素

            if true_label.sum().item() == 0:
                # 这一批里根本没有这个类，IoU 无定义，跳过（记为 nan）
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float()
                union     = torch.logical_or(true_class, true_label).sum().float()
                iou_per_class.append(((intersect + smooth) / (union + smooth)).item())

        return np.nanmean(iou_per_class)   # 忽略 nan 求平均
```

其中「这一类在本批里不存在就记 nan 再 `nanmean`」这一步是必须的。如果不跳过，某些 batch 里刚好没有实变影，分母为 0，分数会直接爆掉或者被算成 0，把平均值拖得完全失真。

**再看验证集表现。** 训练日志里每个 epoch 都会打印训练和验证的 loss、mIoU、像素准确率。这里有两个必须理解的现象：

- **像素准确率看上去很漂亮，mIoU 却低得多。** 这不是 bug，正是背景像素太多导致的——模型把背景猜对了，准确率就上去了；但病灶边缘只要差一点，IoU 就掉得厉害。这也再次说明为什么不能拿准确率当评估指标。
- **训练 mIoU 往往高于验证 mIoU。** 这说明模型开始记住训练集的标注风格（尤其是 Radiopedia 那种更精细的标注），泛化到 MedSeg 风格时打折扣。这也是我们坚持用 MedSeg 前 24 张当验证集的原因——用同源数据验证，才能看出「在测试集上会怎样」。

**最后生成提交。** 测试图只有 10 张，逐张推理，再从 4 个通道里取出前两个（磨玻璃和实变），最后缩回原始尺寸并拉平：

```python
# 每张测试图缩放成 256×256，转成张量后逐张送进网络
image_batch = np.stack([val_augs(image=img)['image'] for img in test_images_medseg], axis=0)

output = np.zeros((10, 256, 256, 4))
for i in range(10):
    # test_predict 内部做了 ToTensor + Normalize + softmax
    output[i] = test_predict(model, image_batch[i]).cpu().numpy()

# 概率 > 0.5 记为 1（该像素属于这一类）
test_masks_prediction = output > 0.5

# 只保留前两个通道（磨玻璃、实变），把后两类直接丢掉
# 再用最近邻插值放大 2 倍，回到 512×512 的原始分辨率
test_pred_original = scipy.ndimage.zoom(test_masks_prediction[..., :-2], (1, 2, 2, 1), order=0)

# 拉平成长表：Id 是行号，Predicted 是 0/1
pd.DataFrame(
    data=np.stack((np.arange(len(test_pred_original.ravel())),
                   test_pred_original.ravel().astype(int)), axis=-1),
    columns=['Id', 'Predicted']
).set_index('Id').to_csv('sub.csv')
```

这里 `[..., :-2]` 的切片是最容易忽略的一步：如果偷懒把 4 个通道全提交，行数会变成 10,485,760，是正确行数的两倍，提交直接判格式错误。另外 `order=0` 是最近邻插值的参数，同样是为了保证放大之后仍然是干净的 0/1，不会插出 0.5 这种值。

**结果与改进方向。** 这条 U-Net + EfficientNet-B2 的路线在验证集上能稳定地分割出磨玻璃影和实变影的主体区域，但边界处偏粗——这是交叉熵损失的通病，它只关心「每个像素分对没有」，不关心「预测区域和真实区域重叠多少」。

材料里还有几条不同的改进路线，我们都没有作为主线，但值得记一笔：

- **换损失函数**。改用 Dice 系数或 IoU 相关的损失，直接优化「重叠度」，对不平衡数据更友好。有位选手把 IoU、敏感度、特异度、F1、Dice 全都写成了 Keras 的度量函数，训练时一起监控。
- **换更轻的编码器**。用 MobileNetV3 替代 EfficientNet-B2，再配合量化（quantization）把模型压小，适合部署到算力有限的设备上。
- **换更强的架构**。用 Mask2Former 这类基于 Transformer 的分割模型，或者 ELUNet 这类为医学图像设计的变体。
- **做 K 折交叉验证**。有选手做了「排除 Radiopedia 只用 MedSeg」的 5 折训练，好处是验证分数更能反映测试集水平，代价是训练数据少了 829 张。

---

## 三、总结

本文中我们完成了 COVID-19 肺部 CT 病灶分割的完整流程。我们首先用 EDA 摸清了数据：MedSeg 100 张、Radiopedia 829 张、测试 10 张，全部是 512×512 单通道，掩码有 4 个通道，而且 `background` 的像素数远远超过两个目标病灶，类别严重不平衡。我们还验证了两个关键事实——数据里没有 NaN 或格式问题，以及磨玻璃影和实变影**完全互不重叠**，后者让我们可以放心地把四通道压成单通道整数标签、当成四分类问题来做。其次我们做了针对性的清洗：把 CT 值截断到 [-1500, 500]、只取 5%～95% 分位之间的像素算出均值 -451.65 和标准差 458.57（并且只在 Radiopedia 上算一次，另外两个集复用）、用 `argmax` 把独热掩码压成类别图、按 24/76/829 划分验证与训练。接着我们用 `segmentation_models_pytorch` 搭了 U-Net + EfficientNet-B2（ImageNet 预训练、单通道输入、5 层编码器、4 类输出），配了旋转 360°、随机裁剪缩放、水平翻转三种增强（掩码统一用最近邻插值），用交叉熵 + AdamW + OneCycleLR 训练 10 轮，并手写了「连续 7 次不进步就停」的早停和按 mIoU 保存 checkpoint 的逻辑。最后我们手写了逐类计算、忽略空类的 mIoU 函数做评估，推理时只取前两个通道、用最近邻放大回 512×512、拉平成 5,242,880 行的提交文件。

这个比赛给我们最大的启发是：**在不平衡的分割任务里，评估指标和损失函数要分开想。** 我们用 IoU 评估是对的，因为它不被大量背景像素掩盖；但训练用的交叉熵并不直接优化 IoU，这就造成了「loss 降得很漂亮、边界却不够贴」的落差。下一步最该做的不是换更大的网络，而是把 Dice 或 IoU 相关的损失加进来。另外，**医学影像的预处理有一套固定套路**：按 HU 范围截断、用分位数估统计量、统计量只在训练集上算一次然后全局复用。这些步骤看起来琐碎，但只要有一处做反（比如给测试集单独做标准化），本地分数就会和线上完全对不上。
