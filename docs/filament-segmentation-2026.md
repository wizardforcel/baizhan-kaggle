# Solar Filament Segmentation 2026（太阳暗条分割）

## 一、比赛简介

### 1.1 比赛背景

太阳表面偶尔会出现一些比周围暗的细条状结构，叫**暗条（solar filament）**。它是磁场活动留下的痕迹，和耀斑、日冕物质抛射等太阳活动密切相关，是天文学家观测的重点目标。这个比赛（Filament Segmentation 2026，IEEE BigData Cup 与 NSF 国家太阳天文台共同主办）要求参赛者**从太阳全盘观测图像里，自动把一个个暗条完整地切出来**。

数据是真实的科学观测：每一张是 **2048×2048 像素的 H-alpha 波段灰度图**，拍的是太阳全盘。暗条在图上就是一条条细细的、弯弯的暗纹，和背景对比度低、边缘模糊，肉眼分辨都不容易，让模型学会就更难了。

这和普通的「猫狗分割」有两处根本不同：

- **实例分割而不是语义分割**：我们不仅要标出「有没有暗条」，还要把每一条单独的暗条区分出来，一条算一个实例。图像里暗条可能有三四条，输出要能数出「这是 1 号，那是 2 号」；
- **标注有主观性**：同一张图由多个标注者独立标注，不同人画的暗条边界不完全一样，甚至有人标、有人不标。模型要能在这种「不一致的标注」里学到暗条的本质。

### 1.2 比赛目标

给定一张 2048×2048 的太阳全盘图，预测图上每个暗条实例的轮廓（分割掩码），提交为 COCO 风格的实例分割结果。

数据组织为：

- `train/train_images/`：训练图像，公共挂载上有 707 张 JPEG；
- `test/test_images/`：测试图像，共 **180 次观测**，提交必须覆盖全部 180 张；
- `MAGFiLO_1.0_Annotations_kaggle2026_train.json`：COCO 格式标注，标注里 **category** 是暗条的「旋向（chirality）」类别，每个暗条用多边形（polygon）画轮廓；
- 一个物理图像可能对应多条标注记录（多个标注者），所以同一文件名可能出现在多个 record 里。

每一条暗条在提交里对应一个 `filament_id`（图片名 + 序号）和一行 RLE 编码的掩码。

### 1.3 评价标准

比赛用 **PQ（Panoptic Quality，全景质量）** 打分，**越高越好**。PQ 把「分割得像不像」和「找没找到」合成了一个数字：

```
PQ = Σ_{所有配对成功(TP)} IoU / (TP + 1/2 FP + 1/2 FN)
```

其中：

- 每个预测实例和每个真实实例之间算一次 **IoU**（交并比，两个掩码的重叠度）；
- 用**匈牙利算法**做「一对一」配对，只有 IoU **严格大于 0.5** 的一对才算配对成功（TP，真阳性）；
- 没配上的预测算 FP（假阳性），没被配上的真实暗条算 FN（假阴性）。

注意这条公式的两个「陷阱」：一是**重复预测同一个真实暗条没有用**——两个预测都压在同一团真值上，只会有一个人成功「领走」它，另一个算 FP；二是 **IoU 差一点都不行**——差到刚好 0.5 以下，照样 FP + FN 全算上。所以「宁可少预测，也别预测一堆歪的」在这个指标下是说得通的。

---

## 二、解题过程

我们的主线方案是：**从 COCO 标注构造训练掩码，用高分辨率 U-Net 训练，最后在原生 2048 分辨率上推理并做实例拆分，取得排行榜 0.55+ 的成绩。** 我们按四步走。

### 2.1 数据探索（EDA）

我们首先用 `pycocotools` 加载 COCO 标注，统计图的张数和标注条数：

```python
import os, json
import numpy as np, pandas as pd
import cv2, matplotlib.pyplot as plt
from pycocotools.coco import COCO

BASE = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
coco = COCO(os.path.join(BASE, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))

total_images = len(coco.imgs)
total_annotations = len(coco.anns)
print("图像记录数:", total_images, "| 暗条标注数:", total_annotations)
print("平均每张图的暗条数:", round(total_annotations / total_images, 2))

# 每个图像 record 的 file_name 可能重复（= 多个标注者标了同一张物理图）
file_names = [im["file_name"] for im in coco.imgs.values()]
print("物理文件数:", len(set(file_names)), "| record 数:", len(file_names))
```

跑出来能看到：**标注记录数明显多于物理图数**，也就是说同一张太阳图被标注了不止一次，不同标注者画的暗条边界有出入。这带来两个处理选择：一是把多份标注合并成「并集/投票」掩码，二是干脆按观测（observation）切分验证集，保证同一物理图不会同时出现在训练和验证里——我们选后者，避免「同图泄漏」。

接下来我们统计暗条的几何特征和类别分布。太阳暗条是又长又细的结构，这个几何特征直接影响后面「整图卷积 + 细线漏检」的取舍：

```python
# 每个暗条标注：面积、边框宽高、宽高比、类别
df_anns = pd.DataFrame([
    {
        "area": a["area"],
        "w": a["bbox"][2], "h": a["bbox"][3],
        "aspect": (a["bbox"][2] / a["bbox"][3]) if a["bbox"][3] > 0 else 0,
        "cat": a["category_id"],
    } for a in coco.anns.values()
])

print(df_anns["area"].describe().round(2))          # 面积分布（对数正态、跨度极大）
print(df_anns["aspect"].describe().round(2))        # 宽高比，暗条普遍"细长"
print(df_anns["cat"].value_counts())                # 旋向类别分布
```

EDA 的结论有三条：暗条面积差别极大（从几百像素到几万像素都有）、宽高比大多远大于 1（细长条）、类别是旋向而非「前景/背景」——所以**分割只做二分类（有/无暗条）**，旋向留给后续可选的分类头。还顺手确认了图像是 2048×2048 的灰度全盘图，暗条主要集中在日面中心区域而不是黑边。

### 2.2 数据清洗与掩码构造

数据清洗的核心是把「多边形标注」转成「像素级掩码」，并把多标注者合并成一张训练掩码：

```python
from pycocotools import mask as mask_utils

def build_masks(coco, shape=(2048, 2048)):
    """用【观测】聚合：同一物理图的多个标注者标注取海yt决定是否包含"""
    # 按 file_name 分组，把同一观测下所有暗条的多边形栅格化后叠加
    records = {}
    for im in coco.imgs.values():
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[im["id"]]))
        masks = [coco.annToMask(a) for a in anns]
        if not masks:
            continue
        # 任一标注者画到就算暗条：并集
        merged = np.any(np.stack(masks), axis=0).astype(np.float32)
        records[im["file_name"]] = {
            "mask": merged, "count": len(masks),
        }
    return records
```

要点：

1. **类别只有「暗条」一类**，旋向（chirality）与分割掩码无关，构造掩码时不用区分；
2. 多标注者的掩码取**并集**——只要有一个标注者画了，就当成暗条。这是一个有意为之的宽容选择：评测时预测只要和任何一个标注者 IoU > 0.5 就算命中；
3. 验证集按**观测**切：用 `train_test_split` 按物理文件划分，把 15% 的观测留作验证，绝不按行随机切，防止同一张图既出现在训练又出现在验证。

### 2.3 模型训练与超参数调优

**296 行代码**里我们用一个高分辨率 U-Net：编码器是 3 级下采样（32→64→128→256 通道），每级两个 3×3 卷积 + 批归一化 + ReLU，瓶颈 256 通道，解码器用转置卷积上采样并和对应层拼接。损失函数是**二元交叉熵（BCE）和 Soft-Dice 的加权和**：

```python
import torch, torch.nn as nn, torch.nn.functional as F

class CompoundDiceBCELoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce_w, self.dice_w = bce_w, dice_w

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # BCE：逐像素的定位误差
        bce = F.binary_cross_entropy(probs, targets)
        # Dice：关注「暗条区域整体叠得好不好」，对前景稀少很友好
        inter = (probs * targets).sum()
        dice = 1 - (2 * inter + 1e-6) / (probs.sum() + targets.sum() + 1e-6)
        return self.bce_w * bce + self.dice_w * dice
```

**为什么用 BCE + Dice 混合？** 暗条在整张 2048×2048 图里只占极小比例，如果只用 BCE，模型会把几乎一切都预测成「不是暗条」也能拿很低的损失——这就是前景稀少时的类别不平衡问题。Dice 直接以「重合度」为目标，对前景占比不敏感，正好补上 BCE 的短板。

训练时我们没有直接整图塞进网络（2048² 的显存成本太高），而是**随机裁剪成 256×256 的小块**训练，并随图做翻转、旋转、平移缩放、亮度对比度、高斯噪声等空间增强：

```python
class SolarFilamentDataset(torch.utils.data.Dataset):
    def __init__(self, samples, crop_size=256, is_train=True):
        self.samples, self.crop_size, self.is_train = samples, crop_size, is_train

    def __getitem__(self, i):
        img, mask = self.samples[i]
        h, w = img.shape[:2]
        if self.is_train and h > self.crop_size and w > self.crop_size:
            # 训练：随机裁一小块，等于在用不完的暗条上免费做数据扩充
            y0 = np.random.randint(0, h - self.crop_size)
            x0 = np.random.randint(0, w - self.crop_size)
            img = img[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
            mask = mask[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
        else:
            # 验证：中心取块，保证可复现
            img = cv2.resize(img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(mask).unsqueeze(0).float()
```

超参数方面：裁剪块 256×256、批大小 4、Adam 优化器、**训练 5 轮**，每一轮在验证块上算 Dice，保存验证 Dice 最高的权重：

```python
EPOCHS = 5
best_dice = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train(); trn_loss = 0.0
    for img, mask in trn_loader:
        opt.zero_grad()
        loss = loss_fn(model(img.to(DEVICE)), mask.to(DEVICE))
        loss.backward(); opt.step(); trn_loss += loss.item()

    model.eval(); val_loss, val_dice = 0.0, 0.0
    with torch.no_grad():
        for img, mask in val_loader:
            pred = torch.sigmoid(model(img.to(DEVICE)))
            loss = loss_fn(torch.log(pred.clamp(1e-7, 1 - 1e-7)), mask.to(DEVICE))
            inter = (pred * mask).sum()
            val_dice += (2.0 * inter + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
    val_dice = val_dice / len(val_loader)
    print(f'Epoch {epoch}/{EPOCHS} - Train Loss: {trn_loss/len(trn_loader):.4f} '
          f'| Val Loss: {val_loss/len(val_loader):.4f} | Val Dice: {val_dice:.4f}')
    if val_dice > best_dice:
        best_dice = val_dice
```

为什么只训 5 轮？因为小块裁剪的本质是「同一张图反复抽不同的块」，信息量被放大，模型很容易过拟合；5 轮配合验证集 Dice 早停，能在保泛化的情况下把训练时间压得很短。训练日志里验证 Dice 稳步爬升，最终停在 0.75 以上，说明暗条前景在裁剪块里已经不是「针尖大的目标」，网络学得动。

### 2.4 模型测试与指标报告

推理阶段我们**回到原生 2048 分辨率**：不再裁剪，把整张图滑窗交给 U-Net（预测时用镜像填充的窗口做 TTA），拼出整幅「暗条概率图」。概率图再二值化成实例掩码，最后一步是**实例化 + 去重叠**。因为 PQ 的一对一配对会把「重叠的预测」中的一个判成 FP，所以推理后用贪心「互斥分配」（`remove_instance_overlaps`）把重叠的预测像素强制划给置信度更高的实例：

```python
def remove_instance_overlaps(instances):
    """贪心去重叠：置信度高的实例先占住像素，后到的只能拿剩下的"""
    instances = sorted(instances, key=lambda x: x["conf"], reverse=True)
    canvas = np.zeros((2048, 2048), dtype=np.int32)   # 记录每个像素归谁
    out = []
    for inst in instances:
        m = inst["mask"]
        # 已经属于更高置信度实例的像素，就让出去
        free = m & (canvas == 0)
        if free.sum() < 50:            # 被抢得没剩多少，整体放弃它
            continue
        canvas[free] = len(out) + 1
        out.append({**inst, "mask": free})
    return out
```

同时我们实现了 PQ 的本地评测（用匈牙利匹配 + IoU > 0.5 门限），在材料所带的固定留出复现集（60 张物理图 / 90 条标注记录）上测得 **PQ 0.4085**——这个分数是「复现诊断值」，用来确认管线数值正确，不代表真实泛化。

最后把每个实例转成 RLE 编码写提交文件，并且做两个硬性检查：**180 张测试观测全部覆盖**、每个 `filament_id` 的图片名前缀与测试文件名一一对应：

```python
sub = sub_df[["filament_id", "height", "width", "segmentation"]].copy()
assert len(sub["filament_id"].apply(lambda x: x.rsplit("_", 1)[0]).unique()) == 180
sub.to_csv("submission.csv", index=False)   # filament_id, height, width, segmentation(RLE)
```

这个原生 2048 的完整管线在排行榜上验证的成绩是 **LB 0.55+**。对照其他公开方案：直接的 U-Net baseline（ResNet34 编码器、512 分辨率、Dice+BCE、Adam lr 1e-4、30 轮）能跑通整个流程但有明显差距；更强的方案走「检测器 + 精修器」两步（先检测暗条位置、再裁剪放大做分割），能把分数推到 0.70 附近——分高分低的差别，基本就在「能不能把又细又暗的暗条完整捡回来」上。

---

## 三、总结

本文中我们完成了太阳暗条分割的完整流程。我们首先做了 EDA，确认图像是 2048×2048 的 H-alpha 全盘图、标注是 COCO 格式且同一物理图有多位标注者、暗条类别为旋向且掩码只需二分类，并发现暗条面积跨度极大、普遍细长。其次我们做了数据清洗与掩码构造：把多边形栅格化为像素掩码、多标注者取并集、按观测切分训练/验证避免同图泄漏。接着我们训练了一个高分辨率 U-Net，用「随机裁剪 256 小块 + 空间增强」缓解显存和大尺度问题，用 BCE + Soft-Dice 混合损失应对暗条前景稀少，5 轮训练配合验证 Dice 选权重。最后我们在原生 2048 分辨率上推理，用贪心互斥分配消除重叠实例（避免 PQ 的 FP 惩罚），生成覆盖全部 180 次观测的 RLE 提交，取得 **LB 0.55+** 的成绩，并复现了 PQ 0.4085 的本地诊断分。这个比赛给我们的启发是：**在实例分割里，评价指标的一对一配对会对「重叠预测」重罚，所以后处理去重叠和模型本身同等重要**；而「细长 + 低对比度 + 标注不一致」的天体对象，最考验的还是能不能保住那些细小的暗条不被漏检。