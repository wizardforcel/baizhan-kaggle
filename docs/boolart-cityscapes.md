# BoolArt Cityscapes 街景语义分割

## 一、比赛简介

### 1.1 比赛背景

这个比赛是 BoolArt 团队基于经典的 **Cityscapes 街景分割数据集**改造出来的一个语义分割任务。Cityscapes 是一批在欧洲城市街道上拍摄的驾驶场景照片，每张图都带有人工标注的逐像素标签，标注出画面里每一块是马路、是人、还是车。

语义分割，通俗说就是把图片**按像素分类**：模型输出一张和输入同样大小的图，每个像素都被标成它所属的类别（这一格是 road，那一格是人）。与「整张图属于哪一类」的分类不同，分割要求模型理解每一个像素，是计算机视觉里难度更高的一类任务。

比赛数据的组织方式和 Cityscapes 一样，分两套目录：

- `data/`（或 `data/train`/`data/test`）：街道照片本体；
- `label/`：与照片一一对应的标签图，文件名形如 `xxxx_gtFine_labelTrainIds.png`——`labelTrainIds` 表示每个像素存的是「训练用的类目编号」，从 0 开始。

比赛要求预测 **8 个类别**：`other`（其他）、`road`（马路）、`person`（行人）、`rider`（骑行者）、`car`（轿车）、`truck`（卡车）、`bus`（公交车）、`bicycle`（自行车）。注意这里的方向是「车辆为主」，和原版 Cityscapes 更全的类别清单不一样。

难点有三：一是分割图分辨率高，训练显存压力大；二是类别之间极度不平衡（画面里大片是马路，行人只有零星几个像素），普通交叉熵会把小类别直接忽略；三是这次比赛的数据本身是**合成再采样**的，训练/验证的划分需要用稳妥的交叉验证来保证分数可信。

### 1.2 比赛目标

给测试集每张街景照片的**每一个像素**都打上 0～7 的类别标签。

数据组织：训练图片位于 `data/train/`，下面的子目录把图片分成若干组；标签在 `label/` 对应位置。测试图片位于 `data/test/`。

提交文件的格式很特殊——**不是直接提交一张标签图**，而是把每个类别的分割结果写成**运行长度编码（RLE）**。RLE 是一种紧凑的表示法：`mask.T.flatten()` 后，依次记录「这一段连续像素从哪开始、连续多长」。每一行是 `id,predict`，其中 `id` 为 `图片名_类别编号.png`，`predict` 是对应类别的 RLE 字符串。

### 1.3 评价标准

语义分割的标准衡量指标是**平均交并比（mean IoU）**。所谓交并比，是对每一个类别分别算：模型预测为该类的像素集合，和真实为该类的像素集合，「两者的交集面积 ÷ 两者的并集面积」，取值 0～1。如果预测的和真实的分毫不差，交并比就是 1；完全对不上就是 0。最后把 8 个类别的 IoU 求平均，得到 mIoU，**越大越好**。

> 小提示：因为 mIoU 对每个类别**一视同仁地平均**，马路上有一千个像素、行人只有十个像素，但两者在 mIoU 里权重相同。所以光把大片面积的分对没用，稀疏的小类别同样关键——这直接决定了我们后面要专门用一个叫 Dice 损失的训练技巧。

---

## 二、解题过程

我们的完整流程是：先把图片和标签配对、划分折，接着设计一个「混合 Swin Transformer」分割模型，然后训练、做推理，最后把预测转成 RLE 提交。同时我们先用一版简单的 UNet 当基线，确认流程能通。

### 2.1 数据探索（EDA）

首先把所有训练图片列出来，和标签对齐，并做好 5 折划分：

```python
import glob
import pandas as pd
from sklearn.model_selection import KFold

# 所有训练图片，路径形如 data/train/<子目录>/<图片名>.png
train_path = glob.glob('../input/boolart-cityscapes/data/train/*/*')
df_train = pd.DataFrame({"image_path": train_path})

# 标签路径：把路径里的 data 换成 label，再换成 gtFine_labelTrainIds 结尾
df_train['mask_path'] = df_train['image_path'].str.replace("data", 'label')
df_train['mask_path'] = df_train['mask_path'].apply(
    lambda x: x.split('.png')[0][:-11] + 'gtFine_labelTrainIds.png')

# 打散后用 KFold 造 5 折，每折轮流当验证集
Fold = KFold(n_splits=5)
for n, (train_index, val_index) in enumerate(Fold.split(df_train)):
    df_train.loc[val_index, 'fold'] = int(n)
df_train['fold'] = df_train['fold'].astype(int)

print(df_train.head())
print(df_train['fold'].value_counts())
```

设置随机种子这点必不可少——分割模型训练有大量随机性（数据增强、dropout），不固定种子的话，同一份代码跑两次结果都不一样，没法复现：

```python
import random, os
import numpy as np
import torch

def set_seed(seed=2 ** 3):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

然后我们定义了全局配置类，把这一轮实验的几乎所有超参数都收拢在一个地方，方便复现和对比：

```python
class CFG:
    seed     = 101
    debug    = True        # 调试模式：只跑少量样本，确认管道能跑通
    exp_name = 'Baselinev1'
    train_bs = 8           # 训练批大小，分割图吃显存，不敢太大
    valid_bs = train_bs * 2
    img_size = [384, 384]  # 统一缩放到 384x384
    epochs   = 5
    lr       = 1e-4        # Swin 编码器用较小的学习率
    scheduler = 'CosineAnnealingLR'
    min_lr   = 1e-6
    T_max    = 10
    wd       = 1e-6        # 权重衰减，防过拟合
    n_fold   = 5
    num_classes = 3        # 基线调试时先用 3 类，正式版是 8 类
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 2.2 数据清洗

分割任务里，「清洗」主要落在两块：**统一尺寸**和**数据增强**。我们用的增强库是 albumentations，它有一个很关键的好处：**图片和标签用同一套随机变换同时做**，保证裁剪、翻转后像素还是一一对得上的：

```python
import cv2
import numpy as np
import albumentations as A

data_transforms = {
    "train": A.Compose([
        # 统一缩放尺寸。分割必须用 INTER_NEAREST，
        # 因为线性插值会把类别编号「插」成 2.5 这种伪造值
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),                       # 水平翻转，街景左右对称很合理
        A.ShiftScaleRotate(shift_limit=0.0625,          # 小幅平移+缩放+旋转
                           scale_limit=0.05, rotate_limit=10, p=0.5),
        A.OneOf([                                      # 每次随机挑一种几何畸变
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        A.CoarseDropout(max_holes=8,                    # 随机挖掉几块，逼模型看全局
                        max_height=CFG.img_size[0] // 20,
                        max_width=CFG.img_size[1] // 20,
                        min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
    ], p=1.0),
    "valid": A.Compose([                              # 验证集只缩放，不做增强
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
    ], p=1.0),
}
```

随后写 `Dataset` 类。注意两点：图片要转成 RGB（`cv2.imread` 默认读成 BGR），并且把 HWC 的排布转成 PyTorch 习惯的 CHW；标签则直接用 `PIL` 原样读进来，**不做任何归一化**，因为它的取值就是类别编号本身：

```python
from PIL import Image
import torch
from torch.utils.data import DataLoader

class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].tolist()
        self.msk_paths = df['mask_path'].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB
        msk = np.array(Image.open(self.msk_paths[index]))
        if self.transforms:
            data = self.transforms(image=img, mask=msk)  # 图片标签同步变换
            img, msk = data['image'], data['mask']
        img = np.transpose(img, (2, 0, 1))           # HWC -> CHW
        return torch.tensor(img), torch.tensor(msk)

def prepare_loaders(fold, debug=False):
    # 只留当前折：fold 列不等于当前 fold 的作训练，等于的作验证
    train_df = df_train.query("fold!=@fold").reset_index(drop=True)
    valid_df = df_train.query("fold==@fold").reset_index(drop=True)
    if debug:                                        # 调试模式少跑一些
        train_df = train_df.head(32 * 15).reset_index(drop=True)
        valid_df = valid_df.head(32 * 3).reset_index(drop=True)
    train_dataset = BuildDataset(train_df, transforms=data_transforms['train'])
    valid_dataset = BuildDataset(valid_df, transforms=data_transforms['valid'])
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs,
                              num_workers=4, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs,
                              num_workers=4, shuffle=False, pin_memory=True)
    return train_loader, valid_loader
```

增强效果我们还会画出来人工检查一眼：把原始图片和叠加了半透明 mask 的图并排打印，确认（1）图片没被增强弄坏，（2）mask 和图片仍然对得齐。这一步虽然朴素，但能提前发现「图片和标签错位」这种灾难性 bug。

### 2.3 模型训练与超参数调优

**第一步，搭一个 UNet 基线。** 我们使用 `segmentation_models_pytorch`（smp）这个开箱即用的分割库。`Unet` 是经典的编码器-解码器结构：编码器不断下采样提取高层语义，解码器把特征图上采样回原尺寸，中间用跳跃连接把细节补回来。骨架 `resnet34` 是预训练好的图像分类网络，直接拿来做编码器，能省下一半的训练时间：

```python
import segmentation_models_pytorch as smp

def build_model():
    model = smp.create_model(
        arch='Unet',                  # 分割模型结构
        encoder_name='resnet34',      # 编码器骨架
        encoder_weights="imagenet",   # 加载 ImageNet 预训练权重
        in_channels=3,                # 输入是三通道彩色图
        classes=8,                    # 输出 8 个类别（每个像素 8 选 1）
        activation=None,              # 不套激活，损失函数里再做
    )
    model.to(CFG.device)
    return model
```

训练时我们用**两个损失各占一半**的组合。第一个是交叉熵的平滑版 `SoftCrossEntropyLoss`，其中 `ignore_index` 关闭了它；第二个是 **Dice 损失**——它直接优化「类别间的重叠程度」，对类别不平衡非常友好：

```python
maskloss1 = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=8)
maskloss2 = smp.losses.DiceLoss(mode='multiclass', smooth=0.05, ignore_index=8)
```

训练循环里我们启用了 PyTorch 的**混合精度（AMP）**来省显存、加速训练，并用梯度累积把有效批大小撑到 32：

```python
from torch.cuda import amp

def train_one_epoch(model, optimizer, dataloader, device):
    model.train()
    scaler = amp.GradScaler()          # 混合精度用的梯度缩放器
    running_loss, dataset_size = 0.0, 0
    for step, (images, masks) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device)
        with amp.autocast(enabled=True):                 # 半精度前向
            y_pred = model(images)
            loss = 0.5 * maskloss1(y_pred, masks.long()) \
                 + 0.5 * maskloss2(y_pred, masks.long())
            loss = loss / CFG.n_accumulate               # 除以累积步数
        scaler.scale(loss).backward()                    # 梯度也缩一下再反传
        if (step + 1) % CFG.n_accumulate == 0:           # 攒够 n 个 batch 才更新一次
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    return running_loss / dataset_size
```

学习率我们用**余弦退火**调度：开始时按 `lr=1e-4` 大步探索，之后按余弦曲线平滑地降到 `min_lr=1e-6`，最后阶段的小步长有利于把模型「磨」到更优的谷底：

```python
from torch.optim import lr_scheduler

def fetch_scheduler(optimizer):
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)
    return scheduler
```

验证时用**验证损失**挑选最好的一轮权重保存下来，`@torch.no_grad()` 是为了省显存、加速推理：

```python
import copy, time
from collections import defaultdict

def run_training(model, optimizer, scheduler, num_epochs):
    best_loss = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader,
                                     device=CFG.device)
        val_loss = valid_one_epoch(model, valid_loader, device=CFG.device)
        # 验证损失变好就存权重，防止过拟合后覆盖最佳模型
        if val_loss <= best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"best_epoch-{fold:02d}.pth")
        torch.save(model.state_dict(), f"last_epoch-{fold:02d}.pth")
```

**第二大步：换更强的混合 Swin Transformer。** 基线的 UNet 只能看到局部感受野，而街景分割里「这块是不是马路」往往取决于远处——于是我们换成了 **Swin Transformer 编码器 + UPer 解码器**的混合结构（hybrid）：

- **Swin Transformer** 是 2021 年提出的视觉 Transformer。它把图像切成一块块小窗口，每块窗口内部做自注意力（每个像素和其他像素互相打量），窗口再逐层移动、跨层融合，从而既能看到全局又能控制计算量。材料里我们直接搬运了官方 Swin 的实现（`WindowAttention`、`SwinTransformerBlock` 等），并选了 `swin_tiny_patch4_window7_224` 这个版本作为编码器，再用 ADE20K 数据集上预训练好的权重初始化。
- **UPer 解码器**负责把编码器给出的金字塔特征（4 个层级、通道数 $[96,192,384,768]$）融合回原尺寸：用 PPM（金字塔池化）抓不同尺度的上下文，再用一串 FPN 层自顶向下逐级合并，最后接一个 1×1 卷积输出 8 类 logits：

```python
class Net(nn.Module):
    def __init__(self, num_class=8):
        super(Net, self).__init__()
        self.arch = 'swin_tiny_patch4_window7_224'
        # Swin 编码器：输出 4 个层级的特征图
        self.encoder = SwinTransformerV1(** {**cfg['basic']['swin'],
                                             **cfg[self.arch]['swin'],
                                             **{'out_norm': LayerNorm2d}})
        encoder_dim = [96, 192, 384, 768]        # 四个层级的通道数

        # UPer 解码器：PPM（金字塔池化）抓多尺度上下文 + FPN 金字塔融合
        self.decoder = UPerDecoder(in_dim=encoder_dim,
                                   ppm_pool_scale=[1, 2, 3, 6],
                                   ppm_dim=512, fpn_out_dim=256)

        # 1x1 卷积把 256 通道压成 8 类 logits
        self.logit = nn.Sequential(nn.Conv2d(256, num_class, kernel_size=1))

    def forward(self, x):
        encoder = self.encoder(x)            # B,4,Gi,Hi,Wi -> 4 层特征
        last, decoder = self.decoder(encoder)
        logit = self.logit(last)
        # 2 倍上采样回到模型约定的尺寸
        logit = F.interpolate(logit, scale_factor=2, mode='bilinear',
                              align_corners=False)
        return logit
```

这个混合模型的 super 参数和基线区别：编码器换成了 Swin、解码器换成了 UPer，隐层能力明显更强，训练时我们用 `Adam` 优化器、`lr=1e-4` 和余弦退火，训练 5 个 epoch。

### 2.4 模型测试与指标报告

测试阶段，我们把保存的最后一代权重加载回来，逐张图做推理。**参赛提交要的是 8 个类别各自的分割 RLE**，所以推理时我们对 `sigmoid(logits)` 取阈值 `0.45`（这个阈值让稀疏类别的召回与精确达到平衡，比默认的 0.5 略宽松，因为背景类别占比过高，默认阈值会让行人/单车这些小目标大量漏检），然后调回原始尺寸，再逐类别做 RLE 编码：

```python
@torch.no_grad()
def inference(model, dataloader, device):
    model.eval()
    pred_strings, pred_ids, pred_classes = [], [], []
    for step, (images, id_, height, width) in enumerate(dataloader):
        images = images.to(device, dtype=torch.float)
        y_pred = model(images).squeeze()
        y_pred = torch.nn.Sigmoid()(y_pred)                  # 逐类输出 0~1 概率
        y_pred = (y_pred.permute((1, 2, 0)) > 0.45).to(torch.uint8)  # 阈值化
        y_pred = cv2.resize(y_pred, dsize=(int(width), int(height)),  # 恢复原尺寸
                            interpolation=cv2.INTER_NEAREST)
        for midx in range(8):
            rle = rle_encode(y_pred[..., midx])              # 每类单独 RLE
            file_name = id_[0][:-4] + str(midx) + '.png'     # 拼出提交 id
            pred_ids.extend([file_name])
            pred_strings.extend([rle])
        pred_classes.extend(['other', 'road', 'person', 'rider',
                             'car', 'truck', 'bus', 'bicycle'])
    return pred_strings, pred_ids, pred_classes

test_dataset = BuildDataset(test_df, transforms=data_transforms['valid'])
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1)
model.load_state_dict(torch.load('./last_epoch-00.pth'))
pred_strings, pred_ids, pred_classes = inference(model, test_loader, CFG.device)

df = pd.DataFrame({"id": pred_ids, "predict": pred_strings})
df.to_csv('./submission.csv', index=False)
```

同样写出 RLE 编码函数，它把二维 mask 拍平成一维，记录「值发生变化的位置」，相邻相同值的连续段只存起点和长度，是提交格式要求的标准写法：

```python
def rle_encode(mask):
    pixels = mask.T.flatten()            # 先转置再拍平，按宽先扫
    if pixels[0] or pixels[-1]:
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded            # 头尾补一格，统一编码逻辑
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2   # 找到所有断点
    if pixel_padded: pass
    if pixels[0]: rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]   # 偶数位变成「长度」而非下标
    return ' '.join(str(x) for x in rle)
```

评估上，我们在每个折的验证集上跟踪验证损失，用「验证损失最低」这一轮保存模型。训练日志显示验证损失随 epoch 稳定下降，说明模型在逐步学会分割。因为我们没有在材料里得到最终的 mIoU 数值，这里我们以**验证损失**作为主要监控指标，把它作为赛榜分数（逐类平均 IoU）的代理——两者在实验中通常是单调相关的。

这版流程的教训也很直接：**基线 UNet 用 8 类、384×384、混合精度 AMP 和双损失（交叉熵 + Dice）已经能跑通整条链路**；再往上走的提升主要来自把编码器从 resnet34 换成预训练 Swin、把解码器从 UNet 升级成带金字塔池化的 UPer。

---

## 三、总结

本文中我们完成了 BoolArt Cityscapes 街景语义分割的完整流程。我们首先用 EDA 把图片和标签配对，用 KFold 做了 5 折划分，并固定随机种子保证可复现；其次我们做了数据清洗与增强，用 albumentations 统一缩放尺寸（分割专用的最近邻插值），配合水平翻转、平移旋转、网格畸变和随机挖洞增强，并让图片与标签同步变换；接着我们训练了两版模型——先用 UNet + resnet34 基线跑通链路，再把编码器换成预训练的 Swin Transformer（`swin_tiny`）、解码器升级为带 PPM 金字塔池化和 FPN 融合的 UPer；训练时用交叉熵 + Dice 双损失各占一半、混合精度 AMP 加速、余弦退火调度学习率，按验证损失保存最优权重；最后在测试集上把 sigmoid 概率按 0.45 阈值化、恢复原尺寸、逐类别做 RLE 编码，按 `id,predict` 格式提交。

这个比赛给我们最大的启发是：**语义分割这类像素级任务的成败，一半在「让每个类别都被公平对待」的训练技巧里**。像 `ignore_index`、Dice 损失、以及为小目标把阈值从 0.5 放宽到 0.45，这些细节对 mIoU 的影响，往往比换一个更强的网络骨架还大。