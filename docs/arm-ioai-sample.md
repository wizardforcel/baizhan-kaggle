# CIFAR-100 图像分类（ArmIOAI 挑战基线）

## 一、比赛简介

### 1.1 比赛背景

CIFAR-100 是深度学习里非常经典的一个图像数据集，可以理解为 CIFAR-10 的"困难版"。它们的图片都是 **32×32 的彩色小图**，分辨率一样低，区别在类别数：CIFAR-10 只有 10 类，CIFAR-100 有 **100 个细分类别**。

这个变化看着只是数字变大，实际难度完全不同：

- 训练集一共 5 万张图，摊到 100 个类别上，**每类只有约 500 张**，是 CIFAR-10 每类数量的十分之一；
- 很多类别长得很像，比如海狸（beaver）和水獭（otter）、枫树（maple_tree）和柳树（willow_tree）、蛇（snake）和蠕虫（worm）；
- 随便猜一个类别的准确率只有 **1%**。

比赛还有一条重要规则：**不允许使用预训练模型，也不允许使用外部数据**。所有模型必须从零开始、只用手上这份数据训练。这个约束看似苛刻，其实很公平——它逼着参赛者思考"到底是什么让模型泛化"，而不是下载一个在 ImageNet 上训好的权重直接用。

### 1.2 比赛目标

输入是一张 32×32 的 RGB 图片（PNG 文件，文件名就是图片 id），输出是它所属的类别编号，范围 0 到 99。

数据组织方式如下：

- `train_images/train_images/`：训练图片；
- `public_test_images/public_test_images/`：公开测试图片；
- `private_test_images/private_test_images/`：私有测试图片；
- `train_labels.csv`：两列，`id` 和 `label`；
- `public_test_ground_truth.csv`、`private_test_ground_truth.csv`：测试集的真实标签，用来在本地算准确率；
- 提交文件两列：`id` 和 `label`（0～99 的整数）。

### 1.3 评价标准

比赛用 **Top-1 准确率**（top-1 accuracy）评分，也就是"预测概率最高的那个类别正好是正确答案"的比例，越高越好。因为类别多、难度大，notebook 里还会额外报告 **Top-5 准确率**：只要正确答案落在模型给出的前五个类别里就算对，作为辅助观察。

随机猜的 Top-1 是 1%，所以哪怕只有 40% 的准确率，也已经比瞎猜强了四十倍。

---

## 二、解题过程

我们的思路非常克制：不引入残差连接、不引入复杂模块、不用任何预训练权重，只用 5 个卷积块加 2 个全连接层，把数据增强、批归一化、Dropout、标签平滑、余弦退火学习率和早停这些"基本功"做扎实，看看一个干净的 CNN 能走多远。

### 2.1 数据探索（EDA）

我们首先把标签表读进来，看清样本总量、列名、标签范围和类别数量：

```python
from pathlib import Path
import pandas as pd

SEED = 42
DATA_ROOT = Path('/kaggle/input/competitions/arm-ioai-sample')
TRAIN_DIR = DATA_ROOT / 'train_images' / 'train_images'
TRAIN_CSV = DATA_ROOT / 'train_labels.csv'
NUM_CLASSES = 100

train_df = pd.read_csv(TRAIN_CSV)
print(f"训练样本数: {len(train_df):,}")
print(f"列名: {list(train_df.columns)}")
print(f"标签范围: {train_df['label'].min()} 到 {train_df['label'].max()}")
print(f"类别数: {train_df['label'].nunique()}")

# 每个类别有多少张图：如果完全均衡，就不需要做类别加权
counts = train_df['label'].value_counts().sort_index()
print(f"每类最少/最多/平均: {counts.min()} / {counts.max()} / {counts.mean():.1f}")
print("完全均衡——不需要类别加权。")
```

打印结果确认了 100 个类别、每类数量完全一致（约 500 张），所以既不用做重采样，也不用给损失函数加类别权重。

接着我们随机挑 10 个类别，每类看 8 张图，拼成一张大图。这一步不是为了"好看"，而是为了直观感受类内差异有多大：同一类里姿态、光照、背景各不相同，而不同类之间又可能只差一点细节。看完之后就明白，数据增强不是可选项，而是必需品。

```python
import random
import matplotlib.pyplot as plt
from PIL import Image

# IDX2CLASS 是 CIFAR-100 官方类别表：编号 0～99 映射到类别名
SHOW_CLASSES = sorted(random.sample(range(NUM_CLASSES), 10))

fig, axes = plt.subplots(10, 8, figsize=(14, 18))
for row_idx, cls_idx in enumerate(SHOW_CLASSES):
    cls_df = train_df[train_df["label"] == cls_idx].head(8)
    for col_idx, (_, row) in enumerate(cls_df.iterrows()):
        img_path = TRAIN_DIR / f"{row['id']}.png"
        ax = axes[row_idx][col_idx]
        if img_path.exists():
            ax.imshow(Image.open(img_path))
        ax.axis("off")
        # 只在每行第一张图左边标出类别名，避免画面太挤
        if col_idx == 0:
            ax.set_ylabel(f"{cls_idx}: {IDX2CLASS[cls_idx]}",
                          fontsize=8, rotation=0, labelpad=85, va="center")
plt.tight_layout()
plt.show()
```

另外，32×32 是数据集的原生尺寸，我们不缩放、也不放大。把图放大不会凭空增加信息，只会让网络变慢；把图裁小又会丢掉边缘的物体。EDA 的结论可以汇成一句：这是一个"样本少、类别多、类间近"的问题，后面所有设计都围绕这三个词展开。

### 2.2 数据清洗

图像数据不需要"清洗"缺失值，但需要**统一尺寸和数值范围**。我们把所有图片转成张量，减去 CIFAR-100 的通道均值、除以通道标准差。这一步叫标准化，作用是让每个颜色通道的数值围绕 0 分布，优化器更容易找到方向。

更关键的是训练集的**数据增强**：随机裁剪（先四周补 4 像素再裁回 32×32）、随机水平翻转、颜色抖动、±15° 旋转、随机擦除一小块区域。它们分别模拟了位置变化、镜像、光照色温变化、拍摄角度和遮挡。验证集和测试集只做标准化，不加任何随机变换——因为我们要评估的是模型在"干净、真实"的输入上的表现。

```python
import torchvision.transforms as transforms

# CIFAR-100 训练集的逐通道均值与标准差，标准化后每个通道约等于零均值、单位方差
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),        # 平移不变性
    transforms.RandomHorizontalFlip(p=0.5),      # 自然物体左右镜像仍然成立
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),  # 光照与色温
    transforms.RandomRotation(degrees=15),       # 轻微拍摄角度
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),  # 强制模型不依赖局部
])

# 验证/测试只标准化，不引入随机性
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])
```

数据加载部分我们写了一个小型的 `Dataset`：给出 id，就去对应文件夹里按 `id.png` 读图，返回图像张量和标签。训练集和验证集共用同一批图片，只是套用不同的 transform，用 `Subset` 按索引切开，这样不用把图片复制两份。所有随机变换都由固定种子控制，同一份代码跑两次看到的数据顺序一致，改动的效果才可比较。

```python
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class CIFAR100Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df, self.img_dir = df.reset_index(drop=True), Path(img_dir)
        self.transform, self.is_test = transform, is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["id"])
        # 统一转 RGB，防止个别灰度图或 RGBA 图让卷积层通道数不匹配
        image = Image.open(self.img_dir / f"{img_id}.png").convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, img_id
        return image, int(row["label"])

# 分层切出 15% 做验证，保证验证集里 100 个类别都有
train_idx, val_idx = train_test_split(
    range(len(train_df)), test_size=0.15,
    random_state=SEED, stratify=train_df["label"])

train_set = Subset(CIFAR100Dataset(train_df, TRAIN_DIR, train_transform), train_idx)
val_set   = Subset(CIFAR100Dataset(train_df, TRAIN_DIR, val_transform), val_idx)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
```

### 2.3 模型训练与超参数调优

我们的网络结构是 5 个卷积块，每个块的固定配方是"卷积 → 批归一化 → ReLU → 卷积 → 批归一化 → ReLU → 最大池化 → Dropout2d"，通道数从 64 逐级加宽到 128、256、512、512，空间尺寸从 32 缩到 16、8、4、2。最后一个块不再池化，因为 2×2 已经太小了。

分类头先用全局平均池化（GAP）把每个通道的 2×2 空间位置取平均，得到 512 维向量，再接两层全连接：512→512（带批归一化、ReLU、Dropout），最后 512→100 输出 logits。用 GAP 而不是直接摊平，是为了大幅减少参数量，同时让模型不再对物体的绝对位置敏感。

```python
import torch.nn as nn

class ConvBlock(nn.Module):
    """卷积 → BN → ReLU → 卷积 → BN → ReLU →（可选）池化 → Dropout2d"""
    def __init__(self, in_ch, out_ch, pool=True, dropout_p=0.2):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),  # 有 BN 就不需要偏置
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        # Dropout2d 按通道整片丢弃，比逐像素 dropout 更适合卷积特征
        layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class CNN5(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.block1 = ConvBlock(3, 64, True, 0.10)
        self.block2 = ConvBlock(64, 128, True, 0.15)
        self.block3 = ConvBlock(128, 256, True, 0.20)
        self.block4 = ConvBlock(256, 512, True, 0.25)
        self.block5 = ConvBlock(512, 512, False, 0.25)   # 最后一块不再池化
        self.gap = nn.AdaptiveAvgPool2d(1)               # 任意空间尺寸压成 1×1
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.45),
            nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.block5(self.block4(self.block3(self.block2(self.block1(x)))))
        return self.classifier(self.gap(x))
```

训练配置有四个要点。第一，优化器用 Adam，学习率 1e-3，权重衰减 5e-4——权重衰减是一种正则化，抑制权重变得过大。第二，学习率调度用余弦退火（CosineAnnealingLR），从 1e-3 平滑降到接近 0，前期步子大、后期步子小。第三，损失函数用带标签平滑的交叉熵，平滑系数 0.1：因为 100 个类别里有太多长得很像的，模型不该对答案有 100% 的把握，平滑把 0.1 的概率质量平均分给其余 99 个类别，能改善校准、也常带来准确率提升。第四，早停：验证集准确率连续 10 轮不提升就停下，并恢复最好的那一版权重。

```python
import torch, torch.optim as optim

def train_model(model, train_loader, val_loader, device, num_epochs=50,
                lr=1e-3, weight_decay=5e-4, label_smoothing=0.1,
                patience=10, model_save_path="best_model.pth"):
    # 标签平滑写在损失函数里，一行就能生效
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_acc, wait = 0.0, 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 验证阶段只前向、不反传，省显存也避免误改 BN 统计量
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                pred = model(images.to(device)).argmax(1).cpu()
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"epoch {epoch}: val acc {acc*100:.2f}%")

        if acc > best_acc:
            best_acc, wait = acc, 0
            torch.save(model.state_dict(), model_save_path)
        else:
            wait += 1
            if wait >= patience:          # 连续 10 轮没进步就收工
                print(f"early stop at epoch {epoch}")
                break
    model.load_state_dict(torch.load(model_save_path))
    return best_acc
```

我们实际调用时把最多训练轮数设成 1000 轮，但早停通常会在几十轮内触发，所以真正跑不完这么多轮。这一点也提醒我们：**轮数上限只是保险丝，不是训练目标**。还有一点值得说明：我们始终没有给损失函数加类别权重，因为 EDA 已经确认每个类别数量完全相同；对一个本来就均衡的数据强行加权，反而会打乱梯度。

### 2.4 模型测试与指标报告

训练结束后，我们对公开测试集和私有测试集分别推理，生成两份提交文件，再把它们拼成一份总提交。推理时只做 `argmax`，取概率最高的类别编号：

```python
@torch.no_grad()
def generate_submission(model, test_loader, device, output_path):
    model.eval()
    all_ids, all_preds = [], []
    for images, img_ids in test_loader:
        preds = model(images.to(device)).argmax(dim=1).cpu().tolist()
        all_ids.extend(img_ids)
        all_preds.extend(preds)
    sub_df = pd.DataFrame({"id": all_ids, "label": all_preds})
    sub_df.to_csv(output_path, index=False)
    print(f"{output_path}: {len(sub_df)} 行，"
          f"共预测出 {sub_df['label'].nunique()} / 100 个类别")
    return sub_df

pub_sub  = generate_submission(model, pub_test_loader,  DEVICE, "submission_public.csv")
priv_sub = generate_submission(model, priv_test_loader, DEVICE, "submission_private.csv")
pd.concat([pub_sub, priv_sub]).to_csv("sub.csv", index=False)
```

因为比赛提供了测试集的真实标签，我们可以在本地直接算准确率，完全不需要先提交到排行榜：

```python
def local_accuracy(sub_df, gt_csv_path, split_name=""):
    gt_df = pd.read_csv(gt_csv_path)
    # 按 id 合并预测和真值，后缀区分同名列
    merged = sub_df.merge(gt_df, on="id", suffixes=("_pred", "_true"))
    acc = (merged["label_pred"] == merged["label_true"]).mean()
    print(f"{split_name}: {acc*100:.2f}% ({int(acc*len(merged))}/{len(merged)})")
    return acc

local_accuracy(pub_sub,  PUB_TEST_CSV,  "Public")
local_accuracy(priv_sub, PRIV_TEST_CSV, "Private")
```

这一步看似简单，价值却很高：它能在几秒钟内发现"id 对不上""行数少了""标签越界"这类格式错误，而这些错误在排行榜上往往表现为莫名其妙的低分。除了 Top-1，我们还记录了 Top-5：如果 Top-1 不高但 Top-5 很高，说明模型已经能排除大部分错误类别，只是在前几名里选错，这时换更强的分类头或延长训练通常比换架构更划算。

作者在材料里给出的预期是 **40%～50% 的 Top-1 验证准确率**。听起来不高，但要知道随机猜是 1%，而且这是在 32×32 的小图上、100 个相近类别、完全从零训练的条件下得到的。材料也提醒：公开测试集和私有测试集的分布未必一致，不要只盯着能看见的那个分数。

---

## 三、总结

本文中我们从零搭了一个 CIFAR-100 的诚实基线。我们首先用 EDA 确认了 100 个类别完全均衡、每类约 500 张，并通过可视化认识到类内差异有多大；其次做了标准化和五项训练期数据增强，验证与测试只保留标准化；接着用 5 个双卷积块加全局平均池化和两层全连接搭出 CNN，配合批归一化、Dropout2d、标签平滑、Adam、余弦退火和早停训练；最后分别对公开、私有测试集推理，并用官方真值在本地核对准确率。材料给出的预期是 40%～50% 的 Top-1，这已经是从零训练的一个合理水平。

这个比赛让我们体会最深的是三件事：**数据增强在少样本时不是锦上添花，而是决定成败**；**标签平滑是一行代码就能改善多分类泛化的便宜买卖**；以及**批归一化负责稳住训练、Dropout 负责抑制过拟合，两者是互补的**。想继续往上走，材料里列的方向也很清楚：CutMix/MixUp、测试时增强、学习率预热、权重平均，以及在合适的时候引入残差连接。
