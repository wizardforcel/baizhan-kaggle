# 墨迹检测挑战（ELIVA 26 Ink Detection）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的任务是**墨迹检测（Ink Detection）**：给一小块古代手稿或卷轴的"表面体积"扫描，判断每一个像素上有没有墨水。

所谓"表面体积"，可以理解成一摞从上到下拍下来的切片。扫描设备把样本一层一层地"切"开成像，每一层存成一张 TIFF 图片，合起来就是一个三维数据块。墨迹不一定只出现在最表面：它可能渗进纤维内部，所以在不同的深度切片上，墨迹的明暗表现会变化。只看一张切片往往不够，要把上下若干层放在一起看，才能判断"这里到底有没有字"。

这和著名的"维苏威挑战"（Vesuvius Challenge）属于同一类问题：用 CT 或显微扫描去"读"那些已经碳化、不敢展开的古卷。ELIVA 26 把这个任务做成了一个 Kaggle 练习赛，数据组织得更简单：每个 patch（小块区域）一个文件夹，里面有切片、有效区域掩码和墨迹标注。

### 1.2 比赛目标

训练集和测试集都按 patch 组织。每个 patch 目录里通常有三样东西：

- `surface_volume/`：一叠 `.tif` 切片，文件名是数字（`0.tif`、`1.tif`……），数字越小表示越靠近表面；
- `mask.png`：有效区域掩码，标出这块扫描里哪些像素是真实数据、哪些是边缘空白；
- `inklabels_rle.csv`（只有训练集有）：墨迹标注，用 RLE 字符串表示。

我们要输出的是每个测试 patch 的墨迹掩码，按 RLE 编码写进提交文件。换句话说，输入是"一摞切片 + 有效区域"，输出是"同样大小的 0/1 掩码"。

### 1.3 评价标准

从提交格式（像素级 RLE 掩码）可以看出，这是一个**像素级二分类分割**任务。评分关注的应该是预测的墨迹区域和真实墨迹区域**重合得有多好**，常用的是 F1 分数或 Dice 系数：

```
F1 = 2 × 预测对的墨迹像素数 / (预测墨迹总数 + 真实墨迹总数)
```

取值在 0 到 1 之间，越高越好。它和准确率（accuracy）的区别很关键：一张图里墨迹像素往往只占很小一部分，如果模型全预测"没有墨迹"，准确率可能高达 95% 以上，但 F1 是 0。所以这类任务绝不能只看准确率。

材料没有给出比赛确切的指标名称和权重，我们按"分割重合度"来理解，不编造具体公式。

---

## 二、解题过程

材料提供的是一份**数据可视化**笔记，没有模型训练代码。所以我们的路线是：先把数据格式彻底搞清楚（尤其是 RLE 编解码和切片顺序），再基于这个格式整理出一套最小可行的分割方案。训练部分会明确标注为"伪代码"，不是原笔记的执行结果。

### 2.1 数据探索（EDA）

我们首先定位训练集和测试集，列出所有 patch 目录：

```python
from pathlib import Path

TRAIN_SPLIT_DIR = Path('/kaggle/input/competitions/eliva-26-ink-detection/train')
TEST_SPLIT_DIR = Path('/kaggle/input/competitions/eliva-26-ink-detection/test')

# 只保留目录，过滤掉可能混在里面的杂项文件
train_patches = sorted([p for p in TRAIN_SPLIT_DIR.iterdir() if p.is_dir()])
test_patches = sorted([p for p in TEST_SPLIT_DIR.iterdir() if p.is_dir()])

print('训练 patch 数:', len(train_patches))
print('测试 patch 数:', len(test_patches))
```

知道 patch 数量很重要：如果训练 patch 只有个位数，就不能指望模型学到很强的泛化能力，交叉验证也必须按 patch 划分，不能按像素随机划分（同一块区域的相邻像素几乎一模一样，随机划分等于把答案抄进验证集）。

其次，我们确认切片是按**数字顺序**排列的。文件名是 `0.tif`、`1.tif`、`10.tif` 这样的字符串，如果直接按字符串排序，`10.tif` 会排在 `2.tif` 前面，深度顺序就乱了：

```python
def numeric_slice_key(path: Path):
    # 文件名是纯数字时按整数排，否则退回按名字排
    return int(path.stem) if path.stem.isdigit() else path.name

def first_tiff_path(patch_dir: Path) -> Path:
    # 取表面体积里的第一张切片（数字最小的那张）
    tiffs = sorted((patch_dir / 'surface_volume').glob('*.tif'),
                   key=numeric_slice_key)
    if not tiffs:
        raise FileNotFoundError('这个 patch 没有 TIFF 切片')
    return tiffs[0]
```

这个小函数是整个流程的地基。后面无论是"取前 K 层做输入"还是"取中间某层做可视化"，都必须保证顺序是数值顺序，否则模型看到的深度关系是错的。

接着，我们把一个 patch 的三样东西同时读出来看：墨迹标注、有效区域、第一张切片。先读标注需要解码 RLE：

```python
def rle_decode(rle, shape):
    """把 Kaggle 风格的 RLE 字符串解成二维 0/1 掩码"""
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # 标注可能为空：None、NaN 或空字符串都表示"整块没有墨迹"
    if rle is None or (isinstance(rle, float) and np.isnan(rle)):
        return flat.reshape(shape)
    rle = str(rle).strip()
    if not rle:
        return flat.reshape(shape)

    # RLE 是"起点, 长度, 起点, 长度..."，起点从 1 开始数
    values = np.asarray([int(v) for v in rle.split()], dtype=np.int64)
    starts = values[0::2] - 1     # 转成从 0 开始的数组下标
    lengths = values[1::2]
    for start, length in zip(starts, lengths):
        flat[start:start + length] = 1
    return flat.reshape(shape)
```

RLE 是 "Run-Length Encoding"（游程编码）的缩写。它不存每个像素，只存"从第几个像素开始、连续多少个 1"。因为墨迹是成片出现的，这种编码能大幅压缩体积。**最容易错的地方是下标**：Kaggle 的 RLE 起点从 1 开始，而 Python 数组从 0 开始，所以必须减 1。

然后我们把它和 `mask.png`、第一张切片放在一起显示：

```python
from PIL import Image
import matplotlib.pyplot as plt

def load_patch_triplet(patch_dir: Path):
    # 有效区域掩码：大于 0 的像素才算真实数据
    mask = np.asarray(Image.open(patch_dir / 'mask.png')) > 0

    # 墨迹标注：形状必须和 mask 一样，才能逐像素对应
    rle_path = patch_dir / 'inklabels_rle.csv'
    df = pd.read_csv(rle_path)
    label = rle_decode(df.loc[0, 'Predicted'], mask.shape)

    # 第一张表面切片，仅用于观察
    tiff_path = first_tiff_path(patch_dir)
    first_slice = np.asarray(Image.open(tiff_path))
    return label, mask, first_slice, tiff_path

def show_patch_triplet(patch_dir: Path):
    label, mask, first_slice, tiff_path = load_patch_triplet(patch_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(label * 255, cmap='gray')
    axes[0].set_title('inklabels_rle.csv')
    axes[1].imshow(mask * 255, cmap='gray')
    axes[1].set_title('mask.png')
    axes[2].imshow(first_slice, cmap='gray')
    axes[2].set_title('first surface TIFF')
    plt.tight_layout()
    plt.show()

for patch_dir in train_patches:
    show_patch_triplet(patch_dir)
```

三张图并排看，能得到几个直观认识：第一，墨迹标注只出现在有效区域内部，边缘空白处没有标签；第二，第一张切片通常偏暗、对比度低，肉眼不一定能直接看出字；第三，标注是"稀疏"的，白色像素只占一小块。这三点直接决定了后面的处理方式：**损失函数要能对付类别不平衡，评估必须在有效区域内做，输入要多取几层而不是只看第一层**。

### 2.2 数据清洗

这份数据的"清洗"主要是格式对齐，而不是去噪。

第一，**统一形状**。标注、掩码、切片的尺寸必须完全一致。我们从 `mask.png` 取形状，再把 RLE 解成同样形状；如果某张切片的尺寸和掩码对不上，就要报错而不是硬塞：

```python
def load_rle_label(patch_dir: Path, shape):
    rle_path = patch_dir / 'inklabels_rle.csv'
    df = pd.read_csv(rle_path)
    if 'Predicted' not in df.columns:
        raise ValueError(f'{rle_path} 里没有 Predicted 列')
    # 取第一行的 RLE 字符串解码；空值会在 rle_decode 里变成全 0
    return rle_decode(df.loc[0, 'Predicted'], shape)
```

第二，**处理空标注**。有些 patch 可能整块没有墨迹，RLE 是空的。如果代码没考虑这种情况，`int('')` 会直接抛异常。我们在 `rle_decode` 里对 `None`、`NaN` 和空字符串都返回全 0 掩码，保证流程不中断。

第三，**有效区域之外不算分**。`mask.png` 为 0 的地方是扫描边缘的空白，那里既没有纤维也没有墨迹。训练时这些像素要排除在损失之外，预测时也要把结果乘上掩码，否则模型会在无意义的区域浪费容量，甚至学会"边缘全是墨"的假规律。

```python
# 训练/预测时都只保留有效区域内的像素
valid = mask.astype(bool)
# 伪代码：计算损失前先展平，只取 valid 为 True 的位置
# loss = criterion(pred[valid], label[valid])
```

还有一点值得注意：切片是 16 位或 8 位的灰度图，不同 patch 的亮度范围可能不一样。常见做法是对每个 patch 单独做归一化（比如减去均值、除以标准差），而不是拿全局固定值。因为扫描条件不同，绝对亮度不可比，相对明暗才有意义。

### 2.3 模型训练与超参数调优

材料里**没有训练代码**，所以这一节是我们根据数据格式整理的伪代码方案，目的是说明"拿到这种三维数据该怎么下手"，不代表原笔记跑过或得到过某个分数。

思路是 **2.5D 分割**：不直接把整个三维体积塞进 3D 网络（显存吃不消），而是每次取同一个 (x, y) 位置上、上下相邻的 K 张切片，叠成 K 个通道，当成一张"多通道图片"输入二维 U-Net。这样既利用了深度信息，又保持了二维卷积的高效。

```python
# 伪代码：2.5D 分割模型的结构示意
import torch
import torch.nn as nn

class InkUNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 编码器：逐层下采样，提取从细到粗的特征
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 解码器：逐层上采样，把特征还原到原分辨率
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=2),
        )

    def forward(self, x):
        # 输出每个像素"有墨"的分数，后面接 sigmoid
        return self.decoder(self.encoder(x))
```

输入张量的构造方式是：对每个 patch，把 K 张连续切片沿通道维堆起来，形状是 `(K, H, W)`。K 取多少要在验证集上试——太小看不到墨迹的深度变化，太大显存吃紧且相邻层高度冗余。

损失函数用 **BCE + Dice** 的组合。BCE（二元交叉熵）逐像素判断"是不是墨"，Dice 则直接优化"预测区域和真实区域的重合度"，正好对应 F1。因为墨迹像素很少，单用 BCE 容易让模型偏向"全预测背景"，加上 Dice 能把注意力拉回正类：

```python
# 伪代码：组合损失
bce = nn.BCEWithLogitsLoss()
def dice_loss(pred, target, eps=1e-6):
    # pred 先过 sigmoid 变成 0~1 的概率
    p = torch.sigmoid(pred)
    intersection = (p * target).sum()
    return 1 - (2 * intersection + eps) / (p.sum() + target.sum() + eps)

loss = bce(pred, target) + dice_loss(pred, target)
```

数据划分必须**按 patch 分**，不能按像素分。同一个 patch 里相邻像素共享几乎相同的切片邻域，随机按像素切会让验证集"见过"训练集的纹理，分数虚高。我们按 patch 编号做 K 折，留出的整块 patch 完全不参与训练。

超参数方面，材料没有给出可参考的数值，我们也不编造。实际做法应该是：固定一个较小的学习率、按验证集 Dice 做早停、对 K（切片数）、输入尺寸、损失权重做小范围网格搜索。因为这类任务训练一轮很慢，网格不宜大，先保证"能稳定训练、验证 Dice 不崩"，再谈精调。

### 2.4 模型测试与指标报告

推理时，我们对每个测试 patch 取同样的 K 层切片，前向得到概率图，再按验证集选定的阈值二值化，最后乘上 `mask.png` 去掉无效区域：

```python
# 伪代码：测试 patch 的预测与后处理
with torch.no_grad():
    logits = model(batch_slices)          # 形状 (1, 1, H, W)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

# 阈值不是固定 0.5，而是在验证集上选让 Dice 最高的那个
binary = (prob >= best_threshold).astype(np.uint8)
binary *= mask.astype(np.uint8)           # 有效区域之外强制为 0
```

二值掩码要转成 RLE 才能提交。编码是解码的逆过程，材料里的写法很巧妙：

```python
def rle_encode(mask):
    """把 0/1 掩码编码成 Kaggle 风格的 RLE 字符串"""
    pixels = mask.astype(np.uint8).ravel()
    # 首尾各补一个 0，这样"变化点"一定成对出现
    pixels = np.concatenate([[0], pixels, [0]])
    # 找出所有值发生变化的位置（从 1 开始计数）
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # 偶数位是起点、奇数位是长度：长度 = 下一个起点 - 当前起点
    changes[1::2] -= changes[::2]
    return ' '.join(str(x) for x in changes)
```

我们逐行解释这段"看起来像魔法"的代码。假设掩码是 `0 0 1 1 1 0`，首尾补 0 后变成 `0 0 0 1 1 1 0 0`。值发生变化的位置是第 3 位（0→1）和第 6 位（1→0），所以 `changes = [3, 6]`。把偶数位（长度位）减去前一个起点：`6 - 3 = 3`，得到 `[3, 3]`，意思是"从第 3 个像素开始，连续 3 个 1"。这正好是 Kaggle RLE 要求的"1 起始下标 + 长度"格式。

生成提交文件前，我们做两项自检：第一，把编码后的字符串再解码回来，和原掩码逐像素比对，必须完全一致；第二，确认每个 patch 的 RLE 长度之和等于掩码里 1 的个数：

```python
# 往返校验：编码再解码，应该和原掩码完全相同
assert np.array_equal(rle_decode(rle_encode(binary), binary.shape), binary)

# 长度校验：所有 run 的长度加起来，等于墨迹像素总数
runs = [int(v) for v in rle_encode(binary).split()]
assert sum(runs[1::2]) == int(binary.sum())
```

关于分数，材料只有可视化代码，没有任何训练结果或排行榜成绩，所以我们不报告 Dice/F1 的具体数字。能确定的评估方法是：在留出的整块 patch 上，只统计有效区域内的像素，计算 Dice 或 F1；同时单独看"有墨 patch"和"无墨 patch"的表现——如果模型在无墨 patch 上大量误报，总分会被拉低，但在平均指标里不容易看出来。

---

## 三、总结

本文中我们完成了 ELIVA 26 墨迹检测任务的数据解析与方案设计。我们首先列出训练和测试的 patch 目录，用数值排序保证 TIFF 切片的深度顺序正确；其次实现了 Kaggle 风格 RLE 的解码与编码，处理了空标注、形状对齐和有效区域掩码，明确了"掩码之外不计分"；接着把标注、掩码和第一张切片并排可视化，确认墨迹稀疏、单层切片对比度低，从而定下"多切片输入 + BCE 与 Dice 组合损失 + 按 patch 划分验证"的 2.5D 分割思路；最后给出推理、阈值选择、掩码后处理和 RLE 往返校验的完整流程。

这个比赛让我们学到的是：**先把数据格式吃透，再谈模型**。RLE 的 1 起始下标、切片文件名的数值排序、有效区域掩码，这些细节任何一个弄错，模型训得再好也交不出正确的提交文件。反过来，只要格式对齐、验证划分不泄漏，哪怕是一个结构简单的 U-Net，也能成为一个扎实的起点。
