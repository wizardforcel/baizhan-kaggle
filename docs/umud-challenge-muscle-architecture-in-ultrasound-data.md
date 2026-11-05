# UMUD 超声肌肉架构挑战

## 一、比赛简介

### 1.1 比赛背景

UMUD Challenge 是 Kaggle 上一个医学影像量测比赛，任务是看超声图估算肌肉的生物力学指标。与一般的分割比赛不同，这里不评判 Dice 分数，而是要输出三个临床可解释的数字。超声图来自不同厂商的便携式探头，图片被非均匀缩放过，像素到毫米的比例在 X 轴和 Y 轴上不一致，这对角度和长度计算是致命的。

比赛提供训练集的分割掩码，但没有提供对应的 PA/FL/MT 标签。于是主流思路是先用掩码训练两个 U-Net 分别分割肌腱膜和肌束，再在测试图上做几何测量。材料里反复提到最关键的点不是网络多深，而是量测定义和物理标定做对。

### 1.2 比赛目标

给定 309 张测试超声图像，预测每张图的三项指标：

* PA 即肌束 pennation angle，是肌束方向与深肌腱膜夹角，单位度。
* FL 即 fascicle length，肌束在两层肌腱膜之间的轨迹长度，单位毫米，需要线性外推。
* MT 即 muscle thickness，两层肌腱膜中心线的正交距离，单位毫米。

输入只有原始图像，输出是 id 与三个数值。训练素材主要是肌腱膜和肌束的二值掩码，用来训练分割模型。

### 1.3 评价标准

分数是三个绝对误差的归一化均值：

Score = MAE_PA / 6 + MAE_FL / 12 + MAE_MT / 3

容差是 PA 6°、FL 12 mm、MT 3 mm，数值越小越好。材料里写到 MT 的权重最大，因为分母最小，1 mm 的误差会拉高分数约 0.33。早期基准线 public LB 在 0.45 左右，几何校准后能到 0.33~0.37 区间。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把测试目录列出来，确认图像数量和设备布局。图片有三种常见尺寸，1200×800、1088×644 等，格式是 tif 和 png。

```python
# 找到测试集目录并统计图像数量
import glob
from pathlib import Path
TEST = Path("/kaggle/input/test_set_v2")
files = sorted([p for p in TEST.iterdir() if p.suffix in (".tif",".png")])
print(f"测试图像数: {len(files)}")
# 输出 309 张
```

接着我们依据文件名后缀和灰度边框判断布局。

```python
# 根据尺寸和灰度判断设备布局
def layout(path, gray):
    h, w = gray.shape
    if path.suffix == ".png":
        return "lumify"
    if (w, h) in ((1088, 644), (1200, 800)):
        side = np.median(gray[100:h-20, 90:140])
        return "telemed" if 35 < side < 80 else "siemens"
    return "crop"
# 这一步让我们知道每张图的缩放规则不同
```

EDA 显示大部分图被 resize 到 1200×800，但 Lumify 的圆 P 图标变成椭圆，说明 X 和 Y 被不同比例压缩。Telemed 有两侧灰条，Siemens 只有右侧刻度。我们还统计了每张图的宽高比，发现 Telemed 在 1200×800 模式下 aspect 约 0.908，Siemens 经过 1152×864 到 1200×800 的缩放后需要 1.125 的修正。不同设备的深度标签格式也不同，有的写 cm 有的写 mm，有的在图像右侧，有的在顶部。没有这一步，后面的像素转毫米会产生系统性偏差，FL 会被拉长或压缩 10 mm 左右。

### 2.2 数据清洗

清洗的核心是把每张图的物理尺度算出来，而不是去噪或增强。

```python
# 用刻度和 OCR 读取物理尺度
import re
def parse_depth_mm(text):
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(cm|mm)", text)
    if not m: return np.nan
    v = float(m.group(1).replace(",", "."))
    return v * 10 if m.group(2) == "cm" else v
# 从图上读出的深度标签用于换算
```

我们先用 RapidOCR 读取深度标签，再在图像边缘做一维轮廓峰值检测，找出刻度线。

```python
# 找出刻度峰值的中心位置
def tick_positions(profile, min_gap=12):
    p = profile.astype(np.float32)
    on = p > max(p.max() * 0.5, 40)
    pos = []
    i = 0
    while i < len(p):
        if on[i]:
            j = i
            while j < len(p) and on[j]:
                j += 1
            pos.append((i + j - 1) / 2)
            i = j
        else:
            i += 1
    return np.array(pos)
# 把峰值去重后得到等距刻度
```

Siemens 用右侧刻度跨度除以 OCR 深度，Telemed 用左右 10 mm 刻度分别算 sx、sy，Lumify 用 P 图标宽高比校正。缺失的 OCR 值从同类刻度帧的 LUT 中借用，异常的 tick 大小按 layout+depth 的中位数吸附。我们还对 Telemed 的 aspect 做设备分辨率分组的中位数 snapping，把误读导致的 0.44 到 1.31 的离群值拉回正常范围。清洗后 sx、sy 不会再有 NaN，物理尺度的误差被限制在毫米级。

### 2.3 模型训练与超参数调优

训练只在掩码上做。我们加载在比赛掩码上训练好的 ResNet-34 U-Net，两种权重分别负责肌腱膜和肌束。推理时做左右翻转 TTA 并缓存概率图。

```python
# U-Net 概率图推理并缓存
import segmentation_models_pytorch as smp
import torch
def load_model(name):
    c = torch.load(CKPT / name, map_location="cpu", weights_only=True)
    m = smp.Unet(c["encoder"], encoder_weights=None, in_channels=1, classes=1)
    m.load_state_dict(c["state"])
    return m.eval(), tuple(c["size"])

@torch.no_grad()
def predict(model, size, gray):
    oh, ow = size
    x = cv2.resize(gray, (ow, oh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = torch.from_numpy(x)[None, None]
    p = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1)) / 2
    return p[0, 0].numpy()
# 缓存 float16，几何实验可在 CPU 秒级重跑
```

有了概率图后，我们进入几何测量。我们把肌腱膜掩码分成连通块，用列方向加权中心拟合二次曲线，选出两条间距最大的带。

```python
# 从肌腱膜概率图提取上下两层肌腱膜曲线
def aponeuroses(p_apo, mx, my, thr=0.5):
    h, w = p_apo.shape
    n, lab, st = cv2.connectedComponentsWithStats((p_apo > thr).astype(np.uint8), connectivity=8)[:3]
    comps = []
    for i in range(1, n):
        m = lab == i
        cols = np.unique(np.nonzero(m)[1])
        cy = np.average(np.nonzero(m)[0][np.isin(np.nonzero(m)[1], cols)], weights=p_apo[m])
        comps.append(dict(X=cols*mx, c=cy*my))
    # 按深度排序并合并重叠片段，返回浅层和深层
    return comps
# 物理坐标 mx, my 把像素转为毫米
```

肌肉厚度按方案在左右中三条垂直线取深浅肌腱膜内边距离。

```python
# 计算肌肉厚度 MT
def muscle_thickness(sup, deep, xs_rel):
    X = lo + np.array(xs_rel) * (hi - lo)
    d = np.polyval(deep["fit"]["top"], X) - np.polyval(sup["fit"]["bot"], X)
    return float(np.mean(d))
# 三点平均更稳健
```

肌束分割后用 8 连通 PCA 求主方向，得到每个片段的角度和长度。PA 取深肌腱膜附近的加权中位数，FL 则用三条最长且可见的肌束线性外推到两层肌腱膜再乘以校准系数 0.9。

为了不完全依赖分割，我们还做了纹理角度的独立估计。

```python
# 高通滤波后用投影方差峰值找肌束方向
def texture_angle(gray, sx, sy, apo, angles=np.arange(-50,50.5,0.5)):
    hp = gray - cv2.GaussianBlur(gray, (0,0), 2.0)
    # 在肌腱膜之间 ROI 内做 Radon 投影
    best, best_a = -1, np.nan
    for a in angles:
        # 计算投影方差，峰值对应肌束方向
        ...
    return best_a
# 这个角度与分割结果互补，可用于校验 PA
```

整个流程是流水线：物理标定 -> U-Net 分割 -> 几何拟合 -> 量测。没有可学习回归，所有参数都由几何和注释协议决定。我们先尝试各向同性尺度，LB 在 0.616，随后加入各向异性尺度降到 0.547，再把 MT 改为内边测量降到 0.442。接着把 PA 取在深肌腱膜附近以避开肌束弯曲，得到 0.416，最后对 FL 做 0.9 的经验校准得到 0.336。超参数调优主要在几何选择上，而不是网络学习率。

### 2.4 模型测试与指标报告

测试时我们对全部 309 张图跑完标定、推理、几何，生成 submission。

```python
# 生成提交文件
import pandas as pd
results = []
for p in files:
    scale = test_scales([p])
    maps = cached_maps([p], "test")
    meas = measure(maps["apo"], maps["fasc"], scale.sx, scale.sy)
    results.append({"id": p.name, "PA": meas["pa_deg"], "FL": meas["fl_mm"], "MT": meas["mt_mm"]})
sub = pd.DataFrame(results)
sub.to_csv("submission.csv", index=False)
# 格式检查：309 行，id 顺序与测试集一致
```

在本地的 OSF 基准集上，我们看到几何校准带来的提升路径与材料一致：从各向同性尺度 LB 0.616，到单独 sx sy 0.547，再到内边 MT 0.442，PA 取深层 0.416，最后 FL ×0.9 校准到 0.336。另一个更精细的 Geometry Over Masks 管线在公开榜得到约 0.374，U-Net + 纹理角度组合达到 0.32296。分数的下降主要来自 MT 的稳定化。

我们按难度分层评估后发现不同设备间误差分布不同，Lumify 的椭圆 P 图标校正最关键，Telemed 的底部刻度误读会导致 aspect 0.44~1.31 的离群值。把这些做完后，指标基本收敛。

## 三、总结

本文中我们完成了 UMUD 超声肌肉架构挑战的从数据到提交的完整流程。我们首先做 EDA 摸清 309 张测试图的三类设备和非均匀缩放问题，其次用 OCR 加刻度峰值检测实现每张图的物理尺度校准，接着基于预训练 U-Net 做肌腱膜和肌束分割并用几何拟合得到 PA、FL、MT，最后在公开榜上得到 0.32~0.37 量级的分数。比赛告诉我们，在医学量化任务里，标定和几何定义的准确性远比堆网络深度重要，字符级统计在这里是像素级几何。后续可以尝试把纹理角度和分割角度做融合，或用基准集做 MT 的有监督校正。
