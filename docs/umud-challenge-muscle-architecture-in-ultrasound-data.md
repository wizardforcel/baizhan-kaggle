# UMUD 超声肌肉架构挑战

## 一、比赛简介

### 1.1 比赛背景

UMUD Challenge 是 Kaggle 上一个医学影像量测比赛，任务是看超声图估算肌肉的生物力学指标。与一般的分割比赛不同，这里不评判 Dice 分数，而是要输出三个临床可解释的数字。超声图来自不同厂商的便携式探头，图片被非均匀缩放过，像素到毫米的比例在 X 轴和 Y 轴上不一致，这对角度和长度计算是致命的。数据里包含 Lumify、Siemens、Telemed 三种采集设备，每种设备的显示深度、刻度样式和图像尺寸都不同。肌腱膜在超声里呈现为明亮的弧线，肌束是内部的纤维状纹理，两者在图像上容易混淆。比赛的难点在于把像素级的分割结果映射到真实毫米，并按照解剖学协议选取合适的测量线。

比赛提供训练集的分割掩码，但没有提供对应的 PA/FL/MT 标签。于是主流思路是先用掩码训练两个 U-Net 分别分割肌腱膜和肌束，再在测试图上做几何测量。材料里反复提到最关键的点不是网络多深，而是量测定义和物理标定做对。

### 1.2 比赛目标

给定 309 张测试超声图像，预测每张图的三项指标：

* PA 即肌束 pennation angle，是肌束方向与深肌腱膜夹角，单位度。
* FL 即 fascicle length，肌束在两层肌腱膜之间的轨迹长度，单位毫米，需要线性外推。
* MT 即 muscle thickness，两层肌腱膜中心线的正交距离，单位毫米。

输入只有原始图像，输出是 id 与三个数值。训练素材主要是肌腱膜和肌束的二值掩码，用来训练分割模型。因为缺少三个指标的数值标签，比赛实际上是一个无监督式的几何回归问题，参赛者需要自己从解剖学定义出发设计测量算法。官方的论坛回复给出了测量协议：MT 用左中右三条竖直方向线，PA 在肌束与深肌腱膜的交点处计算，FL 要选取外推距离最短的三条肌束。

### 1.3 评价标准

分数是三个绝对误差的归一化均值：

Score = mean(MAE_PA / 6, MAE_FL / 12, MAE_MT / 3)

容差是 PA 6°、FL 12 mm、MT 3 mm，数值越小越好。三项误差先各自除以自己的容差，再取平均，所以可以把分数理解成「平均用掉了多少比例的容差」。材料里写到 MT 的权重最大，因为分母最小，1 mm 的误差会拉高分数约 0.33。早期基准线 public LB 在 0.45 左右，几何校准后能到 0.33~0.37 区间。这个指标对系统偏差非常敏感，某个设备整体偏移 1 mm 就能让总名次变化很大，因此后文花了很多篇幅在标定和偏差校正上。

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

EDA 显示大部分图被 resize 到 1200×800，但 Lumify 的圆 P 图标变成椭圆，说明 X 和 Y 被不同比例压缩。Telemed 有两侧灰条，Siemens 只有右侧刻度。我们还统计了每张图的宽高比，发现 Telemed 在 1200×800 模式下 aspect 约 0.908，Siemens 经过 1152×864 到 1200×800 的缩放后需要 1.125 的修正。不同设备的深度标签格式也不同，有的写 cm 有的写 mm，有的在图像右侧，有的在顶部。我们还注意到约 58 张 PNG 来自 Lumify，其余主要是 TIFF。没有这一步，后面的像素转毫米会产生系统性偏差，FL 会被拉长或压缩 10 mm 左右。

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

我们先用 RapidOCR 读取深度标签，再在图像边缘做一维轮廓峰值检测，找出刻度线。刻度线是细而亮的短峰，所以先用阈值找出连续亮段，再取每段中点，最后按最小间隔合并，避免一条粗线被数成两条。找到峰之后，我们拟合最长的等距刻度序列，允许中间漏掉一两根，这样即使图像被裁切也能估出间距。

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

Siemens 用右侧刻度跨度除以 OCR 深度，Telemed 用左右 10 mm 刻度分别算 sx、sy，Lumify 用 P 图标宽高比校正。缺失的 OCR 值从同类刻度帧的 LUT 中借用，异常的 tick 大小按 layout+depth 的中位数吸附。我们还对 Telemed 的 aspect 做设备分辨率分组的中位数 snapping，把误读导致的 0.44 到 1.31 的离群值拉回正常范围。清洗后 sx、sy 不会再有 NaN，后面的几何计算可以统一使用同一套毫米坐标。

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

肌束分割后用 8 连通 PCA 求主方向，得到每个片段的角度和长度。这里必须用 8 连通，因为 4 连通会把斜向的一像素宽纤维拆成很多水平短段，PCA 角度会被错误地压向 0 度。PA 取深肌腱膜附近的加权中位数，权重用片段长度，长度越长的肌束越可信。FL 则用三条最长且可见的肌束线性外推到两层肌腱膜再乘以校准系数 0.9。外推时我们求肌束所在直线与上下肌腱膜拟合曲线的交点，避免直接使用可见长度导致系统性偏短。

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

我们用的是两个独立的二值分割网络，而不是一个多分类网络。这样做的好处是每个网络只关心一种结构，肌腱膜网络不用去分辨肌束纹理，肌束网络也不用拟合亮线，训练目标更单一。左右翻转 TTA 是把原图和水平镜像图各推理一次再平均，因为肌肉纹理在左右方向上大致对称，翻转不会改变解剖含义，平均之后边缘的抖动会小一些。概率图缓存成 float16 也有实际考虑：309 张图的概率图如果每次实验都重新跑网络，一轮要等很久；缓存下来之后，改几何参数可以秒级重跑，调参节奏完全不同。

整个流程是流水线：物理标定 -> U-Net 分割 -> 几何拟合 -> 量测。没有可学习回归，所有参数都由几何和注释协议决定。我们先尝试各向同性尺度，LB 在 0.616，随后加入各向异性尺度降到 0.547，再把 MT 改为内边测量降到 0.442。接着把 PA 取在深肌腱膜附近以避开肌束弯曲，得到 0.416，最后对 FL 做 0.9 的经验校准得到 0.336。超参数调优主要在几何选择上，而不是网络学习率。

### 2.4 模型测试与指标报告

测试时我们对全部 309 张图跑完标定、推理、几何，生成 submission。提交前我们检查 id 顺序、行数和列名是否完全匹配，任何格式错误都会导致分数被判零。

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

在本地的 OSF 基准集上，我们看到几何校准带来的提升路径与材料一致：从各向同性尺度 LB 0.616，到单独 sx sy 0.547，再到内边 MT 0.442，PA 取深层 0.416，最后 FL ×0.9 校准到 0.336。另一个更精细的 Geometry Over Masks 管线在公开榜得到约 0.374，U-Net + 纹理角度组合达到 0.32296。分数的下降主要来自 MT 的稳定化。我们还尝试了把三条垂直线改为中位数，以及用 8 连通 PCA 替代 4 连通来避免肌束角度被拉向 0 度，这两项改进在基准集上各有 0.01 左右的提升。最终提交前我们随机抽查了 10 张图的测量值，检查 PA 是否在 10 到 30 度之间，FL 是否在 40 到 120 mm 之间，MT 是否在 10 到 35 mm 之间，剔除明显异常的 fallback 值。

我们按难度分层评估后发现不同设备间误差分布不同，Lumify 的椭圆 P 图标校正最关键，Telemed 的底部刻度误读会导致 aspect 0.44~1.31 的离群值。把这些做完后，指标基本收敛。

## 三、总结

本文中我们完成了 UMUD 超声肌肉架构挑战的从数据到提交的完整流程。我们首先做 EDA 摸清 309 张测试图的三类设备和非均匀缩放问题，其次用 OCR 加刻度峰值检测实现每张图的物理尺度校准，接着基于预训练 U-Net 做肌腱膜和肌束分割并用几何拟合得到 PA、FL、MT，最后在公开榜上得到 0.32~0.37 量级的分数。比赛告诉我们，在医学量化任务里，标定和几何定义的准确性远比堆网络深度重要，像素级几何在这里比表征学习更关键。即使分割概率图有少量噪声，只要肌腱膜的曲线拟合稳定，MT 的误差仍然可控。反之，如果像素到毫米的转换错了 5%，所有指标都会系统性偏移。我们在总结中保留了纹理角度作为独立验证的做法，这在分割不确定的情况下仍能提供合理的 PA 估计。后续可以尝试把纹理角度和分割角度做融合，或用基准集做 MT 的有监督校正，进一步压低系统偏差。
