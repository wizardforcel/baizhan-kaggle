# Soil Grain Size From Photos Writeup

## 一、比赛简介

### 1.1 背景
我们参与的是 Kaggle 比赛 Predicting Soil Grain Size Distributions from Images。该比赛要求从土壤表面照片推断其质量粒径分布。训练集包含 24 个不同土壤样本，共 127 张训练照片，测试集包含 10 个土壤样本，共 35 张照片，全部照片总数为 162 张。每个土壤样本只有一个真实的累积粒径曲线，多个照片仅用于提供不同视角和拍摄条件。

数据采集设备存在明显差异，训练照片由 Motorola 和 Samsung 手机拍摄，测试照片则来自 iPhone。这种设备差异导致 JPEG 压缩、色彩处理和镜头特性不同，纹理外观会发生系统性变化。更重要的是，可见表面纹理与实验室筛分测得的质量分布并非一一对应，特别是在细粒端，肉眼难以分辨单个颗粒，模型必须从统计纹理推断级配曲线。

比赛背景论文和描述指出，累积分布在对数直径轴上呈现平滑单调特性，且不同土壤的 D50 跨越多个数量级。我们需要在这种跨设备、跨尺度的条件下实现泛化。

### 1.2 目标
我们的目标是为每个土壤样本预测 11 个粒径支点上的累积质量百分比。

```python
SUPPORT = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
# 11 个粒径支点，对应累计质量百分比，从黏土到砾石
```

预测曲线必须满足物理约束：单调不减、取值在 0 到 100 之间、最后一列固定为 100%。例如在 0.063 mm 处预测值为 60%，表示约 60% 的质量细于该直径。输出格式与 sample_submission.csv 保持一致。

### 1.3 评价标准
官方评价指标为 Earth Mover Distance，其对数加权形式与我们本地代理误差一致。我们使用对数间隔权重近似。

```python
import numpy as np

SUPPORT = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
W = np.diff(np.log10(SUPPORT))  # 对数间隔权重

def emd(true, pred):
    # 计算加权 L1 差异，忽略最后一列常数 100
    return float(np.mean(np.abs(np.asarray(true)[:, :-1] - np.asarray(pred)[:, :-1]) @ W))

def fix(p):
    # 保证单调性和上界，防止预测出现下降
    p = np.maximum.accumulate(np.clip(np.asarray(p, dtype=float), 0, 100), axis=1)
    p[:, -1] = 100.0
    return p
```

分数越低越好。我们的初始基准提交得分为 75.52636，经过特征工程和模型融合后，最终第七版提交得分为 36.03689。

## 二、解题过程

### 2.1 EDA
我们首先读取 Training_labels_updated.csv 并检查曲线形态。所有 24 条训练曲线在对数坐标下基本平滑，但存在平台段，这是实验室筛分插值造成的。

```python
import pandas as pd
import unicodedata, re

def toks(s):
    # 去重音、大小写统一，数字单独处理便于模糊匹配
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return tuple(str(int(t)) if t.isdigit() else t for t in re.findall(r"[a-z]+|\d+", s))

def norm(s):
    return "".join(toks(s))
```

我们计算 D10、D30、D50、D60 等分位点，得到不均匀系数 Cu 和曲率系数 Cc。统计显示 clay% 与 cobble% 呈负相关，存在 15 个不同采集地点。地点分组信息对交叉验证很关键，避免同一地点的土壤同时出现在训练和验证中导致泄漏。

```python
LOGD = np.log10(SUPPORT)
def d_at(c, p):
    # 对数线性插值得到 p% 累计对应的直径
    return float(10 ** np.interp(p, c, LOGD))
```

通过主成分分析发现前 3 个分量可解释超过 90% 的方差，说明曲线低秩。我们绘制中位数曲线和 IQR 带，基线中位数曲线在细粒端过于保守。

### 2.2 数据清洗
文件名存在重音符号、大小写、空格和括号编号差异。我们实现去重音模糊匹配将照片映射到 sample_id。

```python
def contains(seq, sub):
    n = len(sub)
    return n > 0 and any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))

def match_id(key, lookup):
    kt = toks(key)
    hits = [(len(k), v) for k, v in lookup.items() if contains(kt, k)]
    if not hits:
        return None
    m = max(h[0] for h in hits)
    best = {v for l, v in hits if l == m}
    return best.pop() if len(best) == 1 else None
```

相机物理尺度不同。我们读取 ppm_updated.csv，按相机和实际图像尺寸换算有效像素每毫米。

```python
def cam_ppm_table(ppm_df, cams):
    # 找到 ppm 列和相机名称列，返回每台相机的像素每毫米
    # ...
    return out, spec, ppm_col, cam_col

CAM_PPM, SPEC_LONG, ppm_col, cam_col = cam_ppm_table(ppm, sorted(meta.cam.unique()))
# 按满分辨率换算实际拍摄的 ppm
meta["eff_ppm"] = [base * max(w, h) / sl for base, sl, w, h in zip(CAM_PPM.values(), SPEC_LONG.values(), meta.w, meta.h)]
TARGET_PPM = float(chk.eff_ppm.min())
```

我们统一物理尺度，将所有图像重采样到相同的像素/毫米基准，避免设备差异干扰纹理。训练图像普遍比测试图像低分辨率，需要下采样到最粗相机的尺度。

### 2.3 训练调参
我们先实现 DoG 文理特征，参考公开方法，用多尺度高斯差捕捉颗粒边缘。

```python
SIGMAS = np.array([.25, .5, 1., 2., 4., 8.])

def dog_vector(image, ppm):
    # DoG 特征，用于捕捉不同尺度的颗粒纹理
    gray = np.asarray(image.convert("L"))
    factor = 4.5 / ppm
    gray = cv2.resize(gray, (round(gray.shape[1]*factor), round(gray.shape[0]*factor)))
    smooth = [cv2.GaussianBlur(gray, (0,0), float(s*4.5)) for s in SIGMAS]
    # 计算相邻高斯差的标准化标准差
    feats = []
    for a, b in zip(smooth[:-1], smooth[1:]):
        feats.append(np.log(float(np.std(b-a))/(float(gray.std())+1e-6)+1e-6))
    return np.array(feats)  # 5 维
```

随后加入冻结的 DINOv2-small，提取 384 维均值 patch 特征，不对模型微调以避免过拟合。

```python
def dino_vector(image, ppm):
    # DINOv2 特征，冻结模型不微调
    side = 112 * ppm
    crop = image.crop(...)  # 中心裁剪 112mm 区域
    # 预处理并推理
    hidden = session.run(["last_hidden_state"], {...})[0]
    return hidden[0, 1:].mean(axis=0)  # 384 维
```

特征提取后我们按地点做 GroupKFold，避免同一地点的不同土壤泄漏。回归分支使用 PLS 和 Ridge，对数空间拟合 Weibull 曲线。

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

def valid_cdf(prediction):
    result = np.maximum.accumulate(np.clip(np.asarray(prediction, float), 0., 100.), axis=-1)
    result[..., -1] = 100.
    return result
```

第七版方案对两个基线做等权融合：纹理 Weibull 分支 + DINOv2 PLS 分支，并继承 SAM 粗颗粒校正。

```python
# 粗颗粒 SAM 测量校正
# 对 H374 等样本使用 SAM 分割获得可见大颗粒面积，校准质量比例
# 校正后在本地验证提升，但需控制过拟合
```

我们在训练中尝试不同 ppm 对齐、不同 DoG σ、不同融合权重。本地代理误差采用对数面积代理。

```python
def local_proxy_error(truth, prediction):
    # 本地代理误差，用于快速迭代
    return (np.abs(np.asarray(truth)[..., :-1] - np.asarray(prediction)[..., :-1]) * WIDTHS).sum(-1)
```

我们还尝试了单土壤校准和 Weibull 参数优化，但发现对公开集泛化帮助有限。最终保留物理尺度对齐、DoG+ DINOv2、地点交叉验证的组合。

### 2.4 测试指标
我们保留第七版提交作为最终结果。历史记录如下：

- 初始提交：75.52636
- 第七版融合方案：36.03689
- 后续使用 SAM 专科颗粒分割与单土壤校准的尝试：38.74508

SAM 粗颗粒校正在本地验证上有所改善，但公开分数反而上升。我们认为公开集相机分布与训练不同，过度依赖可见面积会高估粗粒质量。DINOv2 特征在跨设备泛化上更稳健，Ridge 对异常值不敏感，PLS 能捕捉曲线低秩结构。

预测流程最终为：读取原始照片 → 物理尺度对齐与 JPEG 处理 → 提取 DoG 与 DINOv2 → 地点交叉验证 → 应用 SAM 校正 → 导出 CSV。所有照片的特征均缓存，避免重复推理。我们在本地使用 EMD 加权评估，确保单调约束始终满足。

## 三、总结
本文中我们完成了一个从图像到土壤粒径分布的完整建模流程。我们首先进行了 EDA，了解了 24 个训练土壤的曲线分布和相机差异，发现在对数坐标下曲线低秩且地点效应明显；其次清洗了文件名去重音模糊匹配问题，并统一了不同手机的物理尺度 ppm，确保跨设备纹理可比；接着构建了 DoG 纹理特征与冻结 DINOv2 语义特征的双分支，结合 Ridge 与 PLS 回归并做地点交叉验证，最后通过 SAM 粗颗粒校正微调但控制过拟合。最终我们在公开榜单上取得了 36.03689 的成绩，较初始 75.52636 有显著提升，而失败实验 38.74508 也提醒我们跨设备泛化需谨慎，避免过度依赖可见面积估计。该方案在 CPU 上可复现，特征缓存和物理尺度对齐是关键工程实践。
