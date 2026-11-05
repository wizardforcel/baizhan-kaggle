# Calibration-Free EMG 手势识别挑战

## 一、比赛简介

### 1.1 比赛背景
Aura Calibration-Free EMG Challenge 是关于可穿戴生物信号的人机交互挑战。表面肌电信号 sEMG 能捕捉前臂动作电位的时空叠加，但传统解码受个体差异、电极位移、皮肤阻抗等多因素影响严重。本次比赛要求在零样本跨用户泛化的条件下识别手势，测试用户完全不在训练分布中。

比赛使用 8 通道前臂环形 sEMG 数据配合 9 自由度惯性传感器 IMU。难点在于电极旋转会带来空间位移，且不同受试者 DC 基线漂移可达上万 ADC 数值，必须做校准自由设计。

### 1.2 比赛目标
给定一段连续的多模态记录，预测每个 6 维手势类别：
`fist`、`open`、`pinch`、`point`、`rest`、`thumb-up`。
训练集为 28 个受试者的连续录制，测试集为 5 个未见受试者的 437 个预分段动作窗口。

训练文件包含列 `t`、`ch1~ch8`、`ax,ay,az,gx,gy,gz,roll,pitch,yaw`、`label`。测试文件按 `action_id` 分段，未提供标签。我们需要输出每个 `action_id` 的手势标签。

### 1.3 评价标准
使用宏平均 F1 分数 Macro-F1，6 类等权重：
```
Macro F1 = 1/C sum_c 2*P_c*R_c/(P_c+R_c)
```
该指标对少数类更敏感，需要同时兼顾 5 个主动手势和 rest。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读入 28 个训练 Excel 文件，统计文件数和时间长度。

```python
import os, glob
import pandas as pd
DATA_DIR = '/kaggle/input/competitions/vr-calibration-free-emg-challenge-pieeg'
train_files = sorted(glob.glob(os.path.join(DATA_DIR, 'train', '*.xlsx')))
# 查看每个文件时长和采样率
for f in train_files[:3]:
    df = pd.read_excel(f)
    dur = df['t'].max() - df['t'].min()
    print(os.path.basename(f), '时长', dur, '行数', len(df))
```

每个训练文件约 180 秒，采样率约 230 行/秒，8 EMG 通道 + IMU + 光学追踪 + 每行标签。测试文件已被分段，共 437 个 action。

```python
TARGET_CLASSES = ['fist','open','pinch','point','rest','thumb-up']
# 统计各手势时长的分布
durations = []
labels = []
for f in train_files:
    df = pd.read_excel(f)
    df['clean_label'] = df['label'].where(df['label'].isin(TARGET_CLASSES), None)
    df['segment_id'] = (df['clean_label']!=df['clean_label'].shift(1)).cumsum()
    for seg, g in df.groupby('segment_id'):
        lbl = g['clean_label'].iloc[0]
        if lbl and len(g)>=30:
            durations.append(g['t'].max()-g['t'].min())
            labels.append(lbl)
print('总片段', len(labels))
# 主动手势约 2.97s，rest 约 1.31s
```

主动手势时长稳定在 2.97 秒，rest 更短且分散，说明指导拍子对主动手势更严格。8 通道能量极坐标图显示不同手势在环形上的激活位置不同，为后续旋转不变特征提供依据。

### 2.2 数据清洗
原始 sEMG 受半电池电位漂移影响，需要 20-100Hz 带通滤波并做线性去趋势。

```python
from scipy.signal import butter, filtfilt, detrend
def design_bandpass(lowcut=20.0, highcut=100.0, fs=250.0, order=4):
    nyq = 0.5*fs
    low, high = lowcut/nyq, highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def preprocess_emg(emg_raw):
    # emg_raw shape (N,8)
    emg_detr = detrend(emg_raw, axis=0)
    b,a = design_bandpass()
    emg_filt = filtfilt(b, a, emg_detr, axis=0)
    return emg_filt
```

我们保留所有原始列，不做删点。极短段 <30 样本仍保留，因标签有效。只做滤波和去趋势，避免删除密文中本可用的标点信息。

### 2.3 模型训练与超参数调优
为解决电极旋转问题，采用 8 点离散傅里叶变换模值作为空间旋转不变特征。

```python
import numpy as np
def spatial_fourier_features(emg_seg):
    # 计算每通道能量
    e = np.sqrt(np.mean(emg_seg**2, axis=0))
    # 8点DFT模值
    F = np.abs(np.fft.fft(e, n=8))
    return F[:4]  # 0-3阶
```

我们将 EMG 滤波后的功率包络与 IMU 动态加速度融合，做 GroupKFold 按 session_id 分 5 折，避免同一受试者在训练和验证同时出现。

```python
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
gkf = GroupKFold(n_splits=5)
for tr_idx, val_idx in gkf.split(X, y, groups=session_ids):
    clf = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31)
    clf.fit(X[tr_idx], y[tr_idx])
```

最终使用异构三集成：LightGBM、XGBoost、CatBoost + 多尺度时序卷积网络。概率经 Nelder-Mead 阈值优化以最大化 Macro-F1。

### 2.4 模型测试与指标报告
在 5 折分组验证上，模型对主动手势召回稳定。由于泄漏问题，部分队伍利用 action 持续时间和固定录制协议达到 1.0 公示分，实际跨用户泛化需依赖旋转不变特征和 IMU 融合。

```python
from sklearn.metrics import f1_score
y_pred = clf.predict(X_val)
macro_f1 = f1_score(y_val, y_pred, average='macro')
print('Fold Macro-F1:', macro_f1)
```

最终提交文件按 sample_submission 格式输出每个 action_id 的 6 类预测。

## 三、总结
本文中我们对 Calibration-Free EMG 手势识别任务进行了从信号预处理到跨用户泛化的完整研究。我们首先做了数据探索，了解了 28 个训练受试者连续录制和 5 个测试受试者分段的结构；其次对 8 通道 sEMG 做 20-100Hz 零相位带通滤波和去趋势清洗；接着构建了旋转不变的空间傅里叶特征并融合 IMU 动力学，使用 GroupKFold 按受试者分组训练 LightGBM/XGBoost/CatBoost 集成；最后在测试集上得到模型预测并评估 Macro-F1。材料中亦指出，部分公开解法利用录制时长和固定顺序的泄漏达到完美分数，真实场景仍需校准自由特征。
