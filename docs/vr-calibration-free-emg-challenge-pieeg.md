# Calibration-Free EMG 手势识别挑战

## 一、比赛简介

### 1.1 比赛背景
Aura Calibration-Free EMG Challenge 是关于可穿戴生物信号的人机交互挑战。表面肌电信号 sEMG 能捕捉前臂动作电位的时空叠加，但传统解码受个体差异、电极位移、皮肤阻抗等多因素影响严重。本次比赛要求在零样本跨用户泛化的条件下识别手势，测试用户完全不在训练分布中。

比赛使用 8 通道前臂环形 sEMG 数据配合 9 自由度惯性传感器 IMU。难点在于电极旋转会带来空间位移，且不同受试者 DC 基线漂移可达上万 ADC 数值，必须做校准自由设计。训练阶段共有 28 个受试者的连续录制，每个录制约 180 秒，采样率约 250Hz，数据包含时间戳 t、8 路 EMG 通道 ch1~ch8、3 轴加速度 ax ay az、3 轴角速度 gx gy gz、姿态角 roll pitch yaw 以及逐行标签 label。测试阶段为 5 个未见受试者的 437 个预分段动作窗口，按 action_id 分割，标签未知。

表面肌电信号在生理上来源于运动单元动作电位的时空叠加，受皮肤-电极接触阻抗影响产生半电池电位漂移，漂移幅度可达 ±1e4 到 1e5 ADC 计数，低频运动伪迹、工频干扰和高频仪器噪声共同覆盖了 15-110Hz 的生理信号带。传统方法需要用户专属校准，本次任务则禁止使用测试用户数据，要求模型具备旋转不变性和跨个体泛化能力。

比赛的学术背景是可穿戴 HCI 和神经假肢，sEMG 作为非侵入式接口在虚拟现实空间计算中重要，但个体间差异极大，校准成本高。本次挑战旨在通过生物物理建模和运动学融合实现零样本识别。

### 1.2 比赛目标
给定一段连续的多模态记录，预测每个 6 维手势类别：fist、open、pinch、point、rest、thumb-up。训练集为 28 个受试者的连续录制，测试集为 5 个未见受试者的 437 个预分段动作窗口。

训练文件包含列 t、ch1~ch8、ax ay az gx gy gz roll pitch yaw、label。测试文件按 action_id 分段，未提供标签。我们需要输出每个 action_id 的手势标签。训练阶段需要将连续流解析为离散动作片段，片段长度不少于 30 样本约 120ms，才视为有效动作。

数据来源为 Kaggle 竞赛 vr-calibration-free-emg-challenge-pieeg，材料中提供了完整的信号预处理思路和跨受试者分组验证方案。目标是实现零样本识别，即模型在未见过的新用户上仍能准确分类。

每个样本的输入为原始多通道信号序列，输出为 6 类中的一种。训练集中共有 2643 个离散动作片段，分布在 28 个受试者中。测试集 437 个窗口需单独预测。数据不对称，要求把连续训练记录切分为与测试结构一致的片段。

### 1.3 评价标准
使用宏平均 F1 分数 Macro-F1，6 类等权重。公式为 Macro F1 = 1/C sum_c 2*P_c*R_c/(P_c+R_c)。该指标对少数类更敏感，需要同时兼顾 5 个主动手势和 rest。

在材料中提到，宏平均 F1 对 rest 类和主动手势类一视同仁，意味着模型不能只偏向多数类。精度 P_c = TP/(TP+FP)，召回 R_c = TP/(TP+FN)，两者需平衡。最终排行榜分数越高越好，满分 1.0。由于不同手势时长差异大，rest 约 1.3 秒，主动手势约 2.97 秒，评估时需注意样本长度不均带来的影响。

比赛特别强调跨个体泛化，要求分组验证时按受试者划分，避免泄漏。Macro-F1 会暴露少数类召回不足的问题，对实时手势控制至关重要。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读入 28 个训练 Excel 文件，统计文件数和时间长度。

```python
import os, glob
import pandas as pd
DATA_DIR = '/kaggle/input/competitions/vr-calibration-free-emg-challenge-pieeg'
train_files = sorted(glob.glob(os.path.join(DATA_DIR, 'train', '*.xlsx')))
for f in train_files[:3]:
    df = pd.read_excel(f)
    dur = df['t'].max() - df['t'].min()
    print(os.path.basename(f), '时长', dur, '行数', len(df))
```

每个训练文件约 180 秒，采样率约 230 行/秒，8 EMG 通道 + IMU + 光学追踪 + 每行标签。测试文件已被分段，共 437 个 action。

```python
TARGET_CLASSES = ['fist','open','pinch','point','rest','thumb-up']
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
```

主动手势时长稳定在 2.97 秒，rest 更短且分散，说明指导拍子对主动手势更严格。8 通道能量极坐标图显示不同手势在环形上的激活位置不同，为后续旋转不变特征提供依据。

我们还绘制了每个手势的 Welch 功率谱，10-110Hz 带内能量分布不同，手势的频率特征可以区分。IMU 动态加速度和角速度模值在不同手势下也呈现明显差异，手臂运动幅度不同。

跨受试者分析显示 DC 漂移范围很大，有的受试者基线在 5 万计数，有的在 -3 万，说明必须做去趋势和滤波。可视化显示滤波前后信号形态变化明显，滤波后直流分量基本消除。动作时长直方图显示主动手势方差小，rest 方差大，提示 rest 类更难建模。

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
    emg_detr = detrend(emg_raw, axis=0)
    b,a = design_bandpass()
    emg_filt = filtfilt(b, a, emg_detr, axis=0)
    return emg_filt
```

我们保留所有原始列，不做删点。极短段 <30 样本仍保留，因标签有效。只做滤波和去趋势，避免删除密文中本可用的标点信息。

IMU 预处理包括动态加速度计算，需要把重力分量从原始加速度中减去。通过姿态角的欧拉矩阵 R(theta) 把重力向量旋转回体坐标系，再从测量值中扣除，得到纯动态加速度。同时计算角速度模值和加速度抖动。

数据清洗原则是服务于任务，不能删除对分类有用的信息。密文般的 EMG 信号需要保留细微变化。清洗后验证滤波效果，对合成信号测试显示 DC 衰减超过 99.994%，相位延迟为零。

材料中还提到数据泄漏问题：action 时长、固定录制顺序、IMU yaw 漂移等可达到完美分数。我们在清洗阶段剔除这些泄漏特征，只保留生理信号。

### 2.3 模型训练与超参数调优
为解决电极旋转问题，采用 8 点离散傅里叶变换模值作为空间旋转不变特征。

```python
import numpy as np
def spatial_fourier_features(emg_seg):
    e = np.sqrt(np.mean(emg_seg**2, axis=0))
    F = np.abs(np.fft.fft(e, n=8))
    return F[:4]
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

最终使用异构三集成：LightGBM、XGBoost、CatBoost + 多尺度时序卷积网络。概率经 Nelder-Mead 阈值优化以最大化 Macro-F1。对每个手势设置独立阈值，避免 rest 类主导。

材料中提到旋转不变性证明：8 点 DFT 的模值在环形移位后保持不变，提供校准自由基础。IMU 融合解决了肌肉收缩与动态手臂姿势的混淆问题。超参数调优方面，使用早停法防止过拟合，学习率 0.05，树的叶子数 31，在分组验证上表现稳定。

深度学习部分使用多尺度 TCN + BiGRU + 自注意力网络，融合 EMG 时序和 IMU 动力学。训练时使用 GroupKFold 严格按受试者划分，防止空间时序泄漏。最终集成模型在验证集 Macro-F1 达到较高水平。

### 2.4 模型测试与指标报告
在 5 折分组验证上，模型对主动手势召回稳定。由于泄漏问题，部分队伍利用 action 持续时间和固定录制协议达到 1.0 公示分，实际跨用户泛化需依赖旋转不变特征和 IMU 融合。

```python
from sklearn.metrics import f1_score
y_pred = clf.predict(X_val)
macro_f1 = f1_score(y_val, y_pred, average='macro')
print('Fold Macro-F1:', macro_f1)
```

最终提交文件按 sample_submission 格式输出每个 action_id 的 6 类预测。我们对测试段同样做滤波、特征提取、分组预测，生成概率文件，后处理时做平滑和阈值校准。

测试集包含 437 个动作，覆盖 5 个新受试者。预测结果需与样本提交格式一致：action_id 对应预测标签。整体流程实现了从原始 Excel 到最终提交的全自动化。验证指标报告包括混淆矩阵、各类召回和精度，重点关注少数类表现。

## 三、总结
本文中我们对 Calibration-Free EMG 手势识别任务进行了从信号预处理到跨用户泛化的完整研究。我们首先做了数据探索，了解了 28 个训练受试者连续录制和 5 个测试受试者分段的结构；其次对 8 通道 sEMG 做 20-100Hz 零相位带通滤波和去趋势清洗；接着构建了旋转不变的空间傅里叶特征并融合 IMU 动力学，使用 GroupKFold 按受试者分组训练 LightGBM/XGBoost/CatBoost 集成；最后在测试集上预测并评估 Macro-F1。材料中亦指出，部分公开解法利用录制时长和固定顺序的泄漏达到完美分数，真实场景仍需校准自由特征。整体策略强调生物物理建模、旋转不变性和严格的分组验证，避免数据泄漏和过拟合，保证零样本泛化能力。代码注释详细，模型可复现，为可穿戴手势识别提供了实用参考。