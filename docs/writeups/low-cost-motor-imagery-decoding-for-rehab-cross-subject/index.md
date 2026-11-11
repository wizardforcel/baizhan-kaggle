# Low Cost Motor Imagery Decoding for Rehab Cross Subject

## 一、比赛简介

### 1.1 比赛背景
Low Cost Motor Imagery Decoding for Rehab Cross-Subject 比赛延续单受试者康复解码任务，目标是在未见过的受试者上泛化。本次任务提供 17 位校准受试者的训练数据（S001–S007, S009–S012, S014, S016–S020），每位受试者包含 50 个 CSV 记录，共 825 个试次（move 与 rest 各约 412/413）。数据采用与单受试者相同的 8 电极布放（Fz、C3、Cz、C4、PO7、Pz、PO8、Oz），采样率 250 Hz。受试者间信号幅度与阻抗存在明显差异，要求模型学习可迁移的空间-时频特征。

### 1.2 比赛目标
给定训练集的多受试者 EEG 试次，模型需对跨受试者测试集的每个试次预测属于 move（运动尝试）还是 rest（休息）。测试集来自未在训练中出现的新受试者，要求模型不依赖受试者特定校准。

### 1.3 评价标准
材料使用跨受试者组群交叉验证估计泛化性能，报告总体准确率与平衡准确率。GroupKFold 将 17 位受试者划分为 5 折，每折 3~4 位未见受试者作为验证，重复评估模型在新个体上的表现。指标数值越高越好。

## 二、解题过程

### 2.1 数据探索（EDA）
使用自定义加载器读取每个 CSV，按事件标记切分试次。

```python
import pandas as pd
import numpy as np

loader = CSV_loader(freq=250, overlap=0.5, epoch_length=1.0)
all_data = [[],[]]
for f in train_files:
    temp = loader.load_MI_CSV(f)
    all_data[0].extend(temp[0])  # rest
    all_data[1].extend(temp[1])  # move
X = np.concatenate(all_data, axis=0)
y = np.concatenate([np.zeros(len(all_data[0]))+i for i in range(2)])
```

数据形状为 (n_trials, n_samples, n_channels)。我们统计各受试者的试次数与类平衡。

```python
# 简单 z-score 归一化
def z_score(X):
    return (X - np.mean(X)) / np.std(X)
for i in range(X.shape[0]):
    X[i] = z_score(X[i])
```

材料显示整体类比约为 1:1，跨受试者差异主要体现在幅度与频谱。

### 2.2 数据清洗
预处理包括 4~25 Hz 带通滤波以保留 μ 与 β 节律，以及 50 Hz 工频陷波。

```python
from scipy import signal
b,a = signal.butter(4, [4,25], btype='bandpass', fs=250)
X = signal.filtfilt(b,a,X, axis=1)
b,a = signal.iirnotch(50/(250/2), 30.0)
X = signal.filtfilt(b,a,X, axis=1)
```

对于测试数据，采用相同滤波与 z-score，然后送入模型。

### 2.3 模型训练与超参数调优
示例笔记本给出 CSP + SVM 基线。

```python
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

class CSP_SVM:
    def __init__(self):
        self.pipeline = Pipeline([
            ('csp', CSP()),
            ('svm', SVC(class_weight='balanced'))
        ])
```

训练时对 X 做轴转置以符合 CSP 输入。

```python
classifier = CSP_SVM()
scores = cross_val_score(classifier.pipeline, X, y, cv=5)
```

材料还提供更强的端到端深度学习方案，使用受试者分组的 GroupKFold：

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
```

深度模型直接在原始时序上学习频带分解与空间滤波，禁止使用手工 CSP 特征，输入仅做 1~45 Hz 带通与 50 Hz 陷波，并进行运行时标准化。

### 2.4 模型测试与指标报告
CSP-SVM 在训练集上做 5 折交叉验证，报告平均准确率。

```python
print(f"5 Fold Accuracy: {classifier.evaluate(X, y)*100:.3f}%")
```

测试阶段逐文件加载测试 CSV，用 bandpass 滤波、z-score 标准化后做单试次预测，最后对同一试次内的多 epoch 投票决定 move/rest。

```python
predictions = []
for epoch in data:
    epoch = signal.filtfilt(b,a,epoch,axis=0)
    predictions.append(classifier.predict(epoch.T)[0])
# 投票
target = 'move' if predictions.count(1) > predictions.count(0) else 'rest'
```

GroupKFold 的深度实验在 17 受试者上输出 Out-of-Fold Accuracy 与 Balanced Accuracy，作为跨受试者泛化的最终评估。

## 三、总结
本文中我们完成了跨受试者低成本运动想象解码的建模流程。我们首先对 17 位校准受试者的 825 个试次进行带通滤波与 z-score 归一化，确认类平衡且存在跨个体幅度差异。接着以 CSP-SVM 为基线，使用 5 折交叉验证评估，并在材料中参考了基于 GroupKFold 的端到端深度学习架构，该架构在不使用手工特征的情况下学习可迁移的空间-时频表征。最后对未见受试者的测试试次做滤波、归一化与预测，并按 epoch 投票生成提交文件。该任务的关键在于跨受试者泛化与抗个体差异的预处理。
