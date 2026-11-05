# Low Cost Motor Imagery Decoding for Rehab

## 一、比赛简介

### 1.1 比赛背景
Low Cost Motor Imagery Decoding for Rehab（Single Subject）关注用低成本、便携式脑电设备解码中风患者的运动意图，为机器人手部外骨骼康复训练提供触发信号。数据由 g.tec Unicorn Hybrid 系统采集，只有 8 个电极（Fz、C3、Cz、C4、PO7、Pz、PO8、Oz），采样率约 250 Hz。每位受试者包含 3 个训练文件和 1 个测试文件，时间戳依次递增。实验范式要求受试者在“休息”和“运动尝试”之间切换。

### 1.2 比赛目标
输入是多通道 EEG 时间序列和事件标记，目标是对测试段中每个 cue 试次判断属于 Rest 还是 Motor Attempt。提交文件包含 ID 与 TARGET，单受试者测试集共 680 个试次，其中 408 个用于公开榜、272 个用于私有榜。

### 1.3 评价标准
比赛以分类准确率（accuracy）为指标。材料中的基线对每位受试者分别做交叉验证，最后汇总平均准确率。由于两类样本接近 1:1，普通准确率即可反映解码性能。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读取每位受试者的 3 个训练文件和 1 个测试文件。

```python
from pathlib import Path
import pandas as pd

competition_dir = Path('/kaggle/input/competitions/low-cost-motor-imagery-decoding-for-rehab/')
def read_subject_files(subject):
    train = []
    for i in range(1, 4):
        train.append(pd.read_csv(competition_dir / f"Training set/S{subject:03d}-{i:03d}_eeg.csv",
                                 dtype={'Marker_val': object}))
    test = pd.read_csv(competition_dir / f"Single subject test set/S{subject:03d}_test.csv")
    return train, test
```

材料说明每个文件时间步长为 0.003996 或 0.003997 秒（1/250 s），准备阶段约 3 秒、cue 阶段约 5 秒。

```python
signals = ['Fz','C3','Cz','C4','PO7','Pz','PO8','Oz']
data = pd.concat(train + [test], keys=range(len(train)+1), names=['file','ignore'])
```

我们把四个文件拼起来，保留 file 列以区分试次来源。

### 2.2 数据清洗
我们从 Marker_val 中解析 trial_start、cue_start、cue_stop 和 target。

```python
def extract_markers_from_df(df):
    marker = df[['file','Marker_val']].dropna()
    rows = []
    idx = 0
    while idx < len(marker):
        if marker.iloc[idx] == 'exp_stop':
            break
        if marker.iloc[idx] in ['trial_start','trail_start']:
            rows.append((marker.iloc[idx], marker.index[idx], marker.index[idx+1],
                         marker.index[idx+2], marker.iloc[idx+1]))
        idx += 1
    return pd.DataFrame(rows, columns=['file','trial_start','cue_start','cue_stop','target'])
```

预处理包含三步：共平均参考、50 Hz 陷波、去趋势。

```python
from scipy.signal import iirnotch, filtfilt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 共平均参考，抵消各通道共同噪声
data -= data.mean(axis=1, keepdims=True)
# 50 Hz 工频陷波
b, a = iirnotch(50/(250/2), 30.0)
signal = filtfilt(b, a, signal, axis=0)
# 多项式拟合去除缓慢漂移
drift = make_pipeline(PolynomialFeatures(include_bias=False), LinearRegression())
drift.fit(np.arange(len(signal)).reshape(-1,1), signal)
signal -= drift.predict(np.arange(len(signal)).reshape(-1,1))
```

每个试次截取 cue_start 到 cue_stop，并把长度对齐到最短试次。

```python
X = [s[:n_timesteps] for s in X]
X = np.stack(X).reshape(len(X), -1)
y = np.array(y)
```

### 2.3 模型训练与超参数调优
基线使用标准化 + RBF 核 SVC。

```python
model = make_pipeline(
    StandardScaler(),
    SVC(C=1, kernel='rbf')
)
```

对每位受试者分别训练和验证，避免跨受试者信息泄漏。

```python
for subject in range(1, 21):
    if subject in [8, 13, 15]:
        continue
    # 每位受试者独立交叉验证
    score = cross_val_score(model, X[mask_cv], y[mask_cv]).mean()
```

材料同时给出更强的双轨方案：Track 1 使用 Filter Bank Riemannian Tangent Space（FBRTS）把协方差矩阵映射到切空间，再用 L2 正则线性 SVM；Track 2 使用 EEGNet-4,2 端到端卷积网络。两者均采用受试者内 5 折分层交叉验证。

### 2.4 模型测试与指标报告
用训练文件拟合后预测该受试者的 40 个测试试次，汇总所有受试者得到 680 行提交。

```python
preds = []
for subject in subjects:
    model.fit(X[mask_tr], y[mask_tr])
    preds.append(model.predict(X[mask_te]))
submission = pd.DataFrame({'TARGET': np.hstack(preds)}).rename_axis(index='ID')
submission.to_csv('submission.csv')
```

材料中的基线在每位受试者上打印交叉验证准确率并取平均，最终输出 680 行提交文件。

## 三、总结
本文中我们完成了单受试者低成本运动想象解码的基线流程。我们首先读取 8 通道 EEG 与事件标记，确认每位受试者包含 3 个训练文件和 1 个测试文件。其次做共平均参考、50 Hz 陷波和多项式去趋势，并把 cue 段展平成特征向量。接着以标准化加 RBF-SVM 为主模型，逐受试者进行交叉验证，同时参考了 Riemannian 切空间与 EEGNet 两条更强的技术路线。最后对 17 位受试者的测试试次预测并生成 680 行提交文件。该任务的关键在于逐受试者建模和稳定的伪迹处理。
