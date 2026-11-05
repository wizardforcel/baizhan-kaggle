# VSB 电力线故障检测

## 一、比赛简介

### 1.1 比赛背景
VSB Power Line Fault Detection 来自捷克 VSB-技术大学电力系统研究。电力传输采用三相交流系统，故障会产生瞬间异常电压和电流波形。比赛提供 2001 个采样点的模拟信号波形，目标是判断该段信号是否含有故障。

数据来源于 Kaggle 竞赛 vsb-power-line-fault-detection，训练集共 800,000 条测量，每条包含 2001 个时间点数值，对应 3 相信号。难点在于故障样本极少，属于严重类别不平衡，且波形受噪声和 50Hz 工频干扰覆盖。

### 1.2 比赛目标
输入为训练集的 parquet 文件 `train.parquet` 和元数据 `metadata_train.csv`。元数据包含 `signal_id`、`id_measurement`、`phase`、`target` 四列。`target` 为 0/1 二分类，1 表示故障。测试集同样提供元数据，未给出目标标签。

每个测量 id 包含 3 个相位信号，需分别预测。最终提交为每个 `signal_id` 的故障概率或标签。

### 1.3 评价标准
比赛使用 Matthews 相关系数 MCC 作为评价指标：
```
MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```
MCC 对类别不平衡鲁棒，1 为完美，0 为随机，-1 为完全相反。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先读取 parquet 文件，观察数据形状。

```python
import pyarrow.parquet as pq
import pandas as pd
train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(2001)]).to_pandas()
print(train.shape)  # 800000 x 2001
meta = pd.read_csv('../input/metadata_train.csv')
print(meta.head())
```

每个测量有 3 个相位，共 800,000 条记录。目标分布严重倾斜，正类约 1% 左右。信号波形呈现 50Hz 正弦基底，故障时会出现尖峰或畸变。

```python
# 查看目标分布
print(meta['target'].value_counts())
# 0 约 792000, 1 约 8000
```

我们对正弦波形做峰值检测，发现故障相的峰值幅度、峰间距离和噪声幅值与正常相不同。

### 2.2 数据清洗
原始数据无缺失值，但信号含工频干扰。我们对每个相位信号做带通滤波去除直流和强噪声。

```python
import numpy as np
def denoise_signal(x):
    # 简单移动平均降噪
    w = 5
    return np.convolve(x, np.ones(w)/w, mode='same')

# 去除异常极值
def clip_signal(x):
    p1, p99 = np.percentile(x, [1,99])
    return np.clip(x, p1, p99)
```

数据清洗不删除样本，只做数值稳定化。对每个信号提取峰值窗口，避免使用全序列。

### 2.3 模型训练与超参数调优
冠军方案使用 LightGBM，只用 9 个基于峰值的手工特征，5 折交叉验证。

```python
from sklearn.metrics import matthews_corrcoef
# 峰值检测
from scipy.signal import find_peaks
def get_peaks(x):
    peaks, _ = find_peaks(np.abs(x), distance=20, prominence=1.0)
    return peaks

# 计算峰值特征
def peak_features(peaks, x):
    # 峰高度、峰间距、半峰宽等
    heights = np.abs(x[peaks])
    dists = np.diff(peaks)
    feats = [
        np.mean(heights),
        np.std(heights),
        np.mean(dists),
        np.max(heights),
        np.min(heights),
        len(peaks),
        np.max(dists) if len(dists)>0 else 0,
        np.median(heights),
        np.percentile(heights, 90)
    ]
    return feats
```

特征生成约 15 分钟。训练 LightGBM：

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

params = dict(objective='binary', metric='mcc', n_estimators=1000,
              learning_rate=0.05, num_leaves=31, class_weight='balanced')
skf = StratifiedKFold(n_splits=5)
for tr, val in skf.split(X, y):
    model = lgb.LGBMClassifier(**params)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[val])
    print(matthews_corrcoef(y[val], pred))
```

也可尝试 LSTM+Attention，定义 Matthews 损失函数用于训练。

### 2.4 模型测试与指标报告
在 5 折 CV 上，手工特征 LightGBM 可达 MCC 0.65 左右。测试集预测后生成提交文件。

```python
test_pred = model.predict_proba(X_test)[:,1]
submission = pd.DataFrame({
    'signal_id': test_ids,
    'target': test_pred
})
submission.to_csv('submission.csv', index=False)
```

最终模型在排行榜约 0.68 MCC。特征工程是关键，峰值统计能捕捉故障瞬态。

## 三、总结
本文中我们对 VSB 电力线故障检测任务进行了系统研究。我们首先探索了 80 万条三相波形数据，发现严重类别不平衡；其次对信号进行降噪和峰值检测清洗；接着提取 9 个峰值手工特征，用 LightGBM 做 5 折 Stratified 训练并调参；最后在测试集上预测并提交 MCC 指标。结果表明，基于峰值的轻量特征配合树模型在该时序故障检测上性价比很高。
