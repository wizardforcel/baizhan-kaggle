# Decoding the Human Brain：从脑磁图里读出「你看到了什么」

## 一、比赛简介

### 1.1 比赛背景

这个比赛叫 DecMeg2014（Decoding the Human Brain），是 2014 年 Kaggle 上的一场脑科学比赛。它研究的问题是：**能不能只看大脑活动的信号，就判断出一个人刚才看到的是人脸还是乱七八糟的图案？**

数据是用**脑磁图（MEG）**记录下来的。MEG 是一种无创的脑成像技术，受试者头上戴一顶布满传感器的「头盔」，传感器能测到神经元活动产生的极微弱磁场。比赛时，受试者坐在屏幕前，屏幕上每隔一会儿闪一张图：一半是**人脸**，一半是把人脸打乱后的**噪声图案**。每闪一次叫一个「试次」（trial），记录下这一次刺激前后 1 秒左右的脑磁信号。

这件事的难点在于信号非常微弱、噪声很大，而且每个人的大脑响应都不一样。同一个模式在甲身上明显，在乙身上可能就看不出来。

### 1.2 比赛目标

数据以 MATLAB 的 `.mat` 文件给出，每个受试者一个文件：

- 训练集是 `train_subject01.mat` 到 `train_subject16.mat`，共 16 个人；
- 测试集是 `test_subject17.mat` 到 `test_subject23.mat`，共 7 个人。

每个文件里有几个变量：

- `X`：形状是 `(试次数, 306, 375)`。306 是传感器通道数，375 是时间点（采样率 250 Hz，约 1.5 秒）；
- `y`：每个试次的标签，1 表示看到了人脸，0 表示看到了噪声图（测试集没有这一列）；
- `sfreq`：采样率，250 Hz；
- `tmin`：这段信号的起始时间，通常是 -0.5 秒（0 秒是刺激出现的时刻）；
- 测试文件里还有 `Id`，是每个试次的编号，提交时要用。

我们要输出每个测试试次属于「人脸」的概率或标签，提交文件两列：`Id,Prediction`。

整个训练集一共 **9414 个试次**，测试集一共 **4058 个试次**。

### 1.3 评价标准

评价指标是 **AUC（ROC 曲线下面积）**。用大白话说：把测试集里所有「人脸」试次和所有「噪声」试次两两配对，看模型给「人脸」打的分比「噪声」高的比例有多大。全部排对是 1.0，瞎排是 0.5。所以 AUC 衡量的是**排序能力**，而不是「猜对多少个」。

这一点很关键：因为 AUC 只看分数的高低顺序，我们提交的应该是**连续的概率或分数**，而不是硬邦邦的 0/1。把概率压成 0/1 会丢掉大量信息，分数会明显变差。

---

## 二、解题过程

我们走的是一条很「经典机器学习」的路线：把脑电信号当成时间序列，先滤波、截时间窗、拉平成一个长向量，再扔给支持向量机。下面分四步讲。

### 2.1 数据探索（EDA）

我们首先把 `.mat` 文件解压、读进来，并把 16 个受试者的数据拼在一起：

```python
import os
import numpy as np
from scipy.io import loadmat

# 逐个读取训练文件，把 X 和 y 竖着拼起来
train_data, train_labl = None, None
for i in range(1, 17):
    path = "train/data/train_subject%02d.mat" % i
    mat = loadmat(path, squeeze_me=True)
    X, y = mat["X"], mat["y"]
    if train_data is None:
        train_data, train_labl = X, y
    else:
        train_data = np.concatenate((train_data, X), axis=0)
        train_labl = np.concatenate((train_labl, y), axis=0)

print(train_data.shape)   # (9414, 306, 375)
print(train_labl.shape)   # (9414,)
```

读出来确认了三件事：训练集 9414 个试次，每个试次 306 个通道、375 个时间点；`y` 里只有 0 和 1 两个值，没有别的杂值。

接着我们检查有没有缺失值和标签不平衡：

```python
# 有没有 NaN？有没有意料之外的标签？
print(np.isnan(train_data).any())
print(np.unique(train_labl))

# 每个类别各有多少个试次
import collections
print(collections.Counter(train_labl.tolist()))
```

结果是：没有 NaN；标签只有 0 和 1；两类各有 **4707** 个，正好一半一半。这说明比赛把正负样本配平了，不需要再做重采样，也不会出现「全猜多数类」就能拿高分的情况。

我们还顺手数了每个人的试次数：

```python
# 每个受试者大约 578~594 个试次，16 个人加起来 9414
for i in range(1, 17):
    print(i, loadmat("train/data/train_subject%02d.mat" % i)["X"].shape[0])
```

每个人五百多个样本，说多不多、说少不少。这意味着如果按受试者切验证集，留出 4 个人就只有两千多条样本，估出来的分数会有一定波动，读结果时要留出误差范围。

然后我们画了第一个试次第一个通道的原始波形，以及它的功率谱：

```python
import matplotlib.pyplot as plt
from scipy import signal

plt.figure(figsize=(16, 4))
plt.plot(train_data[0][0])          # 第 1 个试次、第 1 个通道
plt.xlabel("Time")
plt.ylabel("Value")

# 用 Welch 方法估计功率谱密度，采样率 250Hz
f, psd = signal.welch(train_data[0][0], fs=250.0)
plt.figure(figsize=(16, 4))
plt.semilogy(f, psd)                # 纵轴取对数，低频能量才看得清
plt.xlabel("frequency [Hz]")
plt.ylabel("PSD")
plt.show()
```

波形图上有明显的低频漂移（整体缓慢上下起伏），频谱图则在 50 Hz 附近有一个尖峰——那是**工频干扰**，也就是墙上插座 220V/50Hz 交流电带来的噪声。这两张图直接决定了我们下一步要做什么滤波。

### 2.2 数据清洗

**第一步，去掉不用的通道。** 原始数据有 306 个通道，但其中 162 个是辅助通道（比如眼电、参考通道等），真正记录脑磁的只有剩下的 144 个。我们只保留这 144 路：

```python
# nx=306，去掉前 162 个辅助通道，只留 144 路脑磁信号
x_keep = np.empty((n_samples, 306 - 162, 375))
for i in range(n_samples):
    for j in range(144):
        x_keep[i][j] = data[i][162 + j]
```

通道从 306 降到 144，特征量直接少了一半多，后面训练会快很多，也少了很多纯噪声。

**第二步，滤波。** 针对 EDA 里看到的两类噪声，我们设计了两级滤波：

```python
from scipy.signal import butter, lfilter, iirnotch

def apply_filters(sig):
    # 低通滤波：保留 100Hz 以下
    # 250Hz 采样时奈奎斯特频率是 125Hz，0.8 对应的截止频率就是 100Hz
    b_low, a_low = butter(5, 0.8, btype="low")

    # 陷波滤波：专门掐掉 50Hz 工频
    # 第二个参数 30 是品质因数，越大陷波越窄
    b_not, a_not = iirnotch(50.0, 30.0, 250.0)

    filtered = lfilter(b_low, a_low, sig)   # 先去高频噪声
    filtered = lfilter(b_not, a_not, filtered)  # 再掐工频
    return filtered
```

低通的理由是：脑磁信号里有用的成分大多在几十赫兹以下，更高频的基本是肌电和环境噪声。陷波的理由更直接——50 Hz 的尖峰不是大脑产生的，留着只会干扰模型。

**第三步，截时间窗。** 刺激出现在 0 秒，但真正有用的响应通常在刺激后几百毫秒内。我们只保留 **0～0.5 秒**这一段：

```python
def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
    """把 (试次, 通道, 时间) 的三维数据变成二维特征矩阵"""
    beginning = int(round((tmin - tmin_original) * sfreq))  # 0.5s -> 第 125 点
    end = int(round((tmax - tmin_original) * sfreq))        # 1.0s -> 第 250 点
    XX = XX[:, :, beginning:end]        # 只留 125 个时间点

    # 把 144 个通道 × 125 个时间点拉平成一个 18000 维的长向量
    XX = XX.reshape(XX.shape[0], -1)

    # 每个特征单独做 z-score：减均值、除标准差
    XX = XX - XX.mean(0)
    XX = XX / (XX.std(0) + 1e-8)
    return np.nan_to_num(XX)
```

144 × 125 = **18000**，这就是每个试次的最终特征数。z-score 是必要的：不同通道的幅值量级可能差很多，不归一化的话，数值大的通道会主导 SVM 的距离计算。

### 2.3 模型训练与超参数调优

我们用的是**支持向量机（SVM）**。选它的原因很实际：特征有 18000 维、样本只有 9414 条，属于典型的「宽而浅」问题，SVM 在这类高维小样本数据上表现稳定，而且不需要漫长的迭代。

```python
from sklearn.svm import SVC

# 默认使用 RBF 核，C=1.0，gamma='scale'
clf = SVC()
print("Training.")
clf.fit(X_train, y_train)

print("Predicting.")
y_pred = clf.predict(X_test)
```

RBF 核的直觉是：它不要求两类数据在原始空间里被一条直线分开，而是把每个样本映射到高维，再找一个「弯曲」的分界面。对脑磁信号这种模式复杂、方向不固定的数据，这比线性核合适。

**关于调参。** 材料里的做法是直接用默认参数的 `SVC()`，没有做网格搜索。我们认同这个取舍：18000 维、9414 个样本，训练一次已经相当耗时，而 SVM 的两个关键参数 `C`（惩罚强度）和 `gamma`（RBF 的「影响半径」）在这类数据上通常不会带来数量级的差别，收益远小于「把时间窗选对、把噪声滤掉」这类特征层面的工作。所以我们的精力主要花在了 2.2 的清洗上。

**一个必须说明的坑：** 材料里的评估是在**训练集自身**上算的：

```python
from sklearn.metrics import classification_report
print(classification_report(y_train, clf.predict(X_train)))
```

这样算出来的分数只能说明「模型有没有学会」，不能说明「模型能不能泛化」。因为它见过这些样本。真正可信的做法是按**受试者**切分：比如用 1～12 号训练、13～16 号验证，这样才能模拟「遇到一个没见过的人」。如果随机按试次切分，同一个人的数据会同时出现在训练和验证里，分数会系统性偏高。

### 2.4 模型测试与指标报告

在训练集上，模型的分类报告是：

```
              precision    recall  f1-score   support

           0       0.95      0.97      0.96      4707
           1       0.97      0.95      0.96      4707

    accuracy                           0.96      9414
   macro avg       0.96      0.96      0.96      9414
weighted avg       0.96      0.96      0.96      9414
```

两类都是 0.96 的 F1，说明模型没有偏袒任何一类。但请记住，这是**样本内**的 0.96，真实泛化分数一定会低一些——材料里没有给出验证集分数，我们不应该把这个数字当成排行榜成绩。

还有一件更重要的事：**比赛的指标是 AUC，不是准确率**。上面这套流程输出的是硬标签 0/1，如果直接这样提交，等于把模型辛苦算出的「置信度」扔掉了。正确做法是输出连续的决策分数：

```python
# decision_function 给出样本到分类面的带符号距离，数值越大越像正类
scores = clf.decision_function(X_test)

# 按比赛格式写提交文件：Id + 连续分数
with open("Submission.csv", "w") as f:
    f.write("Id,Prediction\n")
    for i in range(len(scores)):
        f.write("%s,%f\n" % (ids_test[i], scores[i]))
```

`decision_function` 和 `predict_proba` 的区别值得说一句：前者是「离分界面多远」，可以是任意实数；后者是「概率」，在 0～1 之间。AUC 只关心顺序，两者都能用；但 SVM 默认不输出概率（需要额外开启 `probability=True`，训练会慢很多），所以直接用 `decision_function` 更划算。

如果要在本地估一个可信的 AUC，我们会这么做：

```python
from sklearn.metrics import roc_auc_score, roc_curve

# 用留出的受试者做验证，而不是随机切试次
mask_val = subject_id >= 13          # 13~16 号当验证集
clf.fit(X_train[~mask_val], y_train[~mask_val])
val_scores = clf.decision_function(X_train[mask_val])

print("AUC =", roc_auc_score(y_train[mask_val], val_scores))
fpr, tpr, _ = roc_curve(y_train[mask_val], val_scores)   # 可以画 ROC 曲线看
```

---

## 三、总结

本文中我们完成了 Decoding the Human Brain 比赛的一次基线尝试。我们首先读入 16 个受试者的 `.mat` 文件，确认了数据形状是 9414 × 306 × 375、两类标签各 4707 个、没有缺失值；又通过波形和功率谱发现了低频漂移与 50 Hz 工频干扰。其次我们做了三步清洗：丢掉 162 个辅助通道只留 144 路脑磁、用 100 Hz 低通加 50 Hz 陷波滤波、截取刺激后 0～0.5 秒并拉平成 18000 维特征，再做逐特征 z-score。接着我们用默认参数的 RBF 核 SVM 训练分类器。最后我们在训练集上得到 0.96 的准确率和 0.96 的宏平均 F1，同时明确指出这是样本内分数，并给出按受试者切分验证、用 `decision_function` 输出连续分数以适配 AUC 的正确做法。

这个比赛让我们体会到两件事。第一，**脑科学数据的成败大半在预处理**：通道选择、滤波、时间窗，每一步都在决定模型能看到什么，模型本身反而是最标准化的一环。第二，**评估方式必须和比赛指标对齐**：用准确率衡量一个 AUC 比赛，用样本内分数估计泛化能力，都是看起来很合理、实际上会骗自己的做法。
