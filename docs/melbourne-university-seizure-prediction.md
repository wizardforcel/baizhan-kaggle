# Melbourne University Seizure Prediction

## 一、比赛简介

### 1.1 比赛背景

Melbourne University Seizure Prediction 是 Kaggle 上一个经典的脑电图二分类比赛。数据来自 Melbourne 大学、AES、MathWorks 和 NIH 的合作项目，包含 3 位癫痫患者的连续颅内脑电图 iEEG 记录。每条样本是一个 .mat 文件，记录 16 个通道的脑电片段，任务是判断该片段属于发作间期还是发作前状态。

比赛数据很“脏”，原始标签里混杂了不确定标签，官方提供了 `train_and_test_data_labels_safe.csv` 标记哪些样本的标签是可信的（safe=1），不同患者的可用样本数差异很大。

### 1.2 比赛目标

输入是一组脑电 .mat 文件，输出是二分类概率。训练集的文件名里编码了病人编号、信号编号和真实标签，测试集没有标签。提交文件要求两列：File, Class，其中 Class 是 0/1 预测标签。

训练数据包含 train_1/2/3，test_1/2/3_new 等目录，标签文件里有 image 路径、safe 标记和 class。

### 1.3 评价标准

比赛使用 ROC AUC 作为主要评价指标。AUC 越高越好，衡量模型在发作前和发作间期之间的区分能力。材料里直接用 sklearn 的 roc_auc_score，且 XGB 的 eval_metric 设为 auc。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先确认数据结构，列出输入目录、读取标签文件并统计每个病人的安全样本数：

```python
import pandas as pd
labels = pd.read_csv('../input/train_and_test_data_labels_safe.csv')
for i in range(1,4):
    safe_labels = labels.loc[(labels.safe == 1) & (labels.image.str.contains('{}_'.format(i)))]
    print(i, ':', len(safe_labels))
```

打印结果显示病人1 有 720 条安全样本，病人2 有 1986 条，病人3 有 2058 条。可见样本量严重不均衡。

接着我们查看 .mat 文件的结构。每个文件包含 dataStruct，里有 data 和 channelIndices。我们用 scipy.io.loadmat 读取，得到 16 列对应 16 个电极通道的数据：

```python
from scipy.io import loadmat
def mat_to_pandas(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])
```

读取后数据形状为 (样本点数, 16)，说明每条记录是多通道时间序列。

### 2.2 数据清洗

标签文件里 safe=0 的样本标签不可靠，我们在训练时只保留 safe=1 的样本。文件名编码规则为 patient_id_id_result.mat，我们从文件名解析 patient_id 和真实标签：

```python
import os
def parse_file(path):
    id_str = os.path.basename(path)[:-4]
    arr = id_str.split('_')
    patient = int(arr[0])
    _id = int(arr[1])
    result = int(arr[2])
    return patient, result
```

特征提取阶段，对每个 .mat 文件计算每通道的均值和绝对值均值，并记录文件大小：

```python
import numpy as np
tables = mat_to_pandas(fl)
for f in tables.columns:
    mean = tables[f].mean()
    absmean = np.mean(np.abs(tables[f]))
```

这样我们把 16 通道的时间序列压缩成 32 个统计特征，外加 file_size 和 patient_id，共 34 列特征。

### 2.3 模型训练与超参数调优

我们采用 XGBoost 二分类模型，特征列是 avg_0~avg_15、absavg_0~absavg_15、file_size、patient_id。参数与 Seizure Boosting 脚本一致：

```python
import xgboost as xgb
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.02,
    "max_depth": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "seed": random_state,
}
```

使用 5 折交叉验证，每折用 20% 数据做验证，early_stopping_rounds=50。训练时打印每轮的 train-auc 和 eval-auc：

```python
gbm = xgb.train(params, dtrain, num_boost_round=1000,
                evals=watchlist, early_stopping_rounds=50, verbose_eval=True)
score = roc_auc_score(y_valid, gbm.predict(dvalid))
```

我们还调用 get_importance 查看特征重要性，帮助确认 avg/absavg 特征确实有区分力。

### 2.4 模型测试与指标报告

在验证集上报告 AUC，Seizure Boosting 脚本的单折结果通常在 0.64~0.72 之间。用全部训练数据重新拟合后对测试集预测，生成提交文件：

```python
def create_submission(score, test, prediction):
    with open('submission.csv', 'w') as f:
        f.write('File,Class\n')
        for id, p in zip(test['Id'], prediction):
            patient = id // 100000
            fid = id % 100000
            f.write(f'{patient}_{fid}.mat,{p}\n')
```

我们同时展示一种“无 Matlab”技巧——按病人分配固定概率也能在公开榜上得到 0.64 AUC，提醒读者该技巧不可用于私有榜。

## 三、总结

本文中我们完成了 Melbourne University Seizure Prediction 的完整流程。我们首先通过 EDA 摸清了数据规模、病人样本不平衡和 16 通道脑电的结构，其次只保留 safe 标签并用文件名解析 patient_id，接着对每个 .mat 文件提取每通道均值、绝对均值和文件大小作为特征，然后用 XGBoost binary logistic 配合 5 折交叉验证训练，最后在验证集上得到 AUC 约 0.65 左右并生成提交文件。该比赛说明在小样本医疗信号上，简单统计特征+强梯度提升仍是可靠起点，但过拟合和病人差异是主要风险。
