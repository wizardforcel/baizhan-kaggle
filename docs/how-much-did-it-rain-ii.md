# How Much Did It Rain? II
## 一、比赛简介
### 1.1 比赛背景
这个比赛延续了 “How Much Did It Rain?” 的思路：用雷达在每小时内的多次扫描记录，预测地面雨量计在这一小时里收集到的降雨量。雷达扫描不是连续的，而是每分钟一个快照，因此同一场雨在时间上是一串序列。比赛材料中的 `rain_regression` 页面把任务定义为回归问题，目标是用雷达反射率、差分反射率、比差分相移等物理量估计 `Expected`。
### 1.2 比赛目标
训练数据是 `train.zip`，每行是一次雷达观测。主要列包括：
- `Id`：一个雨量计在一小时内的观测组编号；
- `minutes_past`：观测发生在小时内的第几分钟；
- `radardist_km`：雨量计到雷达的距离；
- `Ref`、`RefComposite`、`RhoHV`、`Zdr`、`Kdp` 及其 5x5 邻域统计量；
- `Expected`：该小时的实际降雨量，单位毫米。
测试集只有前 23 列，我们要预测每个 `Id` 对应的 `Expected`。
### 1.3 评价标准
比赛使用平均绝对误差（Mean Absolute Error, MAE）作为指标。材料中写道 “Model will be considered good if we reach Mean Absolute Error value of 25.5”。MAE 越小越好，它衡量预测值跟真实值平均差多少毫米。

## 二、解题过程
### 2.1 数据探索（EDA）
我们首先读取训练集，并查看目标值的分布。
```python
import pandas as pd
import numpy as np

# 读取压缩的训练文件
train_df = pd.read_csv("../input/how-much-did-it-rain-ii/train.zip")
print(train_df.shape)   # (13765201, 24)

# 目标列的基本统计
print(train_df['Expected'].describe())
```
材料中的统计显示，`Expected` 的均值约 132.2，标准差约 603.1，最大值 33017.7，而 50% 分位数只有约 1.016。说明降雨量分布极端偏斜：大多数小时几乎不下雨，少数小时雨量非常大。这个特点决定了我们不能直接最小化平方误差，否则模型会被极端值带偏，因此比赛选用 MAE 是合理的。
```python
# 查看每个 Id 有多少条雷达记录
group_sizes = train_df.groupby('Id').size()
print(group_sizes.describe())
```
一个 `Id` 代表一小时内的多次扫描，最多 19 条记录。也就是说每条样本本质上是一个长度不超过 19 的时间序列。
### 2.2 数据清洗
我们其次处理缺失值。材料中的做法是：先去掉 `Ref` 为空的所有 `Id`，因为这些组没有可用的雷达反射率。
```python
# 找出 Ref 非空的 Id
good_ids = set(train_df.loc[train_df['Ref'].notna(), 'Id'])
train_df = train_df[train_df['Id'].isin(good_ids)]
print(train_df.shape)   # (9125329, 24)

# 剩下的缺失值统一填 0
train_df.reset_index(drop=True, inplace=True)
train_df.fillna(0.0, inplace=True)
```
这样做的原因是雷达缺测时，填 0 表示“没有观测”，比用均值填补更符合物理含义。接着我们把雨量限制在合理范围内。材料使用 `THRESHOLD = 73`，只保留 `Expected < 73` 的样本。
```python
THRESHOLD = 73
train_df = train_df[train_df['Expected'] < THRESHOLD]
print(train_df.shape)   # (8926102, 24)
```
去掉极端值是为了让模型专注于常见的降雨区间，避免个别几万毫米的记录主导损失。
### 2.3 模型训练与超参数调优
我们接着把每个 `Id` 的记录整理成固定长度的序列。
```python
N_FEATURES = 22
train_groups = train_df.groupby("Id")
train_size = len(train_groups)
MAX_SEQ_LEN = train_groups.size().max()   # 19

# 用 0 填充，得到 (样本数, 19, 22) 的三维数组
X_train = np.zeros((train_size, MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)
y_train = np.zeros(train_size, dtype=np.float32)
for i, (_, group) in enumerate(train_groups):
    X = group.values
    seq_len = X.shape[0]
    # 第 0 列是 Id，最后一列是 Expected，取中间的 22 个特征
    X_train[i, :seq_len, :] = X[:, 1:23]
    y_train[i] = X[0, -1]
```
模型方面，我们使用带 Masking 的双向 LSTM。Masking 告诉网络哪些时间步是补零的，计算时跳过它们，否则补零会被当成真实的“零降雨”观测。
```python
from keras.layers import Input, Dense, LSTM, TimeDistributed, Flatten
from keras.layers import Dropout, Masking, Layer, BatchNormalization, Bidirectional
from keras.models import Model
from keras.optimizers import Nadam

class NonMasking(Layer):
    """Flatten 不支持 mask，所以在它前面把 mask 去掉"""
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)
    def compute_mask(self, input, input_mask=None):
        return None
    def call(self, x, mask=None):
        return x

def get_model_deep(shape=(MAX_SEQ_LEN, N_FEATURES)):
    inp = Input(shape)
    x = Dense(16)(inp)
    x = BatchNormalization()(x)
    x = Masking(mask_value=0.)(x)          # 跳过补零的时间步
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Dense(64))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Dense(1))(x)
    x = BatchNormalization()(x)
    x = NonMasking()(x)
    x = Flatten()(x)                       # 把 19 个时间步拼成一个向量
    x = Dropout(0.5)(x)
    x = Dense(1)(x)                        # 输出一个雨量值
    return Model(inp, x)

model = get_model_deep()
# 直接用 MAE 作为损失，和比赛指标一致
model.compile(optimizer=Nadam(lr=0.001), loss='mae')
```
训练时我们用 20% 数据做验证，并加入早停和学习率衰减。
```python
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=2)

history = model.fit(
    X_train, y_train,
    batch_size=1024,
    epochs=30,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)
```
### 2.4 模型测试与指标报告
最后我们对测试集做同样的分组、补零，然后预测并生成提交文件。
```python
test_df = pd.read_csv("../input/how-much-did-it-rain-ii/test.zip")
test_df = test_df.reset_index(drop=True).fillna(0.0)

test_groups = test_df.groupby("Id")
X_test = np.zeros((len(test_groups), MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)
test_ids = []
for i, (gid, group) in enumerate(test_groups):
    X = group.values
    X_test[i, :X.shape[0], :] = X[:, 1:23]
    test_ids.append(gid)

y_pred = model.predict(X_test, batch_size=1024)
submission = pd.DataFrame({'Id': test_ids, 'Expected': y_pred.reshape(-1)})
submission.to_csv('submission.csv', index=False)
```
材料中还记录了一个更简单的基线：按 `Id` 对特征取均值后，用线性回归、随机森林和 SVR 做回归，在线性回归上 MAE 约 4.99。这个数字看起来比 LSTM 小，但它是在去掉大于均值的样本、只取 1 万条采样的条件下得到的，不能直接跟全量比赛的 MAE 比较。我们的主线仍是序列模型，因为它保留了“第几分钟下雨”这个关键的时间信息。

## 三、总结
本文中我们完成了 How Much Did It Rain? II 的完整流程。我们首先通过 EDA 发现降雨量分布极端偏斜，大多数样本接近零而少数样本极大；其次去掉了 `Ref` 缺失的 `Id`、把缺失值填 0，并过滤掉 73mm 以上的极端雨量；接着把每个 `Id` 的雷达记录补齐成 19 步的序列，用带 Masking 和 BatchNormalization 的双向 LSTM 做回归，损失函数直接用 MAE；最后对测试集分组预测并生成 `Id, Expected` 提交文件。这个比赛让我们体会到，处理时序表格数据时，“按 Id 分组、补齐、加 mask”往往比单纯把每行当独立样本更有效。
