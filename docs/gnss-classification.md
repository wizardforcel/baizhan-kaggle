# GNSS Classification

## 一、比赛简介

### 1.1 比赛背景

GNSS（全球导航卫星系统，如 GPS）在智能手机上的定位精度受环境影响很大。这个比赛给出香港三个典型城市场景里手机采集的 GNSS 原始观测数据，要求根据卫星信号特征判断观测是否满足某种质量条件（标签 `Label` 为二值）。这类问题在无导航路测、楼宇遮挡场景中很常见。

原始数据 `GNSS_raw_train.csv` 有约 4.6 万行，按地理位置分成三段：0~10390 行为尖沙咀（Tsim Sha Tsui），10391~38143 行为黄埔（Whampoa），之后是旺角（Mong Kok）。数据列包含经纬度、卫星编号（`Satelite_Code`）、信号码（`Code_L1`）等特征。

### 1.2 比赛目标

给定每行 GNSS 观测的特征，预测 `Label` 的值（0/1）。训练集有 `Label` 列，测试集 `GNSS_raw_test.csv` 只有特征，提交文件 `sample_submission.csv` 里有 `id` 与 `Predict` 两列。

```python
train_data = pd.read_csv("../input/gnss-classification/GNSS_raw_train.csv")
test_data  = pd.read_csv("../input/gnss-classification/GNSS_raw_test.csv")
submission = pd.read_csv("../input/gnss-classification/sample_submission.csv")
```

### 1.3 评价标准

提交列 `Predict` 为解析后的二值整数，竞赛按预测与真实 `Label` 是否一致（准确率）评分。材料内部用 LightGBM 的 AUC 作为训练过程监控指标，最终通过阈值把概率转成类别：

```python
pred = model.predict(X_valid) > THR   # THR = 0.52
score = np.sum(y_valid == pred) / len(pred)
```

---

## 二、解题过程

我们按 EDA、清洗、建模、测试四步走，模型用 LightGBM 二分类基线。

### 2.1 数据探索（EDA）

首先可视化卫星轨迹。把每行按经纬度画成地图散点，用 `Location` 着色可以验证场景划分，用 `Label` 着色可以观察正负样本的空间分布：

```python
import plotly.express as px
def visualize_trafic(DF, key):
    center = {"lat": np.mean(DF['Latitude']), "lon": np.mean(DF['Longitude'])}
    fig = px.scatter_mapbox(DF, lat="Latitude", lon="Longitude",
                            color=key, zoom=13, center=center, height=400, width=400)
    fig.update_layout(mapbox_style='stamen-terrain',
                      title_text="GPS trafic")
    fig.show()

visualize_trafic(df, 'Location')
visualize_trafic(df, 'Label')
```

从地图上看，三个场景的轨迹确实分布在不同街区，且 `Label` 与场景有关联——不同位置的信号遮挡情况不同。

### 2.2 数据清洗

首先丢掉 `Code_L1` 为空的样本（与自身比较会得到 NaN 的删除法）：

```python
train_data = train_data[train_data.Code_L1 == train_data.Code_L1]
```

`Satelite_Code` 是文本形式的卫星编号，用 LabelEncoder 转成整数：

```python
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train_data['Satelite_Code'] = labelencoder.fit_transform(train_data['Satelite_Code'])
test_data['Satelite_Code']  = labelencoder.fit_transform(test_data['Satelite_Code'])
```

因为三个场景的数据是连续拼接的，材料按位置把数据划分成 3 折，保证每个场景都能单独做验证，避免同一场景同时出现在训练与验证造成的数据重合。

### 2.3 模型训练与超参数调优

特征取测试集全部列（`Code_L1`、`Satelite_Code`、经纬度等），标签 `Label`：`X = train_data[columns]; y = train_data['Label']`。

LightGBM 参数来自材料：

```python
THR = 0.52
params = {
    'objective': 'binary',
    'learning_rate': 0.25,
    'boosting_type': 'gbdt',
    'min_data_in_leaf': 600,
    'max_bin': 196,
    'feature_fraction': 0.4,
    'lambda_l1': 36, 'lambda_l2': 80,
    'max_depth': 16,
    'num_leaves': 1000,
    'metric': 'auc',
    'n_jobs': -1,
}
```

训练时每个折用 `eval_set`，加 `early_stopping_rounds=10` 防止过拟合：

```python
model = lgb.LGBMClassifier(**params, n_estimators=8000)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_valid, y_valid)],
          verbose=100, early_stopping_rounds=10)
pred = model.predict(X_valid) > THR
```

`n_estimators=8000` 配合早停，让模型在验证集不再变好时立即停下，比固定轮数更省时。随机森林、深度学习等其他方案在材料里也有尝试，但 LightGBM 是作为“第二跑”最有代表性的基线。

### 2.4 模型测试与指标报告

三个折都训练后，把每个模型对测试集的预测概率平均，再按阈值二值化：

```python
for model in models:
    submission['Predict'] += model.predict(test_data[columns])
submission['Predict'] /= len(models)
submission['Predict'] = submission['Predict'] > THR
submission['Predict'] = submission['Predict'].astype(int)
submission.to_csv('first_sub.csv', index=False)
```

材料中只跑了第 2 折（0.52 阈值）来报告验证准确率，模型平均后再对提交做二值化。因为特征重要性可以导出，我们还能进一步分析哪些卫星特征最影响 `Label`，为下一轮特征筛选做准备。

---

## 三、总结

本文中我们完成了 GNSS Classification 的解题流程。我们首先用 EDA 把训练数据的三个场景画在地图上，确认标签与地理位置相关。其次删除了 `Code_L1` 的空值并编码卫星编号，按场景分成 3 折。接着用调好正则的 LightGBM 做二分类，AUC 监控、early stopping + 0.52 阈值转类别。最后平均多折预测生成提交。

这个比赛给我们的启发是：对连续采集的时序状态数据，折叠要跟随数据来源（场景）而不是随机打乱，否则会高估本地分数；同时二分类问题里阈值的选择对提交成绩影响很大，不要随手用默认的 0.5。