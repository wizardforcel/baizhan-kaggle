# GNSS Classification

## 一、比赛简介

### 1.1 比赛背景

GNSS（Global Navigation Satellite System，全球导航卫星系统）就是手机导航里用的卫星定位系统，GPS、北斗、Galileo 都属于这一类。手机里的接收芯片会同时“听”到十几颗卫星的信号，根据信号的到达时间算出自己到每颗卫星的距离，再解算出经纬度。但在城市里，高楼会把卫星信号挡住、反射，形成所谓的“城市峡谷”和多径效应，导致定位点漂移、精度下降。

这个比赛给出的是一批真实的 GNSS 原始观测数据，采集地点在香港的三个典型城区：尖沙咀（Tsim Sha Tsui）、黄埔（Whampoa）和旺角（Mong Kok）。每条记录对应某一时刻、某一颗卫星的一次观测，包含经纬度、卫星编号、信号码等字段。任务是判断这条观测对应的状态标签 `Label`，本质上是一个二分类问题：根据卫星信号特征，判断这次观测是否满足某个质量条件。

### 1.2 比赛目标

输入是训练集 `GNSS_raw_train.csv` 和测试集 `GNSS_raw_test.csv`。训练集约 4.6 万行，测试集列结构与训练集一致，只是没有 `Label`。提交样例 `sample_submission.csv` 有 `id` 与 `Predict` 两列，我们要为每条测试记录给出 0/1 预测。

```python
train_data = pd.read_csv("../input/gnss-classification/GNSS_raw_train.csv")
test_data  = pd.read_csv("../input/gnss-classification/GNSS_raw_test.csv")
submission = pd.read_csv("../input/gnss-classification/sample_submission.csv")
```

数据是按时段连续采集的，前 10390 行属于尖沙咀，10391 到 38142 行属于黄埔，之后属于旺角。这个行号切分在材料里被用于给样本打场景标签，也用于划分验证折。

### 1.3 评价标准

比赛按提交的 `Predict` 与真实 `Label` 是否一致来评分，也就是分类准确率，越高越好。材料在训练过程中用 AUC 作为监控指标，因为 AUC 对阈值不敏感，能更稳定地反映模型排序能力；最后再选一个阈值把概率转成 0/1：

```python
THR = 0.52
pred = model.predict(X_valid) > THR
score = np.sum(y_valid == pred) / len(pred)   # 验证集准确率
```

阈值 0.52 略高于 0.5，说明正负样本的比例或代价并不完全对称，稍微保守一点能提高准确率。

---

## 二、解题过程

我们按 EDA、数据清洗、模型训练与调参、模型测试与指标报告四步走，主线是 LightGBM 二分类。

### 2.1 数据探索（EDA）

最直观的探索方式是把观测点画到地图上。每一行都有 `Latitude` 与 `Longitude`，把同一场景的点连起来，就能看到手机的移动轨迹：

```python
import plotly.express as px

def visualize_trafic(DF, key):
    # 以所有点的经纬度均值作为地图中心
    center = {"lat": np.mean(DF['Latitude']), "lon": np.mean(DF['Longitude'])}
    fig = px.scatter_mapbox(DF, lat="Latitude", lon="Longitude",
                            color=key, zoom=13, center=center,
                            height=400, width=400)
    fig.update_layout(mapbox_style='stamen-terrain',
                      margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      title_text="GPS trafic")
    fig.show()

# 先用场景标签着色，确认三个区域
visualize_trafic(df, 'Location')
# 再用 Label 着色，观察正负样本在空间上的分布
visualize_trafic(df, 'Label')
```

从地图上能看出三个场景确实位于不同街区，轨迹沿着街道分布；`Label` 的分布与位置有明显关联，某些路段的正样本密集、另一些路段负样本密集。这说明“在哪里”本身就是一个很强的特征，模型应当同时利用经纬度和卫星信号。

接着统计标签比例和数值特征的分布：

```python
print(df['Label'].value_counts(normalize=True))
print(df[['Latitude','Longitude','Code_L1']].describe())
```

结果会告诉我们正负样本是否均衡，以及经纬度的取值范围是否合理。如果标签严重不均衡，单纯看准确率会产生误导，需要同时看 AUC 或正类的召回率。

我们还会逐个字段理解含义：`Satelite_Code` 表示是哪颗卫星；`Code_L1` 与卫星播发的 L1 频点信号有关，信号质量会随遮挡、天气和接收机姿态变化；经纬度是接收机解算出的位置，本身可能已经被多径误差污染。把这些字段放在一起看，可以建立一个直觉：模型要学的不是“卫星长什么样”，而是“在什么位置、看到哪些卫星、信号呈现什么组合时，标签更容易为正”。这也是为什么树模型很适合——它擅长从多个特征的组合中划出决策边界。

从数据规模看，训练集有四万多行，特征只有十几列，属于“样本多、特征少”的表格任务。这种数据上深度网络未必比梯度提升树更好，因为树模型对特征尺度不敏感、训练快、可解释性强，调参成本也低。

另外，把 `Location` 作为颜色画一遍地图，再把 `Label` 作为颜色画一遍，两张图放在一起，可以看到标签与地理位置是有关联的：部分区域的正样本更密集，另一些区域以负样本为主。这说明位置特征（经纬度）可能比卫星信号特征更有判别力，同时也意味着如果验证集和训练集来自同一片区域，模型可能只是记住了“这一片的坐标”，换一个场景就失效。因此我们在划分验证集时坚持按场景切，而不是随机切。

### 2.2 数据清洗

第一件事是处理空值。材料用了一个很巧的写法：拿 `Code_L1` 与自身比较，NaN 不等于 NaN，从而筛掉这些行：

```python
# NaN != NaN 恒为真，这一行等价于保留 Code_L1 非空的行
train_data = train_data[train_data.Code_L1 == train_data.Code_L1]
```

第二件事是把 `Satelite_Code` 从文本转成整数。它是卫星的编号（例如 G01、G02 这类），对模型来说是类别特征，不能按字符串处理：

```python
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
train_data['Satelite_Code'] = labelencoder.fit_transform(train_data['Satelite_Code'])
test_data['Satelite_Code']  = labelencoder.fit_transform(test_data['Satelite_Code'])
```

这里必须用**同一个** LabelEncoder，先 `fit` 训练集再 `transform` 测试集，否则同一颗卫星在两边可能被编成不同整数，模型学到的规律就错位了。

第三件事是划分验证集。因为数据是按场景连续采集的，随机 KFold 会让相邻时刻的样本同时出现在训练和验证里，造成“时间近邻泄露”。材料用行号把数据切成三段，每段对应一个场景，再用 `FOLD` 列做按场景的验证：

```python
train_data['FOLD'] = 0
train_data.iloc[0:10393]['FOLD'] = 0
train_data.iloc[10393:38144]['FOLD'] = 1
train_data.iloc[38144:]['FOLD'] = 2
```

这样每次留出一个完整场景做验证，更接近“模型去一个没见过的城区还能不能用”的真实问题。

### 2.3 模型训练与超参数调优

特征取测试集除 `id` 外的全部列，标签为 `Label`：

```python
columns = test_data.columns.tolist()[1:]
X = train_data[columns]
y = train_data['Label']
```

LightGBM 的参数在材料里给得比较具体：

```python
params = {
    'objective': 'binary',
    'learning_rate': 0.25,
    'boosting_type': 'gbdt',
    'min_data_in_leaf': 600,
    'max_bin': 196,
    'feature_fraction': 0.4,
    'lambda_l1': 36,
    'lambda_l2': 80,
    'max_depth': 16,
    'num_leaves': 1000,
    'metric': 'auc',
    'n_jobs': -1,
}
```

LightGBM 属于梯度提升树家族，思想是训练一长串“弱学习器”（深度很浅的决策树），每棵新树都去纠正前面所有树一起犯的错，最终把它们加权求和成一个强模型。和随机森林这种各树独立的“并行”方法不同，梯度提升是“串行”地去修补残差，所以通常需要控制学习率并配合早停，否则会过度拟合训练集。LightGBM 相比传统 GBDT 还做了直方图分箱（把连续特征切成有限个 bin），训练更快、内存更省，因此是 Kaggle 上表格数据比赛最常用的模型。

这组参数有几个明显特点：学习率 0.25 偏大，配合 `n_estimators=8000` 和早停使用；`min_data_in_leaf=600` 很大，抑制叶子过细；`lambda_l1=36`、`lambda_l2=80` 是较强的正则；`feature_fraction=0.4` 让每棵树只看 40% 的特征。整体是“强正则 + 早停”的思路，适合数据量大但特征噪声也大的场景。

训练时把验证集传进去做早停：

```python
for fold_n in range(3):
    if fold_n != 1:          # 材料里演示时只跑了第 2 折
        continue
    X_train = train_data[train_data['FOLD'] != fold_n]
    X_valid = train_data[train_data['FOLD'] == fold_n]
    y_train = X_train['Label']
    y_valid = X_valid['Label']

    model = lgb.LGBMClassifier(**params, n_estimators=8000)
    model.fit(X_train[columns], y_train,
              eval_set=[(X_train[columns], y_train),
                        (X_valid[columns], y_valid)],
              verbose=100, early_stopping_rounds=10)
    pred = model.predict(X_valid[columns]) > THR
    score = np.sum(y_valid == pred) / len(pred)
    print('valid score: {0:.4f}'.format(score))
```

`early_stopping_rounds=10` 表示验证集 AUC 连续 10 轮不再提升就停止，这样既不用猜迭代次数，也能避免在训练集上过拟合。

### 2.4 模型测试与指标报告

三个折都训练完后，把每个模型对测试集的预测概率相加再平均，最后统一用阈值二值化：

```python
for model in models:
    submission['Predict'] += model.predict(test_data[columns])

submission['Predict'] /= len(models)
submission['Predict'] = submission['Predict'] > THR
submission['Predict'] = submission['Predict'].astype(int)
submission.to_csv('first_sub.csv', index=False)
```

集成多个折的模型可以降低单次划分的偶然性。材料里还导出了特征重要性：

```python
fold_importance = pd.DataFrame()
fold_importance["feature"] = columns
fold_importance["importance"] = model.feature_importances_
fold_importance["fold"] = fold_n + 1
feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
```

特征重要性告诉我们哪些卫星信号字段对 `Label` 贡献最大，后续可以据此删掉无用列、做更轻的模型，或者针对重要特征做更细的统计。

---

## 三、总结

本文中我们完成了 GNSS Classification 的完整流程。我们首先把训练数据的经纬度画在地图上，用场景标签和 `Label` 分别着色，确认三个城区的位置分布以及标签与空间位置的相关性。其次删除了 `Code_L1` 的空值，用同一个 LabelEncoder 编码卫星编号，并按场景把数据切成三折以避免时间近邻泄露。接着用强正则的 LightGBM 做二分类，AUC 监控、早停控制轮数，再用 0.52 的阈值把概率转成 0/1。最后平均多折模型生成提交文件。

这个比赛给我们的启发主要有两点。第一，**验证集的划分方式必须匹配数据的采集方式**：连续时序数据随机切分会严重高估本地分数，按场景/时间段切分才能反映真实泛化能力。第二，**二分类的阈值值得单独调**：模型输出的概率排序好，不代表默认 0.5 就是最优切点，材料里用 0.52 就是在验证集上试出来的。把这些细节做扎实，往往比换一个更复杂的模型更能提升最终成绩。

如果继续优化，我们可以沿着三条路走：一是特征工程，在经纬度之外补上相邻观测的差分速度、卫星仰角等时序特征，让模型能感知“运动状态”；二是把三个折的模型真正全跑完并接一个逻辑回归做第二层融合，尽量利用每个折学到的差异；三是针对特征重要性做更细的错误分析，看看哪些场景是当前模型稳定失分的点。每一步都应当在验证集上记录准确率和 AUC，避免只凭排行榜的一次提交下结论。
