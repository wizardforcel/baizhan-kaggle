# Forest Cover Type (Kernels Only)

## 一、比赛简介

### 1.1 比赛背景
Forest Cover Type 是 Kaggle 上的经典机器学习比赛，源自美国林务局的森林覆盖数据。数据来自科罗拉多州 30m×30m 的森林小格子，每个格子被标注为 7 种树种覆盖类型之一：Spruce/Fir、Lodgepole Pine、Ponderosa Pine、Cottonwood/Willow、Aspen、Douglas-fir、Krummholz。特征包括地形、距离水源/道路/火点、光照、野生区域和土壤类型等，共 54 个特征。比赛的“Kernels Only”版本只提供了很小的训练集，用来模拟大家只能用公开 Jupyter 内核的数据情况。

### 1.2 比赛目标
给定训练集 `train.csv` 和测试集 `test.csv`，预测测试集每条样本的 `Cover_Type`。训练集字段有 56 列：`Id`、`Elevation`、`Aspect`、`Slope`、`Horizontal_Distance_To_Hydrology`、`Vertical_Distance_To_Hydrology`、`Horizontal_Distance_To_Roadways`、`Hillshade_9am`、`Hillshade_Noon`、`Hillshade_3pm`、`Horizontal_Distance_To_Fire_Points`、4 个 `Wilderness_Area` 二值特征、40 个 `Soil_Type` 二值特征和目标 `Cover_Type`。测试集只有 55 列，没有标签。材料显示训练集形状为 15120×56，测试集 565892×55，7 类均衡分布，每类 2160 条。

### 1.3 评价标准
排行榜使用分类准确率，预测正确的样本数占总样本的比例，越高越好。训练集类别完全均衡，但测试集真实分布偏斜，因此本地验证时我们也要注意样本权重。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先把训练数据读进来，确认形状和缺失值情况。

```python
import pandas as pd
# 读取训练数据
train_data = pd.read_csv('../input/forest-cover-type-kernels-only/train.csv')
print(train_data.shape)  # (15120, 56)
# 查看目标分布
print(train_data['Cover_Type'].value_counts())
# 输出每类 2160 条，完全均衡
```

打印信息显示没有缺失值，所有特征都是整型。`Soil_Type` 和 `Wilderness_Area` 是 0/1 的二值特征。然后我们观察特征的数值范围和分布，发现 `Elevation`、`Aspect`、`Slope` 量级不同，并用 KDE 看每个覆盖类型下的分布差异。

```python
import seaborn as sb
# 查看不同 Cover_Type 下 Slope 的分布
sb.kdeplot(x=train_data['Slope'], hue=train_data['Cover_Type'], palette='tab10')
# Aspect 在各类型下都呈双峰分布
sb.kdeplot(x=train_data['Aspect'], hue=train_data['Cover_Type'], palette='tab10')
```

EDA 告诉我们特征尺度不一致，需要考虑标准化或树模型，同时分布差异是可分的。

### 2.2 数据清洗
我们其次检查 Id 列和零相关特征。Id 只是索引，应删除。

```python
def drop_features(data, features):
    # 删除指定列，若列存在才删
    for feature in features:
        if feature in data.columns:
            data = data.drop(columns=[feature])
    return data

# 删除 Id 和与标签零相关的土壤类型
train_data = drop_features(train_data, ['Id','Soil_Type7','Soil_Type8','Soil_Type15'])
```

再用 Z 分数检测异常值，连续特征超过 3 个标准差的样本占约 5.8%，我们直接剔除以减少噪声。

```python
import numpy as np
features = ['Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Horizontal_Distance_To_Fire_Points',
            'Elevation','Slope','Aspect']
outlier_idx = set()
for f in features:
    z = (train_data[f] - train_data[f].mean())/train_data[f].std(ddof=0)
    idx = train_data[abs(z) > 3].index
    outlier_idx.update(idx.tolist())
train_data = train_data.drop(outlier_idx)
print(train_data.shape)  # 约 14236, 52
```

数据没有缺失，清洗主要做列删除和异常点剔除。

### 2.3 模型训练与超参数调优
我们接着做特征工程。基于距离的直觉，计算水源的对角距离和水平距离均值。

```python
def diagonal_distance(data, x, y):
    # 计算水平垂直距离的欧氏距离
    return np.sqrt(data[x]**2 + data[y]**2)

def mean_value(data, feature_list):
    # 多个特征的平均值
    s = sum(data[f] for f in feature_list)
    return s / len(feature_list)

Hydrology_distances = ['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology']
train_data['Diagonal_Distance_To_Hydrology'] = diagonal_distance(train_data, *Hydrology_distances)

Horizontal_distances = ['Horizontal_Distance_To_Hydrology',
                        'Horizontal_Distance_To_Roadways',
                        'Horizontal_Distance_To_Fire_Points']
train_data['Mean_Horizontal_Distance'] = mean_value(train_data, Horizontal_distances)
```

第二个思路是大量衍生特征，如材料中 Top 10 方案的 `add_feats`：组合距离差、和、归一化阴影、Aspect 的正弦余弦、二值化等。我们把训练和测试合并后统一生成相同特征，保持分布一致。

```python
def add_feats(df):
    # 距离之和与之差
    df['HF1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['HR1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways']
    # 阴影均值
    df['Shadiness_mean_hillshade'] = (df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']) / 3
    # 山地交互
    df['Slope*Elevation'] = df['Slope'] * df['Elevation']
    return df
```

模型上先用 RandomForest 作为基线，10 折分层交叉验证。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

Y = train_data['Cover_Type']
X = train_data.drop(columns=['Cover_Type'])
clf = RandomForestClassifier(random_state=0)
skf = StratifiedKFold(n_splits=10)
scores = []
for train_idx, valid_idx in skf.split(X, Y):
    clf.fit(X.iloc[train_idx], Y.iloc[train_idx])
    pred = clf.predict(X.iloc[valid_idx])
    scores.append(accuracy_score(Y.iloc[valid_idx], pred))
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
# CV accuracy: 0.788 +/- 0.041
```

为了进一步提升，我们参考 Ensembling 方案，使用 heamy 堆叠多模型：RandomForest、ExtraTrees、LightGBM、LogisticRegression、GaussianNB 和 XGBoost 作为第一层，第二层再次用 LogisticRegression 或 XGBoost 融合概率。

```python
from heamy.pipeline import ModelsPipeline
from heamy.estimator import Classifier
pipeline = ModelsPipeline(rf, et, et1, lgbc, logr, gnb, xgb_first)
stack_ds = pipeline.stack(k=5, seed=1337)
stacker = Classifier(dataset=stack_ds, estimator=LogisticRegression, parameters={'C':5}, use_cache=False)
results = stacker.validate(k=5, scorer=log_loss)
# Mean log_loss 约 0.263
```

### 2.4 模型测试与指标报告
我们最后把清洗和特征工程流程封装为 `transform_features`，对测试集同等处理，用训练好的 RandomForest 预测并生成提交文件。

```python
test_data = pd.read_csv('../input/forest-cover-type-kernels-only/test.csv')
test_data = transform_features(test_data)  # 同样的删列、特征工程
y_test = clf.predict(test_data)
submission = pd.DataFrame({'Id': test_data['Id'], 'Cover_Type': y_test})
submission.to_csv('submission.csv', index=False)
```

本地 10 折交叉验证准确率为 0.788±0.041，堆叠集成后在分层验证上取得更稳定的 log loss 约 0.262，显示模型在不同的森林覆盖类型上均有较好区分能力。

## 三、总结
本文中我们基于 Forest Cover Type Kernels Only 比赛的公开内核材料完成了从数据探索到提交的完整流程。我们首先进行了 EDA，确认训练集 15120 条、7 类均衡、无缺失；其次做了 Id 去除和 Z 分数异常剔除；接着通过对角距离、均值距离、阴影交互等特征工程提升表达，随后用 RandomForest 基线得到 78.8% 的交叉验证准确率，并用 heamy 堆叠 RandomForest、ExtraTrees、LightGBM、XGBoost 等模型降低验证 log loss；最后在测试集上生成了提交文件。
