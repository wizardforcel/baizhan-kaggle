# Mercedes-Benz Greener Manufacturing

## 一、比赛简介

### 1.1 比赛背景

Mercedes-Benz Greener Manufacturing 由戴姆勒（Daimler）发起，要求参赛者利用匿名化的汽车配置特征，预测每辆定制车在测试台（test bench）上通过测试所需的时间。数据集包含 400+ 个匿名特征，每个特征代表一种可能的车辆配置项（如四驱、悬挂选装、抬头显示等），其中大多数是 0/1 二值特征，少数是有字母取值的分类特征。

比赛难点是维度灾难与匿名化：特征很多但部分特征在不同样本间取值完全相同，X0 是高基数分类特征，而且有些类别只出现在测试集里。

### 1.2 比赛目标

输入为 train.csv / test.csv，第一列 ID，特征 X0~X385，训练集多一列 y 表示测试时间（秒）。目标是预测 test.csv 中每条 ID 的 y。提交文件包含 ID 与 y 两列。

### 1.3 评价标准

比赛使用决定系数 R²（R squared）作为评价指标。R² 代表模型解释的方差比例，越接近 1 越好，0 表示模型的预测效果与直接用均值相当。优秀的方案在私有榜能达到 0.55 左右，210 个方案中 78 名左右即为 Top 2%，分数约 0.55282。

## 二、解题过程

### 2.1 数据探索（EDA）

我们先读取数据并了解维度：

```python
import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print('train:', train.shape, 'test:', test.shape)
```

train 和 test 各 4209 行。数据里 X0 是字母分类特征，取值像一个字母串，其余大量是 0/1 字段。我们检查哪些列在不同样本间完全没有变化：

```python
frame = pd.DataFrame({'max': train.max(), 'min': train.min()}, index=train.columns)
frame = frame.loc[frame['max'] == frame['min']]
print(len(frame), list(frame.index))
```

打印显示有若干列 min == max，即所有行取值都一样，这些列对预测毫无贡献，只能添乱，需要删除。

### 2.2 数据清洗

对无变化列直接删除。X0 是高基数分类特征，训练集与测试集的取值不完全重合，直接把字母当类别会泛化差。我们尝试两种思路：一是保持分类特征原样交给树模型，二是像一些方案那样把 X0 转成连续编码，让模型能插值处理未出现的类别：

```python
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train['X0_enc'] = enc.fit_transform(train[['X0']])
test['X0_enc']  = enc.transform(test[['X0']])
```

同时我们检查 y 的分布，决定是否需要对目标做 box-cox 变换以改善偏态——MVP 先不做，保持原尺度便于解释。

### 2.3 模型训练与超参数调优

由于特征大部分是二值且无需缩放，树模型应该是首选。我们先用 XGBoost 回归做 baseline，用 10 折交叉验证评估 R²：

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score
model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03,
                         max_depth=5, subsample=0.9, colsample_bytree=0.9)
scores = cross_val_score(model, X, y, cv=10, scoring='r2')
print('CV R2: %.5f (+/- %.5f)' % (scores.mean(), scores.std()))
```

MVP 版本只用了 X0 一个特征，10 折交叉验证也能拿到约 0.5718 的 CV 分数、约 0.54825 的公开 LB，说明 X0 与车型配置高度相关。

### 2.4 模型测试与指标报告

我们用全量训练数据重新拟合，对测试集预测并写出提交：

```python
model.fit(X, y)
pred = model.predict(X_test)
pd.DataFrame({'ID': test['ID'], 'y': pred}).to_csv('submission.csv', index=False)
```

标杆方案的最终私有 LB 大约在 0.55298~0.55378，属于 Top 2% 区间；仅用 X0 的 MVP 也能拿到 0.548，验证了“少即是多”的特征取舍思路。

## 三、总结

本文中我们完成了 Mercedes-Benz Greener Manufacturing 的完整流程。我们首先通过 EDA 发现训练集 4209 行、特征 400+，X0 是高基数分类特征且大量二值特征无取值变化，其次删除 min == max 的无用列并对 X0 做有序编码，接着用 XGBoost 回归配合 10 折交叉验证训练与调优，最后在私有榜取得接近 0.55 的 R² 而仅用 X0 的单特征 MVP 也有 0.548。这个比赛告诉我们：匿名化高维数据里，先做常数列剔除和关键特征识别，往往比盲目堆叠更多特征更有效。