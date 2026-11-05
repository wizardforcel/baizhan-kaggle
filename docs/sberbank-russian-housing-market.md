# Sberbank 俄罗斯房地产市场

## 一、比赛简介

### 1.1 比赛背景

Sberbank 是俄罗斯历史最悠久、规模最大的银行，为了帮助租户、开发商和贷款方做出更稳健的决策，银行希望能够对真实房地产的价格进行预测。俄罗斯房地产市场总体相对稳定，但在经济波动较强的环境下，房价会受到房源特征和宏观经济因素的复杂影响。比赛的发起者提供了包含房屋特征和宏观经济数据的丰富数据集，参赛者需要利用这些特征预测房价。数据来源于 Sberbank 的真实交易，涵盖建筑年代、面积、楼层、位置以及周边设施等信息，同时还包含宏观经济指标，可以与商品数据通过时间戳进行关联。

### 1.2 比赛目标

给定训练集和测试集的房屋属性，预测每一笔交易的价格。训练文件训练数据 30471 条、292 列，测试数据 7662 条、291 列。训练数据包含 id、timestamp、full_sq、life_sq、floor、max_floor、material、build_year、num_room、kitch_sq 等描述房屋本身的特征，以及 raion_popul、green_zone_part、children_preschool 等描述区域人口和公共设施的特征，同时还有 macro.csv 提供俄罗斯宏观经济和金融部门的时间序列数据，可通过 timestamp 与训练测试数据合并。目标列为 price_doc，表示房价。测试集只有 id 和特征列，没有 price_doc，提交文件需要包含 id 和 price_doc 两列。

### 1.3 评价标准

比赛采用均方根对数误差 RMSLE 作为评价指标，即对真实值和预测值取对数后再计算均方根误差。RMSLE 取值越小越好，常用的实现是 `np.sqrt(mean_squared_log_error(np.exp(y_true), y_pred))`。由于价格分布右偏，使用对数误差可以降低对高价样本的惩罚，让模型更关注相对误差而非绝对误差。

## 二、解题过程

我们按数据探索、数据清洗、模型训练与调优、模型测试与指标报告四步推进。

### 2.1 数据探索（EDA）

我们首先把训练和测试文件从压缩包中读取出来，确认数据形状和基本统计信息。

```python
import pandas as pd
import zipfile

train_zip = zipfile.ZipFile('/kaggle/input/sberbank-russian-housing-market/train.csv.zip')
test_zip = zipfile.ZipFile('/kaggle/input/sberbank-russian-housing-market/test.csv.zip')

df_train = pd.read_csv(train_zip.open('train.csv'))
df_test = pd.read_csv(test_zip.open('test.csv'))

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)
```

代码先解压并读取 csv，然后打印形状。训练集是 30471 行 292 列，测试集是 7662 行 291 列，说明特征基本一致，只是缺了目标列。我们看到数据集包含 int64、object 和 float64 三种类型，特征数量非常多，需要进一步了解缺失情况和分布。

接着我们查看总体描述和目标分布。

```python
# 看训练集的数值摘要
df_train.describe()
# 目标列的对数分布
import numpy as np
df_train['price_doc_log'] = np.log(df_train['price_doc'])
df_train['price_doc_log'].describe()
```

我们发现 full_sq 的均值约为 54.2，life_sq 只有 24088 个非空值，说明大量房屋缺乏居住面积记录；build_year 的统计显示最小值异常大，提示有脏数据。price_doc 整体呈现右偏分布，对数化后更加接近正态，这为后续建模提供了依据。我们还检查了 price_doc 与部分特征的皮尔逊相关性，例如 build_count_before_1920、leisure_count_1000 等与价格有弱相关，大量特征与目标的相关性接近零，说明特征工程和选择是关键。

### 2.2 数据清洗

我们发现训练集中存在大量的缺失值。先对部分特征用众数填充，部分用均值填充，再处理时间列和类别列。

```python
cols_fillna_mode = ['floor','product_type','num_room','state','hospital_beds_raion']
cols_fillna_mean = ['life_sq','metro_min_walk','metro_km_walk']

for col in cols_fillna_mode:
    df_train[col].fillna(df_train[col].mode().iloc[0], inplace=True)
    df_test[col].fillna(df_train[col].mode().iloc[0], inplace=True)

for col in cols_fillna_mean:
    df_train[col].fillna(df_train[col].mean(), inplace=True)
    df_test[col].fillna(df_train[col].mean(), inplace=True)
```

代码用众数填补离散特征，如楼层、物业类型等，用均值填补连续特征，如居住面积、到地铁的时间等。这种做法避免了用零填充带来的偏差，保持了分布特性。

随后我们对时间戳进行解析并对类别特征进行编码。

```python
# 时间戳拆分为年、月、日
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'], errors='coerce')
df_train['year'] = df_train['timestamp'].dt.year
df_train['month'] = df_train['timestamp'].dt.month

from sklearn.preprocessing import OrdinalEncoder
cat_cols = ['product_type','sub_area','ecology']
ohe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df_train[cat_cols] = ohe.fit_transform(df_train[cat_cols])
```

把 timestamp 拆分为年、月、日，可以让模型捕捉季节性和宏观趋势。OrdinalEncoder 把文本类别映射为整数，便于树模型使用。我们还对大量右偏的数值特征做了 log1p 变换，减小异常值影响，并把目标列统一用对数处理，使得损失函数与 RMSLE 目标一致。

### 2.3 模型训练与超参数调优

我们先做 80/20 分层划分，用对数目标训练基线模型。

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error

X = df_train_encoded.drop(['price_doc_log'], axis=1)
y = df_train_encoded['price_doc_log']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2022)

tree = DecisionTreeRegressor(random_state=2022, max_depth=5, min_samples_split=20)
tree.fit(X_train, y_train)
tree_pred = np.exp(tree.predict(X_val))
rmsle = np.sqrt(mean_squared_log_error(np.exp(y_val), tree_pred))
print('RMSLE:', rmsle)
```

我们首先用决策树做基线，设置 max_depth=5 防止过拟合，min_samples_split=20 增加稳健性。训练后在验证集上 RMSLE 约为 0.496，说明简单树可以捕捉部分规律，但仍有提升空间。

接着我们改用 XGBoost 并加入早停策略。

```python
import xgboost as xgb

dmatrix_train = xgb.DMatrix(X_train, y_train)
dmatrix_val = xgb.DMatrix(X_val, y_val)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'verbosity': 0
}
model = xgb.train(xgb_params, dmatrix_train, num_boost_round=1000,
                  evals=[(dmatrix_val, 'val')],
                  early_stopping_rounds=20, verbose_eval=20)
```

我们设置学习率 0.05，树深 5，列采样 0.7，通过 early_stopping_rounds=20 在验证集 rmse 停止下降时终止训练，避免过拟合。训练日志显示验证 rmse 从 14.37 快速下降到约 0.466。随后我们在验证集上计算 RMSLE 得到约 0.466，相比决策树有明显提升。我们还尝试用 SHAP 分析特征重要性，发现 full_sq、life_sq、area_m 等面积相关特征以及位置距离特征贡献较大。

### 2.4 模型测试与指标报告

在确定模型后，我们用全部训练数据重新拟合，再对测试集预测并生成提交文件。

```python
# 用全量数据训练
full_model = xgb.train(xgb_params, xgb.DMatrix(X, y), num_boost_round=model.best_iteration)

# 测试预测并反对数
y_test_log = full_model.predict(xgb.DMatrix(X_test))
y_test = np.exp(y_test_log)

submission = pd.DataFrame({'id': df_test['id'], 'price_doc': y_test})
submission.to_csv('submission.csv', index=False)
```

我们用验证集上的最佳迭代次数在全量数据上重新训练，得到更稳定的模型。对测试集预测后做指数反变换，得到价格的原始尺度。最后生成符合 Kaggle 要求的 submission.csv，包含 id 与 price_doc 两列。

在验证集上最终 RMSLE 约 0.4657，说明模型在对数空间拟合良好。整体来看，面积特征、建筑年代、到地铁的距离以及区域人口结构对价格预测影响显著。

## 三、总结

本文中我们完成了 Sberbank 俄罗斯房地产市场预测的完整流程。我们首先通过 EDA 明确了数据的规模和分布，训练集 30471 行 292 列，目标列 price_doc 右偏严重。其次我们进行了数据清洗，对缺失值分别用众数和均值填补，拆分时间戳并对类别特征做 Ordinal 编码，同时对数值特征做 log1p 变换并把目标列取对数。接着我们用决策树做基线，RMSLE 约 0.496，然后使用 XGBoost 并通过早停调参，验证集 RMSLE 降至约 0.466。最后我们在全量数据上重训并生成提交文件，完成模型测试与指标报告。整个方案体现了“先清洗、再特征、后模型”的入门思路。
