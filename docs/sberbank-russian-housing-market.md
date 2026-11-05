# Sberbank 俄罗斯房地产市场

## 一、比赛简介

### 1.1 比赛背景

Sberbank 是俄罗斯历史最悠久、规模最大的银行。银行发起这场比赛，是希望通过机器学习的方式对房地产价格做出预测，从而让租户、开发商和贷款方在做决策时更有信心。住房成本对普通消费者和开发商而言都是一笔很大的投资，价格判断错了，损失会很直接。俄罗斯的房地产市场整体上比较稳定，但国家经济环境波动较大，利率、通胀、居民收入等宏观因素都会影响房价，因此用房屋特征去预测价格并不是一件简单的事。除了面积、卧室数量、楼层、建筑年份这些房屋本身的属性，房屋所在位置、到地铁站的距离、周边学校与医院的数量、区域人口结构等也会显著影响价格。更麻烦的是，这些特征之间还存在复杂的交互关系，比如同样是三居室，在市中心和郊区价格可能相差数倍。比赛提供的数据集正是围绕这些维度构建的，既包含单笔房屋交易的微观特征，也包含俄罗斯宏观经济与金融部门的时间序列数据。参赛者需要把这两类信息结合起来，才能做出比较准确的价格预测。

### 1.2 比赛目标

比赛的目标是预测每一笔房屋交易的价格。训练数据共 30471 条样本、292 个特征列；测试数据共 7662 条样本、291 个特征列。训练数据中，id 是交易编号，timestamp 是交易时间，full_sq 是房屋总面积，life_sq 是居住面积，floor 是所在楼层，max_floor 是建筑总楼层，material 是建筑材料，build_year 是建筑年份，num_room 是房间数，kitch_sq 是厨房面积，state 是房屋状态。区域类特征包括 raion_popul 区域人口、green_zone_part 绿地占比、children_preschool 幼儿园数量、school_quota 学校配额、hospital_beds_raion 医院床位数、university_top_20_raion 名校数量、shopping_centers_raion 商场数量等。距离类特征包括 metro_min_walk 到地铁的步行时间、kindergarten_km 到幼儿园的距离、park_km 到公园的距离、kremlin_km 到克里姆林宫的距离等。除此之外，macro.csv 提供了俄罗斯宏观经济和金融部门的时间序列数据，可以通过 timestamp 与房屋数据合并。目标列是 price_doc，表示这笔交易的价格。测试集不包含 price_doc，参赛者最终需要提交一个包含 id 和 price_doc 两列的文件。

### 1.3 评价标准

比赛采用均方根对数误差 RMSLE 作为评价指标。它的做法是先对真实价格和预测价格分别取对数，再计算两组对数值的均方根误差。用代码表示就是 `np.sqrt(mean_squared_log_error(np.exp(y_true), y_pred))`，其中 `np.exp` 是把对数还原成原始价格。RMSLE 的数值越小越好，理想情况下接近零。与普通均方根误差相比，RMSLE 对高价房的绝对误差不那么敏感，更关注预测值相对于真实值的比例偏差，这正好符合房价分布右偏、高价样本少但数值大的特点。也正因为如此，后面我们会把目标列先取对数再训练模型，让模型的优化目标和比赛的评分标准保持一致。

## 二、解题过程

我们按数据探索、数据清洗、模型训练与调优、模型测试与指标报告四步进行。

### 2.1 数据探索（EDA）

我们首先把训练和测试文件从压缩包中读取出来，确认数据的形状和基本信息。

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

代码解压并读取 csv，然后打印形状。训练集是 30471 行 292 列，测试集是 7662 行 291 列，说明测试集和训练集的特征基本一致，只是少了目标列 price_doc。数据集包含三种数据类型 int64、object、float64，特征数量非常多，直接建模前需要先弄清哪些列有缺失、哪些列分布异常。

接着我们查看数值摘要和目标分布。

```python
df_train.describe()
import numpy as np
df_train['price_doc_log'] = np.log(df_train['price_doc'])
df_train['price_doc_log'].describe()
```

从摘要可以看到，full_sq 的均值约为 54.2，标准差约 38.0，最小值是 0，最大值达到 5326，说明面积跨度极大，可能存在录入错误或特殊物业。life_sq 只有 24088 个非空值，缺少了六千多条记录；floor 有 30304 个非空值，max_floor 只有 20899 个非空值，说明楼层信息缺失比较严重。build_year 的最大值异常达到 20052007，明显是数据错误，这类值需要在清洗阶段处理。price_doc 本身右偏严重，取对数后分布更接近正态，因此我们新增 price_doc_log 作为建模目标。我们还计算了 price_doc 与各个特征的皮尔逊相关系数，发现 build_count_before_1920、leisure_count_1000、cafe_sum_2000_min_price_avg 等与价格有弱正相关，而大量特征与目标几乎不相关，说明特征筛选和特征工程对提升模型很重要。

### 2.2 数据清洗

我们发现训练集中存在大量缺失值，先对离散特征用众数填充，连续特征用均值填充，再处理时间列和类别列。

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

代码用众数填补离散特征，比如楼层、物业类型、房间数、房屋状态等，用均值填补连续特征，比如居住面积、到地铁的步行时间等。对离散特征用均值没有意义，所以必须用众数；对连续特征用众数会重复同一个值，也不合适，所以用均值。这里还有一个细节：测试集的缺失值必须用训练集的统计量来填，否则会引入测试集信息，造成数据泄漏。我们还对 16 个包含缺失值的字符串列，比如 nuclear_reactor_raion、product_type、radiation_raion 等，使用 SimpleImputer 的 most_frequent 策略统一填充，保证训练和测试的缺失处理方式完全一致。

随后我们对时间戳进行解析并对类别特征进行编码。

```python
df_train['timestamp'] = pd.to_datetime(df_train['timestamp'], errors='coerce')
df_train['year'] = df_train['timestamp'].dt.year
df_train['month'] = df_train['timestamp'].dt.month
df_train['day'] = df_train['timestamp'].dt.day

from sklearn.preprocessing import OrdinalEncoder
cat_cols = ['product_type','sub_area','ecology']
ohe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df_train[cat_cols] = ohe.fit_transform(df_train[cat_cols])
```

把 timestamp 拆分为年、月、日，可以让模型捕捉房价的季节性和宏观趋势，因为不同年份的房价水平本身就有系统差异。OrdinalEncoder 把文本类别映射成整数，便于树模型分裂；`handle_unknown='use_encoded_value'` 和 `unknown_value=-1` 则保证测试集出现训练时没见过的类别时不会报错。我们还对大量右偏的数值特征做了 log1p 变换，压缩极端值的影响；目标列统一取对数，使损失函数与 RMSLE 的评价方式对齐。对于与目标相关性极低的特征，我们也按相关性排序后舍弃了尾部的一部分，以减少噪声和计算量。

### 2.3 模型训练与超参数调优

我们先做 80/20 划分，用对数目标训练基线模型。

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

我们首先用决策树做基线，设置 max_depth=5 限制树的深度，防止它把训练集背下来；min_samples_split=20 要求一个节点至少有一定数量的样本才继续分裂，让模型更稳健。训练完成后在验证集上计算 RMSLE，结果约为 0.496。这个分数不算差，说明房屋特征确实包含价格信号，但决策树只能做分段常数预测，拟合能力有限，所以还有明显提升空间。

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

我们设置学习率 eta 为 0.05，让每一棵树只走一小步，整体更稳定；max_depth 为 5 控制单棵树的复杂度；colsample_bytree 为 0.7 表示每棵树随机使用七成特征，降低过拟合风险；objective 使用 reg:squarederror 做回归，eval_metric 设为 rmse 方便观察对数目标上的误差。early_stopping_rounds=20 的作用是：如果验证集误差连续 20 轮没有下降，就提前停止训练，既省时间又避免过拟合。训练日志显示验证 rmse 从 14.37 快速下降，最终稳定在 0.466 左右。随后我们在验证集上计算 RMSLE，得到约 0.4657，相比决策树有明显提升。我们还用 SHAP 分析了特征重要性，发现 full_sq、life_sq、area_m 等面积相关特征，以及 metro_min_walk、kremlin_km 等位置距离特征贡献较大。调参时我们重点保持了较小的学习率和适中的树深，在收敛速度和泛化能力之间取得平衡。

### 2.4 模型测试与指标报告

在确定模型后，我们用全部训练数据重新拟合，再对测试集预测并生成提交文件。

```python
full_model = xgb.train(xgb_params, xgb.DMatrix(X, y), num_boost_round=model.best_iteration)

y_test_log = full_model.predict(xgb.DMatrix(X_test))
y_test = np.exp(y_test_log)

submission = pd.DataFrame({'id': df_test['id'], 'price_doc': y_test})
submission.to_csv('submission.csv', index=False)
```

我们用验证集上得到的最佳迭代次数 model.best_iteration，在全量数据上重新训练一个模型，这样既保留了早停选出的合适树数量，又能利用全部样本的信息。对测试集预测出来的结果是对数价格，需要做指数反变换回到原始价格尺度，再和 id 一起组成提交文件。最后检查 submission.csv 的列名、行数和测试集是否一致，确认没有缺失值或负数。

在验证集上最终 RMSLE 约为 0.4657，说明模型在对数空间拟合良好。整体来看，面积特征、建筑年代、到地铁的距离以及区域人口结构对价格预测影响显著，而单纯堆叠大量弱相关特征并不会带来同等幅度的提升。

## 三、总结

本文中我们完成了 Sberbank 俄罗斯房地产市场预测的完整流程。我们首先通过 EDA 明确了数据的规模和分布，训练集 30471 行 292 列，目标列 price_doc 右偏严重，面积特征跨度大，楼层和居住面积存在较多缺失值。其次我们进行了数据清洗，对缺失值分别用众数和均值填补，拆分时间戳并对类别特征做 Ordinal 编码，同时对数值特征做 log1p 变换并把目标列取对数。接着我们用决策树做基线，验证集 RMSLE 约 0.496，然后使用 XGBoost 并通过早停调参，验证集 RMSLE 降至约 0.4657。最后我们在全量数据上重新训练并生成提交文件，完成模型测试与指标报告。整个方案体现了先清洗、再特征、后模型的基础思路，也说明了对数变换与早停策略在房价回归任务中的实际价值，适合作为入门者的参考。
