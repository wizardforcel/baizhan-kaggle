# Zillow 房价误差预测（Zillow Prize）

## 一、比赛简介

### 1.1 比赛背景

Zillow 是美国一家做房产估价的网站，它最有名的产品叫 **Zestimate**：只要输入一个地址，网站就会给出一个估计房价。这个估计值是用一套自动估价模型算出来的，但模型不可能永远准确，实际成交价和估计值之间总会有偏差。

这个比赛要参赛者做的，不是直接预测房价，而是**预测 Zestimate 的误差**。换句话说，Zillow 已经给出了一个估价，我们要判断「这个估价大概会偏高还是偏低、偏多少」。听起来有点绕，但这样做有两个好处：一是把问题从「预测绝对价格」变成「预测相对误差」，数值范围更小、更稳定；二是 Zillow 可以直接拿预测结果去修正自己的估价系统。

比赛数据来自美国几个县的房产记录，包括房屋面积、卧室数量、浴室数量、建造年份、地税、经纬度等几十个字段。训练集提供了 2016 年和 2017 年真实成交记录，以及每笔成交对应的误差值；测试集只给房产属性，要求我们预测这些房产的误差。

这个比赛的难点在于：第一，误差本身的分布非常集中，大多数成交的误差都在 0 附近，模型很容易学成「全部预测 0」；第二，房产特征缺失严重，很多字段有一半以上是空值；第三，同一个房产可能在两年里多次成交，训练集和测试集之间还存在时间和地理上的重叠，划分验证集时必须小心。

### 1.2 比赛目标

输入是每个房产的属性表。主要字段包括：

- `parcelid`：房产编号，可以理解成房子的身份证号；
- `calculatedfinishedsquarefeet`：建筑面积；
- `bedroomcnt`、`bathroomcnt`：卧室数、浴室数；
- `yearbuilt`：建造年份；
- `latitude`、`longitude`：经纬度；
- `taxvaluedollarcnt`、`taxamount`：评估价和税额；
- `fips`、`regionidcity`、`regionidzip`：县、城市、邮编等地区编码。

训练集多一列 `logerror`，也就是我们要预测的目标。它的定义是：

```
logerror = log(Zestimate) - log(实际成交价)
```

如果 `logerror` 是正数，说明 Zestimate 估高了；如果是负数，说明估低了。测试集要求我们为每个 `parcelid` 输出一个 `logerror` 预测值，提交文件包含 `parcelid` 和 `201610`、`201611`、`201612`、`201710`、`201711`、`201712` 等月份列。

### 1.3 评价标准

比赛使用 **RMSE（Root Mean Squared Error，均方根误差）** 来评价预测结果：

```
RMSE = sqrt( mean( (真实 logerror - 预测 logerror)^2 ) )
```

它的意思是：先把每个样本的误差平方，取平均，再开根号。因为先平方再开方，大误差会被放大，所以 RMSE 对「偶尔预测得特别离谱」的情况很敏感。分数越低越好，0 表示完全预测正确。

作为参照，材料里给出的「全部预测训练集平均值」这个最笨的基线，RMSE 大约是 **0.0837**；比赛排行榜上比较好的成绩能到 **0.064** 左右（材料中一份完整方案的 Public Score 是 0.06416，Private Score 是 0.07497）。这说明这个任务的提升空间其实不大，能把 RMSE 从 0.0837 降到 0.065 附近就已经很不错了。

---

## 二、解题过程

我们的主线是：先把 2016 和 2017 两年的房产属性、成交记录拼成一张大表，然后做缺失值处理和特征工程，接着用随机森林和 XGBoost 分别建模，最后用验证集 RMSE 选模型、生成提交文件。下面按四步展开。

### 2.1 数据探索（EDA）

我们首先把四个主文件和提交样例读进来，看看数据规模：

```python
import pandas as pd

data_dir = "/kaggle/input/zillow-prize-1/"

# 2016、2017 两年的房产属性表
X_train_2016 = pd.read_csv(f"{data_dir}properties_2016.csv")
X_train_2017 = pd.read_csv(f"{data_dir}properties_2017.csv")

# 两年的成交记录，里面带目标列 logerror
y_train_2016 = pd.read_csv(f"{data_dir}train_2016_v2.csv")
y_train_2017 = pd.read_csv(f"{data_dir}train_2017.csv")

# 提交样例，告诉我们要预测哪些 parcelid
df_submission = pd.read_csv(f"{data_dir}sample_submission.csv")

print("2016 属性表:", X_train_2016.shape)
print("2016 成交表:", y_train_2016.shape)
print("2017 属性表:", X_train_2017.shape)
print("2017 成交表:", y_train_2017.shape)
print("提交样例:", df_submission.shape)
```

输出是：

```
2016 属性表: (2985217, 58)
2016 成交表: (90275, 3)
2017 属性表: (2985217, 58)
2017 成交表: (77613, 3)
提交样例: (2985217, 7)
```

也就是说，属性表有将近 300 万行、58 列，但真正有成交记录的只有 9 万条左右。这是一个典型的「属性多、标签少」的问题：大量房产只出现在测试集里，没有历史成交价可参考，模型只能靠属性去推断误差。

接着我们看目标变量的分布：

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 只取 2016 年的成交记录做示例
train_df = y_train_2016.copy()
train_df["transactiondate"] = pd.to_datetime(train_df["transactiondate"])

# 画出 logerror 的直方图
plt.figure(figsize=(8, 5))
sns.histplot(train_df["logerror"], bins=100, kde=True)
plt.xlabel("logerror")
plt.ylabel("count")
plt.title("Distribution of logerror")
plt.show()

print(train_df["logerror"].describe())
```

从图上看，`logerror` 基本围绕 0 对称分布，绝大多数值落在 -0.2 到 0.2 之间，只有极少数样本跑到 -3 或 +3。这告诉我们两件事：第一，目标本身接近正态，适合用回归模型；第二，极端值虽然少，但平方之后会严重拉高 RMSE，所以后面要考虑把极端值剔除或做截断。

我们还按月份统计了成交笔数：

```python
# 从交易日期里提取月份
train_df["transaction_month"] = train_df["transactiondate"].dt.month

# 统计每个月的成交数量
month_count = train_df["transaction_month"].value_counts().sort_index()
print(month_count)

plt.figure(figsize=(10, 5))
sns.barplot(x=month_count.index, y=month_count.values)
plt.xlabel("Month")
plt.ylabel("Number of transactions")
plt.show()
```

结果显示成交有明显的季节性：春季和夏季成交多，冬季成交少。这个信息可以做成 `transaction_month` 特征，让模型知道「这笔交易发生在几月」，因为不同月份的估价误差可能有系统性差异。

最后我们检查同一个房产是否重复出现：

```python
# 统计每个 parcelid 出现的次数
counts = train_df["parcelid"].value_counts().value_counts()
print(counts)
```

输出显示：90,026 个房产只成交过一次，123 个房产成交过两次，只有 1 个房产成交过三次。重复比例很低，说明大部分样本是独立的。但这也提醒我们，不能简单按行随机划分验证集，因为同一个房产的两次成交如果被分到训练和验证两边，模型会「记住」这个房产，导致验证分数虚高。更稳妥的做法是按 `parcelid` 分组划分，或者用时间切分——用 2016 年训练、2017 年验证。

### 2.2 数据清洗

数据清洗的第一步是把成交表和属性表按 `parcelid` 合并。成交表只有 `parcelid`、`logerror`、`transactiondate` 三列，属性表有 58 列，合并之后才是完整的训练数据：

```python
def load_housing_data(housing_path):
    properties_2016 = pd.read_csv(f"{housing_path}/properties_2016.csv")
    properties_2017 = pd.read_csv(f"{housing_path}/properties_2017.csv")
    train_2016 = pd.read_csv(f"{housing_path}/train_2016_v2.csv")
    train_2017 = pd.read_csv(f"{housing_path}/train_2017.csv")

    # 左连接：以成交记录为准，把对应的房产属性拼进来
    train_2016 = pd.merge(train_2016, properties_2016,
                          how="left", on="parcelid")
    train_2017 = pd.merge(train_2017, properties_2017,
                          how="left", on="parcelid")

    # 两年数据纵向拼在一起
    all_properties = pd.concat([properties_2016, properties_2017],
                               ignore_index=True)
    all_training = pd.concat([train_2016, train_2017],
                             ignore_index=True)
    return all_properties, all_training

all_properties, housing = load_housing_data(
    "/kaggle/input/zillow-prize-1"
)
print(housing.shape)
```

合并之后，我们做了三件清洗工作。

第一，**去重**。同一个房产在同一天可能有多条记录，重复样本会让模型重复学习同一条信息。我们按 `parcelid` 和 `transactiondate` 两个字段去重，保留最后一条：

```python
def drop_duplicates(housing):
    # subset 指定用哪几列判断重复
    # keep="last" 表示保留最后出现的那条
    return housing.drop_duplicates(
        subset=["parcelid", "transactiondate"],
        keep="last",
        ignore_index=True
    )

housing = drop_duplicates(housing)
print("去重后:", housing.shape)
```

第二，**处理极端值**。前面看到 `logerror` 有少量极端值，这些值可能是录入错误，也可能是特殊交易（比如亲属之间低价过户）。材料里的做法是只保留距离均值 2.5 个标准差以内的样本：

```python
import numpy as np

# 计算均值和标准差
mean = housing["logerror"].mean()
std = housing["logerror"].std()

# 只保留 2.5 个标准差以内的样本
housing = housing[
    (housing["logerror"] > mean - 2.5 * std) &
    (housing["logerror"] < mean + 2.5 * std)
].copy()

print("去掉极端值后:", housing.shape)
```

这里要注意，极端值只能从训练集里删，验证集和测试集不能动。因为比赛要评估的是模型在真实分布上的表现，如果验证集也被清洗过，分数就会偏乐观。

第三，**处理缺失值**。这是 Zillow 数据里最麻烦的部分。我们先统计每一列的缺失比例：

```python
# 统计所有列的缺失比例，按从高到低排序
missing_ratio = all_properties.isnull().mean().sort_values(ascending=False)
missing_df = pd.DataFrame({"Missing Ratio": missing_ratio})

# 只看缺失最严重的前 35 列
print(missing_df.head(35))
```

结果里有一批字段缺失超过 95%，比如 `finishedsquarefeet13`、`finishedsquarefeet15`、`storytypeid` 等。这些字段即使能填，填出来的值也基本是猜的，反而会引入噪声，所以我们直接删掉。对于缺失比例中等、且业务上重要的字段，我们分类型填充：

- 计数类字段（车库数量、壁炉数量、泳池面积等）缺失通常表示「没有」，填 0；
- 类别字段（空调类型、供暖类型、房屋用途等）填众数；
- 连续数值字段（建筑面积、浴室数、地价等）填中位数。

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 缺失表示“没有”的字段，填 0
zero_cols = [
    "yardbuildingsqft17", "fireplacecnt", "poolcnt",
    "garagecarcnt", "garagetotalsqft", "pooltypeid2",
    "poolsizesum", "decktypeid", "taxdelinquencyflag"
]

# 类别字段，填出现最多的值
mode_cols = [
    "airconditioningtypeid", "heatingorsystemtypeid", "unitcnt",
    "fips", "propertylandusetypeid", "regionidcounty", "yearbuilt"
]

# 连续数值字段，填中位数
median_cols = [
    "buildingqualitytypeid", "lotsizesquarefeet", "bathroomcnt",
    "bedroomcnt", "calculatedfinishedsquarefeet",
    "structuretaxvaluedollarcnt", "landtaxvaluedollarcnt",
    "latitude", "longitude"
]

# ColumnTransformer 可以按列分组做不同的填充
imputer = ColumnTransformer([
    ("fill_zero", SimpleImputer(strategy="constant", fill_value=0), zero_cols),
    ("fill_mode", SimpleImputer(strategy="most_frequent"), mode_cols),
    ("fill_median", SimpleImputer(strategy="median"), median_cols),
], remainder="passthrough")
```

除了缺失值，我们还做了几组特征工程：

- 用 `yearbuilt` 算出房龄 `house_age = 2016 - yearbuilt`，因为「房龄」比「建造年份」更直观；
- 从 `transactiondate` 拆出年、月、日、季度，捕捉成交时间的季节性；
- 把 `fips`、`regionidcity`、`regionidzip` 等地区编码当成类别特征，做独热编码；
- 对建筑面积、地价等偏态严重的数值特征做 Box-Cox 变换，让分布更接近正态。

这些处理我们都封装进 sklearn 的 `Pipeline`，这样训练和预测时可以用同一套流程，不会出现「训练时填了中位数、预测时忘了填」的问题。

### 2.3 模型训练与超参数调优

我们先把成交记录分成特征 `X` 和目标 `y`，再留出一部分验证集。材料中的端到端方案采用 90% 训练、10% 验证的切分，并固定 `random_state=42`，这样每次运行都能得到同样的划分：

```python
from sklearn.model_selection import train_test_split

# logerror 是回归目标，不能把它留在输入特征里
X = housing.drop("logerror", axis=1)
y = housing["logerror"]

# 先留出验证集，训练过程中只看训练集拟合参数
X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.9, random_state=42
)
print(X_train.shape, X_val.shape)
```

我们把预处理步骤装进流水线。对线性模型，流水线会依次执行删列、填补、类型转换、生成房龄、类别编码、数值缩放、多项式特征和偏态变换；对 XGBoost，则保留更多原始字段，因为树模型可以原生处理缺失值：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 交易日期先拆成年、月、日、季度，再交给模型
date_feat_creator = CreateDateFeatures()
year_feat_creator = CreateYearFeatures(
    date_features={"yearbuilt": "house_age"}
)

# 类别列用独热编码，遇到测试集新类别时忽略而不是报错
feature_encoder = ColumnTransformer([
    ("ohe_cats", OneHotEncoder(handle_unknown="ignore"), cat_vars)
], remainder="passthrough")

xgb_preprocessor = Pipeline([
    ("feature_dropper", FeatureDropper(features_to_drop=["parcelid"])),
    ("date_feat_creator", date_feat_creator),
    ("year_feat_creator", year_feat_creator),
    ("feature_encoder", feature_encoder),
])

X_train_ready = xgb_preprocessor.fit_transform(X_train)
X_val_ready = xgb_preprocessor.transform(X_val)
```

我们首先试了线性回归、Ridge、Lasso、随机森林等模型。线性模型优点是快、容易解释，但房价误差和地理位置、面积之间的关系明显不是简单直线。随机森林把很多棵相互独立的决策树平均起来，能捕捉非线性关系，因此我们用随机搜索和网格搜索调它的主要参数：

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# 先用随机搜索粗略寻找区域，避免把所有组合都遍历一遍
param_grid = {
    "max_samples": [0.1, 0.2],
    "max_features": ["auto", 0.4, 0.6, 0.8],
    "max_depth": [50, 70, 90],
    "min_samples_leaf": [16, 32, 48],
}
rf = RandomForestRegressor(random_state=42)
rf_random = RandomizedSearchCV(
    rf, param_distributions=param_grid,
    n_iter=20, cv=3, n_jobs=-1, random_state=42
)
rf_random.fit(X_train_ready, y_train)
print(rf_random.best_params_)
```

进一步的网格搜索得到随机森林的一组较好参数：`n_estimators=400`、`max_samples=0.3`、`max_depth=70`、`max_features=0.6`、`min_samples_leaf=32`。这里 `n_estimators` 是树的数量，数量越多通常越稳定，但训练也越慢；`max_depth` 控制树的深度，太深会记住训练样本；`min_samples_leaf` 限制叶节点最少样本数，可以减轻过拟合。

```python
best_params = {
    "n_estimators": 400,
    "max_samples": 0.3,
    "max_depth": 70,
    "max_features": 0.6,
    "min_samples_leaf": 32,
    "random_state": 42,
}

ran_forest = RandomForestRegressor(**best_params)
ran_forest.fit(X_train_ready, y_train)
```

随后我们把重点放在 XGBoost 上。梯度提升树和随机森林的区别是：随机森林里的树彼此独立，XGBoost 则一棵接一棵地训练，每棵新树都专门修正前面模型的错误，所以通常精度更高，但也更容易过拟合。

```python
import xgboost as xgb

# 学习率较大时收敛快；树的数量先设高，再交给早停决定何时结束
xgb_model = xgb.XGBRegressor(
    learning_rate=0.3,
    n_estimators=10000,
    max_depth=6,
    random_state=42,
    n_jobs=4,
)

# 验证集 MAE 连续 10 轮不再下降就停止
xgb_model.fit(
    X_train_ready, y_train,
    eval_set=[(X_val_ready, y_val)],
    eval_metric="mae",
    early_stopping_rounds=10,
    verbose=False,
)
```

调参时我们重点观察 `max_depth`、`learning_rate`、`n_estimators`。材料中的建议是用较小学习率、较大的树数量，再依靠验证集早停；`max_depth` 在 1 到 6 之间逐一测试，浅树不容易过拟合。除此之外，我们还试验了 `subsample`、`colsample_bytree`、`reg_lambda`、`reg_alpha` 和 `gamma`，它们分别控制行抽样、列抽样和正则化强度。我们不盲目增加参数，而是用交叉验证比较每一次改变对 MAE 和 RMSE 的影响。

### 2.4 模型测试与指标报告

我们用 MAE 和 RMSE 一起检查模型。MAE 是绝对误差的平均值，容易理解；RMSE 会放大大误差，更接近比赛的关注点。计算函数如下：

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def get_eval_metrics(model, X_eval, y_true):
    # 同一个模型只在验证特征上预测一次，避免重复计算
    y_pred = model.predict(X_eval)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(
        y_true, y_pred, squared=False
    )
    print("MAE:", mae)
    print("RMSE:", rmse)
    return mae, rmse

rf_mae, rf_rmse = get_eval_metrics(
    ran_forest, X_val_ready, y_val
)
xgb_mae, xgb_rmse = get_eval_metrics(
    xgb_model, X_val_ready, y_val
)
```

材料中的基线是 MAE 约 0.0533、RMSE 约 0.0837。调好参数的随机森林在验证集上得到 MAE **0.05121**、RMSE **0.08174**，比直接预测均值明显好。XGBoost 采用三折交叉验证时，材料记录的三个折 RMSE 分别约为 0.0543、0.0533、0.0717，平均约 0.0597；不同折之间有差距，说明时间和地理分布会影响结果，单次随机切分不能完全代表排行榜表现。

确认模型和预处理流程以后，我们用全部可用的训练记录重新拟合，并把预测结果写进提交样例。测试属性表也必须经过同一个 `xgb_preprocessor`，不能重新计算一套填充值或编码：

```python
# 用全部训练数据重新拟合预处理器和模型
X_all = housing.drop("logerror", axis=1)
y_all = housing["logerror"]
X_all_ready = xgb_preprocessor.fit_transform(X_all)

xgb_model.fit(X_all_ready, y_all, verbose=False)

# 测试数据按相同顺序处理，保证列和训练阶段一致
test_ready = xgb_preprocessor.transform(all_properties)
test_pred = xgb_model.predict(test_ready)

# 以 sample_submission 的列名为准，避免提交格式错误
submission = df_submission.copy()
submission["201610"] = test_pred
submission["201611"] = test_pred
submission["201612"] = test_pred
submission["201710"] = test_pred
submission["201711"] = test_pred
submission["201712"] = test_pred
submission.to_csv("submission.csv", index=False)
```

提交前我们还会检查行数、列名和 `parcelid` 是否与样例一致，并随机查看几行预测值。材料中的端到端方案报告了 Public Score 0.06416、Private Score 0.07497，但它同时说明由于比赛已经结束，后来没有重新提交模型。因此我们把这些数字当作材料中的参考结果，而不是声称本地代码重新跑出了同样的排行榜分数。

## 三、总结

本文中我们完成了 Zillow 房价误差预测的完整流程。我们首先读取了 2016、2017 两年的房产属性和成交记录，确认属性表约有 298 万行、58 个字段，训练成交记录分别有 90275 和 77613 行。其次我们围绕 `logerror` 做 EDA，观察了目标分布、交易月份和重复房产，并按 `parcelid` 与交易日期去重、删除严重缺失字段、按字段含义用 0、众数或中位数填补缺失值。接着我们从交易日期生成时间特征、从建造年份生成房龄，把类别列独热编码，并用 Pipeline 保证训练和测试使用同一套处理；模型方面我们比较了线性回归、随机森林和 XGBoost，用随机搜索、网格搜索、交叉验证和早停调整参数。最后我们用 MAE 和 RMSE 评估验证集，随机森林得到约 0.0817 的 RMSE，XGBoost 的交叉验证平均 RMSE 约 0.0597，再把测试预测写成比赛要求的月份列。

这个比赛最重要的经验有三点。第一，遇到大型表格时，先理解字段含义再处理缺失值，不能把所有空值不加思考地填成同一个数字。第二，时间和房产编号可能造成数据泄漏，随机划分验证集并不总是可靠。第三，树模型虽然能处理复杂的非线性关系，但仍然需要通过早停、抽样和正则化控制过拟合。把这些步骤放进可复用的 Pipeline 后，我们不仅得到一个预测结果，也得到了一套更容易检查和复现的机器学习流程。
