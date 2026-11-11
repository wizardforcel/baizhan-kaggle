# Boston Housing 房价回归（SenatorovAI Boston Housing Regression）

## 一、比赛简介

### 1.1 比赛背景

Boston Housing（波士顿房价）是机器学习里非常经典的一个入门数据集，很多教程都用它来演示「回归」是怎么一回事。本次比赛由 SenatorovAI 组织，就是把这份经典数据整理成一场正式的 Kaggle 比赛，让参赛者用一套完整的流程（读数据、训练模型、生成提交文件）来预测房价。

数据采集自美国波士顿地区不同街区的房产记录，每个街区都记录了 13 个和房屋、环境有关的特征。这个任务的难点在于：房价不是由单一因素决定的，而是被犯罪率、通勤距离、税收、学校质量等一堆因素共同影响，而且很多因素和房价的关系并不是简单的直线关系。例如「每户有多少房间」对房价是正向作用，但到了一定程度后边际效果就减弱了，这种非线性关系恰恰是回归模型的考点。

### 1.2 比赛目标

我们要根据特征预测房价，输入是训练集和测试集两套表格：

- 训练集（`train.csv`）：包含 13 个特征列和 1 个目标列 `medv`；
- 测试集（`test.csv`）：只有特征列和 `Id` 编号，没有目标列；
- 提交样例（`sample_submission.csv`）：以 `Id` 和 `MEDV` 两列给出格式模板。

13 个特征列的含义分别是：

| 特征 | 含义 |
| --- | --- |
| `crim` | 城镇人均犯罪率（越高越不利） |
| `zn` | 面积超过 25000 平方英尺的住宅用地比例 |
| `indus` | 城镇中工业用地所占比例 |
| `chas` | 是否紧邻查尔斯河（1 是，0 否） |
| `nox` | 空气中氮氧化物浓度，衡量污染程度 |
| `rm` | 每套住宅的平均房间数 |
| `age` | 1940 年以前建成的房屋占比 |
| `dis` | 到波士顿五个就业中心的加权距离 |
| `rad` | 径向公路通达性指数 |
| `tax` | 每 1 万美元的房产税率 |
| `ptratio` | 城镇中师生比（越小教育越好） |
| `b` | 由居民构成推算的指标 |
| `lstat` | 低社会地位人口所占百分比 |

我们要做的也就是：利用训练集学会「特征 → 房价」的规律，然后对测试集预测每个街区的 `medv`，即**家庭价格中位数**（单位是千美元），并生成提交文件上传。

### 1.3 评价标准

从各份提交代码来看，参赛者不约而同用**均方根误差（Root Mean Squared Error，简称 RMSE）**来评估自己的模型。在了解它之前，先解释两个更基础的概念：误差是我们预测值和真实值之间的差距；均方误差（MSE）是把所有样本的误差先平方再加起来求平均，这样能让大误差被放大。对均方误差再开一次平方根，就得到 RMSE，它的单位跟房价一致（这里是千美元），更好理解。RMSE 越低，说明预测越接近真实房价，排行榜名次越高。除了 RMSE，参赛者通常还会顺便报告平均绝对误差（MAE）和决定系数 R²（越接近 1 越好）。

## 二、解题过程

整场比赛我们可以总结成一条主线：**先做 EDA 看清数据，再做少量清洗和特征工程，然后从简单的线性回归 baseline 起步，最后用一堆更高级的模型放到同一个框架里比较，选出 RMSE 最低的一个提交。** 我们按四步来走。

### 2.1 数据探索（EDA）

我们首先把比赛要求的三个文件读进来，看看数据规模、列名和数据格式：

```python
# numpy 做数值计算，pandas 读写表格，都是 Kaggle 的标配库
import numpy as np
import pandas as pd

# 从比赛目录读取三个文件：训练集、测试集、提交样例
train_path = "/kaggle/input/competitions/senatorov-ai-boston-housing-regression/train.csv"
test_path = "/kaggle/input/competitions/senatorov-ai-boston-housing-regression/test.csv"
sample_path = "/kaggle/input/competitions/senatorov-ai-boston-housing-regression/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

# 打印规模，确认列名完整、测试集确实没有目标列
print("train shape:", train_df.shape)
print("test shape:", test_df.shape)
print("train columns:", list(train_df.columns))
print("test columns:", list(test_df.columns))

# 看一眼前几行，确认数据格式
train_df.head()
```

我们看到三个文件的列都能对应上，训练集里带着目标列 `medv`，测试集只有特征。接着我们检查每个特征的类型、有没有缺失值，并画出目标变量的分布：

```python
# 查看每一列的数据类型，确认是否都是数值型
train_df.dtypes

# 统计每一列的缺失值个数，为后续清洗做准备
train_df.isna().sum()

# 输出训练集的整体统计信息（均值、标准差、四分之一分位数等）
train_df.describe()

# 画出目标变量 medv 的直方图，直观地看房价分布
train_df['medv'].hist()
```

所有列都是数值型，也没有缺失值，数据相当干净。不过直方图显示了一个重要现象：`medv` 在 50（千美元）左右有一个很高的柱子。这是因为原版 Boston Housing 数据里，超过 50 的房价当年被统一记录成了 50，形成了截断。我们后面清洗时专门处理这一点。

### 2.2 数据清洗

由于数据没有缺失值，我们的清洗工作主要集中在**特征工程**上。特征工程是指：在原始特征的基础上，组合、变换出一些更有信息量的新列。我们构造了每个房间对应的税率、房价中位的平方项等六个组合特征：

```python
# 组合特征函数：在原始 13 个特征的基础上造出新特征
def add_features(df):
    df = df.copy()                    # 复制一份，避免破坏原数据
    if 'RM' in df.columns and 'AGE' in df.columns:
        # 平均每个住户的房间数：RM 除以住户数（AGE 加 1 防止除零）
        df['rooms_per_household'] = df['RM'] / (df['AGE'] + 1)
    if 'CRIM' in df.columns and 'ZN' in df.columns:
        # 犯罪率与住宅区比例的比值，衡量「又乱又高密度」的程度
        df['crime_ratio'] = df['CRIM'] / (df['ZN'] + 1)
    if 'TAX' in df.columns and 'RM' in df.columns:
        # 每间房摊到的税，把税收和房间数两个信息合在一起
        df['tax_per_room'] = df['TAX'] * df['RM']
    if 'CHAS' in df.columns and 'RM' in df.columns:
        # 是否临河 × 房间数，捕捉河景房里的额外价值
        df['chas_rm'] = df['CHAS'] * df['RM']
    if 'LSTAT' in df.columns:
        df['lstat_sq'] = df['LSTAT'] ** 2      # 低社会地位比例的平方
    if 'RM' in df.columns:
        df['rm_sq'] = df['RM'] ** 2            # 房间数的平方，捕捉非线性
    return df
```

构造这些特征的原因很朴素：把两个特征相乘或平方，相当于给模型提供了「一个特征依赖另一个特征」的线索，这种线索单靠线性回归是发现不了的。回到前面 EDA 留心到的截断问题，我们把目标值超过 50 的训练样本删掉，避免那些被记成固定值的样本干扰学习：

```python
# 把特征和标签分开：丢掉目标列和 Id 编号
X = train_df.drop([target, 'Id'], axis=1)
y = train_df[target]

# 先加组合特征，再按标签过滤
X = add_features(X)

# 保留房价低于 50 的样本，去掉被截断的高价位记录
mask = y < 50
X = X[mask]
y = y[mask]
```

### 2.3 模型训练与超参数调优

我们首先训练一个最简单的**线性回归（Linear Regression）**模型。线性回归假设「每个特征乘上一个系数、再加一个偏置」就能得到房价，它是最基本的回归模型，最适合当 baseline（基准线）：

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 按 8:2 切分训练集和验证集，固定随机种子保证结果可复现
xtrain, xvalid, ytrain, yvalid = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 创建并训练线性回归模型
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# 在验证集上预测，并与真实值比较
y_valid_pred = regressor.predict(xvalid)
mse = mean_squared_error(yvalid, y_valid_pred)
mae = mean_absolute_error(yvalid, y_valid_pred)
rmse = np.sqrt(mse)                  # 对 MSE 开方，得到 RMSE
r2 = r2_score(yvalid, y_valid_pred)

print("MSE:", mse, "MAE:", mae, "RMSE:", rmse, "R2:", r2)
```

验证集上的四项指标都打印出来了。我们还画了一张「真实房价 vs 预测房价」的散点图：如果模型足够好，点应该密密麻麻地贴在一条 45 度对角的直线附近。基线结果能跑通，但线性回归对非线性关系无能为力，所以我们接着尝试**标准化 + 对数变换**的改良版：先用 StandardScaler 把每个特征缩放到均值 0、方差 1（避免量纲差异影响线性模型），再对目标房价取对数，让分布更对称，训练完再取指数还原：

```python
from sklearn.preprocessing import StandardScaler

# 标准化：让每个特征都是「均值为 0、标准差为 1」，便于模型收敛
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit 学习均值方差，transform 应用缩放
X_val_scaled = scaler.transform(X_val)           # 验证集用同一套均值方差，不能重新 fit

# 关键招式：对目标取对数再训练，减弱高房价长尾的干扰
model = LinearRegression()
model.fit(X_train_scaled, np.log(y_train))

# 预测时先取指数，把预测值还原回「千美元」这个量纲
y_pred_val = np.exp(model.predict(X_val_scaled))
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
r2_val = r2_score(y_val, y_pred_val)
print(f"RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}")
```

对数变换是回归比赛里处理长尾目标的常用手段，能明显改善验证集分数。最后我们把提升空间释放到最大：同时训练六种模型——带正则化的线性模型 Ridge（岭回归）和 Lasso，以及四个树模型随机森林、梯度提升、XGBoost、LightGBM。让它们在同一个验证集上竞争，选 RMSE 最低的作为最终模型：

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 用字典集中定义超参数：alpha 控制正则强度，n_estimators 是树的数量
models = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.01, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbosity=0),
    'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1),
}

results = {}
best_model, best_rmse = None, float('inf')   # 初始化最佳模型为空，分数正无穷
for name, model in models.items():
    # 线性模型必须用标准化后的数据，树模型用原始数据即可，所以分开处理
    if name in ['Ridge', 'Lasso']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    results[name] = rmse
    print(f"{name}: RMSE = {rmse:.4f}")
    if rmse < best_rmse:                 # 记录当前最好的模型
        best_rmse, best_model = rmse, name

print(f"最佳模型: {best_model}, RMSE = {best_rmse:.4f}")
```

这里「超参数」是指模型里那些不是靠训练学出来、而是需要我们提前指定的数值，比如树的个数、学习率、正则化强度。我们按经验给每组模型选了合理的初始超参数，让它们在同一验证集上公平比较。这样既不盲目逐个微调，又能快速找到「这堆模型里谁最擅长抓房价规律」。我们还给线性回归做了 5 折交叉验证（把训练数据切成 5 份、轮流拿 4 份训练 1 份验证），得到更稳定的 RMSE 估计。

### 2.4 模型测试与指标报告

选出最佳模型之后，剩下的工作就是：在测试集上预测，写出符合提交模板的文件。我们提取最佳模型的预测结果，填进 `Id` 和 `MEDV` 两列：

```python
# 用最佳模型对测试集做最终预测（线性模型走标准化分支，其余走原始特征分支）
if best_model in ['Ridge', 'Lasso']:
    test_pred = results[best_model]['model'].predict(X_test_scaled)
else:
    test_pred = results[best_model]['model'].predict(X_test)

# 按提交模板生成文件：测试集的每个 Id 对应一个预测房价
submission = pd.DataFrame({'Id': test['Id'], 'MEDV': test_pred})
submission.to_csv('submission.csv', index=False)   # 不写行索引，符合竞赛格式
print(submission.head())
```

把 `submission.csv` 上传后，系统会按真实房价计算我们的分数。在本地我们则用验证集做最终汇报：把六种模型在验证集上的 RMSE 从低到高排成一张表，分数最低的就是最佳模型；同时交叉验证给出的 RMSE 可以说明这个分数是稳定的，不是运气好。通过这条「baseline → 对数变换 → 多模型竞争」的路线，我们的模型从只能拟合直线关系的线性回归，一路升级到能处理复杂非线性结构的树模型集成方案，提交文件里每一行的房价预测都来自验证集上的胜者模型。

## 三、总结

本文中我们完成了一场经典的波士顿房价预测回归比赛。我们首先读取并检查了训练集、测试集和提交样例的结构，确认了 13 个特征列和 1 个目标列 `medv`，通过描述统计和直方图发现了房价在 50 千美元处被截断的现象；其次我们做了数据清洗和特征工程，用 `add_features` 构造了六个组合特征，并删除了被截断的高价样本，再对目标取对数压缩长尾；接着我们训练了线性回归 baseline，用标准化加对数变换把它提升了一档，最后把岭回归、Lasso、随机森林、梯度提升、XGBoost 和 LightGBM 六种模型放进同一个验证集里比较超参数组合，选出 RMSE 最低的作为最佳模型；然后我们让最佳模型对测试集预测，生成符合模板的 `submission.csv`，并用 RMSE、MAE、R² 和 5 折交叉验证汇报了模型质量。本次比赛使用的页面包括线性回归 baseline 笔记本、加入特征工程与对数变换的改进方案，以及融合六种模型的高级集成方案。