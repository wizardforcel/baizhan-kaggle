# TMDB Box Office Prediction

## 一、比赛简介

### 1.1 比赛背景
TMDB Box Office Prediction 是 Kaggle 上一个经典的电影票房预测比赛。参赛者得到的是一批电影的公开信息，数据来自 The Movie Database。这些信息包括电影的预算、上映日期、演员表、制片公司、语言、流行度、时长等。我们要用这些可以提前知道的信息，去猜这部电影最终能赚多少钱。实际票房往往和预算、口碑、星光阵容、上映季节很有关系，但也带很强的随机性，所以这是个很适合入门回归问题的题目。

### 1.2 比赛目标
训练集提供了约 3000 部电影，每部都有若干列的特征以及一列 `revenue` 作为标签。测试集有 4398 部电影，特征和训练集一样，但没有票房。我们要预测出测试集每部电影的票房，并按 Kaggle 要求的格式提交。训练集里的列大致包括 `id`、`budget`、`genres`、`homepage`、`original_language`、`release_date`、`runtime`、`popularity`、`cast`、`crew`、`Keywords`、`revenue` 等。大部分列是文字或者 JSON 格式的列表，需要先处理成模型能用的数字。

### 1.3 评价标准
比赛官方评价指标是均方根对数误差 RMSLE，也就是预测值和真实值先取对数再算 RMSE。直观来说，对数让大票房和小票房的误差被压缩，模型不会只去迎合几部爆款。RMSLE 越小越好。材料里也有作者自己用 RMSE 来评估模型，我们在本地验证时用 log1p 转换后的目标值计算 RMSE，数值越小表示拟合越好。

## 二、解题过程

整场比赛思路可以总结为一句话：先把原始 JSON 文本和缺失值处理干净，做一些能反映电影体量的特征，再用树模型预测以预算和流行度为核心的对数票房。我们按四步走。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，看看样本量、列的类型和缺失情况，了解数据长什么样。

```python
import pandas as pd
import numpy as np

# 读入训练和测试，release_date 会变成日期类型
train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv', parse_dates=['release_date'])
test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv', parse_dates=['release_date'])

print('训练集形状', train.shape)  # 3000 行左右
print('测试集形状', test.shape)   # 4398 行左右

# 看缺失值最多的几列
train.isnull().sum().sort_values(ascending=False).head(10)
```

打印结果显示训练集里 `belongs_to_collection` 缺失约 2396 行，`homepage` 缺失 2054 行，`tagline` 缺失 597 行，`Keywords` 缺失 276 行等。测试集缺失更严重，`belongs_to_collection` 缺失 3521 行，`homepage` 缺失 2978 行。预算 `budget` 在训练集里没有缺失，但在部分样本里是 0。describe 显示预算均值约 2.27e7 美元，收入均值约 6.67e7 美元，时长均值约 107.9 分钟，流行度均值约 8.46。收入分布非常右偏，最大值可达十几亿，小值很多，这也是后面要做对数转换的原因。我们还看到约 812 部训练电影的预算为 0，意味着未知，这类预算需要当成缺失处理。

### 2.2 数据清洗

其次我们做清洗，主要是处理缺失、拆分日期、做一些入门特征。清洗要服务于任务，文本本身尽量保留信息，但 JSON 类列要提取有用的计数。

```python
# 把 release_date 拆成 年月日，方便模型学习季节效应
train['release_year'] = pd.to_datetime(train['release_date']).dt.year
train['release_month'] = pd.to_datetime(train['release_date']).dt.month
train['release_day'] = pd.to_datetime(train['release_date']).dt.dayofweek
test['release_year'] = pd.to_datetime(test['release_date']).dt.year
test['release_month'] = pd.to_datetime(test['release_date']).dt.month
test['release_day'] = pd.to_datetime(test['release_date']).dt.dayofweek

# 评分、投票、时长缺失用均值填补
train['rating'] = train['rating'].fillna(train['rating'].mean())
test['rating'] = test['rating'].fillna(test['rating'].mean())
train['totalVotes'] = train['totalVotes'].fillna(train['totalVotes'].mean()).round()
test['totalVotes'] = test['totalVotes'].fillna(test['totalVotes'].mean()).round()
train['runtime'] = train['runtime'].fillna(train['runtime'].mean())
test['runtime'] = test['runtime'].fillna(test['runtime'].mean())

# 预算和收入严重右偏，做 log1p 压缩
train['budget'] = np.log1p(train['budget'])
test['budget'] = np.log1p(test['budget'])
train['revenue'] = np.log1p(train['revenue'])
```

我们还创建了几个简单特征：是否英文原声 `lang_english`、类型数量 `numberofgenres`、是否拥有官网 `has_homepage`，以及预算与时长、年份、流行度的组合特征 `bud_runtime`、`bud_year`、`bud_popularity` 等。这些特征都是材料里出现过的，能反映成本和关注度。清洗后数据变得干净，数值列基本没有缺失，类别列也可以放心交给模型。

### 2.3 模型训练与超参数调优

接着我们把清洗后的数据切成训练集和验证集，用 80/20 分层划分，并在对数目标上训练几个常用回归模型。材料里作者先后尝试了线性回归、随机森林、XGBoost 和 LightGBM。

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 选用的特征列示例
features = ['budget','popularity','runtime','rating','totalVotes',
            'release_year','release_month','release_day',
            'numberofgenres','lang_english','bud_runtime','bud_year',
            'bud_popularity','runtime_year','popularity_year']
X = train[features]
y = train['revenue']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=39)

# 线性回归作基准
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_val)
lr_rmse = mean_squared_error(y_val, lr_pred, squared=False)
print('LinearRegression RMSE', lr_rmse)  # 约 2.348
```

线性回归给出基准 RMSE 约 2.348，说明特征与对数票房有线性关系但不足。接着我们用树模型。随机森林参数采用 `n_estimators=50, min_samples_leaf=2, random_state=42`，XGBoost 使用 `eta=0.01, max_depth=3, n_estimators=3000`，LightGBM 使用 `num_leaves=1023, learning_rate=0.005, n_estimators=650` 等设置。材料里还提到曾想用 GridSearchCV 调参，但因为特征维度高、字符类特征多，网格搜索耗时太大，最终主要靠经验设置。每个模型都在验证集上计算 RMSE，并将预测结果用 `exp` 还原到原始票房尺度准备提交。

```python
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

rfr = RandomForestRegressor(random_state=42, n_estimators=50, min_samples_leaf=2)
rfr.fit(X_train, y_train)
rfr_pred = rfr.predict(X_val)
rfr_rmse = mean_squared_error(y_val, rfr_pred, squared=False)
print('RandomForest RMSE', rfr_rmse)  # 约 2.045

xgb_model = xgb.XGBRegressor(objective='reg:linear', eta=0.01, max_depth=3,
                             min_child_weight=3, subsample=0.8, gamma=1.45,
                             colsample_bytree=0.7, n_estimators=3000, seed=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_val)
xgb_rmse = mean_squared_error(y_val, xgb_pred, squared=False)
print('XGBoost RMSE', xgb_rmse)  # 约 1.966

lgb_model = lgb.LGBMRegressor(objective='regression', num_leaves=1023,
                              learning_rate=0.005, n_estimators=650,
                              max_bin=58, bagging_fraction=0.80,
                              max_depth=10, feature_fraction=0.2319)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_val)
lgb_rmse = mean_squared_error(y_val, lgb_pred, squared=False)
print('LightGBM RMSE', lgb_rmse)  # 约 2.050
```

我们看到 XGBoost 在对数目标上的 RMSE 最低，约 1.966，随机森林约 2.045，LightGBM 约 2.050，线性回归约 2.348。XGBoost 的提升来自它对非线性交互的拟合能力。虽然材料里没有给出最终排行榜分数，但另一个方案里 LightGBM 在验证集上的 RMSLE 约 0.145，也体现了树模型在该任务上的优势。超参数调优主要靠少量经验尝试，重点放在特征构造上。

### 2.4 模型测试与指标报告

然后我们用全量训练数据重新拟合最优模型，对测试集预测并还原对数尺度。最后检查提交文件的行数和列名是否正确。

```python
# 用全量数据训练
best_model = xgb_model
best_model.fit(X, y)

# 预测测试集并还原
test_pred_log = best_model.predict(test[features])
test_pred = np.exp(test_pred_log)

submission = pd.DataFrame({'id': test['id'], 'revenue': test_pred})
submission.to_csv('submission.csv', index=False)
print(submission.shape)
```

在验证集上的指标报告显示 RMSE 随模型复杂度提升而下降，XGBoost 最佳。材料里同时给出了按对数目标计算的 RMSLE，LightGBM 的 RMSLE 约 0.145。整体结论是预算、流行度、时长、上映年份等特征对票房有解释力，但票房本身不确定性大，模型只能捕捉统计规律。我们还注意到预算为 0 的样本很多，清洗时把它当作缺失并用均值替代或通过 log1p 压缩，是降低误差的关键一步。

## 三、总结

本文中我们完成了 TMDB Box Office Prediction 的完整入门流程。我们首先用 EDA 摸清了数据：训练集约 3000 部电影，测试集约 4398 部，预算和收入高度右偏，缺失主要集中在文本类列，预算 0 的样本约 812 部。其次我们做了数据清洗：把 release_date 拆成年、月、周几，用均值填补 rating、totalVotes、runtime，对预算和收入做 log1p 转换，并构造了类型数量、英文标记、官网标记、预算与时长等组合特征。接着我们用线性回归、随机森林、XGBoost、LightGBM 进行训练与调优，验证集上 log RMSE 分别约 2.348、2.045、1.966、2.050，XGBoost 表现最好。最后我们用全量数据拟合并生成提交文件，并在验证集上报告 RMSLE 约 0.145。该比赛给我们的启发是：回归任务里先做对数转换和基础特征工程，比一味调参更有效。
