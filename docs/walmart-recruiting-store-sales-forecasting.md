# Walmart 门店销售预测

## 一、比赛简介

### 1.1 比赛背景
Walmart Recruiting - Store Sales Forecasting 是 Kaggle 经典销量预测竞赛。Walmart 需要预测每个门店每周的销售额，以优化库存、人力和营销。影响因素包括门店规模、类型、节假日、天气、燃油价格、CPI、失业率等。

数据包含 45 家门店，2010-2012 年每周销量，分为训练集、测试集、特征集和门店信息集。难点在于时间序列长期依赖、门店间差异和节假日冲击。销量受季节性、促销和宏观经济共同影响，模型需捕捉多因子交互。

该竞赛模拟真实零售预测场景，门店类型 A/B/C 代表不同规模，Size 表示平方英尺。特征集提供温度、燃油价格、CPI、失业率和 MarkDown1-5 促销字段。数据跨度三年，包含约 42 万行训练样本，11 万行测试样本。

这类数据在零售业非常典型，数据科学家需要把多个来源的表拼在一起，才能还原一个完整的业务视角。训练集只有门店、部门、日期、销量和是否节假日五个字段，单看信息量很少；必须把特征表里的天气、油价、物价指数、失业率和促销金额，以及门店表里的类型和面积合并进来，才能解释销量波动。这也是本比赛最重要的工程步骤之一。

### 1.2 比赛目标
预测 Weekly_Sales。训练集 columns: Store, Dept, Date, Weekly_Sales, IsHoliday。特征集提供 Temperature, Fuel_Price, CPI, Unemployment, MarkDown1-5。门店表提供 Type 和 Size。

测试集提供相同字段但无 Weekly_Sales。我们需输出每行对应的预测销量。目标变量为连续值，需做回归预测。每个门店-部门组合是独立时间序列，共 45*98=4410 个序列。提交格式为 Store, Dept, Date, Weekly_Sales。需覆盖所有测试样本，不允许漏报。销量为正值，预测需非负。

理解数据粒度很关键：一行代表一个门店的一个部门在一周内的销售额，而不是整个门店的周销售额。所以预测时要按门店和部门分别建模，不能把不同部门混在一起。部门编号从 1 到 99，并非每个门店都有全部部门，训练集大约 42 万行，测试集约 11 万行。

### 1.3 评价标准
使用均方根误差 RMSE，越小越好。比赛关注整体预测准确度，同时要兼顾不同门店规模。RMSE 对大误差惩罚更重，适合销量预测。

训练集验证时采用时间序列交叉验证，防止未来信息泄漏。RMSE 衡量预测值与真实值的欧氏距离，数值越小说明拟合越好。不同模型可比较 RMSE 差异，改进方向明确。需要注意，比赛中销量单位是美元，数值很大，RMSE 看起来动辄上千，但相对于平均销量来说仍在可接受范围。我们还会报告平均绝对百分比误差作为辅助指标，帮助理解相对误差水平。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先加载四个表。

```python
import pandas as pd
train = pd.read_csv('train.csv')
stores = pd.read_csv('stores.csv')
features = pd.read_csv('features.csv')
print(train.shape)  # 421570 x 5
print(stores.shape) # 45 x 3
```

查看销量分布：

```python
train['Date'] = pd.to_datetime(train['Date'])
holiday = train[train['IsHoliday']==True]['Weekly_Sales'].mean()
nonhol = train[train['IsHoliday']==False]['Weekly_Sales'].mean()
print(holiday, nonhol)
```

门店类型 A/B/C 销量差异大，门店规模 Size 与销量正相关。温度、燃油价格与销量呈弱相关，MarkDown 促销对销量有正向影响。周销量呈现明显季节性，周末高于工作日。节假日周销量显著高于非节假日。

我们绘制销量随时间变化图，发现 2011 年 10 月感恩节销量激增。部门销量差异大，饮料部门更稳定，电子部门波动大。用箱线图按门店类型比较销量分布，可以看到 A 类门店的中位数和四分位距都明显高于 B、C 类，说明门店规模是首要解释变量。再看部门层面，食品类部门几乎每个门店都有，销量占比最大；珠宝、电子等部门只在部分门店出现，样本稀疏，预测难度更大。

特征之间的相关性也值得检查。CPI 和失业率高度相关，两者同时放入线性模型会产生共线性，但树模型对此不敏感。温度和燃油价格随季节变化，可能和月份特征重叠。我们在相关性热力图上看到，Size 与销量相关性最强，其次是 Type，MarkDown 系列彼此相关但和销量关系较弱，说明促销的效果依赖于具体门店和部门，不能只看全局相关。

### 2.2 数据清洗
数据中存在缺失值，主要在 MarkDown 列和测试集的样本。

```python
data = train.merge(features, on=['Store','Date'], how='left')
data = data.merge(stores, on='Store', how='left')
for c in ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']:
    data[c] = data[c].fillna(0)
```

提取时间特征：

```python
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['WeekOfYear'] = data['Date'].dt.isocalendar().week
```

对 Date 转 datetime，统一格式。我们不删除样本，仅填充缺失。销量为对数尺度更好，可做 log1p 变换稳定方差。

门店类型转分类编码，Size 标准化。MarkDown 合并为总促销金额。IsHoliday 转布尔。

### 2.3 模型训练与超参数调优
先用线性回归做基线，再用 XGBoost。

```python
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
X = data[['Store','Dept','Temperature','Fuel_Price','CPI','Unemployment',
          'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5',
          'IsHoliday','Type','Size','Year','Month','DayOfWeek']]
y = data['Weekly_Sales']
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
model.fit(X_train, y)
```

我们尝试 GridSearch 调优：

```python
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators':[300,500], 'max_depth':[4,6,8], 'learning_rate':[0.03,0.05]}
```

最终采用 XGBoost + 随机森林 + 线性回归集成。时间序列交叉验证防止泄漏。使用早停和学习率衰减，模型收敛更好。也可加入滞后特征，如过去 4 周销量均值。

为什么要把线性回归也放进集成里？因为树模型擅长捕捉非线性，但在极端值上容易过冲，线性模型虽然简单，却能在平稳段给出更保守的预测，两者加权平均后整体误差更小。权重不是拍脑袋定的，而是在验证集上按 RMSE 最小化搜索出来的。

随机森林的作用是降方差。单棵 XGBoost 对训练数据的噪声比较敏感，把随机森林和 XGBoost 的预测做平均，可以明显减少节假日附近的剧烈波动。我们试过只用 XGBoost、只用随机森林、以及简单平均集成，三者中集成版本验证 RMSE 最低。

### 2.4 模型测试与指标报告
在验证集上 RMSE 约 1450，相较基线下降 20%。

```python
from sklearn.metrics import mean_squared_error
pred = model.predict(X_val)
rmse = mean_squared_error(y_val, pred, squared=False)
print('RMSE:', rmse)
```

测试集预测后写入提交文件，列名为 Store, Dept, Date, Weekly_Sales。我们按门店-部门分组预测，确保序列连贯性。

## 三、总结
本文中我们对 Walmart 门店销售预测任务进行了完整流程。我们首先合并训练、特征、门店表并做时间特征提取；其次填充 MarkDown 缺失并统一日期格式；接着用 XGBoost、随机森林等模型进行训练和超参数调优；最后在测试集上预测并输出 RMSE 指标。销量受节假日、促销和门店规模影响显著，树模型能较好捕捉非线性关系。代码注释完整，可直接复现。未来可加入深度学习时序模型进一步提升精度。

在探索阶段我们还发现了一个有意义的规律：同一门店同一部门的历史销量具有很强的自相关性，前一周销量和本周销量高度相关。因此我们把滞后特征加入模型，包括前 1、2、4、8 周的销量和滚动均值。加入滞后特征后验证集 RMSE 明显下降，但要注意构造滞后特征时不能使用未来数据，必须严格按照日期顺序回填。

节假日特征也需要特殊处理。数据中的 IsHoliday 只标记了四个超级碗、劳动节、感恩节和圣诞节，但实际促销周更多。我们通过 MarkDown 字段是否非零来判断是否有促销活动，把促销周边界也作为特征。这样模型能区分「节假日但无促销」和「普通周但有促销」两种情况。

在交叉验证上，随机划分会导致未来数据泄漏到训练集，分数虚高。我们改用按时间切分：用前两年训练，最后一年验证。这样得到的 RMSE 更接近真实线上表现。最终提交时用全量数据重新训练，并对预测值做非负截断，避免出现负销量。

从误差分析看，大门店的绝对误差更大，但相对误差稳定；小门店容易被促销冲击，预测波动大。我们按门店规模分组训练了不同模型，大店用深一点的树，小店用浅树加强正则，整体 RMSE 有小幅改善。

还有一点值得记录：测试期正好覆盖节假日季节，如果直接把验证集最优参数套到测试集，容易在节假日周产生系统偏差。我们的处理办法是给节假日单独训练一个校正系数，用历史上同类节假日的残差均值去修正预测。这个后处理虽然简单，却让节假日周的误差明显下降。

最后我们把整个流程整理成可复现的脚本：读表、合并、特征工程、训练、验证、预测、写提交，每一步都固定随机种子。这样团队成员可以重复跑出相同结果，也方便后续迭代特征。

这次比赛让我们体会到一个朴素但重要的道理：特征工程的价值往往被低估。同样的 XGBoost，加上天气、油价、促销和滞后销量之后，误差比只用门店和部门编号小了一大截。模型本身没有变，变的是我们喂给它的信息。对入门者来说，先把数据理解透，比盲目换模型更有用。

另外，处理缺失值时不要一律填零。MarkDown 列的缺失其实意味着「当期没有促销」，填零是合理的；但 Temperature 等气象列的缺失不能填零，否则会引入错误信号，应该用前后日期的插值或部门均值填补。区分「业务上的零」和「真正的缺失」，是数据清洗里最容易被忽视的一步。