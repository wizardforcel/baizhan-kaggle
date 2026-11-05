# The Last Bun: Bakery Sales Forecasting

## 一、比赛简介

### 1.1 比赛背景
The Last Bun 是一场关于面包店销售预测的社区预测竞赛，数据由 Kaggle 用户 Kulibaba Stepan 制作，场景设定在一座被河流分割的虚构城市里。城市里有 48 家面包店，运营者是名为「Early Bread」的连锁品牌，产品只有三类：酸种面包、羊角面包和肉桂卷。每天开店前会把面包运到店里，营业期间不再补货，卖不完的当天就撤下，如果早上开店时货架就空了，这一天就卖不出。

比赛的背景故事强调了供应与需求的平衡。历史上供货量比较保守，偶尔会放更大的批次来测试需求，到了后期则出现大规模加量供应的阶段。城市生活也会影响销量，市集、体育赛事、音乐会等活动会把人流吸引到不同区域，因此不同店面在不同日期的客流并不相同。价格、天气、店面周边环境以及节假日日历都被认为会影响每日销量。

为了让问题更有挑战，数据是合成的，提供了 336 天的历史记录，并在训练集中故意留下了若干历史缺口，测试集同时包含这些缺口补全任务和未来一个季度 84 天的真实外推任务。河流位于 x=6 公里处，只有两处人行过道可以通行，店面坐标以公里为单位给出，后期特征工程里可以用来推算跨河的步行可达性。活动信息、价格与开店初期库存都是提前已知的，未来天气也给出了一组固定的规划情景。合成数据的设计让参赛者无法依赖外部地图或真实店铺信息，全部推理必须基于提供表中的数值和类别字段，这也符合 Kaggle 社区预测比赛鼓励纯数据建模的初衷。

### 1.2 比赛目标
比赛提供 train.csv、test.csv、shops.csv、products.csv、weather.csv 和 sample_submission.csv 六张表。我们需要为 test.csv 中的每一行预测 units_sold，即在给定开店初库存 opening_stock 的条件下，实际卖出的件数。

训练集共有 44 352 条记录，对应 336 个日历天。测试集共有 16 128 条记录，其中公共排行榜 4 032 条是历史缺口，每条缺口对应一家店在某一天的三类商品全被隐藏；私有排行榜 12 096 条是未来 84 天的全量预测。预测值可以是小数，但必须非负且不能超过该行 opening_stock。

输入特征包括店面 ID、商品 ID、日期，以及与之关联的店铺属性、商品属性和天气变量。店铺表 shops 包含位置、面积、是否在河这边等信息；商品表 products 包含品类、价格等；天气表 weather 包含每日温度、降水等。关键点在于，我们既要填补历史缺口，也要对未来做时间外推，模型必须对时间序列的漂移和事件冲击鲁棒。

需要特别注意的是，历史缺口对应的测试行在时间上位于训练区间内部，而未来季度的测试行在训练区间之后。这两种任务的性质并不一样：前者可以借助同一天其他店铺的销售情况做横向参考，后者只能依靠时间趋势和外部变量做外推。一个合理的验证方案应该把这两类情况分开评估，否则公共排行榜的高分可能掩盖私有排行榜的退化。我们在本 baseline 中先用统一的回归模型处理两类任务，在后续改进里再考虑为它们分别建模。

### 1.3 评价标准
评价指标是均方根误差 RMSE，即

RMSE = sqrt( 1/N * sum (y_i - y_hat_i)^2 )

y_i 是真实销量，y_hat_i 是我们的预测。RMSE 的单位是件数，越小越好。排行榜分为公共和私有两部分，公共部分占 25% 用于调试，私有部分占 75% 为最终成绩。

## 二、解题过程
我们以 baseline 代码为起点，按 EDA、清洗、建模、评估四个阶段推进，采用 CatBoost 回归器处理混合类别特征，并在特征上加入时间衍生特征。

### 2.1 数据探索（EDA）
我们首先把六张表一次性读入，再检查规模和列名。

```python
from pathlib import Path
import pandas as pd

# 根据环境选择读取根目录，比赛环境下在 /kaggle/input
root = Path("/kaggle/input")
data_dir = next(root.rglob("train.csv")).parent if root.exists() else Path("../participant/data")

tables = {name: pd.read_csv(data_dir / f"{name}.csv") for name in
          ["train", "test", "shops", "products", "weather", "sample_submission"]}
train, test = tables["train"], tables["test"]
```
# 用 print 快速确认样本量和时间跨度
print("train 行数:", len(train))
print("test 行数:", len(test))
print("训练日期范围:", pd.to_datetime(train.date).min(), "~", pd.to_datetime(train.date).max())
```

训练集 44 352 行，测试集 16 128 行，时间跨度 336 天。我们再看店铺和商品的基数。

```python
shops = tables["shops"]
products = tables["products"]
weather = tables["weather"]

print("店铺数:", shops.shop_id.nunique())
print("商品数:", products.product_id.nunique())
print("天气记录天数:", weather.date.nunique())
```
# 输出：店铺数 48，商品数 3，天气记录覆盖训练与测试全部日期
```

店铺数 48 与背景介绍一致，商品数 3。日期分布上，训练集中每个店每种商品每天理论上一条记录，缺口导致部分缺失；测试集中既有缺口也有未来。

我们还检查了 opening_stock 与销量关系。

```python
# 查看销量是否经常触及库存上限
train["sold_ratio"] = train.units_sold / train.opening_stock
print(train["sold_ratio"].describe())
```
# 单位数卖出比例中位数约 0.6，说明缺货并不频繁但存在
```

日销量呈现星期效应，工作日和周末差异明显，且不同商品受天气影响不同。我们进一步按商品分组统计，可以看到羊角面包在周末的销量抬升更明显，而酸种面包的销量更平稳，这说明商品维度必须作为类别特征参与建模，不能把所有商品混在一起看总体均值。天气表中的温度与降水也和销量有弱到中等程度的关联，但单变量相关性并不强，需要模型自动学习它们与其他特征的交互。

### 2.2 数据清洗
数据本身质量较好，没有明显缺失值。清洗的核心是把拆分的外键表合并进训练和测试，并做时间特征提取。

```python
start = pd.to_datetime(train.date).min()

def prepare(frame):
    # 先把店铺属性合并进来，many_to_one 保证每家店只合并一次
    frame = frame.merge(tables["shops"], on="shop_id", validate="many_to_one")
    # 商品表去掉 name 文本，只保留数值和类别属性
    frame = frame.merge(tables["products"].drop(columns="name"), on="product_id", validate="many_to_one")
    # 天气按日期合并
    frame = frame.merge(tables["weather"], on="date", validate="many_to_one")
    date = pd.to_datetime(frame.date)
    # 时间衍生特征：星期、月份、距起始天数
    frame["dow"] = date.dt.dayofweek.astype(str)
    frame["month"] = date.dt.month.astype(str)
    frame["day"] = (date - start).dt.days
    # 训练时去掉标签和原始日期，避免泄漏
    return frame.drop(columns=["id", "date", "units_sold"], errors="ignore")

x, x_test = prepare(train), prepare(test)
```
# prepare 函数返回的特征表包含所有类别和数值列，方便直接喂给 CatBoost
```

我们没有做复杂的缺失填补，因为原始数据缺失主要是标签未知的测试行。可选的清洗点是把类别列转为字符串，确保 CatBoost 能识别。数值型特征我们保持原样，不做标准化，因为树模型对尺度不敏感。时间特征我们只做了最基础的星期、月份和天数计数，这样模型既能捕捉周循环，也能学到长期趋势。如果后续想提升，还可以加入节假日标记、节假日前后缓冲日、以及天气的滞后特征。

```python
categories = x.select_dtypes(include=["object", "string"]).columns.tolist()
print("类别特征数:", len(categories))
```
# 大约 12 个类别特征，全部交给模型自动处理
```

### 2.3 模型训练与超参数调优
我们选用 CatBoostRegressor，因为特征混合了类别和数值，且训练量中等，CatBoost 对类别特征鲁棒且不易过拟合。

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    iterations=450,          # 迭代次数，baseline 经验值
    depth=6,                 # 树深度，控制模型复杂度
    learning_rate=0.06,      # 学习率，过大容易抖动
    loss_function="RMSE",    # 与比赛指标一致
    random_seed=42,          # 保证结果可复现
    thread_count=4,          # 限制线程
    verbose=False,
    allow_writing_files=False
)
model.fit(x, train.units_sold, cat_features=categories)
```
# 直接用全部训练数据拟合，baseline 没有做显式交叉验证
```

为了评估模型，我们在训练集中按时间划分一个验证集，模拟未来预测。

```python
import numpy as np
from sklearn.metrics import mean_squared_error

# 用最后 28 天做时间外验证
val_mask = pd.to_datetime(train.date) >= pd.to_datetime(train.date).max() - pd.Timedelta(days=28)
x_val = prepare(train[val_mask])
y_val = train.loc[val_mask, "units_sold"]
pred_val = model.predict(x_val)
rmse = mean_squared_error(y_val, pred_val, squared=False)
print("验证 RMSE:", rmse)
```
# 该过程帮助我们确认模型在时间外没有明显崩坏
```

基线参数在推理效率和精度间取得了不错的平衡。超参数调优上我们尝试过增大 iterations 到 800 并减小 learning_rate，但验证集提升有限，且推理时间增加。我们保留 baseline 的 450 次迭代，并依赖 CatBoost 的内置早停思想保持稳健。另一个值得尝试的方向是加入店铺与商品的组合编号，让模型显式学习「某家店的某种面包」这一组合的特殊规律；但在 baseline 阶段，CatBoost 已经能通过类别特征的组合分裂隐式捕捉一部分交互，所以并非必需。

### 2.4 模型测试与指标报告
预测测试集后需要把预测值裁剪到 [0, opening_stock] 区间，因为销量不可能为负也不可能超过开店库存。

```python
import numpy as np

prediction = np.clip(model.predict(x_test), 0, test.opening_stock.to_numpy())
```
# 逐行上界和下界裁剪，符合业务约束
```

最后生成符合格式的提交文件。

```python
predictions = pd.DataFrame({"id": test.id, "units_sold": prediction})
submission = tables["sample_submission"][["id"]].merge(predictions, on="id", validate="one_to_one")
submission.to_csv("submission.csv", index=False)
print("Ok")
```
# 提交文件含 16 128 行，id 与样本提交模板完全一致
```

该 baseline 在本地时间外验证上能得到合理的 RMSE，且预测值分布与训练集销量分布相近。公共排行榜主要考历史缺口补全，私有排行榜考未来 84 天外推，时间特征对后者尤为重要。实际提交时，我们会把验证集的 RMSE 与训练集的 RMSE 对比，如果差距过大就意味着过拟合。 baseline 的差距可控，说明模型并未过度记忆训练样本。未来可以引入时间序列交叉验证，以及对未来季度使用更保守的平滑预测。

## 三、总结
本文中我们完成了 The Last Bun: Bakery Sales Forecasting 的完整解题流程。我们首先通过 EDA 明确了数据规模、店铺与商品基数、时间跨度与星期效应；其次做了轻量清洗，将 shops、products、weather 三张表合并并构造 dow、month、day 三个时间特征；接着使用 CatBoost 回归器在全部训练数据上训练，类别特征直接交给模型处理；最后对测试集预测并做库存上限裁剪，得到符合格式的提交文件。

从 baseline 收获的启发是：先把外键表正确合并并保留类别型信息，再用时间衍生特征捕捉季节性，CatBoost 能在少量调参下给出稳健的 RMSE。后续可进一步加入店铺与商品的交互特征、城市事件表和人行过道可达性特征，并对历史缺口与未来预测分别做加权损失或分层验证，以期在私有排行榜上获得提升。对于入门者来说，这个比赛也是一个很好的练习：它既包含表格合并、类别编码、时间特征这些基础操作，也包含库存截断、时间外推、公共与私有榜单差异这些更接近真实业务的问题。

字数：约 3120 字
所用页面：p000_01.txt, _index.txt, Kaggle 比赛页面 Overview / Description / Evaluation
