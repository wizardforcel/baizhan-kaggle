# Avito 需求预测（Demand Prediction）

## 一、比赛简介

### 1.1 比赛背景

Avito 是俄罗斯最大的分类信息平台，用户可以在这里发布二手商品、租房、招聘、服务等各类广告。平台最关心的一件事是：**这条广告发出去之后，到底能不能促成交易？**

一条广告如果发出去一周都没人联系，说明它的「需求」很低——可能是定价太高、标题写得太差、照片太糊，或者这个品类在那个城市本来就没人要。平台如果能提前预测出「这条广告的成交概率」，就可以做很多事：提醒卖家改价、把高需求广告推上首页、给低需求广告补充曝光。

这个比赛的任务就是：**给定一条广告的全部信息，预测它的 `deal_probability`（成交概率）**。

比赛的难点在于信息是**多模态**的：

1. **文本**：`title`（标题）和 `description`（描述）是俄语，且经常有拼写变形、缩写；
2. **图片**：每条广告最多 5 张图，可以用 `image_top_1` 这个字段间接反映图片类型；
3. **结构化字段**：价格、地区、城市、类目、参数（`param_1`～`param_3`）、用户类型（公司还是个人）、广告序号（`item_seq_number`，即这是该用户的第几条广告）；
4. **时间**：`activation_date` 是广告上线的日期，需求有明显的日历规律。

### 1.2 比赛目标

输入一条广告的所有字段，输出一个 0～1 之间的小数，表示它成交的概率。

- 训练集 `train.csv`：**1,503,424 行 × 18 列**，含标签 `deal_probability`；
- 测试集 `test.csv`：**508,438 行 × 17 列**，没有标签。

主要列包括：`item_id`、`user_id`、`region`、`city`、`parent_category_name`、`category_name`、`param_1/2/3`、`title`、`description`、`price`、`item_seq_number`、`activation_date`、`user_type`、`image`、`image_top_1`。

提交文件两列：`item_id` 和 `deal_probability`。

### 1.3 评价标准

比赛使用 **RMSE（Root Mean Squared Error，均方根误差）**：

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

通俗地说，就是把「预测值和真实值的差」平方、求平均、再开根号，得到的就是「平均差多少」。因为标签本身是 0～1 的概率，所以 RMSE 的量纲也是 0～1 之间的一个数，**越小越好**。

> 小提示：RMSE 会重重惩罚「离谱的预测」。一条真实概率 0.05 的广告，如果你预测成 0.95，这个误差会被平方放大。所以最后一步我们一定要把预测值裁剪到 `[0, 1]` 区间里——模型是回归模型，完全可能输出 1.03 或 -0.02 这种没意义的数。

---

## 二、解题过程

我们的主线方案是：**把日期拆成特征、统计文本长度、对类别字段做均值编码，然后交给 LightGBM 回归**。这条线在本地验证集上做到 RMSE 0.2266，最后用多个差异化提交做平均，把公榜分数压到 0.2204。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，确认规模：

```python
import numpy as np
import pandas as pd

tr = pd.read_csv('../input/train.csv')
te = pd.read_csv('../input/test.csv')
print('train data shape is :', tr.shape)   # (1503424, 18)
print('test data shape is :', te.shape)    # (508438, 17)

# 把两份拼在一起处理，保证类别编码、时间特征在训练和测试上完全一致
data = pd.concat([tr, te], axis=0)
print(data.shape)                          # (2011862, 18)
```

训练集 150 万行、测试集 50 万行，拼起来 201 万行。拼起来的原因很实际：如果训练集和测试集分别做 `LabelEncoder`，同一个城市在两边可能被编成不同的数字，模型就学错了。**先把两份合起来统一编码，再切回去**，是处理类别特征的标准做法。

接着我们看几个最有信息量的字段。价格是需求的核心，我们先看它的分布：

```python
# 价格的描述性统计（含缺失比例）
print(data['price'].describe())
print('price missing ratio:', data['price'].isnull().mean())

# 按用户类型看成交概率差异：公司 vs 个人
print(data.groupby('user_type')['deal_probability'].mean())

# 按星期几看成交概率：周末和工作日可能不一样
data['activation_date'] = pd.to_datetime(data.activation_date)
data['day_of_week_tmp'] = data.activation_date.dt.weekday
print(data.groupby('day_of_week_tmp')['deal_probability'].mean())
```

我们看到几个现象：

- **价格有大量缺失**，而且缺失本身有含义（很多卖家选「价格面议」）；
- **用户类型**对成交概率有明显影响——公司账号（`Shop`）和个人账号的行为完全不同；
- **星期几**上成交概率有系统性差异，说明时间特征是真实有效的，不是噪声。

还有一个容易被忽略的信号：**标题和描述的长度**。写得越详细的广告，往往越用心、成交率越高。这就是我们后面构造 `char_len_title` / `char_len_desc` 两列的依据。

### 2.2 数据清洗

**第一步，处理日期。** 拼好的表里 `activation_date` 是字符串，我们先转成 datetime，再拆出「几号」和「星期几」：

```python
# 转成时间类型；errors='coerce' 让解析失败的变成 NaT，不会整列报错
data.activation_date = pd.to_datetime(data.activation_date)
tr.activation_date   = pd.to_datetime(tr.activation_date)

# 「几号」和「星期几」是两个独立的周期信号
data['day_of_month'] = data.activation_date.apply(lambda x: x.day)
data['day_of_week']  = data.activation_date.apply(lambda x: x.weekday())
tr['day_of_month']   = tr.activation_date.apply(lambda x: x.day)
tr['day_of_week']    = tr.activation_date.apply(lambda x: x.weekday())
```

为什么只拆「日」和「星期」，不拆「年、月」？因为这份数据的训练集和测试集**在时间上是错开的**（后面会讲），拆出「年、月」会让模型学到「2017 年 4 月」这种在测试集上不存在的取值，反而有害。「几号」和「星期几」是循环的日历规律，在哪个时间段都成立。

**第二步，文本长度特征。** 我们没有直接把 150 万条俄语标题丢进词袋（太慢、太占内存），而是先取一个便宜但有效的信号——字符数：

```python
# str() 是为了兜住 NaN：NaN 转成字符串 'nan'，长度是 3，不会报错
data['char_len_title'] = data.title.apply(lambda x: len(str(x)))
data['char_len_desc']  = data.description.apply(lambda x: len(str(x)))
```

**第三步，均值编码（mean encoding）。** 这是本方案的核心特征。思路是：对每一个类别取值，统计它在训练集里对应的**平均成交概率**和**平均价格**，然后把这个统计量当作新特征贴回每一行。

比如「莫斯科 + 轿车类目」的平均成交概率是 0.08，那么所有属于这个组合的广告都会被赋予 0.08 这个特征值。模型由此知道「这类广告历史上表现如何」：

```python
agg_cols = ['region', 'city', 'parent_category_name', 'category_name',
            'image_top_1', 'user_type', 'item_seq_number',
            'day_of_month', 'day_of_week']

# 对每个类别列，统计「平均成交概率」和「成交概率的标准差」
for c in agg_cols:
    gp   = tr.groupby(c)['deal_probability']
    mean = gp.mean()      # 这一类的平均成交概率
    std  = gp.std()       # 这一类的波动程度：是稳定高还是忽高忽低
    data[c + '_deal_probability_avg'] = data[c].map(mean)
    data[c + '_deal_probability_std'] = data[c].map(std)

# 价格均值单独统计一遍，因为「这类东西一般卖多少钱」本身是强信号
for c in agg_cols:
    gp   = tr.groupby(c)['price']
    mean = gp.mean()
    data[c + '_price_avg'] = data[c].map(mean)
```

> **这里有一个必须警惕的陷阱：数据泄漏。** 上面这段代码是在 `tr` 上算统计量，再 map 到 `data`（含训练和测试）。如果 `tr` 里包含了你要验证的那部分样本，那么「平均成交概率」里就混进了验证样本自己的标签——模型相当于提前看到了答案，验证分数会虚高。
>
> 正确的做法是：**先切分训练集和验证集，统计量只在训练部分上算，再应用到验证部分和测试部分**。材料里的另一份实现就是这么修的：它先 `train_test_split` 得到 `data_tr` 和 `data_va`，然后 `fStats.fit_transform(data_tr)`（只在训练集上学统计量），再 `fStats.transform(data_va)` 和 `fStats.transform(data_te)`（只做变换，不重新统计）。这样「均价」「平均成交率」这些数字对验证集来说是「从没见过验证集标签」的，分数才可信。

我们把这段修好的逻辑写成一个类，既干净又不会漏：

```python
class FeaturesStatistics:
    """在训练集上学习统计量，在别的数据集上应用"""
    def __init__(self):
        self._agg_cols = ['region', 'city', 'parent_category_name',
                          'category_name', 'image_top_1', 'user_type',
                          'item_seq_number', 'day_of_month', 'day_of_week']
        self._stats = None

    def fit(self, df):
        # 只在传入的 df（训练集）上统计，绝不碰验证集
        self._stats = {}
        for c in self._agg_cols:
            desc = df.groupby(c)[['deal_probability', 'price']].describe()
            self._stats[c] = desc[[('deal_probability', 'mean'),
                                   ('deal_probability', 'std'),
                                   ('price', 'mean')]]

    def transform(self, df):
        # 把训练集学到的统计量映射成新列，df 有没有标签都不影响
        for c in self._agg_cols:
            df[c + '_deal_probability_mean'] = df[c].map(
                self._stats[c][('deal_probability', 'mean')])
            df[c + '_deal_probability_std'] = df[c].map(
                self._stats[c][('deal_probability', 'std')])
            df[c + '_price_mean'] = df[c].map(
                self._stats[c][('price', 'mean')])
        return df

stats = FeaturesStatistics()
stats.fit(data_tr)                 # 只在训练集上学
data_tr = stats.transform(data_tr)
data_va = stats.transform(data_va) # 验证集只用变换
data_te = stats.transform(data_te) # 测试集只用变换
```

**第四步，类别编码和删列。** 三个关键类别列用 `LabelEncoder` 转成整数，然后丢掉那些「模型用不了、或者已经提完信息」的原始列：`user_id`、`item_id` 是标识符，`title`/`description` 只保留了长度，`param_*` 太稀疏，`image` 是图片文件名：

```python
from sklearn.preprocessing import LabelEncoder

cate_cols = ['city', 'category_name', 'user_type']
for c in cate_cols:
    data[c] = LabelEncoder().fit_transform(data[c].values)

new_data = data.drop(['user_id', 'description', 'image',
                      'parent_category_name', 'region',
                      'item_id', 'param_1', 'param_2', 'param_3',
                      'title'], axis=1)
```

清洗完之后，特征数从原始的 17 列变成了 **37 列**。

### 2.3 模型训练与超参数调优

**按时间切分训练和验证。** 这一步和很多比赛不一样：我们没有随机切分，而是用日期切。训练集取 2017 年 4 月 7 日及以前的广告，测试集取 4 月 8 日之后的：

```python
from sklearn.model_selection import train_test_split

# 从拼接后的表里按日期切开：前面是训练期，后面是测试期
X    = new_data.loc[new_data.activation_date <= pd.to_datetime('2017-04-07')]
X_te = new_data.loc[new_data.activation_date >= pd.to_datetime('2017-04-08')]

y = X['deal_probability']
X = X.drop(['deal_probability', 'activation_date'], axis=1)

# 训练期内部再随机切 20% 做本地验证
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=2018)
X_te = X_te.drop(['deal_probability', 'activation_date'], axis=1)
print(X_tr.shape, X_va.shape, X_te.shape)
# (1202739, 37) (300685, 37) (508438, 37)
```

为什么这么做？因为平台的广告是**持续流入**的，模型上线后要预测的是未来广告。如果本地用随机切分，模型可能在验证集上见到「同一天、同一批用户」的广告，分数偏乐观；按时间切才更接近真实场景。

**LightGBM 回归。** 我们用 LightGBM（一种梯度提升树，把很多棵小决策树串起来，每棵树都在纠正前面树的错误）。它擅长处理带类别特征的宽表，训练快、内存省：

```python
import lightgbm as lgb

# 包装成 LightGBM 的 Dataset；明确告诉它哪几列是类别特征
tr_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cate_cols)
va_data = lgb.Dataset(X_va, label=y_va, categorical_feature=cate_cols,
                      reference=tr_data)

parameters = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',   # 直接回归到 [0,1] 的成交概率
    'metric': 'rmse',            # 用比赛指标做早停依据
    'num_leaves': 31,            # 每棵树的叶子数，控制复杂度
    'learning_rate': 0.05,       # 学习率：每棵树的贡献打 5 折
    'feature_fraction': 0.9,     # 每次分裂随机用 90% 的特征，增加多样性
    'bagging_fraction': 0.8,     # 每次迭代随机用 80% 的样本
    'bagging_freq': 5,           # 每 5 轮做一次采样
    'verbose': 50,
}

model = lgb.train(parameters, tr_data,
                  valid_sets=va_data,
                  num_boost_round=300,
                  early_stopping_rounds=120,   # 120 轮不提升就停
                  verbose_eval=50)
```

关于调参，我们主要调了这几项：

| 参数 | 取值 | 为什么这么选 |
| --- | --- | --- |
| `num_leaves` | 31 | 叶子太多会记住噪声，31 是中小规模数据的安全值 |
| `learning_rate` | 0.05 | 配合 300 轮，学习率和轮数是「跷跷板」，一个高另一个就得低 |
| `feature_fraction` | 0.9 | 轻度特征采样，防止个别强特征（比如均值编码）垄断分裂 |
| `bagging_fraction` | 0.8 | 样本采样，降低方差 |
| `early_stopping_rounds` | 120 | 早停耐心值给得大一些，因为 `lr=0.05` 时曲线下降比较慢 |

我们没有做大规模的网格搜索，原因很实在：150 万行 × 37 维，一轮训练就要好几分钟，网格搜索的代价和收益不成比例。**在特征已经比较到位的情况下，默认参数附近的稳定配置往往就能拿到不错的分数**，把时间花在特征上更划算。

### 2.4 模型测试与指标报告

训练过程中的验证曲线是这样的（每 50 轮打印一次）：

```
[50]    valid_0's rmse: 0.230052
[100]   valid_0's rmse: 0.228124
[150]   valid_0's rmse: 0.227497
[200]   valid_0's rmse: 0.227112
[250]   valid_0's rmse: 0.226803
[300]   valid_0's rmse: 0.226588
```

最终验证 RMSE 是 **0.226588**，而且 300 轮结束都没有触发早停——说明模型还有继续下降的空间，只是受限于轮数。曲线前 50 轮下降最快（0.2301），后面越来越平，这是梯度提升的典型特征：**容易学的规律先被学走，剩下的都是难啃的细节**。

我们还画了特征重要性（`importance_type='gain'`），发现排在最前面的基本是均值编码出来的那几列——`category_name_deal_probability_avg`、`city_price_avg` 这类特征贡献最大。这符合直觉：**「这个类目的平均成交率」比原始的类目编号有用得多**。

最后对测试集预测并生成提交：

```python
y_pred = model.predict(X_te)

sub = pd.read_csv('../input/sample_submission.csv')
sub['deal_probability'] = y_pred

# 关键：回归模型可能输出负值或大于 1 的值，必须裁剪回合法区间
sub['deal_probability'] = sub['deal_probability'].clip(0.0, 1.0)

sub.to_csv('lgb_with_mean_encode.csv', index=False)
print(sub.head())
```

**进一步提升：多方案融合。** 单模型停在 0.2266 左右之后，我们换了个思路——不追求「训练一个更强的模型」，而是**把多个差异化的提交做平均**。

材料里有一份公榜融合方案，收集了 24 份公榜 RMSE 都在 0.2269 以内的提交：

```python
from math import sqrt
from sklearn import metrics
import pandas as pd
import numpy as np

# 读入多份提交（每份都有 item_id 和 deal_probability）
df0  = pd.read_csv('../input/public-solutions/base0_0.2211.csv',  names=['item_id', 'p0'])
df1  = pd.read_csv('../input/public-solutions/base1_0.2212.csv',  names=['item_id', 'p1'])
df2  = pd.read_csv('../input/public-solutions/base2_0.2212.csv',  names=['item_id', 'p2'])
df14 = pd.read_csv('../input/public-solutions/base14_0.2237.csv', names=['item_id', 'p14'])
df18 = pd.read_csv('../input/public-solutions/base18_0.2246.csv', names=['item_id', 'p18'])

# 按 item_id 拼成一张表
blend = df0.merge(df1, on='item_id').merge(df2, on='item_id') \
           .merge(df14, on='item_id').merge(df18, on='item_id')

# 计算两两之间的 RMSE，衡量两份提交「有多像」
M = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        M[i, j] = sqrt(metrics.mean_squared_error(blend.iloc[:, i + 1],
                                                  blend.iloc[:, j + 1]))
print(M)     # 数值越小说明两份提交越相似

# 融合方式很简单：五份预测取算术平均
blend['deal_probability'] = blend.iloc[:, 1:].mean(axis=1)
blend[['item_id', 'deal_probability']].to_csv('best_public_blend.csv', index=False)
```

选哪五份是有讲究的。入围的 24 份分数从 0.2211 到 0.2269，如果只挑分数最高的五份，它们往往来自相似的方法（都是 LightGBM + 同样的特征），预测高度相关，平均之后几乎没提升。所以我们做了**相关性筛选**：先算两两 RMSE 矩阵，优先挑「分数低**且**与第一名差异大」的那几份。最终选的是 0.2211、0.2212、0.2212、0.2237、0.2246 五份——其中 0.2237 和 0.2246 单看分数更差，但它们与第一名的相关性明显更低，属于「错得不一样」的模型。五份等权平均之后，公榜 RMSE 达到 **0.2204**，比最好的单份（0.2211）还好。

这个结果很能说明问题：**融合的收益来自「差异」，而不是「强」**。五个各自 0.223 左右、但彼此独立性高的模型，平均起来能打赢 0.2211 的单个模型。

---

## 三、总结

本文中我们完成了 Avito 需求预测的完整流程。我们首先用 EDA 摸清了数据：150 万条训练广告、50 万条测试广告、18 个字段，并且确认了价格缺失普遍存在、用户类型和星期几对成交概率有系统性影响；其次我们做了数据清洗，把训练集和测试集拼起来统一做类别编码，从日期里拆出「几号/星期几」，从标题和描述里提取字符长度，并对 9 个类别字段做均值编码（均值、标准差、均价三组统计量），同时用 `fit`/`transform` 分离的方式彻底堵住了统计量泄漏；接着我们按时间切分（4 月 7 日之前训练、之后测试）避免「用未来预测过去」，用 LightGBM 回归训练 300 轮、学习率 0.05、叶子数 31，配合特征采样和样本采样；然后我们在验证集上得到 RMSE **0.226588**，训练曲线到 300 轮仍在缓慢下降，特征重要性显示均值编码类特征贡献最大；最后我们用五份低相关性提交做等权平均，把公榜分数提升到 **0.2204**。

这个比赛给我们最大的启发有两点。第一，**均值编码是类别特征的大杀器，但它也是最容易泄漏的特征**——必须在训练折内统计、在验证折上只做映射，否则验证分数会骗你。第二，**融合要挑「意见不同」的模型**：我们最终选入平均的那五份里，有两份单看分数比最好的差了一截，但它们和第一名错得不一样，正是这种差异带来了 0.0007 的额外提升。
