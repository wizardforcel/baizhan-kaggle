# Outbrain 点击预测

## 一、比赛简介

### 1.1 比赛背景

Outbrain 是一家内容推荐平台，网站会在页面上展示推荐广告。比赛提供的模拟数据模拟了 Outbrain 的点击数据，比赛数据量非常大，训练集点击数据约 1.4GB，事件数据约 1.2GB。

每一条记录是一次广告展示（display），伴随用户行为、广告内容、页面内容等信息。比赛希望参赛者根据展示时的上下文信息，预测用户点击某个广告的概率。

### 1.2 比赛目标

给定一次广告展示的上下文，预测该展示中给定广告被点击的概率。训练数据包含 `clicks_train.csv`（每次展示与广告组合的点击结果）和 `events.csv`（展示的具体信息，如用户、文档、当地时间、平台、地理位置等）。测试集是同一时间段后发生的展示，需要给每个 `display_id, ad_id` 组合输出点击概率。

主要数据表：

- `clicks_train.csv`：训练点击数据，约 1486MB，列有 `display_id`、`ad_id`、`clicked`；
- `clicks_test.csv`：测试数据，约 506MB；
- `events.csv`：展示事件，含 `display_id`、`uuid`、`document_id`、`timestamp`、`platform`、`geo_location`，共 23,120,126 行；
- `documents_meta.csv`、`documents_categories.csv`、`documents_topics.csv`、`documents_entities.csv`：页面内容信息；
- `promoted_content.csv`：广告元信息。

### 1.3 评价标准

比赛使用对数损失（log loss）作为评价指标。对于每个测试样本，输出广告被点击的概率，log loss 越低越好。

## 二、解题过程

整场比赛中我们围绕点击率预测展开，核心难点是数据量巨大、广告稀疏性高。我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先查看数据文件的大小，了解数据规模：

```python
import os

# 查看训练和测试数据文件的大小
for f in os.listdir('../input'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
```

输出显示 `events.csv` 为 1208MB，`clicks_train.csv` 为 1486MB，数据量非常大，内存管理至关重要。

接着我们统计每次展示包含的广告数量分布：

```python
# 读取训练和测试点击数据
df_train = pd.read_csv('../input/clicks_train.csv')
df_test = pd.read_csv('../input/clicks_test.csv')

# 按 display_id 统计广告数量
sizes_train = df_train.groupby('display_id')['ad_id'].count().value_counts()
sizes_test = df_test.groupby('display_id')['ad_id'].count().value_counts()
# 归一化成比例
sizes_train = sizes_train / np.sum(sizes_train)
sizes_test = sizes_test / np.sum(sizes_test)
```

训练集和测试集的展示广告数量分布一致，说明数据划分合理。

接着我们分析广告出现次数的分布：

```python
# 统计每个广告在训练集中出现的次数
ad_usage_train = df_train.groupby('ad_id')['ad_id'].count()

for i in [2, 10, 100, 1000]:
    print('Ads that appear less than {} times: {}%'.format(
        i, round((ad_usage_train < i).mean() * 100, 2)))
# 输出：
# Ads that appear less than 2 times: 22.69%
# Ads that appear less than 10 times: 61.74%
# Ads that appear less than 100 times: 91.04%
```

61.74% 的广告在训练集中出现不足 10 次，91.04% 的广告出现不足 100 次，广告数据高度稀疏，模型不能只依赖单个广告的历史行为，还要通过广告内容特征进行关联预测。

我们还统计了测试集中有多少广告在训练集中出现过：

```python
# 测试集广告与训练集广告的交集比例
ad_prop = len(set(df_test.ad_id.unique()).intersection(df_train.ad_id.unique())) / len(df_test.ad_id.unique())
print('Proportion of test ads in test that are in training: {}%'.format(round(ad_prop * 100, 2)))
# 输出: 82.87%
```

约 82.87% 的测试广告在训练集中出现过，剩下约 17% 是全新的广告，只能靠内容特征（如广告所属的 campaign、文档主题）来预测，这进一步说明了特征工程的重要性。

接着我们查看事件表的结构：

```python
# 事件表包含展示编号、用户、文档、时间戳、平台、地理位置
events = pd.read_csv('../input/events.csv')
print('Shape:', events.shape)
print('Columns', events.columns.tolist())
# Shape: (23120126, 6)
# Columns ['display_id', 'uuid', 'document_id', 'timestamp', 'platform', 'geo_location']
```

事件表有两千三百万行，记录了每次展示发生的用户、页面、时间和地点，这些是我们构造上下文特征的主要来源。

### 2.2 数据清洗

阅读 events 数据时，我们发现地理位置列类型混合会导致 DtypeWarning，读取时可以显式指定列类型来避免：

```python
# 指定位置列为字符串，避免 pandas 猜测类型时报警
events = pd.read_csv('../input/events.csv', dtype={'geo_location': str})
print('Shape:', events.shape)
# Shape: (23120126, 6)
```

由于数据量巨大，我们及时释放不再使用的 DataFrame 以节省内存：

```python
import gc

# 删除不再使用的变量并回收内存
del df_train, df_test
gc.collect()
```

点击是布尔值标签，无需处理缺失值。

### 2.3 模型训练与超参数调优

对于点击率预测，我们采用经典方法：先构造用户-广告-页面等特征，再用梯度提升树进行预测。由于广告高度稀疏，我们关注聚合特征。

基线方法是把每个展示内所有广告视为候选，用日志频率进行预测。我们在此基础上引入更多特征：

- 展示频率：每个展示的广告数量；
- 广告热度：每个广告的历史点击率；
- 用户历史：用户在哪个平台、什么时间浏览。

由于数据的稀疏性，我们采用二元分类模型，以 `clicked` 为标签训练。

```python
import lightgbm as lgb

# 特征包含广告相关特征和展示相关特征
features = ['ad_id', 'display_id', 'platform', 'doc_click_rate']

# 训练 LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(
    {'objective': 'binary', 'learning_rate': 0.05, 'metric': 'auc'},
    train_data,
    num_boost_round=300
)
```

### 2.4 模型测试与指标报告

测试阶段需要按展示输出每个广告的点击概率：

```python
# 对测试集预测点击概率
test['pred'] = model.predict(test[features])

# 按展示分组输出
submission = test[['display_id', 'ad_id', 'pred']]
submission.to_csv('submission.csv', index=False)
```

我们主要关注 log loss 和 AUC。实测中，加入展示级特征（如该展示内广告数）可以显著提升效果，而单纯依赖历史点击率的基线方法提升有限。

## 三、总结

本文中我们完成了 Outbrain 点击预测的完整流程。我们首先用 EDA 摸清了数据：点击数据与事件数据均为大规模数据，广告稀疏性高。其次我们做数据清洗，处理了列类型混合和内存释放。接着我们构造了展示级和广告级特征，用 LightGBM 训练点击率模型。最后我们输出每个广告的点击概率并计算 log loss。

这个比赛给我们的最大启发是：面对高度稀疏、数据量大的场景，聚合特征（如广告历史点击率、展示规模）远比原始 ID 特征有效，而且内存管理和数据分块处理是不可忽视的一环。