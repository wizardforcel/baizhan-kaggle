# Instacart Market Basket Analysis

## 一、比赛简介

### 1.1 比赛背景
Instacart 是一家美国的生鲜杂货配送公司。用户在手机上下单，平台上的采购员（shopper）去附近超市按单采购，再把商品送到用户家里。这个过程听起来简单，但对平台来说，库存、拣货路线、推荐和促销都依赖一个核心问题：用户下一单会买什么？

这个比赛公开了约 20 万名用户的匿名订单历史，希望参赛者根据过去的行为预测未来订单。数据被切分成三部分：prior 是用户的历史订单，train 是带有答案的「未来订单」，test 是真正需要预测的未来订单。每个用户只会属于 train 或 test 其中一组。这样的设计让我们能用 prior 构造特征，用 train 训练和验证，最后对 test 做预测。

对入门者来说，这个比赛是一个非常好的推荐系统练习：它把「预测用户下一单买什么」转化成了「预测每个用户—商品组合是否会再次购买」的二分类问题，既直观又可以用常见的梯度提升树解决。

### 1.2 比赛目标
比赛的目标是为每个 test 用户预测下一单的商品集合。数据里有以下文件：

- `orders.csv`：订单主表，包含 `order_id`、`user_id`、`eval_set`、`order_number`、`order_dow`、`order_hour_of_day`、`days_since_prior_order`；
- `order_products__prior.csv`：历史订单商品明细，包含 `order_id`、`product_id`、`add_to_cart_order`、`reordered`；
- `order_products__train.csv`：训练用未来订单商品明细；
- `products.csv`：商品表，包含 `product_id`、`product_name`、`aisle_id`、`department_id`；
- `aisles.csv`、`departments.csv`：商品所属货架和部门。

材料里的方案把问题定义成用户—商品级别：先构造一张以 `(user_id, product_id)` 为索引的表，每行代表用户买过某个商品，再为每行计算各种统计特征，最后预测 `reordered` 的概率。

### 1.3 评价标准
比赛使用平均 F1 分数（mean F1）。对每个用户，比较预测商品集合与真实商品集合：

```
precision = |交集| / |预测集合|
recall    = |交集| / |真实集合|
F1 = 2 * precision * recall / (precision + recall)
```

然后对所有用户取平均。也就是说，对每个用户，我们既要尽量猜中他会买的商品，又不能把太多不相关的商品塞进订单。材料里有一篇 notebook 的标题是「F1 0.38」，这是该方案报告的成绩，说明 0.38 已经是一个比较扎实的基线。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读取订单表和商品明细，查看数据规模。

```python
import pandas as pd
import numpy as np
import gc

orders = pd.read_csv('../input/orders.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
products = pd.read_csv('../input/products.csv')

print('orders:', orders.shape)
print('order_products_prior:', order_products_prior.shape)
print('products:', products.shape)
orders.head()
```

`orders` 每条记录是一次订单，`eval_set` 标明它属于 prior、train 还是 test。材料里说明有约 20 万用户，每个用户有 4 到 100 次订单。`order_number` 是用户订单的时间顺序，`days_since_prior_order` 是距离上一单的天数，这两个字段对行为建模很有用。

接着看每个用户的订单数。

```python
user_orders = orders.groupby('user_id')['order_id'].count()
print(user_orders.describe())
```

输出显示订单数从 4 到 100 不等，平均在 10 单以上。prior 订单占绝大多数，train 和 test 各只有一次「未来订单」。这个结构意味着：每个用户的历史行为很丰富，但目标只有一个订单。

我们再看商品维度和热门商品。

```python
print('商品数:', len(products))

top_products = (order_products_prior['product_id']
                .value_counts().head(20))
print(top_products)
```

结果显示，少数商品被大量购买，长尾商品很多。热门商品在预测中往往有更高先验概率，但只推荐热门商品会导致所有用户的订单雷同，F1 不会高。

我们还统计了用户购买的商品种类数。

```python
user_product_count = (order_products_prior
                      .groupby('user_id')['product_id']
                      .nunique())
print(user_product_count.describe())
```

平均每个用户买过几十种商品，最多的有几百种。这说明「候选商品」的规模是可控的：我们只需要对每个用户历史买过的商品打分，而不需要对全部几万种商品打分。

### 2.2 数据清洗
我们其次把订单与商品明细合并，构造用户—商品主表。

```python
# 合并订单和 prior 商品明细
prd = orders.merge(order_products_prior, on='order_id', how='left')

# 关键：只保留 prior，避免把 train/test 的未来信息泄漏进特征
prd = prd[prd['eval_set'] == 'prior'].copy()
```

这一步是整个流程的红线。train 和 test 的订单是我们要预测的目标，如果把它们的商品提前算进特征，模型就会「偷看答案」，线下分数虚高，线上完全失效。

构造用户—商品表：

```python
uxp = prd.groupby(['user_id', 'product_id'])[['order_id']].count()
uxp.columns = ['uxp_total_bought']
uxp = uxp.reset_index()
```

这样每行代表一个用户买过的一个商品，`uxp_total_bought` 是购买次数。后续所有特征都往这张表上合并。

商品名称里面可能有空格和逗号，我们统一格式，方便按名字检索。

```python
products['product_name'] = (products['product_name']
                            .str.replace(' ', '_').str.lower())
products['product_name'] = (products['product_name']
                            .str.replace(',', '-').str.lower())
```

内存管理也是清洗的一部分。材料里反复使用 `gc.collect()`，因为 prior 明细有几千万行，中间表很容易把 17 GB 内存撑爆。

```python
# 删除不再需要的中间表，主动回收内存
del order_products_prior
gc.collect()
```

### 2.3 模型训练与超参数调优
我们接着构造三类特征：商品特征、用户特征、用户—商品交互特征。

商品层面：

```python
# 商品的平均再购率：买过它的人里有多少比例会再次购买
product_var = prd.groupby('product_id')[['reordered']].mean()
product_var.columns = ['reorder_ratio']

# 商品在购物车中的平均位置，越靠前说明越可能是计划购买
product_var['mean_add_to_cart_order'] = (
    prd.groupby('product_id')[['add_to_cart_order']].mean())

# 商品被多少个不同用户买过，代表流行度
product_var['unique_users'] = (
    prd.groupby('product_id')['user_id'].nunique())
```

用户层面：

```python
# 用户的总订单数、平均订单大小、平均下单间隔
user_var = prd.groupby('user_id').agg(
    total_orders=('order_number', 'max'),
    avg_basket=('order_id', 'nunique'),
    avg_days=('days_since_prior_order', 'mean'))
```

用户—商品交互特征最能体现个人习惯：

```python
# 最近 5 单中购买某商品的次数
prd['order_number_back'] = (prd.groupby('user_id')['order_number']
                            .transform('max') - prd.order_number + 1)
prd5 = prd[prd.order_number_back <= 5]
last_five = prd5.groupby(['user_id', 'product_id'])[['order_id']].count()
last_five.columns = ['times_last5']
last_five['times_last5_ratio'] = last_five.times_last5 / 5

# 购买总次数与购买窗口长度之比
times = prd.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
total_orders = prd.groupby('user_id')[['order_number']].max()
total_orders.columns = ['total_orders']
first_order = (prd.groupby(['user_id', 'product_id'])[['order_number']]
               .min().reset_index())
```

还有一类很有用的特征：用户第几次购买某商品后，连续多少单没有买。材料里的 `item_N2_ratio`、`item_N3_ratio` 等就是类似思路。

```python
# 计算一个商品在用户订单序列中的间隔
def gap_features(group):
    order_numbers = group.sort_values('order_number')['order_number']
    gaps = order_numbers.diff().fillna(0)
    return pd.Series({
        'first_order_number': order_numbers.min(),
        'last_gap': gaps.iloc[-1] if len(gaps) else 0,
        'mean_gap': gaps.mean()})
```

把所有特征合并到 `uxp` 上，得到训练表。

```python
for df in [last_five, order_ratio, product_var]:
    uxp = uxp.merge(df, on=['user_id', 'product_id'], how='left')
```

标签的构造：对 train 用户，如果某商品出现在 train 订单里，标签为 1，否则为 0。

```python
train_users = orders[orders.eval_set == 'train']['user_id'].unique()
train_data = uxp[uxp.user_id.isin(train_users)].copy()

train_labels = (order_products_train[['order_id', 'product_id']]
                .merge(orders[['order_id', 'user_id']], on='order_id'))
train_data = train_data.merge(train_labels,
                              on=['user_id', 'product_id'], how='left')
train_data['label'] = (~train_data['order_id'].isna()).astype(int)
```

模型使用 XGBoost。

```python
import xgboost as xgb

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,          # 控制树的复杂度
    'eta': 0.05,             # 学习率
    'subsample': 0.8,        # 行采样
    'colsample_bytree': 0.8, # 列采样
    'min_child_weight': 3
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
model = xgb.train(params, dtrain, num_boost_round=1000,
                  evals=[(dvalid, 'valid')],
                  early_stopping_rounds=50)
```

调参时我们关注 `max_depth`、`eta`、`min_child_weight` 三个参数。树太深会记住用户历史里的偶然波动，树太浅学不到交互特征。`eta` 越小需要越多轮，但通常泛化更好。

真正决定 F1 的还有每个用户保留多少个商品。我们按预测概率排序，取前 N 个。

```python
def top_n_per_user(pred_df, n=10):
    return (pred_df.sort_values(['user_id', 'prob'], ascending=[True, False])
            .groupby('user_id').head(n))
```

N 的选择必须用验证集调。取太少召回低，取太多精确率低。材料里的最佳 N 在 10 左右，具体取决于用户历史订单的平均大小。

### 2.4 模型测试与指标报告
我们按用户计算 F1。

```python
def f1_for_user(true_items, pred_items):
    tp = len(set(true_items) & set(pred_items))
    if tp == 0:
        return 0.0
    precision = tp / len(pred_items)
    recall = tp / len(true_items)
    return 2 * precision * recall / (precision + recall)
```

对所有验证用户取平均：

```python
scores = []
for user_id, group in valid_predictions.groupby('user_id'):
    true_items = valid_truth.loc[user_id]
    pred_items = group['product_id'].tolist()
    scores.append(f1_for_user(true_items, pred_items))

print('Mean F1:', np.mean(scores))
```

材料里的主线方案报告约 0.38 的 F1。这个成绩的意义是：只用历史行为，不借助任何商品内容或用户画像，模型已经能猜中相当一部分商品。

最后生成提交文件：

```python
# 每个 test 用户一行，products 列是空格分隔的商品编号
submission = (pred_df.groupby('user_id')['product_id']
              .apply(lambda ids: ' '.join(map(str, ids)))
              .reset_index())
submission.columns = ['user_id', 'products']
submission.to_csv('submission.csv', index=False)
```

检查 test 用户数是否与 orders 中一致，避免漏用户。

## 三、总结

本文中我们完成了 Instacart Market Basket Analysis 的完整流程。我们首先通过 EDA 了解了 20 万用户、4 到 100 次订单的数据结构，确认 prior 是历史行为、train/test 是未来订单。其次我们严格只使用 prior 数据构造用户—商品表，清洗商品名并主动释放内存，避免数据泄漏。然后我们构造了商品再购率、购物车位置、最近 5 单购买次数、购买间隔等特征，用 XGBoost 预测每个用户—商品组合的再购概率，并用验证集选择 Top-N 截断。最后我们按用户平均 F1 评估，得到约 0.38 的分数，并按格式生成提交。

这个比赛给我们的最大启发是：推荐问题的特征工程比模型本身更重要。用户最近的行为、商品的热度和个人偏好交互，是预测下一单的关键；同时必须严格区分历史与未来，否则离线分数会虚高。此外，F1 这个指标决定了我们不能只追求「买得准」，还要在召回和精确之间找平衡，Top-N 的 N 本身就是一个需要认真调的超参数。
