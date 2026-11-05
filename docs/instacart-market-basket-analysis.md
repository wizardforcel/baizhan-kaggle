# Instacart Market Basket Analysis

## 一、比赛简介

### 1.1 比赛背景
Instacart 是一家在美国和加拿大提供生鲜杂货配送服务的公司。用户通过 App 下单，Instacart 的采购员到附近超市购买商品并送到家里。这个比赛公开了约 20 万名用户的订单历史，希望参赛者根据用户过去的行为，预测他们下一单会买哪些商品。

对 Instacart 来说，这样的预测很有商业价值：如果平台能提前知道用户下一单大概需要什么，就可以优化库存、推荐商品、安排采购路线。数据里的订单分为 prior、train 和 test 三类：prior 是用户过去的历史订单，train 是带有答案的“未来订单”，test 是需要预测的“未来订单”。每个用户只会属于 train 或 test 其中一组。

### 1.2 比赛目标
比赛目标不是直接列出下一单的商品，而是对每个“用户—商品”组合预测 `reordered`，即这个用户下一单会不会再次购买该商品。训练时使用 prior 订单构造特征，用 train 订单中的真实商品作为标签训练模型；预测时对 test 用户的候选商品打分，最后为每个用户挑出概率最高的一批商品组成订单。

数据文件包括：
- `orders.csv`：订单表，包含 `order_id`、`user_id`、`eval_set`、`order_number`、`order_dow`、`order_hour_of_day`、`days_since_prior_order`；
- `order_products__prior.csv`：历史订单中的商品明细，包含 `order_id`、`product_id`、`add_to_cart_order`、`reordered`；
- `order_products__train.csv`：训练用的未来订单商品明细；
- `products.csv`：商品表，包含 `product_id`、`product_name`、`aisle_id`、`department_id`；
- `aisles.csv` 和 `departments.csv`：商品所属货架和部门。

### 1.3 评价标准
比赛使用平均 F1 分数（mean F1）评估：对每个用户，比较预测商品集合和真实商品集合的交集，计算每个用户的 F1，再对所有用户取平均。F1 是精确率和召回率的调和平均，既不能乱猜太多，也不能漏掉太多。最终分数越高越好。在 Kaggle 上，0.38 左右已经是不错的成绩，材料中有一篇 notebook 的标题就叫 “F1 0.38”。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读取订单表和商品明细，看看数据规模和用户行为。

```python
import pandas as pd
import numpy as np
import gc

# 读取主表
orders = pd.read_csv('../input/orders.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
products = pd.read_csv('../input/products.csv')

# 查看订单表的规模和列
print(orders.shape)
orders.head()
```
`orders` 里每条记录是一次订单，`eval_set` 告诉我们是 prior、train 还是 test。材料里说明有 20 万用户，每个用户有 4 到 100 次订单。

我们接着看用户的订单数分布。

```python
# 每个用户的订单数量
user_orders = orders.groupby('user_id')['order_id'].count()
print(user_orders.describe())

# 订单数量与 eval_set 的关系
orders.groupby('eval_set')['order_id'].count()
```
prior 订单占绝大多数，train 和 test 各只有一次“未来订单”。这意味着我们能用大量历史行为来构造特征。

再看商品维度和重复购买情况。

```python
# 商品总数
print('products:', len(products))

# 历史订单中每个商品被购买的次数
top_products = order_products_prior['product_id'].value_counts().head(20)
print(top_products)
```
材料里的特征表 `uxp` 以 `user_id` 和 `product_id` 为索引，第一列 `uxp_total_bought` 表示某个用户买过某个商品的总次数。我们还统计了商品的 one-shot 比例、最近 5 单购买次数、用户订单总数、首次购买序号等。

```python
# 每个用户购买的商品种类数
user_product_count = order_products_prior.groupby('user_id')['product_id'].nunique()
print(user_product_count.describe())
```
### 2.2 数据清洗
我们其次把订单和商品明细合并，形成“用户—商品”主表。

```python
# 先合并订单与历史商品明细
prd = orders.merge(order_products_prior, on='order_id', how='left')

# 只保留 prior 订单，避免把 train/test 的未来信息泄漏进特征
prd = prd[prd['eval_set'] == 'prior']
```
这里的关键是只使用 prior 数据。train 和 test 的订单是我们要预测的目标，如果提前把它们的商品放进特征，就会造成数据泄漏。

然后我们建立用户—商品表。

```python
# 统计每个用户买每个商品的总次数
uxp = prd.groupby(['user_id', 'product_id'])[['order_id']].count()
uxp.columns = ['uxp_total_bought']
uxp = uxp.reset_index()
```
商品名称里可能有空格和逗号，我们统一替换成下划线，方便后续分析。

```python
# 统一商品名的格式，方便按名称搜索和展示
products['product_name'] = products['product_name'].str.replace(' ', '_').str.lower()
products['product_name'] = products['product_name'].str.replace(',', '-').str.lower()
```
内存也是清洗的一部分。Kaggle 的笔记本内存有限，所以材料里反复使用 `gc.collect()` 和 `del` 删除中间表。

```python
# 删除不再需要的中间表，释放内存
del order_products_prior
gc.collect()
```
### 2.3 模型训练与超参数调优
我们接着构造特征。材料里的特征非常丰富，可以分成三类：用户特征、商品特征、用户—商品交互特征。

```python
# 1) 商品层面的再购率
product_var = prd.groupby('product_id')[['reordered']].mean()
product_var.columns = ['reorder_ratio']

# 2) 商品在购物车中的平均位置
product_var['mean_add_to_cart_order'] = prd.groupby('product_id')[['add_to_cart_order']].mean()

# 3) 商品被多少用户买过
product_var['unique_users'] = prd.groupby('product_id')['user_id'].nunique()
```
用户—商品交互特征更能反映个人习惯。

```python
# 最近 5 单中购买某商品的次数
prd['order_number_back'] = prd.groupby('user_id')['order_number'].transform(max) - prd.order_number + 1
prd5 = prd[prd.order_number_back <= 5]
last_five = prd5.groupby(['user_id', 'product_id'])[['order_id']].count()
last_five.columns = ['times_last5']
last_five['times_last5_ratio'] = last_five.times_last5 / 5

# 用户购买某商品的总次数与购买窗口长度之比
times = prd.groupby(['user_id', 'product_id'])[['order_id']].count()
times.columns = ['Times_Bought_N']
total_orders = prd.groupby('user_id')[['order_number']].max()
total_orders.columns = ['total_orders']
first_order = prd.groupby(['user_id', 'product_id'])[['order_number']].min().reset_index()
```
这些特征的含义很直观：如果用户最近 5 单里经常买某商品，那么下一单再买的概率就高。

然后我们训练 XGBoost。

```python
import xgboost as xgb

# 构造训练集：train 用户及其历史购买过的商品
train_users = orders[orders.eval_set == 'train']['user_id'].unique()
train_data = uxp[uxp.user_id.isin(train_users)].copy()

# 标签：在 train 订单中是否再次购买
train_labels = order_products_train[['order_id', 'product_id']]
train_labels = train_labels.merge(orders[['order_id', 'user_id']], on='order_id')
train_data = train_data.merge(train_labels,
                              on=['user_id', 'product_id'], how='left')
train_data['label'] = (~train_data['order_id'].isna()).astype(int)
```
材料里的 XGBoost 配置强调使用早停和交叉验证。

```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
model = xgb.train(params, dtrain, num_boost_round=1000,
                  evals=[(dvalid, 'valid')],
                  early_stopping_rounds=50)
```
调参时我们重点看 `max_depth`、`eta` 和 `min_child_weight`。树太深容易记住用户的历史噪声，树太浅又学不到交互特征。我们通过验证集上的 F1 而不是 logloss 来选择最终轮数，因为比赛关心的是排序后的 Top-N 商品集合。

预测时，对每个 test 用户的所有历史商品打分，再按概率排序取前 N 个。N 的选择直接影响 F1：取太少召回低，取太多精确率低。我们在验证集上尝试不同的 N，找到平均 F1 最高的截断值。

```python
# 对每个用户按概率排序，取前 N 个商品
def make_order(pred_df, top_n=10):
    return (pred_df.sort_values(['user_id', 'prob'], ascending=[True, False])
                    .groupby('user_id').head(top_n))
```
### 2.4 模型测试与指标报告
我们最后对验证用户计算 F1。

```python
def f1_for_user(true_items, pred_items):
    # 交集大小决定 TP，精确率和召回率都依赖它
    tp = len(set(true_items) & set(pred_items))
    if tp == 0:
        return 0.0
    precision = tp / len(pred_items)
    recall = tp / len(true_items)
    return 2 * precision * recall / (precision + recall)
```
把所有验证用户的 F1 取平均，就得到与比赛一致的指标。

```python
scores = []
for user_id, group in valid_predictions.groupby('user_id'):
    true_items = valid_truth.loc[user_id]
    pred_items = group['product_id'].tolist()
    scores.append(f1_for_user(true_items, pred_items))

print('Mean F1:', np.mean(scores))
```
材料里的方案报告了约 0.38 的 F1。这个分数说明，仅仅根据历史购买行为做再购预测，已经能抓住相当一部分用户需求。为了进一步提升，材料里还探索了关联规则、K-means 客户分群、PCA、商品嵌入和矩阵分解等方法，但主线仍然是特征工程加 XGBoost。

最后生成提交文件。

```python
# 每个 test 用户一行，products 列是空格分隔的商品编号
submission = pred_df.groupby('user_id')['product_id'].apply(
    lambda ids: ' '.join(map(str, ids))).reset_index()
submission.columns = ['user_id', 'products']
submission.to_csv('submission.csv', index=False)
```
我们检查测试用户数量是否与 `orders` 中的 test 用户一致，避免漏掉用户。

## 三、总结

本文中我们完成了 Instacart Market Basket Analysis 的完整流程。我们首先通过 EDA 了解了 20 万用户、4 到 100 次订单的结构，确认 prior 是历史行为、train/test 是未来订单。其次我们只使用 prior 数据构造用户—商品表，清洗商品名并释放中间表内存，避免数据泄漏。然后我们构造了商品再购率、购物车位置、最近 5 单购买次数、购买窗口比例等特征，用 XGBoost 预测每个用户—商品组合的再购概率，并用验证集选择 Top-N 截断。最后我们按用户平均 F1 评估，得到约 0.38 的分数，并按比赛格式输出每个用户的商品列表。

这个比赛给我们的最大启发是：推荐问题的特征工程比模型本身更重要。用户最近的行为、商品的热度和个人偏好交互，是预测下一单的关键；同时必须严格区分历史与未来，否则离线分数会虚高。
