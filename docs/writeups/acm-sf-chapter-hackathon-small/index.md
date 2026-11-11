# ACM 旧金山分会黑客马拉松（小规模版）

## 一、比赛简介

### 1.1 比赛背景

这是 ACM 旧金山分会（ACM San Francisco Chapter）举办的一场小规模黑客马拉松比赛，题目来自一家电商网站的真实搜索日志。用户在该网站上搜索游戏名称，然后从搜索结果里点开一个或多个商品。比赛把这批「搜索→点击」记录公开出来，要求参赛者**根据用户的一次搜索行为，预测他最终点开了哪个商品**。

这个问题的难点有两层。第一层是**信息太少**：一行数据里真正有意义的只有「用户输入了什么搜索词」，以及两个时间戳；用户 id 是一串随机字符，商品类目和商品编号又几乎是绑定的。第二层是**类别太多**：候选商品数量很大，而同一句搜索词下可能有十几个长得很像的游戏，模型必须在「搜的词」和「点开的商品」之间学到映射，不能瞎猜热门商品。

### 1.2 比赛目标

比赛提供三份文件：

- `train.csv`：训练集，共 42365 行、6 列，
  - `user`：用户的匿名 id；
  - `sku`：被点击商品的编号（**这就是标签**）；
  - `category`：商品类目；
  - `query`：用户输入的搜索词；
  - `click_time`：点击发生的时间；
  - `query_time`：用户发起搜索的时间。
- `test.csv`：测试集，列和训练集基本相同，但没有 `sku` 列；
- `popular_skus.csv`、`small_product_data.xml`：热门商品清单与商品元数据，用来辅助理解 sku 的含义。

我们要给测试集的每一行填上一个商品编号，按比赛要求输出一列 `Sku`。

### 1.3 评价标准

我们手头的材料没有公布这场比赛的官方评分公式。从任务形态看，这是一个标准的**多分类**问题——「多分类」的意思就是答案不止两个，而是有成千上万个候选商品，模型每次要从里面挑一个。所以材料里的作者是拿**分类准确率**来衡量的——预测正确的商品数占总样本的比例，越高越好。这里要特别说明：商品种类非常多，随机乱猜的准确率接近零，所以哪怕准确率只有 0.3 出头，也说明模型确实从搜索词里学到了真实的对应关系。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**把「搜索词」和「时间」变成数字，其余列全部交给一个带网格搜索的随机森林。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先把训练集、测试集读进来，并看看输入目录里还有哪些辅助文件：

```python
import os
import numpy as np
import pandas as pd

# 读入训练集和测试集
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# 看看输入目录还有什么, 决定哪些能当辅助信息
print(os.listdir('../input'))
# ['popular_skus.csv', 'small_product_data.xml', 'train.csv',
#  'popular_skus.py', 'test.csv']

# 训练集大小: 42365 行 x 6 列
print(train_data.shape)
train_data.head()
```

打印前几行能看到数据的真实样子：一位用户在 2011 年 9 月 25 日 13:35 左右搜索 `Gears of war`，先点开商品 `9854804`，一分钟后又点开同款游戏附近的另一个商品 `2670133`；两分钟后他换成搜索 `Assassin creed`，点了 `9984142`。也就是说，**一个用户会连续搜不同的词、点不同的商品**，每行数据就是一次独立的「搜索→点击」事件。我们要预测的 `sku` 是商品编号，可能的取值非常多，这就构成了一个典型的「多分类」标签。

其次我们数一数搜索词有多少种、商品被点击得是否均匀：

```python
# 有多少种不同的搜索词
print(train_data['query'].nunique())   # 约 960 种

# 每个商品被点击的次数, 看分布是否均衡
train_data['sku'].value_counts().head(10)
```

搜索词大约 960 种，数量不大，可以做「词表编码」；而商品的点击次数分布非常不均——少数热门商品被点上很多次，大量商品只有零星几次点击。这意味着如果不管不顾地训练，模型很容易学会「无脑预测热门商品」，所以后面我们必须留心准确率的含义。

### 2.2 数据清洗

其次我们处理时间列。原始的 `click_time`、`query_time` 是带时区、带微秒的字符串，长度还不统一（有的到秒、有的带 6 位微秒），直接当字符串用没法比较先后。我们的做法是先把它补成统一的 26 位微秒格式，再去掉所有非数字字符，压成一个整数：

```python
def pad_time(s):
    """把时间字符串补成 26 位的微秒格式 YYYY-MM-DD HH:MM:SS.ffffff"""
    if len(s) == 19:                    # 只到秒, 补 6 个 0 当微秒
        return s + '.' + '0' * 6
    return s + (26 - len(s)) * '0'      # 微秒位数不足的, 在末尾补齐

def to_int_time(s):
    """去掉所有标点, 把时间压成一个可比较大小的整数"""
    return int(''.join(c for c in pad_time(s) if c.isdigit()))

# 对训练集的两列时间做同样的变换
X = train_data.copy()
X['click_time'] = X['click_time'].apply(to_int_time)
X['query_time'] = X['query_time'].apply(to_int_time)
```

这样一来，「点击比搜索晚了几秒」「两次点击间隔多久」这类时间先后关系，就可以用整数差直接算出来了。之所以要绕这一圈，是因为模型只认数字：字符串形式的日期直接比较会被当成普通文本，只有压成整数，才能正确表达「谁先谁后、差多少」。

接着我们处理搜索词。同一个词有人大写、有人小写，比如 `Gears of war` 和 `gears of war` 应该算同一个词，所以我们先统一转成小写，再按出现顺序给每个词编一个号：

```python
# 统一转小写, 避免大小写把同一个词拆成两类
query_lower = X['query'].str.lower()

# 按出现顺序建立 词 -> 编号 的字典
query_type = query_lower.unique()
query_dict = {q: i for i, q in enumerate(query_type)}
print(len(query_dict))         # 约 960
print(query_dict['ncaa 2011']) # 959, 说明这个词排在词表最后一个

# 用编号替换原始文本
X['query'] = query_lower.map(query_dict)
```

最后我们决定输入特征的取舍：把 `user`（随机字符串，语义太弱）和 `category`（商品类目与 sku 几乎一一对应，留着等于直接泄露答案）从特征里去掉，只保留 `query`、`click_time`、`query_time` 三列作为输入，把 `sku` 作为标签。这样模型学到的才是「搜索行为 → 商品」的映射，而不是「类目 → 商品」的捷径。

```python
# 只保留三个数值特征, 其余列丢掉
feature_cols = ['query', 'click_time', 'query_time']
X = X[feature_cols]

# sku 是我们的预测目标
y = train_data['sku']

# 测试集也要做同样的清洗, 才能喂给同一个模型
test_X = test_data.copy()
test_X['click_time'] = test_X['click_time'].apply(to_int_time)
test_X['query_time'] = test_X['query_time'].apply(to_int_time)
test_X['query'] = test_X['query'].str.lower()
# 注意: 测试集里出现训练集没见过的词时, 统一映射成一个「未知」编号
test_X['query'] = test_X['query'].map(query_dict).fillna(len(query_dict)).astype(int)
test_X = test_X[feature_cols]
```

### 2.3 模型训练与超参数调优

接着我们划分数据、选模型。我们从训练集里切出 25% 作为验证集，固定随机种子保证结果可复现；模型选随机森林——它由很多棵决策树组成，每棵树看一部分样本、一部分特征，最后大家一起「投票」决定预测结果。这种「人多嘴杂」的机制让它对离群值不敏感、也不需要做归一化，很适合我们这种「特征少、类别多、还带几个大整数」的表格数据：

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 75% 训练, 25% 验证; random_state 固定, 保证每次切分一样
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# 随机森林基线: 20 棵树, 先把流程跑通
clf_rf = RandomForestClassifier(n_estimators=20)
clf_rf.fit(X_train, y_train)
```

基线跑通后，我们做网格搜索。需要调的是三个参数：树的最大深度（太深会记住训练集、太浅学不动）、叶节点最少样本数（控制叶子不要太碎）、内部节点再分裂的最少样本数。我们用 5 折交叉验证，让每一组参数都被公平地评估 5 次：

```python
from sklearn.model_selection import GridSearchCV

# 每个参数给一串候选值, 让网格搜索自己挑
parameters = {
    'max_depth': range(9, 20, 2),            # 树深 9, 11, ..., 19
    'min_samples_leaf': range(30, 101, 20),  # 叶子最少 30~90 个样本
    'min_samples_split': range(10, 101, 10), # 内部节点最少 10~100 个样本
}

# 5 折交叉验证; n_jobs=-1 表示用满所有 CPU 核心
grid_search_cv_clf = GridSearchCV(
    RandomForestClassifier(n_estimators=20),
    parameters, cv=5, n_jobs=-1)

grid_search_cv_clf.fit(X_train, y_train)

# 交叉验证得分最高的那组参数, 拿来当最终模型
best_clf = grid_search_cv_clf.best_estimator_
print(grid_search_cv_clf.best_params_)
```

搜索结果把树深限制在中间档、叶子保持一定大小——这和我们前面的观察一致：商品类别太碎，叶子切得太细只会把训练集里的噪声也背下来，验证集反而更差。

### 2.4 模型测试与指标报告

最后我们在没参与训练的 25% 验证集上打分。评分方式就是最简单的准确率：

```python
# 用最终模型对验证集预测, 并计算准确率
score = best_clf.score(X_test, y_test)
print('验证集准确率:', score)   # 0.36716389728096677
```

我们得到约 **0.367** 的准确率——大约每三次预测能猜中一次。这个数字看起来不高，但要放到任务背景里理解：候选商品成千上万，随机猜测的期望准确率接近 0；很多商品在训练集里只出现过一两次，模型根本没有机会学会它们，这部分样本相当于「不可能得分」。刨掉这些之后还能拿到三成多，说明模型确实抓住了「搜索词 → 商品」的主要对应关系，而不是靠运气。

确认模型可用后，我们对测试集做同样的预测，并按比赛要求写出提交文件：

```python
# 测试集已经过和时间、搜索词一样的清洗
y_pred = best_clf.predict(test_X)

# 比赛要求的提交格式: 只有一列, 列名 Sku
output = pd.DataFrame({'Sku': y_pred})
output.to_csv('sku_predict.csv', index=False)

# 最后核对一遍: 行数必须和测试集一致, 列名必须是 Sku
print(output.shape, test_data.shape[0])
```

我们要特意强调这一步的核对：黑客马拉松的提交系统只认固定格式，行数少一行、列名写成 `sku` 小写，成绩都会被判无效——这类错误比模型差几个百分点更致命。

---

## 三、总结

本文中我们完成了一场小规模黑客马拉松的完整流程。我们首先读了训练集和测试集，确认这是一个 42365 行的「搜索→点击」多分类问题，并发现搜索词约 960 种、商品点击分布极不均衡。其次我们做了数据清洗：把长短不一的时间字符串补成微秒格式后压成整数，把搜索词统一小写并按词表编号，只保留 query、click_time、query_time 三列作为特征，同时把同样的清洗套用到测试集上。接着我们用随机森林做基线，再用 5 折网格搜索调整树深、叶节点和分裂阈值。最后我们在验证集上得到约 0.367 的准确率，并按要求的格式输出了 sku_predict.csv。

这个比赛给我们的启发是：**当特征少、类别多时，把「格式问题」处理干净（时间能不能比大小、同一个词有没有被拆成两个）往往比换更复杂的模型更有效。**