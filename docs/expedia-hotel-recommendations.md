# 亿客行酒店推荐（Expedia Hotel Recommendations）

## 一、比赛简介

### 1.1 比赛背景

Expedia（亿客行）是全球知名的在线旅游网站。这个比赛给出一段真实的**搜索-点击/预订日志**：用户在 2013~2015 年间发起酒店搜索（多少成人、多少孩子、住几晚、去哪儿），并最终点击或预订了某些酒店。酒店海量，没法一个个精确预测，主办方把相似酒店归并成 **100 个酒店簇（hotel_cluster）**，问题就变成「给这次搜索预测它最可能成交的几个簇」。

数据量很大：`train.csv` 有 **3767 万** 行（`wc -l` 数出 37670294），包含 23 个原始字段；`test.csv` 是需要预测的搜索记录；`destinations.csv` 提供目的地（destination）的向量表示。

这个比赛的经典坑是**数据泄漏（leakage）**：训练集里 `is_booking`（是否真实预订）和 `cnt`（这条记录被看了几次）这两列在测试集里是不存在的，但它们和成交强相关。如果直接拿它们当特征，本地验证分会爆表，上线却直接失效。

### 1.2 比赛目标

输入是一次搜索的所有上下文（站点、用户位置、渠道 channel、入住/退房日期 `srch_ci/srch_co`、人数、目的地 `srch_destination_id`、酒店国家 `hotel_country`、酒店市场 `hotel_market` 等），输出该次搜索最可能的 **5 个酒店簇**，并按可信度从高到低排列。

### 1.3 评价标准

比赛用 **MAP@5（Mean Average Precision at 5）** 打分，通俗说：对每次搜索，看真实成交的酒店簇在你给的 Top-5 里排在第几位，**排得越靠前得分越高**。下面这段演示算得非常清楚：

```python
import ml_metrics as metrics
actual = [1]
# 正确答案 1 排在第一位 → AP@5 = 1.0
metrics.apk(actual, [1,2,3,4,5], 5)   # 1.0
# 正确答案 1 被排到第 2 位 → AP@5 只剩一半
metrics.apk(actual, [2,1,3,4,5], 5)   # 0.5
# 被排到第 5 位 → 只有 0.2
metrics.apk(actual, [4,2,3,5,1], 5)   # 0.2
```

MAP@5 就是所有测试搜索的 AP@5 再取平均。它同时惩罚「没命中」和「命中但排太后」，满分 1.0，越高越好。

当一次搜索的成交簇有多个时，计算规律相同：所有答案都在前几名的，即使顺序换来换去也照样满分，但一旦被挤到后面就开始掉分：

```python
actual = [1, 2]
metrics.apk(actual, [1, 2, 3, 4, 5], 5)   # 两个答案都在前三 → 1.0
metrics.apk(actual, [2, 1, 3, 4, 5], 5)   # 顺序调换不惩罚 → 1.0
metrics.apk(actual, [3, 2, 1, 4, 5], 5)   # 一个被挤到第 3 → 0.583
metrics.apk(actual, [4, 2, 3, 5, 1], 5)   # 排太后 → 0.45
```

这解释了为什么很多提交把「最热门的簇」排在前面：MAP@K 对「答案是否落在前 5」高度敏感，热门兜底能稳定地命中一两个答案，比追求完美排序稳妥得多。

---

## 二、解题过程

我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步走。主线选用**基于计数的热门统计方案**——它在数据泄漏没法用、又没有强特征时，是性价比最高的起步模型，也被证明是很多高分队伍的基础。

### 2.1 数据探索（EDA）

数据有 3767 万行，直接全部读进内存不现实，先看前几行认识字段，再抽样分析：

```python
import pandas as pd

# 先看列名和数据长什么样
!head -n 5 ../input/train.csv
# date_time,site_name,posa_continent,user_location_country,...,
# srch_destination_id,srch_destination_type_id,is_booking,cnt,
# hotel_continent,hotel_country,hotel_market,hotel_cluster

# 采样 10 万行快速探索
data = pd.read_csv("../input/train.csv", nrows=100000)
data['hotel_cluster'].value_counts().plot.bar()
# 画出来像一条往下掉的幂律尾巴：
# 少数热门簇占了大部分点击，大部分簇无人问津
```

如果不想只取前 N 行、而是想抽到「特定用户」的子集，可以用**分块迭代读取**——把大文件按每 1000 行一块地读，边读边筛，最后拼起来：

```python
# 只读 user_id 一列就够采样了，省内存
users = pd.read_csv(train_path, usecols=['user_id'])['user_id'].unique()
select_users = np.random.choice(users, 100000, replace=False)

# 分块读整个大文件，只保留这些用户的行
iter_csv = pd.read_csv(train_path, iterator=True, chunksize=1000)
select = pd.concat([chunk.loc[chunk.user_id.isin(select_users)]
                    for chunk in iter_csv])
```

这样既不用把 3767 万行全部载入内存，又能拿到一个「覆盖特定用户群」的代表性子集来做画图、分布分析。

接下来我们用 **卡方检验（chi2）** 粗筛，看哪些离散特征和酒店簇相关。办法是先把类别列独热（get_dummies），再对每列算卡方得分，取 Top-100 特征统计它们的来源：

```python
from sklearn.feature_selection import chi2

CATEGORICALS = ["site_name", "posa_continent", "user_location_country",
                "user_location_region", "user_location_city", "is_mobile",
                "is_package", "channel", "srch_destination_type_id",
                "hotel_continent", "hotel_country", "hotel_market",
                "srch_destination_id"]

data_cat_dummies = pd.get_dummies(data[CATEGORICALS].astype('category'))
sample_n = 10000   # 特征列很多，卡方检验只用 1 万行采样即可
chi2_scores = chi2(data_cat_dummies.sample(sample_n),
                   data['hotel_cluster'].loc[data_cat_dummies.sample(sample_n).index])
```

得到的 Top-100 特征来源统计说明：**酒店市场、搜索目的地、酒店国家、用户所在城市** 这四类字段与酒店簇关系最密切，是必须进模型的核心变量。

### 2.2 数据清洗

清洗分三件事：

第一是**防泄漏**。EDA 明确把 `is_booking`、`cnt` 标成泄漏列——它们是「结果」不是「原因」，测试集根本没有。所以训练模型时必须把它们丢掉，只保留搜索侧字段：

```python
# 泄漏列：如果保留，本地交叉验证会虚高到难以置信
drop_cols = ['is_booking', 'cnt']
train = train.drop(columns=drop_cols, errors='ignore')
```

第二是处理**日期字段的脏数据**。主题探索中发现部分 `srch_ci`（入住日期）明显错乱（比如 2016 年以后、或年份在 2013 之前），统一回填成 `2015-01-01`，避免把模型时间轴打乱：

```python
def frame_date_deal(frame, datename):
    # 缺失和越界的日期统一塞到 2015-01-01
    frame[datename] = frame[datename].fillna('2015-01-01')
    return frame
```

第三是**给每次搜索打上时间季**。把日期按 4 个月的季度归组，作为分特征，捕捉「旺季/淡季」的周期影响：

```python
# 把 2013-2016 的日期按季度映射成 0~11 的整数季度号
def month_to_quarter(month):      # 伪代码示例
    return (month - 1) // 4
```

### 2.3 模型训练与超参数调优

第一版模型是**计数统计基线（Leakage solution 思路）**:直接统计「在给定条件下，哪些酒店簇最常成交」，提交时按热度取 Top-5。我们维护几套字典：按「用户城市 × 出发地距离」、按「目的地 × 酒店国家 × 酒店市场」、按「目的地」、按「酒店国家」，分别统计每个酒店簇的加权出现次数，其中 **真实预订（is_booking=1）加权 20 次、普通点击加权 3 次**，最后再叠加一个全站人气的兜底字典：

```python
from collections import defaultdict
from heapq import nlargest

best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))  # 城市×距离 → 簇计数
best_hotels_search_dest = defaultdict(lambda: defaultdict(int))  # 目的地×国家×市场
best_hotel_country = defaultdict(lambda: defaultdict(int))       # 仅酒店国家
popular_hotel_cluster = defaultdict(int)                          # 全站人气

# 统计：预订权重远大于普通点击，因为预订才是「真成交」
for line in train_file:
    is_booking = int(arr[18])
    append_1 = 3 + 17 * is_booking      # 预订 +17 → 权重 20，点击权重 3
    if srch_destination_id != '':
        best_hotels_search_dest1[srch_destination_id][hotel_cluster] += append_1
    if hotel_country != '':
        best_hotel_country[hotel_country][hotel_cluster] += append_2
    popular_hotel_cluster[hotel_cluster] += 1
```

预测时按优先级依次填充 Top-5：先查「城市×距离」，再查「目的地×国家×市场」，再退到「目的地」、再到「酒店国家」，最后用全站最热门的簇补底，保证每行都能凑满 5 个不重复的簇：

```python
# 生成提交：按上面四档条件依次取簇，直到凑满 5 个
for each test row:
    filled = []
    for source in (best_hotels_od_ulc, best_hotels_search_dest,
                   best_hotels_search_dest1, best_hotel_country):
        top = nlargest(5, source.get(key, {}).items())   # 取热度最高 5 个
        for cluster, _ in top:
            if cluster not in filled and len(filled) < 5:
                filled.append(cluster)
    # 还差的用全站人气兜底
```

样本量超大是这版方案的另一个压力：3767 万行无法一次性放进内存，所以统计阶段用**逐行流式读取**（`while` 循环 + `defaultdict` 累加），既能控制内存，也天然是常数级时间。

换更强的模型时，材料中的实验路线是：对 100 个簇做**一对多 + 投票**。为每对簇 (i, j) 训练一个小的梯度提升分类器（`GradientBoostingClassifier`，约 20~30 棵树、深度 2），100 个簇就会得到一组「簇 vs 簇」的分类器，预测时多个分类器投票给出每个簇的归属概率，再按概率排序取 Top-5。效果上限更高，但训练成本是 100 倍的分类器。材料里这份尝试因数据拆分方式有误（测试里混入了真实标签做调参）而失败，提醒我们：**切分数据时要确保验证集真的没被模型见过**。

### 2.4 模型测试与指标报告

我们在训练集尾部切出最近一段作验证：统计阶段的字典只从早期数据构建，验证段只用于打分，用 `ml_metrics.mapk` 计算 MAP@5：

```python
import ml_metrics as metrics

# preds: 每个验证搜索预测的 5 个簇 ['1 2 3 4 5' 字符串]
# actual: 真实成交簇列表
score = metrics.mapk(actual, preds, 5)
print("MAP@5:", score)
```

计数基线通常能拿到 **0.40 上下的 MAP@5**，已经显著高于随机猜（100 选 5 的随机期望几乎为 0），是后续一切模型的地基。再往上提分主要靠两点：用矩阵分解把 `destinations.csv` 的目的地向量压成隐特征融入模型，以及用 XGBoost/LightGBM 等树模型在「热门统计 + 结构化特征」上做多分类（100 类）。

探索材料里还列过三类「候选解法」，虽然没跑通但思路值得记录：一是**基于内容的过滤**——把用户点过的酒店簇在目的地隐空间里找最近邻，推荐最近的 5 个；二是**协同过滤**——做用户 × 酒店簇的矩阵分解，用内积打分；三是**因子分解机（Factorization Machine）**——把全体离散、数值特征做特征交叉后由因子分解机统一打分，兼顾上面两种。这三条路都需要先把 23 个原始字段压缩成稠密向量，和我们的计数基线互补，是后续融合或替换模型时的主要候选。

材料里没有给出最终的公开榜分数，我们的处理方法是在本地用同一口径的 MAP@5 反复对比，分数不涨就不换方案。报告时除了总榜分，还会按搜索是否真实预订拆开看 MAP@5——**预订样本的命中通常比纯点击样本难**，这个差距能直观反映模型在「硬样本」上的表现。

---

## 三、总结

本文中我们完成了 Expedia 酒店推荐比赛的完整流程。我们首先用 EDA 认识了 3767 万行的搜索日志，发现酒店簇呈幂律分布，并用卡方检验筛出了「酒店市场、目的地、酒店国家、用户城市」等核心特征；其次做了数据清洗，重点是**丢弃 `is_booking`、`cnt` 两个泄漏列**，并回填脏日期、离散化时间；接着记录了两种建模路线——计数统计热门方案（预订权重 20、点击权重 3，多级兜底凑 Top-5）和基于簇对投票的梯度提升方案；最后用 MAP@5 在时间切分的验证集上评估。

这个比赛给我们的最大启发是：**防泄漏是这类比赛的第一课**。`is_booking`、`cnt` 这两列能让本地分数漂亮得离谱，但它们本质上属于「未来信息」；真实竞赛里拿它当特征，榜上会立刻原形毕露。判断一个特征是否泄漏，只需问一句：**预测的时候我手里的表里真的有这一列吗？**