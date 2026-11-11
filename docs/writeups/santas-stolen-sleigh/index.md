# 圣诞老人的被盗雪橇（Santas Stolen Sleigh）

## 一、比赛简介

### 1.1 比赛背景
圣诞老人把礼物分别放在全球各地的坐标点上，雪橇被偷走后需要重新规划投递路线。比赛给出的训练数据是 gifts.csv，里面记录了每件礼物的编号、收货位置的纬度、经度以及重量。材料中出现的北极坐标是 (90,0)，所有行程都从北极出发，最后必须回到北极。礼物的地理分布覆盖全球，重量差异较大，投递顺序直接影响驯鹿的累积负担。我们从用户提交的代码里看到，常见的思路是先把相近位置的礼物聚在一起，再在每个小区域内部按重量做贪心装箱，最后用局部搜索调整停靠顺序。题目本质上是一个带容量限制的带权旅行商变种，因为每条路程的负担会随着已交付礼物重量减少而动态变化，这让简单的最近邻策略不够理想。

### 1.2 比赛目标
给定 gifts.csv 中的 10 万条礼物记录，要求为每件礼物分配一个 TripId。提交文件只需要两列 TripId 和 GiftId。同一行程中所有礼物的重量之和不能超过 1000，这一点在所有材料里都被反复检查。每个行程从北极出发，按分配的顺序依次访问礼物点，最后返回北极。代码里统一使用 haversine 计算两点间的球面距离，sleigh_weight 固定为 10，表示空载雪橇的重量。目标不是最小化总旅行距离，而是最小化距离与当前载重的乘积之和，这意味着越重的礼物越应该靠近北极交付，轻的礼物可以放在行程末尾。

### 1.3 评价标准
比赛使用加权驯鹿疲劳度 Weighted Reindeer Weariness 作为指标。它的计算方式是累加每段路程距离乘以当前载重。材料中的实现先把行程结束时加上一次返回北极的空载权重，再在遍历停靠点时逐步减去已交付的礼物重量。指标越小越好，负担越轻。材料里提到的 Top1 基准是 12389228461.1，所有本地方案都以这个数值作为参照。理解这个指标时可以把每一段路程想成一次体力消耗，载重越重、距离越长，消耗就越大，所以把重礼物安排在行程前段会比安排在后段更吃亏。我们在后续所有实验里都用同一个函数计算分数，保证不同方案之间可以横向比较。

## 二、解题过程

整场比赛的思路可以概括成一句话：先把礼物在空间上聚类以保证路线局部性，再用重量限制做贪心装箱，最后用邻居交换优化停靠顺序。我们按数据探索、数据清洗、模型训练与超参数调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先把输入文件读进来，确认数据规模和列的含义。材料里多次出现 gifts.shape 为 (100000, 4)，说明总共有 10 万件礼物。

```python
import pandas as pd
import numpy as np
# 读取礼物数据，GiftId、Latitude、Longitude、Weight 四列
df = pd.read_csv('../input/gifts.csv')
# 读取样本提交，熟悉 TripId 与 GiftId 的对应格式
sample = pd.read_csv('../input/sample_submission.csv')
# 打印整体形状，确认数据量
print('数据形状:', df.shape)
# 查看尾部几行，肉眼确认数值范围
print(df.tail(5))
# 简单统计重量，判断是否需要归一化
print(df['Weight'].describe())
```
这段代码是我们在本地复现 EDA 时的起点。通过 tail 可以看到 GiftId 最大到 100000，Latitude 在 -86 到 40 之间，Longitude 跨度很大，Weight 最小为 1，最大接近 11。没有缺失值的提示，基本可以认为数据是完整的。重量分布呈现右偏，绝大多数礼物重量集中在 1 到 5 之间，少数较重的礼物会对装箱产生较大影响。我们还注意到经纬度覆盖范围接近全球，这意味着简单的按经度排序无法保证空间连续性，必须借助聚类来压缩行程长度。

接着我们统计地理分布的直观形态，代码里常用散点图把经度纬度以重量为大小画出来。

```python
import matplotlib.pyplot as plt
# 为了可视化，把重量映射到点大小
plt.figure(figsize=(12, 8))
plt.scatter(df['Longitude'], df['Latitude'], s=df['Weight']*5, alpha=0.3)
plt.xlabel('经度')
plt.ylabel('纬度')
plt.title('礼物地理分布与重量')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()
```
散点图显示礼物遍布全球，但密度在低纬度区域更高。重量分布相对均匀，没有极端异常值。这一观察让我们决定后续先用空间聚类把相近的点圈在一起，减少不必要的跨洋飞行。南北半球的分布不均衡提示我们可以先按纬度做粗略分区，再在每个分区内部做精细聚类，这样可以避免 KMeans 在全球尺度上被极点拉偏。

### 2.2 数据清洗

其次我们检查是否满足重量限制以及坐标的有效性。材料里的验证函数会在发现超重的 Trip 时直接抛出异常。

```python
north_pole = (90, 0)
weight_limit = 1000
sleigh_weight = 10
# 检查合并后的表是否每条 Trip 重量超限
def check_weight(all_trips):
    sums = all_trips.groupby('TripId')['Weight'].sum()
    # 若任意 Trip 超过限制就报错
    if any(sums > weight_limit):
        raise Exception('存在超重行程')
    return True
# 确认纬度经度在合法范围内
df['Latitude'] = df['Latitude'].clip(-90, 90)
df['Longitude'] = df['Longitude'] % 360
```
清洗的过程非常轻。我们没有做文本清洗，也没有归一化坐标，因为距离计算依赖真实经纬度。唯一的硬性约束是重量上限，所有后续的装箱逻辑都围绕 1000 这个阈值展开。材料里多次出现的验证函数提醒我们，在任何一次拆分或合并行程后，都必须重新检查重量和，避免产生不可提交的解。坐标的合法化处理主要是为了防止极少数记录的经度超出 -180 到 180 范围导致 haversine 计算异常。

清洗后我们还保留了原始坐标，不做平移或缩放，因为 haversine 直接使用经纬度更准确。极短的距离或极大的距离都会被权重放大，所以空间聚类的效果比传统的数据标准化更重要。这个决定和一般机器学习流程完全不同：通常我们会对地理坐标做中心化和缩放，但这里一旦缩放就会破坏球面距离的比例关系，反而误导聚类结果。我们宁可让经纬度的量纲保持原始状态。

### 2.3 模型训练与超参数调优

接着我们进入路线构建。第一步是空间聚类，材料里常见的写法是使用 KMeans 把礼物分成 1600 个簇，然后在每个簇内按重量贪心装箱。

```python
from sklearn.cluster import KMeans
# 只用经纬度做聚类，保持原始度量
usecols = ['Latitude', 'Longitude']
kmeans = KMeans(n_clusters=1600, random_state=0, n_jobs=-1, tol=1e-3)
kmeans.fit(df[usecols])
df['cluster'] = kmeans.predict(df[usecols])
# 贪心装箱：同一簇内按原始顺序累加重量，超过1000就开启新行程
trip_id = 0
weight_packed = 0
trip_ids = []
for _, row in df.iterrows():
    if weight_packed + row['Weight'] > weight_limit:
        trip_id += 1
        weight_packed = row['Weight']
    else:
        weight_packed += row['Weight']
    trip_ids.append(trip_id)
df['TripId'] = trip_ids
```
这段代码体现了我们的核心假设：空间相近的礼物放在同一行程里，行驶距离会更短。超参数 n_clusters 决定了聚类粒度，材料里尝试 1600 得到约 3010 个行程，平均每程约 33 件礼物。聚类后再做贪心装箱的好处是保证了每个行程的地理集中性，避免把相隔半个地球的礼物硬塞进同一个行程。缺点是贪心装箱可能在单簇内部产生重量利用率不均的情况，我们后续会用局部交换来补救。

为了进一步降低疲劳度，我们在每条行程内部做邻居交换优化。材料里的 bb_sort 通过局部交换两相邻停靠点来降低路径成本。

```python
from haversine import haversine
def path_opt_test(order):
    # 计算一条顺序的累加疲劳度
    total = 0.0
    load = sum(w for _, w in order) + sleigh_weight
    prev = north_pole
    for _, loc, w in order:
        d = haversine(prev, loc)
        total += d * load
        load -= w
        prev = loc
    total += haversine(prev, north_pole) * sleigh_weight
    return total
# 简单的相邻交换搜索，提升局部顺序
def improve_order(lst):
    improved = True
    while improved:
        improved = False
        for i in range(len(lst)-1):
            if path_opt_test(lst) > path_opt_test(lst[:i]+[lst[i+1],lst[i]]+lst[i+2:]):
                lst[i], lst[i+1] = lst[i+1], lst[i]
                improved = True
    return lst
```
超参数调优主要集中在聚类数量和交换迭代次数上。我们没有做网格搜索，因为字符级特征在这里不适用，空间聚类的计算成本更高。经验上 1600 个簇配合重量贪心已经能得到一个合理的基准，后续的局部交换负责把分数再压下去。交换策略只动相邻两点，是因为在大规模行程上全局重排序成本太高，而相邻交换可以在保持总体结构的前提下逐步找到局部最优。

### 2.4 模型测试与指标报告

然后我们用材料中的加权疲劳度函数对完整方案进行评估。

```python
def weighted_trip_length(stops, weights):
    # 把行程结束的返回北极加入计算
    tuples = [tuple(x) for x in stops.values]
    tuples.append(north_pole)
    weights = list(weights) + [sleigh_weight]
    dist = 0.0
    prev_stop = north_pole
    prev_weight = sum(weights)
    for loc, w in zip(tuples, weights):
        dist += haversine(loc, prev_stop) * prev_weight
        prev_stop = loc
        prev_weight -= w
    return dist
def weighted_reindeer_weariness(all_trips):
    # 检查超重并累加所有行程疲劳度
    if any(all_trips.groupby('TripId')['Weight'].sum() > weight_limit):
        raise Exception('超重')
    total = 0.0
    for t in all_trips['TripId'].unique():
        trip = all_trips[all_trips['TripId']==t]
        total += weighted_trip_length(trip[['Latitude','Longitude']], trip['Weight'])
    return total
score = weighted_reindeer_weariness(df)
print('当前分数:', score)
```
在本地跑通的 KMeans 贪心方案得到 Cur 约 18148803018.1，Top1 基准为 12389228461.1，相对差距约为 -46%。这说明空间聚类加贪心装箱已经能产生可行解，但离最优还有较大空间。材料里后续的改进是通过更精细的网格划分和局部交换把数值逐步压低。我们最后会把 TripId 与 GiftId 写成 submission.csv，并再次检查每条 Trip 的重量和行数是否与样本格式一致。为了避免提交错误，我们还会随机抽取几条行程手动计算疲劳度，对比代码输出。

## 三、总结

本文中我们完成了圣诞老人被盗雪橇比赛的完整流程。我们首先用 EDA 摸清了数据规模和地理分布：10 万件礼物，坐标覆盖全球，重量在 1 到约 11 之间，重量上限为 1000。其次我们做了最少量的数据清洗，只做坐标范围校验和超重检查，不做任何会破坏地理信息的变换。接着我们用 KMeans 空间聚类配合重量贪心装箱构建初始行程，再用邻居交换做局部顺序优化，超参数主要围绕聚类数量和交换迭代次数。最后我们在本地用加权驯鹿疲劳度评估方案，得到一个可行解并与 Top1 基准对比，验证了模型的可行性。我们在实验中发现，即使是简单的聚类加装箱，只要保证每个行程的重量合规，就能得到一个可以提交的基准解。

这个比赛给我们的最大启发是：路线优化问题里，特征工程要跟着几何走。空间邻近性比单纯的重量均衡更能降低累积疲劳，评估时必须考虑载重随交付递减的动态特性，否则会高估实际成本。
