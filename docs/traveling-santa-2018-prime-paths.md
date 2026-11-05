# Traveling Santa 2018: Prime Paths

## 一、比赛简介

### 1.1 比赛背景

Traveling Santa 是 Kaggle 在 2018 年圣诞节前后举办的一场路线优化比赛。它的底子是旅行商问题：给出一堆城市，要求找一条从起点出发、经过每个城市一次、最后回到起点的最短路线。旅行商问题是经典的 NP-Complete 问题，城市一多，穷举所有排列根本不现实。

比赛在这个基础上加了一个“质数转折”。城市编号如果是质数，房子会给驯鹿留下胡萝卜。材料里写得很清楚：如果驯鹿团队没有每 10 步访问一次质数城市，下一段路会比平常多花 10%。换成代码的说法，就是每第 10 次移动，如果这段路的起点城市不是质数，距离要乘 1.1。

城市坐标不是常见的经纬度，而是人为设计的一张平面图。把 197,769 个点画出来，能看到驯鹿和圣诞树的轮廓。城市 0 就是北极点，路线从它出发，最后也要回到它。

### 1.2 比赛目标

输入文件只有一张表 `cities.csv`，包含三列：`CityId`、`X`、`Y`。`CityId` 从 0 连续编号到 197,768，其中 0 是北极点。每一行代表一个城市，`X` 和 `Y` 是它在平面上的坐标。

输出文件需要一列 `Path`，按顺序写出要访问的城市编号。路线必须从 0 开始，访问所有城市且每个城市只出现一次，最后再回到 0。比赛材料里的读取和保存代码都会检查 `tour[0] == 0` 和 `tour[-1] == 0`。

### 1.3 评价标准

评价标准是整条路线的总距离加上质数惩罚。几何距离用两点之间的欧氏距离累加；每当步数是 10 的倍数时，如果这一步的起点城市不是质数，就把这一段距离乘 1.1，也就是额外多算 10%。

用公式表达就是：总分 = 所有路段长度之和 + 0.1 × 所有“第 10 步且起点非质数”的路段长度之和。分数越低越好。材料里的实现大多写成 `sum(distances) + 0.1 * sum(penalties)`，其中 `penalties` 只收集那些触发惩罚的路段。

惩罚项占总距离的比例很小，通常不到 1%。这意味着先把路线几何长度做短，比一开始就死磕质数间隔更划算。

## 二、解题过程

我们按数据探索、数据清洗、模型训练与超参数调优、模型测试与指标报告四步来走。下面每一小节都是先放一段代码，再用大白话解释我们为什么这么做。

### 2.1 数据探索（EDA）

我们首先把城市表读进来，确认数据规模和字段含义。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读入城市表：CityId 是城市编号，X/Y 是平面坐标
cities = pd.read_csv('../input/cities.csv')
print(cities.shape)      # 输出 (197769, 3)
print(cities.head())

# 把所有城市画成散点图；点设得很小，否则 19 万个点会糊成一片
cities.plot.scatter(x='X', y='Y', s=0.07, figsize=(15, 10))
# 城市 0 是北极点，单独用红点标出来
north_pole = cities[cities.CityId == 0]
plt.scatter(north_pole.X, north_pole.Y, c='red', s=20)
plt.show()
```

打印结果告诉我们，训练数据共有 197,769 行、3 列。散点图上的点并不是随机撒的，而是排成了一只驯鹿和几棵树的形状。城市 0 在图里单独用红点标出，位置就是整张图的“北极”。这一步让我们直观理解了题目：我们要在这张图上找一条绕遍所有点再回到红点的路线。

接着我们判断哪些城市编号是质数。

```python
def is_prime(n):
    """判断城市编号是不是质数：只能被 1 和自身整除"""
    if n < 2:
        return False
    i = 2
    # 只需试到根号 n，比它大的因子一定和更小的因子配对出现
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True

# 给每个城市打上“是否质数”的标记，后面算惩罚要用
cities['IsPrime'] = cities.CityId.apply(is_prime)

# 数一数质数城市和非质数城市各有多少
print(cities['IsPrime'].value_counts())
# False    179967
# True      17802
```

统计结果是 17,802 个质数城市、179,967 个非质数城市，质数大约占 9%。质数城市在散点图上分布得比较均匀，没有挤在某一个角落。材料里也提到“质数城市大约是十分之一”，这和“每 10 步需要访问一次质数城市”的规则正好对得上。

如果不想手写试除法，也可以用 `sympy` 现成的质数筛。材料里的遗传算法方案就是这么做的：

```python
from sympy import sieve

# sieve.primerange 会一次性列出 0 到城市总数之间的所有质数
pnums = list(sieve.primerange(0, cities.shape[0]))
# 用编号是否落在质数列表里，给每个城市打标记
cities['isprime'] = cities.index.isin(pnums)
print(len(pnums))       # 质数城市个数
```

这种写法比逐个调用 `is_prime` 更快，适合后面反复检查路线。`cities.shape[0]` 是 197,769，所以筛的范围正好覆盖所有城市编号。

### 2.2 数据清洗

这份数据本身很干净：没有缺失值，没有重复行，坐标也都是有效数字。我们没有做坐标归一化，也没有删掉任何城市，因为每个城市都必须被访问一次。

需要“清洗”的其实是评分规则。下面的函数把“每 10 步检查起点是否为质数，不是就乘 1.1”的逻辑写进了距离计算。

```python
def total_distance(route):
    """给定一条路线（城市编号列表），算总长度和惩罚带来的额外距离"""
    coords = cities[['X', 'Y']].values
    total = 0.0      # 纯几何长度
    extra = 0.0      # 因 10% 惩罚多出来的部分

    for i in range(1, len(route)):
        a, b = route[i-1], route[i]
        # 两点间欧氏距离
        d = np.hypot(coords[b,0]-coords[a,0], coords[b,1]-coords[a,1])

        # 每走 10 步检查一次：如果这一步的起点不是质数，距离要乘 1.1
        if i % 10 == 0 and not is_prime(a):
            extra += d * 0.1
            d *= 1.1

        total += d

    print('总距离:', total)
    print('惩罚占比: {:.3%}'.format(extra / total))
    return total, extra
```

我们特意把 `extra` 单独累加，这样能看出惩罚到底占多大比例。材料里的几个基线结果都显示，惩罚只占总距离的 0.85% 到 0.99% 左右。也就是说，如果为了迁就质数城市把路线绕远，多出来的几何距离很容易超过省下的惩罚，这也是后面很多方案“优化质数反而没提升”的原因。

我们还写了一个小检查，确保提交路线合法：

```python
def check_route(route, n_cities=197769):
    """检查路线是否从 0 出发、覆盖所有城市、最后回到 0"""
    assert route[0] == 0 and route[-1] == 0, '首尾必须是北极点'
    # 去掉最后回到 0 的重复点，剩下的应该恰好是 0 到 n_cities-1
    visited = route[:-1]
    assert len(visited) == n_cities, '城市数量不对'
    assert len(set(visited)) == n_cities, '有城市被重复访问或遗漏'
    return True
```

清洗阶段我们没有做任何会改变数据分布的操作，只是建立统一的评分和校验函数。后面所有实验都用同一把尺子量，结果才可比较。

### 2.3 模型训练与超参数调优

这一节其实是“路线求解器”的搭建和调参。我们先从最笨的基线开始，再逐步换成更聪明的启发式算法。

#### 最笨的基线和按 X 排序

```python
# 方案一：完全不动脑子，按 CityId 原顺序走，最后回到 0
route = list(cities.CityId) + [0]
total_distance(route)   # 总距离约 446884407.52

# 方案二：按 X 坐标从左到右扫一遍，减少来回横跳
temp = cities.drop(cities.index[0]).sort_values('X')
route = [0] + list(temp.CityId) + [0]
total_distance(route)   # 总距离约 196478628.47
```

按编号顺序走，总距离约 4.47 亿，惩罚约 379 万，占 0.85%。改成按 X 坐标排序后，总距离立刻降到约 1.96 亿，惩罚约 194 万，占 0.99%。这个对比很直观：编号顺序没有任何几何意义，而按 X 排序至少让路线从左扫到右，少了很多来回横跳。

#### 最近邻构造

```python
IDs = cities.CityId.values
coords = np.array([cities.X.values, cities.Y.values]).T

position = coords[0]       # 从北极点出发
route = [0]
IDs = np.delete(IDs, 0)
coords = np.delete(coords, 0, axis=0)

while len(IDs) > 0:
    # 计算当前位置到所有未访问城市的距离
    dist = np.linalg.norm(coords - position, axis=1)
    idx = dist.argmin()          # 挑最近的一个
    route.append(IDs[idx])       # 记进路线
    position = coords[idx]       # 走过去
    IDs = np.delete(IDs, idx)    # 已访问，从候选里删掉
    coords = np.delete(coords, idx, axis=0)

route.append(0)                  # 最后回到北极点
total_distance(route)            # 总距离约 1812602.19
```

最近邻的思路是：站在当前城市，永远走向最近的未访问城市。它把总距离从约 1.96 亿压到了约 181 万，提升了两个数量级。材料记录的运行时间约 4 分钟。这个结果说明，哪怕是最简单的贪心，只要顺着几何结构走，也能得到可用的初始解。不过材料也提到，这个成绩仍然比大约 85% 的参赛提交要差。

#### 带质数惩罚的最近邻

```python
cities['IsPrime'] = cities.CityId.apply(is_prime)
IDs = cities.CityId.values
coords = np.array([cities.X.values, cities.Y.values]).T
primes = cities.IsPrime.values

position = coords[0]
route = [0]
IDs = np.delete(IDs, 0)
coords = np.delete(coords, 0, axis=0)
primes = np.delete(primes, 0)

count = 0
while len(IDs) > 0:
    count += 1
    dist = np.linalg.norm(coords - position, axis=1)

    # 每第 10 步，把非质数城市的距离乘 1.10
    # 相当于提前把惩罚算进选择里，逼算法优先考虑质数城市
    if count % 10 == 0:
        dist[primes == False] *= 1.10

    idx = dist.argmin()
    route.append(IDs[idx])
    position = coords[idx]
    IDs = np.delete(IDs, idx)
    coords = np.delete(coords, idx, axis=0)
    primes = np.delete(primes, idx)

route.append(0)
total_distance(route)   # 总距离约 1812550.50
```

把惩罚提前算进距离选择后，总距离从 1,812,602 降到 1,812,550，只改善了 52。材料里也记录了这个现象：总惩罚反而从 17,892 升到 18,004。原因不难理解，惩罚项本来就只有 1% 左右，强行在每第 10 步挑质数城市，可能让几何路线绕远，省下的惩罚抵不过多走的路。

#### 2-opt 局部搜索

有了初始路线，下一步是局部优化。材料里最常出现的是 Kostya Atarik 的 2-opt：如果路径中有两个城市离得很近，就尝试把这两点之间的片段反转，看看总分会不会下降。

```python
import numba
from math import sqrt
from sklearn.neighbors import KDTree

XY = np.stack((cities.X.values, cities.Y.values), axis=1).astype(np.float32)
is_not_prime = np.array([0 if is_prime(i) else 1 for i in cities.CityId])

@numba.jit(nopython=True)
def cities_distance(offset, id_from, id_to):
    """算一段距离，offset 用来判断是不是第 10 步"""
    xy_from, xy_to = XY[id_from], XY[id_to]
    dx, dy = xy_from[0]-xy_to[0], xy_from[1]-xy_to[1]
    d = sqrt(dx*dx + dy*dy)
    # 每 10 步检查一次起点是否为质数
    if offset % 10 == 9 and is_not_prime[id_from]:
        return 1.1 * d
    return d
```

实际优化时，我们先用 KDTree 为每个城市找出最近的若干个邻居，组成“候选城市对”，然后只在这些对对应的路径片段上尝试反转。每次反转后重新算这一段的分，只有分数下降才接受。材料记录的初始分数约 1,516,693.57，经过多轮 2-opt 后降到约 1,516,424，改善约 270。这个数字看起来不大，但在 150 万级别的总分上，已经是实打实的进步。

2-opt 的关键是“只检查近邻”。如果对所有城市对都尝试反转，19 万个城市会产生天文数字的组合；KDTree 把候选缩小到每个城市附近的几十个点，才让搜索变得可行。

#### 遗传算法和聚类

除了局部搜索，材料里还有队伍尝试遗传算法。思路是先把城市聚类成若干区域，再用遗传算法在区域之间和区域内部搜索路线。

```python
from sympy import sieve
from sklearn.mixture import GaussianMixture

# 先拿到所有质数城市编号
pnums = list(sieve.primerange(0, cities.shape[0]))
cities['isprime'] = cities.index.isin(pnums)

# 用高斯混合模型把地图切成 350 个簇
# 每个簇可以看成一个“片区”，减少遗传算法要处理的城市数量
mclusterer = GaussianMixture(
    n_components=350,
    tol=0.01,
    random_state=66,
    verbose=1
)
cities['mclust'] = mclusterer.fit_predict(cities[['X', 'Y']].values)
print(cities['mclust'].value_counts().head())
```

材料里的遗传算法方案把城市聚成 350 个簇，然后在簇的层面做交叉和变异。遗传算法需要调的主要是交叉率、变异率和种群大小：交叉率太高容易破坏已经不错的路段，太低又搜索不动；变异率太高会退化成随机游走，太低则容易卡在局部最优。这个方案的好处是能跳出 2-opt 的局部最优，缺点是要跑很多代，时间成本高。

#### Concorde 和坐标缩放

另一条路线是直接调用专业的 TSP 求解器 Concorde。Concorde 内部把距离四舍五入成整数，而我们的坐标带很多小数，直接传进去会损失精度。材料里的做法是把坐标整体乘 1000。

```python
from concorde.tsp import TSPSolver
import time

cities_scaled = cities.copy()
# Concorde 的 EUC_2D 会把距离取整，坐标乘 1000 可以保留更多小数信息
cities_scaled.iloc[:, 1:] *= 1000

solver = TSPSolver.from_data(
    cities_scaled.X,
    cities_scaled.Y,
    norm="EUC_2D"
)
t = time.time()
tour_data = solver.solve(time_bound=60.0, verbose=True, random_seed=42)
print('求解耗时:', time.time() - t)
print('是否找到路线:', tour_data.found_tour)
```

材料记录，坐标缩放把结果从 1,532,958.68 改善到了 1,524,756.18。这个提升说明，求解器本身很强，但输入精度不够会白白浪费它的能力。当然，Concorde 只优化纯几何距离，不会自动处理质数惩罚，所以拿到的路线还要用我们自己的评分函数检查一遍。

### 2.4 模型测试与指标报告

最后一步是把路线放到完整评分函数里算总分，并生成提交文件。

```python
def score(tour, percent=0.1):
    """完整评分：几何距离 + 每 10 步非质数起点的 10% 惩罚"""
    dists = []
    for i in range(1, len(tour)):
        x = cities.X.values[tour[i]] - cities.X.values[tour[i-1]]
        y = cities.Y.values[tour[i]] - cities.Y.values[tour[i-1]]
        dists.append(np.hypot(x, y))

    # 第 10、20、30... 段，如果起点不是质数，就计入惩罚
    penalties = [
        dists[i]
        for i in range(9, len(dists), 10)
        if not is_prime(tour[i])
    ]
    return sum(dists) + percent * sum(penalties)

final_score = score(tour)
print('最终分数:', final_score)

# 提交格式：一列 Path，首尾都是 0
pd.DataFrame({'Path': tour}).to_csv('submission.csv', index=False)
print('提交行数:', len(tour))
```

测试阶段我们主要看三个数：总分、惩罚占比、路线合法性。总分用来比较不同方案；惩罚占比用来判断质数约束有没有被合理满足；路线合法性用前面的 `check_route` 检查，确保 197,769 个城市一个不少、首尾都是 0。

从材料记录看，最笨的编号顺序约 4.47 亿，按 X 排序约 1.96 亿，最近邻约 181 万，带质数惩罚的最近邻约 181 万，2-opt 可以进入 151 万区间，第 8 名的方案用 cumsum 技巧把分数从 15162xx 进一步压到 15149xx。质数惩罚始终只占 1% 左右，说明真正拉开差距的还是几何路线的优化。

## 三、总结

本文中我们完成了 Traveling Santa 2018 Prime Paths 的完整探索流程。我们首先用 EDA 摸清了数据：197,769 个城市，其中 17,802 个是质数城市，坐标连起来构成驯鹿和树的图案。其次我们做了最少量的数据清洗，把质数判断和 10% 惩罚写进统一的评分函数，并用校验函数保证路线合法。接着我们从按编号顺序、按 X 排序、最近邻这些基线出发，逐步加入质数惩罚，再用 2-opt 局部搜索、遗传算法和 Concorde 求解器继续优化。最后我们在全量路线上重新评分并生成提交文件。

这个比赛给我们的最大启发是：规则要先吃透，优化要抓大放小。质数惩罚虽然有趣，但只占总分的 1% 左右；把几何路线做短，再在近邻对上做局部搜索，才是从百万级降到 150 万级的关键。遇到大规模组合优化问题时，先用简单启发式拿到一个能跑的初始解，再上 2-opt、k-opt 或专业求解器，比一上来就追求完美要现实得多。
