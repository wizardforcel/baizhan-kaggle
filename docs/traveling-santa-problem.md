# Traveling Santa Problem

## 一、比赛简介

### 1.1 比赛背景

Traveling Santa Problem 是圣诞老人送礼物的旅行商问题变体。圣诞老人需要从北极出发拜访所有城市，每座城市只访问一次，最后返回起点。每个城市有坐标 x、y，目标是最小化总行程距离。

这个问题经典的 TSP 难以精确求解，特别是当城市数量达到 15 万时。比赛鼓励使用启发式算法如最近邻、遗传算法、2-opt 等来构造近似最优路径。

### 1.2 比赛目标

给定 santa_cities.csv 包含城市 id 和坐标，输出一个访问所有城市的路径序列，从 id 0 开始，依次列出所有城市 id，最后返回起点。路径长度即总行程距离。

训练集无标签，比赛提供基准路径用于对比。目标是比随机路径更短的距离。

### 1.3 评价标准

比赛使用总行程距离作为评价指标，距离越小越好。以随机基准为参考，进步越大越好。没有明确的分数阈值，排行榜按距离排序。

## 二、解题过程

整场比赛的思路可以概括成一句话：先可视化城市分布，再用最近邻构造初始路径，最后用局部优化降低距离。我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步是读取城市数据并观察分布：

```python
import pandas as pd
df = pd.read_csv('/kaggle/input/traveling-santa-problem/santa_cities.csv')
print(df.shape)
```

城市总数约 15 万，坐标范围较大。我们绘制散点图：

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(16,10))
plt.plot(df.x, df.y, 'k,', alpha=0.3)
plt.plot(df.x[0], df.y[0], 'bx')
plt.title('All cities (first in blue)')
plt.show()
```

城市分布相对均匀，没有明显的聚类。起点在坐标系左上区域。

### 2.2 数据清洗

数据本身干净，无缺失值。清洗主要是计算距离函数：

```python
def total_distance(df_path):
    dist = (np.sqrt((df_path.x - df_path.x.shift())**2 +
                    (df_path.y - df_path.y.shift())**2)).sum()
    return round(dist,2)
```

我们将路径 id 合并到坐标表，方便计算距离。清洗时确保路径包含所有城市且不重复。

### 2.3 模型训练与超参数调优

我们首先实现最近邻启发式：

```python
def nearest_neighbor(df):
    unvisited = set(df.id)
    current = 0
    route = [current]
    unvisited.remove(current)
    while unvisited:
        next_city = min(unvisited, key=lambda c: 
                        ((df.loc[c,'x']-df.loc[current,'x'])**2 +
                         (df.loc[c,'y']-df.loc[current,'y'])**2)**0.5)
        route.append(next_city)
        unvisited.remove(next_city)
        current = next_city
    route.append(0)
    return route
```

最近邻很快但质量一般。接着我们用 2-opt 局部优化：

```python
def two_opt(route, df):
    improved = True
    while improved:
        improved = False
        for i in range(len(route)-2):
            for j in range(i+2, len(route)):
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                if total_distance(merge(new_route)) < total_distance(merge(route)):
                    route = new_route
                    improved = True
    return route
```

超参数主要是迭代次数，受计算资源限制。我们在小样本上先验证算法有效性。

### 2.4 模型测试与指标报告

生成路径后，我们计算总距离并绘图：

```python
route = nearest_neighbor(df)
df_path = pd.merge(pd.DataFrame({'id':route}), df, on='id')
print('Distance:', total_distance(df_path))
plot_path(route, df)
```

本地测试显示最近邻距离约数百万公里，具体数值取决于城市分布。2-opt 后距离显著下降。我们把最终路径保存为提交文件，格式为 id 序列。

## 三、总结

本文中我们完成了 Traveling Santa Problem 的完整流程。我们首先通过 EDA 了解城市分布，接着实现距离计算函数。然后我们用最近邻构造初始路径，再用 2-opt 进行局部优化。最后我们计算总距离并生成提交文件。

这个比赛给我们的启发是：大规模 TSP 无法精确求解，启发式构造加局部优化是实用策略，路径可视化有助于发现改进空间。
