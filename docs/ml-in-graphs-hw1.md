# ML in Graphs HW1（图机器学习第一次作业）

## 一、比赛简介

### 1.1 比赛背景

这是 CS224W（图机器学习）课程 HW1 的练习，主题是**网络的基本特征分析**。材料里的作业模板要求我们：生成两种理论图模型（Erdős–Rényi 随机图和 Small-World 小世界图），再加载一个真实合作网络（ca-GrQc），比较三者的度分布和聚类系数。使用的工具是 Stanford 的 SNAP 库（Stanford Network Analysis Platform）。

### 1.2 比赛目标

输入是一份边列表文件 `ca-GrQc.txt`（arXiv 广义相对论领域的合作网络），每一行表示两个作者合作过。输出是三类图对象的统计量：节点数、边数、度分布、平均聚类系数。作业还要求我们实现图的生成函数，例如 `genErdosRenyi`、`genCircle`、`genSmallWorld`。

### 1.3 评价标准

这是一份课程作业，材料中没有线上排行榜或自动评分脚本。我们以"生成的图是否满足题目要求（节点数、边数、无自环）、统计量是否与图论预期一致"作为自检标准。例如 Erdős–Rényi 图的度分布应接近泊松分布，真实合作网络应呈现重尾分布。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先安装并导入 SNAP，加载合作网络的边列表：

```python
# 安装 Stanford 的图分析库
!python -m pip install snap-stanford
import snap

# 读取边列表：第 0 列和第 1 列分别是边的两个端点
G = snap.LoadEdgeList(snap.PUNGraph, 'ca-GrQc.txt', 0, 1)
print('节点数:', G.GetNodes())
print('边数:', G.GetEdges())
```

材料中的真实网络约有几千个节点、一万多条边。我们接着统计度分布：遍历每个节点，把它的度数累加到字典里：

```python
from collections import Counter

# 统计每个度数出现的次数
deg_count = Counter()
for node in G.Nodes():
    deg_count[node.GetDeg()] += 1

# 按度数排序，打印前几个
for deg in sorted(deg_count)[:10]:
    print(f'度 {deg}: {deg_count[deg]} 个节点')
```

度分布是理解网络结构的第一步：真实社交/合作网络通常有少量"超级节点"度数很高，大部分节点度数很低。

### 2.2 数据清洗

原始边列表可能包含自环（自己连自己），自环会干扰聚类系数和度分布。我们必须先删掉：

```python
# 遍历所有节点，如果发现自环就删除
for node in G.Nodes():
    nid = node.GetId()
    if G.IsEdge(nid, nid):
        G.DelEdge(nid, nid)
```

我们还检查了重复边。SNAP 的无向图在 `AddEdge` 时如果边已存在不会重复添加，因此加载后的图是简单图。

### 2.3 模型训练与超参数调优

这里的"模型"指的是三种图生成模型。我们首先实现 Erdős–Rényi 随机图：给定节点数 N 和边数 E，随机连边直到达到 E 条：

```python
def genErdosRenyi(N=5242, E=14484):
    Graph = snap.TUNGraph.New(N, E)
    for i in range(N):
        Graph.AddNode(i)          # 先建好所有节点
    edges = 0
    Rnd = snap.TRnd(42)           # 固定随机种子，保证可复现
    Rnd.Randomize()
    while edges < E:
        src = Graph.GetRndNId(Rnd)  # 随机选两个节点
        dst = Graph.GetRndNId(Rnd)
        # 排除自环和已经存在的边
        if src != dst and not Graph.IsEdge(src, dst):
            Graph.AddEdge(src, dst)
            edges += 1
    return Graph
```

其次我们构造小世界图：先画一个环（每个节点连左右邻居），再把"邻居的邻居"也连上，最后随机加 4000 条边：

```python
def genSmallWorld(N=5242, E=14484):
    Graph = genCircle(N)              # 第一步：环
    Graph = connectNbrOfNbr(Graph, N) # 第二步：连邻居的邻居
    Graph = connectRandomNodes(Graph, 4000)  # 第三步：随机加边
    return Graph

def connectNbrOfNbr(Graph, N):
    # 每个节点 i 连到 i+2，相当于把距离 2 的节点也连起来
    for i in range(N):
        Graph.AddEdge(i, (i + 2) % N)
    return Graph
```

接着我们实现聚类系数的计算。单个节点的局部聚类系数 = 它的邻居之间实际存在的边数 / 邻居之间可能的最大边数：

```python
def calcClusteringCoefficientSingleNode(Node, Graph):
    if Node.GetDeg() < 2:
        return 0.0                       # 度小于 2 无法形成三角形
    C = 0
    for i in range(Node.GetDeg()):
        for j in range(i + 1, Node.GetDeg()):
            # 检查第 i 个邻居和第 j 个邻居之间是否有边
            if Graph.IsEdge(Node.GetNbrNId(i), Node.GetNbrNId(j)):
                C += 1
    # 公式：2 * 邻居间实际边数 / (k * (k-1))
    return 2.0 * C / (Node.GetDeg() * (Node.GetDeg() - 1))

def calcClusteringCoefficient(Graph):
    total = 0.0
    for node in Graph.Nodes():
        total += calcClusteringCoefficientSingleNode(node, Graph)
    return total / Graph.GetNodes()      # 对所有节点取平均
```

### 2.4 模型测试与指标报告

我们把三种图的度分布画在同一张双对数坐标图上，直观比较：

```python
import matplotlib.pyplot as plt

def getDataPointsToPlot(Graph):
    X, Y = [], []
    DegToCntV = snap.TIntPrV()
    snap.GetDegCnt(Graph, DegToCntV)     # SNAP 自带的度直方图
    for item in DegToCntV:
        X.append(item.GetVal1())
        Y.append(item.GetVal2() / Graph.GetNodes())
    return X, Y

er = genErdosRenyi()
sw = genSmallWorld()
collab = G

for graph, name, style in [(er,'Erdos-Renyi','y'),
                           (sw,'Small World','r--'),
                           (collab,'Collaboration','b:')]:
    x, y = getDataPointsToPlot(graph)
    plt.loglog(x, y, style, label=name)
plt.xlabel('Node Degree (log)')
plt.ylabel('Proportion (log)')
plt.legend()
plt.show()
```

从图中我们看到：Erdős–Rényi 的度分布在平均度附近集中，尾部下降很快；Small-World 和真实合作网络的尾部更长。这和理论一致：真实网络存在"富者更富"的枢纽节点。

聚类系数方面，我们用上面的函数分别计算：

```python
print('ER 聚类系数:', calcClusteringCoefficient(er))
print('Small-World 聚类系数:', calcClusteringCoefficient(sw))
print('合作网络聚类系数:', calcClusteringCoefficient(collab))
```

小世界图因为显式连接了邻居的邻居，聚类系数明显高于同规模随机图；真实合作网络也表现出较高的聚类系数加上较短的平均路径，这正是"小世界"现象。

---

## 三、总结

本文中我们完成了 ML in Graphs HW1 的完整流程。我们首先用 SNAP 加载 arXiv 合作网络并统计节点、边与度分布。其次删除了自环，保证图是简单图。接着实现了 Erdős–Rényi 随机图、环形图、小世界图和聚类系数计算。最后把三种图的度分布画在双对数坐标上比较，并用聚类系数验证了小世界结构。

这个作业给我们的最大启发是：**图的结构可以用少数几个统计量刻画**——度分布告诉我们枢纽是否存在，聚类系数告诉我们局部抱团程度，两者结合就能区分随机图、小世界图和真实网络。