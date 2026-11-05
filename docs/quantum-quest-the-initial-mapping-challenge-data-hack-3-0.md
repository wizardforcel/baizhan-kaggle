# QuantumQuest 初始映射挑战（DataHack 3.0）

## 一、比赛简介

### 1.1 比赛背景

量子程序写出来之后，要真正放到量子芯片上运行，中间还隔着一道坎：**逻辑量子比特和物理量子比特不是一一对应的**。我们写代码时习惯说「第 0 个比特、第 1 个比特」，这只是逻辑编号；芯片上真正存在的比特有自己的编号，而且彼此之间的连接关系是固定的——有的比特挨着，能直接做两比特门；有的隔得远，必须先「绕路」。

于是编译器要做的第一件事，就是决定**把哪个逻辑比特放到哪个物理比特上**，这件事叫**初始映射（initial mapping）**。映射得好，后续的两比特门大多数能直接执行，需要插入的交换门就少，线路就短、错误就少；映射得差，线路会被拉长好几倍，跑出来的结果基本没法用。

这个比赛要我们做的，就是给一个量子线路和一块硬件的连接图，自动给出一个尽量好的初始映射。

### 1.2 比赛目标

输入有两样：

- 一段 **QASM 文本**：这是量子线路的标准描述格式，里面一行一行写着「在哪些比特上做什么门」，例如 `cx q[0],q[1];` 表示在第 0、1 号比特上做一个受控非门；
- 一个**拓扑列表**：描述物理芯片上哪些比特之间有连线，例如 `[[0,1],[1,2],[2,3]]`。

输出是一个**排列列表**：第 `i` 个位置上的数字，表示逻辑比特 `i` 被放到哪个物理比特上。评测脚本会拿这个映射去编译线路，再看最终结果好不好。

### 1.3 评价标准

材料里没有给出具体的评分公式，我们只能从任务形态推断：评测会把这个映射交给编译器，编译出一版能在目标硬件上执行的线路，然后看它的**执行代价**（通常体现为线路深度、交换门数量或整体保真度）。映射越贴合硬件连接，代价越低，分数越好。因为材料没有给出具体数值，我们在下文不会编造任何分数，只讲方法本身。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先要把两种输入「看成图」。这是整道题最关键的一步认识：**线路和硬件，本质上都是图**。

- 线路图：每个量子比特是一个节点；如果两个比特之间出现两比特门，就在它们之间连一条边。一条边出现得越多，说明这两个比特「交流」越频繁，越应该被放到相邻的物理位置上。
- 硬件图：每个物理比特是一个节点；芯片上有一条连线，就有一条边。

我们先把 QASM 解析出来，统计每个比特参与了多少次两比特门——这个次数就是节点的「度」，也是后面喂给模型的唯一特征：

```python
from qiskit import QuantumCircuit
import numpy as np

def qasm_to_graph(qasm_code):
    """把 QASM 线路变成 (节点数, 边列表, 每个节点的度)"""
    qc = QuantumCircuit.from_qasm_str(qasm_code)
    num_nodes = qc.num_qubits

    edges = []
    degrees = np.zeros(num_nodes, dtype=np.float32)

    # 只关心两比特门，单比特门不产生「谁和谁必须相邻」的约束
    for instruction in qc.data:
        if len(instruction.qubits) == 2:
            # find_bit 把「第几个比特对象」翻译成「编号」
            i = qc.find_bit(instruction.qubits[0]).index
            j = qc.find_bit(instruction.qubits[1]).index
            edges.append([i, j])
            edges.append([j, i])   # 无向图，正反都存一份
            degrees[i] += 1        # 每出现一次，两端的度都加一
            degrees[j] += 1

    return num_nodes, edges, degrees

# 用一段很小的示意线路看看效果
demo = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[0],q[3];
"""
print(qasm_to_graph(demo))
```

跑完之后我们会得到 4 个节点、6 条有向边（3 条无向边各存两次）、以及 `[2, 2, 1, 1]` 这样的度序列。度的含义很直白：数字越大，这个逻辑比特越「忙」，越需要放到连接多的物理位置上。

接着看硬件拓扑。它是以边列表给出的，我们需要自己数出有多少个物理比特，并算每个物理比特的度：

```python
import numpy as np

def topology_to_graph(topology_list):
    """把硬件连线列表变成 (节点数, 边列表, 每个节点的度)"""
    if not topology_list:
        return 0, [], np.array([], dtype=np.float32)

    # 最大编号 + 1 就是物理比特总数
    num_nodes = max(max(edge) for edge in topology_list) + 1
    degrees = np.zeros(num_nodes, dtype=np.float32)
    edges = []

    for src, dst in topology_list:
        edges.append([src, dst])
        degrees[src] += 1

    return num_nodes, edges, degrees

# 一条 4 个比特的直线链
topo = [[0, 1], [1, 2], [2, 3]]
print(topology_to_graph(topo))
```

这里有一个容易忽略的细节：拓扑列表只给了「有向」的一条边，而芯片连接是双向的。材料里的做法是**只在建图时按给出的方向存一次**，因为后面的图神经网络会把消息沿边传递，方向信息在聚合时会被自然处理。我们先照原样保留，不擅自补反向边，避免改变节点度的统计口径。

### 2.2 数据清洗

这份数据的「脏」不在缺失值，而在**结构上的冗余和不一致**。我们处理了三件事。

第一，**线路里可能有重复的两比特门**。同一个比特对之间做了十次门，如果每次都往边列表里塞一条，边会重复十次，模型看到的连接强度就会被放大。我们的处理是允许重复——因为重复次数正好代表了「这两个比特有多亲密」，这是有用的信号，不是噪声。但我们同时保留了度统计，让模型自己决定怎么用。

第二，**硬件拓扑里可能出现重复边或自环**。真实硬件的描述文件不一定干净。我们加了一道简单的去重和自环过滤：

```python
def clean_topology(topology_list):
    """去掉自环和重复的边，保证建图时不出现奇怪的结构"""
    seen = set()
    cleaned = []
    for src, dst in topology_list:
        if src == dst:            # 自环对映射没有意义，直接扔掉
            continue
        key = (min(src, dst), max(src, dst))   # 无向边统一成小号在前
        if key in seen:
            continue              # 重复边只保留一条
        seen.add(key)
        cleaned.append([key[0], key[1]])
    return cleaned
```

第三，**节点数可能对不上**。线路有 `num_logical` 个比特，硬件有 `num_physical` 个比特，通常后者更大（芯片总有空闲比特）。如果线路比特数超过物理比特数，这个映射根本不可能完成，属于非法输入。我们的做法是在入口处直接检查，早失败早报错：

```python
def check_feasible(num_logical, num_physical):
    """先确认物理比特够不够用，避免后面算出非法排列"""
    if num_logical > num_physical:
        raise ValueError(
            f"逻辑比特 {num_logical} 多于物理比特 {num_physical}，无法映射"
        )
    # 多出来的物理比特会被闲置，这是正常的
    print(f"逻辑 {num_logical} -> 物理 {num_physical}，"
          f"空闲 {num_physical - num_logical} 个")
```

### 2.3 模型训练与超参数调优

我们采用的是一个**双塔图神经网络（GNN）+ 相似度匹配**的结构。名字听着复杂，拆开很好懂：

1. 用一个小型的图卷积网络（GCN）分别给线路图和硬件图「编码」，每个节点变成一个向量，向量里浓缩了它周围的结构信息；
2. 把两边的向量做点积，得到一个 `逻辑数 × 物理数` 的相似度矩阵，分数越高表示「这个逻辑比特适合放在这个物理比特上」；
3. 用一个贪心算法从矩阵里挑出分数最高的一对一对配上。

先看模型定义。两条支路结构完全一样，只是作用在不同的图上：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class QubitMapperGNN(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        # 逻辑图支路：输入 1 维（度），输出 hidden_dim 维
        self.conv1_log = GCNConv(1, hidden_dim)
        self.conv2_log = GCNConv(hidden_dim, hidden_dim)
        # 物理图支路，结构与上面完全对称
        self.conv1_phys = GCNConv(1, hidden_dim)
        self.conv2_phys = GCNConv(hidden_dim, hidden_dim)

    def forward(self, logical_data, physical_data):
        # 第一层 + ReLU，让模型能学到非线性的结构模式
        x = F.relu(self.conv1_log(logical_data.x, logical_data.edge_index))
        log_emb = self.conv2_log(x, logical_data.edge_index)

        x = F.relu(self.conv1_phys(physical_data.x, physical_data.edge_index))
        phys_emb = self.conv2_phys(x, physical_data.edge_index)

        # 点积得到相似度矩阵，形状 (逻辑数, 物理数)
        return torch.matmul(log_emb, phys_emb.T)
```

有一点必须说清楚：材料里的 `get_initial_mapping` 在推理时是**现场实例化一个随机初始化的模型**，代码注释里写着「真实提交时应在这里 `load_state_dict` 加载预训练权重」。也就是说，这份材料给出的是**接口和结构**，不是一个训练好的成品。我们照实说明，不假装它已经训练过。

模型的输入要用 PyG 的 `Data` 对象包装。我们把前面算好的度当作节点特征，边列表转成 `edge_index`（形状 `2 × 边数`）：

```python
import torch
from torch_geometric.data import Data

def build_pyg_data(num_nodes, edges, degrees):
    """把 (节点数, 边列表, 度) 打包成 PyG 认识的 Data 对象"""
    if edges:
        # PyG 要求 edge_index 是 2 x E 的 long 张量
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # 线路里可能一个两比特门都没有，这时给一个空边集
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 节点特征只有一列：度。view(-1, 1) 把它变成 N x 1
    x = torch.tensor(degrees, dtype=torch.float).view(-1, 1)
    return Data(x=x, edge_index=edge_index)
```

关于超参数，材料只给了一个具体值：`hidden_dim=16`。我们的理解是，节点特征本身只有「度」这一个维度，信息量很小，隐藏层开太大容易过拟合，16 维已经够表达「连接多/少」这种粗粒度差异。另外层数只有 2 层，因为 GCN 每多一层，节点能「看到」的邻居就远一跳；两层已经能覆盖到二跳邻居，再深收益递减，还容易把不同区域的节点信息混在一起（过平滑）。

然后是推理阶段最关键的**贪心匹配**。相似度矩阵里不能简单地每行取最大值，因为两个逻辑比特可能都最喜欢同一个物理比特，会撞车。正确做法是「全局取最大 → 划掉对应行列 → 重复」：

```python
import numpy as np

def greedy_match(sim_matrix, num_logical):
    """从相似度矩阵中贪心地挑出一一对应的映射"""
    sim = sim_matrix.copy()          # 不破坏原矩阵
    mapping = {}
    assigned = set()

    for _ in range(num_logical):
        # 在剩下的矩阵里找全局最大值的位置
        li, pi = np.unravel_index(np.argmax(sim), sim.shape)
        mapping[int(li)] = int(pi)
        assigned.add(int(pi))

        # 这一行、这一列都作废，保证不会再被选中
        sim[li, :] = -np.inf
        sim[:, pi] = -np.inf

    # 按逻辑比特编号排好，输出成排列列表
    return [mapping[i] for i in range(num_logical)]
```

这段代码有三个地方值得展开：

- 用 `-np.inf` 而不是 0 来屏蔽，是因为点积结果可能是负数，用 0 会误伤；
- 每一轮都重新找全局最大，而不是按逻辑比特顺序逐个挑，可以避免「先挑的人占了好位置、后挑的人没得选」；
- 循环次数固定为逻辑比特数，保证每个逻辑比特都被分配到唯一一个物理比特。

### 2.4 模型测试与指标报告

因为材料里没有现成的评测分数，我们的「测试」只能做**合法性检查**。一个映射最容易犯两种错：长度不对、出现重复。我们写了两条断言把它们挡住：

```python
def validate_layout(layout, num_logical, num_physical):
    """检查输出的排列是否合法"""
    # 1) 长度必须等于逻辑比特数
    assert len(layout) == num_logical, \
        f"长度应为 {num_logical}，实际 {len(layout)}"

    # 2) 每个物理比特最多用一次，且编号要在范围内
    assert len(set(layout)) == len(layout), "出现重复的物理比特"
    for p in layout:
        assert 0 <= p < num_physical, f"物理比特 {p} 越界"

    print("映射合法：", layout)
```

除了合法性，我们还能做一个**直观的质量指标**：统计「线路里出现过的两比特门，有多少在映射后落在硬件真正相连的一对位置上」。这个比例越高，说明映射越贴合硬件，后续需要插入的交换门越少。这个指标不需要跑评测器，本地就能算：

```python
def edge_hit_rate(logical_edges, layout, physical_edges):
    """线路里的两比特门有多少能直接落在硬件连线上"""
    # 把物理连线也统一成无序对，方便查表
    phys_pairs = {tuple(sorted(e)) for e in physical_edges}

    hits, total = 0, 0
    for i, j in logical_edges:
        # 只统计无向边，前面正反各存了一次，这里去重
        if i > j:
            continue
        total += 1
        pair = tuple(sorted((layout[i], layout[j])))
        if pair in phys_pairs:
            hits += 1

    rate = hits / total if total else 1.0
    print(f"直连命中率 = {hits}/{total} = {rate:.2%}")
    return rate
```

这个「直连命中率」是我们在本地判断映射好坏的主要依据。命中率高，说明大部分两比特门不用搬比特；命中率低，就要回去检查是不是模型还没训练、或者贪心匹配把关键节点配错了。

最后我们把整个流程串起来，跑一遍完整接口：

```python
def get_initial_mapping(qasm_code, topology_list):
    """评测脚本调用的主函数：QASM + 拓扑 -> 排列列表"""
    # 1) 清洗并建图
    topology_list = clean_topology(topology_list)
    n_log, edges_log, deg_log = qasm_to_graph(qasm_code)
    n_phys, edges_phys, deg_phys = topology_to_graph(topology_list)

    # 2) 先确认物理比特够用
    check_feasible(n_log, n_phys)

    # 3) 编码成图数据
    log_data = build_pyg_data(n_log, edges_log, deg_log)
    phys_data = build_pyg_data(n_phys, edges_phys, deg_phys)

    # 4) 推理（真实提交时这里 load_state_dict 载入训练好的权重）
    model = QubitMapperGNN(hidden_dim=16).eval()
    with torch.no_grad():
        sim = model(log_data, phys_data).numpy()

    # 5) 贪心匹配并检查
    layout = greedy_match(sim, n_log)
    validate_layout(layout, n_log, n_phys)
    return layout
```

---

## 三、总结

本文中我们完成了 QuantumQuest 初始映射挑战的完整方案。我们首先把问题重新表述成「图与图的匹配」——线路图看两比特门的连接，硬件图看芯片连线；其次我们做了结构层面的数据清洗，去掉自环和重复边，并在一开始就检查逻辑比特数不超过物理比特数；接着我们用两条对称的 GCN 支路分别编码线路和硬件，把节点向量点积成相似度矩阵，隐藏维取 16、层数取 2，兼顾表达力和过拟合风险；然后我们用「全局取最大、划掉行列」的贪心算法把矩阵变成一一对应的排列；最后我们用长度、唯一性、越界三类断言检查映射合法性，并用「直连命中率」在本地衡量映射质量。

这个方案最大的优点是**结构清晰、可直接替换**：只要把随机初始化的 GCN 换成真正训练过的权重，整个流程不用改一行。它给我们的启发是，很多看起来和深度学习无关的工程问题，只要找对「表示方式」，就能变成一个标准的图学习任务。
