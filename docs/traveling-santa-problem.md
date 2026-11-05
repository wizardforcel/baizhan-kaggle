# Traveling Santa Problem

## 一、比赛简介

### 1.1 比赛背景
圣诞老人要给 150000 个城市送礼物，最短的送礼路线是什么？这就是 Kaggle 的经典圣诞系列赛 Traveling Santa Problem 里要解决的问题。题目本质上是一个欧氏旅行商问题（TSP），每个城市都可以用二维平面上的坐标 x、y 来表示，城市之间的路程就是两点之间的直线距离。

从材料 `p001_01.txt` 里可以看到，比赛提供了 `santa_cities.csv` 这个数据集，里面只有三列：id、x、y，一共 150000 行。还有一个 `random_paths_benchmark.csv` 用来提供随机路线作为基准参考。题目规模非常大，穷举所有可能的排列是不可能的，这是典型的 NP-hard 组合优化问题，入门者需要用贪心、局部搜索等启发式方法来求解，而不是用机器学习模型去“学习”路线。

所谓欧氏距离，就是我们在纸上量两点之间直线距离的公式：横坐标差平方加纵坐标差平方再开根号。对圣诞老人来说，两个城市之间没有道路限制，可以直接飞过去，所以路线长度就是一段段直线距离相加。城市数量是 15 万，如果要把所有访问顺序都试一遍，排列数量是一个天文数字，任何计算机都算不完。因此比赛考察的不是“预测”，而是“搜索”：怎样在有限时间里构造一条尽量短的闭环路线。

材料里还注释了 `from concorde.tsp import TSPSolver`，说明大家会考虑用专业的 TSP 求解器，但对 15 万个点的规模，精确求解不可行，只能先用简单启发式生成一个可用的解。

### 1.2 比赛目标
输入是 `santa_cities.csv`，每行一个城市的 id、x 坐标、y 坐标。输出是提交文件里的一列路径，记录圣诞老人访问城市的顺序。材料里的基准代码用 `path1` 作为列名：

```python
mySub = pd.DataFrame({'path1':path})
mySub.to_csv("sub.csv", index=False)
```

路径必须从城市 0 开始，最后回到城市 0，且每个城市恰好访问一次。我们从代码 `nearest_neighbour` 的实现里可以看到路径构造时 `path = [0,]`，最后 `path.append(0)` 封回起点。这个规则和旅行商的标准定义一致。

训练集和测试集在这里是同一个含义：我们只能基于给出的 `santa_cities.csv` 生成路线，没有隐藏的标签，整个流程就是“读坐标→构造路线→计算总长度→提交”。

### 1.3 评价标准
评价指标是总路程距离，越小越好。材料里定义了一个 `total_distance` 函数：

```python
def total_distance(df_path):
    # 计算相邻城市之间的欧氏距离之和，这是我们用来评估路线好坏的指标
    dist = (np.sqrt((df_path.x - df_path.x.shift())**2 +
                    (df_path.y - df_path.y.shift())**2)).sum()
    return round(dist,2)
```

这说明排行榜上的分数就是路径上所有相邻城市的欧氏距离累加，精度到小数点后两位。分数越低表示路线越短，圣诞老人走的越少。这个指标直接对应 TSP 的目标函数，和是否使用机器学习无关，启发式搜索的目标就是把这个数字压下来。

## 二、解题过程

我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）
我们首先读取数据，确认规模和字段是否齐全：

```python
import pandas as pd
import numpy as np

path = "/kaggle/input/traveling-santa-problem/"
df = pd.read_csv(path+"santa_cities.csv")  # 读取城市坐标表
sub = pd.read_csv(path+"random_paths_benchmark.csv")  # 读取随机基准路线
```

读完后材料显示 `150000 rows × 3 columns`，id、x、y 三列齐全。为了直观感受分布，我们把所有城市点画在二维平面上：

```python
plt.figure(figsize=(16,10))
plt.plot(df.x, df.y, 'k,', alpha=0.3)  # 用小点画出所有城市，alpha 让重叠处可见
plt.plot(df.x[0], df.y[0], 'bx')       # 把起点城市 0 用蓝色叉标出来
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('All cities (first in blue)', fontsize=18)
plt.show()
```

这一步的大白话解释是：城市在平面上散布得很开，起点 0 处于某个角落，单纯看图就知道任意起点到远处城市的距离差异很大，贪心路线会非常依赖“下一个最近点”的选择。

我们还用 `plot_path` 把一条路线连起来看：

```python
def plot_path(path, df, size=8, verbose=True):
    df_path = pd.merge(pd.DataFrame({'id':path}), df, on='id', how="left")
    plt.figure(figsize=(size,size))
    plt.plot(df_path.x, df_path.y, linewidth=1)  # 画出路线连线
    plt.plot(df_path.x[0], df_path.y[0], 'ro')   # 起点用红色圆标出
    plt.title(str(len(set(path)))+' cities: '+str(total_distance(df_path)), fontsize=18)
    plt.show()
```

从材料得到的示例 `0 2 5 1 8 3 4 9 6 7 0` 说明路线要闭环，标题里会显示城市数和总距离，这让我们在调参时能快速看到视觉效果。

### 2.2 数据清洗
这份数据本身非常干净，没有缺失值，也没有需要归一化的特征。我们做的唯一检查是确认 id 是否连续且坐标没有重复：

```python
# 检查 id 列是否连续，确认城市总数
ids = set(df['id'].unique())
print('城市数:', len(ids))
# 检查是否有重复 id
print('是否有重复 id:', df['id'].duplicated().any())
```

大白话说：坐标是原始测量值，距离计算需要真实的数值，不能做标准化或转小写等文本清洗。清洗的关键不是改数据，而是确认我们后面拼接路径时 `merge` 能正确对上城市坐标。材料里没有做任何归一化或去噪，这也是 TSP 的常规做法。

为什么不把坐标缩放到 0 到 1 之间？因为缩放在这里没有好处，它只是把所有距离同时乘上一个常数，最短的路线仍然是同一条，反而会让我们失去对真实里程的直觉。我们更愿意保留原始坐标，这样 `total_distance` 算出来的数字就接近真实的行程距离，方便和基准比较。另一个容易踩的坑是合并时的行顺序：`merge` 会按照路径里的 id 顺序返回坐标，如果先对坐标表排序再合并，顺序就会乱掉，所以代码里始终保持路径表在左边。

我们还注意到代码里直接用 `df.x.shift()` 来算相邻距离，这要求路径 DataFrame 的顺序就是访问顺序，清洗时只要保证 `id` 作为主键唯一即可。

### 2.3 模型训练与超参数调优
这里的“模型”其实是路线构造的启发式算法。我们首先实现贪心最近邻（Nearest Neighbour）作为基线：

```python
def nearest_neighbour(df):
    # 从城市 0 出发，每次挑距离当前城市最近的未访问城市
    ids = df.index.values[1:]
    xy = np.array([df.x.values, df.y.values]).T[1:]
    path = [0,]
    for _ in tqdm(range(len(df)-1)):
        last_x, last_y = df.x[path[-1]], df.y[path[-1]]
        dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)
        nearest_index = dist.argmin()  # 找最近邻
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        xy = np.delete(xy, nearest_index, axis=0)
    path.append(0)  # 回到起点
    return path
```

我们首先在 `df[:100]` 上试跑，得到一条闭合环路，视觉上能快速判断是否出现大量交叉。接着我们依次在 1000 个、10000 个城市上跑，观察运行时间和路线质量，随着规模增大，贪心策略会产生更多“短视”结果，因为太早选了最近的点会把远处的点留到后面，导致最后的长跳。这种现象在材料里通过 `plot_path` 的不同 size 展示出来。

最近邻的计算量是平方级别的，每次都要和剩下的所有城市算距离，时间增长很快。代码里用 `tqdm` 显示进度，就是为了让运行过程可见。我们并没有调超参数，而是通过改变输入规模来观察算法的稳定性和耗时。材料里注释了 `from concorde.tsp import TSPSolver`，说明如果规模允许，可以换成精确求解器来得到小规模的最优解作为参考。我们在实践中把贪心得到的路线作为初始解，后面再用局部搜索（如 2-opt）去交换两条边来降低总距离。代码层面表现为在 `nearest_neighbour` 之后再调用改善函数，这里材料只展示了最基础的贪心。

最后我们把全量 150000 个城市的贪心路线生成并保存为提交文件：

```python
path = nearest_neighbour(df)
mySub = pd.DataFrame({'path1':path})
mySub.to_csv("sub.csv", index=False)
```

### 2.4 模型测试与指标报告
我们用前面定义的 `total_distance` 来评估不同规模下的路线，直接在标题里显示距离：

```python
path_small = nearest_neighbour(df[:100])
plot_path(path_small, df[:100], size=8, verbose=True)
```

从材料里可以看到，跑 100、1000、10000 以及全量 150000 的路线都会调用 `plot_path` 画图，标题里会输出城市数和总距离。正是通过这种分层测试，我们能判断贪心策略在规模增大时是否还保持可接受的距离增长。

我们还对比了 `random_paths_benchmark.csv` 里的随机路线，材料里读取了这个文件但没有给出具体数值。原则上我们会用 `total_distance` 计算随机基线和贪心基线的距离差异，用“比随机短多少”来说明启发式是有用的。

在提交之前，我们会检查路径是否合法，避免因为漏掉城市或重复访问而被判错：

```python
# 提交的路径应包含所有城市各一次，并且首尾都是 0
path_set = set(path)
print('是否覆盖全部城市:', len(path_set) == len(df))
print('起点和终点:', path[0], path[-1])
```

这一步看似简单，但在 TSP 比赛里非常关键：只要少一个城市，整条路线就不合法，分数会直接作废。材料里最后的 `to_csv` 没有 `index=True`，正符合 Kaggle 提交规范。

最后我们随机抽几行提交文件，核对列名是 `path1`、行数与城市数一致，确认格式没有问题后才点击提交。指标报告的口径始终是 `total_distance` 算出的总距离，越低越好；视觉上的路线交叉越少，通常也意味着距离越短。

## 三、总结

本文中我们完成了 Traveling Santa Problem 的完整探索与基线构建。我们首先通过 EDA 确认了 150000 个城市的坐标分布，并用可视化把城市散点和候选路线画出来，让启发式搜索变得直观。其次我们做了最小程度的数据清洗，确认 id 唯一且坐标完整，没有做不必要的归一化。接着我们用贪心最近邻算法生成了从城市 0 出发的闭合回路，并在 100、1000、10000、150000 四个规模上分别测试，最后用 `total_distance` 计算总欧氏距离并生成提交文件。

从材料可以看出，最近邻是一种非常直观但短视的启发式，随着城市数增加路线交叉增多，距离会偏大。距离计算依赖欧氏几何，路径顺序的变化会直接影响总和，这也是 TSP 被称为组合爆炸问题的原因。

这个比赛给入门者的最大启发是：TSP 这类组合优化问题，特征工程没有用武之地，关键在于对搜索策略的理解和局部改进。我们用可视化来帮助判断路线质量，用 `total_distance` 来量化好坏，用分层规模测试来控制计算成本。后续如果要提升成绩，可以在贪心初始解的基础上加入 2-opt、模拟退火等局部搜索，或者对城市先做空间聚类再拼接子路线。材料中注释的 Concorde 求解器提示了小规模精确解的方向，但对 15 万点规模，启发式仍然是可行的起点。对初学者来说，这个比赛最重要的收获不是拿到多高的排名，而是亲手体会「构造一个解」和「评估一个解」之间的循环：先写一个能跑的版本，再用距离函数量化它，然后针对最长的那些路段做局部调整，如此反复迭代，路线就会一点点变短。
