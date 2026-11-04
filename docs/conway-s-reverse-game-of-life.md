# 康威生命游戏的逆向还原（Conway's Reverse Game of Life）

## 一、比赛简介

### 1.1 比赛背景

**康威生命游戏**（Conway's Game of Life）是数学家 John Conway 在 1970 年提出的一种「细胞自动机」。它规则极简，却能源源不断地产生复杂图案，几十年来一直是程序员和数学爱好者的玩具。

玩法是这样的：想象一张方格纸，每个格子里住着一个细胞，细胞只有两种状态——**活着（1）** 或者 **死着（0）**。每一轮（称为一「代」），所有格子同时按下面三条规则更新：

1. 一个活细胞周围 8 个邻居里，如果有 2 个或 3 个活细胞，它就继续活着；
2. 一个活细胞周围活邻居少于 2 个（太孤单）或多于 3 个（太拥挤），它就死掉；
3. 一个死格子周围正好有 3 个活细胞，它就「复活」。

规则只有三条，但只要给定一个初始图案，往后推演多少代都能算得清清楚楚。这个过程是**确定性的**：同样的初始状态，一定演化出同样的结果。

而这个比赛做的事情正好相反：**给出演化之后的棋盘，让你倒推出演化之前的棋盘。** 正向推演是「照着规则数邻居」，谁都会算；反向推演却难得多——因为正向过程会丢失信息，很多不同的初始棋盘会演化成同一个结果，你只能猜「最可能是哪一个」。

比赛用的棋盘是 **20×20**，也就是 400 个格子。每条样本都带一个 `delta`，表示从起始帧到结束帧经过了几代。delta 越大，丢的信息越多，倒推就越难。数据里的 delta 从 1 一直到 5。

### 1.2 比赛目标

训练集 `train.csv` 每一行是一条样本，包含三部分：

- `id`：样本编号；
- `delta`：经过了多少代；
- `stop.1` ～ `stop.400`：结束帧（演化之后）的 400 个格子，取值 0 或 1；
- `start.1` ～ `start.400`：起始帧（演化之前）的 400 个格子，这就是我们要预测的答案。

测试集 `test.csv` 只有 `id`、`delta` 和 `stop.1` ～ `stop.400`，没有 `start.*` 那 400 列。我们要给每条测试样本补上 400 个 0/1 预测值，列名必须是 `start.1` 到 `start.400`。

所以这是一个**逐格二分类**问题：对 400 个格子各预测一次「它原来活没活」，只是这些格子彼此之间有空间关系，不能当成 400 个独立问题来看。

### 1.3 评价标准

这个比赛按「预测的 400 个格子和真实格子差多少」打分。材料里有两个数字可以对上：一位选手在本地验证集上的逐格准确率是 **0.8530**，他提交后拿到的排行榜分数是 **0.14294**；0.8530 的错误率正好是 0.1470，与 0.14294 很接近。由此可以推断，指标是**逐格比较的误差**（通常写作 MSE，即均方误差）——把 400 个格子的预测值和真值相减、平方、再取平均。**这个数字越低越好**，0 表示 400 个格子全猜对。

> 小提示：因为 delta 不同难度差别很大，后面我们会**按 delta 各训练一个模型**，而不是用一个模型硬扛所有难度。

---

## 二、解题过程

整场比赛的主线可以概括成一句话：**先用 Cython 写一个高速的生命游戏模拟器，用它无限量地生成「起始帧 + 结束帧」配对，再训一个卷积神经网络学会反过来推。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步我们先把三个文件读进来，看看数据规模和列名：

```python
# numpy 做数值计算，pandas 读写表格
import numpy as np
import pandas as pd

# index_col=0 表示把 csv 的第一列当作行索引，避免多出一列 Unnamed
train_df = pd.read_csv('../input/train.csv', index_col=0)
test_df = pd.read_csv('../input/test.csv', index_col=0)

# 看形状和列数：每行应该是 id + delta + 400 个 stop + 400 个 start
print(train_df.shape)
print(test_df.shape)

# 只看列名的前缀，确认 stop.* 和 start.* 各 400 个
print(train_df.columns[:5])
```

打印出来我们确认了两件事：第一，`stop.` 开头的列和 `start.` 开头的列各自正好 400 个，对应 20×20 的棋盘；第二，测试集比训练集少了 `start.*` 这 400 列，符合预期。

接着我们想看看 delta 的分布——这决定了我们要不要分难度建模：

```python
# delta 表示从起始帧到结束帧隔了几代，看看每一档有多少样本
print(train_df['delta'].value_counts().sort_index())
```

delta 覆盖了 1 到 5 这几档。这不只是「看看数量」，它直接决定了我们的建模策略：**每一档 delta 单独训一个模型**。因为 delta=1 的题目里结束帧和起始帧只差一代，很多格子其实没变；而 delta=5 的题目里两者可能已经面目全非，两者的难度完全不在一个量级，硬塞给同一个模型只会互相拖累。

然后我们把一行数据重新摆回 20×20 的方阵，确认「400 列」这个展平顺序没有搞错：

```python
# 取第一条 delta=1 的样本，把 400 个数还原成 20×20 的网格
row = train_df[train_df.delta == 1].iloc[0]
start_grid = row[[f'start.{i}' for i in range(1, 401)]].values.reshape(20, 20)

# 用字符直接打印，1 用 ■ 表示活细胞，0 用 · 表示死格子
for line in start_grid:
    print(''.join('■' if v else '·' for v in line))
```

打印出来的图案是一个二维棋盘，而不是一串乱序的点——这说明 `start.1` 到 `start.400` 就是**按行优先**排列的：第 1 行占前 20 个，第 21 到 40 个是第 2 行，以此类推。这一点必须确认，否则 `reshape(20, 20)` 之后棋盘会被搅成一团，卷积网络就学不到相邻关系了。

最后我们看了一下训练集的规模：

```python
# 训练集一共多少行
print(len(train_df))
```

材料里另一份代码的训练日志写着「Train on 45000 samples, validate on 5000 samples」，两者相加正好是 5 万，说明训练集约 **50000 条**。5 万条样本要训一个 400 输出的卷积网络，数量偏紧、而且 delta 一分就只剩 1 万条一档，这也是我们后面决定「自己造数据」的直接原因。

### 2.2 数据清洗

这份数据本身非常干净：没有缺失值，没有文本字段需要处理，每个格子都是规整的 0/1。所以「清洗」在这里主要指**形状整理**，而不是删改内容。

我们需要把 pandas 的 400 列宽表变成卷积网络认识的 `(样本数, 20, 20, 1)` 四维张量：前两维是长宽，最后一维是「通道数」，因为只有一个二值通道，所以是 1。

```python
# 按前缀挑出 stop.* 的列，顺序必须保持 stop.1 → stop.400
stop_cols = [f'stop.{i}' for i in range(1, 401)]

# 取成 numpy 数组，再 reshape 成 20×20 的网格
# -1 表示这一维让 numpy 自己算（等于样本条数）
X = test_df[stop_cols].values.reshape(-1, 20, 20, 1).astype('uint8')

print(X.shape)   # (样本数, 20, 20, 1)
```

还有一件小事我们特意没做：**没有做归一化**。常规图像任务会把像素缩放到 0～1 或做标准化，但这里每个格子本来就是 0 或 1，已经在最小尺度上了。一位选手在材料里也验证过这一点——他试过打开和关掉 `normalize`，结果「没有太大差别」。所以我们保持原样，省掉一次没必要的变换。

### 2.3 模型训练与超参数调优

这一节是整个比赛的核心。我们做了三件事：写模拟器、造数据、训网络。

**第一件：写一个高速的生命游戏模拟器。** 训练一个网络需要几十万甚至上百万条样本，如果每条都靠真实数据，根本不够用。但生命游戏的规则是确定的，我们可以**自己造数据**——随机撒一堆活细胞，往后推 5 代得到结束帧，再往前就是起始帧。麻烦在于纯 Python 跑 400 个格子 × 若干代太慢，所以我们用 **Cython**（把 Python 代码编译成 C 速度的工具）重写了核心循环：

```python
%%cython
# 关掉边界检查和除法检查，让生成的 C 代码跑得更快
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int calc_neighs(unsigned char[:, :] field, int i, int j, int n, int k):
    """数 (i, j) 周围 8 个邻居里有多少个活细胞"""
    cdef int neighs = 0
    cdef int i_min = i - 1, i_pl = i + 1
    cdef int j_min = j - 1, j_pl = j + 1
    # 逐个方向判断是否越界，越界就不数，相当于棋盘外面全是死格子
    if i_min >= 0:
        if j_min >= 0: neighs += field[i_min, j_min]
        neighs += field[i_min, j]
        if j_pl < k:   neighs += field[i_min, j_pl]
    if j_min >= 0:     neighs += field[i, j_min]
    if j_pl < k:       neighs += field[i, j_pl]
    if i_pl < n:
        if j_min >= 0: neighs += field[i_pl, j_min]
        neighs += field[i_pl, j]
        if j_pl < k:   neighs += field[i_pl, j_pl]
    return neighs
```

有了「数邻居」这个零件，推演一代就是把三条规则套上去：活细胞有 2 或 3 个邻居就活，死格子有 3 个邻居就复活，其余全死。把这一步循环 `moves` 次，就能得到任意代之后的棋盘。

**第二件：写一个数据生成器。** 我们不预先把数据全部生成好存到磁盘（那样太占地方），而是写一个「要多少给多少」的生成器，训练时实时造样本：

```python
def data_generator(delta=1, batch_size=32, ravel=True):
    """无限循环地吐出 (结束帧, 起始帧) 配对，配合 fit_generator 使用"""
    while True:
        # 随机初始化：每个格子以 0.5 的概率是活细胞
        batch = np.split(
            np.random.binomial(1, 0.5, (20 * batch_size, 20)).astype('uint8'),
            batch_size)
        # Yy 是起始帧：先把随机棋盘推 5 代，当作一个"合法"的中间状态
        Yy = [make_move(state, 5) for state in batch]
        # Xx 是结束帧：在起始帧基础上再推 delta 代
        Xx = [make_move(state, delta) for state in Yy]
        if ravel:
            # 展平成 400 维向量，给全连接层用
            yield np.array([y.ravel() for y in Yy]), np.array([x.ravel() for x in Xx])
        else:
            # 保留 20×20 的形状并加一维通道，给卷积层用
            yield np.array(Xx)[:, :, :, np.newaxis], np.array(Yy)[:, :, :, np.newaxis]
```

这里有个小细节值得说：我们不是直接拿随机棋盘当起始帧，而是**先推 5 代**。原因是纯随机的 0/1 棋盘在生命游戏里几乎瞬间就塌掉了（活细胞太密，大量被挤死），推 5 代之后得到的才是「看起来像正常棋盘」的状态，用它当起点造出来的样本更接近真实分布。

**第三件：设计网络。** 我们用的是**全卷积**结构：不放池化层，每层都保持 20×20 的大小，最后用 1 个卷积核输出一个通道，再过 `sigmoid` 压到 0～1 之间，当作每个格子「原来是活的」的概率。

```python
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization

def create_model(n_hidden_convs=2, n_hidden_filters=128, kernel_size=5):
    nn = Sequential()
    # 第一层要声明输入形状 (20, 20, 1)
    # padding='same' 保证输出还是 20×20，边界信息不丢
    nn.add(Conv2D(n_hidden_filters, kernel_size, padding='same',
                  activation='relu', input_shape=(20, 20, 1)))
    nn.add(BatchNormalization())   # 归一化，让深层网络更好训练
    for i in range(n_hidden_convs):
        nn.add(Conv2D(n_hidden_filters, kernel_size, padding='same', activation='relu'))
        nn.add(BatchNormalization())
    # 最后压成 1 个通道，sigmoid 输出 0~1 的概率
    nn.add(Conv2D(1, kernel_size, padding='same', activation='sigmoid'))
    # 逐格二分类，用二元交叉熵；accuracy 方便我们盯训练进度
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return nn
```

材料里还试过一个带 `MaxPool2D` + `Conv2DTranspose`（先缩小再放大）的版本，作者自己在注释里写了「这个比纯 Conv2D 效果差」。我们采纳这个结论，**只保留全卷积版本**，不做无谓的折腾。

**关于分难度训练。** 因为 delta 越大越难，我们干脆训练 **5 个独立的模型**，delta=1 到 5 各一个，各自只用对应难度的数据。这样每个模型可以专心学一种「模糊程度」，不用在内部做取舍。超参数上我们把隐藏卷积层加到 6 层、每层 256 个滤波器（`n_hidden_convs=6, n_hidden_filters=256`），并对每个 delta 单独挑了层数：

```python
models = []
for delta in range(1, 6):
    model = create_model(n_hidden_convs=6, n_hidden_filters=256)
    # 监控训练 loss，连续 9 轮改善不到 0.001 就提前停，省时间也防过拟合
    es = EarlyStopping(monitor='loss', patience=9, min_delta=0.001)
    model.fit_generator(
        data_generator(delta=delta, ravel=False),
        steps_per_epoch=500,     # 每轮 500 个 batch
        epochs=50,
        verbose=1,
        callbacks=[es])
    models.append(model)
```

训练日志显示，网络在第 1 轮结束时逐格准确率就有 0.8436，之后稳定爬升，到第 18 轮左右达到 0.9064，然后开始变平缓。这个曲线很典型：**逆向生命游戏的大部分规律很容易学**（比如「孤立的一个活细胞下一轮必死」），难的是边界处那些模棱两可的格子。

作为对照，我们也跑过一个更朴素的版本：两层卷积 + 池化 + 全连接，最后一层写成 `Dense(400, activation='softmax')`。那个版本训练 10 轮，验证准确率从第 1 轮起就一直卡在 0.8551 纹丝不动，loss 只在小数点后第四位缓慢下降——因为 `softmax` 会强行让 400 个格子的概率加和为 1，等于假设「整张棋盘只有一个格子是活的」，这跟真实情况完全相反。这个失败案例提醒我们：**输出层的设计要和问题的结构对上**，逐格独立的二分类就该用逐格独立的 `sigmoid`。

### 2.4 模型测试与指标报告

训练完成后，我们按每条测试样本的 delta 把它送进对应的那个模型：

```python
# 提交表的行索引和测试集一致，列是 start.1 ~ start.400
submit_df = pd.DataFrame(
    index=test_df.index,
    columns=['start.' + str(i) for i in range(1, 401)])

for delta in range(1, 6):
    mod = models[delta - 1]                       # 取出这一档难度对应的模型
    # 只挑出 delta 等于当前值的行，取 stop.* 列并摆成 20×20×1
    delta_df = test_df[test_df.delta == delta].iloc[:, 1:].values.reshape(-1, 20, 20, 1)
    # 预测出概率后四舍五入成 0/1，再展平回 400 列
    submit_df[test_df.delta == delta] = (
        mod.predict(delta_df).reshape(-1, 400).round(0).astype('uint8'))

submit_df.to_csv('cnns_40.csv')
```

这里我们踩过一个坑，值得记下来：一开始直接拿 `predict` 出来的**概率**去写提交文件，结果分数很差。因为指标比的是 0/1 的差值，输出 0.6 这种小数会被当成「错了一大半」，而它其实已经很接近「是活的」了。加上 `.round(0).astype('uint8')` 把概率硬压成 0 或 1 之后，分数才正常。材料里另一位选手也明确提到过这个疑虑——「还是觉得四舍五入可能有问题」。

最终我们的提交拿到 **0.14294** 的排行榜分数。这个结果说明模型确实学到了东西——如果只是乱猜，400 个格子里总会有大量判反；但离满分还有很长的路，主要差距集中在 delta 大的样本上，那些棋盘演化了好几代，原本的信息已经被抹得差不多了。

另外我们注意到一个现象：在**自己造的合成数据**上，模型能达到 0.90 以上的逐格准确率，但按榜单分数 0.14294 反推，测试集上的准确率只有 0.857 左右。这个落差来自分布差异——我们造数据用的是「随机撒 50% 活细胞再推 5 代」，而真实数据的起始棋盘未必是这么来的。这提醒我们，**生成式数据增强一定要想清楚它的分布假设**，否则本地指标会虚高。

---

## 三、总结

本文中我们完成了康威生命游戏逆向还原的完整流程。我们首先用 EDA 摸清了数据：每条样本是 20×20 的棋盘展平成 400 列 `stop.*` 和 400 列 `start.*`，delta 分成 1 到 5 五档，训练集约 5 万条，我们逐档确认了列是按行优先排列的。其次我们做了最轻量的「清洗」——只是把 400 列宽表 reshape 成 20×20×1 的张量，因为格子本来就是 0/1，归一化反而多余。接着我们用 Cython 写了一个高速的生命游戏模拟器，再包装成无限吐数据的生成器，解决了训练样本不够的问题；然后用全卷积网络（6 层隐藏卷积、256 个滤波器、`sigmoid` 单通道输出）按 delta 分 5 个模型分别训练，并用 `EarlyStopping` 控制训练轮数。最后我们把测试集按 delta 分流预测、四舍五入成 0/1 写进提交文件，拿到 0.14294 的排行榜分数。

这个比赛给我们最大的启发有两点。第一，**当正向过程是确定的时候，可以自己造无限多的训练数据**——这比想尽办法做数据增强要划算得多。第二，**问题的结构要反映到网络结构上**：400 个格子在空间上是相邻的，所以用卷积而不是全连接；它们彼此独立，所以用逐格 `sigmoid` 而不是 400 路 `softmax`。同一个任务，输出层选错，模型就卡在原地学不动。
