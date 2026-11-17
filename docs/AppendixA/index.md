# 附录A NumPy快速参考

NumPy（全称 Numerical Python）是 Python 里做科学计算的基础库，几乎所有机器学习库（pandas、scikit-learn、PyTorch、TensorFlow）都在它的基础上构建。如果你想在 Kaggle 上处理表格数据、图像、或者做任何数值计算，NumPy 是绕不过去的工具。

本附录面向零基础初学者，用大白话讲清楚 NumPy 最核心的内容：怎么创建数组，数组有哪些属性和索引方式，怎么做数学和统计运算，广播机制和矩阵运算怎么用，怎么产生随机数，怎么改变数组形状和合并拆分，怎么读写文件。

> **Mermaid 图表：NumPy 在数据处理流程中的位置**

```mermaid
flowchart LR
    A[原始数据 CSV/图片] --> B[NumPy 数组 ndarray]
    B --> C[数据清洗 缺失值/归一化]
    C --> D[pandas / scikit-learn 建模]
    D --> E[模型预测]
    style B fill:#bbf,stroke:#333
```

---

## A.1 数组创建

NumPy 的核心对象是 **ndarray**（N 维数组），简称“数组”。它和 Python 列表最大的区别：数组里所有元素类型必须相同，且支持对整个数组直接做向量化运算，不用写 `for` 循环。

### 安装与导入

```bash
pip install numpy
```

Kaggle Notebooks 里 NumPy 已经预装，直接导入，约定成俗起别名 `np`：

```python
import numpy as np
```

### 1. 从列表创建：`np.array()`

把 Python 列表转成 NumPy 数组，最常用的方法。

```python
# 一维数组（就像一个列表）
a = np.array([1, 2, 3, 4, 5])
print(a)           # [1 2 3 4 5]
print(type(a))     # <class 'numpy.ndarray'>
print(a.shape)     # (5,)  1 维，5 个元素

# 二维数组（就像一个表格/矩阵）
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]
print(b.shape)     # (2, 3)  2 行 3 列
```

> 💡 **大白话**：`shape` 告诉你数组的“形状”。一维数组的 shape 是 `(元素个数,)`，二维数组的 shape 是 `(行数, 列数)`。

### 2. 创建全 0 或全 1 数组：`np.zeros()` / `np.ones()`

```python
# 创建一个 3 行 4 列 的全 0 数组
zeros = np.zeros((3, 4))
print(zeros)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 创建一个 2 行 3 列 的全 1 数组
ones = np.ones((2, 3))
print(ones)
# [[1. 1. 1.]
#  [1. 1. 1.]]
```

注意：`zeros` 和 `ones` 默认创建的是**浮点数**（`0.0`、`1.0`），不是整数。

### 3. 创建等差数列：`np.arange()` / `np.linspace()`

```python
# 0 到 9，步长为 1
print(np.arange(10))      # [0 1 2 3 4 5 6 7 8 9]

# 1 到 9，步长为 2（不包含 10）
print(np.arange(1, 10, 2))  # [1 3 5 7 9]

# 在 0 到 1 之间均匀取 5 个数（包含两端）
print(np.linspace(0, 1, 5))  # [0.   0.25 0.5  0.75 1.  ]
```

`arange` 适合“已知步长”的情况，`linspace` 适合“已知起止和个数”的情况。

### 4. 其他常用创建函数

```python
# 单位矩阵（对角线为 1，其余为 0）
print(np.eye(3))
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 对角线为指定值的矩阵
print(np.diag([1, 2, 3]))
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]

# 全是指定值的数组
print(np.full((2, 3), 7))
# [[7 7 7]
#  [7 7 7]]

# 未初始化的数组（值不定，创建快）
print(np.empty((2, 3)))
```

---

## A.2 数组属性与索引

### 1. 数组属性

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

print(a.shape)    # (2, 3)   形状：行数 x 列数
print(a.ndim)     # 2        维度数
print(a.size)     # 6        元素总个数
print(a.dtype)    # int64    元素数据类型
print(a.itemsize) # 8        每个元素占用的字节数
```

### 2. 改变形状（不改变数据）

```python
a = np.array([1, 2, 3, 4, 5, 6])

print(a.reshape((2, 3)))
# [[1 2 3]
#  [4 5 6]]

print(a.ravel())      # 展平成一维（返回视图）
print(a.flatten())    # 展平成一维（返回副本）

print(a.T)            # 转置，行列互换
print(a.transpose())  # 和 .T 一样
```

### 3. 一维数组索引

下标从 0 开始，也可以用负数从后往前数：

```python
a = np.array([10, 20, 30, 40, 50])

print(a[0])   # 10   第一个元素
print(a[1])   # 20   第二个元素
print(a[-1])  # 50   最后一个元素
print(a[-2])  # 40   倒数第二个元素
```

### 4. 切片：取一部分元素

切片语法是 `数组[开始:结束:步长]`，开始包含，结束不包含。

```python
a = np.array([10, 20, 30, 40, 50])

print(a[1:4])    # [20 30 40]   从下标 1 到下标 3（不含 4）
print(a[:3])     # [10 20 30]   从开头到下标 2
print(a[2:])     # [30 40 50]   从下标 2 到末尾
print(a[::2])    # [10 30 50]   每隔一个取一个
print(a[::-1])   # [50 40 30 20 10]  倒序
```

### 5. 二维数组索引

二维数组用 `数组[行, 列]` 来索引：

```python
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(b[0, 0])    # 1   第一行第一列
print(b[0, 2])    # 3   第一行第三列
print(b[2, 1])    # 8   第三行第二列
print(b[-1, -1])  # 9   最后一行最后一列

# 二维数组切片
print(b[0:2, 0:2])   # [[1 2]
                     #  [4 5]]   前 2 行前 2 列
print(b[1, :])       # [4 5 6]   第二行所有列
print(b[:, 1])       # [2 5 8]   第二列所有行
```

### 6. 花式索引与布尔索引

```python
a = np.array([10, 20, 30, 40, 50])

# 花式索引：传入一个下标数组
print(a[[0, 2, 4]])   # [10 30 50]

# 布尔索引：只取满足条件的元素
print(a[a > 25])      # [30 40 50]

# np.where：找出满足条件的元素下标
print(np.where(a > 25))  # (array([2, 3, 4]),)
```

> ⚠️ **重要陷阱**：NumPy 的切片返回的是**视图**（view），不是副本。你通过切片修改了元素，原数组也会跟着变。如果你想要一个完全独立的副本，要用 `.copy()`：

```python
a = np.array([1, 2, 3])
b = a[0:2]      # b 是 a 的视图
b[0] = 999
print(a)        # [999 2 3]  原数组也变了！

c = a[0:2].copy()  # c 是独立副本
c[0] = 0
print(a)        # [999 2 3]  原数组不变
```

---

## A.3 数学与统计函数

NumPy 提供了大量现成的数学和统计函数，可以直接对整个数组或某一轴求值。

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.sum())        # 21   所有元素之和
print(a.mean())       # 3.5  所有元素的平均值
print(a.std())        # 1.414...  标准差
print(a.var())        # 2.0   方差
print(a.min())        # 1    最小值
print(a.max())        # 6    最大值
print(a.argmin())     # 0    最小值的下标（展开后）
print(a.argmax())     # 5    最大值的下标（展开后）
print(a.median())     # 3.5  中位数
print(np.percentile(a, 50))  # 3.5  百分位数
print(np.diff(a))     # [[1 1]
                     #  [1 1]]   相邻元素之差
print(np.cumsum(a))   # [ 1  3  6 10 15 21]  累积和
print(np.sort(a))     # [[1 2 3]
                     #  [4 5 6]]  排序
print(np.argsort(a))  # [[0 1 2]
                     #  [0 1 2]]  排序后的下标
print(np.unique(a))   # [1 2 3 4 5 6]  去重后的唯一值
```

按轴求值（`axis` 参数）：

```python
# axis=0：竖着消掉行方向，对每一列做运算（按列）
print(a.sum(axis=0))   # [5 7 9]

# axis=1：横着消掉列方向，对每一行做运算（按行）
print(a.sum(axis=1))   # [6 15]
```

> 💡 **记住口诀**：`axis=0` 竖着算（按列），`axis=1` 横着算（按行）。

---

## A.4 广播机制与矩阵运算

### 1. 广播机制

广播让不同形状的数组能直接相加相乘，NumPy 会自动把较小的数组“扩展”到和大数组一样的形状。

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a + 10)   # [[11 12 13]
                #  [14 15 16]]  每个元素都加 10

print(a * 2)    # [[ 2  4  6]
                #  [ 8 10 12]]  每个元素都乘 2
```

### 2. 矩阵运算

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(A @ B)           # [[19 22]
                       #  [43 50]]  矩阵乘法（@ 运算符）
print(np.dot(A, B))    # 和 @ 一样
print(np.transpose(A)) # [[1 3]
                       #  [2 4]]  转置

print(np.diagonal(A))  # [1 4]  对角线元素
print(np.trace(A))     # 5    对角线元素之和

# 线性代数
print(np.linalg.inv(A))   # A 的逆矩阵
print(np.linalg.det(A))   # A 的行列式
print(np.linalg.norm(A))  # A 的范数
vals, vecs = np.linalg.eig(A)   # 特征值和特征向量
print(np.linalg.svd(A))   # 奇异值分解
```

---

## A.5 随机数

```python
# 设置随机种子，让每次运行的“随机”结果都一样（调试时很有用）
np.random.seed(42)

# 生成 0 到 1 之间的随机小数
print(np.random.rand(2, 3))     # 2 行 3 列
print(np.random.randn(2, 3))    # 标准正态分布（均值 0，标准差 1）

# 生成随机整数
print(np.random.randint(0, 10, size=(3, 3)))  # 0 到 9 之间的随机整数

print(np.random.normal(0, 1, (2, 3)))   # 正态分布，均值 0，标准差 1
print(np.random.uniform(0, 1, (2, 3)))  # 均匀分布，0 到 1 之间

# 从数组中随机选取
arr = np.array([1, 2, 3, 4, 5])
print(np.random.choice(arr, size=3))   # 随机选 3 个元素

np.random.shuffle(arr)   # 打乱数组顺序（原地修改）
print(arr)

print(np.random.permutation(arr))  # 返回打乱后的新数组，原数组不变
```

> 💡 **小窍门**：做机器学习划分训练集/验证集时，经常要用随机数。为了让实验可重复（你和别人跑同样的代码得到同样的结果），记得在开头设 `np.random.seed(42)`，42 只是个常用数字，换成其他数字也行。

---

## A.6 形状操作与合并

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 按行/列合并
print(np.concatenate([a, b]))      # [1 2 3 4 5 6]
print(np.vstack([a, b]))           # [[1 2 3]
                                   #  [4 5 6]]  垂直堆叠
print(np.hstack([a, b]))           # [1 2 3 4 5 6]  水平堆叠
print(np.stack([a, b], axis=1))    # [[1 4]
                                   #  [2 5]
                                   #  [3 6]]  沿新轴堆叠

# 拆分
arr = np.arange(12)
print(np.split(arr, 3))            # 把 arr 分成 3 份
print(np.hsplit(arr.reshape(3, 4), 2))  # 按列分成 2 份
print(np.vsplit(arr.reshape(3, 4), 3))  # 按行分成 3 份

# 重复与扩维
print(np.tile(a, 2))               # [1 2 3 1 2 3]  重复
print(np.repeat(a, 2))             # [1 1 2 2 3 3]  每个元素重复
print(np.expand_dims(a, axis=1))   # [[1]
                                   #  [2]
                                   #  [3]]  增加一维
print(np.squeeze(np.expand_dims(a, axis=1)))  # [1 2 3]  去掉长度为 1 的维度
```

---

## A.7 文件读写

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

# 保存为 .npy 文件（NumPy 原生格式）
np.save('data.npy', a)
a2 = np.load('data.npy')   # 读取 .npy 文件

# 保存多个数组到 .npz 文件
np.savez('data.npz', x=a, y=a + 1)
np.savez_compressed('data_compressed.npz', x=a, y=a + 1)  # 压缩保存

# 从文本文件读取（CSV 等）
# np.genfromtxt('data.csv', delimiter=',')
# np.loadtxt('data.txt')
```

> 💡 **小窍门**：`.npy` 适合保存单个数组，`.npz` 适合保存多个数组。如果数据是 CSV 格式，通常直接用 pandas 的 `read_csv` 更方便。

---

## A.8 小结

本附录快速过了一遍 NumPy 最常用的操作：

1. **数组创建**：`np.array()`、`np.zeros()`、`np.ones()`、`np.arange()`、`np.linspace()`、`np.eye()`、`np.diag()`、`np.full()`、`np.empty()`、`np.random.rand/randint/randn/normal/uniform/choice`。
2. **数组属性**：`.shape`、`.ndim`、`.size`、`.dtype`、`.itemsize`。
3. **索引与切片**：`a[i]`、`a[i, j]`、`a[start:end]`，负数下标从后往前数；花式索引 `a[[0, 2, 4]]`；布尔索引 `a[a > 4]`；`np.where()`。
4. **视图与副本**：切片返回视图，修改会影响原数组；要独立副本用 `.copy()`。
5. **形状变换**：`.reshape()`、`.ravel()`、`.flatten()`、`.T`/`.transpose()`。
6. **数学与统计**：`sum()`、`mean()`、`std()`、`var()`、`min()`、`max()`、`argmin()`、`argmax()`、`median()`、`percentile()`、`diff()`、`cumsum()`、`sort()`、`argsort()`、`unique()`，可配合 `axis=0/1`。
7. **广播与矩阵运算**：`+` `-` `*` `/`、`@` 或 `np.dot()`、`.T`、`np.linalg.inv/det/norm/eig/svd`。
8. **随机数**：`np.random.seed()`、`rand()`、`randn()`、`randint()`、`normal()`、`uniform()`、`choice()`、`shuffle()`、`permutation()`。
9. **形状操作与合并**：`concatenate()`、`vstack()`、`hstack()`、`stack()`、`split()`、`hsplit()`、`vsplit()`、`tile()`、`repeat()`、`expand_dims()`、`squeeze()`。
10. **文件读写**：`np.save()`/`np.load()`、`np.savez()`/`np.savez_compressed()`。

掌握这些，你就能读懂大部分 Kaggle Notebook 里的 NumPy 代码了。

---

> **下一步建议**：NumPy 处理的是“数值数组”，但实际比赛数据往往带有列名、缺失值、类别特征，这时候需要更强大的工具——**pandas**。建议继续阅读附录 B（pandas 快速参考）。