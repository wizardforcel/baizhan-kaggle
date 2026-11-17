# 附录B NumPy快速入门

NumPy（全称 Numerical Python）是 Python 里做科学计算的基础库，几乎所有机器学习库（pandas、scikit-learn、PyTorch、TensorFlow）都在它的基础上构建。如果你想在 Kaggle 上处理表格数据、图像、或者做任何数值计算，NumPy 是绕不过去的工具。

本附录面向零基础初学者，用大白话讲清楚 NumPy 最核心的概念：什么是数组，怎么创建数组，怎么做运算，怎么索引切片，怎么改变形状。

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

## B.1 什么是 NumPy 数组

Python 自带的列表（list）可以存任意类型的数据，但做数值计算又慢又不方便。NumPy 提供了一种新的容器叫 **ndarray**（n-dimensional array，N维数组），简称“数组”。

数组和列表最大的区别有两点：

1. **元素类型必须相同**：数组里所有元素都得是同一种类型（全是整数，或全是浮点数，或全是字符串）。这样 NumPy 才能在底层用紧凑的方式存储，计算速度快得多。
2. **支持向量化运算**：你可以直接对整个数组做加减乘除，而不需要写 `for` 循环一个一个元素地算。

### 安装 NumPy

如果你还没安装，打开命令行输入：

```bash
pip install numpy
```

如果你在 Kaggle Notebooks 里写代码，NumPy 已经预装好了，直接 `import` 即可。

### 导入 NumPy

约定成俗，大家都给 NumPy 起个别名 `np`，以后代码里就用 `np.` 开头：

```python
import numpy as np
```

---

## B.2 创建数组

### 1. 从列表创建：`np.array()`

最常用的方法，就是把 Python 列表转成 NumPy 数组。

```python
import numpy as np

# 一维数组（就像一个列表）
a = np.array([1, 2, 3, 4, 5])
print(a)           # [1 2 3 4 5]
print(type(a))     # <class 'numpy.ndarray'>
print(a.shape)     # (5,)  表示1维，有5个元素

# 二维数组（就像一个表格/矩阵）
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]
print(b.shape)     # (2, 3)  2行3列
```

> 💡 **大白话解释**：`shape` 告诉你数组的“形状”。一维数组的 shape 是 `(元素个数,)`，二维数组的 shape 是 `(行数, 列数)`。

### 2. 创建全0或全1数组：`np.zeros()` / `np.ones()`

```python
# 创建一个 3行4列 的全0数组
zeros = np.zeros((3, 4))
print(zeros)
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 创建一个 2行3列 的全1数组
ones = np.ones((2, 3))
print(ones)
# [[1. 1. 1.]
#  [1. 1. 1.]]
```

注意：`zeros` 和 `ones` 默认创建的是**浮点数**（`0.0`、`1.0`），不是整数。

### 3. 创建等差数列：`np.arange()` / `np.linspace()`

```python
# 0 到 9，步长为1
print(np.arange(10))      # [0 1 2 3 4 5 6 7 8 9]

# 1 到 9，步长为2（不包含10）
print(np.arange(1, 10, 2))  # [1 3 5 7 9]

# 在 0 到 1 之间均匀取 5 个数（包含两端）
print(np.linspace(0, 1, 5))  # [0.   0.25 0.5  0.75 1.  ]
```

`arange` 适合“已知步长”的情况，`linspace` 适合“已知起止和个数”的情况。

### 4. 创建随机数：`np.random`

```python
# 生成一个 2行3列 的随机数（0到1之间的小数）
print(np.random.rand(2, 3))

# 生成一个 3x3 的随机整数（0到9之间）
print(np.random.randint(0, 10, size=(3, 3)))

# 设置随机种子，让每次运行的“随机”结果都一样（调试时很有用）
np.random.seed(42)
print(np.random.rand(3))
```

> 💡 **小窍门**：做机器学习划分训练集/验证集时，经常要用随机数。为了让实验可重复（你和别人跑同样的代码得到同样的结果），记得在开头设 `np.random.seed(42)`，42 只是个常用数字，换成其他数字也行。

---

## B.3 数组的数据类型

NumPy 数组里所有元素类型必须一致。创建时可以用 `dtype` 参数指定：

```python
a = np.array([1, 2, 3], dtype='float64')
print(a)       # [1. 2. 3.]
print(a.dtype) # float64

b = np.array([1, 2, 3], dtype='int32')
print(b.dtype) # int32
```

常用的 `dtype`：

| dtype | 说明 |
|-------|------|
| `int32` / `int64` | 32位/64位整数 |
| `float32` / `float64` | 32位/64位浮点数 |
| `bool` | 布尔值 |
| `object` | Python 对象（任意类型） |

如果你不指定，NumPy 会自动推断：全是整数就用 int，有小数就用 float。

---

## B.4 数组索引与切片

### 1. 一维数组索引

和 Python 列表一样，下标从 0 开始，也可以用负数从后往前数：

```python
a = np.array([10, 20, 30, 40, 50])

print(a[0])   # 10   第一个元素
print(a[1])   # 20   第二个元素
print(a[-1])  # 50   最后一个元素
print(a[-2])  # 40   倒数第二个元素
```

### 2. 切片：取一部分元素

切片语法是 `数组[开始:结束:步长]`，开始包含，结束不包含。

```python
a = np.array([10, 20, 30, 40, 50])

print(a[1:4])    # [20 30 40]   从下标1到下标3（不含4）
print(a[:3])     # [10 20 30]   从开头到下标2
print(a[2:])     # [30 40 50]   从下标2到末尾
print(a[::2])    # [10 30 50]   每隔一个取一个
print(a[::-1])   # [50 40 30 20 10]  倒序
```

### 3. 二维数组索引

二维数组用 `数组[行, 列]` 来索引：

```python
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(b[0, 0])    # 1   第一行第一列
print(b[0, 2])    # 3   第一行第三列
print(b[2, 1])    # 8   第三行第二列
print(b[-1, -1])  # 9   最后一行最后一列
```

二维数组切片：

```python
print(b[0:2, 0:2])   # [[1 2]
                     #  [4 5]]   前2行前2列
print(b[1, :])       # [4 5 6]   第二行所有列
print(b[:, 1])       # [2 5 8]   第二列所有行
```

> ⚠️ **重要陷阱**：NumPy 的切片返回的是**视图**（view），不是副本。也就是说，你通过切片修改了元素，原数组也会跟着变。如果你想要一个完全独立的副本，要用 `.copy()`：

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

## B.5 数组运算：向量化

NumPy 最强大的地方就是**向量化运算**：你可以直接对整个数组做加减乘除，不需要写 `for` 循环。

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)   # [5 7 9]   对应元素相加
print(a - b)   # [-3 -3 -3]
print(a * b)   # [4 10 18] 对应元素相乘
print(a / b)   # [0.25 0.4 0.5]

print(a + 10)  # [11 12 13] 每个元素都加10
print(a * 2)   # [2 4 6]    每个元素都乘2
```

> **大白话**：传统的 Python 列表不能直接做这种“对应元素相加”，你得写 `for` 循环一个一个加。NumPy 数组直接用 `+` 号，代码短而且运行快。

### 矩阵乘法：`@` 或 `np.dot()`

如果你要做真正的矩阵乘法（不是对应元素相乘），用 `@` 运算符：

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(A @ B)   # [[19 22]
               #  [43 50]]
```

---

## B.6 常用聚合函数

NumPy 提供了很多现成的统计函数，可以直接对整个数组或某一轴求值：

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.sum())        # 21   所有元素之和
print(a.mean())       # 3.5  所有元素的平均值
print(a.max())        # 6    最大值
print(a.min())        # 1    最小值
print(a.std())        # 标准差
print(a.median())     # 中位数

# 按列或按行求和（axis=0 表示按列，axis=1 表示按行）
print(a.sum(axis=0))  # [5 7 9]   每一列加起来
print(a.sum(axis=1))  # [6 15]    每一行加起来
```

`axis` 参数容易搞混，记住这个口诀：

- `axis=0`：**竖着**消掉行方向，对每一列做运算（按列）。
- `axis=1`：**横着**消掉列方向，对每一行做运算（按行）。

---

## B.7 布尔索引：根据条件筛选元素

布尔索引就是“只取满足条件的元素”，这在数据清洗里非常常用。

```python
a = np.array([1, 5, 3, 8, 2, 7])

# 找出所有大于 4 的元素
mask = a > 4        # mask 是 [False  True False  True False  True]
print(a[mask])      # [5 8 7]

# 或者直接一步到位
print(a[a > 4])     # [5 8 7]
```

多个条件可以用 `&`（且）和 `|`（或），注意要加括号：

```python
print(a[(a > 2) & (a < 7)])   # [5 3 5]  大于2且小于7
print(a[(a == 1) | (a == 8)]) # [1 8]   等于1或等于8
```

---

## B.8 改变数组形状：reshape / flatten / transpose

```python
a = np.arange(12)   # [0 1 2 ... 11]

# 改成 3行4列
b = a.reshape((3, 4))
print(b.shape)      # (3, 4)

# 展平成一维
c = b.flatten()
print(c.shape)      # (12,)

# 转置（行列互换）
d = b.T
print(d.shape)      # (4, 3)
```

`reshape` 要求新形状的元素总数必须和原来一样（比如 12 个元素只能变成 3×4、2×6、1×12 等，不能变成 3×5）。

---

## B.9 小结

本附录快速过了一遍 NumPy 最常用的操作：

1. **创建数组**：`np.array()`、`np.zeros()`、`np.ones()`、`np.arange()`、`np.linspace()`、`np.random.rand/randint()`。
2. **查看形状**：`.shape`，一维是 `(n,)`，二维是 `(行, 列)`。
3. **数据类型**：`.dtype`，创建时可用 `dtype` 指定。
4. **索引与切片**：`a[i]`、`a[i, j]`、`a[start:end]`，负数下标从后往前数。
5. **视图与副本**：切片返回视图，修改会影响原数组；要独立副本用 `.copy()`。
6. **向量化运算**：`+` `-` `*` `/` 直接对整个数组操作，不用写 `for` 循环。
7. **矩阵乘法**：`@` 或 `np.dot()`。
8. **聚合函数**：`sum()`、`mean()`、`max()`、`min()`、`std()`、`median()`，可配合 `axis=0/1`。
9. **布尔索引**：`a[a > 4]` 取满足条件的元素。
10. **形状变换**：`reshape()`、`flatten()`、`.T` 转置。

掌握这些，你就能读懂大部分 Kaggle Notebook 里的 NumPy 代码了。

---

> **下一步建议**：NumPy 处理的是“数值数组”，但实际比赛数据往往带有列名、缺失值、类别特征，这时候需要更强大的工具——**pandas**。建议继续阅读附录C（pandas 数据处理快速入门）。