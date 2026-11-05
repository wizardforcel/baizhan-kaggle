# Introducing Kaggle Scripts

## 一、比赛简介

### 1.1 比赛背景

Introducing Kaggle Scripts 是 Kaggle 在 2015 年推出 Scripts 功能时开设的演示性比赛/练习场。它的目的不是解决某个具体预测任务，而是让用户在 Kaggle 提供的容器环境里编写并运行脚本，熟悉数据读取、依赖库和输出文件的流程。

材料中包含大量参赛脚本，有的只是打印 `Hello World`，有的是环境包清单，也有高性能计算脚本，例如用分段埃拉托斯特尼筛法在几秒内找出数千万个素数。

### 1.2 比赛目标

参赛者在 Kaggle 容器中提交一段可运行的脚本，脚本读取 `../input` 下的数据（如果有），并把结果写入 `submission.csv`。由于是开放性练习，没有固定的输入输出格式。

### 1.3 评价标准

该比赛没有统一的预测准确率指标，重点在于脚本能否在运行时限内成功执行、输出文件是否符合磁盘限制。Python Sieve 脚本的作者特别提到 Kaggle Scripts 有 512MB 磁盘限制，因此只计算到约 9.45 亿以控制输出大小。

## 二、解题过程

我们以「Python Sieve：4 秒找出 4800 万素数」为主线，展示如何在一个受资源限制的脚本环境中完成高性能计算。

### 2.1 数据探索（EDA）

我们先了解运行环境：Python 版本、已安装的包、CPU 数量。

```python
# 列出 Kaggle 容器里已安装的包，方便确认可用依赖
import pip
import pandas as pd

output_list = []
for package in sorted(pip.get_installed_distributions(), key=lambda x: x.project_name):
    output_list.append((package.project_name, package.version))

df = pd.DataFrame(output_list, columns=['pip_project_name', 'version'])
print(df.head(3))
df.to_csv("submission.csv", index=False)  # 输出到当前目录
```

这段脚本说明 Kaggle 环境自带常用科学计算库。接着我们查看 CPU 核数，决定并行策略。

```python
import multiprocessing
n_CPUs = multiprocessing.cpu_count()
print("Number of CPUs detected:", n_CPUs)
```

### 2.2 数据清洗

本题的「数据清洗」主要针对输出：素数结果要写成紧凑的 CSV，且不能超过 512MB。我们先实现整数到 ASCII 的快速转换。

```python
import numpy

def ints_tostring(ints_arr, line_chars=10):
    # 把整数数组转成 ASCII 字节串，每行一个数
    # 比 numpy 的 tofile 更快，也方便直接写入 CSV
    buf = numpy.zeros(shape=(len(ints_arr), line_chars), dtype=numpy.int8)
    buf[:, line_chars-1] = 10  # 10 是换行符
    for buf_ix in range(line_chars-2, -1, -1):
        numpy.mod(ints_arr, 10, out=buf[:, buf_ix])
        buf[:, buf_ix] += 48  # 48 是字符 '0' 的 ASCII
        ints_arr //= 10
    return buf.tostring()
```

这样避免了逐行格式化带来的巨大开销。

### 2.3 模型训练与超参数调优

这里的「模型」是素数筛算法。我们用分段筛法，把大区间切成小段，每段独立计算。

```python
import math

def sieve_primes(n):
    # 找出 n 以内的素数，只跟踪奇数以节省一半内存
    sieve = numpy.ones(n // 2, dtype=bool)
    limit = int(math.sqrt(n)) + 1
    for i in range(3, limit, 2):
        if sieve[i // 2]:
            # 从 i*i 开始把 i 的倍数标记为非素数
            sieve[i*i // 2 :: i] = False
    prime_indexes = numpy.nonzero(sieve)[0][1:]
    primes = 2 * prime_indexes.astype(numpy.int32) + 1
    return primes
```

我们把总区间按 CPU 数切分，用 `multiprocessing.Pool` 并行处理。

```python
def multiprocess_calc_primes(n, n_processes, file_name):
    seg_size = n // n_processes
    seg_ranges = [(s, min(n, s+seg_size)) for s in range(2, n, seg_size)]

    # 每个进程负责一个区间，返回该区间的素数串
    processes = multiprocessing.Pool(n_processes)
    prime_strings = processes.map(seg_prime_string, seg_ranges)

    with open(file_name, "wb") as fout:
        for prime_string in prime_strings:
            fout.write(prime_string)
```

超参数调优体现在：分段大小与进程数匹配 CPU 核数，上界设为 9.45 亿以控制输出约 4800 万行，刚好落在 512MB 限制内。

### 2.4 模型测试与指标报告

我们用计时器测量单进程与多进程的运行时间，验证结果正确性与性能。

```python
from time import time

start = time()
primes = sieve_primes(10_000_000)
elapsed = time() - start
print(f"找到 {len(primes)} 个素数，用时 {elapsed:.3f} 秒")
```

最终脚本在 Kaggle 上运行约 3 到 4 秒，输出 4800 万行素数，文件略小于 512MB。

```python
# 命令行入口
if __name__ == '__main__':
    upper_bound = (1000 - 55) * 1_000_000  # 9.45 亿
    multiprocess_calc_primes(upper_bound, multiprocessing.cpu_count(), 'primes_multiprocess.csv')
```

## 三、总结

本文中我们完成了 Introducing Kaggle Scripts 中的高性能脚本实践。我们首先用 EDA 检查容器环境、CPU 数量与可用包。其次我们实现了紧凑的整数转 ASCII 输出，控制文件大小。接着我们用分段埃拉托斯特尼筛法配合多进程并行计算素数。最后计时验证脚本在数秒内完成并满足 512MB 磁盘限制。这个练习说明，在受限的计算环境里，算法选择与输出格式同样重要。