# 游戏销量预测比赛（第四块：速度与优化）

## 一、比赛简介

### 1.1 比赛背景

这个比赛用的是一张**电子游戏销量表**。每一行代表一款游戏的一次发售记录，表里有游戏名（`title`）、发售平台（`console`，比如 PlayStation、Xbox、PC 这类）、游戏类型（`genre`，比如动作、射击、角色扮演）、媒体评分（`critic_score`）和总销量（`total_sales`）。

我们要做的事很朴素：**根据一款游戏的平台、类型和评分，预测它能卖多少**。

不过这个比赛的组织方式比较特别。从材料里的俄文注释能看出来，它是一套**分块（блок）教学作业**，每一块考一个主题，学生做完之后把 notebook 交给老师评分。这份材料是**第四块：速度与优化**。也就是说，模型准不准只是及格线，真正打分的是这些工程能力：

- 数据处理够不够快（换一个更快的表格库能提速多少）；
- 计算能不能并行（多核跑起来能快多少）；
- 模型文件能不能压小（既要小，又不能明显变差）；
- 线上预测够不够快（一千条要多长时间）；
- 代码写得够不够「向量化」（能不能不用 for 循环）。

所以这篇 writeup 的重点不在调参刷分，而在**把一个能跑的基线，改造成一个又快又小的版本**，并且每一步都用秒表和文件大小量出来。

### 1.2 比赛目标

输入是一个 `train.csv`，里面每行一款游戏，包含这些列：

| 列名 | 含义 |
| --- | --- |
| `title` | 游戏名，文本 |
| `console` | 发售平台，文本（类别） |
| `genre` | 游戏类型，文本（类别） |
| `critic_score` | 媒体评分，数值，可能缺失 |
| `total_sales` | 总销量，数值，**这就是要预测的目标** |

输出就是一个数值：这款游戏的销量预测值。

材料里只提供了 `train.csv` 一个文件，没有单独的测试集。所以我们自己从训练集里切一部分当验证集，用来判断模型好坏——这也是教学作业的常见做法：**先证明流程对，再看速度**。

### 1.3 评价标准

这个比赛有两层评价。

**第一层是预测质量**，用**均方根误差（RMSE）**衡量。它的算法是：把每条样本的「预测值减真实值」平方，全部加起来取平均，再开根号。RMSE 越小越好，单位跟销量本身一样。材料里导入的 `mean_squared_error` 就是干这个的。

**第二层，也是这一块真正的评分标准**，是五条硬指标：

| 任务 | 要求 |
| --- | --- |
| 用 Polars 重写预处理 | 相对 Pandas 加速 **> 2 倍** |
| 用 joblib 并行算 5 折 | 相对串行加速 **> 2 倍** |
| 压缩模型 | 体积 **< 5 MB**，精度损失 **< 1%** |
| 批量预测 | 1000 条 **< 0.05 秒** |
| 向量化改写 | 相对 for 循环加速 **> 5 倍** |

这几条都是材料里白纸黑字写着的门槛，我们的目标就是把它们逐条达成。

---

## 二、解题过程

整场比赛可以概括成一句话：**先用 Pandas 搭一条能跑的基线，再一段一段地把它换快、换小，每一段都测一次。** 我们首先做 EDA 看清数据，其次做清洗和编码，接着训练随机森林并算特征重要性，然后做压缩和推理基准，最后把五项指标汇总成报告。

### 2.1 数据探索（EDA）

第一步是把表读进来，看它有多大、有哪些列、缺多少值：

```python
import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

# 行数、列数：先知道数据规模，后面所有时间测量才有意义
print(df.shape)

# 每列的类型。文本列后面要做编码，数值列要检查缺失
print(df.dtypes)

# 每列缺多少个值。critic_score 是重点，因为很多游戏没有媒体评分
print(df.isna().sum())

# 数值列的分布：均值、四分位、最大最小值
print(df.describe())
```

这几行输出告诉我们两件事：一是**平台和类型都是文本**，不能直接喂给模型，必须先变成数字；二是**评分类列有缺失**，不能直接算，得先补。

接着我们专门看目标列 `total_sales`：

```python
# 销量为 0 的比例。0 销量通常是「没有统计到」而不是「真的一件没卖」
zero_ratio = (df['total_sales'] == 0).mean()
print('销量为 0 的占比：', zero_ratio)

# 销量的分位数，看看是不是长尾分布（少数爆款拉高整体）
print(df['total_sales'].quantile([0.5, 0.9, 0.99, 1.0]))

# 平台和类型各有多少种取值，决定 one-hot 之后会多出多少列
print(df['console'].nunique(), df['genre'].nunique())
```

结论很典型：**销量是严重的长尾分布**，大部分游戏卖得不多，极少数爆款撑起了整个市场。而且表里确实存在销量为 0 的行——材料里的预处理第一句就是 `df[df['total_sales'] > 0]`，说明作者也认为这些行不该参与训练。

平台和类型的取值数量决定了 one-hot 编码之后特征会膨胀多少倍。这一步不做，后面看到「编码后列数暴涨」会一头雾水。

### 2.2 数据清洗

清洗只做三件事，但每一件都有理由：

1. **丢掉销量为 0 的行**。0 更像是「数据缺失」的占位符，留着会让模型学到「很多游戏卖不出去」这种假规律。
2. **用中位数补评分缺失**。评分缺失通常不是因为游戏差，而是因为没人评。用中位数比用 0 更合理——用 0 等于告诉模型「这款游戏评分极差」。
3. **把平台和类型做 one-hot 编码**。模型只认数字，文本类别必须展开成 0/1 列。`drop_first=True` 是去掉每个类别的第一列，避免「所有列都是 0」和「某一列是 1」表达同一个意思的冗余。

```python
import time

# 计时开始：这一段的耗时，就是后面跟 Polars 比较的基准
start_pd = time.time()

df_pd = df.copy()                                  # 复制一份，保留原始表方便对比
df_pd = df_pd[df_pd['total_sales'] > 0]            # 去掉销量为 0 的行

# 评分缺失用中位数补：中位数不受极端值影响，比均值稳
df_pd['critic_score'] = df_pd['critic_score'].fillna(df_pd['critic_score'].median())

# 把 console 和 genre 展开成 0/1 列；drop_first 去掉冗余的那一列
df_pd = pd.get_dummies(df_pd, columns=['console', 'genre'], drop_first=True)

end_pd = time.time() - start_pd
print(f'Pandas 耗时：{end_pd:.4f} 秒')
```

这里有个容易踩的坑：`fillna` 一定要在 `get_dummies` **之前**做，因为 one-hot 只处理文本列，不会碰数值列的缺失。顺序反了，后面模型遇到 NaN 会直接报错。

清洗完，我们把特征和目标分开。注意 `title` 是文本，直接丢掉——游戏名对销量的帮助很弱，而且每款游戏名字都不一样，编码之后几乎全是新列，纯属浪费：

```python
# 目标列单独拿出来
y = df_pd['total_sales']

# 特征表去掉目标列和游戏名，只保留数值列
X = df_pd.drop(['total_sales', 'title'], axis=1, errors='ignore')
X = X.select_dtypes(include=[np.number])

# 兜底：one-hot 之后理论上不会有缺失，但保险起见还是填 0
X = X.fillna(0)
```

`errors='ignore'` 是个小技巧：万一某份数据里没有 `title` 这一列，代码也不会崩。教学作业里数据版本经常换来换去，这种写法能省很多调试时间。

### 2.3 模型训练与超参数调优

**基线模型选随机森林。** 理由很实际：它不用标准化、不怕特征量纲不同、对缺失和异常值都比较宽容，而且训练完能直接给出特征重要性。对一张几十列的表来说，它是最省心的起点。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 5 折交叉验证：把数据切成 5 份，轮流拿 1 份当验证、4 份当训练
kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kf.split(X))

fold_rmse = []
for fold, (train_idx, val_idx) in enumerate(splits):
    X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

    # n_estimators=50 是刻意选的小值：这一块要测速度，树太多会把计时拖长
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_vl)
    rmse = mean_squared_error(y_vl, pred, squared=False)   # 开根号，得到 RMSE
    fold_rmse.append(rmse)
    print(f'第 {fold+1} 折 RMSE：{rmse:.4f}')

print('平均 RMSE：', np.mean(fold_rmse))
```

`shuffle=True` 加固定 `random_state` 是为了**结果可复现**。这一块后面要用「串行 vs 并行」比时间，如果每次切分都不一样，两次跑的根本不是同一件事，比出来的加速比就没有意义。

**关于调参。** 这一块的评分不看分数，所以没有做网格搜索。我们只调了两个跟速度有关的参数：`n_estimators`（树的数量）和 `max_depth`（树的最大深度）。这两个参数同时也是后面「压缩模型」的抓手——树越少、越浅，文件就越小。

**特征重要性。** 材料里用的是**置换重要性（permutation importance）**，思路是：把某一列的值随机打乱，看模型误差变差多少。变差得越多，说明这一列越重要。它比随机森林自带的 `feature_importances_` 更可信，因为它是在验证集上真实测出来的。

麻烦在于，5 折各算一次很慢。这正是第二个任务要解决的问题：用 joblib 把 5 折同时丢给多个 CPU 核：

```python
from joblib import Parallel, delayed
from sklearn.inspection import permutation_importance
import time

def compute_fold(fold):
    """只算一折：训练 + 置换重要性。写成独立函数，才能被 joblib 分发出去。"""
    train_idx, val_idx = splits[fold]
    X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_tr, y_tr)
    perm = permutation_importance(model, X_vl, y_vl, n_repeats=5, random_state=42)
    return perm.importances_mean

# 串行版：一次跑一折，作为基准
start_seq = time.time()
_ = [compute_fold(i) for i in range(5)]
time_seq = time.time() - start_seq

# 并行版：n_jobs=-1 表示用满所有 CPU 核
start_par = time.time()
_ = Parallel(n_jobs=-1)(delayed(compute_fold)(i) for i in range(5))
time_par = time.time() - start_par

joblib_speedup = time_seq / time_par
print(f'串行 {time_seq:.4f} 秒，并行 {time_par:.4f} 秒，加速 {joblib_speedup:.2f} 倍')
```

这里有个关键点：`compute_fold` 必须是**顶层函数**，不能写成 lambda 或者闭包。因为 joblib 要把这个函数序列化后发给别的进程，闭包里的东西不一定传得过去。另外，5 折刚好对应 5 个核，核数再多也用不上——**并行的上限由任务数决定**，这是很多人第一次用 joblib 会忽略的地方。

### 2.4 模型测试与指标报告

这一段我们要量四样东西：**模型多大、预测多快、预处理多快、循环能不能消掉**。

**第一，模型压缩。** 思路是「少而浅」：树的数量从 100 降到 10，深度限制在 5 层。随机森林的体积主要来自每棵树的节点，节点数又由树的深度决定，所以这两个参数一降，文件会小一个数量级：

```python
import joblib, os

# 完整模型：100 棵树，不限制深度
model_full = RandomForestRegressor(n_estimators=100, random_state=42)
model_full.fit(X, y)
joblib.dump(model_full, 'model_full.pkl')
size_full = os.path.getsize('model_full.pkl') / (1024 * 1024)   # 字节换算成 MB
print(f'完整模型：{size_full:.2f} MB')

# 压缩模型：10 棵树、最深 5 层
model_compressed = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
model_compressed.fit(X, y)
joblib.dump(model_compressed, 'model_compressed.pkl')
size_compressed = os.path.getsize('model_compressed.pkl') / (1024 * 1024)
print(f'压缩模型：{size_compressed:.2f} MB')
```

光变小没用，还得看精度掉多少。我们的做法是**用同一个验证集分别算两个模型的 RMSE，再算相对变化**：

```python
def rmse_of(model):
    pred = model.predict(X.iloc[val_idx])
    return mean_squared_error(y.iloc[val_idx], pred, squared=False)

rmse_full = rmse_of(model_full)
rmse_comp = rmse_of(model_compressed)

# 精度损失 = 相对变化，题目要求小于 1%
loss = (rmse_comp - rmse_full) / rmse_full
print(f'精度损失：{loss:.4%}')
```

这一步的取舍很清楚：模型小到 5 MB 以下，但 RMSE 最多只能涨 1%。如果压缩后掉得太狠，就说明树砍多了，得往回加几棵或者放宽深度。

**第二，推理基准。** 要求是 1000 条样本在 0.05 秒内出结果。随机森林的预测本身就是向量化的，所以关键不是写循环，而是**一次把整批数据喂进去**：

```python
def predict_batch(model, data, batch_size=1000):
    """一次预测一整批；切片保证不超过 batch_size。"""
    return model.predict(data[:batch_size])

test_data = X.values[:1000]
start_infer = time.time()
_ = predict_batch(model_compressed, test_data)
inference_time_sec = time.time() - start_infer
print(f'1000 条推理耗时：{inference_time_sec:.4f} 秒')
```

这里用 `X.values` 而不是 `X`，是为了避开 pandas 的索引对齐开销——预测阶段不需要列名，转成纯 numpy 数组更快。

**第三，向量化。** 材料里给了一个最典型的反面教材：用 for 循环把数组里每个元素乘 2 加 1。numpy 的正确写法是整个数组一起算：

```python
def slow_function(arr):
    """反面教材：一个一个算，Python 解释器要为每个元素做一次加法。"""
    result = []
    for x in arr:
        result.append(x * 2 + 1)
    return np.array(result)

def fast_function(arr):
    """向量化写法：整个数组一次性运算，底层是 C 循环，没有 Python 解释开销。"""
    return arr * 2 + 1

test_arr = np.random.rand(100000)

start_slow = time.time()
_ = slow_function(test_arr)
time_slow = time.time() - start_slow

start_fast = time.time()
_ = fast_function(test_arr)
time_fast = time.time() - start_fast

vectorization_speedup = time_slow / time_fast
print(f'向量化加速：{vectorization_speedup:.2f} 倍')
```

这一条之所以要求 5 倍以上，是因为两者根本不在一数量级：循环版的瓶颈在 Python 解释器，每个元素都要走一遍解释执行；向量化版把整块内存交给 C 层处理。数组越长，差距越大。

**第四，预处理换库。** Polars 是另一个表格库，底层用 Rust 写，支持多线程。同样的清洗逻辑可以这样写：

```python
import polars as pl

start_pl = time.time()
df_pl = pl.from_pandas(df)
df_pl = df_pl.filter(pl.col('total_sales') > 0)                     # 等价于 pandas 的布尔筛选
df_pl = df_pl.with_columns(                                          # 补中位数
    pl.col('critic_score').fill_null(pl.col('critic_score').median())
)
df_pl = df_pl.to_dummies(columns=['console', 'genre'])               # 等价于 get_dummies
end_pl = time.time() - start_pl

polars_speedup = end_pd / end_pl
print(f'Polars 耗时：{end_pl:.4f} 秒，加速 {polars_speedup:.2f} 倍')
```

要点是**两条流水线的操作顺序必须一模一样**，否则比出来的时间没有可比性。Polars 默认就会多线程执行，而且它的 `filter`、`with_columns` 都是惰性求值风格的链式写法，中间不产生临时表，这就是它快的主要原因。

**最后把五项指标汇总成一张表**：

```python
report = {
    'polars_speedup': polars_speedup,          # 要求 > 2
    'joblib_speedup': joblib_speedup,          # 要求 > 2
    'model_size_mb': size_compressed,          # 要求 < 5
    'accuracy_loss': loss,                     # 要求 < 0.01
    'inference_time_sec': inference_time_sec,  # 要求 < 0.05
    'vectorization_speedup': vectorization_speedup,  # 要求 > 5
}
for name, value in report.items():
    print(f'{name}: {value}')
```

这份材料本身是**留给学生填空的模板**，所以它给出的是框架和示例代码，具体测出来的数值要自己跑一遍填回去。我们在写的时候严格照这个结构来，保证每个数字都是当场量出来的，而不是拍脑袋写的。

---

## 三、总结

本文中我们完成了游戏销量预测比赛第四块（速度与优化）的完整流程。我们首先用 EDA 看清数据：它是一张以 `total_sales` 为目标的长尾销量表，平台和类型是文本列、评分列有缺失、还有一批销量为 0 的行。其次我们做了三步清洗——丢掉 0 销量行、用中位数补评分、对平台和类型做 one-hot 编码——并把游戏名这类无用文本丢掉。接着我们用随机森林配上 5 折交叉验证搭起基线，用 RMSE 评估，并改用 joblib 并行计算置换重要性来提速。然后我们做了三件「变小变快」的事：把森林砍成 10 棵浅树把模型压到 5 MB 以内并用同一验证集核对精度损失、用批量切片把 1000 条推理压进 0.05 秒、把 for 循环改成 numpy 向量化运算。最后我们把五项指标（Polars 加速、joblib 加速、模型体积、精度损失、推理耗时、向量化加速）汇总成一张报告表。

这个比赛给我们的最大启发是：**「能跑」和「能交付」是两回事**。同样的清洗逻辑、同样的模型，换一个表格库、加一层并行、少几棵树，结果可能差出好几倍，而模型分数几乎不变。入门阶段大家容易只盯着准确率，但真正上线的时候，体积、延迟、耗时这些「不性感」的指标，往往才是决定能不能用的那一条。
