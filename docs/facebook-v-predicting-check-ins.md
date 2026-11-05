# Facebook V: Predicting Check Ins（预测签到地点）

## 一、比赛简介

### 1.1 比赛背景

Facebook 虚拟构造了一个「世界」：在一个 **10km × 10km** 的正方形区域里，放了 **超过 10 万个地点（place）**，然后模拟了一部手机信号定位数据。这份数据故意做得很「脏」——坐标带噪声、有误差，贴近真实场景中「定位不准」的情况，用来考验参赛者处理带噪声定位数据的能力。

比赛数据是这一段时间内全部用户的签到（check-in）记录：每次签到给出了这次签到的坐标、定位精度和时间，但**没有告诉你用户签到的到底是哪个商家（place_id）**，这正是我们要预测的东西。

材料里给了一个关键提示：训练集中有整整 **2911 万个**签到样本、10 万个类别（`place_id`）——类别数量多到大部分监督学习算法无法直接在整张表上训练，所以主流思路是**把地图切成小块，只在局部小格子内做分类**。

### 1.2 比赛目标

给定一次签到事件的坐标和时间，预测用户所在的地点是哪一家，并且要给出**最可能的 3 个候选地点**（按概率从高到低排列）。

数据列含义：

- `row_id`：签到事件编号；
- `x`、`y`：坐标，取值都在 0～10 之间，单位是公里；
- `accuracy`：定位精度，数值越大定位越不准；
- `time`：签到时间戳；
- `place_id`：地点编号，这是我们要预测的目标（测试集里没有这一列）。

训练集约 2911 万行，测试集约 860 万行，每个测试样本要在提交文件里填上「空格分隔的 3 个地点编号」。

### 1.3 评价标准

比赛采用 **MAP@3（Mean Average Precision @ 3，前 3 名的平均准确率）**。它比只看第一次预测对没对要温和一点：如果正确答案排在提供的第一位，得 1 分；排在第二位，得 1/2；排在第三位，得 1/3；三个都没排进去就不得分。MAP@3 就是所有测试样本得分的平均，越接近 1 越好。这要求模型不能只给出一个「最有把握」的答案，还要把后面两个备选也排对。

---

## 二、解题过程

我们的整体策略是：**先把 10km×10km 的地图用网格切成很多小格子，再把每个格子里的样本单独拿出来分类**。这样每个小格子里的类别（地点）只有几十个，问题规模立刻变小，K 近邻（KNN）之类的方法就变得可行了。我们按四步走。

### 2.1 数据探索（EDA）

我们首先把数据读进来，观察基本统计量：

```python
import numpy as np
import pandas as pd

train = pd.read_csv('../input/train.csv')

# 看每个字段的统计量
train.describe()
# row_id 均值、x/y 都在 0~10、
# accuracy 最小 1、最大 1033、
# time 最小 1、最大 786239
print("不同地点的数量:", train['place_id'].nunique())
# 108390，正好印证了「10 万个类」
```

数据的关键事实浮出水面：

- 坐标 `x`、`y` 都在 0～10 之间，验证了 10km×10km 的地图设定；
- `time` 的范围是 1～786239，`accuracy` 的范围是 1～1033；
- 训练集有 **108390** 个不同的 `place_id`。

接下来我们调查 `time` 列的单位——这决定了后面要做哪种时间特征：

```python
# 把时间按 5000 份分段，画直方图看有没有周期性
train['timeBin'] = (train['time'] / (786239.0 / 5000)).apply(int)
timeHist = train['timeBin'].value_counts()
timeHist.sort_index().plot()
```

直方图上有规律的「小尖峰」，尖峰之间的间隔大约是 10000。直觉提示：**一周有 10080 分钟**，于是我们用「分钟」假设做验证——如果 `time` 的单位是分钟，那把每周每个小时的人数画出来，应该呈 24 小时周期并且一周首尾能接上：

```python
# 假设 time 单位是分钟：除以 60 得到小时，再对每周 168 小时计数
hourly = (train['time'] / 60).apply(int)
act = [len(hourly[(hourly % 168) == i]) for i in range(168)]
pd.Series(act).plot()   # 出现 24 小时周期、周末和工作日差异，一周首尾相近
```

对比「秒」假设画出来的图（完全看不出 24 小时周期，和随机图没法区分），「分钟」假设显然更合理。**结论：`time` 的单位是分钟。** 之后我们就可以放心地把时间拆成「一天中的小时」「一周中的星期几」等特征。

EDA 还告诉我们两个建模要点：一是不同地点的签到量相差巨大（有的地点几千次、有的只有几次），少数派地点很难学；二是地点在空间上是**成团聚集**的，同一个地点的坐标方差往往很小，所以「空间近邻」是极强的信号。

### 2.2 数据清洗与特征工程

清洗方面：坐标只落在 0～10 的格子内，几乎没有明显异常值；真正要下功夫的是**特征工程**。

我们把时间戳（分钟数）换算成具体日期，再抽出小时、星期、月份等面向周期的特征。注意到所有特征乘上了一组「特征权重」`fw`——这是为了让不同单位的特征在距离计算时**处于同一量级**，例如 `x`、`y` 范围是 0～10，而 `accuracy` 最大上千，如果不缩放，距离就会被 `accuracy` 主导：

```python
def prepare_data(df, n_cell_x=20, n_cell_y=40):
    # 特征权重：把 x、y、hour、weekday、day、month、year 缩放到合适的尺度
    # 这组数值是通过实验调出来的（作者戏称为"黑魔法"）
    fw = [500, 1000, 4, 3, 1./22., 2, 10]
    df.x = df.x.values * fw[0]
    df.y = df.y.values * fw[1]

    # 把时间戳（分钟）换算成日期，2014-01-01 作为起点
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]')
    d_times = pd.DatetimeIndex(
        initial_date + np.timedelta64(int(mn), 'm') for mn in df.time.values
    )
    # 抽出周期特征并乘上权重
    df['hour']    = d_times.hour * fw[2]          # 一天中的小时
    df['weekday'] = d_times.weekday * fw[3]       # 一周中的第几天
    df['day']     = (d_times.dayofyear * fw[4]).astype(int)  # 一年中的第几天
    df['month']   = d_times.month * fw[5]         # 月份
    df['year']    = (d_times.year - 2013) * fw[6] # 相对 2013 的年差
    df = df.drop(['time'], axis=1)
    return df
```

这里有一个被许多参赛者验证过的直觉：**地点与时间强相关**——同一家餐厅、酒吧，用户往往在特定时段签到，所以「星期几 + 几点」这类时间特征对区分地点很有用。

更进一步的方案会把时间做成「循环」特征，用 `sin` / `cos` 表示，避免「23:59」和「00:01」被当成巨远两点：

```python
def feature_engineering(df):
    # 一日分成 288 段（每 5 分钟一段），用 sin/cos 表示，头尾是连续的
    minute = 2 * np.pi * ((df["time"] // 5) % 288) / 288
    df['minute_sin'] = (np.sin(minute) + 1).round(4)
    df['minute_cos'] = (np.cos(minute) + 1).round(4)
    # 一年的第几天、一周的第几天也各自做成 sin/cos
    day = 2 * np.pi * ((df['time'] // 1440) % 365) / 365
    df['day_of_year_sin'] = (np.sin(day) + 1).round(4)
    df['day_of_year_cos'] = (np.cos(day) + 1).round(4)
    # 定位精度取对数，压缩极端值
    df['accuracy'] = np.log10(df['accuracy'])
    return df
```

同时我们把地图切成 20×40 的网格，给每个点打上「格子号」，并在每个格子里**砍掉签到次数太少的地点**（这些地点样本不够学出可靠模型，留着只会引入噪声）：

```python
def prepare_data_with_grid(df, n_cell_x=20, n_cell_y=40):
    # 把 0~10 的坐标投影到 20×40 的格子上
    size_x, size_y = 10. / n_cell_x, 10. / n_cell_y
    eps = 0.00001
    xs = np.where(df.x.values < eps, 0, df.x.values - eps)
    ys = np.where(df.y.values < eps, 0, df.y.values - eps)
    pos_x = (xs / size_x).astype(int)
    pos_y = (ys / size_y).astype(int)
    df['grid_cell'] = pos_y * n_cell_x + pos_x
    return df

# 每个格子内只保留出现 ≥5 次的地点
train = train[train.groupby('place_id')['x'].transform('count') >= 5]
```

### 2.3 模型训练与超参数调优

我们使用 **K 近邻（KNN）** 作为主力模型。理由很实在：签到数据里「离目标最近的那些历史签到」是预测哪个地点的最强证据——一个坐标点周围最近的 25 条历史签到都来自某家咖啡店，那这次签到也大概率是这家店。KNN 记住训练点，按距离找「最近邻居」，再让邻居们投票。

为了让远近邻居影响力不同，我们给 KNN 配上「距离加权」（越近权重越大），距离度量用曼哈顿距离。每个格子单独训练一个 KNN：

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

def process_one_cell(df_train, df_test, grid_id, th=5):
    # 只取当前格子内的训练与测试数据
    df_cell_train = df_train.loc[df_train.grid_cell == grid_id]
    # 砍掉签到次数少于 th 的地点
    place_counts = df_cell_train.place_id.value_counts()
    df_cell_train = df_cell_train.loc[
        place_counts[df_cell_train.place_id.values] >= th]
    df_cell_test = df_test.loc[df_test.grid_cell == grid_id]
    row_ids = df_cell_test.index

    # 用 LabelEncoder 把 10 位数的地点编号编码成连续整数
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id', 'grid_cell'], axis=1).values.astype(int)
    X_test = df_cell_test.drop(['grid_cell'], axis=1).values.astype(int)

    # 距离加权的 KNN，取 25 个最近邻居，曼哈顿距离
    clf = KNeighborsClassifier(n_neighbors=25, weights='distance',
                               metric='manhattan')
    clf.fit(X, y)
    # 取概率最高的 3 个地点
    y_pred = clf.predict_proba(X_test)
    pred_labels = le.inverse_transform(
        np.argsort(y_pred, axis=1)[:, ::-1][:, :3])
    return pred_labels, row_ids
```

需要说明超参数是怎么定的：`n_neighbors=25`、`weights='distance'`、`metric='manhattan'` 以及前面那组特征权重，是材料里的数据老手通过大量网格搜索 + 本地验证比较出来的经验值。基础版用 25 个邻居，增强版格子内核（加时间切分、边界扩充）会把邻居数调到 37；邻居数太少噪声大、太多又看不清局部结构，25～37 是实测比较稳的范围。

主流程是遍历所有格子：先按格子分配测试样本，其次在每个格子里训练并预测，然后把各格子的结果拼回去，形成每个样本的 3 个候选地点：

```python
def process_grid(df_train, df_test, n_cell_x=20, n_cell_y=40, th=5):
    """遍历 20×40=800 个格子，聚合预测结果并生成提交文件"""
    preds = np.zeros((df_test.shape[0], 3), dtype=int)
    for g_id in range(n_cell_x * n_cell_y):
        pred_labels, row_ids = process_one_cell(df_train, df_test, g_id, th)
        preds[row_ids] = pred_labels            # 把该格子的预测写回对应行
    return preds
```

增强版还会把时间维度也切成几段（`t_cuts`），并在格子边界外再扩一圈邻居数据（`x_border_aug`、`y_border_aug`）——因为格子边缘的样本，它的近邻很可能在隔壁格子里，扩边界后边缘样本的预测会稳很多。

**验证集怎么来？** 比赛不允许我们随便拿测试集看分数，所以我们在训练集上按「第几天」切分：前面 N 天做训练，后面留几天做验证，保证验证集的时间在训练集之后，模拟真实预测场景：

```python
def validation_split(df, val_start_day):
    """把按分钟记录的时间换算成天数，按天切出验证集"""
    day = df['time'] // 1440            # 一天 1440 分钟
    df_val = df.loc[day >= val_start_day].copy()
    df = df.loc[day < val_start_day].copy()
    return df, df_val
```

### 2.4 模型测试与指标报告

我们用 MAP@3 自定义打分函数来评估。打分逻辑正是比赛规则：正确答案排第一得 1 分、排第二得 1/2、排第三得 1/3：

```python
def mapk_precision(labels, preds):
    """根据验证集真实地点 labels 和预测的前 3 地点列表算 MAP@3"""
    scores = []
    for true, pred_3 in zip(labels, preds):
        s = 0.0
        for i, p in enumerate(pred_3):          # i = 0,1,2 对应排名
            if p == true:
                s += 1.0 / (i + 1)              # 第 1/2/3 名分别加 1, 1/2, 1/3
                break
        scores.append(s)
    return np.mean(scores)
```

在按天切出的验证集上（用最后一段时间的签到当验证、前面的当训练），增强版 KNN 流水线的得分为 **0.531413（MAP@3）**。这个分数怎么解读？随机瞎猜 3 个地点，期望得分约为 3/108390，几乎为 0；0.53 意味着平均每个样本的第一个候选有相当高比例直接命中正确答案，剩下的靠第 2、3 名补充，说明「空间 + 时间近邻」这套信号非常有效。

确定模型可用后，我们把整份训练数据（不再留验证）全部拿去训练，对 860 万个测试样本做同样流程，按 `row_id,place_id` 的格式输出，`place_id` 部分是 3 个编号用空格连接：

```python
# 生成提交文件：注意一行里要写空格分隔的 3 个候选地点
ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')
ds_sub.name = 'place_id'
ds_sub.to_csv('sub_knn.csv', index=True, header=True, index_label='row_id')
```

---

## 三、总结

本文中我们完成了 Facebook 签到地点预测的完整流程。我们首先做了 EDA，确认训练集有 2911 万条签到、10.8 万个地点，并且通过直方图的周期性和每周活动曲线论证了**时间戳的单位是分钟**。其次我们做了特征工程，把时间拆成小时、星期、月等周期性特征（还尝试了 sin/cos 循环编码），对精度取对数，并把 10km×10km 的世界按 20×40 网格切块、砍掉低频地点。接着我们在每个小格子里训练带距离加权的 KNN（曼哈顿距离、25 个近邻），并介绍了按天切验证集、边界扩充、时间切分等实践经验。最后我们用自定义 MAP@3 打分函数在验证集上测得约 **0.5314** 分。这条「分而治之 + 近邻投票」的路子告诉我们：当类别数量极大时，与其在全局硬训一个 10 万类的分类器，不如把空间切开、在局部用最简单的模型把问题解决掉。