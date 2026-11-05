# CareerCon 2019：帮助机器人导航（Help Navigate Robots）

## 一、比赛简介

### 1.1 比赛背景

CareerCon 2019 是一道**传感器时间序列分类**题。想象一个四足机器人正在各种地面上行走，它身上装着惯性测量单元（IMU，就是手机里用来感知倾斜和晃动的那个东西），每时每刻记录机器人的姿态和运动。

比赛给出的任务是：**根据一小段 IMU 读数，判断机器人脚下踩的是什么地面**。地面一共有 9 种，比如混凝土（concrete）、软 PVC（soft_pvc）、木地板（wood）、瓷砖（tiled）、地毯（carpet）、硬瓷砖（hard_tiles）等等。

这个问题为什么重要？因为机器人如果能知道自己踩在什么材质上，就能调整步态：在地毯上可以走得随意些，在光滑瓷砖上就要放慢、防止打滑。所以这是一道很有实际意义的「感知 + 分类」题。

它的主要难点有三个：

1. **数据是时间序列**，每条样本不是一行而是一小段 128 个时刻的读数，需要先把一段信号压缩成特征；
2. **样本是按「连续行走的片段」切出来的**，同一个 `group_id` 下的多条序列其实来自同一段行走，验证集划分不当会造成「信息泄漏」，本地分数虚高；
3. **类别严重不均衡**，有的地面样本很多，有的只有二十几条。

### 1.2 比赛目标

训练特征表 `X_train.csv` 的列：

- `row_id`：行号；
- `series_id`：序列编号，**一个 series 就是一段 128 个时刻的读数**；
- `measurement_number`：这段序列里的第几个时刻（0～127）；
- `orientation_X/Y/Z/W`：四元数表示的三维姿态；
- `angular_velocity_X/Y/Z`：三轴角速度；
- `linear_acceleration_X/Y/Z`：三轴线性加速度。

标签表 `y_train.csv` 的列是 `series_id`、`group_id`、`surface`。测试集 `X_test.csv` 有同样的特征列，`sample_submission.csv` 要求为每个 `series_id` 给出一个 `surface`。

也就是说，输入是**一段 10 通道、128 步的信号**，输出是**一个地面类别**。

### 1.3 评价标准

比赛使用**分类准确率（Categorization Accuracy）**：预测对的地面类别数 ÷ 总序列数。9 分类，随机猜的准确率大约是 1/9 ≈ 0.11。准确率越高越好，满分 1.0。

---

## 二、解题过程

我们的主线是：**先把每条 128 步的序列压缩成一组统计特征和物理量特征，再用随机森林分类，并且用按 group_id 分组的验证集来挑选模型。** 这条路线在公开讨论里对应排行榜 0.71 左右的成绩，是性价比很高的基线。

### 2.1 数据探索（EDA）

我们首先读入三个表，确认数据的形状和一一对应关系：

```python
import pandas as pd
import numpy as np

train = pd.read_csv("../input/X_train.csv")
test  = pd.read_csv("../input/X_test.csv")
label = pd.read_csv("../input/y_train.csv")
sub   = pd.read_csv("../input/sample_submission.csv")

# 每条序列固定 128 个时刻，所以总行数 = 序列数 × 128
print(train.shape, "序列数:", train.series_id.nunique())
print(train.measurement_number.min(), train.measurement_number.max())
```

训练特征表有 487,680 行，序列数是 3,810，正好 487680 = 3810 × 128。这说明**每条序列长度固定为 128**，不用做变长对齐，省了很多事。

接着看标签分布。我们按 `series_id` 去重后再统计，避免同一条序列的 128 行被重复计数：

```python
# groupby 后取 first，保证每个 series_id 只算一次
counts = train_label = pd.read_csv("../input/y_train.csv")
print(counts.groupby("series_id").first()["surface"].value_counts())
```

结果如下：

| 地面 | 序列数 |
| --- | --- |
| concrete | 779 |
| soft_pvc | 732 |
| wood | 607 |
| tiled | 514 |
| fine_concrete | 363 |
| hard_tiles_large_space | 308 |
| soft_tiles | 297 |
| carpet | 189 |
| hard_tiles | 21 |

可以看到两个问题：一是**极不均衡**，最多的 `concrete` 有 779 条，最少的 `hard_tiles` 只有 21 条；二是 `hard_tiles` 只有一个 `group_id`。这意味着任何随机切分都很可能把 `hard_tiles` 全切到一边，验证集上根本学不到这一类。

我们还发现一个更隐蔽的问题：**同一个 `group_id` 下的多条序列来自同一段连续行走**。如果把同一 group 的序列同时分到训练和验证里，模型会「记住」这段路，验证分数虚高，但排行榜上会打回原形。因此后面必须**按 group_id 整组划分**。

### 2.2 数据清洗

原始数据本身没有缺失值（10 个传感器列都是满的），但有几件事必须做：

1. **降低内存**：487,680 行 × 多列浮点数，直接用 `float64` 会占几十 MB，我们把能窄化的列转成 `float32`/`int16`，材料里的做法把内存从 48.37 MB 降到 14.88 MB，减少约 69%；
2. **特征工程后产生的无穷大和缺失值要清零**：除以零、取对数等操作会产生 `inf`/`nan`，随机森林不能直接吃这些值；
3. **丢弃 `row_id`、`measurement_number` 这类纯编号列**，它们没有物理意义。

内存优化函数的核心逻辑是「按取值范围选择最窄的数据类型」：

```python
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            # 整数列：能放进 int16 就不放 int32
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                else:
                    df[col] = df[col].astype(np.int32)
            # 浮点列：优先 float32
            else:
                df[col] = df[col].astype(np.float32)
    return df

# 特征工程之后再统一清理异常值
train.replace([np.inf, -np.inf], 0, inplace=True)
test.replace([np.inf, -np.inf], 0, inplace=True)
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
```

这里用 0 而不是均值来填，是因为这些异常值大多是「除以零」造成的，本身没有物理意义，填 0 等于告诉模型「这里不可用」，比编造一个均值更诚实。

### 2.3 模型训练与超参数调优

这一步的核心是**特征工程**：把 128 × 10 的原始信号压缩成每条序列一行的特征。

我们首先造几个物理量。比如四元数的模长、归一化四元数、三轴角速度的合速度、三轴加速度的合加速度，以及把四元数转成欧拉角（直观的俯仰/横滚/偏航）：

```python
import math

def quaternion_to_euler(x, y, z, w):
    """把四元数转成欧拉角，方便刻画机器人姿态"""
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))       # 夹到 [-1,1]，防止 asin 越界
    Y = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)
    return X, Y, Z

def add_physics(data):
    # 四元数模长与归一化
    data["mod_quat"] = np.sqrt(data.orientation_X**2 + data.orientation_Y**2
                               + data.orientation_Z**2 + data.orientation_W**2)
    # 三轴角速度合成一个「总角速度」
    data["total_angular_velocity"] = np.sqrt(
        data.angular_velocity_X**2 + data.angular_velocity_Y**2 + data.angular_velocity_Z**2)
    # 三轴加速度合成一个「总加速度」
    data["total_linear_acceleration"] = np.sqrt(
        data.linear_acceleration_X**2 + data.linear_acceleration_Y**2
        + data.linear_acceleration_Z**2)
    # 加速度 / 角速度，刻画「转得慢但冲得猛」的程度
    data["acc_vs_vel"] = data.total_linear_acceleration / data.total_angular_velocity
    return data
```

合成量的意义在于：不同地面最直观的差别，是**颠簸强度**和**转向时的角速度**。把三轴合成一个标量，模型就更容易学到「这种地面整体更晃」这类规律。

然后我们对每个物理量，按 `series_id` 做**组内统计**：均值、最大值、最小值、标准差、绝对值的极值，以及一阶差分和二阶差分的平均。这一步把 128 个时刻压成一个数：

```python
def series_features(data):
    df = pd.DataFrame()
    for col in data.columns:
        if col in ["row_id", "series_id", "measurement_number"]:
            continue                      # 编号列不参与统计
        g = data.groupby("series_id")[col]
        df[col + "_mean"] = g.mean()      # 平均水平
        df[col + "_max"]  = g.max()
        df[col + "_min"]  = g.min()
        df[col + "_std"]  = g.std()       # 波动大小
        df[col + "_abs_max"] = g.apply(lambda x: np.max(np.abs(x)))
        # 一阶差分的绝对均值：信号「抖得多厉害」
        df[col + "_mean_abs_change"] = g.apply(
            lambda x: np.mean(np.abs(np.diff(x))))
        # 二阶差分：抖动的「加速度」
        df[col + "_mean_change_of_abs_change"] = g.apply(
            lambda x: np.mean(np.diff(np.abs(np.diff(x)))))
    return df
```

一路统计下来，训练表从每条序列 128 行变成了每条序列一行、共 260 个特征（材料中的 `train.shape` 是 `(3810, 260)`）。

**关于验证策略。** 这是这道题最关键的一步。因为排行榜的类别分布和训练集不一样，直接用普通分层抽样会高估分数。材料里的做法参考了社区用「排行榜探测」得到的各类别权重，手工挑出若干 `group_id` 组成验证集，使验证集的类别比例尽量贴近排行榜：

```python
def create_valid_set(label):
    # 每个类别在验证集里应占的比例（来自排行榜探测讨论）
    ldict = {
        "concrete": 0.16, "soft_pvc": 0.18, "wood": 0.06, "tiled": 0.03,
        "fine_concrete": 0.10, "hard_tiles_large_space": 0.12,
        "soft_tiles": 0.23, "carpet": 0.05, "hard_tiles": 0.07,
    }
    # 按 group_id 整组挑，绝不把同一段行走拆到两边
    cv_set = {
        "concrete": [0], "soft_pvc": [69], "wood": [2], "tiled": [28],
        "fine_concrete": [36], "hard_tiles_large_space": [16],
        "soft_tiles": [4, 17], "carpet": [52], "hard_tiles": [27],
    }
    val_df = pd.DataFrame()
    for key, ids in cv_set.items():
        for i in ids:
            val_df = pd.concat([val_df, label[label.group_id == i]])
    # hard_tiles 只有一个 group，只能同时放进训练和验证，否则验证集里一条都没有
    return val_df
```

这段代码里最值得学习的是最后一行注释：`hard_tiles` 在整个数据集里只有一个 `group_id`，如果严格按组隔离，验证集就永远见不到这个类别。所以只能**破例**让它同时出现在两边——这是一个「为了能评估而牺牲一点严格性」的务实取舍。

训练用的是随机森林，500 棵树。随机森林的好处是对特征尺度不敏感、能处理非线性、对不均衡也相对稳健：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 500 棵树，固定随机种子保证可复现
model = RandomForestClassifier(n_estimators=500, random_state=13)
model.fit(x_train, y_train)

# 看训练和验证的准确率，判断有没有过拟合
print("训练准确率:", accuracy_score(y_train, model.predict(x_train)))
print("验证准确率:", accuracy_score(y_val, model.predict(x_val)))
```

我们没有把树的数量往 1000 以上堆，因为 500 棵已经能把误差稳定下来，再翻倍收益很小、训练时间却线性增加。真正的提升来自好的特征和合理的验证集，而不是更多的树。

### 2.4 模型测试与指标报告

在按 group 构造的验证集上，我们得到的**验证准确率约 0.64**；把模型提交到排行榜后，**公开榜分数约 0.71**。

这两个数字的差距本身就很有信息量：验证集是「人工仿造排行榜分布」的，仍然和真实测试集有出入。这也解释了社区里大家共同的困惑——本地交叉验证和排行榜对不上。我们在报告指标时坚持两件事：

1. **同时看训练准确率和验证准确率**，两者差距大就说明过拟合；
2. **以排行榜为准做最终判断**，因为验证集再精心构造，也只是近似。

最后生成提交文件：

```python
# 用同样的特征工程处理测试集
test_feat = series_features(add_physics(test.copy()))
# 用训练好的森林预测每个 series 的地面
pred = model.predict(test_feat)

submission = pd.DataFrame({
    "series_id": test_feat.index,   # groupby 后的索引就是 series_id
    "surface": pred,
})
submission.to_csv("submission.csv", index=False)
assert len(submission) == test.series_id.nunique()   # 行数必须等于序列数
```

这里要特别小心索引的对齐：`series_features` 返回的表以 `series_id` 为索引，如果打乱了顺序，预测结果就会和 `series_id` 错位，分数会莫名其妙地低。

---

## 三、总结

本文中我们完成了 CareerCon 2019 机器人地面分类的完整流程。我们首先用 EDA 确认了 3,810 条序列、每条 128 步、10 个 IMU 通道，并发现 9 类地面严重不均衡、`hard_tiles` 只有 21 条。其次我们做了内存压缩和无穷值清理，把特征表从 48 MB 压到 15 MB 左右。接着我们从物理量出发构造了四元数模长、合角速度、合加速度、欧拉角等中间量，再按序列做均值/极值/标准差/差分统计，把每条 128 步的信号压成 260 维特征，并用随机森林 500 棵树分类。最后我们用按 `group_id` 整组划分、并仿照排行榜类别分布的验证集评估，得到验证准确率约 0.64、排行榜约 0.71。

这个比赛给我们最大的教训是：**时间序列题里，验证集的划分方式比模型本身更能决定成败**。序列是从连续行走里切出来的，`group_id` 就是防泄漏的钥匙；一旦按行随机切分，模型会提前看到同一段路，本地分数好看但榜上难看。材料里还有一条不同的思路：有队伍用 `group_id` 把碎片序列重新拼接成长序列、用 tsfresh 提取大量时序特征再做元集成，最终进入第 11 名——这提醒我们，**理解数据是怎么被切出来的，往往比换模型更能提分**。
