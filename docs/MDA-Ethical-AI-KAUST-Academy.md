# MDA Ethical AI KAUST Academy

## 一、比赛简介

### 1.1 比赛背景

这个比赛的数据来自一段一段的**录像**（recording）。录像里有很多个"拍摄单元"（unit）——可以理解成画面被切成若干块，或者广场上被划出的若干区域——每一分钟统计一次该单元里出现了多少人。把这些数字排成一张表，就是「行是分钟、列是单元、格子里是人数」的二维表格。

难点在于，主办方把表格里的**一部分格子挖掉了**，只留下编号，让人去填。这种被挖掉的格子，起始 notebook 里直接叫它"洞"（hole）。洞不止一种：有的是要我们**预测未来**的（`kind` 为 `forecast`，给你"当前时刻"和"往后看多少分钟"，让你算出那时候有多少人），有的是别的类型，主办方在提交样例里给了参考值。

从比赛名看，它和 KAUST Academy 的 Ethical AI 课程有关；数据同时提供了"摄像头表"和"H3 广场表"两套，起始 notebook 的说明是"两张表都能跑"，因为我们写的是与具体场景无关的通用流程。**需要说明的是，材料里只给了一份起始 notebook，没有比赛页面的完整说明，所以下面凡是涉及到指标定义这类材料没写清的地方，我们都按代码里的证据做最小推断，并明确标出来。**

对人流做预测本身也是个敏感话题：同样的技术可以用来优化广场照明和疏散通道，也可以用来盯着具体的人。这也是这类比赛被放在"Ethical AI"名义下的原因——**只统计人数、不识别身份**，是这条技术路线能被接受的前提。

### 1.2 比赛目标

数据目录里有三个文件：

- `train.csv`：训练集，一行是「某段录像里、某个单元、某一分钟」的人数，列为 `recording`（录像编号）、`unit`（单元编号）、`minute_of_day`（当天的第几分钟）、`people`（人数）。
- `test.csv`：测试集，结构和训练集一样，但 `people` 大多是空的；额外多了 `id`、`kind`、`now_minute_of_day`、`horizon` 几列。`id` 不为空的行，才是需要我们填的洞。
- `sample_submission.csv`：提交样例，两列 `id` 和 `people`，里面已经填好了参考值。

我们的任务是：对 `kind == "forecast"` 的每一个洞，根据"当前时刻（`now_minute_of_day`）之前"的信息，预测 `horizon` 分钟之后该单元的人数；其他类型的洞保留参考值不动。最后输出一个只含 `id`、`people` 两列的 CSV。

### 1.3 评价标准

材料没有直接写出指标公式。起始 notebook 里有一句关键的话：提交样例里的参考基线"得分为 1"（the reference baselines, which score 1），参赛者做完预测后要争取**比 1 更好**。由此我们可以推断，这是一个**相对参考基线的误差型分数**：1 表示和基线打平，越小越好。至于它具体是哪种误差、怎么归一化，材料里没有，我们不臆测。

不过这不影响我们做事：不管外面套的是什么公式，核心都是同一件事——**让预测的人数尽量接近真实人数**。所以下面我们自己评估时，看的就是"预测值和真实值差多少"。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**把每个人数表看成一幅随时间滚动的"热力图"，用当前时刻之前几分钟的统计量，去预测 H 分钟之后的变化量。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步是找到数据并读进来。Kaggle 的输入目录结构每次挂载都可能不一样，所以起始 notebook 用了一种笨但稳的办法：在 `/kaggle/input` 下递归查找同时含有三个文件的目录。

```python
import os
import numpy as np
import pandas as pd

INPUT = "/kaggle/input"
# 数据可能挂在 input 下的任意子目录，逐个目录找同时含三个文件的文件夹
DATA = next(r for r, _, f in sorted(os.walk(INPUT, followlinks=True))
            if {"train.csv", "test.csv", "sample_submission.csv"} <= set(f))
print("data:", DATA)

train = pd.read_csv(os.path.join(DATA, "train.csv"))
test = pd.read_csv(os.path.join(DATA, "test.csv"))
sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))

# 测试集里 id 不为空的行，才是要我们填的洞；其余行是已公布的参考值
holes = test[test["id"].notna()].copy()
print(f"{len(train):,} 条训练行, {len(test) - len(holes):,} 条已公布测试行, {len(holes):,} 个洞")
print(holes["kind"].value_counts().to_string())
```

这段打印很关键：它一下子告诉我们三个数字——训练行数、测试集里已经公开的行数、以及真正要填的洞有多少。接着 `kind` 的分布告诉我们洞里有多少是 `forecast`。**我们只处理 `forecast` 那一类**，剩下的原样保留参考值，这是最省事也最不容易出错的做法。

第二步是把长表变成宽表。所谓长表，就是"一行一个格子"；宽表则是"行是分钟、列是单元"，看起来就像一张随时间滚动的热力图。这一步是整个方案的基础，因为后面所有"往前看 15 分钟"的操作，都依赖时间轴是连续的。

```python
def tables(df):
    """把长表转成 {录像编号: 分钟 × 单元 的人数矩阵}"""
    df = df[df["people"].notna()]          # 没公布人数的行先丢掉，它们在透视表里会变成 NaN
    out = {}
    for rec, g in df.groupby("recording"):
        # 行是分钟、列是单元，值是人数
        t = g.pivot_table(index="minute_of_day", columns="unit", values="people")
        # 录像中间可能整分钟缺失，补成连续网格，这样"往前数 6 行"就等于"往前 5 分钟"
        out[rec] = t.reindex(np.arange(t.index.min(), t.index.max() + 1))
    return out

T_train, T_test = tables(train), tables(test)
```

这里有个容易踩的坑：如果直接 `pivot_table`，中间缺掉的分钟会被跳过，那么"取最后 6 行"就不等于"取最近 5 分钟"。所以我们用 `reindex` 把分钟轴补成连续的等差数列，缺的位置自动是 `NaN`。

第三步，我们想看看每段录像的规模，以及数据有多稀疏：

```python
for rec, t in T_train.items():
    print(rec, t.shape, round(t.notna().mean().mean(), 3))
```

`t.shape` 是（分钟数, 单元数），`t.notna().mean()` 是这张表里"有数字的格子"占多大比例。这个比例是理解整场比赛的钥匙：**如果一张表有一半格子是空的，那"往前找最近一个有效值"就不能只回看 1 分钟，得允许回看几分钟。** 后面特征里的"最多回看 5 分钟"就是从这里来的。

### 2.2 数据清洗

这份数据的清洗非常轻，因为我们不想动原始信息。具体做了三件事：

**第一，只保留有真实人数的行。** `people` 为空的格子，要么是洞，要么是本来就没记录，两种情况都不该当成"0 人"——把"没有观测"当成"零"是这类任务里最常见的错误，会让模型以为某个地方一直没人。

**第二，把时间轴补全。** 上面 `reindex` 做过了。补出来的 `NaN` 不填充，一直留到特征计算那一步再处理。

**第三，明确写下"只用当前时刻之前的信息"。** 这一点严格来说不是清洗，而是纪律。比赛里如果偷看未来（比如用 `now` 之后的真实人数算特征），本地分数会好看得离谱，上线就崩。起始 notebook 在取数据时写的是 `t.loc[now - 15:now]`，**右端点就是 `now`，一秒钟的未来都不碰**。

还有一个细节：人数分布是严重长尾的——大多数格子是 0 到几个人，偶尔有几百人的大场面。如果直接对原始人数做回归，模型会被大数值牵着走。所以我们在特征和标签里都取了对数：`np.log1p(x)`，也就是 `log(1 + x)`，这个变换能把 0 也安全地纳入，并且把"从 1 人涨到 10 人"和"从 100 人涨到 110 人"看成量级相近的变化。

### 2.3 模型训练与超参数调优

**特征。** 每个单元我们只给四个数：最近一个有效人数（最多回看 5 分钟）、最近 15 分钟的平均人数、整张表最近 5 分钟的活跃程度，以及预测跨度 `horizon`。全部取对数，因为这些量跨好几个数量级。

```python
def features(t, now, H):
    # 只取 now 之前 15 分钟的窗口，右端点就是 now，不使用任何未来数据
    a = t.loc[now - 15:now].to_numpy(float)

    last = np.full(a.shape[1], np.nan)
    # 从窗口末尾往前逐行扫，每列取最近一个有效值，所以最多回看 6 行 = 5 分钟
    for row in a[-6:]:
        last = np.where(np.isfinite(row), row, last)

    with np.errstate(all="ignore"):           # 空窗口求均值会产生 NaN，这是预期行为
        mean15 = np.nanmean(a, 0)             # 每个单元最近 15 分钟的平均人数
        table = np.nanmean(np.log1p(a[-6:]))  # 整张表最近 5 分钟的平均活跃度（全表共用一个数）

    return pd.DataFrame({
        "log_last": np.log1p(last),      # 当前水平
        "log_mean15": np.log1p(mean15),  # 近期趋势
        "log_table": table,              # 全局背景：整段录像整体忙不忙
        "horizon": float(H),             # 要预测多远之后
    }, index=t.columns)


X, y = [], []
for rec, t in T_train.items():
    # 从第 20 分钟起，每 3 分钟取一个起点；末尾留 5 分钟，保证最早的一个跨度也够
    for now in t.index[20:-5:3]:
        for H in (5, 15, 30):
            if now + H not in t.index:
                continue
            f = features(t, now, H)
            # 学"变化量"而不是绝对值：log(未来) - log(当前)
            target = np.log1p(t.loc[now + H].to_numpy(float)) - f["log_last"].to_numpy()
            ok = np.isfinite(target)     # 未来格子是洞或空，就没有监督信号，跳过
            X.append(f[ok]); y.append(target[ok])

X, y = pd.concat(X), np.concatenate(y)
```

这里最值得说的是**为什么预测变化量**。人数有明显的"惯性"：这一分钟 80 人，下一分钟大概率还是七八十人。如果直接让模型预测"15 分钟后有多少人"，模型会把大量精力花在学这个惯性上；改成预测 `log(未来) - log(现在)`，惯性就被人工扣掉了，模型只需要学"会往上飘还是往下飘、飘多少"。这个技巧在时间序列里叫**残差建模**，是通用做法。

另外，`horizon` 被当成一个普通特征塞进模型，这样同一个模型就能同时处理 5、15、30 分钟三种跨度，不用训三个模型。

**模型。** 我们用 `HistGradientBoostingRegressor`，这是一种基于直方图的梯度提升树，速度快、对缺失值友好、几乎不用调参，很适合做基线。

```python
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(
    loss="absolute_error",   # 人数长尾，绝对误差比平方误差更不容易被极端值带偏
    max_iter=200,
    learning_rate=0.05,
    random_state=0,
)
# 人数多的单元给更大权重：判断"100 还是 120"比判断"0 还是 2"更重要
w = np.maximum(np.expm1(X["log_last"]), 1.0)
model.fit(X, y, sample_weight=w)
```

三个超参数的选择理由都不复杂：`loss="absolute_error"` 对应的是"平均绝对误差"，它惩罚的是绝对偏差而不是平方偏差，遇到大场面不会被一个异常值拖垮；`max_iter=200` 配合 `learning_rate=0.05` 是一组很常见的保守组合，树的数量够多、每棵树走得够慢，不容易过拟合；`sample_weight` 取 `expm1(log_last)` 也就是还原出来的原始人数，等于告诉模型"人多的格子错了更严重"。

**关于调参。** 我们没有做网格搜索，原因有两个。一是这份数据的每个样本不是独立的——同一段录像里相邻的 `now` 高度相关，随机划分验证集会严重高估效果；二是特征只有四个，模型容量很小，`max_iter` 在 100 到 300 之间变化时结果差别不大。相比之下，**把特征做得更贴近物理含义**（比如加上"最近 5 分钟与前 15 分钟的差值"来刻画趋势）比调参更有价值。这也是我们想强调的一点：小特征集的任务里，调参的边际收益通常低于特征工程。

### 2.4 模型测试与指标报告

预测阶段的组织和训练阶段是对称的：按「录像、当前时刻、跨度」分组，每组算出特征，预测变化量，再加回当前值。

```python
# 先把样例里的参考值全部抄下来，非 forecast 的洞就靠它们了
pred = dict(zip(sample["id"], sample["people"]))

fc = holes[holes["kind"] == "forecast"]
for (rec, now, H), g in fc.groupby(["recording", "now_minute_of_day", "horizon"]):
    t = T_test[rec]
    f = features(t, int(now), int(H))
    f = f.reindex(g["unit"])          # 按这一组里出现的单元对齐顺序
    # 模型输出的是对数空间的变化量，先加回当前值，再用 expm1 还原成人数
    p = np.expm1(f["log_last"].to_numpy() + model.predict(f))
    for i, v in zip(g["id"], p):
        if np.isfinite(v):
            pred[i] = max(0.0, float(v))   # 人数不能为负，兜底截断

submission = pd.DataFrame({"id": holes["id"], "people": [pred[i] for i in holes["id"]]})
submission.to_csv(os.path.join(OUT, "submission.csv"), index=False)
print(submission.shape)
```

三个细节值得展开：

1. **`reindex(g["unit"])`**。`features` 返回的行索引是"整段录像的全部单元"，而这一组 `g` 里可能只有其中一部分单元有洞。不做对齐，预测值就会张冠李戴。这是宽表方案最常见的 bug。
2. **`np.expm1` 加回当前值**。训练时标签是 `log1p(未来) - log1p(现在)`，所以预测时 `log1p(未来) = 预测值 + log1p(现在)`，再 `expm1` 得到人数。少写一步，输出就永远是零点几。
3. **`max(0.0, v)`**。模型在极端情况下可能吐出负数（尤其是当前值是 0、模型又预测了一个负变化量），人数不可能是负数，截断一下最稳。

至于分数，材料里没有给出这份起始 notebook 的本地验证分数，我们也没有额外造一个。按照样例"参考基线得分为 1"的说法，这个 notebook 的定位就是**一个能跑通、能提交、比随便填要好一点的起点**。它的价值不在于分数，而在于把"数据长什么样、什么能看什么不能看、怎么对齐"这几件事讲清楚了。

如果要在它基础上继续做，我们会按这个顺序加东西：一是用**按录像分组**的验证方式（同一段录像不能同时出现在训练和验证里），这样估出来的分数才可信；二是给每个单元单独学一条"日周期曲线"，因为人流有很强的时间规律；三是把"当前值"从简单回看改成指数平滑，让特征对噪声更稳。

---

## 三、总结

本文中我们完成了 MDA Ethical AI KAUST Academy 人数补洞任务的一条完整基线。我们首先把杂乱的输入目录定位出来，读入训练、测试和提交样例三张表，发现真正要填的只有 `kind == "forecast"` 那一部分洞。其次我们把"一行一个格子"的长表转成"分钟 × 单元"的宽表，并补全时间轴，保证"往前看 5 分钟"这句话在代码里真的成立。接着我们构造了四个特征——最近有效人数、近 15 分钟均值、全表活跃度、预测跨度，全部取对数，并让模型去学对数空间里的**变化量**而不是绝对值。然后我们用绝对误差损失的直方图梯度提升树训练，并按原始人数给样本加权。最后我们按「录像、当前时刻、跨度」分组做预测，把变化量加回当前值、还原成人数、截断负值，再与样例里的参考值合并成提交文件。

这次练习让我们记住三件事：**宽表操作一定要对齐索引**，否则预测会错位；**特征只能用当前时刻之前的数据**，这条纪律比任何模型都重要；以及**当样本之间高度相关时，随机划分的验证分会骗人**，必须按录像分组来评估。
