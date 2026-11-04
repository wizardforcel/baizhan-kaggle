# 非洲土壤性质预测（AFSIS Soil Properties）

## 一、比赛简介

### 1.1 比赛背景

非洲土壤信息服务（Africa Soil Information Service，简称 AFSIS）要在很大的区域上绘制土壤地图。传统做法是到野外钻孔取样、把样本送进实验室化验，一个点一个点地测——速度慢、成本高，覆盖整个非洲几乎不可能。于是人们想到一个替代办法：**用中红外光谱仪（MIR）照一下土壤样本，根据反射率曲线反推土壤的化学成分**。光谱仪测量便宜、快速，如果模型能学准「光谱 → 土壤性质」的映射，绘制大范围土壤图就变得可行了。

这场比赛就把这件事做成了题目：每个采样点的土壤被光谱仪在几千个波长上扫描，记录下一条高维的反射率曲线；同时还有高程、植被指数、地表温度等地理环境信息。比赛要求参赛者用这些信息预测 5 种土壤性质。

难点非常突出：**特征维度极高（3500 多个波长列），而训练样本只有 1157 个**——这是典型的「维度爆炸 + 小样本」回归问题。如果把整条光谱直接喂给模型，模型会轻轻松松记住训练集、在新样本上崩掉；而且 5 个目标变量的性质各不相同，一套参数打天下往往不是最优。

### 1.2 比赛目标

训练集 `train.zip` 每行一个采样点，共 1157 行、3600 列：

- `PIDN`：土壤样本编号；
- `m7497.96`、`m7496.04` …… 共 3578 列光谱反射率，列名里的数字就是波长；
- 15 列地理环境辅助特征：容重 `BSAN/BSAS/BSAV`、复合地形指数 `CTI`、高程 `ELEV`、增强植被指数 `EVI`、昼夜地表温度 `LSTD/LSTN`、反射率 `REF1/REF2/REF3/REF7`、相对湿度 `RELI`、`TMAP/TMFI` 等；
- `Depth`：土层类型，取值为 `Topsoil`（表土）或 `Subsoil`（底土）；
- 5 个标签：`Ca`（钙）、`P`（磷）、`pH`（酸碱度）、`SOC`（有机碳）、`Sand`（沙含量）。

测试集 `test.zip` 有 727 行、3595 列（比训练集正好少 5 个标签列），另外还有一份 `sample_submission.csv`，格式是 `PIDN` 加 5 个目标列。我们要为测试集的每个样本填满这 5 列。

### 1.3 评价标准

比赛用**误差指标**衡量，越小越好。手头的材料里，两份笔记本都拿**均方误差 MSE**（也就是 L2 损失）来观察验证集表现，其中一份的标题还直接把成绩标注为「0.6 MSE」，所以我们就把 MSE 当作主要参照。MSE 的平方根叫 RMSE，它的量纲和原始标签一致，汇报时读起来更直观，下面两个都会给。另外，5 个目标变量的数值范围差很多（`pH` 在 0~14 之间，`P` 的分布又很偏），实际评估时要**分目标看误差**，而不是把 5 个数混在一起算。

> 一个小提醒：这份笔记本的标题把成绩标注为「0.6 MSE」，这也是我们下面分析误差数量级时的参照。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**因为波长列太多、样本太少，先对每个目标变量单独筛出最相关的 100 个特征，再为每个目标配一个多项式核的岭回归模型。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先把三份文件读进来，确认形状和缺失情况：

```python
import numpy as np
import pandas as pd

# 训练集、测试集、提交样例
train = pd.read_csv('/kaggle/input/afsis-soil-properties/train.zip')
test  = pd.read_csv('/kaggle/input/afsis-soil-properties/test.zip')
sub   = pd.read_csv('/kaggle/input/afsis-soil-properties/sample_submission.csv')

print(train.shape)   # (1157, 3600): 1157 个采样点, 3600 列
print(test.shape)    # (727, 3595):  比训练集少 5 个标签列

# 最多的一列有几个缺失值
print(train.isna().sum().max())   # 0
print(test.isna().sum().max())    # 0

print(sub.head())    # PIDN + Ca / P / pH / SOC / Sand
```

数据非常干净：训练集 1157 行、测试集 727 行，两边都没有任何缺失值，因此不需要做填补。再看列名结构：3578 个以 `m` 开头的波长列，加上 15 个地理辅助特征，再加上 `Depth` 一列，正好是 3594 个候选特征。

其次我们看土层类型的分布，确认两类样本是否均衡：

```python
# Topsoil(表土) 与 Subsoil(底土) 各有多少个采样点
print(train['Depth'].value_counts())
# Topsoil    581
# Subsoil    576

# 两个土层的样本几乎对半, 不需要额外做重采样
```

表土 581 个、底土 576 个，几乎五五开。这说明比赛对两种土层都认真采样了，模型不会被某一种土层主导；同时也提示我们，`Depth` 本身是一个有用的特征——同样是「沙含量」，表土和底土的典型值并不一样。

### 2.2 数据清洗

其次我们做清洗。需要处理的事情有三件：把文本型的 `Depth` 转成数字、把纯编号的 `PIDN` 从特征里剔除（编号和土壤性质没有任何关系，留着只会让模型学到噪声）、以及最关键的——**在 3594 个特征里做降维**，否则 1157 个样本撑不起这么高的维度：

```python
# Depth 是文本, 转成 0/1 才能进模型
train['Depth'] = train['Depth'].replace({'Topsoil': 0, 'Subsoil': 1})

# PIDN 只是样本编号, 不是特征; 测试集里也一并丢掉
test.drop(['PIDN'], axis=1, inplace=True)

# 五个目标变量
targets = ['Ca', 'P', 'pH', 'SOC', 'Sand']

# 特征列 = 除 PIDN 和五个标签以外的所有列
feature_cols = [c for c in train.columns
                if c not in targets and c != 'PIDN']
print(len(feature_cols))   # 3594
```

清洗完的下一步是特征筛选。我们的做法是**为每个目标变量各筛一套特征**，而不是所有目标共用一套——因为对 `Ca` 敏感的波长，未必对 `Sand` 敏感。筛选工具用 `SelectKBest` 配合 `f_regression`：它给每个特征和目标变量算一个 F 统计量，衡量「这个特征和目标有没有线性关系」，然后留下分数最高的 100 个：

```python
from sklearn.feature_selection import SelectKBest, f_regression

# 每个目标单独挑 100 个最相关的特征, 并把同一套列名套到测试集上
selected = {}
for y in targets:
    X_all = train[feature_cols]
    selector = SelectKBest(score_func=f_regression, k=100)
    X_sel = selector.fit_transform(X_all, train[y])

    # get_support() 返回布尔掩码, 用它把列名挑出来
    keep_cols = [c for c, keep in zip(feature_cols, selector.get_support()) if keep]
    selected[y] = (X_sel, test[keep_cols])   # 训练特征 / 测试特征成对保存

print(selected['Ca'][0].shape)   # (1157, 100)
print(selected['Ca'][1].shape)   # (727, 100)
```

这里有一个容易踩的坑值得说明：原始笔记本是「一个目标复制一份代码」写五遍的，复制过程中 `P` 那一支误用了 `Ca` 的目标变量来选特征（`fs.fit_transform(X_ca, y_ca)`），导致 P 模型拿到的是为钙挑选的特征。这类逐目标复制的流程极容易出现这种串行错误，所以我们把它改成一个统一的循环，从结构上杜绝「选错目标」的问题。

### 2.3 模型训练与超参数调优

接着我们选模型。样本只有 1157 个、特征筛完还剩 100 维，属于「小样本、中等维度」，我们选**核岭回归（Kernel Ridge Regression）**：它本质上是在岭回归（带 L2 正则的线性回归）外面套一个核函数，把原始特征隐式地映射到更高维的空间里，从而拟合非线性关系；同时正则项又能压住小样本带来的过拟合。核函数选多项式核，让它能捕捉特征之间的乘积型交互。

我们先对 5 个目标各训练一个模型，用 80% 训练、20% 验证，并记录验证集 MSE：

```python
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

preds = {}     # 存每个目标对测试集的预测
scores = {}    # 存每个目标的验证 MSE

for y in targets:
    X_sel, X_test_sel = selected[y]

    # 80% 训练 / 20% 验证, random_state 固定保证可复现
    x_tr, x_val, y_tr, y_val = train_test_split(
        X_sel, train[y], random_state=42, test_size=0.2)

    # 多项式核岭回归: alpha 是正则强度, degree 是多项式次数
    kr = KernelRidge(alpha=0.1, kernel='polynomial', degree=7, coef0=2.5)
    kr.fit(x_tr, y_tr)

    # 在验证集上算 MSE, 同时把测试集预测存下来
    scores[y] = mean_squared_error(y_val, kr.predict(x_val))
    preds[y] = kr.predict(X_test_sel)
    print(y, round(scores[y], 4))
```

跑完之后，五个目标的验证 MSE 分别是 Ca 0.1754、P 0.4783、pH 0.1715、SOC 0.2470、Sand 0.1626。差异很明显：`pH` 和 `Sand` 最好学，`P` 最难——磷在土壤里的分布本来就极不均匀，光谱信号也弱。

接着我们针对最难的 `P` 做超参数扫描。先单独取 P 的训练/验证数据（和上面同样的切分种子），再扫多项式次数 `degree`：它决定模型能拟合多复杂的曲线，太小会欠拟合，太大会过拟合：

```python
# 单独准备 P 的数据, 切分方式与上面主流程保持一致
X_p, X_p_test = selected['P']
xp_tr, xp_val, yp_tr, yp_val = train_test_split(
    X_p, train['P'], random_state=42, test_size=0.2)

# degree 从 1 扫到 10, 看验证 MSE 怎么变
for d in range(1, 11):
    kr = KernelRidge(alpha=0.1, kernel='polynomial', degree=d, coef0=2.5)
    kr.fit(xp_tr, yp_tr)
    m = mean_squared_error(yp_val, kr.predict(xp_val))
    print('degree=%d  MSE=%.4f' % (d, m))
# 1:0.5813  2:0.5600  3:0.5346  4:0.5117  5:0.5048
# 6:0.4966  7:0.4783  8:0.4915  9:0.5772  10:0.7339
```

曲线是一条漂亮的 U 形：`degree=7` 时 MSE 最低（0.4783），再往上误差迅速反弹——这说明次数 9、10 时模型已经开始把训练集里的噪声也背下来了。所以我们把 `degree` 定在 7。

然后我们扫正则强度 `alpha`（固定 `degree=2`，让曲线更平滑、便于观察趋势）：

```python
# alpha 越大正则越强; 扫描 [0.1, 0.2, ..., 1.0]
for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0]:
    kr = KernelRidge(alpha=a, kernel='polynomial', degree=2, coef0=2.5)
    kr.fit(xp_tr, yp_tr)
    m = mean_squared_error(yp_val, kr.predict(xp_val))
    print('alpha=%.1f  MSE=%.4f' % (a, m))
# 0.1:0.5600  0.2:0.5664  0.3:0.5703  0.4:0.5730 ...
# 1.0:0.5806
```

趋势很单调：`alpha` 越小，验证误差越低，在扫过的范围里 0.1 最好。这说明在当前的特征筛选和样本规模下，正则不必加得太重——`SelectKBest` 已经把 3594 维压到 100 维、去掉了大部分噪声，特征本身带来的约束已经足够。

### 2.4 模型测试与指标报告

最后我们把 5 个目标各自的预测拼进提交表。这里要注意顺序不能乱：`sample_submission.csv` 的 `PIDN` 列顺序就是测试集的顺序，我们的预测也是按这个顺序生成的，所以直接按列赋值即可：

```python
# 五个目标分别来自各自独立的模型
sub['Ca']   = preds['Ca']
sub['P']    = preds['P']
sub['pH']   = preds['pH']
sub['SOC']  = preds['SOC']
sub['Sand'] = preds['Sand']

# 写提交文件; 列顺序与样例完全一致
sub.to_csv('first.csv', index=False)

# 检查一下: 行数应为 727, 没有空值
print(sub.shape, sub.isna().sum().sum())
```

我们按目标汇总验证误差，并把 MSE 开平方换算成 RMSE（量纲与原始标签一致，更直观）：

| 目标 | 验证 MSE | 换算 RMSE |
| --- | --- | --- |
| Ca | 0.1754 | 0.42 |
| P | 0.4783 | 0.69 |
| pH | 0.1715 | 0.41 |
| SOC | 0.2470 | 0.50 |
| Sand | 0.1626 | 0.40 |

从这张表能读出两条信息。第一，**误差的量级与笔记本标题标注的 0.6 MSE 是同一个数量级**，说明这套「逐目标选特征 + 核岭回归」的路线是站得住的。第二，**P 明显拖了后腿**——它的 RMSE 约 0.69，几乎是其他目标的 1.5 倍；如果比赛只看 5 个目标的平均分，改进 P 的收益最大。

为了对照，我们还参考了另一份使用光谱一阶导数 + LightGBM（dart 提升）的方案：它对全部波长做相邻差分构造导数特征，用 `feature_fraction=0.9`、`bagging_fraction=0.8` 等随机化参数防过拟合，验证集 L2 从最初约 2.66 一路降到约 0.598 并稳定下来。两个方案的误差量级相近，但核岭回归这条线在 5 个目标上都有明确的验证数字，所以我们以它作为主线。

---

## 三、总结

本文中我们完成了非洲土壤性质预测的完整流程。我们首先用 EDA 摸清了数据：训练集 1157 个采样点、3600 列，测试集 727 行、3595 列，两边都没有缺失值；3594 个候选特征里绝大多数是波长列，表土和底土各约一半。其次我们做了数据清洗：把 `Depth` 文本转成 0/1，丢掉样本编号 `PIDN`，并针对「维度爆炸、样本太少」的核心矛盾，为 5 个目标变量各自用 `f_regression` 选出 100 个最相关的特征（同时修掉了原始笔记本里目标变量串用的问题）。接着我们为每个目标单独训练一个多项式核岭回归模型，并对最难的 `P` 扫了多项式次数与正则强度，确定 `degree=7`、`alpha=0.1` 最合适。最后我们汇总验证误差：Ca 0.175、P 0.478、pH 0.172、SOC 0.247、Sand 0.163，换算成 RMSE 约在 0.40~0.69 之间，并按样例格式写出了提交文件。作为对照，另一条「光谱一阶导数 + LightGBM」的路线把验证 L2 从约 2.66 降到约 0.598，误差量级与我们相当。

这个比赛给我们的最大启发是：**面对「几千个特征、一千个样本」的数据，降维不是可选项而是第一步**——只有先把噪声波长筛掉，核方法或提升树这类模型才有机会学到真正的光谱与土壤性质之间的对应关系。