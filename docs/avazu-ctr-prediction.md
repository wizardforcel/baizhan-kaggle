# Avazu 移动广告点击率预测（Click-Through Rate Prediction）

## 一、比赛简介

### 1.1 比赛背景

你在手机 App 里刷信息流时，屏幕上方或中间会插进一条广告。广告主为这次展示付了钱，但他真正关心的是——**你会不会点它**。如果系统能提前估出「这个用户在这个位置看到这条广告，有多大概率会点」，就可以把出价高的、点击率高的广告排在前面，让同样的流量赚更多钱。这个概率就叫 **点击率**（Click-Through Rate，简称 CTR）。

Avazu 是一家移动广告平台，它把自己平台上 10 天的真实曝光日志公开出来，办了这场比赛。每一行就是**一次广告展示**（impression），我们要预测这次展示会不会被点击。

这个比赛的难点几乎全在「数据规模」和「数据形态」上。

第一，**数据极大**。训练集有 **40,428,967 条**记录，24 列。用 pandas 直接读进来，内存会直接爆掉，所以连读文件都得用分块（chunk）的方式。

第二，**几乎全是类别特征，而且基数极高**。`device_ip` 在训练集里有 2,893,925 个不同取值，`device_id` 有 931,588 个，`site_id` 有 3870 个。这些列不能当数字用，也不能简单 one-hot。

第三，**正负样本极不均衡**。整份数据里点击率大约 **17%**，也就是五次展示才有一次点击。模型如果什么都不学、一律预测「不点」，准确率也有 83%，但这个模型毫无价值。

第四，**训练集和测试集的时间不重叠**。`hour` 这一列在训练集里有 240 个不同取值（10 天），而测试集只有 24 个（1 天）。这说明测试集是「未来的 1 天」，模型必须在没见过的时段上仍然有效——这正是真实广告系统的要求。

### 1.2 比赛目标

给定一次广告展示的全部属性，预测这次展示被点击的概率。核心的列如下：

| 列名 | 含义 |
| --- | --- |
| `id` | 每条展示的编号 |
| `click` | 标签，1 表示被点击，0 表示没有 |
| `hour` | 展示时间，格式是 `YYMMDDHH` |
| `C1` | 匿名类别特征 1 |
| `banner_pos` | 广告位序号（广告出现在页面第几个位置） |
| `site_id` / `site_domain` / `site_category` | 广告所在站点、域名、类别 |
| `app_id` / `app_domain` / `app_category` | 广告所在 App、域名、类别 |
| `device_id` / `device_ip` / `device_model` | 设备标识、IP、机型 |
| `device_type` / `device_conn_type` | 设备类型、联网方式 |
| `C14` ~ `C21` | 8 个匿名类别特征 |

训练集 40,428,967 行 × 24 列；测试集 4,577,464 行 × 23 列（少一个 `click`）。我们要给每条测试记录输出一个 0～1 之间的点击概率。

### 1.3 评价标准

比赛用 **Logloss（对数损失）** 评分，公式是：

$$\text{Logloss} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

用大白话解释：对每一条样本，看模型给「真实答案」打了多少概率，取对数、加负号。模型越确信正确答案，这一项越接近 0；模型越确信错误答案，惩罚越大，而且是**按指数级增长**的。

关键点是：**Logloss 越低越好**，而且它惩罚的是「过度自信的错误」。一个模型说「这条 99% 会点」，结果没点，会挨非常重的罚；说「这条 55% 会点」，结果没点，罚得就轻。所以 Logloss 逼着模型输出**校准过的概率**，而不是只排出个先后顺序。

另外要注意：因为 log(0) 是负无穷，实际计算时必须把概率裁剪到一个很小的范围内，否则一条极端预测就能让整个分数变成 `inf`。

---

## 二、解题过程

面对四千万行数据，我们的策略是：**先想办法把数据缩到能放进内存的规模，再把所有列交给 CatBoost 这种原生支持类别特征的模型，最后在验证集上报告 Logloss。** 整个过程围绕「省内存」和「别让模型记住用户 ID」两个目标展开。

### 2.1 数据探索（EDA）

我们首先解决「怎么把数据读进来」这个问题。直接用 `pd.read_csv` 读 4000 万行会占用几十 GB 内存，所以这里用 `chunksize` 分块迭代，一次只处理 100 万行：

```python
import pandas as pd

train_file = '../input/avazu-ctr-prediction/train.gz'
test_file  = '../input/avazu-ctr-prediction/test.gz'

# 训练集分块读：chunksize=10**6 表示每次只取 100 万行
reader = pd.read_csv(train_file, chunksize=10**6, iterator=True)

# 测试集只有 457 万行，可以一次读进内存
test = pd.read_csv(test_file, compression='gzip')

# 先看一眼测试集的列，训练集多一列 click
print(test.shape)          # (4577464, 23)
print(list(test.columns))
```

读完第一块之后，我们想确认一件事：**训练集和测试集的特征分布是否一致**。做法是逐个特征统计唯一值个数，比较两边：

```python
# 逐个特征比较「训练集有多少种取值」「测试集有多少种取值」
for col in test.columns:
    train_len = len(train[col].astype(str).value_counts())
    test_len  = len(test[col].astype(str).value_counts())
    print(col, 'train: {}  test: {}'.format(train_len, test_len))

# hour      train: 240   test: 24
# C1        train: 7     test: 7
# site_id   train: 3870  test: 2825
# app_id    train: 6135  test: 3952
# device_id train: 931588  test: 291759
# device_ip train: 2893925 test: 1077199
# C14       train: 2519  test: 1257
```

结果透露了两件事。

一是**测试集的取值种类普遍少于训练集**，比如 `site_id` 训练集 3870 种、测试集 2825 种。这不是数据有问题，而是因为测试集只有 1 天、训练集有 10 天，见过的站点自然更少。反过来说，**训练集里包含了测试集出现过的所有取值**，所以不用担心遇到完全没见过的类别。

二是 `hour` 的 240 对 24 —— 训练 10 天、测试 1 天。这提醒我们：**不能用随机划分验证集来模拟真实场景**。随机划分会让验证集里的时间点散布在训练时间中间，模型可以「偷看」前后时段；而真实测试集是未来的一整天，时间上完全在训练数据之后。材料里的作者仍然用了随机划分（这是这套方案的一个弱点），我们在总结里会指出。

接着看标签本身。我们统计点击比例：

```python
# click 的均值就是整体点击率
print(train['click'].value_counts(normalize=True))
# 1    0.170239
# 0    0.829761

# 也就是说，整体 CTR 约为 17%
```

17% 的点击率意味着：**如果模型给每条样本都输出 0.17，Logloss 大约是 0.46 左右；如果乱给极端值，分数会迅速恶化**。这也给了我们一个「及格线」的概念——任何模型的 Logloss 都应该显著低于「常数预测」的水平。

我们还做了一件有意思的事：算每个特征和标签的**互信息**（mutual information，衡量「知道这个特征能减少多少对标签的不确定性」）：

```python
from sklearn.feature_selection import mutual_info_classif

# 先手工构造一个组合特征：把设备 IP、机型、ID 拼起来当作「用户指纹」
train_data['user_info'] = (train_data.device_ip
                           + train_data.device_model
                           + train_data.device_id)
train_data = train_data.drop(['device_id', 'device_ip', 'device_model'], axis=1)

mi = mutual_info_classif(train_data, y_label, discrete_features=True)
mi = pd.Series(mi, index=train_data.columns).sort_values(ascending=False)
print(mi.head(8))

# user_info    0.386558
# C14          0.037118
# site_id      0.036873
# C17          0.034175
# site_domain  0.032669
# app_id       0.023342
# C21          0.021779
# C19          0.016762
```

`user_info` 的互信息是 0.387，比第二名 `C14`（0.037）高出一个数量级。这说明**「这个用户是谁」是预测点击最强的信号**——同一个人反复看到同类广告，点不点是有习惯的。

但这里埋着一个陷阱：如果把 `device_id` 这类几乎唯一的高基数列直接喂给模型，模型会「背答案」——它记住「这个 ID 在训练集里点过」，然后在验证集上完全失灵，因为验证集里是另一批 ID。另一份材料里的作者因此把三个设备列合并成一个 `user_info`，用「组合后的用户指纹」代替三个独立的高基数列。我们最后选择的主线是**保留原始的三列，交给 CatBoost 的 ordered target statistics 去处理**——它会按顺序逐条更新类别统计量，本身就是为了抑制这种「背答案」问题而设计的。互信息分析在这条主线里的价值，是让我们知道「用户维度最重要」，而不是让我们手工删列。

### 2.2 数据清洗

清洗的第一步是**降采样**。四千万行里 83% 是「没点击」的负样本，这些样本提供的信息很有限，却占了绝大部分内存和训练时间。我们的做法是在每个分块里做**负样本下采样**，让正负样本变成 1 : 1：

```python
import gc
import pandas as pd

train = pd.DataFrame()
for i, chunk in enumerate(reader):
    # 先随机取 65% 的行，进一步压缩数据量
    chunk = chunk.sample(frac=.65, replace=False, random_state=516)

    # 负样本只保留和正样本一样多的条数，让正负比例变成 1:1
    neg_samp = chunk[chunk['click'] == 0].sample(
        n=len(chunk[chunk['click'] == 1]),
        random_state=2021,
    )

    # 正样本全留，负样本只留下采样后的那部分
    train = pd.concat([train, neg_samp, chunk[chunk['click'] == 1]], axis=0)

    # 显式删掉临时变量并回收内存，否则循环几轮就 OOM 了
    del neg_samp
    gc.collect()

print(train.shape)            # (8925858, 24)
print(train['click'].value_counts())
# 1    4462929
# 0    4462929
```

最终得到 **8,925,858 行**，正负各 4,462,929 条，比例正好 1 : 1。

这里要讲清楚一个容易误解的点：**下采样会改变预测的概率含义**。原始点击率是 17%，下采样成 1 : 1 之后，模型学到的「基准概率」变成了 50%。如果直接把模型输出当作真实点击率提交，Logloss 会很差。常见的补救办法是事后做概率校正（把预测概率按采样比例换算回去）。材料里的这份方案没有做这一步，这是它 Logloss 只到 0.5657 的原因之一——严格来说，这个分数还有明显的改进空间。

第二步是**决定哪些列进入模型**：

```python
# 标签单独拿出来
y_train_full = train[['click']]

# 训练特征去掉 click 和 id：
#   id 是每条展示的唯一编号，纯粹是噪音，留着只会让模型过拟合
x_train = train.drop(['click', 'id'], axis=1)

# 测试集只需要去掉 id
x_test = test.drop(['id'], axis=1)

# 80/20 划分，stratify 保证两边点击率一致
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train_full, test_size=0.20, stratify=y_train_full, random_state=256,
)

# 划分完就可以释放大表了
del train
gc.collect()
```

第三步是**把列类型统一**。`hour`、`C1`、`banner_pos` 这些列在读进来时是整数，`site_id`、`app_id`、`device_ip` 是字符串。如果混着喂给模型，CatBoost 会把整数列当成数值特征去切分，而「把 hour=14102100 和 14102101 当成连续的数值」是完全没有意义的。所以我们要么统一转成字符串，要么在 `cat_features` 里把它们全部声明为类别。我们选后者，因为 CatBoost 支持直接声明整数列是类别：

```python
# 22 个特征全部按类别处理
# 这一步是整场比赛的关键：一旦声明为类别，CatBoost 内部会用
# target statistics（目标统计）编码，而不是按数值大小切分
cat_features = list(range(22))
print(x_train.dtypes.value_counts())
# int64      13
# object      9
```

### 2.3 模型训练与超参数调优

我们用的模型还是 **CatBoost**。选它的理由很直接：22 个特征全是类别，其中还有几个基数上万，而 CatBoost 原生就是为「大量类别特征」设计的——它内部会做 target statistics（把每个类别替换成「这个类别对应的平均点击率」），并用一套叫 ordered boosting 的技巧避免目标泄漏。

训练代码如下：

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=25,              # 只建 25 棵树：数据量太大，先跑通再说
    learning_rate=0.9,          # 学习率调高，配合很少的树数快速收敛
    task_type='GPU',            # 用 GPU 训练，CPU 上跑四千万行太慢
    loss_function='Logloss',    # 和比赛指标一致
    gpu_ram_part=0.9,           # 允许使用 90% 的显存
    boosting_type='Plain',      # 简化版的建树方式，显存占用更低
    max_ctr_complexity=2,       # 只做 1~2 个特征的组合统计，控制显存
    depth=6,                    # 树深 6，类别特征多时不宜太深
    gpu_cat_features_storage='CpuPinnedMemory',  # 类别特征放内存，省显存
)

model.fit(
    x_train, y_train,
    eval_set=(x_val, y_val),    # 提供验证集，用于观察和选轮次
    cat_features=cat_features,  # 声明全部 22 列都是类别
    verbose=10,                 # 每 10 轮打一次日志
)
```

几个参数的取舍值得解释：

- **`iterations=25` 而不是几百**：这不是因为 25 棵树最好，而是因为在这个数据规模下，每多一轮都要付出可观的时间和显存。作者的原话是，CatBoost 在做预测时还会撞上 memory limit（内存上限），所以整套方案是「先能跑完」优先。
- **`max_ctr_complexity=2`**：CatBoost 默认会尝试 1～4 个特征的组合统计。组合越多，类别组合的取值数量越爆炸，显存越吃紧。限制成 2 是在效果和资源之间取的折中。
- **`gpu_cat_features_storage='CpuPinnedMemory'`**：把类别特征表放在主机内存（pinned memory）里而不是显存里，是这套方案能在 Kaggle 的 GPU 上跑完的关键。

训练日志显示，Logloss 从第 0 轮的 0.6347 一路降到第 24 轮的 **0.5655**，而且每轮都还在下降，说明 25 棵树远远没有训够——如果资源允许，继续加树应该还能再降。

### 2.4 模型测试与指标报告

训练完成后，我们在验证集上算两个指标：

```python
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

# 用「最好的那一轮」做预测，而不是最后一轮
y_val_pred = model.predict(
    x_val,
    prediction_type='Probability',
    ntree_start=0,
    ntree_end=model.get_best_iteration(),
    thread_count=-1,
)

print('Out of folds logloss is {:.4f}'.format(log_loss(y_val, y_val_pred[:, 1])))
print('Out of folds roc_auc_score is {:.4f}'.format(roc_auc_score(y_val, y_val_pred[:, 1])))

# Out of folds logloss is 0.5657
# Out of folds roc_auc_score is 0.7727
```

**Logloss 0.5657，AUC 0.7727。**

怎么读这两个数？

先说 Logloss。前面提过，如果一律输出 0.5（因为我们已经把数据下采样成 1 : 1），Logloss 就是 `-log(0.5) = 0.693`。我们的 0.5657 比这个「什么都不学」的基线低了 0.13，说明模型确实学到了东西。但距离优秀还有明显差距，而且训练日志里 Logloss 一直到第 24 轮都还在下降——这说明模型的瓶颈主要不是「学不动」，而是「没训够」以及「下采样没有做概率校正」。

再说 AUC。0.7727 表示：随机抽一条点击记录和一条未点击记录，模型有 77% 的概率给点击的那条更高分。这个数字在 CTR 任务里算中等偏上，和 Logloss 反映的结论一致——**模型能排序，但概率校准得不好**。这两个指标的差异本身就是一条诊断信息：AUC 只看相对顺序，不受下采样影响；Logloss 看绝对概率，被下采样直接拖累。所以「AUC 还行、Logloss 很差」几乎总是意味着概率尺度出了问题。

作为对照，材料里还有一份在 300 万行样本上跑的决策树：训练准确率 0.8341、测试准确率 0.8331，AUC 只有 **0.7053**。准确率看起来比 CatBoost 还高，但那是被 83% 的负样本「撑」起来的——它其实主要在预测「不点」。另一份用 PCA 把特征压到 2 维再跑 XGBoost 的实验，AUC 只有 **0.6906**。这两个对照都说明：**在这种高基数类别数据上，把类别压成数值（PCA、哈希、直接当数字）会大量丢信息**。

最后生成提交文件：

```python
# 对测试集预测概率
y_test = model.predict(
    x_test,
    prediction_type='Probability',
    ntree_end=model.get_best_iteration(),   # 只用有效的那部分树
    thread_count=-1,
)

# 按官方样例提交的格式写文件
submission = pd.read_csv('../input/avazu-ctr-prediction/sampleSubmission.gz',
                         compression='gzip')
submission = pd.DataFrame({'id': submission['id'], 'click': y_test[:, 1]})
submission.to_csv('submission.csv', index=False)

# 检查形状和取值范围：概率必须落在 [0, 1]
print(submission.shape)
print(submission['click'].describe())
```

提交前我们特意检查了概率的取值范围。CTR 比赛里一个常见的翻车方式是：模型输出的是 logit 或者被下采样扭曲过的分数，直接提交上去 Logloss 会非常难看。

---

## 三、总结

本文中我们完成了 Avazu 移动广告点击率预测的完整流程。我们首先用分块读取解决了「四千万行读不进内存」的问题，并通过逐列统计唯一值确认了训练集覆盖了测试集的全部取值、训练集有 10 天而测试集只有 1 天、整体点击率约 17%。其次我们做了针对性的清洗：每个分块先取 65%，再把负样本下采样到和正样本等量，最终得到 8,925,858 行、正负各半的数据；去掉纯编号的 `id` 列；把 22 个特征全部声明为类别，避免模型把编号当数值切分。接着我们用 CatBoost 在 GPU 上训练，设 25 轮、学习率 0.9、树深 6、组合统计上限 2，并把类别特征表放在主机内存里以节省显存。最后我们在 20% 的验证集上得到 Logloss **0.5657**、AUC **0.7727**，并与决策树（AUC 0.7053）和 PCA + XGBoost（AUC 0.6906）两个对照做了比较。

这场比赛给我们三条很实用的经验。第一，**面对超大数据，先想「怎么变小」再想「怎么变好」**：分块读取、负样本下采样、及时 `gc.collect()`，这些工程手段决定了你能否跑完第一轮。第二，**下采样会改变概率的含义**：它让 AUC 看起来还行，却让 Logloss 严重失真，所以如果比赛指标是 Logloss，采样之后一定要做概率校正，否则分数会被系统性拖累。第三，**类别特征不要硬转数值**：把 `site_id` 当数字切分、或者用 PCA 把类别压成两维，都会丢掉大量信息——这也是为什么 CatBoost 这种原生支持类别特征的模型在这类比赛里几乎是默认选择。
