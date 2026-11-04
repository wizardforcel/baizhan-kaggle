# Criteo 展示广告点击率预估（Display Advertising Challenge）

## 一、比赛简介

### 1.1 比赛背景

你在网页上看到的每一条广告，背后都有一场**几毫秒内完成的拍卖**。广告平台要在这一瞬间决定：这块位置展示哪条广告、出价多少。判断依据只有一个——**这条广告被这个人点击的概率有多大**。

这个概率叫 **CTR**（Click-Through Rate，点击率）。它之所以重要，是因为广告按点击付费：曝光不收钱，用户点了才收钱。平台高估了 CTR，就会把位置卖给没人点的广告，白白浪费流量；低估了，又会错过真正能赚钱的广告。

Criteo 是一家做展示广告重定向的公司，它公开了一份真实的点击日志。每一条记录就是**一次广告曝光**：

- `label`：这一次曝光有没有被点击（1 = 点了，0 = 没点），这是我们要预测的目标；
- `I1` ～ `I13`：13 个**整数型**特征，是计数类的数值（比如「这个用户历史上看过多少条同类广告」）；
- `C1` ～ `C26`：26 个**类别型**特征，全部做过哈希处理，所以看起来像 `e5ba7672` 这样的 8 位十六进制串，我们不知道它具体代表什么，但知道「相同字符串 = 同一个类别」。

难点有三个：

1. **类别特征基数极大且长尾**。材料里的统计显示，`C3` 在训练集里有 **356058** 种不同取值，`C12` 有 **317159** 种。绝大多数取值只出现一两次，如果每个取值都开一个独立参数，模型根本学不动。
2. **训练集和测试集的分布对不上**。材料里明确记录了这个现象：很多类别取值在测试集里出现，却在训练集里从没出现过，导致「本地交叉验证分数」和「线上榜单分数」差距很大。
3. **正负样本不平衡**。材料里算出的多数类基线是 **77.337%**，也就是说正类（被点击）只占两成多。这时候模型只要全部猜「没点」就能拿到 77% 的准确率，所以绝不能用准确率当指标。

### 1.2 比赛目标

数据是 **Tab 分隔的纯文本，没有表头**，所以第一件事就是自己给列起名。按材料里的写法，一共 40 列：

```text
label, I1, I2, ..., I13, C1, C2, ..., C26
```

测试集比训练集少一列 `label`，只有 39 列。我们要给每一条测试记录预测一个 0 到 1 之间的**点击概率**，提交文件包含两列：`Id`（记录编号）和 `Predicted`（概率）。

材料里出现了两个规模完全不同的数据集，这一点很重要：

- **样本集** `dac_sample.txt`：官方提供的抽样版本，**100000 行 × 40 列**，拿来快速验证模型结构够不够用；
- **完整数据**：训练集 **1005000 行 × 40 列**（材料是分块读取、每块抽 5% 攒出来的），测试集 **6042135 行 × 39 列**。

材料里还有一句很关键的注释：**测试集样本量是训练集的 6 倍多**。这意味着训练数据相对稀缺，任何能「省着用数据」的技巧都很值钱。

### 1.3 评价标准

这个比赛用 **log loss**（对数损失，也叫交叉熵）打分。材料里的模型把损失设成 `Logloss`，评估时调用 `log_loss`，提交的也是概率而不是 0/1，这两点都指向同一个结论。

log loss 的算法是：对每条样本，取模型给「真实答案」的概率，求它的对数，再取负号，最后对所有样本求平均。写成公式是：

```
loss = -[ y·log(p) + (1-y)·log(1-p) ]
```

- 如果真实是 1，模型给的概率 p 越接近 1，`log(p)` 越接近 0，损失越小；
- 如果真实是 1，模型只给了 0.01，`log(0.01) ≈ -4.6`，损失就会非常大。

所以它衡量的是**「你对自己说的话有多自信，以及说得对不对」**。**这个指标越低越好**，完美预测时是 0。

为什么不用准确率？因为准确率只看四舍五入之后对不对——模型给 0.49 和给 0.001，在准确率眼里完全一样，但在 log loss 眼里差着十万八千里。**log loss 逼着模型输出有意义的概率，而不是模糊的中间值**，这正是广告出价真正需要的。

---

## 二、解题过程

我们走了两条路线：先用小样本集跑通 **DLRM**（Deep Learning Recommendation Model，一种专为推荐/CTR 设计的神经网络结构），再在全量数据上用 **CatBoost** 做特征工程和梯度提升。我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

**第一步：分块读取，均匀抽样。** 全量训练集很大，直接读会爆内存，所以我们用 `chunksize` 分块，并在**每一块里都抽 5%**，攒够 100 万行就停：

```python
import pandas as pd, numpy as np, time

# 数据没有表头，先按位置起名：1 个标签 + 13 个整数特征 + 26 个类别特征
continue_var = ['I' + str(i) for i in range(1, 14)]
cat_features = ['C' + str(i) for i in range(1, 27)]
col_names_train = ['Label'] + continue_var + cat_features
col_names_test = col_names_train[1:]        # 测试集没有 Label 列

# 分块读取：一次读 10 万行，读完一块就丢弃，内存占用恒定
reader = pd.read_csv('/kaggle/input/criteo-dataset/dac/train.txt', sep='\t',
                     names=col_names_train, chunksize=100000, iterator=True)
train = pd.DataFrame()
for i, chunk in enumerate(reader):
    if train.shape[0] > 1000000:            # 攒够 100 万行就停
        break
    # 关键：每一块都抽 5%，而不是只取文件开头的前 N 行
    train = pd.concat([train, chunk.sample(frac=.05, replace=False, random_state=911)],
                      axis=0)
```

这里有个非常值得记住的细节，材料里的原话大意是「只采样数据集中的前 n 行是不够的，要尽量均匀地读取」，而且**这样处理后成绩提升了 0.005 左右**。因为日志数据往往按时间排序，前 100 万行可能全来自同一天甚至同一个小时，分布和整体差异很大。分块抽样保证每一段时间都有代表。**在时序数据上抽样，随机性比数量更重要。**

**第二步：看缺失和基数。** 读完数据先看每列非空多少、有多少种取值：

```python
# non-null 计数直接告诉我们每列缺了多少
train.info()

# 逐列比较"训练集有多少种取值"和"测试集有多少种取值"
ls = list(train.columns)
ls.remove('Label')
for col in ls:
    train_len = len(train[col].astype(str).value_counts())
    test_len = len(test[col].astype(str).value_counts())
    print(col, ' | train: {} | test: {}'.format(train_len, test_len))
```

结果暴露了两个大问题：

1. **缺失很多**。`I1` 和 `I10` 只有 551544 个非空值（总共 1005000 行），缺了近一半；`I12` 只有 229927 个非空值；`C19`、`C20`、`C25`、`C26` 各有 571005 个，`C22` 只有 229940 个。
2. **测试集的类别种类远多于训练集**。`C3` 训练集 356058 种、测试集 **1654604** 种；`C12` 是 317159 对 1426969；`C16` 是 244079 对 1029668。原因很简单——我们只用了 5% 的训练数据，而测试集是全量的。材料里把这件事写得很直白：**「很多 id 类类别字段的枚举值在测试集中出现，却未在训练集中出现，这也导致了提交后的分数和 local cv 分数极为不一致」**。

这一点决定了后面的编码策略：**绝不能给每个类别开一个独立参数**，因为测试集里会不断冒出训练集没见过的取值。

**第三步：确认基线。** 我们数了一下两个类别的数量：

```python
label_counts = train.groupby('Label')['I1'].count()
print(f'Baseline: {max(label_counts.values) / sum(label_counts.values) * 100}%')
# 输出：Baseline: 77.337%
```

77.337% 就是「全部猜多数类」的准确率。任何模型如果连这个数都超不过，说明它什么都没学到。

### 2.2 数据清洗

**第一步：合并长尾类别。** 这是整份方案里最重要的一步。思路是：**保留出现频率最高的那部分取值，其余的统统归成一个叫 `longtail` 的类别**。这样既压住了基数，又保证了训练集和测试集用的是同一套「词典」：

```python
# 按"测试集取值种类是否超过训练集"把类别列分成两类：
# short（种类少、≤100）用 0.7，unbalance（种类多）用 0.5
for col in short:
    d1 = train[col].astype(str).value_counts()          # 按出现次数排序
    envalue = d1[: int(len(d1) * 0.7)].index            # 取前 70% 的取值
    # 不在名单里的，一律替换成 longtail
    train[col] = np.where(train[col].isin(envalue), train[col], 'longtail')
    test[col]  = np.where(test[col].isin(envalue),  test[col],  'longtail')

for col in unbalance:
    d1 = train[col].astype(str).value_counts()
    envalue = d1[: int(len(d1) * 0.5)].index            # 长尾更严重，只留前 50%
    # 注意这里多保留了 NaN：缺失本身是有信息的，不能并进 longtail
    train[col] = np.where((train[col].isin(envalue)) | (train[col].isna()), train[col], 'longtail')
    test[col]  = np.where((test[col].isin(envalue))  | (test[col].isna()),  test[col],  'longtail')
```

效果非常明显：`C3` 从训练 356058 种 / 测试 1654604 种，变成了 **178030 种 / 51215 种**。测试集的类别数甚至比训练集还少了——因为测试集里那些训练集没见过的稀有取值，现在全部落进了 `longtail` 这一个桶里。**这一步把「模型没见过这个取值」从灾难降级成了「模型把它当成稀有类别处理」。**

两个细节值得注意：

- **为什么两组比例不同？** 取值少的列长尾不严重，留 70%；取值多的列为了压基数只能留 50%。
- **为什么 `unbalance` 那组要额外保留 NaN？** 因为后面我们打算给缺失单独填 `'unknown'`，如果先把缺失并进 `longtail`，就再也分不出「稀有取值」和「没填」了。

**第二步：填缺失，但按字段类型分开处理。** 连续型字段用「同组均值」填，类别字段填一个明确的未知标记：

```python
# 连续型字段：先按 C7 分组求均值填充
# 为什么要分组？因为 C7 是个强相关的类别列，同类样本的数值分布更接近
fill_mean = lambda x: x.fillna(x.mean())
for col in continue_var:
    train[col] = train[col].groupby(train['C7']).apply(fill_mean)
    test[col]  = test[col].groupby(test['C7']).apply(fill_mean)
    # 如果整组都缺，组均值还是 NaN，就用全局均值兜底
    train[col] = train[col].fillna(test[col].mean())
    test[col]  = test[col].fillna(test[col].mean())

# 类别型字段：缺失统一填 'unknown'，让它自成一个类别
train = train.fillna('unknown')
test  = test.fillna('unknown')
```

这里体现的是「**缺失值填补要利用相关性**」。直接填全局均值等于假设「缺失是随机的」；但如果缺失本身和某些特征有关（比如某个渠道的曝光从来不回传数值），按 `C7` 分组填就能保住这部分结构。而类别字段填 `'unknown'` 而不是众数，是因为**「没填」和「填了最常见的值」在业务上是两回事**，合并会把信号抹掉。

**第三步：造交叉特征。** 材料里做了一个简单但有效的尝试——把两个数值特征相乘：

```python
# I13 和 I6 相乘：单独看两个计数意义有限，
# 乘起来能表达"两个行为同时发生的强度"，是二阶交互的廉价近似
train['I13_I6'] = train['I13'] * train['I6']
test['I13_I6']  = test['I13'] * test['I6']
```

它后来在重要性排名里排到第 6 位（4.564），说明确实有用。这是 CTR 任务的通用套路：**线性模型看不见特征交互，得靠人手造；树模型虽能自己分裂出来，但直接给现成的乘积能省下很多分裂次数。**

材料里还试过其他清洗手段，但最后都注释掉了：把连续特征分箱（`pd.qcut` 分 6 箱）、用 `RobustScaler` 做标准化、用 `TargetEncoder` 按标签均值编码、用 `CountEncoder` 按频次编码。这些在「长尾合并已经压住基数」的前提下，收益不如直接交给 CatBoost 的类别处理机制。

### 2.3 模型训练与超参数调优

**路线一：DLRM（小样本集上跑通结构）。** DLRM 的核心思想是：**类别特征查表变成稠密向量（embedding），数值特征过一个小 MLP，然后把所有向量两两做内积，捕捉二阶交互**。材料里用 TensorFlow 实现了原始论文的结构。

先看「两两内积」这一层。它把 N 个向量两两做点积，只保留上三角（不重复、也不含自己和自己）：

```python
class SecondOrderFeatureInteraction(tf.keras.layers.Layer):
    """把所有特征向量两两做内积，捕捉二阶交互"""
    def __init__(self, self_interaction=False):
        super().__init__()
        self.self_interaction = self_interaction

    def call(self, inputs):
        batch_size = tf.shape(inputs[0])[0]
        concat_features = tf.stack(inputs, axis=1)            # (batch, N, dim)
        dot_products = tf.matmul(concat_features,
                                 concat_features, transpose_b=True)   # (batch, N, N)
        ones = tf.ones_like(dot_products)
        mask = tf.linalg.band_part(ones, 0, -1)               # 取上三角
        out_dim = int(len(inputs) * (len(inputs) + 1) / 2)
        if not self.self_interaction:
            mask = mask - tf.linalg.band_part(ones, 0, 0)     # 再去掉对角线
            out_dim = int(len(inputs) * (len(inputs) - 1) / 2)
        # 把上三角展平成一维向量，作为"交互特征"
        return tf.reshape(tf.boolean_mask(dot_products, mask), (batch_size, out_dim))
```

内积越大，说明两个 embedding 在这个空间里越「同向」，模型于是自动学出「某类用户 × 某类广告 = 高点击」这种组合规律。`self_interaction=False` 是因为自己和自己内积只是模长的平方，没有信息量，去掉能省一半维度。

整个模型串起来是这样：

```python
class DLRM(tf.keras.Model):
    def __init__(self, embedding_sizes, embedding_dim, arch_bot, arch_top, self_interaction):
        super().__init__()
        # 每个类别列一张独立的 embedding 表，行数 = 该列的取值种类数
        self.emb = [tf.keras.layers.Embedding(size, embedding_dim) for size in embedding_sizes]
        self.bot_nn = MLP(arch_bot, out_activation='relu')     # 数值特征 → 同维度向量
        self.top_nn = MLP(arch_top, out_activation='sigmoid')  # 输出点击概率
        self.interaction_op = SecondOrderFeatureInteraction(self_interaction)

    def call(self, input):
        input_dense, input_cat = input
        # 类别列拆成 26 列，各自查表
        emb_x = [E(x) for E, x in zip(self.emb, tf.unstack(input_cat, axis=1))]
        dense_x = self.bot_nn(input_dense)                     # 数值 → 2 维
        Z = self.interaction_op(emb_x + [dense_x])             # 全部两两内积
        z = tf.concat([dense_x, Z], axis=1)                    # 拼起来喂给顶层
        return self.top_nn(z)
```

超参数上，我们选了很小的规模：`embedding_dim=2`、`arch_bot=[8, 2]`、`arch_top=[128, 64, 2]`、`self_interaction=False`，总参数量 **536,504**。优化器是 Adam、学习率 `1e-4`、batch 128、最多 100 轮，并挂上「连续 6 轮没进步就停」的早停。

**训练结果很说明问题。** 第 1 轮训练准确率 0.7718、验证 0.7814；到第 9 轮，训练准确率爬到 **0.8986**，验证准确率却掉到 **0.7418**，验证 loss 从 0.5032 一路涨到 **0.6894**。这是再典型不过的**过拟合**：模型开始背训练集里的样本，而不是学规律。最终测试集准确率 **0.7813**、loss 0.4766——比 77.337% 的基线只高了一点点。

> 小提示：材料里 DLRM 的输出层用了 `sigmoid`，损失函数却写成 `BinaryCrossentropy(from_logits=True)`。这两个设置是冲突的——`from_logits=True` 的意思是「输入里已经是对数几率，我来做 sigmoid」，可输入本身已经过了一次 sigmoid。这等于对概率又压了一次，训练信号被严重削弱。如果改成 `from_logits=False`（或者去掉输出层的 sigmoid），效果应该会明显不同。**这类「输出层和损失函数不匹配」的错误在 CTR 任务里非常常见，值得专门检查一遍。**

**路线二：CatBoost（全量数据上做特征工程 + 梯度提升）。** 相比神经网络，CatBoost 有三个直接的优势：原生处理类别特征（不需要 embedding）、对高基数类别有专门的统计编码、在小数据上更稳。我们先按 8:2 分层切分训练/验证集：

```python
from sklearn.model_selection import train_test_split

y_train = train[['Label']]
x_train = train.drop(['Label'], axis=1)

# stratify=y_train：保证验证集里的正负比例和训练集一致
# 不打乱分布的话，验证集的 logloss 会随切分方式剧烈波动
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.20, stratify=y_train, random_state=256)
```

然后先训一个「探路模型」，纯粹为了拿到每个特征的重要性：

```python
from catboost import CatBoostClassifier

# 探路：轮数故意开得很少（100 轮），只为快速拿到重要性排名
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.4,     # 学习率很大，配合少轮数快速收敛
    task_type='GPU',
    loss_function='Logloss',   # 和比赛指标对齐
    depth=8,
)
fit_model = model.fit(x_train, y_train,
                      eval_set=(x_val, y_val),
                      cat_features=cat_features,   # 声明哪些列是类别列
                      verbose=10)
feature_im = fit_model.feature_importances_
```

`cat_features` 是 CatBoost 的核心卖点：不用做独热或目标编码，模型内部用**有序目标统计**把类别替换成「该类别下的正样本比例」，并用排列避免标签泄漏。对 `C3` 这种 17 万种取值的列，这是唯一可行的办法。

重要性排名出来之后，我们砍掉了最不重要的 30%：

```python
feimp = pd.DataFrame({'feature': x_train.columns,
                      'importance': feature_im}).sort_values('importance', ascending=False)
# 只保留前 70% 的特征
feimp = feimp.iloc[: int(feimp.shape[0] * 0.7), ]

# 用选出来的特征重新切数据，并同步更新类别列名单
x_train = x_train[feimp['feature']]
x_val   = x_val[feimp['feature']]
test    = test[feimp['feature']]
cat_features0 = [c for c in feimp['feature'] if 'C' in c]
```

排名前列的是 `I6`（14.897）、`C7`（7.180）、`C15`（6.327）、`I11`（5.290）、`C3`（4.599）和我们自己造的 `I13_I6`（4.564）。排在最后的 `C5` 只有 0.0637——它几乎不贡献任何信息，留着只会增加噪声和训练时间。**用模型自己的重要性做特征筛选，比凭直觉删列可靠得多。**

最后用选出的特征重新训练并预测：

```python
model = CatBoostClassifier(iterations=100, learning_rate=0.4, task_type='GPU',
                           loss_function='Logloss', depth=8)
fit_model = model.fit(x_train, y_train, eval_set=(x_val, y_val),
                      cat_features=cat_features0, verbose=10)

# 只用到"最佳迭代"为止的树，避免后面那些开始过拟合的树参与预测
y_test_pred = model.predict(test, prediction_type='Probability',
                            ntree_end=model.get_best_iteration(),
                            thread_count=-1, verbose=None)[:, 1]
y_val_pred = model.predict(x_val, prediction_type='Probability',
                           ntree_start=0,
                           ntree_end=model.get_best_iteration(),
                           thread_count=-1, verbose=None)[:, 1]
```

`ntree_end=model.get_best_iteration()` 这个参数很关键。CatBoost 训练时会在验证集上记录每一轮的表现，最佳轮数是 98（总共 100 轮）。如果直接把全部 100 棵树都用来预测，就等于把最后两棵已经过拟合的树也算进去了。**训练时多留几轮、预测时只用最好的那几轮，是梯度提升的标准做法。**

### 2.4 模型测试与指标报告

**验证集指标。** 我们用 `log_loss` 直接算验证集的分数：

```python
from sklearn.metrics import log_loss, confusion_matrix
import numpy as np

print('Out of folds logloss is {:.4f}'.format(log_loss(y_val, y_val_pred)))
# 输出：Out of folds logloss is 0.4663
```

**0.4663** 就是这份方案的核心成绩。作为参照，如果对每条样本都懒洋洋地猜 0.5，log loss 会正好等于 `-ln(0.5) ≈ 0.693`，所以 0.4663 说明模型确实从特征里学到了东西；但和「能稳定区分点击与否」的理想水平相比，还有相当的距离。DLRM 在小样本集上的测试 loss 是 0.4766，两者大致在同一档——区别在于 CatBoost 这份是全量数据上的结果。

**混淆矩阵。** 我们再把概率按 0.5 切成 0/1，看看分类的具体形态：

```python
y_val_class = np.where(y_val_pred > 0.5, 1, 0)
cm = confusion_matrix(y_val, y_val_class)
# 归一化：每一行除以该行的总数，得到"每个真实类别被预测成什么"的比例
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_norm)
# [[0.94194271 0.05805729]
#  [0.68630989 0.31369011]]
```

读法是：真实为 0 的样本里 94.2% 判对了；真实为 1 的样本里**有 68.6% 被漏掉**，只抓出 31.4%。这不意外——正类只占两成多、阈值又是 0.5，模型天然偏向多数类。**这再次说明不能用准确率：准确率看着接近 0.79，真正关心的「点击」却漏了七成。**

**提交文件。** 因为 log loss 对概率的**绝对数值**敏感（不是只看排序），材料里做了一件很聪明的事——同时提交三份，把预测概率整体乘以不同系数，看榜单更认哪一个：

```python
# 直接读官方的随机提交样例，保证 Id 的顺序和行数完全正确
submission = pd.read_csv('../input/criteo-display-ad-challenge/random_submission.zip',
                         compression='zip')

# 原始预测
pd.DataFrame({'Id': submission['Id'], 'Predicted': y_test_pred}) \
    .to_csv('submission.csv', index=False)
# 整体压低 2%：如果模型系统性高估，这一份会更好
pd.DataFrame({'Id': submission['Id'], 'Predicted': y_test_pred * 0.98}) \
    .to_csv('submission2.csv', index=False)
# 整体抬高 2%：如果模型系统性低估，这一份会更好
pd.DataFrame({'Id': submission['Id'], 'Predicted': y_test_pred * 1.02}) \
    .to_csv('submission3.csv', index=False)
```

这个技巧在 log loss 类比赛里特别好用：模型排序对了、但概率整体偏大或偏小，乘一个系数就能修正，**完全不需要重新训练**。乘 0.98 和 1.02 就是在试探模型的概率校准偏了多少。提交文件一共 **6042135 行**，`Id` 从 60000000 开始递增。

---

## 三、总结

本文中我们完成了 Criteo 展示广告点击率预估的完整流程。我们首先用 EDA 摸清了数据：40 列由 1 个标签、13 个整数特征和 26 个哈希后的类别特征组成，多数类基线是 77.337%；`C3` 在训练集有 356058 种取值、测试集有 1654604 种，`I12` 缺了将近八成。其次我们做了针对性的清洗：分块读取并在每块里抽 5%（这一改动带来约 0.005 的提升）、按「测试集种类是否超过训练集」把类别列分成两组、分别保留前 70% 和前 50% 的高频取值、其余归入 `longtail`（`C3` 因此从 35 万/165 万降到 17.8 万/5.1 万）、连续特征按 `C7` 分组均值填补并以全局均值兜底、类别缺失填 `'unknown'`、并造了 `I13_I6` 交叉特征。接着我们先在小样本集上实现了 DLRM（embedding 维度 2、两两内积的二阶交互、总参数 536504），观察到训练准确率升到 0.8986 而验证跌到 0.7418 的明显过拟合；再在全量数据上用 CatBoost（`Logloss`、depth=8、lr=0.4、原生类别处理）先评估特征重要性、砍掉最弱的 30% 后重训。最后我们用 `log_loss` 得到 0.4663 的验证分数，用归一化混淆矩阵确认正类召回只有 31.4%，并把预测概率按原值、×0.98、×1.02 三份一起提交，用榜单反推模型的校准偏差。

这个比赛给我们最大的启发有两点。第一，**CTR 任务里最贵的资源不是算力，而是「类别特征的分布一致性」**。训练集和测试集的类别分布一旦错位，本地验证分数就完全不可信；长尾合并这种看起来粗糙的操作，恰恰是让两边对齐的最有效手段。第二，**小样本上验证结构、大样本上验证效果**。DLRM 在 10 万行上跑一轮只要 9 秒，能快速暴露「输出层与损失函数不匹配」「模型容量过大」这类结构性问题；等问题改完，再上全量数据做真正的特征工程。
