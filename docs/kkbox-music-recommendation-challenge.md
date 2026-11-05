# KKBox 音乐推荐挑战（KKBox's Music Recommendation Challenge）

## 一、比赛简介

### 1.1 比赛背景

KKBox 是台湾最大的在线音乐串流平台，用户可以在手机 App 上在线收听音乐。这个比赛的数据来自真实的用户收听行为。比赛的目的是：判断用户听过一首歌之后，在接下来一个月里会不会「回头」再听同一首歌。

换个生活化的说法：你打开音乐 App，点了一首歌，听完了。平台想知道，接下来 30 天你还会不会再点这首歌？如果会，说明这首歌对你有吸引力；如果不会，说明它只是一次性的尝试。对音乐平台来说，这个判断能直接用于「猜你喜欢」和「每日推荐」。

数据由几张表组成：用户第一次可观察的收听事件、用户基本信息、歌曲信息、歌曲额外信息。每个「用户-歌曲」组合只有一条记录。这些表合在一起，可以帮我们理解用户偏好和歌曲特征。

### 1.2 比赛目标

给定一次收听事件，预测它在未来一个月内是否会被重复收听。核心表有：

训练表 `train.csv` 包含：

- `msno`：用户匿名 ID；
- `song_id`：歌曲匿名 ID；
- `source_system_tab`：触发这次收听的系统标签，比如 my library、discover、search；
- `source_screen_name`：用户看到的界面布局名称；
- `source_type`：进入点，比如 online-playlist、local-library；
- `target`：标签，1 表示一个月内有重复收听，0 表示没有。

测试表 `test.csv` 只有前五列，没有 `target`，需要我们预测。

另外还有：

- `songs.csv`：歌曲长度、genre_ids、artist_name 等；
- `members.csv`：用户城市、生日（`bd`）、性别、注册方式等；
- `song_extra_info.csv`：歌曲名称和 ISRC（国际标准录音编码）。

### 1.3 评价标准

比赛使用 **AUC（Area Under the ROC Curve，ROC 曲线下面积）** 作为评价指标。AUC 衡量的是：模型把「会重复收听的样本」排在「不会重复收听的样本」前面的概率。直观理解就是「随机抽一个正样本和一个负样本，模型给正样本打分更高的概率」。

AUC 越高越好：0.5 等于瞎猜，1.0 是完美排序。这个指标只看排序，不看具体概率值，所以特别适合推荐、广告这类「排序」任务。

## 二、解题过程

我们的主线是：先合并用户表和歌曲表得到更丰富的特征，再做缺失值编码、数据类型压缩、年龄异常值清洗，然后加入「歌曲重复收听率」等统计特征，最后用 LightGBM 三折交叉验证出预测。下面按四步展开。

### 2.1 数据探索（EDA）

第一步是把主要数据表读进来，看看规模和列名：

```python
import numpy as np
import pandas as pd

# 读取四张主要表
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
songs = pd.read_csv('../input/songs.csv')
members = pd.read_csv('../input/members.csv')

print(train.shape)   # 样本数和列数
print(train.head())
```

训练集有 7377418 条样本，6 列。这是一个相当庞大的表，接近 740 万行，所以后面处理内存时要格外小心。

我们统计一下 `source_system_tab` 的分布，看看用户主要通过哪些入口听歌：

```python
# 看看各个入口的样本量
train['source_system_tab'].value_counts()
```

结果里 my library 和 discover 出现最多，说明大多数用户是通过「我的音乐库」和「发现」页面来听歌的。这两个入口的推荐逻辑很不一样，所以入口本身就是一个有信息量的特征。

接着我们合并表，把用户特征和歌曲特征接到训练集上：

```python
# 先按用户 ID 合并成员表
train_members = pd.merge(train, members, on='msno', how='inner')

# 再按歌曲 ID 合并歌曲表；用 outer 保留没有匹配上的歌曲信息
train_merged = pd.merge(train_members, songs, on='song_id', how='outer')
print(train_merged.shape)
```

合并之后列数增加到 18 列。合并的代价是可能出现缺失：用户没填的字段、歌曲库里没登记的字段，都会变成 NaN。

我们检查哪些列有缺失、缺多少：

```python
# 找出含缺失值的列
missing_cols = train_merged.columns[train_merged.isnull().any()].tolist()
print(missing_cols)
print(train_merged[missing_cols].isnull().sum())
```

结果显示 `gender` 缺失最多，约 489 万条——很多用户根本没有登记性别。歌曲表里的 `composer`、`lyricist` 也大量缺失。这提示我们：**缺失不是异常，而是数据本身的常态**，需要用一种方式告诉模型「这里没有信息」。

再看目标标签的分布：

```python
# 统计标签为 0 和 1 的样本数
data = train_merged.groupby('target').aggregate({'msno':'count'}).reset_index()
print(data)
```

正负样本大致五五开，属于比较平衡的数据。这意味着我们不需要像处理严重不平衡那样去调类别权重，可以把精力放在特征和模型上。

### 2.2 数据清洗

数据有几处需要处理：缺失值、数据类型、年龄异常值。

第一，缺失值填充。对数值列的缺失，我们用 -5 填。为什么用 -5 而不是 0 或均值？因为 0 和均值都可能是真实取值，模型会分不清「这个值是 0」和「这个值缺失」。用 -5 这种现实中不可能出现的值，模型就能学到「-5 代表缺失」这个独立含义。

```python
# 数值列缺失用 -5 填充，让模型能区分「缺失」
object_cols = list(train_merged.select_dtypes(include=['float']).columns)
for col in object_cols:
    train_merged[col] = train_merged[col].fillna(np.int(-5))
```

第二，数据类型压缩。740 万行 × 18 列如果全用 64 位浮点，内存会迅速爆掉。我们根据每列的实际取值范围，把它降到能装下的最小整数类型：

```python
def change_datatype(df):
    """按取值范围把 float 降成更小的整型，省内存"""
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        if np.max(df[col]) <= 127 and np.min(df[col]) >= -128:
            df[col] = df[col].astype(np.int8)      # 1 字节
        elif np.max(df[col]) <= 32767 and np.min(df[col]) >= -32768:
            df[col] = df[col].astype(np.int16)     # 2 字节
```

这一步对实际比赛的运行时间帮助非常大：内存降下来，后面的合并、分组、训练都会快很多。

第三，年龄异常值。`bd`（birthday，生日）列里有诸如 -43 岁、1051 岁这样的记录，明显是录入错误。我们把年龄限制在合理范围：

```python
def remove_outlier(df_in, col_name):
    """只保留 12 到 45 岁之间的记录，去掉明显异常值"""
    fence_low = 12
    fence_high = 45
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

df_final_train = remove_outlier(train_merged, 'bd')
```

对城市、性别、注册方式这类类别列，我们直接转成 `category` 类型。这样既省内存，又能让 LightGBM 原生地按类别处理。

第四，特征工程。我们觉得「这首歌本身容易被重复听」是很重要的信号，所以按歌曲统计重复收听的比率：

```python
# 按 song_id 汇总：重复收听次数和总播放次数
song_count = train.loc[:, ["song_id", "target"]]
song_count1 = song_count.groupby(["song_id"], as_index=False).sum() \
    .rename(columns={"target": "repeat_count"})
song_count2 = song_count.groupby(["song_id"], as_index=False).count() \
    .rename(columns={"target": "play_count"})

# 合并后算重复收听百分比
song_repeat = song_count1.merge(song_count2, how="inner", on="song_id")
song_repeat["repeat_percentage"] = round(
    (song_repeat['repeat_count'] * 100) / song_repeat['play_count'], 1)

train = train.merge(song_repeat, on="song_id", how="left")
```

这一步让模型不仅知道「这个用户是谁」，还知道「这首歌整体上是不是一首让人想再听的歌」。后面我们也可以对艺术家做同样的统计。

### 2.3 模型训练与超参数调优

我们选 LightGBM 作为主模型。原因很实际：数据量大、类别特征多，LightGBM 训练快、内存友好，而且对类别特征有原生支持。

首先用带类型的 `dtype` 读数据，从源头上控制内存：

```python
# 直接指定每列类型，避免读进来再转换浪费内存
train = pd.read_csv('../input/train.csv', dtype={
    'msno': 'category',
    'source_system_tab': 'category',
    'source_screen_name': 'category',
    'source_type': 'category',
    'target': np.uint8,
    'song_id': 'category'
})
```

接着从成员表里提取时间特征。注册时间和到期时间原本是日期字符串，我们分别取年份，并计算会员时长：

```python
# 从注册时间、到期时间里取出年份
members['registration_year'] = members['registration_init_time'].apply(
    lambda x: int(str(x)[0:4]))
members['expiration_year'] = members['expiration_date'].apply(
    lambda x: int(str(x)[0:4]))

# 会员总天数 = 到期日 - 注册日
members['membership_days'] = (
    members['expiration_date'] - members['registration_init_time']).dt.days

# 把天数按 200 天一段粗分，减少取值种类
members['membership_days'] = (members['membership_days'] // 200).astype('category')
```

然后合并所有特征，用 3 折交叉验证训练 LightGBM，并把三折的测试集预测取平均：

```python
from sklearn.model_selection import KFold
import lightgbm as lgb

kf = KFold(n_splits=3)
predictions = np.zeros(shape=[len(test)])

for train_indices, validate_indices in kf.split(train):
    train_data = lgb.Dataset(
        train.drop(['target'], axis=1).loc[train_indices, :],
        label=train.loc[train_indices, 'target'])
    val_data = lgb.Dataset(
        train.drop(['target'], axis=1).loc[validate_indices, :],
        label=train.loc[validate_indices, 'target'])

    params = {
        'objective': 'binary',       # 二分类
        'boosting': 'gbdt',          # 梯度提升树
        'learning_rate': 0.2,        # 学习率偏大，配合 100 轮快速收敛
        'num_leaves': 2**8,          # 每棵树的叶子数 = 256
        'bagging_fraction': 0.95,    # 每次迭代随机用 95% 数据
        'bagging_freq': 1,
        'feature_fraction': 0.9,     # 每次迭代随机用 90% 特征
        'metric': 'auc'
    }

    lgbm_model = lgb.train(params, train_data, 100, valid_sets=[val_data])
    predictions += lgbm_model.predict(test.drop(['id'], axis=1))

predictions = predictions / 3    # 三折取平均，降低方差
```

参数这样设的理由是：`num_leaves=256` 给模型足够的复杂度；`learning_rate=0.2` 配上 100 轮能在可接受时间内收敛；`bagging_fraction` 和 `feature_fraction` 都是随机采样，相当于给每棵树不同的数据视角，能有效抑制过拟合。

### 2.4 模型测试与指标报告

训练时我们打开了验证集监控，日志里的 AUC 变化如下：

```
[1]  valid_0's auc: 0.733548
[20] valid_0's auc: 0.77346
[80] valid_0's auc: 0.770918
```

读法是：第 1 轮 AUC 就有 0.733，说明特征本身已经很有区分度；到第 20 轮升到 0.773；第 80 轮时略降到 0.771。这提示模型在第 20~80 轮之间基本收敛，再训下去可能开始轻微过拟合。

三折之间有波动，但整体平均 AUC 稳定在 0.72~0.77 之间，说明模型泛化能力不错，没有明显「折与折之间打架」的现象。

最后把平均后的概率写入提交文件：

```python
submission = pd.read_csv('../input/sample_submission.csv')
submission.target = predictions        # 填入预测概率
submission.to_csv('submission.csv', index=False)
```

提交前必须检查一件事：**`id` 列的顺序要和测试集严格一致**。如果顺序打乱，即使预测内容正确也会被判错。这一步看似琐碎，却是很多比赛翻车的地方。

## 三、总结

本文中我们完成了 KKBox 音乐推荐挑战的完整流程。我们首先用 EDA 确认数据规模接近 740 万行、缺失值普遍、用户主要在「我的音乐库」和「发现」两个入口听歌。其次我们对数值缺失用 -5 编码、压缩数据类型节省内存、过滤不合理的年龄，并统计了歌曲层面的重复收听率。接着我们用 LightGBM 配合 3 折交叉验证训练，学习率 0.2、256 个叶子、随机采样防过拟合，在验证集上得到约 0.72~0.77 的 AUC。最后我们对三折预测取平均并生成提交文件。

这个比赛让我们体会到：在推荐类问题里，**从行为数据里统计出来的「重复率」特征往往比原始属性更有用**；而面对大数据表，缺失值编码、内存压缩这些「脏活」做得好不好，直接决定后面能不能顺利跑起来。