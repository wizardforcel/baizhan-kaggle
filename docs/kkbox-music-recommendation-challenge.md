# KKBox 音乐推荐挑战

## 一、比赛简介

### 1.1 比赛背景

KKBox 是台湾最大的在线音乐串流平台，用户可以通过手机 App 收听音乐。这个比赛的数据来自真实的用户收听记录。比赛希望找出那些听完一次后，还会在一个月内再次收听的歌曲。简单来说，就是预测用户会不会「回头听」同一首歌。

数据包含用户的第一次可观察收听事件，每个用户-歌曲对只有一条记录。平台提供了用户基本信息、歌曲信息、歌曲额外信息等表。这些信息可以帮助我们理解用户喜好和歌曲特征。

### 1.2 比赛目标

给定一次收听事件，预测它在未来一个月内是否会被重复收听。输入表主要有两张：

训练表 `train.csv` 包含：
- msno：用户匿名 ID
- song_id：歌曲匿名 ID
- source_system_tab：触发事件的系统标签，比如 my library、discover、search
- source_screen_name：用户看到的布局名称
- source_type：进入点，比如 online-playlist、local-library
- target：标签，1 表示一个月内有重复收听，0 表示没有

测试表 `test.csv` 只有前五列，没有 target。

另外还有 `songs.csv` 包含歌曲长度、genre_ids、artist_name 等信息，`members.csv` 包含用户城市、生日、性别、注册方式等，`song_extra_info.csv` 包含歌曲名称和 ISRC。

### 1.3 评价标准

比赛使用 AUC 作为评价指标。AUC 全称 Area Under Curve，即 ROC 曲线下面积。它衡量模型把正例排在负例前面的能力。AUC 越高越好，0.5 是随机猜测，1.0 是完美预测。

## 二、解题过程

我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步是把数据读进来，看看规模和列名：

```python
# 读取主要数据表，看看有多少条样本
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
songs = pd.read_csv('../input/songs.csv')
members = pd.read_csv('../input/members.csv')

print(train.shape)
print(train.head())
```

训练集有 7377418 条样本，6 列。能看到 source_system_tab 里 my library 和 discover 出现最多，说明用户多是通过「我的音乐库」和「发现」功能来听歌。

接着我们合并表，把用户特征和歌曲特征接上来：

```python
# 把用户表和歌曲表合并到训练集，丰富特征
train_members = pd.merge(train, members, on='msno', how='inner')
train_merged = pd.merge(train_members, songs, on='song_id', how='outer')
print(train_merged.shape)
```

合并后列数增加到 18 列。查看缺失值情况：

```python
# 检查哪些列有缺失
missing_cols = train_merged.columns[train_merged.isnull().any()].tolist()
print(missing_cols)
print(train_merged[missing_cols].isnull().sum())
```

发现 gender 列缺失最多，约 489 万条，因为很多用户没登记性别。歌曲表里 composer、lyricist 也大量缺失。这提示我们要做缺失值填充。

我们还计算了 target 的分布：

```python
# 看正负样本比例
data = train_merged.groupby('target').aggregate({'msno':'count'}).reset_index()
print(data)
```

正负样本大致五五开，数据是平衡的，不会一开始就偏向某一方。

### 2.2 数据清洗

数据有很多缺失值，还需要统一格式。我们先用 -5 填补数值列的缺失：

```python
# 对数值列的缺失值用 -5 填充，区分于真实值
object_cols = list(train_merged.select_dtypes(include=['float']).columns)
for col in object_cols:
    train_merged[col] = train_merged[col].fillna(np.int(-5))
```

这样模型能区分「缺失」和「正常值」。接着把数据类型压缩，减少内存：

```python
# 根据取值范围把 float 转成更小的整型，节省内存
def change_datatype(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        if (np.max(df[col]) <= 127 and np.min(df[col]) >= -128):
            df[col] = df[col].astype(np.int8)
        elif (np.max(df[col]) <= 32767 and np.min(df[col]) >= -32768):
            df[col] = df[col].astype(np.int16)
```

我们还发现年龄列 bd 有异常值，比如 -43、1051 岁。这些是数据录入错误。我们用经验值过滤：

```python
# 去掉年龄异常值，只保留 12 到 45 岁之间
def remove_outlier(df_in, col_name):
    fence_low = 12
    fence_high = 45
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out
df_final_train = remove_outlier(train_merged, 'bd')
```

对类别列我们直接转成 category 类型，既省内存又方便 LightGBM 处理。

在特征工程上，我们按歌曲、艺术家、用户计算重复收听率：

```python
# 计算每首歌的重复收听比例，作为新特征
song_count = train.loc[:,["song_id","target"]]
song_count1 = song_count.groupby(["song_id"],as_index=False).sum().rename(columns={"target":"repeat_count"})
song_count2 = song_count.groupby(["song_id"],as_index=False).count().rename(columns={"target":"play_count"})
song_repeat = song_count1.merge(song_count2,how="inner",on="song_id")
song_repeat["repeat_percentage"] = round((song_repeat['repeat_count']*100) / song_repeat['play_count'],1)
train = train.merge(song_repeat,on="song_id",how="left")
```

这一步让模型能知道这首歌本身是否容易被回头听，而不是只靠用户特征。

### 2.3 模型训练与超参数调优

我们选了 LightGBM 做基线，因为它对类别特征友好、训练快。先把表合并并做特征衍生：

```python
# 加载数据并设类别类型，减少内存占用
train = pd.read_csv('../input/train.csv', dtype={'msno':'category',
    'source_system_tab':'category','source_screen_name':'category',
    'source_type':'category','target':np.uint8,'song_id':'category'})
```

合并歌曲和用户信息后，我们提取了 registration_year、expiration_year、membership_days 等时间特征：

```python
members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['membership_days'] = (members['expiration_date'] - members['registration_init_time']).dt.days
members['membership_days'] = (members['membership_days']//200).astype('category')
```

然后用 3 折交叉验证训练 LightGBM：

```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
predictions = np.zeros(shape=[len(test)])

for train_indices, validate_indices in kf.split(train):
    train_data = lgb.Dataset(train.drop(['target'],axis=1).loc[train_indices,:], label=train.loc[train_indices,'target'])
    val_data = lgb.Dataset(train.drop(['target'],axis=1).loc[validate_indices,:], label=train.loc[validate_indices,'target'])
    params = {
        'objective':'binary',
        'boosting':'gbdt',
        'learning_rate':0.2,
        'num_leaves':2**8,
        'bagging_fraction':0.95,
        'bagging_freq':1,
        'feature_fraction':0.9,
        'metric':'auc'
    }
    lgbm_model = lgb.train(params, train_data, 100, valid_sets=[val_data])
    predictions += lgbm_model.predict(test.drop(['id'],axis=1))
predictions = predictions/3
```

参数里 num_leaves 设为 256，让模型有足够复杂度；learning_rate 0.2 相对大，配合 100 轮快速收敛。bagging_fraction 和 feature_fraction 做随机采样，防止过拟合。

### 2.4 模型测试与指标报告

训练过程中我们监控每轮的 AUC：

```
[1]  valid_0's auc: 0.733548
[20] valid_0's auc: 0.77346
[80] valid_0's auc: 0.770918
```

AUC 在前 20 轮快速上升，之后趋于平稳，约在 0.77 左右。3 折平均的 AUC 稳定在 0.72~0.77 之间，说明模型泛化能力不错。

最后我们生成提交文件：

```python
submission = pd.read_csv('../input/sample_submission.csv')
submission.target = predictions
submission.to_csv('submission.csv', index=False)
```

提交时要注意 id 列顺序必须和测试集一致，否则会被判错误。

## 三、总结

本文中我们完成了 KKBox 音乐推荐挑战的完整流程。我们首先通过 EDA 发现数据规模大、缺失值多、source_system_tab 为主要入口。其次做了缺失值填充、数据类型压缩、年龄异常值过滤等清洗工作，并在歌曲和用户层面加入重复收听率特征。接着用 LightGBM 建立基线模型，并通过 3 折交叉验证得到约 0.77 的 AUC。最后生成预测文件提交。

这个比赛提示我们，在推荐类问题里，行为序列衍生的统计特征（如歌曲重复率）往往比原始属性更有效。同时类别特征的正确处理和缺失值的合理编码，对模型效果影响很大。
