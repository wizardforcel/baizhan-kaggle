# See Click Predict Fix

## 一、比赛简介

### 1.1 比赛背景

SeeClickFix 是一个让居民报告城市问题的平台。用户可以提交道路坑洞、垃圾、树木、交通等问题，并在网页或手机端查看、评论和投票。一个问题获得的投票数，通常表示它受到多少人的关注。城市管理者可以利用这些信息了解哪些问题更紧急，因此预测投票数有实际意义。

本比赛的材料来自一个 Kaggle notebook，题目名称是 SeeClickFix。数据中的每一行对应一个问题，既有问题发生的地理位置，也有标题、详细描述、创建时间、来源设备等信息。难点在于：投票数是一个非负整数，而且大多数问题的票数很少，少数问题的票数却明显更高；同时，文本描述长短不一，部分描述为空，位置和时间也会影响用户是否看到问题。我们不能只依靠一个字段，而要把文字、时间和空间信息放在一起考虑。

材料里的探索显示，完整训练表有 223129 行，字段包括 `id`、`latitude`、`longitude`、`summary`、`description`、`num_votes`、`num_comments`、`num_views`、`source`、`created_time` 和 `tag_type`。在建模版本中，读入后的表有 223120 行，原因是代码在清洗过程中重新建立了索引。`description` 存在缺失，`source` 和 `tag_type` 也不是每条记录都有值。我们需要在保留有效信息的同时，避免把缺失值直接送进文本模型。

### 1.2 比赛目标

我们的目标是：给定一个 SeeClickFix 问题的已知信息，预测它会得到多少个 `num_votes`。输入可以包含两段文字：`summary` 是较短的摘要，`description` 是详细描述；还可以使用纬度、经度、创建时间和问题类型等结构化字段。输出是一个票数预测值，或者把票数看成若干类别时输出 50 个类别的概率。

材料里实际尝试了两条路线。第一条路线把 `num_votes` 看成 0 到 49 的离散类别，模型最后输出 50 个概率。第二条路线把 `num_votes` 看成连续数值，最后一层只输出一个线性值。两条路线都使用了问题文本，区别是第一条路线用分类准确率诊断，第二条路线用均方误差诊断。我们下面以材料中更完整的路线为主，同时说明另一条路线如何帮助理解任务。

### 1.3 评价标准

材料开头定义了 `rmsle`：先对预测值和真实票数分别做 `log1p`，再计算均方根误差。这个指标会降低极大票数样本对整体结果的压倒性影响，适合票数这种长尾分布；数值越小越好。材料同时定义了 `rmsle_v2`，它是普通的均方根误差形式。实际 notebook 还打印了回归模型的 MSE，以及分类模型的 `categorical_accuracy`，这些是本地验证时的辅助指标，不能和 RMSLE 混为一谈。

因此我们报告结果时遵守材料中的口径：回归部分说明使用 MSE 观察误差，分类部分报告 categorical accuracy；如果要对齐比赛的对数误差，则使用材料给出的 RMSLE 函数计算。材料没有给出最终 Kaggle 排名分数，所以我们不虚构排行榜成绩，只报告 notebook 中确实打印出来的验证数值。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读入训练表并查看每一列的类型和缺失情况。先摸清数据，比直接训练模型更重要，因为缺失的描述、没有标签的问题类型都会决定后面的清洗方式。

```python
# 导入 pandas，先从表格层面了解数据，而不是马上训练模型
import pandas as pd
import numpy as np

# 读取比赛训练数据，材料中的分隔符是逗号
TRAIN = pd.read_csv('../input/see-click-predict-fix/train.csv', sep=',')

# info 能同时显示行数、列名、非空数量和数据类型
TRAIN.info()

# 统计每个字段缺少多少值，帮助我们决定清洗策略
missing = TRAIN.isnull().sum().sort_values(ascending=False)
print(missing)

# 先看票数的基本统计，确认目标是非负整数且是否有长尾
print(TRAIN['num_votes'].describe())
```

材料给出的完整表有 223129 行。`latitude` 和 `longitude` 是浮点数，`num_votes`、`num_comments`、`num_views` 是整数，文本列是对象类型。`description` 只有 106819 条非空记录，说明详细描述缺失非常常见；`tag_type` 只有 53415 条非空记录，不能把它当作所有样本都有的必填特征。

接着我们观察票数的分布。材料先保留 `num_votes < 50` 的样本，再绘制所有问题和有 `tag_type` 问题的频数图。这样做不是随便删数据，而是为了观察主要分布，并让后面的 50 类分类输出有明确的类别范围。

```python
# 先截取材料中模型实际使用的主要票数范围
eda = TRAIN[TRAIN['num_votes'] < 50].copy()

# 缺失描述填成一个空格，后面才能安全地调用字符串处理函数
eda['description'] = eda['description'].fillna(' ')

# 将经纬度保留三位小数，便于按近似位置统计问题数量
eda['latitude'] = eda['latitude'].round(3)
eda['longitude'] = eda['longitude'].round(3)

# 查看每个票数占全部记录的比例，观察票数是否集中在低值
vote_rate = eda['num_votes'].value_counts(normalize=True).sort_index()
print(vote_rate.head(10))

# 只看有问题类型标签的记录，比较基线子集的票数分布
baseline = eda[eda['tag_type'].notna()]
print(baseline['num_votes'].value_counts(normalize=True).sort_index().head(10))
```

EDA 的数值结果很有代表性：全部数据的 `num_votes` 均值是 1.506624，方差是 1.526166；把票数限制在 1 到 49 后，均值变成 2.836344，方差变成 3.090021；再取有 `tag_type` 的基线子集，均值是 2.751568，方差是 3.414642。由此我们看到票数主要集中在低值，但仍然有高票问题，不能只预测一个固定均值。

我们还按问题类型统计平均票数和样本数量，并检查位置是否存在重复问题。材料特别查看了 Oakland 区域和同一经纬度的多个问题，这说明地理位置可能携带稳定的社区或街区信息。文本方面，`description` 的最大词数是 924、最小是 0、平均约 6.175852；`summary` 最大 49、最小 1、平均约 3.049646。详细描述虽然平均很短，但确实存在很长的异常样本。

```python
# 从创建时间中拆出小时、星期和年份，先观察时间特征能否被统计
eda['hour'] = pd.to_datetime(eda['created_time']).dt.hour.astype(str)
eda['dayofweek'] = pd.to_datetime(eda['created_time']).dt.weekday.astype(str)
eda['year'] = pd.to_datetime(eda['created_time']).dt.year.astype(str)

# 按问题类型计算平均票数，均值高的类型可能更容易引起关注
mean_by_type = eda.groupby('tag_type')['num_votes'].mean()
print(mean_by_type.sort_values(ascending=False).head())

# 按经纬度统计同一近似位置的记录数，帮助我们识别重复地点
location_counts = eda.groupby(['latitude', 'longitude']).size()
print(location_counts.sort_values(ascending=False).head())

# 查看一个具体区域的类型均值，材料中使用了 37.80 到 38 的纬度范围
okland = eda[(eda['latitude'] >= 37.80) & (eda['latitude'] <= 38)]
print(okland.groupby('tag_type')['num_votes'].mean().sort_values(ascending=False))
```

### 2.2 数据清洗

我们其次处理文本和结构化字段。材料中的清洗原则比较克制：先把缺失的 `description` 填为空格，再用 `gensim.utils.simple_preprocess` 把 `description` 和 `summary` 变成小写、去掉无关符号后的词序列。这样做的目的不是把文字变得漂亮，而是让分词器能够稳定地建立词表。我们没有删除整行缺失 `tag_type` 的记录，只有在明确做 baseline 实验时才筛出 `tag_type` 非空的行。

```python
# 复制数据，避免清洗动作意外修改最初读入的表
train = TRAIN[TRAIN['num_votes'] <= 50].copy()

# 详细描述为空时用空格代替，避免文本预处理遇到 NaN
train['description'] = train['description'].fillna(' ')

# simple_preprocess 会把文本切成适合建词表的规范化词序列
import gensim
train['description'] = train['description'].apply(
    lambda x: ' '.join(gensim.utils.simple_preprocess(x))
)
train['summary'] = train['summary'].apply(
    lambda x: ' '.join(gensim.utils.simple_preprocess(x))
)

# 经度和纬度保留三位小数，减少极细碎的地理类别数量
train['latitude'] = train['latitude'].round(3)
train['longitude'] = train['longitude'].round(3)

# 重新建立连续索引，后面按照数组下标切分训练和验证数据
train.reset_index(drop=True, inplace=True)
print(train[['summary', 'description', 'latitude', 'longitude']].head())
```

我们接着把时间字段转成三个简单的离散特征：小时、星期几和年份。材料还给出了经纬度小数位的解释：保留三位大致对应街区或街道尺度，比保留很多无意义的小数更适合做 one-hot 编码。`source` 虽然存在网页、Android、iPhone 等值，但材料中的 one-hot 主线使用的是时间和位置五列，因此我们不擅自把未稳定处理的字段塞进模型。

```python
# 时间拆分成类别特征，模型可以分别学习白天、周末和年份差异
time_value = pd.to_datetime(train['created_time'])
train['hour'] = time_value.dt.hour.astype(str)
train['dayofweek'] = time_value.dt.weekday.astype(str)
train['year'] = time_value.dt.year.astype(str)

# 把五个结构化字段拼在一起，准备交给同一个编码器
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
features = np.stack((
    train['hour'], train['dayofweek'], train['year'],
    train['latitude'], train['longitude']
), axis=-1)
onehot_train = encoder.fit_transform(features)

# 记录编码后的列数，建模时输入层需要使用这个维度
print(onehot_train.shape)
```

文本清洗后，我们用 Keras tokenizer 建立统一词表。`summary` 和 `description` 必须使用同一个 tokenizer，否则同一个单词在两段文本中可能对应不同编号。随后将详细描述补齐到 100 个词，将摘要补齐到 50 个词，过长文本截断，过短文本补零。补齐的重点是让神经网络可以接收形状一致的数组，而不是声称 100 或 50 是数据的自然长度。

```python
# 一个 tokenizer 同时学习摘要和详细描述，保证词编号一致
from keras.preprocessing import text, sequence
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(train['description']) + list(train['summary']))

# 文字先转成整数编号，再补齐到神经网络要求的固定长度
description_train = tokenizer.texts_to_sequences(train['description'])
description_train = sequence.pad_sequences(description_train, maxlen=100)

summary_train = tokenizer.texts_to_sequences(train['summary'])
summary_train = sequence.pad_sequences(summary_train, maxlen=50)

# 打印形状，确认每行都拥有固定数量的词位
print(description_train.shape)
print(summary_train.shape)
```

### 2.3 模型训练与超参数调优

我们接着建立文本模型。材料使用预训练的 GloVe 词向量（通过 `glove_w2v.txt` 加载），词向量矩阵的维度是 200。预训练向量可以把相近词映射到相近的数字空间，减少从零学习词义的压力。模型分别处理 `description` 和 `summary`：先查表得到词向量，再用双向 CuDNNLSTM 读取前后文，最后同时取最大池化和平均池化，保留一段文字中最强的信号和整体平均信号。

```python
# 根据词表建立 200 维的预训练词向量矩阵
from gensim.models import KeyedVectors

def build_matrix(word_index, embedding, size):
    # 预先填零，词向量文件中找不到的词保持为零向量
    matrix = np.zeros((len(word_index) + 1, size))
    for word, index in word_index.items():
        try:
            matrix[index] = embedding[word]
        except KeyError:
            # 未登录词没有向量时不报错，交给模型学习有限的零表示
            pass
    return matrix

# 材料中使用了转换后的 GloVe 文件，并指定文本格式
embedding = KeyedVectors.load_word2vec_format(
    '../input/glove2word2vec/glove_w2v.txt', binary=False
)
embedding_matrix = build_matrix(tokenizer.word_index, embedding, 200)
del embedding
print(embedding_matrix.shape)
```

我们把模型超参数明确列出来。分类实验使用 `BATCH_SIZE=64`、`LSTM_UNITS=200`、`EPOCHS=10`、`MAX_LEN=100` 和 `CLASSES=50`。`LSTM_UNITS` 越大，模型能记住的文本模式越多，但计算和过拟合风险也会增加。材料还尝试过 LSTM 单元为 300 的版本，以及 GoogleNews 300 维词向量的回归版本，这些属于调参方向，而不是材料给出的最终排行榜结论。

```python
# 这些数值来自材料的分类模型设置，先固定它们再比较验证结果
BATCH_SIZE = 64
LSTM_UNITS = 200
EPOCHS = 10
MAX_LEN = 100
CLASSES = 50

from keras.layers import Input, Embedding, Bidirectional, CuDNNLSTM
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Dense, Dropout, concatenate
from keras.models import Model

def build_model(embedding_matrix, one_hot_shape):
    # 两个输入分别承载详细描述和摘要，避免混淆文本长度
    words = Input(shape=(MAX_LEN,))
    summary = Input(shape=(50,))
    one_hot = Input(shape=(one_hot_shape,))

    # 词向量层固定预训练权重，降低小样本词义学习难度
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])

    # 摘要走同样的结构，得到第二份文本表示
    y = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(summary)
    y = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(y)
    y = concatenate([GlobalMaxPooling1D()(y), GlobalAveragePooling1D()(y)])

    # 把两段文本和时间空间特征合并，再用全连接层预测 50 个类别
    hidden = concatenate([x, y, one_hot])
    hidden = Dense(1000, activation='relu')(hidden)
    hidden = Dropout(0.4)(hidden)
    hidden = Dense(800, activation='relu')(hidden)
    hidden = Dropout(0.4)(hidden)
    result = Dense(CLASSES, activation='softmax')(hidden)
    model = Model(inputs=[words, summary, one_hot], outputs=result)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    return model
```

分类标签按照材料的写法建立为 50 列。这里的写法把第 `sample` 列赋成 `sample`，并不是标准的“一个位置赋 1”的 one-hot 编码，而是材料中真实使用的目标构造方式。因此我们保留它作为 notebook 的可复现描述，同时提醒读者：如果重新实现，应该确认标签编码和损失函数是否严格匹配。

```python
# 依据材料把票数写入 50 列目标矩阵，便于复现原 notebook
# 这里不擅自修改原始写法，以免改变材料中的实验口径
y_train = np.zeros(shape=(len(train), CLASSES))
for index, sample in enumerate(train['num_votes']):
    if sample < CLASSES:
        y_train[index, sample] = sample

# 固定随机抽取 20% 作为验证集，材料使用随机状态 999 的思路
from sklearn.model_selection import KFold
kfold = KFold(n_splits=2, shuffle=True, random_state=999)
test_idx = train.sample(frac=0.2, random_state=999).index
train_idx = train.index[~train.index.isin(test_idx)]

# 训练时把文本数组和 one-hot 结构化特征一起送入模型
model = build_model(embedding_matrix, onehot_train.shape[1])
model.fit(
    [description_train[train_idx], summary_train[train_idx], onehot_train[train_idx]],
    y_train[train_idx], batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2
)
```

### 2.4 模型测试与指标报告

我们最后在留出的 20% 数据上测试。材料中分类模型训练到第 10 个 epoch 时，训练集 categorical accuracy 达到 0.6049；两次验证实验打印的准确率分别是 0.4895609025 和 0.4930249977，随后输出的总体平均准确率是 0.491。这个结果只能说明 notebook 的分类诊断情况，不能被改写成官方排行榜分数。

```python
# 在未参与拟合的样本上评估，避免只看训练集准确率
pred_metrics = model.evaluate(
    [description_train[test_idx], summary_train[test_idx], onehot_train[test_idx]],
    y_train[test_idx]
)

# Keras 会按 compile 时的顺序返回 loss 和 categorical_accuracy
print(model.metrics_names)
print(pred_metrics)
print('Accuracy %.3f' % pred_metrics[1])
```

材料还尝试了连续回归模型：最后一层改成 `Dense(1, activation='linear')`，损失函数使用 MSE，优化器仍然使用 Adam；另一份实验使用 GoogleNews 的 300 维词向量，并让学习率按 `1e-3 * (0.6 ** global_epoch)` 衰减。代码会在验证集上计算并打印 MSE，但材料没有给出一个完整、可信的最终 MSE 数值，因此我们不在这里补写分数。材料还出现过一次池化层维度错误，说明调参时必须检查每层输入输出形状。

```python
# 回归路线把目标保留为一个数，而不是 50 个分类概率
from keras.layers import Dense

def build_regression_head(hidden):
    # 线性输出允许模型给出连续票数，适合观察回归误差
    result = Dense(1, activation='linear')(hidden)
    return result

# 材料中的回归训练使用 MSE，MSE 是预测值与真实值差的平方平均
regression_model.compile(loss='mse', optimizer='adam')
regression_model.fit(
    [description_train[train_idx], summary_train[train_idx]],
    train['num_votes'].iloc[train_idx], batch_size=64, epochs=1, verbose=2
)
pred = regression_model.predict(
    [description_train[test_idx], summary_train[test_idx]]
).flatten()
print('MSE %.3f' % np.mean(np.square(pred - train['num_votes'].iloc[test_idx])))
```

如果要按比赛的对数误差检查预测，我们使用材料开头的 RMSLE 函数，并先把预测值裁剪到非负范围。裁剪是因为票数不能为负，而线性回归输出可能出现负数；这一步只影响评估输入，不会凭空增加真实分数。我们同时保留 MSE，方便和 notebook 的已有输出对照。

```python
# RMSLE 要求预测值和真实值都不能小于零，所以先做非负裁剪
def rmsle(h, y):
    h = np.maximum(np.asarray(h), 0)
    y = np.maximum(np.asarray(y), 0)
    # log1p 能稳定处理票数为 0 的样本，再计算均方根差
    return np.sqrt(np.square(np.log1p(h) - np.log1p(y)).mean())

# 对验证集预测，并用同一批样本同时报告两种误差
pred = regression_model.predict(
    [description_train[test_idx], summary_train[test_idx]]
).reshape(-1)
y_valid = train['num_votes'].iloc[test_idx].to_numpy()
print('RMSLE %.6f' % rmsle(pred, y_valid))
print('MSE %.6f' % np.mean(np.square(pred - y_valid)))
```

我们还需要检查提交文件的形状和列名。材料片段主要展示了训练和验证，没有给出完整的最终提交分数，因此这里的检查重点是工程正确性：预测行数应和测试集一致，票数预测应保持非负，保存时不应额外写入 pandas 索引。这样可以避免模型本身没有问题，却因为文件格式错误而无法评测。

```python
# 读取测试表，使用训练阶段已经拟合好的 tokenizer 和 encoder
TEST = pd.read_csv('../input/see-click-predict-fix/test.csv', sep=',')
TEST['description'] = TEST['description'].fillna(' ')
TEST['description'] = TEST['description'].apply(
    lambda x: ' '.join(gensim.utils.simple_preprocess(x))
)
TEST['summary'] = TEST['summary'].apply(
    lambda x: ' '.join(gensim.utils.simple_preprocess(x))
)

# 测试数据必须使用同样的长度和同样的编码器，不能重新 fit
description_test = sequence.pad_sequences(
    tokenizer.texts_to_sequences(TEST['description']), maxlen=100
)
summary_test = sequence.pad_sequences(
    tokenizer.texts_to_sequences(TEST['summary']), maxlen=50
)

# 连续回归预测需要截断负数，再写成比赛要求的提交表
prediction = regression_model.predict([description_test, summary_test]).reshape(-1)
prediction = np.maximum(prediction, 0)
submission = pd.DataFrame({'id': TEST['id'], 'num_votes': prediction})
submission.to_csv('submission.csv', index=False)
print(submission.shape, submission.head())
```

## 三、总结

本文中我们完成了 See Click Predict Fix 的一套可复现解题流程。我们首先通过 EDA 认识了数据：完整训练表有 223129 行，字段同时包含摘要、详细描述、位置、时间、来源和问题类型；票数均值为 1.506624，低票样本很多，但仍有高票问题。其次我们做了数据清洗：补齐缺失描述，把文本规范化，纬度和经度保留三位小数，并从创建时间中拆出小时、星期和年份。接着我们使用共享 tokenizer、100 词的详细描述序列、50 词的摘要序列和 200 维预训练词向量，建立双向 LSTM、池化层和全连接层组成的模型，同时尝试了 50 类分类和单值回归两条路线。然后我们使用 Adam、批大小 64、训练 10 个 epoch，以及 0.4 的 Dropout 等材料中出现的设置进行验证。最后我们在留出的 20% 样本上报告了分类准确率：两次实验分别约为 0.490 和 0.493，整体平均约为 0.491；回归实验保留 MSE 及材料定义的 RMSLE 作为诊断指标，但没有把材料未给出的分数写成结论。

这个比赛给我们的启发是，城市问题的关注度不是由文字单独决定的。摘要和详细描述能告诉模型问题说了什么，时间能提示用户何时更容易看到问题，位置和问题类型则反映社区与事件类别。我们也看到，清洗和标签设计必须保持一致：分类输出、目标矩阵和评价指标必须互相匹配；回归输出则要注意非负约束和长尾分布。材料中出现的池化维度错误提醒我们，每次调整 LSTM 或池化结构后，都要检查输入张量的维度，再进行完整训练。