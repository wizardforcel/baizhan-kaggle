# 预测 StackOverflow 关闭问题

## 一、比赛简介

### 1.1 比赛背景
StackOverflow 是全球最大的程序员问答社区。为了保证内容质量，社区会关闭不符合规范的问题。关闭原因分成几类：不是真实问题、不具建设性、离题、过于本地化，以及保持开放的正常问题。这个比赛要求我们根据问题发布时的信息，预测它最终会不会被关闭、以及被归为哪一类。

这道题的难点在于**信息不完整**。我们只能看到发帖当时的快照，比如标题、正文、标签、作者声望、作者注册时间、发帖时间，看不到之后的评论和投票。模型必须从这些静态信息里判断问题的“命运”。另外，类别之间界限模糊，比如“不具建设性”和“离题”经常重叠，人工标注本身也有噪声。

### 1.2 比赛目标
训练集 `train-sample.csv` 包含以下列：

- `PostId`：帖子编号；
- `PostCreationDate`：发帖时间；
- `OwnerUserId`：发帖用户编号；
- `OwnerCreationDate`：用户注册时间；
- `ReputationAtPostCreation`：发帖时的声望；
- `OwnerUndeletedAnswerCountAtPostTime`：发帖时该作者未被删除的回答数；
- `Title`：标题；
- `BodyMarkdown`：正文；
- `Tag1` 到 `Tag5`：最多五个标签；
- `PostClosedDate`：关闭时间；
- `OpenStatus`：目标列，取值为 open、not a real question、not constructive、off topic、too localized。

测试集结构相同，只是没有 `OpenStatus`。我们要给每条测试样本预测五类中的一类。

### 1.3 评价标准
比赛以**多分类准确率**为核心指标，也就是预测正确的样本数除以总样本数。五分类的随机猜测准确率是 20%，模型要做的是明显高于这个水平。训练时我们也可以用交叉熵损失来观察模型是否在学，但最终汇报用准确率。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读入数据，看看字段和样本。

```python
import pandas as pd
import numpy as np

# 读入训练样本
df = pd.read_csv('../input/predict-closed-questions-on-stack-overflow/train-sample.csv')
print(df.shape)
df.head()
```

前几行显示：`Title` 和 `BodyMarkdown` 是长文本，`Tag1` 到 `Tag5` 是标签，`OpenStatus` 是字符串类别。`PostClosedDate` 只有被关闭的帖子才有值。

接着看目标分布。

```python
# 统计每个类别的样本数
print(df['OpenStatus'].value_counts())
```

`open` 类最多，其余四类加起来约占一半。类别不均衡，但不算极端，所以不需要做重采样，只要在切分验证集时做分层抽样即可。

我们还检查了缺失值。

```python
# 每列缺失数量
print(df.isnull().sum().sort_values(ascending=False).head(10))
```

结果主要集中在 `Tag2` 到 `Tag5` 和 `PostClosedDate`。标签缺失是正常的，因为很多问题只打一个标签；`PostClosedDate` 缺失说明这条问题没有被关闭。关键在于，`PostClosedDate` 是**事后信息**，如果把它当特征，模型等于提前偷看了答案，线下分数会虚高、线上却用不了。所以我们在建模前必须把它删掉。

我们再看标签的取值数量。

```python
# Tag1 有多少种不同取值
print('Tag1 nunique:', df['Tag1'].nunique())
# 前 20 个高频标签
print(df['Tag1'].value_counts().head(20))
```

`Tag1` 有上千种取值。如果直接 One-Hot，会生成上千列稀疏特征，既占内存又容易过拟合。所以我们后面改用二进制编码。

时间字段也值得看。

```python
# 把发帖时间转成日期类型，看年份范围
df['PostCreationDate'] = pd.to_datetime(df['PostCreationDate'])
print(df['PostCreationDate'].dt.year.value_counts())
```

数据覆盖 2008 到 2012 年。不同年份的社区规范不同，早期问题可能更容易被关闭，所以年份本身就是一个有用特征。

最后看文本长度。

```python
# 标题和正文的字符数
df['title_len'] = df['Title'].str.len()
df['body_len'] = df['BodyMarkdown'].str.len()
print(df[['title_len', 'body_len']].describe())
```

正文长度差异很大，短的几十个字符，长的上万字符。长度本身和“是否被关闭”有关：太短的问题往往信息不足，更容易被判为“不是真实问题”。

### 2.2 数据清洗

我们的清洗分三步：处理时间、处理文本、处理类别。

第一步，从两个时间字段里拆出有用的部分。

```python
def datetime_features(df):
    """把发帖时间和作者注册时间拆成星期、月份、年份、小时、分钟"""
    df = df.copy()

    # 转成 datetime 类型
    df['PostDate'] = pd.to_datetime(df['PostDate'])
    df['OwnrDate'] = pd.to_datetime(df['OwnrDate'])

    # 发帖时间的各个分量
    df['PostDay'] = df['PostDate'].dt.dayofweek
    df['PostMonth'] = df['PostDate'].dt.month
    df['PostYear'] = df['PostDate'].dt.year
    df['PostHour'] = df['PostDate'].dt.hour
    df['PostMin'] = df['PostDate'].dt.minute

    # 作者注册时间的各个分量
    df['OwnrDay'] = df['OwnrDate'].dt.dayofweek
    df['OwnrMonth'] = df['OwnrDate'].dt.month
    df['OwnrYear'] = df['OwnrDate'].dt.year
    df['OwnrHour'] = df['OwnrDate'].dt.hour
    df['OwnrMin'] = df['OwnrDate'].dt.minute

    # 账号年龄：发帖时间减去注册时间，单位秒
    df['AccAge'] = (df['PostDate'] - df['OwnrDate']) / np.timedelta64(1, 's')

    # 原始时间列已经拆完，删掉
    del df['PostDate']
    del df['OwnrDate']
    return df
```

这些分量看起来零碎，但组合起来能表达“新用户深夜发帖”“老用户工作日提问”这类模式。`AccAge` 尤其重要：账号越新，越可能不熟悉社区规则，问题被关闭的概率越高。

第二步，把标题和正文合并。

```python
# 标题包含核心问题，正文包含细节，拼在一起让模型同时看到
df['BodyMarkdown'] = df['Title'] + ' ' + df['BodyMarkdown']
df = df.drop(columns=['Title'])
df = df.rename(columns={'BodyMarkdown': 'QuestionText'})
```

第三步，类别编码。

```python
import category_encoders as ce

# 时间分量取值不多，用二进制编码压成几位
cols = ['PostDay', 'PostMonth', 'PostYear', 'PostHour', 'PostMin',
        'OwnrDay', 'OwnrMonth', 'OwnrYear', 'OwnrHour', 'OwnrMin']
encoder = ce.binary.BinaryEncoder(cols=cols)
sub_cols = encoder.fit_transform(df[cols])
df = pd.concat([df, sub_cols], axis=1)
df = df.drop(columns=cols)

# Tag1 取值太多，同样用二进制编码
tag_encoder = ce.binary.BinaryEncoder()
tag_bits = tag_encoder.fit_transform(df['Tag1'])
df = pd.concat([df, tag_bits], axis=1)
df = df.drop(columns=['Tag1'])
```

二进制编码的做法是：先把类别映射成整数，再把整数写成二进制，每一位变成一列 0/1。比如 1000 个类别只需要约 10 列，比 One-Hot 的 1000 列省得多，也保留了类别之间的一部分数值关系。

最后把目标字符串映射成整数。

```python
target_map = {
    'not a real question': 0,
    'not constructive': 1,
    'off topic': 2,
    'open': 3,
    'too localized': 4
}
df['OpenStatus'] = df['OpenStatus'].map(target_map)
```

映射顺序不影响多分类结果，只要训练和提交时一致即可。

清理完之后，我们还需要把训练集再切出一部分做验证。因为测试集没有标签，只有本地验证分数和排行榜的相关性高，我们才敢放心提交。

```python
from sklearn.model_selection import train_test_split

# 按目标分层切分，保证验证集里五类比例和训练集一致
train_df, val_df = train_test_split(
    df, test_size=0.2, shuffle=True, random_state=42,
    stratify=df['OpenStatus']
)

# 元特征：去掉目标列和文本列
train_meta = train_df.drop(['OpenStatus', 'QuestionText'], axis=1)
train_target = train_df['OpenStatus']
val_meta = val_df.drop(['OpenStatus', 'QuestionText'], axis=1)
val_target = val_df['OpenStatus']
```

这里 `stratify=df['OpenStatus']` 是一句很重要的代码：如果不分层，随机切分可能让验证集里某一类样本特别少，导致分数忽高忽低，无法判断模型到底进步没有。

### 2.3 模型训练与超参数调优

我们采用“文本 + 元特征”的双输入模型。文本走 LSTM，元特征直接拼接，最后一起过全连接层。这样做的原因是：标题正文提供语义，声望、时间、标签提供上下文，两条信息互补。

先做文本向量化。

```python
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

MAX_TOKENS = 10000
MAX_LEN = 150

# 把文本转成整数序列，最多保留 1 万个词，每条截断/补齐到 150 个词
text2vec = TextVectorization(max_tokens=MAX_TOKENS,
                             output_sequence_length=MAX_LEN)
text2vec.adapt(train_text)
```

`TextVectorization` 会统计词频，把最常见的词编号为 1、2、3……，罕见词统一归到保留编号。`MAX_LEN=150` 是权衡：太长会增加计算量，太短会丢掉正文后半段的信息。

然后搭模型。

```python
from tensorflow.keras.layers import Input, Embedding, LSTM, Concatenate, Dense
from tensorflow.keras import Model

# 文本输入：每条是一个字符串
text_input = Input(shape=(1,), dtype=tf.string, name='text_input')
x = text2vec(text_input)

# 词嵌入：每个词变成 50 维向量
x = Embedding(input_dim=MAX_TOKENS, output_dim=50)(x)

# LSTM 把词序列压成一个 128 维的语义向量
x = LSTM(128)(x)

# 元特征输入：58 维
meta_input = Input(shape=(train_meta.shape[1:]), name='meta_input')

# 拼接语义向量和元特征
y = Concatenate()([x, meta_input])

# 两层全连接，激活函数用 selu
y = Dense(256, activation='selu', kernel_initializer='lecun_normal')(y)
y = Dense(64, activation='selu')(y)

# 输出 5 类概率
y = Dense(5, activation='softmax')(y)

model = Model(inputs=[text_input, meta_input], outputs=y)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

这里用 `sparse_categorical_crossentropy` 而不是 `categorical_crossentropy`，因为标签是整数 0 到 4，不是 One-Hot 向量，前者省去了一步转换。

我们选择 LSTM 是因为问题文本不是无序的词袋：标题中的“why”“how”与正文中的代码片段顺序，会影响问题类型。Embedding 先把相似词映射到相近的向量位置，LSTM 再按顺序读取这些向量，最后得到一条摘要式的语义表示。元特征没有经过 LSTM，而是直接拼接，是因为声望、年份、小时等字段已经是单个数值，强行当成词序列反而会丢失它们的含义。

训练时加两个回调：学习率衰减和早停。

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 验证损失 5 轮不降，学习率减半
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, verbose=True)
# 验证损失 20 轮不降，停止训练并恢复最好的权重
early_stop = EarlyStopping(monitor='val_loss', patience=20,
                           restore_best_weights=True, verbose=True)

history = model.fit(
    x=[train_text, train_meta],
    y=train_target,
    batch_size=256,
    epochs=100,
    validation_data=([val_text, val_meta], val_target),
    callbacks=[reduce_lr, early_stop],
    shuffle=True
)
```

**关于调参。** 我们主要调整了三个量：`MAX_LEN`、LSTM 单元数、全连接层宽度。`MAX_LEN` 从 100 提到 150 有小幅提升，再往上收益不明显；LSTM 从 64 提到 128 提升明显，再大就开始过拟合；全连接层用 256 和 64 的组合比较稳。我们没有做系统的网格搜索，因为每个 epoch 要十几秒，全量搜索成本太高。

### 2.4 模型测试与指标报告

训练过程中，训练准确率一路升到 0.82，但验证准确率在第 4 到第 10 轮之间见顶，之后开始回落。这是典型的过拟合信号，早停帮我们回到了最好的那一轮。

```python
# 在验证集上评估
loss, acc = model.evaluate([val_text, val_meta], val_target)
print('val loss:', loss)
print('val accuracy:', acc)
```

输出约为 `loss 0.9909`、`accuracy 0.6196`。也就是说，模型能把约 62% 的验证样本分对，明显高于 20% 的随机水平。

这个数字没有继续往上走，原因有两个。一是数据本身噪声大：同样一段问题，不同管理员的判断都可能不同，人工标签存在上限。二是文本被截断到 150 个词，长问题后半段的代码和报错信息被丢掉了。我们试过把 `MAX_LEN` 加到 300，验证准确率只提高了不到一个百分点，但训练时间翻倍，性价比不高。

我们进一步看混淆矩阵，发现模型最容易混淆的是 `not constructive` 和 `off topic`，这两类的语义确实接近；`open` 类的召回率最高，因为它样本多、特征也更典型。

```python
from sklearn.metrics import classification_report
pred_probs = model.predict([val_text, val_meta])
pred_labels = pred_probs.argmax(axis=1)
print(classification_report(val_target, pred_labels))
```

最后对测试集做同样的处理并生成提交文件。

```python
test_pred = model.predict([test_text, test_meta]).argmax(axis=1)

submission = pd.DataFrame({
    'PostId': test['PostId'],
    'OpenStatus': test_pred
})
submission.to_csv('submission.csv', index=False)
```

## 三、总结

本文中我们完成了 StackOverflow 关闭问题预测。我们首先通过 EDA 了解类别分布、标签取值数量和时间跨度，并识别出 `PostClosedDate` 这个事后泄漏列。其次把发帖时间、注册时间拆成周期特征并计算账号年龄，把标题和正文合并，用二进制编码处理高基数标签。接着构建 LSTM 文本分支与元特征分支融合的神经网络，用早停和学习率衰减控制过拟合。最后在验证集上得到约 0.62 的准确率。这个比赛让我们体会到，文本任务里“文本 + 结构化特征”的双输入结构，往往比单纯把文本喂给模型更稳，而高基数类别用二进制编码比 One-Hot 更实用。
