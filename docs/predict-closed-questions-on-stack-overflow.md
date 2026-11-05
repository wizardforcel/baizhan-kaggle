# 预测 StackOverflow 关闭问题

## 一、比赛简介

### 1.1 比赛背景
StackOverflow 是程序员常去的问答社区，站长会把不符合规范的问题关闭。关闭原因有“不是真实问题”、“不具建设性”、“离题”、“已关闭”、“过于本地化”等。比赛期望用帖子的元信息和正文特征预测问题最终状态。

### 1.2 比赛目标
输入包含帖子创建时间、作者注册时间、声望、回答数、标题、正文、标签等字段。目标列 `OpenStatus` 是一个字符串，取值有 open、not a real question、not constructive、off topic、too localized 五类。训练集和测试集结构一致，需要预测测试集的类别标签。

### 1.3 评价标准
比赛以多分类准确率或对数似然为核心指标，模型需要区分 5 个类别。准确率越高越好。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先读入训练样本，查看列名和样本分布。

```python
import pandas as pd
df = pd.read_csv("../input/predict-closed-questions-on-stack-overflow/train-sample.csv")
df.head()
```

列包括 PostId、PostCreationDate、OwnerUserId、OwnerCreationDate、ReputationAtPostCreation、OwnerUndeletedAnswerCountAtPostTime、Title、BodyMarkdown、Tag1~Tag5、PostClosedDate、OpenStatus。OpenStatus 包含 open 和四种关闭原因。

```python
# 类别分布
df['OpenStatus'].value_counts()
```

open 类最多，其次是 not constructive，大家分布不均，需要分层抽样。

### 2.2 数据清洗
我们把时间特征拆成年、月、日、小时、分钟计算作者年龄和发帖时段。标题和正文合并成 QuestionText。

```python
import pandas as pd
import numpy as np
def datetime_features(df):
    df["PostDate"] = pd.to_datetime(df["PostDate"])
    df["OwnrDate"] = pd.to_datetime(df["OwnrDate"])
    df["PostDay"] = df["PostDate"].dt.dayofweek
    df["PostMonth"] = df["PostDate"].dt.month
    df["AccAge"] = (df["PostDate"] - df["OwnrDate"])/np.timedelta64(1, 's')
    return df
```

标签特征取值太多，直接 one-hot 会爆炸，我们用二进制编码压缩。

```python
import category_encoders as ce
encoder = ce.binary.BinaryEncoder()
df_encoded = encoder.fit_transform(df[['Tag1']])
```

目标映射为 0~4。

```python
target_map = {"not a real question":0,"not constructive":1,"off topic":2,"open":3,"too localized":4}
df["OpenStatus"] = df["OpenStatus"].map(target_map)
```

### 2.3 模型训练与超参数调优
我们采用文本+元特征的深度模型。文本用 TextVectorization 转成序列，Embedding+ LSTM 提取语义，元特征直接拼接。

```python
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Input, Concatenate, Dense
from tensorflow.keras import Model
MAX_TOKENS = 10000
MAX_LEN = 150
EMBED_DIM = 50

text_input = Input(shape=(1,), dtype='string')
x = TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=MAX_LEN)(text_input)
x = Embedding(input_dim=MAX_TOKENS, output_dim=EMBED_DIM)(x)
x = LSTM(128)(x)

meta_input = Input(shape=(58,))
y = Concatenate()([x, meta_input])
y = Dense(256, activation='selu', kernel_initializer='lecun_normal')(y)
y = Dense(64, activation='selu')(y)
y = Dense(5, activation='softmax')(y)

model = Model(inputs=[text_input, meta_input], outputs=y)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

训练使用早停和学习率衰减。

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
             EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]

history = model.fit([train_text, train_meta], train_target, batch_size=256, epochs=100,
                    validation_data=([val_text, val_meta], val_target), callbacks=callbacks)
```

### 2.4 模型测试与指标报告
在验证集上最好的验证准确率约为 0.62，训练准确率随 epoch 上升。

```python
loss, acc = model.evaluate([val_text, val_meta], val_target)
print(acc)  # 0.6196
```

说明文本特征对区分关闭原因有效，但仍有提升空间。

## 三、总结
本文中我们完成了 StackOverflow 关闭问题预测。我们首先做时间特征工程并合并文本。其次对标签做映射并用二进制编码处理高基数标签。接着构建 LSTM 文本模型与元特征融合，验证准确率约 0.62。最后生成提交文件。比赛提示我们，文本与数值特征结合比单用文本更稳健，时间窗口特征也很关键。
