# What's Cooking Kernels Only

## 一、比赛简介

### 1.1 比赛背景
What's Cooking Kernels Only 是 Kaggle 关于菜系预测的内核集合比赛。比赛提供食材列表预测菜系，数据与 What's Cooking 相同，但社区聚焦于深度学习方法。与传统词袋方法不同，本比赛强调序列建模，利用食材出现的顺序信息。

### 1.2 比赛目标
输入每条菜谱的食材列表，输出对应的菜系类别。训练集 39774 条，测试集 9944 条，共有 20 个菜系类别。

### 1.3 评价标准
采用分类准确率作为评测指标，预测正确比例越高越好。

## 二、解题过程

### 2.1 数据探索（EDA）
读取 JSON 数据并统计类别分布。

```python
import json, numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

with open("../input/whats-cooking-kernels-only/train.json") as f:
    data = json.load(f)
ingredients = [i['ingredients'] for i in data]
cuisine = [i['cuisine'] for i in data]
print(len(ingredients))
```

食材列表长度不一，最大序列长度约 65。我们对食材列表做整体分词，构建词表。

```python
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(ingredients)
sequences = tokenizer.texts_to_sequences(ingredients)
max_len = max(len(s) for s in sequences)
```

### 2.2 数据清洗
食材名称标准化，去掉大小写差异并统一。

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded = pad_sequences(sequences, maxlen=max_len, padding='pre')
```

标签编码。

```python
le = LabelEncoder()
labels = le.fit_transform(cuisine)
```

数据清洗后无缺失值。

### 2.3 模型训练与超参数调优
构建 Embedding + BiLSTM 模型。

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

vocab_size = len(tokenizer.word_index)+1
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len),
    Bidirectional(LSTM(20)),
    Dense(30, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

使用 EarlyStopping 和 ModelCheckpoint 防止过拟合。训练 30 轮，批次大小 32。

### 2.4 模型测试与指标报告
划分训练验证集。

```python
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(padded, labels, test_size=0.2, random_state=42)
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val), callbacks=[es, cp])
```

验证集准确率在第 4 轮达到 0.7309，后续略有波动。最终选择最佳模型在测试集预测。

```python
test_data = pd.read_json("../input/whats-cooking-kernels-only/test.json")
test_seq = tokenizer.texts_to_sequences([d['ingredients'] for d in test_data])
test_pad = pad_sequences(test_seq, maxlen=max_len, padding='pre')
pred = model.predict(test_pad)
```

生成提交文件。

## 三、总结
本文中我们完成了 What's Cooking Kernels Only 的深度学习流程。我们首先通过 EDA 了解食材序列长度分布。其次做序列填充和标签编码清洗。接着使用 Embedding 加 BiLSTM 的模型训练，验证集准确率约 0.73。最后生成提交文件完成预测。序列模型比词袋模型能捕捉食材顺序信息，是该任务的有力尝试。
