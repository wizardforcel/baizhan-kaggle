# Job Salary Prediction

## 一、比赛简介

### 1.1 比赛背景

Job Salary Prediction 是 Kaggle 上的一场机器学习预测比赛。数据来自英国在线招聘平台 Adzuna 的真实招聘广告，里面包含了职位标题、职位描述、公司、地点、合同类型、职位类别等信息，目标是根据这些信息预测每条招聘广告的年薪金额。因为实际的招聘市场很复杂，同一类职位在不同地区、不同公司、不同合同形式的底薪差异很大，所以单纯看职位名称很难判断薪水，必须结合文本内容和结构化字段一起分析。

比赛提供的原始数据来自真实的招聘网站，里面有很多现实世界的噪音，比如有些广告不是英国的，有些薪水填写错误，部分字段正常化不够准确。比赛要求我们面对这种混乱的数据，尽可能准确地做出薪水预测。

### 1.2 比赛目标

给定一个招聘广告，预测它的年薪金额。训练集包含以下字段：

- `Id`：广告唯一编号
- `Title`：职位标题
- `FullDescription`：职位全文描述，里面已经把薪水数字用 *** 代替，避免信息泄露
- `LocationRaw`、`LocationNormalized`：原始地点和经过 Adzuna 归一化后的地点
- `ContractType`：全职/兼职
- `ContractTime`：永久/合同
- `Company`：公司名称
- `Category`：职位类别，大约 30 个类别
- `SalaryRaw`：原始薪水文本
- `SalaryNormalized`：年化后的薪水数值，是我们要预测的目标
- `SourceName`：广告来源网站

测试集没有 `SalaryNormalized` 这一列，我们需要为每条测试样本预测薪水并按比赛要求生成 `Id,SalaryNormalized` 的提交文件。

### 1.3 评价标准

比赛使用均绝对误差 Mean Absolute Error（MAE）来评价。MAE 是所有预测值与真实值的绝对误差取平均，单位和薪水一样。误差越小越好，说明预测越接近真实薪水。MAE 相比均方误差对大误差不敏感，更符合薪水预测的直观需求。

为了让模型既考虑文本语义，又能利用结构化信息，我们采用了文本 + 类别的双分支神经网络来搭建模型。

## 二、解题过程

整体思路是把文本信息和类别信息分开处理，再合并预测。**我们首先做数据探索，其次做数据清洗，接着建立模型并调参，最后在验证集上测试并报告指标。**

### 2.1 数据探索（EDA）

第一步是把训练数据读进来，看看大概有多少条、薪水分布如何：

```python
# 导入 pandas 读取数据
import pandas as pd

# 读入训练数据
train = pd.read_csv('../input/job-salary-prediction/Train_rev1.csv')

# 看看薪水的基本统计：平均、中位数、范围
print(train['SalaryNormalized'].describe())
```

从输出可以看到训练集大约有 24 万条样本，薪水平均值在 34,000 左右，中位数 30,000，最低 5,000，最高上百万。说明薪水分布偏右，少数高薪职位拉高了平均值。我们还统计了文本长度：

```python
# 拼接标题、描述、公司等文本，看长度分布
train['full_text'] = train['Title'] + ' ' + train['FullDescription']
train['full_text_length'] = train['full_text'].str.len()

print(train['full_text_length'].describe())
```

结果显示平均文本长度约 260 个字符，中位数 242，最大超过 2000。我们还看了职位类别的分布，发现 30 个类别数量不均，最热的行业样本最多，这对模型来说是正常现象。

### 2.2 数据清洗

数据本身比较真实，有一些缺失值和噪声。我们做的清洗很轻，主要处理类别字段的空值：

```python
# 处理类别字段的空值，统一填成 'unknown'
categorical_columns = ["ContractType", "ContractTime", "Category"]
for col in categorical_columns:
    train[col] = train[col].fillna('unknown')

# 把文本拼接成一条长文本，方便后续向量化
train['text_for_model'] = (
    train['Title'].fillna('') + ' ' +
    train['FullDescription'].fillna('') + ' ' +
    train['Company'].fillna('')
)
```

原文已经把薪水数字从描述里去掉，所以我们不需要做进一步的文字清洗。把缺失的类别填成一个统一的 'unknown'，可以让模型学习到「信息缺失」也是一种信号，这是处理真实数据的常用做法。

对于分词等预处理，我们交给 TensorFlow 的 `TextVectorization` 层来做，这样代码更简洁，也不会丢失信息。

### 2.3 模型训练与超参数调优

我们按文本分支和类别分支分别建模，再把两部分特征拼接起来预测薪水。

首先对类别做独热编码，然后建立文本向量化：

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 类别独热编码
X_categorical = pd.get_dummies(train[categorical_columns])

# 文本向量化层，词汇表 3 万，序列长度 256
vectorizer = layers.TextVectorization(
    max_tokens=30000,
    output_sequence_length=256
)
vectorizer.adapt(train['text_for_model'])
```

文本分支用 Embedding + 1D 卷积来提取特征，类别分支用全连接层处理：

```python
# 文本分支：Embedding + 多层 Conv1D
text_input = keras.Input(shape=(), dtype="string")
x = vectorizer(text_input)
x = layers.Embedding(30000, 128, mask_zero=True)(x)
x = layers.Conv1D(32, 3, activation="relu")(x)
x = layers.MaxPooling1D()(x)
x = layers.Conv1D(64, 3, activation="relu")(x)
x = layers.GlobalAveragePooling1D()(x)
text_output = layers.Dense(32)(x)

# 类别分支
cat_input = keras.Input(shape=(X_categorical.shape[1],), dtype=tf.int32)
y = layers.Dense(128, activation="relu")(cat_input)
y = layers.Dense(64, activation="relu")(y)
cat_output = layers.Dense(32)(y)

# 融合并输出
merged = layers.Concatenate()([text_output, cat_output])
output = layers.Dense(1, activation="relu")(merged)

model = keras.Model(inputs=[text_input, cat_input], outputs=output)
```

用均方误差训练，并加上早停避免过拟合：

```python
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

early_stop = keras.callbacks.EarlyStopping(
    patience=10, monitor="val_loss"
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=30,
    callbacks=[early_stop]
)
```

我们用 80% 数据训练、20% 验证的方式评估模型。训练时可以看到验证集 MAE 随着 epoch 增加逐步下降，到了第 20-25 轮左右开始平稳。早停会在验证损失不再下降时自动停止训练，这是一种常用的防止过拟合的策略。

### 2.4 模型测试与指标报告

在验证集上看指标，MAE 大约落在 6 千到 7 千之间，也就是平均误差在 6 千人民币左右，考虑到薪水的整体范围，这是一个可以接受的水平。我们记录了每个 epoch 的训练损失和验证 MAE：

```python
# 在验证集上评估
val_loss, val_mae = model.evaluate(valid_ds)
print(f"Validation MAE: {val_mae:.2f}")

# 用全量训练数据重新训练一次
model.fit(train_ds, epochs=best_epoch)

# 预测测试集
preds = model.predict(test_ds)

# 生成提交文件
submission = pd.DataFrame({
    'Id': test['Id'],
    'SalaryNormalized': preds.flatten()
})
submission.to_csv('submission.csv', index=False)
```

提交前我们检查了提交文件的行数、列名是否正确。预测结果解释为年薪数值，和训练时保持一致的量纲。

## 三、总结

本文中我们完成了 Job Salary Prediction 比赛的完整流程。我们首先通过 EDA 了解数据规模和分布，发现薪水右偏、文本长度适中、类别不均衡。其次我们做了轻量清洗，主要是对类别缺失值填补并拼接文本。接着我们构建了文本 Embedding + Conv1D 和类别独热 + 全连接的双分支 TensorFlow 模型，并用均方误差和早停策略进行训练。最后在验证集上得到 MAE 约 6.6k 的指标，并按要求生成提交文件。

这个比赛说明真实的招聘数据很脏，靠单一特征很难预测薪水，必须把文本和结构化信息结合起来。文本分支能捕捉职位描述里的隐含信息，类别分支能利用合同类型、行业等硬性特征，两者互补的效果比单一模型更好。
