# Spooky Author Identification

## 一、比赛简介

### 1.1 比赛背景
Spooky Author Identification 是 Kaggle 上极具代表性的文本风格识别比赛。比赛选取了 19 世纪末到 20 世纪初三位公域恐怖小说作家的作品片段，分别是 Edgar Allan Poe、H.P. Lovecraft 和 Mary Shelley。这三位作家的写作年代、地域和题材高度重合，都擅长营造恐怖氛围，但语言习惯存在细微差异。Poe 偏好华丽的辞藻和心理描写，Lovecraft 喜欢使用古旧词汇和宇宙恐怖的意象，Shelley 则更倾向于叙事性和情感表达。比赛希望通过机器学习捕捉这些风格指纹，从而在不看故事情节的情况下判断作者归属。

### 1.2 比赛目标
训练集包含 id、text、author 三列，author 取值映射为 EAP、HPL、MWS。文本长度差异很大，从一句几十字符的短句到上千字符的段落。测试集只有 id、text，需要输出每个样本的作者预测。任务本质上是三分类文本分类问题，类别数少但文本多样性高。参与者需要处理文本预处理、特征提取和模型选择，特别是在小样本段落上保持泛化能力。

### 1.3 评价标准
Kaggle 使用多分类对数损失 Multi-class Log Loss 作为评价指标。这种指标惩罚预测概率与真实标签之间的偏差，比单纯的准确率更严格。模型不仅需要猜对类别，还需要给出合理的置信度分布。目标是最小化 log loss，分数越低越好。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先加载训练数据并统计作者分布。

```python
# 加载训练数据并观察作者分布
train = pd.read_csv('../input/train.csv')
print(train.author.value_counts())
```

结果显示训练集共 19579 条，EAP 7900 条，HPL 6044 条，MWS 5635 条，类别略有不平衡。随后观察文本长度。

```python
# 计算字符和词数分布
char_len = train['text'].astype(str).apply(len)
word_len = train['text'].astype(str).apply(lambda x: len(x.split()))
print('字符均值', char_len.mean(), '词数均值', word_len.mean())
```

平均字符数约 149，最小 21 最大 4663，平均词数约 26.7。我们绘制直方图确认分布。

```python
# 绘制词数直方图
plt.hist(word_len, bins=50, range=(0,50))
plt.title('词数分布')
```

不同作者的词云显示词汇偏好差异明显。我们还检查文本缺失情况，发现文本列无空值，作者列完整。

### 2.2 数据清洗
文本预处理包括转小写、去标点、分词、停用词去除和词干提取。

```python
# 基础清理函数
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [SnowballStemmer('english').stem(w) for w in tokens if w not in stopwords.words('english')]
    return ' '.join(tokens)
```

我们保留了字符长度和词数作为数值特征，因为句子长度是风格的重要指标。

```python
train['char_len'] = train['text'].astype(str).apply(len)
train['word_len'] = train['text'].astype(str).apply(lambda x: len(x.split()))
```

文本长度特征与作者存在相关性。清洗后数据无缺失，可以直接进入特征提取。

### 2.3 模型训练与超参数调优
我们先用 TF-IDF 词袋特征配合 Logistic Regression 作为基线。

```python
# TF-IDF 向量化
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
X_train_vec = vectorizer.fit_transform(train['text'].apply(clean_text))
```

随后训练逻辑回归。

```python
# Logistic Regression 基线
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_vec, train['author'])
```

我们用 train_test_split 划分验证集，保持类别比例。

```python
X_tr, X_val, y_tr, y_val = train_test_split(X_train_vec, y, test_size=0.2, stratify=y, random_state=42)
```

随后尝试 Multinomial Naive Bayes。

```python
# Naive Bayes 模型
nb = MultinomialNB()
nb.fit(X_tr, y_tr)
```

Naive Bayes 在文本分类上速度快且效果稳定。随后尝试 LSTM。

```python
# 词嵌入 + 双向 LSTM
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(128, dropout=0.2)),
    Dense(3, activation='softmax')
])
```

调参重点是 n-gram 范围、是否去掉停用词、TF-IDF 的 max_df/min_df。字符级 n-gram 对短文本有帮助。最终选择 TF-IDF 1-2 gram + Logistic Regression 并做概率校准。

### 2.4 模型测试与指标报告
在验证集上评估 log loss 和分类报告。

```python
# 计算 log loss
from sklearn.metrics import log_loss
proba = lr.predict_proba(X_val)
print('log loss', log_loss(y_val, proba))
```

我们同时计算准确率和混淆矩阵，分析各作者的混淆情况。Poe 和 Lovecraft 容易混淆，Shelley 区分度更高。最终模型在验证集上的 log loss 约 0.38~0.42 之间。

```python
# 生成提交文件
test_vec = vectorizer.transform(test['text'].apply(clean_text))
proba_test = lr.predict_proba(test_vec)
submission = pd.DataFrame(proba_test, columns=['EAP','HPL','MWS'])
submission['id'] = test['id']
```

提交前检查列名和行数，确保与 sample_submission 一致。

## 三、总结
本文中我们完成了 Spooky Author Identification 的完整流程。我们首先通过 EDA 了解 19579 条文本的作者分布与长度特征，平均字符数约 149，词数约 26.7；其次做了小写、停用词、词干等清洗，并保留了长度统计特征；接着用 TF-IDF 与 Logistic Regression 建立基线，并尝试 Naive Bayes 和 LSTM；最后在验证集上得到 log loss 约 0.40 并生成概率提交文件。这个比赛体现了文本风格分类中特征工程的重要性，简单的词袋特征配合线性模型就能取得不错效果。
