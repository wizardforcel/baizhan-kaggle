# Transfer Learning on Stack Exchange Tags

## 一、比赛简介

### 1.1 比赛背景

Stack Exchange 是一个由很多问答社区组成的网站。每个问题通常有一个标题、一段正文和若干标签，标签可以帮助读者快速判断问题属于哪个知识方向。这个比赛的特别之处在于：我们能看到六个领域的带标签问题，却要把知识迁移到另一个没有标签的领域。材料中的六个训练领域是 biology、cooking、crypto、diy、robotics 和 travel；测试集是 Physics 领域的问题。也就是说，训练数据和测试数据不是同一个分布，这违反了普通机器学习“训练和测试看起来相似”的常见假设。我们不能简单记住某个领域的词，而要尽量学习问题文字与标签之间更一般的关系。

```python
# 先列出比赛文件，确认六个有标签的来源领域和一个测试文件
import os
import pandas as pd

# 这些文件名来自比赛材料，名称本身也帮助我们理解任务结构
print(os.listdir('../input/transfer-learning-on-stack-exchange-tags'))

# 六个训练语料各自有 tags，测试语料没有 tags
biology = pd.read_csv('../input/transfer-learning-on-stack-exchange-tags/biology.csv')
cooking = pd.read_csv('../input/transfer-learning-on-stack-exchange-tags/cooking.csv')
crypto = pd.read_csv('../input/transfer-learning-on-stack-exchange-tags/crypto.csv')
diy = pd.read_csv('../input/transfer-learning-on-stack-exchange-tags/diy.csv')
robotics = pd.read_csv('../input/transfer-learning-on-stack-exchange-tags/robotics.csv')
travel = pd.read_csv('../input/transfer-learning-on-stack-exchange-tags/travel.csv')
test = pd.read_csv('../input/transfer-learning-on-stack-exchange-tags/test.csv')
```

我们首先读取六个训练文件和测试文件，并确认训练行包含 `id`、`title`、`content`、`tags` 四类信息。测试文件保留 `id`、`title`、`content`，需要我们预测 `tags`。材料还给出了一个直观例子：biology 的问题会带有 `ribosome`、`translation` 等标签，而测试问题会出现粒子、弦论、蒙特卡洛等 Physics 主题。领域迁移的难点就在这里：测试领域的专有词可能没有在训练领域中直接出现。

### 1.2 比赛目标

比赛目标是给 Physics 测试问题生成一个或多个标签。一个问题可以同时属于多个主题，因此这不是只能选择一个类别的普通分类，而是**多标签分类**：同一行可以对应一组标签。训练数据的 `title` 是问题标题，`content` 是带 HTML 标记的正文，`tags` 是用空格分隔的标签字符串。我们要从标题和正文中提取文字特征，再输出与测试 `id` 对应的标签字符串。

```python
# 把六个来源表放进列表，便于使用同一套流程检查列名
sources = [biology, cooking, crypto, diy, robotics, travel]
for name, frame in zip(['biology', 'cooking', 'crypto', 'diy', 'robotics', 'travel'], sources):
    # 每个训练表都应当包含标签，否则无法做监督学习
    print(name, frame.columns.tolist())

# 测试表用于最终预测，所以只检查输入列，不访问不存在的 tags
print('test columns:', test.columns.tolist())
print('one training row:', sources[0].loc[0, ['title', 'content', 'tags']].to_dict())
```

我们把任务理解成“由问题文字预测标签集合”。材料统计了 87000 个问题、225138 个标签使用记录和 4268 个不同标签；每个问题平均有 2.59 个标签，最少 1 个、最多 5 个，标准差是 1.20。这些数字说明输出不是单一词，而是一组数量不固定的标签。我们还要注意，题目给出的标签集合很大，如果把全部标签都直接作为模型目标，训练成本会明显增加。

### 1.3 评价标准

材料中的方案使用 Micro F1 作为主要报告指标。F1 可以把“预测出的标签有多少正确”和“真实标签有多少被找回”合并起来；Micro 的计算方式是先把所有问题、所有标签的正确与错误累积，再计算总体的精确率和召回率。它适合多标签问题，因为一个问题可能有多个正确答案。分数越高越好。材料记录的训练集 Micro F1 为 0.50，验证集为 0.48，而提交到 Kaggle 后的测试集 Micro F1 为 0.011；这也反映了跨领域迁移的困难。

```python
# 多标签任务不能只看单个类别准确率，所以同时报告 Micro 和 Macro F1
from sklearn.metrics import accuracy_score, f1_score, classification_report

# y_true 和 y_pred 都是“行=问题、列=标签”的 0/1 矩阵
print('accuracy:', accuracy_score(y_true, y_pred))
print('micro f1:', f1_score(y_true, y_pred, average='micro'))
print('macro f1:', f1_score(y_true, y_pred, average='macro'))
print(classification_report(y_true, y_pred))
```

我们在本地把训练集切成训练部分和验证部分，用验证集的 Micro F1 比较模型。这里的验证分数只能说明模型在六个已知领域中的表现，不能保证在 Physics 上同样好；最终测试分数由比赛平台计算。这样报告指标时，我们既不把本地验证结果冒充排行榜结果，也能看到迁移造成的落差。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把六个训练表纵向合并，并把领域名称保存到新列 `category`。这样做的原因是：后面既要把所有领域放在一起训练，也要按领域观察样本量和标签关系。探索时我们查看每个领域的行数、每行标签数量、标签总数和跨领域共享情况。

```python
# 只读取 EDA 需要的列，减少无关内容干扰
import itertools
import collections
import numpy as np

categories = ['cooking', 'robotics', 'travel', 'crypto', 'diy', 'biology']
frames = []
for category in categories:
    frame = pd.read_csv(f'../input/transfer-learning-on-stack-exchange-tags/{category}.csv',
                        usecols=['id', 'title', 'content', 'tags'])
    frame['category'] = category  # 保留来源领域，方便比较分布
    frames.append(frame)

combined_data = pd.concat(frames, ignore_index=True)
tag_lists = [text.strip().split() for text in combined_data['tags']]
all_tags = list(itertools.chain(*tag_lists))
print(len(combined_data), len(all_tags), len(set(all_tags)))
print(np.mean([len(x) for x in tag_lists]), min(map(len, tag_lists)), max(map(len, tag_lists)))
```

我们从材料得到的整体统计是 87000 个问题、4268 个独立标签，平均每题 2.59 个标签。标签数量分布也不均匀：约 30% 的问题有 2 个标签，只有 8% 的问题有 5 个标签。按领域看，DIY 大约占 30%，Robotics 约占 3.2%，因此 Robotics 的训练样本相对少；但它在语言和词汇上可能更接近 Physics，样本少反而会成为迁移瓶颈。我们还检查了标签是否跨领域共享，结果显示 4148 个标签只出现在 1 个领域，101 个出现在 2 个领域，只有少数标签出现在更多领域。这说明直接依赖跨领域同名标签并不充分，需要从文字模式中学习。

```python
# 统计每个标签出现在哪些领域，判断迁移时有没有共享词汇
shared = pd.DataFrame(index=set(all_tags))
for category, frame in combined_data.groupby('category'):
    counts = collections.Counter(itertools.chain.from_iterable(
        row.strip().split() for row in frame['tags']
    ))
    shared[category] = pd.Series(counts)

# 一个标签在多少个领域出现过；数值越大表示跨领域共享越明显
shared['categories_appears'] = shared.fillna(0).astype(bool).sum(axis=1)
print(shared['categories_appears'].value_counts())
```

我们还统计了标签与标题、正文之间的重合。这个检查能帮助我们判断标签是否常常直接出现在问题文字里：如果重合较高，文字模型更容易学；如果重合很低，就需要依靠上下文。EDA 的重点不是画图本身，而是先弄清样本量、标签长短、领域不平衡和标签重叠，再决定清洗和建模方案。

### 2.2 数据清洗

我们其次处理 HTML、空格和重复标题。原始 `content` 中有 `<p>`、`<em>` 等 HTML 标记，若直接分词，标记名可能被当成普通词，影响 TF-IDF。材料采用正则表达式删除 HTML 标签，并对标题和正文做首尾空格清理；之后按 `title` 去重，避免相同问题在训练中重复出现。

```python
import re

# HTML 标签只是网页格式，不是问题语义，所以先移除它们
 def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_space(text):
    # 首尾空格不会提供主题信息，去掉后能避免产生无意义差异
    return text.strip()

for frame in frames:
    frame['title'] = frame['title'].apply(remove_html_tags).apply(remove_space)
    frame['content'] = frame['content'].apply(remove_html_tags).apply(remove_space)

combined_data = pd.concat(frames, ignore_index=True)
combined_data = combined_data.drop_duplicates(subset=['title'])
combined_data.reset_index(drop=True, inplace=True)
```

我们清洗后把标题和正文分别分词，再去标点、转小写、删除停用词，最后使用 SnowballStemmer 做词干化。停用词是“the”“and”这类在很多问题中都出现、区分度低的词；词干化则把词形压缩到相近的形式。材料还特别加入了一批额外停用词，以减少语料中的常见噪声。标签本身不删除，因为它们是监督学习的目标；测试集也必须使用同样的文字清洗流程。

```python
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
stop_words = set(['a', 'an', 'the', 'and', 'or', 'is', 'are', 'to', 'of', 'in'])

def clean_text(text):
    # 先切成词，再保留字母词，避免标点成为特征
    words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    # 词干化可以把相近词形压缩，减少特征数量
    return ' '.join(stemmer.stem(word) for word in words)

combined_data['title_clean'] = combined_data['title'].apply(clean_text)
combined_data['content_clean'] = combined_data['content'].apply(clean_text)
combined_data['text'] = combined_data['title_clean'] + ' ' + combined_data['content_clean']
```

我们没有把清洗理解成越多越好，而是保持训练和测试完全一致。清洗后的 `text` 由标题和正文合并而成，作为后续 TF-IDF 的输入。对于多标签目标，我们把标签按空格拆开，统计每个标签出现次数，并按频率排序。材料中的取舍是只保留最高频的 500 个标签：它们覆盖 86968 个问题中的 78276 个，约占 90%；这样可以控制计算量，同时保留绝大多数问题的标签信息。没有命中这 500 个标签的问题在这条基线中被移除。

```python
from sklearn.feature_extraction.text import CountVectorizer

# binary=True 表示只关心标签是否出现，不重复累计同一题中的次数
label_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '), binary=True)
label_matrix = label_vectorizer.fit_transform(combined_data['tags'])
counts = label_matrix.sum(axis=0).A1
sorted_tags = np.argsort(-counts)

# 只取最高频的 500 个标签，降低 OneVsRest 需要训练的分类器数量
top_indices = sorted_tags[:500]
top_labels = [label_vectorizer.get_feature_names_out()[i] for i in top_indices]
target = label_matrix[:, top_indices]
```

### 2.3 模型训练与超参数调优

我们接着用分层思路准备多标签目标和文字特征。材料中的最终基线采用 `TfidfVectorizer`，最多保留 10000 个特征，使用 1 到 3 元词组。TF-IDF 可以理解为给关键词加权：在某个问题中出现、但不是所有问题都出现的词，权重会更高；常见的无区分度词权重会降低。1-gram、2-gram、3-gram 同时使用，既看单词，也看短语。

```python
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 先切分再 fit，避免验证集的词频信息泄漏到训练过程
x_train, x_valid, y_train, y_valid = train_test_split(
    combined_data, sparse.csr_matrix(target), test_size=0.20, random_state=42
)

vectorizer = TfidfVectorizer(
    max_features=10000,                 # 控制稀疏矩阵规模，材料中的基线参数
    ngram_range=(1, 3),                 # 同时学习单词、二元词组和三元词组
    tokenizer=lambda x: x.split(' ')
)
TrainData = vectorizer.fit_transform(x_train['text'])
ValidData = vectorizer.transform(x_valid['text'])
```

我们然后使用 `OneVsRestClassifier`。它的直观做法是：对每个标签单独训练一个“有这个标签/没有这个标签”的二分类器，最后把所有分类器的结果拼成标签集合。材料给出的主模型是逻辑回归，参数为 `C=0.9`、`penalty='l1'`、`solver='saga'`，并设置 `n_jobs=-1` 并行处理标签。`C` 控制正则化强度，L1 正则会把一部分不重要的特征压成零，适合高维稀疏文本。

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# OneVsRest 把多标签任务拆成多个独立的二分类任务
classifier = OneVsRestClassifier(
    LogisticRegression(
        C=0.9,             # 材料中的正则化参数，控制模型复杂度
        penalty='l1',      # 让部分无用词特征的权重变成零
        solver='saga'      # 支持稀疏输入和 L1 正则
    ),
    n_jobs=-1             # 标签分类器可以并行训练，缩短等待时间
)
classifier.fit(TrainData, y_train)
predictions = classifier.predict(ValidData)
```

我们把超参数调优理解成在有限计算量下做合理取舍。材料中保留了 `max_features=10000`、`ngram_range=(1,3)`、逻辑回归 `C=0.9` 的主方案；同时还试过 `OneVsRestClassifier(MultinomialNB(alpha=0.35))` 作为朴素贝叶斯对照。标签数从全部 4268 个缩减到 500 个，也是重要的资源型超参数。由于训练域和测试域差异极大，继续把模型做得更复杂并不能保证 Physics 测试集提升，因此我们优先保持流程清楚、特征和参数可复现。

### 2.4 模型测试与指标报告

我们最后在验证集上计算准确率、Micro F1、Macro F1 和分类报告。对多标签矩阵而言，准确率只有在整行标签都完全正确时才算正确，通常比较严格；Micro F1 更适合观察所有标签决策的总体效果，Macro F1 则会让每个标签贡献更接近，能暴露少数标签表现不佳的问题。

```python
# 在验证集上同时评估多标签预测，避免只看一个数字
from sklearn.metrics import accuracy_score, f1_score, classification_report

prediction_train = classifier.predict(TrainData)
prediction_valid = classifier.predict(ValidData)

print('Train Accuracy:', accuracy_score(y_train, prediction_train))
print('Train Micro F1:', f1_score(y_train, prediction_train, average='micro'))
print('Valid Accuracy:', accuracy_score(y_valid, prediction_valid))
print('Valid Micro F1:', f1_score(y_valid, prediction_valid, average='micro'))
print('Valid Macro F1:', f1_score(y_valid, prediction_valid, average='macro'))
print(classification_report(y_valid, prediction_valid))
```

材料记录的结果是训练集 Micro F1 为 0.50，验证集 Micro F1 为 0.48。这个结果说明模型在六个训练领域内部已经学到了一些有效的文字—标签关系，但不能直接推断它能准确处理 Physics。我们把完整训练流程重新用于测试文本，并使用训练阶段已经拟合好的 TF-IDF 向量器进行 `transform`，不能在测试集上重新 `fit`，否则词表会发生变化。

```python
# 测试集只能使用训练集已经确定的清洗和词表
 test['title'] = test['title'].apply(remove_html_tags).apply(remove_space)
test['content'] = test['content'].apply(remove_html_tags).apply(remove_space)
test['text'] = test['title'].apply(clean_text) + ' ' + test['content'].apply(clean_text)

test_features = vectorizer.transform(test['text'])
test_predictions = classifier.predict(test_features)

# 将 0/1 标签矩阵还原成空格分隔的标签字符串
result = []
for row in test_predictions:
    result.append(' '.join(label for label, flag in zip(top_labels, row) if flag))
submission = pd.DataFrame({'id': test['id'], 'tags': result})
submission.to_csv('submission.csv', index=False)
```

我们还参考材料中的概率后处理：先用 `predict_proba` 得到每个标签的概率，再用 0.03 作为阈值决定是否输出。阈值越低，输出的标签越多，召回可能上升但误报也会增加；阈值越高则相反。最终材料记录 Kaggle 测试集 Micro F1 为 0.011，远低于本地验证集 0.48。我们把主要原因解释为领域和词汇分布完全不同，而不是把低分归因于代码运行错误。这个对比也正是本比赛“transfer learning”主题的关键实验结果。

## 三、总结

本文中我们完成了 Stack Exchange 标签迁移比赛的一套可复现基线。我们首先读取 biology、cooking、crypto、diy、robotics、travel 六个训练领域，并认识到测试集来自 Physics，训练和测试分布不同。其次我们做了 EDA：合并 87000 个问题，统计 4268 个独立标签、平均每题 2.59 个标签，并观察到 DIY 样本约占 30%、Robotics 约占 3.2%，大部分标签只在一个领域出现。接着我们删除 HTML 标签、清理空格、按标题去重，把标题和正文合并，分词、去停用词、转小写并词干化。然后我们把高频标签缩减到 500 个，用 1 到 3 元词组的 TF-IDF 表示文字，再用 OneVsRest 逻辑回归完成多标签训练；主要参数是 `max_features=10000`、`ngram_range=(1,3)`、`C=0.9`、L1 正则和 `saga` 求解器。最后我们用 Accuracy、Micro F1、Macro F1 和分类报告评估，并生成 `id,tags` 格式的提交文件。

材料中的训练 Micro F1 为 0.50，验证 Micro F1 为 0.48，但测试提交 Micro F1 只有 0.011。这个落差提醒我们：在同一分布内表现不错的 TF-IDF 模型，面对新的 Physics 领域时可能无法识别新词和新概念。后续如果要继续提升，方向应当围绕跨领域词汇迁移、领域相似度、标签语义和更可靠的阈值选择，而不是只盯着训练集分数。