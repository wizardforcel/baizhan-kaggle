# Random Acts of Pizza（随机披萨行动）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的名字叫 Random Acts of Pizza（随机披萨行动），数据来自 Reddit 上的一个叫 `r/Random_Acts_Of_Pizza` 的小社区。这个社区的活动很简单：如果你饿了、穷了、遇到了困难，可以发一个帖子哭诉一下自己的处境，请求别人给自己买一份披萨；社区里的好心人看到后，可能会回答你「告诉我你的地址」，然后真的下单给你送一份披萨。

比赛就是从这个社区的真实历史帖子中整理出来的：主办方抓取了几千条「求披萨」的请求记录，每条记录记录了这个帖子当时的标题、正文，以及发帖人的各种资料（比如账号注册多久了、发过多少帖子、收到过多少个赞），再标注上这位发帖人**最终到底有没有收到披萨**（`requester_received_pizza` 这个字段）。我们参赛者的任务，就是根据这些历史数据学会预测：一个新帖子发出来，这个人有多大几率真的能收到披萨。

这个任务有几个天然的难点。**首先，数据严重不平衡**：真正能收到披萨的请求是少数，大部分请求最后都石沉大海，所以即便我们瞎猜「全都不会收到披萨」，准确率也有七成以上，但那样并没有真正学会任何东西。**其次，有用的信息很杂**：除了帖子文字这种「软信息」，还有一堆发帖人的行为特征。**最后，要注意数据泄漏**：训练集里有不少字段是「发帖之后」才产生的，比如帖子的最终点赞数、发帖人后来的发帖数量——这些信息在预测的时候根本拿不到，必须特别小心。

### 1.2 比赛目标

主办方给了两个 JSON 文件：`train.json` 和 `test.json`。我们用工具读出来后，训练集有 **4040 行、32 列**，测试集有 **1631 行、17 列**。每一行是一次请求（request），对应的预测对象就是 32 列里的标签列 `requester_received_pizza`，它是一个布尔值：`True` 表示收到了披萨，`False` 表示没有。

特征大致可以分成两类：

- **文本特征**：`request_text`（请求正文）、`request_text_edit_aware`（考虑编辑过的正文）、`request_title`（请求标题）。
- **行为特征**：`requester_account_age_in_days_at_request`（发帖时账号已注册天数）、`requester_number_of_posts_at_request`（发帖时的总发帖数）、`requester_number_of_comments_at_request`、`requester_upvotes_plus_downvotes_at_request`（正的点赞数）等。注意很多字段都带 `at_request` 或 `at_retrieval` 后缀，前者是「发帖那一刻」的值，后者是「后来抓数据时」的值，后者在真实预测时是拿不到的。

我们要用训练集里这些特征训练一个二分类模型，然后对测试集的每一条记录预测它「收到披萨」的概率。

### 1.3 评价标准

比赛用的是 **ROC AUC（也叫 AUC，ROC 曲线下的面积）** 作为评价指标。大白话解释一下：AUC 衡量的是「模型给正例（真收到披萨的人）打的分，比给负例（没收到披萨的人）打的分离」的概率。AUC = 0.5 相当于瞎猜，AUC = 1.0 是完美排序。**这个指标越高越好**。正因为用 AUC 而不是准确率，所以就算正例很少（只有约四分之一），我们也不会被「全猜负例」的高准确率骗到。

---

## 二、解题过程

我们整体思路是：**把「文本信息」和「行为信息」都转成特征，拼接起来丢给逻辑回归模型，并用网格搜索调超参数。** 下面按四个步骤来讲。

### 2.1 数据探索（EDA）

我们首先读入两个 JSON 文件，`json_normalize` 的作用是把嵌套的 JSON 展开成一行一行的表格：

```python
import json
import pandas as pd
from pandas.io.json import json_normalize

# JSON 里是嵌套结构，用 json_normalize 展开成表格，每一行是一条请求
train = json_normalize(json.load(open('../input/train.json')))
test = json_normalize(json.load(open('../input/test.json')))
print("Train :", train.shape[0], "rows", train.shape[1], "columns")
print("Test :", test.shape[0], "rows", test.shape[1], "columns")
```

打印出来和我们预期的一样：`Train: 4040 rows 32 columns`，`Test: 1631 rows 17 columns`。我们接着比较两边的列，搞清楚测试集里缺了哪些信息：

```python
# 训练集有而测试集没有的列，基本都是 "at_retrieval"（事后抓取）的字段和标签
in_train_not_in_test = set(train.columns) - set(test.columns)
print(in_train_not_in_test)
```

打印结果里，训练集独有的一批列除了标签 `requester_received_pizza` 和正文 `request_text` 之外，都是形如 `number_of_upvotes_of_request_at_retrieval`、`requester_number_of_posts_at_retrieval` 这样的**事后字段**。这提醒我们：真正能用来预测的只有 17 个共享列，事后字段必须全部丢掉，否则就是数据泄漏——测试集上根本没有这些信息，用了就等于作弊。

我们再看看成功率的分布。把训练数据按 8:2 切成开发集（dev）和验证集（valid），数一数正例比例：

```python
from sklearn.model_selection import train_test_split

# 只保留训练集和测试集都有的列，否则模型会用到测试时拿不到的信息
shared = list(set(train.columns) & set(test.columns)) + ['requester_received_pizza']
train = train[shared]

# 8 成训练、2 成验证
dev_data, valid_data, dev_labels, valid_labels = train_test_split(
    train, train['requester_received_pizza'], test_size=0.2)

# 看看正例比例
success = dev_data[dev_data.requester_received_pizza == True]
print(round(len(success)/len(dev_data)*100, 2), "% successful requests")
# 输出：24.41 % successful requests
```

我们看到只有 **24.41%** 的请求最后成功了，数据明显不平衡。这意味着我们后续不能只看准确率，而要看 AUC 这种对不平衡不敏感的指标。

接着我们做时间维度的 EDA。`unix_timestamp_of_request_utc` 是发帖的时间戳（1970 年以来的秒数），我们把它转成日期，再拆出月份、星期几、小时，画出「全年请求量」「一周分布」「一天 24 小时分布」三张图，并用红叉叠加成功请求的比例：

```python
# 把时间戳转成可读时间，再拆出月份、星期、小时
dev_data['datetime'] = pd.to_datetime(dev_data.unix_timestamp_of_request_utc, unit='s')
dev_data['day'] = dev_data.datetime.dt.dayofweek
dev_data['hour'] = dev_data.datetime.dt.hour
dev_data['month'] = dev_data.datetime.dt.month

# 画出每个月请求量的变化（蓝点）和每个月成功率（红叉）
for i in range(12):
    monthly = dev_data[dev_data.month == i]
    plt.plot(i, len(monthly)/len(dev_data)*100, 'bo')
    if len(monthly) > 0:
        plt.plot(i, len(monthly[monthly.requester_received_pizza == True])/len(monthly)*100, 'rx')
```

观察图可以发现请求量在一年里是有波动的，但成功率在不同月份差别不大，说明时间本身不是特别强的影响因素。接着我们对账号年龄、发帖数、评论数等数值特征做箱线图对比，比较成功组和失败组的分布差异。最后，我们还做了词云统计文本里出现最多的词，看看披萨请求里一般都写些什么。

### 2.2 数据清洗

清洗我们分两块来做。

**行为字段的清洗。** 我们只保留带 `_at_request` 后缀的字段（发帖前就能拿到），丢弃所有 `_at_retrieval` 字段；`giver_username_if_known`（是否已知道赠与人）在请求发出时基本都是 `N/A`，我们把它转成 0/1 标记。另外，`requester_upvotes_plus_downvotes_at_request` 和 `requester_upvotes_minus_downvotes_at_request` 这两列名字绕口，其实我们可以从它们反推出「赞」和「踩」的数量，还可以算一个赞踩比：

```python
# upvotes_plus = 赞 + 踩，upvotes_minus = 赞 - 踩
# 所以 (和 + 差)/2 就是赞，(和 - 差)/2 就是踩
mat['upvotes'] = (mat['upvotes_plus_downvotes'] + mat['upvotes_minus_downvotes']) / 2
mat['downvotes'] = (mat['upvotes_plus_downvotes'] - mat['upvotes_minus_downvotes']) / 2
# 赞踩比：加 1 是为了防止除零
mat['votes_ratio'] = mat['upvotes'] / (mat['upvotes'] + mat['downvotes'] + 1)
```

**文本字段的清洗。** 英文文本不能直接喂给模型，我们要先转小写、分词、去掉停用词（stopwords，比如 the、a、this 这些没信息量的虚词）、去掉标点和纯数字。因为正文可能被编辑过，作者建议用 `request_text_edit_aware`（考虑编辑的正文）和 `request_title` 拼接起来使用：

```python
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# 停用词 + 标点组成黑名单，这些词几乎没有预测价值
en_stopwords = set(stopwords.words('english'))
blacklist = set.union(en_stopwords, set(string.punctuation))

def tokenize(s):
    """把整段文字清洗成一串词（空格分隔），过滤掉虚词、标点、数字"""
    tokens = ""
    for sentence in sent_tokenize(s.lower()):      # 先转小写，再按句子切
        for word in word_tokenize(sentence):        # 句子再切成单词
            if len(word) and word not in blacklist and not word.isdigit():
                tokens += word + " "
    return tokens[:-1]

# 标题清洗后拼一个空格，再接正文，作为"全部文本"
dev_title_tokens = dev_data['request_title'].apply(tokenize)
dev_request_tokens = dev_data['request_text_edit_aware'].apply(tokenize)
dev_all_tokens = dev_title_tokens + ' ' + dev_request_tokens
```

### 2.3 模型训练与超参数调优

特征准备好之后，我们把四类特征拼在一起：**行为数值特征 + 文本 TF-IDF 特征 + 长度特征**。作者还尝试了另外两种更花哨的文本表示（这里只做介绍，主线不用）：一个是用 LDA 主题模型把每篇文本压成 5 个主题概率；另一个是用 Google 预训练的 Word2Vec 300 维向量，把一篇文章的词语向量取平均作为全篇的向量。

文本特征我们用 TF-IDF。TF-IDF 的大白话解释：一个词在一篇文章里出现越多，它越重要（TF 词频）；但如果它在**所有**文章里到处都出现，它的重要性就得打折（IDF 逆文档频率）。我们把段落「是否在超过一定数量的文章里出现才保留」设为 `min_df=5`，并且用 1~2 个词的组合来捕捉短语：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

# min_df=5：只在少于 5 篇文章里出现的词直接丢弃，压缩维度
# ngram_range=(1,2)：既看单个词，也看相邻两个词的组合
# sublinear_tf=True：用 1+log(tf) 代替原始词频，削弱刷词的影响
vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 2), norm='l2', sublinear_tf=True)
dev_bow = vectorizer.fit_transform(dev_all_tokens)

# 先用带 L1 惩罚的逻辑回归做特征选择：L1 会把不重要的特征系数压成 0
lr = LogisticRegression(C=1, penalty='l1').fit(dev_bow, dev_labels)
model = SelectFromModel(lr, prefit=True)
# 只保留被 L1 认为"有用"的词，样本量小，压缩维度能防止过拟合
dev_pruned_bow = model.transform(dev_bow)
```

拼接好全部特征后，我们训练一个逻辑回归二分类器。逻辑回归可以理解成一个「把一堆特征加权求和，再经过一个压缩到 0~1 之间的函数，输出就是属于正类的概率」的模型，简单好调。我们用 5 折交叉验证在 `C` 上做网格搜索——`C` 越小正则化越强，模型越不容易过拟合：

```python
from scipy.sparse import hstack
from sklearn.model_selection import GridSearchCV

# 行为特征 + 文本特征横向拼接，形成完整输入
dev_input = hstack([dev_non_textual, dev_pruned_bow])

# 在 0.005 到 0.1 之间均匀取 100 个 C 值做网格搜索
lr = LogisticRegression()
parameters = {'C': np.linspace(0.005, 0.1, 100)}
gs = GridSearchCV(lr, parameters, cv=5)   # 5 折交叉验证
gs.fit(dev_input, dev_labels)
print(gs.best_params_)
# 输出：{'C': 0.1}
```

结果告诉我们，搜索范围内最大的 `C = 0.1` 最好。这个值落在搜索区间边缘，说明还可以往更大的方向继续找，但提升空间不大，我们就先采用它。

### 2.4 模型测试与指标报告

模型训练好之后，我们在没有参与训练、也没参与调参的验证集（那 20% 的独立样本）上评估真实效果。先算比赛指标 AUC，再算各组的准确率：

```python
from sklearn.metrics import roc_auc_score

# 预测"收到披萨"的概率，AUC 需要概率而不是 0/1 标签
pred_valid_prob = gs.predict_proba(valid_input)[:, 1]
pred_valid_labels = gs.predict(valid_input)

auc = roc_auc_score(valid_data['requester_received_pizza'], pred_valid_prob)
print(auc)
# 输出：0.7833353557416172

# 分别统计"成功组"和"失败组"里预测对的占比
success_count = (valid_data['requester_received_pizza'] == True).sum()
failure_count = (valid_data['requester_received_pizza'] == False).sum()
success_correct = ((valid_data['requester_received_pizza'] == True) & (pred_valid_labels == True)).sum()
failure_correct = ((valid_data['requester_received_pizza'] == False) & (pred_valid_labels == False)).sum()
print(round(success_correct / success_count*100, 2), "% success")
print(round(failure_correct / failure_count*100, 2), "% failure")
# 输出：33.17 % success   99.17 % failure
```

我们的验证集 AUC 约 **0.783**。拆开看两个组的预测表现：失败组（没有收到披萨）预测准确率高达 **99.17%**，而成功组只有 **33.17%**，总体准确率 **82.43%**。这个现象的原因很直接：正例本来就少（约四分之一），模型为了把 AUC 拉高会把大部分样本判成负例，所以失败组很容易猜对，成功组则很难召回。这也再次印证了**用准确率评估不平衡问题会让人产生错觉**，只有 AUC 才能公允地衡量模型的排序能力。

作为对照，我们还尝试了 XGBoost（梯度提升树）方案：用 `binary:logistic` 目标函数、`max.depth = 4`、`eval_metric = "auc"`，跑 10 轮后训练集 AUC 达到 0.807。但用 10 折交叉验证后验证集 AUC 只有约 0.61，而且特征重要性名列前茅的竟然是 `unix_timestamp_of_request`（时间戳），说明它多数是在记忆数据里某些时间段整体被拒绝的规律，泛化能力不如前面那条「文本 + 行为 + 逻辑回归」的主线。最终我们以逻辑回归方案作为提交的方案。

---

## 三、总结

本文中我们完成了 Random Acts of Pizza 比赛的完整流程。我们首先做 EDA，读入数据后确认了训练集 4040 行、测试集 1631 行，并发现数据的两大特点：一是正例只有约 24.41%，严重不平衡；二是训练集含有一批「事后抓取」的 `at_retrieval` 字段，属于数据泄漏，必须丢弃。其次我们做了清洗：只保留请求时刻能拿到的 `at_request` 字段，文本统一转小写、分词、去停用词和标点，并顺手从点赞差反推出赞踩比。接着我们构建了「行为数值特征 + TF-IDF 文本特征 + L1 特征选择」的流水线，用逻辑回归加 5 折交叉验证在 C 上做网格搜索，得到最优 C = 0.1。最后在独立验证集上得到 AUC 约 0.783，失败组预测准确率 99.17%、成功组 33.17%、总体 82.43%。我们另外尝试的 XGBoost 方案交叉验证 AUC 只有约 0.61，因此没有采用。

这个比赛给我们的最大启发有两层：一是**看指标要跟任务匹配**，面对不平衡数据要相信 AUC 而不是准确率；二是**处理特征要先想清楚"预测当下你能拿到什么"**，把所有事后信息剔除干净，模型才有真正的泛化能力。