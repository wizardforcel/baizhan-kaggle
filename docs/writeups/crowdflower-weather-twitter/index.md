# CrowdFlower 天气推文预测（Weather Twitter）

## 一、比赛简介

### 1.1 比赛背景

社交网络上每天都有人在聊天气：「外面下暴雨了」「今天 30 度热死了」「周末要去滑雪」。这些随口一说的话里，其实藏着真实的天气信息——地点、天气现象、冷热程度、甚至说话人的心情。

CrowdFlower 办的这场比赛，就是把这些英文推文收集起来，交给众包工人（crowdworker）逐条标注。众包的意思是：一条推文不只给一个人看，而是给好几个人看，每个人独立判断「这条推文有没有提到下雨」「有没有提到风」，最后把几个人的判断平均起来，得到一个 0 到 1 之间的分数。所以标签不是干净的 0 或 1，而是像 0.403、0.597 这样的小数——这代表「大约四成的人认为有，六成的人认为没有」。

这个比赛真正的难点在于：**推文是极其口语化、充满缩写和噪音的短文本**。一条推文可能只有五个单词，还夹杂着 `{link}`、`@mention`、`#话题` 这些占位符，模型必须从这么一点点文字里判断天气属性。

### 1.2 比赛目标

训练集有 77946 行、28 列；测试集有 42157 行、4 列。两边的差集就是我们要预测的东西：

- 输入列（训练集和测试集都有）：`id` 编号、`tweet` 推文正文、`state` 州名、`location` 城市/地区名；
- 输出列（只有训练集有）：`s1`～`s5`、`w1`～`w4`、`k1`～`k15`，一共 24 列，每列都是 0～1 之间的小数。

也就是说，一条推文要同时预测 24 个属性，这是一个**多输出回归**问题（multi-output regression）。数据文件夹里的 `variableNames.txt` 记录的就是这 24 个列名。

### 1.3 评价标准

比赛对 24 个标签列分别计算误差，再汇总成一个总分，**分数越低越好**。材料里这份 notebook 的做法是对每一列单独算 RMSE（均方根误差，Root Mean Squared Error）：先把预测值和真实值的差平方、求平均、再开根号，得到的就是「平均偏离多少个 0 到 1 之间的单位」。把 24 列的 RMSE 放在一起看，就能判断整体水平。

这份 notebook 在排行榜上的公开分数是 **0.16372**。考虑到标签本身就在 0～1 之间，0.16 左右的误差并不算小——它说明有一批标签（尤其是语义性强的那些）确实很难从几十个单词里判断出来。

---

## 二、解题过程

我们的整体思路是：**把能拼的文本都拼起来做 TF-IDF，再补上几个手工统计特征，最后给 24 个标签各训练一个线性回归器。** 下面按数据探索、清洗、训练、测试四步展开。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，看看数据规模，并且用「训练集有、测试集没有」这个条件自动找出标签列：

```python
import pandas as pd
import numpy as np

# 训练集、测试集、提交样例
train = pd.read_csv("/kaggle/input/crowdflower-weather-twitter/train.csv")
test  = pd.read_csv("/kaggle/input/crowdflower-weather-twitter/test.csv")
sub   = pd.read_csv("/kaggle/input/crowdflower-weather-twitter/sampleSubmission.csv")

print(train.shape)   # (77946, 28)
print(test.shape)    # (42157, 4)

# 训练集比测试集多出来的列，就是 24 个待预测的标签
target_cols = train.columns.difference(test.columns)
target = train[target_cols]        # 先把标签单独存一份
train.drop(target_cols, axis=1, inplace=True)   # 特征里只留输入列

print(len(target_cols))   # 24
```

这个写法比手写 24 个列名更稳：以后列名变了也不会漏。

接着我们看数据长什么样。推文列里已经提前把链接和 @ 替换成了占位符，比如：

```
0    Jazz for a Rainy Afternoon:  {link}
1    RT: @mention: I love rainy days.
2    Good Morning Chicago! Time to kick the Windy C...
3    Preach lol! :) RT @mention: #alliwantis this t...
```

我们能直接读出两个信息：一是文本很短，二是 `{link}`、`@mention` 这种占位符本身就是有用的信号——带链接的推文往往是天气预报或新闻，带 @ 的往往是转发。

再看标签的取值。以第一行为例，`s3=1.0`、`w1=0.8`、`w3=0.2`、`k9=1.0`、`k15=1.0`，其余大多是 0。0.8 和 0.2 加起来正好是 1；另一条推文的 `s4=0.403`、`s5=0.597` 加起来也是 1，这印证了前面说的「多个标注者投票后取平均」。

最后我们统计一下缺失情况：`location` 和 `state` 都有不少空值。这不是脏数据，而是真实情况——很多人发推文根本不带定位。

### 2.2 数据清洗

我们做的第一件事，是把「缺失」本身变成一个特征，而不是简单填掉：

```python
for data in (train, test):
    # 有没有写地点，可能和推文内容风格有关，所以单独记一个 0/1 标志
    data['missing_loc']   = data['location'].isnull().astype(int)
    data['missing_state'] = data['state'].isnull().astype(int)

    # 拼接字符串前先填空串，否则 'nan' 会被当成一个单词进入词表
    data['location'] = data['location'].fillna('')
    data['state']    = data['state'].fillna('')

    # 把推文、州、城市合成一段文本，交给后面的 TF-IDF 统一处理
    data['full_text'] = data['tweet'] + ' ' + data['state'] + ' ' + data['location']
```

为什么要拼起来？因为「Arizona（亚利桑那）」这种州名和「热」高度相关，「Minnesota（明尼苏达）」和「冷、雪」高度相关。把地名放进同一段文本，模型就能学到「地名 + 天气词」的搭配。

第二件事是把美国州的缩写展开成全名。推文里写 `CA`，词表里却是 `California`，不统一就会丢掉大量信号：

```python
import re

# 州名缩写对照表（原表共 51 项，这里只列一部分示意）
states = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "CA": "California",
          "CO": "Colorado", "FL": "Florida", "IL": "Illinois", "NY": "New York",
          "TX": "Texas", "WA": "Washington", "NYC": "New York"}

def cleanpunc(sentence):
    # 问号、感叹号、引号、井号直接删；逗号句号括号换成空格，避免把两个词粘在一起
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/|:]', r' ', cleaned)
    return cleaned

def acronym(s, dct):
    # 逐词查表：能查到就换成全名，查不到就原样保留
    return ' '.join([dct.get(i, i) for i in s.split()])

for data in (train, test):
    data['full_txt2'] = [acronym(cleanpunc(t), states) for t in data['full_text']]
```

第三件事是手工构造几个统计特征。文本长度这类信息 TF-IDF 抓不到，但它确实有用：长推文更可能在认真描述天气，短推文更可能是随口一句：

```python
for data in (train, test):
    # 词数、字符数，以及平均词长（+1 是为了防止除以 0）
    data['num_word'] = data['tweet'].apply(lambda x: len(x.split()))
    data['num_char'] = data['tweet'].apply(len)
    data['avg_word_length'] = data['num_char'] + 1 / data['num_word']

    # 原始数据已经把链接和 @ 换成了固定占位符，所以直接查字符串即可
    data['link_word']    = data['full_text'].str.contains('{link}', regex=False).astype(int)
    data['mention_word'] = data['full_text'].str.contains('@mention', regex=False).astype(int)
```

最后是标准的文本清洗：去掉残留网址、去掉标点、只保留纯字母单词、去掉 the / is / a 这类停用词（stopword，出现频率极高但没有区分度的词），全部转小写：

```python
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))

def cleanhtml(sentence):
    return re.sub(r'http\S+', '', sentence)   # 去掉残留的网址

for data in (train, test):
    docs = []
    for sent in data['full_text']:
        text = cleanpunc(cleanhtml(sent))
        # 只留字母词，并且过滤停用词；转小写是为了让 Rain 和 rain 算同一个词
        words = [w.lower() for w in text.split()
                 if w.isalpha() and w.lower() not in stopwords]
        docs.append(' '.join(words))
    data['clean_text'] = docs
```

这里看起来和前面的 `full_txt2` 有点重复：那一份是「把州名缩写换掉」的中间产物，这一份 `clean_text` 才是最终喂给 TF-IDF 的输入。多走一遍换来的是两条路径互不干扰——词表替换出了问题，可以单独拿 `full_txt2` 来核对。

### 2.3 模型训练与超参数调优

我们首先把清洗后的文本变成 TF-IDF 向量。TF-IDF 的意思是「词频 × 逆文档频率」：一个词在这条推文里出现得越多、在所有推文里出现得越少，权重就越高。所以像 `weather` 这种到处都是的词会被自动压低，`hail`、`tornado` 这种有辨识度的词会被抬高。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# ngram_range=(1,3)：单词、二元词组、三元词组都统计，"rainy day" 这种搭配也能被抓到
# max_df=0.75：出现在超过 75% 文档里的词丢掉，太常见没信息量
# min_df=5：出现次数少于 5 次的词丢掉，太罕见多半是拼写错误
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.75, min_df=5)
train_vec = vectorizer.fit_transform(train['clean_text'])
test_vec  = vectorizer.transform(test['clean_text'])   # 测试集只能 transform，不能重新 fit

# 用 SVD 把高维稀疏向量压成 2 维，作为两个额外的稠密特征
svd = TruncatedSVD(n_components=2, random_state=0)
train_svd = svd.fit_transform(train_vec)
test_svd  = svd.transform(test_vec)
```

SVD（奇异值分解）在这里的作用是「降维」：把几万维的词袋压成 2 个最能代表整体差异的方向，让线性模型也能感受到「这条推文整体偏哪一类」。

接下来我们把四块特征横向拼在一起：TF-IDF 稀疏矩阵、三个数值特征、四个 0/1 标志位、SVD 两列。拼接前必须做归一化，否则 `num_char` 是几十上百的量级，会把 0～1 的二值特征完全盖住：

```python
import scipy as sp
from sklearn.preprocessing import normalize

extra_cols = ['num_word', 'num_char', 'avg_word_length',
              'missing_loc', 'missing_state', 'link_word', 'mention_word']

X = sp.sparse.hstack((
    vectorizer.fit_transform(train['clean_text']),          # 文本特征（稀疏）
    normalize(train[['num_word', 'num_char', 'avg_word_length']].values),  # 数值特征归一化
    sp.csr_matrix(train[['missing_loc', 'missing_state',
                         'link_word', 'mention_word']].values),            # 0/1 标志位
    sp.csr_matrix(train_svd),                               # SVD 两列
), format='csr')

print(X.shape)   # (77946, 32439)  三万多个特征

# 测试集必须走完全一样的流程，否则特征对不上
test_sp = sp.sparse.hstack((
    vectorizer.transform(test['clean_text']),
    normalize(test[['num_word', 'num_char', 'avg_word_length']].values),
    sp.csr_matrix(test[['missing_loc', 'missing_state',
                        'link_word', 'mention_word']].values),
    sp.csr_matrix(test_svd),
), format='csr')
```

然后是训练。我们一开始试过梯度提升树（GradientBoostingRegressor）、SGD 回归、Lasso、LinearSVR 和普通线性回归，最后发现**岭回归（Ridge）**在这份高维稀疏数据上又稳又快。Ridge 就是在普通最小二乘上加一个惩罚项，防止系数被个别特征拉得过大，对共线性很强的文本特征特别合适。

我们的做法是「一列一个模型」：

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt

# 80% 训练、20% 验证；这里不 stratify，因为标签是连续值
x, x_test, y, y_test = train_test_split(X, target, test_size=0.2, random_state=0)

clf = Ridge(random_state=25)
dict_preds = {}
for col in target.columns:          # 24 个标签逐个训练
    clf.fit(x, y[col])
    rmse = sqrt(mean_squared_error(y_test[col], clf.predict(x_test)))
    print(col, "RMSE: %.4f" % rmse)
    dict_preds[col] = clf.predict(test_sp)   # 顺便把测试集预测存起来
```

跑出来的 RMSE 差异非常大：

- 最好的是 `k6`，RMSE 只有 **0.0216**，`k8` 是 **0.0324**，`k3` 是 **0.0471**；
- 最差的是 `w1`，RMSE **0.2833**，`s2` 是 **0.2436**，`w3` 是 **0.2315**。

这说明 24 个标签的难度完全不是一个量级。`k6` 这类标签绝大多数样本都是 0，模型只要一直预测「接近 0」就能拿到很低的误差；而 `w1`、`s2` 这类标签本身分布更均衡、更依赖语义理解，线性模型就很难压下去。看懂这一点，比盲目调参有用得多。

关于调参，真正值得动的是三块：TF-IDF 的 n-gram 范围（这里定在 1～3，再大维度会爆炸）、`min_df`/`max_df` 的过滤强度、以及 Ridge 的正则系数。由于每一列都要单独训练，24 列 × 多组参数的成本很高，而且提升幅度很小，所以我们把精力放在了特征拼接和「链式回归」上，没有做大规模网格搜索。

链式回归（RegressorChain）的思路是：24 个标签之间其实有关联（比如「下雨」和「雷暴」经常同时出现），与其各训各的，不如排成一条链，让后一个标签的模型把前面标签的预测值也当成输入：

```python
from sklearn.multioutput import RegressorChain

# 以 Ridge 为基学习器，按列顺序串成一条链
regressor = RegressorChain(Ridge(random_state=25))
regressor.fit(x, y)

predictions = regressor.predict(x_test)
print(sqrt(mean_squared_error(y_test, predictions)))   # 0.1602
```

整体 RMSE 是 **0.1602**。这里是 24 列的所有预测值放在一起算一个总误差，和前面逐列算完再分别看不是同一口径；它和排行榜上的 0.16372 属于同一量级，可以作为方案有效性的交叉印证。链式回归的意义在于：标签之间共享信息时，让模型「看到」前面的预测是有帮助的。

### 2.4 模型测试与指标报告

确认方案可用后，我们用全部训练数据重新拟合，再对测试集预测。这里有两个必须做的收尾动作：**截断越界值**和**对齐提交列顺序**。

```python
# 用全量训练数据重训（多喂数据总没坏处）
regressor.fit(X, target)

pred2 = regressor.predict(test_sp)
cols = target.columns

df_res2 = pd.DataFrame(pred2, columns=cols)
df_res2[df_res2 < 0] = 0    # 标签定义域是 0~1，负数和大于 1 的预测没有意义
df_res2[df_res2 > 1] = 1

df_res2['id'] = sub['id']            # 贴回 id
df_res2 = df_res2[sub.columns]       # 严格按 sampleSubmission 的列顺序重排
df_res2.to_csv('sub2.csv', index=False)
```

截断这一步很关键：Ridge 是线性模型，遇到极端特征时完全可能输出 -0.3 或 1.4。这些值放在 RMSE 里会被平方放大，白白拉高误差；截到 [0, 1] 几乎只会让分数变好，不会变坏。

最后我们核对三件事：提交文件的行数是不是 42157、列名是不是和 `sampleSubmission.csv` 完全一致、`id` 有没有重复或缺失。比赛里因为格式问题拿不到分数的例子并不少见，这一步花不了几分钟。

材料里还有另一份 notebook，走的是更轻的路线：只用 `max_features=100` 的 TF-IDF，加 `MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50))`，验证集 R² 是 **0.417**。它证明「词表小一点 + 树模型」也能跑通，但整体精度不如上面的 Ridge 方案，所以我们主写这一条。

---

## 三、总结

本文中我们完成了 CrowdFlower 天气推文预测的完整流程。我们首先通过列名差集自动识别出 24 个标签，并确认了标签是多个标注者投票平均后的小数；其次我们把推文、州、城市拼成一段文本，展开州名缩写、过滤停用词，并把「地点是否缺失」「是否带链接」「是否带 @」变成显式特征；接着我们用 1～3 元 TF-IDF 加 SVD 降维，与数值特征横向拼接成三万维的稀疏矩阵，给每个标签单独训练 Ridge 回归，又用 RegressorChain 利用标签之间的相关性；最后在全量数据上重训、把预测截断到 [0, 1]、按样例文件对齐列顺序输出，验证集整体 RMSE 约 0.1602，排行榜公开分数 0.16372。

这个比赛给我们最深的印象是：**多输出任务的难点不在模型，而在「每个标签值不值得花同样的力气」**。`k6` 的 RMSE 只有 0.02，`w1` 却有 0.28，差了十几倍。如果只盯着总分调参，很容易把时间浪费在那些本来就接近常数的标签上；先按列拆开看误差，才知道真正该攻的是哪几个属性。
