# 搜索结果相关性评分（CrowdFlower Search Results Relevance）

## 一、比赛简介

### 1.1 比赛背景

在电商网站里搜索「wireless mouse」，搜索引擎会返回一屏商品。排在前面的那个，真的是无线鼠标吗？标题里写着「无线鼠标」但其实是鼠标垫的商品，该打几分？

这个问题难在**「相关」是主观的**。同一条商品，有人觉得完全符合需求，有人觉得差得远。所以这份数据没有让一个人拍板，而是让**三位标注员各自独立打分**：每条样本的标签取三人的**中位数**（`median_relevance`），另外把三人打分的**方差**（`relevance_variance`）也一并给出——方差大，说明连标注员自己都吵不清楚，这条样本的「标准答案」本身就不太可靠。

每条样本是一个三元组：**搜索词**（`query`）、**商品标题**（`product_title`）、**商品描述**（`product_description`），标签是 1～4 分的相关性，分越高越相关。这是一个典型的**文本相关性建模**任务，本质上要回答「这段话和这个词说的是不是同一件事」。

难点集中在三处：

1. **标签有序且极不平衡**。4 分有 6171 条，占了六成；1 分只有 774 条。模型只要全猜 4，准确率就有六成，看起来很美，实际没用。
2. **文本脏、长短悬殊**。约 24% 的商品描述是空的，不少描述里还嵌着整段 HTML（`<table>`、图片链接），长度从 0 一路到 19436 字符。
3. **评价指标很特殊**。这个比赛用**二次加权 Kappa（QWK）**，它不看「猜对没有」，而看「猜得离正确答案有多远」——把 4 分猜成 3 分罚得很轻，猜成 1 分罚得很重。

### 1.2 比赛目标

训练集 `train.csv` 有 **10158 行、6 列**：

- `id`、`query`、`product_title`、`product_description`；
- `median_relevance`：三位标注员打分的中位数，也就是要预测的目标；
- `relevance_variance`：三人打分的方差，衡量标注一致性。

测试集 `test.csv` 有 **22513 行、4 列**，没有后两列。我们要给每条测试样本预测一个 **1～4 的整数**，提交文件两列：`id` 和 `prediction`。官方样例提交里每条都是 3——这是最「安全」的常数猜测，也是我们的第一个参照物。

注意一个关键事实：**`relevance_variance` 不能当特征**。它在测试集里根本不存在，用它训练等于作弊，线上一定翻车。

### 1.3 评价标准

比赛用 **Quadratic Weighted Kappa（QWK，二次加权 Kappa）** 打分。它原本是统计学里衡量「两个评分者一致程度」的指标，取值范围 **-1 到 1**，0 表示「和瞎猜一样」，1 表示完全一致。

算法分五步（材料里一步步手算演示过）：

1. 先做一个**混淆矩阵** `O`，`O[i][j]` 是「真实 i 分、预测 j 分」的样本数；
2. 再造一个**权重矩阵**，差距越大罚得越重：`w[i][j] = (i-j)² / (K-1)²`，四分类时 `K=4`，分母是 16；
3. 分别统计真实和预测的**打分直方图**；
4. 两个直方图做**外积**，得到「纯靠运气」时应该出现的分布 `E`；
5. 归一化后，用「观测到的加权分歧」除以「期望的加权分歧」，再取 `1 - 比值`。

写成公式是：

```
QWK = 1 - Σ w[i][j]·O[i][j] / Σ w[i][j]·E[i][j]
```

**这个指标把「顺序」看得比「对错」重得多。** 材料里有个绝妙的对比：两组预测的准确率**都是 0**（一个都没猜中），但第一组把真实值 `[0,4,2,3]` 预测成 `[1,0,0,0]`，QWK 是 **-0.18**（比瞎猜还差）；第二组预测成 `[1,3,1,2]`，虽然也全错，但每个都只差一两档，QWK 有 **0.68**。这就是为什么这种比赛绝不能拿准确率当指标——准确率会把「差一点」和「差得离谱」当成一回事。

---

## 二、解题过程

我们走的主线是：**清洗文本 → TF-IDF 抽词 → SVD 降维 → SVM 分类**，用 QWK 当调参目标。我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步，把两份数据读进来，先看形状和缺失：

```python
import pandas as pd
import numpy as np

train = pd.read_csv('/kaggle/input/crowdflower-search-relevance/train.csv.zip')
test  = pd.read_csv('/kaggle/input/crowdflower-search-relevance/test.csv.zip')

print(train.shape)          # (10158, 6)
print(test.shape)           # (22513, 4)

# info() 一次看清类型与非空数量
# 关键发现：product_description 只有 7714 个非空 → 缺了约 24%
train.info()
```

测试集比训练集大了一倍多（22513 对 10158），意味着训练数据相对稀缺，模型容量不能开太大。而描述缺失的 7714/10158 这个比例，直接决定了后面不能删样本，只能填空。

第二步，看目标分布和文本的基本形态：

```python
# 标签分布：4 分 6171 条占六成，1 分只有 774 条，严重不平衡
print(train['median_relevance'].value_counts().sort_index())
# 1: 774   2: 1476   3: 1737   4: 6171
print(train['median_relevance'].mean())        # 3.3098

# 一共有 261 个不同的搜索词，同一个 query 会对应多个商品
print(train['query'].nunique())                # 261
print(train['query'].value_counts().head(3))   # wireless mouse 113 次

# 商品标题几乎不重复：91.78% 只出现一次
print((train['product_title'].value_counts() == 1).sum())   # 9323
```

这三组数字各有用途。标签分布告诉你**评估必须用 QWK**；261 个 query 说明数据是「按搜索词分组」的，不同 query 的候选商品完全不同，做交叉验证时最好按 query 分组；标题唯一的比例高达 91.78%，说明靠「记住某条标题」是学不到东西的，必须学词语之间的匹配关系。

第三步，量一量文本长度，并检查 HTML：

```python
from bs4 import BeautifulSoup
import numpy as np

# 标题长度：中位数 52 字符，99% 在 114 以内，最长 250
title_len = train['product_title'].str.len()
print(np.percentile(title_len, [50, 90, 99, 99.9]))   # 52 / 79 / 114 / 179.5

# 描述长度差异极大：三成是 0，一半在 264 字符内，最长 19436
desc_len = train['product_description'].fillna('').str.len()
print(np.percentile(desc_len, [30, 50, 90, 99]))       # 134 / 264.5 / 901 / 1930

# 抽查 HTML：id=290 的描述开头就是 <table>，不清理会把标签当词
print(train.loc[train.id == 290, 'product_description'].iloc[0][:30])
```

长度分布说明两件事：一是**标题短、信息密度高**，是主要信号；二是**描述长短相差上百倍**，直接拼进 TF-IDF 会让长描述的权重失控，所以后面要么截断，要么单独处理。HTML 检查则确认了脏数据的形态——材料里 `id=290` 的描述整段是表格标签加样式串，`BeautifulSoup` 能从中提取出正文。

### 2.2 数据清洗

**第一步：缺失填空串，不删样本。** 24% 的描述是空的，删掉就丢了四分之一的监督信号，而这些样本的标题和查询词明明是完好的：

```python
# 描述缺失填空串：保留样本，让模型至少还能用 query + title
train['product_description'] = train['product_description'].fillna('')
test['product_description']  = test['product_description'].fillna('')

# 确认没有残留 NaN
print(train.isna().sum().sum())   # 0
```

**第二步：文本清洗三步走——去 HTML、加前后缀、词干化。** 材料里最有意思的技巧是给词加「位置前缀」：

```python
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import text
import re

stemmer = PorterStemmer()

def clean_row(query, title, desc):
    # 1) 去掉 HTML 标签，只留正文文本
    q = BeautifulSoup(str(query), 'html.parser').get_text(' ')
    t = BeautifulSoup(str(title), 'html.parser').get_text(' ')
    d = BeautifulSoup(str(desc),  'html.parser').get_text(' ')
    # 2) 给 query 的词加 q 前缀、给 title 的词加 z 前缀
    #    同一个词出现在哪一侧就成了不同的 token，模型能学到"词在查询里"这层信息
    q = ' '.join('q' + w for w in q.split(' '))
    t = ' '.join('z' + w for w in t.split(' '))
    s = q + ' ' + t + ' ' + d
    # 3) 只保留字母数字，标点和换行一律换成空格
    s = re.sub('[^a-zA-Z0-9]', ' ', s)
    # 4) 词干化：mouse / mice 之外，lights/lighting/lighted 都折叠成 light
    return ' '.join(stemmer.stem(w) for w in s.split(' '))

train_text = [clean_row(*row) for row in
              train[['query', 'product_title', 'product_description']]
              .itertuples(index=False)]
```

加 `q` / `z` 前缀的好处是**保留了字段来源**：TF-IDF 本来会把 query 和 title 的词混在一个词表里，「wireless 出现在查询里」和「wireless 出现在标题里」对相关性的含义完全不同，前缀让这两种情况变成两个特征。代价是停用词表要跟着改——`the` 被删了，`qthe`、`zthe` 还留着，得手动补进停用词：

```python
# 加了前缀后，"qthe"/"zthe" 不再是停用词，要把整份停用词表复制成带前缀的版本
STOP = text.ENGLISH_STOP_WORDS.union(
    {'q' + w for w in text.ENGLISH_STOP_WORDS} |
    {'z' + w for w in text.ENGLISH_STOP_WORDS})
```

**第三步：按标注一致性过滤训练样本。** 既然数据给了 `relevance_variance`，就可以拿它筛掉「连标注员都意见不一」的样本：

```python
split = int(len(train) * 0.8)          # 前 80% 训练，后 20% 验证
train_0, dev = train[:split], train[split:]

# 方差越大，标签越不可信。先只保留方差 < 1.0 的样本
clean_1 = train_0[train_0.relevance_variance < 1.0]    # 7558 条
# 更严格的版本：方差 < 0.5，只剩 6206 条
clean_2 = train_0[train_0.relevance_variance < 0.5]
print(len(train_0), len(clean_1), len(clean_2))        # 8126 / 7558 / 6206
```

我们最终用了第一版（7558 条）。这是一次**用标签质量换数据量**的权衡：方差小于 1 的样本里，三人打分最多相差一档，标签基本可信；砍得太狠（< 0.5）会丢掉 1900 多条样本，模型反而学不够。另外，长描述超过 10000 字符的样本只占极小比例（99.96% 都在 99.97 分位以内），材料建议直接截断，避免个别超长文本把 TF-IDF 的统计量带偏。

### 2.3 模型训练与超参数调优

**主线：TF-IDF → SVD → 标准化 → SVM。** 思路是先把文本变成稀疏词袋，再压成稠密向量，最后交给 SVM 分类。整条链路用 `sklearn` 的 `Pipeline` 串起来：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import pipeline, metrics

# 把 QWK 包成 sklearn 认识的打分器，调参时直接最大化它
kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better=True)

clf = pipeline.Pipeline([
    # 1) 1~5 元 n-gram 的 TF-IDF：从单词到五词短语都算特征
    #    sublinear_tf 用 1+log(tf) 压高频词，smooth_idf 避免除零
    ('tfv', TfidfVectorizer(min_df=3, ngram_range=(1, 5),
                            sublinear_tf=1, smooth_idf=1,
                            stop_words='english', strip_accents='unicode',
                            token_pattern=r'\w{1,}')),
    # 2) 稀疏高维 → 400 维稠密向量；SVM 在稠密低维上才算得动
    ('svd', TruncatedSVD(n_components=400)),
    # 3) 标准化：SVD 各维尺度不同，不缩放会被方差大的维度主导
    ('scl', StandardScaler()),
    # 4) RBF 核 SVM 做 1~4 的四分类
    ('svm', SVC(C=10)),
])
```

这条流水线每一环都有明确的分工。**TF-IDF 的 `ngram_range=(1,5)`** 是关键：单看「light」和「bulb」没意义，但「christmas light bulb」这个五元组能直接对上查询。**SVD 降维**解决的是维度灾难——1～5 元词表动辄几十万维，SVM 在这种稀疏空间里算距离极慢且噪声大，压到 400 维后既保留了主要语义方向，又能被 `StandardScaler` 处理。**标准化**不能省，因为 SVD 各主成分的方差依次递减，不缩放的话 SVM 只看得到前几个成分。

**第二步：用 QWK 做网格搜索。** 只搜两个最关键的参数——SVD 维度和 SVM 的正则强度 `C`：

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'svd__n_components': [200, 400],
              'svm__C': [10, 12]}

model = GridSearchCV(estimator=clf, param_grid=param_grid,
                     scoring=kappa_scorer,   # 直接用比赛指标选参数
                     cv=2, n_jobs=-1, refit=True, verbose=10)
model.fit(X, y)                              # X 是 TF-IDF 矩阵，y 是 1~4 标签

print("Best score: %0.3f" % model.best_score_)   # 0.557
print(model.best_params_)                        # {'svd__n_components': 400, 'svm__C': 10}
```

`cv=2` 看起来太少，但这里是被算力逼的：一个 TF-IDF + SVD(400) + SVM 的完整训练要几分钟，4 组参数 × 5 折 = 20 次训练太贵，2 折已经能分出高下。这也提醒我们：**调参的折数要和单次训练成本匹配**，不是越多越好。

**第三步：一条回归路线做对照。** 材料里另一份方案不把标签当四分类，而是先把 1～4 线性映射到 [0,1]，用回归模型去拟合，再映射回整数：

```python
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, cohen_kappa_score, make_scorer

# 1~4 → [0,1]：让回归模型在连续空间里学"相关程度"，而不是四个孤立类别
y_train_reg = [(v - 1) / 3 for v in train_y]

def reg_scorer(true, pred):
    # 回归输出先裁到 [0,1]，再还原成 1~4 的整数，最后算 kappa
    pred = [int(round(min(1, max(0, p)) * 3 + 1)) for p in pred]
    true = [int(round(t * 3 + 1)) for t in true]
    return cohen_kappa_score(true, pred)

svr = GridSearchCV(SVR(),
                   {'C': [1, 2, 5, 10], 'epsilon': [0.1, 0.2, 0.5],
                    'kernel': ('linear', 'rbf', 'poly', 'sigmoid')},
                   scoring=make_scorer(reg_scorer), cv=5, n_jobs=8)
svr.fit(train_x, y_train_reg)   # 48 组参数 × 5 折 = 240 次训练，约 9.3 分钟
print(svr.best_params_)          # {'C': 1, 'kernel': 'sigmoid'}
```

这条路线的好处是**利用了标签的有序性**：回归模型知道「4 比 3 更接近真实值」，而普通分类器把 1/2/3/4 当成四个互不相干的类别。它在验证集上的 MSE 是 **0.0677**、QWK 是 **0.2765**——比主线低不少，说明在这个任务上「四分类 + QWK 直接优化」更对路。顺带一提，材料里这个 `reg_scorer` 调的是 `cohen_kappa_score` 的默认参数，也就是**不加权**的 kappa，而比赛用的是二次加权版本，这是一个应该修正的小 bug。

**第四步：另一条「近邻特征」路线。** 材料的第三份方案思路完全不同：不学文本，而是看「和这条样本最像的几条样本，标签是什么」：

```python
from sklearn.metrics import pairwise

# 用 word2vec 把每条描述压成 300 维平均向量，再两两算余弦相似度
a = pairwise.cosine_similarity(embeddings)
for i in a.index:
    # 找最像的 6 条，把它们的标签统计当成特征
    iii = a.iloc[i].sort_values(ascending=False)[:6].index
    meann.append(embeddings.iloc[iii]['target'].mean())   # 邻居标签均值
    maxx.append(embeddings.iloc[iii]['target'].max())     # 邻居最高分
    # 再数一数邻居里 1/2/3/4 分各有几个，作为四维计数特征
# 训练时按标注方差加权：方差越大权重越低，等于相信"标注一致"的样本
weight = 1 / (1 + df['relevance_variance'])
```

这份方案还用了 Levenshtein 编辑距离和 Jaccard 相似度做同样的「找邻居、统计邻居标签」操作，再把所有特征喂给 XGBoost，用 `LeaveOneOut` 逐条留一验证，评分同样用加权 kappa。它更像「基于案例的推理」——**如果两段描述几乎一样，它们的相关性分数也应该差不多**。缺点是计算量大（两两相似度矩阵是平方复杂度），而且它是在单个 query 的子集上做的实验。

### 2.4 模型测试与指标报告

**主线的验证分数。** 网格搜索的输出来自两折交叉验证，最优组合是 SVD 400 维 + `C=10`，平均 QWK 为 **0.557**。四组参数的逐折分数很能说明问题：

```text
svm__C=10, svd__n_components=200 → 0.5316 / 0.5133
svm__C=12, svd__n_components=200 → 0.5327 / 0.5132
svm__C=10, svd__n_components=400 → 0.5635 / 0.5503   ← 最好
svm__C=12, svd__n_components=400 → 0.5580 / 0.5500
```

两个规律很清楚：**维度从 200 提到 400 带来稳定提升**（说明文本语义确实需要更大的空间才装得下），**C 从 10 加到 12 反而略降**（说明正则已经到位，再放松就开始拟合噪声）。

**自己实现一遍 QWK，确认指标没算错。** 比赛指标特殊，最稳妥的做法是手写一遍、和 `sklearn` 对拍：

```python
from sklearn.metrics import confusion_matrix

def quadratic_weighted_kappa(actual, predicted, n=5):
    # 步骤 1：混淆矩阵 O，O[i][j] = 真实 i 分、预测 j 分的样本数
    O = confusion_matrix(actual, predicted, labels=range(n)).astype(float)
    # 步骤 2：权重矩阵，差距越大罚越重：(i-j)^2 / (K-1)^2
    w = np.array([[(i - j) ** 2 / (n - 1) ** 2 for j in range(n)]
                  for i in range(n)])
    # 步骤 3：两边的打分直方图
    hist_a = np.array([(actual == i).sum() for i in range(n)], dtype=float)
    hist_p = np.array([(predicted == i).sum() for i in range(n)], dtype=float)
    # 步骤 4：期望矩阵 = 外积，代表"纯靠运气"时的分布
    E = np.outer(hist_a, hist_p)
    # 步骤 5：观测分歧 / 期望分歧，再取 1 减
    return 1 - (w * (O / O.sum())).sum() / (w * (E / E.sum())).sum()
```

材料在手算时用的就是这五步，而且在两个小例子上验证过：两个都「全错」的预测分别得到 -0.18 和 0.68，与 `sklearn` 的 `cohen_kappa_score(..., weights='quadratic')` 完全一致。**在比赛里先把指标实现对，比调模型重要得多**——指标算错，后面所有调参都是白干。

**常数预测的基线。** 官方样例提交里每条都是 3。这个基线值得算一下：如果预测全是同一个常数，那么「观测矩阵」和「期望矩阵」在加权求和后完全相同，QWK 恒等于 **0**。也就是说，一个什么都学不到、只会交 3 的模型，分数正好是「和瞎猜一样」；任何有意义的模型都必须明显大于 0。

**生成提交。** 最后用最优模型给 22513 条测试样本预测整数标签：

```python
# 用网格搜索选出的最优流水线给测试集打分
preds = model.predict(X_test)          # SVC 直接输出 1~4 的整数类别

# 提交格式：id 顺序必须和测试集完全一致，prediction 是 1~4 的整数
submission = pd.DataFrame({'id': test.id.values, 'prediction': preds})
submission.to_csv('submission.csv', index=False)

# 检查一下预测分布：如果全挤在 4 上，说明模型被不平衡带偏了
print(submission['prediction'].value_counts())
```

这里必须强调 `prediction` 的类型：QWK 只定义在**离散有序标签**上，如果输出连续值（比如 3.47），要么被判格式错误，要么被按错误的方式取整，分数会莫名其妙地差。SVC 天然输出整数，回归路线则必须手动做「裁到 [0,1] → 四舍五入 → 还原到 1～4」这三步。

---

## 三、总结

本文中我们完成了 CrowdFlower 搜索结果相关性评分的完整流程。我们首先用 EDA 摸清了数据：训练集 10158 行、测试集 22513 行；标签从 1 到 4，4 分占 6171 条、1 分只有 774 条，严重不平衡；共有 261 个搜索词，91.78% 的商品标题只出现一次；商品描述有约 24% 缺失，长度从 0 到 19436 字符不等，而且部分描述里嵌着 HTML 表格。其次我们做了针对性的清洗：把缺失描述填成空串而不是删样本、用 `BeautifulSoup` 剥掉 HTML、给 query 的词加 `q` 前缀、给标题的词加 `z` 前缀以保留字段来源、用 Porter 词干化把同源词折叠、把带前缀的停用词补进停用词表，并按标注员打分的方差 `< 1` 过滤训练样本（剩 7558 条）。接着我们搭了两条路线：主线是 TF-IDF（1～5 元）→ SVD（400 维）→ 标准化 → RBF 核 SVM，用二次加权 Kappa 当网格搜索的评分函数，在 200/400 维和 C=10/12 之间选了 400 维配 C=10；对照路线是把标签映射到 [0,1] 做 SVR 回归再取整，另一份方案则用 word2vec 句向量、余弦相似度、编辑距离和 Jaccard 找近邻、统计邻居标签作为特征，配合按标注方差加权的 XGBoost 和留一验证。最后我们在两折交叉验证上得到 **0.557** 的 QWK，手写了一遍五步 QWK 计算与 `sklearn` 对拍，确认了常数预测的基线正好是 0，并在测试集上输出 1～4 的整数预测、按 `id` 与 `prediction` 两列生成提交文件。

这个比赛给我们最大的启发是：**评价指标决定了模型该怎么设计。** QWK 罚的是「离正确答案的距离」，不是「对错」，所以把 4 分猜成 3 分几乎不亏、猜成 1 分要重罚——这直接解释了为什么「四分类 + 直接优化 QWK」优于「回归后取整」，也解释了为什么准确率在这里毫无参考价值。第二个启发是**文本任务里，特征工程的重点在「保留结构」而不是「去噪」**：给 query 和 title 的词加前缀、把三人的打分方差当成样本可信度，这些操作都不改变词本身，只是把「这个词来自哪里」「这个标签有多可信」这两层本来会丢失的信息保留了下来。最后，**先把指标实现对，再谈调参**——如果一个比赛用了不常见的指标，手写一遍、和官方库对拍，是最省时间的保险。

