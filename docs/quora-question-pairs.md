# Quora 问题对去重（Quora Question Pairs）

## 一、比赛简介

### 1.1 比赛背景

Quora 是一个大型问答社区，每月有上亿访客。人多了就会出现一个很自然的问题：**同一个问题会被不同的人反复提问**。比如「怎样才能学会游泳」和「如何快速学会游泳」，问的其实是同一件事。如果系统能自动识别出「这两个问题是重复的」，就可以把后来者直接引导到已有的回答上，既省了回答者的力气，也让提问者更快拿到答案。

这个比赛给的就是一批**问题对**，每对包含两个问题，人工标注了它们是不是重复的。我们要训练一个模型，对任意一对问题输出一个 0 到 1 之间的概率，表示「它们是重复的」有多大的可能性。

这个任务的难点在于**「重复」不是「字面相同」**：

- 有的问题用词几乎一样，但意思不同；
- 有的问题一个字都不重合，却是同一个意思，比如「怎样减肥」和「如何瘦下来」；
- 问题都很短，平均只有十来个词，模型能抓的线索很少；
- 还有大量拼写错误、网络用语、数学公式标记，噪声很重。

### 1.2 比赛目标

训练集 `train.csv` 只有六列：

- `id`：行号；
- `qid1`、`qid2`：两个问题各自的编号；
- `question1`、`question2`：两个问题的正文；
- `is_duplicate`：标签，1 表示重复，0 表示不重复。

测试集 `test.csv` 只有 `test_id`、`question1`、`question2`——**注意它没有给 qid**，这个细节后面会带来一点麻烦，也催生了一个很有名的特征。我们要为每个 `test_id` 输出一个概率值 `is_duplicate`。

### 1.3 评价标准

比赛使用 **LogLoss（对数损失）**：

$$
\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log p_i + (1-y_i)\log(1-p_i)\right]
$$

其中 $y_i$ 是真实标签（0 或 1），$p_i$ 是我们预测的概率。直白地说：**如果真实是 1，我们给的 probability 越接近 1，惩罚越小；如果真实是 0，我们给的越接近 0，惩罚越小**。分数越低越好。

LogLoss 有两个重要特点，直接决定了后面的所有策略：

1. **它看的是概率本身，不只是排序**。所以「把正例排前面」还不够，概率的绝对值也要校准到和真实比例接近；
2. **它对称地惩罚两种错误**，而且对「 confidently wrong」（很自信地答错）惩罚极重。

也正因为如此，材料里先算了一个「什么都不学、全部预测训练集均值」的基线：训练集正例率 36.92%，全填 0.369198，训练集上的 LogLoss 是 **0.658527384**。这个数字就是「零模型」的水平，任何模型都必须明显低于它才有意义。

## 二、解题过程

我们的思路是一条很典型的「**先简单后复杂**」路线：先用统计量看清数据，再做基础文本清洗，然后从「词重合比例」这个简单特征出发，逐步加入 TF-IDF 加权和「共同邻居」这个关键特征，最后用 XGBoost 输出概率。

### 2.1 数据探索（EDA）

我们首先看数据的规模和标签分布。这一步决定了后面所有的验证策略：

```python
import numpy as np
import pandas as pd

train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

print('训练集形状 =', train.shape)
print('测试集形状 =', test.shape)
print('重复对占比 = {:.2f}%'.format(train['is_duplicate'].mean() * 100))

# 把两列问题编号拼起来，看看一共有多少个不同的问题
qids = pd.Series(train['qid1'].tolist() + train['qid2'].tolist())
print('训练集中不同问题数 =', qids.nunique())
print('出现多次的问题数 =', (qids.value_counts() > 1).sum())
print('出现次数最多的问题 =', qids.value_counts().max())
```

输出是：训练集 **404,290** 对，测试集 **2,345,796** 对；重复对占 **36.92%**；训练集里共有 **537,933** 个不同问题，其中 **111,780** 个问题出现了不止一次，最多的一个问题出现了 **160 次以上**。

这里有两个发现值得单独说。

第一，**测试集比训练集大得多**。训练集文件 63.4MB，测试集 314.02MB。材料提醒说，测试集里很多行是自动生成的「凑数问题」，用来防止人工标注，真正计分的行数可能少得多。这一点在后面的验证里会造成麻烦：本地验证集的分布和排行榜的分布不一致。

第二，**训练集正例 36.92%，测试集却不是**。材料引用另一份分析估算测试集正例只有约 **16.5%**。这个差异非常关键，因为 LogLoss 对概率绝对值敏感——如果模型在 37% 正例的数据上训练，输出的平均概率就会偏高，直接搬到只有 17% 正例的测试集上，分数会被拉坏。

接着我们看文本本身的统计特征：

```python
# 把两列问题合成一个长 Series，统一按字符串处理
train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist()).astype(str)

# 字符数：均值、标准差、最大值
n_char = train_qs.apply(len)
print('字符数 mean={:.2f} std={:.2f} max={}'.format(
    n_char.mean(), n_char.std(), n_char.max()))

# 词数：按空格简单切分
n_word = train_qs.apply(lambda x: len(x.split(' ')))
print('词数 mean={:.2f} std={:.2f} max={}'.format(
    n_word.mean(), n_word.std(), n_word.max()))
```

结果是：平均每条问题 **59.82 个字符**、标准差 31.96、最长 1169 个字符；平均 **11.06 个词**、标准差 5.89、最长 237 个词。也就是说绝大多数问题在 10 个词左右，属于典型的短文本。

我们还统计了几种「表面符号」的出现比例：

```python
# 这些比例能告诉我们哪些「表面特征」值得做成特征
print('含问号 = {:.2f}%'.format(np.mean(train_qs.apply(lambda x: '?' in x)) * 100))
print('含 [math] 标记 = {:.2f}%'.format(np.mean(train_qs.apply(lambda x: '[math]' in x)) * 100))
print('含句号 = {:.2f}%'.format(np.mean(train_qs.apply(lambda x: '.' in x)) * 100))
print('首字母大写 = {:.2f}%'.format(np.mean(train_qs.apply(lambda x: x[0].isupper())) * 100))
print('含大写字母 = {:.2f}%'.format(np.mean(train_qs.apply(lambda x: any(y.isupper() for y in x))) * 100))
print('含数字 = {:.2f}%'.format(np.mean(train_qs.apply(lambda x: any(y.isdigit() for y in x))) * 100))
```

数字是：含问号 **99.87%**、含 `[math]` 标记 **0.12%**、含句号 **6.31%**、首字母大写 **99.81%**、含大写字母 **99.95%**、含数字 **11.83%**。

这组数字的价值在于**排除**：既然几乎所有问题都有问号、都以大写字母开头，那这两项就几乎没有区分度，不值得单独做特征。真正可能有信息的是句号（只有 6.31% 有）、数字（11.83%）和 `[math]` 标记。

### 2.2 数据清洗

这份数据的清洗目标不是「去空值」，而是**把文本变成可以比较的集合**。我们做了三件事。

第一步，**统一小写、去掉停用词**。比较两个问题时，「the」「is」「how」这些词几乎每条都有，它们对判断是否重复没有帮助，反而会稀释真正有信息的名词和动词：

```python
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

def to_word_set(text):
    """把一条问题变成「去掉停用词后的小写词集合」"""
    words = str(text).lower().split()
    # 用集合而不是列表：集合天然去重，也方便求交集
    return {w for w in words if w not in stops}
```

第二步，**把「问题对」当成两个集合来比较**，而不是拿两串字符串直接比。这一个小转换是整篇 writeup 的核心：一旦变成集合，就可以用「交集大小 / 并集大小」这种简单又稳定的度量，而不必依赖词序。

第三步，**处理空集合**。材料里特别提到：自动生成的问题里有些整句都是停用词，去掉停用词之后就什么都不剩了。这时如果直接做除法会得到除零错误或者 `NaN`，我们的处理是返回 0：

```python
def word_match_share(row):
    """两个问题的词重合比例：交集大小 / 总词数"""
    q1 = to_word_set(row['question1'])
    q2 = to_word_set(row['question2'])
    # 任意一边为空（全是停用词）就直接判为不重合，避免除零
    if not q1 or not q2:
        return 0.0

    shared_1 = [w for w in q1 if w in q2]
    shared_2 = [w for w in q2 if w in q1]
    # 分子是「两边共享的词各数一遍」，分母是「两边词数之和」
    # 这样得到的值在 0 到 1 之间，1 表示两个词集完全一样
    return (len(shared_1) + len(shared_2)) / (len(q1) + len(q2))

train['word_match'] = train.apply(word_match_share, axis=1, raw=True)
```

还有一步是**统一编号**。测试集没有 qid，但后面的「共同邻居」特征需要把它和训练集放进同一张图里，所以材料给测试集里的每个新问题依次分配了编号：训练集 qid 最大值是 **537,933**，测试集新问题从 **538001** 开始往后排。这一步很不起眼，却是后面那个「魔法特征」能成立的前提。

### 2.3 模型训练与超参数调优

我们先做**单个特征的强度评估**。材料用 AUC（只看看排序能力，不受概率尺度影响）来比较特征：

```python
from sklearn.metrics import roc_auc_score

print('word_match 的 AUC =', roc_auc_score(train['is_duplicate'], train['word_match']))
```

结果是 **0.780432704935**。一个只用「词重合比例」的标量特征就能到 0.78 的 AUC，说明这个方向是对的——重复问题通常共享较多实词。

**（1）TF-IDF 加权。** 上面的做法把每个词看得一样重，但「exercise」显然比「and」更能说明问题。所以我们给每个词算一个权重：**越常见的词权重越小**，越罕见的词权重越大，出现次数太少的（比如只出现一次的拼写错误）直接忽略：

```python
from collections import Counter
import numpy as np

def get_weight(count, eps=10000, min_count=2):
    """词频越高权重越低；只出现一次的噪声词权重直接给 0"""
    if count < min_count:
        return 0
    return 1 / (count + eps)

# 用整个训练集的所有词统计词频
words = ' '.join(train_qs).lower().split()
counts = Counter(words)
weights = {w: get_weight(c) for w, c in counts.items()}
```

其中 `eps=10000` 是一个平滑项，作用是**压住极罕见词的权重**。如果不加平滑，一个只出现两次的生僻词权重会比常用词高几千倍，模型会被噪声带跑。

有了权重，我们把「数词数」改成「加权求和」，得到 `tfidf_word_match`：

```python
def tfidf_word_match(row):
    """和 word_match 结构一样，只是把「个数」换成「权重之和」"""
    q1 = to_word_set(row['question1'])
    q2 = to_word_set(row['question2'])
    if not q1 or not q2:
        return 0.0

    shared = [weights.get(w, 0) for w in q1 if w in q2] + \
             [weights.get(w, 0) for w in q2 if w in q1]
    total  = [weights.get(w, 0) for w in q1] + \
             [weights.get(w, 0) for w in q2]

    # 分母为 0 说明两边的词都没见过，按不重合处理
    return np.sum(shared) / np.sum(total) if np.sum(total) > 0 else 0.0

train['tfidf_word_match'] = train.apply(tfidf_word_match, axis=1, raw=True)
```

有意思的是，这个「理论上更合理」的特征单独看 AUC 反而下降了：**0.770480229222**，比原始版本的 0.780432704935 还低一点。材料对此的判断是：它的价值不在于单独更强，而在于**提供了和原始特征不同的信息**，两个一起用可能比任何一个单独用都好。这也提醒我们，评估特征不能只看单变量指标。

**（2）重采样对齐正例率。** 前面说过训练集正例 36.92%、测试集约 16.5%，而 LogLoss 对概率绝对值敏感。所以材料做了一个很直接的处理：**把训练集的负例过采样**，让正例比例降到接近测试集的水平：

```python
pos = train[train['is_duplicate'] == 1]
neg = train[train['is_duplicate'] == 0]

p = 0.165   # 目标正例率，来自对测试集分布的估算
# scale 是「负例需要放大到原来的几倍」
scale = (len(pos) / (len(pos) + len(neg))) / p - 1
while scale > 1:
    neg = pd.concat([neg, neg])   # 每次翻倍，直到剩余倍数小于 1
    scale -= 1
neg = pd.concat([neg, neg[:int(scale * len(neg))]])   # 处理剩下的小数部分
```

跑完之后正例率变成 **0.19124366100096607**，也就是约 19.1%。为什么目标是 16.5%，实际却是 19.1%？因为负例只能成倍复制，无法精确切到任意比例，19.1% 是这种「翻倍 + 截断」能达到的最近值。这个精度对 LogLoss 已经够用了。

**（3）XGBoost 建模。** 模型本身很简单：两个特征，一棵棵决策树往上加。我们的参数是 `eta=0.02`（学习率，小一点更稳）、`max_depth=4`（树很浅，防止过拟合两个特征）、最多 400 轮、验证集 50 轮不提升就早停：

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 只放两个特征：原始重合比例 + TF-IDF 加权重合比例
X = train[['word_match', 'tfidf_word_match']].fillna(0)
y = train['is_duplicate'].values

# 注意这里是在「重采样之后」的数据上切分验证集
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=4242)

params = {
    'objective': 'binary:logistic',   # 输出 0~1 的概率
    'eval_metric': 'logloss',         # 和比赛指标一致
    'eta': 0.02,                      # 学习率
    'max_depth': 4,                   # 树深度
}

d_tr = xgb.DMatrix(X_tr, label=y_tr)
d_va = xgb.DMatrix(X_va, label=y_va)

bst = xgb.train(params, d_tr, num_boost_round=400,
                evals=[(d_tr, 'train'), (d_va, 'valid')],
                early_stopping_rounds=50, verbose_eval=10)
```

训练日志里，验证 LogLoss 从第 0 轮的 **0.683252** 一路降到第 399 轮的 **0.363904**。注意它**跑满了 400 轮都没触发早停**——说明模型还没饱和，继续加轮数或加特征都还有空间。

**（4）魔法特征：共同邻居数。** 这是这个比赛最有名的一个技巧，材料里说它能带来约 **0.045** 的提升。思路是：**把每个问题看成图上的一个节点，如果两个问题组成过一个「问题对」，就在它们之间连一条边**。那么对于一对问题 (A, B)，如果它们在图里有很多**共同邻居**，说明它们和同一批问题都「有关系」，它们自己很可能也是重复的。

```python
from collections import defaultdict

# 把训练集和测试集的问题对都放进来建图
all_pairs = pd.concat([train[['question1', 'question2']],
                       test[['question1', 'question2']]], axis=0).reset_index(drop=True)

# 邻接表：question -> 和它出现在同一个对里的所有问题
neighbors = defaultdict(set)
for q1, q2 in zip(all_pairs['question1'], all_pairs['question2']):
    neighbors[q1].add(q2)
    neighbors[q2].add(q1)     # 边是无向的，两个方向都要存

def common_neighbors(row):
    """两个问题的共同邻居数量"""
    return len(neighbors[row['question1']] & neighbors[row['question2']])

train['q1_q2_intersect'] = train.apply(common_neighbors, axis=1, raw=True)
test['q1_q2_intersect']  = test.apply(common_neighbors, axis=1, raw=True)
```

这个特征为什么有效？直白地说：**重复的问题往往「共享同一批邻居」**。如果 A 和 B 都被拿去和 C、D、E 比较过，而 A、B 又确实是一个意思，那么 A 和 B 的共同邻居就会很多。反之，两个毫不相干的问题，即使各自有很多邻居，交集通常也是空的。

不过它有一个必须说清楚的**风险**：建图时用了测试集的问题对，这是一种「直推式」（transductive）做法——测试数据的结构参与了特征计算。在离线比赛里这样做是允许的，但如果模型要上线到真实场景，新问题进来时图会变，这个特征就没法直接算了。我们在报告里会把这个限制写明。

有了这个特征，我们把三个特征一起喂给 XGBoost，树的深度仍然保持很浅。

### 2.4 模型测试与指标报告

测试阶段我们做三件事：算本地 LogLoss、生成提交、核对预测分布。

```python
from sklearn.metrics import log_loss

# 在验证集上预测概率并计算 LogLoss
p_va = bst.predict(xgb.DMatrix(X_va))
print('验证 LogLoss = {:.6f}'.format(log_loss(y_va, p_va)))

# 对测试集预测：注意测试集也要先算同样的特征
X_test = test[['word_match', 'tfidf_word_match', 'q1_q2_intersect']].fillna(0)
p_test = bst.predict(xgb.DMatrix(X_test))

sub = pd.DataFrame({'test_id': test['test_id'], 'is_duplicate': p_test})
sub.to_csv('submission.csv', index=False)
print('提交行数 =', len(sub))
```

只用两个基础特征的版本，本地验证 LogLoss 是 **0.363904**，公开排行榜是 **0.35460**。作为对照，全部预测均值的「零模型」是 0.658527384，公开榜大约 0.55。也就是说，两个简单特征就把损失砍掉了将近一半。

这里有一个很值得注意的现象：**本地验证 0.3639，排行榜 0.3546，排行榜反而更好**。正常情况下本地和线上应该接近，甚至本地略好。出现这种情况，最可能的原因就是我们前面反复提到的分布差异——本地验证集来自重采样后的训练集（正例约 19%），而测试集的正例率是另一个数。这也说明：**在这类 LogLoss 比赛里，验证集的构造方式本身就是一个需要调的超参数**，不能随便切一刀就当真。

三个特征的版本在材料里被记录为带来约 **0.045 的提升**，这也是「魔法特征」这个名字的由来——一个几乎不需要训练、只用图结构数一数邻居的特征，效果超过了很多复杂的文本模型。

最后我们做两项提交检查：

```python
# 1) 行数必须和测试集完全一致，不能多也不能少
assert len(sub) == len(test), '提交行数与测试集不一致'

# 2) 预测概率的均值应该接近我们对测试集正例率的估计
print('预测均值 = {:.4f}'.format(sub['is_duplicate'].mean()))
print('预测最小值 = {:.4f}，最大值 = {:.4f}'.format(
    sub['is_duplicate'].min(), sub['is_duplicate'].max()))
```

如果预测均值离 0.165 太远（比如 0.37，那就是没做重采样），或者出现大量 0 和 1 的极端值（LogLoss 会爆炸），基本可以断定流程有问题。**提交前看一眼概率分布**，比事后在排行榜上猜原因划算得多。

---

## 三、总结

本文中我们完成了 Quora 问题对去重的完整流程。我们首先用 EDA 摸清数据：训练集 404,290 对、测试集 2,345,796 对、训练集正例 36.92%、共 537,933 个不同问题、平均每条问题 59.82 个字符和 11.06 个词，并发现测试集正例率只有约 16.5%，与训练集严重不一致；其次我们做了面向任务的清洗，统一小写、去掉停用词、把问题转成词集合、处理全停用词导致的空集合，并给测试集新问题分配了连续编号；接着我们先做「词重合比例」这个标量特征（AUC 0.7804），再加 TF-IDF 加权版本（AUC 0.7705），然后把负例过采样到正例约 19.1% 以对齐测试分布，用 `eta=0.02`、`max_depth=4`、400 轮的 XGBoost 训练，验证 LogLoss 0.363904、公开榜 0.35460；然后我们引入了「共同邻居数」这个魔法特征，把问题对看成图上的边、统计两个问题的邻居交集大小，带来约 0.045 的提升；最后我们生成提交并检查行数与预测分布。

这个比赛给我们三点启发。第一，**在 LogLoss 比赛里，概率校准和模型精度一样重要**——同样的排序能力，概率整体偏高就会白白丢分，所以重采样不是可选项。第二，**特征工程可以借用数据结构**：把「问题对」看成图，一个数邻居交集的朴素操作，效果胜过很多复杂的文本模型。第三，**验证集要小心构造**，本地分数和排行榜不一致时，先怀疑验证集的分布，而不是急着调模型。
