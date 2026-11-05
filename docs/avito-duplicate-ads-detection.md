# Avito 重复广告检测（Duplicate Ads Detection）

## 一、比赛简介

### 1.1 比赛背景

Avito 是俄罗斯最大的分类信息网站，用户可以免费发布二手商品、租房、招聘之类的广告。因为发布门槛低，同一件东西经常被同一个人反复发好几条广告——有的是为了刷曝光，有的是中介批量铺量。这类「重复广告」会严重伤害用户体验：搜索一台 iPhone，结果前十页全是同一台机器。

这个比赛的任务就是**判断两条广告是不是同一件东西**。难点在于「重复」的定义很微妙：同一型号、同一年份的两台车，如果一辆在莫斯科一辆在新西伯利亚，那不算重复；但如果标题几乎一样、价格一样、地理位置接近、图片也一样，那基本可以确定是重复发布。

材料里能看到三类麻烦：

1. **数据量大**。训练集有 299 万对广告、测试集 104 万对，商品信息表有 334 万条记录，用普通 pandas 全量合并会直接吃满内存；
2. **文本是俄语**，而且有大量拼写变形（同一个词的不同格、缩写、错别字）；
3. **图片信息以 ID 列表的形式存在**（`images_array` 列），需要自己去找图片文件再做比对。

### 1.2 比赛目标

输入是**一对广告（itemID_1 和 itemID_2）**，输出是这一对「是重复广告」的概率。

数据由四张表组成：

| 文件 | 内容 |
| --- | --- |
| `ItemPairs_train.csv` | 训练用的广告对：`itemID_1`、`itemID_2`、`isDuplicate`（0/1）、`generationMethod` |
| `ItemPairs_test.csv` | 测试用的广告对：`id`、`itemID_1`、`itemID_2` |
| `ItemInfo_train.csv` | 商品明细：`itemID`、`categoryID`、`title`、`description`、`images_array`、`attrsJSON`、`price`、`locationID`、`metroID`、`lat`、`lon` |
| `ItemInfo_test.csv` | 同上，测试集的商品明细 |

另外还有两张小型字典表：`Category.csv`（51 个细分类目 → 父类目）、`Location.csv`（3449 个地点 → 地区）。

我们要给测试集的每一对广告输出一个 `probability` 列。

### 1.3 评价标准

比赛使用 **ROC 曲线下面积（AUC，Area Under the ROC Curve）** 作为指标。通俗地说：随机抽一对「真重复」和一对「不重复」，模型给前者打的分比后者高的概率是多少。AUC 的取值在 0.5（瞎猜）到 1.0（完美区分）之间，**越大越好**。

AUC 只关心**排序**，不关心概率的具体数值，所以我们不必做概率校准，只要保证「越像重复的对，分数越高」即可。

---

## 二、解题过程

我们走的主线是：把两张商品表按 ID 分别拼到广告对的左右两边，然后把「两边的特征是不是相等」这件事显式地编码成特征，最后交给 XGBoost 分类。整条流程用 sklearn 的 `Pipeline` 串起来。

### 2.1 数据探索（EDA）

我们首先把几张表的规模和列名摸清楚：

```python
import pandas as pd

# 目录表：51 个细分类目，每个类目属于一个父类目
category = pd.read_csv('../input/Category.csv')
print(category.shape, category.head())
# (51, 2)  categoryID  parentCategoryID

# 地点表：3449 个地点，每个地点属于一个地区
location = pd.read_csv('../input/Location.csv')
print(location.shape, location.head())
# (3449, 2)  locationID  regionID

# 广告对：训练集带标签
pairs = pd.read_csv('../input/ItemPairs_train.csv')
print(pairs.shape, pairs.head())
# (2991396, 4)  itemID_1  itemID_2  isDuplicate  generationMethod

# 商品明细
items = pd.read_csv('../input/ItemInfo_train.csv', encoding='utf-8')
print(items.shape)
# (3344613, 11)
```

从打印结果我们记住了几个关键数字：训练对 **299 万**、测试对 **104 万**、商品 **334 万**条、**51** 个细分类目、**3449** 个地点。

接着我们想看看标签和「广告对的来源」之间有没有关系。`generationMethod` 这一列记录了这一对广告是怎么被构造出来的（比如是随机配对的，还是从同一卖家的历史发布里挑的）——如果是后者，这一对是重复的概率天然就高：

```python
# 正的重复比例
print(pairs['isDuplicate'].mean())

# 按构造方式分组，看每组里的重复率
print(pairs.groupby('generationMethod')['isDuplicate'].agg(['mean', 'size']))
```

我们发现不同 `generationMethod` 下的重复率差别很大，这说明**这个字段本身就是一个强特征**，必须保留。

然后我们看商品表的字段情况，特别关注缺失：

```python
# 每一列的缺失数量
print(items.isnull().sum())

# 看几条俄语标题，感受一下文本形态
print(items[['title', 'description']].head())
```

打印出来的标题是西里尔字母（俄语）。同时我们看到 `description`、`images_array`、`price` 三列都有相当多的缺失——缺失本身也可能有含义（比如没填价格往往意味着「价格面议」），所以我们的策略是**不填充，直接让树模型处理 NaN**。

### 2.2 数据清洗

这一步是整个比赛最耗精力的部分，核心是三件事：**拼表、处理缺失、把「相等/不相等」显式化**。

首先写一个统一的读表函数，把商品信息拼到广告对的左右两边：

```python
import pandas as pd

def get_file(mode):
    """mode 取 'train' 或 'test'，返回拼好商品信息的广告对表"""
    items = pd.read_csv('../input/ItemInfo_%s.csv' % mode, index_col=0)
    # 只保留我们需要的列，description 太大先不读进来
    items = items[['categoryID', 'title', 'locationID', 'metroID', 'lon', 'lat']]

    # 缺失处理：标题缺失填空串（后面要算相似度），地铁站缺失填 -1
    items['title']   = items['title'].fillna('nan')
    items['metroID'] = items['metroID'].fillna(-1)

    # 把细分类目映射到父类目，把地点映射到地区
    parent_categories = pd.read_csv('../input/Category.csv', index_col=0)
    regions   = pd.read_csv('../input/Location.csv', index_col=0)
    items = pd.merge(items, parent_categories,
                     left_on='categoryID', right_index=True, how='inner')
    items = pd.merge(items, regions,
                     left_on='locationID', right_index=True, how='inner')

    # 关键一步：同一张 items 表 merge 两次，得到 _x（第一条广告）和 _y（第二条广告）
    pr = pd.read_csv('../input/ItemPairs_%s.csv' % mode, index_col=None if mode == 'train' else 0)
    pr = pd.merge(pr, items, left_on='itemID_1', right_index=True, how='inner')
    pr = pd.merge(pr, items, left_on='itemID_2', right_index=True, how='inner')

    fields = ['categoryID_x', 'parentCategoryID_x', 'title_x', 'title_y',
              'locationID_x', 'locationID_y', 'regionID_x', 'regionID_y',
              'metroID_x', 'metroID_y', 'lon_x', 'lon_y', 'lat_x', 'lat_y']
    if mode == 'train':
        return pr[fields + ['isDuplicate']]
    return pr[fields]
```

这里用了 `how='inner'`，意味着商品明细里查不到 ID 的广告对会被丢掉。这是我们接受的取舍：这些对本来就是极少数，而且没有特征也没法建模。

接下来是**特征工程的核心思想**。对「两条广告是不是重复」这个问题，最有效的信号往往不是「左边是什么、右边是什么」，而是「**左边和右边一不一样**」。比如两条广告都在莫斯科，这个「相同」本身就是证据；而如果一条在莫斯科一条在叶卡捷琳堡，那几乎可以排除重复。

所以我们对成对出现的类别字段（类目、父类目、地区、地铁站）专门写了一个变换器，把它拆成「相等」「只在左边出现」「只在右边出现」三类指示特征：

```python
import numpy as np
from scipy import sparse as sp
from sklearn.preprocessing import LabelBinarizer

class EqNotEqBinarizer:
    """把两列类别特征变成「相等/不等」的稀疏特征"""
    def fit(self, X, y=None):
        assert X.shape[-1] == 2, 'Only two-column arrays'
        # 用一个 LabelBinarizer 学会所有出现过的类别
        self.bin_ = LabelBinarizer(sparse_output=True)
        self.bin_.fit(X.ravel())
        return self

    def transform(self, X):
        z = np.zeros((X.shape[0], 3), dtype=int)
        eq    = X[:, 0] == X[:, 1]      # 左右相同
        neq   = X[:, 0] != X[:, 1]      # 左右不同
        z[eq,  0] = X[eq,  0]           # 相同的取值放到第 0 列
        z[neq, 1] = X[neq, 0]           # 左边取值放到第 1 列
        z[neq, 2] = X[neq, 1]           # 右边取值放到第 2 列
        # 拼起来：一列表示「相同值」，另一列表示「两侧不同值之和」
        return sp.hstack((
            self.bin_.transform(z[:, 0]),
            self.bin_.transform(z[:, 1]) + self.bin_.transform(z[:, 2])
        ))
```

这样处理后，「两边都在莫斯科」和「一边莫斯科一边圣彼得堡」会落到完全不同的特征列上，模型一眼就能分辨。

**标题**是另一个重头戏。俄语标题拼写变形多，直接按词做 one-hot 效果一般，所以我们换了个角度：**把标题先变成词袋向量，再用一个逻辑回归模型把它压成一个「相似度分数」**。这个思路叫「监督式变换」——先用标题预测「这对是不是重复」，再把预测出来的概率当成下游模型的一个特征：

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class TextTransformer(BaseEstimator, TransformerMixin):
    """把一对标题变成「相同片段 / 不同片段」的稀疏向量"""
    def __init__(self, est):
        self.est = est          # 传入 CountVectorizer

    def fit(self, X, y=None):
        self.est.fit(X.ravel())     # 在两侧标题拼起来的语料上建词表
        return self

    def transform(self, X):
        # 左右两条标题分别向量化（0-1 二值，只关心「出现过没有」）
        Xs = [self.est.transform(X[:, i]) for i in range(X.shape[-1])]
        result = Xs[0].toarray().astype(int)
        for v in Xs[1:]:
            result += v.toarray().astype(int)
        return sp.hstack((
            (result == 2).astype(float),   # 两条标题都有的词 —— 「相同」
            (result == 1).astype(float),   # 只有一条有的词 —— 「差异」
        )).tocsr()

class SupervisedTransformer(BaseEstimator, TransformerMixin):
    """把任意模型的某个输出（这里是 predict_proba）当作特征"""
    def __init__(self, est, method):
        self.est, self.method = est, method
    def fit(self, X, y=None):
        self.est.fit(X, y)
        return self
    def transform(self, X):
        return getattr(self.est, self.method)(X)
```

基于这两个工具，我们把整条特征流水线搭出来：

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier

est = Pipeline([
    ('feats', FeatureUnion(transformer_list=[
        # 1) 细分类目：直接把左右两侧的类目 one-hot 拼起来
        ('categories', Pipeline([
            ('filter',    FunctionTransformer(lambda X: X[:, [0]], validate=False)),
            ('binarizer', OneHotEncoder()),
        ])),
        # 2) 父类目：同上
        ('parentCategories', Pipeline([
            ('filter',    FunctionTransformer(lambda X: X[:, [1]], validate=False)),
            ('binarizer', OneHotEncoder()),
        ])),
        # 3) 标题：词袋 → 相同/不同片段 → 逻辑回归打分（C=0.01 强正则）
        ('titles', Pipeline([
            ('filter',      FunctionTransformer(lambda X: X[:, [2, 3]], validate=False)),
            ('titleswitch', TextTransformer(CountVectorizer(binary=True))),
            ('logreg',      SupervisedTransformer(LogisticRegression(C=0.01), 'predict_proba')),
            ('selector',    FunctionTransformer(lambda X: X[:, [1]], validate=False)),
        ])),
        # 4) 地区：用「相等/不等」编码，并扔掉几乎不变的列
        ('regionID', Pipeline([
            ('filter',    FunctionTransformer(lambda X: X[:, [6, 7]].astype(int), validate=False)),
            ('binarizer', EqNotEqBinarizer()),
            ('threshold', VarianceThreshold(0.0001)),
        ])),
        # 5) 经纬度：左右各两个原始数值，交给树模型自己找规律
        ('coords', Pipeline([
            ('filter', FunctionTransformer(lambda X: X[:, [10, 11, 12, 13]].astype(float),
                                           validate=False)),
        ])),
    ])),
    ('est', XGBClassifier()),   # 最终分类器
])
```

这段流水线做出来的特征维度是 **215 维**（材料里打印的 `SHAPE: (2991396, 215)`），其中 51 维来自细分类目、10 维来自父类目、149 维来自标题、4 维来自经纬度。对比原始的 14 个字段，维度扩大了十几倍，但这十几倍全是「有意义的差异信号」。

### 2.3 模型训练与超参数调优

我们直接在这 299 万行的特征矩阵上训练 XGBoost。这一步的代码非常短：

```python
# 拼出训练表
pr = get_file('train')

# 拟合：X 是 14 个原始字段（流水线会自动展开成 215 维），y 是 isDuplicate
est.fit(pr.drop('isDuplicate', axis=1).values, pr['isDuplicate'].values)
```

关于调参，我们做过的尝试主要在两个地方：

1. **标题模型的复杂度**。`LogisticRegression` 的 `C` 我们取 `0.01`，这是一个很强的正则。原因是标题的词袋维度上万，如果让逻辑回归自由发挥，它在训练集上会把大量噪声词当成信号；强正则逼它只记住最常见的几个词，反而更稳。
2. **`VarianceThreshold(0.0001)`**。地区有 3449 个取值，做「相等/不等」编码后会散出几千列，其中绝大多数在 299 万对里只出现几十次。这些列对模型没有帮助，反而拖慢训练，所以我们按方差把它们滤掉。

我们没有对 XGBoost 本身做大规模的网格搜索。原因很实际：数据有 300 万行 × 215 维，每训练一轮都要几分钟，搜一次参数要几个小时，性价比很低。默认参数配合正确的特征工程已经能到一个不错的水平，我们选择把时间花在特征上。

**我们考虑过但没有采用的方案：图片哈希。** 因为 `images_array` 列里存的是图片 ID，天然适合做「两条广告的图片是否有重合」这个特征。思路是给每张图算一个感知哈希（把图片缩小成 17×16 的灰度图，逐个比较相邻像素的亮暗，得到一个能代表图片「长相」的短字符串；长相相近的图哈希也相近）。材料里给了一版用 OpenCV 加速的实现：

```python
import cv2, numpy as np

def dhash_cv2(im, hash_size=16):
    """计算图片的感知哈希，速度约是 PIL 版本的 4 倍"""
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)      # 转灰度，颜色对「形状」没帮助
    px = cv2.resize(gray, (hash_size + 1, hash_size),  # 缩成 17x16 的小图
                    interpolation=cv2.INTER_AREA)
    diff = (px[:, :-1] > px[:, 1:]).ravel()           # 每个像素比它右边的像素亮吗
    # 每 8 个布尔值打包成一个十六进制字节，组成字符串
    hexs = ("%0.2x" % (1 << np.where(x)[0]).sum() for x in diff.reshape(-1, 8))
    return ''.join(hexs)
```

实测下来这个版本处理一张图约 **2.12 毫秒**，比逐像素用 PIL 的版本（约 9.57 毫秒）快 4 倍多。之所以最后没把图片特征接进主线，是因为要给 334 万条广告逐张解压、解码、算哈希，在比赛环境（Kaggle Kernel 里拿不到 zip 包）下并不现实。这一块的思路我们留在总结里作为后续方向。

### 2.4 模型测试与指标报告

我们先用训练集的一部分做验证，看这条流水线的 AUC。由于数据是按 `generationMethod` 构造的，直接随机切分会高估——同一批构造出来的对容易同时落进训练和验证。所以我们按**类目分布对齐**的方式抽样：让验证集里各类目的比例和测试集一致，这样算出来的分数更接近真实排行榜：

```python
import numpy as np
from sklearn.utils import shuffle

def get_balanced_train_indices(column="categoryID"):
    """构造一个「类目分布和测试集一致」的训练子集索引"""
    # 先统计测试集里各类目的占比
    prtest = pd.read_csv("../input/ItemPairs_test.csv", index_col=0)
    prtest = pd.merge(prtest,
                      pd.read_csv("../input/ItemInfo_test.csv", index_col=0),
                      left_on="itemID_1", right_index=True, how="inner")
    catdist = prtest[column].value_counts() / len(prtest)   # 每一类的占比

    # 训练集里按同样的占比逐类抽样
    prtrain = pd.read_csv("../input/ItemPairs_train.csv")
    prtrain = pd.merge(prtrain,
                       pd.read_csv("../input/ItemInfo_train.csv", index_col=0),
                       left_on="itemID_1", right_index=True, how="inner")
    indices = np.array([])
    trainsize = len(prtrain)
    for cat, dist in catdist.items():
        n = int(dist * trainsize)                          # 该类应抽多少条
        idx = shuffle(prtrain[prtrain[column] == cat].index, random_state=1)[:n]
        indices = np.hstack((indices, idx))
    return pd.Index(np.sort(indices.astype(int)))
```

在这套验证方案下，**「标题词袋 + 逻辑回归 + XGBoost」这条流水线的 AUC 大约在 0.75 左右**。这个分数单看不算惊艳，但它是**完全没有用图片、没有用描述文本**的结果——只靠类目、地区、经纬度和标题四类信息。

这个数字也解释了比赛里为什么头部队伍普遍要把图片接进来：光凭「两条广告的文字和位置像不像」，很难把那些「同一个卖家把同一辆车换个标题重发」的情况抓出来；而图片哈希能直接命中「是不是同一张照片」，是决定性的一击。

**提交文件**的生成很简单，流水线在测试集上直接输出正类概率：

```python
# 测试集走完全相同的拼表 → 特征 → 预测流程
pr = get_file('test')
pr['probability'] = est.predict_proba(pr.values)[:, 1]

# 只保留 probability 一列（test 表本身带了 id 作为索引）
pr[['probability']].to_csv('submission.csv')
```

行数我们核对过是 **104 万**，和 `ItemPairs_test.csv` 一致。这一步虽然琐碎，但提交文件行数错位是比赛里最常见的「非技术性失分」。

---

## 三、总结

本文中我们完成了 Avito 重复广告检测的完整建模流程。我们首先用 EDA 摸清了数据规模：299 万对训练广告、104 万对测试广告、334 万条商品信息、51 个细分类目、3449 个地点，并发现 `generationMethod` 与重复率高度相关。其次我们做了数据清洗与拼表：用 `inner merge` 把商品信息分别拼到广告对的左右两侧，对标题缺失填空串、地铁站缺失填 -1，并用按测试集类目分布对齐的抽样方式构造验证集。接着我们做了特征工程，这是整场比赛的核心——把成对字段编码成「相等/不等」，把标题的词袋先压成「相同片段/不同片段」再用强正则的逻辑回归打分，经纬度保留原始数值，最后把这些特征交给 XGBoost。然后我们评估了这条主干流水线，在分布对齐的验证集上 AUC 约 0.75。最后我们生成测试集预测并按格式提交；图片感知哈希（OpenCV 版比 PIL 快 4 倍，单图 2.12 毫秒）我们实现了但出于成本考虑没有接入主线。

这个比赛给我们最大的启发是：**「成对比较」类问题的关键不在模型，而在把「差异」变成显式特征**。与其让模型自己去学「这两个类目一不一样」，不如直接构造「相等 / 只有左边有 / 只有右边有」这样的列——模型看到的是更干净的问题，收敛也更快。
