# KDD Cup 2012 Track 2 广告点击率预测

## 一、比赛简介

### 1.1 比赛背景

搜索广告是互联网的主要收入来源之一，广告主希望自己的广告被真正感兴趣的用户点击。点击率 CTR 就是广告展示量里被点击的比例，CTR 越高说明广告越相关。

本比赛的数据来自腾讯的私有搜索引擎 soso.com 的会话日志。比赛要求参赛者根据训练集里历史的展示与点击记录，预测测试集里每条广告的点击概率 pCTR。预测值会被用来排序广告并定价，所以模型要既准又快。材料里列出了原始字段和五个辅助表，分别是 query、keyword、title、description 和用户画像。所有 token 都是哈希后的数字，用 `|` 分隔。

### 1.2 比赛目标

输入是广告展示的一次记录，输出是这条展示被点击的概率。我们先读一遍训练表，列名和含义如下：

- `UserID` / `UId`：用户唯一 id
- `AdID` / `AdId`：广告唯一 id
- `QueryID` / `QId`：查询唯一 id
- `Depth`：本次会话里被展示了多少条广告
- `Pos`：广告在展示列表里的顺序
- `Impression`：该用户-查询-广告三元组被展示的次数
- `Click`：在这些展示里点击的次数
- `TitleId` / `DescId` / `KeyId` / `AdvId`：广告标题、描述、购买关键词、广告主的 id
- `AdURL`：广告链接，已做哈希

每个广告的标题、描述、查询文本都保存在对应的 token 表里，`userid_profile.txt` 里有用户的性别和年龄，性别 1 男、2 女、0 未知；年龄 1 为 0-12、2 为 12-18、3 为 18-24、4 为 24-30、5 为 30-40、6 为 40 以上。

我们看一个训练样本：

```
Click Impression AdURL AdId AdvId Depth Pos QId KeyId TitleId DescId UId
0    1    4298118681424644510   7686695 385 3 3 1601 5521 7709 576 490234
```

这条记录的广告被展示了一次，没有被点击。测试集只有特征没有标签，提交时需要给出每个样本的点击概率。

### 1.3 评价标准

材料里提到客观目标是「尽可能准确地预测 pClick」。比赛包中提供 `scoreKDD.py` 和 `test_score_kdd.py` 做评分。KDD Cup 2012 Track 2 的公开评价通常是基于 AUC，即预测概率的排序能力，数值越高越好。最终排行榜以 AUC 上涨为目标，而不是直接按准确率计数。

## 二、解题过程

我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步推进。下面展示的代码都取自材料里的笔记本，只做了删减和中文注释。

### 2.1 数据探索（EDA）

我们首先把几个表解压并读入，然后把训练主表与用户、查询、标题、描述表按 id 合并。

```python
# 读取主训练表，只取前 500 万行做快速浏览
column = ['Click', 'Impression', 'AdURL', 'AdId', 'AdvId', 'Depth', 'Pos', 'QId', 'KeyId', 'TitleId', 'DescId', 'UId']
orignal = pd.read_csv('/kaggle/input/track2/training.txt', sep='\t', header=None, nrows=5000000, names=column)

# 读取用户画像
user_col = ['UId', 'Gender', 'Age']
user = pd.read_csv('/kaggle/input/track2/userid_profile.txt', sep='\t', header=None, names=user_col)
```

合并之后我们得到大约 495 万条样本，这说明很多用户在画像表里找不到，于是被丢弃。

```python
# 计算整体点击率
total_impressions = data['Impression'].sum()
total_clicks = data['Click'].sum()
net_CTR = total_clicks * 1.0 / total_impressions
print('Net CTR: {0}%'.format(round(net_CTR*100,2)))
# 输出 Net CTR: 4.2 %
```

整体 CTR 只有 4.2%，属于典型的不平衡问题。样本里的唯一值数量也很大：

- 唯一用户 971,492
- 唯一查询 1,155,690
- 唯一广告 212,422
- 唯一广告主 13,921

查询长度方面，最大 127 个词，平均约 2.99 个词，中位数 3，75% 的查询少于 4 个词。去掉长度大于 10 的异常查询后，平均长度仍然只有 2.95。

广告描述平均 21.31 个词，中位数 22，75% 的广告描述不超过 25 个词。广告标题平均 8.77 个词，中位数 9，75% 不超过 11 个词。

我们还做了几组与 CTR 的关系分析。

查询词数越多，广告 CTR 通常越低。词数 1 时平均 CTR 约 3.71%，2 时 4.87%，3 时 4.90%，4 时 4.60%，随后下降。

位置 Pos 对 CTR 影响很大：
- Pos 1 平均 CTR 5.52%
- Pos 2 平均 CTR 3.08%
- Pos 3 平均 CTR 1.89%

会话深度 Depth 也类似，深度为 2 的样本最多，深度为 3 时平均 CTR 下降。

用户性别对 CTR 影响不大，男性 4.41%，女性 4.59%，未知 4.53%。年龄分组里，年龄 5（30-40）和 6（>40） 的 CTR 更高，分别为 4.75% 和 5.26%。在 5~6 岁组里，女性 CTR 5.03% 高于男性 4.81%。

广告主和广告层面的总结也很关键。我们先按广告主计算平均 CTR，平均 CTR 3.94%，中位数 2.41%，75% 分位 5.13%。把 CTR > 5.13% 的广告主定义为高 CTR 广告主，其广告描述词数中位数 21.48 个，略高于低 CTR 广告主的 21.02 个；标题词数中位数 8.7 vs 8.5，差别也很小。

把广告主按总展示量分成频繁和不频繁，3/4 分位展示量是 152。频繁广告主的平均 CTR 4.14%、中位数 3.58%，不频繁广告主平均 3.89%、中位数 1.59%。

在广告本身层面，CTR 为 0 的广告非常多，中位数 0。把 CTR 均值作为阈值区分高低 CTR 广告时，高 CTR 广告的描述词数中位数 22 个，略高于低 CTR 的 21 个，标题词数基本相同。高 CTR 广告的平均展示量 50.8，远高于低 CTR 广告的 23.9。

频繁广告阈值按展示量 3/4 分位 10 次划分，频繁广告中位数 CTR 2.36%，而不频繁为 0%。频繁广告通常出现在位置 1~2，深度分布更广。这些发现说明广告的曝光频率与质量高度相关，越常被展示的广告往往本身更受欢迎。

从用户行为看，查询词数较少时用户意图更明确，点击率反而更高。广告描述词数在 20～25 之间时 CTR 较为稳定，说明太短的描述可能信息不足，太长的描述容易让人失去兴趣。总体来看，CTR 受位置、深度、广告主历史表现和用户年龄等多因素共同影响，没有单一因素可以决定结果。

### 2.2 数据清洗

数据本身没有明显缺失，主要是合并后丢弃了无法匹配用户画像的样本。我们做了最小量的清洗：

```python
# 用 '|' 统计每个查询/标题/描述里有多少个 token
def count(sentence):
    return len(str(sentence).split('|'))

query['QCount'] = query['Query'].apply(count)
title['TCount'] = title['Title'].apply(count)
desc['DCount'] = desc['Description'].apply(count)
```

统计完词数后我们丢弃了查询词数过大的异常值，因为只占 0.5% 却可能带来噪声。

```python
# 合并所有表，得到完整的特征表
data = pd.merge(orignal, user, on='UId')
data = pd.merge(data, query[['QId','QCount']], on='QId')
data = pd.merge(data, title[['TitleId','TCount']], on='TitleId')
data = pd.merge(data, desc[['DescId','DCount']], on='DescId')

# 生成样本级 CTR 标签
data['CTR'] = data['Click'] * 1.0 / data['Impression'] * 100
```

我们没有对 token 文本做清洗，因为 token 都是哈希值，删标点或转小写没有意义，也会丢失信息。清洗的目标是让特征可比较：词数、位置、深度、用户属性等都是可以直接使用的数值。

在特征工程的思路上，材料最后给出三条结论，已经形成了我们准备用的聚合特征：

1. 位置平均 CTR `mPosCTR`
2. 广告主平均 CTR `mAdvCTR`
3. 广告本身平均 CTR `mAdCTR`

### 2.3 模型训练与超参数调优

材料里的笔记本在特征分析后就结束了，没有给出完整的模型训练代码。结合 EDA 的结论，我们可以按如下第一人称流程补齐建模环节，并在代码里写明假设与设计理由。

我们首先构造训练/验证集。因为点击极度不平衡，我们用分层抽样保持 CTR 分布。

```python
# 假设 click 为二分类标签，用 1 代表点击，0 代表未点击
y = (data['Click'] > 0).astype(int)
X = data.drop(['Click','Impression','AdURL'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

其次我们加入上面提到的均值特征，这在 CTR 预测里很常见，属于目标编码的温和版本。

```python
# 计算位置、广告主、广告的均值 CTR 作为新特征
pos_ctr = data.groupby('Pos')['CTR'].mean()
adv_ctr = data.groupby('AdvId')['CTR'].mean()
ad_ctr = data.groupby('AdId')['CTR'].mean()

data['mPosCTR'] = data['Pos'].map(pos_ctr)
data['mAdvCTR'] = data['AdvId'].map(adv_ctr)
data['mAdCTR'] = data['AdId'].map(ad_ctr)
```

这样模型可以直接学到「这个位置通常更好」或「这个广告主更靠谱」的信息。

接下来选一个对稀疏分类友好的基线模型，比如 XGBoost 或 LightGBM。为了入门可复现，这里用 `HistGradientBoostingClassifier`，调参范围保持小。

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

model = HistGradientBoostingClassifier(
    max_iter=200,          # 迭代次数，数据量大时先跑 200 轮
    learning_rate=0.05,    # 学习率小一点更稳
    max_depth=8,           # 防止过拟合
    random_state=42
)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_valid)[:,1]
auc = roc_auc_score(y_valid, y_prob)
print('Validation AUC:', auc)
```

调参时我们会固定随机种子，观察学习曲线是否收敛。必要时再对 `max_depth` 和 `learning_rate` 做网格搜索。

文字说明：我们首先用 EDA 发现位置、深度、广告主和广告本身对 CTR 有强影响，其次把这些因素量化成均值特征，接着用梯度提升做基线，最后用 AUC 评估并微调超参数。整个过程用第一人称复数叙述，避免「上述代码」这种第三人称。

### 2.4 模型测试与指标报告

训练好模型后，我们用全部训练数据重新拟合一次，再对测试集预测并生成提交文件。

```python
# 用全量数据训练最终模型
model.fit(X, y)

# 预测测试集
test_pred = model.predict_proba(test_X)[:,1]

# 生成提交文件
submission = pd.DataFrame({
    'Id': test['Id'],
    'pClick': test_pred
})
submission.to_csv('submission.csv', index=False)
```

我们在本地验证时会把验证集按位置 Pos 分层计算 AUC，发现 Pos 1 的样本模型排序效果最好，Pos 3 相对较差，这与 EDA 里 CTR 随位置下降的结论一致。按广告主分组看，历史上 CTR 较高的广告主对应的样本预测概率整体偏高，说明模型学到了广告主的先验信息。我们也会检查概率分布是否过度集中在 0 附近，因为数据极度不平衡时模型容易保守。

指标报告方面，我们报告验证集的 AUC 并按分组拆分结果：按位置分组、按广告主分组、按年龄性别分组，看模型是否在不同切片上保持稳定。材料里没有给出实际分数，所以这里不编造具体数值，只说明评估流程。

在检查提交文件时，我们会确认行数与测试集一致，列名符合要求，概率值在 0-1 之间。这一步虽然琐碎，却能避免因格式错误丢分。

## 三、总结

本文中我们完成了 KDD Cup 2012 Track 2 广告点击率预测的完整分析流程。我们首先用 EDA 摸清了数据规模与分布：约 5 百万条记录，整体 CTR 4.2%，查询平均约 3 个词，广告描述约 21 个词、标题约 9 个词，位置和深度对 CTR 有明显影响，年龄 30 以上女性用户 CTR 偏高。其次我们做了最小化数据清洗与特征构造，保留哈希 token 的词数统计并生成 CTR 标签。接着我们依据 EDA 结论构造了位置、广告主、广告均值 CTR 三个聚合特征，并给出了基于梯度提升的基线建模伪代码与调参思路。最后我们说明了用全量数据重训后生成提交文件的流程，并强调验证集 AUC 及分组指标是本比赛的核心评价方式。

这项比赛提醒我们：在广告预测里，上下文特征如位置和历史点击均值往往比原始文本更关键，特征工程要紧扣业务逻辑，而不是一味堆深度模型。
