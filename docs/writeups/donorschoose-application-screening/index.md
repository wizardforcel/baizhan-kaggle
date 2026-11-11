# DonorsChoose 项目申请书筛选（Application Screening）

## 一、比赛简介

### 1.1 比赛背景

DonorsChoose.org 是一个美国的公益众筹网站，公立学校的老师可以在上面发布「项目申请书」——比如「我想给班里的孩子买一套科学实验器材」「我们需要一批课外读物」——写清楚用途和预算，然后由爱心人士捐款支持。

网站每天会收到大量申请。为了保护捐款人的信任，平台需要先筛掉那些看上去不靠谱、描述太含糊或者不符合规范的申请，只把合格的申请书展示给捐款人。这个「谁来审」的工作如果全靠人工，既慢又累。这个比赛的目的，就是让算法**自动判断一份申请书会不会被通过**。

这件事的难点在于：判断标准没有写在纸上。一段文字到底「像不像会被批准」，取决于措辞、语气、老师的经验、申请的物资、学校所在地等等，是典型的**文本分类 + 结构化特征**混合问题。而且标签是历史审核结果，带有人的主观性，模型不可能做到 100% 准。

### 1.2 比赛目标

数据分三个文件：

- `train.csv`：每行一个申请书，包含 `id`（编号）、`teacher_id`（老师编号）、`teacher_prefix`（称谓）、`school_state`（州）、`project_submitted_datetime`（提交时间）、`project_grade_category`（年级）、`project_subject_categories`（科目大类）、`project_subject_subcategories`（科目小类）、`project_title`（标题）、`project_resource_summary`（物资说明）、`teacher_number_of_previously_posted_projects`（老师之前发过几次申请）、`project_essay_1`～`project_essay_4`（几段申请作文），以及目标列 `project_is_approved`（1 表示通过，0 表示没通过）。
- `resources.csv`：每个申请书申请的物资明细，有 `id`、`description`（物品名）、`quantity`（数量）、`price`（单价）。
- `test.csv`：和训练集同样的列，但没有 `project_is_approved`。

我们要给每个测试申请输出一个 **0～1 之间的概率**，表示它会被批准的可能性。

### 1.3 评价标准

比赛使用 **ROC 曲线下面积（AUC，Area Under the ROC Curve）**。

AUC 衡量的是「排序能力」：随便挑一个真实通过的项目和一个真实没通过的项目，模型给通过的那个打分更高的概率有多大。全部排对是 1.0，瞎猜是 0.5。它只关心**谁排在谁前面**，不关心具体概率值，因此对类别不平衡不敏感——这个比赛里通过的申请占多数，如果看准确率，「全猜通过」就能很高，但 AUC 仍会是 0.5，一眼就能识破。AUC 越高越好。

---

## 二、解题过程

我们的主线思路分两条：一条是**简单可靠的基线条**——把文本变成 TF-IDF 向量、把资源信息聚合成数字特征，喂给 LightGBM；另一条是**冠军方案**——用多种深度网络（GRU、LSTM、胶囊网络）加梯度提升树做集成。我们会以基线为主线讲清楚每一步，最后再介绍冠军方案的集成思路。

### 2.1 数据探索（EDA）

我们首先把三个表读进来，并把物资明细按申请书编号汇总。因为 `resources.csv` 是一行一件物品，一个申请可能对应十几行，必须聚合成「一个申请一行」才能和主表合并：

```python
import pandas as pd
import numpy as np

# 读入三个文件；提交时间要解析成日期类型，后面按时间做特征
train = pd.read_csv("../input/train.csv", parse_dates=['project_submitted_datetime'])
resources = pd.read_csv("../input/resources.csv")
test = pd.read_csv("../input/test.csv", parse_dates=['project_submitted_datetime'])

# 每件物资先算出小计：数量 × 单价
resources['cost'] = resources['quantity'] * resources['price']

# 按申请书 id 聚合，得到四个「整份申请」级别的数字
resources_aggregated = resources.groupby('id').agg({
    'description': ['nunique'],       # 一共申请了多少种不同物品
    'quantity': ['sum'],              # 物品总件数
    'cost': ['mean', 'sum']           # 平均单价、总花费
})
resources_aggregated.columns = ['unique_items', 'total_quantity', 'mean_cost', 'total_cost']
resources_aggregated.reset_index(inplace=True)

# 左连接回主表
train = pd.merge(train, resources_aggregated, how='left', on='id')
test = pd.merge(test, resources_aggregated, how='left', on='id')
```

合并完之后，我们先看目标分布和「老师经验」这个特征：

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 通过率大概是多少
print(train['project_is_approved'].mean())

# 把「之前发过几次申请」分成三档：0~2 次、2~10 次、10 次以上
bins = [0., 2., 10.]
train["experience"] = np.digitize(train['teacher_number_of_previously_posted_projects'], bins=bins)

# 画出每一档的通过率
sns.pointplot(x="experience", y="project_is_approved", data=train, ci=95)
plt.ylabel('Approval rate')
plt.show()
```

这张图说明**老师经验和通过率有关系**：第一次发申请的新手老师和发过很多次的老手，通过率并不一样。原因可能是老手更懂平台喜欢什么样的措辞。

接着我们看四个物资特征。这里有个技巧：原始数值的分布又长又散（少数申请要几万美元），直接画图看不出趋势，所以我们**按分位数把每个特征切成 6 桶**，再看每桶的通过率：

```python
def get_quantile_based_boundaries(feature_values, num_buckets):
    """按分位数切桶，保证每桶样本数差不多"""
    boundaries = np.arange(1.0, num_buckets) / num_buckets
    quantiles = feature_values.quantile(boundaries)
    return [quantiles[q] for q in quantiles.keys()]

# 以 unique_items 为例：切成 6 桶后看通过率
quantiles = get_quantile_based_boundaries(train['unique_items'], 6)
train["unique_items_binned"] = np.digitize(train['unique_items'], bins=quantiles)
sns.pointplot(x="unique_items_binned", y="project_is_approved", data=train, ci=95)
plt.show()
```

四个特征的趋势很清楚：

- 申请的物品种类越多，通过率**越低**；
- 物品总件数越多，通过率**越低**；
- **平均单价越高，通过率越高**；
- 总花费越高，通过率**越低**。

第三条最有意思：与其申请「一大堆便宜的东西」，不如申请「少数几样好一点、贵一点的东西」，前者更容易被拒。这其实是一个隐含的审核偏好，靠人肉想是想不出来的，必须让数据说话。

### 2.2 数据清洗

文本数据里有几处必须处理的「历史包袱」：

**第一，早期和中期的表结构变了。** 2016 年 5 月 7 日之前的申请有四段作文，之后的只有两段。如果不处理，同一个字段在不同时期含义不一样，模型会学乱。我们的做法是把早期的四段合并成两段，让所有样本的字段含义统一：

```python
import datetime

# 对 2016-05-07 之前的样本，把 essay_3/4 并进 essay_1/2
mask = train.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7)
train.loc[mask, 'project_essay_1'] = train.loc[mask, 'project_essay_1'] + ' ' + train.loc[mask, 'project_essay_2']
train.loc[mask, 'project_essay_2'] = train.loc[mask, 'project_essay_3'] + ' ' + train.loc[mask, 'project_essay_4']
train.drop(['project_essay_3', 'project_essay_4'], axis=1, inplace=True)
# 测试集做完全一样的处理
mask_t = test.project_submitted_datetime.dt.date < datetime.date(2016, 5, 7)
test.loc[mask_t, 'project_essay_1'] = test.loc[mask_t, 'project_essay_1'] + ' ' + test.loc[mask_t, 'project_essay_2']
test.loc[mask_t, 'project_essay_2'] = test.loc[mask_t, 'project_essay_3'] + ' ' + test.loc[mask_t, 'project_essay_4']
test.drop(['project_essay_3', 'project_essay_4'], axis=1, inplace=True)
```

**第二，文本里混着转义字符。** 从网页导出的文本会带 `\r`、`\n` 这些换行符，它们不是真正的文字，会污染分词。我们统一替换成空格：

```python
# 把转义换行和连续空格压成一个空格
for col in ['project_essay_1', 'project_essay_2', 'project_title', 'project_resource_summary']:
    train[col] = train[col].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
    test[col] = test[col].apply(lambda x: x.replace('\\r', ' ').replace('\\n', ' ').replace('  ', ' '))
```

**第三，类别特征要编码。** 科目大类、小类都是文字，无法直接进模型。这里我们对高基数的类别用 **目标编码（target encoding）**：把每个类别替换成「该类别历史通过率的平均值」。为了防止过拟合，还加了平滑——样本少的类别向全局平均值靠拢。

```python
def target_encode(trn_series, tst_series, target, min_samples_leaf=1, smoothing=1):
    """用类别对应的历史通过率替换类别名；样本少时向全局均值收缩"""
    temp = pd.concat([trn_series, target], axis=1)
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # 样本越多，越信任它自己的均值；样本越少，越靠近全局均值
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    prior = target.mean()
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # 分别应用到训练集和测试集
    ft_trn = pd.merge(trn_series.to_frame(), averages, on=trn_series.name, how='left')[target.name].fillna(prior)
    ft_tst = pd.merge(tst_series.to_frame(), averages, on=trn_series.name, how='left')[target.name].fillna(prior)
    return ft_trn, ft_tst
```

### 2.3 模型训练与超参数调优

**文本转向量。** 我们用了 TF-IDF：一个词在**这一篇**里出现得越多（TF）、在**所有文章**里出现得越少（IDF），权重就越高。这样「的、了、我们」这类到处都是的词自动被压低。对标题和作文分别建向量器：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

# 作文用 1~3 个词的组合（uni/bi/trigram），最多保留 2000 个特征
vectorizer = TfidfVectorizer(stop_words=stop, ngram_range=(1, 3),
                             max_df=0.9, min_df=5, max_features=2000)
vectorizer.fit(train['project_essay_1'])
train_essay_1 = vectorizer.transform(training_examples['project_essay_1'])
val_essay_1   = vectorizer.transform(validation_examples['project_essay_1'])
test_essay_1  = vectorizer.transform(test_examples['project_essay_1'])

# 标题用 1~2 个词的组合
title_vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2),
                            max_df=0.9, min_df=5, max_features=2000)
title_vec.fit(train['project_title'])
train_title = title_vec.transform(training_examples['project_title'])
```

`max_df=0.9` 表示「在 90% 以上文档里都出现的词直接丢掉」，`min_df=5` 表示「出现少于 5 次的词也丢掉」，两者一起把词表控制在 2000 维，既省内存又减少噪声。

**拼接特征。** 把 TF-IDF 出来的稀疏矩阵和数值、类别特征横向拼在一起：

```python
from scipy.sparse import hstack, csr_matrix

# 把数值特征转成稀疏矩阵，再和文本特征拼起来
training_features = hstack([
    train_title, train_essay_1, train_essay_2,
    csr_matrix(training_examples[num_cols].values)
]).tocsr()
```

**训练 LightGBM。** 我们对比了 XGBoost 和 LightGBM，发现两者 AUC 很接近，但 LightGBM 跑得更快，所以选它。参数按资料里的设置：

```python
import lightgbm as lgb

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',      # 二分类
    'metric': 'auc',            # 直接用 AUC 做早停依据
    'max_depth': 16,
    'num_leaves': 31,
    'learning_rate': 0.025,
}

model = lgb.train(
    params,
    lgb.Dataset(training_features, training_targets),
    num_boost_round=10000,
    valid_sets=[lgb.Dataset(validation_features, validation_targets)],
    verbose_eval=100,
    early_stopping_rounds=100   # 验证 AUC 连续 100 轮不涨就停
)
```

这里的 `early_stopping_rounds=100` 很关键：树模型轮数越多越容易过拟合，让验证集自动决定「什么时候该停」，比人工拍一个轮数靠谱。

**冠军方案的思路。** 第一名的做法不是单个模型，而是**三层集成**。第一层训练很多个差异很大的模型：GRU-ATT（带注意力机制的循环网络）、Bi-LSTM、胶囊网络、Bi-GRU + Conv1D 的组合，以及深度 17 的 LightGBM、深度 7 的 XGBoost。每个模型都保存**折外预测（OOF）**——即用 8 折交叉验证时，每条训练样本由「没在它上面训练过」的那个模型给出的预测。第二层用 HillClimb（一种逐个试权重、贪心地挑出最佳组合的算法，做了 4 份）和一个 XGBoost 堆叠器（1 份）来融合。最终深度网络贡献了 54% 的权重，梯度提升树贡献 46%。

他们的特征工程也很值得学：「厨房水槽法」——先把能想到的特征全造出来，让模型自己去挑。比如统计每段文本里大写字母的数量（大喊大叫的申请可能被扣分）、比较 essay_1 和 essay_2 的相似度（偷懒的老师两段写一样）、统计拼写错误数、用 Textacy 算各种文本统计量。其中胶囊网络单模型的本地 AUC 达到了 **0.81251**。

### 2.4 模型测试与指标报告

我们训练时让 LightGBM 打印验证 AUC：

```python
# 训练日志（节选）
# [100]  valid_0's auc: 0.746033
# [400]  valid_0's auc: 0.771107
# [800]  valid_0's auc: 0.774174
# [892]  valid_0's auc: 0.774457   <- 最佳
# Early stopping, best iteration is: [892]
```

可以看到，前 100 轮 AUC 从 0.746 快速涨到 0.77 附近，之后每 100 轮只涨千分之几，第 892 轮达到 **0.774457** 后就基本不动了。这说明：

1. 文本 + 数值特征的基线是有效的，AUC 明显高于 0.5 的瞎猜线；
2. 提升在 0.77 附近见顶，单纯加树的数量没用，想再往上必须靠**更强的文本表示**（深度网络）和**更多特征**。这也正是冠军方案要堆那么多模型的原因。

确认模型可用后，用最佳轮数对测试集预测，输出概率：

```python
submission = pd.DataFrame()
submission['id'] = test['id']
# 用 best_iteration 而不是 10000，避免用过头
submission['project_is_approved'] = model.predict(test_features,
                                                  num_iteration=model.best_iteration)
submission.to_csv('submission.csv', index=False)
```

这里要特别注意 `num_iteration=model.best_iteration`。如果直接预测全部 10000 轮，等于用了大量在验证集上已经证明会过拟合的树，AUC 反而会下降。

---

## 三、总结

本文中我们完成了 DonorsChoose 申请书筛选的完整流程。我们首先通过 EDA 发现了几条有价值的规律：老师经验与通过率相关，申请的物品种类越多、总件数越多、总花费越高，通过率越低，而平均单价越高反而越容易通过。其次我们做了数据清洗：把早期四段作文合并成两段以统一字段含义、清掉转义换行符、用带平滑的目标编码处理高基数类别。接着我们把标题和作文做成 1～3 gram 的 TF-IDF、与聚合后的物资数字特征拼成稀疏矩阵，用 LightGBM 训练并靠验证 AUC 早停。最后我们在验证集上得到 **0.774457** 的 AUC，也看到这个基线在 0.77 附近见顶；冠军方案则用 GRU-ATT、Bi-LSTM、胶囊网络加 LGB/XGB 做三层集成，单是胶囊网络就有 0.81251 的本地 AUC。

这个比赛给我们最大的启发是：**结构化特征和文本特征同样重要，而且特征工程要敢于「多做」**。物资的聚合数字看起来平平无奇，切成分位数桶后却能暴露出清晰的审核偏好；而冠军把能想到的特征全部造出来再交给模型筛选，本质上是用算力换洞察。
