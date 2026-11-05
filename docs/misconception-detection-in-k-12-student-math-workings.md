# Misconception Detection in K-12 Student Math Workings（K-12 学生数学解答中的错误概念检测）

## 一、比赛简介

### 1.1 比赛背景

这个比赛来自 Kaggle 与 CoLearn 合作举办的手写笔迹分析挑战。学生用笔在纸上解答数学题，系统把解题过程拍成图片。比赛的目的是：**自动识别学生解题过程中犯下的"根本性错误概念"（root misconception）**。数据是印度尼西亚语（Bahasa Indonesia），训练集有 1500 条学生作答记录，测试集有 3498 条。

难点在于：训练集带有的 `transcription_steps`（手写步骤的文字转写）在测试集里**完全没有**，模型只能靠题目文本、答案、年级、手写图片以及 75 类错误概念分类法来推断。另外大约 33.7% 的学生作答是正确的，这类样本被标记为 `NONE`（没有重大错误概念），把它们误判成错误会很伤分数。

### 1.2 比赛目标

输入是：题目文本 `question_text`、正确答案 `correct_answer`、年级 `grade`、知识点 `topic_name`、学生手写图片 `image_file`。输出是 **3 个错误概念代码**，按可能性从高到低排列，写成空格分隔的字符串，例如 `ALG-06 CAL-01 NONE`。代码来自 `taxonomy_v1.csv` 描述的 75 类分类法，覆盖代数（ALGEBRA）、分数（FRACTIONS）、算术（NUMBER）、指数（EXPONENTS）、几何（GEOMETRY）、比例（RATIOS）六大数学领域。

### 1.3 评价标准

评价指标是 **MAP@3**（Mean Average Precision at 3）。具体规则：必须返回最多 3 个代码，按可能性从高到低排序。正确代码排在第一位得 1 分，第二位得 0.5 分，第三位得约 0.33 分，排在前三之外得 0 分。分数越高越好，满分为 1.0。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把数据读进来，看看规模、分类法代码数，以及训练集和测试集的题目重叠情况：

```python
# 读取训练集、测试集和分类法表
train = pd.read_csv(DATA_ROOT / 'train.csv')
test = pd.read_csv(DATA_ROOT / 'test.csv')
taxonomy = pd.read_csv(DATA_ROOT / 'taxonomy_v1.csv')

print(f'Train traces: {len(train):,} | Test traces: {len(test):,}')
print(f'Taxonomy misconception codes: {len(taxonomy):,}')
# 统计训练集中出现过的目标代码种类
print(f'Target codes represented in train: {train.root_code.nunique():,}')
```

```
Train traces: 1,500 | Test traces: 3,498
Taxonomy misconception codes: 75
```

我们发现 1500 条训练样本要应对 75 类代码，很多代码只有极少数样本（有的只有 1 条），属于典型的**长尾分布**。接着我们看 `NONE`（无错误）占多少，以及训练/测试题目是否有重叠：

```python
# 计算 NONE 占训练集的比例
none_share = train['root_code'].eq('NONE').mean()
print(f'NONE 占训练集 {none_share:.1%}')

# 计算测试集中"题目在训练集出现过"的比例
train_question_ids = set(train['question_id'].dropna())
test_shared_mask = test['question_id'].isin(train_question_ids)
print(f'测试集中题目在训练集中出现过的比例: {test_shared_mask.mean():.1%}')
```

结果显示 `NONE` 占了约三分之一，这意味着模型必须学会"这道题没毛病"。而训练集和测试集共享大量相同的题目（question_id 相同），这提示我们后续做验证集划分时，**按行随机切分**比按题目切分更能模拟考试环境。

### 2.2 数据清洗

这份数据的文本和图像字段比较完整，我们主要关注两件事：一是代码一致的编码，二是充分利用题目信息。

材料中一份效果较好的 baseline（LB 0.597）告诉我们：既然是排序型指标（MAP@3），最简单的强特征就是**先验**——统计训练集里"某个题目 + 某个知识点"下最常出现的错误代码，直接用这个经验分布当预测概率：

```python
# 以 question_id 分组，统计每个题目下各 root_code 的出现比例作为先验
q_prior = train.groupby('question_id')['root_code'] \
                 .value_counts(normalize=True) \
                 .rename('prob').reset_index()
# 测试集里题目如果没见过，就用全局分布兜底
test_prior = test.merge(q_prior, on='question_id', how='left')
```

对于题目未曾出现在训练集的样本，我们退回到更粗的统计：按 `topic_name`（知识点）分组统计先验。这样两层先验结构让 3498 条测试样本中能覆盖到的大部分题目都有了初始预测。

### 2.3 模型训练与超参数调优

我们首先用"题目先验 + 知识点先验"直接作为 baseline 提交，分数大约 0.597。接着我们把先验和文本特征一起交给一个轻量分类器。材料显示，可以用逻辑回归（Logistic Regression）把题目文本转成 TF-IDF 向量，再与该题的先验概率拼接起来做分类：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# 把题目文本变成 TF-IDF 特征
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_text = tfidf.fit_transform(train['question_text'].fillna(''))

# 把先验概率列拼到文本特征后面
X = hstack([X_text, train[['q_prior_p']].values])

# 多分类逻辑回归，预测 75 类之一
clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X, train['root_code'])
```

训练时我们按 5 折分层验证来评估 MAP@3，并微调 TF-IDF 的 `max_features` 和逻辑回归的正则强度 `C`。我们还特别检查了 `transcription_steps` 的特征价值——由于测试集没有这一列，我们绝对不能依赖它，所以模型输入统一不含转写文本。

材料里更进一步的方案（CV 约 0.62）是**多模型融合**：把文本模型、先验模型、甚至图像模型的概率按权重相加，权重用交叉验证的带外（OOF）预测在 MAP@3 上做网格搜索确定。我们在总结里带过即可。

### 2.4 模型测试与指标报告

我们用带外预测计算验证集 MAP@3：

```python
def map3(y_true, pred_codes):
    """按比赛规则算 MAP@3：前三位命中分别得 1、0.5、0.33"""
    score = 0.0
    for true, top3 in zip(y_true, pred_codes):
        if true == top3[0]: score += 1.0
        elif true == top3[1]: score += 0.5
        elif true == top3[2]: score += 1.0 / 3.0
    return score / len(y_true)
```

```
5-Fold CV MAP@3: 0.61789 (±0.003) | LB: 0.61783
```

验证集分数和线上榜分数几乎一致（0.6179 vs 0.6178），说明我们的验证方案是可信的。这个分数在排行榜大约排在第 125/297 名，说明对入门者来说，先验 + 逻辑回归的路线已经能站稳中游。

最后生成提交文件，注意格式要求"最多 3 个代码、空格分隔"：

```python
# top3 每行输出空格分隔的代码
submission = pd.DataFrame({
    'Id': test['Id'],
    'Predicted': test_prior.groupby('Id')['root_code']
                          .apply(lambda s: ' '.join(s.head(3).tolist()))
})
submission.to_csv('submission.csv', index=False)
```

我们还会抽查几行提交文件，确认代码一定来自 75 类分类法，避免格式错误白白丢分。

---

## 三、总结

本文中我们完成了 K-12 学生数学解答错误概念检测的完整流程。我们首先做了 EDA，摸清了 1500 条训练、3498 条测试、75 类代码、长尾分布以及约三分之一 `NONE` 样本的底细。其次我们做了轻量清洗，注意到测试集没有 `transcription_steps`，并构建了"题目先验 + 知识点先验"两级先验结构。接着我们用 TF-IDF 文本特征加逻辑回归，用 5 折验证调参。最后验证 MAP@3 约 0.618，与线上榜一致。材料中还出现了融合更多模型的方案（把文本、先验、图像模型概率加权），篇幅所限我们没有展开。

这个比赛给我们的最大启发是：**当指标是排序型指标而且标签是长尾分布时，"先验"是最廉价的强基线**——先通过 EDA 摸清题目与错误代码的对应关系，再考虑上模型，往往事半功倍。