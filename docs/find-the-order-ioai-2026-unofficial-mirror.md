# 找出顺序（Find the Order，IOAI 2026 非官方镜像）

## 一、比赛简介

### 1.1 比赛背景

IOAI 是 International Olympiad in Artificial Intelligence（国际人工智能奥林匹克）的缩写，是一项面向中学生的国际性人工智能竞赛。Kaggle 上的这个比赛名字里带有 “unofficial mirror”，意思是**非官方镜像赛**：组织者把 IOAI 2026 的一道题目搬到了 Kaggle 平台上，让没有到场参赛的人也能拿同样的数据练手。

这道题叫 Find the Order，直译就是「找出顺序」。现实生活里有很多类似的判断：两封邮件哪封先发出、两条新闻哪条先发布、两段代码哪个版本更早、两张照片哪张先拍。人做这种判断时，往往要把两条信息放在一起比较，而不是单独看其中一条。比赛想考的就是这种**成对比较**的能力。

这类任务和普通的单条分类不一样：模型看到的不是一条样本，而是「一对」样本。两条记录之间可能只有很细微的差别，比如日期写法、措辞的新旧、事件提到的先后线索。难点也在这里——模型必须学会比较，而不是分别给两条记录打分。

### 1.2 比赛目标

材料里的 starter notebook 展示了数据的位置和提交格式。测试数据放在 `find_the_order_test` 目录下，包含两个文件：

- `test_pairs.csv`：每行是一对需要判断的记录，`pair_id` 是这对样本的编号，其余列存放这一对的两条记录内容；
- `sample_submission.csv`：提交样例，列名是 `pair_id` 和 `before`。

我们要为每个 `pair_id` 填一个 `before` 值。从列名和 starter 的写法看，`before` 是一个 0/1 标记，表示「这一对记录中，前一条是否排在后面那条之前」。也就是说，输出的是一个二分类判断，而不是具体的时间戳。

需要说明的是，starter 只演示了如何读取**测试对**和生成合法提交，并没有附带完整的训练流程；材料里也没有保留训练集的具体列名和条数。因此下面凡是涉及训练数据的部分，我们只讲方法和思路，不虚构字段名与数字。

### 1.3 评价标准

提交列是 0/1 标签，因此这类比赛通常按**准确率（Accuracy）**评分：预测对的 pair 数除以总 pair 数，越高越好，满分是 1.0。

材料里没有给出官方指标说明，我们根据提交格式做这个合理推断，并在本地验证时同样用准确率。这里要特别提醒一个陷阱：如果一份提交把所有 pair 都填成 0，它的准确率就等于「真实标签为 0 的比例」。如果数据里正负样本各占一半，全填 0 也能拿到大约 0.5 分。所以**看起来五成的分数并不代表模型有用**，必须和「全填同一个值」的基线对比。

---

## 二、解题过程

整场比赛可以拆成四步：先看清测试对长什么样，再把成对的记录整理成模型能吃的特征，接着训练一个二分类器，最后按格式提交并用本地验证估计效果。我们按这四步走。

### 2.1 数据探索（EDA）

我们首先把测试对和提交样例读进来，确认文件路径、列名、行数，以及 `pair_id` 是否能和提交样例一一对应：

```python
from pathlib import Path
import pandas as pd

# 比赛数据目录：starter 里给的是 Kaggle 的只读输入路径
INPUT = Path('/kaggle/input/competitions/'
             'find-the-order-ioai-2026-unofficial-mirror/find_the_order_test')

# 读取测试对和提交样例
pairs = pd.read_csv(INPUT / 'test_pairs.csv')
sample = pd.read_csv(INPUT / 'sample_submission.csv')

# 先看规模：有多少对需要判断
print(f'{len(pairs)} test pairs')

# 看前几行，确认一对记录是怎么摆放的
display(pairs.head())
display(sample.head())
```

运行后 starter 会打印测试对的数量，但材料里没有保留具体数字，所以这里不引用条数。我们重点关注的是结构：`pairs` 的一行对应一对记录，`sample` 的一行对应一个待填的 `pair_id`。

接着我们检查三件事：`pair_id` 是否唯一、两个文件的 `pair_id` 集合是否一致、以及记录列有没有缺失。前两件事决定提交能不能对上号，第三件事决定要不要做缺失处理：

```python
# 1) pair_id 必须唯一，否则提交会出现重复行
assert pairs['pair_id'].is_unique
assert sample['pair_id'].is_unique

# 2) 测试对和提交样例的编号集合应当完全一致
assert set(pairs['pair_id']) == set(sample['pair_id'])

# 3) 看看除 pair_id 外的记录列有没有空值
print(pairs.isnull().sum())

# 4) 看看一对里的两条记录是否有可能完全一样
#    这里用除 pair_id 外的所有列做比较
record_cols = [c for c in pairs.columns if c != 'pair_id']
same = (pairs[record_cols].nunique(axis=1) == 1).sum()
print('两条记录完全相同的对数：', same)
```

这段检查帮我们排除了两类低级错误：编号对不上、以及重复提交。如果发现同一对里两条记录完全相同，那这对样本本身没有可判断的信息，只能靠先验比例去猜。

最后我们看一眼训练标签的分布（如果有训练集的话）。这一步很重要，因为它决定了「全填 0」到底能拿多少分：

```python
# 假设训练集里同样有 pair_id 和 before 两列
# 统计 0 和 1 的比例，判断数据是否平衡
# 若 0 占 70%，那么全填 0 的基线准确率就是 0.70
print(train['before'].value_counts(normalize=True))
```

从任务形式看，这类「谁在前」的配对通常是正负各半构造出来的，但我们不把这一点当成事实，而是要求在实际数据上先算一遍再决定基线。

### 2.2 数据清洗

这份数据的清洗重点不在「去脏」，而在「保证成对结构不被破坏」。我们做了三件事。

第一，统一两条记录的文本格式。前后空格、全角半角、大小写这些差异，如果不处理，模型可能把「格式不同」误当成「时间不同」：

```python
def normalize(text):
    """把一条记录变成统一的字符串形式"""
    if pd.isna(text):
        return ''                 # 缺失统一成空串，避免后续报错
    text = str(text)
    text = text.replace('　', ' ')   # 全角空格换成半角
    text = ' '.join(text.split())        # 把连续空白压成一个空格
    return text.strip()

# 对两条记录列分别做归一化
# 列名以实际数据为准，这里用 rec_a / rec_b 代指
pairs['rec_a'] = pairs['rec_a'].map(normalize)
pairs['rec_b'] = pairs['rec_b'].map(normalize)
```

第二，处理空记录。一对里如果有一条是空的，模型就没法比较，我们给这类样本打一个标记，而不是直接删掉——测试集不能删，删了提交就对不上编号了：

```python
# 标记「有一条为空」的样本，让模型自己学这类样本该偏向哪边
pairs['one_empty'] = ((pairs['rec_a'] == '') ^ (pairs['rec_b'] == '')).astype(int)

# 统计一下有多少这样的对，做到心里有数
print('含空记录的对数：', pairs['one_empty'].sum())
```

第三，固定「谁在前」的语义。训练时我们必须保证 A、B 两条记录的摆放顺序和标签含义一致，否则模型会学到相反的关系。做法是：训练集里随机交换 A、B 的位置，同时把标签取反。这样模型就不会偷懒地记住「左边总是更早」这种位置偏置：

```python
import numpy as np

# 随机选一半样本交换两条记录，并把标签 0/1 对调
swap = np.random.rand(len(train)) < 0.5
train.loc[swap, ['rec_a', 'rec_b']] = train.loc[swap, ['rec_b', 'rec_a']].values
train.loc[swap, 'before'] = 1 - train.loc[swap, 'before']
```

这一步是成对任务里非常关键的「数据增强」，成本几乎为零，但能显著减少位置带来的虚假规律。

### 2.3 模型训练与超参数调优

材料里只给了一个「全填 0」的 starter，没有现成模型。下面是我们根据这类任务的一般做法整理的方案，**属于设计思路，不是材料里已经跑出的结果**。

核心想法是：不要分别给两条记录打分，而是**把两条记录拼成一条「比较特征」**。具体来说，对每条记录抽取长度、数字占比、年份/日期样式、标点密度等轻量特征，然后构造「A 减 B」的差值和「A、B 拼接」的文本，一起喂给分类器：

```python
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

def hand_features(s):
    """从一条记录里抽几个便宜但有用的统计特征"""
    return pd.Series({
        'len': len(s),                                  # 字符长度
        'digits': sum(ch.isdigit() for ch in s) / (len(s) + 1),   # 数字占比
        'punct': sum(not ch.isalnum() and not ch.isspace() for ch in s) / (len(s) + 1),
        'has_year': int(bool(re.search(r'(19|20)\d{2}', s))),     # 是否出现年份
        'upper': sum(ch.isupper() for ch in s) / (len(s) + 1),    # 大写字母占比
    })

# 分别抽取两条记录的统计特征
fa = pairs['rec_a'].map(hand_features)
fb = pairs['rec_b'].map(hand_features)

# 用「差值」表达谁更长、谁数字更多，这一步让模型直接看到比较结果
diff = fa.values - fb.values
diff = pd.DataFrame(diff, columns=[c + '_diff' for c in fa.columns])

# 文本部分：把两条记录用特殊分隔符拼起来，再做字符级 TF-IDF
# 分隔符让模型知道「这里是分界线」
pair_text = pairs['rec_a'] + ' [SEP] ' + pairs['rec_b']
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=50000)
X_text = tfidf.fit_transform(pair_text)

# 数值特征和文本特征拼在一起
X = hstack([X_text, diff.values, fa.values, fb.values]).tocsr()
```

这里我们用**字符级 n-gram** 而不是词级，是因为记录可能很短、拼写不规整，字符片段比完整单词更稳。`char_wb` 表示只在词边界内切字符，能减少跨词的噪音。

分类器我们选逻辑回归做第一版，因为它训练快、系数可以解释，方便我们判断特征有没有学反：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# C 是正则强度的倒数：C 越小正则越强，越不容易过拟合
# 成对任务特征维度高、样本相对少，先用中等强度的正则
clf = LogisticRegression(C=1.0, max_iter=2000, solver='liblinear')

# 分层 5 折交叉验证，保证每折正负比例一致
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print('每折准确率：', scores)
print('平均准确率：', scores.mean())
```

调参时我们只动三个地方：TF-IDF 的 `ngram_range`、`max_features`、以及逻辑回归的 `C`。先用小网格粗扫，再在最优附近细看。我们没有去堆很复杂的模型，因为成对任务的上限往往取决于「特征里有没有真正的先后线索」，而不是分类器的复杂度。

如果数据量足够，第二版可以换成 LightGBM，把手工特征和 TF-IDF 的稀疏输出一起喂进去：

```python
import lightgbm as lgb

# LightGBM 对稀疏特征支持一般，这里只用稠密的手工特征做对比
model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(diff.values, y)
```

我们主要看它和逻辑回归的验证分数差多少；如果差距在误差范围内，就保留更简单的逻辑回归，省得后面解释不清。

### 2.4 模型测试与指标报告

本地验证用分层 K 折，指标就是准确率。但只看准确率不够，我们一定要把它和两个基线放在一起比：

```python
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

# 基线 1：全填 0，就是 starter 的做法
zero_acc = accuracy_score(y, np.zeros(len(y)))

# 基线 2：按训练集比例随机猜
dummy = DummyClassifier(strategy='stratified', random_state=42)
dummy.fit(X, y)
dummy_acc = accuracy_score(y, dummy.predict(X))

print('全填 0 基线：', zero_acc)
print('按比例随机基线：', dummy_acc)
print('我们的模型：', scores.mean())
```

只有当我们模型的准确率**明显高于这两个基线**时，才能说模型学到了东西。如果只比全填 0 高一点点，那很可能只是碰巧押中了多数类，而不是真的会判断顺序。

starter 的提交代码如下，它的作用只是保证格式正确：

```python
# 从样例里复制 pair_id，保证行数和顺序完全一致
submission = sample[['pair_id']].copy()

# starter 故意把所有预测都设成 0，只用来验证提交格式
submission['before'] = 0

# 提交文件必须正好是这两列，且 pair_id 不重复
assert submission.columns.tolist() == ['pair_id', 'before']
assert submission['pair_id'].is_unique

submission.to_csv('/kaggle/working/submission.csv', index=False)
```

等模型训练好之后，我们把 `submission['before']` 换成模型对测试对的预测即可，其余格式保持不变。提交前我们还会做一次检查：行数是否等于 `sample_submission.csv`、`pair_id` 顺序是否一致、预测值是否只包含 0 和 1。

关于分数，我们必须诚实：材料里没有保留任何排行榜成绩，starter 本身也不是预测模型。因此这篇 writeup 不报告具体分数，只报告验证协议和基线对比的方法。这也是我们想强调的一点——**在没有可信验证的情况下，不要用一个看起来像分数的数字安慰自己**。

---

## 三、总结

本文中我们围绕 Find the Order 这个成对排序任务，搭起了一套从数据检查到模型验证的完整框架。我们首先读了 `test_pairs.csv` 和 `sample_submission.csv`，确认 `pair_id` 唯一且两个文件能对上号，并检查了成对记录里的空值和重复。其次我们做了针对成对结构的清洗：统一文本格式、标记空记录，并通过随机交换 A/B 位置同时翻转标签来消除位置偏置。接着我们把两条记录拼成「比较特征」——手工统计量的差值加上字符级 TF-IDF，用逻辑回归做第一版模型，并准备了 LightGBM 作为对照。最后我们用分层 5 折交叉验证评估准确率，并坚持把结果和「全填 0」「按比例随机」两个基线放在一起比较。

这个比赛给我们的最大启发是：**成对任务的胜负手在「比较」而不是「分类」**。如果只把两条记录分别变成向量再拼接，模型也能跑，但很难学到「谁更早」这种相对关系；显式构造差值、交换增强、固定语义，才是让模型真正学会比较的关键。另外，starter 的全 0 提交提醒我们：格式正确的提交不等于有效的模型，看到分数之前先问一句「这个分数比瞎猜好多少」。
