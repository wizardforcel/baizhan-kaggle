# Russian State Exam 模拟器（Russian State Exam Simulator）

## 一、比赛简介

### 1.1 比赛背景
这个比赛是围绕俄罗斯统一国家考试（ЕГЭ, EGE）的模拟题预测而做的。考试体系里，同一场考试由多道选择题组成，每道题有 1~n 个选项，题目本身常带有题干正文和上下文。主办方特意不提供本场比赛的训练集——目的是检验模型在“零样本”条件下的迁移能力。真正的训练数据要从同组织的另一场“多选题分类”比赛中借用，两场的数据题库不重叠，使用借用数据训练并在新试卷上预测是比赛认可的、且意在测试通用化能力的路径。

我们需要理解的第一个关键点是试卷结构。原始数据是考试模拟的任务表，每行是一道题，字段包含 `paper`（试卷编号）、`exam`（考试类型）、`subject`（科目）、`question`（题目文字）、`context`（题干正文或阅读材料）、`n_options`（选项个数）、`option_1...option_n`（具体选项内容）、`position`（题目在试卷中的顺序）、`id`（唯一编号）。一个 `paper` 对应一次考试：同一个科目同一等级的 8~12 道题，按 `position` 排序，形成完整的模拟试卷。主办方强调按试卷计分更有信息量，整体平均分会掩盖模型在某些科目上的强弱，这是我们设计评估时需要牢记的前提。

从任务设计来看，这是一场典型的跨域迁移问题。我们面对的不是一个标准的有监督分类任务，而是“无标注的新域 + 有标注的老域”。比赛的哲学是，好的模型应该能从过去的考试题库中学会“什么是好的选项”，而不是死记特定题目的答案。题目间的差异很大，科目涵盖语文、数学、物理、化学、历史等，语言为俄语，字符集和词形变化丰富，这对特征设计提出了挑战。

### 1.2 比赛目标
我们拿到的是 `test.csv`，其中每一行是一道需要预测的题目。我们要为每道题输出 `Category`，也就是正确答案的选项编号。测试集中 `paper` 共有数千张，覆盖多个 `exam` 和 `subject`。因为没有本场训练标签，我们必须先把另一场多选题比赛的 `train.csv` 当作唯一的监督信号来学习，然后把学到的模型套用到本场的 `test.csv` 上。最终提交文件要按 `Id` 顺序给出 `Category`，官方按题目准确率评分。

具体来说，输入是一张包含题目描述、上下文、选项的表，输出是每个题目对应的正确选项序号。我们需要在没有本场标签的情况下完成预测，这意味着模型的泛化能力比记忆能力更重要。我们还要兼顾试卷层面的一致性：同一个试卷内的题目往往属于同一科目、同一难度，模型在同一试卷上表现的一致性也是我们关心的隐含目标。

### 1.3 评价标准
官方排行榜用整体准确率，即预测正确的题目占全部测试题目的比例。常数基线——对所有题都猜 1——大约是 0.263。零样本起步的基线在 0.318 左右，已经超过常数。比赛还建议我们自己按试卷拆分评估：同一试卷内各题预测的正确率能看出模型是否均匀，或是否在某些科目上特别强。对第一人称的我们而言，目标是让借用数据的基线尽可能稳定，并为后续引入大语言模型（LLM）留出清晰的评分骨架。

评价指标简单直白，但解读需要细致。整体准确率容易受不同科目题目数量的影响，如果模型在高频科目上表现好，整体分就会被拉高。反之，如果模型在少数科目上表现极差，整体分也会受损。因此我们需要同时关注科目维度的表现差异。比赛材料中反复提到“per-paper scores are far more informative than the single leaderboard number”，这提醒我们不要只盯着一个总分，而要看分布。最终提交只要求一个答案，但我们的内部分析必须多维度。

## 二、解题过程

### 2.1 数据探索（EDA）
首先我们用 glob 把所有输入里的 `test.csv` 找出来，筛出包含 `paper` 列的那个文件，这就是本场的测试集。

```python
import glob, pandas as pd

# 找出包含 paper 列的 test.csv
test = next(d for d in (pd.read_csv(f) for f in glob.glob('/kaggle/input/**/test.csv', recursive=True))
            if 'paper' in d.columns)
print(test.shape)
print('papers:', test.paper.nunique(), 'subjects:', test.groupby(["exam","subject"]).ngroups)
```

输出显示试卷数量和科目组合。接着我们看每张试卷的题目数分布：

```python
print(test.groupby('paper').size().describe()[['count','min','max']])
print(test.groupby(['exam','subject']).size().sort_values(ascending=False))
```

这里我们看到一张试卷通常 8~12 题，不同科目题目总量差异很大。拿一张试卷实际看效果：

```python
p = test[test.paper == test.paper.iloc[0]]
for _, r in p.head(3).iterrows():
    print(f"{r.position}. {str(r.question)[:90]}")
```

从样例中我们体会到题目是俄语原文，`context` 字段在需要时提供段落背景，选项以 `option_1...` 存在。我们进一步确认每个题目选项数不同，`n_options` 用来循环构造选择对。

为了更深入地了解数据结构，我们统计了各科目的选项数分布：

```python
# 统计每个题目的选项数量
option_counts = test['n_options'].value_counts().sort_index()
print(option_counts)
```

这一步告诉我们并不是所有题目都是四选一，有些是三选一或五选一，这会影响到我们的基线策略——常数猜 1 的基线期望值并不是简单的 1/4，而取决于选项数分布的平均值。我们还查看了上下文字段的填充率：

```python
# 看上下文是否存在
has_context = test['context'].astype(str).str.strip().ne('')
print('有上下文的比例:', has_context.mean())
```

有上下文的题目往往是阅读理解类，题干更长，选项之间的区分度可能更高。从 EDA 中我们得出几个结论：试卷是完整的评价单元，题目大多是俄语文本，选项数多变，上下文在部分题目中提供额外信息，科目间分布不均。这些洞察直接影响了我们后面的特征构造方式。

### 2.2 数据清洗
原始数据基本干净，但我们需要把“题目+选项”展开成可以做文本分类的行。核心函数 `pairs` 把每道题拆成多个选择对，标签为题目的正确选项。

```python
def pairs(df, labelled=True):
    rows = []
    for _, r in df.iterrows():
        stem = str(r.question)
        # 如果有上下文，把它拼到题干前面
        if isinstance(r.context, str) and r.context.strip():
            stem += ' ' + r.context
        # 为每个选项生成一行
        for i in range(1, int(r.n_options)+1):
            rows.append({'gid': r.id, 'opt': i,
                         'text': stem + ' ||| ' + str(r[f'option_{i}']),
                         'y': int(labelled and r.answer == i)})
    return pd.DataFrame(rows)
```

注意我们没有删停用词，也没有做俄语分词，只是把题干和选项用 `|||` 拼接成一条文本。这样字符级特征可以捕捉选项词的局部模式。对于测试集 `labelled=False`，我们只生成 `text`，不生成 `y`，后续只预测概率。

在读取另一场多选题比赛的 `train.csv` 时，我们同样用 `pairs` 展开，得到训练对的数量级。

```python
train = pd.read_csv(hits[0])
P_tr = pairs(train)
print('borrowed', len(train), 'labelled tasks')
```

展开后每道题有 `n_options` 行，正负样本比例自然不均衡，我们随后用 `predict_proba` 取每个题目里得分最高的选项。

数据清洗的重点在于保持文本完整性。俄语文本中的标点、大小写、特殊符号都可能携带信息，我们没有进行常规的文本清洗如转小写、去标点、去停用词。因为字符级 n-gram 本身对噪声有一定鲁棒性，过度的清洗反而会丢失区分信息。我们只做了一件事：把 `question`、`context`、`option` 拼接成统一的文本字段，确保所有输入格式一致。`pairs` 函数还负责处理选项数量不一致的问题，动态循环到 `n_options`，避免硬编码。

我们还检查了是否有空值：

```python
# 检查关键字段空值
print(test[['question','n_options']].isnull().sum())
```

确认核心字段完整后，我们进入特征工程阶段。清洗的目标是“最小干预”，只做格式统一和结构化，不改变文本原意。

### 2.3 模型训练与超参数调优
因为题目和选项都是短文本，我们采用字符级 TF-IDF 而非词级，避免俄语形态变化带来的稀疏问题。俄语同一个词可能因为格、数、性发生变化，词级方法会把它们看成不同词；字符片段则能保留共同的局部结构，对跨题库迁移更稳。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4),
                      min_df=2, max_features=60_000)
X = vec.fit_transform(P_tr.text)
model = LogisticRegression(max_iter=1000, C=2.0).fit(X, P_tr.y)
```

`analyzer='char_wb'` 表示在词边界内做字符 n-gram，对选择题这种短文本很有效。`ngram_range=(2,4)` 捕捉两到四个字符的模式，`min_df=2` 去掉只出现一次的噪声，`max_features=60_000` 控制维度。LogisticRegression 用 `C=2.0`，相对默认更宽松，配合 `max_iter=1000` 确保收敛。这个基线在材料里给出约 0.318 的自我评分，明显高于常数基线 0.263。

训练完成后我们把模型套用到本场测试集：

```python
P_te = pairs(test, labelled=False)
P_te['s'] = model.predict_proba(vec.transform(P_te.text))[:,1]
pick = P_te.loc[P_te.groupby('gid').s.idxmax()]
```

对每道题 `gid` 取概率最高的选项即为预测答案。最后按比赛格式整理提交文件：

```python
sub = pd.DataFrame({'Id': pick.gid.values, 'Category': pick.opt.values})
sub = sub.set_index('Id').loc[test.id].reset_index()
sub.to_csv('submission.csv', index=False)
```

我们还构建了按试卷统计的评估骨架，用于后续校准：

```python
merged = test.merge(sub, left_on='id', right_on='Id')
by_paper = merged.groupby(['exam','subject','paper']).Category.agg(
    n='size', picked_1=lambda s: (s==1).mean())
```

虽然没有真实标签，我们仍能看到预测分布，为后续引入 LLM 时做“模型分歧分析”提供依据。

在调参方面，我们没有进行大规模网格搜索。原因是本场没有验证标签，任何在本场测试集上反复试参数都会造成隐性过拟合。我们把参数选择限定在借用数据上：字符范围取 2 到 4，特征上限取 60,000，文档频次至少为 2，逻辑回归的正则参数取 2.0。这些参数的共同目标是让模型既能看见俄语词形的局部模式，又不会因为只出现一次的片段而记住训练题。

我们还对比了两个常数策略的含义。一个是无论题目类型都选择 1，另一个是按照 `n_options` 的先验频率选择最常见的选项。材料中明确报告了常数约 0.263，而字符模型约 0.318，所以我们把字符模型作为主要方案。若继续改进，可以沿着材料给出的方向加入 LLM，但必须保持一题一答案的输出约束，不能让模型输出解释文字混入提交文件。

### 2.4 模型测试与指标报告
本场比赛没有公开验证集，我们只能在本地用 borrowed 数据做内部评估。材料里给出零样本基线的自我评分约 0.318。对照常数基线 0.263，我们的字符级 TF-IDF + LogisticRegression 已经建立了合理的起步点。

在 2.3 里我们已经实现了按试卷聚合的统计，能直观看出不同科目、不同考试级别的预测倾向差异。如果后续拿到了 leaderboard 分数，我们可以用同样的 `by_paper` 骨架把分数映射回试卷维度，判断模型是均匀中等，还是在部分科目上很强、部分科目上很弱。这正是材料一开始强调的——平均分会掩盖 variance，而按试卷拆分能给出更具操作性的信息。

我们还把基线作为第二意见与 LLM 方案对比。材料最后给出了一段可复用的提示词模板，把一道题、它的选项和一问一答的要求组织起来，强制模型只输出一个数字：

```python
r = test.iloc[0]
# 把每个选项拼接成一行
opts = '\n'.join(f'{i}. {r[f"option_{i}"]}'
                 for i in range(1, int(r.n_options) + 1))
# 上下文存在时才插入阅读材料段落
passage = f'ТЕКСТ:\n{r.context}\n\n' if isinstance(r.context, str) and r.context.strip() else ''

PROMPT = f'''Ты сдаёшь {r.exam} по предмету «{r.subject}».

{passage}ЗАДАНИЕ:
{r.question}

ВАРИАНТЫ:
{opts}

Ответь одним числом — номером верного варианта.'''
print(PROMPT)
```

这段模板的意义在于：难度不在提示词本身，而在内容理解上。对我们的评分骨架来说，它正好可以和字符基线的预测做“模型分歧分析”——两个模型对同一题给出不同答案时，正是最值得人工检查的位置。

## 三、总结
本文中我们完成了 Russian State Exam 模拟器的零样本基线搭建。我们首先确认了本场没有训练集，测试集以 `paper` 为单元组织，题目为俄语选择题。其次我们从同组织的另一场多选题比赛借用标签，用 `pairs` 函数把题干+上下文+选项展开成文本对。接着我们采用字符级 word-boundary TF-IDF（2-4 gram）和 LogisticRegression（C=2.0）做基线，得到约 0.318 的自我评分，优于常数基线 0.263。最后我们按试卷聚合预测结果，并输出标准提交格式，同时保留了按试卷评估的骨架，为后续引入大语言模型提供可扩展框架。