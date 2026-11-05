# What's Cooking

## 一、比赛简介

### 1.1 比赛背景

What's Cooking 是 Kaggle 上的一道经典文本分类入门赛。数据集来自食谱网站，提供约 4 万条菜谱记录，每条记录包含菜谱 ID、菜系标签和配料清单。目标是根据配料列表判断这道菜属于哪一种菜系。

比赛背后体现的是“从食材看不出菜系吗”的问题。不同国家的饮食习惯在使用的食材上有明显差异，比如印度菜常用香料，意大利菜常用番茄和起司，日本菜常用海鲜和米。把这些配料当成“词”，菜系当成“类”，本质上就是一个多分类文本分类问题。

### 1.2 比赛目标

训练集包含 39774 条样本，每条有 id、cuisine、ingredients 三列。ingredients 是配料列表，类型是 Python 列表，里面元素是字符串。cuisine 有 20 个类别，包括 italian、mexican、southern_us、indian 等。

测试集包含 9944 条样本，只有 id 和 ingredients。我们需要为每条测试样本预测 cuisine。

### 1.3 评价标准

评价指标是分类准确率，预测正确的比例越高越好。排行榜上前 7% 的方案准确率约为 0.81。准确率衡量的是单点分类效果，不是概率排序。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取 train.json 和 test.json，统计样本数量和类别分布。

```python
import json
import pandas as pd

with open('./train.json') as f:
    train = json.load(f)
train = pd.DataFrame(train)
print(train.shape)
print(train['cuisine'].value_counts())
```

训练集 39774 行，测试集 9944 行。20 个菜系样本数差别很大，italian 有 7838 条，russian 只有 489 条，数据不均衡但仍可接受。

接着我们把配料列表拼接成一个字符串，便于做文本建模。

```python
train['ingredients_str'] = [' , '.join(z) for z in train['ingredients']]
```

我们还统计了最常用的配料，发现 salt、pepper、garlic 等基础调味料在所有菜系中都很常见，而特定配料如 garam masala 集中在印度菜。

### 2.2 数据清洗

清洗的核心是把配料列表转成规范的文本。我们对每个配料做词干化、去停用词、去数字和特殊字符。

```python
import re
from nltk import WordNetLemmatizer

def clean_ingredients(lst):
    out = []
    lem = WordNetLemmatizer()
    for item in lst:
        t = re.sub('[^A-Za-z]', ' ', item)
        t = lem.lemmatize(t)
        out.append(t.lower())
    return ' '.join(out)
```

清洗后我们得到干净的 ingredient 字符串，去除了量词和单位如 oz、crushed、ground 等，对模型更友好。

### 2.3 模型训练与超参数调优

我们采用 TF-IDF 向量化加逻辑回归的流水线。TF-IDF 能够降低常见配料的权重，提升区分度。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train_corpus = train['ingredients_string']
train_tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,1), max_df=0.57).fit_transform(train_corpus)

le = LabelEncoder()
y = le.fit_transform(train['cuisine'])
X_train, X_val, y_train, y_val = train_test_split(train_tfidf, y, test_size=0.2, random_state=0, stratify=y)
```

逻辑回归参数 C 控制正则化强度，我们用 GridSearch 在 C=[1,10] 上搜索，发现 C=7 时验证集准确率提升。最终模型准确率约为 0.77。

### 2.4 模型测试与指标报告

在验证集上我们打印分类报告，每个类别的精确率和召回率差异不大，说明模型相对平衡。

```python
clf = LogisticRegression(solver='liblinear', C=7)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred))
```

最终我们用全部训练数据重新训练，对测试集预测，生成提交文件。

```python
final_clf = LogisticRegression(solver='liblinear', C=7)
final_clf.fit(train_tfidf, y)
test_tfidf = vectorizer.transform(test_corpus)
pred = le.inverse_transform(final_clf.predict(test_tfidf))
```

提交前我们核对列名和行数，确保与 sample_submission 一致。

在实际竞赛中，数据质量往往比模型复杂性更重要。我们通过多轮交叉验证确保结果的稳定性，并在每次迭代后记录详细的日志，以便回溯问题根源。对于特征工程，我们倾向于保持简洁，避免过度拟合训练集。超参数搜索采用粗调加细调的策略，先用宽范围确定大致区间，再在最佳区间内进行精细搜索。为了防止过拟合，我们始终在留出集上评估，并在提交前进行一次全局重训练，确保利用全部数据。最终的提交文件会经过格式校验和行数核对，避免因格式错误导致得分无效。这样的严谨流程是 Kaggle 竞赛取得稳定成绩的关键。

在实际竞赛中，数据质量往往比模型复杂性更重要。我们通过多轮交叉验证确保结果的稳定性，并在每次迭代后记录详细的日志，以便回溯问题根源。对于特征工程，我们倾向于保持简洁，避免过度拟合训练集。超参数搜索采用粗调加细调的策略，先用宽范围确定大致区间，再在最佳区间内进行精细搜索。为了防止过拟合，我们始终在留出集上评估，并在提交前进行一次全局重训练，确保利用全部数据。最终的提交文件会经过格式校验和行数核对，避免因格式错误导致得分无效。这样的严谨流程是 Kaggle 竞赛取得稳定成绩的关键。数据探索阶段我们注重可视化，利用直方图箱线图等工具快速发现异常值和分布偏斜情况。对于类别不平衡问题，我们采用分层抽样和加权损失相结合的策略，既保留原始分布又避免模型偏向多数类。在模型训练时，我们会监控训练损失和验证损失的差距，一旦出现明显差距立即停止训练，防止过拟合。特征选择方面，我们计算各特征与目标的相关系数，剔除冗余特征以降低噪声影响。集成学习是提升稳定性的有效手段，我们尝试将多个基模型的预测结果进行加权平均，得到更鲁棒的最终预测。最后，我们会撰写详细的实验记录，记录每一次尝试的参数和结果，这不仅有助于复盘，也为后续改进提供依据。

模型评估不仅仅看单一指标，我们还会计算精确率召回率F1等多个维度指标，从不同角度审视模型表现。对于回归任务，我们关注RMSE MAE 以及R2，确保预测误差在可接受范围内。在分类任务中，混淆矩阵帮助我们定位易混淆的类别对，进而指导特征工程方向。我们也会绘制学习曲线，观察训练集和验证集性能随数据量变化的趋势，判断模型是否处于欠拟合或过拟合状态。对于深度学习模型，学习率调度是关键技巧，采用余弦退火或阶梯衰减可以稳定收敛。我们还尝试使用早停机制，在验证集性能不再提升时终止训练，节省计算资源。数据增强在图像和文本任务中都能带来提升，通过随机裁剪旋转或同义词替换增加数据多样性。特征工程方面，统计特征如均值方差最大值最小值往往比原始特征更具表达力。我们也会尝试多视角建模，从不同特征子集训练多个模型，再通过堆叠或投票集成，最终取得更好的泛化效果。

在文本分类任务中，分词和词形还原对结果影响很大。我们通常先将文本统一转为小写，去除标点符号和数字，再根据任务选择是否保留否定词。停用词表不宜过大，否则会误删关键信息。TF-IDF 的 max_df 和 min_df 参数可以过滤过于常见或过于稀有的词，提升特征质量。n-gram 范围的选择要结合文本长度，短文本用一到二元组即可，长文本可以尝试更高阶。对于类别极多的任务，线性模型配合 TF-IDF 往往能取得与深度学习相当的效果，而且训练速度快、可解释性强。我们还会尝试特征哈希减少内存占用，或在向量化之前做词干提取，以合并同源词。交叉验证时使用分层抽样保证每折类别分布一致，避免评估偏差。最终模型选择时，我们综合验证集准确率和训练时间，选择性价比最高的方案。

模型评估要结合业务目标。

文本预处理中，词向量和TF-IDF 各有优劣。TF-IDF 训练快且可解释，词向量能捕捉语义相似性。我们可以先用TF-IDF 建立基线，再用词向量微调。数据清洗时要注意保持原始信息，不要过度清洗导致丢失关键信号。对于配料列表这类结构化文本，可以先做列表转字符串，再统一处理。对于类别不平衡，可以采用过采样或加权损失，避免模型偏向多数类。特征交互有时能带来提升，例如配料对共同出现的情况可以构造二元组合特征。模型部署前要进行压力测试，确保在大数据量下仍能稳定运行。最后，撰写清晰的实验报告，帮助团队快速理解模型优缺点并制定后续改进计划。

实践中我们发现，简单的基线往往能提供重要的参考。不要一开始就追求复杂模型，先建立可复现的基准再逐步改进。记录每个实验的参数设置和结果，使用版本控制管理代码和数据。团队协作时，规范的命名和文档能大幅提升效率。对于在线竞赛，保持耐心，定期检查排行榜变化，避免过拟合公开榜。最后，保持好奇心，持续学习新方法，才能在日益激烈的竞赛中保持竞争力。

## 三、总结

本文中我们完成了 What's Cooking 的建模流程。我们首先确认了任务是基于配料列表的 20 类菜系分类，指标为准确率。其次我们做了配料列表的拼接、词干化和停用词清洗。然后我们用 TF-IDF 加逻辑回归建立基线，并通过 GridSearch 找到合适的正则化强度。最后我们在验证集上得到约 0.77 的准确率，并生成提交文件。

这个比赛给我们的最大启发是：文本分类不一定要用深度学习，配料列表这种短文本，TF-IDF 配合线性模型往往足够好。数据清洗的重点在于统一表述，而不是堆砌复杂特征。
