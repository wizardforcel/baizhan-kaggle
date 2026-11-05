# Unimelb 科研经费申请预测（Predict Grant Applications）

## 一、比赛简介

### 1.1 比赛背景

这个比赛来自墨尔本大学（University of Melbourne）的科研经费申请数据。学校把过去的经费申请记录整理成表格，希望我们根据一份申请的基本信息、研究方向和申请人的历史记录，判断这份申请最后是否成功。材料中说明，数据包含 249 个特征，特征大致分成三类：经费和项目本身的信息、研究方向的信息，以及申请团队成员的脱敏信息。所谓“脱敏”，就是把真实姓名等隐私换成编号，但仍然保留了一些可以用于建模的结构化字段。

训练数据来自 2004 年到 2008 年提交的 8,707 份申请；测试数据来自 2009 年以及 2010 年上半年的 2,176 份申请。实际读入训练文件后，训练表是 8,708 行、252 列，测试表是 2,176 行、252 列。多出来的一列通常是表格中用于标识申请的 `Grant.Application.ID`，目标列是 `Grant.Status`。这个时间顺序带来一个实际难点：测试集来自较晚的年份，训练集和测试集的分布不一定完全一样。

这份数据的第二个难点是缺失值很多。申请人信息按最多 15 个位置展开，例如 `Person.ID.1` 到 `Person.ID.15`。实际数据里后面一些位置几乎没有记录，某些列甚至整列都是空值。同时，研究方向还使用 `RFCD.Code` 和 `SEO.Code` 这类编码，初学者不能把它们直接当成自然语言，也不能只凭列名猜测每个编号的含义。因此，我们需要先了解缺失情况，再决定哪些列可以保留，哪些列应该删除或填补。

### 1.2 比赛目标

我们的输入是一条科研经费申请记录，输出是它的 `Grant.Status`。这个字段是二分类标签：材料中的样例使用 0 和 1 表示两种申请结果。测试文件没有这一列，我们要为每个测试申请生成一个预测值或预测概率。

表格中的字段包括：申请编号 `Grant.Application.ID`，赞助方 `Sponsor.Code`，项目类别 `Grant.Category.Code`，合同金额区间 `Contract.Value.Band...see.note.A`，开始日期 `Start.date`；研究方向由五组 `RFCD.Code`、`RFCD.Percentage` 和五组 `SEO.Code`、`SEO.Percentage` 表示；申请人部分则包含出生年份、出生国家、部门、学院、是否拥有博士学位、在大学工作的年数、过去成功和失败的申请数量，以及 A、B、C 等历史记录字段。测试集的列结构应与训练集保持一致，但要去掉 `Grant.Status`，提交时再把预测结果写回示例文件的对应位置。

我们的目标不是只让模型记住某个申请编号，而是让模型学会“什么样的研究方向、团队经历和申请背景更容易获得资助”。因此，数据清洗时要特别注意编号列和时间顺序，模型评估时也不能只看训练集上的分数。

### 1.3 评价标准

比赛材料中的解决方案使用 ROC 曲线和 AUC（曲线下面积）来检查模型。ROC AUC 可以用大白话理解为：随机拿一份成功申请和一份失败申请，模型把成功申请排在前面的概率有多大。AUC 为 0.5 时接近随机排序，越接近 1 越好。最终提交应当保存连续的预测概率，而不是简单地把概率切成 0 和 1；这样排行榜可以在所有可能的阈值下评价模型的排序能力。

在本地开发阶段，材料里的几份 notebook 也使用了 `accuracy`、`classification_report` 和混淆矩阵。准确率表示猜对的比例，精确率和召回率则分别回答“预测为成功的有多少是真的成功”和“所有成功申请中找回了多少”。这些指标适合帮助我们理解模型，但它们和排行榜使用的概率排序评价不是一回事，所以我们最后要同时保存概率、准确率和 AUC 计算代码。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取训练集、测试集和提交样例，确认文件名、行数、列数以及目标列是否存在。这里把 `low_memory=False` 写出来，是因为原始表中有一些列同时出现数字和字符串，pandas 会给出混合类型警告；关闭分块推断可以让我们更完整地读取表格。

```python
import numpy as np
import pandas as pd

# 关闭分块类型推断，避免同一列中数字和字符串混合时被错误猜类型
train = pd.read_csv('../input/unimelb/unimelb_training.csv', low_memory=False)
test = pd.read_csv('../input/unimelb/unimelb_test.csv', low_memory=False)
sample = pd.read_csv('../input/unimelb/unimelb_example.csv', low_memory=False)

# 先确认数据规模和目标列，防止把测试集或提交样例当成训练集
print('train shape:', train.shape)
print('test shape:', test.shape)
print('sample shape:', sample.shape)
print(train[['Grant.Application.ID', 'Grant.Status']].head())
```

从材料得到的规模是训练集 `(8708, 252)`，测试集有 2,176 行。训练表的列从 `Grant.Application.ID` 延伸到 `Unnamed: 251`，既有数值列，也有 70 个对象类型列。我们接着查看每一列的缺失数量。缺失统计不能只看总的缺失单元格，因为“某一列完全为空”和“某一列只有少数几行为空”对建模的影响不同。

```python
# 统计每一列有多少个缺失值，并计算缺失比例，便于发现整列为空的字段
missing_count = train.isnull().sum().sort_values(ascending=False)
missing_rate = (missing_count / len(train)).sort_values(ascending=False)
missing_table = pd.DataFrame({
    'missing_count': missing_count,
    'missing_rate': missing_rate
})
print(missing_table.head(30))

# 统计每行的缺失数量，帮助判断是否有极端的空记录
row_missing = train.isnull().sum(axis=1)
print(row_missing.describe())
```

我们看到很多与第 14、15 位申请人有关的字段有 8,708 个缺失值，`Unnamed: 251` 也是全空列；`Home.Language.1` 有 7,910 个缺失，`Contract.Value.Band...see.note.A` 有 3,563 个缺失，`With.PHD.1` 有 3,595 个缺失。研究方向的 `RFCD` 字段通常有 634 个缺失，`SEO` 字段通常有 664 个缺失。这样的缺失模式说明申请团队人数并不固定，空值本身也可能表示“没有第 N 位成员”，不能把所有空值都简单理解为测量错误。

我们再查看目标变量。清理掉缺失较多的行后，材料中的 `Grant.Status` 分布为 0 类 3,902 条、1 类 3,341 条，两类数量比较接近。因此不需要为了平衡类别而大量复制样本，但训练时仍可以使用 `class_weight='balanced'`，避免模型过度偏向数量较多的类别。

```python
# 目标列只在训练集出现，先用 value_counts 检查是否存在严重类别不平衡
print(train['Grant.Status'].value_counts(dropna=False))

# 交叉统计目标和关键字段的组合，观察是否有明显的空值或类别异常
print(pd.crosstab(train['Grant.Status'], train['Start.date']).head())
```

### 2.2 数据清洗

我们首先尝试材料中较直接的一条路线：只保留前 40 列，把缺失比例很高、含义不稳定或难以编码的字段去掉，再删除剩余字段仍有空值的行。原 notebook 删除了 `Country.of.Birth.1`、`Sponsor.Code`、`Grant.Category.Code`、`Start.date`、合同金额、语言、博士学位和在校年数等字段，得到 31 列；继续 `dropna()` 后得到 7,243 行。这个方法简单，适合先建立基线，但它会丢掉不少样本。

```python
# 先保留目标列和前面一组较常用的项目、研究方向、第一位申请人字段
data1 = train.iloc[:, 1:40].copy()

# 删除缺失太多或需要额外业务解释的字段，避免少数空值主导模型
drop_cols = [
    'Country.of.Birth.1', 'Sponsor.Code', 'Grant.Category.Code',
    'Start.date', 'Contract.Value.Band...see.note.A',
    'Home.Language.1', 'With.PHD.1',
    'No..of.Years.in.Uni.at.Time.of.Grant.1'
]
data2 = data1.drop(columns=drop_cols)

# 这条基线只使用完整行，便于先观察模型是否能学习到信号
data3 = data2.dropna()
print('clean shape:', data3.shape)
print(data3['Grant.Status'].value_counts())
```

更完整的路线来自另一份材料。我们把训练集和测试集在行方向拼接，先把 `Grant.Status` 删除，再区分数值列和类别列。数值列缺失时填 0，类别列缺失时填字符串 `NA`。这里的 0 不是说真实数值一定为零，而是给模型一个统一的占位符；类别列使用 `NA` 是为了让“缺失”成为一个可学习的类别。之后我们把类别字段做独热编码，70 个类别列展开成 1,647 个列，再和 181 个数值列拼接。

```python
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

# 训练集和测试集必须使用完全相同的列，所以先拼接后编码
all_data = pd.concat([train, test], ignore_index=True)
X_all = all_data.drop(columns=['Grant.Status'])

# 数值缺失先填 0；这样不会把字符串混入数值矩阵
numeric_cols = X_all.select_dtypes(exclude=['object']).columns
X_real_zeros = X_all[numeric_cols].fillna(0.0)

# 类别缺失保留为一个明确的类别，避免直接丢掉整行
categorical_cols = X_all.select_dtypes(include=['object']).columns
X_cat = X_all[categorical_cols].fillna('NA').astype(str)

# 独热编码把“某个类别”变成 0/1 列，模型才能处理字符串字段
encoder = DictVectorizer(sparse=False)
X_cat_oh = encoder.fit_transform(X_cat.T.to_dict().values())

# 数值列尺度差异很大，标准化可以避免大数值列压过小数值列
scaler = StandardScaler()
X_real_scaled = scaler.fit_transform(X_real_zeros)
X_all_matrix = np.hstack([X_real_scaled, X_cat_oh])
print(X_real_zeros.shape, X_cat_oh.shape, X_all_matrix.shape)
```

材料中拼接后的数值列数量是 181，类别列数量是 70，独热编码后为 1,647 列，合计 1,828 个输入特征。我们取前 8,708 行作为训练部分、后 2,176 行作为测试部分。严格的工程实践会只用训练集估计标准化参数，避免测试信息进入预处理；这里为了复现材料，保留了“先拼接再编码”的做法，同时在解释结果时把它看作一种 Kaggle 中常见的特征对齐方法。

### 2.3 模型训练与超参数调优

我们先用逻辑回归作为基线。逻辑回归可以理解为：给每个特征一个权重，把所有特征加权后转换成成功概率。它速度快、容易解释，适合先确认数据清洗和编码是否正确。材料中的简单 30 特征版本在分层训练集上准确率约为 0.54，成功类别的召回率只有 0.18；只使用两个百分比字段的决策树准确率约为 0.54，说明少量字段不足以解决问题。使用 5 个近邻的 KNN 准确率约为 0.58，略有提升但仍然不是稳定方案。

我们随后采用带 L1 正则化的逻辑回归。L1 正则化会把不重要特征的权重压到 0，相当于一边训练一边做特征选择；面对独热编码后 1,828 个特征，这比把所有变量都强行保留下来更稳。`class_weight='balanced'` 会按照类别数量调整损失，防止模型只偏向数量较多的一类。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

# 先切出分层验证集，保证 0、1 两类在训练和验证中的比例接近
X_train, X_valid, y_train, y_valid = train_test_split(
    X_all_matrix[:len(train)], train['Grant.Status'],
    test_size=0.3, stratify=train['Grant.Status'], random_state=0
)

# L1 需要 liblinear 求解器；balanced 用来降低类别比例差异的影响
estimator_lasso = LogisticRegression(
    solver='liblinear', penalty='l1', class_weight='balanced'
)

# C 越大正则越弱，我们用交叉验证挑选一个不过度约束的值
param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
optimizer = GridSearchCV(estimator_lasso, param_grid, cv=3, n_jobs=-1)
optimizer.fit(X_train, y_train)

print('best C:', optimizer.best_params_)
print('best CV score:', optimizer.best_score_)
```

这次 3 折网格搜索的最佳参数是 `C=1`，材料记录的最佳交叉验证得分为 `0.837571206018007`。这个分数是交叉验证阶段的默认分类准确率，不应直接当成排行榜 AUC。它说明在“数值填补 + 类别独热 + L1 逻辑回归”的完整特征空间里，模型比只使用 30 个字段的基线更有学习能力。

材料中还尝试了随机森林。随机森林是许多棵决策树的组合，能够处理非线性关系。一个 notebook 设置了 `n_estimators=15`、`max_depth=15`，并报告了 0.94 的准确率；但它先用完整的 `X, Y` 拟合，再在从同一批数据切出的 `X_test` 上预测，训练集信息已经泄露到评估集，因此这个 0.94 不能作为可靠的泛化成绩。另一份 notebook 在同一训练数据上得到随机森林准确率 0.7283、AUC 0.8179，同样属于训练内评估，只能作为特征有信号的参考，不能替代严格的留出验证。

### 2.4 模型测试与指标报告

我们首先在没有参与拟合的验证集上报告准确率、分类报告和 AUC。分类报告能同时显示两类的精确率、召回率和 F1；AUC 则使用连续概率而不是硬分类结果，和比赛的概率排序更接近。代码中用 `[:, 1]` 取类别 1 的概率，避免把成功和失败的方向写反。

```python
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# 验证集只用于评估，不再调用 fit，避免把答案泄露给模型
valid_pred = optimizer.best_estimator_.predict(X_valid)
valid_prob = optimizer.best_estimator_.predict_proba(X_valid)[:, 1]

print('accuracy:', accuracy_score(y_valid, valid_pred))
print(classification_report(y_valid, valid_pred))
print('roc_auc:', roc_auc_score(y_valid, valid_prob))
```

对于简单清洗路线，材料给出的逻辑回归留出结果是准确率 0.54，0 类召回率 0.87、1 类召回率 0.18；KNN 的准确率是 0.58，宏平均 F1 约为 0.57。这些结果提醒我们：只看准确率可能会掩盖模型对成功申请的识别能力不足。我们应该优先使用完整特征版本，并用交叉验证选出的 `C=1` 重新拟合全部训练数据。

```python
# 选择交叉验证确认过的模型，并在全量训练数据上重新拟合
best_model = optimizer.best_estimator_
best_model.fit(X_all_matrix[:len(train)], train['Grant.Status'])

# 测试集没有标签，只输出每一行属于类别 1 的概率
test_matrix = X_all_matrix[len(train):]
test_probability = best_model.predict_proba(test_matrix)[:, 1]

# 提交文件保留申请编号，保证每个概率能对应到正确的申请
submission = pd.DataFrame({
    'Grant.Application.ID': test['Grant.Application.ID'],
    'Grant.Status': test_probability
})
submission.to_csv('submission.csv', index=False)
print(submission.head())
```

最后我们检查提交文件的行数、列名和概率范围。测试集有 2,176 行，所以提交文件也应有 2,176 条预测；`Grant.Application.ID` 必须来自测试集，`Grant.Status` 应该是 0 到 1 之间的概率。若比赛要求的提交样例列顺序不同，就应以 `unimelb_example.csv` 的列顺序为准，而不是凭记忆手工改名。

```python
# 提交前做最基本的格式检查，避免模型正确却因文件格式丢分
assert len(submission) == len(test)
assert list(submission.columns) == ['Grant.Application.ID', 'Grant.Status']
assert submission['Grant.Status'].between(0, 1).all()
assert submission['Grant.Application.ID'].equals(test['Grant.Application.ID'])
```

## 三、总结

本文中我们完成了 Unimelb 科研经费申请预测的完整流程。我们首先通过 EDA 确认了训练集有 8,708 行、252 列，测试集有 2,176 行，并发现第 14、15 位申请人的许多字段完全为空，其他项目、研究方向和申请人字段也存在不同程度的缺失。其次我们尝试了删除高缺失列和完整行的简单清洗路线，得到 7,243 行、31 列的基线数据；接着采用数值缺失填 0、类别缺失填 `NA`、独热编码和标准化的完整路线，把输入扩展为 1,828 个特征。然后我们使用带 L1 正则化和类别权重的逻辑回归，并对 `C` 在 `[0.01, 0.05, 0.1, 0.5, 1, 5, 10]` 上做 3 折搜索，得到最佳 `C=1`，材料记录的交叉验证准确率为 `0.837571206018007`。最后我们用未参与拟合的验证集检查准确率、分类报告和 ROC AUC，再用全量训练集拟合并生成测试概率提交文件。

我们还学到一个重要的评估教训：随机森林在训练集上报告的 0.94 准确率或 0.8179 AUC，并不能自动说明模型真的能预测未来申请，因为训练数据和评估数据可能已经混在一起。只有分层留出、交叉验证和提交前的格式检查同时做好，结果才更可信。对于这个比赛，下一步可以继续研究时间分层验证、类别编码方式以及随机森林等非线性模型，但每次改动都应保持独立验证，不能用训练内分数替代真实的泛化能力。
