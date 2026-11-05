# 信用评分贷款状态预测（findata-creditscoring）

## 一、比赛简介

### 1.1 比赛背景

信用评分（credit scoring）是金融风控里最经典的一类问题。银行或贷款机构在放款前会收集申请人的收入、负债、信用历史、工作年限、破产记录等信息，然后判断这笔贷款未来会不会出问题。如果模型能提前识别「高风险」贷款，机构就可以选择拒贷、提高利率或降低额度，从而减少损失。

这个比赛给出了一份贷款申请人的历史数据 `credit_train.csv`。每一行是一位借款人的贷款记录，目标是预测 `Loan Status`（贷款状态）。在原始数据里它通常是 `Charged Off`（已核销/违约）和 `Current`（正常还款）这样的文本标签，模型要做的就是判断这笔贷款最终属于哪一类。

数据有几个典型难点。第一，**缺失值多**：收入、信用分、逾期月数这些字段经常是空的，而且缺失本身可能就带有信息（不愿填收入的人风险可能不同）。第二，**类别列多**：贷款期限、房屋所有权、贷款用途、工作年限都是文本，需要编码成数字。第三，**类别不平衡**：正常还款的贷款通常远多于违约贷款，准确率会被多数类主导，必须看清楚模型到底学到了什么。

### 1.2 比赛目标

训练集 `credit_train.csv` 的每一行是一笔贷款，包含编号列（`Loan ID`、`Customer ID`）、若干数值特征、若干类别特征，以及目标列 `Loan Status`。我们要训练一个二分类模型，输入一位借款人的特征，输出这笔贷款的预测类别。

材料里出现过的特征列包括：`Term`（贷款期限）、`Credit Score`（信用分）、`Annual Income`（年收入）、`Years in current job`（当前工作年限）、`Home Ownership`（房屋所有权）、`Purpose`（贷款用途）、`Months since last delinquent`（距上次逾期的月数）、`Maximum Open Credit`（最大授信额度）、`Bankruptcies`（破产次数）、`Tax Liens`（税务留置次数）等。编号列不参与预测，只用于区分样本。

### 1.3 评价标准

材料里用 `accuracy_score` 评估模型，所以评价指标是**分类准确率（Accuracy）**：预测正确的样本数除以总样本数，越高越好，满分 1.0。

准确率直观，但在不平衡数据上容易骗人。假设 90% 的贷款都是正常还款，一个「全部预测为正常」的模型也能拿 0.9 的准确率，可它一个违约都抓不到。所以我们在报告准确率的同时，必须看混淆矩阵和每一类的精确率、召回率。这一点在下面会反复强调。

---

## 二、解题过程

我们的主线是：先做 EDA 摸清缺失和量纲，再做填补和编码，然后比较 MLP、随机森林等模型，最后用准确率和训练/测试分数差来选模型。过程中我们发现材料里的 notebook 有一个**非常严重的 bug**，会单独拿出来讲，因为它比任何调参技巧都更值得引以为戒。

### 2.1 数据探索（EDA）

我们首先读取训练集，查看列名、类型和缺失情况：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取贷款训练数据
data_scoring = pd.read_csv("../input/findata-creditscoring/credit_train.csv")

# 看前几行，确认字段含义和标签写法
data_scoring.head()

# 看每列的类型、非空数量
data_scoring.info()

# 统计每列缺失比例，按从高到低排序
missing = data_scoring.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0])
```

`info()` 会告诉我们哪些列是数值、哪些是文本。这一步很关键，因为后面填补和编码的策略要按类型分开定：数值列用中位数，类别列用众数或单独一类。

接着我们对几个关键数值列做描述统计，看看量纲和异常值：

```python
# 逐个看关键数值列的均值、标准差、最小最大和分位数
for col in ['Credit Score', 'Annual Income',
            'Months since last delinquent',
            'Maximum Open Credit', 'Bankruptcies', 'Tax Liens']:
    print(f"\n=== {col} ===")
    print(data_scoring[col].describe())
```

我们重点关注三件事：

- `Credit Score` 的正常范围大概在 300 到 850 之间，如果出现 0 或几千，说明有脏数据；
- `Annual Income` 的均值和最大值差距可能非常大，右偏严重；
- `Months since last delinquent` 和 `Maximum Open Credit` 的缺失比例通常很高，填补方式会明显影响模型。

然后看目标列的分布：

```python
# 目标列有多少种取值、各多少条
print(data_scoring['Loan Status'].value_counts())
print(data_scoring['Loan Status'].value_counts(normalize=True))
```

材料里没有保留具体输出，所以我们不引用具体条数。但从业务常识和数据规模看，正常还款的样本会占多数，这决定了后面不能只看准确率。

最后看类别列的取值分布。如果某一类占比极小（比如只有几笔），one-hot 之后会形成几乎全零的列，对模型没什么帮助，反而增加维度：

```python
# 看四个主要类别列各有哪些取值
for col in ['Term', 'Home Ownership', 'Purpose', 'Years in current job']:
    print(f"\n=== {col} ===")
    print(data_scoring[col].value_counts())
```

### 2.2 数据清洗

我们首先处理缺失值。材料里的策略很直接：`Years in current job` 是文本列，缺失统一填成 `'10+ years'`；其余数值列用**中位数**填补：

```python
# 工作年限是类别列，缺失单独归为「10 年以上」
# 这样填的理由是：年限缺失往往意味着资历较老或信息未采集
data_scoring['Years in current job'].fillna('10+ years', inplace=True)

# 数值列用中位数填补
# 中位数比均值更抗极端值，适合收入、授信这类右偏数据
for col in ['Credit Score', 'Annual Income',
            'Months since last delinquent',
            'Maximum Open Credit', 'Bankruptcies', 'Tax Liens']:
    data_scoring[col].fillna(data_scoring[col].median(), inplace=True)
```

这里用中位数而不是均值，是因为收入、授信额度这类字段常有极大的异常值，均值会被拉高，用它填补等于给缺失样本安上一个不真实的「高收入」标签。

其次我们把类别列做 one-hot 编码。one-hot 的意思是「一个类别一列，是就填 1，不是就填 0」，这样模型不会误以为类别之间有大小关系：

```python
# 对四个类别列做 one-hot，列名前缀保留原列名方便追溯
data_scoring = pd.concat([
    data_scoring,
    pd.get_dummies(data_scoring['Years in current job'], prefix='Years in current job'),
    pd.get_dummies(data_scoring['Term'], prefix='Term'),
    pd.get_dummies(data_scoring['Home Ownership'], prefix='Home Ownership'),
    pd.get_dummies(data_scoring['Purpose'], prefix='Purpose'),
], axis=1)

# 编码完成后删掉原始文本列，避免重复信息
data_scoring.drop(['Years in current job', 'Term', 'Home Ownership', 'Purpose'],
                  axis=1, inplace=True)
```

然后我们把目标列从文本转成 0/1。`factorize()` 会按出现顺序给每个唯一值编一个整数，并返回映射关系：

```python
# 把 Loan Status 的文本标签转成 0/1 整数
data_scoring['Loan Status'] = data_scoring['Loan Status'].factorize()[0]

# 确认目标只有两个取值
print(data_scoring['Loan Status'].unique())
```

最后切分特征和标签：

```python
# X 是特征，y 是标签
X = data_scoring.drop('Loan Status', axis=1)
y = data_scoring['Loan Status']

# 训练集 67%、测试集 33%，固定随机种子保证可复现
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)

print(X_train.shape, X_test.shape)
```

注意这里 `X` 里仍然包含 `Loan ID` 和 `Customer ID` 两列编号。材料前面虽然对它们做过 `factorize`，但 `drop` 时用了 `inplace=False` 且没有接收返回值，所以实际上并没有删掉。编号是随机标识，和违约风险没有关系，正确做法是显式删除。我们在这里指出来，并在下面修正：

```python
# 编号列对预测没有意义，必须真正删掉
X = X.drop(columns=['Loan ID', 'Customer ID'], errors='ignore')
```

### 2.3 模型训练与超参数调优

材料里比较了三个模型：MLP（多层感知机，一种神经网络）、线性回归、随机森林。这里要特别提醒：**线性回归是用来预测连续值的，不能直接做分类**；材料里却拿它和分类器放在一张表里比 `r2_score`，这是不严谨的。我们正确的比较对象应该是 MLP 和随机森林。

先看 MLP。材料里的参数是 `hidden_layer_sizes=(3,)`，意思是只有一个隐藏层、层里只有 3 个神经元。这个容量太小了，连特征之间的简单交互都学不全。我们改用 `(64, 32)` 两层，并保留 `max_iter=300`：

```python
from sklearn.neural_network import MLPClassifier

# 两层隐藏层：64 和 32 个神经元
# random_state 固定，保证每次训练结果一致
clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=300,
    random_state=1,
)

# 用真实数据训练——注意，不是用 make_classification 造的假数据
clf.fit(X_train, y_train)
```

再看随机森林。材料里的参数是 200 棵树、`criterion='entropy'`、`max_depth=3`、`min_samples_leaf=4`、`bootstrap=False`。`max_depth=3` 对随机森林来说非常浅，等于把每棵树限制成三个判断问题，模型会严重欠拟合。我们保留树数量和分裂准则，但把深度放开：

```python
from sklearn.ensemble import RandomForestClassifier

# 200 棵树，用信息熵衡量分裂质量
# max_depth=None 让树长到自然停止，再用 min_samples_leaf 控制过拟合
rfc = RandomForestClassifier(
    n_estimators=200,
    criterion='entropy',
    max_depth=None,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=33,
)

rfc.fit(X_train, y_train)
```

**这里必须停下来讲材料里最大的坑。** 原 notebook 在切分完真实数据之后，紧接着写了：

```python
# 原 notebook 里的这段代码会把真实数据覆盖掉
X, y = make_classification(n_samples=100, random_state=1)
train, test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
```

`make_classification` 是 sklearn 用来**生成玩具数据**的函数，`n_samples=100` 表示造 100 条完全虚构的样本。这两行一执行，前面辛苦清洗的真实贷款数据就被彻底替换了。之后训练的 MLP、打印的准确率，全都来自这 100 条假数据。更糟的是，第三个 notebook 在模型训练完之后才往 `data_scoring` 里追加「孩子数量」「房产价格」两列随机数，而那时光模型早就用旧的 `X` 训练完了，加了也白加。

所以我们在这个比赛里做的第一件「调参」其实是**删掉这两行**，让 `X`、`y` 始终指向真实数据。这不是模型问题，是数据泄漏/数据错位问题，但它对结果的影响比任何超参数都大。

### 2.4 模型测试与指标报告

修好数据之后，我们在测试集上评估两个模型，并同时看训练分数和测试分数：

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate(model, name):
    """输出训练/测试准确率，以及测试集的分类报告"""
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} 训练准确率：{train_acc:.4f}")
    print(f"{name} 测试准确率：{test_acc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))
    print(confusion_matrix(y_test, model.predict(X_test)))

evaluate(clf, 'MLP')
evaluate(rfc, 'Random Forest')
```

读结果时我们看三件事：

1. **测试准确率是否明显高于多数类基线。** 如果多数类占 80%，模型只有 0.81，那基本等于没学；
2. **训练和测试的差距。** 材料里用「差距小于 0.15」作为筛选条件，这个阈值比较宽松，但思路是对的：差距太大说明过拟合；
3. **混淆矩阵里少数类的召回率。** 违约样本被漏掉多少，直接决定模型有没有业务价值。

材料里还记录了一张「模型—训练分数—测试分数」的对比表，并用 `result_best = result[(train_score - test_score).abs() < 0.15]` 过滤掉过拟合的模型，再按测试分数排序。这个流程本身没问题，前提是分数必须来自真实数据。由于原 notebook 的分数来自 100 条合成样本，我们不引用那些数字，只保留这套评估方法。

如果要把模型用于提交，流程和训练时完全一致——取出测试表的特征列、做同样的填补和 one-hot、用训练好的模型预测：

```python
# 假设测试文件是 credit_test.csv，字段和训练集相同（不含 Loan Status）
test_df = pd.read_csv("../input/findata-creditscoring/credit_test.csv")

# 测试集也要经过和训练集完全相同的清洗步骤
# 注意：one-hot 的列必须和训练时对齐，缺少的列补 0
test_df['Years in current job'].fillna('10+ years', inplace=True)
for col in ['Credit Score', 'Annual Income', 'Months since last delinquent',
            'Maximum Open Credit', 'Bankruptcies', 'Tax Liens']:
    test_df[col].fillna(data_scoring[col].median(), inplace=True)

# 用训练好的随机森林预测
pred = rfc.predict(test_df[X.columns])

# 生成提交
pd.DataFrame({'id': test_df['Loan ID'], 'Loan Status': pred}).to_csv(
    'submission.csv', index=False
)
```

这里有一个容易忽略的点：测试集的 one-hot 列可能和训练集不完全一样（比如测试集里出现了一个训练集没见过的 `Purpose`）。正确做法是先 `reindex(columns=X.columns, fill_value=0)` 对齐，而不是直接 `predict`，否则列顺序错位会让模型输出完全错误的结果。

---

## 三、总结

本文中我们完成了信用评分贷款状态预测的完整流程，并纠正了材料里一个会毁掉全部结果的错误。我们首先读取 `credit_train.csv`，用 `info()` 和描述统计摸清了数值列、类别列和缺失情况，并确认了目标列 `Loan Status` 的分布。其次我们用中位数填补数值缺失、把工作年限缺失归为 `'10+ years'`、对四个类别列做 one-hot，并把目标转成 0/1。接着我们比较了 MLP 和随机森林，指出原 notebook 用 `make_classification` 生成的 100 条假数据覆盖了真实数据，导致所有分数失效，并修正了这个问题。最后我们用准确率、分类报告和混淆矩阵评估模型，强调在不平衡数据上必须看少数类的召回率，而不是只盯总准确率。

这个比赛给我们的最大教训是：**先检查数据流，再谈模型调参**。一行 `make_classification` 就能让前面所有 EDA 和清洗白做，而它藏在代码中间，不仔细读根本发现不了。另一个教训是，编号列、过浅的树、只有 3 个神经元的网络，这些「看起来能跑」的设置其实都在悄悄限制模型。把数据对齐、把容量调合理，比换更花哨的算法有用得多。
