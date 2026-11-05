# Mental Health 抑郁状态预测

## 一、比赛简介

### 1.1 比赛背景

Mental Health 数据集在 Kaggle 上常被用来研究公共健康和心理健康问题，也是入门机器学习时很常见的一个练习题目。数据来自人群调查，记录了个体的生活方式、人口统计、心理状态等信息，这些信息都是通过问卷和日常记录收集起来的，来源比较可信但整理起来比较琐碎。

抑郁是一种常见但容易被忽视的心理健康问题。很多人有了抑郁倾向却不知道，也没有及时得到帮助。提前判断一个人是否有抑郁风险，可以让医生或家人更早介入，提供必要的精神支持和干预措施，这对改善人群健康状况很有意义。所以这个比赛的目标不再只是把准确率刷高，而是希望借助机器学习，从已有的、容易获得的结构化数据中发现抑郁的早期信号。

由于心理健康标签往往比较敏感，而且这类标签一般靠医院或调查机构统计，所以数据规模通常不会很大，不同特征列也经常有缺失值。这给建模带来了两个难点：一是类别特征和数值特征混杂在一起，需要先做统一的预处理；二是缺失值如果处理不当，会让树模型在分裂时把缺失本身当成有价值的信息，影响稳定性。我们后面会一一应对这些问题。

### 1.2 比赛目标

给定训练集 `train.csv` 和测试集 `test.csv`。训练集中每条样本包含一个 `id` 列用于标识样本，一个目标列 `Depression` 告诉我们这个人是否抑郁，此外还有多个特征列。特征里既有数值型，比如各种得分、时长、次数一类的连续量，也有类别型，比如用字符串表示的选项，材料里按 `object` 类型识别出来。

我们要做的事情是：用训练集中的特征和标签教会模型什么样的人更容易有抑郁倾向，然后在没有标签的测试集上为每条样本预测出一个 `Depression` 值，最后按比赛要求把 `id` 和预测结果两列写成 `submission.csv` 提交。整个解决方案属于监督学习中的分类问题。

### 1.3 评价标准

从材料中的脚本来看，验证时使用 `sklearn.metrics.accuracy_score` 计算准确率，CatBoost 训练时也把 `eval_metric` 设成了 `'Accuracy'`，因此比赛的评分指标是分类准确率。准确率的含义很简单：把预测正确的样本数除以总样本数，得到的就是正确预测的比例。这个数值越接近 1.0 表示模型整体越好。

需要说明的是，准确率适合在类别分布比较均衡的时候使用。如果某个类别的样本特别多，即使模型总是猜多数类也能拿到很高的准确率，所以我们做 EDA 时也关注标签分布，尽量避免被这种假高分误导。

---

## 二、解题过程

整场比赛的思路可以概括为：先读入数据，把特征的类型和缺失情况摸清楚；其次做统一的数据清洗，包括去掉无意义的 id 列、填充缺失值、把类别特征编码成数字；接着用 CatBoost 训练分类器并在验证集上检查效果；最后对测试集预测并生成提交文件。下面按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先把训练集、测试集和示例提交文件读进来。示例提交文件很重要，它能告诉我们输出格式应该是什么样，避免最后因为格式不对而丢分。

```python
# 导入常用的数据处理库
import pandas as pd
import numpy as np

# 把三个文件分别读进来
TRAIN_PATH = '/kaggle/input/sch2-reg-2026-d3-1/train.csv'
TEST_PATH = '/kaggle/input/sch2-reg-2026-d3-1/test.csv'
SUBMISSION_PATH = '/kaggle/input/sch2-reg-2026-d3-1/sample_submission.csv'

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SUBMISSION_PATH)

print("Loading data...")
```

读入后我们需要确认几件事：数据有多少行多少列、特征有哪些、目标列叫什么、有没有缺失值。我们对数据的了解越清楚，后面的预处理就越有把握。材料中目标列被命名为 `Depression`，id 列命名为 `id`。id 只是样本编号，不像特征那样携带预测信息，所以我们在建模前把它去掉。

```python
# 配置目标列和 id 列
TARGET = 'Depression'
ID_COL = 'id'

# 按 dtype 判断类别特征和数值特征
cat_features = [col for col in train.columns 
                if train[col].dtype == 'object' and col != TARGET]
num_features = [col for col in train.columns 
                if train[col].dtype != 'object' and col != TARGET]

print(f"Categorical features: {len(cat_features)}")
print(f"Numerical features: {len(num_features)}")
```

从打印出的结果中我们可以看出，特征被分成了两类：类别特征是那些存字符串的列，数值特征是存数字的列。这个划分很关键，因为只有先知道特征类型，我们才知道每种特征该用哪种方式清洗。

接着我们看一下缺失值分布和标签分布，评估数据质量。

```python
# 统计每列缺失值个数
print(train.isna().sum())

# 查看标签的分布，判断类别是否均衡
print(train[TARGET].value_counts())

# 看特征数据的基本统计量
print(train.describe())
```

缺失值个数能告诉我们哪些列需要填充；标签分布能提醒我们准确率这个指标是否可信；`describe()` 则给出数值列最常见、最少见、平均水平的量级，帮助我们理解特征的含义。做完这几步，我们心里就有了一张清晰的数据地图，可以进入清洗阶段了。

### 2.2 数据清洗

数据本身没有明显重复，最需要处理的是两件事：缺失值和类别编码。

先处理缺失值。缺失值对于树模型来说有其特殊性：填错了会引入噪声，完全不管又可能让模型在缺失问题上过度拟合。材料采取的方案是简单而可靠的通用做法，类别特征用字符串 "Missing" 填充，数值特征用中位数填充。前者让模型把缺失当成一个独立类别看待，后者对整个分布的中枢影响最小。

```python
# 为类别特征填充缺失值：用 "Missing" 表示缺失，等于新增一个类别
for col in cat_features:
    train[col] = train[col].fillna('Missing')
    test[col] = test[col].fillna('Missing')

# 为数值特征填充缺失值：用训练集里该列的中位数
for col in num_features:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)
```

这里有个细节我们特别注意：数值列填充用的中位数是从训练集计算的，然后训练集和测试集都用同一个值。这样测试集的信息不会泄漏到训练里，避免影响我们对模型公平性的判断。

接着我们把 id 列去掉，因为它不携带任何可学习的规律，留着反而可能让模型记住编号。

```python
# 去掉 id 列，避免模型学到无意义的编号信息
if ID_COL in train.columns:
    train = train.drop(ID_COL, axis=1)
if ID_COL in test.columns:
    test = test.drop(ID_COL, axis=1)
```

最后是类别特征的编码。CatBoost 本身支持类别特征，但为了减少出错的概率、让流程更可控，材料选择先用 LabelEncoder 把每个类别映射成一个整数。这样每个取值对应一个唯一的数字，模型可以按数字区分不同类别。

```python
from sklearn.preprocessing import LabelEncoder

# 对每个类别特征做 Label 编码
for col in cat_features:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
```

这里把训练集和测试集拼在一起再拟合编码器，是一个容易踩坑的细节：如果只用训练集拟合，测试集里万一出现训练集没见过的新取值，`transform` 就会报错。合并后拟合，可以保证两边所有取值都有一一对应的数字，编码过程不会中断。

### 2.3 模型训练与超参数调优

清洗干净之后，我们把特征和标签拆开，准备训练模型。出于验证需要，我们从训练数据里再切出一部分当作验证集，用验证集评估我们的模型好坏，而不是直接去猜测试集的表现。

```python
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

X = train.drop(TARGET, axis=1)
y = train[TARGET]

# 80% 训练，20% 验证；固定随机种子让结果可复现
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

之所以固定 `random_state=42`，是为了让每次运行的划分都完全一样。这样我们调整参数时看到的效果差异只来自模型本身，而不是随机划分带来的波动，便于我们判断哪个参数更好。另外，80/20 是一个常见的默认比例：训练数据太少，模型学不到足够规律；验证数据太少，评估结果又不够稳定，这个比例在多数表格型任务上够用。

模型我们选用 CatBoost。它是一个梯度提升树模型，也就是很多棵决策树一棵接一棵地学习前一次的预测误差，最后把所有树的结果加起来作为最终预测。它对类别特征和缺失值都比较友好，训练速度也快，很适合这种表格型数据。

```python
model = CatBoostClassifier(
    iterations=2000,           # 树的数量，越多拟合越充分，但要防过拟合
    learning_rate=0.03,        # 学习率，越小每棵树的贡献越保守
    depth=6,                   # 树深度，控制模型的复杂度
    eval_metric='Accuracy',    # 优化目标：准确率
    verbose=200,               # 每 200 次迭代打印一次进度，方便观察
    early_stopping_rounds=100, # 验证分数连续 100 轮不提升就提前停止
    random_seed=42             # 固定随机种子，保证可复现
)

model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

`early_stopping_rounds=100` 是防止过拟合的关键一招：CatBoost 会在每一轮迭代后检查验证集表现，如果连续 100 轮都没有进步，就认为继续训练只会过拟合，主动停下来。这样 `iterations` 设得再大也不用担心训练过头，省时又省心。

### 2.4 模型测试与指标报告

训练结束后，第一步是在没见过的验证集上做预测，并计算准确率，这是我们最关心的指标。

```python
from sklearn.metrics import accuracy_score

# 在验证集上预测
val_pred = model.predict(X_val)

# 预测正确的样本占比就是准确率
acc = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {acc:.5f}")
```

准确率输出会告诉我们模型在整体上的表现。需要提醒的是，这里是在一个 80/20 随机切分的验证集上评估，它接近于测试集的表现，但并不是测试集本身。如果对结果满意，就可以进入最后的预测和提交环节。此外，验证集的划分也让我们可以在不提交的情况下反复试验不同方案，这是 Kaggle 比赛中控制风险的关键习惯。

确认模型可用后，我们做两个动作：一是用全部训练数据重新训练一次，让模型看到更多样本；二是对测试集生成预测，并按示例格式写出提交文件。

```python
# 用全部训练数据重新训练，充分利用样本
model.fit(X, y)

# 对测试集做预测
test_predictions = model.predict(test)

# 按示例提交文件的格式生成提交内容
submission = pd.DataFrame({
    'id': sample_submission['id'],
    'Depression': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("Success! 'submission.csv' saved.")
```

最后我们还要检查一下提交文件：列名是不是 `id` 和 `Depression`，行数是不是和测试集一样，前几行的 id 是否对得上。Kaggle 提交时格式错误会直接得零分，这个琐碎的检查能避免白白丢分，所以我们每次都会做。

---

## 三、总结

本文中我们完成了 Mental Health 抑郁状态预测比赛的完整流程。我们首先通过 EDA 读入了数据，按数据类型把特征划分成类别和数值两类，查了缺失值和标签分布，对数据有了整体把握；其次做了数据清洗，去掉无意义的 id 列，类别特征填 "Missing"、数值特征填中位数，并用 LabelEncoder 把类别编码成整数；接着用 CatBoost 训练分类器，配合验证集和早停机制保证泛化；最后在验证集上计算准确率，用全量数据重新训练并生成提交文件。

整个方案走的是入门建模的标准路径：先了解数据，再针对特征类型做清洗，然后选一个对表格数据友好的树模型，最后老老实实地评估和提交。虽然材料里只铺开了一条主线，但这条主线把每个环节背后的道理都讲清楚了，比如缺失值为什么填 "Missing"、编码为什么要合并训练集和测试集、为什么要固定随机种子、早停为什么有用。对我们这些入门者来说，这些为什么比一堆花哨技巧更有价值。如果以后想继续提升，可以考虑对比不同的树模型、尝试更多的超参数组合，这些都是在这条主线上自然的延伸。

所用页面：p000_01.txt
