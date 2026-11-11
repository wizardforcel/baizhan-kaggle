# Tutors - Expected Math Exam Results

## 一、比赛简介

### 1.1 比赛背景

这个比赛的名字可以直译为“导师预期的数学考试成绩”。从 Kaggle 页面标题和材料中的字段可以确认，题目希望我们根据导师及课程相关信息，预测学生的平均数学考试成绩。数据不是一张数学试卷，而是一张描述辅导关系的表格：每一行代表一名学生或一条辅导记录，里面同时有导师的背景、授课价格、经验年限，以及学生在多个科目上的情况。我们要做的是一个**回归问题**。回归的意思很简单：结果不是几个固定类别，而是一个连续的数字，例如预测平均考试分数为 70.5。

材料中出现的字段包括 `age`（年龄）、`years_of_experience`（教学或辅导经验年数）、`lesson_price`（课程价格）、`qualification`（资格或资质等级），以及 `physics`、`chemistry`、`biology`、`english`、`geography`、`history` 等科目相关字段。训练集还包含目标列 `mean_exam_points`，测试集则没有这列。`Id` 只是提交时用来识别样本的编号，不能把它当成有教育意义的特征。

这个题目的难点并不是文本处理，而是如何把多个数值特征组合起来。年龄、经验、资质和价格可能有不同的量纲；有些变量单独看作用不大，但组合起来可能反映导师的教学能力。材料中的方案没有大量复杂的特征工程，而是用直方图和相关性热力图先了解数据，再加入一个“资质乘以经验”的交互特征，最后使用自己实现的回归树和梯度提升模型。这里的梯度提升可以理解成：先用一棵树做一个不太好的预测，再让下一棵树专门修正前一棵树的错误，重复多次以后得到更稳定的结果。

### 1.2 比赛目标

我们的输入是训练表和测试表。训练表至少包含以下列：

- `Id`：样本编号；
- `age`：年龄特征；
- `years_of_experience`：经验年数；
- `lesson_price`：课程价格；
- `qualification`：资格或资质数值；
- `physics`、`chemistry`、`biology`、`english`、`geography`、`history`：各科目相关数值；
- `mean_exam_points`：训练集中的目标值，即平均数学考试分数。

测试集使用相同的输入特征，但不提供 `mean_exam_points`。我们的任务是为测试集的每一行预测一个平均考试分数，并按照 `Id,mean_exam_points` 两列写入提交文件。预测值必须是连续数值，而不是四舍五入后的类别标签。

### 1.3 评价标准

材料中的两个方案都把回归结果写成了 `R2_score`，并且同时定义了均方误差 MSE。决定系数 R² 用来衡量模型比“直接预测平均值”好多少，计算形式是

\[
R^2=1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}。
\]

其中，`y_i` 是真实分数，`\hat y_i` 是预测分数，`\bar y` 是真实分数的平均值。R² 越接近 1，说明模型解释的数据变化越多；如果 R² 接近 0，说明模型和直接猜平均值差不多；如果低于 0，说明模型还不如这个简单基线。MSE 则是预测误差平方的平均值，越小越好。

两份材料没有给出 Kaggle 排行榜的官方评分说明或最终分数，所以我们不把本地 R² 当作官方排行榜分数。本文把代码中明确实现的 R² 和 MSE 作为本地验证指标，提交时仍应以比赛页面的隐藏测试评分规则为准。这样写可以避免把本地实验结果和平台最终分数混为一谈。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取训练集和测试集，然后用 `head`、`describe`、直方图和相关性热力图检查数据。EDA 是探索性数据分析的缩写，目的不是立刻训练模型，而是先回答几个基本问题：列名是否正确、数值是否能正常读取、各变量的范围是否异常、特征之间是否存在明显关系。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 先读入训练集和测试集，训练集有目标列，测试集没有目标列
train = pd.read_csv('/kaggle/input/tutors-expected-math-exam-results/train.csv')
test = pd.read_csv('/kaggle/input/tutors-expected-math-exam-results/test.csv')

# 先看几行，确认文件路径和列名没有读错
print(train.head())
print(test.head())

# describe 能快速给出均值、标准差、最小值和分位数，便于发现异常范围
print(train.describe().T)
print(test.describe().T)
```

我们其次查看每个数值列的分布。材料中的代码分别对训练集和测试集画了直方图，并使用 15 个箱子展示数值大致集中在哪里。直方图对于入门者很有帮助：如果某列大多数值挤在很窄的区间，或者出现少量远离主体的值，就值得进一步检查。这里我们不凭空删除异常值，因为材料没有证明这些值是录入错误；它们可能正是比赛数据的一部分。

```python
# 训练集和测试集分开画图，便于比较两者的数值范围是否相近
train.hist(figsize=(12, 14), bins=15, grid=True)
plt.suptitle('训练集各数值列的分布')
plt.show()

test.hist(figsize=(12, 14), bins=15, grid=True)
plt.suptitle('测试集各数值列的分布')
plt.show()
```

接着我们计算相关性矩阵。相关性可以粗略表示两个数值列是否一起增大或一起减小，范围通常在 -1 到 1 之间。它不是因果关系，也不能单独决定删掉某个字段，但可以帮助我们了解目标列与输入列的线性关系，并观察特征之间是否有重复信息。

```python
# 只对数值列计算相关性，避免把 Id 等非解释性字段混进分析
corr = train.corr(numeric_only=True)

# 颜色越深，表示线性相关程度的绝对值越高；数字便于精确查看
plt.rcParams['figure.figsize'] = (14.0, 12.0)
sns.heatmap(corr, annot=True, fmt='.2g', vmin=-1, vmax=1,
            center=0, linewidths=2, cmap='GnBu')
plt.title('训练集相关性热力图')
plt.show()
```

### 2.2 数据清洗

我们然后进行数据清洗和特征准备。材料没有报告缺失值填补、重复行删除或类别编码，因此我们不添加未经证实的处理。这里最重要的清洗动作是把 `Id` 排除在模型特征之外，把 `mean_exam_points` 单独拿出来当目标。这样可以防止模型把样本编号误认为有实际含义。

材料中的第一个方案还加入了一个交互特征：`qualification * years_of_experience`。交互特征就是把两个字段相乘，让模型能够直接看到“资质和经验同时较高”这种组合，而不是完全依靠树自己一步一步找出关系。

```python
# 交互特征表达“资质和经验共同作用”，训练集和测试集必须同时创建
train['qualification * years_of_experience'] = (
    train['qualification'] * train['years_of_experience']
)
test['qualification * years_of_experience'] = (
    test['qualification'] * test['years_of_experience']
)

# 测试集没有目标列，因此从测试集去掉 Id 后得到可用特征名
features = test.drop(columns='Id').columns
target = 'mean_exam_points'

# 用完全相同的列顺序生成训练矩阵和测试矩阵，避免列错位
X = train[features].values
y = train[target].values
test_set = test[features].values

print('训练特征形状:', X.shape)
print('目标形状:', y.shape)
print('测试特征形状:', test_set.shape)
```

材料还展示了主成分分析 PCA 和标准化函数。PCA 可以把多个相关变量压缩成更少的综合变量；标准化则把每列变成类似的尺度。这里的代码只对 `years_of_experience`、`lesson_price`、`qualification` 做了一个主成分，并且把把 PCA 结果拼回特征矩阵的语句注释掉了。因此我们把它作为认识方法的实验，不把它误写成最终模型必需步骤。

```python
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, data):
        # 转成浮点数，保证协方差计算和矩阵乘法不会因为整数截断
        data_ = data.astype(float)
        cov_matrix = data_.T.dot(data_)
        eig_values, eig_vectors = np.linalg.eig(cov_matrix)

        # 按特征值从大到小排列，优先保留能解释更多变化的方向
        eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:, i])
                     for i in range(len(eig_values))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        w = np.hstack([eig_pairs[i][1].reshape(data.shape[1], 1)
                       for i in range(self.n_components)])
        return data_.dot(w)

# 只把三个相关字段压成一个主成分，观察它能否代表共同变化
pca = PCA(n_components=1)
train_component = pca.fit_transform(
    train[['years_of_experience', 'lesson_price', 'qualification']]
)
print(train_component[:5])
```

### 2.3 模型训练与超参数调优

我们接着采用材料的主线：自己实现回归树，再把多棵树组合成随机批量梯度提升。回归树的每个内部节点都会选择一个特征和阈值，把数据分成左右两组；叶子节点直接输出这一组目标值的平均数。材料把叶子预测写成 `np.mean`，并把最小叶子样本数设置为 5，这能避免树继续切分到只有极少数样本的区域。

```python
class Node:
    def __init__(self, index, t, true_branch, false_branch):
        # index 表示用哪一列比较，t 表示比较用的阈值
        self.index = index
        self.t = t
        self.true_branch = true_branch
        self.false_branch = false_branch


class Leaf:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        # 回归叶子用平均值，比用某一条样本的值更稳定
        self.prediction = np.mean(self.labels)


class Tree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    def dispersion(self, labels):
        # 用标准差表示节点内目标值的分散程度，越小表示越集中
        return np.std(labels)

    def quality(self, left_labels, right_labels, current_dispersion):
        # 按左右子节点样本比例加权，衡量切分后分散程度下降了多少
        p = len(left_labels) / (len(left_labels) + len(right_labels))
        return (current_dispersion - p * self.dispersion(left_labels)
                - (1 - p) * self.dispersion(right_labels))
```

我们在分裂节点时遍历特征和候选阈值，只保留左右两边都至少有 5 个样本的切分。材料中的 `find_best_split` 还可以随机抽取部分特征索引，这就是随机化的一部分：每次树不一定都查看全部特征，既能降低计算量，也能让多棵树之间有差异。

```python
    def split(self, data, labels, index, t):
        # 小于等于阈值的一侧进入 true 分支，其余进入 false 分支
        left = np.where(data[:, index] <= t)
        right = np.where(data[:, index] > t)
        return data[left], data[right], labels[left], labels[right]

    def find_best_split(self, data, labels):
        min_leaf = 5
        current_dispersion = self.dispersion(labels)
        best_quality, best_t, best_index = 0, None, None

        # 逐列寻找能够最大幅度降低分散程度的切分
        for index in range(data.shape[1]):
            for t in np.unique(data[:, index]):
                true_data, false_data, true_labels, false_labels = \
                    self.split(data, labels, index, t)
                # 子节点太小会让预测不稳定，所以跳过这种切分
                if len(true_data) < min_leaf or len(false_data) < min_leaf:
                    continue
                current_quality = self.quality(
                    true_labels, false_labels, current_dispersion
                )
                if current_quality > best_quality:
                    best_quality = current_quality
                    best_t = t
                    best_index = index
        return best_quality, best_t, best_index
```

然后我们把多棵树组合成梯度提升模型。材料中的参数是 `n_trees=10`、`max_depth=16`、`alpha=0.1`、`iter_num=1500`。`alpha` 可以理解为每一轮修正的步长，值太大可能会冲过头，值太小则需要更多轮迭代。树的数量和深度控制模型容量：树越多、越深，模型越灵活，但也更容易记住训练集。

```python
class StGradBoost:
    def __init__(self, max_depth, n_trees, alpha, coef, iter_num):
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.alpha = alpha
        self.coef = coef
        self.iter_num = iter_num
        self.trees = []

    # fit 和 predict 的具体树构建过程按材料中的 Tree 实现调用
    def fit(self, X, y):
        # 每次重新训练前清空旧树，避免重复调用 fit 时累积模型
        self.trees = []
        # 先把第一棵树的平均预测作为初始结果，再逐轮修正残差
        # 这里保留为与材料一致的模型接口，实际 build_tree 使用 Tree 类
        return self

    def predict(self, X):
        # 多棵树的预测按 alpha 和系数加权后求和
        return np.array([
            sum(self.alpha * coef * tree.predict([x])[0]
                for tree, coef in zip(self.trees, self.coef))
            for x in X
        ])
```

上面代码展示了模型接口和核心组合方式；材料中的完整 `fit` 还包括残差计算、随机批量取样和重复迭代。我们调参时不采用大网格搜索，因为每组参数都要重新构造许多树，计算成本较高。我们先固定一套清楚、可复现的参数，再通过验证集的 R² 和 MSE 判断是否值得调整。材料中最终示例使用 10 棵树、最大深度 16、步长 0.1 和 1500 次迭代，这些参数可以作为入门基线。

### 2.4 模型测试与指标报告

最后我们把训练数据随机打乱，再按 70% 和 30% 划分训练集与验证集。材料中的 `tts` 函数固定 `random_state=42`，所以重复运行时划分结果一致。固定随机种子很重要：如果每次验证集不同，我们就无法判断分数变化究竟来自模型改动，还是来自数据划分变化。

```python
def tts(X, y, test_size, random_state):
    # 固定种子，让每次实验都使用相同的随机排列
    np.random.seed(random_state)
    cut = int(len(X) * (1 - test_size))
    order = np.random.permutation(X.shape[0])
    X_shuffled, y_shuffled = X[order], y[order]

    # 前 70% 用来学习，后 30% 只用于检查泛化能力
    return (X_shuffled[:cut], X_shuffled[cut:],
            y_shuffled[:cut], y_shuffled[cut:])

X_train, X_valid, y_train, y_valid = tts(
    X, y, test_size=0.3, random_state=42
)

n_trees = 10
coefs = [1] * n_trees
model = StGradBoost(max_depth=16, n_trees=n_trees,
                    alpha=0.1, coef=coefs, iter_num=1500)
model.fit(X_train, y_train)
```

我们分别计算训练集和验证集上的 R²。训练集分数很高而验证集明显较低，通常意味着模型过拟合；两者都较低，则说明模型还没有学到足够规律。材料只给出了 R² 和 MSE 的实现，没有给出打印出的具体数值，因此我们在 writeup 中不虚构分数，而是把计算过程保留下来。

```python
def r2_score(y_pred, y_real):
    # 分母表示真实值自身的总波动，分子表示预测残差的平方和
    ss_total = np.sum((y_real - np.mean(y_real)) ** 2)
    ss_res = np.sum((y_real - y_pred) ** 2)
    return 1 - ss_res / ss_total


def mean_squared_error(y_real, prediction):
    # MSE 对大误差更敏感，适合观察预测偏差是否集中在少数样本
    return np.mean((y_real - prediction) ** 2)

y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)
print('训练 R2:', r2_score(y_train_pred, y_train))
print('验证 R2:', r2_score(y_valid_pred, y_valid))
print('验证 MSE:', mean_squared_error(y_valid, y_valid_pred))
```

验证方案确定后，我们最后使用全部训练数据重新拟合模型，再预测测试集。提交文件保留测试集原来的 `Id`，把预测数组命名为 `mean_exam_points`。写文件后还应该检查列名、行数和前几行，避免模型已经训练正确，却因为提交格式错误而无法评分。

```python
# 验证参数确定后，用全部带标签的数据学习，尽量利用可用样本
model.fit(X, y)
test_pred = model.predict(test_set)

# Kaggle 提交需要保留 Id，并把预测列写成目标列名
submission = pd.concat([test['Id'], pd.Series(test_pred)], axis=1)
submission = submission.rename(columns={0: 'mean_exam_points'})

# index=None 避免把 pandas 的行号额外写成一列
submission.to_csv('StGBoost_predict.csv', index=False)
print(submission.head())
print('提交行数:', len(submission))
```

## 三、总结

本文中我们完成了 Tutors - Expected Math Exam Results 的一套回归解题流程。我们首先通过 `head`、`describe`、直方图和相关性热力图检查数据，确认了问题围绕 `mean_exam_points` 展开，输入包括年龄、经验、课程价格、资格以及多个科目字段。其次我们进行保守的数据清洗：排除 `Id`，分离目标列，并按照材料加入 `qualification * years_of_experience` 交互特征；对于没有证据表明是错误的数据，我们没有擅自删除。接着我们学习了 PCA、标准化和回归树的作用，并以材料中的树模型为基础，把多棵回归树组合成随机批量梯度提升模型，采用最大深度 16、10 棵树、步长 0.1 和 1500 次迭代的示例参数。然后我们用固定随机种子的 70%/30% 划分做本地验证，用 R² 和 MSE 同时观察拟合效果与误差大小。最后我们用全部训练数据重新训练，生成包含 `Id` 和 `mean_exam_points` 的提交文件。

这份材料带给我们的重要经验是，入门阶段不一定要马上堆叠很多复杂库。我们可以先明确目标列和特征列，再用直方图、相关性和一个简单的交互特征理解数据；随后把树的切分、叶子平均值、残差修正和评估公式写出来。这样做虽然不保证得到排行榜最高分，却能把梯度提升从一个黑盒模型变成一组可以逐步检查的步骤。对于下一轮实验，我们可以在不改变数据划分的前提下比较树深度、树数量和步长，并同时观察训练 R²、验证 R² 与 MSE，防止只追求训练集分数而忽略模型在新样本上的表现。
