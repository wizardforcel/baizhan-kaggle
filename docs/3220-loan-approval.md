# 3220 贷款审批挑战（Loan Approval）

## 一、比赛简介

### 1.1 比赛背景

银行在发放贷款之前，要先判断这笔钱给不给借。能自动、准确地判断"这位申请人的贷款会不会被批准"，对银行来说既省人力又能减少坏账。这个挑战给出一份贷款申请记录数据集，要求我们根据申请人的各项属性，预测他的贷款最终是否被批准（`loan_status`）。

数据里既有数字列也有文本列，文本列包括住房情况（`person_home_ownership`）、贷款等级（`loan_grade`）、是否有违约记录（`cb_person_default_on_file`）等；同时还包含少量缺失值。所以这个比赛主要考察三件事：把文本列变成数字、处理缺失值、再训练一个分类器。材料来源于一位参赛者的单份笔记本 `loan_challenge`，我们按它的思路整理成下面的流程。

### 1.2 比赛目标

训练集 `train.csv` 里的每一行是一位贷款申请人，包含编号 `id`、贷款金额 `loan_amnt`、贷款利率 `loan_int_rate`、还款占收入比例 `loan_percent_income`，以及我们要预测的目标列 `loan_status`（1 表示批准，0 表示拒绝）等。我们先用训练集学会从特征预测 `loan_status`，然后对测试集 `test.csv`（没有 `loan_status` 列）的每一行预测标签，最后按 `id + loan_status` 的格式生成提交文件。

### 1.3 评价标准

材料里没有明文写出官方评分指标，但从参赛代码看，模型输出的是 0/1 二分类标签，并用 `np.mean(y_pred == y_test)` 来评估性能——这算的就是猜对的比例。因此我们推断评测指标为**分类准确率**：预测对的样本数除以总样本数，越高越好。为保险起见，我们全程以准确率为主线来评价模型。

## 二、解题过程

整场比赛的思路可以概括成一句话：**先把文本列编码成数字、补好缺失值，再用一个从零实现的逻辑回归做二分类。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先读入训练集，看看都有哪些列、每列是什么类型：

```python
import pandas as pd

# 读入训练集；id 只是编号，对预测没用，先丢到一边
raw_data = pd.read_csv('../input/train.csv').drop('id', axis=1)
raw_data.head()

# 查看每列的名称、类型、缺失情况
raw_data.info()
```

`info()` 显示数据里既有 `int64`、`float64` 这样的数值列，也有好几列是 `object`（字符串）。字符串没法直接喂给模型，这是后面要重点处理的。我们还发现部分列存在缺失值，也需要想办法补上。

接着我们用 `describe()` 和直方图快速摸一遍数值特征的取值范围：

```python
import matplotlib.pyplot as plt

# 看每列的均值、标准差、最小/最大值等统计量
raw_data.describe()

# 给所有特征画直方图，快速检查分布形状
raw_data.hist(figsize=(16, 12))
plt.show()
```

`describe()` 让我们知道贷款金额、利率这些列大概落在什么区间；直方图则帮我们判断某个特征是不是严重偏斜——如果大部分样本都挤在一个很窄的范围里，模型很难从它身上学到有效的区分信息。

再来看目标本身长什么样，也就是批准和拒绝各占多少比例：

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 统计 loan_status 的类别分布，并画成柱状图
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=raw_data, x="loan_status")
# 在每个柱子顶上标出所占百分比
total = len(raw_data)
for p in ax.patches:
    percentage = f"{100 * p.get_height() / total:.1f}%\n"
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha="center", va="center")
plt.title("Loan Status Approved vs Rejected")
plt.show()
```

柱状图和 `value_counts()` 会告诉我们批准/拒绝两类各有多少条，这样就能看出数据有没有**类别不平衡**——如果某一类占比特别高，模型为了刷准确率会倾向于全猜它，必须心里有数。

我们还想知道哪些特征和"是否批准"关系最大。于是计算每个特征跟 `loan_status` 的相关系数，并画成热力图：

```python
# 计算各特征两两之间的相关系数，只看它们与目标 loan_status 的关系
corr = raw_data.corr()
sns.heatmap(
    corr[["loan_status"]].sort_values(by="loan_status", ascending=False),
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Feature Correlation with Approved Status")
plt.show()
```

相关系数取值在 -1 到 1 之间，越接近 0 说明关系越弱。我们根据这张图和排序后的相关系数，挑出与目标关系最密切的 6 个特征，后面只拿这些列去训练：

```python
# 选出的特征：贷款等级、还款占收入比、利率、住房情况、违约记录、贷款金额
features = ['loan_grade', 'loan_percent_income', 'loan_int_rate',
            'person_home_ownership', 'cb_person_default_on_file', 'loan_amnt']
```

### 2.2 数据清洗

这份数据要处理两件事：**文本列转数字**和**缺失值填充**。文本列我们用 sklearn 的 `LabelEncoder` 处理：把列里出现的每种取值依次编上号，比如 `租房` 编成 0、`自己买房` 编成 1，模型就能用了。缺失值我们逐列用该列的中位数来填，中位数比均值更抗极端值，是处理缺失的常见选择。

```python
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

def label_encoder(data):
    """找出所有字符串类型的列名"""
    names = data.columns
    encode = []
    for i in range(len(names)):
        if data[names[i]].dtypes == object:
            encode.append(names[i])
    print(encode)
    return encode

def label_transform(data, encode):
    """把字符串列按出现顺序转成整数编号"""
    for x in encode:
        data[x] = le.fit_transform(data[x])
        print(le.classes_)

def clean(data):
    """用每列的中位数填充缺失值"""
    cols = data.columns
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
    return data

# 先补缺失，再把文本列编码成数字
raw_data = clean(raw_data)
label_transform(raw_data, label_encoder(raw_data))
```

清洗之后再用 `info()` 和 `isnull().sum()` 各看一眼，确认所有列都已经变成数值类型、缺失值数量为 0，就可以放心建模了。

需要提醒的是：`LabelEncoder` 编出来的号码只是"第几种取值"，数字大小本身没有含义。对 `loan_grade` 这种本身就有顺序的列（等级越高风险越大）问题不大，但对 `person_home_ownership` 这类纯类别列，编号会给线性模型人为制造"租房 > 买房"的错觉。更讲究的做法是用独热编码，不过本方案为了流程简单沿用了编号，我们可以把它当作后续可优化的点。

### 2.3 模型训练与超参数调优

首先把特征和目标拆开，并对特征做**标准化**：把每个特征的均值变成 0、标准差变成 1，公式是 `z = (x - 均值) / 标准差`。为什么必须这么做？我们准备用梯度下降来训练模型，如果某个特征数值特别大（比如贷款金额动辄上万，而 0/1 标志最多是 1），梯度会在不同方向上尺度悬殊，收敛很慢甚至"爆炸"；统一尺度能加快收敛，也让正则化对每个特征更公平。

```python
import numpy as np

target = ['loan_status']
# 取出选定的 6 个特征，转成浮点数
X_raw = raw_data[features].values.astype(float)
y = raw_data[target].values

# 标准化：z = (x - mean) / std
mean = np.mean(X_raw, axis=0)
std = np.std(X_raw, axis=0)
X_scaled = (X_raw - mean) / std
```

接下来划分训练集和测试集。我们手动实现一次按 80% / 20% 的比例切分：先把所有样本的下标随机打乱，前 80% 当训练集、后 20% 当测试集。这里固定随机种子 `seed(42)`，保证每次运行的切分结果一致，方便复现实验。

```python
np.random.seed(42)                                # 固定随机种子，实验可复现
indices = np.random.permutation(len(X_scaled))    # 随机打乱所有样本的下标
split_idx = int(0.8 * len(X_scaled))              # 前 80% 当训练集

train_idx, test_idx = indices[:split_idx], indices[split_idx:]
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
```

这次我们决定自己从零实现一个逻辑回归（Logistic Regression），它是分类任务里最基础也最好解释的模型：先把特征的线性组合 `w·x + b` 算出来，塞进 sigmoid 函数，得到一个 0 到 1 之间的概率，概率大于 0.5 就预测为 1。我们采用**小批量梯度下降**来学习权重：每一轮随机抽一小批（默认 32 条）样本，计算梯度，沿负梯度方向小步更新参数；同时支持 L1 / L2 正则化，防止把权重学得过大导致过拟合。

```python
class LogisticRegressionFromScratch:
    """从零实现的逻辑回归，训练方式是小批量梯度下降"""
    def __init__(self, lr=0.01, n_iters=1000, batch_size=32,
                 l1_ratio=0.0, l2_ratio=0.0):
        self.lr = lr              # 学习率：每次更新迈多大步子
        self.n_iters = n_iters    # 迭代轮数
        self.batch_size = batch_size
        self.l1_ratio = l1_ratio  # L1 正则强度（Lasso）
        self.l2_ratio = l2_ratio  # L2 正则强度（Ridge）
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # 先把输入夹在 [-500, 500]，防止 exp 计算溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        num_samples, num_features = X.shape

        self.weights = np.zeros(num_features)   # 权重从 0 开始
        self.bias = 0.0

        for _ in range(self.n_iters):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)          # 每轮先打乱，防止批次规律固定

            for start in range(0, num_samples, self.batch_size):
                batch = indices[start:start + self.batch_size]
                X_batch, y_batch = X[batch], y[batch]
                b = len(X_batch)

                # 1. 前向传播：线性组合 -> sigmoid，得到预测概率
                linear_pred = np.dot(X_batch, self.weights) + self.bias
                y_pred = self._sigmoid(linear_pred)

                # 2. 计算梯度：预测与真实值的差是梯度的核心
                error = y_pred - y_batch
                dw = (1 / b) * np.dot(X_batch.T, error)
                db = (1 / b) * np.sum(error)

                # 3. 加上正则项对梯度的贡献
                if self.l1_ratio > 0:
                    dw += self.l1_ratio * np.sign(self.weights)  # L1 让部分权重变 0
                if self.l2_ratio > 0:
                    dw += self.l2_ratio * self.weights            # L2 让权重整体变小

                # 4. 沿负梯度方向更新参数
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)

    def predict(self, X, threshold=0.5):
        # 概率超过 0.5 判为批准(1)，否则拒绝(0)
        prob = self.predict_proba(X)
        return np.where(prob >= threshold, 1, 0)
```

关于超参数，我们做了几次小规模试验后选定：学习率 `lr=0.08`（太大容易来回震荡、太小收敛太慢）、迭代 `n_iters=3000` 轮、L2 正则强度 `l2_ratio=0.01`（轻微约束权重防止过拟合），L1 设为 0 因为我们并不追求把某些权重压成 0。

### 2.4 模型测试与指标报告

训练完成后，我们在留下没见过的 20% 数据上测试：

```python
model = LogisticRegressionFromScratch(lr=0.08, n_iters=3000, l2_ratio=0.01)
model.fit(X_train, y_train)

# 在测试集上预测，准确率 = 猜对的比例
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
```

笔记本打印出了测试集上的准确率（原笔记本的打印输出没有把具体数值保留下来，所以这里以我们自己跑出来的打印结果为准）。因为逻辑回归学到的权重可以解释，我们还把每个特征对应的权重打印出来，看模型认为谁更重要：

```python
# 打印每个特征学到的权重：绝对值越大说明它对结果越有判别力
print("=====  Trained Model Weights ======")
for name, w in zip(features, model.weights):
    print(f'{name:<28} | {w:>10.4f}')
print(f'{"Intercept / Bias (b)":<28} | {model.bias:>10.4f}')
```

最后处理测试集并生成提交文件。测试集要和训练集走**一模一样的清洗流程**：去掉 `id`、只保留选中的特征、把文本列编码成数字、再标准化，然后用训练好的模型预测并组装成比赛要求的格式。

```python
# 读入测试集，去掉 id、只留选中的特征
test_data = pd.read_csv('../input/test.csv').drop('id', axis=1)
test = test_data[features].copy()

# 沿用训练时的标签编码（更严谨的做法是把训练时学到的映射拿过来用）
label_transform(test, label_encoder(test))
test = test.values.astype(float)

# 标准化（理想情况应该用训练集的均值/标准差来转换测试集）
mean = np.mean(test, axis=0)
std = np.std(test, axis=0)
test_scaled = (test - mean) / std

# 预测并组装提交文件：id + loan_status
y_pred = model.predict(test_scaled)
col0 = pd.read_csv('../input/sample_submission.csv').drop('loan_status', axis=1)
col1 = pd.Series(y_pred, name='loan_status')
submission = pd.concat([col0, col1], axis=1)
submission.to_csv('submission.csv', index=False)
```

## 三、总结

本文中我们完成了 3220 贷款审批挑战的完整流程。我们首先用 EDA 查看了类别分布和特征与目标的相关系数，从中挑出 6 个最重要的特征；其次做了两步清洗——用 `LabelEncoder` 把文本列变成数字、用中位数填补缺失值；接着对特征做标准化，并手写了一个带小批量梯度下降和 L1/L2 正则化的逻辑回归，选定学习率 0.08、迭代 3000 轮、L2 强度 0.01；最后在保留的 20% 数据上打印准确率，对测试集按 `id + loan_status` 格式生成提交文件。

这场比赛数据量不大、模型也比较基础，但完整展示了"二分类 + 逻辑回归"从数据到提交的常规套路。我们最大的体会是：**清洗和特征选择常常比模型本身更决定分数下限**——文本列编码得对不对、缺失值补得合不合理、特征选得好不好，都会直接体现在最终准确率上。