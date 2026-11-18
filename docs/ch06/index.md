# 第6章 表格数据比赛详解：贷款审批（Loan Approval）

3220 贷款审批挑战是一个适合新手入门的表格二分类比赛。银行在发放贷款之前，要先判断这笔钱给不给借。能自动、准确地判断"这位申请人的贷款会不会被批准"，对银行来说既省人力又能减少坏账。

这个比赛给出一份贷款申请记录数据集，要求我们根据申请人的各项属性，预测他的贷款最终是否被批准（`loan_status`）。数据里既有数字列也有文本列，还包含少量缺失值。所以这个比赛主要考察三件事：把文本列变成数字、处理缺失值、再训练一个分类器。

与房价预测（回归）和信用评分（AUC 指标）不同，这个比赛主要考察**分类准确率**，且模型可以从零实现逻辑回归，非常适合理解"二分类 + 逻辑回归"从数据到提交的完整套路。

---

## 6.1 比赛背景与数据

### 比赛背景

银行在发放贷款之前，要先判断这笔钱给不给借。能自动、准确地判断"这位申请人的贷款会不会被批准"，对银行来说既省人力又能减少坏账。

### 数据结构

训练集 `train.csv` 里的每一行是一位贷款申请人，包含编号 `id`、贷款金额 `loan_amnt`、贷款利率 `loan_int_rate`、还款占收入比例 `loan_percent_income`，以及我们要预测的目标列 `loan_status`（1 表示批准，0 表示拒绝）等。文本列包括住房情况（`person_home_ownership`）、贷款等级（`loan_grade`）、是否有违约记录（`cb_person_default_on_file`）等。

测试集 `test.csv` 没有 `loan_status` 列。我们要对测试集的每一行预测标签，最后按 `id + loan_status` 的格式生成提交文件。

### 评价标准

材料里没有明文写出官方评分指标，但从参赛代码看，模型输出的是 0/1 二分类标签，并用 `np.mean(y_pred == y_test)` 来评估性能——这算的就是猜对的比例。因此我们推断评测指标为**分类准确率**：预测对的样本数除以总样本数，越高越好。

---

## 6.2 探索性数据分析（EDA）

在清洗和建模之前，先把数据摊开看，了解都有哪些列、每列是什么类型、有没有缺失值、目标类别是否平衡、哪些特征与目标关系更密切。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读入训练集；id 只是编号，对预测没用，先丢到一边
raw_data = pd.read_csv('/kaggle/input/3220-loan-approval/train.csv').drop('id', axis=1)

# 查看每列的名称、类型、缺失情况
print(raw_data.info())
```

`info()` 显示数据里既有 `int64`、`float64` 这样的数值列，也有好几列是 `object`（字符串）。字符串没法直接喂给模型，这是后面要重点处理的。我们还发现部分列存在缺失值，也需要想办法补上。

接着我们用 `describe()` 和直方图快速摸一遍数值特征的取值范围：

```python
# 看每列的均值、标准差、最小/最大值等统计量
print(raw_data.describe())

# 给所有特征画直方图，快速检查分布形状
raw_data.hist(figsize=(16, 12))
plt.show()
```

`describe()` 让我们知道贷款金额、利率这些列大概落在什么区间；直方图则帮我们判断某个特征是不是严重偏斜——如果大部分样本都挤在一个很窄的范围里，模型很难从它身上学到有效的区分信息。

再来看目标本身长什么样，也就是批准和拒绝各占多少比例：

```python
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=raw_data, x="loan_status")
total = len(raw_data)
for p in ax.patches:
    percentage = f"{100 * p.get_height() / total:.1f}%"
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha="center", va="center")
plt.title("Loan Status Approved vs Rejected")
plt.show()

print(raw_data['loan_status'].value_counts())
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

---

## 6.3 数据清洗

这份数据要处理两件事：**文本列转数字**和**缺失值填充**。

### 文本列转数字

机器学习模型只认数字，不能直接处理 "租房"、"自己买房" 这样的文字。我们用 sklearn 的 `LabelEncoder` 处理：把列里出现的每种取值依次编上号，比如 `租房` 编成 0、`自己买房` 编成 1，模型就能用了。

需要注意的是：`LabelEncoder` 编出来的号码只是"第几种取值"，数字大小本身没有含义。对 `loan_grade` 这种本身就有顺序的列（等级越高风险越大）问题不大，但对 `person_home_ownership` 这类纯类别列，编号会给线性模型人为制造"租房 > 买房"的错觉。更讲究的做法是用独热编码，不过本方案为了流程简单沿用了编号，我们可以把它当作后续可优化的点。

### 缺失值填充

缺失值我们逐列用该列的中位数来填，中位数比均值更抗极端值，是处理缺失的常见选择。

以下是完整清洗代码：

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
    print("需要编码的字符串列:", encode)
    return encode

def label_transform(data, encode):
    """把字符串列按出现顺序转成整数编号"""
    for x in encode:
        data[x] = le.fit_transform(data[x])
        print(f"{x} 的类别映射:", le.classes_)

def clean(data):
    """用每列的中位数填充缺失值"""
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)
    return data

# 先补缺失，再把文本列编码成数字
raw_data = clean(raw_data)
encode_cols = label_encoder(raw_data)
label_transform(raw_data, encode_cols)

# 确认清洗结果
print("\n清洗后数据类型：")
print(raw_data.dtypes)
print("\n剩余缺失值总数:", raw_data.isnull().sum().sum())
```

清洗之后再用 `info()` 和 `isnull().sum()` 各看一眼，确认所有列都已经变成数值类型、缺失值数量为 0，就可以放心建模了。

---

## 6.4 特征工程

### 特征选择

从 EDA 的相关系数热力图中，我们已经选出了 6 个与目标关系最密切的特征：

- `loan_grade`：贷款等级
- `loan_percent_income`：还款占收入比
- `loan_int_rate`：贷款利率
- `person_home_ownership`：住房情况
- `cb_person_default_on_file`：是否有违约记录
- `loan_amnt`：贷款金额

只保留这 6 个特征，可以去掉噪声，也能让模型更专注于真正有信息量的输入。

### 标准化

接下来把特征和目标拆开，并对特征做**标准化**：把每个特征的均值变成 0、标准差变成 1，公式是 `z = (x - 均值) / 标准差`。

为什么必须这么做？我们准备用梯度下降来训练模型，如果某个特征数值特别大（比如贷款金额动辄上万，而 0/1 标志最多是 1），梯度会在不同方向上尺度悬殊，收敛很慢甚至"爆炸"；统一尺度能加快收敛，也让正则化对每个特征更公平。

```python
target = ['loan_status']
# 取出选定的 6 个特征，转成浮点数
X_raw = raw_data[features].values.astype(float)
y = raw_data[target].values

# 标准化：z = (x - mean) / std
mean = np.mean(X_raw, axis=0)
std = np.std(X_raw, axis=0)
X_scaled = (X_raw - mean) / std
```

---

## 6.5 模型训练与调优

### 划分训练集和测试集

我们手动实现一次按 80% / 20% 的比例切分：先把所有样本的下标随机打乱，前 80% 当训练集、后 20% 当测试集。这里固定随机种子 `seed(42)`，保证每次运行的切分结果一致，方便复现实验。

```python
np.random.seed(42)                                # 固定随机种子，实验可复现
indices = np.random.permutation(len(X_scaled))    # 随机打乱所有样本的下标
split_idx = int(0.8 * len(X_scaled))              # 前 80% 当训练集

train_idx, test_idx = indices[:split_idx], indices[split_idx:]
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("训练集形状:", X_train.shape)
print("测试集形状:", X_test.shape)
```

### 从零实现逻辑回归

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

### 训练模型

```python
model = LogisticRegressionFromScratch(lr=0.08, n_iters=3000, l2_ratio=0.01)
model.fit(X_train, y_train)
```

训练完成后，我们可以打印每个特征学到的权重，看模型认为谁更重要：

```python
print("=====  Trained Model Weights ======")
for name, w in zip(features, model.weights):
    print(f'{name:<28} | {w:>10.4f}')
print(f'{"Intercept / Bias (b)":<28} | {model.bias:>10.4f}')
```

权重的绝对值越大，说明该特征对结果的判别力越强。

---

## 6.6 模型测试与提交

### 在测试集上评估

训练完成后，我们在留下没见过的 20% 数据上测试：

```python
# 在测试集上预测，准确率 = 猜对的比例
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
```

### 生成提交文件

最后处理测试集并生成提交文件。测试集要和训练集走**一模一样的清洗流程**：去掉 `id`、只保留选中的特征、把文本列编码成数字、再标准化，然后用训练好的模型预测并组装成比赛要求的格式。

```python
# 读入测试集，去掉 id、只留选中的特征
test_data = pd.read_csv('/kaggle/input/3220-loan-approval/test.csv').drop('id', axis=1)
test = test_data[features].copy()

# 沿用训练时的标签编码
encode_cols = label_encoder(test)
label_transform(test, encode_cols)
test = test.values.astype(float)

# 标准化（理想情况应该用训练集的均值/标准差来转换测试集）
mean_test = np.mean(test, axis=0)
std_test = np.std(test, axis=0)
test_scaled = (test - mean_test) / std_test

# 预测并组装提交文件：id + loan_status
y_pred = model.predict(test_scaled)
col0 = pd.read_csv('/kaggle/input/3220-loan-approval/sample_submission.csv').drop('loan_status', axis=1)
col1 = pd.Series(y_pred, name='loan_status')
submission = pd.concat([col0, col1], axis=1)

# 上线前核对
assert len(submission) == len(test_data), "提交行数与测试集不一致"
assert list(submission.columns) == ['id', 'loan_status'], "列名不一致"
assert submission['loan_status'].isnull().sum() == 0, "预测结果包含缺失值"

submission.to_csv('submission.csv', index=False)
print("提交文件已生成：submission.csv")
print(submission.head())
```

**核对说明**：
- 行数必须与测试集一致，不能多也不能少。
- 列名必须是 `id` 和 `loan_status`，与 `sample_submission.csv` 严格匹配。
- `id` 顺序与测试集一一对应，不能打乱。
- 预测结果不能有缺失值，且必须是 0 或 1。

把 `submission.csv` 上传到 Kaggle 比赛页面，你就能在排行榜上看到自己的准确率了。

---

## 6.7 小结

本章我们以 3220 贷款审批挑战为例，完整解析了一场表格二分类比赛：

1. **比赛背景**：预测贷款是否会被批准，`loan_status` 为目标（1 批准，0 拒绝），评价指标为分类准确率。
2. **EDA**：查看了类别分布和特征与目标的相关系数，从相关系数热力图中挑出 6 个最重要的特征（`loan_grade`、`loan_percent_income`、`loan_int_rate`、`person_home_ownership`、`cb_person_default_on_file`、`loan_amnt`）。
3. **数据清洗**：用 `LabelEncoder` 把文本列变成数字，用中位数填补缺失值。注意 `LabelEncoder` 对纯类别列会人为制造顺序关系，后续可优化为独热编码。
4. **特征工程**：只保留与目标相关性高的 6 个特征；对特征做标准化（均值为 0、标准差为 1），加快梯度下降收敛，让正则化更公平。
5. **模型训练**：手写了一个带小批量梯度下降和 L1/L2 正则化的逻辑回归，选定学习率 0.08、迭代 3000 轮、L2 强度 0.01。训练后可打印每个特征的权重，解释模型认为谁更重要。
6. **模型测试与提交**：在保留的 20% 数据上打印准确率；对测试集按 `id + loan_status` 格式生成提交文件，上传前核对行数、列名、ID 顺序、缺失值。

这场比赛数据量不大、模型也比较基础，但完整展示了"二分类 + 逻辑回归"从数据到提交的常规套路。我们最大的体会是：**清洗和特征选择常常比模型本身更决定分数下限**——文本列编码得对不对、缺失值补得合不合理、特征选得好不好，都会直接体现在最终准确率上。

下一章我们将详解另一个表格比赛：推土机拍卖价格预测（Blue Book for Bulldozers），看看在表格回归场景下，上述清洗与建模流程有什么共通之处和不同之处。
