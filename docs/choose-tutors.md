# 选择家教（Choose Tutors）

## 一、比赛简介

### 1.1 比赛背景

这个比赛模拟的是家教平台上的一个真实业务问题：**一个学生面对一位家教，最后会不会选他？**

平台手里有一份家教档案，记录了这位家教的年龄、从业年限、课时费、资格等级，以及他能教哪些科目（物理、化学、生物、英语、地理、历史），还有他所带学生的平均考试分数。比赛要求我们根据这些信息，预测学生最终是否会选择这位家教。

这个任务看起来简单，但它同时包含了几类现实中的难点：

1. **标签不均衡**：被选中的家教只占一部分，绝大多数是「没被选」；
2. **特征之间有隐含关系**：课时费不是随便定的，它和家教的资格、学生水平有关，直接看价格会误判；
3. **样本量有限**：特征只有十来个，稍不注意就会过拟合，必须靠正则化和验证集把控。

### 1.2 比赛目标

训练集 `train.csv` 的列有：

- `Id`：样本编号；
- `age`：家教年龄；
- `years_of_experience`：从业年限；
- `lesson_price`：课时费；
- `qualification`：资格等级；
- `physics`、`chemistry`、`biology`、`english`、`geography`、`history`：六门科目，1 表示能教、0 表示不能教；
- `mean_exam_points`：学生平均考试分数；
- `choose`：目标列，1 表示被选中、0 表示没被选中。

测试集 `test.csv` 有同样的特征列，但没有 `choose`。我们要为每个 `Id` 输出一个 0～1 之间的数值，表示「被选中」的概率。

### 1.3 评价标准

比赛使用 **ROC-AUC（受试者工作特征曲线下面积）** 作为评价指标。

ROC 曲线把「预测为选中的概率」从高到低排一遍，每取一个阈值就得到一个「真正例率」和「假正例率」，画成曲线。AUC 就是这条曲线下的面积：**它衡量的是「随机抽一个被选中的样本和一个没被选中的样本，模型把前者排得更高的概率」**。所以 AUC 不关心你阈值定在哪，只看排序能力。取值 0～1，0.5 相当于乱猜，越高越好。

正因为 AUC 只看排序，提交文件里要交的是**概率**而不是 0/1 标签——这也是材料里所有方案的共同做法。

---

## 二、解题过程

我们的整体路线是：先做 EDA 找出哪些因素和「被选中」有关，再基于业务常识造几个新特征（比如「课时费相对水平是否划算」），最后用一个带 L2 正则的逻辑回归输出概率。逻辑回归的好处是简单、可解释、不容易过拟合，很适合这种中小规模表格数据。

### 2.1 数据探索（EDA）

我们首先把数据读进来，确认样本量、缺失值和标签分布：

```python
import pandas as pd
import numpy as np

# 读取训练集和测试集
df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 样本量、缺失值、标签分布一次看清楚
print("训练集行数:", len(df))
print("缺失值总数:", df.isnull().sum().sum())
print(df["choose"].value_counts(normalize=True))
```

结果显示数据**没有任何缺失值**，省去了插补的麻烦。目标列 `choose` 是二分类，而且两类比例并不对半，属于典型的**不均衡数据**——这也是为什么后面要用 AUC 而不是准确率：在不均衡时，准确率会被多数类「绑架」，AUC 更能反映真实排序能力。

接着我们看各门科目「能教」与「不能教」时，被选中的比例差多少：

```python
subjects = ["physics", "chemistry", "biology", "english", "geography", "history"]

for sub in subjects:
    # 分成两组，分别算 choose 的均值，也就是「被选中的比例」
    rate_no = df[df[sub] == 0]["choose"].mean() * 100
    rate_yes = df[df[sub] == 1]["choose"].mean() * 100
    print(f"{sub}: 不能教 {rate_no:.2f}% -> 能教 {rate_yes:.2f}%")
```

我们发现科目能力确实影响选择率，尤其理科方向更明显。材料里进一步统计了「能教几门理科（物理+化学+生物）」与选择率的关系，结论是**三门理科都能教的家教，选择率大约达到二分之一**，明显高于只能教一两门的。这说明「能力覆盖面」本身就是一个强特征。

然后我们关注课时费。课时费的绝对值很难直接比较，因为资格高、学生分数高的家教本来就该贵。材料里的做法是先用最小二乘拟合出一个「市场公允价」，再看实际价格偏离公允价多少：

```python
# 用 mean_exam_points 和 qualification 解释 lesson_price，拟合一个「公允价」公式
Y = df["lesson_price"].values
X = np.column_stack([
    df["mean_exam_points"].values,
    df["qualification"].values,
    np.ones(len(df)),                 # 常数项
])

# 最小二乘法：直接解正规方程，得到三个系数
W, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
print("公允价 = %.2f * 平均分 + %.2f * 资格 + %.2f" % (W[0], W[1], W[2]))

# 实际价格 - 公允价：负数表示「比该有的便宜」，也就是划算
df["pred_price"] = X.dot(W)
df["price_diff"] = df["lesson_price"] - df["pred_price"]
```

把 `price_diff` 分箱后统计选择率，可以看到「越划算（负得越多）选择率越高」的趋势。这给了我们一个很重要的建模线索：**不要直接喂原始价格，而要喂「相对公允价的偏离」**。

### 2.2 数据清洗

这份数据的清洗工作主要是三件事：

1. **没有缺失值**，所以不需要插补；
2. **标准化**。年龄、课时费、平均分的量纲差很远，逻辑回归对尺度敏感，必须把每个特征变成「减均值、除标准差」；
3. **训练集和测试集用同一套均值、标准差**。这是一个容易被忽略的细节：测试集的标准化参数必须来自训练集，否则相当于提前偷看了测试数据的分布。

我们写一个小函数统一处理：

```python
def standardize_by_train(train, test, cols):
    """用训练集的均值/标准差标准化，测试集套用同一套参数"""
    mean = train[cols].mean()
    std = train[cols].std()
    std[std == 0] = 1e-9            # 防止常数列除零
    train_s = (train[cols] - mean) / std
    test_s = (test[cols] - mean) / std
    return train_s, test_s
```

另外，材料里也试过用**过采样（oversampling）**把少数类补到和多数类一样多，这在逻辑回归上能缓解不均衡；但在 AUC 这个指标下，过采样带来的提升有限，反而可能让概率偏高。我们的主线仍然用原始分布训练，把重点放在特征工程上。

### 2.3 模型训练与超参数调优

我们首先在原始特征之外造了几组「有业务含义」的特征：

```python
def add_features(d, price_w):
    d = d.copy()
    # 能教的理科门数：物理+化学+生物
    d["science_count"] = d["physics"] + d["chemistry"] + d["biology"]
    # 每一分对应的课时费：衡量「性价比」
    d["price_per_point"] = d["lesson_price"] / d["mean_exam_points"]
    # 每一级资格对应的课时费
    d["price_per_qual"] = d["lesson_price"] / d["qualification"]
    # 相对公允价的偏离（负数=划算）
    pred = d["mean_exam_points"] * price_w[0] + d["qualification"] * price_w[1] + price_w[2]
    d["price_diff"] = d["lesson_price"] - pred
    # 物理和化学是否会同时教，交叉项
    d["physics_chemistry"] = d["physics"] * d["chemistry"]
    return d
```

这里的每一个新特征都对应一句人话：`science_count` 是「能力广度」，`price_per_point` 是「性价比」，`price_diff` 是「贵不贵」，`physics_chemistry` 是「理科组合」。相比让模型自己从原始列里瞎猜，我们直接把业务判断告诉它，效果更稳。

然后我们实现一个**带 L2 正则的逻辑回归**。之所以自己写，是因为可以完全控制损失函数和正则项，也方便观察训练过程：

```python
class LogisticRegressionL2:
    def __init__(self, lr=0.1, l2_reg=1.0, epochs=4000, tol=1e-6):
        self.lr, self.l2_reg = lr, l2_reg
        self.epochs, self.tol = epochs, tol

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)          # 权重
        self.b = 0.0                  # 偏置
        prev_loss = float("inf")
        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-z))           # sigmoid 得到概率
            p = np.clip(p, 1e-15, 1 - 1e-15)        # 防止 log(0)
            # 交叉熵损失 + L2 正则
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            loss += (self.l2_reg / (2 * m)) * np.sum(self.w ** 2)
            if abs(prev_loss - loss) < self.tol:    # 收敛就提前停
                break
            prev_loss = loss
            # 梯度下降：数据梯度 + 正则梯度
            dw = (1.0 / m) * (X.T @ (p - y)) + (self.l2_reg / m) * self.w
            db = (1.0 / m) * np.sum(p - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))
```

写好模型后，我们在训练集内部切出 20% 做**分层验证**（`stratify=y`，保证验证集的正负比例和整体一致），然后用网格搜索挑学习率和正则强度：

```python
from sklearn.model_selection import train_test_split

# 训练集/验证集按 8:2 切分，stratify 保持正负样本比例
X_tr, X_val, y_tr, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

best_auc, best = 0.0, {}
for lr in [0.05, 0.1, 0.2]:          # 学习率
    for reg in [0.1, 1.0, 5.0, 10.0]:  # L2 正则强度
        model = LogisticRegressionL2(lr=lr, l2_reg=reg, epochs=4000)
        model.fit(X_tr, y_tr)
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val))
        if val_auc > best_auc:
            best_auc, best = val_auc, {"lr": lr, "l2_reg": reg}
```

我们这样理解两个超参数：**学习率**决定每步走多远，太大会来回震荡、太小则收敛慢；**L2 正则**决定「惩罚大权重」的力度，越大模型越保守、越不容易过拟合。在这个数据上，正则强度取中间档比较合适，太小会过拟合、太大又会欠拟合。

### 2.4 模型测试与指标报告

选好参数后，我们用**全部训练数据**重新训练一次（多喂数据通常更稳），再对测试集输出概率：

```python
from sklearn.metrics import roc_auc_score

# 用最优超参数在全部标准化后的训练数据上重训
final_model = LogisticRegressionL2(lr=best["lr"], l2_reg=best["l2_reg"], epochs=5000)
final_model.fit(X_scaled, y)

# 验证集上的 AUC（重训前记录的 best_auc 作为参考）
print("验证集最佳 AUC:", round(best_auc, 4))

# 测试集输出「被选中」的概率
test_preds = final_model.predict_proba(X_test_scaled)
```

评价时只看两个数：**验证集 AUC** 和**训练集 AUC**。如果训练 AUC 很高、验证 AUC 明显低，说明过拟合，要加大正则；如果两者都低，说明特征不够或者模型太简单。我们的逻辑回归在标准化 + 正则之后，训练和验证的差距被控制在一个可接受的范围。

最后按比赛格式生成提交文件：

```python
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "choose": test_preds,      # 注意是概率，不是 0/1
})
submission.to_csv("submission.csv", index=False)

# 检查行数和测试集一致、概率都在 [0,1]
assert len(submission) == len(test_df)
assert submission["choose"].between(0, 1).all()
```

这一步必须确认提交的是**概率**。如果手滑交了 0/1，AUC 会退化成「只有一个阈值的准确率」，分数会掉得很厉害。

---

## 三、总结

本文中我们完成了「选择家教」这道二分类题的完整流程。我们首先用 EDA 发现数据无缺失、标签不均衡，并确认了科目能力、理科覆盖面、课时费相对公允价的偏离都与「被选中」有关。其次我们做了最小化的清洗：统一用训练集的均值和标准差做标准化，没有做多余的文本或异常值处理。接着我们围绕业务含义构造了理科门数、性价比、价格偏离、科目交叉等特征，用一个带 L2 正则的自写逻辑回归输出概率，并在分层验证集上网格搜索学习率和正则强度。最后我们用全量数据重训，按比赛要求输出概率，并对行数和取值范围做了检查，以 ROC-AUC 作为排序能力的度量。

这个比赛让我们体会到：**表格数据的关键往往不在模型多复杂，而在特征是否说出了业务逻辑**。「课时费贵不贵」远不如「课时费相对公允价贵不贵」有用，而后者需要我们先把公允价拟出来。材料里还有用随机森林、甚至从零实现决策树的方案，它们在给出概率和可解释性上各有取舍，但核心特征思路是一致的。
