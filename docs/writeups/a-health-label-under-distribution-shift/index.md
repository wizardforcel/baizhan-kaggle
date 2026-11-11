# A Health Label Under Distribution Shift

## 一、比赛简介

### 1.1 比赛背景

这个比赛的数据来自 Kaggle Playground 系列的第 S5E12 期（Playground S5E12）。Playground 是 Kaggle 定期举办的"人人都能参加"的练习赛：数据是**合成**的，不需要医学或行业背景，目的就是让大家练手。这一期的上游数据是 Mohan Krishna Thalla 整理的 **Diabetes Health Indicators Dataset**（糖尿病健康指标数据集），一份由问卷和体检指标构成的表格，每条记录对应一个人，标签是"是否被诊断为糖尿病"。

本次我们要读的这份材料，是作者 Clare Laurent 写的一份**独立研究型 notebook**，她没有去刷排行榜，而是拿同一份数据回答一个更有意思的问题：

> **总体的 AUC 很稳定，是不是就意味着模型对不同人群都同样可靠？**

答案是不一定。AUC 衡量的是"随机抽一个病人和一个健康人，模型给病人的分更高的概率"，它是一个**排序**指标，只看谁排在前面，不看分数本身准不准。如果模型在某个亚群体上系统性地把分数压低了（这叫**校准偏差**），排序可能还是对的、AUC 还是很高，但一旦你拿某个固定阈值去筛人，这个群体就会被打包漏掉。这就是所谓**分布漂移**（distribution shift）下的隐患：训练时的人群构成，和将来实际使用的人群构成不一样，模型的表现会悄悄地变。

作者在 notebook 开头就写明了立场：这是一份**学习练习**，不能用来诊断糖尿病，也不能用来验证任何筛查设备。这句话贯穿全文，也是我们要认真对待的部分。

### 1.2 比赛目标

数据目录下是三个标准文件：

- `train.csv`：训练集。第一列是样本编号，中间是各种健康指标（体重指数 `bmi`、腰臀比 `waist_to_hip_ratio`、血压、胆固醇、年龄、性别、吸烟史等），最后一列是标签 `diagnosed_diabetes`（0 或 1）。
- `test.csv`：测试集。列结构和训练集去掉标签后一致。
- `sample_submission.csv`：提交样例。第一列是编号，第二列是 `diagnosed_diabetes`，要求填**概率**而不是 0/1。

所以任务是：给每个人算一个"患糖尿病"的概率，输出到提交文件里。

### 1.3 评价标准

这是一个二分类问题，评价指标是 **ROC AUC**（Receiver Operating Characteristic 曲线下的面积）。大白话解释：把所有样本按模型给的分数从高到低排队，AUC 就是"一个真正的病人排在一个健康人前面"的概率。完全瞎猜是 0.5，完美是 1.0，越高越好。

AUC 有两个特点，恰好是这份材料想讨论的：

1. **它只看排序，不看数值。** 把所有人的分数同时乘以 2，AUC 一点不变。
2. **它是把所有人混在一起算的。** 如果模型对男性很准、对女性很差，只要总体排序还过得去，AUC 依然可能很好看。

notebook 在报告里同时算了准确率、平衡准确率、对数损失、AUC 和 MCC（马修斯相关系数），但**比赛只看 AUC**。我们在下面也会把 AUC 作为主指标，其他几个当作诊断。

---

## 二、解题过程

整份材料的思路可以概括成一句话：**先做一份可审计的、固定参数的基线，再用一次"预先声明"的消融实验去检验结论，最后把局限写清楚。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步不是看数据，而是先确认"我拿到的文件对不对"。notebook 把实验的全部设置写进一个 `CONFIG` 字典，然后用提交样例的列名反过来校验数据：

```python
from pathlib import Path
import numpy as np
import pandas as pd

# 把整个实验的"合同"写在最前面：目标列、任务类型、要消融的字段
CONFIG = {
    "episode": "s5e12",
    "targets": ["diagnosed_diabetes"],
    "family": "binary",                       # 二分类
    "ablation": ["bmi", "waist_to_hip_ratio"],# 第二个实验要拿掉的两列
    "date_columns": [],
}

# 数据可能挂在 input 下的任意位置，按数据集名去找目录
candidates = [p.parent for p in Path("/kaggle/input").rglob("train.csv")
              if CONFIG["episode"] in str(p).lower()]
DATA = candidates[0]

# 用样例文件的列名当"合同"：第一列必须是 ID，后面必须正好是目标列
schema = pd.read_csv(DATA / "sample_submission.csv", nrows=3).columns.tolist()
id_col = schema[0]
assert schema[1:] == CONFIG["targets"], "提交格式和配置对不上"
```

为什么这么写？因为比赛中真正让人抓狂的不是模型不好，而是**提交格式错了**。先读 `sample_submission.csv`、把列名当成契约去断言，可以在写第一行建模代码之前就把"输出长什么样"钉死。后面所有产出都围着这个 `schema` 转。

第二步是抽样和统计缺失。训练文件可能有几百万行，全读进内存不一定撑得住。notebook 的做法很讲究：**分块读 + 蓄水池抽样**，一边读一边统计缺失，最后留下 12 万行。

```python
MAX_ROWS = 120000
reservoir = None
rows_seen = 0
missing = None
rng = np.random.default_rng(20261026)

# 每次读 10 万行，读完就丢；只保留"随机优先级"最小的 12 万行
for chunk in pd.read_csv(DATA / "train.csv", chunksize=100000):
    rows_seen += len(chunk)
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)

    # 给每一行发一个随机数，数字越小越优先留下
    chunk["_sampling_priority"] = rng.random(len(chunk))
    reservoir = chunk if reservoir is None else pd.concat([reservoir, chunk], ignore_index=True)
    reservoir = reservoir.nsmallest(MAX_ROWS, "_sampling_priority")

train = reservoir.drop(columns="_sampling_priority").reset_index(drop=True)
assert train[id_col].is_unique, "ID 有重复"
assert not train[CONFIG["targets"]].isna().any().any(), "标签有缺失，不要偷偷填"

# 把缺失统计落盘，作为审计证据
pd.DataFrame({"field": missing.index,
              "missing_rows": missing.values.astype(int)}).to_csv("missingness.csv", index=False)
```

这段代码有三处值得学：

- **`nsmallest` 而不是 `sample`。** 如果每读一块就 `sample` 一次，最终结果会严重偏向"读得早"的那部分行。给每行一个固定随机数、永远保留最小的那 12 万行，才能保证整体均匀。这就是蓄水池抽样。
- **`rows_seen` 单独记。** 最后要报告"源文件有多少行、实际用了多少行"，读者才知道你没有偷偷用全量数据夸大结果。
- **标签缺失直接断言失败。** 把没有标签的行删掉或填个默认值，等于自己给自己造样本，这在任何比赛里都是大忌。

第三步是看标签分布和字段类型：

```python
# 每个字段有多少缺失，打印出来看哪些列"很可能没法用"
print(pd.read_csv("missingness.csv").sort_values("missing_rows", ascending=False).head(10))

# 标签有多少个值、各占多少
print(train["diagnosed_diabetes"].value_counts(normalize=True))
```

材料没有给出这份数据的具体行数和患病比例，我们也不编。这一步的结论方向是明确的：**这是一个类别不均衡的二分类问题**，正类（患病）通常占少数，这也是为什么后面要报告"平衡准确率""MCC"这些对不均衡不那么敏感的诊断指标，而不是只看准确率。

### 2.2 数据清洗

这份 notebook 的清洗哲学非常明确，写在注释里：**只做"无状态"的变换**（stateless transforms）。

什么叫无状态？就是"这一列的每个值怎么处理，只取决于这个值自己，不取决于别的行"。比如"把性别转成字符串""把无穷大换成 NaN""把日期拆成年月日"，都是无状态的。反过来，"用全表中位数填空""用目标均值编码类别"，都是**有状态**的——它们要从数据里学一个统计量，一旦用了验证集或测试集的数据，就是泄漏。

```python
def features(frame, omit=()):
    """把原始表变成模型能吃的表。omit 里的列直接不取，用于消融实验。"""
    x = frame[[c for c in feature_columns if c not in omit]].copy()

    for c in list(x.columns):
        if c in CONFIG.get("date_columns", []):
            # 日期列拆成三个数字，树模型才用得上
            dt = pd.to_datetime(x.pop(c), errors="coerce")
            x[c + "_year"] = dt.dt.year
            x[c + "_month"] = dt.dt.month
            x[c + "_weekday"] = dt.dt.dayofweek
        elif not pd.api.types.is_numeric_dtype(x[c]):
            # 文本列：缺失单独变成一个类别 "__MISSING__"，而不是填众数
            x[c] = x[c].astype("string").fillna("__MISSING__").astype(str)
        else:
            # 数值列：转不动就变 NaN，无穷大也当成缺失，交给 CatBoost 处理
            x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return x
```

两个设计选择很讲究。**缺失值变成一个新类别**，而不是填众数：在健康数据里，"没有记录"本身可能就是信息（比如没量过腰围的人，和量过的人不一样），强行填一个"平均腰围"会把这种信息抹掉。**无穷大也当缺失**：某些比值型字段（比如 `waist_to_hip_ratio`）在分母为 0 时会算出 `inf`，不处理就会让整个模型崩掉。

接着是标签编码和划分。二分类要把标签映射成 0/1，并且**按标签分层**抽样，保证训练集和验证集的正负比例一致：

```python
classes = sorted(train["diagnosed_diabetes"].unique().tolist())
mapping = {v: i for i, v in enumerate(classes)}
y = train["diagnosed_diabetes"].map(mapping).to_numpy()
print("类别编码:", mapping)

indices = np.arange(len(train))
# stratify=y：按标签分层，避免验证集里病人比例忽高忽低
fit_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=20261026, stratify=y
)
assert set(fit_idx).isdisjoint(val_idx), "训练和验证有重叠"
```

固定随机种子、固定 20% 留出、只切一次——这三点看起来朴素，其实是这份材料最重要的方法论：**先定好实验，再看分数**。如果反复换划分方式直到验证集好看，那验证集就变成了第二个训练集。

### 2.3 模型训练与超参数调优

模型用的是 **CatBoost**，一种专门擅长处理类别特征的梯度提升树。参数是**预先固定**的，没有在验证集上搜索：

```python
from catboost import CatBoostClassifier

def model_fit(x, target):
    # 找出文本列，交给 CatBoost 用它的方式处理（不用我们手动 one-hot）
    cat_cols = x.select_dtypes(include=["object", "string"]).columns.tolist()

    params = dict(
        iterations=350,          # 树的棵数，固定
        depth=6,                 # 每棵树深度，固定
        learning_rate=0.08,      # 学习率，固定
        random_seed=20261026,
        verbose=False,
        thread_count=4,
        allow_writing_files=False,   # 不在工作目录里拉一堆临时文件
    )

    model = CatBoostClassifier(loss_function="Logloss", **params)
    model.fit(x, target, cat_features=cat_cols)
    return model
```

三个超参数的选择理由：`iterations=350` 配合 `learning_rate=0.08` 是一组偏保守的组合，训练快、不容易过拟合；`depth=6` 是表格数据的经验值，再深在小样本上容易记住噪声；**不对验证集调参**是这份材料的核心纪律——它想证明的是"一个不做任何优化的固定模型能到什么水平"，而不是"我能刷到多高"。

**基线。** 光有模型还不够，必须有一个"什么都不学"的对照。notebook 用训练集的**正类比例**作为常数预测：

```python
def constant_prediction(target, n):
    # 不看任何特征，直接输出训练集里的类别频率
    freq = np.bincount(target, minlength=len(classes)) / len(target)
    return np.tile(freq, (n, 1))
```

如果 CatBoost 打不过这个常数预测，说明特征里根本没有可用信息，后面的所有讨论都失去意义。这个对照在真正的比赛里经常被省略，但它是判断"模型到底有没有学到东西"的最便宜办法。

**评估函数。** 因为这份 notebook 是一份"通用骨架"，评估函数写得比较全：

```python
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, log_loss, matthews_corrcoef)

def score(truth, p):
    labels = p.argmax(axis=1)          # 概率最大的那一类
    result = {
        "accuracy": float(accuracy_score(truth, labels)),
        # 平衡准确率：先分性别/类别各自算准确率再平均，对不均衡更公平
        "balanced_accuracy": float(balanced_accuracy_score(truth, labels)),
        "log_loss": float(log_loss(truth, p, labels=np.arange(len(classes)))),
        # 比赛主指标：只看排序，不看概率绝对值
        "AUC": float(roc_auc_score(truth, p[:, 1])),
        "MCC": float(matthews_corrcoef(truth, labels)),
    }
    return result
```

这里就能看出为什么材料开头说"稳定的 AUC 可能掩盖校准差异"：`log_loss` 会惩罚"分数虚高"（说 0.99 结果不是），AUC 不会；`balanced_accuracy` 会惩罚"某一类整体判错"，AUC 也不一定。**几个指标摆在一起看，比只看一个更能发现模型在哪个方向上出了问题。**

### 2.4 模型测试与指标报告

**第一个实验：基线对照。** 在同一个留出集上同时算常数预测和 CatBoost 的分数：

```python
X = features(train)
baseline = constant_prediction(y[fit_idx], len(val_idx))
model = model_fit(X.iloc[fit_idx], y[fit_idx])
pred = model.predict_proba(X.iloc[val_idx])

results = {
    "constant": score(y[val_idx], baseline),   # 什么都不学
    "boosting": score(y[val_idx], pred),       # 固定参数的 CatBoost
}
print(results)
```

常数预测的 AUC 必然正好是 0.5（所有分数一样，排序完全随机），所以这个对照的价值在于给其他指标一个"地板"：准确率等于正类比例、MCC 等于 0。CatBoost 只要明显高过这条线，就说明特征是有用的。

**第二个实验：预先声明的消融。** 这是整份材料最有教学价值的部分。在**看分数之前**就写死：拿掉 `bmi` 和 `waist_to_hip_ratio` 这两列，其余完全不变，再看分数掉多少。

```python
# 消融名单在 CONFIG 里写死，不是看到结果之后才挑的
omit = [c for c in CONFIG["ablation"] if c in feature_columns]
reduced = features(train, omit)                       # 同样的预处理，只是少两列
alternate = model_fit(reduced.iloc[fit_idx], y[fit_idx])
alt_pred = alternate.predict_proba(reduced.iloc[val_idx])

results["availability_ablation"] = score(y[val_idx], alt_pred)
```

为什么强调"预先声明"？因为如果先跑一遍、发现掉得不多、再把这个结果说成"我们本来就想验证的事"，那就是典型的**事后编故事**。预先声明的好处是：结论是可证伪的——掉多少就是多少，不能反悔。

而且要注意作者对消融结果的解读态度：**消融是诊断，不是部署许可**。就算拿掉 `bmi` 分数几乎不变，也不代表"实际场景里没有 bmi 也能用"，因为这两列的相关信息可能已经被别的列（比如体重、身高等）间接带进来了。

**分布压力测试。** 材料要求"报告一个预先声明的分布压力测试"，并给出了一份模型卡（model card），里面明确列出**禁止用途**（excluded uses）：不能用于诊断糖尿病、不能用于验证筛查设备。notebook 里给出的压力测试思路是**对抗验证**：训练一个分类器去区分"这行来自训练集还是测试集"，如果它很容易区分（AUC 明显高于 0.5），说明两个集合的特征分布已经不一样了，模型到测试集上很可能失灵。

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier

# 造一个"来源"标签：0 = 训练集，1 = 测试集
domain_X = pd.concat([X, features(test)], ignore_index=True)
domain_y = np.r_[np.zeros(len(X)), np.ones(len(test))]

# 如果这个分类器能轻松分开两边，说明存在分布漂移
domain_auc = cross_val_score(
    HistGradientBoostingClassifier(max_iter=100, random_state=20261026),
    domain_X, domain_y, cv=3, scoring="roc_auc"
).mean()
print("domain AUC:", domain_auc)
```

判别要点：**domain AUC 接近 0.5 是好消息**（两边几乎无法区分），但它**不能证明没有问题**——它只检查了特征边缘分布，查不出"特征相同但标签关系变了"这种条件漂移。所以作者把它当成"多个检查之一"，而不是"通过/不通过"的开关。

**提交文件的生成。** 最后用全量抽样数据重新拟合，再分块预测测试集，每读一块就和样例文件逐行比对：

```python
final = model_fit(X, y)                        # 用全部抽样数据重训一次
sample = pd.read_csv(DATA / "sample_submission.csv")
submission = sample.copy()
submission[CONFIG["targets"]] = submission[CONFIG["targets"]].astype(float)

offset = 0
for chunk in pd.read_csv(DATA / "test.csv", chunksize=100000):
    n = len(chunk)
    # 逐行核对 ID 顺序，防止分块读取和样例对不上
    assert np.array_equal(chunk[id_col].to_numpy(), sample.iloc[offset:offset + n, 0].to_numpy())

    p = final.predict_proba(features(chunk))
    submission.loc[submission.index[offset:offset + n], "diagnosed_diabetes"] = p[:, 1]
    offset += n

assert offset == len(sample)
submission.to_csv("submission.csv", index=False)

# 读回来再验一遍：列名、行数、ID 顺序、没有 NaN
roundtrip = pd.read_csv("submission.csv")
assert roundtrip.columns.tolist() == schema and len(roundtrip) == len(sample)
```

分块读取是为了内存，逐块断言 ID 顺序是为了正确性。最后"读回来再验一遍"这一步很值得学：**写出去的文件和内存里的对象是两回事**，CSV 可能有精度、编码、换行的问题，读回来验证才是真的验证。

**关于榜单上的第一名。** 材料末尾附了冠军 wind1234 的做法：用**爬山法**（hill climbing）从多个模型里挑互补的组合，再对排序变换后的预测做 **Ridge 回归**，最好的是一个 top-36 的 Ridge 集成，`alpha=10`，用"截止日期之后"的数据做验证，牺牲了一点交叉验证分数换取更强的泛化。作者明确声明：**这是原作者的比赛发现，不是通用配方，我们的 notebook 不声称复现了这个分数**。这句话本身就是一种很好的科研态度。

---

## 三、总结

本文中我们完成了一份"A Health Label Under Distribution Shift"的独立研究型基线。我们首先把实验的"合同"——目标列、任务类型、消融字段——写进配置，用提交样例的列名反向校验数据格式。其次我们用分块读取加蓄水池抽样，在不超过 12 万行的前提下做了一次种子固定的均匀抽样，同时把每个字段的缺失行数落盘，并拒绝任何形式的标签填补。接着我们对特征只做无状态变换：文本列的缺失变成独立类别，数值列的无穷大转成缺失，再由 CatBoost 自己处理类别特征；标签按分层切出固定的 20% 留出集。然后我们用一个参数完全固定的 CatBoost（350 轮、深度 6、学习率 0.08）和"只输出类别频率"的常数基线做对照，并做了一次**事先声明**的消融实验——拿掉 `bmi` 和 `waist_to_hip_ratio`，在同一批留出行上比较。最后我们分块预测测试集、逐块核对 ID 顺序、读回提交文件做格式复验，并给出了分布压力测试（对抗验证）和一份写清"禁止用途"的模型卡。

这份材料教给我们的，与其说是技术，不如说是**分寸**：AUC 高不等于模型可以拿去筛人，消融掉分不多不等于那些字段可以不要，对抗验证通过不等于没有分布漂移。材料最后反复强调的那句话值得我们抄下来——**这次练习不诊断糖尿病，也不验证任何筛查设备。**
