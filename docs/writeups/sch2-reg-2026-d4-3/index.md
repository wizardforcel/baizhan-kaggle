# 肝硬化结局多分类预测（Multi-Class Prediction of Cirrhosis Outcomes）

## 一、比赛简介

### 1.1 比赛背景

肝硬化是一种慢性肝病：健康的肝细胞被纤维组织取代，肝脏逐渐变硬、失去功能。医生在随访肝硬化患者时，会定期做血液和生化检查，记录一系列指标，并根据病情变化把患者的状态归入不同结局。如果能在早期就判断出患者接下来更可能处于哪个结局，医生就可以提前调整随访频率和治疗方案。

这个比赛（Multi-Class Prediction of Cirrhosis Outcomes）给出一批肝硬化患者的临床记录，要求我们根据这些记录预测患者的 `Status`。`Status` 是一个**三分类**标签，材料里的提交列名 `Status_C`、`Status_CL`、`Status_D` 分别对应三个类别：

- `C`：Censored，删失/存活，指在随访期内患者仍然存活、没有发生终点事件；
- `CL`：Censored due to liver transplant，因接受肝移植而删失；
- `D`：Death，死亡。

理解这三个类别的含义很重要，因为它决定了我们要预测的是什么：不是简单的「生病/没生病」，而是「患者最终走到哪一步」。三个类别之间有天然的临床关联（比如 `C` 和 `D` 是两种相反的结局，`CL` 则介于中间），模型需要学会区分这种细微差别。

这类医学表格数据有几个常见难点。第一，样本量通常不大，模型容易过拟合；第二，临床指标之间往往高度相关（比如多个肝功能指标同向变化），存在冗余；第三，类别可能不均衡，如果直接用准确率评估，模型可能偏向多数类。因此这场比赛选择 **Log Loss（对数损失）** 作为评价指标，而不是准确率。

### 1.2 比赛目标

比赛提供训练集、测试集和提交样例：

- 训练集包含 `id`、患者的临床特征列，以及目标列 `Status`；
- 测试集包含同样的特征和 `id`，但没有 `Status`；
- 提交样例有三列概率：`id`、`Status_C`、`Status_CL`、`Status_D`。

我们要对测试集里的每位患者，输出属于三个类别的**概率**，而不是单个硬标签。也就是说，提交文件里每一行有三个数，分别表示「这位患者结局是 C / CL / D 的可能性」，三个数加起来应当接近 1。

### 1.3 评价标准

比赛使用 **多分类对数损失（Multi-class Log Loss）**，越低越好。

用大白话解释：模型对每个样本给出三个概率，比如 `[0.7, 0.2, 0.1]`。我们取出真实类别对应的那个概率，取它的对数，再取负号，就得到这个样本的损失。如果模型给真实类别很高的概率（比如 0.9），损失就很小；如果给真实类别很低的概率（比如 0.01），损失就会非常大。最后对所有样本取平均。

公式写出来是：

```
LogLoss = -1/N * Σ log(p_true)
```

其中 `p_true` 是模型给真实类别的预测概率，`N` 是样本数。

这个指标有两个特点值得记住：

1. **它惩罚「自信地猜错」**。如果模型斩钉截铁地给某个错误类别 0.99 的概率，损失会爆炸式增长。所以 Log Loss 逼着模型输出「诚实的概率」，而不是盲目自信。
2. **它和准确率不是一回事**。一个模型可能准确率很高，但概率校准很差，Log Loss 依然很难看。因此我们不能只满足于「猜对最多」，还要让概率分布尽量贴近真实。

对我们入门者来说，Log Loss 还有一个直观的理解方式：它衡量的是模型的「惊讶程度」。模型越是被真实结果吓一跳，损失越大；越是从容地给出高概率，损失越小。

---

## 二、解题过程

我们的整体方案是：**先做基本的 EDA 确认数据规模和目标分布，再把类别特征统一做标签编码，然后用 XGBoost 的多分类目标函数配合 5 折分层交叉验证训练，按折平均概率，最后按三个类别输出提交文件。** 这条路线的特点是简单、可复现，而且非常适合作为多分类问题的第一版基准。

### 2.1 数据探索（EDA）

我们首先把三份 CSV 读进来，确认数据的形状和列名：

```python
import pandas as pd
import numpy as np

# 训练集、测试集、提交样例
train = pd.read_csv('/kaggle/input/sch2-reg-2026-d4-3/train.csv')
test  = pd.read_csv('/kaggle/input/sch2-reg-2026-d4-3/test.csv')
submission_sample = pd.read_csv('/kaggle/input/sch2-reg-2026-d4-3/sample_submission.csv')

# 先看形状，判断样本量级
print("Train shape:", train.shape)
print("Test shape :", test.shape)

# 看提交样例的列名，确认要输出哪几个概率列
print(submission_sample.columns.tolist())
```

打印出来的提交列名非常关键，它直接告诉我们类别有几个、分别叫什么。我们看到的是 `id`、`Status_C`、`Status_CL`、`Status_D`，也就是三个类别。这个信息决定了后面模型必须使用「三分类」目标，而不是二分类。

接着我们看目标变量的分布：

```python
# 每类有多少位患者
print(train['Status'].value_counts())

# 看每类占比，判断是否不均衡
print(train['Status'].value_counts(normalize=True))
```

从分布能看出三类样本并不是完全均等的。这一点很重要：如果某一类特别少，模型在训练时会倾向于忽略它；而 Log Loss 对少数类的错误预测又特别敏感。因此后面我们用了**分层交叉验证**（`StratifiedKFold`），保证每一折里三个类别的比例都和整体一致，不会出现某一折里某个类别几乎消失的情况。

然后我们检查特征类型和缺失情况：

```python
# 数据类型一览：哪些是数值、哪些是字符串
print(train.dtypes)

# 逐列统计缺失值
print(train.isnull().sum())
```

这一步让我们确认了两件事：第一，数据里存在 `object` 类型的类别特征（比如某些检查结果被记录成文字类别），需要编码；第二，数值特征大多已经是数字格式，不需要做复杂的解析。材料中的流程没有做缺失值插补，因为 XGBoost 可以自动处理缺失值——它会为缺失值学习一个默认的分裂方向，这也是我们选它的原因之一。

最后我们简单看一下 `id` 列：

```python
# id 只是编号，和病情无关
print(train['id'].head())
print(test['id'].head())
```

`id` 在训练集和测试集里各自连续，它只是一个行标识。如果把它当特征，模型可能会学到「id 越大越容易怎样」这种毫无意义的规律，所以我们建模前一定要把它删掉。

### 2.2 数据清洗

我们其次做数据清洗和编码。这一步的目标只有一个：把所有列都变成模型能吃的数字，同时保证训练集和测试集的编码规则完全一致。

**第一步：删除 `id`。** 训练集和测试集都要删，但测试集的 `id` 要先单独存下来，因为最后提交时还要用它：

```python
# 训练集直接删掉 id
train = train.drop('id', axis=1)

# 测试集的 id 先保存，最后写提交文件时要用
test_ids = test['id']
test = test.drop('id', axis=1)
```

这里有个小细节：如果先删了 `test['id']` 又没有保存，最后就不知道每一行预测对应哪位患者，提交文件会错位。所以顺序必须是「先存后删」。

**第二步：编码目标变量。** XGBoost 的多分类要求标签是 0、1、2 这样的整数，不能是 `C`、`CL`、`D` 这样的字符串。我们用 `LabelEncoder` 做映射：

```python
from sklearn.preprocessing import LabelEncoder

# 把 C/CL/D 映射成 0/1/2
le_target = LabelEncoder()
train['Status'] = le_target.fit_transform(train['Status'])

# 打印映射关系，方便后面把预测结果还原成类别名
print(dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))
```

注意这里只对训练集 `fit_transform`，因为测试集根本没有 `Status` 列。最后生成提交文件时，我们不需要把数字还原成字符串——提交列名本身就是 `Status_C`、`Status_CL`、`Status_D`，只要保证概率列的**顺序**和编码顺序一致即可。

**第三步：编码类别特征。** 我们自动找出所有 `object` 类型的列，逐列做标签编码。关键技巧是：**把训练集和测试集对应的列拼起来一起 `fit`**，这样测试集里出现、但训练集里没出现过的新类别也能被编码，不会报错：

```python
# 找出所有字符串类型的特征列
cat_cols = [col for col in train.columns if train[col].dtype == 'object']
print("Categorical columns:", cat_cols)

for col in cat_cols:
    le = LabelEncoder()
    # 合并训练集和测试集的取值，保证编码表覆盖所有类别
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)

    # 用同一张编码表分别转换，保证两边含义一致
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
```

这里我们特意用了 `.astype(str)`。原因是有些类别列里可能混有数字和文字（比如 `"1"` 和 `"unknown"`），统一转成字符串可以避免 `LabelEncoder` 因为类型不一致而报错。

**第四步：确认最终特征矩阵。** 编码完成后，训练集里除了 `Status` 全是数值，测试集的列也和训练集一一对应：

```python
# X 是特征，y 是标签
X = train.drop('Status', axis=1)
y = train['Status']

# 检查两边列名和顺序是否完全一致
assert list(X.columns) == list(test.columns)

print("X shape     :", X.shape)
print("test shape  :", test.shape)
```

这个 `assert` 虽然只有一行，却非常有价值。列顺序不一致是表格比赛里最隐蔽的错误之一：模型训练时第 3 列是「年龄」，预测时第 3 列却变成了「胆红素」，程序不会报错，但结果会莫名其妙地差。用断言提前拦住，能省下大量排查时间。

### 2.3 模型训练与超参数调优

我们接着训练模型。选择 XGBoost 的理由很直接：它是梯度提升树的一种，擅长处理中小规模的表格数据，对缺失值和特征尺度都不敏感，而且原生支持多分类。

**多分类的目标函数。** XGBoost 用 `multi:softprob` 输出每个类别的概率，配合 `num_class=3` 告诉它一共有三类：

```python
import xgboost as xgb

params = {
    'objective': 'multi:softprob',  # 输出每个类别的概率
    'num_class': 3,                 # 三个结局：C / CL / D
    'eval_metric': 'mlogloss',      # 用多分类对数损失监控训练
    'learning_rate': 0.05,          # 每棵树只走一小步
    'max_depth': 6,                 # 树深适中，兼顾拟合和泛化
    'n_estimators': 1000,           # 最多 1000 轮，靠早停决定实际轮数
    'subsample': 0.8,               # 每棵树用 80% 样本
    'colsample_bytree': 0.8,        # 每棵树用 80% 特征
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50     # 验证集 50 轮不改善就停
}
```

这几个参数的分工值得说清楚：

- `learning_rate` 控制每棵树的贡献。设小一点（0.05），模型学得慢但更稳；设大一点（0.3），收敛快但容易过拟合。
- `max_depth` 控制单棵树的复杂度。深度 6 意味着最多做 6 次分裂，能捕捉特征之间的交互，但不至于把每个样本都单独记住。
- `subsample` 和 `colsample_bytree` 是两种随机化手段：前者每轮随机抽一部分样本，后者每轮随机抽一部分特征。它们让每棵树看到的「视角」不同，集成起来更稳。
- `early_stopping_rounds` 是最实用的防过拟合工具：训练时盯着验证集的 `mlogloss`，如果连续 50 轮都没有下降，就停下来，避免白跑后面几百轮。

**5 折分层交叉验证。** 因为样本量不大，单次划分的验证结果偶然性太大。我们用 `StratifiedKFold` 把数据分成 5 份，每次拿 4 份训练、1 份验证，轮流做 5 次，最后把 5 次的测试预测平均：

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 存每一折对测试集的概率预测，最后取平均
test_probs = np.zeros((len(test), 3))
val_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]

    model = xgb.XGBClassifier(**params)

    # 训练时传入验证集，早停才有依据
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 用概率算 Log Loss，这正是比赛指标
    val_pred = model.predict_proba(X_val)
    score = log_loss(y_val, val_pred)
    val_scores.append(score)
    print(f"Fold {fold} Log Loss: {score:.5f}")

    # 把这一折模型对测试集的预测累加，最后除以折数
    test_probs += model.predict_proba(test) / skf.get_n_splits()

print("Average Log Loss:", np.mean(val_scores))
```

**关于超参数调优。** 我们在这个比赛里没有做大规模的网格搜索，理由有两点。第一，数据规模不大，5 折交叉验证每跑一遍就要训练 5 个模型，如果再做几十组参数组合，时间成本很高。第二，XGBoost 在默认参数附近通常已经很稳，真正影响结果的往往是「有没有做交叉验证」「有没有按折平均」这些流程层面的决定，而不是某个参数多调了 0.01。

我们的调参策略是「先固定一组经验参数，用交叉验证拿到一个可信的基线，再围绕关键参数小范围试探」。具体来说，我们主要关注三个旋钮：`max_depth`（3～8）、`learning_rate`（0.03～0.1）和 `n_estimators`（配合早停自动决定）。材料的最终配置是深度 6、学习率 0.05、最多 1000 轮加早停。这个组合在验证集上的 Log Loss 稳定，且不会因为树太深而过拟合。

另外，`early_stopping_rounds` 放在参数字典里而不是 `fit` 调用里，是一个版本兼容的小细节：新版 XGBoost 推荐在构造函数里指定早停，否则会给出警告。我们在材料里看到作者特意把它移到了 `params` 中，就是为了消掉这条警告。

### 2.4 模型测试与指标报告

训练完成后，我们首先汇总每一折的验证分数，看模型是否稳定：

```python
# 每折的 Log Loss
for i, s in enumerate(val_scores, 1):
    print(f"Fold {i}: {s:.5f}")

# 平均值和标准差：标准差小说明模型在不同折上表现一致
print("Mean Log Loss:", np.mean(val_scores))
print("Std  Log Loss:", np.std(val_scores))
```

如果五折分数彼此接近，说明模型没有依赖某一次特定的数据划分，结果更可信；如果某一折明显偏高，通常意味着那一折里少数类样本太少，或者存在一些特别难判的病例。

接着我们直接用平均后的概率生成提交文件：

```python
# 三个概率列的列名和顺序必须与样例一致
submission_df = pd.DataFrame(
    test_probs,
    columns=['Status_C', 'Status_CL', 'Status_D']
)

# 把 id 插回第一列
submission_df.insert(0, 'id', test_ids)

# 保存
submission_df.to_csv('submission.csv', index=False)
print(submission_df.head())
```

这里有几个必须检查的点：

1. **列顺序**。`test_probs` 的第 0 列对应 `LabelEncoder` 编码后的 0 号类别，第 1 列对应 1 号，第 2 列对应 2 号。`LabelEncoder` 是按字母序编码的，所以 `C` 是 0、`CL` 是 1、`D` 是 2，正好和 `Status_C`、`Status_CL`、`Status_D` 的顺序一致。如果编码顺序和提交列名对不上，Log Loss 会非常难看。
2. **行数**。`test_probs` 的行数必须等于测试集行数，`test_ids` 的长度也要一致，否则 `insert` 会报错。
3. **概率归一化**。`predict_proba` 输出的每行三个概率天然加起来等于 1，不需要再手动归一化；如果做了额外的指数或缩放，反而可能破坏这个性质。

我们最后打开提交文件看了前几行，确认每行三个概率都在 0 到 1 之间、且和为 1，`id` 也能和测试集对上。确认无误后才提交。

---

## 三、总结

本文中我们完成了肝硬化结局多分类预测的完整流程。我们首先用 EDA 确认了数据形状、目标 `Status` 的三类分布（`C`、`CL`、`D`）以及提交文件的三个概率列，并检查了特征类型和缺失情况。其次我们做了克制的数据清洗：删除无信息的 `id`、用 `LabelEncoder` 把目标映射成 0/1/2、把训练集与测试集的类别列合并后统一编码，并用断言保证两边特征列顺序一致。接着我们选用 XGBoost 的 `multi:softprob` 目标函数，配置深度 6、学习率 0.05、最多 1000 轮并开启 50 轮早停，在 5 折分层交叉验证下训练，每折都用 `mlogloss` 监控并计算验证集的 Log Loss。最后我们把 5 折模型对测试集的概率预测取平均，按 `Status_C`、`Status_CL`、`Status_D` 的顺序生成提交文件。

这个比赛给我们的启发主要有两点。第一，**多分类问题的输出是概率，不是标签**。Log Loss 评价的是概率质量，所以我们必须提交 `predict_proba` 的结果，并且要保证概率列的顺序和类别编码顺序严格对应——这是一个很容易出错、但一旦出错代价极大的细节。第二，**分层交叉验证在小样本医学数据上尤其重要**。它既能让评估更稳定，又能通过「按折平均测试预测」起到集成学习的效果，相当于免费获得了一点模型融合的收益。对入门者来说，把这两件事做对，往往比追求复杂模型更能提升最终成绩。
