# Classification with an Academic Success Dataset

## 一、比赛简介

### 1.1 比赛背景

学术成功预测是教育数据挖掘领域的一个重要应用。简单来说，就是根据学生在学习过程中留下的各类记录，提前判断他最终能否顺利完成学业。这个问题之所以有现实价值，是因为**退学预测**可以帮助学校和教育机构及时发现可能面临困难的学生，从而提前安排辅导、奖学金或其他干预措施，尽可能减少学生流失。

这个比赛使用的数据是一个关于学生学术表现的真实记录集，其中包含大量学生在学期间的指标。这些指标可能包括课程成绩、出勤情况、入学背景、家庭环境等多方面的信息。预测学业结果并不是一件容易的事，因为学生的最终表现往往是由多种因素共同决定的，有些因素能直接反映学习能力，有些因素则是间接的、甚至带有噪声的。

比赛的数据以 CSV 文件的形式提供，方便参赛者用熟悉的工具处理。这类比赛不需要自己收集数据，难点全部集中在**特征理解和建模**上：哪些特征真正影响学业结果、类别型特征如何编码、用什么模型才能取得最好的准确率，这些都是我们需要逐一解决的问题。

### 1.2 比赛目标

本次比赛的任务是一个**分类问题**：给定一条学生的特征记录，预测他最终的学业结果。从代码中可以确认，目标列名为 `Target`，其中包含 `Graduate`（毕业）和 `Dropout`（退学）两个类别，这是一个二分类任务。

我们可以从代码中梳理出数据的结构：

- 训练集（train.csv）：包含 `id` 列、若干特征列和 `Target` 目标列，是我们用来训练模型的样本；
- 测试集（test.csv）：包含 `id` 列和同样的特征列，但没有 `Target` 列，需要我们自己预测；测试集没有标签，我们无法直接看到正确答案，只能通过本地划分验证集来评估模型的好坏；
- 提交样例（sample_submission.csv）：规定了提交文件的格式，每一行对应一条测试样本，需要给出预测的 `Target` 值。

特征列里既有数值型特征，也有类别型特征。从代码中可以看出，类别型特征是通过 `dtype == 'object'`（即数据在 pandas 里被识别为文本/对象类型）来筛查出来的。类别型特征不能直接输入给数学模型，必须先转换成数字，这也是数据预处理的重点之一。

### 1.3 评价标准

比赛使用**准确率（Accuracy）**作为评价指标。

准确率的含义非常直观：模型预测正确的样本数占总样本数的比例。在二分类问题里，就是「把所有学生分成毕业/退学两类，猜对了多少个」。准确率越高越好，满分为 1.0。

准确率的计算方式是：

正确预测的样本数 ÷ 总样本数

因为这个比赛关心的是学生能否顺利毕业，每一次预测错误都意味着错过了一次可能的干预机会，所以使用准确率作为指标既容易理解，又贴合实际场景。我们后面在交叉验证中，也使用 `accuracy_score` 来评估每一折的表现，和比赛的评价方式保持一致。

## 二、解题过程

这个比赛的思路比较清晰：首先读取数据并了解基本情况，然后对类别型特征做编码处理，接着用 LightGBM 模型配合交叉验证进行训练，最后把预测结果整理成提交文件。我们按 EDA、数据清洗、模型训练与调优、模型测试四个步骤来走。

### 2.1 数据探索（EDA）

我们首先读取训练集、测试集和提交样例三个文件，看看数据的基本形状：

```python
# 导入常用的数据处理库：pandas 负责读写表格，numpy 负责数值计算
import pandas as pd
import numpy as np

# 定义数据文件路径
TRAIN_PATH = '/kaggle/input/sch2-reg-2026-d3-4/train.csv'
TEST_PATH = '/kaggle/input/sch2-reg-2026-d3-4/test.csv'
SAMPLE_SUB_PATH = '/kaggle/input/sch2-reg-2026-d3-4/sample_submission.csv'

# 读取三个 CSV 文件
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SAMPLE_SUB_PATH)

# 打印数据形状，确认训练集和测试集分别有多少行多少列
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
```

`shape` 返回的是 `(行数, 列数)` 的元组。通过这个结果，我们可以快速判断训练集和测试集的规模，以及两者特征列数量是否一致。特征列数量一致是基本的检查项，否则测试集和训练集就没有可比性。

数据规模清楚之后，我们来看看每一列的类型。对于一个分类任务，先分清哪些列是数值型、哪些列是类别型，才能决定后面的编码方案：

```python
# 查看每一列的数据类型
print(train.dtypes)

# 筛选出所有类别型特征
# object 类型在 pandas 里通常表示文本/字符串
cat_cols = [col for col in train.columns if train[col].dtype == 'object' and col != 'Target']
print(f"类别型特征数量: {len(cat_cols)}")
```

在建模之前，我们还需要确认一个容易忽略的细节：训练集和测试集里是否都带有 `id` 列。`id` 只是样本编号，本身不携带任何预测信息，如果忘了处理，它会被误当作一个特征：

```python
# 检查训练集和测试集是否包含 id 列
if 'id' in train.columns:
    print(f"训练集包含 id 列，共 {train['id'].nunique()} 个唯一样本编号")
if 'id' in test.columns:
    print(f"测试集包含 id 列，共 {test['id'].nunique()} 个唯一样本编号")
```

最后我们确认一下目标变量的取值情况。目标列名为 `Target`，从代码注释可以看出它的取值是 `Graduate`（毕业）和 `Dropout`（退学）。知道类别种类，才能确定这是一个二分类问题，也才能在后面正确设置模型的输出：

```python
# 查看目标变量的全部取值
print(train['Target'].unique())
```

### 2.2 数据清洗

数据清洗的第一步，是把 `id` 列从特征中去掉。`id` 只是每一行样本的编号，比如第 1 条样本、第 2 条样本，这个编号不会影响学业结果，如果把 `id` 留下来，模型反而会学到一些毫无意义的规律：

```python
# 移除 id 列，它不具有预测能力
if 'id' in train.columns:
    train = train.drop('id', axis=1)
if 'id' in test.columns:
    test = test.drop('id', axis=1)

# 移除之后，特征矩阵里就不再有 id 了
print(f"移除 id 后训练集列数: {train.shape[1]}")
```

第二步是把目标变量 `Target` 从文本转成数字。模型只能处理数值，所以我们需要用 `LabelEncoder`（标签编码器）把 `Graduate` 和 `Dropout` 分别映射成 0 和 1：

```python
# 导入标签编码器
from sklearn.preprocessing import LabelEncoder

# 创建编码器实例
target_encoder = LabelEncoder()

# 对 Target 列进行编码：Graduate -> 0, Dropout -> 1
train['Target'] = target_encoder.fit_transform(train['Target'])

# 打印编码后的类别顺序，方便对照
print(target_encoder.classes_)
```

`fit_transform` 会先统计出所有不同的类别，再依次替换成数字。编码器记住的类别顺序记录在 `classes_` 属性里，测试阶段还需要用它把预测结果翻译回原始的文本标签。

第三步也是整个清洗过程里最关键的一步：处理类别型特征。特征列里的文本不能直接输入模型，我们同样用 `LabelEncoder` 把它们逐列转成数字。这里有一个容易被忽略的坑——**必须同时用训练集和测试集的数据来拟合编码器**：

```python
# 对每个类别型特征单独进行标签编码
for col in cat_cols:
    le = LabelEncoder()
    # 合并训练集和测试集的这一列，一起拟合编码器
    full_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full_data)
    # 用同一个编码器分别转换训练集和测试集
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
```

为什么要合并两边的数据来拟合？因为如果某个类别只出现在测试集里、训练集里没有，而编码器又只在训练集上拟合过，就会报「未知类别」的错误。合并拟合可以保证训练集和测试集使用同一套编码规则，不会出现对不上号的情况。这也是数据清洗里防止数据不一致的一个常见做法。这一步同时避开了对类别型特征的繁琐手动映射，把每个特征都统一成了模型可以直接吃的整数。

做完以上处理后，我们把特征矩阵和目标向量分开。习惯上我们用大写 `X` 表示特征、小写 `y` 表示目标：

```python
# 分离特征矩阵和目标向量
X = train.drop('Target', axis=1)
y = train['Target']

print(f"特征矩阵形状: {X.shape}")
print(f"目标向量形状: {y.shape}")
```

至此，`X` 里全是数值型特征，`y` 里是编码后的 0/1 标签，数据已经可以交给模型了。

### 2.3 模型训练与超参数调优

我们选用的模型是 **LightGBM**。它是梯度提升树（GBDT）家族的优秀实现，每次训练一棵决策树，不断弥补之前所有树的预测误差，最终把很多棵弱树叠加成强模型。LightGBM 对混合类型的表格数据非常友好，训练速度快、准确率也高，是这类比赛的主流选择。

训练之前，我们先把交叉验证的方案定下来。我们选用 **5 折分层交叉验证**：

```python
# 导入模型库和交叉验证工具
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# 设置 5 折交叉验证
# StratifiedKFold 会保证每一折里的类别比例和整体一致
N_SPLITS = 5
folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
```

交叉验证的思路是把训练数据平均切成 5 部分，每次拿其中 4 部分训练、剩下 1 部分验证，循环 5 次让每一部分都被验证过一次。`StratifiedKFold`（分层交叉验证）更进一步，它确保每一折中毕业/退学的比例和全量数据接近，避免某些折恰好全是某一类、把评估结果带偏。

接着我们准备好存放结果的容器，然后进入训练循环：

```python
# test_preds 用于累加测试集上每一折的预测概率
test_preds = np.zeros((len(test), len(target_encoder.classes_)))
# scores 记录每一折的验证准确率
scores = []

print("开始训练...")

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    # 按索引切出本折的训练集和验证集
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
```

循环体里，我们用一组保守的超参数创建 LightGBM 分类器：

```python
    # 创建 LightGBM 分类器
    model = lgb.LGBMClassifier(
        n_estimators=1000,   # 最多训练 1000 棵树
        learning_rate=0.03,  # 学习率：每棵树只贡献一小步，防止步子迈太大
        num_leaves=31,       # 每棵树的叶子节点数，控制模型复杂度
        max_depth=-1,        # -1 表示不限制树的最大深度
        random_state=42,     # 固定随机种子，结果可复现
        verbose=-1           # 关闭训练过程的冗长日志
    )

    # 训练模型，并指定验证集用于早停
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
```

这里有两个超参数值得说明。`learning_rate=0.03` 是一个较小的学习率：提升树靠一步一步修正误差前进，步子太大容易错过最优解，步子小一些虽然慢，但通常更稳。`early_stopping`（早停）则是防止过拟合的关键：训练过程中每训练一轮，模型就在验证集上评估一次，如果连续 50 轮验证准确率都没有提升，就自动停止训练。这样既能自动决定到底需要多少棵树，又避免了训练过头。`num_leaves` 设成 31，与 `max_depth=-1` 配合，是一个在经验上既不太简单也不太复杂的合理起点。

每一折训练完成后，我们立刻在当折的验证集上评估，并把测试集的预测概率累加起来：

```python
    # 在本折验证集上做预测，计算准确率
    val_pred = model.predict(X_val)
    score = accuracy_score(y_val, val_pred)
    scores.append(score)
    print(f"Fold {fold+1} Accuracy: {score:.5f}")

    # 用当前模型预测测试集概率，除以折数后累加
    test_preds += model.predict_proba(test) / N_SPLITS
```

注意我们这里预测的是**概率**而不是直接给出类别。`predict` 只给出 0 或 1，而 `predict_proba` 给出属于每个类别的概率值。把 5 折的概率求平均，等价于让 5 个形状不同但都在完整数据上学过的模型「投票」——单折模型的判断可能偏激，平均之后往往更稳健，这也是一种轻量的集成策略。

5 折循环结束后，我们打印平均准确率，对模型做一个整体评估：

```python
# 计算 5 折的平均准确率
print(f"\nAverage Accuracy: {np.mean(scores):.5f}")
```

平均准确率可以近似看作模型在未知数据上的预期表现。由于我们没有修改任何超参数就拿到了一个稳定的基线分数，这说明 LightGBM 在这份数据上「开箱即用」。如果时间充裕，下一步本可以做网格搜索：把 `num_leaves`、`learning_rate`、`max_depth` 等参数各取几个候选值组合起来，逐一评估，找出最优组合。不过早停已经让模型自动停在了一个较优的训练轮次，模型的复杂度和迭代次数都由验证集说了算，因此进一步调参的收益空间往往有限。

### 2.4 模型测试与指标报告

交叉验证结束后，我们拿到了测试集上累加好的平均概率，现在需要把它变成最终的类别标签。我们用 `argmax` 取出概率最大的类别下标，再通过之前保存的编码器映射回文本：

```python
# argmax 取概率最大的那一类，得到类别下标
final_predictions_indices = np.argmax(test_preds, axis=1)

# 用编码器反变换回原始标签（Graduate / Dropout）
final_predictions_labels = target_encoder.inverse_transform(final_predictions_indices)
```

这里的逻辑是：`test_preds` 是 (样本数, 2) 的矩阵，第 0 列和第 1 列分别是对应 0、1 两个类别的概率，`argmax(axis=1)` 沿着样本方向取出每行概率最大的位置，得到每个样本被判为哪个类别。

然后我们把预测结果填进提交文件并保存：

```python
# 把预测结果写入提交文件的 Target 列
submission['Target'] = final_predictions_labels

# 保存提交文件，index=False 表示不额外写入行号列
submission.to_csv('submission.csv', index=False)
print("submission.csv 创建成功!")
```

提交之后我们还要做一次「复盘」，把交叉验证期间的分数拿出来仔细看一眼。只看一个平均分容易掩盖问题，值得关心的是 5 折之间的差异：

```python
# 输出每一折的准确率、平均分和标准差
print(f"各折准确率: {[f'{s:.5f}' for s in scores]}")
print(f"平均准确率: {np.mean(scores):.5f}")
print(f"标准差: {np.std(scores):.5f}")
```

如果 5 折的分数非常接近，说明模型对数据的划分不敏感，泛化能力比较稳定；如果某几折明显偏低，就要想一想是不是那些折里恰好包含了模型不擅长处理的样本。

最后，我们再检查一遍预测结果的分布是否合理。一个二分类模型如果预测出来的全是同一类，那多半是出问题了：

```python
# 统计预测结果里两个类别各占多少，确认分布没有明显偏差
pred_counts = pd.Series(final_predictions_labels).value_counts()
print("预测结果分布:")
print(pred_counts)
```

把预测分布和训练集里两个类别的真实比例对照，如果相距不远，说明模型学到了数据本身的类别结构，输出是可信的。至此，训练、评估、提交的完整流程就走完了。

## 三、总结

本文中我们完成了 Classification with an Academic Success Dataset 比赛的完整流程。我们首先通过 EDA 读取了训练集和测试集，检查了数据规模、列类型、`id` 列和目标变量的取值。其次我们进行了数据清洗：删除了没有预测能力的 `id` 列，用 `LabelEncoder` 把目标变量 `Target` 从文本转成数字，并对类别型特征做了编码，特别注意合并训练集和测试集来拟合编码器，避免出现未知类别的问题。接着我们用 LightGBM 模型结合 5 折分层交叉验证进行训练，用早停机制自动确定训练轮数，并对测试集做了多折概率平均。最后我们通过反编码生成了提交文件，并复盘了各折准确率和预测分布的稳定性。

这个比赛给我们的主要收获有两点：一是**数据预处理的一致性比算法本身更脆弱**——训练集和测试集必须共享同一套编码规则，任何不一致都会在提交时暴露出来；二是**早停和交叉验证是表格数据建模的有效保障**——它们让模型自动停在合适的位置，让我们在不盲目调参的情况下就拥有了一个稳定可靠的基线。