# CASCO 车险赔付预测（findata-casco）

## 一、比赛简介

### 1.1 比赛背景

CASCO（КАСКО）是俄罗斯车险里的一种商业险，和强制性的交强险不同，它保的是**车主自己的车**：车辆被撞、被划、被盗、被水淹，都可以申请赔付。保险公司在承保时要根据车型、驾驶人、地区、历史出险记录等信息，预估这份保单未来可能产生多少赔付，再决定收多少保费。

这个比赛给出一批历史保单的结构化数据，要求参赛者建立一个回归模型，预测目标列的数值。数据来自 `findata-casco`，训练集和测试集分别是 `casco_train.csv` 和 `casco_test.csv`。这类问题和 Kaggle 上常见的房价预测很像：一行一个样本，很多数值列和类别列，目标是一个连续值。

难点主要有三个。第一，**特征多且类型混杂**，数值列和文本类别列混在一起，需要模型自己找到合适的处理方式。第二，**目标值的量级差异大**，小额赔付和大额赔付可能相差几个数量级。第三，**评价指标是百分比误差**，这会让模型对小数值样本的误差特别敏感，不能只盯着大额样本拟合。

### 1.2 比赛目标

训练集 `casco_train.csv` 的最后一列是目标列，前面所有列是特征；测试集 `casco_test.csv` 只有特征，第一列是样本编号。我们要为测试集每一行预测一个目标值，并按下面的格式提交：

- `id`：测试样本编号，取自测试集第一列；
- `target`：模型预测的数值。

材料里的 notebook 用 `train.columns[:-1]` 取特征、`train.columns[-1]` 取目标，说明目标列固定放在最后一列。至于目标具体是赔付金额还是保费，材料没有明说，我们按「连续数值回归目标」来处理，不虚构业务含义。

### 1.3 评价标准

作者在 AutoML 里设置 `eval_metric="mape"`，即 **MAPE（Mean Absolute Percentage Error，平均绝对百分比误差）**。它的算法是：对每个样本算「预测值减真实值，取绝对值，再除以真实值」，把所有样本的这个比例取平均：

```
MAPE = mean( |预测值 - 真实值| / |真实值| )
```

MAPE 越低越好，0 表示完全预测正确。它和 RMSE 的区别在于：RMSE 按绝对误差惩罚，大额样本主导分数；MAPE 按相对误差惩罚，一个 100 元的样本错 50 元（50%）比一个 10000 元的样本错 500 元（5%）更严重。所以优化 MAPE 时，模型必须同时照顾小额样本，不能只把大额样本拟合好。

---

## 二、解题过程

我们的主线是用 **MLJAR AutoML** 自动完成特征工程、模型训练、堆叠和集成。AutoML 的意思是「自动机器学习」：我们只准备好表格和评估指标，它自己尝试多种模型和预处理组合。这样做的好处是省去大量手工试错，特别适合这种特征类型混杂、我们又不了解业务细节的数据。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，看看表头、规模和目标列的位置：

```python
import numpy as np
import pandas as pd

# 读取训练集和测试集
train = pd.read_csv("/kaggle/input/findata-casco/casco_train.csv")
test = pd.read_csv("/kaggle/input/findata-casco/casco_test.csv")

# 看前几行，确认特征长什么样
display(train.head())

# 看规模：多少行、多少列
print("训练集形状：", train.shape)
print("测试集形状：", test.shape)

# 看每列的类型和缺失情况
print(train.info())
missing = train.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0])
```

我们重点关注两件事：一是目标列是否在最后一列，这决定后面怎么切分特征；二是缺失比例最高的几列，如果某列缺失太多，AutoML 的特征选择可能会把它丢掉，我们要心里有数。

接着我们单独看目标列的分布。这一步对回归任务非常关键，因为 MAPE 对目标值的量级很敏感：

```python
# 目标列是最后一列
y_col = train.columns[-1]

# 描述统计：均值、标准差、最小最大、分位数
print(train[y_col].describe())

# 看偏度：大于 0 表示右偏，即少数大值拉长了尾巴
print("偏度：", train[y_col].skew())

# 画直方图，直观看看分布形状
import matplotlib.pyplot as plt
train[y_col].hist(bins=50)
plt.title("Target distribution")
plt.show()
```

如果目标明显右偏，常见做法是做对数变换，把长尾压平。不过这次我们把变换的事交给 AutoML——它在 Compete 模式下会自动尝试不同的目标变换和特征工程组合。我们手工只做最基本的检查，避免和自动流程重复。

然后我们确认训练集和测试集的特征列是否一致：

```python
# 去掉目标列后的特征名
x_cols = train.columns[:-1].tolist()

# 测试集应当包含完全相同的特征列
print("测试集缺少的列：", set(x_cols) - set(test.columns))
print("测试集多出的列：", set(test.columns) - set(x_cols))

# 第一列一般是 id，提交时要用
print("测试集第一列：", test.columns[0])
```

这一步能提前发现「训练集有、测试集没有」的列。如果存在这种列，模型训练时用了它，预测时就会报错，必须在训练前删掉。

### 2.2 数据清洗

因为用的是 AutoML，我们把清洗限制在「不改数据分布」的最小范围，剩下的交给自动流程。具体做了两件事。

第一，明确特征列和目标列，绝不把 id 或目标列误当成特征：

```python
# 特征列 = 除最后一列以外的所有列
x_cols = train.columns[:-1].tolist()

# 目标列 = 最后一列
y_col = train.columns[-1]

# 确认目标列不在特征里，避免「用答案预测答案」的泄漏
assert y_col not in x_cols
```

第二，把 obviously 无用的 id 类列从特征里排除。材料里的做法比较直接：直接把 `train.columns[:-1]` 全部当特征。如果第一列是行号或客户编号，树模型一般不会因为它而变差太多，但为了让模型更干净，我们更倾向于删掉它。不过为了和材料保持一致，下面仍然按原做法走。

关于缺失值，我们没有手工填。原因是 AutoML 内部会针对不同模型选择不同的填补策略（数值列用中位数或均值，类别列用众数或单独一类），手工填反而可能限制它的搜索空间。这一点和传统手工建模不同：**用了 AutoML，就要把「预处理」也当成它搜索的一部分**，不要提前替它做决定。

### 2.3 模型训练与超参数调优

我们首先用 `Compete` 模式跑一版。Compete 是 MLJAR 专门为比赛设计的模式：在有限时间内自动做特征工程（包括 golden features 搜索、kmeans 聚类特征、特征选择），训练多种模型并做堆叠（stacking）：

```python
# 安装并导入 MLJAR（Kaggle 环境里通常需要先装）
!pip install -U mljar-supervised
from supervised.automl import AutoML

# Compete 模式：自动特征工程 + 多模型 + 堆叠
automl = AutoML(
    mode="Compete",
    eval_metric="mape",              # 按比赛的 MAPE 优化
    model_time_limit=5 * 60,         # 每个模型最多训练 5 分钟
    total_time_limit=2 * 3600,       # 整个搜索最多 2 小时
    features_selection=True,         # 开启特征选择
    golden_features=True,            # 开启 golden features 搜索
    algorithms=['Random Forest', 'Extra Trees', 'Xgboost', 'CatBoost'],
)

# 用全部训练数据拟合；AutoML 自己切分验证集
automl.fit(train[x_cols], train[y_col])
```

这里的 `algorithms` 只保留了四种树模型。逻辑回归、神经网络这类模型在结构化表格上不一定更强，但在时间有限时少放几种，能让每种模型分到更多训练时间，搜索更充分。

接着我们跑第二版，手动指定交叉验证和集成方式。这一版的目的不是「再碰碰运气」，而是把验证策略固定成 3 折、打乱、分层，让不同模型之间的分数更可比：

```python
# 第二版：手动控制验证策略和集成规模
custom = AutoML(
    mode="Compete",
    eval_metric="mape",
    model_time_limit=5 * 60,
    total_time_limit=60 * 60,        # 这一版只给 1 小时
    top_models_to_improve=3,         # 对最好的 3 个模型继续微调
    start_random_models=5,           # 先随机搜 5 个起点
    hill_climbing_steps=0,           # 不做爬山，节省时间
    features_selection=True,
    golden_features=False,           # 关掉 golden features，减少过拟合风险
    stack_models=True,               # 开启堆叠
    train_ensemble=True,             # 训练集成
    validation_strategy={
        "validation_type": "kfold",
        "k_folds": 3,
        "shuffle": True,
        "stratify": True,            # 回归任务里按目标分箱后分层
    },
    algorithms=['Random Forest', 'Extra Trees', 'Xgboost', 'CatBoost'],
)

custom.fit(train[x_cols], train[y_col])
```

两版的差别值得说清楚：

- 第一版开启 `golden_features`，让 AutoML 自动构造两两组合特征，探索性更强，但耗时也更大；
- 第二版关掉 `golden_features`，改用 `top_models_to_improve` 对头部模型做微调，并打开 `stack_models` 和 `train_ensemble`，更偏向「把已有模型整合好」，而不是继续找新特征。

这其实是一种常见的调参思路：**先用宽搜找方向，再用窄搜做精修**。如果两版都从头到尾开着所有功能，时间会浪费在重复搜索上。

训练过程中，MLJAR 会为每个模型保存验证集上的 MAPE。我们可以通过 `report()` 打开完整的训练报告，看每个模型的分数、训练时长、特征重要性：

```python
# 打印训练报告（在 Notebook 里会渲染成带图表的页面）
custom.report()

# 也可以直接取排行榜，按验证分数排序
leaderboard = custom.get_leaderboard()
print(leaderboard[['name', 'metric_value', 'train_time']].head(10))
```

我们主要看两件事：一是最好的模型比中位数模型好多少，如果差距很小，说明数据本身的可预测性有限；二是训练时间和分数的关系，如果某个模型训了很久却没什么提升，下次就可以把它从候选里去掉。

### 2.4 模型测试与指标报告

训练完成后，我们对测试集做预测，并按比赛格式生成提交文件：

```python
# 用第一版模型对测试集预测
preds = automl.predict(test[x_cols])

# 用第二版模型对测试集预测
preds_custom = custom.predict(test[x_cols])

# 提交格式：id 取自测试集第一列，target 是预测值
submission = pd.DataFrame({
    'id': test.iloc[:, 0],
    'target': preds_custom,
})

# 保存
submission.to_csv('2_submission.csv', index=False)

# 检查：行数是否等于测试集、有没有空值
print(submission.shape)
print(submission['target'].isna().sum())
```

提交前我们做了三项检查：

1. **行数**：`submission` 的行数必须和 `casco_test.csv` 一致，否则 Kaggle 会直接拒收；
2. **空值**：AutoML 偶尔会在某些样本上给出 NaN，必须确认没有；
3. **id 对齐**：`id` 必须来自测试集本身，不能自己重新编号。

指标方面，我们需要回到 MAPE 的定义去读分数。假设验证集 MAPE 是 0.2，意思不是「平均每个样本错 0.2 元」，而是「平均每个样本错了真实值的 20%」。这个数字对目标值的量级不敏感，但对小目标非常敏感：如果真实值是 10，预测成 20，单个样本就贡献了 100% 的误差。所以看 MAPE 时，我们还会额外看一遍误差最大的前几个样本，确认模型不是在少数小值样本上崩掉。

材料里没有保留 AutoML 各模型的最终分数，因此我们不在这里引用具体数字。我们只能说清楚验证协议：两版 AutoML 都用 K 折交叉验证、按 MAPE 选模型，最终提交用的是第二版（手动控制验证策略、开启堆叠和集成）的预测。如果要复现，按上面的参数重跑即可得到当时的分数。

---

## 三、总结

本文中我们完成了 CASCO 车险数据的回归建模。我们首先读取 `casco_train.csv` 和 `casco_test.csv`，确认目标列在最后一列、特征列在训练集和测试集之间一致，并检查了目标分布和缺失情况。其次我们做了最小化的清洗，只固定特征列和目标列的边界，把缺失填补、类别编码、特征变换都留给 AutoML 自动搜索。接着我们跑了两版 MLJAR AutoML：第一版用 Compete 模式开启 golden features 做宽搜，第二版固定 3 折交叉验证、关闭 golden features、开启堆叠和集成做精修。最后我们按 `id + target` 的格式生成提交，并强调用 MAPE 读分数时要关注相对误差，尤其是小目标样本。

这个比赛让我们体会到 AutoML 的正确用法：**它不是「一键出分」的魔法，而是一个需要你定义好评估指标、时间预算和搜索范围的工具**。指标设错，它会把模型优化到错误的方向；时间给太少，它只能跑完默认模型；验证策略不统一，两版结果就没法比较。把这三件事定清楚，AutoML 才能真正帮上忙。
