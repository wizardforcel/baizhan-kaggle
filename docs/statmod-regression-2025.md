# Statmod, regression-2025

## 一、比赛简介

### 1.1 比赛背景

Statmod 是以统计与机器学习教学为目的的 Kaggle 练习赛系列。它并不是某个具体行业的商业比赛，而更像是一门统计建模课程的配套练习：数据规模不大、评价标准明确、流程完整，很适合刚接触 Kaggle 的入门者熟悉「读数据、清洗、建模、提交」这一整套动作。本场是 2025 年的回归赛，由 Kaggle 用户 Nina Golyandina 主办。从比赛页面可以看到，它只有 30 名参赛者、8 支队伍和 52 次提交，属于典型的小规模教学赛。

赛题的英文描述是：This is the second statmod test competition for predicting the ranking of the film 'Pulp Fiction' (Криминальное чтиво) - the variable Expected in the train data。

俄语描述为：Данные по оценке фильмов и результатам тестов. Нужно по набору оценок фильмов и результатам тестов предсказать, насколько человеку понравится фильм Криминальное чтиво。

用大白话说，这是一场用电影评分和心理测试数据来预测用户对电影《低俗小说》的主观喜爱的比赛。每个受访者都填写了对多部电影的评分，并完成了若干测试题。我们要用这些历史评分和测试结果，来预测他在《低俗小说》这一题上的期望评分。

数据集的列名是俄语，文件名为 `pred_crim_train.csv` 与 `pred_crim_test.csv`。文件名里的 `crim` 并不是指犯罪，而是指《Криминальное чтиво》，即《低俗小说》的俄文名。Kaggle 数据页明确说明：Data are comma-delimited. You should predict the column "Expected" (Pulp Fiction = Криминальное чтиво)。训练集包含要预测的列 `Expected`，测试集没有该列。数据页还提到共有 325 列，每一行对应一名受访者。

### 1.2 比赛目标

我们拿到两张表：

训练表 `pred_crim_train.csv` 包含受访者 Id、大量电影评分及测试特征列，以及目标列 `Expected`，即受访者对《低俗小说》的评分。

测试表 `pred_crim_test.csv` 包含相同的 Id 与特征列，但没有 `Expected`。

我们的任务是，基于训练集学出特征到 `Expected` 的映射关系，对测试集的每个 Id 给出一个实数预测 `Predicted`，并以 `Id, Predicted` 两列的 CSV 提交。问题的核心是回归：不是判断类别，而是预测一个连续的数值评分。训练集和测试集的结构完全一致，只是测试集缺少目标列。

### 1.3 评价标准

官方说明 Evaluation 为 RMSE，即均方根误差。

RMSE 的定义是先算每个样本的预测值与真实值的差，平方后求均值再开根号。用公式写出来是：先把每个样本的误差平方，求和后除以样本数，最后开平方根。数值越小越好，说明整体预测更接近真实评分。平方的惩罚会让大偏差被更严厉地扣分，所以模型要尽量避免出现离谱的极端预测。

举个例子：某个样本预测 8 分、真实 6 分，误差是 2，平方后是 4；另一个样本误差只有 1，平方后是 1。平方运算把大误差的权重放大了。最后开根号是为了把误差单位拉回到和评分相同的尺度，方便我们直观理解「平均差了几分」。比赛提交格式为 `Id, Predicted` 两列，与训练集的 Id 一一对应。

## 二、解题过程

我们按 EDA、清洗、训练调优、测试报告四步走。下面每一步都配上一段可直接运行的代码和中文说明。

### 2.1 数据探索（EDA）

EDA，全称 Exploratory Data Analysis，中文叫探索性数据分析。这一步的目标不是马上建模，而是先搞清楚数据长什么样：能不能读进来、有多少行多少列、目标列是什么类型、有没有缺失。摸清楚这些底细，后面建模才不会踩坑。

我们先想把数据读进来。这里有一个容易踩的坑：俄语数据常使用 `cp1251` 编码，直接用默认编码读取会抛 UnicodeDecodeError。用 try/except 先尝试默认编码，失败时改用 cp1251，回退方案保证数据一定能读进 pandas。

pandas 的 `read_csv` 默认按 UTF-8 编码解析文件。如果文件实际是 cp1251，某些西里尔字母会变成非法的字节序列，pandas 就会抛出 `UnicodeDecodeError`。`try/except` 的写法是：先尝试主方案，出错就执行备用方案，而不是让程序直接崩溃。这种「失败后回退」的思路在处理来源不明的数据时非常实用，尤其是俄语、中文这类非英文编码的 CSV。

```python
import pandas as pd
import numpy as np

TRAIN_PATH = '/kaggle/input/statmod-regression-2025/pred_crim_train.csv'
TEST_PATH  = '/kaggle/input/statmod-regression-2025/pred_crim_test.csv'

# 尝试默认编码读取，失败时用 cp1251
try:
    print('尝试默认编码读取...')
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
except UnicodeDecodeError:
    print('默认失败，改用 cp1251 编码读取...')
    train = pd.read_csv(TRAIN_PATH, encoding='cp1251')
    test  = pd.read_csv(TEST_PATH, encoding='cp1251')

print('训练集形状：', train.shape)
print('测试集形状：', test.shape)
```

这里 `train.shape` 返回 `(行数, 列数)`，先用它确认数据规模。材料里还通过 `os.walk` 遍历 `/kaggle/input` 目录打印出所有输入文件，目的是确认数据文件是否成功挂载到 Kaggle 环境，这也是 EDA 的第一步。读完后我们先看数据基本信息，了解列数和目标分布。

```python
import numpy as np
import pandas as pd

# 看前 3 行，观察列的含义
print(train.head(3))

# 看列类型统计，区分数值列和对象列
print(train.dtypes.value_counts())

# 看目标列 Expected 的基本统计
print(train['Expected'].describe())
```

从形状可以看到训练集约 325 列，目标列 `Expected` 是一个连续值。列名是俄语，解释起来比较困难，但数值列可以直接用于建模。EDA 的目的是确认数据能读入、目标是连续值、没有明显读取错误，为下一步清洗做准备。

### 2.2 数据清洗

我们做了四步清洗：分离目标与 Id、对齐列顺序、填充缺失值、只保留数值列。目的是让训练和测试在完全一致的特征空间里作比较，并消除 NaN 带来的计算报错。

```python
target = 'Expected'
id_col = 'Id'

# 从训练集中分离特征和目标
X = train.drop(columns=[target, id_col], errors='ignore')
y = train[target]

# 测试集只去掉 Id
X_test = test.drop(columns=[id_col], errors='ignore')
```

分离之后要确保测试集列顺序和训练集完全一致，否则模型会把特征错配。

```python
# 按训练集列顺序对齐测试集
X_test = X_test[X.columns]
```

接下来用均值填充缺失值。均值填充简单易懂，对数值列不会大幅改变分布。

```python
# 用均值填充训练和测试的缺失值
X = X.fillna(X.mean(numeric_only=True))
X_test = X_test.fillna(X_test.mean(numeric_only=True))
```

为了避免类别编码的复杂度，基线方案只保留数值列。

```python
# 只保留数值型列
X = X.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

print('保留特征列数：', X.shape[1])
```

这样清洗后，训练集和测试集特征数量一致、没有 NaN、全部为数值类型，可以直接喂给梯度提升模型。丢弃类别列会损失信息，但保证了流程稳定，适合做基线。

### 2.3 模型训练与超参数调优

我们选用 XGBoost 的 `XGBRegressor`。它对表格数据效果稳定，内置对缺失值的处理，支持早停防止过拟合。

首先用 80/20 分割出验证集，保持随机种子可复现。

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

模型参数采用常见的安全配置：

```python
# 构造 XGBoost 回归模型
model = xgb.XGBRegressor(
    n_estimators=1000,      # 最多训练 1000 棵树
    learning_rate=0.05,     # 学习率较小，更稳定
    max_depth=6,            # 树深度中等，防止过拟合
    n_jobs=-1,              # 用满 CPU
    random_state=42,
    early_stopping_rounds=50  # 验证 50 轮无提升就停止
)

# 训练并用验证集监控
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)
```

注意在新版 XGBoost 中 `early_stopping_rounds` 需要放在构造函数里，而不是 `fit` 参数里，材料里特别标注了这个坑。我们没有做网格搜索，因为在教学基线里先让流程跑通比盲目调参更重要。后续可尝试类别编码补回、目标 log 变换或 Optuna 贝叶斯调参来提升分数。

### 2.4 模型测试与指标报告

训练结束后，先在验证集上计算 RMSE，评估模型是否学到规律。

```python
val_preds = model.predict(X_val)

# 计算均方根误差
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print(f'验证集 RMSE：{rmse:.5f}')

# 看一下预测范围是否合理
print('预测最小/最大：', val_preds.min(), val_preds.max())
print('真实最小/最大：', y_val.min(), y_val.max())
```

为了判断模型是否有效，我们可以和一个简单的基线对比：全用训练集均值预测。

```python
baseline_rmse = np.sqrt(np.mean((y_val - y_train.mean()) ** 2))
print('预测均值的 RMSE：', baseline_rmse)
print('模型提升比例：', (baseline_rmse - rmse) / baseline_rmse)
```

如果模型 RMSE 明显低于基线，说明特征确实有预测力。最后生成提交文件。

```python
final_preds = model.predict(X_test)

submission = pd.DataFrame({
    'Id': test[id_col],
    'Predicted': final_preds
})

submission.to_csv('submission.csv', index=False)
print(submission.head())
print('已保存 submission.csv，行数：', len(submission))
```

提交前确认行数、列名 `Id, Predicted` 与模板一致，且预测值无 NaN 或 inf。至此整个流程闭环。

## 三、总结

本文中我们完成了 Statmod 回归 2025 从数据读取到提交的完整流程。我们首先处理了俄语编码 cp1251 的读取问题并做了初步 EDA，确认了目标列 Expected 是《低俗小说》的评分；其次进行数据清洗，分离目标与 Id、对齐列顺序、用均值填充缺失并只保留数值列以保证基线稳定；接着用 XGBoost 回归训练模型，采用 80/20 划分、学习率 0.05、最大深度 6 与早停 50 轮，并修正了 early_stopping_rounds 位置的版本差异；最后在验证集上报告 RMSE，与均值基线对比后生成符合 Kaggle 格式的 submission.csv。

这场赛题告诉我们，先让流程跑通再谈分数。丢弃类别列和简单均值填充虽然牺牲了一部分信息，却换来了一个可复现的基线。等基线稳定后，再逐步加入类别编码、特征工程和更精细的调参，每一步的增益都清晰可见，这正是数据科学工程化的基本思路。
