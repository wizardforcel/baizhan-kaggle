# The Winton Stock Market Challenge

## 一、比赛简介

### 1.1 比赛背景

The Winton Stock Market Challenge 是 Kaggle 举办的一场金融时间序列预测比赛。Winton Capital Management 是策略对冲基金的代表，比赛提供的是合成的股票特征序列，要求参赛者预测未来两天的股票收益率。数据里有大约 25 个 Feature 字段，为 1 到 30 之间编号，每个样本还有过去 180 天左右的滚动收益率 Ret_175 到 Ret_180，以及目标列 Ret_PlusOne、Ret_PlusTwo，分别代表未来第 1 天和第 2 天的收益率。还有两个权重列 Weight_Intraday、Weight_Daily，用于衡量不确定性。

这类比赛的难点很典型：特征很多但含义不明。材料里有一个名为 MacroPCA with Unknown Features 的笔记本，作者专门讨论了 25 个特征中存在大量缺失值，且部分特征可能是类别变量。正是这种「不知道特征到底是什么」的设定，让特征工程变得困难：你不能靠领域知识解释每个列，必须靠统计方法挖掘信号。MacroPCA 的作者指出，部分特征看起来像是分类变量，比如 Feature_1、Feature_9、Feature_10、Feature_13、Feature_15、Feature_16、Feature_20，这些列的取值范围很窄，可能是离散编码。如果把它们当成连续变量处理，模型会学到错误的距离关系。

金融数据还有一个共同点：收益率通常很小、方差很微弱，而且噪声很大。模型如果只追求「平均绝对误差」小，就很容易在零值上得到漂亮分数，但从投资角度看，真正值钱的是把非零机会找出来。所以比赛往往要求预测在后续评估中要综合权重和方向性。作者在最后的提交脚本里尝试把 XGBoost 预测与中位数预测做加权融合，就是为了避免模型在噪声上过度自信。

### 1.2 比赛目标

训练数据包含 Id、数十个 Feature、一段历史收益率序列、两个未来收益率目标以及两个权重列。测试数据没有未来收益率列，提交文件需要输出对未来收益的预测。材料中的示例只聚焦 Ret_PlusOne 和 Ret_PlusTwo，这两列代表未来两天的累积收益率。

由于特征名称都是 Feature_1、Feature_2 这种无意义名字，任务本质是一个纯数值回归问题：给定一组高维、缺失率极高的数值特征，预测两个连续的未来收益值。没有类别标签，也不做分类，只做点预测。

### 1.3 评价标准

官方评价指标在材料里未明确写出名称，但从代码中可以看出，作者用 `mean_absolute_error` 做内部验证，并把结果与「全零预测」比较。全零预测的意思是完全不预测，预测值全为 0。金融收益率的平均绝对值通常在千分之几到百分之几之间，所以 MAE 的量级在 0.01 左右。作者在例子里把 y 轴限制在 0 到 0.021，说明模型要比全零预测显著更小才算有用。

从 Winton 类比比赛中常见做法看，最终排行榜通常使用加权均方误差或基于方向的评分，并考虑不确定性权重。内部验证时，Pseudo-Huber 损失（reg:pseudohubererror）被选为目标损失，它是对线性损失的平滑近似，对异常值不那么敏感。简单说，平方损失会把极端误差放大，而线性损失更稳定，适合收益率这种有肥尾分布的数据。

## 二、解题过程

### 2.1 数据探索（EDA）

数据读取时先用 zipfile 解压训练文件。训练集有 40,000 行、约 211 列，包含 25 个 Feature 列、历史收益率列和两个目标列。测试集结构类似，只是目标列缺失。

```python
from zipfile import ZipFile
import pandas as pd

zip_file = ZipFile('../input/the-winton-stock-market-challenge/train.csv.zip')
df = pd.read_csv(zip_file.open('train.csv'))
df.head()
```

从打印的摘要可以看出特征的离散度差异很大。Feature_1 的值集中在 1 到 10，NA 比例高达 33,313 行；Feature_2 是连续值，NA 约 9,146 行；Feature_7 的数值在 33,800 到 99,861 之间，量级比其它特征大千倍；Feature_8 的值在 0.0098 到 0.3650 之间。不同特征的缺失率从几百到三万不等，说明数据不是随机缺失，很可能有业务原因：某些特征只在特定资产上存在。

```python
summary = df.describe()
print(summary.loc[['Min.', '1st Qu.', 'Median', 'Mean', '3rd Qu.', 'Max.', "NA's"]])
```

这种极端的量级差异让许多机器学习模型感到不舒服。我们后面会看到作者在 PCA 实验里专门讨论是否需要先做缩放。因为特征本身尺度相似度有限，贸然缩放可能反而抹去信息，所以要做对比实验。

另一个观察是历史收益率列 Ret_175 到 Ret_180 的值非常小，数量级在 1e-4 左右，且存在大量 NaN。因为每只资产上市时间不同，早期没有足够历史，缺失是必然的。这种情况需要先用滚动统计补足，或者直接把缺失当成信息保留。训练集里 180 天的收益序列并不都是完整的，有的资产只有最近几十天的记录。如果我们强行用均值填补，会把「没有历史」伪装成「历史收益为零」，反而误导模型。因此我们更倾向于保留缺失，让模型自己决定如何处理。

### 2.2 数据清洗

清洗的核心是缺失值处理。金融数据里缺失不等于「未知」，它往往意味着该资产在该时间段不适用。材料里有两个方向：MacroPCA 实验使用专门处理缺失值的 PCA 算法 CellWise，鲁棒于行缺失和列缺失；XGBoost 实验则采用特征选择来规避缺失问题。

我们参考 XGBoost 线路，先选出十个最重要特征。作者先用一次 XGBoost 训练得到特征重要性，然后手动挑出最重要的十个：

```python
# 从完整特征集中挑出十个最重要列
select = [1, 5, 6, 13, 14, 18, 22, 25, 26, 27]
X_train = X[:30000, select]
X_val = X[30000:35000, select]
X_test = X[35000:, select]
```

这种「先用模型选特征，再用这十个特征重新训练」的做法叫做特征预筛选。它的好处是降低过拟合风险：原始特征数可能上百，很多特征之间高度相关，全部一起喂给模型会让树模型在噪声上开枝散叶。把特征压到十个，模型更容易学到稳定规律。

对缺失值本身，XGBoost 有内置的处理方式：它在分裂时把缺失样本送到默认值收益更高的那一侧，相当于会自动学习缺失的模式。所以我们不需要提前做均值填补，直接把 NaN 留给模型处理。只是要注意，数值型特征必须先用 `pd.to_numeric` 统一类型，否则字符串的 NaN 会导致类型错误。

### 2.3 模型训练与超参数调优

我们采用 XGBoost 回归器，使用 Pseudo-Huber 损失 `reg:pseudohubererror`，这是一种对异常值不敏感的损失函数，比平方损失更稳定。训练过程先做粗调，再逐步精调。

```python
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

param = {
    'learning_rate': 0.1,
    'verbosity': 0,
    'objective': 'reg:pseudohubererror',
    'tree_method': 'gpu_hist',
    'n_estimators': 100,
    'n_jobs': -1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 0,
}
```

这里的 `tree_method='gpu_hist'` 用 GPU 加速训练，`subsample` 和 `colsample_bytree` 都是 0.8，表示每次训练只随机选 80% 的样本和 80% 的特征，这是一种正则化手段，防止树模型记死训练集。

调参用自己写的网格搜索工具 `mysearch`。它可以同时调一维或二维参数网格，返回最好 MAE 和对应的参数组合。

```python
# 示例：同时调 max_depth 和 min_child_weight
param1 = {'max_depth': [1, 3, 5]}
param2 = {'min_child_weight': [1, 3, 5]}
score, best_param = mysearch(X_train, y1_train, X_val, y1_val,
                             param, param1, param2)
```

结果显示 max_depth=5、min_child_weight=3 效果最好。最后精调得到的模型参数是：

```python
best_xgbr = XGBRegressor(
    objective='reg:pseudohubererror',
    tree_method='gpu_hist',
    max_depth=5,
    min_child_weight=3,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.9,
    alpha=0,
    learning_rate=0.01,
    n_estimators=700,
)
```

学习率降到 0.01、树的数量提升到 700，配合 0.9 的子采样，是典型的「慢学、多学」策略，更容易收敛到稳定解。

训练时我们把训练集和验证集合并起来再训练一次，因为调参后不再需要验证集。我们对 Ret_PlusOne 和 Ret_PlusTwo 各自训练一个模型，因为两个目标有不同的噪声水平。

```python
best_xgbr.fit(X_train_val, y1_train_val)
y1_test_pred = best_xgbr.predict(X_test)
best_xgbr.fit(X_train_val, y2_train_val)
y2_test_pred = best_xgbr.predict(X_test)
```

### 2.4 模型测试与指标报告

评估时我们先算基准：全零预测的 MAE。基准值等于真实值的平均绝对值。如果模型不能比全零预测更低，那就毫无价值。

```python
from sklearn.metrics import mean_absolute_error

benchmark = [np.abs(y1_test).mean(), np.abs(y2_test).mean()]
fitted = [
    mean_absolute_error(y1_test_pred, y_test[:, 0]),
    mean_absolute_error(y2_test_pred, y_test[:, 1]),
]
```

材料里的图表显示 XGBoost 模型在两个目标上都明显优于全零预测。最终作者把模型预测与中位数预测做加权融合：`y_new_pred = 0.1 * y_xgb + 0.9 * median(y)`。这种融合是为了降低过拟合，因为 XGBoost 在小样本上容易过度学习噪声，而中位数预测是极度保守但稳健的策略。

```python
y1_new_pred = 0.1 * y1_new_pred + 0.9 * np.median(y[:, 0])
y2_new_pred = 0.1 * y2_new_pred + 0.9 * np.median(y[:, 1])
```

最后按提交样例格式拼装结果。材料里样本提交有 60 个预测列，前 58 列是历史位置的占位，末两列才是我们预测的 Ret_PlusOne 和 Ret_PlusTwo。作者用 `np.hstack` 把占位数组、模型预测、中位数预测拼接起来，写进 submission.csv。

从整体看，这个流程的价值不在于调出了顶尖分数，而在于思路：先用模型选特征降维，再用对异常值鲁棒的损失函数训练，最后用保守策略融合防止过拟合。作者也提到最终排名约 30 名，说明基线思路是可靠的，但要冲到前列，需要更多特征工程，比如 MacroPCA 提取的潜在因子。值得注意的是，MacroPCA 实验显示去除疑似类别特征后，PCA 的预测能力反而提升，这进一步支持「先筛选、再建模」的策略。

## 三、总结

本文中我们完成了 The Winton Stock Market Challenge 的基线流程。我们首先用 zipfile 加载合成的特征和历史收益率，发现特征量级差异极大且缺失严重；其次通过 XGBoost 特征重要性筛选出十个核心特征，直接把缺失值交给模型处理而不做硬填补；接着用 Pseudo-Huber 损失训练 XGBoost 回归器，并用网格搜索找到 max_depth=5、min_child_weight=3 等参数；然后比较 XGBoost 与全零预测的 MAE，确认模型确实捕捉到信号；最后在测试集上训练、用中位数融合做保守平滑并按样例格式提交。

这个练习给我们最大的启发是：在特征含义不明、缺失严重、噪声极高的金融数据上，「少即是多」往往更有效。先用模型选出真正有预测力的少数特征，再用鲁棒损失和保守融合控制过拟合，是一种可迁移的通用策略。
