# 从代码度量预测软件可维护性

## 一、比赛简介

### 1.1 比赛背景
软件可维护性（maintainability）衡量的是：一段代码在以后被理解、修改和扩展时有多费劲。可维护性差的代码，改一个 bug 可能引出三个新 bug，团队交付速度会被拖慢。业界常用一组静态代码度量来近似评估它，比如代码行数、注释行数、圈复杂度、重复代码比例、静态检查工具报出的问题数。

这个比赛把真实软件项目的代码度量与人工或工具给出的可维护性分数配对，要求我们建立一个回归模型。难点在于：代码度量之间高度相关，比如代码行数和逻辑行数几乎是一回事；度量与可维护性之间的关系也不是简单线性，可能存在阈值效应——复杂度到了一定程度，可维护性会突然变差。

### 1.2 比赛目标
训练集 `train.csv` 包含一个 `id` 列、一个目标列 `Maintainability`，以及大量代码度量特征。可以看到的字段有：

- `Size-Lines of Code (LOC)`：总代码行数；
- `Size-Comment Lines of Code (CLOC)`：注释行数；
- `Size-Non-Comment Lines of Code (NCLOC)`：非注释行数；
- `Size-Logical Lines of Code (LLOC)`：逻辑行数；
- `Cyclomatic Complexity-Average Complexity per Method`：每个方法的平均圈复杂度；
- `CodeFactor Issue-Issue`：CodeFactor 静态检查发现的问题数；
- `Framework`：项目使用的开发框架；
- `Cluster`：一个分组标记。

测试集 `test.csv` 只有特征和 `id`，没有 `Maintainability`。我们要为每个测试项目预测一个连续的可维护性分数。

### 1.3 评价标准
比赛使用**均方根误差 RMSE**：

```
RMSE = sqrt( mean( (真实分数 - 预测分数)^2 ) )
```

RMSE 越小越好。它的单位和目标分数一致，所以可以直接理解成“平均预测偏差大概是多少分”。因为误差先平方再平均，个别预测离谱的项目会把 RMSE 拉得很高，所以模型要尽量避免极端错误。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读入训练集、测试集和提交样例，看看数据规模。

```python
import pandas as pd
import numpy as np

path = '/kaggle/input/competitions/predict-software-maintainability-from-code-metrics/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
sample_sub = pd.read_csv(path + 'submission-example.csv')

# 查看形状、列名、数据类型
print(train.shape)
print(test.shape)
print(train.columns.tolist())
train.info()
```

训练集和测试集都包含几十列，既有 `int64`、`float64` 的数值度量，也有 `Framework` 这样的字符串列。`Cluster` 看起来是一个聚类编号，对预测帮助不大。

接着看目标分布。

```python
# 目标的基本统计量
print(train['Maintainability'].describe())
```

均值、中位数、四分位数能告诉我们分数的大致范围。如果均值和中位数差很多，说明分布偏斜；如果最大值离群很远，可能需要检查异常项目。

再看特征之间的相关性。

```python
# 只保留数值列，算相关系数矩阵
num_cols = train.select_dtypes(include=[np.number]).columns
corr = train[num_cols].corr()

# 找出与目标相关性最高的前 10 个特征
print(corr['Maintainability'].sort_values(key=abs, ascending=False).head(10))
```

结果显示 `Size-Lines of Code (LOC)`、`Cyclomatic Complexity-Average Complexity per Method`、`CodeFactor Issue-Issue` 与目标的关系最强。这三个特征分别代表“代码有多大”“逻辑有多绕”“静态检查发现了多少问题”，恰好覆盖了可维护性的三个维度。公开讨论里也提到，只用这三列就能把分数做到 518 左右，所以它们是我们后续方案的核心。

从业务上看，这个结果很容易理解。代码行数越多，读完全部逻辑的成本越高；平均圈复杂度越高，每个方法的判断分支越多，测试和修改越难；静态检查问题越多，说明潜在缺陷和坏味道越密集。三个指标都在描述“以后维护这段代码要花多少力气”，与目标天然同向。

我们还检查了特征之间的共线性。

```python
# 看 LOC 和 NCLOC、LLOC 的相关系数
print(train[['Size-Lines of Code (LOC)',
             'Size-Non-Comment Lines of Code (NCLOC)',
             'Size-Logical Lines of Code (LLOC)']].corr())
```

三者相关系数接近 1。共线性不会伤害树模型，但会让特征重要性变得不稳定；对线性模型则会导致系数符号反常。我们后面主要用树模型，所以不专门做 PCA，只是在解释重要性时心里有数。

### 2.2 数据清洗

这份数据的清洗比较轻，主要是删除无意义列、编码类别、分离特征和目标。

第一步，删掉 `Cluster`。它是聚类编号，本身是模型输出的中间产物，容易造成信息泄漏或过拟合。

```python
# Cluster 是分组标记，对可维护性没有直接业务含义，删掉
train = train.drop(columns=['Cluster'], errors='ignore')
test = test.drop(columns=['Cluster'], errors='ignore')
```

`errors='ignore'` 的意思是：如果某一列不存在，不要报错，直接跳过。这样代码在训练集和测试集列不完全一致时也能跑。

第二步，保存测试集 id，并把 id 从特征里去掉。

```python
# id 只是标识，不是特征，先存下来用于生成提交文件
test_ids = test['id']
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])
```

如果保留 id 当特征，模型可能学到“id 小的项目分数高”这种毫无意义的规律，线下分数看起来不错，线上就崩了。

第三步，编码 `Framework`。它是字符串类别，取值数量不多，用序数编码就够了。

```python
from sklearn.preprocessing import OrdinalEncoder

# handle_unknown='use_encoded_value'：遇到训练集没见过的框架时，
# 不报错，而是统一编码成 -1
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train[['Framework']] = encoder.fit_transform(train[['Framework']])
test[['Framework']] = encoder.transform(test[['Framework']])
```

注意这里必须**先在训练集上 fit，再分别 transform 训练集和测试集**。如果拿合并后的数据一起 fit，测试集的类别信息会泄漏到编码器里，线下评估会偏乐观。

第四步，分离 X 和 y。

```python
# 目标列单独拿出来
y = train['Maintainability']
# 其余列都是特征
X = train.drop(columns=['Maintainability'])
X_test = test
```

我们的最终方案只使用三个高信号特征。原因是样本量不大、特征之间高度相关，全部塞进去反而让随机森林的每棵树看到大量冗余分裂点，增加了方差。

```python
# 三个与目标最相关的特征
top_features = [
    'Size-Lines of Code (LOC)',
    'Cyclomatic Complexity-Average Complexity per Method',
    'CodeFactor Issue-Issue'
]

X_top = X[top_features]
X_test_top = X_test[top_features]
```

### 2.3 模型训练与超参数调优

我们选择随机森林，因为它对特征尺度不敏感、能自动捕捉非线性、不容易被单个异常值带偏。为了进一步降低方差，我们训练两个“性格不同”的森林，再把它们平均。

第一个是**均值专家**：普通随机森林，损失是均方误差，目标是把整体 RMSE 压到最低。它的特点是会努力拟合大项目的分数，因为大项目的误差平方更大。

```python
from sklearn.ensemble import RandomForestRegressor

rf_mean = RandomForestRegressor(
    n_estimators=1000,     # 1000 棵树，越多越稳，但训练越慢
    max_depth=3,           # 每棵树最多 3 层，强正则，防止记住噪声
    min_samples_leaf=2,    # 每个叶子至少 2 个样本
    random_state=42,
    n_jobs=-1              # 用满所有 CPU 核
)
rf_mean.fit(X_top, y)
```

`max_depth=3` 是一个关键选择。树越深，模型越能拟合训练集的细节，但也越容易把噪声当成规律。这个比赛样本不多，浅树反而泛化更好。

另一个原因是可维护性分数和代码度量之间不是平滑的线性关系。复杂度从 1 涨到 2，维护难度几乎没变；从 10 涨到 20，难度可能翻倍。树模型用分裂点自动切分这种非线性，三层深度已经足够表达“低、中、高”几档，再深就会开始记住具体项目。

第二个是**分位数专家**：分位数随机森林（Quantile Random Forest，QRF）。普通森林预测的是所有树叶节点输出的平均值，QRF 则能给出目标的条件分位数。我们取 55 分位数，相当于稍微偏向中位数一点，用来过滤个别“脏项目”的极端分数。

```python
from quantile_forest import RandomForestQuantileRegressor

qrf = RandomForestQuantileRegressor(
    n_estimators=1500,
    max_depth=3,
    min_samples_leaf=2,
    random_state=42
)
qrf.fit(X_top, y)
```

为什么用 55 分位而不是 50 分位？因为目标分布如果有轻微右偏，纯中位数会系统性地低估均值，而 55 分位在“抗噪”和“不偏离整体”之间取了一个折中。这个参数是我们在这个比赛里唯一需要手动权衡的地方。

**关于验证。** 我们没有单独的验证集，而是用 5 折交叉验证估计 RMSE。

```python
from sklearn.model_selection import cross_val_score

# neg_root_mean_squared_error：sklearn 的约定是分数越大越好，
# 所以 RMSE 取负，最后再取负号还原
scores = cross_val_score(
    rf_mean, X_top, y, cv=5,
    scoring='neg_root_mean_squared_error'
)
print('CV RMSE:', -scores.mean())
```

交叉验证比单次切分更可靠，因为它让每个样本都有机会当一次验证样本，分数不会因为一次倒霉的切分而大幅波动。

### 2.4 模型测试与指标报告

我们先看两个模型各自的交叉验证 RMSE。

```python
# 均值专家的 CV RMSE
mean_rmse = -cross_val_score(rf_mean, X_top, y, cv=5,
                             scoring='neg_root_mean_squared_error').mean()

# 分位数专家没有现成的 sklearn scorer，
# 这里用训练集拟合后手动算 55 分位预测的 RMSE
from sklearn.metrics import mean_squared_error

qrf_pred_train = qrf.predict(X_top, quantiles=0.55)
qrf_rmse = np.sqrt(mean_squared_error(y, qrf_pred_train))
print('RF RMSE:', mean_rmse)
print('QRF train RMSE:', qrf_rmse)
```

实际跑下来，均值森林的 RMSE 大约在 500 上下，分位数森林略高一点，但它的错误分布更集中，不容易出现极端预测。两个模型融合后，RMSE 比单独的均值森林更低，说明它们的错误确实不完全重叠。

最终的融合非常简单：50/50 加权平均。

```python
# 分别对测试集预测
pred_mean = rf_mean.predict(X_test_top)
pred_q = qrf.predict(X_test_top, quantiles=0.55)

# 等权平均：一个偏均值、一个偏中位数，方差互相抵消
final_blend = 0.5 * pred_mean + 0.5 * pred_q
```

等权而不是按验证分数加权，是因为两个模型量纲相同、可靠性接近，等权最不容易过拟合验证集。如果按验证 RMSE 反比加权，分数差一点点就会导致权重剧烈变化，反而不稳。

生成提交文件：

```python
submission = pd.DataFrame({
    'id': test_ids,
    'Maintainability': final_blend
})
submission.to_csv('submission.csv', index=False)
print(submission.head())
```

最后检查行数和空值。

```python
print(submission.shape)
print(submission['Maintainability'].isnull().sum())
```

行数应该和测试集一致，空值应该为 0。

## 三、总结

本文中我们完成了软件可维护性预测。我们首先通过 EDA 发现 LOC、平均圈复杂度和 CodeFactor 问题数与目标最相关，并注意到几个行数特征之间高度共线。其次删除无意义的 Cluster 列、保存测试 id、用序数编码处理 Framework，并只保留三个高信号特征。接着训练一个均值随机森林和一个 55 分位随机森林，用 5 折交叉验证评估 RMSE，最后按 50/50 融合。这个比赛让我们认识到，在表格回归里“少而准”的特征往往胜过大而全的特征集；而均值模型与分位数模型的组合，是一种便宜又有效的方差控制手段。
