# 久违的线性回归（Old Good Linear Regression）

## 一、比赛简介

### 1.1 比赛背景

「久违的线性回归」是一个教学性质的小型回归比赛，数据文件是 `train_ds.csv`、`test_ds.csv` 和 `sample_submission_file.csv`。这是一份匿名的表格数据：特征名是 `X1` 到 `X10` 这样的编号，目标列叫 `target`。它非常适合当作回归问题的入门练手——数据量不大、没有复杂的文本或图像，重点在于把「建模整个流程」走通：读数据、看分布、做特征清洗、训练模型、生成提交。

### 1.2 比赛目标

给定样本的 10 个数值特征 `X1` ~ `X10`，预测目标 `target`。训练集同时包含全部特征和目标，测试集只有特征，我们要为测试集补上 `target` 预测值。提交文件只有两列：`id` 和 `target`。

### 1.3 评价标准

比赛使用**均方根误差（RMSE）**作为评分指标，越低越好：

```
RMSE = sqrt( mean( (预测值 - 真实值)^2 ) )
```

也就是先求预测值和真实值差值的平方，取平均，再开根号。RMSE 的单位和 `target` 一致，能够直观反映「平均而言，预测偏了多少」。选手的 notebook 里也同时打印了 R²（拟合优度），用于辅助判断模型解释了多少方差。

---

## 二、解题过程

我们按照「EDA → 清洗 → 建模 → 报告」的标准四步来走，途中不断对比线性回归、梯度提升树和核岭回归三种模型。

### 2.1 数据探索（EDA）

我们首先读入三个文件，看看形状和信息：

```python
import numpy as np
import pandas as pd

df_train  = pd.read_csv("/kaggle/input/competitions/old-good-linear-regressio/train_ds.csv")
df_test   = pd.read_csv("/kaggle/input/competitions/old-good-linear-regressio/test_ds.csv")
df_sample = pd.read_csv("/kaggle/input/competitions/old-good-linear-regressio/sample_submission_file.csv")

print(f"Train shape: {df_train.shape}")
print(f"Test shape:  {df_test.shape}")
print(f"Sample shape: {df_sample.shape}")

df_train.info()      # 看列类型和是否有缺失
df_train.describe().round(2)   # 看每列的均值、标准差、分位数
```

接着我们检查缺失值，并统计每个特征列的均值，确认数据里没有「漏填」的情况：

```python
print(df_train.isnull().sum())
```

结果全是 0，说明没有缺失值，数据整体很干净。

然后我们做一个「特征与目标的关系」检查：把每个特征 `X` 和目标 `target` 画成散点图，再看特征的核密度分布和相关系数热力图：

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 逐个特征和目标画散点图，直观判断线性关系
X_cols = [col for col in df_train.columns if col != "target"]
for col in X_cols:
    plt.scatter(df_train[col], df_train["target"])
    plt.title(col)
    plt.show()

# 画所有特征的相关系数热力图，找特征之间是否有强共线性
sns.heatmap(df_train.corr(), annot=True, cmap='Reds')
```

散点图让我们对每个特征和目标的关系有了直观感觉，热力图则帮我们发现特征之间是否存在严重的相关（共线性）。这组检查决定了后面「哪些特征值得保留、哪些可以去掉」。

### 2.2 数据清洗

数据没有缺失值，我们主要处理的是**异常值**。通过 `describe()` 我们看到 `X10` 的分布存在明显的极端大值，这类异常会对线性回归的拟合产生不成比例的影响。我们用「99% 分位数截断」的办法把它们压回来：

```python
# 取 X10 的 99% 分位数作为上限
upper_limit = df_train['X10'].quantile(0.99)

# 比这个上限还大的值一律截断到上限，只影响右边的长尾
df_train['X10'] = np.clip(df_train['X10'], a_min=None, a_max=upper_limit)
df_test['X10']  = np.clip(df_test['X10'],  a_min=None, a_max=upper_limit)
```

之所以只用上限截断而不用下限，是因为从分布看主要是**大值异常**。注意测试集也必须做一模一样的截断，否则训练和测试数据的分布就不一致了。

接下来我们做特征标准化。线性相关的模型对特征的量纲敏感，我们把所有特征减去均值、除以标准差，让它们都落在同一尺度上：

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_train[feature_cols])   # 用训练集学均值方差
X_test_scaled = scaler.transform(df_test[feature_cols])   # 用同一套参数变换测试集
```

这里有一个关键习惯：**先 `fit` 再 `transform`**——特征缩放的那些均值和标准差只能从训练集统计，绝不能把测试集混进去学，否则就是数据泄漏。

### 2.3 模型训练与超参数调优

我们把数据按 80% 训练、20% 验证切分，然后在同一份数据上对比三种模型。

**第一，驯服过拟合的线性回归。** 作为本比赛的「正主」，线性回归是最简单也最值得先跑的基线：

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 切分训练、验证集；固定随机种子保证结果可复现
X = df_train.drop(columns=["target"])
y = df_train["target"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 线性回归只是一个最小二乘拟合，没有超参数
model = LinearRegression()
model.fit(X_train, y_train)
```

线性回归对每个特征学一个权重，加权求和得到预测值。它简单、可解释，也是本次比赛评审的主线。

**第二，梯度提升树。** 特征之间的真实关系大概率不是纯线性的，我们把 LightGBM 也放进来对比：

```python
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(n_estimators=300, random_state=42)
lgb_model.fit(X_train, y_train)
```

树模型不需要特征缩放（它对单调变换不敏感），但我们对 `X10` 异常值的处理对它同样有好处，所以两者共用同一套清洗后的数据。

**第三，核岭回归。** 我们再试了一种带核技巧的岭回归，用 RBF 核去捕捉特征的非线性：

```python
from sklearn.kernel_ridge import KernelRidge

# RBF 核 = 高斯核，gamma 控制单个样本影响半径
gaussian_model = KernelRidge(kernel='rbf', gamma=0.1)
gaussian_model.fit(X_train, y_train)
```

关于超参数：线性回归没有参数；LightGBM 的 `n_estimators=300` 和核岭回归的 `gamma=0.1` 都是先试出来的合理默认值。由于数据量不大，我们主要靠验证集分数挑选模型，而不是做密集的网格搜索。

### 2.4 模型测试与指标报告

在验证集上，我们对三种模型分别计算 RMSE 和 R²：

```python
from sklearn.metrics import mean_squared_error, r2_score

# 线性回归的验证集评估
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(f"Root Mean Square error (RMSE) : {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")
```

线性回归已经在验证集上给出了不错的 RMSE，说明特征 `X1`~`X10` 和目标之间确有明显的线性成分；LightGBM 因为能捕捉非线性，通常在验证集上还会略优一点，但涨幅有限。核岭回归和 LightGBM 谁更好，取决于数据的真实结构，我们以验证集的 RMSE 为准。

由于 `target` 的取值接近 0/1（材料里的 notebook 同时打印了分类准确率，说明目标可能接近二值），我们也尝试了把线性回归的输出在阈值 0.5 处二值化、再算准确率的做法，作为对问题性质的交叉验证。

最后，我们用训练好的模型对测试集做预测并生成提交文件：

```python
# 在测试集上预测，输出连续值
final_predictions = model.predict(df_test[feature_cols])

submission = pd.DataFrame({
    'id': df_test.index,
    'target': final_predictions
})
submission.to_csv('submission.csv', index=False)
```

比赛的材料里两个 notebook 内容高度重合，分别是「线性回归」和「逻辑回归」两个角度，可见这道题的标签既可以当回归处理、也可以当近似分类处理。我们以回归为主线，并把分类视角作为校验。

---

## 三、总结

本文中我们完成了「久违的线性回归」这道入门回归赛的完整流程。我们首先用 EDA 看清楚了数据：`X1`~`X10` 共 10 个特征、一个 `target`，没有缺失值，并通过散点图、KDE 和分析相关性初步理解了特征与目标的关系。其次我们做了两件清洗：把 `X10` 的极端大值用 99% 分位数截断，再对全部特征做标准化（并对测试集使用同一个标准化的参数）。接着我们对比了线性回归、LightGBM 和核岭回归三种模型，用 80/20 切分和 RMSE 选型。最后我们在验证集上报告了 RMSE 和 R²，并生成了 `id + target` 格式的提交文件。

这个比赛给我们的最大启发是：**回归问题里，最简单的线性模型永远值得先跑**。它既是下限（baseline），也是理解数据的透镜——一旦它表现尚可，说明特征与目标是线性可解释的；如果它不行，再去上树模型和非线性核也不迟。同时，任何对训练集做的变换（截断、标准化）都要原样施加到测试集，这是避免线上翻车的铁律。