# CEUPE 大数据分析：钻石价格预测（Diamonds Price）

## 一、比赛简介

### 1.1 比赛背景

这是 CEUPE（一个开设大数据分析课程的教育机构）组织的课程比赛，题目非常经典：**根据钻石的物理特征预测它的价格**。数据就是著名的 `diamonds` 数据集，每一行是一颗钻石的物理测量值和售价。

要理解这个任务，先要知道珠宝行业评价钻石的「4C 标准」：

- **Carat（克拉）**：重量。1 克拉等于 0.2 克。不到 1 克拉的钻，还会再细分成 100「分」，0.75 克拉就是 75 分。
- **Cut（切工）**：从 Excellent（极好）到 Poor（差）。切工影响钻石对光线的反射，切得好才够闪。
- **Color（颜色）**：GIA 的颜色等级从 D（无色）到 Z（浅黄或浅褐）。越接近无色越贵。
- **Clarity（净度）**：从 Flawless（无瑕）到 I3（有明显内含物）。钻石在高温高压下形成，几乎不可能完全纯净，所以净度越高越稀有。

这四个维度之外，数据里还有几个几何测量值：`depth`（深度百分比）、`table`（台面宽度百分比）、`x`（长度）、`y`（宽度）、`z`（高度），单位是毫米。

价格跨度非常大。材料里给了一个直观的对比：1 克拉的钻石可以从 1,500 美元卖到 16,000 美元以上；2 克拉的可以从 6,000 美元卖到 80,000 美元。同样重量的钻石，因为切工、颜色、净度不同，价格能差十倍。

这个任务的难点有三个：第一，`carat` 和 `x/y/z` 高度相关，几乎是在重复描述「这块石头有多大」，线性模型会遇到多重共线性；第二，价格分布右偏，还有不少极端值；第三，三个类别特征（`cut`、`color`、`clarity`）有天然的顺序，但直接编成 0/1/2 又会让模型误以为「差值」有意义。

### 1.2 比赛目标

输入是一颗钻石的物理特征：`carat`、`cut`、`color`、`clarity`、`depth`、`table`、`x`、`y`、`z`。输出是它的价格 `price`。

训练集 `diamonds_train.csv` 含 `price` 列，测试集 `diamonds_test.csv` 不含。提交文件是两列：`id` 和预测价格。

### 1.3 评价标准

这是一个**回归任务**，比赛衡量的是「预测价格和真实价格差多少」。材料里的几份 notebook 都用两个指标来评估：

- **RMSE（均方根误差）**：预测值和真实值之差的平方，求平均后再开根号。单位和价格一样是美元，越接近 0 越好。
- **R²（决定系数）**：模型解释了价格波动的多大比例，越接近 1 越好，等于 0 就相当于「永远猜平均值」。

所以我们的目标很明确：把 RMSE 压下去、把 R² 提上来。

---

## 二、解题过程

整场比赛的思路可以概括成：**先看清数据的长相，把明显错误的记录和极端值处理掉，再用 `Pipeline` 把「补缺 + 标准化 + one-hot」和「随机森林」打包在一起，最后用随机搜索调参。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，看看形状、类型、缺失值和基本统计量：

```python
import pandas as pd
import numpy as np

train = pd.read_csv('../input/ceupe-big-data-analytics/diamonds_train.csv')
test  = pd.read_csv('../input/ceupe-big-data-analytics/diamonds_test.csv')

print(train.shape)             # 行数、列数
print(train.info())            # 每列的类型，顺便看有没有空值
print(train.isnull().sum())    # 这份数据没有缺失值
print(train.describe())        # 均值 / 标准差 / 分位数，先建立“量纲感”
```

`describe()` 里最值得注意的是 `x`、`y`、`z` 的最小值都是 0。一颗钻石长、宽、高不可能为 0，这说明数据里有明显录错的记录，后面要处理。

接着我们画直方图和相关性热力图：

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 七个数值列各画一张直方图，看分布是“对称”还是“长尾巴”
numeric_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']
train[numeric_cols].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# 相关性热力图：两个变量越接近 1 越“同涨同跌”，越接近 -1 越“此消彼长”
corr = train.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
```

从图里我们读出四件事：

1. `carat` 和 `price` 的相关系数是 **0.92**，说明重量是价格最主要的驱动因素；
2. `carat` 和 `x` 的相关系数高达 **0.98**，`carat` 与 `y`、`z` 之间也在 0.95～0.98，四个变量基本在说同一件事——「钻石有多大」，放在线性模型里会造成冗余；
3. `x`、`y`、`z` 与 `price` 的相关系数约 **0.87～0.89**，同样是强相关；
4. `depth` 和 `price` 的相关系数只有 **-0.02**，几乎没关系；`table` 和 `price` 只有 **0.13**；但 `depth` 和 `table` 之间有 **-0.29**，说明「钻得深的石头台面往往小一点」。

还有一个重要的形态：`carat` 和 `price` 散点图不是一条直线，而是一条向上弯的曲线——2 克拉以下接近线性，2 克拉以上价格几乎指数级往上冲。这提示我们，树模型（能自动切分段）可能比线性模型更合适。

### 2.2 数据清洗

根据 EDA 的发现，我们做两步清洗。

第一步，删掉尺寸为 0 的记录。一颗钻石不可能「没有长度」，这类记录多半是人工录入错误，留着只会教坏模型。

第二步，用 IQR 规则处理极端值。IQR 的思路是：把数据从中间往外扩，超出「上下四分位之外 1.5 倍箱体长度」的点就当作异常值。这个规则不假设数据服从正态分布，比「3 倍标准差」更稳健。

```python
# 1) 去掉尺寸为 0 的“无体积”钻石：三个维度必须同时为正
train = train[(train['x'] > 0) & (train['y'] > 0) & (train['z'] > 0)]

# 2) IQR 规则：超出 [Q1 - 1.5*IQR, Q3 + 1.5*IQR] 的点视为极端值
def drop_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return df[(df[col] >= lo) & (df[col] <= hi)]

before = len(train)
# 只对材料里确认有明显异常的三列做处理，避免把正常的贵钻也误删
for col in ['table', 'depth', 'carat']:
    train = drop_outliers(train, col)
print('清洗前', before, '清洗后', len(train))
```

这里我们要克制：IQR 是一把「双刃剑」。删得太狠，会把本来就稀少的高价钻石（恰恰是最有商业价值的那批）全删掉；删得太松，极端值又会把模型往错误方向拉。所以我们只对材料里明确出现异常的三列动手，并且记录清洗前后各删了多少行，方便回溯。

### 2.3 模型训练与超参数调优

清洗完之后，我们把「预处理」和「模型」串成一条 `Pipeline`。这样做有三个好处：第一，交叉验证时预处理只在训练折上拟合，不会把验证折的信息泄漏进来；第二，调参时可以把「补缺策略」这种预处理参数和「树的棵数」一起搜；第三，整条流程可以一键保存、一键复用。

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM_FEATS = ['carat', 'depth', 'table', 'x', 'y', 'z']
CAT_FEATS = ['cut', 'color', 'clarity']
FEATS = NUM_FEATS + CAT_FEATS
TARGET = 'price'

# 数值列：中位数补缺（比均值更抗极端值）→ 标准化（消除量纲差异）
numeric_tf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# 类别列：先填一个占位符 → one-hot 编码
# 不能直接把 Fair/Good 编成 0/1/2，否则模型会误以为“差值”有意义
categorical_tf = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

# ColumnTransformer 把两套预处理并行作用在各自的列上，再拼起来
preprocessor = ColumnTransformer([
    ('num', numeric_tf, NUM_FEATS),
    ('cat', categorical_tf, CAT_FEATS),
])
```

模型我们选随机森林。理由很直接：它由很多棵决策树投票，能自动捕捉「2 克拉以上价格陡增」这种非线性关系，对异常值也不像线性回归那么敏感；而且它几乎不需要特征缩放，工程上很省心。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     RandomizedSearchCV)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42)),
])

# 80% 训练、20% 验证；固定随机种子，保证结果可复现
tr, va = train_test_split(train, test_size=0.2, random_state=42)

# 5 折交叉验证，统一用 RMSE
# sklearn 里 neg_root_mean_squared_error 是“负 RMSE”，所以取负号后越大越好
scores = cross_val_score(model, train[FEATS], train[TARGET],
                         scoring='neg_root_mean_squared_error',
                         cv=5, n_jobs=-1)
print('CV RMSE:', -scores.mean())
```

调参用随机搜索而不是网格搜索。原因很实际：三个参数全组合是 2×6×4=48 组，每组还要跑 5 折，网格搜索太慢；随机搜索只抽 32 组，能在差不多的时间里覆盖更大的参数范围，性价比更高。

```python
param_grid = {
    # 补缺策略也可以当成超参数来搜
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
    # 树越多通常越稳，但训练越慢
    'regressor__n_estimators': [16, 32, 64, 128, 256, 512],
    # 树太深容易记住训练集，太浅又学不到规律
    'regressor__max_depth': [2, 4, 8, 16],
}

search = RandomizedSearchCV(model, param_grid, cv=5, n_iter=32,
                            scoring='neg_root_mean_squared_error',
                            n_jobs=-1, verbose=1, random_state=42)
search.fit(train[FEATS], train[TARGET])
print('best:', search.best_params_)
print('best CV RMSE:', -search.best_score_)
```

材料里还试过另一种组合：用**目标编码**（TargetEncoder）把每个类别替换成「该类别的平均价格」，再接 XGBoost。目标编码对高基数类别更友好，但也更容易过拟合，所以材料里加了 `smoothing=5` 做平滑。我们在下面会把它作为对照方案。

```python
from xgboost import XGBRegressor
from category_encoders import TargetEncoder
from sklearn.metrics import r2_score, mean_squared_error

# 目标编码：用“该类别的平均价格”替换类别名，smoothing 防止小样本类别被极端值带跑
encoder = TargetEncoder(smoothing=5)
X_enc = encoder.fit_transform(train[FEATS], train[TARGET])

x_tr, x_va, y_tr, y_va = train_test_split(X_enc, train[TARGET],
                                          test_size=0.2, random_state=42)
xgb = XGBRegressor()
xgb.fit(x_tr, y_tr)

pred = xgb.predict(x_va)
print('R2  :', round(r2_score(y_va, pred), 4))
print('RMSE:', round(mean_squared_error(y_va, pred, squared=False), 2))

# VAF（方差解释比例）：和 R² 类似，但更直接地看“残差方差占原方差多少”
residual_var = np.var(y_tr - xgb.predict(x_tr))
vaf = (1 - residual_var / np.var(y_tr)) * 100
print(f'VAF: {vaf:.2f}%')
```

材料里的结论是：在对比的几个模型里，XGBoost 的 R² 表现最好，而且交叉验证分数和训练分数接近，说明没有明显过拟合。

### 2.4 模型测试与指标报告

调参结束后，我们用搜索到的最优模型对测试集预测，并按比赛格式生成提交文件。这里有一个小但重要的步骤：**把预测价格夹到合理区间**。价格不可能为负；训练集里的最高价也远低于 20,000，所以把预测值截断到 `[0, 20000]` 既能避免荒唐的负数，也能避免个别树「抽风」给出离谱高价。

```python
# 用最优参数在测试集上预测
y_pred = search.predict(test[FEATS])

submission = pd.DataFrame({'id': test['id'], 'price': y_pred})

# 价格下限是 0；上限按训练数据的实际范围截到 20000，防止个别离谱预测
submission['price'] = submission['price'].clip(0, 20000)

submission.to_csv('diamonds_rf.csv', index=False)
print(submission.describe())   # 检查预测值的分布是否合理
```

最后我们做一次「合理性检查」：看预测价格的均值、最小值、最大值是不是落在合理范围，和训练集的价格分布大致对得上。如果预测均值明显偏离，往往说明特征对齐出了问题（比如测试集多了一列、类别拼写不一致）。

材料里报告的核心结果可以概括为：

- 随机森林方案用 5 折交叉验证、以 RMSE 为评分标准完成调参，最终用 `clip(0, 20000)` 生成提交；
- 对照方案里，XGBoost 的 R² 最高，交叉验证与训练分数接近，没有明显过拟合；
- 目标编码的 `smoothing=5` 在小样本类别上起到了稳定作用；
- VAF 检查用来确认模型确实解释了数据里的大部分方差，而不是只拟合了噪声。

---

## 三、总结

本文中我们完成了 CEUPE 钻石价格预测的完整流程。我们首先用 `describe()`、直方图和相关性热力图摸清了数据：`carat` 和 `price` 相关系数 0.92，`carat` 与 `x/y/z` 的相关系数在 0.95～0.98 之间，`depth` 与价格几乎无关，价格和重量之间是向上弯的非线性关系。其次我们删掉了 `x/y/z` 为 0 的坏记录，并用 IQR 规则对 `table`、`depth`、`carat` 做极端值处理，同时保留清洗前后的行数用于回溯。接着我们用 `ColumnTransformer` 把「中位数补缺 + 标准化」和「占位符补缺 + one-hot」并行拼起来，接上随机森林组成 `Pipeline`，用 5 折交叉验证和 32 组随机搜索调 `n_estimators`、`max_depth` 和补缺策略。最后我们用最优模型预测测试集，把价格截断到 `[0, 20000]` 后生成提交，并用 R²、RMSE、VAF 做了对照评估，其中 XGBoost 加目标编码的方案 R² 最高且没有明显过拟合。

这个比赛给我们的最大启发是：**入门项目里，先把数据看清、把流程搭对，比换模型更重要**。相关性热力图直接告诉我们哪些特征在重复，`Pipeline` 保证了预处理不会泄漏，随机搜索在有限预算里覆盖了更大的参数空间。这些「基本功」在任何一个表格回归任务里都能直接复用。
