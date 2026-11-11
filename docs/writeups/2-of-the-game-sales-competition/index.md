# 游戏销量预测比赛（第二模块：特征工程）

## 一、比赛简介

### 1.1 比赛背景

这个比赛要求我们根据一款游戏的各项属性，预测它的**总销量**（`total_sales`）。数据记录的是游戏发行层面的信息，包括游戏登陆的主机平台（`console`）、类型（`genre`）、发行商（`publisher`）、开发商（`developer`）、发售年份（`release_year`）以及媒体评分（`critic_score`）等。目标值 `total_sales` 是一个连续的数字，所以这是一个典型的**回归**问题——和分类问题不同，回归要预测的不是"属于哪一类"，而是一个具体数值。

材料是一份面向教学的分模块比赛模板：比赛被拆成若干"模块"（Block），第一模块做最基础的线性回归基线，第二模块专门练**特征工程**。模板里反复强调一句话：**所有统计量（平均值、中位数等）只能在训练集上计算**，否则就会用到测试集的信息，属于"数据泄漏"，分数会虚高而在真实提交时崩掉。这一点也是本模块的核心考点。

### 1.2 比赛目标

输入是训练集 `train.csv` 里的游戏记录，输出是该游戏的总销量。整场比赛被拆成模块后，这个第二模块的"提交"比较特殊：不是直接交预测值，而是交一张**答案表**，里面填上我们自己算出的几个结果，包括：新建了多少个特征、用了什么编码方式、怎么处理异常值、最优模型叫什么、它的 RMSE 是多少、以及第一模块线性回归基线的 RMSE 是多少。模板用固定的 `Id=0` 来标识这张表。

### 1.3 评价标准

从模板的代码和字段命名看，评价指标是 **RMSE（均方根误差，Root Mean Squared Error）**：先算每个样本"预测值减真实值"的平方，取平均，再开根号。它衡量的是预测值和真实值平均差多远，**越小越好**。模板还要求拿本模块最优模型的 RMSE 去和第一模块的线性回归基线比较，看特征工程到底带来了多少提升。

## 二、解题过程

需要先说明一点：材料本身是一份教学模板，代码里很多地方写着"学生在此处写代码"，最终只给出各步骤的框架和示例。所以下面有些代码是我们**根据模板要求和示例整理的伪代码**，重点是把每一步的思路讲清楚，而不是逐行复刻某位参赛者的提交。

### 2.1 数据探索（EDA）

我们首先把训练集读进来，确认数据规模和列的类型：

```python
import pandas as pd

# 读入训练集
df = pd.read_csv('train.csv')
print(df.shape)      # 先看有多少行、多少列
df.head()            # 看前几行长什么样
df.info()            # 看每列的类型和缺失情况
```

从列名可以判断出两类特征：一类是**类别列**，比如 `console`、`genre` 取值不多，`publisher`、`developer` 取值可能有成百上千个；另一类是**数值列**，比如 `release_year`、`critic_score`，以及我们要预测的 `total_sales`。我们还要留意哪些列有缺失——模板在基线模型里专门对 `critic_score` 做了中位数填充，说明这一列确实存在缺失。

接下来看目标值的分布：

```python
# 看总销量的分布：大多数游戏销量很小，少数爆款极大
print(df['total_sales'].describe())
df['total_sales'].hist(bins=50)
```

销量数据往往是"长尾"的：绝大部分游戏卖得不多，少数爆款拉高了整体范围。这会直接影响我们后面怎么处理异常值。

### 2.2 数据清洗与特征工程

这一节是本模块的重点，模板把它拆成三个任务。

**第一个任务：创造新特征。** 模板要求至少造 3 个新特征。我们选了几个可从数据直接算出来的：

```python
# 1) 游戏年龄：发售越早的游戏，累计销量通常越高（卖了更久）
df['game_age'] = 2026 - df['release_year']

# 2) 发行商实力：该发行商在训练集里的平均销量
#    注意：统计量只在训练集上算，再映射到全体数据，避免数据泄漏
publisher_mean = df.groupby('publisher')['total_sales'].mean().to_dict()
df['publisher_power'] = df['publisher'].map(publisher_mean)

# 3) 平台热度：该主机平台在训练集里的平均销量
console_mean = df.groupby('console')['total_sales'].mean().to_dict()
df['platform_momentum'] = df['console'].map(console_mean)

# 4) 类型热度：该游戏类型在训练集里的平均销量
genre_mean = df.groupby('genre')['total_sales'].mean().to_dict()
df['genre_popularity'] = df['genre'].map(genre_mean)

# 5) 是不是续作：标题里带 2、3、II、III 的，往往有前作粉丝基础
#    （标题列名以实际数据为准，这里用 name 示意）
df['is_sequel'] = df['name'].astype(str).str.contains(
    r'(\b2\b|\b3\b|II|III)'
).astype(int)
```

这里最容易出错的是 `publisher_power` 这类"分组平均销量"特征：如果我们用全量数据（训练集 + 测试集）来算平均值，等于提前知道了测试集的销量，模型会严重高估自己。正确做法是**只用训练集算平均值**，再通过 `map` 把结果翻译到测试集行上；测试集里没见过的发行商，就留空、之后用中位数补。

**第二个任务：处理类别特征。** 类别列要转成数字。模板建议：取值少的列（`console`、`genre`）直接做独热编码（One-Hot Encoding），也就是给每个取值开一列 0/1 标志；取值多的列（`publisher`、`developer`）先只保留出现最多的前 10 个，其余合并成 `Other`，再做独热编码。这样可以避免一次多出上千列。

```python
# 取值少的类别：直接独热编码
df_encoded = pd.get_dummies(df, columns=['console', 'genre'], drop_first=True)

# 取值多的类别：只保留出现最多的前 10 个，剩下的归为 Other
top_10_publishers = df['publisher'].value_counts().head(10).index
df['publisher'] = df['publisher'].apply(
    lambda x: x if x in top_10_publishers else 'Other'
)
df_encoded = pd.get_dummies(df_encoded, columns=['publisher'], drop_first=True)
```

`drop_first=True` 的意思是每个类别少留一列当"基准"，避免几列加起来恒等于 1 造成完全共线，这会让线性模型无法稳定求解。

为什么 `publisher` 要先只留前 10 个？因为发行商可能有成百上千种，如果每个都开一列，特征维度会爆炸，而且大部分发行商只有一两个游戏、统计上毫无稳定性。把它们合并成 `Other` 相当于"不认识的小发行商统一归一类"，模型仍然能利用"大厂 vs 小厂"的差异。另外，上面这些新特征在遇到缺失时（比如测试集里出现了训练集没见过的发行商、或者 `release_year` 缺失），`map` 会得到空值，我们在训练模型时会统一交给中位数填充器处理。

**第三个任务：处理异常值。** 模板给了四种选择：IQR 法（把落在 `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]` 之外的值视为异常）、Z-score 法（`|z| > 3`）、Winsorize（把异常值直接替换成边界值）、以及不处理。我们选择 **Winsorize**，因为销量本来就是长尾分布，直接删掉爆款样本会丢掉重要信息，而剪到边界既压住了极端值又保住了样本。

```python
# 用分位数算出上下界
Q1 = df['total_sales'].quantile(0.25)
Q3 = df['total_sales'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Winsorize：把超出边界的值剪到边界上，而不是删除
df['total_sales_clean'] = df['total_sales'].clip(lower, upper)
```

### 2.3 模型训练与超参数调优

先复现第一模块的**基线模型**：只用 `critic_score` 和 `Year` 两列做线性回归，缺失的评分用中位数补。基线的作用是给我们一个"及格线"，后面任何复杂模型至少要打得过它才算有用。

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 基线特征：只有两个
X_base = df[['critic_score', 'Year']].copy()
X_base['critic_score'] = X_base['critic_score'].fillna(X_base['critic_score'].median())
y_base = df['total_sales']

# 留 20% 做验证集
X_tr, X_val, y_tr, y_val = train_test_split(
    X_base, y_base, test_size=0.2, random_state=42
)

model_base = LinearRegression()
model_base.fit(X_tr, y_tr)

# RMSE = 均方误差开根号，越小越好
baseline_rmse = np.sqrt(mean_squared_error(y_val, model_base.predict(X_val)))
print(f'Baseline RMSE: {baseline_rmse:.3f}')
```

然后训练两个更强的模型：随机森林（`RandomForestRegressor`）和 XGBoost（`XGBRegressor`）。随机森林是把很多棵决策树的结果平均起来，能自动处理非线性和特征交互；XGBoost 则是一棵一棵"接力"地补足前面模型的误差（梯度提升）。两者在表格数据上都非常能打。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import xgboost as xgb

# 用清洗后的销量作为目标，特征去掉原始销量列，防止"偷答案"
X = df_encoded.drop(['total_sales', 'total_sales_clean'], axis=1)
y = df_encoded['total_sales_clean']

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 数值特征里仍有缺失，统一用中位数补
imputer = SimpleImputer(strategy='median')
X_tr_imp = imputer.fit_transform(X_tr)
X_val_imp = imputer.transform(X_val)   # 注意：用训练集拟合的规则转换验证集

# 随机森林
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_tr_imp, y_tr)
rmse_rf = np.sqrt(mean_squared_error(y_val, rf.predict(X_val_imp)))

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_tr_imp, y_tr)
rmse_xgb = np.sqrt(mean_squared_error(y_val, xgb_model.predict(X_val_imp)))

print(f'RandomForest RMSE: {rmse_rf:.3f}')
print(f'XGBoost RMSE: {rmse_xgb:.3f}')
```

模板还要求用交叉验证（`cv=3` 或 `cv=5`）来评估，避免"某一次切分运气好"造成误判：

```python
from sklearn.model_selection import cross_val_score

# cv=5：把数据切成 5 份，轮流用 4 份训练、1 份验证，取平均
scores = cross_val_score(xgb_model, X_tr_imp, y_tr,
                         cv=5, scoring='neg_root_mean_squared_error')
# sklearn 对误差类指标取负数，所以要乘 -1 换回正数
print('CV RMSE:', -scores.mean())
```

关于超参数，本模块只要求跑通流程和比较模型，不做深度搜索；模板里两个模型都用 `n_estimators=100` 这个默认级别，先看哪个 RMSE 更低，再决定后续要不要加树的数量、调学习率。

读 RMSE 时要注意它和销量是同一个单位：RMSE 是 1.5，就表示"平均每个游戏的预测销量和真实销量差了大约 1.5（销量单位）"。所以比较两个模型时，只要看谁的 RMSE 更小；而和基线相比改进多少，就是把两个 RMSE 相减，得到的是"平均少差多少"。这也是模板最后让我们填 `block2_baseline_rmse` 和 `block2_best_rmse` 两栏的原因——它想看特征工程到底把误差压下去了多少。

### 2.4 模型测试与指标报告

最后把结果整理成比赛要求的答案表。我们把最优模型的名字、它的 RMSE、基线 RMSE、编码方式和异常值处理方法都填进去，并算出改进幅度：

```python
# 把各步骤的结论汇总成一张表
answers = {
    'Id': 0,
    'block2_num_features': 5,            # 我们新建了 5 个特征
    'block2_best_rmse': round(min(rmse_rf, rmse_xgb), 3),
    'block2_best_model': 'XGBoost' if rmse_xgb < rmse_rf else 'RandomForest',
    'block2_encoding_method': 'OneHot',  # 用独热编码处理类别列
    'block2_outlier_method': 'Winsorize',# 用剪边法处理异常值
    'block2_baseline_rmse': round(baseline_rmse, 3),
}

submission_df = pd.DataFrame([answers])
submission_df.to_csv('submission_block2.csv', index=False)

# 顺便打印改进幅度，确认特征工程真的有效
print('Improvement:',
      answers['block2_baseline_rmse'] - answers['block2_best_rmse'])
```

提交前我们还会逐项自查：`block2_num_features` 和我们实际新建的特征数是否一致、`best_model` 是不是真的对应了更低的 RMSE、两处 RMSE 是不是都保留了三位小数——模板里的 `1.234`、`2.456` 只是占位符，必须换成我们自己跑出来的真实数值，否则等于交了一张假答案。

## 三、总结

本文中我们完成了游戏销量预测比赛第二模块的完整流程。我们首先读入数据、确认了类别列、数值列和目标列的结构，并发现总销量呈长尾分布；其次构造了游戏年龄、发行商实力、平台热度、类型热度等新特征，特别强调所有分组统计量只在训练集上计算以杜绝数据泄漏，然后用"少类别直接独热、多类别保留前 10 再独热"的方式编码类别列，用 Winsorize 把极端销量剪到边界；接着复现了第一模块"评分 + 年份"的线性回归基线，并用随机森林和 XGBoost 各训练一个模型、用 5 折交叉验证评估；最后把最优模型、RMSE、编码方式和异常值处理方法汇总成答案表提交。

这个模块给我们最重要的三点经验是：**一是特征工程的价值要靠基线来量化**，没有基线就不知道新特征到底有没有用；**二是所有统计量必须来自训练集**，这是表格类比赛最隐蔽也最致命的错误；**三是异常值不一定要删**，像销量这种长尾数据，剪边往往比删除更合适。