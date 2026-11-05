# 电子游戏日本销量预测（findata-videogames）

## 一、比赛简介

### 1.1 比赛背景

这份数据是一份电子游戏销售记录。每一行是一款游戏，包含它的名称、平台（PlayStation、Xbox、PC 等）、发行年份、类型（动作、角色扮演、体育等）、发行商、开发商、玩家评分、媒体评分，以及它在不同地区的销量。地区分成北美（`NA_Sales`）、欧洲（`EU_Sales`）、日本（`JP_Sales`）和其他地区（`Other_Sales`），单位通常是百万份。

比赛要求预测**日本市场的销量 `JP_Sales`**。这是一个回归问题：输入游戏的各种属性，输出一个连续的销量数字。它和房价预测的本质一样——根据已知特征估计一个数值目标。

难点有三个。第一，**销量分布极度右偏**：大多数游戏只卖几万份，少数爆款能卖几百万份，直接回归会被少数大值主导。第二，**类别特征多**：平台、类型、发行商、开发商都是文本，而且取值很多，编码方式会明显影响效果。第三，**信息不完整**：很多游戏缺评分、缺发行年份，而缺失本身可能和游戏规模有关。

### 1.2 比赛目标

训练集是 `Video_Games.csv`，测试集是 `Video_Games_Test.csv`，提交样例是 `Video_Games_Sample.csv`。训练集包含目标列 `JP_Sales`，测试集没有；测试集还有一个 `Id` 列，预测前要把它去掉，因为它只是行号。提交文件的格式是：

- 第一列是测试样本的编号（取自提交样例）；
- `JP_Sales` 列填模型预测的销量。

除日本销量外，数据里还有北美、欧洲、其他地区三个地区的销量。这里有一个非常重要的细节：**预测日本销量时，不能把日本销量本身放进特征**。北美和欧洲销量可以作为特征使用，因为它们和日本销量相关，而且在测试集里是已知的；但如果把 `JP_Sales` 也算进「全球销量」，就等于把答案泄漏给模型了。

### 1.3 评价标准

材料里的 notebook 标题写着 “RMSE is TOP”，代码里也用 `mean_squared_error` 开根号算 RMSE，所以评价指标是 **RMSE（Root Mean Squared Error，均方根误差）**：

```
RMSE = sqrt( mean( (预测值 - 真实值)^2 ) )
```

RMSE 越低越好，单位和目标一样（百万份）。它先平方再开方，所以对大的错误惩罚特别重：错 10 万的样本比错 1 万的样本严重 100 倍。这也是为什么后面要对目标做对数变换——把「绝对误差」变成「相对误差」，让模型不至于为了几个爆款牺牲掉大量普通游戏。

---

## 二、解题过程

我们的主线是：先做 EDA 看清分布和缺失，再把年份、评分、销量这几类特征整理好，然后用 CatBoost、XGBoost、LightGBM 各训几个版本做折外预测（OOF），最后平均或堆叠成提交。整个过程围绕一个核心思想：**先把数据变成模型能吃的形状，再谈模型**。

### 2.1 数据探索（EDA）

我们首先读取训练集和测试集，确认规模和重复情况：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取训练集、测试集
train = pd.read_csv('/kaggle/input/findata-videogames/Video_Games.csv')
test = pd.read_csv('/kaggle/input/findata-videogames/Video_Games_Test.csv')

# 看形状和类型
display(train.shape, test.shape)
display(train.info(), test.info())

# 测试集的 Id 只是行号，预测前删掉
test = test.drop(['Id'], axis=1)

# 检查重复行：完全一样的记录会让模型高估某些模式
print('训练集重复行数：', train.duplicated().sum())
print('测试集重复行数：', test.duplicated().sum())
```

接着我们逐列统计缺失比例。这一步在销量预测里特别重要，因为评分和发行年份的缺失不是随机的：

```python
# 每列的缺失百分比
for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing * 100)))

# 看前几行，确认销量列的量级
display(train.head(5))
```

然后我们画数值特征的直方图、箱线图和 Q-Q 图，判断分布形状和异常值：

```python
# 数值列的直方图：看偏度和双峰
train.select_dtypes(include=[np.number]).hist(figsize=(16, 12), bins=50)
plt.show()

# 箱线图：看离群点
train.select_dtypes(include=[np.number]).plot(
    kind='box', subplots=True, layout=(4, 4), figsize=(16, 12)
)
plt.show()
```

从业务常识看，销量列一定是右偏的：大量游戏销量接近 0，少数爆款拉出长尾。这提示我们后面要做 `log1p` 变换。评分列则可能有很多 0（表示没有评分）和缺失，需要单独处理。

我们还看了特征之间的相关性。这里用 **Spearman 相关系数**而不是 Pearson，因为 Spearman 只看排名，对右偏和异常值更稳健：

```python
# Spearman 相关矩阵：只对数值列计算
corr = train.select_dtypes(include=[np.number]).corr(method='spearman')

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Spearman correlation')
plt.show()
```

最后看类别特征的分布，确认平台、类型、评级各有多少种取值：

```python
# 每个类别列画柱状图，取前若干类
for col in ['Platform', 'Genre', 'Rating']:
    if col in train.columns:
        train[col].value_counts().head(20).plot(kind='bar')
        plt.title(col)
        plt.show()
```

这些图帮助我们判断：哪些类别样本很少（编码后几乎是噪声）、哪些类别需要合并。

### 2.2 数据清洗

清洗分四步：处理类别缺失、处理数值缺失、处理发行年份、构造新特征。

第一步，类别列缺失填 `'Unknown'`。平台、评级缺失不代表游戏不存在，单独归为一类反而能让模型学到「信息缺失」的模式：

```python
def handle_categorical_missing(df):
    df = df.copy()
    if 'Developer' in df.columns:
        df['Developer'].fillna('Unknown', inplace=True)
    if 'Rating' in df.columns:
        df['Rating'].fillna('Unknown', inplace=True)
    return df
```

第二步，数值缺失。`User_Score` 里可能混着字符串 `'tbd'`（to be determined，待定），直接 `fillna` 处理不了，要先转成数字、把无法转换的变成 NaN，再加一个「是否缺失」的指示列：

```python
def handle_numeric_missing(df):
    df = df.copy()

    # 评分缺失往往意味着游戏太小众，单独标记这个信息
    if 'Critic_Score' in df.columns:
        df['Critic_Score_missing'] = df['Critic_Score'].isna().astype(int)
    if 'User_Score' in df.columns:
        # 'tbd' 会被 coerce 成 NaN
        df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
        df['User_Score_missing'] = df['User_Score'].isna().astype(int)

    # 评分和评分人数缺失填 0，表示「没有记录」
    for col in ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    return df
```

第三步，处理发行年份。缺年份的行直接删掉，因为年份是构造「游戏年龄」的基础；然后用固定年份 2023 减去年份，得到 `Game_Age`：

```python
def handle_year_of_release(df, target_year=2023):
    df = df.copy()
    if 'Year_of_Release' in df.columns:
        df = df.dropna(subset=['Year_of_Release'])
        # 游戏越老，年龄越大；这个特征比原始年份更稳定
        df['Game_Age'] = target_year - df['Year_of_Release']
    return df
```

第四步，构造新特征。我们算「除目标地区外的全球销量」、是否爆款（销量超过 100 万份）、以及媒体评分和用户评分的差异：

```python
def create_new_features(df, target_column='JP_Sales'):
    df = df.copy()

    # 除目标地区外的三个地区销量之和
    sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    sales_columns = [c for c in sales_columns
                     if c in df.columns and c != target_column]

    if sales_columns:
        df['Global_Sales'] = df[sales_columns].sum(axis=1)
        # 100 万份是行业里常用的「大作」门槛
        df['Is_Bestseller'] = (df['Global_Sales'] > 1.0).astype(int)

    # 媒体评分满分 100，用户评分满分 10，先统一量纲再相减
    if 'Critic_Score' in df.columns and 'User_Score' in df.columns:
        df['Score_Difference'] = df['Critic_Score'] - df['User_Score'] * 10

    return df
```

这里 `target_column` 参数是关键：训练时目标列是 `JP_Sales`，构造 `Global_Sales` 时会自动把它排除；预测测试集时目标列不存在，四个地区销量都能用。同一个函数在训练和预测时行为一致，避免了「训练用了答案、预测没有答案」的错位。

### 2.3 模型训练与超参数调优

编码环节我们分三种情况处理：类别少的用 one-hot，类别多的用频次编码，标识类的用标签编码：

```python
from sklearn.preprocessing import LabelEncoder

def encode_categorical_features(df):
    df = df.copy()

    # Platform / Genre / Rating 类别不多，one-hot 最稳
    for col in ['Platform', 'Genre', 'Rating']:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

    # Publisher 有几百个取值，one-hot 会爆炸，改用出现频率
    if 'Publisher' in df.columns:
        freq = df['Publisher'].value_counts(normalize=True)
        df['Publisher_Freq'] = df['Publisher'].map(freq).fillna(0)

    # Developer 和 Name 只是标识，用整数编号即可
    for col in ['Developer', 'Name']:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))

    return df
```

销量类特征还做了 `log1p` 变换。`log1p(x)` 等于 `log(1+x)`，好处是 0 也能算，而且能把长尾压平：

```python
def apply_log_transformation(df, target_column='JP_Sales'):
    df = df.copy()
    log_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales',
                   'Global_Sales', 'Critic_Count', 'User_Count']
    # 绝不把目标列取对数后又留在特征里
    log_columns = [c for c in log_columns
                   if c in df.columns and c != target_column]

    for col in log_columns:
        df['log_' + col] = np.log1p(df[col])
    return df
```

做完这些，原始列（年份、发行商、平台原文、各地区销量等）就可以删掉了，避免同一信息重复出现。训练集和测试集还要做列对齐：以两边的并集为准，缺的列补 0，保证模型看到的列顺序完全一致：

```python
def align_columns(train_df, test_df, target_column='JP_Sales'):
    # 目标列和 Id 不参与对齐
    drop_cols = [target_column, 'Id']
    train_aligned = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_aligned = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    # 并集列名，缺的补 0
    all_columns = sorted(set(train_aligned.columns) | set(test_aligned.columns))
    for col in all_columns:
        if col not in train_aligned.columns:
            train_aligned[col] = 0
        if col not in test_aligned.columns:
            test_aligned[col] = 0

    return train_aligned[all_columns], test_aligned[all_columns]
```

最后用方差阈值去掉几乎不变的列。一列如果 97% 以上都是同一个值，它对模型几乎没有区分能力，留着只会增加噪音：

```python
from sklearn.feature_selection import VarianceThreshold

# threshold=0.03 表示方差低于 0.03 的列被删掉
selector = VarianceThreshold(threshold=0.03)
X_selected = selector.fit_transform(X)
X = pd.DataFrame(X_selected, columns=X.columns[selector.get_support()])

# 测试集只保留训练集选中的列
test_aligned = test_aligned[X.columns]
```

模型方面，第一版我们训练九个模型：三个 CatBoost、三个 XGBoost、三个 LightGBM，参数只在树深、树数量上做小幅变化，用 7 折交叉验证生成折外预测（OOF）：

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb

FOLDS = KFold(n_splits=7, shuffle=True, random_state=42)

models = [
    ('cat_1', CatBoostRegressor(verbose=0, random_state=42)),
    ('cat_2', CatBoostRegressor(verbose=0, random_state=42, iterations=400)),
    ('cat_3', CatBoostRegressor(verbose=0, random_state=42, depth=6)),
    ('xgb_1', xgb.XGBRegressor(random_state=42, n_jobs=1)),
    ('xgb_2', xgb.XGBRegressor(random_state=42, n_jobs=1, n_estimators=400)),
    ('xgb_3', xgb.XGBRegressor(random_state=42, n_jobs=1, max_depth=5)),
    ('lgb_1', LGBMRegressor(random_state=42, n_jobs=1)),
    ('lgb_2', LGBMRegressor(random_state=42, n_jobs=1, n_estimators=400)),
    ('lgb_3', LGBMRegressor(random_state=42, n_jobs=1, num_leaves=31)),
]

for name, model in models:
    oof = np.zeros(len(X))
    pred = np.zeros(len(test_aligned))

    for fold, (trn_idx, val_idx) in enumerate(FOLDS.split(X, y)):
        # 用当前折的训练部分拟合，验证部分只用来打分
        model.fit(X.iloc[trn_idx], y.iloc[trn_idx])
        oof[val_idx] = model.predict(X.iloc[val_idx])

        # 测试集预测按折平均，降低单次预测的波动
        pred += model.predict(test_aligned) / FOLDS.n_splits

    # OOF 分数才是可信的验证分数
    print(name, 'OOF RMSE:', np.sqrt(mean_squared_error(y, oof)))
```

九个模型里，有些可能因为参数或数据问题训练失败，代码里用 `try/except` 跳过，保证不会因为一个模型报错就中断整轮实验。

第二版我们做了三处改进：对目标做 `log1p`、用 `CatBoostEncoder` 处理高基数类别、用 `TimeSeriesSplit` 按游戏年龄排序做时间序列验证，最后用 Ridge 做堆叠：

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from category_encoders import CatBoostEncoder

# 目标取对数，模型优化的是相对误差
y_log = np.log1p(train_aligned['JP_Sales'])

# 按游戏年龄排序，再做时间序列切分
X_sorted = X.sort_values('Game_Age')
y_sorted = y_log.loc[X_sorted.index]

cat_cols = [c for c in X_sorted.columns if X_sorted[c].dtype == 'object']
FOLDS = TimeSeriesSplit(n_splits=5)

for fold, (trn_idx, val_idx) in enumerate(FOLDS.split(X_sorted)):
    X_tr = X_sorted.iloc[trn_idx].copy()
    X_val = X_sorted.iloc[val_idx].copy()
    y_tr = y_sorted.iloc[trn_idx]

    # 类别编码只用训练折的标签统计，避免验证折信息泄漏
    encoder = CatBoostEncoder(cols=cat_cols, random_state=42)
    X_tr[cat_cols] = encoder.fit_transform(X_tr[cat_cols], y_tr)
    X_val[cat_cols] = encoder.transform(X_val[cat_cols])

    model = CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05,
        loss_function='RMSE', verbose=0, random_state=42,
    )
    model.fit(X_tr, y_tr)
```

堆叠（stacking）的意思是：先用多个基模型生成 OOF 预测，再把这些预测当成新特征，训练一个很简单的「元模型」来组合它们。这里元模型用 Ridge 回归：

```python
# oof_results 的每一列是一个基模型的 OOF 预测
meta = Ridge(alpha=1.0)
meta.fit(oof_results.drop(columns='target'), oof_results['target'])

# 用元模型组合各基模型对测试集的预测
final_predictions = meta.predict(test_predictions)
```

这里必须指出材料里的一个 bug：第二版在得到 `meta.predict` 的结果后又做了一次 `np.expm1`。前面每个基模型的预测已经 `expm1` 回到原始销量尺度了，元模型输出的也是原始尺度，再取一次 `expm1` 等于重复反变换，会把预测值指数级放大。我们在复现时去掉了这一步，只在「模型输出还在 log 尺度」时才做 `expm1`。

### 2.4 模型测试与指标报告

验证时我们只看一个数：**OOF RMSE**。它把 7 折（或 5 折）的验证预测拼成完整的一份，再和真实销量比，等价于「每个样本都被没见过的模型预测过一次」。

```python
from sklearn.metrics import mean_squared_error

# 第一版：九个模型取平均
final_predictions = test_predictions.mean(axis=1)

# 用 OOF 报告每个模型的 RMSE
for name in oof_df.columns:
    if name == 'target':
        continue
    rmse = np.sqrt(mean_squared_error(oof_df['target'], oof_df[name]))
    print(f'{name}: RMSE = {rmse:.4f}')
```

材料里没有保留各模型的具体 RMSE 数值，所以这里不引用分数。我们报告的是流程和选择标准：**OOF RMSE 最低的模型或组合胜出**。如果多个模型分数接近，我们优先选平均后的结果，因为平均能降低方差，对测试集更稳。

生成提交前还有几项清理：

```python
# 用提交样例的编号列作为第一列
sample = pd.read_csv('/kaggle/input/findata-videogames/Video_Games_Sample.csv')

# 1) 空值用均值补
final_predictions = pd.Series(final_predictions).fillna(
    pd.Series(final_predictions).mean()
)

# 2) 销量不可能为负，负值截断到 0
final_predictions = np.maximum(final_predictions, 0)

# 3) 写入目标列
sample['JP_Sales'] = final_predictions

# 4) 强制转成数值，处理残留的空串
sample['JP_Sales'] = pd.to_numeric(sample['JP_Sales'], errors='coerce')
sample['JP_Sales'] = sample['JP_Sales'].fillna(0)

# 5) 保留 6 位小数，避免科学计数法影响提交解析
sample.to_csv('submission.csv', index=False, float_format='%.6f')

print(sample['JP_Sales'].min(), sample['JP_Sales'].max())
```

这五步看起来琐碎，但每一步都对应一种常见的提交失败原因：空值、负销量、字符串混入、格式不合法。比赛里因为提交格式丢分，比模型差一点更可惜。

---

## 三、总结

本文中我们完成了电子游戏日本销量的回归预测。我们首先读取训练集和测试集，检查规模、重复行、逐列缺失比例，并用直方图、箱线图、Spearman 相关矩阵和类别分布图摸清了销量右偏、评分大量缺失这两个关键特点。其次我们做了四步清洗：类别缺失归为 `'Unknown'`、`User_Score` 的 `'tbd'` 转成缺失并加指示列、删掉缺年份的行并构造 `Game_Age`、以及用「排除目标地区」的方式构造全球销量和爆款标记。接着我们对平台、类型、评级做 one-hot，对发行商做频次编码，对开发商和名称做标签编码，对销量类特征做 `log1p`，对齐训练集和测试集的列并用方差阈值筛特征，然后训练了三个 CatBoost、三个 XGBoost、三个 LightGBM，用 7 折 OOF RMSE 评估。最后我们尝试了 `log1p` 目标、`CatBoostEncoder`、`TimeSeriesSplit` 和 Ridge 堆叠，修正了材料里重复 `expm1` 的 bug，并对预测做空值、负值和格式清理后生成提交。

这个比赛给我们最大的体会是：**回归任务里，目标的尺度比模型的选择更重要**。同一个 CatBoost，在原始销量上优化 RMSE 和在 `log1p` 销量上优化 RMSE，学到的完全是两种东西——前者盯着几个爆款，后者照顾大多数普通游戏。另一个体会是提交前的「收尾」不能省：负值截断、空值填补、列对齐、小数格式，这些都不提升模型分数，但任何一项出错都会让分数归零。
