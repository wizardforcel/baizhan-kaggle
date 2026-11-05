# PUBG 决赛名次预测

## 一、比赛简介

### 1.1 比赛背景

绝地求生（PUBG，PlayerUnknown's Battlegrounds）是 Kaggle 上的经典回归预测比赛。玩家在每局比赛中完成一连串行为：移动、开枪、捡装备、救队友等。比赛给了每名玩家的回合级统计数据，要求预测她/他的最终名次百分比 `winPlacePerc`——即在本场比赛中排名超过多少比例的玩家。

比赛背景很有趣：想知道是「杀敌多」还是「走位稳」更能决定吃鸡。数据集来自官方实战记录，包含训练集 `train_V2.csv` 和测试集 `test_V2.csv`。

### 1.2 比赛目标

输入是每个玩家在单局比赛中的行为统计，输出是她的最终决赛名次百分比。`winPlacePerc` 越大表示名次越靠前（0~1 之间）。提交文件需要包含 Id 和预测的 `winPlacePerc`。

```python
# 读入训练数据，确认特征列
train = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
pd.options.display.max_columns = None
train.head()
```

我们可以看到列包括 `Id、` `matchId、` `groupId、` `assists、` `boosts、` `damageDealt、` `DBNOs、` `headshotKills` 等共计上百列统计量。

### 1.3 评价标准

比赛使用均方根误差 RMSE，越小越好，表示预测名次与真实名次的差距越小。为了加速实验，我们在本地常用平均绝对误差 MAE 与 RMSE 结合，同时绘制预测值与真实值的散点图，观察残差分布。

## 二、解题过程

### 2.1 数据探索（EDA）

我们先统计样本数量、缺失情况以及 `winPlacePerc` 的分布：

```python
# 查看 winPlacePerc 的分布
train['winPlacePerc'].describe()
plt.hist(train['winPlacePerc'], bins=50)
```

名次百分比集中在极端两端：大量玩家排名靠后，也有不少排名靠前者。`kills`、`damageDealt`、`walkDistance` 等特征与名次存在负相关，而 `revives`、`teamKills` 等行为同样重要。

我们按 `matchType`（solo、duo、squad、FPP、TPP）切分观察不同模式下名次分布的差异，并用相关热力图找出一组与 `winPlacePerc` 相关性最高的特征，如 `killPlace`、`killPoints`、`walkDistance`、`rideDistance` 等。

### 2.2 数据清洗

数据清洗主要包括处理缺失和生成衍生特征。首先对缺失值进行填充，`winPlacePerc` 缺失用均值填充，其它数值缺失用中位数填充；类别列 `matchType` 用独热编码。

我们参照知名内核《Effective Feature Engineering》进行特征工程。最核心的是**按组聚合特征**：对同一次比赛同队的玩家取平均、总和、排名：

```python
def mean_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    return df.merge(agg, suffixes=['', '_mean'], how='left', on=['matchId', 'groupId'])
```

我们还做了如下派生特征：

```python
def total_distance(df):
    # 徒步距离 + 骑行距离：总移动量
    df['totalDistance'] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
    return df

def killPlace_over_maxPlace(df):
    # 击杀名次占最大名次的比例，体现击杀效率
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    return df

def players_in_team(df):
    # 队伍人数
    df['players_in_team'] = df.groupby(['matchId', 'groupId'])['Id'].transform('count')
    return df
```

这些特征试图捕捉「个人表现 vs 团队表现」「移动能力 vs 战斗能力」等信息。随后对同场比赛内的特征做百分位排名，生成 `_rank` 特征，帮助模型区分不同比赛的难度。

### 2.3 模型训练与超参数调优

我们使用 LightGBM 做回归基线，原因在于它对大规模数据快、对稀疏特征友好：

```python
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

target = 'winPlacePerc'
cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType', target]
cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
X = df[cols_to_fit]
y = df[target].fillna(df[target].mean())

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
model = LGBMRegressor(random_state=0)
model.fit(X_train, y_train)
```

我们尝试了一系列实验：只用原始特征、加入团队聚合、加入总距离和击杀效率特征、加入 matchType 独热编码等。实验通过 `run_experiment` 统一评估 MAE。实验结果表明，团队均值及排名特征以及 `totalDistance`、`killPlace_over_maxPlace` 等自定义特征对 RMSE 有明显改善。

我们没有做复杂的网格搜索，而是固定随机种子并依赖 LightGBM 的默认参数，配合特征工程取胜。

### 2.4 模型测试与指标报告

在验证集上预测后计算 MAE 和 RMSE，并绘制预测 vs 真实的散点图：

```python
y_pred = model.predict(X_valid)
print("MAE:", mean_absolute_error(y_valid, y_pred))
```

我们还计算了排列重要性，发现 `walkDistance`、`rideDistance`、`killPoints`、`headshotKills` 和团队平均特征对模型贡献最大。残差分布显示模型在极端名次上低估略多，这符合战斗类游戏的长尾特性。

最终我们在全量训练集上用最佳预处理重新拟合，对测试集预测生成提交文件：

```python
model.fit(X, y)
test_pred = model.predict(X_test)
submission = pd.DataFrame({'Id': test_ids, 'winPlacePerc': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
```

## 三、总结

本文中我们完成了 PUBG 决赛名次预测的完整流程。我们首先用 EDA 摸清 `winPlacePerc` 的分布和核心特征相关性；其次对缺失值填充并按比赛队伍做聚合、排名、总距离等特征工程；接着用 LightGBM 做回归基线，并通过对比实验验证了自定义特征的有效性；然后在验证集上报告 MAE/RMSE 并用排列重要性解释关键特征；最后在全量数据上训练并生成提交文件。

这个比赛告诉我们：在游戏类数据里，个人统计和团队上下文必须同时考虑，合适的特征工程往往比调参更能带来提升。
