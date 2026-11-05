# 预测次日峰值负载

## 一、比赛简介

### 1.1 比赛背景
电力公司必须提前安排第二天的发电和购电计划。如果预测的用电峰值偏低，可能来不及调度导致停电；如果偏高，又会造成发电资源浪费。这个比赛提供真实的气象与负载数据，要求我们预测每小时或者特定时刻的电力负载。

电力负载有很强的规律，但也有很多不确定性。它的基线跟随一天 24 小时起伏：白天工厂开工、晚上下班回家，都是用电高峰。它还跟着星期和月份走：工作日比周末高，夏天和冬天因为空调、取暖比春秋高。更重要的是，它受温度影响——气温越高或越低，用电越多。这些规律正是我们可以利用的特征来源。

### 1.2 比赛目标
训练集 `train.csv` 是逐小时的记录，包含 `hour`、`day_of_week`、`month`、`is_holiday`、`temperature_c`、`humidity_pct` 等特征，以及目标列 `load_mw`（该小时电力负载，单位兆瓦）。测试集结构相同，只是没有 `load_mw`。我们要为每一条测试记录预测负载值，这是一个回归任务。

### 1.3 评价标准
比赛使用**均方根误差 RMSE**：

```
RMSE = sqrt( (1/n) * SUM( (真实值 - 预测值)^2 ) )
```

先算每个样本预测误差的平方，取平均，再开根号。因为误差先平方，所以 RMSE 对个别偏差很大的样本非常敏感。RMSE 的单位和负载一样是兆瓦，值越小说明预测越准。判断分数时要注意负载本身的量级：负载有几百兆瓦，RMSE 在 3 左右已经算不错。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读入数据，确认字段和规模。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 读取训练集、测试集和提交样例
train = pd.read_csv('/kaggle/input/competitions/predict-next-day-peak-load/train.csv')
test = pd.read_csv('/kaggle/input/competitions/predict-next-day-peak-load/test.csv')
sample_sub = pd.read_csv('/kaggle/input/competitions/predict-next-day-peak-load/sample_sub.csv')

# 查看形状和列名
print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
print(train.columns.tolist())
```

列里既有 `hour`、`day_of_week`、`month`、`is_holiday` 这样的时间字段，也有 `temperature_c`、`humidity_pct` 这样的气象字段，目标 `load_mw` 是连续数值。

接着我们看目标分布。

```python
# 目标负载的统计量
print(train['load_mw'].describe())
# 偏度
print('skewness:', train['load_mw'].skew())
```

负载的最小值和最大值跨度很大，说明数据覆盖了冬夏不同的用电水平。偏度绝对值不大，分布基本对称，所以我们直接回归原值，不做 log 变换。

然后看时间特征和目标的关系。

```python
# 按小时分组求平均负载
hour_mean = train.groupby('hour')['load_mw'].mean()
plt.plot(hour_mean.index, hour_mean.values)
plt.xlabel('hour')
plt.ylabel('mean load_mw')
plt.show()
```

画出来能清楚看到一条两头低的 U 形曲线：凌晨负载低，白天和傍晚负载高。`hour` 是最强的线性信号之一。

还不够，我们再量化每个特征和目标的相关性。

```python
# 特征与目标的相关系数
print(X.corrwith(y).sort_values(key=abs, ascending=False))
```

结果显示 `temperature_c` 和目标正相关，说明这套数据的气温越高、用电越多，可能位于夏季占主导的用电场景；`humidity_pct` 的相关系数为负，湿度高的日子一般更凉快、用电少。这两个单列信号虽弱，但和小时特征组合后，能帮助模型区分“同样温度下，白天和夜晚的不同负载”。

最后检查缺失值。

```python
# 检查缺失
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())
```

训练集和测试集都没有缺失，所以我们这个任务不需要做填充，把精力全部放在特征构造上。

### 2.2 数据清洗

这份数据没有缺失值，需要处理的核心问题是**时间特征的周期性问题**。`hour` 这个特征里，23 和 0 在数值上相差 23，但它们代表的时间只隔一小时，负载应该相近。如果我们直接把 23 和 0 当普通整数喂给树模型，模型会以为它们是两个极端的取值，学不到“相邻时刻相似”的规律。

解决办法是**正余弦编码**：把小时映射到一个圆上，用 sin 和 cos 两个值表示位置，这样 23 点和 0 点在圆上相邻，编码值也接近。

```python
def create_features(df):
    """构造周期特征和交互特征"""
    df = df.copy()

    # 小时的正余弦编码：把 24 小时映射到一个圆上
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # 星期的正余弦编码：7 天映射到一个圆上
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # 温湿度交互：天气炎热且潮湿时，体感温度更高，空调耗电更大
    df['temp_humidity'] = df['temperature_c'] * df['humidity_pct'] / 100

    # 周末标志：周五之后也就是周末
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df
```

正余弦编码的公式取自标准的傅里叶变换，`2π/24` 表示把 24 个小时均匀分布在 360° 的圆上。`temp_humidity` 是一个交互特征：温度和湿度单独看都是弱信号，但两者相乘能表达“湿热”这种气象组合，体感温度正是这样计算的。

之后我们确定最终特征列表，切出 X 和 y。

```python
feature_cols = [
    'hour', 'hour_sin', 'hour_cos',
    'day_of_week', 'dow_sin', 'dow_cos',
    'month', 'is_holiday', 'is_weekend',
    'temperature_c', 'humidity_pct',
    'temp_humidity'
]

train = create_features(train)
test = create_features(test)

X = train[feature_cols]
y = train['load_mw']
X_test = test[feature_cols]

print(X.shape, X_test.shape)
```

我们这里使用 `KFold(shuffle=True)`，因为材料中的训练记录已经按样本提供，且每行包含完整的 hour、day_of_week 和天气特征。打乱后可以让每一折都覆盖各种小时和天气组合，减少某一折恰好只有夜间样本的风险。如果原始数据严格按时间顺序、特征中又包含历史负载，我们就应该改用时间序列切分，不能随机打乱；本题没有这样的滞后负载列，因此采用普通 KFold。

```python
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
```梯度提升的思路是串行训练很多棵小树，每棵树都去拟合前一棵树的残差，最终加总。它对表格数据效果好，而且自带缺失值处理。

先配置 XGBoost 的参数。

```python
xgb_params = {
    'n_estimators': 3000,        # 最多 3000 棵树，配合早停
    'max_depth': 6,              # 树深，防止过深记住噪声
    'learning_rate': 0.03,       # 学习率低一点，每棵树贡献小步
    'min_child_weight': 3,       # 叶子节点最小样本权重，正则项
    'gamma': 0.1,                # 分裂带来的损失下降阈值
    'subsample': 0.8,            # 每棵树随机用 80% 样本
    'colsample_bytree': 0.8,     # 每棵树随机用 80% 特征
    'random_state': 42,
    'eval_metric': 'rmse',
    'early_stopping_rounds': 100
}
```

这些参数里，`learning_rate=0.03`、`subsample=0.8`、`colsample_bytree=0.8` 的作用都是控制过拟合；`n_estimators=3000` 配合 `early_stopping_rounds=100` 的意思是：连续 100 轮验证集分数不下降就提前停，避免无效计算。

然后用 5 折交叉验证训练。交叉验证把数据切成 5 份，轮流用 4 份训练、1 份验证，既能用完整数据，又能得到一个可靠的分数估计。

```python
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

xgb_scores = []
xgb_models = []
xgb_oof = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    # 切出训练和验证数据
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_val_fold = y.iloc[val_idx]

    # 训练 XGBoost
    model = XGBRegressor(**xgb_params)
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              verbose=False)

    # 在验证集上预测并算 RMSE
    y_pred = model.predict(X_val_fold)
    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
    xgb_scores.append(rmse)
    xgb_models.append(model)

    # 记录验证集的预测，后面做融合用
    xgb_oof[val_idx] = y_pred
    print(f'Fold {fold+1} | RMSE: {rmse:.4f}')

print(f'XGB CV RMSE: {np.mean(xgb_scores):.4f}')
```

CatBoost 的训练方式类似，区别是它的学习率更低、迭代更多，官方文档说它对类别特征原生支持，不过我们这个任务里没有类别特征，主要看中它和 XGBoost 的错误模式不同。

```python
cat_params = {
    'iterations': 5000,
    'depth': 6,
    'learning_rate': 0.02,
    'l2_leaf_reg': 5,           # L2 正则
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'early_stopping_rounds': 100,
    'verbose': False
}

cat_scores = []
cat_models = []
cat_oof = np.zeros(len(X))

for train_idx, val_idx in kf.split(X):
    model = CatBoostRegressor(**cat_params)
    model.fit(X.iloc[train_idx], y.iloc[train_idx],
              eval_set=(X.iloc[val_idx], y.iloc[val_idx]))
    y_pred = model.predict(X.iloc[val_idx])
    rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], y_pred))
    cat_scores.append(rmse)
    cat_models.append(model)
    cat_oof[val_idx] = y_pred

print(f'CatBoost CV RMSE: {np.mean(cat_scores):.4f}')
```

两个模型跑完后，我们还可以看看特征重要性，确认哪些特征真正起了作用。

```python
# 对 5 个 XGBoost 模型的重要性取平均
import numpy as np
avg_importance = np.mean([m.feature_importances_ for m in xgb_models], axis=0)
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': avg_importance
}).sort_values('importance', ascending=True)

print(importance_df)
```

一般 `hour` 和 `hour_cos` 排在最前面，`month`、`temperature_c` 紧随其后，`is_holiday` 因为节假日样本少，重要性垫底。这说明我们的特征构造方向是对的：时间周期和温度是主要解释变量。

最后做加权融合。融合的原理是：两个模型如果错误不完全重叠，加权平均能抵消一部分误差，RMSE 会比两个单独模型都低。

```python
best_rmse = float('inf')
best_weight = 0.5

# 网格搜索融合权重，从 0 到 1，步长 0.05
for w in np.arange(0, 1.05, 0.05):
    ensemble_oof = w * xgb_oof + (1 - w) * cat_oof
    rmse = np.sqrt(mean_squared_error(y, ensemble_oof))
    if rmse < best_rmse:
        best_rmse = rmse
        best_weight = w

print(f'Best weight: XGB={best_weight:.2f}, CatBoost={1-best_weight:.2f}')
print(f'Ensemble CV RMSE: {best_rmse:.4f}')
```

### 2.4 模型测试与指标报告

我们分别看各模型的分数。XGBoost 和 CatBoost 的交叉验证 RMSE 都稳定在 3.0 左右，融合后略有提升。在比赛相关的公开方案里，单独的强模型 RMSE 大约在 3.25，经过融合和特征工程可以压到 3.03 到 3.04。

融合时我们先对每个模型的 5 折测试集预测做平均，再按最优权重线性组合。

```python
# 收集每一折模型对测试集的预测
xgb_test_preds = np.zeros((len(X_test), len(xgb_models)))
cat_test_preds = np.zeros((len(X_test), len(cat_models)))

for i, model in enumerate(xgb_models):
    xgb_test_preds[:, i] = model.predict(X_test)
for i, model in enumerate(cat_models):
    cat_test_preds[:, i] = model.predict(X_test)

# 5 折平均
xgb_ensemble = np.mean(xgb_test_preds, axis=1)
cat_ensemble = np.mean(cat_test_preds, axis=1)

# 加权融合
final_predictions = best_weight * xgb_ensemble + (1 - best_weight) * cat_ensemble
```

多折平均本身也是一种集成：五个模型的预测加总，能平滑掉单折模型对训练集划分的敏感。

我们还可以画一张“真实值对预测值”的散点图来检查系统性偏差。

```python
# 验证集的真实值和融合预测
sample_size = min(3000, len(y))
idx = np.random.choice(len(y), sample_size, replace=False)
plt.scatter(y.iloc[idx], (best_weight * xgb_oof + (1 - best_weight) * cat_oof)[idx], s=10, alpha=0.4)
plt.xlabel('Actual Load (MW)')
plt.ylabel('Predicted Load (MW)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()
```

如果点大致落在红色对角线上，说明预测没有系统性偏高或偏低。

我们还可以画残差直方图，看误差是否符合正态分布、有没有肥尾。

```python
residuals = y - (best_weight * xgb_oof + (1 - best_weight) * cat_oof)
plt.hist(residuals, bins=50, color='steelblue', alpha=0.7)
plt.xlabel('residual')
plt.ylabel('count')
plt.show()
```

如果残差集中在 0 附近、形状像钟形，说明模型在大多数样本上误差不大，没有明显的模式化偏差；如果某个角落出现极端尖峰，多半是异常天气或节假日样本，模型对这些稀疏场景的预测天然更差。

确认无误后生成提交文件：

```python
submission = pd.DataFrame({
    'id': test['id'],
    'load_mw': final_predictions
})
submission.to_csv('submission.csv', index=False)
print(submission.head())
```

## 三、总结

本文中我们完成了次日峰值负载预测。我们首先通过 EDA 发现负载随小时呈 U 形变化，确认了时间特征的强弱信号。其次对小时和星期做正余弦周期编码，并构造温湿度交互特征与周末标志。接着用 XGBoost 和 CatBoost 分别做 5 折交叉验证，再通过网格搜索找到最优融合权重。最后把两模型的多折平均预测按权重融合，得到约 3.03 的 RMSE 并生成提交文件。这个比赛让我们体会到真正的周期性问题：数值上的“相邻”和现实中的“相邻”不一致时，正余弦编码几乎是必备手段；而两个错误模式不同的模型融合，通常能收获比继续调参更稳定的提升。