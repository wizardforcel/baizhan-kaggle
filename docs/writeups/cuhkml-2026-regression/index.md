# CUHKML 2026 回归赛（Regression）

## 一、比赛简介

### 1.1 比赛背景

这是香港中文大学（CUHK）商学院 2026 年机器学习练习赛的回归赛道，和分类赛道**共用同一份业务表格数据**。数据里有 `id`、`action_type`（动作类型）、`expected_use_time`（预计使用时间）以及其他若干业务特征。

回归赛要回答的问题是：**这条记录的「预计使用时间」是多少？**

它和分类赛的区别只有一处，但很关键：分类赛预测的是「是不是配送」这个二选一的标签，回归赛预测的是一个**连续的数值**。所以：

- 分类用 log loss 衡量，回归用 **RMSE**（均方根误差）衡量；
- 分类的输出是 0/1，回归的输出必须是一个实数，**不能取整**；
- 分类赛必须把 `expected_use_time` 从特征里删掉（它是同门回归赛的答案），而回归赛里它就是我们要预测的目标。

这类表格回归任务的难点和分类赛一样，不在模型，而在**数据整理**：同一列在训练集和测试集里类型可能不一致、两边列可能对不齐、数值列可能有缺失。把这些处理干净，一个标准的梯度提升树就能拿到不错的结果。

### 1.2 比赛目标

数据目录里有两个主文件：

- `dataframe_train.csv`：带标签的训练集，包含 `id`、`expected_use_time` 和若干业务特征；
- `dataframe_test.csv`：测试集，只有 `id` 和特征。

我们要给测试集每一行预测一个 `expected_use_time` 的数值。

提交模板是 `regression_example.csv`，同样只有两列：

| 列名 | 含义 |
| --- | --- |
| `order` | 行顺序 |
| `expected_use_time` | 预测出来的时间数值（连续值，不是类别） |

和分类赛一样，做法是**在官方模板上原地填一列**，这样行序、列名、行数都不会出错。

### 1.3 评价标准

评价指标是 **RMSE（Root Mean Squared Error，均方根误差）**，**越低越好**。

用大白话解释：把每条样本的「预测值减真实值」平方，全部加起来求平均，再开根号。它衡量的是「平均而言我们差了多少」。平方的作用是让正负误差不互相抵消，并且**对大误差惩罚更重**——差 10 的样本，误差贡献是差 1 的 100 倍。开根号是为了让单位回到原来的量纲，方便理解。

所以优化 RMSE 时，模型会倾向于**避免离谱的预测**，宁可每条都差一点点，也不要有一条错得特别远。

---

## 二、解题过程

我们的思路可以概括成一句话：**先把训练集和测试集的列对齐、把混杂类型统一编码、用中位数补缺失，再用 LightGBM 和 XGBoost 各跑五折，最后对半混合填进官方模板。** 我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步走。

### 2.1 数据探索（EDA）

我们首先把数据读进来，把目标列和 ID 列分出来。这里注意一个和分类赛的差别：`expected_use_time` 在回归赛里是**目标**，所以要从特征表里 `drop` 掉，但它是被当作标签用的，不是被删掉不用：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

train = pd.read_csv('/kaggle/input/competitions/cuhkml-2026-regression/dataframe_train.csv')
test  = pd.read_csv('/kaggle/input/competitions/cuhkml-2026-regression/dataframe_test.csv')

target_col = 'expected_use_time'
id_col = 'id'

# 目标是一个连续数值，直接取出来当 y，不做任何离散化
y = train[target_col].values

# 特征表去掉目标列和 ID 列（ID 只是编号，对预测没有意义）
X = train.drop(columns=[target_col, id_col], errors='ignore')
X_test = test.drop(columns=[id_col], errors='ignore')
```

接着我们看数据的规模、目标的分布，以及有哪些列需要特殊处理：

```python
# 规模和目标的分布：均值、标准差、最小最大值、分位数
print('train:', train.shape, 'test:', test.shape)
print(pd.Series(y).describe())

# 目标如果有极端长尾，RMSE 会被少数大值主导，需要提前知道
print('skew:', pd.Series(y).skew())

# 每列的类型、缺失数、唯一值个数，用来找出混杂类型和缺失列
info = pd.DataFrame({
    'dtype': X.dtypes.astype(str),
    'missing': X.isna().sum(),
    'n_unique': X.nunique(dropna=False),
})
print(info.sort_values('missing', ascending=False).head(15))

# 两边列的差集，确认有没有没对齐的列
print('only in train:', set(X.columns) - set(X_test.columns))
print('only in test :', set(X_test.columns) - set(X.columns))
```

这一步的核心是找出**哪些列是字符串、哪些列两边类型不一样、哪些列缺值多**。在表格比赛里，这几件事通常比调参更能决定最终分数。

### 2.2 数据清洗

我们做了三件事：列对齐、类型统一、缺失填充。逻辑和分类赛完全一致，因为两个赛道用的是同一份数据。

**第一件，列对齐。** 只保留两边都有的列：

```python
# 只保留训练集和测试集都存在的列，避免特征维度对不上
common_cols = [col for col in X.columns if col in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]
```

**第二件，类型统一。** 关键是**两边都要检查类型**，只检查训练集会漏掉「测试集里混进字符串」的情况：

```python
for col in common_cols:
    # 训练集和测试集都检查，只要有一边是字符串就整列按类别处理
    is_train_object = X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col])
    is_test_object  = X_test[col].dtype == 'object' or pd.api.types.is_string_dtype(X_test[col])

    if is_train_object or is_test_object:
        # 先统一转成字符串，这样数字、文字、缺失值可以放进同一个编码器
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)

        # 训练集和测试集拼起来一起 fit，
        # 保证同一个类别在两边拿到同一个编号
        le = LabelEncoder()
        le.fit(pd.concat([X[col], X_test[col]]))
        X[col] = le.transform(X[col])
        X_test[col] = le.transform(X_test[col])
    else:
        # 数值列：用训练集中位数填缺失。
        # 中位数比均值稳健，不会被极端值带偏
        median_val = X[col].median()
        if pd.isna(median_val):        # 整列都空的极端情况
            median_val = 0
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
```

**第三件，转成纯数值矩阵。** 树模型对 `object` 类型不友好，统一转 `float32`，既清掉类型残留又省内存：

```python
# 全部转成 float32 纯数值矩阵，确保没有任何 object 残留
X_arr = X.values.astype(np.float32)
X_test_arr = X_test.values.astype(np.float32)
print('aligned:', X_arr.shape, X_test_arr.shape)
```

关于「在训练集加测试集上一起 fit 编码器」，我们如实说明：这在 Kaggle 里很常见，好处是编码绝对一致，代价是用到了测试集的分布信息。如果要严格评估泛化能力，应该只在训练集上 fit，把测试集里的新类别统一映射成「未知」编号。这里我们沿用材料的做法。

### 2.3 模型训练与超参数调优

模型用 LightGBM 和 XGBoost，两个都是梯度提升树。验证方式是 5 折交叉验证（KFold，打乱，随机种子 42），每折的验证集预测拼起来就是完整的 OOF 预测。

**第一个模型：LightGBM 回归。** 参数和分类赛几乎一样，只把 `objective` 换成 `regression`、`metric` 换成 `rmse`：

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb   = np.zeros(len(train))   # 每折验证集预测，拼成完整 OOF
preds_lgb = np.zeros(len(test))    # 测试集预测，五折累加后取平均

lgb_params = {
    'objective': 'regression',     # 回归任务
    'metric': 'rmse',              # 和比赛指标一致
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,         # 小学习率配多棵树，通常更稳
    'num_leaves': 31,
    'max_depth': -1,               # 不限制深度，用叶子数控制复杂度
    'feature_fraction': 0.8,       # 每棵树只用 80% 特征，防过拟合
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_arr, y)):
    model_lgb = lgb.LGBMRegressor(**lgb_params, n_estimators=1500)
    # 早停：验证集 RMSE 连续 50 轮不降就停
    model_lgb.fit(X_arr[train_idx], y[train_idx],
                  eval_set=[(X_arr[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    oof_lgb[val_idx] = model_lgb.predict(X_arr[val_idx])
    # 测试集预测按折数平均，五折各贡献 1/5
    preds_lgb += model_lgb.predict(X_test_arr) / kf.n_splits

    # 注意 mean_squared_error 返回的是 MSE，要开根号才是 RMSE
    rmse = np.sqrt(mean_squared_error(y[val_idx], oof_lgb[val_idx]))
    print(f'Fold {fold+1} LGBM RMSE: {rmse:.4f}')
```

这里有个容易写错的地方值得强调：`mean_squared_error` 返回的是**均方误差 MSE**，比赛要的是**均方根误差 RMSE**，必须再开一次根号。忘了开根号，分数会看起来「大得离谱」。

**第二个模型：XGBoost 回归。** 目标函数用 `reg:squarederror`，也就是直接优化平方误差，和 RMSE 完全对应：

```python
oof_xgb   = np.zeros(len(train))
preds_xgb = np.zeros(len(test))

xgb_params = {
    'objective': 'reg:squarederror',   # 平方误差，与 RMSE 一致
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,                    # 限制深度，控制树的复杂度
    'subsample': 0.8,                  # 每棵树只抽 80% 样本
    'colsample_bytree': 0.8,           # 每棵树只抽 80% 特征
    'random_state': 42,
    'tree_method': 'hist',             # 直方图算法，速度快
    'n_jobs': -1,
    # 新版 XGBoost 的早停参数必须写在构造函数里，写在 fit() 里会报错
    'early_stopping_rounds': 50,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_arr, y)):
    model_xgb = xgb.XGBRegressor(**xgb_params, n_estimators=1500)
    model_xgb.fit(X_arr[train_idx], y[train_idx],
                  eval_set=[(X_arr[val_idx], y[val_idx])],
                  verbose=False)

    oof_xgb[val_idx] = model_xgb.predict(X_arr[val_idx])
    preds_xgb += model_xgb.predict(X_test_arr) / kf.n_splits

    rmse = np.sqrt(mean_squared_error(y[val_idx], oof_xgb[val_idx]))
    print(f'Fold {fold+1} XGB RMSE: {rmse:.4f}')
```

我们特意记下了材料里标为「FIXED」的那条注释：早期版本的写法是把 `early_stopping_rounds` 传给 `fit()`，新版 XGBoost 改成了必须在构造函数里指定。这类版本差异在 Kaggle 上非常常见，遇到莫名报错时先查一查参数该放哪儿。

**关于超参数调优。** 材料里没有网格搜索，参数是直接给定的（学习率 0.05、`num_leaves` 31、`max_depth` 6、采样 0.8），树的棵数交给早停决定。这是一套很实用的默认组合：小学习率保证收敛平稳，`num_leaves` 31 对中等规模表格够用，0.8 的采样比例在防过拟合和保留信息之间比较平衡。真要进一步提升，优先级应该是「补更多有效特征」而不是「把学习率调到 0.048」。

### 2.4 模型测试与指标报告

先看两个单模型和混合的结果。混合方式是各占 0.5，这是最简单、最不容易翻车的融合：

```python
# 两个模型各自的整体 OOF RMSE
cv_lgb = np.sqrt(mean_squared_error(y, oof_lgb))
cv_xgb = np.sqrt(mean_squared_error(y, oof_xgb))

# 0.5 / 0.5 混合后的 OOF RMSE
final_oof = oof_lgb * 0.5 + oof_xgb * 0.5
cv_final  = np.sqrt(mean_squared_error(y, final_oof))

print(f'Overall LGBM RMSE   : {cv_lgb:.4f}')
print(f'Overall XGB  RMSE   : {cv_xgb:.4f}')
print(f'ENSEMBLE BLEND RMSE : {cv_final:.4f}')
```

材料里这几个分数同样是运行时格式化打印的，抽取出的文本没有保留具体数值，所以我们只报告**评估方法和配置**，不编造分数。看结果的顺序是：先看五折之间 RMSE 波动大不大（波动大说明数据划分不稳，或者有异常值在捣乱），再看两个模型的整体分数是否接近（差太多说明其中一个可能没收敛好），最后看混合有没有比两个单模型都好。如果混合后反而更差，说明两个模型高度相关，这时候换模型种类比加权重更有用。

接着看特征重要性：

```python
# 用最后一折的模型看特征重要性，确认模型没有依赖可疑的列
lgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_lgb.feature_importances_,
}).sort_values('importance', ascending=False)

xgb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model_xgb.feature_importances_,
}).sort_values('importance', ascending=False)

print(lgb_importance.head(10).to_string(index=False))
print(xgb_importance.head(10).to_string(index=False))
```

如果排名靠前的全是 ID 类或者接近 ID 的列，就要警惕模型记住了样本而不是学到了规律。

**最后是提交。** 回归赛这一步比分类赛省事：官方模板 `regression_example.csv` 里的 `expected_use_time` 直接填连续预测值，**不做任何取整**——因为 RMSE 就是按连续值算的，一旦取整反而会引入误差：

```python
# 读官方模板，在它上面原地填，保证行序和列名正确
example_sub = pd.read_csv(
    '/kaggle/input/competitions/cuhkml-2026-regression/regression_example.csv')

# 两个模型的测试集预测各占一半，直接填连续值
example_sub['expected_use_time'] = preds_lgb * 0.5 + preds_xgb * 0.5

# 固定列顺序，去掉 pandas 索引列再写盘
example_sub = example_sub[['order', 'expected_use_time']]
example_sub.to_csv('submission.csv', index=False)

# 最后确认行数和模板一致、列名正确
print(example_sub.head(10))
print('rows:', len(example_sub), 'columns:', list(example_sub.columns))
```

这里顺带对比一下分类赛：分类赛的作者把概率四舍五入成了 0/1，那一步其实是和 log loss 指标相矛盾的；而回归赛这里保留了完整的小数，是**正确**的做法。同一个作者、同一份数据，两个赛道的提交处理正好形成了对照——**提交格式要跟着指标走**，这是我们从这两场比赛里学到的同一个道理。

---

## 三、总结

本文中我们完成了 CUHKML 2026 回归赛的完整流程。我们首先用 EDA 确认了任务形态：目标是连续的 `expected_use_time`，我们用 `describe()` 和偏度看了它的分布，并统计了每列的类型、缺失和唯一值，找出需要特殊处理的列以及两边没对齐的列。其次我们做了三件清洗：只保留两边共有的列做对齐；对「任一边是字符串」的列统一转成字符串后在训练集加测试集上一起 fit `LabelEncoder`；数值列用训练集中位数补缺失，最后全部转成 `float32` 纯数值矩阵。接着我们用 5 折交叉验证分别训练 LightGBM（学习率 0.05、`num_leaves` 31、特征采样 0.8、早停 50 轮）和 XGBoost（`reg:squarederror`、学习率 0.05、深度 6、行列采样 0.8、`hist` 加速、早停参数写进构造函数），并在每一折都记得把 MSE 开根号成 RMSE。然后我们在 OOF 上算两个单模型和 0.5/0.5 混合的 RMSE，用特征重要性检查有没有可疑列。最后我们把连续的混合预测直接填进官方模板，固定列顺序、去掉索引列写出提交文件。

这个比赛给我们的最大启发是：**回归和分类的骨架几乎一样，差别全在「细节的物理意义」上**。目标连续，所以损失用平方误差、输出不做取整、评估要记得开根号；指标是 RMSE，所以模型会主动躲开离谱的预测。把这些和指标对齐的细节想清楚，比多试几组超参数更值钱。
