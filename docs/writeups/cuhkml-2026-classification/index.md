# CUHKML 2026 分类赛（Classification）

## 一、比赛简介

### 1.1 比赛背景

这是香港中文大学（CUHK）商学院在 2026 年组织的一场机器学习练习赛，分**分类**和**回归**两个赛道，用的是同一份业务表格数据。材料里的数据列名很有业务味：有一个 `id` 列、一个 `action_type` 列（记录这条记录对应的动作类型，其中有一类是 `DELIVERY`），还有一个 `expected_use_time` 列（预计使用时间）。

分类赛要回答的问题是：**这条记录的动作是不是「配送」（DELIVERY）？**

这类比赛最大的难点不在模型有多复杂，而在**表格数据常见的三个坑**：

1. **字段类型混杂**。同一列在训练集里是数字，在测试集里却可能混进了字符串，直接丢给模型会报错；
2. **特征列不完全对齐**。训练集和测试集的列不一定一模一样；
3. **标签泄漏**。同门比赛共用一份数据，回归赛的标签是 `expected_use_time`，如果分类赛不小心把它当成特征，就等于提前看到了答案。

所以我们的整条流水线，重点是**把数据整理干净、把两边对齐**，模型部分反而很标准：LightGBM 和 XGBoost 各跑一遍，再各占一半混合。

### 1.2 比赛目标

数据目录里有两个主文件：

- `dataframe_train.csv`：带标签的训练集，包含 `id`、`action_type`、`expected_use_time` 以及其他业务特征；
- `dataframe_test.csv`：测试集，只有 `id` 和特征，没有标签。

我们要预测的是「是否为 DELIVERY」，也就是把 `action_type` 这一列变成二分类标签：是 DELIVERY 记 1，其余记 0。

提交格式比较特别。官方模板叫 `classification_example.csv`，只有两列：

| 列名 | 含义 |
| --- | --- |
| `order` | 行顺序（整数） |
| `action_type_DELIVERY` | 该行是否为 DELIVERY（0 或 1） |

也就是说，我们不是提交一个 `id` 加一个预测，而是**在官方给的模板上原地填一列**。这个细节后面会专门讲，因为它是材料里作者踩坑最多的地方。

### 1.3 评价标准

评价指标是**对数损失（log loss）**，**越低越好**。

用大白话解释：模型对每条样本输出一个 0～1 之间的概率，log loss 衡量的是「模型对自己的判断有多自信，以及判断错的时候被罚得有多重」。猜得越准、越自信，损失越小；**自信地猜错**会被罚得非常狠。所以模型不能光求「对」，还要输出**校准良好**的概率。

这里有一个材料里没有明说、但从代码能看出来的矛盾，我们后面会重点讨论：作者用 log loss 做交叉验证，但最终提交时却把概率四舍五入成了 0 和 1 两个整数。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**先把两边的列对齐、把混杂类型统一编码、把泄漏列删掉，再用 LightGBM 和 XGBoost 各跑五折，最后对半混合填进官方模板。** 我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步走。

### 2.1 数据探索（EDA）

我们首先把两个文件读进来，并把标签二值化。注意这里用的是 `== 'DELIVERY'` 而不是直接拿 `action_type` 的原始类别去训练——因为比赛只关心「是不是配送」这一个二分类问题：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

train = pd.read_csv('/kaggle/input/competitions/cuhkml-2026-classification/dataframe_train.csv')
test  = pd.read_csv('/kaggle/input/competitions/cuhkml-2026-classification/dataframe_test.csv')

target_col = 'action_type'
id_col = 'id'

# 比赛要的是二分类：是不是 DELIVERY。是就记 1，其余全部记 0
y = (train[target_col].astype(str) == 'DELIVERY').astype(int).values

# 注意这里多删了一列 expected_use_time：
# 它是同门回归赛的标签，留在特征里就是标签泄漏，必须删掉
X = train.drop(columns=[target_col, id_col, 'expected_use_time'], errors='ignore')
X_test = test.drop(columns=[id_col], errors='ignore')
```

接着我们摸一摸数据的底：有多少行、标签是否均衡、哪些列混杂了类型、有没有缺失值。

```python
# 行数、列数：确认训练集和测试集规模
print('train:', train.shape, 'test:', test.shape)

# 标签分布：如果 DELIVERY 占比很低，就是一个不平衡问题
print(y.mean())                  # DELIVERY 在训练集里的比例
print(pd.Series(y).value_counts())

# 每一列的类型和缺失情况，找出需要特殊处理的列
info = pd.DataFrame({
    'dtype': X.dtypes.astype(str),
    'missing': X.isna().sum(),
    'n_unique': X.nunique(dropna=False),
})
print(info.sort_values('missing', ascending=False).head(15))

# 训练集和测试集各自有哪些列，差集就是没对齐的部分
print('only in train:', set(X.columns) - set(X_test.columns))
print('only in test :', set(X_test.columns) - set(X.columns))
```

这一步的目的很明确：**先找出哪些列是字符串、哪些列训练集和测试集的类型不一样、哪些列缺值多**。表格比赛里，模型调参带来的提升往往还不如把这几件事处理对。

### 2.2 数据清洗

这是整场比赛最花心思的一节。我们做了三件事：列对齐、类型统一、缺失填充。

**第一件，列对齐。** 只保留两边都有的列，避免训练时特征维度对不上：

```python
# 只保留训练集和测试集都存在的列，顺序也按训练集来
common_cols = [col for col in X.columns if col in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]
```

**第二件，类型统一。** 这一步是材料里的关键修复。同一列可能「训练集是数字、测试集混了字符串」，所以判断条件必须**两边都检查**，只要有一边是字符串，就整列按类别处理：

```python
for col in common_cols:
    # 关键：训练集和测试集都要检查，只查一边会漏
    is_train_str = X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col])
    is_test_str  = X_test[col].dtype == 'object' or pd.api.types.is_string_dtype(X_test[col])

    if is_train_str or is_test_str:
        # 先统一转成字符串，这样 NaN 和数字也能混在一起处理
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)

        # 把训练集和测试集拼起来一起 fit，
        # 保证同一个类别在两边拿到同一个整数编码
        le = LabelEncoder()
        le.fit(pd.concat([X[col], X_test[col]]))
        X[col] = le.transform(X[col])
        X_test[col] = le.transform(X_test[col])
    else:
        # 数值列：用训练集的中位数填缺失（中位数比均值抗异常值）
        median_val = X[col].median()
        if pd.isna(median_val):      # 整列都是空的极端情况
            median_val = 0
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
```

关于「拼起来一起 fit」这一点要说明一下：这在 Kaggle 里很常见，好处是训练集和测试集的类别编码绝对一致，坏处是用了测试集的分布信息。如果追求严格的线上泛化评估，应该只在训练集上 fit，对测试集里没见过的类别统一映射到一个「未知」编号。这里作者选了前一种，我们如实记录。

**第三件，彻底变成纯数值数组。** 树模型对 `object` 类型很敏感，最后统一转成 `float32`，既消除类型残留，又省一半内存：

```python
# 全部转成 float32 的纯数值矩阵，确保没有任何 object 类型残留
X_arr = X.values.astype(np.float32)
X_test_arr = X_test.values.astype(np.float32)
print('aligned:', X_arr.shape, X_test_arr.shape)
```

### 2.3 模型训练与超参数调优

模型部分我们用了两个在表格比赛里最稳的选手：LightGBM 和 XGBoost。两者都是「梯度提升树」——用很多棵小决策树一步步纠正前面树的错误，对表格数据几乎不用做特征工程就能有好效果。

**验证方式**是 5 折交叉验证（KFold，打乱，随机种子 42）。每一折拿 80% 训练、20% 验证，五折轮流，最后每个样本都恰好被预测过一次，这批预测叫 OOF（out-of-fold），可以拿来算一个比较可信的本地分数。

**第一个模型：LightGBM。** 关键参数是学习率 0.05、叶子数 31、特征采样比例 0.8，树的数量最多 1500 但用早停控制：

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb   = np.zeros(len(train))     # 存每折验证集的预测，拼成完整的 OOF
preds_lgb = np.zeros(len(test))      # 存测试集预测，五折累加后取平均

lgb_params = {
    'objective': 'binary',            # 二分类
    'metric': 'binary_logloss',       # 和比赛指标一致
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,            # 小学习率 + 多树，通常更稳
    'num_leaves': 31,
    'max_depth': -1,                  # 不限制深度，交给叶子数控制复杂度
    'feature_fraction': 0.8,          # 每棵树只用 80% 的特征，防过拟合
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_arr, y)):
    model_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=1500)
    # 早停：验证集损失连续 50 轮不降就停，避免白跑和过拟合
    model_lgb.fit(X_arr[train_idx], y[train_idx],
                  eval_set=[(X_arr[val_idx], y[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    oof_lgb[val_idx] = model_lgb.predict_proba(X_arr[val_idx])[:, 1]
    # 测试集预测按折数平均，五折各贡献 1/5
    preds_lgb += model_lgb.predict_proba(X_test_arr)[:, 1] / kf.n_splits
    print(f'Fold {fold+1} LGBM LogLoss: {log_loss(y[val_idx], oof_lgb[val_idx]):.4f}')
```

**第二个模型：XGBoost。** 结构类似，但参数风格不同：最大深度限制在 6，行采样和列采样都是 0.8，用 `hist` 算法加速：

```python
oof_xgb   = np.zeros(len(train))
preds_xgb = np.zeros(len(test))

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.05,
    'max_depth': 6,                   # 和 LightGBM 的叶子数对应，控制树的复杂度
    'subsample': 0.8,                 # 每棵树只抽 80% 的样本
    'colsample_bytree': 0.8,          # 每棵树只抽 80% 的特征
    'random_state': 42,
    'tree_method': 'hist',            # 直方图算法，大数据上更快
    'n_jobs': -1,
    'early_stopping_rounds': 50,      # 早停必须写在构造参数里
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_arr, y)):
    model_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=1500)
    model_xgb.fit(X_arr[train_idx], y[train_idx],
                  eval_set=[(X_arr[val_idx], y[val_idx])], verbose=False)

    oof_xgb[val_idx] = model_xgb.predict_proba(X_arr[val_idx])[:, 1]
    preds_xgb += model_xgb.predict_proba(X_test_arr)[:, 1] / kf.n_splits
    print(f'Fold {fold+1} XGB LogLoss: {log_loss(y[val_idx], oof_xgb[val_idx]):.4f}')
```

材料里专门标注了一个版本兼容的坑：**XGBoost 新版本的早停参数必须写在构造函数里**，写在 `fit()` 里会报错。另外两个模型的折划分用的是同一个 `kf`，所以两边的 OOF 是按同一套索引拼出来的，可以直接对比、直接混合。

**关于超参数调优。** 材料里没有出现网格搜索或随机搜索，参数是直接给定的（学习率 0.05、`num_leaves` 31、`max_depth` 6、采样 0.8），靠早停自动决定树的数量。这种「一套合理默认值 + 早停」的做法在表格比赛里其实很常见：参数对树模型的最终效果影响远小于数据质量和特征，与其花几小时搜索，不如先把 2.2 节的对齐做扎实。

### 2.4 模型测试与指标报告

先看混合的效果。两个模型各占一半，在 OOF 上算整体 log loss：

```python
# 0.5 / 0.5 的加权平均，是最简单也最不容易翻车的融合方式
cv_final = log_loss(y, oof_lgb * 0.5 + oof_xgb * 0.5)
print(f'FINAL ENSEMBLE LOGLOSS: {cv_final:.4f}')
```

材料里这三行分数（单折 LGBM、单折 XGB、整体混合）都是运行时格式化打印的，抽取出来的文本没有保留具体数值，所以我们这里只报告**评估方法和配置**，不编造分数。从流程上说，我们看的顺序是：先看每一折的 log loss 是否稳定（波动大说明数据划分不稳），再看两个模型单独的 OOF 分数，最后看混合后有没有比单模型更好。如果混合没有提升，说明两个模型太像了，这时候加模型种类比加模型数量更有用。

接着我们看特征重要性，确认模型在依赖哪些列：

```python
# 取最后一折训练好的模型，看它给每个特征打了多少分
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

如果排在前面的全是某个 ID 类的列，就要警惕过拟合；如果 `expected_use_time` 出现在榜单里，那说明 2.1 节的删除没生效——这是我们特意检查的一点。

**最后是提交。** 这一步是材料里作者调试最久的地方。官方模板要求 `action_type_DELIVERY` 是**严格的 0/1 整数**，`order` 列也必须是纯整数，否则校验器会报「混合类型」错误：

```python
# 直接读官方模板，在它上面原地填，保证行序和列名完全正确
example_sub = pd.read_csv(
    '/kaggle/input/competitions/cuhkml-2026-classification/classification_example.csv')

# 两个模型的测试集概率混合后，按 0.5 为界切成 0/1
final_preds = preds_lgb * 0.5 + preds_xgb * 0.5
binary_labels = np.round(final_preds).astype(int)

example_sub['action_type_DELIVERY'] = binary_labels

# order 列如果混了文字，先转成数字，再用前后值补齐空洞，最后强转整数
example_sub['order'] = pd.to_numeric(example_sub['order'], errors='coerce')
example_sub['order'] = example_sub['order'].bfill().ffill().astype(int)

# 固定列顺序，去掉 pandas 的索引列再写盘
example_sub = example_sub[['order', 'action_type_DELIVERY']]
example_sub.to_csv('submission.csv', index=False)
```

写完我们还会打印一下 `np.unique(example_sub['action_type_DELIVERY'])`，确认里面真的只有 0 和 1，没有残留的小数。

**这里必须补一句诚实的评价。** 作者在前面用 log loss 评估模型，最后却把概率四舍五入成了 0/1。这两件事是矛盾的：log loss 奖励的是**校准良好的概率**，把 0.51 和 0.99 都变成 1，等于把模型最有价值的信息丢掉了。如果线上指标确实是 log loss，这样提交会把分数拉低不少；只有当校验器真的只接受硬标签时，四舍五入才是必要的妥协。材料没有说明线上指标的确切定义，所以我们把这个矛盾如实指出来，而不假装它是一个「精妙技巧」。

---

## 三、总结

本文中我们完成了 CUHKML 2026 分类赛的完整流程。我们首先用 EDA 摸清了数据结构：`action_type` 二值化成「是否 DELIVERY」，并因为同门回归赛的标签是 `expected_use_time` 而把它从特征里删掉，避免标签泄漏。其次我们做了三件清洗：只保留两边共有的列做对齐，对「任一边是字符串」的列统一转字符串后在训练集加测试集上一起 fit `LabelEncoder`，数值列用训练集中位数填缺失，最后全部转成 `float32` 纯数值矩阵。接着我们用 5 折交叉验证分别训练 LightGBM（学习率 0.05、`num_leaves` 31、特征采样 0.8、早停 50 轮）和 XGBoost（学习率 0.05、深度 6、行列采样 0.8、`hist` 加速、早停 50 轮），并把早停参数按新版本要求写进构造函数。然后我们在 OOF 上评估两个单模型并做 0.5/0.5 混合，用特征重要性检查有没有可疑的泄漏列。最后我们把混合概率填进官方模板，把 `order` 强制转成整数、把标签切成 0/1，写出提交文件。

这个比赛给我们的最大启发有两个。第一，**表格比赛的大头在数据对齐，不在调参**：列不一致、类型混杂、标签泄漏，这三件事任何一件没处理好，再好的模型也白搭。第二，**评估指标和提交格式要说同一种语言**：用 log loss 调模型却交硬标签，等于把辛苦校准的概率扔了；遇到这种情况，应该先搞清楚校验器到底要什么，再决定怎么提交。
