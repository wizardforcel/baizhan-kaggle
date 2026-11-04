# Allstate 理赔金额预测（Claims Severity）

## 一、比赛简介

### 1.1 比赛背景

Allstate 是美国一家大型保险公司，卖车险、家财险等。客户出了事故来报案，公司要赔多少钱，事先得有个估计：估得准，公司才能合理定价、留够准备金；估得离谱，要么亏钱，要么收多了保费把客户吓跑。过去这件事靠理赔员凭经验判断，现在 Allstate 想用机器学习来做。

这个比赛给出的是一批**匿名化**的保单数据：客户的性别、婚姻状况、车型、投保项目等都被换成了 `cat1`、`cat2`……这样的代号，金额相关的连续变量换成了 `cont1`……`cont14`。我们要根据这些特征，预测这张保单最终的**理赔金额 `loss`**。

之所以要把数据匿名化，是因为保险数据涉及个人隐私，不能直接公开。代价是我们不知道每一列到底是什么含义，只能纯粹靠数据本身的统计规律去建模——这也让比赛变成了一场「特征工程 + 模型调参」的硬仗。

### 1.2 比赛目标

给定一张保单的 130 个特征，预测它的理赔金额 `loss`（一个连续的美元数值）。

- 训练集：**188318 行、131 列**（130 个特征 + 1 个目标 `loss`）；
- 测试集：**125546 行、130 列**（没有 `loss`）；
- 特征构成：116 个类别特征 `cat1`～`cat116`（取值是字母/代号）+ 14 个连续特征 `cont1`～`cont14`；
- 目标 `loss`：理赔金额，从个位数到几万美元都有，分布**极度右偏**——绝大多数理赔是小额，少数大额把平均值拉得很高。

数据本身很干净：材料里检查后确认**没有缺失值**，所以清洗的重点不在「补空」，而在「处理异常值」和「处理偏态」。

### 1.3 评价标准

比赛使用 **MAE（平均绝对误差，Mean Absolute Error）**，公式是：

```
MAE = (1/n) · Σ |真实值 - 预测值|
```

也就是把每张保单的预测误差取绝对值，再求平均。**MAE 越低越好**，单位是美元。比如 MAE = 1100，意思是平均每张保单我们猜错 1100 美元。

这个指标有两个特点会直接影响我们的建模思路。第一，它对**离群值不敏感**——不像均方误差（MSE）会把大误差平方放大，MAE 只看绝对差距，所以偶尔猜错一个天文数字不会毁掉整个分数。第二，正因为 MAE 关注的是「中位数附近的表现」，模型会更倾向于预测「中间值」，而我们后面要做的 **log 变换** 和 **fair 损失** 都是在对付这件事。

> 小提示：比赛里大家常说的「1100 分」就是指 MAE 约 1100 美元。这个比赛的分数集中在 1100~1130 之间，每往下压 1 美元都很难。

---

## 二、解题过程

我们的主线是：**先用对数变换把右偏的金额「拉正」，再把 116 个类别特征编码、并手工组合出更有信息量的交叉特征，最后用 XGBoost 配合 fair 损失和 10 折交叉验证来训练。** 下面分四步讲。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，看看规模、类型和缺失：

```python
import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

print(train.shape)   # (188318, 131)
print(test.shape)    # (125546, 130)

# 看看各列的类型：cat* 是 object，cont* 是 float
print(train.dtypes.value_counts())

# 检查缺失值：这个比赛数据很干净，应该都是 0
print(train.isnull().sum().sum())
```

结果确认：训练集 188318×131，测试集 125546×130，**没有缺失值**。既然不缺数据，我们的注意力就转向两件事：目标分布长什么样、有没有极端值。

```python
# 看目标 loss 的统计量
print(train['loss'].describe())
# 平均值远大于中位数，说明少数大额理赔把分布拉长了（右偏）
print('偏度:', train['loss'].skew())
```

`loss` 的分布是一条长长的右尾巴：大量保单在几百到几千美元，少数能到几万。这种分布直接喂给模型会有问题——模型会花大量精力去拟合那几个大额样本，反而在占多数的小额样本上不准。所以我们先做一次可视化，确认「取对数之后是不是接近钟形」：

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# 原始 loss：右边拖得很长
axes[0].hist(train['loss'], bins=100)
axes[0].set_title('原始 loss 分布')

# 取 log 之后：形状接近对称的钟形，适合回归
axes[1].hist(np.log(train['loss']), bins=100)
axes[1].set_title('log(loss) 分布')
plt.show()
```

右边那张图明显更「正态」，这就是我们后面要对 `loss` 做对数变换的直接依据。另外我们还顺便统计了一下极端值：`loss` 大于 40000 的样本只有**个位数**，属于录入异常或极端个例，训练时可以考虑剔除。

### 2.2 数据清洗

清洗分三块：内存压缩、异常值处理、类别编码。

第一块是**内存压缩**。188318×131 的表格用默认的 64 位浮点数存，光训练集就要 **189.65 MB**，再加上后续做特征组合会翻好几倍，普通笔记本很容易爆内存。我们把每一列「降级」到能装下它取值范围的最小类型：

```python
def reduce_mem_usage(df):
    """把数值列转成更省内存的类型，数值不变但体积大幅下降"""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            # 整数列：按范围选 int8 / int16 / int32
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                else:
                    df[col] = df[col].astype(np.int32)
            else:
                # 浮点列统一用 float32，精度足够、体积减半
                df[col] = df[col].astype(np.float32)
        else:
            # 文本列：类别少就用 category 类型，内部只存编号
            df[col] = df[col].astype('category')
    return df

train = reduce_mem_usage(train)
# 内存从 189.65 MB 降到 27.70 MB
```

这一步把内存从 **189.65 MB** 压到 **27.70 MB**，等于同样的内存能多装六七倍的数据，对后面反复试验帮助很大。

第二块是**目标变换和异常值**。我们把极端大额样本剔掉，再对 `loss` 取对数。这里用的是 `log(loss + 200)`——加 200 是为了避免 `loss` 很小时 `log` 变成绝对值很大的负数（`log(0)` 是负无穷）：

```python
# 只保留 loss <= 40000 的样本：更大的只有个位数，属于异常
train = train[train['loss'] <= 40000].reset_index(drop=True)

# 对数变换：把右偏分布拉成近似正态；+200 防止小值取 log 后过负
y = np.log(train['loss'] + 200)

# 预测时记得反变换回来：exp(预测值) - 200 才是美元金额
# pred_loss = np.exp(pred_log) - 200
```

第三块是**类别编码**。116 个类别列是 `A`、`B`、`C`……这样的字母，模型只认数字，所以我们用 `LabelEncoder` 把它们统一映射成 0、1、2……。注意：这样映射出来的数字**没有大小含义**（`B` 不一定比 `A` 大），对树模型来说这没关系——树模型只看「是不是等于某个值」，不看数值大小，所以 LabelEncoder 对 XGBoost、LightGBM 这类模型是安全的：

```python
from sklearn.preprocessing import LabelEncoder

# 找出所有类别列：第 0 列是 id，第 1~116 列是 cat1~cat116
cat_cols = ['cat' + str(i) for i in range(1, 117)]

for col in cat_cols:
    le = LabelEncoder()
    # 训练集和测试集一起 fit，保证同一个字母映射成同一个数字
    le.fit(pd.concat([train[col], test[col]]).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
```

### 2.3 模型训练与超参数调优

这个比赛真正的差距在**特征组合**。116 个类别特征单看信息量有限，但两两交叉往往能挖出强信号——比如 `cat80` 和 `cat87` 单独看都平平，组合起来却能把某类保单区分得很干净。我们的做法是把若干对「黄金组合」拼成新列，再用 26 进制把它们编码成一个整数：

```python
# 材料里验证过有增益的若干特征对
COMB_FEATURE = [
    'cat80,cat87', 'cat57,cat12', 'cat80,cat32,cat69,cat13',
    'cat79,cat10', 'cat18,cat42', 'cat80,cat32', 'cat80,cat87,cat13',
    'cat80,cat32,cat69', 'cat80,cat32,cat13', 'cat80,cat69',
    # ...（共 35 组）
]

for comb in COMB_FEATURE:
    cols = comb.split(',')
    # 用 '_' 把几列拼成一个字符串，例如 'A_B_C'
    new_col = '_'.join(cols)
    for df in (train, test):
        df[new_col] = df[cols[0]].astype(str)
        for c in cols[1:]:
            df[new_col] = df[new_col] + '_' + df[c].astype(str)
    # 把组合字符串当成新类别，再 LabelEncoder 成整数
    le = LabelEncoder()
    le.fit(pd.concat([train[new_col], test[new_col]]))
    train[new_col] = le.transform(train[new_col])
    test[new_col]  = le.transform(test[new_col])
```

**损失函数是另一个关键。** 标准的平方损失会让模型被少数大额样本牵着走，而比赛评的是 MAE。材料里用了一个专门针对 MAE 的 **fair 损失**（也叫 fair objective），它对误差的增长比平方损失温和，对大误差不那么敏感，跟 MAE 的口味更一致：

```python
fair_constant = 2.0

def fair_obj(preds, dtrain):
    """Fair loss 的一阶、二阶导数，供 XGBoost 自定义目标使用"""
    labels = dtrain.get_label()
    x = preds - labels
    # 一阶导（梯度）
    grad = fair_constant * x / (abs(x) * (abs(x) + fair_constant))
    # 二阶导（海森矩阵）
    hess = fair_constant ** 2 / (abs(x) + fair_constant) ** 2
    return grad, hess

def xg_eval_mae(preds, dtrain):
    """自定义评估函数：直接算 MAE，训练时能看到跟比赛一致的指标"""
    labels = dtrain.get_label()
    return 'mae', np.mean(np.abs(preds - labels))
```

训练用 **10 折交叉验证**：把数据分成 10 份，轮流用 9 份训练、1 份验证，最后把 10 个模型的预测平均。这样做的好处是充分利用了每一份数据，而且平均之后方差更小、更稳：

```python
import xgboost as xgb
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=2016)
pred_test = np.zeros(len(test))
cv_scores = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
    dtrain = xgb.DMatrix(train.iloc[tr_idx].drop(['id', 'loss'], axis=1), label=y.iloc[tr_idx])
    dvalid = xgb.DMatrix(train.iloc[va_idx].drop(['id', 'loss'], axis=1), label=y.iloc[va_idx])

    params = {
        'eta': 0.01,                 # 学习率小一点，配合多轮迭代更稳
        'max_depth': 12,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 100,
        'silent': 1,
        'objective': fair_obj,       # 用 fair 损失代替默认的平方损失
        'eval_metric': xg_eval_mae,  # 每轮都看 MAE
    }
    model = xgb.train(params, dtrain, num_boost_round=2000,
                      evals=[(dvalid, 'valid')], early_stopping_rounds=100,
                      verbose_eval=False)

    # 验证集预测 → 反 log 变换 → 算 MAE
    va_pred = model.predict(xgb.DMatrix(train.iloc[va_idx].drop(['id', 'loss'], axis=1)),
                            ntree_limit=model.best_ntree_limit)
    va_pred = np.exp(va_pred) - 200
    cv_scores.append(np.mean(np.abs(va_pred - train['loss'].iloc[va_idx])))
    print('fold', fold, 'MAE:', cv_scores[-1])

    # 测试集累加，10 折最后取平均
    pred_test += model.predict(xgb.DMatrix(test.drop(['id'], axis=1)),
                               ntree_limit=model.best_ntree_limit)

pred_test /= 10
pred_test = np.exp(pred_test) - 200
print('CV MAE:', np.mean(cv_scores))
```

关于调参，我们试过用 `GridSearchCV` 在 `max_depth`、`min_child_weight`、`colsample_bytree` 上搜索，但 10 折 × 多组参数的成本太高。实际比较有效的是：**`eta` 压到 0.01、搭配 `early_stopping_rounds` 自动决定树的数量**，这样既不容易过拟合，也不用猜「训多少轮」。`min_child_weight` 设大（100）是为了防止树在极少数样本上长出「专记特例」的叶子。

### 2.4 模型测试与指标报告

最后把 10 折的测试预测取平均，按提交格式写出：

```python
# pred_test 已经是平均后、反 log 变换回美元的值
sub = pd.DataFrame({'id': test['id'], 'loss': pred_test})
sub.to_csv('submission.csv', index=False)
print(sub.head())
```

**结果报告。** 我们的主线方案（特征组合 + fair 损失 + 10 折 XGB）在公榜上拿到 **1108.72665**。作为对照，材料里还有几条路：

| 方案 | 公榜 MAE |
| --- | --- |
| XGBoost 入门版（log(y+200)、默认平方损失、无特征组合） | 1128.9 |
| CatBoost（默认参数、只做基础编码） | 1122 |
| XGB + LightGBM + CatBoost 三模型平均 | 1113.12994 |
| 特征组合 + fair 损失 + 10 折 XGB（我们的主线） | **1108.72665** |

从表里能看清提升的来源：入门版 1128.9 → 加了特征组合和 fair 损失降到 1108.7，一共降了 **20 美元**左右；而单纯「堆三个模型做平均」只能到 1113.1，还不如把特征做扎实。这也印证了我们一开始的判断：**这个比赛的胜负手在特征，不在模型数量**。

---

## 三、总结

本文中我们完成了 Allstate 理赔金额预测的完整流程。我们首先用 EDA 摸清了数据：训练集 188318×131、测试集 125546×130、没有缺失值，但目标 `loss` 严重右偏，需要取对数才能让模型学得动。其次我们做了三件清洗工作：用类型降级把内存从 189.65 MB 压到 27.70 MB、剔除 `loss > 40000` 的个位数异常样本、对 116 个类别列做 LabelEncoder 编码。接着我们围绕「特征组合」下功夫，手工拼出 35 组类别交叉特征，并用 fair 损失配合 10 折交叉验证训练 XGBoost，学习率压到 0.01 靠早停决定轮数。最后我们反 log 变换生成提交，公榜 MAE 达到 **1108.72665**，比入门版的 1128.9 改善了约 20 美元。

这个比赛给我们的最大启发是：**面对匿名化的表格数据，特征工程比模型选择更重要。** 116 个看不出含义的类别列，靠两两交叉就能挖出强信号；而堆三个模型做平均的收益（1113.1）远不如把特征做对（1108.7）。另外，**评价指标决定损失函数**——比赛评 MAE，我们就用对误差更宽容的 fair 损失，而不是默认的平方损失，这一步本身就值好几分。