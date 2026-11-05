# Porto Seguro 安全驾驶预测

## 一、比赛简介

### 1.1 比赛背景
Porto Seguro 是巴西一家大型汽车保险公司。车险业务的核心问题是：哪些投保人下一年会出险并提出理赔？如果能提前识别高风险客户，保险公司就可以更合理地定价，也能把服务资源投向真正需要的人。这个比赛把真实的匿名化保单数据交给参赛者，要求我们建立一个二分类模型。

这份数据有两个显著特点。第一，**正样本极少**。绝大多数车主一年内不会理赔，只有约 3.6% 的样本是正例。第二，**特征全部匿名**。我们看不到“年龄”“车龄”这类直白字段，只能看到 `ps_ind_01`、`ps_reg_03`、`ps_car_13` 这样的代号。代号里的 `ind`、`reg`、`car` 分别暗示个人、地区、车辆三类信息，`calc` 则是平台计算出来的统计量。匿名化保护了隐私，但也让特征工程只能靠数据本身的分布去猜。

另一个难点是缺失值。数据里很多列用 `-1` 表示“没有记录”，而不是标准的空值。如果我们直接把 `-1` 当成普通数字喂给模型，模型会误以为这是一个很小的真实取值，从而学到错误的规律。所以识别并正确处理 `-1` 是这个比赛数据清洗的第一步。

### 1.2 比赛目标
训练集文件 `train.csv` 有 595212 行、58 列，其中 1 列是 `id`，1 列是目标 `target`，其余 56 列是特征。测试集 `test.csv` 有 892816 行、57 列，比训练集少了 `target` 列。我们要为测试集每一行输出一个 0 到 1 之间的概率，表示该车主会理赔的可能性。

特征按前缀可以分成四组：

- `ps_ind_*`：个人相关特征，包括若干二值列和类别列；
- `ps_reg_*`：地区相关特征，主要是连续数值；
- `ps_car_*`：车辆相关特征，既有类别列也有连续列；
- `ps_calc_*`：平台自动计算出的统计特征，数量多、单列信息量弱。

需要特别注意的是，带 `_cat` 后缀的是类别特征，带 `_bin` 后缀的是二值特征，其余多为连续或有序数值。这个后缀约定对后面的编码方式非常关键。

### 1.3 评价标准
比赛使用**标准化 Gini 系数**（Normalized Gini Coefficient）。Gini 系数和 ROC 曲线下的面积 AUC 是等价的，只差一个线性变换：

```
Gini = 2 × AUC - 1
```

AUC 衡量的是“随机抽一个正样本和一个负样本，模型把正样本排在负样本前面的概率”。AUC 为 0.5 相当于瞎猜，为 1.0 相当于完美排序。换算成 Gini 后，瞎猜是 0，完美是 1。排行榜上的分数越高越好。

这个指标只关心**排序**，不关心概率的绝对大小。换句话说，把所有预测概率同时乘以 10，Gini 不变。所以我们在训练时不需要刻意把概率校准到真实发生率，只要把高风险样本排在前面即可。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，确认数据规模和列结构。

```python
# 导入常用库
import pandas as pd
import numpy as np

# 数据目录
data_path = '/kaggle/input/porto-seguro-safe-driver-prediction/'

# 读取训练集、测试集、提交样例，把 id 设为索引方便对齐
train = pd.read_csv(data_path + 'train.csv', index_col='id')
test = pd.read_csv(data_path + 'test.csv', index_col='id')
submission = pd.read_csv(data_path + 'sample_submission.csv', index_col='id')

# 查看形状：训练集 59 万行 58 列，测试集 89 万行 57 列
print(train.shape, test.shape)
```

输出是 `((595212, 58), (892816, 57))`。测试集比训练集还大，这在 Kaggle 里比较少见，说明主办方留了更多数据用于最终评估。

接着我们看目标分布。

```python
# 统计正负样本数量
print(train['target'].value_counts())
# 计算正样本比例
print('positive rate:', train['target'].mean())
```

正样本比例约为 0.036，也就是 3.6%。如果模型把所有样本都预测为 0，准确率能到 96.4%，但这个“准确率”毫无意义。这也解释了为什么比赛不用准确率而用 Gini：在极端不平衡的数据上，准确率会被负样本淹没，排序指标才能区分模型好坏。

然后我们检查缺失值。

```python
# 统计每一列里 -1 的个数
missing_count = (train == -1).sum()
# 只显示确实存在 -1 的列，并按数量排序
print(missing_count[missing_count > 0].sort_values(ascending=False).head(10))
```

可以看到 `ps_reg_03`、`ps_car_03_cat`、`ps_car_05_cat` 等列有大量 `-1`。这些列的缺失比例甚至超过一半。`ps_car_14` 也有不少缺失。后续处理时，我们要么保留 `-1` 让树模型自己分裂，要么把它转成 NaN 再补一个缺失指示列。

我们还统计了类别特征的取值个数。

```python
# 找出所有类别特征
cat_cols = [c for c in train.columns if c.endswith('_cat')]
# 看每个类别特征有多少种取值
for c in cat_cols:
    print(c, train[c].nunique())
```

大部分类别特征只有 2 到 5 种取值，只有 `ps_car_11_cat` 有 100 多种。取值少的适合 One-Hot，取值多的适合目标编码或留给树模型。

最后看特征与目标的关系。我们把二值特征按目标均值画柱状图，发现有些特征的正样本率明显高于整体，例如某些 `ps_ind` 列的差异能达到两三倍。这说明特征是有信号的，不是纯噪声。

### 2.2 数据清洗

我们的清洗原则是：**合并训练集和测试集一起处理，保证编码一致；不做会丢信息的粗暴填充；让树模型自己处理缺失。**

第一步，把训练集和测试集拼起来，并去掉目标列。

```python
# 合并训练集和测试集，ignore_index 让行号重新排列
all_data = pd.concat([train, test], ignore_index=True)
# 去掉目标列，剩下的才是特征
all_data = all_data.drop('target', axis=1)

# 记录训练集行数，后面用它把数据切回去
num_train = train.shape[0]
```

合并处理的好处是：One-Hot 编码器能看到训练集和测试集的全部类别，不会出现“测试集里有一个训练集没见过的类别，编码器直接报错”的情况。

第二步，对类别特征做 One-Hot 编码。

```python
from sklearn.preprocessing import OneHotEncoder

# 选出所有以 _cat 结尾的列
cat_features = [c for c in all_data.columns if c.endswith('_cat')]

# 创建 One-Hot 编码器
onehot = OneHotEncoder()
# 在合并后的数据上拟合并转换，得到稀疏矩阵
encoded_cat = onehot.fit_transform(all_data[cat_features])
print(encoded_cat.shape)
```

One-Hot 的意思是：一个类别列有 3 种取值，就变成 3 列 0/1，样本属于哪一类，哪一列就是 1。它不引入“大小关系”，适合树模型和线性模型。

第三步，挑选数值特征并删掉一部分弱特征。

```python
# 这些特征要么几乎全是同一个值，要么信息量太低，先删掉
drop_features = ['ps_ind_14', 'ps_ind_10_bin', 'ps_ind_11_bin',
                 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_car_14']

# 保留：不是类别、不是 calc、也不在删除名单里的列
remaining_features = [c for c in all_data.columns
                      if (not c.endswith('_cat'))
                      and ('calc' not in c)
                      and (c not in drop_features)]

# 把数值特征转成稀疏矩阵
from scipy import sparse
num_matrix = sparse.csr_matrix(all_data[remaining_features])
```

`calc` 系列是平台算出来的统计量，单独看每一列信息很弱，而且数量多，容易让模型过拟合。在基线方案里我们先整体剔除，后面如果有时间可以再试“只保留其中几列”。

第四步，把 One-Hot 矩阵和数值矩阵横向拼起来，再按原来的行数切回训练集和测试集。

```python
# 横向拼接：数值特征 + One-Hot 类别特征
all_sparse = sparse.hstack([num_matrix, encoded_cat], format='csr')

# 前 num_train 行是训练集，后面是测试集
X = all_sparse[:num_train]
X_test = all_sparse[num_train:]
y = train['target'].values
```

这里 `X` 是稀疏矩阵，因为 One-Hot 之后绝大多数位置是 0，用稀疏存储能省下大量内存。

### 2.3 模型训练与超参数调优

我们选择 LightGBM，原因是它训练快、能直接处理稀疏矩阵、对缺失值友好，而且支持自定义评估函数。为了让本地验证分数和排行榜一致，我们先实现 Gini 计算。

```python
def eval_gini(y_true, y_pred):
    """计算标准化 Gini 系数，输入真实标签和预测概率"""
    # 确保两个数组长度一致
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # 对角线：把样本按数量等分后，每一份的累计比例
    L_mid = np.linspace(1 / n_samples, 1, n_samples)

    # 按预测值从大到小排序真实标签，算预测的洛伦兹曲线
    pred_order = y_true[y_pred.argsort()]
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    G_pred = np.sum(L_mid - L_pred)

    # 按真实值排序，算完美预测的洛伦兹曲线
    true_order = y_true[y_true.argsort()]
    L_true = np.cumsum(true_order) / np.sum(true_order)
    G_true = np.sum(L_mid - L_true)

    # 两者相除得到标准化 Gini
    return G_pred / G_true
```

这个函数看起来绕，其实就一句话：Gini 衡量的是“模型排序曲线”和“完美排序曲线”之间的面积比例。分子是模型得到的面积，分母是完美模型得到的面积，相除就把结果归一化到 0 到 1。

接下来写 LightGBM 的自定义评估接口。

```python
def gini_feval(preds, dtrain):
    """LightGBM 的自定义评估函数，返回 (指标名, 值, 是否越大越好)"""
    labels = dtrain.get_label()
    return 'gini', eval_gini(labels, preds), True
```

第三个返回值 `True` 表示这个指标越大越好，LightGBM 会根据它决定早停方向。

然后我们做 5 折分层交叉验证。分层的意思是：每一折里正负样本比例都和整体一致，避免某一折正样本特别少导致分数抖动。

```python
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

# 5 折，打乱顺序，固定随机种子保证可复现
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1991)

# 训练参数：二分类目标，学习率 0.01，随机种子 0
params = {
    'objective': 'binary',
    'learning_rate': 0.01,
    'force_row_wise': True,
    'random_state': 0
}

# 用来收集每个验证样本的预测值
oof_val_preds = np.zeros(X.shape[0])
# 用来累加测试集的预测值
oof_test_preds = np.zeros(X_test.shape[0])

for idx, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    print('#' * 20, 'fold', idx + 1, '#' * 20)

    # 切出本折的训练集和验证集
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    # 转成 LightGBM 的数据集对象
    dtrain = lgb.Dataset(X_train, y_train)
    dvalid = lgb.Dataset(X_valid, y_valid)

    # 训练：最多 1000 轮，验证集 100 轮不提升就早停
    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=1000,
        valid_sets=dvalid,
        feval=gini_feval,
        early_stopping_rounds=100,
        verbose_eval=100
    )

    # 验证集预测，填回对应位置
    oof_val_preds[valid_idx] = model.predict(X_valid)
    # 测试集预测，每折取平均
    oof_test_preds += model.predict(X_test) / folds.n_splits

    # 打印本折 Gini
    print('fold gini:', eval_gini(y_valid, oof_val_preds[valid_idx]))
```

**关于超参数。** 我们没有做大规模网格搜索，主要基于两点考虑。第一，LightGBM 在几百万行、几百维稀疏特征上，学习率 0.01 配合早停已经能稳定收敛，盲目加大学习率反而容易过拟合。第二，Gini 是排序指标，对树的数量不像 logloss 那么敏感，早停通常停在 500 到 900 轮之间。我们固定 `num_boost_round=1000`、`early_stopping_rounds=100`，让每一折自己决定什么时候停。

五折跑完后，各折的 Gini 大约是 0.289、0.275、0.281、0.271、0.288，波动在 0.02 以内。最后把所有验证集的预测拼起来算整体 Gini：

```python
print('OOF gini:', eval_gini(y, oof_val_preds))
```

输出约为 0.2805。

### 2.4 模型测试与指标报告

我们先看整体 OOF 分数。0.2805 的 Gini 换算成 AUC 是：

```
AUC = (Gini + 1) / 2 = 0.640
```

也就是说，模型在随机抽一对正负样本时，有 64% 的概率把正样本排在前面。相比 50% 的随机水平，这是一个明确的提升。作为参考，比赛里很多公开基线在 0.28 左右，头部方案能到 0.29 以上，差距主要来自特征工程和集成，而不是模型本身。

再看分折结果。第 2 折只有 0.275，明显低于第 1 折的 0.289。我们检查后发现，第 2 折的正样本数量和其它折差不多，差异主要来自样本随机性——在只有 1.7 万个正样本的情况下，单折分数本来就会抖动。这也提醒我们：**不要用单折分数判断模型好坏，要看 OOF 汇总。**

确认模型稳定后，我们用所有训练数据重新训练一次，再对测试集预测。严格来说，前面每折的模型已经对测试集做了平均，可以直接用 `oof_test_preds` 提交；如果想利用全部数据，也可以再训练一个全量模型做加权平均。我们选择直接用五折平均，因为它天然带有集成效果，更稳。

```python
# 把五折平均后的测试集预测填进提交文件
submission['target'] = oof_test_preds

# 保存，index 保留 id
submission.to_csv('submission.csv', index=True)
```

最后我们检查提交文件的行数是否等于测试集行数、列名是否为 `target`、有没有空值。

```python
print(submission.shape)          # 应该是 (892816, 1)
print(submission['target'].isnull().sum())  # 应该是 0
print(submission.head())
```

这一步看起来琐碎，但比赛中因为提交格式错误而白跑一次的情况并不少见。

## 三、总结

本文中我们完成了 Porto Seguro 安全驾驶预测的完整流程。我们首先通过 EDA 确认了数据规模、3.6% 的正样本比例，以及大量以 `-1` 表示的缺失值。其次我们把训练集和测试集合并，对类别特征做 One-Hot 编码，剔除信息量弱的 `calc` 特征，并把数值特征与编码结果拼成稀疏矩阵。接着我们用 LightGBM 做 5 折分层交叉验证，自定义 Gini 评估函数并配合早停，得到 OOF Gini 约 0.2805。最后我们把五折平均的测试集预测写入提交文件，并检查了格式。

这个比赛给我们最大的启发是：**在极度不平衡的表格数据上，评估指标的选择比模型选择更重要。** 如果用准确率，一个全预测 0 的模型就能拿 96 分，但它没有任何业务价值；换成 Gini 之后，模型必须真正学会排序，分数才反映真实能力。另一个启发是缺失值不一定非要填：树模型能够把 `-1` 当成一个单独的取值来分裂，保留它反而保住了“这条记录缺失”这一信息。
