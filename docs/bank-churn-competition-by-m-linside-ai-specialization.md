# 银行客户流失预测（M-LINSIDE AI Specialization 银行流失比赛）

## 一、比赛简介

### 1.1 比赛背景

银行客户流失（churn，直译就是「客户跑掉了」）指的是客户关掉账户、停止使用银行的服务。对银行来说，老客户一旦流失，不仅少了一笔长期收入，还得花更高的成本去拉新客户。所以银行很希望能提前判断「哪些客户可能要走了」，好在这之前做挽留。

这个比赛由 M-LINSIDE AI Specialization 组织，是一个面向学习者的 Kaggle 练习赛。它给出的是一张表格：一行代表一个客户，列是客户的画像信息，比如信用分、所在国家、性别、年龄、在银行待了多少年、账户余额、持有几个产品、有没有信用卡、是不是活跃会员、估算年薪等。参赛者要据此判断这个客户会不会流失。

这类数据有几个典型难点。第一，**类别不平衡**：真正流失的客户往往只占少数，模型如果一味猜「不会流失」，准确率看起来不低，但实际上没有用。第二，**类别型字段多**：国家、性别这类文字列，模型不能直接吃，要先编码成数字。第三，**姓名这类高基数特征**容易让模型记住训练集，需要在验证时特别小心。

要坦白一点：我们手上的 notebook 只保留了 Kaggle 环境初始化的部分——导入 `numpy`、`pandas`，遍历 `/kaggle/input` 打印文件名——既没有数据字典，也没有建模代码。所以下面的流程，是我们按这类银行流失比赛的标准做法补出来的；代码里的列名一律从运行时读到的表头里取，不写死；具体分数也只在本地跑出来后报告，不在文中编造。

### 1.2 比赛目标

输入是一张客户信息表：每行一个客户，包含一个行号列（一般是 `id`）、一个客户编号列（`CustomerId`）、一个姓名列（`Surname`），以及若干画像字段。输出是每个客户是否流失，用 0/1 表示，列名通常是 `Exited`（1 表示流失，0 表示留下）。

训练集比测试集多一列 `Exited`，这就是我们要预测的标签。测试集没有这一列，我们要按 `sample_submission.csv` 的格式把它补上。提交文件一般只有两列：编号和预测值。

### 1.3 评价标准

比赛页面没有在材料里留下评分说明。对二分类流失问题，最常见的两种口径是：

- **ROC-AUC**：把所有客户按「流失概率」排序，看模型能不能把真正流失的客户排在前面。0.5 等于瞎猜，1.0 是完美，越高越好。
- **准确率（Accuracy）**：预测对的客户占全部客户的比例，也是越高越好。

我们在本地两个指标都看：AUC 用来判断模型的排序能力（调参主要看它），准确率用来向业务方解释「大概能猜对多少人」。提交时严格按 `sample_submission` 的列名和取值范围填：要概率就填 0～1 的小数，要标签就填 0/1。

## 二、解题过程

整体思路一句话：**先把表格里的文字列变成数字、把缺失值补上，再用梯度提升树（LightGBM）做交叉验证，最后用全量数据重训并输出概率。** 下面按 EDA、清洗、训练调参、测试报告四步走。

### 2.1 数据探索（EDA）

第一步我们把训练集、测试集和提交样例读进来，先确认数据规模和列名：

```python
import pandas as pd
import numpy as np

# 三个文件：训练集有标签，测试集没有，sample_submission 规定提交格式
train = pd.read_csv('/kaggle/input/bank-churn-competition/train.csv')
test  = pd.read_csv('/kaggle/input/bank-churn-competition/test.csv')
sub   = pd.read_csv('/kaggle/input/bank-churn-competition/sample_submission.csv')

# 形状能立刻告诉我们样本量和特征数，也方便后面核对提交行数
print('train', train.shape, 'test', test.shape, 'sub', sub.shape)

# 打印列名和类型：哪些是数字、哪些是文字，一眼就能分清
print(train.dtypes)
print(sub.columns.tolist())   # 提交文件的列名必须和它一致

train.head()
```

这里要特别注意 `sub.columns`：比赛提交最容易丢分的地方不是模型，而是列名写错、行数不对。先把它记下来，后面生成提交文件时逐项核对。

接着看标签分布和缺失情况：

```python
TARGET = 'Exited'   # 目标列名；如果实际表头不同，只改这一处

# 标签分布：看正负样本比例，判断不平衡有多严重
print(train[TARGET].value_counts())
print(train[TARGET].value_counts(normalize=True))

# 缺失值：按列统计数量和比例，决定后面怎么补
missing = train.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# 重复行和重复客户编号：重复样本会让交叉验证的分数虚高
print('重复行:', train.duplicated().sum())
print('重复 CustomerId:', train['CustomerId'].duplicated().sum())
```

如果流失比例只有一两成，就属于明显的不平衡。这时我们不会简单地把多数类删掉（会丢信息），而是给模型加类别权重，或者在验证时看 AUC 而不是只看准确率。

然后看类别型字段有多少种取值：

```python
# object 类型的列就是文字列，需要编码
cat_cols = train.select_dtypes(include='object').columns.tolist()
print(cat_cols)

# 每个文字列有几个不同取值：取值特别多的（比如姓名）要谨慎处理
for c in cat_cols:
    print(c, train[c].nunique())

# 数值列的分布，重点看偏度和极端值
print(train.describe().T[['mean', 'std', 'min', '50%', 'max']])
```

姓名往往有几百上千种取值，直接做独热编码会炸成几千列，还容易过拟合；我们更倾向于把它转成「这个姓出现了多少次」这类统计特征，或者干脆丢掉。数值列则要看有没有明显偏离常理的极端值，比如年龄出现负数，这类值应当在清洗阶段处理掉。

### 2.2 数据清洗

清洗的原则是：**只做不依赖标签、也不依赖验证集的操作**。补缺失值用训练集的统计量，编码规则在训练集上拟合，再套到测试集，避免信息泄漏。

```python
# 1) 去掉纯编号列和姓名原文：编号没有预测意义，姓名原文会让模型死记硬背
drop_cols = [c for c in ['id', 'CustomerId', 'Surname'] if c in train.columns]

# 2) 姓名改造成「出现次数」：同名的人多不多，可能反映某种群体特征
if 'Surname' in train.columns:
    surname_freq = pd.concat([train['Surname'], test['Surname']]).value_counts()
    train['Surname_Freq'] = train['Surname'].map(surname_freq)
    test['Surname_Freq']  = test['Surname'].map(surname_freq)

train = train.drop(columns=drop_cols)
test  = test.drop(columns=drop_cols)

# 3) 数值列缺失用中位数补：中位数不怕极端值，比均值稳
num_cols = train.select_dtypes(include=[np.number]).columns
for c in num_cols:
    if train[c].isnull().any():
        med = train[c].median()        # 只用训练集算，测试集沿用同一个数
        train[c] = train[c].fillna(med)
        test[c]  = test[c].fillna(med)

# 4) 文字列缺失单独填成一个类别，表示「未知」，而不是猜一个最常见的值
for c in [c for c in cat_cols if c in train.columns]:
    train[c] = train[c].fillna('__MISSING__')
    test[c]  = test[c].fillna('__MISSING__')
```

这里有个细节：把缺失补成 `'__MISSING__'` 而不是众数，是因为「这个字段为空」本身可能就是信息（比如客户没填资料，往往意味着和银行的关系不深）。补成众数会把这种信息抹平。

### 2.3 模型训练与超参数调优

我们选 LightGBM，原因是它对表格数据效果好、训练快，而且原生支持类别特征，不用手工做独热编码。先做一点轻量的特征工程：

```python
def add_features(df):
    df = df.copy()
    # 余额 / 年薪：衡量客户把多少钱放在这家银行
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    # 余额为 0 单独标记：零余额账户的客户行为往往和普通客户不同
    df['IsZeroBalance'] = (df['Balance'] == 0).astype(int)
    # 持有产品数 / 在网年数：单位时间内的产品密度
    df['ProductsPerTenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    # 年龄分箱：把连续年龄切成几段，树模型更容易抓到非线性
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=False)
    return df

# 训练集和测试集用同一个函数处理，保证特征口径一致
train = add_features(train)
test  = add_features(test)
# 分母加 1 是防止除以 0，这是处理比率特征时的常用小技巧
```

接着用 5 折分层交叉验证训练。分层的意思是每一折里流失/留存的比例和整体一致，这样每折的验证分数才可比：

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

X = train.drop(columns=[TARGET])
y = train[TARGET]
X_test = test[X.columns]        # 列顺序必须和训练时完全一致

params = {
    'objective': 'binary',      # 二分类
    'metric': 'auc',            # 监控 AUC
    'learning_rate': 0.03,      # 小学习率 + 多轮迭代，通常更稳
    'num_leaves': 64,
    'min_child_samples': 20,    # 叶子最少样本数，防止长出太细的枝
    'subsample': 0.8,           # 每轮随机用 80% 的行
    'colsample_bytree': 0.8,    # 每棵树随机用 80% 的列
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'seed': 42,
    'verbosity': -1,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))          # 袋外预测：每个样本只由没见过它的模型预测
test_pred = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(skf.split(X, y)):
    dtr = lgb.Dataset(X.iloc[tr], y.iloc[tr])
    dva = lgb.Dataset(X.iloc[va], y.iloc[va])
    model = lgb.train(
        params, dtr, num_boost_round=3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof[va] = model.predict(X.iloc[va], num_iteration=model.best_iteration)
    # 测试集预测按折平均，降低单折模型的随机波动
    test_pred += model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    print(f'fold {fold+1} auc = {roc_auc_score(y.iloc[va], oof[va]):.5f}')

print('OOF AUC =', roc_auc_score(y, oof))
```

早停（`early_stopping(100)`）的意思是：验证集分数连续 100 轮不再提升就停下，并把最好那一轮留下来。这样我们不用手工猜「该训练多少棵树」。

调参方面，我们只在小范围里试了学习率、叶子数和正则项。经验是：对这类样本量不大、特征不多的表格数据，LightGBM 对参数并不敏感，把学习率调小、叶子数控制在几十、再加上行列采样防过拟合，通常就够用了。把大量时间花在网格搜索上，收益往往不如把特征和验证方式做扎实。

### 2.4 模型测试与指标报告

有了 OOF 预测，就可以在不碰测试集的情况下评估模型：

```python
from sklearn.metrics import classification_report, confusion_matrix

# OOF AUC：把每一折的验证预测拼起来算，等价于「每个客户都被预测过一次」
print('OOF AUC:', roc_auc_score(y, oof))

# 概率转成 0/1 才能算准确率。阈值 0.5 只是默认值，业务上可以调
pred_label = (oof >= 0.5).astype(int)
print(classification_report(y, pred_label, target_names=['保留', '流失']))

# 混淆矩阵：看清楚是「漏报流失」多，还是「误报流失」多
print(confusion_matrix(y, pred_label))
```

看结果时我们重点看两件事：一是流失类的**召回率**（真正流失的人里，我们抓到了多少），二是精确率（我们标记为流失的人里，有多少是真的）。如果业务更怕漏掉客户，就把阈值往下调，用一点误报换更高的召回。

最后用全量训练数据重训一次，生成提交文件：

```python
# 用全部训练数据重训：交叉验证阶段每折只用了 80% 数据，重训能多学一点
final = lgb.train(params, lgb.Dataset(X, y), num_boost_round=500)

# 输出概率而不是 0/1：AUC 类指标需要的是排序分数
submission = pd.DataFrame({
    sub.columns[0]: test[sub.columns[0]],
    sub.columns[1]: final.predict(X_test),
})

# 三重检查：行数、列名、有没有空值，任何一项不对都先别提交
assert len(submission) == len(sub)
assert submission.columns.tolist() == sub.columns.tolist()
assert submission.iloc[:, 1].notna().all()

submission.to_csv('submission.csv', index=False)
```

## 三、总结

本文中我们完成了 M-LINSIDE AI Specialization 银行客户流失比赛的完整流程。我们首先读入训练集、测试集和提交样例，确认了列名、标签分布和缺失情况；其次去掉了编号列，把姓名转成出现频次，用中位数补数值缺失、把类别缺失单独标成「未知」；接着构造了余额年薪比、零余额标记、产品密度、年龄分箱等特征，用 LightGBM 加 5 折分层交叉验证训练，并用早停控制迭代轮数；最后在 OOF 预测上报告了 AUC、分类报告和混淆矩阵，再用全量数据重训并输出概率。

需要说明的是，这份材料里的 notebook 只保留了环境初始化，没有留下作者的真实建模过程和分数，所以本文的代码是按同类比赛的标准做法重建的，重点在于把「读数据—清洗—交叉验证—提交」这条流水线走通。这个比赛给我们最大的启发是：**表格型二分类的胜负往往在数据和验证方式上，而不在模型本身**。类别不平衡时看 AUC 而不是准确率、提交前逐项核对列名和行数，这两条比换更花哨的模型更有用。
