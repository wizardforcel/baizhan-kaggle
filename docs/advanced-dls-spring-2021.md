# 电信客户流失预测（Advanced DLS Spring 2021）

## 一、比赛简介

### 1.1 比赛背景

电信运营商最怕的一件事是**客户流失**（churn）：用了很久的客户突然转到别家。挽留一个老客户的成本，通常远低于拉一个新客户，所以运营商愿意花钱预测「谁可能要跑」，提前给折扣、给合约、给关怀。

这个比赛就是让参赛者根据客户的历史资料，预测他会不会流失。它是 DLS（Deep Learning School，莫斯科物理技术学院开设的深度学习学校）进阶班第 3 模块的 Kaggle 作业，比赛页面名为 `advanced-dls-spring-2021`。材料里收集了上百份学员提交的 notebook，大家的做法集中在两条路线上：**逻辑回归**和 **CatBoost 梯度提升**，少数人再加一层 stacking。

数据是一张很典型的「客户画像表」：每个客户一行，包含在网时长、月消费、总消费、性别、是否老年人、是否有伴侣、是否有孩子，以及电话、网络、在线安全、备份、设备保护、技术支持、在线电视、电影订阅、手机合约、电子账单、付款方式等一系列「有 / 没有」的服务标记。它有两大特点：

- **类别极不平衡。** 流失的客户只占少数（材料里有的页面统计约为四分之一，有的页面按另一套口径算出来约三分之一），模型如果无脑全猜「不流失」，准确率能到七成以上，但一点用都没有。
- **大部分列是「是 / 否」的二值列。** 真正连续取值的只有三列：在网时长、月消费、总消费。这意味着特征工程的空间不大，模型选择和小心的数据清洗更关键。

### 1.2 比赛目标

训练集 `train.csv` 有 5000 行、20 列：`ClientPeriod`、`MonthlySpending`、`TotalSpent` 是三个数值列，其余 16 列是类别列，最后一列 `Churn` 是标签（1 表示流失，0 表示留下）。注意表里**没有单独的 Id 列**，19 个特征加 1 个标签正好 20 列。测试集 `test.csv` 没有 `Churn`，行号从 0 开始。

我们要输出**每个测试客户流失的概率**，提交文件两列：`Id`（就是测试集的行号 0、1、2……）和 `Churn`。注意 `Churn` 这里是 0 到 1 之间的小数，不是 0/1 的硬标签——因为评价指标看的是排序。

### 1.3 评价标准

比赛用 **ROC-AUC**（ROC 曲线下面积）评分，越高越好，0.5 等于瞎猜，1.0 是完美。

用大白话解释：随机抽一个「真的流失了」的客户和一个「真的留下了」的客户，让模型分别给两人的流失概率打分。如果流失那位拿到的分数更高，就算模型赢一次；ROC-AUC 就是「赢的次数占所有配对的比例」。它只关心**排序对不对**，不关心概率的绝对值准不准，所以特别适合类别不平衡的场景——模型不需要把概率校准到「3%」，只要能把高风险的人排在前面就行。

另外，材料里有一份 **adversarial validation**（对抗验证）的笔记：把训练集标成 0、测试集标成 1，训练一个分类器去区分两者。如果 AUC 接近 0.5，说明训练集和测试集长得一样，本地交叉验证的分数可以放心参考；如果 AUC 很高，说明两者分布不同，本地分数就不可信了。这份笔记跑出来的结果**接近 0.5**，所以我们后面可以放心用交叉验证来选模型。

---

## 二、解题过程

我们的主线是：首先做 EDA 摸清三列数值特征和标签的关系，其次修掉 `TotalSpent` 里的脏值和重复行，接着用「标准化 + 独热编码 + 逻辑回归」拿到一个稳定的基线，然后用 CatBoost 利用它原生的类别特征支持做提升，最后把预测概率写成提交文件并对比两边的分数。

### 2.1 数据探索（EDA）

我们首先读数据，确认行列数、每列的类型，并直接看前几行：

```python
import pandas as pd
import numpy as np

df = pd.read_csv('/kaggle/input/advanced-dls-spring-2021/train.csv')
print(df.shape)          # (5000, 20)
print(df.head(8))

# info() 会告诉我们每列的类型和非空数量
# 这里能发现 TotalSpent 是 object（字符串），而不是 float
df.info()
```

`df.info()` 暴露了一个关键问题：`TotalSpent` 明明应该是数字，却被 pandas 读成了字符串。这说明这一列里混进了非数字的内容，如果不处理，后面标准化会直接报错。

接着我们看标签分布和特征相关性。相关性用 `LabelEncoder` 把类别列临时编码成整数再算——注意这只是为了「看」，不是最终的特征处理：

```python
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# 先看流失比例，判断不平衡有多严重
print(df['Churn'].value_counts(normalize=True))

cat_features = ['HasMultiplePhoneNumbers', 'HasInternetService',
                'HasOnlineSecurityService', 'HasOnlineBackup',
                'HasDeviceProtection', 'HasTechSupportAccess',
                'HasOnlineTV', 'HasMovieSubscription',
                'HasContractPhone', 'PaymentMethod']

corr = df.copy()
# LabelEncoder 一次只能编一列，所以要循环
for c in cat_features:
    corr[c] = LabelEncoder().fit_transform(corr[c])

plt.figure(figsize=(14, 12))
sns.heatmap(corr.corr(), annot=True, cmap='coolwarm',
            linewidths=0.5, fmt='.2f')
plt.title('特征相关性热力图')
plt.show()
```

热力图给出的信号非常清楚：

- `HasContractPhone` 与流失的相关系数约为 **-0.4**，是全部特征里最强的；
- `ClientPeriod`（在网时长）约为 **-0.35**，排第二；
- `HasInternetService` 约为 **0.31**，方向是正的；
- `IsBillingPaperless` 约 **0.19**，`MonthlySpending` 约 **0.18**；
- `Sex` 和 `HasPhoneService` 几乎为 0。

读法是：**签了手机合约、在网时间长的客户不容易走；用光纤上网、用电子账单、月消费高的客户反而更容易走**。这符合常识——长期合约把人和运营商绑在一起，而光纤用户对网速更挑剔，遇到不满意更容易换一家。

### 2.2 数据清洗

脏值只有一个来源，但很隐蔽：`TotalSpent` 里有 **9 行**是空格字符串 `" "`，不是空值也不是数字。我们把这 9 行单独打出来看，发现它们的 `ClientPeriod` 都等于 0——也就是**刚开户、还没产生第一笔账单的新客户**。

```python
# 用 to_numeric 强制转换，转不动的会变成 NaN，借此把脏行揪出来
bad = df[pd.to_numeric(df['TotalSpent'], errors='coerce').isna()]
print('坏值行数:', len(bad))
print(bad[['ClientPeriod', 'MonthlySpending', 'TotalSpent']].head(10))

# 空格换成 0 再转浮点：新客户的总消费本来就是 0
df['TotalSpent'] = df['TotalSpent'].replace(' ', 0).astype(float)

# 重复行检查：完全相同的行 28 行，
# 去掉标签后仍相同的行有 41 行——说明有同一客户被记了两次、标签却不同
print('完全重复:', df.duplicated(keep=False).sum())
print('去掉标签后重复:', df.drop(columns='Churn').duplicated(keep=False).sum())

# 只按特征去重，保留第一条。标签冲突的行无论如何都预测不准，留着只会干扰模型
df = df.drop_duplicates(subset=df.columns.difference(['Churn']), keep='first')
```

这里有两个决定值得展开。第一，**空格填 0 而不是删掉**。新客户的总消费确实是 0，这是真实信息，不是缺失；删掉反而丢样本。材料里也有另一种做法：把 `MonthlySpending` 加到 `TotalSpent` 上、同时把 `ClientPeriod` 加 1，理由是「新客户虽然还没结账，但第一个月已经消费了」。两种做法在分数上差别很小，我们选了更直接的填 0。

第二，**按特征去重、不按（特征 + 标签）去重**。完全相同的两行如果标签不同，模型无论怎么学都不可能同时猜对；保留下来只会让训练目标自相矛盾。去掉标签后仍有 41 行重复，说明这类冲突客观存在，这是这个数据集分数上限的一部分。

还有一类「伪类别」要合并。很多服务列有三个取值：`Yes`、`No`、`No internet service`（或 `No phone service`）。后两者本质都是「没有」，只是措辞不同，如果当成三个类别，独热编码会多出两列几乎没人用的特征，还会让树模型多切几次：

```python
patterns = {
    'No': 0, 'No internet service': 0, 'No phone service': 0,
    'Yes': 1,
    'Male': 0, 'Female': 1,
    'DSL': 1, 'Fiber optic': 2,
    'Month-to-month': 0, 'One year': 1, 'Two year': 2,
    'Credit card (automatic)': 0, 'Bank transfer (automatic)': 1,
    'Mailed check': 2, 'Electronic check': 3,
}

# 顺序编码保留了"合约越长 / 付款方式风险越高"的次序信息，
# 比无脑独热更适合树模型；逻辑回归那边仍然会再做一次独热
X_train = df.replace(patterns).drop(columns='Churn')
y_train = df['Churn']
```

### 2.3 模型训练与超参数调优

**第一条线：逻辑回归。** 逻辑回归对特征的尺度敏感，所以数值列要先标准化；类别列要做独热编码。把这两步和模型串成 `Pipeline`，再用 `GridSearchCV` 按 ROC-AUC 选正则强度 `C`：

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

num_features = ['ClientPeriod', 'MonthlySpending', 'TotalSpent']
cat_and_bool = ['Sex', 'IsSeniorCitizen', 'HasPartner', 'HasChild',
                'HasPhoneService', 'HasMultiplePhoneNumbers',
                'HasInternetService', 'HasOnlineSecurityService',
                'HasOnlineBackup', 'HasDeviceProtection',
                'HasTechSupportAccess', 'HasOnlineTV',
                'HasMovieSubscription', 'HasContractPhone',
                'IsBillingPaperless', 'PaymentMethod']

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num_features),      # 数值列标准化
    ('cat', OneHotEncoder(), cat_and_bool),       # 类别列独热编码
])

pipe = Pipeline([('preprocess', preprocess),
                 ('logreg', LogisticRegression(random_state=42))])

# C 越小正则越强。这里在 5 折交叉验证上按 roc_auc 挑最好的 C
param_grid = {'logreg__C': [150, 120, 100, 90, 70, 60, 50, 10, 1, 0.1],
              'logreg__penalty': ['l2'],
              'logreg__solver': ['lbfgs'],
              'logreg__max_iter': [100]}
search = GridSearchCV(pipe, param_grid, scoring='roc_auc',
                      cv=5, n_jobs=-1, refit=True)
search.fit(X_train, y_train)
print('最佳参数:', search.best_params_)
print('交叉验证 AUC:', search.best_score_)
```

这一套跑出来的 5 折交叉验证 ROC-AUC 是 **0.8452**。逻辑回归的好处是稳、快、可解释：系数的大小和方向直接告诉我们哪个特征在推高流失风险。

**第二条线：CatBoost。** CatBoost 是梯度提升树的一种，特点是**原生支持类别特征**——不用独热编码，它自己会用「目标统计」的方式处理高基数列（比如 `PaymentMethod`）。所以我们把原始字符串列直接喂进去：

```python
from catboost import CatBoostClassifier

# 重新读一遍原始数据，只做 TotalSpent 的修复，不做编码
df_raw = pd.read_csv('/kaggle/input/advanced-dls-spring-2021/train.csv')
df_raw['TotalSpent'] = df_raw['TotalSpent'].replace(' ', 0).astype(float)

X = df_raw.drop(columns='Churn')
y = df_raw['Churn']

params = {'depth': [4], 'learning_rate': [0.05], 'iterations': [200],
          'l2_leaf_reg': [1], 'subsample': [0.5],
          'grow_policy': ['Lossguide'], 'random_seed': [42]}

model_cb = CatBoostClassifier(
    silent=True,
    cat_features=cat_and_bool,          # 告诉 CatBoost 哪些列是类别
    eval_metric='AUC:hints=skip_train~false')

# grid_search 会在内部做交叉验证，refit=True 表示最后用最佳参数重训一遍
best = model_cb.grid_search(params, X, y, refit=True)
print('最佳参数:', best['params'])
```

调参的搜索范围是：树的数量 200 / 250 / 300，深度 4 / 5 / 6，L2 正则 1 / 3，行采样 0.5 / 0.7，学习率 0.03 / 0.05，生长策略 `SymmetricTree` 和 `Lossguide`。最后选中的是比较浅、比较小的配置：**深度 4、200 棵树、学习率 0.05、行采样 0.5、`Lossguide`**。数据只有 5000 行，树太深、太多会直接过拟合，浅树反而泛化更好。

材料里也有人用更大的 CatBoost（1000 棵树、学习率 0.03、深度 6、`bagging_temperature=1`、早停 50 轮），效果也不错。这说明书不在「参数必须多少」，而在**用交叉验证盯住过拟合**。

### 2.4 模型测试与指标报告

训练完之后，我们对测试集输出**概率**而不是 0/1。提交文件的 `Id` 用行号，`Churn` 用 `predict_proba` 的第二列（即「流失」这一类的概率）：

```python
X_test = pd.read_csv('/kaggle/input/advanced-dls-spring-2021/test.csv')

# 测试集必须走和训练集一模一样的清洗：空格填 0、转浮点
X_test['TotalSpent'] = X_test['TotalSpent'].replace(' ', 0).astype(float)

# predict_proba 返回两列：[不流失的概率, 流失的概率]，我们只要第二列
proba = model_cb.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({'Id': range(len(proba)), 'Churn': proba})
submission.to_csv('submission.csv', index=False)
print(submission.shape, submission.head())
```

这里最容易出错的是**测试集漏做清洗**。如果训练时把空格换成了 0，测试时忘了换，CatBoost 遇到字符串会直接报错；逻辑回归那边则是标准化器遇到无法转换的值直接崩。所以我们的习惯是把清洗写成一个函数，训练集和测试集都调它，保证两边一模一样。

我们得到的分数如下（都来自材料里不同页面的记录，不是我们编的）：

| 方案 | 分数 | 说明 |
| --- | --- | --- |
| 逻辑回归（标准化 + 独热） | 0.8452 | 5 折交叉验证 ROC-AUC |
| CatBoost（原生类别特征） | 0.855+ | 公开排行榜 |
| 其他学员的集成方案 | 0.85126 / 0.85153 | 公开排行榜 |

读这张表要注意两件事。第一，**0.8452 和 0.855 不是同一个东西**：前者是本地交叉验证，后者是排行榜，两者不能直接相减说「提升了 1 个百分点」。第二，**分数集中在 0.85 附近**，说明这个数据集的信息量就这么多——从 0.845 到 0.855 只差 1 个百分点，但排名可能差很多，因为大家都很接近。

为什么本地 CV 和排行榜能对得上？前面那份对抗验证给了答案：训练集和测试集几乎无法区分（AUC 接近 0.5），两者同分布，所以 5 折 CV 是一个可信的估计。如果当时对抗验证的 AUC 是 0.9，我们就不能相信 0.8452 这个数字了。

另外，材料里还有一类做法值得一提：**stacking**。把逻辑回归、随机森林、K 近邻、决策树、CatBoost 的折外预测（OOF）拼成一张新特征表，再训一个元模型。思路上它是把「不同模型看问题的角度」合起来，但代价是要严格保证每一层的训练都不看到当行标签，否则分数会虚高。这条路在我们的主线里没有展开，只在总结里提一句。

---

## 三、总结

本文中我们完成了电信客户流失预测的完整流程。我们首先用 `info()` 发现 `TotalSpent` 被读成字符串，用相关性热力图确认了「合约、在网时长是护城河，光纤和电子账单是风险信号」；其次把 9 行新客户的空格总消费填成 0，按特征处理掉 41 行重复记录（其中包含标签互相冲突的样本），并把「没有网络服务 / 没有电话服务」合并成「没有」；接着用标准化加独热编码的逻辑回归拿到 0.8452 的 5 折交叉验证 AUC，再用原生支持类别特征的 CatBoost 把公开排行榜推到 0.855 以上；最后用对抗验证确认训练集和测试集同分布，从而确认本地 CV 可信。

这个比赛给我们的最大启发有两点。一是**不平衡数据别看准确率**，要看 ROC-AUC 这种只关心排序的指标；模型不需要把概率校准得很准，只要把高风险客户排到前面。二是**分数越接近上限，清洗和验证的可信度就越重要**——当所有人都能跑到 0.85 时，决定成败的往往不是换更复杂的模型，而是有没有把 9 行脏值、41 行重复、以及训练测试分布一致性这些事情处理干净。
