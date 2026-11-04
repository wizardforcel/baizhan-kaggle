# 银行客户流失二分类（Binary Classification with a Bank Churn Dataset）

## 一、比赛简介

### 1.1 比赛背景

Binary Classification with a Bank Churn Dataset 是一道标准的银行客户流失二分类题。数据是合成的银行客户画像，一行一位客户。材料里的两份 notebook 其实是在同一个比赛上分工：一份专门做 EDA 和变量关系分析（`EDA_and_Relationships.ipynb`），一份做模型实现与提交（`ML_Implementation.ipynb`），本篇 writeup 把这两份拼成一条完整流水线。

这类表格数据对新手非常友好：字段不多、没有太复杂的文本和图片，缺失值也几乎不存在。真正的难点只有两个——**类别不平衡**（流失的人永远是少数）和**类别字段怎么编码**（国家、性别怎么变成数字），其余部分都是套路。

### 1.2 比赛目标

输入是 `train.csv`，每行一个客户，字段如下：

- `id`、`CustomerId`：行号和客户编号，无预测意义；
- `Surname`：客户姓氏，高基数类别特征；
- `CreditScore`：信用分；`Age`：年龄；`Tenure`：在银行多少年；`Balance`：账户余额；`EstimatedSalary`：估算年薪；
- `Geography`：所在国家（France/Germany/Spain）；`Gender`：性别；
- `NumOfProducts`：持有几个产品；`HasCrCard`：有没有信用卡；`IsActiveMember`：是不是活跃会员；
- `Exited`：标签，1 表示流失，0 表示留下。

输出是每个测试客户流失的概率，提交文件是 `id` + `Exited` 两列。测试集路径和训练集是同一个数据目录下的 `test.csv`。

### 1.3 评价标准

提交文件要求填概率（而不是 0/1 标签），所以比赛按 **ROC-AUC** 评价，越高越好；0.5 等于纯猜，1.0 是完美。材料里的代码除了算 ROC-AUC，还额外报告准确率、绘制 ROC 曲线和混淆矩阵，理由是：AUC 回答「排序好不好」，准确率回答「业务上能猜对多少人」，两个视角都要。

## 二、解题过程

整体思路：**EDA 先摸清哪些字段和流失强相关，清洗时构造一批比率特征，然后金字塔式地试三个模型——逻辑回归打底、随机森林进阶、LightGBM 加 10 折交叉验证收尾。** 我们按四步走。

### 2.1 数据探索（EDA）

第一步读数据，确认形状、缺失值和目标分布：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('/kaggle/input/competitions/'
                    'binary-classification-with-bank-churn-dataset/train.csv')
test  = pd.read_csv('/kaggle/input/competitions/'
                    'binary-classification-with-bank-churn-dataset/test.csv')

print('训练集', train.shape, '测试集', test.shape)
print('缺失值：训练集', train.isnull().sum().sum(), '测试集', test.isnull().sum().sum())
print('重复行/重复 CustomerId：', train.duplicated().sum(), train['CustomerId'].duplicated().sum())

print(train['Exited'].value_counts(normalize=True))   # 流失占比
```

两份数据都没有缺失值、也没有重复，省掉了最麻烦的环节。金字塔的第一步，是画出数值特征与流失的关系、以及特征之间的相关性。注意这里的目的不是追求画面漂亮，而是**给特征工程带路**：

```python
# 数值特征的分布图：先看 CreditScore、Age、Balance 这些列长什么样
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, c in zip(axes.flatten(), num_cols):
    sns.histplot(train[c], kde=True, ax=ax)
    ax.set_title(c)
plt.tight_layout()
plt.show()

# 相关系数热力图：数字越接近 0 说明线性关系越弱
corr = train[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
              'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# 分国家看流失率：地中海 vs 德国的业务结构不一样
sns.barplot(x='Geography', y='Exited', data=train)
plt.show()
```

EDA 部分得到的关键结论，材料里总结得很清楚，我们在后面都用到：

1. `Exited` 类别不平衡，大部分客户留存；
2. `Age`、`NumOfProducts`、`IsActiveMember` 和 `Geography`（尤其是德国）和流失关系最强；
3. `Balance` 在 0 的位置有一个尖峰——大量客户零余额，这批人的流失行为与众不同；
4. `CreditScore` 和 `EstimatedSalary` 单独看和流失的线性相关很弱；
5. 训练集和测试集的分布一致，模型应当能泛化。

再看几张具体关系图收尾。把 `Age` 按是否流失画箱线图，可以看到流失人群的年龄整体偏大；把 `NumOfProducts` 按是否流失画柱状图，持有 1 个产品和持有 2 个产品的客户，流失比例差别非常明显。这两条和「年龄、产品数是最强特征」的结论互相印证，也解释了为什么后面的特征工程里我们专门造了「产品密度」这种组合特征。

```python
# 流失 vs 留存的年龄分布：流失人群整体年龄更大
sns.boxplot(x='Exited', y='Age', data=train)
plt.title('Age vs Exited')
plt.show()

# 每个产品数档位的流失率：1 个产品和 2 个产品是天壤之别
train.groupby('NumOfProducts')['Exited'].mean().plot(kind='bar')
plt.ylabel('流失率'); plt.title('Churn Rate by NumOfProducts')
plt.show()
```

### 2.2 数据清洗

既然没有缺失，清洗主要做两件事：**造特征**和**编码**。

先看特征工程。作者完全照着 EDA 的结论来：既然 `Age` 连续、`Balance` 有零值尖峰、`NumOfProducts` 与流失相关，那就把它们组合成比率、标记和分箱特征：

```python
def feature_engineer(df):
    df = df.copy()
    # 余额/年薪：客户把多大比例的收入放在这家银行
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    # 信用分/年龄：单位年龄上的信用水平
    df['CreditScore_Age_Ratio'] = df['CreditScore'] / (df['Age'] + 1)
    # 在网年数/年龄：占比越高说明客户越小就来了，忠诚度信号
    df['Tenure_Age_Ratio'] = df['Tenure'] / (df['Age'] + 1)
    # 零余额标记：把 EDA 里看到的尖峰转成一个 0/1 特征
    df['IsZeroBalance'] = (df['Balance'] == 0).astype(int)
    # 产品数/在网年数：产品密度
    df['ProductsPerTenure'] = df['NumOfProducts'] / (df['Tenure'] + 1)
    # 年龄分箱：连续年龄按业务常识切成五段
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=False)
    # 姓氏频次：把高基数姓名压缩成一个统计特征
    df['Surname_Freq'] = df['Surname'].map(df['Surname'].value_counts())
    # 建模不需要 id、CustomerId 和 Surname 原文
    return df.drop(columns=['id', 'CustomerId', 'Surname'])

train_fe = feature_engineer(train)
test_fe = feature_engineer(test)
```

所有分母都加 1，是为了防止除数为 0（比如 `Tenure=0`）。这些特征的共同点是从 EDA 的「观察」变成了「可输入模型的数字」，这就是特征工程的全部意义。

然后是编码。`Geography` 和 `Gender` 是文字列，我们做了关键的一步：**把训练集和测试集拼在一起拟合 LabelEncoder**。这样如果测试集出现了训练集没有的国家（虽然这张表不会，但保平安），也不会报「未知类别」的错误：

```python
from sklearn.preprocessing import LabelEncoder

cat_cols = ['Geography', 'Gender']
for c in cat_cols:
    le = LabelEncoder()
    # 拼接训练+测试，一次性 fit：保证两边得到同一套编码规则
    combined = pd.concat([train_fe[c], test_fe[c]], axis=0).astype(str)
    le.fit(combined)
    train_fe[c] = le.transform(train_fe[c].astype(str))
    test_fe[c] = le.transform(test_fe[c].astype(str))

X = train_fe.drop(columns=['Exited'])
y = train_fe['Exited']
X_test = test_fe
print('特征数:', X.shape[1])
```

### 2.3 模型训练与超参数调优

我们按从简到难的顺序训练三个模型。**逻辑回归**是打底模型：它简单、可解释，最重要的是能当「及格线」，告诉我们数据里信号的下限在哪里。注意它需要先做标准化（`StandardScaler`），否则信用分和余额这种量纲差异会拖慢收敛：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

# 先切一个 20% 的验证集给逻辑回归和随机森林用
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 逻辑回归对特征的量纲敏感，先标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# class_weight='balanced'：给少数类（流失）更大的权重，抵消不平衡
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train_scaled, y_train)
print('逻辑回归 AUC:', roc_auc_score(y_val, log_reg.predict_proba(X_val_scaled)[:, 1]))
```

**随机森林**是第二步。树模型不需要标准化，自带特征重要性，正好让我们回头验证 EDA 的结论。参数上，400 棵树、最大深度 10、叶子最少 5 个样本，都指向「够深但不过分」的方向：

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=400, max_depth=10,
                            min_samples_leaf=5, class_weight='balanced',
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print('随机森林 AUC:', roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1]))

# 特征重要性：树的贡献被平摊到各特征上，可以看出谁在真正出力
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(10))
```

**LightGBM** 是最终主角，配 10 折分层交叉验证。为什么要 10 折而不是简单切一个验证集？因为 10 折意味着「每个样本都会被某一个模型预测一次」，把这些袋外预测（OOF）拼起来算的 AUC，比单次切分的验证 AUC 更稳定、更能代表真实水平：

```python
import lightgbm as lgb

params = {
    'objective': 'binary',          # 二分类
    'metric': 'auc',                # 监控 AUC
    'learning_rate': 0.03,          # 小学习率，配合 3000 轮上限
    'num_leaves': 64,
    'min_child_samples': 20,        # 叶子最少样本数，防过拟合
    'subsample': 0.8,               # 行采样
    'colsample_bytree': 0.8,        # 列采样
    'reg_alpha': 0.1,               # L1 正则
    'reg_lambda': 0.1,              # L2 正则
    'seed': 42,
    'verbosity': -1,
}

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
cat_feats = ['Geography', 'Gender']    # 告诉 LightGBM 哪些是类别特征

for fold, (tr, va) in enumerate(kf.split(X, y)):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_feats)
    dva = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_feats)

    model = lgb.train(params, dtr, num_boost_round=3000,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof[va] = model.predict(X_va, num_iteration=model.best_iteration)
    # 测试集预测按折平均，降低单折波动
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    print(f'fold {fold+1} AUC:', roc_auc_score(y_va, oof[va]))

print('OOF AUC:', roc_auc_score(y, oof))
```

`early_stopping(100)` 的意思是验证 AUC 连续 100 轮不涨就停，并保留最好的那轮——我们不必手工猜训练轮数。

训练完之后，我们还用特征重要性（按 gain 排序）交叉验证一次 EDA 的结论。`NumOfProducts`、`IsActiveMember`、`Age`、`Geography`、`Balance` 排在最前面，和「哪几个字段和流失关系最强」的判断完全对得上：

```python
# 用第一棵树模型的 gain 重要性，确认哪些特征在真正出力
gain_imp = pd.Series(
    model.feature_importance(importance_type='gain'), index=X.columns
).sort_values(ascending=False)
gain_imp.head(10).plot(kind='barh')
plt.title('LightGBM 特征重要性（Gain）')
plt.show()
```

调参上，我们只做了一件「大事」：把学习率压到 0.03，把 `num_leaves` 控制在 64，再加上行列采样和正则。实践证明，对这种几千到几万行的表格数据，LightGBM 对参数并不敏感，这套配置已经接近好用的区间，再多调只能换来小数点后第四位的提升。

### 2.4 模型测试与指标报告

把三个模型的 ROC 曲线叠在一起，是最直观的对比方式：

```python
from sklearn.metrics import roc_auc_score, roc_curve

fpr_lr, tpr_lr, _ = roc_curve(y_val, log_reg.predict_proba(X_val_scaled)[:, 1])
fpr_rf, tpr_rf, _ = roc_curve(y_val, rf.predict_proba(X_val)[:, 1])
fpr_lgb, tpr_lgb, _ = roc_curve(y, oof)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic (AUC={roc_auc_score(y_val, log_reg.predict_proba(X_val_scaled)[:,1]):.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'RF (AUC={roc_auc_score(y_val, rf.predict_proba(X_val)[:,1]):.3f})')
plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM OOF (AUC={roc_auc_score(y, oof):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.legend(); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC 曲线对比')
plt.show()
```

最后把三个模型的准确率和 AUC 放进同一张表，作为「模型对比小结」：

```python
from sklearn.metrics import accuracy_score

results = pd.DataFrame({
    '模型': ['逻辑回归', '随机森林', 'LightGBM (10折OOF)'],
    '准确率': [
        accuracy_score(y_val, (log_reg.predict_proba(X_val_scaled)[:, 1] >= 0.5).astype(int)),
        accuracy_score(y_val, (rf.predict_proba(X_val)[:, 1] >= 0.5).astype(int)),
        accuracy_score(y, (oof >= 0.5).astype(int)),
    ],
    'ROC-AUC': [
        roc_auc_score(y_val, log_reg.predict_proba(X_val_scaled)[:, 1]),
        roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1]),
        roc_auc_score(y, oof),
    ],
})
print(results)
```

结论和材料一致：**LightGBM 加 10 折分层交叉验证是三个模型里最强的，也是最终提交的模型**。随机森林次之，逻辑回归作为基线也不算差——它证明了这个任务里信号确实存在。

除了 ROC 曲线，我们还画了 LightGBM 袋外预测的精确率-召回率曲线，以及「预测概率按真实类别」的分布直方图。后者用来回答一个重要问题：模型给出的概率是「软」的（两类样本分得开）还是「只会在 0.5 附近犹豫」？如果分布重叠得很厉害，说明特征信息已经用到极限，再换模型意义也不大；如果分得很开，则说明提升空间还大着：

```python
from sklearn.metrics import precision_recall_curve

# PR 曲线：在不平衡数据上比 ROC 更严格地反映「抓流失」的表现
prec, rec, _ = precision_recall_curve(y, oof)
plt.plot(rec, prec, color='purple')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve (LightGBM OOF)')
plt.show()

# 留下客户和流失客户的预测概率分布：重合越多，说明越难分
sns.histplot(oof[y == 0], color='green', label='留存', kde=True, stat='density', alpha=0.5)
sns.histplot(oof[y == 1], color='red', label='流失', kde=True, stat='density', alpha=0.5)
plt.legend(); plt.title('真实类别的预测概率分布'); plt.show()
```

提交文件同样简单直接：`id` 用测试集的 `id`，`Exited` 用 10 折平均下来的测试概率：

```python
submission = pd.DataFrame({'id': test['id'], 'Exited': test_preds})
submission.to_csv('submission.csv', index=False)
submission.head()
```

材料最后还给了一条后续优化的思路：把 LightGBM 和 CatBoost（原生支持类别特征）或 XGBoost 再融合，或者对 `Geography`/`Surname` 做目标编码——这些都可以在本地验证后再决定要不要上。

## 三、总结

本文中我们完成了银行客户流失二分类比赛的完整流程。我们首先做 EDA，确认了数据没有缺失、没有重复，流失样本占少数，且 `Age`、`NumOfProducts`、`IsActiveMember`、`Geography`（尤其德国）与流失关系最强，`Balance` 存在零值尖峰；其次做数据清洗，构造了余额年薪比、信用分年龄比、零余额标记、产品密度、年龄分箱和姓氏频次等特征，并把国家、性别用「训练+测试拼接」的 LabelEncoder 统一编码；接着训练了逻辑回归、随机森林、LightGBM 三个模型，LightGBM 配 10 折分层交叉验证和早停，用袋外预测评估；最后对比三个模型的 ROC-AUC 和准确率，确认 LightGBM 最强，用它输出测试概率提交。这套「逻辑回归打底 → 随机森林验证 → LightGBM 收尾」的节奏，适合任何一张字段不多、类别不平衡的表格数据。

这个比赛给我们的最大启发是：**EDA 的结论要能直接指挥特征工程**。作者不是凭空造了十几个特征，而是照着「年龄和产品数最相关、零余额行为特殊、姓名需要压缩」这三条发现来造；反过来，造完特征再回头用随机森林的特征重要性验证一遍。数据洞察、特征、模型三者之间形成闭环，才是表格数据比赛的正确姿势。