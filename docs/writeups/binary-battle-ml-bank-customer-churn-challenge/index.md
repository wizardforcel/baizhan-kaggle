# 银行客户流失预测（Binary Battle ML）

## 一、比赛简介

### 1.1 比赛背景

Binary Battle ML: Bank Customer Churn Challenge 是一道银行客户流失预测的二分类题。数据是我们熟悉的银行客户画像：信用分、所在国家（法国/德国/西班牙）、性别、年龄、在银行多少年（Tenure）、账户余额、持有几个产品、有没有信用卡、是不是活跃会员、估算年薪，外加一个客户姓名（Surname）。`Exited` 是标签，1 表示客户流失，0 表示保留下来。

训练集一共 **165,034** 行客户记录，流失率约 **21.16%**——也就是说约五分之一的人流失了（材料里直接打印出 `Churn Rate 0.211599`）。样本量不小，但存在明显的类别不平衡；另外 `Surname` 这种高基数类别特征，训练集和测试集的分布一旦漂移，模型就会死记名字而泛化失败。比赛的难点主要在前两点：怎么处理不平衡、怎么把握高基数特征。

### 1.2 比赛目标

输入是训练集 `train.csv`，每行一个客户，字段包括 `id`（行号）、`CustomerId`（客户编号）、`Surname`（姓名）、`CreditScore`、`Geography`、`Gender`、`Age`、`Tenure`、`Balance`、`NumOfProducts`、`HasCrCard`、`IsActiveMember`、`EstimatedSalary`，以及标签 `Exited`。输出是每个测试客户流失的概率，提交文件是 `id` + `Exited` 两列。

`Exited` 在提交时要填 0～1 之间的概率而不是 0/1 标签，这就暗示评价指标是排序类指标（AUC），而不是只看谁分对分错的准确率。

### 1.3 评价标准

比赛按 **ROC-AUC** 评价。AUC 的含义是：随机取一个流失客户和一个留存客户，模型把流失客户排在前面的概率。0.5 等于瞎猜，1.0 是完美。材料里的两份主代码都明确按 AUC 做验证（XGBoost 的 `eval_metric='auc'`、报告里 `Overall CV AUC`），融合后的模型也主要看 AUC 是否提升。

## 二、解题过程

整场比赛的思路是：**先做特征工程（交叉特征 + Surname 统计特征），再训练多个梯度提升树模型，最后用加权平均把它们的概率融合在一起。** 我们按 EDA、清洗、训练、测试四步走。

### 2.1 数据探索（EDA）

第一步是读数据、看整体情况。材料里直接给出了三个 KPI：总客户数 165,034、流失率 0.211599、平均年龄 38.125888、平均余额 55478.086689。

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('../input/binary-battle-ml-bank-customer-churn-challenge/train.csv')
test  = pd.read_csv('../input/binary-battle-ml-bank-customer-churn-challenge/test.csv')

print('训练集', train.shape, '测试集', test.shape)

# 流失率：约 21%，属于明显不平衡，后面验证不能只看准确率
print('流失率', train['Exited'].mean())
print(train['Exited'].value_counts(normalize=True))

# 缺失值总个数：先确认这份数据干不干净
print('训练集缺失值总数', train.isnull().sum().sum())
print('测试集缺失值总数', test.isnull().sum().sum())
```

两份数据都没有缺失值，这省掉了补缺失的环节，清洗重点就落在**类别编码**和**分布一致性**上。

接下来我们做了两件 EDA。第一件是看每个特征和流失的关系，第二件是检验训练集和测试集分布是否一致——后者用的是 KS 检验：

```python
from scipy.stats import ks_2samp

# 对数值特征做 KS 检验：p 值小说明训练/测试分布差异大，这样的特征要警惕
def compare_train_test(train, test):
    rows = []
    for col in train.columns:
        if col == 'Exited':
            continue
        if train[col].dtype in ['int64', 'float64']:
            stat, p = ks_2samp(train[col].dropna(), test[col].dropna())
            rows.append({'feature': col, 'type': '数值', 'value': stat, 'p_value': p})
        else:
            # 类别特征比「分布差异绝对值之和」，越小越一致
            td = train[col].value_counts(normalize=True)
            dd = test[col].value_counts(normalize=True)
            idx = td.index.union(dd.index)
            diff = np.abs(td.reindex(idx, fill_value=0) - dd.reindex(idx, fill_value=0)).sum()
            rows.append({'feature': col, 'type': '类别', 'value': diff, 'p_value': np.nan})
    return pd.DataFrame(rows)

result = compare_train_test(train, test)
result['flag'] = result.apply(
    lambda r: 'OK' if (r['type'] == '数值' and r['p_value'] > 0.05)
    or (r['type'] == '类别' and r['value'] < 0.1) else 'DIFF', axis=1)
print(result.sort_values('value', ascending=False))
```

这个检验很有用：如果某个特征在训练集和测试集里分布差得离谱，模型在训练集上学到的一切，在测试集上都会失效。检查下来这份数据的分布基本一致，只有少数特征的差异需要留意。

我们还看了两个业务的「老问题」：德国客户的流失率明显更高；另外材料的图显示，不同国家的流失率不同，性别上男性流失率略低。材料的 KDE 图还显示，年龄越大流失率越高，`Age` 是区分「会不会流失」最直观看得出来的连续特征。这些方向性的结论不需要精确到小数，只要能指导后面的特征工程——比如给不平衡处理和类别编码提供依据就够了。

```python
# 年龄按标签画密度曲线：两条曲线分得开，说明这个特征有区分度
sns.kdeplot(data=train, x='Age', hue='Exited', fill=True)
plt.title('按流失状态看年龄分布')
plt.show()

# 按国家看流失率：德国明显更高，说明 Geography 必须进模型
train.groupby('Geography')['Exited'].mean().sort_values().plot(kind='bar')
plt.ylabel('流失率'); plt.title('分国家流失率')
plt.show()
```

### 2.2 数据清洗

清洗分成三步：合并、造交叉特征、编码。

第一步，把训练集和测试集拼到一张表里处理。这样做的目的很简单：**Surname 这类特征的所有取值（包括测试集里出现、训练集里没有的名字）都能被统一编码，避免测试集一编码就出现「训练时没见过」的陌生值**。

```python
# 合并后统一做特征工程，最后再拆回两份
train['is_train'] = 1
test['is_train'] = 0
test['Exited'] = np.nan                       # 测试集没有标签，先占个位
all_data = pd.concat([train, test], axis=0).reset_index(drop=True)

# 交叉特征 1：只持有一个产品——这批客户的流失行为很特殊
all_data['SingleProduct'] = (all_data['NumOfProducts'] == 1).astype(int)

# 交叉特征 2：有信用卡但从不活跃——成员关系较浅的信号
all_data['CardButInactive'] = ((all_data['HasCrCard'] == 1) &
                                (all_data['IsActiveMember'] == 0)).astype(int)

# 交叉特征 3：零余额客户，行为和其他客户明显不同
all_data['ZeroBalance'] = (all_data['Balance'] == 0).astype(int)

# 交叉特征 4：平均到每个产品上的余额
all_data['Balance_Per_Product'] = all_data['Balance'] / all_data['NumOfProducts']
```

第二步处理 Surname。直接给姓名做独热编码会造出几千列，还容易过拟合；我们把它拆成「前 5 个字符 + 长度 + 姓氏出现频次」三个统计特征，既保住了信息，又大幅压缩了维度：

```python
# 姓氏前缀：很多姓氏共享前缀，从「每个名字」粗化到「每类名字」
all_data['Surname_prefix'] = all_data['Surname'].str[:5]
all_data['Surname_len'] = all_data['Surname'].str.len()
# 姓氏在整个合并数据里出现了多少次：稀有姓氏可能对应特殊人群
all_data['Surname_freq'] = all_data['Surname'].map(all_data['Surname'].value_counts())

# 低基数类别（国家、性别）做独热编码
all_data = pd.get_dummies(all_data, columns=['Geography', 'Gender'], drop_first=False)

# 高频数的姓氏前缀再用频次编码补一个数值特征
prefix_freq = all_data['Surname_prefix'].value_counts()
all_data['Surname_prefix_freq'] = all_data['Surname_prefix'].map(prefix_freq)

# 剩下的高基数文字列统一标签编码（编号当类别代号，不是当作有序数字）
from sklearn.preprocessing import LabelEncoder
for col in ['Surname', 'Surname_prefix']:
    all_data[col] = LabelEncoder().fit_transform(all_data[col])

# 训练/测试真正建模前，去掉纯 id 列和原始姓名字段
all_data_encoded = all_data.drop(columns=['id', 'CustomerId', 'Surname', 'Surname_prefix'])
```

最后拆回训练集和测试集：

```python
train_enc = all_data_encoded[all_data_encoded['is_train'] == 1].copy()
test_enc  = all_data_encoded[all_data_encoded['is_train'] == 0].copy()

y = train_enc['Exited']
X = train_enc.drop(columns=['Exited', 'is_train'])
X_test = test_enc.drop(columns=['Exited', 'is_train'])

# 分层切 20% 做验证集：正负比例和整体一致，分数比纯随机切更可信
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
```

### 2.3 模型训练与超参数调优

这一场我们把五种模型都试了一遍：随机森林、梯度提升（GradientBoosting）、XGBoost、LightGBM 和逻辑回归。思路是让不同模型「投票」——模型之间差异越大，融合后的抵抗力越强。

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=500, max_depth=12,          # 500 棵树、深度限制在 12
        min_samples_split=20, min_samples_leaf=10,  # 限制叶子，防过拟合
        random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=3000, learning_rate=0.025,  # 多轮 + 小学习率
        max_depth=4, subsample=0.8, max_features=0.8, random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=3000, learning_rate=0.02, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,     # 行列都采样，防止过拟合
        objective='binary:logistic', eval_metric='auc',
        early_stopping_rounds=100, random_state=42, n_jobs=-1),
    'LightGBM': LGBMClassifier(
        n_estimators=3000, learning_rate=0.02,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        objective='binary', random_state=42, n_jobs=-1),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
}

results = []
trained = {}
for name, model in models.items():
    # boosting 模型用验证集做早停，其余直接 fit
    if name in ('XGBoost', 'LightGBM'):
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, prob)
    results.append({'model': name, 'AUC': auc})
    trained[name] = model
    print(name, auc)

print(pd.DataFrame(results).sort_values('AUC', ascending=False))
```

早停在这场比赛里作用很大：3000 棵树是上限，当验证集 AUC 连续 100 轮不再上升就提前停，省时间也不容易过拟合。

树上三个模型（GradientBoosting、XGBoost、LightGBM）各有各的弱点，我们最后用**加权软投票**把它们的概率融合，权重是按验证集表现大致调的：0.4 / 0.34 / 0.26。

```python
# 三个模型的概率分别取出来
p_gb = trained['GradientBoosting'].predict_proba(X_val)[:, 1]
p_xgb = trained['XGBoost'].predict_proba(X_val)[:, 1]
p_lgb = trained['LightGBM'].predict_proba(X_val)[:, 1]

# 加权平均：表现最好的给最大权重。融合后 AUC 通常比单个模型更稳
pred_avg = 0.4 * p_gb + 0.34 * p_xgb + 0.26 * p_lgb
print('融合后验证 AUC:', roc_auc_score(y_val, pred_avg))
```

融合为什么有用？一句话：**不同模型犯的错不在同一个地方**。三个 boosting 模型的结构和随机性不同，各自把概率平均一下，那些「只有个别模型自信」的错误就会被压下去。材料里也直接说，融合后的验证 AUC 比任何单个模型都高、也更稳定。

### 2.4 模型测试与指标报告

验证阶段除了 AUC，我们还按阈值 0.5 转成标签，看准确率和混淆矩阵，确认在「现实业务」层面的表现：

```python
from sklearn.metrics import accuracy_score, confusion_matrix

# 概率转 0/1：阈值默认 0.5，但业务上可以为了召回率往下调
y_pred_label = (pred_avg >= 0.5).astype(int)
print('Accuracy:', accuracy_score(y_val, y_pred_label))

# 混淆矩阵：数清楚「流失被漏报」和「留存被误报」各有多少
cm = confusion_matrix(y_val, y_pred_label)
print(pd.DataFrame(cm,
                   index=['实际留存', '实际流失'],
                   columns=['预测留存', '预测流失']))
```

读这张表时要注意一件事：验证集里流失客户只占两成左右，所以准确率天然会偏高，一个「永远猜留存」的模型也能有接近八成的准确率。也正因为如此，我们把 AUC 当作真正的选择依据，准确率和混淆矩阵只用来向业务方解释误判的代价——漏掉一个会流失的客户，和误判一个不会流失的客户，成本完全不同。

最后，用全量训练数据重新训练三个 boosting 模型，再对测试集做同样的加权平均，生成提交文件。这一步和交叉验证阶段不同：**交叉验证时每折只用了 80% 的数据，重训则把 100% 的数据都用上**，通常能稳定带来一点点提升。

```python
# 重训：三个模型都在全量训练数据上调好，再用同一组权重预测测试集
trained['GradientBoosting'].fit(X, y)
trained['XGBoost'].fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
trained['LightGBM'].fit(X, y, eval_set=[(X_val, y_val)], eval_metric='auc')

test_pred = (
    0.4 * trained['GradientBoosting'].predict_proba(X_test)[:, 1]
    + 0.34 * trained['XGBoost'].predict_proba(X_test)[:, 1]
    + 0.26 * trained['LightGBM'].predict_proba(X_test)[:, 1]
)

submission = pd.DataFrame({'id': test['id'], 'Exited': test_pred})
submission.to_csv('submission.csv', index=False)
submission.head()
```

另外值得一提的是材料的另一份代码：只用 XGBoost 加 5 折分层交叉验证，也能得到不错的 OOF AUC，并且代码更短。对我们来说，比赛的最后几步其实没有标准答案——模型可以是一个也可以是多个，关键是每一步都要能在验证集上自圆其说。

## 三、总结

本文中我们完成了 Binary Battle ML 银行客户流失比赛的完整流程。我们首先做了 EDA：确认了 165,034 行训练数据、约 21.16% 的流失率，并用 KS 检验检查了训练集和测试集的分布差异；其次做了数据清洗，把训练测试合并后构造了单产品、无活跃信用卡、零余额、单产品余额等交叉特征，把 Surname 拆成前缀、长度和频次三个统计特征，低基数类别做独热编码，高基数文字列做标签编码；接着训练了随机森林、梯度提升、XGBoost、LightGBM 和逻辑回归五个模型，对三个 boosting 模型用 0.4/0.34/0.26 的权重做软投票融合，验证 AUC 高于任何单个模型；最后用全量数据重训三个模型，加权平均后输出概率提交文件。材料里另一条思路是只用 XGBoost 做 5 折分层交叉验证，也能拿到稳定的 OOF AUC。

这个比赛给我们最大的启发是：**小步快跑比追求单个模型更值得**。特征工程上，与其给几千个姓氏做独热编码，不如用频次这类统计特征；模型上，与其执着于调出一个神通广大的单模型，不如让几个差异化的 boosting 模型各发挥所长再融合。