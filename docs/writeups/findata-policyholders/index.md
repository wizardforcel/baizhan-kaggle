# 车险投保风险分类（findata-policyholders）

## 一、比赛简介

### 1.1 比赛背景

这份数据来自一家经营车辆保险的保险公司。客户来投保时，公司需要判断：这份车险合同未来会不会「出险」，也就是客户会不会申请理赔。会出险的合同叫**风险合同**，不会出险的叫**非风险合同**。如果模型能提前把高风险合同挑出来，公司就可以调整核保策略，比如提高保费、增加免赔条款，或者直接拒保。

材料里的俄文说明写得很清楚：任务是为车辆保险合同建立风险预测模型，目标 `target` 是风险类别，`1` 表示风险合同，`0` 表示非风险合同，评价指标是 ROC-AUC。数据文件有三个：

- `insclass_train.csv`：训练集，带 `target` 标签；
- `insclass_test.csv`：测试集，只有特征；
- `insclass_sample.csv`：提交样例。

数据的特点是**匿名化**：特征名大多是 `variable_1`、`variable_2` 这样的编号，没有业务含义。我们不知道每一列具体代表什么，只能靠统计规律和模型自己去学。这也意味着特征工程的空间有限，重点要放在预处理、模型比较和验证策略上。

### 1.2 比赛目标

训练集每一行是一份保险合同，包含一个 `id` 列、若干数值特征和类别特征，以及目标列 `target`。测试集有相同的特征列，但没有 `target`。我们要为每份测试合同输出一个**风险概率**，取值在 0 到 1 之间，越大表示越可能出险。

需要注意输出的是概率而不是硬标签。比赛按 ROC-AUC 评分，只要概率的排序正确就行，不需要自己卡一个 0.5 的阈值。这一点和准确率不同：准确率只看「分对没分对」，ROC-AUC 看「风险高的合同是不是真的排得更靠前」。

材料里还列出了一批「看起来是数字、实际上是类别」的二元特征：`variable_2`、`variable_3`、`variable_8`、`variable_10`、`variable_11`、`variable_23`、`variable_24`、`variable_25`、`variable_26`。它们只有两个取值，应该按类别处理，而不是当成连续数值去标准化。

### 1.3 评价标准

**ROC-AUC（Receiver Operating Characteristic - Area Under Curve）** 是二分类里常用的排序指标。用大白话解释：

1. 从真实的风险合同里随机抽一个，再从非风险合同里随机抽一个；
2. 看模型给风险合同打的分是不是比非风险合同高；
3. 如果每次都能排对，AUC 就是 1；如果完全随机，AUC 就是 0.5。

所以 ROC-AUC 衡量的是「模型把正样本排在负样本前面的能力」，越高越好，满分 1.0。它的好处是不受阈值影响，也不受类别比例影响；坏处是不关心概率的绝对值校准——模型说 0.9 还是 0.6 不重要，只要排序对就行。

---

## 二、解题过程

我们的主线是：先做 EDA 看清目标分布和特征类型，再用 `ColumnTransformer` 把数值列和类别列分开预处理，然后横向比较五种模型，最后选验证 ROC-AUC 最好的一个输出概率。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，确认规模和列结构：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子，保证结果可复现
RANDOM_STATE = 42

# 读取训练集、测试集、提交样例
train_data = pd.read_csv('/kaggle/input/findata-policyholders/insclass_train.csv')
test_data = pd.read_csv('/kaggle/input/findata-policyholders/insclass_test.csv')
sample_submission = pd.read_csv('/kaggle/input/findata-policyholders/insclass_sample.csv')

# 看规模
print(train_data.shape, test_data.shape, sample_submission.shape)

# 看前几行，确认列名和标签位置
display(train_data.head())
display(test_data.head())
display(sample_submission.head())
```

接着看数据类型和缺失情况。`info()` 会列出每一列的非空数量和类型，`isnull().sum()` 则能精确定位哪些列有缺失：

```python
# 训练集和测试集的列类型、非空数量
train_data.info()
test_data.info()

# 只打印有缺失的列，避免输出太长
missing_train = train_data.isnull().sum()
print(missing_train[missing_train > 0])

missing_test = test_data.isnull().sum()
print(missing_test[missing_test > 0])
```

匿名数据里经常有大量缺失，而且缺失本身可能就是信息（比如某类客户就是不填某一项）。我们没有直接删列，而是把缺失留给后面的 `SimpleImputer` 处理，让它按列类型选择中位数或众数。

然后看目标分布：

```python
# 目标列计数和比例
target_counts = train_data['target'].value_counts()
print(target_counts)
print(f"风险合同占比：{target_counts[1] / len(train_data) * 100:.2f}%")

# 画柱状图直观看看是否平衡
sns.countplot(x='target', data=train_data)
plt.title('Target distribution')
plt.show()
```

材料里没有保留具体比例，所以我们不引用数字。但从业务上看，出险的合同通常是少数，这会影响后面看混淆矩阵时的判断：即使模型把大多数样本判成 0，准确率也可能不低，所以我们必须用 ROC-AUC 而不是准确率来选模型。

接着区分类别列和数值列。除了 `object` 类型的列，前面提到的那些二元特征也要算作类别：

```python
# 文本类型的列
categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()

# 这些列名字是数字，但只取两个值，本质是类别
binary_features = ['variable_2', 'variable_3', 'variable_8', 'variable_10',
                   'variable_11', 'variable_23', 'variable_24',
                   'variable_25', 'variable_26']

# 合并成完整的类别列清单，并去掉重复
for f in binary_features:
    if f in train_data.columns and f not in categorical_features:
        categorical_features.append(f)

# 剩下的数值列（排除 target）
numerical_features = [c for c in train_data.select_dtypes(include=['int64', 'float64']).columns
                      if c != 'target' and c not in categorical_features]

print('类别列：', categorical_features)
print('数值列：', numerical_features)
```

最后我们看数值特征和目标的相关性，找出哪些列值得重点关注：

```python
# 只对数值列算相关系数，target 放最后
corr = train_data[numerical_features + ['target']].corr()

# 热力图：红色正相关，蓝色负相关
plt.figure(figsize=(16, 12))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation matrix')
plt.show()

# 按与 target 的相关性排序
print(corr['target'].sort_values(ascending=False))
```

相关性只能反映线性关系，树模型能捕捉非线性和交互，所以我们不据此删特征，只用它来确认没有哪一列泄漏了目标。

### 2.2 数据清洗

我们首先把特征和标签分开，再按 8:2 做**分层切分**。分层（stratify）的意思是让训练集和验证集里风险合同的比例保持一致，避免验证集恰好抽到太多或太少正样本：

```python
from sklearn.model_selection import train_test_split

# 特征和标签分开
X = train_data.drop('target', axis=1)
y = train_data['target']

# 80% 训练、20% 验证，stratify=y 保证两边正负比例一致
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
```

接着我们定义两套预处理流程。数值列先填中位数再标准化；类别列先填众数再做 one-hot：

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 数值列：中位数填补 → 标准化（减均值除标准差）
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# 类别列：众数填补 → one-hot
# handle_unknown='ignore' 让测试集出现新类别时不报错，而是全部填 0
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

# 把两套流程按列拼起来
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features),
])
```

这里有两个细节值得解释。

第一，`handle_unknown='ignore'` 很重要。训练集里没出现过的类别，测试集里可能出现；如果不设这个参数，`OneHotEncoder` 会直接抛错。设成 `ignore` 后，这些未知类别会被编码成全 0，模型仍然能给出预测。

第二，标准化对逻辑回归和 SVM 是必需的，对随机森林、XGBoost、LightGBM 其实无所谓——树模型只看分裂点，不看数值的绝对大小。但为了统一流程，我们对所有模型都用同一个 `preprocessor`，这样比较的是模型本身，而不是预处理差异。

### 2.3 模型训练与超参数调优

我们写两个函数：一个负责「训练 + 在验证集算 ROC-AUC」，另一个负责「在全部训练数据上做 5 折交叉验证」。这样每个模型都有两个分数，可以互相印证：

```python
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

def train_evaluate_model(model, X_train, y_train, X_val, y_val, preprocessor):
    """训练模型并在验证集上计算 ROC-AUC"""
    # 预处理和模型串成一个流水线，避免验证时重复写代码
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)

    # 取正类的概率，而不是硬标签
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    return pipeline, roc_auc_score(y_val, y_pred_proba)

def cross_validate_model(model, X, y, preprocessor, cv=5):
    """在全部训练数据上做分层 K 折交叉验证"""
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(
        pipeline, X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring='roc_auc',
    )
    return scores.mean(), scores.std()
```

然后我们准备五个候选模型，覆盖线性模型、bagging 和 boosting 三大类：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = {
    '逻辑回归': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    '随机森林': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    '梯度提升': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    'LightGBM': LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1),
}
```

选这五个模型的理由：逻辑回归是线性基线，能告诉我们「不做非线性变换能到多少分」；随机森林是 bagging 代表，对缺失和异常值稳健；梯度提升、XGBoost、LightGBM 是 boosting 代表，通常在表格数据上最强。先跑默认参数，是为了看清哪一类模型更适合这份数据，再决定要不要细调。

训练和比较的循环如下：

```python
results = {}
trained_models = {}

for name, model in models.items():
    print(f'训练模型：{name}')

    # 80/20 验证集上的分数
    trained_model, val_auc = train_evaluate_model(
        model, X_train, y_train, X_val, y_val, preprocessor
    )

    # 5 折交叉验证的均值和标准差
    cv_mean, cv_std = cross_validate_model(model, X, y, preprocessor)

    results[name] = {'val_auc': val_auc, 'cv_mean': cv_mean, 'cv_std': cv_std}
    trained_models[name] = trained_model

    print(f'  验证 ROC-AUC：{val_auc:.4f}')
    print(f'  交叉验证 ROC-AUC：{cv_mean:.4f} ± {cv_std:.4f}')
```

我们看结果时遵循一个原则：**单个验证集分数可能波动，交叉验证均值加减标准差更可信**。如果两个模型的分数差小于一个标准差，我们倾向于选更简单的那个（比如逻辑回归），因为简单模型更稳、更好解释。

超参数方面，材料里没有做网格搜索，只用了默认参数加 `max_iter=1000`（保证逻辑回归收敛）。这是一个合理的起点：在匿名特征、样本量未知的情况下，盲目细调很容易过拟合验证集。如果时间允许，我们的下一步会是给 LightGBM 调 `num_leaves`、`learning_rate`、`n_estimators`，给随机森林调 `max_depth`、`min_samples_leaf`，并且始终用交叉验证分数而不是单次验证分数来选。

### 2.4 模型测试与指标报告

训练完五个模型后，我们按验证 ROC-AUC 排序，取最好的一个：

```python
# 整理成表格并按验证分数降序
results_df = pd.DataFrame([
    {'模型': name,
     '验证 ROC-AUC': r['val_auc'],
     '交叉验证 ROC-AUC': f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}"}
    for name, r in results.items()
]).sort_values('验证 ROC-AUC', ascending=False)

print(results_df)

# 选出最好的模型
best_name = results_df.iloc[0]['模型']
best_model = trained_models[best_name]
print('最佳模型：', best_name)
```

接着画 ROC 曲线和混淆矩阵。ROC 曲线越靠左上角越好；混淆矩阵则告诉我们具体错在哪一类：

```python
from sklearn.metrics import roc_curve, confusion_matrix, classification_report

# 验证集上的正类概率
y_val_proba = best_model.predict_proba(X_val)[:, 1]

# ROC 曲线
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
plt.plot(fpr, tpr, label=f'{best_name} (AUC={results[best_name]["val_auc"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='random (AUC=0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# 默认 0.5 阈值下的混淆矩阵和分类报告
y_val_pred = best_model.predict(X_val)
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))
```

这里要提醒：ROC-AUC 高不代表 0.5 阈值下的分类效果好。AUC 只看排序，而混淆矩阵看的是固定阈值下的硬判决。如果业务上更在意「少漏掉风险合同」，可以把阈值调低，用召回率换精确率；如果更在意「别误伤好客户」，就把阈值调高。比赛只交概率，所以阈值不影响分数，但理解这一点对落地很重要。

最后我们对测试集输出概率并生成提交：

```python
# 取正类概率，而不是 0/1 标签
test_proba = best_model.predict_proba(test_data)[:, 1]

# 提交格式：id + target（概率）
submission = pd.DataFrame({
    'id': test_data['id'],
    'target': test_proba,
})
submission.to_csv('submission.csv', index=False)

# 检查：行数、空值、取值范围
print(submission.shape)
print(submission['target'].isna().sum())
print(submission['target'].min(), submission['target'].max())
```

材料里没有保留各模型的具体分数，所以这篇 writeup 不引用数字。我们报告的是完整的验证协议：分层 8:2 切分、5 折分层交叉验证、ROC-AUC 选模型、概率提交。按这套流程重跑，就能得到当时的排行榜结果。

---

## 三、总结

本文中我们完成了车险投保风险的二分类建模。我们首先读取 `insclass_train.csv`、`insclass_test.csv` 和提交样例，确认了数据规模、缺失情况和目标分布，并把二元编号特征从数值列里拆出来归入类别列。其次我们用 `ColumnTransformer` 搭了两套预处理：数值列中位数填补加标准化，类别列众数填补加 one-hot，并用 `handle_unknown='ignore'` 兜住测试集的新类别。接着我们比较了逻辑回归、随机森林、梯度提升、XGBoost、LightGBM 五种模型，每个模型都同时看 8:2 验证分数和 5 折交叉验证分数。最后我们按验证 ROC-AUC 选出最佳模型，画出 ROC 曲线和混淆矩阵，并对测试集输出正类概率生成提交。

这个比赛让我们更清楚地理解了 ROC-AUC：它衡量的是排序能力，不是概率校准，也不是固定阈值下的准确率。也正因为如此，提交概率比提交硬标签更合适——把「怎么卡阈值」留给业务方，模型只负责把风险从低到高排好。另一条经验是，面对匿名特征，不要急着做复杂的特征工程，先把缺失、类型、验证策略这三件事做扎实，往往比多造几个特征更有效。
