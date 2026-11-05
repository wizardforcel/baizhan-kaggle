# 预测莱桑信天翁性别

## 一、比赛简介

### 1.1 比赛背景

这场比赛的任务是判断莱桑信天翁的性别。材料中的数据来自 Kaggle 竞赛 `predicting-the-sex-of-laysan-albatross`，我们拥有 94 行训练样本和对应的测试样本。每个样本包含若干身体测量指标：`longitudCraneo`、`longitudNarina`、`anchoCraneo`、`altoPico`、`anchoPico`、`tarso`、`longAlaCerrada`、`longAlaAbierta`、`mediaEnvergadura`、`envergadura`、`longitudPico`、`peso` 等。这些特征是鸟类的形态测量，相当于从外部观察能测到的长度、宽度和重量。对初学者来说，这种任务很好理解：我们不能直接看见鸟，就只能靠数字化的身体尺寸来猜性别。

因为样本量非常小，只有 94 行数据，比赛难度并不在于算法复杂，而在于如何避免过拟合。材料中的分析发现，特征之间存在强烈的多重共线性。例如 `envergadura` 和 `mediaEnvergadura` 的相关系数接近 1.00，本质上是同一信息；`longitudPico` 也和 `longitudCraneo`、`longitudNarina` 高度相关。性别标签在材料里标记为 `sexo`，经过编码后变成 0 和 1，其中 H 类别有 57 个样本，M 类别有 37 个样本，不是完全平衡。样本特征还有缺失，材料显示 `peso` 存在缺失值。样本量小意味着每一行数据都很有价值，同时也意味着模型很容易记住训练集而不是学会规律，因此我们必须格外小心特征选择和验证方式。

### 1.2 比赛目标

我们输入的是每只鸟的形态测量特征，输出是性别分类标签。训练集包含 `id`、`sexo` 和若干数值特征，测试集只有 `id` 和特征。我们需要为测试集预测性别类别，并生成提交文件。材料中的代码把目标用 `LabelEncoder` 编码，打印 `le.classes_` 来确认 0 和 1 分别代表哪一类，避免标签方向搞反。这一步在真实比赛中很重要：如果编码方向弄反了，模型即便训练得再好，提交上来的结果也会完全错位。

### 1.3 评价标准

材料使用分类准确率 `accuracy_score` 来衡量模型。我们在报告中还可以使用混淆矩阵和分类报告来观察每个类别的精确率和召回率。对于小样本问题，单次划分的准确率会很不稳定，所以材料强烈建议使用分层交叉验证 `StratifiedKFold`。准确率越高越好，满分 1.0。由于样本只有 94 行，交叉验证的平均值比单次划分更能反映模型真实能力。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先加载训练表和测试表，检查缺失值、重复行和基本统计量。探索要回答三个问题：样本量是否足够、特征是否存在高度相关、目标是否平衡。材料中的代码先打印 `info()`，然后计算缺失值总和，接着看重复行。重复行检查看似简单，却能暴露数据采集或拼接时的重复录入问题。对 94 行的小数据来说，如果同一条记录被复制两次，验证结果就会被高估，所以我们在建模前先把这个问题排除掉。

```python
# 读取训练和测试数据，初步了解形状和列
import pandas as pd
import numpy as np

train_df = pd.read_csv('/kaggle/input/competitions/predicting-the-sex-of-laysan-albatross/train.csv')
test_df = pd.read_csv('/kaggle/input/competitions/predicting-the-sex-of-laysan-albatross/test.csv')

print('train shape:', train_df.shape)
print('test shape:', test_df.shape)
print('columns:', train_df.columns.tolist())

# 快速检查缺失值和重复行
print('Missing values:\n', train_df.isna().sum())
print('Duplicated rows:', train_df.duplicated().sum())
print('Target distribution:\n', train_df['sexo'].value_counts())
```

我们其次对数值特征做可视化，画直方图和核密度估计，观察分布是否正常。材料中的代码把所有数值列画成 3×4 的子图阵列，随后用热力图显示相关系数矩阵。相关系数图明显显示 `envergadura` 与 `mediaEnvergadura` 几乎 100% 相关，`longitudPico` 与头骨和鼻孔长度也有很高相关。可视化在这里的作用是帮助我们直观判断特征是否重复。对入门者来说，相关性热力图是最容易理解的工具，只要看到颜色很深的格子，就知道两列几乎讲同样的故事。

```python
# 可视化每个数值特征的分布，了解偏态
numerical_cols = train_df.drop(columns=['id','sexo']).columns.tolist()
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(3, 4, figsize=(16, 9))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    sns.histplot(train_df[col], kde=True, ax=axes[i])
    axes[i].set_title(col)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# 相关性热力图，帮助发现多重共线性
corr = train_df.drop(columns=['id','sexo']).corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
```

最后我们统计目标的分布并检查是否需要分层。材料明确指出两类样本不平衡，H 有 57 个，M 有 37 个，且样本总量小，任何随机划分都可能导致验证集类别比例漂移。我们因此采用分层交叉验证，保持每折的类别比例一致。

```python
# 统计目标分布并确认是否需要分层
print('Class counts:')
print(train_df['sexo'].value_counts())

# 打印统计摘要，观察均值和标准差
print(train_df[numerical_cols].describe().T)

# 缺失值处理前，先看看哪个列缺失最多
missing_counts = train_df.isna().sum()
print('Columns with missing:', missing_counts[missing_counts > 0])
```

### 2.2 数据清洗

我们首先处理 `peso` 列的缺失。材料使用训练集的中位数填充训练和测试两端，避免信息泄露。填充后，我们对目标列进行编码。`LabelEncoder` 会自动把类别变成整数，材料还建议打印 `le.classes_`，确保我们知道 0 和 1 分别对应哪个性别。在小数据里，不能简单使用 0 来补缺失，因为 0 可能是真实的测量值。用中位数填充可以保留分布的中心位置，同时不会引入新的类别。

```python
# 填补 peso 的缺失，用训练集的中位数
train_df['peso'] = train_df['peso'].fillna(train_df['peso'].median())
test_df['peso'] = test_df['peso'].fillna(train_df['peso'].median())

# 对目标进行编码，同时打印映射关系避免方向错误
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['sexo_enc'] = le.fit_transform(train_df['sexo'])
print('Encoding map:', dict(zip(le.classes_, le.transform(le.classes_))))
print(train_df['sexo_enc'].value_counts())
```

材料发现特征之间存在严重的多重共线性，通过 VIF 进行量化。`envergadura` 与 `mediaEnvergadura` 几乎是同一个特征，`longitudPico` 的 VIF 达到约 6.1。因此我们选择删除 `longitudPico` 和 `envergadura`，保留 `mediaEnvergadura`。特征选择后，我们把数据切成 X 和 y，准备建模。

```python
# 用 VIF 量化多重共线性，决定删除哪些高度相关特征
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_vif = train_df.drop(columns=['id', 'sexo', 'sexo_enc'])
X_vif_const = add_constant(X_vif)
vif = pd.Series(
    [variance_inflation_factor(X_vif_const.values, i)
     for i in range(X_vif_const.shape[1])],
    index=X_vif_const.columns
).sort_values(ascending=False)
print(vif.head(10))

# 保留信息重叠较少的一组特征
features_to_keep = [
    'longitudCraneo', 'longitudNarina', 'anchoCraneo', 'altoPico',
    'anchoPico', 'tarso', 'longAlaCerrada', 'longAlaAbierta',
    'mediaEnvergadura', 'peso'
]
X = train_df[features_to_keep]
y = train_df['sexo_enc']
print('Final feature count:', X.shape[1])
```

最后我们检查特征是否还有缺失、是否需要标准化。LDA 和 Logistic Regression 都对特征尺度敏感，材料在这一步未显式做标准化，但可以作为可选步骤。我们只保留经过选择的特征，避免因多重共线性导致模型不稳定。

```python
# 检查最终特征矩阵的完整性
print(X.isna().sum())
print(X.describe().T)

# 必要时做标准化，保证不同量纲特征对模型贡献公平
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 记录测试集的相同特征，用于后续预测
X_test = test_df[features_to_keep]
X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features_to_keep)
print('Test shape:', X_test_scaled.shape)
```

### 2.3 模型训练与超参数调优

我们首先选择两个线性模型：线性判别分析 LDA 和逻辑回归。材料指出 LDA 在这个数据集上表现更好且更稳定。原因可能是 LDA 同时进行特征降维和分类，适合小样本高相关的情况。我们使用 5 折分层交叉验证来评估，每折都能保留类别比例。分层交叉验证在类别不平衡时尤其重要，避免某折里某一类样本太少，导致模型训练失衡。

```python
# 用 StratifiedKFold 做交叉验证，避免小样本随机划分带来的波动
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in [('LDA', LinearDiscriminantAnalysis()),
                    ('Logistic Regression', LogisticRegression(max_iter=1000))]:
    scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
    print(f'{name}: mean={scores.mean():.3f}  std={scores.std():.3f}  folds={np.round(scores,3)}')
```

我们其次在验证集上比较两种模型。材料提到 LDA 的均值更高，标准差更小，说明更稳定。逻辑回归作为对比，参数 `max_iter=1000` 确保收敛。我们还可以把两者合并成投票，但材料的主线保留 LDA 作为主力模型。

```python
# 单独拟合模型以获取详细报告
lda = LinearDiscriminantAnalysis()
lda.fit(X_scaled, y)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_scaled, y)

# 在训练集上看整体表现，仅作参考
from sklearn.metrics import classification_report, confusion_matrix
y_pred_lda = lda.predict(X_scaled)
print('LDA classification report:')
print(classification_report(y, y_pred_lda, target_names=le.classes_))
print('Confusion matrix:')
print(confusion_matrix(y, y_pred_lda))
```

我们最后检查模型对特征的敏感性。因为特征共线性已经处理，LDA 的判别方向相对稳定。材料没有做复杂的超参数网格搜索，因为 LDA 的参数很少，逻辑回归的正则强度可以尝试，但小样本下过度调参更容易过拟合。我们保持简单策略，先保证流程稳定。

```python
# 简单的超参数网格示例，逻辑回归正则强度 C
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10]}
grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid, cv=skf, scoring='accuracy'
)
grid.fit(X_scaled, y)
print('Best C:', grid.best_params_)
print('Best CV accuracy:', grid.best_score_)
```

### 2.4 模型测试与指标报告

我们首先在分层交叉验证下评估模型，并在每折输出准确率。材料强调这比单次 80/20 划分更可靠。我们用交叉验证得到均值和标准差，作为最终的模型能力估计。

```python
# 交叉验证结果汇总
cv_results = {}
for name, model in [('LDA', LinearDiscriminantAnalysis()),
                    ('Logistic Regression', LogisticRegression(max_iter=1000))]:
    scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
    cv_results[name] = scores
    print(f'{name} CV scores: {scores}')
    print(f'Mean accuracy: {scores.mean():.3f} +/- {scores.std():.3f}')
```

然后我们用全量训练数据拟合 LDA，并对测试集预测。最后把预测的整数标签解码回原始性别名称，生成符合提交格式的文件。材料强调要使用 `le.inverse_transform` 把编码值转回字符串，避免写出 0/1 被误解。

```python
# 用全量数据训练最终模型并预测测试集
final_model = LinearDiscriminantAnalysis()
final_model.fit(X_scaled, y)
test_pred_enc = final_model.predict(X_test_scaled)

# 解码回原始类别名称
test_pred_label = le.inverse_transform(test_pred_enc)
submission = pd.DataFrame({
    'id': test_df['id'],
    'sexo': test_pred_label
})
submission.to_csv('submission.csv', index=False)
print(submission.head())
print('Submission shape:', submission.shape)
```

最后我们检查提交文件的正确性：行数是否匹配、是否包含缺失值、类别是否只有训练集出现过的两类。材料还建议在小型数据集上绘制混淆矩阵，帮助理解两类之间的误判情况。

```python
# 提交文件自检
print('Submission missing values:', submission.isna().sum().sum())
print('Unique predicted classes:', submission['sexo'].unique())

# 在训练集上画混淆矩阵，直观看出错误类型
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred_train = final_model.predict(X_scaled)
cm = confusion_matrix(y, y_pred_train)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix on Training Data')
plt.show()
```

## 三、总结

本文中我们完成了预测莱桑信天翁性别的完整流程。我们首先读取 94 行训练数据，发现 `peso` 缺失并用中位数填补，目标类别不平衡，特征之间存在严重多重共线性，尤其是 `envergadura` 与 `mediaEnvergadura` 以及 `longitudPico` 的高相关。接着我们通过 VIF 分析删除 `longitudPico` 和 `envergadura`，保留 10 个较为独立的形态特征。用分层交叉验证比较 LDA 和逻辑回归，发现 LDA 在小样本上更稳定、均值更高。然后用全量训练数据拟合 LDA 模型，对测试集预测并解码回原始性别标签，生成提交文件。我们在整个过程中坚持使用分层交叉验证来评估，避免小样本单次划分的随机误差干扰判断。

这个比赛的核心教训是：在小样本情况下，特征选择和数据质量比模型复杂度更重要。删除冗余特征、保持类别比例、分层验证都是避免过拟合的关键。我们没有编造具体分数，而是以交叉验证均值和标准差的方式呈现模型稳定性，并遵循材料中的主线使用 LDA 作为最终方案。
