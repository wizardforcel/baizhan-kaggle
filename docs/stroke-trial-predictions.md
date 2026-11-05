# Stroke Trial Predictions（卒中试验预测）

## 一、比赛简介

### 1.1 比赛背景

卒中是导致死亡和残疾的主要原因之一，临床试验中对患者 6 天内的生存情况进行预测具有重要的医学价值。该比赛的数据来源于临床试验记录，包含患者的基线特征和随访信息。参赛者需要利用训练集中的历史数据学习患者死亡风险的规律，并在未公开的测试集上给出预测。

材料中提到本项目由 Abdullah Gul University 支持，作为 COMP 468 Machine Learning 课程的期末项目。数据文件为 Train1.csv 和 Test1.csv，训练集中包含目标列 DEAD6，表示患者是否在 6 天内死亡。特征中还存在 DEAD1、DEAD24、DEAD48 等时间点的死亡标记，这些信息在实际预测中属于未来信息，不能用于训练。

这个任务有几个典型难点。第一，临床数据的缺失很常见，不同患者的检查项目并不完全一致，直接丢掉缺失行会浪费大量样本，需要合适的填充策略。第二，特征中既有年龄这样的数值列，也有类别型变量，两种列必须分别处理，不能直接套同一个变换。第三，短期内死亡的患者在样本中通常占少数，属于类别不平衡问题，如果不做处理，模型会倾向于把所有人都预测成“存活”。第四，随访时间点的死亡标记与目标高度相关，一旦不小心把它们放进特征，模型在本地验证时表现会很好，但到了测试集就失效，这种问题叫数据泄露，是本次比赛最需要警惕的坑。

### 1.2 比赛目标

给定患者的临床特征，预测其在试验期间是否死亡。训练集的列包括样本标识 ID、年龄 AGE 等临床变量，以及目标列 DEAD6。测试集包含同样的特征列但不含目标，需要我们给出 PatientDied 的预测。

训练集中的关键列可以大致分为三类：标识列 ID 用于对齐样本；基线特征如年龄 AGE、实验室指标和治疗信息等用于预测；随访死亡标记 DEAD1、DEAD8、DEAD24、DEAD48 和目标 DEAD6。后者都与“死亡”相关，其中 DEAD1 等是发生在预测时间点之后的信息，不能使用。测试集通常和训练集保持相同特征结构，只是没有目标列，需要我们在同样的特征上进行预测。预测结果的提交格式需要与样例一致，ID 列必须严格对齐，目标列用字符 Y 表示死亡、N 表示存活。

在代码中，我们先将目标列 DEAD6 重命名为 PatientDied，便于理解。最终的提交格式要求用 ‘Y’ 表示死亡，‘N’ 表示存活，并保持与官方样例提交文件相同的 ID 顺序。

### 1.3 评价标准

在本地验证中，代码使用了多种分类指标：准确率 accuracy、精确率 precision、召回率 recall、F1 分数 f1_score 以及 ROC-AUC roc_auc_score。模型训练阶段的网格搜索使用 roc_auc 作为优化目标，这说明在医学预测场景中，更关注模型区分存活与死亡的能力，而不仅仅是整体准确率。

准确率是预测正确的样本占全部样本的比例，直观但在类别不平衡时会失真；精确率衡量“预测为死亡的人里有多少真的死亡”；召回率衡量“真正死亡的人里有多少被找出来”；F1 是精确率和召回率的调和平均，在两者之间做平衡。ROC-AUC 则可以理解为：随机抽一个死亡患者和一个存活患者，模型给死亡患者更高风险分的概率，取值越接近 1 说明区分能力越强。在临床场景中，漏掉一个高风险患者通常比多提醒一次更严重，因此召回率和 ROC-AUC 值得特别关注，网格搜索也正是以 ROC-AUC 做优化。

混淆矩阵和 ROC 曲线被用于可视化比较 Logistic Regression、Random Forest 和 XGBoost 三个模型的表现。

## 二、解题过程

整场比赛的思路可以概括为：先了解数据分布和缺失情况，再剔除数据泄露列并做常规预处理，然后用多个模型对比并对 XGBoost 做超参数调优，最后在验证集上评估并生成提交文件。我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先读取训练数据并查看基本信息，了解列的类型和样本数量。

```python
import pandas as pd
import numpy as np
# 读取训练集和测试集，指定编码避免乱码
train_df = pd.read_csv('/kaggle/input/competitions/stroke-trial-predictions/Train1.csv', encoding='latin1')
test_df = pd.read_csv('/kaggle/input/competitions/stroke-trial-predictions/Test1.csv', encoding='latin1')
# 将目标列重命名为更易读的名字
train_df = train_df.rename(columns={'DEAD6': 'PatientDied'})
```

接下来我们打印数据集概览并分析缺失值，目标变量的分布以及年龄与死亡的关系。

```python
print(train_df.info())
# 计算每列缺失值个数和占比
missing_values = train_df.isnull().sum()
missing_percentage = (missing_values / len(train_df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing_values, 'Percentage (%)': missing_percentage})
print(missing_df.sort_values(by='Missing Count', ascending=False).head(10))
# 查看目标变量的类别计数和占比
target_counts = train_df['PatientDied'].value_counts()
target_pct = train_df['PatientDied'].value_counts(normalize=True) * 100
for idx in target_counts.index:
    print(f"Class {idx}: {target_counts[idx]} rows ({target_pct[idx]:.2f}%)")
```

在可视化部分，我们绘制了目标分布计数图、年龄与死亡的箱线图以及数值特征的相关性热力图，帮助直观看出类别不平衡、年龄对死亡的可能影响以及特征间的相关性。信息概览能让我们知道每列的数据类型和样本数，缺失值统计帮助决定后续是否需要填充和选择合适的填充策略。目标分布的打印结果直接显示了正负类占比，是判断是否需要类别加权的依据。箱线图可以看出死亡组的年龄中位数是否显著高于存活组，相关性热力图则提醒我们避免使用彼此高度相关的冗余特征。这些探索为后续的缺失值处理和建模提供了依据，也让我们对数据泄露问题保持警觉。

### 2.2 数据清洗

我们其次进行数据清洗，重点是剔除数据泄露列并准备数值与类别特征的预处理流程。

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

target_column = 'PatientDied'
# DEAD1、DEAD8、DEAD24、DEAD48 属于未来信息，必须剔除
leakage_columns = ['DEAD1', 'DEAD8', 'DEAD24', 'DEAD48']
# 准备特征矩阵，去掉目标、泄露列和ID
X = train_df.drop(columns=[target_column] + leakage_columns + ['ID'], errors='ignore')
y = train_df[target_column]
# 测试集同样去掉 ID
X_kaggle_test = test_df.drop(columns=['ID'], errors='ignore')
```

这里最关键的判断是哪些列不能作为特征。DEAD1、DEAD8、DEAD24、DEAD48 分别是第 1、8、24、48 天（或对应随访点）的死亡标记，它们和 DEAD6 描述的是同一件事的不同时间切片，留作特征等于把答案提前告诉模型。ID 只是行号，没有预测意义，也要去掉。使用 errors='ignore' 是为了在列名不完全一致时不报错，保证流程更稳健。

然后自动区分数值列和类别列，构建统一的预处理流程。

```python
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 数值特征：中位数填充 + 标准化
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# 类别特征：众数填充 + 独热编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])
```

清洗后，数据泄露被避免，缺失值通过中位数和众数填充，数值特征被标准化，类别特征被编码，统一进入模型训练。数值特征使用中位数填充是对异常值更稳健的选择，避免均值受极端值影响；标准化让不同量纲的指标处于同一尺度，有利于基于距离的模型。类别特征用众数填充缺失，然后独热编码避免把类别当成有序数字。ColumnTransformer 把两者整合进一个步骤，保证在交叉验证时训练集的统计量不会泄漏到验证集，Pipeline 再把预处理和模型绑定在一起，避免手动转换时出错。

### 2.3 模型训练与超参数调优

接着我们做分层训练验证集划分，保证死亡与存活的比例在训练集和验证集保持一致。

```python
from sklearn.model_selection import train_test_split
# 20% 作为验证集，stratify 保持类别比例
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```

分层抽样很重要，因为正类样本较少，随机的划分可能导致验证集里正类过少，指标波动很大。固定 random_state 让结果可复现。划分完成后，特征和标签保持一一对应，为后面的模型训练做好准备。

随后我们同时训练三个基准模型：Logistic Regression、Random Forest 和 XGBoost，并把预处理与模型封装进 Pipeline。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

log_reg_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor), ('model', log_reg_model)])
pipeline_lr.fit(X_train, y_train)

rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor), ('model', rf_model)])
pipeline_rf.fit(X_train, y_train)

xgb_model = XGBClassifier(scale_pos_weight=5, eval_metric='logloss', random_state=42)
pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb_model)])
pipeline_xgb.fit(X_train, y_train)
```

三个模型各有分工。Logistic Regression 是线性基线，训练快、可解释性强；Random Forest 通过多棵决策树投票，能捕捉非线性和特征交互；XGBoost 是梯度提升树，通常在这种表格型临床数据上表现较强。Logistic Regression 和 Random Forest 都设置了 class_weight='balanced'，让少数类在损失函数中占更大权重；XGBoost 用 scale_pos_weight=5 提高正类权重，缓解类别不平衡。三者都使用同一个 preprocessor，保证比较公平。

然后对 XGBoost 进行超参数调优，使用 GridSearchCV 在 3 折分层交叉验证下搜索最佳参数。

```python
from sklearn.model_selection import GridSearchCV
param_grid_xgb = {
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1],
    'model__n_estimators': [100, 200]
}
grid_search_xgb = GridSearchCV(
    estimator=pipeline_xgb,
    param_grid=param_grid_xgb,
    cv=3,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)
grid_search_xgb.fit(X_train, y_train)
best_pipeline_xgb = grid_search_xgb.best_estimator_
```

网格搜索以 ROC AUC 为目标，最终得到最优参数组合的 XGBoost 流水线，作为后续提交的主模型。参数网格覆盖了树的深度、学习率和树的数量：max_depth 控制每棵树的复杂度，太小会欠拟合，太大会过拟合；learning_rate 控制每棵树的贡献，学习率小通常需要更多树；n_estimators 是树的数量。3 折分层交叉验证在计算成本和估计稳定性之间取了折中，n_jobs=-1 则让搜索并行跑满 CPU。需要注意，网格搜索是套在完整 Pipeline 上的，所以每一折的预处理都只在训练折上拟合，验证折保持干净。

### 2.4 模型测试与指标报告

然后我们在本地验证集上对三个模型进行全面评估，计算准确率、精确率、召回率、F1 和 ROC-AUC，并绘制混淆矩阵和 ROC 曲线。

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve

models_dict = {
    'Logistic Regression (Baseline)': pipeline_lr,
    'Random Forest (Comparison)': pipeline_rf,
    'XGBoost (Tuned Best Model)': best_pipeline_xgb
}

for model_name, current_pipeline in models_dict.items():
    predictions = current_pipeline.predict(X_val)
    probabilities = current_pipeline.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, predictions)
    prec = precision_score(y_val, predictions, zero_division=0)
    rec = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    auc = roc_auc_score(y_val, probabilities)
    print(f"{model_name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={auc:.4f}")
```

评估结果汇总成表格，并通过可视化比较不同模型的混淆矩阵和 ROC 曲线，帮助判断召回率在医学场景中的重要性。打印的指标让三个模型一目了然，混淆矩阵可以看出误报和漏报的具体数量，ROC 曲线则展示了在不同阈值下召回率和误报率的权衡。代码把调优后的 XGBoost 作为最佳候选，但最终提交时仍然保留概率输出，并通过阈值 0.5 映射到 Y/N 标签，以符合比赛的提交格式要求。整个流程强调可复现性：随机种子固定、分层划分、Pipeline 封装，保证别人跑同一份代码得到的结果与我们一致。

最后我们使用调优后的 XGBoost 在全量训练数据上重新拟合，生成测试集预测并保存提交文件。

```python
best_pipeline_xgb.fit(X, y)
# 对齐官方样例的 ID 顺序并生成特征矩阵
test_indexed = test_df.set_index('ID')
X_final_test = test_indexed.reindex(final_official_sample['ID']).reset_index()
X_final_features = X_final_test.drop(columns=['ID'], errors='ignore').reindex(columns=X.columns, fill_value=0)

probabilities = best_pipeline_xgb.predict_proba(X_final_features)[:, 1]
final_labels = ['Y' if prob >= 0.5 else 'N' for prob in probabilities]

submission_df = pd.DataFrame({
    'ID': final_official_sample['ID'],
    'PatientDied': final_labels
})
submission_df.to_csv('submission.csv', index=False)
```

再次拟合全量数据是为了让模型利用全部信息，验证集只用于调参和评估。在生成提交文件时，必须严格对齐官方样例的 ID 顺序，否则即使预测完全正确也会因行序错位被判错。我们先用 reindex 把测试集按样例 ID 排序，再用相同的列顺序生成特征矩阵，缺失的列用 0 填充。概率大于 0.5 时记为 ‘Y’，否则记为 ‘N’，这也是提交文件中要求的标签形式。保存为 CSV 后，通常会再检查行数是否一致、列名是否正确，避免格式错误导致白白丢分。

我们首先完成 EDA，其次清洗并剔除数据泄露，接着训练三个模型并对 XGBoost 进行网格搜索调优，然后在验证集上做指标报告，最后用全量数据生成符合格式的提交文件。

## 三、总结

本文中我们完成了 Stroke Trial Predictions 比赛的完整流程。我们首先通过 EDA 了解了训练数据的缺失情况、目标变量分布以及年龄等特征与死亡的关系，其次剔除了 DEAD1、DEAD8、DEAD24、DEAD48 等数据泄露列并构建了数值与类别特征的统一预处理流水线。接着我们用 Logistic Regression、Random Forest 和 XGBoost 三个模型进行对比，并对 XGBoost 使用 GridSearchCV 在 max_depth、learning_rate、n_estimators 上进行 ROC-AUC 优化。然后我们在验证集上报告了准确率、精确率、召回率、F1 和 ROC-AUC，并可视化了混淆矩阵与 ROC 曲线。最后我们用调优后的模型在全量数据上训练并生成了 Y/N 格式的提交文件。

这个比赛提醒我们，在医学预测任务中，数据泄露的排查、类别不平衡的处理以及召回率的重视都比单纯追求准确率更关键。Pipeline 与分层抽样保证了评估的可靠性，网格搜索帮助我们在复杂度与性能之间找到平衡点，而严格的提交格式对齐则是把模型表现真正转化为分数的关键一步。从数据理解到提交格式的每个环节都需要耐心检查，模型分数只是整个工作中的一部分。
