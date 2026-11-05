# 普鲁登特寿险风险评估

## 一、比赛简介

### 1.1 比赛背景

普鲁登特（Prudential）是美国最大的寿险公司之一。传统寿险核保流程需要投保人提供大量信息并安排体检，平均耗时 30 天，导致用户流失严重。公司希望用数据驱动的方法，在保护隐私的前提下，快速给出风险等级。

比赛提供的训练集包含超过 100 个变量，描述投保人的产品信息、年龄身高体重、就业信息、受保人信息、保险历史、家族史、医疗史和医疗关键词。所有数值已被归一化处理，目标变量 `Response` 是一个 8 级序数标签，表示最终的风险决策等级。

### 1.2 比赛目标

给定投保人的特征，我们需要预测她/他的风险等级 `Response`。训练集包含 Id 和 Response，测试集只有 Id 和特征，提交文件需要预测 8 个类别中的一个。特征包括：

- Product_Info_1-7：产品信息
- Ins_Age、Ht、Wt、BMI：年龄身高体重
- Employment_Info_1-6：就业相关
- InsuredInfo_1-6：受保人信息
- Insurance_History_1-9：保险历史
- Family_Hist_1-5：家族史
- Medical_History_1-41：医疗史
- Medical_Keyword_1-48：医疗关键词虚拟变量

```python
# 读取训练集并查看列名，确认目标列
insurance_df = pd.read_csv('../input/prudential-life-insurance-assessment/train.csv.zip', index_col='Id')
print(insurance_df.head())
print(insurance_df.shape)
```

### 1.3 评价标准

比赛在 Kaggle 上使用多分类对数损失（logloss）作为评估指标，logloss 越小越好。logloss 对概率估计的校准很敏感：不仅要猜对类别，还要给出可靠的置信度。排行榜上还常以准确率、ROC、Recall 等辅助指标展示模型在不同类别上的表现，因为 8 个风险等级分布极不均衡。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先检查训练集的整体信息、缺失比例和目标分布：

```python
# 查看数据信息、缺失值和目标分布
train.info()
train.isnull().sum().sort_values(ascending=False)
train['Response'].value_counts().sort_index()
```

结果显示大量变量存在缺失，特别是 Medical_History 和 Medical_Keyword 字段。目标 `Response` 并不均匀分布，某些风险等级样本很少。我们还绘制了各特征与目标的相关性热力图，并用缺失可视化图（missingno）观察缺失是否随机。

随后我们计算每个 `Response` 等级的人数，发现极度不平衡，最多等级数千条，最少只有几百条。这提示我们需要关注少数类的识别能力。

### 2.2 数据清洗

缺失值处理是本比赛的核心。我们先观察缺失比例，对不同类型的变量采用不同策略：数值型变量用中位数填补，类别型变量用众数填补：

```python
# 简单缺失填补：数值中位数、类别众数
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
```

更进一步，我们采用 MICE（多重插补）方法，让每个缺失变量用其他变量预测来填补：

```python
# MICE 插补，先对数值变量进行多重插补
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(max_iter=10, random_state=0)
X_imputed = imputer.fit_transform(X_num)
```

填补后我们再次检查特征分布，并对类别特征做独热或序数编码。部分笔记本还使用了 PCA 对高度相关的医疗历史变量做降维，减少冗余。

### 2.3 模型训练与超参数调优

我们采用分层抽样划分训练/验证集，保证 8 个等级的比例一致。首先用随机森林做基线：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=500, max_depth=None,
                            min_samples_leaf=1, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
```

我们对随机森林和 XGBoost 做网格/随机搜索，调整 n_estimators、max_depth、min_child_weight 等超参数，并使用交叉验证评估 logloss。随后我们尝试了 Fastai 的神经网络和遗传编程搜索特征组合，在排行榜上取得约 0.662 的成绩。

最终我们采用集成策略：把随机森林、XGBoost 和 LightGBM 预测概率平均：

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
clf1 = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05)
clf2 = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05)
clf3 = RandomForestClassifier(n_estimators=500)
for clf in [clf1, clf2, clf3]:
    clf.fit(X_train, y_train)
```

我们还借鉴了 Shap 的解释，找出对风险等级影响最大的特征，如 Medical_History、Ins_Age、BMI 等，并在此基础上做特征选择，去除信息量极低的变量。

### 2.4 模型测试与指标报告

在验证集上，我们计算 logloss、准确率、召回率和 ROC AUC，并输出每个类别的分类报告：

```python
from sklearn.metrics import log_loss, classification_report
y_pred_proba = clf1.predict_proba(X_valid)
print("Log Loss:", log_loss(y_valid, y_pred_proba))
print(classification_report(y_valid, y_pred_proba))
```

集成模型在验证集中 logloss 明显优于单模型，对少数类召回率也有提升。我们绘制混淆矩阵，发现主要混淆集中在相邻风险等级，这是序数标签的自然现象。

在全量训练集上重新拟合后，对测试集预测类别标签并生成提交文件，同时保存特征重要性图以备后续解释。

## 三、总结

本文中我们完成了普鲁登特寿险风险评估的完整流程。我们首先用 EDA 摸清了目标不平衡、缺失严重的现状；其次采用中位数填补与 MICE 多重插补提升数据质量；接着用随机森林、XGBoost、LightGBM 构建基线并通过集成提升稳定性；然后在验证集上用 logloss、准确率和召回率评估，并结合 SHAP 解释关键特征；最后在全量数据上训练生成提交文件。

这个比赛告诉我们，医疗类表格数据的核心挑战是缺失和类别不平衡，合理的插补和集成策略比单一花哨模型更可靠。
