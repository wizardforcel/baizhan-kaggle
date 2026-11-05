# 普鲁登特寿险风险评估

## 一、比赛简介

### 1.1 比赛背景

普鲁登特（Prudential）是美国最大的寿险公司之一。传统寿险核保流程非常繁琐：投保人需要填写大量问卷、提供就业和家族病史、甚至安排体检，整个流程平均要花 30 天。在「一键下单、即时满足」的消费习惯下，这种体验让很多人望而却步。数据显示，只有约 40% 的美国家庭拥有个人寿险，其中一个重要原因就是投保过程太慢、太麻烦。

普鲁登特希望用数据驱动的方法改善这一点：如果能够根据申请人已经提供的信息，快速、准确地判断她/他的风险等级，就能大幅缩短核保时间，同时保持隐私边界。这个比赛就是把真实的核保问题抽象成一个机器学习任务——给一批脱敏的申请人特征，预测风险等级。

比赛数据包含超过一百个变量，全部经过归一化处理，所以你看不到原始的身高体重数字，只能看到 0 到 1 之间的相对值。这些变量可以分成几组：产品信息（Product_Info）、年龄身高体重（Ins_Age、Ht、Wt、BMI）、就业信息（Employment_Info）、受保人信息（InsuredInfo）、保险历史（Insurance_History）、家族史（Family_Hist）、医疗史（Medical_History）和医疗关键词（Medical_Keyword）。目标 `Response` 是一个 8 级序数标签，表示最终的风险决策等级。

这里的「序数」很关键：8 个等级不是随便排列的类别，而是有大小顺序的，1 级和 2 级的距离，理论上比 1 级和 8 级更近。但比赛的评价指标是多分类对数损失，它把 8 个等级当成彼此独立的类别，并不利用顺序信息。这给建模带来一个有趣的取舍：是用多分类模型直接优化指标，还是用回归模型利用顺序信息、再把连续输出映射回类别。

### 1.2 比赛目标

给定投保人的特征，我们需要预测她/他的风险等级 `Response`。训练集包含 `Id` 和 `Response`，测试集只有 `Id` 和特征，提交文件需要为每一行预测 8 个类别中的一个。特征分组如下：

- Product_Info_1-7：产品信息
- Ins_Age、Ht、Wt、BMI：年龄身高体重
- Employment_Info_1-6：就业相关
- InsuredInfo_1-6：受保人信息
- Insurance_History_1-9：保险历史
- Family_Hist_1-5：家族史
- Medical_History_1-41：医疗史
- Medical_Keyword_1-48：医疗关键词虚拟变量

我们先把训练集读进来，确认列数和目标列：

```python
# 读取训练集，把 Id 设为索引，避免它被当成特征
insurance_df = pd.read_csv('../input/prudential-life-insurance-assessment/train.csv.zip',
                           index_col='Id')
print(insurance_df.shape)       # 一百多列，最后一列是 Response
print(insurance_df['Response'].value_counts().sort_index())
```

### 1.3 评价标准

比赛在 Kaggle 上使用多分类对数损失（logloss）作为评估指标，logloss 越小越好。它的计算方式是：模型对每个样本输出 8 个类别的概率，取真实类别对应的那个概率，取对数，加负号，再对所有样本求平均。如果模型对真实类别给出很高的概率，logloss 就小；如果给出很低的概率，logloss 就会非常大。

logloss 有一个重要特点：它不仅惩罚「猜错类别」，更惩罚「过度自信地猜错」。一个模型如果把某个样本 99% 的概率押在错误类别上，会遭到极重的惩罚。所以优化 logloss 实际上是在要求模型给出**校准良好**的概率，而不是只追求准确率。

排行榜上还常以准确率、精确率、召回率、ROC AUC 等辅助指标展示模型在不同类别上的表现，因为 8 个风险等级分布极不均衡，只看准确率会被多数类掩盖。例如，如果模型把所有样本都预测成最多的那一类，准确率可能不低，但对少数高风险人群完全失去识别能力，这在保险业务里是不可接受的。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先检查训练集的整体信息、缺失比例和目标分布：

```python
# 查看数据类型和非空数量
train.info()

# 统计每列缺失数量，按缺失从多到少排序
train.isnull().sum().sort_values(ascending=False)

# 看 8 个风险等级各有多少样本
train['Response'].value_counts().sort_index()
```

结果显示大量变量存在缺失，尤其是 `Medical_History` 和 `Medical_Keyword` 字段。这在保险问卷里很常见：不是每个申请人都会做同样的检查，也不是每个人都愿意填写完整的家族病史，所以缺失本身就是一种「信息未提供」的状态。

目标 `Response` 的分布并不均匀：中间等级样本多，两端等级样本少，最少的等级只有几百条，最多的有数千条。这种不平衡意味着，如果只看总体准确率，模型很容易偏向多数类；我们必须关注少数类的召回率，否则高风险申请人会被漏掉。

我们还绘制了各特征与目标的相关性热力图，并用缺失可视化库（missingno）观察缺失是否随机。如果缺失是随机分布的，用中位数填补问题不大；如果缺失和某些特征强相关（比如某个年龄段的人更少填某项），那就要考虑把「是否缺失」也作为一个特征。我们还计算了各等级下关键变量的均值，比如 BMI 是否随风险等级上升而上升，年龄分布是否在不同等级间有明显差异。

### 2.2 数据清洗

缺失值处理是本比赛的核心。我们先观察缺失比例，对不同类型的变量采用不同策略：数值型变量用中位数填补，类别型变量用众数填补：

```python
from sklearn.impute import SimpleImputer

# 数值列用中位数，类别列用众数
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
```

中位数比均值更抗极端值，适合医疗和保险这种有长尾的字段；众数则是类别列里最常见的取值，虽然粗糙，但不会凭空造出新类别。

更进一步，我们采用 MICE（多重插补）方法。MICE 的思路是：把每个有缺失的变量轮流当作「目标」，用其他变量去预测它，反复迭代若干轮，直到填补值稳定。这样填出来的值比简单中位数更贴近变量之间的关系：

```python
# 启用迭代插补器（sklearn 需要先导入 experimental 模块）
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# max_iter=10 表示最多迭代 10 轮
imputer = IterativeImputer(max_iter=10, random_state=0)
X_imputed = imputer.fit_transform(X_num)
```

填补后我们再次检查特征分布，确认没有出现不合理的负数或超出范围的取值。对类别特征（比如 `Product_Info_2` 这种字母加数字的编码）我们做了独热编码或序数编码；对高度相关的医疗历史变量，部分方案用 PCA 做了降维，减少冗余和共线性。

值得一提的是，有些方案还用孤立森林（Isolation Forest）或方差分析（ANOVA）做特征筛选，剔除信息量极低的变量。这一步不是必须的，因为树模型对无关特征有一定鲁棒性，但减少特征可以加快训练、降低过拟合风险。

### 2.3 模型训练与超参数调优

我们采用分层抽样划分训练/验证集，保证 8 个等级的比例在两边一致。首先用随机森林做基线，因为它对缺失和非线性关系都比较稳，而且能直接输出特征重要性：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# stratify=y 保证验证集的类别比例和训练集一致
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

rf = RandomForestClassifier(
    n_estimators=500,      # 500 棵树，足够稳定
    max_depth=None,        # 先不限制深度
    min_samples_leaf=1,
    n_jobs=-1,             # 用满所有 CPU 核心
    random_state=42
)
rf.fit(X_train, y_train)
```

随后我们对随机森林和 XGBoost 做网格搜索和随机搜索，调整 `n_estimators`、`max_depth`、`min_child_weight`、`subsample`、`colsample_bytree` 等超参数，用交叉验证的 logloss 选最优。XGBoost 的优点是训练快、正则化强，在类别不平衡的表格数据上通常比随机森林更准。

我们还尝试了两条更「重」的路线。一条是 Fastai 的神经网络：把表格特征做嵌入和标准化后喂给多层感知机，用带权重的损失函数缓解类别不平衡。另一条是遗传编程（Genetic Programming）：让程序自动搜索特征组合和变换，在排行榜上取得约 0.662 的成绩。遗传编程的思想是模拟生物进化——随机生成一批「特征表达式」，保留表现好的，交叉变异产生下一代，多代之后可能找到人工想不到的组合。不过它计算量大、可解释性差，更适合作为探索而不是主力方案。

最终我们采用集成策略：把随机森林、XGBoost 和 LightGBM 的预测概率平均：

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 三个基学习器，分别捕捉不同的偏差
clf1 = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
clf2 = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, random_state=42)
clf3 = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)

for clf in [clf1, clf2, clf3]:
    clf.fit(X_train, y_train)

# 概率平均：三个模型各投一票，比单模型更稳
proba = (clf1.predict_proba(X_valid) +
         clf2.predict_proba(X_valid) +
         clf3.predict_proba(X_valid)) / 3
```

概率平均之所以有效，是因为三个模型的错误不完全相同：XGBoost 和 LightGBM 都是提升树，但分裂策略和正则不同；随机森林是装袋法，偏差方向又不一样。把它们平均，能抵消一部分随机误差。

我们还借鉴了 SHAP 做模型解释，找出对风险等级影响最大的特征。SHAP 的核心思想来自博弈论：把每个特征看成「合作者」，计算它给最终预测贡献了多少。结果显示 `Medical_History` 系列、`Ins_Age`、`BMI` 等特征贡献最大，这和保险业务常识一致——既往病史和年龄体重本来就是核保最看重的因素。在这个基础上我们做了一轮特征选择，去掉信息量极低的变量，再重新训练集成模型。

### 2.4 模型测试与指标报告

在验证集上，我们计算 logloss、准确率、召回率和 ROC AUC，并输出每个类别的分类报告：

```python
from sklearn.metrics import log_loss, classification_report, roc_auc_score

# 用集成后的概率计算 logloss
print("Log Loss:", log_loss(y_valid, proba))

# 每个类别的精确率、召回率、F1
y_pred = proba.argmax(axis=1)
print(classification_report(y_valid, y_pred))

# 多分类 ROC AUC，平均方式用 macro
print("ROC AUC:", roc_auc_score(y_valid, proba, multi_class='ovr', average='macro'))
```

集成模型在验证集上的 logloss 明显优于任何单模型，对少数类的召回率也有提升。我们绘制了混淆矩阵，发现主要混淆集中在相邻风险等级，这是序数标签的自然现象：模型很难把 3 级和 4 级完全分开，但很少把 1 级和 8 级搞混。这个观察也提示我们，如果比赛允许，用回归加阈值映射的方法可能能利用顺序信息，但既然指标是多分类 logloss，我们还是以概率输出为准。

在全量训练集上重新拟合后，对测试集预测类别标签并生成提交文件，同时保存特征重要性图和 SHAP 摘要图，方便业务方理解模型依据：

```python
# 用全部训练数据重新训练三个模型
for clf in [clf1, clf2, clf3]:
    clf.fit(X, y)

# 测试集概率平均，取最大概率对应的类别
test_proba = (clf1.predict_proba(test) +
              clf2.predict_proba(test) +
              clf3.predict_proba(test)) / 3
test_pred = test_proba.argmax(axis=1)

# 按比赛格式生成提交文件
submission = pd.DataFrame({'Id': test_ids, 'Response': test_pred})
submission.to_csv('submission.csv', index=False)
```

提交前我们检查了行数、ID 顺序、类别取值是否都在 1 到 8 之间，避免因为格式问题白白丢分。

## 三、总结

本文中我们完成了普鲁登特寿险风险评估的完整流程。我们首先用 EDA 摸清了目标不平衡、缺失严重、特征分组的现状，并分析了不同风险等级下关键变量的差异；其次采用中位数/众数填补处理基础缺失，再用 MICE 多重插补利用变量间关系提升数据质量，并对类别特征编码、对高相关变量做 PCA 降维；接着用随机森林、XGBoost、LightGBM 构建基线，通过网格/随机搜索调参，尝试 Fastai 神经网络和遗传编程搜索特征组合，最终用概率平均做集成；然后在验证集上用 logloss、准确率、召回率和 ROC AUC 评估，并结合 SHAP 解释关键特征；最后在全量数据上训练生成提交文件。

这个比赛告诉我们，医疗和保险类表格数据的核心挑战是缺失和类别不平衡。合理的插补能保留更多信息，集成策略能提升稳定性，而模型解释则让业务方敢用模型。对我们初学者来说，这个比赛是一个很好的综合练习：它把数据清洗、特征工程、模型集成和结果解释串成了一条完整的链路，也让我们第一次认真思考「预测概率校准」和「类别顺序信息」这些进阶话题。