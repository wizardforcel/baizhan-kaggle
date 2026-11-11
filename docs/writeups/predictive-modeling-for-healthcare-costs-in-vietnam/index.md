# 越南医疗费用预测

## 一、比赛简介

### 1.1 比赛背景

越南医疗费用预测比赛来源于 Kaggle 上的 Predictive Modeling for Healthcare Costs in Vietnam 竞赛。医疗费用是个人、家庭和保险公司共同面对的现实问题：一次住院可能花掉一个家庭几个月的收入，而保险公司需要提前估算每个人未来可能产生的费用，才能合理设计保费和准备赔付资金。如果能够根据投保人的基本信息、就诊频率和历史记录，提前预测出她/他一年内大约要花多少医疗费，就能帮助保险机构做资源分配、帮助个人做财务规划，也能让公共卫生部门识别高风险人群。

这个比赛使用的数据是越南医疗保险体系中的脱敏样本。所谓脱敏，是指把姓名、身份证号这类可以直接识别个人的字段去掉或替换掉，只保留用于建模的统计特征。数据里既有数值型字段，也有性别、地区之类的类别字段，而且不少字段存在缺失。真实业务数据几乎不可能整整齐齐，所以这个比赛真正的难点并不是把模型堆得多深，而是：怎样在缺失严重、目标分布偏斜的情况下，做出一个稳定、可解释、不会因为个别极端值而崩掉的费用预测。

另一个容易被忽略的难点是目标变量本身。医疗费用通常呈现明显的右偏分布：大多数人花费很少甚至为零，少数重症患者花费极高。这种「长尾」会让平方误差被少数高值样本主导，模型容易学会「保守地预测一个中间值」，从而在高费用人群上系统性低估。我们在后面的 EDA 和残差分析里会反复回到这一点。

### 1.2 比赛目标

给定训练集中的一批特征列，我们要预测每位参保人的医疗费用目标值 `y`。训练集的结构大致是：第一列是样本编号 `ID`，中间是若干个人口学、就诊和费用相关特征，最后一列是目标 `y`。测试集只有 `ID` 和特征列，没有 `y`，我们需要为每一行填上预测值。

值得特别注意的是，训练集里还存在几个「只在训练集出现」的列，例如 `FRE_NGOAITRU`（门诊次数）、`FRE_NOITRU`（住院次数）、`COST_NGOAITRU`（门诊费用）、`COST_NOITRU`（住院费用）以及目标 `y`。这些列在测试集中并不存在。如果直接把训练集所有列都喂给模型，模型会「偷看」到测试时根本拿不到的信息，线下分数好看、线上一塌糊涂。所以我们在构造特征时必须先取训练集和测试集的列交集，只保留两边都有的字段。

```python
# 读取训练集和测试集，先确认数据形状
train = pd.read_csv('/kaggle/input/competitions/predictive-modeling-for-healthcare-costs-in-vietnam/train.csv')
test = pd.read_csv('/kaggle/input/competitions/predictive-modeling-for-healthcare-costs-in-vietnam/test.csv')
print("Train Dataset Shape:", train.shape)
print("Test Dataset Shape:", test.shape)

# 找出只存在于训练集的列：这些是目标或与目标直接相关的字段
# 绝不能当作特征使用，否则会造成严重的标签泄漏
only_in_train = [c for c in train.columns if c not in test.columns]
print("只在训练集出现的列：", only_in_train)
```

### 1.3 评价标准

比赛使用均方根误差 RMSE（Root Mean Squared Error）作为评价指标。它的算法很直白：先把每个样本的预测值和真实值相减，平方，把所有平方加起来求平均，最后开根号。用公式写就是 RMSE 等于「误差平方的平均值」的平方根。因为有平方，大误差会被放大，所以 RMSE 对少数预测得特别离谱的样本非常敏感；又因为最后开了根号，它的单位和目标 `y` 一致，可以直接理解成「平均偏差多少」。

RMSE 越低越好，完美预测时等于 0。为了判断模型到底有没有用，我们在本地除了算 RMSE，还会算一个「均值基线」：不管输入什么，都预测训练集的平均费用。如果模型连这个基线都打不过，说明它什么也没学到。此外我们还会看平均绝对误差 MAE 和决定系数 R²，从不同角度了解模型的误差结构。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先检查训练集的整体信息，包括每一列的数据类型、非空数量和缺失比例：

```python
# 查看列类型、非空计数，快速了解哪些列需要清洗
print(train.info())

# 统计每列缺失数量，按缺失从多到少排序
missing_values = train.isnull().sum()
missing_values[missing_values > 0].sort_values(ascending=False)
```

从输出可以看到，部分频率类和费用类字段存在缺失，而且缺失并不是均匀分布的：有的列缺得多，有的列几乎不缺。这一点很重要，因为它决定了我们不能简单地「一行有缺失就删掉」——那样会损失大量样本，还可能把某一类人群系统性删掉，让模型产生偏差。

接着我们看目标变量 `y` 的分布：

```python
# 目标的描述统计：均值、中位数、最大值、零值比例
print("Target mean:", train['y'].mean())
print("Target median:", train['y'].median())
print("Target maximum:", train['y'].max())
print("Percentage of zero values:", round((train['y'] == 0).mean() * 100, 2), "%")

# 画直方图，直观看到右偏和零值堆积
plt.hist(train['y'], bins=50, edgecolor='black', alpha=0.75)
plt.xlabel('Healthcare Cost Target (y)')
plt.ylabel('Number of Individuals')
plt.title('Distribution of the Healthcare Cost Target')
plt.show()
```

直方图显示两个特征：一是大量样本的费用接近 0，二是一条被少数高费用样本拉得很长的右尾。均值明显大于中位数，这也是右偏分布的典型信号。这意味着如果我们只用 MSE 训练，模型会被高费用样本牵着走；但如果我们把高值全删掉，又会丢失最有商业价值的信息。我们的策略是保留全部样本，但在评估时同时看整体 RMSE 和残差分布，确认模型没有在高值段彻底失效。

然后我们看年龄和性别的分布：

```python
# 年龄分布：检查是否有异常年龄（比如负数或过大值）
print(train['AGE'].describe())
plt.hist(train['AGE'], bins=20, edgecolor='black', alpha=0.75)
plt.xlabel('Age')
plt.ylabel('Number of Individuals')
plt.title('Age Distribution of Policyholders')
plt.show()

# 性别分布：确认类别编码是否只有有限几种取值
gender_counts = train['GENDER'].value_counts(dropna=False).sort_index()
print(gender_counts)
```

年龄大致集中在劳动人口区间，极端值不多；性别是离散编码，取值种类有限，适合做类别处理。我们还把数值特征和目标 `y` 的相关系数算出来，挑出相关性最高的十几个变量画热力图：

```python
# 只取数值列，计算与目标的相关系数绝对值并排序
numeric_data = train.select_dtypes(include=[np.number])
target_correlations = numeric_data.corr()['y'].abs().sort_values(ascending=False)

# 取相关性最高的 12 个变量（含目标自身）画相关矩阵
top_columns = target_correlations.head(12).index.tolist()
correlation_matrix = numeric_data[top_columns].corr()
plt.figure(figsize=(11, 9))
plt.imshow(correlation_matrix, aspect='auto', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.xticks(range(len(top_columns)), top_columns, rotation=90)
plt.yticks(range(len(top_columns)), top_columns)
plt.title('Correlation Heatmap of Selected Numerical Variables')
plt.show()
```

从热力图里我们能看到，就诊频率、年龄、BMI 一类变量与费用有中等程度的相关，而有些变量之间本身高度相关（比如不同种类的次数统计），这提示我们后面可以考虑做特征筛选或让树模型自己处理冗余。

### 2.2 数据清洗

数据清洗主要做三件事：缺失值填补、类别编码、特征对齐与缩放。

第一，缺失值填补。我们采取的是最朴素也最稳的策略：数值列用中位数，类别列用众数。中位数比均值更抗极端值，这对右偏的费用类字段尤其重要；众数则是类别列里出现最多的取值，虽然粗糙，但不会引入不存在的类别。

```python
# 对训练集逐列填补：类别列用众数，数值列用中位数
for col in train.columns:
    if train[col].dtype == 'object':
        train[col] = train[col].fillna(train[col].mode()[0])
    else:
        train[col] = train[col].fillna(train[col].median())

# 测试集必须用同样的规则单独填补，避免把训练集的标签信息带进去
for col in test.columns:
    if test[col].dtype == 'object':
        test[col] = test[col].fillna(test[col].mode()[0])
    else:
        test[col] = test[col].fillna(test[col].median())

# 填补后确认没有残留缺失
print("Train Missing:", train.isnull().sum().sum())
print("Test Missing:", test.isnull().sum().sum())
```

这里有一个初学者常犯的错误：先把训练集和测试集拼在一起再填补。这样做看似方便，实际上会让测试集的分布影响训练过程，属于一种轻微的数据泄漏。更稳妥的做法是像上面这样，各自用训练集/测试集自身的统计量填补；如果追求严格，应该只用训练集的统计量去填测试集。

第二，类别编码。模型只认数字，所以要把字符串类别转成整数。我们用 `LabelEncoder` 给每个类别分配一个编号：

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# 逐列找出 object 类型，编码成 0、1、2……
for col in train.columns:
    if train[col].dtype == 'object':
        train[col] = le.fit_transform(train[col].astype(str))

for col in test.columns:
    if test[col].dtype == 'object':
        test[col] = le.fit_transform(test[col].astype(str))
```

对于树模型来说，标签编码已经够用，因为它可以按阈值切分；但如果用神经网络或线性模型，标签编码会人为制造「大小关系」，这时更适合独热编码。我们在神经网络方案里就额外做了标准化。

第三，特征对齐与缩放。我们取训练特征和测试特征的列交集，确保模型看到的列完全一致；然后用 `StandardScaler` 把数值特征变成均值 0、标准差 1：

```python
# 只保留训练和测试共有的特征列
common_cols = X.columns.intersection(test.columns)
X = X[common_cols]
test = test[common_cols]

# 标准化：神经网络对特征尺度敏感，必须做
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)   # 注意：验证集只能用训练集拟合出的 scaler
test_scaled = scaler.transform(test)
```

为什么验证集和测试集只能用 `transform` 而不能 `fit_transform`？因为标准化参数（均值和标准差）必须是「只看训练数据」学出来的，否则验证集的信息就漏进了预处理，线下评估会偏乐观。

### 2.3 模型训练与超参数调优

我们尝试了两条路线，一条是神经网络，一条是梯度提升树。

第一条是 Keras 全连接网络。结构是三层隐藏层，宽度依次 128、64、32，前两层后加 Dropout 抑制过拟合，最后一层只有一个神经元，输出费用预测值：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),          # 随机丢弃 30% 神经元，防止网络死记训练集
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)               # 回归任务，输出一个连续值
])

# 回归用均方误差做损失，同时监控平均绝对误差
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 早停：验证损失连续 10 个 epoch 不下降就停下，并恢复最好的权重
early_stop = EarlyStopping(monitor='val_loss', patience=10,
                           restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_valid_scaled, y_valid),
    epochs=100, batch_size=32,
    callbacks=[early_stop], verbose=1
)
```

训练完成后我们画了训练损失和验证损失曲线。两条曲线如果一起下降，说明模型还在学；如果训练损失继续降、验证损失开始升，就是过拟合的信号，早停会自动把权重退回到验证损失最低的那一轮。

第二条是 `HistGradientBoostingRegressor`，也就是直方图梯度提升树。它的优点是能原生处理缺失值、训练快、对特征尺度不敏感，非常适合这种结构化表格数据。我们把预处理和模型串成一条 `Pipeline`：

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

# 数值列：中位数填补，并额外加一列「是否缺失」的指示特征
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median', add_indicator=True))
])

# 类别列：众数填补 + 序数编码；遇到没见过的类别编码成 -1 而不是报错
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_columns),
    ('categorical', categorical_pipeline, categorical_columns)
])

regressor = HistGradientBoostingRegressor(
    learning_rate=0.05,      # 学习率小一点，配合更多轮数更稳
    max_iter=400,            # 最多 400 棵树
    max_leaf_nodes=31,       # 每棵树最多 31 个叶子
    min_samples_leaf=20,     # 叶子最少 20 个样本，抑制过拟合
    l2_regularization=1.0,   # L2 正则
    random_state=42
)

model_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', regressor)
])
```

这里 `add_indicator=True` 是一个小而重要的设计：它把「这个值原本是不是缺失」变成一列 0/1 特征。很多时候「缺失」本身就有信息，比如某项检查没做，可能意味着申请人比较健康，所以把缺失指示保留下来，比单纯填一个中位数更有表达力。

我们没有做大规模的网格搜索。原因是样本量和特征维度都不算小，每跑一组参数都要重新拟合一整条流水线，时间成本高；而且从几次小规模试验看，`learning_rate=0.05`、`max_iter=400` 这一组已经比较稳，继续细调带来的提升有限。我们把精力放在了特征对齐、缺失处理和误差分析上。

### 2.4 模型测试与指标报告

我们先在 80/20 划分出的验证集上评估。评估流程是：在训练部分拟合整条流水线，在验证部分预测，把负的预测裁剪成 0（费用不可能为负），再和均值基线对比：

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 用训练部分拟合，验证部分只做预测
model_pipeline.fit(X_train, y_train)
validation_predictions = model_pipeline.predict(X_validation)

# 费用不可能为负，把负预测截断到 0
validation_predictions = np.clip(validation_predictions, 0, None)

# 模型 RMSE
validation_rmse = np.sqrt(mean_squared_error(y_validation, validation_predictions))

# 均值基线：无论输入什么都预测训练集平均值
baseline_predictions = np.full(len(y_validation), y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_validation, baseline_predictions))

print("Validation RMSE:", round(validation_rmse, 6))
print("Mean baseline RMSE:", round(baseline_rmse, 6))
print("Improvement over baseline:", round(baseline_rmse - validation_rmse, 6))
```

结果显示，梯度提升模型的验证 RMSE 明显低于均值基线，说明特征里确实存在可学习的信号。我们同时报告 MAE 和 R²：

```python
# 三个指标一起看：RMSE 对大误差敏感，MAE 更稳健，R² 表示解释了多少方差
print("RMSE:", np.sqrt(mean_squared_error(y_validation, validation_predictions)))
print("MAE :", mean_absolute_error(y_validation, validation_predictions))
print("R2  :", r2_score(y_validation, validation_predictions))
```

只看一个总分是不够的。我们画了两张诊断图：一张是「真实值 vs 预测值」的散点图，理想情况下点应该落在对角线上；另一张是残差直方图，残差等于真实值减预测值，理想情况下应该围绕 0 对称分布。

```python
# 散点图：如果点系统性地落在对角线下方，说明高费用段被低估
plt.scatter(y_validation, validation_predictions, alpha=0.35)
plt.plot([y_validation.min(), y_validation.max()],
         [y_validation.min(), y_validation.max()], linestyle='--')
plt.xlabel('Actual Target Value')
plt.ylabel('Predicted Target Value')
plt.title('Actual versus Predicted Healthcare Cost')
plt.show()

# 残差分布：均值应接近 0，标准差反映误差的离散程度
residuals = y_validation - validation_predictions
print("Mean residual:", residuals.mean())
print("Residual standard deviation:", residuals.std())
```

从散点图我们看到，低费用段的预测比较贴近对角线，但高费用段的点明显偏向下方，也就是模型倾向于低估极端高费用。这是平方损失加右偏目标的典型结果：模型不敢预测太大的值，因为一旦猜错惩罚很重，于是选择「保守」。这提醒我们，如果业务上更关心高费用人群，可以考虑对目标取对数、或者用分位数损失，而不是一味优化整体 RMSE。

为了解释模型，我们用排列重要性（permutation importance）：把某一列随机打乱，看验证集误差变差多少，变差越多说明这一列越重要。这个方法比树模型自带的「分裂次数重要性」更贴近真实预测贡献。

```python
from sklearn.inspection import permutation_importance

# 在验证集上重复 5 次打乱每一列，记录 RMSE 的平均恶化程度
importance_result = permutation_importance(
    model_pipeline, X_validation, y_validation,
    scoring='neg_root_mean_squared_error',
    n_repeats=5, random_state=42, n_jobs=-1
)

feature_importance = pd.DataFrame({
    'Feature': X_validation.columns,
    'Importance': importance_result.importances_mean
}).sort_values('Importance', ascending=False).head(12)
print(feature_importance)
```

结果显示就诊频率、年龄、BMI 一类特征贡献最大，这和 EDA 阶段的相关系数分析基本一致，说明模型学到的东西是合理的，而不是靠某个偶然的 ID 列作弊。

确认方案可用后，我们在全量训练数据上重新拟合一次，再对测试集预测：

```python
# 用全部训练样本重新训练（数据更多，通常更稳）
model_pipeline.fit(X, y)

# 对测试集做同样的预处理和预测
test_predictions = model_pipeline.predict(X_test)

# 费用不能为负，做同样的截断
test_predictions = np.clip(test_predictions, 0, None)

print("Minimum prediction:", test_predictions.min())
print("Maximum prediction:", test_predictions.max())
print("Mean prediction:", test_predictions.mean())

# 按提交格式写出：ID + y.hat
submission = pd.DataFrame({'ID': test_ids, 'y.hat': test_predictions})
submission.to_csv('/kaggle/working/submission_baseline.csv', index=False)
```

最后我们检查提交文件的行数是否和测试集一致、有没有空值、预测值范围是否合理。这些检查看起来琐碎，但比赛里因为列名写错或行数不对而白白丢分的情况并不少见。

## 三、总结

本文中我们完成了越南医疗费用预测的完整流程。我们首先用 EDA 摸清了数据的基本面貌：目标 `y` 右偏且大量为零、若干频率与费用字段存在缺失、年龄与性别分布正常、就诊频率和 BMI 等特征与费用有中等相关；其次我们做了最少必要的数据清洗，数值列用中位数填补、类别列用众数填补、字符串类别做标签编码，并且严格只保留训练集和测试集共有的特征列，避免把只在训练集出现的费用字段当成特征造成泄漏；接着我们尝试了两条建模路线，一条是带 Dropout 和早停的三层 Keras 全连接网络，另一条是把缺失填补、序数编码和 `HistGradientBoostingRegressor` 串成流水线的梯度提升方案，并在 80/20 验证集上比较；然后在验证集上同时报告 RMSE、MAE 和 R²，用均值基线做对照，用散点图和残差分布检查误差结构，再用排列重要性解释关键特征；最后我们在全量数据上重新训练，对测试集预测、裁剪负值并按格式生成提交文件。

这个比赛给我们最大的启发有三点。第一，**特征对齐比模型选择更重要**：如果不去检查哪些列只在训练集出现，模型会在线下表现得异常好，上线却完全失效。第二，**缺失值处理要结合业务含义**：中位数填补简单可靠，但加上「是否缺失」指示特征往往能保留更多信息。第三，**评价一个回归模型不能只看一个 RMSE**：右偏目标会让模型在高值段系统性低估，只有把散点图和残差分区间看清楚，才知道模型真正强在哪里、弱在哪里。对我们这些初学者来说，这个比赛是一个很好的练习：它不需要复杂的深度学习技巧，却把数据泄漏、缺失处理、目标偏斜和误差分析这些最基础也最容易踩坑的问题都摆在了面前。
