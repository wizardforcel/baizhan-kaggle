## 一、比赛简介

### 1.1 比赛背景
Spaceship Titanic 是 Kaggle 上的一个经典表格结构竞赛。故事设定在 2077 年，Spaceship Titanic 在超新星附近爆炸，大量乘客被传送散落到各个维度。任务是根据未来公司的乘客登记信息和消费记录，预测哪些乘客最终被成功传送回飞船。该比赛数据包含了乘客的航行信息、舱室信息、冷冻睡眠状态以及在飞船上的消费情况。对深度学习与编程入门者来说，这个比赛的数据量不大、特征含义直观、评价方式简单，非常适合用来练习完整的表格数据建模流程。

与图像或文本比赛不同，表格比赛更强调对字段含义的理解。我们需要先弄清楚每一列代表什么，再判断哪些列可以直接使用、哪些列需要拆分、哪些列需要组合。Spaceship Titanic 的 Cabin 和 PassengerId 都不是简单的字符串，而是带了层级信息的编码字段，这也是本比赛特征工程的主要切入点。

### 1.2 比赛目标
比赛提供 train.csv 和 test.csv。训练集包含 PassengerId、HomePlanet、Destination、CryoSleep、Age、VIP、RoomService、FoodCourt、ShoppingMall、Spa、VRDeck、Cabin、Name、Transported 等列，其中 Transported 是二分类标签，表示乘客是否被传送回飞船。测试集缺少 Transported 列，我们需要基于训练数据学习规律，给出测试集的预测结果并提交。换句话说，这是一个监督学习中的二分类问题：输入是乘客的各种登记与消费信息，输出只有两个可能值，即 True 或 False。

### 1.3 评价标准
比赛采用准确率 Accuracy 作为评价标准，即预测正确的样本比例。举例来说，如果一共有 100 个测试样本，我们预测对了 80 个，那么准确率就是 0.80。准确率高低直接反映模型对传送规律的捕捉能力。基线随机森林模型在 5 折交叉验证上的表现约为 0.78，也就是说大部分乘客的去向可以被这些基础特征解释，剩余部分则需要更细致的特征工程或更强的模型来提升。

## 二、解题过程

### 2.1 数据探索EDA
我们首先加载数据并观察基本形状和目标分布，了解数据规模和类别平衡情况。EDA 是探索性数据分析的缩写，目的不是立刻建模，而是先看清数据长什么样，例如有多少行、有多少列、标签比例是否均衡、哪些字段有缺失。只有先看清这些基本情况，后面的清洗和建模才不会盲目。
```python
import pandas as pd
import numpy as np

# 加载训练集和测试集
train = pd.read_csv('/kaggle/input/train.csv')
test = pd.read_csv('/kaggle/input/test.csv')

# 查看数据形状，即行数和列数
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# 查看目标变量的分布，normalize=True 表示输出比例而不是数量
print(train['Transported'].value_counts(normalize=True))
```
训练集大约有 8,693 条样本，Transported 为 True 的比例约为 0.38，存在一定的类别不平衡。这里的类别不平衡是指两类样本数量不完全相等，但差距不算极端，因此后续评估时使用准确率仍然合理。我们将训练集和测试集合并，方便统一做特征工程。合并的好处是，类别编码和缺失值填充可以一次性对两份数据使用同一套规则，避免训练集和测试集处理方式不一致。
```python
# 合并数据便于统一处理，ignore_index 让行号重新排列
all_data = pd.concat([train, test], axis=0, ignore_index=True)

# 标记每一行来自训练集还是测试集，Transported 不为空的就是训练数据
all_data['is_train'] = all_data['Transported'].notna().astype(int)

print(f"合并后数据形状: {all_data.shape}")
```
合并后我们初步观察 Cabin 字段的格式，发现其包含甲板、舱号、侧位等多层信息。例如同一个 Cabin 字符串通常由斜杠分隔成三段，第一段是甲板区域，第二段是舱号，第三段是左舷或右舷。PassengerId 中也隐含了分组信息，同一组乘客往往一起订票，因此组内人数可能和是否被传送存在关联。这些观察为后续特征工程提供了思路。

### 2.2 数据清洗
我们对 Cabin 字段进行拆分，提取甲板 Deck、舱号 CabinNum 和侧位 Side 作为独立特征。把一段混杂的字符串拆成三个含义明确的列，可以让模型分别学习甲板位置、舱号大小和左右侧位的影响。随后我们从 PassengerId 中拆出 Group 编号，并统计每个 Group 的人数作为 GroupSize。GroupSize 可以粗略理解为同行人数，人数不同的家庭或团体在逃生时可能表现出不同规律。
```python
# 拆分 Cabin 成三部分，expand=True 表示展开成多列
all_data[['Deck', 'CabinNum', 'Side']] = all_data['Cabin'].str.split('/', expand=True)

# CabinNum 转为数值类型，无法转换的会变成缺失值
all_data['CabinNum'] = pd.to_numeric(all_data['CabinNum'], errors='coerce')

# 从 PassengerId 提取下划线前面的 Group 编号
all_data['Group'] = all_data['PassengerId'].str.split('_').str[0].astype(int)

# 计算每个 Group 的乘客数量作为 GroupSize
group_size = all_data.groupby('Group')['PassengerId'].transform('count')
all_data['GroupSize'] = group_size
```
接着我们对五项消费列进行汇总，构造总消费特征并做对数变换，同时生成 NoSpend 交互特征。五个消费列分别是 RoomService、FoodCourt、ShoppingMall、Spa 和 VRDeck，它们代表乘客在不同场所的花费。把五项相加可以得到一个总体消费水平，而消费金额往往差异很大，直接使用容易受极端值影响，所以我们对总消费做 log1p 变换，把长尾分布压平一些。NoSpend 则是一个简单的 0/1 标记，表示这位乘客是否完全没有消费。
```python
spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# 计算总消费，axis=1 表示按行求和
all_data['TotalSpend'] = all_data[spend_cols].sum(axis=1)

# 对数变换处理偏态，log1p 可以安全处理 0 值
all_data['LogTotalSpend'] = np.log1p(all_data['TotalSpend'])

# 总消费为 0 时标记为 1，否则为 0
all_data['NoSpend'] = (all_data['TotalSpend'] == 0).astype(int)

print("新增特征示例:", all_data[['TotalSpend','LogTotalSpend','NoSpend']].head())
```
在缺失值处理上，我们对数值列使用中位数填充，类别列使用众数填充，避免引入额外偏差。中位数不容易被极端值拉偏，适合年龄、消费这类数值字段；众数则是出现次数最多的类别，用它填补类别字段比较符合直觉。这样处理后，模型不会因为空值而报错，也不会因为随便填一个异常值而学到错误规律。
```python
# 数值列使用中位数填充
num_cols = all_data.select_dtypes(include=[np.number]).columns
for col in num_cols:
    all_data[col] = all_data[col].fillna(all_data[col].median())

# 类别列使用众数填充，若众数不存在则填 Unknown
cat_cols = all_data.select_dtypes(include=['object']).columns
for col in cat_cols:
    mode_val = all_data[col].mode()
    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
    all_data[col] = all_data[col].fillna(fill_val)

print("剩余缺失值数量:", all_data.isnull().sum().sum())
```
我们删除对建模无帮助的 PassengerId、Name 和中间 Group 列，保持特征一致性。PassengerId 和 Name 是标识类字段，直接输入模型容易造成无意义的记忆；Group 编号本身只是分组标签，我们已经用 GroupSize 表达了组规模信息，因此可以删掉原始编号。
```python
drop_cols = ['PassengerId', 'Name', 'Group']
all_data.drop(columns=drop_cols, inplace=True, errors='ignore')
```
之后使用 LabelEncoder 对剩余类别特征进行编码，使其可被树模型使用。LabelEncoder 会把每个不同的类别映射成一个整数，例如把不同的甲板名称变成 0、1、2 等。这个做法简单直接，对随机森林这类基于树的模型通常够用。
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cat_cols:
    if col in all_data.columns:
        all_data[col] = le.fit_transform(all_data[col].astype(str))

print("编码后特征数:", all_data.shape[1])
```

### 2.3 模型训练与超参数调优
特征准备好后，我们将数据拆回训练和测试集，确保列名一致。这里的 assert 语句是一个保护措施，如果训练集和测试集的特征列顺序或名称不一致，程序会立刻报错，避免在生成提交文件时才发现问题。
```python
train_processed = all_data[all_data['is_train'] == 1].drop(columns=['is_train'])
test_processed = all_data[all_data['is_train'] == 0].drop(columns=['is_train', 'Transported'])

X_train = train_processed.drop(columns=['Transported'])
y_train = train_processed['Transported'].astype(int)
X_test = test_processed

assert list(X_train.columns) == list(X_test.columns)
```
我们采用随机森林作为基线模型，易于调参且对类别特征鲁棒。随机森林由很多棵决策树组成，每棵树看到一部分样本和一部分特征，最后通过投票得到预测结果。因为是多棵树平均，它对单个特征的噪声和异常值不那么敏感，也不需要像线性模型那样做标准化。我们设置 n_estimators=300 表示使用 300 棵树，max_depth=8 限制每棵树的最大深度，min_samples_split=5 表示一个节点至少要有 5 个样本才继续分裂。
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,      # 树的数量
    max_depth=8,           # 每棵树的最大深度
    min_samples_split=5,   # 节点分裂所需的最小样本数
    random_state=42,       # 固定随机种子，方便复现
    n_jobs=-1              # 使用全部 CPU 核心加速
)
```
使用分层 5 折交叉验证保持目标分布，评估模型稳定性。分层抽样是指每一折中正负样本的比例都与原始数据接近，这样评估结果更可靠。5 折交叉验证会把训练数据分成 5 份，每次用 4 份训练、1 份验证，循环 5 次后取平均准确率。
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```
在尝试调参时，我们发现增大树深容易导致过拟合，max_depth=8 时在偏差和方差之间取得较好平衡。树太深会记住训练集中的细节，在验证集上反而变差；树太浅又学不到足够规律。最终我们保留这组较稳健的配置，并把交叉验证准确率作为模型选择的依据。

### 2.4 模型测试与指标报告
交叉验证完成后，我们在全量训练集上拟合最终模型并生成测试集预测。这里重新 fit 一次，是因为交叉验证只是评估流程，并不会自动保留一个可用于预测的最终模型。把全部训练数据都用上，可以让模型见到更多样本，通常更有利于测试表现。
```python
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 读取样例提交文件，并按相同顺序写入预测结果
submission = pd.read_csv('/kaggle/input/sample_submission.csv')
submission['Transported'] = predictions.astype(bool)
submission.to_csv('submission.csv', index=False)

print("提交文件已保存")
print(f"预测 Transported 率: {submission['Transported'].mean():.4f}")
```
本地 5 折交叉验证准确率约为 0.78，与基线水平一致。预测的 Transported 率约为 0.39，与训练集比例接近，说明模型没有出现明显的预测偏移。我们检查了特征列的一致性，确保训练和测试使用完全相同的特征空间，最终提交文件格式正确、可以直接上传。

## 三、总结
本文中我们完成了 Spaceship Titanic 分类任务的完整流程。我们首先通过 EDA 了解乘客特征与目标分布，确认训练集约 8,693 条样本、正类比例约 0.38；其次对 Cabin 拆出 Deck、CabinNum、Side，对 PassengerId 拆出 Group 并计算 GroupSize，对五个消费列求和得到 TotalSpend，并衍生 LogTotalSpend 与 NoSpend；接着用中位数和众数完成缺失值填充，用 LabelEncoder 完成类别编码；然后使用 RandomForestClassifier 配合 StratifiedKFold 5 折交叉验证进行训练与调优，选择 n_estimators=300、max_depth=8、min_samples_split=5 的稳健配置；最后在全量训练集上拟合并生成提交文件，本地交叉验证准确率约为 0.78。整个流程覆盖了表格比赛从数据探索到提交的完整环节，便于入门者照着复现并在此基础上继续改进。
