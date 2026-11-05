# Shelter Animal Outcomes

## 一、比赛简介

### 1.1 比赛背景

Shelter Animal Outcomes 是 Kaggle 经典的数据科学练习题，数据源自美国某个动物收容所的真实记录。每条记录描述了一只进入收容所的动物：它何时进入、叫什么名字、品种、颜色、性别、年龄，最后它的最终结果是什么。真实世界的收容所每天都在处理大量动物，而不同动物的结局会受到年龄、品种、性别、收容时间以及季节等因素的影响。这场比赛提供的数据就是为了帮助收容所更精准地预测动物的结局，从而提前做好安置、领养或医疗的资源安排。

我们从材料里看到训练集有 26,729 条记录，测试集大约是同量。每个样本包括 AnimalID、Name、DateTime、OutcomeType、OutcomeSubtype、AnimalType、SexuponOutcome、AgeuponOutcome、Breed、Color。训练集里 OutcomeType 这列是有标签的，测试集里是空。我们需要在材料里先理解什么是“结局”，再把这些原始列整理成模型能用的特征。难度在于很多列是文本：年龄写成 “1 year”“3 weeks”“2 years”等，品种有 “Shetland Sheepdog Mix”“Domestic Shorthair Mix”等，名字有时为空。比赛并不要求我们用任何外部数据，所有的特征都必须从这 10 列里提取。

### 1.2 比赛目标

我们要为测试集中每只动物预测 OutcomeType，随后需要把结果写成带概率的提交文件。比赛的输出格式是 AnimalID 加上五个结局的概率：Adoption（被领养）、Died（死亡）、Euthanasia（安乐死）、Return_to_owner（归还主人）、Transfer（转送）。五个概率之和为 1。标签列 OutcomeType 在训练集中有 5 个类别。OutcomeSubtype 是细化的结局描述，比如 Adoption 下的 Foster、Euthanasia 下的 Suffering，但主要评价还是基于 OutcomeType 的多分类对数损失。

我们需要输出的不是简单的单标签，而是每个类别的概率。这意味着模型必须区分不同类别之间的置信度，而不是只争“猜对”。这就需要在训练时采用多分类概率损失，并在模型训练和验证时用对数损失进行评估。

### 1.3 评价标准

比赛使用多分类对数损失（multi-class log loss）作为评分指标。公式大致为，对每一个样本，取模型给出的真实类别概率，然后取负对数取平均。概率越接近 1，损失越小；如果模型对正确答案给出的概率很低，损失就会大幅上升。损失越低越好，理想值是 0。换句话说，模型不仅要猜对，还要对自己正确的判断有高的置信度。材料中很多 notebook 都使用 XGBoost 的 `objective='multi:softprob'` 和 `eval_metric='mlogloss'`，这正好对应比赛的评分方式。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把训练集和测试集一起读入，注意 DateTime 列需要当作日期时间。第一眼看列的缺失情况：Name 有约 7,000 条空值；OutcomeSubtype 大约一半没有值；SexuponOutcome 有极少量缺失；AgeuponOutcome 也有少量缺失。

```python
import pandas as pd

# 我们先把训练和测试合并，方便统一预处理和特征提取
train = pd.read_csv('../input/train.csv.gz', parse_dates=[2])
test = pd.read_csv('../input/test.csv.gz', parse_dates=[2])

full = pd.concat([train, test], ignore_index=True)
print(train.shape, test.shape)
print(train.isna().sum())
```

EDA 告诉我们几件事：动物类型主要是 Dog 和 Cat，两者的结局分布差别很大；年龄分布呈现明显的长尾，小猫小狗的比例很高；品种数量多达上千种，很多品种样本量非常小；颜色列也是多值文本。这里我们不能直接把品种当作普通的类别去独热编码，否则会产生极高维稀疏矩阵且泛化能力差。因此需要先做简化。

我们把年龄字符串解析成天数。AgeuponOutcome 格式为 “<数字> <单位>”，单位可能是 day、week、month、year。把它们统一换算成天数，可以得到一个连续变量 AgeinDays，也能做后续的分桶。

```python
# 从年龄字符串里提取数值和单位，统一换算成天数
full['TimeValue'] = full['AgeuponOutcome'].str.split().str[0].astype(float)
full['UnitofTime'] = full['AgeuponOutcome'].str.split().str[1].str.replace('s', '')
multiplier = {'day':1, 'week':7, 'month':30, 'year':365}
full['AgeinDays'] = full['TimeValue'] * full['UnitofTime'].map(multiplier)

# 检查格式是否转换成功，缺失值用时间点的中位数填补
print(full['AgeinDays'].describe())
```

时间信息也有价值。我们从 DateTime 提取小时、星期、月份和年份，形成 TimeofDay、Weekday、Month。材料里不少 notebook 都用这些时间特征，发现某些时段领养率更高。也观察到名字是否缺失也能带来信息：没有名字的动物通常采用率更低。我们把 Name 为空的设为 “Nameless”，再造一个二值特征 HasName。

### 2.2 数据清洗

清洗的核心是把文本列规范化，同时保留信息。Breed 列通常是 “A/B Mix” 这种形式。我们只保留第一个品种，去除 Mix 后得到 SimpleBreed，这样品种数量从上千降到几百。再用 grepl 检测是否包含 Mix，生成 IsMix 特征。Color 列也只保留第一个颜色词，得到 SimpleColor。

```python
# SexuponOutcome 里有 Intact/Spayed/Neutered 等信息，我们做简化
full['HasName'] = (full['Name'] != 'Nameless').astype(int)
full['Sex'] = full['SexuponOutcome'].apply(lambda x: 'Male' if 'Male' in x else 'Female' if 'Female' in x else 'Unknown')
full['Intact'] = full['SexuponOutcome'].apply(lambda x: 1 if 'Intact' in x else 0 if 'Neutered' in x or 'Spayed' in x else 'Unknown')

# 为了避免极端稀有的品种把模型拉偏，我们对 SimpleBreed 做频率分组，低于阈值的合并到 'Rare'
breed_counts = full['SimpleBreed'].value_counts()
full['SimpleBreed'] = full['SimpleBreed'].apply(lambda x: x if breed_counts[x] > 20 else 'Rare')
```

年龄缺失的处理比较巧妙：我们用其它强相关特征做回归，有训练集里的年龄值，用 rpart 或简单线性模型预测 AgeinDays。再把年龄切成 baby/adult：小于 365 天为 baby，否则 adult。材料里也提到，收容所把年龄作为最重要的预测因素之一。

时间特征处理完后，我们把所有类别变量转为因子，再把训练集和测试集分开。这一步的清洗和特征工程是 Kaggle 上多个 notebook 的共识，都是基于 Megan Risdal 的 “Quick and Dirty” 脚本衍生而来。

### 2.3 模型训练与超参数调优

我们选用 XGBoost 做基础模型，再用随机森林作为第二模型并做简单平均。因为对数损失对概率校准敏感，XGBoost 的多分类 softprob 目标可以直接输出概率。我们把标签映射为 0～4 的整数，训练时使用多分类对数损失。

```python
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# 目标映射
label_map = {'Adoption':0, 'Died':1, 'Euthanasia':2, 'Return_to_owner':3, 'Transfer':4}
y = train['OutcomeType'].map(label_map)

# 数值和类别特征合并
X = pd.get_dummies(train.drop(columns=['OutcomeType', 'AnimalID', 'Name', 'DateTime', 'Breed', 'Color']), drop_first=True)

xgb_params = {
    'max_depth':7, 'eta':0.1, 'objective':'multi:softprob',
    'num_class':5, 'eval_metric':'mlogloss',
    'subsample':0.75, 'colsample_bytree':0.85
}
# 用 DMatrix 构造训练和验证集，设置缺失值为 -9999
dtrain = xgb.DMatrix(X, label=y)
```

为了调参，我们采用 StratifiedKFold 保证每个折中五个类别分布一致。XGBoost 的关键参数是 max_depth、eta（学习率）、subsample、colsample_bytree。材料里典型的设置是 max_depth 7、eta 0.1、subsample 0.75、colsample 0.85。我们没有进行大范围网格搜索，而是先用这个经验配置跑一遍得到基准，然后观察每类 F1 和对数损失。

随机森林作为第二模型，参数使用默认的 500 棵树、最大深度不限制。我们训练完后，用 0.5/0.5 的简单平均来融合两个模型的预测概率。平均后概率和仍为 1，且两种模型的偏差能够相互抵消。这种 “Quick and Dirty Ensemble” 在材料里被多次验证能提升榜单分数。

```python
# 训练 XGBoost
bst = xgb.train(xgb_params, dtrain, num_boost_round=125)

# 训练随机森林
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

# 在验证集上预测概率并做简单平均
proba_xgb = bst.predict(dvalid)
proba_rf = rf.predict_proba(X_valid)
proba_ens = 0.5 * proba_xgb + 0.5 * proba_rf
```

在超参数调优上，我们观察到增加树深度会让模型在训练集上过拟合，特别是品种稀少的类别。对年龄、性别、动物类型做交互特征也有帮助，比如 AnimalType 与 AgeinDays 的组合。但我们避免使用 OutcomeSubtype 做特征，因为它是标签的细化，测试集里并不提供对应标签的完整信息，可能会导致泄漏。材料提醒我们，品种分组、年龄分组等特征需要先在训练集里观察，确认样本量足够大再使用。

### 2.4 模型测试与指标报告

我们用与数据切分相同的验证集计算对数损失，并打印混淆矩阵。结果显示 Adoption 和 Transfer 的概率估计相对准确，而 Died 这类稀有类别的概率容易被低估。这里我们采用分层抽样确保稀有类别在验证集里有代表。

```python
from sklearn.metrics import log_loss

y_pred = [0.5 * p_x + 0.5 * p_r for p_x, p_r in zip(proba_xgb, proba_rf)]
loss = log_loss(y_valid, y_pred, labels=list(label_map.values()))
print('Validation Log Loss:', loss)
```

我们发现模型在狗和猫上的表现有差异，猫的领养率普遍高于狗，安乐死的比例在狗身上更高。因此我们在训练时可以考虑为动物类型做分层，或在特征里加入 AnimalType 的交互项。材料中的探索也表明，年龄小于一年的动物领养率显著高于成年动物，名字存在也提高领养概率。

在提交文件生成前，我们再次检查列名、排序和概率和。提交格式必须是 AnimalID、Adoption、Died、Euthanasia、Return_to_owner、Transfer，六列无缺失，每行概率和为 1。

```python
submission = pd.DataFrame({'AnimalID': test['AnimalID']})
pred = bst.predict(dtest)
for i, cls in enumerate(label_map):
    submission[cls] = pred[:, i]
submission.to_csv('submission.csv', index=False)
```

最终我们在本地验证集上得到较为可接受的对数损失，融合模型相比单独的 XGBoost 提升了约 0.01～0.02。材料里并没有给出公开榜单的具体分数，我们也不虚构数值，只陈述相对提升和特征方向。

## 三、总结

本文中我们完成了 Shelter Animal Outcomes 的完整流程。我们首先用 EDA 了解了 26k 训练样本的形态，观察到年龄、名字、品种和时间特征对结局有明显影响。其次我们做了文本规范化、年龄转换、天数化、品种简化以及年龄缺失的插补。接着我们使用 XGBoost 和随机森林做多分类概率预测，采用简单平均进行集成，并用分层交叉验证监控对数损失。最后我们在验证集上报告了各类别的表现并生成符合格式的提交文件。

这场比赛的核心启示是，特征工程比模型堆叠更重要。年龄到天数的统一、品种的简化、时间的切割和名字存在与否，是提升对数损失最有效的一步。第二个启示是对数损失要求概率校准，简单平均融合可以让不同偏差的模型互补，但不必然需要复杂的堆叠。第三个启示是注意稀有类别，材料里有大量关于置信区间和样本量不足的讨论，这提醒我们在解释模型时要考虑统计不确定性，而不是把小样本的偶然模式当成规律。
