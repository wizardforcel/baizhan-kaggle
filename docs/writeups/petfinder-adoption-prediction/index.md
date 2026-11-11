# PetFinder 领养速度预测（PetFinder Adoption Prediction）

## 一、比赛简介

### 1.1 比赛背景

PetFinder.my 是马来西亚一个很有名的动物领养平台，从 2008 年开始做动物福利，数据库里已经积累了超过 15 万只动物的资料。它一边和动物爱好者、媒体、企业合作，一边帮收容所和救助人把待领养的宠物信息放到网上，让更多流浪动物有机会找到新家。

现实里，全世界每天都有大量流浪动物在街头受苦，或者在收容所里被安乐死。如果一只动物能更快被领养，它活下来的机会就更大，也会多一个家庭因为领养而变得完整。因此平台很关心一个问题：**一只宠物挂到网上之后，到底要过多久才会被领养？**

这个问题和宠物资料里的很多信息有关。平台自己的观察是，领养率和在线资料里的元数据强相关，比如描述文字写得好不好、照片拍得怎么样。PetFinder 甚至试验过一个叫 "Cuteness Meter"（可爱度测量器）的小工具，用照片里的特征给宠物的可爱程度打分。这个比赛希望参赛者根据这些资料预测宠物的领养速度，将来把模型做成工具，反过来指导收容所优化宠物资料，减少动物受苦和被安乐死。

这个比赛的数据有一个特点：它同时包含**自由文本**（宠物描述）、**表格数据**（年龄、品种、颜色、是否绝育等）和**图片**（每只宠物可能有多张照片）。信息很丰富，但也意味着要处理不同形式的数据。我们这篇 writeup 跟随材料里的做法，重点使用表格字段和描述文本，图片本身不直接读入模型。

### 1.2 比赛目标

训练集 `train.csv` 有 14993 行、24 列；测试集 `test.csv` 有 3972 行、23 列，比训练集少一列目标值。每一行代表一只待领养的宠物，主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `Type` | 动物种类，1 是狗，2 是猫 |
| `Name` | 宠物名字（有缺失） |
| `Age` | 年龄，单位是月 |
| `Breed1` / `Breed2` | 主品种、次品种的编号 |
| `Gender` | 性别 |
| `Color1` / `Color2` / `Color3` | 三种颜色编号 |
| `MaturitySize` | 成年体型 |
| `FurLength` | 毛长 |
| `Vaccinated` / `Dewormed` / `Sterilized` | 是否接种疫苗、驱虫、绝育 |
| `Health` | 健康状况 |
| `Quantity` | 同一份资料里宠物的数量 |
| `Fee` | 领养费用 |
| `State` | 所在州 |
| `RescuerID` | 救助人编号 |
| `VideoAmt` / `PhotoAmt` | 视频数、照片数 |
| `Description` | 自由文本描述 |
| `PetID` | 宠物编号 |
| `AdoptionSpeed` | **要预测的目标**，领养速度等级 |

目标 `AdoptionSpeed` 分成 5 个有序等级：

- `0`：上架当天就被领养；
- `1`：1 到 7 天内被领养；
- `2`：8 到 30 天内被领养；
- `3`：31 到 90 天内被领养；
- `4`：上架超过 100 天还没被领养。

注意题目里没有 90 到 100 天之间的样本，等级 4 直接表示"超过 100 天"。我们要给测试集里每只宠物预测它属于哪一级，提交文件只有 `PetID` 和 `AdoptionSpeed` 两列。

### 1.3 评价标准

比赛用**二次加权卡帕系数**（Quadratic Weighted Kappa，简称 QWK）来评分。这个名字听起来吓人，其实可以这样理解：假设有两个人给同一批宠物按领养速度打分，卡帕系数衡量的是"这两个人的打分有多一致"。完全一致是 1，和随机乱猜差不多是 0，比随机还差就可能出现负数。

"二次加权"的意思是：预测错得越远，扣分越重。比如真实等级是 0（当天领养），我们猜成 1 只差一格，损失较小；猜成 4 就差四格，损失按平方增长，惩罚很大。这样设计很合理，因为领养速度本身就是有顺序的，把"当天领养"错判成"超过 100 天"显然比错判成"一周内领养"严重得多。

分数越高越好，满分是 1.0。由于这个指标惩罚"次序错得远"的预测，模型不仅要猜对大概方向，还要尽量让预测等级靠近真实等级。

---

## 二、解题过程

整场比赛我们走了一条比较稳的路线：**先把表格和文本整理成干净的特征，再用多个模型做交叉验证对比，最后挑泛化最好的一套提交。** 下面按数据探索、数据清洗、模型训练与调参、测试与指标报告四步来讲。

### 2.1 数据探索（EDA）

我们首先把训练集、测试集和提交样例读进来，确认数据规模和列结构：

```python
import numpy as np
import pandas as pd

# 三个文件都从比赛数据目录读取
train = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
test = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv")
submission = pd.read_csv("../input/petfinder-adoption-prediction/test/sample_submission.csv")

# 训练集 14993 行 24 列，测试集 3972 行 23 列（少一列目标值）
print(train.shape)   # (14993, 24)
print(test.shape)    # (3972, 23)

# 看前 5 行，确认每一列的实际取值
train.head()
```

读进来后我们看到，训练集有 24 列，其中既有 `Type`、`Age`、`Fee` 这样的纯数值或编号字段，也有 `Description` 这样的长文本，还有 `PetID` 这种只用来标识样本的编号。测试集和训练集列名基本一致，只是没有 `AdoptionSpeed`，因此两份数据可以合并起来做统一的特征处理。

其次我们看目标变量的分布，这一步很关键，因为它决定了后面能不能随便用"准确率"来评价模型：

```python
# 五个领养速度等级各有多少只宠物
print(train['AdoptionSpeed'].value_counts())
# 4    4195   超过 100 天仍未领养
# 2    4036   8 到 30 天
# 3    3257   31 到 90 天
# 1    3090   1 到 7 天
# 0     410   当天领养

# 再看猫和狗在这五个等级上的分布差异
pd.crosstab(train['Type'], train['AdoptionSpeed'])
```

分布明显不均衡：等级 4 最多，有 4195 只；等级 0 最少，只有 410 只，大约占 2.7%。这意味着如果模型偷懒，把所有宠物都预测成 4，也能拿到不低的"准确率"，但对比赛指标没有好处。我们后面要特别关注少数类，也要用能体现"次序"的评分方式来做交叉验证。

接着我们看缺失值和重复行：

```python
# 缺失值集中在两列：Name 缺 1257 条，Description 缺 12 条
print(train.isnull().sum())

# 检查重复行，结果是 0，说明训练集没有完全重复的记录
print(train.duplicated().sum())
```

从结果看，绝大多数字段都是完整的，只有名字缺得比较多、描述缺了十几条。名字对领养速度的预测价值有限，而且缺失比例不小，我们后面会直接把它从特征里去掉；描述缺失的样本很少，可以用空字符串或 0 填充。

我们还借助三张辅助表了解了品种和动物构成：品种表里有 241 种狗、66 种猫，训练集里狗大约占 54%，测试集里大约占 53%。训练集中有 5 只动物没有主品种信息，材料里的做法是把这几条直接删除；另外还有约 2716 条是跨品种的混血，其余 12272 条是单一品种。EDA 里还有一个有意思的发现：猫在 7 天内被领养的比例比狗高，混血动物往往比纯种动物更快被领养，尤其是猫。

### 2.2 数据清洗

第二步做数据清洗。我们的原则是：**文本字段要"既保留信息又清理噪声"，类别字段要"把编号翻译成人能看懂、模型也能用的形式"。**

我们先从 `Description` 里抽统计特征，再去掉标点、统一转小写：

```python
from string import punctuation

def clean_text(data):
    """从描述文本中提取长度类特征，并做基础清理"""
    # 字符数、词数、平均每词字符数
    data['text_char_count'] = data['Description'].apply(len)
    data['text_word_count'] = data['Description'].apply(lambda x: len(x.split()))
    data['text_density'] = data['text_char_count'] / (data['text_word_count'] + 1)

    # 标点数量、首字母大写的词数、全大写词数
    # 这些特征可以粗略反映描述的"热闹程度"和强调语气
    data['text_punctuation_count'] = data['Description'].apply(
        lambda x: len([c for c in x if c in punctuation]))
    data['title_word_count'] = data['Description'].apply(
        lambda x: len([w for w in x.split() if w.istitle()]))
    data['text_upper_case'] = data['Description'].apply(
        lambda x: len([w for w in x.split() if w.isupper()]))

    # 去掉标点并转小写，方便后续词袋和情感分析
    data['Description'] = data['Description'].apply(
        lambda x: x.translate(str.maketrans('', '', punctuation)))
    data['Description'] = data['Description'].str.lower()
    return data

train = clean_text(train)
test = clean_text(test)
```

这样做的理由是：描述长度、标点密度这些"元特征"往往比描述的具体内容更稳定，长而用心的描述可能意味着救助人更上心，也可能只是模板文案，模型可以从数据里自己学。清理后的正文则可以继续做词频或情感特征。

接着我们把品种编号翻译成品种名，并处理"混血"和"混物种"：

```python
# 读入品种表，里面有 BreedID、BreedName 和 Type
breeds = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/breed_labels.csv')

# 建立两个映射：编号 -> 品种名，编号 -> 狗/猫
mapping = dict(breeds[['BreedID', 'BreedName']].values)
map_animal = dict(breeds[['BreedID', 'animal']].values)

train['Main_Breed'] = train['Breed1'].map(mapping)
train['Main_Breed_animal'] = train['Breed1'].map(map_animal)
train['Sec_Breed_animal'] = train['Breed2'].map(map_animal)

# 删除主品种和次品种分属猫、狗的极少数记录
# 这种记录不是混血，而是数据本身有问题，留着会干扰模型
mask = train['Sec_Breed_animal'].notna() & (
    train['Main_Breed_animal'] != train['Sec_Breed_animal'])
train = train[~mask]

# 只有两列都有品种、且品种不同，才算真正的跨品种混血
def get_cross_breed(breed1, breed2):
    return 'Yes' if (breed2 != 0) & (breed1 != breed2) else 'No'

train['Cross_Breed'] = train.apply(
    lambda x: get_cross_breed(x['Breed1'], x['Breed2']), axis=1)
```

然后我们把体型、毛长、疫苗、驱虫、绝育、健康这些用数字编码的字段映射成文字。这样做不是为了"好看"，而是因为原始编号没有大小含义，直接当数值喂给模型会让模型误以为 3 比 1 "大"，映射成类别后再做 one-hot 才符合实际语义：

```python
# 每个数字编码对应的真实含义
mapping_maturity = {1: 'Small', 2: 'Medium', 3: 'Large', 4: 'Extra Large', 0: 'Not Specified'}
mapping_fur = {1: 'Short', 2: 'Medium', 3: 'Long', 0: 'Not Specified'}
mapping_vet = {1: 'Yes', 2: 'No', 3: 'Not Sure'}
mapping_health = {1: 'Healthy', 2: 'Minor Injury', 3: 'Serious Injury', 0: 'Not Specified'}

# 训练集和测试集必须用同一套映射，保证编码一致
for col, mp in [('MaturitySize', mapping_maturity),
                ('FurLength', mapping_fur),
                ('Vaccinated', mapping_vet),
                ('Dewormed', mapping_vet),
                ('Sterilized', mapping_vet),
                ('Health', mapping_health)]:
    train[col] = train[col].map(mp)
    test[col] = test[col].map(mp)
```

我们还删掉了 `Name`、`RescuerID`、`PetID` 等字段：名字缺失多且自由度过高，编号类字段对预测没有直接意义，留着容易让模型记住训练集。文本统计特征与描述正文合并后，少数缺失值统一用 0 填充。到这一步，数据就整理成了一张可以直接做特征工程的表。

### 2.3 模型训练与超参数调优

第三步是模型训练与调参。我们首先把所有类别列做 one-hot 编码，把数值列缩放到 0 到 1 之间：

```python
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# 需要 one-hot 的类别列
cat_cols = ['Type', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
            'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
            'Main_Breed', 'Main_Breed_animal', 'Cross_Breed', 'description_sentiment']
# 需要缩放的数值列
float_cols = ['Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt',
              'Score', 'Magnitude', 'Sentiment']

# handle_unknown='ignore'：测试集出现训练时没见过的类别也不报错
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
ohe.fit(train[cat_cols])

train_x = pd.DataFrame(ohe.transform(train[cat_cols]),
                       columns=ohe.get_feature_names(cat_cols), index=train.index)
test_x = pd.DataFrame(ohe.transform(test[cat_cols]),
                      columns=ohe.get_feature_names(cat_cols), index=test.index)

# MinMaxScaler 把数值压到 0~1，避免 Fee 这类大数值支配模型
mm = MinMaxScaler()
for col in float_cols:
    train_x[col] = mm.fit_transform(train[[col]])
    test_x[col] = mm.transform(test[[col]])
```

这里有一个容易踩的坑：`MinMaxScaler` 和 `OneHotEncoder` 都只能在训练集上 `fit`，再对测试集 `transform`。如果先把训练集和测试集拼起来一起拟合，就等于提前看到了测试集的分布，属于数据泄漏，交叉验证的分数会虚高。

接着我们试了一次 PCA 降维，看看能不能去掉冗余特征：

```python
from sklearn.decomposition import PCA

# 保留 95% 的方差，让 PCA 自己决定留多少维
pca = PCA(0.95, random_state=0)
pca.fit(train_x)

train_pca = pca.transform(train_x)
test_pca = pca.transform(test_x)

# 材料里的结果：特征被压缩到 35 列
print(train_pca.shape)   # (14988, 35)
```

PCA 是一种"把很多相关特征合并成少数几个综合特征"的方法，适合特征多、样本少、容易过拟合的场景。但降维也会损失一部分细节信息，所以我们没有直接认定它更好，而是把原始特征和 PCA 特征都拿去做交叉验证对比。

然后我们同时训练四个模型，统一用 5 折交叉验证，并且同时记录 F1 和二次加权卡帕：

```python
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# 把比赛的评分标准包装成 sklearn 能识别的 scorer
kappa_scorer = make_scorer(cohen_kappa_score, weights='quadratic')

models = [
    ('RF', RandomForestClassifier()),          # 随机森林
    ('GB', GradientBoostingClassifier()),       # 梯度提升树
    ('DT', DecisionTreeClassifier()),           # 单棵决策树
    ('Xgb', XGBClassifier()),                   # XGBoost
]

for name, model in models:
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv = cross_validate(model, train_x, train['AdoptionSpeed'],
                        cv=kfold,
                        scoring={'F1': 'f1_weighted', 'kappa': kappa_scorer},
                        return_train_score=True)
    # 同时打印训练集和验证集的 kappa，差距大就说明过拟合
    print(name, cv['test_kappa'].mean(), cv['train_kappa'].mean())
```

结果很有意思。在原始特征上，XGBoost 的验证 kappa 最高，约 0.3368；但它的训练 kappa 是 0.6392，验证和训练差得比较多。随机森林的训练 kappa 高达 0.9976，几乎把训练集背下来了，验证只有 0.3165，是典型的过拟合。用 PCA 特征后，所有模型的分数都下降了，最好的梯度提升也只有 0.2482，说明这次降维丢掉了对预测有用的信息，我们决定不用 PCA 特征。

接着我们做了一轮随机搜索式的调参，给每个模型换上更有针对性的参数：

```python
# 依据上一轮结果，给每个模型加上正则化或复杂度限制
models = [
    ('RF', RandomForestClassifier(class_weight='balanced', n_estimators=200,
                                  max_features=0.3, max_depth=50,
                                  criterion='entropy')),
    ('Xgb', XGBClassifier(n_estimators=200)),
    ('GB', GradientBoostingClassifier(n_estimators=200, max_depth=25,
                                      loss='deviance', learning_rate=0.5)),
    ('DT', DecisionTreeClassifier(class_weight='balanced', max_features=0.3,
                                  max_depth=5, criterion='entropy')),
]

# 仍然用 5 折交叉验证和同一套评分
for name, model in models:
    cv = cross_validate(model, train_x, train['AdoptionSpeed'],
                        cv=kfold, scoring={'F1': 'f1_weighted', 'kappa': kappa_scorer},
                        return_train_score=True)
    print(name, cv['test_kappa'].mean(), cv['train_kappa'].mean())
```

调参后随机森林的验证 kappa 升到约 0.3407，是这一轮里最高的，但训练 kappa 仍然接近 0.997，过拟合没有解决；XGBoost 的验证 kappa 约 0.3328；梯度提升的验证 kappa 约 0.3051。分数虽然略有提高，但"训练远好于验证"的问题一直存在，这提醒我们光靠调参不够，还得从特征层面做减法。

作为对照，我们还试了另一条路线：把 one-hot 后的特征直接喂给 CatBoost。CatBoost 是专门处理类别特征的梯度提升库，内置了过拟合检测器：

```python
from catboost import CatBoostClassifier

cat_params = {
    'iterations': 15000,    # 最多训练 15000 轮
    'learning_rate': 0.03,  # 小学习率配多轮数，通常更稳
    'depth': 7,             # 每棵树的最大深度
    'verbose': 1000,        # 每 1000 轮打印一次进度
    'od_wait': 1000,        # 连续 1000 轮验证分数不升就停
    'eval_metric': 'Accuracy',
}
model = CatBoostClassifier(**cat_params)

# 传入验证集，让过拟合检测器自己决定什么时候停
model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

材料里的训练日志显示，模型在第 1304 轮达到最好的验证准确率 0.4027，随后因为连续 1000 轮没有提升而被提前停止，最终只保留了前 1305 轮。这个结果说明 CatBoost 的早停机制确实能防止继续无效训练，但单看准确率并不能完全代表卡帕分数，所以我们最终仍以交叉验证的 kappa 作为选型依据。

### 2.4 模型测试与指标报告

最后一步是特征筛选、最终训练和提交。我们先用一个梯度提升模型算出每个特征的重要度：

```python
# 在全量训练集上拟合，拿到特征重要度
gb = GradientBoostingClassifier()
gb.fit(train_x, target)

importance = pd.DataFrame({
    'Attribute': train_x.columns,
    'Importance': gb.feature_importances_,
}).sort_values('Importance', ascending=False)

# 只保留重要度大于 0 的特征
# one-hot 之后有大量几乎不出现的品种列，它们只会增加噪声
important_features = importance[importance['Importance'] > 0]['Attribute'].values

# 材料里排在最前面的特征是：
# Age 0.2266、PhotoAmt 0.1181、Mixed Breed 0.1081、
# Sterilized_No 0.0616、Quantity 0.0511
print(importance.head())
```

这个结果和常识一致：年龄、照片数量、是否混血、是否绝育、同窝数量，都是领养者很在意的信息。有意思的是 `PhotoAmt` 的重要度排得很靠前，说明"照片多不多"比很多文字描述更能影响领养速度，这也呼应了比赛背景里提到的照片特征。

接着我们用筛出来的重要特征重新交叉验证，并训练最终模型：

```python
# 只用重要特征再做一次 5 折交叉验证
cv = cross_validate(gb, train_x[important_features], target,
                    cv=kfold,
                    scoring={'F1': 'f1_weighted', 'kappa': kappa_scorer},
                    return_train_score=True)

print('test_kappa:', cv['test_kappa'].mean())    # 0.3327
print('train_kappa:', cv['train_kappa'].mean()) # 0.4039

# 在全量训练集上重新拟合，然后预测测试集
gb.fit(train_x[important_features], target)
pred = gb.predict(test_x[important_features])

# 按比赛格式输出：PetID + AdoptionSpeed
output = pd.DataFrame({
    'PetID': test_petID,
    'AdoptionSpeed': pred,
})
output.to_csv('submission.csv', index=False)
```

最终这套模型的验证 kappa 约 0.3327，训练 kappa 约 0.4039，两者的差距比调参后的随机森林小得多，说明虽然分数不是最高，但泛化更可信。材料的结论也是选它做提交，而不是选训练集上几乎满分、验证集却掉下来的随机森林。

我们也把几个关键数字整理成一张对照表，方便回顾：

| 方案 | 验证 kappa | 训练 kappa | 结论 |
| --- | --- | --- | --- |
| XGBoost（原始特征） | 0.3368 | 0.6392 | 分数不错，但过拟合明显 |
| 随机森林（调参后） | 0.3407 | 0.9971 | 验证最高，严重过拟合 |
| 梯度提升（重要特征） | 0.3327 | 0.4039 | 分数略低，泛化最稳，最终采用 |
| PCA 后的梯度提升 | 0.2482 | 0.4145 | 降维损失信息，弃用 |

提交前我们还检查了输出文件的行数和列名：行数应当和测试集一样是 3972，列名必须是 `PetID` 和 `AdoptionSpeed`。比赛里因为提交格式不对而白丢分的情况并不少见，这一步不能省。

---

## 三、总结

本文中我们完成了 PetFinder 领养速度预测的完整流程。我们首先通过 EDA 摸清了数据：训练集 14993 条、测试集 3972 条，目标分成 0 到 4 五个有序等级，其中"超过 100 天未领养"最多、"当天领养"只占约 2.7%，数据明显不均衡；同时确认了缺失主要集中在 `Name` 和 `Description` 两列，训练集没有重复行。其次我们做了数据清洗：从描述文本里抽出长度、标点、大小写等统计特征并清理正文，把品种和体型、毛长、绝育等数字编码翻译成类别，删掉无意义的主品种缺失记录和猫狗混物种记录，并构造了"是否混血"特征。接着我们做了特征工程与模型对比：类别列 one-hot、数值列 MinMax 缩放，试过 95% 方差的 PCA，用 5 折交叉验证同时看 F1 和二次加权卡帕，比较了随机森林、梯度提升、决策树和 XGBoost，还试了带早停的 CatBoost。最后我们按特征重要度筛出真正有用的列，用梯度提升做最终模型，验证 kappa 约 0.3327、训练 kappa 约 0.4039，生成测试集提交文件。

这次比赛给我们最大的启发有两点。第一，**评价指标决定建模方向**：领养速度是有序等级，二次加权卡帕会重罚"错得远"的预测，所以不能只盯准确率，交叉验证也必须用同一个指标。第二，**分数最高不等于方案最好**：随机森林在验证集上略高，但训练集几乎满分，泛化风险很大；反而是特征更少、训练验证差距更小的梯度提升模型更值得提交。特征重要度还告诉我们，年龄、照片数量、是否混血这些简单信息，往往比长篇描述更有预测力。
