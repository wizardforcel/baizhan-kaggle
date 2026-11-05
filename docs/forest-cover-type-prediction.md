# Forest Cover Type Prediction

## 一、比赛简介

### 1.1 比赛背景

这场比赛要解决的是一个多分类问题：根据一块土地的地图测量信息，判断它最主要的森林覆盖类型。材料说明，研究区域位于美国北科罗拉多州 Roosevelt National Forest 的四个荒野区域，每条记录代表一块 30m × 30m 的地块。七种覆盖类型分别是 Spruce/Fir、Lodgepole Pine、Ponderosa Pine、Cottonwood/Willow、Aspen、Douglas-fir 和 Krummholz。

材料 `p001_16.txt` 指出，研究区包含 Rawah、Neota、Comanche Peak 和 Cache la Poudre 四个荒野区域，这个背景说明了为什么 `Wilderness_Area` 会有地理解释力。

这些数据把自然环境翻译成了表格。比如 `Elevation` 是海拔，`Aspect` 是方位角，`Slope` 是坡度；水文、道路和火灾点用距离表示；`Hillshade_9am`、`Hillshade_Noon`、`Hillshade_3pm` 是不同时间的阴影指数。除此之外，`Wilderness_Area1` 到 `Wilderness_Area4` 表示四个荒野区域，`Soil_Type1` 到 `Soil_Type40` 表示土壤类型。难点在于：特征既有连续数值，也有大量 0/1 指示列，森林类型又可能由海拔、距离、土壤和区域的组合共同决定。

### 1.2 比赛目标

我们需要用训练集学习 `Cover_Type`，再为测试集每一行预测一个 1 到 7 的类别。材料中的训练集有 15120 行、56 列，其中包括 `Id`、54 个输入特征和目标列 `Cover_Type`。七个类别各有 2160 条训练记录，因此目标分布是均衡的。输入字段可以按三组理解：第一组是海拔、坡度、距离、阴影等数值变量；第二组是 4 个荒野区域二值变量；第三组是 40 个土壤类型二值变量。提交时还要读取 `sampleSubmission.csv`，并将预测类别写入 `Cover_Type`。

### 1.3 评价标准

材料中的提交标题明确使用 `accuracy`，教程也在交叉验证和搜索时设置 `scoring="accuracy"`。因此我们采用分类准确率：预测正确的样本数除以样本总数，数值越高越好。准确率适合本题的原因是七个类别数量相等；如果模型只偏向某一类，也不能靠类别数量不均衡来取得虚高分数。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取训练数据，检查表头、形状、数据类型和类别分布。这样做是为了先确认哪些列可以直接用于模型，以及目标是否存在明显偏斜。

```python
import numpy as np
import pandas as pd

# 先同时读取训练集、测试集和提交模板，保证后面列名一致
train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')
submission = pd.read_csv('../input/forest-cover-type-prediction/sampleSubmission.csv')

# 用形状和缺失值检查数据是否完整，避免把 Id 或空列误当成特征
print(train.shape)
print(train.dtypes.value_counts())
print(train.isna().sum().sum())
print(train.groupby('Cover_Type').size())
```

我们看到训练集是 `(15120, 56)`，所有字段都是整数类型，缺失值统计为 0。`Cover_Type` 的 1 到 7 类都出现 2160 次，说明训练集是均衡的。接着我们查看数值特征的分布和相关关系：材料指出连续变量并不都服从正态分布，`Slope`、水文距离、道路距离和火灾点距离有右偏现象；阴影变量之间、海拔与道路或火灾点距离之间存在较明显的关系。这提醒我们不要只看单列大小，还要考虑组合关系。

```python
# Id 只是样本编号，不表示地理规律，因此探索特征关系时先排除它
num_features = train.drop(columns=['Id', 'Cover_Type'])

# 描述统计帮助我们观察量纲、极值和偏斜，不把不同单位简单混为一谈
print(num_features.describe().T)

# 查看数值变量之间的相关性，为后续构造交互特征提供线索
corr = num_features.corr()
print(corr['Elevation'].sort_values(ascending=False).head())
```

我们还可以按区域和土壤类型统计覆盖类别。材料中的 EDA 将 4 个荒野区域、40 个土壤类型从独立的 0/1 列还原成类别后作图，发现覆盖类型与这些类别有联系；其中 `Soil_Type4` 出现在所有森林覆盖类型中。这个结果支持后面把土壤列重新解释为更有含义的特征，而不是盲目删除它们。

为了把这种关系看得更直接，我们把 0/1 指示列还原成一个类别编号，再统计每类覆盖类型分别落在哪些区域。这里不能把 40 列土壤指标当成 40 个独立数字，因为它们本来描述的是同一件事的不同取值。

```python
# 找出每条记录属于哪个荒野区域：4 列里值为 1 的那一列就是区域编号
area_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]
train['Wilderness_Area'] = train[area_cols].values.argmax(axis=1) + 1

# 按覆盖类型和区域做交叉统计，观察哪个区域更容易出现哪类森林
print(pd.crosstab(train['Cover_Type'], train['Wilderness_Area']))
```

交叉统计的结果显示，不同区域里的覆盖类型比例并不完全相同，这说明区域信息并不是噪声。不过荒野区域只有 4 个取值，单靠它无法区分全部七类，所以我们仍然需要海拔、坡度、距离和土壤等信息一起参与判断。

### 2.2 数据清洗

我们其次处理输入列。`Id` 只是编号，不能帮助模型识别森林类型，所以从训练和测试特征中去掉。材料中的基础教程把数据拆成数值列、荒野区域列和土壤列；原始字段没有缺失，因而这里不需要凭空填补数据。对于特征工程中新产生的列，我们仍然统一用 0 填补可能出现的空值，并把数值类型压缩到合适的整数或浮点类型。

```python
# 标签单独保存，模型只接收真正的输入特征
y = train['Cover_Type']
X = train.drop(columns=['Id', 'Cover_Type']).copy()
X_test = test.drop(columns=['Id']).copy()

# 原始表没有缺失；工程特征可能产生空值，所以统一用 0 表示缺少该组合信息
X = X.fillna(0)
X_test = X_test.fillna(0)

# 用相同的列顺序对齐训练集和测试集，避免预测时特征错位
X_test = X_test[X.columns]
```

接着我们把距离、海拔和水文信息变成组合特征。材料中的做法包括道路距离取 `log(v+1)`，构造海拔减垂直水文距离，以及水文、道路、火灾点之间的和与差。这些变量的直观含义是：同一个地块的地形高度和周围资源距离可能共同反映环境条件；取对数可以减弱道路距离很大时的影响。

```python
def misc_features(data):
    df = data.copy()
    # 距离可能跨度很大，log(v+1) 能在保留顺序的同时压缩极端值
    df['Horizontal_Distance_To_Roadways_Log'] = np.log1p(
        df['Horizontal_Distance_To_Roadways'])
    # 用差值表达海拔与水文位置的相对关系
    df['Water_Elevation'] = (
        df['Elevation'] - df['Vertical_Distance_To_Hydrology'])
    # 用和、差表达水文、道路和火灾点之间的相对位置
    df['Hydro_Fire_1'] = (df['Horizontal_Distance_To_Hydrology'] +
                          df['Horizontal_Distance_To_Fire_Points'])
    df['Hydro_Fire_2'] = abs(df['Horizontal_Distance_To_Hydrology'] -
                             df['Horizontal_Distance_To_Fire_Points'])
    df['Hydro_Road_1'] = abs(df['Horizontal_Distance_To_Hydrology'] +
                             df['Horizontal_Distance_To_Roadways'])
    df['Hydro_Road_2'] = abs(df['Horizontal_Distance_To_Hydrology'] -
                             df['Horizontal_Distance_To_Roadways'])
    return df.fillna(0)

X = misc_features(X)
X_test = misc_features(X_test)
```

完成距离和海拔的组合后，我们处理土壤列。材料指出，原始的 40 个 `Soil_Type` 列其实是对同一个土壤变量的独热编码，拆成很多列只是为了方便某些模型使用。对树模型来说，把 40 列还原成一个有序类别反而更好：既能减少特征数量，也方便后面根据 USFS 的 ELU 编码提取气候区和地质区。

```python
# 40 个 0/1 列中只会有一列为 1，所以加权求和就得到 1 到 40 的土壤编号
def soil_to_category(data):
    df = data.copy()
    df['Soil_Type'] = 0
    for i in range(1, 41):
        df['Soil_Type'] += i * df[f'Soil_Type{i}']
    return df

X = soil_to_category(X)
X_test = soil_to_category(X_test)
# 原始 40 列已经包含在 Soil_Type 中，后面可以统一删除
soil_features = [f'Soil_Type{i}' for i in range(1, 41)]
```

有了土壤编号，我们就能查 ELU 编码表。ELU 编码的第一位表示气候区，第二位表示地质区；材料还根据土壤描述把地表覆盖分成五档、把岩石大小分成三档。这些新列不是凭空造出来的，而是把原始数据说明里已经存在、但被编码隐藏的信息重新展开。

### 2.3 模型训练与超参数调优

我们先用决策树理解交叉验证。单棵未限制的树在训练集上准确率为 1.00，但测试集准确率只有 0.76，说明树把训练样本记得太细，产生了过拟合。普通 10 折交叉验证的平均准确率是 0.7854947702813675；使用 `StratifiedKFold` 后平均准确率是 0.7860821951852012。分层交叉验证会让每一折都尽量保持七类比例，更适合均衡的多分类数据。

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# 固定随机状态，保证每次拆分可复现，便于公平比较参数
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=15)
tree = DecisionTreeClassifier(random_state=12)

# 每一折都用不同的验证数据，平均值比一次随机切分更稳
scores = cross_val_score(tree, X, y, cv=skf, scoring='accuracy')
print(scores.mean(), scores.std())
```

然后我们用网格搜索调节 `max_depth`、`max_features` 和 `min_samples_split`。网格搜索会逐一尝试给出的组合；材料中的参数候选为深度 5、15、30、50、65、80，特征比例 0.2、0.4、0.6、0.8，最小分裂样本数 2、5、8、10、20。搜索得到的决策树参数是 `max_depth=15`、`max_features=0.8`、`min_samples_split=2`，交叉验证最佳分数为 0.7906383714396853。随机搜索用 50 次尝试也得到同一组最佳参数。这个结果还说明，深度 80 的树并不比深度 15 更强，可见控制树的复杂度比一味加深更重要；对这类结构化数据，与其用一棵深树硬记所有边界，不如让很多棵浅一点的树合作。

```python
from sklearn.model_selection import GridSearchCV

# 用有限的候选范围控制树的复杂度，避免只追求训练集分数
param_grid = {
    'max_depth': (5, 15, 30, 50, 65, 80),
    'max_features': (0.2, 0.4, 0.6, 0.8),
    'min_samples_split': (2, 5, 8, 10, 20)
}
search = GridSearchCV(
    DecisionTreeClassifier(random_state=12), param_grid=param_grid,
    cv=skf, scoring='accuracy', n_jobs=-1)
search.fit(X, y)
print(search.best_params_)
print(search.best_score_)
```

为了得到材料中更好的结果，我们最后采用 ExtraTreesClassifier。它可以理解为许多随机化决策树的集合，单棵树的偶然错误可以由投票平均掉。基础数据不做特征工程时，12 折交叉验证的平均准确率为 0.87943；加入数值组合特征和土壤语义特征后，平均准确率提高到 0.91038。

我们把 40 个土壤指示列编码为一个 `Soil_Type`，再用 ELU 编码提取 `Climatic_Zone` 和 `Geologic_Zone`，并加入 `Surface_Cover`、`Rock_Size`。同时，我们构造土壤与荒野区域的交互项，例如 `Climate_Area2`、`Rock_Area1` 和 `Surface_Area4`，最后删除原始 40 个土壤列，达到降维目的。树模型不需要把这些类别强行标准化，反而能直接利用这些离散切分。

下面是从 ELU 编码提取气候区和地质区的写法。字典里的数字是 USFS 给每个土壤类型分配的生态土地类型编号，我们只取它的第 1 位和第 2 位，就避免了让模型自己去猜「26 号土壤和第 3 个区域内在一起意味着什么」。

```python
# 土壤编号 1 到 40 对应的 ELU 编码，来自原始数据的说明
ELU_CODE = {1: 2702, 2: 2703, 3: 2704, 4: 2705, 5: 2706, 6: 2717,
            7: 3501, 8: 3502, 9: 4201, 10: 4703, 11: 4704, 12: 4744,
            # 中间编号省略，实际使用时补全到 40
            40: 8776}

def zone_features(df):
    df = df.copy()
    # 第 1 位是气候区，有从低海拔到高海拔的自然顺序
    df['Climatic_Zone'] = df['Soil_Type'].apply(
        lambda x: int(str(ELU_CODE[x])[0]))
    # 第 2 位是地质区，只是类别编号，没有大小顺序
    df['Geologic_Zone'] = df['Soil_Type'].apply(
        lambda x: int(str(ELU_CODE[x])[1]))
    # 土壤组合与荒野区域的交互项，帮助模型看到「某区域内某类土壤」的效果
    df['Climate_Area4'] = df['Wilderness_Area4'] * df['Climatic_Zone']
    df['Rock_Area1'] = df['Wilderness_Area1'] * df['Rock_Size']
    return df

X = zone_features(X)
X_test = zone_features(X_test)
# 删除原始 40 个土壤列，只保留语义更清楚的派生特征，减少维度
X = X.drop(columns=soil_features)
X_test = X_test.drop(columns=soil_features)
```

我们还要注意，特征工程的结果必须用交叉验证检验，而不能只看训练集准确率。下面我们用 ExtraTreesClassifier 做 12 折验证：每一折只用训练部分拟合模型，在验证部分打分，同时把每折模型对测试集的概率保存下来，供最后平均。

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

# 用 12 折分层验证评估特征工程，test_preds 用于平均测试集概率
skf12 = StratifiedKFold(n_splits=12, shuffle=True, random_state=0)
test_preds = np.zeros((X_test.shape[0], 7))
scores = []
for fold, (train_idx, valid_idx) in enumerate(skf12.split(X, y)):
    model = ExtraTreesClassifier(
        n_estimators=118, min_samples_split=2, max_features=14,
        random_state=0, n_jobs=-1)
    # 每一折只在训练部分拟合，防止验证标签泄漏
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    valid_pred = model.predict(X.iloc[valid_idx])
    scores.append(accuracy_score(y.iloc[valid_idx], valid_pred))
    # 平均概率比只使用某一折的预测更稳
    test_preds += model.predict_proba(X_test)

print(np.mean(scores))
pred = np.argmax(test_preds, axis=1) + 1
```

### 2.4 模型测试与指标报告

我们最后使用每折模型对测试集输出概率，再对 12 折结果求和并取最大类别。这样做与只训练一棵树相比，能减少一次随机抽样带来的波动。预测完成后，我们按照提交模板保留 `Id`，把类别写入 `Cover_Type`，并保存为 CSV。

```python
# 概率累加后取最大下标，得到每条测试记录的最终类别
submission['Cover_Type'] = pred

# 不保存 pandas 自动生成的行号，否则提交列数会多一列
submission.to_csv('features_submission.csv', index=False)

# 在本地验证集上报告准确率，提交前先确认预测长度和标签范围
print(len(submission))
print(submission['Cover_Type'].value_counts().sort_index())
```

材料给出的特征工程方案在 12 折验证中的各折准确率约为 0.90476 到 0.91746，平均准确率为 0.91038，总训练时间为 62.47 秒。对应的原始特征基线平均准确率为 0.87943、训练时间为 70.64 秒。也就是说，土壤类型的语义拆分、环境距离交互项和降维不只是增加列数，而是让模型看到了更有解释力的环境关系；同时，最终方案的训练时间也低于基线。

## 三、总结

本文中我们完成了 Forest Cover Type Prediction 的一条完整解题流程：我们首先检查了 15120 行训练数据、字段类型、缺失值和七类均衡分布；其次删除无预测意义的 `Id`，并处理新特征可能产生的空值；接着用水文、道路、火灾点和海拔的组合关系扩展数值特征，再从 40 个土壤指示列中提取气候区、地质区、地表覆盖和岩石大小；然后用分层交叉验证和网格搜索理解决策树的过拟合与参数选择；最后使用 12 折 ExtraTreesClassifier 平均测试概率并生成提交文件。

从材料中的结果看，原始特征的 ExtraTrees 平均准确率是 0.87943，加入特征工程后达到 0.91038。这个案例说明，表格数据的提升不一定来自更复杂的模型，也可以来自对字段含义的理解：把土壤编号还原成环境信息，把距离变量组合成相对关系，再用交叉验证检验每一步是否真的有效。