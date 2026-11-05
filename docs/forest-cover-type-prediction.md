# Forest Cover Type Prediction：森林覆盖类型预测

## 一、比赛简介

### 1.1 比赛背景

一片森林里长什么树，和海拔、坡度、日照、土壤、离水源多远都有关系。美国森林管理局把这些信息整理成表格，并为每个地点标注了 7 种森林覆盖类型。人工去调查每一块地很慢，所以比赛希望用机器学习根据地形和土壤记录自动判断覆盖类型。

这是一份典型的结构化表格分类数据：没有图片、没有文字，只有数字列和一个类别标签。它的难点不在于数据量特别大，而在于很多列是 one-hot 编码的土壤类别，单看一列信息很少，只有把土壤背后的生态含义恢复出来，模型才容易学到规律。

### 1.2 比赛目标

训练集 `train.csv` 有 15120 行、56 列，测试集没有 `Cover_Type` 这一列。特征包括：

- `Elevation`、`Aspect`、`Slope`：海拔、坡向和坡度；
- 到水文、道路、火点的水平/垂直距离；
- `Hillshade_9am`、`Hillshade_Noon`、`Hillshade_3pm`：三个时间的山体阴影指数；
- `Wilderness_Area1~4`：荒野区域的二进制变量；
- `Soil_Type1~40`：40 个土壤类型的二进制变量；
- `Cover_Type`：目标标签，表示 1~7 中的一种森林覆盖类型。

我们的输出不是概率，而是每个测试样本的一个整数类别，并且 `Id` 必须和原测试集保持一一对应。

### 1.3 评价标准

比赛使用多分类准确率（accuracy）：预测正确的行数除以总行数，越高越好。训练集的 7 个类别各有 2160 行，类别完全均衡，所以不需要人为给某类加权。材料中的方案还用 `recall`、`F1` 或 `log_loss` 做辅助分析，但最终提交仍然是 `Cover_Type` 类别预测。

---

## 二、解题过程

我们首先读取训练和测试表，检查类别、缺失值和特征分组；其次把连续特征做组合、把土壤 one-hot 列还原成有含义的生态特征；接着用 ExtraTrees 建立基线并用分层交叉验证调参数；最后在所有训练数据上重新训练，用每折模型的概率平均产生提交文件。

### 2.1 数据探索（EDA）

我们先读入文件并确认标签列：

```python
import numpy as np
import pandas as pd

train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')
submission = pd.read_csv('../input/forest-cover-type-prediction/sampleSubmission.csv')

# 标签只存在于训练集，测试集没有 Cover_Type
print(train.shape)                         # (15120, 56)
print(train.columns.difference(test.columns))

# 把 1~7 编码为 0~6，方便 sklearn 模型使用
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train['Cover_Type'] = encoder.fit_transform(train['Cover_Type'])
print(train['Cover_Type'].value_counts())
```

我们还检查了原始列的类型和范围。`Elevation`、距离、光照是连续/整数数值；`Wilderness_Area` 和 `Soil_Type` 是只有 0、1 的指示变量。因为树模型不要求所有特征具有相同量纲，所以我们没有做标准化，但会特别关注极端值和几乎恒定的列。

```python
# 先看有没有缺失值，再看哪些列只有一个取值
print(train.isnull().sum().sum())
constant_cols = [c for c in train.columns if train[c].nunique() <= 1]
print('常量列:', constant_cols)

# 观察土壤 one-hot 编码的稀疏程度
soil_cols = [f'Soil_Type{i}' for i in range(1, 41)]
soil_rate = train[soil_cols].mean().sort_values()
print(soil_rate.head())
```

很多土壤列的 1 的比例不到 1%，它们并不是没有信息，而是表示非常少见的土壤。我们不能简单按「出现少」全部删除；相反，我们把土壤编号映射到 ELU（Ecological Landtype Unit，生态土地类型单元）编码，提取其中的气候带和地质带。

### 2.2 数据清洗

我们先处理连续列中的缺失值，再把距离、海拔等变量组合成更符合地理直觉的量：

```python
import math

def misc_features(data):
    df = data.copy()
    # 用 float64 计算，避免交互项的中间结果溢出
    for col in df.columns:
        if df[col].dtype.name.startswith('float'):
            df[col] = df[col].astype('float64')

    # 对数变换压缩道路距离的长尾
    df['Horizontal_Distance_To_Roadways_Log'] = \
        df['Horizontal_Distance_To_Roadways'].apply(lambda v: math.log(v + 1))
    # 海拔减去垂直水文距离，得到水面相对高度
    df['Water Elevation'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    # 水源、火点、道路之间的和与差，表达地理关系
    df['Hydro_Fire_1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['Hydro_Fire_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['Hydro_Road_1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
    df['Hydro_Road_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])

    # 缺失值统一补 0，并压缩数值类型减少内存
    df.fillna(0, inplace=True)
    return df

train = misc_features(train)
test = misc_features(test)
```

接着我们把 40 个土壤指示列合成一个 `Soil_Type` 编号。one-hot 的意思是「40 列中只有一列为 1」，因此用加权求和可以恢复原来的编号：

```python
ELU_CODE = {
    1:2702, 2:2703, 3:2704, 4:2705, 5:2706, 6:2717,
    7:3501, 8:3502, 9:4201, 10:4703, 11:4704, 12:4744,
    13:4758, 14:5101, 15:5151, 16:6101, 17:6102, 18:6731,
    19:7101, 20:7102, 21:7103, 22:7201, 23:7202, 24:7700,
    25:7701, 26:7702, 27:7709, 28:7710, 29:7745, 30:7746,
    31:7755, 32:7756, 33:7757, 34:7790, 35:8703, 36:8707,
    37:8708, 38:8771, 39:8772, 40:8776
}

def categorical_encoding(input_df):
    df = input_df.copy()
    df['Soil_Type'] = 0
    for i in range(1, 41):
        # 只有对应 one-hot 列为 1 时，才加上土壤编号 i
        df['Soil_Type'] += i * df[f'Soil_Type{i}']
    return df

train = categorical_encoding(train)
test = categorical_encoding(test)
```

ELU 编码的第一位表示气候带，第二位表示地质带。我们用字符串取位的方法构造两个离散特征：

```python
def add_elu_features(input_df):
    df = input_df.copy()
    df['Climatic_Zone'] = input_df['Soil_Type'].apply(
        lambda x: int(str(ELU_CODE[x])[0]))
    df['Geologic_Zone'] = input_df['Soil_Type'].apply(
        lambda x: int(str(ELU_CODE[x])[1]))
    return df

train = add_elu_features(train)
test = add_elu_features(test)
```

在材料的完整方案里，我们还按照土壤说明把表面覆盖类型（`Surface_Cover`）和石块大小（`Rock_Size`）编码出来，并构造 `Climate_Area2`、`Rock_Area1` 等土壤×荒野区域交互项。最后删除原来的 40 个 `Soil_Type1~40`，避免同一信息重复出现，既减少维度，也让树更容易找到有意义的分裂点。

### 2.3 模型训练与超参数调优

我们选用 ExtraTrees（极端随机树）作为主模型。它和随机森林一样由多棵树投票，但分裂阈值也带有更强的随机性，通常能降低树之间的相关性。材料中的交叉验证函数同时完成三件事：保存每折的验证预测、把各折测试概率相加、记录特征重要性。

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

features = [c for c in train.columns if c not in ['Id', 'Cover_Type']]
X = train[features]
y = train['Cover_Type']
X_test = test[features]

NUM_FOLDS = 12
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)
test_preds = np.zeros((X_test.shape[0], 7))
scores = np.zeros(NUM_FOLDS)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    model = ExtraTreesClassifier(
        n_estimators=118,        # 树的数量，增加后投票更稳定
        min_samples_split=2,    # 允许树继续细分，保持非线性表达能力
        max_features=14,        # 每次分裂随机查看 14 个特征
        random_state=0,
        n_jobs=-1)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    valid_pred = model.predict(X.iloc[va_idx])
    scores[fold] = accuracy_score(y.iloc[va_idx], valid_pred)
    # 测试集用概率而不是硬标签，最后平均更稳定
    test_preds += model.predict_proba(X_test)

print('CV accuracy:', scores.mean())
```

调参时我们先在原始特征上建立 baseline，12 折平均准确率为 **0.87943**；加入普通的距离、海拔交互项后，再加入土壤生态特征和交互项，最终平均准确率提升到 **0.91038**。材料中最终配置的训练时间约 62.47 秒，说明在这份一万多行的数据上，118 棵树已经足够稳定。

这里我们使用分层交叉验证（`StratifiedKFold`），原因是它会保证每一折里 7 个类别的比例相近。如果只随机切分而不分层，某一折偶然少了某个类别，准确率会有很大的随机波动，调参结论也不可靠。

### 2.4 模型测试与指标报告

原始特征的 baseline 是 0.87943，完整特征工程后的 12 折结果如下：

```python
print('每折准确率:', np.round(scores, 5))
print('平均准确率:', round(scores.mean(), 5))
print('训练总时间:', round(times.sum(), 2), 's')

# 多折概率相加后取最大类别，LabelEncoder 的结果是 0~6
pred_encoded = np.argmax(test_preds, axis=1)
# 写提交前还原为原始标签 1~7
pred_cover_type = encoder.inverse_transform(pred_encoded)

submission['Cover_Type'] = pred_cover_type
submission.to_csv('features_submission.csv', index=False)
```

最终 12 折的准确率大约在 0.90476 到 0.91746 之间，平均 **0.91038**。这个提升不是来自换一个更复杂的模型，而是来自对土壤信息的解释：把 40 个稀疏的 one-hot 列还原为气候带、地质带、表面覆盖和岩石大小，模型看到的是更接近真实生态规律的特征。

提交前我们还做了三项格式检查：`Id` 行数是否和测试集相同、`Cover_Type` 是否全部在 1~7、CSV 是否没有多余索引列。尤其是最后一项，`to_csv(..., index=False)` 不能省，否则会多出一列无名索引，导致提交文件格式不符合要求。

---

## 三、总结

本文中我们完成了 Forest Cover Type Prediction 的完整分类流程。我们首先通过 EDA 确认训练集有 15120 行、56 列，并发现土壤 one-hot 变量背后隐藏着生态信息；其次对数值列做长尾变换、构造水源/道路/火点的距离交互，再把土壤编号映射到 ELU 代码并提取气候带、地质带、表面覆盖和岩石大小；接着使用 `n_estimators=118`、`max_features=14` 的 ExtraTrees 进行 12 折分层交叉验证，原始特征 baseline 为 0.87943，特征工程后提升到 0.91038；最后平均各折测试概率、还原标签编码，生成符合格式的 `features_submission.csv`。

这个比赛给我们的启发是：**one-hot 列不一定只能当作哑变量**。如果数据字典里包含更深一层的编码规则，我们就应该把它还原出来；对入门者来说，先理解每列代表什么，再决定是否标准化、删除或组合，往往比盲目增加模型复杂度更有效。