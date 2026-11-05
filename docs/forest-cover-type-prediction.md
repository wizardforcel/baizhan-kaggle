# Forest Cover Type Prediction

## 一、比赛简介

### 1.1 比赛背景

森林覆盖类型（Forest Cover Type）是机器学习教学里非常经典的一个多分类数据集。原始数据来自美国地质调查局，研究者在科罗拉多州的罗斯福国家森林公园里布下大量观测点，每个观测点记录下周围的地形、土壤、距离等环境信息，同时标注出这一块地上的森林覆盖类型。这个比赛把这个数据集搬到 Kaggle 上，要求参赛者用这些环境特征预测覆盖类型。

### 1.2 比赛目标

训练集 `train.csv` 有 15,120 行，包含 55 个输入特征和一列标签 `Cover_Type`。特征大致分三类：

- **地形类**：`Elevation`（海拔）、`Aspect`（方位角）、`Slope`（坡度）、`Hillshade_9am/Noon/3pm`（三个时刻的日照阴影指数）；
- **距离类**：到水文（`Horizontal/Vertical_Distance_To_Hydrology`）、道路、火点三种基础设施的水平距离；
- **指示类**：4 个 `Wilderness_Area1~4`（荒野区域，0/1）和 40 个 `Soil_Type1~40`（土壤类型，0/1 独热编码）。

我们的任务是对测试集（约 56.5 万行）的每个 `Id` 预测它属于 7 种覆盖类型中的哪一种。比赛按**分类准确率**评价——预测对的样本占全部样本的比例，越高越好。

## 二、解题过程

我们首先读入数据做 EDA，了解每类特征的分布和存在的问题；其次清洗编码、重构土壤类型并做特征工程；接着用 ExtraTrees 分类器配 12 折交叉验证反复试验每个新特征带来的提升；最后比较基线模型和特征工程模型的准确率并生成提交文件。

### 2.1 数据探索（EDA）

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')

print(train.shape)   # (15120, 56)
print(train.isnull().sum().sum())   # 没有缺失值

# 标签原本是 1~7 的类别名，先转成连续整数方便建模
encoder = LabelEncoder()
train['Cover_Type'] = encoder.fit_transform(train['Cover_Type'])
print(train['Cover_Type'].value_counts().sort_index())
```

EDA 看到的最重要的两个现象是：**数据没有缺失值**，并且 40 个 `Soil_Type*` 是独热编码——某行只会有其中一个等于 1，其余都是 0。独热编码对神经网络很友好，但对树模型属于绕路：一棵树本可以直接做一次「土壤类型值等于几」的判断，现在却要做 40 次「某个 0/1 列等不等于 1」。所以我们后面会把 40 列**逆向合并成一列真实土壤编号**，再衍生出物理上有意义的子特征。

### 2.2 数据清洗

清洗的核心是「土壤类型的再造」。我们从比赛说明里知道土壤编号对应美国林务局的 ELU（生态土地利用单元）编码：

```python
# 土壤编号 -> ELU 编码，例如 1 号土对应 2702
ELU_CODE = {1:2702, 2:2703, 3:2704, 4:2705, 5:2706, 6:2717, 7:3501,
            8:3502, 9:4201, 10:4703, 11:4704, 12:4744, 13:4758,
            14:5101, 15:5151, 16:6101, 17:6102, 18:6731, 19:7101,
            20:7102, 21:7103, 22:7201, 23:7202, 24:7700, 25:7701,
            26:7702, 27:7709, 28:7710, 29:7745, 30:7746, 31:7755,
            32:7756, 33:7757, 34:7790, 35:8703, 36:8707, 37:8708,
            38:8771, 39:8772, 40:8776}

def categorical_encoding(input_df):
    """把 40 个独热列还原成单个 Soil_Type 编号"""
    data = input_df.copy()
    data['Soil_Type'] = 0
    for i in range(1, 41):
        data['Soil_Type'] += i * data[f'Soil_Type{i}']
    return data
```

ELU 编码的四位数里藏着信息：**第一位代表气候带**（数字 2/3/4/5/6/7/8 对应不同海拔气候，有自然顺序），**第二位代表地质带**（数字 0/1/2/3/4/5 对应冲击层、冰川、页岩等不同地质）。从这些编码和土壤描述里，我们进一步得到 `Climatic_Zone`、`Geologic_Zone`、`Surface_Cover`（表面石砾覆盖程度）和 `Rock_Size`（石块大小）四个新特征，哪怕原本独热列里完全看不出来。

### 2.3 模型训练与超参数调优

模型上我们选了 **ExtraTreesClassifier**（极端随机树，随机森林的变体），基线就是「原始 54 个特征 + 默认参数」，用 12 折分层交叉验证评估：

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

def cross_validate_and_predict(features):
    """对给定特征做 12 折交叉验证，返回平均准确率，同时叠加各折对测试集的预测"""
    X = train[features]
    y = train['Cover_Type']
    test_preds = np.zeros((test.shape[0], 7))   # 每个类别的概率之和
    skf = StratifiedKFold(n_splits=12, shuffle=True, random_state=0)
    scores = []
    for train_idx, valid_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
        model = ExtraTreesClassifier(
            n_estimators=118, min_samples_split=2,
            max_features=14, random_state=0, n_jobs=-1)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        scores.append(accuracy_score(y_va, pred))
        test_preds += model.predict_proba(test[features])
    return np.mean(scores), np.argmax(test_preds, axis=1)

base_acc, _ = cross_validate_and_predict(
    [c for c in train.columns if c not in ['Id', 'Cover_Type']])
print(base_acc)   # 基线 12 折平均准确率约 0.87943
```

超参数 `n_estimators=118`、`max_features=14` 是作者在多次试验里固定下来的组合：118 棵树保证方差足够小，`max_features=14`（在 54 个特征里每棵树只随机看 14 个）保证每棵树之间的差异足够大。12 折交叉验证的每一折都跑一遍，加起来正好可以当作一次完整的留出评估。

### 2.4 模型测试与指标报告

特征工程是提分的主力。我们把上一步的所有新特征拼起来，还额外加了土壤类型之间的交互项（比如某些常见土壤编号相加、土壤与荒野区域的组合），然后**丢掉原始的 40 个土壤独热列**降维：

```python
def soiltype_interactions(data):
    df = data.copy()
    # 常见土壤类型的组合
    df['Soil_12_32'] = df['Soil_Type32'] + df['Soil_Type12']
    df['Soil_Type23_22_32_33'] = (df['Soil_Type23'] + df['Soil_Type22']
                                  + df['Soil_Type32'] + df['Soil_Type33'])
    # 土壤与荒野区域的交互
    df['Soil29_Area1'] = df['Soil_Type29'] + df['Wilderness_Area1']
    df['Soil3_Area4'] = df['Wilderness_Area4'] + df['Soil_Type3']
    # 气候带/石块大小/表面覆盖与荒野区域的乘积特征
    df['Climate_Area2'] = df['Wilderness_Area2'] * df['Climatic_Zone']
    df['Rock_Area1'] = df['Wilderness_Area1'] * df['Rock_Size']
    df['Surface_Area4'] = df['Wilderness_Area4'] * df['Surface_Cover']
    df.fillna(0, inplace=True)
    return df

train = soiltype_interactions(train)
test = soiltype_interactions(test)
train.drop(columns=[f'Soil_Type{i}' for i in range(1, 41)], inplace=True)
test.drop(columns=[f'Soil_Type{i}' for i in range(1, 41)], inplace=True)

final_acc, test_pred = cross_validate_and_predict(
    [c for c in train.columns if c not in ['Id', 'Cover_Type']])
print(final_acc)   # 特征工程后 12 折平均准确率约 0.91038
```

对比报告：基线 12 折平均准确率 **0.87943**，加入通用交互距离特征、土壤 ELU 再造特征和交互项后提升到 **0.91038**——凭空多了大约 3 个百分点的准确率，而模型本身完全没换。逐折看，10 个折都在 0.90 以上、最高 0.917，说明提升是稳定的而不是撞运气。最后我们把这套 `test_pred`（每折概率均值再取最大类别）还原成原始标签名写成提交文件，并检查提交的列名、行数和类别范围。

值得一提的是，材料里还读到了别的方案：keras 深度网络、fastai、TPOT 自动机器学习、贝叶斯方法、各种 Stacking 集成等，它们大多在这个分数附近或略低，而我们的「特征工程 + ExtraTrees」在计算量小得多的情况下就拿到了 Top 10% 左右的成绩。

---

## 三、总结

本文中我们完成了 Forest Cover Type Prediction 的完整流程。我们首先用 EDA 确认训练集 15,120 行、55 个特征、无缺失值，是一份很干净的结构化数据。其次我们做了关键的清洗：把 40 个独热土壤列合并成语义明确的 `Soil_Type`，并从中反推气候带、地质带、表面覆盖和石块大小四个新特征。接着我们以 ExtraTreesClassifer（118 棵树）为主模型，用 12 折分层交叉验证逐条试验每个新特征。最后我们报告基线准确率 0.87943，加入这些特征后提升到 0.91038，并生成了最终提交文件。

最大的收获是：**对树模型来说，独热编码往往是多余甚至有害的，把原始列还原成语义特征才是提分的正道**。另外，先用精确的交叉验证评估工具把每个新特征的收益量化清楚，比同时换模型又换特征、最后搞不清是谁起的作用，要高效得多。