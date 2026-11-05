# Well Log Reconstruction Challenge DataHack 3.0

## 一、比赛简介

### 1.1 比赛背景

Well Log Reconstruction Challenge 是 DataHack 3.0 举办的一场石油地质方向的数据竞赛。比赛源于油气勘探中的一项实际问题：在钻井过程中，会采集多种测井曲线，例如伽马射线 GR、电阻率 RDEP、声波时差 DTC、泥浆电阻率 RMED 等。这些曲线反映了地下地层的岩性、孔隙度和流体性质，是储层评价的核心依据。

实际作业中，由于仪器故障、泥浆干扰或测井段遗漏，测井曲线会出现大量缺失。参赛者需要在给定的部分曲线作为输入的情况下，重建缺失的四条目标曲线，帮助地质工程师补全井日志。数据以 LAS 文件格式提供，每口井一个文件，记录深度 DEPT 对应的多条测井值。

### 1.2 比赛目标

训练集由多口井的 LAS 文件组成，每口井包含完整的测井曲线，测试集提供了部分曲线，目标曲线 GR、RDEP、DTC、RMED 在测试集中缺失或部分缺失，需要预测。

输入特征是除目标外的其余测井曲线，输出是四条目标曲线的数值。每个样本对应一个深度点，带有多维特征。比赛要求预测每口井每个深度的四条曲线，提交文件需要包含 ID、WELL、DEPT、GR、RDEP、DTC、RMED。

### 1.3 评价标准

比赛使用均方根误差 RMSE 作为评价指标，对四条目标曲线分别计算 RMSE 并取平均，或者以整体 RMSE 衡量。RMSE 越小越好，表示预测值与真实值的偏差越小。数据中缺失值用 -999.25 标记，需要在建模前转换为空值。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先使用 lasio 读取所有的 LAS 文件，统计井的数量、每口井的深度范围和曲线数量。

```python
import os, glob
import pandas as pd
import numpy as np

DATA_DIR = '/kaggle/input/well-log-reconstruction-challenge-data-hack-3-0'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
train_files = glob.glob(os.path.join(TRAIN_DIR, '*.las'))
df_list = []
for file in train_files:
    las = lasio.read(file)
    df = las.df()
    df['DEPT'] = df.index
    df['WELL'] = las.well.WELL.value if las.well.WELL.value else os.path.basename(file).split('.')[0]
    df_list.append(df)
train_df = pd.concat(df_list, ignore_index=True)
```

合并后我们检查列名大小写问题，发现 LAS 文件中列名大小写不统一。为统一处理，我们把所有列名转为大写。

```python
train_df.columns = train_df.columns.str.upper()
test_df.columns = test_df.columns.str.upper()
```

随后我们统计各目标曲线的缺失比例。比赛规定缺失标记为 -999.25，我们先把该值替换为 NaN，再统计缺失率。GR 缺失最严重，其次是 RMED，说明重建任务挑战较大。

### 2.2 数据清洗

数据清洗的核心是处理缺失标记和列名不一致。

```python
train_df.replace(-999.25, np.nan, inplace=True)
test_df.replace(-999.25, np.nan, inplace=True)
train_df.columns = train_df.columns.str.upper()
test_df.columns = test_df.columns.str.upper()
```

目标列为 GR、RDEP、DTC、RMED。训练时我们只保留目标全部存在的行，以避免标签缺失导致模型学习错误。

```python
TARGETS = ['GR', 'RDEP', 'DTC', 'RMED']
train_clean = train_df.dropna(subset=TARGETS)
```

特征列选择遵循“在训练和测试中都存在的列”，排除 ID、WELL、DEPT 和目标列本身。

```python
features = [col for col in test_df.columns if col in train_df.columns and col not in ['ID','WELL','DEPT'] + TARGETS]
```

最后对特征做中位数插补，因为测井曲线存在局部缺失，直接删除会损失大量样本。

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(train_clean[features])
X_test_imputed = imputer.transform(test_df[features])
```

### 2.3 模型训练与超参数调优

作为基线，我们选择随机森林回归器，它天然支持多输出回归，且对非线性关系和缺失值不敏感。

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_imputed, train_clean[TARGETS])
```

训练时我们保持树的深度为 10，防止过拟合，同时树的数量设为 50 以平衡速度与精度。我们没有做复杂的超参数搜索，先确保基线能够跑通并生成合理预测。

在超参数调优思路上，我们可以逐步增加 n_estimators 到 200，尝试 max_depth 在 8 到 20 之间，以观察在验证集上的 RMSE 变化。也可以引入特征重要性分析，剔除对目标贡献极小的测井曲线，以降低噪声。

### 2.4 模型测试与指标报告

模型训练完成后，我们在测试集上做预测，并按提交格式整理。

```python
preds = model.predict(X_test_imputed)
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'WELL': test_df['WELL'],
    'DEPT': test_df['DEPT'],
    'GR': preds[:, 0],
    'RDEP': preds[:, 1],
    'DTC': preds[:, 2],
    'RMED': preds[:, 3]
})
submission = submission[['ID','WELL','DEPT','GR','RDEP','DTC','RMED']]
submission.to_csv('submission.csv', index=False)
```

由于测试集没有标签，我们在训练集上做留一井的交叉验证，模拟真实泛化能力。结果显示 GR 的 RMSE 相对较高，说明伽马射线对岩性变化更敏感，需要更精细的特征工程。

我们还在验证集上绘制了预测值与真实值的散点图，发现低值区间拟合较好，高值区间存在轻微低估，这与随机森林对极端值的保守估计有关。

## 三、总结

本文中我们完成了 Well Log Reconstruction Challenge 的基线流程。我们首先用 lasio 解析 LAS 文件并统一列名，确认缺失标记为 -999.25。其次我们清洗了数据，剔除目标缺失的行并用中位数插补特征缺失。接着我们用随机森林多输出回归器训练四条目标曲线，保持树的深度和数量适中以防过拟合。最后我们在测试集上生成了符合格式的提交文件，并在内部验证中观察到 GR 预测相对最难。

这个比赛给我们的最大启发是：工业数据往往以特殊文件格式出现，数据读取和列名统一是第一步；多输出回归可以用单一模型统一学习相关曲线，避免分别建模带来的不一致。
