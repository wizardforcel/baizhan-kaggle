# Santander 产品推荐（Santander Product Recommendation）

## 一、比赛简介

### 1.1 比赛背景

Santander 银行希望根据客户历史信息，预测每个客户在下一个月会新购买哪些产品。数据来自西班牙零售银行的匿名客户表，每行记录一个客户在某个月的产品持有情况。特征列是客户基本信息、渠道、收入等，产品列是 24 个二值变量，分别表示该客户是否持有 “储蓄账户”“信用卡”“房贷”等产品。时间跨度从 2015 年 1 月到 2016 年 5 月，训练集包含约 1,300 万行，约 93 万个客户。

数据有两大特点。第一，同一个客户在不同月份会重复出现，列信息在不同月份是基本一致的，只有产品列会随时间变化。第二，测试集是 2016 年 6 月，任务是给定 2016 年 5 月的状态，预测 6 月会新增的产品。我们要做的是跨月的新增预测，而不是一次性的分类。

### 1.2 比赛目标

训练集 `train_ver2.csv` 包含客户特征和 24 个产品指示变量，时间列 `fecha_dato` 为每月一记，间隔约 28 天。测试集 `test_ver2.csv` 的 `fecha_dato` 全是 2016-06-28，我们需要为每个客户在 7 个产品槽位里填入最有可能新增的产品 ID，提交文件为 `ncodpers` 和 `products`，`products` 是用空格分隔的最多 7 个产品名。

实际比赛中只有少数客户会在一个月内新增产品，大多数客户的产品保持不变。按照材料中 2016 年 5 月的统计，931,453 名客户里只有 29,712 人有新增，比例约 3.19%。因此 MAP@7 的上限远低于 1，预测要兼顾召回与排序。

### 1.3 评价标准

比赛使用 **Mean Average Precision at 7（MAP@7）**。对每个客户取预测的前 7 个产品，计算该客户的 Average Precision，再对所有客户取均值。平均精度考虑了位置，排在前面的正确产品会获得更高分数。如果某个客户当月没有新增产品，该客户的精度记为 0。排序的重要性在于，前 7 个位置里越靠前的正确越好，这促使我们把最可能的产品排在最前面。

---

## 二、解题过程

我们先对数据做探索，了解产品分布、每月新增量和特征质量；接着做清洗和特征工程，把类别编码、收入缺失和时间信息规范化；然后用历史产品持有作为滞后特征，训练一个多分类 XGBoost，预测每个客户会新增哪个产品；最后按客户生成前 7 个产品并输出提交。

### 2.1 数据探索（EDA）

我们首先读取训练集，了解行数、列数和时间分布。

```python
import pandas as pd

train = pd.read_csv("../input/train_ver2.csv")
print(train.shape)   # (13647309, 48)
print(train["fecha_dato"].nunique())
print(train["ncodpers"].nunique())
print(train.columns[24:])
```

结果显示训练集 13,647,309 行，48 列，前 24 列是客户特征，后 24 列是产品。客户数量约 929,000，时间点从 2015-01-28 到 2016-05-28，共 17 个月。产品列是二值 0/1。

我们统计各产品的使用率和新增情况。

```python
prods = train.columns[24:].tolist()
usage = (train[prods] == 1).mean().sort_values(ascending=False)
print(usage.head(10))

# 以 2016-05-28 为例，看新增比例
may = train[train["fecha_dato"] == "2016-05-28"]
apr = train[train["fecha_dato"] == "2016-04-28"]
merged = pd.merge(apr[["ncodpers"] + prods], may[["ncodpers"] + prods],
                  on="ncodpers", suffixes=("_apr", "_may"))
adds = (merged[[c+"_may" for c in prods]].sum(axis=1) -
        merged[[c+"_apr" for c in prods]].sum(axis=1))
print("新增客户数:", (adds > 0).sum(), "总客户数:", len(adds))
```

产品使用率最高的是 `ind_cco_fin_ult1`（当前账户），约 84% 的单产品客户只持有该产品。另一个常见组合是当前账户 + “定存”“借记卡”。新增客户仅占总客户的 3% 左右，说明大多数客户的产品稳定。年龄分布呈双峰，35-55 岁和 20 多岁是主要群体；参与产品最多的客户往往收入较高、渠道为网银。

### 2.2 数据清洗

我们其次对特征做映射和缺失处理。类别列很多，用 `-99` 表示缺失。

```python
mapping_dict = {
    'ind_empleado' : {'-99':0, 'N':1, 'B':2, 'F':3, 'A':4, 'S':5},
    'sexo' : {'V':0, 'H':1, '-99':2},
    'segmento' : {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1,
                  '01 - TOP':2, '-99':3},
    # ... 其他列同理
}
cat_cols = list(mapping_dict.keys())
def getIndex(row, col):
    val = str(row[col]).strip()
    return mapping_dict[col].get(val, mapping_dict[col]['-99'])
```

收入列 `renta` 缺失较多，我们按省份均值填补。省份不在数据里时用整体未知均值。

```python
renta_dict = {
    'ALBACETE': 76895, 'ALICANTE': 60562, 'ALMERIA': 77815,
    # ... 省份到均值的映射
    'UNKNOWN': 103689
}
def getRent(row):
    rent = row['renta']
    if rent in ['NA', '']:
        prov = row['nomprov'] if row['nomprov'] not in ['NA',''] else 'UNKNOWN'
        rent = renta_dict.get(prov, renta_dict['UNKNOWN'])
    else:
        rent = float(rent)
    # 归一化到 0-1
    return round((rent - 0) / 1500000, 6)
```

年龄和客户资历做截断与归一化。

```python
def getAge(row):
    age = row['age']
    if age in ['NA', '']:
        age = 40
    age = float(age)
    age = max(18, min(100, age))
    return round((age - 18) / 82, 4)

def getCustSeniority(row):
    s = row['antiguedad']
    if s in ['NA', '']:
        s = 0
    s = float(s)
    s = max(0, min(256, s))
    return round(s / 256, 4)
```

时间信息转成整数。

```python
def getMonth(row): return int(row['fecha_dato'].split('-')[1])
def getjoinMonth(row):
    fa = row['fecha_alta']
    if fa in ['NA','']:
        return 6
    return int(fa.split('-')[1])
```

最后删除几乎全空列 `ult_fec_cli_1t` 与 `conyuemp`，其余缺失统一填充为 0 或均值。

### 2.3 模型训练与超参数调优

比赛的预测目标是“新增产品”。我们先把每月的产品状态和前一月状态对齐，得到新增标签。

```python
import numpy as np
import xgboost as xgb
from operator import sub

target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
               'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
               'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
               'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
               'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
               'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1',
               'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
target_cols = target_cols[2:]  # 使用 22 个产品
```

我们只在 4 月/5 月和 2016 年 4 月/5 月构建样本。`cust_dict` 保存 5 月的状态，`lag_cust_dict` 保存 4 月的状态，用于生成新增标签 `new_products = max(current - prev, 0)`。特征包括编码后的类别、归一化年龄、收入、资历、月份、入网月份，以及上一月的产品持有和去年同期产品持有。

```python
param = {
    'objective':'multi:softprob',
    'eta':0.09,
    'max_depth':6,
    'num_class':22,
    'eval_metric':'mlogloss',
    'min_child_weight':12,
    'subsample':0.85,
    'colsample_bytree':0.9,
    'seed':845895
}
xgtrain = xgb.DMatrix(train_X, label=train_y)
model = xgb.train(list(param.items()), xgtrain, num_rounds=70)
```

我们把每个客户的每种新增产品当成一个独立样本，训练多分类模型。训练后对测试客户生成概率，取每个客户概率最高的 7 个产品作为预测。超参数相对保守，深度 6、学习率 0.09、采样 0.85，70 轮能避免过拟合，因为样本量很大但正样本稀疏。

为了评估，我们在 2015-06-28 验证。验证 MAP@7 的实现如下。

```python
def apk(actual, predicted, k=7, default=0.0):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return default
    return score / min(len(actual), k)

def mapk(actual, predicted, k=7):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
```

验证集上的 MAP@7 会随验证月份变化，材料里记录过 0.042663 和 0.036466 两个值，都明显高于“只推最热产品”的朴素基线。进一步的可优化点是使用多模型融合、为每个产品单独训练二分类器，或加入更长的序列信息。

### 2.4 模型测试与指标报告

最后用 2016-04-28 和 2016-05-28 构建训练集，预测 2016-06-28。

```python
model = xgb.train(list(param.items()), xgtrain, num_rounds=70)
preds = model.predict(xgtest)
test_preds = []
for i in range(len(preds)):
    prob = preds[i]
    top7 = np.argsort(prob)[-7:][::-1]
    test_preds.append([target_cols[idx] for idx in top7])
```

提交文件需要每行一个客户的 7 个产品，用空格拼接。

```python
out = pd.DataFrame({
    'ncodpers': test_ids,
    'products': [' '.join(p) for p in test_preds]
})
out.to_csv('submission.csv', index=False)
print(out.head())
```

指标上，本地验证 MAP@7 在 0.036 到 0.043 之间，取决于用哪个月做验证。材料里 2016 年 5 月的理论上限只有 0.0319，而当时排行榜最高分约 0.0303，说明这个指标整体分数很低，0.001 的提升都很难。原因是有新增产品的客户只占约 3%，其余客户的分数固定为 0，模型能争取的空间本来就小。

---

## 三、总结

本文中我们完成了 Santander 产品推荐的完整流程。我们首先通过 EDA 发现训练集有 13,647,309 行、929,000 个客户、24 个产品，新增比例仅约 3%，当前账户是最核心产品。其次我们对类别列做映射，用省份均值填补收入缺失，并归一化年龄、收入、资历等数值特征。接着我们构建滞后特征，用 4 月和 5 月的数据预测 6 月新增，多分类 XGBoost 在验证集上达到 MAP@7 约 0.036，明显优于最热产品基线。最后我们生成每个客户的前 7 个产品预测，提交格式符合要求。

这个比赛的核心不是模型多复杂，而是特征里的时间滞后和样本构造是否正确。新增标签稀疏会导致正负极不平衡，多分类框架配合概率排序，能在 MAP@7 上比较稳定地取得提升。同时，类别映射和时间对齐的细节，比调参更关键。