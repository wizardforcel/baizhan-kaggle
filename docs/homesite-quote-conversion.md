# Homesite Quote Conversion

## 一、比赛简介

### 1.1 比赛背景
Homesite 是房屋保险提供商，客户在网站上请求报价后是否会最终购买，决定了定价策略的有效性。比赛提供匿名化的客户、销售、房产与覆盖信息，要求预测 QuoteConversion_Flag。

数据约 26 万条，298 个特征，完全匿名，特征名以 Field、GeographicField 等命名。类别不平衡明显，正例比例低。做特征工程时几乎没有业务语义可用，只能依赖统计筛选。

### 1.2 比赛目标
给定训练集的匿名特征，预测测试集客户是否会购买报价，输出二分类概率。训练集列包括 QuoteNumber、Original_Quote_Date、QuoteConversion_Flag 以及 295 个匿名 Field。

### 1.3 评价标准
比赛以 ROC-AUC 为评价指标，衡量模型区分正负样本的能力，数值越高越好。相比准确率，AUC 对类别不平衡更友好，适合绝大多数样本都是未购买的情况。

---
## 二、解题过程

我们采用特征选择 + 序数编码 + LightGBM 的流水线，先降维再调参。

### 2.1 数据探索（EDA）

先观察数据规模与缺失。

```python
import pandas as pd
df = pd.read_csv(r'train.csv.zip')
print(df.shape)               # (260753, 299)
print(df['QuoteConversion_Flag'].value_counts(normalize=True))
```

正例比例约 10% 左右，严重不平衡。特征以类别为主，多数为字符型，存在少量数值型。我们查看每个特征的唯一值数，匿名字段中约 60% 的特征唯一值数超过 100，说明信息密度低。

```python
cols = [c for c in df.columns if c.startswith('Field')]
print(sum(df[col].nunique() > 100 for col in cols))
```

列间相关性分析显示绝大多数特征几乎独立，匿名编排导致组合规则难以推断。因此我们决定不做深层交叉特征，仅做单特征筛选与序数编码。

### 2.2 数据清洗

所有特征均视为类别。原始列包含时间与销售字段，可能引入泄漏，我们在 RFE 流程中剔除 Original_Quote_Date、SalesField8。为降低内存，读取后及时释放原始 DataFrame。

```python
df_RFE.drop(columns=['Original_Quote_Date','SalesField8'], axis=1, inplace=True)
del df
import gc; gc.collect()
```

使用 OrdinalEncoder 编码，unknown_value 设为 -99，避免测试集新类别报错。编码后整个数据转为整数，LightGBM 可直接训练。

```python
from sklearn.preprocessing import OrdinalEncoder
pipe = Pipeline([
    ('label_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-99)),
    ('model', gbm.LGBMClassifier())
])
```

不同特征的基数差异很大，序数编码保留了树模型学习阈值的能力，又避免 one-hot 导致的维度爆炸。

### 2.3 模型训练与超参数调优

我们先做特征选择，两种策略：

1. Mutual Information 选出 50 个特征，模型在验证集 ROC-AUC 约 0.93。
2. Recursive Feature Elimination 选出 50 个特征，模型更强。

互信息衡量一个特征与标签的依赖程度，值越大说明越有用。RFE 则反复训练模型、删掉最不重要的特征，直到剩下指定数量。

```python
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(X_filled, y, discrete_features=True)
selected = X_filled.columns[mi.argsort()[-50:]]
```

```python
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier
rfe = RFE(LGBMClassifier(n_estimators=100), n_features_to_select=50, step=10)
rfe.fit(X_filled, y)
selected = X_filled.columns[rfe.support_]
```

RFE 得到的特征组合在验证集上明显优于互信息，说明“联合贡献”比单变量依赖更接近真实预测需求。

```python
RFE_params = {
    'boosting_type': 'gbdt',
    'num_leaves': 105,
    'learning_rate': 0.14,
    'feature_fraction': 0.89,
    'bagging_fraction': 1,
    'min_child_samples': 65,
    'lambda_l2': 4.7
}
```

参数由 Optuna 调优得到。Optuna 会在试验中提前终止表现差的组合，因此可以用较大搜索空间。我们对两组特征分别训练 LightGBM，再做平均融合。融合能降低单一特征筛选带来的方差。

```python
y_predict_mutual = GBM1.predict_proba(X_val)[:,1]
y_predict_RFE = GBM2.predict_proba(X_val)[:,1]
y_avg = (y_predict_mutual + y_predict_RFE) / 2
```

### 2.4 模型测试与指标报告

在本地验证集上，RFE 模型 ROC-AUC 约 0.96，Mutual Information 模型约 0.93，融合后接近 0.96+。我们使用分层抽样划分，验证集分布与全量一致。

```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

预测概率在 0.01~0.9 之间分布，正例被集中推向高分。提交文件使用平均概率。

```python
submission = pd.DataFrame({
    'QuoteNumber': test['QuoteNumber'],
    'QuoteConversion_Flag': y_avg
})
submission.to_csv('submission.csv', index=False)
```

最终私测得分 0.9627。说明特征筛选与模型融合的组合在匿名数据上具有可迁移性。

---
## 三、总结

本文中我们完成了 Homesite Quote Conversion 的分类建模。我们首先通过 EDA 确认 26 万样本、298 特征、约 10% 正例的规模与严重不平衡。其次把所有匿名特征视为类别并做序数编码清洗，及时释放内存。接着用 Mutual Information 与 RFE 两套特征选择策略，结合 Optuna 调优的 LightGBM 训练模型并做平均融合。再用分层验证评估 ROC-AUC，最后在排行榜获得 0.9627。

该比赛表明在高度匿名数据上，特征选择比原始特征堆砌更重要，光谱编码配合树模型已足够强大。融合多个特征视角可有效降低方差。
