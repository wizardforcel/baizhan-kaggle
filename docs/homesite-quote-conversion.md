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

正例比例约 10% 左右，严重不平衡。特征以类别为主，多数为字符型，存在少量数值型。

### 2.2 数据清洗

所有特征均视为类别。原始列包含时间与销售字段，可能引入泄漏，我们在 RFE 流程中剔除 Original_Quote_Date、SalesField8。

```python
df_RFE.drop(columns=['Original_Quote_Date','SalesField8'], axis=1, inplace=True)
```

使用 OrdinalEncoder 编码，unknown_value 设为 -99，避免测试集新类别报错。

```python
from sklearn.preprocessing import OrdinalEncoder
pipe = Pipeline([
    ('label_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-99)),
    ('model', gbm.LGBMClassifier())
])
```

### 2.3 模型训练与超参数调优

我们先做特征选择，两种策略：

1. Mutual Information 选出 50 个特征，模型在验证集 ROC-AUC 约 0.93。
2. Recursive Feature Elimination 选出 50 个特征，模型更强。

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

参数由 Optuna 调优得到。我们对两组特征分别训练 LightGBM，再做平均融合。

```python
y_predict_mutual = GBM1.predict_proba(X_val)[:,1]
y_predict_RFE = GBM2.predict_proba(X_val)[:,1]
y_avg = (y_predict_mutual + y_predict_RFE) / 2
```

### 2.4 模型测试与指标报告

在本地验证集上，RFE 模型 ROC-AUC 约 0.96，Mutual Information 模型约 0.93，融合后接近 0.96+。

提交文件使用平均概率。

```python
submission = pd.DataFrame({
    'QuoteNumber': test['QuoteNumber'],
    'QuoteConversion_Flag': y_avg
})
submission.to_csv('submission.csv', index=False)
```

最终私测得分 0.9627。

---
## 三、总结

本文中我们完成了 Homesite Quote Conversion 的分类建模。我们首先通过 EDA 确认数据规模与类别不平衡。其次把所有匿名特征视为类别并做序数编码清洗。接着用 Mutual Information 与 RFE 两套特征选择策略，结合 Optuna 调优的 LightGBM 训练模型并做平均融合。最后在验证集上达到 ROC-AUC 约 0.96，并在排行榜获得 0.9627。

该比赛表明在高度匿名数据上，特征选择比原始特征堆砌更重要，光谱编码配合树模型已足够强大。
