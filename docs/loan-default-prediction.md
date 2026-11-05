# Loan Default Prediction

## 一、比赛简介

### 1.1 比赛背景
Loan Default Prediction 比赛使用银行贷款申请的匿名特征数据，目标是判断借款人是否会违约。训练表 train_v2.csv.zip 包含 105471 条样本、771 列，其中 f1~f778 为申请特征，loss 为违约金额。测试表 test_v2.csv.zip 结构相同但无标签。任务属于典型的二分类信用风险建模，特征多且包含数值与类别混合。

### 1.2 比赛目标
将连续的 loss 值转化为二分类标签 Classifier：loss>0 标记为违约 1，否则为 0。模型需学习特征到违约概率的映射，并在测试集上输出二分类预测。材料中常见的做法是直接使用 loss 作为回归目标或先二值化再分类。

### 1.3 评价标准
材料中通过 5 折交叉验证的准确率评估模型。报告给出 SVM 85.62%、Random Forest 86.50%、Naive Bayes 78.12%、Logistic Regression 87%。准确率越高越好，模型选择以交叉验证均值为准。

## 二、解题过程

### 2.1 数据探索（EDA）
读取训练数据，确认规模与列。

```python
import pandas as pd
data = pd.read_csv('../input/loan-default-prediction/train_v2.csv.zip', low_memory=False)
print(data.shape)
```

输出 (105471, 771)。查看 loss 分布。

```python
data['loss'].plot(kind='density')
```

loss 大部分为 0，少数正值，呈现强失衡。

```python
data['loss'].value_counts().head()
```

我们将 loss 二值化：

```python
data['Classifier'] = [0 if x==0 else 1 for x in data['loss']]
```

### 2.2 数据清洗
训练集删除目标列后作为特征，使用 SelectKBest 进行特征筛选。

```python
from sklearn.feature_selection import SelectKBest, chi2
features = train.drop(columns=['Classifier','loss'])
X = features.values
y = train['Classifier'].values
```

部分列存在混合类型，读取时使用 low_memory=False。材料中未做缺失值填充，默认使用现有数值。

### 2.3 模型训练与超参数调优
使用 5 折交叉验证比较多个基线。

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def crossvalidation(X_train, y_train, model):
    accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=5)
    return accuracies.mean()
```

模型示例：

```python
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

svm = SVC()
rf = RandomForestClassifier(n_estimators=10, criterion='entropy')
nb = GaussianNB()
lr = LogisticRegression()
```

材料报告 SVM 85.62%、Random Forest 86.50%、Naive Bayes 78.12%、Logistic Regression 87% 的交叉验证准确率。Logistic Regression 在此基线上表现最佳。

特征降维实验也在材料中出现，使用 PCA 对 771 维特征降维后训练模型。

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)
```

### 2.4 模型测试与指标报告
在选定模型上拟合全量训练集并预测测试集。

```python
testdata = pd.read_csv('../input/loan-default-prediction/test_v2.csv.zip')
X_test = testdata.drop(columns=['id'])
model.fit(X, y)
pred = model.predict(X_test)
```

提交文件需包含 id 与预测标签。材料中通过 cross_val_score 手动计算性能，避免使用未定义的变量。

## 三、总结
本文中我们完成了 Loan Default Prediction 的建模流程。我们首先读取 10 万级、771 维的贷款数据并将 loss 二值化为违约标签，观察到目标严重失衡。其次采用 SelectKBest 与 PCA 进行特征筛选，通过 5 折交叉验证比较 SVM、Random Forest、Naive Bayes 与 Logistic Regression。最后以 Logistic Regression 的 87% 交叉验证准确率作为基线，并生成测试集预测提交。材料显示简单的线性模型在该任务上具有竞争力，特征工程与失衡处理是进一步提升的关键。
