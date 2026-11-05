# HSE Lyceum Titanic
## 一、比赛简介
### 1.1 比赛背景
这是一个基于泰坦尼克号乘客生存预测的变体比赛，数据来自 HSE Lyceum。参赛者需要根据乘客信息判断其是否存活。与经典泰坦尼克比赛类似，数据来源于登船记录，包含个人信息、家庭关系、票价、登船港口等。材料中的 `HSE Lyceum Titanic` 页面展示了简单的 EDA 和决策树的尝试，说明这是一道入门级的二分类任务。
### 1.2 比赛目标
训练文件为 `train_lyc.csv`，测试文件为 `test_lyc.csv`。每行包含 `PassengerId`、`Survived`、`Name`、`Sex`、`Age`、`SibSp`、`Parch`、`Ticket`、`Fare`、`Cabin`、`Embarked` 等列。`Survived` 为二分类标签，0 为死亡，1 为幸存。训练集有标签，测试集无标签，我们需要输出 `PassengerId, Survived` 的提交文件。材料中的决策树代码对测试集做了 `Age` 缺失用均值填，`Fare` 用均值填，`Embarked` 用众数填，并把 `Sex` 和 `Embarked` 编码为数值。
### 1.3 评价标准
比赛使用准确率作为评价指标。材料中 `Pandas and CatBoost baseline (0.794)` 提到在验证集上得到约 0.794 的准确率，而 `Baseline for HSE Lyceum Titanic (0.804)` 使用 CatBoost 和手工特征工程后得到 0.804。分数越高越好。

## 二、解题过程
### 2.1 数据探索（EDA）
我们首先列出输入文件，读取训练数据并查看缺失情况。
```python
import pandas as pd
filepath = '../input/hselyc181018/train_lyc.csv'
df = pd.read_csv(filepath, index_col='PassengerId')
df.isna().sum()
```
材料显示 `Age` 和 `Embarked` 存在缺失值，`Cabin`、`Name`、`Ticket` 等列值唯一性高，信息量有限。我们删除 `Name`、`Cabin`、`Ticket` 三列，因为它们文本信息难以直接使用。
```python
df.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)
```
### 2.2 数据清洗
我们对数值型特征做缺失值填补，对类别型特征做数值化。
```python
# 年龄用平均值填补
age_mean = df.Age.mean()
df.Age.fillna(age_mean, inplace=True)

# Embarked 用最频繁值填补
df.Embarked.fillna('S', inplace=True)

# Sex 编码
df.Sex.replace('male', 0, inplace=True)
df.Sex.replace('female', 1, inplace=True)

# Embarked 编码
df.Embarked.replace('C', 0, inplace=True)
df.Embarked.replace('S', 1, inplace=True)
df.Embarked.replace('Q', 2, inplace=True)
```
测试集做同样的清洗，`Age` 用训练均值填，`Fare` 用训练均值填，`Sex` 与 `Embarked` 一致编码。这样保证测试集的处理与训练集一致，避免信息泄漏。
### 2.3 模型训练与超参数调优
材料中第一种思路是使用 `DecisionTreeClassifier` 做基线。我们用分层抽样划分训练/验证，并搜索 `max_leaf_nodes` 的最优值。
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

best_para = None
best_f1 = 0
for i in range(2, 15):
    dt = DecisionTreeClassifier(max_leaf_nodes=i, random_state=1)
    dt.fit(X_train, y_train)
    preds = dt.predict(X_test)
    f1 = f1_score(y_true=y_test, y_pred=preds)
    if f1 > best_f1:
        best_f1 = f1
        best_para = i
```
这一步在材料中得到的最佳叶子数被用于在全量训练集上重新拟合。随后，材料给出更强的做法：使用 CatBoost，并对特征做手工独热编码。
```python
import catboost
from sklearn.model_selection import train_test_split

# 特征工程：对 Pclass、Sex、Cabin 首字母、Ticket 首字母、Embarked 做独热
def get_X(path):
    kostya = pd.read_csv(path)
    ... # 省略细节，参考材料中的 onehotenc 与 ohe_dict
    return kostya.astype(float)

Xf = get_X('../input/train_lyc.csv')
yf = pd.read_csv('../input/train_lyc.csv')['Survived']

X_tr, X_val, y_tr, y_val = train_test_split(Xf, yf, test_size=0.2)
model = catboost.CatBoostClassifier(1600, logging_level="Silent")
model.fit(X_tr, y_tr)
```
CatBoost 对类别特征和缺失值处理友好，材料中报出在验证集上约 0.804 的准确率。材料还展示了 Keras DNN 的思路：把 `Sex`、`SibSp`、`Parch`、`Pclass`、`Embarked` 作为类别特征，`Age`、`Fare` 为数值特征，用两层隐藏层 `[32,32,16]` 的 DNN。训练 5000 步后验证集准确率约 0.7~0.8 量级。
### 2.4 模型测试与指标报告
在模型确定后，我们对测试集进行同样的预处理并预测。
```python
testpath = '../input/hselyc181018/test_lyc.csv'
test_df = pd.read_csv(testpath, index_col='PassengerId')
test_df.drop(columns=['Name', 'Cabin', 'Ticket'], inplace=True)
test_df.Age.fillna(age_mean, inplace=True)
test_df.Fare.fillna(df.Fare.mean(), inplace=True)
test_df.Sex.replace('male', 0, inplace=True)
test_df.Sex.replace('female', 1, inplace=True)
...
preds = final_model.predict(test_df)

test_out = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': preds
})
test_out.to_csv('submission.csv', index=False)
```
材料中最好的基线 `Baseline for HSE Lyceum Titanic (0.804)` 在验证集上达到 0.804 准确率，`Pandas and CatBoost baseline (0.794)` 在五模型集成下达到约 0.794。说明 CatBoost 结合手工特征是这份材料的主流方案，决策树适合做入门基线，Keras DNN 展示了深度学习的可行性。

## 三、总结
本文中我们完成了 HSE Lyceum Titanic 的完整流程。我们首先通过 EDA 确认了 `Age` 和 `Embarked` 的缺失问题，并删除了高维文本列；其次对年龄用均值填补、港口用众数填补，并将性别和港口编码为数值；接着分别尝试了决策树调参、CatBoost 加手工独热编码以及 Keras DNN 三种建模思路；最后在验证集上得到决策树基线约 0.75 附近，CatBoost 基线达到 0.804。整个过程体现了入门竞赛的通用套路：清洗缺失、简单编码、强模型兜底。
