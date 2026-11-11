# 健康结局预测的数据预处理（AI MATCH Workshop）

## 一、比赛简介

### 1.1 比赛背景

AI MATCH 是一个面向医学与人工智能交叉领域的工作坊（workshop）。这个比赛的任务是**根据患者的一组临床特征，预测治疗后的健康结局**。表格里的每一行是一个病例，列是各种检查、化验、体征和基本信息。

医学表格数据有几个很典型的特点，也是这个工作坊想让大家练习的地方：

- **数值和类别混在一起。** 有的列是年龄、指标数值这样的连续量，有的列是「有 / 无」「某种分型」这样的类别。
- **有缺失值。** 临床上漏做某项检查很常见，不是数据录入错误。
- **类别列的取值可能在新数据里没见过。** 训练集里只有三种分型，测试集突然冒出第四种，如果编码方式没处理好，程序会直接报错。

所以这份材料并不是一份「冲榜方案」，而是一份**数据预处理的教学示例**：它把「数值列怎么填、类别列怎么填、两类列怎么拼成一张模型能吃的矩阵」讲得很细。材料里没有训练任何模型，也没有报告分数——这一点我们在后面会明确说明，不替它编造结果。

### 1.2 比赛目标

数据有两个文件：

- `Train.csv`：第一列是样本编号，最后一列是结局标签，中间是特征；
- `Test.csv`：和训练集同样的列，但没有最后一列标签。

标签列里出现的是两种字符串：`"CR/CRI"` 和 `"NR"`。从命名看，这是临床疗效的常用分级——`CR` 大致对应「完全缓解」，`CRI` 是「完全缓解但某些指标未完全恢复」，`NR` 是「无缓解」。数据提供方已经把前两者合并成了一个取值 `"CR/CRI"`，这份示例再把它记作正类 1，把 `"NR"` 记作 0，于是问题就是**二分类**：这个病人有没有获得缓解。

我们要给测试集每一行预测一个 0 / 1 标签（或者缓解的概率）。

### 1.3 评价标准

材料里**没有给出评价指标**，也没有任何分数。这是一个教学用的预处理示例，作者在把特征矩阵拼好之后就结束了。

从任务形式看，这是一个二分类问题，常见的指标会是 ROC-AUC 或准确率；但既然材料没写，我们就不在这里断言。我们在下一节只复现它做的事情，并在最后说明「如果要继续往下走，应该怎么接模型、怎么评估」。

---

## 二、解题过程

我们首先把训练集和测试集读进来，确认列的结构；其次把特征和标签分开，并把三档疗效合并成 0/1；接着按数据类型把列分成数值列和类别列；然后分别给两类列设计填充和编码流水线；最后用 `ColumnTransformer` 把两条流水线并成一条，得到一张可以直接喂给模型的矩阵。

### 2.1 数据探索（EDA）

我们首先读数据，直接看整张表。这里用 `display()` 而不是 `print()`，是因为在 notebook 里它会渲染成带滚动的表格，列多的时候看起来更方便：

```python
import numpy as np
import pandas as pd

# 训练集和测试集都用 pandas 读进来
train_df = pd.read_csv(
    '/kaggle/input/aimatch-workshop-predicting-health-outcomes/Train.csv')
test_df = pd.read_csv(
    '/kaggle/input/aimatch-workshop-predicting-health-outcomes/Test.csv')

display(train_df)
display(test_df)
```

看完表格，我们关心的第一件事是：**第一列和最后一列到底是什么**。第一列是编号，它唯一标识一个样本，但对预测没有意义；最后一列是标签。于是我们把它们切出去：

```python
# iloc[:, 1:-1] 的意思是：所有行，从第 1 列到倒数第 2 列（左闭右开）
# 这样正好跳过第一列的编号和最后一列的标签
X_train = train_df.iloc[:, 1:-1]
y_train = train_df.iloc[:, -1]

# 三档疗效合并成二分类：CR / CRI 都算"缓解"，NR 算"未缓解"
# 注意要先 replace 再 astype(int)，否则对象类型没法直接转整数
y_train = y_train.replace({'CR/CRI': 1, 'NR': 0}).astype(int)

# 测试集没有标签，整张表都是特征（编号列后面再处理）
X_test = test_df

display(X_train)
display(y_train)
```

这里 `replace` 的写法说明训练集里的标签其实是**已经合并过的字符串** `"CR/CRI"`，不是三个独立值。也就是说，数据提供方在导出时就把 CR 和 CRI 并成了一档。这个细节很重要：如果我们按「三分类」去建模，就会和主办方的定义对不上。

接着我们逐个检查列的数据类型。这一步的作用是**自动**把列分成数值组和类别组，而不是手工维护两张名单——手工名单在列很多的时候特别容易漏：

```python
def id_str_and_cat_cols(df):
    """按 pandas 的 dtype 把列分成数值列和类别列。"""
    num_cols, cat_cols = [], []
    for col_name in df.columns:
        print(col_name, df[col_name].dtype)
        print('-' * 20)

        # int64 / float64 当数值处理；
        # object（字符串）一律当类别处理
        if df[col_name].dtype in ['int64', 'float64']:
            num_cols.append(df[col_name].name)
        else:
            cat_cols.append(df[col_name].name)
    return num_cols, cat_cols

numeric_features, categorical_features = id_str_and_cat_cols(X_train)
print('数值列:', len(numeric_features))
print('类别列:', len(categorical_features))
```

这个函数有一个值得注意的假设：**它把 `int64` 一律当数值**。如果某一列其实是「编号型」的整数（比如科室代码 1、2、3），把它当连续数值送进模型是不合适的。这份材料的数据里没有出现这种情况，但换成别的数据集时，这个函数需要人工复核一遍。

### 2.2 数据清洗

这个工作坊把「清洗」拆成了两条流水线，分别对应两类列。先说数值列：**用中位数填缺失，再做标准化**。

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_transformer = Pipeline(steps=[
    # 中位数不受极端值影响，比均值更适合医学指标（偶尔有离谱的高值）
    ('imputer', SimpleImputer(strategy='median')),
    # 标准化：减均值、除标准差，让每个特征量纲一致
    ('scaler', StandardScaler()),
])

Xt_num = numeric_transformer.fit_transform(X_train[numeric_features])
print(Xt_num.shape)      # (样本数, 数值列数)

# 转回 DataFrame 只是为了肉眼检查，真正建模时 numpy 数组就够
Xt_num_df = pd.DataFrame(Xt_num, columns=numeric_features)
display(Xt_num_df.head())
```

为什么数值列要标准化？因为不同指标的尺度差得很远：有的在 0 到 1 之间，有的在几百上千。逻辑回归、神经网络、K 近邻这些模型对尺度敏感，不标准化的话，数值大的列会主导距离和梯度。树模型倒是不在乎，但统一做一遍没有坏处。

再说类别列：**用众数填缺失，再做独热编码**。

```python
from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline(steps=[
    # 类别列用出现最多的取值填，因为没法算"中位数"
    ('imputer', SimpleImputer(strategy='most_frequent')),
    # handle_unknown='ignore'：测试集出现训练集没见过的类别时，
    # 不报错，而是编码成全 0。这一条是防止线上崩掉的关键
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

Xt_cat = categorical_transformer.fit_transform(X_train[categorical_features])
print(Xt_cat.shape)

# 把编码后的列名取出来，方便知道每一列代表哪个类别
ohe = categorical_transformer.named_steps['onehot']
feature_names = ohe.get_feature_names_out(categorical_features)
Xt_cat_df = pd.DataFrame(Xt_cat.toarray(), columns=feature_names)
display(Xt_cat_df.head())
```

`handle_unknown='ignore'` 是这个工作坊最实用的一个细节。默认情况下，`OneHotEncoder` 遇到没见过的类别会抛错；而在真实场景里，测试集出现新类别几乎是必然的（新药、新分型、新科室）。设成 `ignore` 之后，这些样本的对应列全是 0，模型仍然能给出预测，只是少了这一项信息。这比让整个预测流程崩掉要好得多。

还有一个容易忽略的点：`OneHotEncoder` 输出的是**稀疏矩阵**（大部分位置是 0）。类别一多，矩阵会非常宽，用 `toarray()` 转成稠密矩阵可能吃掉大量内存。这份示例转成 DataFrame 只是为了展示，真正训练时应该直接把稀疏矩阵交给模型。

### 2.3 模型训练与超参数调优

单个流水线只能处理一种类型的列，所以最后一步是用 `ColumnTransformer` 把两条流水线**并排**接起来：数值列走数值那条，类别列走类别那条，各自处理完再横向拼在一起。

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    # 第一个元素是这条流水线的名字，第二个是流水线对象，
    # 第三个是"这条流水线处理哪些列"
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

# 一次性拟合 + 转换，得到最终的特征矩阵
Xt_all = preprocessor.fit_transform(X_train)

# 给拼好的矩阵配上列名：数值列名 + 独热编码后的列名
Xt_all_cols = pd.DataFrame(Xt_all.toarray() if hasattr(Xt_all, 'toarray')
                           else Xt_all)
Xt_all_cols.columns = np.append(Xt_num_df.columns.to_numpy(),
                                Xt_cat_df.columns.to_numpy())
display(Xt_all_cols.head())
print('最终特征矩阵:', Xt_all_cols.shape)
```

把预处理封装成 `ColumnTransformer` 有三个好处，都值得记住：

1. **不会漏列。** 只要一开始把列分好组，后面新增列会自动进入对应的流水线。
2. **不会泄漏。** `fit_transform` 只在训练集上调用；到测试集时只调用 `transform`，填充用的中位数、标准化用的均值方差都来自训练集。
3. **可以整体进交叉验证。** 把 `preprocessor` 和一个分类器串成 `Pipeline`，交叉验证的每一折都会重新拟合预处理，不会出现「用全量数据算中位数」这种泄漏。

材料里已经 `import` 了 `LogisticRegression`，但**代码到拼好特征矩阵就结束了**，没有 `fit` 任何模型。所以我们不报告任何模型分数。如果要继续往下走，最自然的接法是：

```python
# 下面这段是根据材料里的 import 和预处理结构整理的思路，
# 不是原材料里跑过的代码
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

model = Pipeline(steps=[
    ('preprocess', preprocessor),          # 预处理也放进流水线
    ('clf', LogisticRegression(max_iter=1000, random_state=42)),
])

# 分层 5 折：每一折里正负样本比例和整体一致，
# 小样本医学数据尤其需要分层，否则某一折可能几乎没有正例
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train,
                         cv=cv, scoring='roc_auc')
print('5 折 AUC:', scores.mean(), '±', scores.std())
```

这里把 `preprocessor` 放进 `Pipeline` 是关键。如果先对全量 `X_train` 做 `fit_transform`、再拿处理后的矩阵去做交叉验证，每一折的验证集其实已经参与了填充和标准化的拟合，分数会偏高。放进流水线之后，每一折都会在「只看这一折训练部分」的前提下重新算中位数和均值，评估才是诚实的。

### 2.4 模型测试与指标报告

材料没有模型，自然也没有测试和指标。我们不在这里编一个分数，只把「如果按这份预处理继续做，测试阶段要注意什么」讲清楚：

```python
# 思路整理：训练完成后，对测试集只做 transform，绝不能再 fit
X_test_features = X_test.iloc[:, 1:]       # 去掉第一列的编号
Xt_test = model.named_steps['preprocess'].transform(X_test_features)

# 输出概率而不是硬标签：后续想调阈值、算 AUC 都方便
proba = model.predict_proba(Xt_test)[:, 1]

submission = pd.DataFrame({
    'id': X_test.iloc[:, 0],               # 用测试集第一列的编号
    'outcome': proba,
})
submission.to_csv('submission.csv', index=False)
print(submission.shape)
```

这段里有两条纪律。第一，**测试集只能 `transform`，不能 `fit_transform`**。一旦在测试集上重新拟合，填充值和标准化参数就变了，训练和推理的坐标系不一致。第二，**提交前要核对编号列**。这份数据的第一列是编号，如果我们把它当特征喂进模型，模型会学到「编号越大越容易缓解」这种毫无意义的规律；反过来，如果提交时忘了带上编号，行就对不上。

我们还会做一次格式自检：提交文件的行数必须等于测试集行数，编号不能重复、不能缺失，概率必须在 0 到 1 之间。这些检查很琐碎，但比赛里因为格式问题白交一次的情况并不少见。

---

## 三、总结

本文中我们完成了 AI MATCH 健康结局预测比赛的预处理流程复现。我们首先读入训练集和测试集，把第一列的编号和最后一列的标签切出去；其次把 `CR/CRI` 合并为正类、`NR` 记为负类，把三档疗效简化成二分类；接着按 pandas 的 dtype 自动把列分成数值组和类别组；然后用「中位数填充 + 标准化」处理数值列，用「众数填充 + 独热编码（忽略未知类别）」处理类别列；最后用 `ColumnTransformer` 把两条流水线拼成一张完整的特征矩阵，并说明了继续接逻辑回归、做分层交叉验证和生成提交文件时要注意的细节。

这份材料的价值不在于分数，而在于它把**表格预处理的标准动作**演示得很完整。我们从中得到三点启发：第一，数值列和类别列要用不同的填充策略，不能一把梭；第二，`handle_unknown='ignore'` 是上线安全的必需品，不是可选项；第三，预处理必须和模型一起放进交叉验证的流水线里，否则「用全量数据算中位数」这种泄漏会让评估结果虚高。
