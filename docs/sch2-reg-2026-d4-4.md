# 最优肥料预测（Predicting Optimal Fertilizers）

## 一、比赛简介

### 1.1 比赛背景

农作物能不能长得好，离不开合适的肥料。不同作物、不同土壤条件对氮（Nitrogen）、磷（Phosphorous）、钾（Potassium）的需求不同；温度、湿度、土壤水分也会影响肥料的吸收效果。农户如果只凭经验选肥料，可能造成成本浪费，甚至因为施肥不当损害作物。这个比赛希望用数据分析来回答一个实际问题：给定土壤和天气条件，哪一种肥料最适合当前地块？

比赛名是 **Predicting Optimal Fertilizers**。训练数据中的每一行代表一次「环境条件—最佳肥料」配对，环境字段既有数值（氮、磷、钾、温度、湿度、水分等），也有类别（作物类型、土壤类型等）。目标列是 `Fertilizer Name`，里面记录了推荐的肥料名称。

这是一项多分类任务，难点在于：第一，很多肥料的名称相似，类别之间的边界可能很细；第二，环境变量之间存在交互关系，比如同样的氮含量，在不同湿度下可能需要不同肥料；第三，比赛不是只要一个答案，而是要求我们给出**前三个候选肥料**。这意味着模型的概率排序非常重要，即使第一名猜错，只要正确答案排进前三，仍然可能获得部分成绩。

### 1.2 比赛目标

我们需要完成以下工作：

- 从训练集读取环境特征和目标列 `Fertilizer Name`；
- 对测试集中的每一行环境条件，预测最可能的三种肥料；
- 将三个肥料名称按可能性从高到低用空格连接，填入提交文件的 `Fertilizer Name` 列；
- 保留测试集的 `id`，确保每条预测和原样本对应。

材料里的特征列包括 `Nitrogen`、`Phosphorous`、`Potassium`、`Temparature`、`Humidity`、`Moisture` 等数值变量，以及需要编码的类别变量。目标列 `Fertilizer Name` 不直接作为特征，而是先编码成整数，交给多分类模型训练。

### 1.3 评价标准

这类比赛通常使用 **MAP@3（Mean Average Precision at 3）** 或等价的 Top-3 评价方式。用大白话解释：模型对每一行输出三个候选答案，第一名猜中得分最高；如果正确答案在第二名或第三名，仍然能得到较低但有效的分数；如果完全不在前三名，则这一行得分为 0。

材料里还用 `top_k_accuracy_score(y_val, val_proba, k=3)` 对验证集进行检查。Top-3 Accuracy 的定义是：真实肥料是否出现在模型概率最高的三个类别里。它比普通 Accuracy 宽松，因为它允许模型保留两个备用答案；但它也更符合实际决策场景——农户可以从三个候选里结合成本和经验做最后选择。

因此，我们在训练时不能只调用 `predict()` 得到一个类别，而要用 `predict_proba()` 得到所有肥料类别的概率，然后对每一行排序，取概率最高的三类。排序顺序就是提交文件的顺序。

---

## 二、解题过程

我们采用的主线很清楚：**先检查训练集和测试集的规模及字段，再构造有农业含义的交互特征，接着把类别列和目标列编码，使用 XGBoost 的多分类概率模型训练，最后从概率矩阵中取每行 Top-3 肥料名称并生成提交。**

### 2.1 数据探索（EDA）

我们首先读取三个文件，确认数据是否成功加载：

```python
import pandas as pd
import numpy as np

# 训练集有目标列，测试集没有；提交样例告诉我们输出格式
train = pd.read_csv('/kaggle/input/sch2-reg-2026-d4-4/train.csv')
test  = pd.read_csv('/kaggle/input/sch2-reg-2026-d4-4/test.csv')
submission = pd.read_csv('/kaggle/input/sch2-reg-2026-d4-4/sample_submission.csv')

print("Train Shape:", train.shape)
print("Test Shape :", test.shape)
print(train.head())
print(test.head())
```

我们首先关注三个问题。

**第一，训练集和测试集的行数不同，但特征列应该相同。** 训练集额外多出 `Fertilizer Name`，测试集没有它。`id` 只是样本标识，不能让模型把它当作环境条件。

**第二，目标是多类别字符串。** 我们查看目标列的类别数量和样例：

```python
# 看推荐肥料一共有多少种，以及每种出现次数
print(train["Fertilizer Name"].nunique())
print(train["Fertilizer Name"].value_counts().head())

# 看数据类型，找出需要编码的字符串特征
print(train.dtypes)
```

如果某些肥料样本较少，就可能出现类别不均衡；如果类别数量较多，模型输出的概率矩阵也会比较宽。无论类别多少，后面的 `num_class` 都应该从编码器里自动读取，而不是手写一个容易错的数字。

**第三，查看数值特征之间可能的关系。** 材料里的第二份方案直接构造了 NPK 总量、温度与湿度的乘积等特征，说明原始列之间的组合关系比单列更有信息。我们用简单的描述统计确认各数值列的尺度：

```python
# 只看数值列的统计量，了解均值、最小值、最大值和离散程度
numeric_cols = [
    "Nitrogen", "Phosphorous", "Potassium",
    "Temparature", "Humidity", "Moisture"
]
print(train[numeric_cols].describe().T)
```

这一步让我们知道不同字段的量纲不一样：氮磷钾是养分含量，温度是温度单位，湿度和水分又是另一种尺度。对于树模型来说不需要标准化，但构造比例或乘积时需要注意分母为 0 的情况。

### 2.2 数据清洗

我们其次处理 `id`、类别特征和目标标签。

**第一步：不把 `id` 当成特征。** 我们让特征列只包含真正描述土壤和天气的信息：

```python
# 找出字符串类型的特征，但排除目标列和 id
cat_cols = [
    col for col in train.columns
    if train[col].dtype == 'object'
    and col not in ['id', 'Fertilizer Name']
]
print("Categorical features:", cat_cols)
```

**第二步：为类别特征建立统一编码。** 训练集和测试集可能出现不同的类别取值，所以我们把两边拼起来拟合 `LabelEncoder`，再分别转换：

```python
from sklearn.preprocessing import LabelEncoder

# 每个类别列单独建立编码器，避免不同列的类别互相混淆
feature_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    # 拼接 train/test，确保测试集可能出现的类别也在编码表里
    full_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full_data)

    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
    feature_encoders[col] = le
```

把字符串转为整数只是为了让模型能读取，并不表示类别之间真的有「大小关系」。XGBoost 的树会通过切分点组合这些整数，实际效果通常比线性模型直接把编码当连续变量更稳。

**第三步：把目标肥料名称编码成整数。** 多分类模型需要 `0, 1, 2, ...` 的标签，所以我们单独为目标列建一个编码器：

```python
# 目标编码器负责保存「数字 ↔ 肥料名称」的对应关系
target_le = LabelEncoder()
train["target_encoded"] = target_le.fit_transform(train["Fertilizer Name"])

# 打印类别表，确认编码器知道所有肥料名称
print("Target classes:", list(target_le.classes_))
```

这里必须保留 `target_le`。训练时模型只认识数字，提交时我们必须把 Top-3 的数字还原成肥料名称；没有编码器就无法完成这个逆转换。

**第四步：准备训练矩阵。** 我们从特征列里排除 `id`、原始目标和目标编码临时列：

```python
# 只保留可用于预测的环境特征
feature_cols = [
    c for c in train.columns
    if c not in ["id", "Fertilizer Name", "target_encoded"]
]

X = train[feature_cols]
y = train["target_encoded"]
X_test = test[feature_cols]

print("X shape:", X.shape)
print("X_test shape:", X_test.shape)
```

### 2.3 模型训练与超参数调优

我们接着构造交互特征。单独看氮、磷、钾只能知道某一种养分的含量，不能知道总体施肥水平；单独看温度和湿度也无法表达「热且潮湿」的环境。因此我们加入以下四个特征：

```python
# NPK 总量：把三种主要养分合成一个总体施肥水平
train["npk_sum"] = (
    train["Nitrogen"] + train["Phosphorous"] + train["Potassium"]
)
test["npk_sum"] = (
    test["Nitrogen"] + test["Phosphorous"] + test["Potassium"]
)

# 温度 × 湿度：表达热湿环境的联合影响
train["temp_x_humidity"] = train["Temparature"] * train["Humidity"]
test["temp_x_humidity"] = test["Temparature"] * test["Humidity"]

# 水分 × 温度：表达土壤水分和温度共同造成的环境状态
train["moisture_x_temp"] = train["Moisture"] * train["Temparature"]
test["moisture_x_temp"] = test["Moisture"] * test["Temparature"]

# 氮磷比例：分母加 1，避免 Phosphorous 为 0 时除零
train["N_P_ratio"] = train["Nitrogen"] / (train["Phosphorous"] + 1)
test["N_P_ratio"] = test["Nitrogen"] / (test["Phosphorous"] + 1)
```

这些特征不是随便相乘，而是对应具体含义：`npk_sum` 表示养分总量，`temp_x_humidity` 表示温湿组合，`moisture_x_temp` 表示土壤水分与热量的相互作用，`N_P_ratio` 表示氮磷结构。材料里的方案先做这些特征工程，再做类别编码，说明作者认为它们能帮助模型识别肥料之间的差别。

**模型选择。** 我们使用 XGBoost 的 `multi:softprob` 模式。`multi:softprob` 和普通的 `multi:softmax` 不同：前者输出每一个类别的概率，后者只输出概率最高的类别。因为比赛要求 Top-3，必须选前者：

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    objective="multi:softprob",  # 输出所有类别概率，才能取 Top-3
    num_class=len(target_le.classes_),
    n_estimators=800,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",          # 直方图算法，训练速度更快
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)
```

这里的超参数体现了「多棵弱树逐步累加」的思路：学习率设置为 0.02，单棵树影响较小；树数量设为 800，给模型足够多的修正机会；深度设为 6，可以学习温度、湿度、养分之间的交互，但不会无限加深。`subsample` 和 `colsample_bytree` 都是 0.8，增加随机性，降低过拟合。

材料里还有一份方案使用 `n_estimators=1000`、`learning_rate=0.05`，并直接在全量数据上训练。这两组参数可以看作不同的调优尝试：前者学习更慢、轮数更多，后者收敛更快。我们选择带验证集的方案来检查 Top-3 效果：

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# 分层抽样，确保验证集里的每种肥料比例和训练集大致一致
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=50000, random_state=42, stratify=y
)

model.fit(x_train, y_train)

# 同时查看硬标签准确率和 Top-3 准确率
val_pred = model.predict(x_val)
val_proba = model.predict_proba(x_val)

print("Accuracy:", accuracy_score(y_val, val_pred))
print("Top-3 Accuracy:", top_k_accuracy_score(y_val, val_proba, k=3))
```

这里 `test_size=50000` 是材料中采用的一个较小验证集设定，目的在于让 Kaggle kernel 更容易运行。我们不把验证集用于最终提交，而是只用它比较参数和检查模型是否能够把正确肥料排进前三。

**关于调参。** 我们调的重点不是盲目增加树的深度，而是观察验证集的 Top-3 Accuracy。因为比赛只关心前三名，参数应该围绕概率排序进行选择。我们试验了两条路线：一条是加入 `npk_sum`、温湿乘积和比例特征后使用 800 棵树、学习率 0.02；另一条是更简单的编码加单模型方案，使用 1000 棵树、学习率 0.05。材料中还提到，排行榜靠前的方法往往是「类别编码 + 单个 XGBoost + 直接输出 Top-3」，这说明一条短而稳定的流水线可能胜过复杂的多模型堆叠。

验证完成后，我们用全部训练数据重新拟合：

```python
# 验证参数确认后，重新用全量数据训练，尽量利用每一行样本
model.fit(X, y)

# 对测试集输出每个肥料类别的概率
probs = model.predict_proba(X_test)
print("Probability matrix shape:", probs.shape)
```

### 2.4 模型测试与指标报告

我们首先在验证集上报告两种指标：普通 Accuracy 和 Top-3 Accuracy。普通 Accuracy 只看第一名是否正确，Top-3 Accuracy 看正确答案是否进入前三：

```python
from sklearn.metrics import classification_report, confusion_matrix

# 第一名预测的准确率
print("Accuracy:", accuracy_score(y_val, val_pred))

# 前三名覆盖真实答案的比例
print("Top-3 Accuracy:", top_k_accuracy_score(y_val, val_proba, k=3))

# 查看每种肥料的分类报告，排查某些类别是否特别难
print(classification_report(
    y_val,
    val_pred,
    target_names=target_le.classes_
))
```

如果 Top-3 Accuracy 明显高于普通 Accuracy，说明模型虽然有时第一名会错，但把正确肥料放进了候选前三。这对比赛评分是好事，也更符合「给农户一份候选清单」的实际用途。

然后我们从测试集概率矩阵中取每一行最大的三个索引。`argsort` 默认从小到大排序，所以先取最后三个，再用 `[:, ::-1]` 反转成从高到低：

```python
# 每行取概率最大的三个类别索引，并按概率从高到低排列
# argsort 是升序，因此切最后三列后要反转
top3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

# 把数字索引还原成肥料名称
top3_classes = target_le.inverse_transform(top3_indices.flatten())
top3_classes = top3_classes.reshape(top3_indices.shape)

# 每一行的三个肥料名称用空格连接，符合提交格式
pred_strings = [" ".join(row) for row in top3_classes]
```

这里不能直接调用 `model.predict()`，因为它只返回一个类别，无法产生第二、第三候选；必须保留完整概率矩阵，才能做排序。

最后我们把预测写入提交样例：

```python
# 保留测试集 id，保证预测和样本一一对应
submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": pred_strings
})

# 保存提交文件
submission.to_csv("submission.csv", index=False)
print(submission.head())
```

我们还可以用概率变换做一个小实验：材料中的第二份方案把概率提升到 `1.2` 次方后再排序。这种变换会让高概率更加突出，但因为每行的顺序通常不会发生太大变化，收益需要用验证集确认，不能凭感觉直接使用：

```python
# 可选实验：放大高概率类别的差异，排序前先做幂变换
sharpened_probs = probs ** 1.2
sharpened_top3 = np.argsort(sharpened_probs, axis=1)[:, -3:][:, ::-1]

# 只有当验证集 Top-3 变好时，才考虑采用这个版本
print("原始与放大概率的形状是否一致：", probs.shape == sharpened_probs.shape)
```

提交前我们检查了三件事：第一，提交行数等于测试集行数；第二，每一行正好有三个以空格分隔的肥料名称；第三，三个名称都来自训练集编码器的类别表，没有出现数字索引或拼写错误。完成这些检查后，文件才具备上传条件。

---

## 三、总结

本文中我们完成了最优肥料预测比赛的完整流程。我们首先通过 EDA 检查训练集、测试集和提交样例，确认目标列是 `Fertilizer Name`，并观察了氮、磷、钾、温度、湿度、水分等字段的类型和尺度。其次我们不把 `id` 当作特征，把训练集和测试集的类别列拼接后统一使用 `LabelEncoder`，再用单独的目标编码器把肥料名称转换成整数。接着我们构造了 `npk_sum`、`temp_x_humidity`、`moisture_x_temp`、`N_P_ratio` 四个有农业含义的交互特征，使用 XGBoost 的 `multi:softprob` 训练多分类模型，并通过分层验证集比较 Accuracy 和 Top-3 Accuracy。最后我们用 `predict_proba` 得到完整概率矩阵，逐行取概率最高的三个类别，反向还原为肥料名称并写入 `submission.csv`。

这个比赛最重要的启发是：**评价指标决定输出形式**。如果只看普通准确率，输出一个最可能类别就够了；但 MAP@3/Top-3 允许三个候选，我们就必须保留完整概率，用排序产生候选列表。第二个启发是特征工程要有领域含义：NPK 总量和氮磷比例描述养分结构，温度乘湿度描述环境组合，它们比机械地堆叠很多无意义列更容易帮助模型。最后，材料中的结果也说明，在表格多分类任务上，类别编码加一个经过合理设置的 XGBoost，往往已经是强而稳定的基线，不一定需要复杂的模型集成。
