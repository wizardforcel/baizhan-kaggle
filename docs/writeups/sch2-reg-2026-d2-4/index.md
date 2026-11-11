# 卡路里消耗预测（sch2-reg-2026-d2-4）

## 一、比赛简介

### 1.1 比赛背景

卡路里（Calorie）是衡量能量的一种单位，我们运动、思考、呼吸都会消耗能量。这场名为「Predict Calorie Expenditure」的比赛（slug 是 sch2-reg-2026-d2-4）要我们预测「一个人做一次运动消耗了多少卡路里」。

比赛的输入是一张典型的**表格数据**（tabular data）：每一行对应一次运动记录，里面有人的身体信息（体重、身高、性别等）和运动过程中的信息（运动时长、心率等）。这类数据在健身 App、智能手环和健康监测软件里非常常见——设备能方便地记录时长和心率，而准确地估计消耗能帮助用户制定锻炼计划、管理体重。

它的难点有三个方面。第一，卡路里消耗不是一个能用单一公式写死的量，它同时受体型、性别、运动强度等多重因素影响，需要模型从数据里学规律。第二，数据里既有数值列（体重、身高、时长、心率），也有文本列（性别），而常用的树模型（比如 XGBoost）只能吃数字，不能直接吃字符串。第三，比赛用的评价指标是 RMSLE（对数空间上的均方根误差），它衡量的是「相对误差」，模型必须适应这种打分方式，而不是笼统地最小化原始的绝对误差。

### 1.2 比赛目标

给定一条运动记录，预测这次锻炼消耗的卡路里数值。训练集 `train.csv` 里有目标列 `Calories`，测试集 `test.csv` 没有这一列，我们要为每条测试样本预测出一个 `Calories`，填到官方给的 `sample_submission.csv` 里提交。

根据参考代码我们能确认的字段如下：

| 列名 | 含义 | 说明 |
| --- | --- | --- |
| `id` | 样本编号 | 辅助字段，不参与建模 |
| `Sex` | 性别 | 文本列，需要编码 |
| `Gender` | 性别 | 文本列，需要编码 |
| `Weight` | 体重 | 用来构造 BMI 特征 |
| `Height` | 身高 | 用来构造 BMI 特征 |
| `Duration` | 运动时长 | 用来构造 Intensity 特征 |
| `Heart_Rate` | 心率 | 用来构造 Intensity 特征 |
| `Calories` | 卡路里消耗 | 目标值，只在训练集出现 |

### 1.3 评价标准

比赛使用 **RMSLE**（Root Mean Squared Logarithmic Error，均方根对数误差）作为评价指标，参考代码里的注释明确写了「Log transform target for RMSLE metric」。它的计算方式是：先把预测值和真实值都取 `log(1+x)`，再算这两组对数之间的 RMSE。因为真实值和预测值都套了对数，所以 RMSLE 衡量的是**相对偏差**：把 100 卡预测成 10 卡（缩小成十分之一）和预测成 1000 卡（放大成十倍），在对数空间里的误差是相同的。它的优势是对极端值不敏感，个别特别离谱的预测不会被放大成灾难性的误差。

RMSLE 的值越小越好，完美预测时是 0。这个指标也直接决定了我们的建模策略：训练时对目标列做同样的 `log1p` 变换，模型学到的就是「对数空间里的卡路里」，正好和评分方式对齐，这是本场比赛中最重要的一个技巧。

---

## 二、解题过程

整场比赛我们的主线思路用一句话概括：**把文本列统一编码成数字，构造 BMI 和运动强度两个特征，对目标做 log1p 变换后训练一棵 XGBoost 回归树，最后再把预测还原成卡路里。** 下面我们按数据探索、数据清洗、模型训练、模型测试四步来走。

### 2.1 数据探索（EDA）

我们首先把三个文件读进来，看看数据长什么样、有哪些列：

```python
# numpy 负责数值计算，pandas 负责读写和处理表格
import numpy as np
import pandas as pd

# 三个文件都放在固定的 input 目录下，路径写完整
TRAIN = '/kaggle/input/sch2-reg-2026-d2-4/train.csv'
TEST  = '/kaggle/input/sch2-reg-2026-d2-4/test.csv'
SUB   = '/kaggle/input/sch2-reg-2026-d2-4/sample_submission.csv'

train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)
submission = pd.read_csv(SUB)

# 打印训练集结构，确认每一列的类型（object 就是文本列）
train.info()
```

我们从 `train.info()` 的输出里看到训练集除了 `id` 和 `Calories`，剩下的特征大致分两类：一类是 `Weight`、`Height`、`Duration`、`Heart_Rate` 这样的数值列，描述人的身体和运动状态；另一类是 `Sex`、`Gender` 这样的文本列，说明同一个信息可能以两种字段形式出现，后面编码时要用统一的办法一次性处理。测试集的结构和训练集一致，只是少了 `Calories`；提交样例只有 `id` 和 `Calories` 两列，这决定了我们最后只要往 `Calories` 里填预测即可。

接着我们快速探查数值列的分布和文本列的情况：

```python
# 数值列的量纲差别很大，先统一看一眼分布
print(train[['Weight', 'Height', 'Duration', 'Heart_Rate']].describe())

# 统计每个文本列的唯一值个数，判断后面编码的工程量
print(train.select_dtypes(include=['object']).nunique())

# 目标 Calories 只能是非负数，且通常右偏，
# 所以后面要用 log1p 压缩分布，这里先记住它的形态
print(train['Calories'].describe())
```

这一步的探查结果告诉我们两件重要的事。第一，数值列之间的量纲差别很大：体重是几十的数量级，而心率是上百的数量级，模型必须自己平衡这种尺度差异。第二，目标列 `Calories` 是一个只能取非负数的连续值，而且像卡路里这种数据通常是右偏的——大多数人消耗适中，少数人消耗很高。右偏的数据直接做回归时，那些大值会主导损失函数，所以参考实现选择对目标做 `log1p` 变换，把偏斜的分布压得接近对称，这个策略和 RMSLE 指标是配套的。

### 2.2 数据清洗

其次我们处理「文本列不能直接进模型」的问题。我们采用的清洗策略是：**先把训练集和测试集拼在一起，再统一对文本列做标签编码**：

```python
from sklearn.preprocessing import LabelEncoder

# 给两个来源各打一个标记，拼接之后还能再拆回去
train['is_train'] = 1
test['is_train'] = 0

# 拼之前先丢掉训练集的 Calories：它是标签，不能被当成特征去编码
df = pd.concat([train.drop(columns=['Calories']), test], axis=0).reset_index(drop=True)

# 用 select_dtypes 自动找出所有文本列（比如 Sex、Gender），
# 而不是硬编码列名，这样以后加新列也不用改代码
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    print(f"Encoding categorical column: {col}")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
```

这一段有三个关键设计。第一，拼接之前先把训练集的 `Calories` 丢掉了，否则这个标签列也会被当成一个「特征」参与编码，等于把答案偷偷送给了模型，还会造成评估失真。第二，我们给两个来源各打一个 `is_train` 标记，拼接、编码、构造特征全部在合并表上进行，处理完再按标记拆回去，保证训练集和测试集经历了完全相同的变换。第三，用 `select_dtypes(include=['object'])` 一次性找出**所有**文本列来编码，而不是写死列名——这样即使数据里出现了参考代码没提到的其他文本列，也能被自动处理，这也是参考实现特别标注的「FIX」写法。

另外值得一提的是，这段代码里并没有出现 `fillna` 之类的缺失值处理，说明这份数据比较干净，各列基本都有值，我们也就没有再额外填补或删除行。

处理完文本列之后，我们构造了两个新特征，来补充原始字段表达不出来的信息：

```python
# BMI = 体重(kg) / 身高(m)^2，是衡量体型胖瘦的常用指标，
# 体型不同，做同样强度的运动消耗的卡路里也不同
if 'Weight' in df.columns and 'Height' in df.columns:
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

# 运动强度：时间越长、心率越快，做的“功”越多，
# 用一个乘积把它们合成一个特征，给模型更直接的信息
if 'Duration' in df.columns and 'Heart_Rate' in df.columns:
    df['Intensity'] = df['Duration'] * df['Heart_Rate']
```

我们构造的两个特征都有明确的身体或运动含义。`BMI`（身体质量指数）是体重除以身高的平方得到的，是衡量一个人胖瘦的常用指标：同样身高的人，越重 BMI 越大。体型不同，消耗自然不同，所以把 BMI 交给模型能帮它区分个体差异。`Intensity`（运动强度）是时长和心率的乘积，虽然两个字段分开看也有信息，但相乘之后更接近「这次运动总共做了多少功」的直觉：时间越长、心率越快，做功越多，越可能消耗更多卡路里。代码里用 `if 'Weight' in df.columns` 这样的条件判断兜底，是考虑到比赛中列名可能有变动，缺列时直接跳过这一步，避免程序崩溃。

### 2.3 模型训练与超参数调优

接着我们把合并表拆回训练集和测试集，选好特征和目标，开始训练模型。我们选的模型是 **XGBoost 回归树**，它是表格数据上最常用、效果也很稳的模型家族——它由很多棵决策树串起来，每棵树都在纠正前面所有树残差剩下的部分：

```python
import xgboost as xgb

# 按来源标记拆回训练/测试，辅助列（is_train、id）不再有用
train_proc = df[df['is_train'] == 1].drop(columns=['is_train', 'id'])
test_proc  = df[df['is_train'] == 0].drop(columns=['is_train', 'id'])

X = train_proc
# 指标是 RMSLE，等价于对 log1p 后的标签做 RMSE，
# 所以在对数空间里训练，让学习目标和评分方式对齐
y = np.log1p(train['Calories'])

model = xgb.XGBRegressor(
    n_estimators=1000,     # 树数量多一点，配合小学习率，拟合得更细
    learning_rate=0.05,    # 每棵树只“纠正”一点点，学得稳
    max_depth=6,           # 单棵树别太深，浅一点不容易过拟合
    subsample=0.8,         # 每棵树只随机抽 80% 的样本训练
    colsample_bytree=0.8,  # 每棵树只随机抽 80% 的特征
    random_state=42,       # 固定随机种子，跑几次结果一致
    n_jobs=-1              # 用满所有 CPU 核心，训练更快
)
model.fit(X, y)
```

这里最值得说明的是 `y = np.log1p(train['Calories'])` 这一行。RMSLE 的评分方式是「先取对数再算 RMSE」，如果我们直接拿原始卡路里去拟合，模型的平方损失和评分指标就不一致，可能把学习重点放在那几个消耗特别大的样本上。把目标先取 `log1p`，训练时的损失就变成了对数空间里的 RMSE，和比赛评分完全对齐，这是这套方案效果好的关键。

模型参数上，我们沿用了一套对树模型来说比较稳健的组合。`n_estimators=1000` 配 `learning_rate=0.05` 是「多棵浅树慢慢学」的思路：学习率小，每棵树只修正一点点，1000 棵树累积起来足以拟合复杂规律，又不容易一上来就过拟合。`max_depth=6` 限制单棵树的深度，`subsample=0.8` 和 `colsample_bytree=0.8` 让每棵树只随机拿到 80% 的样本和 80% 的特征，都能起到抗过拟合的作用。`random_state=42` 固定随机种子，保证结果可复现。需要说明的是，参考 Notebook 采用的是「一组稳妥参数直接训练」的做法，没有跑大规模网格搜索；我们把调参的重心放在「让训练目标和评价指标一致」这个更本质的问题上，后续想进一步提升，可以在此基础上加交叉验证和早停。

### 2.4 模型测试与指标报告

然后我们在拆好的测试集上生成预测，还原成卡路里并写出提交文件：

```python
# 模型输出的是“对数空间里的卡路里”，先取 e 的幂还原成真值
preds_log = model.predict(test_proc)
preds = np.expm1(preds_log)

# 卡路里不可能是负数，把预测值钳到 0 以上
preds = np.maximum(preds, 0)

# 填进样例文件的 Calories 列，生成提交文件，
# index=False 表示不写多余的行号
submission['Calories'] = preds
submission.to_csv('submission.csv', index=False)

# 提交前看一眼前几行，确认列名和行数都对
print(submission.head())
```

模型输出的是我们训练时用的「对数空间值」，所以先用 `np.expm1` 把它还原成真正的卡路里——`expm1` 正是 `log1p` 的逆运算。还原之后，理论上可能出现负数：训练标签全是非负的，但个别样本模型可能预测出很「保守」的低值，取指数之后小于 1，再减 1 就变成负的。实际中卡路里不可能是负数，所以我们用 `np.maximum(preds, 0)` 把预测结果钳到 0 以上。最后把预测填进 `sample_submission.csv` 的 `Calories` 列，再打印前几行确认列名、行数都正确——提交文件格式错误而白白丢分，是比赛中很常见的低级失误。

关于最终分数，我们要诚实说明：参考 Notebook 采用的是「全量训练、直接提交」的策略，没有在本地划分验证集、打印出验证分数，所以材料里没有可引用的具体 RMSLE 得分。但评分逻辑可以从原理上确定——因为我们在训练时用的就是 `log1p(Calories)` 做目标，XGBoost 最小化的平方误差恰好等价于优化 RMSLE，预测时再 `expm1` 还原，整条链路和评价指标完全一致；`max(0, x)` 的钳制也保证了提交的预测值符合「卡路里非负」的常识。RMSLE 只关心相对误差、对极端值不敏感，因此即便个别样本预测偏差较大，总分也不会被拖垮太多。

---

## 三、总结

本文中我们完成了卡路里消耗预测（sch2-reg-2026-d2-4）的完整流程。我们首先用 EDA 摸清了数据：训练集包含 `id`、目标列 `Calories`，以及 `Weight`、`Height`、`Duration`、`Heart_Rate` 等数值列和 `Sex`、`Gender` 等文本列，测试集结构相同只是缺少目标列。其次我们做数据清洗：把训练集和测试集拼接后，用统一的 `LabelEncoder` 处理所有文本列，处理完去掉辅助标记再拆回，保证两边经历完全一致的变换。接着我们构造了 `BMI`（体重除以身高平方）和 `Intensity`（时长乘心率）两个带物理含义的特征，并把目标列做 `log1p` 变换后交给一棵参数稳健的 XGBoost 回归树。然后我们在测试集上预测，用 `expm1` 还原卡路里、钳掉负数，按照样例格式写出提交文件。最后我们说明了 RMSLE 指标的评分逻辑，确认「训练目标与评价指标对齐 + 基础特征工程」这条主线贯穿了整场比赛。

这场比赛给我们最大的启发是：**评价指标不是最后才关心的东西，它应该在最前面就决定你的训练策略。** 因为指标是对数空间的 RMSE，我们就对目标做 `log1p`；因为提交不允许负值，我们就加 `max(0, x)`。同时，把文本列编码、特征构造这些步骤放在「拼接后的整表」上统一执行，能避免训练集和测试集处理不一致这种隐蔽的坑——这些都是表格数据比赛里非常通用、可以反复套用的经验。