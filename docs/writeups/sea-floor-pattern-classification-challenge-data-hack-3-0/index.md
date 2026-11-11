# 海底地貌分类挑战（SeaFloor Pattern Classification - DataHack 3.0）

## 一、比赛简介

### 1.1 比赛背景

这次比赛叫 **SeaFloor Pattern Classification（海底地貌模式分类）**，是 DataHack 3.0 系列比赛中的一个分类任务。所谓「海底地貌」，通俗说就是海底表面天然形成的不同形态——比如平坦的沉积平原、起伏的沙丘、陡峭的海山、幽深的海沟，不同区域的反射信号在特征上会呈现出不同的「模式」。这种分类在海洋测绘、海底资源勘探、水下导航等领域都有实际用途：如果能让机器自动把海底网格分成不同类型的区域，就能省下大量人工判读的成本。

难点在于，这类数据往往不是单张图片，而是把每个采样点整理成了一行特征记录，也就是我们常说的**表格数据（tabular data）**。特征里既有数值型的测量值，也有像区域类别、信号强弱这种**类别特征（categorical feature）**——类别特征的值不是数字，而是像「沙质」「岩质」这样的一串名称，机器学习模型不能直接理解，需要先转换。比赛希望参赛者在这些混合特征上找出规律，训练一个分类器，自动给每片海底区域贴上一个类别标签。

### 1.2 比赛目标

给定每个采样点的特征，预测它的地貌类别。材料中的代码把问题建模成了**多分类（multiclass classification）**问题：每条样本属于若干类别中的某一个，类别在 `class_mapping.csv` 里用 `class_label`（类别名称）和 `class_id`（类别编号）两种方式描述，模型最终输出的是编号。

训练集表格 `train.csv` 主要包含以下几类信息：

- `id`：样本编号，用来唯一区分每条记录；
- `class_label`：目标标签，即这条记录真实属于的地貌类别；
- 若干特征列：既包括 `category_type`（区域类型）、`signal_intensity`（信号强度）、`size_category`（大小类别）、`shape_category`（形状类别）这类**类别特征**，也包括没有被显式编码的其他数值特征。

测试集 `test.csv` 只有特征没有标签。我们要做的是：用训练集学出一个分类器，再对测试集里每条样本判断它最可能是哪个类别，最后把预测结果按比赛要求的格式写进 `submission.csv`。

### 1.3 评价标准

比赛采用的指标是 **宏平均 F1 分数（Macro Average F1-Score）**。我们先解释一下 F1 是什么：对于每个类别，模型预测时会产生「精确率」（预测成这个类别的里面有多大比例真的属于它）和「召回率」（真正属于它的里面有多大比例被找出来），F1 把这两个数调和成一个综合值，介于 0 和 1 之间，越接近 1 越好。**宏平均**的意思是先分别算出每个类别的 F1，再对所有类别取算术平均。由于每个类别被平等对待，宏平均 F1 对「预测概数多的类别」更敏感，因此比普通准确率更适合类别不均衡的数据。材料里的代码全程用它评估，分数越高模型越好。

---

## 二、解题过程

整体思路可以概括成一句话：**把类别特征统一编码成数字，用带早停的 LightGBM 集成树模型做分类，再用 5 折分层交叉验证估计真实水平并生成提交文件。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四个步骤依次进行。

### 2.1 数据探索（EDA）

我们首先把三个数据文件读进来，看一眼表格长什么样、有多少行多少列。材料中的 notebook 还习惯性地遍历了一遍输入目录，以便确认文件名是否和代码里的路径一致：

```python
import numpy as np
import pandas as pd  # 表格读写和处理的常用库

import os
# 遍历输入目录，打印出所有文件，便于核对路径有没有写错
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 按比赛的数据路径读入三个文件
train = pd.read_csv('/kaggle/input/sea-floor-pattern-classification-challenge-data-hack-3-0/train.csv')
test = pd.read_csv('/kaggle/input/sea-floor-pattern-classification-challenge-data-hack-3-0/test.csv')
class_mapping = pd.read_csv('/kaggle/input/sea-floor-pattern-classification-challenge-data-hack-3-0/class_mapping.csv')

# 打印两表的形状：有多少行样本、多少列特征
print(f"Train shape: {train.shape}, Test shape: {test.shape}")
```

这里最关键的一步是发现 `class_mapping.csv` 的存在——它起到了「类别对照表」的作用。因为材料里说训练集的目标列叫 `class_label`，很可能存的是类别名称，而模型输出的编号又必须对应 `class_id`，所以我们必须先把「名称 → 编号」的映射关系建出来，后面清洗才有依据。

接着我们对类别做一次快速的分布检查。多分类任务里，我们总是先确认两个问题：一共几个类别？每类样本量差多少？这决定了后面要不要处理类别不均衡：

```python
# 打印每个类别编号在映射表里是否唯一（类别数）
print(class_mapping.head())
print("类别总数：", len(class_mapping))

# 统计训练集里每个类别各有多少条，看看是否均衡
train['class_label'].value_counts().to_frame().head()
```

由于条件所限，我们没能记录下具体的类别数和样本数，但从比赛设计来看，这是一个类别数量不多的多分类任务。检查均衡性的意义在于：如果某些类别特别少，宏平均 F1 就会主要被这些「少数类」拖累，后续评估和调参都要格外小心。

### 2.2 数据清洗

看过数据结构之后，我们开始做清洗。这里其实有两个动作：一是把训练集的类别名称 `class_label` 映射成整数编号 `class_id`，让标签变成模型能读的 0、1、2……；二是把特征和标签分开，并把类别特征用 **LabelEncoder（标签编码器）** 转成数字。

标签编码有一个容易踩的坑：测试集里可能出现训练集没见过的类别值。比如某个类别特征在训练集里只有 A、B 两种取值，测试集却多了一个 C。如果只用训练集去拟合编码器，再一次转换 C 时就会报错。为了避免这种问题，我们把训练集和测试集的该列拼在一起，让编码器「见」过所有取值后再转换，这一步虽然简单，却是让程序稳定跑通的关键：

```python
# 映射：类别名称 -> 类别编号（整数）
mapping_dict = dict(zip(class_mapping['class_label'], class_mapping['class_id']))
train['class_label'] = train['class_label'].map(mapping_dict)

# 分离特征 X 和标签 y；测试集没有标签，只保留特征
X = train.drop(columns=['id', 'class_label'])
y = train['class_label']
X_test = test.drop(columns=['id'])

# 需要编码的类别特征列表
cat_features = ['category_type', 'signal_intensity', 'size_category', 'shape_category']
for col in cat_features:
    le = LabelEncoder()  # 把字符串类别转成 0,1,2... 的数字
    # 先拼接训练和测试两列再拟合，保证测试集里所有取值都被见过
    combined = pd.concat([X[col], X_test[col]]).astype(str)
    le.fit(combined)
    # 把两边的原文都替换成编码后的数字
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
```

转出来的数字本身没有大小意义（比如「沙质 = 0、岩质 = 1」并不代表岩质比沙质「大」），但对树模型来说这种编码通常是够用的，因为树在分裂时会把编码当作用来判断「等于某个值」的离散标记，而不是当连续数字。这种「训练测试一起拟合」的做法，我们在后面的模型里还会用到 LightGBM 自带的类别特征支持，这里先把它变成数字，是为了统一格式、避免 null 值。

### 2.3 模型训练与超参数调优

清洗完成后，我们进入核心环节：训练模型。我们选用的是 **LightGBM**，一个基于梯度提升树的机器学习库。用大白话说，梯度提升树就是「训练一棵一棵的小决策树，每一棵都专门去纠正前面所有树犯的错，最后把这些树的结论加权求和」。它对表格数据通常效果又快又好，是分类比赛里最常见的工具之一。

训练之前先想清楚评估方式。由于测试集没有标签，我们无法在本地直接算排行榜分数。常用的替代方案是 **K 折交叉验证（K-Fold Cross Validation）**：把训练集切成 K 份，轮流拿 1 份当验证集、其余 K-1 份当训练集，总共训练 K 次，把 K 次验证分数平均起来，就得到了对模型真实水平的较可靠估计。材料选择 K=5，并特意用了**分层抽样（StratifiedKFold）**，保证每一折里各类别的样本比例和全体数据一致，这样验证分数才不会因为某折恰好缺了某个类别而失真：

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

# 5 折分层交叉验证，shuffle 打乱顺序，固定随机种子保证结果可复现
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# oof_preds：记录每一条训练样本在它当验证集那一折时被预测出的类别，
# 这样所有训练样本最终都被「从未见过它们的模型」预测过一次，是真实验证的积累
oof_preds = np.zeros(len(train))
# test_preds_proba：累加每一折模型对测试集输出的各类别概率，最后取平均
test_preds_proba = np.zeros((len(test), len(class_mapping)))
fold_scores = []
```

然后我们写训练主循环。每一折里，我们新建一个 `LGBMClassifier`，用**早停（early stopping）**来控制训练轮数：每训练 50 棵「树」检查一次验证分数，如果连续没有提升就提前停止，其余参数如表所示：

- `n_estimators=1000`：最多允许训练 1000 棵树，上限给足，防止一开始就被限制住；
- `learning_rate=0.05`：学习率，每棵树贡献的步长，调小一点能让模型学得更稳；
- `max_depth=8`：单棵树的最大深度，防止树太深导致过拟合；
- `class_weight='balanced'`：按类别样本量的反比自动加权，让样本少的类别得到更多重视，正好对应宏平均 F1 的评价方式。

```python
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold + 1} ---")
    # 按索引切出本折的训练集和验证集
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # 初始化 LightGBM 分类器
    model = lgb.LGBMClassifier(
        n_estimators=1000,    # 树的上限
        learning_rate=0.05,  # 每棵树的学习步长
        max_depth=8,         # 限制树的深度，防止过拟合
        class_weight='balanced',  # 类别均衡加权，照顾少数类
        random_state=42,
        n_jobs=-1            # 使用所有 CPU 核心加速
    )

    # 训练时把验证集传给模型，配合 early_stopping 自动决定树的数量
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_features,  # 告诉 LightGBM 哪些列是类别特征
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # 对验证集做预测，并记录到 oof_preds 的对应位置
    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds

    # 计算本折的宏平均 F1 并保存
    fold_f1 = f1_score(y_val, val_preds, average='macro')
    fold_scores.append(fold_f1)
    print(f"Fold {fold + 1} Macro F1-Score: {fold_f1:.4f}")

    # 用本折模型预测测试集的概率，累加到数组里，最后除以 N_FOLDS 求平均
    test_preds_proba += model.predict_proba(X_test) / N_FOLDS
```

关于超参数调优，我们在材料里看到的方案是：设定合理上限（1000 棵树、深度 8），再用**早停**在训练过程中自动选出最合适的树数量，这本质上就是一种「自动调参」——我们不需要手动试很多组参数，而是让模型自己决定在哪里停下来。`class_weight='balanced'` 和分层交叉验证这两处设置，也都服务于同一个目标：让宏平均 F1 不被少数类别拖垮。

### 2.4 模型测试与指标报告

5 折训练完成后，我们把所有折的验证预测收集到一起，计算一个总体的 **Out-of-Fold（OOF）宏平均 F1**。OOF 分数的含义是：每一条训练样本都只被「没认真学习过它」的那一折模型预测过，所以它比单次随机划分的验证分数更接近模型在未知数据上的真实表现：

```python
# 用全体 oof_preds 与真实标签 y 对比，计算总体宏平均 F1
cv_score = f1_score(y, oof_preds, average='macro')
print("=" * 40)
print(f"Overall OOF Macro F1-Score: {cv_score:.4f}")
print("=" * 40)

# 打印每一折的分数，检查各折之间是否稳定
print("各折分数：", fold_scores)
```

材料中并没有记录下最终的 OOF 分数数值，因此我们这里不虚构任何数字。但代码的逻辑已经说明了我们评估模型的方式：先保证每一折内部稳定，再看总体 OOF 分数。如果各折分数忽高忽低，说明模型对数据的划分方式很敏感，这时候更该怀疑特征或预处理，而不是急着调参。

验证觉得可行之后，最后一步是生成提交文件。测试集上每一折都产出过一套预测概率，我们对这些概率取平均（**概率平均（probability averaging）**），再取概率最大的类别当作最终预测。这样做比只取单一折的结果更稳健，是集成学习里最简单也最常用的一招：

```python
# 把五折平均概率里概率最大的类别挑出来作为最终预测
final_test_preds = np.argmax(test_preds_proba, axis=1)

# 按提交格式拼装：id 原样保留，class_label 用预测的类别编号
submission = pd.DataFrame({
    'id': test['id'].astype(str).str.zfill(4),  # id 补零成 4 位，与样例格式一致
    'class_label': final_test_preds
})

# 保存成 CSV，无索引列
submission.to_csv('submission.csv', index=False)
print("submission.csv successfully created with Stratified K-Fold predictions!")
```

注意这里有个细节：提交时把 `id` 用 `zfill(4)` 补零成了固定 4 位。比赛对提交格式往往有严格要求，数值相同但格式不同的 `id` 也可能导致提交失败，所以这一步看似不起眼，实际上很重要，我们在写代码时对此做了专门处理。

---

## 三、总结

本文中我们完成了「海底地貌分类挑战」的一整套解题流程。我们首先做了 EDA：读入训练集、测试集和类别映射表，确认了表的形状、列含义和类别基本信息。其次做数据清洗：我们把类别名称映射成整数编号，并对 `category_type`、`signal_intensity`、`size_category`、`shape_category` 这几个类别特征做了标签编码，为了避免测试集出现训练集没见过的类别值，编码时特意把两份数据拼在一起拟合。接着我们训练模型：选用 LightGBM 梯度提升树，配合 `class_weight='balanced'` 和 5 折**分层**交叉验证，用早停自动决定树的数量，并在每折对测试集做概率预测后取平均。最后我们生成了总体 OOF 宏平均 F1 作为模型自评指标，并按比赛格式（`id` 补零到 4 位）输出了 `submission.csv`。

这个比赛带给我们最有价值的经验有三点：第一，**读数据前先找「映射表」**——多分类比赛的 `class_mapping.csv` 往往决定了标签到底该怎么编码，方向搞对了后面才不会返工；第二，**类别特征要统一用同一套编码器处理训练和测试数据**，这是表格数据清洗中最容易踩的坑；第三，**评价方式决定模型设计**——既然用宏平均 F1 打分，就既要靠分层交叉验证保证每一折类别均衡，也要靠 `class_weight='balanced'` 在模型层面照顾少数类，两头一起努力才能让最终分数站得住。