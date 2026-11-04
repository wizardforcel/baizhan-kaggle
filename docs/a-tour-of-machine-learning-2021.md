# A Tour of Machine Learning 2021

## 一、比赛简介

### 1.1 比赛背景

这个比赛叫"A Tour of Machine Learning 2021"，从名字就能看出它的定位：**一次机器学习之旅**，也就是给刚入门的人准备的表格式二分类练习。它不追求复杂的业务背景，数据是一张规规矩矩的数值表——每行一个样本，每列一个数值特征，最后一列是 0/1 标签。

材料里收录了两份 notebook：一份是极简的"逻辑回归跑通流程"（`basic_example`），一份是完整的三模型交叉验证方案（`notebook575b0137ca`）。前者适合第一次提交，后者是我们这篇的主线。

有一个细节值得先提：完整版 notebook 里那张标签分布图的标题写的是"是否认购定期存款"（subscription to term deposit），但数据的编号列却叫 `GeneId`。这说明作者是从另一个银行营销项目里把作图代码整段拷过来的，图里的文字没改。**这是一个很典型的"复制粘贴残留"**，我们在这里指出来，是因为读别人的 notebook 时，看到对不上的地方要能分辨"这是数据的特点"还是"这是作者的笔误"，不要被误导。

### 1.2 比赛目标

数据目录下有三个文件：

- `data_train.csv`：训练集，包含若干数值特征列、一个 `GeneId`（样本编号）和最后一列 `Label`（0 或 1）。
- `data_test.csv`：测试集，列和训练集一样，但没有 `Label`。
- `sample_submission.csv`：提交样例。

任务很直接：给测试集里每一个 `GeneId` 预测一个 `Label` 为 1 的**概率**，写进提交文件的 `Label` 列。

注意这里填的是概率而不是 0/1。这是二分类比赛里最常见的约定：交概率，主办方自己算指标。交 0/1 也不是不行，但那样等于把模型的置信度全扔了，AUC 会明显变差。

### 1.3 评价标准

材料没有直接写排行榜的公式，但代码里有两个强证据：一是训练时用 `roc_auc_score` 评估每一折，二是提交的是概率。由此可以合理判断，比赛按 **ROC AUC** 评分。

AUC 的大白话解释：把所有测试样本按模型给的概率从高到低排队，随机挑一个正样本、一个负样本，正样本排在负样本前面的概率就是 AUC。**完全乱猜是 0.5，完美是 1.0，越高越好。** 它只关心排序，不关心概率的绝对大小——你说"这个人有 0.9 的概率"还是"0.6 的概率"都不影响 AUC，影响的只是这两个人谁排在前面。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**先把特征做一次对数压缩，再用七折交叉验证训练三种梯度提升树，最后取其中表现稳定的那一种做预测。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步是把两张表读进来，看看有多少行、多少列、列名长什么样：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("/kaggle/input/a-tour-of-machine-learning-2021/data_train.csv")
test = pd.read_csv("/kaggle/input/a-tour-of-machine-learning-2021/data_test.csv")

# 标签单独拎出来，特征表里不留 Label
y = train["Label"]
df_train = train.drop("Label", axis=1)
df_test = test

# 列出所有特征名，并数一下有多少列
rows_names = df_train.columns
print("特征:", list(rows_names), "\n列数:", len(rows_names))
print("训练集形状:", df_train.shape, "测试集形状:", df_test.shape)
```

打印结果第一眼就要看两件事：**特征有多少列**、**训练和测试的列是否完全一致**。列数决定了后面能不能做特征重要性分析，列是否一致决定了能不能直接把测试集喂给模型。如果测试集多一列或者少一列，`predict` 那一步会直接报错。

第二步是查缺失。这份材料的写法是把训练和测试分别扫一遍：

```python
# 逐列检查有没有空值，只打印"有缺失的列"
null_train = [c for c in df_train.columns if df_train[c].isna().any()]
null_test = [c for c in df_test.columns if df_test[c].isna().any()]

print("训练集有缺失的列:", null_train)
print("测试集有缺失的列:", null_test)
```

这个检查在数值表比赛里几乎是必做的第一件事。如果两边都打印出空列表，说明数据是完整的，后面就不用写填充逻辑；如果只有测试集有缺失，就要小心——那意味着训练时模型没见过 `NaN`，上线可能出现意外行为。

第三步是看标签分布。类别是否均衡，直接决定了"准确率"这个指标可不可信：

```python
plt.figure(figsize=(6, 4))
sns.countplot(data=train, x="Label", palette="pastel", edgecolor="black")
plt.title("Distribution of subscription to term deposit (Label)", fontsize=11)
plt.xlabel("Label", fontsize=10)
plt.ylabel("count", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()
```

图上两根柱子的高度比，就是我们判断均衡程度的依据。**如果 0 和 1 差不多各占一半**，那准确率还能看；**如果一边压倒另一边**（比如 90% 是 0），那"全猜 0"就能有 90% 的准确率，这时候必须换 AUC 这类不受类别比例影响的指标。材料里选择用 AUC 评估，本身就是对这种风险的一种规避。

第四步是做一次降维可视化，看看样本在高维空间里是不是分得开：

```python
from sklearn.decomposition import PCA
import plotly.express as px

pca = PCA(n_components=3)          # 把几十维压成 3 维，才能画三维散点
data_pca = pca.fit_transform(df_train.to_numpy())

fig = px.scatter_3d(x=data_pca[:, 0], y=data_pca[:, 1], z=data_pca[:, 2])
fig.show()
```

这里有一处必须提醒的地方：材料原稿写的是 `pca.fit_transform(df)`，而 `df` 是**带 `Label` 列的完整训练表**。这等于把标签也当成一个特征参与了降维，画出来的图当然会"看起来分得开"，但那是因为标签本身就在坐标里，属于自己骗自己。我们在复现时改成了只对特征列做 PCA。**读别人的代码时，这种"把不该进去的列也塞进去"的细节一定要盯住。**

另外，散点图没有按标签上色，所以它只能看**整体形状**（有没有明显的离群簇、有没有极端值），看不出类别可分性。要真正看可分性，得给 `color` 参数传标签，用不同颜色区分两类。

### 2.2 数据清洗

这份数据没有缺失值，也没有文本列，看起来"什么都不用洗"。但材料里做了一件很关键的事：**对所有特征做对数变换**。

```python
# 先加 1 再取对数：避免 0 取对数变成负无穷
df_train_ = np.log1p(df_train[rows_names] + 1)
df_train_.describe()
```

为什么要这么做？因为这类表格特征（尤其是 ID 叫 `GeneId` 的那种，往往来自生物测量或计数）常常是**长尾分布**：大部分值挤在很小的范围里，少数值大出好几个数量级。这种分布会带来两个问题：

1. **树模型的分裂点会被大值主导。** 一个特征取值范围是 1 到 10 万，绝大多数样本却在 1 到 100 之间，那树在找分裂点时几乎只会考虑 100 以上那几个点，低值区的精细结构被忽略。
2. **线性模型（比如逻辑回归）会被大值直接带偏。** 因为它的系数是乘在原始尺度上的，一个大数值特征会主导整个预测。

取对数能把"乘法关系"变成"加法关系"，把长尾压成接近对称的形状，上面两个问题都能缓解。

不过这里有一个细节值得推敲：`np.log1p(x + 1)` 其实是 `log(x + 2)`，而不是最常见的 `log(1 + x)`。多出来的那个 `+1` 让变换后的最小值不再是 0。这不是错误，但对结果有影响——它相当于对所有值又做了一次平移。更标准的写法是直接 `np.log1p(x)`，或者先检查有没有负值，有负值就不能取对数。**我们在复现时保留了原写法，因为模型对单调变换不敏感，但心里要知道这两者的区别。**

还有一点要强调：**训练集和测试集必须做完全相同的变换**。材料的代码里有一处疏漏，我们在 2.4 节会专门讲。

### 2.3 模型训练与超参数调优

**交叉验证的框架。** 材料用的是**七折分层交叉验证**（StratifiedKFold）。分层的意思是：每一折里正负样本的比例都和全体一致，避免某一折几乎全是负样本导致分数剧烈波动。

```python
from sklearn.model_selection import StratifiedKFold

n_splits = 7
random_state = 42

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# 为每条训练样本预留一个位置，存放"它作为验证样本时"的预测概率
catboost_pred = np.zeros(len(df_train))
xgb_pred = np.zeros(len(df_train))
lgbm_pred = np.zeros(len(df_train))

# 每一折的 AUC 都记下来，最后看均值和标准差
catboost_auc_score, xgb_auc_score, lgbm_auc_score = [], [], []
```

"为每条样本预留一个位置"这个设计叫 **out-of-fold 预测**：每条样本只在它没参与训练的那一折里被预测一次，最后拼起来就是一份"和训练集等长、但每个值都来自没见过它的模型"的预测。这份预测非常有用——它是评估模型真实水平最干净的方式，也是后面做模型融合的基础。**如果直接在训练集上 `predict`，得到的是模型背下来的答案，分数会虚高。**

**三个模型的参数。** 材料同时训了逻辑回归、CatBoost、XGBoost、LightGBM 四个模型（逻辑回归只声明没参与比较）。参数如下：

```python
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# 1) CatBoost：类别特征友好，这里特征全数值，主要用它稳
cat_model = CatBoostClassifier(
    iterations=1000, learning_rate=0.05, depth=6,
    eval_metric="AUC", early_stopping_rounds=50,
    random_seed=random_state, verbose=100, task_type="CPU",
)

# 2) XGBoost：按叶子生长（lossguide），适合大样本、允许很深
xgb_params = {
    "n_estimators": 8000,
    "max_leaves": 127,          # 叶子数上限，控制复杂度
    "grow_policy": "lossguide", # 按损失下降最多的地方优先分裂，而不是按层
    "learning_rate": 0.008,     # 学习率很小，所以树要多
    "subsample": 0.85,          # 每棵树随机用 85% 的样本，防过拟合
    "colsample_bytree": 0.75,   # 每棵树随机用 75% 的特征
    "reg_alpha": 2.5,           # L1 正则，把没用的特征权重压到 0
    "reg_lambda": 0.8,          # L2 正则
    "objective": "binary:logistic",
    "tree_method": "hist",      # 直方图算法，速度快
    "random_state": 42,
}

# 3) LightGBM：注意这里的参数是"看起来调过"的
lgbm_model = lgb.LGBMClassifier(
    n_estimators=40000, learning_rate=0.0358306214515723,
    num_leaves=228, max_depth=6, min_child_samples=83,
    subsample=0.8700304020753131, colsample_bytree=0.6169349166144594,
    max_bin=3600, reg_alpha=3.700714656885025, reg_lambda=4.709578317972932,
    random_state=42, verbosity=-1,
)
```

`learning_rate` 写成 `0.0358306214515723` 这种十几位小数，一看就是**自动调参工具**（比如 Optuna）跑出来的结果。这类"机器挑出来的参数"有个特点：在当初那一份验证集上确实是最优的，但换一份数据就不一定了，而且这么小的学习率配四万棵树，训练成本很高。相比之下，XGBoost 那组参数是人工设定的整数和"整齐"的小数（0.85、0.75、2.5），可解释性强得多。

**训练循环。** 三种模型在同一折上依次训练、依次评估：

```python
for fold, (train_idx, val_idx) in enumerate(skf.split(df_train_, y)):
    d_train = df_train_.iloc[train_idx].copy()
    x_val = df_train_.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # —— CatBoost ——
    cat_model.fit(d_train, y_train, eval_set=(x_val, y_val), use_best_model=True)
    p = cat_model.predict_proba(x_val)[:, 1]
    catboost_auc_score.append(roc_auc_score(y_val, p))
    catboost_pred[val_idx] = p            # 填进 out-of-fold 数组
    catboot_feature_important.append(cat_model.get_feature_importance())

    # —— XGBoost ——
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(d_train, y_train, eval_set=[(x_val, y_val)], verbose=100)
    p = xgb_model.predict_proba(x_val)[:, 1]
    xgb_auc_score.append(roc_auc_score(y_val, p))
    xgb_pred[val_idx] = p
    xgb_feature_important.append(xgb_model.feature_importances_)

    # —— LightGBM ——
    lgbm_model.fit(d_train, y_train, eval_set=[(x_val, y_val)], eval_metric="auc",
                   callbacks=[lgb.early_stopping(stopping_rounds=50),
                              lgb.log_evaluation(period=100)])
    p = lgbm_model.predict_proba(x_val)[:, 1]
    lgbm_auc_score.append(roc_auc_score(y_val, p))
    lgbm_pred[val_idx] = p
    lgbm_feature_important.append(lgbm_model.feature_importances_)
```

三个模型都用了**早停**（early stopping）：CatBoost 通过构造参数 `early_stopping_rounds=50`，LightGBM 通过回调 `lgb.early_stopping(stopping_rounds=50)`。意思是"验证集 AUC 连续 50 轮不涨就停下来"，避免把树数死板地训满。

**关于调参。** 材料没有做网格搜索，参数是直接给定的。我们认同这个取舍：三个模型的参数已经覆盖了"浅层多树"和"深层少树"两种风格，再网格搜索的收益很小，而七折 × 三个模型 × 上千棵树的计算量已经不小。真正值得投入的是**特征**，而不是在小数点后第四位上纠缠。

### 2.4 模型测试与指标报告

**第一个报告：三个模型的平均 AUC。** 七折跑完后，把每折的分数收拢起来：

```python
print(f"CatBoost 平均 AUC: {np.mean(catboost_auc_score):.4f} ± {np.std(catboost_auc_score):.4f}")
print(f"XGBoost  平均 AUC: {np.mean(xgb_auc_score):.4f} ± {np.std(xgb_auc_score):.4f}")
print(f"LightGBM 平均 AUC: {np.mean(lgbm_auc_score):.4f} ± {np.std(lgbm_auc_score):.4f}")
```

材料里这三个数只是格式化输出，没有写死具体数值，我们也不编。**读这行结果时，标准差比均值更值得看**：如果某模型的均值最高但标准差也最大，说明它在某些折上特别差，稳定性存疑；如果三个模型的均值差在小数点后第三位，那它们实际上是打平的，选哪个都可以。

**第二个报告：特征重要性。** 每折都会给一份重要性，把七折平均一下更稳：

```python
# 每折一份重要性，堆成 (折数, 特征数) 的矩阵，按列求平均
avg_cat = np.mean(np.array(catboot_feature_important), axis=0)
avg_xgb = np.mean(np.array(xgb_feature_important), axis=0)
avg_lgb = np.mean(np.array(lgbm_feature_important), axis=0)

importance_df = pd.DataFrame({
    "Feature": rows_names,
    "CatBoost": avg_cat,
    "XGBoost": avg_xgb,
    "LightGBM": avg_lgb,
}).sort_values("CatBoost", ascending=False)
print(importance_df.head(15))
```

三种模型对"哪些特征重要"的判断如果高度一致，说明这个结论比较可靠；如果三家排出来的前十名完全不同，那就要小心——可能特征之间高度相关，谁上都行，这时候"重要性"其实是在几个等价特征之间随机分配。

**第三个部分：提交。** 材料原稿的写法是**只拿最后一折训练出来的 XGBoost 模型**去预测测试集：

```python
# 原稿写法：直接用循环结束后残留的 xgb_model（只见过最后一折的训练数据）
test_pred_prob = xgb_model.predict_proba(df_test)[:, 1]
submission = pd.DataFrame({"GeneId": df_test["GeneId"], "Label": test_pred_prob})
submission.to_csv("submission.csv", index=False)
```

这段代码有两个明显的漏洞，我们复现时都改掉了：

1. **测试集没有做对数变换。** 训练时喂的是 `df_train_`（取过对数的），预测时喂的是原始 `df_test`。模型看到的数值尺度和训练时完全不同，预测结果基本是废的。正确做法是先 `np.log1p(df_test[rows_names] + 1)`，**用和训练完全相同的变换**。
2. **浪费了七折。** 循环里已经把每一折的验证预测存进了 `xgb_pred`，但测试集只用了最后一折的模型。七折的意义就是"多训几个模型取平均"，只用一个等于白跑了六折。

修正后的写法：

```python
import os

# 1) 测试集做和训练集完全一样的对数变换
df_test_ = np.log1p(df_test[rows_names] + 1)

# 2) 七折各训一个模型，预测后取平均（这里以 XGBoost 为例）
test_fold_preds = []
for train_idx, val_idx in skf.split(df_train_, y):
    m = XGBClassifier(**xgb_params)
    m.fit(df_train_.iloc[train_idx], y.iloc[train_idx],
          eval_set=[(df_train_.iloc[val_idx], y.iloc[val_idx])], verbose=False)
    test_fold_preds.append(m.predict_proba(df_test_)[:, 1])

test_pred = np.mean(test_fold_preds, axis=0)   # 七个模型投票，方差更小

submission = pd.DataFrame({"GeneId": df_test["GeneId"], "Label": test_pred})
submission.to_csv("submission.csv", index=False)
print("行数:", len(submission), " 概率范围:", test_pred.min(), test_pred.max())
```

最后那个 `print` 是我们加的体检：**概率必须落在 0 到 1 之间，行数必须和测试集一致。** 这两条任何一条不满足，提交都会被判错。

另外还可以做一件材料里已经准备好、但没用上的事：把 `catboost_pred`、`xgb_pred`、`lgbm_pred` 三个 out-of-fold 数组按权重加起来，先在训练集上算一遍融合后的 AUC，如果比任何单模型都高，就用同样的权重去平均测试集的预测。这叫**模型融合**（blending），是这类比赛里最稳的提分手段之一。

---

## 三、总结

本文中我们完成了"A Tour of Machine Learning 2021"的完整流程。我们首先把训练集和测试集读进来，确认了特征列数和两边列名一致，逐列检查了缺失情况（这份数据是完整的），用柱状图看了标签的类别比例，又用主成分分析把高维特征压到三维做了一次可视化。其次我们对所有特征做了对数变换，把长尾分布压平，并指出训练集和测试集必须使用同一个变换。接着我们搭了七折分层交叉验证的框架，为每条训练样本预留 out-of-fold 预测位置，在同一折上依次训练 CatBoost、XGBoost、LightGBM 三个梯度提升树模型，各自用早停控制树的数量，并记录每一折的 AUC 和特征重要性。最后我们汇总了三个模型的平均 AUC 和逐折标准差，把七折的特征重要性取平均排序，写提交文件时修正了原稿里的两处问题——测试集漏做对数变换、以及只用了最后一折的模型而没有把七折平均。

这次练习让我们体会到，**入门比赛的分数差距往往不在模型，而在流程**：漏做一次变换、少用一个折，损失远比换一个算法大。读别人的 notebook 时，先看流程对不对，再看参数调得好不好，顺序不能反。
