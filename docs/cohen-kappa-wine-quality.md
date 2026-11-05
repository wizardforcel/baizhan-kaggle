# 葡萄酒品质评分的二次加权 Kappa（Cohen Kappa Wine Quality）

## 一、比赛简介

### 1.1 比赛背景

葡萄酒的品质通常由品酒师打分：同一批酒让几位专家盲品，给一个整数分数（比如 5 分、6 分、7 分）。这批数据来自经典的 UCI Wine Quality 数据集，每行是一种酒的**理化检测结果**，包括：

- 固定酸度（fixed acidity）、挥发性酸度（volatile acidity）、柠檬酸（citric acid）；
- 残糖（residual sugar）、氯化物（chlorides）；
- 游离二氧化硫（free sulfur dioxide）、总二氧化硫（total sulfur dioxide）；
- 密度（density）、pH 值、硫酸盐（sulphates）、酒精度（alcohol）；
- 以及我们要预测的 `quality`——专家给出的品质评分。

难点在于：**品质不是「非黑即白」的类别，而是有顺序的等级**。把 6 分猜成 7 分只是「稍微差一点」，猜成 3 分就离谱了。普通准确率会把这两种错误一视同仁，而真正的业务需求是「错得越远罚得越重」。

### 1.2 比赛目标

输入是每瓶酒的理化指标，输出是整数品质分。训练集里 `quality` 取值在 3 到 8 之间，但分布极不均衡：绝大多数是 5 分和 6 分，3 分和 8 分非常稀少。测试集只给理化指标和 `id`，要填上 `quality`。

提交文件只有两列：`id` 和 `quality`，每一行对应测试集里的一瓶酒。

### 1.3 评价标准

指标是 **Quadratic Weighted Kappa（QWK，二次加权 Kappa）**。

先说 Kappa。两个人（或人和模型）给同一批样本打等级，一致率看起来很高，但可能有很大一部分是「碰巧一致」——比如 80% 的酒都是 6 分，闭着眼睛全填 6 分也有一致率。Kappa 做的事就是**扣掉碰巧一致的部分**：

$$\kappa = 1 - \frac{\text{实际的不一致}}{\text{碰巧会有的不一致}}$$

「二次加权」是说：不一致不是简单数个数，而是按「分差」的平方加权。差 1 分罚 1，差 2 分罚 4，差 3 分罚 9。分母里的期望不一致也用同样的权重算。这样：

- 完全一致：QWK = 1；
- 和瞎猜差不多：QWK = 0；
- 比瞎猜还差：QWK 可以是负数。

所以 QWK **越高越好**，而且它鼓励模型「大方向别错」，不苛求每个边界都分毫不差。

---

## 二、解题过程

我们的主线是：**先把评分当回归问题做，得到连续的「品质分」，再用一组阈值把它切成整数等级；同时用 XGBoost 与 CatBoost 两个模型集成，最后在 out-of-fold 预测上专门优化阈值。** 下面按四步说明。

### 2.1 数据探索（EDA）

第一步是读数据、看形状、看缺失、看分布：

```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns

train = pd.read_csv('/kaggle/input/cohen-kappa-wine-quality/train.csv')
test  = pd.read_csv('/kaggle/input/cohen-kappa-wine-quality/test.csv')
print(train.shape, test.shape)
print(train.isna().sum().sum())     # 缺失值总数

# 品质分布：这是全篇最重要的一张图
sns.histplot(train['quality'], bins=range(3, 10), kde=True)
plt.xticks(range(3, 9))
plt.show()
```

图上能清楚看到：**5 分和 6 分占了绝大多数，3 分与 8 分只有个位数或几十个**。这个不均衡直接决定了两件事：一是不该用普通准确率评估（全猜 6 分也能混到很高准确率）；二是训练时要考虑类别权重，否则模型会完全忽略稀有等级。

接着看理化指标与品质的相关性：

```python
# 去掉 id 后算相关矩阵
corr = train.drop(columns=['id']).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
```

材料里这个数据集的相关性有两个结论：

1. **酒精度与品质正相关最明显**——度数高的酒往往评分更高；
2. **挥发性酸度与品质负相关**——挥发酸高了会有醋味，评分低。
3. 另外，`free sulfur dioxide` 与 `total sulfur dioxide`、`fixed acidity` 与 `pH`、`density` 与 `alcohol` 之间都强相关，说明原始特征里存在明显的冗余和物理耦合。

这些相关性给了我们做特征工程的依据：**不要只把 11 个原始列丢进去，要把有物理意义的比值和差值造出来**。

### 2.2 数据清洗

这份数据本身很干净：没有缺失值，也没有明显需要删除的异常行。材料的处理重点在**特征工程**，我们造了 9 个新特征，每一个都有化学上的理由：

```python
def add_features(frame):
    x = frame.copy()
    eps = 1e-6   # 防止除以 0

    # 总酸度 = 固定酸 + 挥发酸 + 柠檬酸，衡量整体酸感
    x['total acidity'] = x['fixed acidity'] + x['volatile acidity'] + x['citric acid']

    # 结合态二氧化硫 = 总量 - 游离量，抗氧化能力的另一半
    x['bound sulfur dioxide'] = x['total sulfur dioxide'] - x['free sulfur dioxide']

    # 游离硫占比：防腐效率与气味风险的平衡
    x['free total sulfur ratio'] = x['free sulfur dioxide'] / (x['total sulfur dioxide'] + 1.0)

    # 挥发酸 / 固定酸：坏酸相对好酸的比例，与异味直接相关
    x['volatile fixed ratio'] = x['volatile acidity'] / (x['fixed acidity'] + eps)

    # 糖 / 酒精：甜度与发酵程度的关系
    x['sugar alcohol ratio'] = x['residual sugar'] / (x['alcohol'] + eps)

    # 硫酸盐与酒精的乘积：两者都是品质的正向信号
    x['sulphates alcohol product'] = x['sulphates'] * x['alcohol']

    # 酒精与密度交互：酒精越高密度越低，乘积刻画综合「酒体」
    x['alcohol density interaction'] = x['alcohol'] * x['density']

    # 固定酸 / pH：酸度对酸碱度的直接体现
    x['acidity pH ratio'] = x['fixed acidity'] / (x['pH'] + eps)

    # 分子态二氧化硫近似：游离硫中真正有抗菌活性的部分随 pH 急剧变化
    x['molecular sulfur proxy'] = x['free sulfur dioxide'] / np.power(10.0, x['pH'])
    return x

X = add_features(train.drop(columns=['id', 'quality']))
X_test = add_features(test.drop(columns=['id']))
```

最后那个 `molecular sulfur proxy` 值得多说一句：二氧化硫的杀菌能力主要来自「分子态」，而分子态的比例由 pH 决定——pH 每降 1，分子态大约多 10 倍。所以用 `游离硫 / 10^pH` 做近似，比单看游离硫更能反映真实的防腐强度。

我们没有做标准化，也没有做缺失填补——因为树模型（XGBoost、CatBoost）对特征的单调变换不敏感，量纲和分布形状都不影响分裂点的选择。

### 2.3 模型训练与超参数调优

**为什么用回归而不是分类？** 这是 QWK 任务的一个关键技巧。QWK 惩罚的是「距离」，而分类模型把 5 分和 6 分当两个互不相干的类别，学不到「它们挨着」这件事。回归模型输出连续值，天然带有序关系，最后只要选好切分阈值，就能兼顾「顺序」和「类别」。

我们训练两个模型，各取所长：

**XGBoost 回归器。** 树的数量 1800，深度 5，学习率 0.018（小学习率 + 多棵树是常见的稳健配置），行采样 0.82、列采样 0.86 防过拟合，L2 正则 6.0，并用早停（连续 140 轮不提升就停）。另外给每个样本加了**类别权重**：稀有等级权重更大，避免模型只顾 5、6 分。

**CatBoost 分类器。** 用 `MultiClass` 损失，输出 6 个等级的概率，再按概率加权求和得到期望分数（比如 P(5)=0.6、P(6)=0.4，期望就是 5.4）。CatBoost 的对称树和有序提升对中小数据集比较稳，和 XGBoost 形成互补。

两者都在 **5 折分层交叉验证**下训练，并保存 out-of-fold（OOF）预测——每个样本的预测都来自「没见过它」的那一折。OOF 是后面调阈值唯一可信的依据。

```python
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from catboost import CatBoostClassifier

SEED, N_SPLITS = 42, 5
CLASS_VALUES = np.arange(3, 9)       # 等级 3~8

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_xgb = np.zeros(len(train)); oof_cat = np.zeros(len(train))
test_xgb = np.zeros(len(test)); test_cat = np.zeros(len(test))

# 类别权重：越稀有的等级权重越大（开方压缩，避免过度补偿）
counts = np.bincount(y - 3, minlength=6)
cat_class_weights = (len(y) / (6 * counts)) ** 0.22

for fold, (tr, va) in enumerate(cv.split(X, y), 1):
    xgb = XGBRegressor(
        objective='reg:squarederror', n_estimators=1800, max_depth=5,
        learning_rate=0.018, min_child_weight=5.0,
        subsample=0.82, colsample_bytree=0.86,
        reg_alpha=0.12, reg_lambda=6.0, gamma=0.01,
        tree_method='hist', n_jobs=-1,
        random_state=7200 + fold - 1, early_stopping_rounds=140)
    xgb.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
    oof_xgb[va] = xgb.predict(X.iloc[va])
    test_xgb += xgb.predict(X_test) / N_SPLITS   # 5 折测试预测求平均

    cat = CatBoostClassifier(
        loss_function='MultiClass', iterations=1100, depth=7,
        learning_rate=0.035, l2_leaf_reg=12.0, random_strength=0.4,
        bagging_temperature=1.2, border_count=254,
        class_weights=cat_class_weights.tolist(),
        random_seed=2026, verbose=False, allow_writing_files=False)
    cat.fit(X.iloc[tr], y[tr] - 3,
            eval_set=(X.iloc[va], y[va] - 3),
            early_stopping_rounds=130, verbose=False)
    # 概率加权求期望分数，得到连续输出
    oof_cat[va] = cat.predict_proba(X.iloc[va]) @ CLASS_VALUES
    test_cat += (cat.predict_proba(X_test) @ CLASS_VALUES) / N_SPLITS
```

注意 CatBoost 训练时标签减了 3（变成 0~5），因为多分类要求标签从 0 开始连续编号；预测时再乘回 `CLASS_VALUES`（3~8）还原。

**融合。** 两个模型的连续预测按 **0.70（XGB）: 0.30（CatBoost）** 加权。这个比例不是随便拍的，而是根据 OOF 表现试出来的：XGBoost 单独更强，占大头；CatBoost 提供一点差异化的纠偏。

```python
XGB_WEIGHT, CAT_WEIGHT = 0.70, 0.30
oof_blend  = XGB_WEIGHT * oof_xgb  + CAT_WEIGHT * oof_cat
test_blend = XGB_WEIGHT * test_xgb + CAT_WEIGHT * test_cat
```

**阈值优化：本篇最核心的一步。** 回归输出的是 5.37 这样的连续值，要变成整数等级，需要 5 个切分点 $t_1<\dots<t_5$：小于 $t_1$ 判 3 分，$[t_1,t_2)$ 判 4 分，以此类推。最直接的想法是四舍五入（等价于阈值 3.5、4.5、…、7.5），但既然评分分布严重不均衡，固定阈值一定不是最优——比如 8 分太少，模型几乎不会输出大于 7.5 的值，那么 8 分就永远预测不出来。

我们的做法是**直接以 QWK 为目标函数搜索阈值**：

```python
from scipy.optimize import minimize

def fast_qwk(y_true, y_pred):
    """快速计算 QWK：把 3~8 平移到 0~5，用 6x6 混淆矩阵一次算完"""
    a = np.clip(y_true.astype(np.int16), 3, 8) - 3
    b = np.clip(y_pred.astype(np.int16), 3, 8) - 3
    observed = np.bincount(a * 6 + b, minlength=36).reshape(6, 6).astype(float)
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    grid = np.arange(6, dtype=float)
    weights = ((grid[:, None] - grid[None, :]) ** 2) / 25.0   # 二次权重
    return 1.0 - np.sum(weights * observed) / np.sum(weights * expected)

def apply_thresholds(pred, thresholds):
    # digitize 返回「落在第几个区间」，+3 还原成 3~8 的等级
    return np.digitize(pred, np.sort(np.asarray(thresholds))) + 3

def optimize_thresholds(y_true, pred, seed=42):
    rng = np.random.default_rng(seed)
    # 起点 1：常规四舍五入的边界
    # 起点 2：一组经验边界
    # 起点 3：按训练集真实类别频率取分位数（保持预测分布与真实分布一致）
    counts = np.array([(y_true == c).sum() for c in CLASS_VALUES], dtype=float)
    base = np.quantile(pred, np.cumsum(counts)[:-1] / counts.sum())
    starts = [np.array([3.5, 4.5, 5.5, 6.5, 7.5]),
              np.array([5.10, 5.27, 5.53, 6.02, 6.44]), base]
    # 再加 14 组在 base 附近抖动的小随机起点，避免陷入局部最优
    starts += [np.sort(base + rng.normal(0, 0.22, 5)) for _ in range(14)]

    def loss(t):
        t = np.sort(t)
        if np.min(np.diff(t)) < 0.03:   # 阈值必须严格递增且不能太挤
            return 2.0
        return -fast_qwk(y_true, apply_thresholds(pred, t))  # 最小化负 QWK

    best_t, best_score = starts[0], -1.0
    for start in starts:   # 多起点 + Nelder-Mead 单纯形法
        res = minimize(loss, np.sort(start), method='Nelder-Mead',
                       options={'maxiter': 900, 'xatol': 2e-4, 'fatol': 1e-7})
        cand = np.sort(res.x)
        score = fast_qwk(y_true, apply_thresholds(pred, cand))
        if score > best_score:
            best_t, best_score = cand, score

    # 最后用逐步缩小的网格做局部爬山，把边界再磨细一点
    for step in (0.08, 0.03, 0.012, 0.004):
        improved = True
        while improved:
            improved = False
            for j in range(5):
                for direction in (-1.0, 1.0):
                    cand = best_t.copy()
                    cand[j] += direction * step
                    if np.min(np.diff(cand)) < 0.03:
                        continue
                    score = fast_qwk(y_true, apply_thresholds(pred, cand))
                    if score > best_score + 1e-12:
                        best_t, best_score, improved = cand, score, True
    return best_t, best_score
```

这里有几处工程细节值得学：

- **多起点**：QWK 作为阈值函数是不光滑、非凸的，单次优化很容易卡在局部最优；用 3 个有依据的起点加 14 个随机扰动起点，取最好的结果。
- **约束阈值间距**：如果两个阈值挤在一起（比如差 0.001），就会有一个等级几乎不可能被预测出来，QWK 也会因为分母退化而抖动。我们要求最小间隔 0.03。
- **快速 QWK**：直接用 `bincount` 构造混淆矩阵，比反复调用 sklearn 的 `cohen_kappa_score` 快很多；优化要调用上千次，速度很关键。

### 2.4 模型测试与指标报告

阈值在 OOF 预测上分别对 XGB、CatBoost 和融合结果优化，得到三个分数：

```python
xgb_t, xgb_score     = optimize_thresholds(y, oof_xgb,   seed=11)
cat_t, cat_score     = optimize_thresholds(y, oof_cat,   seed=22)
thresholds, blend_score = optimize_thresholds(y, oof_blend, seed=99)
print(xgb_score, cat_score, blend_score)
print(np.round(thresholds, 6))
```

材料的记录是：**5 折 OOF 融合 QWK 约 0.4288**；如果做**交叉拟合的阈值估计**（在一部分折上选阈值、在另一部分折上评估），分数是 **0.4265**。后者更低，但更诚实——因为「在全部 OOF 上调阈值、再用同一批 OOF 报分」会有一点点乐观偏差，阈值本身也拟合了这批数据。这个差距（约 0.002）说明阈值过拟合不算严重，但确实存在，值得记录。

**一个必须提防的坑：阈值不能在全量 OOF 上随意调。** 为了得到更可信的估计，可以把 OOF 再按折切开：用第 1–4 折的 OOF 选阈值，在第 5 折上评估 QWK，轮换一遍，最后求平均。这就是「交叉拟合阈值」的思想，也是 0.4265 这个数字的来源。

最后生成提交文件。注意测试预测的连续值必须用**同一组** OOF 上选出的阈值来切：

```python
submission = pd.DataFrame({
    'id': test['id'],
    'quality': apply_thresholds(test_blend, thresholds).astype(int),
})
submission.to_csv('submission.csv', index=False)

# 检查预测分布：如果某个等级一个都没有，说明阈值可能有问题
print(submission['quality'].value_counts().sort_index())
```

检查提交分布这一步很实用：如果 3 分和 8 分一个都没预测出来，那模型在稀有等级上的 QWK 贡献就是零，可以考虑是否需要轻微调整阈值间距来「放出」少数极端预测——但这类调整必须小心，很容易变成对排行榜的过拟合。

---

## 三、总结

本文中我们完成了葡萄酒品质评分的建模。我们首先用 EDA 确认了数据没有缺失、品质分布严重偏向 5 分和 6 分、酒精度与品质正相关而挥发性酸度负相关，并发现多个理化指标之间存在物理上的强相关。其次我们围绕化学含义构造了 9 个特征，包括总酸度、结合态二氧化硫、游离硫占比、挥发酸与固定酸之比、分子态二氧化硫近似等，没有做标准化（树模型不需要）。接着我们把评分当回归问题训练，用 XGBoost 回归器（1800 棵树、深度 5、小学习率、类别权重）和 CatBoost 多分类器（概率加权求期望）分别做 5 折交叉验证，按 0.70:0.30 融合，并用多起点 Nelder-Mead 加局部爬山，直接在 OOF 上优化 5 个等级切分阈值。最后得到 OOF QWK 约 0.4288，交叉拟合阈值后约 0.4265，并用同一组阈值切测试集生成提交文件。

这个比赛教给我们最重要的一课是：**评估指标有「结构」时，模型和决策要分开设计**。QWK 关心的是有序等级和分差平方，所以用回归保留顺序、再用阈值切分类别，比直接上分类器更贴合目标；而阈值本身就是一组可优化的参数，必须在 OOF 上调、并用交叉拟合检验，否则很容易把「阈值记住了验证集」误当成模型变强。