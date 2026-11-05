# MARL-1D 竞技场 AI（MARL-1D Arena AI）

## 一、比赛简介

### 1.1 比赛背景

MARL-1D Arena AI 是游戏 AI 与多智能体强化学习（Multi-Agent Reinforcement Learning）领域的一场趣味比赛。比赛把「两名斗士在一条一维战术通道上互相对战」这个场景抽象成一个 1D 竞技场：通道一共 7 格，双方在这条线上移动、进攻和防守。比赛要求我们解决一个非常基础的问题——**价值函数估计（Value Function Estimation）**。

所谓价值函数估计，就是回答「给定战斗中间的一瞬间，谁能赢」这个问题的概率版。比赛把战斗中每一个「冻结的时刻」（mid-battle snapshot）作为一条样本：它记录了双方当前的血量、耐力、姿态（poise）、位置等状态，我们要预测的是 **Player 0 获胜的概率**。这个问题既是传统监督学习（预测一个 0~1 的数），也是强化学习中学会评估局势的第一步。

### 1.2 比赛目标

比赛提供三个文件：

- `train.csv`：500 条已标注的战斗快照，每条 70+ 个特征，目标是 `win_probability`（Player 0 的胜率，由引擎从该局面做多次前向模拟得到的经验胜率）；
- `test.csv`：700 条未标注的快照（65+ 列），要预测 `win_probability`；
- `sample_submission.csv`：提交模板，只有 `id` 和 `win_probability` 两列。

特征分成了几组系统：**血量经济**（`player_hp_ratio`、`opp_hp_ratio`、`hp_advantage`）、**耐力/能量**（`en_advantage`、`has_rest`，动作要消耗能量，能量耗尽只能恢复）、**姿态/破防**（`poise_advantage`、`opp_near_break`，重击消耗姿态，姿态归零会进入硬直）、**空间走位**（`distance`、`center_control`、`at_melee_range`）。每行内部还带 `player_*`（Player 0）和 `opp_*`（对手）两套对称字段。

### 1.3 评价标准

比赛用 **AUC-ROC**（曲线下面积）来评价，越高越好。它的直观含义是：随机抽一个 Player 0 赢的局面和一个 Player 0 输的局面，模型给「赢的场景」打出更高分数的概率。AUC = 0.5 相当于瞎猜，AUC 越接近 1 越接近完美排序。因为比赛只有 500 条训练数据，官方 starter 给的粗略基准是 **0.64** 左右，好的方案能冲到 0.80 以上。

---

## 二、解题过程

我们的思路是：**先做 EDA 理解四大游戏系统和目标的关系，然后做领域特征工程，用 5 折分层交叉验证比较「无学习启发式 → 逻辑回归 → 梯度提升树」三个基线，最后提交并分析特征重要性。** 下面按四步走。

### 2.1 数据探索（EDA）

我们首先加载数据并检查规模、缺失值和目标分布：

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"Train: {train_df.shape} | Test: {test_df.shape}")
# 训练集 500 行 x 70+ 列，测试集 700 行 x 65+ 列
print("Missing train:", train_df.isnull().sum().sum())  # 0，数据很干净
```

然后重点看目标 `win_probability` 的分布。它是从该局面做多次前向模拟得到的经验胜率，所以接近 0 和 1 的是「一边倒」的局面，中间 0.4～0.6 的是势均力敌的僵局：

```python
win_probs = train_df["win_probability"]
print("Mean:", win_probs.mean(), "| Std:", win_probs.std())
print("Min/Max:", win_probs.min(), "/", win_probs.max())

# 统计有多少局面是寡众悬殊、多少是五五开的僵局
contested = (win_probs >= 0.4) & (win_probs <= 0.6)
print(f"Contested (40-60%): {contested.sum()}/{len(train_df)} ({contested.mean():.1%})")
```

接着我们看哪些原始特征和目标的相关性最高，用「相关最强的 7 个 + 相关最弱（最负）的 7 个」画柱状图：

```python
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
# 去掉 id 和模拟统计列，只看真实状态特征
numeric_cols = [c for c in numeric_cols if c not in
                ["id", "state_seed", "win_probability", "rollout_count",
                 "win_count", "loss_count", "draw_count", "mean_final_hp_delta"]]

corr = train_df[numeric_cols].corrwith(train_df["win_probability"]).sort_values()
print("Top 5 positive:", corr.tail(5)[::-1])
print("Top 5 negative:", corr.head(5))
```

相关系数大致告诉我们两件事：**血量类的特征（`hp_advantage`、`player_hp_ratio`）对胜率影响最大**，这是符合直觉的——血多的占优；耐力、姿态、距离的单独相关性弱一些，但它们和其他因素**交互**时很关键（比如「低血量 + 近身 + 对手没耐力」才是真正危险）。这提示我们后面要做交互特征，而不是只用单列。

### 2.2 数据清洗

这份数据几乎没有缺失值，清洗工作主要是两件：一是**把模型不需要的模拟统计列和 id 排除掉**；二是**用领域知识构造交互特征**。关键的交互特征如下：

```python
def engineer_features(df):
    data = df.copy()

    # 1. 综合资源优势：血量最重要（权重 2.5），耐力其次（1.2），姿态再其次（0.8）
    data["net_resource_advantage"] = (
        data["hp_advantage"] * 2.5 +
        data["en_advantage"] * 1.2 +
        data["poise_advantage"] * 0.8
    )

    # 2. 持续作战能力：血量占比 x 耐力占比，衡量还能撑几个回合
    data["player_staying_power"] = data["player_hp_ratio"] * data["player_en_ratio"]
    data["opp_staying_power"] = data["opp_hp_ratio"] * data["opp_en_ratio"]
    data["staying_power_delta"] = data["player_staying_power"] - data["opp_staying_power"]

    # 3. 致命威胁：血量低于 25 且在近身距离（<=2格）内才真的致命
    data["lethal_threat_opp"] = ((data["opp_hp"] < 25) & (data["distance"] <= 2)).astype(float)
    data["lethal_threat_self"] = ((data["player_hp"] < 25) & (data["distance"] <= 2)).astype(float)
    data["lethal_pressure_delta"] = data["lethal_threat_opp"] - data["lethal_threat_self"]

    # 4. 墙角陷阱：自己贴墙且对手控制中路，退无可退
    data["corner_trap_danger"] = (data["near_wall_self"] * data["center_control"]).astype(float)

    return data

train_feat = engineer_features(train_df)
test_feat = engineer_features(test_df)
```

这些新特征体现了战斗系统的核心逻辑：单纯的「血量差」没有炸毁整个局面，但「血少 + 近身 + 对手能耗尽 + 被逼到墙角」才是真正的死局。构造完成后，我们从所有列里挑选出 36 个建模特征，并把缺失值补 0：

```python
feature_cols = ["distance", "hp_advantage", "en_advantage", "poise_advantage",
                "net_resource_advantage", "staying_power_delta",
                "lethal_threat_opp", "lethal_threat_self", "corner_trap_danger", ...]

X_train = train_feat[feature_cols].fillna(0)
X_test = test_feat[feature_cols].fillna(0)
```

### 2.3 模型训练与超参数调优

我们依次建立三个基线，用 **5 折分层交叉验证**打分，保证每一折里「一边倒」和「五五开」的局面占比都和整体一致。

**基线 1：纯领域启发式（不用机器学习）。** 只把血量差、耐力差、姿态差做一个加权和，再过一次 sigmoid 变成概率，系数都是手工定的：

```python
def heuristic_predict(df):
    """纯粹的领域启发式：血多、耐足的占优，不需要训练"""
    logit = (df["hp_advantage"] * 2.2 +
             df["en_advantage"] * 1.0 +
             df["poise_advantage"] * 0.7)
    return 1.0 / (1.0 + np.exp(-logit))   # sigmoid 压缩到 0~1

heuristic_auc = roc_auc_score(y_binary, heuristic_predict(train_feat))
print("Baseline 1 (Heuristic) AUC:", heuristic_auc)   # 约 0.64
```

**基线 2：逻辑回归 + 标准化。** 用 5 折交叉验证一套完整的 `StandardScaler → LogisticRegression` 流程，把每个特征的贡献交给模型去学：

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 先把目标转成二分类：>=0.5 视为「Player 0 赢」
y_binary = (train_feat["win_probability"] >= 0.5).astype(int)

scaler = StandardScaler()                # 对数值特征做标准化，利于逻辑回归收敛
X_scaled = scaler.fit_transform(X_train)

oof = np.zeros(len(train_feat))          # 存放折外预测，避免自评
for train_idx, val_idx in kf.split(X_scaled, y_binary):
    clf = LogisticRegression(C=0.2, max_iter=1000, random_state=42)
    clf.fit(X_scaled[train_idx], y_binary.iloc[train_idx])
    oof[val_idx] = clf.predict_proba(X_scaled[val_idx])[:, 1]

lr_auc = roc_auc_score(y_binary, oof)
print("Baseline 2 (Logistic Regression) OOF AUC:", lr_auc)   # 约 0.72
```

**基线 3：梯度提升树（HistGradientBoosting）。** 树模型天然能捕捉「低血量 + 近身」这种阈值型交互，正适合战斗规则。它的超参数我们验证后用了：树最大迭代 60 轮、学习率 0.05、最大深度 3、叶子最少样本 20、L2 正则 2.0：

```python
oof_gbdt = np.zeros(len(train_feat))
for train_idx, val_idx in kf.split(X_train, y_binary):
    gbdt = HistGradientBoostingClassifier(
        max_iter=60, learning_rate=0.05, max_depth=3,
        min_samples_leaf=20, l2_regularization=2.0,
        random_state=42)                # 小学习率 + 小深度，降低过拟合
    gbdt.fit(X_train.iloc[train_idx], y_binary.iloc[train_idx])
    oof_gbdt[val_idx] = gbdt.predict_proba(X_train.iloc[val_idx])[:, 1]

gbdt_auc = roc_auc_score(y_binary, oof_gbdt)
print("Baseline 3 (GBDT) OOF AUC:", gbdt_auc)   # 约 0.76
```

我们特意没有用 `train_test_split` 简单切一刀，而是用 5 折交叉验证，因为只有 500 条样本，一次切分的结果方差太大，交叉验证的结论更可靠。调参上我们认准一个原则：**500 条数据经不起复杂模型**，所以深度压到 3、加 L2 正则，宁可欠拟合也不愿过拟合。

### 2.4 模型测试与指标报告

三个基线的横向对比非常清晰：

| 模型 | 5 折 OOF AUC |
| --- | --- |
| Baseline 1：领域启发式（手工加权 + sigmoid） | 约 0.64 |
| Baseline 2：逻辑回归（标准化 + C=0.2） | 约 0.72 |
| Baseline 3：梯度提升树（HistGradientBoosting） | 约 0.76 |

启发式能到 0.64 说明领域直觉的方向是对的；逻辑回归把它提升到 0.72，说明模型能自动学出更准的权重；梯度提升树再提升到 0.76，印证了**交互特征和阈值型规律**在这种战斗场景里确实存在。

我们用 permutation importance（置换重要性：随机打乱某个特征，看模型准确率掉多少）找出对 GBDT 最重要的特征：

```python
from sklearn.inspection import permutation_importance

gbdt_full = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.05, random_state=42)
gbdt_full.fit(X_train, y_binary)

imp = permutation_importance(gbdt_full, X_train, y_binary,
                             n_repeats=10, random_state=42)
top = pd.Series(imp.importances_mean, index=X_train.columns).sort_values(ascending=False).head(10)
print(top)
```

结果里，`distance`、`hp_advantage`、`net_resource_advantage`、`lethal_threat_*` 这类特征排在最前面，说明**血量、距离和致死威胁是决定胜率的核心**，和 EDA 阶段的判断一致。

生成提交时，我们把 5 折各模型在测试集上的预测取平均，并用 `clip(0, 1)` 保证概率落在合法区间：

```python
# test_preds_gbdt 是 5 折模型对测试集预测的平均
submission = pd.DataFrame({
    "id": test_df["id"],
    "win_probability": np.clip(test_preds_gbdt, 0.001, 0.999)  # 避免 0/1 极端值
})
submission.to_csv("submission.csv", index=False)
```

材料还提示了进一步提升的方向：`marl1d.Engine` 是官方的**确定性模拟引擎**，可以从不明确的局面做蒙特卡洛前向推演（rollout），或者干脆**生成更多合成训练数据**——把 500 条扩到 5000 条以上，再配合 GBDT 就有机会冲上 0.80 甚至 0.85。

---

## 三、总结

本文中我们完成了 MARL-1D Arena AI 比赛的 starter 全流程。我们首先用 EDA 确认了 500 条训练样本、目标 `win_probability` 的经验胜率属性，以及血量类特征与胜率相关性最高。其次我们做了领域特征工程，构造了综合资源优势、持续作战能力、致死威胁和墙角陷阱四类交互特征。接着我们用 5 折分层交叉验证依次比较了领域启发式、逻辑回归和梯度提升树，AUC 从约 0.64 一路升到约 0.76。最后我们用置换重要性确认了核心特征是血量、距离和致死威胁，并生成了 `submission.csv`。

这个比赛给我们的最大启发是：**小样本问题里，领域知识比模型复杂度更重要**。启发式原本只有 0.64，但把「低血量 + 近身距离 + 对手被逼墙角」这些规则变成特征后，简单的树模型立刻就能跨过 0.76；而想要更进一步，就得跳出静态表格，用官方提供的模拟引擎生成更多数据——这也正是强化学习思路在监督学习场景里的漂亮应用。