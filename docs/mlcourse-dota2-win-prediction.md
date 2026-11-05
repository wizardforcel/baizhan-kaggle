# ML Course Dota 2 Win Prediction

## 一、比赛简介

### 1.1 比赛背景

Dota 2 是竞技 MOBA 游戏，每局比赛由两支 5 人队伍对战，最终胜负取决于每名玩家的经济、经验、击杀和团队配合。Kaggle 上 mlcourse-dota2-win-prediction 比赛提供了在比赛进行时点采集的每名玩家状态特征，要求模型预测 Radiant 队是否获胜。比赛源自 ML 入门课程的实践案例，强调特征工程和集成学习的思路。

### 1.2 比赛目标

给定训练集的匹配特征和胜负标签，预测测试集每局比赛 Radiant 获胜的概率。训练样本包含约 246 列原始特征，来源于 10 名玩家各自的 24 余项统计（击杀、死亡、助攻、金钱、经验、等级、视野等）加上比赛上下文信息。目标列 `radiant_win` 为二分类布尔值。

### 1.3 评价标准

比赛使用 ROC AUC 作为评价指标，值越高越好。AUC 不依赖阈值，能够反映模型区分胜负的能力，适合不平衡的比赛结果。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先检查数据文件：

```python
import pandas as pd, os
train_raw = pd.read_csv(os.path.join(DATA_DIR, 'train_features.csv'))
targets   = pd.read_csv(os.path.join(DATA_DIR, 'train_targets.csv'))
print('训练特征形状:', train_raw.shape)   # 约 (样本数, 246)
print('目标类分布:')
print(targets['radiant_win'].value_counts(normalize=True))
```

样例中 Radiant 胜率接近 0.5，类别大致平衡。我们接着查看特征列名的规律：每个玩家对应 `player_<角色>_level`, `player_<角色>_gold` 等字段，前 5 名属于 Radiant，后 5 名属于 Dire。

### 2.2 数据清洗

原始列名重复、冗长，我们统一重命名并做轻量缺失处理：

```python
# 把列名里无关的符号替换为空格，便于后续处理
train_raw.columns = [c.replace('-', '_') for c in train_raw.columns]
train_raw = train_raw.fillna(-1)   # Dota 数据通常没有缺失，用 -1 填充
```

我们还把特征按队伍拆分：Radiant 列和 Dire 列分别聚合，防止标签泄漏。随后通过 `match_id_hash` 把特征与目标合并：

```python
train = train_raw.merge(targets, on='match_id_hash', how='left')
```

### 2.3 模型训练与超参数调优

我们首先做团队级特征工程：对每支队伍的 5 名玩家求和、均值，再计算 Radiant 与 Dire 的差值和比值。

```python
# 假设列名以 'radiant_' 开头或后缀区分对手
def team_agg(df, prefix):
    cols = [c for c in df.columns if prefix in c]
    team_sum = df[cols].sum(axis=1)
    team_mean = df[cols].mean(axis=1)
    return pd.concat([team_sum, team_mean], axis=1)

rad_feat = team_agg(train, 'radiant')
dire_feat = team_agg(train, 'dire')
diff = rad_feat.sub(dire_feat.values, axis=0)
diff.columns = [c + '_diff' for c in diff.columns]
```

进而构造领域复合特征：金钱优势、击杀优势、经验优势、视野优势等。

```python
train['gold_adv'] = (train[[f'radiant_player{i}_gold' for i in range(5)]].sum(axis=1) -
                     train[[f'dire_player{i}_gold' for i in range(5)]].sum(axis=1))
train['kill_adv'] = (train[[f'radiant_player{i}_kills' for i in range(5)]].sum(axis=1) -
                     train[[f'dire_player{i}_kills' for i in range(5)]].sum(axis=1))
```

模型采用梯度提升集成：LightGBM、XGBoost、CatBoost 各自做 5 折分层交叉验证，early stopping。

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

params = {'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':31}
auc_scores = []
for tr_idx, val_idx in skf.split(X, y):
    tr_data = lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx])
    val_data = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx], reference=tr_data)
    model = lgb.train(params, tr_data, valid_sets=[val_data],
                      num_boost_round=1000, callbacks=[lgb.log_evaluation(0)])
    pred = model.predict(X.iloc[val_idx])
    auc_scores.append(roc_auc_score(y.iloc[val_idx], pred))
print('CV AUC:', sum(auc_scores)/len(auc_scores))
```

超参数调优我们主要围绕学习率、树深/叶数和早停轮数进行小范围网格搜索，材料中 LightGBM 最终 CV AUC 约 0.82876，XGBoost 0.82203，CatBoost 0.82284。

我们最后对三模型输出概率取均值做简单平均集成，最终 CV AUC 提升至约 0.83039，并在线上榜达到 0.84385。

### 2.4 模型测试与指标报告

在验证集上我们报告 AUC 和准确率：

```python
from sklearn.metrics import accuracy_score
y_pred_prob = ensemble.predict_proba(X_val)[:,1]
y_pred = (y_pred_prob > 0.5).astype(int)
print('AUC:', roc_auc_score(y_val, y_pred_prob))
print('Accuracy:', accuracy_score(y_val, y_pred))
```

结果显示 AUC 在 0.82~0.83 区间。最后用全量训练数据重新训练集成模型，预测测试集概率并按样例格式输出：

```python
test_pred = ensemble.predict_proba(test_X)[:,1]
submission = pd.DataFrame({'match_id_hash': test_ids,
                          'radiant_win_prob': test_pred})
submission.to_csv('submission.csv', index=False)
```

提交文件包含 `match_id_hash` 和 `radiant_win_prob` 两列，概率范围在 0~1 之间。

---

## 三、总结

本文中我们完成了 Dota 2 胜负预测的完整流程。我们首先做了 EDA，确认特征来自每名玩家的状态，目标大致平衡。其次做了团队聚合和差值特征工程，创造了更具解释性的比赛优势指标。接着用 LightGBM、XGBoost、CatBoost 三模型做 5 折交叉验证并集成，验证 AUC 约 0.83。最后生成符合格式的概率提交文件。

这个比赛给我们的最大启发是：**在竞技游戏数据里，手工程造的"团队对比特征"往往比原始单个玩家特征更有效**，只要把模型换成强力的梯度提升集成并做好交叉验证，就能稳定提升 AUC。