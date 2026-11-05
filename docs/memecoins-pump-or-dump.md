# Memecoins: Pump or Dump

## 一、比赛简介

### 1.1 比赛背景

Memecoins: Pump or Dump 是 Kaggle 上一个模拟 Pump.fun 链上 meme 币交易的比赛。Pump.fun 发行于 Solana 链上，每个 token 会先经过“bonding curve”曲线交易阶段，再迁移到 PumpSwap 池子。比赛要求我们对每个 token 决定是否“买入”，并设置一个止盈倍数（take-profit，tp），从而最大化总收益。

比赛的特殊之处在于它不是简单的分类问题，而是带门槛的决策问题：买错了要亏损（考虑 0.3 的往返交易成本），不买得 0，买对并触发止盈则赚 tp-1。

### 1.2 比赛目标

输入为 train.csv / test.csv（token 列表）、pumpfun_token_creation.csv（发币记录）、pfamm_migrations.csv（迁移记录）以及每天的交易明细（trades day 分组）。测试提交对每个 token 输出一个止盈倍数，tp=0 表示不买。

train.csv 关键列：mint（币地址）、mt（迁移时间）、max_x（小时内最大倍数）、final_x（小时末倍数）、finalx（旧命名）。

### 1.3 评价标准

评分规则是总分单位 PnL，TP 为 0 得 0 分；买入某 token 后，若小时内的最大倍数（sustained max_x）达到 tp，则得 tp-1-0.3；否则持有到小时末得 final_x-1-0.3，地板是 -1。总 PnL 越高越好。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取 train/test 与创建记录，统计时间跨度与 hit3 比例：

```python
import pandas as pd
train = pd.read_csv(f'{DATA}/train.csv')
test  = pd.read_csv(f'{DATA}/test.csv')
cre   = pd.read_csv(f'{DATA}/pumpfun_token_creation.csv',
                    usecols=['mint','created_ts','dev_balance','bundled_buys','bundle_size'])
print(f'train {len(train)} tokens | test {len(test)} tokens')
```

结果显示训练集覆盖数十天，测试集在某几天。接着我们发现训练集里 max_x >= 3 的比例通常只有 5%~10% 左右，说明“涨到 3 倍”本身是少数事件，选择很重要。

### 2.2 数据清洗

我们按迁移时间换算成 UTC 天，便于逐日对齐交易磁带；创建记录有重复 mint 时要按 created_ts 排序去重；旧版本文件用 finalx 命名，统一改成 final_x：

```python
train = train.rename(columns={'finalx':'final_x'})
for _d in (train, test):
    _d['day'] = pd.to_datetime(_d.mt, unit='s').dt.strftime('%Y-%m-%d')
cre = cre.drop_duplicates('mint')
```

为了防止未来函数的泄漏，我们只用迁移前后的短窗口特征（ts_rel < 60），消除“看到未来”的风险。

### 2.3 模型训练与超参数调优

我们构建六个低成本特征：迁移前 60 秒内执行的 curve 交易数 n_curve、失败 curve 交易 n_failed、迁移后 60 秒 pool 交易 n_pool、创建到迁移的时间差 age_s、开发者余额 dev_balance、打包买入 bundled_buys。随后用 LightGBM 拟合“是否达到 5x”的二分类目标。拟合后按训练日顺序做 walk-forward 验证，选出阈值。

```python
import lightgbm as lgb
model = lgb.LGBMClassifier(objective='binary')
model.fit(X_train, y5x, categorical_feature=categoricals)
```

调参时我们固定超参，重点是验证策略不泄漏：用前几天的数据训练、后几天的数据验证，跑通后才确定入库。

### 2.4 模型测试与指标报告

我们用 walk-forward 模拟真实交易：每天先用截止到该天的数据训练，给出分数并过滤线，再与当天实际的 max_x/final_x 计算 PnL。评分计算公式：

```python
def pnl(tp, max_x, final_x):
    tp = np.asarray(tp, float)
    hit  = (max_x >= tp) & (tp > 0)
    held = np.maximum(final_x - 1 - 0.3, -1.0)
    return np.where(tp <= 0, 0.0, np.where(hit, tp - 1 - 0.3, held))
```

我们最终采用 tp=5（或 8）的策略，仅对分数最高的前几个 token 持仓，7 天交叉检验可取得正的总 PnL。

## 三、总结

本文中我们完成了 Memecoins: Pump or Dump 的完整流程。我们首先通过 EDA 发现命中 3 倍以上的 token 比例低，其次清洗了重复 mint 和旧列名并把时间统一到 UTC 天，接着设计六类“迁移前后窗口内”的低泄漏特征并用 LightGBM 拟合止盈命中目标，最后用 walk-forward 回测得到正 PnL 并提交。本比赛证明动态决策 + 收益率设定比单纯分类更贴近实战，防泄漏的设计是制胜关键。