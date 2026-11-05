# MAD Chairs Tournament 2026

## 一、比赛简介

### 1.1 比赛背景
MAD Chairs 是 IEEE SSCI 2027 MAD Chairs Tournament 中的博弈游戏。每轮 6 名玩家从资源集合 {A,B,C,D,E} 或跳过（skip）中选择。选择某资源的玩家若该轮无人与其重复，则获得该轮胜利。本次测试模拟 3 名挑战者玩家对抗 3 名对手玩家，每场比赛包含 6 名玩家共 20 轮。历史记录影响策略决策，需要在重复博弈中动态平衡探索与利用。

### 1.2 比赛目标
参赛者需实现 submission(position, round, history, cache) 函数，返回本轮选择的资源。策略名为 challenger，具体玩家为 challengerA/B/C。系统会用已有的策略 turntaking、equalize、caste、radicaleq、rotate、random 等进行对抗，并用 TrueSkill 输出技能分。

训练阶段无标注数据，参赛者需自行设计博弈策略，history DataFrame 记录每轮历史选择。

### 1.3 评价标准
比赛以策略间的胜率与 TrueSkill 排名为评估依据。材料示意结果文件包含每场比赛各玩家选择及胜率百分比，以及策略统计表。最终提交文件为 skill score，数值越高越好。

## 二、解题过程

### 2.1 数据探索（EDA）
读取比赛日程。

```python
import pandas as pd
schedule = pd.read_csv("/kaggle/input/datasets/chrissantoslang/schedule/schedule.csv")
rounds = 20
resources = ["A","B","C","D","E"]
num_players = 6
```

schedule 指定每场比赛中玩家的策略分配。history DataFrame 包含 Round1..Round20 列。

```python
# 查看历史结构
history.head()
```

历史记录用于策略更新。

### 2.2 数据清洗
历史记录本身已清洗，直接使用。我们维护 cache 字典记录每位玩家的累计胜利。

```python
if cache.get("reqRound") != round:
    cache["reqRound"] = round
    updatePopularity(round, cache, history)
```

popularity 记录每种资源被选频率，wins 记录累计胜利。无缺失处理，数据完整。

### 2.3 模型训练与超参数调优
材料给出若干基准策略。randomN 在第 1 轮随机，后续若前一轮获胜则重复选择，若失利则以 1/n 概率转向未被选资源。

```python
def randomN(n, position, round, history, cache):
    if round == 1:
        return random.choice(resources)
    previous = history[f"Round{round-1}"].tolist()
    if random.random() < 1/n and cache["wins"].loc[position-1, f"Round{round-1}"] == 0:
        options = [b for b in resources if b not in previous]
        return random.choice(options)
    return previous[position-1]
```

radicaleq 为资源均衡策略，资源分配给累计胜利最少的玩家。

```python
def radicaleq(position, round, history, cache):
    if cache.get("reqRound") != round:
        cache["reqRound"] = round
        updatePopularity(round, cache, history)
        sorted_res = [r[0] for r in sorted(cache["popularity"].items(), key=lambda item: item[1])]
        sorted_pos = sorted(range(num_players), key=lambda p: cache["wins"]["total"].tolist()[p])
        cache["radicaleq"] = [s[1] for s in sorted(zip(sorted_pos, (sorted_res + extra)))]
    return cache["radicaleq"][position-1]
```

equalize 在 radicaleq 上加入 30% 衰减的偏离计数，将偏离玩家置于 outgroup，优先保障 ingroup 平衡。

我们可在此基础上设计 challenger 策略，例如在前期探索资源分布，后期根据 popularity 与对手历史模式做逆向选择。

### 2.4 模型测试与指标报告
对每个策略计算胜率。

```python
strategy_name = "My_strategy"
results = []
for match in schedule.itertuples():
    # 模拟 20 轮
    for r in range(1, rounds+1):
        move = submission(position, r, history, cache)
    # 计算胜负
```

统计结果写入 {strategy_name}_results.csv 与 {strategy_name}_stats.csv，最后生成 {strategy_name}_submission.csv 仅含 skill score。

## 三、总结
本文中我们梳理了 MAD Chairs Tournament 2026 的博弈规则与评测流程。我们首先读取赛程与资源列表，确认每场比赛 6 人 20 轮的设置。其次维护历史与缓存实现资源流行度与累计胜利的追踪。接着参考了 randomN、radicaleq、equalize 等基准策略，并以过往轮次的重复与转向行为设计挑战者策略。最后通过模拟对战统计胜率并输出 TrueSkill 技能分。材料表明动态均衡资源的策略在该重复博弈中具优势，未来可加入对手建模与记忆衰减进行进一步改进。
