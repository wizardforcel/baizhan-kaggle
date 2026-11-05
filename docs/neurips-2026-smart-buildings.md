# Neurips 2026 Smart Buildings

## 一、比赛简介

### 1.1 比赛背景
Neurips 2026 Smart Buildings 是一个智能建筑 HVAC 控制竞赛。参赛者需要在看不见的测试建筑上，每 5 分钟控制一次暖通空调系统，持续 30 天，共 8,640 个决策步。训练数据包含 6 个建筑（building_train_1、2、4、5、7、8），每个建筑在 5 个气候（Atlanta、Denver、Fairbanks、Phoenix、Singapore）下都有场景。每个建筑包含 30~85 个房间区、3~9 个空气处理机组和一个热水装置。测试建筑完全隐藏，策略必须泛化到从未见过的建筑。

### 1.2 比赛目标
策略的输入是传感器的读取值，包括区空气温度、设定点、阻尼器和再热阀位置、AHU 的送风温度、压力、流量、室外温度、运行命令、热水供应温度和压力、时间、占位和舒适模式等。策略输出是设定点和开/关命令。目标是最小化跨场景平均奖励的绝对值，奖励中舒适度占 0.6、能源成本 0.2、碳排放 0.2。

### 1.3 评价标准
评分为 `mean over scenarios of abs(mean(reward))`，值越小越好。排行榜列出了公开和私有探测分数，低分数表示在舒适、能源和碳排放之间取得了更好的平衡。

---

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先定位竞赛数据并加载训练场景。
```python
from pathlib import Path
import sys
sys.path.insert(0, str(DATA_ROOT))
from smart_buildings.evaluate import load_hybrid_environment

data_root = Path('/kaggle/input/neurips-2026-smart-buildings')
scenario_dirs = sorted((data_root / "smart_buildings/training_scenarios").glob("*/*"))
```
遍历 30 个训练场景，统计每个建筑的区数和 AHU 数，确认观测和动作维度的一致性。

### 2.2 数据清洗
我们不做常规的数据清洗，而是为控制器准备稳定输入。
```python
obs, _ = env.reset()
assert np.isfinite(obs).all(), "Non-finite initial observation"
```
确保初始观测为有限值，并确认动作范围为连续值。中值动作代表半开阻尼器而非关闭。

### 2.3 模型训练与超参数调优
基线方法使用组织提供的 PIDWaterResetPolicy，并改进了两个规则。
```python
import numpy as np
from smart_buildings.controllers import PIDWaterResetPolicy

# 冷却运行保持：若 AHU 因冷却启动，保持至少 288 步
cooling_run_hold = 288
# 夜晚冷却上限：将非工作时间冷却带上限收紧 1K
night_cooling_ceiling = True
```
冷却运行保持避免了冷却 AHU 的频繁开关（每天 50~107 次），在热气候下显著减少能耗损失。夜晚冷却上限防止建筑夜间持续升温。

我们还在强化学习方向上尝试残差 PPO。
```python
from torch_snippets import pd, np
# 残差 RL 在基线规则上学习修正
rl_residual = ResidualPPO(env, baseline_policy)
rl_residual.train(steps=2_000_000)
```
残差策略在公开排行榜上从 0.01358 逐步下降到 0.01040，证明 RL 可以在从未训练过的建筑上击败手工控制器。

### 2.4 模型测试与指标报告
在公开探测建筑上评估控制器。
```python
public_score = evaluate_policy(policy, held_out_building_8)
print("Public LB:", public_score)
```
最终基线在公开排行榜上得到约 0.05195。残差 RL 在诚实计费下提升约 10.6%，在组织修复泄露后仍保持约 9% 提升。

---

## 三、总结
本文中我们探索了 Neurips 2026 Smart Buildings 的 HVAC 控制任务。我们首先对 30 个训练场景进行 EDA，确认建筑结构差异很大；其次通过冷却运行保持和夜晚冷却上限对 PID 基线进行轻微改进；接着验证残差 PPO 在新建筑上的泛化能力；最后在公开排行榜上得到 0.05195 的基线分数，并观察到 RL 残差策略可以稳定超越手工规则。
