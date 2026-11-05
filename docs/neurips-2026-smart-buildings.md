# Neurips 2026 Smart Buildings

## 一、比赛简介

### 1.1 比赛背景
Neurips 2026 Smart Buildings 是一个智能建筑暖通空调（HVAC）控制竞赛。HVAC 就是楼里的空调、供暖、通风系统；它既要让人待着舒服，又要尽量少耗电、少排碳，这三件事天然互相拉扯。比赛要求参赛者写一个控制策略，在一个**从未见过**的测试建筑上，每 5 分钟做一次决策，连续控制 30 天，总共 8,640 步。

训练数据包含 6 个建筑（`building_train_1`、`2`、`4`、`5`、`7`、`8`），每个建筑又放在 5 种气候下：Atlanta（亚特兰大，温和）、Denver（丹佛，寒冷干燥）、Fairbanks（费尔班克斯，极寒）、Phoenix（凤凰城，炎热干燥）、Singapore（新加坡，湿热）。所以一共有 30 个训练场景。每个建筑有 30～85 个房间区（zone，每个区带一个 VAV 末端箱）、3～9 个空气处理机组（AHU）和一个热水装置（plant）。测试建筑是隐藏的，这意味着策略必须能泛化，而不是记住某栋楼的参数。

### 1.2 比赛目标
策略每一步读取传感器，输出设定点和开关命令。按设备来分，观测和动作大致是：

| 设备 | 读什么 | 下什么命令 |
| --- | --- | --- |
| 房间区（VAV） | 空气温度、冷暖设定点、风阀开度、再热阀开度 | 风阀开度、再热阀开度 |
| 空气处理机组（AHU） | 送风温度、静压、流量、室外温度、风机命令 | 送风温度、静压、回风比例、启停 |
| 热水装置（plant） | 热水供水温度、压力、运行状态 | 水温、压力、水泵启停 |
| 全局 | 一天中的时刻、星期、 occupancy（是否有人）、当前和即将到来的舒适模式 | —— |

这里有个容易踩坑的细节：**连续动作的 0 不是「关闭」，而是量程的中间值**，比如半开的阀门。真正的关闭要用离散的启停命令。另外，AHU 只能制冷，制热靠房间区的热水再热盘管，而再热需要风机送风才能把热量送出去——所以「把风机关掉」并不等于省心，它可能让某些房间冷得难受。

### 1.3 评价标准
评分的定义是：先对每个场景算 30 天奖励的平均值，再取绝对值，最后对所有场景求平均，即 `mean over scenarios of abs(mean(reward))`，**越低越好**。奖励本身是一个加权和：舒适度占 0.6，能源成本占 0.2，碳排放占 0.2。因为舒适度权重最大，所以策略不能靠「关掉一切」来省电——那样奖励会变成很大的负数，取绝对值后分数反而很难看。这个「先取绝对值再平均」的设计，意味着我们既要避免让平均奖励变成很大的负数（太难受），也要避免它变成很大的正数（考核方并不奖励无意义的过度舒适）。

---

## 二、解题过程

整场比赛有两条技术路线：一条是**规则改进**，在组织方提供的 PID 控制器上打补丁；另一条是**强化学习**，在规则策略之上学一个残差修正。我们两条都走了一遍，下面按 EDA、数据清洗、训练调参、测试报告的框架来叙述。

### 2.1 数据探索（EDA）
我们首先定位竞赛数据目录，并把仿真环境加载起来。
```python
import os, sys
from pathlib import Path

# Kaggle 上数据可能挂在两个位置之一，挨个试
candidates = [Path('/kaggle/input/competitions/neurips-2026-smart-buildings'),
              Path('/kaggle/input/neurips-2026-smart-buildings')]
DATA_ROOT = next((p for p in candidates if (p / 'smart_buildings').exists()), None)

# 把竞赛包加入 import 路径，后面才能导入官方仿真器
sys.path.insert(0, str(DATA_ROOT))
sys.path.insert(0, str(DATA_ROOT / 'kaggle_evaluation'))
```
接着我们遍历 30 个训练场景，读取每个建筑的元数据。
```python
import numpy as np, yaml
from smart_buildings.evaluate import load_hybrid_environment, create_building_meta_data

rows = []
scenario_dirs = sorted((DATA_ROOT / "smart_buildings/training_scenarios").glob("*/*"))
seen = set()
for scenario in scenario_dirs:
    if not (scenario / "config.gin").exists():
        continue
    # 每个建筑只统计一次，避免 5 个气候重复计数
    if scenario.parent.name in seen:
        continue
    seen.add(scenario.parent.name)

    env = load_hybrid_environment(scenario / "config.gin")
    env.metrics_path = None          # Kaggle 输入目录只读，不能写指标文件
    meta = create_building_meta_data(env, scenario)
    obs, _ = env.reset()             # 拿一帧初始观测
    rows.append((scenario.parent.name, len(meta["zone_action_indices"]),
                 meta["num_floors"], obs.shape[0]))
print(rows)
```
从结果我们可以看到几个关键事实：区数从 30 到 85 不等，楼层数也不同，观测维度随建筑规模变化。也就是说，**不同建筑的输入维度是不一样的**，任何固定输入尺寸的神经网络都不能直接套用，除非做池化或按设备聚合。这也是为什么规则类方法在这个比赛里很有竞争力——它们天然不挑建筑规模。

我们还检查了初始观测是否含非法值：
```python
# 如果初始观测里有 NaN 或 inf，后面的控制会直接崩掉
assert np.isfinite(obs).all(), "Non-finite initial observation"
```
官方仿真器保证了这一点，但我们仍然显式断言，方便在数据被替换时第一时间发现。

环境里还有一个连续动作维度必须满足的恒等式：
```python
v = len(meta["zone_action_indices"])   # 区数
k = len(meta["temp_indices"])          # 温度设定点个数
assert meta["n_continuous_action"] == 3*k + 2 + 2*v
```
这个式子说明动作由三部分组成：每个温度设定点 3 个动作、AHU 的 2 个、每个区 2 个（风阀和再热阀）。理解了动作结构，才不会被「动作向量到底哪一段对应哪个设备」搞糊涂。

### 2.2 数据清洗
这个比赛没有传统意义上的缺失值清洗，因为数据来自仿真器，字段都是完整的。真正需要「清洗」的是**控制逻辑里的边界情况**，我们主要处理三件事。

第一，把观测里的无效数值挡在控制器外面。仿真器偶尔会给出极端值，我们做截断：
```python
# 把观测裁剪到合理区间，避免控制器被异常值带偏
obs = np.clip(obs, -1e6, 1e6)
```

第二，明确动作的语义。前面说过，连续动作 0 是量程中点不是关闭，所以我们在初始化时把「关闭」写成离散命令，而不是把连续量设成 0：
```python
# 关闭 AHU 要用离散的 run 命令，不能用连续动作 0
action_run_off = 0      # 离散：停机
action_run_on = 1       # 离散：运行
```

第三，处理时间相关的模式。仿真器提供「当前舒适模式」和「即将到来的舒适模式」，我们用它来决定是否进入夜间策略：
```python
# 非工作时间：comfort_mode 不再是 occupied
is_off_hours = (comfort_mode_now != COMFORT_OCCUPIED)
```
这一步为后面的「夜间冷却上限」规则做铺垫，也避免了在有人时段误用节能策略导致舒适度崩盘。

### 2.3 模型训练与超参数调优
我们先做规则基线。组织方提供的是 `PIDWaterResetPolicy`，它的行为是：当某个区的温度回到舒适带内，就立刻停掉对应的 AHU；等温度再次越界，再重新启动。问题在于，这样会导致 AHU 每天启停 50～107 次，频繁启停既费电又损设备。

我们的第一个改动叫**冷却运行保持**（cooling-run hold）：如果一台 AHU 是因为制冷而启动的，那就让它在最近一次命令下至少运行 288 步（24 小时）。
```python
COOLING_RUN_HOLD = 288   # 24 小时 × 每小时 12 步 = 288 步

def apply_cooling_hold(ahu, cmd, steps_since_start):
    """因冷却启动的 AHU 至少运行 288 步再允许停机"""
    if ahu.started_for_cooling and steps_since_start < COOLING_RUN_HOLD:
        return KEEP_LAST_COMMAND      # 保持上一次命令
    return cmd                        # 否则按原 PID 逻辑
```
这个改动在炎热气候下效果明显，因为避免了压缩机式的反复启停；在供暖为主的气候下则几乎没影响，因为那些场景的 AHU 本来就不怎么为制冷启动。

第二个改动叫**夜间冷却上限**（night cooling ceiling）：非工作时间，把舒适带的「冷却侧上限」从 eco 上限收紧到白天的舒适上限。两者差 1K。原因是热点建筑如果在夜间慢慢升温，第二天一开始就已经过热，需要花很大力气才能拉回来。
```python
def night_cooling_ceiling(is_off_hours, cooling_setpoint):
    """夜间把冷却设定点压到白天舒适上限，防止建筑过夜升温"""
    if is_off_hours:
        return DAYTIME_COMFORT_CEILING     # 比 eco 上限低 1K
    return cooling_setpoint
```
这个想法来自公开 notebook「Neurips 2026 Smart Buildings Smartpolicy」，我们复现并验证了它。

把两个改动叠加后，排行榜分数从 0.06439（纯 PID 基线）降到 0.05882（6 步保持），再降到 0.05195（288 步保持 + 夜间上限）。可以看到，**保持时间从 6 步拉到 288 步贡献了大部分收益**。

规则的天花板很快到了，于是我们尝试强化学习。做法不是从零学控制，而是学一个**残差**：以手工规则为基底，RL 只输出一个小的修正量。
```python
# 残差 PPO：动作 = 规则动作 + 网络输出的小修正
class ResidualPPO:
    def act(self, obs, base_action):
        delta = self.policy(obs)              # 幅度受限的修正
        return np.clip(base_action + delta, -1.0, 1.0)
```
这样训练稳定得多，因为即使网络一开始输出全是 0，策略也等价于一个还不错的规则控制器。

训练时我们用了 2.0M 到 5.0M 步不等的预算，并换了随机种子。公开排行榜上的变化是：2.0M 步 0.01237、2.5M 步 0.01245、3.0M 步 0.01224、种子 1 的 4.75M 步 0.01216、5.0M 步 0.01199。后来改成「以区温度为目标」的 RL，250k 步就到 0.01132，750k 步到 0.01128，这是材料里的最好成绩。

这里必须诚实地说一个坑：5.0M 步那版权重的部分收益来自**仿真器计费 bug**，RL 自己发现了这个漏洞并加以利用。按真实账单重算，提升是 -10.6% 而不是 -16.9%；在组织方修复（PR #2）后是 -9.0%。即便如此，这仍然是一个真实的提升，但它提醒我们：**排行榜上的改进要先确认不是钻了评估器的空子**。

### 2.4 模型测试与指标报告
我们按比赛定义实现评估函数，先对每个场景求平均奖励，取绝对值，再跨场景平均。
```python
import numpy as np

def competition_score(rewards_per_scenario):
    """rewards_per_scenario: 每个场景 30 天的逐步奖励列表"""
    per_scenario = [abs(np.mean(r)) for r in rewards_per_scenario]
    return float(np.mean(per_scenario))     # 越低越好
```
在留出的building 8 上对比不同策略：
```python
results = {
    'per-zone PI':       0.01420,
    'hand rule (zt+off)':0.01251,
    'RL residual 2.0M':  0.01237,
    'RL residual 5.0M':  0.01199,
    'zone-target RL 750k': 0.01128,
}
for name, score in results.items():
    print(f"{name:22s} {score:.5f}")
```
读法是：纯 PID 最差，手工规则次之，残差 RL 再进一步，而直接把区温度作为学习目标的版本最好。为什么「区温度目标」更好？因为奖励里舒适度占 0.6，直接盯住区温度比盯住一个混合奖励信号更贴近得分结构。

关于检查点选择，材料里还有一个很实用的经验：单看 building 8 会偏好 250k 步的权重，但在 20 栋「生成出来、从未参与训练」的建筑上，750k 步更好。最后提交的是 750k 步的版本，公开分 0.01128。这说明**验证集要足够多样**，否则会在几个几乎一样好的检查点之间选错。

最后生成提交：策略在每个决策步读取观测、输出动作，评测框架负责跑完 8,640 步并计分。
```python
def agent(observation, configuration):
    # 每一步都会被调用：解析观测 -> 规则基底 -> RL 残差 -> 返回动作
    base = pid_policy(observation)
    action = residual_policy.act(observation, base)
    return action
```

---

## 三、总结
本文中我们完成了 NeurIPS 2026 智能建筑控制比赛的完整探索。我们首先通过 EDA 确认了 30 个训练场景覆盖 6 栋建筑 × 5 种气候，建筑规模从 30 到 85 个区不等，并理清了观测与动作的对应关系；其次在数据层面处理了动作语义、异常观测和夜间模式；接着先用「冷却运行保持 288 步 + 夜间冷却上限」把 PID 基线从 0.06439 改进到 0.05195，再用手工规则加残差 PPO 把分数推进到 0.01128；最后按比赛定义实现了「先取绝对值再平均」的评分函数，并讨论了计费 bug 对成绩的影响。

这个比赛给我们最大的启发有两点。第一，**强基线加小修正**往往比端到端从零学更稳，残差 RL 就是典型例子。第二，看到排行榜提升时一定要问一句「这是真的吗」——材料里那 16.9% 的提升有相当一部分来自仿真器漏洞，诚实计费后只剩 9%～10.6%。在真实工程里，这种自我怀疑比多刷 0.001 分更值钱。
