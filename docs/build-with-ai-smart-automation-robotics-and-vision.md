# Build with AI：智能自动化、机器人与视觉（Build with AI — Smart Automation, Robotics and Vision）

## 一、比赛简介

### 1.1 比赛背景

这场比赛由 GDG Windsor、DataRealm Humanoid Application Lab 和 Mirsee Robotics 联合举办，主题是**把 AI 用到工厂车间的机器人上**。官方平台是一台 **Mirsee MH3 人形机器人**——28 个自由度、轮式底盘、双臂操作，运行在 **Genesis World** 物理仿真环境里（材料中的仿真版本为 Genesis World 1.4.5）。

比赛要解决的是制造业里一个真实的两难：

- **传统 PLC / 有限状态机**（可编程逻辑控制器，工厂里最常见的控制方式）非常可靠、节拍稳定，但一旦遇到没写进程序的意外——零件歪了 12 毫米、表面多了道划痕、过道里滚进来一个零件——它就**直接停机报警**，等人走过去处理，一停就是几十秒；
- **端到端的视觉-语言-动作模型（VLA）**很灵活，看得懂没见过的场景，但它是个「黑盒」：一个坐标幻觉就可能让机械臂撞上夹具，而且工厂 Wi-Fi 一抖，等云端推理的控制回路就卡住了。

所以比赛的核心不是「谁的大模型更强」，而是：**怎样把确定性的安全控制和灵活的视觉语言推理结合起来，既不停线、又不出安全事故**。

### 1.2 比赛目标

参赛者要基于官方的 `mh3_genesis` 代码库，交付一套可运行的车间自动化方案。系统每一拍的输入大致是：

- 机器人状态：关节位置、关节速度、末端位姿、关节力矩；
- 感知输入：头部双目 RGB-D 点云、基准标记（fiducial marker）的识别置信度；
- 环境事件：人手是否进入安全区、传送带/托盘上是否有障碍物、零件是否被遮挡。

系统的输出必须是一个**有界的动作决策**：正常抓取、在安全范围内微调位姿后重抓、安全暂停等待，或者升级给人工。最终还要提交一份 `submission_evaluation_metrics.json`，把仿真中测到的指标写清楚。

### 1.3 评价标准

比赛看的是仿真基准中的一组工程指标（都来自官方的评测脚本）：

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| Task Success Rate（TSR，任务成功率） | 成功完成抓取放置的回合占比 | 越高越好 |
| Safety Violation Rate（安全违规率） | 发生碰撞/越界/伤人风险的回合占比 | **越低越好，理想为 0** |
| MTTR（平均恢复时间） | 出现异常到恢复正常所用的平均秒数 | 越低越好 |
| P95 VLM Latency（VLM 延迟 95 分位） | 视觉语言模型推理耗时的 95% 分位 | 越低越好 |

这四个指标是有张力的：一味求稳（比如遇事就停）会让 MTTR 爆炸；一味求快（让模型直接控制机械臂）会让安全违规率上升。好的方案要同时把它们压住。

---

## 二、解题过程

### 2.1 场景与故障分析（EDA）

这场比赛没有传统的 CSV 数据集，我们的「EDA」是**把车间里会出的故障先列清楚、估出频率**。材料给出的故障模型是：

- 15% 的回合出现**过道障碍卡滞**（corridor obstacle jam）；
- 18% 出现**零件表面划伤/未定位**（scratched bracket）；
- 22% 出现**工厂 Wi-Fi 丢包**；
- 8% 出现 **VLM 坐标幻觉**（输出了错误的偏移量）；
- 6% 出现**人员闯入安全区**。

这五类故障决定了架构必须回答三个问题：**谁能保证安全？谁负责看懂异常？断网了怎么办？** 我们据此把系统分成四层，并规定了一条铁律：**确定性代码拥有 100 Hz 的运动与安全回路，VLM 只能作为异步诊断顾问，在机器人停稳（SAFE_HOLD）后才被调用。**

```python
import numpy as np

def sample_faults(rng):
    """按材料中的故障率，为一个仿真回合采样故障组合"""
    return {
        "obstacle_jam":       rng.random() < 0.15,   # 过道卡滞
        "scratched_bracket":  rng.random() < 0.18,   # 零件划伤/未定位
        "wifi_drop":          rng.random() < 0.22,   # Wi-Fi 丢包
        "vlm_hallucination":  rng.random() < 0.08,   # VLM 输出错误偏移
        "human_intrusion":    rng.random() < 0.06,   # 人员闯入
    }
```

### 2.2 数据清洗：把不可信的输出挡在门外

这里的「清洗」不是洗表格，而是**清洗模型的输出**。VLM 会给出一个 JSON 建议，比如「把抓取点向右移 14 毫米」。这个建议不能直接执行，必须先过一个**确定性闸门**：

1. 置信度低于 0.78 → 拒收；
2. 偏移量的无穷范数 `|Δp|∞` 超过 45 毫米 → 拒收（防止「幻觉式大跳」）；
3. 目标点落在安全工作区之外（安全函数 `h(x) < 0`）→ 拒收。

被拒收不是简单报错，而是触发一次「用本地确定性种子重新问一遍」，再不行才升级给人工。这样既过滤了脏数据，又不会让流程卡死：

```python
MAX_CONFIDENCE_GAP = 0.78          # 低于这个置信度一律不信
MAX_OFFSET_MM      = 45.0          # 单次修正的最大偏移（无穷范数）

def validate_vlm_proposal(proposal, in_safe_region):
    """确定性交接闸门：所有 VLM 输出必须过这一关"""
    if proposal.get("confidence", 0.0) < MAX_CONFIDENCE_GAP:
        return False, "LOW_CONFIDENCE"
    if max(abs(proposal.get("dx_mm", 0.0)),
           abs(proposal.get("dy_mm", 0.0))) > MAX_OFFSET_MM:
        return False, "OFFSET_TOO_LARGE"
    if not in_safe_region:
        return False, "OUTSIDE_SAFE_WORKCELL"
    return True, "ACCEPTED"
```

### 2.3 系统构建与参数调优

**第一层：确定性安全内核（100 Hz）。** 每一拍先检查两个硬约束：末端力是否超过 35 N、人是否进入安全距离。任意一个触发，立刻急停（E-STOP），**任何 AI 输出都不能覆盖它**。如果没有异常且视觉置信度正常，就直接执行抓放；否则进入安全暂停，把现场快照交给下一层：

```python
class DeterministicSafetyController:
    MAX_FORCE_N   = 35.0     # 末端力超过 35N 视为碰撞
    MAX_OFFSET_MM = 25.0     # 允许 VLM 建议的最大修正量

    def step(self, scenario):
        # 1) 硬安全：碰撞力或人员闯入 -> 立刻急停，不问 AI
        if scenario["force_n"] > self.MAX_FORCE_N or scenario["human_in_zone"]:
            return {"mode": "DETERMINISTIC_ESTOP",
                    "action": "IMMEDIATE_SAFE_STOP", "vlm_called": False}

        # 2) 一切正常：按原计划抓取放置
        if scenario["fiducial_conf"] >= 0.80 and not scenario["obstruction_detected"]:
            return {"mode": "DETERMINISTIC_NOMINAL",
                    "action": "EXECUTE_PICK_PLACE", "vlm_called": False}

        # 3) 非致命异常：进入安全暂停，由上层决定是否请教 VLM
        return {"mode": "SAFE_HOLD", "action": "PAUSE_AND_DIAGNOSE", "vlm_called": True}
```

**第二层：行为树（Behaviour Tree）。** 行为树是一种把「顺序执行、失败回退、并行监控」写成树状结构的控制方式，比状态机更容易验证和扩展。我们用 `py_trees` 组织：主序列是「检查安全 → 抓取 → 放置」；任何一个子节点失败，就回退到「诊断并恢复」子树；恢复成功后重新进入主序列：

```python
import py_trees

def build_tree():
    # 主流程：安全自检 -> 抓取 -> 放置
    pick_place = py_trees.composites.Sequence(
        name="PickPlace",
        children=[safety_check(), grasp(), place()])

    # 恢复流程：请求 VLM 诊断 -> 校验建议 -> 执行微调后重抓
    recover = py_trees.composites.Sequence(
        name="DiagnoseAndRecover",
        children=[request_vlm_diagnosis(), validate_proposal(), regrasp()])

    # Fallback：主流程失败就转恢复；恢复也失败则升级给人工
    root = py_trees.composites.Selector(
        name="Root",
        children=[pick_place, recover, escalate_to_operator()])
    return root
```

**第零层：编译期验证。** 行为树跑起来之前，我们还做了一件别人不做的事：用 SMT 求解器 **Z3 在离线阶段形式化证明 4 条安全不变量（INV-1～INV-4）永远不可能被违反**（证明结果全部为 `UNSAT`，即「找不到反例」），并把最终的行为树用 SHA-256 哈希锁定。这样一来，当管理者问「这台机器人会不会在有人进入 0.85 米范围时移动」，答案不再是模型的概率，而是一条可审计的形式化证明。

**第三层：本地边缘 VLM + 150 毫秒看门狗。** 关键的工程决策是 **VLM 跑在本地边缘设备上、不依赖云端**。理由很直接：Wi-Fi 丢包率高达 22%，云端方案一旦断网就只能干等；而本地推理即使在断网时也能在 150 毫秒的预算内返回结果。超时没返回，看门狗就切到「安全暂停 + 升级人工」。

**参数怎么定。** 我们没有梯度可训练，所谓「调参」就是**用消融实验（ablation）确定每个安全阈值**：把置信度阈值、偏移上限、看门狗时长分别放开或收紧，看哪一组能在不增加安全违规的前提下把 MTTR 压低。最终锁定的组合是：置信度 0.78、偏移 45 mm、看门狗 150 ms、控制频率 100 Hz。

### 2.4 仿真测试与指标报告

我们用 **1000 个蒙特卡洛回合**（同样的随机种子 20261114）对比 5 种架构，每种架构跑同样的故障组合：

```python
def run_shopfloor_monte_carlo(n_episodes=1000, seed=20261114):
    rng = np.random.default_rng(seed)
    records = []

    for ep in range(n_episodes):
        faults = sample_faults(rng)
        needs_vlm = faults["obstacle_jam"] or faults["scratched_bracket"]

        for arch in ARCHITECTURES:      # 5 种架构逐一模拟
            completed, safe, mttr = simulate_one(arch, faults, rng)
            records.append({
                "episode": ep, "architecture": arch,
                "task_completed": int(completed),
                "safety_violation": int(not safe),
                "mttr_s": mttr,
                "needs_vlm": needs_vlm,
            })
    return pd.DataFrame(records)
```

再按架构汇总四项指标：

```python
summary = df.groupby("architecture", as_index=False).agg(
    Task_Success_Rate_Pct=("task_completed", lambda s: round(100 * s.mean(), 2)),
    # 只在"需要 VLM"的回合里统计自动恢复率
    Anomaly_Recovery_Rate_Pct=("task_completed",
        lambda s: round(100 * s[df.loc[s.index, "needs_vlm"]].mean(), 2)),
    Safety_Violation_Rate_Pct=("safety_violation", lambda s: round(100 * s.mean(), 2)),
    Mean_Recovery_Time_s=("mttr_s", lambda s: round(s.mean(), 2)),
)
```

结果非常能说明问题：

| 架构 | 安全违规率 | 任务完成 | 平均恢复时间 |
| --- | --- | --- | --- |
| 纯 PLC / 状态机 | 0% | 遇异常即停，等人工 38～65 秒 | 很长 |
| 无约束端到端 VLA | **14.3%** | 灵活但危险 | 60～120 秒 |
| 云端 VLM + 行为树（无本地回退） | 0% | 断网时卡住，**91.8%** | 18～32 秒 |
| 消融：去掉第二层 Z3 闸门 | 出现 **2.1%** 危险冲撞 | 高 | 约 1.9 秒 |
| **AEGIS-BT（完整四层）** | **0%** | 每个回合都完成任务 | 1.38～2.35 秒 |

读法很清楚：

- **PLC 是安全的，但不「智能」**——它遇到没写过的异常只能停线等人；
- **端到端 VLA 很灵活，但没有安全边界**——14.3% 的回合出了安全问题，这在真实工厂是不可接受的；
- **云端方案受制于网络**——22% 的丢包率直接把完成率压到 91.8%；
- **消融实验证明第二层闸门是关键**——去掉它以后，2.1% 的「危险冲撞」会溜过去；
- **完整的四层架构同时做到了 0% 违规和高完成率**，MTTR 稳定在 2 秒左右，比人工处理快一个数量级。

最后把关键指标写进官方要求的 JSON：

```python
import json
from pathlib import Path

submission_metrics = {
    "project": "AEGIS-BT: Autonomous Edge-Guided Industrial System",
    "robot_platform": "Mirsee MH3 28-DOF Wheeled Bimanual Humanoid (Genesis World)",
    "simulation_metrics": {
        "vlm_handoffs_triggered": 3,
        "vlm_proposals_accepted": 2,
        "adversarial_hallucinations_blocked": 1,   # 被闸门拦下的错误建议
        "safety_preemption_latency_ms": 0.85,      # 从发现危险到执行急停的时间
        "safety_violations_executed": 0,
    },
    "monte_carlo_1000ep_benchmark": summary.to_dict(orient="records"),
}
Path("submission_evaluation_metrics.json").write_text(
    json.dumps(submission_metrics, indent=2), encoding="utf-8")
```

---

## 三、总结

本文中我们完成了 Build with AI 车间机器人方案的完整设计与评测：首先我们分析了五类真实车间故障及其发生频率，明确了「确定性代码管安全、VLM 只管诊断」的分层原则；其次我们把「数据清洗」落实为对 VLM 输出的确定性闸门——置信度、偏移上限、安全区三重校验；接着我们构建了四层 AEGIS-BT 架构：Z3 验证的行为树编译、100 Hz 确定性安全内核、结构化交接闸门、本地边缘 VLM 与 150 ms 看门狗，并用消融实验确定各项安全阈值；最后我们用 1000 回合蒙特卡洛在 5 种架构上做了对比，完整方案取得 0% 安全违规、每回合都完成任务、平均恢复时间 1.38～2.35 秒的结果，而端到端 VLA 的违规率高达 14.3%，云端方案在断网时完成率只有 91.8%。

这个比赛给我们最大的启发是：**在工业场景里，「更聪明的模型」不等于「更好的系统」**。真正的关键是给 AI 划一条它越不过去的安全边界，并且让它在地断网时也能工作——把 VLM 关在 100 Hz 安全回路之外，反而让整个系统既安全又能干活。