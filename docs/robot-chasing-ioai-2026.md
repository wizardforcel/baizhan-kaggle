# Robot Chasing IOAI 2026

## 一、比赛简介

### 1.1 比赛背景
IOAI 2026 的 Robot Chasing 赛道是一个网格世界决策任务。机器人拥有 8x8 的格子地图观测，地图上分布着不同颜色的钥匙、球和箱子，机器人需要根据文字任务指令（例如“pick up the red key”或“put the blue ball”）完成拾取与放置。观测数据以 JSON 形式给出，测试阶段还包含 `pretrain`、`pretest` 等额外划分，但本方案只使用 `train`。

### 1.2 比赛目标
输入为每条观测：`image`（8x8x2 的整数矩阵，含义是物体 ID 和颜色 ID）、`robot_id`、`direction`（机器人朝向）、`mission`（任务文字）以及 `carrying`（当前搬运的物体）。输出为 6 类动作：`up`、`down`、`left`、`right`、`pick up`、`drop`。每帧的标注就是一个动作。

### 1.3 评价标准
指标是动作预测的准确率，即预测正确的动作数占总帧数的比例。材料中给出的最小基线（总是预测 DROP，动作 5）的公榜分数约为 0.03638，作为对照的弱基线。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先用可视化脚本检查了数据集的单帧结构。

```python
import json
from pathlib import Path
import numpy as np

train = json.loads(Path('data/train/observations.json').read_text())
labels = np.asarray(json.loads(Path('data/train/labels.json').read_text()), dtype=np.int8)
print(len(train), len(labels))
```

图像矩阵的两个通道分别表示物体 ID 和颜色 ID，物体 ID 中 `key=5`、`ball=6`、`box=7`，机器人用 `10` 表示。我们统计了每个 `robot_id` 的样本量，发现 6 个机器人各约 10000 条带标签数据。

我们还检查了任务的类型分布。

```python
def parse_mission(text):
    if text.startswith(('put the','place the','move the')):
        task = 2
    elif text.startswith(('pick up','take the','grab the','collect the')):
        task = 1
    else:
        task = 0
    return task
```

任务被归为 0 导航、1 拾取、2 放置三类，这决定了后文的阶段划分。

### 2.2 数据清洗
我们从观测里提取机器人和任务实体位置。

```python
OBJECT = {'key':5,'ball':6,'box':7}
COLOUR = {'red':0,'green':1,'blue':2,'purple':3,'yellow':4,'grey':5}

def find_cell(grid, entity):
    cells = np.argwhere((grid[:,:,0]==entity[0]) & (grid[:,:,1]==entity[1]))
    return tuple(map(int, cells[0])) if len(cells) else (-1,-1)
```

同时把机器人朝向解析成方向向量，方便前向格子计算。

```python
DIRECTIONS = ((-1,0),(1,0),(0,-1),(0,1))
def inside(cell):
    return 0 <= cell[0] < 8 and 0 <= cell[1] < 8
```

我们构建了可通行判断，把墙（物体 ID 2）和不可穿越的格子排除。

```python
def walkable(grid, cell):
    return inside(cell) and grid[cell[0],cell[1],0] in (1,10)
```

### 2.3 模型训练与超参数调优
核心步骤是用 BFS 计算到目标的距离地图，再把“执行每个动作后的距离”作为特征。

```python
from collections import deque
INF = 99

def distance_map(grid, goals):
    distances = np.full((8,8), INF, dtype=np.int16)
    queue = deque()
    for goal in goals:
        if walkable(grid, goal) and distances[goal]==INF:
            distances[goal]=0; queue.append(goal)
    while queue:
        cell = queue.popleft()
        for nxt in neighbours(cell):
            if walkable(grid, nxt) and distances[nxt]==INF:
                distances[nxt]=distances[cell]+1
                queue.append(nxt)
    return distances
```

我们把机器人两侧两步内的物体、距离、是否前进能缩短距离等 7 个方向相关特征都放进特征行。

```python
def feature_row(row):
    f = {'robot_id':int(row['robot_id']), 'direction':int(row['direction']), 'task':task, 'phase':phase}
    f['distance_now'] = distance_now
    for action,(dr,dc) in enumerate(DIRECTIONS):
        one = (robot_cell[0]+dr, robot_cell[1]+dc)
        can_move = walkable(grid, one)
        f[f'a{action}_can_move'] = int(can_move)
        f[f'a{action}_distance'] = distance_after
        f[f'a{action}_improves'] = int(distance_after < distance_now)
    # 整个网格的物体与颜色
    for r in range(8):
        for c in range(8):
            f[f'cell_{r}_{c}_object'] = int(grid[r,c,0])
            f[f'cell_{r}_{c}_colour'] = int(grid[r,c,1])
    return f
```

然后为每个机器人训练一个 CatBoost 分类器。

```python
from catboost import CatBoostClassifier

def make_model(iterations):
    return CatBoostClassifier(iterations=iterations, depth=8, learning_rate=0.07,
        loss_function='MultiClass', eval_metric='Accuracy', l2_leaf_reg=5,
        random_seed=42, thread_count=-1, verbose=False, allow_writing_files=False)
```

调参策略：先用 700 轮训练一个小模型，用 10% 验证集做 early stopping。

```python
probe = make_model(700)
probe.fit(X.iloc[fit_idx], labels[fit_idx],
          eval_set=(X.iloc[valid_idx], labels[valid_idx]),
          early_stopping_rounds=80, verbose=False)
```

我们还做了一步规则融合：对合法的 `pick` / `drop` 概率乘以放大系数，对非法动作直接掩码为负值。

```python
def logic_predict(frame, probabilities, config):
    pick_boost, drop_boost, mask_invalid = config
    adjusted = probabilities.copy()
    can_pick = frame['ready_pick'].to_numpy(dtype=bool)
    can_drop = frame['ready_drop'].to_numpy(dtype=bool)
    adjusted[can_pick,4] *= pick_boost
    adjusted[can_drop,5] *= drop_boost
    if mask_invalid:
        adjusted[~can_pick,4] = -1.0
        adjusted[~can_drop,5] = -1.0
    return adjusted.argmax(axis=1)
```

我们在放大系数 `1.0,1.25,1.5,2.0,3.0,6.0` 的组合网格里挑选验证集准确率最高的配置。

### 2.4 模型测试与指标报告
对每个机器人，我们记录 raw 验证分数和叠加逻辑后的分数。

```python
from sklearn.metrics import accuracy_score
print('Raw mean validation score:', round(100*np.mean(raw_validation_scores),4))
print('Logic mean validation score:', round(100*np.mean(validation_scores),4))
```

在验证集上，逻辑融合平均能给每个机器人带来约 1 到 3 个百分点的提升。我们用最优配置重新拟合最终模型，并对测试集预测。

```python
final_model = make_model(iterations)
final_model.fit(X.iloc[indices], labels[indices], verbose=False)
test_idx = np.flatnonzero(test_robot_ids == robot_id)
test_probabilities = final_model.predict_proba(X_test.iloc[test_idx])
predictions[test_idx] = logic_predict(X_test.iloc[test_idx], test_probabilities, best_config)
```

最后写出 `predictions.json` 和 `submission.csv`。

```python
json.dump(predictions.tolist(), open('predictions.json','w'))
pd.DataFrame({'id': range(len(test)), 'action': predictions}).to_csv('submission.csv', index=False)
```

## 三、总结
本文中我们完成了 Robot Chasing 的动作预测任务。我们首先解析观测与任务，把 8x8 网格、机器人朝向、搬运状态等信息整理成结构化特征；其次用 BFS 距离地图构造动作感知特征并加入规则动作；接着为每个机器人单独训练 CatBoost，并用 early stopping 和小网格规则融合调优；最后在验证集上报告原始与融合后的准确率并生成提交文件。整个过程说明了网格世界中结合规划（BFS）与学习的有效性：纯模型负责把状态映射成动作，规则负责约束非法动作。