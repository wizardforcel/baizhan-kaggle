# Robo Vision Challenge Data Hack 3.0

## 一、比赛简介

### 1.1 比赛背景
Robo Vision Challenge 是 DataHack 3.0 的一个机器人视觉任务。参赛者需要根据预先提取的图像特征，预测机器人下一步要执行 6 次拾取动作的坐标。任务包含两个子任务 `task1` 和 `task2`，每个子任务要求对碗和块两种物体各给出 3 个拾取点。特征已经由主办方从图像中提取出来，包括复杂度分数、空间熵、边缘密度、主色调、饱和度均值、纹理复杂度、对称性分数、质心坐标、频谱指数、梯度相干性等 14 个数值特征。

### 1.2 比赛目标
训练集包含 `image_id`、`task_type`、`complexity_score` 等 14 个特征以及 6 个 `action_i` 字符串，格式为 `x,y`。测试集只有特征，需要输出 6 个动作的坐标。我们为每个动作输出 `x_pick,y_pick,x_place,y_place` 四个数。`x_place`、`y_place` 在材料中由任务固定的目标位置给出。

### 1.3 评价标准
评价指标是均方根误差 RMSE，越小越好。对所有 12 个坐标维度的预测误差取均方根。若预测值超出物理范围，需要做裁剪。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先定位数据路径并列出所有文件。

```python
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
base_dir = find_dataset_path()
train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))
```

特征列表固定为 14 个，我们检查缺失值。

```python
features = [
    'complexity_score','spatial_entropy','edge_density',
    'dominant_hue','saturation_mean','texture_complexity',
    'symmetry_score','centroid_x','centroid_y','spectral_index',
    'gradient_coherence','local_variance_ratio','frequency_energy','contrast_ratio'
]
X = train_df[features].fillna(0).values
```

我们把 `action_i` 字符串解析成 12 个目标列。

```python
def parse_action_targets(row):
    targets = []
    for i in range(6):
        action_str = row[f'action_{i}']
        if pd.isna(action_str):
            targets.extend([0.0, 0.0])
        else:
            parts = str(action_str).split(',')
            targets.extend([float(parts[0]), float(parts[1])])
    return pd.Series(targets)
```

### 2.2 数据清洗
特征中存在少量缺失，我们用 0 填充，这在视觉特征中通常代表未检测到该属性。

```python
X_test = test_df[features].fillna(0).values
```

坐标目标也被限制在物理范围内。

```python
x_pick = np.clip(pred_coords[i*2], -0.3, 0.3)
y_pick = np.clip(pred_coords[i*2 + 1], -0.8, -0.2)
```

裁剪防止模型输出不合理的坐标。

### 2.3 模型训练与超参数调优
我们采用多输出随机森林，5 折交叉验证。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
test_preds = np.zeros((len(test_df), 12))
oof_rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    oof_rmse_scores.append(fold_rmse)
    test_preds += model.predict(X_test) / N_SPLITS
```

模型参数 `n_estimators=100`、`max_depth=10` 是在计算资源和稳定性的折中。随机森林对特征间的非线性关系鲁棒，适合图像统计特征。

### 2.4 模型测试与指标报告
训练结束后我们计算平均 OOF RMSE。

```python
print(f"Average Out-Of-Fold RMSE: {np.mean(oof_rmse_scores):.4f}")
```

最后根据每个测试样本的 `task_type`，把固定放置位置与预测拾取位置拼接成提交格式。

```python
TARGETS = {
    'task1': [(-0.2000, -0.3000), (0.0000, -0.5000), (0.2000, -0.7000), ...],
}
for idx, row in test_df.iterrows():
    image_id = row['image_id']
    task = row['task_type']
    targets = TARGETS[task]
    pred_coords = test_preds[idx]
    predicted_actions = {}
    for i in range(6):
        x_pick = np.clip(pred_coords[i*2], -0.3, 0.3)
        y_pick = np.clip(pred_coords[i*2 + 1], -0.8, -0.2)
        x_place, y_place = targets[i]
        predicted_actions[f'action_{i}'] = f"{x_pick:.4f},{y_pick:.4f},{x_place:.4f},{y_place:.4f}"
```

提交文件按 `image_id` 排序输出。

## 三、总结
本文中我们完成了 Robo Vision Challenge 的坐标预测任务。我们首先确认了 14 维图像特征并解析了动作标签；其次用 0 填充缺失并对预测坐标做裁剪；接着用 5 折随机森林做多输出回归；最后按任务拼接固定放置点生成提交文件。整个流程体现了在特征已提取的情况下，简单稳健的集成模型即可给出可用的机器人动作预测。
