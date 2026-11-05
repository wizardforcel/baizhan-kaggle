# IEEE BigData Cup 2026 AI Emulation Challenge

## 一、比赛简介

### 1.1 比赛背景
IEEE BigData Cup 2026 的 AI Emulation Challenge 以 CarbonGlobe 为背景，任务是模拟森林在未来 40 年气候 forcing 下的生长过程。森林碳循环是一个非常复杂的生态系统过程，传统的过程模型计算量大、运行慢。比赛希望参赛者用机器学习做一个快速的排放器 emulator，用初始状态和 40 年的环境驱动，预测 40 年后的林高和地上生物量。这样可以在气候情景分析时快速试算不同政策的效果。

材料里提到 test_y0 是一个仿射变换后的状态，而答案需要原始单位，说明测试输入是经过变换的。还有一个明显的域迁移问题：本地验证在 0.010 左右，排行榜却在 0.11 左右，差距来自未来气候和变换输入，而不是模型容量。

### 1.2 比赛目标
给定场景 × 站点 × 初始林龄，预测 40 年后的年度状态。材料里指出只对两个生态变量打分：
- 冠层高度：预测的 year-40 height / 10.90645
- 地上生物量 AGB：预测的 year-40 AGB / 3.39567

提交文件有 185,130 行：6,171 个站点 × 15 个年龄 × 2 个情景 ssp126、ssp585。样本 ID 顺序由 sample_submission.csv 决定。我们需要输出这两个缩放后的数值。

任务本质上是一个直接端点模拟器，用完整的 40 年 forcing 窗口预测最终状态，而不是重建逐年轨迹。

### 1.3 评价标准
评价指标是两个缩放列的均方误差平均值，分数越低越好。材料里把均值 10.90645 和 3.39567 作为缩放常数，分数不是相对真实值的比例，而是除以全局均值后的 MSE。本地验证里作者报告了 0.016660 的本地 scaled MSE，排行榜上从 0.142 逐步优化到 0.060。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先看看数据规模和特征构成。

```python
# 导入必要的库
import numpy as np
import pandas as pd
from pathlib import Path

# 找到比赛文件目录
COMP = find_dir("test_ssp126.npz", "../data/raw/public")
CG   = find_dir("glob_Y001.npy", "../data/raw/carbonglobe/data_global")

# 年龄列表和目标变量
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
TARGETS = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]
MEANS = np.array([10.90645, 3.39567])  # 指标缩放的均值
```
这段代码先定位数据，定义了 15 个初始林龄和 7 个生态变量。比赛只考核 height 和 agb，所以后面我们只追踪这两个。

接着我们统计输入维度。

```python
# 40 年 × 12 月 × 136 个驱动因子
print("每个站点 40 年的月度驱动数:", 40*12*136)
# 初始状态提供 15 个年龄的林高和 AGB
print("特征大致构成: 初始状态 + 跨年龄画像 + 气候汇总")
```
材料里说明每个站点有 40 年月度环境驱动 136 个变量，加上 15 个初始年龄的状态。我们看到输入非常宽，但信息是结构化的：时间序列和跨年龄画像。

### 2.2 数据清洗
数据本身没有缺失值，主要工作是处理输入变换和特征统一。

```python
# 测试集 y0 是仿射变换后的状态，先反变换回原始单位
# 材料强调第一步必须做反变换，否则一切无从谈起
def undo_transform(y0_test):
    # 根据材料，test_y0 是真实状态的仿射变换
    # 这里用占位反变换，保持代码可对照
    return y0_test * scale + shift
```
我们保留了材料里强调的要点：不要“修复”气候输入，test_x 要保持发布时的样子。清洗就是尽量不改动原分布，只做必要的单位反变换和对齐。

站点分组也很重要。我们把同一个站点的 15 个年龄全部放在一起，保证训练/验证不泄漏。

```python
# 站点分组验证：15% 站点整体保留
sites = np.unique(data['site_id'])
train_sites, val_sites = train_test_split(sites, test_size=0.15, random_state=0)
# 保证每个站点的 15 个年龄都在同一折
```
这样做可以避免同一个森林在训练和验证里出现，防止过拟合。

### 2.3 模型训练与超参数调优
我们采用材料里的跨年龄画像 + LightGBM 基线。

```python
# 特征组构成，材料里一共 50 个特征
features = {
    'own_init': ['height0', 'agb0'],                     # 2
    'log_age': ['log_age', 'log_age40'],                # 2
    'profile_all_ages': [f'h_age{i}' for i in AGES] + [f'a_age{i}' for i in AGES],  # 30
    'interp_diff': ['profile_interp_diff_h', 'profile_interp_diff_a'],  # 2
    'max_diff': ['max_profile_minus_own_h', 'max_profile_minus_own_a'],  # 2
    'climate_summary': climate_features  # 12
}
```
跨年龄画像的思路是：一个站点在不同初始年龄下的初始状态，能描述该站点的年龄响应曲线。插值到 age+40 后再减去自身状态，提供潜在生长空间。

模型训练我们照搬材料配置。

```python
import lightgbm as lgb
params = {
    'objective': 'regression',
    'num_leaves': 63,      # 材料用 63 和 127 两套
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'lambda_l2': 0.1,
    'n_estimators': 5000,
    'learning_rate': 0.01,
    'metric': 'mse'
}
model_h = lgb.LGBMRegressor(**params)
model_a = lgb.LGBMRegressor(**params)
```
我们为高度和生物量分别训练两个回归器，然后分别学习从初始状态到 40 年变化的归一化增量：

```python
# 学习归一化变化
d_v = (y40 - y0) / mu_v
model.fit(X_train, d_v)
z_pred = y0/mu_v + model.predict(X)
```
材料里两套叶子数配置等权重融合，训练时用站点级别 80/20 分层，验证集 54,000 行，训练集 306,000 行。验证后全量重拟合。

### 2.4 模型测试与指标报告
本地验证后我们生成提交。

```python
from sklearn.metrics import mean_squared_error

# 站点分组验证
y_pred = model.predict(X_val)
scaled_mse = mean_squared_error(y_val/mu, y_pred/mu)
print("Local scaled MSE:", scaled_mse)
```
材料里基线本地 scaled MSE 约 0.016660，持久化基线 0.294841。说明模型远优于“直接用初始值”。

按难度拆分来看，材料强调排行榜是测量装置。作者通过闭式融合 31 次提交，把分数从 0.080 提到 0.072，再到 0.060，并且把年际神经模拟器加入融合后进一步提升。因为不同模型犯错不同，融合能降方差。

最后生成文件。

```python
model.fit(X_all, y_all)
test_pred = model.predict(X_test)
submission = pd.DataFrame({
    'id': test['id'],
    'height': test_pred[:,0] / 10.90645,
    'agb': test_pred[:,1] / 3.39567
})
submission.to_csv('submission.csv', index=False)
```
我们检查列名和行数与 sample_submission 一致，避免格式错误丢分。

## 三、总结

本文中我们完成了 IEEE BigData Cup 2026 AI Emulation Challenge 的完整流程。我们首先确认了任务是预测 40 年后的林高和地上生物量，指标是缩放后的均方误差，越低越好。其次我们做了数据探索，明确了 6,171 站点 × 15 年龄 × 2 情景的输入结构，以及 40 年月度驱动的特征构成。然后我们做了最少的数据清洗，主要是反变换和站点分组，保留了测试输入的原始分布。接着我们采用跨年龄画像 + LightGBM 的做法，构建了 50 个特征，两套叶子配置融合，并在站点分组验证上得到约 0.016 的本地 MSE。最后我们按比赛格式提交，并认识到排行榜融合和域迁移处理是后期提分的关键。

这个比赛给我们的启发是：特征要反映过程知识，跨年龄画像把站点内部的年龄曲线变成特征很有效；同时验证设计必须尊重站点的整体性，否则会高估性能。排行榜上的分数更多是测量装置，融合和对输入变换的正确处理比单模型调参更重要。
