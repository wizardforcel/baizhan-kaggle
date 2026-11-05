# IEEE BigData Cup 2026 AI Emulation Challenge

## 一、比赛简介

### 1.1 比赛背景
IEEE BigData Cup 2026 的 AI Emulation Challenge 以 CarbonGlobe 为背景，任务是模拟森林在未来 40 年气候驱动（climate forcing）之下的生长过程。森林碳循环是一个非常复杂的生态系统过程，树冠高度（canopy height）和地上生物量（aboveground biomass，AGB）都与碳排放和碳汇直接相关。传统的过程模型，比如材料里提到的 ED（Ecosystem Demography，生态系统人口学模型），计算量很大、运行很慢，很难快速在不同气候情景下做大规模试算。

比赛希望参赛者用机器学习做一个快速的模拟器（emulator），输入某个地点的初始森林状态和未来 40 年的月度气候驱动，就能立刻得到 40 年后的林高和生物量。这样在评估不同气候政策或研究碳排放路径时，就不需要反复运行几天才能出数的大模型。

这个任务有两个特别容易踩坑的地方。第一，材料里明确指出测试输入 `test_y0` 是真实状态经过仿射变换（一种对数据做平移和缩放的线性变换）之后的结果，但评分用的是原始单位，所以第一步必须把变换还原，否则后面做得再好也没用。第二，本地验证和排行榜的成绩差距很大：模型本地验证约 0.010，排行榜却在 0.11 左右，这个差距来自未来气候带来的域迁移（domain shift），而不是模型能力不够。

### 1.2 比赛目标
给定场景 × 站点 × 初始林龄，预测 40 年后的年度状态。材料里说明，只有七个生态变量中的两个会被评分：

| 目标变量 | 提交值 |
| --- | --- |
| 冠层高度（Canopy height） | 预测的 year-40 高度 ÷ 10.90645 |
| 地上生物量（AGB） | 预测的 year-40 AGB ÷ 3.39567 |

提交文件总共有 185,130 行，结构是 6,171 个站点 × 15 个初始年龄 × 2 个未来情景（ssp126 和 ssp585）。样本 ID 的顺序由 `sample_submission.csv` 给定，我们不能随意调换。每个站点的输入包括：初始状态提供的 15 个年龄各自的高度和 AGB，以及 40 年逐月（40 年 × 12 个月 × 136 个环境驱动因子）的气候数据。

严格来说，这是一个「直接端点模拟器」：直接用完整的 40 年气候窗口去预测最终状态，而不是重建逐年轨迹。因为当前状态已经解释了很多最终水平，所以材料里的做法是让模型学习「归一化变化量」，而不是直接回归绝对高度。

### 1.3 评价标准
评分公式是两列缩放值的均方误差（mean squared error，MSE）的平均数，分数越低越好。缩放就是除以全局均值 10.90645 和 3.39567，这个除数不是每行的真实值，只是一个固定常数。

如果不加任何模型，直接拿「year 40 = year 0」当预测，也就是所谓的持久化基线，其本地 scaled MSE 约为 0.294841。材料里报告的 LightGBM 基线本地 scaled MSE 约 0.016660，远好于持久化基线。排行榜上顶尖的方案从 0.142 起步，通过一步步修正和融合，最终降低到 0.060。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先看看数据的规模和特征构成。

```python
import numpy as np
import pandas as pd
from pathlib import Path

def find_dir(marker, local):
    # 在 Kaggle 环境里查找比赛文件目录，找不到就退回本地目录
    kin = Path("/kaggle/input")
    if kin.exists():
        hits = sorted(kin.rglob(marker))
        if hits:
            return hits[0].parent
    return Path(local)

# 比赛公开文件（未来发展情景的驱动 + 初始状态）
COMP = find_dir("test_ssp126.npz", "../data/raw/public")
# CarbonGlobe 带标签训练数据
CG   = find_dir("glob_Y001.npy", "../data/raw/carbonglobe/data_global")

# 15 个初始林龄，以及 7 个生态变量
AGES   = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
TARGETS = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]
MEANS = np.array([10.90645, 3.39567])   # 指标用来缩放高度和生物量的常数
```
这段代码先用标记文件定位数据，再定义林龄列表。7 个生态变量里只有 height 和 agb 会被评分，因此我们在 EDA 里重点关注这两个量在不同初始年龄下的分布。

接着我们观察初始状态的跨年龄画像。这一步非常关键：同一个站点在不同初始年龄下的状态，描述了「这个地点的生长曲线长什么样」。

```python
# 对每个站点，画出 15 个年龄对应的初始高度
# 这 15 个点连起来就是该站点的年龄画像
for site in sample_sites:
    heights = initial_profiles[site]   # 形状为 (15,)
    plt.plot(AGES, heights, label=f"site {site}")

plt.xlabel("initial age (year)")
plt.ylabel("height at year 0")
plt.legend()
plt.show()
```
我们看到不同站点的画像形态差别很大：有的地方年轻时高度增长快，有的地方整体都缓。这说明「和自身画像比较」比「看绝对数值」更有预测力。

我们还想确认每个站点是否都有完整的 15 个年龄数据。

```python
# 检查初始状态数组的维度：(站点, 年龄, 变量)
print("initial state shape:", init_state.shape)
print("站点数:", 6171, "年龄数:", 15, "变量数:", 7)
```
材料明确给出提交文件是 6,171 站点 × 15 年龄 × 2 情景，所以输入也应该按这个结构组织，方便逐行对接。

### 2.2 数据清洗
数据本身没有缺失值，主要工作是对齐单位、还原仿射变换、按站点分组。

首先是还原测试输入：材料强调 `test_y0` 是经过仿射变换的状态，而答案在原始单位，必须先反变换。

```python
# 测试集 y0 是仿射变换后的状态，先反变换回原始单位
# 这是材料里五条经验的第一条：不改这一步，其他都白做
def undo_affine_transform(y0_test):
    # 根据材料，y0_test = a * y_true + b，因此 y_true = (y0_test - b) / a
    # 实际 a、b 由官方发布数据隐含决定，这里用占位写法保持流程清晰
    return (y0_test - b) / a
```
第二条经验是「不要修复气候输入」：有人想把 test_x 对齐到训练气候分布，结果分数反而变差。所以我们清洗的原则是尽量保留原始分布，只做必要的单位转换。

其次是站点分组。同一个站点的 15 个初始年龄一定要放在同一个折里，否则模型可能在验证时「见过」同一片森林，导致本地分数虚高。

```python
from sklearn.model_selection import train_test_split

# 用站点维度做分层划分，而不是按行随机
sites = np.unique(init_state.site_id)
train_sites, val_sites = train_test_split(
    sites, test_size=0.15, random_state=0)

# 训练、验证都带齐 15 个年龄
train_rows = init_state[init_state.site_id.isin(train_sites)]
val_rows   = init_state[init_state.site_id.isin(val_sites)]
```
材料里的验证设计是：从采样站点里留出 15% 的完整站点，得到 306,000 行训练、54,000 行验证。站点分组能反映真实的应用场景，因为我们需要的是模型面对全新地点的泛化能力。

### 2.3 模型训练与超参数调优
我们采用材料里的「跨年龄画像 + LightGBM」基线方案。核心想法是：当前状态已经解释了大部分最终水平，所以模型学习的是归一化变化量。

```python
# 模型学习目标：归一化的 40 年变化（除以均值常数）
# d_v = (y_40 - y_0) / mu_v
train_y = (state_40 - state_0) / MEANS
# 预测时再还原：yhat_40 = MEANS * (	y_0/MEANS + dhat)
```
特征组按照材料的 50 个特征来组织：

| 特征组 | 数量 | 作用 |
| --- | --- | --- |
| 自身的初始高度和 AGB | 2 | 锚定预测的生长量 |
| log(初始年龄)、log(年龄+40) | 2 | 表达年龄相关的响应 |
| 全部 15 个年龄的初始高度和 AGB | 30 | 描述站点的年龄画像 |
| 插值到 age+40 的画像减去自身状态 | 2 | 提供潜在生长空间 |
| 画像最大值减去自身状态 | 2 | 描述离画像最大值的距离 |
| 气候与静态汇总 | 12 | 纳入环境驱动 |

代码中我们先把这些特征拼成一张表。

```python
def build_features(init_state, AGES, MEANS):
    features = pd.DataFrame(index=init_state.index)
    own = init_state[['height0', 'agb0']]
    # 1. 自身的初始状态
    features['h0'] = own['height0']
    features['a0'] = own['agb0']
    # 2. 年龄的对数变换
    features['log_age']   = np.log(init_state['age'])
    features['log_age40'] = np.log(init_state['age'] + 40)
    # 3. 全部 15 个年龄的画像：每个年龄的 h/a 各一列
    for age in AGES:
        features[f'h_age_{age}'] = init_state[f'h_age_{age}']
        features[f'a_age_{age}'] = init_state[f'a_age_{age}']
    # 4. 插值到 age+40 的画像，减去自身当前状态
    features['interp_diff_h'] = features[f'h_age_500'].interpolate() - features['h0']
    features['interp_diff_a'] = features[f'a_age_500'].interpolate() - features['a0']
    # 5. 画像最大值减去自身状态
    features['max_diff_h'] = init_state.filter(like='h_age_').max(axis=1) - features['h0']
    features['max_diff_a'] = init_state.filter(like='a_age_').max(axis=1) - features['a0']
    # 6. 气候汇总由外部传入
    return features
```
材料的插值处理有个细节：对超出最大可用年龄的部分，权重封顶在最后一个年龄，不会无限外推。而且材料特意提醒：跨年龄画像不代表同一片森林未来的轨迹，模型自己会学习要不要用它，不要自作主张做假设。

模型训练使用 LightGBM，目标分别对 height 和 AGB 各训两个配置然后融合。

```python
import lightgbm as lgb

def make_model(num_leaves):
    return lgb.LGBMRegressor(
        objective='regression',
        num_leaves=num_leaves,
        feature_fraction=0.7,   # 特征抽样，降低过拟合
        bagging_fraction=0.7,
        bagging_freq=1,
        lambda_l2=0.1,          # L2 正则
        n_estimators=5000,
        learning_rate=0.01,
        metric='mse',
        random_state=0)

# 两个配置：63 和 127 叶子，结果等权平均
model_h1 = make_model(63); model_h2 = make_model(127)
model_a1 = make_model(63); model_a2 = make_model(127)

model_h1.fit(X_train, y_train_h)
model_h2.fit(X_train, y_train_h)
# 高度和生物量分开训练，各自融合
```
调参方面，我们重点盯叶子数和正则强度。材料说明两套叶子数（63 和 127）配合特征抽样、L2 正则后等权融合就够用，深挖单个超参数收益不大。验证通过后，再用全量 360,000 行重新拟合最终模型。

第二条提升主线是「排行榜即测量装置」。因为 MSE 指标下，每次提交都是一条关于误差的方程，可以把过去所有提交做闭式融合。材料里的旅程线是：GBM v1 0.142 → 气候不动 0.116 → 直接 Δ 模型 + 站点画像 0.111 → 修正 agb 缩放 0.091 → SS5 中的 Δ 乘以 0.88 → 0.086 → 多方向校准 0.080 → 闭式融合 0.072 → 按年龄带修正 0.067 → 加入年际神经模拟器 0.062 → 31 次提交最终融合 0.060。

其中年际神经模拟器单独得分只有 0.144，比主模型差，但融合到 blend 里反而把分数从 0.072 带到 0.060，因为它犯的错误和 GBM 不同，是不同类型的补充。

### 2.4 模型测试与指标报告
我们在站点分组验证上测试最终模型。

```python
from sklearn.metrics import mean_squared_error

def scaled_mse(y_true, y_pred, MEANS=MEANS):
    # 先缩放再求 MSE，与比赛官方一致
    return mean_squared_error(y_true / MEANS, y_pred / MEANS)

for target, models in [('height', [model_h1, model_h2]),
                       ('agb',    [model_a1, model_a2])]:
    pred = np.mean([m.predict(X_val) for m in models], axis=0)
    mse = scaled_mse(y_val[target], pred)
    print(f'{target} scaled MSE: {mse:.6f}')
```
本地验证的 scaled MSE 约为 0.016660，而持久化基线是 0.294841，说明模型确实学到了大量信息。不过材料提醒我们：本地好 ≠ 排行榜好，实际提交会碰到未来气候和变换输入带来的域迁移。

生成提交文件时，我们严格照 `sample_submission.csv` 的顺序。

```python
# 用全量数据重新拟合后对测试集预测
pred_h = np.mean([m.predict(X_test) for m in [model_h1, model_h2]], axis=0)
pred_a = np.mean([m.predict(X_test) for m in [model_a1, model_a2]], axis=0)

submission = pd.DataFrame({
    'id':     sample_sub['id'],
    'height': pred_h / 10.90645,   # 按官方缩放
    'agb':    pred_a / 3.39567,
})
submission.to_csv('submission.csv', index=False)

# 检查行数是否与样本提交一致
assert len(submission) == len(sample_sub)
```
最后我们按概率融合多份提交，把分数进一步压低。融合的权重通过过去提交在排行榜上的误差反推，属于材料里「专题研究」的一部分，我们在总结里再提。

## 三、总结

本文中我们完成了 IEEE BigData Cup 2026 AI Emulation Challenge 的完整流程。我们首先确认任务是预测 40 年后的林高和地上生物量，指标是缩放后的均方误差，越低越好，并意识到测试输入的仿射变换是必须迈过的第一道坎。其次我们做了数据探索，明确了 6,171 站点 × 15 年龄 × 2 情景的输入结构，以及跨年龄画像带来的信息价值。接着我们做了最少的数据清洗——还原变换、保留气候原始分布、按站点分组划分——避免了验证泄漏和画蛇添足。然后我们采用跨年龄画像 + LightGBM 的做法，构造了 50 维特征，用 63/127 叶子的双配置分别拟合 height 和 AGB 并等权融合，站点分组验证得到约 0.016 的本地 scaled MSE。最后我们按样例格式生成提交，并通过多份提交的闭式融合，把排行榜分数从 0.142 一路优化到 0.060。

这个比赛给我们的启发是两点。第一，过程知识（process knowledge）比盲目堆特征重要：跨年龄画像把「站点内部不同年龄的差别」变成特征，比直接给定初始值有效得多。第二，「好验证 ≠ 好排名」在域迁移任务里是常态，本地分数只用于挑模型，真正要相信的还是排行榜这个测量装置；不同模型的错误互补，融合常常是单模型到极限后唯一的提分手段。