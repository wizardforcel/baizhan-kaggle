# ClimX 极端气候模拟与不确定性量化（UQ Track）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的全名是 **ClimX: Extreme-Aware Earth System Emulation and Probabilistic Uncertainty Quantification**，由欧洲空间局（ESA $\Phi$-lab）发起，核心数据来自挪威地球系统模式 **NorESM2-MM**。

先说背景。气候科学家研究未来气候，靠的是「地球系统模式」（General Circulation Model，GCM）：把大气、海洋、陆面放进一大堆物理方程里，一天一天地往前推演。这种模式非常准确，但也非常贵——跑一次几十年、上百年的模拟，要占用超级计算机很长时间。于是出现了一个折中办法：**气候模拟器（emulator）**。它的思路是「用便宜的机器学习模型，去学贵模式输入和输出之间的映射」——给进去温室气体浓度、气溶胶排放，直接吐出温度、降水等气象场，速度快几个数量级，可以用来做大量情景下的风险评估。

这个 UQ Track 关心的不只是「预测得准不准」，更是「预测得**有多不确定**」。真实的未来有很多种可能，好的模拟器应该给出一整套可能的结局（一个概率分布），而不是一个孤零零的数字。比赛的难点有三个：

1. 必须先预测**逐日气象场**（气温、最高温、最低温、降水、比湿、气压、风速），不允许直接预测极端指数；
2. 极端指数是由逐日场用固定公式算出来的，不能绕开；
3. 最终要给出 15 个 ETCCDI 极端指数在 9 个分位数上的概率预测。

### 1.2 比赛目标

输入是「外强迫」数据：温室气体浓度（CO2、N2O、CH4、CFC11、CFC12 等）、气溶胶（黑碳、硫酸盐、有机质等）的空间分布，以及时间信息。输出是每一天、每个格点上的 7 个气象变量。

参赛数据分两档：

- **全分辨率参考网格**：历史模拟 1850–2014 年共 60,224 个日步，空间 192×288；情景推演 2015–2100 年共 31,389 天，训练用 SSP1-2.6、SSP3-7.0、SSP5-8.5，测试用没见过的 SSP2-4.5。
- **精简网格**：为了在 Kaggle 的 GPU 上快速验证，官方又给了一份 12×18 的抽稀版本（空间降采样 16 倍）。

提交文件是「4,755,456 行 × 139 列」的长表，每一行对应一个（区域、年份、指数），列则是各分位数的预测值。

### 1.3 评价标准

指标是 **CRPS**（Continuous Ranked Probability Score，连续排序概率分），再经过区域加权和方差归一化，变成 **r-nNSE**（归一化 Nash–Sutcliffe 效率）。用大白话说：

- CRPS 衡量「你给的概率分布」和「真实观测值」差多远，既惩罚预测偏离，也惩罚「不敢给分布、只给一个点」；
- 它按纬度余弦加权，并按 IPCC AR6 的陆地参考区域分区统计，避免极地和高纬地区主导分数；
- 最后用历史气候的方差做归一化，分数落在 $(-1, 1]$ 之间，**越大越好**，1 表示完美。

当预测被压缩成 9 个分位数 $\tau \in \{0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95\}$ 时，CRPS 可以用 pinball loss（分位数损失）近似：

$$\text{CRPS}(F, y) \approx \frac{2}{K}\sum_{k=1}^{K}\rho_{\tau_k}(y - \hat q_{\tau_k})$$

也就是说：**每个分位数都往自己该在的位置上靠**，谁偏了谁就被罚。

---

## 二、解题过程

我们的整体思路分四步：先看清数据里的强迫与气象场长什么样；其次把每日数据整理成「一行一个格点」的训练张量；接着用一个带概率输出的神经网络同时学均值与方差，并对降水单独建模；最后用随机集合采样得到分位数并生成提交文件。

### 2.1 数据探索（EDA）

第一步是把所有 NetCDF 文件列出来，确认每份文件里有什么变量、时间跨度多长。气候数据的一个常见坑是日历——NorESM 用的是「noleap」（没有闰年）日历，普通的 `datetime` 解析不了，所以我们先装 `cftime`：

```python
import cftime, xarray as xr
from pathlib import Path

# 气候模式常用 noleap 日历，cftime 才能正确解码
DATA_DIR = Path('/kaggle/input/climx-uq')
available_nc = sorted(DATA_DIR.glob('*.nc'))
print(f"发现 {len(available_nc)} 个 NetCDF 文件")

# 测试集只有强迫数据，没有气象场标签
ds_test_forc = xr.open_dataset(
    DATA_DIR / 'test_ssp245_forcing.nc',
    use_cftime=True   # 关键：允许非标准日历
)
# 看时间长度与网格大小
print(ds_test_forc.sizes['time'], ds_test_forc.sizes['lat'], ds_test_forc.sizes['lon'])
print(list(ds_test_forc.data_vars))
```

输出的时间长度是 31,389 天（86 年），空间是 12×18 的精简网格，强迫变量包括 CO2、N2O、CH4、气溶胶等。这告诉我们两件事：一是测试情景（SSP2-4.5）在训练里**没出现过**，模型必须学会「按强迫推演」，而不是背住某条情景；二是每天有 216 个格点，训练样本数会非常可观。

接着我们画几条 CO₂ 轨迹，确认三个训练情景的差异：

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 6))
# 三条训练情景的 CO2 边界条件，看它们随时间如何分叉
for name in ['ssp126', 'ssp370', 'ssp585']:
    ds = xr.open_dataset(DATA_DIR / f'train_{name}.nc', use_cftime=True)
    co2 = ds['CO2_LBC'].values.squeeze()
    # 强迫有时带空间维，取空间平均得到一条一维曲线
    if co2.ndim > 1:
        co2 = co2.mean(axis=-1)
    ax.plot(co2, label=name)
ax.set_xlabel('Year index')
ax.set_ylabel('CO2 (ppm)')
ax.legend()
```

三条曲线在 2015 年起点相同，到 2100 年分得很开（低排放、中排放、高排放），这正是模拟器要学的「输入差异 → 输出差异」。

然后我们看历史气象场的气候态，重点是两样东西：温度的空间分布，以及**昼夜温差 DTR = tasmax − tasmin**。为什么要看 DTR？因为最高温和最低温如果被模型分开预测，很容易出现 `tasmin > tasmax` 这种物理上不可能的结果，后面必须加约束。

```python
# 取 1961–1990 作为气候基准期
tas_base = ds_hist['tas'].sel(time=slice('1961', '1990'))
tas_clim = tas_base.mean(dim='time').values - 273.15   # K 转摄氏度

# 昼夜温差：最高温减最低温
dtr = (ds_hist['tasmax'].sel(time=slice('1961','1990'))
       - ds_hist['tasmin'].sel(time=slice('1961','1990'))).mean(dim='time').values

# 画全球分布，确认空间结构合理（陆地温差大、海洋小）
plt.pcolormesh(ds_hist['lon'], ds_hist['lat'], tas_clim, cmap='Spectral_r')
plt.colorbar(label='Mean tas (C)')
plt.title('1961-1990 baseline climatology')
```

图上可以看到：高纬陆地最冷、赤道最热，DTR 在干旱内陆最大（十几 K），在海洋和湿润地区只有几 K。这些结构是后面检查模型输出的「常识基准」。

降水还有一个特殊现象：**大量格点、大量日子是 0**。我们统计了一下湿日比例和湿日强度分布，发现如果直接对原始降水做回归，模型会输出「到处 0.5～2.0 mm/day 的毛毛雨」——因为均方误差的最优解就是条件均值，而条件均值被大量干日和少量暴雨拉扯，正好落在中间。这就是材料里说的「drizzle catastrophe」（毛毛雨灾难），它会让干旱指数 CDD 的 $R^2$ 低到 −900 这种离谱数值。这个发现直接决定了后面的建模方式。

### 2.2 数据清洗

这份数据的「清洗」主要不是去脏值，而是**统一单位、构造特征、处理物理约束**。

第一，把所有量换成物理上直观的单位：降水从 kg m⁻² s⁻¹ 乘 86400 变成 mm/day；气压除以 1e5 变成 hPa 量级；比湿乘 1000 变成 g/kg。这样各变量的数值范围接近，神经网络才好训练。

第二，把强迫展开成「每个格点一行」的特征矩阵。温室气体是全球标量，要广播到所有格点；气溶胶本身是空间场，直接按格点取值；再加两组谐波特征——空间经纬度的正余弦、以及一年中第几天的正余弦。谐波的作用是让模型知道「这里是哪里」「现在是什么季节」，否则它只能看到强迫，分不清赤道和极地。

```python
# 空间谐波：把经纬度编码成周期特征，避免模型在 0/360 经度处断开
lat_rad = np.deg2rad(lats)
lon_rad = np.deg2rad(lons)
spatial_features = np.column_stack([
    np.sin(lat_rad), np.cos(lat_rad),
    np.sin(lon_rad), np.cos(lon_rad),
    np.sin(2*lat_rad), np.cos(2*lat_rad),
])

# 时间谐波：一年周期，告诉模型现在是几月
doy = (t_idx % 365) + 1
doy_feat = np.tile(np.array([
    np.sin(2*np.pi*doy/365.25),
    np.cos(2*np.pi*doy/365.25)
], dtype=np.float32), (n_pixels, 1))

# 拼成一天的输入：(格点数, 10个强迫 + 空间谐波 + 2个季节谐波)
x_day = np.hstack([ghg_broadcast, aerosol_mat, spatial_features, doy_feat])
```

第三，预计算历史基准阈值。WSDI（暖昼持续指数）要数「最高温超过历史 90 分位的连续天数」，CSDI 要数「最低温低于历史 10 分位的连续天数」，R95p 要数「降水超过历史湿日 95 分位的累计量」。这些阈值必须从**历史期**算出来，不能从测试期算，否则就是信息泄漏：

```python
# 只用历史期（1850–2014）计算基准阈值
hist_tas_flat = ds_hist['tas'].values.reshape(-1, n_pixels)
hist_pr_flat  = ds_hist['pr'].values.reshape(-1, n_pixels) * 86400.0

# 每个格点的 90/10 分位阈值
threshold_t90 = np.percentile(hist_tas_flat, 90, axis=0)
threshold_t10 = np.percentile(hist_tas_flat, 10, axis=0)

# R95p 只统计湿日（>=1 mm/day）的 95 分位
threshold_r95p = np.zeros(n_pixels)
for p in range(n_pixels):
    wet_p = hist_pr_flat[:, p][hist_pr_flat[:, p] >= 1.0]
    threshold_r95p[p] = np.percentile(wet_p, 95) if len(wet_p) >= 20 else 5.0
```

第四，训练时每 15 天取一个样本（`sample_stride = 15`）。原因是相邻日期的气象场高度相关，全取会浪费显存且信息重复；抽稀后不同情景加起来仍有足够多的独立样本。

### 2.3 模型训练与超参数调优

模型叫 `ClimX_UQNet`，是一个多层感知机（MLP），但输出不是普通的数值，而是**概率分布的参数**。结构上有三块：

- **特征融合 + 残差主干**：输入层 → LayerNorm → 两个带残差连接的 MLP 块（GELU 激活）。残差连接让深层网络更容易训练。
- **热力与动力头**：一次输出 6 个变量的**均值** $\mu$ 和 **对数方差** $\log\sigma^2$。方差就是「不确定性」——模型自己学会「哪里我拿不准」。
- **零膨胀降水头**：先判断「今天下不下雨」（一个 logit），再对湿日预测对数强度的均值和方差。

```python
class ClimX_UQNet(nn.Module):
    def __init__(self, in_features, hidden_dim=128):
        super().__init__()
        self.fc_in = nn.Linear(in_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        # 两个残差块，每块里两层全连接
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

        # 6 个连续气象变量：均值 + 对数方差
        self.head_thermal_mean   = nn.Linear(hidden_dim, 6)
        self.head_thermal_logvar = nn.Linear(hidden_dim, 6)

        # 降水：是否下雨 / 湿日强度均值 / 湿日强度方差
        self.head_pr_occurrence     = nn.Linear(hidden_dim, 1)
        self.head_pr_intensity_mean = nn.Linear(hidden_dim, 1)
        self.head_pr_intensity_logvar = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.gelu(self.ln1(self.fc_in(x)))
        h = h + self.block1(h)   # 残差，梯度更容易回传
        h = h + self.block2(h)

        thermal_mean   = self.head_thermal_mean(h)
        # 夹住对数方差范围，防止训练早期方差爆炸或塌缩到 0
        thermal_logvar = torch.clamp(self.head_thermal_logvar(h), min=-6.0, max=4.0)

        pr_logit      = self.head_pr_occurrence(h)
        pr_int_mean   = self.head_pr_intensity_mean(h)
        pr_int_logvar = torch.clamp(self.head_pr_intensity_logvar(h), min=-4.0, max=4.0)
        return thermal_mean, thermal_logvar, pr_logit, pr_int_mean, pr_int_logvar
```

输入维度是「10 个强迫 + 空间谐波维度 + 2 个季节谐波」，隐藏层宽度 128。这个规模对 216 个格点 × 数万天来说足够，又不会过拟合。

损失函数是模型的关键。热力部分用**高斯负对数似然（NLL）**：

$$\mathcal{L}_{\text{thermal}} = \frac{1}{2}\left[\log\sigma^2 + \frac{(y-\mu)^2}{\sigma^2}\right]$$

它比 MSE 多了一项 $\log\sigma^2$：如果模型把某处方差预测得很大，虽然残差项变小了，但会被 $\log\sigma^2$ 惩罚。于是模型必须在「承认不确定」和「保持精度」之间找平衡，这正好是 UQ 任务想要的。

降水部分用**零膨胀模型**：先用二元交叉熵判断干湿（阈值 0.5 mm/day），再只在湿日上算对数正态 NLL。

```python
# 一次前向：拿到全部 5 组参数
th_mean, th_logvar, pr_logit, pr_int_mean, pr_int_logvar = model(batch_x)

# 1) 连续变量：高斯 NLL，方差项自动做不确定性校准
th_loss = 0.5 * torch.mean(
    th_logvar + (batch_th - th_mean)**2 / torch.exp(th_logvar))

# 2) 降水发生概率：干湿二分类
is_wet = (batch_pr >= 0.5).float().unsqueeze(1)
bce_loss = F.binary_cross_entropy_with_logits(pr_logit, is_wet)

# 3) 湿日强度：在湿日子集上算对数正态 NLL
wet_mask = (batch_pr >= 0.5)
if wet_mask.sum() > 0:
    target_log_pr = torch.log(batch_pr[wet_mask] + 1e-4).unsqueeze(1)
    intensity_loss = 0.5 * torch.mean(
        pr_int_logvar[wet_mask]
        + (target_log_pr - pr_int_mean[wet_mask])**2
          / torch.exp(pr_int_logvar[wet_mask]))
else:
    intensity_loss = torch.tensor(0.0, device=DEVICE)

total_loss = th_loss + bce_loss + intensity_loss
```

优化器用 AdamW（`lr=1e-3`、`weight_decay=1e-4`），配余弦退火学习率（10 个 epoch 从 1e-3 降到 5e-5），batch size 4096，梯度裁剪 2.0。训练 10 个 epoch，日志里分别打印总损失、热力 NLL 和降水损失——分开看很重要：如果总损失在降但降水损失不降，说明模型在「放弃降水、专攻温度」。

超参数方面，隐藏层宽度在 128 附近试过更大的值，收益不明显却显著变慢；学习率用到 1e-3 以上容易在早期出现方差夹紧饱和。最终这套配置在 Kaggle 的双 T4 上跑完 10 个 epoch 是可接受的。

关于物理一致性，我们在采样阶段加了两条硬约束：`tasmax >= tas + 0.5`、`tasmin <= tas - 0.5`。这是先验知识——日最高温不可能比日均温还低，最低温不可能比日均温还高。用 `np.maximum` / `np.minimum` 一行就能保证。

### 2.4 模型测试与指标报告

UQ 任务的测试不是「预测一次」，而是**生成一个集合**。我们对测试期每个格点、每一天采样 $M=20$ 条随机实现：热力变量按 $\mu + \epsilon\sigma$ 采样（共享同一个噪声 $\epsilon$，保证 tas/tasmax/tasmin 之间相关），降水先用伯努利决定下不下雨，湿日再按对数正态采强度。

```python
N_REALIZATIONS = 20   # 集合成员数，越多分位数越稳，但越慢
ensemble_indices = {k: np.zeros((N_REALIZATIONS, n_years, n_lat, n_lon))
                    for k in INDEX_NAMES}

for m in range(N_REALIZATIONS):
    for y_idx, year in enumerate(years):
        # 按年组织，逐日预测后聚合成年度极端指数
        for t in range(days_in_year):
            eps_t = np.random.randn(n_pixels)          # 热力变量共享噪声
            tas_m    = tas_pred + eps_t * tas_sigma
            # 物理约束：最高温高于均温，最低温低于均温
            tasmax_m = np.maximum(tasmax_pred + eps_t * tasmax_sigma, tas_m + 0.5)
            tasmin_m = np.minimum(tasmin_pred + eps_t * tasmin_sigma, tas_m - 0.5)

            # 降水：先决定是否下雨，再采强度
            pr_gate = (np.random.rand(n_pixels) < rain_prob).astype(np.float32)
            pr_m = pr_gate * np.exp(pr_mu + np.random.randn(n_pixels) * pr_sigma)

            # 按 ETCCDI 定义累加成年度指数
            ensemble_indices['TXx'][m, y_idx] = tasmax_m.max(axis=0).reshape(n_lat, n_lon)
            ensemble_indices['TNn'][m, y_idx] = tasmin_m.min(axis=0).reshape(n_lat, n_lon)
            ensemble_indices['SU'][m, y_idx]  = (tasmax_m >= 298.15).sum(axis=0).reshape(n_lat, n_lon)
            ensemble_indices['FD'][m, y_idx]  = (tasmin_m < 273.15).sum(axis=0).reshape(n_lat, n_lon)
```

15 个指数都用向量化写法算出来，包括 TXx（年最高温）、TNn（年最低温）、SU（夏日数）、TR（热带夜）、FD（霜冻日）、ID（冰日）、WSDI/CSDI（持续暖/冷期，需要数连续天数）、GSL（生长季长度）、Rx5day（5 日最大降水）、CDD/CWD（连续干/湿期）、R95p、R99p 等。

接着从 20 条实现里取 9 个分位数。**这是整个 UQ 方案最巧妙的一点**：分位数来自真实采样的集合，因此天然满足单调性 $q_{0.05} \le q_{0.10} \le \dots \le q_{0.95}$，绝不会出现「5% 分位比 50% 分位还大」的分位数交叉。如果直接让网络输出 9 个分位数，就必须额外加排序约束或惩罚项。

```python
QUANTILES = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]

# 在每个 (指数, 年, 格点) 上，沿集合维取分位数
submission_dict = {}
for idx_name, arr in ensemble_indices.items():
    # arr: (M, years, lat, lon) -> 沿 axis=0 取分位
    qs = np.quantile(arr, QUANTILES, axis=0)   # (9, years, lat, lon)
    for qi, q in enumerate(QUANTILES):
        col = f"{idx_name}_{q}"               # 例如 TXx_0.5
        submission_dict[col] = qs[qi].flatten()
```

最后把精简网格（12×18）上采样回完整网格（192×288），拼成 139 列、4,755,456 行的提交表，并做三项体检：行数与样例一致、没有 NaN、没有无穷值。同时检查每一行的 9 个分位数是否单调——这是提交前的最后一道保险。

需要说明的是：材料里没有给出我们的验证集 r-nNSE 具体数值，所以我们不在这里编造分数。但从训练曲线看，总损失、热力 NLL 与降水损失都在 10 个 epoch 内稳定下降并趋于平缓；从提交分布看，TXx 的中位数落在合理的气温范围内，分位数没有交叉。方案的主要收益来自结构设计而非某个调参技巧：零膨胀降水解决了「毛毛雨灾难」，集合采分位数解决了单调性，方差头解决了「模型不知道自己不知道」的问题。

---

## 三、总结

本文中我们完成了 ClimX UQ Track 的完整建模流程。我们首先用 EDA 摸清了数据结构：测试情景 SSP2-4.5 不在训练集里、空间网格是 12×18、强迫包含全球温室气体与空间气溶胶；同时发现了降水的零膨胀特性，以及直接回归会导致「毛毛雨灾难」。其次我们做了面向任务的清洗：统一单位、把强迫广播到格点、加空间与季节谐波、只用历史期预计算 ETCCDI 阈值以避免泄漏、按 15 天抽稀采样。接着我们用带残差连接的 MLP 搭建概率模拟器：热力变量输出均值与对数方差并用高斯 NLL 训练，降水单独用「发生概率 + 湿日强度」的零膨胀模型。然后我们训练 10 个 epoch（AdamW + 余弦退火 + 梯度裁剪），并在采样时加入最高温/最低温的物理约束。最后我们用 20 条随机实现生成集合，从集合中取 9 个分位数，上采样后生成 4,755,456 行 × 139 列的提交文件，并逐项校验格式与单调性。

这个比赛给我们最大的启发是：**在不确定性量化任务里，「怎么输出」比「网络多深」更重要**。同样的 MLP，直接回归只能给出一个点估计；换成预测分布参数、再用集合近似分位数，就能同时拿到校准过的不确定性和物理上自洽的极端指数。而降水这种「大量零 + 少量极端」的变量，逼着我们放弃 MSE，改用零膨胀生成式建模——这也是气候模拟与普通回归任务最本质的区别。
