# ClimX 气候模拟挑战（Main Track）

## 一、比赛简介

### 1.1 比赛背景

Climate X（ClimX）是一场「气候模型模拟」比赛。物理学家用超级计算机跑气候模型（比如挪威的 NorESM2-MM），输入温室气体排放情景，就能输出未来几十年的气温、降水等数据。但这类模型跑一次要花巨额算力，于是比赛提出了一个新问题：**能不能训练一个「模拟器（emulator）」，快速近似出气候模型的输出？**

比赛分主赛道（Main Track）和轻赛道（Lite Track）。主赛道要求预测 NorESM2-MM 模型在 **SSP2-4.5 排放情景**下，**2015～2100 年**逐年的 **15 个极端气候指数**（比如极端高温 `TXx`、连续干旱天数 `CDD`、强降水日数 `R10mm` 等），覆盖全球 192×288 的网格。

一块地形真实的地球网格是这道题最大的难点：

1. **分辨率高**：192×288 = 55,296 个格点，86 年、15 个指标，输出规模巨大；
2. **逐格点评分**：评价指标在每个格点上独立计算，山区、海岸线、海陆交界处哪怕整体气候态偏了一点点，分数都会被狠狠扣掉；
3. **数据量巨大**：完整的历史输出文件有 31.9 GB，必须想尽办法压缩缓存。

### 1.2 比赛目标

官方提供两种粒度的数据：

- **lite 数据（12×18 粗网格）**：历史文件 `hist_train_1850_1931`、`hist_train_1932_2014`，三个训练情景 `train_ssp126/370/585`，以及 `test_forcings_ssp245`；
- **完整数据（192×288 细网格）**：全分辨率的 NorESM2-MM 历史输出 1932～2014。

输入特征分两类：

- 气候变量：`tas`（气温）、`tasmax`、`tasmin`、`pr`（降水）、`huss`、`psl`、`sfcWind`；
- 强迫数据：温室气体（`CO2_LBC`、`N2O_LBC`、`CH4_LBC` 等）和气溶胶（`SO2`、`BC_AX` 等）逐年序列。

我们要输出 15 个极端指数列（`TXx`、`TNn`、`SU`、`TR`、`FD`、`ID`、`WSDI`、`CSDI`、`GSL`、`Rx5day`、`CDD`、`CWD`、`R95p`、`R10mm`、`SDII`），提交文件按「时间 → 纬度 → 经度」展开成一张大表：每行是某个格点某年的 15 个值，外加一列 `row_id`。

### 1.3 评价标准

比赛使用 **区域加权 nNSE（region-wise normalized Nash–Sutcliffe Efficiency）**，计算分四步：

1. 对每个格点，计算「预测值随时间变化」与「真实值随时间变化」的 R²；
2. 把 R² 变换成 nNSE：`nNSE = R² / (2 − R²)`；
3. 按 AR6 陆地分区，在每个区域内用「余弦纬度权重」求平均；
4. 所有区域的分数再平均，最后对 15 个指标再平均。

简化理解：它衡量的是「我们预测的每一年变化」和「模型真实跑出来的变化」有多贴近，**逐格点、逐指标、逐区域**加权。分数越高越好，越接近 1 越好，负数说明预测比直接看平均气候态还差。

---

## 二、解题过程

我们采用的是一套经典的地球科学基线，叫 **Δ-mapping（增量映射）**，分为三个阶段：

1. **全球模型**：用温室气体 + 全球平均气溶胶做特征，岭回归预测逐年全球平均地表温度（GMST）；
2. **模式缩放**：每个格点、每个月、每个变量拟合 `mean = a + b × GMST`，得到「强迫变化场」；
3. **增量映射**：把「某一年的真实历史天气」叠加上「强迫变化」，得到未来每日天气，再由每日天气算出 15 个指数。

这套做法的核心信念是：**未来天气 ≈ 昨天天气的变异 + 人类排放强加的缓慢变化**。

### 2.1 数据探索（EDA）

作为一个数据管线的第一件事，我们先把 lite 数据找出来、把配置和变量名定下来：

```python
TARGETS = ['tas', 'tasmax', 'tasmin', 'pr', 'huss', 'psl', 'sfcWind']   # 要模拟的 7 个日变量
GHG     = ['CO2_LBC', 'N2O_LBC', 'CH4_LBC', 'CFC11eq_LBC', 'CF2CL2_LBC']  # 温室气体强迫
AER     = ['BC_AX', 'BC_N', 'SO2', 'SO4_PR', 'OM_NI']                     # 气溶胶强迫
INDEX_COLS = ['TXx', 'TNn', 'SU', 'TR', 'FD', 'ID', 'WSDI', 'CSDI', 'GSL',
              'Rx5day', 'CDD', 'CWD', 'R95p', 'R10mm', 'SDII']            # 要提交的 15 个指数
REF_YEARS   = np.arange(1961, 2015)      # 用于拼每日天气的历史年份池
BASE_PERIOD = (1961, 1990)               # 计算极端阈值的气候基准期
```

然后我们把所有情景数据读进来，先看形状和年份范围：

```python
def load_scenario(paths, with_targets=True):
    """读一个情景：返回经纬度、逐年/逐日时间、气候变量和逐年强迫"""
    dss = [xr.open_dataset(p, decode_times=False).isel(forcing_scenario=0) for p in paths]
    D = {'lat': dss[0].lat.values, 'lon': dss[0].lon.values}
    # 手动解析 noleap 日历，得到 (year, day_of_year)
    yd = [noleap_year_doy(x['time']) for x in dss]
    D['year'] = np.concatenate([a for a, _ in yd])
    D['doy']  = np.concatenate([b for _, b in yd])
    if with_targets:
        for v in TARGETS:
            D[v] = np.concatenate([x[v].values for x in dss]).astype(np.float32)
    # 气溶胶先取全球加权平均（用纬度余弦权重），温室气体直接保留年序列
    return D

H   = load_scenario(hist_paths)                   # 历史：1850~2014
SS  = {s: load_scenario(p) for s, p in train_paths.items()}  # 三个训练情景
TE  = load_scenario(test_path, with_targets=False)            # 测试：SSP2-4.5
```

打印结果显示历史数据跨越 1850～2014，测试数据是 SSP2-4.5、预测期覆盖 2015～2100 共 86 年，lite 网格是 12×18。我们还做了个边界检查：**每年必须是 365 天**（climx 用的是无闰年日历），最后一年可能因截断少一天，但不允许多于 365。这种检查能及早拦住解析 bug。

我们还有一步「EDA」性质的探针——预测全球平均温度在 2015→2100 的升温幅度：

```python
G_test = predict_gmst(gm, TE)
warming = G_test[-10:].mean() - G_test[:10].mean()
print('Predicted GMST change 2015->2100 (SSP2-4.5): %.2f K' % warming)
```

它把「整个 86 年的任务」浓缩成一个数字，也顺便校验：如果升温几度都对不上直觉，说明强迫数据的处理出了问题。

### 2.2 数据清洗

climx 的数据清洗在深度学习里很少见，它的重点不是去缺失值，而是**处理不友好的文件格式和巨大的体量**：

**第一，无闰年日历的手动解码。** 数据用 CF 标准时间格式（比如「days since 1850-01-01」），但 Kaggle 环境没装 `cftime`，很多函数读不了这种日历。我们手写了解码：

```python
MONTH_START = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])

def noleap_decode(units, values):
    """把 'days since ...' 时间戳解析成 (年, 年内第几天)"""
    _, ref = units.split(' since ')
    date, clock = (ref.split(' ') + ['00:00:00'])[:2]
    y, m, d = (int(x) for x in date.split('-'))
    # 无闰年：一年固定 365 天，用月份累计表换算
    days = y * 365 + MONTH_START[m - 1] + (d - 1) + np.asarray(values)
    days = np.floor(days + 1e-6).astype(np.int64)   # 容忍浮点舍入误差
    return days // 365, days % 365
```

**第二，量化和压缩缓存。** 完整分辨率历史文件有 31.9 GB，不可能每次都现读。我们把温度和气压存成 `int16`（0.01 K 的步长），降水等其他字段存成 `float16`，转成内存映射（memmap）的 `.npy` 缓存：

```python
CODEC = {   # 变量: (存储类型, 偏移, 缩放)，即 stored = (x - offset) / scale
    'tas': ('int16', 273.15, 0.01),      # 温度偏移到 273.15K，精度 0.01K
    'pr':  ('float16', 0.0, 1 / 86400),  # 降水转成 mm/s 量级
    ...
}

def build_cache(nc_path, cache_dir, years, variables):
    # 找年份范围，把选出的变量按 CODEC 量化写入 memmap
    for v in variables:
        mm = np.lib.format.open_memmap(tmp, mode='w+', dtype=CODEC[v][0], shape=...)
        mm[:] = np.round((values - off) / scale)
    np.savez(os.path.join(cache_dir, 'meta.npz'), year=..., lat=..., lon=...)
```

这样 31.9 GB 的原始文件被压成约 8.7 GB 的缓存，使用完后删除原始文件、以后直接读缓存，内存占用和运行时间都大幅下降。**「把不适合直接用的数据转成适合自己的缓存格式」正是这道题清洗环节的精髓。**

### 2.3 模型训练与超参数调优

**阶段一：全球平均温度模型。** 用每年的温室气体浓度和全球平均气溶胶，岭回归拟合该年的 GMST。`ridge_alpha` 控制正则强度，越大气温曲线越平滑：

```python
def fit_global(all_sets):
    years = []
    X, y = [], []
    for name, D in all_sets.items():
        # 特征：温室气体 + 气溶胶的当年值；标签：当年全球平均气温
        feats = np.column_stack([D['F'][g] for g in GHG + AER])
        gmst, _ = global_mean_temperature(D)   # 纬度余弦加权平均 tas
        years += list(D['F']['year']); X.append(feats); y.append(gmst)
    # 岭回归 = 带 L2 正则的最小二乘，alpha 越大曲线越稳
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(np.vstack(X), np.concatenate(y))
    return ridge
```

**阶段二：模式缩放。** 对每个格点、每个月份、每个变量，拟合一个简单回归：**该月气候均值 = a + b × 当年的 GMST**。它表达的是「全球每暖 1 度，这个格点这个月的温度该变多少」：

```python
for var in TARGETS:
    for c in cells:
        for m in months:
            # 从历史年份里收集 (GMST_t, 该格点该月该变量的值)
            gm = np.array([...])      # 全球平均温度序列
            vv = np.array([...])      # 该格点该月该变量的均值
            a, b = np.polyfit(gm, vv, 1)   # 1 次多项式即线性回归
            ps[var][c][m] = (a, b)
```

**阶段三：Δ-mapping 逐日天气模拟。** 对目标年 t，我们并非凭空生成天气，而是**从 1961～2014 里挑一个「参考年」，把它的逐日天气整体平移/缩放，使它的气候均值等于模式缩放预测出的值**：

```python
def apply_delta(var, base_daily, forced_change):
    # 降水等正量变量用「乘性」增量，温度用「加性」增量
    if var in POSITIVE:
        return base_daily * forced_change
    return base_daily + forced_change

def ref_year_for(target_year, member):
    # 每个 ensemble 成员选不同的参考年，最后把多个成员平均
    return REF_YEARS[(target_year * 7 + member * 11) % len(REF_YEARS)]
```

**Ensemble 平均。** 参考年份的选择有随机性——用 1963 年的天气还是 1998 年的天气，得到的单年值会有差异。我们把所有参考年都试一遍（`N_MEMBERS` 个成员），每个成员算一套指数，最后取平均。**平均掉的是「参考年份恰好偏冷/偏暖」的运气，剩下的是稳定的强迫信号**。这是主赛道提分最有效的手段之一。

**「清洗」贯穿始终的一个细节：** 计算极端指数前，必须用 1961～1990 基准期按「年内第几天 ± 5 天窗口」算分位数阈值（比如 90 分位是高温阈值、95 分位是强降水阈值）。阈值和天气一样，都来自真实历史，不掺人任何未来信息。

### 2.4 模型测试与指标报告

**验证策略。** 我们拿一个没参与训练的情景 **SSP3-7.0** 当验证集：用其他数据拟合模型，再在 SSP3-7.0 上算官方指标，评估「没见过的情景」下的泛化能力。材料里先报告全球模型在 SSP3-7.0 上的 GMST 误差（以 RMSE 记），再报告 15 指数的 rnNSE：

```python
def region_nnse(true, pred, lat, mask):
    """复刻官方 rnNSE.py：逐格点 R2 -> nNSE -> 区域余弦加权平均 -> 15 指数平均"""
    cw = np.cos(np.deg2rad(lat))[:, None] * np.ones(mask.shape)   # 纬度权重
    per = {}
    for k in INDEX_COLS:
        gt, p = true[k].astype(float), pred[k].astype(float)
        var = gt.var(0)                                   # 真实值随时间变化的方差
        # 每个格点单独算 R2
        r2 = 1 - ((gt - p) ** 2).mean(0) / np.maximum(var, eps)
        nn = r2 / (2 - r2)                                # R2 -> nNSE
        # 在 AR6 各区域内按余弦权重平均，再对区域平均
        vals = [region_mean(nn, mask, rc, cw) for rc in regions]
        per[k] = np.mean(vals)
    return np.mean(list(per.values())), per
```

**结果解读。** 材料记录了一个非常反直觉的现象：

- lite（12×18 粗网格）基线在本地验证集上能有 **+0.19** 左右的 rnNSE；
- 但简单地把粗网格结果上采样（upsample）到 192×288 后，排行榜分数暴跌到约 **−0.46**；
- 原因正如前面「难点」所说：上采样会错位所有格点的气候均值，山区和海岸线全都对不上，而官方指标是**逐格点**算 R² 的。

所以轻赛道改成**全分辨率直接做 Δ-mapping**，让每个格点都在自己的原生坐标上预测。这条全分辨率基线最终在排行榜拿到 **0.03368**。分数为正，说明预测确实比「每格点用常值气候态」要好，尽管提升还很小——这道题真正要拼的，就是那 0.03 之后的每一分。

**提交文件。** 指数已经位于原生 192×288 网格，按 时间→纬度→经度 展开成一行一个格点，加上 `row_id` 列，另存为 parquet：

```python
sub = pd.DataFrame({k: idx_pred[k].reshape(-1).astype(np.float32) for k in INDEX_COLS})
sub.insert(0, "row_id", np.arange(len(sub), dtype=np.int64))

# 行数必须 = 86 年 × 192 × 288，且不允许出现 NaN
assert len(sub) == len(years_out) * 192 * 288
assert np.isfinite(sub[INDEX_COLS].to_numpy()).all()

sub.to_parquet("submission.parquet", index=False, compression="zstd")
```

这两个断言里，行数断言防「维度错位」的灾难性 bug，`isfinite` 断言防把含 `NaN` 的表交上去——评委根本不会接受带空值的预测。

---

## 三、总结

本文中我们完成了 ClimX 主赛道的完整流程。我们首先用 EDA 摸清了 lite 与 full 两套数据：7 个日变量、5 种温室气体和 5 种气溶胶强迫、15 个极端指数、SSP2-4.5 情景 2015～2100 共 86 年 192×288 网格。其次我们做了针对性的「清洗」：手写 noleap 日历解码摆脱 cftime 依赖，把 31.9 GB 的历史文件量化压缩成 8.7 GB 的 memmap 缓存。接着我们建了三阶段模拟器——岭回归学全球增温、模式缩放学各格点对增温的响应、Δ-mapping 用真实历史天气叠加强迫变化，并用多参考年 ensemble 平均降低随机性。最后我们在 SSP3-7.0 保留集上复刻官方 rnNSE 验证，确认 lite 上采样到 full 是分数暴跌的原因后，改做全分辨率 Δ-mapping，拿到排行榜 0.03368 的基线。

这个比赛最大的启发是：**评估指标的数据粒度，决定了整个技术方案**。正因为指标是逐格点算 R² 的，上采样这种「偷懒」才会把分数打成负数；反过来，一旦我们用全分辨率原生格点做预测，即使方法一样，分数立刻转正。材料里还有用物理约束神经网络（PINN）做逐格点天气模拟的进阶方案，但它要处理的数据量和训练成本又是另一个量级——我们的 Δ-mapping 基线证明，了解地球物理的结构（天气=历史天气+强迫变化），很多时候比堆模型更管用。