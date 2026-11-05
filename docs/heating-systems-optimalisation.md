# Heating Systems Optimalisation

## 一、比赛简介

### 1.1 比赛背景
这是一个家庭供暖系统的最优设备选型比赛。比赛给出一栋房屋一整年的逐小时天气和用电负荷数据，要求参赛者为三种不同的设计目标分别选出一套设备方案。评分用的模拟器是隐藏的，但主办方提供了完整的物理规格说明 `physics_spec.md`，以及十个参考设计的精确答案 `reference_results.csv`，可以用来校验自己写的模拟器是否正确。

这套系统包含的设备很多：空气源热泵、光伏（PV）板、PVT 光伏热板、电热水箱、气体锅炉、季节性储热罐（MgSO4 和石灰石储热）、太阳能集热器、蓄电池，以及一辆每年耗电约 4,000 kWh 的电动汽车。比赛的难点不是训练一个模型，而是把物理规则用逐小时的仿真模拟准确复刻出来，再在巨大的组合空间里搜索最优配置。

### 1.2 比赛目标
输入数据 `train_2024.csv` 包含 2024 年整年的逐小时序列：室外温度 `T_out_C`、屋顶单位面积太阳辐照 `g_roof35_Wm2`、空间采暖需求 `space_heat_kwh`、生活热水需求 `hot_water_kwh`、家庭用电 `household_el_kwh`、地板供暖供水温度 `supply_temp_C`、电动汽车是否在家 `car_home` 以及行驶耗电 `car_drive_kwh`。设备目录 `catalogue.csv` 列出了所有可选设备及其单位。

比赛需要为三个目标各设计一套方案，三套方案可以不同：

- 设计 A —— 最省钱：10 年总成本最低；
- 设计 B —— 2 万预算：设备投资额不超过 20,000 欧元，且年采购能量（从电网买电 + 买天然气）最少；
- 设计 C —— 生态上限：设备投资额不超过 100,000 欧元，且年采购能量最少。

每套方案用一个固定格式的向量提交，包含全部 13 个设备字段的值，没选用的设备写 0。提交格式要求 `pub`（2024 公测年）和 `priv`（隐藏年份）两块完全相同，每块 13 个值。

### 1.3 评价标准
比赛的定义是让参与者把隐藏模拟器跑得到的“真实分数”最小化。主办方给出的期望公共分数计算方式是：

```
score = (cost10_A / cost10_nothing
       + bought_B / bought_nothing
       + bought_C / bought_nothing) / 3
```

其中 `cost10` 表示最终方案 10 年的总成本（设备投资 + 10 × 年电费燃气费账单），`bought` 表示年采购能量（电网购电 kWh + 天然气 kWh）。分母用的是什么都不装方案的数值。因此分数是一个相对比例：完全不装设备分数为 1，分数越低代表三个目标平均优化得越好。第一名通常在 0.3 以下。

## 二、解题过程

由于评分模拟器是隐藏的，我们无法直接“训练”模型，解题思路变成三步：先把物理规则逐小时写成仿真器，然后用参考答案验证仿真器对错，最后用网格搜索在这些设备组合上寻找最优解并用能量流向图辅助分析。

### 2.1 数据探索（EDA）
我们先读取逐小时气象数据，并提取月份、小时、年内天数等时间特征：

```python
import os, numpy as np, pandas as pd

IN = '/kaggle/input/competitions/heating-systems-optimalisation'
df = pd.read_csv(f'{IN}/train_2024.csv', parse_dates=['time_utc'])
cat = pd.read_csv(f'{IN}/catalogue.csv')
ref = pd.read_csv(f'{IN}/reference_results.csv')

# 添加时间特征，便于按月、按小时查看负荷分布
df['month'] = df.time_utc.dt.month
df['hour'] = (df.time_utc.dt.hour + 1) % 24
df['doy'] = df.time_utc.dt.dayofyear

print(df.shape)
print(cat.head())
print(ref.head())
```

查看数据后我们发现：空间采暖负荷集中在冬季，生活热水全年都有，家庭用电昼夜差异明显，而太阳辐照显然只有白天和晴天才有。这意味着光伏发的电如果不配蓄电池或电解水储热，就无法覆盖傍晚的用电高峰。我们还发现行驶耗电集中在几个固定时段，说明电动汽车在工作日的用车习惯比较规律。

参考结果表给出十个设计（R1～R10）的逐行答案，比如纯电网自购、7 kW 热泵、7 kW 热泵加 5 kWp 光伏等。我们用这些答案来校准仿真器。

### 2.2 数据清洗
这份数据没有缺失值，也不需要对负荷做清洗，因为它直接来自真实参考系统。清洗主要发生在“设备参数”和“物理常数”层面。

热泵的能效比 COP 是一个必须精确建模的关键参数。规格书规定 COP = η × 卡诺效率，两个换热器各取 5 K 温差，η 在数据表点之间线性插值，随供水温度的升高而下降：

```python
def eta(t_src, t_sink):
    """从数据表点插值出 35℃ 和 55℃ 两档的换热器效率"""
    e35 = np.interp(t_src, [-7, 2, 7], [0.4998, 0.6042, 0.6419])
    e55 = np.interp(t_src, [-7, 7],    [0.4387, 0.5554])
    f = min(max((t_sink - 35) / 20, 0), 1)   # 供水温度越高越接近 55℃ 档
    return (1 - f) * e35 + f * e55

def cop(t_src, t_sink):
    """空气源热泵的 COP：η × 卡诺效率，两侧各加 5 K 温差"""
    th, tc = t_sink + 5 + 273.15, t_src - 5 + 273.15
    return max(1.0, eta(t_src, t_sink) * th / max(th - tc, 1.0))
```

我们把这种做法视为“清洗”，因为参数插值如果不按规格书严格做，仿真结果就会完全错误。

### 2.3 模型训练与超参数调优
真正的“模型”是逐小时的能量平衡仿真器。它的逻辑是：优先让热泵供生活热水（源温 55 ℃）和空间采暖，不足的部分交给老锅炉补足；然后处理电动汽车充电；最后用光伏、蓄电池和电网之间的功率平衡来结算电量和电费：

```python
def simulate(hp_kw=0.0, pv_kwp=0.0, battery_kwh=0.0):
    """逐小时仿真一套设备组合，返回年度汇总指标"""
    imp = exp = gas = 0.0      # 电网购电、售电、天然气
    b = 0.0                    # 蓄电池电量
    car = 30.0                 # 汽车电量
    for i in range(len(T)):
        cap = hp_kw
        dw, sh = DW[i], SH[i]
        # 先用热泵满足生活热水，再满足空间采暖
        for need, tk in ((dw, 55.0), (sh, TS[i])):
            q = min(need, cap)
            if q > 0:
                imp  # 热泵耗电按 q/COP 计
                cap -= q
                if tk == 55.0: dw -= q
                else:          sh -= q
        gas += (dw + sh) / 0.888              # 老锅炉补足，效率 88.8%
        car -= DRV[i]                          # 行驶耗电
        need = min(11.0, 25.0 - car) if HOME[i] and car < 25.0 else 0.0
        s = pv_kwp * G[i] * 0.85 - EL[i] - e_hp - need
        if s > 0:                              # 光伏有剩余，先充电池再卖电
            u = min(s, 0.5 * battery_kwh, (battery_kwh - b) / 0.95)
            b += u * 0.95
            exp += s - u
        else:                                  # 光伏不足，从电池取、从电网买
            u = min(-s, 0.5 * battery_kwh, b * 0.95)
            b -= u / 0.95
            imp += -s - u
    return dict(cost10=..., bought=imp + gas, ...)
```

我们用参考答案校验仿真器。材料给出的示例是：纯电网自购 R1、7 kW 热泵 R2、7 kW 热泵加 5 kWp 光伏 R3，对比 `grid_import_kwh`、`export_kwh`、`gas_kwh` 与 `reference_results.csv` 是否一致。R4 以后需要热水箱才能正确模拟，所以我们在实现水箱模型前把它留作练习。

参数调优在这里表现为设备组合的搜索。我们先用网格搜索扫描热泵和光伏：

```python
# 遍历热泵 0~8 kW、光伏 0~13 kWp、电池 0~10 kWh 的组合
grid = [(h, p, bt) for h in (0,3,4,5,6,7,8)
        for p in (0,2,4,6,8,10,13) for bt in (0,5,10)]
res = {g: simulate(*g) for g in grid}

# 分别选出三个目标的最优组合
A = min(grid, key=lambda g: res[g]['cost10'])
B = min([g for g in grid if res[g]['capex'] <= 20000],
        key=lambda g: res[g]['bought'])
C = min([g for g in grid if res[g]['capex'] <= 100000],
        key=lambda g: res[g]['bought'])
```

材料还给出了一套典型方案作为参考：A 选 4 kW 热泵 + 2 kWp 光伏；B 选 3 kW 热泵 + 4 kWp 光伏 + 5 kWh 电池；C 选 7 kW 热泵 + 13 kWp 光伏 + 10 kWh 电池。我们的网格搜索应当给出接近这些坐标的结果。网格搜索结束后，如果要进一步提高，可以把热水箱、MgSO4（5.3 欧元/kWh）和石灰石（3.7 欧元/kWh）储热罐也纳入组合，并套用智能 48 小时控制器决定何时用便宜的储热供热水。

### 2.4 模型测试与指标报告
我们按比赛公式计算期望公共分数，并输出每个目标的成本明细：

```python
nothing = simulate()               # 什么都不装的基线
score = (res[A]['cost10'] / nothing['cost10']
       + res[B]['bought']  / nothing['bought']
       + res[C]['bought']  / nothing['bought']) / 3
print('expected public score:', round(score, 4))
```

我们还用能量流向可视化检查系统的合理性：光伏的每一度电去了哪里、电网供了多少、燃气锅炉烧了多少，以及电池和储热罐的充放电曲线。材料中的 `system visualiser` 把整年能量按“能源输入 → 采集 → 电力 → 转换 → 储热 → 使用/输出”六层绘制成桑基图，并用月度曲线展示储罐的填充与放空。这一步能发现时序上的 bug，比如发现电池在夜间莫名放电、热泵 COP 在寒冬掉到接近 1 之类的异常。

最后我们把三套设计写入标准提交格式：

```python
KEYS = list(cat.name)
def design(g):
    d = {k: 0.0 for k in KEYS}          # 未选择的设备填 0
    d.update(hp_kw=g[0], pv_kwp=g[1], battery_kwh=g[2])
    return d
sub = pd.DataFrame({
    'id': [f'{p}{t}_{k}' for p in ('pub_','priv_') for t in 'ABC' for k in KEYS],
    'value': [design(D[t])[k] for p in ('pub_','priv_') for t in 'ABC' for k in KEYS]
})
sub.to_csv('submission.csv', index=False)
```

提交后，我们以期望公共分数作为量化指标报告：分数越低越好，为 1 表示和什么装备都不买一样，任何小于 1 的方案都说明优化有效。

## 三、总结
本文中我们完成了 Heating Systems Optimalisation 的完整流程。我们首先用 EDA 分析了逐小时的气象与负荷数据，确认采暖、热水、用电和太阳辐照的时序特征；其次按规格书精确实现了热泵 COP 插值并把它纳入仿真的“清洗”步骤；接着构建了逐小时能量平衡仿真器，用参考答案做校验，再通过网格搜索选出三套方案；最后用比赛公式计算期望公共分数，并用桑基图、月度储罐曲线做合理性检查，按标准格式生成了提交文件。

这个比赛的核心启示是：对一个隐藏模拟器来描述的系统，最可靠的“模型”往往不是机器学习模型，而是忠于物理规则的确定性仿真器；先把仿真器在校验集上算准，之后的搜索优化才有意义。