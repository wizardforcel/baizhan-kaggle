# 使用 SARIMA 模型预测

## 一、比赛简介

### 1.1 比赛背景

这个比赛要求我们用 SARIMA 模型对多条时间序列做预测。数据中有一个 `train.csv` 和一个 `test.csv`。训练集包含多列：`Serie` 是序列编号，`y` 是观测值，`indice` 大概是序列内部的时间索引；测试集也包含 `Serie`、`indice` 等列，并且同样有 `y` 列待填。材料把每条序列按 `Serie` 分组，保存为训练序列字典 `series_train`。一共有多条序列，它们在趋势、季节性和尺度上差异很大。

SARIMA 是 ARIMA 的扩展。ARIMA 组合了自回归项、差分项和移动平均项，用来描述时间序列的自身规律；SARIMA 额外添加了季节项，可以表达周期性规律。材料的分析显示六条序列各具特点：第一条从 600 左右增长到 2500 左右，被趋势主导；第三条围绕 0 波动；第二、三、五、六条的滞后 12 期自相关较强，提示年度季节性。因此比赛难度在于不能让所有序列共用一套固定参数，而要为每条序列找到合适的季节周期和阶数。对入门者来说，这意味着要先把每条序列单独看待，而不是直接堆加模型。

### 1.2 比赛目标

我们输入的是按 `Serie` 分组的历史序列，输出是每条序列未来若干步的预测值。`horizons` 字典通过测试集分组统计得到：`{s: test.groupby('Serie').size()[s]}`，表示每条序列要预测多少个未来时间点。我们最终需要按测试集的顺序写出每一行的 `y` 预测值，并保存为 `sample_submission.csv`，保留 `id` 列和 `y` 列。材料中的代码用 `pred_df` 先复制测试表，再把预测值按 `Serie` 和 `indice` 对齐回正确位置。

### 1.3 评价标准

材料使用 MAE，也就是平均绝对误差，作为模型调参的目标函数。MAE 计算每个预测值和真实值之差的绝对值，再求平均，单位与序列本身相同。在 Optuna 优化中，材料把每条序列最后一段历史当作验证集，用验证集 MAE 选择最优超参数。MAE 越小越好。由于测试集没有公开标签，最终排行榜分数由比赛平台计算，我们在本文中只报告验证方式的 MAE 值，不编造最终榜分数，这符合材料给出的信息。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取训练集和测试集，确认序列数量、每条序列的训练长度和要预测的步数。材料把按 `Serie` 分组的训练序列存成字典，用测试集分组统计出每条序列的预测步数，并汇总成一张概览表。这一步确认了我们的输出规模：一共要为几条序列各生成对应的预测步数，任何一条写错都会让提交文件的行数或顺序对不上。

```python
# 读取数据，按 Serie 分组整理训练序列
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = '/kaggle/input/competitions/prediction-with-sarima-models/%s'
train = pd.read_csv(path % 'train.csv', encoding='utf-8-sig')
test = pd.read_csv(path % 'test.csv', encoding='utf-8-sig')

print('train shape:', train.shape)
print('test shape:', test.shape)
print('series in train:', train['Serie'].nunique())
print('series in test:', test['Serie'].nunique())
print(train.head())
```

接着我们把训练序列拆成字典，并统计每条序列的长度和测试集的预测步数。这样我们能知道每条序列有多少历史数据，以及未来需要预测多长。

```python
# 每条训练序列转成 numpy 数组，方便后续建模
series_train = {
    s: train.loc[train['Serie'] == s, 'y'].to_numpy(dtype=float)
    for s in sorted(train['Serie'].unique())
}

# 测试集按 Serie 统计样本数，即每条序列的预测步数
horizons = test.groupby('Serie').size().to_dict()
print('horizons:', horizons)

# 汇总每条序列的训练长度和预测步数
overview = pd.DataFrame(
    {'n_train': {s: len(y) for s, y in series_train.items()},
     'test_horizon': horizons}
)
overview.index.name = 'Serie'
print(overview)
```

我们其次画图观察每条序列的形状。材料强调序列差异很大：有的序列持续增长，有的围绕 0 波动，有的存在季节周期。我们还需要计算自相关函数 ACF 在候选周期 4、6、12 处的值，作为季节性的初步证据。ACF 可以理解为“当前值和若干步之前的值有多像”，数值越大说明周期性越明显；这一步让我们在调参前先对数据有一个直观判断。

```python
# 画出所有训练序列，观察趋势与波动
from statsmodels.tsa.stattools import acf

fig, axes = plt.subplots(3, 2, figsize=(13, 9))
for ax, (s, y) in zip(axes.ravel(), sorted(series_train.items())):
    ax.plot(y, lw=1.5)
    ax.set_title(f'Series {s}  (n={len(y)})')
    ax.set_xlabel('Index')
    ax.set_ylabel('y')
    ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# 计算滞后 4、6、12 的自相关，作为季节周期参考
rows = []
for s, y in sorted(series_train.items()):
    acf_vals = acf(y, nlags=24, fft=False)
    rows.append({'Serie': s,
                 'ACF lag 4': acf_vals[4],
                 'ACF lag 6': acf_vals[6],
                 'ACF lag 12': acf_vals[12]})
print(pd.DataFrame(rows).set_index('Serie').round(3))
```

从 ACF 表中我们看到，第二、三、五、六条序列在滞后 12 处的自相关较高，提示年度季节性；第一条被趋势主导；第四条自相关较弱。材料并不手动固定周期，而是让 Optuna 自己从候选周期中为每条序列选择 `m`，这样既保留先验知识又避免武断。

### 2.2 数据清洗

时间序列模型的输入相对简单，但我们必须保证序列顺序正确。材料中的代码没有对序列做插值或填充，而是直接按 `Serie` 过滤出 `y` 序列。我们首先确认测试集每条序列拥有完全相同的 `indice` 密度，若测试的 `indice` 没有跳过，则预测步数就等于测试行数。其次要保证预测结果在写回测试集时，按 `indice` 排序对齐，而不是按分组后字典遍历的先后顺序。

```python
# 检查测试集的 indice 是否连续，避免对齐错位
for s in sorted(test['Serie'].unique()):
    sub = test.loc[test['Serie'] == s, 'indice'].sort_values()
    diff = sub.diff().dropna().unique()
    print(f'Series {s}: n={len(sub)}, index diff unique={np.unique(diff)}')
```

对于这类序列数据，缺失值处理要格外小心。材料没有描述缺失值，代码直接用 `as_numpy` 读取，也没有可见的填充逻辑。通常做法是先补全序列缺失或判断是否等间隔时间序列。我们按照材料的证据采取保守写法：不擅自补插。

```python
# 检查训练序列是否含缺失或非有限数值
for s, y in series_train.items():
    bad = np.isnan(y).sum() + np.isinf(y).sum()
    print(f'Series {s}: invalid values = {bad}')

# 测试集也做同样的完整性检查
print('test invalid:', test['y'].isna().sum())
```

最后我们确定候选周期。材料认为季节周期可能为 4、6 或 12，同时保留 `m = 1` 表示无季节性。序列之间的尺度差异让我们推断数据可能存在周、月或年级别的规律，但不会手动指定，而是交给搜索选择。

```python
# 候选季节周期留给 Optuna 选择，m=1 代表没有季节项
candidate_m = [1, 4, 6, 12]
print('seasonal candidates:', candidate_m)
```

### 2.3 模型训练与超参数调优

我们首先定义 SARIMA 的目标函数。它接收试参数组合，把序列最后 `h` 步当作验证集，其余部分训练模型，再预测 `h` 步并计算 MAE。材料中加了两个保护措施：当阶数之和超过 6 时直接返回无穷大，避免过度复杂模型；当模型拟合失败时也返回无穷大，让 Optuna 自动丢弃无效组合。

```python
# 定义 SARIMA 超参数优化的目标函数
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_objective(trial, y, h):
    """用最后 h 步做验证，返回预测 MAE。"""
    # 候选季节周期，1 表示无季节性
    m = trial.suggest_categorical('m', [1, 4, 6, 12])
    p = trial.suggest_int('p', 0, 3)
    d = trial.suggest_int('d', 0, 1)
    q = trial.suggest_int('q', 0, 3)
    if m > 1:
        P = trial.suggest_int('P', 0, 2)
        D = trial.suggest_int('D', 0, 1)
        Q = trial.suggest_int('Q', 0, 2)
    else:
        P = D = Q = 0
    trend = trial.suggest_categorical('trend', ['n', 'c', 't', 'ct'])

    # 参数太多容易过拟合，直接放弃
    if p + q + P + Q > 6:
        return float('inf')

    y_train, y_val = y[:-h], y[-h:]
    try:
        model = SARIMAX(
            y_train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, m),
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=200)
        pred = np.asarray(res.forecast(steps=h))
        return float(np.mean(np.abs(pred - y_val)))
    except Exception:
        # 拟合失败也算无效组合
        return float('inf')
```

其次我们用 Optuna 对每条序列独立搜索。每条序列有各自的季节周期和趋势，因此价值取向不同。材料设定每序列最多 50 次试验、180 秒超时，并使用 TPE 采样器。

```python
# 对每条序列独立运行 Optuna 搜索
import optuna

SEED = 42
N_TRIALS = 50
TIMEOUT_PER_SERIES = 180
optuna.logging.set_verbosity(optuna.logging.WARNING)

studies = {}
for s, y in sorted(series_train.items()):
    h = horizons[s]
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        lambda t: sarima_objective(t, y, h),
        n_trials=N_TRIALS,
        timeout=TIMEOUT_PER_SERIES,
    )
    studies[s] = study
    print(f"Series {s}: validation MAE = {study.best_value:.4f} | "
          f"best params = {study.best_params}")
```

搜索完成后，我们汇总每条序列的最优参数和验证 MAE。材料还展示了手工修正：例如第 1 条序列最终使用 `m=4, p=2, d=1, q=2, P=1, D=1, Q=2, trend='ct'`，第 4 条序列最终使用 `m=12, p=4, d=1, q=0, P=2, D=1, Q=0, trend='c'`。这些值来自材料中的最终模型代码，我们在写报告时只引用它们的参数，不作为普适最优结果，但足够说明每条序列选择不同参数的重要性。

```python
# 汇总每序列最优参数
rows = []
for s, study in sorted(studies.items()):
    bp = study.best_params
    m = bp['m']
    rows.append({
        'Serie': s,
        'order': f"({bp['p']},{bp['d']},{bp['q']})",
        'seasonal_order': (f"({bp.get('P',0)},{bp.get('D',0)},"
                           f"{bp.get('Q',0)},{m})") if m > 1 else '-',
        'trend': bp['trend'],
        'MAE_validation': round(study.best_value, 4),
        'n_trials': len(study.trials),
    })
best_params_table = pd.DataFrame(rows).set_index('Serie')
print(best_params_table)
```

### 2.4 模型测试与指标报告

我们最后用完整的训练序列重新拟合 SARIMA，并用 `get_forecast` 得到 `h` 步的预测均值和 95% 置信区间。材料对第 1 条和第 4 条序列使用手工确认的最佳参数，其余用 Optuna 搜索结果。这个步骤说明，调参结果并不总是一步到位，还需要结合可视化和领域知识微调。

```python
# 用完整序列拟合最终模型并预测未来 h 步
forecasts = {}

for s, y in sorted(series_train.items()):
    bp = studies[s].best_params
    if s == 1:
        bp = {'m': 4, 'p': 2, 'd': 1, 'q': 2,
              'P': 1, 'D': 1, 'Q': 2, 'trend': 'ct'}
    if s == 4:
        bp = {'m': 12, 'p': 4, 'd': 1, 'q': 0,
              'P': 2, 'D': 1, 'Q': 0, 'trend': 'c'}

    m = bp['m']
    order = (bp['p'], bp['d'], bp['q'])
    seasonal_order = (bp.get('P', 0), bp.get('D', 0),
                      bp.get('Q', 0), m)
    h = horizons[s]

    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        trend=bp['trend'],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=300)
    fc = res.get_forecast(steps=h)
    pred_mean = np.asarray(fc.predicted_mean)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    forecasts[s] = pred_mean

    print(f'Series {s}: order={order} seasonal={seasonal_order} '
          f'AIC={res.aic:.1f} forecast head={pred_mean[:3]}')
```

然后我们测试预测范围是否合理，并检查是否有缺失。比如第 1 条序列原本在 600 到 2500 之间，预测值不应出现极端的负值或无穷大。我们还可以把历史曲线和预测曲线画在同一张图里，观察预测是否与历史趋势衔接。

```python
# 检查每序列预测的基本统计
for s, pred in forecasts.items():
    print(f'Series {s}: n_pred={len(pred)}, min={pred.min():.3f}, '
          f'max={pred.max():.3f}, nan={np.isnan(pred).sum()}')

# 绘制某条序列的历史与预测对比图
s = 1
y = series_train[s]
fc = forecasts[s]
plt.figure(figsize=(10,5))
plt.plot(np.arange(len(y)), y, label='Train', lw=1.5)
plt.plot(np.arange(len(y), len(y)+len(fc)), fc, label='Forecast',
         color='orange', lw=1.8)
plt.axvline(len(y)-0.5, color='gray', ls='--', lw=0.8)
plt.title(f'SARIMA forecast for Series {s}')
plt.legend()
plt.show()
```

最后我们把预测值写回测试表。为了保证顺序正确，我们为每条序列的测试行按 `indice` 排序后再填入预测，再输出 `id` 和 `y` 两列。材料最终保存为 `sample_submission.csv`。

```python
# 把预测值按 indice 对齐写回测试表
pred_df = test.copy()
pred_df['y'] = np.nan
for s, pred_mean in forecasts.items():
    idx = pred_df.index[pred_df['Serie'] == s]
    row_order = pred_df.loc[idx].sort_values('indice').index
    pred_df.loc[row_order, 'y'] = pred_mean

# 保留比赛要求的 id 与预测值列，输出提交文件
pred_df[['id', 'y']].to_csv(
    'sample_submission.csv', index=False, encoding='utf-8-sig'
)
print(pred_df[['id', 'Serie', 'indice', 'y']].head(10))
print('submission rows:', len(pred_df))
print('any missing y:', pred_df['y'].isna().sum())
```

## 三、总结

本文中我们完成了 SARIMA 时间序列预测的完整流程。我们首先读取训练和测试数据，按 `Serie` 分组后统计每条序列的训练长度与预测步数，借助曲线和滞后 4、6、12 期的自相关判断趋势与季节性。其次我们确认序列没有缺失和顺序错位后，定义 SARIMA 目标函数，用最后 `h` 步做验证计算 MAE，并对阶数和季节周期设置范围。然后我们用 Optuna 对每条序列分别搜索最优超参数，对个别序列再根据观察手工微调。最后我们用完整序列拟合 SARIMA，得到未来 `h` 步的预测均值和置信区间，并按照 `Serie` 和 `indice` 把预测值对齐写入提交文件。

这个比赛给我们的主要启发是，时间序列模型不能采用“一把钥匙开所有锁”的通用配置。不同序列的趋势、季节性和波动尺度差异很大，为每条序列独立搜索参数并配合绘制历史与预测图进行校验，是保证预测合理性的关键。同时，验证集的构造必须贴近真实任务：用最后 `h` 步当验证，正好模拟我们未来要做的事情，这样调参才有意义。对入门者来说，先理解序列的图形和自相关，再动手调参，往往比直接运行一个默认模型更可靠。