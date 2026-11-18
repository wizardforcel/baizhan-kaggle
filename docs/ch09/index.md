# 第9章 时序数据比赛详解：门店销售预测（Rossmann Store Sales）

Rossmann Store Sales 是 Kaggle 上经典的时序回归比赛。Rossmann 是一家在德国运营超过 3000 家药店的连锁企业，比赛提供 1115 家门店 2006 年至 2014 年的日销售额历史数据，任务是在测试期对未来六周的销售额进行预测。

这个比赛的数据包含 `train.csv`、`test.csv` 和 `store.csv` 三份文件。门店特征包含类型、商品组合、竞争距离、促销信息、州假日等，交易特征包含周几、是否营业、促销、学校假期等。评价指标是 **RMSPE（均方根百分比误差）**，对低销量样本更敏感，模型需要同时拟合大店和高峰日和小店和平日。

与房价预测（普通表格回归）不同，这是一个**时序回归**任务，数据有明显的时间顺序、周周期和年度节假日波动，必须按时间切分验证集，防止未来信息泄漏。

本章我们完整走一遍 Rossmann 比赛的解题流程：EDA → 清洗（关店日处理、缺失值、日期拆分、PromoInterval 拆分） → 特征工程（门店静态特征 + 日期动态特征） → 按时间切分验证 → XGBoost 建模（对数目标 + RMSPE） → 提交。

---

## 9.1 比赛背景与数据

### 比赛背景

Rossmann 是一家在德国运营超过 3000 家药店的连锁企业。比赛提供 1115 家门店 2006 年至 2014 年的日销售额历史数据，任务是在测试期对未来六周的销售额进行预测。

### 数据结构

比赛数据包含三份文件：

| 文件 | 含义 |
|------|------|
| `train.csv` | 训练集，约 1,017,209 行，包含日期、门店、销售额、客流量、是否营业、促销、学校假期、州假日等 |
| `test.csv` | 测试集，约 41,088 行，没有 `Sales` 列，需要预测 |
| `store.csv` | 门店静态信息，约 1115 行，包含门店类型、商品组合、竞争距离、竞争开业日期、促销区间等 |

训练集和 `store.csv` 通过 `Store` 列合并，得到每个门店每天的完整特征。

### 评价指标

比赛使用 **RMSPE（Root Mean Squared Percentage Error，均方根百分比误差）**：

$$
\text{RMSPE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} w_i \cdot (y_i - \hat{y}_i)^2}
$$

其中 $w_i = 1/y_i^2$（当 $y_i \neq 0$ 时），对低销量样本更敏感。这意味着模型不仅要拟合大店和高峰日，也要关注小店和平日，不能只顾着大单子。

---

## 9.2 探索性数据分析（EDA）

先读取数据并合并：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('/kaggle/input/rossmann-store-sales/train.csv')
test  = pd.read_csv('/kaggle/input/rossmann-store-sales/test.csv')
store = pd.read_csv('/kaggle/input/rossmann-store-sales/store.csv')

# 合并门店静态信息
df = train.merge(store, how='left', left_on='Store', right_on='Store')

print("训练集形状:", df.shape)   # (1017209, ...)
print("测试集形状:", test.shape)  # (41088, ...)
```

### 基本统计

```python
print(df['Sales'].describe())
```

基本描述显示 `Sales` 的均值约 5773，标准差 3849，最大值 41551，且存在 0 销量，说明门店可能关店或促销期异常。

### 检查 Open 字段

```python
# Open=0 的行 Sales 恒为 0
print(df.groupby(df['Open'])['Sales'].mean())
```

发现 `Open=0` 的行 Sales 恒为 0，说明门店关店当天没有销售。促销 `Promo` 占约 38% 的天数，学校假期占 18%。

### Sales 与 Customers 的关系

```python
temp_df = df[df['Open'] == 1]
sns.scatterplot(data=temp_df, x='Sales', y='Customers', hue='Promo', alpha=0.3)
plt.title("Sales vs Customers colored by Promo")
plt.show()
```

Sales 与 Customers 高度相关，且门店类型与竞争距离存在差异。

### 日期范围和周期性

```python
df['Date'] = pd.to_datetime(df['Date'])
print(df['Date'].min(), df['Date'].max())  # 2006-07-01 至 2015-07-31
```

日期范围覆盖 2007-07-01 至 2015-07-31，测试集在 2015-08-01 之后。时间序列显示明显的周周期和年度节假日波动。

---

## 9.3 数据清洗

### 关店日处理

关店日的 Sales 被设为 0，我们保留这些行但在训练时通过 `Open` 标记区分。通常只训练营业日（`Open == 1`）：

```python
df = df[df['Open'] == 1]  # 只训练营业日
```

### CompetitionDistance 缺失处理

`CompetitionDistance` 中有 172817 个 0 值，我们用 NaN 标记，让 XGBoost 自己处理缺失：

```python
df['CompetitionDistance'] = df['CompetitionDistance'].replace(0, np.nan)
```

### 类别变量编码

门店类型和商品组合是类别变量，先用 LabelEncoder 编码：

```python
from sklearn.preprocessing import LabelEncoder

store = store.copy()
store['StoreType'] = LabelEncoder().fit_transform(store['StoreType'].astype(str))
store['Assortment'] = LabelEncoder().fit_transform(store['Assortment'].astype(str))
```

### 日期拆分

对于日期，拆分为年、月、日：

```python
df['Date'] = pd.to_datetime(df['Date'])
df['Day']   = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year']  = df['Date'].dt.year
```

### PromoInterval 拆分

`PromoInterval` 是形如 `Jan,Apr,Jul,Oct` 的字符串，我们把它拆成月份标志：

```python
monthToNum = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
              'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

join_with = store['PromoInterval'].str.split(',', expand=True)
store['0_PromoInterval'] = join_with[0].map(lambda x: monthToNum(x) if pd.notna(x) else np.nan)
# 可以继续拆分第1、2、3个月份...
```

### 竞争开业时间和促销开始时间

竞争开业时间和促销开始时间被统一转换为整数日期：

```python
store['CompetitionOpen'] = store['CompetitionOpenSinceYear'].astype(str) + \
                           store['CompetitionOpenSinceMonth'].astype(str).str.zfill(2)
df['DateInt'] = df['Date'].dt.strftime('%Y%m%d').astype(int)
```

---

## 9.4 特征工程

特征包括门店静态特征和日期动态特征：

```python
features = [
    'Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'StateHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpen',
    'PromoSince', '0_PromoInterval', 'Day', 'Month', 'Year', 'DateInt'
]

X_train = df[features]
```

### 特征说明

| 特征类别 | 具体特征 | 说明 |
|---------|---------|------|
| 门店静态特征 | Store, StoreType, Assortment, CompetitionDistance | 门店本身的属性，随时间基本不变 |
| 日期动态特征 | DayOfWeek, Day, Month, Year, DateInt | 捕捉周周期、月度周期、年度趋势 |
| 促销特征 | Promo, SchoolHoliday, StateHoliday, PromoInterval | 促销和假期对销量的影响 |
| 竞争特征 | CompetitionDistance, CompetitionOpen | 竞争对手距离和开业时间 |

---

## 9.5 模型训练与调优

### 目标变换

Sales 右偏，我们对目标取对数，让分布更对称：

```python
y = np.log1p(df['Sales'])
```

### 模型选择：XGBoost 回归

```python
import xgboost as xgb
from sklearn.model_selection import KFold

model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42
)
```

### 自定义 RMSPE 评估函数

```python
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1.0 / (y[ind] ** 2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe
```

### 按时间切分验证集

时序数据不能随机切分验证集，必须按时间顺序切（训练在前，验证在后），防止数据泄漏：

```python
# 例如：用前 80% 时间作为训练，后 20% 作为验证
split_idx = int(0.8 * len(df))
X_train_part = X_train.iloc[:split_idx]
y_train_part = y.iloc[:split_idx]
X_val_part   = X_train.iloc[split_idx:]
y_val_part   = y.iloc[split_idx:]

model.fit(
    X_train_part, np.log1p(df.iloc[:split_idx]['Sales']),
    eval_set=[(X_val_part, np.log1p(df.iloc[split_idx:]['Sales']))],
    early_stopping_rounds=50,
    verbose=False
)
```

### 超参数

超参数初步设为 `n_estimators=1000, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8`。可以通过 5 折时间序列交叉验证取平均，提升稳定性：

```python
kf = KFold(n_splits=5, shuffle=False)  # 时序数据不 shuffle
oof = np.zeros(len(df))
for tr_idx, val_idx in kf.split(df):
    model.fit(df.iloc[tr_idx][features], np.log1p(df.iloc[tr_idx]['Sales']))
    oof[val_idx] = np.expm1(model.predict(df.iloc[val_idx][features]))
```

---

## 9.6 模型测试与提交

### 在验证集上计算 RMSPE

```python
pred = np.expm1(model.predict(X_val_part))
rmspe_val = rmspe(pred, np.expm1(y_val_part))
print('Validation RMSPE:', rmspe_val)
```

### 对测试集预测

最终用全部训练数据拟合模型，对测试集预测：

```python
# 合并测试集与门店信息，构造相同的特征
test = test.merge(store, how='left', left_on='Store', right_on='Store')
test['Date'] = pd.to_datetime(test['Date'])
test['Day']   = test['Date'].dt.day
test['Month'] = test['Date'].dt.month
test['Year']  = test['Date'].dt.year
# ... 其他特征工程与训练集保持一致

X_test = test[features]
model.fit(X_train, np.log1p(df['Sales']))
test_pred = np.expm1(model.predict(X_test))

submission = pd.DataFrame({'Id': test['Id'], 'Sales': test_pred})
submission.to_csv('submission.csv', index=False)

# 核对清单
assert len(submission) == len(test), "提交行数与测试集不一致"
assert submission['Sales'].isnull().sum() == 0, "预测结果包含缺失值"
assert (submission['Sales'] >= 0).all(), "销售额不能为负"
print("提交文件已生成：submission.csv")
```

从材料看，使用特征工程后的 XGBoost 模型在公榜上属于前 1% 的水平，验证集 RMSPE 明显低于基线。

---

## 9.7 小结

本章我们以 Rossmann 门店销售预测为例，完整解析了一场时序回归比赛：

1. **比赛背景**：1115 家门店 2006~2014 年日销售额，预测未来六周销售额，评价指标为 RMSPE（对低销量更敏感）。
2. **EDA**：发现 Sales 均值约 5773，存在 0 销量（关店日），明显的周周期和年度节假日波动，Promo 占约 38% 天数，Sales 与 Customers 高度相关。
3. **数据清洗**：关店日只训练营业日，`CompetitionDistance` 中 0 值替换为 NaN，日期拆分为年/月/日，`PromoInterval` 字符串拆成月份标志，竞争开业时间和促销开始时间转换为整数日期。
4. **特征工程**：门店静态特征（类型、商品组合、竞争距离）与日期动态特征（周几、月份、年份、促销、假期）的组合。
5. **模型训练**：目标取对数（`log1p`），XGBoost 回归，自定义 RMSPE 评估函数，**按时间切分验证集**防止数据泄漏，超参数 `n_estimators=1000, max_depth=8, learning_rate=0.05` 等。
6. **模型测试与提交**：预测后用 `expm1` 还原销售额，5 折时间序列交叉验证取平均提升稳定性，上传前核对行数、列名、缺失值和负值。

这个比赛说明了在时间序列零售预测中，**特征工程和目标变换比单纯增加模型复杂度更关键**，而且**按时间切分验证集是防止未来信息泄漏的前提**——否则验证分数会好得虚假，排行榜上会崩盘。

下一章我们将详解另一个时序比赛：交通流量预测（TrafficFlowBench），看看在另一个时序场景下，上述流程有什么共通之处和不同之处。
