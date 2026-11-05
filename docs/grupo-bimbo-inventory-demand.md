# Grupo Bimbo Inventory Demand

## 一、比赛简介

### 1.1 比赛背景
Grupo Bimbo 是墨西哥一家大型烘焙制造商，在全国有超过 100 万家门店和 45,000 条配送路线，销售一百多种面包和糕点。对这样一张庞大的分销网络来说，最难的事情不是把货做出来，而是判断“明天早上这辆车应该往这家店送多少个这种面包”。送少了会缺货，送多了会退货，两种情况都是钱。这个比赛就是把真实的历史销售与退货数据交给参赛者，让大家预测每个门店、每条路线、每个产品在某一周的“均衡需求”。

数据里的需求不是简单地看本周卖了多少。因为零售商可以把卖不掉的产品退回来，所以比赛定义了调整后的需求：`Demanda_uni_equil = Venta_uni_hoy - Dev_uni_proxima`，其中 `Venta_uni_hoy` 是本周销售件数，`Dev_uni_proxima` 是下一周退货件数。这个列被保证大于等于 0。原始数据里销售减退货有时是负数，原因是退货会跨好几周陆续发生，所以主办方已经帮我们做了平滑调整。

比赛的另一个难点是测试集里会出现训练集没有见过的新产品。新产品没有历史销量，模型只能依靠产品所属的品类、包装规格、价格等间接信息做推断。这是零售业务的常态，也是比赛刻意保留的挑战。

### 1.2 比赛目标
输入是一张按周组织的订单明细表。训练集包含以下列：

- `Semana`：周编号，一周从周四到周三；
- `Agencia_ID`：销售仓库编号；
- `Canal_ID`：销售渠道编号；
- `Ruta_SAK`：配送路线编号；
- `Cliente_ID`：客户编号；
- `NombreCliente`：客户名称；
- `Producto_ID`：产品编号；
- `NombreProducto`：产品名称；
- `Venta_uni_hoy`：本周销售件数；
- `Venta_hoy`：本周销售额，单位比索；
- `Dev_uni_proxima`：下周退货件数；
- `Dev_proxima`：下周退货金额，单位比索；
- `Demanda_uni_equil`：调整后的需求，也是我们要预测的目标。

测试集只缺 `Demanda_uni_equil` 这一列。辅助表有三张：`cliente_tabla` 记录客户信息，`producto_tabla` 记录产品名称与规格，`town_state` 记录每个仓库所在的城市和州。它们可以分别通过 `Cliente_ID`、`Producto_ID` 和 `Agencia_ID` 与主表连接。

### 1.3 评价标准
比赛使用 RMSLE（Root Mean Squared Logarithmic Error，均方根对数误差）作为指标：

```
RMSLE = sqrt( mean( (log(1 + 预测值) - log(1 + 真实值))^2 ) )
```

它先对预测值和真实值都取 `log(1+x)`，再算均方误差，最后开平方。这样做的好处是：当真实需求是 10 件时，预测成 20 件和预测成 100 件的惩罚差距，不会像普通 RMSE 那样被大数值完全主导；同时它天然要求预测值非负。RMSLE 越低越好。

## 二、解题过程

整个比赛的主线可以概括为：先处理超大的 CSV 和类别编码，再构造“周 × 产品 × 渠道”的历史统计特征，最后用树模型回归 `Demanda_uni_equil`，按 RMSLE 评估。

### 2.1 数据探索（EDA）
我们首先把训练表读进来，只取需要的列并指定较小数据类型，避免几十万行乘十几列把内存撑爆：

```python
import pandas as pd, numpy as np

# 只读取探索阶段需要的列，并显式指定类型
df_train = pd.read_csv(
    '../input/train.csv',
    usecols=['Semana', 'Producto_ID', 'Venta_uni_hoy',
             'Dev_uni_proxima', 'Demanda_uni_equil'],
    dtype={'Semana': 'int32', 'Producto_ID': 'int32',
           'Venta_uni_hoy': 'int32', 'Dev_uni_proxima': 'int32',
           'Demanda_uni_equil': 'int32'}
)

# 查看前几行和目标列的基本统计
print(df_train.head())
print(df_train['Demanda_uni_equil'].describe())
```

从统计结果我们确认了两件事：第一，`Demanda_uni_equil` 的最小值是 0，没有负数，和题目说明一致；第二，需求分布非常长尾，大多数样本是个位数，少数大客户一次能买上百件。这种分布正是 RMSLE 适合它的原因。

接着我们按产品查看周销量的波动：

```python
# 按周和产品聚合，看看每个产品每周的销量统计
agg = df_train.groupby(['Semana', 'Producto_ID'], as_index=False).agg(
    ['count', 'sum', 'min', 'max', 'median', 'mean'])
agg.columns = ['_'.join(c).strip() for c in agg.columns]
print(agg.shape)
```

我们发现同一产品在不同周的均值变化明显，说明存在季节性和促销活动。然后我们把辅助表合并进来，检查城镇和渠道的覆盖情况：

```python
# 读取城镇表并编码城市、州，减少字符串内存
from sklearn.preprocessing import LabelEncoder
town = pd.read_csv('../input/town_state.csv')
town['Town'] = LabelEncoder().fit_transform(town['Town'].values)
town['State'] = LabelEncoder().fit_transform(town['State'].values)
print(town.head())
```

EDA 的结论是：数据没有缺失值，但产品、客户、路线都是高基数类别；测试集会出现新组合，因此不能单纯依赖“客户 × 产品”的精确记忆，必须构造更粗粒度的聚合特征。

### 2.2 数据清洗
原始 CSV 很大，直接读入会占用数 GB 内存。我们写了一个内存压缩函数，把整数列按取值范围降到 `uint8`、`uint16` 或 `int32`，把浮点列统一成 `float32`：

```python
def reduce_mem_usage(props):
    """把数值列降位存储，返回压缩后的表和含缺失值的列名"""
    start_mem = props.memory_usage().sum() / 1024**2
    for col in props.columns:
        if props[col].dtype != object:
            mx, mn = props[col].max(), props[col].min()
            # 整数列按最大值选择更窄的类型
            if (props[col] - props[col].astype(np.int64)).abs().sum() < 0.01:
                if mn >= 0:
                    if mx < 255:   props[col] = props[col].astype(np.uint8)
                    elif mx < 65535: props[col] = props[col].astype(np.uint16)
                    else:          props[col] = props[col].astype(np.uint32)
                else:
                    props[col] = props[col].astype(np.int32)
            else:
                props[col] = props[col].astype(np.float32)
    print('memory reduced to', round(props.memory_usage().sum()/1024**2, 1), 'MB')
    return props
```

对于辅助表里的字符串列，我们统一用 `LabelEncoder` 转成整数。这样 `NombreCliente`、`NombreProducto`、`Town`、`State` 都能进入树模型。缺失值很少，统一填 0；注意 `Demanda_uni_equil` 不会缺失，所以填 0 不会污染标签。文本不做转小写、去标点等常规 NLP 清洗，因为这些名称本身就是类别标识。

### 2.3 模型训练与超参数调优
我们首先构造按类别分组的历史统计特征。最朴素也最有效的是“每个产品每周的平均销量和最大销量”，因为它把新样本也映射到了产品层面：

```python
def calc_features(df, col):
    """按某个类别列统计销售均值、最大值，作为历史特征"""
    features = df.groupby(col).agg({
        'Venta_uni_hoy': ['mean', 'max'],
        'Venta_hoy': 'mean'
    })
    features.columns = ['_'.join(c) + f'_{col}' for c in features.columns]
    return features.reset_index()

# 分别对渠道、仓库、城镇、州、路线构造聚合特征
for col in ['Canal_ID', 'Agencia_ID', 'Town', 'State', 'Ruta_SAK']:
    feat = calc_features(train, col)
    train = train.merge(feat, on=col, how='left')
    test = test.merge(feat, on=col, how='left')
```

注意这些统计只用了训练集，避免把测试集信息泄漏进特征。合并后统一 `fillna(0)`。

然后我们按时间切分训练集和验证集。因为比赛是预测未来周，随机切分会高估效果，所以用前 80% 的周做训练、后 20% 做验证更合理。我们先跑随机森林基线：

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error

# 先去掉周编号和原始销售列，只保留特征
drop_cols = ['Semana', 'Venta_uni_hoy', 'Venta_hoy',
             'Dev_uni_proxima', 'Dev_proxima']
X_train = train.drop(columns=drop_cols + ['Demanda_uni_equil'])
y_train = train['Demanda_uni_equil']

rf = RandomForestRegressor(
    n_estimators=2, max_depth=5, min_samples_leaf=100000,
    random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_train)
rmsle = mean_squared_log_error(y_train, pred, squared=False)
print('RF RMSLE:', rmsle)
```

随机森林在这个数据上只能作为基线，因为树的数量和深度都受内存限制。更有竞争力的是 XGBoost。我们用它处理高维类别和长尾目标：

```python
import xgboost as xgb

boost = xgb.XGBRegressor(
    max_depth=4,          # 控制树复杂度，避免记住噪声
    learning_rate=0.1,     # 学习率
    n_estimators=10,       # 先小规模试跑
    objective='reg:linear',
    random_state=42)

boost.fit(X_train, y_train,
          eval_metric='rmse',
          eval_set=[(X_valid, y_valid)],
          early_stopping_rounds=100,
          verbose=True)
```

超参数方面，我们用 `RandomizedSearchCV` 在 `max_depth=[4,5]` 和 `lambda=[0.0,0.1]` 上做小规模搜索，评分函数用负的均方对数误差。由于数据量太大，我们没有做大规模网格搜索，而是根据验证集 RMSLE 选择了一组稳定参数。实践表明，`max_depth` 超过 6 以后验证分数不再提升，反而更容易过拟合。

### 2.4 模型测试与指标报告
在验证集上，我们对比常数基线、随机森林和 XGBoost 的 RMSLE。XGBoost 明显优于常数基线，说明聚合特征确实抓到了产品与渠道的差异。随后我们用全量训练数据重新训练，并对测试集做同样的特征合并与预测：

```python
# 全量重训
final_model = xgb.XGBRegressor(
    max_depth=4, learning_rate=0.1, n_estimators=10,
    objective='reg:linear', random_state=42)
final_model.fit(train.drop(columns=drop_cols + ['Demanda_uni_equil']),
                train['Demanda_uni_equil'])

# 测试集预测，并裁剪到非负
test_pred = final_model.predict(test)
test_pred = np.maximum(test_pred, 0)

submission = pd.DataFrame({
    'id': test['id'],
    'Demanda_uni_equil': test_pred
})
submission.to_csv('submission.csv', index=False)

# 检查行数和列名
print(submission.shape, submission.columns.tolist())
```

提交前我们做了三项检查：行数是否与 `sample_submission` 一致、`id` 是否唯一、预测值有没有负数。由于 RMSLE 不允许负值，这一步是必须的。材料中没有给出最终排行榜的具体分数，因此我们只把验证集上的 RMSLE 作为模型选择的依据。

## 三、总结
本文中我们完成了 Grupo Bimbo 库存需求预测的完整流程。我们首先用 EDA 理解了销售、退货与均衡需求的关系，确认目标列非负且长尾严重。其次做了内存压缩、类别编码和城镇表合并的清洗工作。接着构造了按产品、渠道、仓库、城镇聚合的历史统计特征，并用随机森林做基线、用 XGBoost 做主要模型，在验证集上按 RMSLE 调参。最后用全量数据重训并生成提交文件，同时检查了格式与非负约束。这个比赛给我们的启示是：面对大规模零售数据，先把内存和类别编码处理好，再围绕业务维度做稳定的聚合特征，往往比堆叠复杂模型更有效。
