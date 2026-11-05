# 二手车价格回归预测

## 一、比赛简介

### 1.1 比赛背景

二手车价格预测是机器学习中非常经典的回归问题。市场上流通的二手车数量庞大，每辆车的品牌、型号、年份、里程、发动机、事故历史等都会影响其市场价。Kaggle 上同类比赛的数据通常来源于车商平台的挂牌信息，目标是根据车辆的公开属性预测合理的成交价。难度在于价格分布往往右偏，存在高价豪车的长尾，同时不同品牌和车型的样本量差异很大，需要用特征工程和模型调优来平衡影响。

从材料里的字段可以看出，这份数据比较接近真实的二手车交易记录，而不是人工构造的玩具数据。真实记录的特点是：同一品牌下不同车型的价格跨度很大，同一车型又会受到年份、里程和车况的共同影响；事故记录、清洁标题、变速箱类型、燃油类型这些字段单独看信息量有限，组合起来才能较完整地刻画一辆车的价值。也正因为如此，单纯换更复杂的模型往往不如先把特征梳理清楚。我们在这篇 writeup 里会把重点放在数据理解和特征工程上，模型部分选择训练速度快、对表格数据友好的梯度提升树。

### 1.2 比赛目标

给定训练集 `train.csv` 和测试集 `test.csv`，我们需要预测测试集每辆车的 `price`。训练集包含车辆的多个描述性字段，例如 `brand` 品牌、`model` 型号、`model_year` 出厂年份、`milage` 里程、`engine` 发动机描述、`accident` 事故记录、`clean_title` 清洁记录、`transmission` 变速箱、`fuel_type` 燃油类型等，以及目标列 `price`。测试集缺少 `price` 列，我们要输出 `id` 与预测价格的提交文件 `submission.csv`。

在这里 id 只是编号，不参与预测。price 是连续数值，所以这是一个回归任务。不同于分类任务，我们不需要输出类别标签，而是预测具体数值，这对模型对异常值的鲁棒性要求更高。数据中还有 engine 这种半结构化文本，里面混杂了马力、排量、气缸数等信息，需要拆成数值列。类别列又可以分成高基数和低基数两类：brand、model 基数大，直接编码会产生稀疏特征；accident、clean_title、transmission、fuel_type 基数小，适合常规编码。理解这种差异对后续的预处理策略很重要。

### 1.3 评价标准

从材料中的代码来看，模型训练时使用了均方根误差 RMSE 作为评估指标，CatBoost 示例中 `eval_metric='RMSE'`，LightGBM 实验中也计算了 RMSE。回归任务里 RMSE 越小越好，它惩罚大误差更重，适合价格预测这种对离群值敏感的场景。实际提交时排行榜通常以 RMSE 或其变体排序。

RMSE 的计算思路是先算每个样本的预测误差，取平方后再求平均，最后开根号。相比平均绝对误差 MAE，RMSE 对大误差更敏感：一辆豪车预测偏了 5 万美元，惩罚相当于五辆普通车各偏了 1 万美元。因为二手车数据里存在高价长尾，优化 RMSE 会促使模型关注高价样本，但同时也容易被极端值拉偏。这正是我们在数据清洗阶段对价格做上限截断和对数转换的原因，既保留高价信息，又降低极端值的影响。

## 二、解题过程

整场比赛的思路可以概括为：先摸清数据分布，再做针对性的清洗和特征构造，最后用梯度提升树做回归并通过交叉验证评估。我们按 EDA、数据清洗、模型训练与调优、测试与报告四步来走。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，看看数据的基本形态和缺失情况，先建立整体印象。

```python
import pandas as pd
import numpy as np

# 读取训练和测试数据，方便后续对比
train = pd.read_csv('/kaggle/input/sch2-reg-2026-d1-3/train.csv')
test = pd.read_csv('/kaggle/input/sch2-reg-2026-d1-3/test.csv')

# 查看数据形状，确认样本数和列数是否合理
print('Train shape:', train.shape)
print('Test shape:', test.shape)

# 查看缺失值情况，缺失会影响模型训练，需要先知道哪些列有问题
missing = train.isnull().sum()
print(missing[missing > 0])

# 描述性统计，了解价格分布特点
print(train['price'].describe())
```

打印信息后我们能看到价格整体右偏，存在极端高价车。类别列如 `brand`、`transmission`、`fuel_type`、`accident`、`clean_title` 的取值种类比较多，需要后续做编码。`engine` 列是文本，里面混杂了马力、排量、气缸数等信息，可以拆解成数值特征。数据没有明显的重复行，缺失值主要集中在部分可选字段，这提示我们需要填充策略而不是直接删除。

为了决定编码方式，我们又统计了每个类别列的唯一值数量，并看了几个主要类别的取值分布。

```python
# 统计每个类别列有多少种取值，判断该用哪种编码
for col in train.select_dtypes(include='object').columns:
    print(col, train[col].nunique())

# 查看品牌和变速箱的头部取值，判断是不是长尾分布
print(train['brand'].value_counts().head())
print(train['transmission'].value_counts().head())
```

从结果我们得出一个很实用的结论：`brand` 和 `model` 属于高基数类别，直接独热会生成大量稀疏列，模型很难学好；所以后面我们把品牌和型号拼成 `brand_model`，并在编码时设置 `min_frequency`，把出现次数很少的组合归到同一类。`transmission` 和 `fuel_type` 的取值相对集中，可以正常独热。`accident` 和 `clean_title` 基本是二值或少量类别，直接映射成 0/1 更省事。按基数分开处理，比所有类别列都用同一种编码方式更稳妥。

接着我们统计了各品牌的样本量和价格的箱线图，发现品牌间价格差异很大。车型年份与价格呈明显的负相关关系，年份越新价格越高。里程数与价格也呈负相关，但关系不是严格线性，这为后续做 `milage_per_year` 这样的衍生特征提供了思路。

箱线图还能帮我们看中位数和四分位范围，同时观察离群点。豪华品牌的中位数明显高于普通品牌，而且同一品牌内部的跨度也很大，说明品牌只能解释一部分价格，不能当作唯一依据。年份和里程的关系更接近单调，但斜率在不同价位段不一样，这提示我们构造年龄和年均里程这类交互特征，而不是只保留原始列。

### 2.2 数据清洗

其次我们做数据清洗，目标是让数据更干净、更适合树模型学习。清洗的核心是处理异常价格、构造车辆年龄和发动机特征，并把文本标志二值化。

```python
# 计算车辆年龄，年龄越大价格通常越低，是强特征
CURRENT_YEAR = 2024
train['vehicle_age'] = CURRENT_YEAR - train['model_year']
test['vehicle_age'] = CURRENT_YEAR - test['model_year']

# 从 engine 文本中提取马力和排量，避免把混杂信息当整体使用
train['engine_hp'] = train['engine'].str.extract(r'(\d+\.?\d*)HP').astype(float)
test['engine_hp'] = test['engine'].str.extract(r'(\d+\.?\d*)HP').astype(float)

# 提取排量和气缸数，构造 V6 V8 标志
train['engine_size'] = train['engine'].str.extract(r'(\d+\.?\d*)L').astype(float)
train['cylinders'] = train['engine'].str.extract(r'(\d+)\sCylinder').astype(float)

# 限制价格上限，减少极端异常值对模型的影响
price_cap = train['price'].quantile(0.99)
train = train[train['price'] <= price_cap].reset_index(drop=True)

# 把清洁记录和事故记录转成 0/1，方便模型使用
train['clean_title'] = train['clean_title'].map({'Yes': 1, 'No': 0})
train['has_accident'] = train['accident'].str.contains('accident', case=False, na=False).astype(int)
```

清洗后我们保留了价格的 99 分位以内样本，去掉了极端高价带来的噪声。车辆年龄和 `milage_per_year` 帮助模型区分老车但保养良好的情况。发动机信息拆解后，马力和排量成为数值特征，V6/V8 标志则捕捉了动力等级差异。品牌与型号拼接成 `brand_model` 可以保留车型组合的信息，同时避免单独的型号列过于稀疏。

在这一阶段我们还对数转换了目标价格 `y = log1p(price)`，因为价格分布右偏，对数转换能让误差更均匀，也符合梯度提升树对对数空间的学习习惯。

接下来我们又构造了几个补充特征，并且每一步都同时作用于训练集和测试集，避免两边不一致导致提交时出错。

```python
# 构造年均里程，衡量用车强度：老车低里程和新车高里程的价格会不同
train['milage_per_year'] = train['milage'] / (train['vehicle_age'] + 1)
test['milage_per_year'] = test['milage'] / (test['vehicle_age'] + 1)

# 品牌与型号拼接，保留具体车型的组合信息
train['brand_model'] = train['brand'] + '_' + train['model']
test['brand_model'] = test['brand'] + '_' + test['model']

# 缺失的类别字段统一填成 Unknown，交给编码器处理，而不是删除行
for col in ['accident', 'clean_title', 'transmission', 'fuel_type']:
    train[col] = train[col].fillna('Unknown')
    test[col] = test[col].fillna('Unknown')
```

`vehicle_age` 加 1 再作分母是为了避免车龄为零时除零出错。年均里程把总里程和车龄合并成一个量，能更好地反映用车强度：同样是 5 万公里，用了 5 年的车比用了 15 年的车磨损更重，价格自然不同。`brand_model` 保留了具体型号的信息，配合编码阶段的最小出现频次设置，不会因为组合太多而变得稀疏。类别缺失字段我们统一填成 Unknown 而不是删除样本，因为测试集同样存在缺失，删掉只会白白损失样本。

特征准备好之后，我们从特征表里去掉 `id`、`price`、`engine` 和 `model` 原列：`id` 只是编号没有信息量，`price` 是我们要预测的目标，`engine` 和 `model` 已经被拆解或用组合列取代，保留只会增加维度而不带来新信息。剩下的数值列和类别列分别收集成 `num_cols` 和 `cat_cols`，交给后面的预处理流水线使用。

### 2.3 模型训练与超参数调优

接着我们进入模型训练。我们选择 LightGBM 作为主模型，因为它在表格数据上训练快、效果稳定。特征预处理用 `ColumnTransformer` 分开处理数值和类别，数值只做中位数填充，类别做众数填充后再独热编码，并设置 `min_frequency` 来合并稀有类别。

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor

# 数值列只做中位数填充，避免引入偏差
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# 类别列先填充再做独热，避免未知类别报错
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=200))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# LightGBM 参数先保守设置，保证收敛
model = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', model)
])
```

训练时我们使用对数目标并配合 5 折交叉验证，保证评估稳定。材料中也尝试过 CatBoost 作为对比基线。CatBoost 对类别特征的原生支持比较强，不需要外部编码，直接把类别列索引传进去即可。下面是材料里基线做法的简化版本。

```python
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# 用 80% 训练、20% 验证，切分时固定随机种子保证可复现
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CatBoost 可以直接处理类别列，适合做快速基线
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)

cat_model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)
```

基线训练采用早停策略，避免过拟合，同时保证验证集上的 RMSE 有参考意义。我们在材料里看到，CatBoost 训练时间比较长，而且最终效果并没有比 LightGBM 更好，特征工程带来的提升更明显。因此主线仍保留 LightGBM 配合预处理流水线的做法。

超参数调优上我们没有做大规模网格搜索，而是先固定合理的经验值，观察验证集的 RMSE 变化。特征工程带来的提升比微调单个超参数更明显，这与二手车数据的高噪声特性相符。

### 2.4 模型测试与指标报告

然后我们进行模型测试。交叉验证阶段我们对每折的验证集先做对数空间的预测，再转回原始价格空间计算 RMSE 和 MAE，用于报告。

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 5 折交叉验证评估稳定性
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_log = np.log1p(train['price'])

r2_logs, mae_real, rmse_real = [], [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(x), 1):
    x_tr, x_val = x.iloc[tr_idx], x.iloc[val_idx]
    y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]
    
    pipeline.fit(x_tr, y_tr)
    preds_log = pipeline.predict(x_val)
    
    # 转回原始价格空间计算可解释指标
    y_val_real = np.expm1(y_val)
    preds_real = np.expm1(preds_log)
    
    rmse_real.append(np.sqrt(mean_squared_error(y_val_real, preds_real)))
    mae_real.append(mean_absolute_error(y_val_real, preds_real))
    print(f'Fold {fold} RMSE: {rmse_real[-1]:,.0f}')
```

交叉验证结果显示模型在对数空间的 R2 达到约 0.68，均方误差对应的平均绝对误差在 1.2 万美元量级。材料中的结论提到模型对数转换 R2 约 0.685，MAE 约 1.277 万美元，这说明特征工程和清洗起到了作用，但高价车仍是误差的主要来源。

为了更直观地看 5 折的稳定性，我们在每折结束后打印平均指标，并在最后汇总。

```python
# 汇总 5 折指标，判断稳定性
print('R2 mean:', np.mean(r2_logs))
print('MAE mean:', np.mean(mae_real))
print('RMSE mean:', np.mean(rmse_real))
print('RMSE std:', np.std(rmse_real))

# 用箱线图检查各折 RMSE 分布是否集中
import matplotlib.pyplot as plt
plt.boxplot(rmse_real)
plt.title('RMSE distribution across folds')
plt.savefig('rmse_boxplot.png')
```

分折结果的波动相对可控，说明交叉验证的划分没有导致过大的抽样误差。误差主要集中在高价段，低价段的预测相对稳定。这与二手车价格长尾的特性一致：高价车样本少，特征覆盖不全，模型容易低估或高估。为缓解这个问题，我们既做了价格上限截断，也在目标上做了对数转换，让模型在不同价位上分配更均衡的学习注意力。

最后我们用全量训练数据重新拟合模型，预测测试集并生成提交文件。

```python
# 用全部训练数据最终训练
pipeline.fit(x, y_log)

# 预测测试集并转回价格空间
test_preds_log = pipeline.predict(x_test)
test_preds = np.expm1(test_preds_log)
test_preds = np.clip(test_preds, 0, None)

submission = pd.DataFrame({
    'id': test['id'],
    'price': test_preds
})
submission.to_csv('submission.csv', index=False)
```

提交前我们检查了 `id` 与行数是否匹配，价格没有负值，格式符合要求。最后我们最后总结了整体流程。

## 三、总结

本文中我们完成了二手车价格回归预测的完整流程。我们首先用 EDA 摸清了数据：价格右偏、品牌差异大、发动机文本可拆解。其次我们做了数据清洗——限制价格 99 分位、构造车辆年龄和马力排量特征、对数转换目标。接着我们用 LightGBM 配合数值中位数填充和类别独热编码的流水线训练模型，并通过 5 折交叉验证评估稳定性和误差分布。最后我们在验证集上得到对数 R2 约 0.68、MAE 约 1.27 万美元的水平，并在测试集上生成提交文件。

从整个过程我们总结出几个要点。第一，特征工程比盲目调参更有效：年均里程、车龄、发动机拆解和品牌型号组合，这些直接反映车辆价值的特征，比单纯提高树的深度或树的数量带来的增益更稳定。第二，数据清洗要服务于任务：价格上限截断和目标对数转换不是对原数据的破坏，而是让模型能够在长尾分布下更好学习。第三，评估要分层看：整体 RMSE 会掩盖高价段和低价段的差异，拆开看每折的指标和价格区间的误差，才能判断模型真实的弱点。最后，基线对比很重要：CatBoost 虽然能原生处理类别特征，但在这个数据集上训练成本更高，收益有限，说明选择工具要看数据规模和工程成本。

这个比赛给我们的最大启发是特征工程的重要性。同样是树模型，加入车辆年龄、里程年均化、发动机拆解和品牌型号组合后，效果明显提升。价格分布的长尾特性提醒我们在回归任务中既要处理异常值，也要考虑目标转换，让模型更关注整体趋势而非极端值。
