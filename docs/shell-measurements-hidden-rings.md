# Shell Measurements: Hidden Rings

## 一、比赛简介

### 1.1 比赛背景

这是一场贝壳年龄预测回归比赛。数据来自 Abalone（鲍鱼）测量数据，比赛把真实的年龄用 `Rings` 表示：贝壳上的环数通常与年龄有关。我们不能直接看到年龄，只能根据贝壳的性别和尺寸、重量等记录来估计环数。题目的特别之处在于，它不是普通的“猜一个整数”，而是使用对数尺度上的误差来评价。这样做会让小年龄样本的相对误差和大年龄样本的相对误差更可比，也提醒我们不能只追求训练集中最大的环数。

材料说明这场比赛来自 Kaggle Playground S4E4，原始测量数据来自 UCI Abalone。数据文件保留原样，训练集、测试集和提交样例分别提供。这个任务的难点有三点：第一，输入都是连续测量值，环数却是离散的整数；第二，`Sex` 是类别特征，不能简单当成普通数字；第三，评价指标是 RMSLE，最合适的训练目标和最后输出之间要经过 `log1p` 与 `expm1` 的转换。我们还要保证提交文件中的 id 顺序与测试集完全一致。

### 1.2 比赛目标

训练集每行包含 `id`、`Sex`、`Length`、`Diameter`、`Height`、`Whole weight`、`Whole weight.1`、`Whole weight.2`、`Shell weight` 和目标列 `Rings`。这些列分别描述性别、长度、直径、高度、整体重量以及不同组成部分的重量。测试集有相同的输入列，但没有 `Rings`。我们的目标是用训练集学习一个回归模型，为测试集每行预测环数，并写出两列的 `submission.csv`：一列是原样保留的 `id`，另一列是预测的 `Rings`。

`id` 只负责对齐输出，不应该被模型当作贝壳的测量信息。`Sex` 是类别变量，其他测量列应转成数值。预测值允许是小数，但必须是非负数；例如模型预测 8.6 个环并不违反提交格式，因为评价的是误差，不要求输出整数。

### 1.3 评价标准

比赛使用均方根对数误差 RMSLE（Root Mean Squared Logarithmic Error）：

\[
RMSLE=\sqrt{\frac{1}{n}\sum_i(\log(1+\hat y_i)-\log(1+y_i))^2}
\]

其中 `y` 是真实环数，`ŷ` 是预测环数。RMSLE 越低越好，理想值是 0。因为公式先做 `log1p`，预测 3 和真实 6 的差距会以相对比例的方式体现，而不是只看相差了几个环。预测和真实值都必须非负，否则对数没有意义。材料还给出了一个直观例子：真实目标为 3 和 8 时，直接取原始均值 5.5 并不是 RMSLE 下的最佳常数；在对数空间求平均再还原，结果是 5。因此我们从一开始就围绕对数目标设计训练流程。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取三个文件，检查列名、id 是否重复、训练测试是否重叠，并统计缺失值和目标范围。这样做的目的不是马上训练，而是先确认数据可以安全地进入流水线。

```python
from pathlib import Path
import numpy as np
import pandas as pd

# 先定位同时包含三个比赛文件的数据目录，避免误读其他比赛的数据
DATA = Path('../data')
train = pd.read_csv(DATA / 'train.csv')
test = pd.read_csv(DATA / 'test.csv')
sample = pd.read_csv(DATA / 'sample_submission.csv')

# 这些检查能尽早发现列名、顺序和 id 对不上的问题
assert list(sample.columns) == ['id', 'Rings']
assert train['id'].is_unique and test['id'].is_unique
assert sample['id'].is_unique
assert np.array_equal(test['id'], sample['id'])
assert set(train['id']).isdisjoint(test['id'])

features = train.drop(columns=['id', 'Rings'])
print('Rows:', len(train), 'train;', len(test), 'test')
print('Features:', list(features.columns))
print('Missing values:', int(features.isna().sum().sum()))
print('Repeated feature rows:', int(features.duplicated().sum()))
print('Rings range:', int(train.Rings.min()), int(train.Rings.max()))
```

我们其次把 `id` 和 `Rings` 从特征表中排除。`id` 是输出定位符，并不携带测量含义；`Rings` 只在训练时作为目标。EDA 还检查了目标不为空且不小于 0。材料明确提醒我们，重复特征行虽然值得报告，但不能据此宣称模型能推广到新的生物种群；如果未来数据包含同一只动物、同一养殖场或时间序列，就应该使用分组或时间切分，而不是无条件采用随机切分。

### 2.2 数据清洗

我们接着做最少而明确的数据清洗：类别列缺失时填成一个特殊字符串，数值列强制转换为数字；比例特征遇到 0 分母时先换成缺失值，最后把无穷大替换为缺失。这样做比直接删除整行更稳妥，因为零分母只是某个派生比例无法计算，并不代表整条测量记录无效。

```python
def prepare(frame, engineered=False):
    # id 和目标列只用于对齐或训练，不能把它们作为输入特征
    x = frame.drop(columns=['id', 'Rings'], errors='ignore').copy()
    # 把缺失性别变成可被 CatBoost 识别的类别，而不是混入 NaN
    x['Sex'] = x['Sex'].fillna('<missing>').astype(str)
    for col in x.columns.drop('Sex'):
        # errors='raise' 可以在数据格式异常时立即停止，而不是悄悄产生错误特征
        x[col] = pd.to_numeric(x[col], errors='raise')
    if engineered:
        # 分母为零时使用 NaN，避免产生无穷大比例
        whole = x['Whole weight'].replace(0, np.nan)
        length = x['Length'].replace(0, np.nan)
        diameter = x['Diameter'].replace(0, np.nan)
        x['weight1_share'] = x['Whole weight.1'] / whole
        x['weight2_share'] = x['Whole weight.2'] / whole
        x['shell_share'] = x['Shell weight'] / whole
        x['length_diameter_ratio'] = x['Length'] / diameter
        x['height_length_ratio'] = x['Height'] / length
        x['size_product'] = x['Length'] * x['Diameter'] * x['Height']
    return x.replace([np.inf, -np.inf], np.nan)
```

我们没有把 `Sex` 做成任意的 0、1、2。数字编码会暗示类别之间存在大小关系，例如把 M 当成 2 并不表示它比 F“大两倍”。CatBoost 可以直接接收类别列，因此保留字符串更符合数据含义。我们还比较了两套特征：一套只使用原始测量值，另一套增加重量占比、长度直径比、高度长度比以及长宽高乘积。所有派生量只依赖当前行的测量值，没有使用目标统计，因此不会把验证集答案泄漏进训练过程。

### 2.3 模型训练与超参数调优

我们然后把目标转换为 `log1p(Rings)`。这一步让模型直接在比赛指标更接近的尺度上学习。模型选择 CatBoostRegressor，因为它能处理数值特征和类别特征的混合，并且不需要我们把 `Sex` 粗暴编码。材料给出的固定设置是 3 折交叉验证、每折 700 棵树、深度 7、学习率 0.05、L2 正则 5。我们使用相同的折分比较原始特征和比例特征，避免因为验证样本变化而误判改动效果。

```python
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

SEED = 20261025
FOLDS = 3
ITERATIONS = 700
splits = list(KFold(n_splits=FOLDS, shuffle=True,
                    random_state=SEED).split(train))
target_log = np.log1p(train['Rings'].to_numpy(dtype=float))
oof, test_log, records = {}, {}, []

for name, engineered in [('raw', False), ('ratios', True)]:
    x = prepare(train, engineered)
    xt = prepare(test, engineered)
    oof[name] = np.full(len(train), np.nan)
    test_log[name] = np.zeros(len(test))
    for fold, (fit_idx, valid_idx) in enumerate(splits):
        # 固定树数和折分，才能公平比较两套特征
        model = CatBoostRegressor(
            iterations=ITERATIONS, depth=7, learning_rate=0.05,
            loss_function='RMSE', l2_leaf_reg=5,
            random_seed=SEED + fold, thread_count=4,
            verbose=False, allow_writing_files=False)
        model.fit(x.iloc[fit_idx], target_log[fit_idx],
                  cat_features=['Sex'])
        oof[name][valid_idx] = model.predict(x.iloc[valid_idx])
        test_log[name] += model.predict(xt) / FOLDS
```

这里的 OOF（out-of-fold）预测很重要：每一行验证样本都由没有见过它的模型预测，因而能比训练集内预测更诚实。我们没有根据某一次验证分数临时修改树数，而是固定 700 棵树，让两套特征在同一个实验条件下比较。材料也特别提醒，反复复用同一折分会使 OOF 分数只是开发阶段估计，不应把它当成对未来数据的保证。

最后我们在对数空间做等权融合。这样不是把两个已经还原到环数的结果简单平均，而是先平均两个模型对于相对规模的判断，再用 `expm1` 回到原始环数。我们可以用 RMSLE 函数检查每折和整体表现，并与训练折目标的对数均值常数作比较。

```python
def rmsle(y, prediction):
    # 比赛要求非负且有限，先检查能避免错误结果被写入提交文件
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if not (np.isfinite(y).all() and np.isfinite(prediction).all()):
        raise ValueError('RMSLE requires finite values.')
    if (y < 0).any() or (prediction < 0).any():
        raise ValueError('RMSLE requires nonnegative values.')
    return float(np.sqrt(np.mean((np.log1p(prediction) -
                                  np.log1p(y)) ** 2)))

# 在 log1p 空间等权融合，最后才还原到 Rings
blend_oof_log = 0.5 * oof['raw'] + 0.5 * oof['ratios']
blend_test_log = 0.5 * test_log['raw'] + 0.5 * test_log['ratios']
pred_oof = np.maximum(0., np.expm1(blend_oof_log))
pred_test = np.maximum(0., np.expm1(blend_test_log))
summary = {name: rmsle(train.Rings,
             np.maximum(0., np.expm1(values)))
           for name, values in oof.items()}
summary['fixed_log_blend'] = rmsle(train.Rings, pred_oof)
print(summary)
```

这次“超参数调优”更像是受控实验：我们没有进行庞大的网格搜索，而是固定模型参数、固定折分，比较 raw 与 ratios，再采用事先确定的 0.5/0.5 融合。对于小而结构明确的测量表格，这种做法更容易复现，也更容易解释。若继续改进，材料建议先冻结独立 holdout，再研究更小模型、原始数据增强和外层折上的融合选择，并记录运行时间和不确定性。

### 2.4 模型测试与指标报告

我们最后检查误差是否集中在某个环数范围。RMSLE 在对数尺度上计算，所以按目标区间统计平方对数误差，能帮助我们判断模型是否只在尾部样本上吃亏，而不是凭感觉把预测值强行推向训练最大值。

```python
diagnostics = pd.DataFrame({
    'id': train.id, 'Rings': train.Rings,
    'prediction': pred_oof
})
diagnostics['squared_log_error'] = (
    np.log1p(diagnostics.prediction) -
    np.log1p(diagnostics.Rings)
) ** 2
diagnostics['ring_range'] = pd.cut(
    diagnostics.Rings, [-np.inf, 5, 10, 15, np.inf])
by_range = diagnostics.groupby('ring_range', observed=False).agg(
    rows=('Rings', 'size'),
    mean_squared_log_error=('squared_log_error', 'mean'))
by_range['rmsle'] = np.sqrt(by_range['mean_squared_log_error'])
print(by_range)
```

材料没有给出可直接复述的排行榜分数，因此我们不虚构一个数。我们会报告固定折分下的每个模型 RMSLE、融合后的整体 OOF RMSLE，以及按环数区间的诊断表。这样读者能知道这个数字来自哪种验证协议，也不会把开发估计误解为真正的测试集成绩。特别是，随机划分的合成样本不能证明模型一定适用于全新的生物群体。

确认 OOF 结果和约束后，我们使用测试集预测生成提交文件。写出文件前后都检查行数、列名、id 顺序、非负性和有限性，防止模型本身没问题却因为提交格式失败。

```python
submission = sample.copy()
submission['Rings'] = pred_test
assert len(submission) == len(test)
assert list(submission.columns) == ['id', 'Rings']
assert np.array_equal(submission.id, sample.id)
assert np.isfinite(submission.Rings).all()
assert (submission.Rings >= 0).all()
submission.to_csv('submission.csv', index=False)

# 重新读回文件，验证磁盘上的内容与内存中的结果一致
check = pd.read_csv('submission.csv')
assert np.array_equal(check.id, sample.id)
assert np.allclose(check.Rings, submission.Rings)
assert not check.isna().any().any()
print('Submission shape:', check.shape)
print(check.head())
```

## 三、总结

本文中我们完成了 Shell Measurements: Hidden Rings 的一套可复现回归流程。我们首先用 EDA 检查了训练集、测试集和样例提交的列结构、id 唯一性、数据重叠、缺失值、重复特征行以及 `Rings` 的非负范围。其次我们只做必要的数据清洗：把缺失的 `Sex` 变成类别字符串，把其他列转成数字，并在计算比例时安全处理零分母。接着我们把 `Rings` 转换到 `log1p` 空间，使用能处理类别变量的 CatBoostRegressor，在固定的 3 折上比较原始测量和比例、尺寸派生特征，再在对数空间做 0.5/0.5 融合。最后我们通过 OOF RMSLE、环数区间误差表和严格的提交文件检查来评估模型。

这个比赛最重要的启发是：先理解评价指标，再决定训练目标。RMSLE 不是把原始误差简单平方，所以对数变换不是装饰，而是让模型学习目标与比赛评分方向一致。第二个启发是特征工程要尊重物理含义：重量占比和尺寸乘积可以补充原始列，但零分母、单位变化和数据重复都必须被明确记录。第三个启发是可复现性本身也是解题的一部分：固定种子、固定折分、保留 OOF 预测、检查提交 id，才能知道一次分数提升到底来自真正的改进，还是来自验证过程被不小心改变了。
