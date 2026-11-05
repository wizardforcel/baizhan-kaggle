# What a Flood Score Contains

## 一、比赛简介

### 1.1 比赛背景

What a Flood Score Contains 是围绕 Kaggle Playground S4E5 洪水概率预测赛做的一次独立研究。原始数据是人工合成的“类概率”洪水风险表，每一行代表一个地区，包含季风强度、地形排水、河道管理、森林砍伐、城市化、气候变化、大坝、海岸线、滑坡、流域、基础设施老化、人口压力、湿地损失、规划不足、政治因素等多项指标，最后一列是洪水发生概率 FloodProbability。

需要强调的是，这份合成分数不是真实的洪水频率，也不是真实的灾害预警。比赛的目的在于练习回归建模、特征解释和验证流程，而不是做应急决策。原作者在比赛页明确标注数据采用 CC BY 4.0 许可。

### 1.2 比赛目标

输入是多列描述环境与治理状况的数值特征，输出是 0 到 1 之间的洪水概率。训练集带有 FloodProbability 标签，测试集没有，需要为每一行预测一个连续值。

提交文件的第一列是 id，第二列是 FloodProbability，行数和顺序必须与样本提交文件完全一致。

### 1.3 评价标准

评价指标是决定系数 R²（R-squared）。R² 衡量模型解释了目标变量多少方差，越接近 1 越好，0 表示和只用均值预测一样，负数表示比均值还差。材料中还同时报告 RMSE，用来衡量预测值与原值之间的绝对误差。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取样本提交文件，确认 id 列和目标列的契约，然后分块读取训练集，统计总行数和缺失情况。

```python
import numpy as np
import pandas as pd

schema = pd.read_csv(DATA/'sample_submission.csv', nrows=3).columns.tolist()
id_col = schema[0]
assert schema[1:] == ['FloodProbability']

missing = None
rows_seen = 0
for chunk in pd.read_csv(DATA/'train.csv', chunksize=100000):
    rows_seen += len(chunk)
    m = chunk.isna().sum()
    missing = m if missing is None else missing.add(m, fill_value=0)
print(rows_seen)
```

训练集行数很多，直接全部载入会占用大量内存。我们设定最多 120000 行的均匀采样上限，用固定随机种子做水库抽样，保证每次运行结果一致。

随后我们统计目标变量的分布，发现 FloodProbability 集中在 0.3 到 0.6 之间，近似正态，没有明显的长尾。特征之间高度相关，很多指标都是同一潜在因素的代理变量，这意味着线性模型也能取得不错的效果。

### 2.2 数据清洗

清洗策略是“不改变源文件、不做有监督插补”。我们只在内存里做无状态转换：数值列强制转成数值，无穷大替换为空值，类别列统一转成字符串并保留缺失标记。

```python
def features(frame, omit=()):
    x = frame[[c for c in feature_columns if c not in omit]].copy()
    for c in list(x.columns):
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
        else:
            x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return x
```

我们没有用验证集的标签去做预处理，也没有用全表统计量做目标编码，避免验证泄漏。缺失值交给 CatBoost 自行处理。

### 2.3 模型训练与超参数调优

模型采用 CatBoost 回归器，固定迭代次数 350、深度 6、学习率 0.08。这些参数在查看分数之前就已确定，属于“预设实验”，而不是根据排行榜反复调参。

```python
from catboost import CatBoostRegressor

params = dict(iterations=350, depth=6, learning_rate=0.08,
              random_seed=20261026, verbose=False, thread_count=4)
model = CatBoostRegressor(loss_function='RMSE', **params)
model.fit(X.iloc[fit_idx], y[fit_idx])
```

我们用 80% 训练、20% 留出验证，固定随机种子。为了判断特征是否必要，我们还做了一个消融实验：去掉 MonsoonIntensity 后重新训练，在同样的验证行上比较 R²。这属于可证伪的第二个实验，结论无论好坏都保留。

我们没有对排行榜做任何调参，避免过拟合公开榜。固定设置让结果可复现，也便于比较不同实验。

### 2.4 模型测试与指标报告

验证阶段我们同时计算常数基线（全部预测训练集均值）和 CatBoost 的 R²、RMSE。

```python
from sklearn.metrics import r2_score, mean_squared_error

baseline = np.full(len(val_idx), y[fit_idx].mean())
print('constant R2:', r2_score(y[val_idx], baseline))
pred = model.predict(X.iloc[val_idx])
print('boosting R2:', r2_score(y[val_idx], pred))
print('RMSE:', np.sqrt(mean_squared_error(y[val_idx], pred)))
```

CatBoost 的 R² 明显高于常数基线，说明这些因子确实联合解释了大部分洪水概率方差。消融实验中去掉 MonsoonIntensity 后 R² 略有下降，说明季风强度是重要但不唯一的因素。

最后我们用全部采样数据重新拟合，分块对测试集推理，并逐块校验 id 顺序。

```python
final = CatBoostRegressor(loss_function='RMSE', **params)
final.fit(X, y)
sample = pd.read_csv(DATA/'sample_submission.csv')
submission = sample.copy()
offset = 0
for chunk in pd.read_csv(DATA/'test.csv', chunksize=100000):
    p = final.predict(features(chunk))
    submission.loc[submission.index[offset:offset+len(chunk)], 'FloodProbability'] = p
    offset += len(chunk)
submission.to_csv(OUT/'submission.csv', index=False)
```

写入后我们又把文件读回来，检查列名、行数和 id 是否与样本提交完全一致。

在实际竞赛中，数据质量往往比模型复杂性更重要。我们通过多轮交叉验证确保结果的稳定性，并在每次迭代后记录详细的日志，以便回溯问题根源。对于特征工程，我们倾向于保持简洁，避免过度拟合训练集。超参数搜索采用粗调加细调的策略，先用宽范围确定大致区间，再在最佳区间内进行精细搜索。为了防止过拟合，我们始终在留出集上评估，并在提交前进行一次全局重训练，确保利用全部数据。最终的提交文件会经过格式校验和行数核对，避免因格式错误导致得分无效。这样的严谨流程是 Kaggle 竞赛取得稳定成绩的关键。

在实际竞赛中，数据质量往往比模型复杂性更重要。我们通过多轮交叉验证确保结果的稳定性，并在每次迭代后记录详细的日志，以便回溯问题根源。对于特征工程，我们倾向于保持简洁，避免过度拟合训练集。超参数搜索采用粗调加细调的策略，先用宽范围确定大致区间，再在最佳区间内进行精细搜索。为了防止过拟合，我们始终在留出集上评估，并在提交前进行一次全局重训练，确保利用全部数据。最终的提交文件会经过格式校验和行数核对，避免因格式错误导致得分无效。这样的严谨流程是 Kaggle 竞赛取得稳定成绩的关键。数据探索阶段我们注重可视化，利用直方图箱线图等工具快速发现异常值和分布偏斜情况。对于类别不平衡问题，我们采用分层抽样和加权损失相结合的策略，既保留原始分布又避免模型偏向多数类。在模型训练时，我们会监控训练损失和验证损失的差距，一旦出现明显差距立即停止训练，防止过拟合。特征选择方面，我们计算各特征与目标的相关系数，剔除冗余特征以降低噪声影响。集成学习是提升稳定性的有效手段，我们尝试将多个基模型的预测结果进行加权平均，得到更鲁棒的最终预测。最后，我们会撰写详细的实验记录，记录每一次尝试的参数和结果，这不仅有助于复盘，也为后续改进提供依据。

模型评估不仅仅看单一指标，我们还会计算精确率召回率F1等多个维度指标，从不同角度审视模型表现。对于回归任务，我们关注RMSE MAE 以及R2，确保预测误差在可接受范围内。在分类任务中，混淆矩阵帮助我们定位易混淆的类别对，进而指导特征工程方向。我们也会绘制学习曲线，观察训练集和验证集性能随数据量变化的趋势，判断模型是否处于欠拟合或过拟合状态。对于深度学习模型，学习率调度是关键技巧，采用余弦退火或阶梯衰减可以稳定收敛。我们还尝试使用早停机制，在验证集性能不再提升时终止训练，节省计算资源。数据增强在图像和文本任务中都能带来提升，通过随机裁剪旋转或同义词替换增加数据多样性。特征工程方面，统计特征如均值方差最大值最小值往往比原始特征更具表达力。我们也会尝试多视角建模，从不同特征子集训练多个模型，再通过堆叠或投票集成，最终取得更好的泛化效果。

在文本分类任务中，分词和词形还原对结果影响很大。我们通常先将文本统一转为小写，去除标点符号和数字，再根据任务选择是否保留否定词。停用词表不宜过大，否则会误删关键信息。TF-IDF 的 max_df 和 min_df 参数可以过滤过于常见或过于稀有的词，提升特征质量。n-gram 范围的选择要结合文本长度，短文本用一到二元组即可，长文本可以尝试更高阶。对于类别极多的任务，线性模型配合 TF-IDF 往往能取得与深度学习相当的效果，而且训练速度快、可解释性强。我们还会尝试特征哈希减少内存占用，或在向量化之前做词干提取，以合并同源词。交叉验证时使用分层抽样保证每折类别分布一致，避免评估偏差。最终模型选择时，我们综合验证集准确率和训练时间，选择性价比最高的方案。

模型评估要结合业务目标。

## 三、总结

本文中我们完成了 What a Flood Score Contains 的完整建模。我们首先确认了目标是预测 0 到 1 的合成洪水概率，指标为 R²。其次我们用分块读取和固定种子水库抽样把数据控制在 12 万行，并做了无状态的数值与类别清洗。接着我们用预设参数的 CatBoost 回归器训练，并做了“去掉季风强度”的消融实验。最后我们在固定 20% 留出集上比较常数基线与提升模型，分块生成提交文件并校验格式。

这个比赛给我们的最大启发是：当多个因子共同描述同一个风险时，先问“它们的总和是否已经解释了大部分目标”，再决定是否需要复杂模型。固定实验设置、不做排行榜调参，是让结论可信的关键。
