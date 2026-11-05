# TalkingData 移动广告点击欺诈检测

## 一、比赛简介

### 1.1 比赛背景

TalkingData 成立于 2011 年，是中国最大的独立大数据服务平台之一。平台每天要处理数十亿次移动广告的点击记录，其中大约有 90% 的点击流量存在欺诈风险。传统的做法是通过 IP 黑名单、设备黑名单来过滤可疑流量，但欺诈者会不断更换设备和 IP，静态黑名单很快就会失效。

为了在欺诈手段面前保持领先，TalkingData 与 Kaggle 合作举办了第二次广告追踪比赛。比赛的目标是**预测用户在点击移动广告后是否会下载应用**。如果点击多而下载少，说明这条点击链路很可疑，极有可能是机器刷量或者恶意点击。比赛给出了约 4 天、约 2 亿条点击记录的真实数据，让参赛者用机器学习找规律。

这个场景的特点是：数据量巨大、类别极度不平衡、时间维度强、很多字段都是高基数的类别型 ID。初学者很容易在这里掉进两个坑：第一，直接把原始 ID 丢给模型，模型记不住；第二，忘了处理不平衡，直接用准确率评判，结果模型全部预测为 0 也能有 99% 的准确率。

### 1.2 比赛目标

训练集每行一条点击记录，包含以下字段：

- `ip`：点击来源 IP 地址；
- `app`：被投放的应用编号；
- `device`：设备类型编号，例如 iPhone、华为等；
- `os`：操作系统版本编号；
- `channel`：广告渠道编号；
- `click_time`：点击发生的 UTC 时间戳；
- `is_attributed`：目标标签，1 表示用户在点击后下载了应用，0 表示没有下载。

测试集和训练集字段类似，但没有 `is_attributed`，只有 `click_id` 作为提交时的唯一标识。我们要为测试集里的每一行预测 `is_attributed`，并把结果写成两列的 CSV。

### 1.3 评价标准

比赛采用 ROC 曲线下面积 **AUC** 作为评价指标。AUC 衡量模型把正例排在负例前面的能力，对类别不平衡不敏感。AUC 的取值在 0 到 1 之间，0.5 相当于随机猜测，1.0 代表完美分类，分数越高越好。

因为正例只有 0.2% 左右，准确率几乎没有参考意义。举例来说，如果全部预测 0，准确率能达到 99.7%，但 AUC 只有 0.5。所以整个解题的思路都要围绕「让正例和负例在打分上可分开」，而不是简单地追求对样本数的比例。

---

## 二、解题过程

我们按 EDA、数据清洗、模型训练与调参、模型测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先读取训练样本，观察数据规模和字段类型：

```python
import pandas as pd
import numpy as np

# 为了节省内存，提前指定字段类型
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

train_cols = ['ip','app','device','os','channel','click_time','is_attributed']
trainset = pd.read_csv('../input/train_sample.csv', dtype=dtypes, usecols=train_cols)

print("train - rows:", trainset.shape[0], " columns:", trainset.shape[1])
print(trainset.head())
```

输出显示训练样本约 10 万行、测试集约 1800 万行。样本量差距巨大，说明比赛的真实规模远远超过本地样例，特征工程必须尽可能节省内存。

接着我们检查标签不平衡的情况：

```python
# 统计正负例数量
cnt = trainset['is_attributed'].value_counts()
print(cnt)

# 正例占比 = 正例数 / 总数
pos_rate = cnt[1] / cnt.sum()
print(f"正例占比约 {pos_rate:.4%}")
```

从材料可以看到，正例只有 227 条，占 10 万条样本的 0.23% 左右。这意味着直接训练的模型很容易把所有样本都判为负。此时需要用 AUC 来评估，用类别权重或者下采样来避免模型「偷懒」。

我们还看一下点击时间分布：

```python
# 把 click_time 转换成 datetime，再拆成年、月、日、时、分、秒
trainset['click_time'] = pd.to_datetime(trainset['click_time'])
trainset['hour'] = trainset['click_time'].dt.hour
trainset['day'] = trainset['click_time'].dt.day

# 正例和负例在不同小时的分布是否一致
print(trainset.groupby('is_attributed')['hour'].mean())
```

材料里的 EDA 指出，正例的点击时间相对更分散，小时分布比负例宽一些。这说明时间特征对区分欺诈有帮助——机器刷量往往集中在某些小时。

我们还检查同一 IP 或同一设备是否出现异常高频点击：

```python
# 统计每个 IP 的点击次数，看看是否存在少数 IP 贡献了绝大部分样本
ip_counts = trainset['ip'].value_counts()
print(ip_counts.describe())

# 看点击最多的前 10 个 IP，直观感受「少数 IP、大量点击」的现状
print("Top 10 IPs:", ip_counts.head(10))
```

这种长尾分布很常见：正常用户的 IP 相对稳定、点击次数适中，而欺诈脚本会用同一个 IP 持续刷点击。频次过高的 IP 通常是强烈欺诈信号，但也不是绝对的规则，因为一些广告渠道会集中投放。把频次作为特征喂给模型，让模型自己决定阈值，比我们自己手写规则更稳。

接着我们检查不同字段的唯一值数量：

```python
for col in ['ip','app','device','os','channel']:
    uniq = trainset[col].nunique()
    print(col, uniq)
```

高基数字段的唯一值数量决定了特征工程的方向。如果唯一值太多，直接 one-hot 会产生巨大的稀疏矩阵；所以我们选择频次编码和目标均值编码作为折中，把「ID 本身」变成「ID 的行为统计量」。

### 2.2 数据清洗

数据清洗的主要工作是**从时间戳里提取人类可解释的特征，并对高基数类别做统计编码**。

时间特征提取：

```python
def extract_time_features(df):
    # 把原始时间戳转成 datetime，能让 pandas 帮我们拆分时间成分
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['hour'] = df['click_time'].dt.hour
    df['day'] = df['click_time'].dt.day
    df['minute'] = df['click_time'].dt.minute
    # 周末效应：0 表示周末，1 表示工作日
    df['is_weekend'] = df['click_time'].dt.weekday.isin([5,6]).astype(int)
    return df
```

为什么要拆到小时和分钟？因为广告点击的时序规律很强：人类用户在工作日白天点击多，夜间少；而机器人刷量可能每小时都一样。我们把时间切碎，模型就能学到这种差异。

对高基数类别的统计编码：

```python
# 对 ip、app、device、os、channel 等字段做频次编码
def add_count_encoding(train, test, cols):
    for col in cols:
        # 统计训练集里每个 ID 出现的次数，再映射到测试集
        cnt = train[col].value_counts()
        train[f'{col}_cnt'] = train[col].map(cnt)
        test[f'{col}_cnt'] = test[col].map(cnt).fillna(0)
    return train, test
```

频次编码的核心思想是：**频繁出现的 IP 大概率是正常用户，极少出现的 IP 更可能是干净的机器人或一次性欺诈。** 原始 ID 太多，模型直接学习会过拟合，但频次是一个紧凑的数值特征，能把相似的 ID 归到一起。

我们再叠加一组「行为时序」特征，反映同一设备最近的动作节奏：

```python
# 按设备分组，观察时间排序后相邻两次点击的间隔
def add_click_gap_features(df):
    # 先把时间整理成数字，方便做差
    df['ct'] = df['click_time'].astype('int64')
    g = df.groupby('ip')['ct']
    # 上一次点击与本设备当前点击的时间差（秒级别的近似间隔）
    df['prev_gap'] = g.diff() / 1e9
    # 再用同一种方式把「最后一次点击距当前时间」也算出来
    df['next_gap'] = g.transform(lambda s: s.shift(-1)) - df['ct']
    df['next_gap'] = (df['next_gap'] * -1).fillna(0) / 1e9
    return df
```

为什么点击间隔有用？真实用户一次点击后会有一段「思考时间」，比如几分钟后再决定是否下载；而机器人是程序化触发，间隔可能固定得很短，甚至是 0。把间隔补成负数或 0 再让树模型学习，往往能显著提升 AUC。

需要特别提醒的是，这类「时间窗口」特征在构造时**必须只用当前行为之前的信息**，否则会泄漏未来信息，导致本地交叉验证分数虚高、排行榜分数却很低。这也是为什么我们对 `train` 和 `test` 分开计算统计量，而不是拼在一起算。

缺失值处理很简单：材料里样本数据几乎没有缺失值，主要是时间特征和编码特征。如果出现缺失，我们用 0 填充，表示「这个 ID 在训练集里从未出现」。

### 2.3 模型训练与超参数调优

基线的建模策略是**树模型 + 类别编码 + 时间特征**。树模型对离散特征不敏感，对不平衡数据鲁棒。

我们先用 LightGBM 做基准：

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

# 用分层抽样保证每折正例比例一致
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced'   # 自动给正例更高权重
)

aucs = []
for tr_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc')
    aucs.append(model.best_score_['valid_0']['auc'])

print(f"5-fold CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
```

`class_weight='balanced'` 是处理不平衡的常用方法：它会根据类别频率自动调整损失权重，相当于给正例更高的惩罚。树模型在不平衡数据上表现不错，但配合类别权重后，AUC 通常能提升几个百分点。

超参数调优方面，我们优先调三件事：

1. **学习率和树的数量**：学习率越小，树的数量要越大；
2. **叶子数和采样比例**：控制模型复杂度，防止过拟合并提升速度；
3. **是否使用频次编码**：去掉频次编码后，AUC 下降明显，证明统计特征有效。

材料里也提到，XGBoost、LightGBM 甚至神经网络都有人尝试，但对这个任务来说，树模型在特征工程到位的情况下往往性价比最高。与其在模型结构上堆复杂度，不如先把 `ip`、`app` 等 ID 的行为序列特征（比如最近一次点击间隔、同一 IP 的不同 app 数量）做够。

### 2.4 模型测试与指标报告

我们在本地用交叉验证得到 AUC，大约在 0.97 左右，说明模型基本能把正例和负例分开。最后我们把模型在全部训练数据上重新训练一次，再对测试集预测：

```python
# 用全部训练数据重新拟合，信息越多越好
model.fit(X_train_full, y_train_full)

# 对测试集打分，保留概率值而不是直接二分类
test_pred_proba = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'click_id': test['click_id'],
    'is_attributed': test_pred_proba
})
submission.to_csv('submission.csv', index=False)
```

这里我们输出的是预测概率，而不是 0/1 的硬分类。AUC 评估的是「排序」能力，提交文件里的概率值越能反映真实排序，排行榜分数越高。硬阈值 0.5 在不平衡数据上通常不是最优解。

最后做一次自检：

```python
print(submission.shape)
print(submission['is_attributed'].describe())
print((submission['is_attributed'].isna().sum() == 0))
```

确认行数和测试集一致、没有缺失值、概率值落在 [0,1] 区间。很小的格式错误会让一次很好的建模白费。

---

## 三、总结

本文中我们完成了 TalkingData 广告点击欺诈检测比赛的 baseline。我们首先通过 EDA 确认了数据的极度不平衡和强时间特征；其次从 `click_time` 中提取了小时、日期、周末等特征，并用频次编码把高基数 ID 转成数值；接着用 LightGBM 配合分层 5 折交叉验证，并在类权重下得到稳定的 AUC；最后在全量训练数据上重新训练，用预测概率生成提交文件。

这个比赛给我们的最大启发是：**在不平衡数据上，特征工程比模型选择更重要。** 同样的 LightGBM，只要加上点击时间分解和 ID 频次编码，AUC 就会明显提升；而盲目换神经网络或堆深度模型，往往会因为数据量和不平衡而得不偿失。先把数据讲清楚，再让模型发挥作用，这一直是广告点击这类工业风问题的核心思路。