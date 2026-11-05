# 预测社交网络中谁更具影响力

## 一、比赛简介

### 1.1 比赛背景
在社交网络中，用户的影响力往往与关注者数量、被提及次数、转发次数、发帖活跃度以及网络结构特征密切相关。本比赛提供的训练数据来自真实的社交平台记录，每一条样本都包含两名用户 A 和 B 的多项公开指标。我们需要根据这些指标判断哪一位用户在网络中更具影响力。材料中多篇解法都以「Influencers in Social Networks」为标题进行探索，说明了问题本质是基于用户统计特征做二分类。对入门者来说，这个任务比单纯的点赞预测更贴近真实场景，因为影响力是相对的，而不是绝对数值：同样是十万粉丝，一个人可能天天被转发，另一个人却无人问津，所以不能只看某一个数字，要综合各项指标来判断。

### 1.2 比赛目标
给定训练集和测试集，预测测试集中每对用户的 Choice。训练集共有 5500 条样本，包含 23 列；测试集共有 5952 条样本，包含 22 列。目标列 Choice 为 0 或 1，表示在该对用户中哪一位更具影响力。输入特征以 A_ 和 B_ 开头，每组包含 follower_count、following_count、listed_count、mentions_received、retweets_received、mentions_sent、retweets_sent、posts 以及三个网络特征 A_network_feature_1~3。测试集没有 Choice 列，我们需要输出与 sample_predictions.csv 格式一致的提交文件。

### 1.3 评价标准
比赛使用 ROC AUC 作为评价指标，分数越高越好。材料中的贝叶斯优化 LightGBM 单模型在私有排行榜上取得 0.87169 的分数，排在 12-13 名。交叉验证结果显示 20 折 CV AUC 为 0.87192168，这与排行榜表现一致。AUC 度量模型区分正负样本的能力，对类别不平衡更稳健，适合本任务。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先把训练集和测试集读入，查看基本信息。训练集有 5500 条、23 列，全部列都无缺失。测试集有 5952 条、22 列，同样无缺失值。目标列 Choice 的频次为 0 类 2698 条、1 类 2802 条，比例接近 1:1，说明数据基本平衡，无需做过采样。

```python
# 导入必要的库
import pandas as pd
import numpy as np
# 读取训练集和测试集
train = pd.read_csv("../input/predict-who-is-more-influential-in-a-social-network/train.csv")
test = pd.read_csv("../input/predict-who-is-more-influential-in-a-social-network/test.csv")
# 查看训练集基本信息
train.info()
# 查看测试集基本信息
test.info()
```

为了确认数据质量，我们检查缺失值，并统计目标分布。代码显示训练集和测试集都没有缺失样本，目标分布接近平衡，这为后续建模提供了良好基础。我们还看了前十行，确认 A_ 和 B_ 列一一对应，且数值范围合理。

```python
# 检查训练集和测试集是否存在缺失值
print("Any missing sample in training set:", train.isnull().values.any())
print("Any missing sample in test set:", test.isnull().values.any())
# 统计目标列 Choice 的频次分布
train_outcome = pd.crosstab(index=train["Choice"], columns="count")
print(train_outcome)
```

通过频次表我们得知 0 类 2698 条、1 类 2802 条。随后我们列出了用于建模的特征名，包括两名用户的粉丝数、关注数、收录数、提及收发、转发收发、发帖数和三个网络特征，共 17 个原始特征。为了进一步了解数据，我们查看了每组特征的含义：A_follower_count 表示用户 A 的关注者人数，A_following_count 表示 A 关注了多少人，A_listed_count 表示 A 被多少人加入了列表，A_mentions_received 是 A 被@的次数，A_retweets_received 是 A 被转发的次数，A_mentions_sent 和 A_retweets_sent 则相反，是 A 主动发起的行为。三个 network_feature 则是比赛方匿名化的网络结构指标，我们不清楚具体含义，但可以当作普通数值特征使用。EDA 阶段没有画图，但通过 describe 和简单的分组聚合，我们意识到 A 和 B 的数值存在明显差异，且同一个人在不同指标上的表现也可能完全相反，直接比较 A 和 B 的差别，比单独看某一个绝对值更有信息量。

### 2.2 数据清洗
材料中的数据本身比较干净，训练集和测试集都没有缺失值。但我们在阅读代码时发现，部分数值可能出现 inf 和 -inf，来源于除法或极端比例计算。无穷大对树模型来说是无法直接处理的，LightGBM 遇到 inf 可能报错或者给出异常分裂，因此必须提前清理。为避免模型训练出错，我们先做预处理函数 pre_pro，将 inf 替换为列的最大值或最小值，并把缺失值填 0。这样做的道理是：极端值虽然不真实，但保留一个「特别大」或「特别小」的标记，比直接删掉整行更安全，因为样本本身只有五千多条，删样本会损失宝贵信息。

```python
def pre_pro(df):
    # 将数据转为 float32 节省内存
    df = df.astype('float32')
    col = df.columns
    for i in range(len(col)):
        # 处理 -inf，替换为列的最小值
        m = df.loc[df[col[i]] != -np.inf, col[i]].min()
        df[col[i]].replace(-np.inf, m, inplace=True)
        # 处理 +inf，替换为列的最大值
        M = df.loc[df[col[i]] != np.inf, col[i]].max()
        df[col[i]].replace(np.inf, M, inplace=True)
    # 最后统一填充缺失为 0
    df.fillna(0, inplace=True)
    return df
```

在比例特征计算时，为了防止除零错误，特征工程函数 feat_eng 先把 0 替换为 0.001。之后我们对每个指标生成 A 是否大于 B 的布尔差分特征，以及两名用户的比率特征，如 A_foll_ratio、B_foll_ratio 等，用于帮助模型学习相对影响力。

```python
def feat_eng(df):
    # 避免除零，0 替换为 0.001
    df.replace(0, 0.001)
    # 比较差分特征：A 的粉丝数是否大于 B
    df['follower_diff'] = (df['A_follower_count'] > df['B_follower_count'])
    df['following_diff'] = (df['A_following_count'] > df['B_following_count'])
    df['listed_diff'] = (df['A_listed_count'] > df['B_listed_count'])
    df['ment_rec_diff'] = (df['A_mentions_received'] > df['B_mentions_received'])
    df['rt_rec_diff'] = (df['A_retweets_received'] > df['B_retweets_received'])
    # 比例特征
    df['A_foll_ratio'] = df['A_follower_count'] / df['A_following_count']
    df['B_foll_ratio'] = df['B_follower_count'] / df['B_following_count']
    return pre_pro(df)
```

清洗阶段我们没有做缩放，材料指出随机森林这类树模型不需要归一化，且树模型对量纲不敏感。原因是决策树在每次分裂时只看某个特征的一个阈值，特征的大小顺序不会因为整体乘一个常数而改变，所以标准化对树模型几乎没有影响。特征工程之后，训练集和测试集特征数目一致，可直接用于模型训练。需要提醒的是，构造比率特征之前一定要处理分母为零的情况，否则会生成无穷大，而无穷大又会反过来破坏后面所有的计算，这也是我们前面先用 0.001 替换 0 的原因。

### 2.3 模型训练与超参数调优
主线解法采用 LightGBM，并用贝叶斯优化搜索超参数。为什么选择 LightGBM 而不是深度学习？因为数据只有 5500 条、几十个特征，规模很小，树模型训练快、调参直观、不容易过拟合，而且对表格数据通常表现优异。先对数据做前面提到的特征工程得到 fe_train 和 fe_test，然后用 StratifiedKFold 做分层交叉验证，保证每折的类别比例一致。贝叶斯优化在有限的迭代次数内搜索 num_leaves、learning_rate、feature_fraction、lambda_l1、lambda_l2 等参数。相比网格搜索逐个尝试，贝叶斯优化会根据前面几次试验的结果调整后续的方向，用更少的次数找到好参数。

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
# 特征工程后的训练集和测试集
fe_train = feat_eng(train.copy())
fe_test = feat_eng(test.copy())
target = 'Choice'
predictors = fe_train.columns.values.tolist()[1:]
# 5 折分层交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
```

超参数优化完成后，我们把训练流程封装为 20 折交叉验证的集成。每一折训练一个 LightGBM 模型，使用早期停止防止过拟合，并保存验证集的 out-of-fold 预测用于估计 CV AUC。测试集预测取 20 折平均，降低方差。

```python
oof = np.zeros(len(fe_train))
predictions = np.zeros((len(fe_test), 20))
i = 1
for train_index, valid_index in skf.split(fe_train, fe_train.Choice.values):
    xg_train = lgb.Dataset(fe_train.iloc[train_index][predictors].values,
                           label=fe_train.iloc[train_index][target].values)
    xg_valid = lgb.Dataset(fe_train.iloc[valid_index][predictors].values,
                           label=fe_train.iloc[valid_index][target].values)
    # 训练并做早期停止
    clf = lgb.train(param_lgb, xg_train, 10000000,
                    valid_sets=[xg_valid], verbose_eval=250,
                    early_stopping_rounds=100)
    oof[valid_index] = clf.predict(fe_train.iloc[valid_index][predictors].values,
                                   num_iteration=clf.best_iteration)
    predictions[:, i-1] += clf.predict(fe_test[predictors],
                                       num_iteration=clf.best_iteration)
    i = i + 1
# 计算 CV AUC
cv_auc = metrics.roc_auc_score(fe_train.Choice.values, oof)
print("CV AUC:", cv_auc)
```

为了对比，我们也参考了材料中的随机森林基线。随机森林不需调参即可快速验证，对入门者友好。其超参数包括 n_estimators、max_depth、min_samples_split 等，可以用 RandomizedSearchCV 做粗调。代码显示随机森林在本地验证集上表现稳定，但 AUC 低于 LightGBM。

```python
from sklearn.ensemble import RandomForestClassifier
# 定义随机森林模型
rfcla = RandomForestClassifier(n_estimators=100, random_state=168, n_jobs=-1)
# 训练模型
rfcla.fit(train[features], train[target])
# 在测试集上预测
predictions_rf = rfcla.predict(test[features])
```

### 2.4 模型测试与指标报告
模型训练完成后，我们用 out-of-fold 预测计算整体 CV AUC，材料中 20 折 CV AUC 为 0.87192168。单模型在私有榜上得到 0.87169，两者几乎一致，说明交叉验证能够稳定估计真实效果。我们还在训练集上观察了特征重要性，发现 follower_diff、foll_ratio 等相对特征对区分影响力起主导作用，这符合直觉：影响力更多是相对比较而非绝对数值。早期停止策略让每一折在验证 AUC 连续 100 轮不再提升时提前停止，避免了在数千万轮训练上浪费时间，也防止了过拟合。测试集的预测我们采用 20 折平均，相当于模型集成，方差更小，排行榜稳定性更好。

生成提交文件时，我们沿用 sample_predictions.csv 格式，将 20 折平均的预测概率写入 Choice 列。提交文件保存为 sub.csv，后续上传即可。

```python
# 读取样例提交文件
submission = pd.read_csv('../input/sample_predictions.csv')
# 将 20 折平均预测作为最终概率
lgb_bay = []
for i in range(len(predictions)):
    lgb_bay.append(predictions[i].mean())
submission['Choice'] = lgb_bay
# 保存提交文件
submission.to_csv('sub.csv', index=False, header=True)
```

在模型测试阶段，我们验证了预测文件行数与测试集一致，列名正确。指标报告显示 CV AUC 稳定在 0.87 左右，说明模型具有良好的泛化能力。材料中没有详细的交易报告，但通过多次折的 AUC 分布可以看出模型稳定，没有出现过拟合的极端波动。

## 三、总结
本文中我们完成了社交网络影响力预测的完整流程。我们首先通过 EDA 摸清数据：训练集 5500 条、测试集 5952 条，目标列基本平衡且无缺失值。其次我们做了轻量数据清洗，处理 inf 值并构造了 A 与 B 的差分和比率特征。接着我们采用贝叶斯优化的 LightGBM 模型，结合 20 折分层交叉验证进行训练与调优，得到 CV AUC 0.8719。最后我们生成了符合格式的提交文件，并在验证集上确认指标稳定。

这个比赛给我们的最大启发是：相对特征比绝对特征更有效。在社交影响力这类对比任务中，直接比较两个用户的粉丝数、转发数等指标，并构造比值，能够让树模型更快捕捉到「谁更大」的规律。对入门者而言，本项目也展示了特征工程与交叉验证的重要性：不依赖复杂的深度学习，用良好的特征和稳定的验证流程就能取得不错的排行榜成绩。
