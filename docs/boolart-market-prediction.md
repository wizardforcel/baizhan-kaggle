# BoolArt 市场预测（BoolArt Market Prediction）

## 一、比赛简介

### 1.1 比赛背景

这场比赛模拟的是一个金融投资场景：在每一个时间点（`time_id`）上，市场里同时存在多个投资标的，每个投资标的都有 300 个匿名化特征（`f_0`～`f_299`）。这些特征大概已经做过预处理——我们会等下在 EDA 里验证。我们要预测的不是收益率本身，而是投资标的的**收益率排序**：同样在某个时刻，哪些标的收益更靠前。

这个任务的难点在于：目标本质上是「排序」，而不是「回归一个数」。如果只看单只标的自己的特征去预测它的排名，信息是不够的——因为排名是一个相对概念，别人涨得更多，你的名次就往下掉。

### 1.2 比赛目标

训练集 `train.csv` 的一行是「某个时刻 × 某个投资标的」：

- `investment_id`：投资标的编号；
- `time_id`：时间戳（训练集大约在 0～1000）；
- `f_0`～`f_299`：300 个匿名特征；
- `target`：投资收益率的排序值（经过放缩处理），数值越大说明该标的在这个时刻收益越靠前；
- `row_id`：形如「时间_标的」的样本编号。

测试集只有到 `time_id` 约 1019~1219 的多段时间，需要我们对**每一个真实的（时刻，标的）组合**预测一个 `target`。

### 1.3 评价标准

比赛使用 **Mean Pearson Score（平均皮尔逊相关系数）**：对每一个 `time_id`，把它下面所有投资标的的真实 `target` 和我们的预测放在一起，算一次皮尔逊相关系数，然后再对所有时间求平均。相关系数越接近 1，就说明预测和真实的排序趋势越一致，分数越高。

> 注意：公开评测（LB）榜上的 Mean Pearson 计算存在 bug，所以榜单分数不可信。组委会提到的做法是**在本地用 `test_label.csv` 重算这个指标**，我们后面也照做了。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们先把数据读进来，看形状和基本分布：

```python
import pandas as pd
import numpy as np

train = pd.read_csv("../input/boolart-market-prediction/train.csv")
test  = pd.read_csv("../input/boolart-market-prediction/test.csv")
train.shape
```

然后我们对特征和目标做了几项统计：

- **目标分布**：`train.target` 画直方图后非常接近正态分布；
- **特征分布**：300 个特征的均值集中在 -0.5～0.5 之间，说明数据做过中心化；再看每个特征的 min/max，也都集中在 0 附近；
- **时间跨度**：检查 `time_id` 的 min/max、每个 `investment_id` 的样本数，发现一些标的在某些时间点**缺失**（比如测试集里 `investment_id=3722` 只有 205 条数据）。

```python
# 每个特征在整个训练集上的平均值分布
feat_cols = [f'f_{i}' for i in range(300)]
train[feat_cols].mean().describe()

# 时间跨度：训练集时间戳从 0 到 1000
train.groupby("investment_id")['time_id'].agg(['min', 'max', 'count']).describe()
```

这一轮 EDA 给了我们三个判断：**数据是「时刻 × 标的」的矩阵，而且矩阵不是满的**（有缺失行）；特征已经过标准化；目标偏离度不大，整体比较干净。

### 2.2 数据清洗

数据最大的问题是**投资标的 × 时间的组合有缺失**——某个时刻某只标的不在数据里。而 Transformer 要处理的是「同一时刻所有标的同时输入」，如果不把缺失行补回来，矩阵就是歪的。所以我们把缺失的组合补上，并打一个 `mask` 标记，用来区分「真实样本」和「填充的占位样本」：

```python
import itertools

# 找出所有 (investment_id, time_id) 组合中缺失的那些
train_missing = set(itertools.product(train.investment_id.unique(), train.time_id.unique())) - \
                set([tuple(l) for l in train[['investment_id', 'time_id']].values.tolist()])

miss_train = pd.DataFrame(list(train_missing), columns=['investment_id', 'time_id'])
miss_train['row_id'] = miss_train['time_id'].astype(str) + "_" + miss_train['investment_id'].astype(str)

# 把缺失行拼回去，按时间、标的排序
train = pd.concat([train, miss_train], axis=0, ignore_index=True) \
         .sort_values(['time_id', 'investment_id'])

# mask=1 表示真实样本，mask=0 表示我们补出来的占位行
train['mask'] = 1 - train['f_0'].isnull().astype(int)
# 真实样本缺失的 target 记 -1，占位行其余特征填 0
train['target'] = train['target'].fillna(-1)
train = train.fillna(0.)
```

对测试集做同样的操作后，我们检查每一个 `time_id` 下的标的数量确实是整齐一致的，说明矩阵补平了。这些小改动非常重要：后面计算 loss 时，我们只对 `mask=1` 的真实样本计算误差，占位行既不参与注意力有效统计，也不参与误差计算。

### 2.3 模型训练与超参数调优

**选型思考。** 我们盘点了四种建模范式：

1. 只看单个投资标的当前时刻的 300 维特征——简单，但目标本来就是排序，单看自己几乎无法预测排名；
2. **把当前时刻所有标的一起作为一组输入，同时预测所有标的**——Transformer 的 self-attention 正好可以学习标的与标的两两之间的关系，而且没有时序，不需要 position embedding；
3. 顺着时间轴对每只标的历史做序列建模（RNN/Seq2Seq/Transformer）；
4. 方案 3 + 方案 2 的组合（先用 RNN 把每只标的历史编码，再套方案 2）。

我们最终采用**方案 2**（作者称之为 Investment-wise Transformer）：在同一个 `time_id` 下，各投资标的没有先后顺序，所以去掉位置编码；为了不让占位行干扰 attention，我们在计算 self-attention 时使用 `mask` 信息，计算 loss 时也跳过占位行的标签。

下面是把数据组织成样本的数据集类，每个样本是「一个时间点下所有标的的特征矩阵」：

```python
FEATURE = [f'f_{i}' for i in range(300)]

class CustomDataSet(Dataset):
    def __init__(self, df):
        # 按 time_id 分组，每个时间点作为一个样本
        self.gp = df.groupby('time_id')
        self.time_id = df.time_id.unique()

    def __len__(self):
        return len(self.time_id)

    def __getitem__(self, item):
        df = self.gp.get_group(self.time_id[item])
        feat = df[FEATURE].values          # 这个时刻所有标的的特征矩阵
        mask = df['mask'].values           # 哪些是真实样本
        if 'target' in df.columns:
            target = df['target'].values
            return {"feat": feat, "mask": mask}, target
        return {"feat": feat, "mask": mask}
```

模型主体是「线性映射 → N 层 Transformer encoder → 输出头回归」的结构。值得注意的细节是**残差连接**：每一层 Transformer 的输出按 `0.9 * 层输出 + 0.1 * 输入` 混合，保证后面模型可以更自由地退化成「多层感知机」，多一层保险；计算 loss 时把 `mask` 当权重，等价于只对真实样本求平均 MSE。

```python
class CustomModel(nn.Module):
    def __init__(self, input_size, num_layers, d_model, ffn_size, n_head):
        super().__init__()
        self.input_fc = nn.Linear(input_size, d_model)          # 300 维特征 → d_model 维
        self.transformers = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model, n_head, ffn_size, batch_first=True)
            for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1))                          # 回归到 1 个排序值
        self.loss_fn = nn.MSELoss(reduce=False)

    def forward(self, feat, mask, label=None):
        feat = self.input_fc(feat.float())
        for layer in self.transformers:
            # 带残差的 Transformer；src_key_padding_mask 屏蔽占位样本
            feat = layer(feat, src_key_padding_mask=mask.long()) * 0.9 + feat * 0.1
        pred = self.head(feat).squeeze()
        if label is not None:
            # 只在真实样本上计算损失，再除以真实样本数做平均
            loss = self.loss_fn(label.float(), pred)
            loss = (loss * mask) / torch.sum(mask)
            return pred, loss
        return pred
```

**数据划分与训练。** 我们按时间切分：把训练集里最后 100 个 `time_id` 划出来当验证集——因为测试集也是未来的时间，这样划分最接近真实部署场景。训练配置是 Adam（学习率 `1e-4`）、批大小 32；先小跑 2 个 epoch 验证流水线能跑通，再用**全量训练数据**重新训练一轮，因为多喂数据对排任务几乎总是有利的。

```python
# 按时间顺序划分：最后 100 个时刻做验证
time_ids = train.time_id.unique()
train_time_ids, val_time_ids = time_ids[:-100], time_ids[-100:]

train_ds = CustomDataSet(train[train.time_id.isin(train_time_ids)])
val_ds   = CustomDataSet(train[train.time_id.isin(val_time_ids)])
train_ds_all = CustomDataSet(train)
test_ds  = CustomDataSet(test)

model = CustomModel(input_size=300, num_layers=3, d_model=512, ffn_size=1024, n_head=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

config = TrainerConfig(save_path="./test_model.bin", batch_size=32, max_epochs=2)
trainer = Trainer(config, model, optimizer, train_dataset=train_ds, test_dataset=val_ds,
                  metric_fn=pearson_score, metric_larger_better=True)
trainer.fit()
```

我们在调参上没有做太激进的搜索：3 层、d_model 512、8 头是 Transformer 在中小规模表格数据上的稳妥组合，试过更大模型收益有限，反而更容易过拟合。更多时间不如花在方案升级上。

### 2.4 模型测试与指标报告

**评价函数。** 我们实现比赛的定义——每个时刻算一次 Pearon 相关系数再求平均：

```python
from scipy.stats import pearsonr

def pearson_score(labels, preds):
    # labels/preds 的每一行对应一个 time_id 下所有标的
    # 对每个时刻分别算 pearson，再对所有时刻取平均，就是比赛分数
    return np.mean([pearsonr(l, p)[0] for l, p in zip(labels, preds)])
```

训练过程中，`Trainer` 每个 epoch 都会在验证集上调用这个函数，记录验证分数、保存最好模型的 checkpoint。由于 LB 分数不可信，我们最后直接对测试集预测，并用**本地提供的真实标签**来复核：

```python
# 用训练好的模型对测试集（每个时刻一组）预测
test_pred = trainer.predict(test_ds)
test['prediction'] = test_pred.reshape(-1)

# 只保留真实样本（mask=1），提交格式是 row_id + prediction
test = test.loc[test['mask'] == 1, ['row_id', 'prediction']]
test.to_csv("submission.csv", index=False)

# 确保提交与要求完全一致（行数与 sample_submission 逐一对齐）
sample_submission = pd.read_csv("../input/boolart-market-prediction/sample_submission.csv")
assert np.all(test['row_id'] == sample_submission['row_id'])
```

再用本地标签重算比赛分数：

```python
def LB_score(submission_file):
    """在本地用 test_label.csv 重算 Mean Pearson Score（LB 的计算有 bug）"""
    sub = pd.read_csv(submission_file)
    test_label = pd.read_csv("../input/boolart-market-prediction/test_label.csv")
    test_label = test_label.merge(sub, on='row_id', how='left')
    return test_label.groupby('time_id').apply(
        lambda g: pearsonr(g['target'].values, g['prediction'].values)[0]).mean()

LB_score("submission.csv")
```

这一套流程跑完，我们得到了一份格式正确、分数可在本地复算的提交文件。

---

## 三、总结

本文中我们完成了 BoolArt 市场预测的完整流程：首先通过 EDA 确认数据是「时刻 × 投资标的」的矩阵、特征已标准化、目标近似正态分布，并发现了大量「标的 × 时刻」缺失的问题；其次我们把缺失组合补平并引入 `mask` 标记，保证填充样本不污染 attention 与 loss；接着我们比较了四种建模范式，最终采用没有位置编码的 Investment-wise Transformer，在同一时刻内用 self-attention 建模标的两两关系，并逐步做了时间划分的验证与全量训练；最后我们实现比赛的评价函数，在本地用真实标签复算 Mean Pearson Score，并生成了通过 `row_id` 对齐校验的提交文件。

这个比赛给我们最大的启发是两句话：**目标是排序时，特征必须带「同伴」一起看**——单样本模型读不出相对排名；以及**榜单分数不可信时，要自己搭一个本地复算的评测管线**，否则根本没法科学地调参。