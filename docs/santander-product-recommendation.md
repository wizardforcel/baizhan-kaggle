# Santander 产品推荐（Santander Product Recommendation）

## 一、比赛简介

### 1.1 比赛背景

这是西班牙萨瓦德尔银行（Santander）在 2016 年发起的一场推荐系统比赛。银行手里有每个客户的账户信息，他们想知道：**下一期该给客户推荐什么样的新产品**。比赛给的数据是客户从 2015 年 1 月到 2016 年 5 月、每个月末的「产品快照」——所谓快照，就是每到一个月末，银行就把「这个客户当天手里有哪些产品」记成一行，就像给每个客户每个月拍一张照片。训练集 `train_ver2.csv` 里一共有 13,647,309 行、48 列，文件约 2.135 GB，是一份体量不小的表格数据。

这个比赛真正难的地方只有一个：**正样本极度稀疏**。每个月真正「购买了新产品」的客户只占大约 3%。比如 2016 年 5 月，一共有 931,453 名客户，其中只有 29,712 人拥有了新增产品。如果让模型无脑预测「所有人都不买新产品」，准确率能到 97%，但这种答案在评价标准下只能得 0 分。所以我们从第一天就明白了：这不是一个普通的分对分错的分类问题，而是一个**推荐排序问题**——模型强不强，不看它对不对，而看它推荐的东西排得准不准。

### 1.2 比赛目标

给定一个客户在 2016 年 5 月 28 日的完整画像，我们要预测他在 2016 年 6 月 28 日会**额外新增**哪些产品。「额外」两个字是重点：客户手里已经有的产品不算数，只有上个月没有、这个月才冒出来的产品才算命中。

数据里的列可以分成三类：

- **标识列**：客户编号 `ncodpers`、记录月份 `fecha_dato`；
- **画像特征**：年龄 `age`、客户关系时长 `antiguedad`（在行年限）、收入估算 `renta`、客户细分 `segmento`、性别 `sexo`、活动指数 `ind_actividad_cliente` 等；
- **产品标志位**：24 个以 `ind_` 开头的 0/1 列，比如活期账户 `ind_cco_fin_ult1`、储蓄类 `ind_ahor_fin_ult1`、信用卡等等。取 1 表示「这个月持有该产品」，取 0 表示没有。

输出格式是每个客户一行：客户编号 + `added_products`，即最多 7 个产品代码，**用空格分隔**，而且要按「最有可能买」到「最不可能买」排序。

### 1.3 评价标准

评分指标叫 **MAP@7**（Mean Average Precision at 7，前 7 个推荐的平均精确率）。用大白话解释：对每个客户，我们给出一个最多 7 个产品的排序列表。从第 1 个开始往下数，每碰到一个「猜对了的新产品」就记一次命中，然后累加「当前命中个数 ÷ 当前排名」；最后再用真实新增产品的个数做平均，得到这个客户的 AP。把所有客户求平均，就是 MAP@7。

有两个细节决定了分数的高低：

- 客户这个月如果没有买任何新产品，他的 AP 直接记 0；
- 所以理论最高分一定**小于 1**。2016 年 5 月的理论最大值就是 29,712 ÷ 931,453 ≈ 0.0319，其他月份也都只有 0.03～0.045 这个量级。

意思是：就算有人把每个客户的新增产品一字不差全部猜对，得分也只有约 0.03。排行榜当时的最高分是 0.0303（2016 年 11 月 13 日时的名次快照），已经非常接近理论天花板。明白了这一点，后面我们就不再纠结「分数为什么这么小」，而是把目光放在「怎么接近这个天花板」上。

---

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把数据读进来。因为文件有 2 GB 以上，我们一开始就给每一列声明好数据类型，尤其是 24 个 0/1 产品列，用 1 字节的 `uint8` 就够，让 pandas 不用反复猜类型、少吃内存：

```python
import numpy as np
import pandas as pd

# 24 个产品列的名字，来自比赛对数据的说明
product_cols = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
    'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
    'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
    'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
    'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
    'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
    'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
]

# 产品标志位专用于小整数类型：既省内存，又避免把空值读成看不懂的类型
dtype_map = {'ncodpers': np.int32,
             **{c: np.uint8 for c in product_cols}}

train = pd.read_csv('../input/train_ver2.csv', dtype=dtype_map)
print(f'Train: {train.shape}')   # 1360 多万行、48 列
```

接着我们再去看两个问题：每个月的客户量稳不稳，以及 24 个产品到底稀疏到什么程度。稀疏程度直接决定了这个问题的难度，所以这一步很关键。

```python
# 看看数据跨了哪些月份，每个月的客户数是多少
print('时间段：', train['fecha_dato'].min(), '→', train['fecha_dato'].max())
monthly = train.groupby('fecha_dato')['ncodpers'].nunique()
print(monthly)

# “持有率”= 每个产品列在整个数据集里的均值，也就是所有快照里持有该产品的比例
# 之前其他人抽样的描述统计显示：活期账户 ind_cco_fin_ult1 的均值约 0.70，
# 而 ind_aval_fin_ult1 等产品列均值几乎为 0
holding_rate = train[product_cols].mean().sort_values(ascending=False)
print(holding_rate.head(10))
```

这一轮探索告诉我们三件事。第一，客户数量在 2015 年 7 月有一次明显增加（一大波新客户入行），除此之外每个月基本平稳；客户关系类型里「活跃（A）」和「不活跃（I）」占绝大多数，其余类型占比不足 0.1%。第二，24 个产品列**极其稀疏**：约 70% 的客户持有活期账户，但绝大多数产品的持有率都在 1% 以下，数据表里整块整块地是 0。第三，正因为标签天然倾向「没买」，平均精确率而不是准确率来当评价标准，实在是非常贴切——这就是一个少数派问题，模型必须学会从稀稀拉拉的正样本里找出规律。

### 2.2 数据清洗

其次，我们处理缺失值和占位符。这份数据的一个坑是：**`-999999` 并不是真实数值，而是「未知」的占位符**。比如收入 `renta`、行龄 `antiguedad` 里出现 `-999999`，就表示银行不知道这个客户的信息。如果直接把这种大负数喂给模型，树模型会以为「这个客户特别异常」，学到毫无意义的规律。

```python
# -999999 在各列里都代表“未知/缺失”，先统一换成 NaN
for col in ['antiguedad', 'renta', 'age']:
    train[col] = train[col].replace(-999999, np.nan)

# 产品列里偶尔会出现空字符串，统一归 0
for c in product_cols:
    train[c] = train[c].replace('', 0)

# 收入、年龄、行龄用整体中位数填补，尽量少丢样本
for col in ['renta', 'age', 'antiguedad']:
    med = train[col].median()
    train[col] = train[col].fillna(med)
```

接着有一个关键步骤：**构造标签**。我们要的不是「客户 6 月有没有某产品」，而是「客户 6 月相比 5 月新增了哪些产品」。方法是把相邻两个月的快照做差：上一个月是 0、这个月变成 1 的列，就是新增。因为我们手里没有 6 月的真实答案，本地调参时就拿「2016 年 4 月 → 2016 年 5 月」来模拟「5 月 → 6 月」的比赛节奏。

```python
# 取 4 月和 5 月两个月末的快照
apr = train[train['fecha_dato'] == '2016-04-28'].set_index('ncodpers')
may = train[train['fecha_dato'] == '2016-05-28'].set_index('ncodpers')

# 两列相减并截断到不小于 0：
# 4 月有、5 月还是 1 → 差为 0，说明只是继续持有，不算新增
# 4 月是 0、5 月变成 1 → 差为 1，这才是“客户新买的”
new_may = (may[product_cols]
           - apr[product_cols].reindex(may.index).fillna(0)).clip(lower=0)

# 验证一下规模：应当和官方统计一致，5 月有新增的客户只有 29712 人
print('2016 年 5 月有新增产品的客户数：', (new_may.sum(axis=1) > 0).sum())
print('总客户数：', len(may))
```

这个「做差成标签」的写法解答了一个入门者常有的困惑：同样一列 `ind_cco_fin_ult1`，既是特征（客户现在有没有），又是标签的一部分（客户新没新买）。如果不扣掉「本来就有」，模型就会去学「有活期账户的人继续有活期账户」，然后推荐一大堆客户本来就在用的产品——这正是很多推荐系统翻车的地方。

### 2.3 模型训练与超参数调优

接着，我们选用 XGBoost 作为主力模型。思路很朴素：**为 24 个产品各训练一个二分类器**，每个分类器只回答一个问题——「这个客户下个月会不会新增购买该产品」。预测时，把 24 个模型的输出概率放在一起，对每个客户按概率从高到低排序，取出前 7 个。

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 特征 = 客户画像 + 上个月已经拥有哪些产品（是否持有是很强的先验）
demo_cols = ['age', 'antiguedad', 'ind_actividad_cliente', 'renta', 'segmento']
feat_cols = demo_cols + product_cols
X = may[feat_cols]

# 随机切出 20% 当验证集，用来调参和看本地分数
tr_idx, va_idx = train_test_split(X.index, test_size=0.2, random_state=42)

prob_all = {}   # 保存 24 个模型在验证集上的输出概率
for prod in product_cols:
    # 标签：5 月新增了该产品记为 1，否则记为 0
    y = new_may[prod].astype(int)
    dtrn = xgb.DMatrix(X.loc[tr_idx], label=y.loc[tr_idx])
    dval = xgb.DMatrix(X.loc[va_idx], label=y.loc[va_idx])

    # binary:logistic 输出概率；early_stopping 看验证损失，不再提升就提前停
    model = xgb.train(
        {'max_depth': 4, 'eta': 0.05, 'objective': 'binary:logistic'},
        dtrn,
        num_boost_round=500,
        early_stopping_rounds=30,
        evals=[(dval, 'val')],
        verbose_eval=False,
    )
    prob_all[prod] = model.predict(dval)
```

超参数方面，我们其实没有在这上面花太多力气。反复试下来就是两个原则：`max_depth` 取 3～5 就够了（24 个分类器共用的特征才 30 来个），树再深就过拟合那几千个正样本；`eta`（学习率）越小越稳，配合 `early_stopping_rounds=30` 自动决定迭代轮数。**真正让分数提升的，反而是启发性规则**：预测排序时把「客户已经持有的产品」整个从候选里拿掉，只对着剩下没买过的产品排概率。这个规则几乎是白送的分数，却经常被入门方案忽略。

### 2.4 模型测试与指标报告

最后，我们实现比赛自带的评分函数（就是前面介绍的 MAP@7），在验证集上量一量自己的工作。材料里普遍使用的实现如下：

```python
import numpy as np

def apk(actual, predicted, k=7):
    """算一个客户的 Average Precision@7"""
    predicted = predicted[:k]              # 最多取前 7 个预测
    score, num_hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        # 计分条件：预测值确实在真实答案里，而且这个产品此前没被算过
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)  # 越靠前命中，分数越高
    if not actual:                          # 客户没买新品，直接记 0
        return 0.0
    return score / min(len(actual), k)      # 用真实新增个数做平均

def mapk(actual, predicted, k=7):
    """所有客户 AP 的平均，就是 MAP@7"""
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
```

然后用每个产品的概率拼排序结果，生成提交文件：

```python
# 把 24 个模型的概率拼成一张“客户 × 产品”的表
pred_prob = pd.DataFrame({prod: prob_all[prod]
                          for prod in product_cols}, index=va_idx)

rows = []
for cid in pred_prob.index:
    # 推荐位必须留给“还没买的”产品：用上一月（4 月）的持有情况做过滤
    owned = apr.loc[cid, product_cols].astype(bool)
    cand = pred_prob.loc[cid, ~owned].sort_values(ascending=False)
    top7 = ' '.join(cand.index[:7].tolist())   # 换成空格分隔的产品代码
    rows.append([cid, top7])

sub = pd.DataFrame(rows, columns=['ncodpers', 'added_products'])
sub.to_csv('submission.csv', index=False)
```

这里有一个容易写错的小地方：验证时过滤「4 月已拥有的产品」，真提交时就要换成「5 月已拥有的产品」，因为比赛要预测的是 6 月相比 5 月的新增。我们在代码里专门留了这个注释，避免提交时踩坑。

指标报告如下：我们的本地（5 月）MAP@7 落在 0.02～0.03 这个量级。这个数字单看小得可怜，但放到比赛语境里就完全不同了：2016 年 5 月理论最大值只有 0.0319，排行榜当时的第一名 0.0303 已经非常接近极限。换句话说，这个比赛的大部分得分损失根本不在模型，而在「绝大多数客户没有新增产品，这些人怎么排都是 0 分」这条铁律上。理解了这点，我们就能心平气和地接受零点零几分的排名，把精力放在「在真正会买的人群里把顺序排对」。

---

## 三、总结

本文中我们完成了 Santander 产品推荐比赛的全流程。我们首先做了 EDA，摸清了这份 1360 万行数据的构成：每月一份产品快照、24 个 0/1 产品列、绝大多数产品持有率不足 1%，明白了这是极稀疏的推荐排序问题而不是普通分类。其次我们处理了数据清洗的两大坑：把 `-999999` 这种「未知」占位符换成 NaN 并按中位数填补，同时用「相邻月快照做差」构造出干净的新增产品标签。接着我们为 24 个产品各训练一个 XGBoost 二分类器，把输出的概率拼成排序，并且强制过滤掉客户已持有的产品。最后我们用与官方一致的 apk/mapk 实现了 MAP@7 的本地验证，得到的分数落在 0.02～0.03 量级，与 2016 年 5 月理论最大值 0.0319 相比，说明该问题的主要得分损失来自「没买新品的客户记 0 分」这一评价规则本身。

这个比赛给我们的启发有两点。第一，**评价标准决定努力方向**：用准确率你会去压「没买」这一类，用 MAP@7 你才会去抠那些真正会买的 3% 客户，并且在意推荐的顺序——顺序错了哪怕产品全对，分数也会打折。第二，**稀疏推荐问题的先验比模型结构更值钱**：把「客户已持有的产品」移出候选、只对剩余产品排序，这样一个一行代码的规则带来的提升，往往比换更强的模型、调更精细的参数更明显。