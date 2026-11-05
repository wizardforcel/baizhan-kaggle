# 国际象棋 ELO 分预测（finding-elo）

## 一、比赛简介

### 1.1 比赛背景

ELO 分是国际象棋里用来衡量棋手水平的一个数字。两个人下棋，赢了加分、输了扣分，分越高说明水平越强。ELO 的好处是它能直接换算成胜率：两个棋手分差越大，高分一方的预期胜率越高。所以如果能从一盘棋的过程反推出两位棋手的 ELO，就等于知道了这盘棋大概是什么水平的对局。

这个比赛给出一批国际象棋对局记录，要求预测每盘棋里**白方和黑方各自的 ELO 分**。数据来自 `chess_data.pkl`，每一行是一盘棋，包含：

- `white`、`black`：白方和黑方的 ELO 分（原始数据里可能是字符串）；
- `result`：对局结果，1 表示白胜、0 表示黑胜、0.5 表示和棋；
- `evals`：引擎评估序列，每个元素是某一步之后的局面评分，正数通常表示白方占优；
- `moves`：走法序列，每一步是一个字符串，比如 `e4`、`Nf3`、`Qxd5`；
- `opening`、`variation`：开局名称和变例。

难点在于：ELO 不是直接出现在数据里的，模型只能从「棋手怎么下」「局面怎么变化」「用了什么开局」这些间接线索去推断。高手下棋失误少、评估波动小；低手经常出现大起大落。模型要学的就是这种「棋局质量」和「棋手水平」之间的关系。

### 1.2 比赛目标

训练数据包含完整的 `white` 和 `black` 两列，测试数据只给对局过程，要我们补上两个 ELO 值。材料里的代码把前 25,000 盘当作训练集，后 25,000 盘当作测试集，说明总共有 50,000 盘左右的对局。

提交文件的列是：

- `Event`：对局编号；
- `WhiteElo`：预测的白方 ELO；
- `BlackElo`：预测的黑方 ELO。

注意这是**两个回归目标**，不是二分类。白方和黑方的分数通常接近（同一场比赛的对手水平不会差太远），但也会有让子、低分棋手挑战高手等情况，所以两个目标要分别建模。

### 1.3 评价标准

材料里用 `np.sqrt(mean_squared_error(...))` 计算误差，所以指标是 **RMSE（均方根误差）**：

```
RMSE = sqrt( mean( (预测 ELO - 真实 ELO)^2 ) )
```

RMSE 越低越好，单位和 ELO 一样是「分」。它对大误差惩罚很重，所以模型不能只在大多数普通对局上准，还要避免在少数异常对局上给出离谱的预测。比如真实 2000 分预测成 3000 分，误差 1000，平方后就是 100 万，会严重拉高整体 RMSE。

---

## 二、解题过程

我们的主线是：先修数据里几处明显错误的标签，再把每一盘棋的评估序列、走法、开局都变成数值特征，最后用 XGBoost 分别回归白方和黑方的 ELO。

### 2.1 数据探索（EDA）

我们首先把 pickle 文件读进来，看看列和规模：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取预处理好的对局数据
data1 = pd.read_pickle('chess_data.pkl')

# 看规模和列名
print(data1.shape)
print(data1.columns.tolist())

# 看前几行
display(data1.head())
```

从材料里的用法看，`white` 和 `black` 读进来时是字符串，`evals` 和 `moves` 是列表。列表类型的列不能直接喂给模型，必须先做统计汇总。

接着我们修正几处明显错误的 `result`。材料里定位到三行，分别改成 0.5、0、1：

```python
# 修正三条记录的结果标签
# 这些是数据录入错误：和棋被记成了胜负，或者胜负记反了
data1.loc[13683, 'result'] = 0.5
data1.loc[15846, 'result'] = 0
data1.loc[33094, 'result'] = 1
```

然后我们把 ELO 列转成浮点数，并构造两个最直观的特征：双方平均分和分差：

```python
# 原始 ELO 是字符串，先转成 float
data1['white'] = data1['white'].astype(float)
data1['black'] = data1['black'].astype(float)

# 平均分：代表这盘棋的整体水平
data1['meanElo'] = (data1['white'] + data1['black']) / 2

# 分差：代表双方实力差距，正数表示白方更强
data1['diffElo'] = data1['white'] - data1['black']
```

这两个特征很重要，因为「两个 2500 分棋手对局」和「两个 1200 分棋手对局」的平均分差很多，但分差可能都是 0。模型需要同时知道「水平高低」和「差距大小」。

再看对局长度和评估序列。`moves` 的长度减 1 就是回合数（走法按半步记录）；`evals` 是逐步评分，直接看原始值意义不大，我们更关心它的波动：

```python
# 回合数：走法数量减 1
data1['lengths'] = data1['moves'].apply(lambda m: len(m) - 1)

# 打印长度分布，看看有没有异常短或异常长的对局
print(data1['lengths'].describe())

# 画直方图
data1['lengths'].hist(bins=50)
plt.title('Game length')
plt.show()
```

评估值还要做截断。引擎评分偶尔会出现 ±10000 这样的极端值（表示「必胜/必败」），如果不处理，均值和标准差都会被少数几步棋带偏：

```python
# 把评估值限制在 [-400, 400]
# 超出这个范围基本等同于「已经赢定/输定」，再大的数值没有额外信息
data1['evals'] = data1['evals'].apply(
    lambda seq: pd.Series(seq).clip(-400, 400).tolist()
)
```

### 2.2 数据清洗

这一步主要处理两类问题：异常对局和空序列。

第一，删除两盘明显异常的对局（材料里给的编号是 14854 和 23781），并只保留长度在 20 到 300 个回合之间的对局。太短的对局信息不足，太长的对局可能是数据错误或特殊赛制：

```python
# 删除两盘异常对局
train = train.drop(labels=[14854, 23781])
train.reset_index(inplace=True, drop=True)

# 只保留 20~300 回合的对局
keep = train[(train['lengths'] >= 20) & (train['lengths'] <= 300)].index
train = train.loc[keep].reset_index(drop=True)
```

第二，处理空的评估序列。有些对局没有引擎评分，`evals` 是空列表。对这类样本，我们没法算均值、标准差，只能填一个「中性值」，并尽量不让它影响整体：

```python
def safe_stats(seq):
    """对评估序列算统计量；空序列返回 None，后面统一填中位数"""
    if len(seq) == 0:
        return pd.Series({'std': np.nan, 'median': np.nan,
                          'max': np.nan, 'min': np.nan})
    s = pd.Series(seq)
    return pd.Series({'std': s.std(), 'median': s.median(),
                      'max': s.max(), 'min': s.min()})

# 展开成四列
stats = data1['evals'].apply(safe_stats)

# 空值用该列的中位数填补
for col in ['std', 'median', 'max', 'min']:
    data1['eval_' + col] = stats[col].fillna(stats[col].median())
```

这里用中位数而不是 0，是因为不同对局的评估尺度不一样，0 不一定代表「中性」；用全体中位数更稳妥。

### 2.3 模型训练与超参数调优

特征分三组：评估序列统计、走法特征、开局编码。

第一组是评估序列的统计量。除了上面的标准差、中位数、最大最小，我们还看了「评估值的逐歩变化」——高手对局评估变化平缓，低手对局经常大起大落：

```python
def eval_changes(seq, cutoff=10000):
    """计算相邻两步评估值的变化量"""
    changes = []
    for j in range(1, len(seq)):
        # 只统计没有出现「必胜/必败」极端值的步
        if -cutoff < seq[j] < cutoff and -cutoff < seq[j-1] < cutoff:
            changes.append(seq[j] - seq[j-1])
    return changes

# 把每盘棋的评估变化序列存起来，后面可以算均值、标准差
data1['eval_changes'] = data1['evals'].apply(eval_changes)
```

第二组是走法特征：第一次走王后（Q）、第一次走王（K）、第一次将军（+）分别发生在第几步。高手往往更早出动重子、更早组织进攻：

```python
def first_move(moves, piece, parity):
    """返回某个棋子在指定奇偶步（白/黑）中第一次出现的步数"""
    for j, mv in enumerate(moves):
        # parity=0 表示白方走，1 表示黑方走
        if j % 2 == parity and piece in mv:
            return j
    # 整盘都没出现，就返回总步数
    return len(moves) - 1

# 白方第一次走王后、走王、将军的步数
data1['queenw'] = data1['moves'].apply(lambda m: first_move(m, 'Q', 0))
data1['kingw'] = data1['moves'].apply(lambda m: first_move(m, 'K', 0))
data1['checkw'] = data1['moves'].apply(lambda m: first_move(m, '+', 0))

# 黑方同理
data1['queenb'] = data1['moves'].apply(lambda m: first_move(m, 'Q', 1))
data1['kingb'] = data1['moves'].apply(lambda m: first_move(m, 'K', 1))
data1['checkb'] = data1['moves'].apply(lambda m: first_move(m, '+', 1))
```

第三组是开局编码。`opening` 和 `variation` 是文本，取值很多，我们先用 `LabelEncoder` 转成整数，再做 one-hot：

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
ohe = OneHotEncoder(sparse_output=False)

# 开局名称和变例分别编码
open_enc = le.fit_transform(data1['opening'])
var_enc = le.fit_transform(data1['variation'])

# one-hot 展开成稀疏的 0/1 矩阵
open_ohe = ohe.fit_transform(open_enc.reshape(-1, 1))
var_ohe = ohe.fit_transform(var_enc.reshape(-1, 1))

# 拼到特征表里
openings = np.hstack([open_ohe, var_ohe])
```

开局特征的作用是告诉模型「这盘棋属于哪一类体系」。比如西西里防御的对局往往比伦敦体系更激烈，评估波动更大，模型可以借此调整对棋手水平的判断。

把所有特征拼好后，我们用 `RobustScaler` 做缩放。它用中位数和四分位距而不是均值和标准差，对异常值更稳健：

```python
from sklearn.preprocessing import RobustScaler

RS = RobustScaler()

# 特征矩阵：去掉目标列、原始列表列和文本列
drop_cols = ['white', 'black', 'meanElo', 'diffElo', 'result',
             'evals', 'opening', 'variation', 'moves', 'lengths']
X = data1.drop(columns=drop_cols, errors='ignore')

# 拼上开局编码
X = np.hstack([X.values, openings])

# 缩放
X = RS.fit_transform(X)
```

模型用 XGBoost。参数是 `max_depth=4`、`n_estimators=250`、`learning_rate=0.05`，这是一组比较保守的设置：树不深、学习率不高、树的数量适中，不容易过拟合：

```python
from xgboost import XGBRegressor

# 白方 ELO 模型
xgbw = XGBRegressor(
    max_depth=4,
    n_estimators=250,
    learning_rate=0.05,
    random_state=42,
)

# 黑方 ELO 模型
xgbb = XGBRegressor(
    max_depth=4,
    n_estimators=250,
    learning_rate=0.05,
    random_state=42,
)
```

我们试过两种建模方式：

1. **直接回归**：分别训练白方模型和黑方模型，各自预测；
2. **均值 + 差值**：先预测平均 ELO 和分差，再用 `均值 + 差值/2` 还原白方、`均值 - 差值/2` 还原黑方。

材料里第二种方式写在注释里，没有作为最终方案。我们最终选直接回归，因为它更简单，两个目标各自优化各自的 RMSE，不用担心中间变量的误差传播。

### 2.4 模型测试与指标报告

我们按时间顺序切分：前 20,000 盘训练、20,000 之后验证，最后用全部训练数据重训并预测后 25,000 盘。先看白方模型的验证误差：

```python
from sklearn.metrics import mean_squared_error

# 前 20000 盘训练，20000 之后验证
x_train = X[:20000]
x_test = X[20000:]
y_train = y[:20000]
y_test = y[20000:]

# 训练白方模型
xgb.fit(x_train, y_train['white'])
xgb_preds = xgb.predict(x_test)

# 计算 RMSE
rmse_white = np.sqrt(mean_squared_error(xgb_preds, y_test['white']))
print('白方 ELO 的 RMSE：', rmse_white)
```

然后我们看误差的分布，而不只是看一个总数。把所有 `预测 - 真实` 的差值画成直方图，如果分布以 0 为中心、左右大致对称，说明模型没有系统性高估或低估：

```python
# 逐样本误差
misses = xgb_preds - y_test['white'].values

# 直方图：看误差是否对称、有没有长尾
pd.Series(misses).hist(bins=50)
plt.title('White ELO prediction error')
plt.show()

# 误差的均值接近 0 说明没有系统偏差
print('平均误差：', misses.mean())
print('误差标准差：', misses.std())
```

接着用全部训练数据重训两个模型，对测试集预测：

```python
# 用全部训练数据重新训练白方和黑方模型
xgbw.fit(X, y['white'])
xgbb.fit(X, y['black'])

# 对测试集（第 25000 行之后）预测
white_preds = xgbw.predict(X2)
black_preds = xgbb.predict(X2)
```

最后组装提交文件。`Event` 是对局编号，从 25001 开始依次递增；两列 ELO 分别是两个模型的输出：

```python
# 测试集对局编号：25001 到 50000
events = list(range(25001, 50001))

# 组装提交表
preds = pd.DataFrame({
    'Event': events,
    'WhiteElo': white_preds,
    'BlackElo': black_preds,
})

# 保存
preds.to_csv('first_preds.csv', index=False)
```

材料里没有保留具体的 RMSE 数值，所以这篇 writeup 不引用分数。我们报告的是完整的验证协议：修正三处结果标签、删除两盘异常对局、只保留 20 到 300 回合的对局、截断评估值到 ±400、按时间顺序切分、分别回归白方和黑方、用 RMSE 评估并检查误差分布。

---

## 三、总结

本文中我们完成了国际象棋对局的 ELO 分预测。我们首先读取 `chess_data.pkl`，修正了三处错误的 `result` 标签，把 ELO 字符串转成数值，并构造了平均分和分差两个基础特征。其次我们清理了异常对局：删掉两盘问题棋局，只保留 20 到 300 回合的记录，把引擎评估截断在 ±400，并对空的评估序列用中位数填补。接着我们构造了三组特征：评估序列的标准差/中位数/最大最小、白黑双方第一次走王后/走王/将军的步数、以及开局和变例的 one-hot 编码，再用 `RobustScaler` 缩放后分别训练白方和黑方的 XGBoost 回归模型。最后我们按时间顺序验证、检查误差分布，并用全部训练数据重训后输出 `Event`、`WhiteElo`、`BlackElo` 三列提交。

这个比赛让我们看到「领域知识怎么变成特征」：ELO 本身没出现，但棋手的水平会体现在评估波动、重子出动时机、开局选择上。把这些间接线索量化，模型就能从棋局过程里反推出棋手水平。另一条经验是切分方式要匹配数据性质——对局数据有时间顺序，按时间切分比随机切分更能反映模型在「未来对局」上的真实表现。
