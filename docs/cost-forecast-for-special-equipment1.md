# 特种设备价格预测（Cost Forecast for Special Equipment）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的任务是**给二手特种设备估价**。

所谓「特种设备」，指的是工程和运输领域那些专门用途的大机器：吊车、挖掘机、装载机、钻机、冷藏车、牵引车等等。这类设备单价高、型号杂、配置差异极大，同一款车换个吊臂长度或者多几千个摩托小时，价格可能差出一大截。买家在二手市场上最头疼的问题就是：**这台机器到底值多少钱？**

比赛提供的数据来自俄语的设备交易信息（列名全是俄文，比如 `Марка` 是品牌、`Пробег` 是里程、`Моточасы` 是发动机工作小时数）。每条记录就是一条挂牌信息，里面既有结构化的参数（年份、功率、载重、臂长），也有自由文本（标题描述、原始链接）。

难点有三个：

1. **数值列是「脏」的**。同一个字段里混着 `12 500`、`12,5`、`1\xa0500`（不换行空格）这类写法，还有「暂无」「不详」这种非数字，必须先抠出数字才能用。
2. **类别列极多且稀疏**。品牌、型号、城市、变速箱、注册状态……加起来二十多列，很多取值只出现几次。
3. **样本量不大、目标跨度大**。设备从几万美元到上百万美元都有，用绝对误差去优化，模型会被贵设备牵着走。

### 1.2 比赛目标

训练集 `train.csv` 每一行是一台设备，包含：

- `id`：记录编号；
- `Ссылка`：原始挂牌链接；
- `Описание`：文字描述；
- 一批类别字段：`Категория`（类别）、`Тип техники`（设备类型）、`Марка`（品牌）、`Модель`（型号）、`Город`（城市）、`Двигатель`（发动机）、`Коробка передач`（变速箱）等；
- 一批数值字段：`Год`（年份）、`Дата`（日期）、`Мощность`（功率）、`Пробег`（里程）、`Масса`（质量）、`Моточасы`（工作小时）、`Грузоподъемность`（载重）、`Длина стрелы`（臂长）等；
- 目标列 `Цена в долларах`：这台设备的成交价（美元）。

测试集没有目标列，我们要给每条测试记录预测一个美元价格，提交文件包含 `id` 和 `USD_price` 两列。

### 1.3 评价标准

材料里的训练代码把 **MAPE**（Mean Absolute Percentage Error，平均绝对百分比误差）当作损失函数，进度条上也一直在打印 `MAPE=xx%`。MAPE 的算法是：对每条样本算「预测值和真实值差多少，再除以真实值」，取绝对值，最后对所有样本求平均。写成公式就是 `|真实 - 预测| / 真实`。

它衡量的是**相对误差**：一台 10 万美元的机器猜错 1 万，和一台 100 万美元的机器猜错 10 万，在 MAPE 眼里「错得一样多」，都是 10%。所以这个指标**越低越好**，而且它天然地让模型对所有价位的设备一视同仁——这正是我们想要的。

---

## 二、解题过程

这份材料给出的方案叫 **RowNet**，一句话概括就是：**不孤立地看一条记录，而是让模型拿它去和训练集里所有设备逐一比较，找出最相似的几台，再参考它们的价格来估价。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步我们把训练集和测试集读进来，先把没用的列扔掉，并把目标列单独拎出来：

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 读入训练集和测试集
train_df = pd.read_csv(
    "/kaggle/input/competitions/cost-forecast-for-special-equipment1/train.csv")
test_df = pd.read_csv(
    "/kaggle/input/competitions/cost-forecast-for-special-equipment1/test.csv")

# 提交文件要用的 id，先存下来，因为后面要把 id 列删掉
test_ids = test_df["id"]

# 链接和文字描述不参与建模：链接是纯噪声，描述是长文本，
# 这里没有做文本建模，所以直接丢掉，省得占内存
drop_cols = ["Ссылка", "Описание"]
train_df.drop(drop_cols, axis=1, inplace=True)
test_df.drop(drop_cols + ["id"], axis=1, inplace=True)

# 目标列：美元价格。训练时单独拿出来，不作为输入特征
y = train_df["Цена в долларах"]
train_features = train_df.drop(columns=["Цена в долларах"])
```

看完列名我们立刻发现一个重要事实：**特征可以干净地分成「类别」和「数值」两组**。类别列里有品牌、型号、城市这种字符串，数值列里则是年份、功率、里程这些本该是数字的字段。但数值列的实际内容并不干净——它们是从网页上抓下来的，常带单位和千分位。所以下一步我们把每一列手工归档：

```python
# 类别列：取值是字符串，需要编码成整数
cat_cols = [
    "Категория", "Тип техники", "Марка", "Модель", "Город",
    "Двигатель", "Коробка передач", "Руль", "VIN", "ПТС",
    "Розыск", "Ограничения", "Дтп и ремонт", "Юр. лицо",
    "Учет", "Топливо", "Регистрация", "Состояние", "Привод",
    "Спальник", "Ходовая часть", "Страна производства",
    "Категория ТС", "Тип кабины", "Особые отметки",
    "Рефрижератор", "Бренд ХОУ", "Проверки",
]

# 数值列：理论上都是数字，但实际带着单位和空格
num_cols = [
    "Год", "Дата", "Мощность", "Пробег", "Масса", "Моточасы",
    "Количество осей", "Высота подъема", "Грузоподъемность",
    "Длина кузова", "Объем ковша", "Грузоподъемность стрелы",
    "Длина стрелы", "Объем кузова", "Рабочая ширина",
    "Высота седла", "Высота вышки", "Глубина бурения",
    "Диаметр бурения", "Количество мест", "Колёсная формула",
    "Период владения", "Залог", "Записи пробега",
]
```

把列分好组之后，我们注意到三列特别稀疏——`Высота вышки`（塔架高度）、`Глубина бурения`（钻孔深度）、`Диаметр бурения`（钻孔直径），只有钻机类设备才有值，别的设备全是空。这三列**不能用中位数填充**，因为「没有这个参数」本身就是信息（说明它不是钻机）。所以后面我们会把它们单独填成 `-1`。

再往下，我们真正关心的是**数值列到底有多脏**。我们随机抽几条 `Пробег` 看，发现形如 `12 500`、`8,5`、`1 200` 的写法混在一起。这意味着我们不能直接 `astype(float)`，必须先把数字抠出来。

### 2.2 数据清洗

清洗分四步。第一步是**把所有文本形状的数字变成真数字**：去掉不换行空格、去掉普通空格、把小数点逗号换成点，最后用正则抓出第一个数字。

```python
for col in num_cols:
    if col not in combined.columns:
        continue
    # 「Дата」「Год」等列有各自的解析方式，跳过统一处理
    if col in skip_cols:
        continue

    extracted = (
        combined[col]
        .astype(str)
        .str.replace("\xa0", "", regex=False)   # 去掉不换行空格（网页里常见）
        .str.replace(" ", "", regex=False)      # 去掉普通空格
        .str.replace(",", ".", regex=False)     # 俄语里逗号当小数点
        .str.extract(r"([-+]?\d*\.?\d+)")[0])   # 只保留第一个数字

    # errors="coerce"：实在不是数字的（比如「нет данных」）变成 NaN
    combined[col] = pd.to_numeric(extracted, errors="coerce")
```

这里有个细节值得强调：我们把训练集和测试集**先拼在一起**再清洗（`combined`）。原因是很多处理步骤要用到统计量，比如中位数填充、标签编码。如果只对着训练集算，测试集里出现的新品牌就没法编码；拼在一起做，两边的编码规则完全一致，也不会出现维度对不上的问题。

第二步是**把几个伪装成字符串的字段拆成数字**。`Дата`（日期）本身没有预测价值，但「距离数据集最新日期过了多少天」有——越新的挂牌越接近当前行情。`Год` 里混着 `2019 г.` 这种写法，要提取四位年份。`Колёсная формула`（轴数配置）写作 `6x4`，我们把它变成 `6*10+4` 这样一个能反映「几轴驱动」的数字：

```python
# 日期：统一解析成 datetime，再换算成"距最新日期多少天"
parsed_dates = pd.to_datetime(combined["Дата"], format="%d.%m.%Y", errors="coerce")
max_date = parsed_dates.max()                       # 数据集里最新的一天
combined["Дата"] = (max_date - parsed_dates).dt.days  # 越新数字越小

# 年份：从 "2019 г." 里抠出 4 位数字
combined["Год"] = pd.to_numeric(
    combined["Год"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce")

# 轴数配置：把 "6x4" 拆成 6 和 4，编码成 64，
# 这样个位数表示驱动轴数，十位数表示总轴数，语义不会丢
wheel = combined["Колёсная формула"].astype(str).str.extract(r"(\d+)x(\d+)")
combined["Колёсная формула"] = (
    pd.to_numeric(wheel[0], errors="coerce") * 10
    + pd.to_numeric(wheel[1], errors="coerce"))
```

第三步是**处理缺失值**，而且我们按「这一列缺了意味着什么」分情况处理：

```python
# 情况一：稀疏的钻机参数，缺失本身就是信息，统一填 -1 当"没有"
for col in ["Высота вышки", "Глубина бурения", "Диаметр бурения"]:
    combined[col] = combined[col].fillna(-1)

# 情况二：类别列，缺失当成一个独立的类别 "-1"
#（不能填众数，因为"没填"和"填了最常见值"是两回事）
for col in cat_cols:
    if col in combined.columns:
        combined[col] = combined[col].fillna("-1").astype(str)

# 情况三：普通数值列，用中位数填充
# 中位数比均值抗极端值——一台设备标价一亿美元这种脏数据不会把填充值带偏
for col in num_cols:
    if col not in combined.columns or col in special_minus_one_cols:
        continue
    combined[col] = combined[col].fillna(combined[col].median())
```

第四步是**把类别列编码成整数**。神经网络只会做浮点运算，字符串它看不懂。这里用 `LabelEncoder` 给每个取值编一个序号：

```python
label_encoders = {}
for col in cat_cols:
    if col not in combined.columns:
        continue
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])
    label_encoders[col] = le      # 存起来，以后要还原或处理新数据时用

# 清洗完再按原来的行数切开：前 len(train_df) 行是训练集，剩下是测试集
X = combined.iloc[:len(train_df)]
X_test = combined.iloc[len(train_df):]
```

到这里，原始表格就变成了一张**全数字表**：每一列都是浮点数，没有空值。这份数据可以直接喂给模型了。

### 2.3 模型训练与超参数调优

这一节是全文的核心。传统做法是把一行特征塞进一个网络（或者 XGBoost）直接回归价格，但那种做法有个天然缺陷：**模型必须把「什么价位的设备长什么样」压缩进固定的参数里**，遇到罕见型号就抓瞎。

RowNet 换了个思路，它把训练集本身当成一本「价格词典」：拿到一台新设备，先翻词典找相似的，再看那些相似的卖多少钱。具体分三步。

**第一步：定义「两台设备有多像」。** 相似度必须同时考虑类别和数值。类别用「是否相等」，数值用三种不同的衰减函数：

```python
def sim_x(q_cat, q_num, X_cat, X_num):
    """计算查询行 q 与训练集每一行的相似度特征
       q_cat: 查询行的类别向量；X_cat: 全训练集的类别矩阵"""
    # 类别相似度：相等记 1，不等记 0，逐列比较
    cat_sim = (X_cat == q_cat.unsqueeze(0)).float()

    # 数值相似度：先算绝对差
    d = torch.abs(X_num - q_num.unsqueeze(0))
    scale = torch.abs(q_num.unsqueeze(0)) + 1e-6   # 用查询值当尺度，避免除零

    # 三种互补的相似度曲线：
    #  sim1 平滑线性衰减，适合"差一点也算接近"
    #  sim2 指数衰减，对"差很多"惩罚更狠
    #  sim3 截断线性，超过尺度直接归零，制造稀疏性
    sim1 = 1.0 / (1.0 + d)
    sim2 = torch.exp(-d / scale)
    sim3 = torch.clamp(1.0 - d / scale, min=0.0)

    # 拼成一个大矩阵：行是训练样本，列是相似度特征
    return torch.cat([cat_sim, sim1, sim2, sim3], dim=1)
```

为什么要三种曲线？因为不同字段的「像」不一样：年份差 1 年和差 10 年是完全不同的概念；功率差 10% 也许无所谓，差一倍就是两种机器。三条曲线给了模型三种「宽容度」去挑，具体用哪条、权重多少，交给后面的可学习参数决定。

**第二步：把相似度变成「投票权重」。** 有了相似度矩阵，模型给每个训练样本打一个分数，再 `softmax` 归一化成权重，最后用这些权重对训练集的价格做加权平均——这就是预测值。这一层我们叫 **L1**，它只看特征相似度：

```python
# M1：查询行与全训练集的相似度矩阵，形状 (训练样本数, 特征数)
M1 = sim_x(X_cat[i], X_num[i], X_cat, X_num)

# w1 是可学习的特征权重，矩阵乘法给每个训练样本打一个相似度总分
s1 = M1 @ model.w1
# 关键一步：把自己排除掉！否则模型只要把权重全压在"自己"身上，
# 就能直接抄到答案，训练会立刻"完美"但毫无泛化能力
s1[i] = -1e9

a1 = torch.softmax(s1, dim=0)          # 归一化成"该参考哪几台"的权重
y1 = (a1 * y).sum()                    # 加权平均，得到第一版估价
```

**第三步：用第一版估价再找一轮。** 这是 RowNet 最巧妙的地方。拿到 `y1` 之后，我们再算一次相似度，但这次**把「价格像不像」也当成一维特征**：

```python
def sim_xy(q_cat, q_num, q_y, X_cat, X_num, Y):
    """在特征相似度的基础上，再加两列"价格相似度" """
    Mx = sim_x(q_cat, q_num, X_cat, X_num)     # 先用原特征算一遍

    # 查询行估计价 q_y 与训练集每台设备真实价的差距
    dy = torch.abs(Y - q_y)
    scale = torch.abs(q_y) + 1e-6

    # 两条价格相似度曲线
    y_sim1 = 1.0 / (1.0 + dy / scale)
    y_sim2 = torch.exp(-dy / scale)

    return torch.cat([Mx, y_sim1.unsqueeze(1), y_sim2.unsqueeze(1)], dim=1)
```

这一层（**L2**）的好处很直观：如果第一轮估出「大概 8 万美元」，第二轮就会自动偏向那些「特征相近、价格也在 8 万附近」的设备，相当于做了一次**基于价格的二次检索**。两轮下来，估价既参考了「长得像的」，也参考了「卖得差不多的」。

**第四步：多专家 + 残差修正。** 光有一次加权平均还不够稳，所以我们让模型同时维护 **4 个「专家头」**（`n_heads=4`），每个头有自己的权重向量，各自给出一个加权平均价。然后再用一个「门控网络」决定该听谁的，同时每个头再输出一个残差去修正系统偏差：

```python
class ThreeLayerMoE(nn.Module):
    """MoE = Mixture of Experts，多个专家各自预测再融合"""
    def __init__(self, d1, d2, hidden=128, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        # 第一层相似度特征的权重（所有专家共用）
        self.w1 = nn.Parameter(torch.ones(d1) * 0.1)
        # 第二层：每个专家一套权重，形状 (专家数, 特征数)
        self.w2 = nn.Parameter(torch.randn(n_heads, d2) * 0.1)

        # 门控网络：输入融合后的特征，输出每个专家的得分
        self.gating = nn.Sequential(
            nn.Linear(d2, hidden), nn.ReLU(), nn.Linear(hidden, n_heads))

        # 每个专家配一个小 MLP 做残差修正
        self.residual_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(d2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            for _ in range(n_heads)])
```

融合时我们用了两条并行的权重来源，再按比例 `g=0.5` 混合：一条是门控网络输出的 `g_weights`（学会「什么情况下信哪个专家」），另一条是 `softmin(-|残差|)`——**哪个专家估得偏离平均值越少，就越信它**。这种做法不需要额外训练就能起到「保守投票」的作用，对回归任务特别稳。

**第五步：损失函数。** 除了主损失 MAPE，我们还加了三个正则项，每个都对应一个具体的问题：

```python
mape = torch.abs((true - pred) / (true + 1e-6))          # 主损失：相对误差
loss_reg = (residual / (true + 1e-6)).pow(2)             # 残差不许太大

# 门控的熵：熵越低说明"选专家"越果断，我们希望它别和稀泥
entropy = -(g_weights * torch.log(g_weights + 1e-8)).sum()

# 专家多样性：把 4 个权重向量两两算余弦相似度，
# 和单位矩阵差得越远说明专家越"雷同"，惩罚它
W_norm = model.w2 / (model.w2.norm(dim=1, keepdim=True) + 1e-8)
div_loss = ((W_norm @ W_norm.T) - torch.eye(model.n_heads)).pow(2).mean()

loss = (mape
        + 0.05 * loss_reg        # lambda_res
        - 0.05 * entropy         # lambda_entropy：注意是减号，鼓励低熵
        + 0.20 * div_loss)       # lambda_div
```

**关于训练方式和超参数。** 这里有一个和常规深度学习很不一样的地方：**batch size 只能是 1**。因为 attention 是在「整张训练表」上做的，每条样本都要遍历全部训练行，没法并行。所以代码里只能一层 `for i in range(n)` 逐条训练。为了让这种昂贵的训练收敛，我们用了几个配套设置：学习率 `1e-2`（比较大，配 Adam）、余弦退火调度（从 1e-2 平滑降到 1e-6）、混合精度（`autocast` + `GradScaler`，省显存又提速）。总共训练 `epochs=10`。

超参数方面，我们注意到一个有意思的细节：`lambda_res` 和 `lambda_entropy` 在函数默认值和实际调用里都是 `0.05`，唯独 **`lambda_div` 从默认的 `0.05` 被提到了 `0.20`**——是三项正则里被加强得最多的一个。这说明作者最担心的是「4 个专家学成一模一样」：一旦权重向量雷同，所谓集成就退化成单个模型，白白多花四倍算力。把多样性项的权重调到其他项的四倍，就是在逼着每个专家去看不同的相似度特征。

### 2.4 模型测试与指标报告

预测阶段和训练阶段的前向过程一模一样，唯一的区别是**不再排除自己**——因为测试样本本来就不在训练集里，没有「自己」可抄。我们把每条测试记录依次和训练集比对、算 L1、算 L2、融合专家，得到最终价格：

```python
@torch.no_grad()          # 预测不需要梯度，关掉能省一大半显存
def predict(model, train_df, train_y, test_df, cat_cols, num_cols, g=0.5):
    model.eval()          # 切到推理模式（关掉 dropout 等训练专用行为）

    # 训练集整体搬到 GPU 上，因为每条测试样本都要和它比一遍
    X_cat_tr = torch.tensor(train_df[cat_cols].values, dtype=torch.float32, device="cuda")
    X_num_tr = torch.tensor(train_df[num_cols].values, dtype=torch.float32, device="cuda")
    y_tr = torch.tensor(np.asarray(train_y), dtype=torch.float32, device="cuda")

    preds = []
    for i in range(len(test_df)):
        # L1：纯特征检索，得到第一版估价
        a1 = torch.softmax(sim_x(X_cat_te[i], X_num_te[i], X_cat_tr, X_num_tr) @ model.w1, dim=0)
        y1 = (a1 * y_tr).sum()

        # L2：带着 y1 再检索一次，同时参考"长得像"和"卖得像"
        M2 = sim_xy(X_cat_te[i], X_num_te[i], y1, X_cat_tr, X_num_tr, y_tr)
        a2 = torch.softmax(M2 @ model.w2.T, dim=0)

        # 每个专家的加权平均价 + 每个专家看到的特征
        y_heads = (a2 * y_tr.unsqueeze(1)).sum(dim=0)
        feats_heads = (a2.unsqueeze(2) * M2.unsqueeze(1)).sum(dim=0)

        # 门控 + 残差 + softmin 融合，和训练时完全一致
        g_weights = torch.softmax(model.gating(feats_heads.mean(dim=0)), dim=0)
        residual_heads = torch.stack([
            model.residual_mlps[h](feats_heads[h]) for h in range(model.n_heads)]).squeeze()
        soft_weights = torch.softmax(-torch.abs(residual_heads), dim=0)

        final_weights = g * g_weights + (1 - g) * soft_weights
        final_weights = final_weights / (final_weights.sum() + 1e-8)

        preds.append(((final_weights * y_heads).sum()
                      + (final_weights * residual_heads).sum()).item())

    return np.array(preds)
```

训练日志里，每个 epoch 结束都会打印一次全训练集的 MAPE，格式是 `Epoch k: MAPE=xx.xxxx%`。我们主要盯两个信号：**一是 MAPE 是否还在下降**（不降了就说明 10 轮够了），**二是四个专家的预测是否开始分化**（如果始终一模一样，说明多样性正则没起作用，需要调大 `lambda_div`）。

最后按比赛要求组装提交文件：

```python
test_preds = predict(t_model, X, y, X_test, cat_cols, num_cols, g=0.5)

submission = pd.DataFrame({
    'id': test_ids,          # 之前存下来的测试集 id
    'USD_price': test_preds  # 预测的美元价格
})
submission.to_csv("three_layer_submission.csv", index=False)
```

这里有一处必须小心：`test_preds` 是模型直接输出的实数，而残差修正项并没有加非负约束，所以理论上可能出现负数或异常大的值。价格不能是负的，而且一旦出现这种预测，MAPE 里 `|真实 - 预测| / 真实` 会直接爆掉。实际提交前值得加一道保险：把预测值截断到训练集价格的合理区间内。

从结果来看，RowNet 的价值不在于「网络比树模型更深」，而在于它换了一种归纳方式：**XGBoost 学的是「特征到价格的分段函数」，RowNet 学的是「该参考哪几台设备」。** 前者在训练集覆盖不到的价格区间会直接外推失败，后者至少能保证预测值落在「相似设备的价格」这个合理范围内。这也是作者把标题写成「一个跑赢 XGBoost 的神经网络」的原因。

---

## 三、总结

本文中我们完成了特种设备价格预测的完整流程。我们首先用 EDA 摸清了数据的结构：一条记录就是一台二手工程设备，特征分成 28 个类别列和 24 个数值列，另外还有两列自由文本（链接和描述）被我们直接丢弃。其次我们做了相当重的数据清洗——数值列里混着不换行空格和逗号小数点，我们用正则统一抠数字；把日期换算成「距最新日期多少天」、把 `6x4` 的轴数配置编码成 64；缺失值按语义分三种情况处理（钻机参数填 -1、类别填 "-1"、普通数值填中位数）；最后用 `LabelEncoder` 把类别变成整数。接着我们实现了 RowNet：先用三种衰减曲线算相似度，`softmax` 得到「参考权重」并排除自身，加权平均出第一版价格；再带着这个价格做第二轮检索；然后用 4 个专家头加门控网络和残差修正融合，损失函数是 MAPE 加上残差、熵、多样性三个正则项，训练时 batch size 固定为 1、学习率 1e-2 配余弦退火、共 10 轮。最后我们复用同一套前向逻辑对测试集逐条预测，组装成 `id` + `USD_price` 的提交文件。

这个比赛给我们最大的启发是：**回归问题不一定非要「把特征映射成数字」，也可以「去历史里找相似的例子」。** 对表格数据来说，后者往往更稳、更可解释——你甚至可以说出「这台吊车之所以估 8 万，是因为训练集里有三台同型号、同吨位、同年份的卖了 7.5 到 8.5 万」。这种「检索 + 加权」的框架，在样本量不大、目标值跨度大的场景里，是树模型之外一条很值得尝试的路。
