# 酒店宾客满意度预测（WUSTL 深度学习课程 2026 秋）

## 一、比赛简介

### 1.1 比赛背景

圣路易斯华盛顿大学（WUSTL）的「深度学习应用」课程在 2026 年秋季用这个比赛做课堂练习。题目听起来很生活化：**给一家酒店打分**。每家酒店有一份档案，里面有房价、房间数、建成年份、离机场多远、有没有游泳池和 spa，也有一大段「自我介绍」、运营方的叙述、员工的视角、住客评论摘录，还有一张酒店照片。我们要根据这些信息，预测这家酒店的**宾客满意度指数**。

这个任务真正的难点不在模型，而在**三种完全不同的输入**要放在一起用：

- **表格数据**：一行行的数字和类别，比如星级、房价、城市；
- **散文数据**：五段自然语言文本，长短不一；
- **图片数据**：每家酒店一张照片。

传统上做表格预测，梯度提升树就够了；文本和图片则要用预训练神经网络先转成向量（embedding，嵌入）。难点在于：**单独用图片或单独用文本去预测满意度，效果都很弱**——一张好看的照片并不能说明这家酒店的服务好。材料里给了一个很直观的数字：只用照片嵌入做岭回归，RMSE 大约是 17，而「所有酒店都猜平均值」的 RMSE 是 19.27，也就是说图片单独用只比瞎猜好一点点。

所以这个比赛的核心思路是：**让图片和文本去解释「表格模型解释不了的那部分」**，也就是残差。这是整篇 writeup 的主线。

### 1.2 比赛目标

训练集 `train.csv` 每行一家酒店，目标列是 `guest_satisfaction_index`，取值 0 到 100。特征可以分成四组：

- 数值列：`star_rating`、`total_rooms`、`year_built`、`year_last_renovated`、`avg_nightly_rate_usd`、`distance_to_airport_km`、`distance_to_city_center_km`、`staff_count`；
- 是 / 否列：`has_restaurant`、`has_pool`、`has_spa`、`has_fitness_center`、`sustainability_certified`；
- 类别列：`brand_affiliation`、`city`、`country`、`region_type`、`property_type`、`parking_type`、`pet_policy`、`accessibility_features`、`architectural_style`；
- 文本列：`description`、`operator_narrative`、`staff_perspective`、`guest_review_excerpt`、`name`；
- 另外还有入住 / 退房时间 `check_in_time`、`check_out_time`，格式是 `"15:30"` 这样的钟点。

图片放在 `images/<id>.jpg`，`id` 同时是行号和文件名。测试集 `test.csv` 有同样的列但没有目标，`sample_solution.csv` 给出提交格式：`id,guest_satisfaction_index`。

### 1.3 评价标准

比赛用 **RMSE（均方根误差）**，越低越好。

大白话：对每家酒店算「预测分减真实分」的平方，全部平均，再开根号。因为分数范围是 0 到 100，RMSE 可以直接理解成「平均猜偏多少分」。RMSE 对离谱的预测惩罚很重（误差先平方了），所以模型不能只顾大多数酒店，还得把少数预测得特别偏的拽回来。

材料给出的参照线非常有参考价值：

| 做法 | 5 折交叉验证 RMSE |
| --- | --- |
| 所有酒店都猜平均值 | 19.27 |
| 只用表格特征的梯度提升树 | 13.66 |
| 加上图片 / 文本残差模型 | 7.25 |
| 再加上「调节变量」搜索（完整流程） | 6.42 |
| 3 个折种子平均（最终提交） | 6.245 |

从 19.27 到 6.245，误差砍掉了差不多三分之二，其中**表格特征是主力**（19.27 → 13.66），图片和文本贡献了第二段（13.66 → 7.25），调节变量搜索又抠掉了最后一段（7.25 → 6.42）。

---

## 二、解题过程

我们首先把训练集和测试集拼成一张表，让所有特征在两边用完全相同的规则计算；其次做表格特征工程，并训练一个梯度提升树基线；接着用 CLIP 提取照片向量、用 MiniLM 和 TF-IDF 提取文本向量；然后把图片和文本模型的目标改成「表格模型的残差」，并搜索哪些表格字段会放大或削弱图片 / 文本的作用；最后做两轮 backfitting，用一个最终的大模型把所有信号合起来，并用三个折种子平均出提交文件。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集**拼在一起**，再去掉目标列。这样做的好处是：所有「频次」「中位数」这类统计量在训练和测试上口径一致，而且不会因为两边分开算导致特征分布有细微差别。下面的 `TRAIN_CSV`、`TEST_CSV`、`SAMPLE_CSV` 都是 notebook 顶部配置单元里写好的路径常量：

```python
import numpy as np
import pandas as pd

TARGET = 'guest_satisfaction_index'
ID = 'id'

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sample = pd.read_csv(SAMPLE_CSV)

# 训练行在前、测试行在后。后面所有特征计算都用这个 df，
# 最后再按行数切回 train / test，保证两边处理完全一样
df = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
n = len(train)
y = train[TARGET].values.astype(float)

print(train.shape, test.shape)
print(train[TARGET].describe().round(2))
print(f'恰好等于 0 的比例: {np.mean(y == 0):.1%}')
print(f'恰好等于 100 的比例: {np.mean(y == 100):.1%}')
```

`describe()` 和两个端点比例要回答的是：分数是集中在中间，还是在两端扎堆。如果 0 分和 100 分各有相当比例的样本，就不能假设目标服从正态分布，用线性模型加平方损失会很别扭；树模型不假设分布形状，只在取值范围内分段拟合，更适合这种「两头都有」的目标。

接着看**缺失值**。这一步在这个比赛里格外重要，因为「缺哪些字段」本身就是信号：

```python
# 每一列的缺失比例
blank = df.isna().mean()
print(blank[blank > 0].map('{:.1%}'.format))

# 每行缺了几个字段（不算文本列），再看这些行的平均满意度
n_blank = train.drop(columns=[TARGET] + TEXT).isna().sum(axis=1)
print(train.groupby(n_blank)[TARGET].agg(['mean', 'count']).round(1))
```

结果很惊人：**缺字段越多的酒店，满意度平均低大约 12 分**。这说明缺失不是随机的，而是「这家酒店的资料不全」本身就和「服务差」相关。所以后面我们不会去「猜」缺失值填什么，而是**给每个有缺失的列加一个「是否缺失」的标记**，让模型自己学这个信号。

我们还确认了图片和文本的规模：训练集大约 8000 行，训练加测试一共 10000 张照片，每张都有对应的 `<id>.jpg`；五段文本里有长有短，`name` 最短，只适合做句子嵌入，不适合做词频统计。文本长度我们也统计了均值、最小值、中位数和最大值，确认没有超长文本撑爆显存。

### 2.2 数据清洗

这个比赛没有「脏数据」要修，清洗的主要工作是**把各种格式统一成模型能吃的数值**，并且保证不泄漏标签。

第一件事是把钟点转成小时数。`"15:30"` 这样的字符串模型看不懂，转成 `15.5` 才有大小含义：

```python
def clock_hours(s):
    """'15:30' -> 15.5；缺失值保持 NaN。"""
    # 用正则把小时和分钟拆成两列，再算小数
    p = s.astype('string').str.extract(r'(\d+):(\d+)').astype(float)
    return p[0] + p[1] / 60
```

第二件事是处理高基数的类别列（比如 `city`，可能有几百个城市）。梯度提升树 `HistGradientBoostingRegressor` 原生支持类别特征，但**最多只支持 255 个水平**，超出的部分只能当缺失处理。所以我们的做法是把出现次数最多的 250 个取值保留，其余合并成 NaN：

```python
def category_codes(s, max_levels=250):
    """把出现最多的 max_levels 个取值编成 0..k-1 的整数，其余变 NaN。"""
    keep = s.value_counts().index[:max_levels]
    s = s.where(s.isin(keep))
    levels = sorted(s.dropna().unique())      # 排序保证编码稳定、可复现
    return s.map({v: i for i, v in enumerate(levels)}).astype(float)
```

注意这里**只用了特征本身的频次，没有用标签**。任何用到标签的编码（比如目标编码）都必须放进交叉验证的折内，否则会把答案泄漏进特征。

第三件事是造派生特征。这部分是表格模型提分的关键：

```python
def tabular_features(df):
    """输入是 train + test 拼好的表，输出纯特征表（不需要标签）。"""
    X = pd.DataFrame(index=df.index)

    # 原始数值列直接保留
    for c in NUMERIC:
        X[c] = df[c].astype(float)

    # 是 / 否列转成 1 / 0，树模型对 0/1 最友好
    for c in YES_NO:
        X[c] = df[c].map({'Yes': 1.0, 'No': 0.0})

    # 无障碍设施是有序的：无 < 基础 < 中等 < 全面
    X['access_ord'] = df['accessibility_features'].map(
        {'No': 0, 'Basic': 1, 'Moderate': 2, 'Comprehensive': 3})

    # 缺失标记：这列缺不缺，本身就是强信号（缺字段的酒店低约 12 分）
    miss_cols = [c for c in df.columns
                 if c not in TEXT + [TARGET, ID] and df[c].isna().any()]
    for c in miss_cols:
        X['miss_' + c] = df[c].isna().astype(float)
    X['n_missing'] = df[miss_cols].isna().sum(axis=1)

    # 业务上有意义的比值和差值
    X['age'] = 2026 - df['year_built']
    X['since_renovation'] = 2026 - df['year_last_renovated']
    X['staff_per_room'] = df['staff_count'] / df['total_rooms']
    X['log_rate'] = np.log1p(df['avg_nightly_rate_usd'])   # 房价右偏，取对数
    X['rate_per_star'] = df['avg_nightly_rate_usd'] / df['star_rating']
    X['n_amenities'] = X[['has_restaurant', 'has_pool',
                          'has_spa', 'has_fitness_center']].sum(axis=1)

    # 同城 / 同星级 / 同物业类型的「相对水平」：
    # 一家酒店贵不贵，要看它和同城同行比，而不是看绝对价格
    for g in ['property_type', 'city', 'star_rating', 'region_type', 'country']:
        for c in ['avg_nightly_rate_usd', 'total_rooms',
                  'distance_to_airport_km', 'distance_to_city_center_km',
                  'staff_count', 'year_built']:
            X[f'{c}_rel_{g}'] = df[c] - df.groupby(g)[c].transform('median')

    return X
```

这段代码里有三个「为什么」。第一，**缺失标记单独成列**，而不是去填一个「猜出来的值」——缺失本身有信息，填成中位数反而把信息抹掉了。第二，**取对数处理房价**，因为房价是长尾分布，少数豪华酒店会把数值拉得非常大，取对数后分布更接近钟形。第三，**「同组相对值」**：一家酒店 300 美元一晚算贵还是便宜，取决于它在什么城市、什么星级；减去同组中位数，就把「绝对价格」变成了「相对同行的溢价」，这个信号比原始价格更接近「宾客觉得值不值」。

所有这些统计（频次、中位数）都只用了特征，**没有用 `guest_satisfaction_index`**，所以即使训练测试拼在一起算，也不存在标签泄漏。

### 2.3 模型训练与超参数调优

**第一步：表格基线。** 我们用的是 `HistGradientBoostingRegressor`（下面简称 HGB），它可以直接处理 NaN 和类别列，不用先做缺失填充和独热编码，非常适合这种「一半数值一半类别」的表：

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

def hgb(params, seed, columns):
    """按列名判断哪些是类别列，交给 HGB 原生处理。"""
    is_cat = np.array([c in CATEGORICAL for c in columns])
    return HistGradientBoostingRegressor(
        categorical_features=is_cat if is_cat.any() else None,
        random_state=seed, **params)

def cv_hgb(X, Xtest, y, folds, params, seed, with_test=True):
    """返回折外预测（OOF）和测试集上各折模型的平均预测。

    OOF 的意思是：每一行都由「没见过这一行」的模型预测，得到的分数
    才是对未见数据的诚实估计。
    """
    oof = np.zeros(len(y))
    test = np.zeros(len(Xtest)) if with_test else None
    for a, b in folds:
        m = hgb(params, seed, X.columns).fit(X.iloc[a], y[a])
        oof[b] = m.predict(X.iloc[b])
        if with_test:
            test += m.predict(Xtest) / len(folds)
    return oof, test

def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))

folds = list(KFold(5, shuffle=True, random_state=42).split(X_train))
TAB_HGB = dict(learning_rate=0.03, max_iter=600, max_leaf_nodes=15,
               min_samples_leaf=30, l2_regularization=1.0)
tab_oof, _ = cv_hgb(X_train, None, y, folds, TAB_HGB, 42, with_test=False)
print('表格 HGB OOF RMSE:', rmse(tab_oof, y))     # 13.66
```

树的数量、叶子数、学习率这些参数是在**同一套折**上定的，只有折一致，不同参数之间的比较才公平。基线的 RMSE 是 13.66，比「猜均值」的 19.27 好了一大截，说明表格特征本身很有信息量。

**第二步：把照片变成向量。** 我们用的是 CLIP 的 ViT-B/32 版本，它能把一张图编码成 512 个数字。这里有个小技巧：把原图和它的**镜像**各编码一次再取平均，这是一种最便宜的数据增强（test-time augmentation），能让向量稍微稳定一点：

```python
def clip_image_embeddings(ids, image_dir, cache_file, batch_size=64):
    """返回 L2 归一化后的 CLIP 图片向量。"""
    out = []
    for i in range(0, len(ids), batch_size):
        ims = [Image.open(os.path.join(image_dir, f'{j}.jpg')).convert('RGB')
               for j in ids[i:i + batch_size]]
        px = proc(images=ims, return_tensors='pt')['pixel_values'].to(DEVICE)

        e = 0
        # 原图和镜像图各算一次，平均后作为这张图的表示
        for x in (px, torch.flip(px, dims=[3])):
            pooled = model.vision_model(pixel_values=x).pooler_output
            e = e + model.visual_projection(pooled)
        out.append((e / 2).float().cpu().numpy())

    emb = np.concatenate(out)
    # 归一化后，向量点积就等于余弦相似度，岭回归的尺度也更稳定
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.savez(cache_file, ids=ids, model=np.array(CLIP_MODEL), emb=emb)
    return emb
```

嵌入是整份 notebook 里最慢的一步（10000 张图），所以我们把它存进缓存文件，并在文件里记下「这份缓存对应的是哪个模型、哪些 id」。下次重跑时只要模型和行号都对得上，就直接读缓存，省掉几分钟。

**第三步：把文本变成向量。** 文本有两条互补的路线。MiniLM 是句向量模型，输出 384 维，擅长抓「语义」；TF-IDF 是传统的词频统计，擅长抓「关键词」。我们对四段长文本用 TF-IDF，对包括 `name` 在内的五段文本都用 MiniLM：

```python
def minilm_embeddings(df, cache_file, batch_size=64):
    """对每一段文本列分别做均值池化，再拼成一个大向量。"""
    blocks = {}
    for c in TEXT:
        texts = df[c].fillna('').astype(str).tolist()
        embs = []
        for i in range(0, len(texts), batch_size):
            b = tok(texts[i:i + batch_size], padding=True, truncation=True,
                    max_length=256, return_tensors='pt').to(DEVICE)
            h = model(**b).last_hidden_state
            # 只对"真实 token"求平均，padding 位置不能算进去
            m = b['attention_mask'].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            embs.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        blocks[c] = np.concatenate(embs).astype(np.float32)
    return np.hstack([blocks[c] for c in TEXT])

def tfidf_matrix(df):
    """四段长文本各建一个词表，1-2 元词组，拼在一起。"""
    mats = [TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                            max_features=50000, sublinear_tf=True)
            .fit_transform(df[c].fillna(''))
            for c in LONG_TEXT]
    return sp.hstack(mats).tocsr()
```

`attention_mask` 那一步很关键：一批文本里短的会被补零到和最长的一样，如果直接对所有位置求平均，短文本的向量会被大量「补位零」稀释。乘上 mask 再除以真实长度，才是真正的句向量。

**第四步：残差堆叠（本场比赛的核心）。** 直接拿图片向量去拟合满意度，RMSE 约 17，几乎没用。但拿它去拟合**表格模型犯的错**，就有用了：

```python
from sklearn.linear_model import Ridge
from scipy import sparse as sp

def interact(M, Z):
    """[M, M*z1, M*z2, ...]：让线性模型的系数随 z 变化。"""
    if Z is None or Z.shape[1] == 0:
        return M
    if sp.issparse(M):
        return sp.hstack([M] + [M.multiply(Z[:, [j]])
                                for j in range(Z.shape[1])]).tocsr()
    return np.hstack([M] + [M * Z[:, [j]] for j in range(Z.shape[1])])

def cv_ridge(M, Mtest, t, folds, alpha, Z=None, Ztest=None):
    """折安全的岭回归：返回 (OOF, 测试预测)，每个都是 (行数, 1+调节变量数)。

    第 0 列是主预测 M·w0；后面每列是某个调节变量对应的斜率 M·w_j，
    最终模型可以自己决定要不要用这些斜率。
    """
    k = 0 if Z is None else Z.shape[1]          # 有几个调节变量
    p = M.shape[1]                              # 嵌入的维度
    F, Ft = interact(M, Z), interact(Mtest, Ztest)
    oof = np.zeros((M.shape[0], 1 + k))
    test = np.zeros((Mtest.shape[0], 1 + k))

    for a, b in folds:
        m = Ridge(alpha=alpha).fit(F[a], t[a])
        oof[b, 0] = m.predict(F[b])
        test[:, 0] += m.predict(Ft) / len(folds)
        if k:
            # 把系数矩阵还原成 (1+k) 组、每组 p 个权重
            W = m.coef_.reshape(1 + k, p)
            for j in range(k):
                oof[b, 1 + j] = M[b] @ W[1 + j]          # 斜率分数
                test[:, 1 + j] += (Mtest @ W[1 + j]) / len(folds)
    return oof, test

r0 = y - tab_oof                     # 表格模型没解释掉的部分
photo_y = cv_ridge(clip[:n], clip[n:], y, folds, alpha=3.0)[0][:, 0]
photo_r0 = cv_ridge(clip[:n], clip[n:], r0, folds, alpha=3.0)[0][:, 0]
print('图片 -> 满意度 RMSE:', rmse(photo_y, y))     # 约 17
print('图片 -> 残差 RMSE:', rmse(photo_r0, r0))     # 明显更好
```

`interact` 把 `[M]` 扩展成 `[M, M·z1, M·z2, ...]` 之后，岭回归学到的就不只是一个固定的图片权重，而是「图片权重 = 主权重 + 每个调节变量 × 一个斜率」。

这就是**残差堆叠**：模型学的不是「这家酒店多少分」，而是「表格模型把这家的分猜高/猜低了多少」。图片、文本各自学一部分残差，学到的预测再作为新特征喂给最终模型。

**第五步：调节变量搜索。** 我们还问了一个更有意思的问题：**照片的作用会不会随表格字段变化？** 比如「好看的照片对贵酒店更重要」。做法是给岭回归加「交互项」——让图片向量的系数不是一个常数，而是 `w0 + z·w1`，其中 `z` 是某个标准化后的表格列。然后贪心地一个一个试，只有当某个字段能把残差 RMSE 至少降低 0.05 时才保留，每个模态最多保留 2 个：

```python
def select_moderators(M, t, cands, folds, label):
    """贪心前向搜索：找出最能放大 / 削弱某个模态作用的表格字段。"""
    P = PCA(32, random_state=0).fit_transform(M)   # 先压到 32 维，加速试验

    def score(names):
        Z = np.column_stack([cands[c] for c in names]) if names else None
        F = interact(P, Z)
        o = np.zeros(len(t))
        for a, b in folds:
            o[b] = Ridge(alpha=1.0).fit(F[a], t[a]).predict(F[b])
        return rmse(o, t)

    chosen, best = [], score([])
    for _ in range(MAX_MODERATORS):
        trials = sorted((score(chosen + [c]), c)
                        for c in cands if c not in chosen)
        # 提升不到 0.05 就停手，避免把噪声当成信号
        if best - trials[0][0] < MIN_MODERATOR_GAIN:
            break
        best, c = trials[0]
        chosen.append(c)
    print(label, '选中的调节变量:', chosen)
    return chosen
```

搜索只在第一个折种子的残差上跑一次，避免反复使用标签造成的过拟合。加不加这一步，分数差别很大：完整流程 6.42，关掉调节变量搜索是 7.25。

**第六步：backfitting 与最终模型。** 第一轮里图片和文本都学「表格的残差」，会学到重叠的东西。于是我们再做两轮 backfitting：**图片模型改学「表格 + 文本解释不了的残差」，文本模型改学「表格 + 图片解释不了的残差」**，各自去补对方的短板。最后把表格特征和所有模态的 OOF 预测拼在一起，交给一个「很多棵很小的树」的 HGB：

```python
FINAL_HGB = dict(learning_rate=0.03, max_iter=5000, max_leaf_nodes=4,
                 min_samples_leaf=20, l2_regularization=1.0,
                 max_features=0.3)
# backfitting 专用的树模型：比最终模型小得多，只负责更新残差目标
BACKFIT_HGB = dict(learning_rate=0.05, max_iter=2000, max_leaf_nodes=4,
                   min_samples_leaf=30, l2_regularization=1.0,
                   max_features=0.3)

def cv_pls(M, Mtest, t, folds, k):
    """取 k 个与残差最相关的 PLS 方向，同样按折训练。"""
    oof = np.zeros((M.shape[0], k))
    test = np.zeros((Mtest.shape[0], k))
    for a, b in folds:
        m = PLSRegression(k, scale=False).fit(M[a], t[a])
        oof[b] = m.transform(M[b])          # 把样本投影到这几个方向上
        test += m.transform(Mtest) / len(folds)
    return oof, test

def with_meta(T, meta, part):
    """表格特征后面接上模态预测：part=0 用 OOF，part=1 用测试预测。"""
    X = T.copy()
    for k, v in meta.items():
        X['m_' + k] = v[part]
    return X

def modality_meta(prefix, spec, t, folds, cands, n):
    """一个模态的岭回归，返回主预测和每个调节变量对应的斜率。"""
    mods = spec['mods']
    Z = np.column_stack([cands[c] for c in mods]) if mods else None
    oof, test = cv_ridge(spec['M'], spec['Mt'], t, folds,
                         spec['alpha'], Z[:n] if mods else None,
                         Z[n:] if mods else None)
    out = {prefix: (oof[:, 0], test[:, 0])}
    for j, c in enumerate(mods):
        out[f'{prefix}_slope_{c}'] = (oof[:, 1 + j], test[:, 1 + j])
    return out

# 每个模态的配置：训练行矩阵、测试行矩阵、岭回归的 alpha、选中的调节变量
img = {'clip': dict(M=clip[:n], Mt=clip[n:], alpha=3.0, mods=img_mods)}
txt = {'tfidf': dict(M=tfidf[:n], Mt=tfidf[n:], alpha=10.0, mods=txt_mods),
       'minilm': dict(M=mini[:n], Mt=mini[n:], alpha=30.0, mods=txt_mods)}
```

三种模态的岭回归强度不一样：图片用 3，TF-IDF 用 10，MiniLM 用 30。特征越宽、越稀疏、噪声越大，通常越需要强的正则来压住系数。这些值是同一套折上试出来的。

配置固定之后，整套堆叠就可以封装成一个「给一个折种子、吐一份预测」的函数：

```python
def run_seed(seed, tab_oof):
    folds = list(KFold(5, shuffle=True, random_state=seed).split(X_train))
    r_img = r_txt = y - tab_oof
    meta = {}

    for rnd in range(N_BACKFIT + 1):          # 第 0 轮 + 2 轮 backfitting
        if rnd > 0:
            # 图片改学"表格+文本"的残差，文本改学"表格+图片"的残差
            txt_meta = {k: v for k, v in meta.items() if k.startswith('txt_')}
            img_meta = {k: v for k, v in meta.items() if k.startswith('img_')}
            r_img = y - cv_hgb(with_meta(X_train, txt_meta, 0), None,
                               y, folds, BACKFIT_HGB, seed,
                               with_test=False)[0]
            r_txt = y - cv_hgb(with_meta(X_train, img_meta, 0), None,
                               y, folds, BACKFIT_HGB, seed,
                               with_test=False)[0]
        meta = {}
        for k, spec in img.items():
            meta.update(modality_meta('img_' + k, spec, r_img, folds, cands, n))
        for k, spec in txt.items():
            meta.update(modality_meta('txt_' + k, spec, r_txt, folds, cands, n))

    # 再补 3 个 PLS 方向，让最终模型能用图片向量的更多维度
    pls_oof, pls_test = cv_pls(img['clip']['M'], img['clip']['Mt'],
                               r_img, folds, 3)
    for j in range(3):
        meta[f'img_pls{j}'] = (pls_oof[:, j], pls_test[:, j])

    return cv_hgb(with_meta(X_train, meta, 0), with_meta(X_test, meta, 1),
                  y, folds, FINAL_HGB, seed)
```

最终模型用 5000 棵**只有 4 个叶子**的树，学习率 0.03，并且每棵树只随机看 30% 的列。为什么是「很多很小的树」而不是「少数很深的树」？因为我们要模型学的是「图片信号 × 表格字段」这类**低阶交互**，小树一次只切一两个变量，正好适合；深树容易记住个别样本，反而过拟合。

### 2.4 模型测试与指标报告

为了降低单次划分的运气成分，我们把整套流程在 **3 个折种子**（42、7、2026）上各跑一遍。每个种子的测试预测是 5 折模型的平均，三个种子再平均，一共 15 个模型：

```python
SEEDS = [42, 7, 2026]
oofs, tests = [], []
for seed in SEEDS:
    o, t = run_seed(seed, tab_oof if seed == 42 else None)
    oofs.append(o)
    tests.append(t)

# 目标范围是 0~100，超出范围的预测没有意义，直接截断
oof = np.clip(np.mean(oofs, axis=0), 0, 100)
pred = np.clip(np.mean(tests, axis=0), 0, 100)

print('各折种子 OOF RMSE:',
      [round(rmse(np.clip(o, 0, 100), y), 3) for o in oofs])
print('种子平均 OOF RMSE:', round(rmse(oof, y), 3))   # 6.245
```

分数汇总如下：

| 阶段 | CV RMSE |
| --- | --- |
| 猜均值 | 19.27 |
| 表格 HGB | 13.66 |
| 残差模态 + 2 轮 backfitting + 最终 HGB（关闭调节变量搜索） | 7.25 |
| 加上调节变量搜索 | 6.42 |
| 3 个折种子平均（提交版本） | 6.245 |

**这个 CV 可信吗？** 这是个好问题。虽然每一行都是由「没见过它」的模型预测，但标签仍然可能通过间接路径影响特征：这一行的标签参与了别的行的残差目标，调节变量搜索也用到了全部标签。所以堆叠的 CV 有可能偏乐观。作者做了一个很扎实的检验：**留出 1600 行完全不参与训练**，在剩下的 6400 行上把整套流程（包括调节变量搜索）重跑一遍。结果留出集上的 RMSE 是 **6.13**，而内部 CV 是 6.51——留出集甚至更好，说明**没有过拟合的迹象**。

最后生成提交文件。格式要求是「和 `sample_solution.csv` 完全一致的表头、每个测试 id 都有一行、行顺序一样」，所以我们用 `merge` 对齐，而不是直接按顺序写：

```python
sub = sample[[ID]].merge(
    pd.DataFrame({ID: test[ID], TARGET: pred}), on=ID, how='left')

# 断言：行数必须一致、不能有 id 没拿到预测
assert len(sub) == len(sample) and sub[TARGET].notna().all()
sub = sub[list(sample.columns)]          # 列顺序也照抄样例
sub.to_csv(SUBMISSION_PATH, index=False)

# OOF 预测单独存一份，方便做误差分析
pd.DataFrame({ID: train[ID], TARGET: oof}).to_csv('oof_predictions.csv', index=False)
```

把 OOF 存下来是一个很实用的习惯：按绝对误差从大到小排序，就能看到模型最离谱的那些酒店长什么样——是照片拍得怪，还是描述写了特殊服务，还是房价极值。这比盯着一个总分有用得多。

---

## 三、总结

本文中我们完成了酒店宾客满意度预测的完整流程。我们首先把训练集和测试集拼成一张表统一计算特征，用目标分布和缺失统计发现「缺字段多的酒店平均低约 12 分」；其次把钟点转成小时、把高基数类别压到 250 个水平、造出年龄、每房员工数、对数房价和「同城同星级相对价格」等派生特征，并给每个缺失列加标记；接着用 HGB 拿到 13.66 的表格基线，用 CLIP 和 MiniLM / TF-IDF 把图片和文本变成向量，再用岭回归去拟合表格模型的**残差**，并用贪心搜索找出能放大图片 / 文本作用的表格字段；然后做两轮 backfitting 让图片和文本互补，用 5000 棵 4 叶小树做最终模型；最后用 3 个折种子平均，得到 6.245 的 CV RMSE，并通过 1600 行留出检验确认 CV 没有虚高。

这个比赛给我们的最大启发是：**弱模态不要直接预测目标，要去预测强模型的残差**。单看图片，RMSE 17 几乎等于瞎猜；但让它解释「表格模型猜错的部分」，它就成了有用的信号。另一个启发是**「缺失」本身可以是特征**——如果数据缺失不是随机的，填一个中位数反而是在抹掉信息。
