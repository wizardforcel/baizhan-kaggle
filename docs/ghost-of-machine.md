# Ghost of Machine

## 一、比赛简介

### 1.1 比赛背景

Ghost of Machine 是 Kaggle 上一个关于「人机混合文本边界定位」的挑战。训练集里每条样本都是一段完整文本，前半部分由人类撰写，后半部分由 AI 模型续写。我们拿到的是拼接好的整段文本，不知道切点具体落在哪个字符上。比赛的目标并不是把 AI 文本“破译”出来，也不是判断整段文本是不是 AI 写的，而是精确地找出「AI 从哪一个字符位置开始接管」。

为什么这个问题值得研究？在内容审核、AI 生成内容标注、版权溯源等场景里，我们经常需要知道一段文字里哪些部分是人写的、哪些部分是机器写的。如果只知道“含有 AI 内容”，却不知道分界在哪里，后续的标注和追责就很难落地。因此这个比赛把问题定义成一个纯粹的“边界定位”任务。

难度在于，人类文本与 AI 续写在用词和句式上非常接近，单看一个词、一句话很难分辨；真正可用的信号往往藏在较长片段的统计规律里。材料里的 EDA 还揭示了一个关键先验：边界几乎总是落在句子起点上，这让我们可以把“字符级回归”转化为“句子级分类”，大幅缩小搜索空间。

### 1.2 比赛目标

输入是一段文本 `text` 和一个样本编号 `id`。训练集额外提供 `boundary_char_index`，表示 AI 部分起始的字符下标（0 表示开头就是 AI，等于文本长度表示整段都是人类写的）。测试集只有 `id` 与 `text`，需要为每条测试样本预测这个边界下标。

数据表结构在代码里体现得很清楚：

```python
# 读取三个文件：训练集、测试集、提交样例
train = pd.read_csv(TRAIN_DIR)   # 列：id, text, boundary_char_index
test  = pd.read_csv(TEST_DIR)    # 列：id, text
sample_sub = pd.read_csv(SAMPLE_SUBMISSION)  # 列：id, boundary_char_index
```

文本长度不一，边界值随文本长度变化，从 0 到 `len(text)` 都有可能。测试集的 `boundary_char_index` 在材料里被临时置为 -1，只是占位，不参与训练。

### 1.3 评价标准

比赛的评分函数在材料里写得很直白：

```python
def comp_score(pred):
    t = truth.loc[pred.index]                  # 取对应的真实边界
    return np.mean(np.exp(-np.abs(pred - t) / 100)) * 100
```

也就是 `score = 100 * mean(exp(-|pred - true| / 100))`。预测完全命中时每一项都是 1，总分 100；偏差 100 个字符时每项约 0.368，总分约 36.8；偏差越大越接近 0。这是一种“容忍小误差、惩罚大误差”的指数衰减指标，和单纯的对错准确率不同，它鼓励模型尽量靠近真实边界，即使不能完全命中也能拿到部分分数。排行榜分数越高越好。

---

## 二、解题过程

整场比赛的主线可以概括为：先把每条长文本拆成句子，用句子的语义/统计特征判断它属于人类部分还是 AI 部分，再把逐句的概率聚合成一条文本的边界。我们按 EDA、数据清洗、模型训练与调参、模型测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步是确认边界到底长什么样。我们读取训练集，打印边界前后各 150 个字符：

```python
train = pd.read_csv(TRAIN_DIR)
for i in range(3):
    t, b = train.text[i], train.boundary_char_index[i]
    print(t[b-150:b], "<<<BATAS>>>", t[b:b+150])
```

打印结果里，`<<<BATAS>>>`（材料作者用的分隔词）左右两边恰好都是完整句子，边界从不落在单词中间。这个观察非常重要：它说明标注者在构造数据时是按句子拼接的，因此我们只需要在句子起点里找边界，不必逐字符扫描。

接着我们定义句子切分规则。不能简单地按句号切，因为缩写、小数点、引号都会造成误切。材料采用的正则是：只有 `.!?` 后面跟空白，并且空白后紧跟大写字母、数字、引号或左括号时，才认为是句子边界：

```python
OPEN = "A-Z0-9\"'("
SPLIT = re.compile(r'(?<=[.!?])\s+(?=[' + OPEN + r'])')

def sent_starts(text):
    # 第一个句子从 0 开始，之后每个匹配的结束位置都是新句子起点
    return [0] + [m.end() for m in SPLIT.finditer(text)]

def separate_sentences(df):
    rows = []
    for r in df.itertuples():
        t, b = r.text, r.boundary_char_index
        st = sent_starts(t)
        ends = st[1:] + [len(t)]
        for i, (s, e) in enumerate(zip(st, ends)):
            # label=1 表示这个句子落在边界之后，即属于 AI 部分
            rows.append((r.id, i, s, t[s:e], int(s >= b)))
    return pd.DataFrame(rows, columns=["id","sent_idx","start","sentence","label"])
```

这样每条文本被拆成若干行，每行记录样本 id、句子序号、句子在原文本中的起始字符位置、句子内容，以及它是否在 AI 区域内。得到的 `row_train` 行数大约是“样本数 × 平均句子数”，既保留了句子顺序，又让每个句子成为独立的分类样本。

我们还检查了拆分后的排序是否稳定：按 `id` 和 `sent_idx` 排序后的索引应当与原始索引一致。这个检查看起来琐碎，但它保证了后面 `decode` 按句子顺序累加概率时不会错位。

### 2.2 数据清洗

这份数据本身很干净：`text` 没有缺失，标签也没有异常值。需要处理的只有两件事。

第一，测试集没有真实边界，材料把 `boundary_char_index` 统一设为 -1，避免拆分句子时误用：

```python
# 测试集没有标签，先用 -1 占位
test['boundary_char_index'] = -1
row_train = separate_sentences(train)
row_test  = separate_sentences(test)
```

第二，不对文本做常规的“清洗”。平时做文本分类会转小写、去标点、去停用词，但这里**不能这么做**：大小写、标点、引号本身就是区分人类与 AI 写作风格的重要线索，删掉等于把信息扔掉。所以句子一律保留原始形式。

另一个容易被忽略的点是数据泄露。同一条文本拆出的多个句子高度相关，如果随机划分训练/验证集，同一段文本的句子可能同时出现在两边，验证分数会虚高。因此我们按 `id` 分组做 5 折交叉验证：

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(row_train, groups=row_train["id"]))
```

`GroupKFold` 保证同一 `id` 的所有句子只会出现在同一折里。虽然这里只是句子级分类，但严谨的分组方式能让我们更真实地估计模型在未见文本上的表现。

### 2.3 模型训练与超参数调优

我们采用「句子嵌入 + 树模型」的方案。先用 `sentence-transformers` 加载 `BAAI/bge-base-en-v1.5`，把每个句子编码成 768 维向量，并做 L2 归一化：

```python
from sentence_transformers import SentenceTransformer

bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
E = bge.encode(row_train.sentence.tolist(),
               batch_size=128,
               normalize_embeddings=True,   # 归一化后余弦相似度更稳定
               show_progress_bar=True)
E_test = bge.encode(row_test.sentence.tolist(),
                    batch_size=128,
                    normalize_embeddings=True)
```

这里用预训练句向量而不是从头训练，是因为训练样本有限，而 bge 已经在大规模语料上学到了通用的语义表示，能捕捉“人类写作”和“AI 续写”在句法、用词上的细微差异。

接下来用 LightGBM 做二分类。为了不浪费数据，我们用 5 折 OOF（out-of-fold）方式得到每个句子的概率：

```python
import lightgbm as lgb

y = row_train.label.values

def oof_lgb(params):
    p = np.zeros(len(row_train))
    for tr_idx, va_idx in folds:
        m = lgb.LGBMClassifier(**params, n_estimators=300, verbosity=-1)
        m.fit(E[tr_idx], y[tr_idx])
        p[va_idx] = m.predict_proba(E[va_idx])[:, 1]
    return p
```

调参用 Optuna，搜索空间包括学习率、叶子数、叶节点最小样本数、列采样、行采样和 L2 正则：

```python
import optuna

def objective(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 64),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    )
    row_train["p_tmp"] = oof_lgb(params)
    # 句子概率 → 文本边界，用比赛指标评估这组参数
    pred = row_train.groupby("id").apply(lambda g: decode(g, col="p_tmp"))
    return comp_score(pred)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1)
```

注意目标函数不是句子级 AUC，而是把句子概率解码成文本边界后的比赛分数 `comp_score`。这样调参直接优化最终指标，比优化中间任务更贴合比赛。

解码函数的思路是：把每个句子的概率转成 log 似然，假设边界在第 k 个句子之前，则第 1~k-1 个句子应属人类（概率 1-p），第 k 个之后应属 AI（概率 p）。取累计对数似然最大的切点：

```python
def decode(g, col="p", eps=1e-6):
    p = np.clip(g[col].values, eps, 1 - eps)
    left  = np.concatenate([[0], np.cumsum(np.log(1 - p))])
    right = np.concatenate([np.cumsum(np.log(p)[::-1])[::-1], [0]])
    k = 1 + np.argmax((left + right)[1:len(p)])
    return g.start.values[k]   # 返回该切点句子的起始字符下标
```

`left` 表示“前 i 个句子都是人类”的累计对数似然，`right` 表示“从 i 到结尾都是 AI”的累计对数似然，两者相加就是边界设在 i 处的总似然。取最大值对应的句子起点，就得到字符级边界。

材料里还给出了另一条基于 BERT 的路线：在真实边界处插入 `[SEP]` 作为标记，把每个 token 轮流遮住并用 MLM 头取该位置的 logit，得到逐 token 的“AI 倾向”序列；再用 1D 卷积（64→128→256 通道）提取上下文，最后用带掩码的 BCE 损失训练。这条路线更重，但能直接输出 token 级边界，适合算力充足的队伍。

### 2.4 模型测试与指标报告

训练完成后，我们用留出的验证折检查句子级概率解码回文本边界后的 `comp_score`。材料的做法是先用全部训练数据拟合最终模型，再对测试集逐句预测并解码：

```python
final = lgb.LGBMClassifier(**study.best_params,
                           n_estimators=300,
                           random_state=42, verbosity=-1)
final.fit(E, row_train.label)

row_test["p"] = final.predict_proba(E_test)[:, 1]
pred_test = row_test.groupby("id").apply(lambda g: decode(g, col="p"))

# 整理成提交格式：id + boundary_char_index
sub = pred_test.astype(int).rename("boundary_char_index").reset_index()
sub.to_csv("submission.csv", index=False)
```

提交文件只有两列，行数必须与测试集一致。我们还会抽查几条样本，确认预测边界确实落在句子起点上、没有越界。

从验证结果看，句子级分类的 AUC 较高，但解码后的边界分数会受“边界前最后一个句子”和“边界后第一个句子”概率误差的影响。这也是为什么我们直接以 `comp_score` 作为调参目标——中间指标好看不代表最终定位准。

---

## 三、总结

本文中我们完成了 Ghost of Machine 的完整解题流程。我们首先通过 EDA 发现边界总是落在句子起点，并把每条长文本拆成句子级二分类样本。其次在数据清洗上只做最小化处理，保留原始大小写和标点，并用 GroupKFold 按文本 id 分组，避免同一段文本的句子泄露到验证集。接着我们用 bge-base-en-v1.5 生成句子嵌入，用 LightGBM 判断句子是否属于 AI 部分，通过 Optuna 直接以比赛指标为目标调参，再用累计对数似然拐点把句子概率解码成字符边界。最后在全量数据上重新训练并生成提交文件。

这个比赛给我们最大的启发是：**把难以直接回归的问题转化成一系列容易判断的小问题，再用概率模型把结果聚合回去**。如果一上来就训练一个“预测字符下标”的回归模型，模型很难学到边界总在句首这个先验；而先做句子分类、再按似然解码，等于把这个先验显式地写进了流程里。另一条经验是评估指标要贯穿始终——调参、验证、最终提交都用同一个 `comp_score`，才能保证每一步优化都指向真正的目标。
