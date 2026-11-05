# T-Pokolenie ML: профиль «Лапки»

## 一、比赛简介

### 1.1 比赛背景

这次我们要写的是 Kaggle 上的 `t-pokolenie-ml-profil-lapki`。从名字看，它属于 T-Pokolenie ML 系列比赛，`профиль` 可以理解为“方向”或“赛道”，`лапки` 在俄语里是“小爪子”的意思。材料里只有一份标题为 `lapki baseline rubert-tiny2` 的 notebook，没有展开讲赛题故事，所以我们不猜测具体业务场景，而是从数据形态和代码出发理解任务。

这类比赛和常见的“给一张图判断猫还是狗”很不一样。它的候选不是固定的两个类别，而是一个所有查询共用的候选池。每个查询会拿到一段文本，候选池里的每个候选也有自己的文本，模型要做的是判断“这个查询最可能对应哪个候选”。这属于文本检索和候选排序问题，也叫 learning to rank。实际生活里，搜商品、找简历、匹配问答对，都是同样的形状。

比赛提供三个核心文件。`train.csv` 里有查询编号 `query_id`、正确答案 `candidate_id` 和查询文本 `cipher`；`test.csv` 里只有 `query_id` 和 `cipher`，没有答案；`candidates.csv` 里有候选编号 `candidate_id` 和候选文本 `text`。也就是说，训练集告诉我们“哪个查询对应哪个候选”，测试集要求我们把这个对应关系推出来。

### 1.2 比赛目标

比赛的目标很具体：对于测试集里的每一个 `query_id`，从候选池中选出 5 个候选，并按照匹配程度从高到低排好序，分别写进 `candidate_id_1`、`candidate_id_2`，一直到 `candidate_id_5`。排在第一位的候选，就是模型认为最可能正确的那个。

这里有两个容易忽略的点。第一，五个候选必须是互不相同的，不能把同一个候选重复写五次。第二，五个候选都必须来自 `candidates.csv`，不能自己编一个编号。比赛不要求模型输出一段解释，也不要求输出概率，只要求给出一个排好序的候选列表。所以我们的任务可以拆成两步：先给每个“查询—候选”对打分，再按分数排序取前五。

训练集里每个查询只有一个正确答案，但测试集里我们不知道答案。模型需要从训练集里学会“什么样的查询文本，应该匹配什么样的候选文本”，然后把这个能力迁移到没见过的查询上。

### 1.3 评价标准

官方指标是 MRR@5，全称是 Mean Reciprocal Rank at 5，也就是“前五名平均倒数排名”。它的计算方式很直白：对于每个查询，看正确答案出现在第几名。如果正确答案排第 1，这个查询得 1 分；排第 2，得 1/2 分；排第 3，得 1/3 分；排第 4，得 1/4 分；排第 5，得 1/5 分；如果正确答案根本不在前五名里，这个查询得 0 分。最后把所有查询的分数加起来，再除以查询数量，就是 MRR@5。

这个指标的特点是“越靠前越值钱”。把正确答案从第 5 名提到第 4 名，只多拿 0.05 分；从第 2 名提到第 1 名，却能多拿 0.5 分。所以模型不仅要“找得到”，还要“排得准”。相比之下，普通的准确率只看第一名对不对，Recall@5 只看前五名里有没有，MRR@5 把位置信息也考虑进去了，更严格，也更贴近真实搜索场景。

材料里的代码除了算 MRR@5，还顺手算了两个辅助指标：`accuracy` 是第一名命中的比例，`recall5` 是前五名里至少命中一个的比例。它们不直接决定排名，但能帮我们判断模型是“完全找不到”还是“找到了但排得不够靠前”。另外，提交文件有严格的格式校验：每个 `query_id` 必须出现且只出现一次，每行必须有 5 个互不相同的候选，候选编号必须都在候选表里，不能多查询也不能少查询。格式错了，分数再高也提交不上去。

## 二、解题过程

我们拿到材料后，先通读了一遍 baseline 的代码结构，然后按照“读数据、看字段、做切分、提特征、训模型、写提交”的顺序复现。下面按照实际做事的顺序展开。

### 2.1 数据探索（EDA）

我们首先把三个 CSV 读进来，并且把候选编号映射成整数下标。这样做的原因是，后面无论是取标签、算分数还是做交叉验证，用整数下标都比用字符串方便，也不容易因为编号格式不一致而出错。

```python
# 用 utf-8-sig 读取，自动去掉 Windows/Excel 常见的 BOM 头
def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

train = read_csv(DATA / 'train.csv')
test = read_csv(DATA / 'test.csv')
candidates = read_csv(DATA / 'candidates.csv')

# 建立 candidate_id 到行号的映射，后面把“正确答案”变成整数下标
candidate_index = {}
for pos, row in enumerate(candidates):
    candidate_index[row['candidate_id']] = pos

# 训练集的 candidate_id 是字符串，这里换成模型好用的整数标签
labels = np.array([candidate_index[row['candidate_id']] for row in train])

# 查询文本在 cipher 列，候选文本在 text 列，先确认列名没有写错
assert 'cipher' in train[0] and 'text' in candidates[0]
print(len(train), len(test), len(candidates))
```

看完字段后，我们对数据有了几个基本判断。第一，查询侧和候选侧的文本是分开存放的，查询在 `train.csv` 和 `test.csv` 里，候选在 `candidates.csv` 里，这意味着我们不需要把两边拼成一个大表，而是分别编码成向量，再做匹配。第二，训练集里每个查询只有一条正确候选，标签是唯一的，所以这可以看成一个“多选一”的排序问题，而不是多标签分类。第三，候选池是所有查询共享的，同一个候选可能被多个查询命中，这一点会直接影响后面的验证集切分。

我们还检查了文本长度。`rubert-tiny2` 最多只处理 256 个 token，如果某条文本特别长，就会被截断，尾部信息会丢失。baseline 的做法是先统计所有查询和候选的最大 token 数，如果超过 256 就主动报错，提醒我们调大 `MAX_LENGTH`，而不是悄悄截断。这个检查看起来不起眼，但能避免“模型效果突然变差却不知道原因”的情况。

材料没有给出训练集、测试集和候选集的具体数量，所以我们不编造数字，只保留代码里的打印语句。对入门者来说，记住一点就够了：先看清楚每一列是什么意思，再决定怎么切分、怎么编码，比一上来就调模型重要得多。

### 2.2 数据清洗

我们其次做的是数据清洗和一致性检查。CSV 文件最常见的坑是编码问题。Windows 或 Excel 导出的文件经常带 BOM，如果直接用普通 `utf-8` 读，第一列的列名可能变成带不可见字符的 `﻿query_id`，后面用 `row['query_id']` 就会报 `KeyError`。所以我们统一用 `utf-8-sig`，它会自动把 BOM 去掉。

第二个检查是文本长度。我们先把所有查询文本和候选文本过一遍 tokenizer，统计最长的 token 数。如果最大值超过 `MAX_LENGTH`，就说明 256 不够用，需要调大截断长度。实际写模型时，我们仍然会打开 `truncation=True` 作为兜底，但兜底不等于可以忽略问题：如果大量文本都被截断，模型看到的信息就是不完整的，指标会受影响。

```python
# 先把所有文本过一遍 tokenizer，确认 256 真的够用
def check_text_lengths(tok, texts):
    maximum = 0
    for text in texts:
        ids = tok.encode(text)
        maximum = max(maximum, len(ids))
    if maximum > MAX_LENGTH:
        raise ValueError('存在超长文本，请调大 MAX_LENGTH')
    return maximum

# 查询在 cipher 列，候选在 text 列；两边都要检查
query_texts = [row['cipher'] for row in train + test]
candidate_texts = [row['text'] for row in candidates]
print(check_text_lengths(tokenizer, query_texts + candidate_texts))
```

第三个处理是向量层面的清洗。候选向量算出来以后，我们先减去所有候选向量的均值，再做 L2 归一化。减去均值是为了去掉候选池里“大家都差不多”的共同偏移，让真正有区分度的方向更突出；归一化是为了让每个候选向量的长度都变成 1，这样两个向量的点积就等于余弦相似度，不会因为某条文本很长、向量模很大，就天然拿到更高的分数。

此外，我们对提交文件也做了校验：每一行必须有 5 个不同的候选，候选必须在候选表里，`query_id` 不能重复也不能缺失。这个校验和官方 `score` 函数的检查逻辑是一致的。提前在本地跑一遍，可以避免辛苦训完模型却因为格式问题白忙一场。

### 2.3 模型训练与超参数调优

我们接着做模型训练和超参数调优。文本编码器用的是 `cointegrated/rubert-tiny2`，这是一个很小的俄语 BERT，参数量少，CPU 也能跑，非常适合做 baseline。我们还固定了模型的 revision，这样以后重新运行时，拉到的是同一份权重，结果可复现，不会因为作者更新模型而悄悄变化。

第一步是切分训练集和验证集。这里不能简单地随机切，因为同一个候选可能出现在多个查询里。如果普通随机切分，同一个候选的向量可能既出现在训练折，又出现在验证折，模型相当于“见过答案”，验证分数会虚高。所以我们按正确候选的编号分组切分，保证训练折和验证折的候选集合没有交集。

```python
# 同一个候选可能出现在多个查询里，普通随机切分会把“同一个候选”
# 同时放进训练和验证，分数会虚高，所以按 label 分组切分
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
fit_idx, val_idx = next(splitter.split(train, groups=labels))

# 再检查一次：训练组和验证组的候选集合不能有交集
fit_candidates = set(labels[fit_idx])
val_candidates = set(labels[val_idx])
assert fit_candidates.isdisjoint(val_candidates)

print('fit:', len(fit_idx), 'val:', len(val_idx))
```

第二步是把文本变成向量。我们用的是 mean pooling，也就是把一句话里所有真实 token 的向量加起来求平均。为什么不直接用 `[CLS]`？因为小模型的 `[CLS]` 未必训练得很充分，而 mean pooling 对所有 token 一视同仁，通常更稳。求平均时要配合 attention mask，把 padding 位置的权重设成 0，否则补齐出来的无效 token 会把平均值拉偏。

```python
def mean_pooling(hidden, mask):
    # mask 是 [batch, seq]，扩一维后才能和 [batch, seq, dim] 逐元素相乘
    mask = mask.unsqueeze(-1)
    # 只对真实 token 求平均，padding 位置的权重是 0
    return (hidden * mask).sum(1) / mask.sum(1)

@torch.no_grad()  # 只做推理，不需要建计算图，省显存也更快
def embed(texts, all_layers):
    outs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        tok = tokenizer(batch, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors='pt')
        out = encoder(**tok, output_hidden_states=True)
        if all_layers:
            # 每一层都做 mean pooling，再拼起来，让查询拿到更丰富的信息
            vecs = [torch.nn.functional.normalize(
                mean_pooling(h, tok['attention_mask']), dim=1)
                for h in out.hidden_states]
            vec = torch.cat(vecs, dim=1)
        else:
            vec = mean_pooling(out.hidden_states[TARGET_LAYER],
                               tok['attention_mask'])
        outs.append(vec.numpy())
    return np.concatenate(outs)
```

在 baseline 里，查询侧和候选侧的处理并不完全一样。查询侧把所有隐藏层都做 mean pooling，然后拼成一个更长的向量；候选侧只取第 0 层，也就是最靠近输入的隐藏层。这样安排的理由是：查询是“主动去找”的一方，给它更丰富的表示，能让线性映射学到更多信息；候选数量通常比查询多，候选侧保持简单可以省计算。两边最后都会做 L2 归一化，保证点积就是余弦相似度。

第三步是训练一个 Ridge 回归模型。这里不是训练分类器，而是学习一个“翻译器”：输入查询向量，输出一个和正确候选向量尽量接近的向量。训练目标是最小化“预测向量”和“正确候选向量”之间的平方误差，再加上 L2 正则，防止权重过大。预测时，我们把查询向量送进 Ridge，得到一个新的向量，再和所有候选向量做点积，分数越高说明越相似。

```python
def predict_scores(model, q_vec, c_vec):
    # Ridge 输出仍在查询空间，先归一化，再和候选向量做点积
    pred = normalize(model.predict(q_vec))
    return pred @ c_vec.T  # 归一化后点积等价于余弦相似度

best_mrr, best_alpha, best_model = -1.0, None, None
for alpha in ALPHAS:
    model = Ridge(alpha=alpha)
    # 用训练折的查询向量去回归“正确候选”的向量
    model.fit(q_vec[fit_idx], c_vec[labels[fit_idx]])
    scores = predict_scores(model, q_vec[val_idx], c_vec)
    quality = measures(rankings(scores), labels[val_idx])
    print('alpha=', alpha, quality)
    # 比赛只看 MRR@5，所以按它选超参数，而不是按准确率
    if quality['mrr5'] > best_mrr:
        best_mrr, best_alpha, best_model = quality['mrr5'], alpha, model
```

`alpha` 是 Ridge 的正则强度。`alpha` 越大，模型越保守，越不容易过拟合；`alpha` 越小，模型越相信训练数据。我们在 `[0.001, 0.01, 0.1, 1, 10]` 这五个值里搜索，每个值都重新拟合一次，然后在验证集上算 MRR@5，最后选验证分数最高的那个。注意我们选的是 MRR@5，而不是准确率，因为比赛的排名只看 MRR@5，准确率高不代表 MRR@5 一定高。

### 2.4 模型测试与指标报告

我们然后做模型测试和指标报告。模型选好以后，不能只看训练时打印的数字，还要把验证集真的写成提交文件，再用官方 `score` 函数的逻辑算一遍。这样做有两个好处：一是能发现格式问题，二是能确认“离线算分”和“提交文件算分”是一致的。

```python
# 用官方 score 的逻辑在本地再算一遍，防止“离线分数”和“提交文件”不一致
def validate_saved(model, indices):
    rows = [train[i] for i in indices]
    scores = predict_scores(model, q_vec[indices], c_vec)
    with tempfile.TemporaryDirectory() as d:
        truth = Path(d) / 'truth.csv'
        pred = Path(d) / 'pred.csv'
        write_answers(truth, rows)       # 只写 query_id 和正确答案
        write_predictions(pred, rows, scores)
        return score(truth, pred, DATA / 'candidates.csv')

local_mrr = validate_saved(best_model, val_idx)
print('local MRR@5:', local_mrr)
```

这段代码会先造一个临时的正确答案文件，再按提交格式写出预测文件，最后调用和官方一致的 `score` 函数。如果本地算出来的 MRR@5 和训练时记录的 `best_mrr` 对不上，就说明排序方向、候选下标或者文件格式有问题，需要马上排查。材料里的 baseline 还用了一个 `assert` 来检查两者是否几乎相等，这个习惯很值得保留。

确认验证流程没问题后，我们用选好的 `alpha` 在全部训练数据上重新拟合一次。验证集的作用只是选超参数，既然超参数已经定了，就没有必要再留出 20% 的数据不用。训练数据越多，线性映射通常越稳。

```python
# 验证集只用来选 alpha，确定后把全部训练数据都用上再拟合一次
final_model = Ridge(alpha=best_alpha)
final_model.fit(q_vec[:len(train)], c_vec[labels])

# query 向量是按 train+test 的顺序一次性算的，所以 test 从 len(train) 开始
test_vec = q_vec[len(train):]
test_scores = predict_scores(final_model, test_vec, c_vec)
top5 = rankings(test_scores)

with open('submission.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(FIELDS)  # query_id, candidate_id_1 ... candidate_id_5
    for row, idx in zip(test, top5):
        # 写回字符串 ID，不能写下标，否则提交校验会报未知候选
        writer.writerow([row['query_id']] + [candidates[i]['candidate_id'] for i in idx])
```

最后写提交文件时，有两个细节。第一，`query_vectors` 是把 `train + test` 放在一起编码的，所以测试集向量要从 `len(train)` 这个位置开始取；如果顺序搞错，预测就会和查询对不上。第二，排序得到的是候选在 `candidates` 列表里的下标，写文件时必须换回 `candidate_id` 字符串，不能把 0、1、2 这样的下标直接写进去。

材料里没有记录最终榜单分数，所以我们不编造具体数值。我们能确认的是：验证集使用按候选分组的切分，指标按 MRR@5 计算，提交格式经过本地校验。如果要把这个 baseline 继续往上推，可以尝试几个方向：换更大的俄语预训练模型；对查询和候选做更细致的文本清洗；用对比学习或双塔模型代替 Ridge；增加 hard negative mining，让模型学会区分“看起来很像但不是答案”的候选；或者把候选侧的多层信息也利用起来。

## 三、总结

本文中我们首先通读了比赛材料，确认这是一个“给查询从候选池中排序”的文本检索任务；其次检查了 `train.csv`、`test.csv`、`candidates.csv` 的字段关系，把候选编号映射成整数标签，并统计了文本长度；接着按正确候选分组切分训练集和验证集，避免同一个候选同时出现在两边造成分数虚高；然后使用 `rubert-tiny2` 提取文本向量，查询侧拼接所有隐藏层，候选侧取第 0 层，两边都做 mean pooling 和 L2 归一化；最后用 Ridge 回归把查询向量映射到候选向量空间，在 `alpha` 网格上按 MRR@5 选超参数，再用全部训练数据重新拟合，排序取前五并写出提交文件。

我们完成的这个 baseline 结构清晰、计算量小，适合在 CPU 上快速跑通。它也有明显的局限：编码器没有微调，Ridge 只能学习线性映射，查询侧和候选侧的层选择带有较强的经验成分，候选数量很大时线性扫描也会变慢。后续如果继续优化，应该优先考虑更强的文本表示和更有针对性的负采样，而不是一味调 `alpha`。对入门者来说，这个比赛最有价值的地方是理解“检索任务怎么切分验证集、怎么定义指标、怎么把向量相似度变成提交文件”这一整条链路。
