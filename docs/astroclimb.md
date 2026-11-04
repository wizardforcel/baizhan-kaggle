# AstroCLIMB 科学图文配对判断

## 一、比赛简介

### 1.1 比赛背景

AstroCLIMB 是一个用来**评测多模态模型**的基准。所谓多模态，就是模型要同时处理文字和图片。这个比赛的任务不是"看图说话"，而是判断**两个对象之间是什么关系**。

每个样本由两个对象组成，每个对象可能是两种情况之一：

- 一段英文文本，通常是论文里的图注（caption）；
- 一张图片，是论文里的插图（figure），在 CSV 里以 base64 编码的 PNG 字符串形式存放。

我们要判断这两个对象属于四级关系中的哪一级：

| 标签 | 含义 |
| --- | --- |
| `same_figure` | 图注描述的正是这张图，即二者是同一张图的图文对 |
| `same_paper` | 来自同一篇论文 |
| `related_papers` | 来自相关但不同的论文 |
| `unrelated_papers` | 来自互不相关的论文 |

按对象的模态组合，样本又分成三类：`image+text`（图 + 文）、`text+text`（文 + 文）、`image+image`（图 + 图）。`same_figure` 只可能出现在 `image+text` 里——一段文字不可能"就是"另一段文字，也不可能"就是"一张图。

这个比赛的难点有三个。第一，数据是一个约 10 GB 的 CSV，两个对象列都是长字符串，不能一次性读进内存。第二，图和文分别由不同的编码器变成向量时，两个向量空间之间天然存在"模态鸿沟"（modality gap），直接算相似度会被这个鸿沟拖累。第三，科学插图信息密度高，坐标轴、刻度、图例都是关键线索，简单的 OCR 反而会引入噪声——材料里实测 OCR 的区分能力只有 0.51～0.53 的 AUC，基本等于瞎猜。

### 1.2 比赛目标

输入一行数据，包含 `id`、`obj_1`、`obj_2` 三列；两个对象各自是"文本"或"base64 图片"。输出是四个标签列，每行**恰好有一个 1**，表示这一对属于哪一级关系。

训练集 `train.csv` 额外带有四个标签列；测试集只有 `id` 和两个对象列。`sample_submission.csv` 的格式是 `id` 加四个标签列。

### 1.3 评价标准

比赛用 **macro-F1** 评分，而且是把所有测试行**混在一起**算一个总分（pooled），不是先算三个子集再取平均。macro-F1 的意思是：先对每个类别分别算 F1（精确率和召回率的调和平均），再对四个类别取平均。这样每个类别权重相同，不会因为某个类别样本多就主导分数。

需要特别注意的是，四个类别本身并不均衡，`same_figure` 是最稀有的一类。所以光看总体准确率会高估模型，必须按类别看 F1。

---

## 二、解题过程

我们的主线是：先把海量字符串解析成"文本 / 图片 + 哈希"的元数据，再把图片解码、缩放、去重，然后用多个预训练编码器（CLIP、SciNCL、DINOv2、Qwen3-VL）把图和文都变成向量，围绕"相似度 + 排名 + 词面重合"构造特征，最后按三种模态组合分别训练 LightGBM 和逻辑回归、做加权融合，并针对最容易混淆的边界加一个二分类"补丁"。

### 2.1 数据探索（EDA）

我们首先按块扫描训练集和测试集，把每一行的两个对象拆成"模态、长度、哈希"三组元数据。判断图片的办法很直接：比赛组织者把所有图片都编码成 PNG，PNG 的 base64 前缀固定是 `iVBORw0KGgo`，所以只要看字符串开头就能分辨。

```python
import hashlib, numpy as np, pandas as pd

LABELS = ['same_figure', 'same_paper', 'related_papers', 'unrelated_papers']
PNG = 'iVBORw0KGgo'          # PNG 的 base64 前缀，组织者统一用 PNG 编码

def scan(path, has_labels, chunksize=200):
    out = []
    for ch in pd.read_csv(path, chunksize=chunksize):
        d = pd.DataFrame({'id': ch['id'].values})
        for k, col in (('1', 'obj_1'), ('2', 'obj_2')):
            s = ch[col].fillna('')
            # 用前缀判断模态；同时记录原始字符串长度，作为信息量的粗略代理
            d['t' + k] = np.where(s.str.startswith(PNG), 'image', 'text')
            d['len' + k] = s.str.len().values
            # 用 md5 做对象级去重与对齐，后面 embedding 的行号全靠它
            d['h' + k] = [hashlib.md5(x.encode('utf-8', 'replace')).hexdigest() for x in s]
        if has_labels:
            d['label'] = ch[LABELS].values.argmax(1)   # 四个 0/1 列压成一个类别
        out.append(d)
    return pd.concat(out, ignore_index=True)

mtr = scan('train.csv', True)
mte = scan('test.csv', False)
```

接着我们把模态组合拼成一个字段，做"模态 × 标签"的交叉表，并检查训练集和测试集之间有没有共享的对象：

```python
# 把两种模态按字典序拼起来，例如 image+text
for d in (mtr, mte):
    d['combo'] = np.where(d.t1 < d.t2, d.t1 + '+' + d.t2, d.t2 + '+' + d.t1)

print(mtr['label'].value_counts())
print(pd.crosstab(mtr.combo, mtr.label, normalize='index').round(3) * 100)
print(pd.crosstab(mtr.combo, mtr.label))     # 看每个组合里各类别的绝对数量

# 训练集和测试集共享多少对象？共享多的话，随机切分会高估泛化
tro = pd.concat([mtr.h1, mtr.h2]); teo = pd.concat([mte.h1, mte.h2])
print('train 对象数:', tro.nunique(), 'test 对象数:', teo.nunique())
print('train ∩ test:', len(set(tro) & set(teo)))
```

这两张表给了我们三个关键结论。第一，`same_figure` 只在 `image+text` 组合里出现，所以三种组合必须**分别建模**，不能混在一起训。第二，`related_papers` 和 `unrelated_papers` 的边界是主要失分点，后面要专门处理。第三，同一个对象会在多行里重复出现，因此"这个对象在这篇论文的其他配对里表现如何"本身就是有用的相对信息，这一点在特征工程里会用上。

我们还顺便统计了 base64 字符串的长度：图片字符串比文本长得多，而且长短差异很大。这说明不能按固定长度截断，必须先解码再统一缩放。

### 2.2 数据清洗

我们其次处理两个对象列。图片的做法是：base64 解码 → 打开 → 转 RGB → 等比缩放到最长边 384 像素 → 存成 JPEG。这里选 384 而不是常见的 224，是有原因的：科学插图的坐标轴标签、刻度数字都很小，缩到 224 会直接糊掉，而这些恰恰是判断"是不是同一张图"的重要线索。

文本的做法是保守清洗：只做 Unicode 规范化（NFKC）、把 HTML 实体还原、把连续空白压成一个空格。**不删 LaTeX、不删单位和编号**，因为像 `SN 2011fe` 这样的标识符正是判断"是不是同一篇论文"的强信号。

```python
import base64, html, re, unicodedata
from io import BytesIO
from PIL import Image

Image.MAX_IMAGE_PIXELS = None      # 关闭超大图警告，避免解压时误报
SIDE = 384
_WS = re.compile(r'\s+')

def clean_text(s):
    # 只做规范化与空白压缩；LaTeX、单位、天体编号全部保留
    s = unicodedata.normalize('NFKC', html.unescape(s))
    return _WS.sub(' ', s).strip()

def handle(s, texts, seen, img_dir):
    """同一个字符串只处理一次：图片落盘，文本进字典，统一返回 md5 哈希"""
    h = hashlib.md5(s.encode('utf-8', 'replace')).hexdigest()
    if h in seen:
        return h
    seen.add(h)
    if s.startswith('iVBORw0KGgo'):
        img = Image.open(BytesIO(base64.b64decode(s)))
        img = img.convert('RGB')
        # thumbnail 保持长宽比，只缩不放
        img.thumbnail((SIDE, SIDE), Image.LANCZOS)
        img.save(f'{img_dir}/{h}.jpg', 'JPEG', quality=90)
    else:
        texts[h] = clean_text(s)
    return h
```

去重是这一步的隐形收益。训练集和测试集里同一个对象会反复出现，如果每个对象都解码一遍，10 GB 的字符串会被反复解开，时间全浪费在重复劳动上；用 md5 做键之后，每个对象只处理一次，后续所有特征都按哈希索引，行号对齐也不会出错。

### 2.3 模型训练与超参数调优

我们接着把每个对象变成向量。材料里用了四个编码空间：

- **CLIP ViT-L/14**：图和文都有，是最常用的跨模态基线；
- **SciNCL**：专门在科学文献上训练过的文本编码器，适合图注；
- **DINOv2**：自监督图像编码器，对科学插图的视觉结构比较敏感；
- **Qwen3-VL-Embedding-2B**：一个模型同时编码图和文，天生把两种模态放在同一个空间里。

前三个是"单模态专家"，第四个是"跨模态桥梁"。材料里的实测很能说明问题：只用 CLIP 时，`image+text` 的 macro-F1 卡在 **0.42**；换上 Qwen3-VL 之后升到 **0.60**。原因不是 Qwen3-VL 更强，而是它从训练方式上就没有模态鸿沟。

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def nrm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)

# 2B 模型在单张 T4/P100 上跑得动，8B 会超时
model = SentenceTransformer('Qwen/Qwen3-VL-Embedding-2B', device='cuda',
                            model_kwargs={'torch_dtype': 'float16'})

# 图片和文本走同一个 encode 接口，输出的向量天然在同一空间
img_vecs = model.encode([{'image': f'{IMGD}/{h}.jpg'} for h in IH],
                        batch_size=8, show_progress_bar=False)
txt_vecs = model.encode([{'text': s[:4000]} for s in TXT],
                        batch_size=16, show_progress_bar=False)

# 统一做 L2 归一化，之后点积就等于余弦相似度
QI, QT = nrm(img_vecs), nrm(txt_vecs)
np.save('emb_qvl_img.npy', QI)
np.save('emb_qvl_txt.npy', QT)
```

有了向量之后，我们构造特征。最直接的是余弦相似度，但只有相似度不够，因为不同对象的"可匹配程度"本来就不一样。我们加了两类修正：

第一类是 **CSLS**（跨域相似度局部缩放）：一个对象的相似度减去它在对方池子里最近 k 个邻居的平均相似度。直观理解是"扣掉这个对象自带的 popularity"，防止某些"万金油"对象对谁都像。

第二类是**排名特征**：这个对象在对方全部对象里排第几名。排名比原始相似度更稳定，因为它只关心相对位置。

此外还有 **2-hop 代理**：图和文不能直接比时（比如图 + 图），先把图映射到"最像的图注"，再在文本空间里比较。材料里把这个思路也用在了交叉编码器上，但实测代理图注的 AUC 只有 0.611/0.684，并不比原 pipeline 的 0.672/0.685 好，所以最后只作为辅助特征，没有当主力。

```python
def topk_mean(Q, P, k, bs=2048):
    """每个向量在对方池子里最相似的 k 个分数的均值，用于 CSLS"""
    out = np.zeros(len(Q), np.float32)
    for i in range(0, len(Q), bs):
        S = Q[i:i+bs] @ P.T
        out[i:i+bs] = np.partition(S, -k, axis=1)[:, -k:].mean(1)
    return out

def rank_in_pool(Q, P, iq, ip, bs=2048):
    """目标对象在对方池子里的名次（越大表示越不像）"""
    rk = np.zeros(len(iq), np.float32)
    for i in range(0, len(iq), bs):
        S = Q[iq[i:i+bs]] @ P.T
        s = S[np.arange(S.shape[0]), ip[i:i+bs]]
        rk[i:i+bs] = (S > s[:, None]).sum(1)
    return np.log1p(rk)      # log 压缩长尾

# a、b 是这一组合里“图索引”和“文索引”，由 pair_index 按 obj_1/obj_2 的模态整理得到
rad_i = topk_mean(QI, QT, 10)
c = (QI[a] * QT[b]).sum(1)          # 配对的余弦相似度
F_csls = 2 * c - rad_i[a] - topk_mean(QT, QI, 10)[b]   # CSLS 修正
F_rank = rank_in_pool(QI, QT, a, b)                     # 图→文的排名
```

把三种模态组合的特征拼好后，我们分别训练模型。主模型是 LightGBM 多分类：700 棵树、学习率 0.05、31 片叶子、列采样 0.3、行采样 0.8、类别权重 balanced。辅助模型是标准化后的逻辑回归。两者都用固定的 5 折分层交叉验证，折的划分在三个子集之间保持一致，这样后面的融合权重才不会因为折不同而产生偏差。

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

FOLDS = StratifiedKFold(5, shuffle=True, random_state=0)

def run_lgb(X, y, Xt, K, seeds=(0,)):
    oof = np.zeros((len(y), K)); te = np.zeros((len(Xt), K))
    for s in seeds:
        for tr, va in FOLDS.split(X, y):
            m = lgb.LGBMClassifier(
                objective='multiclass', num_class=K, n_estimators=700,
                learning_rate=0.05, num_leaves=31, colsample_bytree=0.3,
                subsample=0.8, subsample_freq=1, class_weight='balanced',
                min_child_samples=20, reg_lambda=1.0, max_bin=63,
                random_state=s, verbose=-1).fit(X[tr], y[tr])
            # OOF：验证折上的预测，用来挑融合权重
            oof[va] += m.predict_proba(X[va]) / len(seeds)
            te += m.predict_proba(Xt) / (5 * len(seeds))
    return oof, te

mf1 = lambda y, P, w=None: f1_score(y, (P * (1 if w is None else w)).argmax(1), average='macro')
```

融合分两步。第一步给三个模型（LightGBM 标量特征、LightGBM 全特征、逻辑回归）在单纯形上搜一组权重，使 OOF 的 macro-F1 最大。第二步做**类别乘子**：给每个类别的概率乘一个系数，再把系数往大或往小调，专门补偿稀有类别被压扁的问题。乘子只调概率的相对大小，不改变排序逻辑。

最后是最容易被忽略、但收益很实在的一步：**级联二分类**。多分类模型要同时兼顾三条边界，最容易在 `related_papers` 和 `unrelated_papers` 之间做妥协。我们单独在这两类样本上训一个二分类器，只负责"这两个到底是相关还是不相关"，再把多分类的概率和这个二分类的概率按权重混合、用阈值裁决。材料里记录这一步在 `image+image` 上带来约 +0.007 的提升，虽然不大，但方向稳定。

### 2.4 模型测试与指标报告

我们最后把所有子集的结果汇总。因为排行榜是 pooled macro-F1，我们也在 OOF 上算 pooled 版本：把三个子集的真实标签和预测拼成一个长列表，再统一算一次 F1，而不是对三个子集分数取加权平均。

```python
from sklearn.metrics import f1_score, classification_report

# 每个子集单独训练后，把 OOF 预测拼起来。
# y_all / pred_all / D_all 是三个子集训练结束时按组合名存下的真值、预测和行表
pool_y, pool_p = [], []
for cb in ['image+text', 'text+text', 'image+image']:
    # y 是真实标签名，pred 是融合+乘子+级联之后的预测标签名
    pool_y += list(y_all[cb])
    pool_p += list(pred_all[cb])

pooled = f1_score(pool_y, pool_p, average='macro')   # 这才是和 LB 对齐的算法
print('POOLED OOF macro-F1:', round(pooled, 4))
print(classification_report(pool_y, pool_p,
                            labels=['same_figure', 'same_paper',
                                    'related_papers', 'unrelated_papers'],
                            digits=3))
```

材料里给出的对照数字是：只用单模态编码器的初版方案（plan_c）在三个子集上的 F1 约为 0.42 / 0.63 / 0.47，排行榜约 0.57；加入 Qwen3-VL 联合空间之后的 plan_d_a，三个子集分别达到 **0.6017 / 0.6725 / 0.5135**，pooled OOF 为 **0.6293**。其中 `image+text` 从 0.42 提到 0.60，提升最明显，正好对应"模态鸿沟被填平"这个判断。`image+image` 仍然最弱，只有 0.5135，说明两张科学插图之间可比的信息本来就少。

提交前我们做了两条硬检查：每行四个标签的和必须等于 1；`id` 必须和测试集一一对应且不重复。

```python
LABELS = ['same_figure', 'same_paper', 'related_papers', 'unrelated_papers']
sub = pd.DataFrame({'id': mte.id.values})
for c in LABELS:
    sub[c] = 0

# 每个子集把预测为某类的行，对应标签列置 1
for cb in ['image+text', 'text+text', 'image+image']:
    pos = pd.Index(sub.id).get_indexer(dt_all[cb].id.values)
    for j, c in enumerate(classes_all[cb]):
        sub.loc[pos[pred_all[cb] == j], c] = 1

# 这两条断言能挡住绝大多数“看起来正常”的提交错误
assert len(sub) == len(mte) and sub.id.is_unique
assert (sub[LABELS].sum(1) == 1).all()
sub[['id'] + LABELS].to_csv('submission.csv', index=False)
print(sub[LABELS].sum().to_dict())     # 看预测出的类别分布是否合理
```

材料里还记录了一个反例，很值得写下来：我们试过用 `same_figure` 的图文对学一个"图→文"的线性映射（Procrustes 桥），把 CLIP 的相似度范围从 0.0565 拉宽到 0.1900，看起来是 3.4 倍，但"效应量"几乎没变（1.28 → 1.27），macro-F1 反而从 0.4254 掉到 0.4215。原因是树模型对单调缩放不敏感，多出来的 258 列只是噪声。**把分布的尺度拉宽，不等于把信号变强**——这是这个比赛给我们上的重要一课。

---

## 三、总结

本文中我们完成了一个科学图文配对的关系分类系统。我们首先按块扫描 10 GB 级的 CSV，用 PNG 前缀区分图片与文本、用 md5 做对象去重，并通过交叉表确认三种模态组合必须分开建模；其次把图片解码、等比缩放到 384 像素、文本只做保守规范化；接着用 CLIP、SciNCL、DINOv2 和 Qwen3-VL 四个编码空间生成向量，配合 CSLS、排名、2-hop 代理和词面特征训练 LightGBM 与逻辑回归，做权重融合、类别乘子和 related/unrelated 级联；最后按 pooled macro-F1 汇总 OOF，三个子集达到 0.6017 / 0.6725 / 0.5135，pooled 为 0.6293，并逐行校验提交格式。

这个比赛最核心的收获是**模态鸿沟要用联合训练的编码器解决，而不是靠后处理硬拉**。把 CLIP 的数值范围拉宽 3.4 倍并没有带来任何收益，而换成一个图文联合训练的 Qwen3-VL，`image+text` 直接从 0.42 跳到 0.60。另一个收获是**评估方式要和排行榜一致**：macro-F1 必须 pooled 计算，否则子集大小不同会让本地分数和线上分数对不上。
