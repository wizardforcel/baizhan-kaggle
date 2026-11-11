# ArXiv 标题生成（Title Generation）

## 一、比赛简介

### 1.1 比赛背景

arXiv 是一个收集学术论文预印本的网站，物理、数学、计算机等领域的 researchers 会把自己的论文先放到上面。每篇论文都有两样东西：一段**摘要**（abstract），用几百个词概括全文做了什么；一个**标题**（title），用十几个词把论文的核心贡献浓缩成一句话。

这个比赛做的事情很直观：**只给机器看摘要，让它自己写出论文标题**。对人类来说，读完摘要再起一个标题不算太难；对机器来说，这是一项典型的「文本到文本」任务——输入一段话，输出另一段话。它和机器翻译、自动摘要属于同一类问题，统称为 **seq2seq（序列到序列）**。

这个任务有几个难点。第一，标题很短，信息密度却很高，模型必须在几十个词里抓住论文最核心的贡献，不能把摘要里所有内容都复述一遍。第二，标题和摘要的用词往往不完全一样，摘要里说「我们提出了一种方法」，标题可能写成「一种新的框架」，所以逐字照抄摘要效果不好。第三，评价标准看的是词和词组的重叠程度，模型既要写对关键词，又要控制长度。

还有一个这个比赛特有的情况：**数据划分存在泄漏**。材料里的 EDA 明确指出，测试集里有相当一部分摘要原样出现在训练集中，比例大约是 **43%**；另一份 baseline 则说，有 **431 篇**测试论文同时出现在训练集里。这意味着只要把训练集里的「摘要—标题」对背下来，就能在测试集上拿到一个不低的分数。这个漏洞让「抄答案」成为一个非常强的 baseline，也提醒我们：看排行榜分数时，要分清哪些分是模型真本事，哪些分是数据泄漏送的。

### 1.2 比赛目标

训练集 `train.csv` 有两列：

- `abstract`：论文摘要，一段英文文本；
- `title`：论文标题，我们要预测的目标。

测试集 `test.csv` 只有 `abstract` 一列（外加用于对齐的标识），没有标题。我们要为每一条测试摘要生成一个标题，并按比赛要求提交。

提交文件不是直接交一段标题文本，而是要把标题转换成由 0 和 1 组成的向量，格式是两列 `Id,Predict`。这一点和很多文本比赛不一样，下一节会解释为什么要这样。

### 1.3 评价标准

比赛使用 **F1 分数**，但这里的 F1 不是按「整句话对不对」来算，而是按**词和词组的重叠**来算。具体做法是：

1. 把预测标题和真实标题都去掉标点、转成小写、按空格切成单词；
2. 除了单个单词（unigram），还把所有相邻的两个词拼成一个词组（bigram）、三个词拼成一个词组（trigram），拼的时候用下划线连接；
3. 拿这些词和词组去查一份固定的词表，出现过就记 1，没出现就记 0，于是每个标题变成一个 0/1 向量；
4. 比较预测向量和真实向量，计算 F1。

F1 的公式是精确率和召回率的调和平均：

```
F1 = 2 × 精确率 × 召回率 / (精确率 + 召回率)
```

精确率是「我们写出来的词里有多少是对的」，召回率是「标准答案里的词我们写出了多少」。F1 把两者兼顾起来。分数范围是 0 到 1，**越高越好**，满分 1 表示预测标题的词、二元词组、三元词组和标准答案完全一致。

这个指标有两个特点，后面做模型时会反复用到：第一，它**不要求句子通顺**，只要关键词和关键短语命中就能得分；第二，它**很看重词组**，两个词单独出现不算本事，连在一起出现才说明模型学到了搭配。因此我们在生成标题时，要尽量使用训练集里常见的学术表达，而不是自己造新词。

---

## 二、解题过程

整场比赛我们走了「先摸数据、再做清洗、然后从小模型到大模型、最后按官方指标编码提交」这条路线。我们首先通过 EDA 发现测试集和训练集大量重叠，其次利用这个重叠搭了一个很强的「抄答案」baseline，接着训练了从零开始的 LSTM 序列到序列模型，然后换成预训练的 T5-small 做生成，最后把生成的标题编码成官方要求的 0/1 向量并检查提交格式。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，看看数据规模，以及两份数据之间有没有重叠。重叠检查是这次 EDA 里最重要的一步，因为它直接决定了后面 baseline 能有多强。

```python
import pandas as pd

# 读入训练集和测试集；测试集只有 abstract，没有 title
train_df = pd.read_csv('../input/title-generation/train.csv')
test_df = pd.read_csv('../input/title-generation/test.csv')
print('train:', train_df.shape, 'test:', test_df.shape)

# 训练集内部先去重，避免同一篇论文被模型重复学习
train_df.drop_duplicates(inplace=True)

# 关键检查：测试集的摘要，有多少能在训练集里原样找到
test_abstracts = set(test_df['abstract'])
train_abstracts = set(train_df['abstract'])
overlap = test_abstracts & train_abstracts

# 用重叠条数除以测试集总数，得到泄漏比例
print('overlap ratio:', len(overlap) / len(test_df))
```

跑完我们发现，测试集中大约 **43%** 的摘要和训练集完全一样。这个数字说明比赛的数据划分不是随机切的，同一篇论文可能一部分进了训练、一部分进了测试。对参赛者来说，这是天上掉下来的分数：这 43% 的样本不需要模型生成，直接从训练集里把对应标题取出来就行。

我们接着看标题本身有多长。标题长度决定了模型生成时应该写多少个词，也决定了我们给模型设置的最大长度。

```python
# 统计每条标题包含多少个单词
title_lengths = train_df['title'].str.split().apply(len)

# 看最小值、平均值、最大值和分位数
print(title_lengths.describe())

# 平均长度用来判断生成标题的合理长度
print('mean title length:', title_lengths.mean())

# 顺便看一眼最常见的标题单词，了解学术标题的用词习惯
from collections import Counter
words = [w for title in train_df['title'] for w in title.split()]
print(Counter(words).most_common(20))
```

从统计结果看，论文标题普遍不长，大多数集中在十几个词。这个观察有两个用处：一是后面给 T5 设置 `max_tgt` 时不用设得太大，32 个 token 足够覆盖绝大多数标题；二是做「笨 baseline」时，可以用平均长度来决定填多少个高频词，让预测长度接近真实长度，从而在 F1 上占一点便宜。

### 2.2 数据清洗

我们其次做数据清洗。这份数据的清洗重点不是去停用词，而是处理重复样本和错误样本。

第一件事是训练集去重。测试集里也有重复，但我们不能删测试样本，因为比赛要求对每一条测试数据都给出预测，删掉就会少交行。所以我们只清洗训练集。

第二件事是处理「摘要和标题对不上」的样本。材料里的 EDA 发现，重叠部分里有一条样本的摘要和标题明显错位，还有一条样本的标题存在歧义。这两条会误导模型，我们直接把它们从候选池里剔除。

```python
# 训练集内部去重；测试集保持原样，因为每条都要提交
train_df.drop_duplicates(inplace=True)

# 找出训练集和测试集共享的摘要，这些是后面可以「抄答案」的样本
shared = set(train_df['abstract']) & set(test_df['abstract'])
candidate_df = train_df[train_df['abstract'].isin(shared)].copy()

# 摘要出现次数最多的那一条，往往就是错配样本：多个标题共用一个摘要
bugged_abstract = candidate_df['abstract'].mode()[0]
candidate_df = candidate_df[candidate_df['abstract'] != bugged_abstract]

# 再看剩下数据里出现次数最多的摘要，材料里认为它标题有歧义，也去掉
uncertain_abstract = candidate_df['abstract'].mode()[0]
candidate_df = candidate_df[candidate_df['abstract'] != uncertain_abstract].reset_index(drop=True)

print('clean candidate rows:', len(candidate_df))
```

清洗完之后，我们得到一个「干净的候选池」，里面每一条都是摘要和标题能对上的论文。后面做重复样本匹配时，我们就从这个池子里取标题，避免把错配的标题抄进提交文件。

第三件事是**统一大小写再做匹配**。直接用原始字符串比对，可能因为首字母大小写不同而漏掉重复。所以我们把训练集和测试集的摘要都转成小写，再建立「测试样本编号 → 训练样本编号」的映射。

```python
# 统一转小写，避免大小写差异导致重复样本匹配失败
train_abs = train_df['abstract'].str.lower()
test_abs = test_df['abstract'].str.lower()

duplicate_map = {}

# 对每条测试摘要，在训练集里找完全相同的摘要
for test_id, abstract in enumerate(test_abs):
    hit = (abstract == train_abs)          # 逐条比较，返回布尔序列
    if hit.sum() > 0:
        # idxmax 取出第一个匹配上的训练样本编号
        duplicate_map[test_id] = hit.idxmax()

print('matched test rows:', len(duplicate_map))
```

这段循环看起来有点笨，但逻辑很清楚：测试集有多少条，就扫多少遍训练集。数据量不大时完全够用。匹配上之后，这 43% 的测试样本就可以直接抄训练集的标题，剩下的 57% 才需要真正的生成模型。

### 2.3 模型训练与超参数调优

我们接着进入模型部分。为了看清楚「生成模型到底有没有用」，我们先搭了一个非常简单的规则 baseline，再训练 LSTM，最后上 T5-small。这样每一步的提升都能对上原因，而不是一上来就丢一个大模型进去。

**第一步：规则 baseline。** 思路只有两条：如果测试摘要在训练集里出现过，就直接抄它的标题；如果没出现过，就取摘要的第一句话，最多保留 40 个词，当作标题。

```python
def first_sentence(text, max_words=40):
    """取摘要第一句，并截断到 max_words 个词，防止标题过长"""
    first = text.strip().split('.')[0]        # 按句号切，取第一句
    return ' '.join(first.split()[:max_words])

# 默认预测：摘要第一句
predicted_titles = test_df['abstract'].apply(first_sentence)

# 能匹配上的样本，用训练集的真实标题覆盖掉默认预测
for test_id, train_id in duplicate_map.items():
    predicted_titles.iloc[test_id] = train_df.iloc[train_id]['title']

# 整理成两列，方便后面的编码脚本读取
submission_df = pd.DataFrame({
    'abstract': test_df['abstract'].values,
    'title': predicted_titles.values
})
submission_df.to_csv('predicted_titles.csv', index=False)
```

这个 baseline 没有任何学习，但在这次比赛里非常有效，因为 43% 的测试样本可以做到几乎完全正确。它给后面所有模型设了一个「及格线」：如果 LSTM 或 T5 连这个线都过不了，说明模型还不如抄答案。材料里另一份 baseline 也用了同样的思路，只是把「第一句」换成了「按平均标题长度填最高频的词」，本质都是利用词重叠指标做保守预测。

**第二步：LSTM 序列到序列模型。** 我们用一个双向 GRU 做编码器，把摘要读成一个向量序列；再用一个带注意力机制的 GRU 做解码器，一个词一个词地写出标题。训练时使用 **teacher forcing**：解码器每一步有 50% 的概率看到真实的上一个词，而不是自己上一步的预测，这样训练更稳定。

```python
import torch
import torch.nn as nn
from torchtext.data import Field, BucketIterator

# 用 spacy 做英文分词；lower=True 让词表更紧凑
TEXT = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>',
             include_lengths=True, lower=True)

# 只读取 abstract 和 title 两列，分别作为源端和目标端
fields = [('src', TEXT), ('trg', TEXT)]
dataset = data.TabularDataset(
    path='../input/title-generation/train.csv',
    format='csv', skip_header=True, fields=fields)

# 98% 训练、1% 验证、1% 测试；数据量大时验证集不用切太多
train_data, valid_data, test_data = dataset.split(split_ratio=[0.98, 0.01, 0.01])

# 出现次数少于 7 的词统统当作 <unk>，控制词表大小
TEXT.build_vocab(train_data, min_freq=7)

BATCH_SIZE = 32
train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,              # 按摘要长度排序，减少 padding 浪费
    sort_key=lambda x: len(x.src))
```

模型结构上，编码器是双向 GRU，能把每个词左边和右边的上下文都看一遍；解码器每一步先算注意力权重，决定「现在该看摘要里的哪几个词」，再结合上一个生成的词，预测下一个词。训练超参数是 7 个 epoch、梯度裁剪阈值 1、batch size 32。材料里提到，为了加快实验，作者先只用 3 万条样本训练；如果想复现排行榜成绩，要在全部约 13.5 万条样本上训练 10 个 epoch。这个细节很重要：**小数据跑通流程，全量数据冲分数**，是比赛中很常见的做法。

**第三步：T5-small。** LSTM 是从零开始学的，词表和语言规律都要从这场比赛的数据里现学。T5 是一个已经在海量英文文本上预训练过的「文本到文本」模型，我们只需要在它的基础上做微调（fine-tune），就能借用它已经学会的英语表达能力。

```python
from transformers import T5TokenizerFast, T5ForConditionalGeneration

tokenizer = T5TokenizerFast.from_pretrained('t5-small')

def clean(text):
    """去掉换行、把连续空格压成一个，避免格式噪声"""
    return ' '.join(str(text).replace('\n', ' ').split())

class ArxivDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_src=256, max_tgt=32, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_src, self.max_tgt = max_src, max_tgt
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # T5 习惯在输入前加任务前缀，这里把「写标题」当成摘要任务
        src = self.tok('summarize: ' + clean(self.df.loc[idx, 'abstract']),
                       max_length=self.max_src, truncation=True,
                       padding='max_length', return_tensors='pt')
        if self.is_test:
            # 测试集没有标题，用 -100 占位；-100 在损失里会被忽略
            labels = torch.full((self.max_tgt,), -100, dtype=torch.long)
        else:
            tgt = self.tok(clean(self.df.loc[idx, 'title']),
                           max_length=self.max_tgt, truncation=True,
                           padding='max_length', return_tensors='pt')
            labels = tgt.input_ids.squeeze(0)
        return {'input_ids': src.input_ids.squeeze(0),
                'attention_mask': src.attention_mask.squeeze(0),
                'labels': labels}
```

训练配置是 3 个 epoch、batch size 32、学习率 3e-4，优化器用 AdamW，学习率预热占全部步数的 10%，之后线性衰减。我们还开了混合精度（AMP），让显存占用和训练时间都降下来。这里的学习率比从头训练 LSTM 时更大，是因为 T5 只需要轻微调整，3e-4 在微调里属于常用范围；预热则是为了避免一开始步子太大，把预训练学到的参数冲坏。

关于超参数调优，我们没有做大规模的网格搜索。原因很现实：每换一组参数都要重新跑一遍 Transformer，成本太高。我们的策略是**先固定一组在文本生成任务里比较通用的配置，把流程跑通，再只调最影响结果的几个参数**。最关键的参数是 `max_tgt`：设太小会截断标题，设太大会让模型生成多余内容。根据 EDA 看到的标题长度分布，32 个 token 已经能覆盖绝大多数标题，所以最终定为 32。生成阶段再用 beam search 的宽度 4 做搜索，宽度越大结果通常越稳，但速度越慢。

### 2.4 模型测试与指标报告

我们然后做生成和评估。T5 训练好之后，切换到评估模式，对测试集逐批生成标题。生成时不用普通的「每步选概率最大的词」，而是用 **beam search（束搜索）**：每一步同时保留 4 条候选路径，最后挑整体概率最高的一条。这样能避免模型一条路走到黑，生成的标题更通顺。

```python
model.eval()
pred_titles = []

for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():                     # 只推理，不计算梯度，省显存
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,                   # 和训练时的标题长度保持一致
            num_beams=4,                     # 同时保留 4 条候选，提升生成质量
            early_stopping=True              # 所有候选都生成结束后就停
        )
    # 把 token 编号解码回普通文本，跳过 <pad> 等特殊符号
    pred_titles.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# 保存成「摘要 + 标题」两列，供官方编码脚本使用
result = test_df.copy()
result['title'] = pred_titles
result[['abstract', 'title']].to_csv('predicted_titles.csv', index=False)
```

生成出来的标题还需要转换成官方要求的 0/1 向量。这一步必须和评价标准完全对齐，否则模型写得再好，提交格式不对也拿不到分。具体做法是：去标点、转小写、切词，然后补充二元词组和三元词组，最后按词表逐位判断「这个词有没有出现」。

```python
import string
from nltk.util import ngrams

def encode_title(title, vocab):
    """把一条标题编码成 0/1 向量，和官方评测方式保持一致"""
    # 去掉所有标点，转小写，再按空格切词
    words = title.translate(str.maketrans('', '', string.punctuation)).lower().split()

    # 加入相邻两词、相邻三词组成的词组，用下划线连接
    bigrams = ['_'.join(g) for g in ngrams(words, 2)]
    trigrams = ['_'.join(g) for g in ngrams(words, 3)]
    words.extend(bigrams)
    words.extend(trigrams)

    # 词表里有的位置记 1，没有的记 0
    word_set = set(words)
    return [1 if w in word_set else 0 for w in vocab]

# 每条测试样本有自己对应的词表，逐条编码后写成 Id,Predict
with open('submission.csv', 'w') as f:
    f.write('Id,Predict\n')
    output_id = 0
    for row_idx, row in result.iterrows():
        bits = encode_title(row['title'], vocs[row_idx])
        for bit in bits:
            f.write(f'{output_id},{int(bit)}\n')
            output_id += 1
```

这里有一个容易踩的坑：`vocs.pkl` 里存的不是一份全局词表，而是**每条测试样本对应一份词表**。所以编码时必须用 `vocs[row_idx]`，不能所有样本共用一份。材料里三份 notebook 都反复强调这一点，说明它是提交环节最容易出错的地方。

关于最终成绩，材料里的 notebook 只给出了完整的训练、生成和编码代码，没有贴出可以直接引用的排行榜分数。因此我们不在这里编造一个具体数字。能够确认的是三件事：第一，规则 baseline 依靠 43% 的重复样本，已经能拿到一个相当高的下限；第二，LSTM 版本在材料中用宏平均 BLEU 来观察生成质量，但同样没有记录具体数值；第三，T5-small 借助预训练，在同样数据上通常能生成更通顺、更接近学术标题风格的句子。我们最终选择 T5 作为主线方案，就是因为它把「语言通顺」和「关键词命中」平衡得更好。

我们最后还做了一次提交文件自检：确认行数等于所有测试样本的向量长度之和，`Id` 从 0 开始连续递增，`Predict` 只有 0 和 1，列名严格是 `Id,Predict`。这些检查听起来琐碎，但文本生成比赛里因为编码错位、词表用错而白白丢分的情况并不少见。

---

## 三、总结

本文中我们完成了 ArXiv 标题生成比赛的完整流程。我们首先通过 EDA 发现测试集有约 43% 的摘要原样出现在训练集里，这个数据泄漏让「抄答案」成为一个非常强的 baseline；其次我们做了针对性的数据清洗，去掉训练集重复样本、修正摘要与标题错配的样本，并统一小写建立测试到训练的匹配关系；接着我们先搭了「重复样本直接抄、其余取摘要第一句」的规则 baseline，再训练了带注意力机制的双向 GRU 序列到序列模型，最后微调预训练的 T5-small，用束搜索生成标题；在提交阶段，我们严格按照官方指标，把标题拆成单词、二元词组和三元词组，编码成 0/1 向量。

这个比赛给我们最大的启发有两点。第一，**做 EDA 一定要检查数据泄漏**。如果一上来就闷头训练大模型，很可能辛苦半天还不如直接抄训练集，先摸清数据划分才能知道分数从哪里来。第二，**评价指标决定了模型该往哪个方向优化**。这里的 F1 只看词和词组重叠，不看句子是否优美，所以生成标题时要优先保证关键词和常见搭配命中，而不是追求华丽的表达。把指标翻译成代码、再让提交格式和指标严格对齐，往往比换一个更大的模型更能提分。
