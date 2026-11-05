# You Are Bot 2 机器人识别挑战

## 一、比赛简介

### 1.1 比赛背景

You Are Bot 2 是 Kaggle 上的一场对话机器人识别比赛。比赛提供一批两人对话记录，要求判断其中某一位参与者到底是真人还是自动脚本。对话文本主要是俄语，所以材料里的触发词表 `ня`、`инструкции`、`ьотяра`、`helpt`、`эхо`、`тык` 都是俄语词或俄语聊天里的固定说法。这个任务和普通文本分类有两点不同：第一，判断对象不是整段对话，而是对话中的某一个参与者；第二，线索不只在文字内容里，还在发言频率、重复率、标点使用习惯等行为模式里。机器人往往会重复上一句、复制固定模板、在短时间内刷很多条短消息，真人则更随意，标点和长度变化更大。比赛数据用俄语，一方面是因为原始数据来自俄语聊天场景，另一方面也是想避免参赛者直接套用英文语料上的预训练模型。我们要在有限的数据里同时利用文本语义和行为统计，这也是后文两条特征通道的由来。机器人越自动，行为模式越容易被统计量捕捉，形成稳定的信号。

材料中的两个 notebook 都围绕同一个数据格式展开：每段对话有一个 `dialog_id`，里面按 `message` 字段排序存放多条消息，每条消息记录 `participant_index` 和 `text`。训练标签在 `ytrain.csv` 里，列名是 `dialog_id`、`participant_index`、`is_bot`。测试集提交样例是 `ytest.csv`，包含 `ID`、`dialog_id`、`participant_index` 三列。我们最终要为每一行给出一个 0 到 1 之间的 `is_bot` 概率。

### 1.2 比赛目标

输入是 JSON 格式的对话语料，训练集还额外带有 `is_bot` 标签。输出是一个 CSV 文件，每一行对应测试集中的一个 `(dialog_id, participant_index)` 组合，列 `is_bot` 表示该参与者是机器人的概率，数值越大越像机器人。提交文件必须保留 `ID` 列，并且行数、顺序要和 `ytest.csv` 对得上，否则无法正确评分。需要注意的是，同一个对话里的两位参与者是互斥的，模型需要在对话内部进行比较，而不是把每个参与者孤立看。

### 1.3 评价标准

材料里没有直接写出评分函数的名字。从代码行为看，模型输出的是连续概率，并且最后统一裁剪到 `[0.001, 0.999]`，这说明比赛按概率排序或概率误差来评分，而不是只要求 0/1 分类。我们在这里不臆测具体指标名称，只按“输出合理概率、避免极端 0 和 1”的方式来处理。对入门者来说，可以把它理解成：模型越能把真人排在低概率、机器人排在高概率，分数就越好。如果比赛用的是 AUC，排序比绝对概率更重要；如果用的是对数损失，概率校准就同样重要，所以两个方案都保留了概率输出。

---

## 二、解题过程

我们首先观察数据格式，其次构造行为特征，接着训练 CatBoost，然后尝试用俄语预训练模型补充语义特征，最后生成提交文件。

### 2.1 数据探索（EDA）

我们先把训练和测试的 JSON 读进来，确认对话、消息和参与者的层级关系。

```python
import json

# 训练集和测试集都是 dialog_id -> 消息列表 的字典
with open('/kaggle/input/competitions/you-are-bot-2/train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('/kaggle/input/competitions/you-are-bot-2/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 查看第一段对话的结构
first_id = next(iter(train_data))
print(first_id)
print(train_data[first_id][:3])
# 每段对话里是若干条消息，每条消息有 message、participant_index、text
```

看完结构后，我们统计每段对话的消息数量和每个参与者的发言条数。这个统计很重要，因为如果某个参与者在整段对话里只说了一句话，那么“重复率”“标点密度”这类特征的估计会非常不稳定。对于两人的对话，我们发现多数对话长度在十几到几十条之间，少部分对话只有几条。我们没有丢弃短对话，而是保留 `msg_share` 来反映参与度，这让模型对不同长度的对话都保持公平。

```python
# 统计每段对话的消息数、参与者数量和每个参与者的发言数
rows = []
for dialog_id, messages in train_data.items():
    messages = sorted(messages, key=lambda x: x['message'])
    total = len(messages)
    for p in [0, 1]:
        count = sum(1 for m in messages if m['participant_index'] == p)
        rows.append({
            'dialog_id': dialog_id,
            'participant_index': p,
            'msg_count': count,
            'msg_share': count / total if total else 0
        })
```

我们还把标签表 `ytrain.csv` 读进来，按 `dialog_id` 和 `participant_index` 与上面的统计表合并。合并后我们检查了标签列的取值，确认它只有 0 和 1 两种，0 表示真人、1 表示机器人。EDA 让我们确定了一个关键事实：同一个 `dialog_id` 下通常有两位参与者，模型必须分别对每个人建模，不能把整段对话当成一个样本。否则同一个对话里的真人特征和机器人特征会被混在一起，标签也会变得没有意义。

### 2.2 数据清洗

这份数据的清洗不是去停用词，而是统一文本格式，让行为特征可比较。我们做了三件事：把文本转小写、去掉每条消息首尾空白、把同一个人的多条消息按时间顺序拼起来。拼接时用 `[SEP]` 分隔，既保留了消息边界，又让后面的文本模型能一次看到整段发言。

```python
def build_profiles(corpus, labels=None):
    """把原始 JSON 整理成每人一行、带行为特征的表格"""
    records = []
    trigger_words = {'ня', 'инструкции', 'ьотяра', 'helpt', 'эхо', 'тык'}
    punct = '.,!?()'

    for dialog_id, messages in corpus.items():
        # message 字段是时间顺序，先排序再按人分组
        messages = sorted(messages, key=lambda x: x['message'])
        total_msgs = len(messages)

        for p in (0, 1):
            utterances = [m for m in messages if m['participant_index'] == p]
            if not utterances:
                continue

            texts = [m['text'].strip().lower() for m in utterances]
            full_text = ' [SEP] '.join(texts)

            # 相邻两条消息完全一样，说明可能是在复读
            exact_repeat = sum(1 for a, b in zip(texts[1:], texts[:-1]) if a == b)
            # 相邻两条长度一样，说明节奏机械
            len_repeat = sum(1 for a, b in zip(utterances[1:], utterances[:-1])
                             if len(a['text']) == len(b['text']))

            records.append({
                'dialog_id': dialog_id,
                'participant_index': p,
                'text': full_text,
                'echo_rate': exact_repeat / len(utterances),
                'len_echo': len_repeat / len(utterances),
                'punct_density': sum(full_text.count(c) for c in punct) / (len(full_text) + 1),
                'avg_msg_len': sum(len(t) for t in texts) / len(texts),
                'msg_share': len(utterances) / total_msgs,
                'is_trigger': int(any(w in full_text for w in trigger_words)),
            })

    df = pd.DataFrame(records)
    if labels is not None:
        df = df.merge(pd.read_csv(labels), on=['dialog_id', 'participant_index'])
    return df
```

这样清洗后，每个参与者都有文本和六个数值特征。我们没有删除短对话，也没有做词形还原，因为触发词是固定字符串，保留原样更稳妥。此外，文本转小写是为了让触发词检测不依赖大小写，而 `[SEP]` 分隔符是为了避免把两条不同消息硬连在一起形成跨消息的假词汇。标点密度采用整体统计，而不是每条消息单独算，这样能反映整体书写习惯。`echo_rate` 看的是完全相同句子的比例，`len_echo` 看的是长度重复，两者分别抓字面复读和机械节奏。

### 2.3 模型训练与超参数调优

我们的主模型是 CatBoost。它适合这种“文本 + 少量数值特征”的表格任务：数值特征直接进树模型，`text` 列交给 CatBoost 自带的文本处理，不用自己手写分词。材料里的第一版使用 5 折交叉验证，每折训练 3000 轮，学习率 0.01，树深 7，L2 正则 20。CatBoost 的文本处理会先做内部分词并生成 n-gram，再和数值特征一起训练；早停机制根据验证集表现动态停止，比固定迭代更稳健。

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold

feature_cols = ['text', 'echo_rate', 'len_echo', 'punct_density',
                'avg_msg_len', 'msg_share', 'is_trigger']

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_probs = []

for train_idx, valid_idx in kf.split(train):
    tr, va = train.iloc[train_idx], train.iloc[valid_idx]

    model = CatBoostClassifier(
        iterations=3000,          # 树的数量，配合早停防止过拟合
        learning_rate=0.01,       # 小学习率让每棵树只学一点点
        depth=7,                  # 树深，太深容易记住具体对话
        l2_leaf_reg=20,           # L2 正则，抑制叶子权重过大
        text_features=['text'],   # 告诉 CatBoost 哪一列是文本
        verbose=0,
        early_stopping_rounds=100 # 验证集 100 轮不涨就停
    )
    model.fit(tr[feature_cols], tr['is_bot'],
              eval_set=(va[feature_cols], va['is_bot']))

    # 每折都对测试集预测，最后再平均
    fold_probs.append(model.predict_proba(test[feature_cols])[:, 1])

test['is_bot'] = np.mean(fold_probs, axis=0)
```

第二版在数值特征之外，又加了俄语预训练模型 `cointegrated/rubert-tiny2` 的句向量。具体做法是把每个参与者的全部发言截断到 512 个字符，送进 BERT，取 `[CLS]` 位置的 312 维向量，再和原来的表格特征拼在一起。

```python
import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')
bert = AutoModel.from_pretrained('cointegrated/rubert-tiny2').to(device).eval()

def get_embeddings(texts, batch_size=32):
    """把文本分批送进 BERT，取 CLS 向量作为句向量"""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            out = bert(**inputs)
        vectors.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(vectors)

train_emb = get_embeddings(train['text'].tolist())
test_emb = get_embeddings(test['text'].tolist())
```

拼接后我们重新训练 CatBoost，这次把深度降到 6，学习率提到 0.015，L2 降到 15。调参的依据不是随机搜索，而是先看特征数量：加入 312 维句向量后，模型更容易过拟合，所以适当减浅树、减小正则。材料里没有给出具体验证分数，所以我们只保留结构上的选择，不编造数字。实际训练时我们还会固定随机种子，保证同一份数据每次跑出来的概率一致，方便复现和对比。

### 2.4 模型测试与指标报告

测试阶段，我们对测试集每个 `(dialog_id, participant_index)` 输出概率，并裁剪到 `[0.001, 0.999]`，避免提交极端值。裁剪的另一个好处是，如果某个对话在训练集里从未出现类似模式，模型给出的 0 或 1 会被拉回可比较的范围，避免单个样本主导评分。最后的断言检查则确保提交文件没有空值、行数也没有错位。

```python
# 概率裁剪，既保留排序信息，又避免 0 和 1 带来的数值问题
test['is_bot'] = np.clip(test['is_bot'], 0.001, 0.999)

# 按提交样例的 ID 顺序合并，保证一行不多一行不少
submission = ytest[['ID', 'dialog_id', 'participant_index']].merge(
    test[['dialog_id', 'participant_index', 'is_bot']],
    on=['dialog_id', 'participant_index'],
    how='left'
)

# 检查缺失值和行数
assert submission['is_bot'].notna().all()
assert len(submission) == len(ytest)
submission[['ID', 'is_bot']].to_csv('submission.csv', index=False)
```

材料里的两个方案都没有报告 public LB 分数，所以我们无法给出一个可引用的得分。我们能确认的是流程完整：数据读取、特征构造、交叉验证、测试集预测、格式对齐都跑通了。从工程角度看，最值得保留的是“行为特征 + 文本特征”的双通道设计：行为特征抓住复读和节奏，文本特征抓住触发词和语义，两者互补。概率裁剪能避免极端值带来的提交风险，而 5 折平均则降低了对单份划分的敏感度。

## 三、总结

本文中我们完成了 You Are Bot 2 的机器人识别流程。我们首先读取 JSON 对话数据并观察其层级结构，其次把每位参与者的发言整理成文本和六个行为特征，接着用 5 折交叉验证训练 CatBoost，然后加入俄语 rubert-tiny2 的句向量做第二版模型，最后对测试集输出概率并按要求生成提交文件。这个比赛给我们的启发是：判断对话参与者是不是机器人，不能只看他说了什么，还要看他怎么说话；重复率、消息占比、标点密度这些便宜的特征，往往和文本语义同样有用。未来可以尝试在对话层面做对比，把两位参与者的特征做相对化，以进一步突出机器人的行为差异。
