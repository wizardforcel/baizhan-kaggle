# LegalEval Extract（法律文书抽取质量评分）

## 一、比赛简介

### 1.1 比赛背景
LegalEval Extract 是一个评估「信息抽取质量」的比赛。数据来自俄罗斯法院的司法文书：每一条样本给出一份原文 `source_text`、一个被抽取出来的片段 `extracted_text`，以及这个片段对应的区块名称 `block_name`（例如文书的某个结构性段落）。专家已经为每个抽取结果打了 1 到 5 分，分数越高表示抽取越准确、越完整。比赛要求我们自动给出这个分数，本质上是让模型当「评审」，判断抽取片段有没有漏掉内容、有没有混入无关文本。

### 1.2 比赛目标
训练集包含 `row_id`、`id_sud_resh`、`block_name`、`extracted_text`、`source_text`、`expert_score_mean` 六列；测试集只有前五列。`expert_score_mean` 是 1.0 到 5.0 的连续分数。我们要为测试集每一行预测 `predicted_score`，提交文件包含 `row_id` 和 `predicted_score` 两列。

### 1.3 评价标准
这是一个回归任务，评价指标是预测分数与专家分数之间的误差，通常用平均绝对误差 MAE 或均方根误差 RMSE 衡量，误差越小越好。由于分数被限制在 1 到 5 之间，任何超出范围的值都会被截断回这个区间。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先加载训练集和测试集，确认列名齐全，再看分数的分布。

```python
from pathlib import Path
import pandas as pd

DATA_DIR = Path('/kaggle/input/competitions/legaleval-extract')
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

print('train:', train.shape)
print('test:', test.shape)
print(train['expert_score_mean'].describe())
```

打印出的统计量告诉我们训练集规模、专家打分的均值和中位数，以及最小最大值。接着查看 `block_name` 的分布，了解不同区块的样本量。

```python
# 看看最常见的前 20 种区块
print(train['block_name'].value_counts().head(20))
```

不同 `block_name` 对应的抽取难度不一样，有的区块短、容易判，有的区块长、容易漏，这为后面的误差分析提供了线索。

### 2.2 数据清洗
原始文本里可能包含空字节、连续空白和多余换行。我们写一个轻量清洗函数，并把空值统一成空字符串。

```python
import re
import numpy as np

def clean_text(x):
    # 空值统一成空串，避免后续 len() 报错
    x = '' if x is None else str(x)
    x = x.replace('\x00', ' ')          # 去掉空字节
    x = re.sub(r'[ \t\r\f\v]+', ' ', x) # 连续空白压成一个空格
    x = re.sub(r'\n{3,}', '\n\n', x)    # 最多保留两个换行
    return x.strip()
```

因为模型的输入长度有限，我们还需要一个截断函数：保留开头大部分内容，再保留结尾一小段，中间用标记省略。这样既不丢结论性的结尾，又能控制 token 数。

```python
def smart_truncate(text, max_chars):
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.72)        # 开头占 72%
    tail = max_chars - head - 80        # 结尾留一段
    return text[:head].rstrip() + '\n...[TRUNCATED]...\n' + text[-tail:].lstrip()
```

最后统一列的类型，并把专家分数截断到 1~5。

```python
for df in (train, test):
    for col in ['row_id', 'id_sud_resh', 'block_name', 'source_text', 'extracted_text']:
        df[col] = df[col].fillna('').astype(str)

train['expert_score_mean'] = train['expert_score_mean'].astype(float).clip(1.0, 5.0)
```

### 2.3 模型训练与超参数调优
我们没有从头训练模型，而是使用 Kaggle 环境里现成的因果语言模型 `bond005/meno-lite-0.1`，用「提示词 + 生成」的方式让它直接输出一个分数。提示词里写清楚评分标准，并要求只返回一个数字。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = 'bond005/meno-lite-0.1'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'   # 批量生成时左侧补齐

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map='auto', torch_dtype=torch.float16,
    trust_remote_code=True, low_cpu_mem_usage=True)
model.eval()
```

评分标准（rubric）是效果的关键：它把 1、2、3、4、5 分分别对应到「完全无关」「严重丢失」「部分正确」「基本正确」「几乎完美」，并要求模型只输出一个数字、不要解释。

```python
RUBRIC = '''你是一名法律文书抽取质量评审专家。
请评估 EXTRACTED_TEXT 是否准确、完整地从 SOURCE_TEXT 中抽取了 BLOCK_NAME 对应的区块。
评分范围 1.0 到 5.0：
1.0 = 无关、为空或抽错区块
2.0 = 少量相关，丢失严重
3.0 = 部分正确，有明显遗漏或多余内容
4.0 = 大体正确，有少量遗漏或噪声
5.0 = 几乎完美，完整且准确
只返回一个数字，不要文字、不要 JSON。'''

def build_prompt(row):
    return f'''{RUBRIC}

任务
BLOCK_NAME: {clean_text(row.block_name)}
SOURCE_TEXT:
{smart_truncate(row.source_text, 2600)}
EXTRACTED_TEXT:
{smart_truncate(row.extracted_text, 1400)}
SCORE:'''
```

推理时把样本按 8 条一批送进模型，用 `generate` 生成最多 6 个新 token，再用正则把输出里的第一个数字解析出来，限制在 1~5。

```python
SCORE_RE = re.compile(r'[-+]?\d+(?:[\.,]\d+)?')

def parse_score(text):
    m = SCORE_RE.search(text or '')
    if m is None:
        return 3.0                       # 解析失败时退回中间分
    val = float(m.group(0).replace(',', '.'))
    return float(np.clip(val, 1.0, 5.0))
```

为了让分数更贴近专家，我们又构造了几个纯统计特征：原文长度、抽取长度、长度比、词重合率、是否为空。当抽取为空时直接给 1 分；词重合率高、长度比合理时加分。

```python
WORD_RE = re.compile(r'[а-яa-z0-9]{3,}', re.IGNORECASE)

def lexical_features(source, extracted):
    s, e = clean_text(source).lower(), clean_text(extracted).lower()
    sw, ew = set(WORD_RE.findall(s)), set(WORD_RE.findall(e))
    return {
        'len_ratio': len(e) / max(1, len(s)),
        'word_overlap': len(sw & ew) / max(1, len(ew)),
        'empty_extracted': float(len(e) == 0),
    }
```

最后在 200 条留出样本上做一次轻量校准：先用线性回归把原始输出映射到专家分数，再用 Ridge 把「原始分 + 校准分 + 统计特征」融合起来，选 MAE 最低的方案。

```python
from sklearn.linear_model import LinearRegression, Ridge

lin = LinearRegression().fit(raw.reshape(-1, 1), y)
cal = np.clip(lin.predict(raw.reshape(-1, 1)), 1.0, 5.0)

ridge = Ridge(alpha=2.0).fit(np.column_stack([raw, cal, feat]), y)
blend = np.clip(ridge.predict(np.column_stack([raw, cal, feat])), 1.0, 5.0)
```

### 2.4 模型测试与指标报告
在留出集上分别计算三种方案的 MAE 和 RMSE。

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

mae_raw = mean_absolute_error(y, raw)
rmse_raw = math.sqrt(mean_squared_error(y, raw))
print('MAE raw:', mae_raw, 'RMSE raw:', rmse_raw)
```

我们比较 `raw`、`cal`、`blend` 三种结果的 MAE，选择最低的那个作为最终后处理方式，然后把测试集的原始生成分数送进同样的流程。提交前统一截断到 1~5。

```python
submission = pd.DataFrame({
    'row_id': test['row_id'].astype(str),
    'predicted_score': test_preds.clip(1.0, 5.0),
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(submission.shape)
```

## 三、总结
本文中我们完成了 LegalEval Extract 的评分子任务。我们首先读取数据并观察分数分布与区块分布。其次编写了文本清洗和智能截断函数，把超长文书压到模型可接受的长度。接着使用 `bond005/meno-lite-0.1` 做零样本评分，通过严格的评分提示词约束输出格式，并解析出 1~5 的分数。然后加入长度比、词重合率等统计特征，在留出集上用线性回归和 Ridge 做后处理校准，按 MAE 选择最优方案。最后把结果截断到合法区间并生成 `submission.csv`。整套方案没有修改 Kaggle 运行环境，只依赖镜像里已有的 transformers 和 torch。