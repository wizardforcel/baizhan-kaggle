# 英文文本规范化挑战 Text Normalization Challenge - English Language

## 一、比赛简介

### 1.1 比赛背景

这个比赛是 Kaggle 上的英文文本规范化任务。真实世界里，数字、日期、金额、网址等写法在文字里是一串符号，但在口语中要读成“二零二三年十月五日”“一百二十美元”这样的自然语言。比赛提供的训练数据里，每个 token 都有 `before`（原始写法）和 `after`（规范化后的读法），以及一个 `class` 标签说明它属于哪种类型，比如 `CARDINAL`、`DATE`、`MONEY`、`ELECTRONIC` 等。训练集和测试集都按句子分块，每个 token 有 `sentence_id` 和 `token_id` 定位。

从材料来看，数据来自公开的文本规范化语料，覆盖了日常英语里常见的十几种规范化类型。难点在于同样的字符串在不同语境下可能属于不同 class，比如 “2016-03-31” 在 DATE 里要读成 “the thirty first of March two thousand sixteen”，但孤立出现时又可能被当成 DIGIT。材料里多次提到类重叠的问题，正是这类歧义带来的。

### 1.2 比赛目标

我们需要把测试集里每一行的 `before` 词元转换成语音友好的 `after` 词元。训练集有五列：`sentence_id`、`token_id`、`before`、`after`、`class`。测试集缺少 `class` 和 `after`，只有 `sentence_id`、`token_id`、`before`。

提交文件要求用 `id` 纪录，每条 `id` 是 `sentence_id_token_id` 的拼接，预测列名为 `after`。比赛本质上是分词级别的文本重写任务，不是分类，而是规范化。

### 1.3 评价标准

Kaggle 排行榜使用与规范化准确率相关的分数，材料里多次提到 LB（Leaderboard）分数，例如 XGBoost 加类特定正则的方案达到 LB 0.9915，简单随机森林基线 LB 0.9867。分数越高越好，满分 1.0。没有在材料里给出官方指标名称，我们保守地按排行榜分数理解，即预测 `after` 与标准答案完全一致的比例。

## 二、解题过程

整场比赛的思路是：先用 EDA 了解数据结构，再用训练集构建 before→after 词典覆盖常见词，对未知词按 class 使用正则和规则函数生成读法。我们用第一人称复数叙述整个流程。

### 2.1 数据探索（EDA）

我们首先读入数据，看看样本量和列结构。

```python
# 导入 pandas，读取训练和测试文件
import pandas as pd

# 训练集包含 sentence_id、token_id、before、after、class 五列
train = pd.read_csv("../input/en_train.csv")
test = pd.read_csv("../input/en_test.csv")

# 看形状，确认训练样本远大于测试
print(train.shape)  # 例如约 9.9M 行
print(test.shape)
# 查看前几行，确认列名
train.head()
```
通过 `shape` 我们看到训练集有近千万行，测试集约占整体的 9%。每行是一个 token，不是句子。材料指出每个句子长度集中在 8 到 18 个 token，7 token 的句子出现峰值。

接下来看句子长度分布和 class 构成。

```python
# 按句子统计 token 数
train_sentences = train.groupby("sentence_id")["sentence_id"].count()
train_sentences.describe()
# 绘图观察峰值，材料里发现 7 token 句子特别多

# 看有多少个不同的 class
num_classes = len(train["class"].unique())
print(num_classes)  # 材料里约十几种
# 各 class 样本数
train.groupby("class")["class"].count()
```
结果显示 `PLAIN` 占绝大多数，其余如 `CARDINAL`、`DATE`、`MONEY`、`ELECTRONIC`、`VERBATIM` 等是少数但需要重点处理的类。材料还统计了 `before != after` 的变化比例，约有 66 万行发生了变化，说明大部分 token 是平原词不变的。

我们还检查了句子长度与变化数的关系，材料里发现句子越长，变化数有弱相关，但整体呈弥散云状。位置分析显示句子前部和第 4 位更容易发生变化，和语法结构有关。

### 2.2 数据清洗

这份数据本身很干净，没有缺失值。清洗的关键是构建查找表和为未知词准备回退规则。

我们先统计 before→after 的映射，建立词典。

```python
# 统计训练集中 before 到 after 的映射
from collections import defaultdict

diffs = dict()
total = 0
not_same = 0
for row in train[["before","after"]].values:
    total += 1
    if row[0] != row[1]:
        not_same += 1
    if row[0] not in diffs:
        diffs[row[0]] = dict()
        diffs[row[0]][row[1]] = 1
    else:
        if row[1] in diffs[row[0]]:
            diffs[row[0]][row[1]] += 1
        else:
            diffs[row[0]][row[1]] = 1
print("总行数:", total, "变化行数:", not_same)
```
我们得到近 9.9M 总行，其中约 659k 行发生了变化。这一步是为了让常见词直接查表，查表是最快且最准确的方式。

对于类重叠的问题，材料指出同一字符串可能在 DATE 和 DIGIT 里出现。我们的做法是先查表，若未命中再按预测的 class 走规则函数。为了给规则函数提供训练统计，我们为每个 class 做一次 bag-of-words 统计，取最常见映射。

```python
from collections import defaultdict, Counter

def bagOfWords(cls):
    # 对指定 class 统计 before->after 的最常见映射
    d = defaultdict(list)
    train_cls = train[train["class"]==cls]
    train_list = [(train_cls.iloc[i,3], train_cls.iloc[i,4]) for i in range(train_cls.shape[0])]
    for k,v in train_list:
        d[k].append(v)
    counter_dict = {}
    for key in d:
        c = Counter(d[key]).most_common(1)[0][0]
        counter_dict[key] = c
    return counter_dict

# 为主要 class 构建词袋
plain_trained = bagOfWords("PLAIN")
card_trained = bagOfWords("CARDINAL")
date_trained = bagOfWords("DATE")
# 其他 class 同理...
```
这样即使规则函数没覆盖到的边缘情况，也能先用训练中最常见的映射兜底。

### 2.3 模型训练与超参数调优

材料里主流方案分两步：先用上下文特征预测 class，再按 class 做规范化。我们参考 XGBoost 和类特定正则的方案，先用训练好的 class 预测结果。

```python
# 读取已有 class 预测结果，材料中常用 test_rf.csv
test_df = pd.read_csv("../input/classpredictions/test_rf.csv")
test_df.drop("Unnamed: 0", axis=1, inplace=True)
# 重新生成提交所需的 id
test_df['id'] = test_df[['sentence_id','token_id']].apply(lambda x: str(x[0])+"_"+str(x[1]), axis=1)
```
训练集上的 class 预测本身使用的是 XGBoost 模型，输入特征包含上下文、字符长度、正则匹配等，材料引用 Alpasis 的 XGBoost With Context Label Data 达到 99.637% 的分类准确率。我们直接使用该预测结果进行后续规范化。

规范化的核心是类特定函数，下面以数字和日期为例。

```python
import re
import inflect
p = inflect.engine()

def digit2word(key):
    # 先查词袋，若命中直接返回
    bag_res = bag2word(key, card_trained)
    if bag_res != key:
        return bag_res
    # 否则用 inflect 把数字转成英文单词
    try:
        text = p.number_to_words(key, decimal='point', andword='', zero='o')
        text = text.replace('-',' ').replace(',','')
        return text.lower()
    except:
        return key

def date2word(key):
    bag_res = bag2word(key, date_trained)
    if bag_res != key:
        return bag_res
    # 简单示例：YYYY-MM-DD 转成 "the first of January 2020"
    key = key.strip().replace('/', '-')
    if len(key.split('-')) == 3:
        y,m,d = key.split('-')
        # 这里省略完整月份映射，实际代码里有完整字典
        return f"the {d} of {m} {y}"
    return key

def bag2word(key, bag_dict):
    try:
        return bag_dict[key]
    except:
        return key
```
材料里的完整实现更复杂，包含 `num2words`、`roman`、`datetime` 等库，能处理 `MM/DD/YYYY`、`15th of Jan` 等多种写法。其他 class 如 `MEASURE`、`MONEY`、`ELECTRONIC`、`VERBATIM` 都有类似的 `*2word` 函数，统一先查词袋再走正则。

对于测试集，我们按预测的 class 逐行调用对应函数。

```python
# 简化示例：对 CARDINAL 类应用 digit2word
df_num = test_df[test_df['class']=="CARDINAL"]
out = []
for i in range(df_num.shape[0]):
    out.append(digit2word(df_num.iloc[i,2]))
df_num['after'] = out

# DATE 类同理
df_date = test_df[test_df['class']=="DATE"]
for i in range(df_date.shape[0]):
    df_date.loc[i, 'after'] = date2word(df_date.iloc[i,2])
```
材料里对 `PLAIN`、`PUNCT`、`TIME` 等类直接返回原词或使用词袋兜底，保证不破坏原有信息。整个流程保持第一人称：我们首先预测 class，其次按 class 调用正则，最后合并结果生成提交文件。

### 2.4 模型测试与指标报告

在本地验证时，我们用训练集划分验证集检查规范化准确率。材料里没有给出完整的交叉验证代码，但提供了 LB 分数作为参考。

```python
# 合并所有 class 的结果，生成最终提交
result_df = pd.concat([df_num, df_date, df_let, df_verb, df_elec, df_ord,
                       df_mes, df_dig, df_mon, df_dec, df_tel, df_frac,
                       df_add, df_plain, df_punct, df_time])

# 按 Kaggle 要求的格式输出
result_df[['id','after']].to_csv("submission.csv", index=False)
```
根据材料，XGBoost 加类特定正则的方案在公开排行榜上达到 LB 0.9915，简单随机森林基线约 0.9867。类特定正则在 `CARDINAL`、`DATE`、`MONEY` 等类别上提升明显，因为正则能把未见过的数字、日期正确转换，而单纯的查表会失效。

我们还检查了极端情况：最长句子、7 token 句子等。材料里提到 7 token 句子在末位变化频率极高，这类句子往往是模板化的如 “X as of Y”，规范化集中在日期或数字上，正则函数覆盖得好，准确率就会很高。

## 三、总结

本文中我们完成了一次对英文文本规范化挑战的完整复盘。我们首先通过 EDA 了解了数据是按 token 组织的，句子长度集中在 8-18，`PLAIN` 类占主导但关键类稀少且类重叠现象明显。其次我们做了最少量的数据清洗，重点是构建 before→after 词典和按 class 的 bag-of-words 统计，为后续规则提供兜底。接着我们采用了两阶段方案，先用 XGBoost 预测 token 的 class，再用类特定正则函数生成规范化读法，并在测试集上逐类应用。最后我们按照 Kaggle 提交格式生成 `submission.csv`，材料中该方案的公开 LB 分数约为 0.9915，基线约 0.9867，说明规则驱动的规范化在未见词上很有价值。

这个比赛给我们的启发是：文本规范化里“规则+统计”往往比纯机器学习更稳。查表解决常见词，正则解决系统性模式，模型负责把样本送到正确的处理分支，三者结合才能在排行榜上稳定超过 0.99。
