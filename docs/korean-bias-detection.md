# Korean Bias Detection 韩语偏见检测

## 一、比赛简介

### 1.1 比赛背景

韩语偏见检测比赛关注的是中文以外另一种语言里的社会偏见问题。不同于英语的世界，很多语言领域的偏见检测资料很少，尤其是韩语。比赛提供的训练数据来自收集的韩语评论，想要让模型学会识别评论里是否含有针对特定群体的偏见，比如性别相关的偏见。

这个任务属于文本分类。输入是一段韩语评论，输出是偏见的类型。偏见检测在内容审核、社交媒体管理里很有用，可以帮助平台自动过滤有问题的言论。

### 1.2 比赛目标

给定一段韩语评论，判断它是否含有偏见，以及偏见属于哪一类。材料中的代码显示测试文件是 `test.bias.no_label.csv`，包含 `comments` 列。我们需要为每条评论预测标签，并保存为提交文件。

标签是三分类，用数字表示：
- 0 代表 `none`，没有偏见
- 1 代表 `others`，其他类型的偏见
- 2 代表 `gender`，性别偏见

### 1.3 评价标准

偏见检测通常使用分类准确率或 F1 分数。材料里没有明确给出指标，但同类文本分类比赛常用的指标是准确率，即预测正确的样本比例。越高越好。

---

## 二、解题过程

整场比赛的思路很直接：用现成的韩语预训练模型做文本分类。我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步是查看数据文件，确认列名和数据量：

```python
# 导入常用库
import pandas as pd
import os

# 列出输入文件夹下的所有文件
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

运行后我们能看到训练集和测试集的文件名，确认测试集是 `test.bias.no_label.csv`。

接着读入测试文件，看看有多少条评论：

```python
# 读入测试数据
df = pd.read_csv('../input/korean-bias-detection/test.bias.no_label.csv')

# 看前几行
df.head()
```

我们能看到 `comments` 列里全是韩语文本。材料里没有提供训练集的 EDA，但从模型选择来看，标签只有三类，数据量应该足够让预训练模型直接推理。

### 2.2 数据清洗

韩语文本本身就是干净的，基本不需要复杂清洗。我们只需要确保文本能被模型正确读取：

```python
# 确认 comments 列没有缺失值
df['comments'].isnull().sum()
```

如果有空评论，可以用空字符串填充，避免模型报错。材料里的代码没有做清洗，说明数据质量已经不错，直接送入模型即可。

我们不做分词或大小写转换，因为韩语没有大小写，预训练模型本身就处理好分词。

### 2.3 模型训练与超参数调优

这个比赛没有训练自己的模型，而是直接使用社区提供的韩语偏见检测模型 `beomi/beep-KcELECTRA-base-bias`。这是基于 KcELECTRA 的微调模型，专门针对韩语偏见任务。

我们需要先安装 transformers 库，然后加载模型：

```python
# 安装最新版 transformers
!pip install -U transformers

from transformers import pipeline

# 加载韩语偏见检测模型
pipe = pipeline('text-classification', 
                model='beomi/beep-KcELECTRA-base-bias', 
                device=0)
```

`pipeline` 把分词、模型推理、后处理都封装好了，我们只需要输入文本就能得到标签。

验证模型能否工作：

```python
# 测试一条简单问候语
pipe("안녕하세요")[0]['label']
```

输出应该是 `none`，说明模型能正常运行。

### 2.4 模型测试与指标报告

有了模型后，我们直接对测试集所有评论做预测：

```python
# 对所有评论做预测
df['label'] = df['comments'].map(lambda x: pipe(x)[0]['label'])
```

这里用 `map` 遍历每条评论，`pipe(x)[0]['label']` 取出模型返回的第一个结果的标签。

模型返回的标签是字符串 `none`、`others`、`gender`，需要转成比赛要求的数字：

```python
LABEL_DIC = {
    'none': 0,
    'others': 1,
    'gender': 2,
}

df['label'] = df['label'].map(lambda x: LABEL_DIC[x])
```

最后保存提交文件：

```python
df.to_csv('./submission.csv', index=None)
```
```

我们还可以看看预测结果的分布，确认模型没有把所有样本都分到一个类别：

```python
df.label.value_counts()
```

这一步能帮我们发现模型是否失效，比如所有预测都是 0。

---

## 三、总结

本文中我们完成了韩语偏见检测比赛的完整流程。我们首先通过 EDA 确认了数据结构，测试集只有 `comments` 列，标签需要我们预测。其次我们没有做复杂的数据清洗，直接使用原始韩语文本。接着我们加载了现成的韩语偏见检测模型 `beomi/beep-KcELECTRA-base-bias`，用 transformers 的 pipeline 做推理，并把字符串标签转成数字 0、1、2。最后我们生成了提交文件，并用 `value_counts` 快速检查了预测分布。

这个比赛给我们的启发是：对于小语种任务，使用社区提供的微调模型往往比从零训练更高效。模型已经学会了韩语的语言习惯和偏见特征，我们只需要做推理和格式转换就能完成任务。
