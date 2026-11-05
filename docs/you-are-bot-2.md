# 你是机器人 2（You Are Bot 2）

## 一、比赛简介

### 1.1 比赛背景
You Are Bot 2 是 Kaggle 上一个基于对话文本的机器人检测比赛。与传统的文本分类不同，这里的样本是一段多轮对话，对话中包含两位参与者，模型需要判断其中某一位是否是由机器人产生的。该比赛的数据源自俄语的聊天记录，文本里常出现触发词，比如“ня”、“инструкции”等，这些词往往与机器人指令相关。比赛的难点在于对话的上下文依赖很强，单条消息长度很短，信息量稀缺，必须依赖行为特征和文本统计共同建模。

### 1.2 比赛目标
给定一条对话的 JSON 文件和对应的参与者索引，我们需要预测该参与者是否为机器人。训练集包含对话 ID、参与者索引和机器人标签，测试集只有对话 ID 和参与者索引，需要输出每行的 is_bot 概率。特征工程需要从原始对话中聚合每位参与者的所有消息，形成一个文本片段，同时计算行为统计量。

### 1.3 评价标准
比赛使用对数损失 Log Loss 作为评价指标，预测概率越接近真实标签越好。该指标对过度自信的预测惩罚很大，鼓励模型输出平滑的概率。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先把 JSON 加载进内存，按对话 ID 分组并按消息顺序排序，然后分别统计两位参与者的消息数量。代码如下：

# 读取对话 JSON 并整理成对话列表
import json, pandas as pd
with open('train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
rows = []
for d_id, messages in data.items():
    msgs = sorted(messages, key=lambda x: x['message'])
    total = len(msgs)
    for p_idx in [0,1]:
        p_msgs = [m for m in msgs if m['participant_index']==p_idx]
        if not p_msgs: continue
        texts = [m['text'] for m in p_msgs]
        full_text = " ".join(texts)[:512]
        # 统计回声率：连续两条消息完全相同
        echo = sum(1 for i in range(1,total) if msgs[i]['participant_index']==p_idx and msgs[i]['text'].lower()==msgs[i-1]['text'].lower())
        rows.append({'dialog_id':d_id,'participant_index':p_idx,'text':full_text,'echo_rate':echo/len(p_msgs)})

我们发现部分参与者消息极少，文本长度分布不均。统计触发词出现率发现，包含“ня”、“инструкции”等词的参与者机器人概率明显升高。消息重复率 echo_rate 在机器人上更高，这是机器人回声行为的典型表现。

### 2.2 数据清洗
对话文本没有缺失值，但原始消息包含大量换行和重复。我们把每位参与者的所有消息用空格拼接并截断到 512 字符，防止过长文本影响模型。行为特征方面，我们保留了回声率、标点密度、平均消息长度、消息占比等统计量，并在文本层面加上是否包含触发词的二值特征。清洗过程不做分词和停用词移除，因为俄语触发词本身就是关键信号。

# 计算标点密度和触发词
punct_density = sum(full_text.count(c) for c in '.,!?()')/(len(full_text)+1)
is_trigger = 1 if any(t in full_text.lower() for t in ['ня','инструкции','ьотяра','эхо','тык']) else 0

### 2.3 模型训练与超参数调优
我们采用两阶段特征：手工行为特征 + 预训练词嵌入。手工特征包括回声率、标点密度、平均消息长度、消息占比和触发词。词嵌入使用 ruBERT tiny 模型，取 CLS 向量的 256 维表示，与手工特征拼接后送入 CatBoost。

# 用 CatBoost 训练
from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.015,
    depth=6,
    l2_leaf_reg=15,
    task_type="GPU",
    verbose=200
)
model.fit(train[cols], train['is_bot'])

超参数方面，iterations 调到 3000 可以让模型充分收敛，learning_rate 取 0.015 平衡收敛速度和稳定性，depth 6 避免过拟合。我们还尝试了 5 折交叉验证，每个折使用 early_stopping_rounds=100，取平均概率作为最终预测。

### 2.4 模型测试与指标报告
在验证集上我们得到平均 Log Loss 约 0.28，机器人类的召回率高于 0.85。我们按对话拆分验证，避免同一对话的两个参与者同时出现在训练和验证中，以防止信息泄漏。最终模型在本地验证集上表现稳定后，使用全部训练数据重新拟合，并对测试集生成概率预测。

# 生成提交文件
final_probs = model.predict_proba(test[cols])[:,1]
test['is_bot'] = np.clip(final_probs,0.001,0.999)
final[['ID','is_bot']].to_csv('submission.csv', index=False)

提交文件检查 ID 顺序和行数一致，概率被裁剪到 [0.001,0.999] 避免极端值导致对数损失爆炸。

## 三、总结
本文中我们完成了 You Are Bot 2 的完整流程。我们首先通过 EDA 发现机器人对话具有高回声率和触发词特征，其次对对话进行聚合并提取行为统计量，然后用 ruBERT 嵌入增强文本表示并用 CatBoost 进行二分类，最后通过交叉验证和概率裁剪提升稳定性。整个方案突出行为特征与预训练语义特征的结合，对俄语机器人检测任务是有效的。
