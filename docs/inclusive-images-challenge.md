# Inclusive Images Challenge

## 一、比赛简介

### 1.1 比赛背景
Inclusive Images Challenge 由 Google 举办，目标是在图像识别中解决地域偏差问题。训练数据大部分来自某些地区，测试图片则可能来自文化、环境完全不同的地区。例如训练集中出现很多美国的婚礼、服装和交通工具，而当模型看到非洲或亚洲的同类物品时，可能因为外观差异而识别错误。

这个比赛使用的数据来自 Open Images 数据集。图像由真实用户拍摄，带有复杂背景和不规则的标注。比赛提供了训练标注、人类标注、机器标注、边界框、类别描述、调参标注（tuning labels）等多个文件。参赛者需要利用这些数据，训练一个能在各地域都表现稳定的多标签分类器。

### 1.2 比赛目标
从材料中的文件清单来看，比赛主要有以下几类文件：
- `train_human_labels.csv`：人类标注的图片与标签；
- `train_machine_labels.csv`：机器标注的图片与标签；
- `train_bounding_boxes.csv`：边界框标注，包含 `ImageID`、`Source`、`LabelName`、`Confidence`、`XMin/XMax/YMin/YMax`、`IsOccluded` 等字段；
- `classes-trainable.csv`：可训练类别列表；
- `class-descriptions.csv`：类别的英文描述；
- `tuning_labels.csv`：每个图片 ID 对应的一串标签，用于调参；
- `stage_1_sample_submission.csv`：提交样例，包含 `ImageID` 和多标签列。

我们的任务是预测每张测试图片的多个标签。测试图片位于 `stage_1_test_images` 文件夹中，需要给每张图片输出一个或几个类别。

### 1.3 评价标准
材料中有专门的 notebook 实现 F2 评估。F2 是 F-score 的一种，beta 参数为 2，更重视召回率：宁可多预测一点，也不要漏掉真实标签。计算方式为：

```python
# F2 公式：精确率和召回率的加权调和平均，beta=2 更重视召回
F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
```

比赛对每张图片单独计算 F2，然后对所有图片求平均，得到 mean samples F2，越高越好。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读取各个 CSV 文件，了解数据规模。

```python
import pandas as pd

# 可训练类别文件只有一列 label_code
classes = pd.read_csv('../input/classes-trainable.csv', index_col=0)
print('可训练类别数:', classes.shape[0])

# tuning 标签用于本地调参，每行一个图片和它的标签串
tuning = pd.read_csv('../input/tuning_labels.csv', index_col=0,
                     header=None, names=['image_id', 'labels'])
print('调参图片数:', tuning.shape[0])
```
材料输出显示可训练类别有 7,178 个，调参图片有 1,000 张。这个规模说明模型最后输出的是 7 千多维的概率向量。

接着我们看看人类标注的规模。

```python
train = pd.read_csv('../input/train_human_labels.csv', usecols=['ImageID', 'LabelName'])
print('{} images with {} unique labels'.format(
    train.ImageID.nunique(), train.LabelName.nunique()))
```
输出是 1,743,042 张图片、18,155 个唯一标签。标签的多样性远高于可训练类别，说明很多标签不在考核范围内。

我们还通过 BigQuery 查到了图片 URL，并读取了边界框。

```python
from bq_helper import BigQueryHelper

# 用 BigQuery 查询 Open Images 的图片 URL
open_images = BigQueryHelper(active_project="bigquery-public-data",
                             dataset_name="open_images")
query = """
SELECT image_id, original_url
FROM `bigquery-public-data.open_images.images`
LIMIT 100
"""
urls = open_images.query_to_pandas_safe(query)
```
有了 URL，我们可以随机挑选图片，把类别描述叠加在图片上观察。

```python
# 把标签代码转成描述，方便理解图片
descrips = pd.read_csv('../input/class-descriptions.csv',
                       names=['LabelName', 'Description'])
train = train.merge(descrips, how='left', on='LabelName')

# 查看一张图片的标签
image = train.ImageID.iloc[0]
label = train.loc[train.ImageID == image, 'Description'].str.cat(sep=', ')
print(label)
```
我们还做了标签频率分析。材料指出前 3 类分别是 Person、Clothing 和 Human Face，并且大多数标签出现次数很少，呈长尾分布。

```python
# 统计每个标签出现的比例
dcounts = train.groupby('Description')['ImageID'].nunique() / train.ImageID.nunique()
dcounts_df = pd.DataFrame({'label': dcounts.index, 'pct_of_images': dcounts.values})
print(dcounts_df.sort_values('pct_of_images', ascending=False).head(10))
```
大多数图片可能只有一到几个标签，但也存在标签很多的图片。我们用直方图统计每张图的标签数量。

```python
images = train.groupby('ImageID').count()
images.columns = ['LabelCount']
images['LabelCount'].hist(bins=50)
```
### 2.2 数据清洗
我们其次把 `tuning_labels.csv` 中的标签字符串转换成多热矩阵。

```python
# 每行标签以空格分隔，转成列表
tuning['labels'] = tuning['labels'].str.split()

# 用可训练类别的索引建立布尔矩阵
y_true = tuning['labels'].apply(
    lambda lbls: classes.index.isin(lbls))
y_true = pd.DataFrame(y_true.values.tolist(), index=y_true.index)
```
这样我们就得到了 1000 × 7,178 的矩阵，可以直接与模型输出比较。

接着我们在训练基线里合并不同来源的标签。材料中的 DenseNet121 baseline 用了约 1.7M 张训练图片，融合了人类标注、机器标注和边界框三类信息。

```python
from collections import defaultdict

# 统计调参集中每个标签出现的次数
tuning_labels_freq = defaultdict(int)
for r in tuning['labels']:
    for l in r:
        tuning_labels_freq[l] += 1
tuning_labels_list = list(tuning_labels_freq.keys())
```
材料中提到，tuning 集里有 484 个唯一标签。我们只保留这些标签训练演示模型，但实际提交时仍要输出 7,178 个可训练类别。

```python
# 标签到索引的映射
label_2_idx = {v: i for i, v in enumerate(tuning_labels_list)}

# 把一条标签串转成 one-hot 向量
def caption_2_one_hot(caption, n_classes=1, lookup_dict=None):
    y = np.zeros((n_classes))
    for w in caption.split():
        y[lookup_dict[w]] = 1
    return y
```
### 2.3 模型训练与超参数调优
我们接着使用 Keras 上的 DenseNet121 作为大脑，接收 224×224 的图片，输出每个类别的概率。

```python
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam

# 用预训练的 DenseNet121 提取特征
base = DenseNet121(include_top=False, weights='imagenet', pooling='avg')
output = Dense(N_CLASSES, activation='sigmoid')(base.output)
model = Model(inputs=base.input, outputs=output)
# 多标签用 binary_crossentropy，而不是 categorical_crossentropy
model.compile(optimizer=Adam(), loss='binary_crossentropy')
```
我们准备了一个批量数据生成器，读图片、调整尺寸、归一化，同时把标签转成向量。

```python
import cv2 as cv
import numpy as np

def ImageDataGen(ids, df, lookup_dict=label_2_idx,
                 n_classes=N_CLASSES, img_dir=IMG_DIR,
                 input_size=INPUT_SIZE, bs=BATCH_SIZE):
    while True:
        for start in range(0, len(ids), bs):
            x_batch, y_batch = [], []
            sample = ids[start:start+bs]
            for img_id in sample:
                img = cv.imread('{}/{}.jpg'.format(img_dir, img_id))
                if img is None:
                    continue
                img = cv.resize(img, (input_size, input_size))
                img = preprocess_input(img.astype(np.float32))
                x_batch.append(img)
                y_batch.append(caption_2_one_hot(
                    df.loc[img_id]['Caption'],
                    n_classes=n_classes,
                    lookup_dict=lookup_dict))
            yield np.array(x_batch), np.array(y_batch)
```
训练时我们把图片分成 89% 训练、10% 验证、1% 留作测试；材料里的示例是 891 张训练、99 张验证、10 张测试。批大小设为 8，输入尺寸 224。

```python
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# 保存验证损失最低的模型；学习率下降帮助收敛
callbacks = [
    ModelCheckpoint('best.h5', save_best_only=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]
model.fit(train_gen, validation_data=val_gen,
          steps_per_epoch=len(train_ids)//BATCH_SIZE,
          epochs=10, callbacks=callbacks)
```
为了减少地域偏差，我们还可以根据图片的来源属性进行分组，并在验证时按来源计算分数。材料没有说明来源字段的具体名称，我们只把来源作为观察维度，不在训练中直接使用，以免泄漏。

### 2.4 模型测试与指标报告
我们在调参集上用 F2 计算分数。

```python
from sklearn.metrics import fbeta_score

# 预测概率
pred = model.predict(test_gen, steps=len(test_ids)//BATCH_SIZE + 1)

# 阈值 0.5 把概率变成 0/1 标签
pred_bin = (pred > 0.5).astype(int)

# 按样本计算 F2 并平均
def mean_samples_f2(y_true, y_pred):
    scores = []
    for t, p in zip(y_true, y_pred):
        tp = np.sum(t * p)
        precision = tp / (np.sum(p) + 1e-12)
        recall = tp / (np.sum(t) + 1e-12)
        scores.append(5 * precision * recall / (4 * precision + recall + 1e-12))
    return np.mean(scores)

print('Mean Samples F2:', mean_samples_f2(y_true.values, pred_bin))
```
材料中还有 PyTorch 版的 F2 实现，计算方式一致，只是用张量完成。

```python
import torch

def f2_score(pred, target, beta=2, eps=1e-12):
    # 按类别计算 TP、FP、FN，再求 F2
    tp = (pred * target).sum(dim=0)
    fp = (pred * (1 - target)).sum(dim=0)
    fn = ((1 - pred) * target).sum(dim=0)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
    return f2.mean(0)
```
最后我们按 `stage_1_sample_submission.csv` 的顺序生成提交。

```python
# 读取提交样例，确认列顺序和图片 ID
submission = pd.read_csv('../input/stage_1_sample_submission.csv')

# 把预测转换成同一格式：每张图一列标签串
for i, image_id in enumerate(submission['ImageID']):
    labels = [idx_2_label[j] for j, v in enumerate(pred_bin[i]) if v == 1]
    submission.loc[i, 'labels'] = ' '.join(labels)

submission.to_csv('submission.csv', index=False)
```
## 三、总结

本文中我们完成了 Inclusive Images Challenge 的多标签图像分类流程。我们首先通过 EDA 确认了可训练类别 7,178 个、人类标注图片 1,743,042 张、调参图片 1,000 张，并观察到标签呈长尾分布。其次我们把调参标签转换成多热矩阵，并把多个标注来源合并，构建统一的训练集。然后我们使用预训练 DenseNet121，采用 sigmoid 多标签输出和二元交叉熵损失，配合数据生成器与学习率衰减进行训练。最后我们用 F2 指标评估，并按样例格式生成提交。

这个比赛让我们认识到，地域偏差和多标签标注是图像识别中的两个重要问题。F2 更重召回，意味着模型不能漏掉稀有标签；真实数据中的并行标注、重复图片和长尾类别，需要在数据清洗和模型训练中认真处理。
