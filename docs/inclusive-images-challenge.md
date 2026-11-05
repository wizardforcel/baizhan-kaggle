# Inclusive Images Challenge

## 一、比赛简介

### 1.1 比赛背景
Inclusive Images Challenge 由 Google 主办，目标是解决图像识别中的地域偏差问题。传统的图像分类器在训练时大量依赖欧美地区拍摄的图片，当遇到非洲、东南亚等地区拍摄的相似物品时，识别率会明显下降。原因在于，同一个类别的物品在不同文化背景下外观可能完全不同，比如婚礼服饰、街头交通工具、当地食物等。

这个比赛使用的是 Open Images 数据集，它包含了超过数百万张用户上传的图片，每张图片都带有人工标注的人类标签、机器预测标签和边界框标注。比赛的目标是训练一个能够公平地在各地区都表现稳定的多标签分类模型。参赛者可以利用 tuning 标签来微调模型，并需要在 stage 1 和 stage 2 分别做预测。材料中提供的 `stage_1_test_images`、`stage_2_sample_submission.csv` 以及 `stage_1_attributions.csv` 都表明比赛分为两个阶段，且对归因和解释性有一定要求。

这个比赛不仅考核分类能力，还考核模型对地域差异的适应性，这对入门者来说是一个非常好的实践：模型泛化性不仅仅是数据量问题，数据分布的偏差会让模型在某些群体上失效。

### 1.2 比赛目标
比赛的目标是为每张测试图片预测其包含的多个类别。测试图片来自 Open Images 测试集，图片 ID 列在提交样例里。训练数据包括以下几类：

- `train_human_labels.csv`：人工标注的类别标签，一张图对应多行，每行一个标签；
- `train_machine_labels.csv`：机器生成的标签，用于补充；  
- `train_bounding_boxes.csv`：边界框标注，包含 `ImageID`、`Source`、`LabelName`、`Confidence`、`XMin/XMax/YMin/YMax`、是否遮挡、是否截断等字段；
- `classes-trainable.csv`：可训练类别列表，共 7,178 个类别；
- `class-descriptions.csv`：每个类别的英文描述，便于理解标签含义；
- `tuning_labels.csv`：只有 1,000 张图片的标签，专门供本地调参使用；
- `stage_1_sample_submission.csv` 和 `stage_2_sample_submission.csv`：提交格式样例。

我们的任务是输出每张测试图片的标签集合。标签之间用空格分隔，重复标签会被惩罚。

### 1.3 评价标准
比赛使用平均样本 F2 分数作为评价指标。F2 是 F-score 的一种，beta 参数为 2，相比普通的 F1 更重视召回率。计算公式为：

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F2 = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)
```

其中 TP 是真正例，FP 是假正例，FN 是假反例。因为 F2 更在意召回，所以宁可多预测一些标签，也不要漏掉真实存在的标签。比赛对每张图片分别计算 F2，然后对所有图片求平均，平均值越高越好。

这个指标对多标签任务很合适：如果一张图片里有 10 个标签，模型预测出 9 个并漏掉 1 个，召回率会明显下降，F2 也会下降；如果模型预测了 20 个标签，其中 10 个正确，精确率会下降，但召回率高，F2 仍能保持。材料中提供了 PyTorch 和 Keras 的 F2 实现代码，说明评估过程必须对齐。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先列出所有 CSV 文件，了解数据规模。

```python
import os
from glob import glob

files = glob('../input/inclusive-images-challenge/*.csv')
for f in files:
    df = pd.read_csv(f, nrows=5)
    print(f, df.columns.tolist())
```

可以看到有 `train_bounding_boxes.csv`、`classes-trainable.csv`、`train_human_labels.csv`、`train_machine_labels.csv`、`class-descriptions.csv` 等。每个文件都要看前几行才能确认列名。

接着我们统计可训练类别数。

```python
classes = pd.read_csv('../input/classes-trainable.csv', index_col=0)
print('可训练类别数:', classes.shape[0])  # 7178
```
调参集只有 1,000 张图片。

```python
tuning = pd.read_csv('../input/tuning_labels.csv', index_col=0,
                     header=None, names=['image_id', 'labels'])
print('调参图片数:', tuning.shape[0])
```

人类标注的数据规模更大。

```python
train = pd.read_csv('../input/train_human_labels.csv', usecols=['ImageID', 'LabelName'])
print('图片数:', train.ImageID.nunique(), 
      '标签数:', train.LabelName.nunique())
```
输出是 1,743,042 张图片、18,155 个唯一标签，说明人类标注的标签集合比可训练类别更大，这是正常的，因为部分标签用于未来扩充或评估。

我们还通过 BigQuery 查到了 Open Images 的图片 URL。

```python
from bq_helper import BigQueryHelper

open_images = BigQueryHelper(active_project="bigquery-public-data",
                             dataset_name="open_images")
query = """
SELECT image_id, original_url
FROM `bigquery-public-data.open_images.images`
WHERE image_id IN (SELECT image_id FROM `bigquery-public-data.open_images.labels`)
LIMIT 100
"""
urls = open_images.query_to_pandas_safe(query)
```
有了 URL，我们可以把图片下载下来，配合标签描述做可视化。材料里还用了边界框数据，分析每个类别在图片中的位置分布。边界框有置信度，置信度低的样本可能属于噪声。

```python
boxes = pd.read_csv('../input/inclusive-images-challenge/train_bounding_boxes.csv')
print(boxes['LabelName'].value_counts().head(20))
```

我们还统计了每个类别在图片中的出现频率，材料指出前三个最常见的类别是 Person、Clothing、Human Face，这三个类别几乎占了所有图片的一大部分。标签分布是典型的长尾分布，大多数类别只出现在几百张图片里。

```python
dcounts = train.groupby('LabelName')['ImageID'].nunique() / train.ImageID.nunique()
dcounts_df = pd.DataFrame({'label': dcounts.index, 'pct_of_images': dcounts.values})
print(dcounts_df.sort_values('pct_of_images', ascending=False).head(10))
```

另外，我们统计了每张图片包含的标签数量。

```python
images = train.groupby('ImageID').count()
images.columns = ['LabelCount']
images['LabelCount'].hist(bins=50)
```
大多数图片标签数在 10 个以内，但也有图片标签非常多，说明数据集的复杂性很高。

### 2.2 数据清洗
我们首先把 tuning 标签的标签字符串转换成多热矩阵。

```python
import numpy as np
from collections import defaultdict

# 先统计调参集中出现的标签
tuning_labels_freq = defaultdict(int)
for r in tuning['labels']:
    for l in r.split():
        tuning_labels_freq[l] += 1

tuning_labels_list = list(tuning_labels_freq.keys())
print('调参集唯一标签数:', len(tuning_labels_list))
```

材料里调参集中只有 484 个唯一标签，远小于可训练类别 7,178 个，这是为了让本地调参更快。

```python
# 建立标签到索引的映射
label_2_idx = {v: i for i, v in enumerate(tuning_labels_list)}
idx_2_label = {i: v for v, i in label_2_idx.items()}

# 把标签串转成 one-hot 向量
def caption_2_one_hot(caption, n_classes=len(tuning_labels_list)):
    y = np.zeros(n_classes)
    for w in caption.split():
        if w in label_2_idx:
            y[label_2_idx[w]] = 1
    return y
```

图像数据清洗的主要工作是处理缺失的 URL、无法下载的图片以及类别描述的缺失。

```python
descrips = pd.read_csv('../input/class-descriptions.csv',
                       names=['LabelName', 'Description'])
train = train.merge(descrips, how='left', on='LabelName')
```
合并后可以把机器可读的标签代码转为人类可读的描述，方便分析错误样例。

我们还去掉了重复标注。材料中提到 Kaggle 官方评估会惩罚重复标签，所以对每个图片的标签集合做去重。

```python
train = train.drop_duplicates(subset=['ImageID', 'LabelName'])
tuning = tuning.drop_duplicates(subset=['image_id'])
```

对于边界框数据，我们过滤掉置信度低于 0.1 的框，因为这类框很可能是误检。

```python
low_confidence = boxes[boxes['Confidence'] < 0.1]
print('低置信度框数量:', low_confidence.shape[0])
boxes = boxes[boxes['Confidence'] >= 0.1]
```
这种过滤可以减少噪声带来的负面影响。

### 2.3 模型训练与超参数调优
基线模型采用 Keras 中的 DenseNet121。

```python
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

base = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
# 顶层接 sigmoid，多标签任务专用
out = Dense(N_CLASSES, activation='sigmoid')(base.output)
model = Model(base.input, out)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')
```

因为训练数据量很大，我们使用数据生成器动态加载图片。

```python
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def ImageDataGen(ids, df, lookup_dict=label_2_idx,
                 n_classes=N_CLASSES, img_dir=IMG_DIR,
                 input_size=224, bs=8):
    while True:
        for start in range(0, len(ids), bs):
            x_batch, y_batch = [], []
            sample = ids[start:start+bs]
            for img_id in sample:
                # 从 BigQuery 获取 URL，下载到本地
                img_url = df.loc[img_id, 'url']
                img = cv.imread(img_url)
                img = cv.resize(img, (input_size, input_size))
                img = preprocess_input(img.astype(np.float32))
                x_batch.append(img)
                y_batch.append(caption_2_one_hot(df.loc[img_id, 'Caption']))
            yield np.array(x_batch), np.array(y_batch)
```

训练分两个阶段。第一阶段冻结主干，只训练顶层分类器，用较大的学习率。

```python
for layer in base.layers:
    layer.trainable = False
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')
model.fit(train_gen, validation_data=val_gen, epochs=5)
```

第二阶段解冻部分层，做微调。

```python
for layer in base.layers[-20:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy')
model.fit(train_gen, validation_data=val_gen, epochs=10)
```

为了应对长尾分布，我们可以在损失函数里加入类别权重。材料没有明确给出权重计算方式，但可以按类别出现频率的倒数计算。

```python
class_weights = compute_class_weight('balanced', 
                                     classes=np.arange(N_CLASSES),
                                     y=np.argmax(y_train, axis=1))
```

阈值选择也是超参数。默认阈值 0.5 可能不是最优的，我们可以在验证集上尝试 0.3 到 0.7 之间的阈值，挑出使平均 F2 最高的那个。

```python
best_thresh = 0.5
best_f2 = 0
for t in np.arange(0.3, 0.71, 0.05):
    pred = (proba > t).astype(int)
    f2 = fbeta_score(y_true, pred, beta=2, average='samples')
    if f2 > best_f2:
        best_f2, best_thresh = f2, t
```

材料还提到可以把人类标注、机器标注和边界框标注融合，以获得更完整的标签集合。这种融合需要做标签对齐，确保同一类别在不同标注来源中名称一致。

### 2.4 模型测试与指标报告
我们用调参集计算 F2。

```python
from sklearn.metrics import fbeta_score

y_prob = model.predict(val_gen)
y_pred = (y_prob > best_thresh).astype(int)
f2 = fbeta_score(y_true, y_pred, beta=2, average='samples')
print('Validation mean samples F2:', f2)
```

PyTorch 版的 F2 实现也类似。

```python
import torch

def f2_score(pred, target, beta=2, eps=1e-12):
    tp = (pred * target).sum(dim=0)
    fp = (pred * (1 - target)).sum(dim=0)
    fn = ((1 - pred) * target).sum(dim=0)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
    return f2.mean(0)
```

在真实测试集上，我们需要按提交样例的顺序输出。

```python
submission = pd.read_csv('../input/stage_1_sample_submission.csv')
submission['labels'] = ''

for i, img_id in enumerate(submission['ImageID']):
    probs = model.predict([img_id])
    labels = [idx_2_label[j] for j, p in enumerate(probs[0]) if p > best_thresh]
    submission.loc[i, 'labels'] = ' '.join(set(labels))  # 去重

submission.to_csv('submission.csv', index=False)
```

我们还检查提交文件的行数是否与样例一致，标签是否为空，确保格式正确。

## 三、总结

本文中我们完成了 Inclusive Images Challenge 的多标签图像分类流程。我们首先通过 EDA 确认可训练类别 7,178 个、人类标注图片 1,743,042 张、调参图片 1,000 张，并观察到标签呈长尾分布，常见类别集中在 Person、Clothing、Human Face。其次我们把调参标签转成多热矩阵，对边界框做置信度过滤，并对重复标签去重。接着我们使用预训练 DenseNet121，采用分阶段解冻策略进行训练，并在验证集上调优阈值以最大化 F2。最后我们按样例格式生成提交文件。

这个比赛让我们认识到，地域偏差和多标签标注是图像识别中的两个重要挑战。F2 指标更重视召回，意味着模型不能漏掉真实标签；数据来源多样、标注质量不均，需要在特征构建和清洗上投入更多精力。融合多种标注来源和对阈值的精细调节，常常比单纯堆网络结构更有效。
