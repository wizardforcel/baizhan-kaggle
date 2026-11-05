# iMaterialist Challenge Fashion 2018

## 一、比赛简介

### 1.1 比赛背景
随着购物越来越多地转到网上，商品图片自动识别很有价值。相同商品可能有不同的光照、角度、背景，还可能被遮挡；颜色和款式也经常非常接近，例如皇家蓝和绿松石色。普通图像分类器很容易把这些细微差别混淆。iMaterialist Challenge（Fashion）是 FGVC5 研讨会的一项细粒度视觉分类任务，由 CVPR 与 Google、Wish、Malong Technologies 合作举办。

比赛面对的是服装图片的多标签识别：一张图可能同时包含颜色、服装类别、材质、款式等多个属性。材料中的探索同时使用了图像颜色分析和目标检测，说明任务不仅是把整张图分到一个类别，还要理解图中有哪些服装属性。

### 1.2 比赛目标
训练数据、测试数据和验证数据分别存放在 `train.json`、`test.json`、`validation.json` 中。每个 JSON 里都有 `images`，训练和验证还包含 `annotations`。每条图像记录包含 `imageId` 和图片 URL；标注里包含同一个 `imageId` 对应的 `labelId` 列表。

材料报告的规模是：训练图片 1,014,544 张，测试图片 39,706 张，验证图片 9,897 张；训练集中有 228 个不同标签。我们的任务是把每张测试图片的多个属性标签预测出来，并按照提交格式输出。

### 1.3 评价标准
这是一个多标签分类比赛，一张图片可以有多个正确标签。材料提到官方评估会惩罚重复标签，重复的预测会影响 F-score，因此提交时不能简单地把同一标签重复写入。实际比较的是预测标签集合与真实标签集合的重合程度，预测越准确、越完整越好。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先把 JSON 读进来，检查三个数据集的大小。

```python
import time
import json
import pandas as pd
import numpy as np

script_start_time = time.time()
data_path = "../input"

# 读取 JSON，先用原生 json 保留多标签列表
with open(f"{data_path}/train.json") as f:
    train = json.load(f)
with open(f"{data_path}/test.json") as f:
    test = json.load(f)
with open(f"{data_path}/validation.json") as f:
    validation = json.load(f)

print(len(train['images']), len(test['images']), len(validation['images']))
```
材料里输出的图片数量分别是 1,014,544、39,706 和 9,897。训练集和验证集带有标签，测试集只有图片信息。

接着我们把嵌套 JSON 变成表格。

```python
# images 表保存 URL，annotations 表保存标签
train_img = pd.DataFrame(train['images'])
train_ann = pd.DataFrame(train['annotations'])
train_df = pd.merge(train_img, train_ann, on='imageId', how='inner')

valid_img = pd.DataFrame(validation['images'])
valid_ann = pd.DataFrame(validation['annotations'])
valid_df = pd.merge(valid_img, valid_ann, on='imageId', how='inner')

test_df = pd.DataFrame(test['images'])
```
这样每行就对应一张图片和它的标签列表，后续可以像处理普通 DataFrame 一样统计。

我们还看每个标签出现次数和每张图的标签数。

```python
from collections import Counter

# 把每个图片的列表摊平，统计标签频率
all_labels = []
for labels in train_df['labelId']:
    all_labels.extend(labels)
label_counts = Counter(all_labels)
print('标签数:', len(label_counts))

# 一张图的标签越多，说明多标签问题越明显
train_df['label_count'] = train_df['labelId'].apply(len)
print(train_df['label_count'].describe())
```
材料指出训练集有 228 个不同标签，并且还分析了共现标签、最多标签图片和单标签图片。我们发现标签并不是均匀出现的，少数标签很常见，长尾标签比较少。

### 2.2 数据清洗
我们其次处理标签格式，把标签列表转换成模型需要的多热矩阵。多热矩阵就是每个标签对应一列，图片有这个标签就填 1，没有就填 0。

```python
# 先收集所有标签并建立稳定的编号
unique_labels = sorted(set(all_labels))
label_to_idx = {v: i for i, v in enumerate(unique_labels)}

def labels_to_multi_hot(labels):
    # 多标签任务不能只保留一个类别，所以用向量记录所有标签
    y = np.zeros(len(unique_labels), dtype=np.float32)
    for label in set(labels):
        y[label_to_idx[label]] = 1.0
    return y

Y = np.vstack(train_df['labelId'].map(labels_to_multi_hot))
```
我们使用 `set` 去重，是因为材料明确提到重复标签会影响官方 F-score。图片 URL 也要保留原样，因为它是下载图像的唯一线索，不应该像普通文本那样去掉符号。

为了加快读取，我们把三个数据集里的 `imageId` 转成无符号整数，并只保留需要的列。

```python
# imageId 只用来连接图片和标注，转成整数可以节省内存
for df in [train_df, valid_df, test_df]:
    df['imageId'] = df['imageId'].astype(np.uint32)

train_df = train_df[['imageId', 'url', 'labelId']]
valid_df = valid_df[['imageId', 'url', 'labelId']]
```
### 2.3 模型训练与超参数调优
我们接着尝试材料里的两条思路。第一条是使用图片的平均颜色和主色调作为简单特征，第二条是用目标检测器找图片里的物体，再把检测到的对象作为辅助标签。目标检测输出 bounding box、类别和置信度，置信度阈值在材料示例中使用了 0.1。

```python
# 从检测器中取出框、分数和类别
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores,
     detection_classes, num_detections],
    feed_dict={image_tensor: image_np_expanded}
)

objects = []
for index in range(int(num[0])):
    # 过滤低置信度目标，避免噪声污染标签
    if float(scores[0, index]) > 0.1:
        objects.append(int(classes[0, index]))
```
多标签模型可以使用 DenseNet121 作为图像骨干网络。材料里的 Keras baseline 用 224×224 输入、批大小 8，并将调试数据拆成训练、验证、测试三部分。

```python
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.layers import Dense
from keras.models import Model

# 使用预训练网络提取图像特征，最后一层输出每个标签的概率
base = DenseNet121(include_top=False, weights='imagenet', pooling='avg')
out = Dense(len(unique_labels), activation='sigmoid')(base.output)
model = Model(base.input, out)
model.compile(optimizer='adam', loss='binary_crossentropy')
```
使用 `sigmoid` 而不是 `softmax`，是因为一张图片可以同时属于多个标签。调参时我们优先确认输入尺寸、批大小和检测置信度阈值，再调整学习率和冻结层数；材料没有给出一个统一的最终参数，因此我们把它作为 baseline。

### 2.4 模型测试与指标报告
我们在验证集上将概率转换为标签集合，并删除重复项。

```python
# 概率超过阈值才认为标签存在；阈值应在验证集上调节
proba = model.predict(valid_images)
pred_labels = []
for row in proba:
    ids = np.where(row > 0.5)[0]
    # 保持每张图的标签唯一，避免重复标签扣分
    pred_labels.append(sorted(set(unique_labels[i] for i in ids)))
```
材料中的评估示例把真实标签转换为布尔矩阵，再使用 F-score 计算预测质量。我们可以按图片计算 precision、recall 和 F2，并取平均。

```python
from sklearn.metrics import fbeta_score

# 将预测标签还原为多热矩阵
pred_matrix = np.zeros_like(Y[:len(pred_labels)])
for i, labels in enumerate(pred_labels):
    for label in labels:
        pred_matrix[i, label_to_idx[label]] = 1

# beta=2 更重视召回，适合不漏掉服装属性的场景
score = fbeta_score(Y[:len(pred_labels)], pred_matrix,
                    beta=2, average='samples')
print('validation F2:', score)
```
最后我们用全量训练数据重新拟合，并按照测试集 `imageId` 顺序生成提交文件。由于材料里没有提供最终排行榜数值，我们不虚构分数，只报告验证集上按照官方思路计算的 F2 和标签去重处理。

## 三、总结

本文中我们完成了 iMaterialist Challenge Fashion 2018 的多标签图像识别流程。我们首先读取了 train、test、validation 三个 JSON，确认训练图片约 101 万张、测试图片 39,706 张、验证图片 9,897 张，并统计出 228 个标签。其次我们把图片和标注合并成表格，使用多热向量表示一张图片的多个属性，同时对标签去重。接着我们从颜色、目标检测和 DenseNet121 多标签网络三个方向构造基线，使用 sigmoid 输出多个标签概率，并用 0.1 的检测置信度阈值过滤目标。最后我们把概率转换成唯一标签集合，使用样本级 F2 评估，并按 `imageId` 生成提交。

这个比赛说明，细粒度服装识别不能只看一个类别。颜色、物体位置、标签共现和图像分类器需要结合起来，数据格式转换和标签去重也会直接影响成绩。
