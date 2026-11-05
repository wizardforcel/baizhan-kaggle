# iMaterialist Challenge Fashion 2018

## 一、比赛简介

### 1.1 比赛背景
随着电商的发展，消费者在线购物时大量依赖商品图片。自动对图片中的服装做细粒度属性识别，对推荐系统、搜索排序以及反抄袭都有帮助。与普通的「把整张图片归到一个类别」不同，服装识别要处理一个非常棘手的问题：同一个商品在不同灯光、角度、背景下长相差异很大；不同的商品却可能几乎一模一样，比如皇家蓝和绿松石色在照片里很难区分，还有不同风格的同一件衣服在不同模特身上看起来完全不同。

iMaterialist Challenge（Fashion）是 CVPR 在其细粒度视觉分类工作坊 FGVC5 中举办的一项比赛，由 CVPR 合作伙伴 Google、Wish 和 Malong Technologies 共同发起。比赛的定位不是做一个通用分类器，而是专门针对服装属性，做细粒度、多标签的图像识别。参赛者需要在接近百万张服装图片上识别属性标签，最终选拔的结果会在 FGVC5 工作坊现场展示。

### 1.2 比赛目标
比赛采用 Wish 提供的大规模服装图片数据集。数据以 JSON 格式提供，三个文件分别是 `train.json`、`test.json`、`validation.json`。每个 JSON 里包含两个核心结构：`images` 记录图片的 URL 和 ID，`annotations` 记录每个图片与标签的对应关系。训练数据还额外包含一个样本提交文件，用于对接答案格式。

在材料中，训练图片数量是 1,014,544，张，测试图片数量是 39,706 张，验证图片数量是 9,897 张；训练集中使用到的不同标签总数为 228 个。每张图片可能对应多个标签，例如颜色、款式、材质、部位。标签以 `labelId` 的形式给出，真实标签描述在 `class-descriptions.csv` 中可以查询。

任务的目标就是对于测试图片，预测出属于 228 个细粒度类别中的哪一些。我们需要把每张图片映射到一个标签集合，而不是单一类别。

### 1.3 评价标准
这是一项多标签分类比赛：一张图片可能同时属于多个类别，例如「连衣裙」「红色」「夏季」。因此普通的准确率会高估/低估效果。官方评测使用的是以 F-score 为基础的指标，并对重复标签有惩罚。材料中提到的“f-score”受重复标签影响，说明评价会在每个图片上计算预测集合与真实集合的重合程度，并综合精确率和召回率。

为了鼓励全面覆盖真实标签，比赛会更看重召回率。材料里提到的彩色分析、目标检测都说明了一个事实：如果只看全局图像分类，模型很难捕捉到小面积的服装物体，比如图片里有人、背景也很杂。评价指标对漏掉真实标签很敏感，所以构建特征时必须兼顾全局和局部线索。

## 二、解题过程

### 2.1 数据探索（EDA）
首先我们需要把 JSON 文件读成容易处理的表格格式。

```python
import time
import json
import pandas as pd
import numpy as np

script_start_time = time.time()
data_path = "../input"

# 先用原生 json 读取多层嵌套结构，避免 pandas 读出现问题
with open(f"{data_path}/train.json") as f:
    train = json.load(f)
with open(f"{data_path}/test.json") as f:
    test = json.load(f)
with open(f"{data_path}/validation.json") as f:
    validation = json.load(f)

print("Train images:", len(train['images']))
print("Test images:", len(test['images']))
print("Validation images:", len(validation['images']))
```
材料这里打印出 1,014,544 张训练图、39,706 张测试图和 9,897 张验证图。这个规模已经接近百万级，是一个典型的工业级多标签任务。

接着我们把 `images` 与 `annotations` 用 `imageId` 合并。

```python
# images 里记录 URL，annotations 里记录 labelId
train_img = pd.DataFrame(train['images'])
train_ann = pd.DataFrame(train['annotations'])
train_df = pd.merge(train_img, train_ann, on='imageId', how='inner')

valid_img = pd.DataFrame(validation['images'])
valid_ann = pd.DataFrame(validation['annotations'])
valid_df = pd.merge(valid_img, valid_ann, on='imageId', how='inner')

# 测试集只有图片，没有标签
test_df = pd.DataFrame(test['images'])
```

合并后，我们会得到类似 `imageId, url, labelId` 的三列表格。再对 `imageId` 转为无符号整数，可以略微减少内存占用。

```python
for df in [train_df, valid_df, test_df]:
    df['imageId'] = df['imageId'].astype(np.uint32)

print("训练数据前几行：")
train_df.head()
```
据材料统计，训练集里共有 228 个不同标签。接下来我们统计每个标签出现的次数以及每张图包含的标签数。

```python
from collections import Counter

# 把每张图片的标签列表摊平，统计标签频率
all_labels = []
for labels in train_df['labelId']:
    all_labels.extend(labels)
label_counts = Counter(all_labels)
print("标签总数:", len(label_counts))

# 每张图片的标签数量分布
train_df['label_count'] = train_df.groupby('imageId')['labelId'].transform('count')
print(train_df['label_count'].describe())
```
我们可以发现标签分布极不均匀，有些热门标签出现上万次，冷门标签可能只有几百次出现；同时大部分图片的标签数在 1 到 10 个之间，少量图片标签很多。

我们还进一步分析了标签共现。材料中有一条“最常见的共现标签”，这在多标签任务里很重要：很多服装属性是绑定的，如「红色」和「连衣裙」常常同时出现。知道共现关系可以帮助模型做标签间的正则化。

```python
# 计算标签间的共现矩阵
from itertools import combinations

co_occur = Counter()
for labels in train_df.groupby('imageId')['labelId'].apply(list):
    for a, b in combinations(set(labels), 2):
        co_occur[(a, b)] += 1

most_common_pair = co_occur.most_common(10)
print(most_common_pair)
```
为了理解图像内容，材料中还做了颜色分析和目标检测。颜色分析会从图片中提取主色调，目标检测会用一个预训练的检测器找出图片里的人和服装框，再统计不同类别的出现频次。两类特征都是为了解决细粒度分类时「整体分类器看不清局部物体」的问题。

### 2.2 数据清洗
清洗的第一步是去重和稳定化标签编号。

```python
# 去掉可能存在的重复标注，同一个图片同一个标签只保留一次
train_df = train_df.drop_duplicates(subset=['imageId', 'labelId'])
valid_df = valid_df.drop_duplicates(subset=['imageId', 'labelId'])
```

由于训练集有 228 个标签，而很多工具需要连续编号，我们建立标签到索引的映射。

```python
unique_labels = sorted(set(all_labels))
label_to_idx = {v: i for i, v in enumerate(unique_labels)}
idx_to_label = {i: v for v, i in label_to_idx.items()}
```

多标签任务里，常用的表示是多热向量，就是一个长度为标签数的向量，标签存在就填 1，其余都是 0。

```python
def labels_to_multi_hot(labels):
    y = np.zeros(len(unique_labels), dtype=np.float32)
    for label in set(labels):
        y[label_to_idx[label]] = 1.0
    return y

# 把每个图片的标签变成向量
X_train = train_df[['imageId', 'url']].drop_duplicates()
y_train = X_train['imageId'].apply(lambda id_: labels_to_multi_hot(
    train_df[train_df['imageId']==id_]['labelId'].tolist()))
```

JSON 转成多标签矩阵这个过程非常容易出错。材料里有个“24 秒转换”的技巧：直接用 pandas merge 保留关联，然后用 `groupby` 收集标签列表，避免多次循环。

对图片 URL，我们不做正则清理，只是保留原样。因为 URL 直接用于下载图片，任何改动都会导致链接失效。在清洗时我们不会删除标点符号，也不做转小写，保持原始状态。

### 2.3 模型训练与超参数调优
模型训练的核心思路是把图像分类和辅助特征结合。材料中展示了两条互补的路线：

第一条是颜色和目标检测特征。颜色特征捕捉服装主色调，检测特征捕捉图片里的具体物体。下面演示如何从目标检测器中提取物体标签。

```python
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import requests

sess = tf.Session()
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('faster_rcnn_inception_v2_coco', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes  = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# 对一张图片推理
image_np = np.array(Image.open(BytesIO(requests.get(url).content)).convert('RGB'))
image_np_expanded = np.expand_dims(image_np, axis=0)
boxes, scores, classes, num = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_np_expanded})

objects = []
for i in range(int(num[0])):
    if float(scores[0, i]) > 0.1:   # 过滤低置信度结果
        objects.append(int(classes[0, i]))
```

第二条路线是基于预训练网络的端到端多标签分类。材料里使用 DenseNet121 作为骨干。

```python
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

# 用 ImageNet 预训练权重启动
base = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
# 在顶层加 sigmoid 输出层，适配多标签
x = Dense(len(unique_labels), activation='sigmoid')(base.output)
model = Model(base.input, x)
model.compile(optimizer=Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])
```

为了避免在百万图像规模上一次性加载所有图片，我们使用图像数据生成器动态读取 URL 并裁剪。

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def my_generator(df, batch_size=32, is_train=True):
    while True:
        indices = np.random.permutation(len(df))
        for i in range(0, len(df), batch_size):
            batch = df.iloc[indices[i:i+batch_size]]
            images, labels = [], []
            for _, row in batch.iterrows():
                # 下载、解码、resize 到 224x224
                img = Image.open(BytesIO(requests.get(row['url']).content)).convert('RGB')
                img = img.resize((224, 224))
                images.append(np.array(img))
                labels.append(row['multi_hot'])
            yield np.array(images), np.array(labels)
```

训练时我们逐步解冻网络。先只训练顶层的全连接层，用较大学习率；随后解冻最后几块 DenseNet，用较小学习率微调。这样可以在不忘掉 ImageNet 通用特征的前提下，学习服装细节。

```python
# 阶段 1：冻结主干，只训练顶层
for layer in base.layers:
    layer.trainable = False
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')
model.fit(train_generator, validation_data=valid_generator,
          epochs=5, steps_per_epoch=1000)

# 阶段 2：解冻部分层，微调
for layer in base.layers[-30:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy')
model.fit(train_generator, validation_data=valid_generator,
          epochs=10, steps_per_epoch=1000)
```

超参数方面，我们主要调节 `learning_rate`、`batch_size` 和图像尺寸。显存允许时用 224×224，必要时降到 160×160。阈值选择也是重要超参数：在验证集上尝试 0.3 到 0.7 之间的阈值，挑出平均 F-score 最高的那个。

### 2.4 模型测试与指标报告
我们在验证集上先评估效果。

```python
from sklearn.metrics import f1_score

# 预测概率
y_prob = model.predict(valid_generator)
# 用验证集上确定的最佳阈值
threshold = 0.45
y_pred = (y_prob > threshold).astype(int)
y_true = valid_multi_hot_matrix

# 多标签的样本平均 F1
sample_f1 = f1_score(y_true, y_pred, average='samples')
print("Validation sample F1:", sample_f1)
```

材料还提到数据泄漏问题：验证集里可能出现与训练集高度相似的图片，或者标签重复会导致评估虚高。所以我们先去重，再检查图片来源是否一致。

在生成提交前，我们还做了一轮模型检查。材料里的 EDA 指出，图片下载失败是百万级数据集最常见的问题之一，因此我们记录每个 URL 的下载状态，只对成功下载的图片做推理，失败图片则回退到「用同类商品的平均预测」或者干脆输出空标签。这样做虽然会损失一点覆盖率，但可以避免因一张坏图导致整个批次报错。

另一个检查点是标签范围。训练集里可能有一些标签从未出现在 `classes-trainable.csv` 中，预测时要过滤掉这些非法标签，否则提交会被判为格式错误。我们用一个集合做白名单校验：

```python
valid_labels = set(unique_labels)

# 保证每个预测标签都在可提交类别里
def sanitize(labels):
    return [l for l in set(labels) if l in valid_labels]
```

最后生成提交文件。

```python
# 对测试集做推理
test_probs = model.predict(test_generator)
test_pred_ids = (test_probs > threshold).astype(int)
# 把概率矩阵转成标签列表
submission = []
for i, row in enumerate(test_pred_ids):
    labels = [idx_to_label[j] for j, v in enumerate(row) if v == 1]
    submission.append({'imageId': test_ids[i], 'labels': ' '.join(map(str, labels))})

sub_df = pd.DataFrame(submission)
sub_df.to_csv('submission.csv', index=False)
```

提交文件必须保证 `imageId` 的顺序与样例文件一致，标签之间用空格分隔，不能有重复。重复标签会扣分，材料里特别强调要做 `set(labels)` 去重。

## 三、总结

本文中我们完成了 iMaterialist Challenge Fashion 2018 的多标签服装属性识别流程。我们首先通过 EDA 确认训练图片约 101 万张、测试 39,706 张、验证 9,897 张，标签数 228 个，并观察到标签分布长尾且共现明显。其次我们把 JSON 数据合并成表格，对标签去重并构建多热向量表示。接着我们采用了两种互补方案：基于目标检测和颜色的主动特征提取，以及基于 DenseNet121 的端到端多标签分类，并采用分阶段解冻策略训练。最后我们在验证集上用样本平均 F1 调整阈值，并按样例格式生成去重的提交文件。

这个比赛告诉我们，细粒度视觉分类不能只依赖全局分类器。颜色、物体检测等辅助特征可以弥补网络对局部细节的缺失；多标签任务下对阈值和重复标签的处理，会直接决定最终分数。数据清洗和标签去重同样重要，重复标签在官方评估里会被惩罚。
