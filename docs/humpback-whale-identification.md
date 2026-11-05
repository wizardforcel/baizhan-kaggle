# Humpback Whale Identification
## 一、比赛简介
### 1.1 比赛背景
Humpback Whale Identification 是 Kaggle 上的图像识别比赛，任务是根据座头鲸尾鳍照片识别个体身份。HappyWhale 项目收集了大量的鲸鱼照片，每个个体由其尾鳍花纹唯一识别。比赛背景材料来自 `Happywhale EDA`，训练集约 5 万张图片，涉及多种鲸种，其中座头鲸是主要类别。数据的特点是存在大量重复照片：同一尾鳍被不同角度、不同光照拍摄多次。材料中 `Whale Recognition Model with score 0.78563` 提到训练集和测试集都存在重复图像，处理重复对模型效果至关重要。
### 1.2 比赛目标
训练文件 `train.csv` 包含三列：`image`、`species`、`individual_id`。其中 `image` 是文件名，`individual_id` 为个体标识。目标是根据图像预测个体 id。测试集只有图像文件名，需要提交每个测试图像对应的 `individual_id`。比赛要求对同一尾鳍的不同照片归为同一类，属于细粒度图像识别任务。
### 1.3 评价标准
比赛使用平均精度均值（Mean Average Precision, MAP）作为评价指标。MAP 越高越好，它衡量模型在所有类别上的排序能力。对于每个查询图像，模型给出所有个体的相似度分数，排名越靠前越好。材料中最好的模型分数达到 0.78563，对应公共榜约 78% 的 MAP。

## 二、解题过程
### 2.1 数据探索（EDA）
我们首先读取训练表并查看物种分布。
```python
import pandas as pd
df_train = pd.read_csv("../input/happy-whale-and-dolphin/train.csv")
print(df_train.info())
# RangeIndex: 51033 entries
# 3 columns: image, species, individual_id
```
`image` 列共 51033 张训练图，测试图 27956 张。物种统计显示：
```python
animal_cnt = df_train.species.value_counts()
print(animal_cnt.head(10))
```
材料中出现次数最多的物种是 `bottlenose_dolphin` 10781 张，`beluga_whale` 7443 张，`humpback_whale` 7392 张。比赛聚焦座头鲸，但数据中包含多种鲸种，需要先筛选。
```python
# 筛选出座头鲸
df_whale = df_train[df_train.species == 'humpback_whale']
print(len(df_whale))
```
我们还检查图像重复情况。材料提到同一 `individual_id` 存在多张照片，平均每个个体约 5 张。
### 2.2 数据清洗
清洗的第一步是去重。材料中 `Whale Recognition Model with score 0.78563` 详细描述了重复检测方法：使用感知哈希 phash 判断两张图是否相同，若 phash 相同或仅相差 6 位且尺寸相同且归一化后的均方误差小于 0.1，则视为重复。
```python
from imagehash import phash
from PIL import Image
import numpy as np

def is_duplicate(img1, img2):
    h1 = phash(img1)
    h2 = phash(img2)
    # 判断哈希距离和像素误差
    ...
```
我们为每个重复组选出分辨率最高的图像作为代表，其他图像映射到同一 id。这样训练时模型不会被相同尾鳍的不同裁剪误导。
```python
# 构建 p2h 字典：图片 -> 哈希 id，h2p 字典：哈希 id -> 首选图片
...
```
第二步是图像预处理：对尾鳍做裁剪和归一化。材料提到使用边界框模型先定位尾鳍，再裁剪出感兴趣区域，减少背景干扰。
```python
# 仅示例：读取图像并resize
img = plt.imread(train_jpg_path + '/' + example_pic)
plt.imshow(img)
```
### 2.3 模型训练与超参数调优
主模型采用 Siamese 神经网络。Siamese 网络学习图像间的相似度而非直接分类，适合个体识别这种开集问题。
```python
# 伪代码：构建孪生网络
def build_model(base_model, input_shape):
    input_a = Input(input_shape)
    input_b = Input(input_shape)
    features_a = base_model(input_a)
    features_b = base_model(input_b)
    distance = Lambda(lambda x: K.abs(x[0]-x[1]))([features_a, features_b])
    output = Dense(1, activation='sigmoid')(distance)
    return Model([input_a, input_b], output)
```
材料中 `Data Augmentation Xception model (~0.7889)` 使用 Xception 作为骨干，配合数据增强：随机翻转、旋转、亮度变化、裁剪，得到公共分数约 0.7889。另一篇材料使用 ResNet-50 并加入数据增强，分数约 0.78563。
训练数据构造是关键：材料强调每周期生成图像对 (A,B)，50% 为同个体、50% 为不同个体，且不同个体的配对要选择“难样本”。具体实现是：
```python
# 伪代码：难负样本挖掘
for epoch in range(N_EPOCHS):
    pairs = []
    for _ in range(batch_size):
        if random() < 0.5:
            # 正样本：同个体
            id = random.choice(ids)
            img_a, img_b = sample_two_images(id)
        else:
            # 负样本：不同个体，但特征接近
            id1, id2 = sample_hard_negative_pair(model)
        pairs.append((img_a, img_b, label))
```
这种自适应负样本选择让模型更关注容易混淆的个体，提升 MAP。数据增强方面，材料中提到使用 `ImageDataGenerator` 做水平翻转、随机裁剪、亮度 jitter。
```python
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8,1.2]
)
```
### 2.4 模型测试与指标报告
在测试阶段，我们不能直接做分类，而是为每个测试图像计算它与训练集中所有个体的相似度，取最相似的个体作为预测。材料中 `Whale Recognition Model with score 0.78563` 使用模型嵌入向量做余弦相似度，并通过集成多个模型提升稳定性。
```python
# 生成嵌入
train_embeddings = model.predict(train_images)
test_embeddings = model.predict(test_images)

# 对每个测试图像找到最近邻
for i, emb in enumerate(test_embeddings):
    sims = cosine_similarity(emb, train_embeddings)
    nearest_id = train_ids[np.argmax(sims)]
```
最终生成提交文件：
```python
submission = pd.DataFrame({
    'Image': test_df['image'],
    'Id': predictions
})
submission.to_csv('submission.csv', index=False)
```
材料报告的分数：Xception + 数据增强模型约 0.7889，Siamese ResNet 模型 0.78563。数据清洗后重复图像的数量减少约 10%，MAP 提升明显，说明去重对细粒度识别效果显著。

## 三、总结
本文中我们完成了 Humpback Whale Identification 的完整流程。我们首先通过 EDA 确认了 5 万张多物种图像的分布，筛选出座头鲸并发现大量重复照片；其次使用了感知哈希去重并保留高分辨率代表图像，同时对尾鳍做裁剪预处理；接着采用 Siamese 网络配合难负样本挖掘和 Xception/ResNet 骨干训练相似度模型，并加入数据增强；最后通过最近邻搜索生成个体 id 预测。整个过程体现了细粒度图像识别的核心点：去重、定位、相似度学习和难样本挖掘。
