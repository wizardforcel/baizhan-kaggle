# Yelp 餐厅照片分类（Yelp Restaurant Photo Classification）

## 一、比赛简介

### 1.1 比赛背景
Yelp 是一个让大家点评餐厅、商店和各类本地商户的平台。商户主页上通常挂着很多照片，顾客会拍菜品、拍店内环境、拍招牌，商户自己也会上传宣传照。人看一眼照片就能大致判断“这家店适合吃午饭”“这家店有户外座位”“这家店看起来很贵”，但机器并不知道这些信息。

这个比赛做的事情就是：给一家餐厅一批照片，让模型从照片里自动推断出这家餐厅的 9 个属性。这些属性不是照片级别的，而是商户级别的。也就是说，同一家餐厅的所有照片共享同一组标签，模型需要从这一组照片中学习视觉线索，再预测新商户的标签组合。

比赛数据来自 Yelp 的真实商户照片。材料里显示训练集有 2000 家商户，训练照片共 234842 张，测试照片共 237152 张。照片数量远大于商户数量，说明每家商户平均有上百张照片，信息量很足，但也意味着要处理大量图片。

9 个标签的含义是：
- 0：good_for_lunch，适合吃午饭
- 1：good_for_dinner，适合吃晚饭
- 2：takes_reservations，接受订位
- 3：outdoor_seating，有户外座位
- 4：restaurant_is_expensive，餐厅价格昂贵
- 5：has_alcohol，提供酒精饮料
- 6：has_table_service，有餐桌服务
- 7：ambience_is_classy，环境高档
- 8：good_for_kids，适合带孩子

这是一个**多标签分类**问题：一家餐厅可以同时满足多个属性，标签之间也不互斥。比如一家人均消费较高的餐厅，可能同时“适合晚餐”“接受订位”“环境高档”“有酒精饮料”。所以不能用普通的单标签 softmax，而要给 9 个标签各输出一个 0 到 1 的概率。

### 1.2 比赛目标
输入是商户的照片以及照片到商户的映射，输出是每个商户对应的标签集合。训练集包含三个核心文件：
- `train.csv`：两列，`business_id` 和 `labels`，标签是用空格分隔的数字串；
- `train_photo_to_biz_ids.csv`：两列，`photo_id` 和 `business_id`，把每张训练照片挂到商户上；
- `sample_submission.csv`：提交样例，包含 `business_id` 和 `labels`。

测试集提供 `test_photos` 目录和 `test_photo_to_biz.csv`，我们需要为测试商户生成同样的 `business_id,labels` 格式。`labels` 里放预测为真的标签编号，多个标签用空格隔开。

### 1.3 评价标准
材料里的代码使用 `sklearn.metrics` 的 `f1_score`，并同时打印 micro、macro、samples 三种平均方式。F1 是精确率和召回率的调和平均，简单说就是“既要尽可能找全正确答案，也不能乱报”。micro 把所有标签的预测合在一起算，macro 先算每个标签的 F1 再平均，samples 则先算每家商户的 F1 再平均。比赛评价的是标签集合的匹配程度，F1 越高越好。

## 二、解题过程

整场比赛的思路可以概括成一句话：先把照片和商户标签整理成“图片 + 9 维 0/1 向量”的训练样本，再用预训练的卷积神经网络做多标签分类，最后按商户聚合预测结果。我们按 EDA、数据清洗、模型训练与调优、测试与报告四步来走。

### 2.1 数据探索（EDA）
我们首先把三个 CSV 读进来，看看数据规模和标签格式。
```python
import pandas as pd
import numpy as np

# 训练集：商户 -> 标签串
train = pd.read_csv('../input/train.csv')
print('训练集形状:', train.shape)          # 材料里是 (2000, 2)
print('商户数量:', train.shape[0])

# 照片 -> 商户
train_photo_to_id = pd.read_csv('../input/train_photo_to_biz_ids.csv')
print('训练照片数:', train_photo_to_id.shape[0])  # 234842

# 测试照片 -> 商户
test_photo_to_id = pd.read_csv('../input/test_photo_to_biz.csv')
print('测试照片数:', test_photo_to_id.shape[0])   # 237152

train.head()
```
读进来后我们看到 `labels` 不是数字，而是像 `1 2 3 4 5 6 7` 这样的字符串，需要自己切分。训练集只有 2000 行，但照片有 23 万多张，说明这是一份“商户少、照片多”的数据。

接着我们检查标签缺失。
```python
# 统计标签缺失情况
print('缺失标签数量:', train['labels'].isnull().sum())  # 材料里是 4
missing_rows = train[train['labels'].isnull()]
missing_rows
```
只有 4 家商户没有标签，数量很少，后面可以直接丢掉，不会影响整体分布。

我们还随机抽了几张照片，连同商户编号和标签一起显示出来，直观感受一下图像内容。
```python
import os
import random
import cv2
import matplotlib.pyplot as plt

train_dir = '../input/train_photos'
train_imgs = os.listdir(train_dir)

# 随机抽 8 张训练图片
imgs_samples = random.sample(train_imgs, 8)
plt.figure(figsize=(15, 10))
for i, img_name in enumerate(imgs_samples):
    img = cv2.imread(os.path.join(train_dir, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV 默认 BGR，转成 RGB
    # 通过照片编号找到商户，再找到标签
    biz = train_photo_to_id.loc[
        train_photo_to_id['photo_id'] == int(img_name[:-4]), 'business_id'
    ].values[0]
    labels = train.loc[train['business_id'] == biz, 'labels'].values[0]
    plt.subplot(2, 4, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'{img_name}\nBiz:{biz}\nLabels:{labels}')
plt.show()
```
这一步让我们看到照片内容差异很大：有菜品特写、有店内环境、有店招牌、有户外座位。也解释了为什么同一个商户的标签会同时出现多个：照片里既有食物又有环境。

我们还统计了图片尺寸。
```python
def load_img_shapes(path_to_img):
    return cv2.imread(path_to_img).shape

train_shapes = [load_img_shapes(os.path.join(train_dir, f)) for f in train_imgs]
df_train = pd.DataFrame({'Shapes': train_shapes})
train_counts = df_train['Shapes'].value_counts()
print(train_counts.head(10))
```
结果显示图片的宽高并不统一，有的横图、有的竖图，尺寸也各不相同。因此训练前必须统一缩放，否则无法拼成一个 batch 送进神经网络。

### 2.2 数据清洗
数据清洗的第一步是把标签串解析成 9 维 0/1 向量，并把缺失标签的商户去掉。
```python
# 去掉标签缺失的 4 行
train = train.dropna(subset=['labels']).copy()

# 把 "1 2 3" 这样的字符串切成整数列表
train['labs'] = train['labels'].apply(lambda x: [int(v) for v in str(x).split(' ')])

# 生成 9 维 0/1 向量：属于该类记 1，否则记 0
def to_multihot(label_list):
    return [1 if i in label_list else 0 for i in range(9)]

train['labsint'] = train['labs'].apply(to_multihot)
train[['business_id', 'labels', 'labsint']].head()
```
这一步把“文本标签”变成了模型能吃的数字向量，是后续 Dataset 的核心字段。

第二步是把照片表和标签表合并，得到“照片编号 -> 9 维标签”的训练表。
```python
data = pd.merge(train_photo_to_id, train, on='business_id', how='left')
data = data.dropna(subset=['labels'])
print(data.shape)
data.head()
```
合并后每一行就是一张照片和它所属商户的标签。材料里为了调试方便，只取了前 5000 行做小样本实验；真正训练时可以用全量 23 万多张。

第三步是图像预处理。我们把所有图片统一缩放到 256×256，再转成张量并做标准化。
```python
from torchvision import transforms

# ImageNet 预训练模型常用的均值和标准差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# 训练时的增强：翻转、颜色抖动、随机仿射
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.5, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# 验证时只做缩放和标准化，不加随机增强
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```
训练增强是有意为之：照片的拍摄角度、光线、远近差异很大，随机翻转和仿射能让模型不要死记某一种构图。验证集不能增强，否则评估结果每次都不一样，没法比较模型好坏。

### 2.3 模型训练与超参数调优
我们先把数据集封装成 PyTorch 的 `Dataset`，它的作用是按索引返回一张图片和对应的 9 维标签。
```python
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class YelpDataset(Dataset):
    def __init__(self, data_path, dataframe, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.imgs = dataframe['photo_id'].tolist()
        self.annos = dataframe['labsint'].tolist()
        self.classes = list(range(9))

    def __getitem__(self, idx):
        # 照片文件名是 photo_id + .jpg
        img_path = os.path.join(self.data_path, str(self.imgs[idx]) + '.jpg')
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # 把 0/1 列表转成 float 向量，供 BCE 损失使用
        label = np.array(self.annos[idx], dtype=np.float32)
        return img, label

    def __len__(self):
        return len(self.imgs)
```
注意这里返回的是多标签向量而不是单个类别编号，这是多标签任务和普通分类任务在代码上的关键区别。

我们按 9:1 划分训练集和验证集，并使用 ImageNet 预训练的 ResNeXt50 作为骨干网络。
```python
from sklearn.model_selection import train_test_split
from torchvision import models
import torch.nn as nn

df_train, df_valid = train_test_split(data, test_size=0.1, random_state=42)

class Resnext50(nn.Module):
    def __init__(self, n_classes=9):
        super().__init__()
        # 加载预训练权重，让模型一开始就懂基本的图像特征
        resnet = models.resnext50_32x4d(pretrained=True)
        # 把原来的 1000 类输出层换成 9 类，并加 dropout 防止过拟合
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()  # 每个标签独立输出 0~1 概率

    def forward(self, x):
        return self.sigm(self.base_model(x))
```
这里用 `Sigmoid` 而不是 `Softmax`，因为 9 个标签可以同时为真。比如一家店既适合晚餐又有酒精，两个输出都应该接近 1。

训练参数方面，材料里用的是 Adam 优化器、学习率 1e-4、batch size 32、训练 15 个 epoch。
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Resnext50(n_classes=9).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()  # 二分类交叉熵，对每个标签分别算损失

batch_size = 32
max_epoch = 15

for epoch in range(max_epoch):
    model.train()
    for imgs, targets in train_dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```
`BCELoss` 会把这 9 个标签当作 9 个独立的二分类问题来算，正好符合多标签的设定。学习率用 1e-4 是因为我们加载了预训练权重，太大的学习率会破坏已经学好的特征。

超参数调优上，材料主要调整了 backbone 和输入尺寸，没有做大规模网格搜索。小样本实验里先用较少的 epoch 跑通流程，再观察验证集 F1 是否还在涨。如果训练损失一直降、验证 F1 不涨，就说明过拟合，需要靠 dropout、数据增强和早停来控制。

### 2.4 模型测试与指标报告
在每个 epoch 结束后，我们在验证集上计算 micro、macro、samples 三种 F1。
```python
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(pred, target, threshold=0.5):
    # 概率大于阈值就算预测为该标签
    pred = np.array(pred > threshold, dtype=float)
    return {
        'micro_f1': f1_score(target, pred, average='micro'),
        'macro_f1': f1_score(target, pred, average='macro'),
        'samples_f1': f1_score(target, pred, average='samples'),
        'micro_precision': precision_score(target, pred, average='micro'),
        'micro_recall': recall_score(target, pred, average='micro'),
    }

model.eval()
all_probs, all_targets = [], []
with torch.no_grad():
    for imgs, targets in val_dataloader:
        imgs = imgs.to(device)
        probs = model(imgs).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets.numpy())

result = calculate_metrics(np.vstack(all_probs), np.vstack(all_targets), threshold=0.5)
print(result)
```
材料里的小样本训练日志显示，第 0 个 epoch 时 micro F1 约 0.638、macro F1 约 0.582、samples F1 约 0.610；训练到后面，最好的一次达到 micro F1 约 0.825、macro F1 约 0.789、samples F1 约 0.801。当然，这是在小样本验证集上的结果，不能直接当成排行榜分数，但能说明模型确实学到了有用的视觉特征。

确定阈值也很重要。默认 0.5 不一定最优，因为有些标签本来就很稀少。材料里的做法是打印不同阈值下的 F1，选验证集上最高的那个。
```python
best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.3, 0.71, 0.05):
    score = calculate_metrics(all_probs, all_targets, threshold=thr)['samples_f1']
    if score > best_f1:
        best_f1 = score
        best_thr = thr
print('最佳阈值:', best_thr, '样本 F1:', best_f1)
```
阈值越低，模型越愿意报“有”，召回率上升但精确率可能下降；阈值越高则相反。我们按验证集 F1 来选，而不是拍脑袋定 0.5。

最后生成提交文件。测试集需要先按商户聚合照片，再对每家商户的预测取平均或投票。
```python
test_data = pd.merge(test_photo_to_id, sample_submission, on='business_id', how='left')

# 对每张测试照片预测 9 个概率
test_probs = predict_test_photos(model, test_data)  # 伪代码：内部是 DataLoader 前向传播
test_data['probs'] = list(test_probs)

# 按商户聚合：同一商户所有照片的概率取平均
biz_probs = test_data.groupby('business_id')['probs'].apply(
    lambda x: np.mean(np.vstack(x), axis=0)
)

rows = []
for biz, prob in biz_probs.items():
    # 概率超过阈值的标签组成空格分隔的字符串
    labels = [str(i) for i, p in enumerate(prob) if p >= best_thr]
    rows.append({'business_id': biz, 'labels': ' '.join(labels)})

submission = pd.DataFrame(rows)
submission.to_csv('submission.csv', index=False)
```
聚合这一步很关键：训练时是一家商户的多张照片共享标签，测试时也应该把同一商户的照片放在一起看，单张照片的预测噪声太大。

## 三、总结

本文中我们完成了 Yelp 餐厅照片分类的完整流程。我们首先用 EDA 看清了数据：2000 家商户、23 万多张训练照片、9 个互不排斥的餐厅属性标签，以及只有 4 家商户缺标签。其次我们把空格分隔的标签串解析成 9 维 0/1 向量，丢掉缺失行，并把图片统一缩放、标准化，训练时加入翻转和颜色增强。接着我们加载 ImageNet 预训练的 ResNeXt50，把输出层换成 9 个节点，用 Sigmoid 和 BCE 损失做多标签分类，按 9:1 划分训练验证集，用 Adam、1e-4 学习率训练 15 个 epoch。然后我们在验证集上同时报告 micro、macro、samples 三种 F1，并在 0.3 到 0.7 之间搜索最佳阈值。最后我们按商户聚合测试照片的概率，生成符合样例格式的提交文件。

从材料来看，这个比赛的核心难点不在模型结构，而在“多标签”和“商户级聚合”这两件事。照片本身可以用成熟的预训练 CNN 处理，但标签要按 9 个独立二分类来建模，预测结果还要从单张照片汇总到商户，才能和比赛的提交格式对齐。这个经验对我们以后做类似的多标签图像任务很有帮助。
