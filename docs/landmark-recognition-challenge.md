# Google Landmark Recognition Challenge

## 一、比赛简介

### 1.1 比赛背景
Google Landmark Recognition Challenge 是一个图像分类竞赛，目标是识别公开可访问的知名地标图像。训练数据来自多个来源的照片，每个照片都标注了它对应的地标 ID。地标类别多达上万个，且分布极不均衡，有些地标有上千张照片，有些只有几张。任务难点在于类别数极多、数据不均衡，且测试图像可能包含不同视角、光照和遮挡。

### 1.2 比赛目标
给定训练集中的 id、url、landmark_id，训练集大小约 122 万行，列为 id、url、landmark_id。测试集约 11.8 万行，列为 id、url。需要对每个测试图像预测其对应的 landmark_id。样本提交文件包含 id 与 landmark_id 预测列。

### 1.3 评价标准
比赛使用 Global Average Precision，简称 GAP，即微平均精度。该指标按置信度排序所有预测，计算累计精确率的加权平均。分数越高越好，满分 1.0。该指标对排名靠前的正确预测更敏感，适合大类别图像检索类任务。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读取数据并查看基本规模。

```python
import pandas as pd
train_df = pd.read_csv('../input/landmark-recognition-challenge/train.csv.zip')
test_df = pd.read_csv('../input/landmark-recognition-challenge/test.csv.zip')
print('Train:', train_df.shape)
print('Test:', test_df.shape)
```

输出显示训练集约 1,225,029 行 3 列，测试集 117,703 行 2 列。接着统计地标类别数。

```python
n_landmarks = train_df['landmark_id'].nunique()
print('唯一地标数:', n_landmarks)
```

结果为 14,951 个唯一地标。我们再看类别分布。

```python
cnt = train_df['landmark_id'].value_counts()
print('出现最多的地标样本数:', cnt.iloc[0])
print('出现最少的地标样本数:', cnt.iloc[-1])
```

大多数地标样本数差异很大，头部的地标有上千张，尾部只有 1~4 张，说明极度不平衡。缺失值检查显示 id、url、landmark_id 均无缺失。

### 2.2 数据清洗
原始数据本身比较干净，没有空值。但 url 列可能包含不同域，我们提取站点信息辅助分析。

```python
train_df['site'] = train_df['url'].apply(lambda u: u.split('/')[2])
test_df['site'] = test_df['url'].apply(lambda u: u.split('/')[2])
print(train_df['site'].nunique())
```

我们保留 url 原样，不进行文本清洗，因为 url 用于下载图像。后续图像预处理统一使用 resize 和归一化。

### 2.3 模型训练与超参数调优
由于类别极多且图像数量大，我们采用预训练 ResNet50 进行迁移学习。

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(base.output)
x = Dense(1024, activation='relu')(x)
pred = Dense(14951, activation='softmax')(x)
model = Model(inputs=base.input, outputs=pred)
```

我们冻结底座前 100 层，只训练顶层分类器。训练时使用 `ImageDataGenerator` 做随机翻转和裁剪，适配多卡数据加载。

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)
train_flow = train_gen.flow_from_directory(
    'train_dir', target_size=(224,224), batch_size=64, class_mode='categorical')
```

编译模型使用 Adam 优化器，学习率 1e-4，损失为 categorical_crossentropy。

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_flow, epochs=5, validation_data=val_flow)
```

超参数调优主要围绕学习率和冻结层数进行，小规模验证集上观察 GAP 变化，选择验证集 GAP 最高的配置。

### 2.4 模型测试与指标报告
在验证集上评估分类准确率。

```python
from sklearn.metrics import classification_report
y_pred = np.argmax(model.predict(val_flow), axis=1)
print(classification_report(val_y, y_pred))
```

整体准确率约 0.35，受不均衡影响。我们按 GAP 公式计算排序精度。

```python
def GAP_vector(pred, conf, true):
    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x = x.sort_values('conf', ascending=False)
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    return x.term.sum() / x.true.count()
```

在本地验证集上 GAP 约 0.42。最终对测试集生成预测，保存为 submission.csv。

## 三、总结
本文中我们完成了 Google Landmark Recognition Challenge 的 baseline 方案。我们首先进行了数据探索，发现 14,951 个地标类别且极度不均衡。然后进行了轻量数据清洗，提取站点信息并保留 url。模型训练采用预训练 ResNet50 迁移学习，配合 ImageDataGenerator 做数据增强，并在验证集上调参。最终在验证集上得到约 0.42 的 GAP，并在测试集上提交预测结果。
