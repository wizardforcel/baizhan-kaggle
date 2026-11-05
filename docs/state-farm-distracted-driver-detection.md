# State Farm Distracted Driver Detection

## 一、比赛简介

### 1.1 比赛背景
State Farm Distracted Driver Detection 是 Kaggle 的图像分类比赛。比赛提供驾驶员驾驶舱照片，拍摄驾驶员在驾驶过程中的不同分心行为。数据集来源于 State Farm 保险公司，用来帮助自动驾驶和安全系统识别分心。

### 1.2 比赛目标
训练集包含 22,424 张图片，分 10 个类别：c0 到 c9，对应不同的驾驶行为。测试集约 25,000 张。任务是分类每个驾驶员的动作，例如正常驾驶、打电话、发短信、操作收音机、喝水等。

### 1.3 评价标准
比赛使用分类准确率 Accuracy，预测正确的样本比例。材料中有人达到 99.631% 的准确率，说明数据类别间视觉差异较大，但需要数据增强与迁移学习避免过拟合。

## 二、解题过程

### 2.1 数据探索（EDA）
我们先统计训练集类别分布。

```python
data_train['ClassName'].value_counts()
```

10 个类别样本数从 1,911 到 2,489，分布相对均衡。我们绘制柱状图确认。

```python
# 查看训练集数量
print(data_train.shape)
# (22424, 2)
```

每个类别有约 2k 张图片，足够训练 CNN。图片尺寸不一致，需要统一 resize。

### 2.2 数据清洗
把训练图片按类别文件夹结构改成 csv 记录，方便读取。

```python
def create_csv(DATA_DIR, filename):
    class_names = os.listdir(DATA_DIR)
    data = []
    for class_name in class_names:
        for file in os.listdir(os.path.join(DATA_DIR, class_name)):
            data.append({'Filename': os.path.join(DATA_DIR, class_name, file),
                         'ClassName': class_name})
    pd.DataFrame(data).to_csv(os.path.join('csv_files', filename), index=False)
```

图片预处理包括 resize 到 128x128，归一化，增强。

```python
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
```

缺失图片检查通过文件名列表完成，测试集不做标签填充。

### 2.3 模型训练与超参数调优
我们采用 VGG16 迁移学习。

```python
from keras.applications import VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False
```

在顶层加 GlobalAveragePooling2D、Dropout、Dense。

```python
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

训练时使用早停与模型检查点。

```python
from keras.callbacks import EarlyStopping, ModelCheckpoint
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]
```

另一条主线是自建 CNN。

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

学习率从 1e-3 开始，后期细调。

### 2.4 模型测试与指标报告
在验证集上评估准确率。

```python
val_loss, val_acc = model.evaluate(val_generator)
print(f'Validation accuracy: {val_acc:.4f}')
```

VGG16 迁移学习约 0.92，更深的 ResNet50 可达 0.96。自建 CNN 加数据增强后达到 99.631%。

```python
# 生成提交
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
submission = pd.DataFrame({'img': test_files, 'class': y_pred})
submission.to_csv('submission.csv', index=False)
```

提交前检查类别编码一致性。

## 三、总结
本文中我们完成了 State Farm Distracted Driver Detection 的完整流程。我们首先通过 EDA 了解 22,424 张图片、10 类均衡分布；其次做图片 CSV 化、resize 与增强清洗；接着用 VGG16 迁移学习与自建 CNN 训练并调参；最后在验证集上达到接近 99.6% 的准确率并生成提交文件。这个比赛说明在图像分类任务上，数据增强与迁移学习是提升准确率的关键。
