# iWildCam 2019 FGVC6

## 一、比赛简介

### 1.1 比赛背景

iWildCam 2019 是由 FGVC6 主办的大规模野生动物图像分类比赛。数据来自 Trail Camera（红外相机）采集的自然场景，目标是在 23 个物种类别中识别图像内容。图像噪声大、背景复杂，且同一序列中可能出现多个动物。

比赛提供 `train.csv` 与压缩的 `train_images.zip`，包含 196086 张训练图像。每个样本记录类别 ID、拍摄时间、位置、序列信息等。

### 1.2 比赛目标

给定测试图像，预测其所属物种类别。类别 0 表示空图，其余 1-22 对应具体动物如鹿、狼、熊等。提交文件要求为每个测试样本输出类别标签。

### 1.3 评价标准

比赛使用宏平均 F1 分数（macro F1）作为评价指标，衡量各类别平均性能，值越高越好。它对类别不平衡敏感，要求模型在少数类上也有表现。

## 二、解题过程

我们使用 EDA 理解数据分布，采用预训练 DenseNet121 做迁移学习，并用 TFRecord 加速读取。

### 2.1 数据探索（EDA）

先读取训练 CSV 并统计类别数。

```python
import pandas as pd

train_gt = pd.read_csv('../input/iwildcam-2019-fgvc6/train.csv')
print('训练图像总数:', len(train_gt))
print('类别数:', train_gt.category_id.nunique())
```

输出约 196086 张图，23 个类别。我们映射类别 ID 到名称，并看分布。

```python
classes_wild = {0: 'empty', 1: 'deer', 2: 'moose', 3: 'squirrel', ...}
train_gt['category_label'] = train_gt.category_id.map(classes_wild.get)
category_counts = train_gt.groupby('category_label').size()
print(category_counts.describe())
```

类别分布不均衡，部分物种样本很少。还有多标签情况：部分文件在同一序列中出现多个物种。

### 2.2 数据清洗

图像读取时统一裁剪并归一化。我们先将原始 zip 转换为 TFRecord 加速访问。

```python
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

def parse_example(example):
    features = tf.io.parse_tensor(example, out_type=tf.string)
    img = tf.image.decode_jpeg(features, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    return img
```

对极度不平衡的数据，我们采用类别加权或 oversampling。也可对空图进行筛选，以降低噪声。

### 2.3 模型训练与超参数调优

使用 DenseNet121 预训练模型，冻结底层，仅训练顶层分类头。

```python
mobile_net = tf.keras.applications.DenseNet121(weights='imagenet', input_shape=(192,192,3), include_top=False)
mobile_net.trainable = False

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(23, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

训练采用小批量，steps_per_epoch 控制训练量，先做 5 epoch 预热。

```python
model.fit(ds, epochs=5, steps_per_epoch=60)
```

超参数包括学习率 0.001、batch size 32、输入尺寸 192×192。后续可取消冻结、加入数据增强提升表现。

### 2.4 模型测试与指标报告

训练后在验证集上评估宏 F1，并生成测试预测。

```python
# 预测测试集
preds = model.predict(test_ds)
pred_labels = np.argmax(preds, axis=1)
```

我们检查预测分布，确保没有全集中在多数类的情况，并按比赛格式写入提交文件。

```python
submission = pd.DataFrame({
    'file_name': test_files,
    'category_id': pred_labels
})
submission.to_csv('submission.csv', index=False)
```

训练日志显示准确率从第 1 轮的 0.6885 提升到第 5 轮的 0.7615，说明模型在逐步学习。

## 三、总结

本文中我们完成了 iWildCam 2019 野生动物图像分类。我们首先通过 EDA 确认约 196k 张训练图、23 类极度不均衡，且存在多标签序列。其次我们将图像转为 TFRecord 并归一化。接着使用 DenseNet121 迁移学习配合稀疏交叉熵训练 5 轮。最后在测试集上生成类别预测并保存提交。该方案为大规模自然图像分类提供了可复用的预处理与迁移学习框架。