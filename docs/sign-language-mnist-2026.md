# Sign Language MNIST 2026

## 一、比赛简介

### 1.1 比赛背景

Sign Language MNIST 是一个手语字母识别的图像分类数据集。它模仿了著名的 MNIST 手写数字数据集：每张图片都是 28×28 的灰度图，但图片内容不是数字，而是美国手语（American Sign Language，简称 ASL）的手势字母。数据由不同人的手部照片经过裁剪和缩放得到，共 24 个类别。字母 J 和 Z 需要手部运动才能表示，静态图片无法表达，因此数据集把这两个类别排除在外，只保留 24 个静态字母。

手语识别的意义在于帮助听障人士与不懂手语的人沟通。传统上手语翻译依赖专业译员，成本很高；如果计算机能直接从摄像头图像里识别字母，就能做成实时辅助工具。静态字母识别是这个问题最基础的一步，相当于先把一个个字母认出来，再把它们拼成单词和句子。这个比赛在 2026 年重新整理发布，供入门者练习卷积神经网络（CNN）。

材料里这份 notebook 原本是机器学习的课后作业：作者把 Kaggle 官方教程 `A Simple CNN Model Beginner Guide` 中的 Fashion MNIST 数据集替换成 Sign Language MNIST，网络结构基本沿用教程，只是把最后的分类数从 10 改成 24。因此这篇 writeup 的重点不是发明复杂模型，而是把“数据怎么读、图像怎么预处理、CNN 怎么搭、结果怎么解释”这条完整链路讲清楚。

### 1.2 比赛目标

训练数据有 27,455 张图片，测试数据有 7,172 张。每个 CSV 文件的一行代表一张图片：第一列是 `label`，取值为 0～25 中的整数，对应字母在字母表中的位置；后面 784 列是 28×28 个像素值，取值 0～255。因为 J 和 Z 被跳过，实际出现的标签只有 24 个。

我们的目标是根据像素预测每张测试图片属于哪个字母。由于原始标签不连续（缺少 9 和 25），需要先把剩下的 24 个标签重新映射到 0～23，才能作为神经网络的输出索引。最终提交的是一张图片对应的字母类别，评价标准是分类准确率。

### 1.3 评价标准

比赛使用分类准确率（accuracy）作为指标，也就是预测正确的图片数占全部测试图片的比例，越高越好，满分是 1.0（100%）。材料里的作业要求非常明确：必须把运行时打印出来的真实测试准确率写进报告，不能预先填写任何数字。这一点很重要——分类问题里“看起来差不多”和“真实准确率”往往差距很大，所以我们必须保留独立的测试集，并且只在最后评估一次。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先导入常用库并固定随机种子。固定种子能让数据划分、权重初始化和训练过程尽量可复现，这在课堂作业和比赛中都是基本要求。

```python
import os
import glob
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 固定 Python 和 TensorFlow 的随机种子，保证多次运行结果接近
np.random.seed(42)
tf.random.set_seed(42)

def find_csv(filename):
    """在 Kaggle 输入目录或本地目录中查找 CSV 文件"""
    candidates = (
        glob.glob('/kaggle/input/**/' + filename, recursive=True)
        + glob.glob('./**/' + filename, recursive=True)
    )
    if not candidates:
        raise FileNotFoundError(f'Cannot find {filename}')
    return candidates[0]
```

我们其次读取训练和测试 CSV，查看形状和头几行。训练集是 `(27455, 785)`，测试集是 `(7172, 785)`：785 等于 1 个标签列加 784 个像素列。像素值都是 0～255 的整数，说明图像是 8 位灰度图；每行代表一张已经展平的图片，需要再还原成 28×28 的二维形状，CNN 才能理解空间结构。

我们还把前几张图片画出来看。手语字母的图片差异主要体现在手指的伸展方式和手掌朝向，背景基本是纯黑，手部偏亮。这种“高对比度、主体居中”的特点，说明简单的卷积网络就足以提取边缘和手指轮廓，不必使用非常深的模型。

### 2.2 数据清洗

这份数据的“清洗”主要是格式转换，而不是删数据。我们接着做四件事：分离标签、把标签重新编码、把像素还原成图像、把像素缩放到 0～1。

```python
train_data = pd.read_csv(find_csv('sign_mnist_train.csv'))
test_data = pd.read_csv(find_csv('sign_mnist_test.csv'))

# 原始标签是字母序号 0~25，但 J(9) 和 Z(25) 不存在
y_train_original = train_data['label'].to_numpy()
y_test_original = test_data['label'].to_numpy()

# LabelEncoder 会把出现的 24 个标签重新映射成连续的 0~23
encoder = LabelEncoder()
y_train_all = encoder.fit_transform(y_train_original)
y_test = encoder.transform(y_test_original)

# 用 encoder.classes_ 找回每个编号对应的真实字母
class_names = [string.ascii_uppercase[int(i)] for i in encoder.classes_]

# 像素列转成 float32，再变形为 28×28×1，最后除以 255
X_train_all = train_data.drop(columns=['label']).to_numpy(dtype='float32')
X_test = test_data.drop(columns=['label']).to_numpy(dtype='float32')
X_train_all = X_train_all.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
```

这里的 `reshape(-1, 28, 28, 1)` 中，`-1` 表示行数由数据自动推断，后三个维度分别是高、宽和通道数。因为图片是灰度图，通道数为 1；如果是彩色图就是 3。除以 255 是为了把输入缩放到 0～1，让梯度更稳定，训练收敛更快。

然后我们从训练集里划出 20% 作为验证集。验证集只用于观察训练过程、决定是否早停，绝不参与最终测试。划分时使用 `stratify=y_train_all`，保证每个字母在训练和验证中的比例一致，否则某些样本少的字母可能在验证集里几乎不出现。

```python
X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all,
    test_size=0.20,
    random_state=42,
    stratify=y_train_all   # 按类别分层，保证 24 个字母比例一致
)
print('Train:', X_train.shape, 'Validation:', X_val.shape)
print('Test:', X_test.shape)
assert X_train.shape[1:] == (28, 28, 1)
```

我们特意检查 `num_classes == 24`。如果不做这个断言，很容易在标签编码出错时仍然训练出一个“看起来能跑”的模型，最后才发现输出层和标签数量对不上。

### 2.3 模型训练与超参数调优

我们接着搭建 CNN。结构沿用教程：两层卷积加池化，然后展平、全连接、Dropout，最后用 24 个神经元的 softmax 输出各类别概率。卷积层负责提取局部特征，比如手指边缘、手掌轮廓；池化层负责压缩尺寸、增强平移不变性；全连接层把特征组合成类别判断。

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    # 32 个 3×3 卷积核，padding='same' 保证输出尺寸不变
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # 第二层通道数翻倍，学习更抽象的组合特征
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # Dropout 随机丢弃 30% 的神经元，抑制过拟合
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(24, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # 标签是整数，所以用 sparse 版本
    metrics=['accuracy']
)
model.summary()
```

训练时使用 10 个 epoch、batch size 128，并加 EarlyStopping。EarlyStopping 监控验证损失，连续 3 个 epoch 没有改善就停止，并恢复验证损失最低的那一轮权重。这样既不会训练不足，也避免在验证集上过拟合。

```python
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )],
    verbose=1
)
```

关于调参，我们主要关注三点。第一是学习率：Adam 的默认学习率 0.001 在这个规模上比较稳，太小会训练慢，太大会让损失震荡。第二是 Dropout 比例：0.3 是入门教程的常用值，如果发现训练准确率远高于验证准确率，可以适当加大。第三是 epoch 数：材料用 10 个并配合早停，实际停止的轮数由验证损失决定，因此最终的 `epochs_completed` 可能小于 10。我们没有做大规模网格搜索，因为这份作业的目标是跑通流程、看懂结果，而不是追求排行榜极限。

### 2.4 模型测试与指标报告

训练结束后，我们最后在完全独立的测试集上评估一次，并打印准确率和损失。测试集从头到尾没有参与训练或调参，因此这个数字才是对模型泛化能力的诚实估计。

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Final Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Final Test Loss: {test_loss:.4f}')
print(f'Epochs completed: {len(history.history["loss"])}')
```

除了总准确率，我们还会看混淆矩阵和每个类别的 precision、recall、F1。只看总准确率有个问题：如果某个字母的样本特别多，模型即使完全认不出稀有字母，总分也可能不低。混淆矩阵能暴露“哪些字母经常被认成哪些字母”。

```python
from sklearn.metrics import classification_report, confusion_matrix

pred_proba = model.predict(X_test, batch_size=128, verbose=0)
predicted_labels = np.argmax(pred_proba, axis=1)

print(classification_report(
    y_test, predicted_labels,
    labels=np.arange(24),
    target_names=class_names,
    zero_division=0
))
cm = confusion_matrix(y_test, predicted_labels, labels=np.arange(24))
```

根据材料里的设计，运行后还会保存训练曲线 `training_graphs.png`、样例预测图 `sample_predictions.png`、混淆矩阵 `confusion_matrix.png`，以及 `result_summary.json` 和 `test_predictions.csv`。把真实结果写进 JSON 而不是直接写死在代码里，是一个值得养成的好习惯：报告里的数字来自运行输出，而不是来自猜测。

```python
import json
results = {
    'dataset': 'Sign Language MNIST',
    'validation_examples': int(len(X_val)),
    'test_examples': int(len(X_test)),
    'classes': class_names,
    'epochs_completed': int(len(history.history['loss'])),
    'final_test_accuracy_percent': round(float(test_accuracy) * 100, 2),
    'final_test_loss': float(test_loss)
}
with open('result_summary.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
model.save('hand_sign_cnn.keras')
```

材料没有预先填入任何准确率数字，我们也不在这里编造一个。正确的做法是运行之后把 `result_summary.json` 里的真实值抄进报告。通常这类简单 CNN 在 Sign Language MNIST 上能取得相当高的准确率，因为背景干净、手势区分明显；但具体数值必须以实际运行结果为准。

## 三、总结

本文中我们完成了 Sign Language MNIST 的手语字母分类流程。我们首先阅读了两个 CSV 的形状和像素含义，确认训练集 27,455 张、测试集 7,172 张、24 个类别，并理解了 J 和 Z 因需要动态手势而被排除。其次我们做了标签重编码、像素形状还原和 0～1 缩放，并按类别分层划出 20% 验证集。接着我们搭建了两层卷积加池化的 CNN，用 Adam 优化、sparse 交叉熵损失、Dropout 和 EarlyStopping 训练。最后我们在独立测试集上评估，并用混淆矩阵和分类报告分析每个字母的表现，把真实结果保存为 JSON 和预测文件。

这个项目给我们三点启发。第一，迁移教程时要改的不只是数据集名字：输入形状、类别数、标签编码都可能变化，任何一处对不上都会让模型悄悄失效，所以要写断言检查。第二，验证集和测试集必须严格分开，训练过程中只能看验证集，最终成绩只能来自测试集。第三，评价分类模型不能只看一个总准确率，混淆矩阵能告诉我们模型到底在哪些字母上犯错，这对改进数据和模型都更有指导意义。
