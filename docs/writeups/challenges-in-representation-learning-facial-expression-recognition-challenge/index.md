# 人脸表情识别（Facial Expression Recognition Challenge）

## 一、比赛简介

### 1.1 比赛背景

这个比赛的全名是「Challenges in Representation Learning: Facial Expression Recognition Challenge」，最早是 ICML 2013 的一场表征学习竞赛，后来长期以 FER2013 的名字挂在 Kaggle 上。

数据是从互联网上抓来的人脸照片，被统一裁剪、转成灰度、缩放到 **48×48 像素**。每张照片对应一个人物表情，一共有 7 类：愤怒（Angry）、厌恶（Disgust）、恐惧（Fear）、开心（Happy）、悲伤（Sad）、惊讶（Surprise）、中性（Neutral）。

这道题在深度学习圈里很有代表性，因为它的难度并不低：

1. **分辨率极低**：48×48 的灰度图，五官细节几乎糊成一团，人眼都不一定分得清「恐惧」和「惊讶」；
2. **类别不均衡**：「开心」的样本很多，「厌恶」的样本很少；
3. **标注有噪声**：同一张脸的标签在不同标注者眼里可能不一样，所以准确率的天花板本来就不高。

### 1.2 比赛目标

官方只给一个 CSV 文件 `icml_face_data.csv`，三列：

- `emotion`：表情标签，0～6 的整数；
- `Usage`：这张图属于哪个子集，取值是 `Training`、`PublicTest`、`PrivateTest`；
- `pixels`：一张 48×48 图片被拉平成的 2304 个像素值，用空格隔开。

训练集 28,709 张，公开测试集 3,589 张，私有测试集 3,589 张，合计 35,887 张。

我们要根据 `pixels` 预测 `emotion`。提交时按测试集顺序给每张图一个表情编号。

### 1.3 评价标准

比赛使用**分类准确率（Accuracy）**：预测对的表情数 ÷ 总图片数。7 分类，随机猜的准确率约 1/7 ≈ 0.14。准确率越高越好，满分 1.0。

---

## 二、解题过程

我们的主线是：**把像素还原成 48×48 的灰度图、复制成三通道以复用 ImageNet 预训练模型，然后用 DenseNet201 微调，配合强数据增强和早停。** 这是迁移学习在低分辨率人脸任务上最省事、也最有效的一套组合。

### 2.1 数据探索（EDA）

我们首先把 CSV 读进来，按 `Usage` 拆成训练、公开测试、私有测试三份：

```python
import pandas as pd

dataset = pd.read_csv("../input/icml_face_data.csv")
# 原始列名不统一，先改成我们好用的名字
dataset.columns = ["emotion", "Usage", "pixels"]

# 按官方划分拆成三份
train_dataset      = dataset.loc[dataset.Usage == "Training",    ["emotion", "pixels"]]
public_dataset     = dataset.loc[dataset.Usage == "PublicTest",  ["emotion", "pixels"]]
private_dataset    = dataset.loc[dataset.Usage == "PrivateTest", ["emotion", "pixels"]]

print(dataset.shape)          # (35887, 3)
print(train_dataset.shape)    # (28709, 2)
```

`pixels` 一列是一长串空格分隔的数字，比如 `70 80 82 72 58 ...`。它其实是一张图的 2304 个灰度值按行优先拉平的结果。所以我们第一件事是把它变回数组：

```python
import numpy as np

def pixels_to_array(pixels):
    # 按空格切开，转成 uint8（灰度 0~255）
    return np.array(pixels.split(), "uint8")

train_dataset["pixels"] = train_dataset["pixels"].apply(pixels_to_array)
```

再看表情分布。这一步非常关键，因为它直接决定了我们后面要不要做类别加权：

```python
# 统计每个表情的占比，并按 0~6 排序
emotion_prop = (train_dataset.emotion.value_counts() / len(train_dataset)).sort_index()
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
print(emotion_prop)
print(emotions)
```

结果显示「开心」明显偏多，「厌恶」明显偏少，是典型的长尾分布。如果只用普通交叉熵，模型会倾向于多预测多数类，「厌恶」几乎永远预测不出来。这提示我们要么做类别加权，要么用重采样。

### 2.2 数据清洗

这份数据的「清洗」主要是**形状和通道**：

1. 把 `pixels` 还原成 `(N, 48, 48, 1)`；
2. **复制成三通道** `(N, 48, 48, 3)`，因为 ImageNet 预训练模型接受三通道输入；
3. 把像素值归一化到 0～1 区间（除以 255），让网络训练更稳定。

```python
def image_reshape(data):
    # -1 让 numpy 自动推断样本数；最后一维是通道数
    image = np.reshape(data["pixels"].to_list(), (data.shape[0], 48, 48, 1))
    # 灰度复制三份，凑成 RGB，才能喂给 ImageNet 预训练模型
    image = np.repeat(image, 3, -1)
    return image

X_train = image_reshape(train_dataset)
y_train = train_dataset["emotion"].values
X_val   = image_reshape(private_dataset)   # 私有测试集当验证集
y_val   = private_dataset["emotion"].values
X_test  = image_reshape(public_dataset)    # 公开测试集当最终测试
y_test  = public_dataset["emotion"].values

print(X_train.shape, X_val.shape, X_test.shape)
# (28709, 48, 48, 3) (3589, 48, 48, 3) (3589, 48, 48, 3)
```

这里有一个常见疑问：为什么拿 `PrivateTest` 当验证集？因为比赛排行榜就是按私有测试集算分的，把它留作验证最贴近最终目标；`PublicTest` 则留作最后检查。

**数据增强**是这道题提分的关键。48×48 的图很容易过拟合，我们对训练图做随机水平翻转、平移缩放旋转、透视变换、加噪声和锐化：

```python
import albumentations as A
import cv2

def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),                       # 人脸左右翻转仍然合理
        A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0.4,
                           shift_limit=0.4, p=1,
                           border_mode=cv2.BORDER_REPLICATE),  # 平移/缩放/旋转
        A.Perspective(p=0.6),                          # 轻微透视，模拟拍摄角度
        A.Affine(p=0.5, scale=0.9, rotate=0.5, fit_output=True),
        A.GaussNoise(p=0.5),                           # 加噪声，抗过拟合
        A.Sharpen(p=0.5),
    ])
```

注意 `border_mode=cv2.BORDER_REPLICATE`：旋转后画面边缘会露出空白，用「复制边缘像素」填补，比填黑更自然，不会凭空造出一圈黑边让模型误学。

### 2.3 模型训练与超参数调优

我们选用 **DenseNet201**。DenseNet 的特点是层与层之间密集连接，特征复用充分，在小图上参数效率高。我们加载 ImageNet 预训练权重，去掉原分类头，自己接一个三层全连接 + Dropout 的头：

```python
import tensorflow.keras as keras

def build_model():
    # include_top=False：不要 ImageNet 的 1000 类分类头
    base = keras.applications.DenseNet201(
        include_top=False,
        weights="imagenet",
        input_shape=(48, 48, 3),
    )
    base.trainable = True                 # 整个网络都参与微调

    inputs = keras.Input(shape=(48, 48, 3))
    x = base(inputs)
    x = keras.layers.Flatten()(x)         # DenseNet 输出 (None,1,1,1920)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)      # 强 Dropout 抑制过拟合
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(7, activation="softmax")(x)  # 7 类概率
    return keras.Model(inputs, outputs)

model = build_model()
```

接着编译和训练。优化器用 Adam，学习率 0.0001；损失是 7 类交叉熵；并挂上两个回调：**早停**（验证准确率 7 轮不涨就停，并恢复最好权重）和**学习率衰减**（验证损失 2 轮不降就把学习率乘 0.1）：

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    # 监控验证准确率，耐心 7 个 epoch，恢复最佳权重
    EarlyStopping(monitor="val_accuracy", mode="max",
                  patience=7, min_delta=1e-3, restore_best_weights=True),
    # 验证损失停滞时把学习率缩小 10 倍，最小到 1e-6
    ReduceLROnPlateau(monitor="val_loss", factor=0.1,
                      patience=2, min_lr=1e-6, mode="min"),
]
```

**关于类别不均衡。** 材料里准备了 `class_weight="balanced"` 的加权方案，可以按类别频率给少数类更高权重。但主线训练用的是普通交叉熵：因为强数据增强已经让模型见到更多样的少数类样本，再叠加很大的类别权重反而会让模型过度预测「厌恶」，整体准确率下降。这是一个需要实验权衡的点，不是「一定要加权」。

**关于训练轮数。** 设了 100 个 epoch，但实际上早停会先触发。材料里记录的训练曲线显示，验证准确率在第 6 轮左右就到达 0.679 附近，之后提升很慢甚至回落，说明模型很快进入平台期——这也是低分辨率人脸任务的天花板体现。

### 2.4 模型测试与指标报告

训练过程中我们记录每个 epoch 的训练/验证准确率，观察是否过拟合：

```python
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"],     label="训练准确率")
plt.plot(history.history["val_accuracy"], label="验证准确率")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()
```

材料里的实际曲线大致是：训练准确率从 0.35 一路爬到 0.81，验证准确率从 0.53 升到 **0.6793** 后就不再明显上升。训练和验证之间的差距说明模型已经能很好拟合训练集，但对没见过的脸泛化有限。

我们进一步用**混淆矩阵**看错在哪里：

```python
from sklearn.metrics import confusion_matrix, accuracy_score

pred_val = model.predict(X_val).argmax(axis=1)
print("验证准确率:", accuracy_score(y_val, pred_val))
print(confusion_matrix(y_val, pred_val))
```

混淆矩阵会显示有几类特别容易混：

- **恐惧 vs 惊讶**：两者都是「瞪眼张嘴」，48×48 下几乎一样；
- **悲伤 vs 中性**：表情幅度小，像素差异微弱；
- **厌恶**：样本太少，经常被预测成愤怒或中性。

这不是模型不行，而是数据本身就存在标注歧义。理解这一点，我们才不会盲目地把模型越堆越大。

最后我们对公开测试集做预测并生成提交文件：

```python
# 对公开测试集输出类别概率，取概率最大的那一类
pred_test = model.predict(X_test).argmax(axis=1)

submission = pd.DataFrame({"emotion": pred_test})
submission.to_csv("submission.csv", index=False)
print("提交行数:", len(submission))
```

生成后核对行数是否与测试集一致（3,589 行），并确认标签都在 0～6 范围内，避免格式问题白丢分。

---

## 三、总结

本文中我们完成了 FER2013 人脸表情识别的完整流程。我们首先用 EDA 确认了 35,887 张 48×48 灰度图、7 类表情，并按官方 `Usage` 拆出训练/公开测试/私有测试；同时发现类别长尾严重，「开心」多、「厌恶」少。其次我们把 `pixels` 还原成图片、复制成三通道并归一化，用水平翻转、平移缩放旋转、透视、噪声和锐化做数据增强。接着我们加载 ImageNet 预训练的 DenseNet201，接上 512-256-128 的全连接头和 Dropout，用 Adam（学习率 1e-4）、交叉熵、早停和学习率衰减训练。最后我们在私有测试集上得到约 0.68 的验证准确率，并用混淆矩阵定位出「恐惧/惊讶」「悲伤/中性」这两组典型混淆。

这个比赛让我们学到两件事：**一是迁移学习在低分辨率图像上依然非常有效**，48×48 的图配上 ImageNet 预训练也能到接近 0.7；**二是要看懂错误的来源**——剩下那三成错误里，很大一部分来自标注本身的模糊，继续加大模型未必能解决。材料里还有用 FER+ 标签、Grad-CAM 可视化、以及把多个 CNN 集成起来的方案，本质上都是在「提高标签质量」和「降低方差」上做文章。
