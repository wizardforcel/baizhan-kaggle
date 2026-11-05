# 脏文档去噪（Denoising Dirty Documents）

## 一、比赛简介

### 1.1 比赛背景

图书馆和档案馆里有大量纸质文档被扫描成电子图片。扫描件往往不干净：纸上有咖啡渍、霉斑、水痕，扫描仪本身还会引入噪点，甚至纸张背面的字会透过来。这些「脏东西」会影响后续的 OCR（文字识别），让机器把「3」认成「8」。人工用修图软件一张张擦，成本高得离谱。

这个比赛要做的事情就是：**给模型看成对的图片——左边是脏的扫描件，右边是人工清理干净的版本——让模型学会自动把脏图变干净**。它不是分类也不是检测，而是「图到图」的转换：输入一张灰度图，输出一张同样大小的灰度图，要求输出尽量接近「干净版本」。

这个任务之所以适合用深度学习，是因为脏污的形态千变万化，很难写规则去描述「什么是污渍」；但给足成对的例子，卷积网络可以自己总结出「背景该是什么样、文字该保留成什么样」。

### 1.2 比赛目标

数据分三个目录：

- `train/`：脏的扫描图，共 **144 张**；
- `train_cleaned/`：与 `train/` 一一对应的干净图，也是 144 张，作为训练的「标准答案」；
- `test/`：需要我们去噪的测试图，共 **72 张**。

图片统一是灰度图，尺寸为 **420 × 540**（高 × 宽）。我们要为测试集的每一个像素输出一个 0～1 之间的灰度值。

### 1.3 评价标准

比赛使用**均方根误差（RMSE）**，**越低越好**。算法是：把每个像素的预测值和真实值相减、平方，对所有像素求平均，再开根号。因为像素值已经被缩放到 0～1，所以 RMSE 0.1 大致意味着「平均每个像素差 0.1」，也就是肉眼能看出一点模糊但文字结构还在；RMSE 越小，说明还原得越干净。

---

## 二、解题过程

我们首先解压数据、把图片读成统一格式的数组（EDA 与预处理），其次把像素归一化并切出验证集（数据清洗），接着搭一个卷积自编码器、用 MSE 损失训练（模型训练与调参），最后在测试集上推理、把结果展开成提交格式（测试与报告）。

### 2.1 数据探索（EDA）

数据是压缩包形式，先解压到工作目录，再用 OpenCV 逐张读入。我们把脏图和干净图成对地画出来，确认「脏」到底脏在哪：

```python
import zipfile, os, cv2
import numpy as np
import matplotlib.pyplot as plt

path_zip = '/kaggle/input/denoising-dirty-documents/'
path = '/kaggle/working/'

# 四个压缩包：脏训练图、干净训练图、测试图、提交样例
for name in ['train', 'test', 'train_cleaned', 'sampleSubmission.csv']:
    with zipfile.ZipFile(path_zip + name + '.zip', 'r') as z:
        z.extractall(path)

# 按文件名排序，保证脏图、干净图、测试图各自顺序稳定
train_img = sorted(os.listdir(path + 'train'))
train_cleaned_img = sorted(os.listdir(path + 'train_cleaned'))
test_img = sorted(os.listdir(path + 'test'))

# 8 张图：奇数位置放脏图，偶数位置放对应的干净图，方便上下对照
plt.figure(figsize=(15, 25))
for i in range(0, 8, 2):
    dirty = cv2.imread(path + 'train/' + train_img[i], 0)         # 0 表示灰度读入
    clean = cv2.imread(path + 'train_cleaned/' + train_cleaned_img[i], 0)
    plt.subplot(4, 2, i + 1); plt.imshow(dirty, cmap='gray')
    plt.title('Noise image: {}'.format(train_img[i])); plt.axis('off')
    plt.subplot(4, 2, i + 2); plt.imshow(clean, cmap='gray')
    plt.title('Denoised image'); plt.axis('off')
plt.show()
```

对照图能看出两类噪声：一类是**局部污渍**（一块深色斑点），一类是**整页的底色不均**（纸发黄、发灰，背景不是纯白）。文字本身的笔画在干净图里是清晰的黑色，所以模型要学会的是「保留笔画、抹掉背景」。

### 2.2 数据清洗

图像任务里的「清洗」其实就是**统一格式**。我们写一个 `process_image`，把任意图片变成模型能吃的固定形状：

```python
def process_image(file):
    img = cv2.imread(file)                        # 默认按 BGR 三通道读
    img = np.asarray(img, dtype='float32')
    img = cv2.resize(img, (540, 420))             # 统一成宽 540、高 420
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # 转灰度，只留一个通道
    img = img / 255.0                             # 归一化到 0~1，网络更容易训练
    img = np.reshape(img, (420, 540, 1))          # 补一个通道维，形状 (H, W, C)
    return img
```

这里有两个细节值得解释：**转灰度**是因为文档去噪只关心「黑的程度」，彩色信息反而会引入干扰；**除以 255** 是把像素从 0～255 压到 0～1，配合输出层的 sigmoid（输出范围正好 0～1），损失函数才能稳定下降。

然后分别处理三个目录，并切出 15% 做验证集——验证集不参与训练，只用来判断模型是不是真的学会了「去噪」而不是「背答案」：

```python
train, train_cleaned, test = [], [], []
for f in sorted(os.listdir(path + 'train/')):
    train.append(process_image(path + 'train/' + f))
for f in sorted(os.listdir(path + 'train_cleaned/')):
    train_cleaned.append(process_image(path + 'train_cleaned/' + f))
for f in sorted(os.listdir(path + 'test/')):
    test.append(process_image(path + 'test/' + f))

# 列表转数组：形状 (样本数, 420, 540, 1)
X_train, y_train = np.asarray(train), np.asarray(train_cleaned)
X_test = np.asarray(test)

# 85% 训练、15% 验证；输入和标签必须同步切分
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
```

### 2.3 模型训练与超参数调优

我们用的是**卷积自编码器（Convolutional Autoencoder）**。名字听着玄，其实结构很直白：前半段叫**编码器**，用卷积和池化把图片一层层「压缩」，提取出「哪里有笔画、哪里有污渍」的抽象特征；后半段叫**解码器**，用转置卷积把特征一层层「放大」回原图尺寸；最后输出一张和输入同样大小的干净图。

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose

conv_autoencoder = Sequential()
# ---- 编码器：两次「卷积 + 池化」，尺寸从 420×540 压到 105×135 ----
conv_autoencoder.add(Conv2D(32, (3, 3), input_shape=(420, 540, 1),
                            activation='relu', padding='same'))
conv_autoencoder.add(MaxPooling2D((2, 2), padding='same'))   # 420×540 -> 210×270
conv_autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
conv_autoencoder.add(MaxPooling2D((2, 2), padding='same'))   # 210×270 -> 105×135

# ---- 解码器：两次转置卷积，尺寸放大回 420×540 ----
conv_autoencoder.add(Conv2DTranspose(8, kernel_size=3, strides=2,
                                     activation='relu', padding='same'))
conv_autoencoder.add(Conv2DTranspose(16, kernel_size=3, strides=2,
                                     activation='relu', padding='same'))

# ---- 输出层：1 个通道 + sigmoid，输出 0~1 的灰度值 ----
conv_autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
```

整个模型只有 **7,417 个参数**，比很多人想象的「深度学习模型」小得多。这是因为任务是「保留原图结构、只去除纹理噪声」，网络不需要记住复杂的语义，只需要学会一个局部的「去污滤波器」。

编译时用 **MSE（均方误差）** 做损失——它和比赛的 RMSE 指标是同一个东西（只差一个开根号），优化 MSE 就是在直接优化比赛指标：

```python
from tensorflow.keras.callbacks import EarlyStopping

conv_autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 连续 10 轮验证损失不降就停，并恢复验证损失最低那一轮的权重
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

history = conv_autoencoder.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,          # 最多练 200 轮
    batch_size=16,       # 每批 16 张图
    callbacks=[early_stop])
```

**关于调参。** 材料里的做法是把 epoch 上限设得比较宽（200），靠早停自动决定何时收手；batch size 取 16，是在「显存/内存吃得下」和「梯度足够稳」之间折中。真正影响效果的主要是两点：一是卷积核数量要**逐层递减再递增**（32→16 编码、8→16 解码），形成「漏斗」结构；二是输出层必须用 sigmoid，否则输出会超出 0～1，和标签对不上。

### 2.4 模型测试与指标报告

训练日志显示，验证损失从第 1 轮的 0.2050 一路降到第 200 轮的 **0.0100**，验证 MAE 降到 **0.0403**，而且训练损失和验证损失几乎贴在一起（第 200 轮分别是 0.0102 和 0.0100），说明模型没有明显过拟合。把验证 MSE 开根号，**RMSE 大约是 0.10**，正好对应比赛指标。

我们把损失曲线画出来检查收敛形态：

```python
epoch_loss     = history.history['loss']
epoch_val_loss = history.history['val_loss']
epoch_mae      = history.history['mae']
epoch_val_mae  = history.history['val_mae']

plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(epoch_loss)), epoch_loss, 'b-', label='Train Loss')
plt.plot(range(len(epoch_val_loss)), epoch_val_loss, 'r-', label='Val Loss')
plt.title('Evolution of loss on train & validation'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(epoch_mae)), epoch_mae, 'b-', label='Train MAE')
plt.plot(range(len(epoch_val_mae)), epoch_val_mae, 'r-', label='Val MAE')
plt.title('Evolution of MAE on train & validation'); plt.legend()
plt.show()
```

两条曲线一路平滑下降、没有分叉，说明训练过程是健康的。接着对测试集预测，并把预测图还原到测试图的**原始尺寸**——因为预测是在 420×540 上做的，而每张测试图的实际尺寸可能不同，提交时必须按原图尺寸逐像素填写：

```python
# 测试集推理：形状 (72, 420, 540, 1)
y_pred = conv_autoencoder.predict(X_test, batch_size=16)

ids, vals = [], []
for i, f in enumerate(test_img):
    img = cv2.imread(path + 'test/' + f, 0)      # 读原图，拿到真实宽高
    h, w = img.shape
    # 把 420×540 的预测图缩放到原图尺寸
    pred = cv2.resize(y_pred[i], (w, h))
    imgid = int(f[:-4])                          # 文件名去掉 .png 就是图片编号
    for r in range(h):
        for c in range(w):
            # 提交 id 的格式：图片编号_行号_列号（行列都从 1 开始）
            ids.append(f'{imgid}_{r + 1}_{c + 1}')
            vals.append(pred[r, c])

submission = pd.DataFrame({'id': ids, 'value': vals})
submission.to_csv('submission.csv', index=False)
print('Length of IDs:', len(ids))     # 14230080，与样例提交行数一致
```

我们最后核对了两件事：提交文件一共 **14,230,080 行**，和官方样例提交的行数完全一致；`id` 的格式是「编号_行_列」且行列从 1 开始，`value` 都是 0～1 之间的小数。这两点任何一处写错，提交都会直接判无效。

---

## 三、总结

本文中我们完成了脏文档去噪比赛的完整流程。我们首先解压数据并成对查看脏图与干净图，确认噪声主要是局部污渍和整页底色不均；其次我们统一了图片格式：缩放到 420×540、转灰度、像素归一化到 0～1，并用 85/15 切出验证集；接着我们搭建了一个只有 7,417 个参数的卷积自编码器，编码器用卷积加池化压缩、解码器用转置卷积还原、输出层用 sigmoid，以 MSE 为损失训练 200 轮并配早停；最后我们得到验证 MSE 0.0100、MAE 0.0403（RMSE 约 0.10），并按「编号_行_列」的格式生成了 1423 万行的提交文件。

这场比赛给我们的最大启发是：**图到图任务不一定需要大模型**。当输入和输出共享同一套空间结构（脏图与干净图只差一层噪声）时，一个很浅的小自编码器就能学到「去污」这个局部变换；真正决定成败的反而是数据格式——灰度、归一化、输出层激活函数、提交 id 的拼法，任何一环出错，模型再好也白搭。