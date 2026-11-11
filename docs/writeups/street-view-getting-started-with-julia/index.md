# 街景字符识别入门赛（Street View Getting Started with Julia）

## 一、比赛简介

### 1.1 比赛背景

街景字符识别入门赛是 Kaggle 上的一场入门级比赛。名字里的 `street view` 指街景照片，`getting started` 表示它是给新手练手的比赛，`with Julia` 则透露出一个信息：这场比赛最初是围绕 Julia 这门编程语言设计的。Julia 是一种主打科学计算和数值分析的语言，Kaggle 想借着这场比赛让大家熟悉「用 Julia 跑一个完整的机器学习流程」。不过从我们收集到的参赛代码来看，大家主要还是用 Python、TensorFlow/Keras 和 fastai 来完成，语言本身并不是得分的关键，真正决定分数的是能不能把图里的字符认对。

所谓字符识别，就是给模型看一张很小的字符图片，让它回答「这是哪个字符」。它是 OCR（光学字符识别，Optical Character Recognition）的基本功。现实中的 OCR 系统通常先在一张街景大图里找到文字区域，把文字切成一个个字符，再逐个识别，最后拼成门牌号、路牌或招牌上的字符串。这个比赛跳过了「找文字」和「切字符」两步，直接把切好的单字符小图交给我们，只保留最核心的分类问题：这张图到底是 `A`、`7` 还是 `g`。

从材料里可以看到，训练集有 6283 张图片，测试集有 6220 张图片，都是 BMP 格式，尺寸只有 20×20 像素。标签一共有 62 种：数字 `0` 到 `9` 共十个，大写字母 `A` 到 `Z` 共二十六个，小写字母 `a` 到 `z` 共二十六个。20×20 的分辨率非常低，很多笔画细节已经糊在一起，加上街景拍摄时的光照、模糊和角度变化，人眼有时也难分清楚。fastai 那份 notebook 在训练后列出的易混字符对就很能说明问题：`s` 和 `S` 错了 16 次，`0` 和 `O` 错了 14 次，`O` 和 `o` 错了 10 次，`I` 和 `l` 错了 9 次。这些字符要么只差大小写，要么形状本来就接近，正是这个比赛的主要难点。

### 1.2 比赛目标

比赛给我们的输入有两部分。第一部分是训练图片和标签表 `trainLabels.csv`，标签表只有两列：

| 列名 | 含义 |
| --- | --- |
| `ID` | 图片编号，和图片文件名一一对应，例如 `1704.Bmp` 对应 `ID=1704` |
| `Class` | 这张图片里的字符，例如 `n`、`8`、`T`、`I`、`R` |

第二部分是测试图片，文件名同样是编号，例如 `7899.Bmp`，但没有对应的标签。我们要做的是：对每一张测试图片预测它属于 62 个字符中的哪一个，然后生成一个包含 `ID` 和 `Class` 两列的 CSV 文件。提交样例 `sampleSubmission.csv` 也是这个格式，行数必须和测试集一致。换句话说，这是一个「输入一张小图、输出一个字符标签」的多分类问题，类别数是 62。

训练图片和测试图片都来自同一个街景字符数据集，图片已经被统一裁剪、缩放成 20×20 的小图。因为尺寸太小，模型不可能靠看清笔画的每个细节来识别，更多要靠整体形状、笔画粗细和明暗分布这些「轮廓级」的特征。这也是为什么卷积神经网络特别适合这个任务：卷积核天然擅长捕捉局部形状和边缘。

### 1.3 评价标准

比赛使用分类准确率（accuracy）作为评价指标，也就是「预测正确的图片数除以图片总数」：

```
accuracy = 预测正确的张数 / 测试集总张数
```

准确率越高越好，满分是 1.0。因为一共有 62 个类别，如果完全随机乱猜，平均只能猜对大约 1/62 ≈ 1.6%，所以任何明显高于 1.6% 的结果都说明模型学到了东西。

材料里的 fastai notebook 用的是 `error_rate` 这个指标，它等于 `1 - accuracy`，也就是错误率，越低越好。两种写法说的是同一件事：错误率 0.156 就等价于准确率 0.844。我们在后面的报告里会同时给出这两种口径，避免读者混淆。

---

## 二、解题过程

整场比赛的思路可以概括成一条主线：先把 20×20 的小图整理成统一格式，再用卷积神经网络做 62 类分类；自己从零搭网络大约能到 65% 的验证准确率，而借助在 ImageNet 上预训练过的 ResNet34 做迁移学习，验证错误率能降到 15.6%，也就是约 84.4% 的准确率。我们首先做数据探索，其次做数据清洗，接着比较几种模型并调参，然后做验证与混淆分析，最后生成提交文件。

### 2.1 数据探索（EDA）

我们首先读入标签表，看看数据长什么样、一共有多少种类别。下面的代码只依赖 pandas，读进来之后打印前几行和类别列表：

```python
# pandas 是处理表格数据的常用库，read_csv 可以直接把标签表读成一张表
import pandas as pd

TRAIN_CSV = '../input/street-view-getting-started-with-julia/trainLabels.csv'
train_labels = pd.read_csv(TRAIN_CSV)

# 先看前几行，确认列名确实是 ID 和 Class，顺便观察标签的写法
print(train_labels.head())
# 材料里的输出大致是：ID=1/Class='n'，ID=2/Class='8'，
# ID=3/Class='T'，ID=4/Class='I'，ID=5/Class='R'
print(train_labels.shape)

# 把标签转成字符串后再排序去重，数一数到底有多少个字符类别
classes = sorted(train_labels['Class'].astype(str).unique())
print(len(classes))   # 62
print(classes)        # 0~9、A~Z、a~z，一共 62 个
```

打印结果告诉我们两件关键的事。第一，标签表的结构非常简单，没有缺失值，也没有多余的特征列，`ID` 是图片编号，`Class` 是字符标签。第二，类别数确实是 62，正好对应「十个数字 + 二十六个大写字母 + 二十六个小写字母」。这个数字后面会直接决定模型输出层的宽度。

接着我们打开训练图片目录，检查图片的数量、尺寸和色彩模式，并随机看几张图：

```python
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# 训练图片解压后放在 train/train/ 目录下，文件名就是标签表里的 ID
train_dir = Path('../input/street-view-getting-started-with-julia/train/train')
files = sorted(train_dir.glob('*.Bmp'))
print(len(files))          # 材料里数出来是 6283 张训练图

# 打开第一张，确认尺寸和通道数；这一步决定了网络输入层怎么写
im = Image.open(files[0])
print(im.size, im.mode)    # (20, 20) RGB

# 一次画 8 张，直观感受低分辨率字符图长什么样
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for ax, f in zip(axes.ravel(), files[:8]):
    ax.imshow(Image.open(f))
    ax.set_title(f.stem)   # 文件名去掉扩展名就是 ID，方便和标签对照
    ax.axis('off')
plt.show()
```

这一步让我们确认了图片是 20×20 的彩色 BMP，测试目录 `test/test/` 下则有 6220 张同规格的图片。我们还在浏览过程中发现少数图片其实是灰度图，读出来只有两个维度，没有 RGB 三个通道。如果不处理这类图片，后面按 `(20, 20, 3)` 搭建的网络会在拼接数组时直接报形状错误。这个发现直接引出了下一节的数据清洗。

### 2.2 数据清洗

其次我们处理这些「不规整」的图片，并把像素值缩放到适合训练的范围。清洗主要做四件事：删掉三张有问题的训练灰度图、把灰度图补成三通道、按 ID 对齐标签、把 0~255 的像素值除以 255。下面是我们整理后的训练集读取代码：

```python
import numpy as np

# 重新读一遍标签表，避免受前面探索时改动的影响
train_data = pd.read_csv(TRAIN_CSV)

# 材料里明确点出这三行是灰度图，直接删掉最省事
train_data.drop([283, 2289, 3135], inplace=True)
train_data.reset_index(drop=True, inplace=True)  # 重排行号，后面按位置取数据不会错位

images, labels = [], []
for img_id, cls in zip(train_data['ID'], train_data['Class']):
    img = np.asarray(Image.open(train_dir / f'{img_id}.Bmp'))

    # 少数图片只有两个维度，补成三通道，保证所有样本形状一致
    if img.shape != (20, 20, 3):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    images.append(img / 255.0)   # 缩放到 0~1，梯度更稳定，收敛更快
    labels.append(cls)

X = np.asarray(images, dtype=np.float32)
y = np.asarray(labels)
print(X.shape)   # (6280, 20, 20, 3)
```

这里有一个值得单独说的坑：标签对齐。有的 notebook 不是按 `ID` 查标签，而是按 `1, 2, 3, ...` 的顺序直接读图，然后用 `labels['Class'].iloc[:-1]` 去掉标签表最后一行来「凑齐」数量。这种写法依赖「图片编号连续、且和行号顺序完全一致」这个假设，一旦中间删过图或者编号有缺口就会错位。我们更推荐上面的做法：以图片文件名里的 `ID` 为准去标签表里查 `Class`，这样增删图片都不会影响对应关系。

测试集也要做类似的清洗：解不开的图片要跳过，灰度图要转成三通道。材料里的做法是用 `try/except` 逐张解码，把失败的图片记录下来再删除：

```python
test_dir = Path('../input/street-view-getting-started-with-julia/test/test')
test_files = sorted(test_dir.glob('*.Bmp'))
print(len(test_files))   # 6220 张测试图

good_files = []
for f in test_files:
    try:
        im = Image.open(f).convert('RGB')  # convert('RGB') 把灰度图统一成三通道
        im.load()
        good_files.append(f)
    except Exception:
        pass   # 解不开的图片先跳过，避免预测阶段整个程序崩掉

# 记录被跳过的图片编号，提交前要专门检查这些行有没有补回来
skipped = sorted(set(f.stem for f in test_files) - set(f.stem for f in good_files))
print('skipped:', skipped)   # 材料里测试集有两张图解码失败
```

需要提醒的是，删除测试图片会让最终提交文件少几行，而比赛要求提交行数和测试集完全一致。所以「跳过」只是让预测流程能跑完，最后还是要给这些被跳过的图片补一个预测结果，或者至少确认提交文件的 `ID` 集合和测试集对得上。这个细节看似琐碎，却是很多新手在入门赛里白白丢分的地方。

### 2.3 模型训练与超参数调优

接着我们进入模型训练。为了知道「到底什么水平算好」，我们先跑了一个最简单的基线：把图片缩成 28×28 的灰度图、拉平成一个向量，用最近邻分类器（KNN，K=1）直接按像素距离找最像的训练样本。材料里的这份基线在 10% 的验证集上拿到了 **0.4229** 的准确率。这个结果说明字符形状本身确实有很强的区分度，连「找最像的图」都能到 42%；但它也说明只靠像素距离不够，因为同一个字符在不同光照和字体下像素差异很大。

随后我们搭了一个小型卷积神经网络。卷积层（Conv2D）可以理解成一组「会学习的放大镜」，每个卷积核在图片上滑动，专门寻找某种局部图案，比如竖线、横线、圆弧；池化层（MaxPool2D）把相邻区域合并，保留最强响应，相当于「抓重点、降尺寸」；Dropout 则在训练时随机关掉一部分神经元，防止模型死记训练样本。下面是我们用的网络结构：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split

# 先把彩色图转成灰度并缩放到 28×28，减少参数量
X_gray = np.asarray([cv2.resize(img, (28, 28)) for img in X])
X_gray = X_gray[:, :, :, np.newaxis]   # 增加通道维，形状变成 (N, 28, 28, 1)

# 62 类标签做 one-hot 编码，输出层才能用 softmax 算每类的概率
Labels = pd.get_dummies(y)
Trainx, valx, Trainy, valy = train_test_split(
    X_gray, Labels, test_size=0.2, random_state=42)

cmodel = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPool2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Dropout(0.25),
    Flatten(),                    # 把特征图拉成一维向量，接全连接层
    Dense(128, activation='relu'),
    Dense(62, activation='softmax')   # 62 个字符类别，输出概率分布
])

cmodel.compile(loss='categorical_crossentropy',
               optimizer='rmsprop',
               metrics=['accuracy'])
history = cmodel.fit(Trainx, Trainy, epochs=15, validation_data=(valx, valy))
```

这份网络在材料里的表现是：训练准确率一路涨到 0.78，验证准确率最高到 **0.6555**。训练集和验证集之间拉开了十多个百分点，说明模型已经开始过拟合——它记住了训练图的一些细节，但没学会泛化。我们随后加了数据增强来缓解这个问题，做法是对训练图做小幅平移、缩放，让模型每次看到的图都略有不同：

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 小样本字符识别很容易过拟合，用轻度几何变换「造」更多训练样本
datagen = ImageDataGenerator(
    width_shift_range=0.2,   # 左右平移最多 20%，模拟字符位置偏移
    height_shift_range=0.2,  # 上下平移最多 20%
    zoom_range=0.2,          # 随机缩放，模拟字符大小变化
    rescale=1./255,          # 顺便把像素缩到 0~1
    fill_mode='nearest')     # 平移后空出来的边缘用最近像素补齐
```

我们还试过一条「加深网络」的路线：材料里的 `second_attempt_2` 把三个卷积层直接堆在一起，再池化，最后接 1024 维全连接层，用 Adam 优化器和稀疏交叉熵训练。问题出在 batch size 被设成了 6000，训练日志里每轮只有 `1/1` 步，也就是一轮只更新一次梯度。结果训练到 100 多轮时验证准确率还在 0.55 附近徘徊，收敛非常慢。这个教训很直接：**网络更深不一定更好，超参数配错会把训练效率拖垮**。对小数据集来说，batch 太大意味着每轮看到的梯度估计太少，反而不如小 batch 稳定。

真正带来突破的是迁移学习。我们改用 fastai 的 `cnn_learner`，直接加载在 ImageNet 上预训练好的 ResNet34。ResNet34 在几百万张自然图片上已经学会了识别边缘、纹理、形状这些通用视觉特征，我们只需要在它的基础上用街景字符图做微调（fine-tune），让它把「通用视觉」调整成「字符识别」。代码如下：

```python
from fastai.vision.all import *

# 用文件名里的数字 ID 去标签表查字符，保证图片和标签一一对应
def label_func(fname):
    the_id = int(re.findall("[0-9]+", fname.name)[0])
    return train_labels.loc[train_labels['ID'] == the_id, 'Class'].values.item()

db = DataBlock(
    blocks=(ImageBlock, CategoryBlock),        # 输入是图片，输出是类别
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),  # 留 20% 做验证
    get_y=label_func,
    item_tfms=Resize(128, method='squish'),    # 把 20×20 拉到 128×128
    batch_tfms=aug_transforms(do_flip=False))  # 不做翻转，见下面的解释

dls = db.dataloaders(train_dir)
learn = cnn_learner(dls, resnet34, metrics=error_rate)

# 先让 fastai 自动找一组合适的学习率，避免手工试错
lr_min, lr_steep = learn.lr_find()
print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")
# 材料里的输出：Minimum/10: 1.45e-02, steepest point: 1.32e-02

learn.fine_tune(4, base_lr=1.32e-02)   # 用找到的学习率微调 4 轮
```

这里有两个细节值得解释。第一，`do_flip=False` 是因为字符不能随便翻转：把 `b` 左右翻转会变成 `d`，把 `p` 上下翻转会变成 `b` 附近的形状，翻转后的图在字符世界里可能对应另一个类别，增强反而会制造错误标签。第二，把 20×20 放大到 128×128 并不会凭空增加信息，但能让 ResNet 的预训练卷积核在更接近 ImageNet 的尺度上工作，也让网络有足够的空间做多层下采样。`lr_find` 扫描一系列学习率，找出损失下降最快的那一段，材料里给出的最陡点是 `1.32e-02`，我们就用这个值做微调。

### 2.4 模型测试与指标报告

然后我们在验证集上检查模型到底错在哪里。fastai 提供了 `ClassificationInterpretation`，可以直接列出「最容易被混淆的类别对」：

```python
from fastai.vision.all import ClassificationInterpretation

interp = ClassificationInterpretation.from_learner(learn)

# 只列出混淆次数不少于 5 的类别对，避免输出太长
print(interp.most_confused(min_val=5))
# 材料里的结果：
# [('s', 'S', 16), ('0', 'O', 14), ('O', 'o', 10),
#  ('o', 'O', 10), ('I', 'l', 9), ('c', 'C', 7)]

# 也可以把损失最高的几张图可视化，看看模型最没把握的样本长什么样
interp.plot_top_losses(9, figsize=(10, 6))
```

这份混淆列表和我们的直觉完全一致：错误集中在「大小写只差一点」和「形状本来就像」的字符上。`s` 与 `S`、`c` 与 `C` 是大小写关系，`0` 与 `O`、`o` 与 `O` 是圆形字符，`I` 与 `l` 是竖线字符。在 20×20 的分辨率下，这些区别往往只剩几个像素的宽度差，模型很难稳定区分。知道这一点之后，我们就能判断：继续盲目加深网络收益有限，更有效的方向可能是提高输入分辨率、做针对性增强，或者对易混类别单独处理。

最后是生成提交文件。我们先解压测试集，用同一个 `dls` 构造测试数据加载器，再让模型输出每张图的 62 类概率，取概率最大的那一类作为预测字符：

```python
# 测试图片解压后放在 test/test/ 目录，文件名同样是数字 ID
test_files = get_image_files(test_dir)

tst_dl = dls.test_dl(test_files)
preds, _ = learn.get_preds(dl=tst_dl)

# 对每张图取概率最大的类别下标，再映射回字符标签
pred_idx = preds.argmax(dim=1)
pred_labels = [learn.dls.vocab[i] for i in pred_idx]

# 从文件名里提取数字 ID，和预测结果拼成提交表
pred_ids = [int(re.findall("[0-9]+", f.name)[0]) for f in test_files]
df_pred = pd.DataFrame({"ID": pred_ids, "Class": pred_labels})
df_pred = df_pred.sort_values("ID", ignore_index=True)   # 按 ID 排序，方便核对

df_pred.to_csv("submission.csv", index=False)
print(df_pred.shape)   # 行数应该和测试集图片数一致
```

我们最终得到的验证指标可以放在一起对比：

| 方案 | 验证指标 | 说明 |
| --- | --- | --- |
| 1-NN 最近邻 | 准确率 0.4229 | 把 28×28 灰度图拉平后按像素距离找最近样本 |
| 小型 CNN（从零训练） | 验证准确率 0.6555 | 两层卷积 + 池化 + Dropout，15 轮训练 |
| 加深 CNN（`second_attempt_2`） | 验证准确率约 0.55 | batch size 设为 6000，每轮只有一步更新，收敛很慢 |
| fastai + ResNet34 迁移学习 | 错误率 0.156051，约准确率 0.844 | 预训练权重 + 数据增强 + 自动学习率 |

从这张表可以清楚看到迁移学习的价值：同样只有六千多张训练图，从零训练的 CNN 停在 65% 左右，而预训练过的 ResNet34 把错误率压到了 15.6%。原因并不神秘——小数据集不足以让一个深层网络从随机权重学出稳定的视觉特征，但预训练模型已经带着「怎么看图」的先验知识过来，我们只需要做小幅调整。对入门者来说，这是一条非常实用的经验：**数据少的时候，先找预训练模型，再考虑自己搭网络**。

另外要强调提交前的检查。材料里有的 notebook 在测试阶段删掉了两张解码失败的灰度图，结果预测结果只有 6218 条，而测试集有 6220 张，直接提交就会因为行数不匹配出问题。我们的做法是：预测前先确认 `good_files` 加回被跳过的图片后是否覆盖了全部测试 ID，生成 CSV 后再检查行数和 `ID` 集合，确保一行不多、一行不少。

---

## 三、总结

本文中我们完成了街景字符识别入门赛的完整流程。我们首先用 EDA 摸清了数据：训练集 6283 张、测试集 6220 张，图片都是 20×20 的 BMP，标签共 62 类，对应数字、大写字母和小写字母；同时发现少数图片是灰度图，存在形状不一致的问题。其次我们做了数据清洗：删掉三张有问题的训练灰度图，把灰度图补成三通道，按文件名里的 ID 去标签表对齐标签，并把像素值缩放到 0~1。接着我们比较了多种模型：1-NN 基线拿到 0.4229 的准确率，从零训练的小型 CNN 把验证准确率提到 0.6555，而盲目加深网络、把 batch size 设成 6000 的做法反而让训练几乎停滞。然后我们改用 fastai 加载 ImageNet 预训练的 ResNet34 做迁移学习，配合自动学习率搜索和针对性数据增强，把验证错误率降到 0.156051，相当于约 84.4% 的准确率。最后我们分析了混淆矩阵，确认错误主要集中在 `s/S`、`0/O`、`I/l` 这类形状接近的字符上，并按比赛格式生成了提交文件。

这个比赛给我们最大的启发有两点。第一，**小数据集上迁移学习往往比从零搭网络更划算**：ResNet34 带来的先验知识，让 6283 张图发挥出了远超其数量的效果。第二，**清洗和提交格式的细节不能省**：灰度图会导致形状报错，按行号对齐标签可能错位，删掉测试图会让提交行数不够——这些都不是模型问题，却足以让一次完整的训练白费。对刚入门的人来说，先把数据管好、把流程跑通，再去追求模型分数，是最稳妥的路径。
