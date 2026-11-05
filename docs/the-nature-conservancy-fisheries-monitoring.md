# The Nature Conservancy Fisheries Monitoring

## 一、比赛简介

### 1.1 比赛背景

The Nature Conservancy Fisheries Monitoring 是一个用计算机视觉识别鱼类的图像分类比赛。渔船上的摄像头会从不同角度拍下甲板或船舱中的照片，照片中可能有鱼，也可能没有鱼。我们的任务不是数鱼，也不是画出鱼的边界，而是根据整张图片判断其中最可能出现的鱼类类别。

材料把比赛目标概括为：根据船载摄像头拍摄的不同角度的图片，检测一张图片中出现的鱼类。数据中一共有八个目标类别：`ALB`（Albacore tuna，长鳍金枪鱼）、`BET`（Bigeye tuna，大眼金枪鱼）、`YFT`（Yellowfin tuna，黄鳍金枪鱼）、`DOL`（Mahi Mahi，鲯鳅）、`LAG`（Opah，月鱼）、`SHARK`（鲨鱼）、`OTHER`（有鱼但不属于前面几类）和 `NoF`（没有鱼）。

这道题的难点首先来自图像本身。不同图片的尺寸并不完全相同，拍摄角度、光照、背景和鱼在画面中的位置都会变化；同一类鱼也可能只占很小区域。其次，类别数量并不均衡，`ALB` 的图片明显多于其他类别，少数类别的样本较少。最后，比赛要求我们提交每个类别的概率，而不是只提交一个类别名称，所以模型既要分对，还要给出比较可靠的置信度。

材料还提到比赛分为两个阶段，第二阶段测试数据会在后期发布。由于图片涉及比赛主办方的保密要求，探索时需要注意不要公开具体样本图像；因此本文只讨论材料中已经公开的目录、尺寸、数量和代码流程。

### 1.2 比赛目标

我们拿到的训练数据放在 `train/` 目录下，并按照类别分成八个子目录：`ALB`、`BET`、`DOL`、`LAG`、`NoF`、`OTHER`、`SHARK` 和 `YFT`。目录名就是训练标签。测试数据放在 `test_stg1/` 目录下，测试图片没有目录标签，我们需要为每张测试图片输出八个类别的预测概率。

材料中的探索结果显示，训练集各类图片数量分别为：`ALB` 1719 张、`BET` 200 张、`DOL` 117 张、`LAG` 67 张、`NoF` 465 张、`OTHER` 299 张、`SHARK` 176 张、`YFT` 734 张，一共 3777 张。测试阶段的 `test_stg1/` 中有 1000 张图片。训练图片常见尺寸是 `720_1280_3`，也就是高度 720、宽度 1280、三个颜色通道；材料还观察到训练集有 10 种不同尺寸，最小记录为 `720_1244_3`，最大记录为 `974_1732_3`。

因此，比赛的输入是一张颜色图片，输出是八个类别的概率向量。我们可以把输出理解成一个八格的表格：每一格表示这张图片属于对应鱼类的可能性。八个概率应该共同表达模型的判断，而不是只保留最大的一格。

### 1.3 评价标准

材料中的方案标题明确使用了 `log loss`，例如深度森林方案报告了约 1.4 的 public leaderboard log loss，卷积神经网络方案的标题报告了 `LB 1.279`。因此本比赛使用多分类对数损失（log loss）评价预测概率，分数越低越好。

对一张图片来说，如果真实类别是 `ALB`，模型给 `ALB` 的概率越高，损失就越小；如果模型把很高的概率给了错误类别，损失就会变大。把所有测试图片的损失取平均，就得到排行榜分数。这个指标与只看“猜对还是猜错”不同，它会惩罚过度自信的错误预测，所以我们在测试时要保留完整的概率输出，并且让模型最后一层产生八个类别的概率。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先检查训练目录的结构，确认每个子目录都代表一个类别，再统计各类图片数量。这样做的原因是，图像分类最容易忽略的问题不是模型，而是标签目录是否读对；同时，类别数量分布会直接影响后面的训练和验证划分。

```python
# 导入数组、表格和绘图库，方便统计数据并画出类别分布
import numpy as np
import pandas as pd
from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns

# 读取训练目录下的子目录；目录名就是比赛提供的类别标签
sub_folders = check_output(["ls", "../input/train/"]).decode("utf8").strip().split('\n')
count_dict = {}
for sub_folder in sub_folders:
    # 每个文件对应一张训练图片，因此文件数可以作为该类别的样本量
    num_of_files = len(check_output([
        "ls", "../input/train/" + sub_folder
    ]).decode("utf8").strip().split('\n'))
    count_dict[sub_folder] = num_of_files

# 画柱状图是为了直观看出类别是否严重不均衡
plt.figure(figsize=(12, 4))
sns.barplot(x=list(count_dict.keys()), y=list(count_dict.values()), alpha=0.8)
plt.xlabel('Fish Species')
plt.ylabel('Number of Images')
plt.show()
```

运行后，我们看到 `ALB` 有 1719 张，而 `LAG` 只有 67 张，类别之间差异很大。这个现象说明不能只看总体准确率，也不能简单地把少数类当成噪声删除。少数类别虽然图片少，但它们仍然是需要提交的目标类别。训练时如果随机切分不当，验证集中可能几乎没有某个少数类，评价就会失真，因此后面要使用分层切分。

接着，我们检查测试集规模和图片尺寸。尺寸检查很重要，因为卷积网络要求一个批次中的图片形状一致，而原始图片存在多种尺寸，不能直接全部堆叠成数组。

```python
# 统计第一阶段测试图片数量，确认预测时需要生成多少行结果
num_test_files = len(check_output([
    "ls", "../input/test_stg1/"
]).decode("utf8").strip().split('\n'))
print("Number of test files present :", num_test_files)

# 收集训练集里出现过的图像形状，检查是否所有图片大小一样
train_path = "../input/train/"
different_file_sizes = {}
for sub_folder in sub_folders:
    file_names = check_output([
        "ls", train_path + sub_folder
    ]).decode("utf8").strip().split('\n')
    for file_name in file_names:
        im_array = imread(train_path + sub_folder + "/" + file_name)
        size = "_".join(map(str, list(im_array.shape)))
        different_file_sizes[size] = different_file_sizes.get(size, 0) + 1

# 用图形汇总尺寸，而不是只打印长列表，便于发现主流尺寸
plt.figure(figsize=(12, 4))
sns.barplot(x=list(different_file_sizes.keys()),
            y=list(different_file_sizes.values()), alpha=0.8)
plt.xticks(rotation='vertical')
plt.title("Image size present in train dataset")
plt.show()
```

材料显示，`test_stg1/` 有 1000 张图片，测试集的尺寸分布与训练集很相似。训练集最常见的是 `720_1280_3`，但一共存在 10 种尺寸。因此我们的处理重点是统一尺寸，同时保留颜色通道，不把图片错误地当作灰度图。EDA 阶段还要记住比赛的保密要求，不能为了展示而公开材料中没有公开的样本图片。

### 2.2 数据清洗

我们其次把目录标签转换成图片路径和标签数组，并将所有图片调整到同一个较小尺寸。材料中的基础方案选择 `ROWS = 90`、`COLS = 160`、`CHANNELS = 3`，这样每张图片都会变成 `90×160×3` 的数组。缩小图片会丢掉一部分细节，但可以减少内存占用和计算时间，使基础卷积网络能够运行。

```python
# 设定统一尺寸，保证网络可以把图片组成一个四维批次数组
import os
import cv2
import numpy as np

TRAIN_DIR = '../input/train/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ROWS, COLS, CHANNELS = 90, 160, 3

def get_images(fish):
    # 用类别目录构造相对路径，路径中保留类别信息以便之后生成标签
    fish_dir = TRAIN_DIR + '{}'.format(fish)
    return [fish + '/' + im for im in os.listdir(fish_dir)]

def read_image(src):
    # 统一尺寸是为了让不同原图可以进入同一个神经网络批次
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    return cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)

files, y_all = [], []
for fish in FISH_CLASSES:
    fish_files = get_images(fish)
    files.extend(fish_files)
    # 每一张图片都继承所在文件夹的类别标签
    y_all.extend([fish] * len(fish_files))

y_all = np.array(y_all)
X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)
for i, im in enumerate(files):
    X_all[i] = read_image(TRAIN_DIR + im)
```

这里我们没有把鱼从背景中手工裁剪出来，因为材料中的基础方案没有使用额外的定位步骤，而是直接让卷积网络从整张图片学习颜色、纹理和形状。我们也没有把 `ALB` 的多余图片随意删除；数据不均衡是事实，后面通过分层划分和正则化来降低影响。读图时使用彩色模式，是因为鱼的颜色和水面、甲板的颜色可能是有用线索。

在标签方面，我们要把字符串类别转成神经网络能处理的数值，并进一步转成 one-hot 向量。所谓 one-hot，就是八个位置中只有真实类别的位置为 1，其余位置为 0。这个格式与多分类交叉熵损失匹配。

```python
# 标签编码让字符串类别变成连续整数，便于神经网络计算
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(y_all)
# one-hot 后每个标签有八个位置，和八个鱼类一一对应
 y_all = np_utils.to_categorical(y_all)

# 分层切分保证训练集和验证集都尽量保留各类别比例
X_train, X_valid, y_train, y_valid = train_test_split(
    X_all, y_all,
    test_size=0.2,
    random_state=23,
    stratify=y_all
)
```

材料中的代码使用 `test_size=0.2` 和 `random_state=23`，并指定 `stratify=y_all`。分层切分可以避免少数类别全部落到训练集或验证集之外。这里的验证集只用于本地比较，不能把测试集标签泄漏到训练过程中。

### 2.3 模型训练与超参数调优

我们接着采用材料中的基础卷积神经网络。卷积层可以把相邻像素组合起来，逐步学习边缘、纹理和更复杂的形状；池化层压缩空间尺寸；全连接层把这些特征汇总成八个类别的判断。材料说明这个基础方案没有做参数调优和数据增强，因此我们把它作为清楚、容易复现的起点。

```python
# 导入 Keras 层；卷积层负责提取图像局部特征，Dropout 用来减少过拟合
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras import backend as K

# 用全局均值和标准差做中心化，避免不同图片的亮度尺度差异过大

def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)

model = Sequential()
model.add(Activation(activation=center_normalize,
                     input_shape=(ROWS, COLS, CHANNELS)))
model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu'))
model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
```

我们首先用两层 32 通道卷积提取较基础的图案，再用两层 64 通道卷积提取更丰富的特征。随着层数增加，网络能看到更大的图像区域。材料中的完整网络还继续使用 128 和 256 通道卷积，并在末端加入全连接层和 Dropout；这比一开始就使用很大的输入图片更适合入门复现。

下面补上后半段网络以及训练配置。这里的超参数不是凭空选择，而是来自材料中的可运行方案：优化器使用学习率为 `1e-4` 的 RMSprop，批大小为 64，训练示例把 epoch 设为 1 以避免运行超时，同时设置验证损失连续一段时间不改善时停止。

```python
# 继续增加特征通道，让网络能够学习更复杂的鱼体和背景模式
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# 八个输出对应八个鱼类；输出概率才能计算 log loss
model.add(Dense(len(FISH_CLASSES)))
model.add(Activation('sigmoid'))

optimizer = RMSprop(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

材料中的代码最后使用 `EarlyStopping(monitor='val_loss', patience=4)`，并在训练时传入 `batch_size=64`、`nb_epoch=1`、`validation_split=0.2`、`shuffle=True`。这里的 `shuffle=True` 可以打乱训练样本，避免同一类别的图片连续出现；`val_loss` 与比赛的概率损失方向一致，因此比只看训练准确率更合适。材料还特别说明，如果计算资源允许，可以把 epoch 增加到约 20，这只是材料提出的改进方向，不能把它当作已经完成的最终结果。

除了卷积网络，材料中的另一条路线是 gcForest，也就是深度森林。它不依赖 GPU，而是反复使用随机森林生成概率特征，再把这些特征送入级联森林。作者从每个类别抽取 100 张样本，`LAG` 还加入了论坛发布的 Imagenet 图片；调参后报告本地验证准确率超过 75%，public leaderboard 约为 1.4 log loss。这个结果说明小样本的树模型也有价值，但材料同时说它仍不能和使用完整数据的预训练 VGG 相比。因此我们的主线仍选直接、容易解释的 CNN，并把深度森林作为小样本条件下的可替代方案。

### 2.4 模型测试与指标报告

我们最后读取 `test_stg1/` 中的图片，用和训练集完全相同的方式调整到 `90×160×3`，再调用模型输出八列概率。测试阶段不能重新拟合标签编码，也不能改变类别顺序，否则提交文件中的列会与真实类别错位。

```python
# 测试图片必须使用训练阶段相同的尺寸和颜色通道处理
TEST_DIR = '../input/test_stg1/'
test_files = sorted(os.listdir(TEST_DIR))
X_test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, file_name in enumerate(test_files):
    # 统一预处理保证训练分布和测试分布尽量一致
    X_test[i] = read_image(TEST_DIR + file_name)

# 输出不是单一类别，而是每张图片对应八个类别的概率
pred = model.predict(X_test, batch_size=64, verbose=1)

# 用题目规定的类别名称建立提交表，列顺序不能随意改变
submission = pd.DataFrame(pred, columns=FISH_CLASSES)
submission.insert(0, 'image', test_files)
submission.to_csv('submission.csv', index=False)
```

评价时我们关注 log loss，而不是只统计最大概率对应的类别是否正确。材料中卷积网络方案的标题记录了 `LB 1.279`，深度森林方案记录了约 `1.4` 的 public log loss；这两个数字只能作为材料中相应方案的公开排行榜记录，不能理解为本文重新训练后的新成绩。由于 log loss 越低越好，1.279 在该指标上优于约 1.4，但两者来自不同方案和不同实验设置，比较时还要注意数据量、预训练模型和参数是否一致。

本地测试时，我们可以先在 `X_valid` 上预测并计算验证损失，确认模型能输出合法概率，再生成测试提交。若验证集上的损失明显上升，说明网络可能过拟合；此时可以优先尝试材料提到的更长训练配合早停、数据增强或预训练网络，但这些都需要重新验证，不能直接假设一定变好。

## 三、总结

本文中我们……首先通过目录探索确认了八个鱼类类别、训练图片数量和测试图片规模，发现 `ALB` 样本明显多于其他类别，并且训练图片存在多种尺寸；其次，我们保留彩色通道，把所有图片缩放到 `90×160`，把文件夹名转换成 one-hot 标签，并用分层方式切分训练集和验证集；接着，我们按照材料中的 Keras 方案搭建卷积神经网络，用中心化、卷积、池化、Dropout 和八类别输出学习鱼类图像特征；然后，我们使用 RMSprop、交叉熵和早停控制训练，并理解了批大小、epoch 和验证损失各自的作用；最后，我们把测试图片转成八类概率并按 log loss 评价，参考了材料中 CNN 的 `LB 1.279` 和深度森林约 `1.4` 的公开记录。

这场比赛给我们的重要启发是，图像分类不能只靠堆叠更深的网络。先确认标签目录、类别不均衡和图像尺寸，再保证训练与测试预处理完全一致，往往比盲目调参更重要。若数据量或硬件有限，材料中的深度森林提供了不依赖 GPU 的选择；若资源充足，则可以在这个清晰的基础 CNN 之上继续研究预训练模型、数据增强和更合理的验证划分。