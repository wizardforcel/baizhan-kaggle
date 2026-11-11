# 癫痫发作预测（American Epilepsy Society Seizure Prediction）

## 一、比赛简介

### 1.1 比赛背景

癫痫是一种神经系统疾病，患者的大脑会毫无征兆地突然异常放电，引发抽搐甚至意识丧失。如果能在癫痫发作的**前几分钟**给出预警，病人就可以提前坐下、找人帮忙、或者吃药，能大大降低受伤风险。这个比赛就是美国癫痫学会（American Epilepsy Society）发起的，目标是用脑电图（EEG）提前判断「癫痫快要发作了」。

脑电图（EEG）是把电极贴在头皮上，记录大脑神经活动的电压变化。正常人脑的信号和癫痫发作前后一段时间的大脑信号，在某些统计特征上是不一样的。如果模型能从 EEG 里学会区分这两者的差别，就能实现真正有用的「随身癫痫预警器」。

比赛的难点主要有三个。第一，EEG 信号每分钟有 30 万个采样点，数据非常大，需要先想好怎么压缩和提炼。第二，脑电信号非专业人士很难看懂，神经科学的解释非常复杂。第三，最常见的干扰是两类样本比例严重失衡——没发病的「发作间期」样本远远多于快要发病的「发作前」样本，模型很容易偷懒、把所有样本都猜成没发病。

从数据的文件组织来看，训练数据按照对象（subject）分目录存放，包括 `Patient_1`、`Patient_2` 两个人，以及 `Dog_1` 到 `Dog_5` 五条狗。材料里一致采用的做法是：先只拿 `Patient_1` 的数据把一个分类流水线跑通，再考虑推广到其他对象。

### 1.2 比赛目标

每个 `.mat` 文件都是一段 EEG 记录，去除常见的文件头信息后，文件里面实际存了五种东西：

- `data`：真正的 EEG 数据矩阵，形状是「电极数 × 采样点数」；
- `data_length_sec`：这段记录的时长，单位秒；
- `sampling_frequency`：采样频率，也就是每秒记录多少个点；
- `channels`：每个电极的名字；
- `sequence`：这段信号在整个记录里的序号。

以 `Patient_1_preictal_segment_0001.mat` 为例，我们读出来发现数据矩阵的形状是 `(15, 3000000)`，也就是 15 个电极，每段 300 万个采样点。为什么是 300 万？因为 `data_length_sec` 是 600 秒（10 分钟），采样频率是 5000 赫兹，两者相乘正好是 3000000。换句话说，每个 `.mat` 文件是一段 **10 分钟、15 个电极、每秒采样 5000 次** 的脑电记录。

文件分为两种：文件名带 `interictal` 的表示「发作间期」，也就是离发作还很远、大脑处于正常状态；文件名带 `preictal` 的表示「发作前」，也就是癫痫很快就要发作。我们要做的，就是给一段 EEG 判断它属于这两类中的哪一类（二分类）。测试集里文件名带 `test_segment` 的样本没有标签，我们要把预测结果按 `sampleSubmission.csv` 的格式填好交上去。

### 1.3 评价标准

官方评分细则在材料里没有直接写明，我们只看到数据集目录下有一个 `sampleSubmission.csv` 模板，说明比赛要求对每个测试文件提交预测结果。从材料里的代码看，作者衡量模型好坏用的是**损失（loss）和准确率（accuracy）**两个数字：loss 越小越好，accuracy 是「预测对的样本数 ÷ 总样本数」，越接近 1.0 越好，随机乱猜是 0.5。我们下面也沿用这一套标准，把准确率当作最重要的参考指标。

---

## 二、解题过程

我们的整体思路是：**把原始的脑电时间序列切成 1 秒的小段，转换成「语谱图」这种二维图像，再喂给卷积神经网络（CNN）做二分类。** CNN 天生擅长在图像里找局部模式，而语谱图恰好能把「不同频率随时间的变化」显示成一张图。我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先要知道 `.mat` 文件内部长什么样。这种文件是 MATLAB 的二进制格式，Python 里要用 `scipy.io` 的 `loadmat` 来读。我们拿 Patient_1 的一段「发作前」数据开刀：

```python
# 从 scipy 里导入 loadmat，它负责解析 .mat 格式的文件
from scipy.io import loadmat
import numpy as np

# 指定要读的文件：Patient_1 的一个 preictal（发作前）样本
preictal_path = '/kaggle/input/seizure-prediction/Patient_1/Patient_1/Patient_1_preictal_segment_0001.mat'
preictal_data = loadmat(preictal_path)   # loadmat 把整个文件读成一个 Python 字典

# 字典里键名含 “_segment_” 的那一项才是数据本体，其余是版本号等杂物
segment = dict(filter(lambda item: '_segment_' in item[0], preictal_data.items()))
seg_values = list(segment.values())

# 依次取出：EEG 矩阵、时长（秒）、采样频率、电极名列表
data = seg_values[0][0][0][0]
data_length_sec = seg_values[0][0][0][1]    # 600，也就是 10 分钟
sampling_frequency = seg_values[0][0][0][2] # 5000 赫兹
channels = seg_values[0][0][0][3]           # 15 个电极的名字列表

# 验证我们的推断：电极数 15，采样点数 = 600 秒 × 5000 次/秒 = 3000000
print(data.shape)          # 输出 (15, 3000000)
```

打印出的形状和我们预想的一样：15 行（电极）乘 300 万列（采样点）。这说明每个文件的信息量非常大——如果直接拿 300 万个点去训练，任何模型都会慢得没法用，所以 EDA 的结论就是：**必须先把 10 分钟的原始信号切成小块，并提炼出有代表性的特征。**

接着，我们检查所有文件是不是「规规矩矩」的，防止有的文件被截断或者字段缺失。我们遍历整个输入目录，逐个核对「采样点总数 = 时长 × 采样率」这条公式：

```python
import os
import scipy.io

search_key = '_segment_'   # 用来在字典里找到数据内容的关键词

# 递归遍历整个输入目录，检查每个 .mat 文件
for dirname, _, filenames in os.walk('/kaggle/input/seizure-prediction/'):
    for filename in filenames:
        if filename != 'sampleSubmission.csv':      # 跳过提交模板
            try:
                seg_dict = scipy.io.loadmat(os.path.join(dirname, filename))
                seg = dict(filter(lambda item: search_key in item[0], seg_dict.items()))
                values = list(seg.values())
                # 关键三个数字：采样点总数、时长（秒）、采样频率
                n_samples = values[0][0][0][0].shape[1]
                seconds = values[0][0][0][1]
                fs = values[0][0][0][2]
                # 采样点总数应当严格等于「时长 × 采样率」，不然文件就有问题
                if n_samples != round(seconds * fs):
                    print('数据不一致：', filename)
            except:
                print('读取失败：', filename)
print('检查完成')
```

跑完整个目录，所有文件都满足「采样点数 = 时长 × 采样率」，说明数据整体是干净的。另外一个有意思的发现是：15 个电极在 `channels` 里并不是按相邻位置排列的——比如 `LD_1`、`LD_3`、`LD_4` 这些名字说明电极分布在左右两侧，中间跳过了一两个。这提醒我们，在处理时不要假设电极之间有固定的空间顺序，简单的方式是把每个电极当成独立的一路信号。

### 2.2 数据清洗

原始 EEG 是纯时间序列，直接把这种序列塞给卷积网络效果不好——因为 CNN 擅长的是「找局部小图案」，而一维的电压值里并没有明显的图案。我们做的最关键的清洗步骤，是把原始信号转成**语谱图（spectrogram）**。

什么是语谱图？把一段信号每隔一小段切成一个窗口，对每个窗口做傅里叶变换，算出它包含哪些频率、各频率的强度是多少，然后把「时间 × 频率」画成一张二维热力图。癫痫发作前后，某些频段的能量会规律性地变化，这些变化会以明暗图案的形式呈现在语谱图里，刚好是 CNN 擅长捕捉的东西。

我们把每个 `.mat` 文件按每 5000 个采样点（正好 1 秒）切一个窗口，只取第一个电极的数据，生成一张语谱图，再统一归一化到 0～1，方便网络训练：

```python
from scipy.signal import spectrogram
import numpy as np

all_X, all_Y = [], []
types = ['Patient_1_interictal_segment', 'Patient_1_preictal_segment']

for i, typ in enumerate(types):                 # i=0 是 interictal，i=1 是 preictal
    for j in range(18):                          # 每类只取前 18 个文件，保持两类数量均衡
        fl = '/kaggle/input/seizure-prediction/Patient_1/Patient_1/{}_{}.mat'.format(
            typ, str(j + 1).zfill(4))           # zfill(4) 把编号补成 0001、0002 这种格式
        data = scipy.io.loadmat(fl)              # 读入 .mat 文件
        key = typ.replace('Patient_1_', '') + '_'   # 拼出数据字段名
        d_array = data[key + str(j + 1)][0][0][0]   # 取出 (15, 3000000) 的矩阵

        for m in range(0, 3000000):              # 每 5000 个点 +1 秒切一窗，共 600 窗
            p_secs = d_array[0][m:m + 5000]      # 先只用第一个电极（LD_1）
            f, t, Sxx = spectrogram(p_secs, fs=5000,
                                    return_onesided=False)  # fs 告诉函数采样率
            p_SS = np.log1p(Sxx)                 # 取 log 压一压强度，避免大数值主导训练
            arr = p_SS / np.max(p_SS)            # 除以最大值，把像素值缩到 0~1
            all_X.append(arr)                    # 收集语谱图
            all_Y.append(i)                      # 收集标签 0 或 1
```

另外，我们在另一个 EDA 论坛本里也看到类似的标准化思路：用 `scipy.stats` 的 `zscore` 对语谱图按频率方向做标准化，让每个频段的强度都变成「离均值几个标准差」，相当于把所有通道拉到同样的尺度。两种做法目的一致——都是让数值范围稳定，防止网络因为某个频段数值特别大而偏科。

清洗过后，每张语谱图的大小是 256×22 的灰度图像，一共生成了 21600 张（3600 张 × 6 组，每组 18 个文件的前 6 个我们做了窗口切分后每文件 600 秒）。其中 interictal 和 preictal 各占一半，两类已经对齐，不会出现一类样本把另一类淹没的问题。

### 2.3 模型训练与超参数调优

把特征准备好之后，我们首先把 21600 张语谱图**打乱顺序**，避免出现「前半全是发间期、后半全是发作前」这种灾难性的排列，然后按 21000 张训练、600 张测试来切分。切分完还要把两张语音图在边上补一个「通道」的维度、把标签转成 one-hot 编码，卷积网络才会接受：

```python
import random
import numpy as np

# 把「图、标签」一一配对后整体打乱，否则训练/测试集合会按类别分家
dataset = list(zip(all_X, all_Y))
random.shuffle(dataset)
all_X, all_Y = zip(*dataset)

# 前 21000 张当训练集，剩 600 张当测试集（本里是这么切的）
x_train = np.array(all_X[:21000])
y_train = np.array(all_Y[:21000])
x_test = np.array(all_X[21000:])
y_test = np.array(all_Y[21000:])

# CNN 输入需要「高度 × 宽度 × 通道」四维（再加 batch 维），灰度图通道数为 1
batch_size = 128
num_classes = 2
epochs = 30
img_rows, img_cols = 256, 22      # 这就是语谱图的高和宽

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# 把整数标签 0/1 变成 one-hot：0 -> [1,0]，1 -> [0,1]
from tensorflow import keras
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

接着我们搭建这个比赛里被反复验证好用的网络结构：**先用两层卷积提取图像局部特征，再接池化压缩尺寸，最后用带 Dropout 的全连接层做分类**。超参数（32 个卷积核、5×5 和 3×3 的卷积核尺寸、0.25 和 0.5 的丢弃率）都来自材料的实验记录：

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
# 第一层卷积：用 32 个 5×5 的小窗口扫描图像，抓局部模式；relu 加速训练
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
# 第二层卷积：卷积核换小成 3×3，继续组合出更抽象的特征
model.add(Conv2D(32, (3, 3), activation='relu'))
# 最大值池化：每 2×2 区域只留最大值，图尺寸减半，省计算量也抗过拟合
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))     # 随机丢掉 25% 神经元，防止网络死记训练集
model.add(Flatten())         # 把二维特征拉成一维，才能接全连接层
model.add(Dense(32, activation='relu'))   # 全连接层综合所有特征
model.add(Dropout(0.5))      # 分类前再丢一半，进一步正则化
model.add(Dense(num_classes, activation='sigmoid'))  # 输出 2 个 0~1 概率

# 二分类用二元交叉熵作损失，RMSprop 优化器 + 准确率指标
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()   # 打印每层的形状和参数量，核对网络有没有搭错
```

为什么这个结构有效？因为语谱图上的「频率强度」在图上表现为一块块的明暗区域，第一层 5×5 卷积负责认出「这一块是不是一个亮点」，第二层 3×3 卷积负责把零散亮点组合成更大的结构（比如某段频谱能量的持续变亮），池化层负责把图压缩、保留最突出的信号，最后全连接层再综合全局信息给出「发作前」还是「发作间期」的结论。Dropout 之所以重要，是因为我们的样本只来自 18 个文件、同类之间有很强的相似性，网络很容易把「某个文件」的特征背下来而不是学到真正的「发作前模式」。

### 2.4 模型测试与指标报告

训练时我们直接把 600 张测试图作为 `validation_data` 每轮验证一次，这样可以看到准确率随轮数上升的过程；训练结束后再用 `evaluate` 正式打分：

```python
# 开始训练：每轮结束后都会在 validation_data 上算一次 loss 和准确率
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,     # 每次喂 128 张图，分批次更新权重
    epochs=epochs,             # 把全部数据过 30 遍
    verbose=1,                 # 打印每一轮的进度
    validation_data=(x_test, y_test)   # 每轮都用没见过的 600 张图验证
)

# 训练完毕，在测试集上正式评估
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])       # 0.1305
print('Test accuracy:', score[1])   # 0.9467
```

材料提供的训练日志显示，模型在验证集上的表现是一路爬升的：前几轮验证准确率只有 0.63～0.74，从第 11 轮开始稳定超过 0.9，第 22 轮时验证准确率达到最高的 **0.9517**。训练结束时，在 600 张从未见过的测试语谱图上，最终 `Test loss` 为 **0.1305**，`Test accuracy` 高达 **0.9467**——也就是说，模型能正确区分约 94.7% 的「发作前」和「发作间期」1 秒窗口。

这个结果说明「语谱图 + CNN」这套组合在单个病人（Patient_1）的数据上是有效的。不过我们也要清醒地看到它的局限：这个分数是在 1 秒的小窗口级别上算的，而比赛要求的是对整段 10 分钟文件做判断；更重要的是，我们只在 Patient_1 上训练和测试，而比赛数据里有 7 个对象、电极数量和摆放都不完全一样，单病人模型直接搬到其他对象身上很可能掉分。材料里的另一个方案（微博本里的 PyTorch CNN+GRU）之所以额外用 30 秒的窗口和循环神经网络，就是想通过更长的上下文信息来提升对整段信号的判别能力。

**提交文件。** 确认模型可用后，我们应该把全部训练数据重新训练一遍模型，再对每个 `test_segment` 文件切窗、算语谱图、走同样的前向预测，把每个文件的概率平均成对整段的打分，最后按 `sampleSubmission.csv` 的列名填好保存。因为材料里只演示到测试集打分这一步，真正的提交环节我们保守地按这个流程来组织。

---

## 三、总结

本文中我们完成了癫痫发作预测（American Epilepsy Society Seizure Prediction）的完整流程。我们首先做 EDA，搞清楚每个 `.mat` 文件是 10 分钟、15 个电极、每秒 5000 次采样的脑电记录，并通过全目录扫描确认数据是干净的；其次我们做了数据清洗和特征工程，把原始时间序列切成 1 秒窗口并转成语谱图，还归一化到 0～1；接着我们搭建了一个「两层卷积 + 池化 + 全连接」的小型 CNN，用 21000 张语谱图训练、600 张验证；最后我们在测试集上得到 94.67% 的分类准确率，说明「语谱图 + CNN」对单个病人的发作前/发作间期判别是有效的。材料里还有一些我们没展开的替代探索，包括小波分解（用 pywt 把信号分解成不同频带后再统计）以及用 PyTorch 实现的 CNN+GRU 结合 30 秒窗口的方案，都是对原始信号做「频率信息压缩」思路的变体。

这个比赛给我们的最大启发是：**对原始信号型数据，直接丢给神经网络的收益远不如先做一步「能保留任务相关结构」的特征变换。** 语谱图把「频率随时间变化」这种人对图像直觉上最容易理解、CNN 也最容易学习的结构显式地画了出来，这是它能一击即中的根本原因。同时我们也要记住单病人模型的边界——在真实场景里，泛化到不同对象、不同电极布局，往往比在单个人身上刷到高分更难。