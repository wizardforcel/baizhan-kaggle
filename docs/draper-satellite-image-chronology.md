# 德拉帕卫星影像时间排序（Draper Satellite Image Chronology）

## 一、比赛简介

### 1.1 比赛背景

德拉帕实验室（Draper Laboratory）是一家美国的航空航天与生物医药研究机构。它提供的这场卫星影像比赛，任务很特别：**卫星会隔一段时间就对地面同一个地方拍一张照片，比赛要我们判断同一地点多个时间点的照片，到底谁先谁后**。

组织者把地面划分成许多「场景」（scene，文件名里叫 `set`），每个场景给出 **5 张**不同时间拍摄的卫星照片。这些照片不是整齐划一的：不是同一颗卫星拍的，传感器角度、分辨率、光照条件都不同，甚至有的照片里天空还有云。我们要做的，是抛开这些干扰，找出照片之间真实的先后顺序。

这个任务如果靠人工做会很主观——第一眼看上去 5 张图好像长得差不多，但云的位置、阴影的方向、土地的颜色其实都在「泄漏」时间信息。让算法从像素里自动找出这些线索，正是比赛的目的。

### 1.2 比赛目标

- `train_sm/`：**350 张** JPEG 图，对应 **70 个场景**（每个场景 5 张），文件名形如 `set175_1.jpeg`、`set175_2.jpeg`……`_1` 到 `_5` 表示随机打乱的拍摄批次编号（只是一个编号，**不代表时间顺序**）。`sample_submission.csv` 里给出了正确答案，用于我们本地评估。
- `test_sm/`：**1370 张** JPEG 图，对应 **274 个场景**（`sample_submission.csv` 正好 274 行），文件名同样没有真实顺序。
- 提交文件 `sample_submission.csv`：两列 `setId` 和 `day`。`day` 是 5 个数字的空格分隔串，例如 `1 4 3 2 5`，意思是：`_1` 号图最早拍、`_4` 号图第二早、……、`_5` 号图最晚拍。

我们要做的，就是为每个场景输出这 5 张图的正确时间顺序。

### 1.3 评价标准

比赛用 **标签排序平均精度（Label Ranking Average Precision，简称 LRAP）** 评分。它是「排序类」指标：模型对 5 张图各打一个分，然后看按分数排出来的顺序和真实时间顺序有多接近，**排对一对相邻关系就加分，全部排对为 1.0**。它不要求我们猜中某个具体类，只要求**顺序正确**，所以特别适合这种「排序任务」而不是分类任务。LRAP 越高越好。

---

## 二、解题过程

我们的主线分成两条：第一条是**经典计算机视觉路线**——图像配准（把同一场景的照片对齐）加上基于亮度的特征工程，再配一个随机森林；第二条是**深度路线**——把一组照片当成「视频帧」，用卷积 LSTM 之类的序列模型去学画面随时间怎么变。两条路都不完美，但很能说明这个任务难在哪里。

### 2.1 数据探索（EDA）

我们首先读入文件清单，确认数据规模，并把文件名里的 `set` 和编号拆出来：

```python
import pandas as pd
import glob

# 训练图 350 张，测试图 1370 张，提交样例 274 行
train_files = pd.DataFrame([
    [f, f.split('/')[-1].split('.')[0].split('_')[0],   # set175
     f.split('/')[-1].split('.')[0].split('_')[1]]      # 1~5
    for f in glob.glob('../input/train_sm/*.jpeg')])
train_files.columns = ['path', 'group', 'pic_no']
print(len(train_files))          # 350
print(train_files.head())
```

然后我们看一张图的尺寸和取值范围：

```python
import numpy as np
import matplotlib.pyplot as plt

first = plt.imread('../input/train_sm/set175_1.jpeg')
print(np.shape(first))   # (2329, 3100, 3)，三通道 RGB 彩色图
print(np.min(first), np.max(first))   # 27, 255，0~255 的字节缩放
```

训练集 350 张除以每场景 5 张，正好 **70 个场景**；测试集 1370 张对应 **274 个场景**。每张图约 2300×3100，非常大，直接训练会很慢。

接着我们做一个关键分析——看同一个场景里 5 张图整体亮度的差异。以 `set107` 为例，逐张打印平均亮度：

```python
import cv2

a = []
for l in train_files[train_files['group'] == 'set107'].path:
    im = cv2.imread(l)
    a.append(im.mean())
print(a)
# [125.576, 125.587, 125.605, 125.630, 125.675]
# 5 张图的平均亮度只在小数点后第二位变化
```

这里藏着这个任务的第一个真相：**同一场景不同日期的照片，整体上看几乎一样**。亮度、颜色都只差一点点——时间线索藏在极细微的细节里。我们看到相邻图之间确实呈单调走势，但靠肉眼和单个统计量很难稳定判断。

EDA 还给出一个重要警告：**图像没有配准（registation）**。数据探索的作者直接指出，不同照片的取景框有平移旋转，如果不对齐就把两张图的对应像素相减，得到的结果「基本只是两图的位移差」，看不出真正的时间变化。所以**传统的先配准再比较**是绕不开的第一步。

### 2.2 数据清洗

这个比赛的图像「清洗」主要是三步：归一化、降采样、配准。

**归一化。** 把像素压到 0～1 区间，顺便做亮度归一化，抵消不同传感器曝光差异：

```python
def bnormalize(mat):
    """按像素值分布做亮度归一化，让不同照片的亮度可比"""
    return (mat - mat.min()) / (mat.max() - mat.min())

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   # 转灰度，减少计算量
```

**降采样。** 2300×3100 的图太大，特征检测和训练都受不了。我们用 `INTER_AREA` 插值把图缩到一半甚至更小：

```python
def c_resize(img, ratio=0.5):
    """按比例缩小图片，INTER_AREA 适合缩小场景"""
    dim = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
```

**配准。** 对齐两幅图，才能比较「同样的地面位置」的像素变化。做法是：先用 BRISK/ORB 这类特征描述子找出两图的特征点，再暴力匹配，用 RANSAC 算出单应矩阵 `H`，最后把一张图透视变换到另一张的坐标系里。图片的配准还能把 5 张图**拼接成一张全景图**——如果某张图和其他图重叠最多、内容最一致，往往也能侧面反映它更接近「基准」时间：

```python
kp1, des1 = brisk.detectAndCompute(img1Gray, None)   # 找特征点
kp2, des2 = brisk.detectAndCompute(img2Gray, None)
matches = dm.knnMatch(des1, des2, 2)                  # 暴力匹配
# 用最近邻/次近邻距离比值滤掉误匹配
matches_ = [m for m in matches
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
H, mask = cv2.findHomography(pts2_, pts1_, cv2.RANSAC, 4.0)  # RANSAC 求单应矩阵
```

### 2.3 模型训练与超参数调优

**第一步：亮度特征 + 随机森林（经典路线）。** 思路来自配准观察：虽然整体亮度差很小，但**组内相对位置**可能有用。我们为每张图计算全图及 R/G/B 三通道的平均亮度，再算它相对本组最小值和最大值的偏移 `a`、`b`，把「亮度偏高还是偏低」压成特征，然后用随机森林预测这张图属于第几天：

```python
from sklearn import ensemble
from sklearn import pipeline, grid_search

# 每个场景内取 min/max，构造组内相对特征
df_all['group_min'] = df_all['group'].map(lambda x: df_all[df_all['group'] == x]['im_mean'].min())
df_all['group_max'] = df_all['group'].map(lambda x: df_all[df_all['group'] == x]['im_mean'].max())
df_all['a'] = df_all['im_mean'] - df_all['group_min']   # 离组内最暗的偏移
df_all['b'] = df_all['group_max'] - df_all['im_mean']   # 离组内最亮的偏移

# 随机森林 + 两折交叉验证调参
rfr = ensemble.RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=2016)
model = grid_search.GridSearchCV(rfr, {'max_depth': [6], 'max_features': [1.0]},
                                 n_jobs=-1, cv=2)
model.fit(X_train, y_train)
print('Best CV score:', model.best_score_)
# Best CV score: 0.225714285714
```

这里必须先泼一盆冷水：**5 天里随机猜的准确率是 20%，这个方案只做到 22.6%**，换一组特征（加入 R/G/B 通道的组内相对特征）的变体最高也只到 **29.1%**，另一份复制核做到 17.1%。这说明**纯亮度特征几乎抓不到时间信号**——正如 EDA 所说，5 张图看起来几乎一样，靠整图统计量不足以恢复顺序。这个比赛真正的难度就在这：*简单特征不够，要么配准做精细化比较，要么让网络自己找特征*。

**第二步：序列模型（深度路线）。** 既然 5 张图是同一地点的连续时间拍摄，把每组当作一段「视频」，用视频帧预测的思路：喂进第 `t` 张图，让它预测第 `t+1` 张图的样子。模型把每张图缩成 100×100 的灰度帧，前后两帧组成「源→目标」样本对，用 **ConvLSTM2D + Conv3D** 建模：

```python
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, BatchNormalization

def ssim_loss(y_true, y_pred):
    """结构相似性损失：衡量两张图局部亮度、对比度、结构是否相像"""
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

model = Sequential()
# ConvLSTM 把卷积和时序一起建模：既看空间结构，又看帧间变化
model.add(ConvLSTM2D(filters=256, kernel_size=(3, 3),
                     padding='same', return_sequences=True))
model.add(BatchNormalization())
model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='relu',
                 padding='same'))
model.compile(loss=ssim_loss, optimizer='adadelta')   # 用 SSIM 做损失
```

这个想法的动机是：如果模型能学会「在同一个地方，画面按时间如何演变」，那么反过来拿一组 5 张乱序照片让模型回溯，就能推出合理的时间顺序。不过资料里这份 notebook 只到搭建阶段，没有跑出成型的排序分数——它更像作者对方向的探索，也提示我们这类新型任务当时连方向都在摸索。要拿到合理分数，必须回到**配准 + 精细比较**的路子上（阴影方向、云的位置、植被变化的单调特征），而不是全局统计量。

### 2.4 模型测试与指标报告

我们先在训练集上评估随机森林方案。因为标签是 1~5 的「第几天」，最直观的自评就是**预测对的图片占比**：

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_val)
print('Validation accuracy:', accuracy_score(y_val, y_pred))
# 结果只在 0.20 上下（随机 20%），说明亮度特征不可靠
```

真正的比赛指标 LRAP 会按「排序对多少对相邻关系」来算，比「全对准确率」宽容一点，但亮特征方案预测概率时，按概率降序排出的顺序也基本接近随机——所以它的 LRAP 也好不到哪去。这个结果再次确认：**我们不能把「看起来最简单」的统计特征当作答案**。

生成提交文件是按场景输出的，格式很关键：

```python
# predict_proba 得到每张图属于 1~5 天的概率
df['day1':'day5'] = y_pred

with open('submission.csv', 'w') as f:
    f.write('setId,day\n')
    for sid in df['setId'].unique():
        df1 = df[df['setId'] == sid].copy()
        order = []
        # 对每个候选"天"，按预测概率降序选出当天最可能的图，依次落实顺序
        for j in range(1, 6):
            df1 = df1.sort_values(by='day' + str(j), ascending=False)
            order.append(str(df1['pic_no'].iloc[0]))
            df1 = df1.iloc[1:]          # 已经占用的图要移出候选
        f.write(str(sid) + ',' + ' '.join(order) + '\n')
```

---

## 三、总结

本文中我们完成了德拉帕卫星影像时间排序的完整流程。我们首先通过 EDA 摸清了数据：训练集 350 张图、测试集 1370 张图，每 5 张同场景照片构成一组，图是 2300×3100 的三通道彩色大图、未做配准，且同组照片整体色调几乎一样。其次我们做了图像清洗：亮度归一化、按 INTER_AREA 降采样，并用 BRISK 特征 + RANSAC 单应矩阵把照片对齐、甚至拼成全景。接着我们试了三条路子——基于组内相对亮度的特征 + 随机森林（交叉验证准确率只有 22.6%，接近随机），把每组当视频帧的 ConvLSTM + Conv3D 序列预测（资料只搭到模型阶段），以及靠配准比较阴影、遮挡的精细路线。最后我们按 `setId,day` 的格式生成按概率排序的提交，并解释了为什么亮度统计特征在这个比赛里必然失败。

这个比赛给我们最大的启发是：**「时间」是一种极微弱的信号，粗暴的统计特征根本抓不住它**。5 张卫星图肉眼几乎无法分辨先后，靠整图平均亮度只能做到接近随机；只有配准到像素级、比较阴影的方向和长度、云和植被的逐块变化，或者让卷积网络自己从局部纹理里找时间线索，才有希望。对任何「排序」类视觉任务，先把**对同一目标做精确对齐**做扎实，永远比堆模型更值得。