# 糖尿病视网膜病变检测（Diabetic Retinopathy Detection）

## 一、比赛简介

### 1.1 比赛背景

糖尿病视网膜病变（Diabetic Retinopathy，简称 DR）是糖尿病最常见的并发症之一——长期高血糖会损伤眼底视网膜上的血管，血管可能渗漏、堵塞，甚至长出异常的新生血管，最终导致视力受损乃至失明。资料里提到，患糖尿病 20 年以上的人中，多达 80% 会出现这种病变，而它在早期往往**没有任何明显症状**，等到视力下降再去就医，常常已经错过最佳治疗时机。

因此，定期给糖尿病人拍眼底照片、由医生判断有没有病变，是最有效的筛查手段。但人工读片既慢又依赖专业医生，在糖尿病高发、医疗资源紧张的地区尤其困难。这个比赛的目的，就是让算法**自动给眼底照片分级**，把医生从海量的重复读片中解放出来。

比赛提供的照片是用不同型号、不同质量的眼底相机拍的，光照、对焦、曝光都不统一，有的照片偏暗、有的过曝、有的模糊。比赛方明确说，数据里的图像和标签都带有噪声，参赛者要做出一个**在噪声和差异下依然稳定**的算法。

### 1.2 比赛目标

输入是一张彩色眼底照片，文件名形如 `1_left.jpeg`、`1_right.jpeg`，其中数字是**病人编号**，`left`/`right` 表示左眼还是右眼。每位病人通常左右眼各有一张照片。

每个病人的眼底要被医生打一个 0～4 的严重程度等级：

| 等级 | 含义 |
| --- | --- |
| 0 | 没有病变（No DR） |
| 1 | 轻度（Mild） |
| 2 | 中度（Moderate） |
| 3 | 重度（Severe） |
| 4 | 增殖期（Proliferative DR） |

训练集是 `trainLabels.csv`，两列：`image`（图片名）和 `level`（0～4 的等级）。测试集只有图片，我们要为每张测试图片预测一个 `level`。

这里有一个初学者常踩的坑：Kaggle 原始比赛有约 3.5 万张照片，但很多公开 notebook 挂载的只是**一份 1000 张的抽样子集**（1001 个文件 = 1000 张图 + 1 个 csv）。我们写的时候也会说明这个区别，免得读者把子集的结果和完整比赛的成绩混在一起。

### 1.3 评价标准

比赛使用**二次加权 Kappa 系数（Quadratic Weighted Kappa，简称 QWK）**。

Kappa 系数衡量的不是「猜对了多少」，而是「比瞎猜好多少」。它的想法是：先算一个「纯靠运气按类别比例随机猜」的期望一致率，再看模型的预测和真实标签有多一致，两者相减再归一化。得分为 1 表示完全一致，为 0 表示和随机猜一样，负数表示比随机猜还差。

「二次加权」的意思是：预测错得越离谱，扣分越重。比如真实是 0 级（没病）却预测成 4 级（增殖期），比预测成 1 级要罚得狠得多。这一点非常符合医学场景——把重病患者判成健康，后果远比把轻度判成中度严重。QWK 越高越好。

也正因为这个指标，光看「准确率」是会骗人的：如果 90% 的病人都是 0 级，模型全猜 0 也能有 90% 准确率，但 QWK 会接近 0，等于什么都没学到。这一点我们在后面会专门验证。

---

## 二、解题过程

整个任务的本质是**图像五分类**。我们的主线思路是：先把病人的信息从文件名里拆出来、看清数据分布，再把大图缩小成模型吃得下的尺寸，然后用卷积神经网络（CNN，一种专门处理图像的神经网络）做分类，最后用 QWK 而不是准确率来评估。

### 2.1 数据探索（EDA）

我们首先读进标签表，并从文件名里解析出病人编号，同时检查左右眼的等级是否一致：

```python
import pandas as pd
import numpy as np

# 读入标签表，只有 image 和 level 两列
df = pd.read_csv('../input/trainLabels.csv')
print(df.shape)                 # 完整比赛约 (35126, 2)
print(df['level'].value_counts())  # 看 0~4 各有多少张

# 从 "10_left" / "10_right" 里去掉后缀，得到病人编号
df['PatientID'] = df['image'].str.replace('_left', '').str.replace('_right', '')
print('不同病人数:', df['PatientID'].nunique())   # 约 17563

# 检查同一病人左右眼等级是否相同
pivot = df.pivot(index='PatientID', columns=df['image'].str[-4:], values='level')
pivot.columns = ['left', 'right']
mismatch = (pivot['left'] != pivot['right']).sum()
print('左右眼等级不一致的病人数:', mismatch)       # 约 2240
```

这里有一个很关键的发现：**同一个病人的左右眼，等级经常不一样**（约 2240 个病人如此）。这说明两件事：一是病变确实是按眼独立发展的，不能简单地把一只眼的标签抄给另一只眼；二是同一病人的两张图高度相关，做训练/验证划分时最好**按病人分组**，否则同一病人的左眼在训练集、右眼在验证集，模型等于提前「见过答案」，验证分数会虚高。

接着我们看等级分布，这是这个数据集最大的难点：

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 统计每个等级的图片数量
sns.countplot(x='level', data=df)
plt.title('Label distribution')
plt.show()

# 直接打印比例，方便看清不平衡程度
print(df['level'].value_counts(normalize=True))
```

结果是**极度不平衡**：0 级（没病）占了绝大多数，3 级和 4 级（重度、增殖期）的样本很少。这解释了为什么这个比赛不能只看准确率——模型只要学会「全猜 0」，就能拿到很高的准确率，但真正需要被识别出来的重症病人反而全被漏掉。

### 2.2 数据清洗

图像数据的「清洗」和表格数据不太一样，主要是**尺寸统一和归一化**。

原始眼底照片分辨率非常高（最小也有 2500×1900 左右），直接喂进网络会慢到无法训练。所以第一步是把所有图缩放到统一的小尺寸。我们的做法是：读图 → 双三次插值缩放到 128×128 → 像素值除以 255 归一化到 0～1。

```python
import cv2

WIDTH, HEIGHT, DEPTH = 128, 128, 3

def load_and_resize(path):
    """读一张眼底图，统一缩放到 128x128 并归一化"""
    # 用 cv2 读进来是 BGR 顺序，这里我们只关心形状，
    # 后续如果要用预训练模型需要转成 RGB
    img = cv2.imread(path)
    # 原始图非常大，必须先缩小，否则显存和时间都吃不消
    img = cv2.resize(img, (WIDTH, HEIGHT))
    # 除以 255 让像素落在 0~1，网络更容易收敛
    img = img.astype('float32') / 255.0
    return img
```

这里有一个重要的取舍。资料里的 bronze 方案做过一组对比实验：先把图缩小 20 倍训练一个小网络，准确率约 75%；换成更深的网络，准确率约 79%；再把缩小倍数改成 10 倍（保留更多细节），准确率提升到约 82%。

为什么缩小倍数这么重要？因为 DR 的关键证据——出血点、微动脉瘤、硬性渗出——都是图里**很小很小的斑点**。如果把图缩得太狠，这些斑点直接被抹掉，一张明明有病变的图看起来就和健康图没区别了，模型自然判成 0 级。这也是为什么后面有队伍愿意用更高分辨率、甚至做「渐进式放大」训练。

我们自己训练用的是 128×128，这是精度和速度之间的一个折中，适合先把整条流程跑通。

### 2.3 模型训练与超参数调优

我们先搭一个**从零开始的小 CNN**，把流程跑通、拿到第一个基准。结构很朴素：三组「卷积 → 卷积 → 池化 → Dropout」，最后接全连接层输出 5 类概率。

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam

model = Sequential()
# 第一组卷积：32 个 3x3 卷积核，提取边缘、颜色等低级特征
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))   # 池化把尺寸减半
model.add(Dropout(0.25))                     # 随机丢神经元，抑制过拟合
# 第二组卷积：64 个卷积核，学更复杂的纹理
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# 第三组卷积
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
# 5 个类别，用 softmax 输出概率分布
model.add(Dense(5, activation='softmax'))

# 学习率 1e-3，随训练轮数线性衰减
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-3, decay=1e-3 / 15),
              metrics=['accuracy'])
```

这个网络在 1000 张的子集上训练了 15 轮，验证准确率稳定在 **0.744** 左右。注意 Loss 一直在 0.85～0.97 之间波动、val_loss 甚至略有上升，说明从零训练的小网络很快就学不动了——数据太少，而且类别极不平衡。

于是我们改用**迁移学习**：拿一个在 ImageNet（上百万张自然图片）上预训练好的 ResNet50，把它的卷积层当作「通用特征提取器」直接搬过来，只重新训练最后的分类头。因为底层卷积学的是边缘、角点、纹理这些通用东西，对眼底图同样管用，所以即使只有 1000 张图也能训得动。

```python
from fastai.vision import *
from fastai.metrics import *

# 数据增强：眼底图是圆形的，可以任意旋转，所以把旋转范围开到最大
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=360,
                      max_warp=0, max_zoom=1.1, max_lighting=0.1)

src = (ImageItemList.from_df(df=df, path='./', cols='path')
       .split_by_idx(range(len(train_df) - 1, len(df)))   # 留出验证集
       .label_from_df(cols='level'))
data = (src.transform(tfms, size=sz).databunch(bs=bs, num_workers=0)
        .normalize(imagenet_stats))   # 用 ImageNet 的均值方差做标准化

# 建一个 ResNet50，先冻结卷积层只训练最后的全连接层
learn = create_cnn(data, models.resnet50, metrics=[accuracy, quadratic_kappa])
learn.fit_one_cycle(4, max_lr=2e-3)
```

这里有两个细节值得说：

1. **学习率先用 LR Finder 找**。资料里的做法是先跑 `learn.lr_finder()` 画出「学习率—损失」曲线，看到损失下降最快的位置在 `lr=2e-3` 附近，再用这个值训练。这比拍脑袋定学习率靠谱得多。
2. **数据增强要符合图像语义**。普通照片左右翻转会变成另一张合理照片，但眼底图本身近似圆形、方向无所谓，所以资料里连**竖直翻转和 360° 旋转**都打开了，等于凭空造出很多训练样本，缓解小样本问题。

针对类别不平衡，资料里还用了**过采样**：让少见的 3、4 级样本在每轮里被重复看到更多次，避免模型完全倒向 0 级。

### 2.4 模型测试与指标报告

这是这个比赛最容易被误解的地方。我们先把 QWK 实现出来：

```python
from sklearn.metrics import cohen_kappa_score

def quadratic_kappa(y_true, y_pred):
    """计算二次加权 Kappa；越高越好，0 相当于瞎猜"""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
```

然后我们看训练过程中的验证指标：

```python
# ResNet50 冻结卷积层、训练 4 轮的验证结果
# epoch | valid_loss | accuracy | quadratic_kappa
#   1   |   1.689    |  0.358   |   0.205
#   2   |   1.213    |  0.473   |   0.336
#   3   |   1.082    |  0.542   |   0.389
#   4   |   0.989    |  0.572   |   ...
```

这里就体现出准确率和 QWK 的差别了：第 1 轮准确率只有 0.358，QWK 只有 0.205，几乎等于没学到东西；到第 3 轮准确率升到 0.542，QWK 也升到 0.389。两轮之间准确率涨了，QWK 也涨，说明模型确实在学**等级顺序**，而不只是碰巧猜中大头类别。

如果模型全猜 0 级，在这份不平衡数据上准确率会很高（因为 0 级最多），但 QWK 会接近 0。所以我们要一直盯着 QWK 这个数。

bronze 方案给出的完整比赛成绩更能说明问题：他们在缩小 20 倍的小网络上做到 75% 准确率，深网络 79%，缩小 10 倍后 82% 准确率，但**提交到 Kaggle 后 LB 分数只有约 0.58**（QWK）。准确率 82% 听起来不错，换算成 QWK 却只有 0.58，差距正是被「不平衡 + 相邻等级难以区分」吃掉的。

最后生成提交文件：

```python
# 对测试集每张图片做同样的缩放和归一化，再预测
test_preds = learn.get_preds(DatasetType.Test)
# 取概率最大的类别作为等级
labels = test_preds[0].argmax(dim=1).numpy()

submission = pd.DataFrame({
    'image': test_names,
    'level': labels
})
submission.to_csv('submission.csv', index=False)
```

---

## 三、总结

本文中我们完成了糖尿病视网膜病变检测的完整流程。我们首先通过 EDA 摸清了数据：完整训练集约 3.5 万张、1.7 万多名病人，每名病人左右眼各一张，且约 2240 名病人左右眼等级不一致，标签极度偏向 0 级。其次我们做了图像清洗：把几千万像素的大图统一缩到 128×128 并归一化，同时意识到缩得太狠会抹掉出血点等关键小病灶。接着我们先从零训练一个小 CNN，验证准确率约 0.744 后遇到瓶颈，改用 ImageNet 预训练的 ResNet50 做迁移学习，配合 LR Finder、符合语义的数据增强和过采样。最后我们用二次加权 Kappa 而不是准确率来评估，看到冻结卷积层训练 4 轮后 QWK 从 0.205 升到 0.389，也看到 bronze 方案 82% 准确率在榜单上只对应约 0.58 的 QWK。

这个比赛给我们最大的启发有两点。第一，**医学图像的关键信息往往很小**，分辨率的选择直接决定模型能不能看见病灶，一味缩小图像是在扔掉诊断依据。第二，**评价指标决定你该优化什么**：在不平衡的多级分类里，准确率会被大类淹没，QWK 这种带「错得越远罚得越重」的指标，才真正贴合「别把重病患者判成健康」的临床需求。
