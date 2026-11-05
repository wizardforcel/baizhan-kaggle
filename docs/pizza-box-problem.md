# Pizza Box Problem（披萨盒问题）

## 一、比赛简介

### 1.1 比赛背景

这个比赛来自 Kaggle，题目叫 Pizza Box Problem，材料里的笔记本标题写作“Pizza or Not Pizza?”。翻成大白话就是：给模型看一张图片，让它判断图里是不是披萨。数据由比赛主办方准备好，放在 Kaggle 的 `/kaggle/input/competitions/pizza-box-problem` 目录下，参赛者不能使用外部图片，也不能使用别人预训练好的模型。

这个比赛最特别的地方是规则限制：**最终做预测的模型必须是支持向量机（SVM）**。PCA 这类不依赖标签的无监督预处理是允许的，但预训练模型、深度特征、外部数据统统不许用。主办方这么设计，是想把大家从“下载一个预训练卷积网络直接微调”的捷径上拉回来，逼着参赛者自己设计传统图像特征、自己挑核函数。换句话说，这是一场“带着镣铐跳舞”的比赛：工具箱被限死了，能拉开差距的就是特征工程和对 SVM 的理解。

从任务类型看，它是一个**二分类**问题；但因为评价指标只看排序，我们最后交上去的不是“是/否”，而是一个连续的分数。

### 1.2 比赛目标

比赛提供三个 CSV 文件：

- `train.csv`：训练集，包含 `image`（图片文件名）和 `label`（是不是披萨）两列；
- `test.csv`：测试集，只有 `image` 一列；
- `sample_submission.csv`：提交样例，列名同样是 `image` 和 `label`。

图片本身按文件名存放在 `train/` 和 `test/` 两个子目录里。我们要做的，是给 `test.csv` 里的每张图片算出一个分数，分数越高表示模型越认为它是披萨，然后按 `image,label` 的格式写成 `submission.csv`。

这里要特别注意：虽然列名叫 `label`，提交文件里填的**不是 0 或 1，而是 SVM 的 `decision_function` 原始分数**。原因在下一节解释。

### 1.3 评价标准

比赛使用 **ROC AUC**（Receiver Operating Characteristic Area Under Curve，ROC 曲线下面积）作为指标，越高越好，完美值是 1.0，随机瞎猜是 0.5。

用大白话解释：把测试集里所有正样本（是披萨）和负样本（不是披萨）两两配对，看模型给正样本的分数高于负样本的比例。比例越高，说明模型越能把披萨排在前面。它只关心**排序**，不关心分数具体是多少，也不要求我们自己定一个阈值把分数切成 0/1。这就是为什么提交文件里可以直接放 `decision_function` 的输出：只要披萨的分数普遍比非披萨高，AUC 就高。

也正因为如此，我们训练时用 `roc_auc_score` 来评估，而不是只看准确率。准确率需要先定阈值，阈值一变结果就变；AUC 把阈值这件事绕开了，更适合比较模型。

## 二、解题过程

整场比赛我们走的是最朴素但也最符合规则的一条路：**把图片缩放成固定大小，取原始像素，再补上一组颜色直方图；标准化后用 PCA 降维；最后交给 RBF 核 SVM 分类。** 下面按 EDA、数据清洗、模型训练与调参、测试与指标报告四步展开。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，看看数据规模以及正负样本是否均衡。

```python
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = '/kaggle/input/competitions/pizza-box-problem'
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')

print(train.shape, test.shape)
# normalize=True 输出的是比例而不是条数，方便一眼看出类别是否均衡
print(train.label.value_counts(normalize=True).rename('class balance'))
```

跑完这一步，我们知道了训练集和测试集各有多少张图片，以及两类样本大致各占多少。材料的注释里提到训练集约 1.5k 张图片，这个量级在图像任务里算很小的，后面调参时要小心过拟合。

接着我们随机抽几张图，看看它们的尺寸和颜色模式是否统一。

```python
SEED = 42
np.random.seed(SEED)

# 抽样检查图片的宽高和颜色模式，决定后面要不要统一缩放
for f in train.image.sample(5, random_state=SEED):
    img = Image.open(f'{DATA_DIR}/train/{f}')
    print(f, img.size, img.mode)
```

结果告诉我们，图片的尺寸并不完全一致，颜色模式也可能出现灰度图或带透明通道的图。SVM 吃的是固定长度的数值向量，所以我们必须把每张图都缩放到同一个尺寸，并统一转成三通道 RGB。

然后我们写特征提取函数。思路很直接：一路取缩放后的原始像素，另一路取 HSV 颜色直方图。

```python
IMG_SIZE = 64   # 统一缩放到 64×64，既保留轮廓又控制维度
BINS = 16       # 每个颜色通道分 16 个区间统计

def extract_features(path):
    # convert("RGB") 把灰度图、RGBA 图都强行变成三通道，避免维度对不上
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    # 原始像素：64×64×3 = 12288 个数，除以 255 把范围压到 0~1
    pixels = np.asarray(img, dtype=np.float32).ravel() / 255.0
    # 转成 HSV 再统计直方图：H 管色相、S 管饱和度、V 管亮度
    hsv = np.asarray(img.convert("HSV"))
    hist = np.concatenate([
        np.histogram(hsv[..., c], bins=BINS, range=(0, 255))[0] for c in range(3)
    ]).astype(np.float32) / (IMG_SIZE * IMG_SIZE)
    # 两路特征首尾拼起来，一共 12288 + 48 = 12336 维
    return np.concatenate([pixels, hist])
```

为什么要在像素之外再加颜色直方图？因为材料里明确说“颜色是判断披萨的强线索”。披萨通常有番茄酱的红色、饼底的黄色、芝士的浅黄色，这些颜色在 HSV 空间里比在 RGB 空间里更容易和背景区分开。直方图只统计“某种颜色出现多少次”，不关心它出现在哪个位置，所以对轻微的平移和旋转更宽容，正好补上原始像素对位置敏感的缺点。

特征维度一下子到了 12336，而样本只有约 1500 张，特征比样本多一个数量级。这种“宽而少”的数据直接丢给 SVM 容易过拟合，所以我们在训练前会先用 PCA 看看能压到多少维。

### 2.2 数据清洗

这份数据的“清洗”不是删行删列，而是**把所有图片整理成同一种格式**。具体来说我们做了三件事：

1. 用 `convert("RGB")` 统一颜色通道。灰度图只有一个通道，RGBA 图有四个通道，直接拼在一起会让不同样本的向量长度不一样，模型根本没法训练。
2. 用 `resize((64, 64))` 统一尺寸。原图大小不一，缩放之后每条样本都变成 64×64×3 的固定网格。
3. 用除以 255 的方式归一化。像素原本是 0~255 的整数，除以 255 后落到 0~1，数值范围变小，标准化和 PCA 都会更稳定。

我们把训练集和测试集用**同一个函数**批量处理，确保两边特征的定义完全一致：

```python
# 训练集：对每一行 image 文件名调用同一个特征函数，再用 np.stack 摞成矩阵
X = np.stack([extract_features(f"{DATA_DIR}/train/{f}") for f in train.image])
y = train.label.values

# 测试集必须走完全相同的流程，否则训练和测试的特征含义会对不上
X_test = np.stack([extract_features(f"{DATA_DIR}/test/{f}") for f in test.image])

print("X:", X.shape, "X_test:", X_test.shape)
```

这里我们刻意**没有做数据增强**，也没有做去重、删除异常样本之类的操作。原因有两个：一是 SVM 不像神经网络那样能从大量增强样本里持续受益，增强还会让训练时间成倍增加；二是这个比赛样本本来就少，每一条都可能带着有用信息，贸然删样本风险更大。我们只保证“格式统一、数值可比”，把剩下的判断交给模型。

### 2.3 模型训练与超参数调优

进入建模前，我们先解决“维度太高”的问题。我们用 PCA 把所有主成分的累计解释方差画出来，看看信息到底集中在多少维里。

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# PCA 对量纲敏感，所以先标准化，再拟合 300 个主成分
pca_full = PCA(n_components=300, random_state=SEED).fit(
    StandardScaler().fit_transform(X)
)
cum = np.cumsum(pca_full.explained_variance_ratio_)  # 累计解释方差比例
plt.figure(figsize=(6, 3.5))
plt.plot(cum)
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.grid(alpha=.3)
plt.tight_layout()
plt.show()

# 打印几个候选维数各自保留了多少信息
for k in (50, 100, 200, 300):
    print(f"{k:>3} components -> {cum[k-1]:.1%} variance")
```

这条曲线回答了一个很实际的问题：保留多少个主成分才够。如果曲线在前几十维就迅速抬升、后面变平，说明大部分信息集中在少数方向；如果一直缓慢上升，说明信息摊得很散，压得太狠会丢东西。基于曲线的形状，我们把候选值定在 50、100、200 三档，交给后面的网格搜索去选。

我们接着把标准化、PCA 和 SVM 串成一个流水线。把 PCA 放进流水线里非常关键：交叉验证时，PCA 只会在当前折的训练部分上拟合，不会偷看验证折的数据，避免了信息泄漏。

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def make_pipe(n_components=100, C=3, gamma="scale"):
    # 三步流水线：先标准化，再 PCA 降维，最后 RBF 核 SVM
    return Pipeline([
        ("scale", StandardScaler()),                                # 让每个特征均值 0、方差 1
        ("pca", PCA(n_components=n_components, random_state=SEED)), # 无监督降维，不碰标签
        ("svm", SVC(kernel="rbf", C=C, gamma=gamma)),               # RBF 核负责画非线性边界
    ])
```

为什么选 RBF 核而不是线性核？因为原始像素和颜色直方图之间的关系明显不是线性的：同样是披萨，光照、角度、背景都在变，在特征空间里它们不会乖乖排在线性边界的一侧。RBF 核可以把样本映射到更高维，让弯曲的边界也能被切开。代价是多了一个 `gamma` 参数要调。

在正式调参前，我们先切一个留出集做快速检查，确认整套流程能跑通、方向没跑偏：

```python
from sklearn.model_selection import train_test_split

# 80% 训练、20% 验证；stratify=y 保证两边的正负样本比例一致
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
pipe = make_pipe().fit(X_tr, y_tr)
# decision_function 输出的是到分类超平面的带符号距离，不是概率
val_scores = pipe.decision_function(X_val)
```

我们特意用 `stratify=y`。如果正负样本本来就不完全均衡，随机切分可能让验证集里某一类特别少，算出来的 AUC 会剧烈波动，看起来像模型不稳定，其实只是抽样偏差。

确认流程没问题后，我们开始网格搜索。搜索空间围绕三个最影响结果的参数展开：PCA 保留多少维、SVM 的正则强度 `C`、核宽度 `gamma`。

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    "pca__n_components": [50, 100, 200],          # 降维后保留的主成分数
    "svm__C": [1, 3, 10],                         # C 越大，对错分越不容忍，边界越复杂
    "svm__gamma": ["scale", 1e-3, 3e-3],          # gamma 越大，单个样本影响范围越小
}

# 5 折分层交叉验证，每折都保持正负比例；打分用 ROC AUC，和比赛指标一致
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = GridSearchCV(
    make_pipe(), param_grid,
    scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1
)
search.fit(X, y)

print("Best params:", search.best_params_)
print(f"Best CV AUC: {search.best_score_:.4f}")

# 把所有组合按平均 AUC 从高到低排，前八名用来判断参数是否敏感
res = (pd.DataFrame(search.cv_results_)
       .loc[:, ["param_pca__n_components", "param_svm__C",
                "param_svm__gamma", "mean_test_score", "std_test_score"]]
       .sort_values("mean_test_score", ascending=False))
res.head(8)
```

这个网格一共 3×3×3 = 27 种组合，每种跑 5 折，合计 135 次拟合。因为每次都要重新做标准化和 PCA，计算量不算小，所以我们用了 `n_jobs=-1` 把所有 CPU 核都开起来。

关于 `gamma="scale"`：它的含义是 `1 / (n_features * X.var())`，也就是根据特征数量和方差自动定一个尺度。另外两个固定值 1e-3 和 3e-3 是故意设得比较小，让核宽一些、边界更平滑。我们把这几种放在一起搜，是想看看数据到底需要多复杂的边界。

我们不仅看第一名，也看前八名的分布。如果前几名分数很接近，说明模型对这几个参数不敏感，选哪个都行；如果只有某一个组合特别高、标准差也大，就要警惕它是交叉验证里的偶然波动。

### 2.4 模型测试与指标报告

调参完成后，我们在留出验证集上正式算一遍指标。比赛看 AUC，我们同时算一个准确率作为辅助参考：

```python
from sklearn.metrics import roc_auc_score, accuracy_score

# 用最优流水线在验证集上重新打分
val_scores = search.best_estimator_.decision_function(X_val)
val_auc = roc_auc_score(y_val, val_scores)
# decision_function 以 0 为界：大于 0 判为正类，小于 0 判为负类
val_acc = accuracy_score(y_val, val_scores > 0)

print(f"Validation AUC: {val_auc:.4f}")
print(f"Validation ACC: {val_acc:.4f}")
```

读这两个数时要注意：AUC 衡量的是排序质量，准确率衡量的是“按 0 切一刀”之后的命中比例。因为提交只交分数，AUC 才是和排行榜一致的口径；准确率只是帮我们确认模型没有把正负完全弄反。如果准确率明显低于 0.5，通常说明标签或分数的方向搞反了，而不是模型真的那么差。

`GridSearchCV` 默认 `refit=True`，也就是说搜索结束后它已经自动用**全部训练数据**和最优参数重新拟合了一遍。我们直接拿这个 `best_estimator_` 去预测测试集，不需要再手动 `fit` 一次：

```python
best = search.best_estimator_
# 测试集同样输出连续分数，分数越高越像披萨
test_scores = best.decision_function(X_test)

# 按比赛要求的列名组织提交文件：image 是文件名，label 放分数
submission = pd.DataFrame({"image": test.image, "label": test_scores})
submission.to_csv("submission.csv", index=False)
print(submission.shape)
submission.head()
```

最后一步是提交前的格式自检。比赛里因为列名写错、行数对不上、出现空值而白白丢分的情况非常常见，所以这几行检查我们每次都会跑：

```python
sample_path = f"{DATA_DIR}/sample_submission.csv"
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    # 列名必须和样例完全一致，顺序也不能错
    assert list(submission.columns) == list(sample.columns), "Column mismatch"
    # 图片集合和行数都要对得上，不能多也不能少
    assert set(submission.image) == set(sample.image) and len(submission) == len(sample), "Image ids mismatch"
# 分数列不允许有空值，否则平台可能直接判提交无效
assert submission.label.notna().all()
print("Submission format OK")
```

自检通过，我们才认为这份提交是有效的。

如果还有时间，材料里也列了几条继续往上走的方向：换更丰富的传统特征，比如 HOG（方向梯度直方图，擅长描述轮廓）、LBP（局部二值模式，擅长描述纹理）、SIFT/ORB 加视觉词袋、Gabor 滤波器；换更适合直方图的核函数，比如卡方核或直方图交叉核；把图片尺寸、颜色空间、直方图区间数也纳入搜索；或者训练几个基于不同特征集的 SVM 做集成。这些都没有跳出“只用 SVM”的规则，属于合规的提分路线。

## 三、总结

本文中我们完成了 Pizza Box Problem 的一条合规基线。我们首先用 EDA 摸清了数据：训练集约 1500 张图片，正负两类需要先看比例，图片尺寸和颜色模式并不统一。其次我们做了统一格式的“清洗”：全部转成 RGB、缩放到 64×64、像素除以 255，并把训练集和测试集用同一个函数处理。接着我们构造了两路特征：12288 维原始像素加 48 维 HSV 颜色直方图，合计 12336 维，再用标准化和 PCA 把维度压下来。然后我们把 StandardScaler、PCA、RBF 核 SVM 串成流水线，用 5 折分层交叉验证和 `roc_auc` 指标网格搜索 PCA 维数、`C` 和 `gamma`。最后我们在留出集上报告 AUC 与准确率，用最优模型对测试集输出 `decision_function` 分数，写成提交文件，并做了列名、图片集合、行数和空值四项自检。

这个比赛给我们最大的体会是：**限制往往比自由更能逼出对方法的理解**。不能上预训练模型，我们就必须想清楚“披萨在像素和颜色上到底长什么样”，必须理解 PCA 为什么能降维、RBF 核为什么适合非线性边界、AUC 为什么只关心排序。这条基线本身不算强，但它把整个流程跑通了，也给后续换 HOG、LBP、视觉词袋这些更强的传统特征留好了接口。
