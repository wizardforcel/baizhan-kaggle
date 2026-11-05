# 人体活动数据聚类（Clustering Physical Activity）

## 一、比赛简介

### 1.1 比赛背景

智能手环、运动手表这些可穿戴设备能用传感器感知人在「做什么动作」：是走路、跑步、骑车，还是躺着不动。有了这层信息，就能做健康监测、运动量统计等应用。这个比赛给了我们一组真实的传感器读数，任务是**把不同的活动自动分组**——注意，不是分类，因为**标签是没有提供的**。这是一道无监督学习的聚类题。

数据来自绑在三个部位的 IMU（惯性测量单元）传感器和心率计：

- **手**上（第 3–19 列）：手部传感器的温度、3D 加速度（±16g 和 ±6g 两档）、3D 陀螺仪（测旋转角速度）、3D 磁力计，共 17 列；
- **胸**上（第 20–36 列）：同样的 17 列；
- **脚/踝**上（第 37–53 列）：同样的 17 列；
- 另外还有一列**心率**（次/分）和**时间戳**。

「为什么同一类传感器要量两档加速度？」±16g 档量程大，能捕捉剧烈甩动而不饱和；±6g 档量程小但更细，安静动作读得准。两个一起看，能区分急动和缓动。

### 1.2 比赛目标

每一行是一个采样时刻（大约每秒一个时间点），共 53 列。输入是这些传感器读数，输出是一列整数 `activityID`——即「这一行属于第几类活动」。聚类结果本身没有固定编号顺序，`activityID=0` 不一定对应「走路」，只要**同一种活动标成同一个数字**就算对。

数据里有一个必须警惕的坑：`subject_id`（受试者编号）列。如果把它当特征，模型十有八九会按「人」分组，而不是按「活动」分组——因为每个人的传感器特性不同，聚类算法最容易抓到的差异反而在人与人之间。所以必须把它丢开。

评价指标就是**准确率（accuracy）**：主办方手里藏着真实的活动标签，把我们给的聚类编号和真实标签做最佳匹配后，比对上的比例。分数越高越好。因为聚类拿到的编号和标签只是「顶上号」的关系，所以准确率其实测的是「分组结构对不对」。

### 1.3 评价标准

没有真实标签，怎么挑模型？比赛中两队选手的做法给了我们两个层次的答案：

- **内部指标**：Silhouette（轮廓系数，每个点到自家簇比到邻家的近多少）、Davies–Bouldin（簇内紧凑、簇间分离，越低越好）、Calinski–Harabasz（簇间散度相对簇内散度的比，越高越好）。这些只看特征就够算。
- **外部分数**：直接把聚类结果提交到 Kaggle 排行榜，拿 accuracy。这也是唯一能验证「分组到底对不对」的标准。

一个关键教训是：**两个标准经常打架**。比如轮廓系数高的 DBSCAN，在排行榜上的准确率反而低；轮廓系数差的 KMeans 设置，排行榜分却更高。所以最终选型要「以排行榜为准」，内部指标只用来做初筛。

---

## 二、解题过程

我们采用的是一条被材料验证过的主线：**先做 EDA 摸清缺失规律，其次用「中位数填补 + 手工特征」清洗和增强，然后用 PCA 降维、用肘部法选 K，接着在 KMeans / GMM / 层次聚类之间对比，最后用 MiniBatchKMeans 在完整数据上重训并生成提交文件。**

### 2.1 数据探索（EDA）

第一步是读数据，看形状、类型、缺失、重复和描述统计，并检查每个受试者的样本量：

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df_raw = pd.read_csv('/kaggle/input/clustering-physical-activity/'
                     'Physical_Activity_Monitoring_unlabeled.csv')
print('行 x 列:', df_raw.shape)          # (534601, 53)
print('重复行:', df_raw.duplicated().sum())
print('缺失总数:', df_raw.isna().sum().sum())

# 每个受试者的样本量，看参与者是否均衡
df_raw['subject_id'].value_counts().sort_index()
```

数据显示：约 **53.4 万行**，没有重复，但缺失不少（共 12.5 万个空值）。不同受试者的样本量从 5 万到 7.3 万不等，不均衡。时间戳的差分布说明数据是**连续的时间序列**（有些间隔还是负的，疑似同步误差），不是独立同分布的测量。

然后我们重点分析缺失。材料里有一个非常敏锐的发现：

```python
# 缺失按位置统计：它们集中在哪个部位？
na = df_raw.isna().sum()
print(na[na > 0].sort_values(ascending=False).head(20))
```

结果几乎全部集中在 **`hand*`（手上的传感器）** 列，胸和踝几乎不缺。合理的解释是：**受试者中途把手环摘了**，但胸带和踝带一直戴着。这个「缺失不是随机的」事实决定了填补策略——不能把所有部位一视同仁。

我们还用 IQR（四分位距）法数了各列的离群点，并画了箱线图，结论是：离群点大多是激烈的运动（冲刺、跳跃）造成的，**不是脏数据**，不能删。

最后看相关矩阵。51 个特徵两两之间相关性很高（同一部位的加速度、陀螺仪本就有物理关联）。材料强调：输入列之间高度多重共线，直接喂给基于距离的聚类会「重复放大」某些重复信息，所以需要降维。

### 2.2 数据清洗

清洗分三步：填缺失、造特征、缩放到同一量纲。

**第一步：填补缺失。** 材料里两个 notebook 的细节不同，但更稳的做法是「按受试者分组填中位数」——因为缺失集中在手部传感器，而每个人的手部基线不同，用全表均值会抹掉个体差异；用 `groupby('subject_id').transform` 分组中位数更合理。粗略版本直接填全表 `median()` 也能用，缺失占比很小（最多每人大约 1.5% 的样本），不至于明显带偏聚类。

```python
# 每个受试者内部，用各自的中位数填补（分组中位数比全表中位数更贴合个人基线）
df_filled = df.groupby('subject_id').transform(
    lambda g: g.fillna(g.median())
)
assert df_filled.isna().sum().sum() == 0
```

**第二步：特征工程。** 三轴传感器的关键是**取模长**——向量的长度描述「动得多猛」，与身体朝向无关，而聚类最怕朝向不同造成同一种活动被拆散。具体做三类特征：

```python
# 1) 模长：sqrt(x^2+y^2+z^2)，衡量剧烈程度，与方向无关
X['handAcc16_mag']  = np.linalg.norm(X[['handAcc16_1','handAcc16_2','handAcc16_3']], axis=1)
X['handGyro_mag']   = np.linalg.norm(X[['handGyro1','handGyro2','handGyro3']], axis=1)
X['ankleAcc16_mag'] = np.linalg.norm(X[['ankleAcc16_1','ankleAcc16_2','ankleAcc16_3']], axis=1)

# 2) 部位间比值：不同活动对上半身/下半身的负荷分配不一样
#    例如骑车腿动得多手不动，比值就能把这个区别放大
EPS = 1e-8   # 防除零
X['ratio_hand_ankle_acc16_mag'] = X['handAcc16_mag'] / (X['ankleAcc16_mag'] + EPS)

# 3) 两档加速度的差异：急停急起时 ±16g 与 ±6g 读数差会被放大
X['acc_sensor_diff'] = np.linalg.norm(
    X[['handAcc6_1','handAcc6_2','handAcc6_3']].values
    - X[['handAcc16_1','handAcc16_2','handAcc16_3']].values, axis=1)
```

材料里也记录了「试了就掉分」的负实验：对极度值做 winsorize 截断、把多传感器做 sensor fusion（传感器深度融合）都让榜单分变差。这说明特征不是越多越好，**有没有效要看排行榜**。

**第三步：标准化。** KMeans 靠欧氏距离分簇，量纲大的特征（加速度值动辄几十）会主导距离，量纲小的特征（温度、心率）等于被忽略。所以必须用 `StandardScaler` 把每个特征变成零均值、单位方差。材料还试过 `RobustScaler`，效果差（分出「一大簇 + 一堆碎簇」），弃用。

```python
# 标准化：每个特征减去均值、除以标准差，消除量纲差异
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_fe)
```

### 2.3 模型训练与超参数调优：特征降维与选 K

**降维。** 51 个特征（工程后更多）高度相关，直接聚类会浪费时间又容易被噪声主导。我们做 PCA（主成分分析）：找到少数几个「综合方向」，让第一个方向抓住最大方差、第二个抓住次大，以此类推。取能解释 **95%** 方差的成分数：

```python
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print('原维度:', X_scaled.shape[1],
      '-> PCA 后:', X_pca.shape[1])     # 51 -> 43 个主成分
```

从 51 维降到 43 维，看似只少了一点，但实际效果是：截掉的部分主要是噪声，剩余方向反映真实的物理结构。这个选择还保留了绝大部分信息，不至于过度压缩。

**选 K（簇的个数）。** 聚类最棘手的超参数是 K。材料用了两个信号：

1. **肘部法**（看簇内平方和 WCSS 的拐点）：随着 K 增加，每个点离自己簇中心越来越近，WCSS 单调下降；「拐点」之后下降变缓，说明再加簇收益很小。实现上用一个几何作法：把曲线归一化后，找离「首尾连线」最远的点。
2. **轮廓系数曲线**：在 K=2..20 上扫轮廓系数。材料发现**轮廓系数在大范围上几乎随 K 单调上升**，最好的 K 落在搜索边界上——说明这些活动的形状不是「标准球状簇」，轮廓系数在这里不能单独做决定。

最终我们把 **K 定在 9**，依据是肘部法在 K=9 附近有一个不太明显但可辨的拐点。材料特意强调：作者试过 K=11 会拿到略高的排行榜分（0.28617 vs 0.27428），但那是「照答案调」出来的，原则不允许，所以还是提交 K=9。

```python
# 简化版肘部法：直接看 WCSS 曲线找拐点（这里示意核心思路）
ks = range(4, 21)
wcss = []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(sub_sample)                # 在 1.5 万行子样本上扫，快
    wcss.append(km.inertia_)          # inertia = 簇内平方和
# 拐点在 K=9 附近，我们据此确定 n_clusters=9
```

**算法对比。** 在 K=9、同一份 1.5 万行子样本上，我们并排跑了 KMeans、高斯混合模型（GMM）和层次聚类（Agglomerative，ward 连接），用三个内部指标和训练时间一起看：

```python
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

def evaluate(name, labels, X):
    return {
        'model': name,
        'silhouette':    round(silhouette_score(X, labels), 4),
        'davies_bouldin': round(davies_bouldin_score(X, labels), 4),
        'calinski':      round(calinski_harabasz_score(X, labels), 1),
    }
```

结果很有意思：

- **Agglomerative**：轮廓最好（0.157），但它算距离的复杂度是 $O(n^2)$，53 万行上根本跑不完；这次实验里它还常常退化成「1 个巨簇 + 若干碎簇」。不是实用选项。
- **GMM**：三个指标都垫底，还最慢。
- **KMeans**：轮廓略低，但 Davies–Bouldin（2.17）和 Calinski–Harabasz（1063）两个「紧凑度/分离度」指标最好，而且速度快、能上全量数据。

于是我们选择 KMeans 家族，并在全量数据上改用 **MiniBatchKMeans**——它每次只拿一小批点更新簇中心，比普通 KMeans 快得多，适合 53 万样本。

### 2.4 模型测试与指标报告

最终模型在**完整**的 PCA 空间中训练：

```python
final = MiniBatchKMeans(
    n_clusters=9,
    batch_size=4096,     # 每批 4096 行，内存友好
    n_init=3,            # 初始化 3 次取最好，降低偶然性
    max_iter=200,
    random_state=42,
)
labels = final.fit_predict(X_pca)

# 检查每个簇的占比，防止出现几个点的废簇
print(np.bincount(labels) / len(labels))
```

簇之间大小差异很大，这符合直觉：像「坐着不动」这样的状态在传感器数据里占比极高，而「剧烈冲刺」只占极小一部分。

**编号对齐。** 聚类产生的标签顺序是任意的（第 3 簇不一定要排在第 2 号），为方便主办方核对，也为了提交文件好看，我们把标签重排成第一次出现顺序：先给第一个出现的簇编 0 号，再给下一个新簇编 1 号……

```python
# 按「第一次出现」的顺序重编号（0,1,2,…），保持分组结构不变
unique, idx_first = np.unique(labels, return_index=True)
order = unique[np.argsort(idx_first)]
label_map = {old: new for new, old in enumerate(order)}
labels_ordered = np.vectorize(label_map.get)(labels)

submission = pd.DataFrame({
    'Index': np.arange(len(labels_ordered)),
    'activityID': labels_ordered,       # 与样例文件的列名一致
})
submission.to_csv('submission.csv', index=False)
```

**测试与排行榜。** 提交到 Kaggle 后，K=9 的 MiniBatchKMeans 拿到准确率约 **0.274**，K=11 能到 0.286，但那是「对着排行榜调」出来的，材料明确不用它。另一份 notebook 的做法是 KMeans(K=8)，排行榜分 **0.379**，比我们这个稍微再高一些，但框架完全一致：中位数填补 → 模长/比值特征工程 → StandardScaler → PCA → KMeans。还有一份用了 DBSCAN，轮廓更高但排行榜分只有 0.214，再次印证「内部指标和外部评分不完全一致」。

我们都注意到了：排行榜当前的顶尖成绩约在 **0.85** 附近，说明我们的预处理还是太基础。材料给的方向是：更好的特征工程（比如把时间序列片段化、按活动片段聚合统计量）、更好的降维或聚类算法。但作为一道可完整复现的聚类练习，这条「清洗 → 特征 → 降维 → 选 K → 对比算法 → 提交」的流水线本身是自洽和诚实的——**没有为了分好看去偷看答案的 K**。

---

## 三、总结

本文中我们完成了人体活动传感器数据的聚类任务。我们首先做了 EDA：53.4 万行、51 个可用特征，缺失值高度集中在手部传感器（应该是手环被摘了），时间戳显示这是连续时间序列不是独立样本，还确认离群点大多是激烈运动而非噪声。其次做了数据清洗：按受试者分组用中位数填补手部缺失，构造三轴模长、部位间比值和两档加速度差等特征，再用 StandardScaler 统一量纲。接着用 PCA（保留 95% 方差、51 维到 43 维）降维，用肘部法把 K 定在 9，并在同一子样本上对比了 KMeans、GMM 和层次聚类，最终选 KMeans 家族，在全量数据上改用 MiniBatchKMeans 训练。然后我们对标签按首次出现顺序重编号，生成提交文件，得到排行榜准确率约 0.27（K=9；另一份笔记本在 K=8 拿到 0.379）。

这个比赛给我们最重要的启发是：**无监督聚类没有「唯一正确答案」**。内部指标（轮廓、Davies–Bouldin）衡量的是几何形状好不好，外部指标（排行榜 accuracy）衡量的是分组对不对，两者经常矛盾，选模型要以能验证的那个为准。而「明知 K=11 分数更高却克制不提交」的做法，提醒我们：排行榜上的调参泡沫，会把验证集的作用彻底打穿。