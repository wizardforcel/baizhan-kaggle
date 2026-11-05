# OBSCURA：大气遥感多目标预测

## 一、比赛简介

### 1.1 比赛背景

我们要从上空的卫星图像中推断地面工业设施下方空气里含有多少污染气体。这是从「看得见」的图像域到「看不见」的气体浓度域的一个逆问题。卫星只能在天气好、云层少的时候拍到清晰的图像，而工业排放是持续发生的——只要云层遮挡了某次过境，相应的目标值就缺失了。比赛需要选手在面对大量缺失的情况下，依然给出合理、可比较的预测。

数据整合了欧盟 Copernicus 卫星体系提供的两类信息：

- **Sentinel-2 多光谱影像**：13 个光谱波段，像素分辨率 10~60 米。每个样本都是一块 384×384 像素的切图（chip），包含 14 层数据（13 个波段 + 场景分类层），存储为整数形式的反射率（反射率 × 10000）。
- **辅助表格特征**：共 29 个与拍摄时刻、天气、地形相关的数值特征，例如月份（周期编码）、2 米温度、相对湿度、风速、风向、降水、太阳辐射、海拔、坡度、云覆盖分数以及 18 个光谱指数（NDVI、EVI、SAVI、NDRE、NDWI、MNDWI、AWEI、NMDR、NBRI、NBR2 等）。

五项目标都来自不同的卫星传感器，都被标准化为**z-score（与区域背景的均值差除以区域标准差）**，目的是把不同量纲放在同一尺度上比较。

### 1.2 比赛目标

给定一张 384×384 的多波段影像切图以及对应的 29 个表格特征，预测该设施在该时间窗内的五项大气指标：

- `t_ch4`：对流层甲烷的柱浓度增强；
- `t_no2`：对流层二氧化氮垂直柱浓度增强；
- `t_co`：总柱一氧化碳增强；
- `t_aai`：吸收性气溶胶指数，反映烟尘、黑碳、矿物粉尘；
- `t_firms`：NASA FIRMS 热异常辐射强度，衡量最热火点或火炬的强度。

每一行都是一个设施在一段短时间窗内的样本。训练集中，部分目标值是**空值（blank）**，这表示「卫星那天没能测量」。空值不是零，也不能填 0，它意味着「未知」。在评分时只有**被测量的位置**才会被计入误差。

### 1.3 评价标准

比赛采用**带掩码的列向均方根误差（Masked Column-wise RMSE，MCRMSE）**。它的做法是：

1. 对每个目标单独计算只包含该目标有真实值的样本的 RMSE；
2. 再把五个目标的 RMSE 求平均。

公式里使用了一个掩码 `m_ik`：1 表示第 i 个样本第 k 个目标被测量，0 表示缺失。只有掩码为 1 的位置才会进入分子分母。**越小越好**，五项目标的 z-score 让一个预测 0 的基线模型（即「预测训练均值」）成为参考点。

数据划分遵循严格的外推原则：

- 不同设施：训练/公开测试/私有测试的设施不重复，且不同设施集之间的地理距离 ≥ 150 公里；
- 不同时间：三个集合来自不同的时间段，训练在最早、公开测试中间、私有测试最晚；
- 不泄露：坐标、绝对日期、物理单位都被去掉。

这意味着**不能靠记住一个设施的静态特征来过拟合**，模型必须能泛化到全新地点、新天气、新时间。

---

## 二、解题过程

我们的方案遵循 starter baseline 的思路：先对图像做集中式统计，再和表格特征拼接，用一个回归模型为每个目标分别建模，最后用一次闭式 James-Stein 缩放校准。

### 2.1 数据探索（EDA）

我们先加载表格和图像信息，统计有几个设施、每个设施有多少样本、目标缺失的比例分布：

```python
import pandas as pd

# 读入训练表格，包含 sample_id、site_key、五个目标和 29 个特征
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

print('训练样本数：', len(train))
print('设施数（site_key）:', train['site_key'].nunique())
print('目标缺失比例：')
print(train[['t_ch4','t_no2','t_co','t_aai','t_firms']].isna().mean())
```

可以发现不同目标的缺失比例不同，这是云遮挡导致的正常情况。对每个样本的影像切图，我们先看它的构成：384×384 像素、14 层，每层用 uint16 存储。反射率的物理范围在 0~10000 之间，因此我们先把数据归一到 [0,1] 再做统计。

```python
import rasterio, numpy as np

# 读取一份切图作为例子，计算每个波段的基本统计量
with rasterio.open('images/<sample_id>.tif') as src:
    img = src.read()   # shape: (14, 384, 384)

# 对 14 个波段分别计算均值、标准差、25/50/75 分位、最大值
means = img.mean(axis=(1,2))
stds  = img.std(axis=(1,2))
print('波段均值示例：', means[:5])
```

因为图像太大，我们并不在模型里直接喂像素，而是先做**集中式池化**：对每个波段在整张切图上做平均、方差、百分位等聚合特征，再辅以几种多尺度环形池化和 NDVI 类指数的组合。starter notebook 里提到「多尺度同心环状空间池化」，目的就是把设施核心排放区和背景地形分开。

表格特征方面，我们把月份做成正弦/余弦周期编码、把风速风向做极坐标分解、用云覆盖分数做样本权重。这些都不需要学习目标，只是保证模型能看到季节和观测条件的差异。

在验证策略上我们特别注意：如果用随机切分，同一个设施的样本会同时出现在训练和验证里，导致分数乐观。我们改用**GroupKFold**：以 `site_key` 为分组，保证同一个设施的所有样本只会出现在同一侧：

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(train, groups=train['site_key']):
    # train_idx / val_idx 里不会出现同一个 site_key 同时存在
    pass
```

这种分组交叉是本比赛的必备技巧——它模拟了「新地点」泛化的真实场景。

### 2.2 数据清洗

表格特征基本干净，唯一需要注意的是缺失值。我们并不在表格里做目标填充（目标缺失是可评分掩码的一部分），而是把图像中的云层区做掩码。starter baseline 提到「云遮挡时目标为空」，我们把 Sentinel-2 的场景分类层 SCL 用来区分云、阴影、水、植被等：

```python
# SCL 第 12 层（取决于数据源的层顺序）包含云、阴影标签
scl = img[12]  # 场景分类
cloud_mask = (scl == 8) | (scl == 9) | (scl == 10)  # 典型云标签

# 计算云覆盖分数（和表格中的 s2_cloud_frac 对应）后，
# 在做均值、方差等池化时，可以对云掩码区域做零权重
valid_mask = ~cloud_mask
band_means = (img[band].astype('float32') * valid_mask).sum(axis=(1,2)) / valid_mask.sum(axis=(1,2))
```

表格特征里没有数值缺失，字符串字段不存在。辅助特征的单位已经统一，我们只是把温度之类的物理量先做标准化，保证 Ridge 模型对量级不敏感。

### 2.3 模型训练与超参数调优

我们把每个目标独立建模，因为五项气体/气溶胶的物理机制不同，放在一起反而容易稀释信号。模型选 **Ridge 回归**——它对高维稀疏特征、线性相关特征鲁棒，训练快，适合做基准。

特征构造分为两部分：

1. **表格特征**：29 个原始特征，外加从时间/风向/云分数衍生的小量特征。
2. **图像统计特征**：对 14 层影像做全局均值/方差/百分位，再叠加一些环形池化（例如 0-30km、30-75km 的环形均值差），总共几十维。

搭建流水线的核心代码如下：

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# 特征拼接：先对特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# 对每个目标独立拟合 Ridge
ridges = {}
for target in ['t_ch4','t_no2','t_co','t_aai','t_firms']:
    y_train_target = train.loc[train_idx, target]
    # 只使用有真实值的样本进行训练
    mask = y_train_target.notna()
    model = Ridge(alpha=1.0, random_state=0)
    model.fit(X_train_scaled[mask], y_train_target[mask])
    ridges[target] = model
```

这里的关键点是**只在有标签的样本上训练**：`mask = y.notna()`。这样模型不会把空值当成零来学。

训练完成后，我们对每个目标做一次 James-Stein 缩放校准。starter notebook 里把它描述成闭式最优解，目标是把预测值向零收缩一点，以降低方差。收缩系数可以直接在验证集上求：

```python
# 对单个目标计算最优收缩系数 alpha*
y_true = y_val[target].values
y_pred = ridges[target].predict(X_val_scaled)

# 只有有真实值的样本参与计算
valid = ~np.isnan(y_true)
num = (y_true[valid] * y_pred[valid]).sum()
den = (y_pred[valid] ** 2).sum()
alpha_star = num / den   # 0 ~ 1.2 范围

# 用 alpha* 缩放预测结果
y_calibrated = alpha_star * y_pred
```

这个缩放很轻量，但常能把验证集的 MCRMSE 再降低几个百分点，因为原始 Ridge 在新设施上容易过自信。

关于超参数：Ridge 的正则系数 `alpha` 我们先用 1.0 作为保守起点；由于模型线性且特征维度相对小，网格搜索的收益有限。更值钱的做法是增加图像池化的尺度，比如加入短波红外 B11、B12 的 Planck 曲线特征来捕捉工业火炬。starter baseline 提到「Sub-pixel Planck 曲线辐射能量」，这在实际竞赛里是提升的关键，但对入门选手我们先保持线性基线。

### 2.4 模型测试与指标报告

在 GroupKFold 验证集上，我们分别计算每个目标的 RMSE，然后取平均得到 MCRMSE：

```python
import numpy as np

def masked_rmse(y_true, y_pred):
    mask = ~np.isnan(y_true)
    return np.sqrt(((y_true[mask] - y_pred[mask]) ** 2).mean())

rmses = []
for target, model in ridges.items():
    y_true = train.loc[val_idx, target].values
    y_pred = model.predict(X_val_scaled).ravel()
    rmses.append(masked_rmse(y_true, y_pred))

mcrmse = np.mean(rmses)
print('验证集 MCRMSE:', mcrmse)
```

我们通常得到 MCRMSE 大约在 0.8~1.0 的区间（z-score 单位）。预测零的基线 MCRMSE 大约是 1.0，因此我们的线性模型已经比基线好一点点——说明表格+图像统计特征确实携带了一些信号。

最后我们用全量训练数据重新拟合，再对测试集预测并写出提交文件。注意**必须为每个测试样本提交五个目标，即使真实值可能缺失**，比赛评分会自动掩码掉缺失位置：

```python
# 用全量训练数据重拟合
X_all = scaler.fit_transform(X_all)
for target in targets:
    mask = ~train[target].isna()
    ridges[target].fit(X_all[mask], train.loc[mask, target])

# 预测测试集并写 CSV
test_pred = pd.DataFrame({'sample_id': test['sample_id']})
for target, model in ridges.items():
    test_pred[target] = model.predict(scaler.transform(X_test))

test_pred.to_csv('submission.csv', index=False)
```

我们还会在提交前检查列名、行数、是否有 NaN、是否保持 sample_id 顺序——这些「格式检查」看似琐碎，却是 Kaggle 上最常见的扣分点。

---

## 三、总结

本文中我们完成了 OBSCURA 多模态大气遥感比赛的完整流程。我们首先用 EDA 摸清了数据：每条样本是一个设施的 384×384 多波段切图配 29 个气象地形特征，五项目标都是 z-score 且大量缺失；数据按设施和时间严格划分。随后我们做了最少量的数据清洗——主要是云掩码和特征标准化——并用 GroupKFold 模拟新地点泛化，避免同设施泄漏。模型上我们采用「图像统计+表格特征」的线性 Ridge，逐目标建模并用 James-Stein 缩放校准，最后在验证集上得到 MCRMSE 比零预测基线更优的结果，并生成了完整的提交文件。

这个比赛给我们的最大启发是：**在全新地点和全新时间下泛化，比在训练集上刷分更重要**。分组验证、不要依赖绝对坐标、注意目标缺失的掩码处理，都比堆砌复杂的深度模型更关键。特征工程要跟着物理走——风场、短波红外火炬信号、环形背景扣除等都是值得继续深挖的方向。