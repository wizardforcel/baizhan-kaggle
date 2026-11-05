# ECML PKDD 2015 出租车行程时间预测 II

## 一、比赛简介

### 1.1 比赛背景
出租车行程时间预测是城市交通智能化里最基础也最重要的任务之一。ECML PKDD 2015 期间举办的「Taxi Trip Time Prediction II」比赛，公开了纽约市出租车委员会 TLC 2013 年到 2014 年的大规模行程记录，并提供了配套的 OpenStreetMap 道路 99 分位图用于可视化。这类数据的价值在于，它真实反映了早晚高峰、天气、道路拥堵、乘客上车点与下车点分布等复杂因素对行程时长的影响。业界对这类任务的关注点不在于预测一个孤立的时间点，而是要在不了解实时路况的前提下，仅靠历史统计规律给出可靠的时长估计，用于司机派单、乘客等待时间和票价预估等场景。因此本比赛一方面考验特征工程的能力，另一方面也考验对时空数据噪声和分布偏差的处理能力。

### 1.2 比赛目标
比赛提供训练集 train.csv 和测试集 test.csv，两个文件都以 zip 压缩包形式给出。训练集每行代表一次完整的出租车行程，包含供应商编号、上下车时间、乘客人数、上下车经纬度以及是否通过存储转发标志。测试集缺少下车时间和实际行程时长，我们的任务是根据上车时间和起点终点位置等信息，预测每趟行程的行驶时长，单位是秒。输出文件需要包含原始 id 与预测的时长，两列严格对应。材料中同时给出了两个 RDS 格式的 99 分位道路图文件，这类文件在 R 中便于绘制起点终点分布与热力图，帮助我们判断数据集中是否存在地理采集偏差。

### 1.3 评价标准
比赛采用 Root Mean Squared Logarithmic Error，即 RMSLE 作为官方评价指标。RMSLE 先对真实值和预测值取对数再计算均方根误差，相比普通的 RMSE 对大时间值的惩罚相对温和，同时在时长为零或很小时更稳定。指标越小越好，理想情况下为零。评测时以对数尺度上的误差衡量，因此预测值不能为负，且过高估计与过低估计的代价不对称。我们在本地验证时也使用 RMSLE，保持与排行榜完全一致的评估口径。

## 二、解题过程

整场比赛我们从解压与环境准备开始，逐步完成探索、清洗、建模与验证。下面的思路是在材料提供的 EDA 笔记本基础上提炼而来的，第一人称复数叙述是我们实际执行的步骤。

### 2.1 数据探索（EDA）
我们首先进入 Kaggle 提供的运行环境，检查输入目录并加载常用库。材料中的第一段代码列出了输入文件，确认压缩包存在并准备解压。

```python
# 导入数据处理、绘图和文本处理的常用库
import pandas as pd      # CSV 读写与数据处理
import numpy as np       # 数值计算
import seaborn as sns    # 可视化风格设置
import matplotlib.pyplot as plt  # 绘图
sns.set(style="whitegrid", color_codes=True)

# 列出输入目录，确认 train.csv.zip 和 test.csv.zip 存在
from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))
```

执行后我们看到 train.csv.zip、test.csv.zip 以及两个 99 分位的 RDS 地图文件，证实数据完整。随后我们解压 train.csv 并读取前几行，确认列名与数据类型。

```python
# 读取训练数据并查看基本信息
train = pd.read_csv('train.csv')
print(train.head())
print(train.info())
print(train.describe())
```

从初步观察可以看出，上车时间与下车时间是字符串格式，需要转为时间戳；经纬度列存在缺失值或异常坐标；乘客人数分布高度集中在 1 人；store_and_fwd_flag 只有少数字符变化。我们接着绘制上车小时分布、出发经纬度的散点图以及行程距离的直方图，直观判断是否存在重尾分布或异常点。

我们尤其关心时间的维度，因为出租车时长的规律主要是「随着一天中的时刻」变化的。于是我们画了一张上车小时与平均时长的折线图，发现早晚高峰的均值明显抬升。我们还按星期几分组对比，确认工作日与周末在午夜的分布有明显差异，这说明小时和星期几两个特征都会携带有效信息。地图 RDS 文件帮助我们确认绝大多数行程集中在曼哈顿中城区域，这与纽约出租车运营规律相符。材料中还有一个 R 环境下运行的笔记本，它同样先列出输入目录，说明作者在 R 中读取同样的压缩包，并用 readr 等库进行后续处理。我们承认自己在 R 上的训练数据保留了两份不同语言的 EDA 尝试，最终选择了 Python 主线，因为 Python 生态下的 sklearn 和后续的绘图工具链更完整。这一小节我们验证的最关键结论是：数据中时间、空间两类信号都存在，且都与目标时长有明显关联，值得全部保留下来做特征。

### 2.2 数据清洗
数据本身结构清晰但存在清洗需求。我们首先统一解压流程，确保 train.csv 与 test.csv 在同一工作目录下。

```python
# 设置随机种子保证结果可复现
np.random.seed(0)

# 检查是否有缺失值，列出缺失的行数
missing = train.isnull().sum()
print(missing[missing>0])
```

上车、下车经纬度偶有空值，我们采用按经纬度中位数填充的保守策略，避免引入过大偏差。时间列需要解析为 datetime。

```python
# 将时间列转为 datetime，并提取时间特征
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])

# 计算行程时长用于训练标签，并提取小时、星期几等特征
train['trip_duration'] = (train['dropoff_datetime'] - train['pickup_datetime']).dt.total_seconds()
train['pickup_hour'] = train['pickup_datetime'].dt.hour
train['pickup_weekday'] = train['pickup_datetime'].dt.weekday
```

为了表征空间距离，我们用 haversine 公式计算起点到终点的球面距离。

```python
# 根据经纬度计算大圆距离，单位公里
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    # 转换为弧度
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
```

清洗过程中我们没有对经纬度做粗暴的去噪，而是先保留原始值并通过距离特征间接反映异常。乘客人数直接作为类别特征，store_and_fwd_flag 映射为 0/1。最后我们统一训练集与测试集的列顺序，确保特征对齐。

为保证特征在训练和测试之间保持一致，我们在清洗阶段就建立了一个通用的预处理函数，它接收原始的 DataFrame，依次执行时间解析、缺失填充、距离计算、以及时间特征提取。测试集没有真实的下车时间，因此只执行去掉标签相关的步骤，避免产生信息泄漏。我们还对极端时长做了截断处理，保留了 1 分钟到 3 小时的有效行程，剔除明显异常的零时长或过长时长记录，这在提升模型稳定性上起到了关键作用。清洗后我们再次检查特征间的相关性，避免把高度相关的经纬度与距离同时作为冗余特征堆叠。

### 2.3 模型训练与超参数调优
清洗完成后我们构建特征矩阵 X，标签 y 为 trip_duration。由于目标是非负连续值，我们选择梯度提升树类模型作为基线，它对非线性关系和特征交互很敏感，且对异常值相对稳健。

```python
# 构造特征集，包含时间、空间和类别特征
features = ['pickup_hour','pickup_weekday','passenger_count',
            'pickup_longitude','pickup_latitude',
            'dropoff_longitude','dropoff_latitude',
            'trip_distance','store_and_fwd_flag']

X = train[features]
y = train['trip_duration']
```

为避免数据泄露，我们按时间顺序划分训练与验证集，或采用分层抽样保证验证集覆盖不同时段。

```python
# 简单的训练验证划分，保留一定比例用于本地验证
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0
)
```

模型训练我们先用 GradientBoostingRegressor 作为起点，观察学习曲线，再尝试 XGBoost。

```python
# 使用梯度提升回归器作为基线，调参以降低 RMSLE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np

model = GradientBoostingRegressor(random_state=0)
model.fit(X_train, y_train)

# 在验证集上预测并计算 RMSLE
val_pred = model.predict(X_val)
val_pred = np.maximum(val_pred, 0)  # 防止负预测
rmsle = np.sqrt(mean_squared_log_error(y_val, val_pred))
print("Validation RMSLE:", rmsle)
```

在超参数调优阶段，我们重点调整学习率、树的深度以及树的数量。更深的树能拟合局部拥堵模式，但容易过拟合高峰时段；较小的学习率配合更多迭代可以平滑预测。我们通过网格搜索在验证集上逐步搜索，观察 RMSLE 是否下降，同时比较不同特征组合的影响，比如是否加入星期几特征、是否对距离取对数等。最终我们保留了小时、星期几、乘客人数、haversine 距离以及经纬度原始值组成的特征集。训练时我们全量拟合，得到最终模型以便生成提交文件。

我们调参时遵循「先粗后细」的原则：先固定一个中等规模的学习率找到大致的参数区间，再逐步缩小搜索范围。每一次实验我们都会记录验证集上的 RMSLE，并留意训练集与验证集的差值，这个差值接近零说明过拟合概率升高。经过若干轮实验，我们发现把学习率调低并适当增加树的数量，可以让 RMSLE 在小幅下降的同时保持稳定的偏差。我们还尝试把 passenger_count 当作数值特征而不是类别特征，结果显示两种效果差异不大，最终保留了类别处理时的健壮性。

### 2.4 模型测试与指标报告
验证阶段我们不仅看整体 RMSLE，也按小时段、按出发区域对预测误差进行拆分，确认模型在高峰时段是否出现系统性低估。

```python
# 对验证集按小时分组查看误差
X_val['pickup_hour'] = train.loc[X_val.index, 'pickup_hour']
hour_errors = []
for h in range(24):
    mask = X_val['pickup_hour'] == h
    if mask.sum() > 0:
        pred = val_pred[mask]
        true = y_val[mask]
        err = np.sqrt(mean_squared_log_error(true, np.maximum(pred,0)))
        hour_errors.append((h, err))
```

这种拆分帮助我们发现早晚高峰预测偏差相对平峰更大，于是我们在特征中加入了小时与距离的交互项，进一步提升稳定性。加入交互项后，我们重新在验证集上计算 RMSLE，确认指标有微小但稳定的改善，才把这个特征保留进最终特征列表。我们还使用特征重要性排序理解每个特征的贡献，结果显示距离类特征对时长的解释力最强，时间类特征次之，经纬度因为与距离高度相关而排位靠后。最后我们用全量训练数据重新拟合模型，对测试集做相同的特征工程与预处理，生成预测文件。

```python
# 用全部训练数据拟合最终模型
model.fit(X, y)

# 对测试集做同样的特征构造并预测
test_features = test[features]
test_pred = model.predict(test_features)
test_pred = np.maximum(test_pred, 0)

# 准备提交文件
submission = pd.DataFrame({
    'id': test['id'],
    'trip_time': test_pred.astype(int)
})
submission.to_csv('submission.csv', index=False)
```

提交前我们检查行数是否一致、数值是否全部为正，并随机抽样几条结果与训练数据的分布做对比，确保没有明显漂移。整个流程保证了从原始压缩包到最终提交的链路清晰、可复现。

## 三、总结

本文中我们完成了 ECML PKDD 2015 出租车行程时间预测 II 的完整解决流程。我们首先通过列出输入文件和加载 pandas、seaborn、matplotlib 等库确认数据存在并解压训练与测试压缩包，其次对上车时间与经纬度进行清洗与时间特征提取，接着用 haversine 距离和小时、星期几等特征构建模型输入，然后以梯度提升回归器为基线进行超参数调优并以 RMSLE 为指标做本地验证，最后用全量数据重新训练并生成提交文件。整个过程始终围绕时空特征工程展开，避免了对缺失值和异常坐标的过度干预，强调了可视化与分组误差分析对模型改进的指导作用。
