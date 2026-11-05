# Ghouls, Goblins, and Ghosts... Boo!

## 一、比赛简介

### 1.1 比赛背景

Ghouls, Goblins, and Ghosts… Boo! 是 Kaggle 在万圣节期间推出的经典练习赛。它的设定带点童话色彩：我们手上有一批“怪物”的体检数据，需要判断这只怪物到底是幽灵（Ghost）、哥布林（Goblin）还是食尸鬼（Ghoul）。数据集很小，但特征区分度不错，长期被当作表格分类的入门基准，也是很多人第一次接触多分类、特征编码和模型比较的题目。

数据里包含 5 个特征：`bone_length` 表示骨骼平均长度（0~1 归一化）、`rotting_flesh` 表示腐烂肉体的百分比、`hair_length` 表示毛发平均长度（0~1 归一化）、`has_soul` 表示灵魂所占百分比，以及 `color` 表示怪物的主色。颜色有 white、black、clear、blue、green、blood 六种，是一个典型的名义型变量。目标列 `type` 就是三类别名称。

### 1.2 比赛目标

训练集 371 行、7 列（含 `id`），测试集 529 行、6 列（无 `type`）。我们要为测试集每一行预测 `type`。

```python
# 读取训练集与测试集
train = pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip')
test  = pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip')

print(train.shape)   # (371, 7)
print(test.shape)    # (529, 6)
print(train.columns.tolist())
# ['id','bone_length','rotting_flesh','hair_length','has_soul','color','type']
```

训练样本只有几百条，模型很容易过拟合，因此不能一上来就堆复杂网络；先把数据看透、把特征处理好，往往比换模型更有效。

### 1.3 评价标准

比赛使用多分类准确率（Categorization Accuracy）作为指标，即预测正确的测试样本比例，越高越好，满分 1.0。这个指标很直观：它不关心概率大小，只看最终类别是否一致。因为三个类别样本量不完全均衡，我们在本地也会额外观察混淆矩阵，避免模型靠“全猜多数类”刷到虚高的准确率。材料里多篇内核以验证集 `accuracy_score` 作为模型选择依据，公开榜上 LightGBM 与神经网络方案大致在 0.74~0.75 区间。

---

## 二、解题过程

我们的主线是：先做 EDA 找特征规律，再做颜色编码和标签映射，接着用 LightGBM 建立基准并调参，最后在全量数据上训练、生成提交。

### 2.1 数据探索（EDA）

首先检查缺失值和重复值。材料显示训练集各列都没有缺失：

```python
print(train.isnull().sum())
# id 0, bone_length 0, rotting_flesh 0, hair_length 0, has_soul 0, color 0, type 0
```

接着看数值特征的分布和类别差异。最直观的方式是对每个特征按 `type` 画箱线图：

```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(),
                   ['bone_length','rotting_flesh','hair_length','has_soul']):
    sns.boxplot(x='type', y=col, data=train, ax=ax)
plt.tight_layout(); plt.show()
```

此外，训练集与测试集的 `id` 是交错出现的：训练集前几行是 0、1、2、4、5，测试集前几行是 3、6、9、10、13。这说明主办方是按 id 随机切分的，而不是简单地把前一段给训练、后一段给测试。这个细节有两层含义：一是训练集和测试集的特征分布应当比较接近，本地验证有参考价值；二是 `id` 本身可能隐含采集顺序，但它是编号而非测量值，不能当作特征喂给模型，否则模型可能记住编号与类别的偶然对应关系。

接着对数值特征做描述性统计：

```python
print(train.describe())
```

四个数值特征都落在 0~1 附近，量纲一致，因此不需要额外做标准化；这对树模型没有影响，对神经网络则可以先做 StandardScaler。我们还会检查每个特征的偏度和异常值，确认没有明显离群点把箱线图拉得很长。

从箱线图可以读出几条规律：Ghoul 的骨骼最长，Goblin 次之，Ghost 最短；Ghost 的腐烂肉体比例整体高于另外两类；Ghoul 的毛发最长；Ghost 的灵魂占比最低。这些差异说明数值特征确实携带分类信息，但单看某一个特征无法完全分开三类，需要模型组合多个特征。

再单独看 `color` 这个名义特征，统计每个颜色的样本数，确认类别是否严重失衡，以及是否有些颜色几乎只在某几类出现。材料后续对颜色做了独热编码，说明它是可以被模型利用的信息，但没有任何单一规则能直接代替模型。我们把颜色作为一种离散信号保留，等待模型去组合它与其他数值特征。

最后看类别分布。三类样本数并不相等，Ghoul 偏多、Goblin 相对少一些。这个不均衡会带来一个现象：模型倾向于把边界样本判成多数类，导致 Goblin 的召回率偏低。材料里有多篇内核专门讨论“为什么 Goblin 这么难分”，原因就在于它的特征分布与其他两类有重叠，且样本量较少。

### 2.2 数据清洗

这份数据没有缺失值，清洗的重点是编码。

第一，`color` 是名义变量，不能直接转成 0~5 的整数，因为整数会引入“大小关系”的假信息。材料用 `get_dummies` 做独热编码：

```python
# 为每个颜色生成一列 0/1 指示变量，并删掉原 color 列
train_data = pd.get_dummies(train_data, columns=["color"], prefix=["color"])
# 生成 color_black, color_blood, color_blue, color_clear, color_green, color_white
```

第二，目标列 `type` 需要转成数字。材料给出了两种方式：`LabelEncoder` 和字典映射。作者更偏好映射，因为结果可控、可读：

```python
# 方式一：LabelEncoder（注意要先 astype(str)）
# le = LabelEncoder()
# train_data["type"] = le.fit_transform(train_data["type"].astype(str))

# 方式二：手工映射，顺序明确，便于复现
map_type = {"Ghoul": 1, "Goblin": 2, "Ghost": 0}
train_data.loc[:, "type"] = train_data.type.map(map_type)
```

第三，把 `id` 设为索引而不是特征，避免模型把编号当成规律：

```python
train_data = train_data.set_index('id')
```

测试集也必须做完全相同的编码，并保证列顺序与训练集一致。`get_dummies` 在训练集和测试集上分别调用时，如果某个颜色只在一边出现，列数会不一致，所以更稳妥的做法是先拼接再编码，或按训练集的列对齐测试集。

### 2.3 模型训练与超参数调优

我们首先用默认参数的 LightGBM 建立基准，确认流程能跑通：

```python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
```

然后做训练/验证划分。材料用 60/40 的比例：

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)
```

调参后的版本是：

```python
lgbm_tuned = LGBMClassifier(
    boosting_type='gbdt',
    colsample_bytree=0.6,     # 每棵树随机用 60% 特征，降低方差
    learning_rate=0.01,       # 小学习率配合较多树，泛化更稳
    max_depth=10,
    n_estimators=40,
    num_leaves=30,
    subsample_for_bin=20000
)
lgbm_tuned.fit(X_train, y_train)
y_test_pred = lgbm_tuned.predict(X_test)
score = round(accuracy_score(y_test, y_test_pred), 3)
print(score)   # 0.718
```

验证准确率约 **0.718**。这几个参数的含义在材料里也有解释：`num_leaves` 控制树模型复杂度，叶子数越多模型越能拟合细节；`max_depth` 限制树的深度，防止过深导致过拟合；`learning_rate` 越小通常越稳，需要配合 `n_estimators` 增大；`colsample_bytree` 做特征随机子抽样，相当于每棵树看到不同特征，有助于降低方差。`subsample_for_bin` 控制直方图 bin 的最小样本数，对几百行的小数据设置较大值，可以稳定统计。

调参时我们重点关注三个参数：`num_leaves` 控制树的复杂度，叶子数越多模型越能拟合细节；`max_depth` 限制树的深度，防止过深导致过拟合；`learning_rate` 越小通常越稳，需要配合 `n_estimators` 增大；`colsample_bytree` 和 `subsample` 做行列采样，相当于给模型加正则。由于数据量小，交叉验证的折数不宜太多，否则每折样本太少、分数波动大。

调参时我们重点关注三个参数：`num_leaves` 控制树的复杂度，太大容易记住训练集噪声；`min_data_in_leaf` 控制叶节点最少样本数，几百行数据里应设得相对大一些；`colsample_bytree` 和 `subsample` 做行列采样，相当于给模型加正则。由于数据量小，交叉验证的折数不宜太多，否则每折样本太少、分数波动大。

### 2.4 模型测试与指标报告

模型确定后，用全部训练数据重新拟合一次，再预测测试集：

```python
lgbm_tuned.fit(X, y)                       # X、y 为全量数据
test_pred = lgbm_tuned.predict(test_X)     # 测试集已完成相同编码

submission = pd.DataFrame({'id': test['id'], 'type': test_pred})
submission.to_csv('submission.csv', index=False)
```

我们还会检查提交文件的类别分布是否与训练集接近，以及行数是否与测试集一致。除此之外，用混淆矩阵看每一类的召回率：Goblin 往往是最容易被误判的一类，如果发现它的召回率明显偏低，可以通过类别权重（`class_weight='balanced'`）或对少数类做重采样来缓解。材料里很多内核把精力放在“Goblin 为什么弱”上，本质上就是在做错误分析。

另一个值得注意的点是本地验证分数与排行榜分数的关系。因为训练集只有 371 行，40% 的验证集只有一百多条，单次划分的准确率波动可能有几个百分点。为了避免“调参调到某一次划分的噪声上”，我们在确定最终配置前会用多次不同的 `random_state` 重复划分，观察分数的均值与方差；如果某组参数只是在某一次划分上特别高，就不采信它。最终提交时再用全量训练数据重新拟合，这样能多利用几十条样本的信息。

---

## 三、总结

本文中我们完成了 Ghouls, Goblins, and Ghosts… Boo! 的完整流程。我们首先用 EDA 确认训练集 371 行、测试集 529 行、没有缺失值，并通过箱线图发现骨骼长度、腐烂比例、毛发长度和灵魂占比在三类之间有系统性差异。其次对颜色做独热编码、对标签做字典映射，并把 id 设为索引避免泄露。接着用默认 LightGBM 建立基准，再调参到 `colsample_bytree=0.6`、`learning_rate=0.01`、`num_leaves=30` 等配置，验证集准确率约 0.718。最后在全量数据上拟合，生成符合样例格式的提交文件。

这个比赛虽然小，但把表格分类的关键环节都走了一遍：类别特征怎么编码、小样本怎么防过拟合、类别不均衡怎么做错误分析。它给我们的启发是：**数据量越少，越要把力气花在理解和清洗特征上，而不是盲目换更复杂的模型**；同时本地验证要做得足够细，整体准确率会掩盖掉某一类的系统性失败。

最后需要强调的是可复现性。材料里的很多内核都会在开头固定随机种子，并在结尾打印最终的验证分数与提交文件的 hash。只要我们把 `random_state`、`列顺序`、`get_dummies` 的处理方式固定下来，任何人按照相同的写法都能得到同样的结果，这正是 Kaggle 写作者应该坚持的工程习惯。
