# mytestkmeans

## 一、比赛简介

### 1.1 比赛背景
mytestkmeans 是一个很小的示例比赛，用来让参赛者跑通「读数据、训练模型、写提交文件」的完整流程。它的数据集是一张二维平面上的散点图：每个点有两个坐标 `x` 和 `y`，并且被贴上一个类别标签 `class`。比赛名字里的 kmeans 指的是 K-Means（K 均值聚类），这是一种非常经典的**无监督学习**算法。

所谓无监督学习，就是数据只给特征、不给答案，算法自己去找结构。K-Means 做的事情可以这样理解：假设你手里有一把撒在桌子上的豆子，你想把它们分成 K 堆，就随便选 K 个位置当「堆中心」，然后不断重复两步：一、每个豆子归到离它最近的中心；二、把每个中心挪到它所管的那堆豆子的平均位置。重复到中心不再移动，就分好了。这个比赛的数据很可能就是这样人工或程序生成出来的：先用 K-Means 把二维点分成若干簇，再把簇号当作类别标签。

因为数据是二维的，我们可以在纸上直接画出分界线，所以这个比赛特别适合作为入门练习。也正因为数据结构这么「肉眼可见」，我们可以很容易地判断模型有没有真的学到东西：把预测结果画回散点图，看看颜色分块和原来的团状结构是否对得上即可。这种「可视化验证」的习惯在更复杂的比赛里同样有用，只是到时候要换成看损失曲线、看混淆矩阵、看特征重要性。

### 1.2 比赛目标
比赛提供两个文件：

- `kmeans.training.csv`：训练集，列有 `x`、`y`、`class`，每一行是一个点的坐标和它所属的类别；
- `kmeans.testing.csv`：测试集，只有 `x`、`y` 两列，需要我们预测 `class`。

我们要训练一个分类模型，给测试集里的每个点预测类别，然后按比赛要求生成 `submission.csv`。提交文件需要两列：`index`（测试样本的序号，从 0 开始）和 `class`（预测出来的类别编号）。

这里要注意一个细节：训练集里的类别编号本身没有大小含义，`2` 不比 `1` 大，它只是一个名字。所以这是一个**多分类**问题，而不是回归问题。

### 1.3 评价标准
比赛使用分类准确率（Accuracy）作为评价指标，也就是「预测对的数量 ÷ 总数量」。比如 100 个点猜对 95 个，准确率就是 0.95。准确率越高越好，满分是 1.0。因为类别标签只是编号，不存在「差一点点算对」的情况，猜错就是猜错，所以这是一个很直白的指标。

需要提醒的是，如果各类别样本数量很不均衡，准确率会有点误导：假如 90% 的点都属于同一类，那么无脑全猜这一类也能有 0.9 的准确率。所以做 EDA 的时候要顺便看一眼类别分布。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**把二维点当作特征，用一个决策树学出类别之间的分界线，再对测试集逐点预测。** 下面按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）
我们首先把训练数据读进来，看看它长什么样。
```python
import pandas as pd
import numpy as np

# 读取训练集，文件里每行是一个二维点及其类别
train = pd.read_csv('/kaggle/input/mytestkmeans/kmeans.training.csv')

# 看前五行，确认列名和数值范围
print(train.head())

# 看数据规模和每列类型
print(train.shape)
print(train.dtypes)
```
运行后会看到三列：`x`、`y` 是浮点数，`class` 是整数。数据量不大，可以一次性读进内存。

接下来我们统计每个类别有多少个点：
```python
# 每个类别各有多少样本，检查是否均衡
print(train['class'].value_counts())
```
如果各个类别的数量差不多，说明数据是均衡的；如果差很多，后面看准确率时就要多留个心眼。

我们还想直观地看到点的分布，于是画一张散点图：
```python
import matplotlib.pyplot as plt

# 用颜色区分类别，画出二维散点图
plt.figure(figsize=(6, 6))
for c in sorted(train['class'].unique()):
    subset = train[train['class'] == c]
    plt.scatter(subset['x'], subset['y'], s=10, label=f'class {c}')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Training points')
plt.show()
```
图上通常能看到一坨一坨的「团」，每一团对应一个类别，团与团之间有比较明显的空隙。这正是 K-Means 生成数据的特征——因为它本来就是按「离中心最近」来分簇的，所以簇的形状往往接近圆形或椭圆形。看到这里我们就有了信心：只要模型能学出这些团的边界，分类应该不难。

我们还可以看一眼 `x`、`y` 的取值范围：
```python
# 描述统计：最小、最大、均值、标准差
print(train[['x', 'y']].describe())
```
这能帮我们判断是否需要做标准化。对于决策树这类基于「阈值切分」的模型，特征的尺度其实不影响结果，因为它只关心「比某个值大还是小」；但如果换成 KNN、SVM 或神经网络，标准化就很重要了。

### 2.2 数据清洗
这份数据非常干净：没有缺失值，没有明显的重复行，也没有时间或文本字段需要解析。我们只需要做最基本的检查和处理。

第一，检查缺失值：
```python
# 统计每列的缺失值数量
print(train.isnull().sum())
```
如果输出全是 0，说明不需要填补。

第二，把特征和标签分开。我们不在这一步做标准化，原因是决策树对特征尺度不敏感，而且保留原始坐标便于后面画图对照：
```python
# 特征矩阵 X：两列坐标；标签向量 y：类别
X = train[['x', 'y']].values
y = train['class'].values

print(X.shape, y.shape)   # 应为 (n, 2) 和 (n,)
```
这里用 `.values` 把 DataFrame 转成 NumPy 数组，是因为 scikit-learn 的模型接口更习惯接收数组。

第三，检查测试集的列是否与训练集一致：
```python
test = pd.read_csv('/kaggle/input/mytestkmeans/kmeans.testing.csv')
print(test.columns)       # 应该只有 x、y 两列
print(test.shape)
```
如果测试集多出或少了一列，就要及时调整，否则后面预测会报错。

总体而言，这个比赛几乎没有「清洗」工作量，重点在于理解数据和选对模型。

### 2.3 模型训练与超参数调优
我们选择**决策树分类器**（Decision Tree Classifier）作为主模型。决策树的工作方式很像玩「二十个问题」：它每次挑一个特征和一个阈值，把数据切成两半，比如「x 是否大于 1.2」；然后在每一半里继续切，直到每个叶子节点里的点基本都属于同一类。预测时，一个新的点从树根往下走，最后落到哪个叶子，就判成那个叶子里的多数类。

决策树的优点是：训练快、可解释、不需要标准化，而且能画出很直观的分界线。缺点是单棵树容易「死记硬背」训练数据（过拟合），所以后面我们还会讨论怎么控制。

我们先训练一棵默认参数的树：
```python
from sklearn.tree import DecisionTreeClassifier

# 默认参数的决策树：不限制深度，节点一直分裂到纯为止
dt = DecisionTreeClassifier()
dt.fit(X, y)
```
`fit` 就是让树从数据里学习切分规则。

为了判断模型好不好，我们不能只看训练集，而要留一部分数据做验证。这里用**分层抽样**（stratify），保证训练和验证两部分里各类别的比例一致：
```python
from sklearn.model_selection import train_test_split

# 80% 训练、20% 验证；stratify=y 保持类别比例
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

dt.fit(X_tr, y_tr)
val_pred = dt.predict(X_val)
print("Validation accuracy:", (val_pred == y_val).mean())
```
固定 `random_state` 是为了让每次划分结果一样，方便复现。

接下来做一点超参数调优。决策树最重要的参数是 `max_depth`（最大深度）和 `min_samples_leaf`（叶子最少样本数），它们都用来防止过拟合。深度越大、叶子样本越少，树越复杂，越容易把噪声也学进去。
```python
from sklearn.model_selection import GridSearchCV

# 在几个候选深度和叶子大小上做网格搜索
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, 8, None],
    'min_samples_leaf': [1, 3, 5, 10]
}
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,               # 5 折交叉验证
    scoring='accuracy'
)
grid.fit(X, y)

# 打印最优参数和对应的交叉验证准确率
print(grid.best_params_)
print(grid.best_score_)
```
由于数据是二维的、簇之间分得比较开，通常浅一点的树（比如深度 3～5）就能达到很高的准确率，再深反而可能只带来微小的提升甚至过拟合。这也从侧面说明：**模型复杂度要和数据难度匹配**，不是越复杂越好。

### 2.4 模型测试与指标报告
确定模型后，我们在验证集上做更完整的评估。
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

best_dt = grid.best_estimator_
val_pred = best_dt.predict(X_val)

# 整体准确率（比赛指标）
print("Accuracy:", accuracy_score(y_val, val_pred))

# 每一类的精确率、召回率、F1
print(classification_report(y_val, val_pred))

# 混淆矩阵：看哪两类容易混
print(confusion_matrix(y_val, val_pred))
```
混淆矩阵的行是真实类别、列是预测类别，对角线上的数字就是预测对的个数。如果某两类经常互相混淆，说明它们在空间上挨得太近，这时候可以考虑加特征（比如到各个簇中心的距离）或者换更强的模型。

我们对测试集做预测并生成提交文件：
```python
# 对测试集逐点预测
y_test = best_dt.predict(test[['x', 'y']].values)

# 按要求组装提交文件：index 从 0 开始，class 是预测类别
submission = pd.DataFrame({
    'index': np.arange(len(y_test)),
    'class': y_test
})
submission.to_csv('submission.csv', index=False)
```
最后我们还会检查一下提交文件的行数是否等于测试集行数、列名是否为 `index` 和 `class`，避免因为格式问题白丢分。

材料里还演示了把预测结果画回散点图的做法，方便肉眼检查分界是否合理：
```python
# 把测试点的预测类别画出来，和训练点对照
plt.figure(figsize=(6, 6))
plt.scatter(test['x'], test['y'], c=y_test, s=10, cmap='viridis')
plt.title('Predicted test classes')
plt.show()
```
如果图上的颜色分区和训练时的团状结构吻合，就说明模型确实学到了结构，而不是乱猜。

---

## 三、总结
本文中我们完成了 mytestkmeans 示例比赛的完整流程。我们首先用 EDA 确认数据是二维点集、类别由 K-Means 式的簇结构生成，并检查了类别分布和坐标范围；其次做了最轻量的数据清洗，只检查缺失值并把特征与标签分开；接着用决策树分类器学习分界线，并通过网格搜索在树深度和叶子样本数上做交叉验证调参；最后在验证集上报告准确率、分类报告和混淆矩阵，并把测试集的预测写成 `submission.csv`。

这个比赛虽然简单，但它把机器学习的几个基本环节都串了一遍：**看懂数据、留出验证集、防止过拟合、按比赛格式提交**。对入门者来说，最大的收获不是分数，而是建立起「先探索、再建模、最后验证」的习惯。
