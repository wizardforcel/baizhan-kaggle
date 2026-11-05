# Ghouls, Goblins, and Ghosts... Boo!

## 一、比赛简介

### 1.1 比赛背景

Ghouls, Goblins, and Ghosts… Boo! 是 Kaggle Halloween 主题的练习赛，给出一批神奇生物的测量特征，要求判断它属于 3 类中的哪一类：Ghost（幽灵）、Goblin（哥布林）或 Ghoul（食尸鬼）。这是一个典型的多分类小样本表格比赛，常被用作入门的基准。

数据集包含 5 个数值特征与 1 个颜色类别特征：`bone_length`（骨骼平均长度，0~1 归一化）、`rotting_flesh`（腐烂肉体百分比）、`hair_length`（毛发长度，0~1 归一化）、`has_soul`（灵魂占比百分比），以及 `color`（主色：white/black/clear/blue/green/blood）。目标列 `type` 就是三类名称。

### 1.2 比赛目标

训练集有 371 行，7 列（包含 id），测试集有 529 行，6 列（无 type）。需要为每条测试样本预测 `type`。

```python
train = pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip')
test  = pd.read_csv('/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip')
train.shape  # (371, 7)
test.shape   # (529, 6)
```

标签分布不均衡，三类样本数量不同，这在真实生物数据中很常见。

### 1.3 评价标准

排行榜使用分类准确率（Accuracy），即预测正确的样本比例。准确率越高越好，满分 1.0。材料里多篇内核都在以验证集的 `accuracy_score` 衡量效果，公开榜上有 LightGBM 0.74 左右、神经网络 0.748 左右的成绩，说明难度中等，特征区分度足够。

---

## 二、解题过程

我们按 EDA、清洗、建模、测试四步推进，全部使用材料中出现的常用库。

### 2.1 数据探索（EDA）

训练集没有任何缺失值：

```python
print(train.isnull().sum())
# bone_length 0, rotting_flesh 0, hair_length 0, has_soul 0, color 0
```

描述性统计显示数值特征均在 0~1 之间。对各特征做箱线图可看到明显的类别差异：

```python
import seaborn as sns
sns.boxplot(x='type', y='bone_length', data=train)
```

从材料总结来看：Ghoul 骨骼最长，Goblin 次之，Ghost 最短；Ghost 腐烂肉体百分比最高；Ghoul 毛发最长；Ghost 灵魂占比最低。`color` 与类别也有相关性，不同颜色在三类中出现频率不同。

### 2.2 数据清洗

因为 `color` 是名义型变量，需要编码。材料里采用 `get_dummies` 做独热编码，并把 `type` 用映射转为数字标签：

```python
train_data = pd.get_dummies(train_data, columns=["color"], prefix=["color"])
# 产生 color_black, color_blood, color_blue, color_clear, color_green, color_white

map_type = {"Ghoul":1, "Goblin":2, "Ghost":0}
train_data.loc[:, "type"] = train_data.type.map(map_type)
train_data = train_data.set_index('id')
```

这样数值特征与独热编码拼接后可直接喂给树模型，id 作为索引保留，避免泄露。

### 2.3 模型训练与超参数调优

我们首先做 60/40 划分做快速验证：

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)
```

基准 LightGBM 模型可以直接拟合：

```python
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
```

为提升效果，后续材料里用了调参版本：

```python
lgbm_tuned = LGBMClassifier(
    boosting_type='gbdt',
    colsample_bytree=0.6,
    learning_rate=0.01,
    max_depth=10,
    n_estimators=40,
    num_leaves=30,
    subsample_for_bin=20000
)
lgbm_tuned.fit(X_train, y_train)
y_test_pred = lgbm_tuned.predict(X_test)
score = round(accuracy_score(y_test, y_test_pred), 3)   # 0.718
```

验证集准确率约 0.718。材料中还有神经网络、随机森林、XGBoost、SVM 等基线，以及投票 / 堆叠融合，公开榜上 LightGBM + 神经网络融合可到 0.74 左右。

### 2.4 模型测试与指标报告

在全量训练后生成提交：

```python
lgbm_tuned.fit(X, y)                 # X, y 为全量训练
test_pred = lgbm_tuned.predict(test_X)
submission = pd.DataFrame({'id': test['id'], 'type': test_pred})
submission.to_csv('submission.csv', index=False)
```

我们也用混淆矩阵检查各类的误判情况：Goblin 是最难分的类别，材料里多篇内核专门分析“Why goblins classification is so weak”。最终的提交文件与样例格式一致，`id` 对应 `type`。

---

## 三、总结

本文中我们完成了 Ghouls, Goblins, and Ghosts… Boo! 的完整流程。我们首先用 EDA 摸清了 371 条训练样本、5 个数值+1 个颜色类别特征、无缺失，以及骨骼、腐烂、毛发、灵魂等特征在三类间的明显差异。其次对颜色作独热编码并映射标签，确保模型可直接使用。接着用 LightGBM 先做基准，再调参到 colsample_bytree 0.6、learning_rate 0.01 等配置，验证集准确率约 0.718。最后在全量数据上拟合并生成提交文件。

这个比赛提醒我们：小数据集上先做特征可视化找到区分性强的特征，再用轻量树模型做基准，再逐步加入编码、调参或集成。即使数据只有几百行，只要特征有意义，准确率也能逼近 0.75。
