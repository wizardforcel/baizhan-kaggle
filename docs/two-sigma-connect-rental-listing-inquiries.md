# Two Sigma Connect：纽约租房挂牌关注度预测

## 一、比赛简介

### 1.1 比赛背景

这个比赛叫 Two Sigma Connect，数据来自纽约的租房平台 RentHop。RentHop 上每天会挂出大量公寓房源，租房者看到感兴趣的房子会点进去、打电话或者发邮件询问，这些动作就是“关注”。对中介来说，如果能在房源刚挂出来时就知道它会不会受欢迎，就可以把推广资源集中到更有潜力的房子上，而不是平均用力。

比赛把每套房源整理成一条记录，里面既有结构化的数字，比如几间卧室、几个卫生间、租金多少、经纬度在哪里；也有非结构化的内容，比如一段英文描述、一串设施标签、一组照片链接，以及挂牌时间、楼栋编号、中介编号等。我们要根据这些信息，判断这套房源最终会落在“低关注”“中关注”“高关注”哪一档。

这个任务的难点主要有三个。第一，三档关注度的样本非常不平衡，低关注占了将近七成，模型很容易变成“多数类预测器”。第二，很多字段是高基数类别，比如楼栋编号、中介编号、街道地址，取值成千上万，直接做独热编码会让特征维度爆炸。第三，描述和设施是自由文本，写法不统一，需要先做清理和归并，才能变成模型能吃的数字。

### 1.2 比赛目标

训练数据是 `train.json`，一共 49352 行、15 列；测试数据是 `test.json`，一共 74659 行、14 列。测试集比训练集还大，这一点在比赛中比较少见。两边的字段基本一致，只是测试集没有 `interest_level` 这一列，因为那正是我们要预测的东西。

具体字段可以分成四组：

- 房屋属性：`bathrooms`（卫生间数）、`bedrooms`（卧室数）、`price`（月租金）；
- 位置信息：`latitude`、`longitude`、`street_address`、`display_address`；
- 文本与列表：`description`（房源描述）、`features`（设施标签列表）、`photos`（照片链接列表）；
- 身份与时间：`listing_id`（房源编号）、`building_id`（楼栋编号）、`manager_id`（中介编号）、`created`（挂牌时间）。

训练集多出来的 `interest_level` 就是标签，取值只有 `low`、`medium`、`high` 三种。我们要给测试集的每一行输出这三个类别的概率，而不是只给一个类别名字。

### 1.3 评价标准

比赛使用**多分类对数损失（Multi-class Log Loss）**，有时候也简写成 log loss。它的意思是：模型对每条样本都会给出三个概率，我们只看模型给“真实类别”打的那个概率，然后取对数、加负号、再对所有样本求平均：

$$
\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}
$$

其中 $N$ 是样本数，$p_{i,y_i}$ 是模型认为第 $i$ 条样本属于它真实类别的概率。这个指标**越低越好**。如果模型很自信地给真实类别打了很高的概率，$\log p$ 接近 0，损失就小；如果模型把真实类别的概率打得很低，$\log p$ 会变成一个很大的负数，加负号后损失就很大。换句话说，log loss 不只看“有没有猜对”，还看“猜得有多自信”，瞎猜而且很自信的模型会被重罚。

正因为如此，我们不能只盯着准确率。训练集里低关注度约占 69.47%，中关注度约占 22.75%，高关注度只占 7.78%。一个把所有房源都猜成 `low` 的模型，准确率也能接近 70%，但它的 log loss 会很难看。后面的评估我们要同时看准确率、F1 和混淆矩阵，才能判断模型到底学到了什么。

---

## 二、解题过程

整场比赛我们走的是“先理解数据，再把文本和列表变成数字，最后用梯度提升树做分类”的路线。我们按数据探索、数据清洗、模型训练与调参、模型测试与指标报告四步来做。

### 2.1 数据探索（EDA）

我们首先把训练集和测试集读进来，确认数据规模和标签分布：

```python
import pandas as pd
import numpy as np

# 两个文件都是 JSON 格式，一行对应一套房源
train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')

# 先看规模：训练集 49352 行，测试集 74659 行
print(train_df.shape, test_df.shape)

# 把三档关注度转成 0/1/2，后面的模型只认数字
train_df['target'] = train_df['interest_level'].map(
    {'low': 0, 'medium': 1, 'high': 2}
)

# 看三档各占多少比例，判断类别是否平衡
print(train_df['interest_level'].value_counts(normalize=True))
```

打印结果告诉我们，训练集是 49352 行、15 列，测试集是 74659 行、14 列。三档关注度的比例大约是低 69.47%、中 22.75%、高 7.78%，差距非常明显。这个分布直接决定了我们后面不能只用准确率评估，因为“全部猜低”就能拿到接近 70% 的准确率。

其次我们看照片和描述这两个“非结构化”字段。照片本身是链接列表，我们暂时不读图片内容，只数每套房源有多少张照片；描述则先看它是不是空的：

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 照片数量差异很大，先截断到 10 张，避免少数极端值把图拉变形
train_df['nPhotos'] = train_df['photos'].apply(lambda x: min(10, len(x)))

# 小提琴图：横轴是关注度，纵轴是照片数量
sns.violinplot(x='interest_level', y='nPhotos', data=train_df,
               order=['low', 'medium', 'high'])
plt.show()

# 描述是否为空：没有描述的房源，通常更少人愿意进一步询问
train_df['hasDesc'] = train_df['description'].apply(
    lambda x: len(x.strip()) != 0
)
sns.countplot(x='hasDesc', hue='interest_level', data=train_df,
              hue_order=['low', 'medium', 'high'])
plt.show()
```

从图里能看出两个很直观的规律：照片越少，房源落在低关注度的比例越高；没有描述的房源，也明显更容易无人问津。这两条规律和我们的生活经验一致——照片和文字是租房者最先看到的信息，信息越少，越难让人产生兴趣。

接着我们统计设施标签，并检查经纬度和租金的极端值。`features` 是一个字符串列表，我们先把它摊平，数一数每种设施出现了多少次：

```python
from collections import Counter

# features 是列表，先摊平再统计，才能知道哪些设施最常见
all_features = [f for row in train_df['features'] for f in row]
feature_count = Counter(all_features)
print(feature_count.most_common(10))

# 经纬度两端各去掉 1% 的极端点，避免地图被离群坐标拉偏
lon_low, lon_high = np.percentile(train_df['longitude'], [1, 99])
lat_low, lat_high = np.percentile(train_df['latitude'], [1, 99])
train_geo = train_df[
    train_df['longitude'].between(lon_low, lon_high) &
    train_df['latitude'].between(lat_low, lat_high)
]

# 价格也在 99% 分位数处截断，长尾租金会干扰统计和模型
price_cap = np.percentile(train_df['price'], 99)
train_df['price'] = train_df['price'].clip(upper=price_cap)

# 把房源画到地图上，颜色代表关注度
sns.scatterplot(x='longitude', y='latitude', hue='interest_level',
                data=train_geo, hue_order=['low', 'medium', 'high'], s=10)
plt.show()
```

设施去重后大约有 1760 种写法，最常见的几项是 `elevator`（39560 次）、`cats allowed`（35654 次）、`hardwood floors`（35597 次）、`dogs allowed`（33172 次）和 `doorman`（31538 次）。这些高频设施后面会被我们做成 0/1 特征。地图则显示，不同区域的关注度分布并不一样，曼哈顿核心区和外围区域的房源密度、颜色都有差异，说明经纬度是有用的。

最后我们看了租金。把价格画成分布图后可以看到明显的长尾，少数房源租金特别高；按关注度分组看，极端高价的房源反而更容易落在低关注度。于是我们在统计和建模前把价格截断到 99% 分位数，既保留大部分信息，又不让个别极端值主导模型。

### 2.2 数据清洗

这份数据的缺失值很少，`info()` 显示主要字段都是非空的，所以我们不需要做缺失值填充。真正要处理的是文本里的噪音和类别字段的编码。

我们首先清理描述。原始描述里混着网页换行留下的 `br /`、`br`、`pa` 之类的碎片，这些不是自然语言，会干扰分词和词频统计，所以先把它们删掉。设施标签则统一转成小写、去掉首尾空格，避免 `Doorman` 和 `doorman` 被当成两个不同的东西：

```python
# 描述里混着网页换行留下的碎片，先删掉再进入下一步
def clean_description(text):
    for junk in ['br /', 'br', 'pa']:
        text = text.replace(junk, '')
    return text

train_df['description'] = train_df['description'].apply(clean_description)
test_df['description'] = test_df['description'].apply(clean_description)

# 设施标签大小写不统一，统一转小写并去掉首尾空格
train_df['features'] = train_df['features'].apply(
    lambda xs: [x.strip().lower() for x in xs]
)
test_df['features'] = test_df['features'].apply(
    lambda xs: [x.strip().lower() for x in xs]
)
```

其次我们处理高基数类别。`display_address`、`manager_id`、`building_id`、`street_address` 这四列都是字符串，取值可能上万，直接做独热编码会让特征表变得极其稀疏。我们改用标签编码，把每个不同的字符串映射成一个整数。这里有一个细节：编码器要在训练集和测试集的并集上拟合，这样测试集里出现、训练集里没出现过的地址也能得到一个编号，不会在预测时报错：

```python
from sklearn.preprocessing import LabelEncoder

# 这四列是字符串类别，取值很多，树模型不能直接处理字符串
categorical = ['display_address', 'manager_id',
               'building_id', 'street_address']

for col in categorical:
    encoder = LabelEncoder()
    # 训练集和测试集一起 fit，保证测试集的新类别也有编号
    all_values = list(train_df[col]) + list(test_df[col])
    encoder.fit(all_values)
    train_df[col] = encoder.transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])
```

需要说明的是，标签编码会给类别人为制造“大小关系”，比如编号 100 的楼栋并不一定比编号 10 的楼栋更“大”。这对线性模型很不友好，但树模型只关心“这个编号是否等于某个值”，所以影响相对小一些。我们在材料里看到的主线方案也是这么做的。更严谨的做法是目标编码或者按频率编码，但那会引入额外的泄漏风险，入门阶段先用标签编码更容易跑通。

清洗完文本和类别后，原始的 `description`、`features`、`photos`、`created` 这几列就不再直接喂给模型了。它们的信息已经被我们抽取成数字特征，原始列保留下来只会增加噪音和维度，所以在建模前会统一丢掉。

### 2.3 模型训练与超参数调优

我们接着做特征工程。除了原始的卧室数、卫生间数、租金、经纬度，我们还构造了几类新特征：房间总数、照片数量、设施数量、描述词数、描述里是否包含邮箱、是否包含十位电话号码，以及若干高频设施的 0/1 标记。这样做的原因是，原始字段的表达能力有限，而“房间总数”“有没有留下联系方式”这些组合信息更能反映房源的吸引力：

```python
import re
import string

# 判断描述里有没有十位电话号码：先去掉标点，再按空格切词
def has_phone(text):
    cleaned = re.sub('[' + string.punctuation + ']', '', text)
    return int(any(tok.isdigit() and len(tok) == 10
                   for tok in cleaned.split()))

for df in (train_df, test_df):
    # 房间总数比单独看卧室、卫生间更直观
    df['rooms'] = df['bedrooms'] + df['bathrooms']
    df['num_photos'] = df['photos'].apply(len)
    df['num_features'] = df['features'].apply(len)
    df['num_description_words'] = df['description'].apply(
        lambda x: len(x.split())
    )
    # 描述里留了邮箱或电话，说明中介更急着促成交易
    df['has_email'] = df['description'].apply(
        lambda x: int(bool(re.findall(r'[\w\.-]+@[\w\.-]+', x)))
    )
    df['has_phone'] = df['description'].apply(has_phone)

# 把出现最多的设施做成 0/1 特征
facilities = ['elevator', 'cats allowed', 'hardwood floors',
              'dogs allowed', 'doorman', 'dishwasher',
              'no fee', 'laundry in building', 'fitness center']
for name in facilities:
    for df in (train_df, test_df):
        df[name] = df['features'].apply(lambda xs: int(name in xs))
```

特征准备好后，我们把数据切成训练集和验证集。因为三档比例不平衡，切分时要用 `stratify=y`，保证验证集里的低、中、高三档比例和原数据一致。我们先跑一个不调参的基线模型：1000 棵 XGBoost 树，学习率 0.05，最大深度 10。然后做五折交叉验证，看看模型在不同数据子集上的稳定程度：

```python
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBClassifier

# 去掉标签列和已经抽取完信息的原始文本列
drop_cols = ['interest_level', 'target', 'created',
             'description', 'features', 'photos']
X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
y = train_df['target']

# 七成训练、三成验证；stratify 保证三档比例一致
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=5, stratify=y
)

# 五折交叉验证，每一折都重新训练一个基线模型
kf = KFold(n_splits=5, shuffle=False)
X_train_arr = X_train.values
y_train_arr = y_train.values
scores = []
for tr_idx, va_idx in kf.split(X_train_arr):
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=10,
        random_state=0,
    )
    model.fit(X_train_arr[tr_idx], y_train_arr[tr_idx])
    scores.append(model.score(X_train_arr[va_idx], y_train_arr[va_idx]))

print(scores)
print(np.mean(scores))
```

这一次五折的准确率分别是 0.7479、0.7416、0.7445、0.7347、0.7344，平均约 0.7406。这个数字比“全部猜低”的 0.6947 高，说明模型确实学到了东西，但提升不算大，原因是中、高关注度样本太少，模型很容易偏向多数类。

接着我们用 Optuna 做超参数搜索。Optuna 是一个自动调参工具，它会根据前面试过的参数组合，挑更有可能变好的方向继续试。我们让它搜索五类参数： booster 类型、学习率、最大深度、行采样比例和列采样比例。为了让每次比较公平，所有 trial 都用同一份训练集和验证集：

```python
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score

# 固定一份验证集，调参时反复用同一份，结果才可比
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'booster': trial.suggest_categorical(
            'booster', ['gbtree', 'dart', 'gblinear']
        ),
        'learning_rate': trial.suggest_loguniform(
            'learning_rate', 0.01, 0.1
        ),
        'max_depth': trial.suggest_int('max_depth', 3, 11),
        'subsample': trial.suggest_uniform('subsample', 0.0, 1.0),
        'colsample_bytree': trial.suggest_uniform(
            'colsample_bytree', 0.0, 1.0
        ),
    }
    bst = xgb.train(params, dtrain)
    proba = bst.predict(dvalid)
    pred = proba.argmax(axis=1)
    return f1_score(y_valid, pred, average='micro')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, timeout=600)
print(study.best_params, study.best_value)
```

我们一共跑了 100 轮，时间上限设为 600 秒。搜索过程中验证集上的 micro-F1 最高大约 0.7207，对应的参数是 `booster=gbtree`、`learning_rate` 约 0.0757、`max_depth=11`、`subsample` 约 0.8998、`colsample_bytree` 约 0.6251。可以看到，表现好的 trial 几乎都是 `gbtree`，深度也偏向较大的 11；`dart` 偶尔能接近，但不够稳定；`gblinear` 基本垫底，说明这个任务里的特征交互比较强，线性模型不够用。

除了调参，我们还用 LOFO（Leave One Feature Out）检查了特征的重要性。它的做法是：每次把一列特征删掉，重新训练模型，看验证分数掉多少；掉得越多，说明这列越重要。相比 XGBoost 自带的“分裂次数”重要性，LOFO 更接近“这列特征对最终预测到底有多大贡献”：

```python
from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.metrics import make_scorer, mean_absolute_error

# 四折交叉验证，用平均绝对误差作为评分
cv = KFold(n_splits=4, shuffle=True, random_state=0)
scorer = make_scorer(mean_absolute_error, greater_is_better=False)
features = [col for col in train_df.columns if col != 'target']
dataset = Dataset(df=train_df, target='target', features=features)

lofo_imp = LOFOImportance(dataset, scoring=scorer,
                          model=model, cv=cv)
importance_df = lofo_imp.get_importance()
plot_importance(importance_df)
```

这一步让我们确认，房源的价格、经纬度、房间数、照片数、设施数量这些特征都在起作用，而单纯依赖某一个编号字段并不可靠。它也提醒我们：高基数类别虽然方便，但很容易让模型记住训练集里的具体编号，换到测试集就不灵了。

### 2.4 模型测试与指标报告

调参结束后，我们用最优参数重新训练，并在验证集上算准确率和 micro-F1。这里要特别注意材料里的一个细节：作者最后是用**全部训练数据**重新拟合，再在之前切出来的验证集上评估。因为验证集已经包含在全部训练数据里，这样得到的 0.9057 准确率和 0.9057 的 micro-F1 是偏高的，不能当成真实的泛化能力。更可信的参考是前面五折交叉验证的平均准确率 0.7406，以及 Optuna 搜索时记录到的验证 micro-F1 约 0.7207。

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 用最优参数在全部训练数据上重新拟合
final_model = XGBClassifier(**study.best_params)
final_model.fit(X, y)

# 在验证集上评估；注意 X_valid 已经包含在 X 里，这个分数会偏高
pred = final_model.predict(X_valid)
print('accuracy:', accuracy_score(y_valid, pred))
print('micro f1:', f1_score(y_valid, pred, average='micro'))

# 再看训练集上的混淆矩阵，观察三类之间往哪里混
train_pred = final_model.predict(X)
cm = confusion_matrix(y, train_pred)
print(cm)
```

训练集上的混淆矩阵是：

```
[[33726   543    15]
 [ 2983  8188    58]
 [  471   430  2938]]
```

行是真实类别（依次为低、中、高），列是预测类别。我们可以这样读：

- 低关注度一共 34284 条，其中 33726 条被正确预测，召回率约 98.4%，只有极少数被错判成中或高；
- 中关注度一共 11229 条，其中 8188 条被正确预测，召回率约 72.9%，有 2983 条被错判成低；
- 高关注度一共 3839 条，其中 2938 条被正确预测，召回率约 76.5%，有 471 条被错判成低、430 条被错判成中。

这个矩阵非常典型：多数类 `low` 学得很稳，少数类 `medium` 和 `high` 则经常被模型往 `low` 那边推。原因还是样本不平衡——模型见到的低关注样本太多，稍微拿不准就倾向于猜低。要提高 log loss，重点不是继续把 `low` 猜得更准，而是想办法让模型对中、高关注度给出更合理的概率。

最后我们生成提交文件。比赛要的是三个类别的概率，不是硬标签，所以要用 `predict_proba`。我们按提交样例的列顺序，把高、中、低三个概率分别填进去：

```python
# 对测试集输出三类概率；列顺序要和 sample_submission 一致
proba = final_model.predict_proba(test_df[X.columns])

submission = pd.DataFrame({
    'listing_id': test_df['listing_id'],
    'high': proba[:, 2],    # 类别 2 对应 high
    'medium': proba[:, 1],  # 类别 1 对应 medium
    'low': proba[:, 0],     # 类别 0 对应 low
})
submission.to_csv('submission.csv', index=False)

# 检查行数和列名，避免格式问题白白丢分
print(submission.shape)
print(submission.head())
```

生成后我们会核对两件事：行数是不是 74659，和测试集一致；列名是不是 `listing_id`、`high`、`medium`、`low`。这两步看起来琐碎，但比赛中因为提交格式错误拿不到分数的例子并不少。

---

## 三、总结

本文中我们完成了 Two Sigma Connect 纽约租房挂牌关注度预测的完整流程。我们首先用 EDA 摸清了数据：训练集 49352 行、测试集 74659 行，三档关注度约为低 69.47%、中 22.75%、高 7.78%，照片少、描述空的房源明显更容易冷门，经纬度和租金也携带信息。其次我们做了针对性的数据清洗：删掉描述里的网页碎片，统一设施标签的大小写，把高基数地址和编号做标签编码，并把价格截断到 99% 分位数。接着我们构造了房间总数、照片数、设施数、描述词数、是否留邮箱或电话、高频设施 0/1 标记等特征，用 XGBoost 跑五折交叉验证，平均准确率约 0.7406。然后我们用 Optuna 搜索了 100 组参数，得到的最优验证 micro-F1 约 0.7207，最优配置以 `gbtree`、较深树和较高的行采样率为特征。最后我们看了混淆矩阵，发现低关注度召回率约 98.4%，而中、高关注度只有约 72.9% 和 76.5%，少数类被大量误判成低关注度，这也是 log loss 难以进一步下降的主要原因。

这个比赛给我们最大的启发是：**评估指标决定了优化方向**。如果只看准确率，把所有房源猜成低关注度就能拿到接近 70% 的分数，看起来不差；但 log loss 会惩罚这种“只会押多数类”的策略。另一个教训是**验证集不能被训练集污染**。材料里最后用全量数据重新拟合、再在旧验证集上评估，得到的 0.9057 明显偏高，只有交叉验证和独立验证集上的分数才可信。以后我们再遇到类别不平衡的表格比赛，会优先考虑分层抽样、类别权重、概率校准，以及针对少数类的特征，而不是一味追求准确率。
