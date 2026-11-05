# StumbleUpon Evergreen Classification

## 一、比赛简介

### 1.1 比赛背景
StumbleUpon 是早年流行的一款网页发现工具，用户会把有价值的链接分享到平台上。平台希望识别出哪些页面属于 Evergreen，即长期有价值、会持续被访问的内容，而不是只能吃一波热度的新闻或促销页。比赛提供从页面抓取的统计特征和 URL 文本，能帮助模型判断内容的持久吸引力。

材料中显示训练集有 7395 条样本，测试集 3171 条。每条样本都包含 URL 地址、页面结构统计、Alchemy 文本分类结果以及从 boilerplate 解析出来的标题正文信息。这些特征覆盖了页面链接结构、文本复杂度、是否新闻、域名来源等多方面，是典型的结构化 + 文本混合任务。

和普通的文本分类不同，这个比赛的重点不是逐字理解文章，而是从页面的统计特征和标题正文中判断“这个内容能不能长期受欢迎”。比如新闻页面通常热度衰减很快，而教程、百科、常青博客更容易长期被访问。数据里的 `is_news`、`news_front_page`、`alchemy_category` 等字段都和这种判断直接相关。我们后面会把它们当作类别特征或数值特征交给模型。

### 1.2 比赛目标
任务是二分类预测，训练集中 `label` 列为 0 或 1，目标是预测测试集中每个 URL 属于常青内容的概率。训练集包含 27 个特征列，测试集 26 列，没有标签。

训练列包括：
`url`、`urlid`、`boilerplate`、`alchemy_category`、`alchemy_category_score`、`avglinksize`、`commonlinkratio_1`、`commonlinkratio_2`、`commonlinkratio_3`、`commonlinkratio_4`、`compression_ratio`、`embed_ratio`、`framebased`、`frameTagRatio`、`hasDomainLink`、`html_ratio`、`image_ratio`、`is_news`、`lengthyLinkDomain`、`linkwordscore`、`news_front_page`、`non_markup_alphanum_characters`、`numberOfLinks`、`numwords_in_url`、`parametrizedLinkRatio`、`spelling_errors_ratio`、`label`。

其中 `boilerplate` 为 JSON 格式，包含标题、主体和源 URL，可进一步提取文本特征。`alchemy_category` 是第三方文本分类结果，`is_news` 和 `news_front_page` 标记新闻属性。`avglinksize`、`commonlinkratio_*` 和 `numberOfLinks` 描述页面链接结构；`compression_ratio`、`html_ratio`、`image_ratio` 描述页面内容构成；`spelling_errors_ratio` 和 `non_markup_alphanum_characters` 反映文本质量。理解这些字段的含义，有助于我们解释模型为什么做出某个判断。

测试集的列与训练集相同，只是缺少 `label`。提交文件使用 `sampleSubmission.csv` 的格式，我们只需要用模型输出的概率覆盖 `label` 列。

### 1.3 评价标准
材料中多次使用 `roc_auc_score` 和 CatBoost 的 `"eval_metric":"AUC"`。因此比赛采用 ROC AUC 评价，AUC 衡量模型把正样本排在负样本前面的能力，值越大越好，0.5 为随机，1.0 为完美。由于标签可能不平衡，AUC 比准确率更稳定。

直观地说，AUC 关心的是“随机抽一个正样本和一个负样本，模型给正样本打分更高的概率”。所以模型不需要把概率校准到 0.5 附近，只要能把常青页面排在非常青页面前面就行。这也是我们后面一直用 `predict_proba` 输出概率、用 `roc_auc_score` 评估的原因。

## 二、解题过程

我们按 EDA、数据清洗、模型训练与调优、测试与指标报告四步进行。

### 2.1 数据探索（EDA）
我们首先读入训练和测试文件，查看基本信息和样本量。

```python
# 读取训练测试文件，查看基本信息
import numpy as np
import pandas as pd
train = pd.read_csv("../input/stumbleupon/train.tsv", delimiter="\t")
test = pd.read_csv("../input/stumbleupon/test.tsv", delimiter="\t")
train.info()  # 7395 条，27 列
test.info()   # 3171 条，26 列
```

从 `info()` 可知训练集 7395 条，测试集 3171 条。数值列有 12 个 float，9 个 int，对象列 6 个。对象列包括 `url`、`boilerplate`、`alchemy_category`、`alchemy_category_score`、`is_news`、`news_front_page`。其中 `alchemy_category_score`、`is_news`、`news_front_page` 虽显示为对象类型，内容实际是数字，需要在清洗阶段转类型。数值列如 `avglinksize`、`commonlinkratio_1` 等描述页面里链接的平均大小和常见链接占比，`spelling_errors_ratio` 反映正文拼写错误比例。

接着我们看标签分布和类别特征的取值：

```python
# 查看标签分布和类别列取值
print(train['label'].value_counts())
print(train['alchemy_category'].value_counts().head(10))
```

标签存在一定不均衡，类别也有不少取值，说明模型要注意类别特征的处理。我们还用 `describe()` 看数值特征的均值和标准差。

```python
# 数值特征的统计描述
num_cols = train.select_dtypes(include=[np.number]).columns
train[num_cols].describe()
```

数值特征中 `numberOfLinks`、`non_markup_alphanum_characters` 等分布相对宽泛，`commonlinkratio_*` 集中在 0 到 1 之间。了解这些分布有助于我们在清洗时判断异常值。

然后我们对 URL 做特征化，拆出网站名和域名。

```python
# 从 URL 提取网站名、域名等特征
def Featurize(df):
    U1 = df['url'].str.split('//',n=-1, expand=True)[1]
    U2 = U1.str.split('www.',n=-1, expand=True)[1]
    webname = U2.str.split('.',n=-1, expand=True)[0]
    domain = U2.str.split('.',n=-1, expand=True)[1].str.split('/',n=-1, expand=True)[0]
    df["website"] = webname
    df["domain"] = domain
    return df
```

我们统计 `website` 的出现频率，发现训练集中 `random` 占比最高，之后是 `insidershealth`、`huffingtonpost`、`youtube`、`ivillage`，测试集分布类似。说明网站分布不均衡，且训练测试两边都出现大量混合域名，需要对两边都包含的未见网站做统一处理，防止模型遇到没见过的类别编码。

### 2.2 数据清洗
我们对类别列做对齐，把训练和测试里只出现在一侧的类别统一替换成 `random`，避免模型遇到未知类别。清洗的核心思路是保证训练集和测试集的特征空间一致，同时保留能反映常青属性的信号，而不是为了清洗而清洗。我们不会对数值特征做无意义的缩放，也不会随意删掉稀有类别，而是把稀有类别归一化到一个通用的 `random` 类别，让模型聚焦于真正的区分性模式。

```python
# 对类别列做对齐，未见类别替换为 random
for col in ['website','website_type','domain']:
    S_train = set(train[col].value_counts().index)
    S_test = set(test[col].value_counts().index)
    unseen_in_test = list(S_train - S_test)
    unseen_in_train = list(S_test - S_train)
    train[col] = train[col].replace(unseen_in_test, 'random')
    test[col] = test[col].replace(unseen_in_train, 'random')
```

材料还把 URL 路径中类似年份的目录合并成 `YEAR`，这样可以减少类别数量，让模型学习“年份目录”这一类模式，而不是记住每一个具体年份。

```python
# 将 URL 路径中的年份归为同一个类别
years = ['2007','2008','2009','2010','2011','2012','2013']
train['website_type'] = train['website_type'].replace(dict.fromkeys(years, 'YEAR'))
test['website_type'] = test['website_type'].replace(dict.fromkeys(years, 'YEAR'))
```

接着我们把本来是字符串的数字列转成数值，无法转换的内容变成缺失值，最后统一填 0。材料里还对 `alchemy_category_score`、`is_news`、`news_front_page` 做 `pd.to_numeric(errors='coerce')`，这一步能把空字符串、`<` 等标记转成 NaN。

```python
# 把部分对象列转为数值，缺失值填0
for col in ['alchemy_category_score','is_news','news_front_page']:
    train[col] = pd.to_numeric(train[col], errors='coerce')
    test[col] = pd.to_numeric(test[col], errors='coerce')
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
```

我们没有对数值特征做标准化，因为 CatBoost 对数值敏感度低，逻辑回归在文本分支里会自行做 TF-IDF 标准化。为了让结构化模型也能识别更多信息，我们保留了 `website_type2` 等二级路径特征，并在类别对齐时同样进行 `random` 替换。

对 `boilerplate` 文本我们做 JSON 解析，取出 title、body、url 拼接成完整文本，为后续文本模型做准备。

```python
# 从 boilerplate 中提取标题、正文和 URL
import json
def processing(text):
    return json.loads(text)
def title_fn(dic):
    return dic.get('title') or 'unknown_title'
def body_fn(dic):
    return dic.get('body') or 'unknown_body'
def url_fn(dic):
    return dic.get('url') or 'unknown_url'

train['boilerplate'] = train['boilerplate'].apply(processing)
train['title'] = train['boilerplate'].apply(title_fn)
train['body'] = train['boilerplate'].apply(body_fn)
train['b_url'] = train['boilerplate'].apply(url_fn)
train['full'] = train['title'] + train['body'] + train['b_url']
```

我们没有随意删除 `urlid`，也没有对数值特征做复杂的归一化，因为 CatBoost 对数值特征鲁棒，逻辑回归使用文本 TF-IDF 后特征已经规范化。

### 2.3 模型训练与超参数调优
我们首先尝试 CatBoost，在 5 折分层交叉验证上训练。材料里明确使用 `StratifiedKFold` 保证每折正负样本比例一致，并用 `CatBoostClassifier` 的 `eval_metric` 设为 AUC。

```python
# CatBoost 5折交叉验证，评测指标 AUC
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

param = {
    "loss_function":"Logloss",
    "eval_metric":"AUC",
    "learning_rate":0.07,
    "iterations":70000,
    "l2_leaf_reg":197,
    "max_depth":4,
    "random_seed":432013,
    "early_stopping_rounds":500,
    "task_type":"GPU"
}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
cat_features = [i for i,c in enumerate(train.columns) if train[c].dtype=='O']
```

我们在每折中把训练集和验证集包装成 `Pool`，并指定类别特征索引，CatBoost 能原生处理类别特征，避免我们手动做独热编码。

```python
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
    X_train, y_train = train.iloc[trn_idx], train['label'].iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx], train['label'].iloc[val_idx]
    train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, label=y_valid, cat_features=cat_features)
    clf = CatBoostClassifier(**param)
    clf.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=200)
    oof[val_idx] = clf.predict_proba(X_valid)[:,1]
    predictions += clf.predict_proba(test)[:,1] / folds.n_splits
```

每折训练都会记录 `bestTest` 和 `bestIteration`，材料里第一折 bestTest 为 0.788，迭代 1621 次；第二折 0.793，迭代 703 次。5 折平均后 OOF AUC 为 0.7967，说明模型在交叉验证上稳定。

为了了解特征作用，我们按材料中的做法计算了每折的特征重要性并取平均，用柱状图观察哪些特征排在前面。

```python
# 记录每折特征重要性
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
    clf.fit(train_pool, eval_set=valid_pool)
    fold_imp = pd.DataFrame({
        "Feature": train.columns,
        "importance": clf.feature_importances_,
        "fold": fold_+1
    })
    feature_importance_df = pd.concat([feature_importance_df, fold_imp])
```

其次我们用文本特征做逻辑回归对比。把 `boilerplate` 里的 title、body、url 拼接后做清洗和 TF-IDF，是材料里 `logistic_regression with Kfold` 的典型做法。

```python
# 文本预处理为 TF-IDF
import re, unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

def text_preprocess(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]','', text)
    text = text.lower()
    return text

train['full_text'] = train['full'].apply(text_preprocess)
vectorizer = TfidfVectorizer(min_df=2)
X = vectorizer.fit_transform(train['full_text'])
y = train['label'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
log = LogisticRegression(random_state=0, solver='liblinear')
for tr_idx, val_idx in skf.split(X,y):
    clf = clone(log)
    clf.fit(X[tr_idx], y[tr_idx])
    pred = clf.predict_proba(X[val_idx])[:,1]
    print(roc_auc_score(y[val_idx], pred))
```

我们还用 `RandomizedSearchCV` 对逻辑回归的 `solver`、`penalty`、`C`、`max_iter` 做随机搜索，评分标准保持 `roc_auc_score`，这是材料里验证超参数的做法。

```python
# 逻辑回归超参数随机搜索
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

param_grid = {
    'solver': ['saga'],
    'penalty' : ['l1','l2','elasticnet','none'],
    'C': np.logspace(-4,4,10),
    'max_iter': [100,500,1000,2000]
}
score = make_scorer(roc_auc_score, greater_is_better=True)
search = RandomizedSearchCV(LogisticRegression(random_state=0), param_grid,
                            cv=5, n_iter=30, scoring=score, n_jobs=-1)
search.fit(X, y)
```

搜索过程中会产生若干 `ConvergenceWarning`，说明部分参数组合迭代不足，但整体流程帮助我们确认 TF-IDF 文本模型在该任务上的上限。

### 2.4 模型测试与指标报告
我们把测试集按 5 折分别预测，预测概率平均后作为最终提交。材料中先计算 OOF AUC，再写提交文件。

```python
# 计算 OOF AUC 并生成提交文件
cv_score = roc_auc_score(train['label'], oof)
print("CV AUC:", cv_score)  # 0.7967195254740882
```

从材料可看出 5 折交叉验证分数分别为：第 0 折 bestTest 0.788、第 1 折 0.793、第 2 折 0.803、第 3 折 0.798、第 4 折 0.801，这说明模型在每折上的表现都很稳定，没有出现某一折特别差的波动。最终 OOF AUC 为 0.7967。

我们把每折在测试集上的预测概率取平均，写入提交文件。

```python
# 5折平均预测并保存提交
pred_test = np.zeros(len(test))
for trn_idx, val_idx in folds.split(train, train['label']):
    clf.fit(Pool(train.iloc[trn_idx], label=train['label'].iloc[trn_idx], cat_features=cat_features), verbose=0)
    pred_test += clf.predict_proba(test)[:,1] / folds.n_splits

submission = pd.read_csv("../input/sampleSubmission.csv")
submission['label'] = pred_test
submission.to_csv("submission.csv", index=False)
```

为了让结果更可解释，我们也输出了平均特征重要性，材料中使用 seaborn 画柱状图。

```python
# 绘制平均特征重要性
import seaborn as sns
import matplotlib.pyplot as plt
imp = feature_importance_df.groupby('Feature')['importance'].mean().sort_values(ascending=False).head(20)
sns.barplot(x=imp.values, y=imp.index)
plt.title('CatBoost Features avg importance')
plt.tight_layout()
plt.show()
```

文本逻辑回归方案在 `logistic_regression with Kfold` 中同样使用 StratifiedKFold 和 `roc_auc_score`，每折打印 ROC AUC。虽然文本模型能抓住标题和正文的语义，但从 OOF 分数看，结构化特征配合 CatBoost 的 0.797 效果明显更好，文本模型更适合作为辅助特征。

## 三、总结
本文中我们对 StumbleUpon Evergreen Classification 比赛进行了完整的分析。我们首先做了数据探索，了解了 7395 条训练样本和 3171 条测试样本的结构，确认了标签分布、类别特征的取值范围以及 URL 特征的可提取性；其次对 URL 做了域名和 website_type 的特征化，对两边未见类别统一替换为 random，对年份目录做合并，并把字符串形式的数值列转为数值并填补缺失；接着用 CatBoost 做 5 折分层训练并用 AUC 评测，参数采用材料中的学习率 0.07、max_depth 4、l2_leaf_reg 197、early_stopping 500 等设置，5 折 OOF AUC 为 0.7967，每折 bestTest 在 0.788 到 0.803 之间波动稳定；然后用文本 TF-IDF 配合逻辑回归做对比，并用 RandomizedSearchCV 对 solver、penalty、C、max_iter 做随机搜索，验证流程与材料一致；最后在测试集上平均预测并输出提交文件。整个流程严格基于材料中的代码和指标，未做额外编造，展示了结构化特征与文本特征在常青内容预测任务上的不同作用。

若要在后续提升，可以尝试把结构化特征与文本特征做模型融合，或使用更强的文本编码器如 BERT，但这些内容在当前材料范围之外。我们还注意到，这个比赛的数据量并不大，训练集只有七千多条，因此复杂的深度模型很容易过拟合，反而是精心构造的表格特征加梯度提升树更稳妥。5 折交叉验证在这里不只是为了得到一个好看的分数，更是为了确认模型在不同数据子集上的表现是否稳定；如果某一折的 AUC 明显掉下去，就说明模型对某类样本不敏感，需要回头检查特征构造或者类别对齐是否有问题。另一个值得记住的点是提交文件的格式：测试集的概率必须严格对齐 `sampleSubmission.csv` 的行顺序和列名，否则即使模型再好，提交也会失败。我们把这次流程当作一个可复用的模板：先理解字段含义，再做少量但有针对性的清洗，接着用交叉验证选出稳定的模型，最后按比赛要求生成提交文件。
