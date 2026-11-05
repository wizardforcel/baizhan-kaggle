# 亿客行个性化排序（Expedia Personalized Sort）

## 一、比赛简介

### 1.1 比赛背景

还是 Expedia 的数据，但这回不是「推荐哪座酒店」，而是**给一场搜索结果排序**。当用户在 Expedia 上搜索酒店时，系统会返回一堆候选酒店，这份训练数据记录了每次搜索（`srch_id`）返回的每个候选（`prop_id`）以及当时的各种报价信息——房型星级、评论分、价格、与 8 个竞争对手的比价字段等等，还标注了两个结果：用户有没有**点击**（`click_bool`）、有没有**预订**（`booking_bool`）。

问题看起来是「点不点」的分类问题，但本质是**排序问题**：同一个搜索结果列表里，排在越前面的酒店越可能被用户看到，我们要学的是「什么样的候选应该排到前面」。

原始数据非常大，而且解压后是专有格式，`pd.read_csv` 直接读全量会内存溢出。材料用 `nrows=30123456` 这种「超过文件行数就当读全量」的写法确认：训练集 `train.csv` 共 **9,917,530 行 × 54 列**，测试集 `test.csv` 为 **6,622,629 行 × 50 列**。

### 1.2 比赛目标

输入一次搜索的所有候选酒店及其报价、比价字段，输出每个候选酒店的**相关性强弱排序**，让「用户真正点击/预订的酒店」尽量排在列表前部。相关性我们用一个分数折算：预订（booking）算满分 1、仅点击（click）算 0.2、完全没互动算 0：

```python
def get_target(row):
    """把 booking/click 两个布尔列折算成一个相关性分数"""
    if row.booking_bool > 0:
        return 1        # 预订了：最相关
    if row.click_bool > 0:
        return 0.2      # 只点了没订：次相关
    return 0            # 完全无互动
```

### 1.3 评价标准

比赛用 **NDCG（Normalized Discounted Cumulative Gain，归一化折损累积增益）** 系列指标评估一个 `srch_id` 内的排序质量。直觉是：真正的成交项排得越靠前、且没有成交的垃圾项越少排前，分数越高；排行位置越靠后，贡献被折损得越厉害。`train` 和 `eval` 两组预测相互独立计分，最终按 NDCG 汇总，越高越好。另外可以确认的是，提供方把 `click_bool`、`booking_bool`、`gross_bookings_usd`（成交金额）这三列标为**泄漏列**——评估时不存在，建模必须剔除。

---

## 二、解题过程

我们按 EDA、数据清洗与特征工程、模型训练与调参、测试与指标报告四步走。主线是 **CatBoost 排序模型**，它在每折组（一个 `srch_id`）内直接优化排序指标，比「先分类再排序」更贴合题目。

### 2.1 数据探索（EDA）

先解压读入数据，逐列打印缺失值分布。特征缺失情况非常直观地反映出这份数据的「稀疏性」：

```python
import pandas as pd

# 内存有限，先读前 3000 万行（测试集全部读入）
df = pd.read_csv('train.csv', nrows=30123456)
df_test = pd.read_csv('test.csv')

# 缺失统计（节选）：历史浏览类、比价类字段缺失极多
print(df.isnull().sum())
# visitor_hist_starrating    9412233
# visitor_hist_adr_usd       9409918
# srch_query_affinity_score  9281966
# orig_destination_distance  3216461
# comp5_rate                 5473236
# comp8_rate                 6098487
# ...
```

可以看出：**用户历史行为类字段和竞争对手报价类字段天然缺失严重**——新访客没有历史，竞争对手没报价的不存在可比价。缺失本身就是信息，后续我们用「每行缺失数」当特征。

接着我们检查哪些列是「单值列」（只有一个取值），这类列对模型毫无区分力，必须删掉：

```python
# 单值列：整列只出现一种取值，删掉不丢任何信息
drop_unary_cols = [c for c in list(df) if df[c].nunique(dropna=False) <= 1]

# 泄漏列：训练里的答案列，评估时不存在
target_cols = ["gross_bookings_usd", "click_bool", "booking_bool"]
df = df.drop(columns=drop_unary_cols + target_cols, errors="ignore")
df_test = df_test.drop(columns=drop_unary_cols + target_cols, errors="ignore")
```

删列之后我们可以做一个极简的「特征价值」检查：按相关性分数分两桶（>0 有互动 vs =0 没互动），看每个特征在两组里的均值差多少。差得越明显，这个特征越能区分「值不值得排前面」：

```python
# 按 target 是否 > 0 分组，比较各报价特征的平均值
df.groupby(df["target"] > 0).mean()
```

结果符合直觉：`price_usd`（价格）、`prop_review_score`（评分）、`prop_starrating`（星级）这两组均值差很明显，而 `comp7_rate` 这类缺失率极高的比价特征几乎分不开用户——它们就是我们后来删除或弱化的对象。

### 2.2 数据清洗与特征工程

洗掉无信息和泄漏列之后，我们做**时间特征**和**缺失数特征**两件事：

```python
def featurize_df(df):
    # 时间特征：星期几、一年第几周、几点、第几分钟
    df["weekday"] = df["date_time"].dt.weekday
    df["week_of_year"] = df["date_time"].dt.week
    df["hour"] = df["date_time"].dt.hour
    df["minute"] = df["date_time"].dt.minute
    # 时间戳转成秒级数字：让模型学到连续的时间趋势
    df["time_epoch"] = df["date_time"].astype('int64') // 1e9
    # 每行缺失的特征个数：缺失多 = 信息少 = 相关结构不同
    df["nans_count"] = df.isna().sum(axis=1)
    return df
```

注意一个顺序陷阱：**时间特征必须放在删除泄漏列之后再加**。因为 `nans_count` 会统计列的缺失数，如果它把 `click_bool`、`booking_bool` 也数进去，等于又把泄漏信息带回了特征里。

我们也参考了 `Feature Engineering[K means]` 那份材料的思路：Train 读入后先 `dropna(axis=1, how="any")` 让所有特征没有缺失，之后用 **K-means 聚类**把用户/属性映射到聚类簇号，再把簇号当类别特征喂给模型——聚类把连续、缺失分布复杂的空间压成了少数几个「原型」，对极度稀疏的数据很有用。

### 2.3 模型训练与超参数调优

排序模型缺乏现成的「组内排序」监督信号时，一个常用兜底是先用分类模型。材料里先做了一版**分类/ AUC 模型**：把目标转成相关性分后给 CatBoost 分类器训，评估时看它对正例的区分能力：

```python
from catboost import CatBoostClassifier

clf = CatBoostClassifier(iterations=500, random_seed=42, verbose=100)
clf.fit(X_train, y_train, eval_set=(X_eval, y_eval))
# 分类视角的报告：精确率、召回率，再加 AUC 看对正例的排序能力
print(classification_report(y_eval, clf.predict(X_eval)))
print("AUC:", roc_auc_score(y_eval, clf.predict_proba(X_eval)[:, 1]))
```

这一版的意义是「把相关和不相关的候选分开」，但它并不知道一个组里「谁比谁更值得排前面」——当一组候选全是高分值时，AUC 帮不了排序。所以它只当对照，真正定胜负的还看下面的排序模型。

主力还是**排序模型**。CatBoost 的 `Pool` 支持传入 `group_id`：同一个 `srch_id` 的所有候选构成一组，模型在组内学习「谁该排谁前面」。类别特征（酒店、目的地、星期几）直接通过 `cat_features` 声明，CatBoost 会自动做有序目标编码，省去手工 prepare 的功夫：

```python
from catboost import CatBoost, Pool

categorical_cols = ['prop_id', 'srch_destination_id', 'weekday']

train_pool = Pool(data=X_train, label=y_train,
                  cat_features=categorical_cols,
                  group_id=X_train["srch_id"])   # 关键：声明分组列
eval_pool = Pool(data=X_eval, label=y_eval,
                 cat_features=categorical_cols,
                 group_id=X_eval["srch_id"])

params = {
    'iterations': 2000,
    'custom_metric': ['NDCG', "AUC:type=Ranking"],  # 直接看排序指标
    'random_seed': 42,
    "has_time": True,             # 时间序列数据，禁用随机打乱
    "use_best_model": True,       # 用验证集上的最优迭代次数
}
model = CatBoost(params)
model.fit(train_pool, eval_set=eval_pool)
```

训练/验证切分也很讲究：为了不把「未来」泄漏进「过去」，我们按 `srch_id` 的顺序切——取 `srch_id` 排序后大约 94% 分位的截断点，前 94% 当训练、后 6% 当验证。这样**同一组搜索绝不会同时出现在训练和验证两边**：

```python
cutoff_id = df["srch_id"].quantile(0.94)
X_train = df.loc[df.srch_id <  cutoff_id].drop(["target"], axis=1)
X_eval  = df.loc[df.srch_id >= cutoff_id].drop(["target"], axis=1)
```

调参分两头：一是整体迭代轮数 `iterations` 用 2000 起步，配合 `use_best_model=True` 让模型在验证 NDCG 不再涨时自动取最优轮数；二是靠树深、学习率做小规模搜索。因为数据有千万行，每多跑一组超参都是分钟级的开销，我们优先让 `custom_metric` 直接报 NDCG，把模型的超参调优和「这个特征加不加」的决策绑在一起看。

### 2.4 模型测试与指标报告

训练结束，我们先摘出模型在验证集上的最优排序分，并做 **特征重要性分析**：

```python
# 打印训练/验证两组的最好 NDCG 分
print("best results (train on train):")
print(model.get_best_score()["learn"])
print("best results (on validation set):")
print(model.get_best_score()["validation"])

# 树模型自带的特征重要性（按对分裂贡献排序）
display(model.get_feature_importance(data=train_pool, prettified=True).head(15))
```

再配一份 **SHAP** 值解释——它不仅按重要性排序，还能说明每个特征把排序分往高推还是往低拉，这对理解「价格 vs 位置 vs 比价」谁在驱动成交很有帮助：

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(pd.concat([X_train, X_eval]),
                                    y=pd.concat([y_train, y_eval]))
shap.summary_plot(shap_values, pd.concat([X_train, X_eval]))
```

最后对每个测试 `srch_id`，按模型的预测相关性分降序输出候选酒店的 `prop_id` 列表，提交即可。材料没有披露最终 NDCG 数值，我们的判断依据是验证集上的 NDCG 与 AUC 逐步爬升、且特征重要性识别出「价格、星级、评论分、目的地」等合理变量，说明模型学到了实质规律而非噪声。

---

## 三、总结

本文中我们完成了 Expedia 个性化排序比赛的主流方案梳理。我们首先用 EDA 认识了数据：训练集近 1 亿行、54 列，用户历史和比价类字段缺失严重且存在大量单值列；其次做了清洗与特征工程——删除单值列和三个泄漏列，再从时间上拆出星期/小时/周，并加上每行缺失数特征，还参考了用 K-means 聚类簇做特征的思路；接着用带分组的 CatBoost 排序模型在 `srch_id` 组内直接优化 NDCG，按 `srch_id` 的时间分位切分训练/验证防泄漏；最后用验证集 NDCG 和 SHAP 特征重要性评估模型，输出每组搜索的排序结果。

这个比赛给我们的最大启发是：**排序问题和分类问题要用不同的模型形态去解决**。分类模型在乎「这条候选对不对」，排序模型在乎「一组候选里的相对次序对不对」；在 Expedia 这种一页有十几个候选的世界里，次序就是一切。对应地，切分数据也要按组（一个 `srch_id`）而不是按行随机切，否则同一组的候选会同时在训练和验证里出现，成绩虚高。