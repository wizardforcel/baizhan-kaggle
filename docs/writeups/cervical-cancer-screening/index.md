# 宫颈癌筛查预测（Cervical Cancer Screening）

## 一、比赛简介

### 1.1 比赛背景

宫颈癌是女性常见的恶性肿瘤之一，但如果能定期做筛查（比如宫颈涂片），就能在癌变之前发现问题，大幅降低死亡率。问题是：**哪些人没有来做筛查？** 如果能提前找出「高概率不筛查」的人群，公共卫生部门就可以把有限的宣传和补贴资源投到这些人身上，效果比「全民撒网」好得多。

这个比赛给的是一批匿名的**医保理赔记录**。数据放在一个 SQLite 数据库 `database.sqlite` 里，一共 12 张表：

- `patients_train`、`patients_test`：患者主表，含年龄段、所在州、种族、家庭收入、教育程度等人口学字段，以及目标列 `is_screener`（训练集才有）；
- `diagnosis`、`diagnosis_code`：诊断记录和诊断码字典；
- `procedure`、`procedure_code`：操作/检查记录和操作码字典；
- `prescription`、`drug`：处方和药品；
- `surgical`、`surgical_code`：手术记录和手术码字典；
- `physician`：医生信息；
- `patient_activity`：患者活动记录。

这些表通过 `patient_id` 串起来，形成典型的「一对多」关系：一个患者可以有多条诊断、多条处方、多次手术。比赛的难点也在这里——**原始数据是「长表」，不能直接喂给模型，必须先聚合成「一人一行」**。

另一个难点是数据里有大量「泄漏」性质的信号。比如诊断码 `V72.31` 的含义是「常规妇科检查」，一个人既然做了常规妇科检查，她大概率就是来筛查的。材料里给了一个非常直观的数字：训练集里约 **56%** 的患者至少有过一次 `V72.31`，这一组的筛查率约 **81%**；而不在这一组的人，筛查率只有约 **24%**。也就是说，光看「有没有这个诊断码」，就能把人群分得很开。

### 1.2 比赛目标

输入是一位患者的历史理赔记录（诊断、操作、处方、手术等），输出她属于「筛查者」的概率，也就是 `is_screener` 的预测值（0～1）。提交文件按 `patient_id` 给出一列概率。

### 1.3 评价标准

比赛用 **ROC 曲线下面积（AUC）** 评分。AUC 衡量的是「随机抽一个正样本和一个负样本，模型给正样本打分更高的概率」，取值 0.5 表示和瞎猜一样，1.0 表示完美区分。AUC 越高越好。

材料里也是用 AUC 说话的：用很少的几个特征就能把 AUC 做到接近 **0.93**；某些「组平均」特征单独拿出来就有 **0.88+** 的 AUC；引入「诊断码 × 主治医生」这个交互后，AUC 整整提升了 **1 个百分点**（如果两个主效应已经都在模型里，还能再提升 0.5 个百分点）。

---

## 二、解题过程

整场比赛的思路可以概括成：**先用 SQL 把「一对多」的理赔记录聚合成「一人一行」的特征，再针对性地构造「组平均」类特征，最后用一个不复杂的 XGBoost 分类器打分。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先连上 SQLite，把所有表名列出来，确认数据的组织方式：

```python
import sqlite3
import pandas as pd

# 整个比赛只有一个 SQLite 文件，所有表都靠 patient_id 关联
con = sqlite3.connect('../input/database.sqlite')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
# [('patients_train',), ('patients_test',), ('diagnosis',), ('diagnosis_code',),
#  ('drug',), ('patient_activity',), ('prescription',), ('procedure',),
#  ('procedure_code',), ('surgical',), ('surgical_code',), ...]
```

看到这张表清单，我们心里就有数了：主表只有两张（训练 / 测试），其余十张都是「明细表」，需要按 `patient_id` 聚合。

接着我们看最基本的规律——筛查率和年龄的关系。这里顺手用二项分布给每个年龄组算一个 95% 置信区间，样本小的组区间会很宽，避免我们被小样本的波动骗了：

```python
from scipy import stats

# 人口学字段都在 patients_train 这一张表里
sdf = pd.read_sql_query(
    "SELECT patient_age_group, patient_state, ethinicity, household_income, "
    "education_level, is_screener FROM patients_train;", con)

# 每个年龄组的人数，以及该组的筛查率（is_screener 的平均值）
age_totals = sdf.groupby('patient_age_group').size()
age_rates  = sdf.groupby('patient_age_group').apply(
    lambda x: x.is_screener.sum() / x.shape[0])

# 95% 置信区间 = 二项分布的分位数除以该组人数
ci = [stats.binom.interval(.95, age_totals[g], age_rates[g]) / age_totals[g]
      for g in age_rates.index]
for g, r, (lo, hi) in zip(age_rates.index, age_rates, ci):
    print(f'{g}: {r:.3f}  [{lo:.3f}, {hi:.3f}]')
```

结果很清楚：**筛查率随年龄增长而下降**。这符合直觉——年轻女性可能因为孕产、妇科检查等原因更频繁地接触医疗系统，顺带就做了筛查。

那这个下降趋势是不是所有地区都一样？我们按州再拆一层：

```python
# 每个州、每个年龄组各算一个筛查率
sr_age_state = sdf.groupby(['patient_age_group', 'patient_state']).apply(
    lambda x: x.is_screener.sum() / x.shape[0])
state_totals = sdf.groupby(['patient_age_group', 'patient_state']).size()

# 只取患者最多的 10 个州，样本太小画出来全是噪声
top_states = state_totals.unstack().sum(axis=0).sort_values(ascending=False).index[:10]

# 把 10 条曲线画在一张图上，看“随年龄下降”是不是普遍规律
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(17, 12))
for st in top_states:
    sr_age_state.unstack()[st].plot(ax=ax, label=st, lw=2)
ax.legend()
ax.set_xlabel('Age Group')
ax.set_ylabel('Screener Rate')
```

答案是「不一样」。材料里特别点出了几个州：加州（California）很反常，筛查率不是严格递减，最年轻的组反而比中间年龄组更低；佛罗里达和马里兰整体筛查率偏低，但佛罗里达各年龄组比较平均，马里兰则是年轻组还不错、之后就断崖式下跌。这说明「年龄」不是唯一因素，地区和医疗资源可及性也在起作用。

最后我们看种族、家庭收入、教育程度。结论是：**有差异，但不大**。家庭收入越高越可能筛查；非裔美国人在大多数年龄组低于白人、西班牙裔，但在最年轻的组反而略高；受教育程度越高越可能筛查，不过这个分层在最年轻的组不明显。

### 2.2 数据清洗

原始数据不能直接用，必须先「聚合」。我们按两条线做：一条是「诊断码 / 操作码的统计」，另一条是「构造组平均特征」。这两条线都大量依赖 SQL，因为用 pandas 在几百万行明细上做多次 groupby 会非常慢。

先看诊断码的筛查率。这里有个细节：一个患者同一种诊断会出现很多次，所以要先对 `(patient_id, diagnosis_code)` 去重，否则「出现次数多」的人会把统计带偏。

```python
# 先对 (patient_id, diagnosis_code) 去重，再算每个诊断码对应人群的筛查率
query = """
SELECT ds.diagnosis_code,
       COUNT(ds.diagnosis_code) AS freq,
       AVG(pt.is_screener)      AS screener_rate
FROM (
    SELECT DISTINCT patient_id, diagnosis_code
    FROM diagnosis
    LIMIT 2000000              -- 控制单次查询规模，先取样再逐步放大
) AS ds
INNER JOIN patients_train AS pt ON pt.patient_id = ds.patient_id
GROUP BY ds.diagnosis_code
ORDER BY freq;
"""
dcode_rates = pd.read_sql_query(query, con)

# freq 是“有多少患者出现过这个诊断码”，screener_rate 是这群人的筛查率
print(dcode_rates.tail(15))    # 最常见的诊断码
print(dcode_rates.head(15))    # 最冷门的诊断码
```

材料里跑出的诊断码一共 **11891 个**。其中筛查率最高的一批包括：`V74.5`（性病筛查检查）约 0.860、`V25.09` 约 0.838、`V72.31`（常规妇科检查）约 0.806（在 2,000,000 条样本上是 5,243 人）；筛查率最低的一批包括：`428.0`（充血性心力衰竭）约 0.315、`427.31`（房颤）约 0.335、`491.21`（慢性支气管炎）约 0.341、`496`（慢性气道阻塞）约 0.379、`414.01`（冠状动脉粥样硬化）约 0.375。

读这些数字要小心：高频码和低频码的可信度完全不同。所以我们做了两步保护：第一，只看「出现次数足够多」的码；第二，把样本量作为一列一起看，不单独相信一个百分比。

操作码的分析也照做，结论类似：`87491`、`87591` 这类与筛查直接相关的操作码筛查率在 0.88～0.89；而 `J1270`、`Q4081`、`90999` 这类码的筛查率只有 0.19～0.20。这进一步说明：**理赔记录里的「检查项目」本身就是很强的信号**。

### 2.3 模型训练与超参数调优

这一步是整场比赛的核心。材料里的做法不是堆复杂模型，而是构造 **「组平均」特征**（作者叫它 likelihood 特征）：对某个类别（比如某个诊断码），先算出「有这个特征的人平均筛查率是多少」，再把这个平均值当作这位患者的一个特征。

思路很朴素：如果「有诊断码 A 的人」整体筛查率是 0.8，那么一位有诊断码 A 的患者，她的先验概率也应该接近 0.8。把很多个这样的「组平均」叠起来，模型就能很快抓住人群差异。

关键是要**避免把目标泄漏进特征**。如果直接用全体训练集算平均，再把同一个人的标签算进平均值里，模型就相当于提前看了答案。材料里的处理是：对样本量足够的组（比如至少 100 人）直接算；对样本量小的组，必须用交叉验证的方式算，保证算平均时「没见过这个人」。

```python
# 第一步：抽出所有出现过的 (患者, 诊断码) 组合并去重
# patients_all / diagnosis_head 是作者预处理后的视图，含义等同于主表 + 明细表
con.executescript("""
DROP TABLE IF EXISTS interaction_distincts;
CREATE TABLE interaction_distincts AS
SELECT DISTINCT pt.patient_id, d.diagnosis_code
FROM patients_all AS pt
INNER JOIN diagnosis_head AS d ON d.patient_id = pt.patient_id;
""")

# 第二步：算每个诊断码下人群的平均 is_screener
con.executescript("""
DROP TABLE IF EXISTS interaction_avg;
CREATE TABLE interaction_avg AS
SELECT d.diagnosis_code,
       AVG(CAST(pt.is_screener AS float)) AS avg_is_screener,
       COUNT(DISTINCT pt.patient_id)      AS cnt_patients
FROM patients_train AS pt
INNER JOIN interaction_distincts AS d ON pt.patient_id = d.patient_id
GROUP BY d.diagnosis_code;
""")

# 第三步：把“组平均”搬回患者级。只保留样本量 >= 100 的组合，
# 小样本的平均值极不稳定，等于把噪声当成了特征
risk = pd.read_sql_query("""
SELECT pt.patient_id,
       MAX(CASE WHEN a.cnt_patients >= 100 THEN a.avg_is_screener END) AS risk_max_100,
       MIN(CASE WHEN a.cnt_patients >= 100 THEN a.avg_is_screener END) AS risk_min_100
FROM patients_all AS pt
LEFT JOIN interaction_distincts AS d ON pt.patient_id = d.patient_id
LEFT JOIN interaction_avg AS a ON d.diagnosis_code = a.diagnosis_code
GROUP BY pt.patient_id;
""", con)
```

这套模板可以套在很多特征上：诊断码、操作码、主治医生、以及它们的组合。作者试过 5、10、25、50、100、250 人这几个阈值，阈值越小特征越细，但也越容易过拟合，必须配合交叉验证。材料里提到，很多这样的 `MAX` 特征单独拿出来就有 **0.88+ 的 AUC**。

接下来是筛交互。作者测试了 **120 多个交互**，用「前向选择」一个个试：如果加入某个交互后 AUC 变好，就保留；否则丢掉。用来评估的模型故意做得很简单——XGBoost，`max_depth=6`、`n_estimators=100`——因为这一步的目标是**快速筛特征**，不是刷分。

```python
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def evaluate(features):
    """用交叉验证 AUC 给一组特征打分，数字越大越好。"""
    model = XGBClassifier(max_depth=6, n_estimators=100, n_jobs=-1,
                          eval_metric='auc', use_label_encoder=False)
    return cross_val_score(model, X[features], y, cv=5, scoring='roc_auc').mean()

# 前向选择：每次只加一个候选交互，AUC 提升就留下
selected, best = [], 0.0
for feat in candidate_interactions:
    score = evaluate(selected + [feat])
    if score > best:
        best = score
        selected.append(feat)
        print('add', feat, 'AUC', round(score, 4))
```

材料里筛出来的强交互有三个：

1. **诊断码 × 主治医生**。同样一个诊断码，不同医生开出的筛查率差别很大，有的医生「过度开检查」，有的「开得太少」。这个交互单独就能把 AUC 提升 1 个百分点；即使诊断码和医生两个主效应都已在模型里，它还能再提升 0.5 个百分点。
2. **诊断码 × 索赔类型 × 医生角色**。这三个一起看，比只看诊断码强得多。一个有意思的细节是：对大多数诊断码来说，计划类型为 `HX` 会降低筛查率，但对 `V72.31` 反而会提高。
3. **患者特征 × 诊断/操作**。作者试了很多，提升不大，说明患者人口学特征和具体诊断之间没有特别强的依赖。

另外还有两个时间维度的发现：一是 2008～2014 年间，有诊断/操作记录的人群筛查率整体在缓慢下降；二是自费（cash）患者的筛查率在 2012 年之后突然掉下去，而且再也没回到之前的水平——作者推测可能和当时的经济环境有关。

### 2.4 模型测试与指标报告

最后我们在验证集上算 AUC。注意 AUC 只看「排序」，不依赖 0.5 这个分类阈值，所以我们用的是 `predict_proba` 输出的正类概率，而不是 `predict` 输出的 0/1。

```python
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 分层切分，保证验证集里筛查者/非筛查者的比例和训练集一致
X_tr, X_va, y_tr, y_va = train_test_split(X[selected], y, test_size=0.2,
                                          stratify=y, random_state=0)

model = XGBClassifier(max_depth=6, n_estimators=100, n_jobs=-1,
                      eval_metric='auc', use_label_encoder=False)
model.fit(X_tr, y_tr)

# AUC 用概率算：它衡量的是“正样本得分高于负样本”的概率
auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
print('valid AUC:', round(auc, 4))
```

材料里报告的结果是：只用很少的几个特征，AUC 就能接近 **0.93**；单看那些「组平均」特征，也有 **0.88+**。这个数字说明了一件事：**在这份数据里，特征工程比模型复杂度重要得多**。作者的原话大意是，只要能找对「头表里的哪些交互有用」，根本不需要复杂模型就能达到很高的准确率。

当然，我们也要清醒地看到隐忧：`V72.31` 这种「常规妇科检查」的诊断码，本身就是筛查行为的痕迹，模型学到的高 AUC 有一部分是「事后信息」，而不是真正的「事前预测能力」。如果真要用它来做干预（提前找出该提醒谁），就必须把这类「结果性」特征去掉，重新评估——这也是这场比赛最值得思考的地方。

---

## 三、总结

本文中我们完成了宫颈癌筛查预测的完整流程。我们首先连上 SQLite，确认了 12 张表的结构，并用年龄分组加 95% 置信区间看出「筛查率随年龄下降」这条主线，又按州、种族、收入、教育做了分层，发现地区差异比人口学差异更明显。其次我们用 SQL 把一对多的诊断、操作记录聚合成患者级统计，先对 `(patient_id, code)` 去重，再算每个码的筛查率和样本量，只信样本足够大的码。接着我们构造了「组平均」likelihood 特征，用 100 人这样的样本量阈值控制噪声、用交叉验证防止目标泄漏，并用简单的 XGBoost（`max_depth=6`、`n_estimators=100`）做前向选择，从 120 多个交互里筛出「诊断码 × 主治医生」「诊断码 × 索赔类型 × 医生角色」等强特征。最后我们在验证集上用 AUC 评估，得到接近 0.93 的成绩，并讨论了 `V72.31` 这类特征带来的「结果泄漏」风险。

这个比赛的启发有两点。第一，**对一对多数据，聚合方式和统计口径决定了天花板**：同样是「诊断码平均筛查率」，去不去重、看不看样本量、防不防泄漏，结果差别巨大。第二，**高 AUC 不等于高实用价值**：如果特征是「已经发生了筛查」的痕迹，模型就只能做事后解释，不能做提前干预。想清楚「这个特征在预测时点到底能不能拿到」，比多调几个超参数重要得多。
