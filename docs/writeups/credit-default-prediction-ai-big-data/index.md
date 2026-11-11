# 贷款违约预测（Credit Default Prediction / AI Big Data）

## 一、比赛简介

### 1.1 比赛背景

银行放贷之前，最核心的一道工序叫**信用风险评估**：这个人借了钱，会不会还不上？

这道工序有两个现实约束。第一，**代价不对称**。把好人拒了，银行少赚一笔利息；把坏人放了，银行可能损失整笔本金。所以模型不能只追求「预测得准」，还得保证「真的会违约的人被抓住」。第二，**数据里充满了「人填错了」的痕迹**。申请书是客户自己填的，年收入可能少写一个零，贷款金额可能随手填个 99999999，工作年限可能写成「10+ years」这种没法直接算数的字符串。

这个比赛给的就是一份典型的个人贷款申请表数据：每位申请人的住房情况、年收入、当前工作年限、开立的账户数、信用历史长度、最大授信额度、信用问题次数、上次逾期距今多少个月、破产次数、贷款用途、贷款期限、当前贷款金额、当前信用余额、月负债、信用分，以及最终结果 `Credit Default`（0 = 按时还清，1 = 违约）。

难点集中在三处：

1. **类别不平衡**。真正违约的人是少数，模型很容易学成「全部预测为不违约」，准确率看起来很高，但一个违约都抓不到。
2. **缺失值不是随机缺的**。`Credit Score` 缺失往往意味着「这个人信用记录太短」，`Months since last delinquent` 缺失往往意味着「从来没逾期过」——这两种缺失的含义完全相反，填同一个数就错了。
3. **原始特征不够用**。光看「年收入 5 万」和「月负债 2000」，你判断不了风险；要看「月负债占月收入的多少」。这类**比率特征**必须自己造。

### 1.2 比赛目标

训练集 `train.csv` 有 **17 列**：16 个特征加 1 个目标列 `Credit Default`。特征可以分成三类：

- **类别型**：`Home Ownership`（自有/租房/按揭）、`Years in current job`（当前工作年限，取值形如 `< 1 year`、`2 years`、`10+ years`）、`Purpose`（贷款用途）、`Term`（贷款期限，Short Term / Long Term）；
- **数值型**：`Annual Income`、`Current Loan Amount`、`Current Credit Balance`、`Monthly Debt`、`Credit Score`、`Maximum Open Credit`、`Number of Open Accounts`、`Years of Credit History`、`Months since last delinquent`；
- **计数型**：`Tax Liens`（税务留置）、`Bankruptcies`（破产次数）、`Number of Credit Problems`（信用问题次数）。

测试集没有 `Credit Default` 列，我们要给每条记录预测 0 或 1，提交文件沿用 `sampleSubmission.csv` 的格式，填的是 `Credit Default` 这一列。

### 1.3 评价标准

材料里的模型把 **F1** 直接设成了 CatBoost 的评估指标（`eval_metric='F1'`），并开启了 `auto_class_weights='Balanced'`。这说明比赛是按 F1 打分的。

F1 是**精确率和召回率的调和平均**：

- **精确率（Precision）**：模型说「这个人会违约」的那批人里，真正违约的占多少；
- **召回率（Recall）**：所有真正违约的人里，被模型抓出来多少。

F1 把这两个数揉在一起，任何一个太低都会把 F1 拉下去。**F1 越高越好**，满分是 1。为什么不用准确率？因为在违约者只占少数的情况下，「全部预测为不违约」能拿到很高的准确率，但召回率是 0，F1 也就是 0——**F1 对「偷懒的模型」毫不留情**。

---

## 二、解题过程

我们的主线方案可以概括成一句话：**先用回归模型把两个关键缺失字段补出来，再造一批财务比率特征，最后用 CatBoost 做分类并调参。** 我们按 EDA、数据清洗、模型训练与调参、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

第一步把数据读进来，看形状、类型和缺失：

```python
import pandas as pd
import numpy as np

TRAIN_DATASET_PATH = "/kaggle/input/credit-default-prediction-ai-big-data/train.csv"
TEST_DATASET_PATH  = "/kaggle/input/credit-default-prediction-ai-big-data/test.csv"

df_train_base = pd.read_csv(TRAIN_DATASET_PATH)
df_test_base  = pd.read_csv(TEST_DATASET_PATH)

# info() 一次看清：每列的类型、非空数量
# object 类型的就是类别列，float64/int64 是数值列
print(df_train_base.shape)
df_train_base.info()

# 逐列数缺失，只看缺了的
missing = df_train_base.isnull().sum()
print(missing[missing > 0])
```

缺失集中在五列：`Annual Income`、`Months since last delinquent`、`Bankruptcies`、`Years in current job`、`Credit Score`。这五列恰好覆盖了「收入、逾期、破产、工作、信用分」五个维度，都是风控里最关键的字段——所以不能简单填个中位数了事。

接着我们看目标分布，确认不平衡的程度：

```python
# 0 = 按时还清，1 = 违约
counts = df_train_base['Credit Default'].value_counts().sort_index()
print(counts)

# 违约占比：这个数字越小，越不能拿准确率当指标
print('违约比例: {:.2%}'.format(counts[1] / counts.sum()))
```

违约者是明显的少数派。这一点直接决定了后面三件事：评估要用 F1、训练要开类别权重、切分验证集要用**分层抽样**（保证验证集里违约比例和训练集一致）。

**然后是最有价值的一步：找「不可能的值」。** 真实数据里，异常值往往不是极端大，而是**填表人随手填的哨兵值**：

```python
# 贷款金额填了 99999999，这显然不是真金额，而是"我不知道/不想填"
print((df_train_base['Current Loan Amount'] == 99999999).sum())

# 信用分的正常上限是 850 分，超过 1000 的一定是脏数据
print((df_train_base['Credit Score'] > 1000).sum())

# 信用历史长度超过 55 年，对一个成年借款人来说不合常理
print((df_train_base['Years of Credit History'] > 55).sum())
```

三处都查出了问题。这三个检查看起来不起眼，但如果不管：`99999999` 会把贷款金额的均值直接拉到天上，`Credit Score > 1000` 会让「贷款金额 / 信用分」这类比率特征出现荒谬的小值，模型会被这些点带偏。

**最后一件事：验证「信用分到底有没有区分度」。** 我们画了两条核密度曲线，把违约者和非违约者的信用分分布叠在一起，然后做了两个统计检验：

```python
from scipy.stats import shapiro, mannwhitneyu
from matplotlib import pyplot as plt
import seaborn as sns

# 只看合法范围内的信用分
cs = df_train_base[df_train_base['Credit Score'] < 1000][['Credit Score', 'Credit Default']]

# Shapiro-Wilk：检验"信用分是否服从正态分布"
# p 值很小 → 不是正态，所以后面不能用 t 检验
print(shapiro(cs['Credit Score']))

# Mann-Whitney U：非参数检验，看两组的分布是否相同
# 结论是"两组均值相等的概率被否定"，即违约者和非违约者的信用分确实不同
print(mannwhitneyu(cs[cs['Credit Default'] == 0]['Credit Score'],
                   cs[cs['Credit Default'] == 1]['Credit Score']))

sns.kdeplot(cs[cs['Credit Default'] == 0]['Credit Score'], label='class 0')
sns.kdeplot(cs[cs['Credit Default'] == 1]['Credit Score'], label='class 1')
plt.show()
```

这一步的意义不只是「画个图」。先做正态性检验、发现不满足正态，才改用非参数的 Mann-Whitney U 而不是 t 检验——**这是统计检验的正确顺序**，顺序反了结论就不可信。检验结果确认信用分在两类人之间确实有差异，说明它是有效特征，值得在后面重点用。

### 2.2 数据清洗

**第一步：处理哨兵值和明显错误。** 我们把 `99999999` 换成缺失，再统一用中位数补；同时造一个「这行数据可疑」的标记特征：

```python
# 把哨兵值 99999999 视为缺失：它不是金额，是"没填"
df_train_base.loc[df_train_base['Current Loan Amount'] == 99999999,
                  'Current Loan Amount'] = np.nan

# 中位数从训练集算，测试集复用同一套，避免分布不一致
medians = df_train_base[['Current Loan Amount', 'Annual Income', 'Credit Score']].median()
df_train_base['Current Loan Amount'].fillna(medians['Current Loan Amount'], inplace=True)

def feature_isErrors(df):
    """标记"数据本身有问题"的行——错误本身也是一种信号"""
    df['isErrors'] = 0
    # 信用历史超过 55 年、或信用分超过 1000，都标记为可疑
    df.loc[df['Years of Credit History'] > 55, 'isErrors'] = 1
    df.loc[df['Credit Score'] > 1000, 'isErrors'] = 1
    return df
```

把错误**当成特征而不是删掉**，是一个很实用的取舍：可疑记录往往集中在「资料填得潦草」的客户身上，而这类客户的违约率确实可能不同。删掉就丢了这个信息。

**第二步：用心填缺失。** 这一节是整份方案里最有特色的地方。我们不是简单填中位数，而是**先训两个回归模型，再用模型预测缺失值**：

```python
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

def fit_and_fill_credit_score(df):
    # 先标出哪些行缺信用分，这个标记后面也要保留成特征
    df['Credit_Score_isnan'] = df['Credit Score'].isna().astype(int)

    # 只用"没缺"的行来训练回归模型
    known = df[df['Credit_Score_isnan'] == 0].copy()
    known = known.drop(['Credit Default'], axis=1)     # 目标列不能进特征

    target = 'Credit Score'
    feats = [c for c in known.columns if c != target]
    # 类别列要告诉 CatBoost，让它自己做类别编码
    cat_feats = known[feats].select_dtypes(include=['int64', 'object']).columns.tolist()

    Xtr, Xva, ytr, yva = train_test_split(known[feats], known[target],
                                          test_size=0.3, random_state=42)

    model = CatBoostRegressor(
        eval_metric='R2',          # 回归任务看 R²
        depth=4, iterations=400, learning_rate=0.05, l2_leaf_reg=2,
        one_hot_max_size=20,       # 取值少于 20 种的类别列直接独热
        early_stopping_rounds=40,  # 40 轮不进步就停
        boosting_type='Ordered',   # 小数据上更稳的有序提升
        silent=True, allow_writing_files=False)
    model.fit(Pool(Xtr, ytr, cat_features=cat_feats),
              eval_set=Pool(Xva, yva, cat_features=cat_feats))

    # 只对缺的那批行做预测，填回去
    miss = df['Credit_Score_isnan'] == 1
    Xmiss = df.loc[miss].drop([target, 'Credit Default'], axis=1, errors='ignore')
    df.loc[miss, target] = model.predict(Xmiss)
    return df, model
```

`Annual Income` 用同样的套路再补一次（`depth=6, iterations=500`）。为什么值得为两个字段专门训两个模型？因为**这两个字段太重要了**：信用分和收入是所有财务比率的分母或分子，填错一个数，后面造出来的五六个特征全跟着错。用中位数填等于告诉模型「这个人和大多数人一样」，而用回归填至少能利用「他的贷款金额、账户数、工作年限」这些信息给出一个更贴近的估计。

另外，`Credit_Score_isnan` 和 `Annual_Income_isnan` 这两个标记列我们**留下来了**。因为「收入没填」这件事本身可能和风险有关——不愿填收入的人，也许收入本身就不稳定。

**第三步：其他缺失按语义填。**

```python
# 工作年限缺失：填 "0 years" 而不是众数
# 因为这一栏的空值通常意味着"没有工作"或"刚入职"，不是"忘了填"
df['Years in current job'] = df['Years in current job'].fillna('0 years')

# 逾期月数缺失：填 0，表示从来没逾期过（注意这里 0 是"最好"的值）
df['Months since last delinquent'] = df['Months since last delinquent'].fillna(0)

# 破产次数缺失：填众数（绝大多数人是 0 次）
df['Bankruptcies'] = df['Bankruptcies'].fillna(df['Bankruptcies'].mode()[0])
```

这三行体现了同一条原则：**填什么值，取决于「缺失」在这个业务场景里代表什么**。工作年限缺 = 没工作，逾期月数缺 = 没逾期，这两种「缺」的方向完全相反，一个填字符串、一个填 0。如果统一填中位数，就把方向搞反了。

**第四步：造财务比率特征。** 这是风控建模的核心手艺。原始金额没有可比性——月负债 3000 对月入 3 万的人很轻松，对月入 5000 的人就是灾难。所以我们把绝对金额转成比率：

```python
def build_financial_features(df):
    # 1) 月负债占月收入的比例（DTI，Debt-to-Income）
    #    银行业公认的红线是 36%~43%，超过就属于高风险
    df['Debt_to_Income'] = df['Monthly Debt'] / (df['Annual Income'] / 12)

    # 2) 信用额度使用率：已用额度 / 最大授信
    #    常年刷爆卡的人，现金流通常也紧张
    df['Credit_Utilization'] = df['Current Credit Balance'] / df['Maximum Open Credit']

    # 3) 贷款额是年收入的几倍（LTI，Loan-to-Income）
    #    借的钱相当于几年收入，倍数越高越难还
    df['Loan_to_Income'] = df['Current Loan Amount'] / df['Annual Income']

    # 4) 每个账户平均背着多少余额：债务是否集中在少数账户
    df['Credit_per_Account'] = df['Current Credit Balance'] / df['Number of Open Accounts']

    # 5) 未覆盖债务：贷款金额和当前余额的差额
    #    差额越大，说明这笔钱已经被用得越多
    df['uncovered debt'] = (df['Current Loan Amount'] - df['Current Credit Balance']).abs()

    # 6) 每月可支配的钱 = 月收入 - 月负债
    #    这个值直接反映"还完债还剩多少能生活"
    df['money for living'] = (df['Annual Income'] / 12) - df['Monthly Debt']

    # 7) 贷款额 / 信用分：把"借多少"和"信用多好"放在一起看
    df['division_loan_to_score'] = df['Current Loan Amount'] / df['Credit Score']

    # 8) 账户积累速度：开账户越快的，往往越缺钱
    df['accounts_per_year'] = df['Number of Open Accounts'] / df['Years of Credit History']

    # 9) 高风险标记：DTI 超 40% 或额度使用率超 80%，打一个标记
    df['high_risk_flag'] = ((df['Debt_to_Income'] > 0.4) |
                            (df['Credit_Utilization'] > 0.8)).astype(int)
    return df
```

这九个特征有一个共同点：**它们都是「比率」或「差值」，量纲被消掉了**。这样一个月入 5000 借 1 万的人，和月入 5 万借 10 万的人，在 `Loan_to_Income` 这一维上会被正确地认为「风险相同」。相比之下，如果直接把 `Annual Income` 和 `Current Loan Amount` 丢给树模型，它得靠分裂很多次才能勉强拼出这个比率，效率低得多。

### 2.3 模型训练与超参数调优

**为什么选 CatBoost。** 这份数据里混着类别列（住房情况、贷款用途）、计数列（破产次数）、连续列（收入、信用分），而且样本量不大。CatBoost 有三个正好对症的优点：一是**原生支持类别特征**，不用手工做独热；二是它的**有序提升**（`boosting_type='Ordered'`）在小数据上比普通梯度提升更不容易过拟合；三是它内置了 `auto_class_weights='Balanced'`，一行就能解决类别不平衡。

**构造 Pool 并划分验证集。** CatBoost 要求把所有数据打包成 `Pool` 对象，并显式声明哪些列是类别列：

```python
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

target_name = 'Credit Default'
feature_names = [c for c in df_train.columns if c != target_name]
# 类别列：dtype 是 object 的那些
feature_names_cat = df_train.select_dtypes(include=['object']).columns.tolist()

X = df_train[feature_names]
y = df_train[target_name]

# 分层抽样：保证验证集里违约比例和整体一致
# 如果不分层，随机一切可能切出一个"几乎没有违约者"的验证集，F1 会剧烈波动
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

train_pool = Pool(X_train, y_train, cat_features=feature_names_cat)
valid_pool = Pool(X_valid, y_valid, cat_features=feature_names_cat)
```

**设定目标函数和评估指标。** 这一步决定了模型「往哪个方向优化」：

```python
params_model = {
    'eval_metric': 'F1',              # 和比赛指标对齐，训练时直接看 F1
    'auto_class_weights': 'Balanced', # 按类别频率自动加权，抵消不平衡
    'one_hot_max_size': 20,           # 取值少的类别列用独热，比目标编码更稳
    'early_stopping_rounds': 50,      # 50 轮没进步就停
    'boosting_type': 'Ordered',       # 小数据防过拟合
    'silent': True,
    'allow_writing_files': False,
}
```

这里要强调 `eval_metric='F1'` 和 `auto_class_weights='Balanced'` 的配合：**前者决定「怎么衡量好坏」，后者决定「训练时更在意谁」**。只设 F1 不设权重，模型仍然会倾向于多数类；只设权重不设 F1，早停会按准确率或对数损失判断，可能在 F1 还没到最高时就停下来。

**调参。** 我们用 CatBoost 自带的 `randomized_search` 做随机搜索，而不是全网格：

```python
cbr_model = CatBoostClassifier(**params_model)

params_grid = {
    'depth': [4, 5, 6, 7, 8],                    # 树深：越深越容易过拟合
    'learning_rate': [0.03, 0.1, 0.16, 0.2],     # 学习率
    'iterations': [100, 200, 400],               # 树的棵数
    'l2_leaf_reg': [0.5, 1, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10],  # L2 正则
    'bagging_temperature': [1, 1.5, 2],          # 采样温度，控制随机性
}

gs = cbr_model.randomized_search(
    params_grid, train_pool,
    n_iter=50,          # 只随机试 50 组，而不是把 5×4×3×13×3 全跑一遍
    shuffle=True,
    stratified=True,
    partition_random_seed=42,
    cv=5)               # 5 折交叉验证
```

全网格是 5×4×3×13×3 = 2340 组，每组还要 5 折，根本跑不完。随机搜索只试 50 组，命中好区域的概率却很高——**因为超参数的重要性差别很大，depth 和 learning_rate 的影响远大于 bagging_temperature**，随机采样天然会更频繁地在重要维度上取到不同值。

搜索给出的最优组合是 `depth=8, iterations=400, learning_rate=0.1, l2_leaf_reg=3.5, bagging_temperature=1.5`；而在 `boosting_type='Ordered'` 下，另一组 `depth=4, iterations=200, learning_rate=0.2, l2_leaf_reg=2.5` 也表现不错。最终我们选了一组折中的配置——`depth=6, iterations=400, learning_rate=0.2, l2_leaf_reg=8.5, bagging_temperature=1`。深度 6 比搜索出的 8 浅一些，L2 从 3.5 加大到 8.5，**这是有意往「更保守」的方向调**：在只有几千条样本、且正类很少的任务上，宁可欠拟合一点，也不要让模型记住少数几个违约样本的噪声。

### 2.4 模型测试与指标报告

训练完之后，我们在**训练集和验证集上分别**输出分类报告，并画混淆矩阵：

```python
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

def evaluate_preds(model, y_train, y_valid, X_train, X_valid):
    # 分别在训练集和验证集上预测
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)

    # 训练集报告：看有没有过拟合
    print('Train\n', classification_report(y_train, y_train_pred))
    # 验证集报告：看真实泛化能力
    print('Valid\n', classification_report(y_valid, y_valid_pred))
    # 混淆矩阵：看清楚"漏抓"和"误抓"各有多少
    print(pd.crosstab(y_valid, y_valid_pred))

evaluate_preds(cat_model, y_train, y_valid, X_train, X_valid)
```

看这份报告时，我们关注的是**两类的 F1 是否接近**。如果「不违约」那一类的 F1 有 0.9 多、「违约」只有 0.2 出头，说明模型虽然整体分数不差，但基本没学会识别风险客户——这正是类别不平衡任务里最常见的陷阱。训练集和验证集的差距则反映过拟合程度：训练集 F1 明显高于验证集，就说明该加正则或者减少迭代轮数了。

我们另外算了一个 ROC-AUC 作为辅助参考：

```python
# AUC 衡量"排序能力"：随机抽一个违约者和一个非违约者，模型给违约者打分更高的概率
auc = roc_auc_score(y_valid, cat_model.predict_proba(X_valid)[:, 1])
fpr, tpr, _ = roc_curve(y_valid, cat_model.predict_proba(X_valid)[:, 1])

print('ROC AUC: {:.4f}'.format(auc))
# 注意：这里传的是 predict_proba 的第二列（违约概率），
# 不是 predict 出来的 0/1 标签——用标签算 AUC 会退化成两个点
```

**特征重要性。** CatBoost 可以直接给出每个特征贡献了多少：

```python
feature_importances = pd.DataFrame(
    zip(X_train.columns, cat_model.get_feature_importance()),
    columns=['feature_name', 'importance']
).sort_values('importance', ascending=False)

print(feature_importances.head(20))
```

这一步不只是「看看排名」，它能验证我们前面的假设：如果 `Debt_to_Income`、`Credit_Utilization`、`Loan_to_Income` 这些手工造的特征排在前列，说明**比率化这条路走对了**；如果排在前面全是 `Current Loan Amount` 这种原始金额，说明比率特征没起作用，需要回头看分母有没有算错（比如除以了 0）。材料里还有专门的 SHAP 分析，思路一样，只是能进一步给出「这个特征把预测往哪个方向推、推了多少」。

**生成提交。** 最后对测试集做完全相同的清洗和特征工程（注意必须复用训练集算出的中位数和缺失标记），再预测：

```python
# 测试集走同一套流程：补哨兵值 → 填缺失 → 造比率特征
df_test = data_inst.transform(df_test_base)
df_test = data_inst.features(df_test)

# 预测 0/1
y_pred_test = cat_model.predict(df_test[feature_names])

# 直接在样例提交文件上覆盖目标列，保证行数、列名、顺序都和官方要求一致
final_submit = pd.read_csv(
    "/kaggle/input/credit-default-prediction-ai-big-data/sampleSubmission.csv")
final_submit['Credit Default'] = y_pred_test
final_submit.to_csv('./predictions.csv', index=False)
```

这里有个细节值得学：我们**没有自己从零拼提交表**，而是读入官方的 `sampleSubmission.csv` 再覆盖目标列。这样行数、行序、列名都不可能出错——比赛里因为提交格式翻车的队伍，比因为模型不够好丢分的队伍还多。

---

## 三、总结

本文中我们完成了贷款违约预测的完整流程。我们首先用 EDA 摸清了数据：训练集 17 列，缺失集中在收入、逾期、破产、工作年限、信用分五个关键字段，违约者是明显的少数派；我们还专门排查了三种「不可能的取值」——贷款金额被填成 99999999、信用分超过 1000、信用历史超过 55 年，并用 Shapiro-Wilk 和 Mann-Whitney U 两个检验确认了信用分在违约与非违约人群之间确实有显著差异。其次我们做了有业务含义的清洗：把哨兵值当缺失、把「数据可疑」做成 `isErrors` 标记特征、用 CatBoost 回归模型（`eval_metric='R2'`、`boosting_type='Ordered'`）分别预测并填补信用分和年收入、把工作年限缺失填成 `"0 years"`、逾期月数缺失填成 0、破产次数填众数，并保留了缺失标记列。接着我们造了九个财务比率特征（DTI、额度使用率、LTI、每账户余额、未覆盖债务、月可支配收入、贷款额/信用分、账户积累速度、高风险标记），用 CatBoost 分类器配 `eval_metric='F1'` 和 `auto_class_weights='Balanced'` 训练，用 5 折 `randomized_search` 在深度、学习率、迭代数、L2 正则、采样温度上搜了 50 组，最终选了偏保守的 `depth=6, l2_leaf_reg=8.5`。最后我们在训练集和验证集上分别输出分类报告和混淆矩阵、用 AUC 辅助验证排序能力、用特征重要性反查比率特征是否生效，并把预测结果覆盖到官方样例提交文件上生成最终提交。

这个比赛给我们最大的启发是：**特征工程的价值远大于模型选择。** 把「年收入 5 万、月负债 2000」换成「负债收入比 48%」，信息量是一样的，但对模型来说难度天差地别——前者需要它自己学会做除法，后者直接就是一个可分裂的阈值。第二个启发是**缺失值要按语义填**：同样是空值，工作年限的空代表「没工作」，逾期月数的空代表「没逾期」，方向完全相反，一个填 `"0 years"` 一个填 0，填反了模型学到的就是噪声。最后，在类别不平衡的任务上，**指标、损失、采样三者必须同时对齐**：评估用 F1、训练开类别权重、切验证集用分层抽样，缺一个都会让分数看起来不错但实际没用。
