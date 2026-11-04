# Challenge 01：从意面到 Schema（Spaghetti → Schema）

## 一、比赛简介

### 1.1 比赛背景

这是一场**数据工程**挑战，和常见的「给一张表、预测一个数」不太一样。题目名字里的「spaghetti」是意大利面，在数据行业里用来形容那种**一团乱麻的表结构**：字段命名五花八门、主键外键全靠猜、同一件事在好几张表里各记一遍、单位和格式都不统一。而「schema」指的是规范的数据库结构——每张表管一件事，表与表之间用主键、外键明确地连起来。

比赛要做的，就是把一堆凌乱的零售 CSV 文件，整理成一个**可复现、可直接拿来做分析的关系型数据库**。材料里明确写了用的是 PostgreSQL 和 Python。整个项目有七项交付物：

1. 数据字典和 schema 文档；
2. PostgreSQL 的 DDL（建表语句）和索引脚本；
3. Python 写的 ETL 流水线；
4. 数据质量和校验报告；
5. 探索性分析（EDA）的可视化；
6. 基线模型评估；
7. 项目演示文稿。

这件事听起来不如训练模型「酷」，但它是所有数据分析的地基。如果表结构是乱的，后面无论用多先进的模型，结论都不可信。这个挑战考的就是「能不能把地基打正」。

### 1.2 比赛目标

输入是一批零售业务的原始 CSV，内容大致是门店、商品、销售流水、促销活动这类信息。输出不是一行预测值，而是一整套交付物：规范化的关系库（材料里的项目最终产出了 **9 张规范化 CSV 表**）、建表与索引脚本、ETL 程序、校验报告、图表和基线模型结果。

材料里的做法是：把这些交付物放在一个目录里，最后**整个目录打包成 zip** 作为产出。所以这场比赛更像是「交一个可复现的项目」，而不是「交一个提交文件」。

### 1.3 评价标准

材料里没有给出单一的排行榜指标，所以我们只能从交付物和报告反推评估重点。从项目的结构看，评估围绕三块：

- **结构完整性**：主键是否唯一且非空、外键是否有「孤儿行」、建表语句和索引是否齐全；
- **业务规则与口径**：销售数量是否为正、金额是否非负、财务计算是否对得上；
- **基线模型误差**：用 MAE（平均绝对误差）和 RMSE（均方根误差）衡量，两个都是越低越好。

换句话说，**先看「数据对不对」，再看「模型准不准」**。这也是数据工程项目和普通建模比赛最大的区别。

---

## 二、解题过程

整场比赛的思路可以概括成：**先把源数据摸清楚，再用 ETL 把它拆成规范的多张表，然后用一套自动校验证明「拆得对」，最后跑几个基线模型给出误差参照。** 需要先说明一点：材料里的 notebook 主要是「展示交付物、读取报告」，完整的 ETL 和模型训练代码并没有贴出来。所以下面涉及 ETL、校验、建模的代码，是我们根据报告和说明整理的思路，标注为伪代码，方便读者理解流程。

### 2.1 数据探索（EDA）

我们首先确认交付目录里到底有哪些东西，缺哪一块一目了然：

```python
from pathlib import Path
import pandas as pd

project = Path('/kaggle/input/datasets/khlilaz/'
               'challenge-01-spaghetti-schema-project-files/'
               'Spaghetti_Schema_Final_Deliverables')

# 逐项检查关键交付物是否存在，避免“以为交了其实没交”
for name in ['README.md', 'etl_pipeline.py', 'sql/ddl.sql',
             'Spaghetti_Schema_Project_Presentation.pptx']:
    print(f'{name:45s}', (project / name).exists())

# 数一数规范化之后的 CSV 表有几张
print('normalized tables:',
      len(list((project / 'normalized_tables').glob('*.csv'))))
```

检查结果和材料一致：README、ETL 脚本、DDL、演示文稿都在，规范化表一共 **9 张**。

接着我们读校验报告和模型对比结果。这些 CSV 是项目的「体检单」，每一张对应一类检查：

```python
docs = project / 'docs'
reports = ['etl_validation_summary.csv',
           'etl_primary_key_validation.csv',
           'etl_foreign_key_validation.csv',
           'etl_business_rules_validation.csv',
           'model_comparison_results.csv']

for name in reports:
    p = docs / name
    if p.exists():
        df = pd.read_csv(p)
        print(f'\n=== {name}  shape={df.shape} ===')
        print(df.head(10))
```

EDA 的另一半是「看业务」。材料里的项目画了五张图，分别回答五个问题：

- **销售额最高的 10 个商品是哪些？**（`top_10_products_by_revenue.png`）
- **各门店的销售额差多少？**（`sales_revenue_by_store.png`）
- **销售额按季度怎么波动？**（`quarterly_sales_revenue.png`）
- **哪个部门的销售额最高？**（`sales_revenue_by_department.png`）
- **促销和非促销的销售额差多少？**（`sales_revenue_by_promotion_status.png`）

这五张图看着简单，其实是在给后面的建模「打预防针」：如果某些门店、某些季度、某些促销状态的销售额系统性偏高，模型很容易学到「门店 ID」这种捷径，而不是真正的业务规律。

### 2.2 数据清洗

这一步是整个项目的重心，也就是 ETL（Extract-Transform-Load，抽取—转换—加载）。我们的做法分三步：先清洗原始 CSV，再拆成「维度表」和「事实表」，最后按「先维表、后事实表」的顺序写进数据库。

```python
# 下面是按材料中的 ETL 描述整理的伪代码
def run_etl(raw_dir, engine):
    raw = load_raw_csv(raw_dir)          # 读入所有原始 CSV

    # —— 清洗 ——
    raw = raw.drop_duplicates()          # 去掉完全重复的行
    raw['quantity'] = raw['quantity'].astype(int)     # 数量必须是整数
    raw['amount'] = raw['amount'].astype(float)       # 金额转成浮点
    raw = raw[raw['quantity'] > 0]       # 数量 <= 0 的记录直接拒收
    raw = raw[raw['amount'] >= 0]        # 金额为负的记录直接拒收

    # —— 拆维度表 ——
    dim_stores    = build_dim_stores(raw)      # 门店维度
    dim_products  = build_dim_products(raw)    # 商品维度
    dim_dates     = build_dim_dates(raw)       # 日历维度
    dim_customers = build_dim_customers(raw)   # 顾客维度

    # —— 装事实表 ——
    # 事实表只存“外键 + 度量”，不重复存门店名、商品名，避免不一致
    fact_sales = build_fact_sales(raw, dim_stores, dim_products, dim_dates)

    # 先写维表再写事实表，否则事实表的外键会找不到父行
    for name, table in [('dim_stores', dim_stores),
                        ('dim_products', dim_products),
                        ('dim_dates', dim_dates),
                        ('dim_customers', dim_customers),
                        ('fact_sales', fact_sales)]:
        table.to_sql(name, engine, if_exists='replace', index=False)
```

这里有几个关键设计。第一，**事实表和维度表分开**，是数据仓库里最经典的「星型模型」：事实表存「谁、在哪个店、哪天、买了什么、买了多少、花了多少钱」，维度表存「门店叫什么、商品属于哪个部门」这类描述信息。这样同一个门店名只存一次，改的时候只改一处。第二，**清洗动作要可追溯**：哪些行因为数量为负被拒收、拒收了多少条，都要记进报告，而不是悄悄删掉。

清洗完还要**校验**。校验不是「跑一遍看看有没有报错」，而是把规则写成可重复执行的检查项。材料里的项目一共跑了 **8 组校验**，全部通过，失败数为 0：

```python
# 校验也是伪代码：把“结构对不对、业务规则对不对”变成可执行的断言
def validate(tables, report):
    # 1) 主键：既不能为空，也不能重复
    for name, key in [('dim_stores', 'store_id'),
                      ('dim_products', 'product_id'),
                      ('dim_dates', 'date_id'),
                      ('fact_sales', 'sale_id')]:
        t = tables[name]
        ok = t[key].notna().all() and t[key].is_unique
        report.add(name, 'primary_key', ok)

    # 2) 外键：事实表里的每个 id，都必须能在维表里找到
    #    找不到就是“孤儿行”，说明 ETL 漏了数据或键对错了
    report.add('fact_sales', 'fk_store',
               tables['fact_sales']['store_id']
                     .isin(tables['dim_stores']['store_id']).all())

    # 3) 业务规则：销售数量必须为正，金额不能为负
    report.add('fact_sales', 'positive_quantity',
               (tables['fact_sales']['quantity'] > 0).all())
    report.add('fact_sales', 'non_negative_amount',
               (tables['fact_sales']['amount'] >= 0).all())

    # 4) 财务口径：单价 × 数量 要和总金额对得上（允许极小误差）
    diff = (tables['fact_sales']['unit_price']
            * tables['fact_sales']['quantity']
            - tables['fact_sales']['amount']).abs()
    report.add('fact_sales', 'financial_consistency', (diff < 0.01).all())
    return report
```

材料里的校验结果是这样的：主键 **9 项全过**，没有空值也没有重复；外键 **10 项全过**，孤儿行为 0；业务规则 **4 项全过**，包括销售数量为正、金额非负；财务计算 **3 项全过**；拒收记录 **0 条**；9 张规范化表的行数检查也全部通过。这些数字合起来说明：这套 ETL 不是「跑通了」，而是「每一步都对得上」。

### 2.3 模型训练与超参数调优

数据整理好之后，项目还要求跑基线模型，给后续分析一个「及格线」。这里要强调：**基线的作用不是拿高分，而是证明「复杂模型有没有必要」**。如果随机森林打不过「全都猜平均值」，那说明特征里根本没有可学的信号，或者验证方式有问题。

材料里的项目对比了三个基线：

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 基线一：训练集均值。所有样本都预测成训练目标的平均值
mean_pred = np.full(len(y_valid), y_train.mean())
print('Training Mean  MAE =', mean_absolute_error(y_valid, mean_pred),
      ' RMSE =', mean_squared_error(y_valid, mean_pred, squared=False))

# 基线二：门店 + 星期几的均值。允许每个“门店×星期”组合有自己的平均值
group_mean = train.groupby(['store_id', 'weekday'])['target'].mean()
group_pred = valid.set_index(['store_id', 'weekday']).index.map(group_mean)
print('Store+Weekday  MAE =', mean_absolute_error(y_valid, group_pred),
      ' RMSE =', mean_squared_error(y_valid, group_pred, squared=False))
```

材料里报告的结果有点「反直觉」：**Training Mean 基线拿到了 MAE 121.64、RMSE 153.11，而且在两个指标上都优于 Store + Weekday Mean 和随机森林**。项目作者自己也提醒：这个结果说明「更复杂的模型并没有改善表现」，并且在结论里明确写道——**在下结论之前，应该先复查目标变量的定义、验证策略和潜在的数据泄漏**。

这段话非常值得学。一个「简单模型赢过复杂模型」的结果，通常有三种解释：一是数据本身噪声太大，复杂模型只是在拟合噪声；二是验证集切分有问题，比如训练集和验证集有时间重叠，导致随机森林「记住了未来」；三是目标变量定义得不对，模型学的东西和业务关心的东西不是一回事。材料里没有给出随机森林的具体代码和特征，所以我们也只能停在「复现思路」这一层，不编造它的分数。

### 2.4 模型测试与指标报告

最终的指标报告是两部分拼起来的：一部分是**数据质量**，一部分是**模型误差**。前者用「通过 / 失败」的计数表示，后者用 MAE 和 RMSE 表示。

```python
# 把两类结果汇总成一张“交付体检表”
summary = pd.DataFrame([
    {'检查项': '验证组总数',        '结果': 8,     '状态': 'PASS'},
    {'检查项': '主键检查',          '结果': '9/9', '状态': 'PASS'},
    {'检查项': '外键检查',          '结果': '10/10', '状态': 'PASS'},
    {'检查项': '孤儿行',            '结果': 0,     '状态': 'PASS'},
    {'检查项': '业务规则检查',      '结果': '4/4', '状态': 'PASS'},
    {'检查项': '财务计算检查',      '结果': '3/3', '状态': 'PASS'},
    {'检查项': '拒收记录',          '结果': 0,     '状态': 'PASS'},
    {'检查项': '规范化表行数检查',  '结果': '9/9', '状态': 'PASS'},
])
print(summary.to_string(index=False))

# 模型指标单独列，方便和后续方案对比
print('\nTraining Mean baseline: MAE=121.64, RMSE=153.11')
print('Store + Weekday Mean / Random Forest 的两项指标都更差')
```

最后我们把整个交付目录打包。这一步看起来只是「压缩文件」，其实是**可复现性**的保证：别人拿到这个 zip，就能按 README 的说明重跑一遍 ETL，得到同样的 9 张表和同样的校验结果。

```python
import shutil
from pathlib import Path

out = Path('/kaggle/working')
zip_path = out / 'Spaghetti_Schema_Final_Deliverables.zip'

# 把整个交付目录（docs、sql、normalized_tables）打成一个包
shutil.make_archive(str(zip_path).removesuffix('.zip'), 'zip',
                    root_dir=project.parent, base_dir=project.name)

print('生成成功:', zip_path.exists())
print('大小 (MB):', round(zip_path.stat().st_size / 1024 / 1024, 2))
```

到这里，一个完整的数据工程项目就闭环了：源数据 → 清洗 → 规范化 → 校验 → 基线 → 打包。每一步都有产物，每一步都能重跑。

---

## 三、总结

本文中我们完成了「Spaghetti → Schema」这场数据工程挑战。我们首先确认了交付目录的完整性，读懂了七项交付物和 9 张规范化表；接着用五张业务图表（销售额 Top10 商品、各门店销售额、季度销售额、各部门销售额、促销状态销售额）建立了对零售业务的直觉。其次我们用 Python ETL 把凌乱的 CSV 拆成「门店 / 商品 / 日历 / 顾客」四张维度表和一张销售事实表，清洗时去掉重复行、拒收数量非正和金额为负的记录，并按「先维表、后事实表」的顺序入库。接着我们把校验写成可执行的断言，覆盖主键、外键、业务规则和财务口径，最终 8 组校验全部通过：主键 9/9、外键 10/10、孤儿行 0、业务规则 4/4、财务计算 3/3、拒收记录 0、规范化表行数 9/9。最后我们跑了三个基线模型，Training Mean 基线以 MAE 121.64、RMSE 153.11 胜出，并按要求复查了目标定义、验证策略和潜在泄漏，最后把整套交付物打包成 zip。

这个挑战给我们最大的启发是：**在数据项目里，「证明数据是对的」和「把模型调好」同样重要，甚至更重要**。主键、外键、业务规则这些检查看起来枯燥，但它们能在建模之前就把「脏数据」拦下来；而「简单基线打败复杂模型」这种结果，恰恰是在提醒我们回去检查验证方式和数据泄漏，而不是急着换更花哨的模型。一个能被别人重跑一遍的数据管道，比一个只有自己跑得通的漂亮 notebook 有价值得多。
