# 《百战 Kaggle：机器学习竞赛从零开始实战指南》细纲

> **本书定位**：面向编程语言和深度学习入门者的 Kaggle 竞赛入门书籍。全书不假设读者已有机器学习背景，每一章都从「这是什么」讲起，用大白话解释关键概念，并配套带有详细注释的 Python 代码。关键概念配有 Mermaid 图表，帮助读者建立直观认识。
>
> **内容组织原则**：
> 1. 先按数据集格式把所有 Kaggle 比赛分为四类：表格、时序、图像、文本。
> 2. 第 1 章介绍 Kaggle 比赛的流程和规则。
> 3. 第 2 章介绍通用的探索性数据分析（EDA）。
> 4. 之后分成四个部分，每部分对应一种数据格式。每部分先讲「通用清洗与建模方法与技巧」，再选取若干个代表性比赛，**每个代表比赛独立成章，详解解题过程**。
> 5. 最后介绍模型调优、集成、验证策略、比赛心态，以及从 Kaggle 到实际项目的迁移。
>
> **每一章结构**：x.y 二级章节为最小教学单元；每个二级章节标注是否包含「Python 代码示例」与「Mermaid 图表」，方便后续扩写时直接对齐写作要求。
>
> **内容来源**：参考 `docs/kaggle-writeups/` 下全部 400+ 篇 Kaggle 比赛 Writeup，提炼出贯通各类竞赛的通用流程与常见陷阱。

---

## 前言：为什么要写这本书，以及怎么读

- 什么是 Kaggle，参加比赛能学到什么
- 面向谁：零基础、只会一点 Python、从未参加过竞赛的读者
- 如何阅读：建议章节顺序、代码运行环境、数据获取方式
- 本书配套资源说明：全部代码可在 Kaggle Notebooks 免费运行

---

## 第1章 Kaggle 比赛流程与规则

1.1 什么是 Kaggle，比赛页面长什么样  
- 赛事页面、数据页面、讨论区、排行榜、代码笔记本（Notebooks）的位置与作用  
- 「比赛」与「数据集」的区别  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：Kaggle 比赛页面导览图

1.2 注册账号并加入一场比赛  
- 账号注册、实名认证、接受比赛规则  
- 如何选择适合新手入门的比赛（奖金、截止时间、数据规模、话题熟悉度）  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：无

1.3 理解三份核心文件：训练集、测试集、样例提交  
- train.csv、test.csv、sample_submission.csv 各自包含什么  
- 为什么测试集没有标签，样例提交的格式必须严格匹配  
- 【Python 代码示例】：pandas 读取三份文件并打印形状与列名  
- 【Mermaid 图表】：三份文件的数据流向图

1.4 使用 Kaggle Notebooks：浏览器里直接写代码  
- Notebook 界面、代码单元格与 Markdown 单元格  
- 运行、中断、保存、版本管理的基础操作  
- 【Python 代码示例】：第一个打印 "Hello Kaggle" 的单元格  
- 【Mermaid 图表】：无

1.5 免费 GPU 是什么，什么时候需要开  
- CPU 与 GPU 的直观区别（用画图比喻：顺序计算 vs 并行计算）  
- 图像、文本大模型需要 GPU，表格数据通常 CPU 即可  
- 如何在 Notebook 设置里开启 GPU  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：CPU 与 GPU 计算方式对比图

1.6 把数据下载到本地：Kaggle API 与手动下载  
- 网页端直接下载 ZIP  
- Kaggle API 安装、认证文件配置、`kaggle competitions download` 命令  
- 【Python 代码示例】：用 Python 调用 kaggle API 下载数据（subprocess 调用 CLI）  
- 【Mermaid 图表】：无

1.7 你的第一份提交：哪怕全猜同一个数也能跑通  
- 用训练集目标变量的平均值填满测试集预测列  
- 生成 submission.csv 并上传，查看排行榜排名  
- 目标：跑通「读数据→建模→预测→提交」全链路，而非追求分数  
- 【Python 代码示例】：均值基线提交的完整代码  
- 【Mermaid 图表】：基线提交流程图

1.8 查看排行榜与理解 Public / Private 排名  
- Public 排行榜与 Private 排行榜的区别  
- 为什么排行榜分数会波动，如何看待自己的名次  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：无

1.9 比赛中的禁忌与常见规则  
- 目标泄漏：验证集里不能偷看答案  
- 外部数据的使用限制  
- 提交次数限制与截止时间规则  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：比赛规则红线清单

1.10 小结  
- 本章知识点回顾与下一章预告

---

## 第2章 探索性数据分析（EDA）：先把数据摊开看

2.1 什么是 EDA，为什么建模前必须做 EDA  
- EDA 的直观含义：像侦探勘查现场，先看数据长什么样  
- 跳过 EDA 直接建模会遇到什么麻烦（缺失值爆炸、异常值误导、目标偏态）  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：EDA 在机器学习流程中的位置图

2.2 数值型特征的探索：分布、均值、中位数、偏态  
- 直方图看数值分布，describe 快速查看统计摘要  
- 什么是偏态（右偏、左偏），偏态对模型的影响  
- 异常值的发现：箱线图与 3σ原则  
- 【Python 代码示例】：describe、hist、boxplot 查看房价分布  
- 【Mermaid 图表】：无

2.3 类别型特征的探索：基数、稀有类别、缺失模式  
- 什么是类别型特征，基数（取值种类数）多少才算多  
- 稀有类别会带来什么问题（只出现一两次的类别）  
- 缺失值本身可能就是一种信息（如「没有车库」≠「数据坏了」）  
- 【Python 代码示例】：value_counts 统计类别频次  
- 【Mermaid 图表】：无

2.4 多变量关系与相关性：特征之间和特征与目标的关系  
- 相关系数：两个数值列是否一起变动  
- 散点图看两个数值特征的关系，热力图看相关性矩阵  
- 高度相关的特征可能导致多重共线性  
- 【Python 代码示例】：corr 与 seaborn 热力图  
- 【Mermaid 图表】：特征与目标关系探索图

2.5 目标变量的探索：回归看分布，分类看类别比例  
- 回归任务：目标变量是否偏态，是否需要对数变换  
- 分类任务：各类别样本是否平衡，类别不平衡会带来什么问题  
- 【Python 代码示例】：对数变换前后分布对比，类别比例柱状图  
- 【Mermaid 图表】：目标变量探索决策树

2.6 小结  
- 本章知识点回顾与下一章预告

---

## 第一部分 表格数据：结构化数据的竞赛

表格数据是 Kaggle 中最常见的格式，每一行是一个样本，每一列是一个特征。常见比赛包括房价预测、信用评分、贷款审批、销售预测、点击率预测等。

---

## 第3章 表格数据的通用清洗与建模方法

3.1 缺失值处理：空不等于坏，按语义填充  
- 不同缺失含义：没有泳池 vs 数据丢失  
- 用 'None'、0、中位数、众数、分组中位数等策略填充  
- 何时直接删除含缺失值的行  
- 【Python 代码示例】：按语义填充缺失值  
- 【Mermaid 图表】：缺失值处理决策树

3.2 类别变量编码：让模型读得懂文字  
- 为什么模型只认数字，类别文字必须编码  
- 序号类别（质量等级）用有序映射，非序号类别（街区、城市）用独热编码  
- 独热编码会把一列变成很多列，基数太大时要注意  
- 【Python 代码示例】：LabelEncoder 与 pd.get_dummies  
- 【Mermaid 图表】：类别编码方法选择图

3.3 特征构造：把已有列组合出新列  
- 总面积 = 地下室 + 一楼 + 二楼，总卫浴数，房龄 = 出售年份 - 建成年份  
- 为什么组合特征往往比原始特征更有信息  
- 组合特征的常见模式：加法、乘法、比率、差值  
- 【Python 代码示例】：构造 TotalSF、TotalBath、HouseAge  
- 【Mermaid 图表】：特征组合示意图

3.4 数值变换：对数变换与标准化，让分布更友好  
- 右偏分布（如房价）取 log1p 后更接近正态  
- 标准化让不同量纲的特征在同一量级（如年龄 0~100 vs 收入 0~1000000）  
- 归一化与标准化的区别  
- 【Python 代码示例】：np.log1p 变换目标变量，StandardScaler 标准化特征  
- 【Mermaid 图表】：对数变换前后分布对比图

3.5 表格数据的基线模型：线性回归与岭回归  
- 先跑最简单的模型，建立可比较的基准分数  
- 岭回归的 L2 正则防止多重共线性  
- 【Python 代码示例】：LinearRegression 与 Ridge，5 折交叉验证  
- 【Mermaid 图表】：无

3.6 表格数据的主力模型：随机森林与 XGBoost  
- 决策树的直观原理：不断切分数据，像玩「猜是谁」游戏  
- 随机森林：多棵树投票取平均，减少过拟合  
- XGBoost：梯度提升树，表格数据竞赛的「标配」  
- 树模型对缺失值和异常值的天然耐受性  
- 【Python 代码示例】：RandomForestRegressor 与 XGBRegressor 训练与预测  
- 【Mermaid 图表】：决策树分裂示意图

3.7 模型评估与调参：交叉验证下找不骗人的分数  
- 用交叉验证比较不同模型，避免单次划分的运气成分  
- 调参的基本思路：尝试几组常用参数，看验证集分数  
- 【Python 代码示例】：GridSearchCV 或手动调参 XGBoost  
- 【Mermaid 图表】：无

3.8 生成提交文件与上线前核对清单  
- 预测测试集，按 sample_submission 格式保存 CSV  
- 核对行数、列名、无缺失值、ID 顺序一致  
- 【Python 代码示例】：生成 submission.csv 并断言检查  
- 【Mermaid 图表】：提交前核对清单流程图

3.9 小结  
- 本章知识点回顾与下一章预告

---

## 第4章 表格数据比赛详解：房价预测（House Prices）

4.1 比赛背景与数据  
- Ames 房价数据集介绍，训练集约 1460 条，79 个特征  
- 特征包含数值型与类别型，涉及房屋结构、位置、质量等信息  
- 测试集需预测 SalePrice，提交 Id 与预测价格  
- 【Python 代码示例】：读取 train.csv/test.csv/sample_submission.csv 并查看形状  
- 【Mermaid 图表】：房价数据字段分类图

4.2 探索性数据分析（EDA）  
- SalePrice 呈现右偏，均值被少数高价房拉高  
- 缺失值在 PoolQC、GarageYrBlt、LotFrontage 等列较多  
- 数值特征中 TotalBsmtSF、GrLivArea 与 SalePrice 正相关明显，OverallQual 相关性最高  
- 【Python 代码示例】：describe、corr、散点图查看房价与面积关系  
- 【Mermaid 图表】：无

4.3 数据清洗  
- 缺失值按语义填充：PoolQC、MiscFeature、Alley、Fence、FireplaceQu 填 'None'；GarageYrBlt、GarageArea、GarageCars 填 0  
- LotFrontage 用 Neighborhood 中位数填充，其余类别用众数，数值用中位数  
- 【Python 代码示例】：按语义填充缺失值的完整代码  
- 【Mermaid 图表】：缺失值填充策略图

4.4 特征工程  
- 构造组合特征：TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF；TotalBath = FullBath + 0.5*HalfBath；HouseAge = YrSold - YearBuilt  
- 有序类别映射为数值（ExterQual、KitchenQual 等），其余类别做 LabelEncoder  
- 目标取对数：y_train = np.log1p(SalePrice)  
- 【Python 代码示例】：构造组合特征与对数变换的完整代码  
- 【Mermaid 图表】：特征组合与目标变换图

4.5 模型训练与调优  
- 模型选择 Ridge 回归，L2 正则能防范多重共线  
- 也试过 RandomForest 做对比，树模型在小样本上容易过拟合  
- 5 折交叉验证，评估对数空间 RMSE  
- 【Python 代码示例】：Ridge 回归与 5 折交叉验证代码  
- 【Mermaid 图表】：交叉验证流程图

4.6 模型测试与提交  
- 在对数空间预测，用 np.expm1 还原回原始价格  
- 核对提交行数与测试集一致，列名为 Id 和 SalePrice，无缺失值  
- 生成 submission.csv 并上传  
- 【Python 代码示例】：预测、还原、生成提交文件的完整代码  
- 【Mermaid 图表】：提交前核对清单流程图

4.7 小结  
- 本章知识点回顾：房价预测的完整流程与可迁移模板

---

## 第5章 表格数据比赛详解：信用评分（Give Me Some Credit）

5.1 比赛背景与数据  
- 信用违约预测比赛，预测客户是否会发生违约  
- 数据包含客户特征、信用历史、收入负债等信息  
- 评价指标为 LogLoss，分数越低越好  
- 【Python 代码示例】：读取数据并查看目标变量分布  
- 【Mermaid 图表】：信用评分数据流向图

5.2 探索性数据分析（EDA）  
- 类别不平衡：违约样本较少，非违约样本较多  
- 数值型特征存在缺失值，如月收入、负债数等  
- 部分特征分布偏态，需要变换  
- 【Python 代码示例】：value_counts 查看类别比例，hist 查看数值分布  
- 【Mermaid 图表】：无

5.3 数据清洗  
- 缺失值填充：数值型用中位数，类别型用众数或 'Missing'  
- 异常值处理：极端负债、收入等异常值的截断或转换  
- 【Python 代码示例】：缺失值填充与异常值处理代码  
- 【Mermaid 图表】：缺失值处理决策树

5.4 特征工程  
- 类别变量编码：独热编码或目标编码  
- 构造比率特征：负债收入比、还款负担比等  
- 对偏态特征做对数变换  
- 【Python 代码示例】：构造比率特征与对数变换代码  
- 【Mermaid 图表】：特征构造示意图

5.5 模型训练与调优  
- 基线模型：逻辑回归  
- 主力模型：XGBoost 二分类，处理类别不平衡（scale_pos_weight 或采样）  
- 交叉验证评估 LogLoss  
- 【Python 代码示例】：XGBClassifier 与交叉验证代码  
- 【Mermaid 图表】：无

5.6 模型测试与提交  
- 输出违约概率（0~1 之间），而非 0/1 标签  
- 核对预测概率分布，避免出现大量极端值（0 或 1）  
- 生成 submission.csv 并上传  
- 【Python 代码示例】：概率预测、分布检查、生成提交文件代码  
- 【Mermaid 图表】：提交流程图

5.7 小结  
- 本章知识点回顾：信用评分二分类的完整流程与可迁移模板

---

## 第6章 表格数据比赛详解：贷款审批（Loan Approval）

6.1 比赛背景与数据  
- 贷款是否批准预测比赛  
- 数据包含申请人特征、贷款信息、信用历史等  
- 评价指标为准确率与 F1 分数  
- 【Python 代码示例】：读取数据并查看特征类型  
- 【Mermaid 图表】：贷款审批数据流向图

6.2 探索性数据分析（EDA）  
- 数值型与类别型特征混杂，存在缺失值  
- 目标变量（贷款是否批准）的类别比例  
- 各特征与目标的初步相关性  
- 【Python 代码示例】：dtypes 查看特征类型，value_counts 查看目标分布  
- 【Mermaid 图表】：无

6.3 数据清洗  
- 数值型缺失值用中位数填充，类别型缺失值用众数填充  
- 异常值检测与处理  
- 【Python 代码示例】：缺失值填充代码  
- 【Mermaid 图表】：缺失值处理决策树

6.4 特征工程  
- 类别变量编码：LabelEncoder 或独热编码  
- 构造组合特征：如总收入、贷款收入比  
- 【Python 代码示例】：类别编码与特征构造代码  
- 【Mermaid 图表】：特征构造示意图

6.5 模型训练与调优  
- 随机森林与 XGBoost 对比  
- 交叉验证评估准确率与 F1 分数  
- 调参：树深度、学习率、估计器数量  
- 【Python 代码示例】：RandomForestClassifier 与 XGBClassifier 训练代码  
- 【Mermaid 图表】：无

6.6 模型测试与提交  
- 预测测试集，生成 submission.csv  
- 核对行数、列名、无缺失值  
- 【Python 代码示例】：预测与生成提交文件代码  
- 【Mermaid 图表】：提交流程图

6.7 小结  
- 本章知识点回顾：贷款审批的完整流程与可迁移模板

---

## 第7章 表格数据比赛详解：推土机拍卖价格预测（Blue Book for Bulldozers）

7.1 比赛背景与数据  
- 推土机拍卖价格预测，回归任务  
- 数据包含设备规格、使用情况、拍卖信息等  
- 评价指标为 RMSPE（均方根百分比误差）  
- 【Python 代码示例】：读取数据并查看目标变量分布  
- 【Mermaid 图表】：推土机价格预测数据流向图

7.2 探索性数据分析（EDA）  
- 价格目标变量右偏严重，需要对数变换  
- 类别型特征基数较大（如型号、州别）  
- 日期型特征（如拍卖日期）可拆分  
- 【Python 代码示例】：describe、hist 查看价格分布  
- 【Mermaid 图表】：无

7.3 数据清洗  
- 缺失值处理：数值型填充，类别型填充  
- 日期特征拆分：年、月、日、星期几  
- 【Python 代码示例】：日期拆分与缺失值填充代码  
- 【Mermaid 图表】：日期特征拆分图

7.4 特征工程  
- 大基数类别变量处理：目标编码或频率编码  
- 构造设备年龄、使用年限等特征  
- 目标变量对数变换  
- 【Python 代码示例】：目标编码与特征构造代码  
- 【Mermaid 图表】：特征构造示意图

7.5 模型训练与调优  
- XGBoost 回归为主力模型  
- RMSPE 自定义评估函数  
- 时间切分验证（按拍卖日期顺序切分验证集）  
- 【Python 代码示例】：XGBRegressor 与 RMSPE 评估代码  
- 【Mermaid 图表】：时间切分验证图

7.6 模型测试与提交  
- 预测后用 expm1 还原价格  
- 生成 submission.csv 并上传  
- 【Python 代码示例】：预测、还原、生成提交文件代码  
- 【Mermaid 图表】：提交流程图

7.7 小结  
- 本章知识点回顾：推土机价格预测的完整流程与可迁移模板

---

## 第二部分 时序数据：带时间顺序的竞赛

时序数据的每一行都关联一个时间点，样本之间有先后顺序，不能随便打乱。常见比赛包括门店销售预测、交通流量预测、电力负荷预测、股票价格预测等。

---

## 第8章 时序数据的通用清洗与建模方法

8.1 日期特征拆分：把日期拆成模型能用的数字  
- 把日期列拆成年、月、日、星期几、是否周末、是否节假日  
- 让模型学习星期、月份等周期性规律  
- 【Python 代码示例】：pd.to_datetime 与 dt.day/month/year/weekday  
- 【Mermaid 图表】：无

8.2 滞后特征与滚动统计：用过去的值预测未来  
- 过去几天的销量作为今天的特征（滞后特征，shift）  
- 滚动均值、滚动标准差捕捉近期趋势与波动  
- 滞后特征的阶数怎么选（滞后1天、7天、30天）  
- 【Python 代码示例】：shift 与 rolling 构造滞后特征  
- 【Mermaid 图表】：滞后特征构建示意图

8.3 防止未来信息泄漏：按时间切分验证集  
- 不能随机划分验证集，必须按时间顺序切（训练在前，验证在后）  
- 泄漏会导致本地分虚高、排行榜崩盘  
- TimeSeriesSplit 按时间顺序切分  
- 【Python 代码示例】：按日期切分训练集与验证集  
- 【Mermaid 图表】：时间序列验证集划分图

8.4 时序数据的目标变换：对数变换与差分  
- 销售额等目标常右偏，取 log1p 让分布更对称  
- 差分（今天减昨天）去除趋势，平稳序列更容易建模  
- 【Python 代码示例】：np.log1p 与 df.diff()  
- 【Mermaid 图表】：无

8.5 时序数据的主力模型：XGBoost 回归与 LightGBM  
- 为什么树模型在时序竞赛中依然常用（能处理类别特征、不要求平稳性）  
- 自回归思路：用过去的目标值作为特征  
- 【Python 代码示例】：XGBRegressor 与滞后特征结合  
- 【Mermaid 图表】：时序回归特征构建图

8.6 小结  
- 本章知识点回顾与下一章预告

---

## 第9章 时序数据比赛详解：门店销售预测（Rossmann Store Sales）

9.1 比赛背景与数据  
- Rossmann 门店销售预测，1115 家门店 2006~2014 年日销售额  
- 测试期预测未来六周销售额  
- 评价指标为 RMSPE  
- 【Python 代码示例】：读取 train.csv/test.csv/store.csv 并合并  
- 【Mermaid 图表】：门店销售数据流向图

9.2 探索性数据分析（EDA）  
- Sales 均值约 5773，存在 0 销量（关店日）  
- 明显的周周期和年度节假日波动  
- 促销 Promo 占约 38% 天数，学校假期占 18%  
- Sales 与 Customers 高度相关  
- 【Python 代码示例】：describe、groupby 查看促销与销售关系  
- 【Mermaid 图表】：无

9.3 数据清洗  
- 关店日 Sales=0 处理：只训练营业日，或将 Open 作为特征  
- CompetitionDistance 中 0 值替换为 NaN，让 XGBoost 自己处理缺失  
- 日期拆分为年、月、日  
- 【Python 代码示例】：关店日处理、缺失值替换、日期拆分代码  
- 【Mermaid 图表】：数据清洗流程图

9.4 特征工程  
- PromoInterval 字符串拆成月份标志  
- 竞争开业时间与促销开始时间转换为整数日期  
- 构造门店静态特征与日期动态特征  
- 【Python 代码示例】：PromoInterval 拆分、日期转换代码  
- 【Mermaid 图表】：特征工程流程图

9.5 模型训练与调优  
- 目标取对数：y = np.log1p(Sales)  
- XGBoost 回归，自定义 RMSPE 评估函数  
- 按时间切分验证集，防止数据泄漏  
- 超参数：n_estimators=1000, max_depth=8, learning_rate=0.05 等  
- 【Python 代码示例】：XGBoost 回归与 RMSPE 评估代码  
- 【Mermaid 图表】：无

9.6 模型测试与提交  
- 预测后用 np.expm1 还原销售额  
- 5 折时间序列交叉验证取平均，提升稳定性  
- 生成 submission.csv 并上传  
- 【Python 代码示例】：预测、还原、OOF 预测、生成提交文件代码  
- 【Mermaid 图表】：提交流程图

9.7 小结  
- 本章知识点回顾：门店销售预测的完整流程与可迁移模板

---

## 第10章 时序数据比赛详解：交通流量预测（TrafficFlowBench）

10.1 比赛背景与数据  
- 交通流量时间序列预测比赛  
- 数据包含路段、时间、流量等信息  
- 评价指标为常见回归指标（如 RMSE/MAE）  
- 【Python 代码示例】：读取数据并查看时间范围  
- 【Mermaid 图表】：交通流量数据流向图

10.2 探索性数据分析（EDA）  
- 早晚高峰、周末与工作日差异、节假日效应  
- 不同路段流量分布差异  
- 时间序列的平稳性初步观察  
- 【Python 代码示例】：按小时/星期分组查看流量模式  
- 【Mermaid 图表】：无

10.3 数据清洗  
- 缺失值处理：时间序列插值或前向填充  
- 异常值检测：突然的流量尖峰或断点  
- 【Python 代码示例】：插值填充与异常值检测代码  
- 【Mermaid 图表】：数据清洗流程图

10.4 特征工程  
- 时间特征：小时、星期几、是否工作日、是否节假日  
- 滞后特征：过去 1 小时、1 天、1 周的流量  
- 滚动统计：过去 3 小时、24 小时的平均流量  
- 【Python 代码示例】：时间特征、滞后特征、滚动统计代码  
- 【Mermaid 图表】：特征构造示意图

10.5 模型训练与调优  
- XGBoost/LightGBM 回归为主力模型  
- 按时间切分验证集，防止未来信息泄漏  
- 【Python 代码示例】：树模型回归与时间切分验证代码  
- 【Mermaid 图表】：无

10.6 模型测试与提交  
- 预测测试集，生成 submission.csv  
- 核对提交格式与行数  
- 【Python 代码示例】：预测与生成提交文件代码  
- 【Mermaid 图表】：提交流程图

10.7 小结  
- 本章知识点回顾：交通流量预测的完整流程与可迁移模板

---

## 第11章 时序数据比赛详解：电力负荷预测

11.1 比赛背景与数据  
- 电力需求预测比赛  
- 数据包含时间、气温、负荷等信息  
- 评价指标为回归指标  
- 【Python 代码示例】：读取数据并查看特征分布  
- 【Mermaid 图表】：电力负荷数据流向图

11.2 探索性数据分析（EDA）  
- 季节性：夏季制冷、冬季取暖导致负荷高峰  
- 气温与负荷的关系：极端高温低温负荷都高  
- 周周期与日周期模式  
- 【Python 代码示例】：散点图查看气温与负荷关系  
- 【Mermaid 图表】：无

11.3 数据清洗  
- 缺失值处理：气温、负荷缺失值填充  
- 异常值处理：传感器故障导致的异常读数  
- 【Python 代码示例】：缺失值填充与异常值处理代码  
- 【Mermaid 图表】：数据清洗流程图

11.4 特征工程  
- 日期特征：月份、星期几、小时  
- 气温交叉特征：气温与季节的交互  
- 滞后特征：过去几小时的负荷  
- 【Python 代码示例】：日期拆分、气温交叉特征、滞后特征代码  
- 【Mermaid 图表】：特征构造示意图

11.5 模型训练与调优  
- XGBoost 回归为主力模型  
- 按时间切分验证集  
- 【Python 代码示例】：XGBoost 回归与验证代码  
- 【Mermaid 图表】：无

11.6 模型测试与提交  
- 预测测试集，生成 submission.csv  
- 核对提交格式  
- 【Python 代码示例】：预测与生成提交文件代码  
- 【Mermaid 图表】：提交流程图

11.7 小结  
- 本章知识点回顾：电力负荷预测的完整流程与可迁移模板

---

## 第三部分 图像数据：像素里的竞赛

图像数据的每一个样本是一张图片，由像素值组成。常见比赛包括猫狗分类、病灶检测、卫星图像分类、物体检测等。

---

## 第12章 图像数据的通用清洗与建模方法

12.1 图像在计算机里怎么表示：像素、通道、尺寸  
- 彩色图像 = 三个通道（R、G、B）的数值矩阵  
- 尺寸不一的图片不能直接拼成一个批次，必须统一缩放  
- 【Python 代码示例】：读取图片并打印 shape  
- 【Mermaid 图表】：图像像素矩阵示意图

12.2 读取与预处理图片：缩放、归一化、批次送入  
- 缩放到统一尺寸（如 200×200），像素值除以 255 归一化到 0~1  
- ImageDataGenerator 按批次读图，不占满内存  
- 【Python 代码示例】：Keras ImageDataGenerator 读取与预处理  
- 【Mermaid 图表】：图像预处理流水线

12.3 卷积神经网络直观理解：滤波器、池化、全连接层  
- 卷积层：用小滤波器扫描图像，提取边缘、纹理、部件  
- 池化层：缩小尺寸，保留重要信息  
- 全连接层：把提取的特征映射到类别概率  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：CNN 层结构示意图（卷积→池化→全连接→输出）

12.4 从零训练小型 CNN：猫狗大战示例  
- 搭建四组「卷积+池化」的小型网络，输出 sigmoid 概率  
- 二分类用 binary_crossentropy，优化器 Adam  
- 【Python 代码示例】：Keras Sequential 搭建小型 CNN 并训练  
- 【Mermaid 图表】：无

12.5 过拟合与数据增强：让图片自己变出更多样本  
- 训练准确率高、验证准确率低 = 过拟合  
- 随机旋转、平移、缩放、翻转，让同一张图每次看到略有不同的版本  
- Dropout 随机关掉部分神经元，防止记忆  
- 【Python 代码示例】：ImageDataGenerator 数据增强 + Dropout 层  
- 【Mermaid 图表】：数据增强变换示意图

12.6 迁移学习：借用 ImageNet 上已经学会的「眼睛」  
- 用在大量图像上预训练好的 VGG16/ResNet，只训练最后几层  
- 为什么小数据集上迁移学习效果通常远好于从零训练  
- 【Python 代码示例】：加载预训练 VGG16，冻结卷积层，重训顶层分类器  
- 【Mermaid 图表】：迁移学习迁移路径图

12.7 小结  
- 本章知识点回顾与下一章预告

---

## 第13章 图像数据比赛详解：猫狗大战（Dogs vs. Cats）

13.1 比赛背景与数据  
- 2013 年 Kaggle 经典入门比赛，2.5 万张猫狗照片  
- 训练集猫狗各 12500 张，完全平衡  
- 文件名前三字母即标签（dog/cat），测试集文件名为纯数字  
- 评价指标为 LogLoss，提交「是狗的概率」而非 0/1  
- 【Python 代码示例】：读取 train.zip/test1.zip 并查看文件名结构  
- 【Mermaid 图表】：猫狗数据目录结构图

13.2 探索性数据分析（EDA）  
- 图片尺寸参差不齐，但都是 3 通道彩色图  
- 类别完全平衡，无需处理类别不平衡  
- 背景可能泄露答案（猫多室内，狗多户外），需警惕虚假线索  
- 【Python 代码示例】：遍历图片统计尺寸与通道数  
- 【Mermaid 图表】：无

13.3 数据清洗  
- 统一尺寸：缩放到 200×200（速度与精度的折中）  
- 归一化：像素值除以 255，压到 0~1  
- 划分训练/验证集：从 25000 张里切 20% 做验证  
- 【Python 代码示例】：ImageDataGenerator 读取、缩放、归一化、划分验证集代码  
- 【Mermaid 图表】：图像预处理与划分验证集图

13.4 模型训练：从零训练小型 CNN  
- 四组「卷积+池化」，卷积核数量 32→64→128→128  
- Flatten 后接 512 维全连接，输出 sigmoid 概率  
- 训练 30 轮观察训练与验证准确率  
- 【Python 代码示例】：Keras Sequential 搭建小型 CNN 并训练代码  
- 【Mermaid 图表】：CNN 网络结构图

13.5 解决过拟合：数据增强与 Dropout  
- 训练准确率 0.917，验证准确率 0.863 后开始下滑，出现过拟合  
- 加数据增强（旋转、平移、错切、缩放）和 Dropout(0.2)  
- 验证准确率稳定在 0.81 左右，泛化更稳  
- 【Python 代码示例】：数据增强参数与 Dropout 层代码  
- 【Mermaid 图表】：数据增强变换示意图

13.6 迁移学习：用 VGG16 提升分数  
- 加载 ImageNet 上预训练好的 VGG16，冻结卷积层，只重训顶层  
- 验证准确率普遍能达到 95% 以上，远好于从零训练  
- 【Python 代码示例】：加载预训练 VGG16、冻结层、重训顶层代码  
- 【Mermaid 图表】：迁移学习路径图

13.7 模型测试与提交  
- 对测试集预测「是狗」的概率，shuffle=False 保持文件顺序  
- 生成 submission.csv，列名为 id 和 label  
- 【Python 代码示例】：测试集预测、生成提交文件代码  
- 【Mermaid 图表】：提交流程图

13.8 小结  
- 本章知识点回顾：猫狗分类的完整流程与可迁移模板

---

## 第14章 图像数据比赛详解：糖尿病视网膜病变检测（Diabetic Retinopathy）

14.1 比赛背景与数据  
- 眼底图像多分类比赛，预测视网膜病变严重程度等级  
- 图像数量大，类别存在不平衡  
- 评价指标为 Kappa 或 LogLoss  
- 【Python 代码示例】：读取图像与标签，查看类别分布  
- 【Mermaid 图表】：医学图像数据流向图

14.2 探索性数据分析（EDA）  
- 类别不平衡：轻度病变样本多，重度样本少  
- 图像质量参差不齐，存在模糊、曝光不足等问题  
- 不同等级图像的视觉特征差异  
- 【Python 代码示例】：value_counts 查看类别比例，展示样本图像  
- 【Mermaid 图表】：无

14.3 数据清洗  
- 图像缩放到统一尺寸（如 224×224 或 512×512）  
- 像素归一化到 0~1  
- 质量较差图像的过滤或标记  
- 【Python 代码示例】：图像缩放与归一化代码  
- 【Mermaid 图表】：图像预处理流水线

14.4 特征工程与建模  
- 迁移学习：ResNet/EfficientNet 等预训练模型  
- 冻结底层卷积层，重训顶层分类器  
- 类别不平衡处理：类别权重或过采样  
- 【Python 代码示例】：加载预训练模型、冻结层、设置类别权重代码  
- 【Mermaid 图表】：迁移学习与类别权重图

14.5 模型训练与调优  
- 数据增强：旋转、翻转、缩放，扩充少数类样本  
- 学习率调整、早停机制  
- 多模型集成提升稳定性  
- 【Python 代码示例】：数据增强、早停、集成代码  
- 【Mermaid 图表】：无

14.6 模型测试与提交  
- 输出每个等级的概率，生成 submission.csv  
- 核对提交行数与列名  
- 【Python 代码示例】：概率预测与生成提交文件代码  
- 【Mermaid 图表】：提交流程图

14.7 小结  
- 本章知识点回顾：医学图像多分类的完整流程与可迁移模板

---

## 第15章 图像数据比赛详解：卫星图像分类（Planet / Carvana）

15.1 比赛背景与数据  
- 从太空理解亚马逊：卫星图像多标签分类，或 Carvana 图像抠图  
- 大图分块处理，标签稀少  
- 评价指标为 F2 分数（多标签）或 IoU（分割）  
- 【Python 代码示例】：读取大图并分块  
- 【Mermaid 图表】：卫星图像数据流向图

15.2 探索性数据分析（EDA）  
- 图像分辨率极大，不能直接全部送入模型  
- 标签稀少：多标签分类中，某些标签样本极少  
- 图像内容：森林、农地、水体、道路等  
- 【Python 代码示例】：查看图像尺寸与标签分布  
- 【Mermaid 图表】：无

15.3 数据清洗  
- 大图切分为小块（tiles），便于模型处理  
- 像素归一化  
- 多标签数据的格式统一  
- 【Python 代码示例】：图像分块代码  
- 【Mermaid 图表】：大图分块示意图

15.4 特征工程与建模  
- 迁移学习：Inception/ResNet 预训练模型  
- 多标签分类：输出多个 sigmoid 概率  
- 图像分割（如 Carvana）：U-Net 架构  
- 【Python 代码示例】：多标签分类模型或 U-Net 代码  
- 【Mermaid 图表】：多标签/分割模型结构图

15.5 模型训练与调优  
- 数据增强：旋转、翻转、随机裁剪  
- 学习率调度、早停  
- 测试时增强（TTA）提升稳定性  
- 【Python 代码示例】：数据增强与 TTA 代码  
- 【Mermaid 图表】：无

15.6 模型测试与提交  
- 生成预测标签或分割掩码，保存为指定格式  
- 生成 submission.csv 并上传  
- 【Python 代码示例】：预测与生成提交文件代码  
- 【Mermaid 图表】：提交流程图

15.7 小结  
- 本章知识点回顾：卫星图像/物体检测的完整流程与可迁移模板

---

## 第四部分 文本数据：文字里的竞赛

文本数据的每一个样本是一段文字，需要先转成数字才能送进模型。常见比赛包括情感分析、问题对去重、文本分类、关键词抽取等。

---

## 第16章 文本数据的通用清洗与建模方法

16.1 文本数据的特点：短、吵、不结构化  
- 文本不能直接喂给模型，必须先变成数字  
- 停用词、拼写错误、网络用语等噪声  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：文本转数字的流程图

16.2 文本清洗：小写、去标点、去停用词  
- 统一小写，去掉标点符号，移除 "the/is/how" 这类无区分度的词  
- 【Python 代码示例】：用 NLTK 做英文分词与去停用词  
- 【Mermaid 图表】：无

16.3 词袋模型：把句子变成数字向量  
- 每个词是一个维度，值为该词出现的次数  
- CountVectorizer 的直观用法  
- 【Python 代码示例】：CountVectorizer 将两句话转成向量  
- 【Mermaid 图表】：词袋向量构建示意图

16.4 TF-IDF：让稀有词更有发言权  
- 常见词权重降低，稀有且有区分度的词权重提高  
- TF-IDF 直观解释：一个词在当前文档重要，但在所有文档中罕见  
- 【Python 代码示例】：TfidfVectorizer 计算文本特征  
- 【Mermaid 图表】：TF-IDF 权重计算示意图

16.5 文本相似度特征：两段文字是不是在说同一件事  
- 词重合比例、共同词数、Jaccard 相似度  
- 以「问题对」任务为例，把两个问题转成集合再比较  
- 【Python 代码示例】：计算两个问题的词重合比例  
- 【Mermaid 图表】：问题对特征提取图

16.6 浅层文本模型：逻辑回归、朴素贝叶斯、XGBoost  
- 用 TF-IDF 向量喂给传统机器学习模型  
- 为什么在中小规模文本竞赛中，TF-IDF + 线模型往往够用  
- 【Python 代码示例】：Tfidf + LogisticRegression / XGBoost  
- 【Mermaid 图表】：无

16.7 小结  
- 本章知识点回顾与下一章预告

---

## 第17章 文本数据比赛详解：Quora 问题对去重

17.1 比赛背景与数据  
- Quora 问题对去重，40 万问题对，判断两个问题是否重复  
- 训练集正例 36.92%，测试集正例仅约 16.5%，分布不一致  
- 评价指标为 LogLoss  
- 【Python 代码示例】：读取 train.csv/test.csv 并查看形状  
- 【Mermaid 图表】：问题对数据流向图

17.2 探索性数据分析（EDA）  
- 训练集 404,290 对，测试集 2,345,796 对（测试集远大于训练集）  
- 537,933 个不同问题，111,780 个问题出现多次，最多出现 160 次以上  
- 平均每条问题 59.82 个字符、11.06 个词，属于短文本  
- 含问号 99.87%、首字母大写 99.81%，这两项区分度低  
- 【Python 代码示例】：统计问题数量、字符数、词数、符号比例代码  
- 【Mermaid 图表】：无

17.3 数据清洗  
- 统一小写、去掉停用词、把问题转成词集合  
- 处理全停用词导致的空集合（返回 0 避免除零）  
- 给测试集新问题分配连续编号（训练集 qid 最大值后继续编号）  
- 【Python 代码示例】：文本清洗、转词集合、分配 qid 代码  
- 【Mermaid 图表】：文本清洗流程图

17.4 特征工程  
- 词重合比例：交集大小 / 总词数，AUC 约 0.7804  
- TF-IDF 加权重合比例：稀有词权重更高，单独看 AUC 约 0.7705  
- 共同邻居数（魔法特征）：把问题对看成图上的边，统计两个问题的邻居交集，带来约 0.045 提升  
- 【Python 代码示例】：词重合、TF-IDF、共同邻居特征代码  
- 【Mermaid 图表】：问题对共同邻居图示意图

17.5 模型训练与调优  
- 负例过采样，将正例率从 36.92% 调整到约 19.1%，对齐测试集分布  
- XGBoost：eta=0.02, max_depth=4，400 轮，早停 50 轮  
- 验证 LogLoss 约 0.3639，公开榜约 0.3546  
- 【Python 代码示例】：重采样、XGBoost 训练与早停代码  
- 【Mermaid 图表】：无

17.6 模型测试与提交  
- 输出重复概率（0~1 之间），核对预测均值接近 0.165  
- 检查提交行数与测试集一致，无极端 0/1 值  
- 生成 submission.csv 并上传  
- 【Python 代码示例】：概率预测、分布检查、生成提交文件代码  
- 【Mermaid 图表】：提交流程图

17.7 小结  
- 本章知识点回顾：文本相似度比赛的完整流程与可迁移模板

---

## 第18章 文本数据比赛详解：电影评论情感分析（IMDb）

18.1 比赛背景与数据  
- 电影评论情感分析，二分类（正面/负面）  
- 评价指标为准确率或 LogLoss  
- 【Python 代码示例】：读取评论文本与标签  
- 【Mermaid 图表】：情感分析数据流向图

18.2 探索性数据分析（EDA）  
- 评论长度分布：大多数评论在多少词以内  
- 情感词分布：正面与负面评论的高频词差异  
- 类别比例：正面与负面是否平衡  
- 【Python 代码示例】：评论长度统计、高频词统计代码  
- 【Mermaid 图表】：无

18.3 数据清洗  
- 文本清洗：小写、去标点、去停用词  
- 处理 HTML 标签、特殊字符  
- 【Python 代码示例】：文本清洗代码  
- 【Mermaid 图表】：文本清洗流程图

18.4 特征工程  
- 词袋模型或 TF-IDF 向量化  
- n-gram 特征：捕捉短语（如 "not good"）  
- 【Python 代码示例】：CountVectorizer/TfidfVectorizer 与 n-gram 代码  
- 【Mermaid 图表】：特征向量化示意图

18.5 模型训练与调优  
- 逻辑回归或 XGBoost 作为基线模型  
- 交叉验证评估准确率或 LogLoss  
- 【Python 代码示例】：逻辑回归与 XGBoost 训练代码  
- 【Mermaid 图表】：无

18.6 模型测试与提交  
- 输出正面概率，生成 submission.csv  
- 核对提交格式与行数  
- 【Python 代码示例】：概率预测与生成提交文件代码  
- 【Mermaid 图表】：提交流程图

18.7 小结  
- 本章知识点回顾：情感分析的完整流程与可迁移模板

---

## 第19章 文本数据比赛详解：文本分类与关键词抽取（Tradeshift / Facebook Recruiting III）

19.1 比赛背景与数据  
- 文本多类分类或关键词抽取比赛  
- 文档长度、类别分布、稀有词特征  
- 评价指标为分类指标  
- 【Python 代码示例】：读取文本数据并查看类别分布  
- 【Mermaid 图表】：文本分类数据流向图

19.2 探索性数据分析（EDA）  
- 文档长度分布：短文本还是长文档  
- 类别分布：哪些类别样本多，哪些稀少  
- 高频词与稀有词分析  
- 【Python 代码示例】：文档长度统计、类别比例统计代码  
- 【Mermaid 图表】：无

19.3 数据清洗  
- 文本清洗：小写、去标点、去停用词  
- 处理稀有类别：合并为 "Other" 或单独处理  
- 【Python 代码示例】：文本清洗与稀有类别处理代码  
- 【Mermaid 图表】：文本清洗流程图

19.4 特征工程  
- TF-IDF 向量化  
- n-gram 特征捕捉词组  
- 关键词抽取特征（如 TF-IDF 权重高的词）  
- 【Python 代码示例】：TF-IDF 与 n-gram 特征代码  
- 【Mermaid 图表】：特征构造示意图

19.5 模型训练与调优  
- 线性 SVM、逻辑回归或 XGBoost  
- 多类分类：一对其余（OvR）或 softmax  
- 交叉验证评估  
- 【Python 代码示例】：线性 SVM 或 XGBoost 多类分类代码  
- 【Mermaid 图表】：无

19.6 模型测试与提交  
- 输出类别概率，生成 submission.csv  
- 核对提交格式与行数  
- 【Python 代码示例】：概率预测与生成提交文件代码  
- 【Mermaid 图表】：提交流程图

19.7 小结  
- 本章知识点回顾：文本分类的完整流程与可迁移模板

---

## 第20章 模型评估与验证：先看得分准不准

20.1 评估指标选型：回归看 RMSE/MAE，分类看准确率/AUC/LogLoss  
- 回归任务：RMSE 惩罚大误差更重，MAE 更稳健，RMSPE 适合相对误差敏感的场景  
- 分类任务：准确率直观但类别不平衡时不可靠；AUC 看排序能力，LogLoss 看概率校准  
- 每个指标越大越好还是越小越好，要心里有数  
- 【Python 代码示例】：sklearn 一键计算 RMSE、MAE、AUC、LogLoss  
- 【Mermaid 图表】：指标选择决策树

20.2 交叉验证：把数据切成几份轮流当考卷  
- K 折交叉验证的直观原理：每一份都当一次验证集，平均得分更稳  
- 分层抽样保持各类别比例一致，避免某一折全是负例  
- 为什么单次划分可能运气好/坏，交叉验证更稳  
- 【Python 代码示例】：KFold 与 StratifiedKFold 使用  
- 【Mermaid 图表】：K 折交叉验证流程图

20.3 时间序列验证：按时间顺序切，严禁未来预测过去  
- 时序数据不能随机划分验证集，必须按时间顺序切（训练在前，验证在后）  
- TimeSeriesSplit 按时间顺序切分  
- 泄漏会导致本地分虚高、排行榜崩盘  
- 【Python 代码示例】：TimeSeriesSplit 与按日期切分验证集  
- 【Mermaid 图表】：时间序列验证集划分图

20.4 目标泄漏检查：验证集里不能偷看答案  
- 什么是目标泄漏：特征里包含了只在预测时刻才能知道的信息  
- 典型例子：用未来的销量预测今天的销量，用测试集本身计算特征  
- 如何自查：检查特征的时间戳，看是否包含未来信息  
- 【Python 代码示例】：检查可能的泄漏特征  
- 【Mermaid 图表】：目标泄漏示意图

20.5 为什么本地分和排行榜分经常不一样  
- 验证集构造不同、数据分布不同、公测/私测切分不同  
- 本地分只是参考，不迷信单一指标  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：本地验证与排行榜分差异原因图

20.6 小结  
- 本章知识点回顾：先选对指标，再搞定验证方案，最后才去调参和集成

---

## 第21章 模型调优与集成学习：榜单上的最后几名

21.1 超参数是什么：学习率、树深度、正则强度  
- 模型参数由数据学习，超参数由人提前设定  
- 常见超参数及其直观影响  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：超参数对模型复杂度的影响图

21.2 网格搜索与随机搜索：有策略地尝试  
- GridSearchCV 穷举所有组合，RandomizedSearchCV 随机采样  
- 计算代价与效果的权衡  
- 【Python 代码示例】：GridSearchCV 调参 XGBoost  
- 【Mermaid 图表】：网格搜索示意图

21.3 贝叶斯优化：聪明一点的调参方式  
- 根据历史结果预测哪些超参数组合更值得尝试  
- 适合评估代价高昂的模型（如深度学习）  
- 【Python 代码示例】：Optuna 贝叶斯调参简例  
- 【Mermaid 图表】：贝叶斯优化循环图

21.4 集成学习：投票、平均、加权平均取长补短  
- 多个模型的预测取平均，通常比单一模型更稳  
- 硬投票（多数票）与软投票（概率平均）  
- 【Python 代码示例】：VotingRegressor / VotingClassifier  
- 【Mermaid 图表】：集成学习结构图

21.5 Stacking：用模型去预测模型的预测  
- 第一层模型的预测结果作为第二层模型的输入  
- 元学习器（通常是简单的线性模型）做最终组合  
- 【Python 代码示例】：StackingRegressor 简例  
- 【Mermaid 图表】：Stacking 两层结构图

21.6 早停：验证分数不再提升时停止训练  
- 验证集分数连续若干轮不提升就停止，防止过拟合  
- XGBoost early_stopping_rounds 的用法  
- 【Python 代码示例】：XGBoost early_stopping_rounds  
- 【Mermaid 图表】：早停机制示意图

21.7 小结  
- 本章知识点回顾：调参找更好超参数，集成取多模型之长

---

## 第22章 比赛策略与心态

22.1 选比赛：适合新手入手的比赛类型  
- 数据规模适中、指标直观、基线简单、有充足讨论区资料的比赛  
- 避免初次参赛就选需要大规模 GPU 或专业领域知识的比赛  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：选比赛决策流程图

22.2 先跑通基线，再优化：不要一上来就搞复杂模型  
- 用最简单模型跑出第一个提交，再逐步替换为更强模型  
- 每次只改一处，观察分数变化  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：无

22.3 防过拟合：验证集是唯一真神  
- 训练集分数虚高不可信，以验证集分数为准决策  
- 早停（Early Stopping）在验证分数不再提升时停止训练  
- 【Python 代码示例】：XGBoost early_stopping_rounds  
- 【Mermaid 图表】：早停机制示意图

22.4 提交策略：每天提几次，提交什么版本  
- 排行榜提交次数有限，不要浪费在微小改动上  
- 每次重大改进后再提交，记录每个版本做了什么  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：无

22.5 团队合作与代码共享：Kaggle 团队怎么协作  
- 组队规则、共享 Notebook、代码合并  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：无

22.6 从前 10% 到前 1%：细节决定成败  
- 特征工程的细节、验证方案的严谨、集成的权重调整  
- 顶尖解决方案通常是很多小改进的累加  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：无

22.7 小结  
- 本章知识点回顾与下一章预告

---

## 第23章 从 Kaggle 到实际机器学习项目

23.1 比赛数据与真实业务数据的区别  
- 比赛数据通常已清洗、标签完整；真实数据脏、标签缺失、分布漂移  
- 比赛看排行榜分数，业务看可解释性、稳定性与维护成本  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：比赛 vs 业务数据对比图

23.2 把竞赛 Notebook 变成可维护的项目代码  
- 从单文件 Notebook 拆分为数据处理、特征工程、模型训练、预测四个模块  
- 加入日志、配置文件与单元测试  
- 【Python 代码示例】：模块化代码结构示例  
- 【Mermaid 图表】：机器学习项目目录结构图

23.3 模型部署基础：从 Notebook 到简单 API  
- 用 Flask/FastAPI 把模型封装成 HTTP 接口，接收数据返回预测  
- 【Python 代码示例】：FastAPI 预测接口简例  
- 【Mermaid 图表】：模型部署请求响应图

23.4 继续学习的路径与社区资源  
- Kaggle 论坛、代码笔记本、论文、课程推荐  
- 如何阅读顶尖选手的解决方案报告  
- 【Python 代码示例】：无  
- 【Mermaid 图表】：学习路径建议图

23.5 小结  
- 本书全部知识点回顾

---

## 附录 A：NumPy 快速参考

A.1 数组创建  
- 常用类与函数：`np.array`、`np.zeros`、`np.ones`、`np.empty`、`np.full`、`np.arange`、`np.linspace`、`np.eye`、`np.identity`、`np.diag`、`np.random.rand`、`np.random.randn`、`np.random.randint`、`np.random.random`  
- 用大白话说：NumPy 数组就是装数字的“容器”，机器学习里的数据几乎都要先装进这里才能计算。  
- 【Python 代码示例】：`import numpy as np; a = np.array([1, 2, 3]); b = np.zeros((3, 3)); c = np.linspace(0, 10, 5); d = np.random.randn(2, 3)`  
- 【Mermaid 图表】：数组创建方式对比图

A.2 数组属性与索引  
- 常用属性与方法：`shape`、`ndim`、`size`、`dtype`、`itemsize`、`reshape`、`ravel`、`flatten`、`T`、`transpose`  
- 常用索引：切片 `arr[1:5]`、`fancy indexing` `arr[[0, 2, 4]]`、`boolean indexing` `arr[arr > 0]`、`np.where`  
- 用大白话说：属性告诉你数组长什么样，索引告诉你怎么取出想要的那几个数。  
- 【Python 代码示例】：`print(a.shape, a.dtype); print(a[a > 1]); idx = np.where(a > 1); print(a[idx])`  
- 【Mermaid 图表】：数组索引与切片示意图

A.3 数学与统计函数  
- 常用函数：`np.sum`、`np.mean`、`np.std`、`np.var`、`np.min`、`np.max`、`np.argmin`、`np.argmax`、`np.cumsum`、`np.cumprod`、`np.diff`、`np.sort`、`np.argsort`、`np.unique`、`np.median`、`np.percentile`  
- 用大白话说：对数组求和、平均、最大值、排序、去重等常见统计操作。  
- 【Python 代码示例】：`print(np.mean(a), np.std(a)); print(np.argsort(a)); print(np.unique(a))`  
- 【Mermaid 图表】：无

A.4 广播机制与矩阵运算  
- 常用函数与运算符：`np.dot`、`np.matmul`、`@` 运算符、`np.transpose`、`np.diagonal`、`np.trace`、`np.linalg.inv`、`np.linalg.solve`、`np.linalg.eig`、`np.linalg.svd`、`np.linalg.det`、`np.linalg.norm`  
- 用大白话说：广播让不同形状的数组能直接相加相乘；矩阵运算是线性代数的核心，机器学习模型里到处都是。  
- 【Python 代码示例】：`c = a @ b; inv = np.linalg.inv(b); vals, vecs = np.linalg.eig(b)`  
- 【Mermaid 图表】：矩阵乘法与广播示意图

A.5 随机数  
- 常用函数：`np.random.seed`、`np.random.rand`、`np.random.randn`、`np.random.randint`、`np.random.normal`、`np.random.uniform`、`np.random.choice`、`np.random.shuffle`、`np.random.permutation`  
- 用大白话说：训练模型时要打乱数据、初始化权重，都需要随机数；设 seed 可以让结果可复现。  
- 【Python 代码示例】：`np.random.seed(42); idx = np.random.permutation(len(a)); train_idx = idx[:100]`  
- 【Mermaid 图表】：无

A.6 形状操作与合并  
- 常用函数：`np.concatenate`、`np.vstack`、`np.hstack`、`np.stack`、`np.split`、`np.vsplit`、`np.hsplit`、`np.tile`、`np.repeat`、`np.expand_dims`、`np.squeeze`  
- 用大白话说：把多个数组拼在一起，或者把一个数组拆成几块，调整数组的维度形状。  
- 【Python 代码示例】：`c = np.concatenate([a, b], axis=0); d = np.expand_dims(a, axis=1)`  
- 【Mermaid 图表】：数组合并与拆分示意图

A.7 文件读写  
- 常用函数：`np.save`、`np.load`、`np.savez`、`np.savez_compressed`、`np.genfromtxt`、`np.loadtxt`  
- 用大白话说：把数组保存到磁盘，下次直接读出来用，不用重新生成。  
- 【Python 代码示例】：`np.save('data.npy', a); a2 = np.load('data.npy')`  
- 【Mermaid 图表】：无

A.8 小结  
- NumPy 是 pandas、scikit-learn、PyTorch 的底层基础，掌握数组创建、索引、统计、矩阵运算即可应付大部分竞赛场景。

---

## 附录 B：pandas 快速参考

B.1 读写数据  
- 常用函数：`pd.read_csv`、`pd.read_excel`、`pd.read_json`、`pd.read_sql`、`pd.read_parquet`、`pd.read_html`、`df.to_csv`、`df.to_excel`、`df.to_json`、`df.to_parquet`  
- 用大白话说：把外部数据文件读进 pandas，处理完再写回去。  
- 【Python 代码示例】：`df = pd.read_csv('train.csv'); df.to_csv('submission.csv', index=False)`  
- 【Mermaid 图表】：数据读写流程图

B.2 Series 与 DataFrame 创建  
- 常用类与函数：`pd.Series`、`pd.DataFrame`、`pd.Index`、`pd.MultiIndex`  
- 用大白话说：Series 是一列数据，DataFrame 是一张表（多列），Index 是行列索引。  
- 【Python 代码示例】：`s = pd.Series([1, 2, 3]); df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})`  
- 【Mermaid 图表】：Series 与 DataFrame 结构图

B.3 属性与基本操作  
- 常用属性与方法：`shape`、`columns`、`index`、`dtypes`、`head`、`tail`、`info`、`describe`、`unique`、`nunique`、`value_counts`、`isnull`、`notnull`、`fillna`、`dropna`、`drop`、`reset_index`、`set_index`、`rename`、`astype`  
- 用大白话说：查看数据长什么样、有没有缺失、类型是什么，改名字、改类型、删行列。  
- 【Python 代码示例】：`print(df.info()); print(df['col'].value_counts()); df = df.fillna(0)`  
- 【Mermaid 图表】：无

B.4 索引与筛选  
- 常用方法：`loc`、`iloc`、`at`、`iat`、`切片`、`布尔索引`、`query`、`between`、`isin`、`nlargest`、`nsmallest`、`filter`  
- 用大白话说：按行列名或位置取数据，按条件筛选出想要的行。  
- 【Python 代码示例】：`df.loc[df['age'] > 18, ['name', 'salary']]; df.query('score > 90')`  
- 【Mermaid 图表】：loc 与 iloc 区别示意图

B.5 分组与聚合  
- 常用方法：`groupby`、`agg`、`apply`、`transform`、`filter`、`pivot_table`、`crosstab`、`size`、`count`、`sum`、`mean`、`min`、`max`、`std`、`first`、`last`、`nth`  
- 用大白话说：把数据分成几组，对每组分别求平均、求和、求个数，做透视表。  
- 【Python 代码示例】：`df.groupby('city')['price'].mean(); df.pivot_table(values='sales', index='city', columns='month', aggfunc='sum')`  
- 【Mermaid 图表】：groupby 分组聚合流程图

B.6 合并与连接  
- 常用函数：`pd.merge`、`df.join`、`pd.concat`、`df.append`（已弃用，推荐 concat）  
- 用大白话说：像 Excel 的 VLOOKUP 那样，把两张表按某列拼在一起，或者上下堆叠。  
- 【Python 代码示例】：`pd.merge(df1, df2, on='id', how='left'); pd.concat([df1, df2], axis=0)`  
- 【Mermaid 图表】：表连接方式示意图

B.7 字符串操作  
- 常用方法：`df['col'].str.contains`、`str.split`、`str.replace`、`str.startswith`、`str.endswith`、`str.len`、`str.lower`、`str.upper`、`str.strip`、`str.extract`、`str.extractall`  
- 用大白话说：对文本列做查找、替换、切分、大小写转换等操作。  
- 【Python 代码示例】：`df['email'].str.contains('@'); df['name'].str.split(' ', expand=True)`  
- 【Mermaid 图表】：无

B.8 日期与时间  
- 常用函数与方法：`pd.to_datetime`、`pd.date_range`、`pd.Period`、`pd.Timestamp`、`dt.year`、`dt.month`、`dt.day`、`dt.hour`、`dt.weekday`、`dt.dayofweek`、`dt.is_month_end`、`resample`、`shift`、`rolling`、`expanding`、`ewm`  
- 用大白话说：把字符串日期变成 pandas 能识别的日期类型，按时间分组、滚动计算、移动平均。  
- 【Python 代码示例】：`df['date'] = pd.to_datetime(df['date']); df['month'] = df['date'].dt.month; df.rolling(7).mean()`  
- 【Mermaid 图表】：时间序列滚动窗口示意图

B.9 缺失值与重复  
- 常用方法：`isna`、`isnull`、`notna`、`notnull`、`fillna`、`dropna`、`interpolate`、`duplicated`、`drop_duplicates`、`value_counts(dropna=False)`  
- 用大白话说：发现缺失值、填补缺失值、删除缺失值，处理重复行。  
- 【Python 代码示例】：`print(df.isnull().sum()); df = df.drop_duplicates()`  
- 【Mermaid 图表】：缺失值处理决策树

B.10 小结  
- pandas 是表格数据竞赛的主力工具，读取、筛选、分组、合并、缺失值处理这几项掌握即可。

---

## 附录 C：matplotlib 快速参考

C.1 图形基础  
- 常用函数：`plt.figure`、`plt.subplot`、`plt.subplots`、`plt.show`、`plt.close`、`plt.savefig`、`plt.title`、`plt.xlabel`、`plt.ylabel`、`plt.xlim`、`plt.ylim`、`plt.grid`、`plt.legend`、`plt.tight_layout`  
- 用大白话说：创建画布，设置标题、坐标轴标签、范围、网格、图例，最后显示或保存图片。  
- 【Python 代码示例】：`plt.figure(figsize=(8, 6)); plt.plot(x, y); plt.title('图表'); plt.savefig('plot.png')`  
- 【Mermaid 图表】：绘图流程图

C.2 常见图表  
- 常用函数：`plt.plot`、`plt.scatter`、`plt.bar`、`plt.barh`、`plt.hist`、`plt.boxplot`、`plt.pie`、`plt.errorbar`、`plt.fill_between`、`plt.stackplot`、`plt.step`、`plt.violinplot`  
- 用大白话说：折线图看趋势，散点图看关系，柱状图比大小，直方图看分布，箱线图看离散程度。  
- 【Python 代码示例】：`plt.scatter(df['area'], df['price']); plt.hist(df['price'], bins=30)`  
- 【Mermaid 图表】：常见图表选型图

C.3 坐标轴与刻度  
- 常用函数：`plt.xticks`、`plt.yticks`、`plt.tick_params`、`ax.spines`、`plt.twinx`、`plt.twiny`、`plt.annotate`、`plt.text`、`plt.arrow`  
- 用大白话说：调整刻度标签、坐标轴样式，加双坐标轴，在图上标注文字和箭头。  
- 【Python 代码示例】：`plt.xticks(rotation=45); ax2 = plt.twinx(); ax2.plot(x, z, color='red')`  
- 【Mermaid 图表】：无

C.4 子图与布局  
- 常用函数：`plt.subplots`、`plt.GridSpec`、`plt.tight_layout`、`plt.subplots_adjust`、`plt.subplot2grid`  
- 用大白话说：在一个画布里画多个小图，调整它们之间的间距。  
- 【Python 代码示例】：`fig, axes = plt.subplots(2, 2); axes[0, 0].plot(x, y)`  
- 【Mermaid 图表】：子图布局示意图

C.5 样式与颜色  
- 常用参数：`color`、`linestyle`、`marker`、`linewidth`、`markersize`、`alpha`、`cmap`、`plt.colorbar`、`plt.style.use`、`plt.rcParams`  
- 用大白话说：设置线条颜色、样式、点的形状、透明度、颜色条，让图更好看。  
- 【Python 代码示例】：`plt.plot(x, y, color='red', linestyle='--', marker='o', alpha=0.7)`  
- 【Mermaid 图表】：无

C.6 图例与注释  
- 常用函数：`plt.legend`、`plt.annotate`、`plt.text`、`plt.arrow`、`plt.xlabel`、`plt.ylabel`、`plt.title`  
- 用大白话说：加图例说明每条线代表什么，加注释解释图中的关键点。  
- 【Python 代码示例】：`plt.legend(['训练集', '验证集']); plt.annotate('最高点', xy=(x_max, y_max))`  
- 【Mermaid 图表】：无

C.7 保存与显示  
- 常用函数：`plt.savefig`、`plt.show`、`plt.close`  
- 参数：`dpi`、`bbox_inches`、`format`、`transparent`  
- 用大白话说：把图保存成文件，或者在 Notebook 里直接显示，用完记得关闭释放内存。  
- 【Python 代码示例】：`plt.savefig('output.png', dpi=300, bbox_inches='tight'); plt.show(); plt.close()`  
- 【Mermaid 图表】：无

C.8 小结  
- matplotlib 适合快速画出探索性图表，掌握 plot、scatter、hist、bar、subplots、savefig 即可应付大部分需求。

---

## 附录 D：scikit-learn 快速参考

D.1 数据预处理  
- 常用类：`sklearn.preprocessing.StandardScaler`、`MinMaxScaler`、`RobustScaler`、`MaxAbsScaler`、`Normalizer`、`Binarizer`、`OneHotEncoder`、`OrdinalEncoder`、`LabelEncoder`、`SimpleImputer`、`PolynomialFeatures`、`KBinsDiscretizer`、`FunctionTransformer`、`PowerTransformer`、`QuantileTransformer`  
- 用大白话说：把原始数据变成模型能吃的格式：缺失值填补、数值缩放、类别编码、多项式特征。  
- 【Python 代码示例】：`from sklearn.preprocessing import StandardScaler; scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)`  
- 【Mermaid 图表】：预处理流程图

D.2 特征选择  
- 常用类：`sklearn.feature_selection.SelectKBest`、`SelectPercentile`、`RFE`、`RFECV`、`VarianceThreshold`、`SelectFromModel`  
- 用大白话说：从很多特征里挑出最有用的几个，减少噪声和过拟合。  
- 【Python 代码示例】：`from sklearn.feature_selection import SelectKBest, f_classif; selector = SelectKBest(f_classif, k=10); X_new = selector.fit_transform(X, y)`  
- 【Mermaid 图表】：无

D.3 模型选择与交叉验证  
- 常用函数与类：`sklearn.model_selection.train_test_split`、`KFold`、`StratifiedKFold`、`TimeSeriesSplit`、`GroupKFold`、`cross_val_score`、`cross_val_predict`、`cross_validate`、`GridSearchCV`、`RandomizedSearchCV`、`learning_curve`、`validation_curve`  
- 用大白话说：把数据分成训练和验证，交叉验证更稳，网格搜索帮你找最好的超参数。  
- 【Python 代码示例】：`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2); cv = cross_val_score(model, X, y, cv=5)`  
- 【Mermaid 图表】：交叉验证流程图

D.4 回归模型  
- 常用类：`sklearn.linear_model.LinearRegression`、`Ridge`、`Lasso`、`ElasticNet`、`SGDRegressor`、`sklearn.tree.DecisionTreeRegressor`、`sklearn.ensemble.RandomForestRegressor`、`GradientBoostingRegressor`、`sklearn.svm.SVR`、`sklearn.neighbors.KNeighborsRegressor`  
- 用大白话说：预测连续数值（房价、销量）的模型，线性回归最简单，树模型和 XGBoost 效果通常更好。  
- 【Python 代码示例】：`from sklearn.linear_model import Ridge; model = Ridge(alpha=1.0); model.fit(X_train, y_train)`  
- 【Mermaid 图表】：无

D.5 分类模型  
- 常用类：`sklearn.linear_model.LogisticRegression`、`SGDClassifier`、`sklearn.tree.DecisionTreeClassifier`、`RandomForestClassifier`、`GradientBoostingClassifier`、`sklearn.svm.SVC`、`LinearSVC`、`KNeighborsClassifier`、`sklearn.naive_bayes.GaussianNB`、`MultinomialNB`、`BernoulliNB`  
- 用大白话说：预测类别（是否点击、是否违约）的模型，逻辑回归是基线，随机森林和梯度提升常用。  
- 【Python 代码示例】：`from sklearn.ensemble import RandomForestClassifier; clf = RandomForestClassifier(n_estimators=100); clf.fit(X_train, y_train)`  
- 【Mermaid 图表】：无

D.6 集成模型  
- 常用类：`sklearn.ensemble.VotingClassifier`、`VotingRegressor`、`StackingClassifier`、`StackingRegressor`、`BaggingClassifier`、`BaggingRegressor`、`AdaBoostClassifier`、`AdaBoostRegressor`  
- 用大白话说：多个模型合在一起投票或平均，通常比单个模型更稳、分数更高。  
- 【Python 代码示例】：`from sklearn.ensemble import VotingClassifier; vc = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')`  
- 【Mermaid 图表】：集成学习结构图

D.7 聚类与降维  
- 常用类：`sklearn.cluster.KMeans`、`AgglomerativeClustering`、`DBSCAN`、`GaussianMixture`、`sklearn.decomposition.PCA`、`KernelPCA`、`sklearn.manifold.TSNE`、`TruncatedSVD`、`NMF`  
- 用大白话说：聚类把相似的样本分到一组，降维把高维数据压缩到低维便于可视化。  
- 【Python 代码示例】：`from sklearn.cluster import KMeans; kmeans = KMeans(n_clusters=3); labels = kmeans.fit_predict(X)`  
- 【Mermaid 图表】：无

D.8 管道与评估  
- 常用类与函数：`sklearn.pipeline.Pipeline`、`FeatureUnion`、`make_pipeline`、`make_column_transformer`、`sklearn.metrics.accuracy_score`、`precision_score`、`recall_score`、`f1_score`、`roc_auc_score`、`mean_squared_error`、`mean_absolute_error`、`r2_score`、`confusion_matrix`、`classification_report`、`roc_curve`、`precision_recall_curve`  
- 用大白话说：Pipeline 把预处理和模型串起来避免泄漏，评估函数算出模型好不好。  
- 【Python 代码示例】：`from sklearn.pipeline import Pipeline; pipe = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier())]); print(accuracy_score(y_test, pipe.predict(X_test)))`  
- 【Mermaid 图表】：Pipeline 流程图

D.9 小结  
- scikit-learn 是表格数据竞赛的标准库，预处理、模型、交叉验证、评估四件套掌握即可。

---

## 附录 E：PyTorch 快速参考

E.1 Tensor 基础  
- 常用函数：`torch.tensor`、`torch.zeros`、`torch.ones`、`torch.rand`、`torch.randn`、`torch.arange`、`torch.linspace`、`torch.eye`、`torch.from_numpy`、`.numpy()`、`dtype`、`device`、`requires_grad`、`.detach()`、`.item()`、`.reshape()`、`.view()`、`.transpose()`、`.permute()`、`.squeeze()`、`.unsqueeze()`、`.cat()`、`.stack()`、`.split()`、`.chunk()`、`.index_select()`、`.masked_select()`、`.nonzero()`、`.where()`  
- 用大白话说：Tensor 是 PyTorch 里的多维数组，可以在 GPU 上运行，支持自动求导。  
- 【Python 代码示例】：`import torch; x = torch.tensor([1.0, 2.0], requires_grad=True); y = x * 2; y.backward()`  
- 【Mermaid 图表】：Tensor 与 NumPy 互转图

E.2 张量运算  
- 常用函数：`torch.add`、`sub`、`mul`、`div`、`matmul`、`mm`、`mv`、`dot`、`outer`、`cross`、`sum`、`mean`、`std`、`var`、`min`、`max`、`argmin`、`argmax`、`cumsum`、`cumprod`、`sort`、`argsort`、`unique`、`clamp`、`abs`、`sign`、`exp`、`log`、`sqrt`、`pow`、`sin`、`cos`、`tan`、`sigmoid`、`tanh`、`relu`、`leaky_relu`、`gelu`、`softmax`、`log_softmax`  
- 用大白话说：对 Tensor 做加减乘除、矩阵乘法、激活函数等数学运算。  
- 【Python 代码示例】：`z = torch.matmul(a, b); p = torch.softmax(z, dim=1)`  
- 【Mermaid 图表】：无

E.3 自动微分  
- 常用类与函数：`torch.autograd`、`torch.autograd.grad`、`.backward()`、`torch.no_grad`、`torch.enable_grad`、`torch.set_grad_enabled`、`torch.infer_mode`、`torch.autograd.Function`、`.register_hook()`  
- 用大白话说：自动记录计算过程，反向传播时自动算出梯度，不用手动推导导数。  
- 【Python 代码示例】：`with torch.no_grad(): output = model(input)`  
- 【Mermaid 图表】：前向传播与反向传播流程图

E.4 神经网络模块  
- 常用类：`nn.Module`、`nn.Linear`、`nn.Conv2d`、`nn.Conv1d`、`nn.Conv3d`、`nn.MaxPool2d`、`nn.AvgPool2d`、`nn.AdaptiveAvgPool2d`、`nn.Flatten`、`nn.Dropout`、`nn.BatchNorm1d`、`nn.BatchNorm2d`、`nn.LayerNorm`、`nn.ReLU`、`nn.Sigmoid`、`nn.Tanh`、`nn.LeakyReLU`、`nn.ELU`、`nn.SELU`、`nn.GELU`、`nn.Sequential`、`nn.ModuleList`、`nn.ModuleDict`、`nn.Parameter`、`nn.Embedding`、`nn.LSTM`、`nn.GRU`、`nn.RNN`、`nn.MultiheadAttention`、`nn.Transformer`、`nn.TransformerEncoder`、`nn.TransformerDecoder`  
- 用大白话说：搭建神经网络的积木，全连接、卷积、池化、循环、注意力，组合起来就是模型。  
- 【Python 代码示例】：`class Net(nn.Module): def __init__(self): super().__init__(); self.fc = nn.Linear(10, 1); def forward(self, x): return self.fc(x)`  
- 【Mermaid 图表】：CNN 层结构图

E.5 损失函数  
- 常用类：`nn.MSELoss`、`nn.L1Loss`、`nn.SmoothL1Loss`、`nn.BCELoss`、`nn.BCEWithLogitsLoss`、`nn.CrossEntropyLoss`、`nn.NLLLoss`、`nn.CosineEmbeddingLoss`、`nn.TripletMarginLoss`、`nn.HingeEmbeddingLoss`  
- 用大白话说：衡量模型预测和真实答案的差距，越小越好，反向传播用它来更新参数。  
- 【Python 代码示例】：`criterion = nn.CrossEntropyLoss(); loss = criterion(output, target)`  
- 【Mermaid 图表】：无

E.6 优化器  
- 常用类：`optim.SGD`、`optim.Adam`、`optim.AdamW`、`optim.RMSprop`、`optim.Adagrad`、`optim.Adadelta`、`optim.Adamax`、`optim.LBFGS`  
- 常用方法：`.step()`、`zero_grad()`、`.state_dict()`、`.load_state_dict()`  
- 用大白话说：根据损失函数的梯度，更新模型参数，让模型越来越准。  
- 【Python 代码示例】：`optimizer = optim.Adam(model.parameters(), lr=1e-3); optimizer.zero_grad(); loss.backward(); optimizer.step()`  
- 【Mermaid 图表】：优化器更新流程图

E.7 学习率调度  
- 常用类：`lr_scheduler.StepLR`、`MultiStepLR`、`ExponentialLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`、`OneCycleLR`、`CyclicLR`  
- 常用方法：`.step()`  
- 用大白话说：训练过程中逐渐调小学习率，帮助模型更稳定地收敛。  
- 【Python 代码示例】：`scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1); scheduler.step()`  
- 【Mermaid 图表】：无

E.8 数据加载  
- 常用类：`Dataset`、`DataLoader`、`TensorDataset`、`random_split`、`Subset`、`ConcatDataset`、`Sampler`、`BatchSampler`、`DistributedSampler`  
- 常用参数：`batch_size`、`shuffle`、`num_workers`、`pin_memory`、`collate_fn`、`drop_last`  
- 用大白话说：把数据分批次喂给模型，支持打乱、多进程加载、自定义整批函数。  
- 【Python 代码示例】：`loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)`  
- 【Mermaid 图表】：数据加载器流程图

E.9 训练循环  
- 常用方法：`.train()`、`.eval()`、`.zero_grad()`、`.step()`、`.backward()`、`torch.no_grad()`、`.state_dict()`、`torch.save`、`torch.load`、`.load_state_dict()`、`.to(device)`、`.cuda()`、`.cpu()`  
- 用大白话说：模型训练模式/评估模式切换，梯度清零、反向传播、参数更新，保存和加载模型权重。  
- 【Python 代码示例】：`model.train(); for x, y in loader: optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step(); torch.save(model.state_dict(), 'model.pth')`  
- 【Mermaid 图表】：PyTorch 训练循环图

E.10 小结  
- PyTorch 适合图像、文本等深度学习竞赛，掌握 Tensor、nn.Module、损失函数、优化器、DataLoader 即可搭建并训练模型。

---

## 附录 F：常用 Kaggle 评估指标速查表

- 回归指标：RMSE、MAE、RMSPE、R²  
- 分类指标：准确率、精确率、召回率、F1、AUC、LogLoss  
- 每个指标的计算公式、适用场景、越大越好还是越小越好  
- 【Python 代码示例】：sklearn 一键计算各指标  
- 【Mermaid 图表】：指标选择决策树

---

## 写作约定（供后续扩写参考）

- 每个 x.y 二级章节正面：先讲「这是什么」，再讲「为什么这么做」，最后给「带详细注释的 Python 代码」
- 关键概念必须配一张 Mermaid 图表（流程图、示意图、对比图或决策树）
- 所有代码必须可在 Kaggle Notebooks 免费环境中直接运行
- 语言风格：详细直白，大白话，允许适当废话使文本流畅自然
- 每章结尾附「对应 writeup 列表」，方便读者对照 `docs/kaggle-writeups/` 中的原始赛题深入
