<h1 align="center">百战 Kaggle</h1>

<p align="center"><img src='docs/public/banner.png' /></p>

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

*你已经抓住了石头，现在是挥舞它的时候了！——《天之炽》*

*从 400+ Writeup 中提炼核心洞见，助力选手跑通整个比赛流程。*

## 项目受众

机器学习入门者，想要打比赛获得荣誉的人。

## 在线阅读
https://datawhalechina.github.io/baizhan-kaggle

## 目录

| 章节名 | 简介 | 状态 |
| ---- | ---- | ---- |
| [第1章 Kaggle 比赛流程与规则](docs/ch01/index.md) | 从注册到首次提交的完整走一遍，含 Kaggle CLI 与 MCP 配置 | 已完成 |
| [第2章 探索性数据分析（EDA）](docs/ch02/index.md) | 先把数据摊开看，分布、缺失值、异常值与相关性 | 已完成 |
| [第3章 表格数据的通用清洗与建模方法](docs/ch03/index.md) | 缺失值、类别编码、特征构造、数值变换、基线与主力模型 | 已完成 |
| [第4章 表格数据比赛详解：房价预测（House Prices）](docs/ch04/index.md) | Ames 房价回归，表格比赛解题全流程示例 | 已完成 |
| [第5章 表格数据比赛详解：信用评分（Give Me Some Credit）](docs/ch05/index.md) | 二分类，类别不平衡，LogLoss 指标 | 已完成 |
| [第6章 表格数据比赛详解：贷款审批（Loan Approval）](docs/ch06/index.md) | 贷款是否批准，分类准确率与 F1 分数 | 已完成 |
| [第7章 表格数据比赛详解：推土机拍卖价格预测（Blue Book for Bulldozers）](docs/ch07/index.md) | 回归，时间切分验证，RMSPE 指标 | 已完成 |
| [第8章 时序数据的通用清洗与建模方法](docs/ch08/index.md) | 日期拆分、滞后特征、滚动统计、时间切分验证 | 已完成 |
| [第9章 时序数据比赛详解：门店销售预测（Rossmann Store Sales）](docs/ch09/index.md) | 日销售额预测，RMSPE，按时间切分验证集 | 已完成 |
| [第10章 时序数据比赛详解：交通流量预测（TrafficFlowBench）](docs/ch10/index.md) | 多任务时序，早晚高峰与节假日效应 | 已完成 |
| [第11章 时序数据比赛详解：电力负荷预测](docs/ch11/index.md) | 气象与负载，逐小时电力负载预测 | 已完成 |
| [第12章 图像数据的通用清洗与建模方法](docs/ch12/index.md) | 像素、缩放归一化、CNN、数据增强、迁移学习 | 已完成 |
| [第13章 图像数据比赛详解：猫狗大战（Dogs vs. Cats）](docs/ch13/index.md) | 二分类图像，从零训练 CNN 到 VGG16 迁移学习 | 已完成 |
| [第14章 图像数据比赛详解：糖尿病视网膜病变检测（Diabetic Retinopathy）](docs/ch14/index.md) | 图像多分类，QWK 指标 | 已完成 |
| [第15章 图像数据比赛详解：卫星图像分类（Planet / Carvana）](docs/ch15/index.md) | 多标签分类，F2 指标，U-Net 分割 | 已完成 |
| [第16章 文本数据的通用清洗与建模方法](docs/ch16/index.md) | 文本清洗、词袋模型、TF-IDF、浅层文本模型 | 已完成 |
| [第17章 文本数据比赛详解：Quora 问题对去重](docs/ch17/index.md) | 问题相似度，TF-IDF 与共同邻居特征 + XGBoost | 已完成 |
| [第18章 文本数据比赛详解：电影评论情感分析（IMDb）](docs/ch18/index.md) | 文本多分类，分类准确率 | 已完成 |
| [第19章 文本数据比赛详解：文本分类与关键词抽取（Tradeshift / Facebook Recruiting III）](docs/ch19/index.md) | 多标签文本分类，噪声文本与稀有特征 | 已完成 |
| [第20章 模型评估与验证：先看得分准不准](docs/ch20/index.md) | 指标选型、交叉验证、目标泄漏、本地分与排行榜分差距 | 已完成 |
| [第21章 模型调优与集成学习：榜单上的最后几名](docs/ch21/index.md) | 超参数搜索、投票、Stacking、早停 | 已完成 |
| [第22章 比赛策略与心态](docs/ch22/index.md) | 选比赛、先跑基线、防过拟合、提交策略、团队合作 | 已完成 |
| [第23章 从 Kaggle 到实际机器学习项目](docs/ch23/index.md) | 竞赛到业务的迁移、模块化代码、模型部署 | 已完成 |
| [附录A NumPy 快速参考](docs/AppendixA/index.md) | 数组创建、索引、统计、矩阵运算 | 已完成 |
| [附录B pandas 快速参考](docs/AppendixB/index.md) | Series/DataFrame、读写、筛选、分组、合并 | 已完成 |
| [附录C matplotlib 快速参考](docs/AppendixC/index.md) | 常见图表、坐标轴、样式、保存 | 已完成 |
| [附录D scikit-learn 快速参考](docs/AppendixD/index.md) | 预处理、模型、交叉验证、管道、评估 | 已完成 |
| [附录E PyTorch 快速参考](docs/AppendixE/index.md) | Tensor、nn.Module、损失、优化器、训练循环 | 已完成 |
| [附录F 常用 Kaggle 评估指标速查表](docs/AppendixF/index.md) | 回归/分类指标公式与适用场景 | 已完成 |

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| [张子辰](https://github.com/wizardforcel/) | 项目负责人 | 专注于模型量化、端侧推理和注意力机制改进 |

## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull Request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

