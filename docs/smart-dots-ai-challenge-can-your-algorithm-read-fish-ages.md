# SmartDots AI Challenge – 读懂鱼龄

## 一、比赛简介

### 1.1 比赛背景
SmartDots AI Challenge 是由 ICES 国际海洋考察理事会主办的渔业年龄鉴定竞赛，目的是用机器学习替代人工阅读鱼类耳石耳石年轮的传统方法。耳石是鱼类内耳的钙化结构，生长过程中每年会形成一圈年轮，年轮的宽度与生长速率有关。传统上需要资深渔业专家在显微镜下手工计数，这种方法耗时长且不同实验室之间存在系统性偏差。比赛提供的是 Sole 7a，即北海比目鱼的耳石显微图像，图像分辨率足以看到微米级的纹理变化。任务的核心是从这些高倍显微照片中自动提取年龄相关的视觉模式，而不是依赖人工标注的年龄。

比赛数据来自欧洲渔业管理机构的真实科研数据，每条样本都有捕捞信息、鱼体测量信息和若干张耳石照片。比赛平台提供 API 接口，需要用 eventId 1842 获取样本清单，然后自行下载图片。参赛者不仅要解决图像分类回归问题，还要处理图片获取、GUID 匹配、软标签生成和模型融合的完整流程，这对入门者来说是很好的系统工程练习。

### 1.2 比赛目标
给定比赛 Event 1842 的样本列表，每条样本包含若干张耳石照片，照片的 URL 由系统生成。我们需要为每个 fishID 输出一个 ModalAge，年龄范围通常在 1 到 20 岁之间。训练阶段比赛没有直接提供真实年龄标签，材料里采用的常见做法是先用 Von Bertalanffy 生长公式从鱼长生成一个软标签作为监督信号。数据列包括 fishID、tblSampleID、sampleID、fishLength、fishWeight、catchDate 以及本地图片路径。我们需要把这些信息整理成 DataFrame，再与本地下载的图片路径对应起来。测试集只有图片和鱼长，我们要据此填上年龄预测，格式需要符合 API 提交规范，包括 email、institute、nameAIAlgorithm 等元数据。

### 1.3 评价标准
比赛线上使用平均绝对误差 MAE 作为评判，预测年龄与真实年龄的绝对差值的平均值越小越好。MAE 对异常值不敏感，适合年龄这种离散且分布偏斜的目标。本地我们用验证集的 MAE 跟踪效果，并通过学习曲线观察训练是否过拟合。材料中的最终提交策略是取 CNN 预测与 Von Bertalanffy 生长基线的平均作为 blend，这样可以兼顾图像纹理和生长规律的先验信息，避免模型在极端鱼长上出现不合理的预测。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先读入 Event JSON，了解样本数量和物种。

```python
# 读取比赛事件元数据
with open(INPUT_DIR / 'Event_1842.json') as f:
    event_data = json.load(f)
samples = event_data['samples']
print(f"Species: {event_data['species']}")
print(f"Total samples: {len(samples)}")
```

结果显示物种为 Sole 7a，样本共 332 张磁盘图片，提交图片需要下载。随后我们把样本里的 GUID 映射到磁盘文件。

```python
# 统计磁盘上已有的图片
all_images = list(INPUT_DIR.rglob('*.jpg'))
disk_lookup = {p.stem.upper(): str(p) for p in all_images}
print(f'Images on disk: {len(all_images)}')
```

发现已匹配 0 张，需要下载缺失的提交图片。我们用线程池并发下载，并把鱼长信息整理成 DataFrame。

```python
# 构建提交 DataFrame
submit_df['vb_age'] = submit_df['fishLength'].apply(von_bertalanffy_age)
print(submit_df['vb_age'].value_counts().sort_index())
```

VB 年龄分布集中在 1-10 岁，鱼长范围从几十毫米到 300 多毫米。年轮清晰度与鱼长高度相关。

### 2.2 数据清洗
耳石照片命名规则统一为 GUID。我们按 GUID 匹配图片，缺失的自动下载，已有的拷贝到统一目录。

```python
# 拷贝已有图片
for guid in overlap:
    src = Path(disk_lookup[guid])
    dest = SUBMIT_IMG_DIR / src.name
    if not dest.exists():
        shutil.copy(src, dest)
```

图片裁剪到 224x224 并做归一化。训练变换加入随机构图和颜色抖动，验证只做 resize。

```python
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE+32, IMG_SIZE+32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
```

极短或无法匹配的样本我们用中位年龄 3 作为 fallback，保证标签完整。

### 2.3 模型训练与超参数调优
由于没有真实年龄，我们用 Von Bertalanffy 公式产生的软标签做监督。公式参数取 Linf=370mm、K=0.27、t0=-0.5。

```python
def von_bertalanffy_age(length_mm, Linf=370, K=0.27, t0=-0.5):
    age = t0 - (1.0/K) * np.log(1.0 - length_mm / Linf)
    return max(1, min(20, round(age)))
```

模型用 ResNet18 作为骨干，最后一层换成回归头。

```python
class FishAgeModel(nn.Module):
    def __init__(self):
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.head(feat).squeeze(1)
```

训练采用双学习率，骨干 0.1 倍，头全速，AdamW + CosineAnnealing。

```python
optimiser = torch.optim.AdamW([
    {'params': list(model.backbone.parameters()), 'lr': LR*0.1},
    {'params': list(model.head.parameters()),     'lr': LR},
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
```

每轮用 MSE Loss 训练，验证时计算 MAE。最佳模型保存，训练 30 epoch 后验证 MAE 稳定。

### 2.4 模型测试与指标报告
测试时采用 Test Time Augmentation，对水平、垂直翻转做平均。

```python
tta_tfs = [
    transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize(...)])],
    # 水平翻转
    # 垂直翻转
]
```

随后把 CNN 预测与 VB 基线做简单平均。

```python
submit_df['cnn_age'] = submit_df['local_path'].map(cnn_pred_map)
submit_df['blend_age'] = submit_df.apply(lambda r: round((r['cnn_age']+r['vb_age'])/2), axis=1)
print(submit_df[['fishID','fishLength','vb_age','cnn_age','blend_age']])
```

本地验证 MAE 约 1.2 岁，blend 后略有提升。最终生成提交文件，包含 fishID 与 predicted_age。

## 三、总结
本文中我们完成了 SmartDots AI Challenge 的完整流程。我们首先通过 EDA 了解耳石数据规模与鱼长分布，并用 Von Bertalanffy 公式生成软标签；其次完成了图片下载、匹配与标准化预处理；接着用 ResNet18 配合软标签训练回归模型，并用 TTA 与基线融合提升稳定性；最后在验证集上得到 MAE 约 1.2 岁，并生成提交文件。
