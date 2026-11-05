# SmartDots AI Challenge – 读懂鱼龄

## 一、比赛简介

### 1.1 比赛背景

鱼类年龄鉴定是渔业资源评估里非常基础、却又非常费人力的一步。科学家想知道一群鱼里有多少岁数不同的个体，因为年龄结构会直接影响对种群增长速度、繁殖能力和可捕捞量的判断。最常用的材料是鱼的内耳里一块钙化结构，叫作耳石。耳石会随着鱼一起长大，生长快的季节形成较宽、较浅的轮纹，生长慢的季节形成较窄、较深的轮纹，一年下来就留下了一圈类似树木年轮的痕迹。把耳石放到显微镜下，请有经验的读龄员一圈一圈数，就能估计这条鱼几岁。

问题在于，这件事高度依赖人。读龄员要经过长期训练，盯着高倍显微照片反复比对；一条鱼往往要请多位读龄员各自独立读一遍，再讨论分歧；不同实验室之间的判读习惯还可能不一样，产生系统性偏差。整个过程慢、贵，而且难以完全复现。SmartDots 是 ICES 国际海洋考察理事会推动的一个平台，目标就是把耳石图像、样本信息 and 读龄结果集中管理，并尝试用算法辅助甚至替代部分人工读龄。本场比赛就是在这个背景下提出的：让参赛者的算法直接看耳石照片，输出鱼的年龄。

比赛使用的数据标注为 Sole 7a，也就是鳎鱼（sole）的一个渔业管理单元。每条样本除了若干张耳石显微照片，还带有捕捞日期、鱼长、体重等测量信息。照片不是随比赛文件一起打包好的，而是要先读取比赛事件 JSON，拿到每张图片的下载地址，再自行从平台 API 下载。因此这道题并不是单纯的图像回归题，它还包含元数据解析、文件名与 GUID 匹配、缺失图片补全、软标签构造、模型训练和 API 提交等一整套工程流程。对刚接触深度学习的人来说，这种端到端的练习很有价值，因为它接近真实科研数据项目的形态：数据散落在不同地方，字段名字不统一，图片还可能缺几张。

### 1.2 比赛目标

比赛给参赛者一个事件编号 1842，对应文件 Event_1842.json。这个 JSON 里有事件名称、物种、样本列表，每个样本包含 fishID、tblSampleID、sampleID、鱼长、体重、捕捞日期以及一个 images 列表。images 里的每一项给出图片 URL。我们要做的是：为每个 fishID 给出一个 ModalAge，也就是这条鱼最可能的年龄。ModalAge 这个名字来自平台的实际工作方式，通常指多位读龄员重复读龄后出现次数最多的那个年龄，可以把它理解为一个经过共识的年龄标签。

真正的难点是，训练阶段并没有直接提供可靠的年龄标签。材料里的做法是先用鱼长和 Von Bertalanffy 生长方程算出一个“软标签”，把它当作监督信号。这样做的逻辑是：鱼越长，年龄通常越大，生长方程能把长度映射成一个合理的年龄估计。这个标签不是真实年龄，但它能提供一个可学习的近似目标。于是任务就变成了“用耳石图像去拟合由鱼长推出来的年龄”，模型需要从图像纹理里找到与生长阶段相关的线索。

最终提交需要符合平台 API 的格式，至少包含 email、institute、nameAIAlgorithm、version 和 reports 等字段，reports 里每条记录给出 ID 与 ModalAge。材料里也提供了先保存 CSV 再提交的写法，CSV 的列名是 ID 和 ModalAge。需要特别留意的是，材料中的实现是按“图片”逐行预测的：同一条鱼如果有多张照片，就会出现多个相同 fishID 的记录。严格的提交应当先按 fishID 聚合，再给每条鱼一个年龄；这一点是读代码时要自己看出来的工程细节。

### 1.3 评价标准

比赛线上使用平均绝对误差 MAE 作为评价指标。它的定义很直白：对每条鱼，把预测年龄与真实年龄差的绝对值算出来，再把所有样本的绝对值取平均。MAE 越小，说明预测越接近真实年龄。与均方误差 MSE 相比，MAE 对个别极端错误没那么敏感，因为误差不被平方放大；年龄又是一个取值范围不大、分布偏斜的离散目标，所以用 MAE 来衡量比较符合业务直觉。假设一条鱼真实 5 岁，模型预测 7 岁，这一条的绝对误差就是 2；如果另一条预测 4 岁，误差就是 1；把所有误差平均起来，就是这一批预测的 MAE。

本地训练时，我们也用验证集 MAE 来观察模型有没有学到东西。但这里有一个必须讲清楚的差别：材料里的验证标签同样是 Von Bertalanffy 软标签，而不是人工读出的真实年龄。所以验证 MAE 衡量的其实是“模型预测与生长公式有多一致”，它不能直接等同于线上成绩。验证 MAE 很低，只说明模型很好地拟合了软标签；它是否真的学会了数年轮，还要靠真实标签或线上评测来回答。材料最后的提交策略是把 CNN 预测与生长公式基线做等权平均，这样既利用图像信息，也保留鱼长先验，避免模型在极端鱼长上给出离谱年龄。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先读取比赛事件文件，弄清楚这次任务的对象是什么。打开 Event_1842.json 后，我们打印事件名、物种、样本总数，以及第一个样本包含哪些字段。这样做的目的不是马上建模，而是先确认数据长什么样：图片地址到底藏在哪一层，鱼长字段叫什么名字，样本和图片是一对一还是一对多。对入门者来说，这一步很像拿到一份陌生 Excel 表后先看表头和前几行，先建立整体印象，再决定后面怎么处理。

其次，我们扫描比赛输入目录，把磁盘上已有的 jpg、jpeg、png 图片全部找出来。每张图片的文件名去掉扩展名后就是一个 GUID，我们统一转成大写，建立一个“GUID 到本地路径”的字典。GUID 是平台给每张图片分配的唯一编号，后面的匹配、下载、去重都靠它。这里统一大小写很重要，因为不同系统导出的文件名大小写可能不一致，如果不统一，就会把同一张图当成两张。

接着，我们从事件 JSON 的每个 images 项里解析出 URL，取 URL 最后一段，去掉扩展名并转成大写，得到这次事件需要的 GUID 集合。然后求这个集合与磁盘 GUID 集合的交集，就能知道哪些图片已经存在、哪些还需要下载。材料里的打印语句会分别输出事件 GUID 数、磁盘 GUID 数、已匹配数和待下载数。这个“集合求交”的思路非常实用，比逐条循环判断更清晰，也不容易漏。

然后，材料还从提交表里随机抽了八条样本，画成 2 行 4 列的图片网格，每张图标题写上鱼长和由生长公式算出的年龄。我们通过这张图可以直观看到耳石照片的大致样子：多数是圆形或椭圆形的薄片，中间有明暗相间的环带，边缘清晰程度不一。鱼长从几十毫米到三百多毫米都有，对应的软标签年龄大多落在较小的范围。图像质量、拍摄角度和背景并不完全统一，这也解释了后面为什么要做随机裁剪、翻转和颜色扰动。

最后，我们对 EDA 的结论做一个小结：事件 1842 的物种是 Sole 7a，样本以 fishID 标识，图片以 GUID 标识，两者通过样本里的 images 列表关联；磁盘图片和事件需求之间可能存在缺口，需要下载；鱼长是唯一能稳定拿到的数值型生长信息，因此它自然成为构造软标签的依据。下面这段代码就是我们读取事件、扫描磁盘并计算匹配情况的完整过程。

```python
# 读取比赛事件文件，先了解样本规模与字段结构
with open(INPUT_DIR / 'Event_1842.json', encoding='utf-8') as f:
    event_data = json.load(f)

samples = event_data['samples']
print('Event:', event_data['eventName'])
print('Species:', event_data['species'])
print('Total samples:', len(samples))
print('Sample keys:', list(samples[0].keys()))

# 扫描磁盘上的所有图片，用“文件名去掉扩展名”作为 GUID
all_images = (list(INPUT_DIR.rglob('*.jpg')) +
              list(INPUT_DIR.rglob('*.jpeg')) +
              list(INPUT_DIR.rglob('*.png')))
disk_lookup = {p.stem.upper(): str(p) for p in all_images}
disk_guids = set(disk_lookup.keys())
print('Images on disk:', len(all_images))

# 从图片 URL 末尾解析 GUID，汇总成本次事件需要的图片集合
event_guids = set()
for sample in samples:
    for img in sample['images']:
        guid = img['url'].split('/')[-1].rsplit('.', 1)[0].upper()
        event_guids.add(guid)

# 用集合交集和差集判断已有图片与待下载图片
overlap = event_guids & disk_guids
print('Event GUIDs:', len(event_guids))
print('Already matched:', len(overlap))
print('Need to download:', len(event_guids - disk_guids))
```

### 2.2 数据清洗

我们首先确定以 GUID 作为连接键。事件 JSON 里描述的是“哪条鱼有哪些图片”，磁盘上放的是“哪个 GUID 对应哪个文件”，两边只有 GUID 能对上。我们把所有文件名转成大写，避免大小写差异造成假缺失。对于扩展名，我们也做了统一处理：从 URL 里取最后一段，按最后一个点切分，前半段是 GUID，后半段是扩展名；下载时用 GUID 加扩展名作为目标文件名，这样同一张图不会因为扩展名不同而重复保存。

其次，对于磁盘上已经存在的图片，我们直接复制到统一的提交图片目录。复制前先判断目标文件是否已经存在，存在就跳过，避免重复劳动。这样做的目的是把所有后续要用的图片集中到一个目录，训练、验证和提交都从同一个地方取路径，减少“这张图到底在哪”的混乱。

接着，我们处理缺失图片。材料用 ThreadPoolExecutor 开 8 个线程并发下载，每个请求设置 30 秒超时，并用 raise_for_status 检查 HTTP 状态码。下载失败的样本不会让整个程序崩掉，而是返回 False，最后统计成功和失败的数量。这个设计对真实项目很重要：网络请求总会有超时、404 或者临时故障，如果串行下载，几百张图会等很久；如果一失败就抛异常，整个流程又会被一张坏图卡死。并发加容错是比较稳妥的折中。

然后，我们把样本整理成 DataFrame。每条图片记录一行，字段包括 fishID、tblSampleID、sampleID、fishLength、fishWeight、catchDate 和 local_path。整理完用 dropna 去掉 local_path 为空的记录，因为下游模型必须读到实际文件。这里保留图片级的一行，而不是立刻按鱼聚合，是为了跟材料的实现保持一致；后面我们会指出这种粒度带来的问题以及更严谨的做法。

最后，我们统一图像预处理。训练时的变换是：先把短边缩放到 224+32，再随机裁到 224×224，加入水平和垂直翻转、15 度以内的随机旋转、亮度与对比度各 0.3 的颜色抖动，最后转成张量并按 ImageNet 的均值和标准差归一化。验证时只做缩放到 224×224 和同样的归一化，不加随机扰动，保证每次评估结果可比。自定义 Dataset 负责用 PIL 打开图片、强制转成 RGB，并把标签转成 float32。下面三段代码分别完成图片复制与下载、DataFrame 构造、以及变换与 Dataset 定义。

```python
# 把磁盘上已有的图片复制到统一的提交目录，已存在就跳过
copied = 0
for guid in overlap:
    src = Path(disk_lookup[guid])
    dest = SUBMIT_IMG_DIR / src.name
    if not dest.exists():
        shutil.copy(src, dest)
        copied += 1
print('Copied existing images:', copied)
# 找出仍然缺失的图片，记录鱼号、下载地址和目标路径
missing = []
for sample in samples:
    for img in sample['images']:
        raw = img['url'].split('/')[-1]
        guid = raw.rsplit('.', 1)[0].upper()
        ext = '.' + raw.rsplit('.', 1)[1].lower()
        dest = SUBMIT_IMG_DIR / (guid + ext)
        if not dest.exists():
            missing.append({'fishID': sample['fishID'],'url': img['url'],'dest': dest})
# 单张下载函数：任何异常都返回 False，不让主流程中断
def download_one(item):
    try:
        r = requests.get(item['url'], timeout=30)
        r.raise_for_status()
        item['dest'].write_bytes(r.content)
        return True
    except Exception:
        return False
# 开 8 个线程并发下载，并统计成功与失败数量
ok = fail = 0
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(download_one, m) for m in missing]
    for f in as_completed(futures):
        if f.result(): ok += 1
        else: fail += 1
print('Downloaded:', ok, 'Failed:', fail)
```

```python
# 按 GUID 把本地图片路径写回样本表，一张图片一行
submit_lookup = {p.stem.upper(): str(p) for p in SUBMIT_IMG_DIR.iterdir()}
records = []
for sample in samples:
    for img in sample['images']:
        guid = img['url'].split('/')[-1].rsplit('.', 1)[0].upper()
        records.append({
            'fishID': sample['fishID'],
            'tblSampleID': sample['tblSampleID'],
            'sampleID': sample['sampleID'],
            'fishLength': sample.get('fishLength'),
            'fishWeight': sample.get('fishWeight'),
            'catchDate': sample.get('catchDate'),
            'local_path': submit_lookup.get(guid),
        })

# 没有成功拿到本地文件的记录无法训练，直接丢掉
submit_df = pd.DataFrame(records).dropna(subset=['local_path'])
print('Rows with images:', len(submit_df))
print(submit_df[['fishID', 'fishLength', 'local_path']].head())
```

```python
# 训练变换：先放大再随机裁剪，并加入翻转、旋转和颜色扰动
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# 验证变换只做缩放和归一化，保证评估稳定可比
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# 自定义数据集：读图、转 RGB、应用变换、标签转 float32
class OtolithDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return img, label
```

### 2.3 模型训练与超参数调优

我们首先解决标签问题。由于比赛没有直接给出真实年龄，材料用 Von Bertalanffy 生长方程构造软标签。这个方程的常见形式是 L(t) = Linf × (1 - exp(-K × (t - t0)))，其中 Linf 是理论最大体长，K 是生长速率，t0 是理论出生时的“负年龄”。把它反解，就能从测到的体长 L 推出年龄 t = t0 - (1/K) × ln(1 - L/Linf)。材料采用 Sole 7a 的参数 Linf=370 毫米、K=0.27、t0=-0.5。这个公式有一个前提：L 必须小于 Linf，否则对数里会出现负数；所以代码先把鱼长截断到 Linf 的 99%，再代入计算。算出的年龄还要四舍五入，并夹在 1 到 20 岁之间，避免出现不合理的负年龄或超大年龄。鱼长缺失时，函数返回 3，这是材料选定的中位年龄兜底值。

其次，我们把这个函数应用到提交表的 fishLength 上，得到 vb_age 一列，并打印分布。材料还打印了鱼长的最小值和最大值，以及前几行的 fishID、鱼长和软标签。我们从中可以确认：软标签的取值是离散的整数，集中在 1 到 10 岁附近，鱼长越大，公式给出的年龄越大。这个单调关系是后面融合策略的基础，也是软标签最大的局限：模型虽然只看耳石图像，但监督信号本身是由鱼长决定的，所以它学到的很可能是“如何从耳石推断体长和生长阶段”，而不一定是真正的年轮计数。我们在报告结果时必须诚实地说明这一点。

接着，我们构造训练集和验证集。材料把磁盘上的全部图片都拿来训练：如果某张图的路径能在“路径到 VB 年龄”的映射里找到，就用对应的软标签；找不到的图片一律给默认年龄 3。代码注释里写明训练集使用磁盘上的 332 张图片。验证集则只取提交表里的图片，标签同样是对应的 VB 年龄。这里有一个需要特别指出的问题：训练集包含了全部磁盘图片，而验证集又来自提交图片，两者很可能存在重叠，也就是同一张图片既参加训练又参加验证。这样得到的验证 MAE 会偏乐观，只能当作“流程有没有跑通”的自检，不能当作泛化性能。更严谨的做法是按 fishID 分组划分，保证同一条鱼的所有照片只出现在训练或验证的一侧。

然后，我们搭建模型。骨干网络用 torchvision 提供的 ResNet18，并加载 ImageNet 预训练权重。ResNet18 是残差网络，核心思想是让每一层学习“在原有特征上还需要补充什么”，而不是从零学习完整映射，这样深层网络更容易训练。我们把 ResNet18 最后的全连接层去掉，保留前面的卷积特征提取部分；后面接一个自定义回归头：Dropout 概率 0.3，全连接 512 降到 128，ReLU 激活，再 Dropout 0.2，最后 128 降到 1，输出一个连续的年龄预测值。输出层不加 Sigmoid 之类的限制，是因为年龄回归希望模型自由拟合，最后再在推理阶段做取整和截断。

接着，我们设置优化器。使用 AdamW，并把参数分成两组：骨干网络的学习率是基础学习率 LR 的 0.1 倍，回归头使用完整的 LR。这样做的原因是预训练骨干已经学到了通用图像特征，不希望被新任务一下子冲掉；而新加的回归头是随机初始化的，需要较快地学习。基础学习率取 3e-4，权重衰减 1e-4，学习率调度用 CosineAnnealingLR，T_max 等于总 epoch 数，eta_min 设为 1e-6，让学习率按余弦曲线平滑下降。训练损失用 MSE，因为软标签是连续数值，MSE 能提供平滑的梯度；评估时再换成 MAE，与比赛指标对齐。

最后，我们写训练循环。每个 epoch 先调用 model.train()，遍历训练批次，前向计算预测值，算 MSE 损失，反向传播，用 clip_grad_norm_ 把梯度范数限制在 1.0，防止梯度爆炸，然后更新参数。验证时调用 model.eval() 并关闭梯度计算，收集所有预测值和标签，按“先四舍五入再取绝对误差”的方式计算 MAE。每轮结束调用 scheduler.step()，如果验证 MAE 比历史最好值更低，就保存模型权重。材料设定训练 30 个 epoch，并打印每轮的损失、验证 MAE 和耗时，最后输出最佳验证 MAE。下面的代码依次展示软标签函数、模型结构、数据集划分和训练循环。

```python
# 用 Von Bertalanffy 生长方程从鱼长反推年龄，作为软标签
def von_bertalanffy_age(length_mm, Linf=370, K=0.27, t0=-0.5):
    # 鱼长缺失时返回中位年龄 3，保证标签列没有空值
    if length_mm is None or (isinstance(length_mm, float) and np.isnan(length_mm)):
        return 3
    length_mm = float(length_mm)
    # 对数要求 1 - L/Linf 为正，因此把鱼长截断到 Linf 的 99%
    length_mm = min(length_mm, Linf * 0.99)
    age = t0 - (1.0 / K) * np.log(1.0 - length_mm / Linf)
    # 年龄取整并限制在 1 到 20 岁之间
    return max(1, min(20, round(age)))

# 对每条提交记录计算软标签，并查看分布
submit_df['vb_age'] = submit_df['fishLength'].apply(von_bertalanffy_age)
print(submit_df['vb_age'].value_counts().sort_index())
print('Length range:',
      submit_df['fishLength'].min(), '-',
      submit_df['fishLength'].max(), 'mm')
print(submit_df[['fishID', 'fishLength', 'vb_age']].to_string())
```

```python
# 基于 ImageNet 预训练的 ResNet18 构建年龄回归模型
class FishAgeModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        # 去掉原网络最后的全连接层，只保留特征提取部分
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # 自定义回归头：512 维特征逐步降到 1 维年龄
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.head(feat).squeeze(1)

# 实例化模型并做一次前向检查，确认输出形状正确
model = FishAgeModel(pretrained=True).to(DEVICE)
with torch.no_grad():
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    print('Output shape:', model(dummy).shape)
print('Trainable params:',
      sum(p.numel() for p in model.parameters() if p.requires_grad))
```

```python
# 建立“图片路径 -> 软标签”的映射，训练时按路径查标签
submit_path_set = set(submit_df['local_path'].tolist())
submit_vb_map = dict(zip(submit_df['local_path'], submit_df['vb_age']))

# 训练集使用磁盘上全部图片；没有 VB 映射的图片用默认年龄 3
all_disk_paths = [str(p) for p in all_images]
all_disk_labels = [submit_vb_map.get(str(p), 3) for p in all_images]

# 验证集使用提交图片及其 VB 软标签
val_paths = submit_df['local_path'].tolist()
val_labels = submit_df['vb_age'].tolist()

# 分别构建数据集与数据加载器
train_ds = OtolithDataset(all_disk_paths, all_disk_labels, transform=train_tf)
val_ds = OtolithDataset(val_paths, val_labels, transform=val_tf)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=(DEVICE.type == 'cuda'))

print('Train:', len(train_ds), 'Val:', len(val_ds))
```

```python
# 优化器分组：骨干用小学习率，回归头用完整学习率
optimiser = torch.optim.AdamW([{'params': list(model.backbone.parameters()), 'lr': LR*0.1}, {'params': list(model.head.parameters()), 'lr': LR}], weight_decay=1e-4)
# 余弦退火，让学习率随 epoch 平滑下降
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS, eta_min=1e-6)
best_val_mae = float('inf')
history = {'train_loss': [], 'val_mae': []}
for epoch in range(1, EPOCHS + 1):
    # 训练阶段：前向、算损失、反向传播、裁剪梯度、更新参数
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimiser.zero_grad()
        loss = F.mse_loss(model(imgs), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        train_loss += loss.item()
    # 验证阶段：关闭梯度，收集预测并计算 MAE
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = model(imgs.to(DEVICE)).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(labels.numpy())
    val_mae = np.mean([abs(round(float(p)) - int(t)) for p, t in zip(val_preds, val_true)])
    scheduler.step()
    history['train_loss'].append(train_loss / len(train_loader))
    history['val_mae'].append(val_mae)
    # 只保存验证 MAE 最低的权重，避免最后一轮过拟合
    flag = ''
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), MODEL_DIR / 'best_model.pth')
        flag = ' saved'
    print(f'Epoch {epoch:02d}/{EPOCHS} loss={train_loss/len(train_loader):.4f} val_MAE={val_mae:.3f}{flag}')
print('Best val MAE:', best_val_mae)
```

### 2.4 模型测试与指标报告

我们首先加载验证 MAE 最低的那份权重，并把模型切到 eval 模式。这样 Dropout 会被关闭，BatchNorm 也使用训练期间累积的统计量，推理结果才稳定。材料用 torch.load 并指定 map_location=DEVICE，保证在 GPU 或 CPU 环境下都能正确恢复权重。

其次，我们做测试时增强，也就是 TTA。思路是：同一张图片做几种不改变语义的变换，分别送进模型预测，再把多个预测平均起来，降低单次预测的随机波动。材料用了三种视图：原图缩放、水平翻转、垂直翻转。水平翻转对耳石照片通常是安全的，因为年轮结构没有严格的左右方向；垂直翻转是否合理要看拍摄习惯，材料把它也作为一种扰动。我们对每个视图都构建一个 Dataset 和 DataLoader，批量推理后把同一路径的三次预测收集到字典里。

接着，我们对每个路径的三次预测取平均，再四舍五入并夹到 1 到 20 岁，得到 cnn_age。然后做融合：把 CNN 预测和 Von Bertalanffy 基线做等权平均，再四舍五入，得到 blend_age。这个融合很朴素，但含义清楚：如果模型从图像里学到了额外信息，它会和生长先验互相补充；如果模型在某个鱼长区间偏得很厉害，基线可以把它往回拉一点。材料没有对两个来源做加权搜索，而是直接用 0.5 比 0.5，这样也避免了在软标签验证集上过度调参。

然后，我们生成提交文件。材料里出现了两种写法：一种 CSV 用 predicted_age 作为列名，另一种用 ID 和 ModalAge；JSON 也有 fishID/age 与 ID/ModalAge 两种版本。最后一段代码使用的是 ID 和 ModalAge，并配合 SUBMIT_EVENT_ID、YOUR_EMAIL、YOUR_INSTITUTE、MODEL_NAME 等配置。我们以最后这一段为准，因为它和平台 API 的最终字段要求一致。提交前可以把 requests.post 那行取消注释，向 submitFishAges 接口发送请求，并打印状态码和返回的 JSON。

最后，我们报告指标。材料会打印最佳验证 MAE，并画出训练损失曲线和验证 MAE 曲线。需要再次强调：这个验证 MAE 是相对软标签计算的，而且训练集与验证集存在图片重叠，所以它偏乐观，不能直接当作线上成绩。材料本身没有给出一个固定的排行榜分数，因此我们在这里不编造具体数字，只报告可复现的流程和评估口径。如果要得到可信的本地指标，应当按 fishID 分组划分训练与验证，最好再留出一部分有人工读龄真值的样本；如果一条鱼有多张照片，还应先在图片级预测后按 fishID 取中位数或多数票，再提交每条鱼一个 ModalAge。下面两段代码分别完成 TTA 推理和融合提交。

```python
# 加载最佳权重，并准备三种 TTA 视图：原图、水平翻转、垂直翻转
model.load_state_dict(torch.load(MODEL_DIR / 'best_model.pth', map_location=DEVICE))
model.eval()
# 定义三种测试时增强变换
tta_tfs = [
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomVerticalFlip(p=1.0),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])]),
]
# 对每个视图分别推理，把同一路径的预测收集到一起
all_raw_preds = {path: [] for path in val_paths}
for tf in tta_tfs:
    ds = OtolithDataset(val_paths, val_labels, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            preds = model(imgs.to(DEVICE)).cpu().numpy()
            for j, p in enumerate(preds):
                path = val_paths[i * BATCH_SIZE + j]
                all_raw_preds[path].append(float(p))
# 三次预测取平均，再取整并限制到 1 至 20 岁
cnn_pred_map = {path: max(1, min(20, round(np.mean(preds)))) for path, preds in all_raw_preds.items()}
print('TTA finished for', len(cnn_pred_map), 'images')
```

```python
# 把 CNN 预测写回提交表，并与 VB 基线做等权平均
submit_df['cnn_age'] = submit_df['local_path'].map(cnn_pred_map)
submit_df['blend_age'] = submit_df.apply(
    lambda r: round((r['cnn_age'] + r['vb_age']) / 2), axis=1)

print(submit_df[['fishID', 'fishLength',
                 'vb_age', 'cnn_age', 'blend_age']].to_string())

# 保存平台要求的 CSV：一列 ID，一列 ModalAge
submit_df[['fishID', 'blend_age']].rename(
    columns={'fishID': 'ID', 'blend_age': 'ModalAge'}
).to_csv('/kaggle/working/final_predictions.csv', index=False)
print('Saved final_predictions.csv')

# 构造 API 提交载荷，reports 里每条记录给鱼号和年龄
final_payload = {
    'email': YOUR_EMAIL,
    'tblEventID': SUBMIT_EVENT_ID,
    'institute': YOUR_INSTITUTE,
    'nameAIAlgorithm': MODEL_NAME,
    'version': MODEL_VERSION,
    'reports': [
        {'ID': row['fishID'], 'ModalAge': int(row['blend_age'])}
        for _, row in submit_df.iterrows()
    ],
}
print('Ready to submit', len(final_payload['reports']), 'predictions')
print('Sample:', final_payload['reports'][:3])

# 确认无误后取消注释，向平台接口发送提交
# resp = requests.post(f'{BASE_URL}/submitFishAges',
#                      json=final_payload, timeout=60)
# print('Status:', resp.status_code)
# print('Result:', resp.json())
```

## 三、总结

本文中我们完成了 SmartDots AI Challenge 从数据到提交的完整流程。我们首先读取 Event_1842.json，确认物种为 Sole 7a，理解样本、鱼长与图片 URL 之间的组织方式，并用 GUID 把事件需求和磁盘文件对应起来；其次，我们复制已有图片、用 8 个线程并发下载缺失图片，把样本整理成 DataFrame，并统一了训练与验证的图像变换；接着，我们用 Von Bertalanffy 生长方程把鱼长转换成软标签，以 ResNet18 预训练骨干加回归头搭建模型，采用分组学习率、AdamW、余弦退火、MSE 损失和梯度裁剪训练 30 个 epoch，并按验证 MAE 保存最佳权重；然后，我们用原图、水平翻转、垂直翻转三种视图做 TTA，把平均后的 CNN 预测与生长公式基线做等权融合；最后，我们生成 ID 与 ModalAge 格式的 CSV 和 API 载荷，完成了可提交的预测文件。

这项工作也让我们看到几个必须正视的局限。第一，监督信号来自鱼长公式，而不是人工读龄真值，模型可能在拟合生长先验，而不是真正数清年轮；第二，训练集使用全部磁盘图片、验证集又取自提交图片，两者存在重叠，验证 MAE 偏乐观；第三，材料按图片逐行预测，如果同一条鱼有多张照片，提交前应按 fishID 聚合。后续改进可以围绕这三点展开：引入真实年龄标签、按 fishID 做分组划分、以及把图片级预测聚合成鱼级 ModalAge。即便有这些局限，这个项目仍然很好地展示了深度学习项目的基本骨架：先把数据弄干净，再把标签定义清楚，然后选择一个合适的预训练模型，最后用与业务一致的指标评估并提交。
