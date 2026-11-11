# 糖尿病视网膜病变识别 3（Diabetic Retinopathy Classification 3）

## 一、比赛简介

### 1.1 比赛背景

**糖尿病视网膜病变（Diabetic Retinopathy，简称 DR）** 是糖尿病最常见的并发症之一：长期高血糖会损伤眼底的微小血管，导致眼底出现微血管瘤、出血、渗出等症状，严重时会失明。眼科医生检查这种病，靠的是观察**眼底照片（retina image）**——一张用专用相机拍的、能看到眼底血管和视神经的彩色照片。对病人来说，早发现、早治疗能大幅降低失明风险；问题是眼底照片的阅片门槛高、医生人力宝贵，于是 Kaggle 前后举办了多届「根据眼底照片自动判断病变程度」的比赛。这场比赛编号为 **Diabetic Retinopathy Classification 3**，是其中一个序列版本。

难点在于：病变程度是分级的，医生打分本身就带主观性，相邻等级（比如「轻度」和「中度」）在图片上差别很细微，机器很容易把等级猜偏一档。

### 1.2 比赛目标

任务的输入是一张眼底照片，输出是 5 个等级中的一个，等级编码为 0～4：

- `0`：无病变；
- `1`：轻度；
- `2`：中度；
- `3`：重度；
- `4`：增殖性病变（最严重）。

训练数据是一张 `train.csv` 表格，每一行对应一张图片，主要两列：`id_code`（图片编号）和 `diagnosis`（病变等级，0～4）。图片按编号存放在 `train/` 目录，测试图片在 `test/` 目录。我们要做的是一个**5 分类的图像分类**问题，最后给每张测试图预测一个 0～4 的等级，按比赛要求的格式写进 `submission.csv`。

### 1.3 评价标准

这类糖尿病视网膜病变比赛使用的指标是**二次加权 Kappa（Quadratic Weighted Kappa，缩写 QWK）**，分数范围在 -1 到 1 之间，**越高越好**。它的特别之处在于：它衡量的是「预测等级和真实等级的一致性」，而且**相距越远的错误惩罚越重**——把 0 级预测成 4 级，比把 0 级预测成 1 级扣分多得多。这正好贴合医学场景：猜偏一档还可以接受，猜差三、四档是严重误诊。

---

## 二、解题过程

我们首先加载并观察眼底图片与标签分布（EDA），其次把图片切分为训练/验证集并做数据增强（清洗与预处理），接着用预训练的 ResNet18 训练 5 分类模型并用早停控制训练时长（训练与调优），最后在测试集上推理生成提交文件（测试报告）。

### 2.1 数据探索（EDA）

先读入标签表、列出图片文件，并随机抽一张图看看长什么样：

```python
import pandas as pd, numpy as np, cv2, matplotlib.pyplot as plt
from glob import glob

# 标签表：每行是图片编号 id_code 和病变等级 diagnosis
df = pd.read_csv("kaggle_retina/train.csv")
df.head()

# 找出所有训练图片和测试图片的路径
paths = glob('kaggle_retina/train/*.png')
paths_test = sorted(glob('kaggle_retina/test/*.png'))

# 随机取一条样本：用 cv2 读图，[:,:,::-1] 是把 BGR 翻回 RGB 顺序
idx = np.random.randint(0, len(df))
row = df.iloc[idx]
img = cv2.imread(f"kaggle_retina/train/{row['id_code']}.png")[:, :, ::-1]
plt.imshow(img)
print(f'label: {row["diagnosis"]}')
```

图片是彩色眼底照，能清楚看到血管和视盘。接着我们把所有标签取出来，统计每个等级各有多少张，画一根柱状图：

```python
all_cls = df['diagnosis'].tolist()

# 用 numpy 数每个等级出现的次数，便于画类别分布柱状图
cls, count = np.unique(all_cls, return_counts=True)
plt.bar(cls, count)
```

从分布能直接看出**类别严重不均衡**：大量样本集中在 `0`（无病变），少量样本散布在 1～4。这对模型是个信号——后面必须靠数据增强和合适的训练策略去补足少数类的样本量，不能指望模型「看一眼分布」就会自己学会。

### 2.2 数据清洗与预处理

图片本身不需要清洗噪声，我们重点做三件事：**统一尺寸、归一化、数据增强**。所有图先缩放到 224×224，再除以 255 把像素值压到 0～1。数据增强用的是 `imgaug` 库：随机水平翻转、垂直翻转、旋转 ±45 度、错切 ±16 度、缩放 0.8～1.2 倍。增强的意义在于，训练时每次抽到同一张图，看到的都是「变形」过的版本，相当于免费扩充样本量，也能让模型对旋转、翻转后的眼底图鲁棒（实际上检查拍摄时角度本来就不完全一致）。

```python
import imgaug.augmenters as iaa
from torch.utils.data import Dataset

class RetinaDataset(Dataset):
    def __init__(self, df, aug, img_size=224):
        self.df = df
        self.img_size = img_size
        # 一连串随机增强：翻转、旋转、错切、缩放
        self.transform = iaa.Sequential([
            iaa.Fliplr(0.5),                              # 50% 概率水平翻转
            iaa.Flipud(0.5),                              # 50% 概率垂直翻转
            iaa.Affine(rotate=(-45, 45),                  # 旋转 -45°～45°
                       shear=(-16, 16),                   # 错切
                       scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),  # 随机缩放
        ])
        self.aug = aug   # 训练集传 True，验证/测试集传 False

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(f"kaggle_retina/train/{row['id_code']}.png")[:, :, ::-1]
        img = cv2.resize(img, (self.img_size, self.img_size))   # 统一尺寸
        if self.aug:                                            # 训练才做增强
            img = self.transform.augment_image(image=img)
        img = img / 255.                                        # 归一化到 0~1
        # 转成 torch 张量：颜色通道从最后一个放到第一个，即 (C, H, W)
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        label = torch.tensor(row["diagnosis"], dtype=torch.long)
        return img, label
```

然后按 **8:2** 把标签表切成训练集和验证集，切分时固定随机种子（`random_state=5566`）保证可复现，验证集不增强、只用真实图片做评测：

```python
from sklearn.model_selection import train_test_split

# 80% 训练、20% 验证
train_df, val_df = train_test_split(df, test_size=0.2, random_state=5566)

# 训练集开增强，验证集不开；DataLoader 按 batch 送数据
train_ds = RetinaDataset(train_df, aug=True,  img_size=224)
val_ds   = RetinaDataset(val_df,  aug=False, img_size=224)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=64)
```

### 2.3 模型训练与超参数调优

模型我们直接翻了 **预训练好的 ResNet18**：它在 ImageNet（一千万张自然图片）上已经见过很多「边缘、纹理」特征，我们只把最后的全连接层换成输出 5 个类别的版本，然后从头微调。这比从零训练省力得多，也正是迁移学习的意义——眼底病变的某些纹理模式，和自然图像学到的底层特征是可以复用的。

```python
import torch
import torch.nn as nn
from torchvision import models

class RetinaClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = models.resnet18(pretrained=True)   # 用 ImageNet 预训练权重
        # 把最后的全连接层换成 5 个输出，对应 0~4 五个等级
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

model = RetinaClassifier(num_classes=5).to(device)
# 多分类用交叉熵损失，优化器用 Adam（自适应学习率，省心）
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
```

训练循环是最标准的 PyTorch 写法：前向传播算损失 → 反向传播算梯度 → 优化器更新参数。每个 epoch 里我们都把「整轮平均损失」和「整轮准确率」记下来，方便观察有没有过拟合：

```python
from tqdm.auto import tqdm

def train(dataloader, model, loss_fn, optimizer):
    model.train()                       # 切到训练模式
    total_loss, total_correct = 0, 0
    n = len(dataloader.dataset)
    for x, y in tqdm(dataloader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()           # 清空上一批的梯度
        pred = model(x)                 # 模型给出每个类别的分数
        loss = loss_fn(pred, y)         # 与真实标签算交叉熵损失
        loss.backward()                 # 反向传播求梯度
        optimizer.step()                # 按梯度更新一次参数
        total_loss += loss.item()       # 累加本批损失
        total_correct += (pred.argmax(dim=1) == y).sum().item()  # 本批猜对几个
    return total_loss / n, total_correct / n
```

训练一共设置了 100 轮，但配了**早停（early stopping）**：盯着验证集损失，如果连续 5 轮都不再下降，就认为继续训练只会过拟合，直接停。同时每轮都把验证损失最低的那次权重单独存成 `best.pth`，最后用的一定是「验证集上最好的那一版」，而不是最后一版：

跑训练循环之前，先写一个和 `train` 对称的验证函数——它只做前向计算、不更新参数，并关掉梯度记录：

```python
def test(dataloader, model, loss_fn):
    model.eval()                        # 切到评测模式（关掉 dropout、BN 统计固定）
    total_loss, total_correct = 0, 0
    n = len(dataloader.dataset)
    with torch.no_grad():               # 评测不需要梯度，省显存也更快
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += loss_fn(pred, y).item()
            total_correct += (pred.argmax(1) == y).sum().item()
    return total_loss / n, total_correct / n
```

```python
EPOCHS = 100
patience, counter, best_loss = 5, 0, float('inf')

for epoch in range(EPOCHS):
    train_loss, train_acc = train(train_loader, model, loss_fn, optimizer)
    val_loss,   val_acc   = test(val_loader,    model, loss_fn)
    torch.save(model.state_dict(), "last.pth")            # 每轮都存一份最新的
    if val_loss < best_loss:                              # 验证损失创新低
        counter = 0
        best_loss = val_loss
        torch.save(model.state_dict(), "best.pth")        # 覆盖保存最好的权重
    else:
        counter += 1
    if counter >= patience:                               # 连续 5 轮没进步
        print("Earlystop!")
        break
```

### 2.4 模型测试与指标报告

验证结束后，我们加载 `best.pth` 那版权重，对测试集逐步做推理。测试集没有标签，所以只需要把每张图的 softmax 输出中得分最高的类别挑出来，作为该图的等级预测。`argmax(dim=1)` 做的就是「取最大分数下标」，得到 0～4 的数字：

```python
# 用验证损失最低的那版权重做最终推理
model.load_state_dict(torch.load('best.pth'))
model.eval().to(device)

# 测试集不加增强，按 id_code 顺序逐批推理
test_df = pd.read_csv('kaggle_retina/submission.csv')   # 只有 id_code 列的模板
test_ds = TestDataset(test_df, img_size=224)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)

y_pred = []
with torch.no_grad():                        # 推理不需要梯度，省显存
    for x in test_loader:
        pred = model(x.to(device))
        y_pred.append(pred.argmax(dim=1))    # 取分数最高的类别
y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

# 按比赛格式写回 submission.csv：id_code + diagnosis
test_df['diagnosis'] = y_pred
test_df.to_csv('submission.csv', index=None)
```

虽然材料里没有记录这个 notebook 在排行榜上的最终 QWK 分数，但从训练-验证曲线能确认：训练准确率稳定上升、验证损失在早停机制下收敛，说明这个「ResNet18 + 增强 + 早停」的管线是健康收敛的。如果还想进一步提升 QWK，方向通常是：换更大的骨干网络（EfficientNet、ResNet50 等）、对少数类别多做补样、以及把损失函数换成对「相差一档」更宽容的加权交叉熵——因为 QWK 对相邻档的错误只轻微惩罚。

---

## 三、总结

本文中我们完成了糖尿病视网膜病变识别 3 的完整流程。我们首先在 EDA 里观察到眼底照片样本和 0～4 五个等级的分布，并发现数据存在严重的类别不均衡；其次我们做了数据预处理：统一缩放到 224×224、归一化到 0～1，并用 imgaug 对训练集做翻转、旋转、错切、缩放增强来扩充样本量；接着我们加载预训练的 ResNet18、替换最后的分类头，用交叉熵加 Adam 训练，并配了「连续 5 轮验证损失不降即停」的早停和最优权重保存；最后我们在测试集上逐批推理、取 argmax 生成 `submission.csv` 提交文件。

这场比赛给我们的最大启发是：**面对小样本和类别不均衡的医学图像任务，图像增强和早停比频繁调超参数更管用**。增强帮我们把几百张少数类图片变成上千个不同的「切片」，早停避免我们为刷训练集准确率而白白把模型练废；而 5 分类跨度大、相邻等级难分的特点，也提醒我们再进一步时要从「分类头设计」和「损失函数」上做文章。