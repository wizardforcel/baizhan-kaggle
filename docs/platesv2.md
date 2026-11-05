# PlatesV2：干净盘子与脏盘子图像分类

## 一、比赛简介

### 1.1 比赛背景

本比赛在 Kaggle 上的 slug 是 `platesv2`，材料中的 notebook 多把它称作“Cleaned vs Dirty V2”。任务很直观：给一张盘子的照片，判断它是干净的还是脏的。数据以 `plates.zip` 的形式提供，解压后得到 `train/cleaned`、`train/dirty` 和 `test` 三个目录。`train` 下面按类别分子文件夹，`test` 下面直接放图片，没有标签。

```python
# 解压后的目录结构大致如下
# plates/
# ├── train/
# │   ├── cleaned/     # 干净盘子，有标签
# │   └── dirty/       # 脏盘子，有标签
# └── test/            # 测试图片，没有标签
```

从材料里的 253 个页面可以看到，这是一个有很多初学者参与的图像二分类比赛。大家用的方法从简单 CNN、ResNet18、ResNet152、VGG16，到 EfficientNet、ViT、DINOv2、YOLO 都有；也有人尝试背景移除、盘子裁剪、半监督、零样本 VQA 等技巧。这个现象说明：比赛本身不难理解，但想把准确率做高，需要在数据清洗、增强和模型选择上花功夫。

材料中的 EDA 还提到两个细节。第一，训练集的两个类别数量相同，类别是平衡的；第二，大多数图片尺寸一致，而且测试集图片数量比训练集还多。测试集更大意味着我们不能只靠“记住训练集”取胜，模型必须真的学会区分干净和脏。另一个细节是数据里混有 macOS 生成的 `.DS_Store` 文件，这类文件不是图片，如果不删掉，后面用 `ImageFolder` 读数据时会报错或者把类别数弄乱。

### 1.2 比赛目标

我们要训练一个模型，输入一张盘子图片，输出 `cleaned` 或 `dirty` 两个标签之一。对训练集，我们有图片和标签；对测试集，我们只有图片。最终要把测试集每张图片的预测结果写成 CSV 提交。

用大白话说，这是一个“看图说话”的任务，只不过答案只有两个词。模型要学会忽略背景、光线、拍摄角度这些与“脏不脏”无关的东西，把注意力放在盘子表面的油渍、食物残渣、水渍等线索上。材料里不少 notebook 都围绕这个思路做文章：有的先把背景抠掉，只留下盘子；有的把盘子区域裁出来；有的用很强的数据增强，让模型见过更多光照和角度。

我们的目标不是复现某一个具体 notebook 的分数，而是把材料里反复出现的流程整理成一条可运行的基线：先看清数据，再做清洗和划分，然后用迁移学习训练，最后在验证集上选模型并生成提交文件。

### 1.3 评价标准

比赛是二分类，材料中的 notebook 主要用准确率（Accuracy）来比较结果。准确率的定义很简单：预测正确的图片数除以图片总数。公式是：

```python
# 准确率：预测正确的图片数 / 总图片数
def accuracy(y_true, y_pred):
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return correct / len(y_true)
```

从索引里的标题可以看到，公开 notebook 报告过 0.91、0.913、0.932、0.934、0.95026 等分数，说明大家确实把准确率当作主要指标。材料没有展开官方评分页的细节，所以我们在本文中统一用验证集准确率来选模型；测试集没有标签，本地无法计算真实准确率，只能生成提交文件后看排行榜。因为两个类别数量相同，准确率不会被多数类“带偏”，用它来比较模型是比较合适的。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先把压缩包解压到可写目录。Kaggle 的 `/kaggle/input` 是只读的，所以不能直接在原目录里删文件或移动图片；材料中的 notebook 也是先解压到 `/kaggle/working`。解压后我们删除 `.DS_Store`，再确认 `train/cleaned`、`train/dirty`、`test` 三个目录都在。

```python
import os
import zipfile

# 材料中有两种挂载路径：/kaggle/input/plates.zip
# 或 /kaggle/input/competitions/platesv2/plates.zip，运行时按实际路径改
zip_path = '/kaggle/input/plates.zip'
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall('/kaggle/working/')

data_root = '/kaggle/working/plates'
print('根目录内容：', os.listdir(data_root))

# 删除 macOS 自动生成的 .DS_Store，避免它被当成图片或类别
for dir_path, _, file_names in os.walk(data_root):
    for name in file_names:
        if name == '.DS_Store':
            os.remove(os.path.join(dir_path, name))
            print('已删除：', os.path.join(dir_path, name))
```

接下来我们统计类别数量和图片尺寸。这个步骤看起来简单，但很重要：如果两个类别数量差很多，就不能只看准确率；如果图片尺寸五花八门，就要统一缩放或裁剪。材料里的统计结果是两类数量相同、大多数图片尺寸一致，所以我们后面可以放心地用 `Resize` 或 `RandomResizedCrop` 把图片变成 224×224。

```python
from collections import Counter
import cv2

train_dir = os.path.join(data_root, 'train')
test_dir = os.path.join(data_root, 'test')
class_names = ['cleaned', 'dirty']

# 1. 看类别是否平衡
for c in class_names:
    n = len(os.listdir(os.path.join(train_dir, c)))
    print(f'{c}: {n} 张')

print('test:', len(os.listdir(test_dir)), '张')

# 2. 统计图片尺寸，看看是否需要统一缩放
sizes = Counter()
for c in class_names:
    d = os.path.join(train_dir, c)
    for f in os.listdir(d):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = cv2.imread(os.path.join(d, f))
        if img is not None:
            sizes[img.shape] += 1

print('最常见的尺寸：', sizes.most_common(5))
```

我们还会随机抽几张图看一看。很多“脏”并不是整张图都脏，而是盘子上有一小块油渍；如果只做全局平均池化，模型可能忽略小区域。这个观察解释了为什么材料里有人做盘子裁剪、背景移除，甚至用目标检测先找盘子。EDA 的目的不是画几张好看的图，而是决定后面数据清洗和增强的方向。

### 2.2 数据清洗

数据清洗分三件事：删垃圾文件、划验证集、处理背景。

第一件已经在 EDA 里做了。第二件是划验证集。材料中有两种做法：一种是把约 1/6 的图片从训练集“移动”到验证集；另一种是“复制”到验证集，保留原训练集。我们更推荐复制，因为复制不会破坏原始数据，后面想重新划分也方便。注意要先划分，再做增强；如果先增强再划分，同一张原图的不同副本可能同时出现在训练集和验证集，造成数据泄漏，验证分数会虚高。

```python
import random
import shutil

random.seed(42)
val_dir = os.path.join(data_root, 'val')

for c in class_names:
    os.makedirs(os.path.join(val_dir, c), exist_ok=True)
    src = os.path.join(train_dir, c)
    files = sorted(os.listdir(src))
    random.shuffle(files)

    n_val = len(files) // 6  # 约 1/6 做验证
    for f in files[:n_val]:
        shutil.copy(os.path.join(src, f), os.path.join(val_dir, c, f))

    print(c, '训练', len(files) - n_val, '验证', n_val)
```

材料作者提到，按 1/6 划分后验证集只有 6 张，远不够调参，所以他想用增强来扩充验证集。这个想法可以理解，但要小心：增强副本只能用来让指标更稳定，不能把同一张原图的副本同时放进训练集。更稳妥的做法是做 5 折交叉验证，或者多换几个随机种子划验证集，看模型排名是否稳定。

第三件是背景处理。材料里有一篇 baseline 用 OpenCV 的 GrabCut 把盘子以外的背景变成白色。GrabCut 的思想是：我们先画一个矩形，告诉算法“盘子大概在这个框里”，算法再根据颜色和边缘把前景和背景分开。这样做的目的是减少桌布、桌面、餐具等背景的干扰，让模型更关注盘子本身。

```python
import numpy as np

def remove_background(img_path, out_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # GrabCut 需要一张掩码，0 是背景，1 是前景，2/3 是可能的背景/前景
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 假设盘子大致在画面中间，留出 15 像素边距
    rect = (15, 15, w - 30, h - 30)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 10, cv2.GC_INIT_WITH_RECT)

    # 只保留确定前景和可能前景
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    fg = img * mask2[:, :, None]

    # 把背景里非黑色的像素涂白，得到白底图
    bg = img - fg
    bg[np.where((bg > [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    out = bg + fg

    cv2.imwrite(out_path, out)
```

不过 GrabCut 不是万能的。如果盘子颜色和桌布很接近，或者盘子上有反光，掩码可能把盘子边缘也切掉。材料里也有人不做背景移除，直接训练，效果同样不错。所以我们的策略是：先把“原图”作为主实验，把“去背景图”作为对照实验；只有在验证集稳定提升时才采用。不能因为某张图看起来更干净就认定它一定更好，一切以验证集为准。

另外，材料里有 notebook 对每张训练图做 40 倍离线增强，把数据集扩大。这个做法能增加样本，但也有两个风险：一是磁盘占用大，二是同一张图的增强副本如果划分不当会造成泄漏。我们更建议训练时在线增强，验证集和测试集只做确定性的缩放和归一化。下面这段代码只演示“一张原图生成多张增强图”的写法，实际使用时把 `copies` 调小，并且只对训练集做。

```python
from PIL import Image
from torchvision import transforms

aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])

# 以材料中的“一张原图生成多张增强图”为例
# 注意：必须先划分训练/验证，再只对训练集做增强
def offline_augment(src_dir, dst_dir, copies=5):
    os.makedirs(dst_dir, exist_ok=True)
    for c in class_names:
        os.makedirs(os.path.join(dst_dir, c), exist_ok=True)
        for f in os.listdir(os.path.join(src_dir, c)):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = Image.open(os.path.join(src_dir, c, f)).convert('RGB')
            img.save(os.path.join(dst_dir, c, f))  # 保留原图
            stem = os.path.splitext(f)[0]
            for i in range(copies):
                aug_img = aug(img)
                aug_img.save(os.path.join(dst_dir, c, f'{stem}_aug{i}.jpg'))

# offline_augment(train_dir, 'train_augmented', copies=5)
```

### 2.3 模型训练与超参数调优

我们采用迁移学习。大白话就是：先用别人在海量图片上训练好的网络（比如 ResNet18），再把最后一层改成“干净/脏”两类。网络前面那些层已经学会了识别边缘、纹理、颜色，我们只需要用比赛数据微调，通常比从零训练快得多，也更不容易过拟合。材料里大量 notebook 都用了 ResNet18、ResNet50、ResNet152、VGG16 等预训练模型，说明迁移学习是这个比赛的主流做法。

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# 训练集：随机裁剪、翻转、旋转、颜色扰动，让模型见过更多变化
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 验证集/测试集：只做确定性的缩放、裁剪和归一化，不能随机
val_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(train_dir, train_tf)
val_ds = datasets.ImageFolder(val_dir, val_tf)

# ImageFolder 会按文件夹名排序，cleaned 在前、dirty 在后
assert train_ds.class_to_idx == {'cleaned': 0, 'dirty': 1}

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                        num_workers=2, pin_memory=True)

print('类别到编号：', train_ds.class_to_idx)
print('训练批次数：', len(train_loader))
```

这里的归一化均值 `[0.485, 0.456, 0.406]` 和标准差 `[0.229, 0.224, 0.225]` 是 ImageNet 的统计值。因为我们用的是 ImageNet 预训练模型，输入也必须按同样的方式归一化，否则预训练权重学到的特征会对不上。材料里还特别提醒“增强要柔和”，所以旋转只取 10 度，颜色扰动也只取 0.3，避免把“脏”的细节增强没了。

接下来定义模型、损失函数和优化器。ResNet18 比较轻，适合做第一条基线；如果显存和时间允许，可以再换 ResNet50 或 ResNet152 做对比。

```python
import torch.nn as nn
from torchvision import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载 ImageNet 预训练的 ResNet18，并把最后的全连接层改成 2 类
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
```

训练循环里我们做四件事：前向传播、计算损失、反向传播、更新参数。每个 epoch 结束后在验证集上算准确率，保存最好的模型；如果连续多个 epoch 验证集不再提升，就提前停止。材料里的训练函数还用了梯度裁剪，把梯度的范数限制在 1.0 以内，防止训练初期梯度爆炸。

```python
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total

best_acc = 0.0
best_state = None
patience, wait = 12, 0

for epoch in range(10):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    val_acc = evaluate(model, val_loader)
    scheduler.step()
    print(f'epoch {epoch + 1}: val_acc={val_acc:.4f}')

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print('验证集不再提升，提前停止')
            break

if best_state is not None:
    model.load_state_dict(best_state)
print('最佳验证准确率：', best_acc)
```

超参数调优方面，材料里的交叉验证 notebook 引入了 `optuna`，说明作者把自动搜索纳入了流程。我们可以围绕学习率、batch size 和权重衰减做一个小范围搜索。每次试验只跑少量 epoch，选出最好的配置后，再用完整 epoch 重新训练。这样做的原因是：验证集很小，搜索次数太多容易“记住”验证集，分数看起来高，换到测试集就掉下来。

```python
import optuna

def objective(trial):
    # 学习率按对数尺度采样，避免只在小范围里打转
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # 复用上面的模型和训练循环，封装成 build_model 与 train_and_validate
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    best_val_acc = train_and_validate(model, optimizer,
                                      batch_size=batch_size,
                                      epochs=3)
    return best_val_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)
print('最佳参数：', study.best_params)
```

如果不想引入 Optuna，也可以手工做网格搜索，比如学习率试 `1e-4`、`3e-4`、`1e-3`，batch size 试 32 和 64。关键不是工具，而是每次只改一个变量，并且用同一份验证集比较。还要注意，数据增强的强度本身也是超参数：增强太弱容易过拟合，增强太强会把油渍、残渣这些关键细节破坏掉，验证集准确率反而下降。

### 2.4 模型测试与指标报告

训练完成后，我们在验证集上报告准确率，并看混淆矩阵。准确率只告诉我们“对了多少”，混淆矩阵还能告诉我们“错在哪”：是把脏盘子判成干净，还是把干净盘子判成脏。对洗碗场景来说，把脏盘子放过去通常比多洗一次干净盘子更严重，所以如果两类错误代价不同，可以调整判定阈值，而不是永远取概率最大的那一类。

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 在验证集上收集真实标签和预测标签
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        pred = model(x).argmax(dim=1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.tolist())

print('验证集准确率：', accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=class_names))
print('混淆矩阵：')
print(confusion_matrix(y_true, y_pred))
```

测试集没有标签，所以不能本地计算真实准确率。我们能做的是：用和验证集完全一样的预处理，对测试集逐张预测，然后检查提交文件的行数、列名和标签取值。测试目录下面没有类别子文件夹，因此不能直接用 `ImageFolder`，需要自己写一个简单的 `Dataset`。

```python
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class TestDataset(Dataset):
    def __init__(self, root, tf):
        self.root = root
        self.tf = tf
        # 只保留图片文件，避免把隐藏文件或说明文件读进来
        self.files = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.root, name)).convert('RGB')
        return self.tf(img), name

test_ds = TestDataset(test_dir, val_tf)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                         num_workers=2, pin_memory=True)

model.eval()
rows = []
with torch.no_grad():
    for x, names in test_loader:
        x = x.to(device)
        pred = model(x).argmax(dim=1).cpu().numpy()
        for name, p in zip(names, pred):
            rows.append((name, class_names[p]))

# 列名要以比赛提供的 sample_submission.csv 为准，这里用 image/label 示意
sub = pd.DataFrame(rows, columns=['image', 'label'])
sub.to_csv('submission.csv', index=False)

# 提交前做三项基本检查
assert len(sub) == len(test_ds)
assert set(sub['label']).issubset(set(class_names))
assert sub['image'].is_unique
print(sub.head())
```

从材料里的公开 notebook 标题看，大家的分数大致在 0.91 到 0.95 之间。这个区间说明两件事：第一，二分类本身不算特别难，迁移学习加合理增强就能拿到不错的基线；第二，想继续往上提升会越来越难，因为剩下的错误往往来自标注不一致、反光、盘子颜色接近背景、脏的区域太小等“人眼也犹豫”的样本。我们在报告结果时应该同时写清楚用的是哪份验证集、有没有做背景移除、有没有用 TTA，否则不同 notebook 的分数不能直接比较。

## 三、总结

本文中我们围绕 PlatesV2 的“干净 vs 脏”二分类任务，走完了一条完整的基线流程。我们首先解压数据、删除 `.DS_Store` 等垃圾文件，统计类别数量和图片尺寸，确认两个类别平衡、大多数图片尺寸一致；其次按约 1/6 的比例划出验证集，强调先划分再增强，避免同一张图的不同副本跨集合造成数据泄漏；接着用 GrabCut 做背景移除的对照实验，并用随机裁剪、翻转、旋转、颜色扰动等在线增强提高模型的泛化能力；然后采用 ImageNet 预训练的 ResNet18 做迁移学习，只替换最后的全连接层，配合交叉熵损失、AdamW、余弦学习率调度、梯度裁剪和早停训练；最后在验证集上计算准确率、混淆矩阵和分类报告，并用自定义 Dataset 对无标签的测试集做推理，生成提交 CSV。

回顾整个过程，最重要的不是把模型换得多复杂，而是把数据边界守好：训练集、验证集、测试集不能混；增强只能作用在训练集；背景移除要用验证集证明有效；超参数搜索不能在小验证集上反复“刷分”。材料里的公开 notebook 分数大多在 0.91 到 0.95 之间，说明这个任务的上限受数据质量、拍摄条件和标注一致性影响很大。后续如果继续提升，可以尝试更大的 ResNet、EfficientNet、ViT 或 DINOv2，做 5 折交叉验证和 TTA，或者先把盘子区域裁剪出来再分类。但无论用什么模型，都应该保留一条简单、可复现的 ResNet18 基线，用它来判断新方法到底有没有真的变好。
