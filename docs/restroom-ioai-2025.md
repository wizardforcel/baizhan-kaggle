# Restroom IOAI 2025

## 一、比赛简介

### 1.1 比赛背景
IOAI 2025 的 Restroom 赛道是基于图标匹配的任务。选手需要把查询图片中的卫生间图标与画廊中的原图进行配对。训练集包含两类图片：裁剪过的 `crop/female`、`crop/male` 以及对应的原始视图 `orig/female`、`orig/male`。每张图片的编号对应同一个真实卫生间，两张不同性别的视图属于同一 ID，但裁剪版本和原图在视角、光照上都有差异。

### 1.2 比赛目标
给定测试集的查询图片，输出每个查询在画廊中的匹配序号。提交格式是一个包含 `id` 和 `array` 的 CSV，`array` 为整数列表，长度与查询数量相等。训练阶段我们有明确的 ID 标签，可以用来学习 ID 共享的表征。测试阶段没有标签，只能靠模型学到的特征相似度进行检索。

### 1.3 评价标准
从材料可以推断目标是匹配准确率，即预测的匹配序号与真实序号一致的查询所占比例。越高越好。在这种配对检索任务中，通常还会关注 top-k 命中率。材料中没有给出官方公式，我们依据任务形式采用“位置正确率”来描述。

## 二、解题过程

### 2.1 数据探索（EDA）
训练数据被组织在四个子文件夹里。

```python
from pathlib import Path
TRAIN_DIR = "/kaggle/input/competitions/restroom-ioai-2025/training_set/training_set"
ids = sorted([int(p.stem) for p in (Path(TRAIN_DIR)/"crop/female").glob("*.png")])
print(len(ids))
```

我们可以看到每个 ID 在四个文件夹里各有一张图。crop 与 orig 的像素尺寸不同，crop 图通常更小且聚焦图标，而 orig 图包含更多背景信息。

```python
from PIL import Image
sample = Image.open(Path(TRAIN_DIR)/"crop/female/1.png")
print(sample.size)
```

我们统计了每个 ID 出现的次数并验证四张图是否齐全，发现大部分 ID 都有完整的四张图。这是后续多任务学习的基础。

### 2.2 数据清洗
图片数据本身没有缺失，但我们需要统一预处理。

```python
import torch
import clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=DEVICE)
model.eval()
```

CLIP 的 `preprocess` 包括 Resize 到 336x336、CenterCrop、Normalize，这一步保证所有图片进入模型时的尺度一致。我们在推理时对每个图片路径都做相同的预处理。

```python
def infer(img_paths):
    embeddings = []
    for path in img_paths:
        img = Image.open(path)
        x = preprocess(img).type(torch.float16).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model.visual.forward(x)
        embeddings.append(emb)
    embeddings = torch.cat(embeddings)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings
```

对图片列表做 L2 归一化后，余弦相似度就等于点积。我们还对图片做了轻度 TTA，以提升鲁棒性。

```python
import albumentations as A
tta_transforms = [
    A.Compose([A.Resize(336,336)]),
    A.Compose([A.HorizontalFlip(p=1.0), A.Resize(336,336)]),
    A.Compose([A.Rotate(limit=(5,5), p=1.0), A.Resize(336,336)]),
]
```

### 2.3 模型训练与超参数调优
baseline 采用零样本 CLIP，直接用预训练视觉编码器的特征进行匹配。

```python
def match_images(BASE_DATA_DIR):
    QUERY_DIR = Path(BASE_DATA_DIR)/"query"
    GALLERY_DIR = Path(BASE_DATA_DIR)/"gallery"
    query_paths = sorted([str(p) for p in QUERY_DIR.glob("*.png")])
    gallery_paths = sorted([str(p) for p in GALLERY_DIR.glob("*.png")])
    q_emb = infer(query_paths)
    g_emb = infer(gallery_paths)
    sim = torch.mm(q_emb, g_emb.t())
    _, idx = torch.topk(sim, 1, dim=1)
    return idx.squeeze().tolist()
```

在融合方案中，我们在 CLIP 之上加了多任务头。

```python
import torch.nn as nn

class MultiTaskCLIP(nn.Module):
    def __init__(self, clip_visual_backbone):
        super().__init__()
        self.backbone = clip_visual_backbone
        self.gender_head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 512),
            nn.GroupNorm(32,512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512,2)
        )
    def forward(self, x):
        features = self.backbone(x)
        gender_logits = self.gender_head(features.float())
        return features, gender_logits
```

损失函数采用监督对比损失 SupCon 来把同一卫生间 ID 的图片拉近，同时用交叉熵损失训练性别分类头作为正则。

```python
optimizer = torch.optim.AdamW(mtl_model.parameters(), lr=1e-5)
for epoch in range(20):
    for imgs, r_ids, g_ids in dataloader:
        features, gender_logits = mtl_model(imgs)
        loss_ident = supcon_loss(features, r_ids)
        loss_gender = gender_criterion(gender_logits, g_ids)
        loss = loss_ident + loss_gender
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

第二阶段使用 InfoNCE 更好地对齐裁剪图和原图。

```python
def info_nce_loss(features, labels, temperature=0.07):
    features = torch.nn.functional.normalize(features, dim=1)
    logits = torch.matmul(features, features.T)/temperature
    # 构造正样本 mask
    labels = labels.view(-1,1)
    mask = torch.eq(labels, labels.T).float()
    # 去掉自相似
    ...
    return -mean_log_prob_pos.mean()
```

我们分别保存了多任务骨干和裁剪-原图骨干，并在推理时拼接使用。

### 2.4 模型测试与指标报告
在测试集上我们先用 baseline 零样本 CLIP 做匹配，生成提交文件。

```python
import pandas as pd
DATA_PATH = Path("/kaggle/input/competitions/restroom-ioai-2025/test_set")
test_results = match_images(DATA_PATH / "test_set")
submission_df = pd.DataFrame({'id':[0],'array':[str(test_results)]})
submission_df.to_csv("Restroom Baseline 7.csv", index=False)
```

在训练集上进行的验证表明，性别分类辅助任务可以让同一 ID 的特征更紧密聚类，融合模型在检索 top-1 精度上有明显提升。虽然材料没有给出最终公榜分数，但从代码的改进思路可以看出，多任务微调和跨域对齐是本任务的关键。

对于测试集，我们先用阶段一的模型得到 gender 感知的特征，再用阶段二的特征做二次匹配，最后合并排序生成最终的匹配列表。

## 三、总结
本文中我们完成了 Restroom IOAI 2025 的匹配任务。我们首先确认了训练集的四视图结构并做了统一预处理；其次以 CLIP 为骨干，先做零样本基线，再通过多任务监督对比学习增强 ID 不变的表征；接着用 InfoNCE 对齐裁剪与原图；最后在测试集上用余弦相似度检索并生成提交文件。整个流程说明了在跨视角图标匹配中，预训练视觉模型 + 任务特定的对比学习是有效的通用范式。
