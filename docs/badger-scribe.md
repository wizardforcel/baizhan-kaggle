# Badger Scribe 档案手稿逐字转录

## 一、比赛简介

### 1.1 比赛背景

Badger Scribe 是一个**历史手稿转录**比赛：给一张扫描的档案页图片，要求输出这一页上所有文字的逐字抄录（verbatim transcription）。"逐字"是关键词——不能把古体拼写改成现代写法，不能补全省略的字母，也不能把看不清的地方写成"此处不可辨认"。

数据来自威斯康星大学（UW）的三个馆藏，对应三种完全不同的书写传统：

- `kade_letters`：德文商业信件，用的是 **Kurrent** 手写体。这是一种 19 世纪德语地区流行的连笔字，和今天印刷体的差别大到"看起来像另一种文字"；
- `dominy_accounts`：账本，大量表格、数字、金额，靠横竖线划分格子；
- `survey_notes`：测量员的野外笔记，既有连笔英文，也有手绘的表格和草图。

比赛还有两条硬约束：**所有模型必须是开放权重**（不能调用闭源 API），**整条流水线要能在单张 96 GB 显存的 GPU 上跑完**。

这个任务的难点不在"识别单个字母"，而在**版面**：一页纸上哪里是正文、哪里是表格、先读哪一块、表格的格子怎么切。材料里说得很清楚：正文类的手稿，识别质量是主要误差来源；而账本和测量笔记的表格部分，**版面分析和阅读顺序才是主要误差来源**。至于 Kurrent，拿一个英文识别器去微调根本不够，必须换一个真正见过这种字体的模型。

### 1.2 比赛目标

输入是一张页面图片（JPG）以及它的元数据：`page_id`（页面编号）、`doc_id`（文档编号，同一封信的多页共享一个）、`category`（三个馆藏之一）。输出是 `page_id` 和 `text` 两列，`text` 是整页的转录结果。

训练集 `train.csv` 有 `page_id`、`doc_id`、`category`、`text` 四列，`text` 里用换行符分隔每一行文字；测试集没有 `text`，需要我们预测。提交文件的格式就是 `page_id,text`。

### 1.3 评价标准

比赛使用**字符错误率 CER**（Character Error Rate），并且做了两个约定：

1. **空白折叠**：先把连续空白压成一个空格再比较，所以用空格还是换行分隔表格单元格不影响分数；
2. **按类别宏平均**：三个 `category` 各算一个 CER，再取平均。这样三个馆藏权重相同，不会因为某个馆藏页数多就主导总分。

CER 的算法是"编辑距离除以参考答案的字符数"，即把预测改成正确答案最少要插入、删除、替换多少个字符。所以 **CER 越低越好，0 表示一字不差**。材料里也同时报告 WER（词错误率）作为参考，但排行榜看的是 CER。

---

## 二、解题过程

我们的主线是搭一条**五段式流水线**：版面分区（正文 / 表格）→ 正文行切分 → 表格单元格切分 → 按馆藏路由到对应语言分支 → 用 CRNN/CTC 逐行识别。每一段都可以单独测量、单独替换，这样分数不好时能立刻知道是哪一段出了问题。

### 2.1 数据探索（EDA）

我们首先看清数据长什么样。`train.csv` 是页级的：一行一页，`text` 里用换行分隔每一行文字，但**没有给出每一行的坐标**。这意味着训练识别器时，要么用外部带行级标注的语料，要么自己先切行、再和 `text` 按从上到下的顺序一一配对。

同时我们建立了一张"馆藏 → 语言分支"的路由表。`kade_letters` 走德语，另外两个走英语。这不是取巧：真实档案馆在数字化时本来就知道每一页属于哪个馆藏，按馆藏路由是完全正当的生产做法。

```python
from pathlib import Path
import pandas as pd

DATA_DIR = Path('/kaggle/input/badger-scribe')
IMAGES_DIR = DATA_DIR / 'images'

# 馆藏 -> 识别器语言分支；Stage 4 的路由完全依赖这张表
LANG_ROUTE = {
    'kade_letters': 'de',
    'survey_notes': 'en',
    'dominy_accounts': 'en',
}

train = pd.read_csv(DATA_DIR / 'train.csv')
print(train['category'].value_counts())        # 三个馆藏各有多少页
print(train['doc_id'].nunique())               # 文档级分组，切验证集时要按它分
print(train[['page_id', 'category']].head())

# 每页的“标准答案行数”，用来判断我们的切行器是否和真值对齐
train['n_lines'] = train['text'].fillna('').str.split('\n').apply(
    lambda lines: sum(1 for ln in lines if ln.strip()))
print(train.groupby('category')['n_lines'].describe())
```

这里最重要的是意识到**同一文档的多页不能分到训练和验证两边**。同一封信的笔迹、纸张、扫描条件都一样，如果随机按页切分，验证集分数会虚高。所以后面所有留出都按 `doc_id` 分组。

### 2.2 数据清洗

扫描件的"清洗"不是去标点，而是**把纸面变成黑白两色，再把版面结构找出来**。我们用了两个基础操作。

第一个是自适应二值化。档案页的照明很不均匀，有的角落发黄、有的地方褪色，全局阈值会把一半页面判成空白。自适应阈值按局部邻域算阈值，能适应这种明暗变化。

第二个是**版线检测**：表格的横线和竖线又长又直，而手写的笔画短且弯。用一根很长的横向核做形态学开运算，只有长度超过页面宽度 35% 的横线能留下来；竖线同理。留下来的就是表格线。

```python
import cv2
import numpy as np

pages = sorted(IMAGES_DIR.glob('*.jpg'))

def binarize(img):
    """自适应阈值：应对档案页的不均匀照明与褪色"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, blockSize=51, C=15)

def ruling_lines(binary, min_line_frac=0.35):
    """用长核开运算分离横/竖版线：只有足够长且笔直的结构能存活"""
    h, w = binary.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(w * min_line_frac)), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(h * min_line_frac))))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    return horizontal, vertical

img = np.array(Image.open(pages[0]))
binary = binarize(img)
h_lines, v_lines = ruling_lines(binary)
print('横向版线像素:', int(np.count_nonzero(h_lines)),
      '纵向版线像素:', int(np.count_nonzero(v_lines)))
```

有了版线信号，版面分区就变成一个很朴素的问题：先把页面按"接近空白的行"切成横向条带，再对每个条带算"版线像素占面积的比例"，超过阈值就判为表格，否则判为正文。

```python
def propose_regions(binary, min_gap=20, min_height=30):
    """按近空白行把页面切成横向条带，返回 (top, bottom) 列表"""
    row_ink = binary.sum(axis=1)
    blank_threshold = row_ink.max() * 0.01 if row_ink.max() else 0
    is_blank = row_ink < blank_threshold

    bands, start, gap = [], None, 0
    for y, blank in enumerate(is_blank):
        if not blank:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:                 # 连续空白足够长才算真正的分隔
                end = y - gap + 1
                if end - start >= min_height:  # 太薄的条带多半是噪声
                    bands.append((start, end))
                start = None
    if start is not None:
        bands.append((start, len(is_blank)))
    return bands

def classify_region(region_binary, ruling_frac_threshold=0.02):
    """表格区域的版线像素占比明显更高；阈值要按实际扫描分辨率微调"""
    h_lines, v_lines = ruling_lines(region_binary)
    frac = (np.count_nonzero(h_lines) + np.count_nonzero(v_lines)) / region_binary.size
    return 'table' if frac > ruling_frac_threshold else 'text'
```

### 2.3 模型训练与超参数调优

接下来是识别。正文区域用**水平投影**切行：把每一行的黑色像素数加起来，形成一条"墨水曲线"，曲线的高峰就是文字行，低谷就是行间空白。表格区域则用版线位置求笛卡尔积：横线给出行的边界，竖线给出列的边界，两两组合就得到每个单元格。

```python
def segment_lines(binary, min_height=12):
    """水平投影切行：墨水密度低于阈值的位置就是行间空白"""
    profile = binary.sum(axis=1)
    threshold = profile.max() * 0.05 if profile.max() else 0
    bands, start = [], None
    for y, v in enumerate(profile):
        if v > threshold and start is None:
            start = y
        elif v <= threshold and start is not None:
            if y - start >= min_height:
                bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(profile)))
    return bands

def _line_positions(hits):
    """把一条几像素粗的版线压成一个中心坐标"""
    positions, start = [], None
    for i, hit in enumerate(hits):
        if hit and start is None:
            start = i
        elif not hit and start is not None:
            positions.append((start + i - 1) // 2)
            start = None
    if start is not None:
        positions.append((start + len(hits) - 1) // 2)
    return positions

def table_grid(region_binary):
    """横线位置 = 行边界，竖线位置 = 列边界，两端补上区域边缘"""
    h_lines, v_lines = ruling_lines(region_binary)
    h, w = region_binary.shape
    rows = sorted(set([0, h - 1] + _line_positions((h_lines > 0).any(axis=1))))
    cols = sorted(set([0, w - 1] + _line_positions((v_lines > 0).any(axis=0))))
    return rows, cols
```

识别器用 CRNN + CTC。CRNN 的意思是"卷积 + 循环网络"：卷积层从图像里提取局部特征，循环网络（这里是双向 LSTM）沿着宽度方向读出一串特征，最后每个时间步输出一个字符概率。CTC 是一种损失函数，它允许模型在不知道"第几个字符对应图像哪一列"的情况下训练，只要求最终解码出来的字符串和答案一致——正好适合行级手写识别这种"没有字符级对齐"的场景。

网络的宽度方向保留为序列维度，高度方向用自适应池化压成 1，所以任意高度的行图都能输入。训练时把同批次的变宽图片补零到最大宽度，CTC 会按每张图的实际宽度计算有效长度。

```python
class CRNN(nn.Module):
    """CNN 提特征 -> BiLSTM 读序列 -> 全连接输出字符概率"""
    def __init__(self, num_classes, cnn_channels=128, rnn_hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 宽高各减半
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 再减半
            nn.Conv2d(64, cnn_channels, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),      # 高度压成 1，宽度保留为时间步
        )
        self.rnn = nn.LSTM(cnn_channels, rnn_hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(rnn_hidden * 2, num_classes)
        self.width_downsample = 4                 # 两次 stride-2 池化，宽度缩 4 倍

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)   # (B, W', C)
        seq, _ = self.rnn(f)
        logits = self.head(seq)                       # (B, W', 类别数)
        return logits.permute(1, 0, 2)                # CTCLoss 要求 (T, B, C)

def train_recognizer(dataset, num_epochs=5, batch_size=16, lr=1e-3):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_lines, num_workers=2)
    model = CRNN(num_classes=len(dataset.charset) + 1).to(device)  # +1 给 CTC 空白符
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()
    for epoch in range(num_epochs):
        total = 0.0
        for images, targets, input_widths, target_lengths in loader:
            logits = model(images.to(device))
            log_probs = logits.log_softmax(2)
            # 输入长度必须按 CNN 下采样后的宽度算，且留在 CPU 上
            input_lengths = torch.clamp(input_widths // model.width_downsample, min=1)
            loss = ctc_loss(log_probs, targets.to(device), input_lengths, target_lengths)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
        print(f'epoch {epoch + 1}: loss {total / len(loader):.4f}')
    return model
```

训练数据分两路。英语分支用 Bentham 数据集（18～19 世纪英文手稿，约 3.5 GB，带 PAGE-XML 行级标注），德语分支用 ICDAR2017 READ 的 Train-A（50 页 Escher 信件，约 21 MB），再混入比赛 `train.csv` 里真实 Kade 页面的行。Kade 页面没有行坐标，所以只能先用我们自己的切行器切，再和真值按顺序配对——**并且只在切出来的行数正好等于真值行数时才采用**，否则整页丢掉。因为行数一旦对不上，后面就会一路错位，把 A 行的图配上 B 行的字，比少用几页更糟。

字符集也要分开：英语用大小写字母、数字和常见标点；德语在英语基础上加 `äöüÄÖÜß` 以及真实转录里出现过的符号。字符集越大，CTC 的分类头越大，所以"够用就好"，不能把整个 Unicode 都塞进去。

### 2.4 模型测试与指标报告

推理时把五段串起来：按区域从上到下遍历，正文区域逐行识别，表格区域逐格识别并按行拼接。表格的拼接规则是"同一行用空格连接、不同行用换行连接"——因为评分会把连续空白折叠，用空格还是换行其实不影响 CER。

```python
def transcribe_page(img, category):
    """Stage 1-5 串起来：页面图 + 馆藏 -> 整页转录"""
    binary = binarize(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    lang = LANG_ROUTE[category]          # 整页共用同一个语言分支
    parts = []

    for region in detect_regions(img):
        region_binary = binary[region['top']:region['bottom']]
        if region['kind'] == 'text':
            for top, bottom in segment_lines(region_binary):
                # 注意要从灰度图裁，CRNN 的第一层是单通道卷积
                crop = gray[region['top'] + top:region['top'] + bottom]
                parts.append(recognize_line(crop, lang))
        else:
            rows, cols = table_grid(region_binary)
            cells = table_cells(region, rows, cols)
            texts = [recognize_line(gray[c['top']:c['bottom'], c['left']:c['right']], lang)
                     for c in cells]
            n_cols = len(cols) - 1
            table_rows = [texts[i:i + n_cols] for i in range(0, len(texts), n_cols)]
            parts.append('\n'.join(' '.join(r) for r in table_rows))
    return '\n'.join(parts)
```

评分必须和官方口径一致：用比赛提供的 `metric.py` 里的 `page_cer`，先折叠空白再算编辑距离。我们拿一页训练集里**整个文档都被留出**的 Kade 页面来测，保证这是真正的样本外数字：

```python
import sys
sys.path.insert(0, str(EVAL_CODE_DIR))
from metric import page_cer, page_wer

sample = train[train['doc_id'] == KADE_VAL_DOC].iloc[0]
sample_img = np.array(Image.open(IMAGES_DIR / f"{sample['page_id']}.jpg"))
prediction = transcribe_page(sample_img, sample['category'])
print('CER:', page_cer(prediction, sample['text']),
      'WER:', page_wer(prediction, sample['text']))
```

材料里报告的实测结果是：同一套零样本流水线，英语 Bentham 页面的 CER 约 **10%**，德语 Escher 页面约 **32%**。这个差距不是偶然。英文输出已经接近逐字，错误多是形近字符（比如 `portable` 被读成 `postable`）；德文输出能认出 `und`、`der`、`wirklich` 这样的单词，也能正确区分 `ſ` 和 `ß`，但整页连起来就不成句子了。原因很明确：Kraken 模型库里的德语模型不是 Kurrent 专用模型，用关键词 `kurrent`、`sütterlin` 去搜也搜不到现成的开放权重。

所以最后一步是把测试集全部转录并生成提交文件：

```python
test = pd.read_csv(DATA_DIR / 'test.csv')
sub = pd.DataFrame({
    'page_id': test['page_id'],
    'text': [transcribe_page(np.array(Image.open(IMAGES_DIR / f'{pid}.jpg')), cat)
             for pid, cat in zip(test['page_id'], test['category'])],
})
sub.to_csv('traditional_submission.csv', index=False)
print(sub.shape, sub['text'].str.len().describe())
```

对德语分支，材料里指了一条更划算的路：`dh-unibe/trocr-kurrent` 是在约 29.3 万行真实 19 世纪 Kurrent 上微调过的 TrOCR，报告 CER 约 **2.8%**。它是行级的、没有自己的版面分析，所以正确用法是"先跑我们的切行，再把每行裁图送进 TrOCR"，而不是替代整条流水线。

---

## 三、总结

本文中我们搭了一条可解释、可替换的手稿转录流水线。我们首先通过 EDA 确认 `train.csv` 是页级标注、没有行坐标，并建立了馆藏到语言分支的路由表；其次用自适应二值化和长核形态学开运算分离版面线，把每页分成正文和表格两类区域；接着分别用水平投影切行、用横竖版线求交切表格单元格，再用 CRNN + CTC 训练英语和德语两个识别器，德语分支额外混入真实 Kade 页面的行级样本；最后把五段串成整页转录，用官方 `page_cer` 在留出文档上评分并生成提交。实测英语约 10% CER、德语约 32% CER，差距的根源是缺少 Kurrent 专用识别器。

这个比赛给我们最大的启发是：**端到端的大模型很方便，但分段流水线的每一段都能被单独测量**。当总分不好时，我们可以立刻判断是版面切错了、行切错了、还是识别器本身不行，而不是对着一个黑箱猜。材料里那条"错误集中在哪里"的提示也很有价值——正文类靠识别，表格类靠版面，不同馆藏要区别对待，这比盲目换更大的模型有效得多。
