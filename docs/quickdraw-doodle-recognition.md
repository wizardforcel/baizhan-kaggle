# Quick, Draw! 涂鸦识别（Doodle Recognition）

## 一、比赛简介

### 1.1 比赛背景

Quick, Draw! 是 Google 做过的一个小游戏：屏幕上出现一个词，比如「猫」「自行车」「生日蛋糕」，玩家要在 20 秒内用鼠标把它画出来，游戏再猜你画的是什么。几年下来，这个游戏收集了**几千万张**手绘涂鸦，每张都带一个官方标签。这些数据成了训练「草图识别」模型的好材料。

这个比赛要求我们根据涂鸦的**笔画轨迹**判断它画的是什么类别。数据里保存的不是图片，而是一串坐标：每一笔是一组 `(x, y)` 点，按落笔顺序排列。这跟普通图像分类不一样——同一只猫，有人先画耳朵，有人先画身体；有人画得大，有人画得小；有人只画三笔，有人画二十笔。**同样的语义，轨迹千差万别**，这就是难点。

### 1.2 比赛目标

训练数据放在 `train_simplified/` 目录下，一个类别一个 CSV 文件，一共 340 个类别。每个文件有这样几列：

- `key_id`：这张涂鸦的唯一编号；
- `countrycode`：玩家所在国家；
- `drawing`：笔画数据，是一个字符串形式的嵌套列表，最内层是 x 坐标序列和 y 坐标序列；
- `recognized`：游戏当时有没有认出来（True / False）；
- `timestamp`：绘制时间；
- `word`：类别名。

测试集是 `test_simplified.csv`，只有 `key_id` 和 `drawing`。我们要给每个 `key_id` 提交**三个候选类别**。

### 1.3 评价标准

比赛使用 **MAP@3**（Mean Average Precision at 3）：对每张图，看我们给出的前三个候选里，正确答案排在第几位。排第一得 1 分，排第二得 1/2，排第三得 1/3，没进前三得 0 分，最后对所有样本取平均。所以满分是 1，越高越好；而且**顺序很重要**——把正确答案放第一，比放第三多拿将近三倍的分。

## 二、解题过程

我们的主线是：把笔画轨迹「画成图」，再用 ImageNet 上预训练过的 DenseNet121 做迁移学习。材料里这条路线在排行榜上达到 **0.925**，是同一批材料里最高的之一。

### 2.1 数据探索（EDA）

我们首先统计数据的规模。把 `train_simplified/` 下的 CSV 一个个读进来数行数：

```python
import os
import pandas as pd
from glob import glob

INPUT_ROOT = '../input/quickdraw-doodle-recognition'
csv_files = glob(os.path.join(INPUT_ROOT, 'train_simplified', '*.csv'))

# 一个文件就是一个类别，文件名就是类别名
words = [os.path.basename(f)[:-4] for f in csv_files]
print('类别数 =', len(words))          # 340

# 统计每个文件的行数，也就是每个类别有多少张涂鸦
row_counts = []
for f in csv_files:
    # 只数行数，不把内容读进内存，否则几百个文件会撑爆内存
    with open(f, 'r', encoding='utf-8') as fh:
        row_counts.append(sum(1 for _ in fh) - 1)   # 减去表头

print('总样本数 =', sum(row_counts))
print('最多/最少 =', max(row_counts), min(row_counts))
```

我们得到 **340 个类别**。另一份 EDA notebook 只读了前 185 个文件，就已经有 **27,098,813** 行，可见整个训练集是千万量级。这也直接决定了后面不能用「一次性读进内存」的做法，必须用生成器一边读一边训练。

接着我们看一条 `drawing` 到底长什么样：

```python
import ast

df = pd.read_csv(csv_files[0], nrows=1)
strokes = ast.literal_eval(df['drawing'].values[0])

print('这笔涂鸦由', len(strokes), '笔组成')
print('第一笔有', len(strokes[0][0]), '个点')
print('x 坐标前五个 =', strokes[0][0][:5])
print('y 坐标前五个 =', strokes[0][1][:5])
```

输出类似 `[[138, 142, 155, 169, ...], [2, ...]]`。也就是说，一笔由两个等长列表组成，一个存 x、一个存 y。所有坐标都落在 0～255 的范围内，这一点很关键——它意味着我们可以直接把坐标当成像素位置来画图，不需要额外缩放。

我们还看了类别和 `recognized` 的关系。`recognized` 表示「游戏当时猜对了没有」，它其实是一个**难度标签**：`False` 的样本是连游戏都没认出来的「难样本」。材料里的做法是训练时**只保留 `recognized == True` 的样本**，避免把太离谱的涂鸦喂给模型。

### 2.2 数据清洗

这份数据不需要去空值，需要的是**把轨迹变成模型能吃的格式**，以及**控制内存**。

第一步是建立标签映射。340 个类别要映射成 0～339 的整数，才能做 one-hot 编码：

```python
import os
import numpy as np

# 固定顺序很重要：训练、验证、提交必须用同一套编号
categories = sorted([
    f[:-4] for f in os.listdir(os.path.join(INPUT_ROOT, 'train_simplified'))
], key=str.lower)

word2id = {w: i for i, w in enumerate(categories)}
NCATS = len(categories)
print('类别数 =', NCATS)
```

第二步是把坐标画成灰度图。我们不用彩色，因为涂鸦本来就没有颜色信息；但我们可以用**灰度深浅表示笔画的先后顺序**——先画的笔画亮一点，后画的暗一点，这样模型就能从图里读出「时间」这个额外线索：

```python
import cv2

BASE_SIZE = 256   # 先在 256x256 的画布上画，再缩小，边缘更平滑

def draw_strokes(raw_strokes, size=64, line_width=6):
    """把一笔涂鸦的坐标序列画成一张灰度图"""
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        # t 是笔画序号；越靠后的笔画颜色越深，编码「作画顺序」
        color = 255 - min(t, 10) * 13
        for i in range(len(stroke[0]) - 1):
            # 相邻两点连成线段，line_width 让线条在缩小时不至于消失
            cv2.line(img,
                     (stroke[0][i],     stroke[1][i]),
                     (stroke[0][i + 1], stroke[1][i + 1]),
                     color, line_width)
    if size != BASE_SIZE:
        # 统一缩到模型输入尺寸，保证一个 batch 里形状一致
        img = cv2.resize(img, (size, size))
    return img
```

这里有两个小设计值得说明：

- **线宽 6**：如果按 1 像素画，缩到 64×64 之后线条会变得断断续续，模型很难看出形状；加粗之后缩放仍然连续。
- **`min(t, 10)`**：只让前 10 笔有颜色差异，后面的笔画统一用最深色。因为大多数涂鸦不超过 10 笔，这样既保留了顺序信息，又不会让颜色跨度太大。

第三步是**打乱文件**。原始数据是一个类别一个文件，如果按文件顺序读，一个 batch 里全是同一个类别，模型会学不到东西。材料里的做法是先把 CSV 行打乱、重新切成 100 个文件（`shuffle-csvs`），训练时从这 100 个文件里随机抽。这样即使顺序读，也能保证类别混合。

### 2.3 模型训练与超参数调优

我们的模型是 **DenseNet121**。先说它是什么：普通卷积网络是一层接一层往下传；ResNet 加了「跨层相加」，让信息能跳过几层；DenseNet 更进一步，把前面所有层的输出**拼接**起来送给后面每一层。好处是特征被反复利用，参数更少，小图上也能训得动。

模型结构很简洁：预训练的 DenseNet121 主干 + 全局平均池化 + 一个 1024 维全连接 + 340 类 softmax。

```python
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy, categorical_crossentropy, top_k_categorical_accuracy

size = 64      # 输入图 64x64，够看清大致形状，又不会太慢
NCATS = 340

# include_top=False：不要 ImageNet 原来的 1000 类输出层
# weights='imagenet'：用 ImageNet 上学到的通用视觉特征做起点
base_model = DenseNet121(include_top=False, weights='imagenet',
                         input_shape=(size, size, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)      # 把二维特征图压成一个向量
x = Dense(1024, activation='relu')(x)
predictions = Dense(NCATS, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# lr=1e-4 是微调预训练模型的常用起点，太大容易把预训练权重冲掉
model.compile(optimizer=Adam(lr=1e-4, decay=1e-9),
              loss='categorical_crossentropy',
              metrics=[categorical_accuracy, top_3_accuracy])
```

数据供给用生成器。关键在于**永远不把全量数据读进内存**：每次只从随机挑的一个 CSV 里读几行，画成图，凑够一个 batch 就交出去。

```python
import pandas as pd
import numpy as np

def image_generator(size, batchsize, csv_files, word2id, max_rows=100000):
    """无限循环的生成器：每次产出一个 batch 的 (图像, one-hot 标签)"""
    while True:
        images, labels = [], []
        while len(images) < batchsize:
            # 随机挑一个类别文件，保证 batch 里类别混合
            path = np.random.choice(csv_files)
            # 随机跳到一个位置读几行；跳过表头用 skiprows
            df = pd.read_csv(path, skiprows=np.random.randint(1, max_rows), nrows=5)
            for _, row in df.iterrows():
                # 只保留游戏自己认得出的样本，过滤掉太离谱的涂鸦
                if not bool(row['recognized']):
                    continue
                img = draw_strokes(ast.literal_eval(row['drawing']), size=size)
                images.append(img)
                labels.append(word2id[row['word']])
                if len(images) >= batchsize:
                    break

        X = np.array(images, dtype=np.float32) / 255.0
        # 灰度图复制成 3 通道，因为 ImageNet 预训练模型要 3 通道输入
        X = np.stack([X, X, X], axis=-1)
        y = keras.utils.to_categorical(labels, NCATS)
        yield X, y
```

训练超参数方面，材料里给出的设置是：**batch size 330、输入 64×64、每轮 10000 步、训练 2 轮**。步数看起来不多，但因为每个 batch 只有 330 张，10000 步就是 330 万张图，已经覆盖了大量样本。学习率用 `ReduceLROnPlateau` 在验证指标不动时自动降下来：

```python
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    # 连续几轮验证 loss 不降就把学习率砍半，帮助模型在后期收敛
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1,
                      verbose=1, min_lr=1e-6),
    # 只保存验证集上最好的模型，防止最后一轮反而变差
    ModelCheckpoint('best_densenet.h5', monitor='val_loss',
                    save_best_only=True, verbose=1),
]

model.fit_generator(
    image_generator(64, 330, train_csvs, word2id),
    steps_per_epoch=10000,
    epochs=2,
    validation_data=image_generator(64, 330, valid_csvs, word2id),
    validation_steps=200,
    callbacks=callbacks,
)
```

为什么只训 2 轮？因为数据量太大，一轮已经见过几百万张不同的图，再训下去收益很小，而每轮要花很久。这也是比赛里很实际的一个取舍：**在有限算力下，宁可多喂数据、少跑轮次**。

### 2.4 模型测试与指标报告

本地评估要自己实现 MAP@3，因为 Keras 只提供 top-k 准确率（只看对没对，不管顺序）。MAP@3 的定义是：正确答案排第 `i` 位就得 `1/(i+1)` 分。

```python
import numpy as np

def apk(actual, predicted, k=3):
    """单个样本的 average precision at k"""
    predicted = predicted[:k]
    score, hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p == actual:
            hits += 1.0
            score += hits / (i + 1.0)   # 第 1 位得 1，第 2 位得 1/2 ...
    return score

def mapk(actuals, predictions, k=3):
    """所有样本取平均"""
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predictions)])

# 从模型输出里取分数最高的三个类别
def top3(predictions):
    # argsort 默认升序，加负号变成降序
    return np.argsort(-predictions, axis=1)[:, :3]
```

材料里这条 DenseNet121 路线的公开排行榜分数是 **0.925**。作为对照，同一批材料里还有：

| 方案 | 公开 LB |
| --- | --- |
| DenseNet121（本方案） | 0.925 |
| Greyscale MobileNet | 0.892 |
| 纯 CNN（Black CNN） | 约 0.77 |
| 随机基线 | 约 0.002 |

从这张表能看出两件事：**迁移学习带来的提升非常大**（0.77 → 0.925），**输入分辨率也不是越高越好**（MobileNet 用 128×128 反而低于 DenseNet 的 64×64，说明模型结构和数据增强的匹配程度更重要）。

提交文件的格式要求是每个 `key_id` 三行，分别对应第 1、2、3 候选：

```python
# 对测试集逐批预测
test_df = pd.read_csv('../input/quickdraw-doodle-recognition/test_simplified.csv')

rows = []
for start in range(0, len(test_df), 1000):
    batch = test_df.iloc[start:start + 1000]
    # 测试集同样要画成图，只是没有标签
    X = np.array([draw_strokes(ast.literal_eval(d), size=64)
                  for d in batch['drawing']], dtype=np.float32) / 255.0
    X = np.stack([X, X, X], axis=-1)

    preds = model.predict(X, batch_size=100)
    top = top3(preds)

    # 每个 key_id 展开成三行：word 是类别名，不是编号
    for kid, three in zip(batch['key_id'].values, top):
        for rank, cls in enumerate(three):
            rows.append({
                'key_id': kid,
                'word': categories[cls],
            })

submission = pd.DataFrame(rows)
submission.to_csv('submission.csv', index=False)
print(submission.shape)   # 应该是 测试集行数 * 3
```

最后我们检查了三件事：行数是不是测试集的三倍、`key_id` 有没有漏、`word` 是不是都在 340 个类别里。比赛里因为提交格式错误拿零分的情况，比模型不够好更常见。

---

## 三、总结

本文中我们完成了 Quick, Draw! 涂鸦识别的完整流程。我们首先用 EDA 摸清了数据：340 个类别、千万级样本、每条记录是一串 0～255 的坐标；其次我们做了面向任务的「清洗」——建立固定标签映射、过滤 `recognized=False` 的难样本、打乱 CSV 保证 batch 内类别混合；接着我们把轨迹画成灰度图，用灰度深浅编码笔画顺序，并用生成器边读边画，避免内存爆掉；然后我们以 ImageNet 预训练的 DenseNet121 为主干，接全局平均池化与 340 类 softmax，用 Adam（lr=1e-4）、batch 330、64×64 输入训练 2 轮、每轮 10000 步；最后我们实现了 MAP@3 评估，并按「每个 key_id 三行」的格式生成提交，公开排行榜达到 0.925。

这个比赛让我们体会到，**数据的表示方式往往比模型更决定成败**。同一串坐标，可以当序列喂给 1D-CNN，也可以画成图喂给 2D-CNN；后者借助 ImageNet 的预训练知识，效果直接上了一个台阶。另外，把「笔画先后」编码成颜色深浅这种小技巧，成本极低，却给模型补上了原本缺失的一维信息。
