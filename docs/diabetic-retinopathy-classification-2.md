# 糖尿病视网膜病变识别 2（Diabetic Retinopathy Classification 2）

## 一、比赛简介

### 1.1 比赛背景

**糖尿病视网膜病变（Diabetic Retinopathy，简称 DR）** 是糖尿病患者眼底血管受损引起的一类病变，也是工作年龄人群失明的主要原因之一。医生诊断时会给病人拍一张**眼底照片**：照片中央是视盘，四周分布着视网膜血管。病变越重，照片上能看到的出血点、硬性渗出、棉絮斑就越多。

这个比赛要求参赛者用深度学习模型自动给眼底照片**分级**。分级一共 5 档（0～4），从「没有病变」一路到「增殖性病变」。它的现实意义很直接：如果模型能代替医生做初筛，偏远地区或缺医少药的地区就能先用机器过一遍，把真正需要专家看的病人挑出来。

难点有三个：一是不同等级之间的视觉差异很细微，相邻等级（比如 1 和 2）经常难以区分；二是数据类别严重不均衡，无病变的样本远多于重病变样本；三是眼底照片的拍摄角度、亮度、清晰度差异很大，模型必须对这些变化鲁棒。

### 1.2 比赛目标

输入是一张眼底照片（RGB 彩色图），输出是 0～4 五个等级中的一个。数据目录的结构是：

- `retina-train/`：训练图片，文件名以 `_等级` 结尾，比如 `xxxx_2.png` 表示这张图是 2 级；
- `retina-test/`：测试图片，只有编号，没有等级；
- 提交文件的列是 `Id`（图片文件名）和 `Category`（预测等级）。

### 1.3 评价标准

比赛使用**二次加权 Kappa（Quadratic Weighted Kappa，QWK）**，取值范围 -1 到 1，**越高越好**。QWK 衡量的是预测等级与真实等级的一致程度，并且**按「差几档」来加权惩罚**：差 1 档只扣一点，差 4 档扣得很重。对医学分级来说这很合理——把 0 级误判成 1 级，复查一下即可；把 0 级误判成 4 级，就是严重的误诊。

---

## 二、解题过程

我们首先搭建数据集类并统计各类别数量（EDA），其次用分层切分和按类别截断来处理不均衡（数据清洗），接着用预训练的 EfficientNetB3 微调并配一套自适应学习率回调（训练与调优），最后在测试集上推理并生成提交文件（测试报告）。

### 2.1 数据探索（EDA）

我们首先写一个自定义数据集类，它扫描 `retina-train/` 目录，从文件名最后一段解析出类别编号，同时顺手统计每个类别有多少张图：

```python
import os
from glob import glob
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, split="retina-train", transformations=None):
        self.transformations = transformations
        self.im_paths, self.im_lbls = [], []
        self.cls_names, self.cls_counts = {}, {}     # 类别名 -> 编号、类别计数
        for path in glob(os.path.join(root, split, "*")):
            # 文件名形如 xxxx_2.png，取最后一段得到等级 "2"
            cls_name = os.path.splitext(os.path.basename(path))[0].split("_")[-1]
            if cls_name not in self.cls_names:
                self.cls_names[cls_name] = len(self.cls_names)   # 给新类别分配编号
            self.cls_counts[cls_name] = self.cls_counts.get(cls_name, 0) + 1
            self.im_paths.append(path)
            self.im_lbls.append(self.cls_names[cls_name])

    def __getitem__(self, idx):
        im = Image.open(self.im_paths[idx]).convert("RGB")
        if self.transformations:
            im = self.transformations(im)   # 缩放 + 转张量
        return im, self.im_lbls[idx]

    def __len__(self):
        return len(self.im_paths)
```

接着把类别计数画成柱状图，看分布到底有多不均衡：

```python
def plot_label_count(df, plot_title):
    vcounts = df['labels'].value_counts()     # 每个等级有多少张图
    labels, values = vcounts.keys().tolist(), vcounts.tolist()
    sns.barplot(x=labels, y=values)
    plt.title(f'Images per Label in {plot_title} data')
    # 在每根柱子的中间写上具体数量，方便读数
    for i in range(len(labels)):
        plt.text(i, values[i] / 2, str(values[i]), ha='center', color='yellow')
    plt.show()
```

柱状图直观地告诉我们：**0 级（无病变）占了绝大多数，4 级（最严重）样本最少**。这种长尾分布如果原样训练，模型会倾向于「多猜 0 级」——因为猜 0 在训练集上「命中率」最高，但这样 QWK 会被少数类的错误拖得很惨。

再看几张随机图片确认图像格式：

```python
# 取一个 batch 的图，反归一化后画出来，标题写真实等级
images, labels = next(valid_gen)
classes = list(valid_gen.class_indices.keys())
for i in range(min(25, len(labels))):
    plt.subplot(5, 5, i + 1)
    plt.imshow(images[i] / 255)          # 转回 0~255 便于显示
    plt.title(classes[np.argmax(labels[i])])
    plt.axis('off')
```

可以看到照片有亮有暗、有偏黄有偏红，拍摄条件差异确实很大——这也印证了后面必须做图像增强。

### 2.2 数据清洗

图像任务里的「清洗」主要是三件事：**按类别截断、分层切分、统一尺寸并增强**。

**第一，按类别截断（trim）。** 既然 0 级样本多到「压死」其他类，我们就把每个类别的样本数限制在一个上限 `max_size` 以内：超过上限的类别随机抽 `max_size` 张，不到上限的类别原样保留。这样多个大类被拉回同一水平，少数类不至于被淹没：

```python
def trim(df, max_size, min_size, column):
    """把每个类别的样本数限制在 [min_size, max_size] 之间"""
    sample_list = []
    groups = df.groupby(column)
    for label in df[column].unique():
        group = groups.get_group(label)
        if len(group) > max_size:
            # 超过上限：分层随机抽 max_size 张
            samples, _ = train_test_split(group, train_size=max_size,
                                          shuffle=True, random_state=123,
                                          stratify=group[column])
            sample_list.append(samples)
        elif len(group) >= min_size:
            sample_list.append(group)      # 在区间内就原样保留
    return pd.concat(sample_list, axis=0).reset_index(drop=True)
```

**第二，分层切分。** 训练/验证按 8:2 切，并指定 `stratify` 让两边各类比例一致，避免验证集里 4 级样本太少、指标抖得没法看：

```python
train_df, valid_df = train_test_split(
    df, train_size=0.8, shuffle=True, random_state=123, stratify=df['labels'])
```

**第三，统一尺寸与增强。** 所有图片缩放到 224×224，训练集额外做随机水平翻转——眼底照片左右翻转后仍然是合法的眼底照片，标签不变，等于免费多了一份数据。这里不手动除以 255，是因为 EfficientNet 在 Keras 里自带输入预处理层，会自己完成缩放：

```python
img_size = (224, 224)
# 训练生成器：随机水平翻转；验证/测试生成器：不做增强
tr_gen = ImageDataGenerator(preprocessing_function=lambda img: img,
                            horizontal_flip=True)
ts_gen = ImageDataGenerator(preprocessing_function=lambda img: img)

train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                       target_size=img_size, class_mode='categorical',
                                       color_mode='rgb', shuffle=True, batch_size=16)
valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                       target_size=img_size, class_mode='categorical',
                                       color_mode='rgb', shuffle=True, batch_size=16)
```

### 2.3 模型训练与超参数调优

模型我们选 **EfficientNetB3**：它是「用神经架构搜索找出来的、按宽度/深度/分辨率三个维度同时放大」的卷积网络家族，在同等参数量下通常比 ResNet 更准。用 `include_top=False` + ImageNet 预训练权重，只取它的卷积特征提取部分，再自己接一个分类头：

```python
base_model = tf.keras.applications.efficientnet.EfficientNetB3(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='max')

model = Sequential([
    base_model,
    # 批归一化：把特征拉回均值 0、方差 1，训练更稳
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    # 全连接层 256 维；同时加 L2 和 L1 正则，压制过拟合
    Dense(256, kernel_regularizer=regularizers.l2(0.016),
          activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu'),
    Dropout(0.45),                       # 随机丢 45% 神经元，进一步防过拟合
    Dense(5, activation='softmax')       # 5 个等级的概率输出
])

# 优化器用 Adamax，损失用多分类交叉熵
model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy',
              metrics=['accuracy'])
```

**训练策略：三段式自适应。** 这里没有用固定的学习率，而是写了一个回调，规则是：

- 训练准确率低于 0.9 时，**盯训练准确率**；高于 0.9 后改成**盯验证损失**——因为在训练集上已经学得差不多，此时真正该关心的是泛化；
- 被盯的指标连续 3 轮不进步，就把学习率乘以 0.5；
- 学习率连续调了 10 次还是不进步，就停止训练；
- 每轮结束把「当前最好」的权重存下来，训练结束后恢复这组权重。

```python
class MyCallback(keras.callbacks.Callback):
    def __init__(self, model, patience, stop_patience, threshold, factor, epochs):
        super().__init__()
        self.model, self.patience = model, patience
        self.stop_patience, self.threshold = stop_patience, threshold
        self.factor, self.epochs = factor, epochs
        self.count, self.stop_count = 0, 0
        self.lowest_vloss = np.inf
        self.best_weights = model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        acc, v_loss = logs.get('accuracy'), logs.get('val_loss')
        # 训练还没到 0.9 就盯 accuracy，到了就盯 val_loss
        monitor_value = acc if acc < self.threshold else -v_loss
        if acc >= self.threshold and v_loss < self.lowest_vloss:
            self.lowest_vloss = v_loss
            self.best_weights = self.model.get_weights()
            self.count = 0
        elif monitor_value <= getattr(self, 'best_monitor', -np.inf):
            self.count += 1
        else:
            self.best_monitor = monitor_value
            self.count = 0
        # 连续 patience 轮不进步：学习率减半
        if self.count >= self.patience:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            tf.keras.backend.set_value(self.model.optimizer.lr, lr * self.factor)
            self.count, self.stop_count = 0, self.stop_count + 1
        # 学习率调了 stop_patience 次仍无起色：停
        if self.stop_count >= self.stop_patience:
            self.model.stop_training = True
```

再配上 `batch_size=16`、`epochs=100`、`patience=3`、`stop_patience=10`、`factor=0.5`、`threshold=0.9`，直接 `model.fit` 即可。这套回调等价于「自动版的 ReduceLROnPlateau + EarlyStopping」，好处是不用手动设几个固定的里程碑轮数。

### 2.4 模型测试与指标报告

训练结束后，我们在训练集和验证集上分别评估一次，看有没有过拟合（训练分数明显高于验证分数就是过拟合的信号）：

```python
train_score = model.evaluate(train_gen, verbose=1)
valid_score = model.evaluate(valid_gen, verbose=1)
print("Train Loss:", train_score[0], "Train Accuracy:", train_score[1])
print("Validation Loss:", valid_score[0], "Validation Accuracy:", valid_score[1])
```

材料里的 notebook 没有贴出具体的数值，所以我们这里不编造分数，只报告方法：看 `history` 里的 `val_accuracy` 曲线，确认它在学习率下调后仍能继续爬升，并且训练/验证两条曲线的差距没有越拉越大。

最后生成提交文件。测试图片按文件名排序读取，保持与样例提交一致的顺序；模型对每张图输出 5 个概率，取概率最大的下标作为预测等级：

```python
test_dir = '../input/diabetic-retinopathy-classification-2/retina-test'
# 测试集不读标签，只按 224×224 读入图片
test_gen = tf.keras.utils.image_dataset_from_directory(
    test_dir, labels=None, label_mode=None, color_mode='rgb',
    image_size=(224, 224), shuffle=False)

preds = model.predict(test_gen)              # 形状 (图片数, 5)
pred_list = [np.argmax(p) for p in preds]    # 每张图取概率最大的等级

files = sorted(os.listdir(test_dir))         # 文件名顺序要和读取顺序一致

submission = pd.DataFrame({'Id': files, 'Category': pred_list})
submission.to_csv('submission.csv', index=False)
```

提交前我们还会核对两件事：行数等于测试图片数、`Category` 的取值都在 0～4 之内。这类低级错误在比赛中丢分最冤。

---

## 三、总结

本文中我们完成了糖尿病视网膜病变识别 2 的完整流程。我们首先用自定义数据集类扫描图片目录、从文件名解析等级并统计类别分布，发现 0 级样本远多于 4 级；其次我们用「按类别截断 + 分层切分 + 水平翻转增强」处理不均衡和样本不足，并把图片统一到 224×224；接着我们以 ImageNet 预训练的 EfficientNetB3 为骨干，接上批归一化、带 L1/L2 正则的全连接层和 Dropout，用 Adamax 加自适应学习率回调训练；最后我们在训练/验证集上评估并把测试图片的 argmax 结果写成 `Id`、`Category` 两列的提交文件。

这场比赛给我们的最大启发是：**医学影像比赛的胜负往往不在模型结构，而在「怎么对付不均衡」**。截断多数类、分层切分、按验证损失而不是训练准确率决定何时停，这三件事做对了，一个标准的迁移学习管线就能跑出可用的结果。材料里还有人尝试了 MobileViT 这类轻量视觉 Transformer，思路上是「用更小的模型换更快的推理」，可以作为后续的改进方向。