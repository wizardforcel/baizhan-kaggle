# SAI ECG 分类挑战

## 一、比赛简介

### 1.1 比赛背景
这次比赛的数据来源于 MIT-BIH Arrhythmia Database，这是一个在心电信号研究里非常经典的公开数据集。心电图记录的是心脏每次跳动时产生的电活动，医生通常通过观察波形里各个波段的位置、形状和间距来判断心律是否正常。数据集把连续的心电记录切成一条一条单独的心跳，每条心跳固定采样 187 个点，并标注了它属于哪一种心律。

因为正常心跳占绝大多数、异常心跳很少，真实数据里存在严重的类别不平衡。材料里提到正常类大约占 83%，剩下四类加起来只有 17%。如果模型只追求总准确率，它完全可以把所有样本都猜成正常，仍然得到很高的分数，但这样的模型在临床上没有价值。因此这个比赛真正的难点不是把总准确率做高，而是让模型在样本很少的异常类别上也能识别出来。

材料里选择的方案是基于 TensorFlow 2.x / Keras 的深度学习模型，整体思路是用一维卷积网络从时序 ECG 波形里提取特征，再用注意力机制把对分类更有用的通道强调出来。由于心跳类别数量少、波形长度固定，模型需要足够深才能学到细微的形态差别，又不能因为样本不平衡而偏向多数类。材料为此同时使用了残差连接、通道注意力和类权重三种手段。

### 1.2 比赛目标
我们要把一条长度为 187 的心电时序信号映射到 5 个心律类别之一。

训练集里每条样本包含：
- 187 个采样点组成的 ECG 波形作为特征；
- 一个整数标签 0 到 4。

五个类别的具体含义是：
- 0 N：Normal Beat，正常心跳；
- 1 S：Supraventricular Premature，房性早搏；
- 2 V：Premature Ventricular Contraction，室性早搏；
- 3 F：Fusion Beat，融合波；
- 4 Q：Unknown / Unclassifiable，未知或无法分类。

测试集只有波形，没有标签，提交时需要给出每个测试样本的类别预测。材料里加载数据使用 `header=None`，因为原始 CSV 没有表头，第一列到第 187 列是波形采样值，最后一列是标签，所以读取后需要手动区分特征列和标签列。

材料里描述的数据规模大约是训练 ~87,000 条心跳，测试 ~21,000 条心跳。每个样本 187 个时间步、1 个通道，输入形状为 (187, 1)。因为五个类别的样本量差距很大，模型评估不能只看总体准确率，还要分门别类地看每个类别的召回情况，否则很容易把「模型只会认正常心跳」误判成「模型效果很好」。

### 1.3 评价标准
比赛以分类准确率作为主要指标，越接近 1 越好，材料里给出的目标准确率是 >98%。准确率的算法很直观：预测正确的样本数除以样本总数。

不过只靠准确率并不够。假设测试集里 83% 都是正常心跳，模型把所有样本都预测成正常，准确率也能有 0.83，但它对真正需要识别的异常心跳一个都没抓到。材料因此同时记录了 AUC、Precision、Recall 以及 Macro / Weighted F1。AUC 衡量模型把正类和负类分开的能力；Precision 是「预测为某类的样本里有多少是真的」，Recall 是「真实某类里有多少被找出来」；F1 是两者的调和平均。Macro F1 先对每个类别单独算 F1 再取平均，不管类别大小一视同仁，所以能暴露少数类被忽略的问题；Weighted F1 则按类别样本数加权，更接近总体表现。验证时我们会用分层抽样保持类别比例，训练时使用类权重平衡损失，这两点都是为了不让评估结果被多数类掩盖。

## 二、解题过程

### 2.1 数据探索（EDA）
我们首先把数据文件读进来，确认样本数量和列数。

```python
# 读取训练和测试 csv，header=None 因为没有表头
train_df = pd.read_csv(Config.TRAIN_PATH, header=None)
test_df = pd.read_csv(Config.TEST_PATH, header=None)

# 打印基本形状
print(f"训练样本数: {train_df.shape[0]:,}")
print(f"测试样本数: {test_df.shape[0]:,}")
```

通过打印发现训练集大约 87k、测试集约 21k，每条样本 188 列，最后 1 列是标签，前 187 列是波形。

接下来检查缺失值和基础统计。

```python
missing = train_df.isnull().sum().sum() + test_df.isnull().sum().sum()
print(f"缺失值数量: {missing} {'✅ 无缺失' if missing == 0 else '⚠️ 有缺失'}")
```

数据完整，没有空值。每条波形的均值和标准差在不同类别间也有区别，可以通过 describe 看出来。基本统计告诉我们信号幅值大致在 -1 到 1 之间，没有明显的离群截断。

接着我们查看类别分布，材料指出正常心跳占比约 83%，其余四类合计约 17%，分布极不均衡。我们用 value_counts 统计并可视化：

```python
class_counts = train_df.iloc[:,-1].value_counts().sort_index()
total = class_counts.sum()

# 柱状图显示每个类别的绝对数和百分比
pct = class_counts / total * 100
# 另存图片 class_distribution.png
```

柱状图直观显示正常类占绝大多数，V 类次之，S、F、Q 非常稀少。为了了解波形形态差异，我们随机抽取每类 3 个样本画图，横轴是 0~187 个时间点，纵线是幅值。可以看到正常波形形态比较规则，异常类在 QRS 波群的位置和振幅上出现明显偏移。材料里还计算了每类的平均波形，用以判断各类波形的典型形态特征差异。

### 2.2 数据清洗
ECG 数据基本完整，没有缺失值。我们做的是轻量预处理，主要是让数据形状适配模型并解决不平衡。

```python
# 分离特征和标签
X = train_df.iloc[:,:-1].values
y = train_df.iloc[:,-1].values.astype(int)

# 1D CNN 需要 (samples, timesteps, channels)
X = X.reshape(X.shape[0], X.shape[1], 1)
X_test = test_df.iloc[:,:-1].values.reshape(-1, 187, 1)
```

因为 1D 卷积期望输入三维张量，第二维是时间步，第三维是通道。我们把原始二维矩阵重塑为 (样本数, 187, 1)，这样卷积核才能在时间维度上滑动。

接着计算类权重，用 sklearn 的 compute_class_weight 让少数类损失被放大。

```python
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(weights))
# 打印每类权重，少数类权重大于 1
```

类权重反映的是样本数与总数比例的倒数。正常类样本多，权重小于 1；Q 类样本极少，权重可能达到 10 以上。这样在 loss 计算时，少数类样本的梯度会得到放大。

数据划分采用分层抽样，测试集比例 0.15，保证验证集类别比例与训练集一致。

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=Config.TEST_SIZE, stratify=y, random_state=Config.SEED
)
```

分层抽样的好处是验证集里每个类别的比例与整体一样，评估结果更可信。最后把标签转为 one-hot，方便 categorical crossentropy 计算。

```python
y_train_cat = keras.utils.to_categorical(y_train, Config.NUM_CLASSES)
y_val_cat = keras.utils.to_categorical(y_val, Config.NUM_CLASSES)
```

one-hot 把整数标签变成向量，如类别 2 会变成 [0,0,1,0,0]，模型输出也是概率分布，便于用交叉熵衡量。

### 2.3 模型训练与超参数调优
材料采用的模型是 1D ResNet 加 Squeeze-and-Excitation 注意力块。ResNet 的残差连接可以避免梯度消失，让网络变深；SE 块做通道级注意力，学习强调对心律判别更重要的特征通道。

核心配置如下：
- 输入形状 (187, 1)
- 初始滤波器 32，Dropout 0.3，L2 正则 1e-4
- SE 比率 16
- 批次大小 64，最大轮数 50，学习率 1e-3
- 学习率调度因子 0.5，耐心 3；EarlyStopping 耐心 10

首先定义 SE 注意力块。思路是先做全局平均池化把时间信息压缩成 1 个向量，再用两个全连接做降维和升维得到通道权重，最后用这个权重逐通道缩放。

```python
def squeeze_excitation_block(x, ratio=16):
    filters = int(x.shape[-1])
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(max(filters // ratio, 1), activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, filters))(se)
    return layers.Multiply()([x, se])
```

再定义残差块，包含两层 Conv1D + BatchNorm + ReLU，主干路径结束后如果需要改变通道或下采样就用 1x1 卷积做 shortcut。

```python
def residual_block(x, filters, stride=1, use_se=True):
    shortcut = x
    x = Conv1D(filters, 3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    if use_se:
        x = squeeze_excitation_block(x, ratio=16)

    if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    return Activation('relu')(x)
```

完整的模型由 stem 块和三个残差阶段组成，最后接全局平均池化和全连接头。

```python
def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = Conv1D(32, 7, 2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, 2, padding='same')(x)

    x = residual_block(x, 64, 1)
    x = residual_block(x, 64, 1)
    x = residual_block(x, 128, 2)
    x = residual_block(x, 128, 1)
    x = residual_block(x, 256, 2)
    x = residual_block(x, 256, 1)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)
```

编译时使用 Adam 优化器，指标包含 accuracy、AUC、Precision、Recall。

```python
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)
```

训练时传入 class_weights，配合 EarlyStopping 和 ReduceLROnPlateau。

```python
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True
)
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3
)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    class_weight=class_weights,
    epochs=50,
    batch_size=64,
    callbacks=[early_stop, lr_scheduler]
)
```

我们没有做大规模网格搜索，而是把精力放在数据划分和类权重上，保持模型可复现。

### 2.4 模型测试与指标报告
训练结束后在验证集上评估。

```python
# 在验证集上预测
y_pred = model.predict(X_val)
y_pred_cls = np.argmax(y_pred, axis=1)

# 混淆矩阵
cm = confusion_matrix(y_val, y_pred_cls)
print(classification_report(y_val, y_pred_cls, target_names=Config.CLASS_NAMES))
```

混淆矩阵告诉我们每个真实类别被预测成哪一类。材料里特别关注了少数类 V、S、F、Q 的召回率，因为这些类在临床上往往是最重要的。我们还会绘制每类的 ROC 曲线和 AUC，观察模型在不同阈值下的表现。

```python
# 计算每类 F1
f1_macro = f1_score(y_val, y_pred_cls, average='macro')
f1_weighted = f1_score(y_val, y_pred_cls, average='weighted')
print(f"Macro F1: {f1_macro:.4f}")
print(f"Weighted F1: {f1_weighted:.4f}")
```

为了可视化训练过程，我们画出 loss、accuracy、AUC 随 epoch 的变化曲线。材料里用绿色虚线标注了目标准确率 98%，可以看到验证准确率随着 epoch 增加逐渐逼近该目标，EarlyStopping 在验证准确率不再提升时停止训练，防止过拟合。

在测试集上生成提交文件时，先用全部训练数据重新训练一次，随后对测试波形预测并保存 id 与类别。

```python
model.fit(X, y_cat, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)
test_pred = model.predict(X_test_final)
submission = pd.DataFrame({
    'id': test_ids,
    'label': np.argmax(test_pred, axis=1)
})
submission.to_csv('submission.csv', index=False)
```

最后我们检查提交文件行数、列名是否与样例一致，避免因格式错误丢分。这是材料里强调的一个容易忽略但很关键的细节。

## 三、总结

本文中我们完成了 SAI ECG 分类挑战的完整流程。我们首先通过 EDA 明确了数据规模、187 长度波形以及严重的类别不平衡；其次做了最小必要的数据清洗、形状重塑、分层划分和类权重计算；接着构建了 1D ResNet 加 SE 注意力模型，并在验证集上监测准确率、AUC、F1 等指标；最后按比赛格式生成提交文件。材料的目标准确率是大于 98%，整个方案的核心在于用残差连接加通道注意力提取细微波形特征，并用类权重与分层抽样缓解不平衡问题。

从实现角度看，这个比赛给我们的启发是三个方面。第一，特征工程要跟着数据特性走，时序 ECG 用一维卷积比二维图像更自然，而且保持原始采样长度不做额外重采样能保留细节。第二，评价体系要多元化，仅看准确率会掩盖少数类表现，Macro F1 和 per-class Recall 更能反映临床价值。第三，模型复杂度和数据增强要平衡，材料没有采用大数据增强，而是通过类权重和注意力机制让模型学会关注稀有心律，这在小样本医学任务里往往更稳定。
