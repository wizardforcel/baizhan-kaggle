# SAI ECG 分类挑战

## 一、比赛简介

### 1.1 比赛背景
这次比赛的数据来源于 MIT-BIH Arrhythmia Database，这是一个在心电信号研究里非常经典的公开数据集。数据集里每条心跳都被切成固定长度的波形，标注了对应的心律类型。因为正常心跳占绝大多数、异常心跳很少，真实数据里存在严重的类别不平衡，这也是比赛里需要处理的核心难点之一。

材料里选择的方案是基于 TensorFlow 2.x / Keras 的深度学习模型，整体思路是用一维卷积网络从时序 ECG 波形里提取特征，再用注意力机制把对分类更有用的通道强调出来。由于心跳类别数量少、波形长度固定，模型需要足够深才能学到细微的形态差别，又不能因为样本不平衡而偏向多数类。

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

测试集只有波形，没有标签，提交时需要给出每个测试样本的类别预测。

材料里描述的数据规模大约是训练 ~87,000 条心跳，测试 ~21,000 条心跳。每个样本 187 个时间步、1 个通道，输入形状为 (187, 1)。

### 1.3 评价标准
比赛以分类准确率作为主要指标，越接近 1 越好，材料里给出的目标准确率是 >98%。

因为类别不平衡，材料里还同时关注了 AUC、Precision、Recall 以及 Macro / Weighted F1，用来检验模型在少数类上的表现是否被拉低。验证时我们会用分层抽样保持类别比例，训练时使用类权重平衡损失。

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

接着我们查看类别分布，材料指出正常心跳占比约 83%，其余四类合计约 17%，分布极不均衡。我们用 value_counts 统计并可视化：

```python
class_counts = train_df.iloc[:,-1].value_counts().sort_index()
total = class_counts.sum()

# 柱状图显示每个类别的绝对数和百分比
pct = class_counts / total * 100
```

为了了解波形形态差异，我们随机抽取每类 3 个样本画图，横轴是 0~187 个时间点，纵线是幅值。可以看到正常波形形态比较规则，异常类在 QRS 波群的位置和振幅上出现明显偏移。

### 2.2 数据清洗
ECG 数据基本完整，没有缺失值。我们做的是轻量预处理，主要是让数据形状适配模型并解决不平衡。

```python
# 分离特征和标签
X = train_df.iloc[:,:-1].values
y = train_df.iloc[:,-1].values.astype(int)

# 1D CNN 需要 (samples, timesteps, channels)
X = X.reshape(X.shape[0], X.shape[1], 1)
```

接着计算类权重，用 sklearn 的 compute_class_weight 让少数类损失被放大。

```python
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(weights))
```

数据划分采用分层抽样，测试集比例 0.15，保证验证集类别比例与训练集一致。

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=Config.TEST_SIZE, stratify=y, random_state=Config.SEED
)
```

最后把标签转为 one-hot，方便 categorical crossentropy 计算。

### 2.3 模型训练与超参数调优
材料采用的模型是 1D ResNet 加 Squeeze-and-Excitation 注意力块。ResNet 的残差连接可以避免梯度消失，让网络变深；SE 块做通道级注意力，学习强调对心律判别更重要的特征通道。

核心配置如下：
- 输入形状 (187, 1)
- 初始滤波器 32，Dropout 0.3，L2 正则 1e-4
- SE 比率 16
- 批次大小 64，最大轮数 50，学习率 1e-3
- 学习率调度因子 0.5，耐心 3；EarlyStopping 耐心 10

构建模型时我们先定义 SE 块和残差块，然后堆叠三段主干，最后接 GlobalAvgPool、Dense(128)、Dropout、Dense(5)。

```python
# 简化的构建思路
inputs = layers.Input(shape=(187, 1))
x = Conv1D(32, 7, padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(2)(x)

# ResBlock + SE 重复
...
outputs = Dense(5, activation='softmax')(x)
model = Model(inputs, outputs)
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
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    class_weight=class_weights,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=[early_stop, lr_scheduler, checkpoint]
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

材料里提供了按类别的 F1、AUC 曲线以及 ROC 曲线可视化，用来检查 V、S、F、Q 这几个少数类是否被模型学习到。目标是整体准确率超过 98%，并且少数类的召回不显著低于多数类。

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

## 三、总结

本文中我们完成了 SAI ECG 分类挑战的完整流程。我们首先通过 EDA 明确了数据规模、187 长度波形以及严重的类别不平衡；其次做了最小必要的数据清洗、形状重塑、分层划分和类权重计算；接着构建了 1D ResNet 加 SE 注意力模型，并在验证集上监测准确率、AUC、F1 等指标；最后按比赛格式生成提交文件。材料的目标准确率是大于 98%，整个方案的核心在于用残差连接加通道注意力提取细微波形特征，并用类权重与分层抽样缓解不平衡问题。
