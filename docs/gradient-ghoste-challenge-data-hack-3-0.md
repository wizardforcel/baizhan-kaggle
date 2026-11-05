# Gradient Ghostes Challenge - DataHack 3.0

## 一、比赛简介

### 1.1 比赛背景

Gradient Ghostes Challenge 是 DataHack 3.0 黑客松里的一个「机器学习 × 密码破译」趣味题。它和常规的 Kaggle 比赛很不一样：不需要折腾大量数据去冲分数，而是给出一个已经训练好的“oracle”模型和一个加密文件，要求参赛者理解这个机器学习模型，再利用它把密文解密成一段可读的 flag。说白了，这场比赛想要检验的是：你能不能把一个训练好的模型当作黑盒，从它的行为里抽出信息，做点“模型本不该让你做的事”。

比赛提供三个文件：

- `market_data.csv`：一份按天排列的金融行情数据，包含 `daily_return`（每日收益率）和若干特征列；
- `oracle.pkl`：一个用 `joblib` 保存的字典，里面有 `model`、`scaler`、`feature_names`、`train_indices` 四个键；
- `secrets.bin`：48 字节的二进制密文，其明文以 `FLAG{` 开头。

### 1.2 比赛目标

原始数据文件夹里有 `market_data.csv`、`oracle.pkl` 和 `secrets.bin`。我们要做的是：读懂 oracle 的预测逻辑，推出一组密钥，把 `secrets.bin` 还原成完整明文，最后生成一个只有一行两列的提交文件：

```python
sub = pd.DataFrame({'id': [0], 'FLAG': [found_flag]})
sub.to_csv('submission.csv', index=False)
```

`id` 固定为 0，`FLAG` 必须是一整段字符串，多一个字符、少一个字符都会被判错。

### 1.3 评价标准

按 flag 是否完全正确评分：解对得满分，解错不得分。没有分数梯度，也不需要调参，属于“零或一”的判断题。这对检查能力的要求更严格——只要有一层密钥推错，最终字符串就会变成乱码，需要在解密链路里加校验来确认成功。

第一次看到这个题，很多人会下意识去找“训练一个更好的分类器”，但这里根本没有测试集需要预测。真正的目标藏在名字里：`oracle.pkl` 里那份 `model` 才能告诉我们密钥。所以第一步不是建模，而是把它的四个键全部加载出来，弄清每个键的用途。这件事几乎决定了后续所有步骤能否走通。

---

## 二、解题过程

我们按 EDA（理解数据结构）、数据清洗（对齐标签与预处理）、密钥推理（替代传统“模型训练”步骤）、测试与提交四步走。

### 2.1 数据探索（EDA）

先把三个文件都读进来，看看 oracle 里到底有什么：

```python
import joblib
import pandas as pd

base_path = '/kaggle/input/gradient-ghoste-challenge-data-hack-3-0'
df = pd.read_csv(f'{base_path}/market_data.csv')
oracle_dict = joblib.load(f'{base_path}/oracle.pkl')

with open(f'{base_path}/secrets.bin', 'rb') as f:
    ciphertext = f.read()

model         = oracle_dict['model']
scaler        = oracle_dict['scaler']
feature_names = oracle_dict['feature_names']
train_indices = oracle_dict['train_indices']

print(df.shape, len(ciphertext), len(feature_names))
```

打印后我们确认了三件事：数据表是一个典型的回归/分类特征表；密文共 48 字节，与之对应的 flag 也应是约 48 个字符；特征名列表告诉我们模型用到了哪些列。`train_indices` 说明 oracle 不是用全部数据训练的，我们必须还原它见过的训练子集，否则后面的特征重要性会失真。

### 2.2 数据清洗

清洗的要点有三：对齐标签、统一缩放、限制训练集。

`daily_return` 是当天的收益率，而模型预测的目标是“下一天”。材料用 `np.roll` 把整列向前移一位，得到每个特征行对应的标签：

```python
returns = df["daily_return"].values
y = np.roll(returns, -1)   # 今天的特征，预测明天的收益率
y[-1] = 0                  # 最后一行没有“明天”，补 0
```

特征必须用 oracle 自带的 `scaler` 变换，而不是重新 `fit_transform`。原因很简单：如果重新拟合标准化器，特征的数值尺度就变了，模型看到的分布与当初训练时不一致，算出的特征重要性也失去意义。最后用 `train_indices` 把数据截到 oracle 真正训练过的部分：

```python
X_raw = df[feature_names].values
X_scaled = scaler.transform(X_raw)

X_train = X_scaled[train_indices]
y_train = y[train_indices]
```

这三步分别对应“标签对不对、尺度对不对、子集对不对”，任何一步出错，密钥 A 都不会正确。

### 2.3 模型训练与超参数调优（密钥推理）

我们不需要训练模型，需要的是从 oracle 里抽出密钥。

**第一步：求密钥 A。** 密钥 A 是加密时使用的 8 个下标，它恰好对应模型最依赖的 8 个特征。找到它们的方法是排列重要性（permutation importance）：把某一列的值随机打乱，看模型误差上升多少。误差上升越多，说明这列越重要。材料里对每个特征重复 30 次取平均，保证排序稳定：

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(model, X_train, y_train,
                              n_repeats=30, random_state=42)
Key_A = np.argsort(perm.importances_mean)[::-1][:8]
print(f"Key A: {Key_A}")
```

`argsort(...)[::-1]` 得到重要性从高到低的排列，再取前 8 个，就是密钥 A。这里的直觉是：构造 flag 的人把“模型最敏感的 8 个特征”选作了密钥的一部分，所以重要性排序本身就是解谜说明书。

**第二步：求密钥 C 与密钥 B。** 加密是三层叠加的：先对明文字符加一个模 128 的位移，接着再加上密钥 A 对应的值，最后与密钥 B 异或。密钥 C 只模 26，可能值只有 26 个，完全可以暴力枚举。密钥 B 是一个 4 字节循环使用的异或密钥。

我们利用已知明文前缀 `FLAG{`：如果猜对了密钥 C，用前 4 个字符 `FLAG` 就能反推出 4 字节的密钥 B，再用第 5 个字符 `{` 验证。推导出来的一组 B 自洽，就说明这组密钥是对的：

```python
known_prefix = "FLAG{"
found_flag = None

for guess_C in range(26):                    # 枚举 26 种可能的 C
    derived_B = []
    # 用 "FLAG" 四个字符反推密钥 B
    for i in range(4):
        expected_v2 = (ord(known_prefix[i]) + guess_C + Key_A[i]) % 128
        b_val = ciphertext[i] ^ expected_v2   # 异或层上，v2 异或 B = 密文
        derived_B.append(b_val)

    # 用 "{" 做交叉验证
    i = 4
    expected_v2 = (ord(known_prefix[i]) + guess_C + Key_A[i % 8]) % 128
    b_check = ciphertext[i] ^ expected_v2

    if b_check == derived_B[i % 4]:
        print(f"Cracked Key C : {guess_C}")
        print(f"Cracked Key B : {derived_B}")
        break
```

三层加密中 `% 128` 指字符都在 ASCII 可打印区间；密钥 A 按 8 个一组循环，密钥 B 按 4 个一组循环，因此下标要分别取模 8 和模 4。

### 2.4 模型测试与指标报告

拿到密钥后，对全部 48 字节按“先还原异或、再减密钥 A、最后减密钥 C”的顺序解密：

```python
plaintext = ""
for j, byte in enumerate(ciphertext):
    v2 = byte ^ derived_B[j % 4]     # 撤掉异或层
    v1 = (v2 - Key_A[j % 8]) % 128   # 减掉特征密钥
    p  = (v1 - guess_C) % 128        # 减掉位移密钥
    plaintext += chr(p)

# 校验：flag 一定以 "}" 结尾
if plaintext.endswith('}'):
    found_flag = plaintext
```

最后一步是质量检查：若 `plaintext.endswith('}')`，说明前缀、中间字节都自洽——任何一个密钥错了，这里都会变成乱码。确认后写入提交文件。

```python
if found_flag:
    sub = pd.DataFrame({'id': [0], 'FLAG': [found_flag]})
    sub.to_csv('submission.csv', index=False)
    print(f"RECOVERED FLAG: {found_flag}")
```

---

## 三、总结

本文中我们完成了 Gradient Ghostes Challenge 的全部流程。我们首先加载 oracle 字典，确认其中包含模型、标准化器、特征名与训练下标；其次把标签对齐到“次日收益率”，用自带 scaler 变换并截取训练子集。接着我们用排列重要性找出模型最依赖的 8 个特征作为密钥 A，再用已知明文 `FLAG{` 暴力枚举 26 种密钥 C，反推并验证 4 字节的密钥 B。最后逐字节逆运算还原整段明文，用结尾的 `}` 校验后生成提交文件。

这个挑战最有意思的地方在于它把“模型可解释性”变成了攻击面：特征重要性本来是帮助我们理解模型行为的工具，在这里却直接充当了解密密钥的一部分。它提醒我们两件事：其一是做这类题时，**复现训练数据的预处理（scaler、训练子集）比什么都重要**，差一步都会导致密钥失真；其二是当解决方案看起来像“暴力枚举”时，只要把搜索空间压缩到可接受的范围（这里只有 26 种），枚举就是最实用、最不容易出错的手段。