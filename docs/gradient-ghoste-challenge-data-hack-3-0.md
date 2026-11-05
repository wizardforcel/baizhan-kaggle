# Gradient Ghostes Challenge - DataHack 3.0

## 一、比赛简介

### 1.1 比赛背景

Gradient Ghostes Challenge 是 DataHack 3.0 黑客松里的一个「机器学习 + 密码破译」趣味题。比赛给出一份金融行情数据 `market_data.csv`、一个已经训练好的 oracle 模型 `oracle.pkl`，以及一段被加密的 `secrets.bin`。密码就藏在模型里：只有正确理解 oracle 模型对哪些特征最敏感，才能推出加密用的密钥，把密文还原成一段可读的 flag。

这类比赛的重点不是刷分数，而是考察参赛者能否读懂别人留下的模型产物（模型、标准化器、特征名、训练集下标），并把它当作密码学里的「旁路信息」使用。

### 1.2 比赛目标

输入是三个文件：

- `market_data.csv`：每天一行，含 `daily_return` 与若干特征列；
- `oracle.pkl`：一个字典，保存 `model`、`scaler`、`feature_names`、`train_indices`；
- `secrets.bin`：48 字节的密文，开头是明文 `FLAG{`。

输出是一个提交文件，只有一行，列为 `id` 与 `FLAG`，值就是解出来的完整 flag 字符串。

### 1.3 评价标准

比赛按提交的 flag 是否正确评分：完全匹配得满分，否则不得分。因此这是一道「零或一」的题，不需要调参，只需要把解密链路走通。

---

## 二、解题过程

我们按 EDA、清洗、建模（这里是「反推密钥」）、提交四步走。

### 2.1 数据探索（EDA）

第一步是加载三个产物，确认 oracle 字典里有哪些键：

```python
import joblib
base_path = '/kaggle/input/gradient-ghoste-challenge-data-hack-3-0'
df = pd.read_csv(f'{base_path}/market_data.csv')
oracle_dict = joblib.load(f'{base_path}/oracle.pkl')

with open(f'{base_path}/secrets.bin', 'rb') as f:
    ciphertext = f.read()          # 48 字节密文

model        = oracle_dict['model']
scaler       = oracle_dict['scaler']
feature_names = oracle_dict['feature_names']
train_indices = oracle_dict['train_indices']
```

`train_indices` 的存在说明 oracle 只在一部分行上训练过，我们必须复现同样的训练子集，否则算出的特征重要性会与加密时使用的不一致。

### 2.2 数据清洗

行情数据需要对齐「预测目标」：模型预测的是下一天的收益率，因此标签要用 `daily_return` 整体上移一行：

```python
returns = df["daily_return"].values
y = np.roll(returns, -1)     # 让今天的特征对应明天的收益
y[-1] = 0                    # 最后一行没有次日，补 0

X_raw = df[feature_names].values
X_scaled = scaler.transform(X_raw)     # 必须用 oracle 自带的 scaler，不能重拟合

X_train = X_scaled[train_indices]
y_train = y[train_indices]
```

这里的关键是「用原来的 scaler」，因为密钥的生成依赖模型在特定数值尺度下的行为，任何重新拟合都会让重要性排序改变。

### 2.3 模型训练与超参数调优（反推密钥）

我们没有重新训练模型，而是把 oracle 当作黑盒，用排列重要性（permutation importance）找出模型最依赖的 8 个特征，作为密钥 A 的下标：

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(model, X_train, y_train,
                              n_repeats=30, random_state=42)
Key_A = np.argsort(perm.importances_mean)[::-1][:8]
print(f"Key A: {Key_A}")
```

排列重要性的做法是把某一列随机打乱，看模型误差变差多少：越重要的特征，打乱后误差上升越多。`n_repeats=30` 表示重复 30 次取平均，让排序更稳定。

接下来是已知明文攻击（KPA）。我们知道 flag 以 `FLAG{` 开头，而加密由三层组成：先加 Key C，再加 Key A（取自特征重要性），最后与 Key B 异或。Key C 是模 26 的位移，只有 26 种可能，可以暴力枚举：

```python
known_prefix = "FLAG{"
for guess_C in range(26):
    derived_B = []
    for i in range(4):                                  # 用 "FLAG" 推导 Key B
        expected_v2 = (ord(known_prefix[i]) + guess_C + Key_A[i]) % 128
        b_val = ciphertext[i] ^ expected_v2
        derived_B.append(b_val)

    i = 4                                               # 用 "{" 验证
    expected_v2 = (ord(known_prefix[i]) + guess_C + Key_A[i % 8]) % 128
    b_val_check = ciphertext[i] ^ expected_v2

    if b_val_check == derived_B[i % 4]:                 # 推出的一组密钥自洽
        ...
```

Key B 是 4 字节循环使用的异或密钥，Key A 是 8 个下标循环使用。只要猜对 Key C，前 5 个字符就能推出一致的 Key B，验证通过即可断定密钥组合正确。

### 2.4 模型测试与指标报告

确认密钥后，对 48 字节逐字节逆运算：

```python
plaintext = ""
for j, byte in enumerate(ciphertext):
    v2 = byte ^ derived_B[j % 4]          # 先撤掉异或
    v1 = (v2 - Key_A[j % 8]) % 128        # 再减掉 Key A
    p  = (v1 - guess_C) % 128             # 最后减掉 Key C
    plaintext += chr(p)

if plaintext.endswith('}'):
    found_flag = plaintext
```

最后检查 `plaintext.endswith('}')`，这是天然的校验条件——flag 一定以右花括号结尾。校验通过后写提交：

```python
sub = pd.DataFrame({'id': [0], 'FLAG': [found_flag]})
sub.to_csv('submission.csv', index=False)
```

提交文件只有一行，id 为 0，FLAG 是解出的完整字符串。

---

## 三、总结

本文中我们完成了 Gradient Ghostes Challenge 的解题。我们首先加载 oracle 字典，确认模型、标准化器、特征名与训练下标都在，并按「预测次日收益」对齐标签、用自带 scaler 做变换。其次用排列重要性找出模型最依赖的 8 个特征作为 Key A。接着用已知明文 `FLAG{` 暴力枚举 26 种 Key C，推出并验证 4 字节的 Key B。最后逆序剥离三层加密，得到完整 flag 并生成只有一行的提交文件。

这个挑战最有意思的地方在于，它把「模型解释性」变成了攻击面：特征重要性本来是给我们看模型行为的工具，在这里却直接充当了密钥。它也提醒我们，做这类题时复现训练数据的预处理（scaler、训练子集）比模型本身更重要，差一个下标都解不出正确的字符串。
