# 图像修复挑战（ELIVA 26 Image Restoration）

## 一、比赛简介

### 1.1 比赛背景

这是一个图像修复（Image Restoration）比赛。所谓图像修复，就是给一张"变差"的图片，尽量把它恢复成清晰、干净的样子。常见的退化有三种：

- **噪声**：画面里出现密密麻麻的颗粒，像老照片或高感光拍摄；
- **模糊**：边缘发虚，细节糊成一片，可能来自失焦、运动或压缩；
- **对比度不足**：整体发灰，亮的不亮、暗的不暗。

ELIVA 26 的比赛数据是一批 `.jpg` 图片，放在 `data26` 目录下。参赛者要写一个程序，把每张图处理后，按比赛要求的格式打包提交。和常见的"训练一个神经网络"不同，这个比赛的入门方案可以用**经典图像处理算法**完成：先去噪，再反卷积去模糊，最后做局部对比度增强。材料里的两份 notebook 正是这么做的。

### 1.2 比赛目标

输入是 `data26` 目录下的一组 JPG 图片。输出不是标签，而是**修复后的图片本身**——把图片编码成 base64 文本，放进 CSV 的一行里。提交表大致长这样：

| id | image（或 data） |
| --- | --- |
| 原图文件名，例如 `0001.jpg` | base64 编码后的图片字符串 |

也就是说，每张测试图对应一行，第一列是文件名，第二列是修复结果的编码。比赛方会把这些字符串解码回图片，再和"干净的原图"比较。

### 1.3 评价标准

图像修复通常用**峰值信噪比（PSNR）**和**结构相似度（SSIM）**来评分，两者都是**越高越好**。

- PSNR 衡量"像素层面的误差有多小"，单位是分贝（dB），预测和原图越接近，数值越大；
- SSIM 衡量"结构、亮度、对比度看起来像不像"，取值通常在 0 到 1 之间，1 表示完全一致。

材料里的代码注释提到 "Essential for SSIM"，说明作者在调整对比度时是冲着 SSIM 去的。不过材料没有给出比赛的具体评分公式，所以我们只说"以 PSNR/SSIM 这类图像质量指标评估"，不编造确切权重。

> 小提示：这类比赛有一个反直觉的地方——**过度处理比不处理更糟**。去噪太狠会把纹理抹平，锐化太猛会产生光晕，SSIM 反而下降。所以参数要保守。

---

## 二、解题过程

我们采用的是一条"经典三阶段"流水线：**非局部均值去噪 → Richardson-Lucy 反卷积去模糊 → CLAHE 局部对比度增强**。整条流水线没有需要训练的权重，所有"学习"都体现在参数选择上。下面按四步说明。

### 2.1 数据探索（EDA）

我们首先把输入目录里的图片列出来，确认文件数量和命名规则：

```python
import os
import glob

INPUT_DIR = '/kaggle/input/eliva-26-image-restoration/data26/'

# 收集所有 jpg，并按文件名排序，保证每次运行的顺序一致
image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, '*.jpg')))
print('图片数量:', len(image_paths))
print('前五个文件:', image_paths[:5])
```

排序这一步看似无关紧要，其实很重要：提交文件的行顺序如果能和测试集稳定对应，后面排查问题会方便很多；如果每次跑出来的顺序都不同，一旦某张图处理失败，很难定位是哪一张。

接着，我们逐张读入，检查图像能不能正常解码、尺寸和通道数是否一致：

```python
import cv2
import numpy as np

for path in image_paths[:5]:
    img = cv2.imread(path)          # OpenCV 默认按 BGR 三通道读入
    if img is None:
        print('读取失败:', path)
        continue
    print(os.path.basename(path), img.shape, img.dtype)
```

`img.shape` 会给出"高、宽、通道数"。如果所有图片都是三通道，我们就可以放心地用彩色处理流程；如果混有灰度图，`cv2.split` 之后通道数会对不上，需要单独处理。材料里的流程默认输入是彩色三通道图，`cv2.fastNlMeansDenoisingColored` 和 `cv2.COLOR_BGR2LAB` 都是按彩色图设计的。

从视觉上看，这类比赛图片通常同时存在颗粒噪声和轻微模糊。我们没有具体的退化参数（比如高斯核大小、噪声强度），所以只能按经验假设一个"小的高斯模糊 + 中等噪声"的模型，这也解释了后面为什么核设得比较小、迭代次数也不高。

### 2.2 数据清洗

图像任务的"清洗"和表格任务不太一样，主要是保证数据能安全进入算法：

1. **跳过读不出来的文件**。`cv2.imread` 失败时返回 `None`，如果直接送进后续函数会报错，整批任务就中断了。
2. **不缩放、不裁剪**。缩放会改变图像的高频信息，等于人为制造模糊；裁剪会改变比赛要求的输出尺寸。我们保持原分辨率。
3. **统一在 BGR 空间读入，在 LAB 空间做对比度**。OpenCV 默认 BGR，而 CLAHE 只对亮度通道操作，所以要先把 BGR 转成 LAB，改完 L 通道再转回来。

```python
def advanced_restore(img):
    """三阶段修复：去噪 → 去模糊 → 局部对比度增强"""
    # 阶段 1：非局部均值去噪
    # h=7 是滤波强度；彩色图用 hColor 控制颜色通道的强度
    denoised = cv2.fastNlMeansDenoisingColored(
        img, None, h=7, hColor=7,
        templateWindowSize=7, searchWindowSize=21)

    # 阶段 2：按通道做 Richardson-Lucy 反卷积
    channels = cv2.split(denoised)
    kernel = np.outer(cv2.getGaussianKernel(5, 1.5),
                      cv2.getGaussianKernel(5, 1.5))
    deblurred_channels = [richardson_lucy(c, kernel, iterations=5)
                          for c in channels]
    deblurred = cv2.merge(deblurred_channels)

    # 阶段 3：CLAHE 提升局部对比度
    lab = cv2.cvtColor(deblurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    final_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return final_img
```

这里每个参数都不是随手写的。`h=7` 是去噪强度：材料注释说"颗粒重就用 10，图本来就平滑就用 3"，7 是折中值。`5×5`、`sigma=1.5` 的高斯核对应"轻微失焦"的假设，核太大反而会把本来清楚的边缘糊掉。`clipLimit=1.5` 控制对比度增强的幅度，数值越大越激进，太大容易把噪声也放大。

### 2.3 模型训练与超参数调优

这一节的名字叫"模型训练"，但这个方案里**没有可训练的权重**，只有一组手工设定的参数。我们把"调参"理解为：在几组候选参数里，选一组视觉上最自然、又不产生明显伪影的。

核心算法是 **Richardson-Lucy 反卷积**。用大白话解释它的迭代过程：

1. 先假设一张"猜测图"（材料里直接初始化成 0.5 的灰色）；
2. 用假设的高斯核把猜测图模糊一遍；
3. 拿"模糊后的猜测图"和"真实观测到的模糊图"相除，得到每个像素该放大还是缩小；
4. 把这个修正量反投影回猜测图，更新它；
5. 重复若干次。

```python
from scipy.signal import convolve2d

def richardson_lucy(image, kernel, iterations=10):
    """迭代反卷积：让'猜测图模糊后'尽量接近输入图"""
    # 归一化到 0~1，避免数值过大导致迭代不稳定
    image = image.astype(float) / 255.0
    direct_kernel = kernel / kernel.sum()
    # 反投影要用转置核（上下、左右都翻转）
    flipped_kernel = np.flipud(np.fliplr(direct_kernel))

    # 初始猜测：整张 0.5 的灰图
    latent_est = np.full(image.shape, 0.5)

    for _ in range(iterations):
        # 当前猜测经过同样的模糊后，和观测图差多少
        relative_blur = image / (convolve2d(latent_est, direct_kernel,
                                           mode='same') + 1e-12)
        # 把差异反投影回去，修正猜测图
        error_step = convolve2d(relative_blur, flipped_kernel, mode='same')
        latent_est *= error_step

    # 截断到合法范围，再变回 0~255 的 uint8
    return (np.clip(latent_est, 0, 1) * 255).astype(np.uint8)
```

关于迭代次数：材料里函数默认写 10 次，但实际调用时传的是 5 次。这是一个典型的取舍——迭代越多，边缘越锐，但噪声和振铃伪影也会被一起放大。5 次是在"变清晰"和"不炸噪声"之间选的保守值。分母加 `1e-12` 是防止除零，这是数值计算里的常规保护。

调参时我们主要看三组对比：

- 去噪强度 `h` 取 3、7、10：太小颗粒还在，太大纹理被抹平；
- 反卷积迭代 5 次和 10 次：10 次更锐，但边缘容易出现"回声"；
- CLAHE 的 `clipLimit` 取 1.5 和 2.0：2.0 更通透，但暗部噪声会明显。

因为比赛没有提供"干净原图"，我们没法在本地算 PSNR/SSIM，只能靠放大局部、肉眼对比。这一点必须诚实说明：**没有参考图，就没有客观的本地分数**，参数选择带有主观性。如果比赛方提供了成对的训练图，正确做法是留出一部分图，算 PSNR/SSIM 再选参数。

### 2.4 模型测试与指标报告

确定参数后，我们遍历全部图片，逐张修复、编码、写入 CSV：

```python
import base64
import pandas as pd

results = []
for path in image_paths:
    img_id = os.path.basename(path)   # 用原文件名作为 id，保留 .jpg 后缀
    img = cv2.imread(path)
    if img is None:
        continue                       # 读不出来的图直接跳过，避免整批失败

    restored = advanced_restore(img)

    # 编码成 PNG 再转 base64：PNG 无损，适合承载处理后的像素
    ok, buffer = cv2.imencode('.png', restored)
    encoded = base64.b64encode(buffer).decode('utf-8')

    results.append({'id': img_id, 'image': encoded})

# 写提交文件，index=False 避免多出一列行号
pd.DataFrame(results).to_csv('submission.csv', index=False)
print('已生成 submission.csv，共', len(results), '行')
```

这里有一个容易踩的坑：材料里的两份 notebook 用了**不同的列名**。修复流水线那份写的是 `id` 和 `image`，而官方示例那份写的是 `id` 和 `data`。提交前必须以比赛的 `sample_submission.csv` 为准，列名错一个字就会判格式错误。我们会在生成文件后做一次核对：

```python
# 读回刚写出的文件，核对列名、行数和 id 是否完整
sub = pd.read_csv('submission.csv')
print(sub.columns.tolist())
print('行数:', len(sub))
print('id 是否唯一:', sub['id'].is_unique)

# 抽查第一行：base64 能否解码回一张图片
sample_bytes = base64.b64decode(sub.loc[0, 'image'])
arr = np.frombuffer(sample_bytes, dtype=np.uint8)
decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
print('解码后的图片形状:', None if decoded is None else decoded.shape)
```

这个"解码回来"的检查非常值得做。base64 编码本身很少出错，但如果图片写成了空字符串、或者列名取错导致取到 NaN，解码会立刻失败。与其等比赛方评分时才发现，不如自己先验一遍。

**指标报告。** 材料没有提供任何本地分数，也没有给出排行榜成绩，所以我们不编造 PSNR/SSIM 数字。我们能报告的是一套可复现的评估流程：如果手上有成对的"退化图—原图"，就对每张修复结果计算 PSNR 和 SSIM，再取平均；如果没有，就只能做视觉抽查——重点看三个地方：平坦区域（比如天空）还有没有颗粒、文字或边缘有没有光晕、暗部有没有被 CLAHE 拉出噪点。

作为对照，官方示例给出的"基线"其实什么都不做，只是对图片做一次 `sigma=5` 的高斯模糊：

```python
# 官方示例的 restore 函数：一个占位实现，用来跑通提交流程
def restore(img):
    # 高斯模糊只会让图更糊，作为基线它说明"格式能提交"，不代表效果好
    return cv2.GaussianBlur(img, (0, 0), sigmaX=5, sigmaY=5)

# 编码函数：示例用 JPG，体积比 PNG 小，但会有二次压缩损失
def encode_img(img):
    ok, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')
```

把这条"模糊基线"和我们的三阶段流水线放在一起看，会发现一个有意思的事实：**基线连方向都是反的**（模糊而不是去模糊），它唯一的作用是验证提交格式。这也提醒我们，看别人的 notebook 时要分清"能跑的示例"和"真正的方案"。

---

## 三、总结

本文中我们完成了 ELIVA 26 图像修复比赛的一套经典处理方案。我们首先用 `glob` 收集 `data26` 下的 JPG 并按文件名排序，逐张读入确认尺寸和通道；其次跳过解码失败的图片、保持原分辨率，并在 BGR 与 LAB 之间转换以便只增强亮度通道；接着用非局部均值去噪、5×5 高斯核的 Richardson-Lucy 反卷积和 CLAHE 三步组成流水线，对去噪强度、迭代次数和对比度上限做了保守的对比选择；最后把修复结果编码成 PNG 的 base64 字符串，按 `id`、`image` 两列生成提交文件，并做了列名、行数、唯一性和"解码回图片"四项校验。

这个比赛让我们认识到：**没有学习参数的方案也可以很讲究**。去噪、反卷积、对比度增强各自的参数都会互相影响，过度处理往往比不处理更伤 SSIM；而在没有成对参考图的情况下，任何"本地分数"都只是自我安慰，老老实实说明评估的局限，比编一个好看的数字更可靠。
