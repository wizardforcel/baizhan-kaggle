# NFL 弃踢分析竞赛（NFL Punt Analytics Competition）

## 一、比赛简介

### 1.1 比赛背景

**美式橄榄球里的弃踢（Punt）**，是进攻方在第四档把球踢向对方半场的动作，目的是用「换场地」换到一个对自己更有利的球场位置。弃踢回合里，踢球方会派两名速度最快的球员（外号 gunner，炮手）全力冲向接球方，而接球方则会派人（外号 jammer，干扰员）在中途拦截他们。因为双方都朝着对方高速对冲，弃踢成了橄榄球里脑震荡（concussion）风险最高的回合之一。

2018 年，美国橄榄球联盟（NFL）在 Kaggle 上举办这场分析竞赛，把 2016、2017 两个赛季的「次世代统计」（Next Gen Stats，球员逐帧追踪数据）公开，请公众通过数据分析找出「什么情况下最容易发生脑震荡」，并据此提出修改规则的建议，目标是**在不破坏比赛观赏性、不砍掉弃踢战术魅力的前提下，降低弃踢回合的受伤率**。

这场比赛有两大难点：一是数据又多又杂——球员的逐帧运动轨迹被拆成了 10 个 CSV 文件；二是「坏结果」样本太少——全程 6681 次弃踢里，被视频复盘确认的脑震荡只有 37 次，不到千分之六。这给「从数据里找规律」带来了巨大的挑战。

### 1.2 比赛目标

比赛提供了 6 类核心数据：

- `game_data.csv`：逐场比赛信息，共 666 场（赛季、周数、球场、草皮、天气、温度等）；
- `play_information.csv`：逐回合信息，共 6681 条，全部是弃踢回合（持球方、场上位置、比赛时钟、比分、一段文字描述）；
- `play_player_role_data.csv`：每个球员在回合里的角色（如 GL/GR 炮手、VLi/VRo 干扰员、PLS 长开球手等）；
- `player_punt_data.csv`：球员的传统位置信息；
- `video_review.csv`：37 次脑震荡事件的复盘（撞击类型、受伤球员在做什么、撞击对象在做什么）；
- `NGS-*.csv`：球员每一帧的轨迹（位置、速度、方向、身体朝向）。

所以这场比赛**没有「猜一个数字」的得分文件**。我们要交付的是一份**规则修改提案**：先通过数据分析论证脑震荡的主要成因，再提出能减少脑震荡、同时不破坏比赛的可执行规则。

### 1.3 评价标准

这是一场由 NFL 评委主观评审的开放题比赛，官方给出了两个评分维度：

1. **方案有效性（Solution Efficacy）**：能不能用数据分析清晰地证明你对脑震荡成因的理解，并说明拟议规则为什么能降低这些伤害；
2. **游戏完整性（Game Integrity）**：提案要务实可行、不破坏橄榄球的策略性与观赏性，同时还要讨论新规则可能引入的新风险。

因此没有一个「越高越好」的数值排行榜。我们自己的评估办法是：把 37 次脑震荡逐条对照拟议规则，数一数其中有多少次能靠规则拦下来——这个「命中数 / 37」就是我们的自评指标。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**先用 EDA 认识数据，再用 NGS 逐帧数据给每个回合计算速度、方向、朝向等运动特征，接着用互信息特征选择找出与脑震荡最相关的因素，最后把发现翻译成可落地的规则。** 我们按 EDA、数据清洗、模型训练、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

我们首先把所有 CSV 读进来，摸清整体规模：

```python
import pandas as pd

gd = pd.read_csv('../input/game_data.csv')               # 比赛信息表
pi = pd.read_csv('../input/play_information.csv')        # 回合信息表
ppd = pd.read_csv('../input/player_punt_data.csv')       # 球员位置表
pprd = pd.read_csv('../input/play_player_role_data.csv') # 球员角色表
vr = pd.read_csv('../input/video_review.csv')            # 脑震荡复盘表

print('游戏场数:', gd.shape[0])      # 666 场
print('弃踢回合数:', pi.shape[0])   # 6681 次
print('脑震荡事件数:', vr.shape[0]) # 37 次
```

输出告诉我们：材料覆盖 666 场比赛、6681 次弃踢，但真正造成脑震荡的只有 37 次。这是一个典型的「极少数正样本」问题——后面做特征选择时，必须想办法处理这种不均衡。

其次我们想看这 6681 次弃踢分别是什么结果。主办方没有直接给「这次是回跑、还是安全接球」的标签，只给了一段文字描述（PlayDescription），我们于是用关键词匹配给每个回合打标签：

```python
# 从文字描述里识别回合结果, 用的是字符串匹配
pi['Touchback'] = pi['PlayDescription'].str.contains('Touchback')   # 端区触球
pi['Fair Catch'] = pi['PlayDescription'].str.contains('fair catch') # 安全接球
pi['Downed'] = pi['PlayDescription'].str.contains('downed')         # 球被封堵在地

# 统计各类结果的占比
print('安全接球占比:', pi['Fair Catch'].mean())   # 24.83%
print('端区触球占比:', pi['Touchback'].mean())

# 被回跑的回合, 我们还可以从文字里抽出回跑码数
pi['return_yards'] = pi[pi['PlayDescription'].str.contains(' for ')] \
    .PlayDescription.str.extract(r'for (\d+) yard')[0].astype(float)
```

占比显示：34.67% 的弃踢被回跑且取得了正码数、24.83% 是安全接球、5.69% 回跑零码、2.42% 回跑负码。这个分布给了我们第一个直觉：**有相当比例的弃踢，既没有带来精彩的回跑，又把球员置于了不必要的撞击风险里。**

### 2.2 数据清洗

其次我们处理 NGS 这个大块头。NGS 文件按「赛季 + 比赛类型 + 周数」拆成了 10 份，例如 `NGS-2016-reg-wk13-17.csv`。为了能按回合自动找到对应的文件，我们写了个查询函数：

```python
def getNGSName(season, week, season_type):
    """根据赛季、比赛类型和周数, 返回对应的 NGS 文件名"""
    if season_type == 'Pre':                  # 季前赛
        return 'pre-' + str(season)
    elif season_type == 'Post':               # 季后赛
        return 'post-' + str(season)
    else:                                     # 常规赛按周平均切成三段
        if week <= 6:
            return 'reg-' + str(season) + '-wk1-6'
        if 7 <= week <= 12:
            return 'reg-' + str(season) + '-wk7-12'
        return 'reg-' + str(season) + '-wk13-17'
```

接着我们把球员角色表整理成「阵型」：用 `比赛号_回合号` 拼出一个全局唯一的键，把每个回合里上场的角色排好序串起来。阵型里名字以 V 开头的（VLi、VRi、VR、VL……）是干扰员，接球方派几个干扰员拦截炮手，是后面分析脑震荡的关键变量：

```python
# 踢球方角色集合, 用它在 pprd 里标出「这个球员属于踢球方还是接球方」
PUNTPOS = {'PLS','PLG','PLT','PLW','PRG','PRT','PRW','PC','PPR','P','GL','GR'}

# 用 比赛号_回合号 拼出全局唯一键
pprd['game_play_key'] = pprd['GameKey'].astype(str) + '_' + pprd['PlayID'].astype(str)

# 按 (回合, 攻防方) 分组, 把该方所有角色排序后拼成阵型字符串
form = (pprd.assign(punting_team=pprd['Role'].isin(PUNTPOS))
             .groupby(['game_play_key', 'punting_team'])['Role']
             .apply(lambda x: ', '.join(sorted(x))).to_frame('form'))
```

我们把接球方阵型按干扰员数量（0~4）统计，发现 2 干扰员阵型出现 3402 次、3 干扰员阵型 1818 次、4 干扰员阵型 1344 次——而这三种出现最多的主流阵型，恰好都发生了脑震荡。这是我们要重点解剖的地方。

### 2.3 模型训练与超参数调优

接着我们进入核心：给每个回合生成运动特征。对每个球员，我们用他在整个回合里每一帧的速度、方向、身体朝向取平均，作为该角色在本回合的特征。NGS 给的是相邻两帧的位移 `dis`，要换算成英里/小时的时速用 `dis * 20.5`；回合的悬空时间（hang time）则用「球被踢出」和「球被接到」两个事件的时间差计算：

```python
def makeFeatureVector(row):
    """根据一个回合, 计算该回合的所有运动特征"""
    ngs = NGSDICT[getNGSName(row['Season_Year'], row['Week'], row['Season_Type'])]
    thisplay = ngs.query('GameKey == %d and PlayID == %d'
                         % (row['GameKey'], row['PlayID']))

    feats = {}
    # 悬空时间 = 球被接到的时间 - 球被踢出的时间(秒)
    kicked = pd.to_datetime(thisplay[thisplay['Event'] == 'punt']['Time']).min()
    caught = pd.to_datetime(thisplay[thisplay['Event'].isin(['punt_received','fair_catch'])]['Time']).min()
    feats['HangTime'] = (caught - kicked).total_seconds()

    # 逐角色统计平均运动状态: 时速(英里)、方向、身体朝向
    for role, gsisid in role_of_each_role.items():
        pl = thisplay[thisplay['GSISID'] == gsisid]
        feats[role + '_AVG SPEED'] = (pl['dis'] * 20.5).mean()  # 一帧位移 x 20.5 -> 时速
        feats[role + '_AVG DIR'] = pl['dir'].mean()             # 运动方向(度)
        feats[role + '_AVG O'] = pl['o'].mean()                 # 身体朝向(度)
    return feats
```

我们开了 10 个进程并行跑这个函数，把 6681 个回合全部算完，得到约 6660 个有效特征向量。每个向量有约一百维特征：悬空时间、场上位置、草皮天气等比赛信息，以及每个角色平均速度/方向/朝向。

特征建好后，要回答「哪些特征和脑震荡关系最大」。正样本只有 37 个，直接算互信息会非常不稳定，所以我们采用「随机采样 + SMOTE 过采样 + 互信息选择」的组合：每次从样本里抽 500 个回合，把脑震荡样本插值补足到约四分之一，再跑一次 SelectKBest，重复 10 次后把分数累加取平均，抵消随机波动：

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

selectResults = {}
for _ in range(10):                    # 重复 10 次, 结果更稳定
    # SMOTE: 给少数类(脑震荡)插值出更多样本, 平衡正负比例
    fixed = setdf.apply(LabelEncoder().fit_transform)
    X_res, y_res = SMOTE(random_state=2).fit_resample(fixed, rs)
    # 按互信息挑出与脑震荡最相关的 50 个特征
    sel = SelectKBest(mutual_info_classif, k=50).fit(X_res, y_res)
    for name, score in zip(fixed.columns, sel.scores_):
        selectResults[name] = selectResults.get(name, 0) + score

# 打印得分最高的 5 个特征(10 次平均分)
for name in sorted(selectResults, key=selectResults.get, reverse=True)[:5]:
    print(name, round(selectResults[name] / 10, 4))
```

得分最高的特征几乎全是 `PLW_AVG O`、`PDL2_AVG DIR`、`PDR2_AVG DIR`、`P_AVG O` 这类**踢球方球员的身体朝向和运动方向**。这和我们逐帧看录像得到的直觉完全一致：踢球方球员为了回追回跑者而转身时，最容易在「追防的视野死角」里被高速撞到——脑震荡往往就发生在这些转身追防的球员身上。

### 2.4 模型测试与指标报告

最后，我们把「特征选择给出的信号」翻译成可验证的结论。我们从 `form` 里取出接球方那半边（`punting_team=False` 的行），先按干扰员数量统计脑震荡在哪类阵型里发生，再看受伤的球员属于哪一方：

```python
def countJammers(df, tag):
    """把阵型字符串数一数, 统计有几个以 V 开头的干扰员"""
    counts = {}
    for f in df['form'].dropna():
        j = sum(1 for r in str(f).split(', ') if r.startswith('V'))
        counts[j] = counts.get(j, 0) + 1
    for n, c in sorted(counts.items()):
        print('阵型含 %d 个干扰员: %d 次(%s)' % (n, c, tag))
    return counts
```

统计结果显示：2 干扰员阵型在 37 次脑震荡里占了 13 次、3 干扰员占 12 次、4 干扰员占 12 次；而在 4 干扰员阵型里，受伤球员属于踢球方的比例高达 83.3%。原因也说得通：干扰员把炮手拦死后，中路的防守被掏空，回跑者就沿边线向前推进，踢球方多名球员被迫转身回追——一旦在回追中被对方球员迎面高速撞上，就是脑震荡。

基于这些证据，我们提出两条规则，并用「37 次里能拦下几次」自测效果：

1. **向己方端区方向阻挡，判罚 15 码**：37 次脑震荡里有 16 次发生在「朝自己端区方向做阻挡」时（16/37）。把这种阻挡直接判为不必要冲撞，等于从规则上消灭了最容易出事的那片「危险区」；
2. **弃踢回合中头盔对头盔撞击，直接罚下场**：同样对应 16/37 次事件，在保留原有判罚的基础上加重惩罚，让球员不敢做危险动作。

我们也明确论证了为什么不采用另外三种「看起来省事」的改动：限制起步阵型会抹杀假弃踢的战术空间；奖励安全接球会破坏进攻平衡；把触球线从 20 码改成 15 码，会逼得踢球方每次都把球一脚踢出端区，减少回跑场面、降低观赏性。

按官方评审的两个维度来复盘自己的提案：方案有效性层面，我们用「特征选择 + 阵型统计 + 录像核对」三重证据回答了「脑震荡从哪来」；游戏完整性层面，我们逐条讨论了新规则的副作用，并故意回避了三类伤筋动骨的改动——这正是这场「以报告为交付物」的比赛最希望看到的交付形态。

---

## 三、总结

本文中我们走完了 NFL 弃踢分析竞赛的完整流程。我们首先从 666 场比赛、6681 次弃踢、37 次脑震荡出发做 EDA，确认了正样本极其稀少，并统计出回跑、安全接球、触球等结果的分布。其次我们做了数据清洗：把按周拆分的 NGS 文件按回合对齐，梳理出每个回合双方的阵型。接着我们给每个回合生成速度、方向、朝向、悬空时间等约一百维特征，用「SMOTE + 互信息特征选择 × 10 轮」找出与脑震荡最相关的是踢球方球员的朝向与方向。最后我们用阵型统计验证了「干扰员越多、踢球方越危险」的机制，提出「向己方端区阻挡判罚 15 码」「头盔对头盔立即罚下」两条规则，并逐条核对 37 次事件，认为 16/37 可以用规则拦下。另外，评分体系里同样被看重的另一支队伍把重点放在「奖励安全接球、扩大无防备球员定义、限制双人包夹炮手」上，思路不同，但都坚持以数据作为论证基石。

这个比赛给我们的最大启发是：**当比赛不看排行榜分数、只评报告时，特征选择、统计验证和「把结论翻译成可执行规则」的能力，比堆一个精度更高的模型更重要。**