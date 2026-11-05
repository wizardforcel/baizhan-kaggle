# Python–PostgreSQL 数据管道优化挑战

## 一、比赛简介

### 1.1 比赛背景

这个比赛研究的对象是一个叫 `kaggle_postgre_challenge` 的参考应用。它做的事情是：把数据库里一堆「半结构化」的原始记录，展开（Expand）成多张规整的关系表。所谓半结构化，指的是数据虽然存在数据库里，但它的内容是一段 JSON 文本——字段个数、字段名字都可能不一样，没法像普通表格那样一行一行地对齐。PostgreSQL 里专门有一个类型叫 `JSONB`，用来存放这种 JSON 数据，它比普通文本多了「可以直接按字段查询」的能力，所以很适合做这类数据的落脚点。

具体来说，原始记录放在 `raw_data(id, raw jsonb)` 这张表里：`id` 是自增编号，`raw` 是一整段 JSON。Expand 操作要做三件事：读出一条 `raw`，在 Python 里检查这条记录里的「来源」（origin）和「去向」（dest）字段是否符合约定，然后把拆出来的值分别插进多张目标表。数据规模会从一万条（$10^4$）一路涨到一百万条（$10^6$），而数据库又是通过 SSH 隧道远程连接的——SSH 隧道可以理解成「把数据库流量包在一层加密通道里传输」，安全但会多出网络延迟。

作者在材料里把基线版的三个瓶颈讲得很清楚：

1. **网络往返次数随数据量线性增长（$O(N)$ 次往返）**。基线是一行一行地发 `INSERT`，或者一次只发一小批。每发一条语句，都要经过一次 TCP/SSH 往返，材料给出的经验值是每次 1～5 毫秒。数据一万行时还能忍，到一百万行时，光「等网络回话」就吃掉了绝大部分时间。
2. **反复在 Python 和数据库之间搬运 JSON**。基线把 `jsonb` 解析成 Python 字典，校验完又把参数重新序列化回 PostgreSQL，同时每插一行还要更新目标表的 B-tree 索引。B-tree 是数据库用来加速查找的树形索引结构，逐行插入意味着索引被反复小改，开销很大。
3. **中途失败不能「续跑」**。一次展开要跑好几分钟；如果中途网络断了，要么把整段事务回滚（前面几分钟白干），要么硬着头皮继续，结果就是表里出现重复数据。这种「跑了半截怎么安全接上」的能力，术语叫**幂等恢复**——同一个操作重复执行多次，结果和只执行一次一样。

### 1.2 比赛目标

比赛要求我们**在不改变最终结果的前提下，把这个 Expand 管道跑得更快、更稳**。

输入是 `raw_data` 表里的记录。每条 `raw` 大致长这样（这是材料里构造基准数据时的形状）：

```json
{"origin": "ORG_7", "dest": "DST_23", "payload": {"val": 10.5, "valid": true}}
```

其中 `origin` 是来源编码，`dest` 是去向编码，`payload.val` 是一个数值，`payload.valid` 是一个布尔标记，表示这条记录是否有效。

输出是展开后的目标关系表。在材料给出的可复现基准里，目标表叫 `expanded_nodes`，列为 `(rec_id, origin, dest, val)`；在真实应用里则是多张规范化后的目标表。除了写数据，管道还要维护一张检查点账本 `expand_checkpoint_ledger(chunk_id, min_id, max_id, row_count, status_hash)`，记录每一批处理到哪儿、写了多少行、这批数据的指纹是什么。

简言之：**输入是异构的 JSONB 原始记录，输出是展开并校验后的规整表格行，外加一份可以据此断点续跑的检查点记录。**

### 1.3 评价标准

材料没有直接贴出排行榜的计分公式，但从作者自带的「可复现基准与等价性验证」脚本可以读出考核的两条主线：

- **结果等价性（bit-exact parity）**：优化后的管道写出的行数、以及 `val` 的求和，必须和基线完全一致。材料里的脚本用一句 `assert cnt1 == cnt2` 把这个要求钉死——数字对不上就直接报错，成绩无从谈起。
- **执行耗时与加速比**：脚本分别给基线和优化版计时，最后打印 `Speedup = 基线时间 / 优化时间`。耗时越短、加速比越高越好。

换句话说，这是一个「**既要对、又要快**」的比赛：先保证一字不差地产出同样的结果，再比谁把网络往返、CPU 解析和索引维护这三块开销压得更低。只快不对没有意义，只对不快拿不到分。

---

## 二、解题过程

整场比赛的思路可以概括成一句话：**把「一行一行搬」换成「一批一批搬」，把「Python 里拆」换成「数据库里拆」，再用检查点账本让中断可以安全续跑。** 我们按 EDA、数据清洗、管道优化与参数调优、测试与指标报告四步来走。

### 2.1 数据探索（EDA）

做性能优化，第一步不是改代码，而是**先造一个能反复跑的、大小可控的数据集**。真实的一百万行数据我们不可能每次都跑一遍，所以材料用一段 Python 直接生成两万条形状逼真的记录做基准。这一步既让我们看清数据分布，也让后面的「基线 vs 优化」对比有统一的尺子：

```python
import json, hashlib, time, sqlite3

N_RECORDS = 20000          # 基准规模：两万条，够暴露瓶颈又不至于跑太久
raw_records = []
for i in range(1, N_RECORDS + 1):
    # origin 每 50 个一循环，dest 每 80 个一循环，模拟「来源/去向取值有限」的真实情况
    # payload.valid 让 i 是 17 的倍数的记录无效，模拟真实数据里的坏行
    raw_records.append((
        i,
        json.dumps({
            "origin": f"ORG_{i % 50}",
            "dest":   f"DST_{i % 80}",
            "payload": {"val": i * 1.5, "valid": (i % 17 != 0)},
        }),
    ))

# 数一数有多少条会被判为无效——无效行不入目标表，但仍要占用读取和解析的成本
invalid = sum(1 for _, s in raw_records if not json.loads(s)["payload"]["valid"])
print("总记录:", N_RECORDS, "无效记录:", invalid, "有效记录:", N_RECORDS - invalid)
```

从生成规则里我们能读出三件事，它们直接决定了后面的优化方向：

- **来源和去向的取值是有限的**（50 种来源、80 种去向），所以展开后的目标表不会有爆炸式增长，真正的问题在「怎么搬」，不在「搬多少」。
- **每 17 条里有 1 条是无效的**（`i % 17 == 0` 时 `valid` 为假），大约占 5.9%。校验必须逐条做，但不能逐条往返数据库——这就指向「在内存里批量校验」。
- **每条记录很小**，只有几个字段。小记录意味着单条 `INSERT` 的「有效载荷」很低，网络往返的固定开销（1～5 毫秒）反而占了大头。这是典型的「**小包高频**」场景，批量化的收益会非常明显。

接着我们把基线版的写法原样跑一遍，确认瓶颈确实存在：

```python
t0 = time.perf_counter()
con1 = sqlite3.connect(":memory:")   # 基准里用内存 SQLite 代替 PostgreSQL，避免依赖外部服务
con1.execute("CREATE TABLE expanded_nodes (rec_id INT PRIMARY KEY, origin TEXT, dest TEXT, val REAL)")

for rid, raw_js in raw_records:                 # 一条一条来：对应基线的逐行 INSERT
    obj = json.loads(raw_js)                    # 每条都做一次 JSON 反序列化
    if obj["payload"]["valid"]:                 # 逐条校验
        con1.execute("INSERT INTO expanded_nodes VALUES (?, ?, ?, ?)",
                     (rid, obj["origin"], obj["dest"], obj["payload"]["val"]))
con1.commit()
t_baseline = (time.perf_counter() - t0) * 1000.0

# 记录基线结果的「指纹」，后面用来做等价性比对
cnt1 = con1.execute("SELECT COUNT(*), SUM(val) FROM expanded_nodes").fetchone()
print("基线耗时(ms):", round(t_baseline, 2), "行数与数值和:", cnt1)
```

我们把这段代码和前面的瓶颈诊断对上了号：两万条记录就是两万次 `json.loads`、两万次 `execute`。虽然这里用内存 SQLite 把网络延迟抹掉了，但「逐条解析 + 逐条执行」的 CPU 开销依然在，规模一大就线性膨胀。这就是我们要动刀的地方。

### 2.2 数据清洗

这个比赛里的「数据清洗」不是去空值、去重复，而是**校验和去重**，而且这两件事都必须又快又准。

第一件事是校验。基线在 Python 里逐条判断 `origin` / `dest` 的 schema（也就是字段是否存在、类型对不对）。优化版把这个动作**批量化和编译化**：用 `orjson`（一个 C 语言实现的快速 JSON 库）一次性把一批记录反序列化，再用编译好的校验器统一过一遍，把有效行挑出来。这样每条记录的解析开销大幅下降，而且解析和校验都在内存里完成，不产生任何数据库往返：

```python
import orjson

def validate_batch(batch, required=("origin", "dest", "payload")):
    """把一批 (id, raw) 记录解析+校验，返回可以直接写库的有效行列表。
       返回列表而不是逐行写库，是为了把「计算」和「IO」彻底分开。"""
    valid_rows = []
    for rid, raw in batch:
        obj = orjson.loads(raw)              # 比标准 json 更快的反序列化
        # 三个必需字段缺一不可，缺了就跳过并计入坏行
        if not all(k in obj for k in required):
            continue
        payload = obj.get("payload") or {}
        if not payload.get("valid"):         # 业务规则：valid 为假的行不入目标表
            continue
        valid_rows.append((rid, obj["origin"], obj["dest"], payload["val"]))
    return valid_rows
```

第二件事是去重，也就是**幂等**。如果一批数据因为重试被写了两次，目标表不能出现重复行。我们的做法是把「去重」交给数据库本身：目标表的主键是 `rec_id`，插入时用 `INSERT OR IGNORE`（在 PostgreSQL 里对应 `ON CONFLICT DO NOTHING`）。这样无论同一批数据被提交多少次，结果都一样——这就是幂等的含义：

```python
# 同一批数据重复提交两次，第二次会被主键挡掉，行数不会翻倍
con2.executemany(
    "INSERT OR IGNORE INTO expanded_nodes VALUES (?, ?, ?, ?)",
    valid_rows,
)
# 关键点：把「去重」的责任放在数据库约束上，而不是靠 Python 自己记账。
# 这样即使进程崩溃后重跑，也不会产生重复数据。
```

做完这两步，我们手上就有了一个干净、可重放的「有效行」列表，下一步就是想办法把它们高效地送进数据库。

### 2.3 管道优化与超参数调优

这是整个比赛的核心。我们把原来的「逐行展开」替换成一个**三阶段、集合式（set-based）的批量展开引擎**。所谓集合式，就是尽量用一条 SQL 处理一整批数据，而不是在程序里循环。

**第一阶段：按主键分页取数（keyset pagination）。** 传统分页用 `OFFSET`，页码越大数据库越要先跳过前面所有行，越翻越慢。keyset 分页改成「记住上一批处理到哪个 id，下一批从它后面取」，无论翻到第几页速度都一样。这批的取数上限就是我们要调的超参数 `batch_size`，材料里默认取 50000：

```python
# 每批取 50000 条；last_checkpoint_id 从检查点账本里读出来，首次为 0
BATCH_SIZE = 50000

FETCH_SQL = """
SELECT id, raw
FROM raw_data
WHERE id > :last_checkpoint_id      -- keyset：只取「上次之后」的数据，避免 OFFSET 越翻越慢
ORDER BY id
LIMIT :batch_size                   -- 批大小：越大往返越少，但单批内存占用越高
"""
# 用参数化查询，既防 SQL 注入，也让数据库可以复用执行计划
```

**第二阶段：用 `COPY` 写进 UNLOGGED 暂存表。** 网络往返次数从「每条一次」降到「每 50000 条一次」，即 $O(N)$ 变成 $O(N/50000)$。写进去的是一张 `UNLOGGED` 表——它不写 WAL（预写日志，数据库用来保证崩溃后能恢复的日志），所以写入特别快；代价是数据库崩溃时这张表的数据会丢，但对我们来说无所谓，因为暂存表只是中转站，真正的数据还在 `raw_data` 里，重跑就行：

```python
# 用 PostgreSQL 的 COPY 协议批量灌入暂存表，而不是一条条 INSERT
COPY_SQL = """
COPY staging_expand (rec_id, origin, dest, val)
FROM STDIN WITH (FORMAT csv)
"""

def load_staging(cur, conn, valid_rows):
    # UNLOGGED 表不写 WAL，写入快；作为一次性中转站，崩了重灌即可
    with cur.copy(COPY_SQL) as copy:
        for row in valid_rows:
            copy.write_row(row)      # 流式写入，内存里不需要攒完整张表
    conn.commit()
```

**第三阶段：在数据库里一次性「扇出」到多张目标表。** 数据已经在暂存表里了，接下来用一条带 CTE 的 SQL 把它分发到各目标表。CTE 就是 `WITH ... AS (...)`，可以理解成「先算出一个临时结果集，再拿它做后续操作」。同一条语句里我们顺手把检查点账本也更新掉，保证「写数据」和「记进度」在同一个事务里，要么都成、要么都不成：

```python
FANOUT_SQL = """
WITH src AS (
    SELECT rec_id, origin, dest, val FROM staging_expand
)
INSERT INTO expanded_nodes (rec_id, origin, dest, val)
SELECT rec_id, origin, dest, val FROM src
ON CONFLICT (rec_id) DO NOTHING;          -- 幂等：重复的 rec_id 直接忽略
"""

LEDGER_SQL = """
INSERT INTO expand_checkpoint_ledger
    (chunk_id, min_id, max_id, row_count, status_hash)
VALUES (:chunk_id, :min_id, :max_id, :row_count, :status_hash)
ON CONFLICT (chunk_id) DO UPDATE SET      -- 重跑同一批时覆盖旧记录，而不是报错
    row_count = EXCLUDED.row_count,
    status_hash = EXCLUDED.status_hash;
"""
```

**关于超参数。** 这个比赛里能调的「超参数」其实是几个工程参数，它们之间存在明确的权衡：

- `batch_size`（材料默认 50000）：批越大，网络往返次数越少、`COPY` 越划算；但单批要在内存里攒下的有效行也越多，进程内存压力越大。50000 是一个折中值。
- 检查点粒度：每批结束就落一次账本，粒度越细，崩溃后重做的量越少，但账本写入次数越多。材料选择的粒度是「一批一记」，和 `batch_size` 绑定。
- 校验的严格程度：字段必须齐全、`valid` 必须为真，这两条是硬规则，不参与调优。

我们判断，继续把 `batch_size` 往上堆，收益会迅速变小——因为往返次数本来就按批数下降，50 已经比 20000 小了两个数量级；再往上加，内存风险反而盖过收益。所以我们把精力放在「三阶段是否真的把三个瓶颈都拆掉了」上，而不是死磕单个数字。

### 2.4 模型测试与指标报告

优化到底有没有用，不能靠嘴说，要靠**同一份数据、同一个断言**来验证。我们把基线版和优化版放在同一个脚本里，用相同的数据各跑一遍，然后比对两边的结果「指纹」（行数 + 数值总和），最后打印加速比：

```python
t1 = time.perf_counter()
con2 = sqlite3.connect(":memory:")
con2.execute("CREATE TABLE expanded_nodes (rec_id INT PRIMARY KEY, origin TEXT, dest TEXT, val REAL)")
con2.execute("""CREATE TABLE expand_checkpoints
                (chunk_id INT PRIMARY KEY, max_id INT, rows_written INT, sha256 TEXT)""")

CHUNK = 5000                                   # 基准里的分块大小，对应真实场景的 batch_size 思路
for cid, start in enumerate(range(0, N_RECORDS, CHUNK)):
    batch = raw_records[start:start + CHUNK]
    valid_rows = validate_batch(batch)          # 先在内存里批量校验
    con2.executemany(                           # 再整批写库，减少执行次数
        "INSERT OR IGNORE INTO expanded_nodes VALUES (?, ?, ?, ?)", valid_rows)
    # 为这一批生成一个短指纹：chunk 编号 + 本批最大 id + 写入行数
    chk_hash = hashlib.sha256(
        f"{cid}:{batch[-1][0]}:{len(valid_rows)}".encode()
    ).hexdigest()[:16]
    con2.execute("INSERT INTO expand_checkpoints VALUES (?, ?, ?, ?)",
                 (cid, batch[-1][0], len(valid_rows), chk_hash))
con2.commit()
t_opt = (time.perf_counter() - t1) * 1000.0

cnt2 = con2.execute("SELECT COUNT(*), SUM(val) FROM expanded_nodes").fetchone()
assert cnt1 == cnt2, f"Mismatch: {cnt1} != {cnt2}"   # 等价性不通过就立即失败
print(f"Records processed: {N_RECORDS} | Valid expanded rows: {cnt2[0]}")
print(f"Baseline Row-by-Row Time : {t_baseline:.2f} ms")
print(f"Optimized Bulk-Checkpointed: {t_opt:.2f} ms "
      f"(Speedup: {t_baseline / max(t_opt, 1e-6):.2f}x, 100% bit-exact parity)")
```

这段脚本输出三行关键信息：

- **`Records processed` / `Valid expanded rows`**：告诉我们一共处理了多少条、最终落库多少条。跑完两万条时，有效行数应当等于「总数减去 17 的倍数那些无效行」。
- **`Baseline Row-by-Row Time`**：基线逐行版耗时。
- **`Optimized Bulk-Checkpointed ... Speedup ... 100% bit-exact parity`**：优化版耗时、加速比，以及「结果完全一致」的声明。能打印出这句，说明 `assert cnt1 == cnt2` 顺利通过。

我们没有把材料里没给出的具体毫秒数和加速倍数写死——因为不同的机器、不同的规模跑出来的数字不一样。但我们读懂了这张「成绩单」的读法：**先看断言有没有过，再看加速比有多大**；断言不过，加速比再高也没有意义。

把视角放回真实的 PostgreSQL 部署上，三阶段方案改善的指标也很具体：

- **网络往返次数**：从每条记录一次（$O(N)$）降到每 50000 条一次（$O(N/50000)$），在两万条规模下就是从两万次降到个位数级别。
- **索引维护**：从「插入一行改一次索引」变成「一条语句批量插入」，B-tree 的更新次数大幅减少。
- **失败恢复**：从「整段事务回滚」变成「从账本里最后一个成功的 `max_id` 继续」，已经完成的部分不会白干，也不会重复。

---

## 三、总结

本文中我们完成了 Python–PostgreSQL 数据管道优化挑战的完整流程。我们首先做 EDA，用生成器造了一份形状逼真的两万条基准数据，看清了「记录小、频率高、取值有限」这几个特点，也复现了基线逐行插入的写法。其次我们做了数据清洗，把校验从逐条判断改成批量解析加编译校验，并把去重交给数据库主键约束，用 `INSERT OR IGNORE` / `ON CONFLICT DO NOTHING` 保证幂等。接着我们把逐行展开换成三阶段集合式引擎：keyset 分页按 id 取数，绕开 `OFFSET` 越翻越慢的问题；用 `COPY` 灌进不写 WAL 的 `UNLOGGED` 暂存表，把往返次数从 $O(N)$ 压到 $O(N/50000)$；再用一条 CTE 语句把暂存数据扇出到目标表，并在同一事务里更新检查点账本。然后我们围绕 `batch_size`、检查点粒度做了工程上的权衡，没有盲目加大批次。最后我们用同一份数据、同一个断言跑了两遍，确认行数和数值总和完全一致，再打印基线耗时、优化耗时和加速比。

这个比赛给我们的最大启发是：**性能优化要先找瓶颈，再动手**。如果不知道时间花在「网络往返」「重复解析」还是「逐行维护索引」上，很容易把代码改得很复杂却一点没变快。而当瓶颈被定位清楚之后，解法往往非常朴素——把「一行一行」改成「一批一批」，把「在程序里循环」改成「交给数据库一次做完」，再用一个检查点账本兜住中途失败的风险。这三条思路不只适用于 PostgreSQL，放到任何「远程数据库 + 大批量写入」的管道上都成立。
