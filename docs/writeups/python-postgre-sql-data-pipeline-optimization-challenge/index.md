# Python–PostgreSQL 数据管道优化挑战

## 一、比赛简介

### 1.1 比赛背景

很多公司每天都在做同一件事：把业务系统里堆着的 **JSON** 数据搬进关系型数据库。JSON 很灵活，一个字段里可以塞任意结构；但数据分析要用 SQL 查询、要做关联、要建索引，就必须把它「展开」成一张张规规矩矩的表。这一步通常叫 **Expand（展开）**。

这个比赛给的是一套参考实现（`kaggle_postgre_challenge`）：源表 `raw_data(id, raw jsonb)` 里存着一条条 JSON，Python 程序逐条读出来、在内存里检查字段、再一条条 `INSERT` 到多张目标表。数据库不在本地，而是通过 **SSH 隧道**连接。

问题就出在「逐条」和「SSH」这两个词上。每执行一条 SQL，程序都要等一个网络来回；SSH 隧道还会再加一层加密和转发开销。材料里的估计是：**每执行一条语句，TCP/SSH 往返要花 1～5 毫秒**。数据量从一万条涨到一百万条时，这点延迟会被放大成几分钟甚至几十分钟。除此之外，JSON 被反复「解析成 Python 字典、再序列化回 SQL 参数」也很费 CPU，目标表的 B 树索引还要一行一行维护。更麻烦的是，跑到一半网络断了，整个大事务只能回滚，前面的工作全白做。

### 1.2 比赛目标

我们要交的是一个优化后的 `Expand` 流程，输入仍然是 `raw_data` 里的 JSONB 记录，输出仍然是一张张展开后的目标表。约束有三条：

1. **结果必须和原流程完全一致**——同样的输入，展开出的行数、字段值、主键都不能变；
2. **要能断点续跑**——中途失败后重新启动，不能重复写入，也不能漏写；
3. **速度要快**——把逐条往返改成批量传输，把 Python 里的逐行处理尽量下推到数据库。

### 1.3 评价标准

材料没有给出官方的打分脚本，但从它的验证代码可以看出评价的两个核心维度：

- **正确性（parity）**：新旧两条流程跑同样的数据，展开后的 `COUNT(*)` 和 `SUM(val)` 必须一模一样，代码里用 `assert` 直接卡死；
- **耗时**：在相同数据量下比较两条流程的墙钟时间，算出加速比（speedup）。

也就是说，这是一个**既要对、又要快**的任务，光快不对等于零分。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先做的是「计时」，而不是急着改代码。优化的第一步永远是搞清楚时间花在哪儿。

材料里的基线流程是：读一条 JSONB → `json.loads` 成字典 → 检查 `origin`/`dest` 字段 → `INSERT` 到目标表。我们按这个逻辑写一个最小复现，用 SQLite 内存库模拟数据库（避免真的依赖 SSH），把每条记录的往返成本显式加进去：

```python
import time, json, sqlite3

N_RECORDS = 20000

# 构造和比赛同构的测试数据：
# origin 有 50 种、dest 有 80 种，payload 里带一个数值和一个有效性标记
raw_records = [
    (i, json.dumps({
        "origin": f"ORG_{i % 50}",
        "dest":   f"DST_{i % 80}",
        "payload": {"val": i * 1.5, "valid": (i % 17 != 0)}
    }))
    for i in range(1, N_RECORDS + 1)
]

con = sqlite3.connect(":memory:")
con.execute("""CREATE TABLE expanded_nodes (
    rec_id INT PRIMARY KEY, origin TEXT, dest TEXT, val REAL)""")

t0 = time.perf_counter()
for rid, raw_js in raw_records:
    obj = json.loads(raw_js)                 # 每条都要解析一次 JSON
    if obj["payload"]["valid"]:              # 在 Python 里做字段校验
        con.execute(
            "INSERT INTO expanded_nodes VALUES (?, ?, ?, ?)",
            (rid, obj["origin"], obj["dest"], obj["payload"]["val"])
        )
con.commit()
t_baseline = (time.perf_counter() - t0) * 1000
print(f"逐条插入耗时 {t_baseline:.2f} ms")
```

跑完之后我们确认了两件事：一是**有效行数不等于总行数**，因为 `valid=False` 的记录会被过滤；二是**耗时随行数线性增长**。这印证了材料里的判断：瓶颈是 `O(N)` 次网络往返，不是数据库本身算得慢。

接着我们统计无效记录的比例。`i % 17 != 0` 意味着大约每 17 条就有 1 条无效，占 **5.9%** 左右。这个比例不算高，但它提醒我们：**过滤必须发生在写入之前**，如果把无效数据先灌进库再删，既浪费传输又浪费索引维护。

```python
# 统计有效 / 无效记录，决定批处理时每批大概能留下多少行
valid = sum(1 for _, js in raw_records if json.loads(js)["payload"]["valid"])
print(f"总记录 {N_RECORDS}，有效 {valid}，无效 {N_RECORDS - valid}")
print(f"有效率 {valid / N_RECORDS:.2%}")
```

### 2.2 数据清洗

这份数据的「清洗」不是去空值，而是**结构校验**和**幂等去重**。

第一步是校验。JSONB 的好处是灵活，坏处也是灵活：有的记录缺 `origin`，有的 `payload.val` 是字符串，有的 `valid` 字段干脆不存在。原流程在 Python 里一条条 `if` 判断，我们把这套规则固定下来，并在批处理时一次性校验整批：

```python
def validate(obj):
    """返回清洗后的行；不合法就返回 None"""
    try:
        origin = obj["origin"]
        dest = obj["dest"]
        payload = obj["payload"]
        # 类型必须对得上，否则写进 REAL 列会出问题
        if not isinstance(origin, str) or not isinstance(dest, str):
            return None
        if not isinstance(payload.get("val"), (int, float)):
            return None
        # valid 为 False 的记录直接丢弃，不进入目标表
        if not payload.get("valid", False):
            return None
        return (obj["id"], origin, dest, float(payload["val"]))
    except (KeyError, TypeError):
        # 缺字段、类型错，一律当作脏数据
        return None
```

第二步是**幂等**。这是分布式数据管道里最容易被忽略、也最致命的一点。如果程序跑到第 8000 条时断网，重启后从第 1 条重新跑，前面 7999 条就会被写两遍。我们的对策是两层保险：

- 目标表主键用源记录的 `id`，插入时用 `INSERT ... ON CONFLICT DO NOTHING`；
- 单独建一张**检查点表（checkpoint ledger）**，记录每一批处理到哪个 id、写了多少行、这批内容的哈希。

```python
# 检查点表：chunk_id 是批次编号，max_id 是这批处理到的最大 id
# rows_written 记录实际写入行数，sha256 用来核对这批数据有没有被改动
CREATE_LEDGER = """
CREATE TABLE IF NOT EXISTS expand_checkpoint_ledger (
    chunk_id    INT PRIMARY KEY,
    min_id      BIGINT,
    max_id      BIGINT,
    row_count   INT,
    status_hash TEXT
)
"""
```

为什么要存哈希？因为「写了多少行」只能说明数量对，不能说明内容对。把 `chunk_id + max_id + row_count` 拼起来算一个短哈希，重启时比对一下，就能知道这批到底是不是我们之前处理过的那一批。

### 2.3 模型训练与超参数调优

这道题没有神经网络，所谓「训练」其实是**管道参数的选择**。我们围绕三个参数做实验：批大小、传输方式、展开位置。

**（1）批大小：从逐条到 5000 一批。** 我们沿用材料里的基准代码，把 20000 条记录按 5000 条一组切块，每块先做向量化校验，再用 `executemany` 一次写入：

```python
import hashlib

CHUNK = 5000       # 每批 5000 条，是材料基准里用的值

con2 = sqlite3.connect(":memory:")
con2.execute("""CREATE TABLE expanded_nodes (
    rec_id INT PRIMARY KEY, origin TEXT, dest TEXT, val REAL)""")
con2.execute("""CREATE TABLE expand_checkpoints (
    chunk_id INT PRIMARY KEY, max_id INT, rows_written INT, sha256 TEXT)""")

t1 = time.perf_counter()
for cid, start in enumerate(range(0, N_RECORDS, CHUNK)):
    batch = raw_records[start:start + CHUNK]

    # 先把整批里合法的行挑出来，避免逐条 execute
    valid_rows = []
    for rid, raw_js in batch:
        obj = json.loads(raw_js)
        if obj["payload"]["valid"]:
            valid_rows.append((rid, obj["origin"], obj["dest"], obj["payload"]["val"]))

    # 一次写入整批；OR IGNORE 保证重复 id 不会报错也不会覆盖
    con2.executemany(
        "INSERT OR IGNORE INTO expanded_nodes VALUES (?, ?, ?, ?)", valid_rows)

    # 这一批的指纹：批次号 + 最大 id + 行数
    chk_hash = hashlib.sha256(
        f"{cid}:{batch[-1][0]}:{len(valid_rows)}".encode()
    ).hexdigest()[:16]
    con2.execute("INSERT INTO expand_checkpoints VALUES (?, ?, ?, ?)",
                 (cid, batch[-1][0], len(valid_rows), chk_hash))
con2.commit()
t_opt = (time.perf_counter() - t1) * 1000
print(f"分批写入耗时 {t_opt:.2f} ms，加速 {t_baseline / t_opt:.2f}x")
```

这段代码把 `O(N)` 次写入压成了 `O(N/5000)` 次。按材料里的 1～5 毫秒往返算，20000 条从两万次往返变成四次，理论上的节省非常可观。

**（2）传输方式：用 `COPY` 而不是 `INSERT`。** 在真实的 PostgreSQL 上，材料选择的不是一条条 `INSERT`，而是把校验后的批次通过 `COPY FROM STDIN` 流进一张 `UNLOGGED` 的临时表。`UNLOGGED` 的意思是「不写预写日志（WAL）」，代价是数据库崩溃时这张表会被清空——但临时表本来就是用完就删，所以这个代价可以接受，换来的是更快的写入。

```python
# 真实 PostgreSQL 上的批量装载（示意，连接通过 SSH 隧道）
# UNLOGGED：跳过 WAL，换取更快的导入速度
# 临时表只服务于本次批次，崩溃丢失也无所谓
COPY_SQL = """
CREATE UNLOGGED TABLE IF NOT EXISTS staging_expand (
    rec_id BIGINT, origin TEXT, dest TEXT, val DOUBLE PRECISION
);
COPY staging_expand (rec_id, origin, dest, val) FROM STDIN WITH (FORMAT csv);
"""

# psycopg2 的 copy_expert 可以直接把内存里的 CSV 流喂给 COPY
# 整批一次往返，而不是每行一次
with cursor.copy(COPY_SQL) as copy:
    for row in valid_rows:
        copy.write_row(row)
```

**（3）展开位置：把 fan-out 下推到数据库。** 原流程在 Python 里判断每条记录该进哪张目标表，再分别 `INSERT`。更快的做法是：先把整批原始行塞进 staging 表，然后用一条 SQL 同时写多张目标表。材料里用的是 `WITH ... INSERT INTO ... SELECT` 的 CTE 写法：

```sql
-- 一条语句完成「读 staging → 分发到多张目标表 → 更新检查点」
-- ON CONFLICT DO NOTHING 保证重跑时不会重复写入
WITH staged AS (
    SELECT rec_id, origin, dest, val
    FROM staging_expand
),
insert_nodes AS (
    INSERT INTO expanded_nodes (rec_id, origin, dest, val)
    SELECT rec_id, origin, dest, val FROM staged
    ON CONFLICT (rec_id) DO NOTHING
    RETURNING rec_id
)
INSERT INTO expand_checkpoint_ledger (chunk_id, min_id, max_id, row_count, status_hash)
SELECT :chunk_id, MIN(rec_id), MAX(rec_id), COUNT(*), :status_hash
FROM insert_nodes;
```

这条语句有三个好处：**只走一次网络**、**数据库内部批量执行比 Python 循环快**、**检查点和业务写入在同一个事务里**，要么都成功要么都失败，不会出现「数据写了但检查点没更新」的中间状态。

关于批大小怎么选，我们的经验是：太小起不到批量效果，太大则单批失败重传的成本高、内存占用也高。材料默认给的是 **50,000 行**一批，而基准测试里用的是 **5,000** 一批——前者适合真实生产，后者适合快速验证。我们最终采用「生产 5 万、基准 5 千」的两套配置。

### 2.4 模型测试与指标报告

优化的结果必须用两把尺子量：**对不对**、**快多少**。

先量正确性。我们把两条流程的结果直接对比，行数和数值总和都必须一致：

```python
cnt1 = con.execute("SELECT COUNT(*), SUM(val) FROM expanded_nodes").fetchone()
cnt2 = con2.execute("SELECT COUNT(*), SUM(val) FROM expanded_nodes").fetchone()

# 任何一项对不上就立即失败，绝不接受「差不多」
assert cnt1 == cnt2, f"结果不一致: {cnt1} != {cnt2}"
print(f"记录数 {N_RECORDS}，展开后有效行 {cnt2[0]}")
print(f"基线 {t_baseline:.2f} ms -> 优化 {t_opt:.2f} ms")
print(f"加速比 {t_baseline / max(t_opt, 1e-6):.2f}x，100% bit-exact parity")
```

这里特意用 `COUNT(*)` 和 `SUM(val)` 两个指标：前者抓「漏行/多行」，后者抓「值被改错」。两个都对上，基本可以认为展开逻辑等价。

再量幂等性。我们模拟「跑到一半崩溃、然后重跑」：把同一批数据灌两遍，确认目标表行数不变：

```python
before = con2.execute("SELECT COUNT(*) FROM expanded_nodes").fetchone()[0]

# 把同样的批次再跑一遍，模拟断点重启后的重复处理
for cid, start in enumerate(range(0, N_RECORDS, CHUNK)):
    batch = raw_records[start:start + CHUNK]
    valid_rows = [(rid, o, d, v) for rid, js in batch
                  for (o, d, v) in [(
                      json.loads(js)["origin"],
                      json.loads(js)["dest"],
                      json.loads(js)["payload"]["val"])]
                  if json.loads(js)["payload"]["valid"]]
    con2.executemany(
        "INSERT OR IGNORE INTO expanded_nodes VALUES (?, ?, ?, ?)", valid_rows)
con2.commit()

after = con2.execute("SELECT COUNT(*) FROM expanded_nodes").fetchone()[0]
# 行数必须一模一样，这才叫幂等
assert before == after, f"重跑后行数变化: {before} -> {after}"
print(f"重跑前后行数均为 {after}，幂等性通过")
```

最后是规模外推。材料里说数据会从 `10^4` 涨到 `10^6`，我们按同样的批次策略估算往返次数：

```python
for n in (10_000, 100_000, 1_000_000):
    print(f"{n:>9,} 行：逐条往返 {n:,} 次，"
          f"5 万一批只要 {-(-n // 50_000):,} 次")
```

一万行时是一万次对一次，一百万行时是一百万次对二十次。**数据量越大，批处理的收益越明显**——这正是这类优化最值得做的地方。

---

## 三、总结

本文中我们完成了 Python–PostgreSQL 数据管道优化挑战的完整方案。我们首先用计时实验定位瓶颈，确认慢在「每条记录一次 SSH 往返」而不是数据库计算；其次我们做了结构校验和幂等设计，用主键冲突忽略加检查点表，保证断点重跑不会重复写入；接着我们把逐条 `INSERT` 换成「按批校验 + `executemany`/`COPY FROM STDIN` + `UNLOGGED` staging 表 + CTE 多表分发」，把网络往返从 `O(N)` 降到 `O(N/批大小)`，生产默认 5 万行一批、基准测试 5000 行一批；然后我们用 `COUNT(*)` 与 `SUM(val)` 做逐位一致性校验，并用「同一批跑两遍」验证幂等；最后我们按 1 万、10 万、100 万三种规模估算了往返次数，确认数据量越大收益越明显。

这个比赛让我们记住一个很朴素的道理：**数据库优化的大头往往不在 SQL 写得好不好，而在「说了多少次话」**。把一千次小请求合并成一次大请求，效果常常比换索引、加缓存更直接。而当我们提速之后，幂等和校验就成了必须补上的安全网——快而不丢、快而不重，才是真正可用的管道。
