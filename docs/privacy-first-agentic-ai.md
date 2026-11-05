# 隐私优先的智能体 AI

## 一、比赛简介

### 1.1 比赛背景

这个题目来自「第六届数据科学与人工智能世界锦标赛」中的 Privacy-First Agentic AI 赛道。它和常见的「给一张表、预测一个数」的比赛不同：任务是在**本地**搭一套能检索文档、能回答问题、但绝不把敏感信息泄漏出去的智能体检索增强生成（RAG）系统。

现实中的医院病历、银行流水、法律卷宗都不能直接上传到第三方云端大模型接口，所以这类系统的核心矛盾是：既要让模型「读到」足够多的资料，又不能让个人身份信息（PII）离开本地。PII 指的是能直接识别个人的数据，比如身份证号、病历号、邮箱、电话、姓名。赛道希望参赛者给出可复现、可审计的隐私防线，而不是只喊一句「我们不上传数据」。

由于这是 Notebook 型比赛，评价不只看模型效果，还看隐私设计是否站得住脚：敏感字段有没有在入库前被处理、检索时有没有按租户和权限隔离、输出前有没有做泄漏检查、审计日志是否可验证。

### 1.2 比赛目标

输入是一批文档（可以是真实的业务文档，也可以用合成的虚构记录做基准测试），输出是两样东西：一是能根据自然语言问题检索出相关文档并生成回答的智能体流程；二是一套可验证的隐私保护与审计证据。

在基准版本里，作者用**完全虚构**的记录构造了 200 条文档、覆盖 8 个医学主题，每个主题 25 条；查询分成 32 条开发查询和 32 条互不重复的留出查询。所有姓名、证件号都是假的，避免把真实病人数据带进公开笔记本。

### 1.3 评价标准

因为没有传统意义上的排行榜标签，我们从两个层面自评：

1. **检索效果**：对每条查询，看检索回来的前若干条文档里有没有命中正确主题，用命中率（recall@k 一类指标）衡量，并给出 Wilson 置信区间，避免小样本下把偶然当成结论。
2. **隐私合规**：所有入库文本必须通过 PII 检测（零命中才算通过）；检索必须带租户和权限过滤；输出前再做一次泄漏扫描，并生成 SHA-256 合规回执。

作者特意强调，检索主题命中率**不等于**临床准确率，这个系统也不是临床助手。

## 二、解题过程

### 2.1 数据探索（EDA）

我们首先检查语料的结构：每条记录包含文档 ID、主题、科室、是否是合成数据、原文和脱敏后文本。构造语料时，我们故意把姓名、证件号、电话、邮箱写进正文，好让后面的脱敏和审计有的放矢：

```python
def make_corpus():
    records = []
    scenarios = ['archive routing', 'document indexing', 'terminology lookup',
                 'topic classification', 'retrieval testing']
    for j in range(25):                       # 每个主题 25 条
        for code, title, dept, desc, _ in TOPICS:
            k = len(records)
            identity = f'Example Person {k:03d}'   # 虚构姓名
            raw = (f'FICTIONAL BENCHMARK RECORD. Patient Name: {identity} | '
                   f'SSN: 999-{k%100:02d}-{k:04d} | Phone: +1-202-555-{100+k%100:04d} | '
                   f'Email: case{k:03d}@example.org\n'
                   f'Topic: {title}. This synthetic document concerns {desc}.')
            clean, _ = Sanitizer().redact(raw)   # 先脱敏再入库
            records.append({'doc_id': f'SYN_{k:04d}', 'topic': code,
                            'topic_name': title, 'department': dept,
                            'is_synthetic': True, 'raw_text': raw,
                            'sanitized_text': clean})
    return records
```

我们统计了每条记录里各类 PII 的出现次数，并检查脱敏前后长度变化，确认四类模式（姓名、SSN、邮箱、电话）都能被命中。同时构造了包含「忽略之前所有指令」「把数据库导出发到 webhook」这类提示注入（prompt injection）的对抗样本，用于后面的安全测试。

### 2.2 数据清洗

清洗就是隐私防线的第一道关：**在文档被向量化之前**，用确定性的正则把敏感字段替换成可逆的保险库令牌。所谓可逆，是指令牌和原文的对应关系只保存在本地加密的 SQLite 小仓库里，模型和向量库永远见不到原文：

```python
PII_PATTERNS = {
    "SSN":   re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "MRN":   re.compile(r"\bMRN-\d{6,10}\b"),
}

def redact_pii(text, salt="yankee_local_enclave_2026"):
    vault = {}                                   # 只在本机保存的还原表
    out = text
    for kind, pat in PII_PATTERNS.items():
        for m in list(pat.finditer(out)):
            raw = m.group(0)
            # 加盐哈希后取前 8 位，保证同一实体得到稳定令牌，又不暴露原文
            tok_id = hashlib.sha256(f"{salt}:{kind}:{raw}".encode()).hexdigest()[:8]
            token = f"<REDACTED_{kind}_{tok_id}>"
            vault[token] = raw
            out = out.replace(raw, token)
    return out, vault

def verify_zero_pii(text):
    # 只要还有任意一类模式能匹配上，就判定脱敏失败
    return all(pat.search(text) is None for pat in PII_PATTERNS.values())
```

这里的「加盐」是指在哈希前混入一段只有本机知道的字符串，防止攻击者用彩虹表反推原文；同一实体每次得到同一个令牌，所以文档之间的关联性还在，但身份本身不可读。入库前我们逐条断言 `verify_zero_pii` 通过，任何一条失败就直接中止。

除了 PII，我们还清洗了提示注入模式，比如检测「ignore previous instructions」「system override」等短语并标记为可疑，这类文本不进入检索结果。

### 2.3 模型训练与超参数调优

这一步没有传统意义的「训练」，我们做的是**向量化与检索配置**。文档用本地句向量模型编码后写入 ChromaDB（一个本地向量数据库），并带上不可篡改的元数据，比如租户 ID、密级、是否已脱敏。检索时强制加上权限过滤：

```python
# 每个文档写入时携带租户和密级，检索时由 Agent 强制过滤
where_filter = {"clearance_level": {"$lte": user_role}}   # 用户只能看自己不高于的密级
results = collection.query(query_texts=[question], n_results=10,
                           where=where_filter)
```

我们用三组固定的 HNSW（一种近似最近邻索引）参数做对照，分别是 `cosine_m16`、`cosine_m32`、`cosine_m64`，每组规定了邻居数、建图参数 `ef_construction` 和搜索参数 `ef_search`：

```python
CONFIGS = [
    {'name':'cosine_m16', 'space':'cosine', 'max_neighbors':16,
     'ef_construction':100, 'ef_search':20},
    {'name':'cosine_m32', 'space':'cosine', 'max_neighbors':32,
     'ef_construction':200, 'ef_search':50},
    {'name':'cosine_m64', 'space':'cosine', 'max_neighbors':64,
     'ef_construction':400, 'ef_search':100},
]
```

模型选择只看**开发查询**的表现，选定配置之后才在 32 条留出查询上评估，避免「拿测试集调参」。为了知道向量检索到底有没有比老办法强，我们还跑了 TF-IDF 对比，以及一个精确向量最近邻的参照结果。

在纯标准库版本里，作者甚至写了一个不依赖任何第三方包的极简向量库：分词后用词频构造稀疏向量并做 L2 归一化，点积就是余弦相似度。这让我们理解向量检索的本质，而不是把它当成黑盒。

### 2.4 模型测试与指标报告

评估分两半。检索侧，对每条查询取回前 k 条文档，看是否命中了正确主题，再用 Wilson 公式给出置信区间：

```python
def wilson(w, n):
    # w 是命中数，n 是总查询数，返回 95% 置信区间
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = w / n
    d = 1 + z*z/n
    a = (p + z*z/(2*n)) / d
    b = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [a-b, a+b]
```

隐私侧，我们对全部入库文本做零 PII 断言，并记录脱敏实体数量；对输出再做一次同样的扫描，任何命中都判失败。每次运行还会把配置、随机种子、库版本、运行时长写进 `run_manifest.json`，并生成 SHA-256 回执，方便事后审计。

需要诚实说明边界：这套 Python 层的 socket 防护只覆盖解释器内的网络调用，原生扩展和子进程的流量不在覆盖范围内，而且这个版本并没有真正测量「外发了多少字节」。作者在第三版里专门纠正了早期「100% 零外发」的夸大说法——隐私声明必须和实际能做到的事情一致，否则审计就失去意义。

## 三、总结

本文中我们完成了一个隐私优先的智能体 RAG 原型。我们首先构造了 200 条虚构医疗记录和 8 个主题的查询，明确区分开发集与留出集；其次在入库前用确定性正则加盐哈希把姓名、证件号、邮箱、电话替换成本地保险库令牌，并断言零 PII 残留；接着把文档写入带租户与密级元数据的 ChromaDB，检索时强制权限过滤，并用三组固定 HNSW 配置和 TF-IDF 参照做开发集选型；然后对留出查询计算主题命中率与 Wilson 置信区间，对输出做二次泄漏扫描并生成审计回执；最后如实说明防护的覆盖边界。

这个比赛让我们认识到，隐私保护不是一句口号，而是一串可以被验证的动作：入库前脱敏、检索时隔离、输出前复核、全程留痕。只有每一步都能拿出证据，系统才谈得上「隐私优先」。
