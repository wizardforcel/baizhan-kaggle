"""对 docs/*.md 做结构合规检查。

用法：
    python tools/check.py                # 检查所有已生成的 doc
    python tools/check.py --all          # 连同 issue 级问题一起列出
"""
import os
import re
import sys

ROOT = r"D:/docs/baizhan-kaggle"
DOCS = os.path.join(ROOT, "docs")

REQ = [
    "一、比赛简介",
    "二、解题过程",
    "三、总结",
    "1.1 比赛背景",
    "1.2 比赛目标",
    "1.3 评价标准",
    "2.1 数据探索",
    "2.2 数据清洗",
    "2.3 模型训练与超参数调优",
    "2.4 模型测试与指标报告",
    "本文中我们",
]

BAD_PAT = re.compile(r"(上述代码|该代码|这个代码|代码做了|我们我们|。。|，，|### 2\.5|### 1\.4)")


def check(path):
    name = os.path.basename(path)
    raw = open(path, encoding="utf-8").read()
    if "\ufffd" in raw:
        return name, "含乱码字符", None
    missing = [k for k in REQ if k not in raw]
    bad = BAD_PAT.findall(raw)
    n_code = raw.count("```")
    issues = []
    if missing:
        issues.append("缺小节: " + ",".join(missing))
    if bad:
        issues.append("疑似语病/多余小节: " + ",".join(set(bad)))
    if n_code < 12:  # 6 段代码 = 12 个 ```
        issues.append(f"代码块偏少({n_code//2} 段)")
    if len(raw) < 2000:
        issues.append("篇幅过短")
    return name, None if not issues else "; ".join(issues), n_code


def main():
    files = sorted(f for f in os.listdir(DOCS) if f.endswith(".md"))
    bad, ok = 0, 0
    for f in files:
        name, issue, n = check(os.path.join(DOCS, f))
        if issue:
            bad += 1
            print(f"FAIL {name}: {issue}")
        else:
            ok += 1
    print(f"checked={len(files)} ok={ok} fail={bad}")


if __name__ == "__main__":
    main()