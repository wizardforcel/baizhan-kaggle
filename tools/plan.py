"""分配任务：从已抽取但未写、未分配的比赛中取一批。

用法：
    python tools/plan.py 24        # 取 24 个 slug，打印并记入 work/assigned.txt
    python tools/plan.py --reset   # 清空分配记录
"""
import os
import sys

ROOT = r"D:/docs/baizhan-kaggle"
TEXT = os.path.join(ROOT, "work/text")
DOCS = os.path.join(ROOT, "docs")
ASSIGNED = os.path.join(ROOT, "work/assigned.txt")


def load_assigned():
    if not os.path.exists(ASSIGNED):
        return set()
    with open(ASSIGNED, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def main():
    if "--reset" in sys.argv:
        open(ASSIGNED, "w").close()
        print("assigned.txt reset")
        return
    n = int(sys.argv[1])
    done = {f[:-3] for f in os.listdir(DOCS) if f.endswith(".md")}
    assigned = load_assigned()
    avail = sorted(
        s for s in os.listdir(TEXT)
        if os.path.isdir(os.path.join(TEXT, s)) and s not in done and s not in assigned
    )
    batch = avail[:n]
    with open(ASSIGNED, "a", encoding="utf-8") as f:
        for s in batch:
            f.write(s + "\n")
    print(" ".join(batch))
    print(f"# picked {len(batch)}, remaining unassigned: {len(avail) - len(batch)}", file=sys.stderr)


if __name__ == "__main__":
    main()
