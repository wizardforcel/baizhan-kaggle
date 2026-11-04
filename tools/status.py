"""列出进度，并可生成一批待写的 slug。

用法：
    python tools/status.py                 # 统计已完成/剩余
    python tools/status.py --next 8        # 打印接下来 8 个待写 slug（空格分隔）
    python tools/status.py --next 8 --offset 16   # 跳过已分配的 16 个
"""
import argparse
import json
import os

ROOT = r"D:/docs/baizhan-kaggle"
MANIFEST = os.path.join(ROOT, "work/manifest.json")
DOCS = os.path.join(ROOT, "docs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--next", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    done = {f[:-3] for f in os.listdir(DOCS) if f.endswith(".md")}
    todo = [s for s in sorted(manifest) if s not in done]
    print(f"total={len(manifest)} done={len(manifest)-len(todo)} todo={len(todo)}")
    if args.next:
        batch = todo[args.offset:args.offset + args.next]
        print(" ".join(batch))


if __name__ == "__main__":
    main()
