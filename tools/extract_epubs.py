"""把 ../kaggle 下的 EPUB 抽取成纯文本，按比赛归并。

用法：
    python tools/extract_epubs.py

输入：D:/docs/kaggle/*.epub
输出：
    work/text/<slug>/p001_01.txt   每个 notebook 一页一文
    work/text/<slug>/_index.txt    该比赛的页面清单
    work/manifest.json             全局索引（slug -> 源文件、页数、字符数）

说明：
  - "Kaggle Kernel - xxx - PageN.epub" 视为比赛 xxx 的第 N 卷，按 N 排序后合并；
  - 其余 "xxx.epub" 视为独立比赛；
  - 同一比赛内部内容完全相同的页只保留一份（很多分卷里是重复的空页）。
"""
import json
import os
import re
import sys
import warnings
import zipfile
from hashlib import md5

from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

SRC = r"D:/docs/kaggle"
OUT = r"D:/docs/baizhan-kaggle/work/text"

MULTI = re.compile(r"^Kaggle Kernel - (.+) - Page(\d+)\.epub$")

BOILER_LINE = re.compile(r"^(¶|In \[\d*\]:|Out\[\d*\]:|In \[ \]:)\s*$")


def html_to_text(raw: str) -> str:
    """把一页 XHTML 转成纯文本，保留代码块原样。"""
    soup = BeautifulSoup(raw, "xml")
    for tag in soup.find_all(["head", "script", "style"]):
        tag.decompose()

    # 代码块里每个 token 都被 <span> 包着，用空分隔符拼接才能还原代码
    for pre in soup.find_all("pre"):
        for a in pre.find_all("a"):
            a.unwrap()
        text = pre.get_text("")
        new = soup.new_tag("pre")
        new.string = text
        pre.replace_with(new)

    text = soup.get_text("\n")
    text = text.replace("\xa0", " ").replace("¶", "")
    lines = [ln.rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if not BOILER_LINE.match(ln.strip())]
    # 折叠连续空行
    out, blank = [], 0
    for ln in lines:
        if ln.strip():
            blank = 0
            out.append(ln)
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip()


def epub_pages(path: str):
    """按文件名顺序返回 (name, text)。"""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if re.search(r"/Text/\d+\.html$", n)]
        names.sort(key=lambda n: int(re.search(r"(\d+)\.html$", n).group(1)))
        for n in names:
            raw = z.read(n).decode("utf-8", "ignore")
            yield n, html_to_text(raw)


def main():
    groups = {}
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith(".epub"):
            continue
        m = MULTI.match(fn)
        if m:
            slug, page = m.group(1), int(m.group(2))
        else:
            slug, page = fn[:-5], 0
        groups.setdefault(slug, []).append((page, fn))

    manifest = {}
    for slug, items in groups.items():
        items.sort()
        d = os.path.join(OUT, slug)
        os.makedirs(d, exist_ok=True)
        seen = set()
        index, total = [], 0
        for page, fn in items:
            path = os.path.join(SRC, fn)
            try:
                pages = list(epub_pages(path))
            except Exception as e:  # 损坏的 epub 跳过但记录
                index.append(f"PAGE {page} {fn} ERROR {e}")
                continue
            idx = 0
            for name, text in pages:
                if len(text) < 200:          # 封面/目录页
                    continue
                h = md5(text.encode("utf-8")).hexdigest()
                if h in seen:                # 同比赛内的重复页
                    continue
                seen.add(h)
                idx += 1
                title = text.splitlines()[0][:120]
                fn_out = f"p{page:03d}_{idx:02d}.txt"
                with open(os.path.join(d, fn_out), "w", encoding="utf-8") as f:
                    f.write(text)
                index.append(f"PAGE {page} | {fn_out} | {len(text):>8} chars | {title}")
                total += len(text)
        with open(os.path.join(d, "_index.txt"), "w", encoding="utf-8") as f:
            f.write(f"SLUG: {slug}\nPAGES: {len(index)}\nCHARS: {total}\n\n" + "\n".join(index))
        manifest[slug] = {
            "sources": [fn for _, fn in items],
            "pages": len(index),
            "chars": total,
        }
        print(f"{slug}: {len(items)} epub -> {len(index)} pages, {total} chars", flush=True)

    os.makedirs(os.path.dirname(r"D:/docs/baizhan-kaggle/work/manifest.json"), exist_ok=True)
    with open(r"D:/docs/baizhan-kaggle/work/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"TOTAL competitions: {len(manifest)}")


if __name__ == "__main__":
    sys.exit(main())
