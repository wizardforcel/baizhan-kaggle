import re, os, sys
slugs = [
    'flag-os-open-computing-global-challenge-track-1',
    'flavours-of-physics',
    'flavours-of-physics-kernels-only',
    'forest-cover-type-kernels-only',
    'forest-cover-type-prediction',
]
if os.path.exists('docs/freesound-audio-tagging.md'):
    slugs.append('freesound-audio-tagging')
for f in slugs:
    p = os.path.join('docs', f + '.md')
    if os.path.exists(p):
        s = open(p, encoding='utf-8').read()
        cnt = len(re.findall(r'[一-鿿]', s))
        blocks = s.count('```python')
        print(f, cnt, blocks)