import pathlib, re
p=pathlib.Path(r'D:\docs\baizhan-kaggle\docs\student-pathways-and-observation-time.md')
txt=p.read_text(encoding='utf-8')
chinese = len(re.findall(r'[\u4e00-\u9fff]', txt))
print('chinese chars', chinese)
print('total chars', len(txt))
