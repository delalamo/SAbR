"""Fetch extra full texts and print sentences matching keyword regexes. usage: s03 key regex [maxn]"""
import os, re, sys, urllib.request, html
C='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/literature_dean/epmc'
extra={'croll_nsmb':'PMC8340564','scalop':'PMC6513161','teplyakov_germline':'PMC4968113','emdb_val2022':'PMC9063848','guloglu2023':'PMC10410447'}
for k,pm in ([] if len(sys.argv)>2 else extra.items()):
    fx=f'{C}/{k}.xml'
    if not os.path.exists(fx):
        try:
            with urllib.request.urlopen(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pm}/fullTextXML',timeout=60) as r: open(fx,'wb').write(r.read())
        except Exception as e: print(k,e)
def text(k):
    x=open(f'{C}/{k}.xml',encoding='utf8',errors='ignore').read()
    x=re.sub(r'<ref-list>.*?</ref-list>','',x,flags=re.S)
    x=re.sub(r'<[^>]+>',' ',x); x=html.unescape(x); return re.sub(r'\s+',' ',x)
if len(sys.argv)>2:
    k,rx=sys.argv[1],sys.argv[2]; n=int(sys.argv[3]) if len(sys.argv)>3 else 30
    t=text(k); sents=re.split(r'(?<=[.!?])\s+(?=[A-Z(])',t)
    hits=[s for s in sents if re.search(rx,s,flags=re.I)]
    for s in hits[:n]: print('-',s[:600])
    print(f'[{len(hits)} hits / {len(sents)} sentences]')
