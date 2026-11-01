import json, sys
from anarci import run_anarci
seqs={}
for l in open(sys.argv[1]):
    e=json.loads(l)
    for p in e['polymer_entities'] or []:
        if p['entity_poly']['type']!='polypeptide(L)': continue
        s=(p['entity_poly']['pdbx_seq_one_letter_code_can'] or '').replace('\n','')
        if 60<=len(s)<=1000: seqs.setdefault(s,len(seqs))
print('unique seqs',len(seqs),file=sys.stderr)
items=[(str(i),s) for s,i in seqs.items()]
out={}
B=2000
for k in range(0,len(items),B):
    chunk=items[k:k+B]
    _,_,details,_=run_anarci(chunk,scheme='imgt',ncpu=8,assign_germline=False)
    for (name,s),det in zip(chunk,details):
        if det: out[s]=[{'chain_type':d['chain_type'],'start':d['query_start'],'end':d['query_end'],'evalue':float(d['evalue'])} for d in det]
    print(k,len(out),file=sys.stderr,flush=True)
json.dump(out,open(sys.argv[2],'w'))
