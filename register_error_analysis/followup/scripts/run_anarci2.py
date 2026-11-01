import json, sys
from anarci import run_anarci
def seqs_from(f, key='pdbx_seq_one_letter_code_can'):
    out=set()
    for l in open(f):
        e=json.loads(l)
        for p in e['polymer_entities'] or []:
            if p['entity_poly']['type']!='polypeptide(L)': continue
            s=(p['entity_poly'][key] or '').replace('\n','')
            if 60<=len(s)<=1000: out.add(s)
    return out
done=seqs_from('/tmp/supp/rcsb/seqs.jsonl')
res=json.load(open('/tmp/supp/rcsb/anarci.json'))
new=sorted(seqs_from('/tmp/supp/meta2/em_meta.jsonl')-done)
print('new seqs',len(new),file=sys.stderr,flush=True)
items=[(str(i),s) for i,s in enumerate(new)]
for k in range(0,len(items),2000):
    chunk=items[k:k+2000]
    _,_,details,_=run_anarci(chunk,scheme='imgt',ncpu=4,assign_germline=False)
    for (n,s),det in zip(chunk,details):
        if det: res[s]=[{'chain_type':d['chain_type'],'start':d['query_start'],'end':d['query_end'],'evalue':float(d['evalue'])} for d in det]
    print(k,len(res),file=sys.stderr,flush=True)
json.dump(res,open('/tmp/supp/meta2/anarci_all.json','w'))
print('done',file=sys.stderr)
