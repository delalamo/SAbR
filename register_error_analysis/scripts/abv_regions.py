import json, urllib.request, pandas as pd, re, collections
from anarci import run_anarci
d=pd.read_pickle('/tmp/analysis/ecod_grouped.pkl')
t=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx').drop(columns=['Unnamed: 0'],errors='ignore')
t[['pdb','chain']]=t.PDB_CHAIN_ID.str.split(':',expand=True); t['pdb']=t.pdb.str.lower()
rr=t.RESIDUE_RANGE.str.extract(r'(-?\d+)\s*-\s*(-?\d+)').astype(int); t['rs'],t['re']=rr[0],rr[1]
v=d[(d.group=='Antibody V')&(d.flag==1)]
keys=set(zip(v.pdb,v.chain))
te=t[[k in keys for k in zip(t.pdb,t.chain)]].copy()
pdbs=sorted({k[0] for k in keys})
Q='''query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id polymer_entities{entity_poly{pdbx_seq_one_letter_code_can}
polymer_entity_instances{rcsb_polymer_entity_instance_container_identifiers{auth_asym_id auth_to_entity_poly_seq_mapping}}}}}'''
mp={}
for i in range(0,len(pdbs),100):
    r=urllib.request.Request('https://data.rcsb.org/graphql',json.dumps({'query':Q,'variables':{'ids':pdbs[i:i+100]}}).encode(),{'Content-Type':'application/json'})
    for e in json.load(urllib.request.urlopen(r,timeout=120))['data']['entries']:
        for p in e['polymer_entities']:
            s=p['entity_poly']['pdbx_seq_one_letter_code_can'].replace('\n','')
            for inst in p['polymer_entity_instances']:
                c=inst['rcsb_polymer_entity_instance_container_identifiers']
                if (e['rcsb_id'].lower(),c['auth_asym_id']) in keys: mp[(e['rcsb_id'].lower(),c['auth_asym_id'])]=(s,c['auth_to_entity_poly_seq_mapping'])
seqs=sorted({s for s,_ in mp.values()})
num,det,_=run_anarci([(str(i),s) for i,s in enumerate(seqs)],scheme='imgt',ncpu=4,assign_germline=False)[1:]
imgt={}
for s,n,dd in zip(seqs,num,det):
    m={}
    if n:
        for dom,info in zip(n,dd):
            numbering,st,en=dom
            idx=st
            for (pos,ins),aa in numbering:
                if aa!='-': m[idx]=(pos,info['chain_type']); idx+=1
    imgt[s]=m
def region(p,ct):
    if p<=26: return 'FR1'
    if p<=38: return 'CDR1'
    if p<=55: return 'FR2'
    if p<=65: return 'CDR2'
    if p<=104: return 'FR3'
    if p<=117: return 'CDR3'
    return 'FR4'
cnt=collections.Counter(); ev=[]
for r in te.itertuples():
    k=(r.pdb,r.chain)
    if k not in mp: continue
    s,amap=mp[k]
    regs=collections.Counter(); ct=None
    for i,lab in enumerate(amap):
        mnum=re.match(r'-?\d+',lab or '')
        if not mnum or lab=='?': continue
        if r.rs<=int(mnum.group())<=r.re and i in imgt[s]:
            pos,ct=imgt[s][i]; regs[region(pos,ct)]+=1
    if not regs: continue
    for g,n in regs.items(): cnt[(ct in 'KL' and 'Light' or 'Heavy',g)]+=n
    ev.append(dict(pdb=r.pdb,chain=r.chain,range=r.RESIDUE_RANGE,shift=r.SHIFT,size=r.SIZE,chain_type=ct,regions='+'.join(sorted(regs)),method=r.METHOD,res=r.RESOLUTION))
ev=pd.DataFrame(ev)
print('errors in Ab V domains mapped:',len(ev),'of',len(te))
print(pd.Series(cnt).unstack().fillna(0).astype(int)[['FR1','CDR1','FR2','CDR2','FR3','CDR3','FR4']].to_string())
any_cdr=ev.regions.str.contains('CDR')
print('errors touching any CDR: %.0f%%; CDR3: %.0f%%'%(100*any_cdr.mean(),100*ev.regions.str.contains('CDR3').mean()))
print(ev['shift'].abs().value_counts().head(5).to_string()); print(ev['size'].describe().round(1).to_string())
ev.to_csv('/tmp/analysis/antibody_V_errors_mapped.csv',index=False)
