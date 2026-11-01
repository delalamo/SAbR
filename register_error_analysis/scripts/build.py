import json, re, pandas as pd, numpy as np, collections
S='/tmp/supp'
# --- errors
t=pd.read_excel(f'{S}/epmc/m-11-00938-sup2.xlsx').drop(columns=['Unnamed: 0'],errors='ignore')
t[['pdb','chain']]=t.PDB_CHAIN_ID.str.split(':',expand=True)
t['pdb']=t.pdb.str.lower()
rr=t.RESIDUE_RANGE.str.extract(r'(-?\d+)\s*-\s*(-?\d+)').astype(int)
t['rs'],t['re']=rr[0],rr[1]
# --- denominator chains
meta={}
for l in open(f'{S}/rcsb/meta.jsonl'):
    e=json.loads(l); meta[e['rcsb_id'].lower()]=e
seqinfo={}
for l in open(f'{S}/rcsb/seqs.jsonl'):
    e=json.loads(l)
    for p in e['polymer_entities'] or []:
        s=(p['entity_poly']['pdbx_seq_one_letter_code_can'] or '').replace('\n','')
        for c in p['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] or []:
            seqinfo[(e['rcsb_id'].lower(),c)]=(s,p['rcsb_polymer_entity']['pdbx_description'],p['entity_poly']['type'])
an=json.load(open(f'{S}/rcsb/anarci.json'))
rows=[]
for pdb,e in meta.items():
    meth=[m['method'] for m in e['exptl']]
    method='EM' if 'ELECTRON MICROSCOPY' in meth else 'X-ray'
    res=(e['rcsb_entry_info']['resolution_combined'] or [None])[0]
    yr=int(e['rcsb_accession_info']['deposit_date'][:4])
    for p in e['polymer_entities'] or []:
        if p['entity_poly']['type']!='polypeptide(L)': continue
        L=p['entity_poly']['rcsb_sample_sequence_length']
        if L>1000 or L<20: continue
        for c in p['rcsb_polymer_entity_container_identifiers']['auth_asym_ids'] or []:
            s,desc,_=seqinfo.get((pdb,c),('', '', ''))
            a=an.get(s,[])
            types=sorted({d['chain_type'] for d in a})
            ab='antibody' if any(x in ('H','K','L') for x in types) else ('TCR' if any(x in ('A','B','G','D') for x in types) else 'other')
            rows.append(dict(pdb=pdb,chain=c,method=method,res=res,year=yr,length=L,ab=ab,
                             n_vdom=len(a),desc=desc,seq=s))
ch=pd.DataFrame(rows)
ch['res_bin']=pd.cut(ch.res,[2.999,3.25,3.5,3.75,4.0,4.5,5.0])
# errors per chain
eg=t.groupby(['pdb','chain']).agg(n_err=('SIZE','size'),err_res=('SIZE','sum')).reset_index()
ch=ch.merge(eg,on=['pdb','chain'],how='left').fillna({'n_err':0,'err_res':0})
ch['flag']=(ch.n_err>0).astype(int)
unmatched=t.merge(ch[['pdb','chain']],on=['pdb','chain'],how='left',indicator=True)
print('errors matched to denominator chains:',(unmatched._merge=='both').mean())
ch.to_pickle('/tmp/analysis/chains.pkl')
# --- CATH domains
cols=['dom','C','A','T','H','S','O','L','I','D','len','res']
dl=pd.read_csv(f'{S}/cath/cath-domain-list.txt',sep=r'\s+',comment='#',names=cols)
dl['pdb']=dl.dom.str[:4]; dl['chain']=dl.dom.str[4]
dl['cath_T']=dl.C.astype(str)+'.'+dl.A.astype(str)+'.'+dl['T'].astype(str)
dl['cath_A']=dl.C.astype(str)+'.'+dl.A.astype(str)
dl['cath_H']=dl.cath_T+'.'+dl.H.astype(str)
bd={}
for l in open(f'{S}/cath/cath-domain-boundaries-seqreschopping.txt'):
    d,segs=l.split()
    out=[]
    for sg in segs.split(','):
        m=re.match(r'(-?\d+)[A-Z]?-(-?\d+)[A-Z]?$',sg)
        if m: out.append((int(m[1]),int(m[2])))
    bd[d]=out
dl['segs']=dl.dom.map(bd)
keys=set(zip(ch.pdb,ch.chain))
dl=dl[[k in keys for k in zip(dl.pdb,dl.chain)]].copy()
dl['dstart']=dl.segs.map(lambda s: min(a for a,b in s) if s else np.nan)
# assign errors to domain with max overlap
errs=collections.defaultdict(list)
for r in t.itertuples(): errs[(r.pdb,r.chain)].append((r.rs,r.re,r.SIZE))
dl=dl.reset_index(drop=True)
dl['n_err']=0; dl['err_res']=0
bykey=dl.groupby(['pdb','chain']).groups
unassigned=0;assigned=0
for k,elist in errs.items():
    if k not in bykey: continue
    idx=list(bykey[k])
    for rs,re_,size in elist:
        best,bo=None,0
        for i in idx:
            ov=sum(max(0,min(b,re_)-max(a,rs)+1) for a,b in dl.at[i,'segs'])
            if ov>bo: best,bo=i,ov
        if best is None: unassigned+=1; continue
        assigned+=1
        dl.at[best,'n_err']+=1; dl.at[best,'err_res']+=bo
print('errors on CATH-chopped chains: assigned',assigned,'no overlap',unassigned)
dl['flag']=(dl.n_err>0).astype(int)
dl=dl.merge(ch[['pdb','chain','method','res','res_bin','year','ab','length']],on=['pdb','chain'])
# antibody V vs C: rank Ig (2.60.40.10) domains along the chain
dl['ig']=dl.cath_H=='2.60.40.10'
dl['ig_rank']=dl[dl.ig].groupby(['pdb','chain']).dstart.rank(method='first')
def grp(r):
    if r.ig and r.ab=='antibody': return 'Antibody V' if r.ig_rank==1 else 'Antibody C'
    if r.ig and r.ab=='TCR': return 'TCR V' if r.ig_rank==1 else 'TCR C'
    if r.ig: return 'Ig-like (2.60.40.10), non-antibody/TCR'
    if r.cath_T=='2.60.40': return 'Other Ig-like topology 2.60.40.x'
    if r.cath_A=='2.60': return 'Other β-sandwich (2.60.x)'
    if r.C==2: return 'Other mainly-β'
    return {1:'Mainly α',3:'α/β',4:'Few SS',6:'Special'}.get(r.C,'other')
dl['group']=dl.apply(grp,axis=1)
dl.drop(columns=['segs']).to_pickle('/tmp/analysis/domains.pkl')
print(dl.group.value_counts())
