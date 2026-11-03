"""Build (1) a SAbDab-style high-quality reference of antibody V domains (PDB-wide, X-ray <=2.8/3.0 A),
(2) the unit-level study table (paper|V-region units) with IMGT CDR lengths matched to the reference,
(3) internal-gap flags per chain from SAbR residue records (for the ImmuneBuilder-style 'no missing residues' filter)."""
import pickle, json, re, numpy as np, pandas as pd, collections
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/literature_dean/tables'
import os; os.makedirs(OUT,exist_ok=True)
db=pickle.load(open(f'{SP}/deeper/cache/seq_novelty/abdb.pkl','rb')); D=db['D']
# entry method/resolution PDB-wide
meth={}; resn={}
with open('/tmp/supp/pdbwide/entries.idx',encoding='latin1') as f:
    for i,l in enumerate(f):
        if i<2: continue
        x=l.rstrip('\n').split('\t')
        if len(x)<8: continue
        pid=x[0].lower(); m=x[7].strip()
        meth[pid]='X-ray' if m in ('X-RAY DIFFRACTION',) else ('EM' if 'ELECTRON MICROSCOPY' in m else m)
        try: resn[pid]=float(x[6].split(',')[0])
        except: resn[pid]=np.nan
ents=D.entries.str.split()
def best(es,cond):
    return any(cond(meth.get(e),resn.get(e,np.nan)) for e in es)
D['hq28']=[best(es,lambda m,r: m=='X-ray' and r<=2.8) for es in ents]     # Nowak 2016 / North 2011 style
D['hq30']=[best(es,lambda m,r: m=='X-ray' and r<=3.0) for es in ents]     # Regep 2017 / Guloglu 2023 style
D['abb2']=[best(es,lambda m,r: m in('X-ray','EM') and r<=3.5) for es in ents] # ImmuneBuilder style (gaps not checked here)
D['any_em']=[any(meth.get(e)=='EM' for e in es) for es in ents]
D['first_year']=pd.to_datetime(D.first_release).dt.year
print('reference V domains',len(D),'hq<=2.8',D.hq28.sum(),'hq<=3.0',D.hq30.sum())
D.drop(columns=['vdom','entries']).to_csv(f'{OUT}/ref_vdomains.csv.gz',index=False)
pickle.dump(D[['ctype','vdom','hq28','hq30','first_year','len_CDR1','len_CDR2','len_CDR3','v_gene_hm','species_hm','n_entries']],open(f'{SP}/deeper/cache/literature_dean/ref.pkl','wb'))
# study chains -> lengths from reference (same ANARCI IMGT numbering)
c=pd.read_csv(f'{SP}/deeper/chains.csv.gz')
cn=pd.read_csv('/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables/chain_novelty.csv.gz',usecols=['pdb','chain','dbrow'])
c=c.merge(cn,on=['pdb','chain'],how='left')   # dbrow = row of the PDB-wide reference (seq_novelty lens, same ANARCI IMGT)
ok=(c.dbrow>=0)&(c.dbrow<len(D)); ix=c.dbrow.where(ok,0).astype(int)
for k,col,fb in [('rL1','len_CDR1','cdr1_len'),('rL2','len_CDR2','L2'),('rL3','len_CDR3','cdr3_len')]:
    c[k]=np.where(ok,D[col].values[ix],c[fb])   # fall back to chains.csv (same ANARCI IMGT) for V domains absent from the reference
print('chains not in reference (fallback lengths):',(~ok).sum())
print('chains matched to reference lengths',c.rL1.notna().mean().round(4), ' agreement with chains.csv cdr3_len', (c.rL3==c.cdr3_len).mean().round(4),' cdr1',(c.rL1==c.cdr1_len).mean().round(4))
# internal gaps in modelled V domain (SAbR records): any missing seq_index between first and last modelled V residue
gap={}
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        d=json.loads(l)
        if d.get('status')!='ok': continue
        si=[r[3] for r in d['residues'] if r[3] is not None]
        an=[r[4] for r in d['residues']]
        if not si: continue
        si=sorted(si); internal=sum(b-a-1 for a,b in zip(si,si[1:]) if b-a>1)
        # n-terminal missing: first anarci imgt position >3
        a0=[x for x in an if x is not None]
        gap[(d['pdb'],d['chain'])]=(internal, (min(a0) if a0 else np.nan))
c['int_gap']=[gap.get((p,ch),(np.nan,np.nan))[0] for p,ch in zip(c.pdb,c.chain)]
c['first_imgt']=[gap.get((p,ch),(np.nan,np.nan))[1] for p,ch in zip(c.pdb,c.chain)]
print('gap info for chains',c.int_gap.notna().mean().round(4))
c.drop(columns=['seq']).to_pickle(f'{SP}/deeper/cache/literature_dean/chains_aug.pkl')
