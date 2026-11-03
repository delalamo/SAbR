"""Canonical constant-domain cis-Pro sites located by sequence motif (C-kappa 'FYP' -> Tyr-Pro cis; CH1 'FPEP' -> Phe-Pro
and Glu-Pro cis), plus kappa Pro115. For each chain: is the site (Pro and preceding residue) modelled, and is it annotated cis?
Tests whether the deficit is explained by the site simply not being built (unbuilt residues)."""
import json, re, sys, warnings, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature
warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D=pd.read_csv('frame.csv.gz')
seq=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','chain','seq','vseq'])
D=D.merge(seq,on=['pdb','chain'],how='left')
want=set(zip(D.pdb,D.chain))
feat={}
for l in open(f'{SP}/deeper/cache/pdb_meta/cis_sheet.jsonl'):
    x=json.loads(l); k=(x['rcsb_id'].split('.')[0].lower(),x['auth'])
    if k in want: feat[k]=x['feats']
imgt={}
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        r=json.loads(l); k=(r['pdb'],r['chain'])
        if k not in want or k in imgt: continue
        imgt[k]={x[4]:x[3] for x in r.get('residues') or [] if x[4] is not None and not x[5]}  # imgt pos (no insertion) -> seq index
rows=[]
for c in D.itertuples():
    k=(c.pdb,c.chain); fs=feat.get(k); s=c.seq if isinstance(c.seq,str) else ''
    if fs is None or not s: continue
    cis=set(); unobs=set()
    for f in fs:
        for p in f['feature_positions']:
            b,e=p['beg_seq_id'],p.get('end_seq_id') or p['beg_seq_id']
            if f['type']=='CIS-PEPTIDE': cis.add(e-1)
            elif f['type']=='UNOBSERVED_RESIDUE_XYZ': unobs.update(range(b-1,e))
    vend=s.find(c.vseq)+len(c.vseq) if isinstance(c.vseq,str) and c.vseq in s else 110
    r={'pdb':c.pdb,'chain':c.chain}
    sites=[]
    if c.ctype=='K':
        m=[mm.start()+2 for mm in re.finditer('FYP',s) if mm.start()>vend]
        if m: sites.append(('ck_YP',m[0]))
    if c.ctype=='H':
        m=[mm.start() for mm in re.finditer('FPEP',s) if mm.start()>vend]
        if m: sites+= [('ch1_FP',m[0]+1),('ch1_EP',m[0]+3)]
    if c.ctype=='K' and k in imgt and 115 in imgt[k]:
        i=imgt[k][115]
        if i<len(s) and s[i]=='P': sites.append(('k_P115',i))
    for nm,i in sites:
        r[f'{nm}_present']=1; r[f'{nm}_modelled']=int(i not in unobs and (i-1) not in unobs)
        r[f'{nm}_cis']=int(i in cis)
    rows.append(r)
S=pd.DataFrame(rows); X=D.merge(S,on=['pdb','chain'],how='left')
X.to_csv('03_canonical_sites.csv.gz',index=False)
ZHOU='10.1038/s41422-021-00487-9'
out=[]
def test(sub,feat,label):
    for drop in [False,True]:
        s=sub[sub.cluster!=ZHOU] if drop else sub
        a=fit_feature(s.assign(x=s[feat].astype(float)),'x','bin',['C(ybin)','res'])
        e=s[s.y==1]
        out.append({'test':label+(' | drop Zhou paper' if drop else ''),'n_err':a['n_err'],'n_err_papers':e.cluster.nunique(),'n_ctrl':a['n_ctrl'],
                    'err':a.get('err_val'),'ctrl':a.get('ctrl_val'),'OR':a.get('OR'),'p':a.get('p'),'p_fisher':a.get('p_unadj'),'note':a.get('note')})
for nm,ct,mask_extra in [('ck_YP','K',None),('ch1_FP','H',None),('ch1_EP','H',None),('k_P115','K','noD')]:
    sub=X[(X.ctype==ct)&(X[f'{nm}_present']==1)]
    if mask_extra=='noD': sub=sub[~(sub.mode_D_true&(sub.y==1))]
    # modelled-ness of the site
    test(sub,f'{nm}_modelled',f'{nm}: site modelled (Pro and i-1 not unobserved)')
    test(sub,f'{nm}_cis',f'{nm}: cis | site present (unmodelled counted as not cis)')
    test(sub[sub[f'{nm}_modelled']==1],f'{nm}_cis',f'{nm}: cis | site modelled')
R=pd.DataFrame(out); R.to_csv('03_canonical_sites_tests.csv',index=False)
pd.set_option('display.width',250); print(R.to_string(index=False,float_format=lambda v:f'{v:.3g}'))
