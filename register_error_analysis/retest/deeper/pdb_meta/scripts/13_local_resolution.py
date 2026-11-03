"""Map-intrinsic local resolution (EMDB validation analysis) for antibody chains and at error sites.
(a) chain level: antibody V-domain local resolution relative to the reported map resolution and to non-antibody residues;
(b) site level: local resolution at each first-occurrence error run (ANARCI IMGT window) vs the same IMGT window in clean
    chains of the same chain type in year-matched control entries (percentile per run)."""
import json, sys, collections, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature, bh, ybin
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/tables'
VA={}
for l in open(f'{SP}/deeper/cache/pdb_meta/emdb_va.jsonl'):
    x=json.loads(l)
    if x.get('status')=='ok': VA[x['pdb']]=x
ent=pd.read_csv(f'{OUT}/em_entry_analysis_frame.csv.gz',low_memory=False)[['pdb','y','res','dep_year','ybin','cluster','l2nV']]
ent=ent[ent.pdb.isin(VA)]
ch=pd.read_csv(f'{SP}/deeper/chains.csv.gz',usecols=['pdb','chain','ctype','err_first','err_any','paper'])
ch=ch[ch.pdb.isin(ent.pdb)]
runs=pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv'); runs=runs[runs.pdb.isin(ent.pdb)]
want=set(zip(ch.pdb,ch.chain)); rec={}
for fn in ['new','snapshot_rest','validation']:
    for l in open(f'{SP}/sabr/out/{fn}.jsonl'):
        x=json.loads(l)
        if (x['pdb'],x['chain']) in want and (x['pdb'],x['chain']) not in rec: rec[(x['pdb'],x['chain'])]=x.get('residues') or []
def resmap(pdb,kind):
    x=VA[pdb].get(kind) or []
    m={}
    for lab,v in x:
        try:
            c,rest=lab.split(':',1); num=rest.split()[0]; m[(c,num)]=v
        except Exception: pass
    return m
rows=[]; site=[]
for pdb in ent.pdb:
    lr=resmap(pdb,'res_lr'); qq=resmap(pdb,'res_q')
    if not lr and not qq: continue
    e=ent[ent.pdb==pdb].iloc[0]
    abkeys=set()
    for c in ch[ch.pdb==pdb].itertuples():
        R=rec.get((pdb,c.chain)) or []
        vals={}; qv={}
        for a in R:
            k=(c.chain,f"{a[0]}{a[1] or ''}")
            abkeys.add(k)
            if k in lr and a[4] is not None: vals[(a[4],a[5] or '')]=lr[k]
            if k in qq and a[4] is not None: qv[(a[4],a[5] or '')]=qq[k]
        if len(vals)<30 and len(qv)<30: continue
        rows.append({'pdb':pdb,'chain':c.chain,'ctype':c.ctype,'err_first':c.err_first,'err_any':c.err_any,'paper':c.paper,'res':e.res,'ybin':e.ybin,
                     'v_lr_mean':np.mean(list(vals.values())) if len(vals)>=30 else np.nan,'imgt_lr':vals,'imgt_q':qv})
    # all chains of antibody: abkeys includes only V residues mapped; non-antibody residues = all residues of chains not in ch
    abch=set(ch[ch.pdb==pdb].chain)
    non=[v for (c,n),v in lr.items() if c not in abch]
    for r in rows:
        if r['pdb']==pdb: r['non_lr_median']=np.median(non) if len(non)>50 else np.nan; r['map_lr_median']=VA[pdb].get('map_lr_median')
C=pd.DataFrame(rows)
C['v_lr_minus_res']=C.v_lr_mean-C.res
C['v_lr_minus_non']=C.v_lr_mean-C.non_lr_median
C['v_lr_ratio_res']=C.v_lr_mean/C.res
D=C[(C.err_first)|(~C.err_any)].copy(); D['y']=D.err_first.astype(int); D['cluster']=D.paper.fillna(D.pdb)
res=[]
for f in ['v_lr_mean','v_lr_minus_res','v_lr_ratio_res','v_lr_minus_non']:
    r=fit_feature(D,f,'cont',['C(ybin)','res','C(ctype)']); r['level']='V chain (local resolution, A; higher = worse)'; res.append(r)
# entry-level: mean over antibody chains vs non-antibody
E=D.groupby('pdb').agg(y=('y','max'),v_lr_minus_non=('v_lr_minus_non','mean'),v_lr_minus_res=('v_lr_minus_res','mean'),res=('res','first'),ybin=('ybin','first'),cluster=('cluster','first')).reset_index()
for f in ['v_lr_minus_res','v_lr_minus_non']:
    r=fit_feature(E,f,'cont',['C(ybin)','res']); r['level']='entry (mean of antibody chains)'; res.append(r)
# site level percentiles
clean=D[D.y==0]
pct=[]
for q in runs.itertuples():
    row=D[(D.pdb==q.pdb)&(D.chain==q.chain)]
    if row.empty: continue
    row=row.iloc[0]
    def win(vals): 
        v=[x for (p,i),x in vals.items() if q.imgt0<=p<=q.imgt1]
        return np.mean(v) if len(v)>=2 else np.nan
    s=win(row.imgt_lr)-row.res
    ref=[win(r.imgt_lr)-r.res for r in clean[clean.ctype==row.ctype].itertuples()]
    ref=np.array([x for x in ref if x==x])
    sq=win(row.imgt_q); refq=np.array([x for x in [win(r.imgt_q) for r in clean[clean.ctype==row.ctype].itertuples()] if x==x])
    pct.append({'pdb':q.pdb,'chain':q.chain,'mode':q.mode,'imgt0':q.imgt0,'imgt1':q.imgt1,
                'site_lr_minus_res':s,'ref_median':np.median(ref) if len(ref) else np.nan,'pct_worse_than_clean':100*np.mean(ref<s) if (s==s and len(ref)>=30) else np.nan,
                'site_q':sq,'ref_q_median':np.median(refq) if len(refq) else np.nan,'q_pct_worse_than_clean':100*np.mean(refq>sq) if (sq==sq and len(refq)>=30) else np.nan})
P=pd.DataFrame(pct)
R=bh(pd.DataFrame(res)); R.to_csv(f'{OUT}/local_resolution_tests.csv',index=False)
P.to_csv(f'{OUT}/local_resolution_sites.csv',index=False)
pd.set_option('display.width',250)
print(len(VA),'VA entries;',ent[ent.pdb.isin([p for p in VA if VA[p].get('res_lr')])].y.value_counts().to_dict(),'with per-residue local resolution (y=1 error)')
print(R[['level','feature','n_err','n_ctrl','err_val','ctrl_val','unit','OR','lo','hi','p','q']].to_string(index=False,float_format=lambda x:f'{x:.3g}'))
if len(P):
    # one value per (pdb-chain-run) -> collapse identical copies: one per pdb+mode
    G=P.groupby(['pdb','mode','imgt0']).pct_worse_than_clean.mean().dropna()
    GQ=P.groupby(['pdb','mode','imgt0']).q_pct_worse_than_clean.mean().dropna()
    print('Q-score site percentile (100 = worst Q among clean chains at same window): n entry-sites',len(GQ),'median',GQ.median(),'frac >75th',(GQ>75).mean(),'Wilcoxon p=',stats.wilcoxon(GQ-50).pvalue)
    both=pd.concat([G,GQ],axis=1,keys=['lr','q']).dropna()
    print('same sites with both metrics: n',len(both),'median local-res pct',both.lr.median(),'median Q pct',both.q.median(),'paired Wilcoxon p',stats.wilcoxon(both.lr,both.q).pvalue)
    print('site percentile (100 = worst local resolution among clean chains at the same IMGT window): n runs',len(P),'n entry-sites',len(G),
          'median',G.median(),'frac >75th pct',(G>75).mean(),'Wilcoxon vs 50 p=',stats.wilcoxon(G-50).pvalue)
    print(P.groupby('mode').pct_worse_than_clean.agg(['median','size']))
