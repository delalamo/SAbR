"""Statistical robustness of the constant-domain and Pro115 cis deficits: clustering, pseudoreplication,
leave-one-paper-out, paper/unit-level tests, cluster bootstrap, and conditioning on entry-wide all-trans models."""
import sys, json, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm, warnings
from scipy import stats
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/user/SAbR/register_error_analysis/retest/deeper/pdb_meta/scripts')
from statlib import fit_feature
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/pdb_meta_cis_constant_and_L3_integrity'
D=pd.read_csv(f'{OUT}/frame_em.csv.gz')
ecis={}
for l in open(f'{SP}/deeper/cache/pdb_meta/rcsb_meta.jsonl'):
    e=json.loads(l); ecis[e['rcsb_id'].lower()]=(e.get('rcsb_entry_info') or {}).get('cis_peptide_count')
D['entry_cis']=D.pdb.map(ecis); D['other_cis']=D.entry_cis-D.n_cis_chain
D['entry_alltrans']=(D.entry_cis==0).astype(float); D['other_alltrans']=(D.other_cis<=0).astype(float)
modes=D.modes.fillna('')
D['mode_D_inv']=modes.str.contains('D|E_FR4'); D['mode_D_true']=modes.str.contains('Dplus|Dminus|E_FR4')
D['c115']=D.cis115.astype(float)
ZHOU='10.1038/s41422-021-00487-9'
sets={
 'Ck':(D[(D.ctype=='K')&D.const_ok],'ccis',['C(ybin)','res']),
 'CH1':(D[(D.ctype=='H')&D.const_ok],'ccis',['C(ybin)','res']),
 'pooled_const':(D[D.const_ok],'ccis',['C(ybin)','res','C(ctype)']),
 'Pro115_inv':(D[(D.ctype=='K')&D.pro115&~(D.mode_D_inv&(D.y==1))],'c115',['C(ybin)','res']),
 'Pro115_corr':(D[(D.ctype=='K')&D.pro115&~(D.mode_D_true&(D.y==1))],'c115',['C(ybin)','res']),
}
rows=[]
def rec(name,test,r,extra=None):
    d={'set':name,'test':test,**{k:r.get(k) for k in ['n_err','n_ctrl','err_val','ctrl_val','OR','lo','hi','p','note']}}
    if extra: d.update(extra)
    rows.append(d); print(name,'|',test,'|',{k:d.get(k) for k in ['err_val','ctrl_val','OR','p','note']})
def glm_or(x,feat,cov):
    f=f'y ~ {feat}'+(' + '+' + '.join(cov) if cov else '')
    m=smf.glm(f,data=x,family=sm.families.Binomial()).fit()
    return np.exp(m.params[feat])
for name,(S,feat,cov) in sets.items():
    S=S.copy(); S['cluster']=S.paper.fillna(S.pdb)
    E=S[S.y==1]
    print(f'\n=== {name}: error chains {len(E)} from {E.cluster.nunique()} papers, {E.unit.nunique()} units, {E.pdb.nunique()} entries; Zhou chains {int((E.cluster==ZHOU).sum())}')
    rec(name,'investigator model (chain, cluster-robust by paper)',fit_feature(S,feat,'bin',cov))
    rec(name,'drop Zhou Q. paper',fit_feature(S[S.cluster!=ZHOU],feat,'bin',cov))
    # leave-one-paper-out over error papers
    lo=[]
    for p in E.cluster.unique():
        r=fit_feature(S[S.cluster!=p],feat,'bin',cov); lo.append((p,r.get('OR'),r.get('p'),int((E.cluster==p).sum())))
    L=pd.DataFrame(lo,columns=['paper','OR','p','n_err_dropped']); L['set']=name
    L.to_csv(f'{OUT}/04_lopo_{name}.csv',index=False)
    rec(name,'leave-one-paper-out: worst case',{'OR':L.OR.max(),'p':L.p.max(),'note':f'paper dropped {L.loc[L.p.idxmax(),"paper"]}'})
    # unit level (one vote per antibody-in-paper, majority of its chains)
    U=S.groupby(['unit','y']).agg(v=(feat,'mean'),cluster=('cluster','first'),ybin=('ybin','first'),res=('res','median'),ctype=('ctype','first')).reset_index()
    U['vb']=(U.v>=0.5).astype(float)
    rec(name,'unit level (antibody-in-paper; majority vote), cluster by paper',fit_feature(U,'vb','bin',cov))
    rec(name,'unit level, drop Zhou Q.',fit_feature(U[U.cluster!=ZHOU],'vb','bin',cov))
    # paper level: one vote per paper (error chains of the paper vs control chains of control papers)
    P=S.groupby(['cluster','y']).agg(v=(feat,'mean')).reset_index(); P['vb']=(P.v>=0.5).astype(int)
    a=P[P.y==1].vb; b=P[P.y==0].vb
    pf=stats.fisher_exact([[a.sum(),len(a)-a.sum()],[b.sum(),len(b)-b.sum()]])
    rec(name,'paper level (majority vote), Fisher unadjusted',{'err_val':f'{a.mean()*100:.1f}% ({a.sum()}/{len(a)})','ctrl_val':f'{b.mean()*100:.1f}% ({b.sum()}/{len(b)})','OR':pf[0],'p':pf[1]})
    a2=P[(P.y==1)&(P.cluster!=ZHOU)].vb
    pf=stats.fisher_exact([[a2.sum(),len(a2)-a2.sum()],[b.sum(),len(b)-b.sum()]])
    rec(name,'paper level, drop Zhou Q.',{'err_val':f'{a2.mean()*100:.1f}% ({a2.sum()}/{len(a2)})','ctrl_val':f'{b.mean()*100:.1f}%','OR':pf[0],'p':pf[1]})
    # cluster bootstrap of the adjusted OR (resample papers within error/clean strata)
    x=S[['y',feat,'cluster','ybin','res','ctype']].dropna()
    cl=x.groupby('cluster'); keys=list(cl.groups); g={k:x.loc[v] for k,v in cl.groups.items()}
    ek=[k for k in keys if g[k].y.max()==1]; ck=[k for k in keys if g[k].y.max()==0]
    rng=np.random.default_rng(11); bs=[]
    for it in range(400):
        sk=list(rng.choice(ek,len(ek)))+list(rng.choice(ck,len(ck)))
        xb=pd.concat([g[k] for k in sk])
        try:
            o=glm_or(xb,feat,[c for c in cov if c!='C(ybin)']+['C(ybin)'])
            if np.isfinite(o): bs.append(o)
        except Exception: pass
    bs=np.array(bs); bs=bs[(bs>0)&(bs<1e6)]
    rec(name,'cluster (paper) bootstrap of adjusted OR, 400 reps',{'OR':np.median(bs),'lo':np.quantile(bs,.025),'hi':np.quantile(bs,.975),'p':(np.sum(bs>=1)+1)/(len(bs)+1),'note':f'one-sided bootstrap p = share of reps with OR>=1; n_ok={len(bs)}'})
    # condition on entry-wide all-trans models
    rec(name,'only chains whose entry has >=1 cis peptide in OTHER chains',fit_feature(S[S.other_cis>0],feat,'bin',cov))
    rec(name,'adjusted for entry all-trans elsewhere (other_alltrans)',fit_feature(S,feat,'bin',cov+['other_alltrans']))
    rec(name,'drop Zhou + only entries with cis elsewhere',fit_feature(S[(S.other_cis>0)&(S.cluster!=ZHOU)],feat,'bin',cov))
R=pd.DataFrame(rows); R.to_csv(f'{OUT}/04_robust.csv',index=False)
# entry level: is 'no cis anywhere else in entry' enriched in error entries (any V chain)?
EN=D.groupby('pdb').agg(y=('y','max'),alltrans=('entry_alltrans','first'),ybin=('ybin','first'),res=('res','first'),cluster=('paper',lambda s:s.fillna('x').iloc[0])).reset_index()
r=fit_feature(EN,'alltrans','bin',['C(ybin)','res']); print('\nentry level: entry has zero cis peptides anywhere',{k:r.get(k) for k in ['err_val','ctrl_val','OR','p']})
r=fit_feature(EN[EN.cluster!=ZHOU],'alltrans','bin',['C(ybin)','res']); print('  drop Zhou',{k:r.get(k) for k in ['err_val','ctrl_val','OR','p']})
# partner chains: clean chains in error entries vs clean chains in clean entries (constant-domain cis)
errent=set(D[D.y==1].pdb)
Cc=D[(D.y==0)&D.const_ok].copy(); Cc['y']=Cc.pdb.isin(errent).astype(int); Cc['cluster']=Cc.paper.fillna(Cc.pdb)
r=fit_feature(Cc,'ccis','bin',['C(ybin)','res','C(ctype)']); print('clean partner chains in error entries vs clean chains elsewhere (const cis)',{k:r.get(k) for k in ['err_val','ctrl_val','OR','p']})
r=fit_feature(Cc[Cc.cluster!=ZHOU],'ccis','bin',['C(ybin)','res','C(ctype)']); print('  drop Zhou',{k:r.get(k) for k in ['err_val','ctrl_val','OR','p']})
