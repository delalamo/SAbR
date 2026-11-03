"""Reproduce DE3 (VH D/E slip -> VL error co-occurrence) and list the contributing entries."""
import pandas as pd, numpy as np, json
from scipy.stats import fisher_exact
import statsmodels.formula.api as smf
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D=f'{SP}/deeper'
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/verify/case_files_DE_DE3_VH_VL_cooccurrence_confounding'
CACHE=f'{D}/cache/verify_DE3_confounding'
MODES=['Dplus_CDR3short','Dminus_CDR3long','E_FR4bulge']
CH=pd.read_csv(f'{D}/chains.csv.gz'); R=pd.read_csv(f'{D}/runs_first_occurrence.csv')
ER=pd.read_pickle(f'{D}/cache/case_files_DE/err_struct.pkl')
bad=set(zip(ER[ER.confirm=='not confirmed'].pdb, ER[ER.confirm=='not confirmed'].chain))
md={}; morig={}
for r in R.itertuples():
    if r.mode in MODES and (r.pdb,r.chain) in bad: continue
    md.setdefault((r.pdb,r.chain),set()).add(r.mode)
    morig.setdefault((r.pdb,r.chain),set()).add(str(r.origin)[:2].strip())
CH['k']=list(zip(CH.pdb,CH.chain))
CH['DE']=[bool(md.get(k,set())&set(MODES)) for k in CH.k]
CH['oth']=[bool(md.get(k,set())-set(MODES)) for k in CH.k]
CH['anyerr']=CH.DE|CH.oth
CH['modes']=[';'.join(sorted(md.get(k,set()))) for k in CH.k]
CH['origins']=[';'.join(sorted(morig.get(k,set()))) for k in CH.k]
CH.to_pickle(f'{CACHE}/chains_flags.pkl')
def entry_table(X):
    rows=[]
    for p,x in X.groupby('pdb'):
        H=x[x.ctype=='H']; L=x[x.ctype!='H']
        if len(H) and len(L):
            rows.append(dict(pdb=p, paper=x.paper.iloc[0], last_author=x.last_author.iloc[0], year=x.year.iloc[0], res=x.res.iloc[0],
              method=x.method.iloc[0], sars2=bool(x.sars2.max()), nH=len(H), nL=len(L), nHseq=H.vseq.nunique(), nLseq=L.vseq.nunique(),
              Hseqs='|'.join(sorted(H.vseq.unique())), Lseqs='|'.join(sorted(L.vseq.unique())),
              H_DE=H.DE.any(), L_DE=L.DE.any(), H_oth=H.oth.any(), L_oth=L.oth.any(), H_any=H.anyerr.any(), L_any=L.anyerr.any(),
              H_lqz=H.lq_z.mean(), L_lqz=L.lq_z.mean(), H_lq=H.lq.mean(), L_lq=L.lq.mean(),
              H_modes=' '.join(f'{c}:{m}' for c,m in zip(H.chain,H.modes) if m), L_modes=' '.join(f'{c}:{m}' for c,m in zip(L.chain,L.modes) if m),
              L_orig=' '.join(f'{c}:{m}' for c,m in zip(L.chain,L.origins) if m), ctypeL=''.join(sorted(L.ctype.unique())),
              irm=x.irm.iloc[0], fit=x.fit.iloc[0], deposit=x.deposit.iloc[0]))
    return pd.DataFrame(rows)
if __name__=='__main__':
    X=CH[(CH.method=='EM')&(CH.vhh==0)]
    T=entry_table(X); T.to_pickle(f'{CACHE}/T_em.pkl')
    TX=entry_table(CH[(CH.method!='EM')&(CH.vhh==0)]); TX.to_pickle(f'{CACHE}/T_xray.pkl')
    print('EM entries',len(T),'H_DE',T.H_DE.sum(),'H_DE&L_oth',(T.H_DE&T.L_oth).sum(),'H_DE&L_DE',(T.H_DE&T.L_DE).sum(),'H_DE&L_any',(T.H_DE&T.L_any).sum())
    print('L_oth|not H_DE', T[~T.H_DE].L_oth.mean(), T[~T.H_DE].L_oth.sum())
    z=T.assign(y=T.L_oth.astype(int),x=T.H_DE.astype(int),yb=pd.cut(T.year,[0,2019,2021,2023,2030]).astype(str))
    m=smf.logit('y ~ x + C(yb) + res',data=z).fit(disp=0,cov_type='cluster',cov_kwds={'groups':pd.factorize(z.paper)[0]})
    print('reproduced adjOR',np.exp(m.params['x']),m.pvalues['x'])
    cols=['pdb','paper','last_author','year','res','sars2','nH','nL','nHseq','nLseq','H_modes','L_modes','L_orig','H_lqz','L_lqz','irm','fit']
    S=T[T.H_DE].sort_values(['paper','pdb'])[cols+['L_oth','L_DE']]
    S.to_csv(f'{OUT}/tables/hDE_entries.csv',index=False)
    pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
    print(S.to_string())
