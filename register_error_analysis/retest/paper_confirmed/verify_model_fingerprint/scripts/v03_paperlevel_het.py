"""Paper-level heterogeneity (one value per paper) for T3/T4/T5/T8a, single-paper dependence of T5 D/E, T7b, and
cluster-robust-SE small-cluster check. Verifier's own code."""
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DEEP='/home/user/SAbR/register_error_analysis/retest/deeper'
MF='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint'
OUT='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_model_fingerprint/tables'
ZHOU='10.1038/s41422-021-00487-9'
L=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); L.append(s)
C=pd.read_csv(f'{SP}/deeper/chains.csv.gz',dtype={'chain':str}).drop(columns=['seq'],errors='ignore')
C=C.merge(pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz',dtype={'chain':str}),on=['pdb','chain'])
C=C[~C.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False)].copy()
C['clu']=C.paper.fillna(C.pdb); C['err_any']=C.err_any.astype(bool); C['err_first']=C.err_first.astype(bool)
C['rest']=C.err_first&~C.paper_case_incl_copies&~C.paper_case_strict
def phet(df,val,lab,meth):
    d=df[df.method==meth]
    ca=d[d.paper_case_strict]; ct=d[d.paper_control]; rs=d[d.rest]; cl=d[~d.err_any]
    pc=ca.groupby('clu')[val].mean(); pk=ct.groupby('clu')[val].mean(); pr=rs.groupby('clu')[val].mean(); pl=cl.groupby('clu')[val].mean()
    ep=pc-pk.mean(); er=pr-pl.mean()
    # bootstrap over papers for paper-level excess (case papers and control papers resampled)
    rng=np.random.default_rng(1); bs=[]
    for _ in range(4000):
        bs.append(rng.choice(pc.values,len(pc)).mean()-rng.choice(pk.values,len(pk)).mean())
    lo,hi=np.percentile(bs,[2.5,97.5])
    w=stats.ttest_ind(ep,er,equal_var=False).pvalue
    epz=ep.drop(ZHOU,errors='ignore')
    P(f'{lab} {meth}: paper-level excess paper {ep.mean():+.3g} (boot {lo:+.3g} to {hi:+.3g}; {len(pc)} papers) vs rest {er.mean():+.3g} ({len(pr)} papers); het Welch p={w:.2g}; without Zhou paper {epz.mean():+.3g}, het p={stats.ttest_ind(epz,er,equal_var=False).pvalue:.2g}; chain-level paper {ca[val].mean()-ct[val].mean():+.3g} (Zhou chains {int((ca.clu==ZHOU).sum())}/{len(ca)}) vs rest {rs[val].mean()-cl[val].mean():+.3g}')
# T8a sheet
CS=pd.read_csv(f'{DEEP}/pdb_meta/tables/chain_cis_sheet.csv.gz',dtype={'chain':str})[['pdb','chain','v_sheet_away']]
S=C.merge(CS,on=['pdb','chain']).dropna(subset=['v_sheet_away'])
for m in ['EM','X-ray']: phet(S,'v_sheet_away','T8a sheet',m)
# T3
T=pd.concat([pd.read_pickle(f'{SP}/deeper/cache/verify_su_gap/tallies.pkl'),pd.read_pickle(f'{SP}/paperfp/tallies_xray.pkl')]); T['chain']=T.chain.astype(str)
V=T[(T.cat=='V')&T.dbin.isin(['d4_10','d11p','clean'])].groupby(['pdb','chain'])[['n_rama','allowed']].sum(); V=V[V.n_rama>=30]; V['pA']=100*V.allowed/V.n_rama
R=C.merge(V.reset_index(),on=['pdb','chain'])
for m in ['EM','X-ray']: phet(R,'pA','T3 rama',m)
# T4 entry gap: V minus non-V
g=T.assign(isV=T.cat=='V').groupby(['pdb','isV'])[['n_rama','allowed']].sum().unstack()
ent=pd.DataFrame({'Vn':g[('n_rama',True)],'VA':100*g[('allowed',True)]/g[('n_rama',True)],'On':g[('n_rama',False)],'OA':100*g[('allowed',False)]/g[('n_rama',False)]})
ent=ent[ent.On>=30]; ent['gap']=ent.VA-ent.OA
eg=C.groupby('pdb').agg(method=('method','first'),clu=('clu','first'),s=('paper_case_strict','any'),i=('paper_case_incl_copies','any'),
    r=('rest','any'),ctl=('paper_control','all'),cl=('err_any',lambda s:(~s).all()))
E=ent.join(eg,how='inner').reset_index()
E['paper_case_strict']=E.s; E['paper_control']=E.ctl&~E.s; E['rest']=E.r&~E.i&~E.s; E['err_any']=~E.cl
for m in ['EM','X-ray']: phet(E,'gap','T4 gap',m)
P(f'T4 EM strict entries {int((E.s&(E.method=="EM")).sum())}, controls {int((E.paper_control&(E.method=="EM")).sum())}')
# T5 D/E without Zhou, T7b without Zhou
F=pd.read_csv(f'{MF}/tables/fw_computed.csv',dtype={'chain':str})[['pdb','chain','rep_rmsd']]
VD=f'{DEEP}/verify/case_files_DE_DE2_framework_distortion_integrity'
F0=pd.concat([pd.read_csv(f'{VD}/fw_recomputed.csv',dtype={'chain':str}),pd.read_csv(f'{VD}/fw_recomputed_extra.csv',dtype={'chain':str})])[['pdb','chain','rep_rmsd']]
X=C.merge(pd.concat([F,F0]).drop_duplicates(['pdb','chain']).dropna(),on=['pdb','chain'])
for m in ['EM','X-ray']: phet(X,'rep_rmsd','T5 fw',m)
A=pd.read_csv(f'{SP}/paperconf/paper_errors_annotated.csv',dtype={'chain':str})
de=A[(A.status=='confirmed first occurrence')&A['mode'].fillna('').str.contains('Dplus|Dminus|E_FR4')]
X['DE']=[k in set(zip(de.pdb,de.chain)) for k in zip(X.pdb,X.chain)]
x=X[(X.method=='EM')]; ct=x[x.paper_control]
for lab,ca in [('D/E all',x[x.paper_case_strict&x.DE]),('D/E no Zhou',x[x.paper_case_strict&x.DE&(x.clu!=ZHOU)]),('non-D/E',x[x.paper_case_strict&~x.DE])]:
    pc=ca.groupby('clu').rep_rmsd.mean()
    P(f'T5 EM {lab}: {len(ca)} chains/{ca.clu.nunique()} papers mean {ca.rep_rmsd.mean():.3f} vs ctrl {ct.rep_rmsd.mean():.3f}: {ca.rep_rmsd.mean()-ct.rep_rmsd.mean():+.3f}; paper-level {pc.mean()-ct.groupby("clu").rep_rmsd.mean().mean():+.3f}')
# T7b within entry: error chain vs clean V neighbours (all chains with rmsd in entries holding a strict chain)
rows=[]
for p,g in X[X.pdb.isin(X[X.paper_case_strict].pdb)].groupby('pdb'):
    e=g[g.paper_case_strict]; c=g[~g.err_any]
    if len(e) and len(c): rows.append(dict(pdb=p,clu=g.clu.iloc[0],method=g.method.iloc[0],d=e.rep_rmsd.mean()-c.rep_rmsd.mean()))
W=pd.DataFrame(rows)
for m in ['EM','all']:
    w=W if m=='all' else W[W.method==m]
    pw=w.groupby('clu').d.mean()
    P(f'T7b {m}: entries {len(w)} mean diff {w.d.mean():+.3f} (Zhou entries {int((w.clu==ZHOU).sum())}, without Zhou {w[w.clu!=ZHOU].d.mean():+.3f} n={int((w.clu!=ZHOU).sum())}); per paper {pw.mean():+.3f} n={len(pw)}, t p={stats.ttest_1samp(pw,0).pvalue:.2g}, Wilcoxon p={stats.wilcoxon(pw).pvalue:.2g}')
# rama within-entry similar
RR=R[R.pdb.isin(R[R.paper_case_strict].pdb)]
rows=[]
for p,g in RR.groupby('pdb'):
    e=g[g.paper_case_strict]; c=g[~g.err_any]
    if len(e) and len(c): rows.append(dict(pdb=p,clu=g.clu.iloc[0],method=g.method.iloc[0],d=e.pA.mean()-c.pA.mean()))
W=pd.DataFrame(rows); pw=W.groupby('clu').d.mean()
P(f'T7a all: entries {len(W)} mean diff {W.d.mean():+.2f}; per paper {pw.mean():+.2f} n={len(pw)} Wilcoxon p={stats.wilcoxon(pw).pvalue:.2g}')
open(f'{OUT}/v03_paperlevel_het.txt','w').write('\n'.join(L)+'\n')
