"""Verifier's own recomputation of the most consequential numbers (T1 Pro8 cis, T3 Ramachandran beyond site,
T5 framework RMSD, T6 VH->VL, T9 constant cis). Uses raw-ish inputs (chains.csv.gz, chain_flags, pdb_meta chain_cis_sheet,
validation tallies, fw values) and own statistics; NOT the investigator's binsuite/common."""
import json, numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
DEEP='/home/user/SAbR/register_error_analysis/retest/deeper'
MF='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/model_fingerprint'
OUT='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_model_fingerprint/tables'
ZHOU='10.1038/s41422-021-00487-9'
C=pd.read_csv(f'{SP}/deeper/chains.csv.gz',dtype={'chain':str}).drop(columns=['seq'],errors='ignore')
C=C.merge(pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz',dtype={'chain':str}),on=['pdb','chain'])
C['amy']=C.title.fillna('').str.contains(r'amyloid(?!-beta)',case=False); C=C[~C.amy].copy()
C['clu']=C.paper.fillna(C.pdb); C['err_any']=C.err_any.astype(bool); C['err_first']=C.err_first.astype(bool)
A=pd.read_csv(f'{SP}/paperconf/paper_errors_annotated.csv',dtype={'chain':str})
cop=set(zip(*A[A.status.isin(['copy of an earlier model','near copy (ledger)','confirmed, loose-ledger doubt'])][['pdb','chain']].values.T))
C['copyc']=[k in cop for k in zip(C.pdb,C.chain)]
C['incl']=C.paper_case_strict|C.paper_case_incl_copies
C['rest']=C.err_first&~C.incl
def hald(a,n1,c,n0):
    b=n1-a; d=n0-c; A_,B,Cc,D=a+.5,b+.5,c+.5,d+.5; o=A_*D/(B*Cc); se=np.sqrt(1/A_+1/B+1/Cc+1/D)
    return o,o*np.exp(-1.96*se),o*np.exp(1.96*se),stats.fisher_exact([[a,b],[c,d]])[1],se
lines=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); lines.append(s)
# ---------------- T1 Pro8
CS=pd.read_csv(f'{DEEP}/pdb_meta/tables/chain_cis_sheet.csv.gz',dtype={'chain':str})
K=C.merge(CS[['pdb','chain','v_pro_pos','v_cis_pos','const_cis_n','n_const_modeled','const_pro_n']],on=['pdb','chain'])
K['pro8']=K.v_pro_pos.fillna('').str.split(';').apply(lambda a:'8' in a)
K['cis8']=K.v_cis_pos.fillna('').str.split(';').apply(lambda a:'8PRO' in a)
k=K[(K.ctype=='K')&K.pro8]
for meth in ['EM','X-ray']:
    m=k[k.method==meth]; ca=m[m.paper_case_strict]; ct=m[m.paper_control]
    o=hald(int(ca.cis8.sum()),len(ca),int(ct.cis8.sum()),len(ct))
    P(f'T1 {meth}: strict {int(ca.cis8.sum())}/{len(ca)} ({ca.clu.nunique()} papers; Zhou {int((ca.clu==ZHOU).sum())}) vs ctrl {int(ct.cis8.sum())}/{len(ct)} ({ct.cis8.mean():.2f}); Haldane OR {o[0]:.3g} ({o[1]:.2g}-{o[2]:.2g}) p={o[3]:.2g}')
m=k[k.method=='EM']; ca=m[m.paper_case_strict]; ct=m[m.paper_control]
# year-matched controls
ctm=ct[ct.year.isin(ca.year.unique())]; P('  EM case years',sorted(ca.year.unique()),'ctrl in same years cis',f'{int(ctm.cis8.sum())}/{len(ctm)} ({ctm.cis8.mean():.2f})')
P('  EM control cis by year', ct.groupby('year').cis8.agg(['mean','size']).round(2).to_dict('index'))
# paper level: share of EM control papers with every kappa Pro8 chain trans
cp=ct.groupby('clu').cis8.agg(['mean','size'])
q=(cp['mean']==0).mean(); q2=(cp[cp['size']>=2]['mean']==0).mean()
P(f'  EM control papers: {len(cp)}; all-trans share {q:.3f} (papers with >=2 chains: {q2:.3f}); P(6 case papers all-trans | random control papers) = {q**6:.2g}')
# exact-ish permutation at paper level: draw 6 control papers at random
casep=ca.groupby('clu').cis8.mean(); P('  case papers cis means',casep.to_dict())
nonA=ca[~ca.err_modes.fillna('').str.contains('A_FR1')]
P(f'  non-FR1 EM strict: {int(nonA.cis8.sum())}/{len(nonA)}, papers {nonA.clu.nunique()}, without Zhou {len(nonA[nonA.clu!=ZHOU])} chains in {nonA[nonA.clu!=ZHOU].clu.nunique()} papers')
ra=m[m.rest]; P(f'  rest EM all years {int(ra.cis8.sum())}/{len(ra)}; rest EM in window {int(ra[ra.in_paper_window].cis8.sum())}/{int(ra.in_paper_window.sum())}')
# copies: which papers contribute
inc=m[m.incl]; P(f'  incl-copies EM {int(inc.cis8.sum())}/{len(inc)} papers {inc.clu.nunique()}; copies-only chains {int((inc.copyc&~inc.paper_case_strict).sum())}')
# ---------------- T9 constant
K['ccis']=(K.const_cis_n>0).astype(float).where((K.n_const_modeled>60)&(K.const_pro_n>0))
cc=K[K.ccis.notna()&K.ctype.isin(['K','H','L'])]
for lab,mm in [('all',cc),('noZhou',cc[cc.clu!=ZHOU])]:
    ca=mm[mm.paper_case_strict]; ct=mm[mm.paper_control]
    o=hald(int(ca.ccis.sum()),len(ca),int(ct.ccis.sum()),len(ct))
    P(f'T9 pooled {lab}: strict {int(ca.ccis.sum())}/{len(ca)} vs ctrl {int(ct.ccis.sum())}/{len(ct)} ({ct.ccis.mean():.2f}) OR {o[0]:.3g} ({o[1]:.2g}-{o[2]:.2g}) p={o[3]:.2g}; by ctype {ca.groupby("ctype").ccis.agg(["sum","size"]).to_dict("index")}')
# ---------------- T3 Ramachandran beyond site (my own aggregation of verifier tallies)
T=pd.concat([pd.read_pickle(f'{SP}/deeper/cache/verify_su_gap/tallies.pkl'),pd.read_pickle(f'{SP}/paperfp/tallies_xray.pkl')]); T['chain']=T.chain.astype(str)
V=T[(T.cat=='V')&T.dbin.isin(['d4_10','d11p','clean'])].groupby(['pdb','chain'])[['n_rama','allowed']].sum()
V=V[V.n_rama>=30]; V['pA']=100*V.allowed/V.n_rama
R=C.merge(V.reset_index(),on=['pdb','chain'])
for meth in ['EM','X-ray']:
    r=R[R.method==meth]; ca=r[r.paper_case_strict]; ct=r[r.paper_control]; rs=r[r.rest]; cl=r[~r.err_any]
    d=ca.pA.mean()-ct.pA.mean()
    pc=ca.groupby('clu').pA.mean(); pk=ct.groupby('clu').pA.mean(); pr=rs.groupby('clu').pA.mean(); pl=cl.groupby('clu').pA.mean()
    nz=ca[ca.clu!=ZHOU]
    P(f'T3 {meth}: chains {len(ca)} ({ca.clu.nunique()} papers) mean {ca.pA.mean():.2f} vs ctrl {ct.pA.mean():.2f} (n {len(ct)}): crude diff {d:+.2f}; Welch p={stats.ttest_ind(ca.pA,ct.pA,equal_var=False).pvalue:.2g}')
    P(f'   paper means: case {pc.mean():.2f} vs ctrl papers {pk.mean():.2f} diff {pc.mean()-pk.mean():+.2f} MW p={stats.mannwhitneyu(pc,pk).pvalue:.2g}; Zhou paper mean {pc.get(ZHOU,np.nan):.2f}; without Zhou chains diff {nz.pA.mean()-ct.pA.mean():+.2f}')
    P(f'   rest: chain diff {rs.pA.mean()-cl.pA.mean():+.2f}; paper-level diff {pr.mean()-pl.mean():+.2f} ({len(pr)} papers)')
    # paper-level heterogeneity: paper means of case papers vs rest papers, each minus (era-matched) control mean
    x=pd.concat([pd.DataFrame({'v':pc-pk.mean(),'g':'paper'}),pd.DataFrame({'v':pr-pl.mean(),'g':'rest'})])
    x2=pd.concat([pd.DataFrame({'v':(pc-pk.mean()).drop(ZHOU,errors='ignore'),'g':'paper'}),pd.DataFrame({'v':pr-pl.mean(),'g':'rest'})])
    P(f'   paper-level heterogeneity (excess over own control pool): paper {x[x.g=="paper"].v.mean():+.2f} vs rest {x[x.g=="rest"].v.mean():+.2f}, Welch p={stats.ttest_ind(x[x.g=="paper"].v,x[x.g=="rest"].v,equal_var=False).pvalue:.2g}, MW p={stats.mannwhitneyu(x[x.g=="paper"].v,x[x.g=="rest"].v).pvalue:.2g}; without Zhou Welch p={stats.ttest_ind(x2[x2.g=="paper"].v,x2[x2.g=="rest"].v,equal_var=False).pvalue:.2g}')
    if meth=='EM':
        # rest restricted to the window (same era) vs window controls
        rw=rs[rs.in_paper_window]; prw=rw.groupby('clu').pA.mean()
        P(f'   rest in window (conkit-missed + non-S1 entries): chain diff vs window ctrl {rw.pA.mean()-ct.pA.mean():+.2f} n={len(rw)}; paper diff {prw.mean()-pk.mean():+.2f} ({len(prw)} papers); paper vs rest-in-window MW p={stats.mannwhitneyu(pc,prw).pvalue:.2g}')
# ---------------- T5 framework RMSD (investigator's computed values + DE2 verifier values; my stats)
F=pd.read_csv(f'{MF}/tables/fw_computed.csv',dtype={'chain':str})[['pdb','chain','rep_rmsd']]
VD=f'{DEEP}/verify/case_files_DE_DE2_framework_distortion_integrity'
F0=pd.concat([pd.read_csv(f'{VD}/fw_recomputed.csv',dtype={'chain':str}),pd.read_csv(f'{VD}/fw_recomputed_extra.csv',dtype={'chain':str})])[['pdb','chain','rep_rmsd']]
FW=pd.concat([F,F0]).drop_duplicates(['pdb','chain']).dropna()
X=C.merge(FW,on=['pdb','chain'])
for meth in ['EM','X-ray']:
    x=X[X.method==meth]; ca=x[x.paper_case_strict]; ct=x[x.paper_control]; rs=x[x.rest]; cl=x[~x.err_any]
    pc=ca.groupby('clu').rep_rmsd.mean(); pk=ct.groupby('clu').rep_rmsd.mean(); pr=rs.groupby('clu').rep_rmsd.mean(); pl=cl.groupby('clu').rep_rmsd.mean()
    P(f'T5 {meth}: strict {len(ca)} chains/{ca.clu.nunique()} papers mean {ca.rep_rmsd.mean():.3f} vs ctrl {ct.rep_rmsd.mean():.3f} (n {len(ct)}) crude {ca.rep_rmsd.mean()-ct.rep_rmsd.mean():+.3f}; MW p={stats.mannwhitneyu(ca.rep_rmsd,ct.rep_rmsd).pvalue:.2g}')
    P(f'   Zhou chains n={int((ca.clu==ZHOU).sum())} mean {ca[ca.clu==ZHOU].rep_rmsd.mean():.3f}; non-Zhou mean {ca[ca.clu!=ZHOU].rep_rmsd.mean():.3f} diff {ca[ca.clu!=ZHOU].rep_rmsd.mean()-ct.rep_rmsd.mean():+.3f}')
    P(f'   paper level: case {pc.mean():.3f} vs ctrl papers {pk.mean():.3f}: {pc.mean()-pk.mean():+.3f}, MW p={stats.mannwhitneyu(pc,pk).pvalue:.2g}; rest chain {rs.rep_rmsd.mean()-cl.rep_rmsd.mean():+.3f} (n {len(rs)}), rest paper {pr.mean()-pl.mean():+.3f} ({len(pr)} papers)')
    xx=pd.concat([pd.DataFrame({'v':pc-pk.mean(),'g':'paper'}),pd.DataFrame({'v':pr-pl.mean(),'g':'rest'})])
    P(f'   paper-level het: paper {xx[xx.g=="paper"].v.mean():+.3f} vs rest {xx[xx.g=="rest"].v.mean():+.3f} Welch p={stats.ttest_ind(xx[xx.g=="paper"].v,xx[xx.g=="rest"].v,equal_var=False).pvalue:.2g}')
    if meth=='EM':
        rw=rs[rs.in_paper_window]; P(f'   rest in window chain diff vs window ctrl {rw.rep_rmsd.mean()-ct.rep_rmsd.mean():+.3f} n={len(rw)}, paper {rw.groupby("clu").rep_rmsd.mean().mean()-pk.mean():+.3f} ({rw.clu.nunique()} papers)')
        P(f'   ctrl years {ct.year.describe()[["min","50%","max"]].to_dict()} case years {ca.year.describe()[["min","50%","max"]].to_dict()}; ctrl 2020-21 mean {ct[ct.year.isin([2020,2021])].rep_rmsd.mean():.3f} n={int(ct.year.isin([2020,2021]).sum())}')
# ---------------- T6 VH -> VL (entry level)
D=C.copy(); D['isH']=(D.ctype=='H')&(D.vhh!=1); D['isL']=D.ctype.isin(['K','L'])
rows=[]
for p,g in D.groupby('pdb'):
    H=g[g.isH]; L=g[g.isL]
    if len(H)==0 or len(L)==0: continue
    rows.append(dict(pdb=p,method=g.method.iloc[0],clu=g.clu.iloc[0],win=bool(g.in_paper_window.iloc[0]),flag=bool(g.paper_flagged_entry.any()),
        Hs=bool(H.paper_case_strict.any()),Hi=bool(H.incl.any()),Hrun=bool(H.err_any.any()),Lf=bool(L.err_first.any()),Ls=bool(L.paper_case_strict.any()),
        Hrest=bool((H.err_first&~H.incl).any())))
E=pd.DataFrame(rows); W=E[E.win]
for meth in ['EM','X-ray']:
    w=W[W.method==meth]
    a=int((w.Hs&w.Lf).sum()); n1=int(w.Hs.sum()); c=int((~w.Hs&w.Lf).sum()); n0=int((~w.Hs).sum())
    o=hald(a,n1,c,n0); P(f'T6 {meth} window entries: VH strict -> VL err_first {a}/{n1} vs {c}/{n0}: OR {o[0]:.3g} ({o[1]:.2g}-{o[2]:.2g}) p={o[3]:.2g}')
    ex=w[w.Hs]; P('   exposed entries by paper:',ex.groupby('clu').Lf.agg(['sum','size']).to_dict('index'))
    # clean reference: unexposed entries with no run on any VH chain
    w2=w[w.Hs|~w.Hrun]; c2=int((~w2.Hs&w2.Lf).sum()); n02=int((~w2.Hs).sum()); o=hald(a,n1,c2,n02)
    P(f'   ref = VH-clean entries: {c2}/{n02}, OR {o[0]:.3g} ({o[1]:.2g}-{o[2]:.2g})')
    # unexposed but flagged by conkit elsewhere / VH with conkit-missed error
    hm=w[~w.Hs&w.Hrest]; P(f'   unexposed entries whose VH has a conkit-missed first-occurrence error: VL err {int(hm.Lf.sum())}/{len(hm)}')
    # paper level
    pe=ex.groupby('clu').Lf.max(); pu=w[~w.Hs&~w.clu.isin(ex.clu)].groupby('clu').Lf.max()
    o=hald(int(pe.sum()),len(pe),int(pu.sum()),len(pu)); P(f'   papers (any entry): exposed {int(pe.sum())}/{len(pe)} vs unexposed papers {int(pu.sum())}/{len(pu)}: OR {o[0]:.3g} ({o[1]:.2g}-{o[2]:.2g}) p={o[3]:.2g}')
    exz=ex[ex.clu!=ZHOU]; a=int(exz.Lf.sum()); o=hald(a,len(exz),c,n0); P(f'   without Zhou: {a}/{len(exz)} OR {o[0]:.3g} ({o[1]:.2g}-{o[2]:.2g}) p={o[3]:.2g}')
open(f'{OUT}/v02_recompute.txt','w').write('\n'.join(lines)+'\n')
