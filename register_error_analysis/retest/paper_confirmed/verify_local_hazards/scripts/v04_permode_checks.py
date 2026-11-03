"""Own-code checks of per-mode and distribution verdicts: D+ CDR3 length within chain type; IGHJ6 within heavy;
mode A first built residue (alignment-based); mode C not-Asp81 (ANARCI numbering of SEQRES, own lookup);
C-flank +1 events by paper; CDR3/FR4 share vs 20%; effective end of the conkit snapshot; starting-model test clustered;
probability of 0/n under the full-set case rate for the 'absent' verdicts. Output: tables/v04.log"""
import json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, binomtest
SP='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
OUT='/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_local_hazards/tables'
LED='/home/user/SAbR/register_error_analysis/retest/deeper/synthesis/artifact_ledger_runs.csv'
log=open(f'{OUT}/v04.log','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True); log.write(s+'\n')
pd.set_option('display.width',250)
MODEMAP={'A_FR1Nterm':'A','F_CDR1':'F','B_CDR2FR3':'B','C_FR3':'C','Dplus_CDR3short':'D+','Dminus_CDR3long':'D-','E_FR4bulge':'E'}
STRICT={'amyloid_fibril','indel_gap_artifact_confirmed','DE_chain_end_artifact','near_copy','not_a_register_shift'}
C=pd.read_csv(f'{SP}/deeper/chains.csv.gz',low_memory=False).merge(pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz'),on=['pdb','chain'])
G=pd.read_pickle(f'{OUT}/v02_gap_frame.pkl.gz'); C=C.merge(G,on=['pdb','chain'],how='left')
AC=json.load(open(f'{SP}/sabr/anarci_cache.json'))
def aa_at(seq,ct,pos):
    for d in AC.get(seq) or []:
        if (d['chain_type'] in 'KL')==(ct in 'KL'):
            for s,p,i in d['pos']:
                if p==pos and not i: return seq[s]
    return None
R=pd.read_csv(f'{SP}/deeper/runs_all.csv').merge(pd.read_csv(LED)[['pdb','chain','imgt0','imgt1','flag']],on=['pdb','chain','imgt0','imgt1'],how='left')
V=pd.read_csv(f'{OUT}/v01_cases_rederived.csv')
R=R.merge(V[V.n_overlap_runs>0][['pdb','chain','imgt0','imgt1','status_v']].drop_duplicates(['pdb','chain','imgt0','imgt1']),on=['pdb','chain','imgt0','imgt1'],how='left')
R['m']=R['mode'].map(MODEMAP).fillna(R['mode'])
R=R.drop(columns=['paper','year','sars2','res','method','ctype']).merge(C,on=['pdb','chain'],how='left')
strict=R[R.status_v.eq('confirmed first occurrence')]
pu=set(R[R.status_v.notna()&(R.status_v!='not confirmed by SAbR/ANARCI')].unit)
rest=R[(~R.copied_or_artifact.astype(bool))&~R.flag.isin(STRICT)&~R.unit.isin(pu)]
def units(d): return d.sort_values(['pdb','chain']).drop_duplicates(['m','paper','vseq'])
ctl=C[C.paper_control].sort_values(['pdb','chain']).drop_duplicates(['paper','vseq'])
# ---------- D+ CDR3 length within chain type
P('=== D+ CDR3 length (L3) by chain type, paper_strict units vs paper-control units')
d=units(strict[strict.m=='D+'])
P(d[['pdb','chain','ctype','method','L3','cdr3_len','j_gene','sars2']].to_string(index=False))
for meth,sel in [('all',lambda x:x),('EM',lambda x:x[x.method=='EM'])]:
    dd=sel(d); kk=sel(ctl)
    for heavy in [True,False]:
        a=dd[dd.ctype.eq('H')==heavy].L3.dropna(); b=kk[kk.ctype.eq('H')==heavy].L3.dropna()
        if len(a): P(f'  {meth} {"heavy" if heavy else "light"}: cases n={len(a)} median {a.median()} vs controls n={len(b)} median {b.median()}; MW p={mannwhitneyu(a,b).pvalue:.3g}')
    # pooled within-type: rank percentile within chain type
    kk=kk.copy(); dd=dd.copy()
    pct=[(kk[kk.ctype.eq('H')==(ct=='H')].L3<l).mean()+0.5*(kk[kk.ctype.eq('H')==(ct=='H')].L3==l).mean() for ct,l in zip(dd.ctype,dd.L3)]
    P(f'  {meth}: within-type percentile of case CDR3 length: {np.round(pct,2).tolist()}; mean {np.mean(pct):.2f}; sign test >0.5: {sum(p>0.5 for p in pct)}/{len(pct)} p={binomtest(sum(p>0.5 for p in pct),len(pct)).pvalue:.3g}')
    P(f'  {meth}: control heavy share {kk.ctype.eq("H").mean():.2f} vs case heavy share {dd.ctype.eq("H").mean():.2f}; pooled-type control median {kk.L3.median()}')
# IGHJ6 within heavy
dh=d[d.ctype.eq('H')]; kh=ctl[ctl.ctype.eq('H')]
a=dh.j_gene.astype(str).str.startswith('IGHJ6'); b=kh.j_gene.astype(str).str.startswith('IGHJ6')
P(f'IGHJ6 within heavy (all methods): {a.sum()}/{len(a)} vs {b.mean()*100:.1f}% of heavy controls; Fisher p={fisher_exact([[a.sum(),len(a)-a.sum()],[b.sum(),len(b)-b.sum()]])[1]:.3g}')
# SARS-CoV-2 D+ adjusted for era: controls released 2020-2021 only
k2=ctl[ctl.year.between(2020,2021)]; a=d.sars2.astype(bool); b=k2.sars2.astype(bool)
P(f'D+ SARS-CoV-2 (all methods): {a.sum()}/{len(a)} vs 2020-21 controls {100*b.mean():.1f}%; Fisher p={fisher_exact([[a.sum(),len(a)-a.sum()],[b.sum(),len(b)-b.sum()]])[1]:.3g}; case years {sorted(d.year.tolist())}')
# ---------- mode A
P('\n=== mode A first built residue (alignment-based, ANARCI IMGT of first built SEQRES residue), EM')
dA=units(strict[(strict.m=='A')&(strict.method=='EM')]); kA=ctl[ctl.method.eq('EM')&ctl.ctype.isin(set(dA.ctype))]
P(dA[['pdb','chain','ctype','imgt0','imgt1','first_built_v','paper']].assign(paper=dA.paper.str[-24:]).to_string(index=False))
a=dA.first_built_v.dropna(); b=kA.first_built_v.dropna()
P(f'  median {a.median()} vs {b.median()}; MW p={mannwhitneyu(a,b).pvalue:.3g}; >3: {(a>3).sum()}/{len(a)} vs {(b>3).mean()*100:.1f}% Fisher p={fisher_exact([[(a>3).sum(),(a<=3).sum()],[(b>3).sum(),(b<=3).sum()]])[1]:.3g}; >1: {(a>1).sum()}/{len(a)} vs {(b>1).mean()*100:.1f}%')
a1=dA.sort_values(['pdb','chain']).drop_duplicates('paper').first_built_v
P(f'  one per paper n={len(a1)}: >3 {(a1>3).sum()}/{len(a1)} Fisher p={fisher_exact([[(a1>3).sum(),(a1<=3).sum()],[(b>3).sum(),(b<=3).sum()]])[1]:.3g}; MW p={mannwhitneyu(a1,b).pvalue:.3g}')
# ---------- mode C not D81
P('\n=== mode C heavy: residue at IMGT 81 (own ANARCI lookup on SEQRES)')
dC=units(strict[strict.m=='C']); P(dC[['pdb','chain','ctype','method','species','v_gene','imgt0','imgt1']].to_string(index=False))
dC=dC[dC.ctype.eq('H')]; dC['a81']=[aa_at(s,c,81) for s,c in zip(dC.seq,dC.ctype)]
kC=ctl[ctl.ctype.eq('H')&ctl.method.eq('EM')].copy(); kC['a81']=[aa_at(s,c,81) for s,c in zip(kC.seq,kC.ctype)]; kC=kC[kC.a81.notna()]
P(dC[['pdb','chain','method','a81','v_gene']].to_string(index=False))
de=dC[dC.method.eq('EM')]; x=(de.a81!='D').sum(); y=(kC.a81!='D').sum()
P(f'  EM heavy: not D81 {x}/{len(de)} vs {y}/{len(kC)} ({100*y/len(kC):.1f}%); Fisher p={fisher_exact([[x,len(de)-x],[y,len(kC)-y]])[1]:.3g}')
# ---------- C-flank events by paper
P('\n=== C-flank (118-128) strict runs excl. cls 1a/4/3: events by paper')
s=strict[~strict.cls.astype(str).isin(['1a','4','3'])]; s=s[(s.imgt0<=128)&(s.imgt1>=118)]
ev=s.drop_duplicates(['paper','seqkey','imgt0','delta'])
P(ev[['pdb','chain','m','imgt0','imgt1','delta','cls','paper']].assign(paper=ev.paper.str[-24:]).to_string(index=False))
pp=ev.groupby('paper').delta.apply(lambda v:(v>0).mean())
P(f'  events {len(ev)}, +1 {(ev.delta>0).sum()}; papers {len(pp)}; papers majority +1: {(pp>0.5).sum()}/{len(pp)}; binomial p (papers) = {binomtest(int((pp>0.5).sum()),len(pp)).pvalue:.3g}; binomial p (events) = {binomtest(int((ev.delta>0).sum()),len(ev)).pvalue:.3g}')
P(f'  cls of strict C-flank events: {ev.cls.astype(str).value_counts().to_dict()}; strict C-flank runs of cls 3 dropped: {len(strict[(strict.cls.astype(str)=="3")&(strict.imgt0<=128)&(strict.imgt1>=118)])}')
# ---------- CDR3/FR4 share vs 20%
u=strict.drop_duplicates(['m','paper','vseq']); k=u.m.isin(['D+','D-','D0','E']).sum()
bt=binomtest(int(k),len(u),0.2,alternative='greater'); ci=binomtest(int(k),len(u)).proportion_ci()
P(f'\n=== CDR3/FR4 share: {k}/{len(u)} = {k/len(u):.2f}, 95% CI {ci.low:.2f}-{ci.high:.2f}; one-sided binomial vs 0.20 p={bt.pvalue:.3g}')
pp=u.groupby('paper').apply(lambda g:g.m.isin(['D+','D-','D0','E']).mean()); P(f'  paper-weighted share (mean over papers of per-paper share): {pp.mean():.2f}')
# ---------- effective end of conkit snapshot
T=pd.read_excel('/tmp/supp/epmc/m-11-00938-sup2.xlsx'); ent=set(T.PDB_CHAIN_ID.str.split(':').str[0].str.lower())
rel=C[C.pdb.isin(ent)].groupby('pdb').release.first()
P(f'\n=== Table S1 entries present in the scan: {len(rel)}; latest release {rel.max()}; releases after 2022-02-02: {(rel>"2022-02-02").sum()}')
rw=rest[rest.in_paper_window].drop_duplicates(['m','paper','vseq'])
P(f'  rest_win (conkit-missed) units {len(rw)}; released after the last Table S1 entry ({rel.max()}): {(rw.release>rel.max()).sum()}; after 2022-02-02: {(rw.release>"2022-02-02").sum()}')
P('  rest_win units released after 2022-01-01:'); P(rw[rw.release>'2022-01-01'][['pdb','chain','m','release','species','v_gene']].to_string(index=False))
x=rw[rw.release>rel.max()]; P(f'  CDR3/FR4 share among late rest_win: {x.m.isin(["D+","D-","D0","E"]).sum()}/{len(x)}; among the others {rw[rw.release<=rel.max()].m.isin(["D+","D-","D0","E"]).sum()}/{(rw.release<=rel.max()).sum()}')
y=rw[rw.release<=rel.max()]; t=[[12,24],[y.m.isin(['D+','D-','D0','E']).sum(),(~y.m.isin(['D+','D-','D0','E'])).sum()]]
P(f'  paper_strict 12/36 vs rest_win restricted to <= last S1 release: Fisher p={fisher_exact(t)[1]:.3g}')
# ---------- starting model clustered
P('\n=== starting model: none recorded vs PDB template, window, cases (strict) + paper controls, any mode')
W=C[C.in_paper_window&(C.paper_case_strict|C.paper_control)].copy()
P('  irm values:',W.irm.value_counts().head(8).to_dict())
W['grp']=np.where(W.irm.astype(str).str.contains('none',case=False),'none',np.where(W.irm.astype(str).str.contains('PDB|template|experimental',case=False),'pdb','other'))
for lvl,dd in [('chain',W),('unit',W.sort_values('paper_case_strict',ascending=False).drop_duplicates(['paper','vseq'])),('paper',W.sort_values('paper_case_strict',ascending=False).drop_duplicates(['paper']))]:
    t=dd[dd.grp.isin(['none','pdb'])].groupby('grp').paper_case_strict.agg(['sum','count'])
    if len(t)==2:
        OR,p=fisher_exact([[t.loc['none','sum'],t.loc['none','count']-t.loc['none','sum']],[t.loc['pdb','sum'],t.loc['pdb','count']-t.loc['pdb','sum']]])
        P(f'  {lvl}: none {t.loc["none","sum"]}/{t.loc["none","count"]} vs pdb {t.loc["pdb","sum"]}/{t.loc["pdb","count"]}: OR {OR:.2f} p={p:.3g}')
# ---------- 'absent' verdicts: P(observed or fewer | full-set case rate)
from scipy.stats import binom
P('\n=== P(observing this few | case rate equal to the full-set case rate)')
for lab,k,n,pr in [('C mouse IGHV5',0,5,0.22),('E mouse IGKV8',0,3,0.17),('E mouse',0,3,0.44),('B atypical CDR2',0,4,0.31),('B nanobody',0,4,0.28),
                   ('Fv-only CDR3/FR4 EM',1,7,0.50),('Fv-only CDR3/FR4 all',3,12,0.50),('A SEQRES N-trunc EM',0,9,0.32),('D/E SEQRES FR4 trunc all',0,12,6/77)]:
    P(f'  {lab}: {k}/{n} vs full-set case rate {pr:.2f}: P(X<={k})={binom.cdf(k,n,pr):.3f}')
log.close()
