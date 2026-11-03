"""Follow-ups: (1) FR1/NT novelty adjusted for SEQRES N-terminal truncation; (2) prior same-antibody structure with copied
errors counted as cases (bias check); (3) period composition: novelty of EM antibody units by half-year vs error rate,
and novelty x period interaction; (4) germline-gene heterogeneity of error rates."""
import numpy as np, pandas as pd, warnings, statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings('ignore')
OUT='/home/user/SAbR/register_error_analysis/retest/deeper/seq_novelty/tables'
U=pd.read_csv(f'{OUT}/units.csv.gz',low_memory=False)
U['yb']=pd.cut(U.dep_year,[0,2018,2019,2020,2021,2022,2023,2100],labels=False)
U['nonhm']=(~U.q_species_all.isin(['human','mouse'])).astype(int); U['vhh_i']=U.vhh_u.fillna(0).astype(int); U['sars_i']=U.sars2_u.astype(int)
U['ntt']=U.seq_nt_trunc.astype(int); U['fr4t']=U.seq_fr4_trunc.astype(int); U['psa']=(U.n_prior_same_ab>0).astype(int)
EM=U[U.method_u=='EM'].copy()
log=open(f'{OUT}/followups.log','w')
def P(*a):
    print(*a); print(*a,file=log)
def lr(d,feat,cov):
    d=d.dropna(subset=[feat]).copy(); d['f']=(d[feat]-d[feat].mean())/d[feat].std()
    f=smf.logit('y ~ f'+cov,data=d).fit(disp=0); return np.exp(f.params['f']),np.exp(f.conf_int().loc['f']).values,f.pvalues['f'],int(d.y.sum()),int((d.y==0).sum())
base=' + C(yb) + res_med + C(ctype) + nonhm + vhh_i + sars_i'
P('== (1) FR1/NT novelty with SEQRES N-terminal truncation as covariate')
for m,lab in [('ANY','any error'),('A_FR1Nterm','mode A')]:
    cases=EM.err_first_u if m=='ANY' else EM.err_first_u&EM.modes_u.fillna('').str.contains(m)
    d=EM[cases|EM.clean].copy(); d['y']=cases[cases|EM.clean].astype(int)
    for feat in ['nn_diff_FR1','nn_any_NT','nn_diff_NT','nn_any_V','nn_diff_V']:
        for cov,cl in [(base,'M2'),(base+' + ntt','M2+N-trunc'),(base+' + ntt + psa','M2+N-trunc+prior same Ab')]:
            OR,ci,p,n1,n0=lr(d,feat,cov); P(f'{lab:10s} {feat:12s} {cl:28s} OR/SD {OR:.2f} ({ci[0]:.2f}-{ci[1]:.2f}) p={p:.3g} n={n1}/{n0}')
    dd=d[(d.psa==0)&(d.nonhm==0)&(d.vhh_i==0)]
    for feat in ['nn_diff_FR1','nn_any_V']:
        OR,ci,p,n1,n0=lr(dd,feat,' + C(yb) + res_med + C(ctype) + sars_i + ntt'); P(f'{lab:10s} {feat:12s} M3+N-trunc                   OR/SD {OR:.2f} ({ci[0]:.2f}-{ci[1]:.2f}) p={p:.3g} n={n1}/{n0}')
P('\n== (2) prior structure of the same antibody: first-occurrence cases vs all flagged (incl. copied/artifact) vs clean')
for lab,cases in [('first-occurrence',EM.err_first_u),('any run incl. copied',EM.err_any_u)]:
    d=EM[cases|EM.clean].copy(); d['y']=cases[cases|EM.clean].astype(int)
    f=smf.logit('y ~ psa'+base,data=d).fit(disp=0)
    P(f'{lab:22s} prior same Ab: cases {d[d.y==1].psa.mean():.3f} (n={d.y.sum()}) clean {d[d.y==0].psa.mean():.3f} (n={(d.y==0).sum()}) adj OR {np.exp(f.params["psa"]):.2f} p={f.pvalues["psa"]:.3g}')
P('\n== (3) period composition (EM units; first-occurrence error vs clean, others excluded)')
d=EM[EM.err_first_u|EM.clean].copy()
d['half']=pd.to_datetime(d.deposit).dt.year.astype(str)+np.where(pd.to_datetime(d.deposit).dt.month<=6,'H1','H2')
tab=d.groupby('half').agg(n=('clean','size'),err_rate=('err_first_u','mean'),prior_same_ab=('psa','mean'),nn_any_V=('nn_any_V','mean'),
    nn_diff_V=('nn_diff_V','mean'),first_V=('gprior_V',lambda x:(x==0).mean()),first_VJ=('gprior_VJ',lambda x:(x==0).mean()),
    nonhm=('nonhm','mean'),vhh=('vhh_i','mean'),nt_trunc=('ntt','mean'),fr4_trunc=('fr4t','mean'),sars2=('sars_i','mean'))
tab=tab[tab.n>=30]
P(tab.round(3).to_string())
tab.to_csv(f'{OUT}/period_composition.csv')
r=stats.spearmanr(tab.err_rate,tab.prior_same_ab); P('Spearman half-year error rate vs prior-same-Ab share', r)
r=stats.spearmanr(tab.err_rate,tab.nn_diff_V); P('Spearman half-year error rate vs mean nn_diff_V', r)
d['peak']=d.dep_year.isin([2020,2021]).astype(int); d['y']=d.err_first_u.astype(int)
for feat in ['nn_any_V','nn_diff_V','psa']:
    dd=d.dropna(subset=[feat]).copy(); dd['f']=(dd[feat]-dd[feat].mean())/dd[feat].std()
    f=smf.logit('y ~ f*peak + C(yb) + res_med + C(ctype) + nonhm + vhh_i + sars_i',data=dd).fit(disp=0)
    P(f'interaction {feat} x 2020-21: OR(f) outside peak {np.exp(f.params["f"]):.2f} p={f.pvalues["f"]:.3g}; ratio in peak {np.exp(f.params["f:peak"]):.2f} p={f.pvalues["f:peak"]:.3g}')
P('\n== (4) V and J gene heterogeneity of first-occurrence error rate (EM units, human/mouse non-nanobody, genes with >=20 units)')
for col in ['q_vg','q_jg']:
    for ct in ['H','K','L']:
        x=d[(d.ctype==ct)&(d.vhh_i==0)&(d.nonhm==0)]; g=x.groupby(col).y.agg(['sum','size']); g=g[g['size']>=20]
        oth=x[~x[col].isin(g.index)]
        t=np.array([g['sum'].values,(g['size']-g['sum']).values]).T
        chi=stats.chi2_contingency(t)
        top=(g['sum']/g['size']).sort_values(ascending=False).head(3)
        P(f'{col} {ct}: {len(g)} genes, chi2 p={chi[1]:.3g}; top: '+', '.join(f'{k} {v:.1%} (n={g.loc[k,"size"]})' for k,v in top.items()))
log.close()
