"""Adversarial recomputation of the most consequential who/where/when numbers with independent code.
Unit flags are rebuilt from chain_flags.csv.gz + runs_first_occurrence.csv + the synthesis ledger (not ww_common.py).
Tests: crude Fisher, year-stratified exact/CMH, logit with year category; leave-one-paper(-cluster)-out.
  T1 China-led last author (EM papers, paper_meta)            : case vs window controls; case vs rest_w; year mix
  T2 Titan Krios (EM entries, pdb_meta)                       : case / rest_w / win_all vs window controls, by year
  T3 capsid target (EM entries and papers, antigen frame)
  T4 no funding record (EM entries, pdb_meta), entry vs paper
  T5 citations (EM papers) with China / leave-one-out
  T6 lab antibody experience (antigen_competition lab_prior_ab; independent of the people_network measure)
"""
import numpy as np, pandas as pd, warnings
from scipy.stats import fisher_exact
from scipy.stats.contingency import odds_ratio
from statsmodels.stats.contingency_tables import StratifiedTable
import statsmodels.api as sm, statsmodels.formula.api as smf
warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad'
D = '/home/user/SAbR/register_error_analysis/retest/deeper'
OUT = '/home/user/SAbR/register_error_analysis/retest/paper_confirmed/verify_who_where_when/tables'
ROWS = []

# ---------------------------------------------------------------- unit flags (own build)
C = pd.read_csv(f'{SP}/deeper/chains.csv.gz', low_memory=False).merge(pd.read_csv(f'{SP}/paperconf/chain_flags.csv.gz'), on=['pdb', 'chain'])
R = pd.read_csv(f'{SP}/deeper/runs_first_occurrence.csv')
L = pd.read_csv(f'{D}/synthesis/artifact_ledger_runs.csv')
R = R.merge(L[['pdb', 'chain', 'imgt0', 'imgt1', 'flag']], on=['pdb', 'chain', 'imgt0', 'imgt1'], how='left')
bad = {'amyloid_fibril', 'indel_gap_artifact_confirmed', 'DE_chain_end_artifact', 'near_copy', 'not_a_register_shift'}
G = set(map(tuple, R[~R.flag.isin(bad)][['pdb', 'chain']].values))
C['gen'] = [k in G for k in zip(C.pdb, C.chain)]
E = C.groupby('pdb').agg(case=('paper_case_strict', 'any'), case_i=('paper_case_incl_copies', 'any'), ctrl=('paper_control', 'all'),
                         win=('in_paper_window', 'all'), gen=('gen', 'any'), paper=('paper', 'first'), method=('method', 'first')).reset_index()
E['rest_w'] = E.gen & ~E.case_i & E.win
del C  # do not shadow patsy C()
E['ctl'] = E.ctrl & ~E.gen & ~E.case_i
print('entries: case', int(E.case.sum()), 'EM', int((E.case & (E.method == 'EM')).sum()), '| rest_w', int(E.rest_w.sum()), '| ctl', int(E.ctl.sum()))

def agg_units(km, key):
    m = km[['pdb', key]].drop_duplicates().merge(E[['pdb', 'case', 'case_i', 'ctl', 'rest_w', 'gen']], on='pdb')
    return m.groupby(key).agg(case=('case', 'any'), case_i=('case_i', 'any'), ctl=('ctl', 'all'), rest_w=('rest_w', 'any'), gen=('gen', 'any')).reset_index()

def ycat(y):
    y = pd.Series(y).astype(float).values
    return np.where(y <= 2019, '<=2019', np.where(y == 2020, '2020', '2021+'))

def test(d, grp, x, strata='yc', label='', ctrl='ctl'):
    """grp vs ctrl (excluding units in grp or gen/case_i from controls); crude Fisher, stratified CMH (exact if zero cells)."""
    dd = d[d[grp] | (d[ctrl] & ~d.gen & ~d.case_i)].dropna(subset=[x]).copy()
    if grp in ('rest_w',):
        dd = d[d[grp] | (d[ctrl] & ~d.gen & ~d.case_i)].dropna(subset=[x]).copy()
    y = dd[grp].astype(int); xx = dd[x].astype(int)
    t = np.array([[((y == 1) & (xx == 1)).sum(), ((y == 1) & (xx == 0)).sum()], [((y == 0) & (xx == 1)).sum(), ((y == 0) & (xx == 0)).sum()]])
    orc, pf = fisher_exact(t); ci = odds_ratio(t, kind='conditional').confidence_interval()
    tabs = []
    for s, g in dd.groupby(strata):
        yy = g[grp].astype(int); x2 = g[x].astype(int)
        tt = np.array([[((yy == 1) & (x2 == 1)).sum(), ((yy == 1) & (x2 == 0)).sum()], [((yy == 0) & (x2 == 1)).sum(), ((yy == 0) & (x2 == 0)).sum()]], float)
        if tt.sum(axis=1).min() > 0 and tt.sum(axis=0).min() > 0: tabs.append(tt + (0.5 if (tt == 0).any() else 0))
    st = StratifiedTable(tabs) if tabs else None
    r = dict(label=label, grp=grp, x=x, n_grp=int(y.sum()), n_ctrl=int((y == 0).sum()), grp_pos=int(t[0, 0]), ctrl_pos=int(t[1, 0]),
             pct_grp=100 * t[0, 0] / t[0].sum(), pct_ctrl=100 * t[1, 0] / t[1].sum(), OR_crude=orc, lo_exact=ci.low, hi_exact=ci.high, p_fisher=pf,
             OR_MH_year=st.oddsratio_pooled if st else np.nan, lo_MH=st.oddsratio_pooled_confint()[0] if st else np.nan,
             hi_MH=st.oddsratio_pooled_confint()[1] if st else np.nan, p_CMH=st.test_null_odds(correction=True).pvalue if st else np.nan)
    ROWS.append(r)
    print(f"{label:55s} {grp:7s} {r['grp_pos']}/{r['n_grp']} ({r['pct_grp']:.0f}%) vs {r['ctrl_pos']}/{r['n_ctrl']} ({r['pct_ctrl']:.0f}%) "
          f"crude {orc:.2f} ({ci.low:.2f}-{ci.high:.2f}) p={pf:.3g} | year-MH {r['OR_MH_year']:.2f} ({r['lo_MH']:.2f}-{r['hi_MH']:.2f}) p={r['p_CMH']:.3g}")
    return r

def direct(a, b, x, label):
    """grp a vs grp b, e.g. case vs rest_w, crude Fisher and year-stratified."""
    dd = pd.concat([a.assign(_g=1), b.assign(_g=0)]).dropna(subset=[x])
    t = pd.crosstab(dd._g, dd[x].astype(int)).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values
    o, p = fisher_exact(t)
    tabs = [pd.crosstab(g._g, g[x].astype(int)).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values.astype(float) for _, g in dd.groupby('yc')]
    tabs = [tt + (0.5 if (tt == 0).any() else 0) for tt in tabs if tt.sum(axis=1).min() > 0 and tt.sum(axis=0).min() > 0]
    st = StratifiedTable(tabs) if tabs else None
    print(f"{label:55s} {t[0,0]}/{t[0].sum()} vs {t[1,0]}/{t[1].sum()} Fisher OR {o:.2f} p={p:.3g} | year-MH {st.oddsratio_pooled if st else np.nan:.2f} p={st.test_null_odds(correction=True).pvalue if st else np.nan:.3g}")
    ROWS.append(dict(label=label, grp='case_vs_rest_w', x=x, grp_pos=int(t[0, 0]), n_grp=int(t[0].sum()), ctrl_pos=int(t[1, 0]), n_ctrl=int(t[1].sum()),
                     OR_crude=o, p_fisher=p, OR_MH_year=st.oddsratio_pooled if st else np.nan, p_CMH=st.test_null_odds(correction=True).pvalue if st else np.nan))

# ---------------------------------------------------------------- T1 China-led (paper_meta, EM papers)
P = pd.read_csv(f'{D}/paper_meta/paper_analysis_table.csv', low_memory=False)
P = P[(P.amyloid_fibril != True)]
km = P[['pkey', 'method', 'pdbs']].assign(pdb=P.pdbs.str.split(';')).explode('pdb'); km['uk'] = km.pkey + '|' + km.method
P['uk'] = P.pkey + '|' + P.method
P = P.merge(agg_units(km, 'uk'), on='uk', how='left')
for c in ['case', 'case_i', 'ctl', 'rest_w', 'gen']: P[c] = P[c].fillna(False).astype(bool)
P['yc'] = ycat(P.dep_year)
PE = P[(P.method == 'EM') & (P.published == True)].copy()
print('\n== T1 China-led last author, EM published papers')
PE['china'] = PE.last_China
test(PE, 'case', 'china', label='China-led, case vs window ctrl')
test(PE, 'rest_w', 'china', label='China-led, rest_w vs window ctrl')
PE['win_all'] = PE.case | PE.rest_w
test(PE.assign(ctl2=PE.ctl), 'win_all', 'china', label='China-led, win_all vs window ctrl')
direct(PE[PE.case], PE[PE.rest_w & ~PE.case], 'china', 'China-led, case vs rest_w')
print('case papers by year x China:\n', PE[PE.case].groupby(['dep_year']).china.agg(['sum', 'count']).T.to_string())
print('rest_w papers by year x China:\n', PE[PE.rest_w & ~PE.case].groupby(['dep_year']).china.agg(['sum', 'count']).T.to_string())
print('controls China share by year:', PE[PE.ctl & ~PE.gen & ~PE.case_i].groupby('dep_year').china.mean().round(2).to_dict())
cc = PE[PE.case & (PE.china == 1)]
print('China-led case papers:', len(cc), '| distinct last authors', cc.last_author.nunique(), '| journals', cc.journal_n.value_counts().to_dict(),
      '| sars2', int(cc.sars2.sum()))
# leave-one-paper-out on China OR
los = []
for k in PE[PE.case].pkey:
    d = PE[PE.pkey != k]; dd = d[d.case | (d.ctl & ~d.gen & ~d.case_i)].dropna(subset=["china"])
    t = pd.crosstab(dd.case, dd.china.astype(int)).reindex(index=[True, False], columns=[1, 0]).values
    los.append(fisher_exact(t)[1])
print('China case vs ctrl, leave-one-paper-out max Fisher p:', round(max(los), 4))
# X-ray
PX = P[(P.method == 'X-ray') & (P.published == True)].copy(); PX['china'] = PX.last_China
test(PX, 'case', 'china', label='X-ray China-led, case vs window ctrl')

# ---------------------------------------------------------------- T5 citations (EM papers)
print('\n== T5 citations per year (log1p), EM published papers; logit with year cat + l2_units + res_median, lab-clustered')
def lg(d, grp, x, rhs, cl='lab'):
    dd = d[d[grp] | (d.ctl & ~d.gen & ~d.case_i)].dropna(subset=[x, 'l2_units', 'res_median']).copy()
    dd['y'] = dd[grp].astype(float); dd['lab'] = dd.last_author.fillna(dd.pkey)
    m = smf.glm(f'y ~ {x} + {rhs}', dd, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(dd[cl])[0]})
    b, s = m.params[x], m.bse[x]
    return np.exp(b), np.exp(b - 1.96 * s), np.exp(b + 1.96 * s), m.pvalues[x], int(dd.y.sum()), int((dd.y == 0).sum())
for lab, rhs, d in [('base', 'C(yc) + l2_units + res_median', PE), ('+China', 'C(yc) + l2_units + res_median + last_China', PE),
                    ('non-China only', 'C(yc) + l2_units + res_median', PE[PE.last_China == 0]),
                    ('China only', 'C(yc) + l2_units + res_median', PE[PE.last_China == 1])]:
    for grp in ['case', 'rest_w']:
        try:
            o, lo, hi, p, n1, n0 = lg(d, grp, 'l_cites_per_year', rhs)
            print(f'  cites {lab:15s} {grp:7s} {n1}/{n0}: OR {o:.2f} ({lo:.2f}-{hi:.2f}) p={p:.3g}')
            ROWS.append(dict(label=f'citations {lab}', grp=grp, x='l_cites_per_year', n_grp=n1, n_ctrl=n0, OR_adj=o, lo_adj=lo, hi_adj=hi, p_adj=p))
        except Exception as e:
            print('  cites', lab, grp, 'failed', str(e)[:60])
lo1 = []
for k in PE[PE.case].pkey:
    o, lo, hi, p, n1, n0 = lg(PE[PE.pkey != k], 'case', 'l_cites_per_year', 'C(yc) + l2_units + res_median'); lo1.append((p, o, k))
print('  citations leave-one-case-paper-out: max p %.3g (OR %.2f, dropped %s)' % max(lo1))
x = PE[PE.case]; y = PE[PE.ctl & ~PE.gen & ~PE.case_i]
print('  cite_pct median case %.2f vs ctrl %.2f' % (x.cite_pct.median(), y.cite_pct.median()))

# ---------------------------------------------------------------- T2 Krios, T4 no funding (pdb_meta EM entries)
F = pd.read_csv(f'{D}/pdb_meta/tables/em_entry_analysis_frame_with_amyloid.csv.gz', low_memory=False)
F = F.drop(columns=[c for c in ['case', 'ctl', 'gen'] if c in F.columns]).merge(E.drop(columns=['method']), on='pdb', how='inner')
F['yc'] = ycat(F.dep_year); F['paper'] = F.doi.fillna(F.pdb)
F['win_all'] = F.case | F.rest_w
print('\n== T2 Krios, EM entries')
for g in ['case', 'rest_w', 'win_all']:
    test(F, g, 'krios', label=f'Krios entry, {g} vs window ctrl')
direct(F[F.case], F[F.rest_w & ~F.case], 'krios', 'Krios entry, case vs rest_w')
c = F[F.ctl & ~F.gen & ~F.case_i]
print('Krios by year (ctrl):', c.groupby('dep_year').krios.mean().round(2).to_dict())
print('Krios by year (case):', F[F.case].groupby('dep_year').krios.agg(['sum', 'count']).T.to_dict())
print('Krios by year (rest_w):', F[F.rest_w & ~F.case].groupby('dep_year').krios.agg(['sum', 'count']).T.to_dict())
print('rest_w non-Krios entries by paper/microscope:', F[F.rest_w & ~F.case & (F.krios == 0)].groupby(['paper', 'im_microscope_model']).pdb.count().to_dict())
# paper level (majority)
FP = F.groupby('paper').agg(krios=('krios', lambda s: float(s.mean() >= 0.5)), fund_none=('fund_none', lambda s: float(s.mean() >= 0.5)),
                            dep_year=('dep_year', 'min'), case=('case', 'any'), case_i=('case_i', 'any'), ctl=('ctl', 'all'), rest_w=('rest_w', 'any'), gen=('gen', 'any')).reset_index()
FP['yc'] = ycat(FP.dep_year); FP['win_all'] = FP.case | FP.rest_w
for g in ['case', 'rest_w', 'win_all']:
    test(FP, g, 'krios', label=f'Krios paper, {g} vs window ctrl')
print('\n== T4 no funding record, EM entries vs papers')
for g in ['case', 'case_i', 'rest_w', 'win_all']:
    test(F, g, 'fund_none', label=f'no funding entry, {g}')
for g in ['case', 'rest_w', 'win_all']:
    test(FP, g, 'fund_none', label=f'no funding paper, {g}')
print('case entries with no funding record by paper:', F[F.case & (F.fund_none == 1)].groupby('paper').pdb.count().to_dict())
# leave-largest-paper-out
big = F[F.case].paper.value_counts().index[0]
test(F[F.paper != big], 'case', 'fund_none', label=f'no funding entry, case, drop {big}')
# paper-clustered logit at entry level for funding
dd = F[F.case | (F.ctl & ~F.gen & ~F.case_i)].copy(); dd['y'] = dd.case.astype(float)
m = smf.glm('y ~ fund_none + C(yc) + res', dd, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(dd.paper)[0]})
print('  no funding entry logit yc+res paper-clustered: OR %.2f (%.2f-%.2f) p=%.3g' % (np.exp(m.params.fund_none), *np.exp(m.conf_int().loc['fund_none']), m.pvalues.fund_none))

# ---------------------------------------------------------------- T3 capsid, T6 lab experience (antigen frame)
A = pd.read_csv(f'{D}/verify/antigen_competition_no_competitive_pressure_confounding/entries_frame.csv.gz', low_memory=False)
A = A.drop(columns=['clean']).merge(E.drop(columns=['method', 'paper']), on='pdb', how='inner')
A = A[A.ab_status2.isin(['bound', 'fiducial-only'])].copy()
A['yc'] = ycat(A.dyear); A['cap'] = (A.cls2 == 'viral capsid').astype(int); A['enz'] = (A.cls2 == 'enzyme').astype(int)
AE = A[A.method == 'EM'].copy(); AE['win_all'] = AE.case | AE.rest_w
print('\n== T3 capsid, EM entries')
for g in ['case', 'case_i', 'rest_w', 'win_all']:
    test(AE, g, 'cap', label=f'capsid entry, {g}')
AP = AE.groupby('paper').agg(cap=('cap', 'max'), enz=('enz', 'max'), dyear=('dyear', 'min'), case=('case', 'any'), case_i=('case_i', 'any'),
                             ctl=('ctl', 'all'), rest_w=('rest_w', 'any'), gen=('gen', 'any'), lab=('lab2', 'first')).reset_index()
AP['yc'] = ycat(AP.dyear); AP['win_all'] = AP.case | AP.rest_w
for g in ['case', 'rest_w', 'win_all']:
    test(AP, g, 'cap', label=f'capsid paper, {g}')
    test(AP, g, 'enz', label=f'enzyme paper, {g}')
print('case capsid papers:', AP[AP.case & (AP.cap == 1)][['paper', 'dyear', 'lab']].values.tolist())
print('case enzyme papers:', AP[AP.case & (AP.enz == 1)][['paper', 'dyear', 'lab']].values.tolist())
# capsid at paper level, one per lab key among case
print('capsid case papers per lab:', AP[AP.case & (AP.cap == 1)].lab.value_counts().to_dict())

print('\n== T6 lab antibody experience (antigen_competition lab_prior_ab; log2(1+n)), EM entries, logit yc+res, paper-clustered')
LX = pd.read_csv(f'{D}/antigen_competition/tables/lab_experience.csv')
AE2 = AE.merge(LX, on='pdb', how='left'); AE2['lexp'] = np.log2(1 + AE2.lab_prior_ab)
for g in ['case', 'case_i', 'rest_w', 'win_all']:
    dd = AE2[AE2[g] | (AE2.ctl & ~AE2.gen & ~AE2.case_i)].dropna(subset=['lexp', 'res']).copy(); dd['y'] = dd[g].astype(float)
    m = smf.glm('y ~ lexp + C(yc) + res', dd, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(dd.paper)[0]})
    print(f'  lab exp entry {g:7s} {int(dd.y.sum())}/{int((dd.y==0).sum())}: OR {np.exp(m.params.lexp):.2f} ({np.exp(m.conf_int().loc["lexp"][0]):.2f}-{np.exp(m.conf_int().loc["lexp"][1]):.2f}) p={m.pvalues.lexp:.3g}')
    ROWS.append(dict(label='lab experience entry (antigen_competition measure)', grp=g, x='lexp', n_grp=int(dd.y.sum()), n_ctrl=int((dd.y == 0).sum()),
                     OR_adj=np.exp(m.params.lexp), p_adj=m.pvalues.lexp))
AP2 = AE2.groupby('paper').agg(lexp=('lexp', 'min'), res=('res', 'median'), dyear=('dyear', 'min'), lab=('lab2', 'first'), case=('case', 'any'), case_i=('case_i', 'any'),
                               ctl=('ctl', 'all'), rest_w=('rest_w', 'any'), gen=('gen', 'any')).reset_index()
AP2['yc'] = ycat(AP2.dyear); AP2['win_all'] = AP2.case | AP2.rest_w
for g in ['case', 'rest_w', 'win_all']:
    dd = AP2[AP2[g] | (AP2.ctl & ~AP2.gen & ~AP2.case_i)].dropna(subset=['lexp', 'res']).copy(); dd['y'] = dd[g].astype(float)
    m = smf.glm('y ~ lexp + C(yc) + res', dd, family=sm.families.Binomial()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(dd.lab.fillna(dd.paper))[0]})
    print(f'  lab exp paper {g:7s} {int(dd.y.sum())}/{int((dd.y==0).sum())}: OR {np.exp(m.params.lexp):.2f} ({np.exp(m.conf_int().loc["lexp"][0]):.2f}-{np.exp(m.conf_int().loc["lexp"][1]):.2f}) p={m.pvalues.lexp:.3g}')
    ROWS.append(dict(label='lab experience paper (antigen_competition measure)', grp=g, x='lexp', n_grp=int(dd.y.sum()), n_ctrl=int((dd.y == 0).sum()),
                     OR_adj=np.exp(m.params.lexp), p_adj=m.pvalues.lexp))

# ---------------------------------------------------------------- SARS-CoV-2 share among 2020+ EM entries
print('\n== SARS-CoV-2, EM entries deposited 2020+')
S = AE[AE.dyear >= 2020].copy(); S['sars2'] = S.sars2.astype(int)
test(S, 'case', 'sars2', label='SARS-CoV-2 entry 2020+, case')
test(S, 'rest_w', 'sars2', label='SARS-CoV-2 entry 2020+, rest_w')
direct(S[S.case], S[S.rest_w & ~S.case], 'sars2', 'SARS-CoV-2 entry 2020+, case vs rest_w')
SP_ = S.groupby('paper').agg(sars2=('sars2', 'max'), dyear=('dyear', 'min'), case=('case', 'any'), case_i=('case_i', 'any'), ctl=('ctl', 'all'), rest_w=('rest_w', 'any'), gen=('gen', 'any')).reset_index()
SP_['yc'] = ycat(SP_.dyear)
direct(SP_[SP_.case], SP_[SP_.rest_w & ~SP_.case], 'sars2', 'SARS-CoV-2 paper 2020+, case vs rest_w')

pd.DataFrame(ROWS).to_csv(f'{OUT}/v02_recomputed.csv', index=False)
