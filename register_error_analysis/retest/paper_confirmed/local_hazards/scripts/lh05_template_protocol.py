"""Starting model and fitting protocol, paper-confirmed set (junction README sections 2-3).
 (a) Template CDR3 length mismatch: reuse junction/tables/template_cdr3.csv (closest same-type antibody domain in the
     recorded PDB starting model, with its CDR3 length). EM chains released in the paper window; outcome = chain carries
     a paper-confirmed error (strict) in CDR3/FR4 (D+/D-/E) or of any mode; comparators = paper-control chains.
 (b) Recorded starting model (irm) and EM fitting protocol (fit): error rate among in-window EM chains that are either
     paper cases or paper controls. Rigid-body-only vs ab initio (Fisher, chain and entry level) and logistic models
     adjusted for resolution and year with entry- and paper-clustered SEs.
Output: ../tables/template_cdr3_paper.csv, protocol_rates.csv, template_protocol.log"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import fisher_exact
from lhutil import *
log = open(f'{O}/template_protocol.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n')
pd.set_option('display.width', 250)
CH = chains()
R = case_runs()
de_keys = {k: set(zip(g.pdb, g.chain)) for k, g in [('strict', R[R.pset.eq('strict') & R.m.isin(['D+', 'D-', 'E'])]),
                                                       ('incl', R[R.pset.isin(['strict', 'loose', 'copy', 'near_copy']) & R.m.isin(['D+', 'D-', 'E'])])]}
CH['case_DE_strict'] = [k in de_keys['strict'] for k in zip(CH.pdb, CH.chain)]
CH['case_DE_incl'] = [k in de_keys['incl'] for k in zip(CH.pdb, CH.chain)]
W = CH[CH.in_paper_window & (CH.paper_case_strict | CH.paper_case_incl_copies | CH.paper_control)].copy()
P('in-window chains used (paper cases incl. copies + paper controls):', len(W), 'EM', int(W.EM.sum()))
# ---------- (a) template CDR3
T = pd.read_csv('/home/user/SAbR/register_error_analysis/retest/junction/tables/template_cdr3.csv')
order = ['other Ab, same CDR3 length', 'CDR3 length differs by 1', 'differs by >=2', 'identical antibody']
P('\n== (a) junction README reproduction (EM, all years, junction runs of any origin):')
P(T[T.EM].groupby('rel').agg(chains=('pdb', 'size'), junction=('junction', 'sum')).reindex(order).to_string())
X = T.merge(W[['pdb', 'chain', 'paper_case_strict', 'paper_case_incl_copies', 'paper_control', 'case_DE_strict', 'case_DE_incl', 'paper', 'EM']], on=['pdb', 'chain'], suffixes=('_t', ''))
X = X[X.EM]
G = X.groupby('rel').agg(chains=('pdb', 'size'), entries=('pdb', 'nunique'), controls=('paper_control', 'sum'), case_any_strict=('paper_case_strict', 'sum'),
                         case_DE_strict=('case_DE_strict', 'sum'), case_DE_incl=('case_DE_incl', 'sum')).reindex(order)
P('\nPaper window, EM chains with a recorded PDB starting model containing a same-type antibody domain:'); P(G.to_string())
G.to_csv(f'{O}/template_cdr3_paper.csv')
P('cases with a non-identical template:'); P(X[(X.paper_case_strict | X.paper_case_incl_copies) & ~X.identical][['pdb', 'chain', 'template', 'cdr3_len', 't_cdr3_len', 'dl', 'rel', 'case_DE_strict', 'paper_case_strict']].to_string(index=False))
for col in ['case_DE_strict', 'paper_case_strict']:
    a = X[X.rel == 'differs by >=2']; b = X[X.rel.isin(['other Ab, same CDR3 length', 'CDR3 length differs by 1'])]
    t = [[a[col].sum(), len(a) - a[col].sum()], [b[col].sum(), len(b) - b[col].sum()]]
    P(f'{col}: template CDR3 differs >=2: {a[col].sum()}/{len(a)} vs same or +-1: {b[col].sum()}/{len(b)}; Fisher p={fisher_exact(t)[1]:.3g}')
# chains of all paper-strict cases: how many have any recorded template at all
cs = CH[CH.paper_case_strict]
P('\npaper_strict chains by recorded starting model:', cs.irm.value_counts().to_dict(), '| in template table:', int(cs.set_index(['pdb', 'chain']).index.isin(list(zip(T.pdb, T.chain))).sum()))
# ---------- (b) starting model / fitting
E = W[W.EM & W.res.notna()].copy()
rows = []
for var in ['irm', 'fit']:
    for lvl, g in E.groupby(var):
        rows.append(dict(var=var, level=lvl, chains=len(g), entries=g.pdb.nunique(), strict=int(g.paper_case_strict.sum()), pct_strict=100 * g.paper_case_strict.mean(),
                         DE_strict=int(g.case_DE_strict.sum()), pct_DE_strict=100 * g.case_DE_strict.mean(), incl=int(g.paper_case_incl_copies.sum()), pct_incl=100 * g.paper_case_incl_copies.mean()))
PR = pd.DataFrame(rows); P('\n== (b) error rate by recorded starting model / fitting protocol (EM, paper window, cases + paper controls)'); P(PR.round(2).to_string(index=False))
P('junction README (EM all years, CDR3/FR4 junction + bulge, all chains): rigid-body only 1.79% (1,226), flexible 1.13% (885), ab initio 0.23% (441); AlphaFold start 0.34% (290), PDB template 1.09%, none 1.70%')
PR.to_csv(f'{O}/protocol_rates.csv', index=False)
for y in ['paper_case_strict', 'case_DE_strict', 'paper_case_incl_copies']:
    a = E[E.fit == 'rigid body only']; b = E[E.fit == 'ab initio']
    t = [[a[y].sum(), len(a) - a[y].sum()], [b[y].sum(), len(b) - b[y].sum()]]
    OR, p = fisher_exact(t); lo, hi = Table2x2(np.array(t, float) + 0.5).oddsratio_confint()
    ea = a.groupby('pdb')[y].any(); eb = b.groupby('pdb')[y].any()
    te = [[ea.sum(), len(ea) - ea.sum()], [eb.sum(), len(eb) - eb.sum()]]
    P(f'{y}: rigid {a[y].sum()}/{len(a)} vs ab initio {b[y].sum()}/{len(b)} chains, Fisher OR {OR:.2f} (Haldane CI {lo:.2f}-{hi:.2f}) p={p:.3g} | entries {ea.sum()}/{len(ea)} vs {eb.sum()}/{len(eb)} p={fisher_exact(te)[1]:.3g} | papers with a case: rigid {a[a[y]].paper.nunique()}, ab initio {b[b[y]].paper.nunique()}')
    for grp in ['pdb', 'paper']:
        try:
            d = E[E.fit.isin(['rigid body only', 'ab initio', 'flexible fit'])].copy(); d['y'] = d[y].astype(int)
            m = smf.logit("y ~ C(fit, Treatment('ab initio')) + res + year", data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d[grp])[0]})
            k = [i for i in m.params.index if 'rigid' in i][0]; ci = np.exp(m.conf_int().loc[k])
            P(f'   adjusted (res, year; cluster {grp}) rigid vs ab initio OR {np.exp(m.params[k]):.2f} ({ci[0]:.2f}-{ci[1]:.2f}) p={m.pvalues[k]:.3g}')
        except Exception as e: P('   model failed', grp, e)
a = E[E.irm.eq('none recorded')]; b = E[E.irm.eq('experimental (PDB) model')]
for y in ['paper_case_strict', 'case_DE_strict']:
    t = [[a[y].sum(), len(a) - a[y].sum()], [b[y].sum(), len(b) - b[y].sum()]]
    P(f'{y}: no starting model recorded {a[y].sum()}/{len(a)} vs PDB template {b[y].sum()}/{len(b)}: Fisher OR {fisher_exact(t)[0]:.2f} p={fisher_exact(t)[1]:.3g}')
af = CH[CH.in_paper_window & CH.EM & CH.irm.eq('AlphaFold model')]
P(f'\nAlphaFold starting models in the paper window: {len(af)} EM chains in {af.pdb.nunique()} entries (first release {af.release.min()}); paper cases among them: {int(af.paper_case_incl_copies.sum())}')
P(f'AlphaFold starting models after the window: {int((~CH.in_paper_window & CH.EM & CH.irm.eq("AlphaFold model")).sum())} EM chains')
log.close()
