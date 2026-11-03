"""Canonical pull, specific CDR lengths, truncated SEQRES termini and Fv-only models on the paper-confirmed set.
 (A) Canonical pull (literature_dean s06 / synthesis s02): for each paper-confirmed run touching a fully modelled CDR
     (+-2) with |SAbR length - ANARCI length| in {1,2}: is the apparent (built) length more common in the curated X-ray
     reference (<=3.0 A, first released <=2019) than the true length? Observed vs expected under a random direction.
 (B) H2-9 (heavy CDR2 IMGT 7 vs 8), kappa L1-17 (IMGT 12 vs 6), lambda L1-13 (IMGT 8 vs 9): literature_dean unit table
     (antibody x paper); err = unit carries a paper-confirmed error; controls = paper-control units.
 (C) SEQRES truncation (seq_novelty units): N-terminal truncation for mode A, FR4 truncation for D/E, crude and
     stratified by where the model starts / ends (the factor that absorbed them in the full set).
 (D) Fv-only model, pooled over CDR3/FR4 modes (D+, D-, E) and over all modes (lh02 unit frame).
Output: ../tables/canonical_pull_paper.csv, canonical_pull_units.csv, length_tests_paper.csv, termini_paper.csv,
fv_only_paper.csv, lengths_termini.log"""
import pickle, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy.stats import fisher_exact, norm
from lhutil import *
log = open(f'{O}/lengths_termini.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n')
pd.set_option('display.width', 250)
CH = chains()
R = case_runs()
LD = f'{D}/literature_dean/tables'
strict_units = set(CH[CH.paper_case_strict].unit); incl_units = set(CH[CH.paper_case_incl_copies].unit)
flag_units = set(CH[CH.paper_flagged_entry].unit)
# ---------------- (A) canonical pull
REF = pickle.load(open(f'{SP}/deeper/cache/literature_dean/ref.pkl', 'rb')); ref = REF[(REF.hq30) & (REF.first_year <= 2019)]
F = {}
for ct in 'HKL':
    for k, col in [('1', 'len_CDR1'), ('2', 'len_CDR2'), ('3', 'len_CDR3')]:
        v = ref[ref.ctype == ct][col]; F[(ct, k)] = (v.value_counts() / len(v)).to_dict(); F[(ct, k, 'n')] = len(v)
fr = lambda ct, k, L: F[(ct, k)].get(L, 0.5 / F[(ct, k, 'n')])
AL = pd.read_csv(f'{LD}/apparent_cdr_lengths_all_chains.csv.gz').set_index(['pdb', 'chain'])
REG = {'1': (27, 38), '2': (56, 65), '3': (105, 117)}
def pull_rows(runs):
    out = []
    for r in runs.itertuples():
        if (r.pdb, r.chain) not in AL.index: continue
        a_ = AL.loc[(r.pdb, r.chain)]
        for k, (a, b) in REG.items():
            if not (r.imgt0 <= b + 2 and r.imgt1 >= a - 2): continue
            an, sa, tru = a_['an' + k], a_['sa' + k], a_['rL' + k]
            if an != tru: continue
            dl = sa - an
            if abs(dl) not in (1, 2): continue
            out.append(dict(pdb=r.pdb, chain=r.chain, unit=a_.unit, method=a_.method, ctype=a_.ctype, cdr=k, true_len=an, app_len=sa, diff=dl, pset=r.pset,
                            f_true=fr(a_.ctype, k, an), f_app=fr(a_.ctype, k, sa), f_other=fr(a_.ctype, k, an - dl)))
    A = pd.DataFrame(out)
    if len(A) == 0: return A
    A = A.sort_values(['unit', 'cdr']).drop_duplicates(['unit', 'cdr'])
    A['more_common'] = (A.f_app > A.f_true).astype(int); A['p_exp'] = 0.5 * ((A.f_app > A.f_true).astype(int) + (A.f_other > A.f_true).astype(int))
    return A
rng = np.random.default_rng(1)
def pb_test(s):
    o = int(s.more_common.sum()); e = s.p_exp.sum(); v = (s.p_exp * (1 - s.p_exp)).sum()
    sims = rng.binomial(1, np.tile(s.p_exp.values, (50000, 1))).sum(1)
    p = max(min(1, 2 * min((sims >= o).mean(), (sims <= o).mean())), 2e-5)
    return dict(n=len(s), obs=o, exp=round(e, 2), excess=o - e, var=v, p=p)
Ap = pull_rows(R[R.pset.eq('strict')]); Ai = pull_rows(R[R.pset.isin(['strict', 'loose', 'copy', 'near_copy'])])
# rest = literature_dean error units minus strict artifacts (synthesis s02 rule) minus paper units
A0 = pd.read_csv(f'{LD}/apparent_length_error_units.csv'); L = pd.read_csv(f'{D}/synthesis/artifact_ledger_runs.csv')
def flagged(row):
    g = L[(L.pdb == row.pdb) & (L.chain == row.chain)]; a, b = REG[str(row.cdr)]; g = g[(g.imgt0 <= b + 2) & (g.imgt1 >= a - 2)]
    return len(g) > 0 and set(g.flag) <= STRICT_LEDGER
A0c = A0[[not flagged(r) for r in A0.itertuples()]]
Ar = A0c[~A0c.unit.isin(incl_units | flag_units)]
rows = []
for lab, s in [('paper_strict', Ap), ('paper_incl', Ai), ('full (minus strict artifacts; synthesis s02)', A0c), ('rest', Ar)]:
    for sl, ss in [('all', s), ('EM', s[s.method == 'EM']), ('X-ray', s[s.method != 'EM'])]:
        if len(ss) == 0: rows.append(dict(set=lab, subset=sl, n=0)); continue
        rows.append(dict(set=lab, subset=sl, **pb_test(ss), units=ss.unit.nunique(), papers=ss.unit.str.split('|').str[0].nunique()))
CP = pd.DataFrame(rows)
# heterogeneity: per-item excess (obs - exp)/n, paper_strict vs rest
for sl in ['all', 'EM']:
    a = CP[(CP.set == 'paper_strict') & (CP.subset == sl)].iloc[0]; b = CP[(CP.set == 'rest') & (CP.subset == sl)].iloc[0]
    if a.n > 0:
        z = (a.excess / a.n - b.excess / b.n) / np.sqrt(a['var'] / a.n ** 2 + b['var'] / b.n ** 2)
        CP.loc[(CP.set == 'paper_strict') & (CP.subset == sl), 'het_p_vs_rest'] = 2 * norm.sf(abs(z))
P('== (A) canonical pull'); P(CP.round(4).to_string(index=False))
P('paper_strict items:'); P(Ap[['pdb', 'chain', 'ctype', 'cdr', 'true_len', 'app_len', 'f_true', 'f_app', 'more_common', 'p_exp']].round(3).to_string(index=False))
CP.to_csv(f'{O}/canonical_pull_paper.csv', index=False); pd.concat([Ap.assign(set='paper_strict'), Ai.assign(set='paper_incl')]).to_csv(f'{O}/canonical_pull_units.csv', index=False)
# one-per-paper sensitivity
Ap1 = Ap.assign(paper=Ap.unit.str.split('|').str[0]).sort_values(['paper', 'cdr']).drop_duplicates('paper')
P('one item per paper (paper_strict):', pb_test(Ap1))

# ---------------- (B) specific lengths
u = pd.read_csv(f'{LD}/units_literature.csv.gz')
u['paper'] = u.unit.str.split('|').str[0]
ctl_units = set(CH[CH.paper_control].unit)
u['ybin_w'] = pd.cut(u.year, [0, 2019, 2020, 2021, 2100], labels=['<=2019', '2020', '2021', '>=2022']).astype(str)
tests = {'heavy CDR2 IMGT7 [H2-9] vs 8': (lambda d: d[(d.ctype == 'H') & d.rL2.isin([7, 8])], lambda d: (d.rL2 == 7).astype(int)),
         'kappa CDR1 IMGT12 [L1-17] vs 6': (lambda d: d[(d.ctype == 'K') & d.rL1.isin([6, 12])], lambda d: (d.rL1 == 12).astype(int)),
         'lambda CDR1 IMGT8 [L1-13] vs 9': (lambda d: d[(d.ctype == 'L') & d.rL1.isin([8, 9])], lambda d: (d.rL1 == 8).astype(int))}
rows = []
Lr = L.groupby('unit').flag.agg(lambda s: set(s) <= STRICT_LEDGER); rm = set(Lr[Lr].index)
for t, (selr, fx) in tests.items():
    for meth in ['EM', 'all']:
        base = u if meth == 'all' else u[u.all_em]
        for cname, cu in [('paper_strict', strict_units), ('paper_incl', incl_units)]:
            d = base[base.unit.isin(cu) | (base.unit.isin(ctl_units) & ~base.unit.isin(incl_units) & ~base.err_any)]
            d = selr(d).assign(x=lambda z: fx(z), y=lambda z: z.unit.isin(cu).astype(int))
            tt = [[int(((d.x == 1) & (d.y == 1)).sum()), int(((d.x == 1) & (d.y == 0)).sum())], [int(((d.x == 0) & (d.y == 1)).sum()), int(((d.x == 0) & (d.y == 0)).sum())]]
            ORf, pf = fisher_exact(tt); lo, hi = Table2x2(np.array(tt, float) + 0.5).oddsratio_confint()
            row = dict(test=t, method=meth, set=cname, err_exposed=f'{tt[0][0]}/{tt[0][0]+tt[0][1]}', err_ref=f'{tt[1][0]}/{tt[1][0]+tt[1][1]}',
                       case_papers=d[d.y == 1].paper.nunique(), fisher_OR=ORf, lo_haldane=lo, hi_haldane=hi, fisher_p=pf,
                       min_det_OR80=min_detectable_or(int(d.y.sum()), (tt[0][1]) / max(1, tt[0][1] + tt[1][1]), int((d.y == 0).sum())))
            try:
                m = smf.logit('y ~ x + res + C(ybin_w)', data=d).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(d.paper)[0]})
                ci = np.exp(m.conf_int().loc['x']); row.update(adj_OR=np.exp(m.params['x']), adj_lo=ci[0], adj_hi=ci[1], adj_p=m.pvalues['x'])
            except Exception as e: row['adj_err'] = str(e)[:60]
            # one case unit per paper
            d1 = pd.concat([d[d.y == 1].drop_duplicates('paper'), d[d.y == 0]])
            t1 = [[int(((d1.x == 1) & (d1.y == 1)).sum()), int(((d1.x == 1) & (d1.y == 0)).sum())], [int(((d1.x == 0) & (d1.y == 1)).sum()), int(((d1.x == 0) & (d1.y == 0)).sum())]]
            row.update(opp_exposed=f'{t1[0][0]}/{t1[0][0]+t1[1][0]} cases exposed', opp_OR=fisher_exact(t1)[0], opp_p=fisher_exact(t1)[1])
            rows.append(row)
        # full / rest reference (synthesis s03 model on all_em & ~copied_only, minus strict artifacts) and heterogeneity
        if meth == 'EM':
            E0 = u[(u.all_em) & (~u.copied_only) & ~u.unit.isin(rm)]
            for lab, dd in [('full (s03 minus strict artifacts)', E0), ('rest', E0[~E0.unit.isin(incl_units | flag_units)])]:
                d = selr(dd).assign(x=lambda z: fx(z), y=lambda z: z.err.astype(int))
                m = smf.logit('y ~ x + C(ybin) + res', data=d).fit(disp=0)
                ci = np.exp(m.conf_int().loc['x'])
                rows.append(dict(test=t, method='EM', set=lab, err_exposed=f'{d[d.x==1].y.sum()}/{(d.x==1).sum()}', err_ref=f'{d[d.x==0].y.sum()}/{(d.x==0).sum()}',
                                 adj_OR=np.exp(m.params['x']), adj_lo=ci[0], adj_hi=ci[1], adj_p=m.pvalues['x'], se_log=m.bse['x']))
LT = pd.DataFrame(rows)
for t in tests:
    a = LT[(LT.test == t) & (LT.method == 'EM') & (LT.set == 'paper_strict')]; b = LT[(LT.test == t) & (LT.set == 'rest')]
    if len(a) and pd.notna(a.adj_OR.iloc[0]) and np.isfinite(a.adj_OR.iloc[0]) and 0 < a.adj_OR.iloc[0] < 1e3:
        sea = (np.log(a.adj_hi.iloc[0]) - np.log(a.adj_lo.iloc[0])) / 3.92
        z = (np.log(a.adj_OR.iloc[0]) - np.log(b.adj_OR.iloc[0])) / np.sqrt(sea ** 2 + b.se_log.iloc[0] ** 2)
        LT.loc[a.index, 'het_p_vs_rest'] = 2 * norm.sf(abs(z))
P('\n== (B) specific CDR lengths'); P(LT.drop(columns=['se_log'], errors='ignore').round(4).to_string(index=False))
LT.to_csv(f'{O}/length_tests_paper.csv', index=False)

# ---------------- (C) SEQRES truncation
SU = pd.read_csv(f'{D}/seq_novelty/tables/units.csv.gz', low_memory=False, usecols=['unit', 'pdb', 'chain', 'method_u', 'clean', 'err_first_u', 'seq_first_imgt', 'seq_last_imgt', 'seq_nt_trunc', 'seq_fr4_trunc', 'ctype'])
FR = pd.read_pickle(f'{O}/lh_frame.pkl.gz')
fb = FR[FR['mode'] == 'A'].drop_duplicates(['pdb', 'chain']).set_index(['pdb', 'chain'])[['first_imgt', 'last_imgt']]
cu = CH.set_index(['pdb', 'chain'])
CTL_FIRST = CH[CH.paper_control].sort_values(['pdb', 'chain']).drop_duplicates('unit').set_index('unit')[['pdb', 'chain']]
def unit_model_ends(units_, case_chains):
    """first/last built IMGT of the case chain (cases) or of the unit's first paper-control chain (controls)"""
    out = {}
    if case_chains:
        cc_ = CH[np.array([k in case_chains for k in zip(CH.pdb, CH.chain)])].sort_values(['pdb', 'chain']).drop_duplicates('unit').set_index('unit')[['pdb', 'chain']]
    else:
        cc_ = CTL_FIRST
    for un in units_:
        if un not in cc_.index: continue
        k = tuple(cc_.loc[un])
        if k in fb.index: out[un] = fb.loc[k].values
    return out
rows = []
for lab, modes, col, endcol, cut in [('mode A: SEQRES starts after IMGT 1', ['A'], 'seq_nt_trunc', 'first_imgt', [0, 1, 2, 3, 200]),
                                     ('modes D/E: SEQRES ends inside FR4 (<125)', ['D+', 'D-', 'E'], 'seq_fr4_trunc', 'last_imgt', [0, 124, 200])]:
    for cname, ps in [('paper_strict', ['strict']), ('paper_incl', ['strict', 'loose', 'copy', 'near_copy'])]:
        rr = R[R.pset.isin(ps) & R.m.isin(modes)]
        cc = set(zip(rr.pdb, rr.chain)); cunits = set(CH[[k in cc for k in zip(CH.pdb, CH.chain)]].unit)
        for meth in ['EM', 'all']:
            s = SU[SU.unit.isin(cunits) | (SU.unit.isin(ctl_units) & SU.clean & ~SU.unit.isin(incl_units))]
            if meth == 'EM': s = s[s.method_u == 'EM']
            ct_types = set(SU[SU.unit.isin(cunits)].ctype)
            s = s[s.ctype.isin(ct_types)].copy(); s['y'] = s.unit.isin(cunits).astype(int); s['x'] = s[col].astype(int)
            ends = {**unit_model_ends(s[s.y == 0].unit, None), **unit_model_ends(s[s.y == 1].unit, cc)}
            s['end'] = s.unit.map(lambda z: ends.get(z, [np.nan, np.nan])[0 if endcol == 'first_imgt' else 1])
            s['endbin'] = pd.cut(s.end, cut).astype(str)
            t = [[int(((s.x == 1) & (s.y == 1)).sum()), int(((s.x == 0) & (s.y == 1)).sum())], [int(((s.x == 1) & (s.y == 0)).sum()), int(((s.x == 0) & (s.y == 0)).sum())]]
            ORc, pc = fisher_exact(t); lo, hi = Table2x2(np.array(t, float) + 0.5).oddsratio_confint()
            st = mh(s.dropna(subset=['end']).assign(case=lambda z: z.y), 'x', y='case', strata='endbin')
            endx = s.assign(e=(s.end > 1) if endcol == 'first_imgt' else (s.end < 125))
            te = [[int((endx.e & (endx.y == 1)).sum()), int((~endx.e & (endx.y == 1)).sum())], [int((endx.e & (endx.y == 0)).sum()), int((~endx.e & (endx.y == 0)).sum())]]
            rows.append(dict(test=lab, set=cname, method=meth, cases_trunc=f'{t[0][0]}/{t[0][0]+t[0][1]}', ctrl_trunc=f'{t[1][0]}/{t[1][0]+t[1][1]}',
                             pct_case=100 * t[0][0] / max(1, t[0][0] + t[0][1]), pct_ctrl=100 * t[1][0] / max(1, t[1][0] + t[1][1]), OR=ORc, lo=lo, hi=hi, p=pc,
                             MH_OR_given_model_end=st.get('OR'), MH_lo=st.get('lo'), MH_hi=st.get('hi'), MH_p=st.get('p'),
                             model_end_case=f'{te[0][0]}/{te[0][0]+te[0][1]}', model_end_ctrl=f'{te[1][0]}/{te[1][0]+te[1][1]}', model_end_p=fisher_exact(te)[1]))
TT = pd.DataFrame(rows)
P('\n== (C) SEQRES truncation (units; controls = clean paper-control units of the same chain types)')
P('full set: mode A 19/59 (32.2%) vs 7.8% of 4,036 clean EM units, OR 5.32; adjusted for model start OR 1.17 (0.60-2.28) to 3.8 (verify SN1). D 4/59, E 2/18 vs 35/4036; adjusted for model end OR 0.67 / 2.7 (verify SN2).')
P(TT.round(3).to_string(index=False)); TT.to_csv(f'{O}/termini_paper.csv', index=False)

# ---------------- (D) Fv-only
UU = pd.read_pickle(f'{O}/analysis_units.pkl.gz')
UU['fv'] = UU.fv_only.map(lambda v: np.nan if pd.isna(v) else float(bool(v)))
rows = []
for lab, modes in [('CDR3/FR4 modes (D+, D-, E)', ['D+', 'D-', 'E']), ('all modes', ['A', 'F', 'B', 'C', 'D+', 'D-', 'E'])]:
    for meth in ['EM', 'all']:
        for cname, ctl in [('paper_strict', 'ctl_paper'), ('paper_incl', 'ctl_paper'), ('full', 'ctl_full'), ('rest', 'ctl_full')]:
            d = UU[UU.cset.isin([cname, ctl]) & UU['mode'].isin(modes) & UU.fv.notna()]
            if meth == 'EM': d = d[d.EM]
            d = d.sort_values(['pdb', 'chain']).drop_duplicates(['cset', 'mode', 'paper', 'vseq'])
            d = d[d.stratum.isin(set(d[d.case == 1].stratum))]
            r = mh(d, 'fv'); g = gee(d, 'fv') if cname.startswith('paper') else {}
            rows.append(dict(scope=lab, method=meth, set=cname, n_case=r.get('n_case'), pct_case=r.get('pct_case'), pct_ctrl_std=r.get('pct_ctrl_std'), OR=r.get('OR'), lo=r.get('lo'),
                             hi=r.get('hi'), p=r.get('p'), se_log=r.get('se_log'), gee_OR=g.get('OR'), gee_lo=g.get('lo'), gee_hi=g.get('hi'), gee_p=g.get('p'),
                             min_det_OR80=min_detectable_or(r.get('n_case'), (r.get('pct_ctrl_std') or np.nan) / 100, r.get('n_ctrl')) if cname.startswith('paper') else np.nan))
FV = pd.DataFrame(rows)
for (sc, me), g in FV.groupby(['scope', 'method']):
    a = g[g.set == 'paper_strict'].iloc[0]; b = g[g.set == 'rest'].iloc[0]
    FV.loc[g[g.set == 'paper_strict'].index, 'het_p_vs_rest'] = z_het(dict(a), dict(b))
P('\n== (D) Fv-only model (mode-stratified MH; deepdive full set: D+ 55% vs 40%, E 61% vs 40%)'); P(FV.drop(columns=['se_log']).round(3).to_string(index=False))
FV.to_csv(f'{O}/fv_only_paper.csv', index=False)
log.close()
