"""Pooled local hazards (deepdive_overall.py definitions) on the paper-confirmed cases.

Features at the slipped strand vs controls at the same IMGT window (stratum = mode x heavy/light):
  gap         unbuilt residues within 5 positions of the window, or (mode A) N-terminus not built (first built > IMGT 3)
  poor        window map fit (mean within-domain percentile of Q-score / RSCC) below the controls' 25th percentile
              for that mode and method (threshold from the full clean set, as deepdive_overall.py)
  gap_or_poor either
  adj         adjacent CDR of atypical length (F:L1, B:L2, C/D/E:L3; A: never)
Case sets (units = antibody x paper x mode):
  paper_strict  Table S1 errors confirmed by SAbR/ANARCI, first occurrence (primary)
  paper_incl    + loose-ledger doubt + copies + near copies (sensitivity)
  full          deeper first-occurrence runs, strict-ledger artifacts removed, all years (reference)
  rest          full minus paper units; rest_win = rest released in the paper window (conkit-missed)
Controls: paper = paper_control chains (in window, entry not in Table S1, no run); full = all chains with no run.
Outputs: ../tables/pooled_hazards.csv, pooled_hazards.log, analysis_units.pkl.gz (unit frame reused by lh03)."""
import numpy as np, pandas as pd
from lhutil import *
log = open(f'{O}/pooled_hazards.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n')
pd.set_option('display.width', 250)
C = chains()
F = pd.read_pickle(f'{O}/lh_frame.pkl.gz')
attrs = ['pdb', 'chain', 'ctype', 'light', 'method', 'EM', 'res', 'year', 'release', 'paper', 'vseq', 'unit', 'species', 'v_gene', 'j_gene', 'v_id',
         'L1', 'L2', 'L3', 'L1_atyp', 'L2_atyp', 'L3_atyp', 'vhh', 'sars2', 'lq_z', 'irm', 'fit', 'last_author', 'paper_control', 'in_paper_window',
         'paper_flagged_entry', 'paper_case_strict', 'paper_case_incl_copies']
F = F.merge(C[attrs], on=['pdb', 'chain'], how='left')
R = case_runs()[['pdb', 'chain', 'imgt0', 'imgt1', 'flag', 'pset', 'copied_or_artifact']]
F = F.merge(R.rename(columns={'copied_or_artifact': 'coa'}), on=['pdb', 'chain', 'imgt0', 'imgt1'], how='left')
F['stratum'] = F['mode'] + np.where(F.light, '_L', '_H')
ADJ = {'A': None, 'F': 'L1_atyp', 'B': 'L2_atyp', 'C': 'L3_atyp', 'D+': 'L3_atyp', 'D-': 'L3_atyp', 'E': 'L3_atyp'}
F['adj'] = [bool(r[ADJ[m]]) if ADJ[m] and pd.notna(r[ADJ[m]]) else False for m, (_, r) in zip(F['mode'], F.iterrows())]
F['gap'] = (F.unmod_near.fillna(0) > 0) | ((F['mode'] == 'A') & F.nterm_missing.fillna(False).astype(bool))
# bottom-quartile threshold: controls (full clean set) of that mode and method
F['meth'] = np.where(F.EM, 'EM', 'X-ray')
q = F[F.role == 'control'].groupby(['mode', 'meth']).win_fit_pct.quantile(0.25)
qp = F[(F.role == 'control') & F.paper_control].groupby(['mode', 'meth']).win_fit_pct.quantile(0.25)
P('25th percentile of window fit, full controls vs paper controls:'); P(pd.concat([q.rename('full'), qp.rename('paper')], axis=1).round(3).to_string())
F['poor'] = [(v < q.get((m, me), np.nan)) if pd.notna(v) else np.nan for v, m, me in zip(F.win_fit_pct, F['mode'], F.meth)]
F['has_fit'] = F.win_fit_pct.notna()
F['gap_or_poor'] = F.gap | (F.poor == True)
# ---------------- case sets
cs = F[F.role == 'case'].copy()
cs['first_clean'] = (~cs.coa.astype(bool)) & ~cs.flag.isin(STRICT_LEDGER)
paper_units_incl = set(cs[cs.pset.isin(['strict', 'loose', 'copy', 'near_copy'])].unit)
paper_units_art = set(cs[cs.pset.eq('artifact')].unit)
sets = {
    'paper_strict': cs[cs.pset.eq('strict')],
    'paper_incl': cs[cs.pset.isin(['strict', 'loose', 'copy', 'near_copy'])],
    'full': cs[cs.first_clean],
    'rest': cs[cs.first_clean & ~cs.unit.isin(paper_units_incl | paper_units_art)],
}
sets['rest_win'] = sets['rest'][sets['rest'].in_paper_window]
sets['rest_post'] = sets['rest'][~sets['rest'].in_paper_window]
def units(d):
    # one unit per antibody x paper x mode, chosen AFTER any method filter (a unit with EM and X-ray chains counts in both)
    return d.sort_values(['pdb', 'chain']).drop_duplicates(['mode', 'paper', 'vseq'])
U = {k: v.assign(case=1, cset=k) for k, v in sets.items()}          # chain-level rows; deduplicated inside each analysis
ctl = F[F.role == 'control']
CT = {'paper': ctl[ctl.paper_control].assign(case=0, cset='ctl_paper'), 'full': ctl.assign(case=0, cset='ctl_full')}
for k, v in U.items():
    u = units(v); ue = units(v[v.EM]); ux = units(v[~v.EM])
    P(k, 'unit x mode', len(u), '| EM', len(ue), 'X-ray', len(ux), '| papers', u.paper.nunique(), '| by mode', u['mode'].value_counts().to_dict())
for k, v in CT.items(): P('controls', k, 'units per mode', len(units(v)) // 7, 'EM', len(units(v[v.EM])) // 7)
pd.concat(list(U.values()) + list(CT.values())).drop(columns=['pos_aa']).to_pickle(f'{O}/analysis_units.pkl.gz')
pd.concat(list(U.values()) + list(CT.values())).to_pickle(f'{O}/analysis_units_full.pkl.gz')   # with pos_aa, for lh03

FEATS = [('gap', 'unbuilt within 5 / N-term not built'), ('poor', 'bottom-quartile map fit at site'), ('gap_or_poor', 'either'), ('adj', 'adjacent CDR atypical length')]
rows = []
def one_per_paper(d):
    return d.sort_values(['pdb', 'chain']).drop_duplicates(['paper']) if d.case.iloc[0] == 1 else d.sort_values(['pdb', 'chain']).drop_duplicates(['paper', 'stratum'])
for meth, sel in [('EM', lambda d: d[d.EM]), ('X-ray', lambda d: d[~d.EM]), ('all', lambda d: d)]:
    for f, lab in FEATS:
        def prep(d):
            d = sel(d)
            if f == 'poor': d = d[d.has_fit]
            return units(d).assign(x=d[f].astype(float))
        res = {}
        for cname, cset, ctl_name in [('paper_strict', 'paper_strict', 'paper'), ('paper_incl', 'paper_incl', 'paper'), ('full', 'full', 'full'),
                                      ('rest', 'rest', 'full'), ('rest_win', 'rest_win', 'paper'), ('rest_post', 'rest_post', 'full')]:
            dd = pd.concat([prep(U[cset]), prep(CT[ctl_name])])
            dd = dd[dd.stratum.isin(set(dd[dd.case == 1].stratum))]
            r = mh(dd, 'x'); res[cname] = r
            row = dict(method=meth, feature=lab, set=cname, controls=ctl_name, **{k: r.get(k) for k in ['n_case', 'n_ctrl', 'pct_case', 'pct_ctrl_std', 'OR', 'lo', 'hi', 'p']})
            if cname.startswith('paper'):
                g = gee(dd, 'x'); row.update(gee_OR=g.get('OR'), gee_lo=g.get('lo'), gee_hi=g.get('hi'), gee_p=g.get('p'), n_papers=g.get('n_clusters'))
                d1 = pd.concat([one_per_paper(prep(U[cset])), one_per_paper(prep(CT[ctl_name]))]); d1 = d1[d1.stratum.isin(set(d1[d1.case == 1].stratum))]
                r1 = mh(d1, 'x'); row.update(opp_n=r1.get('n_case'), opp_pct=r1.get('pct_case'), opp_ctrl=r1.get('pct_ctrl_std'), opp_OR=r1.get('OR'), opp_lo=r1.get('lo'), opp_hi=r1.get('hi'), opp_p=r1.get('p'))
                row['min_det_OR80'] = min_detectable_or(r.get('n_case'), (r.get('pct_ctrl_std') or np.nan) / 100, r.get('n_ctrl'))
            rows.append(row)
        # heterogeneity: paper_strict vs rest (log-OR z) and case-only (paper_strict vs rest, and vs rest_win)
        het = dict(method=meth, feature=lab, set='heterogeneity', het_z_p_vs_rest=z_het(res['paper_strict'], res['rest']),
                   het_z_p_vs_rest_win=z_het(res['paper_strict'], res['rest_win']))
        for other in ['rest', 'rest_win']:
            co = pd.concat([prep(U['paper_strict']).assign(isp=1), prep(U[other]).assign(isp=0)])
            co = co[co.stratum.isin(set(co[co.isp == 1].stratum))]
            g = gee(co.assign(y=co.x), 'isp', y='y')
            het[f'caseonly_OR_vs_{other}'] = g.get('OR'); het[f'caseonly_p_vs_{other}'] = g.get('p')
        rows.append(het)
T = pd.DataFrame(rows)
T.to_csv(f'{O}/pooled_hazards.csv', index=False)
for meth in ['EM', 'X-ray', 'all']:
    P(f'\n===== {meth}')
    t = T[(T.method == meth) & (T.set != 'heterogeneity')]
    P(t[['feature', 'set', 'n_case', 'n_ctrl', 'pct_case', 'pct_ctrl_std', 'OR', 'lo', 'hi', 'p', 'gee_OR', 'gee_lo', 'gee_hi', 'gee_p', 'n_papers', 'opp_n', 'opp_OR', 'opp_lo', 'opp_hi', 'opp_p', 'min_det_OR80']].round(3).to_string(index=False))
    h = T[(T.method == meth) & (T.set == 'heterogeneity')]
    P(h[['feature', 'het_z_p_vs_rest', 'het_z_p_vs_rest_win', 'caseonly_OR_vs_rest', 'caseonly_p_vs_rest', 'caseonly_OR_vs_rest_win', 'caseonly_p_vs_rest_win']].round(3).to_string(index=False))
# deepdive-style raw pooled percentages (controls concatenated over modes), for direct comparison with 48/16, 32/25, 66/37, 44/38
P('\n===== deepdive-style raw pooled % (EM; controls of the case chain class concatenated over modes)')
for cname, ctl_name in [('full', 'full'), ('paper_strict', 'paper'), ('paper_incl', 'paper')]:
    cc = units(U[cname][U[cname].EM])
    k = units(CT[ctl_name][CT[ctl_name].EM])
    kk = pd.concat([k[(k['mode'] == m) & k.light.isin(set(cc[cc['mode'] == m].light))] for m in cc['mode'].unique()])
    P(cname, 'cases', {f: round(100 * cc[f].astype(float).mean(), 1) for f, _ in FEATS}, '| controls', {f: round(100 * kk[f].astype(float).mean(), 1) for f, _ in FEATS})
log.close()
